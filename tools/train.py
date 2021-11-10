import argparse
import logging
import pprint
import mxnet as mx
import numpy as np
import os
import time
import copy
import glob
import importlib
import horovod.mxnet as hvd

from six.moves import reduce
from six.moves import cPickle as pkl

from utils import callback
from utils.memonger_v2 import search_plan_to_layer
from utils.detection_module import DetModule
from utils.lr_scheduler import WarmupMultiFactorScheduler, LRSequential, AdvancedLRScheduler
from utils.load_model import load_checkpoint
from utils.cpu_affinity import bind_cpus_on_ecos
from utils.local_sync import broadcast_params_local
from utils.train_utils import OneCycleScheduler, OneCycleMomentumScheduler


def train_net(config):
    print('Please using new data mean and var')
    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list, pLabelMap = config.get_config(is_train=True)

    pretrain_prefix = pModel.pretrain.prefix
    pretrain_epoch = pModel.pretrain.epoch
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    from utils.logger import config_logger
    config_logger(os.path.join(save_path, "log.txt"))

    model_prefix = os.path.join(save_path, "checkpoint")
    # set up logger
    logger = logging.getLogger()

    begin_epoch = pOpt.schedule.begin_epoch
    end_epoch = pOpt.schedule.end_epoch
    lr_step = pOpt.schedule.lr_step

    # only rank==0 print all debug infos
    if pKv.use_horovod is False:
        if os.environ.get("DMLC_ROLE") == "worker":
            assert pKv.kvstore == "dist_sync", "kv dist require dist_sync, {}".format(pKv.kvstore)
        kv = mx.kvstore.create(pKv.kvstore)
        rank = kv.rank
        num_workers = kv.num_workers
        ctx = [mx.gpu(int(i)) for i in pKv.gpus]
        gpu_per_worker = int(len(pKv.gpus) / num_workers)
        gpu_list = pKv.gpus[rank * gpu_per_worker: (rank + 1) * gpu_per_worker]
        ctx = [mx.gpu(int(i)) for i in gpu_list]
        logger.info("rank:{}/{} gpu list:{}".format(rank, num_workers, gpu_list))
        bind_cpus_on_ecos(rank, num_workers)
    else:
        assert os.environ.get("DMLC_ROLE", "None") == "None"
        # Horovod: initialize Horovod
        hvd.init()
        num_workers = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        local_size = hvd.local_size()
        gpu_per_worker = 3
        logger.info("rank: {}/{}, local rank:{}".format(rank, num_workers, local_rank))
        # Horovod: pin GPU to local rank
        ctx = [mx.gpu(local_rank)]
        bind_cpus_on_ecos(local_rank, local_size)

    sync_flag = pKv.sync_flag
    if sync_flag is False:
        model_prefix += "_{}".format(rank)
    dist_flag = "hvd" if pKv.use_horovod else pKv.kvstore
    logger.info("[{}/{}] {} sync flag:{}".format(rank, num_workers, dist_flag, sync_flag))

    sym = pModel.train_symbol
    # sym.save("src_train.json")
    # raise NotImplementedError

    # setup multi-gpu
    input_batch_size = pKv.batch_image * len(ctx)
    logger.info("input batch size:{}".format(input_batch_size))

    # print config
    # if rank == 0:
    #     logger.info(pprint.pformat(config))

    # load dataset and prepare imdb for training
    image_sets = pDataset.image_set
    roidbs = []
    for data_split in image_sets:
        segments = glob.glob(os.path.join(pDataset.data_root, data_split, '*.roidb'))
        for image_set in segments:
            roidb = pkl.load(open(image_set, "rb"), encoding="latin1")
            roidbs.append(roidb)
    roidb = reduce(lambda x, y: x + y, roidbs)

    # sampling rate
    roidb = [r for idx, r in enumerate(roidb) if idx % pDataset.sampling_rate == 0]
    type_dict = {
        'TYPE_UNKNOWN': 0,
        'TYPE_VEHICLE': 1,
        'TYPE_PEDESTRIAN': 2,
        'TYPE_SIGN': 3,
        'TYPE_CYCLIST': 4,
    }

    if hasattr(pDataset, 'filter_class'):
        assert isinstance(pDataset.filter_class, list)
        filtered_roidb = []
        for r in roidb:
            valid_class = np.any([r["gt_class"] == type_dict[c] for c in pDataset.filter_class], axis=0)
            if valid_class.sum() > 0:
                filtered_roidb.append(r)
        roidb = filtered_roidb

    print("load all training data with {} records".format(len(roidb)))

    from utils.detection_input import PostMergeBatchLoader as Loader
    train_data = Loader(
        roidb=roidb,
        transform=transform,
        data_name=data_name,
        label_name=label_name,
        batch_size=input_batch_size,
        shuffle=True,
        num_worker=gpu_per_worker,
        num_collector=gpu_per_worker,
        worker_queue_depth=8,
        collector_queue_depth=8,
        rank=rank,
        num_partition=num_workers,
    )
    # raise NotImplementedError

    # infer shape
    worker_data_shape = dict(train_data.provide_data + train_data.provide_label)
    print(worker_data_shape)
    for key in worker_data_shape:
        worker_data_shape[key] = (pKv.batch_image,) + worker_data_shape[key][1:]
    arg_shape, _, aux_shape = sym.infer_shape(**worker_data_shape)

    _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
    out_shape_dict = list(zip(sym.get_internals().list_outputs(), out_shape))

    _, out_shape, _ = sym.infer_shape(**worker_data_shape)
    terminal_out_shape_dict = zip(sym.list_outputs(), out_shape)

    if rank == 0:
        logger.info('parameter shape')
        logger.info(pprint.pformat([i for i in out_shape_dict if not i[0].endswith('output')]))

        logger.info('intermediate output shape')
        logger.info(pprint.pformat([i for i in out_shape_dict if i[0].endswith('output')]))

        logger.info('terminal output shape')
        logger.info(pprint.pformat([i for i in terminal_out_shape_dict]))

    # memonger
    if pModel.memonger:
        last_block = pModel.memonger_until or ""
        if rank == 0:
            logger.info("do memonger up to {}".format(last_block))

        type_dict = {k: np.float32 for k in worker_data_shape}
        sym = search_plan_to_layer(sym, last_block, 1000, type_dict=type_dict, **worker_data_shape)

    # load and initialize params
    # if begin_epoch != 0:
    #     arg_params, aux_params = load_checkpoint(model_prefix, begin_epoch)
    # elif pModel.from_scratch:
    #     arg_params, aux_params = dict(), dict()
    # else:
    #     arg_params, aux_params = load_checkpoint(pretrain_prefix, pretrain_epoch)

    if pModel.from_scratch:
        arg_params, aux_params = dict(), dict()
    elif pretrain_prefix:
        arg_params, aux_params = load_checkpoint(pretrain_prefix, pretrain_epoch)
    elif begin_epoch != 0:
        arg_params, aux_params = load_checkpoint(model_prefix, begin_epoch)

    if pModel.random:
        mx.random.seed(int(time.time()))
        np.random.seed(int(time.time()))

    init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
    init.set_verbosity(verbose=True)
    # create solver
    fixed_param_prefix = pModel.pretrain.fixed_param
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]

    mod = DetModule(sym, data_names=data_names, label_names=label_names,
                    logger=logger, context=ctx, fixed_param_prefix=fixed_param_prefix)

    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
             for_training=True, force_rebind=False)

    logger.info("distributed training to init module and sync module")
    mod.init_params(initializer=init, arg_params=arg_params, aux_params=aux_params,
                    allow_missing=True, force_init=False)
    if num_workers > 1:
        logger.info("******* {} sleep 20s ensure all workers initialized params *******".format(rank))
        time.sleep(20)
    (arg_params, aux_params) = mod.get_params()

    if pKv.use_horovod:
        logger.info("using horovod for training to sync module")
        # init all module and sync all module params with horovod
        logger.info("******* {} start to broadcast parameters *******".format(rank))
        if arg_params is not None:
            hvd.broadcast_parameters(arg_params, root_rank=0)
            logger.info("******* {} broadcasted arg_params *******".format(rank))
        if aux_params is not None:
            hvd.broadcast_parameters(aux_params, root_rank=0)
            logger.info("******* {} broadcasted aux_params *******".format(rank))
        mod.set_params(arg_params=arg_params, aux_params=aux_params)
    elif num_workers > 1:
        # kvstore("dist_sync")
        logger.info("using kvstore for training to sync module with local broadcast")
        broadcast_params_local(kv, save_path, arg_params, aux_params)

    eval_metrics = mx.metric.CompositeEvalMetric(metric_list)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=pGen.log_frequency)
    epoch_end_callback = callback.do_checkpoint(model_prefix)
    sym.save(model_prefix + ".json")

    # decide learning rate
    if pKv.use_horovod:
        base_lr = pOpt.optimizer.lr * num_workers
    else:
        base_lr = pOpt.optimizer.lr
    lr_factor = pOpt.schedule.lr_factor if hasattr(pOpt.schedule, 'lr_factor') else 0.1

    lr_epoch = lr_step
    lr_epoch_diff = [e - begin_epoch for e in lr_epoch if e > begin_epoch]
    current_lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(e * len(train_data) / input_batch_size) for e in lr_epoch_diff]
    if rank == 0:
        logging.info('lr {}, lr_epoch_diff {}, lr_iters {}'.format(current_lr, lr_epoch_diff, lr_iters))
    lr_mode = pOpt.optimizer.lr_mode or 'step'
    warmup_iters = int(pOpt.warmup.epoch * len(train_data) / input_batch_size)
    iter_per_epoch = int(len(train_data) / input_batch_size)

    if pOpt.warmup is not None and begin_epoch == 0:
        if rank == 0:
            logging.info(
                'warmup lr {}, warmup epoch {}, warmup step {}'.format(
                    pOpt.warmup.lr,
                    pOpt.warmup.epoch,
                    warmup_iters
                )
            )
        if lr_mode == 'cosine':
            warmup_lr_scheduler = AdvancedLRScheduler(
                mode='linear',
                base_lr=pOpt.warmup.lr,
                target_lr=base_lr,
                niters=warmup_iters
            )
            cosine_lr_scheduler = AdvancedLRScheduler(
                mode='cosine',
                base_lr=base_lr,
                target_lr=0,
                offset=warmup_iters,
                niters=(iter_per_epoch * (end_epoch - begin_epoch)) - warmup_iters
            )
            lr_scheduler = LRSequential([warmup_lr_scheduler, cosine_lr_scheduler])
        else:
            lr_scheduler = WarmupMultiFactorScheduler(
                step=lr_iters,
                factor=lr_factor,
                warmup=True,
                warmup_type=pOpt.warmup.type,
                warmup_lr=pOpt.warmup.lr,
                warmup_step=warmup_iters
            )
    else:
        if lr_mode == 'cosine':
            lr_scheduler = AdvancedLRScheduler(
                mode='cosine',
                base_lr=base_lr,
                target_lr=0,
                offset=warmup_iters - iter_per_epoch * begin_epoch,
                niters=iter_per_epoch * end_epoch
            )
        elif len(lr_iters) > 0:
            lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
        else:
            lr_scheduler = None

    # optimizer
    rescale_grad = 1.0 / len(ctx) / num_workers
    # horovod divide the num_workers implicitly.
    if pKv.use_horovod:
        rescale_grad *= num_workers
    if pOpt.optimizer.type == 'sgd':
        optimizer_params = dict(
            momentum=pOpt.optimizer.momentum,
            wd=pOpt.optimizer.wd,
            learning_rate=current_lr,
            lr_scheduler=lr_scheduler,
            rescale_grad=rescale_grad,
            clip_gradient=pOpt.optimizer.clip_gradient
        )
    elif pOpt.optimizer.type == 'adam':
        optimizer_params = dict(
            # momentum=pOpt.optimizer.momentum,
            wd=pOpt.optimizer.wd,
            learning_rate=current_lr,
            lr_scheduler=lr_scheduler,
            # rescale_grad=rescale_grad,
        )
    elif pOpt.optimizer.type == 'adamw' or pOpt.optimizer.type == 'adamws':

        max_update = int(pOpt.schedule.end_epoch * train_data.total_record / pGen.batch_image)
        begin_update = int(pOpt.schedule.begin_epoch * train_data.total_record / pGen.batch_image)
        # print()
        logging.info('total_record of this process: {}, batch_image :{}, max_update:{}'
                     .format(train_data.total_record, pGen.batch_image, max_update))

        optimizer_params = dict(
            wd=pOpt.optimizer.wd,
            learning_rate=current_lr,
            rescale_grad=rescale_grad,
            beta1=pOpt.optimizer.beta1,
            beta2=pOpt.optimizer.beta2,
            clip_gradient=pOpt.optimizer.clip_gradient,
            clip_weight=pOpt.optimizer.clip_weight if hasattr(pOpt.optimizer, 'clip_weight') else None
        )

        optimizer_params['lr_scheduler'] = OneCycleScheduler(
            max_update=max_update,
            begin_update=begin_update,
            lr_max=pOpt.optimizer.lr,
            div_factor=pOpt.optimizer.div_factor,
            pct_start=pOpt.optimizer.pct_start)

        optimizer_params['beta1'] = OneCycleMomentumScheduler(
            max_update=max_update,
            moms=pOpt.optimizer.beta1,
            pct_start=pOpt.optimizer.pct_start)

    if pKv.fp16:
        optimizer_params['multi_precision'] = True
        optimizer_params['rescale_grad'] /= pGen.scale_loss_shift if hasattr(pGen, 'scale_loss_shift') else 128.0

    opt = pOpt.optimizer.type

    if pKv.use_horovod:
        opt = mx.optimizer.create(pOpt.optimizer.type, **optimizer_params)

        # Horovod: wrap optimizer with DistributedOptimizer
        opt = hvd.DistributedOptimizer(opt)

    # print info
    if rank == 0:
        logging.info("lr:{},lr_scheduler:{}, lr step:{}, rescale_grad:{}".format(current_lr, lr_scheduler, lr_step,
                                                                                 rescale_grad))
        logging.info("num_workers:{}, batch size:{}, len(roidb):{}".format(num_workers, input_batch_size, len(roidb)))
        logging.info("epochs:[{},{}], lr:{}, rescale_grad:{}, epoch size:{}".format(begin_epoch, end_epoch,
                                                                                    current_lr, rescale_grad,
                                                                                    int(len(
                                                                                        train_data) / input_batch_size)))
    # train
    mod.fit(
        train_data=train_data,
        eval_metric=eval_metrics,
        epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback,
        kvstore=None if pKv.use_horovod else kv,
        optimizer=opt,
        optimizer_params=optimizer_params,
        initializer=init,
        allow_missing=True,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=begin_epoch,
        num_epoch=end_epoch,
        rank=rank,
        hvd=hvd if pKv.use_horovod else None,
        save_path=save_path,  # used by kvstore sync aux_params
        sync_flag=sync_flag
    )
    # ensure program finish normally in horovod training
    time.sleep(5)
    logging.info("Training has done")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Detection')
    parser.add_argument('--config', help='config file path', type=str)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config


if __name__ == '__main__':
    train_net(parse_args())
