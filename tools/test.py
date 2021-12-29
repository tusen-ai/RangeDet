import ctypes
_=ctypes.CDLL('./operator_cxx/contrib/contrib_cxx.so')
import argparse
import copy
import glob
import importlib
import os
from threading import Thread

import mxnet as mx
import numpy as np
import six.moves.cPickle as pkl
from processing_cxx import wnms_4c
# from lidardet.processing_cxx import wnms_4c
from six.moves import reduce
from six.moves.queue import Queue

from utils.detection_input import PostMergeBatchLoader as Loader
from utils.detection_module import DetModule
from utils.load_model import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    parser.add_argument('--config', help='config file path', default='', type=str)
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config


# def sample_roidb(roidb):
#     new_roidb = []
#     with open('path/to/range_tools/segname_to_ts_list.pkl', 'rb') as fr:
#         seg_to_ts_dict = pkl.load(fr)
#     for r in roidb:
#         pc_url = r['pc_url']
#         segname = pc_url.split('/')[-2]
#         ts = int(pc_url.split('/')[-1][:-4])
#         if ts in seg_to_ts_dict[segname]:
#             new_roidb.append(r)
#     return new_roidb


def bbox3d_12dim_to_8dim(bbox3d_12dim):
    center_x = np.mean(bbox3d_12dim[:, [0, 2, 4, 6]], axis=1)
    center_y = np.mean(bbox3d_12dim[:, [1, 3, 5, 7]], axis=1)
    z0 = bbox3d_12dim[:, 9]
    height = bbox3d_12dim[:, 10]
    center_z = z0 + height / 2
    length = np.sqrt((bbox3d_12dim[:, 2] - bbox3d_12dim[:, 0]) ** 2 + (bbox3d_12dim[:, 3] - bbox3d_12dim[:, 1]) ** 2)
    width = np.sqrt((bbox3d_12dim[:, 2] - bbox3d_12dim[:, 4]) ** 2 + (bbox3d_12dim[:, 3] - bbox3d_12dim[:, 5]) ** 2)
    heading = bbox3d_12dim[:, 8]
    score = bbox3d_12dim[:, 11]
    return np.stack([center_x, center_y, center_z, length, width, height, heading, score], axis=1)


def bbox3d_10dim_to_11dim(bbox3d_10dim):
    bbox3d_10dim = np.array(bbox3d_10dim, dtype=np.float32)
    bbox3d_4xy = bbox3d_10dim[:, :8]

    bbox3d_bottom = bbox3d_10dim[:, 8:9]
    bbox3d_top = bbox3d_10dim[:, 9:10]

    bbox3d_yaw = np.arctan2(
        bbox3d_4xy[:, 1] - bbox3d_4xy[:, 3],
        bbox3d_4xy[:, 0] - bbox3d_4xy[:, 2]
    )
    bbox3d_height = bbox3d_top - bbox3d_bottom

    try:
        assert (bbox3d_height >= 0).all()
    except AssertionError:
        print('height < 0')

    bbox3d_11dim = np.concatenate(
        [bbox3d_4xy,
         bbox3d_yaw[:, None],
         bbox3d_bottom,
         bbox3d_height],
        axis=1
    )
    return bbox3d_11dim


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    config = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list, pLabelMap = config.get_config(is_train=False)

    sym = pModel.test_symbol

    image_sets = pDataset.image_set
    data_root = pDataset.data_root
    segments = glob.glob(os.path.join(data_root, image_sets, '*.roidb'))
    roidbs = [pkl.load(open(s, "rb"), encoding="latin1") for s in segments]
    roidb = reduce(lambda x, y: x + y, roidbs)

    # create fake roidb for 5 image area
    if hasattr(pDataset, 'remain_image_area'):
        print(f'remain area containing {pDataset.remain_image_area}')
        fake_roidb = []
        for r in roidb:
            for area in pDataset.remain_image_area:
                r_copy = copy.deepcopy(r)
                r_copy.update({'image_area': area})
                fake_roidb.append(copy.deepcopy(r_copy))
        roidb = fake_roidb

    if hasattr(pDataset, 'minival') and pDataset.minival:
        roidb = sample_roidb(roidb)

    for i, x in enumerate(roidb):
        x["rec_id"] = i
    print(len(roidb))

    loader = Loader(
        roidb=roidb,
        transform=transform,
        data_name=data_name,
        label_name=label_name,
        batch_size=1,
        shuffle=False,
        num_worker=12,
        num_collector=2,
        worker_queue_depth=128,
        collector_queue_depth=16
    )
    # kv=None)

    # print(loader.provide_data)
    # sys.exit(0)

    data_names = [k[0] for k in loader.provide_data]

    execs = []
    all_outputs = []

    data_queue = Queue(maxsize=64)
    result_queue = Queue(maxsize=64)


    def eval_worker(i, data_queue, result_queue):
        ctx = mx.gpu(i)
        arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
        mod = DetModule(sym, data_names=data_names, context=ctx)
        mod.bind(data_shapes=loader.provide_data, for_training=False)
        mod.set_params(arg_params, aux_params, allow_extra=False)
        while True:
            batch = data_queue.get()
            mod.forward(batch, is_train=False)
            out = [x.asnumpy() for x in mod.get_outputs()]
            # out = [x for x in mod.get_outputs()]
            result_queue.put(out)


    # pKv.gpus = [0, ]
    workers = [Thread(target=eval_worker, args=(i, data_queue, result_queue)) for i in pKv.gpus]
    for w in workers:
        w.daemon = True
        w.start()


    def data_worker(loader, data_queue):
        for batch in loader:
            data_queue.put(batch)


    data_thread = Thread(target=data_worker, args=(loader, data_queue))
    data_thread.start()

    min_score = pTest.min_score
    output_dict = {1: {}, 2: {}, 3: {}}
    annotation_dict = {}
    mapping = {'veh': 'TYPE_VEHICLE', 'ped': 'TYPE_PEDESTRIAN', 'cyc': 'TYPE_CYCLIST'}
    for idx in range(loader.total_record):
        if idx % 1000 == 0:
            print(f'{idx} records have been processed, a total of {loader.total_record} records')
        r = result_queue.get()

        assert len(pTest.class_names) == 1
        # (50000), (50000, 10), (1,)
        # (50000), (200, 10), (200)
        rid, \
        cls_score, \
        bbox_4pts, \
        keep_inds, \
        gt_bbox, \
        gt_class = [i.squeeze(0) for i in r]

        rid = int(np.asscalar(rid))

        # select valid nms output
        if not pTest.nms.wnms:
            bbox_4pts = bbox_4pts[keep_inds != -1]
            keep_inds = keep_inds[keep_inds != -1]
            cls_score = cls_score[keep_inds]

        det_per_frame = {}
        fg_inds = cls_score > min_score[pTest.class_names[0]]

        final_cls_score = cls_score[fg_inds]
        final_bbox_4pts = bbox_4pts[fg_inds]
        if final_bbox_4pts.shape[0] == 0:
            continue

        final_bbox_4pts_init = final_bbox_4pts.copy()
        final_bbox_8pts = bbox3d_10dim_to_11dim(final_bbox_4pts)
        final_bbox_score = np.concatenate([final_bbox_8pts, final_cls_score[:, None]], axis=1)
        if pTest.nms.wnms:
            final_bbox_score, keep_inds = wnms_4c(
                final_bbox_score,
                pTest.nms.thr_lo,
                pTest.nms.thr_hi,
                pTest.nms.is_3d_iou,
                100
            )
            final_bbox_score = np.array(final_bbox_score).reshape((-1, 12))

        if final_bbox_score.shape[0] == 0:
            continue

        final_bbox_7pts_score = bbox3d_12dim_to_8dim(final_bbox_score)
        det_per_frame[mapping[pTest.class_names[0]]] = final_bbox_7pts_score

        pc_url = roidb[rid]['pc_url']
        frame_name = pc_url.split('/')[-2].replace('segment-', '').replace('_with_camera_labels', '')
        timestamp = int(pc_url.split('/')[-1][:-4])

        output_dict[rid] = {
            'det_xyzlwhyaws': det_per_frame,
            'meta_info': {'name': frame_name, 'timestamp_micros': timestamp}}
        annotation_dict[rid] = gt_bbox

    with open(pTest.model.prefix + '_output_dict_{}e.pkl'.format(pTest.model.epoch), 'wb') as fw:
        pkl.dump(annotation_dict, fw)
        pkl.dump(output_dict, fw)
        print('Output dict has been saved!')

