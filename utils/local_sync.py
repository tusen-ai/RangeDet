import os
import time
import mxnet as mx


def load_params(path):
    save_dict = mx.nd.load(path)
    tmp_args = {}
    tmp_auxs = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            tmp_args[name] = v
        if tp == 'aux':
            tmp_auxs[name] = v
    return tmp_args, tmp_auxs


def sync_params_local(kv, epoch, save_path, arg_params, aux_params):
    rank = kv.rank
    num_workers = kv.num_workers
    # save params
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    param_name = os.path.join(save_path, 'tmp_for_sync_%s_%s.params' % (rank, epoch))
    mx.nd.save(param_name, save_dict)
    while 1:
        isOK = True
        for workeri in range(num_workers):
            params_path = os.path.join(save_path, 'tmp_for_sync_%s_%s.params' % (workeri, epoch))
            if os.path.exists(params_path) is False:
                isOK = False
                print("[{}/{}] wait {} params file to sleep 2s.".format(rank, num_workers, workeri))
                time.sleep(2)
                break
        if isOK:
            break

    print("[{}/{}] sleeping 2s ensure all params saving is done.".format(rank, num_workers))
    time.sleep(2)

    # load all workers'params
    args = dict()
    auxs = dict()
    for workeri in range(num_workers):
        path = os.path.join(save_path, 'tmp_for_sync_%s_%s.params' % (workeri, epoch))
        tmp_arg, tmp_aux = load_params(path)
        for k, v in tmp_arg.items():
            if k in args.keys():
                args[k].append(v)
            else:
                args[k] = [v]
        for k, v in tmp_aux.items():
            if k in auxs.keys():
                auxs[k].append(v)
            else:
                auxs[k] = [v]
    # do allreduce
    for name, block in args.items():
        assert len(block) == num_workers, "length of {} is not equal to num_workers({} vs {})".format(name, len(block),
                                                                                                      num_workers)
        weight = sum(block) / len(block)
        weight.astype(arg_params[name].dtype).copyto(arg_params[name])
    for name, block in auxs.items():
        assert len(block) == num_workers, "length of {} is not equal to num_workers({} vs {})".format(name, len(block),
                                                                                                      num_workers)
        weight = sum(block) / len(block)
        weight.astype(aux_params[name].dtype).copyto(aux_params[name])
    # sleep 2s ensure all workers' allreduce is done
    time.sleep(2)
    try:
        os.remove(param_name)
    except:
        print("[{}/{}] some error when remove {}".format(rank, num_workers, param_name))


def broadcast_params_local(kv, save_path, arg_params, aux_params):
    # save params for rank==0
    rank = kv.rank
    num_workers = kv.num_workers
    param_name = os.path.join(save_path, 'tmp_for_broadcast_0.params')
    if rank == 0:
        save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        mx.nd.save(param_name, save_dict)

    while 1:
        isOK = True
        if os.path.exists(param_name) is False:
            isOK = False
            print("[{}/{}] wait 0 params file to sleep 2s.".format(rank, num_workers))
            time.sleep(2)
        if isOK:
            break
    print("[{}/{}] sleeping 2s ensure params saving is done.".format(rank, num_workers))
    time.sleep(2)
    arg, aux = load_params(param_name)
    # do broadcast
    for name, param in arg.items():
        param.astype(arg_params[name].dtype).copyto(arg_params[name])
    for name, param in aux.items():
        param.astype(aux_params[name].dtype).copyto(aux_params[name])
