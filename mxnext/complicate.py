from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
# from syncbn_customop import *

__all__ = ["normalizer_factory", "bn_count"]

bn_count = [0]


def normalizer_factory(type="local", ndev=None, eps=1e-5 + 1e-10, mom=0.9):
    """
    :param type: one of "fix", "local", "sync"
    :param ndev:
    :param eps:
    :param mom: momentum of moving mean and moving variance
    :return: a wrapper with signature, bn(data, name)
    """
    # sometimes the normalizer may be pre-constructed
    if callable(type):
        return type

    if type == "local" or type == "localbn":
        def local_bn(data, gamma=None, beta=None, moving_var=None, moving_mean=None,
                     name=None, momentum=mom, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    gamma=gamma,
                                    beta=beta,
                                    moving_var=moving_var,
                                    moving_mean=moving_mean,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=False,
                                    momentum=momentum,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)

        return local_bn

    elif type == "fix" or type == "fixbn":
        def fix_bn(data, gamma=None, beta=None, moving_var=None, moving_mean=None,
                   name=None, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    gamma=gamma,
                                    beta=beta,
                                    moving_var=moving_var,
                                    moving_mean=moving_mean,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=True,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)

        return fix_bn

    elif type == "sync" or type == "syncbn":
        assert ndev is not None, "Specify ndev for sync bn"

        def sync_bn(data, gamma=None, beta=None, moving_var=None, moving_mean=None,
                    name=None, momentum=mom, lr_mult=1.0, wd_mult=1.0):
            bn_count[0] = bn_count[0] + 1
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.contrib.SyncBatchNorm(data=data,
                                                gamma=gamma,
                                                beta=beta,
                                                moving_var=moving_var,
                                                moving_mean=moving_mean,
                                                name=name,
                                                fix_gamma=False,
                                                use_global_stats=False,
                                                momentum=momentum,
                                                eps=eps,
                                                ndev=ndev,
                                                key=str(bn_count[0]),
                                                lr_mult=lr_mult,
                                                wd_mult=wd_mult)

        return sync_bn

    elif type == "hvd" or type == "hvd_syncbn":
        def hvd_syncbn(data, gamma=None, beta=None, moving_mean=None, moving_var=None,
                       name=None, momentum=mom, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_hvd_syncbn"
            # output = data
            output, _, _ = mx.sym.Custom(data,
                                         #  gamma=gamma,
                                         #  beta=beta,
                                         #  moving_mean=moving_mean,
                                         #  moving_var=moving_var,
                                         op_name=name,
                                         eps=eps,
                                         momentum=momentum,
                                         fix_gamma=False,
                                         use_global_stats=False,
                                         op_type="SyncBatchNormOp")
            return output

        return hvd_syncbn
    elif type == "in":
        def in_(data, gamma=None, beta=None,
                name=None, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_in"
            name = name.replace("_bn", "_in")
            return mx.sym.InstanceNorm(data=data,
                                       gamma=gamma,
                                       beta=beta,
                                       name=name,
                                       eps=eps,
                                       lr_mult=lr_mult,
                                       wd_mult=wd_mult)

        return in_

    elif type == "gn":
        def gn(data, gamma=None, beta=None,
               name=None, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_gn"
            name = name.replace("_bn", "_gn")
            return mx.sym.contrib.GroupNorm(data=data,
                                            gamma=gamma,
                                            beta=beta,
                                            name=name,
                                            eps=eps,
                                            num_group=32,
                                            lr_mult=lr_mult,
                                            wd_mult=wd_mult)

        return gn
    else:
        raise KeyError("Unknown norm type {}".format(type))
