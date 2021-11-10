import mxnet as mx
import numpy as np
import time
import warnings
import math
from functools import partial


class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start = time.time()

    def toc(self):
        self.time += time.time() - self.start
        self.count += 1

    def get(self):
        return self.time / self.count

    def reset(self):
        self.time = 0
        self.count = 0


class OneCycleScheduler(object):
    """ Reduce the learning rate according to a cosine function
    """

    def __init__(self, max_update, lr_max=0.003, div_factor=10.0, pct_start=0.4, begin_update=0):
        assert isinstance(max_update, int)
        self.max_update = max_update
        self.begin_update = begin_update
        self.lr_max = lr_max
        self.base_lr = lr_max
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.warmup_steps = int(max_update * self.pct_start)

        low_lr = self.lr_max / self.div_factor
        self.lr_phases = (partial(OneCycleScheduler.annealing_cos, low_lr, self.lr_max),
                          partial(OneCycleScheduler.annealing_cos, self.lr_max, low_lr / 1e4))
        print('lr_phases:', self.lr_phases)

    @staticmethod
    def annealing_cos(start, end, pct):
        # print(pct, start, end)
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

    def __call__(self, num_update):
        if self.begin_update > 0:
            num_update += self.begin_update
        if num_update <= self.warmup_steps:
            self.base_lr = self.lr_phases[0](float(num_update) / float(self.warmup_steps))
        elif num_update <= self.max_update:
            self.base_lr = self.lr_phases[1](float(num_update - self.warmup_steps) /
                                             float(self.max_update - self.warmup_steps))
        return self.base_lr


class OneCycleMomentumScheduler(object):
    """ Reduce the momentum according to a cosine function
    """

    def __init__(self, max_update, moms=[0.95, 0.85], pct_start=0.4):
        assert isinstance(max_update, int)
        self.max_update = max_update
        self.moms = moms
        self.pct_start = pct_start
        self.warmup_steps = int(max_update * self.pct_start)

        self.lr_phases = (partial(OneCycleScheduler.annealing_cos, moms[0], moms[1]),
                          partial(OneCycleScheduler.annealing_cos, moms[1], moms[0]))
        print('mom_phases:', self.lr_phases)

    @staticmethod
    def annealing_cos(start, end, pct):
        # print(pct, start, end)
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

    def __call__(self, num_update):
        if num_update <= self.warmup_steps:
            self.mom = self.lr_phases[0](float(num_update) / float(self.warmup_steps))
        elif num_update <= self.max_update:
            self.mom = self.lr_phases[1](float(num_update - self.warmup_steps) /
                                         float(self.max_update - self.warmup_steps))
        return self.mom


def clip_global_norm(arrays, max_norm, check_isfinite=True):
    """Rescales NDArrays so that the sum of their 2-norm is smaller than `max_norm`.

    Parameters
    ----------
    arrays : list of NDArray
    max_norm : float
    check_isfinite : bool, default True
         If True, check that the total_norm is finite (not nan or inf). This
         requires a blocking .asscalar() call.

    Returns
    -------
    NDArray or float
      Total norm. Return type is NDArray of shape (1,) if check_isfinite is
      False. Otherwise a float is returned.

    """

    def _norm(array):
        if array.stype == 'default':
            x = array.reshape((-1,))
            return mx.ndarray.dot(x, x)
        return array.norm().square()

    assert len(arrays) > 0
    ctx = arrays[0][0].context
    total_norm = mx.ndarray.add_n(*[_norm(arr[0]).as_in_context(ctx) for arr in arrays])
    total_norm = mx.ndarray.sqrt(total_norm)
    if check_isfinite:
        if not np.isfinite(total_norm.asscalar()):
            warnings.warn(
                UserWarning('nan or inf is detected. '
                            'Clipping results will be undefined.'), stacklevel=2)
    scale = max_norm / (total_norm + 1e-8)
    scale = mx.ndarray.min(mx.ndarray.concat(scale, mx.ndarray.ones(1, ctx=ctx), dim=0))
    for arr in arrays:
        arr[0] *= scale.as_in_context(arr[0].context)
    if check_isfinite:
        return total_norm.asscalar()
    else:
        return total_norm


@mx.optimizer.Optimizer.register
class AdamW(mx.optimizer.Optimizer):
    """The Adam optimizer with weight decay regularization.
    Updates are applied by::
        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * (m / (sqrt(v) + epsilon) + wd * w)
    Note that this is different from `mxnet.optimizer.Adam`, where L2 loss is added and
    accumulated in m and v. In AdamW, the weight decay term decoupled from gradient
    based update.
    This is also slightly different from the AdamW optimizer described in
    *Fixing Weight Decay Regularization in Adam*, where the schedule multiplier and
    learning rate is decoupled. The BERTAdam optimizer uses the same learning rate to
    apply gradients w.r.t. the loss and weight decay.
    This optimizer accepts the following parameters in addition to those accepted
    by :class:`mxnet.optimizer.Optimizer`.
    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 **kwargs):
        if 'clip_weight' in kwargs:
            self.clip_weight = kwargs.pop('clip_weight')
        super(AdamW, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):  # pylint-disable=unused-argument
        """Initialization for mean and var."""
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def _get_mom(self):
        if isinstance(self.beta1, float):
            return self.beta1
        else:
            return self.beta1(self.num_update)

    def update(self, index, weight, grad, state):
        """Update method."""
        try:
            from mxnet.ndarray.contrib import adamw_update
        except ImportError:
            raise ImportError("Failed to import nd.contrib.adamw_update from MXNet. "
                              "BERTAdam optimizer requires mxnet>=1.5.0b20181228. "
                              "Please upgrade your MXNet version.")
        # print(type(weight), len(weight))
        # assert(isinstance(weight, mx.nd.NDArray))
        # assert(isinstance(grad, mx.nd.NDArray))
        if isinstance(index, list):
            assert len(index) == 1
            index = index[0]
        if isinstance(weight, list):
            assert len(weight) == 1
            weight = weight[0]
        if isinstance(grad, list):
            assert len(grad) == 1
            grad = grad[0]
        if isinstance(state, list):
            assert len(state) == 1
            state = state[0]

        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        beta1 = self._get_mom()

        t = self._index_update_count[index]
        coef1 = 1. - beta1 ** t
        coef2 = 1. - self.beta2 ** t
        lr *= math.sqrt(coef2) / coef1

        kwargs = {'beta1': beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        mean, var = state
        adamw_update(weight, grad, mean, var, out=weight, lr=1, wd=wd, eta=lr, **kwargs)

        if self.clip_weight is not None and len(weight.shape) == 4:
            pass
            # if weight.abs().sum() > 0:
            #     weight[:] = weight * 0.9
            # weight[:] = mx.nd.clip(weight, self.clip_weight * -1, self.clip_weight)


@mx.optimizer.Optimizer.register
class AdamWS(mx.optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 **kwargs):
        super(AdamWS, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):  # pylint-disable=unused-argument
        """Initialization for mean and var."""
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def _get_mom(self):
        if isinstance(self.beta1, float):
            return self.beta1
        else:
            return self.beta1(self.num_update)

    def update(self, index, weight, grad, state):
        """Update method."""
        try:
            from mxnet.ndarray.contrib import adamw_update
        except ImportError:
            raise ImportError("Failed to import nd.contrib.adamw_update from MXNet. "
                              "BERTAdam optimizer requires mxnet>=1.5.0b20181228. "
                              "Please upgrade your MXNet version.")
        # print(type(weight), len(weight))
        # assert(isinstance(weight, mx.nd.NDArray))
        # assert(isinstance(grad, mx.nd.NDArray))
        if isinstance(index, list):
            assert len(index) == 1
            index = index[0]
        if isinstance(weight, list):
            assert len(weight) == 1
            weight = weight[0]
        if isinstance(grad, list):
            assert len(grad) == 1
            grad = grad[0]
        if isinstance(state, list):
            assert len(state) == 1
            state = state[0]

        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        beta1 = self._get_mom()

        t = self._index_update_count[index]
        coef1 = 1. - beta1 ** t
        coef2 = 1. - self.beta2 ** t
        lr *= math.sqrt(coef2) / coef1

        kwargs = {'beta1': beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        mean, var = state
        adamw_update(weight, grad, mean, var, out=weight, lr=1, wd=wd, eta=lr, **kwargs)
        # print(weight.shape)

        if len(weight.shape) == 4:
            weight_mean = weight.mean(keepdims=True, axis=(1, 2, 3))
            weight_std = ((weight - weight_mean) ** 2).mean(keepdims=True, axis=(1, 2, 3)) ** 0.5 + 1e-10
            weight[:] = (weight - weight_mean) / weight_std
