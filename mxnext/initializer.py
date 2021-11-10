# encoding: utf-8
"""
MXNeXt is a wrapper around the original MXNet Symbol API
@version: 0.1
@author:  Yuntao Chen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import mxnet as mx
import numpy as np


def gauss(std):
    return mx.init.Normal(sigma=std)


def one_init():
    return mx.init.One()


def zero_init():
    return mx.init.Zero()

def constant(val):
    return mx.init.Constant(val)
