from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import assigner
import numpy as np
from numba import jit, njit


@njit
def sample_data(data, slice_begin=None, stride_width=None):
    """
    Only supports sampling three-dimensional arrays
    :param data: (C, H, W) or (H, W, C)
    :param c_dim: channel_dim: str 'front' 'rear'
    :param stride_height: int (stride in vertical axis)
    :param stride_width: int (stride in horizontal axis)
    :return: stride sample data
    """
    data = data.copy()

    data_width = data.shape[2]
    s_w = slice(slice_begin, data_width, stride_width)
    data = data[:, :, s_w]
    return data


def class_aware_expand(data, class_target, num_classes):
    """
    num_pts = H * W
    :param data: (num_pts, num_channel)
    :param class_target: (num_pts) # Start from zero and consider the background category, background category is the last dimension
    :param num_classes: (int) num fg classes
    :return:  (num_pts, num_fg_classes, num_channel)
    """
    output_data = np.zeros(shape=(data.shape[0], num_classes + 1, data.shape[1]), dtype=np.float32)
    output_data[:, class_target, :] = data
    return output_data[:, :-1, :]


@jit(nopython=True, nogil=True)
def jit_class_aware_expand(data, class_target, num_classes):
    """
    num_pts = H * W
    :param data: (num_pts, num_channel)
    :param class_target: (num_pts) # Start from zero and consider the background category, background category is the last dimension
    :param num_classes: (int) num fg classes
    :return:  (num_pts, num_fg_classes, num_channel)
    """
    output_data = np.zeros(shape=(data.shape[0], num_classes + 1, data.shape[1]), dtype=np.float32)
    for i in range(data.shape[0]):
        output_data[i, class_target[i], :] = data[i]
    return output_data[:, :-1, :]


def inv_points_frequency(bbox_inds):
    """
    :param bbox_inds: (H * W)
    :return: normalization_weight: (H * W)
    """
    # (H * W)
    num_pts_in_bbox = assigner.get_point_num(
        bbox_inds.astype(np.float32)).reshape(-1)
    normalization_weight = 1 / num_pts_in_bbox
    return normalization_weight
