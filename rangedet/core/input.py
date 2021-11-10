from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import processing_cxx
from utils.detection_input import DetectionAugmentation

from rangedet.core.util_func import jit_class_aware_expand, sample_data

EPS = 1e-3


class LoadRecord(DetectionAugmentation):
    """
    Load data extracted from raw data that needs to be continually processed.
    note:
         Data address recorded by pc_url
    """

    def __init__(self):
        super(LoadRecord, self).__init__()

    def apply(self, input_record):
        # load record data
        pc_url = input_record["pc_url"]
        npkl = np.load(pc_url)

        # (64, 2650, 3)
        input_record['pc_vehicle_frame'] = npkl["pc_vehicle_frame"].astype(np.float32)
        # (64, 2650, 4)
        input_record['range_image'] = npkl["range_image"].astype(np.float32)

        # (64)
        input_record['inclination'] = npkl['inclination'].astype(np.float32)
        # (2650)
        input_record['azimuth'] = npkl['azimuth'].astype(np.float32)

        # mind the gap
        input_record['range_image_mask'] = input_record['range_image'][..., 0:1] > 0
        input_record['pc_vehicle_frame'][~input_record['range_image_mask'][..., 0]] = 0
        input_record['range_image_mask'] = input_record['range_image_mask'].astype(np.float32)


class LoadGTInfo(DetectionAugmentation):
    """
    Convert the format of GT data which already in the input_record dict.
    """

    def __init__(self):
        super(LoadGTInfo, self).__init__()

    def apply(self, input_record):
        input_record["gt_class"] = input_record["gt_class"].astype(np.float32)  # (16)
        input_record["gt_bbox_yaw"] = input_record["gt_bbox_yaw"].astype(np.float32)  # (16)
        input_record["gt_bbox_csa"] = input_record['gt_bbox_csa'].astype(np.float32)  # (16, 7)
        input_record["gt_bbox_imu"] = input_record['gt_bbox_imu'].astype(np.float32)  # (16, 8, 3)
        input_record["meta_data"] = input_record['meta_data'].astype(np.float32)  # (16, 4)
        input_record["points_in_box"] = input_record['points_in_box'].astype(np.float32)  # (16)


class FilterGTClass(DetectionAugmentation):
    """
    Filtering GTs that need to be trained.
    """

    def __init__(self, valid_class):
        super(FilterGTClass, self).__init__()
        self.valid_class = valid_class  # [1]

    def apply(self, input_record):
        if input_record["gt_class"].size > 0:
            valid_class = np.any([input_record["gt_class"] == i for i in self.valid_class], axis=0)
            input_record["gt_class"] = input_record["gt_class"][valid_class]
            input_record["gt_bbox_imu"] = input_record["gt_bbox_imu"][valid_class]
            input_record["gt_bbox_csa"] = input_record['gt_bbox_csa'][valid_class]
            input_record["gt_bbox_yaw"] = input_record["gt_bbox_yaw"][valid_class]
            input_record["points_in_box"] = input_record['points_in_box'][valid_class]

        # Do not use elif here
        if input_record["gt_class"].size == 0:
            input_record["gt_class"] = np.zeros((1,), dtype=np.float32)
            input_record["gt_bbox_imu"] = np.zeros((1, 8, 3), dtype=np.float32)
            input_record["gt_bbox_csa"] = np.zeros((1, 7), dtype=np.float32)
            input_record["gt_bbox_yaw"] = np.zeros((1,), dtype=np.float32)
            input_record["points_in_box"] = np.zeros((1,), dtype=np.float32)


class ProcessMissValue(DetectionAugmentation):
    """
    Fill invalid null values by using method like median filtering.
    """

    def __init__(self):
        super(ProcessMissValue, self).__init__()
        self.pc_fill_value = np.array([0, 0, 0])
        self.range_fill_value = np.array([80, 0, 0, -1])

    @staticmethod
    def fill_noise(data, miss_inds, width):
        data_shift1pxl = data[:, list(range(1, width)) + [0, ], :]
        data[miss_inds, :] = data_shift1pxl[miss_inds, :]
        return data

    def apply(self, input_record):
        range_image = input_record['range_image']
        pc_vehicle_frame = input_record['pc_vehicle_frame']
        range_image_mask = input_record['range_image'][..., 0] > 0

        height, width, _ = range_image.shape
        miss_inds = range_image[:, :, 0] == -1

        range_image = self.fill_noise(range_image, miss_inds, width)
        pc_vehicle_frame = self.fill_noise(pc_vehicle_frame, miss_inds, width)
        range_image_mask = self.fill_noise(range_image_mask[:, :, None], miss_inds, width).squeeze()

        still_miss_inds = range_image[:, :, 0] == -1

        shift_down_2px = range_image[[height - 2, height - 1] + list(range(height - 2)), :, 0]
        shift_top_2px = range_image[list(range(2, height)) + [0, 1], :, 0]
        shift_right_2px = range_image[:, [width - 2, width - 1] + list(range(width - 2)), 0]
        shift_left_2px = range_image[:, list(range(2, width)) + [0, 1], 0]

        car_window_mask = still_miss_inds & ((shift_down_2px != -1) | (shift_top_2px != -1) |
                                             (shift_right_2px != -1) | (shift_left_2px != -1))

        range_image[still_miss_inds, :] = self.range_fill_value
        pc_vehicle_frame[still_miss_inds, :] = self.pc_fill_value

        # How much are the intensity and elongation of car windows
        range_image[car_window_mask, :] = np.array([0, 0, 0, -1])
        pc_vehicle_frame[car_window_mask, :] = np.array([0, 0, 0])

        input_record['car_window_mask'] = car_window_mask.astype(np.float32)[None, :, :]
        input_record['range_image'] = range_image
        input_record['pc_vehicle_frame'] = pc_vehicle_frame
        input_record['range_image_mask'] = range_image_mask.astype(np.float32)[:, :, None]


class SepAndClipData(DetectionAugmentation):
    """
    Do data clipping for data with different attributes separately.
    """

    def __init__(self, param):
        super(SepAndClipData, self).__init__()
        self.clip_data_dict = param.clip_data_dict
        # no need to clip azimuth
        self.clip_data_dict.pop('azimuth')

    def apply(self, input_record):
        input_record['range_value'] = input_record['range_image'][:, :, 0].copy()
        input_record['intensity'] = input_record['range_image'][:, :, 1].copy()
        input_record['elongation'] = input_record['range_image'][:, :, 2].copy()

        input_record['pc_vehicle_frame_x'] = input_record['pc_vehicle_frame'][:, :, 0].copy()
        input_record['pc_vehicle_frame_y'] = input_record['pc_vehicle_frame'][:, :, 1].copy()
        input_record['pc_vehicle_frame_z'] = input_record['pc_vehicle_frame'][:, :, 2].copy()

        (height, width) = input_record['range_value'].shape

        inclination = input_record["inclination"].copy()
        inclination_tile = np.tile(inclination[:, np.newaxis], (1, width))
        input_record['inclination'] = inclination_tile

        input_record['azimuth'] = np.arctan2(
            input_record['pc_vehicle_frame_y'], input_record['pc_vehicle_frame_x'])
        # input_record['azimuth'] = np.clip(input_record['azimuth'], -np.pi, np.pi)

        for clip_name, (clip_min, clip_max) in self.clip_data_dict.items():
            input_record[clip_name] = np.clip(input_record[clip_name], clip_min, clip_max)


class GetUnnormalizedRange(DetectionAugmentation):
    """
    Record the unnormalized range data because GenerateFPNTarget will use it.
    """

    def __init__(self):
        super(GetUnnormalizedRange, self).__init__()

    def apply(self, input_record):
        input_record['unnormalized_range'] = input_record['range_value'][:, :, np.newaxis].copy()


class NormData(DetectionAugmentation):
    """
    Normalized data.
    """

    def __init__(self, param):
        super(NormData, self).__init__()
        self.norm_data_dict = param.norm_data_dict

    def apply(self, input_record):
        for norm_name, (data_mean, data_var) in self.norm_data_dict.items():
            input_record[norm_name] = (input_record[norm_name] - data_mean) / (data_var ** 0.5)


class GetCoordinates(DetectionAugmentation):
    """
    Record the normalized pc_vehicle_frame data because MetaKernel will use it.
    """

    def __init__(self):
        super(GetCoordinates, self).__init__()

    def apply(self, input_record):
        input_record['coord'] = np.stack([
            input_record['pc_vehicle_frame_x'],
            input_record['pc_vehicle_frame_y'],
            input_record['pc_vehicle_frame_z'],
        ], axis=2).copy()


class CombineData(DetectionAugmentation):
    """
    Combining data as input data.
    """

    def __init__(self, param):
        super(CombineData, self).__init__()
        self.combine_name_dict = param.combine_name_dict

    def apply(self, input_record):
        for new_name, name_list in self.combine_name_dict.items():
            new_data = np.stack([input_record[n] for n in name_list], axis=2)  # (64, 2650, 8)
            input_record[new_name] = new_data
            [input_record.pop(n) for n in name_list]


class GetFixedLengthGTBbox(DetectionAugmentation):
    """
    We train the classification head to predict iou-aware classification score,
    so it is required to calculate iou between pred bbox and gt bbox.
    Here it is padded to a fixed length because mxnet's custom op only accepts fixed-length input.
    """

    def __init__(self, param):
        super(GetFixedLengthGTBbox, self).__init__()
        self.class_type = param.class_type
        self.fixed_length = param.fixed_length

    def apply(self, input_record):
        for c_type in self.class_type:
            name = '_'.join(['gt_bbox', c_type[5:8].lower(), 'for_iou_pred'])
            input_record[name] = self.get_fixed_length_gt_bbox(
                input_record['gt_bbox_imu'], input_record['gt_class'], class_type=c_type)

    @staticmethod
    def get_fixed_length_gt_bbox(gt_bbox, gt_class, class_type, fixed_length=200):
        assert gt_bbox.shape[0] == gt_class.shape[0]
        assert gt_bbox.shape[1:] == (8, 3)
        type_dict = {
            'TYPE_UNKNOWN': 0,
            'TYPE_VEHICLE': 1,
            'TYPE_PEDESTRIAN': 2,
            'TYPE_SIGN': 3,
            'TYPE_CYCLIST': 4,
        }
        filtered_gt_bbox = gt_bbox[gt_class == type_dict[class_type]].copy()
        filtered_gt_bbox = filtered_gt_bbox[:, :4, :2].reshape(-1, 8)

        fixed_len_gt_bbox = np.full((fixed_length, 8), EPS, dtype=np.float32)
        fixed_len_gt_bbox[:, :] = np.array([0, 0, 0, EPS, EPS, EPS, EPS, 0], dtype=np.float32)

        if filtered_gt_bbox.size == 0:
            return fixed_len_gt_bbox

        assert filtered_gt_bbox.shape[0] < 200, \
            f"The number of GT boxes is greater than {fixed_length}"
        fixed_len_gt_bbox[:filtered_gt_bbox.shape[0], :] = filtered_gt_bbox
        return fixed_len_gt_bbox


class Bbox3dAssigner(DetectionAugmentation):
    """
    Give each point cloud a label of which 3D bounding box it belongs to.
    """

    def __init__(self, param=None):
        super(Bbox3dAssigner, self).__init__()
        self.height = param.feat_size[0]
        self.width = param.feat_size[1]

    def apply(self, input_record):
        # bbox_inds_each_pt.shape (64 * 2650,) min -1 max ..
        bbox3d_ind_of_each_pt = self.get_faster_bbox3d_ind_assigner(input_record)
        input_record['bbox3d_ind_of_each_pt'] = \
            bbox3d_ind_of_each_pt.reshape((self.height, self.width, 1)).copy()

    @staticmethod
    def get_faster_bbox3d_ind_assigner(input_record):
        is_in_nlz = np.zeros((64, 2650), dtype=np.float32)
        pc_vehicle_frame = input_record['pc_vehicle_frame'].copy()
        gt_bbox = input_record['gt_bbox_imu'].copy()
        pc_mask = input_record['range_image_mask'][:, :, 0].copy()

        radius = np.ones((len(gt_bbox),)) * 100
        radius = radius.astype(np.float32)
        bbox_center = gt_bbox.mean(axis=1)

        max_x = float(gt_bbox[:, :, 0].max())
        min_x = float(gt_bbox[:, :, 0].min())
        max_y = float(gt_bbox[:, :, 1].max())
        min_y = float(gt_bbox[:, :, 1].min())
        max_z = float(gt_bbox[:, :, 2].max())
        min_z = float(gt_bbox[:, :, 2].min())
        max_dist = 20.0

        bbox_inds_each_pt = processing_cxx.assign3D_v2(
            pc_vehicle_frame.reshape(-1, 3),
            gt_bbox.reshape(-1, 24),
            bbox_center.reshape(-1, 3),
            radius.reshape(-1, 1),
            pc_mask.reshape(-1, 1),
            is_in_nlz.reshape(-1, 1),
            max_x, min_x, max_y, min_y, max_z, min_z, max_dist
        ).reshape(-1)
        return bbox_inds_each_pt.reshape(-1)


class GenerateTarget(DetectionAugmentation):
    """
    rpn_reg_target -> Give each point cloud in the 3D bounding box a regression target.
    reg_normalize_weight -> Give each point cloud in the 3D bounding box a normalize weight
    which implements the normalization by dividing by the number of points in the box.
    reg_normalize_weight -> Simply normalized by the number of regression dimensions.
    Note:
        The regression target takes into account the observation orientation.
        rpn_cls_target can be used if the classification head not be used to
        predict iou-aware score.
    """

    def __init__(self, param):
        super(GenerateTarget, self).__init__()
        self.input_height = param.feat_size[0]
        self.input_width = param.feat_size[1]

        self.reg_weight = param.reg_weight
        self.num_reg_dim = len(self.reg_weight)

        self.num_classes = param.num_classes
        self.label_set = param.label_set

    def apply(self, input_record):
        bbox_inds_each_pt = input_record['bbox3d_ind_of_each_pt']

        # rpn_reg_target.shape (169984, 8)
        rpn_reg_target = self.get_rpn_reg_target(
            input_record['pc_vehicle_frame'],
            input_record["gt_bbox_csa"],
            bbox_inds_each_pt
        )

        # normalize weight for reg (169600, 8)
        reg_normalize_weight = self.get_normalization_weight(
            bbox_inds_each_pt
        )
        reg_normalize_weight = np.tile(
            reg_normalize_weight[:, np.newaxis],
            (1, self.num_reg_dim)
        )

        # dim weight for reg (169600, 8)
        rpn_reg_weight = self.get_rpn_reg_weight(
            bbox_inds_each_pt,
            self.reg_weight
        )

        # Start from zero and consider the background category
        # background category is the last dimension (169600,)
        rpn_cls_target = self.get_rpn_cls_target(
            input_record['gt_class'],
            bbox_inds_each_pt
        )

        rpn_cls_target_onehot = np.ones_like(rpn_cls_target)[:, np.newaxis]
        rpn_cls_target_onehot = jit_class_aware_expand(
            rpn_cls_target_onehot,
            rpn_cls_target,
            self.num_classes)
        rpn_cls_target_onehot = rpn_cls_target_onehot[:, 0].reshape(
            (self.input_height, self.input_width, self.num_classes))

        data_list = [rpn_reg_target, reg_normalize_weight, rpn_reg_weight]
        result_list = self.batch_process_class_aware_expand(data_list, rpn_cls_target)
        [rpn_reg_target, reg_normalize_weight, rpn_reg_weight] = result_list

        input_record['rpn_reg_target'] = rpn_reg_target
        input_record['reg_normalize_weight'] = reg_normalize_weight
        input_record['rpn_reg_weight'] = rpn_reg_weight
        input_record['rpn_cls_target'] = rpn_cls_target_onehot

    def batch_process_class_aware_expand(self, data_list, rpn_cls_target):
        if self.num_classes == 1:
            data_list = [data.reshape((
                self.input_height,
                self.input_width,
                self.num_classes * self.num_reg_dim)) for data in data_list]
            return data_list

        result_list = []
        for data in data_list:
            p_result = jit_class_aware_expand(
                data,
                rpn_cls_target,
                self.num_classes
            )
            p_result = p_result.reshape((
                self.input_height,
                self.input_width,
                self.num_classes * self.num_reg_dim))
            result_list.append(p_result)
        return result_list

    def get_rpn_cls_target(self, gt_class, bbox3d_ind):
        bbox3d_ind = bbox3d_ind.reshape(-1)
        label_mapping = {
            label: i for i, label in enumerate(self.label_set, start=0)}
        # Do that because it will be no object in this range image.
        label_mapping.update({0: 0})
        gt_class_mapping = np.array(
            [label_mapping[int(c)] for c in gt_class], dtype=np.int32)
        rpn_cls_target = gt_class_mapping[bbox3d_ind]

        rpn_cls_target[bbox3d_ind == -1] = len(self.label_set)
        assert rpn_cls_target.dtype == np.int32
        return rpn_cls_target

    @staticmethod
    def get_normalization_weight(bbox_inds_each_pt):
        pt_num_in_bbox = processing_cxx.get_point_num(
            bbox_inds_each_pt.reshape(-1).astype(np.float32)
        ).reshape(-1)
        normalization_weight = 1 / pt_num_in_bbox
        normalization_weight[normalization_weight == -1] = 0
        return normalization_weight

    @staticmethod
    def get_rpn_reg_weight(bbox3d_ind, reg_dim_weights):
        assert isinstance(reg_dim_weights, list)
        bbox3d_ind = bbox3d_ind.reshape(-1)
        inbox_pts = bbox3d_ind > -1
        reg_dim_weights = np.array(reg_dim_weights, np.float32)

        num_dim = len(reg_dim_weights)
        rpn_reg_weight = np.zeros((bbox3d_ind.shape[0], num_dim), np.float32)
        rpn_reg_weight[inbox_pts] = reg_dim_weights
        return rpn_reg_weight

    def get_rpn_reg_target(self,
                           pc_vehicle_frame,
                           gt_bbox_3d,
                           bbox3d_ind_of_each_pt,
                           delta_bottom_height=False):
        assert len(pc_vehicle_frame.shape) == 3
        assert len(gt_bbox_3d.shape) == 2
        assert len(bbox3d_ind_of_each_pt.shape) == 3

        pc_vehicle_frame = pc_vehicle_frame.reshape(-1, 3)
        bbox3d_ind_of_each_pt = bbox3d_ind_of_each_pt.reshape(-1)
        inbox_pts = bbox3d_ind_of_each_pt > -1

        # Do that because it will be no object in this range image.
        if inbox_pts.sum() == 0:
            return np.zeros((pc_vehicle_frame.shape[0], 8), dtype=np.float32)

        pc_bbox3d = gt_bbox_3d[bbox3d_ind_of_each_pt]
        pc_azimuth = np.arctan2(pc_vehicle_frame[:, 1], pc_vehicle_frame[:, 0])

        pc_bbox3d_yaw = pc_bbox3d[:, -1]
        pc_delta_yaw = pc_bbox3d_yaw - pc_azimuth
        yaw_sin = np.sin(pc_delta_yaw)
        yaw_cos = np.cos(pc_delta_yaw)

        # rotation_tile = np.tile(rotation, (pc_vehicle_frame.shape[0], 1, 1))
        rotation = self.rot_alone_z(pc_azimuth)

        pc_bbox3d_xyz = pc_bbox3d[:, :3]
        delta_pc_xyz = pc_bbox3d_xyz - pc_vehicle_frame
        delta_observate_pc_xyz = np.einsum('nij,nj->ni', rotation, delta_pc_xyz)

        delta_observate_pc_xyz = \
            np.sqrt(np.abs(delta_observate_pc_xyz)) * np.sign(delta_observate_pc_xyz)
        delta_x_azi_frame = delta_observate_pc_xyz[:, 0]
        delta_y_azi_frame = delta_observate_pc_xyz[:, 1]
        delta_z_azi_frame = delta_observate_pc_xyz[:, 2]

        log_length = np.log(pc_bbox3d[:, 3])
        log_width = np.log(pc_bbox3d[:, 4])
        log_height = np.log(pc_bbox3d[:, 5])

        bottom_height = pc_bbox3d[:, 2] - pc_bbox3d[:, 5] / 2

        if delta_bottom_height:
            pc_reg_target = np.stack(
                (delta_x_azi_frame, delta_y_azi_frame, delta_z_azi_frame,
                 log_width, log_length, log_height, yaw_cos, yaw_sin), axis=1)
        else:
            pc_reg_target = np.stack(
                (delta_x_azi_frame, delta_y_azi_frame, log_width, log_length,
                 yaw_cos, yaw_sin, bottom_height, log_height), axis=1)

        pc_reg_target[~inbox_pts] = 0
        return pc_reg_target

    @staticmethod
    def rot_alone_z(azimuth):
        cos = np.cos(azimuth)
        sin = np.sin(azimuth)
        ones = np.ones_like(azimuth, np.float32)
        zeros = np.zeros_like(azimuth, np.float32)
        # rotation should be clockwise
        rotation = np.stack([cos, sin, zeros,
                             -sin, cos, zeros,
                             zeros, zeros, ones], axis=1)
        rotation = rotation.reshape((-1, 3, 3))
        return rotation


class PadData(DetectionAugmentation):
    """
    Padding data to the specified size.
    """

    def __init__(self, param):
        super(PadData, self).__init__()
        self.pad_name_list = param.pad_name_list
        self.pad_short = param.pad_short
        self.pad_long = param.pad_long

    def apply(self, input_record):
        for name in self.pad_name_list:
            assert len(input_record[name].shape) == 3, f'{name}'
            assert input_record[name].shape[-1] <= 20, f'{name}'
            input_record[name] = self._pad(input_record[name], pad_value=0)

    def _pad(self, data, pad_value):
        shape = (self.pad_short, self.pad_long, data.shape[-1])
        padded_data = np.full(shape, pad_value, np.float32)
        original_h, original_w = data.shape[:2]
        padded_data[:original_h, :original_w] = data
        return padded_data


class TransposeData(DetectionAugmentation):
    """
    Transpose data because network input require the channel dimension to be in the front.
    """

    def __init__(self, param):
        super(TransposeData, self).__init__()
        self.transpose_name_dict = param.transpose_name_dict

    def apply(self, input_record):
        for transpose_name, axis_tuple in self.transpose_name_dict.items():
            input_record[transpose_name] = input_record[transpose_name].transpose(axis_tuple)


class GenerateFPNTarget(DetectionAugmentation):
    """
    The regression targets are arranged into different stride feature map by range value.
    """

    def __init__(self, param):
        super(GenerateFPNTarget, self).__init__()
        self.interval = param.interval
        self.fpn_strides = param.fpn_strides

        self.name_list = param.name_list
        self.name_list_without_mask = param.name_list_without_mask

    def apply(self, input_record):
        if self.name_list:
            mask_dict = self.get_mask_per_stride(input_record)
            self.get_down_sample_data(input_record, self.name_list, self.fpn_strides, mask_dict)
        if self.name_list_without_mask:
            self.get_down_sample_data(
                input_record, self.name_list_without_mask, self.fpn_strides, mask_dict=None)

    def get_mask_per_stride(self, input_record):
        range_value = input_record['unnormalized_range']
        mask_dict = self.get_mask_by_range(range_value, self.interval, self.fpn_strides)
        return mask_dict

    @staticmethod
    def get_mask_by_range(range_value, interval, fpn_strides):
        assert len(range_value.shape) == 3
        assert range_value.shape[0] == 1
        assert len(interval) == len(fpn_strides)
        mask_dict = {}
        for (_, stride) in enumerate(fpn_strides):
            lower_bound, upper_bound = interval[stride]
            mask_dict[stride] = np.array(
                (lower_bound <= range_value) & (range_value < upper_bound)).astype('float32')
        return mask_dict

    @staticmethod
    def get_down_sample_data(input_record, name_list, fpn_strides, mask_dict):
        for name in name_list:
            for stride in fpn_strides:
                data = input_record[name].copy()
                if mask_dict:
                    data = data * mask_dict[stride]
                input_record[name + '_s' + str(stride)] = sample_data(
                    data, slice_begin=stride // 2, stride_width=stride)


class TransAndReshape(DetectionAugmentation):
    """
    Reshape pc_vehicle_frame and range_image_mask data for Decode3DBbox custom op.
    """

    def __init__(self, param):
        super(TransAndReshape, self).__init__()
        self.name_list = param.name_list

    def apply(self, input_record):
        for name in self.name_list:
            if 'pc' in name:
                input_record[name] = input_record[name].reshape(3, -1).transpose(1, 0)
            elif 'range' in name:
                input_record[name] = input_record[name].reshape(-1)
