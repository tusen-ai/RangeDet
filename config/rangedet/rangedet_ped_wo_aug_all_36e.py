import time
from datetime import datetime

import numpy as np


def convertDateToTimestamp(str_date):
    dt = datetime.strptime(str_date, '%Y-%m-%d-%H-%M-%S')
    return time.mktime(dt.timetuple())


from mxnext.complicate import normalizer_factory
import rangedet.core.detection_metric as metric

from rangedet.core.input import \
    LoadRecord, LoadGTInfo, FilterGTClass, \
    SepAndClipData, NormData, PadData, \
    ProcessMissValue, Bbox3dAssigner, GenerateTarget, \
    CombineData, TransposeData, GetUnnormalizedRange, \
    GenerateFPNTarget, GetCoordinates, GetFixedLengthGTBbox, TransAndReshape

from rangedet.symbol.head.builder import RangeRCNN as Detector
from rangedet.symbol.head.builder import RangeRpnHead as RpnHead
from rangedet.symbol.backbone.dla_backbone import DLABackbone as Backbone


# label_set = [1, 2, 4]
# class_names = ['veh', 'ped', 'cyc']

def get_config(is_train):
    class General:
        batch_image = 2 if is_train else 1
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        fp16 = True
        scale_loss_shift = 128

        feat_size = (64, 2650)
        #label_set = [1]
        label_set = [2]
        num_classes = len(label_set)
        #class_names = ('veh',)
        class_names = ('ped',)
        pad_field = (64, 2656)

    class KvstoreParam:
        sync_flag = True
        kvstore = "local"  # "device"
        use_horovod = True
        if is_train and use_horovod:
            gpus = [0]
        else:
            gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_image = General.batch_image
        fp16 = General.fp16

    class NormalizeParam:
        normalizer = normalizer_factory(type="localbn", ndev=len(KvstoreParam.gpus))

    class DatasetParam:
        data_root = 'path/to/datasets/waymo-range'
        if is_train:
            image_set = ('training',)
        else:
            image_set = 'validation'
        sampling_rate = 1
        minival = True
        #filter_class = ['TYPE_VEHICLE']
        filter_class = ['TYPE_PEDESTRIAN']

    class FpnParam:
        fpn_strides = (1, 2, 4)
        strategy = 'range'
        interval = {1: (30, 100), 2: (15, 30), 4: (0, 15)}
        if is_train:
            name_list = [
                'rpn_cls_target',
                'rpn_reg_target',
                'rpn_reg_weight',
                'reg_normalize_weight']
            name_list_without_mask = [
                'pc_vehicle_frame',
                'range_image_mask',
                'coord']
        else:
            name_list = [
                'range_image_mask']
            name_list_without_mask = [
                'pc_vehicle_frame',
                'coord']

    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        fpn_strides = FpnParam.fpn_strides
        batch_image = General.batch_image
        range_image_shape_hw = General.pad_field
        meta_kernel_units = {
            'res1_unit2': dict(
                stride=1,
                meta_func_param='meta_baseline_bias',
                data_channels=64,
                coord_channels=3,
                channel_list=[32, 64],
                kernel_size=3
            )}
        num_block = {'res1': 2, 'res2a': 3, 'res2': 3, 'res3a': 5, 'res3': 5,
                     'agg1': 2, 'agg2': 2, 'agg2a': 1, 'agg3': 2, }
        num_filter = {'res1': 64, 'res2a': 64, 'res2': 128, 'res3a': 128, 'res3': 128,
                      'agg1': 64, 'agg2': 128, 'agg2a': 64, 'agg3': 64, }
        add_data_sc = True

    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image
        feat_size = General.feat_size
        scale_loss_shift = General.scale_loss_shift
        class_names = General.class_names
        num_classes = General.num_classes
        fpn_strides = FpnParam.fpn_strides
        num_reg_delta = 8
        wnms = True

        class loss:
            alpha = 1
            gamma = 2
            reg_loss_weight = 8.0
            cls_loss_weight = 10.0
            iou_type = 'bev'
            # l1 = True
            smooth_l1_scalar = 3

        class head:
            cls_conv_layers = 4
            cls_conv_channel = 128
            reg_conv_layers = 4
            reg_conv_channel = 128

        class all_proposal:
            rpn_pre_nms_top_n = {'veh': 50000, 'ped': 5000, 'cyc': 5000}
            rpn_post_nms_top_n = {'veh': 200, 'ped': 200, 'cyc': 100}
            nms_thr = {'veh': 0.2, 'ped': 0.2, 'cyc': 0.2}

    class RoiParam:
        pass

    class BboxParam:
        pass

    class DetParam:
        fpn_strides = FpnParam.fpn_strides
        class_names = General.class_names

    # data processing
    backbone = Backbone(BackboneParam)
    rpn_head = RpnHead(RpnParam)
    detector = Detector(DetParam)
    if is_train:
        train_sym = detector.get_train_symbol(backbone, rpn_head)
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(backbone, rpn_head)

    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = True
        random = False
        memonger = False
        memonger_until = ""

        class pretrain:
            prefix = ""
            epoch = 36
            fixed_param = []

    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image * 5
            momentum = 0.9
            wd = 0.00001
            clip_gradient = 35
            lr_mode = 'cosine'

        class schedule:
            begin_epoch = 0
            end_epoch = 36
            if end_epoch == 36:
                lr_step = [24, 30]
            elif end_epoch == 18:
                lr_step = [12, 15]

        class warmup:
            type = "gradual"
            lr = 0.0
            epoch = 2

    class TestParam:
        min_score = {'veh': 0.5, 'ped': 0.4, 'cyc': 0.3}
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x
        class_names = General.class_names

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            wnms = True if hasattr(RpnParam, 'wnms') and RpnParam.wnms else False
            thr_lo = 0.1
            thr_hi = 0.5
            is_3d_iou = False

    class GenerateTargetParam:
        feat_size = General.feat_size
        reg_weight = [3, 1, 1, 1, 1, 1, 1, 1]
        label_set = General.label_set
        num_classes = General.num_classes

    class AugParam:
        rotation_interval = (-3.1415 / 4, 3.1415 / 4)
        scale_interval = (0.8, 1.2)
        scale_func = 'scale_bg'
        pasting_class = ['ped']
        min_instance_num = {'ped': 5}
        pasting_num = {'ped': {'short': 10, 'mid': 10, 'long': 50}, }
        min_num_points = 2
        objects_root = DatasetParam.data_root + '/objects_full/objects_pool.pkl'
        no_range_constrain = False
        azimuth_occ_thr = 1
        class_types = [{'veh': 1, 'ped': 2, 'cyc': 4}[name] for name in General.class_names]
        rotate_ins = True
        flip_ins = True
        ins_rotation_interval = (-3.1415 / 4, 3.1415 / 4)
        flip_prob = 0.2
        azimuth_delta = 2 * 3.1415 / 2650

    class LabelMapParam:
        mapping = {1: 1, 2: 2, 3: 3, 4: 4, 0: 5}
        test_mapping = {0: 1, 1: 2, 2: 4}

    class ClipDataParam:
        clip_data_dict = {
            'range_value': (0, 80),
            'intensity': (0, 1),
            'elongation': (0, 1),
            'pc_vehicle_frame_x': (-80, 80),
            'pc_vehicle_frame_y': (-80, 80),
            'pc_vehicle_frame_z': (-5, 10),
            'inclination': (-0.5, 0.1),
            'azimuth': (-np.pi * 2, np.pi / 2)
        }

    class NormDataParam:
        norm_data_dict = {
            'range_value': (20.0, 1500.0),
            'intensity': (0.1, 0.01),
            'elongation': (7.2558375e-02, 2.6764875e-02),
            'pc_vehicle_frame_x': (1.5672500e+00, 3.0740625e+02),
            'pc_vehicle_frame_y': (9.8824875e-01, 2.1913250e+02),
            'pc_vehicle_frame_z': (1.4, 1.0),
            'inclination': (-8.8427375e-02, 9.9001750e-03),
            'azimuth': (-7.8061250e-03, 2.5494125e+00),
        }

    class CombineDataParam:
        combine_name_dict = {
            'input_data':
                [
                    'range_value',
                    'intensity',
                    'elongation',
                    'pc_vehicle_frame_x',
                    'pc_vehicle_frame_y',
                    'pc_vehicle_frame_z',
                    'inclination',
                    'azimuth'
                ]
        }

    class GetFixedLengthGTBboxParam:
        #class_type = ['TYPE_VEHICLE']
        class_type = ['TYPE_PEDESTRIAN']
        fixed_length = 200

    class Bbox3dAssignerParam:
        feat_size = General.feat_size

    class PadDataParam:
        pad_short = General.pad_field[0]
        pad_long = General.pad_field[1]
        if is_train:
            pad_name_list = [
                'input_data',
                'rpn_cls_target',
                'rpn_reg_target',
                'rpn_reg_weight',
                'reg_normalize_weight',
                'range_image_mask',
                'pc_vehicle_frame',
                'unnormalized_range',
                'coord'
            ]
        else:
            pad_name_list = [
                'input_data',
                'range_image_mask',
                'pc_vehicle_frame',
                'unnormalized_range',
                'coord'
            ]

    class TransposeDataParam:
        if is_train:
            transpose_name_dict = {'input_data': (2, 0, 1)}
            transpose_name_dict.update({'rpn_cls_target': (2, 0, 1)})
            transpose_name_dict.update({'rpn_reg_target': (2, 0, 1)})
            transpose_name_dict.update({'rpn_reg_weight': (2, 0, 1)})
            transpose_name_dict.update({'reg_normalize_weight': (2, 0, 1)})

            transpose_name_dict.update({'range_image_mask': (2, 0, 1)})
            transpose_name_dict.update({'pc_vehicle_frame': (2, 0, 1)})
            transpose_name_dict.update({'unnormalized_range': (2, 0, 1)})
            transpose_name_dict.update({'coord': (2, 0, 1)})
        else:
            transpose_name_dict = {'input_data': (2, 0, 1)}
            transpose_name_dict.update({'range_image_mask': (2, 0, 1)})
            transpose_name_dict.update({'pc_vehicle_frame': (2, 0, 1)})
            transpose_name_dict.update({'unnormalized_range': (2, 0, 1)})
            transpose_name_dict.update({'coord': (2, 0, 1)})

    class TransAndReshapeParam:
        if is_train:
            name_list = ['pc_vehicle_frame_s1', 'pc_vehicle_frame_s2', 'pc_vehicle_frame_s4']
        else:
            name_list = ['pc_vehicle_frame_s1', 'pc_vehicle_frame_s2', 'pc_vehicle_frame_s4',
                         'range_image_mask_s1', 'range_image_mask_s2', 'range_image_mask_s4']

    if len(BackboneParam.meta_kernel_units) == 0:
        coord_name = []
    else:
        coord_name = ['coord_s1']

    if is_train:
        transform = [
            LoadRecord(),
            LoadGTInfo(),
            FilterGTClass(General.label_set),
            # RandomWorldFlip(AugParam),
            # RandomRotation(AugParam),
            ProcessMissValue(),
            SepAndClipData(ClipDataParam),
            GetUnnormalizedRange(),
            NormData(NormDataParam),
            GetCoordinates(),
            CombineData(CombineDataParam),
            GetFixedLengthGTBbox(GetFixedLengthGTBboxParam),
            Bbox3dAssigner(Bbox3dAssignerParam),
            GenerateTarget(GenerateTargetParam),
            PadData(PadDataParam),
            TransposeData(TransposeDataParam),
            GenerateFPNTarget(FpnParam),
            TransAndReshape(TransAndReshapeParam),
        ]
        data_name = ["input_data"]

        reg_target_stride = ["rpn_reg_target_s{}".format(s) for s in RpnParam.fpn_strides]
        reg_weight_stride = ["rpn_reg_weight_s{}".format(s) for s in RpnParam.fpn_strides]
        reg_normalize_stride = ["reg_normalize_weight_s{}".format(s) for s in RpnParam.fpn_strides]
        mask_stride = ["range_image_mask_s{}".format(s) for s in RpnParam.fpn_strides]
        gt_bbox_per_class = [f'gt_bbox_{General.class_names[0]}_for_iou_pred']
        pc_stride = ["pc_vehicle_frame_s{}".format(s) for s in RpnParam.fpn_strides]

        label_name = reg_target_stride + reg_weight_stride + \
                     mask_stride + reg_normalize_stride + \
                     pc_stride + gt_bbox_per_class + coord_name
    else:
        transform = [
            LoadRecord(),
            LoadGTInfo(),
            FilterGTClass(General.label_set),
            # RandomWorldFlip(AugParam),
            # RandomRotation(AugParam),
            ProcessMissValue(),
            SepAndClipData(ClipDataParam),
            GetUnnormalizedRange(),
            NormData(NormDataParam),
            GetCoordinates(),
            CombineData(CombineDataParam),
            # GetFixedLengthGTBbox(),
            # Bbox3dAssigner(Bbox3dAssignerParam),
            # GenerateTarget(LocationTargetParam),
            PadData(PadDataParam),
            TransposeData(TransposeDataParam),
            GenerateFPNTarget(FpnParam),
            TransAndReshape(TransAndReshapeParam),
        ]
        data_name = ["input_data", "gt_bbox_imu", "gt_class", "rec_id"]
        pc_stride = ["pc_vehicle_frame_s{}".format(s) for s in RpnParam.fpn_strides]
        mask_stride = ["range_image_mask_s{}".format(s) for s in RpnParam.fpn_strides]

        data_name = data_name + pc_stride + mask_stride + coord_name
        label_name = []

    reg_metric_list = [
        metric.ScalarLoss(
            "L1-s{}".format(s),
            ["rpn_reg_loss_s{}_output".format(s)],
            []) for s in FpnParam.fpn_strides
    ]
    cls_metric_list = [
        metric.ScalarLoss(
            "cls-s{}".format(s),
            ["rpn_cls_loss_s{}_output".format(s)],
            []) for s in FpnParam.fpn_strides
    ]
    metric_list = reg_metric_list + cls_metric_list

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list, LabelMapParam
