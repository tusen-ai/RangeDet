from __future__ import division

import math

import mxnet as mx
from mxnext.simple import conv, relu, add, var, to_fp16, deconv

from rangedet.symbol.backbone.meta_kernel import MetaKernel

__all__ = ["DLABackboneBuilder"]


class DLABackboneBuilder(object):
    coord_sym_list = [
        var('coord_s1'), var('coord_s2'), var('coord_s4'), var('coord_s8'), var('coord_s16')]

    @classmethod
    def basicblock(cls, data, name, filter, stride, dilate, proj):
        norm = cls.param.normalizer

        relu1, is_in_name_list = cls.meta_kernel_conv(data, name, filter)
        if not is_in_name_list:
            conv1 = conv(
                data,
                name=name + "_conv1",
                filter=filter,
                kernel=3,
                stride=1,
                pad=dilate,
                dilate=dilate)
            bn1 = norm(data=conv1, name=name + "_bn1")
            relu1 = relu(data=bn1, name=name + "_relu1")

        conv2 = conv(
            relu1,
            name=name + "_conv2",
            filter=filter,
            kernel=3,
            stride=stride,
            pad=dilate,
            dilate=dilate)
        bn2 = norm(data=conv2, name=name + "_bn2")

        if proj:
            shortcut = conv(
                data,
                name=name + "_sc",
                filter=filter,
                stride=stride,
                no_bias=True)
            shortcut = norm(data=shortcut, name=name + "_sc_bn")
        else:
            shortcut = data

        eltwise = add(bn2, shortcut, name=name + "_plus")
        return relu(eltwise, name=name + "_relu")

    @classmethod
    def meta_kernel_conv(cls, data, name, filter):
        if name in cls.param.meta_kernel_units:
            conv_param = cls.param.meta_kernel_units[name]

            #  define meta kernel func
            feature_stride = conv_param['stride']
            meta_kernel = MetaKernel(
                num_batch=cls.param.batch_image,
                feat_height=cls.param.range_image_shape_hw[0],
                feat_width=cls.param.range_image_shape_hw[1] // feature_stride,
                fp16=cls.param.fp16
            )
            meta_func = getattr(
                meta_kernel,
                conv_param['meta_func_param']
            )
            print('-' * 20, conv_param['meta_func_param'], '-' * 20)

            #  call meta kernel func
            coord = cls.coord_sym_list[int(math.log2(feature_stride))]
            conv_mlp = meta_func(
                name=name,
                data=data,
                coord_data=coord,
                data_channels=conv_param['data_channels'],
                coord_channels=conv_param['coord_channels'],
                channel_list=conv_param['channel_list'],
                norm=cls.param.normalizer,
                conv1_filter=filter,
                kernel_size=conv_param['kernel_size'],
            )

            #  aggregation convolution
            norm = cls.param.normalizer
            bn_mlp = norm(data=conv_mlp, name=name + "point_wise_mlp_bn1")
            relu_mlp = relu(data=bn_mlp, name=name + "point_wise_mlp_relu1")
            conv1 = conv(relu_mlp, name=name + "aggregation_conv1", filter=filter, kernel=1)
            bn1 = norm(data=conv1, name=name + "aggregation_bn1")
            relu1 = relu(data=bn1, name=name + "aggregation_relu1")
            is_in_name_list = True
        else:
            relu1 = None
            is_in_name_list = False

        return relu1, is_in_name_list

    @classmethod
    def res_stage(cls, data, name, num_block, filter, stride, dilate):
        s, d = stride, dilate
        if isinstance(s, int):
            s = (s, s)

        data = cls.basicblock(data, "{}_unit1".format(name), filter, s, d, True)
        for i in range(2, num_block + 1):
            data = cls.basicblock(data, "{}_unit{}".format(name, i), filter, 1, d, False)
        return data

    @classmethod
    def agg_stage(cls, name, data_const, data_upsample, num_block, filter, stride, dilate,
                  deconv_kernel, deconv_stride, deconv_pad):
        norm = cls.param.normalizer
        data_upsample = deconv(
            data_upsample, name=name + "_deconv", filter=filter,
            kernel=deconv_kernel, stride=deconv_stride, pad=deconv_pad)
        data_upsample = norm(data=data_upsample, name=name + "_deconv_bn")
        data_upsample = relu(data=data_upsample, name=name + "_relu")
        eltwise = add(data_const, data_upsample, name=name + "_plus")
        stage_out = cls.res_stage(eltwise, name + "_res", num_block, filter, stride, dilate)
        return stage_out

    @classmethod
    def backbone_factory(cls, data):
        num_block = cls.param.num_block
        num_filter = cls.param.num_filter

        if data is None:
            data = var("data")
        if cls.param.fp16:
            data = to_fp16(data, "data_fp16")

        res1 = cls.res_stage(data, 'res1', num_block['res1'], num_filter['res1'], (1, 1), 1)
        res2a = cls.res_stage(res1, 'res2a', num_block['res2a'], num_filter['res2a'], (1, 2), 1)
        res2 = cls.res_stage(res2a, 'res2', num_block['res2'], num_filter['res2'], (1, 2), 1)
        res3a = cls.res_stage(res2, 'res3a', num_block['res3a'], num_filter['res3a'], (1, 2), 1)
        res3 = cls.res_stage(res3a, 'res3', num_block['res3'], num_filter['res3'], (1, 2), 1)
        agg2 = cls.agg_stage("agg2", res2, res3, num_block['agg2'], num_filter['agg2'], 1, 1,
                             deconv_kernel=(3, 8), deconv_stride=(1, 4), deconv_pad=(1, 2))
        agg1 = cls.agg_stage("agg1", res1, res2, num_block['agg1'], num_filter['agg1'], 1, 1,
                             deconv_kernel=(3, 8), deconv_stride=(1, 4), deconv_pad=(1, 2))
        agg2a = cls.agg_stage("agg2a", res2a, agg2, num_block['agg2a'], num_filter['agg2a'], 1, 1,
                              deconv_kernel=(3, 4), deconv_stride=(1, 2), deconv_pad=(1, 1))
        agg3 = cls.agg_stage("agg3", agg1, agg2a, num_block['agg3'], num_filter['agg3'], 1, 1,
                             deconv_kernel=(3, 4), deconv_stride=(1, 2), deconv_pad=(1, 1))

        if hasattr(cls.param, 'add_data_sc') and cls.param.add_data_sc:
            agg3 = mx.sym.concat(data, agg3, dim=1, name='data_concat')

        output_dict = {1: agg3, 2: agg2a, 4: agg2, 16: res3}

        if hasattr(cls.param, 'fpn_strides'):
            return [output_dict[s] for s in cls.param.fpn_strides]
        else:
            return [agg3, ]

    @classmethod
    def get_backbone(cls, data):
        return cls.backbone_factory(data)


class DLABackbone(object):
    def __init__(self, pBackbone):
        DLABackboneBuilder.param = pBackbone
        self.builder = DLABackboneBuilder()

    def get_rpn_feature(self, data):
        rpn_feature = self.builder.get_backbone(data)
        return rpn_feature
