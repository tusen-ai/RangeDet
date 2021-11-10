from __future__ import division

import mxnet as mx
from mxnext.simple import conv, to_fp16, relu


class MetaKernel(object):
    def __init__(self, num_batch, feat_height, feat_width, fp16, num_frame=1):
        self.num_batch = num_batch
        self.H = feat_height
        self.W = feat_width
        self.fp16 = fp16
        self.num_frame = num_frame

    @staticmethod
    def sampler_im2col(data, name, kernel=1, stride=1, pad=None, dilate=1):
        """ please refer to mx.symbol.im2col """
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilate, int):
            dilate = (dilate, dilate)
        if pad is None:
            assert kernel[0] % 2 == 1, "Specify pad for an even kernel size for {}".format(name)
            pad = ((kernel[0] - 1) * dilate[0] + 1) // 2
        if isinstance(pad, int):
            pad = (pad, pad)

        output = mx.symbol.im2col(
            name=name + "sampler",
            data=data,
            kernel=kernel,
            stride=stride,
            dilate=dilate,
            pad=pad
        )
        return output

    def sample_data(self, name, data, kernel_size):
        """
        data sample
        :param name: str
        :param data: num_batch, num_channel_in, H, W
        :param kernel_size: int default=3
        :return: sample_output: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        """
        sample_output = self.sampler_im2col(
            data=data,
            name=name + "data_",
            kernel=kernel_size,
            stride=1,
            pad=1,
            dilate=1
        )
        return sample_output

    def sample_coord(self, name, coord, kernel_size):
        """
        coord sample
        :param name: str
        :param coord: num_batch, num_channel_in, H, W
        :param kernel_size: int default=3
        :return: coord_sample_data: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        """
        coord_sample_data = self.sampler_im2col(
            data=coord,
            name=name + "coord_",
            kernel=kernel_size,
            stride=1,
            pad=1,
            dilate=1
        )
        return coord_sample_data

    def relative_coord(self, sample_coord, center_coord, num_channel_in, kernel_size):
        """
        :param sample_coord: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        :param center_coord: num_batch, num_channel_in, H, W
        :param num_channel_in: int
        :param kernel_size: int
        :return: rel_coord: num_batch, num_channel_in, kernel_size * kernel_size, H, W
        """
        sample_reshape = mx.sym.reshape(
            sample_coord,
            shape=(
                self.num_batch,
                num_channel_in,
                kernel_size * kernel_size,
                self.H,
                self.W
            )
        )
        center_coord_expand = mx.sym.expand_dims(
            center_coord,
            axis=2
        )
        rel_coord = mx.sym.broadcast_minus(
            sample_reshape,
            center_coord_expand,
            name="relative_dis"
        )
        return rel_coord

    def mlp(self,
            data,
            name,
            in_channels,
            norm,
            channel_list=None,
            b_mul=1,
            no_bias=True,
            use_norm=False):
        """
        :param data: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        :param name: str
        :param in_channels: int
        :param norm: normalizer
        :param channel_list: List[int]
        :param b_mul: int default=1
        :param no_bias: bool default=True
        :param use_norm: bool default=False
        :return: mlp_output_reshape: num_batch, out_channels, kernel_size * kernel_size, H, W
        """
        assert isinstance(channel_list, list)
        x = mx.sym.reshape(
            data,
            shape=(
                self.num_batch * b_mul,
                in_channels,
                -1,
                self.W
            )
        )
        for i, out_channel in enumerate(channel_list):
            x = conv(
                x,
                name=name + "{}_mlp{}".format(self.W, i),
                filter=out_channel,
                kernel=1,
                stride=1,
                pad=0,
                dilate=1,
                no_bias=no_bias
            )
            if i != len(channel_list) - 1:
                if use_norm:
                    x = norm(
                        x,
                        name=name + "{}_mlp_bn{}".format(self.W, i))
                x = relu(
                    x,
                    name + "{}_mlp_relu{}".format(self.W, i))
        mlp_output_reshape = mx.sym.reshape(
            x,
            shape=(
                self.num_batch * b_mul,
                channel_list[-1],
                -1,
                self.H,
                self.W
            )
        )
        return mlp_output_reshape

    def meta_baseline_bias(self,
                           name,
                           data,
                           coord_data,
                           data_channels,
                           coord_channels,
                           channel_list,
                           norm,
                           conv1_filter,
                           kernel_size=3,
                           **kwargs):
        """
        # Without data mlp;
        # MLP: fc + norm + relu + fc;
        # Using normalized coordinates
        :param name: str
        :param data: num_batch, num_channel_in, H, W
        :param coord_data: num_batch, 3, H, W
        :param data_channels: num_channel_in
        :param coord_channels: 3
        :param channel_list: List[int]
        :param norm: normalizer
        :param conv1_filter: int
        :param kernel_size: int default=3
        :param kwargs:
        :return: conv1: num_batch, conv1_filter, H, W
        """
        if self.fp16:
            coord_data = to_fp16(
                coord_data,
                name + 'coord_data_fp16')

        name = name + '_'

        coord_sample_data = self.sample_coord(
            name,
            coord_data,
            kernel_size)
        rel_coord = self.relative_coord(
            coord_sample_data,
            coord_data,
            coord_channels,
            kernel_size)
        weights = self.mlp(
            rel_coord,
            name,
            in_channels=coord_channels,
            channel_list=channel_list,
            norm=norm,
            no_bias=False)

        data_sample = self.sample_data(
            name,
            data,
            kernel_size)
        data_sample_reshape = mx.sym.reshape(
            data=data_sample,
            shape=(
                self.num_batch,
                data_channels,
                kernel_size * kernel_size,
                self.H,
                self.W)
        )

        output = data_sample_reshape * weights
        output_reshape = mx.sym.reshape(
            output,
            shape=(
                self.num_batch,
                -1,
                self.H,
                self.W)
        )
        return output_reshape
