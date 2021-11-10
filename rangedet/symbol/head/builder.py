import mxnet as mx
import mxnext as X
from mxnext.complicate import normalizer_factory
from operator_py import \
    get_sorted_foreground, \
    batch_rotated_iou
from rangedet.symbol.head.loss import vari_focal_loss


class RangeRCNN(object):
    def __init__(self, pDet):
        self.p = pDet
        self.fpn_strides = self.p.fpn_strides
        self.class_names = self.p.class_names

    def get_train_symbol(self, backbone, rpn_head):
        """ define data for training. """

        # input_data
        data = X.var('input_data')

        # target
        rpn_cls_target = [X.var('rpn_cls_target_s{}'.format(s)) for s in self.fpn_strides]
        rpn_reg_target = [X.var('rpn_reg_target_s{}'.format(s)) for s in self.fpn_strides]

        # weight for regression
        rpn_reg_weight = [X.var('rpn_reg_weight_s{}'.format(s)) for s in self.fpn_strides]
        rpn_reg_norm_weight = \
            [X.var('reg_normalize_weight_s{}'.format(s)) for s in self.fpn_strides]

        # weight for classification
        range_ignore_mask = [X.var('range_image_mask_s{}'.format(s)) for s in self.fpn_strides]

        # box and pc for iou prediction
        gt_bbox_dict = \
            {name: X.var('gt_bbox_{}_for_iou_pred'.format(name)) for name in self.class_names}
        pc_vehicle_frame = [X.var('pc_vehicle_frame_s{}'.format(s)) for s in self.fpn_strides]

        rpn_feat_list = backbone.get_rpn_feature(data)
        assert isinstance(rpn_feat_list, list)

        rpn_loss = rpn_head.get_fpn_loss(
            rpn_feat_list,
            rpn_cls_target,
            rpn_reg_target,
            rpn_reg_weight,
            rpn_reg_norm_weight,
            range_ignore_mask,
            gt_bbox_dict,
            pc_vehicle_frame
        )
        return X.group(rpn_loss)

    def get_test_symbol(self, backbone, rpn_head):
        """ define data for testing. """

        # input_data
        data = X.var('input_data')

        # pc and mask for decode box
        pc_vehicle_frame = [X.var('pc_vehicle_frame_s{}'.format(s)) for s in self.fpn_strides]
        range_image_mask = [X.var('range_image_mask_s{}'.format(s)) for s in self.fpn_strides]

        # no be used in this function
        rec_id = X.var('rec_id')
        gt_bbox = X.var('gt_bbox_imu')
        gt_class = X.var('gt_class')

        rpn_feat_list = backbone.get_rpn_feature(data)
        assert isinstance(rpn_feat_list, list)

        output_list = rpn_head.get_fpn_prediction(
            rpn_feat_list,
            pc_vehicle_frame,
            range_image_mask
        )
        return X.group([rec_id, *output_list, gt_bbox, gt_class])


class RangeRpnHead(object):
    def __init__(self, pRpn):
        self.p = pRpn
        self._prefix = ""
        self.fp16 = self.p.fp16
        self._cls_logit = None
        self._bbox_delta = None

        self.batch_size = self.p.batch_image
        self.class_names = self.p.class_names

        self.fpn_strides = self.p.fpn_strides
        self.num_classes = self.p.num_classes
        self.num_reg_delta = self.p.num_reg_delta

        self.cls_loss_weight = self.p.loss.cls_loss_weight
        self.reg_loss_weight = self.p.loss.reg_loss_weight
        self.scale_loss_shift = self.p.scale_loss_shift if self.fp16 else 1.0

    def sep_level_type(self, cls_logit_list, bbox_delta_list, concat_all_level_per_class=False):
        """
        :param cls_logit_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, num_classes, H, W // s) for s in fpn_strides]
        :param bbox_delta_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, num_classes * num_reg_dim, H, W // s) for s in fpn_strides]
        :param concat_all_level_per_class: (bool)
        :return: if concat_all_level_per_class:
                     cls_logit_dict: dict[str: Tensor]
                     tensor.shape (num_batch, 297472)
                     bbox_delta_dict: dict[str: Tensor]
                     tensor.shape (num_batch, 297472, 8)
                else:
                     cls_logit_dict: dict[str: List[Tensor]]
                     tensor.shape [(num_batch, H * W // s) for s in fpn_strides]
                     bbox_delta_dict: dict[str: List[Tensor]]
                     tensor.shape [(num_batch, H * W // s, 8) for s in fpn_strides]

        """
        cls_logit_dict = {class_name: [] for class_name in self.class_names}
        bbox_delta_dict = {class_name: [] for class_name in self.class_names}
        for level, (cls_logit, bbox_delta) in enumerate(zip(cls_logit_list, bbox_delta_list)):

            cls_score = X.reshape(
                cls_logit,
                shape=(self.batch_size, self.num_classes, -1),
                name='cls_score_reshape_lvl_{}'.format(level)
            )

            bbox_delta = X.reshape(
                bbox_delta,
                shape=(self.batch_size, self.num_classes, self.num_reg_delta, -1),
                name='bbox_delta_reshape_lvl_{}'.format(level)
            )

            for i in range(self.num_classes):
                cls_score_per_cls = mx.sym.slice_axis(cls_score, axis=1, begin=i, end=i + 1)
                cls_score_per_cls = mx.sym.squeeze(cls_score_per_cls, axis=1)
                cls_logit_dict[self.class_names[i]].append(cls_score_per_cls)

                bbox_delta_per_cls = mx.sym.slice_axis(bbox_delta, axis=1, begin=i, end=i + 1)
                bbox_delta_per_cls = mx.sym.squeeze(bbox_delta_per_cls, axis=1)
                bbox_delta_per_cls = X.transpose(bbox_delta_per_cls, (0, 2, 1))
                bbox_delta_dict[self.class_names[i]].append(bbox_delta_per_cls)

        if concat_all_level_per_class:
            cls_logit_dict = {
                class_name: mx.sym.concat(*cls_logit_dict[class_name], dim=1)
                for class_name in self.class_names
            }
            bbox_delta_dict = {
                class_name: mx.sym.concat(*bbox_delta_dict[class_name], dim=1)
                for class_name in self.class_names
            }

        return cls_logit_dict, bbox_delta_dict

    def get_iou_target(self, bbox_delta_dict, gt_bbox_dict, pc_list):
        """
        :param bbox_delta_dict: dict[str: List[Tensor]]
               tensor.shape [(num_batch, H * W // s, 8) for s in fpn_strides]
        :param gt_bbox_dict: dict[str: Tensor]
               dict{name: (B, 200, 8) for name in class_name}
        :param pc_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, H * W // s, 3) for s in fpn_strides]
        :return iou_target_list: (List[Tensor]) list of #feature levels.
                tensor.shape [(num_batch, 1, H, W // s) for s in fpn_strides]
        """
        iou_target_dict = {class_name: [] for class_name in self.class_names}
        for name in self.class_names:
            delta_list_per_class = bbox_delta_dict[name]
            gt_bbox_per_class = gt_bbox_dict[name]
            for level, bbox_delta in enumerate(delta_list_per_class):
                pc_per_level = pc_list[level]
                decoded_bbox = mx.sym.contrib.Decode3DBbox(
                    bbox_delta, pc_per_level, is_bin=False
                )
                iou_target = mx.sym.Custom(
                    proposal=decoded_bbox,
                    gt_bbox=gt_bbox_per_class,
                    op_type="batch_rotated_iou",
                    iou_type=self.p.loss.iou_type,
                    name="batch_rotated_iou_" + name + '_lvl_{}'.format(level)
                )
                iou_target = mx.sym.reshape(
                    iou_target, shape=(self.batch_size, 1, 64, -1))
                iou_target = X.block_grad(iou_target)
                iou_target_dict[name].append(iou_target)

        iou_target_list = []
        for level in range(len(pc_list)):
            iou_target_per_level = mx.sym.concat(
                *[iou_target_dict[name][level]
                  for name in self.class_names],
                dim=1
            )
            iou_target_list.append(iou_target_per_level)
        return iou_target_list

    def get_fpn_output(self, conv_feat_list):
        """
        :param conv_feat_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, C, H, W // s) for s in fpn_strides]
        :return: self._cls_logit: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, num_classes, H, W // s) for s in fpn_strides]
                self._bbox_delta: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, num_classes * num_reg_dim, H, W // s) for s in fpn_strides]
        """
        cls_conv_channel = self.p.head.cls_conv_channel
        cls_conv_layers = self.p.head.cls_conv_layers
        reg_conv_channel = self.p.head.reg_conv_channel
        reg_conv_layers = self.p.head.reg_conv_layers

        if hasattr(self.p, 'normalizer'):
            norm = self.p.normalizer
        else:
            norm = normalizer_factory(type='local', ndev=None, mom=0.9)

        self._cls_logit, self._bbox_delta = [], []
        for level, conv_feat in enumerate(conv_feat_list):
            reg_feat = cls_feat = conv_feat

            for i in range(cls_conv_layers):
                cls_feat = X.convnormrelu(
                    norm,
                    cls_feat,
                    kernel=3,
                    filter=cls_conv_channel,
                    name=self._prefix + 'rpn_cls_conv_{}_lvl_{}'.format(i, level),
                    no_bias=True,
                    init=X.gauss(0.01)
                )
            for i in range(reg_conv_layers):
                reg_feat = X.convnormrelu(
                    norm,
                    reg_feat,
                    kernel=3,
                    filter=reg_conv_channel,
                    name=self._prefix + 'rpn_reg_conv_{}_lvl_{}'.format(i, level),
                    no_bias=True,
                    init=X.gauss(0.01)
                )

            cls_logit = X.conv(
                cls_feat,
                filter=self.num_classes,
                name=self._prefix + 'rpn_cls_logit_lvl_' + str(level),
                no_bias=False,
                init=X.gauss(0.01)
            )
            bbox_delta = X.conv(
                reg_feat,
                filter=self.num_reg_delta * self.num_classes,
                name=self._prefix + 'rpn_reg_delta_lvl_' + str(level),
                no_bias=False,
                init=X.gauss(0.01)
            )

            if self.fp16:
                cls_logit = X.to_fp32(
                    cls_logit, 'rpn_cls_logit_lvl_{}_fp32'.format(level))
                bbox_delta = X.to_fp32(
                    bbox_delta, 'rpn_reg_delta_lvl_{}_fp32'.format(level))

            self._cls_logit.append(cls_logit)
            self._bbox_delta.append(bbox_delta)

        return self._cls_logit, self._bbox_delta

    def get_fpn_loss(self,
                     conv_feat_list,
                     cls_target_list,
                     reg_target_list,
                     reg_weight_list,
                     reg_norm_weight_list,
                     range_ignore_mask_list,
                     gt_bbox_dict=None,
                     pc_vehicle_frame_list=None):
        """
        :param conv_feat_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, C, H, W // s) for s in fpn_strides]
        :param cls_target_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, 1, H, W // s) for s in fpn_strides]
        :param reg_target_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, 8, H, W // s) for s in fpn_strides]
        :param reg_weight_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, 8, H, W // s) for s in fpn_strides]
        :param reg_norm_weight_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, 8, H, W // s) for s in fpn_strides]
        :param range_ignore_mask_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, 1, H, W // s) for s in fpn_strides]
        :param gt_bbox_dict: dict[str: Tensor]
               dict{name: (B, 200, 8) for name in class_name}
        :param pc_vehicle_frame_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, H * W // s, 3) for s in fpn_strides]
        :return: loss_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, 1, H, W // s) for s in fpn_strides]
              + tensor.shape [(num_batch, 1, H, W // s) for s in fpn_strides]
        """

        # get cls and reg branch feature
        cls_logit_list, reg_delta_list = self.get_fpn_output(
            conv_feat_list)

        # get iou target for cls loss
        _, bbox_delta_dict = self.sep_level_type(
            cls_logit_list,
            reg_delta_list)
        iou_target_list = self.get_iou_target(
            bbox_delta_dict,
            gt_bbox_dict,
            pc_vehicle_frame_list)

        # calculate loss
        cls_loss_list, reg_loss_list = [], []
        for level in range(len(conv_feat_list)):
            # predict features
            cls_logit = cls_logit_list[level]
            reg_delta = reg_delta_list[level]

            # target
            reg_target = reg_target_list[level]

            # weight
            reg_weight = reg_weight_list[level]
            reg_norm_weight = reg_norm_weight_list[level]

            # extra info
            range_ignore_mask = range_ignore_mask_list[level]
            stride = self.fpn_strides[level]

            # cls loss
            iou_target = iou_target_list[level]
            cls_loss = self.get_vfl_loss(
                cls_logit,
                iou_target,
                range_ignore_mask,
                stride=stride)
            cls_loss_list.append(cls_loss)

            # reg loss
            reg_loss = self.get_normalize_reg_loss(
                reg_delta,
                reg_target,
                reg_weight,
                reg_norm_weight,
                stride=stride)
            reg_loss_list.append(reg_loss)

        return cls_loss_list + reg_loss_list

    def get_vfl_loss(self, iou_logit, iou_target, mask, stride=None):
        """
        :param iou_logit: Tensor (num_batch, 1, H, W // stride)
        :param iou_target: Tensor (num_batch, 1, H, W // stride)
        :param mask: Tensor (num_batch, 1, H, W // stride)
        :param stride: (int)
        :return: vfl_loss: Tensor (num_batch, 1, H, W // stride)
        """
        vfl_loss = vari_focal_loss(
            iou_logit,
            iou_target,
            1.0,
            alpha=self.p.loss.alpha,
            gamma=self.p.loss.gamma)

        range_nonignore_mask = X.block_grad(mask)
        norm = mx.sym.sum(range_nonignore_mask) + 1
        range_nonignore_mask = X.reshape(
            range_nonignore_mask,
            shape=(self.batch_size, 1, 64, -1))

        vfl_loss = mx.sym.broadcast_mul(vfl_loss, range_nonignore_mask)
        vfl_loss = mx.sym.broadcast_div(vfl_loss, norm)

        vfl_loss = X.loss(
            vfl_loss,
            grad_scale=self.scale_loss_shift * self.cls_loss_weight,
            name='rpn_cls_loss_s{}'.format(stride)
            if stride is not None else 'rpn_cls_loss')
        return vfl_loss

    def get_normalize_reg_loss(self,
                               reg_delta,
                               reg_target,
                               reg_weight,
                               reg_norm_weight,
                               stride=None):
        """
        :param reg_delta: Tensor (num_batch, 8, H, W // stride)
        :param reg_target: Tensor (num_batch, 8, H, W // stride)
        :param reg_weight: Tensor (num_batch, 8, H, W // stride)
        :param reg_norm_weight: Tensor (num_batch, 8, H, W // stride)
        :param stride: (int)
        :return: reg_loss: Tensor (num_batch, 8, H, W // stride)
        """
        smooth_l1_scalar = self.p.loss.smooth_l1_scalar \
            if hasattr(self.p.loss, 'smooth_l1_scalar') else 1.0
        if hasattr(self.p.loss, 'l1') and self.p.loss.l1:
            reg_loss = mx.sym.abs(
                (reg_delta - reg_target),
                name=self._prefix + 'rpn_reg_l1'
            )
        else:
            reg_loss = X.smooth_l1(
                (reg_delta - reg_target),
                scalar=smooth_l1_scalar,
                name=self._prefix + 'rpn_reg_smoothl1'
            )

        reg_weight = X.block_grad(reg_weight)
        reg_norm_weight = X.block_grad(reg_norm_weight)
        norm = mx.sym.sum(reg_norm_weight) + 1

        reg_loss = mx.sym.broadcast_div(
            lhs=reg_loss * reg_weight * reg_norm_weight,
            rhs=norm
        ) * self.reg_loss_weight
        reg_loss = X.loss(
            reg_loss,
            grad_scale=self.scale_loss_shift,
            name='rpn_reg_loss_s{}'.format(stride)
            if stride is not None else 'rpn_reg_loss')
        return reg_loss

    def get_fpn_prediction(self,
                           conv_feat_list,
                           pc_vehicle_frame_list,
                           range_image_mask_list):
        """
        :param conv_feat_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, C, H, W // s) for s in fpn_strides]
        :param pc_vehicle_frame_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, H * W // s, 3) for s in fpn_strides]
        :param range_image_mask_list: (List[Tensor]) list of #feature levels.
               tensor.shape [(num_batch, H * W // s) for s in fpn_strides]
        :return: if nms:
                     fg_cls_score: (List[Tensor])
                     tensor.shape [(rpn_pre_nms_top_n, )]
                     final_proposal: (List[Tensor])
                     tensor.shape [(rpn_post_nms_top_n, )]
                     keep_inds: (List[Tensor])
                     tensor.shape [(rpn_post_nms_top_n, )]
                 elif wnms:
                     fg_cls_score: (List[Tensor])
                     tensor.shape [(rpn_pre_nms_top_n, )]
                     decoded_bbox: (List[Tensor])
                     tensor.shape [(rpn_post_nms_top_n, 10)]
                     keep_inds: (List[Tensor])
                     tensor.shape [empty]

        """
        cls_logit_list, bbox_delta_list = self.get_fpn_output(
            conv_feat_list)

        cls_logit_dict, bbox_delta_dict = self.sep_level_type(
            cls_logit_list,
            bbox_delta_list,
            concat_all_level_per_class=True)

        cls_score_dict = {
            k: X.sigmoid(v, name=k + '_sigmoid')
            for k, v in cls_logit_dict.items()}

        all_pc = mx.sym.concat(*pc_vehicle_frame_list, dim=1)
        all_mask = mx.sym.concat(*range_image_mask_list, dim=1)

        output_list = []
        for class_name in self.class_names:
            fg_cls_score, final_proposal, keep_inds = \
                self.get_prediction_of_one_type(
                    cls_score_dict[class_name],
                    bbox_delta_dict[class_name],
                    all_pc,
                    all_mask,
                    self.p.all_proposal.nms_thr[class_name],
                    self.p.all_proposal.rpn_pre_nms_top_n[class_name],
                    self.p.all_proposal.rpn_post_nms_top_n[class_name]
                )
            output_list += [fg_cls_score, final_proposal, keep_inds]
        return output_list

    def get_prediction_of_one_type(self,
                                   cls_score,
                                   bbox_delta,
                                   pc_vehicle_frame,
                                   mask,
                                   nms_thr,
                                   pre_nms_top_n,
                                   post_nms_top_n):
        """
        :param cls_score: Tensor (num_batch, 297472)
        :param bbox_delta: Tensor (num_batch, 297472, num_reg_dim)
        :param pc_vehicle_frame: Tensor (num_batch, 297472, 3)
        :param mask: Tensor (num_batch, 297472)
        :param nms_thr: (float)
        :param pre_nms_top_n: (int)
        :param post_nms_top_n: (int)
        :return: if nms:
                     fg_cls_score: (List[Tensor])
                     tensor.shape [(rpn_pre_nms_top_n, )]
                     final_proposal: (List[Tensor])
                     tensor.shape [(rpn_post_nms_top_n, )]
                     keep_inds: (List[Tensor])
                     tensor.shape [(rpn_post_nms_top_n, )]
                 elif wnms:
                     fg_cls_score: (List[Tensor])
                     tensor.shape [(rpn_pre_nms_top_n, )]
                     decoded_bbox: (List[Tensor])
                     tensor.shape [(rpn_post_nms_top_n, 10)]
                     keep_inds: (List[Tensor])
                     tensor.shape [empty]
        """
        fg_cls_score, fg_bbox_delta, fg_pc_vehicle_frame = \
            mx.sym.Custom(
                cls_score=cls_score,
                bbox_delta=bbox_delta,
                pc=pc_vehicle_frame,
                mask=mask,
                op_type="get_sorted_foreground",
                num_fgs=pre_nms_top_n,
                name=self._prefix + "get_foreground"
            )
        decoded_bbox = mx.sym.contrib.Decode3DBbox(
            fg_bbox_delta,
            fg_pc_vehicle_frame,
            is_bin=False)

        if hasattr(self.p, 'wnms') and self.p.wnms:
            return fg_cls_score, decoded_bbox, mx.sym.zeros(shape=(1,))
        else:
            keep_inds, final_proposal = mx.sym.contrib.NMS3D(
                decoded_bbox,
                nms_thr,
                post_nms_top_n)
            return fg_cls_score, final_proposal, keep_inds
