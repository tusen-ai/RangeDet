import mxnet as mx
import mxnet.numpy as np
from mxnet.ndarray.contrib import RotatedIOU


class BatchRotatedIOU(mx.operator.CustomOp):
    def __init__(self, iou_type):
        super(BatchRotatedIOU, self).__init__()
        self.iou_type = iou_type

    def forward(self, is_train, req, in_data, out_data, aux):
        # input
        proposal = in_data[0]  # (3, 169984, 10)
        gt_bbox = in_data[1]  # (3, 200, 8)

        if self.iou_type == '3d':
            proposal = self.to_box_type_7(proposal.copy())

        iou_map = np.zeros((proposal.shape[0], proposal.shape[1]), dtype=gt_bbox.dtype, ctx=gt_bbox.context)
        for pred_bbox_per_batch, gt_bbox_per_batch, iou_map_per_batch in zip(proposal, gt_bbox, iou_map):
            iou_3d = self.get_iou(pred_bbox_per_batch, gt_bbox_per_batch)
            iou_map_per_batch[...] = iou_3d

        for ind, val in enumerate([iou_map, ]):
            self.assign(out_data[ind], req[ind], val.as_nd_ndarray())

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for ind in range(2):
            self.assign(in_grad[ind], req[ind], 0)

    def get_iou(self, roi_batch, gt_batch):
        if self.iou_type == 'bev':
            iou_mat = RotatedIOU(roi_batch[:, :8], gt_batch).as_np_ndarray()  # [num_roi, num_gt]
        elif self.iou_type == '3d':
            roi_batch[:, -1] = -1 * roi_batch[:, -1]
            gt_batch[:, -1] = -1 * gt_batch[:, -1]
            iou_mat = RotatedIOU(roi_batch.as_nd_ndarray(), gt_batch).as_np_ndarray()  # [num_roi, num_gt]
        else:
            raise Exception("no supported type")

        # iou_mat = iou_mat.as_np_ndarray()
        # iou_mat = np.minimum(1, np.maximum(0, 2 * iou_mat - 0.5))
        iou_mat[np.isnan(iou_mat)] = 0
        iou_mat[np.isinf(iou_mat)] = 0
        iou_mat[iou_mat > 1.0] = 0
        iou_mat[iou_mat < 0] = 0
        iou_mat = iou_mat.max(axis=1)

        return iou_mat

    @staticmethod
    def to_box_type_7(proposal):
        """
        [x1, y1, x2, y2, x3, y3, x4, y4, z0, z1] change to [cen_x, cen_y, cen_z, l, w, h, yaw]
        :param proposal:
        :return:
        """
        proposal_4pts = proposal[:, :, :8].reshape(proposal.shape[0], -1, 4, 2)
        center_xy = proposal_4pts.mean(axis=2)  # [b, n, 2]
        center_z = proposal[:, :, -2:].mean(axis=2, keepdims=True)
        length = ((proposal_4pts[:, :, 0, :] - proposal_4pts[:, :, 1, :]) ** 2).sum(axis=2, keepdims=True) ** 0.5
        width = ((proposal_4pts[:, :, 1, :] - proposal_4pts[:, :, 2, :]) ** 2).sum(axis=2, keepdims=True) ** 0.5
        height = proposal[:, :, -1:] - proposal[:, :, -2:-1]
        yaw = np.arctan2(
            proposal_4pts[:, :, 0, 1] - proposal_4pts[:, :, 1, 1],
            proposal_4pts[:, :, 0, 0] - proposal_4pts[:, :, 1, 0])
        proposal_type7 = np.concatenate([center_xy, center_z, length, width, height, yaw[:, :, None]], axis=2)
        return proposal_type7


@mx.operator.register('batch_rotated_iou')
class BatchRotatedIOUProp(mx.operator.CustomOpProp):
    def __init__(self, iou_type):
        super(BatchRotatedIOUProp, self).__init__(need_top_grad=False)
        self.iou_type = iou_type

    def list_arguments(self):
        return ['proposal', 'gt_bbox']

    def list_outputs(self):
        return ['iou_map']

    def infer_shape(self, in_shape):
        proposal_shape = in_shape[0]  # [b, 169984, 10]
        batch_size = proposal_shape[0]
        sample_num = proposal_shape[1]
        proposal_dim = proposal_shape[2]
        assert proposal_dim == 10

        gt_bbox_shape = in_shape[1]
        assert len(gt_bbox_shape) == 3  # [b, 200, 8]
        max_num_box = gt_bbox_shape[1]
        assert max_num_box == 200
        gt_bbox_dim = gt_bbox_shape[2]
        if self.iou_type == 'bev':
            assert gt_bbox_dim == 8
        elif self.iou_type == '3d':
            assert gt_bbox_dim == 7
        else:
            raise ValueError('Unknown iou type!')

        out_shape = (batch_size, sample_num)

        return in_shape, [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BatchRotatedIOU(self.iou_type)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
