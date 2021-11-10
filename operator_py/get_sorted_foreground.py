import mxnet as mx
import mxnet.ndarray as nd


class GetSortedFGOperator(mx.operator.CustomOp):
    def __init__(self, num_fgs):
        super(GetSortedFGOperator, self).__init__()
        self._num_fgs = num_fgs
        # print('Warning: please remove cls_score noise when test real data')

    def forward(self, is_train, req, in_data, out_data, aux):
        ctx = in_data[0].context

        cls_score = in_data[0]  # (1, 297472)
        bbox_delta = in_data[1]  # (1, 297472, 8)
        pc = in_data[2]  # (1, 297472, 3)
        mask = in_data[3]  # (1, 297472)

        # mask the point need not to be calculated
        cls_score = cls_score * mask

        # take topk points
        topk_scores, topk_inds = nd.topk(cls_score, axis=1, k=self._num_fgs, ret_typ='both')

        # create empty output array
        batch_size = cls_score.shape[0]
        all_fg_bbox_delta = mx.nd.zeros(shape=(batch_size, self._num_fgs, 8), ctx=ctx)
        all_fg_pc = mx.nd.zeros(shape=(batch_size, self._num_fgs, 3), ctx=ctx)
        all_fg_scores = mx.nd.zeros(shape=(batch_size, self._num_fgs), ctx=ctx)

        for i, (bbox_delta_per_batch, pc_per_batch, topk_inds_per_batch, topk_scores_per_batch) \
                in enumerate(zip(bbox_delta, pc, topk_inds, topk_scores)):
            sort_inds_per_batch = nd.argsort(topk_scores_per_batch, axis=0, is_ascend=False)
            sort_inds = topk_inds_per_batch[sort_inds_per_batch]
            all_fg_bbox_delta[i] = bbox_delta_per_batch[sort_inds]
            all_fg_pc[i] = pc_per_batch[sort_inds]
            all_fg_scores[i] = topk_scores_per_batch[sort_inds_per_batch]

        for ind, val in enumerate([all_fg_scores, all_fg_bbox_delta, all_fg_pc]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for ind in range(4):
            self.assign(in_grad[ind], req[ind], 0)


@mx.operator.register('get_sorted_foreground')
class GetSortedFGProp(mx.operator.CustomOpProp):
    def __init__(self, num_fgs):
        super(GetSortedFGProp, self).__init__(need_top_grad=False)
        self._num_fgs = int(num_fgs)

    def list_arguments(self):
        return ['cls_score', 'bbox_delta', 'pc', 'mask']

    def list_outputs(self):
        return ['sorted_fg_score', 'sorted_fg_bbox_delta', 'sorted_fg_pc']

    def infer_shape(self, in_shape):
        cls_score_shape = in_shape[0]
        bbox_delta_shape = in_shape[1]
        pc_shape = in_shape[2]
        mask_shape = in_shape[3]

        assert pc_shape[1] >= self._num_fgs
        assert len(cls_score_shape) == 2
        assert len(bbox_delta_shape) == 3
        assert len(pc_shape) == 3

        out_shape1 = list(cls_score_shape)
        out_shape2 = list(bbox_delta_shape)
        out_shape3 = list(pc_shape)

        out_shape1[1] = self._num_fgs
        out_shape2[1] = self._num_fgs
        out_shape3[1] = self._num_fgs

        return [cls_score_shape, bbox_delta_shape, pc_shape, mask_shape], \
               [tuple(out_shape1), tuple(out_shape2), tuple(out_shape3)]

    def create_operator(self, ctx, shapes, dtypes):
        return GetSortedFGOperator(self._num_fgs)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
