/*!
 * Copyright (c) 2017 by Contributors
 * \file rotated_iou.cu
 * \brief rotated IOU
 * \author Feng Wang
*/
#include <stdio.h>
#include "./rotated_iou-inl.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_RotatedIOU)
.set_attr<FCompute>("FCompute<gpu>", RotatedIOUForward<gpu>);

}
}
