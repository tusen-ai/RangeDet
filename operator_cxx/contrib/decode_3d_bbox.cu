/*!
 * \file 
 * \brief 
 * \author Lve Fan
*/
#include <stdio.h>
#include "./decode_3d_bbox-inl.h"
#include "cuda_utils.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_Decode3DBbox)
.set_attr<FCompute>("FCompute<gpu>", Decode3DBboxForward<gpu>);

}
}
