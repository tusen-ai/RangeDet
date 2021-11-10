/*!
 * \file 
 * \brief 
 * \author 
*/
#ifndef MXNET_OPERATOR_CONTRIB_NMS_3D_INL_H_
#define MXNET_OPERATOR_CONTRIB_NMS_3D_INL_H_

#include <vector>
#include <utility>
#include <mxnet/operator_util.h>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"

namespace mxnet {
  typedef std::vector<mxnet::TShape> ShapeVector;
namespace op {

struct NMS3DParam : public dmlc::Parameter<NMS3DParam> {
  float iou_thres;
  int max_keep;
  bool normal_iou;
  DMLC_DECLARE_PARAMETER(NMS3DParam) {
    DMLC_DECLARE_FIELD(iou_thres)
      .describe("IOU threshold.");
    DMLC_DECLARE_FIELD(max_keep)
      .describe("Max number of box to keep.");
    DMLC_DECLARE_FIELD(normal_iou)
      .set_default(false)
      .describe("Max number of box to keep.");
  }
};

template <typename xpu>
void NMS3DForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                         const std::vector<TBlob>& in_data,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& out_data);

}
}

#endif  // MXNET_OPERATOR_CONTRIB_ROTATED_NMS_INL_H_
