/*!
 * \file 
 * \brief 
 * \author 
*/
#include "./nms_3d-inl.h"

namespace mxnet {
namespace op {

template <>
void NMS3DForward<cpu>(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "NotImplemented";
}

DMLC_REGISTER_PARAMETER(NMS3DParam);

NNVM_REGISTER_OP(_contrib_NMS3D)
.describe("NMS3D foward.")
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<NMS3DParam>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 2;
})
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"boxes"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"idx","bbox_after_nms"};
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  const NMS3DParam param = nnvm::get<NMS3DParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1) << "Input:[boxes]";

  mxnet::TShape dshape = in_shape->at(0);
  CHECK_EQ(dshape.ndim(), 3) << "boxes should be (b, n, 10)";
  CHECK_EQ(dshape[2], 10) << "boxes should be (b, n, 10), including coordinates of 4 corners and 2 z coordinates";

  // out: [b,m]
  out_shape->clear();
  out_shape->push_back(Shape2(dshape[0], param.max_keep));
  out_shape->push_back(Shape3(dshape[0], param.max_keep, 10));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 1);
  int dtype = (*in_type)[0];
  // CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(mshadow::kInt32);
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", NMS3DForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("boxes", "NDArray-or-Symbol", "2D boxes with rotation, 3D tensor")
.add_arguments(NMS3DParam::__FIELDS__());

}
}
