/*!
 * \file decode_3d_bbox.cc
 * \brief decode 3d bbox to laser frame
 * \author Lve Fan
*/
#include "./decode_3d_bbox-inl.h"
#include <dmlc/parameter.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DecodeParam);

NNVM_REGISTER_OP(_contrib_Decode3DBbox)
.describe("Decode3DBbox foward.")
.set_attr_parser(ParamParser<DecodeParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"bbox_deltas", "pc_laser_frame"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"decoded_bbox"};
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 2) << "Input:[bbox_deltas, pc_laser_frame]";

  mxnet::TShape dshape1 = in_shape->at(0);
  CHECK_EQ(dshape1.ndim(), 3) << "box deltas should be (b, n, 8)";
  if (dshape1[2] != 8 && dshape1[2] != 7){
    LOG_FATAL.stream() << "box deltas should be  (b, n, 8), or (b, n, 7) when use bin loss";
  }
  mxnet::TShape dshape2 = in_shape->at(1);
  CHECK_EQ(dshape2.ndim(), 3) << "point cloud in laser frame should be (b,n,3)";
  if (dshape2[2] != 3){
    LOG_FATAL.stream() << "point cloud in laser frame should be (b,n,3)";
  }

  out_shape->clear();
  out_shape->push_back(Shape3(dshape1[0], dshape1[1], 10));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2);
  int dtype = (*in_type)[0];
  // CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", Decode3DBboxForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("bbox_deltas", "NDArray-or-Symbol", "2D boxes with rotation, 3D tensor")
.add_argument("pc_laser_frame", "NDArray-or-Symbol", "2D boxes with rotation, 3D tensor")
.add_arguments(DecodeParam::__FIELDS__());
}
}
