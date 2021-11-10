/*!
 * Copyright (c) 2017 by Contributors
 * \file rotated_iou.cc
 * \brief rotated IOU
 * \author Feng Wang
*/
#include "./rotated_iou-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_RotatedIOU)
.describe("RotatedIOU foward.")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"boxes1", "boxes2"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"ious"};
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 2) << "Input:[boxes1, boxes2]";

  mxnet::TShape dshape1 = in_shape->at(0);
  CHECK_EQ(dshape1.ndim(), 2) << "boxes 1 should be (n, 5) or (n, 7) or (n, 8)";
  if (dshape1[1] != 5 && dshape1[1] != 7 && dshape1[1] != 8){
    LOG_FATAL.stream() << "boxes 1 should be (n, 5) or (n, 7) or (n, 8)";
  }
  mxnet::TShape dshape2 = in_shape->at(1);
  CHECK_EQ(dshape2.ndim(), 2) << "boxes 2 should be (n, 5) or (n, 7) or (n, 8)";
  if (dshape2[1] != 5 && dshape2[1] != 7 && dshape1[1] != 8){
    LOG_FATAL.stream() << "boxes 1 should be (n, 5) or (n, 7) or (n, 8)";
  }

  out_shape->clear();
  out_shape->push_back(Shape2(dshape1[0], dshape2[0]));
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
.set_attr<FCompute>("FCompute<cpu>", RotatedIOUForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("boxes1", "NDArray-or-Symbol", "2D boxes with rotation, 3D tensor")
.add_argument("boxes2", "NDArray-or-Symbol", "2D boxes with rotation, 3D tensor");

}
}
