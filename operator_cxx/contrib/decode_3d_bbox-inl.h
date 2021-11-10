/*!
 * \file
 * \brief decode 3d bbox
 * \author Lve Fan
 */
#ifndef MXNET_OPERATOR_CONTRIB_DECODE_3D_BBOX_INL_H_
#define MXNET_OPERATOR_CONTRIB_DECODE_3D_BBOX_INL_H_

#include <mxnet/operator_util.h>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include "mshadow_op.h"
#include "mxnet_op.h"
#include <dmlc/parameter.h>
#include "operator_common.h"
#include "tensor/init_op.h"

namespace mxnet {
namespace op {

struct DecodeParam : public dmlc::Parameter<DecodeParam> {
  bool is_bin;
  DMLC_DECLARE_PARAMETER(DecodeParam) {
    DMLC_DECLARE_FIELD(is_bin)
      .set_default(false)
      .describe("Max number of box to keep.");
  }
};

const float EPS = 1e-8;
#define MACRO_MAX(x,y) ((x) > (y) ? (x) : (y))
#define MACRO_MIN(x,y) ((x) < (y) ? (x) : (y))
// #define DEBUG

template <typename DType>
struct Point {
  DType x, y;
  MSHADOW_XINLINE Point() {}
  MSHADOW_XINLINE Point(DType _x, DType _y) { x = _x, y = _y; }

  MSHADOW_XINLINE void set(DType _x, DType _y) {
    x = _x;
    y = _y;
  }

  MSHADOW_XINLINE void rotate(DType sin, DType cos) {
    DType _x = x * cos - y * sin;
    DType _y = x * sin + y * cos;
    x = _x;
    y = _y;
  }

  MSHADOW_XINLINE Point operator+(const Point &b) const {
    return Point(x + b.x, y + b.y);
  }

  MSHADOW_XINLINE Point operator-(const Point &b) const {
    return Point(x - b.x, y - b.y);
  }
};

struct Decode3DBboxBinKernelGPU {

  template <typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType *bbox_delta, const DType *pc_laser_frame,
                                  DType *decoded_bbox, int box_type) {
    // assert(box_type == 7);
    DType pc_x = pc_laser_frame[index * 3 + 0];
    DType pc_y = pc_laser_frame[index * 3 + 1];
    DType pc_z = pc_laser_frame[index * 3 + 2];
    // DType pc_z = pc_laser_frame[index * 3 + 2];

    DType azimuth_this_point = atan2(pc_y, pc_x);
    #ifdef DEBUG
    printf("azimuth_this_point:%f\n",azimuth_this_point);
    printf("pc_x:%f\n",pc_x);
    printf("pc_y:%f\n",pc_y);
    #endif

    DType delta_x    = bbox_delta[index * box_type + 0];
    DType delta_y    = bbox_delta[index * box_type + 1];
    DType delta_z    = bbox_delta[index * box_type + 2];
    DType log_width  = bbox_delta[index * box_type + 3];
    DType log_length = bbox_delta[index * box_type + 4];
    DType log_height = bbox_delta[index * box_type + 5];
    DType yaw    = bbox_delta[index * box_type + 6];
    #ifdef DEBUG
    printf("delta_x_input:%f\n",delta_x);
    printf("delta_y_input:%f\n",delta_y);
    printf("log_width:%f\n",log_width);
    printf("log_length:%f\n",log_length);
    printf("cos_yaw:%f\n",cos_yaw);
    printf("sin_yaw:%f\n",sin_yaw);
    printf("z0:%f\n",z0);
    printf("log_height:%f\n",log_height);
    #endif

    DType cos_azimuth = cos(azimuth_this_point);
    DType sin_azimuth = sin(azimuth_this_point);
    #ifdef DEBUG
    printf("cos_azimuth:%f\n",cos_azimuth);
    printf("sin_azimuth:%f\n",sin_azimuth);
    #endif

    DType width  = exp(log_width);
    DType length = exp(log_length);
    DType height = exp(log_height);

    DType delta_x_laser = delta_x * cos_azimuth - delta_y * sin_azimuth;
    DType delta_y_laser = delta_x * sin_azimuth + delta_y * cos_azimuth;
    #ifdef DEBUG
    printf("delta_x_laser:%f\n",delta_x_laser);
    printf("delta_y_laser:%f\n",delta_y_laser);
    #endif

    DType bbox_center_x = pc_x + delta_x_laser;
    DType bbox_center_y = pc_y + delta_y_laser;
    DType bbox_center_z = pc_z + delta_z;
    DType z0 = bbox_center_z - height / 2.0;
    Point<DType> center(bbox_center_x, bbox_center_y);

    DType relative_yaw = yaw;
    DType yaw_laser_frame = relative_yaw + azimuth_this_point;
    #ifdef DEBUG
    printf("relative yaw:%f\n",relative_yaw);
    printf("yaw_laser_frame:%f\n",yaw_laser_frame);
    #endif

    DType sin_yaw_laser = sin(yaw_laser_frame);
    DType cos_yaw_laser = cos(yaw_laser_frame);

    Point<DType> A(0.5 * length, -0.5 * width);
    Point<DType> B(-0.5 * length, -0.5 * width);
    Point<DType> C(-0.5 * length, 0.5 * width);
    Point<DType> D(0.5 * length, 0.5 * width);

    A.rotate(sin_yaw_laser, cos_yaw_laser);
    B.rotate(sin_yaw_laser, cos_yaw_laser);
    C.rotate(sin_yaw_laser, cos_yaw_laser);
    D.rotate(sin_yaw_laser, cos_yaw_laser);
    #ifdef DEBUG
    printf("A.x:%f, A.y:%f\n",A.x, A.y);
    printf("B.x:%f, B.y:%f\n",B.x, B.y);
    printf("C.x:%f, C.y:%f\n",C.x, C.y);
    printf("D.x:%f, D.y:%f\n",D.x, D.y);
    #endif

    A = A + center;
    B = B + center;
    C = C + center;
    D = D + center;

    decoded_bbox[index * 10 + 0] = A.x;
    decoded_bbox[index * 10 + 1] = A.y;
    decoded_bbox[index * 10 + 2] = B.x;
    decoded_bbox[index * 10 + 3] = B.y;
    decoded_bbox[index * 10 + 4] = C.x;
    decoded_bbox[index * 10 + 5] = C.y;
    decoded_bbox[index * 10 + 6] = D.x;
    decoded_bbox[index * 10 + 7] = D.y;
    decoded_bbox[index * 10 + 8] = z0;
    decoded_bbox[index * 10 + 9] = z0 + height;

  }
};

struct Decode3DBboxKernelGPU {

  template <typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType *bbox_delta, const DType *pc_laser_frame,
                                  DType *decoded_bbox, int box_type) {
    assert(box_type == 8);
    DType pc_x = pc_laser_frame[index * 3 + 0];
    DType pc_y = pc_laser_frame[index * 3 + 1];
    // DType pc_z = pc_laser_frame[index * 3 + 2];

    DType azimuth_this_point = atan2(pc_y, pc_x);
    #ifdef DEBUG
    printf("azimuth_this_point:%f\n",azimuth_this_point);
    printf("pc_x:%f\n",pc_x);
    printf("pc_y:%f\n",pc_y);
    #endif

    DType delta_x    = bbox_delta[index * box_type + 0];
    DType delta_y    = bbox_delta[index * box_type + 1];
    DType log_width  = bbox_delta[index * box_type + 2];
    DType log_length = bbox_delta[index * box_type + 3];
    DType cos_yaw    = bbox_delta[index * box_type + 4];
    DType sin_yaw    = bbox_delta[index * box_type + 5];
    DType z0         = bbox_delta[index * box_type + 6];
    DType log_height = bbox_delta[index * box_type + 7];
    #ifdef DEBUG
    printf("delta_x_input:%f\n",delta_x);
    printf("delta_y_input:%f\n",delta_y);
    printf("log_width:%f\n",log_width);
    printf("log_length:%f\n",log_length);
    printf("cos_yaw:%f\n",cos_yaw);
    printf("sin_yaw:%f\n",sin_yaw);
    printf("z0:%f\n",z0);
    printf("log_height:%f\n",log_height);
    #endif

    DType cos_azimuth = cos(azimuth_this_point);
    DType sin_azimuth = sin(azimuth_this_point);
    #ifdef DEBUG
    printf("cos_azimuth:%f\n",cos_azimuth);
    printf("sin_azimuth:%f\n",sin_azimuth);
    #endif

    delta_x = delta_x * fabs(delta_x);
    delta_y = delta_y * fabs(delta_y);
    #ifdef DEBUG
    printf("delta_x_square:%f\n",delta_x);
    printf("delta_y_square:%f\n",delta_y);
    #endif

    DType width  = exp(log_width);
    DType length = exp(log_length);
    DType height = exp(log_height);

    DType delta_x_laser = delta_x * cos_azimuth - delta_y * sin_azimuth;
    DType delta_y_laser = delta_x * sin_azimuth + delta_y * cos_azimuth;
    #ifdef DEBUG
    printf("delta_x_laser:%f\n",delta_x_laser);
    printf("delta_y_laser:%f\n",delta_y_laser);
    #endif

    DType bbox_center_x = pc_x + delta_x_laser;
    DType bbox_center_y = pc_y + delta_y_laser;
    Point<DType> center(bbox_center_x, bbox_center_y);

    DType relative_yaw = atan2(sin_yaw, cos_yaw);
    DType yaw_laser_frame = relative_yaw + azimuth_this_point;
    #ifdef DEBUG
    printf("relative yaw:%f\n",relative_yaw);
    printf("yaw_laser_frame:%f\n",yaw_laser_frame);
    #endif

    DType sin_yaw_laser = sin(yaw_laser_frame);
    DType cos_yaw_laser = cos(yaw_laser_frame);

    Point<DType> A(0.5 * length, -0.5 * width);
    Point<DType> B(-0.5 * length, -0.5 * width);
    Point<DType> C(-0.5 * length, 0.5 * width);
    Point<DType> D(0.5 * length, 0.5 * width);

    A.rotate(sin_yaw_laser, cos_yaw_laser);
    B.rotate(sin_yaw_laser, cos_yaw_laser);
    C.rotate(sin_yaw_laser, cos_yaw_laser);
    D.rotate(sin_yaw_laser, cos_yaw_laser);
    #ifdef DEBUG
    printf("A.x:%f, A.y:%f\n",A.x, A.y);
    printf("B.x:%f, B.y:%f\n",B.x, B.y);
    printf("C.x:%f, C.y:%f\n",C.x, C.y);
    printf("D.x:%f, D.y:%f\n",D.x, D.y);
    #endif

    A = A + center;
    B = B + center;
    C = C + center;
    D = D + center;

    decoded_bbox[index * 10 + 0] = A.x;
    decoded_bbox[index * 10 + 1] = A.y;
    decoded_bbox[index * 10 + 2] = B.x;
    decoded_bbox[index * 10 + 3] = B.y;
    decoded_bbox[index * 10 + 4] = C.x;
    decoded_bbox[index * 10 + 5] = C.y;
    decoded_bbox[index * 10 + 6] = D.x;
    decoded_bbox[index * 10 + 7] = D.y;
    decoded_bbox[index * 10 + 8] = z0;
    decoded_bbox[index * 10 + 9] = z0 + height;

  }
};

template <typename xpu>
void Decode3DBboxForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  const int B = in_data[0].size(0);
  const int N = in_data[0].size(1);
  const int box_type = in_data[0].size(2);
  const DecodeParam param = nnvm::get<DecodeParam>(attrs.parsed);
  bool is_bin = param.is_bin;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bbox_delta = in_data[0].dptr<DType>();
    const DType *pc_laser_frame = in_data[1].dptr<DType>();
    DType *decoded_bbox = out_data[0].dptr<DType>();
    Fill<true>(s, out_data[0], kWriteTo, 0);
    if (is_bin)
      mxnet_op::Kernel<Decode3DBboxBinKernelGPU, xpu>::Launch(
          s, N * B, bbox_delta, pc_laser_frame, decoded_bbox, box_type);
    else
      mxnet_op::Kernel<Decode3DBboxKernelGPU, xpu>::Launch(
          s, N * B, bbox_delta, pc_laser_frame, decoded_bbox, box_type);
  });
}
}  // namespace op
}  // namespace mxnet

#endif
