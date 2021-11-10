/*!
 * Copyright (c) 2017 by Contributors
 * \file rotated_iou-inl.h
 * \brief rotated IoU
 * \author Feng Wang
 */
#ifndef MXNET_OPERATOR_CONTRIB_ROTATED_IOU_INL_H_
#define MXNET_OPERATOR_CONTRIB_ROTATED_IOU_INL_H_

#include <mxnet/operator_util.h>
#include <utility>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

const float EPS = 1e-8;
#define MACRO_MAX(x,y) ((x) > (y) ? (x) : (y))
#define MACRO_MIN(x,y) ((x) < (y) ? (x) : (y))

template <typename DType>
struct Point {
  DType x, y;
  MSHADOW_XINLINE Point() {}
  MSHADOW_XINLINE Point(DType _x, DType _y) { x = _x, y = _y; }

  MSHADOW_XINLINE void set(DType _x, DType _y) {
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

/*!
 * \brief Kernel for computing rotated IoU
 * box_type: 5 or 7
 */
struct RotateIoUKernelGPU {
  template <typename DType>
  MSHADOW_XINLINE static bool isEqual(DType d1, DType d2) {
    return fabs((d1 - d2) / MACRO_MIN(d1, d2)) < EPS;
  }

  template <typename DType>
  MSHADOW_XINLINE static DType cross(const Point<DType> &a,
                                     const Point<DType> &b) {
    return a.x * b.y - a.y * b.x;
  }

  template <typename DType>
  MSHADOW_XINLINE static DType cross(const Point<DType> &p1,
                                     const Point<DType> &p2,
                                     const Point<DType> &p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
  }

  template <typename DType>
  MSHADOW_XINLINE static int check_rect_cross(const Point<DType> &p1,
                                              const Point<DType> &p2,
                                              const Point<DType> &q1,
                                              const Point<DType> &q2) {
    int ret = MACRO_MIN(p1.x, p2.x) <= MACRO_MAX(q1.x, q2.x) &&
              MACRO_MIN(q1.x, q2.x) <= MACRO_MAX(p1.x, p2.x) &&
              MACRO_MIN(p1.y, p2.y) <= MACRO_MAX(q1.y, q2.y) &&
              MACRO_MIN(q1.y, q2.y) <= MACRO_MAX(p1.y, p2.y);
    return ret;
  }

  template <typename DType>
  MSHADOW_XINLINE static int check_in_box2d(const DType *box,
                                            const Point<DType> &p) {
    // params: box (5) [x, y, w, h, angle]

    DType angle_cos = cos(-box[4]),
          angle_sin = sin(
              -box[4]);  // rotate the point in the opposite direction of box
    DType rot_x =
        (p.x - box[0]) * angle_cos + (p.y - box[1]) * angle_sin + box[0];
    DType rot_y =
        -(p.x - box[0]) * angle_sin + (p.y - box[1]) * angle_cos + box[1];
    return (rot_x >= box[0] - box[2] / 2 && rot_x <= box[0] + box[2] / 2 &&
            rot_y >= box[1] - box[3] / 2 && rot_y <= box[1] + box[3] / 2);
  }

  template <typename DType>
  MSHADOW_XINLINE static int check_in_box2d_xyzwlh(const DType *box,
                                            const Point<DType> &p) {
    // params: box (7) [x, y, z, w, l, h, angle]

    DType angle_cos = cos(-box[6]),
          angle_sin = sin(
              -box[6]);  // rotate the point in the opposite direction of box
    DType rot_x =
        (p.x - box[0]) * angle_cos + (p.y - box[1]) * angle_sin + box[0];
    DType rot_y =
        -(p.x - box[0]) * angle_sin + (p.y - box[1]) * angle_cos + box[1];
    return (rot_x >= box[0] - box[3] / 2 && rot_x <= box[0] + box[3] / 2 &&
            rot_y >= box[1] - box[4] / 2 && rot_y <= box[1] + box[4] / 2);
  }

  template <typename DType>
  MSHADOW_XINLINE static int check_in_box2d_8pts(const DType *box,
                                                 const Point<DType> &p) {
    // params: box (8)
    int flag = -1;
    for (int i = 0; i < 4; i++) {
      int j = (i + 1) % 4;
      DType position = (box[2 * j] - box[2 * i]) * (p.y - box[2 * i + 1]) -
                       (box[2 * j + 1] - box[2 * i + 1]) * (p.x - box[2 * i]);
      if (flag == -1)
        flag = (position >= static_cast<DType>(0));
      else {
        if (flag != (position >= static_cast<DType>(0))) return false;
      }
    }
    return true;
  }

  template <typename DType>
  MSHADOW_XINLINE static int intersection(const Point<DType> &p1,
                                          const Point<DType> &p0,
                                          const Point<DType> &q1,
                                          const Point<DType> &q0,
                                          Point<DType> &ans) {
    // fast exclusion
    using namespace std;
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;
    DType A1 = p1.y - p0.y;
    DType B1 = p0.x - p1.x;
    DType C1 = A1 * p0.x + B1 * p0.y;

    DType A2 = q1.y - q0.y;
    DType B2 = q0.x - q1.x;
    DType C2 = A2 * q0.x + B2 * q0.y;

    DType det = A1 * B2 - A2 * B1;

    if (isEqual(det, DType(0.0))) {
      return 0;
    }
    else {
      DType x = (B2 * C1 - B1 * C2) / det;
      DType y = (A1 * C2 - A2 * C1) / det;
      bool online1 = ((min(p0.x, p1.x) < x || isEqual(min(p0.x, p1.x), x))
          && (max(p0.x, p1.x) > x || isEqual(max(p0.x, p1.x), x))
          && (min(p0.y, p1.y) < y || isEqual(min(p0.y, p1.y), y))
          && (max(p0.y, p1.y) > y || isEqual(max(p0.y, p1.y), y))
          );
      bool online2 = ((min(q0.x, q1.x) < x || isEqual(min(q0.x, q1.x), x))
          && (max(q0.x, q1.x) > x || isEqual(max(q0.x, q1.x), x))
          && (min(q0.y, q1.y) < y || isEqual(min(q0.y, q1.y), y))
          && (max(q0.y, q1.y) > y || isEqual(max(q0.y, q1.y), y))
          );
      if (online1 && online2) {
        ans.x = x;
        ans.y = y;
        return 1;
      }
    }
    return 0;
  }

  template <typename DType>
  MSHADOW_XINLINE static void rotate_around_center(const Point<DType> &center,
                                                   const DType angle_cos,
                                                   const DType angle_sin,
                                                   Point<DType> &p) {
    DType new_x =
        (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
    DType new_y =
        -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
  }

  template <typename DType>
  MSHADOW_XINLINE static int point_cmp(const Point<DType> &a,
                                       const Point<DType> &b,
                                       const Point<DType> &center) {
    return atan2(a.y - center.y, a.x - center.x) >
           atan2(b.y - center.y, b.x - center.x);
  }

  template <typename DType>
  MSHADOW_XINLINE static DType box_overlap_xywh(const DType *box_a,
                                                const DType *box_b) {
    // params: box_a (5) [x, y, w, h, angle]
    // params: box_b (5) [x, y, w, h, angle]

    DType a_x = box_a[0], a_y = box_a[1], a_w = box_a[2], a_h = box_a[3],
          a_angle = box_a[4];
    DType b_x = box_b[0], b_y = box_b[1], b_w = box_b[2], b_h = box_b[3],
          b_angle = box_b[4];

    Point<DType> center_a(a_x, a_y);
    Point<DType> center_b(b_x, b_y);

    Point<DType> box_a_corners[5];
    box_a_corners[0].set(a_x - a_w / 2, a_y - a_h / 2);
    box_a_corners[1].set(a_x + a_w / 2, a_y - a_h / 2);
    box_a_corners[2].set(a_x + a_w / 2, a_y + a_h / 2);
    box_a_corners[3].set(a_x - a_w / 2, a_y + a_h / 2);

    Point<DType> box_b_corners[5];
    box_b_corners[0].set(b_x - b_w / 2, b_y - b_h / 2);
    box_b_corners[1].set(b_x + b_w / 2, b_y - b_h / 2);
    box_b_corners[2].set(b_x + b_w / 2, b_y + b_h / 2);
    box_b_corners[3].set(b_x - b_w / 2, b_y + b_h / 2);

    // get oriented corners
    DType a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    DType b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
      rotate_around_center(center_a, a_angle_cos, a_angle_sin,
                           box_a_corners[k]);
      rotate_around_center(center_b, b_angle_cos, b_angle_sin,
                           box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point<DType> cross_points[16];
    Point<DType> poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                            box_b_corners[j + 1], box_b_corners[j],
                            cross_points[cnt]);
        if (flag) {
          poly_center = poly_center + cross_points[cnt];
          cnt++;
        }
      }
    }

    // check corners
    for (int k = 0; k < 4; k++) {
      if (check_in_box2d(box_a, box_b_corners[k])) {
        poly_center = poly_center + box_b_corners[k];
        cross_points[cnt] = box_b_corners[k];
        cnt++;
      }
      if (check_in_box2d(box_b, box_a_corners[k])) {
        poly_center = poly_center + box_a_corners[k];
        cross_points[cnt] = box_a_corners[k];
        cnt++;
      }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point<DType> temp;
    for (int j = 0; j < cnt - 1; j++) {
      for (int i = 0; i < cnt - j - 1; i++) {
        if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
          temp = cross_points[i];
          cross_points[i] = cross_points[i + 1];
          cross_points[i + 1] = temp;
        }
      }
    }

    // get the overlap areas
    DType area = 0;
    for (int k = 0; k < cnt - 1; k++) {
      area += cross(cross_points[k] - cross_points[0],
                    cross_points[k + 1] - cross_points[0]);
    }

    return fabsf(area) / 2.0;
  }

  template <typename DType>
  MSHADOW_XINLINE static DType box_overlap_xyzwlh(const DType *box_a,
                                                const DType *box_b) {
    // params: box_a (5) [x, y, z, w, l, h, angle]
    // params: box_b (5) [x, y, z, w, l, h, angle]

    DType a_x = box_a[0], a_y = box_a[1], a_w = box_a[3], a_l = box_a[4],
          a_angle = box_a[6];
    DType b_x = box_b[0], b_y = box_b[1], b_w = box_b[3], b_l = box_b[4],
          b_angle = box_b[6];

    Point<DType> center_a(a_x, a_y);
    Point<DType> center_b(b_x, b_y);

    Point<DType> box_a_corners[5];
    box_a_corners[0].set(a_x - a_w / 2, a_y - a_l / 2);
    box_a_corners[1].set(a_x + a_w / 2, a_y - a_l / 2);
    box_a_corners[2].set(a_x + a_w / 2, a_y + a_l / 2);
    box_a_corners[3].set(a_x - a_w / 2, a_y + a_l / 2);

    Point<DType> box_b_corners[5];
    box_b_corners[0].set(b_x - b_w / 2, b_y - b_l / 2);
    box_b_corners[1].set(b_x + b_w / 2, b_y - b_l / 2);
    box_b_corners[2].set(b_x + b_w / 2, b_y + b_l / 2);
    box_b_corners[3].set(b_x - b_w / 2, b_y + b_l / 2);

    // get oriented corners
    DType a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    DType b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
      rotate_around_center(center_a, a_angle_cos, a_angle_sin,
                           box_a_corners[k]);
      rotate_around_center(center_b, b_angle_cos, b_angle_sin,
                           box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point<DType> cross_points[16];
    Point<DType> poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                            box_b_corners[j + 1], box_b_corners[j],
                            cross_points[cnt]);
        if (flag) {
          poly_center = poly_center + cross_points[cnt];
          cnt++;
        }
      }
    }

    // check corners
    for (int k = 0; k < 4; k++) {
      if (check_in_box2d_xyzwlh(box_a, box_b_corners[k])) {
        poly_center = poly_center + box_b_corners[k];
        cross_points[cnt] = box_b_corners[k];
        cnt++;
      }
      if (check_in_box2d_xyzwlh(box_b, box_a_corners[k])) {
        poly_center = poly_center + box_a_corners[k];
        cross_points[cnt] = box_a_corners[k];
        cnt++;
      }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point<DType> temp;
    for (int j = 0; j < cnt - 1; j++) {
      for (int i = 0; i < cnt - j - 1; i++) {
        if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
          temp = cross_points[i];
          cross_points[i] = cross_points[i + 1];
          cross_points[i + 1] = temp;
        }
      }
    }

    // get the overlap areas
    DType area = 0;
    for (int k = 0; k < cnt - 1; k++) {
      area += cross(cross_points[k] - cross_points[0],
                    cross_points[k + 1] - cross_points[0]);
    }

    return fabsf(area) / 2.0;
  }

  template <typename DType>
  MSHADOW_XINLINE static DType box_overlap_8pts(const DType *box_a,
                                                const DType *box_b) {
    // params: box_a (8)
    // params: box_b (8)

    Point<DType> box_a_corners[5];
    box_a_corners[0].set(box_a[0], box_a[1]);
    box_a_corners[1].set(box_a[2], box_a[3]);
    box_a_corners[2].set(box_a[4], box_a[5]);
    box_a_corners[3].set(box_a[6], box_a[7]);

    Point<DType> box_b_corners[5];
    box_b_corners[0].set(box_b[0], box_b[1]);
    box_b_corners[1].set(box_b[2], box_b[3]);
    box_b_corners[2].set(box_b[4], box_b[5]);
    box_b_corners[3].set(box_b[6], box_b[7]);

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point<DType> cross_points[16];
    Point<DType> poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                            box_b_corners[j + 1], box_b_corners[j],
                            cross_points[cnt]);
        if (flag) {
          poly_center = poly_center + cross_points[cnt];
          cnt++;
        }
      }
    }

    // check corners
    for (int k = 0; k < 4; k++) {
      if (check_in_box2d_8pts(box_a, box_b_corners[k])) {
        poly_center = poly_center + box_b_corners[k];
        cross_points[cnt] = box_b_corners[k];
        cnt++;
      }
      if (check_in_box2d_8pts(box_b, box_a_corners[k])) {
        poly_center = poly_center + box_a_corners[k];
        cross_points[cnt] = box_a_corners[k];
        cnt++;
      }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point<DType> temp;
    for (int j = 0; j < cnt - 1; j++) {
      for (int i = 0; i < cnt - j - 1; i++) {
        if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
          temp = cross_points[i];
          cross_points[i] = cross_points[i + 1];
          cross_points[i + 1] = temp;
        }
      }
    }

    // get the overlap areas
    DType area = 0;
    for (int k = 0; k < cnt - 1; k++) {
      area += cross(cross_points[k] - cross_points[0],
                    cross_points[k + 1] - cross_points[0]);
    }

    return fabsf(area) / 2.0;
  }

  template <typename DType>
  MSHADOW_XINLINE static DType iou_bev(const DType *box_a, const DType *box_b) {
    // params: box_a (5) [x, y, w, h, angle]
    // params: box_b (5) [x, y, w, h, angle]
    DType sa = box_a[2] * box_a[3];
    DType sb = box_b[2] * box_b[3];
    if (sa < EPS || sb < EPS) return DType(0);
    DType s_overlap = box_overlap_xywh(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
  }

  template <typename DType>
  MSHADOW_XINLINE static DType iou_bev_8pts(const DType *box_a,
                                            const DType *box_b) {
    // params: box_a (8) 4 points
    // params: box_b (8) 4 points
    DType sa = (box_a[2] - box_a[0]) * (box_a[5] - box_a[1]) - (box_a[3] - box_a[1]) * (box_a[4] - box_a[0]);
    sa += (box_a[4] - box_a[0]) * (box_a[7] - box_a[1]) - (box_a[5] - box_a[1]) * (box_a[6] - box_a[0]);
    DType sb = (box_b[2] - box_b[0]) * (box_b[5] - box_b[1]) - (box_b[3] - box_b[1]) * (box_b[4] - box_b[0]);
    sb += (box_b[4] - box_b[0]) * (box_b[7] - box_b[1]) - (box_b[5] - box_b[1]) * (box_b[6] - box_b[0]);
    sa = fabsf(sa) / 2.0;
    sb = fabsf(sb) / 2.0;

    if (sa < EPS || sb < EPS) return DType(0);
    DType s_overlap = box_overlap_8pts(box_a, box_b);
    //printf("%f %f %f\n", sa, sb, s_overlap);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
  }

  template <typename DType>
  MSHADOW_XINLINE static DType iou_3d(const DType *box_a, const DType *box_b) {
    // params: box_a (7) [x, y, z, w, l, h, angle]
    // params: box_b (7) [x, y, z, w, l, h, angle]
    DType sa = box_a[3] * box_a[4] * box_a[5];
    DType sb = box_b[3] * box_b[4] * box_b[5];
    if (sa < EPS || sb < EPS) return DType(0);
    DType s_overlap = box_overlap_xyzwlh(box_a, box_b);
    DType h_overlap = MACRO_MAX(static_cast<DType>(0.0), 
     MACRO_MIN(box_a[2] + box_a[5] / static_cast<DType>(2.0), box_b[2] + box_b[5] / static_cast<DType>(2.0))
     - MACRO_MAX(box_a[2] - box_a[5] / static_cast<DType>(2.0), box_b[2] - box_b[5] / static_cast<DType>(2.0)));
    return s_overlap * h_overlap / fmaxf(sa + sb - s_overlap * h_overlap, EPS);
  }

  template <typename DType>
  MSHADOW_XINLINE static void Map(int index, int n1, int n2,
                                  const DType *boxes1, const DType *boxes2,
                                  DType *ious, int box_type) {
    int b1 = index / n2;
    int b2 = index % n2;
    if (box_type == 5) {
      ious[index] = iou_bev(boxes1 + b1 * box_type, boxes2 + b2 * box_type);
    } else if (box_type == 8) {
      ious[index] = iou_bev_8pts(boxes1 + b1 * box_type, boxes2 + b2 * box_type);
    } else if (box_type == 7) {
      ious[index] = iou_3d(boxes1 + b1 * box_type, boxes2 + b2 * box_type);
    }
  }
};

template <typename xpu>
void RotatedIOUForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  // input: boxes1(N1, 5 or 7 or 8), boxes2(N2, 5 or 7 or 8)
  // output: ious(N1, N2)
  const int N1 = in_data[0].size(0);
  const int N2 = in_data[1].size(0);
  const int box_type = in_data[0].size(1);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *boxes1 = in_data[0].dptr<DType>();
    const DType *boxes2 = in_data[1].dptr<DType>();
    DType *ious = out_data[0].dptr<DType>();
    Fill<true>(s, out_data[0], kWriteTo, -1);
    mxnet_op::Kernel<RotateIoUKernelGPU, xpu>::Launch(
        s, N1 * N2, N1, N2, boxes1, boxes2, ious, box_type);
  });
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ROTATED_IOU_INL_H_
