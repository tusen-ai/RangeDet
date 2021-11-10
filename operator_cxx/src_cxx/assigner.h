#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<iostream>
#include<cmath>
#include <bits/stdc++.h>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

Matrix<int, Dynamic, Dynamic> assign3D_v2(
        const py::EigenDRef <MatrixXf> pc,
        const py::EigenDRef <MatrixXf> bbox,
        const py::EigenDRef <MatrixXf> bbox_center,
        const py::EigenDRef <MatrixXf> bbox_radius,
        const py::EigenDRef <MatrixXf> mask,
        const py::EigenDRef <MatrixXf> is_in_nlz,
        float max_x_,
        float min_x_,
        float max_y_,
        float min_y_,
        float max_z_,
        float min_z_,
        float max_dist
) {
    auto num_pts = pc.rows();
    auto num_bbox = bbox.rows();

    Matrix<float, Dynamic, 3> A = bbox.block(0, 0, num_bbox, 3);
    Matrix<float, Dynamic, 3> B = bbox.block(0, 3, num_bbox, 3);
    Matrix<float, Dynamic, 3> C = bbox.block(0, 6, num_bbox, 3);
    Matrix<float, Dynamic, 3> D = bbox.block(0, 9, num_bbox, 3);
    Matrix<float, Dynamic, 3> E = bbox.block(0, 12, num_bbox, 3);
    // Matrix<float, Dynamic, 3> F = bbox.block(0,15,num_bbox,3);
    // Matrix<float, Dynamic, 3> G = bbox.block(0,18,num_bbox,3);
    // Matrix<float, Dynamic, 3> H = bbox.block(0,21,num_bbox,3);

    Matrix<int, Dynamic, 1> results = Matrix<int, Dynamic, 1>::Constant(num_pts, 1, -1);

    for (int i = 0; i != num_pts; i++) {
        if (mask(i, 0) < 0.5 || is_in_nlz(i, 0) > 0) continue;
        Matrix<float, 1, 3> P = pc.row(i);
        if (P(0) < min_x_ || P(0) > max_x_) continue;
        if (P(1) < min_y_ || P(1) > max_y_) continue;
        if (P(2) < min_z_ || P(2) > max_z_) continue;
        Matrix<float, Dynamic, 1> dist_to_center = (bbox_center.rowwise() - P).rowwise().squaredNorm();
        float min_dist = dist_to_center.minCoeff();
        if (min_dist > max_dist) continue;
        for (int j = 0; j != num_bbox; j++) {
            if (dist_to_center(j, 0) > bbox_radius(j, 0)) continue;
            if (P(2) <= A(j, 2) || P(2) >= E(j, 2)) continue;

            if (P(0) < A(j, 0) && P(0) < B(j, 0) && P(0) < C(j, 0) && P(0) < D(j, 0)) continue;

            if (P(1) < A(j, 1) && P(1) < B(j, 1) && P(1) < C(j, 1) && P(1) < D(j, 1)) continue;

            if (P(0) > A(j, 0) && P(0) > B(j, 0) && P(0) > C(j, 0) && P(0) > D(j, 0)) continue;

            if (P(1) > A(j, 1) && P(1) > B(j, 1) && P(1) > C(j, 1) && P(1) > D(j, 1)) continue;

            Matrix<float, 1, 3> BP = P - B.row(j);

            Matrix<float, 1, 3> BA = A.row(j) - B.row(j);
            auto dot1 = BA(0) * BP(0) + BA(1) * BP(1);
            if (dot1 <= 0) continue;

            Matrix<float, 1, 3> BC = C.row(j) - B.row(j);
            auto dot2 = BC(0) * BP(0) + BC(1) * BP(1);
            if (dot2 <= 0) continue;

            Matrix<float, 1, 3> DP = P - D.row(j);

            Matrix<float, 1, 3> DA = A.row(j) - D.row(j);
            auto dot3 = DA(0) * DP(0) + DA(1) * DP(1);
            if (dot3 <= 0) continue;

            Matrix<float, 1, 3> DC = C.row(j) - D.row(j);
            auto dot4 = DC(0) * DP(0) + DC(1) * DP(1);
            if (dot4 <= 0) continue;

            results(i) = j;
            break;

        }
    }
    return results;
}

Matrix<float, Dynamic, Dynamic> get_point_num(
        const py::EigenDRef <MatrixXf> bbox_inds_each_pt // [N, 1]
) {
    auto num_pts = bbox_inds_each_pt.rows();

    int MAX_BOX_NUM = 500;
    Matrix<float, Dynamic, 1> results = Matrix<float, Dynamic, 1>::Constant(num_pts, 1, -1);
    Matrix<float, Dynamic, 1> num_pt_in_each_bbox = Matrix<float, Dynamic, 1>::Constant(MAX_BOX_NUM, 1, 0);

    for (int i = 0; i != num_pts; i++) {
        auto bbox_inds = bbox_inds_each_pt(i, 0);
        if (bbox_inds < 0) continue;
        num_pt_in_each_bbox(bbox_inds) += 1;
    }
    for (int i = 0; i < num_pts; i++) {
        auto bbox_inds = bbox_inds_each_pt(i, 0);
        if (bbox_inds < 0) continue;
        results(i, 0) = num_pt_in_each_bbox(bbox_inds);
    }
    return results;
}

//PYBIND11_MODULE(assigner, m
//){
//m.def("assign3D_v2", &assign3D_v2);
//m.def("get_point_num", &get_point_num);
//}
