/*!
 * \file 
 * \brief 
 * \author 
*/
#include <stdio.h>
#include <math.h>
#include "./nms_3d-inl.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//#define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;
struct Point {
    float x, y;
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }

    __device__ float norm(){
        return sqrt(x * x + y * y);
    }

    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

__device__ inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

__device__ inline float inner_product(const Point &A, const Point &B, const Point &C){
    // vector:AB * vector:AC
    Point AB = B - A;
    Point AC = C - A;
    return AB.x * AC.x + AB.y * AC.y;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross_3d(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

__device__ inline int check_in_box3d(const float *box, const Point &P){
    // const float MARGIN = 1e-5;
    // 4 points: ABCD
    Point A(box[0],box[1]);
    Point B(box[2],box[3]);
    Point C(box[4],box[5]);
    Point D(box[6],box[7]);
    float dot1 = inner_product(B,A,P);
    if (dot1 < 0) return false;
    float dot2 = inner_product(B,C,P);
    if (dot2 < 0) return false;
    float dot3 = inner_product(D,A,P);
    if (dot3 < 0) return false;
    float dot4 = inner_product(D,C,P);
    if (dot4 < 0) return false;
    return true;
}

__device__ inline int check_in_box3d_anotherway(const float *box, const Point &P){
    const float MARGIN = -1e-2;
    // 4 points: ABCD
    Point A(box[0],box[1]);
    Point B(box[2],box[3]);
    Point C(box[4],box[5]);
    Point D(box[6],box[7]);
    Point AB_vec = B - A;
    Point BC_vec = C - B;
    Point CD_vec = D - C;
    Point DA_vec = A - D;
    auto is_clock_wise = cross(AB_vec,BC_vec);

    #ifdef DEBUG
    printf("AB_vec: (%f, %f)\n", AB_vec.x, AB_vec.y);
    printf("BC_vec: (%f, %f)\n", BC_vec.x, BC_vec.y);
    printf("CD_vec: (%f, %f)\n", CD_vec.x, CD_vec.y);
    printf("DA_vec: (%f, %f)\n", DA_vec.x, DA_vec.y);
    printf("is_clock_wise: %f\n", is_clock_wise);
    #endif

    Point PA_vec = A - P;
    float cross1 = cross(PA_vec, AB_vec);
    if (cross1 * is_clock_wise < MARGIN) {
        #ifdef DEBUG
        printf("cross1: %f, PA.x: %f, PA.y: %f\n", cross1, PA_vec.x, PA_vec.y);
        #endif
        return false;
    }

    Point PB_vec = B - P;
    float cross2 = cross(PB_vec, BC_vec);
    if (cross2 * is_clock_wise < MARGIN){
        #ifdef DEBUG
        printf("cross2: %f, PB.x: %f, PB.y: %f\n", cross2, PB_vec.x, PB_vec.y);
        #endif
        return false;
    }

    Point PC_vec = C - P;
    float cross3 = cross(PC_vec, CD_vec);
    if (cross3 * is_clock_wise < MARGIN) {
        #ifdef DEBUG
        printf("cross3: %f, PC.x: %f, PC.y: %f\n", cross3, PC_vec.x, PC_vec.y);
        #endif
        return false;
    }

    Point PD_vec = D - P;
    float cross4 = cross(PD_vec, DA_vec);
    if (cross4 * is_clock_wise < MARGIN) {
        #ifdef DEBUG
        printf("cross4: %f, PD.x: %f, PD.y: %f\n", cross4, PD_vec.x, PD_vec.y);
        #endif
        return false;
    }
    return true;
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion
    if (check_rect_cross_3d(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    }
    else{
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
    float new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float get_area(const float* box){
    float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3], x3 = box[4], y3 = box[5];
    float edge1 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    float edge2 = (x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2);
    return sqrt(edge1 * edge2);
}

__device__ inline float max4(const float x1, const float x2, const float x3, const float x4){
    float max = -1000000;
    if (x1 > max) max = x1;
    if (x2 > max) max = x2;
    if (x3 > max) max = x3;
    if (x4 > max) max = x4;
    return max;
}

__device__ inline float min4(const float x1, const float x2, const float x3, const float x4){
    float min = 1000000;
    if (x1 < min) min = x1;
    if (x2 < min) min = x2;
    if (x3 < min) min = x3;
    if (x4 < min) min = x4;
    return min;
}

__device__ inline float box_overlap(const float *box_a, const float *box_b){

    // float x_a_min = min4(box_a[0], box_a[2], box_a[4], box_a[6]);
    // float x_a_max = max4(box_a[0], box_a[2], box_a[4], box_a[6]);
    // float y_a_min = min4(box_a[1], box_a[3], box_a[5], box_a[7]);
    // float y_a_max = max4(box_a[1], box_a[3], box_a[5], box_a[7]);

    // float x_b_min = min4(box_b[0], box_b[2], box_b[4], box_b[6]);
    // float x_b_max = max4(box_b[0], box_b[2], box_b[4], box_b[6]);
    // float y_b_min = min4(box_b[1], box_b[3], box_b[5], box_b[7]);
    // float y_b_max = max4(box_b[1], box_b[3], box_b[5], box_b[7]);

    // if (x_a_max < x_b_min || x_a_min > x_b_max || y_a_max < y_b_min || y_a_min > y_b_max) return 0;
    // Point center_a;
    // center_a.set((box_a[0] + box_a[2] + box_a[4] + box_a[6]) / 4.0,
    //              (box_a[1] + box_a[3] + box_a[5] + box_a[7]) / 4.0);
    // // printf("center_a:(%f, %f)\n", center_a.x, center_a.y);
    // Point center_b;
    // center_b.set((box_b[0] + box_b[2] + box_b[4] + box_b[6]) / 4.0,
    //              (box_b[1] + box_b[3] + box_b[5] + box_b[7]) / 4.0);
    // // printf("center_b:(%f, %f)\n", center_b.x, center_b.y);
    // Point two_center_vec = center_a - center_b;
    // // printf("two_center_vec:(%f, %f)\n", two_center_vec.x, two_center_vec.y);
    // float center_dist = two_center_vec.norm();
    // // printf("center_dist:%f\n", center_dist);

    // float area_a = get_area(box_a);
    // float area_b = get_area(box_b);
    // float min_area = area_a < area_b ? area_a : area_b; 

    // if (center_dist < 0.2){
    //     return min_area;
    // }
    // else return 0;

    Point box_a_corners[5];
    box_a_corners[0].set(box_a[0],box_a[1]);
    box_a_corners[1].set(box_a[2],box_a[3]);
    box_a_corners[2].set(box_a[4],box_a[5]);
    box_a_corners[3].set(box_a[6],box_a[7]);

    Point box_b_corners[5];
    box_b_corners[0].set(box_b[0],box_b[1]);
    box_b_corners[1].set(box_b[2],box_b[3]);
    box_b_corners[2].set(box_b[4],box_b[5]);
    box_b_corners[3].set(box_b[6],box_b[7]);

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];


    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                #ifdef DEBUG
                printf("Intersect point (%f, %f)\n", cross_points[cnt].x, cross_points[cnt].y);
                #endif
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box3d_anotherway(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
            #ifdef DEBUG
            printf("Point (%f, %f) in box_a\n", box_b_corners[k].x, box_b_corners[k].y);
            #endif
        }
        if (check_in_box3d_anotherway(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
            #ifdef DEBUG
            printf("Point (%f, %f) in box_b\n", box_a_corners[k].x, box_a_corners[k].y);
            #endif
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    auto thread_id = threadIdx.x;
    for (int i = 0; i < cnt; i++){
        printf("thread: %d, All cross point %d: (%.3f, %.3f)\n", thread_id, i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b) {
    float height_a = box_a[9] - box_a[8];
    float height_b = box_b[9] - box_b[8];
    
    float overlap_height = fminf(box_a[9], box_b[9]) - fmaxf(box_a[8], box_b[8]);
    if (overlap_height < 0) overlap_height = 0;
    float area_a = get_area(box_a);
    float area_b = get_area(box_b);
    float volume_a = area_a * height_a;
    float volume_b = area_b * height_b;
    float overlap_2d = box_overlap(box_a, box_b);
    float volume_overlap = overlap_2d * overlap_height;
    float result = volume_overlap / fmaxf(volume_a + volume_b - volume_overlap, EPS); 
    #ifdef DEBUG
    printf("area_a=%f\n", area_a);
    printf("area_b=%f\n", area_b);
    printf("height_a=%f\n", height_a);
    printf("height_b=%f\n", height_b);
    printf("overlap_height=%f\n", overlap_height);
    printf("volume_a=%f\n", volume_a);
    printf("volume_b=%f\n", volume_b);
    printf("overlap_2d=%f\n", overlap_2d);
    printf("volume_overlap=%f\n", volume_overlap);
    printf("overlap result=%f\n", result);
    #endif
    return result;
}

__device__ inline float iou_normal(float const * const a, float const * const b) {
    float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
    float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0]) * (a[3] - a[1]);
    float Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / fmaxf(Sa + Sb - interS, EPS);
}

__global__ void nms_kernel_3d(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask,
                           bool normal_iou) {
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 10];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 10 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 0];
        block_boxes[threadIdx.x * 10 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 1];
        block_boxes[threadIdx.x * 10 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 2];
        block_boxes[threadIdx.x * 10 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 3];
        block_boxes[threadIdx.x * 10 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 4];
        block_boxes[threadIdx.x * 10 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 5];
        block_boxes[threadIdx.x * 10 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 6];
        block_boxes[threadIdx.x * 10 + 7] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 7];
        block_boxes[threadIdx.x * 10 + 8] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 8];
        block_boxes[threadIdx.x * 10 + 9] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 10 + 9];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 10;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            float iou_st;
            if(normal_iou){
                iou_st = iou_normal(cur_box, block_boxes + i * 10);
            }
            else{
                iou_st = iou_bev(cur_box, block_boxes + i * 10);
            }
            if (iou_st > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

__global__ void prepare_output_kernel_3d(const int N, const int max_keep,
                                      const int col_blocks, 
                                      unsigned long long *mask, 
                                      unsigned long long * remv_cpu, 
                                      int* keep_idx,
                                      const float *boxes,
                                      float *bbox_after_nms) {
   // unsigned long long remv_cpu[col_blocks];
   // memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));
   int num_to_keep = 0;
   for (int i = 0; i < N; i++) {
       if(num_to_keep >= max_keep) {break;}
       int nblock = i / THREADS_PER_BLOCK_NMS;
       int inblock = i % THREADS_PER_BLOCK_NMS;
    //    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
    //        keep_idx[num_to_keep++] = i;
    //        unsigned long long *p = &mask[0] + i * col_blocks;
    //        for (int j = nblock; j < col_blocks; j++) {
    //            remv_cpu[j] |= p[j];
    //        }
    //    }
       if (!(remv_cpu[nblock] & (1ULL << inblock))) {
           for (int k = 0; k < 10; k++){
               bbox_after_nms[num_to_keep * 10 + k] = boxes[i * 10 + k];
           }
           keep_idx[num_to_keep++] = i;
           unsigned long long *p = &mask[0] + i * col_blocks;
           for (int j = nblock; j < col_blocks; j++) {
               remv_cpu[j] |= p[j];
           }
       }
   }
}

template <>
void NMS3DForward<gpu>(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<TBlob>& in_data,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<TBlob>& out_data) {
    using namespace mshadow;
    size_t expected_in = 1;
    size_t expected_out = 2;
    // input: boxes(B,N,10), which is sorted with score
    // output: keep_idx(B,num_boxes)
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);
    CHECK_EQ(in_data[0].shape_[2], 10);
    CHECK_EQ(out_data[0].shape_[0], in_data[0].shape_[0]);

    const NMS3DParam param =
        nnvm::get<NMS3DParam>(attrs.parsed);
    const int B = in_data[0].size(0);
    const int N = in_data[0].size(1);
    const int max_keep = param.max_keep;
    const float iou_thres = param.iou_thres;
    const bool normal_iou = param.normal_iou;
    CHECK_EQ(out_data[0].shape_[1], max_keep);

    Stream<gpu>* s = ctx.get_stream<gpu>();
    auto stream = mshadow::Stream<gpu>::GetStream(s);
    // assume all the data and gradient have the same type
    MSHADOW_TYPE_SWITCH(in_data[0].type_flag_, DType, {
        const float* boxes = in_data[0].dptr<float>();
        int* keep_idx = out_data[0].dptr<int>();
        float* bbox_after_nms = out_data[1].dptr<float>();
        Fill<true>(s, out_data[0], kWriteTo, -1);
        Fill<true>(s, out_data[1], kWriteTo, 0.0);

        const int col_blocks = DIVUP(N, THREADS_PER_BLOCK_NMS);

        unsigned long long *mask_data = NULL;
        CHECK_ERROR(cudaMalloc((void**)&mask_data, B * N * col_blocks * sizeof(unsigned long long)));
        dim3 blocks(DIVUP(N, THREADS_PER_BLOCK_NMS),
                    DIVUP(N, THREADS_PER_BLOCK_NMS));
        dim3 threads(THREADS_PER_BLOCK_NMS);

        unsigned long long *remv_dev = NULL;
        CHECK_ERROR(cudaMalloc((void**)&remv_dev, col_blocks * sizeof(unsigned long long)));
        // iterate through batch
        for(int b = 0; b < B; b++) {
            // calculate overlap matrix
            nms_kernel_3d<<<blocks, threads>>>(N, iou_thres, boxes+b*N*10, mask_data+b*N*col_blocks, normal_iou);
            CHECK_ERROR(cudaMemset(remv_dev, 0, col_blocks * sizeof(unsigned long long)));
            prepare_output_kernel_3d<<<1,1,0,stream>>>(N, max_keep, col_blocks, 
                                                       mask_data+b*N*col_blocks, 
                                                       remv_dev, 
                                                       keep_idx + b * max_keep,
                                                       boxes + b * N * 10,
                                                       bbox_after_nms + b * max_keep * 10);
            cudaError_t err = cudaGetLastError();
            if (cudaSuccess != err) {
                LOG(FATAL) << "CUDA kernel failed : " << cudaGetErrorString(err);
                exit(-1);
            }
        }
        cudaFree(mask_data);
    });
}

NNVM_REGISTER_OP(_contrib_NMS3D)
.set_attr<FCompute>("FCompute<gpu>", NMS3DForward<gpu>);

}
}
