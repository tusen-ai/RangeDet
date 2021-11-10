#include "overlap.h"
//#include "preprocessing.h"
#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace trtplus {

struct Point { //顶点
  float x, y;
};

struct Line { //线
  Point a, b;
  float angle; //极角
  Line &operator=(Line l) {
    a.x = l.a.x;
    a.y = l.a.y;
    b.x = l.b.x;
    b.y = l.b.y;
    angle = l.angle;
    return *this;
  }
};

class OverlapChecker {
private:
  static const int MAX_SIZE = 16;
  static constexpr float EPS = 1e-5;
  int pn, dq[MAX_SIZE], top, bot; //数组模拟双端队列
  int n = 8;
  int pn_start = 8;
  Point p[MAX_SIZE];
  Line l[MAX_SIZE];
  float height1 = -1;
  float height2 = -1;
  float height0 = -1;

public:
  void clear_dq() { memset(dq, 0, sizeof(dq)); }

  static int dblcmp(float k) { //精度函数
    if (fabs(k) < EPS)
      return 0;
    return k > 0 ? 1 : -1;
  }

  static float multi(Point p0, Point p1, Point p2) { //叉积
    return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
  }

  static bool cmp(const Line &l1, const Line &l2) {
    int d = dblcmp(l1.angle - l2.angle);
    if (!d)
      return dblcmp(OverlapChecker::multi(l1.a, l2.a, l2.b)) > 0;
    //大于0取半平面的左半，小于0取右半
    return d < 0;
  }

  void addLine(Line &l, float x1, float y1, float x2, float y2) {
    l.a.x = x1;
    l.a.y = y1;
    l.b.x = x2;
    l.b.y = y2;
    l.angle = atan2(y2 - y1, x2 - x1);
  }

  void getIntersect(Line l1, Line l2, Point &p) {
    float A1 = l1.b.y - l1.a.y;
    float B1 = l1.a.x - l1.b.x;
    float C1 = (l1.b.x - l1.a.x) * l1.a.y - (l1.b.y - l1.a.y) * l1.a.x;
    float A2 = l2.b.y - l2.a.y;
    float B2 = l2.a.x - l2.b.x;
    float C2 = (l2.b.x - l2.a.x) * l2.a.y - (l2.b.y - l2.a.y) * l2.a.x;
    p.x = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1);
    p.y = (C1 * A2 - C2 * A1) / (A1 * B2 - A2 * B1);
  }

  bool judge(Line l0, Line l1, Line l2) {
    Point p;
    getIntersect(l1, l2, p);
    return dblcmp(multi(p, l0.a, l0.b)) > 0;
    //大于0，是p在向量l0.a->l0.b的左边，小于0是在右边，当p不在半平面l0内时，返回true
  }

  bool checkClockwise(Point p0, Point p1, Point p2) { //判断是否点的顺序是顺时针
    return ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) > 0;
  }

  void HalfPlaneIntersect() {
    int i, j;
    std::sort(l, l + n, OverlapChecker::cmp); //极角排序
    for (int i = 0; i < n; ++i){
      auto line = l[i];
      // std::cout << "line: " << line.a.x << " " << line.a.y << " " << line.b.x << " " << line.b.y <<  " " << line.angle <<endl;
    }
    // std::cout << endl;

    for (i = 0, j = 0; i < n; i++)
      if (dblcmp(l[i].angle - l[j].angle) > 0)
        l[++j] = l[i]; //排除极角相同（从了l[1]开始比较）
    for (int i = 0; i < j+1; ++i){
      auto line = l[i];
      // std::cout << "line: " << line.a.x << " " << line.a.y << " " << line.b.x << " " << line.b.y <<  " " << line.angle <<endl;
    }
    int t = j + 1;     //个数
    dq[0] = 0;         //双端队列
    dq[1] = 1;         //开始入队列两条直线
    top = 1;
    bot = 0;
    for (i = 2; i < t; i++) {
      while (top > bot && judge(l[i], l[dq[top]], l[dq[top - 1]])){
        top--;
        // cout << "top: " << top << endl;
      }
      while (top > bot && judge(l[i], l[dq[bot]], l[dq[bot + 1]])){
        bot++;
        // cout << "bot: " << bot << endl;
      }
      dq[++top] = i;
      // std::cout << "top: " << top << endl;
      // std::cout << "dq: " << endl;
      // for (auto x: dq){
      //   std::cout << x << " ";
      // }
      // cout << endl;
    }
    while (top > bot && judge(l[dq[bot]], l[dq[top]], l[dq[top - 1]])){
      top--;
      // cout << "top: " << top << endl; 
    }
    while (top > bot && judge(l[dq[top]], l[dq[bot]], l[dq[bot + 1]])){
      bot++;
      // cout << "bot: " << bot << endl; 
    }
    dq[++top] = dq[bot];
    // cout << "top: " << top << endl; 
    // std::cout << "final dq:" << endl;
    // for (auto x: dq){std::cout << x << " ";}
    // cout << endl;
    for (pn = pn_start, i = bot; i < top; i++, pn++)
      getIntersect(l[dq[i + 1]], l[dq[i]], p[pn]); //更新重复利用p数组
  }

  float getArea(int start, int end) {
    if (end - start < 3)
      return 0;
    float area = 0;

    // std::cout << "start: " << start << " end: " << end << endl;
    // std::cout << "start_x: " << p[start].x << " start_y: " << p[start].y << endl;
    for (int i = start + 1; i < end - 1; i++){
      area += multi(p[start], p[i], p[i + 1]); //利用p数组求面积
      // std::cout << "i: " << i << " area: " << area << endl;
      // std::cout << "pi_x: " << p[i].x << " pi_y: " << p[i].y << endl;
    }
    if (area < 0)
      area = -area;
    return area / 2;
  }

  float getHeight(const float *box){
    return box[10];
  }

  float getOverlapHeight(const float *box1, const float *box2){
    float bot1 = box1[9];
    float height1 = box1[10];
    float top1 = bot1 + height1;
    float bot2 = box2[9];
    float height2 = box2[10];
    float top2 = bot2 + height2;
    float min_top = (top1 > top2) ? top2 : top1;
    float max_bot = (bot1 > bot2) ? bot1 : bot2;
    float overlapHeight = min_top - max_bot;
    if (overlapHeight > 0) return overlapHeight;
    else return 0;
  }

  void readRec(const float *box, int start) {
    for (int k = 0; k < 4; ++k) {
      p[start + k].x = box[k * 2];
      p[start + k].y = box[k * 2 + 1];
    }
    bool tag = checkClockwise(p[start], p[start + 1], p[start + 2]);
    if (tag)
      std::reverse(p + start, p + start + 4);
  }
  float single_overlap(const float *box1, const float *box2, bool _3D = false) {
    float height1 = -1;
    float height2 = -1;
    float overlapHeight = -1;
    if (_3D){
      height1 = getHeight(box1);
      height2 = getHeight(box2);
      overlapHeight = getOverlapHeight(box1, box2);
    }
    readRec(box2, 0);
    float area2 = getArea(0, 4);
    clear_dq();
    readRec(box1, 4);
    // for (Point pt: p){
    //   std::cout << "x: " << pt.x << " y: " << pt.y << endl;
    // }
    // std::cout << endl;

    for (int z = 0; z < 4; ++z) { //读入直线
      addLine(l[z], p[z].x, p[z].y, p[(z + 1) % 4].x, p[(z + 1) % 4].y);
      addLine(l[z + 4], p[z + 4].x, p[z + 4].y, p[(z + 1) % 4 + 4].x,
              p[(z + 1) % 4 + 4].y);
    }
    // TODO: calculate the area outside the loop
    float area1 = getArea(4, 8);
    // for (Point pt: p){
    //   std::cout << "x: " << pt.x << " y: " << pt.y << endl;
    // }
    // std::cout << endl;
    HalfPlaneIntersect();
    // for (Point pt: p){
    //   std::cout << "x: " << pt.x << " y: " << pt.y << endl;
    // }
    // std::cout << endl;
    
    float iou = getArea(pn_start, pn);
    // std::cout << "old area1" << area1 << endl;
    // std::cout << "old area2" << area2 << endl;
    // std::cout << "old iou" << iou << endl;
    if (_3D){
      assert (height1 > 0 && height2 > 0 && overlapHeight >= 0);
      iou *= overlapHeight;
      area1 *= height1;
      area2 *= height2;
    }
    // std::cout << "height1" << height1 << endl;
    // std::cout << "height2" << height2 << endl;
    // std::cout << "new area1" << area1 << endl;
    // std::cout << "new area2" << area2 << endl;
    // std::cout << "new iou" << iou << endl;
    // std::cout << overlapHeight << endl;
    auto result = iou / (area1 + area2 - iou);
    // std::cout << "result" << result << endl;
    return result;
  }
};

template <typename T, typename Index = long> class BBoxHash {
public:
  BBoxHash(T xScale, T yScale) : mXScale(xScale), mYScale(yScale) {}
  void createBBoxMap(const std::vector<T> &bboxes, int box_dim) {
    for (int i = 0; i < bboxes.size() / box_dim; ++i) {
      auto indexes = getHash(bboxes.data() + i * box_dim);
      for (Index idx : indexes) {
        auto iter = mBBoxMap.find(idx);
        if (iter == mBBoxMap.end()) {
          mBBoxMap[idx] = {i};
        } else {
          mBBoxMap[idx].insert(i);
        }
      }
    }
  }
  std::vector<Index> getHash(const T *bbox) {
    std::vector<Index> indexes;
    constexpr auto min_T = std::numeric_limits<T>::min();
    constexpr auto max_T = std::numeric_limits<T>::max();
    T min_4p[2] = {max_T, max_T};
    T max_4p[2] = {min_T, min_T};
    for (int i = 0; i < 4; ++i) {
      min_4p[0] = std::min(min_4p[0], bbox[i * 2]);
      min_4p[1] = std::min(min_4p[1], bbox[i * 2 + 1]);
      max_4p[0] = std::max(max_4p[0], bbox[i * 2]);
      max_4p[1] = std::max(max_4p[1], bbox[i * 2 + 1]);
    }
    std::vector<int16_t> bbox_2point(4, 0);
    bbox_2point[0] = int16_t(std::floor(min_4p[0] / mXScale));
    bbox_2point[1] = int16_t(std::floor(min_4p[1] / mXScale));
    bbox_2point[2] = int16_t(std::ceil(max_4p[0] / mYScale));
    bbox_2point[3] = int16_t(std::ceil(max_4p[1] / mYScale));
    for (int i = bbox_2point[0]; i < bbox_2point[2]; ++i) {
      for (int j = bbox_2point[1]; j < bbox_2point[3]; ++j) {
        indexes.push_back(i * 100 + j);
      }
    }
    return indexes;
  }
  std::unordered_set<Index> getFilterResult(const T *bbox) {
    std::unordered_set<Index> res;
    auto indexes = getHash(bbox);
    for (Index idx : indexes) {
      auto iter = mBBoxMap.find(idx);
      if (iter != mBBoxMap.end()) {
        res.insert(iter->second.begin(), iter->second.end());
      }
    }
    return res;
  }

protected:
  T mXScale, mYScale;
  std::unordered_map<Index, std::unordered_set<Index>> mBBoxMap;
};

inline const std::vector<std::vector<int>> match_sequence(){
  return {
    {0, 1, 2, 3},
    {2, 3, 0, 1},
    {3, 2, 1, 0},
    {1, 0, 3, 2},
  };
}

template <typename T>
const std::vector<T> change_box_seq(const std::vector<T> &src, const std::vector<int>& seq){
  std::vector<T> box4c(8);
  for (int i = 0; i < 4; ++i){
    box4c[i * 2] = src[seq[i] * 2];
    box4c[i * 2 + 1] = src[seq[i] * 2 + 1];
  }
  return box4c;
}

template <typename T>
int match_4c_box(const std::vector<T> &target, const std::vector<T> &src, const std::vector<std::vector<int>>& match_seq){
  std::vector<T> box4c(8);
  std::vector<T> scores;
  for (auto &seq : match_seq){
    T sum_value = 0;
    for (int i = 0; i < 4; ++i){
      sum_value += std::pow(std::abs(target[seq[i] * 2] - src[i * 2]), 2);
      sum_value += std::pow(std::abs(target[seq[i] * 2 + 1] - src[i * 2 + 1]), 2);
    }
    scores.push_back(sum_value);
  }
  auto min_iter = std::min_element(scores.begin(), scores.end());
  int min_idx = min_iter - scores.begin();
  return min_idx;
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> wvnms_4c(std::vector<T> &dets,
                           std::vector<T> &boxvars, std::vector<int> &orders,
                           T thresh, T thresh_vote, T delta) {
  // dets: [N, 8(boxes) + 1(yaw) + 2(score, cls) + 6(vars)]
  // box6s: [N, 6] raw lidar output boxes
  auto bboxHash = BBoxHash<T, int>(100, 100);
  auto overlap_calc = OverlapChecker();

  const T *dets_data = dets.data();
  const T *vars_data = boxvars.data();

  auto ndets = orders.size();
  if (ndets == 0) {
    return {};
  }
  auto box_ndim = boxvars.size() / ndets;
  auto match_seqs = match_sequence();
  auto dets_ndim = dets.size() / ndets;
  bboxHash.createBBoxMap(dets, dets_ndim);
  std::vector<int> suppressed(ndets, 0);
  std::vector<T> keep_dets;
  std::vector<int> keep_inds;
  std::vector<int> neighborhoods, new_neighborhoods;
  std::vector<T> yaw(ndets);
  std::vector<T> ovrs, new_ovrs;
  std::vector<T> scores(ndets);
  std::vector<T> avg(box_ndim);
  std::vector<T> sum1(box_ndim);
  std::vector<T> sum2(box_ndim);
  std::vector<T> sum3(box_ndim);
  for (int i = 0; i < ndets; ++i) {
    yaw[i] = dets[i * dets_ndim + 8];
    scores[i] = dets[i * dets_ndim + 9];
  }

  int i, j;
  T ovr, score_sum, median_yaw, score_max, p, tmp0;
  int score_idx, det_idx;
  for (int _i = 0; _i < ndets; ++_i) {
    i = orders[_i];
    if (suppressed[i] == 1)
      continue;
    ovrs.clear();
    neighborhoods.clear();
    new_neighborhoods.clear();
    auto filter_indexes = bboxHash.getFilterResult(dets_data + dets_ndim * i);
    for (int _j = _i + 1; _j < ndets; ++_j) {
      j = orders[_j];
      if (suppressed[j] == 1)
        continue;
      if (filter_indexes.find(j) == filter_indexes.end())
        continue;
      ovr = overlap_calc.single_overlap(dets_data + i * dets_ndim,
                                        dets_data + j * dets_ndim);
      if (ovr >= thresh)
        suppressed[j] = 1;
      if (ovr > thresh_vote) {
        neighborhoods.push_back(j);
        ovrs.push_back(ovr);
      }
    }
    auto current = std::vector<T>(dets_data + dets_ndim * i, dets_data + dets_ndim * i + 8);
    // lidar filtering
    neighborhoods.push_back(i);
    ovrs.push_back(1);
    // voting
    for (int k = 0; k < sum1.size(); ++k) {
      sum1[k] = T(0);
      sum2[k] = T(0);
      sum3[k] = T(0);
    }
    for (size_t l = 0; l < neighborhoods.size(); ++l) {
      auto neigh_idx = neighborhoods[l] * dets_ndim;
      auto neigh_var_idx = neighborhoods[l] * box_ndim;

      p = std::exp(-std::pow((T(1) - ovrs[l]), 2) / delta);
      auto neigh = std::vector<T>(dets_data + neigh_idx, dets_data + neigh_idx + 8);
      auto neigh_var = std::vector<T>(vars_data + neigh_var_idx, vars_data + neigh_var_idx + 8);

      auto matched_seq_idx = match_4c_box(neigh, current, match_seqs);
      auto box4c_ = change_box_seq(neigh, match_seqs[matched_seq_idx]);
      auto box4c_var_ = change_box_seq(neigh_var, match_seqs[matched_seq_idx]);

      for (int k = 0; k < box_ndim; ++k) {
        // det_idx = neigh_idx + k;
        tmp0 = p / box4c_var_[k];
        //tmp0 = p;
        sum1[k] += p * box4c_[k];
        sum2[k] += tmp0;
        sum3[k] += p;
      }
    }
    for (int k = 0; k < sum1.size(); ++k) {
      keep_dets.push_back(sum1[k] / sum3[k]);
    }
    keep_dets.push_back(yaw[i]);
    keep_dets.push_back(scores[i]);
    keep_dets.push_back(dets[i * dets_ndim + 10]);
    for (int k = 0; k < box_ndim; ++k) {
      keep_dets.push_back(sum3[k] / sum2[k]);
    }
    keep_inds.push_back(i);
  }
  return std::tuple<std::vector<T>, std::vector<int>>{keep_dets, keep_inds};
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> wnms_4c(std::vector<T> &dets,
                           std::vector<int> &orders,
                           T thresh, T thresh_vote, bool _3D = false, int hash_scale = 100) {
  // dets: [N, 8(boxes) + 1(yaw) + 2(bottom, height) + 1(score)]
  // box6s: [N, 6] raw lidar output boxes
  auto bboxHash = BBoxHash<T, int>(hash_scale, hash_scale);
  auto overlap_calc = OverlapChecker();

  const T *dets_data = dets.data();

  auto ndets = orders.size();
  if (ndets == 0) {
    return {};
  }
  auto box_ndim = 11;
  auto match_seqs = match_sequence();
  auto dets_ndim = dets.size() / ndets;
  bboxHash.createBBoxMap(dets, dets_ndim);
  std::vector<int> suppressed(ndets, 0);
  std::vector<T> keep_dets;
  std::vector<int> keep_inds;
  std::vector<int> neighborhoods, new_neighborhoods;
  std::vector<T> yaw(ndets);
  std::vector<T> ovrs, new_ovrs;
  std::vector<T> scores(ndets);
  std::vector<T> avg(box_ndim);
  std::vector<T> sum1(box_ndim);
  std::vector<T> sum2(box_ndim);
  std::vector<T> sum3(box_ndim);
  std::vector<T> neighboryaw;
  for (int i = 0; i < ndets; ++i) {
    yaw[i] = dets[i * dets_ndim + 8];
    scores[i] = dets[i * dets_ndim + 11];
  }

  int i, j;
  T ovr, score_sum, median_yaw, score_max, p, tmp0;
  int score_idx, det_idx;
  for (int _i = 0; _i < ndets; ++_i) {
    i = orders[_i];
    if (suppressed[i] == 1)
      continue;
    ovrs.clear();
    neighborhoods.clear();
    new_neighborhoods.clear();
    // lidar filtering
    neighborhoods.push_back(i);
    ovrs.push_back(1);
    auto filter_indexes = bboxHash.getFilterResult(dets_data + dets_ndim * i);
    for (int _j = _i + 1; _j < ndets; ++_j) {
      j = orders[_j];
      if (suppressed[j] == 1)
        continue;
      if (filter_indexes.find(j) == filter_indexes.end())
        continue;
      ovr = overlap_calc.single_overlap(dets_data + i * dets_ndim,
                                        dets_data + j * dets_ndim,
                                        _3D);
      if (ovr >= thresh)
        suppressed[j] = 1;
      if (ovr > thresh_vote) {
        neighborhoods.push_back(j);
        ovrs.push_back(ovr);
      }
    }
    auto current = std::vector<T>(dets_data + dets_ndim * i, dets_data + dets_ndim * i + 8);
    
    // voting
    for (int k = 0; k < sum1.size(); ++k) {
      sum1[k] = T(0);
      sum2[k] = T(0);
      sum3[k] = T(0);
    }
    
    neighboryaw.clear();
    for (size_t l = 0; l < neighborhoods.size(); ++l) {
      neighboryaw.push_back(yaw[neighborhoods[l]]);
    }
    if (neighborhoods.size() <= 2){
      median_yaw = yaw[i];
    }
    else {
      if (neighborhoods.size() % 2 == 0){
        neighboryaw.push_back(yaw[i]);
      }
      std::sort(neighboryaw.begin(), neighboryaw.end());
      median_yaw = neighboryaw[neighboryaw.size() / 2];
    }
    for (size_t l = 0; l < neighborhoods.size(); ++l) {
      if (std::fmod(std::abs(yaw[neighborhoods[l]] - median_yaw), float(2 * 3.1415926))>= 0.3) {
        //printf("median: %f, filtered %d, %f\n", median_yaw, neighborhoods[l], yaw[neighborhoods[l]]);
        continue;
      }
        
      auto neigh_idx = neighborhoods[l] * dets_ndim;

      auto neigh = std::vector<T>(dets_data + neigh_idx, dets_data + neigh_idx + 8);
      p = scores[neighborhoods[l]];//std::exp(-std::pow((T(1) - ovrs[l]), 2) / delta);

      auto matched_seq_idx = match_4c_box(neigh, current, match_seqs);
      auto box4c_ = neigh;//change_box_seq(neigh, match_seqs[matched_seq_idx]);
      //printf("%d %d\n", i, neighborhoods[l]);
      for (int k = 0; k < 8; ++k) {
        // det_idx = neigh_idx + k;
        //tmp0 = p;
        sum1[k] += p * box4c_[k];
        //printf("%f ", box4c_[k]);
        sum3[k] += p;
      }
      
      for (int k=8; k < box_ndim; k++) {
        sum1[k] += p * dets_data[neigh_idx+k];
        //printf("%f ", dets_data[neigh_idx+k]);
        sum3[k] += p;
      }
      //printf("%f %f\n", p, ovrs[l]);
    }
    for (int k = 0; k < sum1.size(); ++k) {
      keep_dets.push_back(sum1[k] / sum3[k]);
    }
    keep_dets.push_back(scores[i]);
    keep_inds.push_back(i);
  }
  return std::tuple<std::vector<T>, std::vector<int>>{keep_dets, keep_inds};
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> _wnms_csa(std::vector<T> &dets,
                           std::vector<int> &orders,
                           T thresh, T thresh_vote) {
  // dets: [N, 8(boxes) + 1(yaw) + 2(bottom, height) + 1(score)]
  // box6s: [N, 6] raw lidar output boxes
  auto bboxHash = BBoxHash<T, int>(100, 100);
  auto overlap_calc = OverlapChecker();

  const T *dets_data = dets.data();

  auto ndets = orders.size();
  if (ndets == 0) {
    return {};
  }
  auto box_ndim = 11;
  auto match_seqs = match_sequence();
  auto dets_ndim = dets.size() / ndets;
  bboxHash.createBBoxMap(dets, dets_ndim);
  std::vector<int> suppressed(ndets, 0);
  std::vector<T> keep_dets;
  std::vector<int> keep_inds;
  std::vector<int> neighborhoods, new_neighborhoods;
  std::vector<T> yaw(ndets);
  std::vector<T> ovrs, new_ovrs;
  std::vector<T> scores(ndets);
  std::vector<T> avg(box_ndim);
  std::vector<T> sum1(box_ndim);
  std::vector<T> sum2(box_ndim);
  std::vector<T> sum3(box_ndim);
  std::vector<T> neighboryaw;
  for (int i = 0; i < ndets; ++i) {
    yaw[i] = dets[i * dets_ndim + 8];
    scores[i] = dets[i * dets_ndim + 11];
  }

  int i, j;
  T ovr, score_sum, median_yaw, score_max, p, tmp0;
  int score_idx, det_idx;
  for (int _i = 0; _i < ndets; ++_i) {
    i = orders[_i];
    if (suppressed[i] == 1)
      continue;
    ovrs.clear();
    neighborhoods.clear();
    new_neighborhoods.clear();
    // lidar filtering
    neighborhoods.push_back(i);
    ovrs.push_back(1);
    auto filter_indexes = bboxHash.getFilterResult(dets_data + dets_ndim * i);
    for (int _j = _i + 1; _j < ndets; ++_j) {
      j = orders[_j];
      if (suppressed[j] == 1)
        continue;
      if (filter_indexes.find(j) == filter_indexes.end())
        continue;
      ovr = overlap_calc.single_overlap(dets_data + i * dets_ndim,
                                        dets_data + j * dets_ndim);
      if (ovr >= thresh)
        suppressed[j] = 1;
      if (ovr > thresh_vote) {
        neighborhoods.push_back(j);
        ovrs.push_back(ovr);
      }
    }
    auto current = std::vector<T>(dets_data + dets_ndim * i, dets_data + dets_ndim * i + 8);
    
    // voting
    for (int k = 0; k < sum1.size(); ++k) {
      sum1[k] = T(0);
      sum2[k] = T(0);
      sum3[k] = T(0);
    }
    
    neighboryaw.clear();
    for (size_t l = 0; l < neighborhoods.size(); ++l) {
      neighboryaw.push_back(yaw[neighborhoods[l]]);
    }
    if (neighborhoods.size() <= 2){
      median_yaw = yaw[i];
    }
    else {
      if (neighborhoods.size() % 2 == 0){
        neighboryaw.push_back(yaw[i]);
      }
      std::sort(neighboryaw.begin(), neighboryaw.end());
      median_yaw = neighboryaw[neighboryaw.size() / 2];
    }
    for (size_t l = 0; l < neighborhoods.size(); ++l) {
      if (std::fmod(std::abs(yaw[neighborhoods[l]] - median_yaw), float(2 * 3.1415926))>= 0.3) {
        //printf("median: %f, filtered %d, %f\n", median_yaw, neighborhoods[l], yaw[neighborhoods[l]]);
        continue;
      }
        
      auto neigh_idx = neighborhoods[l] * dets_ndim;

      auto neigh = std::vector<T>(dets_data + neigh_idx, dets_data + neigh_idx + 8);
      p = scores[neighborhoods[l]];//std::exp(-std::pow((T(1) - ovrs[l]), 2) / delta);

      auto matched_seq_idx = match_4c_box(neigh, current, match_seqs);
      auto box4c_ = neigh;//change_box_seq(neigh, match_seqs[matched_seq_idx]);
      //printf("%d %d\n", i, neighborhoods[l]);
      for (int k = 0; k < 8; ++k) {
        // det_idx = neigh_idx + k;
        //tmp0 = p;
        sum1[k] += p * box4c_[k];
        //printf("%f ", box4c_[k]);
        sum3[k] += p;
      }
      
      for (int k=8; k < box_ndim; k++) {
        sum1[k] += p * dets_data[neigh_idx+k];
        //printf("%f ", dets_data[neigh_idx+k]);
        sum3[k] += p;
      }
      //printf("%f %f\n", p, ovrs[l]);
    }
    for (int k = 0; k < sum1.size(); ++k) {
      keep_dets.push_back(sum1[k] / sum3[k]);
    }
    keep_dets.push_back(scores[i]);
    keep_inds.push_back(i);
  }
  return std::tuple<std::vector<T>, std::vector<int>>{keep_dets, keep_inds};
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> nms_4c(std::vector<T> &dets,
                           std::vector<int> &orders,
                           T thresh) {
  // dets: [N, 8(boxes) + 1(yaw) + 2(bottom, height) + 1(score)]
  // box6s: [N, 6] raw lidar output boxes
  auto bboxHash = BBoxHash<T, int>(100, 100);
  auto overlap_calc = OverlapChecker();

  const T *dets_data = dets.data();

  auto ndets = orders.size();
  if (ndets == 0) {
    return {};
  }
  auto box_ndim = 11;
  auto match_seqs = match_sequence();
  auto dets_ndim = dets.size() / ndets;
  bboxHash.createBBoxMap(dets, dets_ndim);
  std::vector<int> suppressed(ndets, 0);
  std::vector<T> keep_dets;
  std::vector<int> keep_inds;
  std::vector<T> yaw(ndets);
  std::vector<T> scores(ndets);
  for (int i = 0; i < ndets; ++i) {
    yaw[i] = dets[i * dets_ndim + 8];
    scores[i] = dets[i * dets_ndim + 11];
  }

  int i, j;
  T ovr, score_sum, score_max, p, tmp0;
  int score_idx, det_idx;
  for (int _i = 0; _i < ndets; ++_i) {
    i = orders[_i];
    if (suppressed[i] == 1)
      continue;
    auto filter_indexes = bboxHash.getFilterResult(dets_data + dets_ndim * i);
    for (int _j = _i + 1; _j < ndets; ++_j) {
      j = orders[_j];
      if (suppressed[j] == 1)
        continue;
      if (filter_indexes.find(j) == filter_indexes.end())
        continue;
      ovr = overlap_calc.single_overlap(dets_data + i * dets_ndim,
                                        dets_data + j * dets_ndim);
      if (ovr >= thresh)
        suppressed[j] = 1;
    }
    for (int k = 0; k < box_ndim; ++k) {
      keep_dets.push_back(dets[i * dets_ndim + k]);
    }
    keep_dets.push_back(scores[i]);
    keep_inds.push_back(i);
  }
  return std::tuple<std::vector<T>, std::vector<int>>{keep_dets, keep_inds};
}

} // namespace trtplus

template <typename T> std::vector<T> arrayT2Vector(py::array_t<T> arr) {
  std::vector<T> data(arr.data(), arr.data() + arr.size());
  return data;
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> point4_wvnms_4c(py::array_t<T> &dets,
                                 py::array_t<T> &boxvars,
                                 py::array_t<int> &orders, T thresh,
                                 T thresh_vote, T delta) {
  auto dets_v = arrayT2Vector<T>(dets);
  auto boxvars_v = arrayT2Vector<T>(boxvars);
  auto orders_v = arrayT2Vector<int>(orders);
  return trtplus::wvnms_4c(dets_v, boxvars_v, orders_v, thresh,
                              thresh_vote, delta);
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> point4_wnms_4c(py::array_t<T> &dets,
                                 T thresh,
                                 T thresh_vote,
                                 bool _3D = false,
                                 int hash_scale = 100) {
  auto dets_v = arrayT2Vector<T>(dets);
  int dets_ndim = 12;
  std::vector<int> orders_v(dets_v.size()/dets_ndim);
  std::iota(orders_v.begin(), orders_v.end(), 0); //Initializing
  sort(orders_v.begin(), orders_v.end(), [&](int i,int j){
    return dets_v[i * dets_ndim + 11]>dets_v[j * dets_ndim + 11];} );
  return trtplus::wnms_4c(dets_v, orders_v, thresh, thresh_vote, _3D, hash_scale);
}

template <typename T>
std::tuple<std::vector<T>, std::vector<int>> point4_nms_4c(py::array_t<T> &dets, T thresh) {
  auto dets_v = arrayT2Vector<T>(dets);
  int dets_ndim = 12;
  std::vector<int> orders_v(dets_v.size()/dets_ndim);
  std::iota(orders_v.begin(), orders_v.end(), 0); //Initializing
  sort(orders_v.begin(), orders_v.end(), [&](int i,int j){
    return dets_v[i * dets_ndim + 11]>dets_v[j * dets_ndim + 11];} );
  return trtplus::nms_4c(dets_v, orders_v, thresh);
}