// The Photo Sphere pipeline, plain C++ on OpenCV (no Apple types), following Google Camera 4.1's liblightcycle stage
// by stage (the constants are theirs where they're known; see README.md for the map from their classes to this file).
// Used by PanoStitcher.mm on iOS and by tools/harness.cpp on a Linux runner.
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace pano {

using cv::Mat;
using cv::Matx33d;
using cv::Point2d;
using cv::Vec3d;
using Progress = std::function<void(double, const char *)>;

struct StitchResult {
    std::string path;
    int fullW = 0, fullH = 0, cropW = 0, cropH = 0, left = 0, top = 0;
    bool full = false;
    int images = 0, featurePairs = 0, directPairs = 0;
    double focal = 0, seconds = 0;
    std::vector<Matx33d> R;
    std::vector<int> dropped;        // photos not rendered: nothing held them (photos without a gyro only)
    std::string notes;
    std::string pairReport;          // one line per pair after the adjustment: i j kind inliers median_px
};

struct Bootstrap {
    std::vector<int> placed;
    std::vector<Matx33d> R;
    int edges = 0, rejected = 0, secondChance = 0, root = -1;
    std::string log;                 // what the second chance found for each photo it tried
};

struct Synth {
    std::vector<std::string> images;
    std::vector<Matx33d> priors, truth;
    double focalPx = 0;
    int width = 0, height = 0;
    std::string pano;
};

struct Metrics {
    double rotationRmsDeg = 99, rotationMaxDeg = 99, psnr = 0;
    bool pass = false;
    std::vector<double> perPhotoDeg;
};

inline void makeDirs(const std::string &dir) {
    std::string cur;
    for (size_t i = 0; i <= dir.size(); i++) {
        if (i == dir.size() || dir[i] == '/') {
            if (!cur.empty()) mkdir(cur.c_str(), 0755);
        }
        if (i < dir.size()) cur += dir[i];
    }
}


constexpr double kPi = CV_PI;
constexpr int kMatchWidth = 1600;        // the photos are matched at this size (GCam: image_match_width)
constexpr double kHuberPx = 35.0;        // the match residual's Huber scale, in match pixels (GCam: HuberLoss(35))
constexpr double kRansacInlierDeg = 1.1; // a match is an inlier of a pairwise rotation within this (GCam: 1.1 deg)
constexpr int kRansacMin = 550, kRansacMax = 5000, kRansacEnough = 150;
constexpr int kKeepMatches = 35;         // inliers kept per pair for the global adjustment (GCam: 35, stratified)
constexpr double kRatio = 0.8;           // Lowe's ratio (GCam: 0.64 on squared distances)
constexpr double kAxesNear = 0.9;        // two photos are candidates when their axes are within this times the FOV
constexpr double kLatGateDeg = 9.0;
constexpr double kCycleDeg = 4.0;        // without a gyro, a pair is believed when a third photo closes the triangle within this
constexpr int kStrongEdge = 40;          // ...or when it has this many inliers on its own
constexpr double kSupportPx = 25.0;
constexpr double kFocalPriorPx = 3000.0; // the focal stays near the lens's (exact on an iPhone): this many match pixels per unit of ln(f / f0)
constexpr double kPriorAgreeDeg = 10.0;  // without a gyro, a pair must agree with the verified placement within this      // ...and a photo stays when one of its matches holds within this after the adjustment
constexpr double kGammaMax = 2.5;        // a photo's gamma at most (GCam: 1.75; an iPhone's auto exposure spans more)      // and a match's two rays must lie within this latitude band (GCam: 6 deg bins +-1)

double deg(double r) { return r * 180.0 / kPi; }
double rad(double d) { return d * kPi / 180.0; }
double clamp1(double v) { return std::max(-1.0, std::min(1.0, v)); }

/// Rodrigues: the rotation of angle |w| about w.
Matx33d expSO3(const Vec3d &w) {
    double a = cv::norm(w);
    if (a < 1e-12) return Matx33d::eye();
    Vec3d k = w * (1.0 / a);
    Matx33d K(0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0);
    return Matx33d::eye() + K * std::sin(a) + K * K * (1 - std::cos(a));
}

double angleOf(const Matx33d &R) { return std::acos(clamp1((cv::trace(R) - 1.0) / 2.0)); }

Vec3d unit(const Vec3d &v) {
    double n = cv::norm(v);
    return n > 0 ? v * (1.0 / n) : v;
}

double angleBetween(const Vec3d &a, const Vec3d &b) { return std::acos(clamp1(unit(a).dot(unit(b)))); }

Vec3d forwardOf(const Matx33d &R) { return Vec3d(R(0, 2), R(1, 2), R(2, 2)); }
double pitchOf(const Matx33d &R) { return std::asin(clamp1(-R(1, 2))); }
double yawOf(const Matx33d &R) { return std::atan2(R(0, 2), R(2, 2)); }
double rollOf(const Matx33d &R) { return std::asin(clamp1(R(1, 0))); }  // the right vector's downward component

Matx33d rotY(double a) {
    double c = std::cos(a), s = std::sin(a);
    return Matx33d(c, 0, s, 0, 1, 0, -s, 0, c);
}

/// world-from-camera for a camera looking along (yaw, pitch) with no roll (world: y down).
Matx33d lookAt(double yaw, double pitch, double roll = 0) {
    Vec3d f(std::cos(pitch) * std::sin(yaw), -std::sin(pitch), std::cos(pitch) * std::cos(yaw));
    Vec3d right = f.cross(Vec3d(0, -1, 0));
    if (cv::norm(right) < 1e-6) right = Vec3d(std::cos(yaw), 0, -std::sin(yaw));
    right = unit(right);
    Vec3d down = unit(f.cross(right));
    Matx33d R(right[0], down[0], f[0], right[1], down[1], f[1], right[2], down[2], f[2]);
    return R * expSO3(Vec3d(0, 0, roll));
}

/// The equirectangular canvas: W x W/2, longitude -pi..pi left to right, latitude +pi/2 at the top.
struct Equirect {
    int W, H;
    explicit Equirect(int w) : W(w), H(w / 2) {}
    Vec3d ray(double u, double v) const {
        double lam = (u + 0.5) / W * 2 * kPi - kPi, phi = kPi / 2 - (v + 0.5) / H * kPi, cp = std::cos(phi);
        return Vec3d(cp * std::sin(lam), -std::sin(phi), cp * std::cos(lam));
    }
    Point2d pixel(const Vec3d &d) const {
        double lam = std::atan2(d[0], d[2]), phi = std::atan2(-d[1], std::hypot(d[0], d[2]));
        return Point2d((lam + kPi) / (2 * kPi) * W - 0.5, (kPi / 2 - phi) / kPi * H - 0.5);
    }
};

/// The rotation R with R*a[k] ~ b[k] (Kabsch), over the indices given (all when none).
Matx33d kabsch(const std::vector<Vec3d> &a, const std::vector<Vec3d> &b, const std::vector<int> *idx = nullptr) {
    Mat H = Mat::zeros(3, 3, CV_64F);
    auto add = [&](int k) {
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) H.at<double>(r, c) += a[k][r] * b[k][c];
    };
    if (idx)
        for (int k : *idx) add(k);
    else
        for (int k = 0; k < (int)a.size(); k++) add(k);
    cv::SVD svd(H, cv::SVD::FULL_UV);
    Mat V = svd.vt.t(), U = svd.u;
    Mat D = Mat::eye(3, 3, CV_64F);
    D.at<double>(2, 2) = cv::determinant(V * U.t()) < 0 ? -1 : 1;
    Mat R = V * D * U.t();
    return Matx33d(R);
}

// ---------------------------------------------------------------- features and pairwise rotations

/// A photo shrunk and normalised for the direct alignment (GCam's AlignmentTracker normalised its levels the same way:
/// 128 + 128 (I - mean) / (2 sd), so exposure doesn't matter).
struct Small {
    Mat g, gx, gy;
    double f = 1, cx = 0, cy = 0;
};

Small makeSmall(const Mat &gray, double f, double scale) {
    Small s;
    cv::resize(gray, s.g, cv::Size(), scale, scale, cv::INTER_AREA);
    s.g.convertTo(s.g, CV_32F);
    cv::Scalar mu, sd;
    cv::meanStdDev(s.g, mu, sd);
    s.g = (s.g - mu[0]) * (128.0 / (2 * std::max(1.0, sd[0]))) + 128;
    cv::GaussianBlur(s.g, s.g, cv::Size(3, 3), 0);
    cv::Sobel(s.g, s.gx, CV_32F, 1, 0, 3, 1 / 8.0);
    cv::Sobel(s.g, s.gy, CV_32F, 0, 1, 3, 1 / 8.0);
    s.f = f * scale;
    s.cx = (s.g.cols - 1) / 2.0;
    s.cy = (s.g.rows - 1) / 2.0;
    return s;
}

float bilinear(const Mat &g, double x, double y) {
    int x0 = (int)x, y0 = (int)y;
    double fx = x - x0, fy = y - y0;
    const float *r0 = g.ptr<float>(y0), *r1 = g.ptr<float>(y0 + 1);
    return (float)((r0[x0] * (1 - fx) + r0[x0 + 1] * fx) * (1 - fy) + (r1[x0] * (1 - fx) + r1[x0 + 1] * fx) * fy);
}

/// Refines the rotation taking camera a's rays to camera b's directly on the pixels (GCam's CameraRotationModel /
/// GlobalFlowSolver, Gauss-Newton on brightness differences): the pixels of a with some gradient are transferred into
/// b, and the brightness difference, with b's gradients, is solved for a small rotation (and an affine brightness
/// change), repeated until it settles. true when the fit improved clearly; before/after: the mean |difference|.
bool refineDirect(const Small &a, const Small &b, Matx33d &Rji, double &before, double &after, int &points) {
    std::vector<cv::Point> pts;
    for (int y = 2; y < a.g.rows - 2; y += 2)
        for (int x = 2; x < a.g.cols - 2; x += 2)
            if (std::fabs(a.gx.at<float>(y, x)) + std::fabs(a.gy.at<float>(y, x)) > 3) pts.push_back(cv::Point(x, y));
    if (pts.size() > 3000) {
        std::mt19937 rng(3);
        std::shuffle(pts.begin(), pts.end(), rng);
        pts.resize(3000);
    }
    Matx33d R0 = Rji;
    double alpha = 1, beta = 0, lastStep = 1;
    bool conditioned = false;
    before = after = 0;
    points = 0;
    for (int it = 0; it < 25; it++) {
        Mat A = Mat::zeros(5, 5, CV_64F), rhs = Mat::zeros(5, 1, CV_64F);
        double sumAbs = 0;
        int used = 0;
        for (auto &p : pts) {
            Vec3d d = Rji * Vec3d((p.x - a.cx) / a.f, (p.y - a.cy) / a.f, 1);
            if (d[2] <= 0.1) continue;
            double qx = b.f * d[0] / d[2] + b.cx, qy = b.f * d[1] / d[2] + b.cy;
            if (qx < 1 || qy < 1 || qx >= b.g.cols - 2 || qy >= b.g.rows - 2) continue;
            double Ib = bilinear(b.g, qx, qy), Ia = a.g.at<float>(p.y, p.x);
            double gbx = bilinear(b.gx, qx, qy), gby = bilinear(b.gy, qx, qy);
            double e = Ib - (alpha * Ia + beta);
            // dq/dw: q = f d.xy/d.z + c, d = exp(w) d0 ~ d0 + w x d0 = d0 - [d0]x w
            double s = b.f / d[2];
            double Jd[2][3] = {{s, 0, -s * d[0] / d[2]}, {0, s, -s * d[1] / d[2]}};
            double K[3][3] = {{0, -d[2], d[1]}, {d[2], 0, -d[0]}, {-d[1], d[0], 0}};  // [d0]x
            double row[5];
            for (int k = 0; k < 3; k++) {
                double jx = 0, jy = 0;
                for (int m = 0; m < 3; m++) {
                    jx += Jd[0][m] * -K[m][k];
                    jy += Jd[1][m] * -K[m][k];
                }
                row[k] = gbx * jx + gby * jy;
            }
            row[3] = -Ia;
            row[4] = -1;
            double w = std::fabs(e) <= 20 ? 1.0 : 20 / std::fabs(e);  // (Huber)
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) A.at<double>(r, c) += w * row[r] * row[c];
                rhs.at<double>(r) += -w * row[r] * e;
            }
            sumAbs += std::fabs(e);
            used++;
        }
        if (used < 150) return false;
        if (it == 0) before = sumAbs / used;
        after = sumAbs / used;
        points = used;
        Mat ev;
        cv::eigen(A(cv::Rect(0, 0, 3, 3)), ev);  // (the rotation's conditioning: a plain overlap constrains it poorly)
        conditioned = ev.at<double>(2) > 1e-4 * ev.at<double>(0);
        for (int r = 0; r < 5; r++) A.at<double>(r, r) += 1e-6;
        Mat x;
        if (!cv::solve(A, rhs, x, cv::DECOMP_CHOLESKY)) return false;
        Vec3d w(x.at<double>(0), x.at<double>(1), x.at<double>(2));
        Rji = expSO3(w) * Rji;
        alpha += x.at<double>(3);
        beta += x.at<double>(4);
        lastStep = cv::norm(w);
        if (lastStep < 1e-5) break;
    }
    return conditioned && lastStep < 1e-3 && after <= before && angleOf(Rji * R0.t()) < rad(6);
}

struct LineSeg {
    Point2d a, b;
    double len = 0;
};

struct LineMatch {
    int i = 0, j = 0;
    Point2d ai, bi;  // the segment in i
    Point2d aj, bj;  // the matched segment in j
    double w = 0.5;
};

/// The straight edges of a photo (walls, frames, furniture), the longest first, at match scale.
inline std::vector<LineSeg> detectLines(const Mat &gray) {
    std::vector<LineSeg> out;
    try {
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
        std::vector<cv::Vec4f> ls;
        lsd->detect(gray, ls);
        for (auto &l : ls) {
            LineSeg s;
            s.a = Point2d(l[0], l[1]);
            s.b = Point2d(l[2], l[3]);
            s.len = cv::norm(s.a - s.b);
            if (s.len >= 40) out.push_back(s);
        }
    } catch (const std::exception &) {
    }
    std::sort(out.begin(), out.end(), [](const LineSeg &x, const LineSeg &y) { return x.len > y.len; });
    if (out.size() > 300) out.resize(300);
    return out;
}

inline double distToLine(const Point2d &p, const Point2d &a, const Point2d &b) {
    Point2d d = b - a, v = p - a;
    double len = cv::norm(d);
    return len > 1e-9 ? std::fabs(v.x * d.y - v.y * d.x) / len : cv::norm(v);
}

/// Segments of photo i matched to segments of photo j under the current rotations: the endpoints transferred, the
/// nearest segment of about the same direction that they lie close to and overlap, mutual best (GCam: LineFeatureMatcher).
inline void matchLines(int i, int j, const std::vector<LineSeg> &li, const std::vector<LineSeg> &lj, const Matx33d &Rji, double f, double cxi, double cyi,
                       double cxj, double cyj, int wj, int hj, std::vector<LineMatch> &out, int cap = 60) {
    auto transfer = [&](const Point2d &p, Point2d &q) {
        Vec3d d = Rji * Vec3d((p.x - cxi) / f, (p.y - cyi) / f, 1);
        if (d[2] <= 0.1) return false;
        q = Point2d(f * d[0] / d[2] + cxj, f * d[1] / d[2] + cyj);
        return q.x > -0.2 * wj && q.y > -0.2 * hj && q.x < 1.2 * wj && q.y < 1.2 * hj;
    };
    std::vector<int> bestFor(li.size(), -1), bestBack(lj.size(), -1);
    std::vector<double> bestScore(li.size(), 1e9), bestBackScore(lj.size(), 1e9), secondScore(li.size(), 1e9);
    for (size_t s = 0; s < li.size(); s++) {
        Point2d qa, qb;
        if (!transfer(li[s].a, qa) || !transfer(li[s].b, qb)) continue;
        Point2d dq = qb - qa;
        double lq = cv::norm(dq);
        if (lq < 20) continue;
        for (size_t t = 0; t < lj.size(); t++) {
            Point2d dt = lj[t].b - lj[t].a;
            double cosang = std::fabs(dq.dot(dt)) / (lq * lj[t].len);
            if (cosang < std::cos(rad(8))) continue;
            double d1 = distToLine(qa, lj[t].a, lj[t].b), d2 = distToLine(qb, lj[t].a, lj[t].b);
            if (d1 > 30 || d2 > 30) continue;
            // overlap along the line
            Point2d u = dt * (1.0 / lj[t].len);
            double pa = (qa - lj[t].a).dot(u), pb = (qb - lj[t].a).dot(u);
            double lo = std::max(0.0, std::min(pa, pb)), hi = std::min(lj[t].len, std::max(pa, pb));
            if (hi - lo < 0.3 * std::min(lq, lj[t].len)) continue;
            double score = d1 + d2;
            if (score < bestScore[s]) {
                secondScore[s] = bestScore[s];
                bestScore[s] = score;
                bestFor[s] = (int)t;
            } else if (score < secondScore[s]) {
                secondScore[s] = score;
            }
            if (score < bestBackScore[t]) {
                bestBackScore[t] = score;
                bestBack[t] = (int)s;
            }
        }
    }
    std::vector<std::pair<double, int>> order;
    for (size_t s = 0; s < li.size(); s++)
        if (bestFor[s] >= 0 && bestBack[bestFor[s]] == (int)s && bestScore[s] < 0.6 * secondScore[s]) order.push_back({bestScore[s], (int)s});
    std::sort(order.begin(), order.end());
    for (size_t k = 0; k < order.size() && (int)k < cap; k++) {
        int s = order[k].second, t = bestFor[s];
        LineMatch m;
        m.i = i;
        m.j = j;
        m.ai = li[s].a;
        m.bi = li[s].b;
        m.aj = lj[t].a;
        m.bj = lj[t].b;
        out.push_back(m);
    }
}

struct View {
    int w = 0, h = 0;                // at match scale
    double scale = 1;                // match pixels per photo pixel
    double cx = 0, cy = 0;           // its centre at match scale (a landscape photo among portraits has its own)
    Small s16, s8, s4;               // for the direct alignment: 1/16, 1/8 and 1/4 of match scale
    std::vector<LineSeg> lines;      // its straight edges
    std::vector<cv::KeyPoint> kps;
    Mat desc;
    std::vector<Vec3d> rays;         // camera-frame unit rays of the keypoints
    std::vector<double> lat;         // their latitude under the sensor prior
    Matx33d Rp;                      // the sensor prior, world-from-camera
};

struct PairMatch {
    int i = 0, j = 0;
    std::vector<Point2d> pi, pj;     // match-scale pixels
    std::vector<double> wk;          // per-match weights (the ground's: see groundWeight)
    double w = 1;
    bool sensor = false, direct = false;
    int inliers = 0, putative = 0;
};

Vec3d rayOf(double x, double y, double f, double cx, double cy) { return unit(Vec3d((x - cx) / f, (y - cy) / f, 1)); }

/// A 2-point RANSAC for the rotation taking rays a onto rays b (GCam's ComputeRotation), then a refit on the inliers.
bool ransacRotation(const std::vector<Vec3d> &a, const std::vector<Vec3d> &b, std::mt19937 &rng, Matx33d &R, std::vector<int> &inl) {
    int n = (int)a.size();
    if (n < 2) return false;
    double cosTol = std::cos(rad(kRansacInlierDeg)), cos5 = std::cos(rad(5));
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<int> best;
    for (int it = 0; it < kRansacMax; it++) {
        int p = pick(rng), q = pick(rng);
        if (p == q || a[p].dot(a[q]) > cos5) continue;  // (two rays at least 5 deg apart)
        std::vector<int> two = {p, q};
        Matx33d Rc = kabsch(a, b, &two);
        std::vector<int> cur;
        for (int k = 0; k < n; k++)
            if ((Rc * a[k]).dot(b[k]) > cosTol) cur.push_back(k);
        if (cur.size() > best.size()) best = cur;
        if (it >= kRansacMin && (int)best.size() >= kRansacEnough) break;
    }
    if (best.size() < 2) return false;
    R = kabsch(a, b, &best);
    inl.clear();
    for (int k = 0; k < n; k++)
        if ((R * a[k]).dot(b[k]) > cosTol) inl.push_back(k);
    return inl.size() >= 2;
}

/// Up to kKeepMatches inliers spread over the image: one per grid cell in turn (GCam: MaximalStratifiedSample).
std::vector<int> stratified(const std::vector<int> &inl, const std::vector<cv::KeyPoint> &kps, int w, int h, std::mt19937 &rng) {
    double cell = 0.2 * std::min(w, h);
    int cols = (int)std::ceil(w / cell), rows = (int)std::ceil(h / cell);
    std::vector<std::vector<int>> cells(cols * rows);
    for (int k : inl) {
        int cx = std::min(cols - 1, (int)(kps[k].pt.x / cell)), cy = std::min(rows - 1, (int)(kps[k].pt.y / cell));
        cells[cy * cols + cx].push_back(k);
    }
    for (auto &c : cells) std::shuffle(c.begin(), c.end(), rng);
    std::vector<int> out;
    for (size_t round = 0; (int)out.size() < kKeepMatches; round++) {
        bool any = false;
        for (auto &c : cells)
            if (round < c.size() && (int)out.size() < kKeepMatches) {
                out.push_back(c[round]);
                any = true;
            }
        if (!any) break;
    }
    return out;
}

// ---------------------------------------------------------------- the global adjustment (GCam's bundle adjustment, with Ceres' job done by a small LM)

struct BAProblem {
    int n = 0;
    std::vector<Matx33d> R0;         // the priors
    std::vector<PairMatch> *pairs = nullptr;
    std::vector<LineMatch> *lines = nullptr;
    double f0 = 1;                   // the match-scale focal
    std::vector<double> cxs, cys;    // each photo's centre at match scale
    double ws = 0;                   // the gravity priors' weight (GCam: 0.0125 * matches / images)
    std::vector<double> pitch0, roll0;
};

void baRotations(const BAProblem &P, const std::vector<double> &x, std::vector<Matx33d> &R) {
    R.resize(P.n);
    for (int i = 0; i < P.n; i++) R[i] = expSO3(Vec3d(x[3 * i], x[3 * i + 1], x[3 * i + 2])) * P.R0[i];
}

void baResiduals(const BAProblem &P, const std::vector<double> &x, std::vector<double> &r) {
    std::vector<Matx33d> R;
    baRotations(P, x, R);
    double f = P.f0 * std::exp(x[3 * P.n]);
    r.clear();
    for (auto &pm : *P.pairs) {
        Matx33d Rji = R[pm.j].t() * R[pm.i];  // camera j from camera i
        for (size_t k = 0; k < pm.pi.size(); k++) {
            Vec3d d = Rji * Vec3d((pm.pi[k].x - P.cxs[pm.i]) / f, (pm.pi[k].y - P.cys[pm.i]) / f, 1);
            double z = std::max(d[2], 0.05);  // (a point behind the camera: a large residual, bounded by Huber)
            double w = pm.w * (k < pm.wk.size() ? pm.wk[k] : 1.0);
            r.push_back(w * (f * d[0] / z + P.cxs[pm.j] - pm.pj[k].x));
            r.push_back(w * (f * d[1] / z + P.cys[pm.j] - pm.pj[k].y));
        }
    }
    if (P.lines) {  // (GCam: LineMatchResidual: each segment's transferred endpoints' distances to the other photo's line, both ways)
        for (auto &lm : *P.lines) {
            Matx33d Rji = R[lm.j].t() * R[lm.i], Rij = Rji.t();
            auto side = [&](const Matx33d &Rt, int from, int to, const Point2d &a, const Point2d &b, const Point2d &la, const Point2d &lb) {
                for (const Point2d &p : {a, b}) {
                    Vec3d d = Rt * Vec3d((p.x - P.cxs[from]) / f, (p.y - P.cys[from]) / f, 1);
                    double z = std::max(d[2], 0.05);
                    Point2d q(f * d[0] / z + P.cxs[to], f * d[1] / z + P.cys[to]);
                    Point2d dl = lb - la, v = q - la;
                    double len = cv::norm(dl);
                    r.push_back(lm.w * (len > 1e-9 ? (v.x * dl.y - v.y * dl.x) / len : cv::norm(v)));
                }
            };
            side(Rji, lm.i, lm.j, lm.ai, lm.bi, lm.aj, lm.bj);
            side(Rij, lm.j, lm.i, lm.aj, lm.bj, lm.ai, lm.bi);
        }
    }
    r.push_back(kFocalPriorPx * x[3 * P.n]);  // (the focal stays near the lens's own value)
    for (int i = 0; i < P.n; i++) {  // the gravity priors: tilt, and roll with half the weight (GCam: SensorResidual, RollPitchSensorResidual)
        r.push_back(P.ws * f * (pitchOf(R[i]) - P.pitch0[i]));
        r.push_back(std::fabs(P.pitch0[i]) < rad(81) ? 0.5 * P.ws * f * (rollOf(R[i]) - P.roll0[i]) : 0.0);
    }
}

/// A pair's median reprojection distance in match pixels under the given rotations and focal.
inline double medianResidual(const PairMatch &pm, const std::vector<Matx33d> &R, double f, const std::vector<double> &cxs, const std::vector<double> &cys) {
    Matx33d Rji = R[pm.j].t() * R[pm.i];
    std::vector<double> e;
    for (size_t k = 0; k < pm.pi.size(); k++) {
        Vec3d d = Rji * Vec3d((pm.pi[k].x - cxs[pm.i]) / f, (pm.pi[k].y - cys[pm.i]) / f, 1);
        if (d[2] <= 0.05) {
            e.push_back(1e4);
            continue;
        }
        e.push_back(std::hypot(f * d[0] / d[2] + cxs[pm.j] - pm.pj[k].x, f * d[1] / d[2] + cys[pm.j] - pm.pj[k].y));
    }
    if (e.empty()) return 1e4;
    std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
    return e[e.size() / 2];
}

double baCost(const std::vector<double> &r) {
    double c = 0;
    for (double v : r) {
        double a = std::fabs(v);
        c += a <= kHuberPx ? v * v : 2 * kHuberPx * a - kHuberPx * kHuberPx;
    }
    return c;
}

/// Levenberg-Marquardt with a numeric Jacobian and Huber weights (IRLS): 3 rotation parameters a photo, one log focal.
void bundleAdjust(const BAProblem &P, std::vector<Matx33d> &Rout, double &fOut, int maxIter = 50) {
    int n = P.n, np = 3 * n + 1;
    std::vector<double> x(np, 0.0), r, rt;
    baResiduals(P, x, r);
    double cost = baCost(r), lambda = 1e-3;
    for (int it = 0; it < maxIter; it++) {
        int M = (int)r.size();
        std::vector<double> wgt(M);
        for (int k = 0; k < M; k++) {
            double a = std::fabs(r[k]);
            wgt[k] = a <= kHuberPx ? 1.0 : std::sqrt(kHuberPx / a);
        }
        Mat J(M, np, CV_64F), rw(M, 1, CV_64F);
        std::vector<double> xp = x;
        for (int p = 0; p < np; p++) {
            double h = p < 3 * n ? 1e-5 : 1e-4;
            xp[p] += h;
            baResiduals(P, xp, rt);
            xp[p] = x[p];
            for (int k = 0; k < M; k++) J.at<double>(k, p) = wgt[k] * (rt[k] - r[k]) / h;
        }
        for (int k = 0; k < M; k++) rw.at<double>(k) = wgt[k] * r[k];
        Mat JtJ = J.t() * J, Jtr = J.t() * rw;
        bool improved = false, done = false;
        for (int tries = 0; tries < 10 && !improved; tries++) {
            Mat A = JtJ.clone();
            for (int p = 0; p < np; p++) A.at<double>(p, p) = A.at<double>(p, p) * (1 + lambda) + 1e-9;
            Mat dx;
            if (!cv::solve(A, -Jtr, dx, cv::DECOMP_CHOLESKY)) {
                lambda *= 10;
                continue;
            }
            std::vector<double> xn(np);
            double step = 0;
            for (int p = 0; p < np; p++) {
                xn[p] = x[p] + dx.at<double>(p);
                step += dx.at<double>(p) * dx.at<double>(p);
            }
            baResiduals(P, xn, rt);
            double c1 = baCost(rt);
            if (c1 < cost) {
                done = (cost - c1) < 1e-6 * cost || step < 1e-14;
                x = xn;
                r = rt;
                cost = c1;
                lambda = std::max(1e-7, lambda / 3);
                improved = true;
            } else {
                lambda *= 4;
            }
        }
        if (!improved || done) break;
    }
    baRotations(P, x, Rout);
    fOut = P.f0 * std::exp(x[3 * n]);
}

// ---------------------------------------------------------------- rendering

struct Piece {
    Mat img, mask;   // BGR 8U, mask 8U
    cv::Point tl;
    int src = 0;
};

/// Every pixel outside the mask takes the colour of the nearest pixel inside it (GCam's Expand): the blender's bands then
/// never see the black beyond a photo's edge, which bled into the picture as bright and dark arcs along every seam.
inline void fillOutside(Mat &img, const Mat &mask) {
    Mat outside = mask == 0;
    if (cv::countNonZero(outside) == 0 || cv::countNonZero(mask) == 0) return;
    Mat dist, labels;
    cv::distanceTransform(outside, dist, labels, cv::DIST_L2, 3, cv::DIST_LABEL_PIXEL);
    std::vector<cv::Point> at(1);
    for (int y = 0; y < img.rows; y++) {  // (each inside pixel's label is its own: the labels' positions)
        const uchar *o = outside.ptr<uchar>(y);
        const int *l = labels.ptr<int>(y);
        for (int x = 0; x < img.cols; x++)
            if (!o[x]) {
                if ((int)at.size() <= l[x]) at.resize(l[x] + 1);
                at[l[x]] = cv::Point(x, y);
            }
    }
    for (int y = 0; y < img.rows; y++) {
        const uchar *o = outside.ptr<uchar>(y);
        const int *l = labels.ptr<int>(y);
        cv::Vec3b *p = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; x++)
            if (o[x] && l[x] > 0 && l[x] < (int)at.size()) p[x] = img.at<cv::Vec3b>(at[l[x]]);
    }
}

/// Warps one photo onto the equirect canvas: its bounding box from its border (a pole inside it: the whole width), an
/// inverse map over that box (each canvas pixel to a photo pixel), the photo sampled bilinearly, and the result cut into
/// pieces inside the canvas when it crosses the +-180 edge.
void warpPhoto(const Mat &img, double f, double cx, double cy, const Matx33d &R, const Equirect &eq, int src, std::vector<Piece> &out) {
    int w = img.cols, h = img.rows;
    Matx33d Rt = R.t();
    std::vector<double> us, vs;
    const int N = 64;
    auto proj = [&](double x, double y) {
        Point2d p = eq.pixel(R * rayOf(x, y, f, cx, cy));
        us.push_back(p.x);
        vs.push_back(p.y);
    };
    for (int k = 0; k <= N; k++) {
        double t = (double)k / N;
        proj(t * (w - 1), 0);
        proj(t * (w - 1), h - 1);
        proj(0, t * (h - 1));
        proj(w - 1, t * (h - 1));
    }
    bool north = false, south = false;
    for (int s = 0; s < 2; s++) {
        Vec3d d = Rt * Vec3d(0, s == 0 ? -1 : 1, 0);
        if (d[2] <= 1e-6) continue;
        double px = f * d[0] / d[2] + cx, py = f * d[1] / d[2] + cy;
        if (px >= 0 && px <= w - 1 && py >= 0 && py <= h - 1) (s == 0 ? north : south) = true;
    }
    double u0, u1, v0 = *std::min_element(vs.begin(), vs.end()), v1 = *std::max_element(vs.begin(), vs.end());
    if (north || south) {
        u0 = 0;
        u1 = eq.W - 1;
        if (north) v0 = 0;
        if (south) v1 = eq.H - 1;
    } else {
        double umin = *std::min_element(us.begin(), us.end()), umax = *std::max_element(us.begin(), us.end());
        if (umax - umin > eq.W / 2.0) {  // (across the +-180 edge: the left part moved right of it)
            for (double &u : us)
                if (u < eq.W / 2.0) u += eq.W;
            umin = *std::min_element(us.begin(), us.end());
            umax = *std::max_element(us.begin(), us.end());
        }
        u0 = umin;
        u1 = umax;
    }
    int margin = 3 + eq.W / 400;
    int x0 = (int)std::floor(u0) - margin, x1 = (int)std::ceil(u1) + margin;
    int y0 = std::max(0, (int)std::floor(v0) - margin), y1 = std::min(eq.H - 1, (int)std::ceil(v1) + margin);
    if (x1 - x0 + 1 > eq.W) {
        x0 = 0;
        x1 = eq.W - 1;
    }
    int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    if (bw <= 0 || bh <= 0) return;
    std::vector<double> sl(bw), cl(bw), sp(bh), cp(bh);
    for (int i = 0; i < bw; i++) {
        double lam = (((x0 + i) % eq.W + eq.W) % eq.W + 0.5) / eq.W * 2 * kPi - kPi;
        sl[i] = std::sin(lam);
        cl[i] = std::cos(lam);
    }
    for (int j = 0; j < bh; j++) {
        double phi = kPi / 2 - (y0 + j + 0.5) / eq.H * kPi;
        sp[j] = std::sin(phi);
        cp[j] = std::cos(phi);
    }
    Mat mapx(bh, bw, CV_32F), mapy(bh, bw, CV_32F), mask = Mat::zeros(bh, bw, CV_8U);
    for (int j = 0; j < bh; j++) {
        float *mx = mapx.ptr<float>(j), *my = mapy.ptr<float>(j);
        uchar *mk = mask.ptr<uchar>(j);
        for (int i = 0; i < bw; i++) {
            double dx = cp[j] * sl[i], dy = -sp[j], dz = cp[j] * cl[i];
            double cxw = Rt(0, 0) * dx + Rt(0, 1) * dy + Rt(0, 2) * dz;
            double cyw = Rt(1, 0) * dx + Rt(1, 1) * dy + Rt(1, 2) * dz;
            double czw = Rt(2, 0) * dx + Rt(2, 1) * dy + Rt(2, 2) * dz;
            if (czw <= 1e-6) {
                mx[i] = my[i] = -10;
                continue;
            }
            double px = f * cxw / czw + cx, py = f * cyw / czw + cy;
            mx[i] = (float)px;
            my[i] = (float)py;
            if (px >= 0 && px <= w - 1 && py >= 0 && py <= h - 1) mk[i] = 255;
        }
    }
    Mat warped;
    cv::remap(img, warped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    fillOutside(warped, mask);
    auto piece = [&](int cx0, int cx1, int atx) {  // (columns [cx0, cx1) of the warped box, placed at canvas column atx)
        if (cx1 <= cx0) return;
        Piece p;
        p.img = warped(cv::Rect(cx0, 0, cx1 - cx0, bh)).clone();
        p.mask = mask(cv::Rect(cx0, 0, cx1 - cx0, bh)).clone();
        p.tl = cv::Point(atx, y0);
        p.src = src;
        if (cv::countNonZero(p.mask) > 0) out.push_back(p);
    };
    if (x0 >= 0 && x1 < eq.W) {
        piece(0, bw, x0);
    } else if (x0 < 0) {
        piece(0, -x0, eq.W + x0);
        piece(-x0, bw, 0);
    } else {
        piece(0, eq.W - x0, x0);
        piece(eq.W - x0, bw, 0);
    }
}

/// GCam's exposure matching (ComputeRosetteGammaAdjustment): about 2000 directions spread over the sphere (39 latitude
/// rings 4.54 deg apart), each photo's mean luma over the directions it shares with another, and one gamma per photo
/// minimising sum N_ij (g_i ln L_ij - g_j ln L_ji)^2 + 1/2 sum N_i (g_i - 1)^2, clamped to [1/1.75, 1.75]. Photos without
/// overlap keep gamma 1. The gammas, one per photo.
std::vector<double> matchExposure(const std::vector<Mat> &imgs, const std::vector<Matx33d> &R, double f, std::vector<double> &gammaOut) {
    int n = (int)imgs.size();
    std::vector<std::vector<double>> sum(n, std::vector<double>(n, 0));
    std::vector<std::vector<int>> cnt(n, std::vector<int>(n, 0));
    const int rings = 39;
    for (int k = 0; k < rings; k++) {
        double lat = -kPi / 2 + (k + 0.5) * kPi / rings;
        int m = std::max(1, (int)std::lround(2 * kPi * std::cos(lat) / 0.0792));
        for (int t = 0; t < m; t++) {
            double lon = (t + 0.5) * 2 * kPi / m - kPi;
            Vec3d d(std::cos(lat) * std::sin(lon), -std::sin(lat), std::cos(lat) * std::cos(lon));
            std::vector<std::pair<int, double>> hits;
            for (int i = 0; i < n; i++) {
                if (imgs[i].empty()) continue;
                Vec3d c = R[i].t() * d;
                if (c[2] <= 1e-6) continue;
                double cx = (imgs[i].cols - 1) / 2.0, cy = (imgs[i].rows - 1) / 2.0;
                int px = (int)std::lround(f * c[0] / c[2] + cx), py = (int)std::lround(f * c[1] / c[2] + cy);
                if (px < 0 || py < 0 || px >= imgs[i].cols || py >= imgs[i].rows) continue;
                cv::Vec3b p = imgs[i].at<cv::Vec3b>(py, px);
                hits.push_back({i, (0.114 * p[0] + 0.5871 * p[1] + 0.2989 * p[2]) / 255.0});
            }
            for (auto &a : hits)
                for (auto &b : hits)
                    if (a.first != b.first) {
                        sum[a.first][b.first] += a.second;
                        cnt[a.first][b.first]++;
                    }
        }
    }
    Mat A = Mat::zeros(n, n, CV_64F), b = Mat::zeros(n, 1, CV_64F);
    for (int i = 0; i < n; i++) {
        bool any = false;
        for (int j = 0; j < n; j++) {
            int N = cnt[i][j];
            if (i == j || N <= 5) continue;
            double Lij = sum[i][j] / N, Lji = sum[j][i] / std::max(1, cnt[j][i]);
            if (Lij < 2.5 / 255 || Lji < 2.5 / 255) continue;
            double a = std::log(Lij), c = std::log(Lji);
            A.at<double>(i, i) += 2 * N * a * a + N;
            A.at<double>(i, j) += -2 * N * a * c;
            b.at<double>(i) += N;
            any = true;
        }
        if (!any) {
            A.at<double>(i, i) = 1;
            b.at<double>(i) = 1;
        }
    }
    Mat g;
    gammaOut.assign(n, 1.0);
    if (cv::solve(A, b, g, cv::DECOMP_SVD))
        for (int i = 0; i < n; i++) gammaOut[i] = std::max(1 / kGammaMax, std::min(kGammaMax, std::isfinite(g.at<double>(i)) ? g.at<double>(i) : 1.0));
    return gammaOut;
}

/// One gain per photo from the overlaps' mean luma (Brown and Lowe's gain compensation, solved in log space so large
/// differences fit): sum N_ij (ln g_i - ln g_j - ln(L_ji / L_ij))^2, the mean log gain held at 0 (the sphere keeps the
/// photos' overall brightness), clamped to [1/4, 4]. Applied in linear light.
inline std::vector<double> matchGains(const std::vector<Mat> &imgs, const std::vector<Matx33d> &R, double f, double &maxRatio) {
    int n = (int)imgs.size();
    std::vector<std::vector<double>> sum(n, std::vector<double>(n, 0));
    std::vector<std::vector<int>> cnt(n, std::vector<int>(n, 0));
    const int rings = 39;
    for (int k = 0; k < rings; k++) {
        double lat = -kPi / 2 + (k + 0.5) * kPi / rings;
        int m = std::max(1, (int)std::lround(2 * kPi * std::cos(lat) / 0.0792));
        for (int t = 0; t < m; t++) {
            double lon = (t + 0.5) * 2 * kPi / m - kPi;
            Vec3d d(std::cos(lat) * std::sin(lon), -std::sin(lat), std::cos(lat) * std::cos(lon));
            std::vector<std::pair<int, double>> hits;
            for (int i = 0; i < n; i++) {
                if (imgs[i].empty()) continue;
                Vec3d c = R[i].t() * d;
                if (c[2] <= 1e-6) continue;
                double cx = (imgs[i].cols - 1) / 2.0, cy = (imgs[i].rows - 1) / 2.0;
                int px = (int)std::lround(f * c[0] / c[2] + cx), py = (int)std::lround(f * c[1] / c[2] + cy);
                if (px < 0 || py < 0 || px >= imgs[i].cols || py >= imgs[i].rows) continue;
                cv::Vec3b p = imgs[i].at<cv::Vec3b>(py, px);
                double l = (0.114 * p[0] + 0.5871 * p[1] + 0.2989 * p[2]) / 255.0;
                hits.push_back({i, std::pow(std::max(l, 1e-3), 2.2)});  // (linear light)
            }
            for (auto &a : hits)
                for (auto &b : hits)
                    if (a.first != b.first) {
                        sum[a.first][b.first] += a.second;
                        cnt[a.first][b.first]++;
                    }
        }
    }
    Mat A = Mat::zeros(n, n, CV_64F), b = Mat::zeros(n, 1, CV_64F);
    maxRatio = 1;
    double totalN = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int N = cnt[i][j];
            if (i == j || N <= 5 || cnt[j][i] <= 5) continue;
            double Lij = sum[i][j] / N, Lji = sum[j][i] / cnt[j][i];
            if (Lij < 1e-3 || Lji < 1e-3) continue;
            double r = std::log(Lji / Lij);  // (ln g_i - ln g_j = r)
            maxRatio = std::max(maxRatio, std::exp(std::fabs(r)));
            A.at<double>(i, i) += N;
            A.at<double>(i, j) -= N;
            b.at<double>(i) += N * r;
            totalN += N;
        }
    double prior = 1e-3 * std::max(1.0, totalN / n);  // (just enough to fix the overall level)
    for (int i = 0; i < n; i++) A.at<double>(i, i) += prior;
    Mat x;
    std::vector<double> g(n, 1.0);
    if (cv::solve(A, b, x, cv::DECOMP_SVD)) {
        double mean = 0;
        for (int i = 0; i < n; i++) mean += std::isfinite(x.at<double>(i)) ? x.at<double>(i) : 0;
        mean /= n;
        for (int i = 0; i < n; i++) {
            double v = (std::isfinite(x.at<double>(i)) ? x.at<double>(i) : 0) - mean;
            g[i] = std::exp(std::max(-std::log(4.0), std::min(std::log(4.0), v)));
        }
    }
    return g;
}

inline void applyGain(Mat &img, double g) {
    if (std::fabs(g - 1) < 1e-3) return;
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) lut.at<uchar>(i) = (uchar)std::lround(std::min(1.0, std::pow(std::pow(i / 255.0, 2.2) * g, 1 / 2.2)) * 255.0);
    cv::LUT(img, lut, img);
}

void applyGamma(Mat &img, double g) {
    if (std::fabs(g - 1) < 1e-3) return;
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) lut.at<uchar>(i) = (uchar)std::lround(std::pow(i / 255.0, g) * 255.0);
    cv::LUT(img, lut, img);
}

/// GCam's hole treatment: what no photo covered is filled from the coarsest blend level when small (under 0.5% of the
/// canvas), left black when large. Here: small holes take the colour of their blurred surroundings.
/// Holes (no photo covered them) are filled by inpainting from their surroundings at a quarter of the resolution: small
/// ones vanish, a missing ceiling corner becomes a smooth patch of ceiling. (GCam filled only holes under 0.5% of the
/// sphere, from its coarsest level, and left the rest black; a smooth patch reads better than black.) Beyond a tenth of
/// the sphere missing, nothing is filled: the crop takes care of a missing sky or floor.
inline void fillHoles(Mat &result, const Mat &coverage) {
    Mat hole = coverage == 0;
    int holes = cv::countNonZero(hole);
    if (holes == 0 || holes > 0.1 * hole.total()) return;
    Mat small, smallHole, filled, up;
    cv::resize(result, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::resize(hole, smallHole, small.size(), 0, 0, cv::INTER_NEAREST);
    cv::dilate(smallHole, smallHole, Mat(), cv::Point(-1, -1), 1);
    cv::inpaint(small, smallHole, filled, 3, cv::INPAINT_TELEA);
    cv::resize(filled, up, result.size(), 0, 0, cv::INTER_LINEAR);
    up.copyTo(result, hole);
}




// ---------------------------------------------------------------- the stitch

/// sensors: the rotations come from a gyro (the app's captures). false: they were worked out from the images (bootstrap),
/// so they get no sensor pairs or geometry checks, a weaker gravity prior, and a photo nothing holds is dropped.
inline StitchResult stitch(const std::vector<std::string> &paths, const std::vector<Matx33d> &rotations, double focalPx,
                           int outputWidth, const std::string &outputPath, const Progress &progress, bool sensors = true) {
    auto report = [&](double frac, const char *stage) {
        if (progress) progress(frac, stage);
    };
    double t0 = (double)cv::getTickCount();
    int n = (int)paths.size();
    if (n < 1 || rotations.size() != paths.size()) throw std::runtime_error("no photos");
    std::mt19937 rng(7);

    // ---- features (GCam: FastCornerDetector + OrientedPatchExtractor; here ORB: FAST corners with oriented binary patches)
    std::vector<View> views(n);
    std::vector<Mat> photos(n);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10);  // (a low FAST threshold: plain walls)
    double matchScale = 1, fm = focalPx;
    for (int i = 0; i < n; i++) {
        Mat img = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (img.empty()) throw std::runtime_error("can't read " + paths[i]);
        if (i == 0) {
            matchScale = std::min(1.0, (double)kMatchWidth / std::max(img.cols, img.rows));
            fm = focalPx * matchScale;
        }
        photos[i] = img;
        View &v = views[i];
        // (every photo's long side goes to the match width: one camera, so one match-scale focal, whichever way it was held)
        v.scale = std::min(1.0, (double)kMatchWidth / std::max(img.cols, img.rows));
        v.Rp = rotations[i];
        Mat small, gray;
        cv::resize(img, small, cv::Size(), v.scale, v.scale, cv::INTER_AREA);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        v.w = gray.cols;
        v.h = gray.rows;
        v.cx = (v.w - 1) / 2.0;
        v.cy = (v.h - 1) / 2.0;
        orb->detectAndCompute(gray, cv::noArray(), v.kps, v.desc);
        v.lines = detectLines(gray);
        v.s16 = makeSmall(gray, fm, 0.0625);
        v.s8 = makeSmall(gray, fm, 0.125);
        v.s4 = makeSmall(gray, fm, 0.25);
        v.rays.resize(v.kps.size());
        v.lat.resize(v.kps.size());
        for (size_t k = 0; k < v.kps.size(); k++) {
            v.rays[k] = rayOf(v.kps[k].pt.x, v.kps[k].pt.y, fm, v.cx, v.cy);
            v.lat[k] = std::asin(clamp1(-(v.Rp * v.rays[k])[1]));
        }
        report(0.25 * (i + 1) / n, "finding features");
    }
    double fovLong = 2 * std::atan(std::max(views[0].w, views[0].h) / 2.0 / fm);

    // ---- pairs: the photos whose sensor axes overlap, matched by descriptor within a latitude band, verified by a
    // two-point RANSAC rotation and against the sensors (GCam: SphericalPairwiseMatcher, VerifyMatchInliers/Geometry)
    std::vector<PairMatch> pairs;
    std::vector<std::vector<int>> matched(n);
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int candidates = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (angleBetween(forwardOf(views[i].Rp), forwardOf(views[j].Rp)) < kAxesNear * fovLong) candidates++;
    int seen = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            View &a = views[i], &b = views[j];
            if (angleBetween(forwardOf(a.Rp), forwardOf(b.Rp)) >= kAxesNear * fovLong) continue;
            report(0.25 + 0.25 * (++seen) / std::max(1, candidates), "matching");
            if (a.desc.empty() || b.desc.empty()) continue;
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(a.desc, b.desc, knn, 2);
            std::vector<Vec3d> ra, rb;
            std::vector<int> ka, kb;
            for (auto &m : knn) {
                if (m.size() < 2 || m[0].distance > kRatio * m[1].distance) continue;
                if (std::fabs(a.lat[m[0].queryIdx] - b.lat[m[0].trainIdx]) > rad(kLatGateDeg)) continue;
                ra.push_back(a.rays[m[0].queryIdx]);
                rb.push_back(b.rays[m[0].trainIdx]);
                ka.push_back(m[0].queryIdx);
                kb.push_back(m[0].trainIdx);
            }
            if (ra.size() < 7) continue;  // (GCam: LOW_MATCH_FAIL)
            Matx33d Rvis;                 // camera j from camera i, from the images
            std::vector<int> inl;
            if (!ransacRotation(ra, rb, rng, Rvis, inl)) continue;
            if ((int)inl.size() <= 5 + 0.15 * (int)ra.size()) continue;  // (INLIER_FAIL)
            if (!sensors && ((int)inl.size() < 12 || angleOf(Rvis * (b.Rp.t() * a.Rp).t()) > rad(kPriorAgreeDeg))) continue;  // (a chance match)
            if (sensors) {
                Matx33d Rsens = b.Rp.t() * a.Rp, E = Rvis * Rsens.t();
                double yawTol = rad(std::min(25.0 * (j - i), 65.0));
                Vec3d upj = b.Rp.t() * Vec3d(0, -1, 0);
                if (angleOf(E) > yawTol || angleBetween(E * upj, upj) > rad(12.5)) continue;  // (GEO_FAIL: not what the sensors say)
            }
            std::vector<cv::KeyPoint> kpsA;
            for (int k : inl) kpsA.push_back(a.kps[ka[k]]);
            std::vector<int> all(inl.size());
            for (size_t k = 0; k < inl.size(); k++) all[k] = (int)k;
            std::vector<int> keep = stratified(all, kpsA, a.w, a.h, rng);
            PairMatch pm;
            pm.i = i;
            pm.j = j;
            pm.inliers = (int)inl.size();
            pm.putative = (int)ra.size();
            pm.w = std::sqrt(inl.size() / (double)kKeepMatches);
            for (int k : keep) {
                pm.pi.push_back(Point2d(a.kps[ka[inl[k]]].pt.x, a.kps[ka[inl[k]]].pt.y));
                pm.pj.push_back(Point2d(b.kps[kb[inl[k]]].pt.x, b.kps[kb[inl[k]]].pt.y));
                // the ground (GCam: DownWeightGroundPoints(-45 deg, 0.1)): a match below -45 deg of latitude under the sensors
                // weighs from 1 there down to 0.1 at the nadir (the phone orbits the hand, not the lens: parallax is worst there)
                double lat = a.lat[ka[inl[k]]], lim = rad(-45);
                pm.wk.push_back(lat >= lim ? 1.0 : 1.0 - 0.9 * std::min(1.0, (lim - lat) / (kPi / 2 + lim)));
            }
            pairs.push_back(pm);
            matched[i].push_back(j);
            matched[j].push_back(i);
        }
    }
    int imagePairs = (int)pairs.size(), directPairs = 0, lineMatches = 0;
    std::vector<LineMatch> lines;
    int keptPoints = 0;
    for (auto &p : pairs) keptPoints += (int)p.pi.size();
    // sensor pairs: a weak tie between each photo and the one before when no image match links them (GCam: SENSOR_PAIR,
    // a 3x3 grid transferred through the sensors, weight 0.125 or 0.25 when the axes are near)
    for (int i = 1; i < n && sensors; i++) {
        bool linked = std::find(matched[i].begin(), matched[i].end(), i - 1) != matched[i].end();
        if (linked) continue;
        View &a = views[i - 1], &b = views[i];
        PairMatch pm;
        pm.i = i - 1;
        pm.j = i;
        pm.sensor = true;
        pm.w = angleBetween(forwardOf(a.Rp), forwardOf(b.Rp)) < kAxesNear * fovLong ? 0.25 : 0.125;
        Matx33d Rji = b.Rp.t() * a.Rp;
        for (int gy = 0; gy < 3; gy++)
            for (int gx = 0; gx < 3; gx++) {
                Point2d p(a.w * (0.1 + 0.4 * gx), a.h * (0.1 + 0.4 * gy));
                Vec3d d = Rji * Vec3d((p.x - a.cx) / fm, (p.y - a.cy) / fm, 1);
                if (d[2] <= 0.05) continue;
                Point2d q(fm * d[0] / d[2] + b.cx, fm * d[1] / d[2] + b.cy);
                pm.pi.push_back(p);
                pm.pj.push_back(q);
            }
        if (!pm.pi.empty()) pairs.push_back(pm);
    }

    // ---- the global adjustment: every photo's rotation and one focal length, matches robust, tilt and roll pulled to
    // the sensors; yaw is free and re-anchored to the first photo's afterwards (GCam's gauge)
    report(0.5, "adjusting");
    BAProblem P;
    P.n = n;
    P.pairs = &pairs;
    P.f0 = fm;
    P.ws = keptPoints > 0 ? 0.0125 * keptPoints / n : 0.5;
    if (!sensors) P.ws *= 0.25;  // (the "gravity" came from the images: a weaker pull)
    std::vector<bool> dead(n, false);
    for (int i = 0; i < n; i++) {
        P.cxs.push_back(views[i].cx);
        P.cys.push_back(views[i].cy);
        P.R0.push_back(views[i].Rp);
        P.pitch0.push_back(pitchOf(views[i].Rp));
        P.roll0.push_back(rollOf(views[i].Rp));
    }
    std::vector<Matx33d> R;
    double fAdj = fm;
    if (n >= 2) {
        bundleAdjust(P, R, fAdj);
        double fov = deg(2 * std::atan(std::max(views[0].w, views[0].h) / 2.0 / fAdj));
        if (!(fov > 10 && fov < 150) || !std::isfinite(fAdj)) {  // (a solve gone wrong: the sensors' rotations as they are)
            R = P.R0;
            fAdj = fm;
        }
        Matx33d G = rotY(yawOf(P.R0[0]) - yawOf(R[0]));
        for (auto &Ri : R) Ri = G * Ri;

        // ---- without a gyro, a photo stays only when one of its matches holds under the adjusted rotations (a gyro held
        // every photo for GCam); the others are dropped rather than painted where a chance match put them
        if (!sensors) {
            std::vector<double> best(n, 1e9);
            for (auto &pm : pairs) {
                double med = medianResidual(pm, R, fAdj, P.cxs, P.cys);
                best[pm.i] = std::min(best[pm.i], med);
                best[pm.j] = std::min(best[pm.j], med);
            }
            for (int i = 0; i < n; i++) dead[i] = best[i] < 1e9 && best[i] > kSupportPx;
            std::vector<PairMatch> alive;
            for (auto &pm : pairs)
                if (!dead[pm.i] && !dead[pm.j]) alive.push_back(pm);
            pairs.swap(alive);
        }

        // ---- plain walls: the photos' straight edges matched under the adjusted rotations (GCam's line features), and pairs
        // the features left weak aligned directly on the pixels; both feed a second adjustment, then the lines once more
        report(0.56, "aligning plain walls");
        auto matchAllLines = [&](const std::vector<Matx33d> &Rc) {
            lines.clear();
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (!dead[i] && !dead[j] && angleBetween(forwardOf(Rc[i]), forwardOf(Rc[j])) < 0.9 * fovLong)
                        matchLines(i, j, views[i].lines, views[j].lines, Rc[j].t() * Rc[i], fm, views[i].cx, views[i].cy, views[j].cx, views[j].cy, views[j].w,
                                   views[j].h, lines);
            return (int)lines.size();
        };
        lineMatches = matchAllLines(R);
        int refined = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                View &a = views[i], &b = views[j];
                if (dead[i] || dead[j] || angleBetween(forwardOf(R[i]), forwardOf(R[j])) >= 0.8 * fovLong) continue;
                int inl = 0;
                for (auto &pm : pairs)
                    if (!pm.sensor && pm.i == i && pm.j == j) inl = std::max(inl, pm.inliers);
                if (inl >= 20) continue;
                Matx33d Rji = R[j].t() * R[i];
                double before = 0, after = 0;
                int pts = 0;
                if (!refineDirect(a.s16, b.s16, Rji, before, after, pts)) continue;
                if (!refineDirect(a.s8, b.s8, Rji, before, after, pts)) continue;
                if (!refineDirect(a.s4, b.s4, Rji, before, after, pts)) continue;
                PairMatch pm;
                pm.i = i;
                pm.j = j;
                pm.direct = true;
                pm.w = 0.6;
                for (int gy = 0; gy < 5; gy++)
                    for (int gx = 0; gx < 5; gx++) {
                        Point2d p(a.w * (0.1 + 0.2 * gx), a.h * (0.1 + 0.2 * gy));
                        Vec3d d = Rji * Vec3d((p.x - a.cx) / fm, (p.y - a.cy) / fm, 1);
                        if (d[2] <= 0.05) continue;
                        Point2d q(fm * d[0] / d[2] + b.cx, fm * d[1] / d[2] + b.cy);
                        if (q.x < 0 || q.y < 0 || q.x > b.w - 1 || q.y > b.h - 1) continue;
                        pm.pi.push_back(p);
                        pm.pj.push_back(q);
                    }
                if (pm.pi.size() >= 6) {
                    pairs.push_back(pm);
                    refined++;
                }
            }
        }
        for (int pass = 0; pass < 2 && (refined > 0 || lineMatches > 0); pass++) {
            BAProblem P2 = P;
            P2.R0 = R;
            P2.f0 = fAdj;
            P2.lines = &lines;
            std::vector<Matx33d> R2;
            double f2 = fAdj;
            bundleAdjust(P2, R2, f2);
            double fov2 = deg(2 * std::atan(std::max(views[0].w, views[0].h) / 2.0 / f2));
            if (fov2 > 10 && fov2 < 150 && std::isfinite(f2)) {
                Matx33d G2 = rotY(yawOf(P.R0[0]) - yawOf(R2[0]));
                for (auto &Ri : R2) Ri = G2 * Ri;
                R = R2;
                fAdj = f2;
            }
            if (pass == 0) lineMatches = matchAllLines(R);  // (matched again from the better rotations, then adjusted once more)
        }
        directPairs = refined;
    } else {
        R = P.R0;
    }
    double fFull = fAdj / matchScale;
    std::string pairReport;
    for (auto &pm : pairs) {
        char line[96];
        snprintf(line, sizeof line, "%d %d %c %d %.1f\n", pm.i, pm.j, pm.sensor ? 'S' : pm.direct ? 'D' : 'F', pm.inliers, medianResidual(pm, R, fAdj, P.cxs, P.cys));
        pairReport += line;
    }

    // ---- render: each photo warped onto the canvas, exposure matched, seams cut, blended in bands
    Equirect eq(outputWidth);
    double composeScale = std::min(1.0, 1.3 * (eq.W / (2 * kPi)) / fFull);  // (sources no more than 1.3x the output's resolution)
    std::vector<Piece> pieces;
    Mat coverage = Mat::zeros(eq.H, eq.W, CV_8U);
    report(0.6, "matching exposure");
    std::vector<Mat> smalls(n);
    for (int i = 0; i < n; i++)
        if (!dead[i]) cv::resize(photos[i], smalls[i], cv::Size(), views[i].scale * 0.25, views[i].scale * 0.25, cv::INTER_AREA);
    std::vector<double> gammas;
    double maxRatio = 1;
    gammas = matchGains(smalls, R, fm * 0.25, maxRatio);
    smalls.clear();
    for (int i = 0; i < n; i++) {
        if (dead[i]) {
            photos[i].release();
            continue;
        }
        Mat src = photos[i];
        if (composeScale < 0.999) cv::resize(photos[i], src, cv::Size(), composeScale, composeScale, cv::INTER_AREA);
        applyGain(src, gammas[i]);
        double f = fAdj / views[i].scale * composeScale, cx = (src.cols - 1) / 2.0, cy = (src.rows - 1) / 2.0;
        size_t before = pieces.size();
        warpPhoto(src, f, cx, cy, R[i], eq, i, pieces);
        for (size_t k = before; k < pieces.size(); k++) {
            Mat roi = coverage(cv::Rect(pieces[k].tl, pieces[k].mask.size()));
            cv::bitwise_or(roi, pieces[k].mask, roi);
        }
        photos[i].release();
        report(0.6 + 0.12 * (i + 1) / n, "warping");
    }
    if (pieces.empty()) throw std::runtime_error("nothing to render");
    int np = (int)pieces.size();
    std::vector<cv::Point> corners(np);
    std::vector<cv::UMat> imgs(np), masks(np);
    for (int k = 0; k < np; k++) {
        corners[k] = pieces[k].tl;
        pieces[k].img.copyTo(imgs[k]);
        pieces[k].mask.copyTo(masks[k]);
    }
    report(0.76, "finding seams");
    double seamScale = std::min(1.0, 1024.0 / eq.W);  // (GCam cut its seams on a 512-wide mosaic)
    std::vector<cv::UMat> seamImgs(np), seamMasks(np);
    std::vector<cv::Point> seamCorners(np);
    for (int k = 0; k < np; k++) {
        cv::UMat s;
        cv::resize(imgs[k], s, cv::Size(), seamScale, seamScale, cv::INTER_AREA);
        s.convertTo(seamImgs[k], CV_32F);
        cv::resize(masks[k], seamMasks[k], s.size(), 0, 0, cv::INTER_NEAREST);
        seamCorners[k] = cv::Point((int)std::lround(corners[k].x * seamScale), (int)std::lround(corners[k].y * seamScale));
    }
    try {
        cv::detail::GraphCutSeamFinder seam(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
        seam.find(seamImgs, seamCorners, seamMasks);
    } catch (const std::exception &) {
        cv::detail::VoronoiSeamFinder seam;
        seam.find(seamImgs, seamCorners, seamMasks);
    }
    seamImgs.clear();
    report(0.84, "blending");
    double blendWidth = std::sqrt((double)eq.W * eq.H) * 5.0 / 100.0;
    int bands = std::max(1, (int)std::ceil(std::log2(blendWidth)));
    cv::detail::MultiBandBlender blender(false, bands);
    int pad = std::max(1 << bands, eq.W / 8);  // (a margin each side of the +-180 edge: a piece near one edge is fed again
    // a whole turn away, with the same seam, so the pyramid blends the true neighbours there; GCam wrapped its collapse)
    blender.prepare(cv::Rect(-pad, 0, eq.W + 2 * pad, eq.H));
    for (int k = 0; k < np; k++) {
        cv::UMat dilated, seamUp, m;
        cv::dilate(seamMasks[k], dilated, cv::Mat());
        cv::resize(dilated, seamUp, masks[k].size(), 0, 0, cv::INTER_LINEAR_EXACT);
        cv::bitwise_and(seamUp, masks[k], m);
        cv::UMat img16;
        imgs[k].convertTo(img16, CV_16S);
        blender.feed(img16, m, corners[k]);
        if (corners[k].x < pad) {  // (its part within the margin, a turn to the right)
            int w = std::min(img16.cols, pad - corners[k].x);
            blender.feed(img16(cv::Rect(0, 0, w, img16.rows)), m(cv::Rect(0, 0, w, m.rows)), cv::Point(corners[k].x + eq.W, corners[k].y));
        }
        if (corners[k].x + img16.cols > eq.W - pad) {  // (its part within the margin, a turn to the left)
            int c0 = std::max(0, eq.W - pad - corners[k].x);
            blender.feed(img16(cv::Rect(c0, 0, img16.cols - c0, img16.rows)), m(cv::Rect(c0, 0, m.cols - c0, m.rows)), cv::Point(corners[k].x - eq.W + c0, corners[k].y));
        }
        imgs[k].release();
        report(0.84 + 0.1 * (k + 1) / np, "blending");
    }
    Mat result16, resultMask;
    blender.blend(result16, resultMask);
    Mat result;
    result16(cv::Rect(pad, 0, eq.W, eq.H)).convertTo(result, CV_8U);
    result.setTo(cv::Scalar(0, 0, 0), coverage == 0);
    fillHoles(result, coverage);  // (small holes filled from their surroundings, large ones black, as GCam)

    // ---- crop: a full circle keeps its width and loses rows not fully covered (10 deg at most each end); a partial
    // sphere is cut to what's covered
    report(0.95, "saving");
    Mat colAny, rowAll, rowAny;
    cv::reduce(coverage, colAny, 0, cv::REDUCE_MAX);
    cv::reduce(coverage, rowAny, 1, cv::REDUCE_MAX);
    cv::reduce(coverage, rowAll, 1, cv::REDUCE_MIN);
    bool full = cv::countNonZero(colAny) == eq.W;
    int top = 0, bottom = eq.H - 1, left = 0, right = eq.W - 1;
    if (full) {
        int limit = eq.H / 18;
        while (top < limit && rowAll.at<uchar>(top) == 0) top++;
        while (bottom > eq.H - 1 - limit && rowAll.at<uchar>(bottom) == 0) bottom--;
    } else {
        while (top < eq.H - 1 && rowAny.at<uchar>(top) == 0) top++;
        while (bottom > top && rowAny.at<uchar>(bottom) == 0) bottom--;
        while (left < eq.W - 1 && colAny.at<uchar>(left) == 0) left++;
        while (right > left && colAny.at<uchar>(right) == 0) right--;
    }
    Mat cropped = result(cv::Rect(left, top, right - left + 1, bottom - top + 1));
    if (!cv::imwrite(outputPath, cropped, {cv::IMWRITE_JPEG_QUALITY, 90})) throw std::runtime_error("can't write the panorama");
    StitchResult res;
    res.path = outputPath;
    res.fullW = eq.W;
    res.fullH = eq.H;
    res.cropW = cropped.cols;
    res.cropH = cropped.rows;
    res.left = left;
    res.top = top;
    res.full = full;
    res.images = n;
    res.featurePairs = imagePairs;
    res.directPairs = directPairs;
    res.focal = fFull;
    res.seconds = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    res.R = R;
    res.pairReport = pairReport;
    for (int i = 0; i < n; i++)
        if (dead[i]) res.dropped.push_back(i);
    {
        double gmin = 9, gmax = 0;
        for (int i = 0; i < n; i++) {
            if (dead[i]) continue;
            gmin = std::min(gmin, gammas[i]);
            gmax = std::max(gmax, gammas[i]);
        }
        char buf[200];
        snprintf(buf, sizeof buf, "gains %.2f..%.2f (overlaps differed up to %.2fx), %d line matches, %d dropped, compose scale %.2f, %d pieces, %d bands", gmin, gmax, maxRatio,
                 lineMatches, (int)res.dropped.size(), composeScale, np, bands);
        res.notes = buf;
    }
    report(1.0, "done");
    return res;
}

// ---------------------------------------------------------------- orientations from the photos alone

/// For photos without sensor orientations (a camera app's): every pair is matched, the relative rotations are chained
/// over a spanning tree from the best-connected photo, "up" is the direction all the photos' right vectors are
/// perpendicular to (the phone held upright), and the result is a set of orientations the normal stitch can start from.
inline Bootstrap bootstrap(const std::vector<std::string> &paths, double focalPx, const Progress &progress) {
    auto report = [&](double frac, const char *stage) {
        if (progress) progress(frac, stage);
    };
    Bootstrap out;
    int n = (int)paths.size();
    if (n < 2) return out;
    std::mt19937 rng(5);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10);
    struct F {
        std::vector<cv::KeyPoint> kps;
        Mat desc;
        std::vector<Vec3d> rays;
    };
    std::vector<F> fs(n);
    std::vector<Small> s16(n), s8(n), s4(n);
    double fm = focalPx, cx = 0, cy = 0;
    for (int i = 0; i < n; i++) {
        Mat img = cv::imread(paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) throw std::runtime_error("can't read " + paths[i]);
        double scale = std::min(1.0, (double)kMatchWidth / std::max(img.cols, img.rows));
        if (i == 0) fm = focalPx * scale;
        Mat gray;
        cv::resize(img, gray, cv::Size(), scale, scale, cv::INTER_AREA);
        cx = (gray.cols - 1) / 2.0;
        cy = (gray.rows - 1) / 2.0;
        s16[i] = makeSmall(gray, fm, 0.0625);
        s8[i] = makeSmall(gray, fm, 0.125);
        s4[i] = makeSmall(gray, fm, 0.25);
        orb->detectAndCompute(gray, cv::noArray(), fs[i].kps, fs[i].desc);
        for (auto &k : fs[i].kps) fs[i].rays.push_back(rayOf(k.pt.x, k.pt.y, fm, cx, cy));
        report(0.4 * (i + 1) / n, "finding features");
    }
    struct Edge {
        int i, j, inliers;
        Matx33d Rji;
    };
    std::vector<Edge> edges;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int total = n * (n - 1) / 2, seen = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            report(0.4 + 0.55 * (++seen) / total, "matching every pair");
            if (fs[i].desc.empty() || fs[j].desc.empty()) continue;
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(fs[i].desc, fs[j].desc, knn, 2);
            std::vector<Vec3d> ra, rb;
            for (auto &m : knn) {
                if (m.size() < 2 || m[0].distance > kRatio * m[1].distance) continue;
                ra.push_back(fs[i].rays[m[0].queryIdx]);
                rb.push_back(fs[j].rays[m[0].trainIdx]);
            }
            if (ra.size() < 15) continue;
            Matx33d Rji;
            std::vector<int> inl;
            if (!ransacRotation(ra, rb, rng, Rji, inl)) continue;
            if ((int)inl.size() < 12 || (int)inl.size() <= 5 + 0.15 * (int)ra.size()) continue;
            edges.push_back({i, j, (int)inl.size(), Rji});
        }
    }
    if (edges.empty()) return out;
    // verification: GCam checked each pair against the gyro; without one, a pair is believed when a third photo agrees with
    // it (the three relative rotations around the triangle close within kCycleDeg), or when it is strong on its own
    {
        int m = (int)edges.size();
        std::map<std::pair<int, int>, int> at;
        for (int e = 0; e < m; e++) at[{edges[e].i, edges[e].j}] = e;
        auto rel = [&](int a, int b, Matx33d &Rba) {  // camera b from camera a, when they share an edge
            auto it = at.find({std::min(a, b), std::max(a, b)});
            if (it == at.end()) return false;
            const Edge &e = edges[it->second];
            Rba = a == e.i ? e.Rji : e.Rji.t();
            return true;
        };
        std::vector<Edge> kept;
        for (int e = 0; e < m; e++) {
            bool ok = edges[e].inliers >= kStrongEdge;
            for (int k = 0; k < n && !ok; k++) {
                if (k == edges[e].i || k == edges[e].j) continue;
                Matx33d Rki, Rjk;
                if (!rel(edges[e].i, k, Rki) || !rel(k, edges[e].j, Rjk)) continue;
                ok = deg(angleOf(edges[e].Rji.t() * (Rjk * Rki))) < kCycleDeg;
            }
            if (ok) kept.push_back(edges[e]);
        }
        out.rejected = m - (int)kept.size();
        edges.swap(kept);
        if (edges.empty()) return out;
    }
    // the spanning tree of most inliers, from the photo with the most of them
    std::vector<int> degree(n, 0);
    for (auto &e : edges) {
        degree[e.i] += e.inliers;
        degree[e.j] += e.inliers;
    }
    int root = (int)(std::max_element(degree.begin(), degree.end()) - degree.begin());
    std::vector<Matx33d> R(n, Matx33d::eye());
    std::vector<bool> placed(n, false);
    placed[root] = true;
    for (int placedCount = 1; placedCount < n;) {
        const Edge *best = nullptr;
        bool fromI = false;
        for (auto &e : edges) {
            if (placed[e.i] == placed[e.j]) continue;
            if (!best || e.inliers > best->inliers) {
                best = &e;
                fromI = placed[e.i];
            }
        }
        if (!best) break;
        if (fromI) R[best->j] = R[best->i] * best->Rji.t();  // (Rji = Rj^T Ri)
        else R[best->i] = R[best->j] * best->Rji;
        placed[fromI ? best->j : best->i] = true;
        placedCount++;
    }
    // ---- second chance: a photo no triangle vouched for is tried where its neighbours in time put it. From each of the
    // nearest placed photos before and after it: a search over yaw and pitch of the whole overlap at 1/16 scale by
    // correlation (a plain overlap scores nothing), then the tracker's refinement up to 1/4 scale. It is placed when
    // both neighbours agree within 3 deg, or one correlates strongly. (GCam had the gyro for this.)
    auto correlation = [&](const Small &k, const Small &a, const Matx33d &Rak, int &inside) {
        double sk = 0, sa = 0, skk = 0, saa = 0, ska = 0;
        int total = 0;
        inside = 0;
        for (int y = 2; y < k.g.rows - 2; y += 2)
            for (int x = 2; x < k.g.cols - 2; x += 2) {
                total++;
                Vec3d d = Rak * Vec3d((x - k.cx) / k.f, (y - k.cy) / k.f, 1);
                if (d[2] <= 0.1) continue;
                double qx = a.f * d[0] / d[2] + a.cx, qy = a.f * d[1] / d[2] + a.cy;
                if (qx < 1 || qy < 1 || qx >= a.g.cols - 2 || qy >= a.g.rows - 2) continue;
                double vk = k.g.at<float>(y, x), va = bilinear(a.g, qx, qy);
                sk += vk;
                sa += va;
                skk += vk * vk;
                saa += va * va;
                ska += vk * va;
                inside++;
            }
        if (inside < 0.15 * total || inside < 60) return -1.0;
        double mk = sk / inside, ma = sa / inside;
        double vk = skk / inside - mk * mk, va = saa / inside - ma * ma;
        if (vk < 64 || va < 64) return -1.0;  // (plain: nothing to correlate)
        return (ska / inside - mk * ma) / std::sqrt(vk * va);
    };
    int secondChance = 0;
    for (int k = 0; k < n; k++) {
        if (placed[k]) continue;
        std::vector<int> nb;
        for (int d = 1; d <= 3; d++) {
            if (k - d >= 0 && placed[k - d]) {
                nb.push_back(k - d);
                break;
            }
        }
        for (int d = 1; d <= 3; d++) {
            if (k + d < n && placed[k + d]) {
                nb.push_back(k + d);
                break;
            }
        }
        std::vector<std::pair<Matx33d, double>> found;  // (world-from-camera, correlation at 1/4)
        char buf[200];
        for (int a : nb) {
            double yaw0 = yawOf(R[a]), pitch0 = pitchOf(R[a]), roll0 = rollOf(R[a]);
            double bestC = -1, bestYaw = yaw0, bestPitch = pitch0, bestRoll = roll0;
            auto tryPose = [&](const Small &sk, const Small &sa, double yaw, double pitch, double roll) {
                pitch = std::max(-kPi / 2, std::min(kPi / 2, pitch));
                Matx33d Rak = R[a].t() * lookAt(yaw, pitch, roll);
                int inside;
                double c = correlation(sk, sa, Rak, inside);
                if (c > bestC) {
                    bestC = c;
                    bestYaw = yaw;
                    bestPitch = pitch;
                    bestRoll = roll;
                }
            };
            for (int iy = -15; iy <= 15; iy++)  // coarse: yaw and pitch at 1/16, the neighbour's roll
                for (int ip = -8; ip <= 8; ip++) tryPose(s16[k], s16[a], yaw0 + rad(6 * iy), pitch0 + rad(6 * ip), roll0);
            double c16 = bestC;
            snprintf(buf, sizeof buf, "#%d via #%d: corr %.2f", k, a, c16);
            out.log += buf;
            if (c16 < 0.3) {
                out.log += " (too low); ";
                continue;
            }
            double y1 = bestYaw, p1 = bestPitch, r1 = bestRoll;  // fine: with roll, at 1/8
            bestC = -1;
            for (int iy = -4; iy <= 4; iy++)
                for (int ip = -4; ip <= 4; ip++)
                    for (int ir = -3; ir <= 3; ir++) tryPose(s8[k], s8[a], y1 + rad(1.5 * iy), p1 + rad(1.5 * ip), r1 + rad(3 * ir));
            Matx33d Rak = R[a].t() * lookAt(bestYaw, bestPitch, bestRoll);
            int level = 0, pts = 0, inside = 0;
            double before = 0, after = 0;
            for (int lv = 0; lv < 2; lv++) {  // the tracker's refinement at 1/8 and 1/4, kept when it lowered the error
                const Small &sk = lv == 0 ? s8[k] : s4[k], &sa = lv == 0 ? s8[a] : s4[a];
                Matx33d trial = Rak;
                bool fine = refineDirect(sk, sa, trial, before, after, pts);
                if (fine || (pts >= 150 && after < before)) {
                    Rak = trial;
                    level++;
                }
            }
            double c = correlation(s4[k], s4[a], Rak, inside);
            snprintf(buf, sizeof buf, ", fine %.2f, refined %d/2 (%.0f->%.0f, %d pts), corr %.2f at 1/4; ", bestC, level, before, after, pts, c);
            out.log += buf;
            if (c >= 0.6) found.push_back({R[a] * Rak, c});
        }
        bool ok = false;
        Matx33d Rk;
        if (found.size() == 2 && angleOf(found[0].first * found[1].first.t()) < rad(3)) {
            ok = true;
            Rk = found[0].second >= found[1].second ? found[0].first : found[1].first;
        } else {
            double bestC = 0.8;  // (one neighbour alone must correlate strongly)
            for (auto &f : found)
                if (f.second >= bestC) {
                    ok = true;
                    bestC = f.second;
                    Rk = f.first;
                }
        }
        if (ok) {
            R[k] = Rk;
            placed[k] = true;
            secondChance++;
            snprintf(buf, sizeof buf, "#%d placed (yaw %.0f pitch %.0f). ", k, deg(yawOf(Rk)), deg(pitchOf(Rk)));
            out.log += buf;
        } else if (!nb.empty()) {
            snprintf(buf, sizeof buf, "#%d not placed. ", k);
            out.log += buf;
        }
    }
    out.secondChance = secondChance;
    // "up": perpendicular to every placed photo's right vector; the world's y points down, z along the root's forward
    Mat Mx = Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < n; i++) {
        if (!placed[i]) continue;
        Vec3d x(R[i](0, 0), R[i](1, 0), R[i](2, 0));
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) Mx.at<double>(r, c) += x[r] * x[c];
    }
    Mat evals, evecs;
    cv::eigen(Mx, evals, evecs);
    Vec3d up(evecs.at<double>(2, 0), evecs.at<double>(2, 1), evecs.at<double>(2, 2));
    double sign = 0;
    for (int i = 0; i < n; i++)
        if (placed[i]) sign += up.dot(Vec3d(-R[i](0, 1), -R[i](1, 1), -R[i](2, 1)));  // (the cameras' own up: -y)
    if (sign < 0) up = -up;
    Vec3d ey = -up, f0 = forwardOf(R[root]);
    Vec3d ez = unit(f0 - ey * f0.dot(ey));
    if (cv::norm(ez) < 1e-6) ez = unit(Vec3d(R[root](0, 1), R[root](1, 1), R[root](2, 1)).cross(ey));
    Vec3d ex = ey.cross(ez);
    Matx33d G(ex[0], ex[1], ex[2], ey[0], ey[1], ey[2], ez[0], ez[1], ez[2]);
    for (int i = 0; i < n; i++) {
        if (!placed[i]) continue;
        out.placed.push_back(i);
        out.R.push_back(G * R[i]);
    }
    out.edges = (int)edges.size();
    out.root = root;
    report(1.0, "placed");
    return out;
}

// ---------------------------------------------------------------- the self test's worlds and metrics

/// A synthetic world ("textured": shapes everywhere; "room": plain walls with a few things on them) photographed at
/// each (yaw, pitch) with exposure changes, noise, JPEG, and sensor rotations perturbed by random error and a yaw drift.
inline Synth generate(const std::string &dir, const std::vector<double> &yaws, const std::vector<double> &pitches, double fovDeg,
                      int width, int height, int panoWidth, const std::string &world) {
    makeDirs(dir);
    std::mt19937 rng(11);
    Equirect eq(panoWidth);
    bool room = world == "room";
    Mat pano(eq.H, eq.W, CV_8UC3);
    std::uniform_real_distribution<double> U(0, 1);
    if (room) {  // a plain room: pale walls with soft shading, a ceiling, a wooden floor, a few things on the walls
        for (int y = 0; y < eq.H; y++) {
            double t = (double)y / eq.H;
            for (int x = 0; x < eq.W; x++) {
                double wall = 0.5 + 0.5 * std::cos(x * 8 * kPi / eq.W);  // (four walls, lit a little differently)
                cv::Vec3b c;
                if (t < 0.22) c = cv::Vec3b(236 - 12 * t / 0.22, 238 - 10 * t / 0.22, 240);
                else if (t > 0.72) c = cv::Vec3b(80 + 20 * (t - 0.72), 120 + 20 * (t - 0.72), 160 + 10 * (t - 0.72));
                else c = cv::Vec3b(205 + 18 * wall - 25 * (t - 0.22), 208 + 16 * wall - 25 * (t - 0.22), 210 + 12 * wall - 25 * (t - 0.22));
                pano.at<cv::Vec3b>(y, x) = c;
            }
        }
        for (int x = 0; x < eq.W; x += eq.W / 64) cv::line(pano, cv::Point(x, (int)(eq.H * 0.72)), cv::Point(x + eq.W / 32, eq.H - 1), cv::Scalar(70, 105, 140), 1);  // (floorboards)
        {  // (the faint texture of paint, and a few fixtures, as real ceilings and walls have)
            Mat speck(pano.size(), CV_8SC3);
            cv::randn(speck, 0, 3);
            Mat top = pano(cv::Rect(0, 0, eq.W, (int)(eq.H * 0.72)));
            cv::add(top, speck(cv::Rect(0, 0, eq.W, (int)(eq.H * 0.72))), top, cv::noArray(), CV_8UC3);
            for (int k = 0; k < 6; k++) {
                cv::Point c((int)(U(rng) * eq.W), (int)(eq.H * (0.03 + 0.14 * U(rng))));
                cv::circle(pano, c, (int)(eq.W * (0.004 + 0.01 * U(rng))), cv::Scalar(200, 200, 205), -1, cv::LINE_AA);
                cv::circle(pano, c, (int)(eq.W * (0.004 + 0.01 * U(rng))), cv::Scalar(150, 150, 150), 2, cv::LINE_AA);
            }
        }
        cv::line(pano, cv::Point(0, (int)(eq.H * 0.22)), cv::Point(eq.W - 1, (int)(eq.H * 0.22)), cv::Scalar(150, 150, 150), 2);
        cv::line(pano, cv::Point(0, (int)(eq.H * 0.72)), cv::Point(eq.W - 1, (int)(eq.H * 0.72)), cv::Scalar(90, 90, 90), 3);
        for (int k = 0; k < 28; k++) {  // (pictures, a door, a window, switches: the few things to match)
            cv::Point c((int)(U(rng) * eq.W), (int)(eq.H * (0.3 + 0.35 * U(rng))));
            int w = (int)(eq.W * (0.01 + 0.04 * U(rng))), h = (int)(eq.H * (0.02 + 0.08 * U(rng)));
            cv::Scalar col(90 + 120 * U(rng), 90 + 120 * U(rng), 90 + 120 * U(rng));
            cv::rectangle(pano, c - cv::Point(w, h), c + cv::Point(w, h), col, -1);
            cv::rectangle(pano, c - cv::Point(w, h), c + cv::Point(w, h), cv::Scalar(60, 60, 60), 2);
        }
        for (int k = 0; k < 10; k++) {
            cv::Point c((int)(U(rng) * eq.W), (int)(eq.H * (0.74 + 0.15 * U(rng))));
            cv::ellipse(pano, c, cv::Size((int)(eq.W * 0.01 + U(rng) * eq.W * 0.02), (int)(eq.H * 0.01 + U(rng) * eq.H * 0.02)), 0, 0, 360, cv::Scalar(40 + 60 * U(rng), 40 + 60 * U(rng), 40 + 60 * U(rng)), -1);
        }
    } else {
        // a textured world: sky and ground, a horizon, thousands of shapes and lines, labels at the headings and poles
        for (int y = 0; y < eq.H; y++) {
            double t = (double)y / eq.H;
            cv::Vec3b c = t < 0.5 ? cv::Vec3b(230 - 120 * t, 170 + 60 * t, 120 + 40 * t) : cv::Vec3b(60 + 40 * t, 110 + 50 * t, 90 + 30 * t);
            pano.row(y).setTo(c);
        }
    }
    for (int k = 0; k < (room ? 0 : 4000); k++) {
        cv::Point c((int)(U(rng) * eq.W), (int)(U(rng) * eq.H));
        cv::Scalar col(U(rng) * 255, U(rng) * 255, U(rng) * 255);
        int r = 4 + (int)(U(rng) * eq.W / 60);
        if (U(rng) < 0.5)
            cv::circle(pano, c, r, col, -1, cv::LINE_AA);
        else
            cv::rectangle(pano, c - cv::Point(r, r / 2), c + cv::Point(r, r / 2), col, -1);
    }
    for (int k = 0; k < (room ? 0 : 600); k++) {
        cv::Point a((int)(U(rng) * eq.W), (int)(U(rng) * eq.H));
        cv::Point b = a + cv::Point((int)((U(rng) - 0.5) * eq.W / 8), (int)((U(rng) - 0.5) * eq.H / 8));
        cv::line(pano, a, b, cv::Scalar(U(rng) * 255, U(rng) * 255, U(rng) * 255), 1 + (int)(U(rng) * 3), cv::LINE_AA);
    }
    if (!room) cv::line(pano, cv::Point(0, eq.H / 2), cv::Point(eq.W - 1, eq.H / 2), cv::Scalar(20, 20, 20), 2);
    const char *labels[] = {"FRONT", "RIGHT", "BACK", "LEFT"};
    for (int k = 0; k < 4; k++) {
        int x = (int)(eq.W * (0.5 + 0.25 * k)) % eq.W;
        cv::putText(pano, labels[k], cv::Point(x - 60, eq.H / 2 - 20), cv::FONT_HERSHEY_SIMPLEX, eq.W / 1500.0, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    }
    cv::putText(pano, "UP", cv::Point(eq.W / 2 - 30, eq.H / 12), cv::FONT_HERSHEY_SIMPLEX, eq.W / 1500.0, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(pano, "DOWN", cv::Point(eq.W / 2 - 60, eq.H - eq.H / 12), cv::FONT_HERSHEY_SIMPLEX, eq.W / 1500.0, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::GaussianBlur(pano, pano, cv::Size(3, 3), 0);
    Synth out;
    out.pano = dir + "/truth.jpg";
    cv::imwrite(out.pano, pano, {cv::IMWRITE_JPEG_QUALITY, 95});
    Mat padded;
    cv::copyMakeBorder(pano, padded, 0, 0, 0, 1, cv::BORDER_WRAP);  // (bilinear sampling across the +-180 edge)

    double f = (std::max(width, height) / 2.0) / std::tan(rad(fovDeg) / 2), cx = (width - 1) / 2.0, cy = (height - 1) / 2.0;
    std::normal_distribution<double> N01(0, 1);
    Mat mapx(height, width, CV_32F), mapy(height, width, CV_32F);
    for (int v = 0; v < (int)yaws.size(); v++) {
        double roll = rad(N01(rng) * 2.0);
        Matx33d Rt = lookAt(yaws[v], pitches[v], roll);
        for (int y = 0; y < height; y++) {
            float *mx = mapx.ptr<float>(y), *my = mapy.ptr<float>(y);
            for (int x = 0; x < width; x++) {
                Point2d p = eq.pixel(Rt * rayOf(x, y, f, cx, cy));
                mx[x] = (float)p.x;
                my[x] = (float)p.y;
            }
        }
        Mat view;
        cv::remap(padded, view, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        double gain = 0.85 + 0.3 * U(rng);
        view.convertTo(view, -1, gain, 0);
        Mat noise(view.size(), CV_8SC3);
        cv::randn(noise, 0, 2.5);
        cv::add(view, noise, view, cv::noArray(), CV_8UC3);
        std::string path = dir + "/" + std::to_string(v) + ".jpg";
        cv::imwrite(path, view, {cv::IMWRITE_JPEG_QUALITY, 92});
        // the sensors' idea of it: a random error of about 0.4 degrees (a phone held still, gravity in hand), and a yaw
        // drift growing 0.25 degrees a photo
        Vec3d axis = unit(Vec3d(N01(rng), N01(rng), N01(rng)));
        Matx33d Rp = rotY(rad(0.25 * v)) * expSO3(axis * rad(std::fabs(N01(rng)) * 0.4)) * Rt;
        out.images.push_back(path);
        out.priors.push_back(Rp);
        out.truth.push_back(Rt);
    }
    out.focalPx = f;
    out.width = width;
    out.height = height;
    return out;
}

/// How well a stitch of a synthetic set did: the photos' rotation errors after the best global rotation, and the PSNR
/// against the truth seen in the stitched frame, where covered.
inline Metrics metrics(const StitchResult &result, const std::string &truthPath, const std::vector<Matx33d> &truth) {
    Metrics m;
    int n = (int)truth.size();
    if ((int)result.R.size() != n || n == 0) return m;
    Mat M = Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < n; i++)
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                for (int k = 0; k < 3; k++) M.at<double>(r, c) += result.R[i](r, k) * truth[i](c, k);
    cv::SVD svd(M, cv::SVD::FULL_UV);
    Mat D = Mat::eye(3, 3, CV_64F);
    D.at<double>(2, 2) = cv::determinant(svd.u * svd.vt) < 0 ? -1 : 1;
    Matx33d G(Mat(svd.u * D * svd.vt));
    double sum2 = 0, worst = 0;
    for (int i = 0; i < n; i++) {
        double e = deg(angleOf(G * truth[i] * result.R[i].t()));
        m.perPhotoDeg.push_back(e);
        sum2 += e * e;
        worst = std::max(worst, e);
    }
    m.rotationRmsDeg = std::sqrt(sum2 / n);
    m.rotationMaxDeg = worst;
    Mat out = cv::imread(result.path, cv::IMREAD_COLOR), tr = cv::imread(truthPath, cv::IMREAD_COLOR);
    if (!out.empty() && !tr.empty()) {
        Equirect eq(result.fullW), eqT(tr.cols);
        Mat padded;
        cv::copyMakeBorder(tr, padded, 0, 0, 0, 1, cv::BORDER_WRAP);
        Mat mapx(out.rows, out.cols, CV_32F), mapy(out.rows, out.cols, CV_32F);
        Matx33d Gt = G.t();
        for (int y = 0; y < out.rows; y++) {
            float *mx = mapx.ptr<float>(y), *my = mapy.ptr<float>(y);
            for (int x = 0; x < out.cols; x++) {
                Point2d p = eqT.pixel(Gt * eq.ray(x + result.left, y + result.top));
                mx[x] = (float)p.x;
                my[x] = (float)p.y;
            }
        }
        Mat ref;
        cv::remap(padded, ref, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        Mat gray, covered;
        cv::cvtColor(out, gray, cv::COLOR_BGR2GRAY);
        covered = gray > 0;
        cv::erode(covered, covered, cv::Mat(), cv::Point(-1, -1), 3);
        Mat diff;
        cv::absdiff(out, ref, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);
        cv::Scalar mean = cv::mean(diff, covered);
        double mse = (mean[0] + mean[1] + mean[2]) / 3.0;
        m.psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99;
    }
    m.pass = m.rotationRmsDeg < 0.5 && m.rotationMaxDeg < 1.5 && m.psnr > 20;
    return m;
}

}  // namespace pano
