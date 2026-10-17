// The Photo Sphere pipeline in C++ on OpenCV, following Google Camera 4.1's liblightcycle stage by stage (the constants
// are theirs where they're known): see README.md for the map from their classes to this file.
#import <opencv2/opencv.hpp>
#import <opencv2/stitching.hpp>
#import <Foundation/Foundation.h>
#import "PanoStitcher.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using cv::Mat;
using cv::Matx33d;
using cv::Point2d;
using cv::Vec3d;

namespace {

constexpr double kPi = CV_PI;
constexpr int kMatchWidth = 1600;        // the photos are matched at this size (GCam: image_match_width)
constexpr double kHuberPx = 35.0;        // the match residual's Huber scale, in match pixels (GCam: HuberLoss(35))
constexpr double kRansacInlierDeg = 1.1; // a match is an inlier of a pairwise rotation within this (GCam: 1.1 deg)
constexpr int kRansacMin = 550, kRansacMax = 5000, kRansacEnough = 150;
constexpr int kKeepMatches = 35;         // inliers kept per pair for the global adjustment (GCam: 35, stratified)
constexpr double kRatio = 0.8;           // Lowe's ratio (GCam: 0.64 on squared distances)
constexpr double kAxesNear = 0.9;        // two photos are candidates when their axes are within this times the FOV
constexpr double kLatGateDeg = 9.0;      // and a match's two rays must lie within this latitude band (GCam: 6 deg bins +-1)

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

struct View {
    int w = 0, h = 0;                // at match scale
    double scale = 1;                // match pixels per photo pixel
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
    bool sensor = false;
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
    double f0 = 1, cx = 0, cy = 0;   // match-scale intrinsics
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
            Vec3d d = Rji * Vec3d((pm.pi[k].x - P.cx) / f, (pm.pi[k].y - P.cy) / f, 1);
            double z = std::max(d[2], 0.05);  // (a point behind the camera: a large residual, bounded by Huber)
            double w = pm.w * (k < pm.wk.size() ? pm.wk[k] : 1.0);
            r.push_back(w * (f * d[0] / z + P.cx - pm.pj[k].x));
            r.push_back(w * (f * d[1] / z + P.cy - pm.pj[k].y));
        }
    }
    for (int i = 0; i < P.n; i++) {  // the gravity priors: tilt, and roll with half the weight (GCam: SensorResidual, RollPitchSensorResidual)
        r.push_back(P.ws * f * (pitchOf(R[i]) - P.pitch0[i]));
        r.push_back(std::fabs(P.pitch0[i]) < rad(81) ? 0.5 * P.ws * f * (rollOf(R[i]) - P.roll0[i]) : 0.0);
    }
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
        for (int i = 0; i < n; i++) gammaOut[i] = std::max(1 / 1.75, std::min(1.75, std::isfinite(g.at<double>(i)) ? g.at<double>(i) : 1.0));
    return gammaOut;
}

void applyGamma(Mat &img, double g) {
    if (std::fabs(g - 1) < 1e-3) return;
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) lut.at<uchar>(i) = (uchar)std::lround(std::pow(i / 255.0, g) * 255.0);
    cv::LUT(img, lut, img);
}

/// GCam's hole treatment: what no photo covered is filled from the coarsest blend level when small (under 0.5% of the
/// canvas), left black when large. Here: small holes take the colour of their blurred surroundings.
void fillHoles(Mat &result, const Mat &coverage) {
    Mat holes = coverage == 0, labels, stats, centroids;
    int nc = cv::connectedComponentsWithStats(holes, labels, stats, centroids, 8);
    if (nc <= 1) return;
    double limit = 0.005 * result.cols * result.rows;
    Mat small = Mat::zeros(holes.size(), CV_8U);
    for (int c = 1; c < nc; c++)
        if (stats.at<int>(c, cv::CC_STAT_AREA) < limit) small.setTo(255, labels == c);
    if (cv::countNonZero(small) == 0) return;
    int s = 8;
    Mat img, msk;
    cv::resize(result, img, cv::Size(result.cols / s, result.rows / s), 0, 0, cv::INTER_AREA);
    cv::resize(coverage, msk, img.size(), 0, 0, cv::INTER_NEAREST);
    img.convertTo(img, CV_32FC3);
    Mat m;
    msk.convertTo(m, CV_32F, 1.0 / 255);
    for (int it = 0; it < 40; it++) {  // (the covered colours spread outward, a few pixels an iteration)
        Mat wimg, wm;
        std::vector<Mat> ch(3);
        cv::split(img, ch);
        for (auto &c : ch) {
            cv::multiply(c, m, c);
            cv::GaussianBlur(c, c, cv::Size(5, 5), 0);
        }
        cv::merge(ch, wimg);
        cv::GaussianBlur(m, wm, cv::Size(5, 5), 0);
        Mat fillable = (m == 0) & (wm > 1e-3);
        if (cv::countNonZero(fillable) == 0) break;
        std::vector<Mat> wch(3);
        cv::split(wimg, wch);
        for (auto &c : wch) cv::divide(c, wm, c);
        Mat filled;
        cv::merge(wch, filled);
        filled.copyTo(img, fillable);
        m.setTo(1, fillable);
    }
    Mat up;
    cv::resize(img, up, result.size(), 0, 0, cv::INTER_LINEAR);
    up.convertTo(up, CV_8UC3);
    up.copyTo(result, small);
}

NSError *err(NSString *msg) { return [NSError errorWithDomain:@"PanoStitcher" code:1 userInfo:@{NSLocalizedDescriptionKey : msg}]; }

Matx33d matrixFrom(NSArray<NSNumber *> *a) {
    Matx33d R;
    for (int k = 0; k < 9; k++) R(k / 3, k % 3) = a[k].doubleValue;
    return R;
}

NSArray<NSNumber *> *arrayFrom(const Matx33d &R) {
    NSMutableArray *a = [NSMutableArray array];
    for (int k = 0; k < 9; k++) [a addObject:@(R(k / 3, k % 3))];
    return a;
}

}  // namespace

@implementation PanoStitchResult
@end

@implementation PanoStitcher

+ (nullable PanoStitchResult *)stitchImages:(NSArray<NSString *> *)paths
                                  rotations:(NSArray<NSArray<NSNumber *> *> *)rotations
                                    focalPx:(double)focalPx
                                outputWidth:(int)outputWidth
                                 outputPath:(NSString *)outputPath
                                   progress:(nullable PanoProgress)progress
                                      error:(NSError **)error {
    auto report = [&](double frac, NSString *stage) {
        if (progress) progress(frac, stage);
    };
    double t0 = (double)cv::getTickCount();
    int n = (int)paths.count;
    if (n < 1 || rotations.count != paths.count) {
        if (error) *error = err(@"no photos");
        return nil;
    }
    std::mt19937 rng(7);
    try {

    // ---- features (GCam: FastCornerDetector + OrientedPatchExtractor; here ORB: FAST corners with oriented binary patches)
    std::vector<View> views(n);
    std::vector<Mat> photos(n);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10);  // (a low FAST threshold: plain walls)
    double matchScale = 1, fm = focalPx, cxm = 0, cym = 0;
    for (int i = 0; i < n; i++) {
        Mat img = cv::imread(paths[i].UTF8String, cv::IMREAD_COLOR);
        if (img.empty()) {
            if (error) *error = err([NSString stringWithFormat:@"can't read %@", paths[i]]);
            return nil;
        }
        if (i == 0) {
            matchScale = std::min(1.0, (double)kMatchWidth / std::max(img.cols, img.rows));
            fm = focalPx * matchScale;
        }
        photos[i] = img;
        View &v = views[i];
        v.scale = matchScale;
        v.Rp = matrixFrom(rotations[i]);
        Mat small, gray;
        cv::resize(img, small, cv::Size(), matchScale, matchScale, cv::INTER_AREA);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        v.w = gray.cols;
        v.h = gray.rows;
        cxm = (v.w - 1) / 2.0;
        cym = (v.h - 1) / 2.0;
        orb->detectAndCompute(gray, cv::noArray(), v.kps, v.desc);
        v.rays.resize(v.kps.size());
        v.lat.resize(v.kps.size());
        for (size_t k = 0; k < v.kps.size(); k++) {
            v.rays[k] = rayOf(v.kps[k].pt.x, v.kps[k].pt.y, fm, cxm, cym);
            v.lat[k] = std::asin(clamp1(-(v.Rp * v.rays[k])[1]));
        }
        report(0.25 * (i + 1) / n, @"finding features");
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
            report(0.25 + 0.25 * (++seen) / std::max(1, candidates), @"matching");
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
            Matx33d Rsens = b.Rp.t() * a.Rp, E = Rvis * Rsens.t();
            double yawTol = rad(std::min(25.0 * (j - i), 65.0));
            Vec3d upj = b.Rp.t() * Vec3d(0, -1, 0);
            if (angleOf(E) > yawTol || angleBetween(E * upj, upj) > rad(12.5)) continue;  // (GEO_FAIL: not what the sensors say)
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
    int imagePairs = (int)pairs.size();
    int keptPoints = 0;
    for (auto &p : pairs) keptPoints += (int)p.pi.size();
    // sensor pairs: a weak tie between each photo and the one before when no image match links them (GCam: SENSOR_PAIR,
    // a 3x3 grid transferred through the sensors, weight 0.125 or 0.25 when the axes are near)
    for (int i = 1; i < n; i++) {
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
                Vec3d d = Rji * Vec3d((p.x - cxm) / fm, (p.y - cym) / fm, 1);
                if (d[2] <= 0.05) continue;
                Point2d q(fm * d[0] / d[2] + cxm, fm * d[1] / d[2] + cym);
                pm.pi.push_back(p);
                pm.pj.push_back(q);
            }
        if (!pm.pi.empty()) pairs.push_back(pm);
    }

    // ---- the global adjustment: every photo's rotation and one focal length, matches robust, tilt and roll pulled to
    // the sensors; yaw is free and re-anchored to the first photo's afterwards (GCam's gauge)
    report(0.5, @"adjusting");
    BAProblem P;
    P.n = n;
    P.pairs = &pairs;
    P.f0 = fm;
    P.cx = cxm;
    P.cy = cym;
    P.ws = keptPoints > 0 ? 0.0125 * keptPoints / n : 0.5;
    for (int i = 0; i < n; i++) {
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
    } else {
        R = P.R0;
    }
    double fFull = fAdj / matchScale;

    // ---- render: each photo warped onto the canvas, exposure matched, seams cut, blended in bands
    Equirect eq(outputWidth);
    double composeScale = std::min(1.0, 1.3 * (eq.W / (2 * kPi)) / fFull);  // (sources no more than 1.3x the output's resolution)
    std::vector<Piece> pieces;
    Mat coverage = Mat::zeros(eq.H, eq.W, CV_8U);
    report(0.5, @"matching exposure");
    std::vector<Mat> smalls(n);
    for (int i = 0; i < n; i++) cv::resize(photos[i], smalls[i], cv::Size(), matchScale * 0.25, matchScale * 0.25, cv::INTER_AREA);
    std::vector<double> gammas;
    matchExposure(smalls, R, fm * 0.25, gammas);
    smalls.clear();
    for (int i = 0; i < n; i++) {
        Mat src = photos[i];
        if (composeScale < 0.999) cv::resize(photos[i], src, cv::Size(), composeScale, composeScale, cv::INTER_AREA);
        applyGamma(src, gammas[i]);
        double f = fFull * composeScale, cx = (src.cols - 1) / 2.0, cy = (src.rows - 1) / 2.0;
        size_t before = pieces.size();
        warpPhoto(src, f, cx, cy, R[i], eq, i, pieces);
        for (size_t k = before; k < pieces.size(); k++) {
            Mat roi = coverage(cv::Rect(pieces[k].tl, pieces[k].mask.size()));
            cv::bitwise_or(roi, pieces[k].mask, roi);
        }
        photos[i].release();
        report(0.5 + 0.2 * (i + 1) / n, @"warping");
    }
    if (pieces.empty()) {
        if (error) *error = err(@"nothing to render");
        return nil;
    }
    int np = (int)pieces.size();
    std::vector<cv::Point> corners(np);
    std::vector<cv::UMat> imgs(np), masks(np);
    for (int k = 0; k < np; k++) {
        corners[k] = pieces[k].tl;
        pieces[k].img.copyTo(imgs[k]);
        pieces[k].mask.copyTo(masks[k]);
    }
    report(0.76, @"finding seams");
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
    report(0.84, @"blending");
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
        report(0.84 + 0.1 * (k + 1) / np, @"blending");
    }
    Mat result16, resultMask;
    blender.blend(result16, resultMask);
    Mat result;
    result16(cv::Rect(pad, 0, eq.W, eq.H)).convertTo(result, CV_8U);
    result.setTo(cv::Scalar(0, 0, 0), coverage == 0);
    fillHoles(result, coverage);  // (small holes filled from their surroundings, large ones black, as GCam)

    // ---- crop: a full circle keeps its width and loses rows not fully covered (10 deg at most each end); a partial
    // sphere is cut to what's covered
    report(0.95, @"saving");
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
    std::vector<uchar> buf;
    cv::imencode(".jpg", cropped, buf, {cv::IMWRITE_JPEG_QUALITY, 90});
    NSData *data = [NSData dataWithBytes:buf.data() length:buf.size()];
    if (![data writeToFile:outputPath atomically:YES]) {
        if (error) *error = err(@"can't write the panorama");
        return nil;
    }
    PanoStitchResult *res = [PanoStitchResult new];
    res.path = outputPath;
    res.fullWidth = eq.W;
    res.fullHeight = eq.H;
    res.cropWidth = cropped.cols;
    res.cropHeight = cropped.rows;
    res.cropLeft = left;
    res.cropTop = top;
    res.fullCircle = full;
    res.imagesUsed = n;
    res.pairsMatched = imagePairs;
    res.focalPx = fFull;
    res.seconds = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    NSMutableArray *rots = [NSMutableArray array];
    for (auto &Ri : R) [rots addObject:arrayFrom(Ri)];
    res.rotations = rots;
    report(1.0, @"done");
    return res;

    } catch (const std::exception &e) {
        if (error) *error = err([NSString stringWithFormat:@"stitching failed: %s", e.what()]);
        return nil;
    }
}

// ---------------------------------------------------------------- the self test

+ (nullable NSDictionary *)selfTestGenerateInDirectory:(NSString *)dir
                                                  yaws:(NSArray<NSNumber *> *)yaws
                                               pitches:(NSArray<NSNumber *> *)pitches
                                                fovDeg:(double)fovDeg
                                                 width:(int)width
                                                height:(int)height
                                             panoWidth:(int)panoWidth {
    [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
    std::mt19937 rng(11);
    Equirect eq(panoWidth);
    // a textured world: sky and ground, a horizon, thousands of shapes and lines, labels at the headings and poles
    Mat pano(eq.H, eq.W, CV_8UC3);
    for (int y = 0; y < eq.H; y++) {
        double t = (double)y / eq.H;
        cv::Vec3b c = t < 0.5 ? cv::Vec3b(230 - 120 * t, 170 + 60 * t, 120 + 40 * t) : cv::Vec3b(60 + 40 * t, 110 + 50 * t, 90 + 30 * t);
        pano.row(y).setTo(c);
    }
    std::uniform_real_distribution<double> U(0, 1);
    for (int k = 0; k < 4000; k++) {
        cv::Point c((int)(U(rng) * eq.W), (int)(U(rng) * eq.H));
        cv::Scalar col(U(rng) * 255, U(rng) * 255, U(rng) * 255);
        int r = 4 + (int)(U(rng) * eq.W / 60);
        if (U(rng) < 0.5)
            cv::circle(pano, c, r, col, -1, cv::LINE_AA);
        else
            cv::rectangle(pano, c - cv::Point(r, r / 2), c + cv::Point(r, r / 2), col, -1);
    }
    for (int k = 0; k < 600; k++) {
        cv::Point a((int)(U(rng) * eq.W), (int)(U(rng) * eq.H));
        cv::Point b = a + cv::Point((int)((U(rng) - 0.5) * eq.W / 8), (int)((U(rng) - 0.5) * eq.H / 8));
        cv::line(pano, a, b, cv::Scalar(U(rng) * 255, U(rng) * 255, U(rng) * 255), 1 + (int)(U(rng) * 3), cv::LINE_AA);
    }
    cv::line(pano, cv::Point(0, eq.H / 2), cv::Point(eq.W - 1, eq.H / 2), cv::Scalar(20, 20, 20), 2);
    const char *labels[] = {"FRONT", "RIGHT", "BACK", "LEFT"};
    for (int k = 0; k < 4; k++) {
        int x = (int)(eq.W * (0.5 + 0.25 * k)) % eq.W;
        cv::putText(pano, labels[k], cv::Point(x - 60, eq.H / 2 - 20), cv::FONT_HERSHEY_SIMPLEX, eq.W / 1500.0, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    }
    cv::putText(pano, "UP", cv::Point(eq.W / 2 - 30, eq.H / 12), cv::FONT_HERSHEY_SIMPLEX, eq.W / 1500.0, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(pano, "DOWN", cv::Point(eq.W / 2 - 60, eq.H - eq.H / 12), cv::FONT_HERSHEY_SIMPLEX, eq.W / 1500.0, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::GaussianBlur(pano, pano, cv::Size(3, 3), 0);
    NSString *panoPath = [dir stringByAppendingPathComponent:@"truth.jpg"];
    cv::imwrite(panoPath.UTF8String, pano, {cv::IMWRITE_JPEG_QUALITY, 95});
    Mat padded;
    cv::copyMakeBorder(pano, padded, 0, 0, 0, 1, cv::BORDER_WRAP);  // (bilinear sampling across the +-180 edge)

    double f = (std::max(width, height) / 2.0) / std::tan(rad(fovDeg) / 2), cx = (width - 1) / 2.0, cy = (height - 1) / 2.0;
    std::normal_distribution<double> N01(0, 1);
    NSMutableArray *images = [NSMutableArray array], *priors = [NSMutableArray array], *truth = [NSMutableArray array];
    Mat mapx(height, width, CV_32F), mapy(height, width, CV_32F);
    for (int v = 0; v < (int)yaws.count; v++) {
        double roll = rad(N01(rng) * 2.0);
        Matx33d Rt = lookAt(yaws[v].doubleValue, pitches[v].doubleValue, roll);
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
        NSString *path = [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"%d.jpg", v]];
        cv::imwrite(path.UTF8String, view, {cv::IMWRITE_JPEG_QUALITY, 92});
        // the sensors' idea of it: a random error of about 1.2 degrees, and a yaw drift growing 0.25 degrees a photo
        Vec3d axis = unit(Vec3d(N01(rng), N01(rng), N01(rng)));
        Matx33d Rp = rotY(rad(0.25 * v)) * expSO3(axis * rad(std::fabs(N01(rng)) * 1.2)) * Rt;
        [images addObject:path];
        [priors addObject:arrayFrom(Rp)];
        [truth addObject:arrayFrom(Rt)];
    }
    return @{@"images" : images, @"rotations" : priors, @"truth" : truth, @"focalPx" : @(f), @"width" : @(width), @"height" : @(height), @"pano" : panoPath};
}

+ (NSDictionary *)selfTestMetricsForResult:(PanoStitchResult *)result
                                 truthPano:(NSString *)truthPath
                            truthRotations:(NSArray<NSArray<NSNumber *> *> *)truth {
    int n = (int)truth.count;
    std::vector<Matx33d> Rt(n), Re(n);
    for (int i = 0; i < n; i++) {
        Rt[i] = matrixFrom(truth[i]);
        Re[i] = matrixFrom(result.rotations[i]);
    }
    // the best global rotation G with Re ~ G * Rt, then each photo's error after it
    Mat M = Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < n; i++)
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                for (int k = 0; k < 3; k++) M.at<double>(r, c) += Re[i](r, k) * Rt[i](c, k);
    cv::SVD svd(M, cv::SVD::FULL_UV);
    Mat D = Mat::eye(3, 3, CV_64F);
    D.at<double>(2, 2) = cv::determinant(svd.u * svd.vt) < 0 ? -1 : 1;
    Matx33d G(Mat(svd.u * D * svd.vt));
    double sum2 = 0, worst = 0;
    for (int i = 0; i < n; i++) {
        double e = deg(angleOf(G * Rt[i] * Re[i].t()));
        sum2 += e * e;
        worst = std::max(worst, e);
    }
    double rms = std::sqrt(sum2 / std::max(1, n));
    // PSNR against the truth seen in the stitched frame (a canvas direction d is the truth's G^T d), where covered
    Mat out = cv::imread(result.path.UTF8String, cv::IMREAD_COLOR), tr = cv::imread(truthPath.UTF8String, cv::IMREAD_COLOR);
    double psnr = 0;
    if (!out.empty() && !tr.empty()) {
        Equirect eq(result.fullWidth), eqT(tr.cols);
        Mat padded;
        cv::copyMakeBorder(tr, padded, 0, 0, 0, 1, cv::BORDER_WRAP);
        Mat mapx(out.rows, out.cols, CV_32F), mapy(out.rows, out.cols, CV_32F);
        Matx33d Gt = G.t();
        for (int y = 0; y < out.rows; y++) {
            float *mx = mapx.ptr<float>(y), *my = mapy.ptr<float>(y);
            for (int x = 0; x < out.cols; x++) {
                Point2d p = eqT.pixel(Gt * eq.ray(x + result.cropLeft, y + result.cropTop));
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
        cv::Scalar m = cv::mean(diff, covered);
        double mse = (m[0] + m[1] + m[2]) / 3.0;
        psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99;
    }
    bool pass = rms < 0.5 && worst < 1.5 && psnr > 20;
    return @{@"rotationRmsDeg" : @(rms), @"rotationMaxDeg" : @(worst), @"psnr" : @(psnr), @"pass" : @(pass)};
}

@end
