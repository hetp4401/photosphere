// The pipeline on a Linux box, for fast iteration (the same PanoCore.hpp the app uses):
//   harness synth <textured|room> <outdir> [fovDeg] [panoWidth]   a synthetic world photographed at the dots, stitched, measured
//   harness stitch <photo dir> <focalPx> <out.jpg> [width]        real photos (no sensor data): bootstrap + stitch
//   harness stitchprior <photo dir> <rotations.txt> <focalPx> <out.jpg> [width]   photos with known rotations (9 numbers a line)
#include "../Sources/Native/PanoCore.hpp"
#include <dirent.h>
#include <fstream>
#include <iostream>

static std::vector<std::string> jpegsIn(const std::string &dir) {
    std::vector<std::string> out;
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 4 && (n.substr(n.size() - 4) == ".jpg" || n.substr(n.size() - 4) == ".JPG")) out.push_back(dir + "/" + n);
        }
        closedir(d);
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// The dots as the app's TargetManager lays them out (Targets.swift), for the synthetic worlds.
static void targets(double H, double V, std::vector<double> &yaws, std::vector<double> &pitches) {
    int n0 = (int)std::ceil(2 * M_PI / (0.6 * H));
    for (int k = 0; k < n0; k++) {
        yaws.push_back(k * 2 * M_PI / n0);
        pitches.push_back(0);
    }
    double perRing = std::ceil(2 * M_PI / (0.675 * H));
    for (double sign : {1.0, -1.0}) {
        for (int i = 1; i <= 5; i++) {
            double p = sign * i * 0.6 * V, c = std::cos(p);
            if (c < -0.25 * V) break;
            if (c < 0.05 * V) {
                yaws.push_back(0);
                pitches.push_back(sign * M_PI / 2);
                break;
            }
            int n = std::max(1, (int)std::ceil(perRing * c));
            for (int k = 0; k < n; k++) {
                yaws.push_back(k * 2 * M_PI / n);
                pitches.push_back(p);
            }
        }
    }
}

static void show(double f, const char *s) {
    static std::string last;
    if (s != last) {
        std::cerr << "  " << s << std::endl;
        last = s;
    }
    (void)f;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: synth <world> <outdir> | stitch <dir> <focalPx> <out.jpg> [width] | stitchprior <dir> <rot.txt> <focalPx> <out.jpg> [width]\n";
        return 2;
    }
    std::string mode = argv[1];
    try {
        if (mode == "synth" && argc >= 4) {
            std::string world = argv[2], dir = argv[3];
            double fov = argc > 4 ? atof(argv[4]) : 70.0;
            int panoW = argc > 5 ? atoi(argv[5]) : 4096;
            double H = fov * M_PI / 180, V = 2 * std::atan(std::tan(H / 2) * 1200.0 / 1600.0);
            std::vector<double> yaws, pitches;
            targets(H, V, yaws, pitches);
            pano::Synth s = pano::generate(dir, yaws, pitches, fov, 1600, 1200, panoW, world);
            pano::StitchResult r = pano::stitch(s.images, s.priors, s.focalPx, panoW, dir + "/result.jpg", show);
            pano::Metrics m = pano::metrics(r, s.pano, s.truth);
            {  // the worst-placed photos, with where they look
                std::vector<int> idx(m.perPhotoDeg.size());
                for (size_t k = 0; k < idx.size(); k++) idx[k] = (int)k;
                std::sort(idx.begin(), idx.end(), [&](int a, int b) { return m.perPhotoDeg[a] > m.perPhotoDeg[b]; });
                printf("worst photos:");
                for (size_t k = 0; k < idx.size() && k < 8; k++)
                    printf(" #%d %.2f deg (yaw %.0f pitch %.0f)", idx[k], m.perPhotoDeg[idx[k]], yaws[idx[k]] * 180 / M_PI, pitches[idx[k]] * 180 / M_PI);
                printf("\n");
                std::ofstream pr(dir + "/pairs.txt");
                pr << r.pairReport;
            }
            printf("SELFTEST %s photos=%d featurePairs=%d directPairs=%d rotRMS=%.3f deg rotMax=%.3f deg psnr=%.2f dB crop=%dx%d of %dx%d full=%d seconds=%.1f notes=[%s] %s\n",
                   world.c_str(), r.images, r.featurePairs, r.directPairs, m.rotationRmsDeg, m.rotationMaxDeg, m.psnr, r.cropW, r.cropH, r.fullW, r.fullH,
                   r.full ? 1 : 0, r.seconds, r.notes.c_str(), m.pass ? "PASS" : "FAIL");
            return m.pass ? 0 : 1;
        }
        if (mode == "stitch" && argc >= 5) {
            std::vector<std::string> paths = jpegsIn(argv[2]);
            double focal = atof(argv[3]);
            std::string out = argv[4];
            int width = argc > 5 ? atoi(argv[5]) : 4096;
            auto t0 = std::chrono::steady_clock::now();
            pano::Bootstrap b = pano::bootstrap(paths, focal, show);
            double bs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("bootstrap: placed %zu of %zu, %d matched pairs (%d rejected), root %d, %.0f s; not placed:", b.placed.size(), paths.size(), b.edges, b.rejected, b.root, bs);
            for (size_t i = 0; i < paths.size(); i++)
                if (std::find(b.placed.begin(), b.placed.end(), (int)i) == b.placed.end()) printf(" %s", paths[i].substr(paths[i].rfind('/') + 1).c_str());
            printf("\n");
            std::vector<std::string> used;
            for (int i : b.placed) used.push_back(paths[i]);
            pano::StitchResult r = pano::stitch(used, b.R, focal, width, out, show, false);
            printf("stitch: %d photos, %d feature pairs, %d direct pairs, focal %.0f px, crop %dx%d of %dx%d, full circle %d, %.1f s, notes=[%s]; dropped:",
                   r.images, r.featurePairs, r.directPairs, r.focal, r.cropW, r.cropH, r.fullW, r.fullH, r.full ? 1 : 0, r.seconds, r.notes.c_str());
            for (int i : r.dropped) printf(" %s", used[i].substr(used[i].rfind('/') + 1).c_str());
            printf("\n");
            std::ofstream pr(out + ".pairs.txt");
            pr << r.pairReport;
            std::ofstream rot(out + ".rotations.txt");
            for (size_t k = 0; k < r.R.size(); k++) {
                rot << used[k];
                for (int e = 0; e < 9; e++) rot << " " << r.R[k](e / 3, e % 3);
                rot << "\n";
            }
            return 0;
        }
        if (mode == "stitchprior" && argc >= 6) {
            std::vector<std::string> paths;
            std::vector<cv::Matx33d> R;
            std::ifstream in(argv[3]);
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ss(line);
                std::string p;
                cv::Matx33d m;
                ss >> p;
                for (int e = 0; e < 9; e++) ss >> m(e / 3, e % 3);
                if (!p.empty()) {
                    paths.push_back(p[0] == '/' ? p : std::string(argv[2]) + "/" + p);
                    R.push_back(m);
                }
            }
            pano::StitchResult r = pano::stitch(paths, R, atof(argv[4]), argc > 6 ? atoi(argv[6]) : 4096, argv[5], show);
            printf("stitch: %d photos, %d feature pairs, %d direct pairs, focal %.0f px, crop %dx%d of %dx%d, full circle %d, %.1f s, notes=[%s]\n",
                   r.images, r.featurePairs, r.directPairs, r.focal, r.cropW, r.cropH, r.fullW, r.fullH, r.full ? 1 : 0, r.seconds, r.notes.c_str());
            return 0;
        }
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "bad arguments\n";
    return 2;
}
