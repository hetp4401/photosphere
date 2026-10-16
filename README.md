# PhotoSphere

A full 360° × 180° Photo Sphere camera for iPhone, built the way Google Camera's Photo Sphere worked (its
"lightcycle" code, studied from the 4.1 and 7.4 APKs).

**Install:** https://github.com/hetp4401/photosphere/releases/download/latest/PhotoSphere.ipa — unsigned, so sideload
it with AltStore, SideStore or Sideloadly (they sign it with your Apple ID; free accounts re-sign every 7 days).
Turn on Settings › Privacy & Security › Developer Mode.

## How it works, stage by stage

| Stage | Google Camera (lightcycle) | This app |
|---|---|---|
| Where the camera points | A gyro + accelerometer EKF; no magnetometer, so yaw is relative to the first photo | CoreMotion's attitude with the `xArbitraryZVertical` frame (gyro + accelerometer, no compass); a world frame fixed at the first photo |
| The dots | An equator ring every 0.6·FOV, rings at ±i·0.6·V with ⌈⌈2π/(0.675·H)⌉·cos p⌉ dots, a pole dot when a ring gets within 0.05·V of it; neighbours revealed as dots are captured | `Targets.swift`, the same layout and activation |
| When a photo fires | Camera axis within 2.75° (3.5° when turning) of an active dot, gyro under 0.4 rad/s (0.05 for exposures over 1/40 s, 1.0 under 1/100 s), roll within ±20° of the orientation locked at the first shot | `CaptureViewController.swift`, the same rules, exposure read live from the camera |
| The photo | Full-res JPEG, ~1800 px wide by default | AVFoundation still, made upright, 2016 px on the long side |
| Live mosaic | Preview frames pasted at their sensor poses in OpenGL | The photo's thumbnail pasted at its pose in SceneKit |
| Features | FAST corners, 8×8 oriented patches, 3 levels | ORB (FAST corners with oriented binary patches), 3000 per photo, at 1600 px |
| Pairs | Photos whose sensor axes are within 0.9·FOV; matches within a 6° latitude band, ratio test, a 2-point RANSAC rotation (1.1°, 550–5000 iterations), checked against the sensors (12.5° tilt, 25°·gap yaw); 35 stratified inliers kept, weight √(n/35) | The same, in `PanoStitcher.mm` |
| Global adjustment | Ceres: each photo's rotation and one shared focal, Huber(35 px) reprojection, gravity (tilt) priors with weight 0.0125·matches/images, yaw free then anchored to photo 0 | A small Levenberg–Marquardt with the same residuals, weights and gauge |
| Weakly matched photos | "Sensor pairs": a 3×3 grid transferred through the sensors, weight 0.125/0.25 | The same |
| Projection | Equirectangular, output 8/26/70 MP by setting | Equirectangular, 4096/6144/8192 wide by setting |
| Exposure | One gamma per photo from the overlaps' mean luma | OpenCV's per-photo gain from the overlaps |
| Seams | Graph cut on a 512-wide mosaic, colour + Laplacian cost | OpenCV's graph cut on a 1024-wide mosaic, colour + gradient cost |
| Blending | Fixed-point Laplacian pyramid in YCbCr, 10 levels | OpenCV's multiband blender |
| Crop | A full circle keeps its width and loses rows not fully covered (10° at most); a partial sphere is cut to its coverage | The same |
| Metadata | GPano XMP (`FullPano*`, `CroppedArea*`, `PoseHeadingDegrees`, …) | The same XMP, written by `XMP.swift` |

Not carried over: Google's line features for textureless photos, its ground-point down-weighting, and its gyro-bias
calibration (CoreMotion does that itself).

## Build

`brew install xcodegen && xcodegen generate && open PhotoSphere.xcodeproj`. OpenCV comes from the `yeatse/opencv-spm`
Swift package. Every push runs `.github/workflows/ios.yml`: the self test in the simulator (`-selftest`: a synthetic
world photographed at the dots with the sensors' errors added, stitched, and compared with the truth), then the IPA.
