#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// What a stitch produced: the JPEG (no XMP yet), the full equirect canvas, the crop (the filled part), and the refined
/// camera rotations (world-from-camera, 9 doubles each, row-major).
@interface PanoStitchResult : NSObject
@property (nonatomic, copy) NSString *path;
@property (nonatomic) int fullWidth;
@property (nonatomic) int fullHeight;
@property (nonatomic) int cropWidth;
@property (nonatomic) int cropHeight;
@property (nonatomic) int cropLeft;
@property (nonatomic) int cropTop;
@property (nonatomic) BOOL fullCircle;
@property (nonatomic) int imagesUsed;
@property (nonatomic) int pairsMatched;
@property (nonatomic) double focalPx;
@property (nonatomic) double seconds;
@property (nonatomic, copy) NSArray<NSArray<NSNumber *> *> *rotations;
@end

typedef void (^PanoProgress)(double fraction, NSString *stage);

/// The Photo Sphere pipeline, as Google Camera's lightcycle did it: features and pairwise rotations between the photos
/// whose sensor orientations overlap, a global adjustment of every photo's rotation plus one shared focal length pulled
/// toward gravity, then an equirectangular render with exposure matching, graph-cut seams and multiband blending.
@interface PanoStitcher : NSObject

/// paths: the photos (upright, all the same size). rotations: each photo's world-from-camera matrix from the sensors
/// (camera frame: x right, y down, z forward; world: y down, z the first photo's forward). focalPx: the lens's focal
/// length in pixels of those photos. outputWidth: the width of the full 360x180 canvas (its height is half).
+ (nullable PanoStitchResult *)stitchImages:(NSArray<NSString *> *)paths
                                  rotations:(NSArray<NSArray<NSNumber *> *> *)rotations
                                    focalPx:(double)focalPx
                                outputWidth:(int)outputWidth
                                 outputPath:(NSString *)outputPath
                                   progress:(nullable PanoProgress)progress
                                      error:(NSError *_Nullable *_Nullable)error;

/// Makes a synthetic textured panorama and renders a photo of it at each (yaw, pitch) given, with realistic noise:
/// exposure changes, sensor noise, JPEG, and sensor rotations perturbed by random error plus a slow yaw drift. Returns
/// {"images": [paths], "rotations": [[9]] (the noisy priors), "truth": [[9]], "focalPx", "width", "height", "pano"}.
+ (nullable NSDictionary *)selfTestGenerateInDirectory:(NSString *)dir
                                                  yaws:(NSArray<NSNumber *> *)yaws
                                               pitches:(NSArray<NSNumber *> *)pitches
                                                fovDeg:(double)fovDeg
                                                 width:(int)width
                                                height:(int)height
                                             panoWidth:(int)panoWidth;

/// The same with a choice of world: "textured" (shapes everywhere) or "room" (plain walls, a few things on them).
+ (nullable NSDictionary *)selfTestGenerateInDirectory:(NSString *)dir
                                                  yaws:(NSArray<NSNumber *> *)yaws
                                               pitches:(NSArray<NSNumber *> *)pitches
                                                fovDeg:(double)fovDeg
                                                 width:(int)width
                                                height:(int)height
                                             panoWidth:(int)panoWidth
                                                 world:(NSString *)world;

/// How well a stitch of the synthetic set did: {"rotationRmsDeg", "rotationMaxDeg", "psnr", "pass"}.
+ (NSDictionary *)selfTestMetricsForResult:(PanoStitchResult *)result
                                 truthPano:(NSString *)truthPath
                            truthRotations:(NSArray<NSArray<NSNumber *> *> *)truth;
@end

NS_ASSUME_NONNULL_END
