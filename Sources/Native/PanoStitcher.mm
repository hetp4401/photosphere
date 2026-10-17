// The Objective-C face of the pipeline (PanoCore.hpp): Foundation types in, Foundation types out.
#import "PanoCore.hpp"
#include <memory>
#import <Foundation/Foundation.h>
#import "PanoStitcher.h"

namespace {

std::vector<std::string> strings(NSArray<NSString *> *a) {
    std::vector<std::string> out;
    for (NSString *s in a) out.push_back(s.UTF8String);
    return out;
}

cv::Matx33d matrixFrom(NSArray<NSNumber *> *a) {
    cv::Matx33d R;
    for (int k = 0; k < 9; k++) R(k / 3, k % 3) = a[k].doubleValue;
    return R;
}

NSArray<NSNumber *> *arrayFrom(const cv::Matx33d &R) {
    NSMutableArray *a = [NSMutableArray array];
    for (int k = 0; k < 9; k++) [a addObject:@(R(k / 3, k % 3))];
    return a;
}

NSArray<NSArray<NSNumber *> *> *arraysFrom(const std::vector<cv::Matx33d> &Rs) {
    NSMutableArray *a = [NSMutableArray array];
    for (auto &R : Rs) [a addObject:arrayFrom(R)];
    return a;
}

std::vector<cv::Matx33d> matricesFrom(NSArray<NSArray<NSNumber *> *> *a) {
    std::vector<cv::Matx33d> out;
    for (NSArray<NSNumber *> *r in a) out.push_back(matrixFrom(r));
    return out;
}

pano::Progress progressFrom(PanoProgress block) {
    if (!block) return nullptr;
    return [block](double f, const char *s) { block(f, [NSString stringWithUTF8String:s]); };
}

NSError *err(const std::string &msg) {
    return [NSError errorWithDomain:@"PanoStitcher" code:1 userInfo:@{NSLocalizedDescriptionKey : [NSString stringWithUTF8String:msg.c_str()]}];
}

}  // namespace

@implementation PanoStitchResult
@end

@implementation PanoTracker {
    std::unique_ptr<pano::LiveTracker> tracker_;
}

- (instancetype)initWithShortFovDeg:(double)shortFovDeg {
    if ((self = [super init])) tracker_.reset(new pano::LiveTracker(shortFovDeg * M_PI / 180));
    return self;
}

static cv::Mat grayFrom(UIImage *image) {
    CGImageRef cg = image.CGImage;
    if (!cg) return cv::Mat();
    int w = (int)CGImageGetWidth(cg), h = (int)CGImageGetHeight(cg);
    cv::Mat gray(h, w, CV_8UC1);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceGray();
    CGContextRef ctx = CGBitmapContextCreate(gray.data, w, h, 8, gray.step, cs, kCGImageAlphaNone);
    if (ctx) {
        CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cg);
        CGContextRelease(ctx);
    }
    CGColorSpaceRelease(cs);
    return ctx ? gray : cv::Mat();
}

- (void)addKeyframe:(UIImage *)image rotation:(NSArray<NSNumber *> *)rotation {
    cv::Mat g = grayFrom(image);
    if (!g.empty()) tracker_->addKeyframe(g, matrixFrom(rotation));
}

- (void)removeLast {
    tracker_->removeLast();
}

- (BOOL)track:(CVPixelBufferRef)buffer gyro:(NSArray<NSNumber *> *)gyro {
    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    cv::Mat gray;
    OSType fmt = CVPixelBufferGetPixelFormatType(buffer);
    if (fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
        gray = cv::Mat((int)CVPixelBufferGetHeightOfPlane(buffer, 0), (int)CVPixelBufferGetWidthOfPlane(buffer, 0), CV_8UC1,
                       CVPixelBufferGetBaseAddressOfPlane(buffer, 0), CVPixelBufferGetBytesPerRowOfPlane(buffer, 0));
    } else if (fmt == kCVPixelFormatType_32BGRA) {
        cv::Mat bgra((int)CVPixelBufferGetHeight(buffer), (int)CVPixelBufferGetWidth(buffer), CV_8UC4, CVPixelBufferGetBaseAddress(buffer), CVPixelBufferGetBytesPerRow(buffer));
        cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);
    }
    int pts = 0;
    double step = 0;
    BOOL ok = !gray.empty() && tracker_->track(gray, matrixFrom(gyro), pts, step);
    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    return ok;
}

- (NSArray<NSNumber *> *)drift {
    return arrayFrom(tracker_->drift());
}

- (double)driftDeg {
    return pano::deg(pano::angleOf(tracker_->drift()));
}

@end

@implementation PanoStitcher

+ (nullable PanoStitchResult *)stitchImages:(NSArray<NSString *> *)paths
                                  rotations:(NSArray<NSArray<NSNumber *> *> *)rotations
                                    focalPx:(double)focalPx
                           outputMegapixels:(double)outputMegapixels
                                 outputPath:(NSString *)outputPath
                          priorsFromSensors:(BOOL)priorsFromSensors
                              nadirPatchDeg:(double)nadirPatchDeg
                                   progress:(nullable PanoProgress)progress
                                      error:(NSError **)error {
    try {
        pano::StitchResult r = pano::stitch(strings(paths), matricesFrom(rotations), focalPx, outputMegapixels, outputPath.UTF8String, progressFrom(progress), priorsFromSensors,
                                            nadirPatchDeg);
        PanoStitchResult *res = [PanoStitchResult new];
        res.path = [NSString stringWithUTF8String:r.path.c_str()];
        res.fullWidth = r.fullW;
        res.fullHeight = r.fullH;
        res.cropWidth = r.cropW;
        res.cropHeight = r.cropH;
        res.cropLeft = r.left;
        res.cropTop = r.top;
        res.fullCircle = r.full;
        res.imagesUsed = r.images;
        res.pairsMatched = r.featurePairs;
        res.pairsDirect = r.directPairs;
        res.focalPx = r.focal;
        res.seconds = r.seconds;
        res.rotations = arraysFrom(r.R);
        res.notes = [NSString stringWithUTF8String:r.notes.c_str()];
        NSMutableArray *dropped = [NSMutableArray array];
        for (int i : r.dropped) [dropped addObject:@(i)];
        res.dropped = dropped;
        res.firstYawDeg = r.firstYawDeg;
        return res;
    } catch (const std::exception &e) {
        if (error) *error = err(std::string("stitching failed: ") + e.what());
        return nil;
    }
}

+ (nullable NSDictionary *)bootstrapRotations:(NSArray<NSString *> *)paths focalPx:(double)focalPx progress:(nullable PanoProgress)progress {
    try {
        pano::Bootstrap b = pano::bootstrap(strings(paths), focalPx, progressFrom(progress));
        if (b.placed.empty()) return nil;
        NSMutableArray *idx = [NSMutableArray array];
        for (int i : b.placed) [idx addObject:@(i)];
        return @{@"placed" : idx, @"rotations" : arraysFrom(b.R), @"edges" : @(b.edges), @"rejected" : @(b.rejected), @"secondChance" : @(b.secondChance), @"root" : @(b.root)};
    } catch (const std::exception &) {
        return nil;
    }
}

+ (nullable NSDictionary *)selfTestGenerateInDirectory:(NSString *)dir
                                                  yaws:(NSArray<NSNumber *> *)yaws
                                               pitches:(NSArray<NSNumber *> *)pitches
                                                fovDeg:(double)fovDeg
                                                 width:(int)width
                                                height:(int)height
                                             panoWidth:(int)panoWidth {
    return [self selfTestGenerateInDirectory:dir yaws:yaws pitches:pitches fovDeg:fovDeg width:width height:height panoWidth:panoWidth world:@"textured"];
}

+ (nullable NSDictionary *)selfTestGenerateInDirectory:(NSString *)dir
                                                  yaws:(NSArray<NSNumber *> *)yaws
                                               pitches:(NSArray<NSNumber *> *)pitches
                                                fovDeg:(double)fovDeg
                                                 width:(int)width
                                                height:(int)height
                                             panoWidth:(int)panoWidth
                                                 world:(NSString *)world {
    std::vector<double> ys, ps;
    for (NSNumber *y in yaws) ys.push_back(y.doubleValue);
    for (NSNumber *p in pitches) ps.push_back(p.doubleValue);
    try {
        pano::Synth s = pano::generate(dir.UTF8String, ys, ps, fovDeg, width, height, panoWidth, world.UTF8String);
        NSMutableArray *images = [NSMutableArray array];
        for (auto &p : s.images) [images addObject:[NSString stringWithUTF8String:p.c_str()]];
        return @{@"images" : images, @"rotations" : arraysFrom(s.priors), @"truth" : arraysFrom(s.truth), @"focalPx" : @(s.focalPx),
                 @"width" : @(s.width), @"height" : @(s.height), @"pano" : [NSString stringWithUTF8String:s.pano.c_str()]};
    } catch (const std::exception &) {
        return nil;
    }
}

+ (NSDictionary *)selfTestMetricsForResult:(PanoStitchResult *)result
                                 truthPano:(NSString *)truthPath
                            truthRotations:(NSArray<NSArray<NSNumber *> *> *)truth {
    pano::StitchResult r;
    r.path = result.path.UTF8String;
    r.fullW = result.fullWidth;
    r.fullH = result.fullHeight;
    r.left = result.cropLeft;
    r.top = result.cropTop;
    r.R = matricesFrom(result.rotations);
    pano::Metrics m = pano::metrics(r, truthPath.UTF8String, matricesFrom(truth));
    return @{@"rotationRmsDeg" : @(m.rotationRmsDeg), @"rotationMaxDeg" : @(m.rotationMaxDeg), @"psnr" : @(m.psnr), @"pass" : @(m.pass)};
}

@end
