import SceneKit
import SwiftUI
import UIKit

/// The build's check, run in the simulator with "-selftest": a synthetic world is photographed at the dots the target
/// manager lays out, with the sensors' errors added, shown as the live mosaic would be, stitched through the app's own
/// path, and compared with the truth. The verdict goes to Documents/selftest.txt.
struct SelfTestView: View {
    @StateObject private var model = SelfTestModel()

    var body: some View {
        ZStack {
            if let url = model.panoURL {
                PanoViewer(url: url).ignoresSafeArea()
            } else {
                MosaicReplay(model: model).ignoresSafeArea()
            }
            VStack {
                Spacer()
                Text(model.status).font(.footnote.monospaced()).foregroundStyle(.white).padding(8).background(.black.opacity(0.6)).cornerRadius(6).padding()
            }
        }
        .onAppear { model.start() }
    }
}

@MainActor
final class SelfTestModel: ObservableObject {
    @Published var status = "generating"
    @Published var panoURL: URL?
    @Published var images: [(UIImage, Mat3)] = []
    let hfovDeg = 70.0

    func start() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("selftest").path
        let vfovDeg = 2 * atan(tan(hfovDeg * .pi / 360) * 1200.0 / 1600.0) * 180 / .pi
        let tm = TargetManager(hfov: hfovDeg * .pi / 180, vfov: vfovDeg * .pi / 180)
        let yaws = tm.targets.map { NSNumber(value: $0.yaw) }, pitches = tm.targets.map { NSNumber(value: $0.pitch) }
        Task.detached(priority: .userInitiated) {
            guard let gen = PanoStitcher.selfTestGenerate(inDirectory: dir, yaws: yaws, pitches: pitches, fovDeg: self.hfovDeg, width: 1600, height: 1200, panoWidth: 4096),
                  let paths = gen["images"] as? [String], let priors = gen["rotations"] as? [[NSNumber]], let truth = gen["truth"] as? [[NSNumber]],
                  let focal = gen["focalPx"] as? Double, let truthPano = gen["pano"] as? String else {
                await MainActor.run { self.status = "generation failed" }
                return
            }
            let thumbs = paths.map { CaptureSession.upright(UIImage(contentsOfFile: $0) ?? UIImage(), maxSide: 400) }
            await MainActor.run {
                self.images = zip(thumbs, priors).map { ($0, Orient.matrix($1.map { $0.doubleValue })) }
                self.status = "\(paths.count) synthetic photos, stitching"
            }
            let args = CommandLine.arguments
            if let i = args.firstIndex(of: "-mosaicseconds"), i + 1 < args.count, let secs = Double(args[i + 1]) {
                try? await Task.sleep(nanoseconds: UInt64(secs * 1e9))
            }
            do {
                let raw = (dir as NSString).appendingPathComponent("result.jpg")
                let res = try PanoStitcher.stitchImages(paths, rotations: priors, focalPx: focal, outputWidth: 4096, outputPath: raw) { f, s in
                    Task { @MainActor in self.status = String(format: "%@ %.0f%%", s, f * 100) }
                }
                let metrics = PanoStitcher.selfTestMetrics(for: res, truthPano: truthPano, truthRotations: truth)
                let rms = (metrics["rotationRmsDeg"] as? Double) ?? 99, worst = (metrics["rotationMaxDeg"] as? Double) ?? 99
                let psnr = (metrics["psnr"] as? Double) ?? 0, pass = (metrics["pass"] as? Bool) ?? false
                let line = String(format: "SELFTEST photos=%d pairs=%d rotRMS=%.3f deg rotMax=%.3f deg psnr=%.2f dB crop=%dx%d of %dx%d full=%d seconds=%.1f %@",
                                  res.imagesUsed, res.pairsMatched, rms, worst, psnr, res.cropWidth, res.cropHeight, res.fullWidth, res.fullHeight,
                                  res.fullCircle ? 1 : 0, res.seconds, pass ? "PASS" : "FAIL")
                let packet = XMP.gpanoPacket(fullWidth: Int(res.fullWidth), fullHeight: Int(res.fullHeight), cropWidth: Int(res.cropWidth), cropHeight: Int(res.cropHeight),
                                             left: Int(res.cropLeft), top: Int(res.cropTop), headingDeg: 0, firstDate: Date(), lastDate: Date(), sourceCount: paths.count, usePanoramaViewer: true)
                let data = XMP.embed(packet: packet, in: try Data(contentsOf: URL(fileURLWithPath: raw)))
                let out = CaptureSession.panoramasDir.appendingPathComponent("selftest.jpg")
                try data.write(to: out, options: .atomic)
                try line.write(to: CaptureSession.documents.appendingPathComponent("selftest.txt"), atomically: true, encoding: .utf8)
                print(line)
                await MainActor.run {
                    self.status = line
                    self.panoURL = out
                }
            } catch {
                let line = "SELFTEST error: \(error.localizedDescription) FAIL"
                try? line.write(to: CaptureSession.documents.appendingPathComponent("selftest.txt"), atomically: true, encoding: .utf8)
                await MainActor.run { self.status = line }
            }
        }
    }
}

/// The live mosaic as the capture screen shows it, with the synthetic photos at their (noisy) sensor rotations and a
/// camera that turns by itself.
struct MosaicReplay: UIViewRepresentable {
    @ObservedObject var model: SelfTestModel

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        context.coordinator.overlay = SphereOverlay(view: v, transparent: false)
        context.coordinator.overlay?.setFov(hfovDeg: model.hfovDeg, vfovDeg: 2 * atan(tan(model.hfovDeg * .pi / 360) * 0.75) * 180 / .pi)
        context.coordinator.start()
        return v
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        let c = context.coordinator
        guard c.shown < model.images.count else { return }
        for (img, R) in model.images[c.shown...] { c.overlay?.addPhoto(img, rotation: R) }
        c.shown = model.images.count
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var overlay: SphereOverlay?
        var shown = 0
        private var timer: Timer?
        private var t = 0.0

        func start() {
            timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
                guard let self, let overlay = self.overlay else { return }
                self.t += 1.0 / 30
                let yaw = self.t * 0.35, pitch = 0.35 * sin(self.t * 0.5)
                overlay.setCamera(Mat3(rows: [SIMD3(cos(yaw), 0, sin(yaw)), SIMD3(0, 1, 0), SIMD3(-sin(yaw), 0, cos(yaw))])
                    * Mat3(rows: [SIMD3(1, 0, 0), SIMD3(0, cos(pitch), -sin(pitch)), SIMD3(0, sin(pitch), cos(pitch))]))
            }
        }
    }
}
