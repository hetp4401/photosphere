import Foundation
import Photos
import SwiftUI
import UIKit

/// Runs the native stitcher on a session, writes the GPano metadata and keeps the result in Documents/Panoramas
/// (and in Photos when allowed).
@MainActor
final class StitchJob: ObservableObject {
    @Published var fraction = 0.0
    @Published var stage = "starting"
    @Published var result: URL?
    @Published var summary = ""
    @Published var failure: String?

    func run(session: CaptureSession, outputWidth: Int, saveToPhotos: Bool, nadirPatchDeg: Double) {
        let info = session.info
        let dir = session.dir
        Task.detached(priority: .userInitiated) {
            do {
                let paths = info.shots.map { dir.appendingPathComponent($0.file).path }
                let rotations = info.shots.map { $0.rotation.map { NSNumber(value: $0) } }
                let longSide = Double(max(info.width, info.height))
                let longFov = max(info.hfovDeg, info.vfovDeg) * .pi / 180
                let focalPx = info.focalPx > 0 ? info.focalPx : (longSide / 2) / tan(longFov / 2)
                let raw = FileManager.default.temporaryDirectory.appendingPathComponent("pano_raw.jpg")
                let res = try PanoStitcher.stitchImages(paths, rotations: rotations, focalPx: focalPx, outputWidth: Int32(outputWidth), outputPath: raw.path, priorsFromSensors: true, nadirPatchDeg: nadirPatchDeg) { f, s in
                    Task { @MainActor in
                        self.fraction = f
                        self.stage = s
                    }
                }
                let first = Date(timeIntervalSince1970: info.shots.first?.time ?? info.started)
                let last = Date(timeIntervalSince1970: info.shots.last?.time ?? info.started)
                let realFov = Double(res.cropWidth) / Double(res.fullWidth) * 360
                let packet = XMP.gpanoPacket(fullWidth: Int(res.fullWidth), fullHeight: Int(res.fullHeight), cropWidth: Int(res.cropWidth),
                                             cropHeight: Int(res.cropHeight), left: Int(res.cropLeft), top: Int(res.cropTop),
                                             headingDeg: info.headingDeg ?? 0, firstDate: first, lastDate: last, sourceCount: info.shots.count,
                                             usePanoramaViewer: realFov >= 70)
                let data = XMP.embed(packet: packet, in: try Data(contentsOf: raw))
                let out = CaptureSession.panoramasDir.appendingPathComponent("\(CaptureSession.stamp(first)).jpg")
                try data.write(to: out, options: .atomic)
                try? FileManager.default.removeItem(at: raw)
                if saveToPhotos {
                    let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
                    if status == .authorized || status == .limited {
                        try? await PHPhotoLibrary.shared().performChanges { _ = PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: out) }
                    }
                }
                let text = "\(res.imagesUsed) photos, \(res.pairsMatched) matched pairs, \(res.cropWidth)x\(res.cropHeight) of \(res.fullWidth)x\(res.fullHeight), "
                    + String(format: "%.0f s", res.seconds) + (res.fullCircle ? ", full circle" : "")
                await MainActor.run {
                    self.summary = text
                    self.result = out
                }
            } catch {
                await MainActor.run { self.failure = error.localizedDescription }
            }
        }
    }
}

struct StitchView: View {
    let session: CaptureSession
    let onDone: (URL?) -> Void
    @StateObject private var job = StitchJob()
    @AppStorage("outputWidth") private var outputWidth = 6144
    @AppStorage("saveToPhotos") private var saveToPhotos = true
    @AppStorage("nadirPatchDeg") private var nadirPatchDeg = 0.0
    @State private var share: ShareItem?

    var body: some View {
        VStack(spacing: 24) {
            if let url = job.result {
                PanoViewer(url: url).ignoresSafeArea()
                Text(job.summary).font(.footnote).foregroundStyle(.secondary).padding(.horizontal)
                HStack(spacing: 16) {
                    Button("Share the photos") {  // (the capture itself, zipped: photos and sensor data)
                        if let z = CaptureSession.zip(session.dir) { share = ShareItem(url: z) }
                    }.buttonStyle(.bordered)
                    Button("Done") { onDone(url) }.buttonStyle(.borderedProminent)
                }
                .sheet(item: $share) { ShareSheet(items: [$0.url]) }
            } else if let f = job.failure {
                Text("Stitching failed").font(.title3.bold())
                Text(f).foregroundStyle(.secondary).multilineTextAlignment(.center).padding(.horizontal)
                Button("Back") { onDone(nil) }.buttonStyle(.bordered)
            } else {
                Spacer()
                ProgressView(value: job.fraction).padding(.horizontal, 40)
                Text(job.stage).font(.headline)
                Text("\(session.count) photos").foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding(.vertical)
        .onAppear { job.run(session: session, outputWidth: outputWidth, saveToPhotos: saveToPhotos, nadirPatchDeg: nadirPatchDeg) }
    }
}

struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController { UIActivityViewController(activityItems: items, applicationActivities: nil) }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
