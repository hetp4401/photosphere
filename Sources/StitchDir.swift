import SwiftUI

/// A test mode for real photos without sensor orientations (a camera app's), run with "-stitchdir <folder in tmp>
/// -focalPx <f> [-out <name>] [-width <W>]": their orientations are worked out from the images alone, then they go
/// through the normal stitch. The verdict goes to Documents/<name>.txt, the sphere to Documents/Panoramas/<name>.jpg.
struct StitchDirView: View {
    @StateObject private var model = StitchDirModel()

    var body: some View {
        ZStack {
            if let url = model.panoURL {
                PanoViewer(url: url).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
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
final class StitchDirModel: ObservableObject {
    @Published var status = "starting"
    @Published var panoURL: URL?

    func start() {
        let args = CommandLine.arguments
        func arg(_ k: String) -> String? {
            guard let i = args.firstIndex(of: k), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        let dirName = arg("-stitchdir") ?? "room"
        let focal = Double(arg("-focalPx") ?? "") ?? 1344
        let name = arg("-out") ?? dirName
        let width = Int(arg("-width") ?? "") ?? 4096
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(dirName)
        Task.detached(priority: .userInitiated) {
            let files = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.lowercased().hasSuffix(".jpg") }.sorted()
            let paths = files.map { dir.appendingPathComponent($0).path }
            var lines = ["STITCHDIR \(paths.count) photos, focal \(focal) px"]
            let t0 = Date()
            do {
                let show: (Double, String) -> Void = { f, s in Task { @MainActor in self.status = String(format: "%@ %.0f%%", s, f * 100) } }
                guard let boot = PanoStitcher.bootstrapRotations(paths, focalPx: focal, progress: show),
                      let placed = boot["placed"] as? [NSNumber], let rots = boot["rotations"] as? [[NSNumber]], !placed.isEmpty else {
                    throw NSError(domain: "stitchdir", code: 1, userInfo: [NSLocalizedDescriptionKey: "no photo could be placed"])
                }
                lines.append("bootstrap: placed \(placed.count) of \(paths.count), \(boot["edges"] ?? 0) matched pairs (\(boot["rejected"] ?? 0) rejected), root photo \(boot["root"] ?? -1), \(Int(Date().timeIntervalSince(t0))) s")
                let used = placed.map { paths[$0.intValue] }
                let raw = dir.appendingPathComponent("result.jpg").path
                let res = try PanoStitcher.stitchImages(used, rotations: rots, focalPx: focal, outputWidth: Int32(width), outputPath: raw, priorsFromSensors: false, progress: show)
                let packet = XMP.gpanoPacket(fullWidth: Int(res.fullWidth), fullHeight: Int(res.fullHeight), cropWidth: Int(res.cropWidth), cropHeight: Int(res.cropHeight),
                                             left: Int(res.cropLeft), top: Int(res.cropTop), headingDeg: 0, firstDate: Date(), lastDate: Date(), sourceCount: used.count, usePanoramaViewer: true)
                let data = XMP.embed(packet: packet, in: try Data(contentsOf: URL(fileURLWithPath: raw)))
                let out = CaptureSession.panoramasDir.appendingPathComponent("\(name).jpg")
                try data.write(to: out, options: .atomic)
                lines.append(String(format: "stitch: %d photos, %d feature pairs, %d direct pairs, focal %.0f px, crop %dx%d of %dx%d, full circle %d, %.1f s",
                                    res.imagesUsed, res.pairsMatched, res.pairsDirect, res.focalPx, res.cropWidth, res.cropHeight, res.fullWidth, res.fullHeight, res.fullCircle ? 1 : 0, res.seconds) + " " + res.notes + " dropped: " + res.dropped.map { used[$0.intValue].split(separator: "/").last.map(String.init) ?? "?" }.joined(separator: " "))
                lines.append("DONE")
                await MainActor.run { self.panoURL = out }
            } catch {
                lines.append("error: \(error.localizedDescription)")
            }
            let text = lines.joined(separator: "\n")
            try? text.write(to: CaptureSession.documents.appendingPathComponent("\(name).txt"), atomically: true, encoding: .utf8)
            await MainActor.run { self.status = text }
        }
    }
}
