import ImageIO
import SwiftUI
import UIKit

@main
struct PhotoSphereApp: App {
    var body: some Scene {
        WindowGroup {
            if CommandLine.arguments.contains("-selftest") {
                SelfTestView()
            } else if CommandLine.arguments.contains("-stitchdir") {
                StitchDirView()
            } else {
                RootView()
            }
        }
    }
}

struct RootView: View {
    @State private var capturing = false
    @State private var stitching: CaptureSession?
    @State private var panoramas: [URL] = []
    @AppStorage("outputMegapixels") private var outputMegapixels = 26.0
    @AppStorage("saveToPhotos") private var saveToPhotos = true
    @AppStorage("nadirPatchDeg") private var nadirPatchDeg = 0.0

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        capturing = true
                    } label: {
                        Label("New Photo Sphere", systemImage: "camera.circle.fill").font(.title3)
                    }
                }
                Section("Panoramas") {
                    if panoramas.isEmpty { Text("None yet").foregroundStyle(.secondary) }
                    ForEach(panoramas, id: \.self) { url in
                        NavigationLink {
                            PanoViewer(url: url).ignoresSafeArea().toolbar { ShareLink(item: url) }
                        } label: {
                            HStack {
                                if let img = Self.thumbnail(url) { Image(uiImage: img).resizable().aspectRatio(2, contentMode: .fit).frame(width: 120).clipShape(RoundedRectangle(cornerRadius: 6)) }
                                Text(url.deletingPathExtension().lastPathComponent)
                            }
                        }
                    }
                    .onDelete { idx in
                        idx.map { panoramas[$0] }.forEach { try? FileManager.default.removeItem(at: $0) }
                        reload()
                    }
                }
                Section("Output") {
                    Picker("Quality", selection: $outputMegapixels) {  // (Google Camera's three: low, high, max)
                        Text("Low (8 MP)").tag(8.0)
                        Text("High (26 MP)").tag(26.0)
                        Text("Max (70 MP, slow)").tag(70.0)
                    }
                    Toggle("Save to Photos", isOn: $saveToPhotos)
                    Picker("Hide the ground under me", selection: $nadirPatchDeg) {
                        Text("Off").tag(0.0)
                        Text("Small (15°)").tag(15.0)
                        Text("Medium (25°)").tag(25.0)
                        Text("Large (35°)").tag(35.0)
                    }
                }
            }
            .navigationTitle("PhotoSphere")
            .onAppear(perform: reload)
        }
        .fullScreenCover(isPresented: $capturing) {
            CaptureFlow(onDone: { capturing = false; reload() })
        }
    }

    private func reload() {
        let dir = CaptureSession.panoramasDir
        panoramas = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "jpg" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    static func thumbnail(_ url: URL) -> UIImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 360] as CFDictionary) else { return nil }
        return UIImage(cgImage: cg)
    }
}

/// Capture, then stitch, in one full-screen flow.
struct CaptureFlow: View {
    let onDone: () -> Void
    @State private var session: CaptureSession?

    var body: some View {
        if let session {
            StitchView(session: session) { _ in onDone() }
        } else {
            CaptureView(onFinished: { session = $0 }, onCancel: onDone).ignoresSafeArea()
        }
    }
}

struct CaptureView: UIViewControllerRepresentable {
    let onFinished: (CaptureSession) -> Void
    let onCancel: () -> Void

    func makeUIViewController(context: Context) -> CaptureViewController {
        let vc = CaptureViewController()
        vc.onFinished = onFinished
        vc.onCancel = onCancel
        return vc
    }

    func updateUIViewController(_ uiViewController: CaptureViewController, context: Context) {}
}
