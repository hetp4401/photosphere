import ImageIO
import SwiftUI
import UIKit

@main
struct PhotoSphereApp: App {
    var body: some Scene {
        WindowGroup {
            if CommandLine.arguments.contains("-selftest") {
                SelfTestView()
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
    @AppStorage("outputWidth") private var outputWidth = 6144
    @AppStorage("saveToPhotos") private var saveToPhotos = true

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
                    Picker("Size", selection: $outputWidth) {
                        Text("4096 x 2048 (fast)").tag(4096)
                        Text("6144 x 3072").tag(6144)
                        Text("8192 x 4096 (slow)").tag(8192)
                    }
                    Toggle("Save to Photos", isOn: $saveToPhotos)
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
