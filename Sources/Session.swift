import Foundation
import UIKit

/// A capture in progress: its photos on disk (upright, at most 2016 px on the long side, as Google Camera's ~1800), each
/// with the rotation the sensors gave at the moment the dot was hit.
struct Shot: Codable {
    let file: String
    let rotation: [Double]  // world-from-camera, row-major
    let time: Double
    let exposure: Double
}

struct SessionInfo: Codable {
    var shots: [Shot] = []
    var focalPx: Double = 0          // of the saved photos
    var width: Int = 0
    var height: Int = 0
    var hfovDeg: Double = 0
    var vfovDeg: Double = 0
    var headingDeg: Double?          // compass heading of the first photo, when known
    var started: Double = Date().timeIntervalSince1970
}

final class CaptureSession {
    static let maxSide = 2016
    let dir: URL
    private(set) var info = SessionInfo()

    init() {
        let name = Self.stamp(Date())
        dir = Self.sessionsDir.appendingPathComponent(name)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var sessionsDir: URL { documents.appendingPathComponent("sessions") }
    static var panoramasDir: URL {
        let d = documents.appendingPathComponent("Panoramas")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func stamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f.string(from: date)
    }

    func setCamera(width: Int, height: Int, focalPx: Double, hfovDeg: Double, vfovDeg: Double) {
        info.width = width
        info.height = height
        info.focalPx = focalPx
        info.hfovDeg = hfovDeg
        info.vfovDeg = vfovDeg
    }

    /// The photo made upright and shrunk, written as N.jpg; its thumbnail returned for the live mosaic.
    func add(photo: UIImage, rotation: Mat3, exposure: Double) -> UIImage? {
        let n = info.shots.count
        let upright = Self.upright(photo, maxSide: Self.maxSide)
        guard let data = upright.jpegData(compressionQuality: 0.92) else { return nil }
        let name = "\(n).jpg"
        try? data.write(to: dir.appendingPathComponent(name), options: .atomic)
        if info.width == 0 {
            info.width = Int(upright.size.width)
            info.height = Int(upright.size.height)
        }
        info.shots.append(Shot(file: name, rotation: Orient.flat(rotation), time: Date().timeIntervalSince1970, exposure: exposure))
        save()
        return Self.upright(upright, maxSide: 512)
    }

    func removeLast() {
        guard let last = info.shots.popLast() else { return }
        try? FileManager.default.removeItem(at: dir.appendingPathComponent(last.file))
        save()
    }

    var count: Int { info.shots.count }

    func save() {
        if let data = try? JSONEncoder().encode(info) { try? data.write(to: dir.appendingPathComponent("session.json")) }
    }

    func discard() { try? FileManager.default.removeItem(at: dir) }

    /// The image drawn with its orientation applied, scaled so its long side is at most maxSide.
    static func upright(_ image: UIImage, maxSide: Int) -> UIImage {
        let w = image.size.width, h = image.size.height
        guard w > 0, h > 0 else { return image }
        let s = min(1, CGFloat(maxSide) / max(w, h))
        let size = CGSize(width: (w * s).rounded(), height: (h * s).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
    }
}
