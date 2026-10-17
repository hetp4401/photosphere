import Foundation

/// Google's Photo Sphere metadata (GPano, https://developers.google.com/streetview/spherical-metadata), written into the
/// JPEG as an XMP APP1 segment right after the start-of-image marker, so Google Photos, Street View, Facebook and the
/// viewers that look for it show the file as a sphere.
enum XMP {
    static func gpanoPacket(fullWidth: Int, fullHeight: Int, cropWidth: Int, cropHeight: Int, left: Int, top: Int,
                            headingDeg: Double?, firstDate: Date, lastDate: Date, sourceCount: Int, usePanoramaViewer: Bool) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var fields: [(String, String)] = [
            ("ProjectionType", "equirectangular"),
            ("UsePanoramaViewer", usePanoramaViewer ? "True" : "False"),
            ("CroppedAreaImageWidthPixels", "\(cropWidth)"),
            ("CroppedAreaImageHeightPixels", "\(cropHeight)"),
            ("FullPanoWidthPixels", "\(fullWidth)"),
            ("FullPanoHeightPixels", "\(fullHeight)"),
            ("CroppedAreaLeftPixels", "\(left)"),
            ("CroppedAreaTopPixels", "\(top)"),
            ("FirstPhotoDate", f.string(from: firstDate)),
            ("LastPhotoDate", f.string(from: lastDate)),
            ("SourcePhotosCount", "\(sourceCount)"),
            ("CaptureSoftware", "PhotoSphere for iPhone"),
            ("StitchingSoftware", "PhotoSphere for iPhone"),
            ("LargestValidInteriorRectLeft", "0"),
            ("LargestValidInteriorRectTop", "0"),
            ("LargestValidInteriorRectWidth", "\(cropWidth)"),
            ("LargestValidInteriorRectHeight", "\(cropHeight)"),
        ]
        if let h = headingDeg { fields.append(("PoseHeadingDegrees", String(format: "%.1f", h))) }  // (GCam: only when known)
        let attrs = fields.map { " GPano:\($0.0)=\"\($0.1)\"" }.joined()
        return "<?xpacket begin=\"\u{FEFF}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
            + "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
            + "<rdf:Description rdf:about=\"\" xmlns:GPano=\"http://ns.google.com/photos/1.0/panorama/\"\(attrs)/>"
            + "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    }

    /// The crop a file's GPano metadata gives: (full width, full height, crop width, crop height, crop left, crop top).
    static func gpanoCrop(of url: URL) -> (Double, Double, Double, Double, Double, Double)? {
        guard let h = FileHandle(forReadingAtPath: url.path) else { return nil }
        defer { try? h.close() }
        let head = String(decoding: h.readData(ofLength: 256 * 1024), as: UTF8.self)
        func value(_ k: String) -> Double? {
            for pattern in ["GPano:\(k)=\"", "<GPano:\(k)>"] {
                if let r = head.range(of: pattern) {
                    let rest = head[r.upperBound...]
                    let num = rest.prefix { "0123456789.-".contains($0) }
                    if let v = Double(num) { return v }
                }
            }
            return nil
        }
        guard let fw = value("FullPanoWidthPixels"), let fh = value("FullPanoHeightPixels"), let cw = value("CroppedAreaImageWidthPixels"),
              let ch = value("CroppedAreaImageHeightPixels"), fw > 0, fh > 0, cw > 0, ch > 0 else { return nil }
        return (fw, fh, cw, ch, value("CroppedAreaLeftPixels") ?? 0, value("CroppedAreaTopPixels") ?? 0)
    }

    /// The JPEG with an XMP APP1 segment inserted after SOI (an existing one is left alone: ours comes first).
    static func embed(packet: String, in jpeg: Data) -> Data {
        guard jpeg.count > 4, jpeg[0] == 0xFF, jpeg[1] == 0xD8 else { return jpeg }
        var seg = Data([0xFF, 0xE1])
        let ns = Data("http://ns.adobe.com/xap/1.0/\0".utf8), body = Data(packet.utf8)
        let len = 2 + ns.count + body.count
        seg.append(UInt8(len >> 8))
        seg.append(UInt8(len & 0xFF))
        seg.append(ns)
        seg.append(body)
        var out = Data(jpeg.prefix(2))
        out.append(seg)
        out.append(jpeg.suffix(from: 2))
        return out
    }
}
