import Foundation

/// Google's Photo Sphere metadata (GPano, https://developers.google.com/streetview/spherical-metadata), written into the
/// JPEG as an XMP APP1 segment right after the start-of-image marker, so Google Photos, Street View, Facebook and the
/// viewers that look for it show the file as a sphere.
enum XMP {
    static func gpanoPacket(fullWidth: Int, fullHeight: Int, cropWidth: Int, cropHeight: Int, left: Int, top: Int,
                            headingDeg: Double, firstDate: Date, lastDate: Date, sourceCount: Int, usePanoramaViewer: Bool) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let fields: [(String, String)] = [
            ("ProjectionType", "equirectangular"),
            ("UsePanoramaViewer", usePanoramaViewer ? "True" : "False"),
            ("CroppedAreaImageWidthPixels", "\(cropWidth)"),
            ("CroppedAreaImageHeightPixels", "\(cropHeight)"),
            ("FullPanoWidthPixels", "\(fullWidth)"),
            ("FullPanoHeightPixels", "\(fullHeight)"),
            ("CroppedAreaLeftPixels", "\(left)"),
            ("CroppedAreaTopPixels", "\(top)"),
            ("PoseHeadingDegrees", String(format: "%.1f", headingDeg)),
            ("FirstPhotoDate", f.string(from: firstDate)),
            ("LastPhotoDate", f.string(from: lastDate)),
            ("SourcePhotosCount", "\(sourceCount)"),
            ("CaptureSoftware", "PhotoSphere for iPhone"),
            ("StitchingSoftware", "PhotoSphere for iPhone"),
        ]
        let attrs = fields.map { " GPano:\($0.0)=\"\($0.1)\"" }.joined()
        return "<?xpacket begin=\"\u{FEFF}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
            + "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
            + "<rdf:Description rdf:about=\"\" xmlns:GPano=\"http://ns.google.com/photos/1.0/panorama/\"\(attrs)/>"
            + "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
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
