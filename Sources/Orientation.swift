import CoreMotion
import simd

/// Frames (all right-handed):
/// - camera: x right, y down, z forward (the photo as held, upright in the device's portrait sense);
/// - world: y down (gravity), z the first photo's forward, x to its right. Yaw = atan2(x, z) of a forward vector,
///   pitch = asin(-y) (positive up), roll = asin(y of the camera's right vector).
/// This is also the frame the native stitcher uses, so a sensor rotation goes to it as it is.
typealias Mat3 = simd_double3x3

enum Orient {
    /// device-from-camera (and camera-from-device: it's its own inverse). The device frame is CoreMotion's: x right,
    /// y up the screen, z out of the screen; the back camera looks along -z.
    static let camFlip = Mat3(diagonal: SIMD3(1, -1, -1))

    /// CoreMotion's attitude as reference-from-device. Its documentation leaves the direction implicit, so the matrix is
    /// checked against the gravity vector (device coordinates, unambiguous) and transposed if that fits better.
    static func refFromDevice(_ motion: CMDeviceMotion) -> Mat3 {
        let m = motion.attitude.rotationMatrix
        let R = Mat3(rows: [SIMD3(m.m11, m.m12, m.m13), SIMD3(m.m21, m.m22, m.m23), SIMD3(m.m31, m.m32, m.m33)])
        let g = SIMD3(motion.gravity.x, motion.gravity.y, motion.gravity.z)
        let down = SIMD3<Double>(0, 0, -1)  // (reference frame: z up)
        let a = simd_dot(R.transpose * down, g), b = simd_dot(R * down, g)
        return a >= b ? R : R.transpose
    }

    static func forward(_ R: Mat3) -> SIMD3<Double> { R.columns.2 }
    static func yaw(_ R: Mat3) -> Double { atan2(R.columns.2.x, R.columns.2.z) }
    static func pitch(_ R: Mat3) -> Double { asin(max(-1, min(1, -R.columns.2.y))) }
    static func roll(_ R: Mat3) -> Double { asin(max(-1, min(1, R.columns.0.y))) }

    static func direction(yaw: Double, pitch: Double) -> SIMD3<Double> {
        SIMD3(cos(pitch) * sin(yaw), -sin(pitch), cos(pitch) * cos(yaw))
    }

    static func angle(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        acos(max(-1, min(1, simd_dot(simd_normalize(a), simd_normalize(b)))))
    }

    static func rotY(_ a: Double) -> Mat3 {
        let c = cos(a), s = sin(a)
        return Mat3(rows: [SIMD3(c, 0, s), SIMD3(0, 1, 0), SIMD3(-s, 0, c)])
    }

    /// The world frame from a reference-frame forward vector: y down, z that forward's horizontal part.
    static func worldFromRef(forwardInRef f: SIMD3<Double>) -> Mat3 {
        var h = SIMD3(f.x, f.y, 0)
        if simd_length(h) < 1e-6 { h = SIMD3(0, 1, 0) }
        let ez = simd_normalize(h), ey = SIMD3<Double>(0, 0, -1), ex = simd_cross(ey, ez)
        return Mat3(rows: [ex, ey, ez])
    }

    static func flat(_ R: Mat3) -> [Double] {
        [R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1], R[0, 2], R[1, 2], R[2, 2]]  // row-major (simd subscripts are [column, row])
    }

    static func matrix(_ a: [Double]) -> Mat3 {
        Mat3(rows: [SIMD3(a[0], a[1], a[2]), SIMD3(a[3], a[4], a[5]), SIMD3(a[6], a[7], a[8])])
    }

    /// Into SceneKit's frame (y up, camera looking down -z): both the world and the camera axes flipped in y and z.
    static let sceneFlip = Mat3(diagonal: SIMD3(1, -1, -1))
    static func sceneRotation(_ R: Mat3) -> Mat3 { sceneFlip * R * sceneFlip }
    static func scenePoint(_ d: SIMD3<Double>, at distance: Double) -> SIMD3<Float> {
        SIMD3<Float>(Float(d.x * distance), Float(-d.y * distance), Float(-d.z * distance))
    }
}

/// The world frame of a capture: floating with the camera's heading until the first photo (so the first dot is always
/// straight ahead), fixed from then on.
final class WorldFrame {
    private(set) var fixed: Mat3?

    func worldFromCam(refFromDevice: Mat3) -> Mat3 {
        let refFromCam = refFromDevice * Orient.camFlip
        let wfr = fixed ?? Orient.worldFromRef(forwardInRef: refFromCam.columns.2)
        return wfr * refFromCam
    }

    func fix(refFromDevice: Mat3) {
        fixed = Orient.worldFromRef(forwardInRef: (refFromDevice * Orient.camFlip).columns.2)
    }

    func release() { fixed = nil }
}
