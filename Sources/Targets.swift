import Foundation
import simd

/// The dots to point the camera at, laid out from the lens's field of view as Google Camera's target generator did:
/// an equator ring, rings every 0.6 of the vertical FOV, a single dot at a pole when a ring gets too close to it;
/// revealed as their neighbours are captured.
struct Target {
    enum State { case inactive, active, captured }
    let id: Int
    let yaw: Double
    let pitch: Double
    let ring: Int
    var state: State = .inactive
    var neighbours: [Int] = []
    var direction: SIMD3<Double> { Orient.direction(yaw: yaw, pitch: pitch) }
}

final class TargetManager {
    private(set) var targets: [Target] = []
    private var history: [Int] = []  // the captured dots, in order

    /// hfov, vfov: the photo's horizontal and vertical fields of view as held, in radians. Built at the first photo,
    /// around its direction (yaw 0 of the world frame).
    init(hfov: Double, vfov: Double) {
        let H = hfov, V = vfov
        var list: [Target] = []
        let n0 = Int(ceil(2 * .pi / (0.6 * H)))
        for k in 0..<n0 { list.append(Target(id: list.count, yaw: Double(k) * 2 * .pi / Double(n0), pitch: 0, ring: 0)) }
        let perRing = Double(Int(ceil(2 * .pi / (0.675 * H))))
        for sign in [1.0, -1.0] {
            for i in 1...5 {
                let p = sign * Double(i) * 0.6 * V
                let c = cos(p)
                if c < -0.25 * V { break }
                if c < 0.05 * V {  // (a pole: one dot, then stop)
                    list.append(Target(id: list.count, yaw: 0, pitch: sign * .pi / 2, ring: Int(sign) * i))
                    break
                }
                let n = max(1, Int(ceil(perRing * c)))
                for k in 0..<n { list.append(Target(id: list.count, yaw: Double(k) * 2 * .pi / Double(n), pitch: p, ring: Int(sign) * i)) }
            }
        }
        // neighbours: left and right in a ring, and the closest-yaw dot of each adjacent ring (both ways)
        let rings = Dictionary(grouping: list, by: { $0.ring })
        for ring in rings.keys {
            let members = rings[ring]!.sorted { $0.yaw < $1.yaw }
            if members.count > 1 {
                for (k, t) in members.enumerated() {
                    list[t.id].neighbours.append(members[(k + 1) % members.count].id)
                    list[t.id].neighbours.append(members[(k + members.count - 1) % members.count].id)
                }
            }
            for other in [ring + 1, ring - 1] {
                guard let next = rings[other] else { continue }
                for t in members {
                    let nearest = next.min { yawGap($0.yaw, t.yaw) < yawGap($1.yaw, t.yaw) }!
                    list[t.id].neighbours.append(nearest.id)
                    list[nearest.id].neighbours.append(t.id)
                }
            }
        }
        for i in list.indices { list[i].neighbours = Array(Set(list[i].neighbours)).filter { $0 != i } }
        targets = list
        targets[0].state = .active
    }

    private func yawGap(_ a: Double, _ b: Double) -> Double {
        var d = abs(a - b).truncatingRemainder(dividingBy: 2 * .pi)
        if d > .pi { d = 2 * .pi - d }
        return d
    }

    var total: Int { targets.count }
    var capturedCount: Int { history.count }
    var allCaptured: Bool { history.count == targets.count }

    /// The nearest dot to a forward direction, with its angle.
    func nearest(to forward: SIMD3<Double>, among states: Set<Target.State>) -> (Target, Double)? {
        var best: (Target, Double)?
        for t in targets where states.contains(t.state) {
            let a = Orient.angle(forward, t.direction)
            if best == nil || a < best!.1 { best = (t, a) }
        }
        return best
    }

    /// A photo was taken looking along forward: the nearest dot is captured and its neighbours revealed (GCam's
    /// activate-as-you-go). The dot's id, for undo.
    @discardableResult
    func capture(forward: SIMD3<Double>) -> Int? {
        guard let (t, _) = nearest(to: forward, among: [.active, .inactive, .captured]) else { return nil }
        targets[t.id].state = .captured
        history.append(t.id)
        for nb in targets[t.id].neighbours where targets[nb].state == .inactive { targets[nb].state = .active }
        return t.id
    }

    /// The last capture undone: the dot active again, its neighbours hidden where nothing captured touches them.
    func undo() {
        guard let id = history.popLast() else { return }
        targets[id].state = .active
        for nb in targets[id].neighbours where targets[nb].state == .active {
            let touched = targets[nb].neighbours.contains { targets[$0].state == .captured }
            if !touched && nb != 0 { targets[nb].state = .inactive }
        }
        if history.isEmpty {
            for i in targets.indices { targets[i].state = i == 0 ? .active : .inactive }
        }
    }

    /// GCam's hit cone: 2.75 degrees when still, up to 3.5 when turning fast (gyro speed in rad/s).
    static func hitAngle(gyroSpeed: Double) -> Double {
        let dps = gyroSpeed * 180 / .pi
        let t = max(0, min(1, (dps - 10) / 30))
        return (2.75 + 0.75 * t) * .pi / 180
    }

    /// GCam's speed limit, by the current exposure: long exposures need a steady hand.
    static func speedLimit(exposureSeconds e: Double) -> Double {
        if e > 1.0 / 40 { return 0.05 }
        if e < 1.0 / 100 { return 1.0 }
        return 0.4
    }
}
