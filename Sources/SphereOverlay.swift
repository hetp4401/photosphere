import SceneKit
import UIKit
import simd

/// The sphere drawn around the user while capturing, as Google Camera's: a dark scene seen 1.7x wider than the lens, the
/// dots to aim at (shown only near where the camera points, fading in as it comes), each photo pasted where it was
/// taken (the live mosaic, latest on top), and a faint grid on the ground. The camera node follows the phone.
final class SphereOverlay {
    static let radius: Double = 10
    let scene = SCNScene()
    let cameraNode = SCNNode()
    private let dotsNode = SCNNode()
    private let photosNode = SCNNode()
    private var dotNodes: [Int: SCNNode] = [:]
    private var floatingDot: SCNNode?
    private var photoNodes: [SCNNode] = []
    private var hfov = 55.0, vfov = 70.0
    private var hitAlpha: [Int: Double] = [:]   // the in-range sprite's opacity per dot (GCam crossfades it)
    private var revealStart: CFTimeInterval?     // the one-time showing of every dot after the first photo
    private static let ringImage = makeRing(color: .white)
    private static let ringHit = makeRing(color: UIColor(red: 0.4, green: 1, blue: 0.5, alpha: 1))

    init(view: SCNView, transparent: Bool) {
        view.scene = scene
        view.backgroundColor = transparent ? .clear : UIColor(white: 0.12, alpha: 1)  // (GCam: BACKGROUND_COLOR 0.12)
        view.isPlaying = true
        view.rendersContinuously = true
        view.antialiasingMode = .multisampling4X
        view.allowsCameraControl = false
        view.autoenablesDefaultLighting = false
        let cam = SCNCamera()
        cam.zNear = 0.1
        cam.zFar = 100
        cam.projectionDirection = .horizontal
        cam.fieldOfView = CGFloat(hfov)
        cameraNode.camera = cam
        scene.rootNode.addChildNode(cameraNode)
        scene.rootNode.addChildNode(Self.groundGrid())
        scene.rootNode.addChildNode(photosNode)
        scene.rootNode.addChildNode(dotsNode)
    }

    /// GCam's LineTiledGroundPlaneDrawable: 35 x 35 small crosses on the ground, white at 10%, scaled to this sphere.
    private static func groundGrid() -> SCNNode {
        var verts: [SCNVector3] = []
        var idx: [Int32] = []
        let k = Float(radius), y = -1.7 * k, step = 0.3 * k, arm = 0.03 * k
        for a in 0..<35 {
            for b in 0..<35 {
                let x = (Float(a) - 17) * step, z = (Float(b) - 17) * step
                let base = Int32(verts.count)
                verts += [SCNVector3(x - arm, y, z), SCNVector3(x + arm, y, z), SCNVector3(x, y, z - arm), SCNVector3(x, y, z + arm)]
                idx += [base, base + 1, base + 2, base + 3]
            }
        }
        let g = SCNGeometry(sources: [SCNGeometrySource(vertices: verts)], elements: [SCNGeometryElement(indices: idx, primitiveType: .line)])
        let m = SCNMaterial()
        m.diffuse.contents = UIColor(white: 1, alpha: 0.1)
        m.lightingModel = .constant
        m.readsFromDepthBuffer = false
        m.writesToDepthBuffer = false
        g.firstMaterial = m
        let node = SCNNode(geometry: g)
        node.renderingOrder = -3000
        return node
    }

    /// The photo's fields of view as held, in degrees (the pasted photos are sized by them).
    func setFov(hfovDeg: Double, vfovDeg: Double) {
        hfov = hfovDeg
        vfov = vfovDeg
    }

    /// The part of the lens's width the screen shows, in degrees: the overlay's camera.
    func setVisibleHfov(_ deg: Double) {
        cameraNode.camera?.fieldOfView = CGFloat(deg)
    }

    func setCamera(_ R: Mat3) {
        cameraNode.simdTransform = Self.transform(Orient.sceneRotation(R), position: .zero)
    }

    static func transform(_ R: Mat3, position: SIMD3<Float>) -> simd_float4x4 {
        let c0 = R.columns.0, c1 = R.columns.1, c2 = R.columns.2
        return simd_float4x4(columns: (SIMD4(Float(c0.x), Float(c0.y), Float(c0.z), 0), SIMD4(Float(c1.x), Float(c1.y), Float(c1.z), 0),
                                       SIMD4(Float(c2.x), Float(c2.y), Float(c2.z), 0), SIMD4(position.x, position.y, position.z, 1)))
    }

    private static func makeRing(color: UIColor) -> UIImage {
        let s: CGFloat = 128
        return UIGraphicsImageRenderer(size: CGSize(width: s, height: s)).image { ctx in
            let c = ctx.cgContext
            c.setStrokeColor(UIColor.black.withAlphaComponent(0.35).cgColor)
            c.setLineWidth(18)
            c.strokeEllipse(in: CGRect(x: 14, y: 14, width: s - 28, height: s - 28))
            c.setStrokeColor(color.cgColor)
            c.setLineWidth(12)
            c.strokeEllipse(in: CGRect(x: 14, y: 14, width: s - 28, height: s - 28))
            c.setFillColor(color.withAlphaComponent(0.25).cgColor)
            c.fillEllipse(in: CGRect(x: 26, y: 26, width: s - 52, height: s - 52))
        }
    }

    private func makeDot(_ image: UIImage, withHit: Bool = false) -> SCNNode {
        let plane = SCNPlane(width: 0.9, height: 0.9)
        let m = SCNMaterial()
        m.diffuse.contents = image
        m.lightingModel = .constant
        m.isDoubleSided = true
        m.writesToDepthBuffer = false
        m.readsFromDepthBuffer = false
        plane.firstMaterial = m
        let node = SCNNode(geometry: plane)
        node.constraints = [SCNBillboardConstraint()]
        node.renderingOrder = 10
        if withHit {  // (the activated sprite over it, crossfaded in while the dot is in range)
            let hp = SCNPlane(width: 0.9, height: 0.9)
            let hm = SCNMaterial()
            hm.diffuse.contents = Self.ringHit
            hm.lightingModel = .constant
            hm.isDoubleSided = true
            hm.writesToDepthBuffer = false
            hm.readsFromDepthBuffer = false
            hp.firstMaterial = hm
            let h = SCNNode(geometry: hp)
            h.renderingOrder = 11
            h.opacity = 0
            node.addChildNode(h)
        }
        return node
    }

    /// GCam shows every revealed dot once, after the first photo: they fade in over about a second, hold half a second,
    /// and fade away; from then on a dot shows only near where the camera points.
    func revealDots() { revealStart = CACurrentMediaTime() }

    /// The dots as the target manager has them, drawn as GCam's TargetManager.drawTargetsOrthographic: an active dot within
    /// 12 deg of where the camera points at full size, fading and shrinking (to 0.4) out to 22 deg, hidden beyond; the one in
    /// range crossfades to the highlighted sprite. Called on every motion update (100 Hz).
    func syncDots(_ targets: [Target], inRange: Int?, forward: SIMD3<Double>) {
        floatingDot?.removeFromParentNode()
        floatingDot = nil
        var reveal = 0.0
        if let t0 = revealStart {
            let t = CACurrentMediaTime() - t0
            reveal = t < 1 ? 0.1 + 0.8 * t : (t < 1.5 ? 0.9 : 0.9 * pow(0.985, (t - 1.5) * 60))
            if reveal < 0.02 { revealStart = nil }
        }
        for t in targets {
            let node: SCNNode
            if let n = dotNodes[t.id] {
                node = n
            } else {
                node = makeDot(Self.ringImage, withHit: true)
                node.simdPosition = Orient.scenePoint(t.direction, at: Self.radius * 0.95)
                dotsNode.addChildNode(node)
                dotNodes[t.id] = node
            }
            guard t.state == .active else {
                node.isHidden = true
                continue
            }
            let a = Orient.angle(forward, t.direction) * 180 / .pi
            let f = a < 12 ? 1.0 : (a > 22 ? 0.0 : 1 - (a - 12) / 10)
            let alpha = max(f, reveal)
            node.isHidden = alpha <= 0.01
            node.opacity = CGFloat(alpha)
            let sc = Float(0.4 + 0.6 * max(f, reveal > 0.01 ? 1 : 0))
            node.scale = SCNVector3(sc, sc, sc)
            let h = (hitAlpha[t.id] ?? 0) + 0.06 * ((t.id == inRange ? 1.0 : 0.0) - (hitAlpha[t.id] ?? 0))
            hitAlpha[t.id] = h
            node.childNodes.first?.opacity = CGFloat(h)
        }
    }

    /// Before the first photo: one dot on the horizon straight ahead, wherever the camera points.
    func showFloatingDot(yaw: Double, hit: Bool) {
        if floatingDot == nil {
            floatingDot = makeDot(Self.ringImage)
            dotsNode.addChildNode(floatingDot!)
        }
        floatingDot!.simdPosition = Orient.scenePoint(Orient.direction(yaw: yaw, pitch: 0), at: Self.radius * 0.95)
        floatingDot!.geometry?.firstMaterial?.diffuse.contents = hit ? Self.ringHit : Self.ringImage
    }

    func clearDots() {
        dotsNode.childNodes.forEach { $0.removeFromParentNode() }
        dotNodes.removeAll()
        hitAlpha.removeAll()
        revealStart = nil
        floatingDot = nil
    }

    /// A photo pasted on the sphere where it was taken (a flat quad tangent to the sphere, as GCam's).
    func addPhoto(_ image: UIImage, rotation R: Mat3) {
        let d = Self.radius + 0.6
        let w = 2 * d * tan(hfov * .pi / 360), h = 2 * d * tan(vfov * .pi / 360)
        let plane = SCNPlane(width: CGFloat(w), height: CGFloat(h))
        let m = SCNMaterial()
        m.diffuse.contents = image
        m.lightingModel = .constant
        m.isDoubleSided = true
        m.readsFromDepthBuffer = false  // (GCam: no depth test, drawn in capture order: the latest photo on top)
        m.writesToDepthBuffer = false
        plane.firstMaterial = m
        let node = SCNNode(geometry: plane)
        node.renderingOrder = -2000 + photoNodes.count
        let pos = Orient.scenePoint(Orient.forward(R), at: d)
        node.simdTransform = Self.transform(Orient.sceneRotation(R), position: pos)
        node.opacity = 0
        photosNode.addChildNode(node)
        node.runAction(.fadeIn(duration: 0.35))
        photoNodes.append(node)
    }

    func removeLastPhoto() {
        photoNodes.popLast()?.removeFromParentNode()
    }

    func clearPhotos() {
        photoNodes.forEach { $0.removeFromParentNode() }
        photoNodes.removeAll()
    }
}
