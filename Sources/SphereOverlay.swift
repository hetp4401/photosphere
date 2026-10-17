import SceneKit
import UIKit
import simd

/// The sphere drawn around the user while capturing: the dots to aim at, and each photo taken pasted where it was
/// taken (Google Camera's live mosaic). The camera node follows the phone.
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
    private static let ringImage = makeRing(color: .white)
    private static let ringHit = makeRing(color: UIColor(red: 0.4, green: 1, blue: 0.5, alpha: 1))

    init(view: SCNView, transparent: Bool) {
        view.scene = scene
        view.backgroundColor = transparent ? .clear : .black
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
        scene.rootNode.addChildNode(photosNode)
        scene.rootNode.addChildNode(dotsNode)
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

    private func makeDot(_ image: UIImage) -> SCNNode {
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
        return node
    }

    /// The dots as the target manager has them: active ones shown, the one in range highlighted, the rest hidden.
    func syncDots(_ targets: [Target], inRange: Int?) {
        floatingDot?.removeFromParentNode()
        floatingDot = nil
        for t in targets {
            let node: SCNNode
            if let n = dotNodes[t.id] {
                node = n
            } else {
                node = makeDot(Self.ringImage)
                node.simdPosition = Orient.scenePoint(t.direction, at: Self.radius * 0.95)
                dotsNode.addChildNode(node)
                dotNodes[t.id] = node
            }
            node.isHidden = t.state != .active
            node.geometry?.firstMaterial?.diffuse.contents = t.id == inRange ? Self.ringHit : Self.ringImage
            node.scale = t.id == inRange ? SCNVector3(1.3, 1.3, 1.3) : SCNVector3(1, 1, 1)
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
        plane.firstMaterial = m
        let node = SCNNode(geometry: plane)
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
