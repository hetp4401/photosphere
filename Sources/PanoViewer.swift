import CoreMotion
import SceneKit
import SwiftUI
import UIKit

/// Looks around inside a finished sphere: the phone's motion turns the view, a drag turns it too, a pinch zooms.
struct PanoViewer: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        context.coordinator.attach(view: v, url: url)
        return v
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject {
        private let scene = SCNScene()
        private let cameraNode = SCNNode()
        private let motion = CMMotionManager()
        private var dragYaw = 0.0, dragPitch = 0.0
        private var startYaw = 0.0, startPitch = 0.0
        private var motionR = Mat3(rows: [SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, -1, 0)])  // (no motion: level, ahead)
        private var fov = 75.0

        func attach(view: SCNView, url: URL) {
            view.scene = scene
            view.backgroundColor = .black
            view.isPlaying = true
            view.rendersContinuously = true
            view.antialiasingMode = .multisampling4X
            let sphere = SCNSphere(radius: 10)
            sphere.segmentCount = 96
            let m = SCNMaterial()
            m.diffuse.contents = UIImage(contentsOfFile: url.path)
            m.diffuse.contentsTransform = SCNMatrix4Translate(SCNMatrix4MakeScale(-1, 1, 1), 1, 0, 0)  // (seen from inside)
            m.cullMode = .front
            m.lightingModel = .constant
            sphere.firstMaterial = m
            let node = SCNNode(geometry: sphere)
            node.eulerAngles.y = -.pi / 2  // (the texture's centre, longitude 0, ahead: along -z)
            scene.rootNode.addChildNode(node)
            let cam = SCNCamera()
            cam.zNear = 0.1
            cam.zFar = 50
            cam.projectionDirection = .vertical
            cam.fieldOfView = CGFloat(fov)
            cameraNode.camera = cam
            scene.rootNode.addChildNode(cameraNode)
            view.addGestureRecognizer(UIPanGestureRecognizer(target: self, action: #selector(pan(_:))))
            view.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch(_:))))
            if motion.isDeviceMotionAvailable {
                motion.deviceMotionUpdateInterval = 1.0 / 60
                motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] m, _ in
                    guard let self, let m else { return }
                    self.motionR = Orient.refFromDevice(m) * Orient.camFlip
                    self.update()
                }
            }
            update()
        }

        private func update() {
            // the world is the reference frame with y down, as the stitcher's; the drag adds a yaw and a pitch
            let wfr = Orient.worldFromRef(forwardInRef: SIMD3(0, 1, 0))
            let R = wfr * motionR
            let yaw = Orient.yaw(R) + dragYaw
            let pitch = max(-.pi / 2 + 0.01, min(.pi / 2 - 0.01, Orient.pitch(R) + dragPitch))
            let rotX = Mat3(rows: [SIMD3(1, 0, 0), SIMD3(0, cos(pitch), -sin(pitch)), SIMD3(0, sin(pitch), cos(pitch))])
            let Rv = Orient.rotY(yaw) * rotX
            cameraNode.simdTransform = SphereOverlay.transform(Orient.sceneRotation(Rv), position: .zero)
        }

        @objc private func pan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began {
                startYaw = dragYaw
                startPitch = dragPitch
            }
            let scale = fov * .pi / 180 / Double(g.view?.bounds.height ?? 800)
            dragYaw = startYaw - Double(t.x) * scale
            dragPitch = startPitch + Double(t.y) * scale
            update()
        }

        @objc private func pinch(_ g: UIPinchGestureRecognizer) {
            fov = max(30, min(110, fov / Double(g.scale)))
            g.scale = 1
            cameraNode.camera?.fieldOfView = CGFloat(fov)
        }
    }
}
