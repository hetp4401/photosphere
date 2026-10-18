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

        /// The piece of a sphere (radius r, seen from its centre) that a crop of a full equirect canvas covers, its texture
        /// coordinates running over the crop (0,0 its top-left).
        static func patch(fullW: Double, fullH: Double, cropW: Double, cropH: Double, left: Double, top: Double, radius r: Double) -> SCNGeometry {
            let nu = max(4, Int(160 * cropW / fullW)), nv = max(2, Int(80 * cropH / fullH))
            var verts: [SCNVector3] = [], uvs: [CGPoint] = [], idx: [Int32] = []
            for b in 0...nv {
                let t = Double(b) / Double(nv)
                let phi = Double.pi / 2 - (top + t * cropH) / fullH * Double.pi
                for a in 0...nu {
                    let s = Double(a) / Double(nu)
                    let lam = (left + s * cropW) / fullW * 2 * Double.pi - Double.pi
                    verts.append(SCNVector3(Float(r * cos(phi) * sin(lam)), Float(r * sin(phi)), Float(-r * cos(phi) * cos(lam))))
                    uvs.append(CGPoint(x: s, y: t))
                }
            }
            for b in 0..<nv {
                for a in 0..<nu {
                    let i0 = Int32(b * (nu + 1) + a), i2 = i0 + Int32(nu + 1)
                    idx += [i0, i2, i0 + 1, i0 + 1, i2, i2 + 1]
                }
            }
            return SCNGeometry(sources: [SCNGeometrySource(vertices: verts), SCNGeometrySource(textureCoordinates: uvs)],
                               elements: [SCNGeometryElement(indices: idx, primitiveType: .triangles)])
        }

        func attach(view: SCNView, url: URL) {
            view.scene = scene
            view.backgroundColor = .black
            view.isPlaying = true
            view.rendersContinuously = true
            view.antialiasingMode = .multisampling4X
            // only the part of the sphere the photos cover (GPano's crop of the full canvas), black around it; longitude
            // 0 (the canvas's centre) ahead along -z, east to the right
            let crop = XMP.gpanoCrop(of: url)
            let image = UIImage(contentsOfFile: url.path)
            let size = image.map { ($0.size.width * $0.scale, $0.size.height * $0.scale) } ?? (2, 1)
            let (fw, fh, cw, ch, left, top) = crop ?? (Double(size.0), Double(size.0) / 2, Double(size.0), Double(size.1), 0, (Double(size.0) / 2 - Double(size.1)) / 2)
            let m = SCNMaterial()
            m.diffuse.contents = image
            m.diffuse.minificationFilter = .linear  // (mipmapped and anisotropic: smooth when small or at a slant)
            m.diffuse.magnificationFilter = .linear
            m.diffuse.mipFilter = .linear
            m.diffuse.maxAnisotropy = 8
            m.isDoubleSided = true
            m.lightingModel = .constant
            let geometry = Self.patch(fullW: fw, fullH: fh, cropW: cw, cropH: ch, left: left, top: top, radius: 10)
            geometry.firstMaterial = m
            scene.rootNode.addChildNode(SCNNode(geometry: geometry))
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
