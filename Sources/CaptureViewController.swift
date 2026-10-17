import AVFoundation
import AudioToolbox
import CoreLocation
import CoreMotion
import SceneKit
import UIKit
import simd

/// The capture screen, as Google Camera's Photo Sphere worked: the phone's motion sensors say where the camera points,
/// dots on a sphere say where to point it next, and a photo is taken by itself when the camera settles on a dot.
final class CaptureViewController: UIViewController, AVCapturePhotoCaptureDelegate, CLLocationManagerDelegate {
    var onFinished: ((CaptureSession) -> Void)?
    var onCancel: (() -> Void)?

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "photosphere.capture")
    private var device: AVCaptureDevice?
    private let photoOutput = AVCapturePhotoOutput()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var scnView: SCNView!
    private var overlay: SphereOverlay!
    private let motion = CMMotionManager()
    private let location = CLLocationManager()
    private let world = WorldFrame()
    private var targets: TargetManager?
    private let session = CaptureSession()
    private var hfov = 55.0 * .pi / 180, vfov = 70.0 * .pi / 180  // as held (portrait): short and long sides
    private var lockedRoll: Double?
    private var photoInFlight = false
    private var pendingRotation = matrix_identity_double3x3
    private var lastShotAt = Date.distantPast
    private var gyroSpeed = 0.0
    private var stillFor = 0          // motion updates in a row under the speed limit
    private var movedWhileExposing = 0.0  // the fastest the phone turned while a photo was taken
    private var heading: Double?
    private var finished = false

    private let hintLabel = UILabel()
    private let counterLabel = UILabel()
    private let reticle = UIView()
    private let undoButton = UIButton(type: .system)
    private let doneButton = UIButton(type: .system)
    private let cancelButton = UIButton(type: .system)

    override var prefersStatusBarHidden: Bool { true }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        let preview = AVCaptureVideoPreviewLayer(session: captureSession)
        preview.videoGravity = .resizeAspectFill
        preview.frame = view.bounds
        view.layer.addSublayer(preview)
        previewLayer = preview

        scnView = SCNView(frame: view.bounds)
        scnView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        overlay = SphereOverlay(view: scnView, transparent: true)
        view.addSubview(scnView)
        buildUI()
        checkCameraAccess()
        startMotion()
        if CLLocationManager.headingAvailable() {
            location.delegate = self
            location.startUpdatingHeading()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    private func buildUI() {
        reticle.layer.borderColor = UIColor.white.withAlphaComponent(0.9).cgColor
        reticle.layer.borderWidth = 2
        reticle.layer.cornerRadius = 12
        reticle.isUserInteractionEnabled = false
        view.addSubview(reticle)
        hintLabel.textColor = .white
        hintLabel.font = .systemFont(ofSize: 17, weight: .semibold)
        hintLabel.textAlignment = .center
        hintLabel.numberOfLines = 2
        hintLabel.layer.shadowColor = UIColor.black.cgColor
        hintLabel.layer.shadowOpacity = 0.8
        hintLabel.layer.shadowRadius = 3
        hintLabel.layer.shadowOffset = .zero
        view.addSubview(hintLabel)
        counterLabel.textColor = .white
        counterLabel.font = .monospacedDigitSystemFont(ofSize: 15, weight: .medium)
        counterLabel.textAlignment = .center
        view.addSubview(counterLabel)
        for (b, title, action) in [(cancelButton, "Cancel", #selector(cancelTapped)), (undoButton, "Undo", #selector(undoTapped)), (doneButton, "Done", #selector(doneTapped))] {
            b.setTitle(title, for: .normal)
            b.titleLabel?.font = .systemFont(ofSize: 17, weight: .semibold)
            b.setTitleColor(.white, for: .normal)
            b.backgroundColor = UIColor.black.withAlphaComponent(0.45)
            b.layer.cornerRadius = 22
            b.contentEdgeInsets = UIEdgeInsets(top: 10, left: 20, bottom: 10, right: 20)
            b.addTarget(self, action: action, for: .touchUpInside)
            view.addSubview(b)
        }
        doneButton.backgroundColor = UIColor(red: 0.2, green: 0.6, blue: 1, alpha: 0.9)
        [reticle, hintLabel, counterLabel, cancelButton, undoButton, doneButton].forEach { $0.translatesAutoresizingMaskIntoConstraints = false }
        let g = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            reticle.centerXAnchor.constraint(equalTo: view.centerXAnchor), reticle.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            reticle.widthAnchor.constraint(equalToConstant: 150), reticle.heightAnchor.constraint(equalToConstant: 150),
            hintLabel.topAnchor.constraint(equalTo: g.topAnchor, constant: 24), hintLabel.leadingAnchor.constraint(equalTo: g.leadingAnchor, constant: 24),
            hintLabel.trailingAnchor.constraint(equalTo: g.trailingAnchor, constant: -24),
            counterLabel.topAnchor.constraint(equalTo: hintLabel.bottomAnchor, constant: 8), counterLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cancelButton.leadingAnchor.constraint(equalTo: g.leadingAnchor, constant: 20), cancelButton.bottomAnchor.constraint(equalTo: g.bottomAnchor, constant: -24),
            doneButton.trailingAnchor.constraint(equalTo: g.trailingAnchor, constant: -20), doneButton.bottomAnchor.constraint(equalTo: g.bottomAnchor, constant: -24),
            undoButton.centerXAnchor.constraint(equalTo: view.centerXAnchor), undoButton.bottomAnchor.constraint(equalTo: g.bottomAnchor, constant: -24),
        ])
        undoButton.isHidden = true
        doneButton.isHidden = true
        hintLabel.text = "Point the camera at the horizon"
    }

    // MARK: the camera

    private func checkCameraAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: configureCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in DispatchQueue.main.async { ok ? self.configureCamera() : self.noCamera() } }
        default: noCamera()
        }
    }

    private func noCamera() {
        hintLabel.text = "No camera. Allow it in Settings > PhotoSphere."
    }

    private func configureCamera() {
        sessionQueue.async {
            self.captureSession.beginConfiguration()
            self.captureSession.sessionPreset = .photo
            guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: dev), self.captureSession.canAddInput(input) else {
                self.captureSession.commitConfiguration()
                DispatchQueue.main.async { self.noCamera() }
                return
            }
            self.captureSession.addInput(input)
            if self.captureSession.canAddOutput(self.photoOutput) { self.captureSession.addOutput(self.photoOutput) }
            self.photoOutput.maxPhotoQualityPrioritization = .speed  // (no Deep Fusion: each photo processed alike)
            self.captureSession.commitConfiguration()
            self.device = dev
            // the lens's field of view: AVFoundation gives the sensor's horizontal one (its long side); as held in
            // portrait that is the vertical, and the horizontal follows from the 4:3 photo
            let longFov = Double(dev.activeFormat.videoFieldOfView) * .pi / 180
            let shortFov = 2 * atan(tan(longFov / 2) * 3.0 / 4.0)
            self.captureSession.startRunning()
            DispatchQueue.main.async {
                self.hfov = shortFov
                self.vfov = longFov
                self.overlay.setFov(hfovDeg: shortFov * 180 / .pi, vfovDeg: longFov * 180 / .pi)
                self.session.setCamera(width: 0, height: 0, focalPx: 0, hfovDeg: shortFov * 180 / .pi, vfovDeg: longFov * 180 / .pi)
                if let c = self.previewLayer?.connection { Self.portrait(c) }
            }
        }
    }

    private static func portrait(_ c: AVCaptureConnection) {
        if #available(iOS 17.0, *) {
            if c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        } else if c.isVideoOrientationSupported {
            c.videoOrientation = .portrait
        }
    }

    // MARK: motion

    private func startMotion() {
        guard motion.isDeviceMotionAvailable else {
            hintLabel.text = "No motion sensors on this device"
            return
        }
        motion.deviceMotionUpdateInterval = 1.0 / 100
        motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] m, _ in
            guard let self, let m, !self.finished else { return }
            self.handle(m)
        }
    }

    private func handle(_ m: CMDeviceMotion) {
        let refFromDevice = Orient.refFromDevice(m)
        let R = world.worldFromCam(refFromDevice: refFromDevice)
        let rr = m.rotationRate
        gyroSpeed = simd_length(SIMD3(rr.x, rr.y, rr.z))
        overlay.setCamera(R)
        let f = Orient.forward(R), pitch = Orient.pitch(R), roll = Orient.roll(R)
        let hit = TargetManager.hitAngle(gyroSpeed: gyroSpeed)
        let exposure = device?.exposureDuration.seconds ?? 1.0 / 60
        let limit = TargetManager.speedLimit(exposureSeconds: exposure)
        stillFor = gyroSpeed > limit ? 0 : stillFor + 1
        let tooFast = stillFor < 4  // (under the limit for 40 ms: not just passing through)
        if photoInFlight { movedWhileExposing = max(movedWhileExposing, gyroSpeed) }
        let canShoot = device != nil && !photoInFlight && Date().timeIntervalSince(lastShotAt) > 0.4

        if let targets {
            guard let near = targets.nearest(to: f, among: [.active]) else { return }
            let (t, a) = near
            overlay.syncDots(targets.targets, inRange: a < 15 * .pi / 180 ? t.id : nil)
            let rollOK = abs(pitch) > 40 * .pi / 180 || Self.within(roll, lockedRoll ?? 0, 20 * .pi / 180)
            if a < hit && canShoot {
                if tooFast {
                    hintLabel.text = "Hold still"
                } else if !rollOK {
                    hintLabel.text = "Keep the phone \(lockedRoll == 0 ? "upright" : "turned as the first photo")"
                } else {
                    takePhoto(rotation: R)
                }
            } else if photoInFlight {
                hintLabel.text = "Hold still…"
            } else {
                hintLabel.text = a < 15 * .pi / 180 ? "Centre the dot" : "Turn to the next dot"
            }
        } else {
            let hitting = abs(pitch) < hit
            overlay.showFloatingDot(yaw: Orient.yaw(R), hit: hitting)
            if pitch > 20 * .pi / 180 {
                hintLabel.text = "Tilt down to the horizon"
            } else if pitch < -20 * .pi / 180 {
                hintLabel.text = "Tilt up to the horizon"
            } else if hitting && canShoot {
                if tooFast {
                    hintLabel.text = "Hold still"
                } else {
                    firstPhoto(refFromDevice: refFromDevice)
                }
            } else {
                hintLabel.text = "Centre the dot on the horizon"
            }
        }
    }

    private static func within(_ a: Double, _ b: Double, _ tol: Double) -> Bool {
        var d = abs(a - b).truncatingRemainder(dividingBy: 2 * .pi)
        if d > .pi { d = 2 * .pi - d }
        return d <= tol
    }

    private func firstPhoto(refFromDevice: Mat3) {
        lockLens()
        world.fix(refFromDevice: refFromDevice)
        let R = world.worldFromCam(refFromDevice: refFromDevice)
        let roll = Orient.roll(R)
        let uprights: [Double] = [0, .pi / 2, -.pi / 2, .pi]
        lockedRoll = uprights.min(by: { abs($0 - roll) < abs($1 - roll) })
        targets = TargetManager(hfov: hfov, vfov: vfov)
        overlay.clearDots()
        if let h = heading { session.setHeading(h) }
        takePhoto(rotation: R)
    }

    /// From the first photo on, the lens stays focused at infinity (GCam: FOCUS_MODE_INFINITY), so every photo has the same
    /// field of view (autofocus changes it a little), and the white balance stays as it was.
    private func lockLens() {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }  // (where it focused on the first view: a room's distance)
        if dev.isExposureModeSupported(.locked) { dev.exposureMode = .locked }  // (one exposure for the whole sphere)
        if dev.isWhiteBalanceModeSupported(.locked) { dev.whiteBalanceMode = .locked }
        dev.unlockForConfiguration()
    }

    private func unlockLens() {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
        if dev.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { dev.whiteBalanceMode = .continuousAutoWhiteBalance }
        dev.unlockForConfiguration()
    }

    // MARK: photos

    private func takePhoto(rotation R: Mat3) {
        photoInFlight = true
        pendingRotation = R
        movedWhileExposing = 0
        lastShotAt = Date()
        targets?.capture(forward: Orient.forward(R))
        if let t = targets { counterLabel.text = "\(t.capturedCount) of \(t.total)" }
        hintLabel.text = "Hold still…"
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        settings.photoQualityPrioritization = .speed
        if let c = photoOutput.connection(with: .video) { Self.portrait(c) }
        photoOutput.capturePhoto(with: settings, delegate: self)
        AudioServicesPlaySystemSound(1108)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let rotation = pendingRotation
        let exposure = device?.exposureDuration.seconds ?? 0
        guard error == nil, let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else {
            DispatchQueue.main.async { self.photoFailed() }
            return
        }
        if movedWhileExposing > 2 * TargetManager.speedLimit(exposureSeconds: exposure) {  // (it turned while exposing: blurred)
            DispatchQueue.main.async { self.photoFailed(blurred: true) }
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let thumb = self.session.add(photo: image, rotation: rotation, exposure: exposure)
            DispatchQueue.main.async {
                self.photoInFlight = false
                if let thumb { self.overlay.addPhoto(thumb, rotation: rotation) }
                self.undoButton.isHidden = false
                self.doneButton.isHidden = false
                if let t = self.targets, t.allCaptured { self.finish() }
            }
        }
    }

    private func photoFailed(blurred: Bool = false) {
        photoInFlight = false
        targets?.undo()
        if session.count == 0 { resetToStart() }
        hintLabel.text = blurred ? "Moved too much, hold still" : "The photo failed, try again"
    }

    private func resetToStart() {
        unlockLens()
        targets = nil
        world.release()
        lockedRoll = nil
        overlay.clearDots()
        overlay.clearPhotos()
        counterLabel.text = ""
        undoButton.isHidden = true
        doneButton.isHidden = true
    }

    @objc private func undoTapped() {
        guard !photoInFlight, session.count > 0 else { return }
        session.removeLast()
        targets?.undo()
        overlay.removeLastPhoto()
        if session.count == 0 { resetToStart() } else if let t = targets { counterLabel.text = "\(t.capturedCount) of \(t.total)" }
    }

    @objc private func doneTapped() {
        guard session.count > 0 else { return }
        finish()
    }

    @objc private func cancelTapped() {
        stop()
        session.discard()
        onCancel?()
    }

    private func finish() {
        guard !finished else { return }
        stop()
        onFinished?(session)
    }

    private func stop() {
        finished = true
        motion.stopDeviceMotionUpdates()
        location.stopUpdatingHeading()
        sessionQueue.async { if self.captureSession.isRunning { self.captureSession.stopRunning() } }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        heading = newHeading.trueHeading >= 0 ? newHeading.trueHeading : newHeading.magneticHeading
    }
}
