import AVFoundation
import AudioToolbox
import CoreLocation
import CoreMotion
import ImageIO
import SceneKit
import UIKit
import simd

/// The capture screen, as Google Camera's Photo Sphere worked: the phone's motion sensors alone say where the camera
/// points, dots on a sphere say where to point it next, and a photo is taken by itself when the camera settles on a dot.
/// The view is GCam's: the scene seen 1.7x wider than the lens on a dark ground, the camera's frame a fixed outline in the
/// middle with the live picture inside it until the first photo, then only the photos taken, pasted where they belong.
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
    private var landscape: Bool?   // how the first photo was held (GCam locks the family: upright/upside down, or the two landscapes)
    private var photoInFlight = false
    private var pendingRotation = matrix_identity_double3x3
    private var lastShotAt = Date.distantPast
    private var gyroSpeed = 0.0
    private var stillFor = 0          // motion updates in a row under the speed limit
    private var heading: Double?
    private var finished = false
    private var countdownStart: CFTimeInterval?  // GCam's hold on the start dot (the reticle filling over 1.2 s)
    private var skippedTooFast = false           // on a dot too fast to shoot (GCam: PhotoSkippedTooFast)
    private var tooFastFrom = 0.0                // when to say so: 0.25 s after such a pass ends without a photo, for 0.75 s
    private let outline = CAShapeLayer()         // the camera's frame
    private let countdownLayer = CAShapeLayer()

    private let hintLabel = UILabel()
    private let counterLabel = UILabel()
    private let tipLabel = UILabel()  // GCam's tutorial ("keep the camera centered"), until the first photo
    private let reticle = UIView()
    private let undoButton = UIButton(type: .system)
    private let doneButton = UIButton(type: .system)
    private let cancelButton = UIButton(type: .system)

    override var prefersStatusBarHidden: Bool { true }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor(white: 0.12, alpha: 1)
        UIApplication.shared.isIdleTimerDisabled = true  // (GCam reset the screen timeout at every photo: a capture takes minutes without a touch)
        scnView = SCNView(frame: view.bounds)
        overlay = SphereOverlay(view: scnView, transparent: false)
        view.addSubview(scnView)
        let preview = AVCaptureVideoPreviewLayer(session: captureSession)
        preview.videoGravity = .resizeAspectFill  // (inside the frame's outline, which has the photo's shape: the lens's whole frame)
        view.layer.addSublayer(preview)
        previewLayer = preview
        outline.strokeColor = UIColor.white.cgColor
        outline.fillColor = UIColor.clear.cgColor
        outline.lineWidth = 3
        view.layer.addSublayer(outline)
        buildUI()
        checkCameraAccess()
        startMotion()
        if CLLocationManager.headingAvailable() {
            location.delegate = self
            location.startUpdatingHeading()
        }
    }

    /// GCam's view (LightCycleRenderer): the screen's width spans 1.7x the tangent of the lens's long side, so the camera's
    /// frame is a fixed rectangle about 44% of the width in the middle; the live picture fills it until the first photo.
    private func layoutPreview() {
        scnView.frame = view.bounds
        let tanV = tan(vfov / 2), tanH = tan(hfov / 2)
        overlay.setVisibleHfov(2 * atan(1.7 * tanV) * 180 / .pi)
        let scale = (view.bounds.width / 2) / CGFloat(1.7 * tanV)  // (points per unit of tangent, both ways)
        let fw = 2 * CGFloat(tanH) * scale, fh = 2 * CGFloat(tanV) * scale
        let rect = CGRect(x: view.bounds.midX - fw / 2, y: view.bounds.midY - fh / 2, width: fw, height: fh)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        previewLayer?.frame = rect
        outline.path = UIBezierPath(rect: rect).cgPath
        CATransaction.commit()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        layoutPreview()
    }

    private func buildUI() {
        reticle.layer.borderColor = UIColor.white.withAlphaComponent(0.5).cgColor  // (GCam's reticle: at the centre, half opaque)
        reticle.layer.borderWidth = 2
        reticle.layer.cornerRadius = 22
        reticle.isUserInteractionEnabled = false
        countdownLayer.path = UIBezierPath(arcCenter: CGPoint(x: 22, y: 22), radius: 17, startAngle: -.pi / 2, endAngle: 1.5 * .pi, clockwise: true).cgPath
        countdownLayer.strokeColor = UIColor.white.cgColor
        countdownLayer.fillColor = UIColor.clear.cgColor
        countdownLayer.lineWidth = 6
        countdownLayer.strokeEnd = 0
        reticle.layer.addSublayer(countdownLayer)
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
        tipLabel.text = "Hold your device in a fixed position. Tilt or turn your hand as needed."  // (GCam's words)
        tipLabel.textColor = UIColor.white.withAlphaComponent(0.85)
        tipLabel.font = .systemFont(ofSize: 15, weight: .regular)
        tipLabel.textAlignment = .center
        tipLabel.numberOfLines = 0
        view.addSubview(tipLabel)
        [reticle, hintLabel, counterLabel, tipLabel, cancelButton, undoButton, doneButton].forEach { $0.translatesAutoresizingMaskIntoConstraints = false }
        let g = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            reticle.centerXAnchor.constraint(equalTo: view.centerXAnchor), reticle.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            reticle.widthAnchor.constraint(equalToConstant: 44), reticle.heightAnchor.constraint(equalToConstant: 44),
            hintLabel.topAnchor.constraint(equalTo: g.topAnchor, constant: 24), hintLabel.leadingAnchor.constraint(equalTo: g.leadingAnchor, constant: 24),
            hintLabel.trailingAnchor.constraint(equalTo: g.trailingAnchor, constant: -24),
            counterLabel.topAnchor.constraint(equalTo: hintLabel.bottomAnchor, constant: 8), counterLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            tipLabel.bottomAnchor.constraint(equalTo: cancelButton.topAnchor, constant: -20), tipLabel.leadingAnchor.constraint(equalTo: g.leadingAnchor, constant: 32),
            tipLabel.trailingAnchor.constraint(equalTo: g.trailingAnchor, constant: -32),
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
            self.photoOutput.maxPhotoQualityPrioritization = .balanced  // (the camera's own processing, as the Camera app)
            if #available(iOS 17.0, *), self.photoOutput.isZeroShutterLagSupported {
                self.photoOutput.isZeroShutterLagEnabled = true  // (the frame delivered is the one from the still moment that fired it)
            }
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
                self.layoutPreview()
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
        let R = world.worldFromCam(refFromDevice: refFromDevice)  // (the sensors alone, as GCam's EKF: nothing seen corrects it)
        let rr = m.rotationRate
        gyroSpeed = simd_length(SIMD3(rr.x, rr.y, rr.z))
        overlay.setCamera(R)
        let f = Orient.forward(R), pitch = Orient.pitch(R), roll = Orient.fullRoll(R)
        let hit = TargetManager.hitAngle(gyroSpeed: gyroSpeed)
        let exposure = device?.exposureDuration.seconds ?? 1.0 / 60
        let limit = TargetManager.speedLimit(exposureSeconds: exposure)
        stillFor = gyroSpeed > limit ? 0 : stillFor + 1
        let tooFast = stillFor < 4  // (under the limit for 40 ms: not just passing through)
        let canShoot = device != nil && !photoInFlight && Date().timeIntervalSince(lastShotAt) > 0.4
        // GCam's roll check (DeviceOrientationStatus): within 20 deg of upright or upside down, or of the two landscape ways
        // (the way the first photo was held decides); not checked pointing up past 40 deg (nor, here, near straight down,
        // where the roll is undefined)
        let rollOK = pitch > 40 * .pi / 180 || pitch < -70 * .pi / 180 || Self.rollOK(roll, landscape: landscape)

        if let targets {
            guard let near = targets.nearest(to: f, among: [.active]) else { return }
            let (_, a) = near
            // (GCam's GetTargetInRange: the nearest dot of any kind within 15 deg, highlighted only if it's still to take)
            let any = targets.nearest(to: f, among: [.active, .inactive, .captured])
            let inRange = any.flatMap { $0.1 < 15 * .pi / 180 && $0.0.state == .active ? $0.0.id : nil }
            overlay.syncDots(targets.targets, inRange: inRange, forward: f)
            let now = CACurrentMediaTime()
            if a < hit && canShoot {
                if tooFast {
                    skippedTooFast = true  // (GCam says nothing while on the dot)
                } else if !rollOK {
                    hintLabel.text = "Keep the phone straight"
                } else {
                    skippedTooFast = false
                    tooFastFrom = 0
                    takePhoto(rotation: R)
                }
            } else {
                if skippedTooFast && !photoInFlight {  // (a pass over a dot ended without a photo)
                    skippedTooFast = false
                    tooFastFrom = now + 0.25
                }
                if photoInFlight {
                    hintLabel.text = "Hold still\u{2026}"
                } else {
                    hintLabel.text = a < 15 * .pi / 180 ? "Centre the dot" : "Turn to the next dot"
                }
            }
            if tooFastFrom > 0 && now >= tooFastFrom && now < tooFastFrom + 0.75 { hintLabel.text = "Too fast. Slow down on the dot" }
        } else {
            let hitting = abs(pitch) < hit
            overlay.showFloatingDot(yaw: Orient.yaw(R), hit: hitting)
            var counting = false
            if pitch > 20 * .pi / 180 {
                hintLabel.text = "Tilt down to the horizon"
            } else if pitch < -20 * .pi / 180 {
                hintLabel.text = "Tilt up to the horizon"
            } else if !rollOK {
                hintLabel.text = "Hold the phone straight"
            } else if hitting && canShoot && !tooFast {
                // GCam's start: hold on the dot while the reticle fills (three steps, 1.2 s), the lens focusing once meanwhile
                counting = true
                let now = CACurrentMediaTime()
                if countdownStart == nil {
                    countdownStart = now
                    focusOnce()
                }
                let elapsed = now - (countdownStart ?? now)
                countdownLayer.strokeEnd = CGFloat(min(1, Double(Int(elapsed / 0.4) + 1) / 3))
                hintLabel.text = "Hold still"
                if elapsed >= 1.2 && !(device?.isAdjustingFocus ?? false) {
                    firstPhoto(refFromDevice: refFromDevice)
                    return
                }
            } else {
                hintLabel.text = hitting && tooFast ? "Hold still" : "Centre the dot on the horizon"
            }
            if !counting {
                countdownStart = nil
                countdownLayer.strokeEnd = 0
            }
        }
    }

    /// Within 20 deg of upright or upside down, or of the two landscape ways; once a capture started, only its own pair.
    private static func rollOK(_ roll: Double, landscape: Bool?) -> Bool {
        let tol = 20 * Double.pi / 180
        let portrait = within(roll, 0, tol) || within(roll, .pi, tol)
        let land = within(roll, .pi / 2, tol) || within(roll, -.pi / 2, tol)
        switch landscape {
        case .some(true): return land
        case .some(false): return portrait
        case .none: return portrait || land
        }
    }

    private static func within(_ a: Double, _ b: Double, _ tol: Double) -> Bool {
        var d = abs(a - b).truncatingRemainder(dividingBy: 2 * .pi)
        if d > .pi { d = 2 * .pi - d }
        return d <= tol
    }

    private func firstPhoto(refFromDevice: Mat3) {
        countdownStart = nil
        countdownLayer.strokeEnd = 0
        lockLens()
        world.fix(refFromDevice: refFromDevice)
        let R = world.worldFromCam(refFromDevice: refFromDevice)
        let roll = Orient.fullRoll(R)
        let land = Self.within(roll, .pi / 2, .pi / 4) || Self.within(roll, -.pi / 2, .pi / 4)
        landscape = land
        // the dots laid out by how the phone is held (GCam: landscape, the long side runs along the rings); "closer" lays
        // them out as for a lens 18% narrower: about 46 photos instead of 32, smaller turns between neighbours
        let k = UserDefaults.standard.bool(forKey: "closerDots") ? 0.82 : 1.0
        targets = land ? TargetManager(hfov: vfov * k, vfov: hfov * k) : TargetManager(hfov: hfov * k, vfov: vfov * k)
        tipLabel.isHidden = true
        overlay.clearDots()
        if let h = heading { session.setHeading(h) }
        takePhoto(rotation: R)
        previewLayer?.isHidden = true  // (GCam: from the first photo on, the mosaic and the frame's outline only)
        overlay.revealDots()
    }

    /// From the first photo on the focus stays where the start's one autofocus put it (GCam's path for phones that can't
    /// focus at infinity: one focus at the start dot, then held). Exposure and white balance keep adjusting photo by photo,
    /// as GCam left them (it never locked them); the stitcher evens the photos out.
    private func lockLens() {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
        dev.unlockForConfiguration()
    }

    /// One autofocus at the middle of the frame (GCam's autoFocusIfRequired at the start dot).
    private func focusOnce() {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        if dev.isFocusPointOfInterestSupported { dev.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5) }
        if dev.isFocusModeSupported(.autoFocus) { dev.focusMode = .autoFocus }
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
        lastShotAt = Date()
        targets?.capture(forward: Orient.forward(R))
        if let t = targets { counterLabel.text = "\(t.capturedCount) of \(t.total)" }
        hintLabel.text = "Hold still…"
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg, AVVideoCompressionPropertiesKey: [AVVideoQualityKey: 0.95]])
        settings.photoQualityPrioritization = .balanced
        if let c = photoOutput.connection(with: .video) { Self.portrait(c) }
        photoOutput.capturePhoto(with: settings, delegate: self)  // (the system's shutter sound marks the exposure, as GCam's onShutter)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didCapturePhotoFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
        DispatchQueue.main.async { if self.hintLabel.text == "Hold still\u{2026}" { self.hintLabel.text = "" } }  // (exposed: free to turn)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let rotation = pendingRotation
        let exif = photo.metadata[kCGImagePropertyExifDictionary as String] as? [String: Any]
        let exposure = (exif?[kCGImagePropertyExifExposureTime as String] as? Double) ?? device?.exposureDuration.seconds ?? 0  // (GCam read it from the EXIF)
        guard error == nil, let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else {
            DispatchQueue.main.async { self.photoFailed() }
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let thumb = self.session.add(jpeg: data, photo: image, rotation: rotation, exposure: exposure)
            DispatchQueue.main.async {
                self.photoInFlight = false
                if let thumb { self.overlay.addPhoto(thumb, rotation: rotation) }
                self.undoButton.isHidden = false
                self.doneButton.isHidden = false
                if let t = self.targets, t.allCaptured { self.finish() }
            }
        }
    }

    private func photoFailed() {
        photoInFlight = false
        targets?.undo()
        if session.count == 0 { resetToStart() }
        hintLabel.text = "The photo failed, try again"
    }

    private func resetToStart() {
        unlockLens()
        targets = nil
        world.release()
        landscape = nil
        previewLayer?.isHidden = false
        tipLabel.isHidden = false
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
        guard session.count > 0, !photoInFlight else { return }  // (GCam: not while a photo is being taken)
        finish()
    }

    @objc private func cancelTapped() {
        guard session.count > 0 else { return leave() }
        // (GCam asked before throwing a capture away)
        let alert = UIAlertController(title: "Discard this photo sphere?", message: "The \(session.count) photos taken so far will be deleted.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Keep shooting", style: .cancel))
        alert.addAction(UIAlertAction(title: "Discard", style: .destructive) { _ in self.leave() })
        present(alert, animated: true)
    }

    private func leave() {
        stop()
        session.discard()
        onCancel?()
    }

    private func finish() {
        guard !finished else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.success)  // (GCam played a sound when the capture was complete)
        stop()
        onFinished?(session)
    }

    private func stop() {
        finished = true
        UIApplication.shared.isIdleTimerDisabled = false
        motion.stopDeviceMotionUpdates()
        location.stopUpdatingHeading()
        sessionQueue.async { if self.captureSession.isRunning { self.captureSession.stopRunning() } }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        heading = newHeading.trueHeading >= 0 ? newHeading.trueHeading : newHeading.magneticHeading
    }
}
