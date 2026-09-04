# Test fixtures

Three committed images. All are inputs to the real-inference lanes in
`tests/audit_avanalyze_api.rs` and `src/tests/pixel_door.rs`, which run
against the real Apple Vision framework and therefore only on an Apple host.

```
apollo11_crew.jpg
  source:   https://commons.wikimedia.org/wiki/File:Apollo_11_Crew.jpg
            (original: https://upload.wikimedia.org/wikipedia/commons/3/3d/Apollo_11_Crew.jpg)
  credit:   NASA — the Apollo 11 prime crew (Armstrong, Collins, Aldrin), 1969
  licence:  Public domain (a work of the U.S. federal government)
  fetched:  2026-09-01
  original: 4200x3300, 1628582 bytes
  committed: 640x503, 83864 bytes, produced with
             sips -Z 640 --setProperty formatOptions 70
  sha256:   6e20f9e893b6103539601fae122594ca668b9bbbc77fafad22d0d1c79682e8ee
  why:      three frontal, well-separated faces, so every face's own capture-quality
            and landmark observations must join back to ITSELF; a mis-assignment lands
            keypoints in a neighbour's box.
  verified on this host at the crate's default face gates (rectangles min_confidence 0.1,
  capture min_capture_quality 0.1), Vision requests pinned to Revision3:
    3 faces; confidence 0.8567 / 0.8758 / 0.8781;
    capture_quality 0.4387 / 0.3569 / 0.5088; all three with a complete five-point reduction,
    every point inside its own face's box.
  (the same image detects 3 faces at every scale tried from 384 px to the 4200 px original)
  also carries the 3-D body pose. Verified on this host against Objective-C reading the same
  frame through the same request at Revision1 (1 observation, confidence 1.0, bodyHeight 1.800,
  heightEstimation Reference, 17 joints, none of them carrying a confidence):
    human_root_3D       ( 0.000000,  0.000000,  0.000000)   <- model space is rooted at the hip
    human_left_hip_3D   ( 0.156500,  0.000000,  0.000000)
    human_right_hip_3D  (-0.156500,  0.000000,  0.000000)
    human_center_head_3D(-0.050581,  0.631640,  0.174870)
    human_top_head_3D   (-0.048227,  0.748985,  0.182247)
    human_left_ankle_3D ( 0.244692, -0.612459, -0.078243)
  metres, model space. `src/tests/body_pose.rs` asserts the Rust road against these.

airport_keyframe.jpg
  the desktop's exact keyframe-extraction output for 01_airport.mp4 (AreaResampler downscale to
  288x512 + jpeg-encoder q85) at the first frame whose 3-D body-pose detector raised an
  Objective-C exception. 25930 bytes. Carries no face: it is the zero-face lane for the face
  fusion, and the no-abort regression fixture for every entry point.
  The raise this frame was chosen for is gone, and was never about the frame: it was
  `doesNotRecognizeSelector:`, from the crate sending `confidence` to a
  `VNHumanBodyRecognizedPoint3D`, which has no such selector. The fixture keeps both lanes
  above, and now yields a 3-D pose of its own.

qr_code.jpg
  content:   AVANALYZE-PIXEL-DOOR
  produced:  generated for this repository with segno (QR version 2, error
             correction H, scale 12, border 4), then
             sips -s format jpeg -s formatOptions 90
  committed: 396x396, 18006 bytes
  sha256:    bd93cffe8b6bcb1434838906e3514cd85806e8a6ebf8482d5590a57bb46cfacf
  licence:   none needed -- generated here, not taken from anywhere
  why:       the barcode capability has no positive material in the other two
             fixtures, so a barcode entry point that silently returned nothing
             would pass every other test. Both doors must decode the payload
             above, not merely detect a shape.
  verified on this host: 1 barcode, symbology VNBarcodeSymbologyQR, payload
    AVANALYZE-PIXEL-DOOR, identical through the jpeg door and the pixel door.
```
