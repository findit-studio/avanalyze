# Test fixtures

Two committed images. Both are inputs to the real-inference lane in
`tests/audit_avanalyze_api.rs`, which runs against the real Apple Vision
framework and therefore only on an Apple host.

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

airport_keyframe.jpg
  the desktop's exact keyframe-extraction output for 01_airport.mp4 (AreaResampler downscale to
  288x512 + jpeg-encoder q85) at the first frame whose 3-D body-pose detector raises an
  Objective-C exception. 25930 bytes. Carries no face: it is the zero-face lane for the face
  fusion, and the no-abort regression fixture for every entry point.
```
