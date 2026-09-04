//! AUDIT: the public entry-point surface + the non-macOS stubs.
//!
//! Drives every entry point through the outside vocabulary in
//! `tests/common`, on whatever platform the suite runs.

mod common;

use avanalyze::{
  AnalyzeOptions, AnimalPoser, AppleVisionAnimalPoseOptions, AppleVisionBarcodeOptions,
  AppleVisionBodyPoserOptions, AppleVisionClassificationOptions, AppleVisionFaceLandmarkOptions,
  AppleVisionFaceOptions, AppleVisionHandPoseOptions, AppleVisionPersonMaskerOptions,
  AppleVisionTextOptions, BarcodeDetector, BodyPoser, FaceDetector, FaceLandmarker, HandPoser,
  PersonMasker, PixelFormat, PixelPlane, TextRecognizer, VisionAnalyzer,
};
use common::{
  AnimalPose, Barcode, Face, FaceLandmarks, HandPose, InstanceMask, Plain, Pose, Pose3,
  SegmentationMask, Text,
};

// -- Non-macOS stubs: every entry point refuses, and says why --

#[cfg(not(target_vendor = "apple"))]
mod non_macos {
  use super::*;
  use avanalyze::AnalyzeErrorKind;

  #[test]
  fn analyzer_stub_returns_error() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
    let err = analyzer
      .analyze_keyframe::<Plain>(&[], &options)
      .expect_err("stub must Err");
    assert_eq!(err.kind(), AnalyzeErrorKind::Unsupported);
  }

  #[test]
  fn analyzer_stub_error_mentions_macos() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
    let err = analyzer
      .analyze_keyframe::<Plain>(&[0xFF, 0xD8], &options)
      .expect_err("stub must Err");
    assert!(err.message().contains("macOS"));
  }

  #[test]
  fn analyzer_stub_ignores_data_size() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
    let e1 = analyzer
      .analyze_keyframe::<Plain>(&[], &options)
      .expect_err("empty");
    let e2 = analyzer
      .analyze_keyframe::<Plain>(&vec![0u8; 1024], &options)
      .expect_err("large");
    assert_eq!(e1.kind(), e2.kind());
  }

  #[test]
  fn analyzer_error_has_kind_and_message() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
    let err = analyzer
      .analyze_keyframe::<Plain>(&[], &options)
      .expect_err("stub");
    let _ = format!("{:?}", err.kind());
    assert!(!err.message().is_empty());
    // The error is a real `std::error::Error`, not a record the caller
    // has to reassemble a string from.
    let as_error: &dyn std::error::Error = &err;
    assert!(as_error.to_string().contains("macOS"));
  }

  /// Every entry point's stub, not just the analyzer's: nine surfaces
  /// are nine places the refusal could drift.
  #[test]
  fn every_entry_point_stub_refuses() {
    fn check(err: avanalyze::AnalyzeError) {
      assert_eq!(err.kind(), AnalyzeErrorKind::Unsupported);
      assert!(err.message().contains("macOS"));
    }

    let text = AppleVisionTextOptions::new();
    check(
      TextRecognizer::new(&text)
        .expect("TextRecognizer::new builds its Vision requests on this host")
        .recognize::<Text>(&[], &text)
        .expect_err("stub must Err"),
    );

    let barcode = AppleVisionBarcodeOptions::new();
    check(
      BarcodeDetector::new(&barcode)
        .expect("BarcodeDetector::new builds its Vision requests on this host")
        .detect::<Barcode>(&[], &barcode)
        .expect_err("stub must Err"),
    );

    let face = AppleVisionFaceOptions::new();
    check(
      FaceDetector::new(&face)
        .expect("FaceDetector::new builds its Vision requests on this host")
        .detect::<Face>(&[], &face)
        .expect_err("stub must Err"),
    );

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    check(
      FaceLandmarker::new(&landmarks)
        .expect("FaceLandmarker::new builds its Vision requests on this host")
        .detect::<FaceLandmarks>(&[], &landmarks)
        .expect_err("stub must Err"),
    );

    let poser = AppleVisionBodyPoserOptions::new();
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_2d::<Pose>(&[], &poser)
        .expect_err("stub must Err"),
    );
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_3d::<Pose3>(&[], &poser)
        .expect_err("stub must Err"),
    );

    let hand = AppleVisionHandPoseOptions::new();
    check(
      HandPoser::new(&hand)
        .expect("HandPoser::new builds its Vision requests on this host")
        .detect::<HandPose>(&[], &hand)
        .expect_err("stub must Err"),
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    check(
      AnimalPoser::new(&animal)
        .expect("AnimalPoser::new builds its Vision requests on this host")
        .detect::<AnimalPose>(&[], &animal)
        .expect_err("stub must Err"),
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .instance_masks::<InstanceMask>(&[], &masker)
        .expect_err("stub must Err"),
    );
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .segmentation_masks::<SegmentationMask>(&[], &masker)
        .expect_err("stub must Err"),
    );
  }

  /// The pixel door refuses off Apple exactly as the JPEG door does —
  /// same kind, same message, all eleven methods. A plane is pure
  /// arithmetic and builds fine here; what does not exist off Apple is
  /// Vision, so the refusal has to come from the same place either way.
  #[test]
  fn every_pixel_door_stub_refuses() {
    fn check(err: avanalyze::AnalyzeError) {
      assert_eq!(err.kind(), AnalyzeErrorKind::Unsupported);
      assert!(err.message().contains("macOS"));
    }

    let bytes = [0u8; 4 * 4 * 3];
    let plane = PixelPlane::packed(&bytes, 4, 4, PixelFormat::Rgb8)
      .expect("a plane is arithmetic, and is valid on every target");

    let options = AnalyzeOptions::new();
    check(
      VisionAnalyzer::new(&options)
        .expect("VisionAnalyzer::new builds its Vision requests on this host")
        .analyze_keyframe_pixels::<Plain>(&plane, &options)
        .expect_err("stub must Err"),
    );

    let text = AppleVisionTextOptions::new();
    check(
      TextRecognizer::new(&text)
        .expect("TextRecognizer::new builds its Vision requests on this host")
        .recognize_pixels::<Text>(&plane, &text)
        .expect_err("stub must Err"),
    );

    let barcode = AppleVisionBarcodeOptions::new();
    check(
      BarcodeDetector::new(&barcode)
        .expect("BarcodeDetector::new builds its Vision requests on this host")
        .detect_pixels::<Barcode>(&plane, &barcode)
        .expect_err("stub must Err"),
    );

    let face = AppleVisionFaceOptions::new();
    check(
      FaceDetector::new(&face)
        .expect("FaceDetector::new builds its Vision requests on this host")
        .detect_pixels::<Face>(&plane, &face)
        .expect_err("stub must Err"),
    );

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    check(
      FaceLandmarker::new(&landmarks)
        .expect("FaceLandmarker::new builds its Vision requests on this host")
        .detect_pixels::<FaceLandmarks>(&plane, &landmarks)
        .expect_err("stub must Err"),
    );

    let poser = AppleVisionBodyPoserOptions::new();
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_2d_pixels::<Pose>(&plane, &poser)
        .expect_err("stub must Err"),
    );
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_3d_pixels::<Pose3>(&plane, &poser)
        .expect_err("stub must Err"),
    );

    let hand = AppleVisionHandPoseOptions::new();
    check(
      HandPoser::new(&hand)
        .expect("HandPoser::new builds its Vision requests on this host")
        .detect_pixels::<HandPose>(&plane, &hand)
        .expect_err("stub must Err"),
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    check(
      AnimalPoser::new(&animal)
        .expect("AnimalPoser::new builds its Vision requests on this host")
        .detect_pixels::<AnimalPose>(&plane, &animal)
        .expect_err("stub must Err"),
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .instance_masks_pixels::<InstanceMask>(&plane, &masker)
        .expect_err("stub must Err"),
    );
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .segmentation_masks_pixels::<SegmentationMask>(&plane, &masker)
        .expect_err("stub must Err"),
    );
  }
}

// -- Debug, construction, and options plumbing on every platform --

#[test]
fn every_entry_point_is_debug() {
  assert!(
    format!(
      "{:?}",
      VisionAnalyzer::new(&AnalyzeOptions::new())
        .expect("VisionAnalyzer::new builds its Vision requests on this host")
    )
    .contains("VisionAnalyzer")
  );
  assert!(
    format!(
      "{:?}",
      TextRecognizer::new(&AppleVisionTextOptions::new())
        .expect("TextRecognizer::new builds its Vision requests on this host")
    )
    .contains("TextRecognizer")
  );
  assert!(
    format!(
      "{:?}",
      BarcodeDetector::new(&AppleVisionBarcodeOptions::new())
        .expect("BarcodeDetector::new builds its Vision requests on this host")
    )
    .contains("BarcodeDetector")
  );
  assert!(
    format!(
      "{:?}",
      FaceDetector::new(&AppleVisionFaceOptions::new())
        .expect("FaceDetector::new builds its Vision requests on this host")
    )
    .contains("FaceDetector")
  );
  assert!(
    format!(
      "{:?}",
      FaceLandmarker::new(&AppleVisionFaceLandmarkOptions::new())
        .expect("FaceLandmarker::new builds its Vision requests on this host")
    )
    .contains("FaceLandmarker")
  );
  assert!(
    format!(
      "{:?}",
      BodyPoser::new(&AppleVisionBodyPoserOptions::new())
        .expect("BodyPoser::new builds its Vision requests on this host")
    )
    .contains("BodyPoser")
  );
  assert!(
    format!(
      "{:?}",
      HandPoser::new(&AppleVisionHandPoseOptions::new())
        .expect("HandPoser::new builds its Vision requests on this host")
    )
    .contains("HandPoser")
  );
  assert!(
    format!(
      "{:?}",
      AnimalPoser::new(&AppleVisionAnimalPoseOptions::new())
        .expect("AnimalPoser::new builds its Vision requests on this host")
    )
    .contains("AnimalPoser")
  );
  assert!(
    format!(
      "{:?}",
      PersonMasker::new(&AppleVisionPersonMaskerOptions::new())
        .expect("PersonMasker::new builds its Vision requests on this host")
    )
    .contains("PersonMasker")
  );
}

#[test]
fn analyze_options_is_send() {
  fn assert_send<T: Send>() {}
  assert_send::<AnalyzeOptions>();
}

#[test]
fn multiple_analyzer_constructions() {
  let options = AnalyzeOptions::new();
  for _ in 0..10 {
    let _ = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
  }
}

#[test]
fn analyzer_with_custom_options() {
  let mut opts = AnalyzeOptions::new().with_workers(4);
  opts.classifications_mut().set_min_confidence(0.5);
  opts.classifications_mut().set_max_results(5);
  let analyzer = VisionAnalyzer::new(&opts)
    .expect("VisionAnalyzer::new builds its Vision requests on this host");
  let _ = format!("{analyzer:?}");
}

/// The one knob Apple bakes into a request object still reaches its
/// entry point at construction, clamped or not.
#[test]
fn hand_poser_takes_its_baked_knob() {
  for count in [1usize, 2, 6, 64] {
    let opts = AppleVisionHandPoseOptions::new().with_maximum_hand_count(count);
    let _ = HandPoser::new(&opts).expect("HandPoser::new builds its Vision requests on this host");
  }
}

#[test]
fn default_feature_compiles() {
  // This test file compiles with default features (no serde, no tracing)
  let _ = AnalyzeOptions::new();
}

// -- Defaults are public constants, single-sourced with `new()` --

#[test]
fn default_constants_match_constructed_defaults() {
  let o = AppleVisionClassificationOptions::new();
  assert_eq!(
    o.min_confidence(),
    AppleVisionClassificationOptions::DEFAULT_MIN_CONFIDENCE
  );
  assert_eq!(
    o.max_results(),
    AppleVisionClassificationOptions::DEFAULT_MAX_RESULTS
  );
  assert_eq!(
    AnalyzeOptions::new().num_workers(),
    AnalyzeOptions::DEFAULT_NUM_WORKERS
  );
}

// -- Real inference: every entry point against a real keyframe --

/// Regression for the Vision-framework foreign-exception process abort,
/// now over all nine entry points.
///
/// On certain real keyframes a Vision detector raises an Objective-C
/// `NSException` that unwinds across the objc2/Vision FFI boundary.
/// Rust's `catch_unwind` (used in the crate for a separate Rust-panic
/// quirk in the 3-D body-pose path) cannot catch a foreign exception —
/// one escaping it aborts the entire process with `fatal runtime
/// error: Rust cannot catch foreign exceptions`. Every Vision FFI call
/// is guarded with `objc2::exception::catch`, degrading a raising
/// detector to an empty result.
///
/// The committed fixture is the desktop's exact keyframe-extraction
/// output for `01_airport.mp4` (the `AreaResampler` downscale to
/// 288x512 + `jpeg-encoder` q85) at the first frame whose 3D body-pose
/// detector raises. In a RELEASE / `debug-assertions = false` build —
/// where objc2 compiles out its msg_send verification and the
/// encoding-mismatched `VNHumanBodyRecognizedPoint3D` selector
/// dispatches for real — running this fixture through the *unfixed*
/// code aborts the process; the fix makes it return `Ok`. (Under the
/// default `cargo test` debug profile the same path raises a Rust panic
/// that the existing `catch_unwind` absorbs, so this asserts the
/// end-to-end no-abort / no-panic contract on every profile.)
#[cfg(target_vendor = "apple")]
mod real_inference {
  use super::*;

  use crate::common::Bbox;

  /// The desktop resample emits a 288x512 frame for this clip.
  const JPEG: &[u8] = include_bytes!("fixtures/airport_keyframe.jpg");

  /// The multi-face fixture. Provenance, recorded here and in
  /// `tests/fixtures/README.md`:
  ///
  /// ```text
  /// apollo11_crew.jpg
  ///   source:   https://commons.wikimedia.org/wiki/File:Apollo_11_Crew.jpg
  ///             (original: https://upload.wikimedia.org/wikipedia/commons/3/3d/Apollo_11_Crew.jpg)
  ///   credit:   NASA — the Apollo 11 prime crew (Armstrong, Collins, Aldrin), 1969
  ///   licence:  Public domain (a work of the U.S. federal government)
  ///   fetched:  2026-09-01
  ///   original: 4200x3300, 1628582 bytes
  ///   committed: 640x503, 83864 bytes, produced with
  ///              sips -Z 640 --setProperty formatOptions 70
  ///   sha256:   6e20f9e893b6103539601fae122594ca668b9bbbc77fafad22d0d1c79682e8ee
  ///   why:      three frontal, well-separated faces, so every face's own capture-quality
  ///             and landmark readings must come back to ITSELF; a mis-seated observation
  ///             lands keypoints in a neighbour's box.
  ///   verified on this host at the crate's default face gates (rectangles min_confidence 0.1,
  ///   capture min_capture_quality 0.1), Vision requests pinned to Revision3, over 30 runs:
  ///     3 faces every run; per face, paired by the observation identity the fusion seats on —
  ///       x 0.2092 -> confidence 0.8567, capture_quality 0.4387
  ///       x 0.4405 -> confidence 0.8758, capture_quality 0.3569
  ///       x 0.6731 -> confidence 0.8781, capture_quality 0.5088
  ///     all three with a complete five-point reduction, every point inside its own face's box,
  ///     and the emitted records BIT-IDENTICAL on all 30 runs — while the annotating passes
  ///     returned their observations in spine order only 15/30 (quality) and 15/30 (landmarks).
  ///     Order-independence of the output is the property; the varying return order is the test.
  ///   (the same image detects 3 faces at every scale tried from 384 px to the 4200 px original)
  /// ```
  const CREW: &[u8] = include_bytes!("fixtures/apollo11_crew.jpg");

  /// Two normalized boxes share no area.
  fn boxes_are_disjoint(a: &Bbox, b: &Bbox) -> bool {
    a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y
  }

  /// A normalized point lies within a box, edges included.
  fn point_in_box(bbox: &Bbox, x: f32, y: f32) -> bool {
    x >= bbox.x && x <= bbox.x + bbox.width && y >= bbox.y && y <= bbox.y + bbox.height
  }

  #[test]
  fn analyze_keyframe_does_not_abort_on_real_airport_keyframe() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
    let analysis = analyzer
      .analyze_keyframe::<Plain>(JPEG, &options)
      .expect("analyze_keyframe must return Ok (a partial analysis), never abort the process");

    // Both frame-wide slots are always written — with a real reading or
    // with the engine's sentinel — so a successful call is never a
    // degenerate shell.
    assert!(
      analysis.horizon().is_some(),
      "the horizon slot carries at least the no-detection sentinel"
    );
    assert!(
      analysis.aesthetics().is_some(),
      "the aesthetics slot carries at least the no-detection sentinel"
    );
  }

  #[test]
  fn text_recognizer_runs_on_a_real_keyframe() {
    let options = AppleVisionTextOptions::new();
    let recognizer = TextRecognizer::new(&options)
      .expect("TextRecognizer::new builds its Vision requests on this host");
    let runs = recognizer
      .recognize::<Text>(JPEG, &options)
      .expect("recognize must return Ok");
    for run in &runs {
      assert!(
        run.rank < options.max_candidates_per_observation(),
        "rank stays inside the configured candidate list: {}",
        run.rank
      );
      assert!(!run.text.is_empty(), "an emitted run carries its reading");
    }
  }

  #[test]
  fn barcode_detector_runs_on_a_real_keyframe() {
    let options = AppleVisionBarcodeOptions::new();
    let detector = BarcodeDetector::new(&options)
      .expect("BarcodeDetector::new builds its Vision requests on this host");
    let _ = detector
      .detect::<Barcode>(JPEG, &options)
      .expect("detect must return Ok");
  }

  /// The zero-face lane. The airport keyframe carries no face at all,
  /// so the detection spine is empty and the two annotating passes are
  /// never even performed — which is also what stops them running their
  /// own face detection and returning a face the spine never saw.
  #[test]
  fn face_detector_runs_on_a_real_keyframe() {
    let options = AppleVisionFaceOptions::new();
    let detector = FaceDetector::new(&options)
      .expect("FaceDetector::new builds its Vision requests on this host");
    let faces = detector
      .detect::<Face>(JPEG, &options)
      .expect("detect must return Ok");
    assert!(
      faces.is_empty(),
      "the airport keyframe carries no face, so the fusion emits none: got {} face(s)",
      faces.len()
    );
    for face in &faces {
      if let Some(k) = face.keypoints {
        for (x, y) in k.points() {
          assert!(
            (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y),
            "keypoints are normalized image coordinates: ({x}, {y})"
          );
        }
      }
    }
  }

  /// Every face wears its OWN readings, on a frame that actually has
  /// more than one face in it.
  ///
  /// The fusion feeds the rectangles pass's observations to the other
  /// two through `VNFaceObservationAccepting` and seats what comes back
  /// by each observation's `VNObservation.uuid`, because Vision does
  /// **not** return them in the order it was given. Measured on this
  /// host over 30 runs of this very fixture: the returned uuid set
  /// matched the spine 30/30, while the returned ORDER matched it in
  /// only 15/30 of the capture-quality runs and 15/30 of the landmarks
  /// runs, with every one of the six permutations of three elements
  /// observed at least once. (An independent 30-run measurement on the
  /// same host got 14/30 and 12/30 — the point is that it varies, not
  /// that it has a rate.) A positional read would therefore have
  /// mis-seated about half of these runs.
  ///
  /// This fixture's faces are pairwise DISJOINT, which turns that
  /// failure into something a test can see: the keypoints are projected
  /// out of the box of the observation the resolution seated here, and
  /// compared against the box of the SPINE face at this seat, so a
  /// keypoint landing inside a neighbour's box is a mis-seated
  /// observation. Running this test repeatedly exercises many different
  /// return orders against one expected answer.
  ///
  /// This host yields exactly 3 faces. The assertion is `>= 2` so the
  /// test still MEANS something — a genuinely multi-face attribution —
  /// if a future Vision build finds a different count; the observed
  /// count and readings are recorded on `CREW`. Nothing here depends on
  /// the order the faces come back in, which is not stable across
  /// encodings — nor, for the annotating passes, across runs.
  #[test]
  fn face_detector_attaches_each_face_its_own_readings_on_a_multi_face_keyframe() {
    let options = AppleVisionFaceOptions::new();
    let detector = FaceDetector::new(&options)
      .expect("FaceDetector::new builds its Vision requests on this host");
    let faces = detector
      .detect::<Face>(CREW, &options)
      .expect("detect must return Ok");

    assert!(
      faces.len() >= 2,
      "multi-face attribution is what is under test: got {} face(s)",
      faces.len()
    );

    // The premise every per-face assertion below rests on.
    for (i, a) in faces.iter().enumerate() {
      for b in faces.iter().skip(i + 1) {
        assert!(
          boxes_are_disjoint(&a.bbox, &b.bbox),
          "the fixture's faces must be pairwise disjoint for this test to mean anything: \
           {:?} vs {:?}",
          a.bbox,
          b.bbox
        );
      }
    }

    // Every face carries a DISTINCT capture-quality reading. The three
    // crew members' captures genuinely differ on this fixture, so two
    // faces reading the same value would mean one observation's reading
    // had reached more than one face — the failure the one-to-one uuid
    // resolution has to be incapable of.
    for (i, a) in faces.iter().enumerate() {
      for b in faces.iter().skip(i + 1) {
        assert_ne!(
          a.capture_quality, b.capture_quality,
          "each face wears the capture quality Vision computed for IT: {:?} repeated across two \
           faces",
          a.capture_quality
        );
      }
    }

    for (i, face) in faces.iter().enumerate() {
      // Guaranteed by the default 0.1 gate: an unmeasured face compares
      // as 0.0 and would have been dropped.
      let quality = face
        .capture_quality
        .expect("an emitted face carries a real capture-quality reading under the default gate");
      assert!(
        quality >= 0.1,
        "an emitted face cleared the default capture-quality gate: {quality}"
      );

      // The three face requests are pinned to Revision3, so a complete
      // reduction is a stable property of this fixture; a failure here
      // should be loud rather than silently tolerated.
      let keypoints = face
        .keypoints
        .expect("every face on this fixture reduces to a complete five-point set");

      for (x, y) in keypoints.points() {
        assert!(
          point_in_box(&face.bbox, x, y),
          "every keypoint lies inside its OWN face's box: ({x}, {y}) not in {:?}",
          face.bbox
        );
        for (j, other) in faces.iter().enumerate() {
          if i == j {
            continue;
          }
          assert!(
            !point_in_box(&other.bbox, x, y),
            "a keypoint inside a neighbour's box is a mis-assigned observation: \
             ({x}, {y}) in {:?}",
            other.bbox
          );
        }
      }

      // Top-left origin, so "above" is a SMALLER y.
      assert!(
        keypoints.left_eye().1 < keypoints.mouth_left().1,
        "the left eye centre sits above the left mouth corner: {keypoints:?}"
      );
      assert!(
        keypoints.right_eye().1 < keypoints.mouth_right().1,
        "the right eye centre sits above the right mouth corner: {keypoints:?}"
      );
      let eye_mid_y = (keypoints.left_eye().1 + keypoints.right_eye().1) / 2.0;
      let mouth_mid_y = (keypoints.mouth_left().1 + keypoints.mouth_right().1) / 2.0;
      let nose_y = keypoints.nose_tip().1;
      assert!(
        nose_y > eye_mid_y && nose_y < mouth_mid_y,
        "the nose tip sits between the eye midpoint and the mouth midpoint: \
         nose {nose_y}, eyes {eye_mid_y}, mouth {mouth_mid_y}"
      );
    }
  }

  /// **One detector, five calls, two frames alternating.** The whole
  /// per-call request-state invariant, on the path a caller actually
  /// takes.
  ///
  /// A `FaceDetector` owns three RETAINED `VNRequest`s, so everything
  /// a call leaves on them — the rectangles pass's `results`, the two
  /// annotating passes' `results`, and the `inputFaceObservations` set
  /// on them for the call — outlives that call. A detector whose
  /// per-call state leaked would show it here in the loudest possible
  /// way: the airport keyframe carries no face at all, so any face
  /// coming back from it is one of the crew's, arriving from the frame
  /// before.
  ///
  /// The alternation is what makes the leak visible. `crew, airport,
  /// crew, airport, crew` means every airport call is preceded by a
  /// successful three-face call, and every crew call after the first is
  /// preceded by a zero-face one — so a stale spine, a stale annotation
  /// read, or an `inputFaceObservations` array left holding the
  /// previous frame's observations each has a call in which to show
  /// up. Three faces then zero, three times over, is the property.
  ///
  /// The crew records must also be BIT-IDENTICAL across their three
  /// calls: same boxes, same confidences, same capture qualities, same
  /// five-point reductions. A detector that carried anything across
  /// calls would have to reproduce it exactly to pass this, and there
  /// are two intervening zero-face frames in which to fail to.
  ///
  /// This exercises only the ordinary, non-raising path. Forcing an
  /// Objective-C exception at the five points where the invariant does
  /// its work is not possible without a mock Vision framework — see
  /// `run_requests_extracts_only_after_a_completed_perform` in the
  /// crate's own test module.
  #[test]
  fn one_face_detector_reused_across_frames_carries_no_state_between_them() {
    let options = AppleVisionFaceOptions::new();
    let detector = FaceDetector::new(&options)
      .expect("FaceDetector::new builds its Vision requests on this host");

    let mut crew_runs: Vec<Vec<Face>> = Vec::new();
    for (call, frame) in [CREW, JPEG, CREW, JPEG, CREW].into_iter().enumerate() {
      let faces = detector
        .detect::<Face>(frame, &options)
        .expect("detect must return Ok on every call");
      if call % 2 == 0 {
        assert_eq!(
          faces.len(),
          3,
          "call {call} is the crew frame and this host detects exactly 3 faces on it: got {}",
          faces.len()
        );
        crew_runs.push(faces);
      } else {
        assert!(
          faces.is_empty(),
          "call {call} is the airport frame, which carries NO face — {} face(s) here would be \
           the previous frame's, surviving on a retained request",
          faces.len()
        );
      }
    }

    let first = &crew_runs[0];
    for (run, faces) in crew_runs.iter().enumerate().skip(1) {
      assert_eq!(
        faces, first,
        "the crew frame yields the same records every time it is analysed, whatever ran between: \
         crew run {run} differs from crew run 0"
      );
    }
  }

  #[test]
  fn face_landmarker_runs_on_a_real_keyframe() {
    let options = AppleVisionFaceLandmarkOptions::new();
    let landmarker = FaceLandmarker::new(&options)
      .expect("FaceLandmarker::new builds its Vision requests on this host");
    let _ = landmarker
      .detect::<FaceLandmarks>(JPEG, &options)
      .expect("detect must return Ok");
  }

  /// The 3-D pass is the one that raises on this fixture, so it gets
  /// its own call: `Ok` here is the no-abort contract.
  #[test]
  fn body_poser_runs_both_dimensions_on_a_real_keyframe() {
    let options = AppleVisionBodyPoserOptions::new();
    let poser =
      BodyPoser::new(&options).expect("BodyPoser::new builds its Vision requests on this host");
    let _ = poser
      .detect_2d::<Pose>(JPEG, &options)
      .expect("detect_2d must return Ok");
    let _ = poser
      .detect_3d::<Pose3>(JPEG, &options)
      .expect("detect_3d must return Ok (never abort the process)");
  }

  #[test]
  fn hand_poser_runs_on_a_real_keyframe() {
    let options = AppleVisionHandPoseOptions::new();
    let poser =
      HandPoser::new(&options).expect("HandPoser::new builds its Vision requests on this host");
    let _ = poser
      .detect::<HandPose>(JPEG, &options)
      .expect("detect must return Ok");
  }

  #[test]
  fn animal_poser_runs_on_a_real_keyframe() {
    let options = AppleVisionAnimalPoseOptions::new();
    let poser =
      AnimalPoser::new(&options).expect("AnimalPoser::new builds its Vision requests on this host");
    let _ = poser
      .detect::<AnimalPose>(JPEG, &options)
      .expect("detect must return Ok");
  }

  #[test]
  fn person_masker_runs_both_kinds_on_a_real_keyframe() {
    let options = AppleVisionPersonMaskerOptions::new();
    let masker = PersonMasker::new(&options)
      .expect("PersonMasker::new builds its Vision requests on this host");
    let instances = masker
      .instance_masks::<InstanceMask>(JPEG, &options)
      .expect("instance_masks must return Ok");
    for mask in &instances {
      assert_eq!(
        mask.data.len(),
        mask.width as usize * mask.height as usize,
        "the payload is always one byte per pixel"
      );
    }
    let whole = masker
      .segmentation_masks::<SegmentationMask>(JPEG, &options)
      .expect("segmentation_masks must return Ok");
    for mask in &whole {
      assert_eq!(
        mask.data.len(),
        mask.width as usize * mask.height as usize,
        "the payload is always one byte per pixel"
      );
    }
  }

  /// The input-byte ceiling is enforced by every entry point, not just
  /// the analyzer: one oversized payload, nine structured refusals.
  #[test]
  fn every_entry_point_refuses_an_oversized_payload() {
    use avanalyze::AnalyzeErrorKind;

    // One byte over the documented 64 MiB ceiling.
    let huge = vec![0u8; 64 * 1024 * 1024 + 1];
    fn check(err: avanalyze::AnalyzeError) {
      assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
      assert!(err.message().contains("MAX_INPUT_IMAGE_BYTES"));
    }

    let options = AnalyzeOptions::new();
    check(
      VisionAnalyzer::new(&options)
        .expect("VisionAnalyzer::new builds its Vision requests on this host")
        .analyze_keyframe::<Plain>(&huge, &options)
        .expect_err("oversized input must be refused"),
    );

    let text = AppleVisionTextOptions::new();
    check(
      TextRecognizer::new(&text)
        .expect("TextRecognizer::new builds its Vision requests on this host")
        .recognize::<Text>(&huge, &text)
        .expect_err("oversized input must be refused"),
    );

    let barcode = AppleVisionBarcodeOptions::new();
    check(
      BarcodeDetector::new(&barcode)
        .expect("BarcodeDetector::new builds its Vision requests on this host")
        .detect::<Barcode>(&huge, &barcode)
        .expect_err("oversized input must be refused"),
    );

    let face = AppleVisionFaceOptions::new();
    check(
      FaceDetector::new(&face)
        .expect("FaceDetector::new builds its Vision requests on this host")
        .detect::<Face>(&huge, &face)
        .expect_err("oversized input must be refused"),
    );

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    check(
      FaceLandmarker::new(&landmarks)
        .expect("FaceLandmarker::new builds its Vision requests on this host")
        .detect::<FaceLandmarks>(&huge, &landmarks)
        .expect_err("oversized input must be refused"),
    );

    let poser = AppleVisionBodyPoserOptions::new();
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_2d::<Pose>(&huge, &poser)
        .expect_err("oversized input must be refused"),
    );
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_3d::<Pose3>(&huge, &poser)
        .expect_err("oversized input must be refused"),
    );

    let hand = AppleVisionHandPoseOptions::new();
    check(
      HandPoser::new(&hand)
        .expect("HandPoser::new builds its Vision requests on this host")
        .detect::<HandPose>(&huge, &hand)
        .expect_err("oversized input must be refused"),
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    check(
      AnimalPoser::new(&animal)
        .expect("AnimalPoser::new builds its Vision requests on this host")
        .detect::<AnimalPose>(&huge, &animal)
        .expect_err("oversized input must be refused"),
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .instance_masks::<InstanceMask>(&huge, &masker)
        .expect_err("oversized input must be refused"),
    );
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .segmentation_masks::<SegmentationMask>(&huge, &masker)
        .expect_err("oversized input must be refused"),
    );
  }

  /// The decoded-dimension ceiling (issue #2) is enforced by every entry
  /// point too, through the same shared door as the compressed-byte cap
  /// above: a SOF marker declaring dimensions past
  /// `MAX_DECODED_IMAGE_BYTES` is refused before Vision ever sees the
  /// frame.
  #[test]
  fn every_entry_point_refuses_over_cap_decoded_dimensions() {
    use avanalyze::AnalyzeErrorKind;

    // SOI + one SOF0 segment declaring 65535 × 65535, Nf = 1. No
    // entropy-coded data follows — the preflight returns as soon as it
    // reads the SOF, so none is needed.
    #[rustfmt::skip]
    let huge_dims: &[u8] = &[
      0xFF, 0xD8,             // SOI
      0xFF, 0xC0, 0x00, 0x0B, // SOF0, length 11
      0x08,                   // precision
      0xFF, 0xFF,             // height = 65535
      0xFF, 0xFF,             // width  = 65535
      0x01,                   // Nf = 1
      0x01, 0x11, 0x00,       // one component: id, sampling, quant table
    ];

    fn check(err: avanalyze::AnalyzeError) {
      assert_eq!(err.kind(), AnalyzeErrorKind::RequestFailed);
      assert!(err.message().contains("MAX_DECODED_IMAGE_BYTES"));
    }

    let options = AnalyzeOptions::new();
    check(
      VisionAnalyzer::new(&options)
        .expect("VisionAnalyzer::new builds its Vision requests on this host")
        .analyze_keyframe::<Plain>(huge_dims, &options)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let text = AppleVisionTextOptions::new();
    check(
      TextRecognizer::new(&text)
        .expect("TextRecognizer::new builds its Vision requests on this host")
        .recognize::<Text>(huge_dims, &text)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let barcode = AppleVisionBarcodeOptions::new();
    check(
      BarcodeDetector::new(&barcode)
        .expect("BarcodeDetector::new builds its Vision requests on this host")
        .detect::<Barcode>(huge_dims, &barcode)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let face = AppleVisionFaceOptions::new();
    check(
      FaceDetector::new(&face)
        .expect("FaceDetector::new builds its Vision requests on this host")
        .detect::<Face>(huge_dims, &face)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    check(
      FaceLandmarker::new(&landmarks)
        .expect("FaceLandmarker::new builds its Vision requests on this host")
        .detect::<FaceLandmarks>(huge_dims, &landmarks)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let poser = AppleVisionBodyPoserOptions::new();
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_2d::<Pose>(huge_dims, &poser)
        .expect_err("over-cap decoded dimensions must be refused"),
    );
    check(
      BodyPoser::new(&poser)
        .expect("BodyPoser::new builds its Vision requests on this host")
        .detect_3d::<Pose3>(huge_dims, &poser)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let hand = AppleVisionHandPoseOptions::new();
    check(
      HandPoser::new(&hand)
        .expect("HandPoser::new builds its Vision requests on this host")
        .detect::<HandPose>(huge_dims, &hand)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    check(
      AnimalPoser::new(&animal)
        .expect("AnimalPoser::new builds its Vision requests on this host")
        .detect::<AnimalPose>(huge_dims, &animal)
        .expect_err("over-cap decoded dimensions must be refused"),
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .instance_masks::<InstanceMask>(huge_dims, &masker)
        .expect_err("over-cap decoded dimensions must be refused"),
    );
    check(
      PersonMasker::new(&masker)
        .expect("PersonMasker::new builds its Vision requests on this host")
        .segmentation_masks::<SegmentationMask>(huge_dims, &masker)
        .expect_err("over-cap decoded dimensions must be refused"),
    );
  }

  /// A QR code carrying a known payload, generated for this suite.
  /// Provenance is recorded in `tests/fixtures/README.md`.
  const QR: &[u8] = include_bytes!("fixtures/qr_code.jpg");

  /// Decode a fixture through ImageIO — the very decoder Vision reaches
  /// for behind the JPEG door — and hand back a tight packed-RGB plane
  /// of it, so both doors are compared on one decode rather than two.
  fn rgb_plane_bytes(jpeg: &'static [u8]) -> (u32, u32, Vec<u8>) {
    use core::ffi::c_void;

    use objc2_core_foundation::{CGPoint, CGRect, CGSize};
    use objc2_core_graphics::{
      CGBitmapContextCreate, CGColorRenderingIntent, CGColorSpace, CGContext, CGDataProvider,
      CGImage, CGImageAlphaInfo,
    };

    // SAFETY: `jpeg` is `'static`, so the provider's borrow cannot
    // dangle, and no release callback is needed because nothing is
    // owned. `decode` is null, the documented alternative to a pointer.
    let (width, height, rgba) = unsafe {
      let provider = CGDataProvider::with_data(
        core::ptr::null_mut(),
        jpeg.as_ptr().cast::<c_void>(),
        jpeg.len(),
        None,
      )
      .expect("a data provider over a static fixture");
      let image = CGImage::with_jpeg_data_provider(
        Some(&provider),
        core::ptr::null(),
        true,
        CGColorRenderingIntent::RenderingIntentDefault,
      )
      .expect("the fixture is a decodable JPEG");
      let width = CGImage::width(Some(&image));
      let height = CGImage::height(Some(&image));
      let stride = width * 4;
      let mut rgba = vec![0u8; stride * height];
      let colour_space = CGColorSpace::new_device_rgb().expect("device RGB");
      let context = CGBitmapContextCreate(
        rgba.as_mut_ptr().cast::<c_void>(),
        width,
        height,
        8,
        stride,
        Some(&colour_space),
        CGImageAlphaInfo::NoneSkipLast.0,
      )
      .expect("an RGBA8 bitmap context");
      CGContext::draw_image(
        Some(&context),
        CGRect {
          origin: CGPoint { x: 0.0, y: 0.0 },
          size: CGSize {
            width: width as f64,
            height: height as f64,
          },
        },
        Some(&image),
      );
      drop(context);
      (width, height, rgba)
    };

    let mut packed = Vec::with_capacity(width * height * 3);
    for pixel in rgba.as_chunks::<4>().0 {
      packed.extend_from_slice(&pixel[..3]);
    }
    (
      u32::try_from(width).expect("fixture width fits u32"),
      u32::try_from(height).expect("fixture height fits u32"),
      packed,
    )
  }

  /// Door parity as a roster, through the OUTSIDE vocabulary: every one
  /// of the eleven analysis methods has a pixel twin, every twin runs
  /// against a plane decoded from a real photograph, and every one that
  /// the photograph carries material for comes back with something.
  ///
  /// Presence is what this asserts and why it exists. `run_requests`
  /// reports a caught Objective-C exception as `Ok` carrying the
  /// caller's empty fallback, so a door that silently found nothing —
  /// forever, for every input — would satisfy "returned `Ok`" on a
  /// synthetic image with nothing in it. A count that must be non-zero
  /// would not.
  ///
  /// Three capabilities have no material here and are asserted for
  /// agreement only, honestly: `apollo11_crew.jpg` carries no barcode
  /// (covered by the QR fixture below instead), no animal, and no text
  /// Vision reads at 640 px.
  ///
  /// 3-D body poses were a fourth, back when `detect_3d` could not
  /// produce output through either door. It can now, and the fixture
  /// carries a pose, so it joins the positive set. What that pose's
  /// joints actually contain is asserted against Objective-C's own
  /// reading in `src/tests/body_pose.rs` — a count would not have
  /// caught coordinates that were finite and wrong.
  #[test]
  fn every_entry_point_runs_its_pixel_door_on_a_real_photograph() {
    let (width, height, packed) = rgb_plane_bytes(CREW);
    let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8)
      .expect("a tight RGB plane of the crew fixture");

    /// Both doors agreed, and — where the fixture carries the
    /// capability — the pixel door found something.
    fn agree(capability: &str, jpeg: usize, pixels: usize, carried: bool) {
      assert_eq!(jpeg, pixels, "the two doors must agree on {capability}");
      if carried {
        assert!(
          pixels > 0,
          "the crew fixture carries {capability}, so a pixel door finding none is broken"
        );
      }
    }

    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options)
      .expect("VisionAnalyzer::new builds its Vision requests on this host");
    let jpeg = analyzer
      .analyze_keyframe::<Plain>(CREW, &options)
      .expect("the analyzer's jpeg door must return Ok");
    let pixels = analyzer
      .analyze_keyframe_pixels::<Plain>(&plane, &options)
      .expect("the analyzer's pixel door must return Ok");
    agree(
      "classifications",
      jpeg.classifications().len(),
      pixels.classifications().len(),
      true,
    );
    agree(
      "human subjects",
      jpeg.human_subjects().len(),
      pixels.human_subjects().len(),
      true,
    );
    assert!(pixels.horizon().is_some());
    assert!(pixels.aesthetics().is_some());

    let text = AppleVisionTextOptions::new();
    let recognizer = TextRecognizer::new(&text)
      .expect("TextRecognizer::new builds its Vision requests on this host");
    agree(
      "text",
      recognizer
        .recognize::<Text>(CREW, &text)
        .expect("jpeg")
        .len(),
      recognizer
        .recognize_pixels::<Text>(&plane, &text)
        .expect("pixels")
        .len(),
      false,
    );

    let barcode = AppleVisionBarcodeOptions::new();
    let detector = BarcodeDetector::new(&barcode)
      .expect("BarcodeDetector::new builds its Vision requests on this host");
    agree(
      "barcodes",
      detector
        .detect::<Barcode>(CREW, &barcode)
        .expect("jpeg")
        .len(),
      detector
        .detect_pixels::<Barcode>(&plane, &barcode)
        .expect("pixels")
        .len(),
      false,
    );

    let face = AppleVisionFaceOptions::new();
    let faces =
      FaceDetector::new(&face).expect("FaceDetector::new builds its Vision requests on this host");
    let through_pixels = faces
      .detect_pixels::<Face>(&plane, &face)
      .expect("the face fusion's pixel door must return Ok");
    agree(
      "faces",
      faces.detect::<Face>(CREW, &face).expect("jpeg").len(),
      through_pixels.len(),
      true,
    );
    // The fusion's own product, not merely a count: every face carried
    // through the pixel door must still arrive with its OWN annotations
    // seated on it.
    for found in &through_pixels {
      let keypoints = found
        .keypoints
        .expect("each crew face reduces to five keypoints through the pixel door too");
      for (x, y) in keypoints.points() {
        assert!(
          point_in_box(&found.bbox, x, y),
          "a keypoint seated on the wrong face: ({x}, {y}) outside {:?}",
          found.bbox
        );
      }
      assert!(
        found.capture_quality.is_some(),
        "the capture-quality pass must reach the pixel door's faces too"
      );
    }

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    let landmarker = FaceLandmarker::new(&landmarks)
      .expect("FaceLandmarker::new builds its Vision requests on this host");
    agree(
      "landmark sets",
      landmarker
        .detect::<FaceLandmarks>(CREW, &landmarks)
        .expect("jpeg")
        .len(),
      landmarker
        .detect_pixels::<FaceLandmarks>(&plane, &landmarks)
        .expect("pixels")
        .len(),
      true,
    );

    let poser = AppleVisionBodyPoserOptions::new();
    let bodies =
      BodyPoser::new(&poser).expect("BodyPoser::new builds its Vision requests on this host");
    agree(
      "2-D body poses",
      bodies.detect_2d::<Pose>(CREW, &poser).expect("jpeg").len(),
      bodies
        .detect_2d_pixels::<Pose>(&plane, &poser)
        .expect("pixels")
        .len(),
      true,
    );
    let jpeg_3d = bodies.detect_3d::<Pose3>(CREW, &poser).expect("jpeg");
    let pixel_3d = bodies
      .detect_3d_pixels::<Pose3>(&plane, &poser)
      .expect("pixels");
    agree("3-D body poses", jpeg_3d.len(), pixel_3d.len(), true);
    // A 3-D pose whose joint list is empty is dropped before it reaches
    // the vocabulary, so a non-zero pose count already proves joints
    // were read. Assert the absent confidence here too, on the outside
    // vocabulary: it is the seat the caller sees.
    for pose in jpeg_3d.iter().chain(pixel_3d.iter()) {
      assert!(!pose.joints.is_empty());
      assert!(
        pose.joints.iter().all(|j| j.confidence.is_none()),
        "Apple's 3-D points carry no confidence, so every joint's seat is None"
      );
    }

    let hand = AppleVisionHandPoseOptions::new();
    let hands =
      HandPoser::new(&hand).expect("HandPoser::new builds its Vision requests on this host");
    agree(
      "hand poses",
      hands.detect::<HandPose>(CREW, &hand).expect("jpeg").len(),
      hands
        .detect_pixels::<HandPose>(&plane, &hand)
        .expect("pixels")
        .len(),
      true,
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    let animals =
      AnimalPoser::new(&animal).expect("AnimalPoser::new builds its Vision requests on this host");
    agree(
      "animal poses",
      animals
        .detect::<AnimalPose>(CREW, &animal)
        .expect("jpeg")
        .len(),
      animals
        .detect_pixels::<AnimalPose>(&plane, &animal)
        .expect("pixels")
        .len(),
      false,
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    let masks = PersonMasker::new(&masker)
      .expect("PersonMasker::new builds its Vision requests on this host");
    let instances = masks
      .instance_masks_pixels::<InstanceMask>(&plane, &masker)
      .expect("instance masks' pixel door must return Ok");
    agree(
      "instance masks",
      masks
        .instance_masks::<InstanceMask>(CREW, &masker)
        .expect("jpeg")
        .len(),
      instances.len(),
      true,
    );
    for mask in &instances {
      assert!(
        !mask.data.is_empty() && mask.width > 0 && mask.height > 0,
        "a mask through the pixel door must carry real pixels"
      );
    }
    agree(
      "segmentation masks",
      masks
        .segmentation_masks::<SegmentationMask>(CREW, &masker)
        .expect("jpeg")
        .len(),
      masks
        .segmentation_masks_pixels::<SegmentationMask>(&plane, &masker)
        .expect("pixels")
        .len(),
      true,
    );
  }

  /// The barcode door reads a payload, not merely `Ok`.
  ///
  /// The crew fixture carries no barcode, so without this the barcode
  /// door could return nothing forever and every other assertion would
  /// still pass. The QR fixture exists for exactly that gap.
  #[test]
  fn the_pixel_door_decodes_a_barcode_payload() {
    let (width, height, packed) = rgb_plane_bytes(QR);
    let plane = PixelPlane::packed(&packed, width, height, PixelFormat::Rgb8).expect("plane");

    let options = AppleVisionBarcodeOptions::new();
    let detector = BarcodeDetector::new(&options)
      .expect("BarcodeDetector::new builds its Vision requests on this host");
    let through_jpeg = detector
      .detect::<Barcode>(QR, &options)
      .expect("the jpeg door must return Ok");
    let through_pixels = detector
      .detect_pixels::<Barcode>(&plane, &options)
      .expect("the pixel door must return Ok");

    assert_eq!(through_jpeg.len(), 1, "the fixture carries one QR code");
    assert_eq!(through_pixels.len(), 1, "and the pixel door must read it");
    assert_eq!(through_pixels[0].payload, "AVANALYZE-PIXEL-DOOR");
    assert_eq!(through_pixels[0].payload, through_jpeg[0].payload);
    assert_eq!(through_pixels[0].symbology, through_jpeg[0].symbology);
  }

  /// The pixel door has no SOF preflight because it has nothing to
  /// preflight — the geometry was settled when the caller built the
  /// plane. So the refusal that the JPEG door raises at the door, the
  /// pixel door raises at construction, and there is no way to get an
  /// over-ceiling plane past it and into Vision.
  #[test]
  fn an_over_ceiling_plane_cannot_be_built_at_all() {
    let err = PixelPlane::packed(&[], u16::MAX as u32, u16::MAX as u32, PixelFormat::Rgb8)
      .expect_err("an over-ceiling plane must be refused");
    assert_eq!(err.kind(), avanalyze::AnalyzeErrorKind::RequestFailed);
    assert!(err.message().contains("MAX_DECODED_IMAGE_BYTES"));
  }
}
