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
  PersonMasker, TextRecognizer, VisionAnalyzer,
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
    let analyzer = VisionAnalyzer::new(&options);
    let err = analyzer
      .analyze_keyframe::<Plain>(&[], &options)
      .expect_err("stub must Err");
    assert_eq!(err.kind(), AnalyzeErrorKind::Unsupported);
  }

  #[test]
  fn analyzer_stub_error_mentions_macos() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options);
    let err = analyzer
      .analyze_keyframe::<Plain>(&[0xFF, 0xD8], &options)
      .expect_err("stub must Err");
    assert!(err.message().contains("macOS"));
  }

  #[test]
  fn analyzer_stub_ignores_data_size() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options);
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
    let analyzer = VisionAnalyzer::new(&options);
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
        .recognize::<Text>(&[], &text)
        .expect_err("stub must Err"),
    );

    let barcode = AppleVisionBarcodeOptions::new();
    check(
      BarcodeDetector::new(&barcode)
        .detect::<Barcode>(&[], &barcode)
        .expect_err("stub must Err"),
    );

    let face = AppleVisionFaceOptions::new();
    check(
      FaceDetector::new(&face)
        .detect::<Face>(&[], &face)
        .expect_err("stub must Err"),
    );

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    check(
      FaceLandmarker::new(&landmarks)
        .detect::<FaceLandmarks>(&[], &landmarks)
        .expect_err("stub must Err"),
    );

    let poser = AppleVisionBodyPoserOptions::new();
    check(
      BodyPoser::new(&poser)
        .detect_2d::<Pose>(&[], &poser)
        .expect_err("stub must Err"),
    );
    check(
      BodyPoser::new(&poser)
        .detect_3d::<Pose3>(&[], &poser)
        .expect_err("stub must Err"),
    );

    let hand = AppleVisionHandPoseOptions::new();
    check(
      HandPoser::new(&hand)
        .detect::<HandPose>(&[], &hand)
        .expect_err("stub must Err"),
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    check(
      AnimalPoser::new(&animal)
        .detect::<AnimalPose>(&[], &animal)
        .expect_err("stub must Err"),
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    check(
      PersonMasker::new(&masker)
        .instance_masks::<InstanceMask>(&[], &masker)
        .expect_err("stub must Err"),
    );
    check(
      PersonMasker::new(&masker)
        .segmentation_masks::<SegmentationMask>(&[], &masker)
        .expect_err("stub must Err"),
    );
  }
}

// -- Debug, construction, and options plumbing on every platform --

#[test]
fn every_entry_point_is_debug() {
  assert!(format!("{:?}", VisionAnalyzer::new(&AnalyzeOptions::new())).contains("VisionAnalyzer"));
  assert!(
    format!("{:?}", TextRecognizer::new(&AppleVisionTextOptions::new())).contains("TextRecognizer")
  );
  assert!(
    format!(
      "{:?}",
      BarcodeDetector::new(&AppleVisionBarcodeOptions::new())
    )
    .contains("BarcodeDetector")
  );
  assert!(
    format!("{:?}", FaceDetector::new(&AppleVisionFaceOptions::new())).contains("FaceDetector")
  );
  assert!(
    format!(
      "{:?}",
      FaceLandmarker::new(&AppleVisionFaceLandmarkOptions::new())
    )
    .contains("FaceLandmarker")
  );
  assert!(
    format!("{:?}", BodyPoser::new(&AppleVisionBodyPoserOptions::new())).contains("BodyPoser")
  );
  assert!(
    format!("{:?}", HandPoser::new(&AppleVisionHandPoseOptions::new())).contains("HandPoser")
  );
  assert!(
    format!(
      "{:?}",
      AnimalPoser::new(&AppleVisionAnimalPoseOptions::new())
    )
    .contains("AnimalPoser")
  );
  assert!(
    format!(
      "{:?}",
      PersonMasker::new(&AppleVisionPersonMaskerOptions::new())
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
    let _ = VisionAnalyzer::new(&options);
  }
}

#[test]
fn analyzer_with_custom_options() {
  let mut opts = AnalyzeOptions::new().with_workers(4);
  opts.classifications_mut().set_min_confidence(0.5);
  opts.classifications_mut().set_max_results(5);
  let analyzer = VisionAnalyzer::new(&opts);
  let _ = format!("{analyzer:?}");
}

/// The one knob Apple bakes into a request object still reaches its
/// entry point at construction, clamped or not.
#[test]
fn hand_poser_takes_its_baked_knob() {
  for count in [1usize, 2, 6, 64] {
    let opts = AppleVisionHandPoseOptions::new().with_maximum_hand_count(count);
    let _ = HandPoser::new(&opts);
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
    let analyzer = VisionAnalyzer::new(&options);
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
    let recognizer = TextRecognizer::new(&options);
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
    let detector = BarcodeDetector::new(&options);
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
    let detector = FaceDetector::new(&options);
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
    let detector = FaceDetector::new(&options);
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
    let detector = FaceDetector::new(&options);

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
    let landmarker = FaceLandmarker::new(&options);
    let _ = landmarker
      .detect::<FaceLandmarks>(JPEG, &options)
      .expect("detect must return Ok");
  }

  /// The 3-D pass is the one that raises on this fixture, so it gets
  /// its own call: `Ok` here is the no-abort contract.
  #[test]
  fn body_poser_runs_both_dimensions_on_a_real_keyframe() {
    let options = AppleVisionBodyPoserOptions::new();
    let poser = BodyPoser::new(&options);
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
    let poser = HandPoser::new(&options);
    let _ = poser
      .detect::<HandPose>(JPEG, &options)
      .expect("detect must return Ok");
  }

  #[test]
  fn animal_poser_runs_on_a_real_keyframe() {
    let options = AppleVisionAnimalPoseOptions::new();
    let poser = AnimalPoser::new(&options);
    let _ = poser
      .detect::<AnimalPose>(JPEG, &options)
      .expect("detect must return Ok");
  }

  #[test]
  fn person_masker_runs_both_kinds_on_a_real_keyframe() {
    let options = AppleVisionPersonMaskerOptions::new();
    let masker = PersonMasker::new(&options);
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
        .analyze_keyframe::<Plain>(&huge, &options)
        .expect_err("oversized input must be refused"),
    );

    let text = AppleVisionTextOptions::new();
    check(
      TextRecognizer::new(&text)
        .recognize::<Text>(&huge, &text)
        .expect_err("oversized input must be refused"),
    );

    let barcode = AppleVisionBarcodeOptions::new();
    check(
      BarcodeDetector::new(&barcode)
        .detect::<Barcode>(&huge, &barcode)
        .expect_err("oversized input must be refused"),
    );

    let face = AppleVisionFaceOptions::new();
    check(
      FaceDetector::new(&face)
        .detect::<Face>(&huge, &face)
        .expect_err("oversized input must be refused"),
    );

    let landmarks = AppleVisionFaceLandmarkOptions::new();
    check(
      FaceLandmarker::new(&landmarks)
        .detect::<FaceLandmarks>(&huge, &landmarks)
        .expect_err("oversized input must be refused"),
    );

    let poser = AppleVisionBodyPoserOptions::new();
    check(
      BodyPoser::new(&poser)
        .detect_2d::<Pose>(&huge, &poser)
        .expect_err("oversized input must be refused"),
    );
    check(
      BodyPoser::new(&poser)
        .detect_3d::<Pose3>(&huge, &poser)
        .expect_err("oversized input must be refused"),
    );

    let hand = AppleVisionHandPoseOptions::new();
    check(
      HandPoser::new(&hand)
        .detect::<HandPose>(&huge, &hand)
        .expect_err("oversized input must be refused"),
    );

    let animal = AppleVisionAnimalPoseOptions::new();
    check(
      AnimalPoser::new(&animal)
        .detect::<AnimalPose>(&huge, &animal)
        .expect_err("oversized input must be refused"),
    );

    let masker = AppleVisionPersonMaskerOptions::new();
    check(
      PersonMasker::new(&masker)
        .instance_masks::<InstanceMask>(&huge, &masker)
        .expect_err("oversized input must be refused"),
    );
    check(
      PersonMasker::new(&masker)
        .segmentation_masks::<SegmentationMask>(&huge, &masker)
        .expect_err("oversized input must be refused"),
    );
  }
}
