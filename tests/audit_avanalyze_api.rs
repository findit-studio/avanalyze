//! AUDIT: Public API surface + non-macOS stub (R4, R28, R30)
//!
//! Tests the public VisionAnalyzer API on all platforms, driving it
//! through the outside vocabulary in `tests/common`.

mod common;

use avanalyze::{AnalyzeOptions, AppleVisionClassificationOptions, VisionAnalyzer};
use common::Plain;

// -- R30: Non-macOS stub always errors --

#[cfg(not(target_vendor = "apple"))]
mod non_macos {
  use super::*;
  use avanalyze::AnalyzeErrorKind;

  #[test]
  fn stub_returns_error() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options);
    let err = analyzer
      .analyze_keyframe::<Plain>(&[], &options)
      .expect_err("stub must Err");
    assert_eq!(err.kind(), AnalyzeErrorKind::Unsupported);
  }

  #[test]
  fn stub_error_mentions_macos() {
    let options = AnalyzeOptions::new();
    let analyzer = VisionAnalyzer::new(&options);
    let err = analyzer
      .analyze_keyframe::<Plain>(&[0xFF, 0xD8], &options)
      .expect_err("stub must Err");
    assert!(err.message().contains("macOS"));
  }

  #[test]
  fn stub_ignores_data_size() {
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
  fn error_has_kind_and_message() {
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
}

// -- R28: VisionAnalyzer Debug --

#[test]
fn vision_analyzer_debug() {
  let analyzer = VisionAnalyzer::new(&AnalyzeOptions::new());
  let dbg = format!("{analyzer:?}");
  assert!(dbg.contains("VisionAnalyzer"));
}

// -- R30: AnalyzeOptions is Send --

#[test]
fn analyze_options_is_send() {
  fn assert_send<T: Send>() {}
  assert_send::<AnalyzeOptions>();
}

// -- R30: Multiple constructions --

#[test]
fn multiple_analyzer_constructions() {
  let options = AnalyzeOptions::new();
  for _ in 0..10 {
    let _ = VisionAnalyzer::new(&options);
  }
}

// -- R30: Analyzer with custom options --

#[test]
fn analyzer_with_custom_options() {
  let mut opts = AnalyzeOptions::new().with_workers(4);
  opts.classifications_mut().set_min_confidence(0.5);
  opts.classifications_mut().set_max_results(5);
  let analyzer = VisionAnalyzer::new(&opts);
  let _ = format!("{analyzer:?}");
}

// -- R4: Config feature flag combinations --

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

// -- Process-abort regression: a real Vision keyframe must not SIGABRT --

/// Regression for the Vision-framework foreign-exception process abort.
///
/// `analyze_keyframe` runs ~19 Apple Vision detectors. On certain real
/// keyframes a detector raises an Objective-C `NSException` that unwinds
/// across the objc2/Vision FFI boundary. Rust's `catch_unwind` (used in
/// the crate for a separate Rust-panic quirk in the 3D body-pose path)
/// cannot catch a foreign exception — one escaping it aborts the entire
/// process with `fatal runtime error: Rust cannot catch foreign
/// exceptions`. The fix guards every Vision FFI call with
/// `objc2::exception::catch`, degrading a raising detector to an empty
/// result and returning a partial analysis.
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
#[test]
fn analyze_keyframe_does_not_abort_on_real_airport_keyframe() {
  // The desktop resample emits a 288x512 frame for this clip.
  const JPEG: &[u8] = include_bytes!("fixtures/airport_keyframe.jpg");

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
