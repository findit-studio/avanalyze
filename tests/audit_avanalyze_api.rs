//! AUDIT: Public API surface + non-macOS stub (R4, R28, R30)
//!
//! Tests the public VisionAnalyzer API on all platforms.

use avanalyze::*;

// -- R30: Non-macOS stub always errors --

#[cfg(not(target_vendor = "apple"))]
mod non_macos {
  use super::*;
  use core::num::NonZeroU32;
  use mediatime::{Timebase, Timestamp};

  fn make_params() -> (Uuid7, Uuid7, Timestamp, Dimensions, KeyframeExtractor) {
    let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
    (
      Uuid7::new(),
      Uuid7::new(),
      Timestamp::new(0, tb),
      Dimensions::new(320, 180),
      KeyframeExtractor::Manual,
    )
  }

  #[test]
  fn stub_returns_error() {
    let analyzer = VisionAnalyzer::new(ServiceOptions::new());
    let (sid, kid, pts, dims, ext) = make_params();
    let err = analyzer
      .analyze_keyframe(sid, kid, pts, dims, ext, &[])
      .expect_err("stub must Err");
    assert_eq!(err.code(), ErrorCode::AppleVisionFailed);
  }

  #[test]
  fn stub_error_mentions_macos() {
    let analyzer = VisionAnalyzer::new(ServiceOptions::new());
    let (sid, kid, pts, dims, ext) = make_params();
    let err = analyzer
      .analyze_keyframe(sid, kid, pts, dims, ext, &[0xFF, 0xD8])
      .expect_err("stub must Err");
    assert!(err.message().contains("macOS"));
  }

  #[test]
  fn stub_ignores_data_size() {
    let analyzer = VisionAnalyzer::new(ServiceOptions::new());
    let (sid, kid, pts, dims, ext) = make_params();
    let e1 = analyzer
      .analyze_keyframe(sid, kid, pts, dims, ext, &[])
      .expect_err("empty");
    let e2 = analyzer
      .analyze_keyframe(sid, kid, pts, dims, ext, &vec![0u8; 1024])
      .expect_err("large");
    assert_eq!(e1.code(), e2.code());
  }

  #[test]
  fn error_has_code_and_message() {
    let analyzer = VisionAnalyzer::new(ServiceOptions::new());
    let (sid, kid, pts, dims, ext) = make_params();
    let err = analyzer
      .analyze_keyframe(sid, kid, pts, dims, ext, &[])
      .expect_err("stub");
    let _ = format!("{:?}", err.code());
    assert!(!err.message().is_empty());
  }
}

// -- R28: VisionAnalyzer Debug --

#[test]
fn vision_analyzer_debug() {
  let analyzer = VisionAnalyzer::new(ServiceOptions::new());
  let dbg = format!("{analyzer:?}");
  assert!(dbg.contains("VisionAnalyzer"));
}

// -- R30: ServiceOptions is Send --

#[test]
fn service_options_is_send() {
  fn assert_send<T: Send>() {}
  assert_send::<ServiceOptions>();
}

// -- R30: Multiple constructions --

#[test]
fn multiple_analyzer_constructions() {
  for _ in 0..10 {
    let _ = VisionAnalyzer::new(ServiceOptions::new());
  }
}

// -- R30: Analyzer with custom options --

#[test]
fn analyzer_with_custom_options() {
  let mut opts = ServiceOptions::new().with_workers(4);
  opts.classifications_mut().set_min_confidence(0.5);
  opts.classifications_mut().set_max_results(5);
  let analyzer = VisionAnalyzer::new(opts);
  let _ = format!("{analyzer:?}");
}

// -- R4: Config feature flag combinations --

#[test]
fn default_feature_compiles() {
  // This test file compiles with default features (no serde, no tracing)
  let _ = ServiceOptions::new();
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
/// result and returning a partial `Keyframe`.
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
  use core::num::NonZeroU32;
  use mediaframe::frame::Dimensions;
  use mediaschema::domain::{KeyframeExtractor, Uuid7};
  use mediatime::{Timebase, Timestamp};

  // The desktop resample emits a 288x512 frame for this clip.
  const JPEG: &[u8] = include_bytes!("fixtures/airport_keyframe.jpg");

  let analyzer = VisionAnalyzer::new(ServiceOptions::new());
  let tb = Timebase::new(1, NonZeroU32::new(1000).expect("nonzero den"));
  let keyframe = analyzer
    .analyze_keyframe(
      Uuid7::new(),
      Uuid7::new(),
      Timestamp::new(0, tb),
      Dimensions::new(288, 512),
      KeyframeExtractor::Manual,
      JPEG,
    )
    .expect("analyze_keyframe must return Ok (a partial Keyframe), never abort the process");

  // The returned aggregate must carry the caller-supplied frame
  // metadata — i.e. it is a real, usable Keyframe, not a degenerate
  // shell.
  assert_eq!(keyframe.dimensions(), Dimensions::new(288, 512));
}
