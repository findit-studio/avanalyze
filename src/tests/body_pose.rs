//! Body pose internals: the 3-D height/estimation coupling and the
//! SIMD return-type encoding the 3-D joint accessor depends on.

use objc2::encode::Encode;

use crate::{
  HeightEstimation,
  body_pose::{SimdFloat4, SimdFloat4x4, sanitize_body_height_pair},
};

/// A finite body_height pairs with whatever height_estimation enum
/// Vision reported. The pair is forwarded unchanged.
#[test]
fn sanitize_body_height_pair_finite_preserves_estimation() {
  let measured = HeightEstimation::Measured;
  let (h, e) = sanitize_body_height_pair(1.75, measured);
  assert!((h - 1.75).abs() < 1e-6);
  assert_eq!(e, measured);

  let reference = HeightEstimation::Reference;
  let (h, e) = sanitize_body_height_pair(0.42, reference);
  assert!((h - 0.42).abs() < 1e-6);
  assert_eq!(e, reference);
}

/// When body_height is non-finite, the estimation enum MUST be forced
/// to `Unknown`. Preserving `Measured`/`Reference` while substituting
/// `0.0` would tell consumers there is a known 0-metre subject — a
/// worse semantic than "unknown estimate".
#[test]
fn sanitize_body_height_pair_non_finite_forces_unknown() {
  for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    // Even with a Measured input the result must be Unknown.
    let (h, e) = sanitize_body_height_pair(raw, HeightEstimation::Measured);
    assert_eq!(h, 0.0, "non-finite must collapse to 0.0 (raw = {raw:?})");
    assert_eq!(
      e,
      HeightEstimation::Unknown,
      "non-finite must force Unknown (raw = {raw:?})",
    );
    // Same for Reference.
    let (h, e) = sanitize_body_height_pair(raw, HeightEstimation::Reference);
    assert_eq!(h, 0.0);
    assert_eq!(e, HeightEstimation::Unknown);
  }
}

/// `SimdFloat4x4::ENCODING` must format as `{?=[4]}` to match Clang's
/// `@encode(simd_float4x4)` and the runtime metadata of
/// `-[VNHumanBodyRecognizedPoint3D position]`. An `Encoding::Unknown`
/// element renders as `{?=[4?]}` and silently breaks every msg_send
/// for that selector under `catch_unwind`, so the string is pinned
/// here: a future objc2 upgrade or accidental edit surfaces as a test
/// failure rather than as 3-D poses that always come back empty.
#[test]
fn simd_float4x4_encoding_matches_clang_at_encode() {
  assert_eq!(SimdFloat4::ENCODING.to_string(), "");
  assert_eq!(SimdFloat4x4::ENCODING.to_string(), "{?=[4]}");
}
