//! AUDIT: the `Detections` contract as seen from outside the crate.
//!
//! `tests/common` holds a vocabulary written the way a downstream
//! crate writes one — local structs, foreign traits, no adapter. That
//! it compiles at all is the openness proof; the assertions below are
//! the behaviour proof.

mod common;

use avanalyze::{
  Aesthetics, Analysis, BoundingBox, Chirality, Detections, DocumentSegment, HeightEstimation,
  HorizonInfo, conformance,
};
use common::{Bbox, Plain, Quad, Score};

/// The hard family: an outside vocabulary that accepts what the engine
/// emits satisfies the contract.
#[test]
fn outside_vocabulary_satisfies_the_contract() {
  conformance::assert_contract::<Plain>();
}

/// The documented split, made visible. A vocabulary that stores raw
/// values is legal — the engine filters before it constructs — so it
/// passes `assert_contract` and fails `assert_refuses_invalid`. If
/// this ever stops panicking, the two families have collapsed into one
/// and the refusal family has become mandatory by accident.
#[test]
#[should_panic(expected = "BoundingBox::try_new must refuse")]
fn non_validating_vocabulary_fails_the_refusal_family() {
  conformance::assert_refuses_invalid::<Plain>();
}

/// The bundle's associated-type equalities are real: a box built for
/// one slot is the same type every other slot is built from.
#[test]
fn one_bounding_box_type_serves_every_slot() {
  fn same<T>(_: &T, _: &T) {}
  let for_face =
    <Plain as Detections>::BoundingBox::try_new(0.0, 0.0, 1.0, 1.0).expect("infallible vocabulary");
  let for_text =
    <<Plain as Detections>::TextDetection as avanalyze::TextDetection>::BoundingBox::try_new(
      0.1, 0.1, 0.1, 0.1,
    )
    .expect("infallible vocabulary");
  same(&for_face, &for_text);
}

/// Winding order survives the seam: what the engine passes third is
/// the bottom-right corner, and an implementation that stores the four
/// in arrival order gets a perimeter walk, not a raster scan.
#[test]
fn document_corners_arrive_in_winding_order() {
  let quad = Quad::try_new((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), 0.5)
    .expect("infallible vocabulary");
  assert_eq!(quad.corners[2], (0.9, 0.9), "third corner is bottom-right");
  assert_eq!(quad.corners[3], (0.1, 0.9), "fourth corner is bottom-left");
}

/// The horizon seat takes `(angle, confidence)`, not the
/// `(confidence, …)` order every other scored type uses.
#[test]
fn horizon_takes_angle_first() {
  let horizon = common::Horizon::try_new(-0.75, 0.25).expect("infallible vocabulary");
  assert_eq!(horizon.angle, -0.75);
  assert_eq!(horizon.confidence, 0.25);
}

/// The engine-owned vocabularies default to their "not reported"
/// variant, which is what the Vision mappers fall back to.
#[test]
fn engine_enums_default_to_unknown() {
  assert_eq!(Chirality::default(), Chirality::Unknown);
  assert_eq!(HeightEstimation::default(), HeightEstimation::Unknown);
}

/// An [`Analysis`] over an outside vocabulary starts empty and carries
/// no fabricated sentinel.
#[test]
fn analysis_over_outside_vocabulary_starts_empty() {
  let analysis: Analysis<Plain> = Analysis::default();
  assert!(analysis.faces().is_empty());
  assert!(analysis.horizon().is_none());
  assert!(analysis.aesthetics().is_none());

  let filled = analysis.with_aesthetics(Some(Score::new(0.5, false)));
  assert_eq!(
    filled.aesthetics().map(|s| s.overall_score),
    Some(0.5),
    "the aesthetics slot round-trips"
  );
}

/// A box reads back exactly what it was built from — the property the
/// engine's face-pass join depends on.
#[test]
fn bounding_box_reads_back_what_it_was_built_from() {
  let bbox = Bbox::try_new(0.25, 0.5, 0.125, 0.0625).expect("infallible vocabulary");
  assert_eq!(
    (bbox.x(), bbox.y(), bbox.width(), bbox.height()),
    (0.25, 0.5, 0.125, 0.0625)
  );
}
