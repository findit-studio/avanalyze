//! AUDIT: the output contract as seen from outside the crate.
//!
//! `tests/common` holds a vocabulary written the way a downstream
//! crate writes one — local structs, foreign traits, no adapter. That
//! it compiles at all is the openness proof; the assertions below are
//! the behaviour proof.
//!
//! The conformance calls are **per entry point**, mirroring the
//! engine: only `assert_contract` / `assert_refuses_invalid` take the
//! core bundle, and everything else takes the one type that entry
//! point builds.

mod common;

use avanalyze::{
  Aesthetics, Analysis, BoundingBox, Chirality, Detections, DocumentSegment, FaceDetection,
  FaceKeypoints, HeightEstimation, HorizonInfo, TextDetection, conformance,
};
use common::{
  AnimalPose, Barcode, Bbox, Face, FaceLandmarks, HandPose, InstanceMask, Plain, Pose, Pose3, Quad,
  Score, SegmentationMask, Text,
};

/// The hard family: an outside vocabulary that accepts what the engine
/// emits satisfies the contract, entry point by entry point.
#[test]
fn outside_vocabulary_satisfies_the_contract() {
  conformance::assert_contract::<Plain>();

  conformance::assert_text_accepts::<Text>();
  conformance::assert_barcode_accepts::<Barcode>();
  conformance::assert_face_accepts::<Face>();
  conformance::assert_face_landmarks_accept::<FaceLandmarks>();
  conformance::assert_body_pose_accepts::<Pose>("BodyPoser");
  conformance::assert_body_pose_accepts::<AnimalPose>("AnimalPoser");
  conformance::assert_hand_pose_accepts::<HandPose>();
  conformance::assert_body_pose_3d_accepts::<Pose3>();
  conformance::assert_person_instance_mask_accepts::<InstanceMask>();
  conformance::assert_person_segmentation_accepts::<SegmentationMask>();
}

/// The documented split, made visible. A vocabulary that stores raw
/// values is legal — the engine filters before it constructs — so it
/// passes the accept family and fails the refusal family. If this ever
/// stops panicking, the two families have collapsed into one and the
/// refusal family has become mandatory by accident.
#[test]
#[should_panic(expected = "BoundingBox::try_new must refuse")]
fn non_validating_vocabulary_fails_the_refusal_family() {
  conformance::assert_refuses_invalid::<Plain>();
}

/// The per-entry refusal assertions are just as optional as the
/// bundle's, and this vocabulary fails them for the same reason.
#[test]
#[should_panic(expected = "TextDetection::try_new must refuse")]
fn non_validating_vocabulary_fails_the_text_refusals() {
  conformance::assert_text_refusals::<Text>();
}

/// The bundle's associated-type equalities are real, and so are each
/// entry point's: a box built for a core slot is the same type the
/// text and face entry points are built from.
#[test]
fn one_bounding_box_type_serves_every_entry_point() {
  fn same<T>(_: &T, _: &T) {}
  let for_subject =
    <Plain as Detections>::BoundingBox::try_new(0.0, 0.0, 1.0, 1.0).expect("infallible vocabulary");
  let for_text = <Text as TextDetection>::BoundingBox::try_new(0.1, 0.1, 0.1, 0.1)
    .expect("infallible vocabulary");
  let for_face = <Face as FaceDetection>::BoundingBox::try_new(0.2, 0.2, 0.1, 0.1)
    .expect("infallible vocabulary");
  same(&for_subject, &for_text);
  same(&for_subject, &for_face);
}

/// Text runs carry the provenance the engine threads out of its
/// candidate loop: two readings of ONE region share an `observation`
/// and differ by `rank`, and the box they share cannot tell them
/// apart.
#[test]
fn text_candidates_keep_observation_and_rank() {
  let bbox = Bbox::try_new(0.1, 0.1, 0.2, 0.05).expect("infallible vocabulary");
  let best = Text::try_new("hello", 0.9, bbox.clone(), 3, 0).expect("infallible vocabulary");
  let runner_up = Text::try_new("he11o", 0.4, bbox.clone(), 3, 1).expect("infallible vocabulary");
  assert_eq!(best.observation, runner_up.observation, "one region");
  assert_eq!(best.rank, 0, "rank 0 is Vision's best reading");
  assert_eq!(runner_up.rank, 1);
  assert_eq!(
    best.bbox, runner_up.bbox,
    "both candidates re-use the observation's box — only the provenance pair separates them"
  );
}

/// The five-point reduction arrives as a typed set in canonical
/// alignment order, and its absence is representable beside its
/// presence.
#[test]
fn face_keypoints_arrive_typed_and_optional() {
  let bbox = Bbox::try_new(0.1, 0.1, 0.3, 0.3).expect("infallible vocabulary");
  let keypoints = FaceKeypoints::new(
    (0.35, 0.40),
    (0.65, 0.40),
    (0.50, 0.55),
    (0.38, 0.70),
    (0.62, 0.70),
  );
  let with = Face::try_new(
    bbox.clone(),
    0.9,
    Some(0.5),
    None,
    None,
    None,
    Some(keypoints),
  )
  .expect("infallible vocabulary");
  let without =
    Face::try_new(bbox, 0.9, Some(0.5), None, None, None, None).expect("infallible vocabulary");
  assert_eq!(
    with.keypoints.map(|k| k.points()),
    Some([
      (0.35, 0.40),
      (0.65, 0.40),
      (0.50, 0.55),
      (0.38, 0.70),
      (0.62, 0.70)
    ]),
    "left eye, right eye, nose tip, left mouth corner, right mouth corner"
  );
  assert_eq!(
    without.keypoints, None,
    "a face Vision computed no reduction for carries absence, not an origin-valued set"
  );
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
  assert!(analysis.classifications().is_empty());
  assert!(analysis.human_subjects().is_empty());
  assert!(analysis.animal_subjects().is_empty());
  assert!(analysis.horizon().is_none());
  assert!(analysis.aesthetics().is_none());

  let filled = analysis.with_aesthetics(Some(Score::new(0.5, false)));
  assert_eq!(
    filled.aesthetics().map(|s| s.overall_score),
    Some(0.5),
    "the aesthetics slot round-trips"
  );
}

/// A box reads back exactly what it was built from — the property
/// every consumer reading a detection's geometry depends on.
#[test]
fn bounding_box_reads_back_what_it_was_built_from() {
  let bbox = Bbox::try_new(0.25, 0.5, 0.125, 0.0625).expect("infallible vocabulary");
  assert_eq!(
    (bbox.x(), bbox.y(), bbox.width(), bbox.height()),
    (0.25, 0.5, 0.125, 0.0625)
  );
}
