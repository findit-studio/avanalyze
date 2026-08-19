//! Runnable proof that an output vocabulary fits the engine.
//!
//! The [`Detections`] bundle is a set of signatures; it cannot say
//! that a bounding box must accept the full frame, that the horizon's
//! "nothing detected" sentinel must construct, or that document
//! corners arrive in winding order. Those are the conventions the
//! engine actually relies on, and this module turns them into
//! assertions you can run against your own bundle:
//!
//! ```ignore
//! #[test]
//! fn my_vocabulary_fits_the_engine() {
//!   avanalyze::conformance::assert_contract::<MyVocabulary>();
//! }
//! ```
//!
//! Two families live here, and the split is deliberate:
//!
//! - [`assert_contract`] is the **hard** contract — every value the
//!   engine can legitimately emit must be accepted. Failing it means
//!   detections will silently vanish at runtime.
//! - [`assert_refuses_invalid`] is for **validating** vocabularies
//!   that reject bad input. The engine filters non-finite and
//!   out-of-domain values before construction, so passing this family
//!   is not required to work with the engine — it is a second line of
//!   defence, and an implementation that stores raw floats is entitled
//!   to skip it.
//!
//! Every assertion panics with the trait and the input it fed, so a
//! failure names the seat that is wrong.

use crate::contract::{
  Aesthetics, BarcodeDetection, BodyPose3DDetection, BodyPose3DJoint, BodyPoseDetection,
  BodyPoseJoint, BoundingBox, Chirality, Detection, Detections, DocumentSegment, FaceDetection,
  FaceLandmarkRegion, FaceLandmarksDetection, HandPoseDetection, HeightEstimation, HorizonInfo,
  PersonInstanceMaskDetection, PersonSegmentationMask, SaliencyRegion, SubjectDetection,
  TextDetection,
};

/// A 2×2 mask payload with one foreground pixel — the canonical
/// one-byte-per-pixel shape the engine emits.
const MASK_2X2: &[u8] = &[0, 0, 255, 0];

/// Builds the interior box the accept-family reuses. Panics with the
/// caller's context if the vocabulary refuses it.
fn interior_bbox<B: BoundingBox>() -> B {
  match B::try_new(0.1, 0.2, 0.3, 0.4) {
    Ok(bbox) => bbox,
    Err(_) => panic!("BoundingBox::try_new refused the interior box (0.1, 0.2, 0.3, 0.4)"),
  }
}

fn joint_2d<J: BodyPoseJoint>() -> J {
  match J::try_new("neck", 0.5, 0.5, 0.5) {
    Ok(joint) => joint,
    Err(_) => panic!("BodyPoseJoint::try_new refused an in-range joint"),
  }
}

/// Asserts the whole hard contract: every canonical engine output is
/// accepted, in the argument order the engine uses.
///
/// Panics on the first violation.
pub fn assert_contract<D: Detections>() {
  assert_bounding_box_accepts::<D>();
  assert_detection_accepts::<D>();
  assert_faces_accept::<D>();
  assert_poses_accept::<D>();
  assert_masks_accept::<D>();
  assert_text_and_barcodes_accept::<D>();
  assert_frame_wide_accept::<D>();
}

/// The unit-square domain: both edges of `0.0..=1.0` are inside it,
/// and what a box is built from is what it reads back.
///
/// The read-back matters because the engine joins its two face passes
/// by intersection-over-union computed from these accessors.
pub fn assert_bounding_box_accepts<D: Detections>() {
  assert!(
    D::BoundingBox::try_new(0.0, 0.0, 1.0, 1.0).is_ok(),
    "BoundingBox::try_new must accept the full frame (0, 0, 1, 1)"
  );
  let bbox = interior_bbox::<D::BoundingBox>();
  assert_eq!(
    bbox.x(),
    0.1,
    "BoundingBox::x must read back what it was built from"
  );
  assert_eq!(
    bbox.y(),
    0.2,
    "BoundingBox::y must read back what it was built from"
  );
  assert_eq!(
    bbox.width(),
    0.3,
    "BoundingBox::width must read back what it was built from"
  );
  assert_eq!(
    bbox.height(),
    0.4,
    "BoundingBox::height must read back what it was built from"
  );
}

/// Labels and confidences at both ends of `0.0..=1.0`, and the
/// infallible subject pairing.
pub fn assert_detection_accepts<D: Detections>() {
  for confidence in [0.0_f32, 1.0] {
    assert!(
      D::Detection::try_new("person", confidence).is_ok(),
      "Detection::try_new must accept a confidence at the domain edge"
    );
  }
  let Ok(detection) = D::Detection::try_new("person", 0.5) else {
    panic!("Detection::try_new refused an in-range detection");
  };
  let _ = D::SubjectDetection::new(detection, interior_bbox::<D::BoundingBox>());
}

/// Faces, including the `0.0` capture quality the engine writes for a
/// face the capture-quality pass did not cover, and signed pose
/// angles.
pub fn assert_faces_accept<D: Detections>() {
  let bbox = interior_bbox::<D::BoundingBox>();
  assert!(
    D::FaceDetection::try_new(bbox, 1.0, 0.0, -0.5, 0.25, 3.0).is_ok(),
    "FaceDetection::try_new must accept an unmatched capture quality (0.0) and signed \
     roll/yaw/pitch radians"
  );

  let points = [(0.0_f32, 0.0_f32), (1.0, 1.0), (0.42, 0.13)];
  let Ok(region) = D::FaceLandmarkRegion::try_new("allPoints", &points) else {
    panic!("FaceLandmarkRegion::try_new refused in-range landmark points");
  };
  assert!(
    D::FaceLandmarksDetection::try_new(interior_bbox::<D::BoundingBox>(), 0.0, vec![region])
      .is_ok(),
    "FaceLandmarksDetection::try_new must accept a zero-confidence landmark set"
  );
}

/// The four pose shapes, including 3-D's metre-scale coordinates and
/// the coupled `(0.0, Unknown)` height fallback.
pub fn assert_poses_accept<D: Detections>() {
  let joint = joint_2d::<D::BodyPoseJoint>();
  assert_eq!(
    joint.name(),
    "neck",
    "BodyPoseJoint::name must read back what it was built from — the engine sorts on it"
  );
  assert!(
    D::BodyPoseDetection::try_new(interior_bbox::<D::BoundingBox>(), 0.0, vec![joint]).is_ok(),
    "BodyPoseDetection::try_new must accept a zero-confidence pose"
  );

  for chirality in [Chirality::Unknown, Chirality::Left, Chirality::Right] {
    assert!(
      D::HandPoseDetection::try_new(
        interior_bbox::<D::BoundingBox>(),
        0.0,
        chirality,
        vec![joint_2d::<D::BodyPoseJoint>()],
      )
      .is_ok(),
      "HandPoseDetection::try_new must accept every chirality"
    );
  }

  let Ok(joint_3d) = D::BodyPose3DJoint::try_new("root", -1.5, 2.5, 0.75, 0.5) else {
    panic!("BodyPose3DJoint::try_new refused model-space metres — 3-D joints are not normalized");
  };
  assert_eq!(
    joint_3d.name(),
    "root",
    "BodyPose3DJoint::name must read back what it was built from — the engine sorts on it"
  );
  assert!(
    D::BodyPose3DDetection::try_new(0.5, 1.75, HeightEstimation::Measured, vec![joint_3d]).is_ok(),
    "BodyPose3DDetection::try_new must accept a measured height in metres"
  );
  let Ok(joint_3d) = D::BodyPose3DJoint::try_new("root", 0.0, 0.0, 0.0, 0.0) else {
    panic!("BodyPose3DJoint::try_new refused a zeroed joint");
  };
  assert!(
    D::BodyPose3DDetection::try_new(0.0, 0.0, HeightEstimation::Unknown, vec![joint_3d]).is_ok(),
    "BodyPose3DDetection::try_new must accept the coupled (0.0, Unknown) height fallback"
  );
}

/// Both mask shapes, at the one-byte-per-pixel payload the engine
/// always emits — and note the argument order differs by
/// `instance_index` alone.
pub fn assert_masks_accept<D: Detections>() {
  assert!(
    D::PersonInstanceMaskDetection::try_new(
      interior_bbox::<D::BoundingBox>(),
      0.0,
      0,
      2,
      2,
      MASK_2X2,
    )
    .is_ok(),
    "PersonInstanceMaskDetection::try_new must accept instance index 0 and a width*height payload"
  );
  assert!(
    D::PersonSegmentationMask::try_new(interior_bbox::<D::BoundingBox>(), 0.0, 2, 2, MASK_2X2)
      .is_ok(),
    "PersonSegmentationMask::try_new must accept a width*height payload"
  );
}

/// Text and barcodes — both take their box **last**.
pub fn assert_text_and_barcodes_accept<D: Detections>() {
  assert!(
    D::TextDetection::try_new("a", 0.0, interior_bbox::<D::BoundingBox>()).is_ok(),
    "TextDetection::try_new must accept a single-character run at zero confidence"
  );
  assert!(
    D::BarcodeDetection::try_new(
      "0123456789",
      "VNBarcodeSymbologyQR",
      0.0,
      interior_bbox::<D::BoundingBox>(),
    )
    .is_ok(),
    "BarcodeDetection::try_new must accept Apple's raw symbology string"
  );
  assert!(
    D::SaliencyRegion::try_new(interior_bbox::<D::BoundingBox>(), 0.0).is_ok(),
    "SaliencyRegion::try_new must accept a zero-confidence region"
  );
}

/// The single-valued slots: the horizon sentinel, a signed horizon
/// angle, document winding order, and the signed aesthetics score.
pub fn assert_frame_wide_accept<D: Detections>() {
  assert!(
    D::HorizonInfo::try_new(0.0, 0.0).is_ok(),
    "HorizonInfo::try_new(0.0, 0.0) is the engine's no-detection sentinel and must be accepted"
  );
  assert!(
    D::HorizonInfo::try_new(-0.75, 1.0).is_ok(),
    "HorizonInfo::try_new takes (angle, confidence) — the angle is signed radians and unbounded"
  );

  assert!(
    D::DocumentSegment::try_new((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), 0.5).is_ok(),
    "DocumentSegment::try_new takes corners in winding order (top-left, top-right, \
     bottom-right, bottom-left); a well-formed quad in that order must be accepted"
  );

  let _ = D::Aesthetics::new(0.0, false);
  let _ = D::Aesthetics::new(-1.0, true);
}

/// Asserts the optional refusal family: non-finite and out-of-domain
/// inputs are rejected.
///
/// Not required by the engine — it filters these before construction —
/// but a validating vocabulary should pass, and a vocabulary that
/// stores raw floats will not. Run it only if yours validates.
///
/// Panics on the first violation.
pub fn assert_refuses_invalid<D: Detections>() {
  assert_bounding_box_refusals::<D>();
  assert_confidence_refusals::<D>();
  assert_coordinate_refusals::<D>();
  assert_mask_refusals::<D>();
  assert_document_winding_refusal::<D>();
}

/// A box must stay finite, inside the unit square, and non-degenerate.
pub fn assert_bounding_box_refusals<D: Detections>() {
  assert!(
    D::BoundingBox::try_new(f32::NAN, 0.0, 0.5, 0.5).is_err(),
    "BoundingBox::try_new must refuse a non-finite component"
  );
  assert!(
    D::BoundingBox::try_new(0.0, 0.0, 0.0, 0.5).is_err(),
    "BoundingBox::try_new must refuse a zero-extent box"
  );
  assert!(
    D::BoundingBox::try_new(0.5, 0.0, 0.9, 0.5).is_err(),
    "BoundingBox::try_new must refuse a box that extends past the frame edge"
  );
}

/// Confidences must be finite and inside `0.0..=1.0` wherever they
/// appear.
pub fn assert_confidence_refusals<D: Detections>() {
  for confidence in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
    assert!(
      D::Detection::try_new("person", confidence).is_err(),
      "Detection::try_new must refuse a confidence outside a finite 0.0..=1.0"
    );
    assert!(
      D::SaliencyRegion::try_new(interior_bbox::<D::BoundingBox>(), confidence).is_err(),
      "SaliencyRegion::try_new must refuse a confidence outside a finite 0.0..=1.0"
    );
    assert!(
      D::TextDetection::try_new("a", confidence, interior_bbox::<D::BoundingBox>()).is_err(),
      "TextDetection::try_new must refuse a confidence outside a finite 0.0..=1.0"
    );
    assert!(
      D::HorizonInfo::try_new(0.0, confidence).is_err(),
      "HorizonInfo::try_new must refuse a confidence outside a finite 0.0..=1.0 — note the \
       confidence is the SECOND argument"
    );
  }
}

/// Normalized coordinates must be finite and inside `0.0..=1.0`.
/// 3-D joints are exempt: they are model-space metres.
pub fn assert_coordinate_refusals<D: Detections>() {
  assert!(
    D::BodyPoseJoint::try_new("neck", f32::NAN, 0.5, 0.5).is_err(),
    "BodyPoseJoint::try_new must refuse a non-finite coordinate"
  );
  assert!(
    D::BodyPoseJoint::try_new("neck", 1.5, 0.5, 0.5).is_err(),
    "BodyPoseJoint::try_new must refuse a coordinate outside 0.0..=1.0"
  );
  assert!(
    D::FaceLandmarkRegion::try_new("nose", &[(0.5, 0.5), (f32::NAN, 0.5)]).is_err(),
    "FaceLandmarkRegion::try_new must refuse a non-finite point"
  );
}

/// Mask dimensions must be non-degenerate and the payload non-empty.
pub fn assert_mask_refusals<D: Detections>() {
  assert!(
    D::PersonSegmentationMask::try_new(interior_bbox::<D::BoundingBox>(), 0.5, 0, 2, MASK_2X2)
      .is_err(),
    "PersonSegmentationMask::try_new must refuse a zero-width mask"
  );
  assert!(
    D::PersonInstanceMaskDetection::try_new(interior_bbox::<D::BoundingBox>(), 0.5, 0, 2, 2, &[],)
      .is_err(),
    "PersonInstanceMaskDetection::try_new must refuse an empty payload"
  );
}

/// The winding-order nail: reading the corners in raster order
/// (top-left, top-right, bottom-left, bottom-right) turns every real
/// document into a self-intersecting bow-tie, so a validating
/// vocabulary must refuse one.
pub fn assert_document_winding_refusal<D: Detections>() {
  assert!(
    D::DocumentSegment::try_new((0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), 0.5).is_err(),
    "DocumentSegment::try_new must refuse a bow-tie quad — the third argument is the \
     BOTTOM-RIGHT corner, not the bottom-left"
  );
}
