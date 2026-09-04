//! Runnable proof that an output vocabulary fits the engine.
//!
//! A trait is a set of signatures; it cannot say that a bounding box
//! must accept the full frame, that the horizon's "nothing detected"
//! sentinel must construct, or that document corners arrive in winding
//! order. Those are the conventions the engine actually relies on, and
//! this module turns them into assertions you can run against your own
//! types:
//!
//! ```ignore
//! #[test]
//! fn my_vocabulary_fits_the_engine() {
//!   // The core bundle, for `VisionAnalyzer`:
//!   avanalyze::conformance::assert_contract::<MyBundle>();
//!   // And one call per entry point you actually use:
//!   avanalyze::conformance::assert_text_accepts::<MyText>();
//!   avanalyze::conformance::assert_face_accepts::<MyFace>();
//! }
//! ```
//!
//! The assertions are **per entry point**, mirroring the engine: a
//! consumer that only recognises text runs the text assertions and
//! implements nothing else. Only [`assert_contract`] and
//! [`assert_refuses_invalid`] take the whole [`Detections`] bundle,
//! because that is what [`VisionAnalyzer`](crate::VisionAnalyzer)
//! takes.
//!
//! Two families live here, and the split is deliberate:
//!
//! - The **accept** family is the *hard* contract — every value the
//!   engine can legitimately emit must be accepted. Failing it means
//!   detections will silently vanish at runtime.
//! - The **refusal** family is for *validating* vocabularies that
//!   reject bad input. The engine filters non-finite and out-of-domain
//!   values before construction, so passing this family is not
//!   required to work with the engine — it is a second line of
//!   defence, and an implementation that stores raw floats is entitled
//!   to skip it.
//!
//! Every assertion panics with the trait and the input it fed, so a
//! failure names the seat that is wrong.

use crate::{
  Aesthetics, BarcodeDetection, BodyPose3DDetection, BodyPose3DJoint, BodyPoseDetection,
  BodyPoseJoint, BoundingBox, Chirality, Detection, Detections, DocumentSegment, FaceDetection,
  FaceKeypoints, FaceLandmarkRegion, FaceLandmarksDetection, HandPoseDetection, HeightEstimation,
  HorizonInfo, PersonInstanceMaskDetection, PersonSegmentationMask, SaliencyRegion,
  SubjectDetection, TextDetection,
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

/// Builds the in-range 2-D joint the pose family reuses. `seat` names
/// the entry point the type came from: the three 2-D joint seats are
/// three independent types, so a failure has to say which one refused.
fn joint_2d<J: BodyPoseJoint>(seat: &str) -> J {
  match J::try_new("neck", 0.5, 0.5, 0.5) {
    Ok(joint) => joint,
    Err(_) => panic!("BodyPoseJoint::try_new refused an in-range joint (seat: {seat})"),
  }
}

/// The canonical five-point reduction the engine emits: an upright
/// face's eyes, nose tip, and mouth corners.
fn canonical_keypoints() -> FaceKeypoints {
  FaceKeypoints::new(
    (0.35, 0.40),
    (0.65, 0.40),
    (0.50, 0.55),
    (0.38, 0.70),
    (0.62, 0.70),
  )
}

// ----- the core bundle ------------------------------------------------------

/// Asserts the whole hard contract for the core
/// [`VisionAnalyzer`](crate::VisionAnalyzer) bundle: every canonical
/// output of the eight batched detections is accepted, in the argument
/// order the engine uses.
///
/// The other entry points have their own assertions — see the module
/// docs.
///
/// Panics on the first violation.
pub fn assert_contract<D: Detections>() {
  assert_bounding_box_accepts::<D::BoundingBox>();
  assert_detection_accepts::<D>();
  assert_saliency_accepts::<D>();
  assert_frame_wide_accept::<D>();
}

/// The unit-square domain: both edges of `0.0..=1.0` are inside it,
/// and what a box is built from is what it reads back.
///
/// The read-back matters because those accessors are how every
/// consumer reads a detection's geometry back out.
pub fn assert_bounding_box_accepts<B: BoundingBox>() {
  assert!(
    B::try_new(0.0, 0.0, 1.0, 1.0).is_ok(),
    "BoundingBox::try_new must accept the full frame (0, 0, 1, 1)"
  );
  let bbox = interior_bbox::<B>();
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

/// Salient regions, which both saliency passes share.
pub fn assert_saliency_accepts<D: Detections>() {
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

// ----- per entry point ------------------------------------------------------

/// Text runs, including the provenance pair
/// [`TextRecognizer`](crate::TextRecognizer) threads out of its loop.
pub fn assert_text_accepts<T: TextDetection>() {
  assert!(
    T::try_new("a", 0.0, interior_bbox::<T::BoundingBox>(), 0, 0).is_ok(),
    "TextDetection::try_new must accept a single-character run at zero confidence, at the \
     first observation's best candidate"
  );
  let best = T::try_new("hello", 0.9, interior_bbox::<T::BoundingBox>(), 3, 0);
  let runner_up = T::try_new("he11o", 0.4, interior_bbox::<T::BoundingBox>(), 3, 1);
  assert!(
    best.is_ok() && runner_up.is_ok(),
    "TextDetection::try_new must accept two competing readings of ONE observation — same \
     `observation`, different `rank` — as independently representable values; collapsing them \
     loses the candidate list Vision produced"
  );
  assert!(
    T::try_new(
      "x",
      0.5,
      interior_bbox::<T::BoundingBox>(),
      usize::MAX,
      usize::MAX
    )
    .is_ok(),
    "TextDetection::try_new must accept any in-range `usize` for observation/rank — they are \
     indices, not a bounded vocabulary"
  );
}

/// Barcodes, which take their box **last**.
pub fn assert_barcode_accepts<B: BarcodeDetection>() {
  assert!(
    B::try_new(
      "0123456789",
      "VNBarcodeSymbologyQR",
      0.0,
      interior_bbox::<B::BoundingBox>(),
    )
    .is_ok(),
    "BarcodeDetection::try_new must accept Apple's raw symbology string"
  );
}

/// Faces: capture quality's three-way truth (`Some(q)` measured,
/// `Some(0.0)` measured-and-terrible, `None` never measured), signed
/// pose angles, the pose-angle absence the seat exists to represent,
/// and the five-point reduction both present and absent.
pub fn assert_face_accepts<F: FaceDetection>() {
  assert!(
    F::try_new(
      interior_bbox::<F::BoundingBox>(),
      1.0,
      None,
      Some(-0.5),
      Some(0.25),
      Some(3.0),
      Some(canonical_keypoints()),
    )
    .is_ok(),
    "FaceDetection::try_new must accept a face the capture-quality pass never covered — \
     `None`, not `Some(0.0)` — alongside signed roll/yaw/pitch radians"
  );
  assert!(
    F::try_new(
      interior_bbox::<F::BoundingBox>(),
      1.0,
      Some(0.0),
      None,
      None,
      None,
      None,
    )
    .is_ok(),
    "FaceDetection::try_new must accept a face with no pose angles computed at all and no \
     keypoints derived — the states these seats exist to represent, distinct from a head \
     measured level and from a face whose keypoints all landed at the origin"
  );
  assert!(
    F::try_new(
      interior_bbox::<F::BoundingBox>(),
      1.0,
      Some(0.0),
      Some(0.0),
      None,
      Some(-1.2),
      Some(canonical_keypoints()),
    )
    .is_ok(),
    "FaceDetection::try_new must accept angles that are present, absent, and genuinely \
     zero on the SAME face — Vision reports roll/yaw/pitch independently"
  );
  let measured = F::try_new(
    interior_bbox::<F::BoundingBox>(),
    1.0,
    Some(0.0),
    Some(0.1),
    Some(0.1),
    Some(0.1),
    Some(canonical_keypoints()),
  );
  let unmatched = F::try_new(
    interior_bbox::<F::BoundingBox>(),
    1.0,
    None,
    Some(0.1),
    Some(0.1),
    Some(0.1),
    None,
  );
  assert!(
    measured.is_ok() && unmatched.is_ok(),
    "FaceDetection::try_new must accept a measured-and-terrible face (`Some(0.0)`) and an \
     unmatched face (`None`) as independently representable values in the SAME detection \
     set — the two must never collapse to the same wire value, and the same holds for a \
     face with keypoints beside one Vision computed no reduction for"
  );

  let keypoints = canonical_keypoints();
  assert_eq!(
    keypoints.points(),
    [
      keypoints.left_eye(),
      keypoints.right_eye(),
      keypoints.nose_tip(),
      keypoints.mouth_left(),
      keypoints.mouth_right(),
    ],
    "FaceKeypoints::points is the canonical alignment order — left eye, right eye, nose tip, \
     left mouth corner, right mouth corner"
  );
}

/// One named landmark region, at the interior and both unit-square
/// edges.
pub fn assert_face_landmark_region_accepts<R: FaceLandmarkRegion>() {
  let points = [(0.0_f32, 0.0_f32), (1.0, 1.0), (0.42, 0.13)];
  assert!(
    R::try_new("allPoints", &points).is_ok(),
    "FaceLandmarkRegion::try_new refused in-range landmark points"
  );
}

/// A face carrying its landmark regions.
pub fn assert_face_landmarks_accept<L: FaceLandmarksDetection>() {
  assert_face_landmark_region_accepts::<L::Region>();
  let points = [(0.0_f32, 0.0_f32), (1.0, 1.0), (0.42, 0.13)];
  let Ok(region) = L::Region::try_new("allPoints", &points) else {
    panic!("FaceLandmarkRegion::try_new refused in-range landmark points");
  };
  assert!(
    L::try_new(interior_bbox::<L::BoundingBox>(), 0.0, vec![region]).is_ok(),
    "FaceLandmarksDetection::try_new must accept a zero-confidence landmark set"
  );
}

/// One 2-D pose over its own joint seat. `seat` names the entry point
/// — [`BodyPoser`](crate::BodyPoser) and
/// [`AnimalPoser`](crate::AnimalPoser) build the same trait over
/// different joint rosters, and a failure has to say which refused.
pub fn assert_body_pose_accepts<P: BodyPoseDetection>(seat: &str) {
  let joint = joint_2d::<P::Joint>(seat);
  assert_eq!(
    joint.name(),
    "neck",
    "the joint's name must read back what it was built from — the engine sorts on it \
     (seat: {seat})"
  );
  assert!(
    P::try_new(interior_bbox::<P::BoundingBox>(), 0.0, vec![joint]).is_ok(),
    "BodyPoseDetection::try_new must accept a zero-confidence pose (seat: {seat})"
  );
}

/// A hand pose, over every chirality.
pub fn assert_hand_pose_accepts<P: HandPoseDetection>() {
  let joint = joint_2d::<P::Joint>("HandPoser");
  assert_eq!(
    joint.name(),
    "neck",
    "the hand joint's name must read back what it was built from — the engine sorts on it"
  );
  for chirality in [Chirality::Unknown, Chirality::Left, Chirality::Right] {
    assert!(
      P::try_new(
        interior_bbox::<P::BoundingBox>(),
        0.0,
        chirality,
        vec![joint_2d::<P::Joint>("HandPoser")],
      )
      .is_ok(),
      "HandPoseDetection::try_new must accept every chirality"
    );
  }
}

/// A 3-D pose, including metre-scale coordinates, the absent per-joint
/// confidence, and the coupled `(0.0, Unknown)` height fallback.
pub fn assert_body_pose_3d_accepts<P: BodyPose3DDetection>() {
  // `None` is what the engine passes for EVERY 3-D joint it emits:
  // Apple's 3-D point hierarchy declares no confidence. A vocabulary
  // that refuses it can never receive a 3-D pose at all, so this is
  // the load-bearing case, asserted first.
  let Ok(joint_3d) = P::Joint::try_new("root", -1.5, 2.5, 0.75, None) else {
    panic!(
      "BodyPose3DJoint::try_new refused an absent confidence — the engine passes None for every \
       3-D joint, because Apple's 3-D points carry no confidence"
    );
  };
  assert_eq!(
    joint_3d.name(),
    "root",
    "the 3-D joint's name must read back what it was built from — the engine sorts on it"
  );
  assert!(
    P::try_new(0.5, 1.75, HeightEstimation::Measured, vec![joint_3d]).is_ok(),
    "BodyPose3DDetection::try_new must accept a measured height in metres"
  );
  // A present confidence is still part of the signature, so an
  // implementor must handle it: the seat exists for a revision that
  // reports one, and refusing `Some` would break silently on that day.
  let Ok(joint_3d) = P::Joint::try_new("root", -1.5, 2.5, 0.75, Some(0.5)) else {
    panic!("BodyPose3DJoint::try_new refused model-space metres — 3-D joints are not normalized");
  };
  assert!(
    P::try_new(0.5, 1.75, HeightEstimation::Measured, vec![joint_3d]).is_ok(),
    "BodyPose3DDetection::try_new must accept a joint carrying a confidence"
  );
  let Ok(joint_3d) = P::Joint::try_new("root", 0.0, 0.0, 0.0, None) else {
    panic!("BodyPose3DJoint::try_new refused a zeroed joint");
  };
  assert!(
    P::try_new(0.0, 0.0, HeightEstimation::Unknown, vec![joint_3d]).is_ok(),
    "BodyPose3DDetection::try_new must accept the coupled (0.0, Unknown) height fallback"
  );
}

/// The instance-mask shape, at the one-byte-per-pixel payload the
/// engine always emits — note the argument order differs from
/// [`assert_person_segmentation_accepts`] by `instance_index` alone.
pub fn assert_person_instance_mask_accepts<M: PersonInstanceMaskDetection>() {
  assert!(
    M::try_new(interior_bbox::<M::BoundingBox>(), 0.0, 0, 2, 2, MASK_2X2).is_ok(),
    "PersonInstanceMaskDetection::try_new must accept instance index 0 and a width*height payload"
  );
}

/// The whole-frame mask shape.
pub fn assert_person_segmentation_accepts<M: PersonSegmentationMask>() {
  assert!(
    M::try_new(interior_bbox::<M::BoundingBox>(), 0.0, 2, 2, MASK_2X2).is_ok(),
    "PersonSegmentationMask::try_new must accept a width*height payload"
  );
}

// ----- the refusal family ---------------------------------------------------

/// Asserts the optional refusal family for the core bundle:
/// non-finite and out-of-domain inputs are rejected.
///
/// Not required by the engine — it filters these before construction —
/// but a validating vocabulary should pass, and a vocabulary that
/// stores raw floats will not. Run it only if yours validates.
///
/// Panics on the first violation.
pub fn assert_refuses_invalid<D: Detections>() {
  assert_bounding_box_refusals::<D::BoundingBox>();
  assert_confidence_refusals::<D>();
  assert_document_winding_refusal::<D>();
}

/// A box must stay finite, inside the unit square, and non-degenerate.
pub fn assert_bounding_box_refusals<B: BoundingBox>() {
  assert!(
    B::try_new(f32::NAN, 0.0, 0.5, 0.5).is_err(),
    "BoundingBox::try_new must refuse a non-finite component"
  );
  assert!(
    B::try_new(0.0, 0.0, 0.0, 0.5).is_err(),
    "BoundingBox::try_new must refuse a zero-extent box"
  );
  assert!(
    B::try_new(0.5, 0.0, 0.9, 0.5).is_err(),
    "BoundingBox::try_new must refuse a box that extends past the frame edge"
  );
}

/// Confidences must be finite and inside `0.0..=1.0` wherever they
/// appear in the core bundle.
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
      D::HorizonInfo::try_new(0.0, confidence).is_err(),
      "HorizonInfo::try_new must refuse a confidence outside a finite 0.0..=1.0 — note the \
       confidence is the SECOND argument"
    );
  }
}

/// A text run's confidence must be finite and inside `0.0..=1.0`.
pub fn assert_text_refusals<T: TextDetection>() {
  for confidence in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
    assert!(
      T::try_new("a", confidence, interior_bbox::<T::BoundingBox>(), 0, 0).is_err(),
      "TextDetection::try_new must refuse a confidence outside a finite 0.0..=1.0"
    );
  }
}

/// One 2-D joint seat's coordinate refusals, named by the entry point
/// it came from. Normalized coordinates must be finite and inside
/// `0.0..=1.0`; 3-D joints are exempt, being model-space metres.
///
/// Call it once per joint seat your vocabulary names: a validating
/// vocabulary that guards only one of them leaves the others open.
pub fn assert_joint_2d_refusals<J: BodyPoseJoint>(seat: &str) {
  assert!(
    J::try_new("neck", f32::NAN, 0.5, 0.5).is_err(),
    "BodyPoseJoint::try_new must refuse a non-finite coordinate (seat: {seat})"
  );
  assert!(
    J::try_new("neck", 1.5, 0.5, 0.5).is_err(),
    "BodyPoseJoint::try_new must refuse a coordinate outside 0.0..=1.0 (seat: {seat})"
  );
}

/// Landmark points must be finite.
pub fn assert_face_landmark_region_refusals<R: FaceLandmarkRegion>() {
  assert!(
    R::try_new("nose", &[(0.5, 0.5), (f32::NAN, 0.5)]).is_err(),
    "FaceLandmarkRegion::try_new must refuse a non-finite point"
  );
}

/// Instance-mask payloads must be non-empty.
pub fn assert_person_instance_mask_refusals<M: PersonInstanceMaskDetection>() {
  assert!(
    M::try_new(interior_bbox::<M::BoundingBox>(), 0.5, 0, 2, 2, &[]).is_err(),
    "PersonInstanceMaskDetection::try_new must refuse an empty payload"
  );
}

/// Mask dimensions must be non-degenerate.
pub fn assert_person_segmentation_refusals<M: PersonSegmentationMask>() {
  assert!(
    M::try_new(interior_bbox::<M::BoundingBox>(), 0.5, 0, 2, MASK_2X2).is_err(),
    "PersonSegmentationMask::try_new must refuse a zero-width mask"
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
