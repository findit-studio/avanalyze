//! The output contract: what the engine is allowed to build.
//!
//! [`VisionAnalyzer`](crate::VisionAnalyzer) never names a concrete
//! detection type. It is generic over a [`Detections`] bundle — one
//! marker type that names a full set of output types, each of which
//! implements the matching per-part trait in this module. The engine
//! calls the constructors, the caller owns the vocabulary.
//!
//! Every seat on every constructor is a primitive (`f32` / `u32` /
//! `usize` / `&str` / `&[u8]` / tuples of those) or an engine-owned
//! enum ([`Chirality`], [`HeightEstimation`]). No foreign type crosses
//! the seam, so an implementor picks their own storage, their own
//! validation, and their own error type.
//!
//! # Conventions every implementor inherits
//!
//! - **Coordinates** are normalized to `0.0..=1.0` with a **top-left**
//!   origin (`y` grows down). Apple's Vision framework is lower-left;
//!   the engine performs the flip before it reaches this seam.
//! - **Confidences** are finite and in `0.0..=1.0`. The engine refuses
//!   anything else before construction, so an implementation never has
//!   to defend against `NaN` here — but a validating one may.
//! - **3-D joint coordinates** and **body heights** are model-space
//!   **metres**, not normalized: they are neither flipped nor clamped
//!   nor range-checked by the engine.
//! - **Argument order is load-bearing** where it is unusual, and the
//!   unusual cases are called out on each method. Getting one wrong is
//!   silent: the values still type-check.
//!
//! [`conformance`](crate::conformance) turns those conventions into
//! runnable assertions.

/// Handedness of a detected hand pose.
///
/// Apple reports handedness as `VNChirality`; anything the framework
/// does not classify as left or right — including variants added by a
/// future OS — maps to [`Chirality::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Chirality {
  /// Handedness was not reported, or was reported as a value this
  /// engine does not recognise.
  #[default]
  Unknown,
  /// Left hand.
  Left,
  /// Right hand.
  Right,
}

/// How a 3-D body pose's height estimate was obtained.
///
/// Paired with the body height in metres on
/// [`BodyPose3DDetection::try_new`]. The pair is coupled: when the
/// height reading is not finite the engine emits
/// `(0.0, HeightEstimation::Unknown)` rather than a `0.0` metre
/// subject with a `Measured` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeightEstimation {
  /// No usable estimate.
  #[default]
  Unknown,
  /// Derived from a reference height rather than measured.
  Reference,
  /// Measured from the observation.
  Measured,
}

/// A normalized, axis-aligned bounding box.
///
/// The engine intersects every Vision rectangle with the unit square
/// before it arrives here, so `x + width <= 1.0` and
/// `y + height <= 1.0` hold and both extents are strictly positive.
///
/// The read-back accessors are not decoration: the engine joins the
/// two Vision face passes by intersection-over-union computed from
/// them, so an implementation whose accessors disagree with the values
/// it was constructed from silently changes which faces get a capture
/// quality.
pub trait BoundingBox: Sized {
  /// Why a box was refused.
  type Error;

  /// Builds a box from its top-left corner and extents, all
  /// normalized to `0.0..=1.0`.
  fn try_new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, Self::Error>;

  /// Left edge, as constructed.
  fn x(&self) -> f32;

  /// Top edge, as constructed.
  fn y(&self) -> f32;

  /// Width, as constructed.
  fn width(&self) -> f32;

  /// Height, as constructed.
  fn height(&self) -> f32;
}

/// A labelled, scored classification with no geometry of its own.
///
/// Used both for whole-frame image classification and as the label
/// half of a [`SubjectDetection`].
pub trait Detection: Sized {
  /// Why a detection was refused.
  type Error;

  /// Builds a detection from a non-empty label and a confidence.
  ///
  /// The engine lowercases and trims image-classification labels but
  /// passes animal labels through verbatim, and uses the literal
  /// `"person"` for human subjects — the label vocabulary is Apple's,
  /// not this crate's.
  fn try_new(label: &str, confidence: f32) -> Result<Self, Self::Error>;
}

/// A classification bound to a bounding box: the shape used for both
/// human and animal subjects.
///
/// Infallible — both halves have already been validated by their own
/// constructors.
pub trait SubjectDetection: Sized {
  /// The label half.
  type Detection: Detection;
  /// The geometry half.
  type BoundingBox: BoundingBox;

  /// Pairs a label with its box.
  fn new(detection: Self::Detection, bbox: Self::BoundingBox) -> Self;
}

/// One detected face.
///
/// `capture_quality` does **not** come from the same Vision
/// observation as `bbox`: the engine runs a separate capture-quality
/// pass and joins it to the detection spine by box overlap.
/// `capture_quality` is an `Option<f32>` representing the three states
/// that join can produce: `Some(q)` — Vision measured this face's
/// capture quality and reported `q`; `Some(0.0)` — Vision measured it
/// and found it terrible, a real zero reading; `None` — Vision never
/// measured this face's capture quality at all, whether because the
/// raw reading was nil or because no capture-quality observation
/// overlapped this face's box (a join-miss). `Some(0.0)` and `None`
/// are not interchangeable: collapsing `None` to `Some(0.0)` tells
/// every "quality below X" query that a face the quality pass never
/// covered was scored at the worst possible value.
///
/// `roll` / `yaw` / `pitch` are each an `Option<f32>` in radians.
/// Vision estimates the three independently, and which it reports
/// varies by OS version and by detection path — a face may arrive
/// with all three, some, or none. `Some(0.0)` is a head Vision
/// measured and found level; `None` is an angle Vision never
/// computed. The two are not interchangeable: collapsing `None` to
/// `0.0` tells every "pitched down more than 20°" query that a face
/// Vision never looked at was measured perfectly upright.
pub trait FaceDetection: Sized {
  /// Why a face was refused.
  type Error;
  /// The geometry type this face is built from.
  type BoundingBox: BoundingBox;

  /// Builds a face detection. `capture_quality` is `None` where
  /// Vision never measured this face's capture quality; `roll` /
  /// `yaw` / `pitch` are `None` where Vision did not compute that
  /// angle.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    capture_quality: Option<f32>,
    roll: Option<f32>,
    yaw: Option<f32>,
    pitch: Option<f32>,
  ) -> Result<Self, Self::Error>;
}

/// One named group of face-landmark points.
///
/// The engine emits up to thirteen regions per face, named with
/// Apple's own identifiers (`allPoints`, `faceContour`, `leftEye`, …).
/// Points arrive already projected out of face-relative space into
/// image-normalized, top-left coordinates; a region with no surviving
/// point is never constructed.
pub trait FaceLandmarkRegion: Sized {
  /// Why a region was refused.
  type Error;

  /// Builds a region from its name and its non-empty point list.
  fn try_new(name: &str, points: &[(f32, f32)]) -> Result<Self, Self::Error>;
}

/// A face carrying its landmark regions.
///
/// The `confidence` is the landmark set's own score, not the parent
/// face observation's, and the region count has already been checked
/// against the configured minimum.
pub trait FaceLandmarksDetection: Sized {
  /// Why a landmark detection was refused.
  type Error;
  /// The geometry type this detection is built from.
  type BoundingBox: BoundingBox;
  /// The region type this detection collects.
  type Region: FaceLandmarkRegion;

  /// Builds a landmark detection.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    regions: Vec<Self::Region>,
  ) -> Result<Self, Self::Error>;
}

/// One 2-D pose joint — the shape body, hand, and animal joints share.
///
/// The shape is shared; the vocabularies are not. Apple names a
/// different joint set per skeleton, so the bundle seats them
/// separately — [`Detections::BodyJoint`], [`Detections::HandJoint`],
/// [`Detections::AnimalJoint`]. A vocabulary that keeps one joint type
/// for all three is still legal: it names that type three times.
///
/// The name is Apple's joint identifier, verbatim. The engine sorts
/// joints by this name before building the enclosing pose, because
/// Apple's dictionary iteration order is unspecified; an
/// implementation whose [`name`](BodyPoseJoint::name) does not return
/// what it was constructed with makes pose output non-reproducible.
pub trait BodyPoseJoint: Sized {
  /// Why a joint was refused.
  type Error;

  /// Builds a joint at a normalized, top-left-origin position.
  fn try_new(name: &str, x: f32, y: f32, confidence: f32) -> Result<Self, Self::Error>;

  /// Joint name, as constructed.
  fn name(&self) -> &str;
}

/// A 2-D pose — the shape human bodies and animal bodies share.
///
/// The two are separate bundle seats
/// ([`Detections::BodyPoseDetection`] and
/// [`Detections::AnimalPoseDetection`]) because they collect different
/// joints; the trait is one because the payload is.
///
/// Vision does not report a box for a pose. The engine synthesises one
/// from the bounds of the joints that survived filtering, so the box
/// describes the surviving joints and nothing more; a pose whose
/// joints are collinear has no box and is dropped before it reaches
/// this seam.
pub trait BodyPoseDetection: Sized {
  /// Why a pose was refused.
  type Error;
  /// The geometry type this pose is built from.
  type BoundingBox: BoundingBox;
  /// The joint type this pose collects.
  type Joint: BodyPoseJoint;

  /// Builds a pose from its synthesised box and its non-empty,
  /// name-sorted joint list.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error>;
}

/// One 3-D body-pose joint.
///
/// A vocabulary of its own, seated as [`Detections::Body3Joint`]: the
/// 3-D skeleton is neither the 2-D body's joint set nor its coordinate
/// system.
///
/// `x` / `y` / `z` are model-space **metres**, not normalized
/// coordinates: no flip, no clamp, no `0.0..=1.0` invariant. Only
/// finiteness is enforced upstream.
pub trait BodyPose3DJoint: Sized {
  /// Why a joint was refused.
  type Error;

  /// Builds a 3-D joint at a model-space position in metres.
  fn try_new(name: &str, x: f32, y: f32, z: f32, confidence: f32) -> Result<Self, Self::Error>;

  /// Joint name, as constructed.
  fn name(&self) -> &str;
}

/// A 3-D body pose.
///
/// Alone among the pose types this one carries **no bounding box** —
/// Vision reports 3-D poses in model space, which has no projection
/// back to the frame.
pub trait BodyPose3DDetection: Sized {
  /// Why a pose was refused.
  type Error;
  /// The joint type this pose collects.
  type Joint: BodyPose3DJoint;

  /// Builds a 3-D pose.
  ///
  /// `body_height` is metres, and is coupled to `height_estimation`:
  /// the engine only pairs a non-`Unknown` estimation with a finite
  /// height.
  fn try_new(
    confidence: f32,
    body_height: f32,
    height_estimation: HeightEstimation,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error>;
}

/// A hand pose.
///
/// Built from [`BodyPoseJoint`]s of its own kind — Apple's hand joint
/// set is a different vocabulary from the body's, which is why the
/// bundle seats it as [`Detections::HandJoint`] — and, like
/// [`BodyPoseDetection`], carries a box synthesised from the surviving
/// joints.
pub trait HandPoseDetection: Sized {
  /// Why a pose was refused.
  type Error;
  /// The geometry type this pose is built from.
  type BoundingBox: BoundingBox;
  /// The joint type this pose collects.
  type Joint: BodyPoseJoint;

  /// Builds a hand pose.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    chirality: Chirality,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error>;
}

/// One person-instance segmentation mask.
///
/// `data` is **always** one byte per pixel, `width * height` bytes
/// long, row-major and top-down, regardless of which pixel format
/// Vision produced. `width` / `height` describe the mask buffer, which
/// need not match the source frame's dimensions.
///
/// Note the argument order against [`PersonSegmentationMask::try_new`]:
/// the two differ only by `instance_index` in the middle.
pub trait PersonInstanceMaskDetection: Sized {
  /// Why a mask was refused.
  type Error;
  /// The geometry type this mask is built from.
  type BoundingBox: BoundingBox;

  /// Builds an instance mask. `bbox` is the normalized bounding box of
  /// the mask's foreground pixels.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    instance_index: u32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error>;
}

/// One whole-frame person segmentation mask.
///
/// Same payload contract as [`PersonInstanceMaskDetection`], without
/// an instance index.
pub trait PersonSegmentationMask: Sized {
  /// Why a mask was refused.
  type Error;
  /// The geometry type this mask is built from.
  type BoundingBox: BoundingBox;

  /// Builds a segmentation mask. `bbox` is the normalized bounding box
  /// of the mask's foreground pixels.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error>;
}

/// One recognised text run.
///
/// The box comes **last** here, unlike every other boxed detection.
/// One Vision observation can yield several candidate strings, each of
/// which re-uses the observation's box.
pub trait TextDetection: Sized {
  /// Why a text detection was refused.
  type Error;
  /// The geometry type this detection is built from.
  type BoundingBox: BoundingBox;

  /// Builds a text detection.
  fn try_new(text: &str, confidence: f32, bbox: Self::BoundingBox) -> Result<Self, Self::Error>;
}

/// One decoded barcode.
///
/// `symbology` is Apple's raw `VNBarcodeSymbology` string, not a typed
/// vocabulary. The box comes last, as it does for
/// [`TextDetection`].
pub trait BarcodeDetection: Sized {
  /// Why a barcode was refused.
  type Error;
  /// The geometry type this detection is built from.
  type BoundingBox: BoundingBox;

  /// Builds a barcode detection.
  fn try_new(
    payload: &str,
    symbology: &str,
    confidence: f32,
    bbox: Self::BoundingBox,
  ) -> Result<Self, Self::Error>;
}

/// One salient region.
///
/// The same type serves the attention-based and objectness-based
/// passes; which pass produced a region survives only in the
/// [`Analysis`](crate::Analysis) slot it lands in.
pub trait SaliencyRegion: Sized {
  /// Why a region was refused.
  type Error;
  /// The geometry type this region is built from.
  type BoundingBox: BoundingBox;

  /// Builds a saliency region.
  fn try_new(bbox: Self::BoundingBox, confidence: f32) -> Result<Self, Self::Error>;
}

/// The frame's horizon line.
///
/// Note the argument order: **`angle` first**, confidence second —
/// the opposite of every other scored type here.
///
/// `angle` is radians and unbounded in sign. `try_new(0.0, 0.0)` is
/// the engine's "nothing detected" sentinel and a conforming
/// implementation accepts it; see
/// [`conformance`](crate::conformance).
pub trait HorizonInfo: Sized {
  /// Why a horizon was refused.
  type Error;

  /// Builds a horizon reading from its angle in radians and its
  /// confidence.
  fn try_new(angle: f32, confidence: f32) -> Result<Self, Self::Error>;
}

/// One detected document quadrilateral.
///
/// The corners are passed **top-left, top-right, bottom-right,
/// bottom-left** — perimeter winding order, not raster order. An
/// implementation that reads them as `(TL, TR, BL, BR)` builds a
/// bow-tie out of every real document; the conformance suite pins this
/// by feeding a well-formed quad in winding order.
pub trait DocumentSegment: Sized {
  /// Why a segment was refused.
  type Error;

  /// Builds a document quad from its four normalized corners in
  /// winding order.
  fn try_new(
    top_left: (f32, f32),
    top_right: (f32, f32),
    bottom_right: (f32, f32),
    bottom_left: (f32, f32),
    confidence: f32,
  ) -> Result<Self, Self::Error>;
}

/// The frame's aesthetics score.
///
/// Infallible: Apple's `overall_score` is a **signed** quantity with
/// no `0.0..=1.0` invariant to check. `new(0.0, false)` is the
/// engine's "nothing detected" sentinel.
pub trait Aesthetics: Sized {
  /// Builds an aesthetics reading.
  fn new(overall_score: f32, is_utility: bool) -> Self;
}

/// The full output vocabulary, named by one marker type.
///
/// Implement this on a unit struct that ties together the types you
/// want the engine to build. The associated-type bounds force the
/// parts to agree — a face and a saliency region must be built from
/// the *same* `BoundingBox`, and every pose is built from *its own
/// skeleton's* joint — so a mismatched bundle is a compile error
/// rather than a runtime surprise.
///
/// The joint guarantee is per skeleton family, not across families.
/// Apple's four joint rosters are four different sets, so a body joint
/// and a hand joint are not required to be the same type: body poses
/// collect [`BodyJoint`](Detections::BodyJoint), hand poses
/// [`HandJoint`](Detections::HandJoint), animal poses
/// [`AnimalJoint`](Detections::AnimalJoint), 3-D poses
/// [`Body3Joint`](Detections::Body3Joint). A vocabulary with one joint
/// type for every skeleton names it in every seat and loses nothing.
pub trait Detections {
  /// Geometry shared by every boxed detection.
  type BoundingBox: BoundingBox;
  /// Whole-frame image classifications, and the label half of a
  /// subject.
  type Detection: Detection;
  /// Human and animal subjects.
  type SubjectDetection: SubjectDetection<Detection = Self::Detection, BoundingBox = Self::BoundingBox>;
  /// Detected faces.
  type FaceDetection: FaceDetection<BoundingBox = Self::BoundingBox>;
  /// One named group of landmark points.
  type FaceLandmarkRegion: FaceLandmarkRegion;
  /// A face and its landmark regions.
  type FaceLandmarksDetection: FaceLandmarksDetection<BoundingBox = Self::BoundingBox, Region = Self::FaceLandmarkRegion>;
  /// One joint of the human 2-D body skeleton.
  type BodyJoint: BodyPoseJoint;
  /// Human 2-D poses.
  type BodyPoseDetection: BodyPoseDetection<BoundingBox = Self::BoundingBox, Joint = Self::BodyJoint>;
  /// One joint of the hand skeleton.
  type HandJoint: BodyPoseJoint;
  /// Hand poses.
  type HandPoseDetection: HandPoseDetection<BoundingBox = Self::BoundingBox, Joint = Self::HandJoint>;
  /// One joint of the animal 2-D body skeleton.
  type AnimalJoint: BodyPoseJoint;
  /// Animal 2-D poses. Same trait as
  /// [`BodyPoseDetection`](Detections::BodyPoseDetection), its own
  /// seat: the joint rosters differ.
  type AnimalPoseDetection: BodyPoseDetection<BoundingBox = Self::BoundingBox, Joint = Self::AnimalJoint>;
  /// One joint of the 3-D body skeleton, in model-space metres.
  type Body3Joint: BodyPose3DJoint;
  /// 3-D body poses.
  type BodyPose3DDetection: BodyPose3DDetection<Joint = Self::Body3Joint>;
  /// Per-instance person masks.
  type PersonInstanceMaskDetection: PersonInstanceMaskDetection<BoundingBox = Self::BoundingBox>;
  /// Whole-frame person masks.
  type PersonSegmentationMask: PersonSegmentationMask<BoundingBox = Self::BoundingBox>;
  /// Recognised text runs.
  type TextDetection: TextDetection<BoundingBox = Self::BoundingBox>;
  /// Decoded barcodes.
  type BarcodeDetection: BarcodeDetection<BoundingBox = Self::BoundingBox>;
  /// Attention- and objectness-based salient regions.
  type SaliencyRegion: SaliencyRegion<BoundingBox = Self::BoundingBox>;
  /// The frame's horizon line.
  type HorizonInfo: HorizonInfo;
  /// Detected document quadrilaterals.
  type DocumentSegment: DocumentSegment;
  /// The frame's aesthetics score.
  type Aesthetics: Aesthetics;
}
