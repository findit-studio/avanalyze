//! The core output contract: what the one-pass analyzer is allowed to
//! build.
//!
//! [`VisionAnalyzer`](crate::VisionAnalyzer) never names a concrete
//! detection type. It is generic over a [`Detections`] bundle — one
//! marker type that names the eight cheap seats it fills, each of which
//! implements the matching per-part trait in this module. The engine
//! calls the constructors, the caller owns the vocabulary.
//!
//! Every other capability — text, barcodes, faces, landmarks, poses,
//! masks — is its own entry point with its own single-trait generic
//! parameter, and its trait lives beside that entry point rather than
//! here. A consumer that wants one capability names one type.
//!
//! Every seat on every constructor is a primitive (`f32` / `u32` /
//! `usize` / `&str` / `&[u8]` / tuples of those) or an engine-owned
//! type ([`Chirality`](crate::Chirality),
//! [`HeightEstimation`](crate::HeightEstimation),
//! [`FaceKeypoints`](crate::FaceKeypoints)). No foreign type crosses
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

/// A normalized, axis-aligned bounding box.
///
/// The engine intersects every Vision rectangle with the unit square
/// before it arrives here, so `x + width <= 1.0` and
/// `y + height <= 1.0` hold and both extents are strictly positive.
///
/// The read-back accessors are not decoration: they are how every
/// consumer reads a detection's geometry back out, so an
/// implementation whose accessors disagree with the values it was
/// constructed from silently reports a box the engine never produced.
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

/// The core output vocabulary, named by one marker type.
///
/// Implement this on a unit struct that ties together the types the
/// one-pass analyzer should build. The associated-type bounds force
/// the parts to agree — a subject and a saliency region must be built
/// from the *same* `BoundingBox` — so a mismatched bundle is a compile
/// error rather than a runtime surprise.
///
/// Seven associated types cover all eight
/// [`Analysis`](crate::Analysis) slots: the two subject slots share
/// one type, as do the two saliency slots. Nothing else is named here.
/// A consumer that only recognises text implements
/// [`TextDetection`](crate::TextDetection) on one type and never
/// touches this trait; a consumer that only wants faces implements
/// [`FaceDetection`](crate::FaceDetection) and its
/// [`BoundingBox`](crate::BoundingBox). That is the split: the bundle
/// is the price of the one-pass batch, not the price of admission.
pub trait Detections {
  /// Geometry shared by every boxed detection in the bundle.
  type BoundingBox: BoundingBox;
  /// Whole-frame image classifications, and the label half of a
  /// subject.
  type Detection: Detection;
  /// Human and animal subjects.
  type SubjectDetection: SubjectDetection<Detection = Self::Detection, BoundingBox = Self::BoundingBox>;
  /// Attention- and objectness-based salient regions.
  type SaliencyRegion: SaliencyRegion<BoundingBox = Self::BoundingBox>;
  /// The frame's horizon line.
  type HorizonInfo: HorizonInfo;
  /// Detected document quadrilaterals.
  type DocumentSegment: DocumentSegment;
  /// The frame's aesthetics score.
  type Aesthetics: Aesthetics;
}
