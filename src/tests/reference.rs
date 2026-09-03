//! The reference output vocabulary: `mediaschema`'s validated domain
//! aggregates, wired to this crate's contract.
//!
//! `mediaschema` is a **dev**-dependency — the engine does not link
//! it, and nothing in the public API names it. Wiring it here proves
//! two things the trait definitions alone cannot: that the primitive
//! seats are wide enough for a real, validating vocabulary built by
//! someone else, and — through
//! [`conformance`](crate::conformance) — that such a vocabulary agrees
//! with the conventions the engine relies on.
//!
//! Every `try_new` below forwards to the *inherent* constructor of the
//! mediaschema type; `rustc`'s `unconditional_recursion` lint is the
//! guard against that resolution silently flipping to the trait method.

use bytes::Bytes;
use mediaframe::frame::Dimensions;
use mediaschema::domain::aggregates::video::{self as ms, detections::DetectionError};

use crate::{
  Aesthetics, BarcodeDetection, BodyPose3DDetection, BodyPose3DJoint, BodyPoseDetection,
  BodyPoseJoint, BoundingBox, Chirality, Detection, Detections, DocumentSegment, FaceDetection,
  FaceKeypoints, FaceLandmarkRegion, FaceLandmarksDetection, HandPoseDetection, HeightEstimation,
  HorizonInfo, PersonInstanceMaskDetection, PersonSegmentationMask, SaliencyRegion,
  SubjectDetection, TextDetection,
};

/// mediaschema's video aggregates, named as one output vocabulary.
pub(crate) struct MediaSchema;

impl BoundingBox for ms::BoundingBox {
  type Error = DetectionError;

  fn try_new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, Self::Error> {
    ms::BoundingBox::try_new(x, y, width, height)
  }

  fn x(&self) -> f32 {
    ms::BoundingBox::x(self)
  }

  fn y(&self) -> f32 {
    ms::BoundingBox::y(self)
  }

  fn width(&self) -> f32 {
    ms::BoundingBox::width(self)
  }

  fn height(&self) -> f32 {
    ms::BoundingBox::height(self)
  }
}

impl Detection for ms::Detection {
  type Error = DetectionError;

  fn try_new(label: &str, confidence: f32) -> Result<Self, Self::Error> {
    ms::Detection::try_new(label, confidence)
  }
}

impl SubjectDetection for ms::SubjectDetection {
  type Detection = ms::Detection;
  type BoundingBox = ms::BoundingBox;

  fn new(detection: Self::Detection, bbox: Self::BoundingBox) -> Self {
    ms::SubjectDetection::new(detection, bbox)
  }
}

/// `mediaschema` 0.2.1 has no seat for pose-angle OR capture-quality
/// absence — its own `FaceDetection::try_new` takes plain `f32` for
/// both — and none at all for the five-point reduction. Collapsing
/// `None` to `0.0` and dropping the keypoints here is this ONE
/// reference vocabulary's limitation, not a reopening of the loss
/// `FaceDetection` exists to close: the engine hands this impl real
/// `Option`s, and a vocabulary that CAN represent the absence
/// (`tests/common::Face`) receives them intact.
impl FaceDetection for ms::FaceDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    capture_quality: Option<f32>,
    roll: Option<f32>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    keypoints: Option<FaceKeypoints>,
  ) -> Result<Self, Self::Error> {
    let _ = keypoints;
    ms::FaceDetection::try_new(
      bbox,
      confidence,
      capture_quality.unwrap_or(0.0),
      roll.unwrap_or(0.0),
      yaw.unwrap_or(0.0),
      pitch.unwrap_or(0.0),
    )
  }
}

impl FaceLandmarkRegion for ms::FaceLandmarkRegion {
  type Error = DetectionError;

  fn try_new(name: &str, points: &[(f32, f32)]) -> Result<Self, Self::Error> {
    ms::FaceLandmarkRegion::try_new(name, points.iter().copied())
  }
}

impl FaceLandmarksDetection for ms::FaceLandmarksDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;
  type Region = ms::FaceLandmarkRegion;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    regions: Vec<Self::Region>,
  ) -> Result<Self, Self::Error> {
    ms::FaceLandmarksDetection::try_new(bbox, confidence, regions)
  }
}

impl BodyPoseJoint for ms::BodyPoseJoint {
  type Error = DetectionError;

  fn try_new(name: &str, x: f32, y: f32, confidence: f32) -> Result<Self, Self::Error> {
    ms::BodyPoseJoint::try_new(name, x, y, confidence)
  }

  fn name(&self) -> &str {
    ms::BodyPoseJoint::name(self)
  }
}

impl BodyPoseDetection for ms::BodyPoseDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;
  type Joint = ms::BodyPoseJoint;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    ms::BodyPoseDetection::try_new(bbox, confidence, joints)
  }
}

impl BodyPose3DJoint for ms::BodyPose3DJoint {
  type Error = DetectionError;

  fn try_new(
    name: &str,
    x: f32,
    y: f32,
    z: f32,
    confidence: Option<f32>,
  ) -> Result<Self, Self::Error> {
    // mediaschema's 3-D joint demands an `f32`, so the collapse happens
    // here — in the adapter, one visible line — exactly as this file
    // already collapses `capture_quality`, `roll`, `yaw` and `pitch`.
    // The engine never invents the number; a vocabulary that insists on
    // one chooses its own stand-in.
    ms::BodyPose3DJoint::try_new(name, x, y, z, confidence.unwrap_or(0.0))
  }

  fn name(&self) -> &str {
    ms::BodyPose3DJoint::name(self)
  }
}

impl BodyPose3DDetection for ms::BodyPose3DDetection {
  type Error = DetectionError;
  type Joint = ms::BodyPose3DJoint;

  fn try_new(
    confidence: f32,
    body_height: f32,
    height_estimation: HeightEstimation,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    let height_estimation = match height_estimation {
      HeightEstimation::Unknown => ms::BodyPose3DHeightEstimation::Unknown,
      HeightEstimation::Reference => ms::BodyPose3DHeightEstimation::Reference,
      HeightEstimation::Measured => ms::BodyPose3DHeightEstimation::Measured,
    };
    ms::BodyPose3DDetection::try_new(confidence, body_height, height_estimation, joints)
  }
}

impl HandPoseDetection for ms::HandPoseDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;
  type Joint = ms::BodyPoseJoint;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    chirality: Chirality,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    let chirality = match chirality {
      Chirality::Unknown => ms::HandChirality::Unknown,
      Chirality::Left => ms::HandChirality::Left,
      Chirality::Right => ms::HandChirality::Right,
    };
    ms::HandPoseDetection::try_new(bbox, confidence, chirality, joints)
  }
}

impl PersonInstanceMaskDetection for ms::PersonInstanceMaskDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    instance_index: u32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error> {
    ms::PersonInstanceMaskDetection::try_new(
      bbox,
      confidence,
      instance_index,
      Dimensions::new(width, height),
      Bytes::copy_from_slice(data),
    )
  }
}

impl PersonSegmentationMask for ms::PersonSegmentationMask {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error> {
    ms::PersonSegmentationMask::try_new(
      bbox,
      confidence,
      Dimensions::new(width, height),
      Bytes::copy_from_slice(data),
    )
  }
}

/// `mediaschema` 0.2.1 has no seat for the candidate provenance pair
/// either — same pinned-dependency boundary as the face `Option`s.
/// Dropping `observation` / `rank` here costs this reference
/// vocabulary the ability to tell two readings of one text region
/// apart; `tests/common::Text` keeps both.
impl TextDetection for ms::TextDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;

  fn try_new(
    text: &str,
    confidence: f32,
    bbox: Self::BoundingBox,
    observation: usize,
    rank: usize,
  ) -> Result<Self, Self::Error> {
    let _ = (observation, rank);
    ms::TextDetection::try_new(text, confidence, bbox)
  }
}

impl BarcodeDetection for ms::BarcodeDetection {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;

  fn try_new(
    payload: &str,
    symbology: &str,
    confidence: f32,
    bbox: Self::BoundingBox,
  ) -> Result<Self, Self::Error> {
    ms::BarcodeDetection::try_new(payload, symbology, confidence, bbox)
  }
}

impl SaliencyRegion for ms::SaliencyRegion {
  type Error = DetectionError;
  type BoundingBox = ms::BoundingBox;

  fn try_new(bbox: Self::BoundingBox, confidence: f32) -> Result<Self, Self::Error> {
    ms::SaliencyRegion::try_new(bbox, confidence)
  }
}

impl HorizonInfo for ms::HorizonInfo {
  type Error = DetectionError;

  fn try_new(angle: f32, confidence: f32) -> Result<Self, Self::Error> {
    ms::HorizonInfo::try_new(angle, confidence)
  }
}

impl DocumentSegment for ms::DocumentSegment {
  type Error = DetectionError;

  fn try_new(
    top_left: (f32, f32),
    top_right: (f32, f32),
    bottom_right: (f32, f32),
    bottom_left: (f32, f32),
    confidence: f32,
  ) -> Result<Self, Self::Error> {
    ms::DocumentSegment::try_new(top_left, top_right, bottom_right, bottom_left, confidence)
  }
}

impl Aesthetics for ms::Aesthetics {
  fn new(overall_score: f32, is_utility: bool) -> Self {
    ms::Aesthetics::new(overall_score, is_utility)
  }
}

/// The core bundle names only what the one-pass analyzer builds. The
/// per-entry impls above stand on their own and are exercised by the
/// per-entry conformance assertions, not through this trait.
impl Detections for MediaSchema {
  type BoundingBox = ms::BoundingBox;
  type Detection = ms::Detection;
  type SubjectDetection = ms::SubjectDetection;
  type SaliencyRegion = ms::SaliencyRegion;
  type HorizonInfo = ms::HorizonInfo;
  type DocumentSegment = ms::DocumentSegment;
  type Aesthetics = ms::Aesthetics;
}
