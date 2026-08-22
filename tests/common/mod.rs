//! A third-party output vocabulary, written the way a downstream crate
//! would write one.
//!
//! Nothing here adapts, wraps, or re-exports an avanalyze type: these
//! are plain local structs that implement avanalyze's traits directly
//! from an outside crate. That is the proof the contract is actually
//! open — the in-crate reference implementation cannot show it,
//! because the orphan rule lets a crate implement its own traits for
//! anyone's types.
//!
//! It is also deliberately **non-validating**: every constructor is
//! [`Infallible`] and stores what it was given. Such a vocabulary
//! satisfies `assert_contract` and is expected to fail
//! `assert_refuses_invalid`, which is exactly the split those two
//! families document.
//!
//! And it names **four distinct joint types** — [`BodyJoint`],
//! [`HandJoint`], [`AnimalJoint`], [`Body3Joint`]. Nothing forces that;
//! it is here so that the decoupling is compiled, not merely asserted.
//! A bundle whose skeletons carry genuinely different joint vocabularies
//! must type-check, and the reference implementation in
//! `src/tests/reference.rs` proves the other half — one joint type in
//! every seat is still legal.

// Each test binary that includes this module uses a different subset
// of the vocabulary; the unused remainder is not dead code, it is the
// rest of the contract.
#![allow(dead_code)]

use core::convert::Infallible;

use avanalyze::{
  Aesthetics, BarcodeDetection, BodyPose3DDetection, BodyPose3DJoint, BodyPoseDetection,
  BodyPoseJoint, BoundingBox, Chirality, Detection, Detections, DocumentSegment, FaceDetection,
  FaceLandmarkRegion, FaceLandmarksDetection, HandPoseDetection, HeightEstimation, HorizonInfo,
  PersonInstanceMaskDetection, PersonSegmentationMask, SaliencyRegion, SubjectDetection,
  TextDetection,
};

/// The bundle marker naming every type below.
#[derive(Debug, Clone, Copy)]
pub struct Plain;

#[derive(Debug, Clone, PartialEq)]
pub struct Bbox {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Label {
  pub label: String,
  pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Subject {
  pub detection: Label,
  pub bbox: Bbox,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Face {
  pub bbox: Bbox,
  pub confidence: f32,
  pub capture_quality: f32,
  pub roll: f32,
  pub yaw: f32,
  pub pitch: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkRegion {
  pub name: String,
  pub points: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceLandmarks {
  pub bbox: Bbox,
  pub confidence: f32,
  pub regions: Vec<LandmarkRegion>,
}

/// A joint of the human 2-D skeleton. Same shape as [`HandJoint`] and
/// [`AnimalJoint`], deliberately not the same type: the three rosters
/// name different joints, and a body joint has no business in a hand.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyJoint {
  pub name: String,
  pub x: f32,
  pub y: f32,
  pub confidence: f32,
}

/// A joint of the hand skeleton.
#[derive(Debug, Clone, PartialEq)]
pub struct HandJoint {
  pub name: String,
  pub x: f32,
  pub y: f32,
  pub confidence: f32,
}

/// A joint of the animal 2-D skeleton.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimalJoint {
  pub name: String,
  pub x: f32,
  pub y: f32,
  pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
  pub bbox: Bbox,
  pub confidence: f32,
  pub joints: Vec<BodyJoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnimalPose {
  pub bbox: Bbox,
  pub confidence: f32,
  pub joints: Vec<AnimalJoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandPose {
  pub bbox: Bbox,
  pub confidence: f32,
  pub chirality: Chirality,
  pub joints: Vec<HandJoint>,
}

/// A joint of the 3-D body skeleton, in model-space metres.
#[derive(Debug, Clone, PartialEq)]
pub struct Body3Joint {
  pub name: String,
  pub x: f32,
  pub y: f32,
  pub z: f32,
  pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pose3 {
  pub confidence: f32,
  pub body_height: f32,
  pub height_estimation: HeightEstimation,
  pub joints: Vec<Body3Joint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstanceMask {
  pub bbox: Bbox,
  pub confidence: f32,
  pub instance_index: u32,
  pub width: u32,
  pub height: u32,
  pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationMask {
  pub bbox: Bbox,
  pub confidence: f32,
  pub width: u32,
  pub height: u32,
  pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Text {
  pub text: String,
  pub confidence: f32,
  pub bbox: Bbox,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Barcode {
  pub payload: String,
  pub symbology: String,
  pub confidence: f32,
  pub bbox: Bbox,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Salient {
  pub bbox: Bbox,
  pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Horizon {
  pub angle: f32,
  pub confidence: f32,
}

/// Corners in the order the seat delivers them: top-left, top-right,
/// bottom-right, bottom-left.
#[derive(Debug, Clone, PartialEq)]
pub struct Quad {
  pub corners: [(f32, f32); 4],
  pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
  pub overall_score: f32,
  pub is_utility: bool,
}

impl BoundingBox for Bbox {
  type Error = Infallible;

  fn try_new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      x,
      y,
      width,
      height,
    })
  }

  fn x(&self) -> f32 {
    self.x
  }

  fn y(&self) -> f32 {
    self.y
  }

  fn width(&self) -> f32 {
    self.width
  }

  fn height(&self) -> f32 {
    self.height
  }
}

impl Detection for Label {
  type Error = Infallible;

  fn try_new(label: &str, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      label: label.to_owned(),
      confidence,
    })
  }
}

impl SubjectDetection for Subject {
  type Detection = Label;
  type BoundingBox = Bbox;

  fn new(detection: Self::Detection, bbox: Self::BoundingBox) -> Self {
    Self { detection, bbox }
  }
}

impl FaceDetection for Face {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    capture_quality: f32,
    roll: f32,
    yaw: f32,
    pitch: f32,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      capture_quality,
      roll,
      yaw,
      pitch,
    })
  }
}

impl FaceLandmarkRegion for LandmarkRegion {
  type Error = Infallible;

  fn try_new(name: &str, points: &[(f32, f32)]) -> Result<Self, Self::Error> {
    Ok(Self {
      name: name.to_owned(),
      points: points.to_vec(),
    })
  }
}

impl FaceLandmarksDetection for FaceLandmarks {
  type Error = Infallible;
  type BoundingBox = Bbox;
  type Region = LandmarkRegion;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    regions: Vec<Self::Region>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      regions,
    })
  }
}

impl BodyPoseJoint for BodyJoint {
  type Error = Infallible;

  fn try_new(name: &str, x: f32, y: f32, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      name: name.to_owned(),
      x,
      y,
      confidence,
    })
  }

  fn name(&self) -> &str {
    &self.name
  }
}

impl BodyPoseJoint for HandJoint {
  type Error = Infallible;

  fn try_new(name: &str, x: f32, y: f32, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      name: name.to_owned(),
      x,
      y,
      confidence,
    })
  }

  fn name(&self) -> &str {
    &self.name
  }
}

impl BodyPoseJoint for AnimalJoint {
  type Error = Infallible;

  fn try_new(name: &str, x: f32, y: f32, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      name: name.to_owned(),
      x,
      y,
      confidence,
    })
  }

  fn name(&self) -> &str {
    &self.name
  }
}

impl BodyPoseDetection for Pose {
  type Error = Infallible;
  type BoundingBox = Bbox;
  type Joint = BodyJoint;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      joints,
    })
  }
}

impl BodyPoseDetection for AnimalPose {
  type Error = Infallible;
  type BoundingBox = Bbox;
  type Joint = AnimalJoint;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      joints,
    })
  }
}

impl HandPoseDetection for HandPose {
  type Error = Infallible;
  type BoundingBox = Bbox;
  type Joint = HandJoint;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    chirality: Chirality,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      chirality,
      joints,
    })
  }
}

impl BodyPose3DJoint for Body3Joint {
  type Error = Infallible;

  fn try_new(name: &str, x: f32, y: f32, z: f32, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self {
      name: name.to_owned(),
      x,
      y,
      z,
      confidence,
    })
  }

  fn name(&self) -> &str {
    &self.name
  }
}

impl BodyPose3DDetection for Pose3 {
  type Error = Infallible;
  type Joint = Body3Joint;

  fn try_new(
    confidence: f32,
    body_height: f32,
    height_estimation: HeightEstimation,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      confidence,
      body_height,
      height_estimation,
      joints,
    })
  }
}

impl PersonInstanceMaskDetection for InstanceMask {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    instance_index: u32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      instance_index,
      width,
      height,
      data: data.to_vec(),
    })
  }
}

impl PersonSegmentationMask for SegmentationMask {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    width: u32,
    height: u32,
    data: &[u8],
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      bbox,
      confidence,
      width,
      height,
      data: data.to_vec(),
    })
  }
}

impl TextDetection for Text {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(text: &str, confidence: f32, bbox: Self::BoundingBox) -> Result<Self, Self::Error> {
    Ok(Self {
      text: text.to_owned(),
      confidence,
      bbox,
    })
  }
}

impl BarcodeDetection for Barcode {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(
    payload: &str,
    symbology: &str,
    confidence: f32,
    bbox: Self::BoundingBox,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      payload: payload.to_owned(),
      symbology: symbology.to_owned(),
      confidence,
      bbox,
    })
  }
}

impl SaliencyRegion for Salient {
  type Error = Infallible;
  type BoundingBox = Bbox;

  fn try_new(bbox: Self::BoundingBox, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self { bbox, confidence })
  }
}

impl HorizonInfo for Horizon {
  type Error = Infallible;

  fn try_new(angle: f32, confidence: f32) -> Result<Self, Self::Error> {
    Ok(Self { angle, confidence })
  }
}

impl DocumentSegment for Quad {
  type Error = Infallible;

  fn try_new(
    top_left: (f32, f32),
    top_right: (f32, f32),
    bottom_right: (f32, f32),
    bottom_left: (f32, f32),
    confidence: f32,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      corners: [top_left, top_right, bottom_right, bottom_left],
      confidence,
    })
  }
}

impl Aesthetics for Score {
  fn new(overall_score: f32, is_utility: bool) -> Self {
    Self {
      overall_score,
      is_utility,
    }
  }
}

impl Detections for Plain {
  type BoundingBox = Bbox;
  type Detection = Label;
  type SubjectDetection = Subject;
  type FaceDetection = Face;
  type FaceLandmarkRegion = LandmarkRegion;
  type FaceLandmarksDetection = FaceLandmarks;
  // Four skeletons, four joint types, four pose types. The bundle used
  // to require one joint type for the 2-D three; that it no longer does
  // is what this impl compiles.
  type BodyJoint = BodyJoint;
  type BodyPoseDetection = Pose;
  type HandJoint = HandJoint;
  type HandPoseDetection = HandPose;
  type AnimalJoint = AnimalJoint;
  type AnimalPoseDetection = AnimalPose;
  type Body3Joint = Body3Joint;
  type BodyPose3DDetection = Pose3;
  type PersonInstanceMaskDetection = InstanceMask;
  type PersonSegmentationMask = SegmentationMask;
  type TextDetection = Text;
  type BarcodeDetection = Barcode;
  type SaliencyRegion = Salient;
  type HorizonInfo = Horizon;
  type DocumentSegment = Quad;
  type Aesthetics = Score;
}
