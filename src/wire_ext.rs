//! Local extension traits over [`mediaschema`] wire types.
//!
//! mediaschema currently exposes the protobuf-generated structs as plain
//! `#[derive(Default)]` records with public fields and no constructors. The
//! avanalyze engine pre-dates that shape and was written against a richer
//! `::new(...)` + `.with_*(...)` builder API. Until either side migrates, we
//! bridge with a thin set of extension traits living here.
//!
//! All methods are `#[inline(always)]` and `#[must_use]`; the compiler should
//! optimise them away. Nothing here adds new wire fields — these are pure
//! syntactic adapters.

use bytes::Bytes;
use mediaschema::{
  Aesthetics, AnimalAnalysis, BarcodeDetection, BodyPose3DDetection, BodyPose3DHeightEstimation,
  BodyPose3DJoint, BodyPoseDetection, BodyPoseJoint, BoundingBox, ClassificationDetection,
  Detection, Dimensions, DocumentSegment, ErrorInfo, FaceDetection, FaceLandmarkPoint,
  FaceLandmarkRegion, FaceLandmarksDetection, FeaturePrint, HandChirality, HandPoseDetection,
  HorizonInfo, HumanAnalysis, Id, Keyframe, PersonInstanceMaskDetection, PersonSegmentationMask,
  Point2D, SaliencyRegion, SubjectDetection, TextDetection,
};

// ----- BoundingBox ----------------------------------------------------------

/// Constructor sugar for [`BoundingBox`].
pub trait BoundingBoxExt {
  /// Build a `BoundingBox` from `(x, y, width, height)`.
  #[must_use]
  fn new(x: f32, y: f32, width: f32, height: f32) -> Self;
}

impl BoundingBoxExt for BoundingBox {
  #[inline(always)]
  fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
    Self {
      x,
      y,
      width,
      height,
      ..Default::default()
    }
  }
}

// ----- Dimensions -----------------------------------------------------------

/// Constructor sugar for [`Dimensions`].
pub trait DimensionsExt {
  /// Build a `Dimensions` from `(width, height)`. The wire layer stores `u32`
  /// while the engine works in `u16`; we widen here.
  #[must_use]
  fn new(width: u16, height: u16) -> Self;
}

impl DimensionsExt for Dimensions {
  #[inline(always)]
  fn new(width: u16, height: u16) -> Self {
    Self {
      width: u32::from(width),
      height: u32::from(height),
      ..Default::default()
    }
  }
}

// ----- Point2D --------------------------------------------------------------

#[inline(always)]
fn point2d(x: f32, y: f32) -> Point2D {
  Point2D {
    x,
    y,
    ..Default::default()
  }
}

// ----- ErrorInfo ------------------------------------------------------------

/// Constructor + read-only accessor sugar for [`ErrorInfo`].
pub trait ErrorInfoExt {
  /// Build an `ErrorInfo` from a domain error code and a message.
  #[must_use]
  fn new(code: mediaschema::domain::ErrorCode, message: String) -> Self;
  /// Returns the error code as the domain enum, or its raw wire value if unknown.
  fn code(&self) -> mediaschema::domain::ErrorCode;
  /// Returns the human-readable error message.
  fn message(&self) -> &str;
}

impl ErrorInfoExt for ErrorInfo {
  #[inline(always)]
  fn new(code: mediaschema::domain::ErrorCode, message: String) -> Self {
    Self {
      code: code.as_u32(),
      message,
      ..Default::default()
    }
  }

  #[inline(always)]
  fn code(&self) -> mediaschema::domain::ErrorCode {
    mediaschema::domain::ErrorCode::from_u32(self.code)
  }

  #[inline(always)]
  fn message(&self) -> &str {
    &self.message
  }
}

// ----- Detection ------------------------------------------------------------

#[inline(always)]
fn detection(label: impl Into<String>, confidence: f32) -> Detection {
  Detection {
    label: label.into(),
    confidence,
    ..Default::default()
  }
}

// ----- ClassificationDetection ---------------------------------------------

/// Constructor sugar for [`ClassificationDetection`].
pub trait ClassificationDetectionExt {
  /// Build a `ClassificationDetection` from `(label, confidence)`.
  #[must_use]
  fn new(label: impl Into<String>, confidence: f32) -> Self;
}

impl ClassificationDetectionExt for ClassificationDetection {
  #[inline(always)]
  fn new(label: impl Into<String>, confidence: f32) -> Self {
    Self {
      detection: detection(label, confidence).into(),
      ..Default::default()
    }
  }
}

// ----- SubjectDetection -----------------------------------------------------

/// Constructor sugar for [`SubjectDetection`].
pub trait SubjectDetectionExt {
  /// Build a `SubjectDetection` from `(label, confidence, bbox)`.
  #[must_use]
  fn new(label: impl Into<String>, confidence: f32, bbox: BoundingBox) -> Self;
}

impl SubjectDetectionExt for SubjectDetection {
  #[inline(always)]
  fn new(label: impl Into<String>, confidence: f32, bbox: BoundingBox) -> Self {
    Self {
      detection: detection(label, confidence).into(),
      bbox: bbox.into(),
      ..Default::default()
    }
  }
}

// ----- FaceDetection --------------------------------------------------------

/// Builder sugar for [`FaceDetection`].
pub trait FaceDetectionExt {
  /// Set the bounding box.
  #[must_use]
  fn with_bbox(self, bbox: BoundingBox) -> Self;
  /// Set the detection confidence.
  #[must_use]
  fn with_confidence(self, confidence: f32) -> Self;
  /// Set the capture-quality score.
  #[must_use]
  fn with_capture_quality(self, capture_quality: f32) -> Self;
  /// Set the head-roll angle (radians).
  #[must_use]
  fn with_roll(self, roll: f32) -> Self;
  /// Set the head-yaw angle (radians).
  #[must_use]
  fn with_yaw(self, yaw: f32) -> Self;
  /// Set the head-pitch angle (radians).
  #[must_use]
  fn with_pitch(self, pitch: f32) -> Self;
}

impl FaceDetectionExt for FaceDetection {
  #[inline(always)]
  fn with_bbox(mut self, bbox: BoundingBox) -> Self {
    self.bbox = bbox.into();
    self
  }
  #[inline(always)]
  fn with_confidence(mut self, confidence: f32) -> Self {
    self.confidence = confidence;
    self
  }
  #[inline(always)]
  fn with_capture_quality(mut self, capture_quality: f32) -> Self {
    self.capture_quality = capture_quality;
    self
  }
  #[inline(always)]
  fn with_roll(mut self, roll: f32) -> Self {
    self.roll = roll;
    self
  }
  #[inline(always)]
  fn with_yaw(mut self, yaw: f32) -> Self {
    self.yaw = yaw;
    self
  }
  #[inline(always)]
  fn with_pitch(mut self, pitch: f32) -> Self {
    self.pitch = pitch;
    self
  }
}

// ----- FaceLandmarksDetection ----------------------------------------------

/// Constructor sugar for [`FaceLandmarksDetection`].
pub trait FaceLandmarksDetectionExt {
  /// Build a `FaceLandmarksDetection` from `(bbox, confidence, regions)`.
  #[must_use]
  fn new(bbox: BoundingBox, confidence: f32, regions: Vec<FaceLandmarkRegion>) -> Self;
}

impl FaceLandmarksDetectionExt for FaceLandmarksDetection {
  #[inline(always)]
  fn new(bbox: BoundingBox, confidence: f32, regions: Vec<FaceLandmarkRegion>) -> Self {
    Self {
      bbox: bbox.into(),
      confidence,
      regions,
      ..Default::default()
    }
  }
}

// ----- FaceLandmarkRegion --------------------------------------------------

/// Constructor sugar for [`FaceLandmarkRegion`].
pub trait FaceLandmarkRegionExt {
  /// Build a `FaceLandmarkRegion` from `(name, points)`.
  #[must_use]
  fn new(name: impl Into<String>, points: Vec<FaceLandmarkPoint>) -> Self;
}

impl FaceLandmarkRegionExt for FaceLandmarkRegion {
  #[inline(always)]
  fn new(name: impl Into<String>, points: Vec<FaceLandmarkPoint>) -> Self {
    Self {
      name: name.into(),
      points,
      ..Default::default()
    }
  }
}

// ----- FaceLandmarkPoint ---------------------------------------------------

/// Constructor sugar for [`FaceLandmarkPoint`].
pub trait FaceLandmarkPointExt {
  /// Build a `FaceLandmarkPoint` from `(x, y)`.
  #[must_use]
  fn new(x: f32, y: f32) -> Self;
}

impl FaceLandmarkPointExt for FaceLandmarkPoint {
  #[inline(always)]
  fn new(x: f32, y: f32) -> Self {
    Self {
      x,
      y,
      ..Default::default()
    }
  }
}

// ----- BodyPoseJoint -------------------------------------------------------

/// Constructor + accessor sugar for [`BodyPoseJoint`].
pub trait BodyPoseJointExt {
  /// Build a `BodyPoseJoint` from `(name, x, y, confidence)`.
  #[must_use]
  fn new(name: impl Into<String>, x: f32, y: f32, confidence: f32) -> Self;
  /// Returns the joint name.
  fn name(&self) -> &str;
}

impl BodyPoseJointExt for BodyPoseJoint {
  #[inline(always)]
  fn new(name: impl Into<String>, x: f32, y: f32, confidence: f32) -> Self {
    Self {
      name: name.into(),
      x,
      y,
      confidence,
      ..Default::default()
    }
  }
  #[inline(always)]
  fn name(&self) -> &str {
    &self.name
  }
}

// ----- BodyPoseDetection ---------------------------------------------------

/// Constructor sugar for [`BodyPoseDetection`].
pub trait BodyPoseDetectionExt {
  /// Build a `BodyPoseDetection` from `(bbox, confidence, joints)`.
  #[must_use]
  fn new(bbox: BoundingBox, confidence: f32, joints: Vec<BodyPoseJoint>) -> Self;
}

impl BodyPoseDetectionExt for BodyPoseDetection {
  #[inline(always)]
  fn new(bbox: BoundingBox, confidence: f32, joints: Vec<BodyPoseJoint>) -> Self {
    Self {
      bbox: bbox.into(),
      confidence,
      joints,
      ..Default::default()
    }
  }
}

// ----- BodyPose3DJoint -----------------------------------------------------

/// Constructor + accessor sugar for [`BodyPose3DJoint`].
pub trait BodyPose3DJointExt {
  /// Build a `BodyPose3DJoint` from `(name, x, y, z, confidence)`.
  #[must_use]
  fn new(name: impl Into<String>, x: f32, y: f32, z: f32, confidence: f32) -> Self;
  /// Returns the joint name.
  fn name(&self) -> &str;
}

impl BodyPose3DJointExt for BodyPose3DJoint {
  #[inline(always)]
  fn new(name: impl Into<String>, x: f32, y: f32, z: f32, confidence: f32) -> Self {
    Self {
      name: name.into(),
      x,
      y,
      z,
      confidence,
      ..Default::default()
    }
  }
  #[inline(always)]
  fn name(&self) -> &str {
    &self.name
  }
}

// ----- BodyPose3DDetection -------------------------------------------------

/// Constructor sugar for [`BodyPose3DDetection`].
pub trait BodyPose3DDetectionExt {
  /// Build a `BodyPose3DDetection` from `(confidence, body_height, height_estimation, joints)`.
  #[must_use]
  fn new(
    confidence: f32,
    body_height: f32,
    height_estimation: BodyPose3DHeightEstimation,
    joints: Vec<BodyPose3DJoint>,
  ) -> Self;
}

impl BodyPose3DDetectionExt for BodyPose3DDetection {
  #[inline(always)]
  fn new(
    confidence: f32,
    body_height: f32,
    height_estimation: BodyPose3DHeightEstimation,
    joints: Vec<BodyPose3DJoint>,
  ) -> Self {
    Self {
      confidence,
      body_height,
      height_estimation: height_estimation.into(),
      joints,
      ..Default::default()
    }
  }
}

// ----- HandPoseDetection ---------------------------------------------------

/// Constructor sugar for [`HandPoseDetection`].
pub trait HandPoseDetectionExt {
  /// Build a `HandPoseDetection` from `(bbox, confidence, chirality, joints)`.
  #[must_use]
  fn new(
    bbox: BoundingBox,
    confidence: f32,
    chirality: HandChirality,
    joints: Vec<BodyPoseJoint>,
  ) -> Self;
}

impl HandPoseDetectionExt for HandPoseDetection {
  #[inline(always)]
  fn new(
    bbox: BoundingBox,
    confidence: f32,
    chirality: HandChirality,
    joints: Vec<BodyPoseJoint>,
  ) -> Self {
    Self {
      bbox: bbox.into(),
      confidence,
      chirality: chirality.into(),
      joints,
      ..Default::default()
    }
  }
}

// ----- PersonInstanceMaskDetection -----------------------------------------

/// Constructor sugar for [`PersonInstanceMaskDetection`].
pub trait PersonInstanceMaskDetectionExt {
  /// Build a `PersonInstanceMaskDetection` from
  /// `(bbox, confidence, instance_index, dimensions, data)`.
  #[must_use]
  fn new(
    bbox: BoundingBox,
    confidence: f32,
    instance_index: u32,
    dimensions: Dimensions,
    data: Bytes,
  ) -> Self;
}

impl PersonInstanceMaskDetectionExt for PersonInstanceMaskDetection {
  #[inline(always)]
  fn new(
    bbox: BoundingBox,
    confidence: f32,
    instance_index: u32,
    dimensions: Dimensions,
    data: Bytes,
  ) -> Self {
    Self {
      bbox: bbox.into(),
      confidence,
      instance_index,
      dimensions: dimensions.into(),
      data,
      ..Default::default()
    }
  }
}

// ----- PersonSegmentationMask ----------------------------------------------

/// Constructor sugar for [`PersonSegmentationMask`].
pub trait PersonSegmentationMaskExt {
  /// Build a `PersonSegmentationMask` from `(bbox, confidence, dimensions, data)`.
  #[must_use]
  fn new(bbox: BoundingBox, confidence: f32, dimensions: Dimensions, data: Bytes) -> Self;
}

impl PersonSegmentationMaskExt for PersonSegmentationMask {
  #[inline(always)]
  fn new(bbox: BoundingBox, confidence: f32, dimensions: Dimensions, data: Bytes) -> Self {
    Self {
      bbox: bbox.into(),
      confidence,
      dimensions: dimensions.into(),
      data,
      ..Default::default()
    }
  }
}

// ----- TextDetection -------------------------------------------------------

/// Constructor sugar for [`TextDetection`].
pub trait TextDetectionExt {
  /// Build a `TextDetection` from `(text, confidence, bbox)`.
  #[must_use]
  fn new(text: impl Into<String>, confidence: f32, bbox: BoundingBox) -> Self;
}

impl TextDetectionExt for TextDetection {
  #[inline(always)]
  fn new(text: impl Into<String>, confidence: f32, bbox: BoundingBox) -> Self {
    Self {
      text: text.into(),
      confidence,
      bbox: bbox.into(),
      ..Default::default()
    }
  }
}

// ----- BarcodeDetection ----------------------------------------------------

/// Constructor sugar for [`BarcodeDetection`].
pub trait BarcodeDetectionExt {
  /// Build a `BarcodeDetection` from `(payload, symbology, confidence, bbox)`.
  #[must_use]
  fn new(
    payload: impl Into<String>,
    symbology: impl Into<String>,
    confidence: f32,
    bbox: BoundingBox,
  ) -> Self;
}

impl BarcodeDetectionExt for BarcodeDetection {
  #[inline(always)]
  fn new(
    payload: impl Into<String>,
    symbology: impl Into<String>,
    confidence: f32,
    bbox: BoundingBox,
  ) -> Self {
    Self {
      payload: payload.into(),
      symbology: symbology.into(),
      confidence,
      bbox: bbox.into(),
      ..Default::default()
    }
  }
}

// ----- SaliencyRegion ------------------------------------------------------

/// Constructor sugar for [`SaliencyRegion`].
pub trait SaliencyRegionExt {
  /// Build a `SaliencyRegion` from `(bbox, confidence)`.
  #[must_use]
  fn new(bbox: BoundingBox, confidence: f32) -> Self;
}

impl SaliencyRegionExt for SaliencyRegion {
  #[inline(always)]
  fn new(bbox: BoundingBox, confidence: f32) -> Self {
    Self {
      bbox: bbox.into(),
      confidence,
      ..Default::default()
    }
  }
}

// ----- HorizonInfo ---------------------------------------------------------

/// Constructor sugar for [`HorizonInfo`].
pub trait HorizonInfoExt {
  /// Build a `HorizonInfo` from `(angle, confidence)`.
  #[must_use]
  fn new(angle: f32, confidence: f32) -> Self;
}

impl HorizonInfoExt for HorizonInfo {
  #[inline(always)]
  fn new(angle: f32, confidence: f32) -> Self {
    Self {
      angle,
      confidence,
      ..Default::default()
    }
  }
}

// ----- DocumentSegment -----------------------------------------------------

/// Builder sugar for [`DocumentSegment`].
pub trait DocumentSegmentExt {
  /// Set the top-left corner of the segment.
  #[must_use]
  fn with_top_left(self, xy: (f32, f32)) -> Self;
  /// Set the top-right corner of the segment.
  #[must_use]
  fn with_top_right(self, xy: (f32, f32)) -> Self;
  /// Set the bottom-left corner of the segment.
  #[must_use]
  fn with_bottom_left(self, xy: (f32, f32)) -> Self;
  /// Set the bottom-right corner of the segment.
  #[must_use]
  fn with_bottom_right(self, xy: (f32, f32)) -> Self;
  /// Set the confidence of the segmentation.
  #[must_use]
  fn with_confidence(self, confidence: f32) -> Self;
}

impl DocumentSegmentExt for DocumentSegment {
  #[inline(always)]
  fn with_top_left(mut self, (x, y): (f32, f32)) -> Self {
    self.top_left = point2d(x, y).into();
    self
  }
  #[inline(always)]
  fn with_top_right(mut self, (x, y): (f32, f32)) -> Self {
    self.top_right = point2d(x, y).into();
    self
  }
  #[inline(always)]
  fn with_bottom_left(mut self, (x, y): (f32, f32)) -> Self {
    self.bottom_left = point2d(x, y).into();
    self
  }
  #[inline(always)]
  fn with_bottom_right(mut self, (x, y): (f32, f32)) -> Self {
    self.bottom_right = point2d(x, y).into();
    self
  }
  #[inline(always)]
  fn with_confidence(mut self, confidence: f32) -> Self {
    self.confidence = confidence;
    self
  }
}

// ----- Aesthetics ----------------------------------------------------------

/// Constructor sugar for [`Aesthetics`].
pub trait AestheticsExt {
  /// Build an `Aesthetics` from `(overall_score, is_utility)`.
  #[must_use]
  fn new(overall_score: f32, is_utility: bool) -> Self;
}

impl AestheticsExt for Aesthetics {
  #[inline(always)]
  fn new(overall_score: f32, is_utility: bool) -> Self {
    Self {
      overall_score,
      is_utility,
      ..Default::default()
    }
  }
}

// ----- FeaturePrint --------------------------------------------------------

/// Constructor sugar for [`FeaturePrint`].
pub trait FeaturePrintExt {
  /// Build a `FeaturePrint` from `(data, element_type)`.
  #[must_use]
  fn new(data: Bytes, element_type: u32) -> Self;
}

impl FeaturePrintExt for FeaturePrint {
  #[inline(always)]
  fn new(data: Bytes, element_type: u32) -> Self {
    Self {
      data,
      element_type,
      ..Default::default()
    }
  }
}

// ----- HumanAnalysis -------------------------------------------------------

/// Constructor + builder sugar for [`HumanAnalysis`].
pub trait HumanAnalysisExt {
  /// Build an empty `HumanAnalysis`.
  #[must_use]
  fn new() -> Self;
  /// Set the per-subject detections.
  #[must_use]
  fn with_subjects(self, subjects: Vec<SubjectDetection>) -> Self;
  /// Set the per-face capture-quality detections.
  #[must_use]
  fn with_faces(self, faces: Vec<FaceDetection>) -> Self;
  /// Set the per-face rectangle detections.
  #[must_use]
  fn with_face_rectangles(self, face_rectangles: Vec<FaceDetection>) -> Self;
  /// Set the per-face landmark detections.
  #[must_use]
  fn with_face_landmarks(self, face_landmarks: Vec<FaceLandmarksDetection>) -> Self;
  /// Set the per-person body-pose detections.
  #[must_use]
  fn with_body_poses(self, body_poses: Vec<BodyPoseDetection>) -> Self;
  /// Set the per-person hand-pose detections.
  #[must_use]
  fn with_hand_poses(self, hand_poses: Vec<HandPoseDetection>) -> Self;
  /// Set the per-person 3D body-pose detections.
  #[must_use]
  fn with_body_poses_3d(self, body_poses_3d: Vec<BodyPose3DDetection>) -> Self;
  /// Set the per-person instance-segmentation masks.
  #[must_use]
  fn with_instance_masks(self, instance_masks: Vec<PersonInstanceMaskDetection>) -> Self;
  /// Set the binary segmentation masks.
  #[must_use]
  fn with_segmentation_masks(self, segmentation_masks: Vec<PersonSegmentationMask>) -> Self;
}

impl HumanAnalysisExt for HumanAnalysis {
  #[inline(always)]
  fn new() -> Self {
    Self::default()
  }
  #[inline(always)]
  fn with_subjects(mut self, subjects: Vec<SubjectDetection>) -> Self {
    self.subjects = subjects;
    self
  }
  #[inline(always)]
  fn with_faces(mut self, faces: Vec<FaceDetection>) -> Self {
    self.faces = faces;
    self
  }
  #[inline(always)]
  fn with_face_rectangles(mut self, face_rectangles: Vec<FaceDetection>) -> Self {
    self.face_rectangles = face_rectangles;
    self
  }
  #[inline(always)]
  fn with_face_landmarks(mut self, face_landmarks: Vec<FaceLandmarksDetection>) -> Self {
    self.face_landmarks = face_landmarks;
    self
  }
  #[inline(always)]
  fn with_body_poses(mut self, body_poses: Vec<BodyPoseDetection>) -> Self {
    self.body_poses = body_poses;
    self
  }
  #[inline(always)]
  fn with_hand_poses(mut self, hand_poses: Vec<HandPoseDetection>) -> Self {
    self.hand_poses = hand_poses;
    self
  }
  #[inline(always)]
  fn with_body_poses_3d(mut self, body_poses_3d: Vec<BodyPose3DDetection>) -> Self {
    self.body_poses_3d = body_poses_3d;
    self
  }
  #[inline(always)]
  fn with_instance_masks(mut self, instance_masks: Vec<PersonInstanceMaskDetection>) -> Self {
    self.instance_masks = instance_masks;
    self
  }
  #[inline(always)]
  fn with_segmentation_masks(mut self, segmentation_masks: Vec<PersonSegmentationMask>) -> Self {
    self.segmentation_masks = segmentation_masks;
    self
  }
}

// ----- AnimalAnalysis ------------------------------------------------------

/// Constructor + builder sugar for [`AnimalAnalysis`].
pub trait AnimalAnalysisExt {
  /// Build an empty `AnimalAnalysis`.
  #[must_use]
  fn new() -> Self;
  /// Set the per-subject animal detections.
  #[must_use]
  fn with_subjects(self, subjects: Vec<SubjectDetection>) -> Self;
  /// Set the per-animal body-pose detections.
  #[must_use]
  fn with_body_poses(self, body_poses: Vec<BodyPoseDetection>) -> Self;
}

impl AnimalAnalysisExt for AnimalAnalysis {
  #[inline(always)]
  fn new() -> Self {
    Self::default()
  }
  #[inline(always)]
  fn with_subjects(mut self, subjects: Vec<SubjectDetection>) -> Self {
    self.subjects = subjects;
    self
  }
  #[inline(always)]
  fn with_body_poses(mut self, body_poses: Vec<BodyPoseDetection>) -> Self {
    self.body_poses = body_poses;
    self
  }
}

// ----- Keyframe ------------------------------------------------------------

/// Builder sugar for [`Keyframe`].
///
/// The wire `Keyframe` has many list-typed fields with no aggregate parent.
/// Some setters here drop their input on the floor when the wire schema has
/// no corresponding field (`objects`, `actions`, `mood`, `emotion`,
/// `lighting`, `colors`); they remain on the trait so the engine code keeps
/// composing without changes. The discarded categories all currently feed
/// empty `Vec`s from this engine anyway.
pub trait KeyframeExt {
  /// Set the keyframe identifier (wire `id: Bytes`).
  #[must_use]
  fn with_id(self, id: Id) -> Self;
  /// Set the parent scene identifier (wire `scene_id: Bytes`).
  #[must_use]
  fn with_scene_id(self, scene_id: Id) -> Self;
  /// Set the classifications.
  #[must_use]
  fn with_classifications(self, classifications: Vec<ClassificationDetection>) -> Self;
  /// Set the human-analysis aggregate.
  #[must_use]
  fn with_humans(self, humans: HumanAnalysis) -> Self;
  /// Set the animal-analysis aggregate.
  #[must_use]
  fn with_animals(self, animals: AnimalAnalysis) -> Self;
  /// Set the recognised text detections.
  #[must_use]
  fn with_text_detections(self, text_detections: Vec<TextDetection>) -> Self;
  /// Set the barcode detections.
  #[must_use]
  fn with_barcodes(self, barcodes: Vec<BarcodeDetection>) -> Self;
  /// Set the attention-based saliency regions.
  #[must_use]
  fn with_attention_saliency(self, attention_saliency: Vec<SaliencyRegion>) -> Self;
  /// Set the objectness-based saliency regions.
  #[must_use]
  fn with_objectness_saliency(self, objectness_saliency: Vec<SaliencyRegion>) -> Self;
  /// Set the horizon information.
  #[must_use]
  fn with_horizon(self, horizon: HorizonInfo) -> Self;
  /// Set the document segments.
  #[must_use]
  fn with_document_segments(self, document_segments: Vec<DocumentSegment>) -> Self;
  /// Set the feature print.
  #[must_use]
  fn with_feature_print(self, feature_print: FeaturePrint) -> Self;
  /// Set the aesthetic scores.
  #[must_use]
  fn with_aesthetics(self, aesthetics: Aesthetics) -> Self;
}

impl KeyframeExt for Keyframe {
  #[inline(always)]
  fn with_id(mut self, id: Id) -> Self {
    self.id = id.value;
    self
  }
  #[inline(always)]
  fn with_scene_id(mut self, scene_id: Id) -> Self {
    self.scene_id = scene_id.value;
    self
  }
  #[inline(always)]
  fn with_classifications(mut self, classifications: Vec<ClassificationDetection>) -> Self {
    self.classifications = classifications;
    self
  }
  #[inline(always)]
  fn with_humans(mut self, humans: HumanAnalysis) -> Self {
    self.humans = humans.into();
    self
  }
  #[inline(always)]
  fn with_animals(mut self, animals: AnimalAnalysis) -> Self {
    self.animals = animals.into();
    self
  }
  #[inline(always)]
  fn with_text_detections(mut self, text_detections: Vec<TextDetection>) -> Self {
    self.text_detections = text_detections;
    self
  }
  #[inline(always)]
  fn with_barcodes(mut self, barcodes: Vec<BarcodeDetection>) -> Self {
    self.barcodes = barcodes;
    self
  }
  #[inline(always)]
  fn with_attention_saliency(mut self, attention_saliency: Vec<SaliencyRegion>) -> Self {
    self.attention_saliency = attention_saliency;
    self
  }
  #[inline(always)]
  fn with_objectness_saliency(mut self, objectness_saliency: Vec<SaliencyRegion>) -> Self {
    self.objectness_saliency = objectness_saliency;
    self
  }
  #[inline(always)]
  fn with_horizon(mut self, horizon: HorizonInfo) -> Self {
    self.horizon = horizon.into();
    self
  }
  #[inline(always)]
  fn with_document_segments(mut self, document_segments: Vec<DocumentSegment>) -> Self {
    self.document_segments = document_segments;
    self
  }
  #[inline(always)]
  fn with_feature_print(mut self, feature_print: FeaturePrint) -> Self {
    self.feature_print = feature_print.into();
    self
  }
  #[inline(always)]
  fn with_aesthetics(mut self, aesthetics: Aesthetics) -> Self {
    self.aesthetics = aesthetics.into();
    self
  }
}

// ----- Apple Vision <-> wire enum bridges ---------------------------------

/// Re-named variant aliases for [`HandChirality`] so engine-side code can keep
/// using the short names (`HandChirality::Left`) without churn.
pub const HAND_CHIRALITY_LEFT: HandChirality = HandChirality::HAND_CHIRALITY_LEFT;
/// Right-hand variant alias.
pub const HAND_CHIRALITY_RIGHT: HandChirality = HandChirality::HAND_CHIRALITY_RIGHT;
/// Unknown / unspecified variant alias.
pub const HAND_CHIRALITY_UNKNOWN: HandChirality = HandChirality::HAND_CHIRALITY_UNSPECIFIED;

/// Re-named variant aliases for [`BodyPose3DHeightEstimation`].
pub const BODY_POSE_3D_HEIGHT_ESTIMATION_MEASURED: BodyPose3DHeightEstimation =
  BodyPose3DHeightEstimation::BODY_POSE_3D_HEIGHT_ESTIMATION_MEASURED;
/// Reference-height variant alias.
pub const BODY_POSE_3D_HEIGHT_ESTIMATION_REFERENCE: BodyPose3DHeightEstimation =
  BodyPose3DHeightEstimation::BODY_POSE_3D_HEIGHT_ESTIMATION_REFERENCE;
/// Unknown / unspecified variant alias.
pub const BODY_POSE_3D_HEIGHT_ESTIMATION_UNKNOWN: BodyPose3DHeightEstimation =
  BodyPose3DHeightEstimation::BODY_POSE_3D_HEIGHT_ESTIMATION_UNSPECIFIED;
