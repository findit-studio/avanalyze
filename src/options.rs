#![allow(missing_docs)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


macro_rules! default_options {
  ($($name:ident),+$(,)?) => {
    $(
      impl Default for $name {
        #[cfg_attr(not(tarpaulin), inline(always))]
        fn default() -> Self {
          Self::new()
        }
      }
    )*
  };
}

default_options!(
  AppleVisionClassificationOptions,
  AppleVisionAnimalOptions,
  AppleVisionTextOptions,
  AppleVisionBodyPoseOptions,
  AppleVisionHandPoseOptions,
  AppleVisionAnimalPoseOptions,
  AppleVisionBodyPose3DOptions,
  AppleVisionFaceCaptureOptions,
  AppleVisionFaceRectangleOptions,
  AppleVisionFaceLandmarkOptions,
  AppleVisionHumanSubjectOptions,
  AppleVisionBarcodeOptions,
  AppleVisionSaliencyOptions,
  AppleVisionHorizonOptions,
  AppleVisionDocumentSegmentationOptions,
  AppleVisionAestheticsOptions,
  AppleVisionFeaturePrintOptions,
  AppleVisionPersonInstanceMaskOptions,
  AppleVisionPersonSegmentationOptions,
  ServiceOptions,
);

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_classification_min_confidence() -> f32 {
  0.3
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_classification_max_results() -> usize {
  12
}
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionClassificationOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_classification_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_classification_max_results")
  )]
  max_results: usize,
}

impl AppleVisionClassificationOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_classification_min_confidence(),
      max_results: default_classification_max_results(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_results(mut self, max_results: usize) -> Self {
    self.set_max_results(max_results);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_results(&mut self, max_results: usize) -> &mut Self {
    self.max_results = max_results;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_results(&self) -> usize {
    self.max_results
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_animal_min_confidence() -> f32 {
  0.3
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionAnimalOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_animal_min_confidence"))]
  min_confidence: f32,
}

impl AppleVisionAnimalOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_animal_min_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_min_len() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_text_max_candidates_per_observation() -> usize {
  1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionTextOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_text_min_len"))]
  min_text_len: usize,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_text_max_candidates_per_observation")
  )]
  max_candidates_per_observation: usize,
}

impl AppleVisionTextOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_text_len: default_text_min_len(),
      max_candidates_per_observation: default_text_max_candidates_per_observation(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_text_len(mut self, min_text_len: usize) -> Self {
    self.set_min_text_len(min_text_len);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_text_len(&mut self, min_text_len: usize) -> &mut Self {
    self.min_text_len = min_text_len;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_text_len(&self) -> usize {
    self.min_text_len
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_candidates_per_observation(
    mut self,
    max_candidates_per_observation: usize,
  ) -> Self {
    self.set_max_candidates_per_observation(max_candidates_per_observation);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_candidates_per_observation(
    &mut self,
    max_candidates_per_observation: usize,
  ) -> &mut Self {
    self.max_candidates_per_observation = max_candidates_per_observation;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_candidates_per_observation(&self) -> usize {
    self.max_candidates_per_observation
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_body_pose_min_joint_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBodyPoseOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_body_pose_min_joint_confidence")
  )]
  min_joint_confidence: f32,
}

impl AppleVisionBodyPoseOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: default_body_pose_min_joint_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_hand_pose_min_joint_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_hand_pose_maximum_hand_count() -> usize {
  2
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionHandPoseOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_hand_pose_min_joint_confidence")
  )]
  min_joint_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_hand_pose_maximum_hand_count")
  )]
  maximum_hand_count: usize,
}

impl AppleVisionHandPoseOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: default_hand_pose_min_joint_confidence(),
      maximum_hand_count: default_hand_pose_maximum_hand_count(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_maximum_hand_count(mut self, maximum_hand_count: usize) -> Self {
    self.set_maximum_hand_count(maximum_hand_count);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_maximum_hand_count(&mut self, maximum_hand_count: usize) -> &mut Self {
    self.maximum_hand_count = maximum_hand_count;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn maximum_hand_count(&self) -> usize {
    self.maximum_hand_count
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_animal_pose_min_joint_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionAnimalPoseOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_animal_pose_min_joint_confidence")
  )]
  min_joint_confidence: f32,
}

impl AppleVisionAnimalPoseOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: default_animal_pose_min_joint_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_body_pose_3d_min_joint_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBodyPose3DOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_body_pose_3d_min_joint_confidence")
  )]
  min_joint_confidence: f32,
}

impl AppleVisionBodyPose3DOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_joint_confidence: default_body_pose_3d_min_joint_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_joint_confidence(mut self, min_joint_confidence: f32) -> Self {
    self.set_min_joint_confidence(min_joint_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_joint_confidence(&mut self, min_joint_confidence: f32) -> &mut Self {
    self.min_joint_confidence = min_joint_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_joint_confidence(&self) -> f32 {
    self.min_joint_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_capture_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_capture_min_capture_quality() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceCaptureOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_capture_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_capture_min_capture_quality")
  )]
  min_capture_quality: f32,
}

impl AppleVisionFaceCaptureOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_face_capture_min_confidence(),
      min_capture_quality: default_face_capture_min_capture_quality(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_capture_quality(mut self, min_capture_quality: f32) -> Self {
    self.set_min_capture_quality(min_capture_quality);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_capture_quality(&mut self, min_capture_quality: f32) -> &mut Self {
    self.min_capture_quality = min_capture_quality;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_capture_quality(&self) -> f32 {
    self.min_capture_quality
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_rectangle_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceRectangleOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_rectangle_min_confidence")
  )]
  min_confidence: f32,
}

impl AppleVisionFaceRectangleOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_face_rectangle_min_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_landmark_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_face_landmark_min_region_count() -> usize {
  1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFaceLandmarkOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_landmark_min_confidence")
  )]
  min_confidence: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_face_landmark_min_region_count")
  )]
  min_region_count: usize,
}

impl AppleVisionFaceLandmarkOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_face_landmark_min_confidence(),
      min_region_count: default_face_landmark_min_region_count(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_region_count(mut self, min_region_count: usize) -> Self {
    self.set_min_region_count(min_region_count);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_region_count(&mut self, min_region_count: usize) -> &mut Self {
    self.min_region_count = min_region_count;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_region_count(&self) -> usize {
    self.min_region_count
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_human_subject_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionHumanSubjectOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_human_subject_min_confidence")
  )]
  min_confidence: f32,
}

impl AppleVisionHumanSubjectOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_human_subject_min_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_barcode_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_barcode_min_payload_len() -> usize {
  1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionBarcodeOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_barcode_min_confidence"))]
  min_confidence: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_barcode_min_payload_len"))]
  min_payload_len: usize,
}

impl AppleVisionBarcodeOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_barcode_min_confidence(),
      min_payload_len: default_barcode_min_payload_len(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_payload_len(mut self, min_payload_len: usize) -> Self {
    self.set_min_payload_len(min_payload_len);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_payload_len(&mut self, min_payload_len: usize) -> &mut Self {
    self.min_payload_len = min_payload_len;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_payload_len(&self) -> usize {
    self.min_payload_len
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_saliency_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_saliency_max_regions() -> usize {
  16
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionSaliencyOptions {
  min_confidence: f32,
  max_regions: usize,
}

impl AppleVisionSaliencyOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_saliency_min_confidence(),
      max_regions: default_saliency_max_regions(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_regions(mut self, max_regions: usize) -> Self {
    self.set_max_regions(max_regions);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_regions(&mut self, max_regions: usize) -> &mut Self {
    self.max_regions = max_regions;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_regions(&self) -> usize {
    self.max_regions
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_horizon_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionHorizonOptions {
  min_confidence: f32,
}

impl AppleVisionHorizonOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_horizon_min_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_document_segmentation_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_document_segmentation_max_segments() -> usize {
  16
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionDocumentSegmentationOptions {
  min_confidence: f32,
  max_segments: usize,
}

impl AppleVisionDocumentSegmentationOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_document_segmentation_min_confidence(),
      max_segments: default_document_segmentation_max_segments(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_segments(mut self, max_segments: usize) -> Self {
    self.set_max_segments(max_segments);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_segments(&mut self, max_segments: usize) -> &mut Self {
    self.max_segments = max_segments;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_segments(&self) -> usize {
    self.max_segments
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_aesthetics_min_overall_score() -> f32 {
  -1.0
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionAestheticsOptions {
  min_overall_score: f32,
}

impl AppleVisionAestheticsOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_overall_score: default_aesthetics_min_overall_score(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_overall_score(mut self, min_overall_score: f32) -> Self {
    self.set_min_overall_score(min_overall_score);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_overall_score(&mut self, min_overall_score: f32) -> &mut Self {
    self.min_overall_score = min_overall_score;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_overall_score(&self) -> f32 {
    self.min_overall_score
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_feature_print_min_element_count() -> usize {
  1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionFeaturePrintOptions {
  min_element_count: usize,
}

impl AppleVisionFeaturePrintOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_element_count: default_feature_print_min_element_count(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_element_count(mut self, min_element_count: usize) -> Self {
    self.set_min_element_count(min_element_count);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_element_count(&mut self, min_element_count: usize) -> &mut Self {
    self.min_element_count = min_element_count;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_element_count(&self) -> usize {
    self.min_element_count
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_person_instance_mask_min_confidence() -> f32 {
  0.1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_person_instance_mask_max_instances_per_observation() -> usize {
  16
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionPersonInstanceMaskOptions {
  min_confidence: f32,
  max_instances_per_observation: usize,
}

impl AppleVisionPersonInstanceMaskOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_person_instance_mask_min_confidence(),
      max_instances_per_observation: default_person_instance_mask_max_instances_per_observation(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_instances_per_observation(
    mut self,
    max_instances_per_observation: usize,
  ) -> Self {
    self.set_max_instances_per_observation(max_instances_per_observation);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_instances_per_observation(
    &mut self,
    max_instances_per_observation: usize,
  ) -> &mut Self {
    self.max_instances_per_observation = max_instances_per_observation;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_instances_per_observation(&self) -> usize {
    self.max_instances_per_observation
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_person_segmentation_min_confidence() -> f32 {
  0.1
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AppleVisionPersonSegmentationOptions {
  min_confidence: f32,
}

impl AppleVisionPersonSegmentationOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      min_confidence: default_person_segmentation_min_confidence(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
    self.set_min_confidence(min_confidence);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_confidence(&mut self, min_confidence: f32) -> &mut Self {
    self.min_confidence = min_confidence;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_confidence(&self) -> f32 {
    self.min_confidence
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_num_workers() -> usize {
  1
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ServiceOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_num_workers"))]
  num_workers: usize,
  #[cfg_attr(feature = "serde", serde(default))]
  classifications: AppleVisionClassificationOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  face_capture: AppleVisionFaceCaptureOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  face_rectangles: AppleVisionFaceRectangleOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  face_landmarks: AppleVisionFaceLandmarkOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  human_subjects: AppleVisionHumanSubjectOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  animals: AppleVisionAnimalOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  text: AppleVisionTextOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  body_pose: AppleVisionBodyPoseOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  hand_pose: AppleVisionHandPoseOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  animal_pose: AppleVisionAnimalPoseOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  body_pose_3d: AppleVisionBodyPose3DOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  barcodes: AppleVisionBarcodeOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  attention_saliency: AppleVisionSaliencyOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  objectness_saliency: AppleVisionSaliencyOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  horizon: AppleVisionHorizonOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  document_segments: AppleVisionDocumentSegmentationOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  aesthetics: AppleVisionAestheticsOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  feature_print: AppleVisionFeaturePrintOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  person_instance_masks: AppleVisionPersonInstanceMaskOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  person_segmentation_masks: AppleVisionPersonSegmentationOptions,
}

impl ServiceOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      num_workers: 1,
      classifications: AppleVisionClassificationOptions::new(),
      face_capture: AppleVisionFaceCaptureOptions::new(),
      face_rectangles: AppleVisionFaceRectangleOptions::new(),
      face_landmarks: AppleVisionFaceLandmarkOptions::new(),
      human_subjects: AppleVisionHumanSubjectOptions::new(),
      animals: AppleVisionAnimalOptions::new(),
      text: AppleVisionTextOptions::new(),
      body_pose: AppleVisionBodyPoseOptions::new(),
      hand_pose: AppleVisionHandPoseOptions::new(),
      animal_pose: AppleVisionAnimalPoseOptions::new(),
      body_pose_3d: AppleVisionBodyPose3DOptions::new(),
      barcodes: AppleVisionBarcodeOptions::new(),
      attention_saliency: AppleVisionSaliencyOptions::new(),
      objectness_saliency: AppleVisionSaliencyOptions::new(),
      horizon: AppleVisionHorizonOptions::new(),
      document_segments: AppleVisionDocumentSegmentationOptions::new(),
      aesthetics: AppleVisionAestheticsOptions::new(),
      feature_print: AppleVisionFeaturePrintOptions::new(),
      person_instance_masks: AppleVisionPersonInstanceMaskOptions::new(),
      person_segmentation_masks: AppleVisionPersonSegmentationOptions::new(),
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_workers(mut self, num_workers: usize) -> Self {
    self.set_workers(num_workers);
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_workers(&mut self, num_workers: usize) -> &mut Self {
    self.num_workers = if num_workers == 0 { 1 } else { num_workers };
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn num_workers(&self) -> usize {
    self.num_workers
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn classifications(&self) -> AppleVisionClassificationOptions {
    self.classifications
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn classifications_mut(&mut self) -> &mut AppleVisionClassificationOptions {
    &mut self.classifications
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn face_capture(&self) -> AppleVisionFaceCaptureOptions {
    self.face_capture
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn face_capture_mut(&mut self) -> &mut AppleVisionFaceCaptureOptions {
    &mut self.face_capture
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn face_rectangles(&self) -> AppleVisionFaceRectangleOptions {
    self.face_rectangles
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn face_rectangles_mut(&mut self) -> &mut AppleVisionFaceRectangleOptions {
    &mut self.face_rectangles
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn face_landmarks(&self) -> AppleVisionFaceLandmarkOptions {
    self.face_landmarks
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn face_landmarks_mut(&mut self) -> &mut AppleVisionFaceLandmarkOptions {
    &mut self.face_landmarks
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn human_subjects(&self) -> AppleVisionHumanSubjectOptions {
    self.human_subjects
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn human_subjects_mut(&mut self) -> &mut AppleVisionHumanSubjectOptions {
    &mut self.human_subjects
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn animals(&self) -> AppleVisionAnimalOptions {
    self.animals
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn animals_mut(&mut self) -> &mut AppleVisionAnimalOptions {
    &mut self.animals
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn text(&self) -> AppleVisionTextOptions {
    self.text
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn text_mut(&mut self) -> &mut AppleVisionTextOptions {
    &mut self.text
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn body_pose(&self) -> AppleVisionBodyPoseOptions {
    self.body_pose
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn body_pose_mut(&mut self) -> &mut AppleVisionBodyPoseOptions {
    &mut self.body_pose
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hand_pose(&self) -> AppleVisionHandPoseOptions {
    self.hand_pose
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hand_pose_mut(&mut self) -> &mut AppleVisionHandPoseOptions {
    &mut self.hand_pose
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn animal_pose(&self) -> AppleVisionAnimalPoseOptions {
    self.animal_pose
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn animal_pose_mut(&mut self) -> &mut AppleVisionAnimalPoseOptions {
    &mut self.animal_pose
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn body_pose_3d(&self) -> AppleVisionBodyPose3DOptions {
    self.body_pose_3d
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn body_pose_3d_mut(&mut self) -> &mut AppleVisionBodyPose3DOptions {
    &mut self.body_pose_3d
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn barcodes(&self) -> AppleVisionBarcodeOptions {
    self.barcodes
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn barcodes_mut(&mut self) -> &mut AppleVisionBarcodeOptions {
    &mut self.barcodes
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn attention_saliency(&self) -> AppleVisionSaliencyOptions {
    self.attention_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn attention_saliency_mut(&mut self) -> &mut AppleVisionSaliencyOptions {
    &mut self.attention_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn objectness_saliency(&self) -> AppleVisionSaliencyOptions {
    self.objectness_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn objectness_saliency_mut(&mut self) -> &mut AppleVisionSaliencyOptions {
    &mut self.objectness_saliency
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn horizon(&self) -> AppleVisionHorizonOptions {
    self.horizon
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn horizon_mut(&mut self) -> &mut AppleVisionHorizonOptions {
    &mut self.horizon
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn document_segments(&self) -> AppleVisionDocumentSegmentationOptions {
    self.document_segments
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn document_segments_mut(&mut self) -> &mut AppleVisionDocumentSegmentationOptions {
    &mut self.document_segments
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn aesthetics(&self) -> AppleVisionAestheticsOptions {
    self.aesthetics
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn aesthetics_mut(&mut self) -> &mut AppleVisionAestheticsOptions {
    &mut self.aesthetics
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn feature_print(&self) -> AppleVisionFeaturePrintOptions {
    self.feature_print
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn feature_print_mut(&mut self) -> &mut AppleVisionFeaturePrintOptions {
    &mut self.feature_print
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn person_instance_masks(&self) -> AppleVisionPersonInstanceMaskOptions {
    self.person_instance_masks
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn person_instance_masks_mut(&mut self) -> &mut AppleVisionPersonInstanceMaskOptions {
    &mut self.person_instance_masks
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn person_segmentation_masks(&self) -> AppleVisionPersonSegmentationOptions {
    self.person_segmentation_masks
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn person_segmentation_masks_mut(
    &mut self,
  ) -> &mut AppleVisionPersonSegmentationOptions {
    &mut self.person_segmentation_masks
  }
}
