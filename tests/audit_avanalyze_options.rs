//! AUDIT: Options exhaustive coverage (R26-R27)
//!
//! Tests every Options struct: defaults, builder, setter, accessor,
//! const fn, Copy/Clone, Debug, and ServiceOptions composition.

#![allow(unused_imports)]

use avanalyze::*;

// ===== R26: Classification =====

#[test]
fn classification_defaults() {
  let o = AppleVisionClassificationOptions::new();
  assert_eq!(o.min_confidence(), 0.3);
  assert_eq!(o.max_results(), 12);
}

#[test]
fn classification_builder_chain() {
  let o = AppleVisionClassificationOptions::new()
    .with_min_confidence(0.5)
    .with_max_results(5);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.max_results(), 5);
}

#[test]
fn classification_setter_chain() {
  let mut o = AppleVisionClassificationOptions::new();
  o.set_min_confidence(0.8).set_max_results(1);
  assert_eq!(o.min_confidence(), 0.8);
  assert_eq!(o.max_results(), 1);
}

#[test]
fn classification_copy_clone_debug() {
  let o = AppleVisionClassificationOptions::new();
  let o2 = o;
  assert_eq!(o.min_confidence(), o2.min_confidence());
  let _ = format!("{o:?}");
}

#[test]
fn classification_default_trait() {
  let o = AppleVisionClassificationOptions::default();
  assert_eq!(o.min_confidence(), 0.3);
  assert_eq!(o.max_results(), 12);
}

// ===== R26: Animal =====

#[test]
fn animal_defaults() {
  let o = AppleVisionAnimalOptions::new();
  assert_eq!(o.min_confidence(), 0.3);
}

#[test]
fn animal_builder() {
  let o = AppleVisionAnimalOptions::new().with_min_confidence(0.9);
  assert_eq!(o.min_confidence(), 0.9);
}

#[test]
fn animal_default_trait() {
  let o = AppleVisionAnimalOptions::default();
  assert_eq!(o.min_confidence(), 0.3);
}

// ===== R26: Text =====

#[test]
fn text_defaults() {
  let o = AppleVisionTextOptions::new();
  assert_eq!(o.min_text_len(), 1);
  assert_eq!(o.max_candidates_per_observation(), 1);
}

#[test]
fn text_builder() {
  let o = AppleVisionTextOptions::new()
    .with_min_text_len(3)
    .with_max_candidates_per_observation(5);
  assert_eq!(o.min_text_len(), 3);
  assert_eq!(o.max_candidates_per_observation(), 5);
}

#[test]
fn text_setter() {
  let mut o = AppleVisionTextOptions::new();
  o.set_min_text_len(10).set_max_candidates_per_observation(3);
  assert_eq!(o.min_text_len(), 10);
  assert_eq!(o.max_candidates_per_observation(), 3);
}

// ===== R26: BodyPose =====

#[test]
fn body_pose_defaults() {
  let o = AppleVisionBodyPoseOptions::new();
  assert_eq!(o.min_joint_confidence(), 0.1);
}

#[test]
fn body_pose_builder() {
  let o = AppleVisionBodyPoseOptions::new().with_min_joint_confidence(0.5);
  assert_eq!(o.min_joint_confidence(), 0.5);
}

// ===== R26: HandPose =====

#[test]
fn hand_pose_defaults() {
  let o = AppleVisionHandPoseOptions::new();
  assert_eq!(o.min_joint_confidence(), 0.1);
  assert_eq!(o.maximum_hand_count(), 2);
}

#[test]
fn hand_pose_builder() {
  let o = AppleVisionHandPoseOptions::new()
    .with_min_joint_confidence(0.5)
    .with_maximum_hand_count(4);
  assert_eq!(o.min_joint_confidence(), 0.5);
  assert_eq!(o.maximum_hand_count(), 4);
}

#[test]
fn hand_pose_setter() {
  let mut o = AppleVisionHandPoseOptions::new();
  o.set_min_joint_confidence(0.9).set_maximum_hand_count(6);
  assert_eq!(o.min_joint_confidence(), 0.9);
  assert_eq!(o.maximum_hand_count(), 6);
}

// ===== R26: AnimalPose =====

#[test]
fn animal_pose_defaults() {
  let o = AppleVisionAnimalPoseOptions::new();
  assert_eq!(o.min_joint_confidence(), 0.1);
}

#[test]
fn animal_pose_builder() {
  let o = AppleVisionAnimalPoseOptions::new().with_min_joint_confidence(0.7);
  assert_eq!(o.min_joint_confidence(), 0.7);
}

// ===== R26: BodyPose3D =====

#[test]
fn body_pose_3d_defaults() {
  let o = AppleVisionBodyPose3DOptions::new();
  assert_eq!(o.min_joint_confidence(), 0.1);
}

#[test]
fn body_pose_3d_builder() {
  let o = AppleVisionBodyPose3DOptions::new().with_min_joint_confidence(0.3);
  assert_eq!(o.min_joint_confidence(), 0.3);
}

// ===== R26: FaceCapture =====

#[test]
fn face_capture_defaults() {
  let o = AppleVisionFaceCaptureOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
  assert_eq!(o.min_capture_quality(), 0.1);
}

#[test]
fn face_capture_builder() {
  let o = AppleVisionFaceCaptureOptions::new()
    .with_min_confidence(0.5)
    .with_min_capture_quality(0.7);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.min_capture_quality(), 0.7);
}

#[test]
fn face_capture_setter() {
  let mut o = AppleVisionFaceCaptureOptions::new();
  o.set_min_confidence(0.9).set_min_capture_quality(0.8);
  assert_eq!(o.min_confidence(), 0.9);
  assert_eq!(o.min_capture_quality(), 0.8);
}

// ===== R26: FaceRectangle =====

#[test]
fn face_rectangle_defaults() {
  let o = AppleVisionFaceRectangleOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
}

#[test]
fn face_rectangle_builder() {
  let o = AppleVisionFaceRectangleOptions::new().with_min_confidence(0.6);
  assert_eq!(o.min_confidence(), 0.6);
}

// ===== R26: FaceLandmark =====

#[test]
fn face_landmark_defaults() {
  let o = AppleVisionFaceLandmarkOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
  assert_eq!(o.min_region_count(), 1);
}

#[test]
fn face_landmark_builder() {
  let o = AppleVisionFaceLandmarkOptions::new()
    .with_min_confidence(0.5)
    .with_min_region_count(3);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.min_region_count(), 3);
}

#[test]
fn face_landmark_setter() {
  let mut o = AppleVisionFaceLandmarkOptions::new();
  o.set_min_confidence(0.8).set_min_region_count(5);
  assert_eq!(o.min_confidence(), 0.8);
  assert_eq!(o.min_region_count(), 5);
}

// ===== R26: HumanSubject =====

#[test]
fn human_subject_defaults() {
  let o = AppleVisionHumanSubjectOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
}

#[test]
fn human_subject_builder() {
  let o = AppleVisionHumanSubjectOptions::new().with_min_confidence(0.5);
  assert_eq!(o.min_confidence(), 0.5);
}

// ===== R26: Barcode =====

#[test]
fn barcode_defaults() {
  let o = AppleVisionBarcodeOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
  assert_eq!(o.min_payload_len(), 1);
}

#[test]
fn barcode_builder() {
  let o = AppleVisionBarcodeOptions::new()
    .with_min_confidence(0.5)
    .with_min_payload_len(3);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.min_payload_len(), 3);
}

#[test]
fn barcode_setter() {
  let mut o = AppleVisionBarcodeOptions::new();
  o.set_min_confidence(0.9).set_min_payload_len(10);
  assert_eq!(o.min_confidence(), 0.9);
  assert_eq!(o.min_payload_len(), 10);
}

// ===== R26: Saliency =====

#[test]
fn saliency_defaults() {
  let o = AppleVisionSaliencyOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
  assert_eq!(o.max_regions(), 16);
}

#[test]
fn saliency_builder() {
  let o = AppleVisionSaliencyOptions::new()
    .with_min_confidence(0.5)
    .with_max_regions(32);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.max_regions(), 32);
}

#[test]
fn saliency_setter() {
  let mut o = AppleVisionSaliencyOptions::new();
  o.set_min_confidence(0.7).set_max_regions(64);
  assert_eq!(o.min_confidence(), 0.7);
  assert_eq!(o.max_regions(), 64);
}

// ===== R26: Horizon =====

#[test]
fn horizon_defaults() {
  let o = AppleVisionHorizonOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
}

#[test]
fn horizon_builder() {
  let o = AppleVisionHorizonOptions::new().with_min_confidence(0.5);
  assert_eq!(o.min_confidence(), 0.5);
}

// ===== R26: DocumentSegmentation =====

#[test]
fn document_segmentation_defaults() {
  let o = AppleVisionDocumentSegmentationOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
  assert_eq!(o.max_segments(), 16);
}

#[test]
fn document_segmentation_builder() {
  let o = AppleVisionDocumentSegmentationOptions::new()
    .with_min_confidence(0.5)
    .with_max_segments(32);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.max_segments(), 32);
}

// ===== R26: Aesthetics =====

#[test]
fn aesthetics_defaults() {
  let o = AppleVisionAestheticsOptions::new();
  assert_eq!(o.min_overall_score(), -1.0);
}

#[test]
fn aesthetics_builder() {
  let o = AppleVisionAestheticsOptions::new().with_min_overall_score(0.5);
  assert_eq!(o.min_overall_score(), 0.5);
}

// ===== R26: PersonInstanceMask =====

#[test]
fn person_instance_mask_defaults() {
  let o = AppleVisionPersonInstanceMaskOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
  assert_eq!(o.max_instances_per_observation(), 16);
}

#[test]
fn person_instance_mask_builder() {
  let o = AppleVisionPersonInstanceMaskOptions::new()
    .with_min_confidence(0.5)
    .with_max_instances_per_observation(32);
  assert_eq!(o.min_confidence(), 0.5);
  assert_eq!(o.max_instances_per_observation(), 32);
}

#[test]
fn person_instance_mask_setter() {
  let mut o = AppleVisionPersonInstanceMaskOptions::new();
  o.set_min_confidence(0.8)
    .set_max_instances_per_observation(8);
  assert_eq!(o.min_confidence(), 0.8);
  assert_eq!(o.max_instances_per_observation(), 8);
}

// ===== R26: PersonSegmentation =====

#[test]
fn person_segmentation_defaults() {
  let o = AppleVisionPersonSegmentationOptions::new();
  assert_eq!(o.min_confidence(), 0.1);
}

#[test]
fn person_segmentation_builder() {
  let o = AppleVisionPersonSegmentationOptions::new().with_min_confidence(0.7);
  assert_eq!(o.min_confidence(), 0.7);
}

// ===== R27: ServiceOptions composition =====

#[test]
fn service_options_default_num_workers() {
  let o = ServiceOptions::new();
  assert_eq!(o.num_workers(), 1);
}

#[test]
fn service_options_num_workers_zero_coerces_to_one() {
  let o = ServiceOptions::new().with_workers(0);
  assert_eq!(o.num_workers(), 1, "num_workers=0 must coerce to 1");
}

#[test]
fn service_options_set_workers_zero_coerces() {
  let mut o = ServiceOptions::new();
  o.set_workers(0);
  assert_eq!(o.num_workers(), 1);
}

#[test]
fn service_options_num_workers_large() {
  let o = ServiceOptions::new().with_workers(64);
  assert_eq!(o.num_workers(), 64);
}

#[test]
fn service_options_all_sub_option_accessors() {
  let o = ServiceOptions::new();
  // Verify every sub-option accessor returns without panicking
  let _ = o.classifications();
  let _ = o.face_capture();
  let _ = o.face_rectangles();
  let _ = o.face_landmarks();
  let _ = o.human_subjects();
  let _ = o.animals();
  let _ = o.text();
  let _ = o.body_pose();
  let _ = o.hand_pose();
  let _ = o.animal_pose();
  let _ = o.body_pose_3d();
  let _ = o.barcodes();
  let _ = o.attention_saliency();
  let _ = o.objectness_saliency();
  let _ = o.horizon();
  let _ = o.document_segments();
  let _ = o.aesthetics();
  let _ = o.person_instance_masks();
  let _ = o.person_segmentation_masks();
}

#[test]
fn service_options_mut_accessors_modify() {
  let mut o = ServiceOptions::new();
  o.classifications_mut().set_min_confidence(0.9);
  assert_eq!(o.classifications().min_confidence(), 0.9);
  o.face_capture_mut().set_min_capture_quality(0.5);
  assert_eq!(o.face_capture().min_capture_quality(), 0.5);
  o.hand_pose_mut().set_maximum_hand_count(6);
  assert_eq!(o.hand_pose().maximum_hand_count(), 6);
  o.barcodes_mut().set_min_payload_len(5);
  assert_eq!(o.barcodes().min_payload_len(), 5);
  o.attention_saliency_mut().set_max_regions(32);
  assert_eq!(o.attention_saliency().max_regions(), 32);
  o.document_segments_mut().set_max_segments(8);
  assert_eq!(o.document_segments().max_segments(), 8);
  o.person_instance_masks_mut()
    .set_max_instances_per_observation(4);
  assert_eq!(o.person_instance_masks().max_instances_per_observation(), 4);
}

#[test]
fn service_options_clone() {
  let o = ServiceOptions::new().with_workers(4);
  let o2 = o.clone();
  assert_eq!(o.num_workers(), o2.num_workers());
}

#[test]
fn service_options_debug() {
  let o = ServiceOptions::new();
  let dbg = format!("{o:?}");
  assert!(dbg.contains("ServiceOptions"));
}

#[test]
fn service_options_default_trait() {
  let o = ServiceOptions::default();
  assert_eq!(o.num_workers(), 1);
}

// ===== R26: Edge cases for f32 fields =====

#[test]
fn options_accept_nan_values() {
  // Options are passive data - NaN is stored as-is.
  // Sanitization happens at the extractor level, not the options level.
  let o = AppleVisionClassificationOptions::new().with_min_confidence(f32::NAN);
  assert!(o.min_confidence().is_nan());
}

#[test]
fn options_accept_negative_values() {
  let o = AppleVisionClassificationOptions::new().with_min_confidence(-1.0);
  assert_eq!(o.min_confidence(), -1.0);
}

#[test]
fn options_accept_extreme_values() {
  let o = AppleVisionClassificationOptions::new().with_max_results(usize::MAX);
  assert_eq!(o.max_results(), usize::MAX);
}

// ===== R26: Builder returns self correctly =====

#[test]
fn builder_returns_new_object_not_alias() {
  let o1 = AppleVisionClassificationOptions::new();
  let o2 = o1.with_min_confidence(0.99);
  // o1 should be unchanged (builder returns new value)
  assert_eq!(o1.min_confidence(), 0.3);
  assert_eq!(o2.min_confidence(), 0.99);
}

#[test]
fn setter_modifies_in_place() {
  let mut o = AppleVisionClassificationOptions::new();
  o.set_min_confidence(0.99);
  assert_eq!(o.min_confidence(), 0.99);
}
