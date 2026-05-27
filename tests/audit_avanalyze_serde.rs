//! AUDIT: Serde round-trip coverage (R5, R27)
//!
//! Tests serde serialization/deserialization for all Options structs.
//! Only compiles with --features serde.

#[cfg(feature = "serde")]
mod serde_tests {
  use avanalyze::*;

  fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug>(val: &T) {
    let json = serde_json::to_string(val).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    // Re-serialize to check structural equality
    let json2 = serde_json::to_string(&back).expect("re-serialize");
    assert_eq!(json, json2, "serde round-trip must be stable");
    // Also check Debug equality
    assert_eq!(format!("{val:?}"), format!("{back:?}"));
  }

  #[test]
  fn serde_classification_options() {
    roundtrip(&AppleVisionClassificationOptions::new());
  }

  #[test]
  fn serde_animal_options() {
    roundtrip(&AppleVisionAnimalOptions::new());
  }

  #[test]
  fn serde_text_options() {
    roundtrip(&AppleVisionTextOptions::new());
  }

  #[test]
  fn serde_body_pose_options() {
    roundtrip(&AppleVisionBodyPoseOptions::new());
  }

  #[test]
  fn serde_hand_pose_options() {
    roundtrip(&AppleVisionHandPoseOptions::new());
  }

  #[test]
  fn serde_animal_pose_options() {
    roundtrip(&AppleVisionAnimalPoseOptions::new());
  }

  #[test]
  fn serde_body_pose_3d_options() {
    roundtrip(&AppleVisionBodyPose3DOptions::new());
  }

  #[test]
  fn serde_face_capture_options() {
    roundtrip(&AppleVisionFaceCaptureOptions::new());
  }

  #[test]
  fn serde_face_rectangle_options() {
    roundtrip(&AppleVisionFaceRectangleOptions::new());
  }

  #[test]
  fn serde_face_landmark_options() {
    roundtrip(&AppleVisionFaceLandmarkOptions::new());
  }

  #[test]
  fn serde_human_subject_options() {
    roundtrip(&AppleVisionHumanSubjectOptions::new());
  }

  #[test]
  fn serde_barcode_options() {
    roundtrip(&AppleVisionBarcodeOptions::new());
  }

  #[test]
  fn serde_saliency_options() {
    roundtrip(&AppleVisionSaliencyOptions::new());
  }

  #[test]
  fn serde_horizon_options() {
    roundtrip(&AppleVisionHorizonOptions::new());
  }

  #[test]
  fn serde_document_segmentation_options() {
    roundtrip(&AppleVisionDocumentSegmentationOptions::new());
  }

  #[test]
  fn serde_aesthetics_options() {
    roundtrip(&AppleVisionAestheticsOptions::new());
  }

  #[test]
  fn serde_person_instance_mask_options() {
    roundtrip(&AppleVisionPersonInstanceMaskOptions::new());
  }

  #[test]
  fn serde_person_segmentation_options() {
    roundtrip(&AppleVisionPersonSegmentationOptions::new());
  }

  #[test]
  fn serde_service_options() {
    roundtrip(&ServiceOptions::new());
  }

  #[test]
  fn serde_service_options_custom() {
    let o = ServiceOptions::new().with_workers(4);
    roundtrip(&o);
  }

  // ── Serde default deserialization (missing fields) ──────────

  #[test]
  fn serde_service_options_from_empty_json() {
    // All fields should use their serde(default) values
    let o: ServiceOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(o.num_workers(), 1);
    assert_eq!(o.classifications().min_confidence(), 0.3);
    assert_eq!(o.classifications().max_results(), 12);
    assert_eq!(o.hand_pose().maximum_hand_count(), 2);
  }

  #[test]
  fn serde_service_options_from_partial_json() {
    let json = r#"{"num_workers": 8, "classifications": {"min_confidence": 0.5}}"#;
    let o: ServiceOptions = serde_json::from_str(json).expect("partial json");
    assert_eq!(o.num_workers(), 8);
    assert_eq!(o.classifications().min_confidence(), 0.5);
    // max_results should be default
    assert_eq!(o.classifications().max_results(), 12);
    // Other sub-options should be default
    assert_eq!(o.hand_pose().maximum_hand_count(), 2);
    assert_eq!(o.barcodes().min_payload_len(), 1);
  }

  #[test]
  fn serde_num_workers_zero_coerced_to_one() {
    // `{"num_workers": 0}` must produce the same value as `with_workers(0)`
    // (coerced to 1) — a custom `deserialize_with` mirrors the runtime
    // setter contract at the serde boundary.
    let json = r#"{"num_workers": 0}"#;
    let o: ServiceOptions = serde_json::from_str(json).expect("zero workers json");
    assert_eq!(o.num_workers(), 1, "0 must be coerced to 1");
  }

  // ── Serde with extreme values ───────────────────────────────

  #[test]
  fn serde_extreme_confidence_values() {
    let json = r#"{"min_confidence": 0.0}"#;
    let o: AppleVisionClassificationOptions = serde_json::from_str(json).expect("zero confidence");
    assert_eq!(o.min_confidence(), 0.0);
  }

  #[test]
  fn serde_negative_confidence_values() {
    // serde doesn't validate ranges
    let json = r#"{"min_confidence": -1.0}"#;
    let o: AppleVisionClassificationOptions =
      serde_json::from_str(json).expect("negative confidence");
    assert_eq!(o.min_confidence(), -1.0);
  }
}
