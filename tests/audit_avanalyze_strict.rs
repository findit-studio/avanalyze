//! AUDIT: Stricter post-fix verification (Round 2)
//!
//! Targets the serde coercion fix and deeper adversarial tests.

use avanalyze::*;

// ===== Serde num_workers coercion (fix #2) =====

#[cfg(feature = "serde")]
mod serde_coercion_tests {
  use super::*;

  #[test]
  fn serde_zero_workers_coerced_to_one() {
    let o: ServiceOptions = serde_json::from_str(r#"{"num_workers": 0}"#).unwrap();
    assert_eq!(o.num_workers(), 1);
  }

  #[test]
  fn serde_one_workers_unchanged() {
    let o: ServiceOptions = serde_json::from_str(r#"{"num_workers": 1}"#).unwrap();
    assert_eq!(o.num_workers(), 1);
  }

  #[test]
  fn serde_large_workers_unchanged() {
    let o: ServiceOptions = serde_json::from_str(r#"{"num_workers": 256}"#).unwrap();
    assert_eq!(o.num_workers(), 256);
  }

  #[test]
  fn serde_missing_num_workers_defaults_to_one() {
    let o: ServiceOptions = serde_json::from_str(r#"{}"#).unwrap();
    assert_eq!(o.num_workers(), 1);
  }

  #[test]
  fn serde_max_workers_unchanged() {
    let json = format!(r#"{{"num_workers": {}}}"#, usize::MAX);
    let o: ServiceOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(o.num_workers(), usize::MAX);
  }

  #[test]
  fn serde_negative_workers_rejected() {
    let result = serde_json::from_str::<ServiceOptions>(r#"{"num_workers": -1}"#);
    assert!(result.is_err(), "negative usize must be rejected");
  }

  #[test]
  fn serde_workers_roundtrip_preserves_coercion() {
    let o = ServiceOptions::new().with_workers(0);
    assert_eq!(o.num_workers(), 1);
    let json = serde_json::to_string(&o).unwrap();
    let o2: ServiceOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(o2.num_workers(), 1);
  }

  #[test]
  fn serde_float_workers_rejected() {
    let result = serde_json::from_str::<ServiceOptions>(r#"{"num_workers": 1.5}"#);
    assert!(result.is_err(), "float must be rejected for usize");
  }

  #[test]
  fn serde_string_workers_rejected() {
    let result = serde_json::from_str::<ServiceOptions>(r#"{"num_workers": "two"}"#);
    assert!(result.is_err(), "string must be rejected for usize");
  }

  #[test]
  fn serde_all_sub_options_with_custom_values() {
    let json = r#"{
      "num_workers": 4,
      "classifications": {"min_confidence": 0.5, "max_results": 20},
      "face_capture": {"min_confidence": 0.3, "min_capture_quality": 0.6},
      "face_rectangles": {"min_confidence": 0.4},
      "face_landmarks": {"min_confidence": 0.2, "min_region_count": 3},
      "human_subjects": {"min_confidence": 0.5},
      "animals": {"min_confidence": 0.7},
      "text": {"min_text_len": 5, "max_candidates_per_observation": 3},
      "body_pose": {"min_joint_confidence": 0.3},
      "hand_pose": {"min_joint_confidence": 0.4, "maximum_hand_count": 4},
      "animal_pose": {"min_joint_confidence": 0.2},
      "body_pose_3d": {"min_joint_confidence": 0.3},
      "barcodes": {"min_confidence": 0.5, "min_payload_len": 5},
      "attention_saliency": {"min_confidence": 0.3, "max_regions": 32},
      "objectness_saliency": {"min_confidence": 0.2, "max_regions": 16},
      "horizon": {"min_confidence": 0.4},
      "document_segments": {"min_confidence": 0.3, "max_segments": 8},
      "aesthetics": {"min_overall_score": 0.0},
      "person_instance_masks": {"min_confidence": 0.5, "max_instances_per_observation": 8},
      "person_segmentation_masks": {"min_confidence": 0.4}
    }"#;
    let o: ServiceOptions = serde_json::from_str(json).unwrap();
    assert_eq!(o.num_workers(), 4);
    assert_eq!(o.classifications().min_confidence(), 0.5);
    assert_eq!(o.classifications().max_results(), 20);
    assert_eq!(o.face_capture().min_confidence(), 0.3);
    assert_eq!(o.face_capture().min_capture_quality(), 0.6);
    assert_eq!(o.face_landmarks().min_region_count(), 3);
    assert_eq!(o.text().min_text_len(), 5);
    assert_eq!(o.text().max_candidates_per_observation(), 3);
    assert_eq!(o.hand_pose().maximum_hand_count(), 4);
    assert_eq!(o.barcodes().min_payload_len(), 5);
    assert_eq!(o.attention_saliency().max_regions(), 32);
    assert_eq!(o.document_segments().max_segments(), 8);
    assert_eq!(
      o.person_instance_masks().max_instances_per_observation(),
      8
    );
  }
}

// ===== Deeper adversarial tests =====

#[test]
fn builder_does_not_mutate_original() {
  let original = AppleVisionClassificationOptions::new();
  let modified = original.with_min_confidence(0.99).with_max_results(1);
  assert_eq!(original.min_confidence(), 0.3);
  assert_eq!(original.max_results(), 12);
  assert_eq!(modified.min_confidence(), 0.99);
  assert_eq!(modified.max_results(), 1);
}

#[test]
fn setter_modifies_in_place() {
  let mut o = AppleVisionHandPoseOptions::new();
  let ptr_before = &o as *const _;
  o.set_min_joint_confidence(0.5).set_maximum_hand_count(6);
  let ptr_after = &o as *const _;
  assert_eq!(ptr_before, ptr_after, "setter must modify in-place");
}

#[test]
fn mut_accessor_idempotent() {
  let mut o = ServiceOptions::new();
  let ptr1 = o.classifications_mut() as *const _;
  let ptr2 = o.classifications_mut() as *const _;
  assert_eq!(ptr1, ptr2, "mut accessor must return same address");
}

#[test]
fn modifying_one_option_does_not_affect_others() {
  let mut o = ServiceOptions::new();
  let original_barcode = o.barcodes().min_confidence();
  o.classifications_mut().set_min_confidence(0.99);
  assert_eq!(
    o.barcodes().min_confidence(),
    original_barcode,
    "modifying classifications must not affect barcodes"
  );
}

#[test]
fn new_equals_default_for_all_fields() {
  let a = ServiceOptions::new();
  let b = ServiceOptions::default();
  assert_eq!(a.num_workers(), b.num_workers());
  assert_eq!(
    a.classifications().min_confidence(),
    b.classifications().min_confidence()
  );
  assert_eq!(
    a.hand_pose().maximum_hand_count(),
    b.hand_pose().maximum_hand_count()
  );
  assert_eq!(
    a.barcodes().min_payload_len(),
    b.barcodes().min_payload_len()
  );
}

#[test]
fn options_with_zero_confidence() {
  let o = AppleVisionClassificationOptions::new().with_min_confidence(0.0);
  assert_eq!(o.min_confidence(), 0.0);
}

#[test]
fn options_with_one_confidence() {
  let o = AppleVisionClassificationOptions::new().with_min_confidence(1.0);
  assert_eq!(o.min_confidence(), 1.0);
}

#[test]
fn options_with_negative_confidence() {
  let o = AppleVisionClassificationOptions::new().with_min_confidence(-0.5);
  assert_eq!(o.min_confidence(), -0.5);
}

#[test]
fn options_with_subnormal_float() {
  let tiny = f32::from_bits(1);
  let o = AppleVisionClassificationOptions::new().with_min_confidence(tiny);
  assert_eq!(o.min_confidence().to_bits(), 1);
}

// ===== Serde round-trip with all features =====

#[cfg(feature = "serde")]
#[test]
fn serde_roundtrip_service_options_full() {
  let mut opts = ServiceOptions::new().with_workers(8);
  opts.classifications_mut().set_min_confidence(0.7);
  opts.hand_pose_mut().set_maximum_hand_count(4);
  let json = serde_json::to_string_pretty(&opts).unwrap();
  let back: ServiceOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back.num_workers(), 8);
  assert_eq!(back.classifications().min_confidence(), 0.7);
  assert_eq!(back.hand_pose().maximum_hand_count(), 4);
}
