//! AUDIT: every options field survives its own absence
//!
//! `AnalyzeOptions` carries `serde(default)` on each sub-options field, so a
//! config that omits a whole section is already fine. This file audits the
//! level below that: a section that is *present but partial*. Deserializing a
//! sub-options struct only reaches its fields' own defaults, so a field
//! without `serde(default = "…")` makes the whole section mandatory-in-full —
//! `{"attention_saliency": {"max_regions": 4}}` fails on the missing
//! `min_confidence` rather than filling it in.
//!
//! Each case asserts the absent fields land on the type's public `DEFAULT_*`
//! constant, which is the same source `new()` reads, so a default that drifts
//! in one place fails here.
//!
//! Only compiles with --features serde.

#[cfg(feature = "serde")]
mod serde_default_tests {
  use avanalyze::*;

  /// Re-serialize and re-parse a default-filled value: the defaults serde
  /// supplied must themselves be round-trippable, not just parseable.
  fn restable<T: serde::Serialize + serde::de::DeserializeOwned + core::fmt::Debug>(val: &T) {
    let json = serde_json::to_string(val).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(format!("{val:?}"), format!("{back:?}"));
  }

  // ── Saliency ────────────────────────────────────────────────

  #[test]
  fn saliency_from_empty_json() {
    let o: AppleVisionSaliencyOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_confidence(),
      AppleVisionSaliencyOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(
      o.max_regions(),
      AppleVisionSaliencyOptions::DEFAULT_MAX_REGIONS
    );
    restable(&o);
  }

  #[test]
  fn saliency_from_partial_json() {
    let o: AppleVisionSaliencyOptions =
      serde_json::from_str(r#"{"max_regions": 4}"#).expect("partial json");
    assert_eq!(o.max_regions(), 4);
    assert_eq!(
      o.min_confidence(),
      AppleVisionSaliencyOptions::DEFAULT_MIN_CONFIDENCE
    );
  }

  // ── Horizon ─────────────────────────────────────────────────

  #[test]
  fn horizon_from_empty_json() {
    // Single-field type: `{}` is its only absence case.
    let o: AppleVisionHorizonOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_confidence(),
      AppleVisionHorizonOptions::DEFAULT_MIN_CONFIDENCE
    );
    restable(&o);
  }

  // ── Document segmentation ───────────────────────────────────

  #[test]
  fn document_segmentation_from_empty_json() {
    let o: AppleVisionDocumentSegmentationOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_confidence(),
      AppleVisionDocumentSegmentationOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(
      o.max_segments(),
      AppleVisionDocumentSegmentationOptions::DEFAULT_MAX_SEGMENTS
    );
    restable(&o);
  }

  #[test]
  fn document_segmentation_from_partial_json() {
    let o: AppleVisionDocumentSegmentationOptions =
      serde_json::from_str(r#"{"max_segments": 3}"#).expect("partial json");
    assert_eq!(o.max_segments(), 3);
    assert_eq!(
      o.min_confidence(),
      AppleVisionDocumentSegmentationOptions::DEFAULT_MIN_CONFIDENCE
    );
  }

  // ── Aesthetics ──────────────────────────────────────────────

  #[test]
  fn aesthetics_from_empty_json() {
    // Single-field type: `{}` is its only absence case. Its default is
    // negative (-1.0, "keep every score"), so a zero-valued stand-in would
    // change which frames survive.
    let o: AppleVisionAestheticsOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_overall_score(),
      AppleVisionAestheticsOptions::DEFAULT_MIN_OVERALL_SCORE
    );
    restable(&o);
  }

  // ── Person instance mask ────────────────────────────────────

  #[test]
  fn person_instance_mask_from_empty_json() {
    let o: AppleVisionPersonInstanceMaskOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_confidence(),
      AppleVisionPersonInstanceMaskOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(
      o.max_instances_per_observation(),
      AppleVisionPersonInstanceMaskOptions::DEFAULT_MAX_INSTANCES_PER_OBSERVATION
    );
    restable(&o);
  }

  #[test]
  fn person_instance_mask_from_partial_json() {
    let o: AppleVisionPersonInstanceMaskOptions =
      serde_json::from_str(r#"{"min_confidence": 0.75}"#).expect("partial json");
    assert_eq!(o.min_confidence(), 0.75);
    assert_eq!(
      o.max_instances_per_observation(),
      AppleVisionPersonInstanceMaskOptions::DEFAULT_MAX_INSTANCES_PER_OBSERVATION
    );
  }

  // ── Person segmentation ─────────────────────────────────────

  #[test]
  fn person_segmentation_from_empty_json() {
    // Single-field type: `{}` is its only absence case.
    let o: AppleVisionPersonSegmentationOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_confidence(),
      AppleVisionPersonSegmentationOptions::DEFAULT_MIN_CONFIDENCE
    );
    restable(&o);
  }

  // ── Face keypoints ──────────────────────────────────────────

  #[test]
  fn face_keypoints_from_empty_json() {
    // Single-field type: `{}` is its only absence case.
    let o: AppleVisionFaceKeypointsOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.min_confidence(),
      AppleVisionFaceKeypointsOptions::DEFAULT_MIN_CONFIDENCE
    );
    restable(&o);
  }

  // ── The composed per-entry sections ─────────────────────────

  #[test]
  fn face_options_from_empty_and_partial_json() {
    let o: AppleVisionFaceOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.rectangles().min_confidence(),
      AppleVisionFaceRectangleOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(
      o.capture().min_capture_quality(),
      AppleVisionFaceCaptureOptions::DEFAULT_MIN_CAPTURE_QUALITY
    );
    assert_eq!(
      o.keypoints().min_confidence(),
      AppleVisionFaceKeypointsOptions::DEFAULT_MIN_CONFIDENCE
    );
    restable(&o);

    let o: AppleVisionFaceOptions =
      serde_json::from_str(r#"{"keypoints": {"min_confidence": 0.42}}"#).expect("partial json");
    assert_eq!(o.keypoints().min_confidence(), 0.42);
    assert_eq!(
      o.rectangles().min_confidence(),
      AppleVisionFaceRectangleOptions::DEFAULT_MIN_CONFIDENCE
    );
  }

  #[test]
  fn body_poser_options_from_empty_and_partial_json() {
    let o: AppleVisionBodyPoserOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.pose_2d().min_joint_confidence(),
      AppleVisionBodyPoseOptions::DEFAULT_MIN_JOINT_CONFIDENCE
    );
    assert_eq!(
      o.pose_3d().min_joint_confidence(),
      AppleVisionBodyPose3DOptions::DEFAULT_MIN_JOINT_CONFIDENCE
    );
    restable(&o);

    let o: AppleVisionBodyPoserOptions =
      serde_json::from_str(r#"{"pose_2d": {"min_joint_confidence": 0.6}}"#).expect("partial json");
    assert_eq!(o.pose_2d().min_joint_confidence(), 0.6);
    assert_eq!(
      o.pose_3d().min_joint_confidence(),
      AppleVisionBodyPose3DOptions::DEFAULT_MIN_JOINT_CONFIDENCE
    );
  }

  #[test]
  fn person_masker_options_from_empty_and_partial_json() {
    let o: AppleVisionPersonMaskerOptions = serde_json::from_str("{}").expect("empty json");
    assert_eq!(
      o.instances().min_confidence(),
      AppleVisionPersonInstanceMaskOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(
      o.segmentation().min_confidence(),
      AppleVisionPersonSegmentationOptions::DEFAULT_MIN_CONFIDENCE
    );
    restable(&o);

    let o: AppleVisionPersonMaskerOptions =
      serde_json::from_str(r#"{"instances": {"min_confidence": 0.75}}"#).expect("partial json");
    assert_eq!(o.instances().min_confidence(), 0.75);
    assert_eq!(
      o.instances().max_instances_per_observation(),
      AppleVisionPersonInstanceMaskOptions::DEFAULT_MAX_INSTANCES_PER_OBSERVATION
    );
    assert_eq!(
      o.segmentation().min_confidence(),
      AppleVisionPersonSegmentationOptions::DEFAULT_MIN_CONFIDENCE
    );
  }

  // ── Through the parent ──────────────────────────────────────

  #[test]
  fn analyze_options_from_partial_sections() {
    // The shape a real config file takes: name one knob per section and let
    // the rest default. Every section named here is one whose fields lacked a
    // serde default, so before the fix each of these lines was a hard parse
    // error rather than a default.
    let json = r#"{
      "attention_saliency": {"max_regions": 4},
      "objectness_saliency": {"min_confidence": 0.5},
      "horizon": {},
      "document_segments": {"max_segments": 3},
      "aesthetics": {}
    }"#;
    let o: AnalyzeOptions = serde_json::from_str(json).expect("partial sections");

    assert_eq!(o.attention_saliency().max_regions(), 4);
    assert_eq!(
      o.attention_saliency().min_confidence(),
      AppleVisionSaliencyOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(o.objectness_saliency().min_confidence(), 0.5);
    assert_eq!(
      o.objectness_saliency().max_regions(),
      AppleVisionSaliencyOptions::DEFAULT_MAX_REGIONS
    );
    assert_eq!(
      o.horizon().min_confidence(),
      AppleVisionHorizonOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(o.document_segments().max_segments(), 3);
    assert_eq!(
      o.document_segments().min_confidence(),
      AppleVisionDocumentSegmentationOptions::DEFAULT_MIN_CONFIDENCE
    );
    assert_eq!(
      o.aesthetics().min_overall_score(),
      AppleVisionAestheticsOptions::DEFAULT_MIN_OVERALL_SCORE
    );

    // Sections the config never mentioned still default, as before.
    assert_eq!(
      o.num_workers(),
      AnalyzeOptions::DEFAULT_NUM_WORKERS,
      "an unnamed scalar still defaults"
    );
    assert_eq!(
      o.classifications().max_results(),
      AppleVisionClassificationOptions::DEFAULT_MAX_RESULTS,
      "an unnamed section still defaults"
    );
  }

  /// A config written against the pre-split `AnalyzeOptions` still
  /// parses, and the eleven sections that moved are ignored — exactly
  /// as any other unknown key is, on this type and on every other
  /// options type in this crate.
  ///
  /// `text`, `face_capture`, `face_rectangles`, `face_landmarks`,
  /// `body_pose`, `hand_pose`, `animal_pose`, `body_pose_3d`,
  /// `barcodes`, `person_instance_masks` and
  /// `person_segmentation_masks` moved to the options type of the entry
  /// point that reads them. `AnalyzeOptions` has no home for them any
  /// more, so a serialized config that still names them parses and
  /// those keys are dropped. That is the contract, asserted here rather
  /// than left to be discovered: every one of this crate's options
  /// types is unknown-field tolerant, so tightening only the parent
  /// would be incoherent, and `deny_unknown_fields` is documented by
  /// serde as incompatible with `#[serde(flatten)]` — it would newly
  /// break any downstream that flattens `AnalyzeOptions` into a larger
  /// config.
  ///
  /// The loud signal for the migration is the API break, not the
  /// parser: all eleven `AnalyzeOptions` accessors (`text()`,
  /// `face_capture()`, …) were REMOVED, so code that read one fails to
  /// compile and names the section it lost, and the `Detections` bundle
  /// went from twenty-one associated types to seven. Callers move the
  /// values to `AppleVisionTextOptions`, `AppleVisionFaceOptions` and
  /// the rest.
  #[test]
  fn pre_split_config_still_parses_and_the_moved_sections_are_dropped() {
    // A 0.4.x config: `num_workers`, three surviving core sections
    // carrying non-default values, and all eleven sections that moved.
    let pre_split = r#"{
      "num_workers": 6,
      "classifications": {"min_confidence": 0.55, "max_results": 3},
      "attention_saliency": {"max_regions": 5},
      "aesthetics": {"min_overall_score": 0.25},
      "text": {"min_text_len": 4, "max_candidates_per_observation": 2},
      "face_capture": {"min_confidence": 0.6, "min_capture_quality": 0.4},
      "face_rectangles": {"min_confidence": 0.65},
      "face_landmarks": {"min_confidence": 0.7, "min_region_count": 3},
      "body_pose": {"min_joint_confidence": 0.55},
      "hand_pose": {"min_joint_confidence": 0.5, "maximum_hand_count": 4},
      "animal_pose": {"min_joint_confidence": 0.45},
      "body_pose_3d": {"min_joint_confidence": 0.35},
      "barcodes": {"min_confidence": 0.8, "min_payload_len": 2},
      "person_instance_masks": {"min_confidence": 0.75, "max_instances_per_observation": 3},
      "person_segmentation_masks": {"min_confidence": 0.85}
    }"#;
    let o: AnalyzeOptions = serde_json::from_str(pre_split).expect("a pre-split config parses");

    // The eleven strays do not disturb the eight that remain.
    assert_eq!(o.num_workers(), 6);
    assert_eq!(o.classifications().min_confidence(), 0.55);
    assert_eq!(o.classifications().max_results(), 3);
    assert_eq!(o.attention_saliency().max_regions(), 5);
    assert_eq!(o.aesthetics().min_overall_score(), 0.25);

    // And they leave nothing behind: the parse equals what the same
    // config MINUS the eleven produces, field by field over everything
    // this type still holds.
    let without_strays = r#"{
      "num_workers": 6,
      "classifications": {"min_confidence": 0.55, "max_results": 3},
      "attention_saliency": {"max_regions": 5},
      "aesthetics": {"min_overall_score": 0.25}
    }"#;
    let clean: AnalyzeOptions =
      serde_json::from_str(without_strays).expect("the same config without the strays");

    assert_eq!(o.num_workers(), clean.num_workers());
    assert_eq!(
      o.classifications().min_confidence(),
      clean.classifications().min_confidence()
    );
    assert_eq!(
      o.classifications().max_results(),
      clean.classifications().max_results()
    );
    assert_eq!(
      o.human_subjects().min_confidence(),
      clean.human_subjects().min_confidence()
    );
    assert_eq!(
      o.animals().min_confidence(),
      clean.animals().min_confidence()
    );
    assert_eq!(
      o.attention_saliency().min_confidence(),
      clean.attention_saliency().min_confidence()
    );
    assert_eq!(
      o.attention_saliency().max_regions(),
      clean.attention_saliency().max_regions()
    );
    assert_eq!(
      o.objectness_saliency().min_confidence(),
      clean.objectness_saliency().min_confidence()
    );
    assert_eq!(
      o.objectness_saliency().max_regions(),
      clean.objectness_saliency().max_regions()
    );
    assert_eq!(
      o.horizon().min_confidence(),
      clean.horizon().min_confidence()
    );
    assert_eq!(
      o.document_segments().min_confidence(),
      clean.document_segments().min_confidence()
    );
    assert_eq!(
      o.document_segments().max_segments(),
      clean.document_segments().max_segments()
    );
    assert_eq!(
      o.aesthetics().min_overall_score(),
      clean.aesthetics().min_overall_score()
    );
  }
}
