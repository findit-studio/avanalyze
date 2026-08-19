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
      "aesthetics": {},
      "person_instance_masks": {"min_confidence": 0.75},
      "person_segmentation_masks": {}
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
    assert_eq!(o.person_instance_masks().min_confidence(), 0.75);
    assert_eq!(
      o.person_instance_masks().max_instances_per_observation(),
      AppleVisionPersonInstanceMaskOptions::DEFAULT_MAX_INSTANCES_PER_OBSERVATION
    );
    assert_eq!(
      o.person_segmentation_masks().min_confidence(),
      AppleVisionPersonSegmentationOptions::DEFAULT_MIN_CONFIDENCE
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
}
