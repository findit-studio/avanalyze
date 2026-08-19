mod reference;

#[cfg(target_vendor = "apple")]
mod macos;

use crate::{Analysis, conformance, tests::reference::MediaSchema};

/// The whole point of the dev-dependency: a real, foreign, validating
/// vocabulary — `mediaschema`'s domain aggregates — must satisfy the
/// contract as written, with no adapter type in between.
///
/// If a seat's argument order or domain ever drifts from what a real
/// implementation expects, this fails before the engine ships it.
#[test]
fn reference_vocabulary_satisfies_the_contract() {
  conformance::assert_contract::<MediaSchema>();
}

/// `mediaschema` validates, so it must also pass the optional refusal
/// family — including the bow-tie quad that catches a swapped
/// document-corner argument order.
#[test]
fn reference_vocabulary_refuses_invalid_input() {
  conformance::assert_refuses_invalid::<MediaSchema>();
}

/// A fresh [`Analysis`] is genuinely empty: no fabricated sentinels,
/// nothing for a caller to mistake for a detection.
#[test]
fn analysis_starts_empty() {
  let analysis: Analysis<MediaSchema> = Analysis::new();
  assert!(analysis.classifications().is_empty());
  assert!(analysis.faces().is_empty());
  assert!(analysis.person_segmentation_masks().is_empty());
  assert!(analysis.horizon().is_none());
  assert!(analysis.aesthetics().is_none());
}

/// The three ways into a slot agree, and taking a slot through its
/// exclusive accessor moves the values out without cloning.
#[test]
fn analysis_slots_round_trip_through_every_setter() {
  use mediaschema::domain::aggregates::video as ms;

  let detection = ms::Detection::try_new("cat", 0.75).expect("in-range detection");
  let bbox = ms::BoundingBox::try_new(0.1, 0.2, 0.3, 0.4).expect("in-range box");
  let region = ms::SaliencyRegion::try_new(bbox, 0.5).expect("in-range region");

  let mut analysis: Analysis<MediaSchema> = Analysis::new().with_classifications(vec![detection]);
  assert_eq!(analysis.classifications().len(), 1);

  analysis.set_attention_saliency(vec![region]);
  assert_eq!(analysis.attention_saliency().len(), 1);

  let taken = core::mem::take(analysis.attention_saliency_mut());
  assert_eq!(taken.len(), 1);
  assert!(
    analysis.attention_saliency().is_empty(),
    "the exclusive accessor is the consumption path — taking a slot leaves it empty"
  );

  assert!(
    analysis
      .with_aesthetics(Some(ms::Aesthetics::new(0.5, false)))
      .aesthetics()
      .is_some()
  );
}

/// Off Apple there is no Vision framework, so every call reports the
/// platform error rather than pretending to have analysed anything.
#[cfg(not(target_vendor = "apple"))]
#[test]
fn non_macos_stub_reports_unavailable() {
  use crate::{AnalyzeErrorKind, AnalyzeOptions, VisionAnalyzer};

  let options = AnalyzeOptions::new();
  let analyzer = VisionAnalyzer::new(&options);
  let err = analyzer
    .analyze_keyframe::<MediaSchema>(&[], &options)
    .expect_err("the stub must refuse");
  assert_eq!(err.kind(), AnalyzeErrorKind::Unsupported);
  assert!(err.message().contains("macOS"));
}
