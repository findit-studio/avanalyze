//! The revision readers: each producer's named getters and `Display`
//! agree with the constants `setRevision` pinned it at.

use crate::{
  AnalyzeOptions, AppleVisionBarcodeOptions, AppleVisionBodyPoserOptions, AppleVisionFaceOptions,
  AppleVisionPersonMaskerOptions, AppleVisionTextOptions, BarcodeDetector, BodyPoser, FaceDetector,
  PersonMasker, TextRecognizer, VisionAnalyzer,
};

/// [`VisionAnalyzer::revisions`]'s eight named getters and its
/// `Display` both read back the analyzer's own `setRevision` calls, so
/// all three must agree.
#[test]
fn vision_analyzer_revisions_match_the_pinned_constants() {
  let analyzer = VisionAnalyzer::new(&AnalyzeOptions::new())
    .expect("VisionAnalyzer::new builds its Vision requests on this host");
  let revisions = analyzer.revisions();

  assert_eq!(revisions.classify(), 2);
  assert_eq!(revisions.human_rectangles(), 2);
  assert_eq!(revisions.animals(), 2);
  assert_eq!(revisions.attention_saliency(), 2);
  assert_eq!(revisions.objectness_saliency(), 2);
  assert_eq!(revisions.horizon(), 1);
  assert_eq!(revisions.document_segments(), 1);
  assert_eq!(revisions.aesthetics(), 1);

  assert_eq!(
    revisions.to_string(),
    "classify@2,human_rectangles@2,animals@2,attention_saliency@2,objectness_saliency@2,\
     horizon@1,document_segments@1,aesthetics@1"
  );

  let entries: Vec<(&str, usize)> = revisions.into_iter().collect();
  assert_eq!(entries.len(), 8);
  assert_eq!(entries[0], ("classify", 2));
  assert_eq!(entries[7], ("aesthetics", 1));
}

/// A single detector, a single request: [`BarcodeDetector::revision`]
/// is the one `usize` `log_request_revisions` logs.
#[test]
fn barcode_detector_revision_matches_the_pinned_constant() {
  let detector = BarcodeDetector::new(&AppleVisionBarcodeOptions::new())
    .expect("BarcodeDetector::new builds its Vision requests on this host");
  assert_eq!(detector.revision(), 4);
}

/// A single recognizer, a single request: [`TextRecognizer::revision`]
/// is the one `usize` `log_request_revisions` logs.
#[test]
fn text_recognizer_revision_matches_the_pinned_constant() {
  let recognizer = TextRecognizer::new(&AppleVisionTextOptions::new())
    .expect("TextRecognizer::new builds its Vision requests on this host");
  assert_eq!(recognizer.revision(), 3);
}

/// [`FaceDetector::revisions`]'s three named getters and its `Display`
/// both read back [`FaceDetector::new`]'s own `setRevision` calls.
#[test]
fn face_detector_revisions_match_the_pinned_constants() {
  let detector = FaceDetector::new(&AppleVisionFaceOptions::new())
    .expect("FaceDetector::new builds its Vision requests on this host");
  let revisions = detector.revisions();

  assert_eq!(revisions.face_rectangles(), 3);
  assert_eq!(revisions.face_quality(), 3);
  assert_eq!(revisions.face_landmarks(), 3);
  assert_eq!(
    revisions.to_string(),
    "face_rectangles@3,face_quality@3,face_landmarks@3"
  );
}

/// [`BodyPoser::revisions`]'s two named getters and its `Display` both
/// read back [`BodyPoser::new`]'s own `setRevision` calls.
#[test]
fn body_poser_revisions_match_the_pinned_constants() {
  let poser = BodyPoser::new(&AppleVisionBodyPoserOptions::new())
    .expect("BodyPoser::new builds its Vision requests on this host");
  let revisions = poser.revisions();

  assert_eq!(revisions.body_pose(), 1);
  assert_eq!(revisions.body_pose_3d(), 1);
  assert_eq!(revisions.to_string(), "body_pose@1,body_pose_3d@1");
}

/// [`PersonMasker::revisions`]'s two named getters and its `Display`
/// both read back [`PersonMasker::new`]'s own `setRevision` calls.
///
/// This masker owns two requests, exactly as [`BodyPoser`] does. The
/// getters asserted here are its own and the test above asserts a
/// disjoint pair on the same arity, which is the property the producer
/// marker exists to hold: were `Revisions` keyed by request count
/// alone, one of these two rosters would answer to the other's names.
/// A build of this file is the proof — `person_instance_mask` does not
/// resolve on a `BodyPoser` roster, and `body_pose` does not resolve
/// here.
#[test]
fn person_masker_revisions_match_the_pinned_constants() {
  let masker = PersonMasker::new(&AppleVisionPersonMaskerOptions::new())
    .expect("PersonMasker::new builds its Vision requests on this host");
  let revisions = masker.revisions();

  assert_eq!(revisions.person_instance_mask(), 1);
  assert_eq!(revisions.person_segmentation(), 1);
  assert_eq!(
    revisions.to_string(),
    "person_instance_mask@1,person_segmentation@1"
  );
}
