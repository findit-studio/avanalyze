//! `Debug` over an entry point is pure Rust: it sends no Objective-C
//! message.
//!
//! A derived `Debug` over a struct holding a `Retained<VN…Request>`
//! formats that handle, and `NSObject`'s `Debug` sends `-[description]`
//! to it. That message would run outside `guard_native` and
//! `guard_vision_ffi`, and the selector census in the `floor` module
//! would not list it. So every entry point writes its `Debug` by hand,
//! from the type name and the revisions it cached at construction.
//!
//! The law is stated as an exact rendering. A derived impl renders each
//! request through its `-[description]`, which is `<VNClassifyImageRequest:
//! 0x…>`, so no derived impl can produce the strings below.

use crate::{
  AnalyzeOptions, AnimalPoser, AppleVisionAnimalPoseOptions, AppleVisionBarcodeOptions,
  AppleVisionBodyPoserOptions, AppleVisionFaceLandmarkOptions, AppleVisionFaceOptions,
  AppleVisionHandPoseOptions, AppleVisionPersonMaskerOptions, AppleVisionTextOptions,
  BarcodeDetector, BodyPoser, FaceDetector, FaceLandmarker, HandPoser, PersonMasker,
  TextRecognizer, VisionAnalyzer,
};

/// What a `Debug` rendering must never carry: a formatted Objective-C
/// object (`<VNRequest: 0x…>`), a `Retained` wrapper, or an address.
fn assert_no_objc_text(rendered: &str) {
  for forbidden in ["Retained", "VN", "NS", "0x", "<", ">"] {
    assert!(
      !rendered.contains(forbidden),
      "{rendered:?} contains {forbidden:?}: an Objective-C object was formatted"
    );
  }
}

#[test]
fn vision_analyzer_debug_is_its_name_and_its_revisions() {
  let analyzer = VisionAnalyzer::new(&AnalyzeOptions::new())
    .expect("VisionAnalyzer::new builds its Vision requests on this host");
  let rendered = format!("{analyzer:?}");
  assert_eq!(
    rendered,
    format!(
      "VisionAnalyzer {{ revisions: {:?}, .. }}",
      analyzer.revisions()
    )
  );
  assert_no_objc_text(&rendered);
}

#[test]
fn text_recognizer_debug_is_its_name_and_its_revision() {
  let recognizer = TextRecognizer::new(&AppleVisionTextOptions::new())
    .expect("TextRecognizer::new builds its Vision requests on this host");
  let rendered = format!("{recognizer:?}");
  assert_eq!(
    rendered,
    format!(
      "TextRecognizer {{ revision: {}, .. }}",
      recognizer.revision()
    )
  );
  assert_no_objc_text(&rendered);
}

#[test]
fn barcode_detector_debug_is_its_name_and_its_revision() {
  let detector = BarcodeDetector::new(&AppleVisionBarcodeOptions::new())
    .expect("BarcodeDetector::new builds its Vision requests on this host");
  let rendered = format!("{detector:?}");
  assert_eq!(
    rendered,
    format!(
      "BarcodeDetector {{ revision: {}, .. }}",
      detector.revision()
    )
  );
  assert_no_objc_text(&rendered);
}

#[test]
fn face_detector_debug_is_its_name_and_its_revisions() {
  let detector = FaceDetector::new(&AppleVisionFaceOptions::new())
    .expect("FaceDetector::new builds its Vision requests on this host");
  let rendered = format!("{detector:?}");
  assert_eq!(
    rendered,
    format!(
      "FaceDetector {{ revisions: {:?}, .. }}",
      detector.revisions()
    )
  );
  assert_no_objc_text(&rendered);
}

#[test]
fn body_poser_debug_is_its_name_and_its_revisions() {
  let poser = BodyPoser::new(&AppleVisionBodyPoserOptions::new())
    .expect("BodyPoser::new builds its Vision requests on this host");
  let rendered = format!("{poser:?}");
  assert_eq!(
    rendered,
    format!("BodyPoser {{ revisions: {:?}, .. }}", poser.revisions())
  );
  assert_no_objc_text(&rendered);
}

#[test]
fn person_masker_debug_is_its_name_and_its_revisions() {
  let masker = PersonMasker::new(&AppleVisionPersonMaskerOptions::new())
    .expect("PersonMasker::new builds its Vision requests on this host");
  let rendered = format!("{masker:?}");
  assert_eq!(
    rendered,
    format!("PersonMasker {{ revisions: {:?}, .. }}", masker.revisions())
  );
  assert_no_objc_text(&rendered);
}

/// These three cache no revision, so their rendering is the name alone.
#[test]
fn single_request_entry_points_debug_is_their_name_alone() {
  let landmarker = FaceLandmarker::new(&AppleVisionFaceLandmarkOptions::new())
    .expect("FaceLandmarker::new builds its Vision requests on this host");
  let hand = HandPoser::new(&AppleVisionHandPoseOptions::new())
    .expect("HandPoser::new builds its Vision requests on this host");
  let animal = AnimalPoser::new(&AppleVisionAnimalPoseOptions::new())
    .expect("AnimalPoser::new builds its Vision requests on this host");

  for (name, rendered) in [
    ("FaceLandmarker", format!("{landmarker:?}")),
    ("HandPoser", format!("{hand:?}")),
    ("AnimalPoser", format!("{animal:?}")),
  ] {
    assert_eq!(rendered, format!("{name} {{ .. }}"));
    assert_no_objc_text(&rendered);
  }
}
