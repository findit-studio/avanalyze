//! Off Apple the entry points are unit stubs holding no Objective-C
//! object, and their `Debug` renders the type name alone. This pins
//! that shape, so a stub cannot grow a field whose formatting is not
//! pure Rust without this law noticing; the Apple law is `debug_pure`.

use crate::{
  AnimalPoser, BarcodeDetector, BodyPoser, FaceDetector, FaceLandmarker, HandPoser, PersonMasker,
  TextRecognizer, VisionAnalyzer,
};

#[test]
fn stub_debug_is_the_type_name_alone() {
  assert_eq!(format!("{:?}", VisionAnalyzer), "VisionAnalyzer");
  assert_eq!(format!("{:?}", TextRecognizer), "TextRecognizer");
  assert_eq!(format!("{:?}", BarcodeDetector), "BarcodeDetector");
  assert_eq!(format!("{:?}", FaceDetector), "FaceDetector");
  assert_eq!(format!("{:?}", FaceLandmarker), "FaceLandmarker");
  assert_eq!(format!("{:?}", BodyPoser), "BodyPoser");
  assert_eq!(format!("{:?}", HandPoser), "HandPoser");
  assert_eq!(format!("{:?}", AnimalPoser), "AnimalPoser");
  assert_eq!(format!("{:?}", PersonMasker), "PersonMasker");
}
