//! The crate's system floor, through the test seam: a simulated system
//! version for the calling thread.
//!
//! Every system this suite runs on is at or above the floor, so the
//! refusal below it is reached only by simulation. The seam replaces the
//! system's answer, not the check: each entry point's constructor still
//! makes its own call to the floor check, first, and the check compares
//! the simulated version against the floor the crate is written with.

use crate::{
  AnalyzeError, AnalyzeErrorKind, AnalyzeOptions, AnimalPoser, AppleVisionAnimalPoseOptions,
  AppleVisionBarcodeOptions, AppleVisionBodyPoserOptions, AppleVisionFaceLandmarkOptions,
  AppleVisionFaceOptions, AppleVisionHandPoseOptions, AppleVisionPersonMaskerOptions,
  AppleVisionTextOptions, BarcodeDetector, BodyPoser, FaceDetector, FaceLandmarker, HandPoser,
  PersonMasker, SYSTEM_FLOOR, TextRecognizer, VisionAnalyzer,
  floor::{FLOOR_VERSION, require_floor, seam},
};

/// The refusal `site` makes below the floor.
fn refusal(site: &str) -> String {
  format!("{site}: this system is older than {SYSTEM_FLOOR}, the oldest avanalyze is built on")
}

/// `open` refuses below the floor with the floor's refusal, naming `site`,
/// and is admitted at the floor.
///
/// Admitted means it builds its requests on this host, which every host
/// this suite runs on can.
fn refuses_below_and_is_admitted_at<T>(site: &str, open: impl Fn() -> Result<T, AnalyzeError>) {
  {
    let _below = seam::simulate(14, 9);
    let Err(refused) = open() else {
      panic!("{site} was admitted at 14.9, below the floor");
    };
    assert_eq!(refused.kind(), AnalyzeErrorKind::BelowFloor, "{site}");
    assert_eq!(refused.message(), refusal(site));
  }
  let _at = seam::simulate(15, 0);
  if let Err(refused) = open() {
    panic!("{site} refused at 15.0, the floor: {refused}");
  }
}

#[test]
fn the_floor_is_macos_15_ios_18_tvos_18_visionos_2() {
  assert_eq!(
    SYSTEM_FLOOR,
    "macOS 15.0, iOS 18.0, tvOS 18.0 or visionOS 2.0"
  );
  // The suite runs on macOS, whose floor the check compares against.
  assert_eq!(FLOOR_VERSION, (15, 0));
}

/// The check admits from the floor's own version up, and refuses below
/// it, naming the entry point and the floor.
#[test]
fn the_floor_check_admits_from_15_0_up() {
  for (major, minor) in [(0, 0), (10, 15), (13, 0), (14, 0), (14, 9)] {
    let _simulated = seam::simulate(major, minor);
    let refused = require_floor("Site::new").expect_err("below the floor");
    assert_eq!(
      refused.kind(),
      AnalyzeErrorKind::BelowFloor,
      "{major}.{minor}"
    );
    assert_eq!(refused.message(), refusal("Site::new"));
  }
  for (major, minor) in [(15, 0), (15, 1), (16, 0), (27, 0)] {
    let _simulated = seam::simulate(major, minor);
    require_floor("Site::new").unwrap_or_else(|refused| panic!("{major}.{minor}: {refused}"));
  }
}

/// A simulation ends with its guard, and nests: the system's own answer
/// comes back, and every system this suite runs on is at the floor.
#[test]
fn a_simulation_ends_with_its_guard() {
  {
    let _outer = seam::simulate(14, 0);
    {
      let _inner = seam::simulate(15, 0);
      require_floor("Site::new").expect("the inner simulation is at the floor");
    }
    require_floor("Site::new").expect_err("the outer simulation is back");
  }
  require_floor("Site::new").expect("the system itself is at the floor");
}

#[test]
fn the_vision_analyzer_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("VisionAnalyzer::new", || {
    VisionAnalyzer::new(&AnalyzeOptions::new())
  });
}

#[test]
fn the_text_recognizer_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("TextRecognizer::new", || {
    TextRecognizer::new(&AppleVisionTextOptions::new())
  });
}

#[test]
fn the_barcode_detector_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("BarcodeDetector::new", || {
    BarcodeDetector::new(&AppleVisionBarcodeOptions::new())
  });
}

#[test]
fn the_face_detector_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("FaceDetector::new", || {
    FaceDetector::new(&AppleVisionFaceOptions::new())
  });
}

#[test]
fn the_face_landmarker_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("FaceLandmarker::new", || {
    FaceLandmarker::new(&AppleVisionFaceLandmarkOptions::new())
  });
}

#[test]
fn the_body_poser_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("BodyPoser::new", || {
    BodyPoser::new(&AppleVisionBodyPoserOptions::new())
  });
}

#[test]
fn the_hand_poser_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("HandPoser::new", || {
    HandPoser::new(&AppleVisionHandPoseOptions::new())
  });
}

#[test]
fn the_animal_poser_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("AnimalPoser::new", || {
    AnimalPoser::new(&AppleVisionAnimalPoseOptions::new())
  });
}

#[test]
fn the_person_masker_refuses_below_the_floor() {
  refuses_below_and_is_admitted_at("PersonMasker::new", || {
    PersonMasker::new(&AppleVisionPersonMaskerOptions::new())
  });
}
