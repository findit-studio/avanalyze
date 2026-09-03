#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

//! Apple Vision.framework, wrapped so it only ever *produces*.
//!
//! Nine entry points, each owning exactly the Vision requests its own
//! capability needs:
//!
//! | Entry point | Vision requests | Output |
//! |---|---|---|
//! | [`VisionAnalyzer`] | 8 | [`Analysis`] — the eight cheap slots, one batch |
//! | [`TextRecognizer`] | 1 | `Vec<impl TextDetection>` |
//! | [`BarcodeDetector`] | 1 | `Vec<impl BarcodeDetection>` |
//! | [`FaceDetector`] | 3 | `Vec<impl FaceDetection>` — rectangles first, then capture quality and five keypoints fed those same observations |
//! | [`FaceLandmarker`] | 1 | `Vec<impl FaceLandmarksDetection>` — all thirteen regions |
//! | [`BodyPoser`] | 2 | `Vec<impl BodyPoseDetection>` / `Vec<impl BodyPose3DDetection>` |
//! | [`HandPoser`] | 1 | `Vec<impl HandPoseDetection>` |
//! | [`AnimalPoser`] | 1 | `Vec<impl BodyPoseDetection>` |
//! | [`PersonMasker`] | 2 | `Vec<impl PersonInstanceMaskDetection>` / `Vec<impl PersonSegmentationMask>` |
//!
//! Construct only what you use: a consumer that wants text builds a
//! [`TextRecognizer`], names one output type, and loads exactly one
//! Vision model. The engine mints no identifiers, assembles no
//! aggregate, and depends on no schema crate; Vision.framework is
//! stateless per-request, so workers run in parallel — one entry-point
//! instance per worker thread, never shared.
//!
//! # Two doors, one engine
//!
//! Every method in that table has a twin taking a [`PixelPlane`] —
//! pixels the caller has already decoded — in place of encoded JPEG
//! bytes: `analyze_keyframe_pixels`, `recognize_pixels`,
//! `detect_pixels`, and so on. A caller that holds decoded frames
//! should use them. The JPEG door makes it encode a picture it already
//! has so that Vision can decode it again, and a pipeline running four
//! detectors over one frame pays for that round trip four times.
//!
//! Neither door replaces the other, and they differ in what is handed
//! in and in nothing else — same requests, same options, same ceilings,
//! same per-detector degradation, same output vocabulary — because an
//! entry point's two methods are one body reached two ways.

pub use analysis::*;
pub use analyzer::*;
pub use animal_pose::*;
pub use barcode::*;
pub use body_pose::*;
pub use contract::*;
pub use error::*;
pub use face::*;
pub use face_landmarks::*;
pub use hand_pose::*;
pub use options::*;
pub use person_mask::*;
pub use plane::*;
pub use text::*;

mod analysis;
mod analyzer;
mod animal_pose;
mod barcode;
mod body_pose;
pub mod conformance;
mod contract;
mod error;
mod face;
mod face_landmarks;
mod hand_pose;
mod options;
mod person_mask;
mod plane;
mod text;

#[cfg(target_vendor = "apple")]
mod ffi;

#[cfg(test)]
mod tests;
