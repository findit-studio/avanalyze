//! Animal body pose: its own entry point, its own request.
//!
//! The output trait is [`BodyPoseDetection`] — an animal pose has the
//! same payload as a human 2-D pose — but its joint roster is Apple's
//! animal skeleton, so a vocabulary is free to name a joint type here
//! that it names nowhere else.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::BodyPoseJoint;
#[cfg(target_vendor = "apple")]
use crate::ffi::{
  ImageSource, MAX_POSE_JOINTS, MAX_VISION_RESULTS_PER_FRAME, PoseBudget, PoseJoints,
  ffi_nsstring_to_smolstr, guard_vision_ffi, pose_bbox_from_joint_bounds, read_pose_joints,
  run_requests, sanitize_confidence, vision_point_to_normalized,
};
use crate::{AnalyzeError, AppleVisionAnimalPoseOptions, BodyPoseDetection, PixelPlane};

/// Apple Vision animal body pose — one per worker thread.
///
/// Owns exactly one Vision request; constructing an [`AnimalPoser`]
/// loads no human-pose, face or mask model.
///
/// The retained `VNRequest` carries per-call state across
/// `performRequests` / `results()`, so a poser is not safe to share
/// across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct AnimalPoser {
  request: Retained<VNDetectAnimalBodyPoseRequest>,
}

#[cfg(target_vendor = "apple")]
impl AnimalPoser {
  /// Creates a poser holding the animal-body-pose request at its
  /// pinned revision.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// the request object, so every gate is read per call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionAnimalPoseOptions) -> Self {
    let request = unsafe {
      let request = VNDetectAnimalBodyPoseRequest::new();
      request.setRevision(VNDetectAnimalBodyPoseRequestRevision1);
      request
    };
    Self { request }
  }

  /// Logs the pinned revision of the animal-body-pose request.
  ///
  /// A revision drift changes the joint roster **silently** — same
  /// API, different skeleton.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        animal_body_pose_rev = self.request.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Detects animal body poses in `jpeg_data`.
  pub fn detect<P: BodyPoseDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionAnimalPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    self.detect_on::<P>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Detects animal body poses in already-decoded `pixels`.
  ///
  /// [`detect`](Self::detect) reached without the encode: same request,
  /// same options, same output.
  pub fn detect_pixels<P: BodyPoseDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionAnimalPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    self.detect_on::<P>(ImageSource::Plane(pixels), options)
  }

  /// The one detection body both doors reach.
  fn detect_on<P: BodyPoseDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionAnimalPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.request.clone())] };
    run_requests(source, &requests, Vec::new(), || {
      guard_vision_ffi("animal_body_pose", Vec::new(), || {
        self.extract::<P>(options)
      })
    })
  }

  /// A pose is emitted whole or not at all: when the call's cumulative
  /// [`PoseBudget`] cannot cover the pose in hand the extraction stops
  /// there, and no pose is ever truncated to fit what is left.
  fn extract<P: BodyPoseDetection>(&self, options: &AppleVisionAnimalPoseOptions) -> Vec<P> {
    let Some(results) = (unsafe { self.request.results() }) else {
      return Vec::new();
    };
    let Some(group_name) = (unsafe { VNAnimalBodyPoseObservationJointsGroupNameAll }) else {
      return Vec::new();
    };

    let mut budget = PoseBudget::new();
    let mut body_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Ok(points_by_joint) =
        (unsafe { obs.recognizedPointsForJointsGroupName_error(group_name) })
      else {
        continue;
      };

      // Read the joint dictionary bounded, and pair by keyed lookup.
      // Both the allocation and the enumeration work are capped at
      // `MAX_POSE_JOINTS` regardless of what the dictionary claims
      // about itself, and a dictionary whose count disagrees with what
      // it enumerates is refused outright — dropping the pose entirely,
      // the same contract the old joint-count guard had. The walk
      // charges the call's attempt budget entry by entry as it goes, so
      // a refused read has still paid for what it enumerated and the
      // rejection path is bounded too; once that budget is spent the
      // read comes back `Exhausted` and the extraction stops.
      let pairs = match read_pose_joints(&points_by_joint, MAX_POSE_JOINTS, &mut budget) {
        PoseJoints::Read(pairs) => pairs,
        PoseJoints::Malformed => continue,
        PoseJoints::Exhausted => break,
      };
      let mut joints = Vec::with_capacity(pairs.len());
      let mut name_bytes: usize = 0;
      let mut min_x = f32::INFINITY;
      let mut min_y = f32::INFINITY;
      let mut max_x = f32::NEG_INFINITY;
      let mut max_y = f32::NEG_INFINITY;

      for (joint_name, point) in pairs {
        let Some(name) = ffi_nsstring_to_smolstr(&joint_name) else {
          continue;
        };
        if name.is_empty() {
          continue;
        }

        // Vision normalized points are lower-left origin; flip y for
        // the top-left schema convention. A non-finite raw coordinate
        // is dropped at the source — partial-joint animal-pose lists
        // are valid so we skip only this joint.
        let Some((x, y)) = vision_point_to_normalized(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          options.min_joint_confidence(),
        ) else {
          continue;
        };

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);

        let Ok(joint) = P::Joint::try_new(&name, x, y, confidence) else {
          continue;
        };
        name_bytes = name_bytes.saturating_add(name.len());
        joints.push(joint);
      }

      // Charged before the emptiness and bbox gates below, so the
      // budget reflects the joints this pose actually walked and
      // retained rather than only those a later gate let through.
      if !budget.admit_pose(joints.len(), name_bytes) {
        break;
      }

      if joints.is_empty() {
        continue;
      }

      let Some(bbox) = pose_bbox_from_joint_bounds(min_x, min_y, max_x, max_y) else {
        continue;
      };
      let Some(pose_confidence) = sanitize_confidence(unsafe { obs.confidence() }, 0.0) else {
        continue;
      };

      joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
      if let Ok(pose) = P::try_new(bbox, pose_confidence, joints) {
        body_poses.push(pose);
      }
    }

    body_poses
  }
}

/// Non-macOS stub for [`AnimalPoser`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct AnimalPoser;

#[cfg(not(target_vendor = "apple"))]
impl AnimalPoser {
  /// Constructs a non-macOS stub poser. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionAnimalPoseOptions) -> Self {
    Self
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect<P: BodyPoseDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionAnimalPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_pixels<P: BodyPoseDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionAnimalPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }
}
