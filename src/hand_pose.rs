//! Hand pose: its own entry point, its own request, its own trait.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  ImageSource, MAX_POSE_JOINTS, MAX_VISION_RESULTS_PER_FRAME, PoseBudget, PoseJoints,
  ffi_nsstring_to_smolstr, guard_native, guard_vision_ffi, pose_bbox_from_joint_bounds,
  read_pose_joints, run_requests, sanitize_confidence, vision_point_to_normalized,
};
use crate::{AnalyzeError, AppleVisionHandPoseOptions, BodyPoseJoint, BoundingBox, PixelPlane};

/// Apple's documented maximum for
/// `VNDetectHumanHandPoseRequest::setMaximumHandCount(_:)` at
/// revision 1 — the request becomes invalid above this. The pinned
/// request revision in this crate is revision 1; configurations
/// requesting more must clamp at construction time to avoid an
/// Objective-C exception crossing the FFI boundary.
#[cfg(target_vendor = "apple")]
const MAX_HAND_POSE_MAXIMUM_HAND_COUNT: usize = 6;

/// Handedness of a detected hand pose.
///
/// Apple reports handedness as `VNChirality`; anything the framework
/// does not classify as left or right — including variants added by a
/// future OS — maps to [`Chirality::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Chirality {
  /// Handedness was not reported, or was reported as a value this
  /// engine does not recognise.
  #[default]
  Unknown,
  /// Left hand.
  Left,
  /// Right hand.
  Right,
}

/// A hand pose.
///
/// Built from [`BodyPoseJoint`]s of its own kind — Apple's hand joint
/// set is a different vocabulary from the body's — and, like
/// [`BodyPoseDetection`](crate::BodyPoseDetection), carries a box
/// synthesised from the surviving joints.
pub trait HandPoseDetection: Sized {
  /// Why a pose was refused.
  type Error;
  /// The geometry type this pose is built from.
  type BoundingBox: BoundingBox;
  /// The joint type this pose collects.
  type Joint: BodyPoseJoint;

  /// Builds a hand pose.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    chirality: Chirality,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error>;
}

/// Apple Vision hand pose — one per worker thread.
///
/// Owns exactly one Vision request. Alone among this crate's entry
/// points it bakes a knob into the request object:
/// [`maximum_hand_count`](AppleVisionHandPoseOptions::maximum_hand_count)
/// follows the poser, not the call, and is clamped at construction to
/// Apple's revision-1 maximum.
///
/// The retained `VNRequest` carries per-call state across
/// `performRequests` / `results()`, so a poser is not safe to share
/// across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct HandPoser {
  request: Retained<VNDetectHumanHandPoseRequest>,
}

#[cfg(target_vendor = "apple")]
impl HandPoser {
  /// Creates a poser holding the hand-pose request at its pinned
  /// revision, with `maximum_hand_count` baked in.
  ///
  /// The count is clamped to Apple's documented revision-1 maximum:
  /// the request becomes invalid above 6 and would surface as an
  /// Objective-C exception crossing the FFI boundary, so a
  /// stale/misconfigured option value still produces a usable request.
  ///
  /// # Errors
  ///
  /// Building a Vision request loads a model, and a model load is where
  /// Apple's stack raises instead of returning: on a host whose Neural
  /// Engine is denied it throws, and a throw that crosses into Rust
  /// unguarded takes the process down. This refuses with
  /// [`AnalyzeErrorKind::Environment`](crate::AnalyzeErrorKind::Environment)
  /// instead — the constructor is where a whole entry point can still
  /// be declined, before any frame has been handed to it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: &AppleVisionHandPoseOptions) -> Result<Self, AnalyzeError> {
    let request = guard_native("HandPoser::new", || unsafe {
      let request = VNDetectHumanHandPoseRequest::new();
      request.setMaximumHandCount(
        options
          .maximum_hand_count()
          .min(MAX_HAND_POSE_MAXIMUM_HAND_COUNT),
      );
      request.setRevision(VNDetectHumanHandPoseRequestRevision1);
      request
    })?;
    Ok(Self { request })
  }

  /// Logs the pinned revision of the hand-pose request.
  ///
  /// A revision drift changes the joint roster **silently** — same
  /// API, different skeleton.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        hand_pose_rev = self.request.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Detects hand poses in `jpeg_data`.
  ///
  /// `options` is read per call, but
  /// [`HandPoser::new`] already baked
  /// [`maximum_hand_count`](AppleVisionHandPoseOptions::maximum_hand_count)
  /// into the retained request: that one knob follows the poser, not
  /// the call.
  pub fn detect<P: HandPoseDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionHandPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    self.detect_on::<P>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Detects hand poses in already-decoded `pixels`.
  ///
  /// [`detect`](Self::detect) reached without the encode: same request,
  /// same baked knob, same options, same output.
  pub fn detect_pixels<P: HandPoseDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionHandPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    self.detect_on::<P>(ImageSource::Plane(pixels), options)
  }

  /// The one detection body both doors reach.
  fn detect_on<P: HandPoseDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionHandPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.request.clone())] };
    run_requests(source, &requests, Vec::new(), || {
      guard_vision_ffi("hand_pose", Vec::new(), || self.extract::<P>(options))
    })
  }

  /// A pose is emitted whole or not at all: when the call's cumulative
  /// [`PoseBudget`] cannot cover the pose in hand the extraction stops
  /// there, and no pose is ever truncated to fit what is left.
  fn extract<P: HandPoseDetection>(&self, options: &AppleVisionHandPoseOptions) -> Vec<P> {
    let Some(results) = (unsafe { self.request.results() }) else {
      return Vec::new();
    };

    let mut budget = PoseBudget::new();
    let mut hand_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanHandPoseObservationJointsGroupNameAll)
      }) else {
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
        // is dropped at the source — partial-joint hand lists are
        // valid so we skip only this joint.
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
      if let Ok(pose) = P::try_new(
        bbox,
        pose_confidence,
        map_hand_chirality(unsafe { obs.chirality() }),
        joints,
      ) {
        hand_poses.push(pose);
      }
    }

    hand_poses
  }
}

#[cfg(target_vendor = "apple")]
fn map_hand_chirality(chirality: VNChirality) -> Chirality {
  match chirality {
    VNChirality::Left => Chirality::Left,
    VNChirality::Right => Chirality::Right,
    _ => Chirality::Unknown,
  }
}

/// Non-macOS stub for [`HandPoser`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct HandPoser;

#[cfg(not(target_vendor = "apple"))]
impl HandPoser {
  /// Constructs a non-macOS stub poser. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  ///
  /// # Errors
  ///
  /// Never off Apple: there is no Vision framework to raise, so the
  /// constructor cannot fail. The `Result` is the Apple signature kept
  /// whole, so a caller writes `?` once and compiles on every host.
  pub fn new(_options: &AppleVisionHandPoseOptions) -> Result<Self, AnalyzeError> {
    Ok(Self)
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect<P: HandPoseDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionHandPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_pixels<P: HandPoseDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionHandPoseOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }
}
