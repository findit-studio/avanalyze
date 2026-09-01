//! Human body pose, 2-D and 3-D, behind one door.

#[cfg(target_vendor = "apple")]
use std::panic::{AssertUnwindSafe, catch_unwind};

#[cfg(target_vendor = "apple")]
use objc2::{
  encode::{Encode, Encoding},
  rc::Retained,
};
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  MAX_POSE_JOINTS, MAX_VISION_RESULTS_PER_FRAME, PoseBudget, PoseJoints, ffi_nsstring_to_smolstr,
  finite_f32, guard_vision_ffi, pose_bbox_from_joint_bounds, read_pose_joints, run_requests,
  sanitize_confidence, vision_point_to_normalized,
};
use crate::{AnalyzeError, AppleVisionBodyPoserOptions, BoundingBox};

/// One 2-D pose joint — the shape body, hand, and animal joints share.
///
/// The shape is shared; the vocabularies are not. Apple names a
/// different joint set per skeleton, so each entry point's pose type
/// carries its own joint type: a vocabulary that keeps one joint type
/// for all three is still legal, and one that keeps three is now
/// expressible.
///
/// The name is Apple's joint identifier, verbatim. The engine sorts
/// joints by this name before building the enclosing pose, because
/// Apple's dictionary iteration order is unspecified; an
/// implementation whose [`name`](BodyPoseJoint::name) does not return
/// what it was constructed with makes pose output non-reproducible.
pub trait BodyPoseJoint: Sized {
  /// Why a joint was refused.
  type Error;

  /// Builds a joint at a normalized, top-left-origin position.
  fn try_new(name: &str, x: f32, y: f32, confidence: f32) -> Result<Self, Self::Error>;

  /// Joint name, as constructed.
  fn name(&self) -> &str;
}

/// A 2-D pose — the shape human bodies and animal bodies share.
///
/// [`BodyPoser::detect_2d`] and
/// [`AnimalPoser::detect`](crate::AnimalPoser::detect) both build this
/// trait, each over its own joint type; the trait is one because the
/// payload is.
///
/// Vision does not report a box for a pose. The engine synthesises one
/// from the bounds of the joints that survived filtering, so the box
/// describes the surviving joints and nothing more; a pose whose
/// joints are collinear has no box and is dropped before it reaches
/// this seam.
pub trait BodyPoseDetection: Sized {
  /// Why a pose was refused.
  type Error;
  /// The geometry type this pose is built from.
  type BoundingBox: BoundingBox;
  /// The joint type this pose collects.
  type Joint: BodyPoseJoint;

  /// Builds a pose from its synthesised box and its non-empty,
  /// name-sorted joint list.
  fn try_new(
    bbox: Self::BoundingBox,
    confidence: f32,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error>;
}

/// One 3-D body-pose joint.
///
/// A vocabulary of its own: the 3-D skeleton is neither the 2-D body's
/// joint set nor its coordinate system.
///
/// `x` / `y` / `z` are model-space **metres**, not normalized
/// coordinates: no flip, no clamp, no `0.0..=1.0` invariant. Only
/// finiteness is enforced upstream.
pub trait BodyPose3DJoint: Sized {
  /// Why a joint was refused.
  type Error;

  /// Builds a 3-D joint at a model-space position in metres.
  fn try_new(name: &str, x: f32, y: f32, z: f32, confidence: f32) -> Result<Self, Self::Error>;

  /// Joint name, as constructed.
  fn name(&self) -> &str;
}

/// A 3-D body pose.
///
/// Alone among the pose types this one carries **no bounding box** —
/// Vision reports 3-D poses in model space, which has no projection
/// back to the frame.
pub trait BodyPose3DDetection: Sized {
  /// Why a pose was refused.
  type Error;
  /// The joint type this pose collects.
  type Joint: BodyPose3DJoint;

  /// Builds a 3-D pose.
  ///
  /// `body_height` is metres, and is coupled to `height_estimation`:
  /// the engine only pairs a non-`Unknown` estimation with a finite
  /// height.
  fn try_new(
    confidence: f32,
    body_height: f32,
    height_estimation: HeightEstimation,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error>;
}

/// How a 3-D body pose's height estimate was obtained.
///
/// Paired with the body height in metres on
/// [`BodyPose3DDetection::try_new`]. The pair is coupled: when the
/// height reading is not finite the engine emits
/// `(0.0, HeightEstimation::Unknown)` rather than a `0.0` metre
/// subject with a `Measured` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeightEstimation {
  /// No usable estimate.
  #[default]
  Unknown,
  /// Derived from a reference height rather than measured.
  Reference,
  /// Measured from the observation.
  Measured,
}

#[cfg(target_vendor = "apple")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimdFloat4(pub(crate) [f32; 4]);

#[cfg(target_vendor = "apple")]
unsafe impl Encode for SimdFloat4 {
  // `simd_float4` is an `__attribute__((__ext_vector_type__))` type;
  // Clang intentionally emits NO `@encode` for ext-vector elements, so
  // the matching Rust-side encoding is [`Encoding::None`] (formats as
  // empty string), NOT [`Encoding::Unknown`] (formats as `?`).
  //
  // objc2-encode's `Encoding::None` docstring explicitly calls this
  // out as the SIMD-vector case. The previous `Encoding::Unknown`
  // made the wrapping struct render as `{?=[4?]}`, while Vision's
  // `-[VNHumanBodyRecognizedPoint3D position]` returns `{?=[4]}`
  // (Clang refuses to emit an inner element character) — every
  // msg_send for that selector failed verification on macOS 26.x,
  // and the surrounding `catch_unwind` silently swallowed the
  // panic so 3-D pose detections were always dropped to zero.
  const ENCODING: Encoding = Encoding::None;
}

#[cfg(target_vendor = "apple")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimdFloat4x4 {
  columns: [SimdFloat4; 4],
}

#[cfg(target_vendor = "apple")]
unsafe impl Encode for SimdFloat4x4 {
  // Apple's `simd_float4x4` is a struct-of-vectors. Clang reports
  // `@encode(simd_float4x4)` as `{?=[4]}` — outer struct with no name
  // wrapping an array of 4 whose element type Clang refuses to encode
  // (the element is itself an ext-vector, see [`SimdFloat4`] above).
  // The matching Rust encoding therefore uses `Array(4, &None)` so
  // the inner array element formats to an empty string, producing the
  // literal `[4]` Clang emits.
  const ENCODING: Encoding = Encoding::Struct("?", &[Encoding::Array(4, &Encoding::None)]);
}

/// Apple Vision human body pose, 2-D and 3-D — one per worker thread.
///
/// Owns two Vision requests behind one door, and each method performs
/// only its own: a 2-D consumer never runs the 3-D model, and vice
/// versa. Constructing the pair is cheap — Apple loads the model at
/// perform time, not at request-object construction.
///
/// The retained `VNRequest`s carry per-call state across
/// `performRequests` / `results()`, so a poser is not safe to share
/// across threads; build one per worker.
#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct BodyPoser {
  pose_2d: Retained<VNDetectHumanBodyPoseRequest>,
  pose_3d: Retained<VNDetectHumanBodyPose3DRequest>,
}

#[cfg(target_vendor = "apple")]
impl BodyPoser {
  /// Creates a poser holding both body-pose requests at their pinned
  /// revisions.
  ///
  /// `_options` is unused: Apple bakes no knob this crate exposes into
  /// these request objects, so every gate is read per call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionBodyPoserOptions) -> Self {
    unsafe {
      let pose_2d = VNDetectHumanBodyPoseRequest::new();
      pose_2d.setRevision(VNDetectHumanBodyPoseRequestRevision1);

      let pose_3d = VNDetectHumanBodyPose3DRequest::new();
      pose_3d.setRevision(VNDetectHumanBodyPose3DRequestRevision1);

      Self { pose_2d, pose_3d }
    }
  }

  /// Logs the pinned revision of both body-pose requests.
  ///
  /// A revision drift changes the joint roster **silently** — same
  /// API, different skeleton.
  #[cfg(feature = "tracing")]
  pub fn log_request_revisions(&self) {
    unsafe {
      tracing::info!(
        body_pose_rev = self.pose_2d.revision(),
        body_pose_3d_rev = self.pose_3d.revision(),
        "initialized pinned Apple Vision request revisions"
      );
    }
  }

  /// Detects 2-D human body poses in `jpeg_data`.
  ///
  /// Performs only the 2-D request; the 3-D model is not loaded.
  pub fn detect_2d<P: BodyPoseDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.pose_2d.clone())] };
    run_requests(jpeg_data, &requests, Vec::new(), || {
      guard_vision_ffi("body_pose", Vec::new(), || self.extract_2d::<P>(options))
    })
  }

  /// Detects 3-D human body poses in `jpeg_data`, in model-space
  /// metres.
  ///
  /// Performs only the 3-D request; the 2-D model is not loaded.
  pub fn detect_3d<P: BodyPose3DDetection>(
    &self,
    jpeg_data: &[u8],
    options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.pose_3d.clone())] };
    // `extract_3d` self-guards (inner `objc2::exception::catch` under
    // its `catch_unwind`); a call-site guard here would put
    // `catch_unwind` inside the ObjC barrier and could not catch the
    // foreign exception.
    run_requests(jpeg_data, &requests, Vec::new(), || {
      self.extract_3d::<P>(options)
    })
  }

  /// A pose is emitted whole or not at all: when the call's cumulative
  /// [`PoseBudget`] cannot cover the pose in hand the extraction stops
  /// there, and no pose is ever truncated to fit what is left.
  fn extract_2d<P: BodyPoseDetection>(&self, options: &AppleVisionBodyPoserOptions) -> Vec<P> {
    let Some(results) = (unsafe { self.pose_2d.results() }) else {
      return Vec::new();
    };

    let mut budget = PoseBudget::new();
    let mut body_poses = Vec::with_capacity(results.len().min(MAX_VISION_RESULTS_PER_FRAME));
    for obs in results.iter().take(MAX_VISION_RESULTS_PER_FRAME) {
      let Ok(points_by_joint) = (unsafe {
        obs.recognizedPointsForJointsGroupName_error(VNHumanBodyPoseObservationJointsGroupNameAll)
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

        // Vision normalized points are lower-left origin; flip y for the
        // top-left schema convention before recording the joint or
        // deriving the bbox. A non-finite raw coordinate is dropped at
        // the source — partial-joint lists are valid for body pose so
        // we skip just this joint, not the whole pose.
        let Some((x, y)) = vision_point_to_normalized(unsafe { point.x() }, unsafe { point.y() })
        else {
          continue;
        };
        let Some(confidence) = sanitize_confidence(
          unsafe { point.confidence() },
          options.pose_2d().min_joint_confidence(),
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
        // A pose with only one surviving joint (or perfectly colinear
        // joints) cannot produce a valid axis-aligned bbox; skip it
        // rather than emit a zero-extent box that the domain
        // validator would reject.
        continue;
      };
      // Observation confidence carries the per-pose score; sanitise it
      // against the same `[0, 1]` invariant. A non-finite observation
      // confidence cannot be emitted faithfully — drop the pose.
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

  /// A pose is emitted whole or not at all: when the call's cumulative
  /// [`PoseBudget`] cannot cover the pose in hand the extraction stops
  /// there, and no pose is ever truncated to fit what is left.
  fn extract_3d<P: BodyPose3DDetection>(&self, options: &AppleVisionBodyPoserOptions) -> Vec<P> {
    // Two nested barriers, innermost FIRST. The `VNHumanBodyRecognizedPoint3D`
    // `position`/`confidence` msg_sends below have a `simd_float4x4` return
    // encoding that objc2 rejects: in a DEBUG build objc2's runtime
    // verification turns that into a *Rust panic* (caught by the outer
    // `catch_unwind`), but in a RELEASE build verification is compiled out and
    // the real Objective-C dispatch raises an `NSException` instead. That
    // foreign exception MUST be caught by the inner `objc2::exception::catch`
    // (`guard_vision_ffi`) — if it reached the outer `catch_unwind` it would
    // abort the process (`catch_unwind` cannot catch foreign exceptions). So
    // the ObjC barrier is innermost; the Rust-panic barrier wraps it.
    catch_unwind(AssertUnwindSafe(|| {
      guard_vision_ffi("body_pose_3d", Vec::new(), || {
        let Some(results) = (unsafe { self.pose_3d.results() }) else {
          return Vec::new();
        };
        let Some(group_name) = (unsafe { VNHumanBodyPose3DObservationJointsGroupNameAll }) else {
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

          // Read the joint dictionary bounded, as in the 2-D path: the
          // walk pays the attempt budget entry by entry, a malformed
          // dictionary drops just its own pose, and an exhausted budget
          // stops the extraction.
          let pairs = match read_pose_joints(&points_by_joint, MAX_POSE_JOINTS, &mut budget) {
            PoseJoints::Read(pairs) => pairs,
            PoseJoints::Malformed => continue,
            PoseJoints::Exhausted => break,
          };
          let mut joints = Vec::with_capacity(pairs.len());
          let mut name_bytes: usize = 0;

          for (joint_name, point) in pairs {
            let Some(name) = ffi_nsstring_to_smolstr(&joint_name) else {
              continue;
            };
            if name.is_empty() {
              continue;
            }

            let Some((x, y, z)) = extract_body_pose_3d_coordinates(&point) else {
              continue;
            };
            let raw_confidence: f32 = unsafe { objc2::msg_send![&*point, confidence] };
            let Some(confidence) =
              sanitize_confidence(raw_confidence, options.pose_3d().min_joint_confidence())
            else {
              continue;
            };

            let Ok(joint) = P::Joint::try_new(&name, x, y, z, confidence) else {
              continue;
            };
            name_bytes = name_bytes.saturating_add(name.len());
            joints.push(joint);
          }

          // Charged before the emptiness gate below, so the budget
          // reflects the joints this pose actually walked and retained
          // rather than only those a later gate let through.
          if !budget.admit_pose(joints.len(), name_bytes) {
            break;
          }

          if joints.is_empty() {
            continue;
          }
          let Some(pose_confidence) = sanitize_confidence(unsafe { obs.confidence() }, 0.0) else {
            continue;
          };

          joints.sort_by(|lhs, rhs| lhs.name().cmp(rhs.name()));
          // See `sanitize_body_height_pair` — couples the
          // body_height substitution with the height_estimation enum
          // so `(0.0, UNKNOWN)` is the only fallback for non-finite
          // readings.
          let mapped_estimation =
            map_body_pose_3d_height_estimation(unsafe { obs.heightEstimation() });
          let (body_height, height_estimation) =
            sanitize_body_height_pair(unsafe { obs.bodyHeight() }, mapped_estimation);
          if let Ok(pose) = P::try_new(pose_confidence, body_height, height_estimation, joints) {
            body_poses.push(pose);
          }
        }

        body_poses
      })
    }))
    .unwrap_or_else(|_| {
      #[cfg(feature = "tracing")]
      tracing::warn!("caught panic while extracting human body pose 3D; returning empty result");
      Vec::new()
    })
  }
}

/// Sanitise a raw 3-D body-pose height + height-estimation pair.
///
/// Vision's `bodyHeight()` is metres in model space. When the
/// reading is non-finite, both `body_height` AND `height_estimation`
/// must be neutralised together — substituting `0.0` for the height
/// while preserving a `Measured` or `Reference` enum would tell
/// consumers there is a known 0-metre subject. The pair
/// `(0.0, UNKNOWN)` is the truthful encoding of "no estimate
/// available" and the only consistent fallback.
#[cfg(target_vendor = "apple")]
#[inline]
pub(crate) fn sanitize_body_height_pair(
  raw_height: f32,
  measured_or_reference: HeightEstimation,
) -> (f32, HeightEstimation) {
  match finite_f32(raw_height) {
    Some(finite) => (finite, measured_or_reference),
    None => (0.0, HeightEstimation::Unknown),
  }
}

#[cfg(target_vendor = "apple")]
fn extract_body_pose_3d_coordinates(
  point: &VNHumanBodyRecognizedPoint3D,
) -> Option<(f32, f32, f32)> {
  let transform: SimdFloat4x4 = unsafe { objc2::msg_send![point, position] };
  let translation = transform.columns.get(3)?;
  let x = translation.0[0];
  let y = translation.0[1];
  let z = translation.0[2];
  if !(x.is_finite() && y.is_finite() && z.is_finite()) {
    return None;
  }
  Some((x, y, z))
}

#[cfg(target_vendor = "apple")]
fn map_body_pose_3d_height_estimation(
  estimation: VNHumanBodyPose3DObservationHeightEstimation,
) -> HeightEstimation {
  if estimation == VNHumanBodyPose3DObservationHeightEstimation::Measured {
    HeightEstimation::Measured
  } else if estimation == VNHumanBodyPose3DObservationHeightEstimation::Reference {
    HeightEstimation::Reference
  } else {
    HeightEstimation::Unknown
  }
}

/// Non-macOS stub for [`BodyPoser`].
#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct BodyPoser;

#[cfg(not(target_vendor = "apple"))]
impl BodyPoser {
  /// Constructs a non-macOS stub poser. The options are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(_options: &AppleVisionBodyPoserOptions) -> Self {
    Self
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_2d<P: BodyPoseDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_3d<P: BodyPose3DDetection>(
    &self,
    _jpeg_data: &[u8],
    _options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }
}
