//! Human body pose, 2-D and 3-D, behind one door.

#[cfg(target_vendor = "apple")]
use std::panic::{AssertUnwindSafe, catch_unwind};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_vision::*;

#[cfg(target_vendor = "apple")]
use crate::ffi::{
  ImageSource, MAX_POSE_JOINTS, MAX_VISION_RESULTS_PER_FRAME, PoseBudget, PoseJoints,
  ffi_nsstring_to_smolstr, finite_f32, guard_vision_ffi, pose_bbox_from_joint_bounds,
  read_pose_joints, run_requests, sanitize_confidence, vision_point_to_normalized,
  vn_point3d_position,
};
use crate::{AnalyzeError, AppleVisionBodyPoserOptions, BoundingBox, PixelPlane};

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
///
/// `confidence` is `Option<f32>` because Apple's 3-D point hierarchy
/// carries none. `VNPoint3D` declares `position`, `VNRecognizedPoint3D`
/// adds `identifier`, and `VNHumanBodyRecognizedPoint3D` adds
/// `localPosition` and `parentJoint` — and that is the whole roster.
/// The 2-D road reaches a confidence through a *different* hierarchy
/// (`VNPoint` → `VNDetectedPoint`, where `confidence` is declared, →
/// `VNRecognizedPoint`), which the 3-D family does not descend from.
/// The engine therefore passes `None` at
/// `VNDetectHumanBodyPose3DRequestRevision1`, never a substituted
/// number; the only confidence the 3-D road reports is the
/// observation's, which arrives per pose on
/// [`BodyPose3DDetection::try_new`]. A vocabulary that needs an `f32`
/// makes that collapse itself, where it is visible.
pub trait BodyPose3DJoint: Sized {
  /// Why a joint was refused.
  type Error;

  /// Builds a 3-D joint at a model-space position in metres.
  fn try_new(
    name: &str,
    x: f32,
    y: f32,
    z: f32,
    confidence: Option<f32>,
  ) -> Result<Self, Self::Error>;

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
    self.detect_2d_on::<P>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Detects 2-D human body poses in already-decoded `pixels`.
  ///
  /// [`detect_2d`](Self::detect_2d) reached without the encode: same
  /// request, same options, same output. The 3-D model is still not
  /// loaded.
  pub fn detect_2d_pixels<P: BodyPoseDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    self.detect_2d_on::<P>(ImageSource::Plane(pixels), options)
  }

  /// The one 2-D detection body both doors reach.
  fn detect_2d_on<P: BodyPoseDetection>(
    &self,
    source: ImageSource<'_>,
    options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.pose_2d.clone())] };
    run_requests(source, &requests, Vec::new(), || {
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
    self.detect_3d_on::<P>(ImageSource::Jpeg(jpeg_data), options)
  }

  /// Detects 3-D human body poses in already-decoded `pixels`, in
  /// model-space metres.
  ///
  /// [`detect_3d`](Self::detect_3d) reached without the encode: same
  /// request, same options, same output. The 2-D model is still not
  /// loaded.
  pub fn detect_3d_pixels<P: BodyPose3DDetection>(
    &self,
    pixels: &PixelPlane<'_>,
    options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    self.detect_3d_on::<P>(ImageSource::Plane(pixels), options)
  }

  /// The one 3-D detection body both doors reach.
  ///
  /// `_options` is unused:
  /// [`AppleVisionBodyPose3DOptions`](crate::AppleVisionBodyPose3DOptions)
  /// carries no gate for this crate to read, because the 3-D road
  /// reports nothing per joint to gate on. The parameter stays on the
  /// public doors so every entry point keeps one shape and a future 3-D
  /// knob is additive rather than a signature change.
  fn detect_3d_on<P: BodyPose3DDetection>(
    &self,
    source: ImageSource<'_>,
    _options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    let requests = unsafe { [Retained::cast_unchecked::<VNRequest>(self.pose_3d.clone())] };
    // `extract_3d` self-guards (inner `objc2::exception::catch` under
    // its `catch_unwind`); a call-site guard here would put
    // `catch_unwind` inside the ObjC barrier and could not catch the
    // foreign exception.
    run_requests(source, &requests, Vec::new(), || self.extract_3d::<P>())
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
  fn extract_3d<P: BodyPose3DDetection>(&self) -> Vec<P> {
    // Two nested barriers, innermost FIRST, and the order is load-bearing.
    // Any send below can raise an `NSException`, and Rust's `catch_unwind`
    // CANNOT catch a foreign exception — one reaching it aborts the process.
    // So `objc2::exception::catch` (`guard_vision_ffi`) is innermost.
    //
    // The outer `catch_unwind` catches the other failure: objc2 verifies every
    // message send against the runtime's own metadata in DEBUG builds, and a
    // disagreement is a Rust panic. This path has been bitten by both. It sent
    // `confidence` to a point that declares no such selector, which panicked
    // here in debug and raised `doesNotRecognizeSelector:` into the inner
    // barrier in release; before that, a `simd_float4x4` return encoding objc2
    // rejected panicked here in debug alone. Verification is compiled out of
    // release, so only the inner barrier stands there. Both are kept because
    // each catches what the other cannot.
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

            // No per-joint confidence exists to read. See
            // [`BodyPose3DJoint`]: the whole 3-D point hierarchy declares
            // `position`, `identifier`, `localPosition` and `parentJoint` and
            // nothing else. `None` is the reading, not a fallback for one.
            let Ok(joint) = P::Joint::try_new(&name, x, y, z, None) else {
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

/// How far the bottom row may sit from `(0, 0, 0, 1)` and still be one.
///
/// Deliberately loose. This separates "a transform" from "not a
/// transform", not one transform from another: Apple writes that row as
/// literals, any float noise it could pick up is orders of magnitude
/// below this, and a read that missed its return value is orders of
/// magnitude above — the observed failure put `3e-45` in the corner
/// where a `1` belongs. An exact comparison would buy nothing and would
/// refuse a whole pose over one representation bit.
#[cfg(target_vendor = "apple")]
pub(crate) const AFFINE_TOLERANCE: f32 = 1e-3;

/// The translation of a column-major 4x4 affine transform, in the
/// matrix's own units, or `None` if the sixteen floats are not one.
///
/// Split out of [`extract_body_pose_3d_coordinates`] so it can be tested
/// on matrices this crate chooses rather than only on ones Vision
/// happens to produce. Vision returns an affine transform for every
/// joint of every pose it has ever been handed here, so the rejecting
/// branch is unreachable through the framework — and it is the branch
/// that stands between a broken read and coordinates that look
/// plausible. It is tested directly, in `src/tests/body_pose.rs`.
///
/// A 4x4 affine transform's bottom row is `(0, 0, 0, 1)` by
/// construction, so a read that does not satisfy it did not deliver a
/// transform. That check is not decoration: the ABI defect this was
/// written to catch produced *finite* coordinates — stale stack bytes
/// reinterpreted as floats, around `1e26` metres — which every
/// finiteness test in the crate passed. A wrong answer that looks
/// plausible is the failure mode worth spending four comparisons on.
///
/// NaN fails every comparison here, so a NaN anywhere in the bottom row
/// rejects the matrix rather than passing through it.
#[cfg(target_vendor = "apple")]
pub(crate) fn translation_if_affine(transform: &[f32; 16]) -> Option<(f32, f32, f32)> {
  // Column-major: element (row, col) is `transform[col * 4 + row]`, so
  // the bottom row is every fourth element from index 3.
  let bottom_row_is_affine = transform[3].abs() < AFFINE_TOLERANCE
    && transform[7].abs() < AFFINE_TOLERANCE
    && transform[11].abs() < AFFINE_TOLERANCE
    && (transform[15] - 1.0).abs() < AFFINE_TOLERANCE;
  if !bottom_row_is_affine {
    return None;
  }

  let [x, y, z] = [transform[12], transform[13], transform[14]];
  if !(x.is_finite() && y.is_finite() && z.is_finite()) {
    return None;
  }
  Some((x, y, z))
}

/// Reads a 3-D joint's model-space translation, in metres.
///
/// `-[VNPoint3D position]` returns a column-major `simd_float4x4`, and
/// [`vn_point3d_position`](crate::ffi::vn_point3d_position) exists
/// because Rust cannot receive that return type — see its own
/// documentation for the ABI evidence. The matrix's last column is the
/// translation, and [`translation_if_affine`] takes it only from a
/// matrix that is one.
#[cfg(target_vendor = "apple")]
fn extract_body_pose_3d_coordinates(
  point: &VNHumanBodyRecognizedPoint3D,
) -> Option<(f32, f32, f32)> {
  translation_if_affine(&unsafe { vn_point3d_position(point) })
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
  pub fn detect_2d_pixels<P: BodyPoseDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
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

  /// Non-macOS stub: always reports
  /// [`AnalyzeErrorKind::Unsupported`](crate::AnalyzeErrorKind::Unsupported).
  pub fn detect_3d_pixels<P: BodyPose3DDetection>(
    &self,
    _pixels: &PixelPlane<'_>,
    _options: &AppleVisionBodyPoserOptions,
  ) -> Result<Vec<P>, AnalyzeError> {
    crate::error::unsupported()
  }
}
