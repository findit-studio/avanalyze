//! Body pose internals: the 3-D height/estimation coupling, the affine
//! gate that decides whether a matrix was delivered at all, and the 3-D
//! joint read end to end through Vision.

use core::convert::Infallible;

use crate::{
  AppleVisionBodyPoserOptions, BodyPose3DDetection, BodyPose3DJoint, BodyPoser, HeightEstimation,
  body_pose::{AFFINE_TOLERANCE, sanitize_body_height_pair, translation_if_affine},
};

/// The crew portrait — provenance in `tests/fixtures/README.md`.
const CREW: &[u8] = include_bytes!("../../tests/fixtures/apollo11_crew.jpg");

/// A 3-D vocabulary that stores exactly what the engine hands it and
/// refuses nothing, so a test reads the engine's own output rather than
/// a downstream validator's opinion of it.
#[derive(Debug)]
struct Joint3 {
  name: String,
  x: f32,
  y: f32,
  z: f32,
  confidence: Option<f32>,
}

#[derive(Debug)]
struct Pose3 {
  confidence: f32,
  body_height: f32,
  height_estimation: HeightEstimation,
  joints: Vec<Joint3>,
}

impl BodyPose3DJoint for Joint3 {
  type Error = Infallible;

  fn try_new(
    name: &str,
    x: f32,
    y: f32,
    z: f32,
    confidence: Option<f32>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      name: name.to_owned(),
      x,
      y,
      z,
      confidence,
    })
  }

  fn name(&self) -> &str {
    &self.name
  }
}

impl BodyPose3DDetection for Pose3 {
  type Error = Infallible;
  type Joint = Joint3;

  fn try_new(
    confidence: f32,
    body_height: f32,
    height_estimation: HeightEstimation,
    joints: Vec<Self::Joint>,
  ) -> Result<Self, Self::Error> {
    Ok(Self {
      confidence,
      body_height,
      height_estimation,
      joints,
    })
  }
}

/// A finite body_height pairs with whatever height_estimation enum
/// Vision reported. The pair is forwarded unchanged.
#[test]
fn sanitize_body_height_pair_finite_preserves_estimation() {
  let measured = HeightEstimation::Measured;
  let (h, e) = sanitize_body_height_pair(1.75, measured);
  assert!((h - 1.75).abs() < 1e-6);
  assert_eq!(e, measured);

  let reference = HeightEstimation::Reference;
  let (h, e) = sanitize_body_height_pair(0.42, reference);
  assert!((h - 0.42).abs() < 1e-6);
  assert_eq!(e, reference);
}

/// When body_height is non-finite, the estimation enum MUST be forced
/// to `Unknown`. Preserving `Measured`/`Reference` while substituting
/// `0.0` would tell consumers there is a known 0-metre subject — a
/// worse semantic than "unknown estimate".
#[test]
fn sanitize_body_height_pair_non_finite_forces_unknown() {
  for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    // Even with a Measured input the result must be Unknown.
    let (h, e) = sanitize_body_height_pair(raw, HeightEstimation::Measured);
    assert_eq!(h, 0.0, "non-finite must collapse to 0.0 (raw = {raw:?})");
    assert_eq!(
      e,
      HeightEstimation::Unknown,
      "non-finite must force Unknown (raw = {raw:?})",
    );
    // Same for Reference.
    let (h, e) = sanitize_body_height_pair(raw, HeightEstimation::Reference);
    assert_eq!(h, 0.0);
    assert_eq!(e, HeightEstimation::Unknown);
  }
}

/// A column-major affine transform whose translation is
/// `(x, y, z)` and whose bottom row is exactly `(0, 0, 0, 1)`.
fn affine_with_translation(x: f32, y: f32, z: f32) -> [f32; 16] {
  let mut m = [0.0f32; 16];
  // The rotation block is the identity; only the last column and the
  // bottom-right corner carry anything.
  m[0] = 1.0;
  m[5] = 1.0;
  m[10] = 1.0;
  m[12] = x;
  m[13] = y;
  m[14] = z;
  m[15] = 1.0;
  m
}

/// The translation comes out of the last column, in metres, when the
/// matrix is an affine transform.
#[test]
fn translation_if_affine_reads_the_last_column() {
  assert_eq!(
    translation_if_affine(&affine_with_translation(-1.5, 2.5, 0.75)),
    Some((-1.5, 2.5, 0.75))
  );
  assert_eq!(
    translation_if_affine(&affine_with_translation(0.0, 0.0, 0.0)),
    Some((0.0, 0.0, 0.0)),
    "the root joint sits at the origin, and the origin is a translation like any other"
  );
}

/// The gate the ABI defect had to pass, tested on the shapes it could
/// not have passed.
///
/// Vision returns an affine transform for every joint of every pose,
/// so this branch is unreachable through the framework — which is
/// exactly why it is tested here, on matrices this crate writes down,
/// rather than left to a fixture that can never produce one.
#[test]
fn translation_if_affine_refuses_what_is_not_a_transform() {
  // The observed failure: a read that never received its return value,
  // so the corner where a `1` belongs held `3e-45` and the "metres"
  // were finite, plausible-looking garbage around `1e26`.
  let mut stale_stack = affine_with_translation(1.4e26, -8.9e25, 3.1e26);
  stale_stack[15] = 3e-45;
  assert_eq!(
    translation_if_affine(&stale_stack),
    None,
    "a bottom-right corner that is not 1 means the matrix was never delivered"
  );

  // A transposed read: the translation would be in the bottom row and
  // the bottom row in the last column.
  let mut transposed = [0.0f32; 16];
  transposed[0] = 1.0;
  transposed[5] = 1.0;
  transposed[10] = 1.0;
  transposed[3] = 0.1565;
  transposed[15] = 1.0;
  assert_eq!(
    translation_if_affine(&transposed),
    None,
    "a non-zero entry in the bottom row is not an affine transform"
  );

  // Each of the four bottom-row seats refuses on its own.
  for seat in [3usize, 7, 11] {
    let mut m = affine_with_translation(0.1, 0.2, 0.3);
    m[seat] = 1.0;
    assert_eq!(translation_if_affine(&m), None, "bottom row seat {seat}");
  }
  let mut zero_corner = affine_with_translation(0.1, 0.2, 0.3);
  zero_corner[15] = 0.0;
  assert_eq!(translation_if_affine(&zero_corner), None);

  // A NaN fails every comparison, so it rejects rather than passing
  // through — in the bottom row and in the translation alike.
  for seat in [3usize, 7, 11, 15] {
    let mut m = affine_with_translation(0.1, 0.2, 0.3);
    m[seat] = f32::NAN;
    assert_eq!(translation_if_affine(&m), None, "NaN in seat {seat}");
  }
  for seat in [12usize, 13, 14] {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let mut m = affine_with_translation(0.1, 0.2, 0.3);
      m[seat] = bad;
      assert_eq!(
        translation_if_affine(&m),
        None,
        "a non-finite translation in seat {seat}"
      );
    }
  }
}

/// The tolerance is a band, and both of its edges are deliberate.
#[test]
fn translation_if_affine_tolerates_float_noise_but_not_a_missed_read() {
  let mut noisy = affine_with_translation(0.5, 0.5, 0.5);
  noisy[3] = AFFINE_TOLERANCE / 2.0;
  noisy[7] = -AFFINE_TOLERANCE / 2.0;
  noisy[15] = 1.0 + AFFINE_TOLERANCE / 2.0;
  assert_eq!(
    translation_if_affine(&noisy),
    Some((0.5, 0.5, 0.5)),
    "a representation bit of noise must not cost a whole pose"
  );

  let mut past_the_edge = affine_with_translation(0.5, 0.5, 0.5);
  past_the_edge[3] = AFFINE_TOLERANCE * 2.0;
  assert_eq!(translation_if_affine(&past_the_edge), None);
}

/// The 3-D joint read, end to end, through Vision itself.
///
/// This is the test the capability never had, and it asserts the two
/// things the two defects broke.
///
/// **Presence.** Both of the ways this path has failed returned an
/// empty `Ok`: a selector that did not exist raised into the crate's
/// barriers, and before that a return-type encoding objc2 rejected
/// panicked into them. A pose count that must be non-zero is the only
/// assertion either would have failed.
///
/// **Shape.** Presence alone would not have caught the ABI defect,
/// which returned a full seventeen joints carrying stale stack bytes —
/// finite, so nothing else in the crate objected, and around `1e26`
/// metres. So the joints are checked against the invariants of a human
/// skeleton in model space: rooted at the hip, hips symmetric about the
/// root, head above them, ankle below, nothing further from the root
/// than a person is tall. Garbage cannot satisfy that shape by
/// accident, and neither can a transposed read — which would put the
/// bottom row's zeroes in every joint and collapse the hips onto the
/// root.
///
/// **Why invariants and not the measured numbers.** These coordinates
/// are the output of a neural network. Pinning them exactly would pin
/// this test to one host, one macOS and one execution backend, and CI
/// runs a floating `macos-latest` runner — a correct implementation
/// could go red on a runner change. The numbers this host reads at
/// `VNDetectHumanBodyPose3DRequestRevision1` are recorded here as
/// provenance rather than asserted: root `(0, 0, 0)`, left hip
/// `(0.1565, 0, 0)`, right hip `(-0.1565, 0, 0)`, centre head
/// `(-0.0506, 0.6316, 0.1749)`, left ankle `(0.2447, -0.6125, -0.0782)`.
/// The exact-value assertion the ABI needs lives in
/// `src/tests/ffi.rs`, against an Objective-C object whose matrix is
/// sixteen literals and cannot drift.
#[test]
fn three_d_joints_carry_a_human_skeleton_in_metres() {
  /// No joint of a human skeleton is further from the hip than this.
  /// The defect this bounds put the head at `1.7e26`.
  const MAX_HUMAN_METRES: f32 = 3.0;

  let options = AppleVisionBodyPoserOptions::new();
  let poses = BodyPoser::new(&options)
    .expect("BodyPoser::new builds its Vision requests on this host")
    .detect_3d::<Pose3>(CREW, &options)
    .expect("detect_3d must return Ok");

  assert!(
    !poses.is_empty(),
    "the crew fixture carries a 3-D pose; an empty result is the defect this test exists for"
  );

  for pose in &poses {
    // Pinned at `VNDetectHumanBodyPose3DRequestRevision1`, whose joint
    // roster is a contract rather than an inference.
    assert_eq!(
      pose.joints.len(),
      17,
      "Revision1's 3-D skeleton has 17 joints"
    );

    // Every joint's confidence is absent, always: Apple's 3-D point
    // hierarchy declares none. `None` is the reading, not a default.
    assert!(
      pose.joints.iter().all(|j| j.confidence.is_none()),
      "Vision reports no per-joint confidence on the 3-D road"
    );

    assert!(
      pose.confidence.is_finite() && (0.0..=1.0).contains(&pose.confidence),
      "pose confidence {} is not a confidence",
      pose.confidence
    );
    assert!(
      pose.body_height.is_finite() && (0.5..=MAX_HUMAN_METRES).contains(&pose.body_height),
      "body height {} m is not a human height",
      pose.body_height
    );
    assert_ne!(
      pose.height_estimation,
      HeightEstimation::Unknown,
      "a finite body height must not be paired with Unknown"
    );

    let joint = |name: &str| -> &Joint3 {
      pose
        .joints
        .iter()
        .find(|j| j.name == name)
        .unwrap_or_else(|| panic!("the 3-D skeleton must carry {name}"))
    };

    // Nothing is further from the root than a person is tall. This is
    // the assertion the ABI defect failed: its coordinates were finite,
    // so only a magnitude bound could refuse them.
    for j in &pose.joints {
      for (axis, v) in [("x", j.x), ("y", j.y), ("z", j.z)] {
        assert!(
          v.is_finite() && v.abs() <= MAX_HUMAN_METRES,
          "{}.{axis} is {v} m, which is not a coordinate on a human skeleton",
          j.name
        );
      }
    }

    // Model space is rooted at the hip, so the root joint is the origin
    // exactly — the one coordinate with no measurement error in it.
    let root = joint("human_root_3D");
    assert_eq!((root.x, root.y, root.z), (0.0, 0.0, 0.0));

    // The hips straddle the root: opposite signs, matching magnitudes.
    // A transposed read would read the bottom row instead and put both
    // of them at the origin, so this is where that failure lands.
    let left_hip = joint("human_left_hip_3D");
    let right_hip = joint("human_right_hip_3D");
    assert!(
      left_hip.x > 0.05,
      "the left hip sits to one side of the root, not on it: {}",
      left_hip.x
    );
    assert!(
      right_hip.x < -0.05,
      "the right hip sits to the other side: {}",
      right_hip.x
    );
    assert!(
      (left_hip.x + right_hip.x).abs() < 0.05,
      "the hips are symmetric about the root: {} and {}",
      left_hip.x,
      right_hip.x
    );

    // The skeleton is the right way up, which pins the axis convention.
    let head = joint("human_center_head_3D");
    let ankle = joint("human_left_ankle_3D");
    assert!(
      head.y > left_hip.y + 0.3,
      "the head is above the hips: {} against {}",
      head.y,
      left_hip.y
    );
    assert!(
      ankle.y < left_hip.y - 0.3,
      "the ankle is below the hips: {} against {}",
      ankle.y,
      left_hip.y
    );
  }
}
