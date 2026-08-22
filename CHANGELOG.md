# Changelog

## 0.4.0 — 2026-08-22

Face-pose angles and capture quality stop pretending "not measured" is
"measured at zero."

### Breaking

- **`FaceDetection::try_new`'s `roll` / `yaw` / `pitch` become `Option<f32>`.**
  Vision reports each pose angle independently and optionally — availability
  differs by OS version and by detection path, and some faces arrive with
  none of the three computed. The engine previously collapsed that absence
  with `unwrap_or(0.0)`, so "Vision never computed this angle" and "Vision
  measured a level head" were both written as `0.0` and could not be told
  apart downstream — a consumer filtering "pitched down more than 20°"
  wrongly excluded a 30°-down face whose pitch was simply never computed.
  `Some(0.0)` now means a head Vision actually measured as level; `None`
  means Vision did not compute that angle.
- `extract_faces` (the Vision face-rectangles pass) carries the absence
  through to the contract seat instead of defaulting it away.
- `conformance::assert_faces_accept` follows the seat: it passes
  `Option<f32>` for every angle and gains two canonical cases — a face with
  no angles computed at all, and a face mixing a genuinely-level `Some(0.0)`
  roll with an absent yaw on the same detection.
- `tests/common::Face`, the outside vocabulary, stores the three angles as
  `Option<f32>` and round-trips the absence. `src/tests/reference.rs`'s
  `mediaschema`-backed vocabulary cannot: `mediaschema` 0.2.1 has no seat for
  the absence yet, so that one adapter collapses `None` to `0.0` at the
  boundary — a limitation of the pinned dependency, not a reopening of this
  crate's loss. `mediaschema` / `mediagraph` adopt the `Option` shape in
  their own knife once this publishes.
- **`FaceDetection::try_new`'s `capture_quality` becomes `Option<f32>`** —
  the same collapse, found in the same #18 census, fixed in the same window
  (#20). `sanitize_capture_quality` no longer maps a nil Vision reading to
  `Some(0.0)`, and `matched_capture_quality` no longer annotates a face the
  capture-quality pass never covered with `0.0`; both now produce `None`.
  `Some(0.0)` means Vision measured this face's capture quality and found it
  terrible; `None` means Vision never measured it, whether the raw reading
  was nil or no capture-quality observation overlapped the face's box (a
  join-miss). The two were previously indistinguishable downstream — a
  consumer filtering "quality below X" could not tell a genuinely
  zero-scored face from one the quality pass skipped entirely.
- `extract_faces`'s `min_capture_quality` filter keeps its prior observable
  behaviour: an unmeasured face still compares as `0.0` against the
  threshold, so the default (0.1) still drops it and `min_capture_quality
  == 0.0` still keeps it. What changes is what a face that DOES clear the
  filter unmeasured now carries to the contract seat: `None`, not
  `Some(0.0)`.
- `conformance::assert_faces_accept` follows the seat: it passes
  `Option<f32>` for capture quality and gains an unmatched (`None`) case
  plus a mixed case — a measured-and-terrible (`Some(0.0)`) face and an
  unmatched (`None`) face built in the same detection set, proving the two
  remain independently representable.
- `tests/common::Face` stores `capture_quality` as `Option<f32>` and
  round-trips the absence; `src/tests/reference.rs`'s `mediaschema`-backed
  adapter collapses `None` to `0.0` at the same pinned-dependency boundary
  as the angles, for the same reason.

**Semver.** This breaks a published API — 0.3.0 is on crates.io — so it
rides the next breaking line (0.4.0). The version is not bumped here; the
release stamp is its own commit, matching the 0.3.0 precedent.

## 0.3.0 — 2026-08-22

The joint seats stop pretending the four skeletons are one.

### Breaking

- **The `Detections` bundle names four joint types where it named two.**
  `BodyPoseJoint` and `BodyPose3DJoint` give way to `BodyJoint`, `HandJoint`,
  `AnimalJoint` and `Body3Joint`, and every pose seat binds its own: body poses
  collect `BodyJoint`, hand poses `HandJoint`, animal poses `AnimalJoint`, 3-D
  poses `Body3Joint`. The old single 2-D seat asserted an identity that does not
  exist — Apple names a different joint roster for each skeleton — and so forced
  a vocabulary that models them as three types to collapse them into one. The
  compile-time same-source guarantee is untouched *within* a skeleton family and
  gone across families, which is where it never belonged.
- **New bundle seat: `AnimalPoseDetection`.** Animal 2-D poses had been reusing
  `BodyPoseDetection`'s type; they get their own seat over the same
  `BodyPoseDetection` trait — one trait because the payload is one, two seats
  because the joints are two. `Analysis::animal_body_poses` and its four
  accessors are typed `D::AnimalPoseDetection`.
- **`Detections` therefore names twenty-two output types, up from nineteen.**
  The per-part trait set is unchanged at nineteen: no trait was added, renamed,
  or reshaped — only the bundle's seats and the equalities between them moved.

Migrating a vocabulary that already models one joint type per skeleton is a set
of renames. A vocabulary with a single joint type names it in all three 2-D
seats, names one pose type in both 2-D pose seats, and keeps compiling: the
contract still *permits* the identity, it merely stops *imposing* it. The
reference implementation in `src/tests/reference.rs` is exactly that case.

**Semver.** This breaks a published API — 0.2.0 is on crates.io — so it rides a
**0.3.0**. The version is not bumped here; the release stamp is its own commit.

### Changed

- `conformance::assert_poses_accept` exercises the four joint seats one by one,
  and `assert_coordinate_refusals` now checks all three 2-D seats — a validating
  vocabulary that guards only one joint type hears about the other two. Both
  panic with the seat that refused, not just the trait.
- `tests/common`, the vocabulary written from outside the crate, names four
  genuinely distinct joint types, so the decoupling is compiled rather than
  merely asserted.

### Unchanged

Engine behaviour is preserved verbatim: the same Vision passes, the same
filtering, the same joint sort, the same eighteen `Analysis` slots. Only which
type each joint is built into changed.

## 0.2.0 — 2026-08-20

The engine stops assembling and stops depending.

### Breaking

- **`analyze_keyframe` is generic over an output vocabulary and returns
  detections, not an aggregate.**
  `analyze_keyframe(scene_id, keyframe_id, pts, dimensions, extractor, jpeg)`
  → `analyze_keyframe::<D: Detections>(jpeg, &AnalyzeOptions)`. The five
  carrier parameters were never read — they were moved verbatim into the
  aggregate the engine built — so they are gone along with the aggregate. The
  result is `Analysis<D>`: eighteen flat slots the caller composes into
  whatever record it stores.
- **New: the `Detections` contract.** One bundle trait naming nineteen output
  types, each with its own per-part trait (`BoundingBox`, `FaceDetection`,
  `DocumentSegment`, …). Every constructor seat is a primitive (`f32`, `u32`,
  `usize`, `&str`, `&[u8]`, tuples) or an engine-owned enum, so no foreign type
  crosses the boundary. Associated-type bounds make a mismatched bundle a
  compile error.
- **The three media dependencies are gone.** `mediaschema`, `mediaframe`, and
  `mediatime` leave `[dependencies]` entirely; `mediaschema` and `mediaframe`
  return as **dev**-dependencies for the reference implementation in
  `src/tests/reference.rs`. `bytes` leaves too — mask payloads cross the seam
  as `&[u8]`.
- **`AnalyzeError` replaces the borrowed error record**, with
  `AnalyzeErrorKind::{RequestFailed, Unsupported}` in place of the two
  `ErrorCode` variants. It implements `Display` and `std::error::Error`, so
  callers propagate it with `?` instead of reassembling a string from `code()`
  and `message()`.
- **`ServiceOptions` is now `AnalyzeOptions`**, passed per call rather than
  stored. `VisionAnalyzer::new` takes `&AnalyzeOptions` and consumes exactly
  one knob — `maximum_hand_count`, which Apple bakes into the retained
  request — so the analyzer holds no configuration of its own.
- **Engine-owned enums.** `Chirality` and `HeightEstimation` replace the
  imported handedness and height-estimation vocabularies.
- **`VisionAnalyzer::log_request_revisions` is public** and no longer takes
  service/worker parameters. It was dead code annotated as being called from a
  block that does not exist.

### Added

- `avanalyze::conformance`: assertion families an implementor runs to prove
  their vocabulary fits. `assert_contract` is the hard family (every value the
  engine can emit is accepted, in the argument order the engine uses);
  `assert_refuses_invalid` is the optional family for vocabularies that
  validate. The split is deliberate — the engine filters before it constructs,
  so a vocabulary that stores raw values is legal.
- `DEFAULT_*` associated constants on every options type, single-sourced with
  `new()` and with the serde defaults.
- `tests/common`: a vocabulary implemented from **outside** the crate, which is
  the only way to prove the contract is open (the orphan rule lets a crate
  implement its own traits for anyone's types, so the in-crate reference
  implementation cannot show it).

### Fixed

- **Six options types were mandatory-in-full when deserialized.** The saliency,
  horizon, document-segmentation, aesthetics, person-instance-mask and
  person-segmentation options carried no field-level `serde(default)`, so a
  config that named one knob in such a section — `{"attention_saliency":
  {"max_regions": 4}}` — failed on the field it did *not* name instead of
  filling that field in. Every field now defaults from the same function its
  `DEFAULT_*` constant reads, which is what the `DEFAULT_*` entry above already
  claimed for the whole set.
- **No panic path remains.** The two `expect`s on the horizon sentinel are
  gone: a vocabulary that refuses `try_new(0.0, 0.0)` now costs the horizon
  slot, not the worker thread.
- The keyframe-construction failure mode disappeared with the aggregate — the
  error surface is three paths on Apple targets and one off them.
- Manifest and documentation now describe the crate that exists: the git
  dependency that was never a git dependency, the re-export module that never
  existed, and the changelog that still described a 2022 template are all gone.

### Unchanged

Engine behaviour is preserved verbatim: the per-frame resource budgets, the
IoU ≥ 0.5 join between the two Vision face passes, all three coordinate
conventions, the per-field non-finite policy, the joint sort, the nested
exception barriers, the pinned request revisions, and the silent-refusal
policy.
