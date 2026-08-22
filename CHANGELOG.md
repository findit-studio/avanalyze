# Changelog

## Unreleased

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
