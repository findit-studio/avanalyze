//! The repro, kept as a test: a host whose Neural Engine is denied must
//! REFUSE, and the process must survive to say so.
//!
//! `VNDetectHumanBodyPose3DRequest`'s initialiser loads an ANE model,
//! and on a host that cannot reach the Neural Engine that load raises
//! `EspressoPlanFailure` rather than returning nil. Before the barrier
//! this crate now runs its constructors inside, that raise crossed into
//! Rust unguarded and took the whole process down — `fatal runtime
//! error: Rust cannot catch foreign exceptions`, signal 6 — before any
//! frame had been handed to the poser. A daemon that indexes media has
//! to decline the capability, not die of it.
//!
//! # Why a child process
//!
//! Because the denial is a property of the process, not of a call: it
//! is imposed by `sandbox-exec` at spawn and cannot be entered or left
//! afterwards. So this test re-executes its OWN test binary under a
//! profile that denies the ANE, and asserts on what comes back. The
//! child is told it is the child by [`CHILD_MARKER`], which is also
//! what stops it re-executing itself forever.
//!
//! # What is asserted, and what is not
//!
//! The child must exit ZERO. That is the property that was broken: an
//! abort is 134, a failed assertion is 101, and the defect this test
//! exists for produced the former on every run.
//!
//! Whether the RAISE actually happened is a different question, and
//! this test is careful not to conflate the two. The profile denies the
//! Neural Engine, but a host without one, or a macOS that quietly falls
//! back to the GPU, reaches the model anyway and the constructor
//! succeeds — a real outcome, and not a failure of this crate, but also
//! not proof of anything. So the child reports which branch it took and
//! the parent classifies the run:
//!
//! - a refusal is PROOF, and the child additionally requires it to be
//!   an [`AnalyzeErrorKind::Environment`] one;
//! - a success is INCONCLUSIVE, printed as such rather than passing
//!   quietly, so a profile that has decayed against a future macOS is
//!   visible in the log instead of green forever.
//!
//! Set `AVANALYZE_REQUIRE_ANE_DENIAL` to turn the inconclusive case
//! into a failure. That is for a runner known to have a Neural Engine
//! this profile can withhold; it is not the default because a
//! virtualised macOS runner — GitHub's included — does not expose one
//! to its guest, and a test that fails there would be reporting the
//! runner, not the crate.
//!
//! The sentinel line keeps even the weak case honest: a filter that
//! stopped matching would run zero tests and exit zero, so the parent
//! requires evidence that the child really ran.
//!
//! # And it must not leak, which is new
//!
//! A refusal a caller can retry is only an improvement if retrying is
//! bounded. The failed model load autoreleases as it unwinds — the
//! pipeline object, the exception, its reason, its call-stack arrays —
//! and before the barrier existed none of that mattered, because the
//! process died on the first attempt. So the child builds the poser
//! several times, and the parent runs it under
//! `OBJC_DEBUG_MISSING_POOLS=YES`, where Apple reports every object
//! autoreleased outside a pool as "just leaking". The absence of that
//! phrase is the assertion: it is what says the pool inside
//! `avanalyze_0_6_guard` really spans the failure.

use std::{
  env,
  io::Write as _,
  process::{Command, Stdio},
};

use crate::{AnalyzeErrorKind, AppleVisionBodyPoserOptions, BodyPoser};

/// Set on the re-executed binary. Its presence means "you are the
/// child, do the work"; its absence means "you are the parent, spawn
/// one".
const CHILD_MARKER: &str = "AVANALYZE_ANE_DENIED_CHILD";

/// Printed by the child so the parent can tell a real run from a
/// filter that matched nothing.
const SENTINEL: &str = "avanalyze-ane-denied-child-ran:";

/// Appended to [`SENTINEL`] by the child when the constructor really
/// was refused — the outcome that proves the raise crossed the barrier.
const OBSERVED: &str = "refused";

/// Set this to require the refusal rather than merely report it. For a
/// runner known to expose a Neural Engine the profile below can
/// withhold.
const REQUIRE_DENIAL: &str = "AVANALYZE_REQUIRE_ANE_DENIAL";

/// How many times the child builds the poser.
///
/// One attempt cannot distinguish a bounded refusal from an
/// accumulating one, and the objects a failed load strands are reported
/// per object, so a handful is enough for the missing-pool diagnostic
/// to have something to say if the pool is not there.
const ATTEMPTS: usize = 4;

/// What Apple's runtime prints, once per object, for anything
/// autoreleased with no pool in place. Its absence from the child's
/// stderr is what says the barrier's pool spans the failure.
const NO_POOL_DIAGNOSTIC: &str = "just leaking";

/// `sandbox-exec` is deprecated and undocumented, and it is also the
/// only way to withhold the Neural Engine from a process on a stock
/// macOS without disabling it for the whole machine.
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// The libtest name of the test below, as `--exact` wants it: the
/// module path with the crate name stripped.
///
/// Hand-written because there is no macro that yields it, and checked
/// against [`module_path!`] at runtime so a moved or renamed module
/// fails the test instead of silently filtering everything out.
const TEST_NAME: &str =
  "tests::ane_denied::a_denied_neural_engine_is_refused_and_the_process_lives";

/// Deny the Neural Engine, allow everything else.
///
/// The two halves are both needed and neither is enough: the mach
/// services are how a process asks the ANE daemon for a compiled model,
/// and the IOKit user clients are how it talks to the hardware once it
/// has one. Denying the model FILES instead would be a different test —
/// Vision then fails earlier, reports an `NSError`, and never reaches
/// the raise this is about.
const SANDBOX_PROFILE: &str = r#"(version 1)
(allow default)
(deny mach-lookup (global-name "com.apple.aned"))
(deny mach-lookup (global-name "com.apple.appleneuralengine"))
(deny mach-lookup (global-name "com.apple.aneuserd"))
(deny iokit-open (iokit-user-client-class "AppleH11ANEInterfaceUserClient"))
(deny iokit-open (iokit-user-client-class "H11ANEInUserClient"))
(deny iokit-open (iokit-user-client-class "AppleANEUserClient"))
"#;

#[test]
fn a_denied_neural_engine_is_refused_and_the_process_lives() {
  assert!(
    TEST_NAME.starts_with(module_path!().split_once("::").expect("a nested module").1),
    "TEST_NAME must be this test's libtest path; the module moved to {}",
    module_path!()
  );

  if env::var_os(CHILD_MARKER).is_some() {
    run_inside_the_sandbox();
    return;
  }

  if !std::path::Path::new(SANDBOX_EXEC).exists() {
    // Nothing to deny the engine with. Reported rather than failed:
    // the barrier is proved on synthetic throws in
    // `src/tests/native_barrier.rs`, and this test adds the real
    // Apple raise where the host can produce one.
    println!("{SANDBOX_EXEC} is absent; the denied-ANE repro cannot run on this host");
    return;
  }

  let profile = tempfile::Builder::new()
    .suffix(".sb")
    .tempfile()
    .expect("a temporary file for the sandbox profile");
  profile
    .as_file()
    .write_all(SANDBOX_PROFILE.as_bytes())
    .expect("writing the sandbox profile");
  profile.as_file().sync_all().expect("flushing the profile");

  let binary = env::current_exe().expect("the running test binary's path");
  let child = Command::new(SANDBOX_EXEC)
    .arg("-f")
    .arg(profile.path())
    .arg(&binary)
    .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
    .env(CHILD_MARKER, "1")
    // Report every object autoreleased outside a pool, so the repeated
    // refusal below can be checked for accumulation rather than assumed
    // bounded.
    .env("OBJC_DEBUG_MISSING_POOLS", "YES")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .output()
    .expect("spawning the sandboxed child");

  let stdout = String::from_utf8_lossy(&child.stdout);
  let stderr = String::from_utf8_lossy(&child.stderr);

  assert!(
    child.status.success(),
    "the sandboxed child must exit zero — an abort here is the defect this test exists for \
     (`fatal runtime error: Rust cannot catch foreign exceptions`, signal 6). status: {:?}\n\
     --- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}",
    child.status
  );
  assert!(
    stdout.contains(SENTINEL),
    "the child exited zero without running the test: the `--exact` filter matched nothing, which \
     would make this a vacuous pass.\n--- child stdout ---\n{stdout}\n--- child stderr ---\n\
     {stderr}"
  );
  assert!(
    !stderr.contains(NO_POOL_DIAGNOSTIC),
    "the child stranded objects outside an autorelease pool across {ATTEMPTS} refusals, so a \
     caller that retries accumulates them — the pool in avanalyze_0_6_guard is not spanning the \
     failure.\n--- child stderr ---\n{stderr}"
  );

  // Classify the run rather than let a non-reproduction pass quietly.
  let observed = stdout
    .lines()
    .filter(|line| line.contains(SENTINEL))
    .inspect(|line| println!("{line}"))
    .any(|line| line.contains(OBSERVED));

  if observed {
    return;
  }
  assert!(
    env::var_os(REQUIRE_DENIAL).is_none(),
    "{REQUIRE_DENIAL} is set, so this host was declared able to reproduce the denied-ANE raise — \
     and it did not: BodyPoser::new succeeded under the profile. Either the host has no Neural \
     Engine to withhold, or the profile no longer withholds it.\n--- child stdout ---\n{stdout}"
  );
  println!(
    "INCONCLUSIVE: the process survived, which is what the barrier promises, but this host \
     reached its model under the profile, so the denied-ANE raise was not exercised. Set \
     {REQUIRE_DENIAL} on a host with a Neural Engine to require it."
  );
}

/// The half that runs with the Neural Engine denied.
///
/// It builds the poser whose model load raises, [`ATTEMPTS`] times, and
/// asserts only what the barrier promises: an outcome, of either kind,
/// reached without the process dying, and reached the same way every
/// time. Returning normally from here is the pass; what the repetition
/// is for is the parent's missing-pool check, which needs more than one
/// failure to be worth reading.
fn run_inside_the_sandbox() {
  let options = AppleVisionBodyPoserOptions::new();
  let mut refused = 0usize;
  for attempt in 0..ATTEMPTS {
    match BodyPoser::new(&options) {
      Ok(_poser) => {}
      Err(error) => {
        assert_eq!(
          error.kind(),
          AnalyzeErrorKind::Environment,
          "a denied Neural Engine is an environment refusal, not a bad request: {error}"
        );
        if attempt == 0 {
          println!("{SENTINEL} {OBSERVED} with {error}");
        }
        refused += 1;
      }
    }
  }
  assert!(
    refused == 0 || refused == ATTEMPTS,
    "the refusal must be a property of the host, not of the attempt: {refused} of {ATTEMPTS} \
     attempts refused"
  );
  if refused == 0 {
    println!("{SENTINEL} constructed (this host reached its model anyway)");
  }
}
