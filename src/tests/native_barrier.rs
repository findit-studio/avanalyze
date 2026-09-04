//! The barrier, proved against every outcome it claims — and against
//! the one it deliberately refuses to claim.
//!
//! `src/objc_cxx_barrier.mm` catches three named C++ types and NOTHING
//! else. Apple's frameworks reach exactly one of the three on this host
//! (a denied Neural Engine raises an `NSException`), so the others are
//! reached by `src/objc_cxx_barrier_test.mm`, whose throws are three
//! statements and cannot drift with an OS release.
//!
//! The absence of a `catch (...)` is the part to read first, because it
//! looks like a gap and is the opposite. A Rust panic is foreign to C++
//! exactly as an `NSException` is foreign to Rust, so a catch-all would
//! match one — and the Rust Reference gives NO guarantees for a foreign
//! runtime that disposes of or rethrows a Rust panic payload. Naming
//! every clause is what keeps a panic's crossing supported instead of
//! merely observed:
//! [`a_rust_panic_crosses_the_barrier_and_stays_a_rust_panic`] says it
//! crosses, and
//! [`a_cxx_throw_the_barrier_cannot_name_is_left_to_keep_unwinding`]
//! says the catch-all has not come back.

use std::{
  env,
  panic::{AssertUnwindSafe, catch_unwind},
  process::{Command, Stdio},
  sync::Mutex,
};

use crate::{
  AnalyzeErrorKind,
  ffi::{guard_native, guard_vision_ffi},
};

/// Under `panic = "abort"` none of the tests below prove anything: the
/// raise would die in the abort-on-unwind shim rustc puts on the
/// `extern "C-unwind"` callback, and every assertion here would be
/// measuring a barrier that was never reached. `cargo test` always
/// builds with unwinding, so this never fires today; it is here so that
/// a build which changes that fails loudly instead of going green on a
/// configuration the barrier cannot work in.
const _: () = assert!(
  cfg!(panic = "unwind"),
  "the native barrier needs panic=unwind; see src/objc_cxx_barrier.mm"
);

/// Which synthetic throw to make. Must match the enum in
/// `src/objc_cxx_barrier_test.mm`.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub(super) enum SyntheticThrow {
  /// Return without throwing.
  Nothing = 0,
  /// `[NSException raise:format:]` — the shape Apple's model loads
  /// raise on a denied Neural Engine.
  NsException = 1,
  /// `throw std::runtime_error` — the shape Apple's C++ layers throw
  /// natively, under the Objective-C wrapper that usually converts it.
  StdException = 2,
  /// `throw 42` — a C++ exception with no relation to
  /// `std::exception`, and so the one shape the barrier's clauses
  /// cannot name.
  Int = 3,
  /// An `NSException` whose `-reason` throws a `std::runtime_error`,
  /// so the barrier's own reporting raises while it is reporting.
  UndescribableNsException = 4,
}

/// Makes one of the three synthetic throws, inside whatever barrier the
/// caller has put around it.
///
/// # Safety
///
/// Sound to call from anywhere; the throw it makes is not. The caller
/// must be inside [`guard_native`] (directly or through
/// [`guard_vision_ffi`]), or the exception unwinds into Rust as a
/// foreign exception and aborts the test binary at the first
/// `catch_unwind` it meets — which for a `#[test]` is the harness's
/// own.
pub(super) unsafe fn synthetic_throw(kind: SyntheticThrow) {
  // SAFETY: the callee reads only the discriminant, which is a valid
  // one by construction, and either returns or throws. The caller
  // upholds the barrier requirement above.
  unsafe { avanalyze_0_5_test_throw(kind as i32) }
}

// The test archive `build.rs` compiles with no cargo link directive:
// nothing but this attribute names it, and this attribute exists only
// in a `cfg(test)` build, so a function whose whole purpose is to throw
// never reaches a consumer's binary.
//
// `C-unwind`, not `C`: throwing is the point. A `C` declaration would
// mark the frame `nounwind` and turn each of these into an abort before
// the barrier next door ever saw it.
#[link(name = "avanalyze_0_5_objc_cxx_barrier_test", kind = "static")]
unsafe extern "C-unwind" {
  /// `src/objc_cxx_barrier_test.mm` — four throws and a return.
  fn avanalyze_0_5_test_throw(kind: i32);
}

// The pool observers, in the same archive. `C`, not `C-unwind`: one
// allocates and autoreleases and the other reads a counter, and neither
// can raise.
#[link(name = "avanalyze_0_5_objc_cxx_barrier_test", kind = "static")]
unsafe extern "C" {
  /// Autoreleases one sentinel into the innermost pool — inside a
  /// guarded closure, that is the barrier's own.
  fn avanalyze_0_5_test_autorelease_sentinel();
  /// How many sentinels this process has deallocated.
  fn avanalyze_0_5_test_sentinel_deallocations() -> i32;
}

/// The floor: a guarded call that does not throw returns its value, and
/// the barrier is not in the way of anything.
#[test]
fn a_call_that_does_not_raise_returns_its_value() {
  let got = guard_native("test", || {
    // SAFETY: inside the barrier, and this variant throws nothing.
    unsafe { synthetic_throw(SyntheticThrow::Nothing) };
    7u32
  })
  .expect("a call that does not raise must not be refused");

  assert_eq!(
    got, 7,
    "the guarded call's value crosses the barrier intact"
  );
}

/// An Objective-C exception — the shape a denied Neural Engine
/// produces — becomes an [`AnalyzeErrorKind::Environment`] refusal
/// carrying the exception's own name and reason.
#[test]
fn an_objective_c_exception_becomes_an_environment_refusal() {
  let error = guard_native("test_site", || {
    // SAFETY: inside the barrier.
    unsafe { synthetic_throw(SyntheticThrow::NsException) };
  })
  .expect_err("a raise must be refused, not returned");

  assert_eq!(error.kind(), AnalyzeErrorKind::Environment);
  assert!(
    error
      .to_string()
      .starts_with("apple-vision raised a native exception: "),
    "the new kind renders through `Display` like its siblings: {error}"
  );
  assert!(
    error.message().contains("test_site"),
    "the refusal names the site that raised: {}",
    error.message()
  );
  assert!(
    error.message().contains("AvanalyzeSyntheticException"),
    "the refusal carries the exception's own name: {}",
    error.message()
  );
  assert!(
    error
      .message()
      .contains("a synthetic Objective-C exception"),
    "the refusal carries the exception's own reason: {}",
    error.message()
  );
}

/// A `std::exception` — what Apple's C++ layers throw before their
/// Objective-C wrapper converts it, and the exception class
/// `objc2::exception::catch` provably cannot catch — is refused with
/// `what()` in the message.
#[test]
fn a_cxx_exception_becomes_an_environment_refusal() {
  let error = guard_native("test_site", || {
    // SAFETY: inside the barrier.
    unsafe { synthetic_throw(SyntheticThrow::StdException) };
  })
  .expect_err("a C++ throw must be refused, not returned");

  assert_eq!(error.kind(), AnalyzeErrorKind::Environment);
  assert!(
    error.message().contains("a synthetic std::runtime_error"),
    "the refusal carries what() : {}",
    error.message()
  );
}

/// The barrier renders a caught `NSException` by sending it `-name` and
/// `-reason`, from inside the catch handler — where the sibling clauses
/// of the enclosing `try` are no longer active. A raise from one of
/// those sends therefore does not fall to the next clause; it leaves
/// the trampoline and unwinds into Rust, which is the process death
/// this whole change exists to prevent, reached from the code that was
/// reporting a failure.
///
/// Apple's accessors do not throw. The barrier must not depend on that,
/// and only a receiver that really throws can say whether it does: this
/// one's `-reason` throws a `std::runtime_error`, which `@catch (id)`
/// alone would let straight through.
///
/// A regression here does not fail this test, it kills the binary — so
/// reaching the assertions at all is most of the result.
#[test]
fn an_exception_whose_description_throws_is_still_reported() {
  let error = guard_native("test_site", || {
    // SAFETY: inside the barrier.
    unsafe { synthetic_throw(SyntheticThrow::UndescribableNsException) };
  })
  .expect_err("a raise must be refused, not returned");

  assert_eq!(error.kind(), AnalyzeErrorKind::Environment);
  assert!(
    error.message().contains("whose description threw"),
    "the refusal falls back to a fixed message rather than escaping: {}",
    error.message()
  );
}

/// Set on the re-executed binary by
/// [`a_cxx_throw_the_barrier_cannot_name_is_left_to_keep_unwinding`].
const RESIDUAL_MARKER: &str = "AVANALYZE_CXX_RESIDUAL_CHILD";

/// Printed by that test's child ONLY on the outcome that must not
/// happen.
const RESIDUAL_CAUGHT: &str = "avanalyze-cxx-residual-was-caught:";

/// The named residual: a C++ throw the barrier's clauses cannot name is
/// NOT caught, and the barrier is not allowed to grow a `catch (...)`
/// that would catch it.
///
/// This test exists for the catch-all, not for `throw 42`. A `catch
/// (...)` in `src/objc_cxx_barrier.mm` would make this branch return an
/// error — which reads like an improvement — while also matching every
/// Rust panic, which the Rust Reference gives no guarantees for. The
/// two cannot be separated inside the handler: `__cxa_begin_catch` runs
/// before any check the handler could make. So the catch-all's absence
/// has to be a property something asserts, and this is the assertion.
///
/// It runs in a child process because the outcome under test is the
/// child NOT surviving: an unnamed exception unwinds into Rust, where
/// the Reference leaves it unspecified whether the process aborts or
/// `catch_unwind` yields an opaque error. Both are non-zero exits, and
/// neither is a thing the parent's own harness could report. What the
/// parent requires is therefore the pair: a non-zero child, and no
/// sentinel — the sentinel being printed only if the throw came back as
/// a value, which is exactly the regression.
#[test]
fn a_cxx_throw_the_barrier_cannot_name_is_left_to_keep_unwinding() {
  if env::var_os(RESIDUAL_MARKER).is_some() {
    let outcome = guard_native("residual", || {
      // SAFETY: `throw 42` is not a shape the barrier names, so this is
      // deliberately NOT inside an effective barrier — reaching the
      // next line at all is the regression this test reports.
      unsafe { synthetic_throw(SyntheticThrow::Int) };
    });
    println!("{RESIDUAL_CAUGHT} {outcome:?}");
    return;
  }

  let child = Command::new(env::current_exe().expect("the running test binary's path"))
    .args([
      "--exact",
      "tests::native_barrier::a_cxx_throw_the_barrier_cannot_name_is_left_to_keep_unwinding",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(RESIDUAL_MARKER, "1")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .output()
    .expect("spawning the child that throws");

  let stdout = String::from_utf8_lossy(&child.stdout);
  assert!(
    !stdout.contains(RESIDUAL_CAUGHT),
    "the barrier caught a C++ exception it cannot name, which means a `catch (...)` is back in \
     src/objc_cxx_barrier.mm — and a `catch (...)` matches Rust panics too.\n--- child stdout \
     ---\n{stdout}"
  );
  assert!(
    !child.status.success(),
    "an exception the barrier does not name must keep unwinding, not be absorbed silently. \
     status: {:?}\n--- child stdout ---\n{stdout}",
    child.status
  );
}

/// A Rust panic raised inside a guarded call comes out the other side
/// still a Rust panic, payload intact.
///
/// This is the property the clause list is written around. A panic is a
/// foreign exception from C++'s side of the boundary, so a `catch (...)`
/// would match it, and the Rust Reference gives no guarantees for a
/// foreign runtime that then disposes of or rethrows the payload —
/// "an unwind originated from a Rust runtime must either lead to
/// termination of the process or be caught by the same runtime". Naming
/// every clause is what keeps this crossing in the second case: the
/// panic passes THROUGH the C++ frame and the Rust runtime catches it.
///
/// A regression here can abort this test binary rather than fail it,
/// which is the honest signal for this class of defect and the same
/// convention `src/tests/ffi.rs` uses for the simd shim's raise.
#[test]
fn a_rust_panic_crosses_the_barrier_and_stays_a_rust_panic() {
  let outcome = catch_unwind(AssertUnwindSafe(|| {
    guard_native("test_site", || panic!("a synthetic Rust panic"))
  }));

  let payload = outcome.expect_err("the barrier must not swallow a Rust panic");
  let message = payload
    .downcast_ref::<&str>()
    .copied()
    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
    .expect("a `panic!` with a literal carries a string payload");
  assert_eq!(
    message, "a synthetic Rust panic",
    "the panic reaches the caller as the panic it was, not as a refusal"
  );
}

/// Serializes the two pool tests.
///
/// The deallocation counter is one process-wide number, and libtest
/// runs tests in parallel, so two tests reading it around their own
/// guarded call would each see the other's sentinel and neither
/// assertion would mean anything. Nothing else in this crate creates a
/// sentinel, so holding this for the read-act-read is enough to make
/// the difference attributable.
///
/// Poisoning is recovered from rather than propagated: a panic inside
/// one of these tests is a test failure that has already been reported,
/// and turning it into a second failure in the other test would only
/// hide which one broke.
static POOL_OBSERVATION: Mutex<()> = Mutex::new(());

/// The barrier's autorelease pool pops even when nothing is caught.
///
/// This is the half of the pool that the exception statuses cannot
/// reach. Clang lowers `@autoreleasepool` as a NORMAL-ONLY cleanup: the
/// pop is emitted where the block falls through and where a handler
/// returns, and the landing pad for an exception the frame does not
/// catch resumes without it. In a barrier that catches everything that
/// is invisible; in this one, whose design rests on deliberately NOT
/// catching some things, it strands a pool boundary on the thread every
/// time a panic crosses — and public extraction runs a consumer's own
/// vocabulary constructors inside these guards, so an application that
/// catches such a panic and carries on would accumulate one per
/// attempt.
///
/// So the pool is a C++ RAII guard, whose destructor is emitted as a
/// normal AND EH cleanup. Nothing about that is visible in the pool
/// itself — `objc_autoreleasePoolPop` returns nothing — so what is
/// asserted is the fate of an object autoreleased into it: released
/// when the pool pops, stranded when it does not.
///
/// The panic is raised AFTER the autorelease, so the sentinel is in the
/// pool when the unwind starts. `catch_unwind` then puts this thread
/// back in a state where the count can be read.
#[test]
fn the_pool_pops_when_a_panic_passes_through_uncaught() {
  let _observation = POOL_OBSERVATION.lock().unwrap_or_else(|e| e.into_inner());
  // SAFETY: a counter read; cannot raise.
  let before = unsafe { avanalyze_0_5_test_sentinel_deallocations() };

  let outcome = catch_unwind(AssertUnwindSafe(|| {
    guard_native("test_site", || {
      // SAFETY: allocates and autoreleases into the barrier's pool;
      // cannot raise.
      unsafe { avanalyze_0_5_test_autorelease_sentinel() };
      panic!("a synthetic Rust panic");
    })
  }));
  outcome.expect_err("the barrier must not swallow a Rust panic");

  // SAFETY: as above.
  let after = unsafe { avanalyze_0_5_test_sentinel_deallocations() };
  assert_eq!(
    after,
    before + 1,
    "the pool must pop while the panic unwinds through, or every crossing strands a pool \
     boundary and everything autoreleased after it"
  );
}

/// The same property on the path that IS caught, so the two halves of
/// the pool are asserted separately and a fix to one cannot silently
/// cover the other.
#[test]
fn the_pool_pops_when_an_exception_is_caught() {
  let _observation = POOL_OBSERVATION.lock().unwrap_or_else(|e| e.into_inner());
  // SAFETY: a counter read; cannot raise.
  let before = unsafe { avanalyze_0_5_test_sentinel_deallocations() };

  guard_native("test_site", || {
    // SAFETY: allocates and autoreleases into the barrier's pool.
    unsafe { avanalyze_0_5_test_autorelease_sentinel() };
    // SAFETY: inside the barrier.
    unsafe { synthetic_throw(SyntheticThrow::StdException) };
  })
  .expect_err("a C++ throw must be refused, not returned");

  // SAFETY: as above.
  let after = unsafe { avanalyze_0_5_test_sentinel_deallocations() };
  assert_eq!(
    after,
    before + 1,
    "the pool must pop after the handler has finished reporting"
  );
}

/// The same, through TWO barriers — the nesting production actually
/// runs.
///
/// `with_image` guards the whole per-call preamble, and the `perform`
/// and each extraction inside it guard themselves, so a panic raised on
/// a real detection path crosses two C++ frames rather than one. Each
/// one has to recognise it as foreign and rethrow, and the second has
/// to recognise the exception the FIRST rethrew — which is the case a
/// single-frame test cannot reach.
#[test]
fn a_rust_panic_crosses_nested_barriers_and_stays_a_rust_panic() {
  let outcome = catch_unwind(AssertUnwindSafe(|| {
    guard_native("outer", || {
      guard_native("inner", || panic!("a synthetic Rust panic"))
    })
  }));

  let payload = outcome.expect_err("neither barrier may swallow a Rust panic");
  let message = payload
    .downcast_ref::<&str>()
    .copied()
    .expect("a `panic!` with a literal carries a `&str` payload");
  assert_eq!(
    message, "a synthetic Rust panic",
    "the panic survives being rethrown twice"
  );
}

/// The degrading face of the barrier: [`guard_vision_ffi`] returns its
/// caller's own empty value for a C++ throw, exactly as it always has
/// for an Objective-C one.
///
/// This is what makes the fix invisible to the ninety-odd extraction
/// call sites: none of them changed, and each one's `fallback` is now
/// reached from one more world.
#[test]
fn the_degrading_guard_returns_its_fallback_for_a_cxx_throw() {
  let got = guard_vision_ffi("test_detector", vec![9u8], || {
    // SAFETY: inside the barrier.
    unsafe { synthetic_throw(SyntheticThrow::StdException) };
    vec![1u8, 2, 3]
  });

  assert_eq!(
    got,
    vec![9u8],
    "a caught C++ throw degrades to the detector's empty result"
  );
}

/// The same for an Objective-C throw, which the INNER barrier catches:
/// the two are nested, and the nesting must not have changed what the
/// caller sees.
#[test]
fn the_degrading_guard_returns_its_fallback_for_an_objc_raise() {
  let got = guard_vision_ffi("test_detector", vec![9u8], || {
    // SAFETY: inside the barrier.
    unsafe { synthetic_throw(SyntheticThrow::NsException) };
    vec![1u8, 2, 3]
  });

  assert_eq!(
    got,
    vec![9u8],
    "a caught Objective-C raise degrades to the detector's empty result"
  );
}
