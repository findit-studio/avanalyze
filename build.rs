use std::env::{self, var};

fn main() {
  // Don't rerun this on changes other than build.rs, as we only depend on
  // the rustc version.
  println!("cargo:rerun-if-changed=build.rs");

  // Check for `--features=tarpaulin`.
  let tarpaulin = var("CARGO_FEATURE_TARPAULIN").is_ok();

  if tarpaulin {
    use_feature("tarpaulin");
  } else {
    // Always rerun if these env vars change.
    println!("cargo:rerun-if-env-changed=CARGO_TARPAULIN");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARPAULIN");

    // Detect tarpaulin by environment variable
    if env::var("CARGO_TARPAULIN").is_ok() || env::var("CARGO_CFG_TARPAULIN").is_ok() {
      use_feature("tarpaulin");
    }
  }

  // Rerun this script if any of our features or configuration flags change,
  // or if the toolchain we used for feature detection changes.
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TARPAULIN");

  build_native_half();
}

fn use_feature(feature: &str) {
  println!("cargo:rustc-cfg={}", feature);
}

/// The major.minor version every exported native name carries.
///
/// C has no namespaces, and Cargo permits two semver-incompatible
/// versions of one crate in a single graph, so every function this
/// crate's native half exports, its test object's class, the static
/// archives below and the `links` key in `Cargo.toml` are all scoped by
/// this tag — the reasoning is on `src/objc_simd_shim.m`.
/// [`assert_shim_abi_tag_matches_package_version`] keeps it honest.
const SHIM_ABI_TAG: &str = "0_6";

/// Fails the build when [`SHIM_ABI_TAG`] and the package version drift.
///
/// A version-scoped symbol is worth having only while the scope is
/// truthful. Raising `version` in `Cargo.toml` without raising this tag
/// would republish the previous version's symbol under a new crate
/// version — the very collision the scope exists to prevent — and would
/// do it silently, since nothing else in the build reads either value.
/// This makes that a build error that names what to edit.
fn assert_shim_abi_tag_matches_package_version() {
  let major = var("CARGO_PKG_VERSION_MAJOR").expect("cargo sets CARGO_PKG_VERSION_MAJOR");
  let minor = var("CARGO_PKG_VERSION_MINOR").expect("cargo sets CARGO_PKG_VERSION_MINOR");
  let expected = format!("{major}_{minor}");
  assert!(
    expected == SHIM_ABI_TAG,
    "the native shim's version tag is stale: SHIM_ABI_TAG in build.rs is {SHIM_ABI_TAG}, but the \
     package version is {major}.{minor}. Update SHIM_ABI_TAG to {expected}, then the sites \
     assert_versioned_names_are_consistent lists, which the next build will name one at a time."
  );
}

/// Fails the build when any hand-written copy of the tag disagrees with
/// [`SHIM_ABI_TAG`].
///
/// The four archive names are built from the constant, so they cannot
/// drift. Every other site spells the tag out — the functions the two
/// native halves export, the `@interface` and `@implementation` lines of
/// the classes beside them, the callback typedef, the Rust `extern`
/// declarations facing all of those, the `links` key and the two test
/// archives' `#[link]` attributes — because neither C nor Rust can build
/// a linkage name out of a value at compile time (`#[link_name]` takes a
/// literal, and no macro is allowed there).
///
/// The roster below is complete rather than representative, and it names
/// BOTH sides of every pair. One side moving alone is already loud — a
/// Rust `extern` that no longer matches its definition is a link error,
/// an `@implementation` without its `@interface` is a compile error —
/// but that is no help against the failure this guard exists for, where
/// a bump misses both halves of a name at once. That still builds, still
/// links and still passes the ABI test, while two coexisting versions
/// bind both wrappers to whichever old-named member resolved first. A
/// name this list does not carry is a name the next stamp can leave
/// behind.
///
/// So the constant is the source of truth and this reads the copies back
/// out of the files. Each site is `rerun-if-changed`, so editing one
/// re-runs the check.
fn assert_versioned_names_are_consistent() {
  let tag = SHIM_ABI_TAG;
  // (file, the exact text that must appear, required-to-exist)
  let sites: [(&str, String, bool); 22] = [
    (
      "Cargo.toml",
      format!("links = \"avanalyze_{tag}_objc_simd_shim\""),
      true,
    ),
    (
      "src/objc_simd_shim.m",
      format!("void avanalyze_{tag}_vn_point3d_position("),
      true,
    ),
    (
      "src/objc_simd_shim_test.m",
      format!("id avanalyze_{tag}_test_point3d_new("),
      true,
    ),
    (
      "src/objc_simd_shim_test.m",
      format!("@interface Avanalyze_{tag}_TestPoint3D"),
      true,
    ),
    (
      "src/objc_simd_shim_test.m",
      format!("@implementation Avanalyze_{tag}_TestPoint3D"),
      true,
    ),
    (
      "src/ffi.rs",
      format!("fn avanalyze_{tag}_vn_point3d_position("),
      true,
    ),
    (
      "src/objc_cxx_barrier.mm",
      format!("typedef void (*Avanalyze_{tag}_GuardBody)(void *context);"),
      true,
    ),
    (
      "src/objc_cxx_barrier.mm",
      format!("int32_t avanalyze_{tag}_guard("),
      true,
    ),
    ("src/ffi.rs", format!("fn avanalyze_{tag}_guard("), true),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("void avanalyze_{tag}_test_throw("),
      true,
    ),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("@interface Avanalyze_{tag}_TestUndescribableException"),
      true,
    ),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("@implementation Avanalyze_{tag}_TestUndescribableException"),
      true,
    ),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("@interface Avanalyze_{tag}_TestSentinel"),
      true,
    ),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("@implementation Avanalyze_{tag}_TestSentinel"),
      true,
    ),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("void avanalyze_{tag}_test_autorelease_sentinel("),
      true,
    ),
    (
      "src/objc_cxx_barrier_test.mm",
      format!("int32_t avanalyze_{tag}_test_sentinel_deallocations("),
      true,
    ),
    // Test sources need not survive packaging, so a missing file here is
    // not a defect — a stale name in a present one still is.
    (
      "src/tests/ffi.rs",
      format!("#[link(name = \"avanalyze_{tag}_objc_simd_shim_test\", kind = \"static\")]"),
      false,
    ),
    (
      "src/tests/ffi.rs",
      format!("fn avanalyze_{tag}_test_point3d_new("),
      false,
    ),
    (
      "src/tests/native_barrier.rs",
      format!("#[link(name = \"avanalyze_{tag}_objc_cxx_barrier_test\", kind = \"static\")]"),
      false,
    ),
    (
      "src/tests/native_barrier.rs",
      format!("fn avanalyze_{tag}_test_throw("),
      false,
    ),
    (
      "src/tests/native_barrier.rs",
      format!("fn avanalyze_{tag}_test_autorelease_sentinel("),
      false,
    ),
    (
      "src/tests/native_barrier.rs",
      format!("fn avanalyze_{tag}_test_sentinel_deallocations("),
      false,
    ),
  ];

  for (path, expected, required) in &sites {
    println!("cargo:rerun-if-changed={path}");
    let Ok(source) = std::fs::read_to_string(path) else {
      assert!(
        !required,
        "build.rs cannot read {path}, which must carry the native name {expected}"
      );
      continue;
    };
    assert!(
      source.contains(expected.as_str()),
      "the native version scope is inconsistent: SHIM_ABI_TAG in build.rs is {tag}, so {path} must \
       contain {expected}, and it does not. Every hand-written copy of the tag has to move \
       together, or two avanalyze versions in one graph bind to the same native name again."
    );
  }
}

/// Compiles the one Objective-C message send Rust cannot make itself.
///
/// See `src/objc_simd_shim.m`: rustc's `extern "C"` matches neither of
/// the conventions Apple returns a `simd_float4x4` by, so the call has
/// to be emitted by Clang. It is wanted only where Apple's frameworks
/// are, which is exactly where the rest of this crate's platform half
/// compiles; every other target builds the stubs and needs no C
/// toolchain.
///
/// The file is Objective-C rather than C on purpose. The send is typed,
/// so Clang selects the dispatcher — `objc_msgSend_stret` on x86_64,
/// where the matrix is MEMORY class, and `objc_msgSend` on arm64, where
/// it is a vector aggregate and no `_stret` variant exists at all.
/// Choosing that by hand would put an architecture table in this crate
/// whose wrong cells are memory corruption rather than wrong answers.
///
/// `src/objc_simd_shim_test.m` — the deterministic object that ABI is
/// proved against, since Vision's own numbers are inference pinned to a
/// host — is compiled into an archive of its **own**, and deliberately
/// emits no cargo link directive. A separate translation unit inside the
/// production archive would not have been enough: `-ObjC`, `-all_load`
/// and `-force_load` load every archive member that defines an
/// Objective-C class whether or not anything references it (Apple
/// QA1490), so a consumer linking with `-ObjC` — routine in Apple
/// application builds — would ship a test class and register it in the
/// runtime. An archive nothing links cannot be force-loaded. The only
/// thing that names it is a `#[link]` attribute in `src/tests/ffi.rs`,
/// which exists only in a `cfg(test)` build.
///
/// # The second translation unit, and why it is C++
///
/// `src/objc_cxx_barrier.mm` is compiled beside the shim, under the
/// same version-tag discipline and into an archive of its own. It is
/// Objective-C++ because it has to name BOTH exception worlds in one
/// `try`: `catch (NSException *)` needs the Objective-C half of the
/// compiler and `catch (const std::exception &)` the C++ half, and the
/// two are one mechanism only inside a translation unit that speaks
/// both. `src/objc_cxx_barrier_test.mm` — three synthetic throws Vision
/// cannot be asked for on demand — follows the test archive's rules.
fn build_native_half() {
  println!("cargo:rerun-if-env-changed=DOCS_RS");

  // Before the target gate: a stale tag is a defect on every host, and a
  // build that never reaches Clang should not be the one that hides it.
  // These also emit the `rerun-if-changed` for both Objective-C files.
  assert_shim_abi_tag_matches_package_version();
  assert_versioned_names_are_consistent();

  if var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
    return;
  }
  // docs.rs documents this crate for an Apple target but has no Clang
  // for one, and rustdoc never links, so there is nothing for the
  // object to be missing from. `objc2-exception-helper` — already in
  // this crate's graph, and already compiling an Objective-C file —
  // skips there for exactly this reason.
  if var("DOCS_RS").is_ok() {
    return;
  }

  objc_build()
    .file("src/objc_simd_shim.m")
    .compile(&format!("avanalyze_{SHIM_ABI_TAG}_objc_simd_shim"));

  // The test object, in an archive of its own and with no cargo link
  // directive: `cargo_metadata(false)` suppresses both the
  // `rustc-link-lib` and the `rustc-link-search` this would otherwise
  // emit, so nothing in a consumer's link line names this archive and
  // `-ObjC` has nothing to force-load. `src/tests/ffi.rs` names it in a
  // `#[link]` attribute that only a `cfg(test)` build compiles.
  objc_build()
    .file("src/objc_simd_shim_test.m")
    .cargo_metadata(false)
    .compile(&format!("avanalyze_{SHIM_ABI_TAG}_objc_simd_shim_test"));

  objcxx_build()
    .file("src/objc_cxx_barrier.mm")
    .compile(&format!("avanalyze_{SHIM_ABI_TAG}_objc_cxx_barrier"));

  // `catch (NSException *)` binds `_OBJC_EHTYPE_$_NSException`, which
  // lives in Foundation. Every other Apple framework this crate links
  // arrives through an objc2 crate's own link directive, and Foundation
  // does too — but a barrier that silently stops catching Objective-C
  // exceptions if a dependency's directive ever moves is not a barrier,
  // so the translation unit that needs the symbol names the framework
  // that defines it. A duplicate `-framework` on the link line is free.
  println!("cargo:rustc-link-lib=framework=Foundation");

  // The synthetic throws, under the test archive's rules: its own
  // archive, no cargo link directive, named only by the `#[link]`
  // attribute in `src/tests/native_barrier.rs`.
  objcxx_build()
    .file("src/objc_cxx_barrier_test.mm")
    .cargo_metadata(false)
    .compile(&format!("avanalyze_{SHIM_ABI_TAG}_objc_cxx_barrier_test"));

  // Emitted here rather than left to the production archives' own
  // metadata, so the test archives are findable by name without
  // depending on which `cc::Build` happened to print a search path
  // first. All four land in OUT_DIR.
  println!(
    "cargo:rustc-link-search=native={}",
    var("OUT_DIR").expect("cargo sets OUT_DIR")
  );
}

/// The compiler settings both Objective-C files are built with.
///
/// Deliberately without ARC: the shim retains, releases and stores
/// nothing, so owning that directly is both possible and simpler to
/// audit, and the test object's one `alloc` is balanced by the
/// `Retained` the Rust test puts it in. `objc2-exception-helper`
/// compiles its own shim the same way, for the same reason.
fn objc_build() -> cc::Build {
  let mut build = cc::Build::new();
  build
    .flag("-xobjective-c")
    // Without exception support Clang marks the function `nounwind`,
    // and a raise from the send would unwind out of a frame declared
    // not to unwind. The Rust declaration is `extern "C-unwind"` to
    // match, and `src/tests/ffi.rs` raises through this frame to prove
    // the pair holds.
    .flag("-fobjc-exceptions")
    .flag("-fexceptions");
  build
}

/// The compiler settings both Objective-C++ files are built with.
///
/// `cpp(true)` is what puts libc++ on the link line, which a typed
/// clause list needs at every step: `catch (const std::exception &)`
/// binds that class's typeinfo and calls its virtual `what()`, and the
/// `try` around it binds the C++ personality routine and the `__cxa_*`
/// entry points the unwinder reaches it through. Nothing else in this
/// crate's graph brings that runtime. `-xobjective-c++` is spelled out
/// rather than left to the `.mm` extension so the language cannot depend
/// on how `cc` decided to invoke the driver.
///
/// Exception support is not optional here in the way it is elsewhere:
/// this is the frame the whole barrier is. Without `-fexceptions` there
/// is no `try` at all, and without `-fobjc-exceptions` an
/// `NSException` is not something `catch (NSException *)` can name.
///
/// No ARC, matching the Objective-C half: the barrier retains nothing
/// and stores nothing, and the one object it touches — the exception in
/// flight — is owned by the runtime for the length of the handler.
fn objcxx_build() -> cc::Build {
  let mut build = cc::Build::new();
  build
    .cpp(true)
    .flag("-xobjective-c++")
    .flag("-std=c++17")
    .flag("-fobjc-exceptions")
    .flag("-fexceptions");
  build
}
