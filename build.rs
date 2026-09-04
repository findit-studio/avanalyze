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

  build_objc_simd_shim();
}

fn use_feature(feature: &str) {
  println!("cargo:rustc-cfg={}", feature);
}

/// The major.minor version every exported native name carries.
///
/// C has no namespaces, and Cargo permits two semver-incompatible
/// versions of one crate in a single graph, so the shim's function, its
/// test object's class, the static archive below and the `links` key in
/// `Cargo.toml` are all scoped by this tag — the reasoning is on
/// `src/objc_simd_shim.m`. [`assert_shim_abi_tag_matches_package_version`]
/// keeps it honest.
const SHIM_ABI_TAG: &str = "0_5";

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
/// The archive name is derived from the constant, so it cannot drift.
/// Every other site spells the tag out — a C function name, an
/// Objective-C class name, two Rust `extern` declarations, the `links`
/// key, and the test archive's `#[link]` attribute — because neither C
/// nor Rust can build a linkage name out of a value at compile time
/// (`#[link_name]` takes a literal, and no macro is allowed there). A
/// bump that updated the constant and missed one of those would still
/// build and still pass the ABI test, while two coexisting versions
/// bound both wrappers to whichever old-named member resolved first.
///
/// So the constant is the source of truth and this reads the copies back
/// out of the files. Each site is `rerun-if-changed`, so editing one
/// re-runs the check.
fn assert_versioned_names_are_consistent() {
  let tag = SHIM_ABI_TAG;
  // (file, the exact text that must appear, required-to-exist)
  let sites: [(&str, String, bool); 6] = [
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
      format!("@implementation Avanalyze_{tag}_TestPoint3D"),
      true,
    ),
    (
      "src/ffi.rs",
      format!("fn avanalyze_{tag}_vn_point3d_position("),
      true,
    ),
    // Test sources need not survive packaging, so a missing file here is
    // not a defect — a stale name in a present one still is.
    (
      "src/tests/ffi.rs",
      format!("#[link(name = \"avanalyze_{tag}_objc_simd_shim_test\", kind = \"static\")]"),
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
fn build_objc_simd_shim() {
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

  // Emitted here rather than left to the production archive's own
  // metadata, so the test archive is findable by name without depending
  // on which `cc::Build` happened to print a search path first. Both
  // land in OUT_DIR.
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
