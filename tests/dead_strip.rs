//! Every integration test links with `-dead_strip_dylibs` (`build.rs`),
//! which drops the load command of a framework a binary imports no symbol
//! from. avanalyze imports none from Vision, so here the linker has dropped
//! it, and an entry point still builds its requests: every constructor
//! loads Vision by name once the system floor is met (`require_floor`).

#![cfg(target_os = "macos")]

use std::{
  ffi::{CStr, c_char, c_void},
  process::Command,
};

use avanalyze::{AnalyzeOptions, VisionAnalyzer};

unsafe extern "C" {
  fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
}

const VISION: &CStr = c"/System/Library/Frameworks/Vision.framework/Vision";

/// `RTLD_LAZY | RTLD_NOLOAD` in `<dlfcn.h>`: a handle only where the image
/// is already loaded.
const LOADED_ONLY: i32 = 0x1 | 0x10;

#[test]
fn an_entry_point_builds_where_the_linker_dropped_vision() {
  let binary = std::env::current_exe().expect("the test binary");
  let listed = Command::new("otool")
    .arg("-L")
    .arg(&binary)
    .output()
    .expect("otool, from the developer tools");
  assert!(listed.status.success(), "otool -L {}", binary.display());
  let libraries = String::from_utf8_lossy(&listed.stdout);
  assert!(
    !libraries.contains("Vision.framework"),
    "the linker kept Vision: {libraries}"
  );

  let analyzer = VisionAnalyzer::new(&AnalyzeOptions::new());
  assert!(analyzer.is_ok(), "{:?}", analyzer.err());
  // SAFETY: `VISION` is NUL-terminated; `RTLD_NOLOAD` loads nothing.
  let handle = unsafe { dlopen(VISION.as_ptr(), LOADED_ONLY) };
  assert!(!handle.is_null(), "Vision is not loaded");
}
