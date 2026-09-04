/*
 * A deterministic stand-in for `VNPoint3D`, so the shim's ABI can be
 * proved without asking a neural network for an opinion.
 *
 * `src/objc_simd_shim.m` exists because Rust cannot receive a
 * `simd_float4x4`. Proving it reads one correctly needs an object that
 * returns a matrix by that convention and a matrix whose value is known
 * in advance — which Vision cannot supply, because its numbers come from
 * inference on a photograph and are pinned to a host, an OS release and
 * a neural backend. This class supplies both: the same Objective-C
 * return convention, sixteen values fixed in the source.
 *
 * The matrix is `columns[col][row] = col * 4 + row + 0.5`, so all
 * sixteen entries differ and none equals its transpose partner. A read
 * that transposes, truncates to the low half of each register, loses the
 * last column, or reads a stack buffer the callee never wrote cannot
 * match it by accident.
 *
 * This is test-only code, and it stays out of a consumer's binary
 * because `build.rs` compiles it into an archive of its own that no
 * cargo link directive names. Only the `#[link]` attribute in
 * `src/tests/ffi.rs` does, and that is compiled only under `cfg(test)`.
 *
 * Being a separate translation unit inside the production archive would
 * NOT have been enough, which is worth writing down because it is the
 * intuitive answer and it is wrong: `-ObjC`, `-all_load` and
 * `-force_load` load every archive member that defines an Objective-C
 * class whether or not anything references it (Apple QA1490), and
 * `-ObjC` is routine in Apple application builds. Lazy extraction does
 * not protect a class. An archive nothing links does.
 *
 * Compiled by `build.rs` beside the shim, for Apple targets only, and
 * without ARC — see the ownership note on the constructor.
 */

#import <objc/NSObject.h>
#import <simd/simd.h>

/*
 * The entry a test compares against. Element (row, col) of the returned
 * matrix is `col * 4 + row + 0.5`, which in Apple's column-major memory
 * order is simply `0.5, 1.5, ... 15.5`.
 *
 * Deliberately NOT an affine transform: its bottom row is
 * `(3.5, 7.5, 11.5, 15.5)`. This object exists to prove the sixteen
 * floats cross the boundary intact, and nothing else. The affine gate
 * that reads a translation out of them is pure Rust and is tested
 * directly, on synthetic matrices, in `src/tests/body_pose.rs`.
 */
@interface Avanalyze_0_6_TestPoint3D : NSObject
- (simd_float4x4)position;
@end

@implementation Avanalyze_0_6_TestPoint3D
- (simd_float4x4)position {
  simd_float4x4 matrix;
  for (int col = 0; col < 4; col++) {
    for (int row = 0; row < 4; row++) {
      matrix.columns[col][row] = (float)(col * 4 + row) + 0.5f;
    }
  }
  return matrix;
}
@end

/*
 * Returns a new test point, owned by the caller at +1.
 *
 * This file is compiled without ARC, like the shim beside it, so the
 * `alloc` is balanced by the caller: the Rust side takes the +1 into an
 * `objc2` `Retained`, which releases it on drop. The class name carries
 * the crate's major.minor version for the same reason the shim's
 * function does — an Objective-C class name is a process-wide symbol,
 * and two versions of avanalyze in one graph must not define the same
 * one.
 */
id avanalyze_0_6_test_point3d_new(void) {
  return [[Avanalyze_0_6_TestPoint3D alloc] init];
}
