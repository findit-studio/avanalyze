/*
 * One Objective-C message send that Rust cannot make itself.
 *
 * Apple returns a `simd_float4x4` differently on each architecture, and
 * rustc matches neither. On arm64 it is a homogeneous vector aggregate
 * returned in v0-v3, while rustc returns any vector aggregate larger
 * than 128 bits through the x8 hidden pointer — so a Rust declaration
 * reads a buffer the callee never wrote. On x86_64 it is MEMORY class,
 * returned through a hidden pointer that the Objective-C runtime
 * reaches by a DIFFERENT dispatcher, `objc_msgSend_stret`, which does
 * not exist on arm64 at all.
 *
 * This file makes the call in Objective-C rather than selecting a
 * dispatcher by hand. The receiver is typed by a protocol that declares
 * the method's real signature, so Clang picks the correct dispatcher
 * for the target it is compiling, on every architecture, including any
 * Apple adds later. The alternative — an `#if` per architecture around
 * a cast of `objc_msgSend` / `objc_msgSend_stret` — is a table this
 * crate would have to keep correct against a matrix of architectures
 * and return-type classes, and getting one cell wrong is memory
 * corruption rather than a wrong answer.
 *
 * Compiled only for Apple targets, by `build.rs`, without ARC: nothing
 * here is retained, released, or stored.
 *
 * The exported name carries the crate's major.minor version. C has no
 * namespaces, and Cargo permits two semver-incompatible versions of one
 * crate in a single graph, so an unversioned symbol would publish two
 * definitions of the same name into one process: the linker binds both
 * Rust wrappers to whichever archive member it reaches first, and the
 * day the two shims differ that is a calling-convention mismatch rather
 * than a link error. `build.rs` version-scopes the archive name for the
 * same reason, `Cargo.toml` version-scopes the `links` key, and the
 * build refuses to run if this tag and the package version disagree.
 * `objc2-exception-helper` — already in this crate's graph — versions
 * `objc2_exception_helper_0_1_try_catch` for exactly this reason.
 */

#import <objc/objc.h>
#import <simd/simd.h>
#import <string.h>

/*
 * The signature Clang needs, and nothing else. Declaring it costs no
 * class symbol and no runtime metadata — measured, not assumed: the
 * compiled object's only external definition is the function below — so
 * this name needs no version scope. It exists so the send below is typed
 * rather than dynamic.
 */
@protocol AvanalyzePoint3D
- (simd_float4x4)position;
@end

/*
 * Reads `-[VNPoint3D position]` and copies it into `out` as 16 floats,
 * column-major, exactly as it sits in memory.
 *
 * `receiver` must be a live object responding to `position` with a
 * `simd_float4x4` return, and `out` must have room for 16 floats. The
 * selector is fixed here rather than passed in: a caller that could
 * choose it could name one with a different return type, which is a
 * calling-convention mismatch, which is undefined behaviour.
 *
 * A send that raises unwinds straight through: this function holds no
 * state and owns nothing, so there is nothing to clean up, and
 * `build.rs` compiles it with exception support so the frame carries
 * the unwind information to pass through rather than being `nounwind`.
 * The exception is caught on the Rust side, by the
 * `objc2::exception::catch` every 3-D extraction already runs inside.
 */
void avanalyze_0_6_vn_point3d_position(id receiver, float *out) {
  simd_float4x4 matrix = [(id<AvanalyzePoint3D>)receiver position];
  memcpy(out, &matrix, sizeof matrix);
}
