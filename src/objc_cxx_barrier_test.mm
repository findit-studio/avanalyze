/*
 * Four throws that never happen in production, so the barrier beside
 * them can be proved against each one.
 *
 * `src/objc_cxx_barrier.mm` names three C++ types and leaves everything
 * else unwinding, and it renders what it catches from inside a handler.
 * Apple's frameworks reach exactly one of those cases on this host — a
 * denied Neural Engine raises an `NSException` — and there is no input
 * to Vision that reaches the rest on demand. Asking Vision for them
 * would be asking a neural network to misbehave in a specific way on a
 * specific OS; this file asks `throw` statements instead, which cannot
 * drift.
 *
 * It is test-only code and stays out of a consumer's binary the way
 * `src/objc_simd_shim_test.m` does: `build.rs` compiles it into an
 * archive of its own that emits no cargo link directive, so nothing in
 * a consumer's link line names it. Only the `#[link]` attribute in
 * `src/tests/native_barrier.rs` does, and that is compiled only under
 * `cfg(test)`. That separation is load-bearing here rather than
 * merely tidy: this file DOES define an Objective-C class, and `-ObjC`
 * force-loads every archive member that defines one whether or not
 * anything references it (Apple QA1490). An archive nothing links
 * cannot be force-loaded — so a class whose whole purpose is to throw
 * from an accessor never reaches a consumer's runtime.
 *
 * Compiled by `build.rs` beside the barrier, for Apple targets only,
 * and without ARC. The exported name carries the crate's major.minor
 * for the reason `src/objc_simd_shim.m` spells out.
 */

#import <Foundation/Foundation.h>

#include <cstdint>
#include <stdexcept>

/*
 * Which throw to make. Must match `SyntheticThrow` in
 * `src/tests/native_barrier.rs`.
 */
enum {
  AVANALYZE_TEST_THROW_NOTHING = 0,
  AVANALYZE_TEST_THROW_NSEXCEPTION = 1,
  AVANALYZE_TEST_THROW_STD_EXCEPTION = 2,
  AVANALYZE_TEST_THROW_INT = 3,
  AVANALYZE_TEST_THROW_UNDESCRIBABLE_NSEXCEPTION = 4,
};

/*
 * An `NSException` whose `-reason` throws a C++ exception.
 *
 * The barrier renders a caught `NSException` by sending it `-name` and
 * `-reason`, and it does that from INSIDE a catch handler, where the
 * sibling clauses of the enclosing `try` are no longer active. An
 * exception raised by one of those sends therefore does not fall
 * through to the next clause — it leaves the trampoline and unwinds
 * into Rust, which is the process-fatal outcome the whole file exists
 * to prevent, reached from the code that was reporting a failure.
 *
 * Apple's own accessors do not do this. The point is that the barrier
 * must not DEPEND on their not doing it, and only a receiver that
 * really throws can say whether it depends on it or not.
 *
 * The class name carries the crate's major.minor, like every other
 * Objective-C class this crate defines: a class name is a process-wide
 * symbol and two avanalyze versions in one graph must not define the
 * same one.
 */
@interface Avanalyze_0_5_TestUndescribableException : NSException
@end

@implementation Avanalyze_0_5_TestUndescribableException
- (NSString *)reason {
  throw std::runtime_error("a synthetic throw from -reason");
}
@end

/*
 * An object that counts its own deallocations, so a test can ask
 * whether the barrier's autorelease pool really popped.
 *
 * The pool is the thing under test and it is not observable directly:
 * `objc_autoreleasePoolPop` returns nothing and the runtime exposes no
 * depth. What IS observable is the fate of an object autoreleased into
 * it — released when the pool pops, and stranded when it does not. So
 * this counts `-dealloc`, and the test compares the count across a
 * guarded call.
 *
 * It matters on the path where nothing is caught. Clang lowers
 * `@autoreleasepool` as a normal-only cleanup, so a pass-through unwind
 * skips the pop; a C++ destructor does not. This class is how that
 * difference is asserted rather than described.
 */
@interface Avanalyze_0_5_TestSentinel : NSObject
@end

static int32_t g_avanalyze_test_sentinel_deallocations = 0;

@implementation Avanalyze_0_5_TestSentinel
- (void)dealloc {
  g_avanalyze_test_sentinel_deallocations++;
  [super dealloc];
}
@end

extern "C" {

/*
 * Autoreleases one sentinel into whatever pool is currently innermost.
 *
 * Called from inside a guarded closure, that is the barrier's own pool.
 * Compiled without ARC, so the `autorelease` is written out.
 */
void avanalyze_0_5_test_autorelease_sentinel(void) {
  [[[Avanalyze_0_5_TestSentinel alloc] init] autorelease];
}

/*
 * How many sentinels have been deallocated in this process.
 *
 * Read before and after a guarded call; the difference says whether the
 * pool popped. Not atomic, and not required to be: every caller is a
 * single-threaded test.
 */
int32_t avanalyze_0_5_test_sentinel_deallocations(void) {
  return g_avanalyze_test_sentinel_deallocations;
}

/*
 * Throws one of the four, or returns.
 *
 * The `int` case is deliberately a type with no relation to
 * `std::exception`: it is the one shape the barrier's typed clauses
 * cannot name, and it is what the residual test uses to prove no
 * `catch (...)` has come back.
 */
void avanalyze_0_5_test_throw(int32_t kind) {
  switch (kind) {
    case AVANALYZE_TEST_THROW_NSEXCEPTION:
      [NSException raise:@"AvanalyzeSyntheticException"
                  format:@"a synthetic Objective-C exception"];
      break;
    case AVANALYZE_TEST_THROW_STD_EXCEPTION:
      throw std::runtime_error("a synthetic std::runtime_error");
    case AVANALYZE_TEST_THROW_INT:
      throw 42;
    case AVANALYZE_TEST_THROW_UNDESCRIBABLE_NSEXCEPTION:
      @throw [Avanalyze_0_5_TestUndescribableException
          exceptionWithName:@"AvanalyzeUndescribableException"
                     reason:nil
                   userInfo:nil];
    default:
      break;
  }
}

}  // extern "C"
