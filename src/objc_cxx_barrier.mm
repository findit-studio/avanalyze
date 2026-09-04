/*
 * The barrier a native exception cannot cross.
 *
 * Rust's `std::panic::catch_unwind` aborts the process the moment a
 * FOREIGN exception reaches it — `fatal runtime error: Rust cannot
 * catch foreign exceptions` — and Apple's frameworks raise foreign
 * exceptions. `objc2::exception::catch` covers one half of that: the
 * helper it calls is `@catch (id)`, which matches Objective-C objects
 * and deliberately lets everything else keep unwinding. The other half
 * is C++. Vision sits on CoreML, CoreML on Espresso and ANECF, and
 * those are C++ libraries; every `NSException` Vision raises out of a
 * model load is one their Objective-C wrapper turned into an ObjC
 * object, and nothing in the API contract says the wrapper is total.
 *
 * So this file: one `extern "C"` function that runs a callback inside a
 * C++ `try` and reports what came out as a status code plus a message,
 * instead of letting it unwind into Rust. Objective-C++ rather than C++
 * because `catch (NSException *)` needs the Objective-C half of the
 * compiler, and Objective-C and C++ exceptions are one mechanism only
 * in a translation unit that speaks both.
 *
 * # Every clause is TYPED, and there is deliberately no `catch (...)`
 *
 * A Rust panic is a foreign exception from C++'s point of view, exactly
 * as an `NSException` is a foreign exception from Rust's, so a
 * `catch (...)` here would match one. It must not, and no amount of
 * inspection inside the handler makes it safe to: the Rust Reference
 * ("Rethrowing or Disposing of Rust Panic Payloads") states that there
 * are no guarantees about the behaviour when a foreign runtime disposes
 * of or rethrows a Rust panic payload, and that an unwind originated by
 * the Rust runtime must either terminate the process or be caught by
 * that same runtime. Entering the handler at all is already the
 * unsupported act — libc++abi's `__cxa_begin_catch` runs before any
 * check the handler could make, and it calls `std::terminate` outright
 * if a C++ exception is already being handled on this thread.
 *
 * Every clause is therefore a NAMED C++ type, and an exception this
 * file cannot name is left to keep unwinding. That is what makes a Rust
 * panic's crossing supported rather than merely observed: it passes
 * THROUGH this frame — which carries unwind information, and whose one
 * piece of state, the `AvanalyzeAutoreleasePool` below, is popped by a
 * destructor the unwinder runs as a CLEANUP and not as a handler, so
 * the crossing enters no `__cxa_begin_catch` and disposes of no payload
 * this file does not own — and is caught by the Rust runtime that
 * raised it, which is exactly the arrangement `extern "C-unwind"`
 * exists for. It is also what lets `BodyPoser::extract_3d` keep the
 * `catch_unwind` it runs outside the FFI barrier for objc2's
 * debug-build encoding checks, and what keeps a CONSUMER's panic —
 * their vocabulary type's constructor runs inside an extraction guard —
 * a panic rather than a dead process.
 *
 * # What is not caught, and why that is the right residual
 *
 * A C++ exception whose type derives from neither `std::exception` nor
 * `id` — a bare `throw 42` — is not named here and unwinds into Rust,
 * where it aborts or surfaces as an opaque `catch_unwind` error; the
 * Reference leaves which of the two unspecified. That is the same
 * outcome this crate had before the barrier existed, so nothing
 * regresses, and it buys the guarantee above. It is also not a shape
 * Apple's frameworks have been observed to produce: the ANE refusal
 * that motivated this file raises an `NSException`, and the layers
 * under it throw `std::exception` subclasses.
 * `src/tests/native_barrier.rs` pins the residual with a child process,
 * so re-adding a `catch (...)` here fails a test rather than passing
 * one.
 *
 * # The barrier needs `panic = "unwind"`, and no shape of it could not
 *
 * An Apple exception has to cross ONE Rust frame — the callback — to
 * reach the `try` here, because the code that calls Vision is Rust. A
 * `panic = "abort"` consumer's rustc puts an abort-on-unwind shim on
 * every `extern "C-unwind"` boundary, so under that setting the
 * exception dies in the callback and never arrives.
 *
 * No barrier of this shape can avoid that, and this crate did not
 * acquire the constraint here: `objc2::exception::catch` — used since
 * long before this file existed, and built on exactly the same
 * geometry, a Rust `extern "C-unwind"` callback inside an Objective-C
 * `@try` — documents it in the same words ("if your Rust code is
 * compiled with panic=abort ... this cannot catch the exception"), and
 * `BodyPoser::extract_3d`'s `catch_unwind` is inert there too. The only
 * design that would escape it is one where Rust never appears between
 * the raise and the handler, which means writing every Vision call in
 * Objective-C++ and giving up objc2's typed API for all of them.
 *
 * So the promise this file makes is conditional, and says so wherever
 * it is made: under `panic = "abort"` a consumer gets exactly the
 * behaviour they had before the barrier existed — nothing is newly
 * broken, and nothing is silently weakened.
 *
 * # The callback may not be `noexcept`
 *
 * The Rust side declares the callback `extern "C-unwind"` and installs
 * NO `catch_unwind` inside it, which is not an oversight: a Rust
 * `catch_unwind` between the throw site and this `try` would catch the
 * native exception first and abort on the class mismatch, which is the
 * defect. The containment has to be here, in the frame that can name
 * the exception.
 *
 * Compiled by `build.rs` for Apple targets only, into an archive of its
 * own, without ARC — nothing here retains, releases or stores. The
 * exported name carries the crate's major.minor for the reason
 * `src/objc_simd_shim.m` spells out: C has no namespaces and Cargo
 * permits two semver-incompatible avanalyze versions in one graph.
 */

#import <Foundation/Foundation.h>
#import <objc/runtime.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>

/*
 * The two runtime calls `@autoreleasepool` itself lowers to. libobjc
 * exports both — they are in its `.tbd`, and clang emits calls to them
 * for every `@autoreleasepool` block ever compiled — but no public SDK
 * header declares them, so the declaration is written out here. They
 * are used instead of the language construct for the reason given on
 * `AvanalyzeAutoreleasePool` below.
 */
extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *context);

namespace {

/*
 * An autorelease pool that pops when an exception unwinds through it,
 * which `@autoreleasepool` does NOT.
 *
 * Clang lowers `@autoreleasepool` as a NORMAL-ONLY cleanup: the pop is
 * emitted on the fall-through and matched-handler joins, and the
 * landing pad for an exception this frame does not catch resumes
 * through `_Unwind_Resume` without it. That is invisible in a barrier
 * that catches everything, and it is a leak in this one, whose whole
 * design rests on NOT catching some things. A Rust panic passing
 * through — the supported crossing, raised by objc2's debug-build
 * checks or by a consumer's own vocabulary constructor running inside
 * an extraction guard — would strand a pool boundary on the thread and
 * every object autoreleased after it, once per attempt, for an
 * application that catches the panic and carries on.
 *
 * Measured, not assumed, on arm64 AND x86_64: a sentinel autoreleased
 * inside a language `@autoreleasepool` is deallocated on the normal and
 * matched-catch paths and NOT on the pass-through path, while the same
 * sentinel under this guard is deallocated on all three. A C++
 * destructor is emitted as a normal AND EH cleanup, which is the whole
 * difference. `src/tests/native_barrier.rs` keeps that as a test.
 */
class AvanalyzeAutoreleasePool {
 public:
  AvanalyzeAutoreleasePool() : token_(objc_autoreleasePoolPush()) {}
  ~AvanalyzeAutoreleasePool() { objc_autoreleasePoolPop(token_); }
  AvanalyzeAutoreleasePool(const AvanalyzeAutoreleasePool &) = delete;
  AvanalyzeAutoreleasePool &operator=(const AvanalyzeAutoreleasePool &) = delete;

 private:
  void *token_;
};

}  // namespace

/*
 * What came out of the guarded call. Must match `GuardStatus` in
 * `src/ffi.rs`, which is where these numbers are given meaning; a
 * status this file never returns is refused there rather than guessed
 * at.
 */
enum {
  AVANALYZE_GUARD_COMPLETED = 0,
  AVANALYZE_GUARD_OBJC_EXCEPTION = 1,
  AVANALYZE_GUARD_CXX_EXCEPTION = 2,
};

/*
 * Write a NUL-terminated message into the caller's buffer, truncating
 * rather than growing it. A zero capacity and a null buffer are both
 * "the caller does not want a message", not errors.
 *
 * Truncation can land inside a UTF-8 sequence; the Rust side reads the
 * buffer with a lossy conversion, so a split code point becomes a
 * replacement character rather than a refused message.
 */
static void avanalyze_guard_message(char *message, size_t capacity, const char *format, ...) {
  if (message == NULL || capacity == 0) {
    return;
  }
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(message, capacity, format, arguments);
  va_end(arguments);
}

/*
 * Render an `NSException` as `name: reason`.
 *
 * This runs INSIDE a catch handler, and that is what makes its own
 * exception safety load-bearing rather than tidy. While a handler
 * executes, the sibling clauses of the `try` it belongs to are no
 * longer active: an exception raised here does not fall to the next
 * clause, it leaves `avanalyze_0_6_guard` entirely and unwinds into
 * Rust — the exact outcome this file exists to prevent, reached from
 * inside the code that was reporting one.
 *
 * Both accessors are message sends, so both can raise. `@catch (id)`
 * alone is not enough: it matches Objective-C objects and lets a C++
 * exception keep going, and Apple's accessors sit on the same C++
 * layers everything else here does. So the render carries BOTH — a
 * nested C++ `try` with a typed `std::exception` clause, and the
 * Objective-C `@catch` inside it — and a receiver that will not
 * describe itself gets a fixed string.
 *
 * The clause list is typed for the same reason the outer one is, and
 * leaves the same residual: an exotic C++ throw out of `-name` would
 * still escape. Nothing calls Rust from in here, so the residual is not
 * reachable by any panic, only by an Apple accessor throwing something
 * unrelated to `std::exception` — which would be a stranger thing than
 * this file is built to survive.
 *
 * `-name` is nonnull by contract and `-reason` is not, so both are
 * checked; `-UTF8String` on a nil receiver is nil, which `%s` may not
 * be handed. Its result points into autoreleased storage, valid until
 * the pool in `avanalyze_0_6_guard` drains — which is after this
 * returns and after the message has been copied.
 */
static void avanalyze_guard_render_nsexception(NSException *exception, char *message,
                                               size_t capacity) {
  try {
    @try {
      const char *name = exception.name == nil ? NULL : [exception.name UTF8String];
      const char *reason = exception.reason == nil ? NULL : [exception.reason UTF8String];
      avanalyze_guard_message(message, capacity, "%s: %s", name == NULL ? "NSException" : name,
                              reason == NULL ? "(no reason given)" : reason);
    } @catch (id ignored) {
      (void)ignored;
      avanalyze_guard_message(message, capacity,
                              "an Objective-C exception that could not describe itself");
    }
  } catch (const std::exception &ignored) {
    (void)ignored;
    avanalyze_guard_message(message, capacity,
                            "an Objective-C exception whose description threw");
  }
}

extern "C" {

/*
 * The Rust callback. Not `noexcept`: a native exception raised by
 * anything it calls has to be able to unwind out of it and into the
 * `try` below, which is the whole point.
 */
typedef void (*Avanalyze_0_6_GuardBody)(void *context);

/*
 * Run `body(context)` under a barrier no Apple framework exception can
 * cross, and report what happened.
 *
 * Returns `AVANALYZE_GUARD_COMPLETED` when the callback returned
 * normally, and one of the exception statuses otherwise, with `message`
 * filled in (NUL-terminated, truncated to `message_capacity`). On a
 * completed call `message` is left as the empty string.
 *
 * An exception this function's clauses cannot name — a Rust panic, or a
 * C++ throw of an unrelated type — is not caught and does not return
 * here at all: it keeps unwinding. See the header of this file.
 */
int32_t avanalyze_0_6_guard(Avanalyze_0_6_GuardBody body, void *context, char *message,
                            size_t message_capacity) {
  if (message != NULL && message_capacity > 0) {
    message[0] = '\0';
  }
  /*
   * The pool spans the callback AND the handlers, and both halves of
   * that matter.
   *
   * The callback half is what the recovery made necessary. A failed
   * model load autoreleases as it unwinds — the pipeline object, the
   * exception, its reason, its call-stack arrays — and before this
   * change that did not matter, because the process died. A refusal a
   * caller can retry turns the same objects into an accumulation, one
   * per attempt, with nothing to drain them: Apple reports each as
   * "autoreleased with no pool in place - just leaking" under
   * `OBJC_DEBUG_MISSING_POOLS=YES`. The callers' own pools do not
   * help; `ffi::with_image` opens one INSIDE the guard, and the nine
   * constructors open none at all.
   *
   * The handler half is why the pool is out here rather than around
   * `body` alone. The exception being reported is reachable only while
   * the handler runs, and `-UTF8String` hands back a pointer into
   * autoreleased storage — a pool that drained at the end of the
   * callback would take both away before the message was written.
   *
   * An exception this function does not name unwinds through the pool
   * rather than out of a handler, so the pop runs as a cleanup and the
   * exception keeps going: a cleanup is not a catch, and nothing here
   * disposes of a payload it does not own. That path is exactly why the
   * pool is an RAII guard and not an `@autoreleasepool` block — see
   * `AvanalyzeAutoreleasePool`, whose destructor is the only form of
   * the pop that the unwinder runs.
   *
   * Nothing an ObjC exception owns can be released early by that pop,
   * because no ObjC exception reaches it: `catch (id)` matches every
   * `@throw`n object, so the only exceptions that pass through are C++
   * ones, whose storage `__cxa_allocate_exception` owns, and Rust
   * panics, which the Rust runtime owns.
   */
  AvanalyzeAutoreleasePool pool;
  try {
    body(context);
    return AVANALYZE_GUARD_COMPLETED;
  } catch (NSException *exception) {
    avanalyze_guard_render_nsexception(exception, message, message_capacity);
    return AVANALYZE_GUARD_OBJC_EXCEPTION;
  } catch (id exception) {
    /*
     * `@throw` accepts any object, so an Objective-C exception need not
     * be an `NSException`. `object_getClassName` is a C function that
     * reads the runtime's own metadata rather than sending
     * `-description`, so naming the class cannot raise a second time.
     */
    avanalyze_guard_message(message, message_capacity, "an Objective-C exception of class %s",
                            object_getClassName(exception));
    return AVANALYZE_GUARD_OBJC_EXCEPTION;
  } catch (const std::exception &exception) {
    /* `what()` is `noexcept`, so this handler cannot itself throw. */
    avanalyze_guard_message(message, message_capacity, "%s", exception.what());
    return AVANALYZE_GUARD_CXX_EXCEPTION;
  }
  /*
   * No `catch (...)`. Anything not named above keeps unwinding, on
   * purpose — see "Every clause is TYPED" at the top of this file. Do
   * not add one: `src/tests/native_barrier.rs` fails if it comes back.
   */
}

}  // extern "C"
