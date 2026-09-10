# f3d-runtime foundation probe

This crate links the upstream Asupersync runtime in one Wasm instance. Asupersync
owns futures, task admission, polling, wakeups, and the nonreentrant worker pump.
The static JavaScript fixture supplies timer/MessageChannel callbacks and records
observations. It does not simulate task completion or implement an executor.

`browser` enables the wasm32 probe exports. Native builds re-export the admitted
runtime entry points and provide native unit tests. Native unit tests use manual pump
stepping as a fake host with controllable delays and establish scheduler ordering
only, not browser execution. The Rust code uses
`forbid(unsafe_code)`; wasm-bindgen/js-sys are the named external browser boundary.
Host callback state releases its RefCell borrow before invoking a waker.

The probe exercises three actual timers, a real Asupersync oneshot and local joins,
1,000 MessageChannel callbacks, synchronous host-to-pump reentry, a future
that wakes itself 10,000 times, cooperative cancellation within one chunk, and
region drain before teardown. The burst observation checks for additional polls
between two microtasks in the initial turn, followed by completion, and verifies
that no pump turn executes more than 4 polls via the Rust per-turn counter keyed
by upstream `pump_turns`. The JavaScript `ceil-avg-per-pump-turn` value is a ceiling-average
diagnostic across observer intervals, and archived runs before this change over-claimed
the maximum polls per pump turn by relying on that average. Cooperative cancellation, region
drain before teardown, stale result generation check, fetch abort, post-teardown idle state,
all-turn burst measurement, and unsupported-host detection are verified when observed by
their respective probe completions and test gates. The `stale-result` probe is the
generation-checked publication prototype at probe scope with a probe-local state cell, not scene
state, and the late child is run to completion by region close and its value is discarded by
the generation check, not by dropping the future. The `idle` probe counts host waits registered
through `f3dHost.wait` excluding microtask waits and Rust-initiated fetches through the wrapped
global `fetch`, not the fixture's own event posts; a fetch is counted only until its response promise settles, so a leaked body reader after headers arrive is not detected by this probe. A cancellation check inside a synchronous kernel
cannot promise that a main-thread cancellation callback runs before the browser regains control,
so only observed latency is logged, never a guarantee.
Unsupported-host detection verifies that removed or missing required host functions
and globals (`f3dHost.wait`, `f3dHost.event`, `f3dHost.finish`, `f3dHost.reenter`, `f3dHost.turns`,
`f3dHost.inflight`, `AbortController`, `fetch`, `MessageChannel`, `queueMicrotask`, `setTimeout`) terminate
immediately before building an Asupersync runtime with a defined error within one host turn
and no hang. This covers missing or deleted globals and host functions, not a hook that
is present but fails to invoke its callback.
Events use actual performance.now() values and callback turn identifiers. Error or timeout is failure.

Device loss and physical M5/iPhone runs
remain separate unmet foundation criteria.
Runtime ownership is retained for the test page lifetime. This is development
linkage to an upstream checkout, not an immutable release dependency pin; the Cargo path is
the absolute, nonportable /Users/jemanuel/dp/asupersync so remote builds compile the synced
upstream rather than a stale checkout resolved through the worker-side project alias.
No acceleration, rendering, complete compatibility, or passed browser run is
claimed by the presence of this code. Browser results require an actual Wasm
build and execution in a named installed browser.
