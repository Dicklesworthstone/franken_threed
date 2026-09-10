# f3d-runtime foundation probe

This crate links the upstream Asupersync runtime in one Wasm instance. Asupersync
owns futures, task admission, polling, wakeups, and the nonreentrant worker pump.
The static JavaScript fixture supplies timer/MessageChannel callbacks and records
observations. It does not simulate task completion or implement an executor.

`browser` enables the wasm32 probe exports. Native builds re-export the admitted
runtime entry points and provide one real-future regression. The Rust code uses
`forbid(unsafe_code)`; wasm-bindgen/js-sys are the named external browser boundary.
Host callback state releases its RefCell borrow before invoking a waker.

The probe exercises three actual timers, a real Asupersync oneshot and local joins,
1,000 MessageChannel callbacks, synchronous host-to-pump reentry, a future
that wakes itself 10,000 times, cooperative cancellation within one chunk, and
region drain before teardown. The burst observation checks for additional polls
between two microtasks in the initial turn, followed by completion, and measures
the maximum number of pump polls per pump turn (with observer-granularity turns also reported). Cooperative cancellation, region
drain before teardown, fetch abort, and all-turn burst measurement are verified when
observed by their respective probe completions.
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
