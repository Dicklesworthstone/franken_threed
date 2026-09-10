//! Real Rust futures driven by Asupersync in one browser Wasm instance.
use crate::{BrowserHostServices, RuntimeBuilder, burst::BurstCounter};
use asupersync::{
    cx::ChildRegionSpec,
    runtime::{LocalJoinHandle, PumpDrainOutcome, Runtime, RuntimeHandle},
    types::{Budget, CancelKind, CancelReason},
};
use std::{
    cell::{Cell, RefCell},
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
    task::{Context, Poll, Waker},
};
use wasm_bindgen::{JsCast, prelude::*};

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
    static BURST: RefCell<BurstCounter> = const { RefCell::new(BurstCounter::new()) };
    static STALE_DISCARDED: Cell<u32> = const { Cell::new(0) };
}

// Named browser binding boundary. The static shim supplies host callbacks only;
// it neither polls futures nor replaces the Asupersync scheduler.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = f3dHost, js_name = wait, catch)]
    fn host_wait(source: u32, callback: &js_sys::Function) -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = f3dHost, js_name = event)]
    fn event(probe: &str, step: &str, value: u32);
    #[wasm_bindgen(js_namespace = f3dHost, js_name = finish)]
    fn finish(passed: bool, detail: &str);
    #[wasm_bindgen(js_namespace = f3dHost, js_name = reenter)]
    fn host_reenter() -> bool;
    #[wasm_bindgen(js_namespace = f3dHost, js_name = turns)]
    fn host_turns() -> u32;
    #[wasm_bindgen(js_namespace = f3dHost, js_name = inflight)]
    fn host_inflight(kind: u32) -> u32;
}

#[derive(Default)]
struct WaitState {
    value: Option<u32>,
    waker: Option<Waker>,
}
struct HostWait {
    source: u32,
    started: bool,
    state: Arc<Mutex<WaitState>>,
}
impl HostWait {
    fn new(source: u32) -> Self {
        Self {
            source,
            started: false,
            state: Arc::default(),
        }
    }
}
impl Future for HostWait {
    type Output = Result<u32, JsValue>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(value) = state.value.take() {
            return Poll::Ready(Ok(value));
        }
        state.waker = Some(cx.waker().clone());
        drop(state);
        if !self.started {
            self.started = true;
            let state = Arc::clone(&self.state);
            let callback = Closure::once_into_js(move |value: u32| {
                let waker = {
                    let mut state = state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.value = Some(value);
                    state.waker.take()
                };
                // Release the lock before waking the actual runtime.
                if let Some(waker) = waker {
                    waker.wake();
                }
            });
            if let Err(error) = host_wait(self.source, callback.unchecked_ref()) {
                return Poll::Ready(Err(error));
            }
        }
        Poll::Pending
    }
}

fn require(condition: bool, message: &str) -> Result<(), JsValue> {
    if condition {
        Ok(())
    } else {
        Err(JsValue::from_str(message))
    }
}
fn join_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

struct SelfWaking {
    remaining: u32,
}
impl Future for SelfWaking {
    type Output = u32;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let turn = pump_turns();
        let total = BURST.with(|counter| {
            let mut counter = counter.borrow_mut();
            counter.record(turn);
            counter.total_polls()
        });
        if self.remaining == 0 {
            return Poll::Ready(total);
        }
        self.remaining -= 1;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

#[derive(Default)]
struct PublishedState {
    generation: u32,
    value: Option<u32>,
}

fn publish(state: &RefCell<PublishedState>, generation: u32, value: u32) -> bool {
    let mut state = state.borrow_mut();
    if generation != state.generation {
        STALE_DISCARDED.with(|counter| counter.set(counter.get() + 1));
        event("stale-result", "discarded-generation-mismatch", generation);
        false
    } else {
        state.value = Some(value);
        event("stale-result", "published", value);
        true
    }
}

async fn run(handle: RuntimeHandle) -> Result<(), JsValue> {
    event("timer", "spawn", 0);
    let mut previous = 0;
    for _ in 0..3 {
        let turn = HostWait::new(0).await?;
        require(
            turn > previous,
            "timer callback did not advance an observed host turn",
        )?;
        previous = turn;
        event("timer", "resumed", turn);
    }
    event("timer", "complete", 3);

    let (sender, mut receiver) = asupersync::channel::oneshot::channel::<u32>();
    let producer = handle.spawn_local(async move {
        sender
            .send_blocking(10)
            .map_err(|_| "oneshot receiver dropped")
    });
    let cx = handle.request_cx_with_budget(Budget::new());
    let received = receiver.recv(&cx).await.map_err(join_error)?;
    producer
        .await
        .map_err(join_error)?
        .map_err(JsValue::from_str)?;
    let consumer = handle.spawn_local(async move { received + 32 });
    let value = consumer.await.map_err(join_error)?;
    require(
        value == 42,
        "Asupersync channel/join result differs from 42",
    )?;
    event("channel-join", "complete", value);

    for index in 0..1000 {
        let turn = HostWait::new(1).await?;
        require(turn > previous, "host callback reused a turn identifier")?;
        previous = turn;
        event("host-turn", "resumed", index + 1);
    }
    event("host-turn", "complete", 1000);

    require(
        host_reenter(),
        "running Asupersync pump accepted synchronous reentry",
    )?;
    event("reentrancy", "complete", 1);

    event("burst-first-turn-and-completion", "spawn", 0);
    let burst = handle.spawn_local(SelfWaking { remaining: 10_000 });
    // Two JS microtask observations run before the next browser task. A
    // self-wake must not sneak another pump microtask into that interval.
    let extra_polls = HostWait::new(2).await?;
    require(
        extra_polls == 0,
        "self-waking future bypassed the mandatory host yield",
    )?;
    let total = burst.await.map_err(join_error)?;
    require(total == 10_001, "self-waking future lost a wake or poll")?;
    event("burst-first-turn-and-completion", "complete", total);

    event("burst-all-turns", "spawn", 0);
    let _js_ceil_avg = HostWait::new(4).await?;
    let rust_max = burst_max_polls_per_turn();
    event("burst-all-turns", "rust-max-per-pump-turn", rust_max);
    require(
        rust_max <= 4,
        "burst all turns exceeded configured burst limit 4 by Rust per-turn count",
    )?;
    event("burst-all-turns", "complete", 1);

    // 6. Cooperative cancellation probe
    let cancel_cx = handle.request_cx_with_budget(Budget::new());
    let task_cx = cancel_cx.clone();
    let cleanup_runs = Arc::new(AtomicU32::new(0));
    let cleanup = Arc::clone(&cleanup_runs);
    let observed_chunk = Arc::new(AtomicU32::new(0));
    let obs = Arc::clone(&observed_chunk);

    event("cancellation", "spawn", 0);
    let cancel_task = handle.spawn_local(async move {
        let mut chunk = 0;
        loop {
            if task_cx.is_cancelled() {
                obs.store(chunk, Ordering::SeqCst);
                event("cancellation", "observed", chunk);
                cleanup.fetch_add(1, Ordering::SeqCst);
                break;
            }
            chunk += 1;
            asupersync::runtime::yield_now().await;
        }
    });

    let _ = HostWait::new(3).await?;
    cancel_cx.cancel_fast(CancelKind::User);
    cancel_task.await.map_err(join_error)?;

    require(
        cleanup_runs.load(Ordering::SeqCst) == 1,
        "cancellation cleanup did not run exactly once",
    )?;
    event("cancellation", "complete", 1);

    // 7. Drain before teardown probe
    event("drain", "spawn", 8);
    let drain_cx = handle.request_cx_with_budget(Budget::new());
    let child_region = drain_cx
        .open_child_region(ChildRegionSpec::inherit())
        .await
        .map_err(join_error)?;
    event("drain", "region-open", 0);

    let counter = Arc::new(AtomicU32::new(0));
    for i in 0..8u32 {
        let counter = Arc::clone(&counter);
        child_region
            .cx()
            .spawn(move |_child_cx| async move {
                event("drain", "child-started", i);
                let _ = HostWait::new(1).await;
                counter.fetch_add(1, Ordering::SeqCst);
                event("drain", "child-done", i);
            })
            .map_err(join_error)?;
        event("drain", "spawned", i);
    }

    child_region
        .cancel(CancelReason::user("probe region teardown"))
        .map_err(join_error)?;
    event("drain", "cancel-requested", 0);
    child_region.close().await.map_err(join_error)?;

    let at_close = counter.load(Ordering::SeqCst);
    event("drain", "closed", at_close);
    require(
        at_close == 8,
        "region close returned before all children drained",
    )?;

    // Two more host turns to verify no late publication occurs after region close
    let _ = HostWait::new(1).await?;
    let _ = HostWait::new(1).await?;
    let after = counter.load(Ordering::SeqCst);
    require(
        after == at_close,
        "late publication after region close",
    )?;

    event("drain", "settled", after);
    event("drain", "complete", at_close);

    // 8. Stale result generation check probe
    STALE_DISCARDED.with(|counter| counter.set(0));
    let state = Rc::new(RefCell::new(PublishedState {
        generation: 1,
        value: None,
    }));
    event("stale-result", "spawn", 1);

    let region_a_cx = handle.request_cx_with_budget(Budget::new());
    let region_a = region_a_cx
        .open_child_region(ChildRegionSpec::inherit())
        .await
        .map_err(join_error)?;

    let child_a_state = Rc::clone(&state);
    region_a
        .cx()
        .spawn_local(move |_child_cx| async move {
            let _ = HostWait::new(3).await;
            let _ = HostWait::new(3).await;
            publish(&child_a_state, 1, 1);
        })
        .map_err(join_error)?;

    state.borrow_mut().generation = 2;
    event("stale-result", "replaced", 2);
    region_a
        .cancel(CancelReason::user("replaced by generation 2"))
        .map_err(join_error)?;
    region_a.close().await.map_err(join_error)?;

    require(
        state.borrow().value.is_none(),
        "late child published into replaced region",
    )?;

    let region_b_cx = handle.request_cx_with_budget(Budget::new());
    let region_b = region_b_cx
        .open_child_region(ChildRegionSpec::inherit())
        .await
        .map_err(join_error)?;

    let child_b_state = Rc::clone(&state);
    region_b
        .cx()
        .spawn_local(move |_child_cx| async move {
            let _ = HostWait::new(3).await;
            publish(&child_b_state, 2, 2);
        })
        .map_err(join_error)?;
    region_b.close().await.map_err(join_error)?;

    require(
        state.borrow().value == Some(2),
        "replacement region value missing",
    )?;
    let discarded_count = STALE_DISCARDED.with(|counter| counter.get());
    require(
        discarded_count == 1,
        "stale result discarded count must be 1",
    )?;
    event("stale-result", "complete", discarded_count);

    // 9. Fetch abort probe
    event("fetch-abort", "spawn", 0);
    let fetch_cancel_cx = handle.request_cx_with_budget(Budget::new());
    let task_cx = fetch_cancel_cx.clone();

    let fetch_task: LocalJoinHandle<Result<u32, JsValue>> = handle.spawn_local(async move {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("missing window"))?;
        let controller = web_sys::AbortController::new()?;
        let signal = controller.signal();

        let init = web_sys::RequestInit::new();
        init.set_method("GET");
        init.set_signal(Some(&signal));

        let fetch_promise = window.fetch_with_str_and_init("/slow-resource", &init);
        let response_val = wasm_bindgen_futures::JsFuture::from(fetch_promise).await?;
        let response: web_sys::Response = response_val.dyn_into()?;
        let body = response
            .body()
            .ok_or_else(|| JsValue::from_str("missing response body"))?;
        let reader: web_sys::ReadableStreamDefaultReader = body.get_reader().dyn_into()?;

        let mut bytes_received = 0u32;
        let mut saw_abort = false;

        let is_abort = |val: &JsValue| {
            js_sys::Reflect::get(val, &JsValue::from_str("name"))
                .ok()
                .and_then(|name| name.as_string())
                .as_deref()
                == Some("AbortError")
                || format!("{val:?}").contains("AbortError")
        };

        loop {
            if task_cx.is_cancelled() {
                controller.abort();
                let stream_err = wasm_bindgen_futures::JsFuture::from(reader.read()).await.err();
                if let Some(ref err) = stream_err {
                    if is_abort(err) {
                        saw_abort = true;
                    }
                }
                break;
            }

            match wasm_bindgen_futures::JsFuture::from(reader.read()).await {
                Ok(chunk_val) => {
                    let done = js_sys::Reflect::get(&chunk_val, &JsValue::from_str("done"))
                        .ok()
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if done {
                        break;
                    }
                    let val = js_sys::Reflect::get(&chunk_val, &JsValue::from_str("value"))?;
                    let chunk_arr = js_sys::Uint8Array::new(&val);
                    bytes_received += chunk_arr.length();
                }
                Err(err) => {
                    if is_abort(&err) {
                        saw_abort = true;
                    }
                    break;
                }
            }
        }

        require(saw_abort, "fetch stream read did not reject with AbortError")?;
        Ok::<u32, JsValue>(bytes_received)
    });

    for _ in 0..20 {
        let _ = HostWait::new(0).await?;
    }
    fetch_cancel_cx.cancel_fast(CancelKind::User);
    let joined = fetch_task.await.map_err(join_error)?;
    let bytes_received: u32 = joined?;
    require(
        bytes_received > 0,
        "fetch abort completed without reading any bytes",
    )?;
    event("fetch-abort", "aborted", bytes_received);
    event("fetch-abort", "complete", 1);

    // 10. Post-teardown idle probe
    event("idle", "spawn", 0);
    let _ = HostWait::new(3).await?;
    let _ = HostWait::new(3).await?;
    let _ = HostWait::new(3).await?;

    let w = host_inflight(0);
    let f = host_inflight(1);
    event("idle", "pending-waits", w);
    event("idle", "pending-fetches", f);
    require(w == 0, "leaked host wait after teardown")?;
    require(f == 0, "leaked fetch after teardown")?;

    let p1 = pump_turns();
    let _ = HostWait::new(3).await?;
    let p2 = pump_turns();
    let quiet_turns = p2.saturating_sub(p1);
    event("idle", "quiet-window-pump-turns", quiet_turns);
    require(
        quiet_turns <= 2,
        "pump ran without work during quiet window",
    )?;
    event("idle", "complete", quiet_turns);

    Ok(())
}

const HOST_CAPABILITIES: &[&str] = &[
    "f3dHost.wait",
    "f3dHost.event",
    "f3dHost.finish",
    "f3dHost.reenter",
    "f3dHost.turns",
    "f3dHost.inflight",
    "AbortController",
    "fetch",
    "MessageChannel",
    "queueMicrotask",
    "setTimeout",
];

fn check_host_capabilities() -> Result<(), JsValue> {
    let global = js_sys::global();
    for &name in HOST_CAPABILITIES {
        let is_fn = if let Some(sub) = name.strip_prefix("f3dHost.") {
            let host = js_sys::Reflect::get(&global, &JsValue::from_str("f3dHost"))
                .unwrap_or_else(|_| JsValue::undefined());
            if host.is_undefined() || host.is_null() {
                false
            } else {
                let member = js_sys::Reflect::get(&host, &JsValue::from_str(sub))
                    .unwrap_or_else(|_| JsValue::undefined());
                member.is_function()
            }
        } else {
            let val = js_sys::Reflect::get(&global, &JsValue::from_str(name))
                .unwrap_or_else(|_| JsValue::undefined());
            val.is_function()
        };
        if !is_fn {
            return Err(JsValue::from_str(&format!(
                "unsupported-host: missing capability {name}"
            )));
        }
    }
    Ok(())
}

/// Starts the actual Asupersync task. The page observes completion via the host
/// callback; no second Rust executor is used to drive an exported async function.
#[wasm_bindgen]
pub fn start_probes() -> Result<(), JsValue> {
    require(
        !RUNTIME.with(|runtime| runtime.borrow().is_some()),
        "probe already started",
    )?;
    if let Err(error) = check_host_capabilities() {
        let message = error
            .as_string()
            .unwrap_or_else(|| "unsupported-host: missing capability unknown".to_string());
        let failed_name = message
            .strip_prefix("unsupported-host: missing capability ")
            .unwrap_or("");
        let index = HOST_CAPABILITIES
            .iter()
            .position(|&name| name == failed_name)
            .map_or(0, |idx| idx as u32);
        event("unsupported-host", "missing", index);
        finish(false, &message);
        return Err(error);
    }
    event("unsupported-host", "complete", HOST_CAPABILITIES.len() as u32);

    let runtime = RuntimeBuilder::new()
        .worker_threads(1)
        .browser_host_services(Arc::new(BrowserHostServices::with_burst_limit(4)))
        .build()
        .map_err(join_error)?;
    let handle = runtime.handle();
    let _task = runtime.spawn_local(async move {
        match run(handle).await {
            Ok(()) => finish(true, "real Asupersync browser probes completed"),
            Err(error) => finish(false, &format!("{error:?}")),
        }
    });
    RUNTIME.with(|slot| *slot.borrow_mut() = Some(runtime));
    Ok(())
}

/// Called synchronously from the browser import while an Asupersync task is
/// being polled. The upstream guard must reject entry before locking the worker.
#[wasm_bindgen]
pub fn reenter_probe() -> bool {
    RUNTIME.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(Runtime::browser_pump)
            .is_some_and(|pump| pump.drain_batch(1) == PumpDrainOutcome::ReentrantPrevented)
    })
}

/// Actual number of calls to the self-waking Rust future's `poll` method.
#[wasm_bindgen]
pub fn burst_polls() -> u32 {
    BURST.with(|b| b.borrow().total_polls())
}

/// Maximum number of calls to `SelfWaking::poll` in any single pump turn.
#[wasm_bindgen]
pub fn burst_max_polls_per_turn() -> u32 {
    BURST.with(|b| b.borrow().max_polls_in_a_turn())
}

/// Actual number of pump turns executed by the Asupersync single-worker pump.
#[wasm_bindgen]
pub fn pump_turns() -> u32 {
    RUNTIME.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(Runtime::browser_pump)
            .map_or(0, |pump| pump.pump_turns())
    })
}
