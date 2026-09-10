//! Real Rust futures driven by Asupersync in one browser Wasm instance.
use crate::{BrowserHostServices, RuntimeBuilder};
use asupersync::{
    runtime::{PumpDrainOutcome, Runtime, RuntimeHandle},
    types::Budget,
};
use std::{
    cell::{Cell, RefCell},
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::Arc,
    task::{Context, Poll, Waker},
};
use wasm_bindgen::{JsCast, prelude::*};

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
    static BURST_POLLS: Cell<u32> = const { Cell::new(0) };
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
}

#[derive(Default)]
struct WaitState {
    value: Option<u32>,
    waker: Option<Waker>,
}
struct HostWait {
    source: u32,
    started: bool,
    state: Rc<RefCell<WaitState>>,
}
impl HostWait {
    fn new(source: u32) -> Self {
        Self {
            source,
            started: false,
            state: Rc::default(),
        }
    }
}
impl Future for HostWait {
    type Output = Result<u32, JsValue>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(value) = self.state.borrow_mut().value.take() {
            return Poll::Ready(Ok(value));
        }
        self.state.borrow_mut().waker = Some(cx.waker().clone());
        if !self.started {
            self.started = true;
            let state = Rc::clone(&self.state);
            let callback = Closure::once_into_js(move |value: u32| {
                let waker = {
                    let mut state = state.borrow_mut();
                    state.value = Some(value);
                    state.waker.take()
                };
                // Release the RefCell borrow before waking the actual runtime.
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
        let count = BURST_POLLS.with(|counter| {
            let count = counter.get() + 1;
            counter.set(count);
            count
        });
        if self.remaining == 0 {
            return Poll::Ready(count);
        }
        self.remaining -= 1;
        cx.waker().wake_by_ref();
        Poll::Pending
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
    BURST_POLLS.with(Cell::get)
}
