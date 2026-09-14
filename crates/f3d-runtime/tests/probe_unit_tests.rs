//! Native scheduler regressions driven by manual pump stepping as a fake host;
//! these establish scheduler ordering only and do not establish browser execution.
use asupersync::{
    cx::ChildRegionSpec,
    types::{Budget, CancelKind, CancelReason},
};
use f3d_runtime::{BrowserHostServices, RuntimeBuilder, publication::PublishedState};
use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::Arc,
    task::{Context, Poll},
};

struct YieldOnce {
    yielded: bool,
}

impl YieldOnce {
    fn new() -> Self {
        Self { yielded: false }
    }
}

impl Future for YieldOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[test]
fn asupersync_polls_a_real_local_future() {
    let runtime = RuntimeBuilder::new()
        .worker_threads(1)
        .browser_host_services(Arc::new(BrowserHostServices::new()))
        .build()
        .expect("threadless runtime");
    let value = Rc::new(Cell::new(0));
    let output = Rc::clone(&value);
    let task = runtime.spawn_local(async move {
        output.set(42);
        output
    });
    assert_eq!(value.get(), 0, "spawn must not fabricate a completed task");
    assert!(!task.is_finished());
    assert!(runtime.browser_pump().expect("upstream pump").step());
    assert_eq!(value.get(), 42);
    assert!(task.is_finished());
}

#[test]
fn burst_counter_rejects_five_plus_three_across_two_turns() {
    use f3d_runtime::burst::BurstCounter;

    let mut counter = BurstCounter::new();
    for _ in 0..5 {
        counter.record(1);
    }
    for _ in 0..3 {
        counter.record(2);
    }

    assert_eq!(counter.max_polls_in_a_turn(), 5);
    assert_eq!(counter.total_polls(), 8);
    assert!(
        counter.max_polls_in_a_turn() > 4,
        "bound-4 check must fail when a turn has 5 polls"
    );

    let old_formula = (8.0_f64 / 2.0_f64).ceil() as u32;
    assert_eq!(
        old_formula, 4,
        "old formula ceil(8 / 2) equals 4, which would falsely pass bound 4"
    );
}

#[test]
fn generation_check_discards_stale_publication() {
    let mut state = PublishedState::new(1);
    assert!(state.try_publish(1, 10));
    assert_eq!(state.value(), Some(10));

    let next_gen = state.replace();
    assert_eq!(next_gen, Some(2));
    assert_eq!(state.value(), Some(10), "replace preserves previous value");
    assert!(!state.try_publish(1, 20));
    assert_eq!(state.value(), Some(10));

    assert!(state.try_publish(2, 30));
    assert_eq!(state.value(), Some(30));
}

#[test]
fn publication_exhaustion_at_u32_max_rejects_further_tokens_and_preserves_value() {
    let mut state = PublishedState::new(u32::MAX - 1);
    assert!(state.try_publish(u32::MAX - 1, 100));
    assert_eq!(state.value(), Some(100));

    // MAX-1 -> MAX succeeds:
    let max_gen = state.replace();
    assert_eq!(max_gen, Some(u32::MAX));
    assert_eq!(state.value(), Some(100), "replace preserves previous value");

    // Publishing under active MAX token succeeds:
    assert!(state.try_publish(u32::MAX, 200));
    assert_eq!(state.value(), Some(200));

    // Next replace overflows u32 and explicitly exhausts:
    let exhausted = state.replace();
    assert_eq!(
        exhausted, None,
        "replace beyond u32::MAX must return None (explicit exhaustion)"
    );
    assert_eq!(
        state.value(),
        Some(200),
        "exhaustion preserves previous value"
    );

    // Reject both 0 and MAX tokens after exhaustion:
    assert!(
        !state.try_publish(0, 300),
        "token 0 must be rejected after exhaustion (no wrap to 0)"
    );
    assert!(
        !state.try_publish(u32::MAX, 300),
        "token MAX must be rejected after exhaustion"
    );
    assert_eq!(
        state.value(),
        Some(200),
        "rejected publications must leave previous value untouched"
    );

    // Subsequent replace calls remain exhausted and continue rejecting:
    assert_eq!(state.replace(), None);
    assert!(!state.try_publish(0, 400));
    assert_eq!(state.value(), Some(200));
}

#[test]
fn region_close_drains_all_children_before_returning() {
    let runtime = RuntimeBuilder::new()
        .worker_threads(1)
        .browser_host_services(Arc::new(BrowserHostServices::new()))
        .build()
        .expect("threadless runtime");
    let handle = runtime.handle();
    let counter = Rc::new(Cell::new(0u32));
    let counter_for_children = Rc::clone(&counter);

    let driver = runtime.spawn_local(async move {
        let drain_cx = handle.request_cx_with_budget(Budget::new());
        let child_region = drain_cx
            .open_child_region(ChildRegionSpec::inherit())
            .await
            .expect("open child region");

        for _ in 0..4 {
            let c = Rc::clone(&counter_for_children);
            child_region
                .cx()
                .spawn_local(move |_child_cx| async move {
                    YieldOnce::new().await;
                    YieldOnce::new().await;
                    c.set(c.get() + 1);
                })
                .expect("spawn child");
        }

        child_region
            .cancel(CancelReason::user("test teardown"))
            .expect("cancel child region");
        child_region.close().await.expect("close child region");
    });

    let pump = runtime.browser_pump().expect("upstream pump");
    let mut steps = 0;
    while !driver.is_finished() {
        assert!(steps < 100, "driver task must finish within 100 steps");
        pump.step();
        steps += 1;
    }

    assert_eq!(counter.get(), 4, "counter must be 4 at close");
    for _ in 0..5 {
        pump.step();
    }
    assert_eq!(
        counter.get(),
        4,
        "counter must still be 4 after stepping 5 more times"
    );
}

#[test]
fn cancellation_observed_within_one_chunk_and_cleanup_once() {
    let runtime = RuntimeBuilder::new()
        .worker_threads(1)
        .browser_host_services(Arc::new(BrowserHostServices::new()))
        .build()
        .expect("threadless runtime");
    let cancel_cx = runtime.handle().request_cx_with_budget(Budget::new());
    let task_cx = cancel_cx.clone();

    let chunk_count = Rc::new(Cell::new(0u32));
    let chunk_clone = Rc::clone(&chunk_count);
    let cleanup_counter = Rc::new(Cell::new(0u32));
    let cleanup_clone = Rc::clone(&cleanup_counter);

    let task = runtime.spawn_local(async move {
        let mut chunk = 0u32;
        loop {
            if task_cx.is_cancelled() {
                break;
            }
            chunk += 1;
            chunk_clone.set(chunk);
            YieldOnce::new().await;
        }
        cleanup_clone.set(cleanup_clone.get() + 1);
    });

    let pump = runtime.browser_pump().expect("upstream pump");
    // One step may poll the task more than once up to the pump's burst limit,
    // so only require that at least one chunk ran before cancellation.
    pump.step();
    pump.step();
    pump.step();
    assert!(chunk_count.get() >= 1, "at least one chunk must run before cancellation");

    cancel_cx.cancel_fast(CancelKind::User);
    let chunk_before = chunk_count.get();

    let mut steps = 0;
    while !task.is_finished() {
        assert!(steps < 10, "task must finish within 10 steps");
        pump.step();
        steps += 1;
    }

    assert!(
        chunk_count.get() - chunk_before <= 1,
        "cancellation observed within one chunk, never running an extra chunk"
    );
    assert_eq!(
        cleanup_counter.get(),
        1,
        "cleanup counter must be exactly 1 (cleanup ran once, no leak, no duplication)"
    );
}

