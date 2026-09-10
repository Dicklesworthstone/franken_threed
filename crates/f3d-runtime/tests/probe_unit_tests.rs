//! Native scheduler regression; this does not establish browser execution.
use f3d_runtime::{BrowserHostServices, RuntimeBuilder};
use std::{cell::Cell, rc::Rc, sync::Arc};

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
