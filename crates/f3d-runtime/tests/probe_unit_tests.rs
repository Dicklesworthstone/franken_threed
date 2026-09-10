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
