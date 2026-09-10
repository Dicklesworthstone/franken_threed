//! Unit tests for `f3d-runtime` browser validation probes and lifecycle invariants.

#![forbid(unsafe_code)]

use f3d_runtime::probe::{
    BrowserExecutionHarness, BrowserProbeConfig, ProbeEventKind, ProbeEventLogger, ProbeKind,
    ReentrancyGuardMock,
};
use f3d_runtime::{BrowserRuntimeConfig, BrowserRuntimeCore};
use std::sync::Arc;

#[test]
fn test_timer_sequence_probe_monotonicity() {
    let config = BrowserProbeConfig {
        timer_steps: 3,
        timer_step_ms: 5,
        ..BrowserProbeConfig::default()
    };
    let logger = Arc::new(ProbeEventLogger::new());
    let harness = BrowserExecutionHarness::new(config, Arc::clone(&logger));

    let summary = harness.run_timer_probe();
    assert_eq!(summary.probe, ProbeKind::TimerSequence);
    assert!(summary.passed, "Timer probe must pass monotonic checks");
    assert_eq!(summary.macrotasks_observed, 3);
    assert_eq!(summary.tasks_executed, 1);

    let events = logger.snapshot();
    assert!(events.len() >= 5); // Spawn + 3 Polls + Complete
    assert_eq!(events.first().unwrap().event, ProbeEventKind::Spawn);
    assert_eq!(events.last().unwrap().event, ProbeEventKind::Complete);
}

#[test]
fn test_channel_join_chain_probe_determinism() {
    let config = BrowserProbeConfig::default();
    let logger = Arc::new(ProbeEventLogger::new());
    let harness = BrowserExecutionHarness::new(config, Arc::clone(&logger));

    let summary = harness.run_channel_join_chain_probe();
    assert_eq!(summary.probe, ProbeKind::ChannelJoinChain);
    assert!(summary.passed, "Channel/join chain must produce expected output");
    assert_eq!(summary.tasks_executed, 3);

    let events = logger.snapshot();
    let completes: Vec<_> = events
        .iter()
        .filter(|e| e.event == ProbeEventKind::Complete)
        .collect();
    assert_eq!(completes.len(), 3);
}

#[test]
fn test_host_turn_yield_probe_distinct_macrotasks() {
    let count = 50; // Use bounded count for unit test
    let config = BrowserProbeConfig {
        host_turn_yield_count: count,
        ..BrowserProbeConfig::default()
    };
    let logger = Arc::new(ProbeEventLogger::new());
    let harness = BrowserExecutionHarness::new(config, Arc::clone(&logger));

    let summary = harness.run_host_turn_yield_probe();
    assert_eq!(summary.probe, ProbeKind::HostTurnYield);
    assert!(summary.passed, "All yields must occur in distinct macrotasks");
    assert_eq!(summary.macrotasks_observed, count);
}

#[test]
fn test_reentrancy_guard_mock_rejection() {
    let mock = ReentrancyGuardMock::new();
    let mut inner_attempted = false;
    let mut inner_rejected = false;

    let outer = mock.execute_with_guard(|| {
        let inner = mock.execute_with_guard(|| {
            inner_attempted = true;
        });
        if inner.is_err() {
            inner_rejected = true;
        }
    });

    assert!(outer.is_ok(), "Outer invocation must succeed");
    assert!(!inner_attempted, "Inner closure must not have run");
    assert!(inner_rejected, "Inner reentrant call must be rejected");
}

#[test]
fn test_reentrancy_guard_probe_execution() {
    let config = BrowserProbeConfig::default();
    let logger = Arc::new(ProbeEventLogger::new());
    let harness = BrowserExecutionHarness::new(config, Arc::clone(&logger));

    let summary = harness.run_reentrancy_guard_probe();
    assert_eq!(summary.probe, ProbeKind::ReentrancyGuard);
    assert!(summary.passed, "Reentrancy guard probe must pass");

    let events = logger.snapshot();
    assert!(events
        .iter()
        .any(|e| e.event == ProbeEventKind::ReentrancyPrevented));
}

#[test]
fn test_burst_limit_enforcement() {
    let burst_limit = 16;
    let total_tasks = 50;
    let config = BrowserProbeConfig {
        microtask_burst_limit: burst_limit,
        burst_task_count: total_tasks,
        ..BrowserProbeConfig::default()
    };
    let logger = Arc::new(ProbeEventLogger::new());
    let harness = BrowserExecutionHarness::new(config, Arc::clone(&logger));

    let summary = harness.run_burst_limit_probe();
    assert_eq!(summary.probe, ProbeKind::BurstLimit);
    assert!(summary.passed, "Burst limit must not be exceeded");
    assert!(summary.max_burst_observed <= burst_limit);
    assert_eq!(summary.tasks_executed, total_tasks);
    // 50 tasks with burst limit 16 -> 16, 16, 16, 2 -> 4 turns
    assert_eq!(summary.macrotasks_observed, 4);
}

#[test]
fn test_runtime_core_and_event_serialization() {
    let config = BrowserRuntimeConfig::default();
    let core = BrowserRuntimeCore::new(config);
    let harness = core.create_harness();

    let summaries = harness.run_all_probes();
    assert_eq!(summaries.len(), 5);
    for s in &summaries {
        assert!(s.passed, "Probe {:?} must pass", s.probe);
    }

    let json_lines = core.logger().to_json_lines();
    assert!(!json_lines.is_empty(), "Events must serialize to non-empty string");

    // Validate that each line parses as valid JSON with expected fields
    for line in json_lines.lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("Valid JSON line");
        assert!(v.get("probe").is_some());
        assert!(v.get("event").is_some());
        assert!(v.get("macrotask_id").is_some());
        assert!(v.get("ts_wall_ms").is_some());
    }
}
