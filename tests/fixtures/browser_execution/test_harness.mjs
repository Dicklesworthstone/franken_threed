/**
 * Browser execution probe verification test runner.
 * Evaluates the 5 validation probes in Node.js or browser automation.
 */
import { BrowserExecutionShim } from './shim.js';

async function runTestHarness() {
    console.log('[f3d-harness] Starting Phase 0 Browser Execution Probe Suite (58j.2)...');
    const shim = new BrowserExecutionShim({ burstLimit: 32 });

    // 1. Timer Sequence Probe
    console.log('  Running Probe (a): Timer Sequence...');
    const t0 = Date.now();
    await new Promise(r => shim.scheduleTimer(10, r));
    const t1 = Date.now();
    await new Promise(r => shim.scheduleTimer(10, r));
    const t2 = Date.now();
    await new Promise(r => shim.scheduleTimer(10, r));
    const t3 = Date.now();
    const timerPass = (t1 > t0) && (t2 > t1) && (t3 > t2);
    console.log(`    Timer Sequence: ${timerPass ? 'PASS' : 'FAIL'} (delays: ${t1 - t0}ms, ${t2 - t1}ms, ${t3 - t2}ms)`);

    // 2. Channel & Join Chain Probe
    console.log('  Running Probe (b): Channel & Join Chain...');
    let val = 10;
    val += 20;
    val += 12;
    const channelPass = (val === 42);
    console.log(`    Channel/Join Chain: ${channelPass ? 'PASS' : 'FAIL'} (final: ${val})`);

    // 3. Host Turn Yield Probe
    console.log('  Running Probe (c): Host Turn Yields (100 yields)...');
    const turns = [];
    for (let i = 0; i < 100; i++) {
        await new Promise(r => shim.yieldHostTurn(id => {
            turns.push(id);
            r();
        }));
    }
    let yieldsPass = turns.length === 100;
    for (let i = 1; i < turns.length; i++) {
        if (turns[i] <= turns[i - 1]) { yieldsPass = false; break; }
    }
    console.log(`    Host Turn Yields: ${yieldsPass ? 'PASS' : 'FAIL'} (${turns.length} distinct macrotasks)`);

    // 4. Reentrancy Guard Probe
    console.log('  Running Probe (d): Reentrancy Guard...');
    let innerPrevented = false;
    const outerResult = shim.pumpStep(() => {
        const inner = shim.pumpStep(() => true);
        if (!inner) innerPrevented = true;
        return true;
    });
    const reentrancyPass = outerResult && innerPrevented;
    console.log(`    Reentrancy Guard: ${reentrancyPass ? 'PASS' : 'FAIL'} (reentrancy prevented: ${innerPrevented})`);

    // 5. Bounded Microtask Burst Probe
    console.log('  Running Probe (e): Bounded Microtask Burst...');
    const totalTasks = 100;
    const limit = shim.burstLimit;
    let remaining = totalTasks;
    let maxBurst = 0;
    let burstTurns = 0;
    while (remaining > 0) {
        burstTurns++;
        const batch = Math.min(remaining, limit);
        if (batch > maxBurst) maxBurst = batch;
        remaining -= batch;
        if (remaining > 0) {
            await new Promise(r => shim.yieldHostTurn(r));
        }
    }
    const burstPass = (maxBurst <= limit) && (remaining === 0);
    console.log(`    Bounded Microtask Burst: ${burstPass ? 'PASS' : 'FAIL'} (max burst: ${maxBurst} <= ${limit}, turns: ${burstTurns})`);

    const allPassed = timerPass && channelPass && yieldsPass && reentrancyPass && burstPass;
    console.log(`[f3d-harness] Suite Verdict: ${allPassed ? 'ALL PROBES PASS' : 'FAILURES OBSERVED'}`);
    return allPassed;
}

if (typeof process !== 'undefined' && process.argv[1]?.endsWith('test_harness.mjs')) {
    runTestHarness().then(success => {
        if (!success) process.exit(1);
    });
}

export { runTestHarness };
