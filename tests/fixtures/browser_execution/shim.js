// Browser host callbacks for the actual Asupersync Wasm probe. No executor.
(() => {
  let wasm;
  let hostTurn = 0;
  let finished = false;
  const events = [];
  const pendingStream = [];
  let streamTimer = null;
  function flushStream() {
    if (streamTimer !== null) { clearTimeout(streamTimer); streamTimer = null; }
    if (pendingStream.length === 0) return;
    const batch = pendingStream.splice(0, pendingStream.length);
    fetch('/event', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(batch) }).catch(() => {});
  }
  const channel = new MessageChannel();
  const pending = new Map();
  let nextCallback = 0;
  const burstChannel = new MessageChannel();
  let burstActive = false;
  let burstDone = false;
  let burstCallback = null;
  let burstResultsEmitted = false;
  let baselinePending = false;
  let lastPolls = 0;
  let lastPumpTurns = 0;
  let burstStartPumpTurns = 0;
  let maxPollsPerTurn = 0;
  let turnsObserved = 0;
  let maxPollsPerPumpTurn = 0;
  let maxPumpTurnsPerObservation = 0;
  let observationIndex = 0;
  const anomalies = [];
  let totalAnomalies = 0;

  function observeTurn(source = 9) {
    if (!wasm || !burstActive) return;
    if (baselinePending) {
      lastPolls = wasm.burst_polls();
      lastPumpTurns = wasm.pump_turns ? wasm.pump_turns() : 0;
      burstStartPumpTurns = lastPumpTurns;
      baselinePending = false;
      return;
    }
    observationIndex++;
    const currentPolls = wasm.burst_polls();
    const currentPumpTurns = wasm.pump_turns ? wasm.pump_turns() : 0;
    const pollsDelta = currentPolls - lastPolls;
    const turnsDelta = currentPumpTurns - lastPumpTurns;
    lastPolls = currentPolls;
    lastPumpTurns = currentPumpTurns;
    if (pollsDelta > 0) {
      if (pollsDelta > maxPollsPerTurn) maxPollsPerTurn = pollsDelta;
      turnsObserved++;
    }
    const perPumpTurn = Math.ceil(pollsDelta / Math.max(1, turnsDelta));
    if (turnsDelta >= 1) {
      if (perPumpTurn > maxPollsPerPumpTurn) maxPollsPerPumpTurn = perPumpTurn;
    }
    if (perPumpTurn > 4) {
      totalAnomalies++;
      if (anomalies.length < 5) {
        anomalies.push({
          index: observationIndex,
          pollsDelta,
          turnsDelta,
          currentPolls,
          currentPumpTurns,
          source,
        });
      }
    }
    if (turnsDelta > maxPumpTurnsPerObservation) {
      maxPumpTurnsPerObservation = turnsDelta;
    }
  }

  function emitBurstResults() {
    if (burstResultsEmitted) return;
    burstResultsEmitted = true;
    const totalPumpTurns = wasm && wasm.pump_turns ? wasm.pump_turns() - burstStartPumpTurns : 0;
    globalThis.f3dHost.event('burst-all-turns', 'pump-turns', totalPumpTurns);
    globalThis.f3dHost.event('burst-all-turns', 'max-per-pump-turn', maxPollsPerPumpTurn);
    globalThis.f3dHost.event('burst-all-turns', 'max-pump-turns-per-observation', maxPumpTurnsPerObservation);
    globalThis.f3dHost.event('burst-all-turns', 'max', maxPollsPerTurn);
    globalThis.f3dHost.event('burst-all-turns', 'turns', turnsObserved);
    for (const a of anomalies) {
      globalThis.f3dHost.event('burst-all-turns', 'anomaly-polls', a.pollsDelta);
      globalThis.f3dHost.event('burst-all-turns', 'anomaly-turns', a.turnsDelta);
      globalThis.f3dHost.event('burst-all-turns', 'anomaly-index', a.index);
      globalThis.f3dHost.event('burst-all-turns', 'anomaly-source', a.source);
    }
    globalThis.f3dHost.event('burst-all-turns', 'anomalies', totalAnomalies);
  }

  burstChannel.port1.onmessage = () => {
    observeTurn(9);
    if (burstActive && !burstDone) {
      burstChannel.port2.postMessage(null);
    } else if (burstDone && burstCallback) {
      const cb = burstCallback;
      burstCallback = null;
      emitBurstResults();
      cb(maxPollsPerPumpTurn);
    }
  };

  channel.port1.onmessage = ({ data }) => {
    observeTurn(1);
    const callback = pending.get(data);
    pending.delete(data);
    if (callback) callback(++hostTurn);
  };
  // A Rust panic inside a pump microtask surfaces as an uncaught exception, not as a
  // rejected start_probes() call; report it instead of letting the run look like a hang.
  addEventListener('error', e => globalThis.f3dHost.finish(false, String(e.error?.stack || e.message || e)));
  addEventListener('unhandledrejection', e => globalThis.f3dHost.finish(false, String(e.reason?.stack || e.reason)));
  globalThis.f3dHost = {
    attach(exports) {
      wasm = exports;
      // Host liveness diagnostic: if this never streams, the main thread is frozen
      // (synchronous hang in the Wasm); if it streams with 0 polls, the task was never polled.
      setTimeout(() => this.event('host-diag', 'main-thread-alive-3s', wasm.burst_polls()), 3000);
      setTimeout(() => this.event('host-diag', 'reenter-probe-says-pump-running', wasm.reenter_probe() ? 1 : 0), 3500);
    },
    wait(source, callback) {
      if (source === 0) {
        setTimeout(() => {
          observeTurn(0);
          callback(++hostTurn);
        }, 5);
      } else if (source === 1) {
        const id = ++nextCallback;
        pending.set(id, callback);
        channel.port2.postMessage(id);
      } else if (source === 2) {
        queueMicrotask(() => {
          const before = wasm.burst_polls();
          queueMicrotask(() => callback(wasm.burst_polls() - before));
        });
      } else if (source === 3) {
        setTimeout(() => {
          observeTurn(3);
          callback(++hostTurn);
        }, 5);
      } else if (source === 4) {
        if (burstDone) {
          emitBurstResults();
          queueMicrotask(() => callback(maxPollsPerPumpTurn));
        } else {
          burstCallback = callback;
        }
      } else {
        throw new Error(`Unknown host callback source ${source}`);
      }
    },
    reenter() { return wasm.reenter_probe(); },
    turns() { return turnsObserved; },
    event(probe, step, value) {
      if (probe === 'burst-first-turn-and-completion' && step === 'spawn') {
        burstActive = true;
        burstDone = false;
        burstResultsEmitted = false;
        baselinePending = true;
        maxPollsPerTurn = 0;
        turnsObserved = 0;
        maxPollsPerPumpTurn = 0;
        maxPumpTurnsPerObservation = 0;
        observationIndex = 0;
        anomalies.length = 0;
        totalAnomalies = 0;
        burstChannel.port2.postMessage(null);
      }
      if (probe === 'burst-first-turn-and-completion' && step === 'complete') {
        burstDone = true;
        observeTurn(9);
        burstActive = false;
        if (burstCallback) {
          const cb = burstCallback;
          burstCallback = null;
          emitBurstResults();
          cb(maxPollsPerPumpTurn);
        }
      }
      const event = { probe, step, value, ts_wall: Date.now(), host_time_ms: performance.now(), host_turn: hostTurn };
      events.push(event);
      // Stream each observation immediately so a hang still shows how far the Rust program got.
      // Batched streaming: the browser keepalive budget (~64 KB in flight) dropped
      // ~40% of single-event posts during a 90 s hang, so buffer and flush in chunks.
      pendingStream.push(event);
      if (pendingStream.length >= 25) flushStream();
      else if (streamTimer === null) streamTimer = setTimeout(flushStream, 100);
    },
    finish(passed, detail) {
      if (finished) return;
      finished = true;
      flushStream();
      const result = {
        passed, detail, events, owner: 'asupersync-rust-wasm',
        browser: { userAgent: navigator.userAgent, platform: navigator.platform },
      };
      globalThis.__F3D_PROBE_RESULTS__ = result;
      document.querySelector('#result').textContent = JSON.stringify(result, null, 2);
      fetch('/result', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(result) }).catch(error => console.error('Result delivery failed', error));
      channel.port1.close();
      channel.port2.close();
      burstChannel.port1.close();
      burstChannel.port2.close();
    },
  };
})();
