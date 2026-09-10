// Browser host callbacks for the actual Asupersync Wasm probe. No executor.
(() => {
  let wasm;
  let hostTurn = 0;
  let finished = false;
  const events = [];
  const channel = new MessageChannel();
  const pending = new Map();
  let nextCallback = 0;
  channel.port1.onmessage = ({ data }) => {
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
        setTimeout(() => callback(++hostTurn), 5);
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
        setTimeout(() => callback(++hostTurn), 5);
      } else {
        throw new Error(`Unknown host callback source ${source}`);
      }
    },
    reenter() { return wasm.reenter_probe(); },
    event(probe, step, value) {
      const event = { probe, step, value, ts_wall: Date.now(), host_time_ms: performance.now(), host_turn: hostTurn };
      events.push(event);
      // Stream each observation immediately so a hang still shows how far the Rust program got.
      fetch('/event', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(event), keepalive: true }).catch(() => {});
    },
    finish(passed, detail) {
      if (finished) return;
      finished = true;
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
    },
  };
})();
