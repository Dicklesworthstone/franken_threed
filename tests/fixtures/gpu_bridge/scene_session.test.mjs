import assert from "node:assert/strict";
// A data URL lets this Node-only test consume the browser ESM file independently
// of a repository-wide package.json module-type setting.
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const source = await readFile(new URL("./scene_session.js", import.meta.url), "utf8");
const { WebGpuSceneSession } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
async function until(predicate) {
  for (let i = 0; i < 50; i++) {
    if (predicate()) return;
    await tick();
  }
  assert.fail("Condition did not settle");
}
function packet(marker = 1) {
  const bytes = new Uint8Array(17);
  new DataView(bytes.buffer).setUint32(0, 0x50443346, true);
  bytes[4] = 1;
  bytes[16] = marker;
  return bytes;
}
function fixture(options = {}) {
  const log = [],
    devices = [],
    errors = [];
  const context = {
    configure(config) {
      log.push(["configure", config.device.id, config.format]);
    },
    unconfigure() {
      log.push(["unconfigure"]);
    },
  };
  const canvas = {
    width: 64,
    height: 32,
    getContext: (type) => (type === "webgpu" ? context : null),
  };
  const host = {
    device: null,
    buffers: new Map(),
    textures: new Map(),
    capabilityRecord: null,
    request: 0,
    negotiateGate: null,
    submitGate: null,
    queueGate: null,
    async negotiateAndCreateDevice(profile) {
      const request = ++this.request;
      log.push(["negotiate", profile]);
      if (this.negotiateGate) await this.negotiateGate.promise;
      const lost = deferred();
      const device = {
        id: devices.length + 1,
        lost: lost.promise,
        lose: lost.resolve,
        limits: { maxTextureDimension2D: 4096 },
        queue: { onSubmittedWorkDone: () => this.queueGate?.promise ?? Promise.resolve() },
        destroy() {
          log.push(["device-destroy", device.id]);
          lost.resolve({ reason: "destroyed" });
        },
      };
      devices.push(device);
      if (request !== this.request) {
        device.destroy();
        throw new Error("Device request superseded");
      }
      this.device = device;
      this.capabilityRecord = { preferredCanvasFormat: "bgra8unorm", deviceId: device.id };
      lost.promise.then(() => {
        if (this.device !== device) return;
        this.clearDeviceResources();
        this.device = null;
        this.capabilityRecord = null;
      });
      return this.capabilityRecord;
    },
    clearDeviceResources() {
      this.buffers.clear();
      this.textures.clear();
      log.push(["clear"]);
    },
    destroyDevice() {
      this.request++;
      const device = this.device;
      this.device = null;
      this.capabilityRecord = null;
      this.clearDeviceResources();
      device?.destroy();
    },
    async executePacket(bytes, _context, onSubmitted) {
      assert.ok(this.device, "No submission without a live device");
      log.push(["submit", bytes[16], this.device.id]);
      onSubmitted?.();
      if (this.submitGate) await this.submitGate.promise;
    },
  };
  const session = new WebGpuSceneSession({
    host,
    canvas,
    onError: (e) => errors.push(e),
    ...options,
  });
  return { session, host, canvas, context, log, devices, errors };
}
const submissions = (f) => f.log.filter((e) => e[0] === "submit").map((e) => e[1]);
const aborts = (promise) => assert.rejects(promise, { name: "AbortError" });

test("frame credit limits must be null or positive safe integers", async () => {
  for (const maxFramesInFlight of [false, 0, -1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => fixture({ maxFramesInFlight }), /positive safe integer/);
  }
  for (const maxFramesInFlight of [null, 1, 2, Number.MAX_SAFE_INTEGER]) {
    await fixture({ maxFramesInFlight }).session.close();
  }
});

test("default on-screen scheduling never waits for GPU completion or drops requested frames", async () => {
  const f = fixture(), gpu = deferred();
  const times = [];
  f.host.queueGate = gpu;
  await f.session.load(() => ({
    packet: packet(1),
    frame: ({ time }) => { times.push(time); return packet(times.length + 1); },
  }));
  const requested = [10, 20, 30, 40].map((time) => f.session.render(time));
  await until(() => submissions(f).length === 5);
  await Promise.all(requested);
  assert.deepEqual(times, [10, 20, 30, 40]);
  assert.deepEqual(submissions(f), [1, 2, 3, 4, 5]);
  assert.equal(f.session.snapshot().framesInFlight, 5);
  gpu.resolve();
  await until(() => f.session.snapshot().framesInFlight === 0);
  await f.session.close();
});

test("two opt-in GPU credits bound producers and a high-water acknowledgement retires earlier frames", async () => {
  const f = fixture({ maxFramesInFlight: 2 }), initial = deferred(), second = deferred(), next = deferred();
  let calls = 0;
  f.host.queueGate = initial;
  await f.session.load(() => ({ packet: packet(1), frame: () => packet(++calls + 1) }));
  f.host.queueGate = second;
  await f.session.render(1);
  f.host.queueGate = next;
  const blocked = f.session.render(2);
  await tick();
  assert.equal(calls, 1, "Successful validation receipts do not release GPU credits");
  assert.deepEqual(submissions(f), [1, 2]);
  second.resolve();
  await blocked;
  const fourth = f.session.render(3);
  await until(() => calls === 3);
  await fourth;
  assert.deepEqual(submissions(f), [1, 2, 3, 4]);
  next.resolve();
  initial.resolve();
  await f.session.close();
});

test("load captures completion at submission instead of after its validation receipt", async () => {
  const f = fixture({ maxFramesInFlight: 1 }), gpu = deferred(), validation = deferred();
  let calls = 0;
  f.host.queueGate = gpu;
  f.host.submitGate = validation;
  const loading = f.session.load(() => ({ packet: packet(1), frame: () => packet(++calls + 1) }));
  await until(() => submissions(f).length === 1);
  f.host.queueGate = null;
  validation.resolve();
  await loading;
  const rendering = f.session.render();
  await tick();
  assert.equal(calls, 0, "The initial packet still owns its GPU credit");
  gpu.resolve();
  await rendering;
  assert.equal(calls, 1);
  await f.session.close();
});

test("replacement cancels a blocked producer and drains old GPU work before its factory", async () => {
  const f = fixture({ maxFramesInFlight: 1 }), gpu = deferred();
  let frames = 0, builds = 0;
  f.host.queueGate = gpu;
  await f.session.load(() => ({ packet: packet(1), frame: () => { frames++; return packet(2); } }));
  const blocked = f.session.render(), rejected = aborts(blocked);
  await tick();
  const replacing = f.session.load(() => { builds++; return { packet: packet(3) }; });
  await rejected;
  await tick();
  assert.equal(frames, 0);
  assert.equal(builds, 0);
  gpu.resolve();
  await replacing;
  assert.equal(builds, 1);
  assert.deepEqual(submissions(f), [1, 3]);
  await f.session.close();
});

test("close releases a credit wait and owns a later completion rejection", async () => {
  const f = fixture({ maxFramesInFlight: 1 }), gpu = deferred();
  let calls = 0;
  f.host.queueGate = gpu;
  await f.session.load(() => ({ packet: packet(1), frame: () => { calls++; return packet(2); } }));
  const rendering = f.session.render(), rejected = aborts(rendering);
  await tick();
  await f.session.close();
  await rejected;
  gpu.reject(new Error("completion rejected after shutdown"));
  await tick();
  assert.equal(calls, 0);
  assert.equal(f.session.state, "closed");
  assert.deepEqual(f.errors, []);
});

test("device loss cancels credit waits and old completions cannot fail a successor", async () => {
  const f = fixture({ maxFramesInFlight: 1 }), oldGpu = deferred();
  const builds = [];
  let frames = 0;
  f.host.queueGate = oldGpu;
  await f.session.load(({ capabilities }) => {
    builds.push(capabilities.deviceId);
    return { packet: packet(capabilities.deviceId), frame: () => { frames++; return packet(3); } };
  });
  const rendering = f.session.render(), rejected = aborts(rendering);
  await tick();
  f.host.queueGate = null;
  f.devices[0].lose({ message: "loss while waiting for GPU credit" });
  await rejected;
  await until(() => f.session.state === "ready" && f.devices.length === 2);
  oldGpu.reject(new Error("old device completion"));
  await tick();
  assert.equal(frames, 0);
  assert.equal(f.session.state, "ready");
  assert.deepEqual(builds, [1, 2]);
  assert.equal(f.errors.length, 1);
  await f.session.render();
  assert.equal(frames, 1);
  await f.session.close();
});

test("a rejected GPU completion fails the scene without replaying source callbacks", async () => {
  const f = fixture({ maxFramesInFlight: 1 }), gpu = deferred();
  let frames = 0;
  f.host.queueGate = gpu;
  await f.session.load(() => ({ packet: packet(1), frame: () => { frames++; return packet(2); } }));
  const blocked = f.session.render(), rejected = aborts(blocked);
  await tick();
  gpu.reject(new Error("GPU completion failed"));
  await rejected;
  assert.equal(f.session.state, "failed");
  assert.equal(frames, 0);
  assert.match(f.session.lastError, /GPU completion failed/);
  assert.equal(f.errors.length, 1);
  assert.deepEqual(submissions(f), [1]);
  await f.session.close();
});

test("a failed queue acknowledgement terminates every outstanding credit on that device", async () => {
  const f = fixture({ maxFramesInFlight: 2 }), first = deferred(), second = deferred();
  let frames = 0;
  f.host.queueGate = first;
  await f.session.load(() => ({ packet: packet(1), frame: () => packet(++frames + 1) }));
  f.host.queueGate = second;
  await f.session.render(1);
  const blocked = f.session.render(2), rejected = aborts(blocked);
  await tick();
  assert.equal(f.session.snapshot().framesInFlight, 2);
  first.reject(new Error("queue completion failed"));
  await rejected;
  assert.equal(f.session.state, "failed");
  assert.equal(f.session.snapshot().framesInFlight, 0);
  assert.equal(frames, 1);
  second.reject(new Error("later failure on the terminal queue"));
  await tick();
  assert.equal(f.errors.length, 1);
  assert.match(f.session.lastError, /queue completion failed/);
  assert.deepEqual(submissions(f), [1, 2]);
  await f.session.close();
});

test("failed validation stops queued callbacks and replacement still waits for submitted GPU work", async () => {
  const f = fixture(), gpu = deferred(), validation = deferred();
  let frames = 0, builds = 0;
  await f.session.load(() => {
    builds++;
    return { packet: packet(1), frame: () => packet(++frames + 1) };
  });
  f.host.queueGate = gpu;
  f.host.submitGate = validation;
  const first = f.session.render(), second = f.session.render();
  const failures = Promise.all([assert.rejects(first, /validation failed/), aborts(second)]);
  await until(() => submissions(f).length === 2);
  validation.reject(new Error("validation failed"));
  await failures;
  assert.equal(frames, 1);
  assert.equal(f.session.state, "failed");
  f.host.submitGate = null;
  const reloading = f.session.reload();
  await tick();
  assert.equal(builds, 1, "Failed validation does not cancel issued GPU work");
  gpu.resolve();
  await reloading;
  assert.equal(builds, 2);
  assert.equal(frames, 1);
  assert.deepEqual(submissions(f), [1, 2, 1]);
  await f.session.close();
});

test("initializes a scene and submits explicit frames with negotiated capabilities", async () => {
  const f = fixture();
  let loadContext, frameContext;
  await f.session.load((ctx) => {
    loadContext = ctx;
    return {
      packet: packet(1),
      frame(ctx) {
        frameContext = ctx;
        return packet(2);
      },
    };
  });
  assert.equal(f.session.state, "ready");
  assert.equal(loadContext.capabilities.deviceId, 1);
  assert.equal(loadContext.width, 64);
  assert.equal(loadContext.height, 32);
  await f.session.render(125);
  assert.equal(frameContext.time, 125);
  assert.deepEqual(submissions(f), [1, 2]);
  assert.equal(f.session.snapshot().submittedFrames, 2);
  assert.ok(
    f.log.findIndex((e) => e[0] === "configure") < f.log.findIndex((e) => e[0] === "submit"),
  );
  await f.session.close();
});

test("resize reuses the device and destroys scene resources only after GPU completion", async () => {
  const f = fixture();
  let disposed = 0;
  const sizes = [];
  await f.session.load((ctx) => {
    sizes.push([ctx.width, ctx.height]);
    return {
      packet: packet(),
      dispose() {
        disposed++;
      },
    };
  });
  f.host.buffers.set(1, {
    destroy() {
      f.log.push(["buffer-destroy"]);
    },
  });
  f.host.textures.set(2, {
    texture: {
      destroy() {
        f.log.push(["texture-destroy"]);
      },
    },
  });
  f.host.queueGate = deferred();
  const resizing = f.session.resize(320, 200);
  await tick();
  assert.equal(
    f.log.some((e) => e[0] === "buffer-destroy"),
    false,
  );
  f.host.queueGate.resolve();
  await resizing;
  assert.equal(f.devices.length, 1);
  assert.equal(f.canvas.width, 320);
  assert.equal(f.canvas.height, 200);
  assert.deepEqual(sizes, [
    [64, 32],
    [320, 200],
  ]);
  assert.equal(disposed, 1);
  assert.equal(f.log.filter((e) => e[0] === "buffer-destroy").length, 1);
  assert.equal(f.log.filter((e) => e[0] === "texture-destroy").length, 1);
  await f.session.resize(320, 200);
  assert.equal(sizes.length, 2, "Unchanged dimensions do not rebuild the scene");
  await f.session.close();
  assert.equal(disposed, 2);
});

test("a late uncooperative factory cannot overwrite a successor and is disposed once", async () => {
  const f = fixture(),
    late = deferred();
  let entered = false,
    disposed = 0;
  const loading = f.session.load(() => {
    entered = true;
    return late.promise;
  });
  const rejected = aborts(loading);
  await until(() => entered);
  await f.session.load(() => ({ packet: packet(9) }));
  await rejected;
  late.resolve({
    packet: packet(3),
    dispose() {
      disposed++;
    },
  });
  await until(() => disposed === 1);
  assert.deepEqual(submissions(f), [9]);
  assert.equal(f.session.state, "ready");
  await f.session.close();
  assert.equal(disposed, 1);
});

test("resize supersedes queued sizes and forwards cancellation to the old factory", async () => {
  const f = fixture(),
    block = deferred();
  let oldSignal;
  await f.session.load(() => ({ packet: packet(1) }));
  const loading = f.session.load((ctx) => {
    oldSignal = ctx.signal;
    return block.promise;
  });
  const rejected = aborts(loading);
  await until(() => oldSignal !== undefined);
  const next = f.session.resize(200, 100);
  const nextRejected = aborts(next);
  const final = f.session.load((ctx) => ({ packet: packet(ctx.width === 400 ? 7 : 8) }), {
    width: 400,
    height: 200,
  });
  await Promise.all([rejected, nextRejected, final]);
  assert.ok(oldSignal.aborted);
  assert.deepEqual(submissions(f), [1, 7]);
  assert.equal(f.canvas.width, 400);
  block.resolve({ packet: packet(4) });
  await f.session.close();
});

test("close cancels a pending device request and destroys its late result", async () => {
  const f = fixture();
  f.host.negotiateGate = deferred();
  const loading = f.session.load(() => ({ packet: packet() }));
  const rejected = aborts(loading);
  await until(() => f.log.some((e) => e[0] === "negotiate"));
  const close = f.session.close();
  assert.strictEqual(f.session.close(), close);
  await Promise.all([close, rejected]);
  f.host.negotiateGate.resolve();
  await until(() => f.devices.length === 1);
  assert.equal(f.host.device, null);
  assert.ok(f.log.some((e) => e[0] === "device-destroy"));
  assert.deepEqual(submissions(f), []);
  assert.throws(() => f.session.reload(), /closed/);
});

test("device loss rebuilds scene packets with successor capabilities, with a finite retry budget", async () => {
  const f = fixture();
  const capabilities = [];
  let disposed = 0;
  await f.session.load((ctx) => {
    capabilities.push(ctx.capabilities.deviceId);
    return {
      packet: packet(ctx.capabilities.deviceId),
      dispose() {
        disposed++;
      },
    };
  });
  f.devices[0].lose({ reason: "unknown", message: "simulated loss" });
  await until(() => f.session.state === "ready" && f.devices.length === 2);
  assert.deepEqual(capabilities, [1, 2]);
  assert.deepEqual(submissions(f), [1, 2]);
  assert.equal(disposed, 1);
  f.devices[1].lose({ reason: "unknown", message: "again" });
  await until(() => f.session.state === "failed");
  await tick();
  assert.equal(f.devices.length, 2, "Loss recovery must not spin indefinitely");
  assert.match(f.session.lastError, /again/);
  await f.session.reload();
  assert.equal(f.session.state, "ready");
  assert.equal(f.devices.length, 3);
  assert.equal(f.session.recoveryAttempts, 0);
  await f.session.close();
});

test("a frame producer completing after reload never submits to the new scene", async () => {
  const f = fixture(),
    frame = deferred();
  let started = false;
  await f.session.load(() => ({
    packet: packet(1),
    frame() {
      started = true;
      return frame.promise;
    },
  }));
  const rendering = f.session.render();
  const rejected = aborts(rendering);
  await until(() => started);
  await f.session.load(() => ({ packet: packet(2) }));
  frame.resolve(packet(3));
  await rejected;
  await tick();
  assert.deepEqual(submissions(f), [1, 2]);
  await f.session.close();
});

test("frame submission is serialized and no-op frames do not inflate GPU counts", async () => {
  const f = fixture();
  let frameCount = 0;
  await f.session.load(() => ({
    packet: packet(1),
    frame() {
      return ++frameCount === 3 ? null : packet(2);
    },
  }));
  f.host.submitGate = deferred();
  const first = f.session.render(),
    second = f.session.render();
  await until(() => submissions(f).length === 2);
  await tick();
  assert.equal(frameCount, 1);
  f.host.submitGate.resolve();
  await Promise.all([first, second]);
  await f.session.render();
  assert.equal(f.session.submittedFrames, 3);
  await f.session.close();
});

test("a failed frame stops already queued frames until an explicit reload", async () => {
  const f = fixture();
  let calls = 0;
  await f.session.load(() => ({
    packet: packet(1),
    frame() {
      calls++;
      throw new Error("invalid scene state");
    },
  }));
  const first = f.session.render(),
    second = f.session.render();
  await Promise.all([assert.rejects(first, /invalid scene state/), aborts(second)]);
  assert.equal(calls, 1);
  assert.equal(f.session.state, "failed");
  assert.deepEqual(submissions(f), [1]);
  await f.session.close();
});

test("invalid dimensions leave an existing scene untouched", async () => {
  const f = fixture();
  await f.session.load(() => ({ packet: packet() }));
  const before = f.session.snapshot();
  for (const width of [0, -1, NaN, Infinity, 1.5, 4097]) {
    assert.throws(() => f.session.resize(width, 100), RangeError);
  }
  assert.deepEqual(f.session.snapshot(), before);
  assert.throws(() => f.session.render(NaN), TypeError);
  await f.session.close();
});

test("malformed, detached and shared packet objects never reach the host", async () => {
  const f = fixture();
  let disposed = 0;
  const detached = new Uint8Array(20);
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  for (const value of [
    null,
    new Uint8Array(0),
    new Uint8Array(15),
    {},
    detached,
    new Uint8Array(new SharedArrayBuffer(20)),
  ]) {
    await assert.rejects(
      f.session.load(() => ({
        packet: value,
        dispose() {
          disposed++;
        },
      })),
      TypeError,
    );
  }
  assert.equal(disposed, 6);
  assert.deepEqual(submissions(f), []);
  await f.session.close();
});

test("close during a GPU completion wait does not lose CPU scene disposal", async () => {
  const f = fixture();
  let disposed = 0;
  await f.session.load(() => ({
    packet: packet(),
    dispose() {
      disposed++;
    },
  }));
  f.host.queueGate = deferred();
  const loading = f.session.reload();
  const rejected = aborts(loading);
  await tick();
  await f.session.close();
  await rejected;
  assert.equal(disposed, 1);
  f.host.queueGate.resolve();
});

test("observer exceptions do not break ownership, including reentrant close", async () => {
  const f = fixture({
    onState() {
      throw new Error("observer failure");
    },
    onError() {
      throw new Error("error observer failure");
    },
  });
  await f.session.load(() => ({ packet: packet() }));
  assert.equal(f.session.state, "ready");
  let reentrant;
  f.session.onState = () => {
    reentrant = f.session.close();
  };
  const closed = f.session.close();
  assert.strictEqual(reentrant, closed);
  await closed;
  assert.equal(f.log.filter((e) => e[0] === "unconfigure").length, 1);
});

test("device loss recovery can be disabled and intentional shutdown never reacquires", async () => {
  const f = fixture({ maxRecoveryAttempts: 0 });
  await f.session.load(() => ({ packet: packet() }));
  f.devices[0].lose({ reason: "unknown" });
  await until(() => f.session.state === "failed");
  await f.session.close();
  await tick();
  assert.equal(f.devices.length, 1);
  const g = fixture();
  await g.session.load(() => ({ packet: packet() }));
  await g.session.close();
  await tick();
  assert.equal(g.devices.length, 1);
});

test("error observers may close the session without a failure resurrecting it", async () => {
  const f = fixture();
  let close;
  await f.session.load(() => ({
    packet: packet(),
    frame() {
      throw new Error("frame failure");
    },
  }));
  f.session.onError = () => {
    close = f.session.close();
  };
  await assert.rejects(f.session.render(), /frame failure/);
  await close;
  assert.equal(f.session.state, "closed");
});

test("a device-loss observer can shut down instead of auto-recovering", async () => {
  const f = fixture();
  await f.session.load(() => ({ packet: packet() }));
  f.session.onError = () => {
    f.session.close();
  };
  f.devices[0].lose({ message: "loss handled by application" });
  await tick();
  await tick();
  assert.equal(f.session.state, "closed");
  assert.equal(f.devices.length, 1);
});

test("error-observer reload is newer than the failing frame", async () => {
  const f = fixture();
  let replacement;
  await f.session.load(() => ({
    packet: packet(1),
    frame() {
      throw new Error("retry externally");
    },
  }));
  f.session.onError = () => {
    replacement = f.session.load(() => ({ packet: packet(2) }));
  };
  await assert.rejects(f.session.render(), /retry externally/);
  await replacement;
  assert.equal(f.session.state, "ready");
  assert.deepEqual(submissions(f), [1, 2]);
  await f.session.close();
});

test("frame methods retain their scene receiver", async () => {
  const f = fixture();
  await f.session.load(() => ({
    packet: packet(1),
    value: 11,
    frame() {
      return packet(this.value);
    },
  }));
  await f.session.render();
  assert.deepEqual(submissions(f), [1, 11]);
  await f.session.close();
});

test("close waits for already-started disposal after cancelling a reload", async () => {
  const f = fixture(),
    disposal = deferred();
  let entered = false,
    closed = false;
  await f.session.load(() => ({
    packet: packet(),
    dispose() {
      entered = true;
      return disposal.promise;
    },
  }));
  const loading = f.session.reload(),
    rejected = aborts(loading);
  await until(() => entered);
  const closing = f.session.close().then(() => {
    closed = true;
  });
  await tick();
  assert.equal(closed, false, "close must join existing CPU ownership cleanup");
  disposal.resolve();
  await Promise.all([closing, rejected]);
  assert.equal(closed, true);
});

test("an abort listener can close during reload without the new operation reopening the session", async () => {
  const f = fixture();
  await f.session.load(({ signal }) => {
    signal.addEventListener(
      "abort",
      () => {
        f.session.close();
      },
      { once: true },
    );
    return { packet: packet(1) };
  });
  await aborts(f.session.reload());
  assert.equal(f.session.state, "closed");
  assert.deepEqual(submissions(f), [1]);
});

test("terminal device loss cannot overwrite shutdown requested by an abort listener", async () => {
  const f = fixture({ maxRecoveryAttempts: 0 });
  await f.session.load(({ signal }) => {
    signal.addEventListener(
      "abort",
      () => {
        f.session.close();
      },
      { once: true },
    );
    return { packet: packet() };
  });
  f.devices[0].lose({ message: "terminal loss" });
  await tick();
  await tick();
  assert.equal(f.session.state, "closed");
  assert.equal(f.devices.length, 1);
});

test("closing in the microtask before a queued frame producer prevents invoking it", async () => {
  const f = fixture();
  let calls = 0;
  await f.session.load(() => ({
    packet: packet(),
    frame() {
      calls++;
      return packet(2);
    },
  }));
  const rendering = f.session.render(),
    rejected = aborts(rendering);
  queueMicrotask(() => {
    f.session.close();
  });
  await rejected;
  assert.equal(calls, 0);
  assert.deepEqual(submissions(f), [1]);
});
