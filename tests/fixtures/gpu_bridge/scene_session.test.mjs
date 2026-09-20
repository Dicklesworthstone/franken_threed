import { test } from 'node:test';
import assert from 'node:assert/strict';
// A data URL lets this Node-only test consume the browser ESM file independently
// of a repository-wide package.json module-type setting.
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('./scene_session.js', import.meta.url), 'utf8');
const { WebGpuSceneSession } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const tick = () => new Promise(resolve => setImmediate(resolve));
async function until(predicate) {
  for (let i = 0; i < 50; i++) { if (predicate()) return; await tick(); }
  assert.fail('Condition did not settle');
}
function packet(marker = 1) {
  const bytes = new Uint8Array(17);
  new DataView(bytes.buffer).setUint32(0, 0x50443346, true);
  bytes[4] = 1;
  bytes[16] = marker;
  return bytes;
}
function fixture(options = {}) {
  const log = [], devices = [], errors = [];
  const context = { configure(config) { log.push(['configure', config.device.id, config.format]); },
    unconfigure() { log.push(['unconfigure']); } };
  const canvas = { width: 64, height: 32, getContext: type => type === 'webgpu' ? context : null };
  const host = {
    device: null, buffers: new Map(), textures: new Map(), capabilityRecord: null,
    request: 0, negotiateGate: null, submitGate: null, queueGate: null,
    async negotiateAndCreateDevice(profile) {
      const request = ++this.request;
      log.push(['negotiate', profile]);
      if (this.negotiateGate) await this.negotiateGate.promise;
      const lost = deferred();
      const device = { id: devices.length + 1, lost: lost.promise, lose: lost.resolve,
        limits: { maxTextureDimension2D: 4096 },
        queue: { onSubmittedWorkDone: () => this.queueGate?.promise ?? Promise.resolve() },
        destroy() { log.push(['device-destroy', device.id]); lost.resolve({ reason: 'destroyed' }); } };
      devices.push(device);
      if (request !== this.request) { device.destroy(); throw new Error('Device request superseded'); }
      this.device = device;
      this.capabilityRecord = { preferredCanvasFormat: 'bgra8unorm', deviceId: device.id };
      lost.promise.then(() => {
        if (this.device !== device) return;
        this.clearDeviceResources(); this.device = null; this.capabilityRecord = null;
      });
      return this.capabilityRecord;
    },
    clearDeviceResources() { this.buffers.clear(); this.textures.clear(); log.push(['clear']); },
    destroyDevice() { this.request++; const device = this.device; this.device = null;
      this.capabilityRecord = null; this.clearDeviceResources(); device?.destroy(); },
    async executePacket(bytes) {
      assert.ok(this.device, 'No submission without a live device');
      log.push(['submit', bytes[16], this.device.id]);
      if (this.submitGate) await this.submitGate.promise;
    },
  };
  const session = new WebGpuSceneSession({ host, canvas, onError: e => errors.push(e), ...options });
  return { session, host, canvas, context, log, devices, errors };
}
const submissions = f => f.log.filter(e => e[0] === 'submit').map(e => e[1]);
const aborts = promise => assert.rejects(promise, { name: 'AbortError' });

test('initializes a scene and submits explicit frames with negotiated capabilities', async () => {
  const f = fixture(); let loadContext, frameContext;
  await f.session.load(ctx => { loadContext = ctx; return { packet: packet(1),
    frame(ctx) { frameContext = ctx; return packet(2); } }; });
  assert.equal(f.session.state, 'ready');
  assert.equal(loadContext.capabilities.deviceId, 1);
  assert.equal(loadContext.width, 64);
  assert.equal(loadContext.height, 32);
  await f.session.render(125);
  assert.equal(frameContext.time, 125);
  assert.deepEqual(submissions(f), [1, 2]);
  assert.equal(f.session.snapshot().submittedFrames, 2);
  assert.ok(f.log.findIndex(e => e[0] === 'configure') < f.log.findIndex(e => e[0] === 'submit'));
  await f.session.close();
});

test('resize reuses the device and destroys scene resources only after GPU completion', async () => {
  const f = fixture(); let disposed = 0; const sizes = [];
  await f.session.load(ctx => { sizes.push([ctx.width, ctx.height]);
    return { packet: packet(), dispose() { disposed++; } }; });
  f.host.buffers.set(1, { destroy() { f.log.push(['buffer-destroy']); } });
  f.host.textures.set(2, { texture: { destroy() { f.log.push(['texture-destroy']); } } });
  f.host.queueGate = deferred();
  const resizing = f.session.resize(320, 200);
  await tick();
  assert.equal(f.log.some(e => e[0] === 'buffer-destroy'), false);
  f.host.queueGate.resolve();
  await resizing;
  assert.equal(f.devices.length, 1);
  assert.equal(f.canvas.width, 320);
  assert.equal(f.canvas.height, 200);
  assert.deepEqual(sizes, [[64, 32], [320, 200]]);
  assert.equal(disposed, 1);
  assert.equal(f.log.filter(e => e[0] === 'buffer-destroy').length, 1);
  assert.equal(f.log.filter(e => e[0] === 'texture-destroy').length, 1);
  await f.session.resize(320, 200);
  assert.equal(sizes.length, 2, 'Unchanged dimensions do not rebuild the scene');
  await f.session.close();
  assert.equal(disposed, 2);
});

test('a late uncooperative factory cannot overwrite a successor and is disposed once', async () => {
  const f = fixture(), late = deferred(); let entered = false, disposed = 0;
  const loading = f.session.load(() => { entered = true; return late.promise; });
  const rejected = aborts(loading);
  await until(() => entered);
  await f.session.load(() => ({ packet: packet(9) }));
  await rejected;
  late.resolve({ packet: packet(3), dispose() { disposed++; } });
  await until(() => disposed === 1);
  assert.deepEqual(submissions(f), [9]);
  assert.equal(f.session.state, 'ready');
  await f.session.close();
  assert.equal(disposed, 1);
});

test('resize supersedes queued sizes and forwards cancellation to the old factory', async () => {
  const f = fixture(), block = deferred(); let oldSignal;
  await f.session.load(() => ({ packet: packet(1) }));
  const loading = f.session.load(ctx => { oldSignal = ctx.signal; return block.promise; });
  const rejected = aborts(loading);
  await until(() => oldSignal !== undefined);
  const next = f.session.resize(200, 100);
  const nextRejected = aborts(next);
  const final = f.session.load(ctx => ({ packet: packet(ctx.width === 400 ? 7 : 8) }), { width: 400, height: 200 });
  await Promise.all([rejected, nextRejected, final]);
  assert.ok(oldSignal.aborted);
  assert.deepEqual(submissions(f), [1, 7]);
  assert.equal(f.canvas.width, 400);
  block.resolve({ packet: packet(4) });
  await f.session.close();
});

test('close cancels a pending device request and destroys its late result', async () => {
  const f = fixture(); f.host.negotiateGate = deferred();
  const loading = f.session.load(() => ({ packet: packet() }));
  const rejected = aborts(loading);
  await until(() => f.log.some(e => e[0] === 'negotiate'));
  const close = f.session.close();
  assert.strictEqual(f.session.close(), close);
  await Promise.all([close, rejected]);
  f.host.negotiateGate.resolve();
  await until(() => f.devices.length === 1);
  assert.equal(f.host.device, null);
  assert.ok(f.log.some(e => e[0] === 'device-destroy'));
  assert.deepEqual(submissions(f), []);
  assert.throws(() => f.session.reload(), /closed/);
});

test('device loss rebuilds scene packets with successor capabilities, with a finite retry budget', async () => {
  const f = fixture(); const capabilities = []; let disposed = 0;
  await f.session.load(ctx => { capabilities.push(ctx.capabilities.deviceId); return {
    packet: packet(ctx.capabilities.deviceId), dispose() { disposed++; } }; });
  f.devices[0].lose({ reason: 'unknown', message: 'simulated loss' });
  await until(() => f.session.state === 'ready' && f.devices.length === 2);
  assert.deepEqual(capabilities, [1, 2]);
  assert.deepEqual(submissions(f), [1, 2]);
  assert.equal(disposed, 1);
  f.devices[1].lose({ reason: 'unknown', message: 'again' });
  await until(() => f.session.state === 'failed');
  await tick();
  assert.equal(f.devices.length, 2, 'Loss recovery must not spin indefinitely');
  assert.match(f.session.lastError, /again/);
  await f.session.reload();
  assert.equal(f.session.state, 'ready');
  assert.equal(f.devices.length, 3);
  assert.equal(f.session.recoveryAttempts, 0);
  await f.session.close();
});

test('a frame producer completing after reload never submits to the new scene', async () => {
  const f = fixture(), frame = deferred(); let started = false;
  await f.session.load(() => ({ packet: packet(1), frame() { started = true; return frame.promise; } }));
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

test('frame submission is serialized and no-op frames do not inflate GPU counts', async () => {
  const f = fixture(); let frameCount = 0;
  await f.session.load(() => ({ packet: packet(1), frame() { return ++frameCount === 3 ? null : packet(2); } }));
  f.host.submitGate = deferred();
  const first = f.session.render(), second = f.session.render();
  await until(() => submissions(f).length === 2);
  await tick(); assert.equal(frameCount, 1);
  f.host.submitGate.resolve();
  await Promise.all([first, second]);
  await f.session.render();
  assert.equal(f.session.submittedFrames, 3);
  await f.session.close();
});

test('a failed frame stops already queued frames until an explicit reload', async () => {
  const f = fixture(); let calls = 0;
  await f.session.load(() => ({ packet: packet(1), frame() { calls++; throw new Error('invalid scene state'); } }));
  const first = f.session.render(), second = f.session.render();
  await Promise.all([assert.rejects(first, /invalid scene state/), aborts(second)]);
  assert.equal(calls, 1);
  assert.equal(f.session.state, 'failed');
  assert.deepEqual(submissions(f), [1]);
  await f.session.close();
});

test('invalid dimensions leave an existing scene untouched', async () => {
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

test('malformed, detached and shared packet objects never reach the host', async () => {
  const f = fixture(); let disposed = 0;
  const detached = new Uint8Array(20);
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  for (const value of [null, new Uint8Array(0), new Uint8Array(15), {}, detached, new Uint8Array(new SharedArrayBuffer(20))]) {
    await assert.rejects(f.session.load(() => ({ packet: value, dispose() { disposed++; } })), TypeError);
  }
  assert.equal(disposed, 6);
  assert.deepEqual(submissions(f), []);
  await f.session.close();
});

test('close during a GPU completion wait does not lose CPU scene disposal', async () => {
  const f = fixture(); let disposed = 0;
  await f.session.load(() => ({ packet: packet(), dispose() { disposed++; } }));
  f.host.queueGate = deferred();
  const loading = f.session.reload();
  const rejected = aborts(loading);
  await tick();
  await f.session.close();
  await rejected;
  assert.equal(disposed, 1);
  f.host.queueGate.resolve();
});

test('observer exceptions do not break ownership, including reentrant close', async () => {
  const f = fixture({ onState() { throw new Error('observer failure'); }, onError() { throw new Error('error observer failure'); } });
  await f.session.load(() => ({ packet: packet() }));
  assert.equal(f.session.state, 'ready');
  let reentrant;
  f.session.onState = () => { reentrant = f.session.close(); };
  const closed = f.session.close();
  assert.strictEqual(reentrant, closed);
  await closed;
  assert.equal(f.log.filter(e => e[0] === 'unconfigure').length, 1);
});

test('device loss recovery can be disabled and intentional shutdown never reacquires', async () => {
  const f = fixture({ maxRecoveryAttempts: 0 });
  await f.session.load(() => ({ packet: packet() }));
  f.devices[0].lose({ reason: 'unknown' });
  await until(() => f.session.state === 'failed');
  await f.session.close(); await tick();
  assert.equal(f.devices.length, 1);
  const g = fixture();
  await g.session.load(() => ({ packet: packet() }));
  await g.session.close(); await tick();
  assert.equal(g.devices.length, 1);
});
