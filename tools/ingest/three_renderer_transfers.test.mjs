/** Real public renderer/FIFO and transfer planners with an in-memory target
 * boundary. No GPU shader execution or native pixel-parity claim. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createWebGPURendererClass, createWebGLRendererClass} from './three_renderer.mjs';
import {inspectGpuTargetWrite, gpuTargetWriteData, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';
import {withThreeNativeCapture} from './three_node_shadows.mjs';
import {PerspectiveCamera, Matrix4, Layers, Texture} from '../../upstream/three.js/build/three.core.js';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
class Vector {
  constructor(x = 0, y = 0, z = 0, w = 0) { this.set(x, y, z, w); }
  set(x, y, z = 0, w = 0) { Object.assign(this, {x, y, z, w}); return this; }
  copy(v) { return this.set(v.x, v.y, v.z, v.w); }
}
class Color { constructor(r = 0, g = 0, b = 0) { Object.assign(this, {r, g, b}); } }
class Object3D { children = []; traverse(callback) { callback(this); } }
class Scene extends Object3D { isScene = true; }
class Camera extends PerspectiveCamera {}
const T = {REVISION: '186', Scene, Object3D, Camera, Color, Texture, Matrix4, Layers, Vector2: Vector, Vector4: Vector,
  NoToneMapping: 0, SRGBColorSpace: 'srgb'};
const layout = new Map([
  ['rgba8unorm', [Uint8Array, 4]], ['rgba8unorm-srgb', [Uint8Array, 4]],
  ['rgba16float', [Uint16Array, 4]], ['r32float', [Float32Array, 1]],
  ['rg32float', [Float32Array, 2]], ['rgba32float', [Float32Array, 4]],
]);
function target(format = 'rgba8unorm', width = 3, height = 2) {
  const [Type, components] = layout.get(format), bpp = Type.BYTES_PER_ELEMENT * components;
  return {format, width, height, sampleCount: 1, mipLevelCount: 1, bpp, bytes: new Uint8Array(width * height * bpp)};
}
function fixture({gl = false, three = T, device = null, ...limits} = {}) {
  const init = deferred(), calls = [], entries = new Map(), controls = {fence: null, prepare: null, writeError: null};
  let disposed = false;
  const stale = () => Object.assign(Error('stale generation'), {code: 'THREE_TARGET_STALE'});
  const owner = {
    capture(t) {
      if (disposed) throw Error('owner disposed');
      let e = entries.get(t);
      if (!e) {
        e = {target: t, shape: {...t}, alive: true, references: 0, lifetime: new AbortController(), validation: Promise.resolve(), storage: null};
        entries.set(t, e);
      }
      return e;
    },
    assertCurrent(e) {
      if (disposed || !e.alive || e.shape.width !== e.target.width || e.shape.height !== e.target.height || e.shape.format !== e.target.format) throw stale();
    },
    retainGeneration(e) {
      this.assertCurrent(e); e.references++;
      let held = true;
      return () => { if (held) { held = false; e.references--; } };
    },
    ensure(e) { this.assertCurrent(e); e.storage ??= {whenIdle: () => controls.fence?.promise ?? Promise.resolve()}; },
    writePixels(e, data, options) {
      this.assertCurrent(e);
      const plan = inspectGpuTargetWrite(e.shape, data, options);
      if (plan.empty) return false;
      if (controls.writeError) throw controls.writeError;
      this.ensure(e);
      const packed = gpuTargetWriteData(plan);
      calls.push({kind: 'upload', target: e.target, options, data: packed.data.slice()});
      for (let row = 0; row < plan.size.height; row++) {
        const offset = ((plan.origin.y + row) * e.shape.width + plan.origin.x) * e.target.bpp;
        e.target.bytes.set(packed.data.subarray(row * packed.layout.bytesPerRow, row * packed.layout.bytesPerRow + plan.rowBytes), offset);
      }
      return true;
    },
    copyColor(from, to, options) {
      this.assertCurrent(from); this.assertCurrent(to);
      const plan = inspectGpuTargetCopy(from.shape, to.shape, options);
      if (plan.empty) return false;
      this.ensure(from); this.ensure(to);
      calls.push({kind: 'copy', source: from.target, target: to.target, options});
      const bpp = from.target.bpp, rowBytes = plan.size.width * bpp;
      for (let row = 0; row < plan.size.height; row++) {
        const start = ((plan.sourceOrigin.y + row) * from.shape.width + plan.sourceOrigin.x) * bpp;
        const offset = ((plan.destinationOrigin.y + row) * to.shape.width + plan.destinationOrigin.x) * bpp;
        to.target.bytes.set(from.target.bytes.subarray(start, start + rowBytes), offset);
      }
      return true;
    },
    retire(t) { const e = entries.get(t); if (e) { e.alive = false; e.lifetime.abort(stale()); } },
    async whenIdle() { await Promise.all([...entries.values()].flatMap(e => [e.validation, e.storage?.whenIdle()])); },
    dispose() { disposed = true; for (const t of entries.keys()) this.retire(t); },
  };
  const Base = (gl ? createWebGLRendererClass : createWebGPURendererClass)(three, limits);
  class Harness extends Base {
    async _init() { await init.promise; if (this._disposed) throw Error('disposed during init'); this._initialized = true; return this; }
    async _prepare(destination) {
      if (controls.prepare) await controls.prepare.promise;
      destination._dispatcher = {entry: () => ({key: '00000'})};
    }
    _submit(request) { calls.push({kind: 'draw'}); request.scene.onSubmit?.(); }
  }
  const renderer = new Harness({canvas: {width: 8, height: 8}});
  if (device) renderer._device = device;
  renderer._targets.dispose(); renderer._targets = owner;
  return {renderer, init, owner, entries, calls, controls,
    async ready() { init.resolve(); await renderer.init(); },
    async drain() { while (renderer._drain) await renderer._drain; },
  };
}
const flush = () => new Promise(resolve => setImmediate(resolve));
const counts = (r, requests, bytes) => {
  assert.equal(r.info.f3d.pendingTransfers, requests);
  assert.equal(r.info.f3d.pendingTransferBytes, bytes);
};

test('public target writes and copies cannot overwrite an earlier sampled parent attachment', async () => {
  const f = fixture(), r = f.renderer, a = target(), b = target();
  for (const t of [a, b]) {
    t.texture = {isTexture: true, isRenderTargetTexture: true, renderTarget: t};
    t.depthTexture = {isTexture: true, isDepthTexture: true, renderTarget: t};
  }
  await f.ready(); await r.writeRenderTargetPixelsAsync(b, new Uint8Array(b.bytes.length).fill(7));
  const before = f.calls.length, rejected = [];
  withThreeNativeCapture(r, record => {
    record(b.texture); record(b.depthTexture);
    const refusal = {code: 'THREE_NODE_PROGRAM_SHADOW'};
    rejected.push(assert.rejects(r.writeRenderTargetPixelsAsync(b, new Uint8Array(b.bytes.length).fill(23)), refusal));
    rejected.push(assert.rejects(r.copyRenderTargetPixelsAsync(a, b), refusal));
    rejected.push(assert.rejects(r.copyRenderTargetDepthPixelsAsync(a, b), refusal));
    rejected.push(assert.rejects(r.copyTextureToTextureAsync(a.texture, b.texture), refusal));
    rejected.push(assert.rejects(r.copyStorageBufferToRenderTargetAsync({isStorageBufferAttribute: true}, b), refusal));
    assert.throws(() => r.copyTextureToTexture(a.texture, b.texture), refusal);
    assert.throws(() => r.copyTextureToTexture(a.depthTexture, b.depthTexture), refusal);
    counts(r, 0, 0); assert.equal(r._pending.length, 0); assert.equal(f.calls.length, before);
  });
  await Promise.all(rejected);
  assert.ok(b.bytes.every(value => value === 7));
  await r.writeRenderTargetPixelsAsync(b, new Uint8Array(b.bytes.length).fill(29));
  assert.ok(b.bytes.every(value => value === 29)); counts(r, 0, 0); r.dispose();
});

// Use the actual renderer, compute owner, texture owner and GPU upload path.
// Only shader construction and host image delivery are controlled here.
async function computeFixture(t) {
  const core = await import('../../upstream/three.js/build/three.core.js');
  class Builder {
    build() {
      this.computeShader = `// ${this.compute.name}\n@compute @workgroup_size(1) fn main() {}`;
      this.updateBeforeNodes = this.compute.before ?? []; this.updateNodes = [];
      this.updateAfterNodes = this.compute.after ?? [];
    }
    getBindings() { return this.compute.groups; }
  }
  class NodeFrame {
    frameId = 0;
    update() { this.frameId++; }
    updateNode(run) { run(); } updateBeforeNode(run) { run(); } updateAfterNode(run) { run(); }
  }
  class ImageBitmap {
    constructor(value = 1) { this.width = this.height = 1; this.value = value; this.closeCalls = 0; }
    close() { this.width = this.height = 0; this.closeCalls++; }
  }
  class HTMLImageElement {
    constructor(value = 1) { this.width = this.height = this.naturalWidth = this.naturalHeight = 1; this.complete = true; this.value = value; }
  }
  const deliveries = [], bitmaps = [], saved = new Map(['ImageBitmap', 'HTMLImageElement', 'createImageBitmap', 'structuredClone']
    .map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const clone = globalThis.structuredClone;
  Object.assign(globalThis, {ImageBitmap, HTMLImageElement,
    structuredClone: value => value instanceof ImageBitmap ? new ImageBitmap(value.value) : clone(value),
    createImageBitmap(image) {
      const bitmap = new ImageBitmap(image.value), gate = deferred();
      bitmaps.push(bitmap);
      deliveries.push(() => gate.resolve(bitmap)); return gate.promise;
    }});
  const device = textureDevice(), mapping = deferred(); device.features = new Set();
  const f = fixture({three: {...core, WGSLNodeBuilder: Builder, NodeFrame}, device});
  const allocate = device.createBuffer, encode = device.createCommandEncoder;
  device.createBuffer = options => Object.assign(allocate(options), {mapAsync: () => mapping.promise});
  device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createCommandEncoder = () => ({...encode(),
    copyBufferToBuffer(source, from, destination, to, size) {
      new Uint8Array(destination.data, to, size).set(new Uint8Array(source.data, from, size));
      f.calls.push({kind: 'readback'});
    },
    beginComputePass() {
      let pipeline; const groups = new Map();
      return {setPipeline(value) { pipeline = value; }, setBindGroup(slot, group) { groups.set(slot, group); },
        dispatchWorkgroups() {
          const values = [...groups.values()].flatMap(group => group.entries.flatMap(({resource}) =>
            resource.buffer ? [new Float32Array(resource.buffer.data)[0]] : []));
          f.calls.push({kind: pipeline.label, values});
        }, end() {}};
    }});
  t.after(() => {
    f.renderer.dispose(); mapping.resolve(); deliveries.forEach(deliver => deliver());
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  const image = new HTMLImageElement(), texture = new core.Texture(image);
  texture.generateMipmaps = false; texture.minFilter = core.NearestFilter; texture.needsUpdate = true;
  const attribute = new core.BufferAttribute(new Float32Array([3]), 1); attribute.isStorageBufferAttribute = true;
  const uniform = {isUniformBuffer: true, buffer: new Float32Array([5, 0, 0, 0])};
  const sampled = {texture, isSampledTexture: true, isSampler: true};
  const buffers = [uniform, {isStorageBuffer: true, attribute}];
  const kernel = (name, bindings = []) => Object.assign(new core.EventDispatcher(),
    {isComputeNode: true, name, version: 0, workgroupSize: [1], count: 1, groups: [{bindings}]});
  return {...f, device, mapping, deliveries, bitmaps, image, texture, attribute, uniform, sampled, buffers, kernel, ImageBitmap};
}

for (const create of [createWebGPURendererClass, createWebGLRendererClass])
  test(`${create.name}: transfer limits are validated before construction`, () => {
    for (const n of [0, -1, 1.2, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      assert.throws(() => create(T, {maxTransferBytes: n}), /transfer byte and request limits/);
      assert.throws(() => create(T, {maxTransfers: n}), /transfer byte and request limits/);
    }
  });

test('public pre-init uploads, draw, copy and readback retain a single submission order', async () => {
  const f = fixture(), r = f.renderer, a = target(), b = target(), gate = deferred();
  const input = new Uint8Array(a.bytes.length).fill(3);
  const upload = r.writeRenderTargetPixelsAsync(a, input);
  const scene = new Scene(); scene.onSubmit = () => a.bytes.fill(9);
  const draw = r.renderAsync(scene, new Camera());
  const copy = r.copyRenderTargetPixelsAsync(a, b);
  const read = new Promise((resolve, reject) => r._enqueue({readback: true, resolve, reject,
    issue() { const bytes = b.bytes.slice(); f.calls.push({kind: 'readback'}); return gate.promise.then(() => bytes); }}));
  const next = r.writeRenderTargetPixelsAsync(b, new Uint8Array(b.bytes.length).fill(17));
  input.fill(211);
  counts(r, 3, a.bytes.length + b.bytes.length);
  assert.equal(f.calls.length, 0);
  await f.ready(); await f.drain();
  assert.deepEqual(f.calls.map(c => c.kind), ['upload', 'draw', 'copy', 'readback', 'upload']);
  assert.ok(b.bytes.every(n => n === 17));
  gate.resolve(); assert.ok((await read).every(n => n === 9));
  await Promise.all([upload, draw, copy, next]); counts(r, 0, 0); r.dispose();
});

test('image submission readiness blocks the real FIFO while later readback fences do not', async () => {
  const f = fixture(), r = f.renderer, ready = deferred(), mapped = deferred();
  await f.ready();
  const image = new Promise((resolve, reject) => r._enqueue({transfer: true, awaitSubmission: true, resolve, reject,
    issue: () => ready.promise.then(() => { f.calls.push({kind: 'image-compute'}); })}));
  const read = new Promise((resolve, reject) => r._enqueue({readback: true, resolve, reject,
    issue() { f.calls.push({kind: 'readback'}); return mapped.promise; }}));
  const draw = r.renderAsync(new Scene(), new Camera());
  await flush();
  assert.deepEqual(f.calls, []); assert.ok(r._drain);
  ready.resolve(); await f.drain(); await Promise.all([image, draw]);
  assert.deepEqual(f.calls.map(call => call.kind), ['image-compute', 'readback', 'draw']);
  let complete = false; read.then(() => { complete = true; });
  await flush(); assert.equal(complete, false);
  mapped.resolve(); await read; r.dispose();
});

test('failed image readiness rejects its FIFO entry before following submissions advance', async () => {
  const f = fixture(), r = f.renderer, ready = deferred(), failure = new Error('image decode');
  await f.ready();
  const image = new Promise((resolve, reject) => r._enqueue({transfer: true, awaitSubmission: true, resolve, reject,
    issue: () => ready.promise.then(() => { assert.fail('failed image cannot submit'); })}));
  const rejected = assert.rejects(image, error => error === failure);
  const following = new Promise((resolve, reject) => r._enqueue({transfer: true, resolve, reject,
    issue() { f.calls.push({kind: 'following'}); }}));
  await flush(); assert.equal(f.calls.length, 0);
  ready.reject(failure); await rejected; await following; await f.drain();
  assert.deepEqual(f.calls.map(call => call.kind), ['following']); r.dispose();
});

test('pre-init image kernels serialize nested/list submissions and preserve queued uniform and storage bytes', async t => {
  const f = await computeFixture(t), r = f.renderer;
  const initial = f.kernel('initial'), image = f.kernel('image', [f.sampled, ...f.buffers]);
  const nested = f.kernel('nested', f.buffers), last = f.kernel('last', f.buffers);
  const original = f.attribute.array; let read;
  image.onInitFunction = () => r.compute(initial);
  image.after = [() => {
    read = r.getArrayBufferAsync(f.attribute);
    f.uniform.buffer[0] = 9; f.attribute.array[0] = 7; f.attribute.needsUpdate = true;
    r.compute(nested);
  }];
  const compute = r.compute([image, last]);
  const following = new Promise((resolve, reject) => r._enqueue({transfer: true, resolve, reject,
    issue() { f.calls.push({kind: 'following'}); }}));
  await f.ready(); await flush();
  assert.deepEqual(f.calls.map(call => call.kind), ['initial']); assert.equal(f.deliveries.length, 1);
  f.uniform.buffer[0] = 77; f.attribute.array[0] = 88; f.attribute.needsUpdate = true; f.image.value = 99;
  f.deliveries[0](); await Promise.all([compute, following]); await f.drain();
  assert.deepEqual(f.calls.map(call => call.kind), ['initial', 'image', 'readback', 'nested', 'last', 'following']);
  assert.deepEqual(f.calls.filter(call => call.values?.length).map(call => call.values), [[5, 3], [9, 7], [9, 7]]);
  assert.equal(f.attribute.array, original); assert.equal(original[0], 88);
  assert.equal(f.device.externalCopies[0].source.source.value, 1);
  let mapped = false; read.then(() => { mapped = true; }); await flush(); assert.equal(mapped, false);
  f.mapping.resolve(); assert.equal(new Float32Array(await read)[0], 3);
});

test('pending captured images privatize buffers even when the native upload images are authored bitmaps', async t => {
  const f = await computeFixture(t), r = f.renderer;
  f.texture.flipY = false; f.texture.mipmaps = [new f.ImageBitmap(4)];
  await f.ready();
  const original = f.attribute.array, compute = r.compute(f.kernel('authored', [f.sampled, ...f.buffers]));
  f.uniform.buffer[0] = 77; original[0] = 88; f.attribute.needsUpdate = true;
  await flush(); assert.deepEqual(f.calls, []); assert.equal(f.deliveries.length, 1);
  f.deliveries[0](); await compute; await f.drain();
  assert.deepEqual(f.calls, [{kind: 'authored', values: [5, 3]}]);
  assert.equal(f.attribute.array, original); assert.equal(original[0], 88);
  assert.equal(f.texture.image, f.image); assert.equal(f.texture.mipmaps[0].closeCalls, 0);
});

test('renderer disposal rejects unresolved image compute before delivery and closes late private pixels', async t => {
  const f = await computeFixture(t), r = f.renderer; await f.ready();
  const original = f.attribute.array, pending = r.compute(f.kernel('disposed', [f.sampled, ...f.buffers]));
  const rejected = assert.rejects(pending, {code: 'F3D_RENDERER_DISPOSED'});
  await flush(); assert.equal(f.deliveries.length, 1); assert.equal(f.calls.length, 0);
  r.dispose(); await rejected;
  assert.equal(f.calls.length, 0); assert.equal(f.device.externalCopies.length, 0);
  f.deliveries[0](); await f.drain();
  assert.equal(f.bitmaps[0].closeCalls, 1); assert.equal(f.calls.length, 0);
  assert.equal(f.texture.image, f.image); assert.equal(f.image.width, 1); assert.equal(f.attribute.array, original);
});

for (const [format, [Type, components]] of layout) for (const gl of [false, true])
  test(`${format} ${gl ? 'WebGL' : 'WebGPU'}: snapshots selected rows and raw bits`, async () => {
    const f = fixture({gl}), r = f.renderer, t = target(format, 3, 3);
    const rowBytes = 2 * components * Type.BYTES_PER_ELEMENT, stride = rowBytes * 2;
    const raw = new Uint8Array(4096); raw.fill(211);
    // Includes NaN payload/sign bits when interpreted as float32; copy bytes.
    const first = new Uint8Array(rowBytes), second = new Uint8Array(rowBytes);
    for (let i = 0; i < rowBytes; i++) { first[i] = [1, 0, 192, 127][i % 4]; second[i] = [0, 0, 0, 128][i % 4]; }
    raw.set(first, 16); raw.set(second, 16 + stride);
    const data = new Type(raw.buffer, 8), options = {x: 1, y: 0, width: 2, height: 2, offset: 8, bytesPerRow: stride};
    const pending = r.writeRenderTargetPixelsAsync(t, data, options);
    counts(r, 1, rowBytes * 2);
    options.x = -100; options.width = 300;
    structuredClone(raw.buffer, {transfer: [raw.buffer]});
    await f.ready(); await pending;
    const call = f.calls[0]; assert.equal(call.options.x, 1); assert.equal(call.options.width, 2);
    assert.equal(call.options.bytesPerRow, rowBytes); assert.equal(call.options.offset, 0);
    assert.equal(call.options.flipY, gl);
    assert.deepEqual(call.data, new Uint8Array(gl ? [...second, ...first] : [...first, ...second]));
    counts(r, 0, 0); r.dispose();
  });

test('initialized upload submits synchronously; completion does not block later work', async () => {
  const f = fixture(), r = f.renderer, t = target(), gate = deferred(); await f.ready();
  f.controls.fence = gate;
  const first = r.writeRenderTargetPixelsAsync(t, new Uint8Array(t.bytes.length).fill(4));
  assert.equal(f.calls.length, 1); assert.equal(r._pending.length, 0);
  const second = r.writeRenderTargetPixelsAsync(t, new Uint8Array(t.bytes.length).fill(6));
  assert.equal(f.calls.length, 2); counts(r, 2, t.bytes.length * 2);
  assert.equal(f.owner.capture(t).references, 2, 'each transfer retains the generation through its completion fence');
  let complete = false; const idle = r.waitForGPU().then(() => { complete = true; });
  await flush(); assert.equal(complete, false);
  gate.resolve(); await Promise.all([first, second, idle]); counts(r, 0, 0);
  assert.equal(f.owner.capture(t).references, 0); r.dispose();
});

test('copy regions, mip level, target identity and explicit origin override are captured', async () => {
  const f = fixture({gl: true}), r = f.renderer, a = target('rgba8unorm', 8, 8), b = target('rgba8unorm', 4, 4);
  a.mipLevelCount = 4;
  const options = {sourceX: 1, sourceY: 1, x: 2, y: 2, width: 1, height: 1, sourceMipLevel: 1, flipY: false};
  const pending = r.copyRenderTargetPixelsAsync(a, b, options);
  Object.assign(options, {sourceMipLevel: 20, sourceX: 100, flipY: true});
  r._renderTarget = target();
  await f.ready(); await pending;
  assert.equal(f.calls[0].source, a); assert.equal(f.calls[0].target, b);
  assert.equal(f.calls[0].options.sourceMipLevel, 1); assert.equal(f.calls[0].options.sourceX, 1);
  assert.equal(f.calls[0].options.flipY, false); counts(r, 0, 0); r.dispose();
});

test('byte/request admission is shared, bounded and retryable after cancellation', async () => {
  const f = fixture({maxTransferBytes: 24, maxTransfers: 2}), r = f.renderer, t = target(), controller = new AbortController();
  const p = r.writeRenderTargetPixelsAsync(t, new Uint8Array(24), {signal: controller.signal});
  await assert.rejects(r.writeRenderTargetPixelsAsync(t, new Uint8Array(24)), /shared request or byte budget/);
  const copy = r.copyRenderTargetPixelsAsync(t, target());
  await assert.rejects(r.copyRenderTargetPixelsAsync(t, target()), /shared request or byte budget/);
  counts(r, 2, 24); controller.abort(Error('caller cancelled')); await assert.rejects(p, /caller cancelled/);
  counts(r, 1, 0);
  const retry = r.writeRenderTargetPixelsAsync(t, new Uint8Array(24));
  await f.ready(); await Promise.all([copy, retry]); counts(r, 0, 0); r.dispose();
});

test('a full render FIFO rejects transfers without retaining upload snapshots', async () => {
  const f = fixture(), r = f.renderer;
  for (let i = 0; i < 64; i++) r._enqueue({readback: true, issue() {}, resolve() {}, reject() {}});
  await assert.rejects(r.writeRenderTargetPixelsAsync(target(), new Uint8Array(24)), /Too many render operations/);
  counts(r, 0, 0); r.dispose(); f.init.resolve(); await r._drain?.catch(() => {});
});

for (const which of ['caller', 'source', 'destination', 'renderer'])
  test(`${which} disposal/cancellation rejects queued copies without awaiting init`, async () => {
    const f = fixture(), r = f.renderer, a = target(), b = target(), controller = new AbortController();
    const pending = r.copyRenderTargetPixelsAsync(a, b, {signal: controller.signal});
    const leases = [f.entries.get(a), f.entries.get(b)];
    assert.deepEqual(leases.map(e => e.references), [1, 1]);
    if (which === 'caller') controller.abort(Error('cancelled'));
    if (which === 'source') f.owner.retire(a);
    if (which === 'destination') f.owner.retire(b);
    if (which === 'renderer') r.dispose();
    await assert.rejects(pending); counts(r, 0, 0); assert.equal(f.calls.length, 0);
    assert.deepEqual(leases.map(e => e.references), [0, 0], 'cancellation releases both captured generations');
    f.init.resolve(); await f.drain().catch(() => {}); assert.equal(f.calls.length, 0); r.dispose();
  });

test('queued upload cancellation releases bytes and never submits the snapshot', async () => {
  const f = fixture(), r = f.renderer, t = target(), controller = new AbortController();
  const pending = r.writeRenderTargetPixelsAsync(t, new Uint8Array(24), {signal: controller.signal});
  controller.abort(Error('stop upload')); await assert.rejects(pending, /stop upload/); counts(r, 0, 0);
  await f.ready(); await f.drain(); assert.equal(f.calls.length, 0); r.dispose();
});

test('cancellation after submission rejects completion, without pretending to roll back writes', async () => {
  const f = fixture(), r = f.renderer, t = target(), controller = new AbortController(), gate = deferred(); await f.ready();
  f.controls.fence = gate;
  const pending = r.writeRenderTargetPixelsAsync(t, new Uint8Array(24).fill(7), {signal: controller.signal});
  assert.ok(t.bytes.every(n => n === 7)); controller.abort(Error('stop completion'));
  await assert.rejects(pending, /stop completion/); counts(r, 0, 0); gate.resolve(); r.dispose();
});

for (const which of ['source', 'destination'])
  test(`${which} direct generation edits reject before copy submission`, async () => {
    const f = fixture(), r = f.renderer, a = target(), b = target();
    const pending = r.copyRenderTargetPixelsAsync(a, b);
    (which === 'source' ? a : b).width++;
    await f.ready(); await assert.rejects(pending, /stale generation/);
    assert.equal(f.calls.length, 0); counts(r, 0, 0); r.dispose();
  });

test('native completion errors and source validation errors reach the public promise', async () => {
  for (const which of ['completion', 'source-validation']) {
    const f = fixture(), r = f.renderer, a = target(), b = target(), gate = deferred(); await f.ready();
    if (which === 'completion') f.controls.fence = gate;
    else f.owner.capture(a).validation = gate.promise;
    const pending = r.copyRenderTargetPixelsAsync(a, b);
    gate.reject(Error(which)); await assert.rejects(pending, new RegExp(which)); counts(r, 0, 0); r.dispose();
  }
});

test('deferred draw failure rejects following transfers and is not silently bypassed', async () => {
  const f = fixture(), r = f.renderer, gate = deferred(); f.controls.prepare = gate;
  const draw = r.renderAsync(new Scene(), new Camera()); draw.catch(() => {});
  const pending = r.writeRenderTargetPixelsAsync(target(), new Uint8Array(24));
  await f.ready(); gate.reject(Error('prepare failed'));
  await assert.rejects(draw, /prepare failed/); await assert.rejects(pending, /prepare failed/);
  counts(r, 0, 0); assert.equal(f.calls.length, 0);
  await assert.rejects(r.copyRenderTargetPixelsAsync(target(), target()), /prepare failed/); r.dispose();
});

test('a transfer submission exception rejects its promise without dropping subsequent requests', async () => {
  const f = fixture(), r = f.renderer, a = target(), b = target();
  f.controls.writeError = Error('upload failed');
  const pending = r.writeRenderTargetPixelsAsync(a, new Uint8Array(24));
  const copy = r.copyRenderTargetPixelsAsync(a, b);
  await f.ready(); await assert.rejects(pending, /upload failed/); await copy;
  assert.deepEqual(f.calls.map(c => c.kind), ['copy']); counts(r, 0, 0);
  assert.deepEqual([f.entries.get(a).references, f.entries.get(b).references], [0, 0]); r.dispose();
});

test('invalid options and unsupported transfers fail before GPU work or queue admission', async () => {
  const f = fixture(), r = f.renderer, t = target(); let invoked = false;
  const accessor = Object.defineProperty({}, 'x', {get() { invoked = true; return 0; }});
  for (const options of [null, [], new Date(), accessor, {[Symbol('x')]: 0}, {signal: {}}, {unknown: true},
    {x: -1}, {width: 10}, {bytesPerRow: 2}, {maxBytes: 1}, {flipY: 1}])
    await assert.rejects(r.writeRenderTargetPixelsAsync(t, new Uint8Array(24), options));
  assert.equal(invoked, false);
  await assert.rejects(r.writeRenderTargetPixelsAsync(t, new Float32Array(24)), /matching typed array/);
  await assert.rejects(r.copyRenderTargetPixelsAsync(t, t), /In-place/);
  await assert.rejects(r.copyRenderTargetPixelsAsync(t, target('rgba32float')), /copy-compatible/);
  const multisample = target(); multisample.sampleCount = 4;
  await assert.rejects(r.copyRenderTargetPixelsAsync(t, multisample), /single-sample/);
  await assert.rejects(r.copyRenderTargetPixelsAsync(t, target(), {sourceMipLevel: 2}), /mip level/);
  assert.equal(r._pending.length, 0); assert.equal(f.calls.length, 0); counts(r, 0, 0); r.dispose();
});

test('pre-aborted calls and empty regions allocate no snapshots or native storage', async () => {
  const f = fixture(), r = f.renderer, t = target(), controller = new AbortController(); controller.abort(Error('already stopped'));
  await assert.rejects(r.writeRenderTargetPixelsAsync(t, new Uint8Array(24), {signal: controller.signal}), /already stopped/);
  counts(r, 0, 0); assert.equal(r._pending.length, 0);
  const empty = r.writeRenderTargetPixelsAsync(t, new Uint8Array(0), {width: 0});
  await f.ready(); assert.equal(await empty, false);
  assert.equal(f.owner.capture(t).storage, null); counts(r, 0, 0); r.dispose();
});

test('many transfers share one lifetime/caller listener and remove it after completion', async () => {
  const f = fixture(), r = f.renderer, a = target(), b = target(), controller = new AbortController();
  const countsBySignal = new Map();
  for (const signal of [r._targetLifetime.signal, f.owner.capture(a).lifetime.signal, f.owner.capture(b).lifetime.signal, controller.signal]) {
    const count = {added: 0, removed: 0}; countsBySignal.set(signal, count);
    const add = signal.addEventListener.bind(signal), remove = signal.removeEventListener.bind(signal);
    signal.addEventListener = (...args) => { count.added++; return add(...args); };
    signal.removeEventListener = (...args) => { count.removed++; return remove(...args); };
  }
  const pending = Array.from({length: 64}, () => r.copyRenderTargetPixelsAsync(a, b, {signal: controller.signal}));
  for (const value of countsBySignal.values()) assert.equal(value.added, 1);
  await f.ready(); await Promise.all(pending);
  for (const value of countsBySignal.values()) assert.equal(value.removed, 1);
  counts(r, 0, 0); r.dispose();
});

for (const gl of [false, true]) test(`${gl ? 'WebGL' : 'WebGPU'} public target texture copies compose with the storage facade and retain FIFO capture`, async () => {
  const f = fixture({gl}), r = f.renderer, a = target('rgba8unorm', 4, 4), b = target('rgba8unorm', 4, 4);
  for (const t of [a, b]) t.texture = {isTexture: true, isRenderTargetTexture: true, renderTarget: t};
  await f.ready();
  const gate = deferred(); f.controls.prepare = gate;
  const draw = r.renderAsync(new Scene(), new Camera());
  const before = r.writeRenderTargetPixelsAsync(a, new Uint8Array(a.bytes.length).fill(9));
  const region = {min: {x: 1, y: 1}, max: {x: 3, y: 3}}, position = {x: 0, y: 0};
  assert.equal(r.copyTextureToTexture(a.texture, b.texture, region, position), undefined);
  region.max.x = 999; position.x = 999;
  const after = r.writeRenderTargetPixelsAsync(a, new Uint8Array(a.bytes.length).fill(17));
  assert.equal(f.calls.length, 0);
  gate.resolve(); await Promise.all([draw, before, after]); await r.waitForGPU();
  assert.deepEqual(f.calls.map(call => call.kind), ['draw', 'upload', 'copy', 'upload']);
  const call = f.calls.find(call => call.kind === 'copy');
  assert.equal(call.options.width, 2); assert.equal(call.options.height, 2); assert.equal(call.options.x, 0);
  assert.equal(call.options.sourceMipLevel, 0); assert.equal(call.options.flipY, gl);
  assert.ok(a.bytes.every(n => n === 17));
  const originY = gl ? 2 : 0;
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++)
    assert.equal(b.bytes[(y * 4 + x) * 4], x < 2 && y >= originY && y < originY + 2 ? 9 : 0);
  assert.throws(() => r.copyTextureToTexture(a.texture, b.texture, null, null, 0, 1), {code: 'F3D_RENDERER_UNSUPPORTED'});
  counts(r, 0, 0); r.dispose();
});
