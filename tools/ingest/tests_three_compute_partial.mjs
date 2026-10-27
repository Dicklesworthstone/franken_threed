/** Actual sampled owner + public compute mixin with recording GPU/pool/builder
 * boundaries. No pinned shader compiler or physical-GPU execution claim.
 * Run: node --experimental-vm-modules --test tools/ingest/tests_three_compute_partial.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {createThreeComputeTextures} from './three_compute_textures.mjs';

class Events {
  addEventListener(type, fn) { ((this.listeners ??= {})[type] ??= new Set()).add(fn); }
  removeEventListener(type, fn) { this.listeners?.[type]?.delete(fn); }
  dispose() { for (const fn of [...this.listeners?.dispose ?? []]) fn(); }
}
class Texture extends Events {
  constructor({width = 4, height = 2, layers = 1, channels = 4, Kind = Uint8Array, alignment = 4, array = false} = {}) {
    super();
    const pitch = Math.ceil(width * channels * Kind.BYTES_PER_ELEMENT / alignment) * alignment;
    const data = new Kind(pitch * height * layers / Kind.BYTES_PER_ELEMENT).fill(1);
    Object.assign(this, {isDataTexture: !array, isDataArrayTexture: array, version: 1,
      source: {version: 1, dataReady: true, data: {width, height, depth: layers, data}},
      channels, unpackAlignment: alignment, flipY: false, wrapS: 'clamp-to-edge',
      mipmaps: [], updateRanges: [], onUpdate: null, generateMipmaps: false});
    if (array) this.layerUpdates = new Set();
  }
  get image() { return this.source.data; }
  touch() { this.version++; this.source.version++; }
  clearUpdateRanges() { this.updateRanges.length = 0; }
  clearLayerUpdates() { this.layerUpdates.clear(); }
}
class CompressedTexture extends Texture {
  constructor({width = 8, height = 8, layers = 3, blockWidth = 4, blockHeight = 4, blockBytes = 16} = {}) {
    super(); this.isDataTexture = false; this.isCompressedTexture = this.isCompressedArrayTexture = true;
    this.source.data = {width, height, depth: layers}; this.layerUpdates = new Set();
    Object.assign(this, {blockWidth, blockHeight, blockBytes}); this.mipmaps = [];
    const levels = 1 + Math.floor(Math.log2(Math.max(width, height)));
    for (let level = 0; level < levels; level++) {
      const w = Math.max(1, width >> level), h = Math.max(1, height >> level);
      const size = Math.ceil(w / blockWidth) * Math.ceil(h / blockHeight) * blockBytes * layers;
      this.mipmaps.push({width: w, height: h, data: new Uint8Array(size).fill(level + 1)});
    }
  }
}
const sampled = texture => ({texture, isSampledTexture: true, isSampler: true});
const sampler = texture => ({texture, isSampler: true});
const kernel = bindings => Object.assign(new Events(), {isComputeNode: true, version: 0,
  workgroupSize: [1], count: 1, groups: [{bindings}]});
const bytes = view => [...new Uint8Array(view.data.buffer, view.data.byteOffset, view.data.byteLength)];
const tick = () => new Promise(resolve => setImmediate(resolve));

function poolRig(options = {}) {
  const uploads = [], resident = new Map(), sourceViews = new Map(); let failure = null;
  const storage = {captureBinding: b => ({storage: true, b}), layout: () => ({storageTexture: {format: 'rgba8unorm'}}),
    resolve: () => ({storage: true}), validateDispatch(list) { assert.ok(list.every(s => s.storage)); }, written() {},
    async whenIdle() {}, bindingsFor: r => r, dispose() {}};
  const pool = {
    inspect(t) {
      if (t.isCompressedArrayTexture) {
        const {width, height, depth: layers} = t.image, levels = t.mipmaps.length;
        const {blockWidth, blockHeight, blockBytes} = t;
        return {data: true, blockUploads: true, array: true, manual: true, width, height, layers, levels,
          sampleType: 'float', viewDimension: '2d-array',
          key: JSON.stringify(['compressed', width, height, layers, levels, blockWidth, blockHeight, blockBytes, t.wrapS]),
          uploads: t.mipmaps.map((image, level) => ({image, level, layer: 0, depth: layers,
            sliceBytes: Math.ceil(image.width / blockWidth) * Math.ceil(image.height / blockHeight) * blockBytes}))};
      }
      const array = t.isDataArrayTexture, bpc = t.image.data.BYTES_PER_ELEMENT;
      const viewDimension = array ? '2d-array' : '2d', sampleType = t.sampleType ?? 'float';
      const shape = {data: true, width: t.image.width, height: t.image.height, layers: t.image.depth,
        channels: t.channels, bpc, alignment: t.unpackAlignment, array, layered: array,
        flipY: t.flipY, cube: false, volume: false, integral: sampleType === 'uint' || sampleType === 'sint',
        manual: !!t.mipmaps.length, sampleType, viewDimension};
      shape.key = JSON.stringify([shape, t.wrapS, t.generateMipmaps]); return shape;
    },
    prepare(textures) {
      for (const t of textures) {
        assert.equal(t.updateRanges.length, 0, 'native uploader receives completed immutable pixels');
        assert.equal(t.layerUpdates?.size ?? 0, 0);
        let binding = sourceViews.get(t.source);
        if (!binding) {
          binding = {view: {data: t.image.data ? new t.image.data.constructor(t.image.data) : null,
            mips: t.mipmaps.map(image => new Uint8Array(image.data)), texture: t}, sampler: {wrapS: t.wrapS}};
          sourceViews.set(t.source, binding); uploads.push(binding.view);
          t.onUpdate?.(t);
        }
        resident.set(t, binding);
      }
    },
    binding: t => resident.get(t),
    retain(textures) {
      for (const t of resident.keys()) if (!textures.includes(t)) resident.delete(t);
      for (const source of sourceViews.keys()) if (!textures.some(t => t.source === source)) sourceViews.delete(source);
    },
    async whenIdle() { if (failure) throw failure; },
    dispose() { resident.clear(); sourceViews.clear(); },
  };
  const owner = createThreeComputeTextures({storage, poolFactory: () => pool, ...options});
  return {owner, pool, storage, uploads, resident, fail(error) { failure = error; }};
}
function consume(r, t) {
  const lease = r.owner.captureBinding(sampled(t));
  try { return r.owner.resolve(lease); } finally { r.owner.release(lease); }
}
async function publicRig() {
  const r = poolRig(), events = [], buffers = new Map();
  const gpu = {prepareSync: s => s, async prepare() {},
    upload(key, data) { const buffer = {data: [...data]}; buffers.set(key, buffer); return buffer; },
    getBuffer: key => buffers.get(key) ?? {}, release: key => buffers.delete(key),
    dispatch(shader, groups, dimensions) { events.push({shader, groups, dimensions}); },
    async whenIdle() {}, async read() { return new ArrayBuffer(4); }, dispose() {}};
  const context = vm.createContext({console, ArrayBuffer, DataView, Uint8Array, Uint16Array, Uint32Array,
    Int8Array, Int16Array, Int32Array, Float32Array, Map, Set, WeakMap, WeakSet});
  const synthetic = exports => new vm.SyntheticModule(Object.keys(exports), function () {
    for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
  }, {context});
  const modules = new Map([
    ['./gpu_compute.mjs', synthetic({createGpuCompute: () => gpu})],
    ['./gpu_compute_texture_copy.mjs', synthetic({copyComputeBufferToTarget: async (_a, _b, _c, _d, _e, o) => o.sourceCompletion()})],
    ['./three_storage_textures.mjs', synthetic({createThreeStorageTextures: () => r.storage, inspectThreeStorageTexture() {}})],
    ['./three_textures.mjs', synthetic({createGpuThreeTextures: () => r.pool})],
    ['./three_framebuffer_textures.mjs', synthetic({createThreeFramebufferTextures() { throw new Error('No framebuffer textures in this fixture'); }})],
    ['./three_depth_textures.mjs', synthetic({createThreeDepthTextures() { throw new Error('No detached depth textures in this fixture'); }})],
    ['./three_compute_targets.mjs', synthetic({createThreeComputeTargets() { throw new Error('No native targets in this fixture'); }})],
    ['./three_compute_textures.mjs', new vm.SourceTextModule(await readFile(new URL('./three_compute_textures.mjs', import.meta.url), 'utf8'), {context})],
  ]);
  const module = new vm.SourceTextModule(await readFile(new URL('./three_compute.mjs', import.meta.url), 'utf8'), {context});
  await module.link(name => { assert.ok(modules.has(name), name); return modules.get(name); }); await module.evaluate();
  class Builder {
    build() { this.computeShader = '@compute @workgroup_size(1) fn main() {}';
      this.updateBeforeNodes = this.compute.before ?? []; this.updateAfterNodes = this.compute.after ?? []; }
    getBindings() { return this.compute.groups; }
  }
  class Base {
    constructor() {
      this.isF3DRenderer = this.isWebGPURenderer = this._initialized = true; this._device = {limits: {}, features: new Set()};
      this._targetLifetime = new AbortController(); this._targets = {capture: t => t}; this._backend = {};
      this._nodes = {nodeFrame: {updateNode: fn => fn(), updateBeforeNode: fn => fn(), updateAfterNode: fn => fn()}};
      this.info = {calls: 0, compute: {calls: 0, frameCalls: 0}}; this.queue = [];
    }
    _enqueue(request) { if (this.queueFailure) throw this.queueFailure; this.queue.push(request); }
    async flush() {
      while (this.queue.length) { const request = this.queue.shift();
        try { request.resolve(await request.issue()); } catch (error) { request.reject(error); } }
    }
    async waitForGPU() { await this.flush(); }
    dispose() { this._disposed = true; }
  }
  const Renderer = module.namespace.withThreeCompute(Base, {REVISION: '186', WGSLNodeBuilder: Builder});
  return {...r, renderer: new Renderer(), events};
}
const seen = r => r.events.map(e => bytes(e.groups[0].entries[0].resource));

test('RGBA row deltas preserve unmarked GPU-visible pixels, array identities and source IDs', () => {
  const r = poolRig(), t = new Texture(), data = t.image.data, source = t.source;
  consume(r, t); data.fill(90); t.updateRanges.push({start: 4, count: 4}); t.touch();
  const result = bytes(consume(r, t));
  assert.deepEqual(result, [...Array(4).fill(1), ...Array(4).fill(90), ...Array(24).fill(1)]);
  assert.equal(t.image.data, data); assert.equal(t.source, source); assert.equal(t.updateRanges.length, 0); r.owner.dispose();
});
test('the first partial upload starts with zero storage, not a full CPU upload', () => {
  const r = poolRig(), t = new Texture(); t.image.data.fill(7); t.updateRanges.push({start: 20, count: 4});
  assert.deepEqual(bytes(consume(r, t)), [...Array(20).fill(0), ...Array(4).fill(7), ...Array(8).fill(0)]); r.owner.dispose();
});
test('two blocked versions compose sequential deltas without recopying unmarked writes', () => {
  const r = poolRig(), t = new Texture(); consume(r, t);
  t.image.data.fill(8); t.updateRanges = [{start: 0, count: 4}]; t.touch(); const a = r.owner.captureBinding(sampled(t));
  t.image.data.fill(9); t.updateRanges = [{start: 16, count: 4}]; t.touch(); const b = r.owner.captureBinding(sampled(t));
  t.image.data.fill(77);
  assert.deepEqual(bytes(r.owner.resolve(a)), [...Array(4).fill(8), ...Array(28).fill(1)]);
  assert.equal(t.updateRanges.length, 1, 'old upload must not clear the newer range request');
  assert.deepEqual(bytes(r.owner.resolve(b)), [...Array(4).fill(8), ...Array(12).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]);
  assert.equal(t.updateRanges.length, 0); r.owner.release(a); r.owner.release(b); r.owner.dispose();
});
test('row coalescing preserves source component rounding without sorting or mutating public ranges', () => {
  const r = poolRig(), t = new Texture(); t.image.data.set(Array.from({length: 32}, (_, i) => i + 1));
  const ranges = [{start: 5, count: 2}, {start: 1, count: 3}], original = structuredClone(ranges); t.updateRanges = ranges;
  const lease = r.owner.captureBinding(sampled(t)); assert.deepEqual(ranges, original);
  assert.deepEqual(bytes(r.owner.resolve(lease)), [...Array.from({length: 8}, (_, i) => i + 1), ...Array(24).fill(0)]);
  r.owner.release(lease); r.owner.dispose();
});
test('unpack-aligned rows use padded byte offsets and never upload padding as texels', () => {
  const r = poolRig(), t = new Texture({width: 3, height: 2, alignment: 8}); consume(r, t);
  t.image.data.fill(9); t.updateRanges = [{start: 12, count: 4}]; t.touch();
  const got = bytes(consume(r, t)); assert.deepEqual(got, [...Array(16).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]); r.owner.dispose();
});
test('zero-count row updates retain prior pixels and are acknowledged only on submission', () => {
  const r = poolRig(), t = new Texture(); consume(r, t); t.image.data.fill(6);
  t.updateRanges = [{start: 32, count: 0}]; t.touch(); const lease = r.owner.captureBinding(sampled(t));
  assert.equal(t.updateRanges.length, 1); assert.deepEqual(bytes(r.owner.resolve(lease)), Array(32).fill(1));
  assert.equal(t.updateRanges.length, 0); r.owner.release(lease); r.owner.dispose();
});
test('DataArrayTexture layer updates copy complete selected layers at exact typed bit precision', () => {
  for (const Kind of [Uint8Array, Uint16Array, Int16Array, Float32Array, Uint32Array, Int32Array]) {
    const r = poolRig(), t = new Texture({width: 2, height: 2, layers: 3, channels: 2, Kind, array: true});
    consume(r, t); new Uint8Array(t.image.data.buffer).fill(0xa5); t.layerUpdates.add(1); t.touch();
    const lease = r.owner.captureBinding(sampled(t)); t.image.data.fill(0);
    const view = r.owner.resolve(lease), raw = bytes(view), slice = 8 * Kind.BYTES_PER_ELEMENT;
    const old = new Uint8Array(new Kind(8).fill(1).buffer);
    assert.ok(view.data instanceof Kind); assert.deepEqual(raw, [...old, ...Array(slice).fill(0xa5), ...old]);
    assert.equal(t.layerUpdates.size, 0); r.owner.release(lease); r.owner.dispose();
  }
});
test('first array-layer updates leave every unmentioned layer zeroed', () => {
  const r = poolRig(), t = new Texture({width: 1, height: 1, layers: 3, array: true});
  t.layerUpdates.add(2); assert.deepEqual(bytes(consume(r, t)), [...Array(8).fill(0), ...Array(4).fill(1)]); r.owner.dispose();
});
test('source aliases share the same composed payload and one upload/callback', () => {
  const r = poolRig(), a = new Texture(), b = new Texture(); b.source = a.source;
  consume(r, a); a.image.data.fill(8); a.updateRanges = [{start: 0, count: 4}]; a.touch(); b.version++;
  b.updateRanges = [{start: 16, count: 4}]; let callbacks = 0; a.onUpdate = b.onUpdate = () => callbacks++;
  const x = r.owner.captureBinding(sampled(a)), y = r.owner.captureBinding(sampled(b));
  assert.equal(x.record.pixels, y.record.pixels);
  assert.equal(r.owner.resolve(x), r.owner.resolve(y)); assert.equal(callbacks, 1);
  assert.deepEqual(bytes(r.owner.resolve(y)), [...Array(4).fill(8), ...Array(28).fill(1)]);
  assert.equal(b.updateRanges.length, 1, 'an alias without an upload is not acknowledged');
  r.owner.release(x); r.owner.release(y); r.owner.dispose();
});
test('descriptor/source replacement resets a partial baseline rather than reusing foreign pixels', () => {
  for (const change of [t => { t.wrapS = 'repeat'; }, t => { t.source = {...t.source, data: {...t.image}}; }]) {
    const r = poolRig(), t = new Texture(); consume(r, t); t.image.data.fill(9); t.updateRanges = [{start: 0, count: 4}];
    change(t); t.touch(); assert.deepEqual(bytes(consume(r, t)), [...Array(4).fill(9), ...Array(28).fill(0)]); r.owner.dispose();
  }
});
test('unversioned edits to an acknowledged range/layer request are never silently cleared', () => {
  const r = poolRig(), t = new Texture(); t.updateRanges = [{start: 0, count: 4}];
  const lease = r.owner.captureBinding(sampled(t)); t.updateRanges[0].start = 16;
  r.owner.resolve(lease); assert.equal(t.updateRanges[0].start, 16); r.owner.release(lease); r.owner.dispose();
  const q = poolRig(), a = new Texture({array: true, layers: 2}); a.layerUpdates.add(0);
  const use = q.owner.captureBinding(sampled(a)); a.layerUpdates.add(1); q.owner.resolve(use);
  assert.deepEqual([...a.layerUpdates], [0, 1]); q.owner.release(use); q.owner.dispose();
});
test('upload callbacks observe acknowledgment and can request the next partial version', () => {
  const r = poolRig(), t = new Texture(); t.updateRanges = [{start: 0, count: 4}];
  t.onUpdate = function (texture) {
    assert.equal(this, t); assert.equal(texture, t); assert.equal(t.updateRanges.length, 0);
    t.updateRanges.push({start: 16, count: 4}); t.touch();
  };
  consume(r, t); assert.deepEqual(t.updateRanges, [{start: 16, count: 4}]); r.owner.dispose();
});
test('capture failure restores every previous source version instead of poisoning future deltas', () => {
  const r = poolRig(), a = new Texture(), b = new Texture(); consume(r, a); consume(r, b);
  const original = {...r.owner.diagnostics}; a.image.data.fill(8); a.updateRanges = [{start: 0, count: 4}]; a.touch();
  let stale;
  assert.throws(() => r.owner.captureDispatch(() => { stale = r.owner.captureBinding(sampled(a));
    r.owner.release(stale); throw new Error('second binding rejected'); }), /second binding rejected/);
  assert.deepEqual(r.owner.diagnostics, original); assert.throws(() => r.owner.resolve(stale), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  a.image.data.fill(9); a.updateRanges = [{start: 16, count: 4}]; a.touch();
  assert.deepEqual(bytes(consume(r, a)), [...Array(16).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]); r.owner.dispose();
});
test('replacement budgets charge the old baseline until the replacement commits', () => {
  const r = poolRig({maxSnapshotBytes: 32}), t = new Texture(); consume(r, t);
  t.image.data.fill(9); t.updateRanges = [{start: 0, count: 4}]; t.touch();
  assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_BUDGET'});
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 32});
  assert.equal(t.listeners.dispose.size, 1); r.owner.dispose(); assert.equal(t.listeners.dispose.size, 0);
});
test('invalid partial layouts and rows are refused before any snapshot or upload', () => {
  for (const mutate of [t => t.updateRanges.push({start: 12, count: 8}), t => t.updateRanges.push({start: -1, count: 1}),
    t => { t.flipY = true; t.updateRanges.push({start: 0, count: 4}); },
    t => { t.mipmaps = [{data: t.image.data}]; t.updateRanges.push({start: 0, count: 4}); }]) {
    const r = poolRig(), t = new Texture(); mutate(t);
    assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_RANGES'});
    assert.equal(r.uploads.length, 0); assert.equal(r.owner.diagnostics.snapshotBytes, 0); r.owner.dispose();
  }
  const r = poolRig(), t = new Texture({array: true, layers: 2, width: 1, channels: 1}); t.layerUpdates.add(0);
  assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_RANGES'}); r.owner.dispose();
});
test('disposal invalidates all queued partial versions and releases every charged byte', () => {
  const r = poolRig(), t = new Texture(); consume(r, t); t.updateRanges = [{start: 0, count: 4}]; t.touch();
  const a = r.owner.captureBinding(sampled(t)); t.touch(); const b = r.owner.captureBinding(sampled(t)); t.dispose();
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(r.resident.size, 0);
  for (const use of [a, b]) { assert.throws(() => r.owner.resolve(use), {code: 'THREE_COMPUTE_TEXTURE_STALE'}); r.owner.release(use); }
  r.owner.dispose();
});
test('public compute FIFO consumes cumulative call-time deltas and never later unmarked writes', async () => {
  const r = await publicRig(), t = new Texture(), node = kernel([sampled(t), sampler(t)]); r.renderer.compute(node);
  r.renderer._preparing = true;
  t.image.data.fill(8); t.updateRanges = [{start: 0, count: 4}]; t.touch(); const a = r.renderer.computeAsync(node, [2]);
  t.image.data.fill(9); t.updateRanges = [{start: 16, count: 4}]; t.touch(); const b = r.renderer.computeAsync(node, [3]);
  t.image.data.fill(77); assert.equal(r.events.length, 1); await r.renderer.flush(); await Promise.all([a, b]);
  assert.deepEqual(seen(r), [Array(32).fill(1), [...Array(4).fill(8), ...Array(28).fill(1)],
    [...Array(4).fill(8), ...Array(12).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]]);
  assert.deepEqual(r.events.map(e => e.dimensions[0]), [1, 2, 3]); r.renderer.dispose();
});
test('public invalid dispatch capture cannot change the baseline for the next valid kernel', async () => {
  const r = await publicRig(), t = new Texture(), node = kernel([sampled(t)]); r.renderer.compute(node);
  t.image.data.fill(8); t.updateRanges = [{start: 0, count: 4}]; t.touch(); node.count = -1;
  assert.throws(() => r.renderer.compute(node), {code: 'THREE_COMPUTE_DISPATCH'});
  t.image.data.fill(9); t.updateRanges = [{start: 16, count: 4}]; t.touch(); node.count = 1; r.renderer.compute(node);
  assert.deepEqual(seen(r)[1], [...Array(16).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]); r.renderer.dispose();
});
test('public array-layer updates and compile preparation use the same snapshot path', async () => {
  const r = await publicRig(), t = new Texture({width: 1, height: 1, layers: 2, array: true}), node = kernel([sampled(t)]);
  t.layerUpdates.add(1); await r.renderer.compileComputeAsync(node); assert.equal(r.events.length, 0);
  t.image.data.fill(9); t.layerUpdates.add(0); t.touch(); r.renderer.compute(node);
  assert.deepEqual(seen(r)[0], [9, 9, 9, 9, 1, 1, 1, 1]); r.renderer.dispose();
});
test('after-update nested compute stays ordered and is not mistaken for transaction reentry', async () => {
  const r = await publicRig(), t = new Texture(), child = kernel([sampled(t)]), parent = kernel([sampled(t)]);
  parent.after = [() => { t.image.data.fill(7); t.updateRanges = [{start: 0, count: 4}]; t.touch(); r.renderer.compute(child); }];
  r.renderer._preparing = true; const done = r.renderer.computeAsync(parent); await r.renderer.flush(); await done;
  assert.deepEqual(seen(r), [Array(32).fill(1), [...Array(4).fill(7), ...Array(28).fill(1)]]); r.renderer.dispose();
});
test('partial input validation errors remain dependencies of wait/read/copy completion', async () => {
  const r = await publicRig(), t = new Texture(); t.updateRanges = [{start: 0, count: 4}]; r.renderer.compute(kernel([sampled(t)]));
  const error = new Error('partial input validation failed'); r.fail(error);
  await assert.rejects(r.renderer.waitForGPU(), e => e === error);
  const attribute = {isStorageBufferAttribute: true, count: 1, itemSize: 1, array: new Float32Array(1)};
  await assert.rejects(r.renderer.getArrayBufferAsync(attribute), e => e === error);
  await assert.rejects(r.renderer.copyStorageBufferToRenderTargetAsync(attribute, {}), e => e === error); r.renderer.dispose();
});
test('a refused public FIFO reservation cannot become a later partial-update baseline', async () => {
  const r = await publicRig(), t = new Texture(), node = kernel([sampled(t)]); r.renderer.compute(node);
  t.image.data.fill(8); t.updateRanges = [{start: 0, count: 4}]; t.touch(); r.renderer._preparing = true;
  const error = new Error('FIFO full'); r.renderer.queueFailure = error;
  await assert.rejects(r.renderer.computeAsync(node), e => e === error); await tick();
  r.renderer.queueFailure = null; r.renderer._deferredError = null; r.renderer._preparing = false;
  t.image.data.fill(9); t.updateRanges = [{start: 16, count: 4}]; t.touch(); r.renderer.compute(node);
  assert.deepEqual(seen(r)[1], [...Array(16).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]); r.renderer.dispose();
});

test('source-wide delta history survives disposal of its latest uploading alias', () => {
  const r = poolRig(), a = new Texture(), b = new Texture(); b.source = a.source; consume(r, a); consume(r, b);
  b.image.data.fill(8); b.updateRanges = [{start: 0, count: 4}]; b.touch(); consume(r, b); b.dispose();
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 64});
  a.image.data.fill(9); a.updateRanges = [{start: 16, count: 4}]; a.touch();
  assert.deepEqual(bytes(consume(r, a)), [...Array(4).fill(8), ...Array(12).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]);
  assert.equal(r.owner.diagnostics.snapshotBytes, 32); r.owner.dispose(); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('restoring residency from a shared history does not replay its upload callback', () => {
  const r = poolRig(), a = new Texture(), b = new Texture(); b.source = a.source; consume(r, a); consume(r, b);
  let callbacks = 0; a.onUpdate = b.onUpdate = () => callbacks++;
  b.image.data.fill(8); b.updateRanges = [{start: 0, count: 4}]; b.touch(); consume(r, b); b.dispose();
  assert.equal(callbacks, 1); assert.equal(r.uploads.length, 2);
  assert.deepEqual(bytes(consume(r, a)), [...Array(4).fill(8), ...Array(28).fill(1)]);
  assert.equal(r.uploads.length, 3); assert.equal(callbacks, 1); r.owner.dispose();
});
test('rollback preserves a charged history even when no texture record owns that version', () => {
  const r = poolRig({maxSnapshotBytes: 96}), a = new Texture(), b = new Texture(); b.source = a.source;
  consume(r, a); consume(r, b); b.image.data.fill(8); b.updateRanges = [{start: 0, count: 4}]; b.touch(); consume(r, b); b.dispose();
  a.image.data.fill(7); a.updateRanges = [{start: 16, count: 4}]; a.touch();
  assert.throws(() => r.owner.captureDispatch(() => {
    const lease = r.owner.captureBinding(sampled(a)); r.owner.release(lease); throw new Error('reservation rejected');
  }), /reservation rejected/);
  assert.equal(r.owner.diagnostics.snapshotBytes, 64);
  a.image.data.fill(9); a.updateRanges = [{start: 24, count: 4}]; a.touch();
  assert.deepEqual(bytes(consume(r, a)), [...Array(4).fill(8), ...Array(20).fill(1), ...Array(4).fill(9), ...Array(4).fill(1)]);
  r.owner.dispose(); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('disposing the last source alias releases orphan history bytes and allows a zero-based restart', () => {
  const r = poolRig(), a = new Texture(), b = new Texture(); b.source = a.source; consume(r, a); consume(r, b);
  b.updateRanges = [{start: 0, count: 4}]; b.touch(); consume(r, b); b.dispose(); a.dispose();
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(r.resident.size, 0);
  a.image.data.fill(9); a.updateRanges = [{start: 16, count: 4}]; a.touch();
  assert.deepEqual(bytes(consume(r, a)), [...Array(16).fill(0), ...Array(4).fill(9), ...Array(12).fill(0)]); r.owner.dispose();
});
test('source disposal during transaction rollback cannot resurrect a baseline or listener', () => {
  const r = poolRig(), t = new Texture(); consume(r, t);
  assert.throws(() => r.owner.captureDispatch(() => { t.dispose(); throw new Error('cancelled'); }), /cancelled/);
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(t.listeners.dispose.size, 0); r.owner.dispose();
});
test('compressed array layer updates preserve every unmentioned block at every authored mip', () => {
  for (const profile of [{blockBytes: 8}, {blockBytes: 16}, {width: 12, height: 12, blockWidth: 6, blockHeight: 6},
    {width: 4, height: 4, blockWidth: 1, blockHeight: 1, blockBytes: 4}]) {
    const r = poolRig(), t = new CompressedTexture(profile); consume(r, t);
    for (const mip of t.mipmaps) mip.data.fill(99); t.layerUpdates.add(1); t.touch();
    const view = consume(r, t);
    for (const [level, data] of view.mips.entries()) {
      const slice = data.length / 3;
      assert.deepEqual([...data], [...Array(slice).fill(level + 1), ...Array(slice).fill(99), ...Array(slice).fill(level + 1)]);
    }
    assert.equal(t.layerUpdates.size, 0); r.owner.dispose();
  }
});
test('queued compressed mip deltas retain each call-time block version and acknowledgment', () => {
  const r = poolRig(), t = new CompressedTexture(); consume(r, t);
  for (const mip of t.mipmaps) mip.data.fill(8); t.layerUpdates.add(0); t.touch(); const a = r.owner.captureBinding(sampled(t));
  for (const mip of t.mipmaps) mip.data.fill(9); t.layerUpdates = new Set([2]); t.touch(); const b = r.owner.captureBinding(sampled(t));
  for (const mip of t.mipmaps) mip.data.fill(77);
  const first = r.owner.resolve(a); assert.deepEqual([...t.layerUpdates], [2]); const second = r.owner.resolve(b);
  for (let level = 0; level < second.mips.length; level++) {
    const slice = second.mips[level].length / 3;
    assert.deepEqual([...first.mips[level]], [...Array(slice).fill(8), ...Array(slice * 2).fill(level + 1)]);
    assert.deepEqual([...second.mips[level]], [...Array(slice).fill(8), ...Array(slice).fill(level + 1), ...Array(slice).fill(9)]);
  }
  assert.equal(t.layerUpdates.size, 0); r.owner.release(a); r.owner.release(b); r.owner.dispose();
});
test('shared incoming mip arrays do not incorrectly share unequal historical tail blocks', () => {
  const r = poolRig(), t = new CompressedTexture(); consume(r, t);
  const shared = new Uint8Array(t.mipmaps[2].data.length).fill(9);
  t.mipmaps[2].data = t.mipmaps[3].data = shared; t.layerUpdates.add(0); t.touch();
  const view = consume(r, t), slice = shared.length / 3;
  assert.deepEqual([...view.mips[2]], [...Array(slice).fill(9), ...Array(slice * 2).fill(3)]);
  assert.deepEqual([...view.mips[3]], [...Array(slice).fill(9), ...Array(slice * 2).fill(4)]);
  assert.notEqual(view.texture.mipmaps[2].data, view.texture.mipmaps[3].data); r.owner.dispose();
});
test('compressed partial updates require a complete initial version instead of inventing zero block encodings', () => {
  const r = poolRig(), t = new CompressedTexture(); t.layerUpdates.add(1);
  assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_RANGES'});
  assert.equal(r.uploads.length, 0); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
  t.layerUpdates.clear(); consume(r, t); t.layerUpdates.add(1); t.touch(); consume(r, t); r.owner.dispose();
});
test('replacing a compressed mip array without a version bump prevents stale acknowledgment', () => {
  const r = poolRig(), t = new CompressedTexture(); consume(r, t); t.layerUpdates.add(1); t.touch();
  const use = r.owner.captureBinding(sampled(t)); t.mipmaps[0].data = new Uint8Array(t.mipmaps[0].data.length).fill(90);
  r.owner.resolve(use); assert.deepEqual([...t.layerUpdates], [1]); r.owner.release(use); r.owner.dispose();
});
test('compressed deltas recover the correct history after the last uploading alias is disposed', () => {
  const r = poolRig(), a = new CompressedTexture(), b = new CompressedTexture(); b.source = a.source; b.mipmaps = a.mipmaps;
  consume(r, a); consume(r, b); b.mipmaps.forEach(m => m.data.fill(8)); b.layerUpdates.add(0); b.touch(); consume(r, b); b.dispose();
  a.mipmaps.forEach(m => m.data.fill(9)); a.layerUpdates.add(2); a.touch(); const view = consume(r, a);
  for (const [level, data] of view.mips.entries()) {
    const slice = data.length / 3;
    assert.deepEqual([...data], [...Array(slice).fill(8), ...Array(slice).fill(level + 1), ...Array(slice).fill(9)]);
  }
  r.owner.dispose(); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('a rejected compressed delta is rolled back across the entire mip chain', () => {
  const r = poolRig(), t = new CompressedTexture(); consume(r, t); const before = r.owner.diagnostics.snapshotBytes;
  t.mipmaps.forEach(m => m.data.fill(8)); t.layerUpdates.add(0); t.touch();
  assert.throws(() => r.owner.captureDispatch(() => { const use = r.owner.captureBinding(sampled(t));
    r.owner.release(use); throw new Error('invalid dispatch'); }), /invalid dispatch/);
  assert.equal(r.owner.diagnostics.snapshotBytes, before);
  t.mipmaps.forEach(m => m.data.fill(9)); t.layerUpdates = new Set([2]); t.touch();
  for (const [level, data] of consume(r, t).mips.entries()) {
    const slice = data.length / 3;
    assert.deepEqual([...data], [...Array(slice * 2).fill(level + 1), ...Array(slice).fill(9)]);
  }
  r.owner.dispose();
});
test('compressed partial peak budgets count the whole composed pyramid, not only changed layers', () => {
  const t = new CompressedTexture(), size = t.mipmaps.reduce((total, mip) => total + mip.data.byteLength, 0);
  const r = poolRig({maxSnapshotBytes: size * 2 - 1}); consume(r, t); t.layerUpdates.add(1); t.touch();
  assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_BUDGET'});
  assert.equal(r.owner.diagnostics.snapshotBytes, size); r.owner.dispose(); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('public compressed-array compute uses captured mip histories through compile and FIFO dispatch', async () => {
  const r = await publicRig(), t = new CompressedTexture(), node = kernel([sampled(t)]);
  await r.renderer.compileComputeAsync(node); r.renderer._preparing = true;
  t.mipmaps.forEach(m => m.data.fill(8)); t.layerUpdates.add(0); t.touch(); const a = r.renderer.computeAsync(node);
  t.mipmaps.forEach(m => m.data.fill(9)); t.layerUpdates = new Set([2]); t.touch(); const b = r.renderer.computeAsync(node);
  t.mipmaps.forEach(m => m.data.fill(77)); await r.renderer.flush(); await Promise.all([a, b]);
  assert.equal(r.events.length, 2);
  const view = r.events[1].groups[0].entries[0].resource;
  for (const [level, data] of view.mips.entries()) {
    const slice = data.length / 3;
    assert.deepEqual([...data], [...Array(slice).fill(8), ...Array(slice).fill(level + 1), ...Array(slice).fill(9)]);
  }
  r.renderer.dispose();
});
test('capture transactions reject asynchronous/nested capture without leaking history charges', () => {
  const r = poolRig(), t = new Texture(); consume(r, t); const before = {...r.owner.diagnostics};
  assert.throws(() => r.owner.captureDispatch(async () => {}), {code: 'THREE_COMPUTE_TEXTURE_ASYNC_USE'});
  assert.throws(() => r.owner.captureDispatch(() => r.owner.captureDispatch(() => {})), {code: 'THREE_COMPUTE_TEXTURE_REENTRY'});
  assert.deepEqual(r.owner.diagnostics, before); r.owner.dispose();
});
test('an upload callback exception does not replay the acknowledged source upload on public retry', async () => {
  const r = await publicRig(), t = new Texture(), node = kernel([sampled(t)]); let calls = 0;
  const error = new Error('user upload callback failed'); t.onUpdate = () => { calls++; throw error; };
  assert.throws(() => r.renderer.compute(node), e => e === error);
  assert.equal(r.events.length, 0); r.renderer.compute(node);
  assert.equal(calls, 1); assert.equal(r.events.length, 1); assert.equal(r.uploads.length, 1); r.renderer.dispose();
});
test('a submitted partial upload remains the next delta baseline even if its callback throws', async () => {
  const r = await publicRig(), t = new Texture(), node = kernel([sampled(t)]); r.renderer.compute(node);
  t.image.data.fill(8); t.updateRanges = [{start: 0, count: 4}]; t.touch();
  const error = new Error('after native upload'); t.onUpdate = () => { throw error; };
  assert.throws(() => r.renderer.compute(node), e => e === error); assert.equal(t.updateRanges.length, 0);
  t.onUpdate = null; t.image.data.fill(9); t.updateRanges = [{start: 16, count: 4}]; t.touch(); r.renderer.compute(node);
  assert.deepEqual(seen(r)[1], [...Array(4).fill(8), ...Array(12).fill(1), ...Array(4).fill(9), ...Array(12).fill(1)]); r.renderer.dispose();
});
