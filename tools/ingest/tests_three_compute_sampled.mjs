/** Public renderer/FIFO integration with recording GPU and source-builder seams.
 * Executes the actual compute and sampled-owner modules, not copied algorithms.
 * Run: node --experimental-vm-modules --test tools/ingest/tests_three_compute_sampled.mjs
 * This does not exercise the pinned WGSL compiler or a physical WebGPU device.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

class Events {
  addEventListener(type, fn) { ((this.listeners ??= {})[type] ??= new Set()).add(fn); }
  removeEventListener(type, fn) { this.listeners?.[type]?.delete(fn); }
  dispose() { for (const fn of [...this.listeners?.dispose ?? []]) fn(); }
}
class Texture extends Events {
  constructor(value = 1) {
    super(); this.isDataTexture = true; this.version = 1;
    this.source = {version: 1, dataReady: true, data: {width: 1, height: 1, data: new Uint8Array([value, 2, 3, 4])}};
    this.mipmaps = []; this.updateRanges = []; this.onUpdate = null;
  }
  get image() { return this.source.data; }
  touch(value) { this.image.data[0] = value; this.version++; this.source.version++; }
}
const plain = value => JSON.parse(JSON.stringify(value));
const sampled = texture => ({texture, isSampledTexture: true, isSampler: true});
function kernel(bindings) {
  return Object.assign(new Events(), {isComputeNode: true, version: 0, workgroupSize: [1], count: 1,
    groups: [{bindings}]});
}
async function rig() {
  const events = [], resident = new Map(); let failure = null, builds = 0;
  const gpu = {
    prepareSync(shader) { events.push(['prepare', shader]); return shader; },
    async prepare(shader) { events.push(['compile', shader]); return shader; },
    dispatch(shader, bindings, dimensions) { events.push(['dispatch', shader, bindings, dimensions]); },
    async whenIdle() { events.push(['gpuIdle']); },
    async read() { events.push(['read']); return new ArrayBuffer(4); },
    getBuffer() { return {}; }, release() {}, dispose() { events.push(['gpuDispose']); },
  };
  const storage = {
    captureBinding(binding) { return {storage: true, binding}; },
    layout(snapshot) { assert.equal(snapshot.storage, true); return {storageTexture: {access: 'write-only', format: 'rgba8unorm'}}; },
    resolve() { return {storageView: true}; },
    validateDispatch(snapshots) { assert.ok(snapshots.every(s => s.storage)); },
    written(snapshots) { events.push(['written', snapshots.length]); },
    async whenIdle() {}, bindingsFor(retained) { return retained; },
    dispose() { events.push(['storageDispose']); },
  };
  const pool = {
    inspect(texture) {
      const viewDimension = texture.isData3DTexture ? '3d' : '2d';
      const sampleType = texture.sampleType ?? 'float';
      return {data: true, viewDimension, sampleType, key: viewDimension + sampleType};
    },
    prepare(textures) {
      for (const t of textures) if (!resident.has(t)) {
        resident.set(t, {view: {pixels: [...t.image.data]}, sampler: {nativeSampler: true}});
        events.push(['textureUpload']); t.onUpdate?.(t);
      }
    },
    binding(texture) { return resident.get(texture); },
    retain(textures) { for (const t of resident.keys()) if (!textures.includes(t)) resident.delete(t); },
    async whenIdle() { if (failure) throw failure; },
    dispose() { resident.clear(); events.push(['textureDispose']); },
  };
  const context = vm.createContext({console, ArrayBuffer, DataView, Int8Array, Int16Array, Int32Array,
    Uint8Array, Uint16Array, Uint32Array, Float32Array, Set, Map, WeakMap, WeakSet});
  const synthetic = exports => new vm.SyntheticModule(Object.keys(exports), function () {
    for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
  }, {context});
  const dependencies = new Map([
    ['./gpu_compute.mjs', synthetic({createGpuCompute: () => gpu})],
    ['./gpu_compute_texture_copy.mjs', synthetic({copyComputeBufferToTarget: async (_d, _b, _o, _t, _p, options) => {
      events.push(['copy']); await options.sourceCompletion();
    }})],
    ['./three_storage_textures.mjs', synthetic({createThreeStorageTextures: () => storage, inspectThreeStorageTexture: () => ({})})],
    ['./three_textures.mjs', synthetic({createGpuThreeTextures: () => pool})],
    ['./three_framebuffer_textures.mjs', synthetic({createThreeFramebufferTextures() { throw new Error('No framebuffer textures in this fixture'); }})],
    ['./three_depth_textures.mjs', synthetic({createThreeDepthTextures() { throw new Error('No detached depth textures in this fixture'); }})],
    ['./three_compute_targets.mjs', new vm.SourceTextModule(await readFile(new URL('./three_compute_targets.mjs', import.meta.url), 'utf8'), {context})],
    ['./three_compute_textures.mjs', new vm.SourceTextModule(await readFile(new URL('./three_compute_textures.mjs', import.meta.url), 'utf8'), {context})],
  ]);
  const module = new vm.SourceTextModule(await readFile(new URL('./three_compute.mjs', import.meta.url), 'utf8'), {context});
  await module.link(specifier => {
    const dependency = dependencies.get(specifier); assert.ok(dependency, `Unexpected import: ${specifier}`); return dependency;
  });
  await module.evaluate();
  class Builder {
    constructor(_object, renderer) { this.renderer = renderer; }
    build() {
      builds++;
      for (const group of this.compute.groups) for (const binding of group.bindings) if (binding.isSampler || binding.isSampledTexture)
        this.renderer._backend.utils.getTextureSampleData(binding.textureNode?.value ?? binding.texture);
      this.computeShader = '@compute @workgroup_size(1) fn main() {}';
    }
    getBindings() { return this.compute.groups; }
  }
  class Base {
    constructor() {
      this.isF3DRenderer = this.isWebGPURenderer = this._initialized = true;
      this._device = {limits: {}, features: new Set()}; this._targetLifetime = new AbortController();
      this._nodes = {nodeFrame: {updateNode() {}, updateBeforeNode() {}, updateAfterNode() {}}};
      this._backend = {}; this._targets = {capture(t) { return t; }}; this.queue = [];
      this.info = {calls: 0, compute: {calls: 0, frameCalls: 0}};
    }
    _enqueue(request) { this.queue.push(request); }
    async flush() {
      while (this.queue.length) {
        const request = this.queue.shift();
        try { request.resolve(await request.issue()); } catch (error) { request.reject(error); }
      }
    }
    async waitForGPU() { await this.flush(); }
    dispose() { this._disposed = true; }
  }
  const Renderer = module.namespace.withThreeCompute(Base, {REVISION: '186', WGSLNodeBuilder: Builder});
  return {renderer: new Renderer(), events, resident, get builds() { return builds; }, fail(error) { failure = error; }};
}
const dispatches = r => r.events.filter(e => e[0] === 'dispatch');

test('public compute binds sampled input, sampler and existing writable storage output', async () => {
  const r = await rig(), input = new Texture(8), output = {isStorageTexture: true};
  r.renderer.compute(kernel([sampled(input), {isSampler: true, texture: input}, {...sampled(output), store: true}]));
  const [, shader, groups, dimensions] = dispatches(r)[0];
  assert.deepEqual(groups[0].entries[0].resource.pixels, [8, 2, 3, 4]);
  assert.equal(groups[0].entries[1].resource.nativeSampler, true);
  assert.equal(groups[0].entries[2].resource.storageView, true);
  assert.equal(shader.bindGroupLayouts[0].entries[0].texture.sampleType, 'float');
  assert.equal(shader.bindGroupLayouts[0].entries[1].sampler.type, 'filtering');
  assert.deepEqual(plain(dimensions), [1, 1, 1]);
  assert.deepEqual(r.events.filter(e => e[0] === 'written'), [['written', 1]]);
  assert.equal(r.events.filter(e => e[0] === 'textureUpload').length, 1);
  await r.renderer.waitForGPU(); r.renderer.dispose();
});
test('blocked public FIFO preserves every call-time pixel version and dispatch size', async () => {
  const r = await rig(), texture = new Texture(1), node = kernel([sampled(texture)]);
  r.renderer._preparing = true;
  const size = [2, 1, 1], first = r.renderer.computeAsync(node, size);
  texture.touch(9); size[0] = 3; const second = r.renderer.computeAsync(node, size);
  texture.image.data[0] = 77; size[0] = 88;
  assert.equal(dispatches(r).length, 0); assert.equal(r.renderer.queue.length, 2);
  await r.renderer.flush(); await Promise.all([first, second]);
  assert.deepEqual(dispatches(r).map(d => d[2][0].entries[0].resource.pixels[0]), [1, 9]);
  assert.deepEqual(dispatches(r).map(d => d[3][0]), [2, 3]);
  assert.equal(r.resident.size, 1); assert.equal(r.builds, 1); r.renderer.dispose();
});
test('live texture-node replacement invalidates compiled dimension and sample layouts', async () => {
  const r = await rig(), textureNode = {value: new Texture(2)};
  const binding = {...sampled(textureNode.value), textureNode}, node = kernel([binding]);
  r.renderer.compute(node);
  textureNode.value = Object.assign(new Texture(5), {isDataTexture: false, isData3DTexture: true, sampleType: 'uint'});
  r.renderer.compute(node);
  assert.equal(r.builds, 2);
  const shader = dispatches(r)[1][1];
  assert.equal(shader.bindGroupLayouts[0].entries[0].texture.viewDimension, '3d');
  assert.equal(shader.bindGroupLayouts[0].entries[0].texture.sampleType, 'uint');
  assert.equal(dispatches(r)[1][2][0].entries[0].resource.pixels[0], 5); r.renderer.dispose();
});
test('compileComputeAsync warms source inputs without executing a kernel', async () => {
  const r = await rig(), texture = new Texture(4), node = kernel([sampled(texture)]), progress = [];
  await r.renderer.compileComputeAsync(node, event => progress.push(event.loaded));
  assert.deepEqual(progress, [1]); assert.equal(dispatches(r).length, 0);
  assert.equal(r.events.filter(e => e[0] === 'compile').length, 1);
  r.renderer.compute(node); assert.equal(dispatches(r).length, 1);
  assert.equal(r.events.filter(e => e[0] === 'textureUpload').length, 1); r.renderer.dispose();
});
test('failed captures release leases instead of exhausting version capacity', async () => {
  const r = await rig(), texture = new Texture(), node = kernel([sampled(texture)]); node.count = -1;
  for (let i = 0; i < 270; i++) {
    texture.touch(i % 255);
    assert.throws(() => r.renderer.compute(node), {code: 'THREE_COMPUTE_DISPATCH'});
  }
  node.count = 1; r.renderer.compute(node); assert.equal(dispatches(r).length, 1);
  r.renderer.dispose(); assert.equal(texture.listeners.dispose.size, 0);
});
test('source disposal invalidates a queued public dispatch without executing GPU work', async () => {
  const r = await rig(), texture = new Texture(), node = kernel([sampled(texture)]);
  r.renderer._preparing = true;
  const pending = r.renderer.computeAsync(node); const rejected = assert.rejects(pending, {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  texture.dispose(); await r.renderer.flush(); await rejected;
  assert.equal(dispatches(r).length, 0); assert.equal(r.resident.size, 0); r.renderer.dispose();
});
test('texture upload completion failures propagate through wait, readback and copy', async () => {
  const r = await rig(); r.renderer.compute(kernel([sampled(new Texture())]));
  const error = new Error('sampled texture GPU validation failed'); r.fail(error);
  await assert.rejects(r.renderer.waitForGPU(), e => e === error);
  const attribute = {isStorageBufferAttribute: true, itemSize: 1, count: 1, version: 0, array: new Uint32Array(1)};
  await assert.rejects(r.renderer.getArrayBufferAsync(attribute), e => e === error);
  await assert.rejects(r.renderer.copyStorageBufferToRenderTargetAsync(attribute, {}), e => e === error);
  assert.equal(r.events.filter(e => e[0] === 'read').length, 1);
  assert.equal(r.events.filter(e => e[0] === 'copy').length, 1); r.renderer.dispose();
});
test('renderer disposal releases owned sampled resources and borrowed storage exactly once', async () => {
  const r = await rig(), texture = new Texture(); r.renderer.compute(kernel([sampled(texture)]));
  r.renderer.dispose(); assert.equal(r.resident.size, 0); assert.equal(texture.listeners.dispose.size, 0);
  for (const name of ['gpuDispose', 'textureDispose', 'storageDispose'])
    assert.equal(r.events.filter(e => e[0] === name).length, 1);
});
