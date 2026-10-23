/** Recording seams for native target / public compute tests. The actual target
 * owner, sampled owner, target adapter and compute mixin are loaded from disk.
 * Source constructors, WGSLNodeBuilder and the native device are test doubles;
 * these tests make no shader compilation or physical-GPU execution claim. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
export const plain = value => JSON.parse(JSON.stringify(value));
export class Events {
  addEventListener(type, listener) { ((this.listeners ??= {})[type] ??= new Set()).add(listener); }
  removeEventListener(type, listener) { this.listeners?.[type]?.delete(listener); }
  dispose() { const event = {type: 'dispose', target: this}; for (const listener of [...this.listeners?.dispose ?? []]) listener(event); }
}
export class Texture extends Events {
  constructor() {
    super(); Object.assign(this, {isTexture: true, isRenderTargetTexture: true, version: 0, source: {version: 0},
      type: 1009, format: 1023, colorSpace: '', generateMipmaps: false, internalFormat: null,
      minFilter: 1003, magFilter: 1003, wrapS: 1001, wrapT: 1001, anisotropy: 1});
  }
}
export class RenderTarget extends Events {
  constructor(width = 4, height = 4) {
    super(); Object.assign(this, {width, height, depth: 1, samples: 0, depthBuffer: false, stencilBuffer: false,
      resolveColorBuffer: true, textures: [new Texture()]}); this.texture.renderTarget = this;
  }
  get texture() { return this.textures[0]; }
  setSize(width, height) {
    if (this.width === width && this.height === height) return;
    this.width = width; this.height = height; this.dispose();
  }
}
export class DepthTexture extends Texture {
  constructor(type = 1015, format = 1026) {
    super(); Object.assign(this, {isRenderTargetTexture: false, isDepthTexture: true,
      type, format, flipY: false, compareFunction: null, mipmaps: [], image: {width: 4, height: 4}});
  }
}
export function depthTarget(type = 1015, format = 1026) {
  const target = new RenderTarget(); target.depthBuffer = true;
  target.depthTexture = new DepthTexture(type, format); target.depthTexture.renderTarget = target;
  return target;
}
export const THREE = {REVISION: '186', Texture, DepthTexture, RenderTarget,
  Compatibility: {TEXTURE_COMPARE: 'depthTextureCompare'},
  NeverCompare: 512, LessCompare: 513, EqualCompare: 514, LessEqualCompare: 515,
  GreaterCompare: 516, NotEqualCompare: 517, GreaterEqualCompare: 518, AlwaysCompare: 519,
  DepthFormat: 1026, DepthStencilFormat: 1027, UnsignedInt248Type: 1020,
  UnsignedByteType: 1009, HalfFloatType: 1016, FloatType: 1015, UnsignedIntType: 1014,
  RGBAFormat: 1023, RedFormat: 1028, RGFormat: 1030, NoColorSpace: '', LinearSRGBColorSpace: 'srgb-linear', SRGBColorSpace: 'srgb',
  NearestFilter: 1003, NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearFilter: 1006, LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008,
  ClampToEdgeWrapping: 1001, RepeatWrapping: 1000, MirroredRepeatWrapping: 1002};
export const sampled = texture => ({texture, isSampler: true, isSampledTexture: true});
export const sampler = texture => ({texture, isSampler: true});
export const kernel = bindings => Object.assign(new Events(), {isComputeNode: true, version: 0,
  workgroupSize: [1], count: 1, groups: [{bindings}]});
export const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };

export async function rig({features = [], maxBytes, maxTargets, adapterOptions = {}} = {}) {
  const events = [], storages = [], scopeErrors = [], scopePromises = [], buffers = new Map();
  const lost = deferred(); let samplerFailure = null, builds = 0;
  const device = {features: new Set(features), limits: {}, lost: lost.promise,
    pushErrorScope(type) { events.push(['pushScope', type]); },
    popErrorScope() { events.push(['popScope']); return scopePromises.shift() ?? Promise.resolve(scopeErrors.shift() ?? null); },
    createSampler(descriptor) {
      if (samplerFailure) throw samplerFailure;
      const value = {sampler: plain(descriptor)}; events.push(['sampler', value]); return value;
    },
    queue: {submit() { events.push(['submit']); }, async onSubmittedWorkDone() { events.push(['fence']); }},
  };
  function nativeTarget(_device, options) {
    assert.equal(_device, device);
    const texture = {value: 0, borrowed: false, options,
      createView(descriptor) { const view = {nativeTexture: texture, descriptor: plain(descriptor)}; events.push(['view', view]); return view; }};
    const depth = {...texture, options: {...options, format: options.depthFormat},
      createView(descriptor) { const view = {nativeTexture: depth, descriptor: plain(descriptor)}; events.push(['view', view]); return view; }};
    let busy = false, disposed = false, failure = null, completion = null, mipsDirty = true;
    const storage = {texture, depth,
      allocatedBytes: 0,
      frameAllocationBytes(mipLevel) { assert.equal(mipLevel, 0); return 0; },
      get failed() { return !!failure; },
      withTexture(run) {
        assert.equal(busy, false); assert.equal(disposed, false); busy = texture.borrowed = true;
        try {
          if (options.generateMipmaps && mipsDirty) { events.push(['mips', texture.value]); mipsDirty = false; }
          const result = run(texture); assert.equal(!!result?.then, false, 'native borrow must not cross await');
          return true;
        } finally { busy = texture.borrowed = false; }
      },
      withDepthTexture(run) {
        assert.equal(options.sampleDepth, true); assert.equal(busy, false); assert.equal(disposed, false);
        busy = depth.borrowed = true;
        try { const result = run(depth); assert.equal(!!result?.then, false); return true; }
        finally { busy = depth.borrowed = false; }
      },
      withSampledTextures(run) {
        assert.equal(options.sampleDepth, true); assert.equal(busy, false); assert.equal(disposed, false);
        busy = texture.borrowed = depth.borrowed = true;
        try {
          if (options.generateMipmaps && mipsDirty) { events.push(['mips', texture.value]); mipsDirty = false; }
          const result = run(texture, depth); assert.equal(!!result?.then, false); return true;
        } finally { busy = texture.borrowed = depth.borrowed = false; }
      },
      withFrame(run) {
        assert.equal(busy, false); assert.equal(disposed, false); busy = true;
        try { const result = run({depthView: {nativeTexture: depth}}, texture); mipsDirty = true; return result; } finally { busy = false; }
      },
      async whenIdle() { assert.equal(busy, false, 'completion must start after borrow ends'); if (failure) throw failure; if (completion) await completion; },
      dispose() { assert.equal(busy, false, 'source retirement must wait for borrow unwinding'); disposed = true; events.push(['targetDispose']); },
      fail(error) { failure = error; }, wait(promise) { completion = promise; }, get disposed() { return disposed; },
    };
    storages.push(storage); events.push(['targetAllocate', options]); return storage;
  }
  const gpu = {
    prepareSync(shader) { events.push(['prepare', shader]); return shader; },
    async prepare(shader) { events.push(['compile', shader]); return shader; },
    dispatch(shader, bindings, dimensions) {
      const values = [];
      for (const group of bindings) for (const {resource} of group.entries) if (resource.nativeTexture) {
        assert.equal(resource.nativeTexture.borrowed, true, 'native target must stay borrowed through dispatch');
        values.push(resource.nativeTexture.value);
      }
      events.push(['dispatch', shader, bindings, plain(dimensions), values]); device.queue.submit([]);
    },
    upload(key, data) { const buffer = {data: new data.constructor(data)}; buffers.set(key, buffer); return buffer; },
    captureUpload(key, data, version) { return {key, data, version}; }, releaseUpload() {},
    getBuffer(key) { return buffers.get(key) ?? {}; }, release(key) { buffers.delete(key); },
    async read() { events.push(['read']); return new ArrayBuffer(4); },
    async whenIdle() {}, dispose() { events.push(['gpuDispose']); },
  };
  const storage = {
    captureBinding(binding) { return {storage: true, binding}; },
    layout() { return {storageTexture: {access: 'write-only', format: 'rgba8unorm', viewDimension: '2d'}}; },
    resolve() { return {storageView: true}; },
    validateDispatch(snapshots) { assert.ok(snapshots.every(s => s.storage)); },
    written(snapshots) { events.push(['written', snapshots.length]); }, async whenIdle() {},
    bindingsFor(retained) { return retained; }, dispose() { events.push(['storageDispose']); },
  };
  const context = vm.createContext({console, ArrayBuffer, DataView, Int8Array, Int16Array, Int32Array,
    Uint8Array, Uint16Array, Uint32Array, Float32Array, Map, Set, WeakMap, WeakSet, AbortController});
  const synthetic = exports => new vm.SyntheticModule(Object.keys(exports), function () {
    for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
  }, {context});
  const modules = new Map([
    ['./gpu_render_target.mjs', synthetic({createGpuRenderTarget: nativeTarget})],
    ['./gpu_target_transfer.mjs', synthetic({inspectGpuTargetWrite() { return {empty: false}; }, inspectGpuTargetCopy() { return {empty: false}; }})],
    ['./gpu_depth_transfer.mjs', synthetic({inspectGpuDepthCopy() { return {empty: false}; }})],
    ['./gpu_compute.mjs', synthetic({createGpuCompute: () => gpu})],
    ['./gpu_compute_texture_copy.mjs', synthetic({copyComputeBufferToTarget: async (_d, _b, _o, _t, _p, options) => {
      events.push(['copy']); await options.sourceCompletion();
    }})],
    ['./three_storage_textures.mjs', synthetic({createThreeStorageTextures: () => storage, inspectThreeStorageTexture() {}})],
    ['./three_storage_texture_copy.mjs', synthetic({withStorageTextureCopies: Base => Base})],
    ['./three_textures.mjs', synthetic({createGpuThreeTextures() { throw new Error('Render targets must not use the CPU texture uploader'); }})],
  ]);
  for (const name of ['three_compute_targets', 'three_compute_textures', 'three_render_targets', 'three_compute', 'three_node_shadows'])
    modules.set(`./${name}.mjs`, new vm.SourceTextModule(await readFile(new URL(`./${name}.mjs`, import.meta.url), 'utf8'), {context, identifier: name}));
  const link = specifier => { assert.ok(modules.has(specifier), specifier); return modules.get(specifier); };
  for (const module of modules.values()) if (module.status === 'unlinked') await module.link(link);
  for (const module of modules.values()) if (module.status !== 'evaluated') await module.evaluate();
  const targets = modules.get('./three_render_targets.mjs').namespace.createThreeRenderTargets(THREE,
    {getDevice: () => device, ...(maxBytes ? {maxBytes} : {}), ...(maxTargets ? {maxTargets} : {})});
  class Builder {
    constructor(_object, renderer) { this.renderer = renderer; }
    build() {
      builds++;
      for (const group of this.compute.groups) for (const binding of group.bindings) if (binding.isSampler || binding.isSampledTexture)
        events.push(['sampleData', plain(this.renderer._backend.utils.getTextureSampleData(binding.textureNode?.value ?? binding.texture))]);
      this.computeShader = '@compute @workgroup_size(1) fn main() {}';
      this.updateBeforeNodes = this.compute.before ?? []; this.updateNodes = []; this.updateAfterNodes = this.compute.after ?? [];
    }
    getBindings() { return this.compute.groups; }
  }
  class Base {
    constructor() {
      this.isF3DRenderer = this.isWebGPURenderer = this._initialized = true;
      this._device = device; this._targets = targets; this._targetLifetime = new AbortController();
      this._nodes = {nodeFrame: {updateNode(fn) { fn(); }, updateBeforeNode(fn) { fn(); }, updateAfterNode(fn) { fn(); }}};
      this._backend = {}; this.info = {calls: 0, compute: {calls: 0, frameCalls: 0}}; this.queue = [];
    }
    _enqueue(request) { this.queue.push(request); }
    produce(target, value, depth = 0) {
      const entry = targets.capture(target), issue = () => targets.render(entry, (frame, texture) => {
        texture.value = value; if (frame.depthView) frame.depthView.nativeTexture.value = depth; events.push(['render', value, depth]);
      });
      if (!this._preparing && this._initialized) return issue();
      return new Promise((resolve, reject) => this._enqueue({issue, resolve, reject}));
    }
    async flush() {
      while (this.queue.length) { const request = this.queue.shift();
        try { request.resolve(await request.issue()); } catch (error) { request.reject(error); }
      }
    }
    async waitForGPU() { await this.flush(); }
    dispose() { this._disposed = true; this._targetLifetime.abort(new Error('renderer disposed')); targets.dispose(); }
  }
  const Renderer = modules.get('./three_compute.mjs').namespace.withThreeCompute(Base, {...THREE, WGSLNodeBuilder: Builder});
  const renderer = new Renderer();
  const adapter = modules.get('./three_compute_targets.mjs').namespace.createThreeComputeTargets({targets, device, ...adapterOptions});
  return {renderer, targets, device, adapter, storages, events, lost, scopeErrors, scopePromises,
    get builds() { return builds; }, failSampler(error) { samplerFailure = error; }};
}
export const dispatches = r => r.events.filter(e => e[0] === 'dispatch');
