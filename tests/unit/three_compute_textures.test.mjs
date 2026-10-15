import test from 'node:test';
import assert from 'node:assert/strict';
import {withThreeCompute} from '../../tools/ingest/three_compute.mjs';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {THREE, gpu, deferred} from './storage_texture_fixture.mjs';

const write = texture => ({texture, isSampler: true, isSampledTexture: true, store: true, access: 'writeOnly', mipLevel: 0});
const sample = texture => ({texture, isSampler: true, isSampledTexture: true});
const node = (bindings, extra = {}) => Object.assign(new EventTarget(), {
  isComputeNode: true, workgroupSize: [1], count: 4, version: 1, name: 'texture-test', bindings,
  dispose() { this.dispatchEvent(new Event('dispose')); }, ...extra,
});
function setup(options = {}) {
  const device = gpu(options), builders = [];
  class Builder {
    constructor(object, renderer) { assert.equal(object, null); this.renderer = renderer; builders.push(this); }
    build() {
      for (const binding of this.compute.bindings) if (binding.isSampledTexture) {
        const texture = binding.textureNode?.value ?? binding.texture;
        assert.equal(this.renderer.backend.utils.getTextureSampleData(texture).primarySamples, 1);
      }
      this.computeShader = `// types ${this.compute.bindings.map(b => (b.textureNode?.value ?? b.texture)?.type).join(',')}\n@compute @workgroup_size(1) fn main() {}`;
      this.updateBeforeNodes = this.compute.before ?? [];
      this.updateNodes = this.compute.updates ?? [];
      this.updateAfterNodes = this.compute.after ?? [];
    }
    getBindings() { return [{bindings: this.compute.bindings}]; }
  }
  class Base {
    constructor() {
      this.isF3DRenderer = this.isWebGPURenderer = true;
      this._device = device; this._backend = {device};
      this._initialized = !options.initialization; this._disposed = false;
      this._pending = []; this._drain = this._preparing = null;
      this._targetLifetime = new AbortController();
      this._targets = {bindingsFor: (destination, retained = new Map()) => retained};
      this._nodes = {nodeFrame: {renderId: 0,
        updateBeforeNode(fn) { fn(this); }, updateNode(fn) { fn(this); }, updateAfterNode(fn) { fn(this); }}};
      this.info = {calls: 0, frame: 0, compute: {calls: 0, frameCalls: 0}};
    }
    get backend() { return this._backend; }
    async init() { if (options.initialization) await options.initialization.promise; this._initialized = true; }
    _enqueue(request) {
      this._pending.push(request);
      if (!this._drain) this._drain = (async () => {
        await this.init();
        while (this._pending.length && !this._disposed) {
          const item = this._pending[0];
          if (item.prepare) await item.prepare;
          try { item.resolve(item.issue()); } catch (error) { item.reject(error); }
          this._pending.shift();
        }
      })().finally(() => { this._drain = null; });
    }
    renderTexture(texture) {
      const issue = () => { const binding = this._targets.bindingsFor(null).get(texture); device.calls.push(['render', binding.view]); return binding; };
      return this._drain ? new Promise((resolve,reject) => this._enqueue({issue,resolve,reject})) : issue();
    }
    async waitForGPU() { while (this._drain) await this._drain; }
    dispose() {
      this._disposed = true; this._targetLifetime.abort(new Error('renderer stopped'));
      for (const request of this._pending) request.reject(new Error('renderer stopped'));
      this._pending = [];
    }
  }
  const Renderer = withThreeCompute(Base, {...THREE, WGSLNodeBuilder: Builder, TSL: {context: () => ({isContextNode: true})}});
  return {renderer: new Renderer(), device, builders};
}
function block(renderer) {
  const gate = deferred();
  const done = new Promise((resolve,reject) => renderer._enqueue({prepare: gate.promise, issue() {},resolve,reject}));
  return {...gate, done};
}
const groups = device => device.calls.filter(c => c[0] === 'group').map(c => c[2]);

test('public compute writes a StorageTexture sampled by ordinary renderer bindings without copying', async () => {
  const {renderer,device} = setup(); const texture = new THREE.Texture();
  renderer.compute(node([write(texture)]));
  const binding = await renderer.renderTexture(texture), produced = groups(device)[0].entries[0].resource;
  assert.equal(binding.view.texture, produced.texture);
  assert.equal(device.textures.length, 1); assert.equal(device.buffers.length, 0);
  assert.deepEqual(device.calls.filter(c => ['dispatch','mip','render'].includes(c[0])).map(c => c[0]), ['dispatch','mip','mip','render']);
  await renderer.waitForGPU(); renderer.dispose();
});

test('queued ping-pong texture selections retain each call rather than the last NodeSampledTexture value', async () => {
  const {renderer,device,builders} = setup(), a = new THREE.Texture(), b = new THREE.Texture();
  a.generateMipmaps = b.generateMipmaps = false;
  const binding = {...write(a), textureNode: {value: a}, update() { this.texture = this.textureNode.value; }};
  const kernel = node([binding]), gate = block(renderer);
  const first = renderer.computeAsync(kernel); binding.textureNode.value = b;
  const second = renderer.computeAsync(kernel); const draw = renderer.renderTexture(b);
  assert.equal(device.textures.length, 0);
  gate.resolve(); await Promise.all([first,second,draw,gate.done]);
  assert.equal(groups(device)[0].entries[0].resource.texture, device.textures[0]);
  assert.equal(groups(device)[1].entries[0].resource.texture, device.textures[1]);
  assert.equal(device.calls.find(c => c[0] === 'render')[1].texture, device.textures[1]);
  assert.equal(builders.length, 1); renderer.dispose();
});

test('texture samplers and dispatch sizes are captured while earlier rendering prepares', async () => {
  const {renderer,device} = setup(), texture = new THREE.Texture(); texture.generateMipmaps = false;
  const kernel = node([{isSampler: true,texture}, sample(texture)]), gate = block(renderer);
  const first = renderer.computeAsync(kernel, [1,2]); texture.magFilter = THREE.NearestFilter;
  const second = renderer.computeAsync(kernel, [2,1]); gate.resolve(); await Promise.all([first,second]);
  const bound = groups(device);
  assert.equal(bound[0].entries[0].resource.descriptor.magFilter, 'linear');
  assert.equal(bound[1].entries[0].resource.descriptor.magFilter, 'nearest');
  assert.deepEqual(device.calls.filter(c => c[0] === 'dispatch'), [['dispatch',1,2,1],['dispatch',2,1,1]]);
  renderer.dispose();
});

test('write-to-sample dependency in separate kernels uses the same native allocation', () => {
  const {renderer,device} = setup(), texture = new THREE.Texture(); texture.generateMipmaps = false;
  renderer.compute([node([write(texture)]), node([sample(texture)])]);
  assert.equal(groups(device)[0].entries[0].resource.texture, groups(device)[1].entries[0].resource.texture);
  assert.equal(device.textures.length, 1); renderer.dispose();
});

test('texture precompilation creates reusable native layouts without dispatching or generating mips', async () => {
  const {renderer,device} = setup(), texture = new THREE.Texture(), kernel = node([write(texture)]);
  await renderer.compileComputeAsync(kernel);
  assert.equal(device.calls.some(c => ['compute','mip'].includes(c[0])), false);
  const count = device.calls.filter(c => c[0] === 'layout').length;
  renderer.compute(kernel);
  const descriptor = device.calls.find(c => c[0] === 'pipeline')[1].descriptor;
  assert.equal(descriptor.layout.bindGroupLayouts[0].entries[0].storageTexture.format,'rgba8unorm');
  assert.equal(device.calls.filter(c => c[0] === 'layout').length, count + 1); // One mip-generation layout only.
  await renderer.waitForGPU(); renderer.dispose();
});

test('float32 textureLoad input uses an explicit unfilterable layout with mixed storage buffers', () => {
  const {renderer,device} = setup(), texture = new THREE.Texture(); texture.type = THREE.FloatType;
  const data = {isStorageBufferAttribute: true, itemSize: 1, count: 4, array: new Float32Array(4), version:0, updateRanges:[]};
  renderer.compute(node([sample(texture), {isStorageBuffer:true,attribute:data,access:'readOnly'}]));
  const descriptor = device.calls.find(c=>c[0]==='pipeline')[1].descriptor;
  assert.equal(descriptor.layout.bindGroupLayouts[0].entries[0].texture.sampleType, 'unfilterable-float');
  assert.equal(descriptor.layout.bindGroupLayouts[0].entries[1].buffer.type, 'read-only-storage');
  assert.equal(device.textures[0].descriptor.format, 'rgba32float'); renderer.dispose();
});

test('changing a texture format rebuilds WGSL/layouts while queued generations stay distinct', async () => {
  const {renderer,device,builders} = setup(), a = new THREE.Texture(), b = new THREE.Texture();
  b.type = THREE.HalfFloatType; a.generateMipmaps = b.generateMipmaps = false;
  const binding = {...write(a), textureNode:{value:a}}, kernel = node([binding]), gate=block(renderer);
  const first=renderer.computeAsync(kernel); binding.textureNode.value=b; const second=renderer.computeAsync(kernel);
  gate.resolve(); await Promise.all([first,second]);
  const formats=device.calls.filter(c=>c[0]==='pipeline').map(c=>c[1].descriptor.layout.bindGroupLayouts[0].entries[0].storageTexture.format);
  assert.deepEqual(formats,['rgba8unorm','rgba16float']); assert.equal(builders.length,2); renderer.dispose();
});

test('texture disposal rejects queued dispatch before allocation, not after a misleading success', async () => {
  const {renderer,device} = setup(), texture = new THREE.Texture(), gate = block(renderer);
  const task=renderer.computeAsync(node([write(texture)])); texture.dispose(); gate.resolve();
  await assert.rejects(task,/STALE/); assert.equal(device.textures.length,0); renderer.dispose();
});

test('read/write texture feedback fails before uploads or compute submission', () => {
  const {renderer,device} = setup(), texture = new THREE.Texture();
  assert.throws(()=>renderer.compute(node([sample(texture),write(texture)])),/FEEDBACK/);
  assert.equal(device.textures.length,0); assert.equal(device.calls.some(c=>c[0]==='dispatch'),false); renderer.dispose();
});

test('initialization ordering keeps compute, sampling and subsequent compute in their source order', async () => {
  const initialization=deferred(), {renderer,device}=setup({initialization}), texture=new THREE.Texture(); texture.generateMipmaps=false;
  const a=renderer.computeAsync(node([write(texture)])), b=renderer.renderTexture(texture), c=renderer.computeAsync(node([sample(texture)]));
  initialization.resolve(); await Promise.all([a,b,c]);
  assert.deepEqual(device.calls.filter(c=>['dispatch','render'].includes(c[0])).map(c=>c[0]),['dispatch','render','dispatch']); renderer.dispose();
});

test('texture validation failures propagate through renderer waitForGPU', async () => {
  const {renderer,device}=setup(), texture=new THREE.Texture(); device.validationError('texture allocation failed');
  renderer.compute(node([write(texture)])); await assert.rejects(renderer.waitForGPU(),/texture allocation failed/);
  assert.equal(renderer._storageTextures.allocatedBytes,0); renderer.dispose();
});

test('explicit GPU pipeline layouts are snapshotted, keyed and shared between sync/async preparation', async () => {
  const device=gpu(), compute=createGpuCompute(device), code='@compute @workgroup_size(1) fn main() {}';
  const layouts=[{entries:[{binding:0,visibility:4,texture:{sampleType:'unfilterable-float'}}]}];
  const pending=compute.prepare({code,bindGroupLayouts:layouts}); layouts[0].entries[0].texture.sampleType='float';
  const original=await pending;
  assert.equal(original.descriptor.layout.bindGroupLayouts[0].entries[0].texture.sampleType,'unfilterable-float');
  const changed=compute.prepareSync({code,bindGroupLayouts:layouts}); assert.notEqual(original,changed);
  assert.equal(await compute.prepare({code,bindGroupLayouts:layouts}),changed);
  assert.throws(()=>compute.prepareSync({code,bindGroupLayouts:[{},{}]}),/LAYOUT/); compute.dispose();
});
