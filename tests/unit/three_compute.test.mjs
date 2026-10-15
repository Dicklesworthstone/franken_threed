import test from 'node:test';
import assert from 'node:assert/strict';
import {withThreeCompute} from '../../tools/ingest/three_compute.mjs';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
};
// This is a submission/resource test double, not a WGSL interpreter or GPU
// conformance test. Its kernel callback observes the buffers bound at submit.
function device({mapping = null, compilation = null} = {}) {
  const calls = [], buffers = [];
  const pipeline = descriptor => ({code: descriptor.compute.module.code, getBindGroupLayout: index => index});
  return {
    calls, buffers, limits: {maxComputeWorkgroupsPerDimension: 65535}, lost: new Promise(() => {}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createShaderModule: ({code}) => ({code, getCompilationInfo: async () => {
      if (compilation) await compilation.promise;
      return {messages: []};
    }}),
    createComputePipeline: descriptor => { calls.push(['pipeline']); return pipeline(descriptor); },
    createComputePipelineAsync: async descriptor => { calls.push(['pipelineAsync']); return pipeline(descriptor); },
    createBindGroup: descriptor => descriptor,
    createBuffer({size}) {
      const buffer = {data: new Uint8Array(size), destroyed: 0, destroy() { this.destroyed++; }, unmap() {},
        mapAsync: () => mapping?.promise ?? Promise.resolve(), getMappedRange() { return this.data.buffer; }};
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() {
      const commands = [];
      return {
        beginComputePass() {
          let current, groups = [];
          const dispatch = (...counts) => commands.push(() => {
            calls.push(['dispatch', ...counts]);
            const entries = groups[0]?.entries ?? [];
            calls.push(['bound', entries.map(e => e.resource.buffer)]);
            // Stand-in GPU write: lets tests prove that later host uploads do
            // not erase producer results, without claiming WGSL execution.
            if (current.code.startsWith('// add') && entries.length) {
              const array = new Float32Array(entries[0].resource.buffer.data.buffer);
              const amount = entries[1] ? new Float32Array(entries[1].resource.buffer.data.buffer)[0] : 1;
              for (let i = 0; i < array.length; i++) array[i] += amount;
            }
          });
          return {setPipeline: value => { current = value; }, setBindGroup: (i, value) => { groups[i] = value; },
            dispatchWorkgroups: dispatch,
            dispatchWorkgroupsIndirect: (buffer, offset) => { calls.push(['indirect', buffer, offset]); dispatch(...new Uint32Array(buffer.data.buffer, offset, 3)); }, end() {}};
        },
        copyBufferToBuffer(source, from, target, to, size) {
          commands.push(() => { calls.push(['copy']); target.data.set(source.data.subarray(from, from + size), to); });
        }, finish: () => commands,
      };
    },
    queue: {
      writeBuffer(buffer, offset, source, start = 0, size = source.byteLength - start) {
        calls.push(['upload']); buffer.data.set(new Uint8Array(source.buffer, source.byteOffset + start, size), offset);
      },
      submit(commands) { calls.push(['submit']); for (const batch of commands) for (const execute of batch) execute(); },
      onSubmittedWorkDone: async () => {},
    },
  };
}
function attribute(array, itemSize = 1) {
  return {isStorageBufferAttribute: true, array, itemSize, count: array.length / itemSize, version: 0,
    updateRanges: [], uploads: 0, clearUpdateRanges() { this.updateRanges = []; }, onUploadCallback() { this.uploads++; }};
}
function node(bindings = [], options = {}) {
  const listeners = new Set();
  return {isComputeNode: true, name: 'add', count: 4, dispatchSize: null, workgroupSize: [2], version: 1, bindings,
    addEventListener(type, listener) { if (type === 'dispose') listeners.add(listener); },
    removeEventListener(type, listener) { listeners.delete(listener); },
    dispose() { for (const listener of [...listeners]) listener(); },
    get listenerCount() { return listeners.size; }, ...options};
}
const storage = attribute => ({isStorageBuffer: true, attribute});
function setup(options = {}) {
  const gpu = device(options), initialization = options.initialization;
  const updates = [], builders = [];
  class WGSLNodeBuilder {
    constructor(object, renderer) { assert.equal(object, null); this.renderer = renderer; builders.push(this); }
    build() {
      assert.equal(this.renderer.contextNode.isContextNode, true);
      const source = this.compute;
      this.computeShader = `// ${source.name}\n@compute @workgroup_size(2) fn main() {}`;
      this.updateBeforeNodes = source.before ?? [];
      this.updateNodes = source.updates ?? [];
      this.updateAfterNodes = source.after ?? [];
    }
    getBindings() { return [{bindings: this.compute.bindings}]; }
  }
  class Base {
    constructor() {
      this.isF3DRenderer = this.isWebGPURenderer = true;
      this._device = gpu; this._initialized = !initialization; this._disposed = false;
      this._pending = []; this._drain = this._preparing = null;
      this._targetLifetime = new AbortController();
      this._nodes = {nodeFrame: {renderId: 17, compute: null, renderer: null,
        updateBeforeNode(value) { updates.push('before'); value(this); },
        updateNode(value) { updates.push('current'); value(this); },
        updateAfterNode(value) { updates.push('after'); value(this); }}};
      this.info = {calls: 0, frame: 0, compute: {calls: 0, frameCalls: 0}};
      this.inspector = {beginCompute: () => updates.push('begin'), finishCompute: () => updates.push('finish')};
    }
    async init() { if (initialization) await initialization.promise; this._initialized = true; }
    // Matches the source renderer's transfer FIFO: issue synchronously, resolve
    // its result, then continue without waiting for readback mapping.
    _enqueue(request) {
      this._pending.push(request);
      if (!this._drain) {
        this._drain = (async () => {
          await this.init();
          while (this._pending.length && !this._disposed) {
            const item = this._pending.shift();
            if (item.prepare) await item.prepare;
            try { item.resolve(item.issue()); } catch (error) { item.reject(error); }
          }
        })().finally(() => { this._drain = null; });
      }
    }
    async waitForGPU() { while (this._drain) await this._drain; }
    dispose() {
      this._disposed = true; this._targetLifetime.abort(new Error('renderer stopped'));
      for (const item of this._pending) item.reject(new Error('renderer stopped'));
      this._pending = [];
    }
  }
  const THREE = {REVISION: '186', WGSLNodeBuilder, TSL: {context: () => ({isContextNode: true})}};
  const Renderer = withThreeCompute(Base, THREE), renderer = new Renderer();
  return {renderer, gpu, updates, builders, THREE};
}

test('ordinary compute nodes share GPU storage and do not re-upload stale CPU values', async () => {
  const {renderer, gpu} = setup(), data = attribute(new Float32Array([1, 2, 3, 4]));
  const first = node([storage(data)]), second = node([storage(data)]);
  assert.equal(renderer.compute([first, second]), undefined);
  const result = await renderer.getArrayBufferAsync(data);
  assert.deepEqual([...new Float32Array(result)], [3, 4, 5, 6]);
  assert.deepEqual([...data.array], [1, 2, 3, 4]);
  assert.equal(data.uploads, 1);
  const bound = gpu.calls.filter(c => c[0] === 'bound');
  assert.equal(bound[0][1][0], bound[1][1][0]);
  assert.equal(renderer.info.compute.calls, 1); renderer.dispose();
});

test('per-dispatch source updates produce distinct uniform versions in kernel order', async () => {
  const {renderer, updates} = setup(), data = attribute(new Float32Array([0]));
  let value = 2;
  const uniform = {isUniformBuffer: true, buffer: new Float32Array(4), update() { this.buffer[0] = value; }};
  const first = node([storage(data), uniform], {after: [() => { value = 7; }]}), second = node([storage(data), uniform]);
  renderer.compute([first, second]);
  assert.deepEqual([...new Float32Array(await renderer.getArrayBufferAsync(data))], [9]);
  assert.deepEqual(updates, ['begin', 'after', 'finish']); renderer.dispose();
});

test('compute/readback/compute keep FIFO ordering during initialization without waiting for maps', async () => {
  const initialization = deferred(), mapping = deferred(), {renderer, gpu} = setup({initialization, mapping});
  const data = attribute(new Float32Array([0])), kernel = node([storage(data)]);
  const first = renderer.computeAsync(kernel), read = renderer.getArrayBufferAsync(data), second = renderer.computeAsync(kernel);
  assert.equal(gpu.calls.length, 0); initialization.resolve();
  await second; await first;
  assert.deepEqual(gpu.calls.filter(c => ['dispatch', 'copy'].includes(c[0])).map(c => c[0]), ['dispatch', 'copy', 'dispatch']);
  mapping.resolve();
  assert.deepEqual([...new Float32Array(await read)], [1]);
  assert.deepEqual([...new Float32Array(await renderer.getArrayBufferAsync(data))], [2]); renderer.dispose();
});

test('compute queues behind a pending render preparation and before a later render', async () => {
  const preparation = deferred(), {renderer, gpu} = setup();
  const earlier = new Promise((resolve, reject) => renderer._enqueue({prepare: preparation.promise, resolve, reject, issue() {}}));
  const data = attribute(new Float32Array([0])), run = renderer.computeAsync(node([storage(data)]));
  let seen;
  const draw = new Promise((resolve, reject) => renderer._enqueue({resolve, reject, issue() {
    seen = new Float32Array(gpu.buffers[0].data.buffer)[0]; gpu.calls.push(['render']);
  }}));
  preparation.resolve(); await earlier; await run; await draw;
  assert.equal(seen, 1); renderer.dispose();
});

test('compileComputeAsync prewarms source programs without executing a kernel', async () => {
  const {renderer, gpu, builders} = setup(), data = attribute(new Float32Array([0])), kernel = node([storage(data)]);
  let initialized = 0; kernel.onInitFunction = function ({renderer: target}) { assert.equal(this, kernel); assert.equal(target, renderer); initialized++; };
  const progress = [];
  await renderer.compileComputeAsync(kernel, event => progress.push([event.loaded, event.total]));
  assert.equal(gpu.calls.some(c => c[0] === 'dispatch'), false);
  renderer.compute(kernel); renderer.compute(kernel);
  assert.equal(builders.length, 1); assert.equal(initialized, 1);
  assert.equal(gpu.calls.filter(c => c[0] === 'pipelineAsync').length, 1);
  assert.equal(gpu.calls.filter(c => c[0] === 'pipeline').length, 0);
  assert.deepEqual(progress, [[1, 1]]); renderer.dispose();
});

test('source frame callbacks see the compute renderer and previous frame context is restored', () => {
  const {renderer, updates} = setup(), frame = renderer._nodes.nodeFrame;
  let kernel;
  const check = state => { assert.equal(state.renderer, renderer); assert.equal(state.compute, kernel); assert.equal(state.renderId, 1); };
  kernel = node([], {before: [check], updates: [check], after: [check]});
  renderer.compute(kernel);
  assert.deepEqual(updates, ['begin', 'before', 'current', 'after', 'finish']);
  assert.equal(frame.renderId, 17); assert.equal(frame.compute, null); assert.equal(frame.renderer, null); renderer.dispose();
});

test('source frame and inspector are restored after a callback throws', () => {
  const {renderer, updates} = setup(), frame = renderer._nodes.nodeFrame;
  assert.throws(() => renderer.compute(node([], {updates: [() => { throw new Error('callback failed'); }]})), /callback failed/);
  assert.equal(frame.renderId, 17); assert.equal(frame.compute, null); assert.equal(updates.at(-1), 'finish'); renderer.dispose();
});

test('scalar counts, explicit grids, device-limit tiling and indirect GPU counts dispatch correctly', () => {
  const {renderer, gpu} = setup(), kernel = node([], {count: 9, name: 'noop'});
  renderer.compute(kernel); renderer.compute(kernel, [3, 2, 1]); renderer.compute(kernel, 65536 * 2);
  const indirect = attribute(new Uint32Array([6, 2, 1])); indirect.isIndirectStorageBufferAttribute = true;
  renderer.compute(kernel, indirect);
  assert.deepEqual(gpu.calls.filter(c => c[0] === 'dispatch'), [
    ['dispatch', 5, 1, 1], ['dispatch', 3, 2, 1], ['dispatch', 65535, 2, 1], ['dispatch', 6, 2, 1]]);
  assert.equal(gpu.calls.filter(c => c[0] === 'indirect').length, 1); renderer.dispose();
});

test('vec3 storage acquires r186-compatible padded layout and integer storage is widened', async () => {
  const {renderer} = setup(), data = attribute(new Float32Array([1, 2, 3, 4, 5, 6]), 3);
  const integers = attribute(new Uint8Array([9, 11]));
  renderer.compute(node([storage(data), storage(integers)], {name: 'noop'}));
  assert.equal(data.itemSize, 4); assert.equal(data.count, 2);
  assert.deepEqual([...new Float32Array(await renderer.getArrayBufferAsync(data))], [1, 2, 3, 0, 4, 5, 6, 0]);
  assert.equal(integers.array.constructor, Uint32Array);
  assert.deepEqual([...new Uint32Array(await renderer.getArrayBufferAsync(integers))], [9, 11]); renderer.dispose();
});

test('versioned partial CPU updates preserve producer values outside dirty ranges', async () => {
  const {renderer} = setup(), data = attribute(new Float32Array([0, 10])), kernel = node([storage(data)]);
  renderer.compute(kernel);
  data.array[1] = 30; data.version++; data.updateRanges.push({start: 1, count: 1});
  renderer.compute(kernel);
  assert.deepEqual([...new Float32Array(await renderer.getArrayBufferAsync(data))], [2, 31]);
  assert.equal(data.updateRanges.length, 0); assert.equal(data.uploads, 2); renderer.dispose();
});

test('node disposal permits recompilation without destroying storage shared with other kernels', async () => {
  const {renderer, builders} = setup(), data = attribute(new Float32Array([0])), first = node([storage(data)]), second = node([storage(data)]);
  renderer.compute(first); first.dispose(); renderer.compute(second); renderer.compute(first);
  assert.equal(builders.length, 3);
  assert.deepEqual([...new Float32Array(await renderer.getArrayBufferAsync(data))], [3]);
  renderer.dispose(); assert.equal(first.listenerCount, 0); assert.equal(second.listenerCount, 0);
});

test('version and workgroup-size changes invalidate source program preparation', () => {
  const {renderer, builders} = setup(), kernel = node();
  renderer.compute(kernel); kernel.version++; renderer.compute(kernel);
  kernel.workgroupSize = [4]; renderer.compute(kernel);
  assert.equal(builders.length, 3); renderer.dispose();
});

test('unsupported bindings, invalid nodes and invalid workgroup sizes fail explicitly', async () => {
  const {renderer, gpu} = setup(), data = attribute(new Float32Array([0]));
  assert.throws(() => renderer.compute([node(), {}]), /SOURCE/);
  assert.throws(() => renderer.compute([]), /SOURCE/);
  assert.throws(() => renderer.compute(node([storage(data), {isSampledTexture: true}])), /BINDING/);
  assert.equal(gpu.buffers.length, 0);
  assert.throws(() => renderer.compute(node([], {workgroupSize: [257]})), /DISPATCH/);
  assert.throws(() => renderer.compute(node([], {workgroupSize: [0]})), /DISPATCH/);
  await assert.rejects(renderer.getArrayBufferAsync(data), /has not been uploaded/);
  await assert.rejects(renderer.compileComputeAsync(node(), 1), /OPTIONS/); renderer.dispose();
});

test('disposing during async compilation rejects rather than publishing a stale program', async () => {
  const compilation = deferred(), {renderer} = setup({compilation}), kernel = node();
  const compile = renderer.compileComputeAsync(kernel);
  renderer.dispose(); await assert.rejects(compile, /DISPOSED/);
  compilation.resolve(); assert.equal(kernel.listenerCount, 0);
});

test('ignored queued compute failures are surfaced on the next public compute call', async () => {
  const initialization = deferred(), {renderer} = setup({initialization});
  renderer.compute(node([{isSampledTexture: true}])); initialization.resolve();
  await renderer.waitForGPU(); await Promise.resolve();
  assert.throws(() => renderer.compute(node()), /BINDING/); renderer.dispose();
});

test('the WebGL surface does not pretend to implement WebGPU-only compute', () => {
  const {renderer} = setup(); renderer.isWebGPURenderer = false;
  assert.throws(() => renderer.compute(node()), /ROUTE/); renderer.dispose();
});
