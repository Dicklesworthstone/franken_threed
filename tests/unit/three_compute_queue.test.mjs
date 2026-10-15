import test from 'node:test';
import assert from 'node:assert/strict';
import {withThreeCompute} from '../../tools/ingest/three_compute.mjs';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
};
// Records submission-time buffer contents; this is not a WGSL interpreter.
function setup({initialization = null, compilation = null, mapping = null} = {}) {
  const events = [], buffers = [], builders = [];
  const pipeline = descriptor => ({code: descriptor.compute.module.code, getBindGroupLayout: index => index});
  const device = {
    limits: {maxComputeWorkgroupsPerDimension: 65535}, lost: new Promise(() => {}),
    pushErrorScope() {}, popErrorScope: async () => null,
    createShaderModule: ({code}) => ({code, getCompilationInfo: async () => {
      if (compilation) await compilation.promise;
      return {messages: []};
    }}),
    createComputePipeline: descriptor => { events.push(['compile']); return pipeline(descriptor); },
    createComputePipelineAsync: async descriptor => { events.push(['compileAsync']); return pipeline(descriptor); },
    createBindGroup: descriptor => descriptor,
    createBuffer({size, usage}) {
      const buffer = {size, usage, data: new Uint8Array(size), destroyed: 0, destroy() { this.destroyed++; }, unmap() {},
        mapAsync: () => mapping?.promise ?? Promise.resolve(), getMappedRange() { return this.data.buffer; }};
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() {
      const commands = [];
      return {
        beginComputePass() {
          let current; const groups = [];
          const dispatch = counts => commands.push(() => {
            const dims = counts();
            events.push(['dispatch', current.code, ...dims]);
            const entries = groups[0]?.entries ?? [];
            if (current.code === 'add' && entries.length && dims.every(n => n > 0)) {
              const array = new Float32Array(entries[0].resource.buffer.data.buffer);
              const amount = entries[1] ? new Float32Array(entries[1].resource.buffer.data.buffer)[0] : 1;
              for (let i = 0; i < array.length; i++) array[i] += amount;
            }
          });
          return {setPipeline: value => { current = value; }, setBindGroup: (i, value) => { groups[i] = value; },
            dispatchWorkgroups: (...counts) => dispatch(() => counts),
            dispatchWorkgroupsIndirect: (buffer, offset) => dispatch(() => [...new Uint32Array(buffer.data.buffer, offset, 3)]), end() {}};
        },
        copyBufferToBuffer(source, from, target, to, size) {
          commands.push(() => { events.push(['read']); target.data.set(source.data.subarray(from, from + size), to); });
        },
        copyBufferToTexture(source, target, size) {
          commands.push(() => {
            events.push(['texture-copy']);
            for (let row = 0; row < size[1]; row++) {
              const start = source.offset + row * (source.bytesPerRow ?? size[0] * 4);
              const offset = ((target.origin[1] + row) * target.texture.width + target.origin[0]) * 4;
              target.texture.data.set(source.buffer.data.subarray(start, start + size[0] * 4), offset);
            }
          });
        }, finish: () => commands,
      };
    },
    queue: {
      writeBuffer(buffer, offset, source, start = 0, size = source.byteLength - start) {
        events.push(['upload']); buffer.data.set(new Uint8Array(source.buffer, source.byteOffset + start, size), offset);
      },
      submit(commands) { for (const batch of commands) for (const execute of batch) execute(); },
      onSubmittedWorkDone: async () => {},
    },
  };
  class WGSLNodeBuilder {
    constructor(object, renderer) { assert.equal(object, null); this.renderer = renderer; builders.push(this); }
    build() {
      this.computeShader = this.compute.name;
      this.updateBeforeNodes = this.compute.before ?? [];
      this.updateNodes = this.compute.updates ?? [];
      this.updateAfterNodes = this.compute.after ?? [];
    }
    getBindings() { return [{bindings: this.compute.bindings}]; }
  }
  class Base {
    constructor() {
      this.isF3DRenderer = this.isWebGPURenderer = true;
      this._device = device; this._initialized = !initialization; this._disposed = false;
      this._pending = []; this._drain = this._preparing = null;
      this._targetLifetime = new AbortController();
      this._nodes = {nodeFrame: {renderId: 17, compute: null, renderer: null,
        updateBeforeNode(value) { value(this); }, updateNode(value) { value(this); }, updateAfterNode(value) { value(this); }}};
      this.info = {calls: 0, frame: 0, compute: {calls: 0, frameCalls: 0}};
      this.inspector = {beginCompute() {}, finishCompute() {}};
    }
    async init() { if (initialization) await initialization.promise; this._initialized = true; }
    _enqueue(request) {
      if (this._pending.length >= 64) throw new Error('queue full');
      this._pending.push(request);
      if (!this._drain) {
        this._drain = (async () => {
          await this.init();
          while (this._pending.length && !this._disposed) {
            const item = this._pending[0];
            if (item.prepare) await item.prepare;
            if (this._disposed) return;
            try { item.resolve(item.issue()); } catch (error) { item.reject(error); }
            this._pending.shift();
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
  const Renderer = withThreeCompute(Base, {REVISION: '186', WGSLNodeBuilder, TSL: {context: () => ({isContextNode: true})}});
  const renderer = new Renderer();
  function block() {
    const gate = deferred();
    const done = new Promise((resolve, reject) => renderer._enqueue({prepare: gate.promise, issue() {}, resolve, reject}));
    done.catch(() => {});
    return {open: gate.resolve, done};
  }
  return {renderer, device, events, buffers, builders, block};
}
function attribute(values, itemSize = 1) {
  return {isStorageBufferAttribute: true, array: values, itemSize, count: values.length / itemSize,
    version: 0, updateRanges: [], uploads: 0, clearUpdateRanges() { this.updateRanges = []; }, onUploadCallback() { this.uploads++; }};
}
function node(bindings = [], options = {}) {
  const listeners = new Set();
  return {isComputeNode: true, name: 'add', count: 4, dispatchSize: null, workgroupSize: [2], version: 1, bindings,
    addEventListener(type, listener) { listeners.add(listener); }, removeEventListener(type, listener) { listeners.delete(listener); },
    dispose() { for (const listener of [...listeners]) listener(); }, ...options};
}
const storage = attribute => ({isStorageBuffer: true, attribute});
const read = async (renderer, data) => [...new Float32Array(await renderer.getArrayBufferAsync(data))];

// The first three regressions fail against b6a5d7b's call-at-drain adapter.
test('queued kernels preserve distinct call-time uniforms, even when prewarmed', async () => {
  const {renderer, block} = setup(), data = attribute(new Float32Array([0]));
  let value = 2, updates = 0;
  const uniform = {isUniformBuffer: true, buffer: new Float32Array(4), update() { this.buffer[0] = value; }};
  const kernel = node([storage(data), uniform], {updates: [() => updates++]});
  await renderer.compileComputeAsync(kernel);
  const gate = block(), first = renderer.computeAsync(kernel);
  const middle = renderer.getArrayBufferAsync(data);
  value = 7;
  const second = renderer.computeAsync(kernel);
  value = 99;
  assert.equal(updates, 3); // compile + two source calls; never replay at drain.
  gate.open(); await Promise.all([gate.done, first, second]);
  assert.deepEqual([...new Float32Array(await middle)], [2]);
  assert.deepEqual(await read(renderer, data), [9]);
  assert.equal(updates, 3); renderer.dispose();
});

test('queued CPU versions and dirty ranges preserve intermediate simulation state', async () => {
  const {renderer, block} = setup(), data = attribute(new Float32Array([0, 10])), kernel = node([storage(data)]);
  renderer.compute(kernel); // GPU now [1, 11], CPU still [0, 10].
  const gate = block();
  data.array[1] = 30; data.version++; data.updateRanges = [{start: 1, count: 1}];
  const first = renderer.computeAsync(kernel), middle = renderer.getArrayBufferAsync(data);
  data.array[0] = 100; data.version++; data.updateRanges = [{start: 0, count: 1}];
  const second = renderer.computeAsync(kernel);
  data.array.fill(-500); // Unversioned changes cannot contaminate either upload.
  gate.open(); await Promise.all([first, second, gate.done]);
  assert.deepEqual([...new Float32Array(await middle)], [2, 31]);
  assert.deepEqual(await read(renderer, data), [101, 32]);
  assert.equal(data.uploads, 3); renderer.dispose();
});

test('queued dispatch counts and shader generations are captured rather than read at drain', async () => {
  const {renderer, block, events} = setup(), kernel = node([], {name: 'first', count: 4});
  const gate = block(), a = renderer.computeAsync(kernel);
  kernel.count = 8; kernel.version++; kernel.name = 'second';
  const b = renderer.computeAsync(kernel);
  kernel.count = 20; kernel.name = 'too late';
  gate.open(); await Promise.all([a, b, gate.done]);
  assert.deepEqual(events.filter(e => e[0] === 'dispatch'), [['dispatch', 'first', 2, 1, 1], ['dispatch', 'second', 4, 1, 1]]);
  renderer.dispose();
});

test('queued after-update readback follows its own dispatch and precedes the next kernel', async () => {
  const {renderer, block, events} = setup(), data = attribute(new Float32Array([0]));
  const reads = [], kernel = node([storage(data)], {after: [() => reads.push(renderer.getArrayBufferAsync(data))]});
  const gate = block(), run = renderer.computeAsync([kernel, kernel]);
  assert.equal(reads.length, 2);
  gate.open(); await run;
  assert.deepEqual(await Promise.all(reads.map(async p => [...new Float32Array(await p)])), [[1], [2]]);
  assert.deepEqual(events.filter(e => ['dispatch', 'read'].includes(e[0])).map(e => e[0]), ['dispatch', 'read', 'dispatch', 'read']);
  renderer.dispose();
});

test('pre-initialization onInit kernels stay ahead of the already-queued readback', async () => {
  const initialization = deferred(), {renderer, events} = setup({initialization});
  const data = attribute(new Float32Array([0])), init = node([storage(data)]);
  const kernel = node([storage(data)], {onInitFunction: () => renderer.compute(init)});
  const run = renderer.computeAsync(kernel), result = renderer.getArrayBufferAsync(data);
  initialization.resolve(); await run;
  assert.deepEqual([...new Float32Array(await result)], [2]);
  assert.deepEqual(events.filter(e => ['dispatch', 'read'].includes(e[0])).map(e => e[0]), ['dispatch', 'dispatch', 'read']);
  renderer.dispose();
});

test('an after-update exception does not drop its already-queued kernel', async () => {
  const {renderer, block} = setup(), data = attribute(new Float32Array([0]));
  const gate = block();
  assert.throws(() => renderer.compute(node([storage(data)], {after: [() => { throw new Error('source failed'); }]})), /source failed/);
  gate.open(); await gate.done; await renderer.waitForGPU();
  assert.deepEqual(await read(renderer, data), [1]); renderer.dispose();
});

test('disposed and recreated nodes cannot impersonate an earlier queued generation', async () => {
  const {renderer, block, events} = setup(), kernel = node();
  const gate = block(), old = renderer.computeAsync(kernel);
  kernel.dispose();
  const current = renderer.computeAsync(kernel);
  gate.open(); await assert.rejects(old, /STALE/); await current;
  assert.equal(events.filter(e => e[0] === 'dispatch').length, 1); renderer.dispose();
});

test('zero workgroup dimensions and zero count remain zero; invalid grids do not truncate', () => {
  const {renderer, events} = setup(), kernel = node([], {count: 0, name: 'noop'});
  renderer.compute(kernel); renderer.compute(kernel, [2, 0, 1]); renderer.compute(kernel, [1, 1, 0]);
  assert.deepEqual(events.filter(e => e[0] === 'dispatch'), [
    ['dispatch', 'noop', 0, 1, 1], ['dispatch', 'noop', 2, 0, 1], ['dispatch', 'noop', 1, 1, 0]]);
  for (const counts of [[], [1, 2, 3, 4], [1, NaN], [-1], [Infinity], [65536]])
    assert.throws(() => renderer.compute(kernel, counts), /DISPATCH/);
  renderer.dispose();
});

test('unchanged queued storage versions never replace GPU-produced data', async () => {
  const {renderer, block, events} = setup(), data = attribute(new Float32Array([0])), kernel = node([storage(data)]);
  const gate = block();
  const a = renderer.computeAsync(kernel), b = renderer.computeAsync(kernel), c = renderer.computeAsync(kernel);
  data.array[0] = 999;
  gate.open(); await Promise.all([a, b, c]);
  assert.deepEqual(await read(renderer, data), [3]);
  assert.equal(data.uploads, 1); assert.equal(events.filter(e => e[0] === 'upload').length, 1);
  renderer.dispose();
});

test('readback mappings do not hold the submit FIFO', async () => {
  const mapping = deferred(), {renderer, block, events} = setup({mapping});
  const data = attribute(new Float32Array([0])), kernel = node([storage(data)]), gate = block();
  const first = renderer.computeAsync(kernel), result = renderer.getArrayBufferAsync(data), second = renderer.computeAsync(kernel);
  gate.open(); await Promise.all([first, second]);
  assert.deepEqual(events.filter(e => ['dispatch', 'read'].includes(e[0])).map(e => e[0]), ['dispatch', 'read', 'dispatch']);
  mapping.resolve(); assert.deepEqual([...new Float32Array(await result)], [1]); renderer.dispose();
});

test('waitForGPU also waits for outstanding asynchronous compilation', async () => {
  const compilation = deferred(), {renderer} = setup({compilation});
  const compile = renderer.compileComputeAsync(node());
  let complete = false;
  const idle = renderer.waitForGPU().then(() => { complete = true; });
  await Promise.resolve(); await Promise.resolve(); assert.equal(complete, false);
  compilation.resolve(); await compile; await idle; assert.equal(complete, true); renderer.dispose();
});

test('renderer disposal cancels queued snapshots without any GPU upload', async () => {
  const {renderer, block, events} = setup(), gate = block();
  const run = renderer.computeAsync(node([storage(attribute(new Float32Array([0])))]));
  renderer.dispose(); gate.open();
  await assert.rejects(run, /stopped/); assert.equal(events.some(e => e[0] === 'upload'), false);
});


test('preparation started by a before-update hook still snapshots the current call', async () => {
  const {renderer, block} = setup(), data = attribute(new Float32Array([0]));
  let gate, value = 3;
  const uniform = {isUniformBuffer: true, buffer: new Float32Array(4), update() { this.buffer[0] = value; }};
  const kernel = node([storage(data), uniform], {before: [() => { gate ??= block(); }]});
  const run = renderer.computeAsync(kernel);
  value = 99; uniform.buffer[0] = 99; data.array[0] = 100;
  gate.open(); await run;
  assert.deepEqual(await read(renderer, data), [3]); renderer.dispose();
});


function targetFor(renderer, width = 2, height = 2) {
  const target = {}, texture = {width, data: new Uint8Array(width * height * 4)};
  const entry = {target, alive: true, busy: false, depthDirty: false, dirty: false,
    shape: {format: 'r32float', width, height, sampleCount: 1}, validation: Promise.resolve(),
    lifetime: new AbortController(), storage: {whenIdle: async () => {}}};
  renderer._targets = {
    capture(value) { assert.equal(value, target); return entry; },
    assertCurrent(value) { assert.equal(value, entry); if (!value.alive) throw new Error('stale target'); if (value.error) throw value.error; },
    ensure(value) { this.assertCurrent(value); },
    render(value, callback) {
      this.assertCurrent(value); value.busy = true; value.dirty = value.depthDirty = true;
      try { callback({}, texture); } finally { value.busy = false; }
    },
  };
  return {target, entry, values: () => [...new Float32Array(texture.data.buffer)]};
}

test('public compute-to-target copies preserve compute/copy/draw order without waiting for completion', async () => {
  const {renderer, device, block, events} = setup(), {target, entry, values} = targetFor(renderer);
  const data = attribute(new Float32Array([0, 10, 20, 30])), kernel = node([storage(data)]);
  const gate = block(), fence = deferred();
  device.queue.onSubmittedWorkDone = () => fence.promise;
  const a = renderer.computeAsync(kernel), options = {width: 2, height: 2};
  const copy = renderer.copyStorageBufferToRenderTargetAsync(data, target, options);
  options.flipY = true; options.width = 1;
  let drawn;
  const draw = new Promise((resolve, reject) => renderer._enqueue({resolve, reject, issue() { drawn = values(); events.push(['draw']); }}));
  const b = renderer.computeAsync(kernel);
  gate.open(); await Promise.all([a, b, draw]); // Fence still unresolved.
  assert.deepEqual(drawn, [1, 11, 21, 31]);
  assert.deepEqual(events.filter(e => ['dispatch', 'texture-copy', 'draw'].includes(e[0])).map(e => e[0]),
    ['dispatch', 'texture-copy', 'texture-copy', 'draw', 'dispatch']);
  fence.resolve(); assert.equal(await copy, true);
  assert.deepEqual(await read(renderer, data), [2, 12, 22, 32]);
  assert.equal(entry.depthDirty, false); assert.equal(entry.dirty, true); renderer.dispose();
});

test('copying compute storage never uploads newer CPU values or needsUpdate ranges', async () => {
  const {renderer, events} = setup(), {target, values} = targetFor(renderer);
  const data = attribute(new Float32Array([0, 10, 20, 30]));
  renderer.compute(node([storage(data)]));
  const uploads = events.filter(e => e[0] === 'upload').length;
  data.array.fill(999); data.version++; data.updateRanges = [{start: 0, count: 4}];
  await renderer.copyStorageBufferToRenderTargetAsync(data, target);
  assert.deepEqual(values(), [1, 11, 21, 31]);
  assert.equal(events.filter(e => e[0] === 'upload').length, uploads);
  assert.deepEqual(data.updateRanges, [{start: 0, count: 4}]); renderer.dispose();
});

test('public copies reject missing GPU storage and stale target generations', async () => {
  const {renderer, block, events} = setup(), {target, entry} = targetFor(renderer);
  const data = attribute(new Float32Array(4));
  await assert.rejects(renderer.copyStorageBufferToRenderTargetAsync(data, target), /before copying/);
  renderer.compute(node([storage(data)]));
  const gate = block(), copy = renderer.copyStorageBufferToRenderTargetAsync(data, target);
  entry.alive = false; gate.open();
  await assert.rejects(copy, /stale target/);
  assert.equal(events.some(e => e[0] === 'texture-copy'), false); renderer.dispose();
});
