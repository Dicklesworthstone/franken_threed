import test from 'node:test';
import assert from 'node:assert/strict';
import {withThreeCompute} from '../../tools/ingest/three_compute.mjs';
import {computeAttributeSnapshot} from '../../tools/ingest/gpu_compute_attributes.mjs';

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
        mapAsync: () => mapping?.promise ?? Promise.resolve(),
        getMappedRange(offset = 0, length = size - offset) { return this.data.buffer.slice(offset, offset + length); }};
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
          commands.push(() => { events.push(['read', from, size]); target.data.set(source.data.subarray(from, from + size), to); });
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
      this.info = {calls: 0, frame: 0, compute: {calls: 0, frameCalls: 0},
        memory: {total: 0, readbackBuffers: 0, readbackBuffersSize: 0}};
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
function readbackTarget(maxByteLength) {
  const listeners = new Map();
  return {isReadbackBuffer: true, maxByteLength, buffer: null, _mapped: false,
    addEventListener(type, listener) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(listener); },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    release() { for (const listener of [...listeners.get('release') ?? []]) listener(); },
    dispose() { for (const listener of [...listeners.get('dispose') ?? []]) listener(); }};
}

test('asynchronous compilation joins storage upload validation before progress', async t => {
  const {renderer, device} = setup(), validation = deferred(), data = attribute(new Float32Array([1]));
  const write = device.queue.writeBuffer, pop = device.popErrorScope;
  let upload = false, held = false, successful = false, progress = 0;
  device.queue.writeBuffer = (...args) => { upload = true; return write(...args); };
  device.popErrorScope = () => {
    const result = pop();
    if (upload && !held) { held = true; return validation.promise; }
    return result;
  };
  t.after(() => { validation.resolve(null); renderer.dispose(); });
  const task = renderer.compileComputeAsync(node([storage(data)]), () => progress++).then(() => { successful = true; });
  task.catch(() => {});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(held, true); assert.equal(successful, false);
  validation.resolve({message: 'invalid captured storage upload'});
  await assert.rejects(task, /invalid captured storage upload/);
  assert.equal(progress, 0);
});

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
  const {renderer, device, buffers, block, events} = setup(), data = attribute(new Float32Array([0, 10])), kernel = node([storage(data)]);
  renderer.compute(kernel); // GPU now [1, 11], CPU still [0, 10].
  const gate = block();
  data.array[1] = 30; data.version++; data.updateRanges = [{start: 1, count: 1}];
  const first = renderer.computeAsync(kernel), middle = renderer.getArrayBufferAsync(data);
  const oldPublication = new Promise((resolve, reject) => renderer._enqueue({resolve, reject, issue() {
    assert.throws(() => computeAttributeSnapshot(device, data), /UNPREPARED/);
  }}));
  data.array[0] = 100; data.version++; data.updateRanges = [{start: 0, count: 1}];
  const second = renderer.computeAsync(kernel);
  data.array.fill(-500); // Unversioned changes cannot contaminate either upload.
  gate.open(); await Promise.all([first, second, oldPublication, gate.done]);
  assert.deepEqual([...new Float32Array(await middle)], [2, 31]);
  assert.deepEqual(await read(renderer, data), [101, 32]);
  assert.equal(computeAttributeSnapshot(device, data).buffer, buffers[0]);
  assert.equal(events.filter(e => e[0] === 'upload').length, 3);
  assert.equal(data.uploads, 0, 'native storage uploads leave WebGL callbacks inert'); renderer.dispose();
});

test('H8-sized structured storage preserves packed member bytes and matrix padding', async () => {
  const {renderer, block, events} = setup();
  // H8's position:vec3, velocity:vec3, C:mat3 occupy 20 32-bit words.
  // Include an integer NaN bit pattern: treating fields as JS float values
  // rather than copying storage bytes must not canonicalize it.
  const words = new Uint32Array(40);
  for (let i = 0; i < words.length; i++) words[i] = i % 20 === 3 ? 0x7fc01234 : 0x3f800000 + i;
  const source = new Float32Array(words.buffer), data = attribute(source, 20);
  data.isStorageInstancedBufferAttribute = true;
  const kernel = node([storage(data)], {name: 'packed-struct', count: 2});
  const expected = [...words], gate = block(), run = renderer.computeAsync(kernel);
  words.fill(0); // This unversioned CPU edit must not alter the queued copy.
  gate.open(); await run;
  assert.deepEqual([...new Uint32Array(await renderer.getArrayBufferAsync(data))], expected);
  assert.equal(data.array, source); assert.equal(data.itemSize, 20); assert.equal(data.count, 2);
  assert.equal(events.filter(e => e[0] === 'upload').length, 1);
  assert.equal(data.uploads, 0, 'native storage uploads leave WebGL callbacks inert'); renderer.dispose();
});

test('queued storage replacement retains each captured shape until its own submission', async () => {
  const {renderer, block} = setup(), data = attribute(new Float32Array(20).fill(1), 20);
  const kernel = node([storage(data)], {name: 'packed-struct', count: 1});
  const gate = block(), first = renderer.computeAsync(kernel), before = renderer.getArrayBufferAsync(data);
  data.array = new Float32Array(40).fill(2); data.count = 2; data.version++;
  const second = renderer.computeAsync(kernel);
  gate.open(); await Promise.all([first, second]);
  assert.deepEqual([...new Float32Array(await before)], Array(20).fill(1));
  assert.deepEqual(await read(renderer, data), Array(40).fill(2));
  renderer.dispose();
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
  assert.equal(data.uploads, 0, 'native storage uploads leave WebGL callbacks inert');
  assert.equal(events.filter(e => e[0] === 'upload').length, 1);
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

test('public readback forwards byte ranges and caller targets without copying the full storage buffer', async t => {
  const {renderer, buffers, events} = setup(), data = attribute(new Float32Array([10, 20, 30, 40]));
  t.after(() => renderer.dispose()); renderer.compute(node([storage(data)]));
  const target = new Float32Array([-1, -1, -1]).buffer;
  assert.equal(await renderer.getArrayBufferAsync(data, target, 4, 8), target);
  assert.deepEqual([...new Float32Array(target)], [21, 31, -1]);
  assert.deepEqual([...new Float32Array(await renderer.getArrayBufferAsync(data, null, 12))], [41]);
  assert.deepEqual(events.filter(event => event[0] === 'read'), [['read', 4, 8], ['read', 12, 4]]);
  assert.deepEqual(buffers.filter(buffer => buffer.usage & 1).map(buffer => buffer.size), [8, 4]);
});

test('queued source readback handles reserve immediately and preserve their compute/read/compute position', async t => {
  const mapping = deferred(), {renderer, block, buffers, events} = setup({mapping});
  t.after(() => { mapping.resolve(); renderer.dispose(); });
  const data = attribute(new Float32Array([10, 20, 30, 40])), kernel = node([storage(data)]), target = readbackTarget(16), gate = block();
  const first = renderer.computeAsync(kernel), result = renderer.getArrayBufferAsync(data, target, 4, 8), second = renderer.computeAsync(kernel);
  assert.equal(target._mapped, true); assert.equal(target.buffer, null);
  await assert.rejects(renderer.getArrayBufferAsync(data, target, 0, 4), /READBACK_MAPPED/);
  gate.open(); await Promise.all([first, second]);
  assert.deepEqual(events.filter(event => ['dispatch', 'read'].includes(event[0])).map(event => event[0]), ['dispatch', 'read', 'dispatch']);
  mapping.resolve(); assert.equal(await result, target);
  assert.deepEqual([...new Float32Array(target.buffer)], [21, 31]);
  assert.deepEqual(renderer.info.memory, {total: 16, readbackBuffers: 1, readbackBuffersSize: 16});
  target.release(); await renderer.getArrayBufferAsync(data, target, 12, 4);
  assert.deepEqual([...new Float32Array(target.buffer)], [42]);
  assert.equal(buffers.filter(buffer => buffer.usage & 1).length, 1);
  target.dispose(); assert.deepEqual(renderer.info.memory, {total: 0, readbackBuffers: 0, readbackBuffersSize: 0});
});

for (const action of ['release', 'dispose'])
test(`source ${action} cancels a readback still waiting for submission without affecting reuse`, async t => {
  const {renderer, block, buffers, events} = setup(), data = attribute(new Float32Array([1, 2, 3, 4])), target = readbackTarget(16);
  t.after(() => renderer.dispose()); renderer.compute(node([storage(data)]));
  const gate = block(), first = renderer.getArrayBufferAsync(data, target, 0, 4);
  const rejected = assert.rejects(first, /READBACK_(RELEASED|DISPOSED)/);
  target[action](); assert.equal(target._mapped, false);
  const second = renderer.getArrayBufferAsync(data, target, 8, 8);
  gate.open(); await rejected; await second;
  assert.deepEqual([...new Float32Array(target.buffer)], [4, 5]); assert.equal(target._mapped, true);
  assert.deepEqual(events.filter(event => event[0] === 'read'), [['read', 8, 8]]);
  assert.equal(buffers.filter(buffer => buffer.usage & 1).length, 1);
});

test('pre-init ranged readback handles and failed queue admission release their reservations', async t => {
  const initialization = deferred(), {renderer, events} = setup({initialization});
  t.after(() => { initialization.resolve(); renderer.dispose(); });
  const data = attribute(new Float32Array([3, 6, 9, 12])), target = readbackTarget(8);
  const compute = renderer.computeAsync(node([storage(data)])), result = renderer.getArrayBufferAsync(data, target, 8, 8);
  assert.equal(target._mapped, true); assert.equal(events.length, 0);
  initialization.resolve(); await compute; await result;
  assert.deepEqual([...new Float32Array(target.buffer)], [10, 13]);
  target.release();
  const enqueue = renderer._enqueue, previousDrain = renderer._drain;
  renderer._drain = Promise.resolve(); renderer._enqueue = () => { throw new Error('queue full'); };
  await assert.rejects(renderer.getArrayBufferAsync(data, target, 0, 4), /queue full/);
  assert.equal(target._mapped, false); assert.equal(target.buffer, null);
  renderer._enqueue = enqueue; renderer._drain = previousDrain;
  await renderer.getArrayBufferAsync(data, target, 0, 4);
  assert.deepEqual([...new Float32Array(target.buffer)], [4]);
});

for (const failure of ['resource completion', 'released mapping'])
test(`late ${failure} cannot leak or overwrite a reusable native mapping`, async t => {
  const {renderer, device, buffers} = setup(), data = attribute(new Float32Array([1, 2, 3, 4])), target = readbackTarget(16);
  t.after(() => renderer.dispose());
  const createBuffer = device.createBuffer;
  device.createBuffer = options => {
    const buffer = createBuffer(options), map = buffer.mapAsync;
    buffer.nativeMapped = false; buffer.unmapCount = 0;
    buffer.mapAsync = async (...args) => { await map(...args); buffer.nativeMapped = true; };
    buffer.unmap = () => { buffer.nativeMapped = false; buffer.unmapCount++; };
    return buffer;
  };
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = createEncoder(), copy = encoder.copyBufferToBuffer;
    encoder.copyBufferToBuffer = (source, from, destination, to, size) => {
      assert.equal(destination.nativeMapped, false, 'a mapped staging buffer cannot accept a new GPU copy');
      copy(source, from, destination, to, size);
    };
    return encoder;
  };
  renderer.compute(node([storage(data)]));
  const textures = renderer._getSampledTextureInputs(), idle = deferred();
  textures.whenIdle = () => idle.promise;
  const first = renderer.getArrayBufferAsync(data, target, 0, 4);
  const rejected = assert.rejects(first, failure === 'resource completion' ? /late resource failure/ : /READBACK_RELEASED/);
  await new Promise(resolve => setImmediate(resolve));
  const staging = buffers.find(buffer => buffer.usage === 9);
  assert.equal(staging.nativeMapped, true); assert.equal(target._mapped, true);
  textures.whenIdle = async () => {};
  if (failure === 'resource completion') {
    idle.reject(new Error('late resource failure')); await rejected;
  } else target.release();
  assert.equal(staging.nativeMapped, false); assert.equal(staging.unmapCount, 1);
  assert.equal(target._mapped, false); assert.equal(target.buffer, null);
  await renderer.getArrayBufferAsync(data, target, 4, 4);
  if (failure === 'released mapping') { idle.resolve(); await rejected; }
  assert.equal(target._mapped, true); assert.equal(staging.nativeMapped, true);
  assert.deepEqual([...new Float32Array(target.buffer)], [3]);
  assert.equal(buffers.filter(buffer => buffer.usage === 9).length, 1);
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
  const target = {texture: {}}, texture = {width, data: new Uint8Array(width * height * 4)};
  const entry = {target, alive: true, busy: false, depthDirty: false, dirty: false, references: 0,
    shape: {format: 'r32float', width, height, sampleCount: 1, textures: [target.texture]}, validation: Promise.resolve(),
    lifetime: new AbortController(), storage: {whenIdle: async () => {}}};
  renderer._targets = {
    capture(value) { assert.equal(value, target); return entry; },
    assertCurrent(value) { assert.equal(value, entry); if (!value.alive) throw new Error('stale target'); if (value.error) throw value.error; },
    ensure(value) { this.assertCurrent(value); },
    validateNativeTextures(writes) { for (const {entry} of writes) this.assertCurrent(entry); },
    retainGeneration(value) {
      this.assertCurrent(value); value.references++; let active = true;
      return () => { assert.equal(active, true); active = false; value.references--; };
    },
    withNativeColorDestination({entry: value, texture: source, depth, colorIndex}, callback) {
      assert.equal(source, target.texture); assert.equal(depth, false); assert.equal(colorIndex, 0);
      this.assertCurrent(value); value.busy = true; value.dirty = true;
      try { callback(texture); } finally { value.busy = false; }
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
  assert.equal(entry.references, 1);
  options.flipY = true; options.width = 1;
  let drawn;
  const draw = new Promise((resolve, reject) => renderer._enqueue({resolve, reject, issue() { drawn = values(); events.push(['draw']); }}));
  const b = renderer.computeAsync(kernel);
  gate.open(); await Promise.all([a, b, draw]); // Fence still unresolved.
  assert.deepEqual(drawn, [1, 11, 21, 31]);
  assert.deepEqual(events.filter(e => ['dispatch', 'texture-copy', 'draw'].includes(e[0])).map(e => e[0]),
    ['dispatch', 'texture-copy', 'texture-copy', 'draw', 'dispatch']);
  assert.equal(entry.references, 1, 'pending GPU completion retains the captured target');
  fence.resolve(); assert.equal(await copy, true); assert.equal(entry.references, 0);
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
  await assert.rejects(copy, /stale target/); assert.equal(entry.references, 0);
  assert.equal(events.some(e => e[0] === 'texture-copy'), false); renderer.dispose();
});
