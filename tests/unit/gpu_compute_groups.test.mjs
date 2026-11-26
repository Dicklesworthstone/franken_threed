/** Compute command grouping against a recording WebGPU queue. Reads happen at
 * queue submission, so the suite distinguishes native queue order from encode
 * order; the tiny recorded kernels are not WGSL or pixel-parity evidence. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';

function deviceFixture() {
  const calls = [], observations = [], buffers = [];
  let scopes = 0, lose;
  const lost = new Promise(resolve => { lose = resolve; });
  const device = {
    calls, observations, buffers, lost, lose,
    get scopes() { return scopes; },
    limits: {maxComputeWorkgroupsPerDimension: 65535},
    pushErrorScope() { scopes++; },
    popErrorScope() { assert.ok(scopes > 0); scopes--; return Promise.resolve(null); },
    createBuffer({size, usage}) {
      const buffer = {size, usage, data: new Uint8Array(size), destroyed: false, destroyCount: 0,
        destroy() { this.destroyed = true; this.destroyCount++; }};
      buffers.push(buffer); return buffer;
    },
    createShaderModule: descriptor => descriptor,
    createComputePipeline: descriptor => ({code: descriptor.compute.module.code,
      getBindGroupLayout: index => ({index})}),
    createBindGroup: descriptor => descriptor,
    createCommandEncoder() {
      calls.push('encoder');
      const commands = [];
      return {
        beginComputePass() {
          calls.push('begin');
          let pipeline; const groups = new Map();
          const record = counts => {
            // Native encoding captures the selected pipeline and bind groups;
            // their GPU buffer contents are read only when submitted.
            const selected = pipeline, bindings = new Map(groups);
            commands.push(() => {
              for (const group of bindings.values()) for (const entry of group.entries)
                if (entry.resource.buffer) assert.equal(entry.resource.buffer.destroyed, false, 'submitted binding stays alive');
              const dimensions = counts(), buffer = bindings.get(0)?.entries[0]?.resource.buffer;
              const words = buffer ? new Uint32Array(buffer.data.buffer) : null;
              observations.push({code: selected.code, dimensions, words: words ? [...words] : null});
              if (words && dimensions.every(n => n > 0)) {
                if (selected.code === 'add-one') for (let i = 0; i < words.length; i++) words[i]++;
                if (selected.code === 'double') for (let i = 0; i < words.length; i++) words[i] *= 2;
              }
            });
          };
          return {setPipeline(value) { pipeline = value; }, setBindGroup(index, value) { groups.set(index, value); },
            dispatchWorkgroups(...counts) { record(() => counts); },
            dispatchWorkgroupsIndirect(buffer, offset) { record(() => {
              assert.equal(buffer.destroyed, false, 'submitted indirect arguments stay alive');
              return [...new Uint32Array(buffer.data.buffer, offset, 3)];
            }); },
            end() { calls.push('end'); }};
        },
        finish() { calls.push('finish'); return commands; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, start = 0, size = data.byteLength - start) {
        calls.push('write');
        buffer.data.set(new Uint8Array(data.buffer, data.byteOffset + start, size), offset);
      },
      submit(batches) { calls.push('submit'); for (const batch of batches) for (const run of batch) run(); },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };
  return device;
}
const words = buffer => [...new Uint32Array(buffer.data.buffer)];
const bindings = buffer => [{index: 0, entries: [{binding: 0, resource: {buffer}}]}];

test('one compute group observes all pre-submission uploads and retains dispatch state', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, buffer = owner.upload(key, new Uint32Array([0]));
  const add = owner.prepareSync({code: 'add-one'}), double = owner.prepareSync({code: 'double'});
  const group = owner.createDispatchGroup();
  group.dispatch(add, bindings(buffer), [1]);
  assert.deepEqual(words(buffer), [0]); assert.equal(device.observations.length, 0);
  owner.upload(key, new Uint32Array([10]), 1);
  const selected = bindings(buffer);
  group.dispatch(double, selected, [1]);
  selected[0].entries[0].resource.buffer = {};
  assert.equal(device.calls.filter(call => call === 'submit').length, 0);
  group.submit();
  assert.deepEqual(device.observations.map(row => [row.code, row.words]), [
    ['add-one', [10]], ['double', [11]],
  ]);
  assert.deepEqual(words(buffer), [22]);
  for (const call of ['encoder', 'begin', 'end', 'finish', 'submit'])
    assert.equal(device.calls.filter(value => value === call).length, 1, call);
  assert.throws(() => group.submit(), /GROUP_CLOSED/);
  group.abort(); await owner.whenIdle();
});

test('aborted groups discard every encoded dispatch while preserving issued writes', t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, buffer = owner.upload(key, new Uint32Array([4]));
  const add = owner.prepareSync({code: 'add-one'}), group = owner.createDispatchGroup();
  group.dispatch(add, bindings(buffer), [1]);
  owner.upload(key, new Uint32Array([9]), 1);
  const failure = new Error('source update failed');
  group.abort(failure); group.abort();
  assert.throws(() => group.submit(), error => error === failure);
  assert.throws(() => group.dispatch(add, bindings(buffer), [1]), error => error === failure);
  assert.deepEqual(words(buffer), [9]);
  assert.equal(device.observations.length, 0);
  assert.equal(device.calls.includes('end'), false);
  assert.equal(device.calls.includes('submit'), false);
});

test('a later invalid dispatch invalidates the entire unfinished group', t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const buffer = owner.upload({}, new Uint32Array([4]));
  const add = owner.prepareSync({code: 'add-one'}), group = owner.createDispatchGroup();
  group.dispatch(add, bindings(buffer), [1]);
  assert.throws(() => group.dispatch(add, bindings(buffer), [65536]), /DISPATCH/);
  assert.throws(() => group.submit(), /DISPATCH/);
  assert.deepEqual(words(buffer), [4]);
  assert.equal(device.calls.includes('submit'), false);
});

test('nested compute groups submit independently ahead of their unsubmitted parent', t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const buffer = owner.upload({}, new Uint32Array([2]));
  const add = owner.prepareSync({code: 'add-one'}), double = owner.prepareSync({code: 'double'});
  const outer = owner.createDispatchGroup();
  outer.dispatch(add, bindings(buffer), [1]);
  const inner = owner.createDispatchGroup();
  inner.dispatch(double, bindings(buffer), [1]); inner.submit();
  assert.deepEqual(words(buffer), [4]);
  outer.dispatch(add, bindings(buffer), [1]); outer.submit();
  assert.deepEqual(device.observations.map(row => [row.code, row.words]), [
    ['double', [2]], ['add-one', [4]], ['add-one', [5]],
  ]);
  assert.deepEqual(words(buffer), [6]);
  assert.equal(device.calls.filter(call => call === 'submit').length, 2);
});

test('groups close every error scope before yielding and validate one timestamp interval', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  let starts = 0, cancellations = 0; const validations = [];
  const timestamp = {writes() { starts++; return {}; }, cancel() { cancellations++; },
    validate(promise) { validations.push(promise); }};
  const pipeline = owner.prepareSync({code: 'empty'}), group = owner.createDispatchGroup(timestamp);
  group.dispatch(pipeline, [], [1]); assert.equal(device.scopes, 0);
  await Promise.resolve();
  group.dispatch(pipeline, [], [2]); assert.equal(device.scopes, 0);
  group.submit(); assert.equal(device.scopes, 0);
  assert.equal(starts, 1); assert.equal(validations.length, 1); assert.equal(cancellations, 0);
  await validations[0]; await owner.whenIdle();
  const abandoned = owner.createDispatchGroup(timestamp);
  abandoned.dispatch(pipeline, [], [1]); abandoned.abort(); abandoned.abort();
  assert.equal(cancellations, 1);
});

test('captured backend operations begin and finish once for a complete compute group', t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const pipeline = owner.prepareSync({code: 'empty'}), calls = [], context = {};
  const scope = {context, run(callback) { calls.push('scope'); return callback(); }, backend: {
    beginCompute(value, encoder, descriptor) {
      assert.equal(value, context); calls.push('begin'); return encoder.beginComputePass(descriptor);
    },
    compute(value, node, groups, selected, dispatchSize, pass, dimensions) {
      assert.equal(value, context); assert.equal(selected, pipeline); assert.equal(groups.length, 0);
      calls.push(node.name); assert.deepEqual(dispatchSize, dimensions);
      pass.setPipeline(selected); pass.dispatchWorkgroups(...dimensions);
    },
    finishCompute(value, pass) { assert.equal(value, context); calls.push('finish'); pass.end(); },
  }};
  const group = owner.createDispatchGroup(null, scope);
  group.begin(); group.begin();
  assert.deepEqual(calls, ['scope', 'begin'], 'begin precedes source node updates and is idempotent');
  for (const name of ['a', 'b'])
    group.dispatch(pipeline, [], [1, 1, 1], {scope, node: {name}, dispatchSize: [1, 1, 1]});
  group.submit();
  assert.deepEqual(calls, ['scope', 'begin', 'scope', 'a', 'scope', 'b', 'scope', 'finish']);
  const aborted = owner.createDispatchGroup(null, scope);
  aborted.dispatch(pipeline, [], [1, 1, 1], {scope, node: {name: 'c'}, dispatchSize: [1, 1, 1]});
  aborted.abort();
  assert.equal(calls.filter(call => call === 'finish').length, 1);
});

test('indirect arguments are read at group submission and standalone dispatch stays immediate', t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, buffer = owner.upload(key, new Uint32Array([1, 1, 1]));
  const pipeline = owner.prepareSync({code: 'empty'}), group = owner.createDispatchGroup();
  group.dispatch(pipeline, [], {buffer, offset: 0});
  owner.upload(key, new Uint32Array([3, 2, 1]), 1); group.submit();
  assert.deepEqual(device.observations[0].dimensions, [3, 2, 1]);
  owner.dispatch(pipeline, [], [4]);
  assert.deepEqual(device.observations[1].dimensions, [4, 1, 1]);
});

test('disposing the owner prevents submission and cancels each unsubmitted timestamp', t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  let cancelled = 0;
  const pipeline = owner.prepareSync({code: 'empty'});
  const group = owner.createDispatchGroup({writes: () => ({}), cancel() { cancelled++; }, validate() {}});
  group.dispatch(pipeline, [], [1]);
  owner.dispose();
  assert.throws(() => group.submit(), /DISPOSED/);
  assert.equal(cancelled, 1); assert.equal(device.calls.includes('submit'), false);
});

for (const uniform of [false, true])
test(`queued ${uniform ? 'uniform' : 'storage'} bindings retain replaced allocations until submission`, async t => {
  const device = deviceFixture(), owner = createGpuCompute(device, {maxBytes: 12, maxBuffers: 2});
  t.after(() => owner.dispose());
  const key = {}, first = owner.upload(key, new Uint32Array([7]), 0, {uniform});
  const pipeline = owner.prepareSync({code: 'empty'}), group = owner.createDispatchGroup();
  group.dispatch(pipeline, bindings(first), [1]);
  const second = owner.upload(key, new Uint32Array([11, 13]), 1, {uniform});
  assert.equal(owner.getBuffer(key), second);
  assert.equal(first.destroyed, false);
  assert.equal(owner.allocatedBytes, 12);
  assert.equal(owner.bufferCount, 2);
  group.dispatch(pipeline, bindings(second), [1]);
  group.submit();
  assert.deepEqual(device.observations.map(row => row.words), [[7], [11, 13]]);
  assert.equal(first.destroyCount, 1);
  assert.equal(second.destroyed, false);
  assert.equal(owner.allocatedBytes, 8);
  assert.equal(owner.bufferCount, 1);
  await owner.whenIdle();
});

test('public release removes storage ownership while every open group retains its recorded buffer', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, buffer = owner.upload(key, new Uint32Array([2]));
  const add = owner.prepareSync({code: 'add-one'}), double = owner.prepareSync({code: 'double'});
  const outer = owner.createDispatchGroup(), inner = owner.createDispatchGroup();
  outer.dispatch(add, bindings(buffer), [1]);
  outer.dispatch(add, bindings(buffer), [1]);
  inner.dispatch(double, bindings(buffer), [1]);
  owner.release(key); owner.release(key);
  assert.equal(owner.getBuffer(key), null);
  assert.equal(owner.allocatedBytes, 4);
  assert.equal(owner.bufferCount, 1);
  inner.submit();
  assert.equal(buffer.destroyed, false);
  assert.deepEqual(words(buffer), [4]);
  outer.submit();
  assert.deepEqual(device.observations.map(row => row.words), [[2], [4], [5]]);
  assert.equal(buffer.destroyCount, 1);
  assert.equal(owner.allocatedBytes, 0);
  assert.equal(owner.bufferCount, 0);
  await owner.whenIdle();
});

test('indirect-only buffers retain their accepted allocation across replacement and release', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, first = owner.upload(key, new Uint32Array([2, 3, 4]));
  const pipeline = owner.prepareSync({code: 'empty'}), group = owner.createDispatchGroup();
  group.dispatch(pipeline, [], {buffer: first, offset: 0});
  const second = owner.upload(key, new Uint32Array([5, 6, 7, 8]), 1);
  assert.equal(first.destroyed, false);
  group.dispatch(pipeline, [], {buffer: second, offset: 4});
  owner.release(key);
  assert.equal(second.destroyed, false);
  assert.equal(owner.allocatedBytes, 28);
  group.submit();
  assert.deepEqual(device.observations.map(row => row.dimensions), [[2, 3, 4], [6, 7, 8]]);
  assert.equal(first.destroyCount, 1); assert.equal(second.destroyCount, 1);
  assert.equal(owner.allocatedBytes, 0);
  await owner.whenIdle();
});

test('retained generations remain charged to both byte and count budgets', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device, {maxBytes: 12, maxBuffers: 2});
  t.after(() => owner.dispose());
  const key = {}, buffer = owner.upload(key, new Uint32Array([1, 2]));
  const pipeline = owner.prepareSync({code: 'empty'}), group = owner.createDispatchGroup();
  group.dispatch(pipeline, bindings(buffer), [1]); owner.release(key);
  assert.throws(() => owner.upload({}, new Uint32Array([1, 2])), /BUDGET/);
  const secondKey = {}, second = owner.upload(secondKey, new Uint32Array([3]));
  assert.throws(() => owner.upload(secondKey, new Uint32Array([4, 5])), /BUDGET/);
  assert.equal(owner.getBuffer(secondKey), second);
  assert.equal(buffer.destroyed, false);
  assert.equal(owner.allocatedBytes, 12);
  group.abort();
  assert.equal(buffer.destroyCount, 1);
  assert.equal(owner.allocatedBytes, 4);
  const replacement = owner.upload(secondKey, new Uint32Array([4, 5]));
  assert.equal(second.destroyCount, 1); assert.equal(replacement.destroyed, false);
  await owner.whenIdle();

  const bounded = createGpuCompute(device, {maxBytes: 128, maxBuffers: 1});
  t.after(() => bounded.dispose());
  const old = bounded.upload(key, new Uint32Array([6]));
  const waiting = bounded.createDispatchGroup(), prepared = bounded.prepareSync({code: 'empty'});
  waiting.dispatch(prepared, bindings(old), [1]);
  assert.throws(() => bounded.upload(key, new Uint32Array([7, 8])), /BUDGET/);
  assert.equal(bounded.getBuffer(key), old);
  assert.equal(old.destroyed, false);
  bounded.release(key);
  assert.throws(() => bounded.upload({}, new Uint32Array([9])), /BUDGET/);
  waiting.submit();
  assert.equal(old.destroyCount, 1);
  bounded.upload({}, new Uint32Array([10]));
  await bounded.whenIdle();
});

for (const exit of ['abort', 'invalid dispatch', 'failed submit', 'dispose', 'device loss'])
test(`retired group resources are destroyed once after ${exit}`, async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, buffer = owner.upload(key, new Uint32Array([17]));
  const pipeline = owner.prepareSync({code: 'empty'}), group = owner.createDispatchGroup();
  group.dispatch(pipeline, bindings(buffer), [1]); owner.release(key);
  assert.equal(buffer.destroyed, false);
  if (exit === 'abort') group.abort();
  else if (exit === 'invalid dispatch') assert.throws(() => group.dispatch(pipeline, [], [65536]), /DISPATCH/);
  else if (exit === 'failed submit') {
    const error = new Error('submission failed');
    device.queue.submit = () => { assert.equal(buffer.destroyed, false); throw error; };
    assert.throws(() => group.submit(), value => value === error);
  } else if (exit === 'device loss') {
    device.lose({message: 'lost while recording'}); await Promise.resolve();
  } else owner.dispose();
  assert.equal(buffer.destroyCount, 1);
  assert.equal(owner.allocatedBytes, 0); assert.equal(owner.bufferCount, 0);
  assert.equal(device.observations.length, 0);
  group.abort(); owner.dispose(); owner.dispose();
  assert.equal(buffer.destroyCount, 1);
});

test('indirect submission retains the same selector that admission validated', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const firstKey = {}, secondKey = {};
  const first = owner.upload(firstKey, new Uint32Array([2, 3, 4]));
  const second = owner.upload(secondKey, new Uint32Array([9, 8, 7]));
  const pipeline = owner.prepareSync({code: 'empty'});
  let reads = 0;
  const selector = {get buffer() { reads++; return reads === 1 ? first : second; }, offset: 0};
  const group = owner.createDispatchGroup();
  group.dispatch(pipeline, [], selector); owner.release(firstKey); owner.release(secondKey);
  assert.equal(reads, 1); assert.equal(first.destroyed, false); assert.equal(second.destroyCount, 1);
  group.submit();
  assert.deepEqual(device.observations[0].dimensions, [2, 3, 4]);
  assert.equal(first.destroyCount, 1);
  await owner.whenIdle();
});

test('recording hooks can release selected resources without destroying their pending uses', async t => {
  const device = deviceFixture(), owner = createGpuCompute(device);
  t.after(() => owner.dispose());
  const key = {}, indirectKey = {}, buffer = owner.upload(key, new Uint32Array([3]));
  const indirect = owner.upload(indirectKey, new Uint32Array([1, 1, 1]));
  const pipeline = owner.prepareSync({code: 'add-one'}), selector = {buffer: indirect, offset: 0};
  const create = device.createBindGroup;
  device.createBindGroup = descriptor => { owner.release(key); return create(descriptor); };
  const scope = {context: {}, run(run) { selector.buffer = {}; selector.offset = 99; return run(); }, backend: {
    beginCompute(context, encoder, descriptor) { return encoder.beginComputePass(descriptor); },
    compute(context, node, groups, selected, dispatchSize, pass, captured) {
      assert.notEqual(captured, selector); assert.equal(Object.isFrozen(captured), true);
      pass.setPipeline(selected);
      for (const {index, group} of groups) pass.setBindGroup(index, group);
      pass.dispatchWorkgroupsIndirect(captured.buffer, captured.offset);
    },
    finishCompute(context, pass) {
      owner.release(indirectKey); assert.equal(indirect.destroyed, false); pass.end();
    },
  }};
  const group = owner.createDispatchGroup(null, scope);
  group.dispatch(pipeline, bindings(buffer), selector, {scope, node: {}, dispatchSize: null});
  assert.equal(buffer.destroyed, false);
  group.submit();
  assert.deepEqual(device.observations[0].dimensions, [1, 1, 1]);
  assert.deepEqual(words(buffer), [4]);
  assert.equal(buffer.destroyCount, 1); assert.equal(indirect.destroyCount, 1);
  assert.equal(owner.cachedBindGroupCount, 0);
  assert.equal(owner.allocatedBytes, 0);
  await owner.whenIdle();
});
