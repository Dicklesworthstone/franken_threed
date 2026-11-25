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
      const buffer = {size, usage, data: new Uint8Array(size), destroyed: false,
        destroy() { this.destroyed = true; }};
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
            dispatchWorkgroupsIndirect(buffer, offset) { record(() => [...new Uint32Array(buffer.data.buffer, offset, 3)]); },
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
