import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from './gpu_compute.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';

// Real production owners run against an independent byte-addressed command
// recorder. This is not a WGSL interpreter or a hardware performance benchmark.
function gpu() {
  let lose;
  const calls = {groups: [], layouts: [], buffers: [], submits: [], passes: [], fences: 0};
  const device = {
    limits: {maxBindGroups: 4, maxBindingsPerBindGroup: 16, maxBufferSize: 4096,
      maxStorageBufferBindingSize: 4096, maxUniformBufferBindingSize: 4096, maxComputeWorkgroupsPerDimension: 65535},
    lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope() {},
    popErrorScope() { const error = this.error; this.error = null; return Promise.resolve(error ?? null); },
    createBuffer(descriptor) {
      const buffer = {...descriptor, bytes: new Uint8Array(descriptor.size), destroyed: false,
        destroy() { this.destroyed = true; }, unmap() {}, mapAsync() { return Promise.resolve(); },
        getMappedRange() { assert.equal(this.destroyed, false); return this.bytes.buffer; }};
      calls.buffers.push(buffer); return buffer;
    },
    createShaderModule(descriptor) { return descriptor; },
    createBindGroupLayout(descriptor) { return descriptor; },
    createPipelineLayout(descriptor) { return descriptor; },
    createComputePipeline(descriptor) {
      const pipeline = {descriptor, getBindGroupLayout(index) {
        calls.layouts.push({pipeline, index}); return {pipeline, index};
      }};
      return pipeline;
    },
    createComputePipelineAsync(descriptor) { return Promise.resolve(this.createComputePipeline(descriptor)); },
    createBindGroup(descriptor) {
      this.onCreateGroup?.();
      if (this.throwGroup) { this.throwGroup = false; throw new Error('bind group creation failed'); }
      const group = {layout: descriptor.layout,
        entries: descriptor.entries.map(({binding, resource}) => ({binding,
          resource: Object.prototype.toString.call(resource) === '[object Object]' && 'buffer' in resource ? {...resource} : resource}))};
      calls.groups.push(group); return group;
    },
    createCommandEncoder() {
      const commands = [];
      return {
        beginComputePass() {
          const pass = {pipeline: null, groups: new Map(),
            setPipeline(pipeline) { this.pipeline = pipeline; },
            setBindGroup(index, group) { this.groups.set(index, group); },
            dispatchWorkgroups(...size) { this.direct = size; },
            dispatchWorkgroupsIndirect(buffer, offset) { this.indirect = {buffer, offset}; },
            end() {
              calls.passes.push(this);
              commands.push(() => {
                // Optional deterministic kernel model: dst = src + 1. Read at
                // submission, never at group creation, using independent bytes.
                if (this.pipeline.descriptor.compute.module.code !== 'increment') return;
                const entries = this.groups.get(0).entries;
                const source = entries.find(entry => entry.binding === 0).resource;
                const destination = entries.find(entry => entry.binding === 1).resource;
                assert.equal(source.buffer.destroyed, false); assert.equal(destination.buffer.destroyed, false);
                const sourceOffset = source.offset ?? 0, destinationOffset = destination.offset ?? 0;
                const size = source.size ?? source.buffer.size - sourceOffset;
                const from = new Uint32Array(source.buffer.bytes.buffer, sourceOffset, size / 4);
                const to = new Uint32Array(destination.buffer.bytes.buffer, destinationOffset, size / 4);
                for (let i = 0; i < from.length; i++) to[i] = from[i] + 1;
              });
            }};
          return pass;
        },
        copyBufferToBuffer(source, from, destination, to, size) {
          commands.push(() => destination.bytes.set(source.bytes.slice(from, from + size), to));
        },
        finish() { return commands; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, start = 0, length = data.byteLength - start) {
        buffer.bytes.set(new Uint8Array(data.buffer, data.byteOffset + start, length), offset);
      },
      submit(commandBuffers) { calls.submits.push(commandBuffers); for (const commands of commandBuffers) for (const command of commands) command(); },
      onSubmittedWorkDone() { calls.fences++; return Promise.resolve(); },
    },
  };
  return {device, calls, lose};
}
function setup(options) {
  const state = gpu();
  const owner = createGpuCompute(state.device, options);
  const pipeline = owner.prepareSync({code: 'noop'});
  const aKey = {}, bKey = {};
  const a = owner.upload(aKey, new Uint32Array([1, 2, 3, 4]));
  const b = owner.upload(bKey, new Uint32Array(4));
  return {...state, owner, pipeline, a, b, aKey, bKey};
}
const entries = (a, b) => [{binding: 0, resource: {buffer: a}}, ...(b ? [{binding: 1, resource: {buffer: b}}] : [])];
const group = (a, b, index = 0) => [{index, entries: entries(a, b)}];
const tagged = type => ({[Symbol.toStringTag]: type});
const words = buffer => [...new Uint32Array(buffer.bytes.buffer)];
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

test('128 identical dispatches create one group and one pipeline layout handle, without fences', async () => {
  const {owner, pipeline, a, b, calls} = setup();
  for (let i = 0; i < 128; i++) owner.dispatch(pipeline, group(a, b), [4]);
  assert.equal(calls.groups.length, 1); assert.equal(calls.layouts.length, 1);
  assert.equal(owner.cachedBindGroupCount, 1); assert.equal(calls.fences, 0);
  assert.equal(calls.passes.length, 128); assert.equal(new Set(calls.passes.map(p => p.groups.get(0))).size, 1);
  assert.ok(calls.passes.every(pass => pass.direct.join(',') === '4,1,1'));
  await owner.whenIdle(); owner.dispose();
});

test('cache hits observe new uploads, not bytes captured when the group was created', async () => {
  const {owner, a, b, aKey, bKey, calls} = setup();
  const pipeline = owner.prepareSync({code: 'increment'});
  owner.dispatch(pipeline, group(a, b), [4]);
  assert.deepEqual(words(b), [2, 3, 4, 5]);
  assert.equal(owner.upload(aKey, new Uint32Array([21, 22, 23, 24]), 1), a);
  owner.dispatch(pipeline, group(a, b), [4]);
  assert.deepEqual([...new Uint32Array(await owner.read(bKey))], [22, 23, 24, 25]);
  assert.equal(calls.groups.length, 1); owner.dispose();
});

test('GPU ping-pong keeps two groups and consumes evolving native bytes', async () => {
  const {owner, a, b, calls} = setup();
  const pipeline = owner.prepareSync({code: 'increment'});
  for (let i = 0; i < 30; i++) owner.dispatch(pipeline, i % 2 ? group(b, a) : group(a, b), [4]);
  assert.deepEqual(words(a), [31, 32, 33, 34]);
  assert.equal(calls.groups.length, 2); assert.equal(calls.layouts.length, 1);
  await owner.whenIdle(); owner.dispose();
});

test('automatic layouts never share groups across pipeline identities', async () => {
  const {owner, pipeline, a, calls} = setup();
  const second = owner.prepareSync({code: 'different shader'});
  for (const p of [pipeline, second, pipeline, second]) owner.dispatch(p, group(a), [1]);
  assert.equal(calls.groups.length, 2); assert.equal(calls.layouts.length, 2);
  assert.notEqual(calls.passes[0].groups.get(0), calls.passes[1].groups.get(0));
  await owner.whenIdle(); owner.dispose();
});

test('asynchronously prepared pipelines use the same binding cache', async () => {
  const {owner, a, calls} = setup();
  const pipeline = await owner.prepare({code: 'async shader'});
  owner.dispatch(pipeline, group(a), [1]); owner.dispatch(pipeline, group(a), [2]);
  assert.equal(calls.groups.length, 1); await owner.whenIdle(); owner.dispose();
});

test('binding group indices are part of the key', () => {
  const {owner, pipeline, a, calls} = setup();
  for (const index of [0, 1, 0, 1]) owner.dispatch(pipeline, group(a, null, index), [1]);
  assert.equal(calls.groups.length, 2); assert.equal(calls.layouts.length, 2); owner.dispose();
});

test('entry order and fresh wrapper identity do not prevent reuse', () => {
  const {owner, pipeline, a, b, calls} = setup();
  owner.dispatch(pipeline, group(a, b), [1]);
  owner.dispatch(pipeline, [{index: 0, entries: entries(a, b).reverse()}], [1]);
  assert.equal(calls.groups.length, 1);
  assert.deepEqual(calls.groups[0].entries.map(entry => entry.binding), [0, 1]); owner.dispose();
});

test('binding slots distinguish resource alias partitions', () => {
  const {owner, pipeline, a, b, calls} = setup();
  for (const [from, to] of [[a, a], [a, b], [b, a], [a, a]]) owner.dispatch(pipeline, group(from, to), [1]);
  assert.equal(calls.groups.length, 3); owner.dispose();
});

test('buffer offsets and sizes remain part of each group and select exact bytes', () => {
  const {owner, a, b, calls} = setup();
  const pipeline = owner.prepareSync({code: 'increment'});
  for (const offset of [0, 8, 0]) owner.dispatch(pipeline, [{index: 0, entries: [
    {binding: 0, resource: {buffer: a, offset, size: 8}},
    {binding: 1, resource: {buffer: b, offset, size: 8}},
  ]}], [2]);
  assert.deepEqual(words(b), [2, 3, 4, 5]); assert.equal(calls.groups.length, 2); owner.dispose();
});

test('omitted buffer size does not alias a narrower explicit binding', () => {
  const {owner, pipeline, a, calls} = setup();
  for (const size of [undefined, 4, 8, undefined]) owner.dispatch(pipeline, [{index: 0,
    entries: [{binding: 0, resource: {buffer: a, size}}]}], [1]);
  assert.equal(calls.groups.length, 3); owner.dispose();
});

test('mutating caller descriptors cannot mutate an older cached binding', () => {
  const {owner, pipeline, a, b, calls} = setup();
  const groups = group(a), resource = groups[0].entries[0].resource;
  owner.dispatch(pipeline, groups, [1]);
  resource.buffer = b; resource.offset = 4; resource.size = 4;
  owner.dispatch(pipeline, groups, [1]);
  assert.equal(calls.groups[0].entries[0].resource.buffer, a);
  assert.equal(calls.groups[0].entries[0].resource.offset, 0);
  assert.equal(calls.groups[0].entries[0].resource.size, undefined);
  owner.dispatch(pipeline, group(a), [1]); assert.equal(calls.groups.length, 2); owner.dispose();
});

test('stable native texture views and samplers reuse their binding group', () => {
  const {owner, pipeline, a, calls} = setup();
  const view = tagged('GPUTextureView'), sampler = tagged('GPUSampler');
  const make = () => [{index: 0, entries: [...entries(a), {binding: 1, resource: sampler}, {binding: 2, resource: view}]}];
  owner.dispatch(pipeline, make(), [1]); view.contentVersion = 4; owner.dispatch(pipeline, make(), [1]);
  assert.equal(calls.groups.length, 1); owner.dispose();
});

for (const type of ['GPUTextureView', 'GPUSampler']) test(`${type} replacements cannot reuse an older identity`, () => {
  const {owner, pipeline, calls} = setup();
  const one = tagged(type), two = tagged(type);
  for (const resource of [one, two, one]) owner.dispatch(pipeline, [{index: 0, entries: [{binding: 0, resource}]}], [1]);
  assert.equal(calls.groups.length, 2); owner.dispose();
});

for (const resource of [tagged('GPUExternalTexture'), {}, tagged('GPUTexture')]) test(`volatile or unknown ${Object.prototype.toString.call(resource)} bypasses caching`, () => {
  const {owner, pipeline, calls} = setup();
  for (let i = 0; i < 4; i++) owner.dispatch(pipeline, [{index: 0, entries: [{binding: 0, resource}]}], [1]);
  assert.equal(calls.groups.length, 4); assert.equal(owner.cachedBindGroupCount, 0); owner.dispose();
});

test('external texture anywhere in a mixed group prevents caching that group', () => {
  const {owner, pipeline, a, calls} = setup();
  const resource = tagged('GPUExternalTexture');
  for (let i = 0; i < 3; i++) owner.dispatch(pipeline, [{index: 0,
    entries: [...entries(a), {binding: 1, resource}]}], [1]);
  assert.equal(calls.groups.length, 3); assert.equal(owner.cachedBindGroupCount, 0); owner.dispose();
});

for (const type of ['GPUTextureView', 'GPUSampler', 'GPUExternalTexture']) test(`custom buffer properties do not turn ${type} into a buffer binding`, () => {
  const {owner, pipeline, a, calls} = setup();
  const resource = {...tagged(type), buffer: a};
  // A real GPU interface is used as-is despite arbitrary application metadata.
  const groups = [{index: 0, entries: [{binding: 0, resource}]}];
  for (let i = 0; i < 2; i++) owner.dispatch(pipeline, groups, [1]);
  assert.equal(calls.groups.length, type === 'GPUExternalTexture' ? 2 : 1);
  assert.equal(calls.groups[0].entries[0].resource, resource);
  assert.equal(owner.cachedBindGroupCount, type === 'GPUExternalTexture' ? 0 : 1);
  owner.dispose();
});

test('LRU eviction is bounded and cache hits refresh recency', () => {
  const {owner, pipeline, a, b, calls} = setup({maxCachedBindGroups: 2});
  const c = owner.upload({}, new Uint32Array(4));
  for (const buffer of [a, b, a, c, a, b]) {
    owner.dispatch(pipeline, group(buffer), [1]); assert.ok(owner.cachedBindGroupCount <= 2);
  }
  assert.equal(calls.groups.length, 4); assert.equal(calls.layouts.length, 1); owner.dispose();
});

test('releasing one buffer evicts only groups that reference it', () => {
  const {owner, pipeline, a, b, aKey, bKey, calls} = setup();
  for (const gs of [group(a), group(b), group(a, b)]) owner.dispatch(pipeline, gs, [1]);
  assert.equal(owner.cachedBindGroupCount, 3); owner.release(aKey);
  assert.equal(a.destroyed, true); assert.equal(owner.cachedBindGroupCount, 1);
  owner.dispatch(pipeline, group(b), [1]); assert.equal(calls.groups.length, 3);
  owner.release(bKey); assert.equal(owner.cachedBindGroupCount, 0); owner.dispose();
});

test('resizing a buffer evicts its old groups and preserves unrelated groups', () => {
  const {owner, pipeline, a, b, aKey, calls} = setup();
  owner.dispatch(pipeline, group(a), [1]); owner.dispatch(pipeline, group(b), [1]);
  const next = owner.upload(aKey, new Uint32Array(8), 1);
  assert.notEqual(next, a); assert.equal(a.destroyed, true); assert.equal(owner.cachedBindGroupCount, 1);
  owner.dispatch(pipeline, group(next), [1]); owner.dispatch(pipeline, group(b), [1]);
  assert.equal(calls.groups.length, 3); owner.dispose();
});

test('uniform/storage allocation changes cannot reuse destroyed buffers', () => {
  const {owner, pipeline, a, aKey, calls} = setup();
  owner.dispatch(pipeline, group(a), [1]);
  const next = owner.upload(aKey, new Uint32Array(4), 1, {uniform: true});
  assert.equal(owner.cachedBindGroupCount, 0); assert.equal(a.destroyed, true);
  owner.dispatch(pipeline, group(next), [1]); assert.equal(calls.groups.length, 2); owner.dispose();
});

test('failed replacement admission leaves the existing group usable', () => {
  const {owner, pipeline, a, aKey, calls} = setup({maxBytes: 48});
  owner.dispatch(pipeline, group(a), [1]);
  assert.throws(() => owner.upload(aKey, new Uint32Array(8), 1), {code: 'GPU_COMPUTE_BUDGET'});
  owner.dispatch(pipeline, group(a), [1]); assert.equal(calls.groups.length, 1);
  assert.equal(a.destroyed, false); owner.dispose();
});

test('disabled cache keeps fresh groups but bounded pipeline-local layout handles', () => {
  const {owner, pipeline, a, calls} = setup({maxCachedBindGroups: 0});
  for (let i = 0; i < 5; i++) owner.dispatch(pipeline, group(a), [1]);
  assert.equal(calls.groups.length, 5); assert.equal(calls.layouts.length, 1);
  assert.equal(owner.cachedBindGroupCount, 0); owner.dispose();
});

for (const value of [-1, 1.5, NaN, Infinity, '2', null]) test(`invalid cache limit ${String(value)} fails at creation`, () => {
  assert.throws(() => createGpuCompute(gpu().device, {maxCachedBindGroups: value}), {code: 'GPU_COMPUTE_OPTIONS'});
});

const malformed = [null, {}, [{index: -1, entries: []}], [{index: 4, entries: []}],
  [{index: 0, entries: []}, {index: 0, entries: []}], [{index: 0, entries: null}],
  [{index: 0, entries: [{binding: 16, resource: {}}]}],
  [{index: 0, entries: [{binding: 0, resource: {}}, {binding: 0, resource: {}}]}],
  [{index: 0, entries: [{binding: 0, resource: null}]}]];
for (const [index, groups] of malformed.entries()) test(`invalid binding descriptors ${index} fail before GPU allocation/submission`, () => {
  const {owner, pipeline, calls} = setup();
  assert.throws(() => owner.dispatch(pipeline, groups, [1]), {code: 'GPU_COMPUTE_BINDINGS'});
  assert.equal(calls.groups.length, 0); assert.equal(calls.layouts.length, 0);
  assert.equal(calls.submits.length, 0); owner.dispose();
});

test('a bad later group cannot allocate a valid prefix', () => {
  const {owner, pipeline, a, calls} = setup();
  assert.throws(() => owner.dispatch(pipeline, [...group(a), {index: 2, entries: [{binding: 0, resource: null}]}], [1]),
    {code: 'GPU_COMPUTE_BINDINGS'});
  assert.equal(calls.groups.length, 0); assert.equal(calls.layouts.length, 0); owner.dispose();
});

test('sparse group/entry arrays are rejected before a valid prefix allocates', () => {
  const {owner, pipeline, a, calls} = setup();
  const sparseGroups = [group(a)[0], , {index: 2, entries: []}];
  const sparseEntries = [{index: 0, entries: [entries(a)[0], , {binding: 2, resource: {buffer: a}}]}];
  for (const groups of [sparseGroups, sparseEntries])
    assert.throws(() => owner.dispatch(pipeline, groups, [1]), {code: 'GPU_COMPUTE_BINDINGS'});
  assert.equal(calls.groups.length, 0); assert.equal(calls.layouts.length, 0); owner.dispose();
});

test('evicted reverse indices do not over-evict newer unrelated groups', () => {
  const {owner, pipeline, a, b, aKey, bKey, calls} = setup({maxCachedBindGroups: 1});
  owner.dispatch(pipeline, group(a, b), [1]);
  owner.dispatch(pipeline, group(b), [1]);
  owner.release(aKey);
  assert.equal(owner.cachedBindGroupCount, 1);
  owner.dispatch(pipeline, group(b), [1]); assert.equal(calls.groups.length, 2);
  owner.release(bKey); assert.equal(owner.cachedBindGroupCount, 0); owner.dispose();
});

test('cache state is private to each compute owner, even on a shared device', () => {
  const {device, calls} = gpu();
  const one = createGpuCompute(device), two = createGpuCompute(device);
  const first = one.prepareSync({code: 'noop'}), second = two.prepareSync({code: 'noop'});
  const native = one.upload({}, new Uint32Array(4));
  one.dispatch(first, group(native), [1]); two.dispatch(second, group(native), [1]);
  assert.equal(calls.groups.length, 2);
  one.dispose(); assert.equal(two.cachedBindGroupCount, 1); two.dispose();
});

test('aborting the owner cancels its cache and leaves borrowed resources caller-owned', async () => {
  const controller = new AbortController();
  const {owner, pipeline, a} = setup({signal: controller.signal});
  owner.dispatch(pipeline, group(a), [1]);
  const error = new Error('cancelled compute'); controller.abort(error);
  assert.equal(owner.cachedBindGroupCount, 0);
  await assert.rejects(owner.whenIdle(), cause => cause === error);
});

for (const range of [{offset: -1}, {offset: 0.5}, {size: 0}, {size: Infinity}]) test(`invalid buffer range ${JSON.stringify(range)} fails without cache mutation`, () => {
  const {owner, pipeline, a, calls} = setup();
  assert.throws(() => owner.dispatch(pipeline, [{index: 0, entries: [{binding: 0, resource: {buffer: a, ...range}}]}], [1]),
    {code: 'GPU_COMPUTE_BINDINGS'});
  assert.equal(calls.groups.length, 0); owner.dispose();
});

test('indirect dispatch reuses bindings and retains the owned indirect buffer/offset', () => {
  const {owner, pipeline, a, calls} = setup();
  const indirect = owner.upload({}, new Uint32Array([999, 2, 3, 4]));
  for (let i = 0; i < 4; i++) owner.dispatch(pipeline, group(a), {buffer: indirect, offset: 4});
  assert.equal(calls.groups.length, 1);
  assert.ok(calls.passes.every(pass => pass.indirect.buffer === indirect && pass.indirect.offset === 4)); owner.dispose();
});

test('invalid indirect arguments leave cached bindings and submissions unchanged', () => {
  const {owner, pipeline, a, calls} = setup();
  owner.dispatch(pipeline, group(a), [1]);
  assert.throws(() => owner.dispatch(pipeline, group(a), {buffer: a, offset: 8}), {code: 'GPU_COMPUTE_DISPATCH'});
  assert.equal(calls.groups.length, 1); assert.equal(calls.submits.length, 1); owner.dispose();
});

test('partial updates preserve untouched bytes while retaining the binding', () => {
  const {owner, a, b, aKey, calls} = setup();
  const pipeline = owner.prepareSync({code: 'increment'});
  owner.dispatch(pipeline, group(a, b), [4]);
  owner.upload(aKey, new Uint32Array([90, 20, 91, 92]), 1, {ranges: [{start: 1, count: 1}]});
  owner.dispatch(pipeline, group(a, b), [4]);
  assert.deepEqual(words(b), [2, 21, 4, 5]); assert.equal(calls.groups.length, 1); owner.dispose();
});

test('graphics storage publication still borrows the same compute buffer after cache hits', async () => {
  const {owner, pipeline, calls, device} = setup();
  const attribute = {isStorageBufferAttribute: true, array: new Float32Array([1, 2, 3, 4]), itemSize: 4, count: 1, version: 0};
  const native = owner.upload(attribute, attribute.array);
  const lease = captureComputeStorage(device, attribute);
  owner.dispatch(pipeline, group(native), [1]); owner.dispatch(pipeline, group(native), [1]);
  assert.equal(lease.resolve().buffer, native); assert.equal(calls.groups.length, 1);
  await lease.whenIdle(); owner.dispose();
  assert.throws(() => lease.assertCurrent(), {code: 'GPU_COMPUTE_DISPOSED'});
});

test('disposal clears cache references and prevents reuse', () => {
  const {owner, pipeline, a} = setup();
  owner.dispatch(pipeline, group(a), [1]); owner.dispose();
  assert.equal(owner.cachedBindGroupCount, 0); assert.equal(owner.bufferCount, 0);
  assert.throws(() => owner.dispatch(pipeline, group(a), [1]), {code: 'GPU_COMPUTE_DISPOSED'});
});

test('device loss clears the cache and retains the terminal error', async () => {
  const {owner, pipeline, a, lose} = setup();
  owner.dispatch(pipeline, group(a), [1]); lose({message: 'lost recorder'}); await flush();
  assert.equal(owner.cachedBindGroupCount, 0);
  await assert.rejects(owner.whenIdle(), {code: 'GPU_COMPUTE_DEVICE_LOST'});
});

test('scoped GPU validation failure clears cached groups', async () => {
  const {owner, pipeline, a, device} = setup();
  await owner.whenIdle();
  device.error = {message: 'invalid bind group'};
  owner.dispatch(pipeline, group(a), [1]);
  await assert.rejects(owner.whenIdle(), {code: 'GPU_COMPUTE_VALIDATION'});
  assert.equal(owner.cachedBindGroupCount, 0); assert.equal(a.destroyed, true);
});

test('synchronous creation errors are not inserted into the cache', () => {
  const {owner, pipeline, a, device, calls} = setup();
  device.throwGroup = true;
  assert.throws(() => owner.dispatch(pipeline, group(a), [1]), /bind group creation failed/);
  assert.equal(owner.cachedBindGroupCount, 0); assert.equal(calls.submits.length, 0);
  owner.dispatch(pipeline, group(a), [1]); assert.equal(calls.groups.length, 1); owner.dispose();
});

test('disposal during group creation cannot republish cache entries or submit', () => {
  const {owner, pipeline, a, device, calls} = setup();
  device.onCreateGroup = () => owner.dispose();
  assert.throws(() => owner.dispatch(pipeline, group(a), [1]), {code: 'GPU_COMPUTE_DISPOSED'});
  assert.equal(owner.cachedBindGroupCount, 0); assert.equal(calls.submits.length, 0);
});

test('empty bind groups are cacheable without allocating any native buffers', () => {
  const {device, calls} = gpu(), owner = createGpuCompute(device);
  const pipeline = owner.prepareSync({code: 'empty'});
  for (let i = 0; i < 3; i++) owner.dispatch(pipeline, [{index: 0, entries: []}], [1]);
  assert.equal(calls.groups.length, 1); assert.equal(calls.buffers.length, 0); owner.dispose();
});

test('programs without bind groups retain direct dispatch behavior', () => {
  const {owner, pipeline, calls} = setup();
  owner.dispatch(pipeline, [], [0, 2, 1]);
  assert.equal(calls.groups.length, 0); assert.deepEqual(calls.passes[0].direct, [0, 2, 1]); owner.dispose();
});

test('cache eviction does not destroy borrowed texture or sampler resources', () => {
  const {owner, pipeline} = setup({maxCachedBindGroups: 1});
  let destroyed = 0;
  const first = {...tagged('GPUTextureView'), destroy() { destroyed++; }};
  const second = tagged('GPUTextureView');
  for (const resource of [first, second]) owner.dispatch(pipeline, [{index: 0, entries: [{binding: 0, resource}]}], [1]);
  owner.dispose(); assert.equal(destroyed, 0);
});
