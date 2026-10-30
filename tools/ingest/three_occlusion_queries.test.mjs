import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeOcclusionQueries} from './three_occlusion_queries.mjs';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
}
function fixture(t, options = {}) {
  const lost = deferred(), querySets = [], buffers = [], maps = [], encoded = [];
  let allocationFailure = null;
  const device = {
    lost: lost.promise,
    createQuerySet(descriptor) {
      const query = {...descriptor, samples: new BigUint64Array(descriptor.count), destroyed: 0,
        destroy() {this.destroyed++;}};
      querySets.push(query); return query;
    },
    createBuffer(descriptor) {
      if (allocationFailure?.(descriptor)) throw new Error('allocation failed');
      const buffer = {...descriptor, bytes: new ArrayBuffer(descriptor.size), destroyed: 0, unmapped: 0, reads: 0,
        destroy() {this.destroyed++;},
        mapAsync(mode) {
          assert.equal(mode, 1); const mapping = deferred(); maps.push({buffer: this, ...mapping});
          return mapping.promise;
        },
        getMappedRange() {this.reads++; return this.bytes;},
        unmap() {this.unmapped++;},
      };
      buffers.push(buffer); return buffer;
    },
  };
  const encoder = {
    resolveQuerySet(query, start, count, target, offset) {
      encoded.push(['resolve', start, count, offset]);
      new BigUint64Array(target.bytes, offset, count).set(query.samples.subarray(start, start + count));
    },
    copyBufferToBuffer(source, sourceOffset, target, targetOffset, length) {
      encoded.push(['copy', sourceOffset, targetOffset, length]);
      new Uint8Array(target.bytes, targetOffset, length).set(new Uint8Array(source.bytes, sourceOffset, length));
    },
  };
  const owner = createThreeOcclusionQueries({getDevice: () => device, ...options});
  t.after(() => owner.dispose());
  return {owner, device, lost, querySets, buffers, maps, encoded, encoder,
    failAllocation(predicate) {allocationFailure = predicate;}};
}
const command = (overrides = {}) => ({nodeProgram: {}, instanceCount: 1, count: 3, ...overrides});
function frame(f, state, objects, values, options = {}) {
  const session = f.owner.begin(state, {requested: true, ...options});
  const plan = session.plan(objects.map(object => ({object, tested: true, bundle: false})),
    objects.map(() => command()), objects.length);
  if (plan) {plan.querySet.samples.set(values); plan.resolve(f.encoder);}
  session.commit(); return {session, plan};
}

test('previous-frame uint64 results publish after a successor, while waiting alone never maps the latest frame', async t => {
  const f = fixture(t), state = f.owner.context(null), hidden = {}, visible = {};
  const first = frame(f, state, [hidden, visible], [0n, (1n << 48n) + 7n]);
  assert.equal(state.occluded, undefined); assert.equal(f.maps.length, 0);
  await f.owner.whenIdle(); assert.equal(f.maps.length, 0); assert.equal(f.owner.pendingCount, 1);
  assert.deepEqual(f.encoded, [['resolve', 0, 2, 0], ['copy', 0, 0, 16]]);
  assert.equal(f.buffers[0].usage, 512 | 4); assert.equal(f.buffers[1].usage, 8 | 1);
  assert.deepEqual([...new BigUint64Array(f.buffers[1].bytes)], [0n, (1n << 48n) + 7n]);
  const second = frame(f, state, [hidden, visible], [1n, 0n]);
  assert.equal(f.maps.length, 1); assert.equal(state.occluded, undefined);
  f.maps[0].resolve(); await f.owner.whenIdle();
  assert.equal(state.occluded.has(hidden), true); assert.equal(state.occluded.has(visible), false);
  assert.equal(first.plan.querySet.destroyed, 1); assert.equal(second.plan.querySet.destroyed, 0);
  assert.equal(f.maps[0].buffer.unmapped, 1); assert.equal(f.owner.pendingCount, 1);
  frame(f, state, [hidden, visible], [9n, 9n]);
  assert.equal(state.occluded.has(hidden), true);
  f.maps[1].resolve(); await f.owner.whenIdle();
  assert.equal(state.occluded.has(hidden), false); assert.equal(state.occluded.has(visible), true);
});

test('render-context sharing follows attachment signature, MRT and nesting, independent of identity and dimensions', t => {
  const f = fixture(t);
  const target = () => ({textures: [{}], texture: {format: 1023, type: 1009}, samples: 4,
    depthBuffer: true, stencilBuffer: false, width: 16, height: 16});
  const a = target(), b = {...target(), width: 1024, height: 512, activeCubeFace: 3, activeMipmapLevel: 2};
  assert.equal(f.owner.context(a), f.owner.context(b));
  const baseline = f.owner.context(a), variants = [
    {...target(), textures: [{}, {}]}, {...target(), texture: {format: 1021, type: 1009}},
    {...target(), texture: {format: 1023, type: 1015}}, {...target(), samples: 1},
    {...target(), depthBuffer: false}, {...target(), stencilBuffer: true},
  ];
  for (const variant of variants) assert.notEqual(f.owner.context(variant), baseline);
  assert.notEqual(f.owner.context(null), baseline);
  assert.equal(f.owner.context(a, {id: 11}), f.owner.context(b, {id: 11}));
  assert.notEqual(f.owner.context(a, {id: 11}), f.owner.context(a, {id: 12}));
  assert.notEqual(f.owner.context(a, null, 1), baseline);
  assert.equal(f.owner.context(a, null, 1), f.owner.context(b, null, 1));
});

test('contiguous object groups share a query, source bundles inherit it, and internal draw spans retain ordinary queries', t => {
  const f = fixture(t), state = f.owner.context(null), a = {}, b = {}, c = {}, bundled = {}, empty = {};
  const objects = [
    {object: a, tested: true}, {object: a, tested: true}, {object: b, tested: false},
    {object: c, tested: true}, {object: bundled, tested: true, bundle: true},
    {object: empty, tested: true}, {object: a, tested: true},
  ];
  const commands = objects.map(() => command()); commands[5].instanceCount = 0;
  // Internal spans retain ordinary per-object identities here; only the source
  // BundleGroup marker suppresses a query for the authored bundled draw.
  objects[6].bundle = false;
  const session = f.owner.begin(state, {requested: true}), plan = session.plan(objects, commands, objects.length);
  assert.deepEqual(plan.indices, [0, 0, null, 1, 1, 1, 2]); assert.equal(plan.querySet.count, 3);
  session.cancel(); assert.equal(plan.querySet.destroyed, 1); assert.equal(f.owner.pendingCount, 0);
});

test('empty native and skipped indirect commands do not change query grouping; live indirect draws do', t => {
  const f = fixture(t), a = {}, skipped = {}, indirect = {}, state = f.owner.context(null);
  const objects = [a, skipped, skipped, a, indirect].map(object => ({object, tested: true}));
  const commands = [command(), command({count: 0}), command({nodeIndirect: {skip: true}}),
    command(), command({count: 0, nodeIndirect: {skip: false}})];
  const session = f.owner.begin(state, {requested: true}), plan = session.plan(objects, commands, objects.length);
  assert.deepEqual(plan.indices, [0, 0, 0, 0, 1]); assert.equal(plan.querySet.count, 2);
  session.cancel();
  const onlyBundles = f.owner.begin(state, {requested: true});
  const bundled = onlyBundles.plan([{object: a, tested: true, bundle: true}], [command()], 1);
  assert.deepEqual(bundled.indices, [null]); assert.equal(bundled.querySet.count, 1);
  onlyBundles.cancel();
});

test('disabled query requests leave history alone; requested empty and ArrayCamera frames rotate previous results', async t => {
  const f = fixture(t), state = f.owner.context(null), object = {};
  frame(f, state, [object], [0n]);
  assert.equal(f.owner.begin(state), null); await f.owner.whenIdle(); assert.equal(f.maps.length, 0);
  const array = frame(f, state, [object], [], {arrayCamera: true});
  assert.deepEqual(array.plan.indices, [null]); assert.equal(array.plan.querySet.count, 1);
  assert.equal(f.querySets.length, 2); assert.equal(f.maps.length, 1);
  f.maps[0].resolve(); await f.owner.whenIdle(); assert.equal(state.occluded.has(object), true);
  const empty = frame(f, state, [], []);
  assert.deepEqual(empty.plan.indices, []); assert.equal(empty.plan.querySet.count, 1);
  assert.equal(state.occluded.has(object), true); assert.equal(f.maps.length, 2);
  f.maps[1].resolve(); await f.owner.whenIdle();
  assert.equal(state.occluded.has(object), false);
});

test('unassociated readbacks preserve GPU completion ordering across overlapping empty and ordinary frames', async t => {
  const f = fixture(t), state = f.owner.context(null), firstObject = {}, newestObject = {};
  frame(f, state, [firstObject], [0n]);
  frame(f, state, [firstObject], [], {arrayCamera: true});
  frame(f, state, [newestObject], [0n]);
  assert.equal(f.maps.length, 2); assert.equal(state.occluded, undefined);
  f.maps[0].resolve(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(state.occluded.has(firstObject), true);
  let idle = false; const waiting = f.owner.whenIdle().then(() => {idle = true;});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(idle, false); assert.equal(state.occluded.has(firstObject), true);
  f.maps[1].resolve(); await waiting;
  assert.equal(state.occluded.has(firstObject), false); assert.equal(state.occluded.has(newestObject), false);
  frame(f, state, [], []); assert.equal(f.maps.length, 3);
  f.maps[2].resolve(); await f.owner.whenIdle();
  assert.equal(state.occluded.has(newestObject), true);
});

test('separate pass parts retain their object associations and publish together after every initiated mapping completes', async t => {
  const f = fixture(t), state = f.owner.context(null), hidden = {}, visible = {}, objects = [{object: hidden, tested: true}];
  const session = f.owner.begin(state, {requested: true});
  const first = session.plan(objects, [command()], 1); first.querySet.samples[0] = 0n; first.resolve(f.encoder);
  objects[0] = {object: visible, tested: true};
  const second = session.plan(objects, [command()], 1); second.querySet.samples[0] = 3n; second.resolve(f.encoder);
  const tail = session.plan([], [], 0); tail.resolve(f.encoder);
  assert.equal(tail.querySet.count, 1); assert.deepEqual(tail.indices, []);
  session.commit(); frame(f, state, [], []); assert.equal(f.maps.length, 3);
  f.maps[1].resolve(); f.maps[0].resolve();
  await new Promise(resolve => setImmediate(resolve)); assert.equal(state.occluded, undefined);
  let idle = false; const waiting = f.owner.whenIdle().then(() => {idle = true;});
  await Promise.resolve(); assert.equal(idle, false);
  f.maps[2].resolve(); await waiting;
  assert.equal(state.occluded.has(hidden), true); assert.equal(state.occluded.has(visible), false);
  assert.equal(first.querySet.destroyed, 1); assert.equal(second.querySet.destroyed, 1); assert.equal(tail.querySet.destroyed, 1);
});

test('context, cumulative query and pending-history limits reject before extra allocations and recover after retirement', async t => {
  const f = fixture(t, {maxContexts: 1, maxQueries: 2, maxPending: 2}), state = f.owner.context(null), a = {}, b = {};
  assert.throws(() => f.owner.context(null, null, 1), /THREE_OCCLUSION_LIMIT/);
  assert.throws(() => f.owner.begin({}, {requested: true}), /THREE_OCCLUSION_CONTEXT/);
  const session = f.owner.begin(state, {requested: true});
  session.plan([{object: a, tested: true}], [command()], 1);
  assert.throws(() => session.plan([{object: a, tested: true}, {object: b, tested: true}], [command(), command()], 2), /THREE_OCCLUSION_LIMIT/);
  assert.equal(f.querySets.length, 1); session.cancel();
  frame(f, state, [a], [0n]); frame(f, state, [b], [1n]);
  assert.throws(() => f.owner.begin(state, {requested: true}), /THREE_OCCLUSION_LIMIT/);
  f.maps[0].resolve(); await f.owner.whenIdle(); assert.equal(f.owner.pendingCount, 1);
  const next = f.owner.begin(state, {requested: true}); next.cancel(); assert.equal(f.owner.pendingCount, 1);
});

test('one GPU query set never exceeds the WebGPU count limit even when the render budget is larger', t => {
  const f = fixture(t), session = f.owner.begin(f.owner.context(null), {requested: true});
  const objects = Array.from({length: 4097}, () => ({object: {}, tested: true}));
  assert.throws(() => session.plan(objects, objects.map(() => command()), objects.length), /THREE_OCCLUSION_LIMIT/);
  assert.equal(f.querySets.length, 0); assert.equal(f.buffers.length, 0); session.cancel();
});

test('cancelled preparation keeps the previous result and partial GPU allocations are released exactly once', async t => {
  const f = fixture(t), state = f.owner.context(null), object = {};
  const previous = frame(f, state, [object], [0n]);
  f.failAllocation(descriptor => descriptor.label === 'f3d-occlusion-readback');
  const failed = f.owner.begin(state, {requested: true});
  assert.throws(() => failed.plan([{object, tested: true}], [command()], 1), /allocation failed/);
  failed.cancel(); failed.cancel();
  assert.equal(f.querySets[1].destroyed, 1); assert.equal(f.buffers.at(-1).destroyed, 1);
  assert.equal(previous.plan.querySet.destroyed, 0); assert.equal(f.maps.length, 0);
  assert.throws(() => failed.commit(), /THREE_OCCLUSION_STALE/);
  f.failAllocation(null); frame(f, state, [], []); f.maps[0].resolve(); await f.owner.whenIdle();
  assert.equal(state.occluded.has(object), true);
});

test('mapping rejection terminalizes the owner and retires both old and latest resources', async t => {
  const f = fixture(t), state = f.owner.context(null), object = {}, error = new Error('GPU mapping failed');
  frame(f, state, [object], [0n]); frame(f, state, [object], [1n]);
  const waiting = f.owner.whenIdle(); f.maps[0].reject(error);
  await assert.rejects(waiting, value => value === error);
  assert.equal(state.occluded, undefined); assert.equal(f.owner.pendingCount, 0);
  assert.ok(f.querySets.every(query => query.destroyed === 1)); assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
  assert.throws(() => f.owner.context(null), value => value === error);
});

for (const reason of ['dispose', 'loss']) test(`${reason} prevents delayed mapping from reading or publishing destroyed history`, async t => {
  const f = fixture(t), state = f.owner.context(null), object = {};
  frame(f, state, [object], [0n]); frame(f, state, [object], [1n]);
  const rejection = reason === 'dispose' ? /THREE_OCCLUSION_DISPOSED/ : /THREE_OCCLUSION_LOST/;
  const waiting = assert.rejects(f.owner.whenIdle(), rejection);
  if (reason === 'dispose') f.owner.dispose(); else {f.lost.resolve({message: 'test device loss'}); await Promise.resolve();}
  await waiting; // Neither device loss nor disposal waits for an unresponsive map promise.
  f.maps[0].resolve(); await Promise.resolve(); await Promise.resolve();
  assert.equal(state.occluded, undefined); assert.equal(f.maps[0].buffer.reads, 0);
  assert.equal(f.owner.pendingCount, 0); assert.ok(f.querySets.every(query => query.destroyed === 1));
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
  await assert.rejects(f.owner.whenIdle(), rejection);
});
