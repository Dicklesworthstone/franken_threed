import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeTimestampQueries} from './three_timestamp_queries.mjs';

const deferred = () => { let resolve, reject; const promise = new Promise((done, fail) => { resolve = done; reject = fail; }); return {promise, resolve, reject}; };
function fixture(t, options = {}) {
  const lost = deferred(), queries = [], buffers = [], scopes = [];
  const d = {features: new Set(['timestamp-query']), lost: lost.promise, scopes,
    pushErrorScope(type) { scopes.push(type); }, popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(d.invalid ?? null); },
    createQuerySet(descriptor) {
      const query = {...descriptor, values: [1000000n, 1750000n], destroy() {this.destroyed = true;}};
      queries.push(query); return query;
    },
    createBuffer(descriptor) {
      const buffer = {...descriptor, data: new ArrayBuffer(descriptor.size),
        async mapAsync() { if (d.map) await d.map; }, getMappedRange() {return this.data;},
        unmap() {this.unmapped = true;}, destroy() {this.destroyed = true;}};
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() { return {
      resolveQuerySet(query, start, count, buffer, offset) {new BigUint64Array(buffer.data, offset, count).set(query.values.slice(start, start + count));},
      copyBufferToBuffer(source, from, target, to, size) {new Uint8Array(target.data, to, size).set(new Uint8Array(source.data, from, size));},
      finish() { return {}; },
    }; },
    queue: {submit() {}},
  };
  const owner = createThreeTimestampQueries({getDevice: () => d, enabled: () => true, ...options});
  t.after(() => owner.dispose());
  return {owner, d, queries, buffers, lost};
}

test('one logical call writes start once, preserves its last pass end and publishes milliseconds', async t => {
  const f = fixture(t), context = {}, token = f.owner.begin('render', 'r:1:f5', context);
  const first = token.writes(), last = token.writes();
  assert.equal(first.querySet, last.querySet);
  assert.deepEqual([first.beginningOfPassWriteIndex, last.beginningOfPassWriteIndex, last.endOfPassWriteIndex], [0, undefined, 1]);
  assert.equal(f.owner.getTimestampUID(context), 'r:1:f5');
  assert.equal(f.owner.hasTimestampQuery('r:1:f5'), false);
  assert.equal(await f.owner.resolve(), .75);
  assert.equal(f.owner.getTimestamp('r:1:f5'), .75);
  assert.deepEqual(f.owner.getTimestampFrames('render'), [5]);
  assert.equal(f.owner.pendingCount, 0);
  assert.ok(f.queries.every(query => query.destroyed)); assert.ok(f.buffers.every(buffer => buffer.destroyed && (!(buffer.usage & 1) || buffer.unmapped)));
});

test('render and compute snapshots remain independent while later calls continue during mapping', async t => {
  const f = fixture(t), gate = deferred(); f.d.map = gate.promise;
  f.owner.begin('render', 'r:1:f1').writes(); f.owner.begin('compute', 'c:1:f1').writes();
  const first = f.owner.resolve('render');
  f.owner.begin('render', 'r:2:f2').writes();
  assert.equal(f.owner.resolve('render'), first);
  gate.resolve(); await first;
  assert.deepEqual(f.owner.getTimestampFrames('render'), [1]);
  assert.equal(f.owner.hasTimestampQuery('r:2:f2'), false);
  assert.equal(await f.owner.resolve('compute'), .75);
  assert.deepEqual(f.owner.getTimestampFrames('compute'), [1]);
  assert.equal(await f.owner.resolve('render'), .75);
  assert.deepEqual(f.owner.getTimestampFrames('render'), [2]);
  assert.equal(f.owner.hasTimestampQuery('r:1:f1'), false);
});

test('capacity exhaustion preserves draws and reports unavailable calls without overwriting live queries', async t => {
  const f = fixture(t, {maxQueries: 2, maxFrames: 2});
  const first = f.owner.begin('render', 'r:1:f1'); first.writes();
  for (let frame = 2; frame <= 4; frame++) assert.equal(f.owner.begin('render', `r:1:f${frame}`).writes(), undefined);
  assert.equal(f.queries.length, 1); assert.equal(f.owner.pendingCount, 1);
  assert.equal(await f.owner.resolve(), 0);
  assert.deepEqual(f.owner.getTimestampFrames('render'), [3, 4]);
  assert.equal(f.owner.hasTimestampQuery('r:1:f4'), false);
  f.owner.begin('compute', 'c:1:f5').writes();
  assert.equal(await f.owner.resolve('compute'), .75);
});

test('query validation and device loss reject maps and release every native allocation', async t => {
  {
    const f = fixture(t); f.d.invalid = {message: 'invalid timestamp query'};
    f.owner.begin('render', 'r:1:f1').writes();
    await assert.rejects(f.owner.resolve(), /invalid timestamp query/);
    assert.equal(f.owner.pendingCount, 0); assert.ok(f.queries.every(query => query.destroyed));
  }
  {
    const f = fixture(t); f.d.map = new Promise(() => {});
    f.owner.begin('compute', 'c:1:f2').writes();
    const pending = f.owner.resolve('compute'); pending.catch(() => {});
    f.lost.resolve({message: 'GPU removed'});
    await assert.rejects(pending, /GPU removed/);
    assert.equal(f.owner.pendingCount, 0); assert.ok(f.buffers.every(buffer => buffer.destroyed));
  }
});

test('delayed validation of a measured pass must succeed before timestamps become available', async t => {
  const f = fixture(t), validation = deferred(), token = f.owner.begin('compute', 'c:1:f2');
  token.writes(); token.validate(validation.promise);
  const pending = f.owner.resolve('compute'); pending.catch(() => {});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.owner.hasTimestampQuery('c:1:f2'), false);
  validation.reject(Error('invalid measured dispatch'));
  await assert.rejects(pending, /invalid measured dispatch/);
  assert.equal(f.owner.pendingCount, 0);
  assert.ok(f.queries.every(query => query.destroyed));
});
