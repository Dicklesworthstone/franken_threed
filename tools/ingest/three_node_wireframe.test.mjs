import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodeWireframes} from './three_node_wireframe.mjs';

const edges = [0, 1, 1, 2, 2, 0, 2, 1, 1, 3, 3, 2];
function geometry(indexed = true) {
  const value = new T.BufferGeometry();
  value.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(18), 3));
  if (indexed) value.setIndex([0, 1, 2, 2, 1, 3]);
  return value;
}
function owner(t, options) {
  const value = createThreeNodeWireframes(T, options); t.after(() => value.dispose()); return value;
}

test('wireframe captures source CPU edges at version boundaries without consuming original index ranges', t => {
  const histories = owner(t), source = geometry(), descriptor = {};
  const view = histories.register(descriptor, source), index = view.index;
  source.index.array[0] = 3;
  const first = histories.capture(descriptor); first.install();
  assert.deepEqual(Array.from(index.array), [3, 1, 1, 2, 2, 3, ...edges.slice(6)]);
  assert.equal(first.bytes, 24); assert.equal(first.indexCount, 12); assert.equal(histories.byteLength(descriptor), 24);
  // Dynamic usage alone does not invalidate Geometries' derived-index cache.
  source.index.setUsage(T.DynamicDrawUsage); source.index.array[0] = 0;
  histories.capture(descriptor).install();
  assert.equal(index.array[0], 3);
  source.index.array[5] = 4; source.index.addUpdateRange(0, 1); source.index.needsUpdate = true;
  histories.capture(descriptor).install();
  assert.deepEqual(Array.from(index.array), [0, 1, 1, 2, 2, 0, 2, 1, 1, 4, 4, 2]);
  assert.deepEqual(source.index.updateRanges, [{start: 0, count: 1}], 'the full CPU index drives derived edges');
  assert.equal(view.index, index, 'version changes reuse one private index owner');
  source.index.array[5] = 5; source.index.version--;
  histories.capture(descriptor).install(); assert.equal(index.array[9], 5, 'source compares version inequality');
});

test('nonindexed wireframes retain source loop extents and compose generated instance streams', t => {
  const histories = owner(t), source = geometry(false), descriptor = {};
  source.setDrawRange(1, 3);
  const attributes = {...source.attributes, instanceNode: new T.InstancedBufferAttribute(new Float32Array(8), 4)};
  const base = {attributes, morphAttributes: {position: []}, isInstancedBufferGeometry: true};
  const view = histories.register(descriptor, source, base);
  histories.capture(descriptor).install();
  assert.deepEqual(Array.from(view.index.array), [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3]);
  assert.deepEqual(view.drawRange, {start: 2, count: 6});
  assert.equal(view.attributes, attributes); assert.equal(view.morphAttributes, base.morphAttributes);
  assert.equal(view.isInstancedBufferGeometry, true); assert.equal(source.index, null);
  // The pinned loop reads array.length / 3, rather than position.count.
  source.attributes.position.count = 3;
  source.attributes.position.needsUpdate = true;
  histories.capture(descriptor).install(); assert.equal(view.index.count, 12);
});

test('queued wireframe packets retain independent edge arrays and source identity replacements rebuild', t => {
  const histories = owner(t), source = geometry(), descriptor = {};
  const view = histories.register(descriptor, source), privateIndex = view.index;
  const first = histories.capture(descriptor);
  assert.equal(histories.capture(descriptor).geometry, first.geometry, 'one geometry view per accepted edge snapshot');
  source.setIndex([3, 2, 1, 0, 2, 3]);
  const second = histories.capture(descriptor);
  assert.notEqual(second.geometry, first.geometry);
  assert.notEqual(second.geometry.index, first.geometry.index);
  assert.ok(Object.isFrozen(first.geometry));
  source.index.array.fill(4);
  first.install(); assert.deepEqual(Array.from(privateIndex.array), edges);
  const version = privateIndex.version; first.install(); assert.equal(privateIndex.version, version);
  second.install(); assert.deepEqual(Array.from(privateIndex.array), [3, 2, 2, 1, 1, 3, 0, 2, 2, 3, 3, 0]);
  first.install(); assert.deepEqual(Array.from(privateIndex.array), edges);
  histories.release(descriptor); second.install(); assert.equal(view.index, privateIndex);
  assert.deepEqual(Array.from(first.geometry.index.array), edges, 'another group install cannot replace earlier captured edges');
  assert.deepEqual(Array.from(second.geometry.index.array), [3, 2, 2, 1, 1, 3, 0, 2, 2, 3, 3, 0]);
  const other = {}; histories.register(other, source); histories.capture(other).install();
  assert.deepEqual(Array.from(histories.geometry(other).index.array), Array.from(privateIndex.array));
});

test('wireframe layout changes need a new descriptor and preserve earlier accepted packets', t => {
  const histories = owner(t), source = geometry(), descriptor = {};
  const previous = histories.register(descriptor, source), first = histories.capture(descriptor);
  source.setIndex([0, 1, 2]);
  assert.throws(() => histories.capture(descriptor), {code: 'THREE_NODE_WIREFRAME_CHANGED'});
  const short = {}, view = histories.register(short, source); histories.capture(short).install();
  assert.equal(view.index.count, 6); assert.ok(view.index.array instanceof Uint16Array);
  source.attributes.position.count = 65535; source.index.needsUpdate = true;
  assert.throws(() => histories.capture(short), {code: 'THREE_NODE_WIREFRAME_CHANGED'});
  const wide = {}; histories.register(wide, source); histories.capture(wide).install();
  assert.ok(histories.geometry(wide).index.array instanceof Uint32Array);
  first.install(); assert.deepEqual(Array.from(previous.index.array), edges);
});

test('source disposal releases wireframe history and subsequent use rebuilds from live CPU arrays', t => {
  const histories = owner(t), source = geometry(), descriptor = {};
  const view = histories.register(descriptor, source), pending = histories.capture(descriptor);
  pending.install(); source.index.array[0] = 3;
  source.dispose(); assert.equal(histories.diagnostics.allocatedBytes, 0); assert.equal(histories.diagnostics.geometries, 0);
  pending.install(); assert.equal(view.index.array[0], 0, 'already accepted bytes survive source disposal');
  histories.capture(descriptor).install(); assert.equal(view.index.array[0], 3);
  assert.equal(histories.diagnostics.allocatedBytes, 24);
});

test('wireframe history budgets reject allocation before deriving or changing source state', t => {
  const small = owner(t, {maxBytes: 20}), source = geometry();
  assert.throws(() => small.register({}, source), {code: 'THREE_NODE_WIREFRAME_LIMIT'});
  assert.deepEqual(Array.from(source.index.array), [0, 1, 2, 2, 1, 3]);
  const histories = owner(t, {maxBytes: 40}), descriptor = {};
  histories.register(descriptor, source); histories.capture(descriptor);
  assert.throws(() => histories.register({}, geometry()), {code: 'THREE_NODE_WIREFRAME_LIMIT'});
  assert.equal(histories.diagnostics.allocatedBytes, 24);
});
