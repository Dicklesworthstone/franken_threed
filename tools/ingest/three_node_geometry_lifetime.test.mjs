/** Pinned native geometry disposal and upload histories. The actual source
 * Geometries/Attributes/WebGPUAttributeUtils run against a recording device;
 * these checks compare buffer contents and ownership, not shader execution. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceAttributes from '../../upstream/three.js/src/renderers/common/Attributes.js';
import SourceGeometries from '../../upstream/three.js/src/renderers/common/Geometries.js';
import SourceAttributeUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
import {createThreeNodeBundleGeometry} from './three_node_bundle_geometry.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function geometry(indexed = false) {
  const value = new T.BufferGeometry().setAttribute('position',
    new T.Float32BufferAttribute([0, .85, 0, 1, 0, 0, 0, 1, 0], 3));
  if (indexed) value.setIndex(new T.BufferAttribute(new Uint32Array([0, 1, 2]), 1));
  return value;
}
function sourceOwner(t) {
  const device = geometryDevice(), records = new WeakMap();
  const backend = {device, get(attribute) {
    let record = records.get(attribute);
    if (!record) { record = {}; records.set(attribute, record); }
    return record;
  }, delete: attribute => records.delete(attribute)};
  const utils = new SourceAttributeUtils(backend);
  backend.createAttribute = attribute => utils.createAttribute(attribute, 32 | 8);
  backend.createIndexAttribute = attribute => utils.createAttribute(attribute, 16 | 8);
  backend.updateAttribute = attribute => utils.updateAttribute(attribute);
  backend.destroyAttribute = attribute => utils.destroyAttribute(attribute);
  const info = {memory: {geometries: 0}, render: {calls: 0}, createAttribute() {}, createIndexAttribute() {}, destroyAttribute() {}};
  const attributes = new SourceAttributes(backend, info), manager = new SourceGeometries(attributes, info);
  t.after(() => { manager.dispose(); for (const buffer of device.buffers) buffer.destroy(); });
  return {device, manager, info,
    render(geometry, renderId) {
      info.render.calls = renderId;
      manager.updateForRender({geometry, material: {wireframe: false}, getAttributes: () => Object.values(geometry.attributes)});
    },
    buffer(attribute) {
      const buffer = backend.get(attribute.isInterleavedBufferAttribute ? attribute.data : attribute).buffer;
      assert.ok(buffer && !buffer.destroyed, 'source allocation is live'); return buffer;
    }};
}
function histories(t) {
  const owner = createThreeNodeBundleGeometry(T); t.after(() => owner.dispose()); return owner;
}
function assertUploaded(source, original, captured) {
  const compare = (attribute, view) => {
    const array = view.isInterleavedBufferAttribute ? view.data.array : view.array;
    assert.deepEqual(new Uint8Array(source.buffer(attribute).data, 0, array.byteLength),
      new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
  };
  for (const [name, attribute] of Object.entries(original.attributes)) compare(attribute, captured.attributes[name]);
  if (original.index) compare(original.index, captured.index);
}

for (const indexed of [false, true]) test(`native ${indexed ? 'indexed' : 'nonindexed'} histories preserve partial uploads after repeated disposal`, t => {
  const source = sourceOwner(t), owner = histories(t), original = geometry(indexed), candidate = original.clone(), held = [];
  t.after(() => { for (const packet of held) packet.release(); });
  const both = change => { change(original); change(candidate); };
  const capture = renderId => {
    source.render(original, renderId); const packet = owner.capture(candidate, renderId); held.push(packet);
    assertUploaded(source, original, packet.geometry); return packet;
  };
  const first = capture(1);
  both(value => { value.attributes.position.array[0] = .3; value.dispose(); });
  assert.equal(source.manager.has({geometry: original}), true, 'r186 retains its initialized geometry record');
  assert.equal(source.info.memory.geometries, 0); assert.equal(source.manager._geometryDisposeListeners.size, 0);
  const restored = capture(2);
  assert.equal(restored.geometry.attributes.position.array[0], Math.fround(.3));
  assert.equal(first.geometry.attributes.position.array[0], 0, 'accepted earlier data remains immutable');
  both(value => {
    const position = value.attributes.position;
    position.array[1] = .65; position.addUpdateRange(1, 1); position.needsUpdate = true;
    if (value.index) { value.index.array[0] = 2; value.index.addUpdateRange(0, 1); value.index.needsUpdate = true; }
  });
  const changed = capture(3), buffers = source.device.buffers.length, bytes = owner.diagnostics.allocatedBytes;
  both(value => { value.attributes.position.array[1] = .85; if (value.index) value.index.array[0] = 0; });
  for (let renderId = 4; renderId <= 5; renderId++) {
    both(value => value.dispose());
    assert.equal(owner.diagnostics.allocatedBytes, bytes, 'later disposal retains the source upload history');
    const retained = capture(renderId);
    assert.equal(retained.geometry, changed.geometry);
    assert.equal(retained.geometry.attributes.position.array[1], Math.fround(.65));
    if (indexed) assert.equal(retained.geometry.index.array[0], 2);
  }
  assert.equal(source.device.buffers.length, buffers, 'the source does not recreate buffers or rearm its listener');
  assert.equal(source.manager._geometryDisposeListeners.size, 0);
  both(value => {
    value.attributes.position.array[0] = .45;
    value.attributes.position.addUpdateRange(0, 1); value.attributes.position.needsUpdate = true;
  });
  const next = capture(6);
  assert.equal(next.geometry.attributes.position.array[0], Math.fround(.45));
  assert.equal(next.geometry.attributes.position.array[1], Math.fround(.65), 'later partial updates still preserve unmarked bytes');
  owner.dispose();
  for (const packet of held) packet.release();
  assert.deepEqual(owner.diagnostics, {allocatedBytes: 0, owners: 0, geometries: 0, maxBytes: 64 * 1024 * 1024});
});

test('all initial private views invalidate once and later views share their canonical source lifetime', t => {
  const owner = histories(t), original = geometry(), held = [];
  t.after(() => { for (const packet of held) packet.release(); });
  const view = () => ({geometry: original, attributes: original.attributes, index: original.index, drawRange: original.drawRange});
  held.push(owner.capture(view(), 1), owner.capture(view(), 1));
  original.attributes.position.array[0] = 7; original.dispose();
  assert.equal(owner.diagnostics.owners, 0);
  assert.equal(owner.diagnostics.geometries, 0, 'every previously registered wrapper receives the first event');
  const later = view(), fresh = owner.capture(later, 2); held.push(fresh);
  assert.equal(fresh.geometry.attributes.position.array[0], 7);
  original.attributes.position.array[0] = 9; original.dispose();
  const reused = owner.capture(later, 3), another = owner.capture(view(), 3); held.push(reused, another);
  assert.equal(reused.geometry, fresh.geometry);
  assert.equal(another.geometry.attributes.position.array[0], 7, 'a new wrapper cannot rearm a disposed source');
  assert.equal(original._listeners.dispose.length, 0);
  owner.dispose(); for (const packet of held) packet.release();
  assert.equal(owner.diagnostics.allocatedBytes, 0);
});

test('a different geometry sharing an attribute retains its independent first disposal', t => {
  const source = sourceOwner(t), owner = histories(t), original = geometry(), candidate = original.clone();
  const sourceAlias = new T.BufferGeometry().setAttribute('position', original.attributes.position);
  const candidateAlias = new T.BufferGeometry().setAttribute('position', candidate.attributes.position), held = [];
  t.after(() => { for (const packet of held) packet.release(); });
  const capture = (a, b, renderId) => {
    source.render(a, renderId); const packet = owner.capture(b, renderId); held.push(packet);
    assertUploaded(source, a, packet.geometry); return packet;
  };
  capture(original, candidate, 1); capture(sourceAlias, candidateAlias, 1);
  for (const value of [original, candidate]) { value.attributes.position.array[0] = 3; value.dispose(); }
  const restored = capture(sourceAlias, candidateAlias, 2);
  capture(original, candidate, 2);
  for (const value of [original, candidate]) { value.attributes.position.array[0] = 5; value.dispose(); }
  assert.equal(capture(sourceAlias, candidateAlias, 3).geometry.attributes.position.array[0], 3);
  sourceAlias.dispose(); candidateAlias.dispose();
  assert.equal(capture(original, candidate, 4).geometry.attributes.position.array[0], 5,
    'the alias first disposal still invalidates shared residency');
  assert.equal(restored.geometry.attributes.position.array[0], 3);
});

test('a newly created renderer history observes its own first disposal', t => {
  const original = geometry(), candidate = original.clone(), firstSource = sourceOwner(t), first = histories(t), held = [];
  t.after(() => { for (const packet of held) packet.release(); });
  firstSource.render(original, 1); held.push(first.capture(candidate, 1));
  original.dispose(); candidate.dispose();
  firstSource.render(original, 2); held.push(first.capture(candidate, 2));
  const secondSource = sourceOwner(t), second = histories(t);
  secondSource.render(original, 1); held.push(second.capture(candidate, 1));
  original.attributes.position.array[0] = candidate.attributes.position.array[0] = 4;
  original.dispose(); candidate.dispose();
  firstSource.render(original, 3); const unchanged = first.capture(candidate, 3); held.push(unchanged);
  secondSource.render(original, 2); const fresh = second.capture(candidate, 2); held.push(fresh);
  assertUploaded(firstSource, original, unchanged.geometry); assertUploaded(secondSource, original, fresh.geometry);
  assert.equal(unchanged.geometry.attributes.position.array[0], 0);
  assert.equal(fresh.geometry.attributes.position.array[0], 4);
});

test('private GPU residency recreated by a later event uses accepted bytes, and owner shutdown releases histories', t => {
  const owner = histories(t), original = geometry(), first = owner.capture(original, 1);
  original.dispose(); first.release();
  original.attributes.position.array[1] = .65;
  const accepted = owner.capture(original, 2), device = geometryDevice();
  const gpu = createGpuProgramGeometry(device, accepted.geometry,
    [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}], {attributeTypes: 'webgpu'});
  t.after(() => { gpu.dispose(); accepted.release(); });
  original.attributes.position.array[1] = .85; original.dispose();
  assert.throws(() => programGeometrySnapshot(gpu, device), {code: 'GEOMETRY_GPU_RELEASED'});
  gpu.update();
  assert.equal(new Float32Array(programGeometrySnapshot(gpu, device).vertexBuffers[0].data)[1], Math.fround(.65));
  owner.dispose(); assert.equal(owner.diagnostics.allocatedBytes, 36, 'the accepted lease still owns its bytes');
  accepted.release(); assert.equal(owner.diagnostics.allocatedBytes, 0);
});
