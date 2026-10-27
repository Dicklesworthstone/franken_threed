/** Native wireframe command and upload histories against the pinned source.
 * Pixel execution is covered by the corresponding browser scenarios. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import Geometries from '../../upstream/three.js/src/renderers/common/Geometries.js';
import RenderObject from '../../upstream/three.js/src/renderers/common/RenderObject.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const frames = device => device.snapshots.map(frame => frame.filter(draw =>
  draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size)).filter(frame => frame.length);
const draws = device => frames(device).flat();
const indices = draw => {
  const bytes = draw.contents.get(draw.index.buffer), Type = draw.index.format === 'uint16' ? Uint16Array : Uint32Array;
  return [...new Type(bytes.buffer, bytes.byteOffset, bytes.byteLength / Type.BYTES_PER_ELEMENT)];
};
async function fixture(t, {geometry = new T.PlaneGeometry(1, 1),
  material = new T.MeshBasicNodeMaterial({wireframe: true}), mesh = null} = {}) {
  const device = geometryDevice(); device.limits.maxTextureDimension2D = 8192;
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => descriptor; device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose()); await renderer.init();
  mesh ??= new T.Mesh(geometry, material); mesh.frustumCulled = false;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 5;
  scene.add(mesh); scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const source = new Geometries({delete() {}}, {}); t.after(() => source.dispose());
  const expectedIndex = () => source.getIndex({geometry, material});
  const expectedRange = group => RenderObject.prototype.getDrawParameters.call({object: mesh, material, geometry,
    group, drawRange: geometry.drawRange, getIndex: expectedIndex});
  t.after(() => { geometry.dispose(); material.dispose(); });
  return {device, renderer, scene, camera, geometry, material, mesh, expectedIndex, expectedRange};
}

for (const indexed of [true, false]) test(`ordinary Standard ${indexed ? 'indexed' : 'nonindexed'} wireframe uses native source edges and ranges`, async t => {
  const base = new T.PlaneGeometry(1, 1), material = new T.MeshStandardMaterial({wireframe: true});
  const f = await fixture(t, {geometry: indexed ? base : base.toNonIndexed(), material});
  f.geometry.setDrawRange(3, 3);
  const expected = [...f.expectedIndex().array], range = f.expectedRange(null);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const [draw] = draws(f.device); assert.equal(draws(f.device).length, 1);
  assert.equal(draw.pipeline.primitive.topology, 'line-list'); assert.equal(draw.indexed, true);
  assert.deepEqual(indices(draw).slice(0, expected.length), expected);
  assert.deepEqual(draw.args.slice(0, 3), [range.vertexCount, range.instanceCount, range.firstVertex]);
  assert.equal(material.colorNode, undefined, 'ordinary source material needs no dummy node slot');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('wireframe groups keep distinct source ranges and line pipelines suppress polygon offset', async t => {
  const a = new T.MeshBasicNodeMaterial({wireframe: true, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 4});
  const b = a.clone(), geometry = new T.PlaneGeometry();
  geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 1);
  const f = await fixture(t, {geometry, material: a, mesh: new T.Mesh(geometry, [a, b])});
  t.after(() => b.dispose()); f.geometry.setDrawRange(1, 5);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device); assert.equal(submitted.length, 2);
  for (const [i, draw] of submitted.entries()) {
    const range = f.expectedRange(geometry.groups[i]);
    assert.deepEqual([draw.args[0], draw.args[2]], [range.vertexCount, range.firstVertex]);
    assert.equal(draw.pipeline.primitive.topology, 'line-list');
    assert.equal(draw.pipeline.depthStencil.depthBias ?? 0, 0);
    assert.equal(draw.pipeline.depthStencil.depthBiasSlopeScale ?? 0, 0);
  }
});

test('queued wireframe, fill and range changes retain each accepted topology and draw', async t => {
  const f = await fixture(t); f.material.colorNode = N.vec3(.2, .5, .8);
  f.renderer.render(f.scene, f.camera);
  f.material.wireframe = false; f.geometry.setDrawRange(3, 3); f.renderer.render(f.scene, f.camera);
  f.material.wireframe = true; f.geometry.setDrawRange(0, 3); f.renderer.render(f.scene, f.camera);
  f.material.wireframe = false; f.geometry.setDrawRange(0, 0);
  await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(draw => [draw.pipeline.primitive.topology, draw.args[0], draw.args[2]]),
    [['line-list', 12, 0], ['triangle-list', 3, 3], ['line-list', 6, 0]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('wireframe captures full requested CPU edge versions before ordinary node updates', async t => {
  const f = await fixture(t), original = [...f.expectedIndex().array]; let updates = 0;
  f.material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => {
    if (++updates === 1) {
      f.geometry.index.array.set([2, 1, 0, 3, 2, 0]);
      f.geometry.index.addUpdateRange(0, 1); f.geometry.index.needsUpdate = true;
    }
    return new T.Color(1, 1, 1);
  });
  f.renderer.render(f.scene, f.camera);
  const changed = [...f.expectedIndex().array]; f.renderer.render(f.scene, f.camera);
  f.geometry.index.array.fill(0); // no requested source version: edges remain stale
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device); assert.equal(submitted.length, 3);
  assert.deepEqual(indices(submitted[0]).slice(0, original.length), original);
  assert.deepEqual(indices(submitted[1]).slice(0, changed.length), changed);
  assert.deepEqual(indices(submitted[2]).slice(0, changed.length), changed);
  assert.deepEqual(f.geometry.index.updateRanges, [{start: 0, count: 1}], 'derived wire upload does not consume the original index request');
  assert.equal(updates, 3);
});

test('wireframe geometry composes with source-generated InstancedMesh attributes', async t => {
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicNodeMaterial({wireframe: true});
  material.colorNode = N.vec3(.3, .6, .9);
  const mesh = new T.InstancedMesh(geometry, material, 3);
  for (let i = 0; i < 3; i++) mesh.setMatrixAt(i, new T.Matrix4().makeTranslation(i - 1, 0, 0));
  const f = await fixture(t, {geometry, material, mesh});
  f.renderer.render(f.scene, f.camera); mesh.count = 2; f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(draw => [draw.pipeline.primitive.topology, draw.args[0], draw.args[1]]),
    [['line-list', 12, 3], ['line-list', 12, 2]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('each wireframe material group observes the range object selected for its physical pass', async t => {
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicNodeMaterial({wireframe: true});
  geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 0);
  const f = await fixture(t, {geometry, material, mesh: new T.Mesh(geometry, [material])}); let updates = 0;
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => {
    if (++updates === 1) geometry.drawRange = {start: 0, count: 0};
    return new T.Color(1, 1, 1);
  });
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(draw => [draw.args[0], draw.args[2]]), [[6, 0]]);
  assert.equal(updates, 2, 'empty second group retains its source update');
});

test('two groups using one program retain distinct edge versions in a single GPU submission', async t => {
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicNodeMaterial({wireframe: true});
  geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 0);
  const f = await fixture(t, {geometry, material, mesh: new T.Mesh(geometry, [material])});
  const original = [...f.expectedIndex().array]; let updates = 0;
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => {
    if (++updates === 1) { geometry.index.array.set([2, 1, 0, 3, 2, 0]); geometry.index.needsUpdate = true; }
    return new T.Color(1, 1, 1);
  });
  f.renderer.render(f.scene, f.camera); const changed = [...f.expectedIndex().array];
  await f.renderer.waitForGPU();
  const submitted = frames(f.device); assert.equal(submitted.length, 1); assert.equal(submitted[0].length, 2);
  assert.deepEqual(indices(submitted[0][0]).slice(0, original.length), original);
  assert.deepEqual(indices(submitted[0][1]).slice(0, changed.length), changed);
  assert.notEqual(submitted[0][0].index.buffer, submitted[0][1].index.buffer);
  assert.equal(updates, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const indexed of [true, false]) test(`native ${indexed ? 'indexed' : 'nonindexed'} batch wireframe preserves source subdraw multipliers`, async t => {
  const material = new T.MeshBasicMaterial({wireframe: true}), mesh = new T.BatchedMesh(4, 64, 96, material);
  const source = new T.PlaneGeometry(), geometryId = mesh.addGeometry(indexed ? source : source.toNonIndexed());
  mesh.addInstance(geometryId); mesh.addInstance(geometryId);
  mesh.perObjectFrustumCulled = false; mesh.sortObjects = true;
  const f = await fixture(t, {geometry: mesh.geometry, material, mesh});
  t.after(() => { mesh.dispose(); source.dispose(); });
  f.renderer.render(f.scene, f.camera);
  const expected = [...Array(mesh._multiDrawCount)].map((_, i) =>
    [mesh._multiDrawCounts[i], 1, mesh._multiDrawStarts[i] / mesh._multiDrawBytesPerElement, 0, i]);
  await f.renderer.waitForGPU();
  const submitted = draws(f.device); assert.equal(submitted.length, 2);
  assert.deepEqual(submitted.map(draw => draw.args), expected);
  assert.ok(submitted.every(draw => draw.indexed && draw.pipeline.primitive.topology === 'line-list'));
});

test('compileAsync initializes wire edges and source disposal refreshes requested residency', async t => {
  const f = await fixture(t); await f.renderer.compileAsync(f.scene, f.camera);
  const original = [...f.expectedIndex().array]; f.geometry.index.array.fill(0);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(indices(draws(f.device)[0]).slice(0, original.length), original);
  f.geometry.dispose(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.ok(indices(draws(f.device).at(-1)).every(value => value === 0));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('wireframe topology changes inside node updates fail before publication and recover on the next compile', async t => {
  const f = await fixture(t); let change = true;
  f.material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => {
    if (change) { f.material.wireframe = false; change = false; }
    return new T.Color(1, 1, 1);
  });
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(draws(f.device).length, 0);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(draws(f.device)[0].pipeline.primitive.topology, 'triangle-list');
});
