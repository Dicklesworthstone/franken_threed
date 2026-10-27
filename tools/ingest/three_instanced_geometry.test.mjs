/** Native instanced geometry through the public renderer and owned GPU queue.
 * These source/recording-device checks do not claim GPU pixel parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import RenderObject from '../../upstream/three.js/src/renderers/common/RenderObject.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(THREE), N = THREE.TSL;
function fixture(t, {indexed = true, authored = true, interleaved = false} = {}) {
  const device = geometryDevice();
  device.limits.maxTextureDimension2D = 8192; device.limits.maxBindingsPerBindGroup = 1000;
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => descriptor; device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const base = new THREE.PlaneGeometry(.5, .5);
  const geometry = new THREE.InstancedBufferGeometry().copy(indexed ? base : base.toNonIndexed());
  geometry.instanceCount = 3;
  const offsets = [-.6, 0, 0, 0, 0, 0, .6, 0, 0], colors = [.125, .25, .5, .25, .5, .75, .5, .75, 1];
  let offset, color, owner;
  if (interleaved) {
    const array = new Float32Array(18);
    for (let i = 0; i < 3; i++) { array.set(offsets.slice(i * 3, i * 3 + 3), i * 6); array.set(colors.slice(i * 3, i * 3 + 3), i * 6 + 3); }
    owner = new THREE.InstancedInterleavedBuffer(array, 6, 1);
    offset = new THREE.InterleavedBufferAttribute(owner, 3, 0);
    color = new THREE.InterleavedBufferAttribute(owner, 3, 3);
  } else {
    offset = new THREE.InstancedBufferAttribute(new Float32Array(offsets), 3);
    color = new THREE.InstancedBufferAttribute(new Float32Array(colors), 3);
    owner = color;
  }
  geometry.setAttribute('offset', offset); geometry.setAttribute('color', color);
  const material = authored ? new THREE.MeshBasicNodeMaterial() : new THREE.MeshBasicMaterial();
  material.vertexColors = true;
  if (authored) material.positionNode = N.positionLocal.add(N.attribute('offset', 'vec3'));
  const mesh = new THREE.Mesh(geometry, material); mesh.frustumCulled = false;
  const scene = new THREE.Scene(); scene.add(mesh);
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  return {device, renderer, scene, camera, mesh, geometry, material, offset, color, owner};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const draws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
function instanceStreams(draw) {
  return draw.pipeline.vertex.buffers.flatMap((layout, index) => {
    if (layout.stepMode !== 'instance') return [];
    const bytes = draw.contents.get(draw.streams.get(index));
    return [{layout, values: [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]}];
  });
}
function sourceCount(f) {
  return RenderObject.prototype.getDrawParameters.call({object: f.mesh, geometry: f.geometry, material: f.material,
    group: null, drawRange: f.geometry.drawRange, getIndex: () => f.geometry.index})?.instanceCount ?? 0;
}

for (const indexed of [false, true]) test(`native ${indexed ? 'indexed' : 'nonindexed'} geometry uses source count precedence and instance attributes`, async t => {
  const f = fixture(t, {indexed}); f.mesh.count = 19;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const [draw] = draws(f.device);
  assert.equal(draw.indexed, indexed); assert.equal(draw.args[0], 6);
  assert.equal(draw.args[1], sourceCount(f)); assert.equal(draw.args[1], 3);
  assert.deepEqual(instanceStreams(draw).map(stream => stream.layout.arrayStride), [12, 12]);
  assert.ok(instanceStreams(draw).some(stream => stream.values.join() === [...f.color.array].join()));
  assert.equal(f.renderer.info.render.calls, 1);
});

test('ordinary materials on InstancedBufferGeometry select native programs without authored node slots', async t => {
  const f = fixture(t, {authored: false}); f.mesh.count = 0;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const [draw] = draws(f.device);
  assert.equal(draw.args[1], 3); assert.equal(instanceStreams(draw).length, 1);
  assert.match(draw.pipeline.vertex.module.code, /color/);
  assert.equal(f.mesh.material, f.material); assert.equal(f.geometry.isInstancedBufferGeometry, true);
});

test('queued instance counts follow source node update ordering and keep zero draws without replaying callbacks', async t => {
  const f = fixture(t); let updates = 0, after = 0, requested = 1;
  const gain = N.uniform(1).onObjectUpdate(() => { updates++; f.geometry.instanceCount = requested; return 1; });
  gain.updateAfterType = 'object'; gain.updateAfter = () => { after++; f.geometry.instanceCount = 99; };
  f.material.colorNode = N.attribute('color', 'vec3').mul(gain);
  await f.renderer.init();
  for (requested of [1, 3, 0, 2]) f.renderer.render(f.scene, f.camera);
  assert.equal(updates, 4); assert.equal(after, 4);
  f.geometry.instanceCount = 77; f.mesh.count = 0;
  await drain(f.renderer);
  assert.deepEqual(draws(f.device).map(draw => draw.args[1]), [1, 3, 2]);
  assert.equal(updates, 4); assert.equal(after, 4); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.equal(f.renderer.info.f3d.presentedRenders, 4);
});

test('instanced geometry counts remain live without shader or pipeline recompilation', async t => {
  const f = fixture(t); await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  const count = f.device.pipelines.length;
  for (const instances of [2, 1, 3]) {
    f.geometry.instanceCount = instances; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  }
  assert.deepEqual(draws(f.device).map(draw => draw.args[1]), [2, 1, 3]);
  assert.equal(f.device.pipelines.length, count);
});

for (const interleaved of [false, true]) test(`native ${interleaved ? 'interleaved' : 'separate'} streams preserve source divisor behavior`, async t => {
  const f = fixture(t, {interleaved});
  // r186 WebGPUAttributeUtils uses instance step mode, with no WebGL divisor
  // expansion. Three real source records must remain three distinct records.
  f.owner.meshPerAttribute = 2;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const streams = instanceStreams(draws(f.device)[0]);
  assert.equal(streams.length, interleaved ? 1 : 2);
  assert.ok(streams.some(stream => stream.values.join() === [...f.owner.array].join()));
  if (interleaved) {
    assert.equal(streams[0].layout.arrayStride, 24);
    assert.deepEqual(streams[0].layout.attributes.map(attribute => attribute.offset).sort((a, b) => a - b), [0, 12]);
  }
});

test('interleaved instance partial uploads preserve unrequested GPU-visible bytes', async t => {
  const f = fixture(t, {interleaved: true}); await f.renderer.init();
  await f.renderer.renderAsync(f.scene, f.camera);
  const original = instanceStreams(draws(f.device)[0])[0].values;
  f.owner.array[0] = 50; f.owner.array[3] = .875;
  f.owner.addUpdateRange(3, 1); f.owner.needsUpdate = true;
  await f.renderer.renderAsync(f.scene, f.camera);
  const updated = instanceStreams(draws(f.device)[1])[0].values;
  assert.deepEqual(updated, original.map((value, index) => index === 3 ? .875 : value));
  assert.deepEqual(f.owner.updateRanges, []); assert.equal(f.owner.array[0], 50);
});

test('instance capacity is checked across every shader-consumed stream without clamping', async t => {
  const f = fixture(t); f.geometry.instanceCount = 3;
  f.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0]), 3));
  assert.equal(sourceCount(f), 3, 'source count is not inferred or clamped to the first attribute');
  await f.renderer.init();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(draws(f.device).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('unused instance attributes do not constrain native draw capacity', async t => {
  const f = fixture(t);
  f.geometry.setAttribute('unused', new THREE.InstancedBufferAttribute(new Float32Array([1]), 1));
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(draws(f.device)[0].args[1], 3);
});

test('invalid geometry counts reject rather than falling back to Mesh.count or truncating', async t => {
  for (const count of [-1, .5, Infinity, NaN, 0x100000000]) {
    const f = fixture(t); f.geometry.instanceCount = count; f.mesh.count = 1; await f.renderer.init();
    assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_OBJECT'});
    assert.equal(draws(f.device).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});

test('instanced geometry without per-instance vertex inputs still uses its explicit source count', async t => {
  const f = fixture(t); f.material.vertexColors = false;
  f.material.positionNode = N.positionLocal.add(N.vec3(N.float(N.instanceIndex).mul(.5), 0, 0));
  f.geometry.instanceCount = 7;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(draws(f.device)[0].args[1], 7); assert.equal(instanceStreams(draws(f.device)[0]).length, 0);
});
