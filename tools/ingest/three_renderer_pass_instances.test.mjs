/** Actual facade, scene bridge and instance GPU owners. Recording buffers prove
 * accepted bytes/counts and source upload history; they do not rasterize pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {SSAARenderPass} from '../../upstream/three.js/examples/jsm/postprocessing/SSAARenderPass.js';
import {WebGLAttributes} from '../../upstream/three.js/src/renderers/webgl/WebGLAttributes.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {renderThreePassScene} from './three_pass_state.mjs';
import {createGpuInstanceAttributes, instanceAttributesSnapshot} from './gpu_buffer_geometry.mjs';
import {createWebGLInstanceHistory} from './webgl_instance_history.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGLRendererClass(T, {maxDraws: 64, maxBindings: 64});
async function fixture(t) {
  const device = geometryDevice(); Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [32, 32], format: configuration.format, usage: 16})})};
  const renderer = new Renderer({canvas, device}); await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(60, 1, .1, 100); camera.position.z = 5;
  const mesh = new T.InstancedMesh(new T.PlaneGeometry(), new T.MeshBasicMaterial(), 3);
  mesh.setColorAt(0, new T.Color(1, .5, .25)); mesh.frustumCulled = false; scene.add(mesh);
  t.after(async () => { renderer.dispose(); await renderer._disposePromise; mesh.geometry.dispose(); mesh.material.dispose(); mesh.dispose(); });
  return {renderer, device, scene, camera, mesh};
}
const drain = async renderer => { await renderer.waitForGPU(); await Promise.resolve(); };
function instanceDraws(device) {
  return device.snapshots.flat().flatMap(draw => {
    const layouts = draw.pipeline.vertex.buffers ?? [];
    const matrix = layouts.findIndex(layout => layout.stepMode === 'instance' && layout.arrayStride === 64);
    const color = layouts.findIndex(layout => layout.stepMode === 'instance' && layout.arrayStride === 12);
    if (matrix === -1) return [];
    const values = slot => new Float32Array(draw.contents.get(draw.streams.get(slot)).buffer).slice();
    return [{count: draw.args[1], matrix: values(matrix), color: color === -1 ? null : values(color)}];
  });
}
function oracle() {
  let bound;
  return WebGLAttributes({createBuffer: () => ({}), bindBuffer(_target, buffer) { bound = buffer; },
    bufferData(_target, array) { bound.bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength).slice(); },
    bufferSubData(_target, offset, array, start = 0, count = 0) {
      bound.bytes.set(new Uint8Array(array.buffer, array.byteOffset + 4 * start, 4 * (count || array.length - start)), offset);
    }, deleteBuffer() {}});
}

test('ordinary and verified pass draws retain queued counts, colors and matrix upload history', async t => {
  const f = await fixture(t), {renderer, mesh, scene, camera} = f;
  renderer.render(scene, camera); await drain(renderer);
  mesh.count = 2; mesh.instanceMatrix.array[12] = 90; mesh.instanceColor.array[0] = .875;
  renderThreePassScene(renderer, scene, camera);
  mesh.count = 1; mesh.instanceMatrix.array[13] = .75;
  mesh.instanceMatrix.addUpdateRange(13, 1); mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.array[1] = .125; mesh.instanceColor.addUpdateRange(1, 1); mesh.instanceColor.needsUpdate = true;
  renderThreePassScene(renderer, scene, camera);
  mesh.count = 0; renderer.render(scene, camera);
  mesh.count = 3; mesh.instanceMatrix.array.fill(99); mesh.instanceColor.array.fill(99);
  await drain(renderer);
  const draws = instanceDraws(f.device);
  assert.deepEqual(draws.map(draw => draw.count), [3, 2, 1, 0]);
  assert.equal(draws[1].matrix[12], 0, 'versionless edits stay CPU-only after the ordinary upload');
  assert.equal(draws[1].color[0], 1);
  assert.equal(draws[2].matrix[12], 0); assert.equal(draws[2].matrix[13], .75);
  assert.equal(draws[2].color[0], 1); assert.equal(draws[2].color[1], .125);
  assert.equal(renderer._passInstances.pendingBytes, 0); assert.equal(renderer._materialCaptureValues, 0);
});

test('a cold ordinary draw keeps its inputs when a newer verified draw queues before preparation', async t => {
  const f = await fixture(t); f.renderer.render(f.scene, f.camera);
  f.mesh.count = 2; f.mesh.instanceMatrix.array[12] = 2; f.mesh.instanceMatrix.needsUpdate = true;
  f.mesh.instanceColor.array[0] = .5; f.mesh.instanceColor.needsUpdate = true;
  renderThreePassScene(f.renderer, f.scene, f.camera);
  f.mesh.count = 0; f.mesh.instanceMatrix.array.fill(99); f.mesh.instanceColor.array.fill(99);
  await drain(f.renderer);
  assert.deepEqual(instanceDraws(f.device).map(draw => [draw.count, draw.matrix[12], draw.color[0]]), [[3, 0, 1], [2, 2, .5]]);
});

test('shared raw and captured histories match pinned WebGLAttributes partial ranges and callback release', () => {
  const device = geometryDevice(), mesh = new T.InstancedMesh(new T.BufferGeometry(), new T.MeshBasicMaterial(), 3);
  const history = createWebGLInstanceHistory(), source = mesh.instanceMatrix;
  const reference = source.clone(), gl = oracle(); gl.update(reference, 0);
  const owner = createGpuInstanceAttributes(device, mesh, {instanceHistory: history});
  source.array[12] = reference.array[12] = 9;
  assert.deepEqual(history.capture(source), new Float32Array(gl.get(reference).buffer.bytes.buffer));
  for (const attribute of [source, reference]) {
    attribute.array[13] = 7; attribute.array[14] = 8; attribute.array[15] = 6;
    attribute.addUpdateRange(13, 1); attribute.addUpdateRange(15, 1); attribute.needsUpdate = true;
  }
  const sourceRange = source.updateRanges[0], referenceRange = reference.updateRanges[0];
  gl.update(reference, 0); const accepted = history.capture(source); owner.update();
  assert.deepEqual(sourceRange, referenceRange, 'retained range-object aliases observe the pinned in-place merge');
  assert.deepEqual(sourceRange, {start: 13, count: 3});
  assert.deepEqual(source.updateRanges, reference.updateRanges);
  assert.deepEqual(new Uint8Array(accepted.buffer), gl.get(reference).buffer.bytes);
  assert.deepEqual(new Uint8Array(owner.vertexBuffer.data), gl.get(reference).buffer.bytes);
  for (const attribute of [source, reference]) { attribute.array[20] = 11; attribute.addUpdateRange(20, 0); attribute.needsUpdate = true; }
  owner.update(); gl.update(reference, 0);
  assert.deepEqual(history.capture(source), new Float32Array(gl.get(reference).buffer.bytes.buffer));
  const nextOwner = createGpuInstanceAttributes(device, mesh, {instanceHistory: history});
  assert.deepEqual(new Uint8Array(nextOwner.vertexBuffer.data), gl.get(reference).buffer.bytes);
  source.onUpload(function () { this.array = null; }); source.needsUpdate = true; owner.update(); owner.update();
  assert.equal(instanceAttributesSnapshot(owner, device).instanceCount, 3);
  owner.dispose(); nextOwner.dispose(); history.dispose();
});

test('actual pinned SSAA submits every accepted instance sample after source changes and disposal', async t => {
  const f = await fixture(t), pass = new SSAARenderPass(f.scene, f.camera);
  const read = new T.WebGLRenderTarget(32, 32), write = read.clone();
  t.after(() => { pass.dispose(); read.dispose(); write.dispose(); }); pass.sampleLevel = 2;
  pass.render(f.renderer, write, read);
  f.mesh.count = 0; f.mesh.instanceMatrix.array.fill(99); f.mesh.instanceColor = null;
  f.mesh.dispose(); f.scene.clear();
  await drain(f.renderer);
  const draws = instanceDraws(f.device);
  assert.equal(draws.length, 4); assert.ok(draws.every(draw => draw.count === 3 && draw.matrix[0] === 1 && draw.color[0] === 1));
  assert.equal(f.renderer._passInstances.pendingBytes, 0);
});

test('verified shader overrides consume captured instances through the source-program path', async t => {
  const f = await fixture(t), material = new T.ShaderMaterial({
    vertexShader: 'varying vec3 instanceTint; void main() { instanceTint = instanceColor; gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0); }',
    fragmentShader: 'varying vec3 instanceTint; void main() { gl_FragColor = vec4(instanceTint, 1.0); }',
  });
  t.after(() => material.dispose()); f.scene.overrideMaterial = material;
  renderThreePassScene(f.renderer, f.scene, f.camera);
  f.mesh.count = 1; f.mesh.instanceMatrix.array[12] = .5; f.mesh.instanceMatrix.needsUpdate = true;
  f.mesh.instanceColor.array[0] = .25; f.mesh.instanceColor.needsUpdate = true;
  renderThreePassScene(f.renderer, f.scene, f.camera);
  f.scene.overrideMaterial = null; f.mesh.count = 0; f.mesh.instanceMatrix.array.fill(99); f.mesh.instanceColor.array.fill(99);
  await drain(f.renderer);
  const draws = instanceDraws(f.device);
  assert.deepEqual(draws.map(draw => draw.count), [3, 1]);
  assert.deepEqual(draws.map(draw => [draw.matrix[12], draw.color[0]]), [[0, 1], [.5, .25]]);
  assert.ok(f.device.snapshots.flat().some(draw => draw.pipeline.fragment.module.code.includes('instanceTint')));
});

test('culled and hidden instances do not acknowledge uploads, and shared streams retain the selected bytes', async t => {
  const f = await fixture(t), second = new T.InstancedMesh(f.mesh.geometry, f.mesh.material, 3);
  second.instanceMatrix = f.mesh.instanceMatrix; second.instanceColor = f.mesh.instanceColor;
  second.visible = false; f.scene.add(second);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  f.mesh.visible = false; f.mesh.instanceMatrix.array[12] = 4; f.mesh.instanceMatrix.addUpdateRange(12, 1); f.mesh.instanceMatrix.needsUpdate = true;
  renderThreePassScene(f.renderer, f.scene, f.camera);
  assert.equal(f.mesh.instanceMatrix.updateRanges.length, 1);
  f.mesh.visible = true; renderThreePassScene(f.renderer, f.scene, f.camera);
  f.mesh.visible = false; second.visible = true; f.mesh.instanceMatrix.array[12] = 99;
  await drain(f.renderer);
  const draws = instanceDraws(f.device);
  assert.equal(draws.length, 2); assert.equal(draws[1].matrix[12], 4);
  assert.equal(f.mesh.instanceMatrix.updateRanges.length, 0); second.dispose();
});

test('verified instance capture refuses unsupported hooks and morphs without consuming source ranges', async t => {
  const f = await fixture(t); let calls = 0;
  const a = f.mesh.instanceMatrix; a.addUpdateRange(0, 1); a.needsUpdate = true;
  a.onUpload(() => { calls++; });
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(calls, 0); assert.equal(a.updateRanges.length, 1); assert.equal(f.device.submissions.length, 0);
  a.onUpload(T.BufferAttribute.prototype.onUploadCallback); f.mesh.morphTexture = new T.DataTexture(new Float32Array(4), 1, 1);
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(a.updateRanges.length, 1); f.mesh.morphTexture.dispose(); f.mesh.morphTexture = null;
});

test('automatic LOD and hooks installed after optional capture retain the ordinary source path', async t => {
  const f = await fixture(t), lod = new T.LOD(); f.scene.remove(f.mesh); lod.addLevel(f.mesh, 0); f.scene.add(lod);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(instanceDraws(f.device).length, 1);
  const submitted = f.device.submissions.length;
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(f.device.submissions.length, submitted);
  f.scene.remove(lod); f.scene.add(f.mesh);
  f.renderer.render(f.scene, f.camera);
  let calls = 0;
  f.mesh.onBeforeRender = function (_renderer, _scene, camera) { calls++; assert.equal(this, f.mesh); assert.equal(camera, f.camera); };
  await drain(f.renderer); assert.equal(calls, 1);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('optional shared history capacity preserves ordinary uploads and refuses unavailable capture history', () => {
  const device = geometryDevice(), history = createWebGLInstanceHistory({maxBytes: 64, maxAttributes: 1});
  const a = new T.InstancedMesh(new T.BufferGeometry(), new T.MeshBasicMaterial(), 1);
  const b = new T.InstancedMesh(a.geometry, a.material, 1);
  const first = createGpuInstanceAttributes(device, a, {instanceHistory: history});
  const second = createGpuInstanceAttributes(device, b, {instanceHistory: history});
  b.instanceMatrix.array[12] = 3; b.instanceMatrix.needsUpdate = true; second.update();
  assert.equal(new Float32Array(second.vertexBuffer.data)[12], 3);
  assert.equal(history.bytes, 64);
  assert.throws(() => history.capture(b.instanceMatrix), {code: 'GEOMETRY_GPU_HISTORY'});
  first.dispose(); second.dispose(); history.dispose();
});
