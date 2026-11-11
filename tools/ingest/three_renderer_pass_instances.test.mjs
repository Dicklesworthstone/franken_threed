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
function rigidPackets(device) {
  return device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main' && draw.streams.size)
    .flatMap(draw => {
      const {group, offsets} = draw.groups.get(0), resource = group.entries.find(entry => entry.binding === 0).resource;
      const layout = group.layout.entries.find(entry => entry.binding === 0).buffer;
      const storage = layout.type === 'read-only-storage', stride = layout.minBindingSize;
      const bytes = draw.contents.get(resource.buffer), first = storage ? draw.args[draw.indexed ? 4 : 3] : 0;
      return Array.from({length: storage ? draw.args[1] : 1}, (_, index) => {
        const offset = (resource.offset ?? 0) + (storage ? (first + index) * stride : offsets[0] ?? 0);
        return {draw, words: [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, stride / 4)]};
      });
    });
}
function ordinaryMesh(f) {
  const mesh = new T.Mesh(f.mesh.geometry, f.mesh.material);
  mesh.frustumCulled = false; f.scene.remove(f.mesh); f.scene.add(mesh);
  return mesh;
}
const clipMatrix = (camera, mesh) => {
  const projection = new T.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  for (let column = 0; column < 4; column++)
    projection.elements[column * 4 + 2] = .5 * (projection.elements[column * 4 + 2] + projection.elements[column * 4 + 3]);
  return projection.multiply(mesh.matrixWorld).elements.map(Math.fround);
};

for (const warm of [false, true])
test(`queued ordinary geometry retains accepted vertex uploads and draw ranges (warm=${warm})`, async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), position = mesh.geometry.attributes.position;
  const original = [...position.array];
  if (warm) {
    let callbacks = 0; mesh.onBeforeRender = () => { callbacks++; };
    f.renderer.render(f.scene, f.camera); await drain(f.renderer); f.device.snapshots.length = 0;
    assert.equal(callbacks, 1); mesh.onBeforeRender = T.Object3D.prototype.onBeforeRender;
    mesh.material.side = T.BackSide;
  }
  position.array[0] = .125; mesh.geometry.setDrawRange(0, 3); f.renderer.render(f.scene, f.camera);
  const first = warm ? original : [...position.array];
  position.array[0] = .75; position.array[1] = 99;
  position.addUpdateRange(0, 1); position.needsUpdate = true;
  mesh.geometry.setDrawRange(3, 3); f.renderer.render(f.scene, f.camera);
  const second = first.slice(); second[0] = .75;
  position.array.fill(90); mesh.geometry.setDrawRange(0, 6);
  await drain(f.renderer);
  const packets = rigidPackets(f.device);
  const arrays = packets.map(({draw}) => {
    const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(attribute => attribute.shaderLocation === 0));
    const bytes = draw.contents.get(draw.streams.get(slot));
    return [...new Float32Array(bytes.buffer, bytes.byteOffset, position.array.length)];
  });
  assert.deepEqual(arrays, [first, second]);
  assert.deepEqual(packets.map(({draw}) => [draw.args[0], draw.args[2]]), [[3, 0], [3, 3]]);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('queued geometry retains partial index uploads, attribute selection and material groups', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), geometry = mesh.geometry;
  const red = mesh.material, blue = new T.MeshBasicMaterial({color: new T.Color(.25, .5, .75), transparent: true});
  red.color.setRGB(.75, .5, .25); red.transparent = true; mesh.material = [red, blue];
  t.after(() => blue.dispose());
  geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 1);
  const first = [...geometry.index.array], originalPosition = geometry.attributes.position;
  f.renderer.render(f.scene, f.camera);
  geometry.index.array[0] = 3; geometry.index.array[3] = 0;
  geometry.index.addUpdateRange(3, 1); geometry.index.needsUpdate = true;
  geometry.groups[0].materialIndex = 1; geometry.groups[1].materialIndex = 0;
  f.renderer.render(f.scene, f.camera);
  const second = first.slice(); second[3] = 0;
  geometry.setIndex(null); geometry.clearGroups();
  geometry.setAttribute('position', new T.BufferAttribute(new Float32Array(12).fill(90), 3));
  structuredClone(originalPosition.array.buffer, {transfer: [originalPosition.array.buffer]});
  await drain(f.renderer);
  const packets = rigidPackets(f.device);
  assert.deepEqual(packets.map(({draw}) => {
    const bytes = draw.contents.get(draw.index.buffer);
    return [...new Uint16Array(bytes.buffer, bytes.byteOffset, first.length)];
  }), [first, first, second, second]);
  assert.deepEqual(packets.map(({words}) => words.slice(16, 19)),
    [[.75, .5, .25], [.25, .5, .75], [.25, .5, .75], [.75, .5, .25]]);
  assert.deepEqual(packets.map(({draw}) => [draw.args[0], draw.args[2]]), [[3, 0], [3, 3], [3, 0], [3, 3]]);
  assert.equal(geometry.index, null); assert.equal(f.renderer._passInstances.pendingBytes, 0);
});

test('queued interleaved geometry retains one accepted shared stream and source range effects', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), geometry = new T.BufferGeometry();
  const data = new T.InterleavedBuffer(new Float32Array([-1, -1, 0, .25, .5, .75, 1, -1, 0, .75, .5, .25, 0, 1, 0, 1, 1, 1]), 6);
  geometry.setAttribute('position', new T.InterleavedBufferAttribute(data, 3, 0));
  geometry.setAttribute('color', new T.InterleavedBufferAttribute(data, 3, 3));
  mesh.geometry = geometry; mesh.material.vertexColors = true; t.after(() => geometry.dispose());
  f.renderer.render(f.scene, f.camera); const first = [...data.array];
  data.array[1] = -.5; data.array[2] = .125; data.array[3] = .875; data.array[4] = 90;
  data.addUpdateRange(1, 1); data.addUpdateRange(3, 1); const retainedRange = data.updateRanges[0]; data.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  const second = first.slice(); second.splice(1, 3, -.5, .125, .875);
  assert.deepEqual(retainedRange, {start: 1, count: 3}); assert.deepEqual(data.updateRanges, []);
  data.array.fill(99); await drain(f.renderer);
  assert.deepEqual(rigidPackets(f.device).map(({draw}) => {
    const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(attribute => attribute.shaderLocation === 0));
    assert.deepEqual(draw.pipeline.vertex.buffers[slot].attributes.map(attribute => attribute.shaderLocation), [0, 4]);
    const bytes = draw.contents.get(draw.streams.get(slot));
    return [...new Float32Array(bytes.buffer, bytes.byteOffset, first.length)];
  }), [first, second]);
});

test('geometry disposal separates queued generations and admits fresh versionless CPU bytes on reuse', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), position = mesh.geometry.attributes.position;
  position.array[0] = .25; f.renderer.render(f.scene, f.camera);
  mesh.geometry.dispose(); position.array[0] = .75; f.renderer.render(f.scene, f.camera);
  mesh.geometry.dispose(); position.array[0] = 90;
  await drain(f.renderer);
  const packets = rigidPackets(f.device);
  assert.deepEqual(packets.map(({draw}) => {
    const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(attribute => attribute.shaderLocation === 0));
    return new Float32Array(draw.contents.get(draw.streams.get(slot)).buffer)[0];
  }), [.25, .75]);
  assert.notEqual(packets[0].draw.streams.get(0), packets[1].draw.streams.get(0));
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('an upload hook introduced during preparation keeps its original receiver and updated source bytes', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), position = mesh.geometry.attributes.position;
  f.renderer.render(f.scene, f.camera);
  let calls = 0; position.array[0] = .75; position.needsUpdate = true;
  position.onUpload(function () { calls++; assert.equal(this, position); });
  await drain(f.renderer); assert.equal(calls, 1);
  const {draw} = rigidPackets(f.device)[0];
  const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(attribute => attribute.shaderLocation === 0));
  assert.equal(new Float32Array(draw.contents.get(draw.streams.get(slot)).buffer)[0], .75);
  assert.equal(f.renderer._passInstances.pendingBytes, 0);
});

test('cold ordinary WebGL renders retain each accepted world matrix and geometry selector', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), replacement = new T.BoxGeometry();
  t.after(() => replacement.dispose());
  const expected = [];
  mesh.position.x = .25; f.renderer.render(f.scene, f.camera); expected.push(clipMatrix(f.camera, mesh));
  mesh.geometry = replacement; mesh.position.x = .75; mesh.scale.x = -1;
  f.renderer.render(f.scene, f.camera); expected.push(clipMatrix(f.camera, mesh));
  mesh.position.x = 90; mesh.scale.x = 3; mesh.updateMatrixWorld(); mesh.geometry = null;
  await drain(f.renderer);
  const packets = rigidPackets(f.device);
  assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(({words}) => words.slice(0, 16)), expected);
  assert.deepEqual(packets.map(({draw}) => draw.args[0]), [6, 36]);
  assert.deepEqual(packets.map(({draw}) => draw.pipeline.primitive.frontFace), ['ccw', 'cw']);
  assert.equal(mesh.geometry, null); assert.equal(mesh.matrixWorld.elements[12], 90);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('queued rigid scenes preserve membership and material selectors after removal and insertion', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f), blue = new T.MeshBasicMaterial({color: new T.Color(.25, .5, .75)});
  f.mesh.material.color.setRGB(.75, .5, .25);
  const next = new T.Mesh(mesh.geometry, blue); next.frustumCulled = false;
  const unrelated = new T.Mesh(mesh.geometry, new T.ShaderMaterial());
  t.after(() => { blue.dispose(); unrelated.material.dispose(); });
  mesh.position.x = .25; f.renderer.render(f.scene, f.camera);
  f.scene.remove(mesh); next.position.x = .75; f.scene.add(next); f.renderer.render(f.scene, f.camera);
  f.scene.clear(); f.scene.add(unrelated); mesh.material = blue; next.position.x = 90; next.updateMatrixWorld();
  await drain(f.renderer);
  const packets = rigidPackets(f.device);
  assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(({words}) => words.slice(16, 19)), [[.75, .5, .25], [.25, .5, .75]]);
  assert.deepEqual(packets.map(({words}) => words[12]), [.25, .75].map(x => Math.fround(x * f.camera.projectionMatrix.elements[0])));
  assert.deepEqual(f.scene.children, [unrelated]); assert.equal(mesh.material, blue);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('rigid snapshots preserve parent visibility, layers and group sorting for each accepted call', async t => {
  const f = await fixture(t), first = ordinaryMesh(f), material = new T.MeshBasicMaterial({color: new T.Color(.25, .5, .75), transparent: true});
  first.material.color.setRGB(.75, .5, .25); first.material.transparent = true;
  const second = new T.Mesh(first.geometry, material); second.frustumCulled = false;
  const a = new T.Group(), b = new T.Group(); a.add(first); b.add(second); f.scene.add(a, b);
  a.renderOrder = 1; b.renderOrder = 2; first.renderOrder = 20; second.renderOrder = 10;
  t.after(() => material.dispose());
  f.renderer.render(f.scene, f.camera);
  a.renderOrder = 2; b.renderOrder = 1; second.layers.disableAll();
  f.renderer.render(f.scene, f.camera);
  a.visible = false; b.visible = false; first.layers.disableAll();
  await drain(f.renderer);
  assert.deepEqual(rigidPackets(f.device).map(({words}) => words.slice(16, 19)),
    [[.75, .5, .25], [.25, .5, .75], [.75, .5, .25]]);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('rigid frustum selection and manual matrixWorld values stay attached to the accepted draw', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f);
  mesh.matrixAutoUpdate = false; mesh.matrixWorldAutoUpdate = false; mesh.frustumCulled = true;
  mesh.matrixWorld.makeTranslation(.25, 0, 0); f.renderer.render(f.scene, f.camera);
  const expected = clipMatrix(f.camera, mesh);
  mesh.matrixWorld.makeTranslation(100, 0, 0); f.renderer.render(f.scene, f.camera);
  mesh.matrixWorld.identity(); mesh.geometry.boundingSphere.center.set(99, 99, 99);
  await drain(f.renderer);
  const packets = rigidPackets(f.device); assert.equal(packets.length, 1);
  assert.deepEqual(packets[0].words.slice(0, 16), expected);
  assert.equal(mesh.matrixWorld.elements[12], 0);
});

test('mixed ordinary and instanced draws retain their independent accepted transforms', async t => {
  const f = await fixture(t), ordinary = new T.Mesh(f.mesh.geometry, f.mesh.material);
  ordinary.frustumCulled = false; f.scene.add(ordinary);
  ordinary.position.x = .25; f.mesh.position.x = -.25;
  f.renderer.render(f.scene, f.camera);
  const expected = clipMatrix(f.camera, ordinary);
  ordinary.position.x = 90; f.mesh.position.x = 80; f.scene.updateMatrixWorld();
  await drain(f.renderer);
  const packets = rigidPackets(f.device);
  assert.ok(packets.some(({words}) => words.every((value, index) => index >= 16 || value === expected[index])));
  assert.equal(instanceDraws(f.device).length, 1);
});

test('hooks introduced after an ordinary capture run on original source objects and camera', async t => {
  const f = await fixture(t), mesh = ordinaryMesh(f); f.renderer.render(f.scene, f.camera);
  let calls = 0;
  mesh.onBeforeRender = function (renderer, scene, camera, geometry) {
    calls++; assert.equal(this, mesh); assert.equal(renderer, f.renderer);
    assert.equal(scene, f.scene); assert.equal(camera, f.camera); assert.equal(geometry, mesh.geometry);
  };
  await drain(f.renderer); assert.equal(calls, 1);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

for (const Type of [T.Mesh, T.LineSegments, T.Points])
test(`queued ${Type.name} source programs consume the accepted model-view matrices`, async t => {
  const f = await fixture(t), material = new T.ShaderMaterial({
    vertexShader: 'void main() { gl_PointSize = 2.0; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'void main() { gl_FragColor = vec4(0.25, 0.5, 0.75, 1.0); }',
  });
  const object = new Type(f.mesh.geometry, material); object.frustumCulled = false;
  if (Type === T.Points) object.geometry.setIndex(null);
  f.scene.remove(f.mesh); f.scene.add(object); t.after(() => material.dispose());
  const expected = [], positions = [];
  for (const x of [.25, .75]) {
    object.geometry.attributes.position.array[0] = x; object.geometry.attributes.position.needsUpdate = true;
    object.position.x = x; f.renderer.render(f.scene, f.camera);
    expected.push(new T.Matrix4().multiplyMatrices(f.camera.matrixWorldInverse, object.matrixWorld).elements.map(Math.fround));
    positions.push([...object.geometry.attributes.position.array]);
  }
  object.position.x = 90; object.updateMatrixWorld(); object.geometry.attributes.position.array.fill(90);
  await drain(f.renderer);
  const {reflection} = f.renderer._programSupport.compile(material, object).program;
  const offset = reflection.uniforms.find(uniform => uniform.name === 'modelViewMatrix').offset;
  const matrices = f.device.snapshots.flat().filter(draw => draw.streams.size)
    .map(draw => {
      const {group, offsets} = draw.groups.get(0), resource = group.entries.find(entry => entry.binding === 0).resource;
      const bytes = draw.contents.get(resource.buffer);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + (resource.offset ?? 0) + (offsets[0] ?? 0) + offset, 16)];
    });
  assert.deepEqual(matrices, expected);
  assert.deepEqual(f.device.snapshots.flat().filter(draw => draw.streams.size).map(draw => {
    const location = reflection.attributes.find(attribute => attribute.name === 'position').location;
    const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(attribute => attribute.shaderLocation === location));
    const bytes = draw.contents.get(draw.streams.get(slot));
    return [...new Float32Array(bytes.buffer, bytes.byteOffset, positions[0].length)];
  }), positions);
  assert.equal(f.renderer._materialCaptureValues, 0);
});
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
