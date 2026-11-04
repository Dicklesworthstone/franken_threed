/** Ordinary native fixed draws retain their accepted scene inputs. Actual
 * source objects feed the renderer and the recording device inspects GPU bytes;
 * these assertions do not replace browser pixel comparisons. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {threeSpriteWorldMatrix} from './three_sprites.mjs';

const Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
function fixture(t, Material = T.MeshBasicMaterial) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    return texture;
  };
  device.createSampler = value => ({...value}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100);
  camera.position.z = 4;
  const geometry = new T.PlaneGeometry(), material = new Material({color: new T.Color(.125, .375, .625)});
  const mesh = new T.Mesh(geometry, material); scene.add(mesh);
  t.after(() => { renderer.dispose(); geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, mesh, geometry, material};
}
function fixedPackets(device) {
  return device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main' && draw.streams.size)
    .flatMap(draw => {
      const {group, offsets} = draw.groups.get(0), resource = group.entries.find(entry => entry.binding === 0).resource;
      const layout = group.layout.entries.find(entry => entry.binding === 0).buffer;
      const storage = layout.type === 'read-only-storage', stride = layout.minBindingSize;
      const bytes = draw.contents.get(resource.buffer);
      const first = storage ? draw.args[draw.indexed ? 4 : 3] : 0;
      return Array.from({length: storage ? draw.args[1] : 1}, (_, index) => {
        const offset = (resource.offset ?? 0) + (storage ? (first + index) * stride : offsets[0] ?? 0);
        return {draw, words: [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, stride / 4)]};
      });
    });
}
function clip(camera, world) {
  return new T.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
    .multiply(world).elements.map(Math.fround);
}
function lightWords(draw) {
  const {group} = draw.groups.get(1), resource = group.entries.find(entry => entry.binding === 0).resource;
  const bytes = draw.contents.get(resource.buffer);
  return [...new Float32Array(bytes.buffer, bytes.byteOffset + (resource.offset ?? 0), resource.size / 4)];
}

for (const Material of [T.MeshBasicMaterial, T.MeshPhongMaterial])
test(`queued ${Material.name} draws keep independent world/camera matrices and winding`, async t => {
  const f = fixture(t, Material), expected = [], worlds = [];
  await f.renderer.init();
  for (const [x, scale, cameraZ] of [[.125, [1, 1, 1], 4], [.625, [-1, 2, .5], 6]]) {
    f.mesh.position.x = x; f.mesh.scale.set(...scale); f.camera.position.z = cameraZ;
    f.renderer.render(f.scene, f.camera);
    expected.push(clip(f.camera, f.mesh.matrixWorld)); worlds.push(f.mesh.matrixWorld.elements.map(Math.fround));
    const request = f.renderer._pending.at(-1);
    assert.equal(request.nodeFrame.draws.length, 0, 'ordinary draws keep the fixed shader owner');
    assert.ok(request.nodeFrame.fixedGeometries.get(f.mesh));
  }
  f.mesh.position.x = 9; f.mesh.scale.set(3, 3, 3); f.mesh.updateMatrixWorld();
  f.camera.position.z = 20; f.camera.updateMatrixWorld();
  await drain(f.renderer);
  const packets = fixedPackets(f.device); assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(({words}) => words.slice(0, 16)), expected);
  assert.deepEqual(packets.map(({draw}) => draw.pipeline.primitive.frontFace), ['ccw', 'cw']);
  if (Material === T.MeshPhongMaterial) assert.deepEqual(packets.map(({words}) => words.slice(32, 48)), worlds);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('fixed draws preserve manually authored matrixWorld values without updating source caches at submission', async t => {
  const f = fixture(t); f.mesh.matrixAutoUpdate = false; f.mesh.matrixWorldAutoUpdate = false;
  f.mesh.matrixWorld.makeTranslation(.75, 0, 0);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const expected = clip(f.camera, f.mesh.matrixWorld);
  f.mesh.matrixWorld.makeTranslation(12, 0, 0);
  f.mesh.modelViewMatrix.elements.fill(7); f.mesh.normalMatrix.elements.fill(9);
  await drain(f.renderer);
  assert.deepEqual(fixedPackets(f.device)[0].words.slice(0, 16), expected);
  assert.deepEqual(f.mesh.modelViewMatrix.elements, Array(16).fill(7));
  assert.deepEqual(f.mesh.normalMatrix.elements, Array(9).fill(9));
  assert.equal(f.mesh.matrixWorld.elements[12], 12);
});

for (const change of ['remove', 'add']) test(`accepted fixed scene membership survives a later ${change}`, async t => {
  const f = fixture(t), added = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial());
  t.after(() => { added.geometry.dispose(); added.material.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  if (change === 'remove') f.scene.remove(f.mesh); else f.scene.add(added);
  await drain(f.renderer);
  assert.equal(fixedPackets(f.device).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued fixed scenes preserve each hierarchy, material and transform across replacement', async t => {
  const f = fixture(t), secondMaterial = new T.MeshBasicMaterial({color: new T.Color(.75, .5, .25)});
  const replacement = new T.Mesh(f.geometry, secondMaterial); replacement.position.x = .5;
  t.after(() => secondMaterial.dispose());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  f.scene.remove(f.mesh); f.scene.add(replacement); f.renderer.render(f.scene, f.camera);
  f.scene.clear();
  await drain(f.renderer);
  const packets = fixedPackets(f.device); assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(({words}) => words.slice(16, 19)), [[.125, .375, .625], [.75, .5, .25]]);
  assert.deepEqual(packets.map(({words}) => words[12]), [0, Math.fround(.5 * f.camera.projectionMatrix.elements[0])]);
});

test('fixed scene traversal retains parent layers, group order and per-object render order', async t => {
  const f = fixture(t), secondMaterial = new T.MeshBasicMaterial({color: new T.Color(.75, .5, .25), transparent: true});
  const second = new T.Mesh(f.geometry, secondMaterial), a = new T.Group(), b = new T.Group();
  f.material.transparent = true; a.add(f.mesh); b.add(second); f.scene.add(a, b);
  a.renderOrder = 1; b.renderOrder = 2; f.mesh.renderOrder = 20; second.renderOrder = 10;
  t.after(() => secondMaterial.dispose());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  a.renderOrder = 2; b.renderOrder = 1; second.layers.disableAll();
  f.renderer.render(f.scene, f.camera);
  a.layers.disableAll(); b.layers.disableAll(); f.mesh.layers.disableAll();
  await drain(f.renderer);
  assert.deepEqual(fixedPackets(f.device).map(({words}) => words.slice(16, 19)),
    [[.125, .375, .625], [.75, .5, .25], [.125, .375, .625]]);
});

test('accepted fixed lighting keeps point radiance, world position and attenuation after edits and removal', async t => {
  const f = fixture(t, T.MeshPhongMaterial), light = new T.PointLight(new T.Color(.25, .5, .75), 2, 12, 1);
  light.position.set(1, 2, 3); f.scene.add(light);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  light.position.set(5, 6, 7); light.updateMatrixWorld(); light.color.setRGB(.75, .25, .5);
  light.intensity = 4; light.distance = 20; light.decay = 2; f.scene.remove(light);
  await drain(f.renderer);
  const packets = fixedPackets(f.device); assert.equal(packets.length, 1);
  const words = lightWords(packets[0].draw);
  assert.equal(words[4], 1); assert.deepEqual(words.slice(8, 11), [1, 2, 3]);
  assert.deepEqual(words.slice(12, 15), [.5, 1, 1.5]); assert.equal(words[19], 12); assert.equal(words[22], 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('fixed directional lights retain target matrices even when the target is outside the accepted tree', async t => {
  const f = fixture(t, T.MeshPhongMaterial), light = new T.DirectionalLight();
  light.position.set(0, 0, 3); light.target.position.set(0, 0, -2); light.target.updateMatrixWorld();
  f.scene.add(light);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  light.target.position.set(3, 0, 3); light.target.updateMatrixWorld();
  light.color.setRGB(0, 0, 0);
  await drain(f.renderer);
  const words = lightWords(fixedPackets(f.device)[0].draw);
  assert.deepEqual(words.slice(8, 11), [0, 0, -1]); assert.deepEqual(words.slice(12, 15), [1, 1, 1]);
});

test('fixed sprites retain their world transform, center, material rotation and repeated count', async t => {
  const f = fixture(t), material = new T.SpriteMaterial({color: new T.Color(.25, .5, .75)}), sprite = new T.Sprite(material);
  f.scene.remove(f.mesh); f.scene.add(sprite); sprite.count = 2; sprite.position.x = .5;
  sprite.scale.set(2, 3, 1); sprite.center.set(.25, .75); material.rotation = .25;
  t.after(() => material.dispose());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const expected = clip(f.camera, new T.Matrix4().fromArray(threeSpriteWorldMatrix(T, sprite, material, f.camera)));
  sprite.position.x = 8; sprite.scale.set(4, 1, 1); sprite.updateMatrixWorld(); sprite.center.set(1, 0);
  material.rotation = 2; sprite.count = 0;
  await drain(f.renderer);
  const packets = fixedPackets(f.device); assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(({words}) => words.slice(0, 16)), [expected, expected]);
});

test('fixed ArrayCamera views use accepted matrices and child layer selections', async t => {
  const f = fixture(t), left = new T.PerspectiveCamera(50, 1, .1, 100), right = left.clone();
  left.position.set(-.5, 0, 4); right.position.set(.5, 0, 4); right.layers.set(1);
  left.viewport = new T.Vector4(0, 0, 32, 64); right.viewport = new T.Vector4(32, 0, 32, 64);
  const camera = new T.ArrayCamera([left, right]); camera.layers.enable(1); f.mesh.layers.enable(1);
  f.mesh.position.x = .25;
  await f.renderer.init(); f.renderer.render(f.scene, camera);
  const expected = [left, right].map(view => clip(view, f.mesh.matrixWorld));
  f.mesh.layers.disableAll(); f.mesh.position.x = 9; f.mesh.updateMatrixWorld();
  right.layers.disableAll(); left.position.x = 9; left.updateMatrixWorld();
  await drain(f.renderer);
  assert.deepEqual(fixedPackets(f.device).map(({words}) => words.slice(0, 16)), expected);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
