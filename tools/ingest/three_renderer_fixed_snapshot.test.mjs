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

function addNativeMesh(t, f, node = T.TSL.vec3(.25, .5, .75), parent = f.scene) {
  const material = new T.MeshBasicNodeMaterial(), geometry = new T.PlaneGeometry();
  material.colorNode = node;
  const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; parent.add(mesh);
  t.after(() => { geometry.dispose(); material.dispose(); });
  return mesh;
}
const nativePackets = device => device.snapshots.flat().filter(draw => draw.streams.size && draw.pipeline.vertex.entryPoint === 'main');
function uniformBlocks(draw) {
  const blocks = [];
  for (const {group, offsets} of draw.groups.values()) {
    const layout = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding);
    for (const [index, entry] of layout.entries()) {
      const resource = group.entries.find(value => value.binding === entry.binding).resource, bytes = draw.contents.get(resource.buffer);
      const offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      blocks.push([...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)]);
    }
  }
  return blocks;
}
const colorIn = (draw, color) => uniformBlocks(draw).some(words =>
  words.some((_, i) => color.every((value, j) => words[i + j] === value)));
const worldX = draw => uniformBlocks(draw).find(words => words.length === 20)[16];
const vertex = draw => {
  const bytes = draw.contents.get(draw.streams.get(0));
  return [...new Float32Array(bytes.buffer, bytes.byteOffset, 3)];
};

test('mixed node and ordinary draws keep accepted pose, material, geometry and membership through preparation', async t => {
  const f = fixture(t); f.renderer.sortObjects = false; f.mesh.frustumCulled = false;
  addNativeMesh(t, f);
  const position = f.geometry.attributes.position;
  await f.renderer.init();
  f.mesh.position.x = .25; position.setX(0, .125);
  f.renderer.render(f.scene, f.camera);
  f.mesh.position.x = .75; position.setX(0, .625); position.needsUpdate = true;
  f.material.color.setRGB(.75, .5, .25);
  f.renderer.render(f.scene, f.camera);
  const replacement = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial());
  t.after(() => { replacement.geometry.dispose(); replacement.material.dispose(); });
  f.mesh.position.x = 9; f.mesh.updateMatrixWorld(); f.mesh.geometry = null;
  f.material.color.setRGB(1, 0, 0); position.setX(0, 99);
  f.scene.remove(f.mesh); f.scene.add(replacement);
  await drain(f.renderer);
  const packets = nativePackets(f.device); assert.equal(packets.length, 4);
  assert.deepEqual([worldX(packets[0]), worldX(packets[2])], [.25, .75]);
  assert.deepEqual([vertex(packets[0])[0], vertex(packets[2])[0]], [.125, .625]);
  assert.ok(colorIn(packets[0], [.125, .375, .625])); assert.ok(colorIn(packets[2], [.75, .5, .25]));
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
});

for (const transparent of [false, true]) test(`mixed captures keep NodeFrame effects between source draws (transparent=${transparent})`, async t => {
  const f = fixture(t); f.renderer.sortObjects = false; f.mesh.frustumCulled = false;
  f.material.transparent = transparent;
  const position = f.geometry.attributes.position; position.setX(0, .125);
  let updates = 0, after = 0;
  const node = T.TSL.uniform(.5).onObjectUpdate(() => {
    updates++; f.material.color.setRGB(.625, .375, .125);
    f.mesh.matrixWorld.elements[12] = 7; second.matrixWorld.elements[12] = .5;
    position.setX(0, .875); position.needsUpdate = true;
    return .5;
  });
  node.updateAfterType = 'object'; node.updateAfter = () => { after++; f.material.color.setRGB(.75, .5, .25); };
  const native = addNativeMesh(t, f, T.TSL.vec3(node)); native.material.transparent = transparent;
  const secondGeometry = new T.PlaneGeometry(); secondGeometry.setAttribute('position', position);
  const second = new T.Mesh(secondGeometry, f.material); second.frustumCulled = false; f.scene.add(second);
  t.after(() => secondGeometry.dispose());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.deepEqual([updates, after], [1, 1], 'source node lifecycle completes synchronously');
  f.mesh.matrixWorld.elements[12] = 9; second.matrixWorld.elements[12] = 9;
  f.material.color.setRGB(1, 0, 0); position.setX(0, 99);
  await drain(f.renderer);
  const packets = nativePackets(f.device); assert.equal(packets.length, 3);
  assert.deepEqual([worldX(packets[0]), worldX(packets[2])], [0, .5]);
  assert.ok(colorIn(packets[0], [.125, .375, .625])); assert.ok(colorIn(packets[2], [.75, .5, .25]));
  assert.deepEqual([vertex(packets[0])[0], vertex(packets[2])[0]], [.125, .125],
    'a shared attribute uploads once per source render, before the intervening node writes');
  assert.deepEqual([updates, after], [1, 1], 'deferred preparation never replays node effects');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('fixed-to-mixed transition shares uploaded attribute generations and preserves partial ranges', async t => {
  const f = fixture(t); f.renderer.sortObjects = false; f.mesh.frustumCulled = false;
  const position = f.geometry.attributes.position; position.array.fill(.125);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  // The first fixed request is still queued. Changing CPU bytes alone cannot
  // initialize a second upload history when the next scene gains a node draw.
  position.array[0] = 99; position.array[1] = 77;
  const native = addNativeMesh(t, f); native.geometry = f.geometry;
  f.renderer.render(f.scene, f.camera);
  position.array[0] = .625; position.addUpdateRange(0, 1); position.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  f.scene.remove(native); f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  const packets = f.device.snapshots.flat().filter(draw => draw.streams.size);
  assert.equal(packets.length, 6);
  assert.equal(packets[0].pipeline.vertex.entryPoint, 'vertex_main');
  assert.ok(packets.slice(1).every(draw => draw.pipeline.vertex.entryPoint === 'main'));
  assert.deepEqual(packets.map(vertex), [[.125, .125, .125], [.125, .125, .125], [.125, .125, .125],
    [.625, .125, .125], [.625, .125, .125], [.625, .125, .125]]);
  assert.equal(position.updateRanges.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('mixed source updates can replace a later companion geometry before its physical draw', async t => {
  const f = fixture(t); f.renderer.sortObjects = false; f.mesh.frustumCulled = false;
  const original = new T.PlaneGeometry(), replacement = new T.PlaneGeometry();
  original.attributes.position.array.fill(.125); replacement.attributes.position.array.fill(.625);
  let calls = 0;
  addNativeMesh(t, f, T.TSL.vec3(T.TSL.uniform(.5).onObjectUpdate(() => {
    calls++; later.geometry = replacement; return .5;
  })));
  const later = new T.Mesh(original, f.material); later.frustumCulled = false; f.scene.add(later);
  t.after(() => { original.dispose(); replacement.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 1); later.geometry = null; replacement.attributes.position.array.fill(99);
  await drain(f.renderer);
  const packets = nativePackets(f.device); assert.equal(packets.length, 3);
  assert.deepEqual(vertex(packets[2]), [.625, .625, .625]);
  assert.equal(calls, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('mixed source precompilation retains restored DoubleSide normals across queued draws', async t => {
  const f = fixture(t); f.renderer.sortObjects = false;
  const native = addNativeMesh(t, f, T.TSL.normalView.mul(.25).add(.35));
  native.material.transparent = true; native.material.side = T.DoubleSide; native.scale.x = -1;
  await f.renderer.compileAsync(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const packets = nativePackets(f.device); assert.equal(packets.length, 6);
  const sides = [packets[1], packets[2], packets[4], packets[5]];
  assert.ok(sides.every(draw => draw.pipeline.fragment.module.code.includes('@builtin( front_facing )')));
  assert.deepEqual(sides.map(draw => draw.pipeline.primitive.frontFace), ['ccw', 'cw', 'ccw', 'cw']);
  assert.equal(sides[0].pipeline, sides[2].pipeline); assert.equal(sides[1].pipeline, sides[3].pipeline);
  native.material.needsUpdate = true; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.ok(nativePackets(f.device).slice(-2).every(draw => !draw.pipeline.fragment.module.code.includes('@builtin( front_facing )')));
  assert.equal(native.material.side, T.DoubleSide); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('mixed capture retains cold and warm bundle update ordering before ordinary companions', async t => {
  const f = fixture(t); f.mesh.frustumCulled = false;
  const group = new T.BundleGroup(); group.static = false; f.scene.add(group);
  let calls = 0;
  addNativeMesh(t, f, T.TSL.vec3(T.TSL.uniform(.5).onObjectUpdate(() => {
    calls++; f.material.color.setRGB(calls / 4, .375, .625); f.mesh.matrixWorld.elements[12] = calls / 4; return .5;
  })), group);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  f.material.color.setRGB(1, 0, 0); f.mesh.matrixWorld.elements[12] = 9;
  await drain(f.renderer);
  const packets = nativePackets(f.device); assert.equal(packets.length, 4);
  // Bundle node updates precede ordinary draws; their recorded GPU commands
  // execute afterward. Preserve both orders on cold and warm calls.
  assert.deepEqual([worldX(packets[0]), worldX(packets[2])], [.25, .5]);
  assert.ok(colorIn(packets[0], [.25, .375, .625])); assert.ok(colorIn(packets[2], [.5, .375, .625]));
  assert.equal(calls, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('application material fields on non-renderable groups do not become mixed scene draws', async t => {
  const f = fixture(t), group = new T.Group(); group.material = f.material; f.scene.add(group);
  addNativeMesh(t, f);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(nativePackets(f.device).length, 2);
  assert.equal(group.material, f.material); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const capacity of [3, 1025]) test(`ordinary instances retain upload history through queued mixed transitions (${capacity} matrices)`, async t => {
  const f = fixture(t), instance = new T.InstancedMesh(f.geometry, f.material, capacity);
  f.device.limits.maxUniformBufferBindingSize = 65536; f.renderer.sortObjects = false; instance.count = 2;
  instance.frustumCulled = false; f.scene.remove(f.mesh); f.scene.add(instance);
  instance.setMatrixAt(1, new T.Matrix4().makeTranslation(.5, 0, 0));
  instance.setColorAt(0, new T.Color(.125, .375, .625)); instance.setColorAt(1, new T.Color(.25, .5, .75));
  const position = f.geometry.attributes.position, matrices = instance.instanceMatrix, colors = instance.instanceColor;
  position.array.fill(.125); let before = 0, after = 0, updates = 0;
  for (const attribute of [position, matrices, colors]) Object.defineProperty(attribute, 'onUploadCallback', {
    get() { assert.fail('native instance capture must not observe WebGL upload callbacks'); },
  });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  position.array[0] = 99; colors.array[0] = 99; matrices.array[12] = 1.5;
  instance.onBeforeRender = () => { before++; }; instance.onAfterRender = () => { after++; };
  const native = addNativeMesh(t, f, T.TSL.vec3(T.TSL.uniform(.5).onObjectUpdate(() => { updates++; return .5; })));
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  position.array[0] = .625; position.array[1] = 99; position.addUpdateRange(0, 1); position.needsUpdate = true;
  colors.array[0] = .875; colors.array[1] = 99; colors.addUpdateRange(0, 1); colors.needsUpdate = true;
  matrices.array[12] = 2; matrices.array[28] = 99; matrices.addUpdateRange(12, 1); matrices.needsUpdate = true;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  f.scene.remove(native); f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  assert.deepEqual([before, after, updates], [3, 3, 2], 'source callbacks complete once per selected draw');
  position.array.fill(99); colors.array.fill(99); matrices.array.fill(99);
  await drain(f.renderer);
  const packets = nativePackets(f.device); assert.equal(packets.length, 6);
  const draws = packets.filter(draw => draw.args[1] === 2); assert.equal(draws.length, 4);
  assert.deepEqual(draws.map(vertex), [[.125, .125, .125], [.125, .125, .125], [.625, .125, .125], [.625, .125, .125]]);
  const stream = (draw, stride) => {
    const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.stepMode === 'instance' && layout.arrayStride === stride);
    if (slot < 0) return null;
    const bytes = draw.contents.get(draw.streams.get(slot)); return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  };
  assert.deepEqual(draws.map(draw => [...stream(draw, 12).slice(0, 4)]),
    [[.125, .375, .625, .25], [.125, .375, .625, .25], [.875, .375, .625, .25], [.875, .375, .625, .25]]);
  const matrixValues = draws.map(draw => stream(draw, 64) ?? uniformBlocks(draw).find(words => words.length === 48));
  assert.deepEqual(matrixValues.map(words => [words[12], words[28]]), capacity === 3
    ? [[0, .5], [0, .5], [2, 99], [2, 99]] : [[0, .5], [0, .5], [2, .5], [2, .5]]);
  assert.equal(position.updateRanges.length, 0); assert.equal(colors.updateRanges.length, 0);
  if (capacity > 3) assert.equal(matrices.updateRanges.length, 0);
  assert.deepEqual([before, after, updates], [3, 3, 2], 'deferred preparation never replays callbacks');
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
});

for (const instanceFirst of [false, true]) test(`ordinary shared instances observe aliased dynamic uploads (instance first=${instanceFirst})`, async t => {
  const f = fixture(t), instance = new T.InstancedMesh(f.geometry, f.material, 2);
  instance.frustumCulled = false; f.renderer.sortObjects = false; f.scene.remove(f.mesh);
  const native = addNativeMesh(t, f); native.geometry = f.geometry;
  f.scene.remove(native); f.scene.add(...(instanceFirst ? [instance, native] : [native, instance]));
  const position = f.geometry.attributes.position; position.setUsage(T.DynamicDrawUsage); position.array.fill(.125);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  position.array.fill(.625); f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  position.array.fill(99); await drain(f.renderer);
  const draws = nativePackets(f.device); assert.equal(draws.length, 4);
  assert.deepEqual(draws.map(vertex), [[.125, .125, .125], [.125, .125, .125], [.625, .625, .625], [.625, .625, .625]]);
  assert.equal(position.version, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const kind of ['morph', 'skin', 'morph and skin']) test(`ordinary ${kind} draws retain source deformation through queued mixed transitions`, async t => {
  const f = fixture(t), skinned = kind.includes('skin'), morphed = kind.includes('morph');
  const position = f.geometry.attributes.position; position.array.fill(.125);
  if (morphed) {
    f.geometry.morphAttributes.position = [position.clone()];
    f.geometry.morphTargetsRelative = true;
  }
  const mesh = skinned ? new T.SkinnedMesh(f.geometry, f.material) : new T.Mesh(f.geometry, f.material);
  mesh.frustumCulled = false; f.renderer.sortObjects = false; f.scene.remove(f.mesh); f.scene.add(mesh);
  let tip, updates = 0, before = 0, after = 0, nodes = 0;
  if (skinned) {
    f.geometry.setAttribute('skinIndex', new T.Uint16BufferAttribute(Array.from({length: position.count * 4}, (_, i) => i % 4 === 0 ? 1 : 0), 4));
    f.geometry.setAttribute('skinWeight', new T.Uint8BufferAttribute(Array.from({length: position.count * 4}, (_, i) => i % 4 === 0 ? 255 : 0), 4, true));
    const root = new T.Bone(); tip = new T.Bone(); root.add(tip); mesh.add(root); mesh.bind(new T.Skeleton([root, tip]));
    const update = mesh.skeleton.update; mesh.skeleton.update = function () { updates++; return update.call(this); };
    tip.position.x = .25;
    if (!morphed) {
      const shared = new T.SkinnedMesh(f.geometry, f.material); shared.bind(mesh.skeleton, mesh.bindMatrix);
      shared.frustumCulled = false; shared.position.x = .5; f.scene.add(shared);
    }
  }
  if (morphed) mesh.morphTargetInfluences[0] = .25;
  for (const attribute of Object.values(f.geometry.attributes)) Object.defineProperty(attribute, 'onUploadCallback', {
    get() { assert.fail('native deformation must not observe WebGL upload callbacks'); },
  });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  position.array.fill(99); if (tip) tip.position.x = .5;
  if (morphed) mesh.morphTargetInfluences[0] = .5;
  mesh.onBeforeRender = () => { before++; }; mesh.onAfterRender = () => { after++; };
  const native = addNativeMesh(t, f, T.TSL.vec3(T.TSL.uniform(.5).onObjectUpdate(() => { nodes++; return .5; })));
  f.renderer.render(f.scene, f.camera);
  if (skinned) { f.scene.updateMatrixWorld(true); mesh.skeleton.update(); }
  position.array[0] = .625; position.addUpdateRange(0, 1); position.needsUpdate = true;
  if (morphed) {
    mesh.morphTargetInfluences = [.75];
    f.geometry.morphAttributes.position[0].array.fill(9); f.geometry.morphAttributes.position[0].needsUpdate = true;
  }
  f.renderer.render(f.scene, f.camera);
  f.scene.remove(native); mesh.static = true;
  if (tip) tip.position.x = .875; if (morphed) mesh.morphTargetInfluences[0] = .875;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  assert.deepEqual([before, after, nodes], [3, 3, 2]); if (skinned) assert.equal(updates, 3, 'shared skeleton updates once per frame plus the explicit update');
  position.array.fill(99); if (skinned) mesh.skeleton.boneMatrices.fill(99); if (morphed) mesh.morphTargetInfluences.fill(99);
  await drain(f.renderer);
  const draws = nativePackets(f.device).filter(draw => draw.pipeline.vertex.module.code.includes(skinned ? 'skinWeight' : 'texture_2d_array'));
  const repeated = values => values.flatMap(value => kind === 'skin' ? [value, value] : [value]);
  assert.equal(draws.length, kind === 'skin' ? 8 : 4);
  assert.deepEqual(draws.map(vertex), repeated([[.125, .125, .125], [.125, .125, .125], [.625, .125, .125], [.625, .125, .125]]));
  if (skinned) for (const [i, translation] of repeated([.25, .25, .5, .875]).entries())
    assert.ok(uniformBlocks(draws[i]).some(words => words.length === 32 && words[28] === translation), `source bone palette ${translation} survives deferred preparation`);
  if (morphed) for (const [i, influence] of [.25, .5, .75, skinned ? .875 : .75].entries())
    assert.ok(uniformBlocks(draws[i]).some(words => words.length === 4 && words[0] === influence && words.slice(1).every(value => value === 0)), `source influence ${influence} survives deferred preparation`);
  assert.deepEqual([before, after, nodes], [3, 3, 2]); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const legacy of ['normal-only morph', 'nonstandard palette']) test(`automatic skin capture preserves the existing ${legacy} route`, async t => {
  const f = fixture(t), count = f.geometry.attributes.position.count;
  f.geometry.setAttribute('skinIndex', new T.Uint16BufferAttribute(new Uint16Array(count * 4), 4));
  f.geometry.setAttribute('skinWeight', new T.Float32BufferAttribute(Array.from({length: count * 4}, (_, i) => i % 4 === 0 ? 1 : 0), 4));
  if (legacy === 'normal-only morph') f.geometry.morphAttributes.normal = [f.geometry.attributes.normal.clone()];
  const mesh = new T.SkinnedMesh(f.geometry, f.material), bone = new T.Bone(); mesh.add(bone); mesh.bind(new T.Skeleton([bone]));
  if (legacy === 'nonstandard palette') mesh.skeleton.boneMatrices = new Float64Array(mesh.skeleton.boneMatrices);
  mesh.frustumCulled = false; f.scene.remove(f.mesh); f.scene.add(mesh); await f.renderer.init();
  const captured = f.renderer._captureNodeFrame({scene: f.scene, camera: f.camera, destination: f.renderer,
    overrideMaterial: null, captureCameras: true, renderId: 1});
  assert.equal(captured.objects.has(mesh), false); assert.equal(captured.draws.length, 0);
  assert.equal(captured.bytes, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('growing an empty-capacity instance retains its established fixed upload history', async t => {
  const f = fixture(t), instance = new T.InstancedMesh(f.geometry, f.material, 0);
  f.scene.remove(f.mesh); f.scene.add(instance); instance.frustumCulled = false;
  const position = f.geometry.attributes.position; position.array.fill(.125);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  instance.instanceMatrix = new T.InstancedBufferAttribute(new Float32Array(new T.Matrix4().elements), 16); instance.count = 1;
  position.array.fill(99); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draws = f.device.snapshots.flat().filter(draw => draw.streams.size && draw.args[1] > 0);
  assert.equal(draws.length, 1); assert.deepEqual(vertex(draws[0]), [.125, .125, .125]);
  assert.equal(draws[0].pipeline.vertex.entryPoint, 'vertex_main'); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a refused legacy fixed-shadow transition preserves its prior route and GPU-stale bytes', async t => {
  const f = fixture(t);
  f.renderer.shadowMap.enabled = true; f.mesh.castShadow = true; f.mesh.frustumCulled = false;
  const position = f.geometry.attributes.position; position.array.fill(.125);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  position.array[0] = 99; f.renderer.shadowMap.enabled = false;
  const native = addNativeMesh(t, f);
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
  f.scene.remove(native); f.renderer.shadowMap.enabled = true; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const packets = f.device.snapshots.flat().filter(draw => draw.streams.size);
  assert.equal(packets.length, 2); assert.deepEqual(packets.map(vertex), [[.125, .125, .125], [.125, .125, .125]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer.info.f3d.presentedRenders, 2);
});
