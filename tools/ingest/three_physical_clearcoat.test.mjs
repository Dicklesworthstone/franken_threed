/** Pinned source materials through the native scene bridge and draw packer.
 * This recording device proves bindings, source transitions and queue snapshots;
 * differential browser scenarios check the actual shaded pixels.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createGpuThreeScene} from './three_scene.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
function fixture(type = 'MeshPhysicalMaterial') {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 4096, maxBindingsPerBindGroup: 32});
  const scene = new T.Scene(), geometry = new T.PlaneGeometry(1, 1);
  for (const name of ['uv1', 'uv2', 'uv3']) geometry.setAttribute(name, geometry.attributes.uv.clone());
  const material = new T[type]({clearcoat: .75, clearcoatRoughness: .25, roughness: .6});
  const object = new T.Mesh(geometry, material); object.frustumCulled = false; scene.add(object);
  const light = new T.DirectionalLight(0xffffff, 2); light.position.z = 3; scene.add(light);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 3;
  const textures = new Map();
  function texture(channel = 0) {
    const source = new T.Texture(); source.channel = channel;
    textures.set(source, {view: {}, sampler: {}, version: source.version, sourceVersion: source.source.version});
    return source;
  }
  return {device, scene, geometry, object, material, camera, textures, texture};
}
async function prepare(h, t, options = {}) {
  const bridge = await createGpuThreeScene(h.device, h.scene, {three: T, textures: h.textures,
    textureTransforms: true, renderer: {maxDraws: 16}, ...options});
  t.after(() => bridge.dispose()); return bridge;
}
const frame = () => ({colorView: {}, depthView: {}});
const draws = h => h.device.snapshots.flat().filter(d => d.streams.has(0));
const coatBuffer = d => d.groups.get(1)?.group.entries.find(e => e.binding === 16)?.resource.buffer;
const coat = d => new Float32Array(d.contents.get(coatBuffer(d)).buffer);
const f32 = values => new Float32Array(values);
function uv(draw, slot) {
  const group = draw.groups.get(0), buffer = group.group.entries[0].resource.buffer;
  const words = new Float32Array(draw.contents.get(buffer).buffer);
  return words.slice((group.offsets[0] ?? 0) / 4 + 64 + slot * 8, (group.offsets[0] ?? 0) / 4 + 72 + slot * 8);
}
const matrix = texture => {
  const e = texture.matrix.elements; return f32([e[0], e[3], e[6], 0, e[1], e[4], e[7], 0]);
};

for (const type of ['MeshPhysicalMaterial', 'MeshPhysicalNodeMaterial'])
test(`${type} clearcoat uses native physical shading and live coating uniforms`, async t => {
  const h = fixture(type), bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const first = draws(h).at(-1), pipelines = h.device.pipelines.length, buffers = h.device.buffers.length;
  assert.deepEqual(coat(first), f32([.75, .25, 1, 1]));
  assert.match(first.pipeline.fragment.module.code, /specular_lobe: bool/);
  h.material.clearcoat = .3; h.material.clearcoatRoughness = .8;
  bridge.render(h.camera, frame()); const last = draws(h).at(-1);
  assert.deepEqual(coat(last), f32([.3, .8, 1, 1]));
  assert.deepEqual(coat(first), f32([.75, .25, 1, 1]), 'already submitted uniforms retain their first values');
  assert.equal(coatBuffer(last), coatBuffer(first)); assert.equal(h.device.pipelines.length, pipelines);
  assert.equal(h.device.buffers.length, buffers); assert.equal(bridge.diagnostics.prepareVersion, 1);
  await bridge.whenIdle();
});

test('shared geometry keeps independent mutable clearcoat packets per material', async t => {
  const h = fixture(), other = h.material.clone(); other.clearcoat = .2; other.clearcoatRoughness = .9;
  const object = new T.Mesh(h.geometry, other); object.frustumCulled = false; object.renderOrder = 1; h.scene.add(object);
  const bridge = await prepare(h, t, {renderer: {maxDraws: 16, instancing: true, renderBundles: true}});
  bridge.render(h.camera, frame()); const first = draws(h);
  assert.equal(first.length, 2); assert.notEqual(coatBuffer(first[0]), coatBuffer(first[1]));
  assert.deepEqual(first.map(d => Array.from(coat(d))), [[.75, .25, 1, 1], Array.from(f32([.2, .9, 1, 1]))]);
  const bundles = h.device.bundleEncoders.length; h.material.clearcoat = .5; bridge.render(h.camera, frame());
  assert.equal(h.device.bundleEncoders.length, bundles);
  assert.deepEqual(draws(h).slice(-2).map(d => Array.from(coat(d))), [[.5, .25, 1, 1], Array.from(f32([.2, .9, 1, 1]))]);
  assert.equal(bridge.diagnostics.geometryCount, 1);
});

test('all three coat maps keep independent channels, transforms and XY normal scales', async t => {
  const h = fixture(), maps = [h.texture(1), h.texture(2), h.texture(3)];
  [h.material.clearcoatMap, h.material.clearcoatRoughnessMap, h.material.clearcoatNormalMap] = maps;
  maps.forEach((map, i) => { map.repeat.set(i + 1, i + 2); map.offset.set(i / 8, i / 16); });
  h.material.clearcoatNormalScale.set(.25, 1.5);
  const bridge = await prepare(h, t); bridge.render(h.camera, frame()); const first = draws(h).at(-1);
  assert.deepEqual(coat(first), f32([.75, .25, .25, 1.5]));
  for (let i = 0; i < 3; i++) {
    assert.deepEqual(uv(first, i + 5), matrix(maps[i]));
    assert.equal(first.groups.get(1).group.entries.find(e => e.binding === 11 + 2 * i).resource, h.textures.get(maps[i]).view);
  }
  assert.match(first.pipeline.fragment.module.code, /coat_source_roughness = clearcoat_info.y \* clearcoat_roughness_texel.r/);
  const pipelines = h.device.pipelines.length;
  maps[0].rotation = .7; maps[1].offset.x = .6; maps[2].repeat.y = 4; h.material.clearcoatNormalScale.set(-.5, .75);
  bridge.render(h.camera, frame()); const last = draws(h).at(-1);
  for (let i = 0; i < 3; i++) assert.deepEqual(uv(last, i + 5), matrix(maps[i]));
  assert.deepEqual(coat(last), f32([.75, .25, -.5, .75])); assert.equal(h.device.pipelines.length, pipelines);
  maps[2].channel = 1; assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(h.camera, frame());
  assert.match(draws(h).at(-1).pipeline.vertex.module.code, /out.uv_7 = .*uv1/);
});

test('zero/nonzero transitions prepare the matching source variant without stale coating', async t => {
  const h = fixture(), bridge = await prepare(h, t); bridge.render(h.camera, frame());
  h.material.clearcoat = 0;
  assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(h.camera, frame()); assert.equal(coatBuffer(draws(h).at(-1)), undefined);
  h.material.clearcoat = 1;
  assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(h.camera, frame()); assert.deepEqual(coat(draws(h).at(-1)), f32([1, .25, 1, 1]));
});

test('invalid coating and other active physical lobes fail before new queue work', async t => {
  const h = fixture(), bridge = await prepare(h, t); bridge.render(h.camera, frame());
  for (const [key, value] of [['clearcoat', NaN], ['clearcoatRoughness', Infinity], ['transmission', .5], ['sheen', 1]]) {
    const before = [h.device.writes.length, h.device.submissions.length], previous = h.material[key];
    h.material[key] = value; assert.throws(() => bridge.render(h.camera, frame()), /THREE_SCENE_(VALUE|MATERIAL)/);
    assert.deepEqual([h.device.writes.length, h.device.submissions.length], before); h.material[key] = previous;
  }
  bridge.render(h.camera, frame()); assert.equal(bridge.failed, false);
});
