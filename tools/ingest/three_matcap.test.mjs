/** Source Matcap admission, live bindings and per-use packets. Pixel equations
 * are compared with the pinned renderer by the native_matcap browser scenes. */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
function fixture(type = 'MeshMatcapMaterial') {
  const device = geometryDevice(); device.limits.maxBindingsPerBindGroup = 32;
  const scene = new T.Scene(), geometry = new T.PlaneGeometry(1, 1);
  const material = new T[type]({color: new T.Color(1.4, .3, .2)});
  const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; scene.add(mesh);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 3;
  const textures = new Map();
  const texture = () => {
    const t = new T.Texture();
    textures.set(t, {view: {}, sampler: {}, version: t.version, sourceVersion: t.source.version});
    return t;
  };
  return {device, scene, geometry, material, mesh, camera, textures, texture};
}
async function prepare(h, t, options = {}) {
  const bridge = await createGpuThreeScene(h.device, h.scene, {three: T,
    textures: h.textures, textureTransforms: true, alphaMaps: true,
    renderer: {maxDraws: 8}, ...options});
  t.after(() => bridge.dispose()); return bridge;
}
const frame = () => ({colorView: {}, depthView: {}});
const last = h => h.device.snapshots.flat().filter(d => d.streams.has(0)).at(-1);
function words(draw) {
  const {group, offsets} = draw.groups.get(0), buffer = group.entries[0].resource.buffer;
  return new Float32Array(draw.contents.get(buffer).buffer, offsets[0] ?? 0);
}
const uv = (draw, slot) => words(draw).slice(64 + slot * 8, 72 + slot * 8);
const matrix = t => {
  const e = t.matrix.elements;
  return Float32Array.from([e[0], e[3], e[6], 0, e[1], e[4], e[7], 0]);
};
const views = draw => draw.groups.get(1)?.group.entries.map(e => e.resource) ?? [];

for (const type of ['MeshMatcapMaterial', 'MeshMatcapNodeMaterial'])
test(`${type} renders its fallback without lights or UV attributes and keeps color live`, async t => {
  const h = fixture(type); h.geometry.deleteAttribute('uv');
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const first = last(h), pipelines = h.device.pipelines.length;
  assert.deepEqual(words(first).slice(16, 20), Float32Array.from([1.4, .3, .2, 1]));
  h.material.color.setRGB(.2, 1.8, .7); bridge.render(h.camera, frame());
  assert.deepEqual(words(last(h)).slice(16, 20), Float32Array.from([.2, 1.8, .7, 1]));
  assert.deepEqual(words(first).slice(16, 20), Float32Array.from([1.4, .3, .2, 1]));
  assert.equal(h.device.pipelines.length, pipelines); assert.equal(bridge.diagnostics.prepareVersion, 1);
});

test('generated Matcap coordinates ignore channel and retain independent live matrices', async t => {
  const h = fixture(); h.geometry.deleteAttribute('uv');
  h.material.matcap = h.texture(); h.material.matcap.channel = 3;
  h.material.matcap.repeat.set(.6, .8); h.material.matcap.offset.set(.1, .2);
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const first = last(h), pipelines = h.device.pipelines.length;
  assert.ok(views(first).includes(h.textures.get(h.material.matcap).view));
  assert.deepEqual(uv(first, 1), matrix(h.material.matcap));
  h.material.matcap.channel = 17; h.material.matcap.rotation = .4;
  bridge.render(h.camera, frame());
  assert.deepEqual(uv(last(h), 1), matrix(h.material.matcap));
  assert.notDeepEqual(uv(first, 1), uv(last(h), 1));
  assert.equal(h.device.pipelines.length, pipelines, 'ignored channel edits do not require new geometry or pipelines');
});

test('Matcap, diffuse, normal and opacity textures keep separate bindings and transforms', async t => {
  const h = fixture();
  for (const [i, name] of ['map', 'matcap', 'normalMap', 'alphaMap'].entries()) {
    h.material[name] = h.texture(); h.material[name].offset.set(i * .1, i * .05);
  }
  h.material.alphaTest = .4; h.material.normalScale.set(.7, 1.2);
  h.scene.fog = new T.Fog(0x102030, 1, 9);
  const bridge = await prepare(h, t, {fog: {}}); bridge.render(h.camera, frame());
  const draw = last(h);
  for (const [name, slot] of [['map', 0], ['matcap', 1], ['normalMap', 2], ['alphaMap', 8]]) {
    assert.ok(views(draw).includes(h.textures.get(h.material[name]).view));
    assert.deepEqual(uv(draw, slot), matrix(h.material[name]));
  }
  assert.equal(words(draw)[20], Math.fround(.4));
  await bridge.whenIdle();
});

test('Matcap presence and texture identity changes cross the existing preparation boundary', async t => {
  const h = fixture(), bridge = await prepare(h, t); bridge.render(h.camera, frame());
  h.material.matcap = h.texture();
  assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(h.camera, frame());
  assert.ok(views(last(h)).includes(h.textures.get(h.material.matcap).view));
  h.material.matcap = h.texture();
  assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(h.camera, frame());
  assert.ok(views(last(h)).includes(h.textures.get(h.material.matcap).view));
  h.material.matcap = null; await bridge.prepare(); bridge.render(h.camera, frame());
  assert.equal(bridge.diagnostics.prepareVersion, 4);
});

test('invalid live Matcap transforms fail before queue writes and leave the bridge usable', async t => {
  const h = fixture(); h.material.matcap = h.texture();
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  h.material.matcap.matrixAutoUpdate = false; h.material.matcap.matrix.elements[0] = NaN;
  const before = [h.device.writes.length, h.device.submissions.length];
  assert.throws(() => bridge.render(h.camera, frame()), /UV coefficients must fit finite f32/);
  assert.deepEqual([h.device.writes.length, h.device.submissions.length], before);
  h.material.matcap.matrix.identity(); bridge.render(h.camera, frame());
  assert.equal(bridge.failed, false);
});
