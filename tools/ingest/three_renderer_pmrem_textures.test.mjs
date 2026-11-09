/** Source material-map captures for retained PMREM and owned GPU submission. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass, createPMREMGeneratorClass} from './three_renderer.mjs';
import {createThreePassSnapshots} from './three_pass_snapshot.mjs';
import {createGpuThreeTextures} from './three_textures.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGLRendererClass(T, {maxDraws: 64, maxBindings: 64});
const Generator = createPMREMGeneratorClass(T);
function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.textures = []; device.textureWrites = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
    destination.texture.pixels = bytes; device.textureWrites.push({texture: destination.texture, bytes});
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) for (const draw of pass.draws)
      for (const {group} of draw.groups.values()) for (const entry of group.entries ?? []) if (entry.resource?.texture)
        assert.notEqual(entry.resource.texture.destroyed, true, 'accepted sampled input remains live through submission');
    submit(commands);
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }})};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  return {renderer, device};
}
function texture(value = 32) {
  const map = new T.DataTexture(new Uint8Array(16).fill(value), 2, 2);
  map.needsUpdate = true; return map;
}
function room(map, Type = T.MeshBasicMaterial) {
  const root = new T.Mesh(new T.BoxGeometry(10, 10, 10), new Type({map, side: T.BackSide}));
  root.frustumCulled = false; return root;
}
const drain = async renderer => { await renderer.waitForGPU(); await Promise.resolve(); };
const sceneWrites = device => device.textureWrites.filter(write => write.texture.width === 2 && write.texture.height === 2);

test('textured scene bakes retain distinct marked uploads, stale unmarked bytes and disposal lifetime', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const map = texture(32), root = room(map), generator = new Generator(renderer), outputs = [];
  const bake = () => outputs.push(generator.fromScene(root, .02, .1, 20, {size: 16}));
  bake(); map.image.data.fill(64); bake();
  map.image.data.fill(96); map.needsUpdate = true; bake();
  map.image.data.fill(200); map.addUpdateRange(4, 4); map.needsUpdate = true; bake();
  root.material.map = null; map.image.data.fill(0); map.dispose();
  root.geometry.dispose(); root.material.dispose(); generator.dispose(); await drain(renderer);
  const writes = sceneWrites(device).map(write => Array.from(write.bytes));
  assert.deepEqual(writes, [new Array(16).fill(32), new Array(16).fill(96),
    [...new Array(4).fill(96), ...new Array(4).fill(200), ...new Array(8).fill(96)]]);
  assert.equal(renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
  assert.equal(renderer._sampledTextureInputs.diagnostics.versions, 0);
  assert.equal(renderer._passSnapshots.rootCount, 0);
  for (const output of outputs) output.dispose();
});

test('map transforms are captured independently of upload versions and installs retain each accepted generation', async t => {
  const {renderer} = fixture(t); await renderer.init();
  const map = texture(), root = room(map), owner = createThreePassSnapshots(T), camera = new T.PerspectiveCamera();
  const capture = () => owner.capturePMREMScene(root, camera, renderer._device,
    {captureTexture: source => renderer._capturePMREMTexture(source)});
  const first = capture(), version = map.version;
  map.offset.set(.25, .5); map.repeat.set(3, 2); map.center.set(.1, .2); map.rotation = .6;
  const second = capture(), expected = map.matrix.elements.slice();
  map.offset.set(9, 9); map.repeat.set(9, 9); map.center.set(9, 9); map.rotation = 9; map.matrix.identity();
  first.install(); assert.deepEqual(first.root.material.map.offset.toArray(), [0, 0]);
  second.install(); assert.deepEqual(second.root.material.map.offset.toArray(), [.25, .5]);
  assert.deepEqual(second.root.material.map.repeat.toArray(), [3, 2]);
  assert.equal(second.root.material.map.rotation, .6);
  assert.deepEqual(second.root.material.map.matrix.elements, expected); assert.equal(map.version, version);
  assert.notEqual(second.root.material.map.matrix, map.matrix);
  first.release(); second.release(); owner.dispose(); map.dispose(); root.geometry.dispose(); root.material.dispose();
});

test('hidden, invisible-material, excluded-layer and culled maps do not establish uploaded history', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const map = texture(32), root = room(map), generator = new Generator(renderer), outputs = [];
  const bake = () => outputs.push(generator.fromScene(root, 0, .1, 20, {size: 16}));
  root.visible = false; bake();
  root.visible = true; root.material.visible = false; bake();
  root.material.visible = true; root.layers.set(1); bake();
  root.layers.set(0); root.frustumCulled = true; root.position.z = 100; bake();
  map.image.data.fill(144); root.position.z = 0; bake();
  map.dispose(); generator.dispose(); await drain(renderer);
  assert.deepEqual(sceneWrites(device).map(write => write.bytes[0]), [144]);
  for (const output of outputs) output.dispose(); root.geometry.dispose(); root.material.dispose();
});

test('ordinary material selection captures active diffuse, normal, alpha, gradient and PBR maps only', () => {
  const map = texture(), normal = texture(127), alpha = texture(255), detail = texture(80), camera = new T.PerspectiveCamera();
  const cases = [
    [new T.MeshBasicMaterial({map, alphaMap: alpha}), [map, alpha]],
    [new T.MeshLambertMaterial({map, normalMap: normal}), [map, normal]],
    [new T.MeshPhongMaterial({map, normalMap: normal, specularMap: detail}), [map, normal, detail]],
    [new T.MeshToonMaterial({map, gradientMap: detail}), [map, detail]],
    [new T.MeshStandardMaterial({map, normalMap: normal, roughnessMap: detail, metalnessMap: detail}), [map, normal, detail]],
    [new T.MeshNormalMaterial({normalMap: normal}), [normal]],
    [new T.MeshMatcapMaterial({matcap: detail, map}), [map, detail]],
    [new T.MeshDepthMaterial({map, alphaMap: alpha}), [map, alpha]],
  ];
  for (const [material, expected] of cases) {
    const owner = createThreePassSnapshots(T), mesh = new T.Mesh(new T.BoxGeometry(), material), seen = [], scene = new T.Scene();
    scene.add(mesh, new T.PointLight(0xffffff, 2));
    mesh.frustumCulled = false;
    // This source property is not consumed by any of these shader programs.
    material.unusedTexture = texture(17);
    const capture = owner.capturePMREMScene(scene, camera, geometryDevice(), {captureTexture(source) {
      seen.push(source); return {texture: Object.create(Object.getPrototypeOf(source), Object.getOwnPropertyDescriptors(source)), release() {}};
    }});
    assert.deepEqual(new Set(seen), new Set(expected), material.type);
    capture.release(); owner.dispose(); mesh.geometry.dispose(); material.unusedTexture.dispose(); material.dispose();
  }
  for (const input of [map, normal, alpha, detail]) input.dispose();
});

test('one shared Source across successive bakes and ordinary maps retains the accepted history', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const map = texture(32), alias = map.clone(), root = room(map), generator = new Generator(renderer), outputs = [];
  outputs.push(generator.fromScene(root, 0, .1, 20, {size: 16})); await drain(renderer);
  map.image.data.fill(64); root.material.map = alias;
  outputs.push(generator.fromScene(root, 0, .1, 20, {size: 16})); await drain(renderer);
  assert.ok(sceneWrites(device).every(write => write.bytes[0] === 32));
  alias.needsUpdate = true; outputs.push(generator.fromScene(root, 0, .1, 20, {size: 16}));
  const ordinary = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial({map})), camera = new T.PerspectiveCamera(); camera.position.z = 2;
  renderer.render(ordinary, camera); await drain(renderer);
  assert.equal(sceneWrites(device).at(-1).bytes[0], 64);
  map.dispose(); alias.dispose(); generator.dispose(); await drain(renderer);
  for (const output of outputs) output.dispose();
  for (const mesh of [root, ordinary]) { mesh.geometry.dispose(); mesh.material.dispose(); }
});

test('source-inactive bump and Toon gradient samplers retain fresh first-use bytes when enabled', async t => {
  for (const kind of ['bump', 'gradient']) {
    const {renderer, device} = fixture(t); await renderer.init();
    const inactive = texture(32), normal = texture(127), scene = new T.Scene(), generator = new Generator(renderer);
    const material = kind === 'bump'
      ? new T.MeshNormalMaterial({normalMap: normal, bumpMap: inactive, side: T.BackSide})
      : new T.MeshToonMaterial({gradientMap: inactive, side: T.BackSide});
    const mesh = new T.Mesh(new T.BoxGeometry(10, 10, 10), material); scene.add(mesh, new T.AmbientLight(0xffffff, .3));
    const first = generator.fromScene(scene, 0, .1, 20, {size: 16});
    inactive.image.data.fill(144); // No upload request: this map has never been sampled.
    if (kind === 'bump') { material.normalMap = null; material.needsUpdate = true; }
    else scene.add(new T.PointLight(0xffffff, 2));
    const second = generator.fromScene(scene, 0, .1, 20, {size: 16});
    inactive.dispose(); normal.dispose(); generator.dispose(); await drain(renderer);
    assert.deepEqual(sceneWrites(device).map(write => write.bytes[0]), kind === 'bump' ? [127, 144] : [144], kind);
    first.dispose(); second.dispose(); mesh.geometry.dispose(); material.dispose();
  }
});

test('texture disposal starts fresh residency while earlier scene captures remain queued', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const map = texture(32), root = room(map), generator = new Generator(renderer);
  const first = generator.fromScene(root, 0, .1, 20, {size: 16});
  map.dispose(); map.image.data.fill(96);
  const second = generator.fromScene(root, 0, .1, 20, {size: 16});
  map.dispose(); generator.dispose(); await drain(renderer);
  assert.deepEqual(sceneWrites(device).map(write => write.bytes[0]), [32, 96]);
  first.dispose(); second.dispose(); root.geometry.dispose(); root.material.dispose();
});

test('callback, foreign residency and same-bake Source-alias boundaries fail before accepting scene work', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const map = texture(), root = room(map), generator = new Generator(renderer);
  let callbacks = 0; map.onUpdate = () => { callbacks++; };
  assert.throws(() => generator.fromScene(root, 0, .1, 20, {size: 16}), /upload callbacks/);
  assert.equal(callbacks, 0); map.onUpdate = null;
  const alias = map.clone(), second = room(alias), scene = new T.Scene(); scene.add(root, second);
  assert.throws(() => generator.fromScene(scene, 0, .1, 20, {size: 16}), /Source aliases/);
  const pool = createGpuThreeTextures(device, {three: T, webglVersioning: true}); pool.prepare([map]);
  assert.throws(() => generator.fromScene(root, 0, .1, 20, {size: 16}), /separate scene texture owner/);
  map.dispose(); alias.dispose(); pool.dispose(); generator.dispose();
  assert.equal(renderer._passSnapshots.rootCount, 0);
  for (const mesh of [root, second]) { mesh.geometry.dispose(); mesh.material.dispose(); }
});

test('discarded image-readiness probes observe rejection while awaited preparation keeps the original error', async () => {
  const map = texture(), root = room(map), owner = createThreePassSnapshots(T), expected = new Error('image snapshot failed');
  const pending = Promise.reject(expected); pending.catch(() => {});
  let releases = 0;
  const capture = owner.capturePMREMScene(root, new T.PerspectiveCamera(), geometryDevice(), {captureTexture(source) {
    return {texture: Object.create(Object.getPrototypeOf(source), Object.getOwnPropertyDescriptors(source)),
      owner: {preparation: () => pending}, lease: {}, release() { releases++; }};
  }});
  capture.preparation(); // The renderer's synchronous admission probe does not await this token.
  await assert.rejects(capture.preparation(), error => error === expected);
  await new Promise(resolve => setImmediate(resolve));
  capture.release(); owner.dispose(); assert.equal(releases, 1);
  map.dispose(); root.geometry.dispose(); root.material.dispose();
});

test('a rejected later map rolls back earlier upload history and ordinary-source routing', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const map = texture(11), alpha = texture(255), root = room(map), generator = new Generator(renderer);
  root.material.transparent = true; root.material.alphaMap = alpha; alpha.onUpdate = () => {};
  const submissions = device.submissions.length;
  assert.throws(() => generator.fromScene(root, 0, .1, 20, {size: 16}), /upload callbacks/);
  assert.equal(device.submissions.length, submissions); assert.equal(renderer._passSnapshots.rootCount, 0);
  assert.equal(renderer._sampledTextureInputs.diagnostics.versions, 0);
  assert.equal(renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
  assert.equal(renderer._framebufferBindings().has(map), false, 'failed admission leaves ordinary routing unchanged');
  map.image.data.fill(77);
  const input = renderer._capturePMREMTexture(map);
  assert.equal(input.texture.image.data[0], 77, 'the first successful capture sees current unmarked pixels');
  input.release(); generator.dispose(); map.dispose(); alpha.dispose(); root.geometry.dispose(); root.material.dispose();
});
