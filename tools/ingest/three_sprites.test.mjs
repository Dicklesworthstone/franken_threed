/** Source sprite state through native draw packing; pixel parity is separate. */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {threeSpriteWorldMatrix} from './three_sprites.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
const {vec3} = await import(pathToFileURL(path.join(root, 'build/three.tsl.js')));
const frame = () => ({colorView: {}, depthView: {}});
function fixture(type = 'SpriteMaterial') {
  const device = geometryDevice(); device.limits.maxBindingsPerBindGroup = 32;
  const scene = new T.Scene(), material = new T[type]({color: 0xff8040, opacity: .5});
  const sprite = new T.Sprite(material); scene.add(sprite);
  const camera = new T.PerspectiveCamera(55, 1, .1, 100); camera.position.set(1, 2, 5); camera.lookAt(0, 0, 0);
  return {device, scene, material, sprite, camera};
}
async function prepare(h, t, options = {}) {
  const bridge = await createGpuThreeScene(h.device, h.scene, {three: T, renderer: {maxDraws: 16}, ...options});
  t.after(() => bridge.dispose()); return bridge;
}
const draws = h => h.device.snapshots.flat().filter(d => d.streams.has(0));
function packet(draw) {
  const {group, offsets} = draw.groups.get(0), buffer = group.entries[0].resource.buffer;
  return new Float32Array(draw.contents.get(buffer).buffer, offsets[0] ?? 0, 64);
}
function near(actual, expected, epsilon = 2e-5) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) <= epsilon * Math.max(1, Math.abs(expected[i])), `${i}: ${value} != ${expected[i]}`));
}
function sourcePosition(sprite, material, camera, position) {
  // Independent literal evaluation of r186 SpriteNodeMaterial.setupPositionView:
  // model-view origin, world-column lengths, center offset, then 2D rotation.
  const mv = new T.Vector4(0, 0, 0, 1).applyMatrix4(sprite.matrixWorld).applyMatrix4(camera.matrixWorldInverse);
  const e = sprite.matrixWorld.elements, k = camera.isPerspectiveCamera && !material.sizeAttenuation ? -mv.z : 1;
  const x = (position[0] - (sprite.center.x - .5)) * Math.hypot(e[0], e[1], e[2]) * k;
  const y = (position[1] - (sprite.center.y - .5)) * Math.hypot(e[4], e[5], e[6]) * k;
  const c = Math.cos(material.rotation), s = Math.sin(material.rotation);
  mv.x += c * x - s * y; mv.y += s * x + c * y;
  return mv.applyMatrix4(camera.projectionMatrix).toArray();
}

for (const type of ['SpriteMaterial', 'SpriteNodeMaterial'])
test(`${type} draws camera-facing source geometry with live center, scale and rotation`, async t => {
  const h = fixture(type), parent = new T.Group(); h.scene.add(parent); parent.add(h.sprite);
  // This camera's general inverse produces homogeneous w = 1.0000000000000002.
  h.camera.position.set(1, .6, 5); h.camera.lookAt(0, 0, 0);
  parent.scale.set(-2, 3, .5); parent.rotation.set(.4, -.7, .2); h.sprite.rotation.set(.2, .6, .8);
  h.sprite.center.set(.15, .8); h.material.rotation = .6;
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const first = draws(h).at(-1); assert.equal(first.args[0], 6); assert.equal(first.pipeline.primitive.frontFace, 'ccw');
  const vp = new T.Matrix4().multiplyMatrices(h.camera.projectionMatrix, h.camera.matrixWorldInverse);
  if (h.camera.coordinateSystem === T.WebGLCoordinateSystem)
    for (let c = 0; c < 4; c++) vp.elements[c * 4 + 2] = .5 * (vp.elements[c * 4 + 2] + vp.elements[c * 4 + 3]);
  const billboard = threeSpriteWorldMatrix(T, h.sprite, h.material, h.camera);
  assert.deepEqual([billboard[3], billboard[7], billboard[11], billboard[15]], [0, 0, 0, 1]);
  near(Array.from(packet(first).slice(0, 16)), new T.Matrix4().multiplyMatrices(vp,
    new T.Matrix4().fromArray(billboard)).elements);
  const pipelines = h.device.pipelines.length, publicMatrix = h.sprite.matrixWorld.elements.slice();
  h.sprite.center.set(.9, .25); h.material.rotation = -.3; h.material.sizeAttenuation = false; h.camera.position.z = 8;
  bridge.render(h.camera, frame()); assert.notDeepEqual(packet(draws(h).at(-1)).slice(0, 16), packet(first).slice(0, 16));
  assert.equal(h.device.pipelines.length, pipelines); assert.equal(bridge.diagnostics.prepareVersion, 1);
  assert.deepEqual(h.sprite.matrixWorld.elements, publicMatrix, 'billboard lowering never rewrites the public world matrix');
});

test('billboard projection matches source formula with perspective, orthographic, shear and ignored local Z', () => {
  const h = fixture(); h.sprite.matrixAutoUpdate = false;
  h.sprite.matrix.set(-2, .4, 0, 1, .5, 3, .2, -.5, .3, -.2, .8, -2, 0, 0, 0, 1);
  h.scene.updateMatrixWorld(); h.sprite.center.set(.2, .9); h.material.rotation = 1.1;
  const ortho = new T.OrthographicCamera(-3, 3, 3, -3, .1, 100); ortho.position.set(-2, 1, 5); ortho.lookAt(0, 0, 0);
  for (const camera of [h.camera, ortho]) for (const attenuation of [true, false]) {
    camera.updateMatrixWorld(); h.material.sizeAttenuation = attenuation;
    const world = new T.Matrix4().fromArray(threeSpriteWorldMatrix(T, h.sprite, h.material, camera));
    for (const position of [[-.5, -.5, 0], [.5, .5, 9], [.3, -.1, -4]]) {
      const actual = new T.Vector4(...position, 1).applyMatrix4(world).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix).toArray();
      near(actual, sourcePosition(h.sprite, h.material, camera, position), 1e-12);
    }
  }
});

test('sprite maps and alpha maps use the ordinary independent UV and blending path', async t => {
  const h = fixture(), map = new T.Texture(), alpha = new T.Texture(), textures = new Map();
  for (const texture of [map, alpha]) textures.set(texture, {view: {}, sampler: {}, version: texture.version, sourceVersion: texture.source.version});
  h.material.map = map; h.material.alphaMap = alpha; h.material.alphaTest = .2;
  map.repeat.set(2, 3); alpha.offset.set(.25, .5); alpha.rotation = .2;
  const bridge = await prepare(h, t, {textures, textureTransforms: true, alphaMaps: true}); bridge.render(h.camera, frame());
  const draw = draws(h).at(-1), entries = draw.groups.get(1).group.entries;
  assert.ok(entries.some(e => e.resource === textures.get(map).view)); assert.ok(entries.some(e => e.resource === textures.get(alpha).view));
  // Pinned MaterialNode.OPACITY multiplies the full alpha texture; the final
  // vec4-to-float conversion in NodeBuilder.format selects its red component.
  assert.match(draw.pipeline.fragment.module.code, /rgba\.a \*= alpha_texel\.r;/);
  assert.equal(packet(draw)[20], Math.fround(.2)); assert.equal(packet(draw)[19], .5);
  assert.ok(draw.pipeline.fragment.targets[0].blend);
  const pipelines = h.device.pipelines.length; alpha.offset.x = .75; bridge.render(h.camera, frame());
  assert.equal(h.device.pipelines.length, pipelines);
});

test('sprite visibility, layers, frustum and repeated transparent counts remain source ordered', async t => {
  const h = fixture(), second = new T.Sprite(new T.SpriteMaterial({color: 0x00ff00})); second.position.z = -1; h.scene.add(second);
  const bridge = await prepare(h, t); h.sprite.count = 3; bridge.render(h.camera, frame());
  assert.equal(draws(h).length, 4, 'count preserves repeated alpha blending');
  assert.equal(bridge.diagnostics.sourceDraws, 2);
  h.sprite.count = 0; second.layers.set(2); bridge.render(h.camera, frame()); assert.equal(h.device.snapshots.at(-1).length, 0);
  h.sprite.count = 1; h.sprite.position.set(1000, 0, 0); bridge.render(h.camera, frame()); assert.equal(h.device.snapshots.at(-1).length, 0);
  h.sprite.frustumCulled = false; bridge.render(h.camera, frame()); assert.equal(h.device.snapshots.at(-1).length, 1);
  h.sprite.count = 17; const submitted = h.device.submissions.length;
  assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_LIMIT'}); assert.equal(h.device.submissions.length, submitted);
});

test('ArrayCamera billboards use each visible child view and sprite frustum tests', async t => {
  const h = fixture(); h.sprite.center.set(.1, .9); h.material.rotation = .3; h.sprite.count = 2;
  const left = h.camera, right = new T.PerspectiveCamera(55, 1, .1, 100); right.position.set(-4, 1, 3); right.lookAt(0, 0, 0);
  left.updateMatrixWorld(); right.updateMatrixWorld();
  const camera = new T.ArrayCamera([left, right]); camera.position.z = 5;
  const bridge = await prepare(h, t); bridge.render(camera, frame());
  const recorded = draws(h); assert.equal(recorded.length, 4);
  assert.deepEqual(packet(recorded[0]).slice(0, 16), packet(recorded[1]).slice(0, 16), 'repeat all instances before advancing views');
  assert.notDeepEqual(packet(recorded[0]).slice(0, 16), packet(recorded[2]).slice(0, 16));
  assert.deepEqual(packet(recorded[2]).slice(0, 16), packet(recorded[3]).slice(0, 16));
  right.layers.set(3); bridge.render(camera, frame()); assert.equal(bridge.diagnostics.sourceDraws, 1);
});

test('custom sprite nodes remain explicit refusals and sprite castShadow does not add a caster', async t => {
  const h = fixture('SpriteNodeMaterial'); h.material.positionNode = vec3(1, 0, 0);
  await assert.rejects(createGpuThreeScene(h.device, h.scene, {three: T}), {code: 'THREE_SCENE_MATERIAL'});
  h.material.positionNode = null; h.sprite.castShadow = true;
  const bridge = await prepare(h, t); bridge.render(h.camera, frame()); assert.equal(draws(h).length, 1);
});
