/** Independent source PBR maps through production admission, bindings and draw
 * snapshots. The browser parity scenarios separately execute the generated WGSL.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuThreeScene} from './three_scene.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const frame = () => ({colorView: {}, depthView: {}});
const lastDraw = d => d.snapshots.at(-1)[0];
function packet(draw) {
  const binding = draw.groups.get(0), buffer = binding.group.entries[0].resource.buffer;
  return new Float32Array(draw.contents.get(buffer).buffer, binding.offsets[0] ?? 0);
}
const uv = (draw, slot) => packet(draw).slice(64 + slot * 8, 72 + slot * 8);
const matrix = texture => {
  const e = texture.matrix.elements;
  return Float32Array.from([e[0], e[3], e[6], 0, e[1], e[4], e[7], 0]);
};
function fixture(type = 'MeshStandardMaterial') {
  const device = geometryDevice(); device.limits.maxBindingsPerBindGroup = 32;
  const scene = new T.Scene(), geometry = new T.PlaneGeometry();
  for (const name of ['uv1', 'uv2', 'uv3']) geometry.setAttribute(name, geometry.attributes.uv.clone());
  const material = new T[type]({color: 0x789abc, metalness: .65, roughness: .75});
  const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; scene.add(mesh);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 3;
  const textures = new Map();
  const texture = channel => {
    const source = new T.Texture(); source.channel = channel;
    textures.set(source, {view: {}, sampler: {}, version: source.version, sourceVersion: source.source.version});
    return source;
  };
  return {device, scene, geometry, material, camera, textures, texture};
}
async function prepare(h, t, options = {}) {
  const bridge = await createGpuThreeScene(h.device, h.scene, {three: T, textures: h.textures,
    textureTransforms: true, alphaMaps: true, renderer: {maxDraws: 4}, ...options});
  t.after(() => bridge.dispose()); return bridge;
}

for (const type of ['MeshStandardMaterial', 'MeshStandardNodeMaterial', 'MeshPhysicalMaterial', 'MeshPhysicalNodeMaterial'])
test(`${type}: independent PBR textures, UV channels and live transforms`, async t => {
  const h = fixture(type), metal = h.texture(1), rough = h.texture(3);
  h.material.metalnessMap = metal; h.material.roughnessMap = rough;
  metal.repeat.set(2, 3); metal.offset.set(.125, .25);
  rough.repeat.set(4, 2); rough.rotation = .3;
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const first = lastDraw(h.device), entries = first.groups.get(1).group.entries;
  assert.deepEqual(entries.map(e => e.binding), [23, 24, 25, 26]);
  assert.equal(entries[1].resource, h.textures.get(metal).view);
  assert.equal(entries[3].resource, h.textures.get(rough).view);
  assert.deepEqual(uv(first, 11), matrix(metal)); assert.deepEqual(uv(first, 12), matrix(rough));
  const code = first.pipeline.fragment.module.code;
  assert.match(code, /let metallic = draw_info.options.w \* metallic_texel.b/);
  assert.match(code, /let source_roughness = draw_info.emission_roughness.w \* roughness_texel.g/);
  assert.match(code, /map_uv_0 = vec4<f32>\(vec2<f32>\(.*uv1/);
  assert.match(code, /map_uv_0 = vec4<f32>\(out.map_uv_0.xy, vec2<f32>\(.*uv3/);
  const pipelines = h.device.pipelines.length, buffers = h.device.buffers.length, original = uv(first, 12).slice();
  rough.offset.set(.6, .4); metal.rotation = -.7;
  h.material.roughness = .3; h.material.metalness = .9;
  bridge.render(h.camera, frame()); const second = lastDraw(h.device);
  assert.deepEqual(uv(second, 11), matrix(metal)); assert.deepEqual(uv(second, 12), matrix(rough));
  assert.equal(packet(second)[23], Math.fround(.9)); assert.equal(packet(second)[63], Math.fround(.3));
  assert.deepEqual(uv(first, 12), original, 'prior submitted use keeps its original transform');
  assert.equal(h.device.pipelines.length, pipelines); assert.equal(h.device.buffers.length, buffers);
  assert.equal(bridge.diagnostics.prepareVersion, 1);
});

for (const [source, slot, expected, absent] of [
  ['metalnessMap', 11, /options.w \* metallic_texel.b/, /source_roughness = draw_info.emission_roughness.w \*/],
  ['roughnessMap', 12, /emission_roughness.w \* roughness_texel.g/, /metallic = draw_info.options.w \*/],
]) test(`${source} alone leaves the other scalar factor untouched`, async t => {
  const h = fixture(); h.material[source] = h.texture(2);
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const draw = lastDraw(h.device), code = draw.pipeline.fragment.module.code;
  assert.deepEqual(draw.groups.get(1).group.entries.map(e => e.binding), [slot * 2 + 1, slot * 2 + 2]);
  assert.match(code, expected); assert.doesNotMatch(code, absent);
  assert.equal(packet(draw)[23], Math.fround(.65)); assert.equal(packet(draw)[63], Math.fround(.75));
});

test('independent maps transition to shared, replaced, removed and changed-channel bindings', async t => {
  const h = fixture(), first = h.texture(1), second = h.texture(2);
  h.material.metalnessMap = first; h.material.roughnessMap = second;
  const bridge = await prepare(h, t); bridge.render(h.camera, frame());
  for (const change of [
    () => { second.channel = 3; },
    () => { h.material.roughnessMap = first; },
    () => { h.material.roughnessMap = h.texture(2); },
    () => { h.material.metalnessMap = null; },
    () => { h.material.roughnessMap = null; },
  ]) {
    change(); const before = [h.device.writes.length, h.device.submissions.length];
    assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
    assert.deepEqual([h.device.writes.length, h.device.submissions.length], before);
    await bridge.prepare(); bridge.render(h.camera, frame());
    const entries = lastDraw(h.device).groups.get(1)?.group.entries ?? [];
    if (h.material.roughnessMap === first) assert.deepEqual(entries.map(e => e.binding), [2, 3]);
  }
  assert.equal(bridge.failed, false);
});

test('all physical maps coexist with coating, fog and bundled instances without binding or varying collisions', async t => {
  const h = fixture('MeshPhysicalMaterial'); h.material.clearcoat = .5;
  for (const name of ['map', 'normalMap', 'emissiveMap', 'aoMap', 'clearcoatMap', 'clearcoatRoughnessMap',
    'clearcoatNormalMap', 'alphaMap', 'specularIntensityMap', 'specularColorMap', 'metalnessMap', 'roughnessMap'])
    h.material[name] = h.texture(0);
  h.scene.fog = new T.Fog(0xabcdef, 1, 8);
  const bridge = await prepare(h, t, {fog: {}, renderer: {maxDraws: 4, instancing: true, renderBundles: true}});
  bridge.render(h.camera, frame()); const draw = lastDraw(h.device), entries = draw.groups.get(1).group.entries;
  assert.equal(entries.length, 25); assert.equal(new Set(entries.map(e => e.binding)).size, entries.length);
  assert.equal(entries.find(e => e.binding === 16).resource.size, 16, 'coating keeps its own uniform binding');
  const output = /struct VertexOutput \{([^}]+)\}/.exec(draw.pipeline.vertex.module.code)[1];
  const locations = [...output.matchAll(/@location\((\d+)\)/g)].map(m => Number(m[1]));
  assert.equal(new Set(locations).size, locations.length); assert.ok(Math.max(...locations) < 16);
  bridge.render(h.camera, frame()); await bridge.whenIdle();
});

test('immutable/deformed secondary UVs use compact attribute locations for the new map slots', async t => {
  const d = geometryDevice(); d.limits.maxBindingsPerBindGroup = 32;
  const renderer = await createGpuAnimationRenderer(d, {threeLights: true, textureTransforms: true, maxDraws: 2});
  t.after(() => renderer.dispose());
  const geometry = {vertexCount: 3, vertexBuffer: d.createBuffer({size: 120, usage: 32}), worldMatrix: I,
    whenIdle: async () => {}, vertexLayout: {arrayStride: 40, stepMode: 'vertex', attributes: [
      {shaderLocation: 0, offset: 0, format: 'float32x3'}, {shaderLocation: 1, offset: 12, format: 'float32x3'}]}};
  const mesh = await renderer.addMesh(geometry, {shading: 'metallic-roughness',
    metallicTexture: {view: {}, sampler: {}}, roughnessTexture: {view: {}, sampler: {}},
    mapCoordinates: {metallicTexture: {texCoords: [0, 0, 1, 0, 0, 1]}, roughnessTexture: {texCoords: [1, 1, 0, 1, 1, 0]}}});
  renderer.render({...frame(), viewProjection: I, draws: [mesh], lighting: {viewDirection: [0, 0, 1], lights: []}});
  const draw = lastDraw(d), code = draw.pipeline.vertex.module.code;
  assert.match(code, /@location\(5\) uv_11: vec2<f32>/); assert.match(code, /@location\(6\) uv_12: vec2<f32>/);
  assert.deepEqual(draw.pipeline.vertex.buffers[1].attributes.map(a => a.shaderLocation), [3, 4, 5, 6]);
  for (const options of [
    {shading: 'unlit', metallicTexture: {view: {}, sampler: {}}},
    {shading: 'metallic-roughness', metallicRoughnessTexture: {view: {}, sampler: {}}, roughnessTexture: {view: {}, sampler: {}}},
  ]) await assert.rejects(renderer.addMesh(geometry, options), {code: 'ANIMATION_RENDER_OPTIONS'});
});
