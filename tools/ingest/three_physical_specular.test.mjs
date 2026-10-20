/** Production material admission and submitted bytes. Browser differential
 * scenarios, not this recording device, validate the physical lighting pixels.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {environmentLightingWgsl} from './animation_environment_receiver.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
const f32 = values => Float32Array.from(values);
const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function fixture(type = 'MeshPhysicalMaterial') {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 4096, maxBindingsPerBindGroup: 32});
  const scene = new T.Scene(), geometry = new T.PlaneGeometry(1, 1);
  for (const name of ['uv1', 'uv2', 'uv3']) geometry.setAttribute(name, geometry.attributes.uv.clone());
  const material = new T[type]({ior: 2.1, specularIntensity: .3, specularColor: new T.Color(.2, 1.2, .7), roughness: .4});
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
    textureTransforms: true, alphaMaps: true, renderer: {maxDraws: 8}, ...options});
  t.after(() => bridge.dispose()); return bridge;
}
const frame = () => ({colorView: {}, depthView: {}});
const draws = h => h.device.snapshots.flat().filter(d => d.streams.has(0));
function packet(draw, byteOffset = 0) {
  const group = draw.groups.get(0), buffer = group.group.entries[0].resource.buffer;
  return new Float32Array(draw.contents.get(buffer).buffer, (group.offsets[0] ?? 0) + byteOffset);
}
const factors = draw => packet(draw).slice(64 + 11 * 8, 72 + 11 * 8);
const factorValues = (color, ior, intensity) => f32([...color, 0, ior, intensity, 1, 0]);
const uv = (draw, slot) => packet(draw).slice(64 + slot * 8, 72 + slot * 8);
const matrix = texture => {
  const e = texture.matrix.elements; return f32([e[0], e[3], e[6], 0, e[1], e[4], e[7], 0]);
};

for (const type of ['MeshPhysicalMaterial', 'MeshPhysicalNodeMaterial'])
test(`${type} keeps IOR and colored specular live without material recompilation`, async t => {
  const h = fixture(type), bridge = await prepare(h, t); bridge.render(h.camera, frame());
  const first = draws(h).at(-1), buffers = h.device.buffers.length, pipelines = h.device.pipelines.length;
  assert.deepEqual(factors(first), factorValues([.2, 1.2, .7], 2.1, .3));
  h.material.ior = 1; h.material.specularIntensity = 0; h.material.specularColor.setRGB(2, .5, .25);
  bridge.render(h.camera, frame());
  assert.deepEqual(factors(draws(h).at(-1)), factorValues([2, .5, .25], 1, 0));
  assert.deepEqual(factors(first), factorValues([.2, 1.2, .7], 2.1, .3), 'previous submitted factors retain their values');
  assert.equal(h.device.buffers.length, buffers); assert.equal(h.device.pipelines.length, pipelines);
  assert.equal(bridge.diagnostics.prepareVersion, 1); await bridge.whenIdle();
});

test('specular maps have independent texture bindings, channels and live transforms', async t => {
  const h = fixture(), intensity = h.texture(2), color = h.texture(3);
  h.material.specularIntensityMap = intensity; h.material.specularColorMap = color;
  intensity.repeat.set(2, 3); intensity.offset.set(.125, .25);
  color.repeat.set(4, 2); color.rotation = .3; color.colorSpace = T.SRGBColorSpace;
  const bridge = await prepare(h, t); bridge.render(h.camera, frame()); const first = draws(h).at(-1);
  const entries = first.groups.get(1).group.entries;
  assert.deepEqual(entries.map(e => e.binding), [19, 20, 21, 22]);
  assert.equal(entries[1].resource, h.textures.get(intensity).view);
  assert.equal(entries[3].resource, h.textures.get(color).view);
  assert.deepEqual(uv(first, 9), matrix(intensity)); assert.deepEqual(uv(first, 10), matrix(color));
  const source = first.pipeline.fragment.module.code;
  assert.match(source, /physical_specular_params.y \* specular_intensity_texel.a/);
  assert.match(source, /physical_specular_color.rgb \* specular_color_texel.rgb/);
  assert.match(source, /map_uv_0 = vec4<f32>\(vec2<f32>\(.*uv2/);
  assert.match(source, /map_uv_0 = vec4<f32>\(out.map_uv_0.xy, vec2<f32>\(.*uv3/);
  const pipelines = h.device.pipelines.length;
  intensity.offset.x = .7; color.rotation = -.8; bridge.render(h.camera, frame());
  assert.deepEqual(uv(draws(h).at(-1), 9), matrix(intensity));
  assert.deepEqual(uv(draws(h).at(-1), 10), matrix(color)); assert.equal(h.device.pipelines.length, pipelines);
  color.channel = 1; assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(h.camera, frame());
  assert.match(draws(h).at(-1).pipeline.vertex.module.code, /map_uv_0.xy, vec2<f32>\(.*uv1/);
});

test('all eleven maps coexist with clearcoat, fog and batch draw indices', async t => {
  const h = fixture(); h.material.clearcoat = .6;
  h.material.metalnessMap = h.material.roughnessMap = h.texture();
  for (const name of ['map', 'normalMap', 'emissiveMap', 'aoMap', 'clearcoatMap',
    'clearcoatRoughnessMap', 'clearcoatNormalMap', 'alphaMap', 'specularIntensityMap', 'specularColorMap'])
    h.material[name] = h.texture();
  h.scene.fog = new T.Fog(0xabcdef, 1, 9);
  const bridge = await prepare(h, t, {fog: {}, renderer: {maxDraws: 8, instancing: true, renderBundles: true}});
  bridge.render(h.camera, frame()); const draw = draws(h).at(-1), entries = draw.groups.get(1).group.entries;
  assert.equal(entries.length, 23); assert.equal(new Set(entries.map(e => e.binding)).size, 23);
  const output = /struct VertexOutput \{([^}]+)\}/.exec(draw.pipeline.vertex.module.code)[1];
  const locations = [...output.matchAll(/@location\((\d+)\)/g)].map(m => Number(m[1]));
  assert.equal(new Set(locations).size, locations.length, 'UVs must not collide with fog or the draw index');
  assert.ok(Math.max(...locations) < 16, 'full physical map set fits ordinary WebGPU interstage limits');
  assert.deepEqual(factors(draw), factorValues([.2, 1.2, .7], 2.1, .3));
  assert.match(draw.pipeline.fragment.module.code, /vec3<f32>\(0.04\), 1.0, true, normal/,
    'the coating remains its independent fixed-1.5-IOR lobe');
  await bridge.whenIdle();
});

test('invalid live IOR/specular parameters reject before any queue writes', async t => {
  const h = fixture(), bridge = await prepare(h, t); bridge.render(h.camera, frame());
  for (const [key, value] of [['ior', 0], ['ior', -1], ['ior', NaN], ['specularIntensity', -1], ['specularIntensity', Infinity]]) {
    const before = [h.device.writes.length, h.device.submissions.length], previous = h.material[key];
    h.material[key] = value; assert.throws(() => bridge.render(h.camera, frame()), /THREE_SCENE_VALUE/);
    assert.deepEqual([h.device.writes.length, h.device.submissions.length], before); h.material[key] = previous;
  }
  const before = h.device.writes.length; h.material.specularColor.r = -1;
  assert.throws(() => bridge.render(h.camera, frame()), {code: 'THREE_SCENE_VALUE'});
  assert.equal(h.device.writes.length, before); h.material.specularColor.r = .2;
  bridge.render(h.camera, frame()); assert.equal(bridge.failed, false);
});

function geometry(d) {
  return {vertexCount: 3, vertexBuffer: d.createBuffer({size: 120, usage: 32}), worldMatrix: I,
    whenIdle: async () => {}, vertexLayout: {arrayStride: 40, stepMode: 'vertex', attributes: [
      {shaderLocation: 0, offset: 0, format: 'float32x3'}, {shaderLocation: 1, offset: 12, format: 'float32x3'}]}};
}
const lowFrame = draws => ({...frame(), viewProjection: I,
  lighting: {viewDirection: [0, 0, 1], lights: [{type: 'directional', direction: [0, 0, -1]}]}, draws});
for (const instancing of [false, true])
test(`shared material snapshots physical values for each logical draw and reused bundle (instancing=${instancing})`, async t => {
  const d = geometryDevice(), renderer = await createGpuAnimationRenderer(d, {threeLights: true, instancing, renderBundles: true, maxDraws: 2});
  t.after(() => renderer.dispose()); const mesh = await renderer.addMesh(geometry(d), {shading: 'metallic-roughness'});
  const first = {mesh, ior: 1.2, specularIntensity: .2, specularColor: [1, 0, 0]}, second = {mesh, ior: 2.33, specularIntensity: .8, specularColor: [0, .4, 1]};
  renderer.render(lowFrame([first, second])); first.ior = 1.8; first.specularColor.fill(.5);
  renderer.render(lowFrame([first, second]));
  const a = d.snapshots[0], b = d.snapshots[1];
  assert.deepEqual(packet(a[0]).slice(64, 72), factorValues([1, 0, 0], 1.2, .2));
  assert.deepEqual(packet(a[instancing ? 0 : 1], instancing ? 512 : 0).slice(64, 72), factorValues([0, .4, 1], 2.33, .8));
  assert.deepEqual(packet(b[0]).slice(64, 72), factorValues([.5, .5, .5], 1.8, .2));
  assert.equal(renderer.bundleDiagnostics.builds, 1); assert.equal(renderer.bundleDiagnostics.reuses, 1);
  assert.equal(renderer.drawCallCount, instancing ? 1 : 2); await renderer.whenIdle();
});

test('generic PBR retains its material boundary and original environment receiver', async t => {
  const d = geometryDevice(), renderer = await createGpuAnimationRenderer(d);
  t.after(() => renderer.dispose()); const g = geometry(d);
  for (const option of [{ior: 1.8}, {specularIntensity: .4}, {specularColor: [1, 0, 0]},
    {specularIntensityTexture: {view: {}, sampler: {}}}, {specularColorTexture: {view: {}, sampler: {}}}])
    await assert.rejects(renderer.addMesh(g, {shading: 'metallic-roughness', ...option}), {code: 'ANIMATION_RENDER_OPTIONS'});
  const source = environmentLightingWgsl(1);
  assert.match(source, /mix\(vec3<f32>\(0.04\), base, metallic\)/);
  assert.match(source, /f0 \* dfg.x \+ vec3<f32>\(dfg.y\)/);
  assert.doesNotMatch(source, /dielectric_f0|specular_f90/);
  const native = environmentLightingWgsl(1, true);
  assert.match(native, /mix\(dielectric_f0, base, metallic\)/);
  assert.match(native, /specular_f90 \* dfg.y/);
});

test('registration snapshots physical values before asynchronous pipeline preparation', async t => {
  const d = geometryDevice(), renderer = await createGpuAnimationRenderer(d, {threeLights: true});
  t.after(() => renderer.dispose()); const options = {shading: 'metallic-roughness', ior: 1.4,
    specularIntensity: .6, specularColor: [.2, .3, .4]};
  const pending = renderer.addMesh(geometry(d), options);
  options.ior = 2; options.specularColor.fill(1); options.specularIntensity = 0;
  const mesh = await pending; renderer.render(lowFrame([mesh]));
  assert.deepEqual(packet(d.snapshots.at(-1)[0]).slice(64, 72), factorValues([.2, .3, .4], 1.4, .6));
});
