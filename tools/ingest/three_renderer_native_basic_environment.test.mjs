/** Native non-PBR environments retain the pinned lighting/cube graph.
 * The device records resources and commands; browser cases execute the WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceRenderObject from '../../upstream/three.js/src/renderers/common/RenderObject.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
function cube(t, value = 64) {
  const faces = Array.from({length: 6}, (_, face) => {
    const texture = new T.DataTexture(new Uint8Array([value + face * 8, 64, 128, 255]), 1, 1);
    t.after(() => texture.dispose()); return texture;
  });
  const texture = new T.CubeTexture(faces); texture.needsUpdate = true;
  texture.minFilter = texture.magFilter = T.NearestFilter; texture.generateMipmaps = false;
  t.after(() => texture.dispose()); return texture;
}
function panorama(t, value = 32) {
  const bytes = new Uint8Array(8 * 4 * 4);
  for (let pixel = 0; pixel < 32; pixel++) bytes.set([value + pixel, 64, 192, 255], pixel * 4);
  const texture = new T.DataTexture(bytes, 8, 4);
  texture.mapping = T.EquirectangularReflectionMapping; texture.needsUpdate = true;
  texture.minFilter = texture.magFilter = T.NearestFilter;
  t.after(() => texture.dispose()); return texture;
}
function fixture(t, Material = T.MeshBasicNodeMaterial) {
  const device = geometryDevice(), textureWrites = [], textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data, layout, size) => textureWrites.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice(), layout, size});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: index => ({index})});
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configured.format, usage: configured.usage})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new Material({envMap: cube(t), reflectivity: .375});
  const object = new T.Mesh(new T.SphereGeometry(1, 8, 6), material), scene = new T.Scene(); scene.add(object);
  object.frustumCulled = false;
  const camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld(); scene.updateMatrixWorld();
  object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld); object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
  const programs = createThreeNodePrograms({three: T, renderer}), rawDraws = [];
  renderer._nodeProgramSupport = Object.create(programs, {captureDraw: {value(descriptor, context) {
    const packet = programs.captureDraw(descriptor, context); rawDraws.push({object: context.object, packet}); return packet;
  }}});
  const context = {scene, camera, lights: [], object, material, renderId: 1, frameId: 1};
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) { const result = capture.call(this, request); if (result) captures.push(result); return result; };
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); });
  return {renderer, device, textureWrites, textures, material, object, scene, camera, programs, context, captures, rawDraws,
    compile: () => programs.compile(material, object, context)};
}
function sourceBuilder(f) {
  const source = new T.WGSLNodeBuilder(f.object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.lightsNode = N.lights(f.context.lights);
  source.environmentNode = f.scene.environmentNode ?? (f.scene.environment ? N.texture(f.scene.environment) : null);
  source.build(); return source;
}
const meshPackets = f => f.captures.flatMap(frame => frame.draws).filter(draw => draw.object === f.object);
const rawPackets = f => f.rawDraws.filter(draw => draw.object === f.object).map(draw => draw.packet);

for (const Material of [T.MeshBasicMaterial, T.MeshLambertMaterial, T.MeshPhongMaterial,
  T.MeshBasicNodeMaterial, T.MeshLambertNodeMaterial, T.MeshPhongNodeMaterial])
for (const refract of [false, true])
test(`${Material.name} keeps exact source ${refract ? 'refraction' : 'reflection'} WGSL and material-local precedence`, async t => {
  const f = fixture(t, Material); await f.renderer.init();
  f.material.envMap.mapping = refract ? T.CubeRefractionMapping : T.CubeReflectionMapping;
  f.scene.environment = cube(t, 96); f.scene.environmentNode = N.texture(panorama(t));
  const expected = sourceBuilder(f), descriptor = f.compile(), packet = f.programs.captureDraw(descriptor, f.context);
  assert.equal(descriptor.vertex, expected.vertexShader); assert.equal(descriptor.fragment, expected.fragmentShader);
  assert.match(descriptor.fragment, refract ? /refract\(/ : /reflect\(/);
  assert.deepEqual(sampled(packet), [f.material.envMap]);
  assert.ok(values(packet).includes(.375)); assert.doesNotMatch(descriptor.fragment, /roughnessToMip|iblIrradiance/);
  f.material.envMap = null;
  const removed = f.compile(), inactive = f.programs.captureDraw(removed, {...f.context, renderId: 2});
  assert.notEqual(removed, descriptor); assert.equal(sampled(inactive).length, 0, 'non-PBR materials do not adopt the scene environment');
});

test('source combine operations and mapping changes select fresh guarded variants', async t => {
  const f = fixture(t, T.MeshPhongNodeMaterial); await f.renderer.init();
  const variants = [];
  for (const combine of [T.MultiplyOperation, T.MixOperation, T.AddOperation]) {
    f.material.dispose(); // The pinned renderer aliases nonzero combine modes in its cache.
    f.material.combine = combine;
    const expected = sourceBuilder(f), descriptor = f.compile(); variants.push(descriptor);
    assert.equal(descriptor.fragment, expected.fragmentShader);
  }
  assert.equal(new Set(variants.map(value => value.fragment)).size, 3);
  f.material.combine = T.MultiplyOperation;
  assert.throws(() => f.programs.captureDraw(variants[2], f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const reflected = f.compile(); f.material.envMap.mapping = T.CubeRefractionMapping;
  assert.throws(() => f.programs.captureDraw(reflected, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const refracted = f.compile(); assert.notEqual(refracted, reflected); assert.match(refracted.fragment, /refract\(/);
});

test('pinned Mix/Add cache collisions require an explicit source rebuild boundary', async t => {
  const f = fixture(t, T.MeshPhongMaterial); await f.renderer.init();
  const cacheView = {object: f.object, material: f.material, renderer: {backend: {isWebGPUBackend: true}},
    context: {id: 0}, clippingContextCacheKey: '', getGeometryCacheKey: () => 'same-geometry'};
  const sourceKey = () => SourceRenderObject.prototype.getMaterialCacheKey.call(cacheView);
  f.material.combine = T.MixOperation;
  const firstKey = sourceKey(), first = f.compile();
  f.material.combine = T.AddOperation; f.material.needsUpdate = true;
  assert.equal(sourceKey(), firstKey, 'the pinned renderer ignores this numeric-enum transition even with needsUpdate');
  assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_CACHE'});
  f.material.dispose();
  const rebuilt = f.compile(); assert.notEqual(rebuilt.fragment, first.fragment);
  assert.equal(rebuilt.fragment, sourceBuilder(f).fragmentShader);
});

test('pinned texture environment mapping changes require a fresh authored graph', async t => {
  const f = fixture(t, T.MeshPhongNodeMaterial); await f.renderer.init();
  const texture = f.material.envMap; f.material.envMap = null; f.material.envNode = N.cubeTexture(texture);
  const key = f.material.customProgramCacheKey(), reflected = f.compile();
  texture.mapping = T.CubeRefractionMapping; f.material.needsUpdate = true;
  assert.equal(f.material.customProgramCacheKey(), key, 'texture mappings do not change the pinned graph key');
  assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_CACHE'});
  f.material.envNode = N.cubeTexture(texture);
  assert.notEqual(f.material.customProgramCacheKey(), key);
  const refracted = f.compile(); assert.notEqual(refracted.fragment, reflected.fragment);
  assert.equal(refracted.fragment, sourceBuilder(f).fragmentShader);
});

test('reflectivity, specular strength, refraction and rotation stay live while accepted bytes are immutable', async t => {
  const f = fixture(t); await f.renderer.init();
  f.material.envMap.mapping = T.CubeRefractionMapping; f.material.refractionRatio = .625;
  f.material.specularMap = new T.DataTexture(new Uint8Array([96, 0, 0, 255]), 1, 1); f.material.specularMap.needsUpdate = true;
  t.after(() => f.material.specularMap.dispose());
  const descriptor = f.compile(), first = f.programs.captureDraw(descriptor, f.context), saved = first.uniforms.map(bytes => bytes.slice());
  f.material.reflectivity = .875; f.material.refractionRatio = .8125; f.material.envMapRotation.set(.2, .7, -.3);
  f.material.envMapIntensity = 99; // BasicLightingModel has no PBR intensity multiplier.
  assert.equal(f.compile(), descriptor);
  const second = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  for (const value of [.875, .8125]) assert.ok(values(second).includes(value));
  assert.ok(!values(second).includes(99)); assert.deepEqual(first.uniforms, saved);
  assert.ok(sampled(second).includes(f.material.specularMap));
  const rotation = new T.Matrix4().makeRotationFromEuler(f.material.envMapRotation).transpose().elements.map(Math.fround);
  const packed = values(second);
  assert.ok(packed.some((_, i) => rotation.every((value, j) => packed[i + j] === value)));
});

for (const reference of [false, true])
test(`source ${reference ? 'material-reference' : 'cube-texture'} envNode takes precedence and preserves callback context`, async t => {
  const f = fixture(t, T.MeshLambertNodeMaterial); await f.renderer.init();
  const chosen = cube(t, 80), ignored = f.material.envMap; let updates = 0;
  if (reference) { f.material.envMap = chosen; f.material.envNode = N.materialReference('envMap', 'cubeTexture'); }
  else f.material.envNode = N.cubeTexture(chosen);
  f.material.colorNode = N.uniform(new T.Color(.5, .25, .125)).onObjectUpdate(frame => {
    assert.equal(frame.material, f.material); assert.equal(frame.object, f.object); assert.equal(frame.scene, f.scene); updates++;
    return new T.Color(.5, .25, .125);
  });
  const expected = sourceBuilder(f), descriptor = f.compile(), packet = f.programs.captureDraw(descriptor, f.context);
  assert.equal(descriptor.fragment, expected.fragmentShader); assert.deepEqual(sampled(packet), [chosen]); assert.equal(updates, 1);
  if (!reference) assert.ok(!sampled(packet).includes(ignored));
  f.material.envNode = N.cubeTexture(cube(t, 112));
  assert.throws(() => f.programs.captureDraw(descriptor, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 1, 'graph replacement rejects before callbacks');
  assert.notEqual(f.compile(), descriptor);
});

for (const disabled of ['lighting', 'fragment'])
test(`${disabled} bypass leaves an otherwise unsupported material environment inactive`, async t => {
  const f = fixture(t); await f.renderer.init();
  f.material.envMap = new T.Texture(); t.after(() => f.material.envMap.dispose());
  if (disabled === 'lighting') f.renderer.lighting.enabled = false;
  else f.material.fragmentNode = N.vec4(.25, .5, .75, 1);
  const expected = sourceBuilder(f), descriptor = f.compile(), packet = f.programs.captureDraw(descriptor, f.context);
  assert.equal(descriptor.fragment, expected.fragmentShader); assert.equal(sampled(packet).length, 0);
});

test('mapping or combine mutations during source node updates reject the stale native draw', async t => {
  for (const change of [f => { f.material.combine = T.AddOperation; }, f => { f.material.envMap.mapping = T.CubeRefractionMapping; }]) {
    const f = fixture(t); await f.renderer.init();
    f.material.colorNode = N.uniform(new T.Color()).onObjectUpdate(() => { change(f); return new T.Color(1, 1, 1); });
    const descriptor = f.compile();
    assert.throws(() => f.programs.captureDraw(descriptor, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(f.device.submissions.length, 0);
  }
});

test('ordinary reflective meshes route natively and retain queued environments and controls', async t => {
  const f = fixture(t, T.MeshPhongMaterial); await f.renderer.init();
  f.scene.add(new T.AmbientLight(0xffffff, .5));
  const firstTexture = f.material.envMap;
  f.renderer.render(f.scene, f.camera);
  f.material.envMap = cube(t, 120); f.material.reflectivity = .75; f.material.refractionRatio = .8125;
  f.material.envMap.mapping = T.CubeRefractionMapping; f.renderer.render(f.scene, f.camera);
  f.material.reflectivity = 99; await drain(f.renderer);
  const packets = meshPackets(f); assert.equal(packets.length, 2);
  assert.deepEqual(sampled(rawPackets(f)[0]), [firstTexture]); assert.deepEqual(sampled(rawPackets(f)[1]), [f.material.envMap]);
  assert.ok(values({uniforms: packets[0].nodeUniforms}).includes(.375));
  assert.ok(values({uniforms: packets[1].nodeUniforms}).includes(.75));
  assert.ok(!values({uniforms: packets[1].nodeUniforms}).includes(99));
  assert.equal(f.renderer.info.f3d.presentedRenders, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('removing and restoring an ordinary environment retains source-native material shading', async t => {
  for (const Material of [T.MeshBasicMaterial, T.MeshLambertMaterial, T.MeshPhongMaterial]) {
    const f = fixture(t, Material); await f.renderer.init();
    const texture = f.material.envMap;
    f.renderer.render(f.scene, f.camera);
    f.material.envMap = null; f.material.needsUpdate = true;
    const expected = sourceBuilder(f);
    f.renderer.render(f.scene, f.camera);
    f.material.envMap = texture; f.material.needsUpdate = true;
    f.renderer.render(f.scene, f.camera); await drain(f.renderer);
    const packets = meshPackets(f), raw = rawPackets(f);
    assert.equal(packets.length, 3, `${Material.name} must not return to the fixed shader when its environment is removed`);
    assert.equal(packets[1].binding.nodeProgram.vertex, expected.vertexShader);
    assert.equal(packets[1].binding.nodeProgram.fragment, expected.fragmentShader);
    assert.deepEqual(raw.map(packet => sampled(packet).length), [1, 0, 1]);
    assert.equal(f.renderer.info.f3d.presentedRenders, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});

for (const authored of [false, true])
test(`${authored ? 'texture envNode' : 'ordinary Phong envMap'} converts a ready panorama through six owned cube faces`, async t => {
  const f = fixture(t, authored ? T.MeshPhongNodeMaterial : T.MeshPhongMaterial), input = panorama(t);
  if (authored) { f.material.envMap = null; f.material.envNode = N.texture(input); }
  else f.material.envMap = input;
  const filter = input.minFilter, generate = input.generateMipmaps;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const environment = sampled(rawPackets(f)[0])[0];
  assert.equal(environment.isCubeTexture, true); assert.equal(environment.isRenderTargetTexture, true);
  assert.equal(environment.mapping, T.CubeReflectionMapping); assert.notEqual(environment, input);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 6); assert.equal(f.renderer.info.f3d.presentedRenders, 1);
  assert.equal(input.minFilter, filter); assert.equal(input.generateMipmaps, generate);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 6, 'source cached conversion is reused');
  assert.equal(sampled(rawPackets(f)[1])[0], environment);
  input.mapping = T.EquirectangularRefractionMapping; f.material.needsUpdate = true;
  if (authored) f.material.envNode = N.texture(input);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(environment.mapping, T.CubeRefractionMapping); assert.equal(f.renderer.info.f3d.offscreenRenders, 6);
  assert.match(meshPackets(f)[2].binding.nodeProgram.fragment, /refract\(/); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
