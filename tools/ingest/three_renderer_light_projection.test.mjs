/** Source spotlight projections and authored light colors through native draws.
 * The recording GPU checks WGSL, immutable bindings and source update timing;
 * these checks do not execute shaders or claim browser pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, {Light = T.SpotLight, Material = T.MeshLambertMaterial, shadows = false, features = []} = {}) {
  const device = geometryDevice(), textures = [], writes = [], captures = [];
  device.features = new Set(features);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const texture = {dimension: '2d', sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = desc => ({...desc});
  device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice(), layout, size});
  let configuration;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configuration.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  renderer.shadowMap.enabled = shadows; renderer.shadowMap.type = T.PCFShadowMap;
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false; object.receiveShadow = object.castShadow = shadows;
  const light = new Light(0xffe6cc, 2); light.position.set(-2, 3, 4);
  if (light.shadow) { light.castShadow = shadows; light.shadow.mapSize.set(32, 16); }
  const scene = new T.Scene(); scene.add(object, light); if (light.target) scene.add(light.target);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 5;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const frame = capture.call(this, request); if (frame) captures.push(frame); return frame;
  };
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); light.dispose(); });
  return {renderer, device, textures, writes, captures, material, object, light, scene, camera};
}
function map(t, value = 17, compressed = false) {
  const texture = compressed
    ? new T.CompressedTexture([{width: 4, height: 4, data: new Uint8Array(8).fill(value)}], 4, 4, T.RGBA_S3TC_DXT1_Format)
    : new T.DataTexture(new Uint8Array(16).fill(value), 2, 2);
  texture.minFilter = texture.magFilter = T.LinearFilter; texture.colorSpace = T.SRGBColorSpace;
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const colorView = draw => entries(draw).find(entry => entry.resource?.texture && !entry.resource.texture.format.startsWith('depth')).resource;
const uploaded = (f, texture) => f.writes.filter(write => write.destination.texture === texture);
function uniformValues(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => {
    const bindings = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding);
    return bindings.flatMap((binding, index) => {
      const resource = group.entries.find(entry => entry.binding === binding.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    });
  });
}
const contains = (haystack, needle) => haystack.some((_, index) => needle.every((value, offset) => haystack[index + offset] === value));
function assertSource(f, descriptor) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights([f.light]); builder.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(builder.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(builder.fragmentShader));
}
const programs = f => f.captures.flatMap(frame => frame.draws ?? []).filter(draw => draw.object === f.object || draw.binding?.object === f.object)
  .map(draw => draw.binding.nodeProgram);

for (const compressed of [false, true]) test(`ordinary spotlight material projects ${compressed ? 'compressed' : 'RGBA'} maps through source WGSL`, async t => {
  const f = fixture(t, {features: compressed ? ['texture-compression-bc'] : []}); f.light.map = map(t, 19, compressed);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0], descriptor = programs(f)[0]; assert.ok(draw); assert.ok(descriptor);
  assertSource(f, descriptor); assert.match(descriptor.fragment, /textureSample/);
  assert.equal(colorView(draw).format, compressed ? 'bc1-rgba-unorm-srgb' : 'rgba8unorm-srgb');
  assert.ok(uploaded(f, colorView(draw).texture)[0].bytes.every(value => value === 19));
  assert.equal(f.material.isNodeMaterial, undefined, 'ordinary materials select the native light path without an authored slot');
  assert.equal(f.light.shadow.camera.coordinateSystem, T.WebGPUCoordinateSystem);
  assert.ok(contains(uniformValues(draw), [...new Float32Array(f.light.shadow.matrix.elements)]));
});

test('queued projections capture source matrix changes, stale upload versions and replacement maps', async t => {
  const f = fixture(t), texture = map(t), replacement = map(t, 91); f.light.map = texture;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const firstMatrix = new Float32Array(f.light.shadow.matrix.elements);
  texture.image.data.fill(53); f.light.position.set(2, 4, 6); f.light.target.position.set(.5, -.25, 0); f.light.angle = .4;
  f.renderer.render(f.scene, f.camera); const secondMatrix = new Float32Array(f.light.shadow.matrix.elements);
  f.light.map = replacement; f.renderer.render(f.scene, f.camera);
  texture.image.data.fill(99); replacement.image.data.fill(127); f.light.shadow.matrix.elements.fill(999);
  await f.renderer.waitForGPU();
  const rendered = draws(f), views = rendered.map(colorView), descriptors = programs(f);
  assert.equal(rendered.length, 3); assert.notDeepEqual(firstMatrix, secondMatrix);
  assert.ok(contains(uniformValues(rendered[0]), [...firstMatrix]));
  assert.ok(contains(uniformValues(rendered[1]), [...secondMatrix]));
  assert.ok(contains(uniformValues(rendered[2]), [...secondMatrix]));
  assert.equal(views[0].texture, views[1].texture); assert.notEqual(views[1].texture, views[2].texture);
  assert.ok(uploaded(f, views[0].texture)[0].bytes.every(value => value === 17));
  assert.ok(uploaded(f, views[2].texture)[0].bytes.every(value => value === 91));
  assert.equal(descriptors[0], descriptors[1]); assert.notEqual(descriptors[1], descriptors[2]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('spot projections and source PCF shadows share the native receiver and ordered depth owner', async t => {
  const f = fixture(t, {shadows: true, Material: T.MeshStandardMaterial}); f.light.map = map(t);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const receiver = draws(f).find(draw => entries(draw).some(entry => entry.resource?.compare)); assert.ok(receiver);
  assert.ok(entries(receiver).some(entry => entry.resource?.format === 'rgba8unorm-srgb'));
  assert.ok(entries(receiver).some(entry => entry.resource?.texture?.format.startsWith('depth')));
  assert.ok(f.device.passes.some(pass => pass.desc.depthStencilAttachment?.view?.texture.width === 32 && pass.draws.length));
  assert.ok(contains(uniformValues(receiver), [...new Float32Array(f.light.shadow.matrix.elements)]));
});

test('spotlight TSL Fn colors preserve projection inputs, source cache keys and queued uniforms', async t => {
  const f = fixture(t), texture = map(t), updates = [], gain = N.uniform(.21875).setName('projectedLightGain');
  gain.onRenderUpdate(frame => { assert.equal(frame.renderer, f.renderer); assert.equal(frame.light, undefined);
    assert.equal(frame.scene, f.scene); updates.push(gain.value); return gain.value; });
  f.light.map = {ignored: true};
  f.light.colorNode = N.Fn(([uv]) => { assert.equal(uv.isNode, true); return N.texture(texture, uv.xy).rgb.mul(gain); });
  assert.doesNotThrow(() => N.lights([f.light]).getCacheKey());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  gain.value = .8125; f.renderer.render(f.scene, f.camera); gain.value = .96875; texture.image.data.fill(255);
  await f.renderer.waitForGPU();
  const rendered = draws(f); assert.equal(rendered.length, 2); assert.deepEqual(updates, [.21875, .8125]);
  assert.ok(uniformValues(rendered[0]).includes(.21875)); assert.ok(uniformValues(rendered[1]).includes(.8125));
  assert.equal(programs(f)[0], programs(f)[1]);
  assert.ok(uploaded(f, colorView(rendered[0]).texture)[0].bytes.every(value => value === 17));
  f.light.colorNode = N.Fn(([uv]) => uv.x.mul(.375).add(N.vec3(.125, .25, .5)));
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.notEqual(programs(f)[1], programs(f)[2]);
  assert.equal(entries(draws(f)[2]).filter(entry => entry.resource?.texture).length, 0);
});

for (const Light of [T.DirectionalLight, T.PointLight, T.AmbientLight])
test(`${Light.name} authored color graphs retain shared render updates and immutable draw uniforms`, async t => {
  const f = fixture(t, {Light}), updates = [], tint = N.uniform(new T.Color(.125, .375, .625)).setName('authoredLightTint');
  tint.onRenderUpdate(frame => { assert.equal(frame.renderer, f.renderer); updates.push(tint.value.clone()); return tint.value; });
  f.light.colorNode = tint;
  const second = new T.Mesh(f.object.geometry, f.material); second.frustumCulled = false; second.position.x = 1; f.scene.add(second);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  tint.value.setRGB(.875, .625, .375); f.renderer.render(f.scene, f.camera); tint.value.setRGB(0, 0, 0);
  await f.renderer.waitForGPU();
  const rendered = draws(f); assert.equal(rendered.length, 4); assert.equal(updates.length, 2);
  assert.ok(rendered.slice(0, 2).every(draw => contains(uniformValues(draw), [.125, .375, .625])));
  assert.ok(rendered.slice(2).every(draw => contains(uniformValues(draw), [.875, .625, .375])));
  assertSource(f, programs(f)[0]);
});

for (const kind of ['array projection', 'missing compressed feature', 'spot graph instead of callable', 'spot callable without source hash', 'nonspot callable', 'invalid generated texture'])
test(`native light admission refuses ${kind} before GPU allocation`, async t => {
  const f = fixture(t, {Light: kind === 'nonspot callable' ? T.PointLight : T.SpotLight});
  if (kind === 'array projection') {
    f.light.map = new T.DataArrayTexture(new Uint8Array(32), 2, 2, 2); f.light.map.needsUpdate = true;
    t.after(() => f.light.map.dispose());
  } else if (kind === 'missing compressed feature') f.light.map = map(t, 17, true);
  else if (kind === 'spot graph instead of callable') f.light.colorNode = N.vec3(1);
  else if (kind === 'spot callable without source hash') {
    f.light.colorNode = () => N.vec3(1);
    assert.throws(() => N.lights([f.light]).getCacheKey(), TypeError);
  }
  else if (kind === 'nonspot callable') f.light.colorNode = () => N.vec3(1);
  else {
    const texture = map(t); texture.magFilter = T.LinearMipmapLinearFilter;
    f.light.colorNode = N.Fn(([uv]) => N.texture(texture, uv.xy).rgb);
  }
  await f.renderer.init(); const allocations = f.textures.length;
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), error => {
    assert.equal(error.code, kind === 'missing compressed feature' ? 'THREE_TEXTURE_FEATURE'
      : kind === 'invalid generated texture' ? 'THREE_NODE_PROGRAM_BUILD' : 'THREE_NODE_PROGRAM_LIGHTS', error.cause?.message ?? error.message);
    if (kind === 'invalid generated texture') assert.equal(error.cause?.code, 'THREE_TEXTURE_SAMPLER');
    return true;
  });
  assert.equal(f.textures.length, allocations); assert.equal(f.writes.length, 0); assert.equal(draws(f).length, 0);
});

test('inactive projected lights leave ordinary materials on their fixed path without reading invalid maps', async t => {
  for (const inactive of ['hidden', 'layer', 'disabled']) {
    const f = fixture(t); f.light.map = {invalid: true};
    if (inactive === 'hidden') f.light.visible = false;
    else if (inactive === 'layer') f.light.layers.set(1);
    else f.renderer.lighting.enabled = false;
    await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
    assert.equal(draws(f).length, 0); assert.equal(f.writes.length, 0);
  }
});
