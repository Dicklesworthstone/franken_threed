/** IES/projector source lighting through owned queued draws and shadow passes.
 * Recording-GPU evidence covers shader inputs and command order, not pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, Light, {shadows = false, shadowType = T.PCFShadowMap} = {}) {
  const device = geometryDevice(), textures = [], writes = [], captures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const texture = {dimension: '2d', sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = desc => ({...desc});
  device.queue.writeTexture = (destination, data) => writes.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let configuration;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configuration.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  renderer.shadowMap.enabled = shadows; renderer.shadowMap.type = shadowType;
  const material = Light === T.IESSpotLight ? new T.MeshPhongMaterial() : new T.MeshLambertMaterial();
  const object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false; object.receiveShadow = object.castShadow = shadows;
  const light = new Light(0xffffff, 2); light.position.set(-2, 3, 4);
  light.castShadow = shadows; light.shadow.mapSize.set(32, 16);
  const scene = new T.Scene(); scene.add(object, light, light.target);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 5;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const frame = capture.call(this, request); if (frame) captures.push(frame); return frame;
  };
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); light.dispose(); });
  return {renderer, device, textures, writes, captures, material, object, light, scene, camera};
}
function map(t, value = .25) {
  // Match IESLoader's default single-channel half-float table and filtering.
  const texture = new T.DataTexture(new Uint16Array(8).fill(T.DataUtils.toHalfFloat(value)), 4, 2, T.RedFormat, T.HalfFloatType);
  texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const mapView = draw => entries(draw).find(entry => entry.resource?.texture?.format === 'r16float')?.resource;
const programs = f => f.captures.flatMap(frame => frame.draws ?? []).filter(draw => draw.binding.material === f.material)
  .map(draw => draw.binding.nodeProgram);
function uniforms(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
    .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .flatMap((entry, index) => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    }));
}
const contains = (values, expected) => values.some((_, index) => expected.every((value, offset) => values[index + offset] === value));

for (const Light of [T.IESSpotLight, T.ProjectorLight])
test(`${Light.name} selects native source attenuation before any image or authored color is assigned`, async t => {
  const f = fixture(t, Light);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(draws(f).length, 1);
  const descriptor = programs(f)[0], source = new T.WGSLNodeBuilder(f.object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.lightsNode = N.lights([f.light]); source.build();
  assert.equal(descriptor.vertex, source.vertexShader); assert.equal(descriptor.fragment, source.fragmentShader);
  if (Light === T.IESSpotLight) {
    f.light.iesMap = map(t); f.renderer._advanceAnimationFrame(); await f.renderer.renderAsync(f.scene, f.camera);
    assert.equal(programs(f)[1], descriptor, 'adding a table retains the source first-render cone until invalidation');
    assert.equal(mapView(draws(f)[1]), undefined);
    f.material.needsUpdate = true; await f.renderer.renderAsync(f.scene, f.camera);
    assert.notEqual(programs(f)[2], descriptor); assert.ok(mapView(draws(f)[2]));
  }
});

for (const Light of [T.IESSpotLight, T.ProjectorLight])
test(`${Light.name} ordinary materials capture queued maps, source uniforms and upload history`, async t => {
  const f = fixture(t, Light), texture = map(t), replacement = map(t, .75);
  const slot = Light === T.IESSpotLight ? 'iesMap' : 'map'; f.light[slot] = texture;
  await f.renderer.init(); f.renderer._advanceAnimationFrame(); f.renderer.render(f.scene, f.camera);
  const firstMatrix = new Float32Array(f.light.shadow.matrix.elements);
  texture.image.data.fill(T.DataUtils.toHalfFloat(.875));
  f.light.color.setRGB(.125, .375, .625); f.light.intensity = 1;
  f.light.position.set(2, 4, 6); f.light.aspect = .75;
  f.renderer._advanceAnimationFrame(); f.renderer.render(f.scene, f.camera);
  const secondMatrix = new Float32Array(f.light.shadow.matrix.elements);
  f.light[slot] = replacement;
  f.renderer._advanceAnimationFrame(); f.renderer.render(f.scene, f.camera);
  replacement.image.data.fill(0); f.light.color.setRGB(0, 0, 0);
  await f.renderer.waitForGPU();
  const rendered = draws(f), descriptors = programs(f), views = rendered.map(mapView);
  assert.equal(rendered.length, 3); assert.ok(views.every(Boolean));
  assert.equal(f.material.isNodeMaterial, undefined, 'official ordinary materials need no authored node slot');
  assert.equal(views[0].texture, views[1].texture); assert.notEqual(views[1].texture, views[2].texture);
  for (const [view, expected] of [[views[0], .25], [views[2], .75]]) {
    const writes = f.writes.filter(write => write.destination.texture === view.texture);
    assert.equal(writes.length, 1, 'versionless CPU edits remain GPU-stale');
    assert.ok([...new Uint16Array(writes[0].bytes.buffer)].every(value => value === T.DataUtils.toHalfFloat(expected)));
  }
  assert.ok(contains(uniforms(rendered[0]), [2, 2, 2]));
  assert.ok(rendered.slice(1).every(draw => contains(uniforms(draw), [.125, .375, .625])));
  assert.equal(descriptors[0], descriptors[1]);
  if (Light === T.IESSpotLight) assert.equal(descriptors[1], descriptors[2]);
  else {
    assert.notEqual(descriptors[1], descriptors[2], 'source projected-map identities invalidate the lighting graph');
    assert.notDeepEqual(firstMatrix, secondMatrix);
    assert.ok(contains(uniforms(rendered[0]), [...firstMatrix]));
    assert.ok(contains(uniforms(rendered[1]), [...secondMatrix]));
  }
  assert.equal(f.renderer.info.f3d.route, 'general-webgpu'); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const Light of [T.IESSpotLight, T.ProjectorLight])
for (const shadowType of [T.BasicShadowMap, T.PCFShadowMap, T.VSMShadowMap])
test(`${Light.name} source shadow type ${shadowType} retains caster callbacks, update controls and shared inputs`, async t => {
  const f = fixture(t, Light, {shadows: true, shadowType}), events = [];
  f.light[Light === T.IESSpotLight ? 'iesMap' : 'map'] = map(t);
  f.object.onBeforeShadow = (_renderer, _object, _camera, shadowCamera) => events.push(['before', shadowCamera]);
  f.object.onAfterShadow = (_renderer, _object, _camera, shadowCamera) => events.push(['after', shadowCamera]);
  await f.renderer.init(); f.renderer._advanceAnimationFrame(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.deepEqual(events, [['before', f.light.shadow.camera], ['after', f.light.shadow.camera]]);
  const receiver = draws(f).find(mapView); assert.ok(receiver);
  assert.ok(entries(receiver).filter(entry => entry.resource?.texture).length >= 2, 'attenuation and shadow images share the receiver');
  const shadowPass = f.device.passes.find(pass => pass.desc.depthStencilAttachment?.view?.texture.width === 32 &&
    pass.desc.depthStencilAttachment.view.texture.height === 16 && pass.draws.length);
  const receiverPass = f.device.passes.find(pass => pass.draws.some(draw => draw.pipeline === receiver.pipeline));
  assert.ok(shadowPass); assert.ok(f.device.passes.indexOf(shadowPass) < f.device.passes.indexOf(receiverPass));
  f.light.shadow.autoUpdate = false;
  f.renderer._advanceAnimationFrame(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(events.length, 2, 'disabled automatic shadow updates do not replay source caster hooks');
  f.light.shadow.needsUpdate = true;
  f.renderer._advanceAnimationFrame(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(events.length, 4); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('ordinary projector Fn colors keep per-render callbacks and queued uniforms', async t => {
  const f = fixture(t, T.ProjectorLight), updates = [], gain = N.uniform(.21875).setName('projectorFamilyGain');
  gain.onRenderUpdate(frame => { assert.equal(frame.renderer, f.renderer); updates.push(gain.value); return gain.value; });
  f.light.map = {ignored: true};
  f.light.colorNode = N.Fn(([uv]) => N.vec3(uv.xy, .5).mul(gain));
  f.light.aspect = 1.25;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  gain.value = .8125; f.renderer.render(f.scene, f.camera); gain.value = .96875;
  await f.renderer.waitForGPU();
  assert.deepEqual(updates, [.21875, .8125]);
  assert.ok(uniforms(draws(f)[0]).includes(.21875)); assert.ok(uniforms(draws(f)[1]).includes(.8125));
  assert.equal(programs(f)[0], programs(f)[1]); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const Light of [T.IESSpotLight, T.ProjectorLight])
test(`${Light.name} inactive lights stay inert and active unsupported maps fail before GPU writes`, async t => {
  for (const inactive of ['hidden', 'layer', 'disabled']) {
    const f = fixture(t, Light);
    Object.defineProperty(f.light, Light === T.IESSpotLight ? 'iesMap' : 'map', {get() { throw Error('inactive image read'); }});
    if (inactive === 'hidden') f.light.visible = false;
    else if (inactive === 'layer') f.light.layers.set(1);
    else f.renderer.lighting.enabled = false;
    await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
    assert.equal(draws(f).length, 0); assert.equal(f.writes.length, 0);
  }
  const f = fixture(t, Light), texture = new T.DataArrayTexture(new Uint8Array(32), 2, 2, 2);
  texture.needsUpdate = true; t.after(() => texture.dispose());
  f.light[Light === T.IESSpotLight ? 'iesMap' : 'map'] = texture;
  await f.renderer.init(); const allocations = f.textures.length;
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_LIGHTS'});
  assert.equal(f.textures.length, allocations); assert.equal(f.writes.length, 0); assert.equal(draws(f).length, 0);
});
