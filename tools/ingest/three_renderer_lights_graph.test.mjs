/** Authored material lighting through the retained builder and native owner.
 * Recording GPU checks cover source WGSL, bindings and FIFO capture, not pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, Material = T.MeshLambertNodeMaterial) {
  const device = geometryDevice(), captures = [], textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    destination.texture.pixels = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
  };
  let configuration;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(); scene.add(object);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 5;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const frame = capture.call(this, request); if (frame) captures.push(frame); return frame;
  };
  t.after(() => { renderer.dispose(); material.dispose(); object.geometry.dispose(); });
  return {renderer, device, captures, textures, material, object, scene, camera};
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const programs = f => f.captures.flatMap(frame => frame.draws ?? [])
  .filter(draw => draw.object === f.object || draw.binding?.object === f.object).map(draw => draw.binding.nodeProgram);
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
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
function assertSource(f, descriptor, sceneLights = []) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera;
  builder.lightsNode = N.lights(f.renderer.lighting.enabled ? sceneLights : []); builder.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(builder.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(builder.fragmentShader));
}

for (const Material of [T.MeshLambertNodeMaterial, T.MeshLambertMaterial])
test(`${Material.name} selects explicit lights independently of scene membership, visibility and layers`, async t => {
  const f = fixture(t, Material), selected = new T.AmbientLight(new T.Color(.125, .25, .5), 2);
  const ignored = new T.Light(); f.scene.add(ignored);
  selected.visible = false; selected.layers.set(3);
  f.material.lightsNode = N.lights([selected]);
  await f.renderer.renderAsync(f.scene, f.camera);
  assertSource(f, programs(f)[0], [ignored]);
  assert.equal(draws(f).length, 1); assert.ok(contains(uniformValues(draws(f)[0]), [.25, .5, 1]));
  assert.equal(selected.parent, null, 'explicit light selection does not mutate scene ownership');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('authored irradiance, occlusion and environment contributors retain queued source uniforms and callbacks', async t => {
  const f = fixture(t, T.MeshStandardNodeMaterial), events = [];
  const irradiance = N.uniform(new T.Color(.125, .25, .5)).setName('authoredIrradiance').onObjectUpdate(frame => {
    assert.equal(frame.renderer, f.renderer); assert.equal(frame.scene, f.scene);
    assert.equal(frame.object, f.object); assert.equal(frame.material, f.material);
    events.push(irradiance.value.toArray()); return irradiance.value;
  });
  const occlusion = N.uniform(.375).setName('authoredOcclusion');
  const environment = N.uniform(new T.Color(.1875, .3125, .5625)).setName('authoredEnvironment');
  f.material.lightsNode = N.lights([new T.IrradianceNode(irradiance), new T.AONode(occlusion), new T.EnvironmentNode(environment)]);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = programs(f)[0]; assertSource(f, first);
  irradiance.value.setRGB(.625, .75, .875); occlusion.value = .8125;
  environment.value.setRGB(.4375, .6875, .9375); f.renderer.render(f.scene, f.camera);
  irradiance.value.setRGB(99, 99, 99); occlusion.value = 99; environment.value.setRGB(99, 99, 99);
  assert.deepEqual(events, [[.125, .25, .5], [.625, .75, .875]]);
  assert.equal(f.device.submissions.length, 0);
  await f.renderer.waitForGPU();
  const rendered = draws(f); assert.equal(rendered.length, 2); assert.equal(programs(f)[1], first);
  for (const [draw, color, ao, env] of [[rendered[0], [.125, .25, .5], .375, [.1875, .3125, .5625]],
    [rendered[1], [.625, .75, .875], .8125, [.4375, .6875, .9375]]]) {
    const values = uniformValues(draw); assert.ok(contains(values, color)); assert.ok(values.includes(ao));
    assert.ok(contains(values, env)); assert.ok(!values.includes(99));
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('light list edits and replacement graphs keep every previously accepted lighting packet', async t => {
  const f = fixture(t), first = new T.AmbientLight(new T.Color(.125, .25, .5));
  const second = new T.AmbientLight(new T.Color(.625, .75, .875)), list = [first];
  f.material.lightsNode = N.lights(list);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  list.splice(0, 1, second); f.renderer.render(f.scene, f.camera);
  f.material.lightsNode = N.lights([new T.IrradianceNode(N.uniform(new T.Color(.1875, .3125, .5625)))]);
  f.renderer.render(f.scene, f.camera); const selected = programs(f);
  first.color.setRGB(99, 99, 99); second.color.setRGB(99, 99, 99);
  await f.renderer.waitForGPU();
  assert.equal(draws(f).length, 3); assert.notEqual(selected[0], selected[1]); assert.notEqual(selected[1], selected[2]);
  for (const [index, color] of [[0, [.125, .25, .5]], [1, [.625, .75, .875]], [2, [.1875, .3125, .5625]]])
    assert.ok(contains(uniformValues(draws(f)[index]), color));
  assertSource(f, selected[2]); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const [sceneLighting, materialLighting] of [[false, true], [true, false], [false, false]])
test(`explicit lighting remains active with scene=${sceneLighting} and material=${materialLighting}`, async t => {
  const f = fixture(t), light = new T.AmbientLight(new T.Color(.125, .25, .5));
  f.material.lightsNode = N.lights([light]); f.material.lights = materialLighting;
  f.renderer.lighting.enabled = sceneLighting;
  await f.renderer.renderAsync(f.scene, f.camera);
  assertSource(f, programs(f)[0]); assert.ok(contains(uniformValues(draws(f)[0]), [.125, .25, .5]));
});

test('an explicit analytic light node retains its source shadow prerequisite outside the scene light list', async t => {
  const f = fixture(t), light = new T.DirectionalLight(0xffffff, 2);
  light.position.set(-2, 3, 4); light.updateMatrixWorld(); light.target.updateMatrixWorld();
  light.castShadow = true; light.shadow.mapSize.set(32, 16);
  f.material.colorNode = N.uniform(new T.Color(.5, .375, .25)).setName('selectedShadowReceiver');
  f.material.lightsNode = N.lights([new T.DirectionalLightNode(light)]);
  f.object.receiveShadow = f.object.castShadow = true;
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = T.PCFShadowMap;
  t.after(() => light.dispose());
  await f.renderer.renderAsync(f.scene, f.camera);
  const passes = f.device.submissions.flat(2);
  const shadow = passes.find(pass => pass.desc.depthStencilAttachment?.view.texture.width === 32 && pass.draws.length);
  const receiver = passes.find(pass => pass !== shadow &&
    pass.draws.some(draw => draw.pipeline.fragment.module.code.includes('selectedShadowReceiver')));
  assert.ok(shadow); assert.ok(receiver); assert.ok(passes.indexOf(shadow) < passes.indexOf(receiver));
  const depth = shadow.desc.depthStencilAttachment.view.texture;
  assert.ok(receiver.draws.some(draw => entries(draw).some(entry => entry.resource?.texture === depth)));
  assert.equal(light.parent, null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('unsupported generated lighting resources fail before source updates or native GPU effects', async t => {
  const f = fixture(t), storage = N.storage(new T.StorageBufferAttribute(1, 1), 'float', 1);
  let updates = 0;
  f.material.colorNode = N.uniform(new T.Color()).onObjectUpdate(() => { updates++; });
  f.material.lightsNode = N.lights([new T.IrradianceNode(N.Fn(() => {
    storage.element(0).assign(.5); return N.vec3(1);
  })())]);
  await f.renderer.init();
  const allocations = f.textures.length;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0); assert.equal(f.device.writes.length, 0);
  assert.equal(f.textures.length, allocations); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
