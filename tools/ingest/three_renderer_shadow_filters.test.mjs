/** Authored source shadow filters through the native facade and command owner.
 * The recording device checks captured resources/order, not GPU pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const receiverDraws = device => device.snapshots.flat().filter(draw =>
  draw.pipeline.fragment.module.code.includes('customShadowReceiver'));
const shadowPasses = device => device.passes.filter(pass => {
  const texture = pass.desc.depthStencilAttachment?.view?.texture;
  return texture?.width === 32 && texture?.height === 16 && pass.draws.length;
});
function uniformValues(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => {
    const bindings = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform')
      .sort((a, b) => a.binding - b.binding);
    return bindings.flatMap((binding, index) => {
      const resource = group.entries.find(entry => entry.binding === binding.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    });
  });
}

function fixture(t, Light = T.DirectionalLight, type = T.PCFShadowMap) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.textures = []; device.textureWrites = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
    destination.texture.pixels = bytes; device.textureWrites.push({destination, bytes});
  };
  let configuration, acquired = 0;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { acquired++; return device.createTexture({size: [64, 64], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = type;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.set(0, 2, 6); camera.lookAt(0, 0, 0);
  const material = new T.MeshLambertNodeMaterial();
  material.colorNode = N.uniform(new T.Color(.5, .375, .25)).setName('customShadowReceiver');
  const receiver = new T.Mesh(new T.PlaneGeometry(4, 4), material);
  receiver.position.z = -.5; receiver.receiveShadow = true; scene.add(receiver);
  const caster = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicNodeMaterial({color: 0xd58142}));
  caster.position.set(.25, .25, .25); caster.castShadow = true; caster.frustumCulled = false; scene.add(caster);
  const light = new Light(0xffe6cc, 2);
  light.position.set(-3, 4, 5); light.castShadow = true; light.shadow.mapSize.set(32, 16);
  if (light.shadow.camera.isOrthographicCamera)
    Object.assign(light.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4});
  Object.assign(light.shadow.camera, {near: .1, far: 20}); light.shadow.camera.updateProjectionMatrix();
  scene.add(light, light.target, new T.AmbientLight(0xffffff, .2));
  return {renderer, device, scene, camera, receiver, material, caster, light, acquired: () => acquired};
}

for (const [Light, type, wrap] of [
  [T.DirectionalLight, T.PCFShadowMap, N.Fn], [T.SpotLight, T.BasicShadowMap, filter => filter]
]) {
  test(`${Light.name} authored shadow filters retain source samplers, queued uniforms and pass order`, async t => {
    const f = fixture(t, Light, type), events = [], updates = [];
    const gain = N.uniform(.21875).setName('customShadowGain').onObjectUpdate(frame => {
      assert.equal(frame.object, f.receiver); assert.equal(frame.material, f.material);
      assert.equal(frame.renderer, f.renderer); assert.equal(frame.scene, f.scene);
      updates.push(gain.value); events.push('filter-update'); return gain.value;
    });
    let builds = 0;
    f.light.shadow.filterNode = wrap(inputs => {
      builds++; assert.equal(inputs.shadow, f.light.shadow); assert.equal(inputs.depthLayer?.value ?? inputs.depthLayer, 0);
      assert.equal(inputs.depthTexture.isDepthTexture, true);
      return N.texture(inputs.depthTexture, inputs.shadowCoord.xy).compare(inputs.shadowCoord.z).mul(gain);
    });
    f.caster.onBeforeShadow = () => events.push('before-shadow');
    f.caster.onAfterShadow = () => events.push('after-shadow');
    f.receiver.onBeforeRender = () => events.push('before-receiver');
    f.receiver.onAfterRender = () => events.push('after-receiver');
    await f.renderer.init();
    f.renderer.render(f.scene, f.camera);
    gain.value = .8125; f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
    gain.value = 99;
    assert.deepEqual(updates, [.21875, .8125]); assert.equal(builds, 1);
    assert.equal(f.acquired(), 0); assert.equal(f.device.submissions.length, 0);
    await drain(f.renderer);
    const draws = receiverDraws(f.device), shadows = shadowPasses(f.device);
    assert.equal(draws.length, 2); assert.equal(shadows.length, 2);
    assert.equal(draws[0].pipeline, draws[1].pipeline, 'uniform changes retain the source shader');
    assert.ok(uniformValues(draws[0]).includes(.21875)); assert.ok(uniformValues(draws[1]).includes(.8125));
    assert.ok(!uniformValues(draws[0]).includes(99) && !uniformValues(draws[1]).includes(99));
    const expectedFilter = type === T.PCFShadowMap ? 'linear' : 'nearest';
    for (const draw of draws) {
      assert.match(draw.pipeline.fragment.module.code, /textureSampleCompare/);
      const bindings = entries(draw), depth = f.light.shadow.map.depthTexture;
      const sampler = bindings.map(entry => entry.resource).find(resource => resource?.compare === 'less-equal');
      assert.ok(sampler); assert.equal(sampler.minFilter, expectedFilter); assert.equal(sampler.magFilter, expectedFilter);
      assert.equal(depth.minFilter, type === T.PCFShadowMap ? T.LinearFilter : T.NearestFilter);
      assert.ok(bindings.some(entry => entry.resource?.texture === shadows[0].desc.depthStencilAttachment.view.texture));
    }
    const passes = f.device.submissions.flat(2);
    const parents = passes.filter(pass => pass.draws?.some(draw => draw.pipeline.fragment.module.code.includes('customShadowReceiver')));
    assert.ok(passes.indexOf(shadows[0]) < passes.indexOf(parents[0]));
    assert.ok(passes.indexOf(parents[0]) < passes.indexOf(shadows[1]));
    assert.ok(passes.indexOf(shadows[1]) < passes.indexOf(parents[1]));
    assert.deepEqual(events, [...events.slice(0, 5), ...events.slice(0, 5)], 'logical callbacks retain the same per-frame order');
    assert.ok(events.indexOf('before-shadow') < events.indexOf('after-shadow'));
    assert.ok(events.indexOf('before-receiver') < events.indexOf('filter-update'));
    assert.ok(events.indexOf('filter-update') < events.indexOf('after-receiver'));
    assert.equal(f.renderer._nodeCaptureBytes, 0);
  });
}

test('filter replacement follows the retained shadow graph invalidation lifecycle', async t => {
  const f = fixture(t), first = N.uniform(.21875).setName('initialShadowFilter');
  f.light.shadow.filterNode = N.Fn(({depthTexture, shadowCoord}) =>
    N.texture(depthTexture, shadowCoord.xy).compare(shadowCoord.z).mul(first));
  await f.renderer.renderAsync(f.scene, f.camera);
  const original = receiverDraws(f.device)[0];
  let replacements = 0;
  const second = N.uniform(.8125).setName('replacementShadowFilter');
  f.light.shadow.filterNode = N.Fn(({depthTexture, shadowCoord}) => {
    replacements++; return N.texture(depthTexture, shadowCoord.xy).compare(shadowCoord.z).mul(second);
  });
  f.material.needsUpdate = true; f.renderer._nodes.nodeFrame.update();
  await f.renderer.renderAsync(f.scene, f.camera);
  const cached = receiverDraws(f.device)[1];
  assert.equal(replacements, 0, 'r186 caches its shadow output until the light shadow lifecycle resets it');
  assert.match(cached.pipeline.fragment.module.code, /initialShadowFilter/);
  f.light.dispose(); f.material.needsUpdate = true; f.renderer._nodes.nodeFrame.update();
  await f.renderer.renderAsync(f.scene, f.camera);
  const replacement = receiverDraws(f.device)[2];
  assert.equal(replacements, 1); assert.notEqual(replacement.pipeline, original.pipeline);
  assert.match(replacement.pipeline.fragment.module.code, /replacementShadowFilter/);
  assert.doesNotMatch(replacement.pipeline.fragment.module.code, /initialShadowFilter/);
  assert.ok(uniformValues(replacement).includes(.8125)); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('custom filter generated storage writes fail before node updates or GPU work', async t => {
  const f = fixture(t), attribute = new T.StorageBufferAttribute(1, 1), storage = N.storage(attribute, 'float', 1);
  let callbacks = 0;
  f.material.colorNode = N.uniform(new T.Color(.5, .375, .25)).onObjectUpdate(() => { callbacks++; });
  f.caster.onBeforeShadow = () => callbacks++;
  f.light.shadow.filterNode = N.Fn(() => { storage.element(0).assign(.25); return N.float(1); });
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(callbacks, 0); assert.equal(f.acquired(), 0); assert.equal(f.device.submissions.length, 0);
  assert.equal(f.device.writes.length, 0); assert.equal(f.device.textureWrites.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.scene.overrideMaterial, null);
});

test('falsy custom filters preserve the source default and noncallable values fail preflight', async t => {
  const f = fixture(t); f.light.shadow.filterNode = false;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.match(receiverDraws(f.device)[0].pipeline.fragment.module.code, /textureSampleCompare/);
  f.light.shadow.filterNode = N.float(.5);
  const submissions = f.device.submissions.length, acquired = f.acquired();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_SHADOW'});
  assert.equal(f.device.submissions.length, submissions); assert.equal(f.acquired(), acquired);
});
