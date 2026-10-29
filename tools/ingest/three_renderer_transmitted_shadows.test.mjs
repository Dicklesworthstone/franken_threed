/** Source transmitted-shadow color/depth passes through the native facade.
 * Recorded resources and uniforms prove capture/ordering, not GPU pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const fragment = draw => draw.pipeline.fragment.module.code;
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const textures = draw => entries(draw).map(entry => entry.resource).filter(resource => resource?.texture);
const receiverDraws = device => device.snapshots.flat().filter(draw => fragment(draw).includes('directDiffuse'));
const casterDraws = device => device.snapshots.flat().filter(draw => fragment(draw).includes('transmittedCasterColor'));
const shadowPasses = device => device.passes.filter(pass => {
  const depth = pass.desc.depthStencilAttachment?.view?.texture;
  return depth && depth.width !== 64 && pass.draws.length;
});
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
const contains = (values, expected) => values.some((_, i) => expected.every((value, j) => values[i + j] === value));

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
  Object.assign(renderer.shadowMap, {enabled: true, transmitted: true, type});
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.set(0, 2, 6); camera.lookAt(0, 0, 0);
  const material = new T.MeshLambertMaterial({color: 0xc8b4a0});
  const receiver = new T.Mesh(new T.PlaneGeometry(4, 4), material);
  receiver.position.z = -.5; receiver.receiveShadow = true; receiver.frustumCulled = false; scene.add(receiver);
  const tint = N.uniform(new T.Vector4(.125, .375, .625, .5)).setName('transmittedCasterColor');
  const casterMaterial = new T.MeshBasicMaterial({color: 0xd58142, transparent: true});
  casterMaterial.castShadowNode = tint;
  const caster = new T.Mesh(new T.PlaneGeometry(), casterMaterial);
  caster.position.set(.25, .25, .25); caster.castShadow = true; caster.frustumCulled = false; scene.add(caster);
  const light = new Light(0xffe6cc, Light === T.PointLight ? 30 : 2);
  light.position.set(-3, 4, 5); light.castShadow = true; light.shadow.mapSize.set(16, 16);
  if (light.shadow.camera.isOrthographicCamera)
    Object.assign(light.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4});
  Object.assign(light.shadow.camera, {near: .1, far: 20}); light.shadow.camera.updateProjectionMatrix();
  scene.add(light); if (light.target) scene.add(light.target);
  scene.add(new T.AmbientLight(0xffffff, .2));
  return {renderer, device, scene, camera, receiver, material, caster, casterMaterial, tint, light, acquired: () => acquired};
}

for (const Light of [T.DirectionalLight, T.SpotLight, T.PointLight]) {
  test(`${Light.name} transmitted shadows submit source RGBA and depth before their receiver`, async t => {
    const f = fixture(t, Light), faces = [];
    f.caster.onBeforeShadow = () => faces.push(f.renderer.getActiveCubeFace());
    await f.renderer.renderAsync(f.scene, f.camera);
    const receivers = receiverDraws(f.device), shadows = shadowPasses(f.device), point = Light === T.PointLight;
    assert.equal(receivers.length, 1); assert.equal(shadows.length, point ? 6 : 1);
    assert.deepEqual(faces, point ? [0, 1, 2, 3, 4, 5] : [0]);
    const color = shadows[0].desc.colorAttachments[0].view.texture;
    const depth = shadows[0].desc.depthStencilAttachment.view.texture;
    const inputs = textures(receivers[0]);
    assert.ok(inputs.some(view => view.texture === color)); assert.ok(inputs.some(view => view.texture === depth));
    assert.ok(inputs.filter(view => view.texture === color || view.texture === depth)
      .every(view => view.dimension === (point ? 'cube' : '2d')));
    assert.equal(color.format, 'rgba8unorm');
    assert.match(fragment(receivers[0]), /textureSample\(/); assert.match(fragment(receivers[0]), /textureSampleCompare/);
    const casters = casterDraws(f.device); assert.equal(casters.length, point ? 6 : 1);
    assert.ok(casters.every(draw => contains(uniformValues(draw), [.125, .375, .625, .5])));
    const commands = f.device.submissions.flat(2), parent = commands.find(pass => pass.draws?.some(draw => fragment(draw).includes('directDiffuse')));
    assert.ok(shadows.every(pass => commands.indexOf(pass) < commands.indexOf(parent)));
    assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.scene.overrideMaterial, null);
  });
}

test('ordinary transparent materials select one native transmitted-shadow owner without authored nodes', async t => {
  const f = fixture(t); delete f.casterMaterial.castShadowNode;
  f.casterMaterial.map = new T.DataTexture(new Uint8Array([240, 60, 20, 128]), 1, 1);
  f.casterMaterial.map.needsUpdate = true;
  await f.renderer.renderAsync(f.scene, f.camera);
  const receivers = receiverDraws(f.device), shadows = shadowPasses(f.device);
  assert.equal(receivers.length, 1); assert.equal(shadows.length, 1);
  assert.ok(textures(receivers[0]).some(view => view.texture === shadows[0].desc.colorAttachments[0].view.texture));
  assert.ok(f.device.textureWrites.some(write => write.bytes[3] === 128));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source point shadow construction registers cube metadata before sampling without allocating its GPU images', async t => {
  const f = fixture(t, T.PointLight);
  await f.renderer.compileAsync(f.scene, f.camera);
  const target = f.light.shadow.map;
  assert.ok(target?.isCubeRenderTarget); assert.equal(target.texture.renderTarget, null);
  assert.equal(f.renderer._targets.getTextureTarget(target.texture), target);
  assert.equal(f.renderer._targets.getTextureTarget(target.depthTexture), target);
  assert.equal(f.device.textures.filter(texture => texture.width === 16).length, 0);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.acquired(), 0);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draw = receiverDraws(f.device)[0], shadows = shadowPasses(f.device);
  assert.equal(shadows.length, 6);
  for (const key of ['colorAttachments', 'depthStencilAttachment']) {
    const view = key === 'colorAttachments' ? shadows[0].desc[key][0].view : shadows[0].desc[key].view;
    assert.ok(textures(draw).some(input => input.texture === view.texture));
  }
});

test('queued tint, alpha and shadow resizes retain matching color and depth generations', async t => {
  const f = fixture(t), colors = [[.125, .375, .625, .5], [.75, .25, .125, .8125]];
  await f.renderer.init();
  for (let i = 0; i < colors.length; i++) {
    f.tint.value.fromArray(colors[i]); f.light.shadow.mapSize.set(16 << i, 16 << i);
    f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  }
  f.tint.value.set(99, 99, 99, 99);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.acquired(), 0);
  await drain(f.renderer);
  const receivers = receiverDraws(f.device), shadows = shadowPasses(f.device), casters = casterDraws(f.device);
  assert.equal(receivers.length, 2); assert.equal(shadows.length, 2); assert.equal(casters.length, 2);
  for (let i = 0; i < receivers.length; i++) {
    const color = shadows[i].desc.colorAttachments[0].view.texture;
    const depth = shadows[i].desc.depthStencilAttachment.view.texture;
    assert.equal(color.width, 16 << i); assert.equal(depth.width, 16 << i);
    assert.ok(textures(receivers[i]).some(view => view.texture === color));
    assert.ok(textures(receivers[i]).some(view => view.texture === depth));
    assert.ok(contains(uniformValues(casters[i]), colors[i]));
    assert.ok(!uniformValues(casters[i]).includes(99));
  }
  assert.notEqual(shadows[0].desc.colorAttachments[0].view.texture, shadows[1].desc.colorAttachments[0].view.texture);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('transmitted mode and map type changes follow the retained shadow graph reset', async t => {
  const f = fixture(t);
  await f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.shadowMap.transmitted = false; f.light.shadow.mapType = T.HalfFloatType;
  f.material.needsUpdate = true; f.renderer._nodes.nodeFrame.update();
  await f.renderer.renderAsync(f.scene, f.camera);
  const cached = receiverDraws(f.device)[1];
  assert.ok(textures(cached).some(view => view.texture.format === 'rgba8unorm'),
    'r186 retains its constructed color graph and map type until the source light lifecycle resets');
  f.light.dispose(); f.material.needsUpdate = true; f.renderer._nodes.nodeFrame.update();
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.ok(textures(receiverDraws(f.device)[2]).every(view => view.texture.format.startsWith('depth')));
  f.renderer.shadowMap.transmitted = true; f.light.dispose(); f.material.needsUpdate = true;
  f.renderer._nodes.nodeFrame.update(); await f.renderer.renderAsync(f.scene, f.camera);
  const current = receiverDraws(f.device)[3], map = shadowPasses(f.device).at(-1);
  assert.equal(map.desc.colorAttachments[0].view.texture.format, 'rgba16float');
  assert.ok(textures(current).some(view => view.texture === map.desc.colorAttachments[0].view.texture));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('physical transmission preserves source cast-shadow alpha and material restoration', async t => {
  const f = fixture(t), material = new T.MeshPhysicalMaterial({transmission: .75});
  material.castShadowNode = f.tint; material.alphaTest = .1875;
  f.caster.material = material; f.caster.layers.set(1); f.light.shadow.camera.layers.set(1);
  // The source camera selects this physical caster only for its shadow pass;
  // no visible physical transmission pass is part of this fixture.
  await f.renderer.renderAsync(f.scene, f.camera);
  const draws = casterDraws(f.device); assert.equal(draws.length, 1);
  assert.ok(contains(uniformValues(draws[0]), [.125, .375, .625, .5]));
  assert.ok(uniformValues(draws[0]).includes(.1875));
  assert.match(fragment(draws[0]), /discard/);
  assert.match(fragment(draws[0]), /DiffuseColor/);
  assert.doesNotMatch(fragment(draws[0]), /DiffuseColor\.(?:a|w)\s*=\s*1\.0\s*;/);
  assert.equal(draws[0].pipeline.fragment.targets[0].blend, undefined, 'the source shadow material keeps NoBlending');
  assert.equal(f.caster.material, material); assert.equal(material.transparent, false);
  assert.equal(f.scene.overrideMaterial, null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('unsupported transmitted caster writes still fail before native GPU work', async t => {
  const f = fixture(t), data = N.storage(new T.StorageBufferAttribute(1, 1), 'float', 1);
  f.casterMaterial.castShadowNode = N.Fn(() => { data.element(0).assign(.25); return N.vec4(1); })();
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(f.device.submissions.length, 0); assert.equal(f.device.writes.length, 0);
  assert.equal(f.device.textureWrites.length, 0); assert.equal(f.acquired(), 0);
  assert.equal(f.scene.overrideMaterial, null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
