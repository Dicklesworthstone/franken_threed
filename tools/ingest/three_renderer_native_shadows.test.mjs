/** Native source shadow nodes through the facade and its GPU command owner.
 * The recording device verifies resource identity, callbacks and pass ordering;
 * the browser differential scenarios separately execute the resulting WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';
import {SunLight, SunLightNode} from './fixtures/three_sun_light.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const textureResources = draw => entries(draw).map(entry => entry.resource).filter(resource => resource?.texture);
const storageResources = draw => entries(draw).map(entry => entry.resource)
  .filter(resource => resource?.buffer && (resource.buffer.usage & 128));
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const shader = draw => draw.pipeline.fragment.module.code;
const receiverDraws = device => nativeDraws(device).filter(draw => shader(draw).includes('nativeShadowReceiver'));
const targetPasses = (device, width = 32, height = 16) => device.passes.filter(pass => {
  const texture = pass.desc.depthStencilAttachment?.view?.texture;
  return texture?.width === width && texture?.height === height;
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
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
    destination.texture.pixels = bytes; device.textureWrites.push({destination, bytes});
  };
  const createEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = descriptor => {
    const encoder = createEncoder(descriptor), begin = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = description => {
      const result = begin(description), pass = device.passes.at(-1), viewport = result.setViewport.bind(result);
      result.setViewport = (...values) => { pass.viewport = values; viewport(...values); };
      return result;
    };
    return encoder;
  };
  let configuration, acquired = 0;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { acquired++; return device.createTexture({size: [64, 64], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.set(0, 2, 6); camera.lookAt(0, 0, 0);
  const material = new T.MeshStandardNodeMaterial();
  material.colorNode = N.uniform(new T.Color(.5, .375, .25)).setName('nativeShadowReceiver');
  const receiver = new T.Mesh(new T.PlaneGeometry(4, 4), material);
  receiver.position.z = -.5; receiver.receiveShadow = true; scene.add(receiver);
  const deformation = N.uniform(.125).setName('nativeShadowDeformation');
  const casterMaterial = new T.MeshBasicNodeMaterial({color: 0xd58142});
  casterMaterial.positionNode = N.positionLocal.add(N.vec3(deformation, 0, 0));
  const caster = new T.Mesh(new T.PlaneGeometry(), casterMaterial);
  caster.position.set(.25, .25, .25); caster.castShadow = true; caster.frustumCulled = false; scene.add(caster);
  const noncaster = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicNodeMaterial({color: 0x4b8abe}));
  noncaster.position.x = -1; scene.add(noncaster);
  const light = new T.DirectionalLight(0xffe6cc, 2);
  light.position.set(-3, 4, 5); light.castShadow = true; light.shadow.mapSize.set(32, 16);
  Object.assign(light.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4, near: .1, far: 20});
  light.shadow.camera.updateProjectionMatrix(); scene.add(light, light.target, new T.AmbientLight(0xffffff, .2));
  return {renderer, device, scene, camera, receiver, caster, noncaster, material, casterMaterial,
    light, deformation, acquired: () => acquired};
}

function recordCompute(device) {
  device.computePasses = [];
  device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: index => ({index})});
  device.createComputePipelineAsync = descriptor => Promise.resolve(device.createComputePipeline(descriptor));
  const create = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = descriptor => {
    const encoder = create(descriptor), passes = [];
    return {...encoder, beginComputePass(desc) {
      const pass = {desc, draws: [], groups: new Map(), dispatches: []};
      passes.push(pass); device.computePasses.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; },
        setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...passes]; }};
  };
}

test('native directional PCF submits only visible casters before sampling their depth attachment', async t => {
  const f = fixture(t);
  const hidden = new T.Mesh(f.caster.geometry, f.casterMaterial);
  hidden.castShadow = true; hidden.visible = false; f.scene.add(hidden);
  const otherLayer = new T.Mesh(f.caster.geometry, f.casterMaterial);
  otherLayer.castShadow = true; otherLayer.layers.set(1); f.scene.add(otherLayer);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.device.submissions.length, 0, 'compilation does not render shadow maps');
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const passes = targetPasses(f.device), shadowDraws = passes.flatMap(pass => pass.draws);
  assert.equal(shadowDraws.length, 1, 'noncasters, receivers, invisible objects and unmatched layers do not cast');
  assert.match(shadowDraws[0].pipeline.vertex.module.code, /nativeShadowDeformation/);
  const {frontFace, cullMode} = shadowDraws[0].pipeline.primitive;
  assert.ok(frontFace === 'cw' && cullMode === 'back' || frontFace === 'ccw' && cullMode === 'front',
    'default source caster side is inverted for PCF');
  const receiver = receiverDraws(f.device)[0]; assert.ok(receiver);
  const depth = passes.find(pass => pass.draws.length).desc.depthStencilAttachment.view.texture;
  assert.ok(textureResources(receiver).some(resource => resource.texture === depth), 'receiver borrows the actual shadow depth allocation');
  assert.ok(entries(receiver).some(entry => entry.resource?.compare === 'less-equal'));
  assert.match(shader(receiver), /textureSampleCompare/);
  const commands = f.device.submissions.flat(2);
  assert.ok(commands.indexOf(passes[0]) < commands.findIndex(pass => pass.draws?.some(draw => shader(draw).includes('nativeShadowReceiver'))));
  assert.equal(f.renderer.info.f3d.presentedRenders, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native shadow and visible deformation borrow each queued compute allocation without stale graphics uploads', async t => {
  const f = fixture(t); recordCompute(f.device);
  const count = f.caster.geometry.attributes.position.count, attribute = new T.StorageBufferAttribute(count * 2, 4);
  const storage = N.storage(attribute, 'vec4', attribute.count);
  const index = N.instanceIndex.mul(N.uint(count)).add(N.vertexIndex);
  f.casterMaterial.positionNode = storage.element(index).xyz;
  f.caster.count = 2;
  const kernel = N.Fn(() => { storage.element(N.instanceIndex).addAssign(N.vec4(0)); })().compute(attribute.count);
  attribute.array.fill(.125); await f.renderer.init();
  const first = f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(.625); attribute.needsUpdate = true; f.caster.count = 1;
  f.renderer._nodes.nodeFrame.update();
  const second = f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99); f.caster.count = 99;
  await Promise.all([first, second, drain(f.renderer)]);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  const draws = nativeDraws(f.device).filter(draw => storageResources(draw).some(resource => resource.buffer === buffer));
  assert.equal(draws.length, 4, 'each queued frame has a shadow draw and its visible draw');
  assert.deepEqual(draws.map(draw => draw.args[1]), [2, 2, 1, 1]);
  assert.deepEqual(draws.map(draw => new Float32Array(draw.contents.get(buffer).buffer)[0]), [.125, .125, .625, .625]);
  assert.equal(targetPasses(f.device).flatMap(pass => pass.draws).length, 2);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 2, 'only captured compute versions upload');
  assert.ok(f.device.computePasses.every(pass => [...pass.groups.values()].some(group =>
    group.entries.some(entry => entry.resource?.buffer === buffer))));
  const commands = f.device.submissions.flat(2), shadowPasses = targetPasses(f.device).filter(pass => pass.draws.length);
  for (let index = 0; index < 2; index++) assert.ok(commands.indexOf(f.device.computePasses[index]) < commands.indexOf(shadowPasses[index]));
  assert.equal(f.device.buffers.some(candidate => candidate.usage & 1), false, 'shadow deformation requires no readback allocation');
  assert.equal(f.renderer._nodeCaptureBytes, 0); await f.renderer.waitForGPU();
});

test('native shadow refresh preserves source callback identity, alpha cutoff and frozen autoUpdate', async t => {
  const f = fixture(t), events = [], pixels = new Uint8Array([220, 90, 30, 255, 220, 90, 30, 0]);
  // Source before-updates occur at each draw, so put the receiver before the
  // visible caster to observe its after-shadow mutation in that later capture.
  f.receiver.renderOrder = -1;
  const map = new T.DataTexture(pixels, 2, 1);
  map.minFilter = map.magFilter = T.NearestFilter; map.needsUpdate = true;
  f.casterMaterial.map = map; f.casterMaterial.alphaTest = .5;
  const callback = kind => function (...args) {
    events.push({kind, object: this, args});
    f.deformation.value = kind === 'before' ? .3125 : .8125;
  };
  f.caster.onBeforeShadow = callback('before'); f.caster.onAfterShadow = callback('after');
  f.light.shadow.autoUpdate = false; f.light.shadow.needsUpdate = true;
  await f.renderer.compileAsync(f.scene, f.camera); assert.equal(events.length, 0);
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events.map(event => event.kind), ['before', 'after'], 'source callbacks run synchronously once around capture');
  assert.equal(f.light.shadow.needsUpdate, false);
  for (const event of events) {
    assert.equal(event.object, f.caster);
    assert.equal(event.args[0], f.renderer); assert.equal(event.args[1], f.caster);
    assert.equal(event.args[2], f.light.shadow.camera); assert.equal(event.args[3], f.light.shadow.camera);
    assert.equal(event.args[4], f.caster.geometry); assert.equal(event.args[5].isShadowPassMaterial, true);
    assert.notEqual(event.args[5], f.casterMaterial); assert.equal(event.args[6], null);
  }
  assert.equal(events[0].args[5], events[1].args[5], 'both callbacks receive the same source shadow material');
  await drain(f.renderer);
  const originalMap = f.light.shadow.map;
  const first = targetPasses(f.device).flatMap(pass => pass.draws)[0];
  assert.match(shader(first), /discard/);
  assert.ok(textureResources(first).some(resource => resource.texture.pixels?.every((value, index) => value === pixels[index])));
  const deformed = nativeDraws(f.device).filter(draw => draw.pipeline.vertex.module.code.includes('nativeShadowDeformation'));
  assert.equal(deformed.length, 2);
  assert.ok(uniformValues(deformed[0]).includes(.3125), 'before-shadow mutation reaches the shadow capture');
  assert.ok(uniformValues(deformed[1]).includes(.8125), 'after-shadow mutation reaches the later visible capture');
  f.deformation.value = .75; f.caster.position.x = 1;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(targetPasses(f.device).flatMap(pass => pass.draws).length, 1, 'autoUpdate=false preserves the previous map');
  assert.equal(events.length, 2); assert.equal(f.light.shadow.map, originalMap);
  f.light.shadow.needsUpdate = true;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(targetPasses(f.device).flatMap(pass => pass.draws).length, 2);
  assert.deepEqual(events.map(event => event.kind), ['before', 'after', 'before', 'after']);
  assert.equal(f.light.shadow.needsUpdate, false); assert.equal(f.caster.material, f.casterMaterial);
  assert.equal(receiverDraws(f.device).length, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('throwing source shadow callbacks restore the parent renderer, scene and camera state', async t => {
  const f = fixture(t), background = new T.Color(.1, .2, .3), target = new T.RenderTarget(48, 24);
  f.scene.name = 'Original parent scene'; f.scene.background = background;
  f.receiver.renderOrder = -1; f.camera.layers.enable(2);
  f.renderer.setRenderTarget(target); f.renderer.setClearColor(0x285a8e, .375);
  f.renderer.toneMapping = T.NeutralToneMapping; f.renderer.toneMappingExposure = .75;
  f.renderer.setScissorTest(true);
  const original = {target: f.renderer.getRenderTarget(), renderObject: f.renderer.getRenderObjectFunction(),
    override: f.scene.overrideMaterial, name: f.scene.name, layers: f.light.shadow.camera.layers.mask,
    clearColor: f.renderer.getClearColor(new T.Color()).clone(), clearAlpha: f.renderer.getClearAlpha(),
    autoClear: f.renderer.autoClear, toneMapping: f.renderer.toneMapping,
    exposure: f.renderer.toneMappingExposure, scissorTest: f.renderer.getScissorTest()};
  const failure = new Error('source before-shadow failed'); let before = 0, after = 0;
  f.caster.onBeforeShadow = () => {
    before++; assert.notEqual(f.renderer.getRenderTarget(), target);
    assert.equal(f.scene.overrideMaterial.isShadowPassMaterial, true);
    assert.equal(f.light.shadow.camera.layers.mask, f.camera.layers.mask); throw failure;
  };
  f.caster.onAfterShadow = () => { after++; };
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), error => error === failure);
  await drain(f.renderer);
  assert.equal(before, 1); assert.equal(after, 0);
  assert.equal(f.renderer.getRenderTarget(), original.target);
  assert.equal(f.renderer.getRenderObjectFunction(), original.renderObject);
  assert.equal(f.scene.overrideMaterial, original.override); assert.equal(f.scene.name, original.name);
  assert.equal(f.scene.background, background); assert.equal(f.light.shadow.camera.layers.mask, original.layers);
  assert.deepEqual(f.renderer.getClearColor(new T.Color()), original.clearColor);
  assert.equal(f.renderer.getClearAlpha(), original.clearAlpha); assert.equal(f.renderer.autoClear, original.autoClear);
  assert.equal(f.renderer.toneMapping, original.toneMapping); assert.equal(f.renderer.toneMappingExposure, original.exposure);
  assert.equal(f.renderer.getScissorTest(), original.scissorTest);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.acquired(), 0);
  assert.equal(f.renderer.info.f3d.presentedRenders, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source shadow callbacks cannot overwrite a target already sampled by the pending parent', async t => {
  const f = fixture(t), target = new T.RenderTarget(2, 2, {depthBuffer: false});
  const initial = new Uint8Array([220, 30, 70, 255, 30, 180, 220, 255, 40, 200, 80, 255, 220, 180, 30, 255]);
  f.noncaster.material.colorNode = N.texture(target.texture).rgb.mul(N.uniform(1).setName('earlierShadowTargetRead'));
  f.noncaster.renderOrder = -2; f.noncaster.frustumCulled = false; f.receiver.renderOrder = -1;
  await f.renderer.init(); await f.renderer.writeRenderTargetPixelsAsync(target, initial);
  const uploaded = f.device.textureWrites.at(-1).destination.texture;
  const writes = f.device.textureWrites.filter(write => write.destination.texture === uploaded).length;
  let rejection, callbacks = 0;
  f.caster.onBeforeShadow = () => {
    callbacks++;
    const pending = f.renderer._pending.length;
    rejection = f.renderer.writeRenderTargetPixelsAsync(target, new Uint8Array(16).fill(99))
      .then(() => null, error => error);
    assert.equal(f.renderer._pending.length, pending, 'rejected transfer adds no FIFO request');
    assert.equal(f.device.textureWrites.filter(write => write.destination.texture === uploaded).length, writes);
  };
  f.renderer.render(f.scene, f.camera);
  assert.equal(callbacks, 1); assert.equal((await rejection)?.code, 'THREE_NODE_PROGRAM_SHADOW');
  await drain(f.renderer);
  const earlier = nativeDraws(f.device).find(draw => shader(draw).includes('earlierShadowTargetRead'));
  assert.ok(earlier);
  assert.ok(textureResources(earlier).some(resource => resource.texture === uploaded));
  assert.deepEqual(uploaded.pixels, initial, 'earlier draw keeps the preseeded target contents');
  assert.equal(f.device.textureWrites.filter(write => write.destination.texture === uploaded).length, writes);
  assert.equal(targetPasses(f.device).flatMap(pass => pass.draws).length, 1);
  assert.equal(f.renderer.info.f3d.presentedRenders, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('real SunLight cascades share one atlas and ordinary ShadowMaterial receives both source shadows', async t => {
  const f = fixture(t); f.scene.remove(f.light, f.light.target, f.noncaster);
  const sun = new SunLight(0xffffff, 2); sun.position.set(-4, 10, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(64, 32); sun.shadow.camera.far = 20; f.scene.add(sun);
  f.renderer.library.addLight(SunLightNode, SunLight);
  const shadowMaterial = new T.ShadowMaterial({color: 0x5d6568, opacity: .3});
  f.receiver.material = shadowMaterial; f.receiver.rotation.x = -Math.PI / 2; f.receiver.position.set(0, -.5, 0);
  f.caster.count = 2;
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.device.submissions.length, 0);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const passes = targetPasses(f.device, 128, 32), drawn = passes.filter(pass => pass.draws.length);
  assert.equal(drawn.length, 2, 'both source cascades produce native passes');
  assert.deepEqual(drawn.map(pass => pass.draws.length), [1, 1]);
  assert.deepEqual(drawn.map(pass => pass.draws[0].args[1]), [2, 2], 'ordinary Mesh.count is retained in both cascades');
  const depth = drawn[0].desc.depthStencilAttachment.view.texture;
  assert.equal(drawn[1].desc.depthStencilAttachment.view.texture, depth);
  for (let index = 0; index < 2; index++) {
    const tile = sun.shadow.getViewport(index);
    const expected = [64 * tile.x, 32 - 32 * (tile.y + tile.w), 64 * tile.z, 32 * tile.w];
    assert.deepEqual(drawn[index].viewport.slice(0, 4), expected);
    assert.equal(drawn[index].desc.depthStencilAttachment.depthLoadOp, 'load', 'a cascade preserves the other tile');
  }
  assert.equal(passes.filter(pass => pass.desc.depthStencilAttachment.depthLoadOp === 'clear').length, 1, 'source atlas is cleared once');
  const receivers = nativeDraws(f.device).filter(draw => textureResources(draw).some(resource => resource.texture === depth));
  assert.equal(receivers.length, 1); assert.match(shader(receivers[0]), /textureSampleCompare/);
  assert.ok(receivers[0].pipeline.fragment.targets[0].blend, 'ShadowMaterial keeps its transparent receiver blending');
  assert.equal(f.receiver.material, shadowMaterial); assert.equal(sun.shadow.map.width, 128); assert.equal(sun.shadow.map.height, 32);
  assert.equal(f.renderer.info.f3d.presentedRenders, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const cascades of [false, true]) test(`queued ${cascades ? 'SunLight atlas' : 'directional shadow'} resizes retain each accepted depth generation`, async t => {
  const f = fixture(t);
  let light = f.light;
  if (cascades) {
    f.scene.remove(light, light.target);
    light = new SunLight(0xffffff, 2); light.position.set(-4, 10, 8); light.castShadow = true;
    light.shadow.mapSize.set(32, 16); light.shadow.camera.far = 20;
    f.scene.add(light); f.renderer.library.addLight(SunLightNode, SunLight);
  }
  await f.renderer.compileAsync(f.scene, f.camera);
  const map = light.shadow.map, depthTexture = map.depthTexture;
  const sizes = [[32, 16], [64, 32], [32, 16]];
  for (const [width, height] of sizes) {
    light.shadow.mapSize.set(width, height);
    f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
    assert.equal(light.shadow.map, map); assert.equal(map.depthTexture, depthTexture);
  }
  assert.equal(f.renderer._targets.size, 3, 'all accepted map generations remain live before the FIFO drains');
  await drain(f.renderer); await f.renderer.waitForGPU();
  const receivers = receiverDraws(f.device);
  assert.equal(receivers.length, 3);
  const depths = receivers.map(draw => textureResources(draw).find(resource => resource.texture.format.startsWith('depth')).texture);
  assert.deepEqual(depths.map(texture => [texture.width, texture.height]), sizes.map(([width, height]) => [width * (cascades ? 2 : 1), height]));
  assert.equal(new Set(depths).size, 3, 'returning to an earlier size does not alias its queued contents');
  const commands = f.device.submissions.flat(2);
  for (const depth of depths) {
    const writers = f.device.passes.filter(pass => pass.desc.depthStencilAttachment?.view?.texture === depth && pass.draws.length);
    assert.equal(writers.length, cascades ? 2 : 1);
    const reader = commands.findIndex(pass => pass.draws?.some(draw => textureResources(draw).some(resource => resource.texture === depth)));
    assert.ok(writers.every(pass => commands.indexOf(pass) < reader));
  }
  assert.ok(depths.slice(0, 2).every(texture => texture.destroyed)); assert.notEqual(depths[2].destroyed, true);
  assert.equal(f.renderer._targets.size, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.equal(f.renderer.info.f3d.presentedRenders, 3);
});

test('explicit shadow target resize or disposal still invalidates every queued generation', async t => {
  for (const change of [map => map.dispose(), map => map.setSize(128, 64)]) {
    const f = fixture(t);
    await f.renderer.compileAsync(f.scene, f.camera);
    f.renderer.render(f.scene, f.camera);
    f.light.shadow.mapSize.set(64, 32); f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
    assert.equal(f.renderer._targets.size, 2);
    change(f.light.shadow.map);
    await assert.rejects(drain(f.renderer), {code: 'THREE_TARGET_STALE'});
    assert.equal(f.renderer.info.f3d.presentedRenders, 0);
    assert.equal(f.renderer._targets.size, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});

test('hidden H7 source skins do not poison their captured visible computed sibling', async t => {
  const f = fixture(t), character = new T.Group(); character.name = 'Character';
  f.scene.remove(f.caster); character.add(f.caster); f.scene.add(character);
  const originalMaterial = new T.MeshPhysicalMaterial(), originalGeometry = new T.BufferGeometry();
  const source = new T.SkinnedMesh(originalGeometry, originalMaterial);
  source.name = 'Original source skin'; source.visible = false; source.castShadow = true;
  const hiddenAccess = () => { throw new Error('hidden source render state was inspected'); };
  source.onBeforeRender = source.onAfterRender = source.onBeforeShadow = source.onAfterShadow = hiddenAccess;
  Object.defineProperty(source, 'geometry', {configurable: true, get: hiddenAccess});
  Object.defineProperty(source, 'material', {configurable: true, get: hiddenAccess});
  character.add(source);
  let updates = 0;
  f.material.colorNode = f.material.colorNode.mul(N.uniform(1).onObjectUpdate(() => { updates++; return 1; }));
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.equal(updates, 1); assert.equal(f.acquired(), 0);
  source.visible = true;
  await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.presentedRenders, 1, 'later visibility cannot change an accepted parent or shadow pass');
  assert.equal(targetPasses(f.device).flatMap(pass => pass.draws).length, 1);
  assert.equal(receiverDraws(f.device).length, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
  Object.defineProperty(source, 'geometry', {configurable: true, writable: true, value: f.caster.geometry});
  Object.defineProperty(source, 'material', {configurable: true, writable: true, value: originalMaterial});
  source.onBeforeRender = T.Object3D.prototype.onBeforeRender; source.onAfterRender = T.Object3D.prototype.onAfterRender;
  source.onBeforeShadow = T.Object3D.prototype.onBeforeShadow; source.onAfterShadow = T.Object3D.prototype.onAfterShadow;
  const submissions = f.device.submissions.length;
  f.renderer._nodes.nodeFrame.update();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_OBJECT'});
  assert.equal(updates, 1, 'new visible unsupported source is rejected before application draw updates');
  assert.equal(f.device.submissions.length, submissions); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
