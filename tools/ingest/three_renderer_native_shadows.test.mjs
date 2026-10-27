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

function animationHost(renderer) {
  const callbacks = new Map(); let serial = 0;
  renderer.domElement.ownerDocument = {defaultView: {
    requestAnimationFrame(callback) { const id = ++serial; callbacks.set(id, callback); return id; },
    cancelAnimationFrame(id) { callbacks.delete(id); },
  }};
  return {get pending() { return callbacks.size; }, tick(time) {
    assert.equal(callbacks.size, 1, 'the source clock owns exactly one scheduled host callback');
    const [id, callback] = callbacks.entries().next().value;
    callbacks.delete(id); callback(time);
  }};
}

function pointFixture(t) {
  const f = fixture(t);
  f.scene.remove(f.light);
  const light = new T.PointLight(0xffe6cc, 30, 15);
  light.position.set(1, 2, 3); light.castShadow = true; light.shadow.mapSize.set(16, 16);
  light.shadow.camera.near = .1; light.shadow.camera.far = 15;
  light.shadow.camera.updateProjectionMatrix(); f.scene.add(light);
  return {...f, light};
}

for (const point of [false, true]) {
  test(`first renderAsync submits ${point ? 'all six point faces' : 'directional shadow'} before its parent and already queued later work`, async t => {
    const f = point ? pointFixture(t) : fixture(t), faces = [];
    f.caster.onBeforeShadow = () => faces.push(f.renderer.getActiveCubeFace());
    const later = new T.Scene(), laterMaterial = new T.MeshBasicNodeMaterial();
    laterMaterial.colorNode = N.uniform(new T.Color(.2, .4, .6)).setName('laterNativeReceiver');
    later.add(new T.Mesh(new T.PlaneGeometry(), laterMaterial));
    assert.equal(f.renderer._initialized, false);
    const first = f.renderer.renderAsync(f.scene, f.camera);
    const second = f.renderer.renderAsync(later, f.camera);
    await Promise.all([first, second]);
    const passes = point ? targetPasses(f.device, 16, 16) : targetPasses(f.device);
    const casters = passes.filter(pass => pass.draws.length), commands = f.device.submissions.flat(2);
    const parent = commands.find(pass => pass.draws?.some(draw => shader(draw).includes('nativeShadowReceiver')));
    const following = commands.find(pass => pass.draws?.some(draw => shader(draw).includes('laterNativeReceiver')));
    assert.equal(casters.length, point ? 6 : 1);
    assert.ok(passes.every(pass => commands.indexOf(pass) < commands.indexOf(parent)),
      'deferred capture moves its accepted prerequisites before the parent depth sampling pass');
    assert.ok(commands.indexOf(parent) < commands.indexOf(following), 'preexisting later FIFO requests stay after their parent');
    assert.deepEqual(faces, point ? [0, 1, 2, 3, 4, 5] : [0], 'resuming parent preparation never replays source callbacks');
    const depth = casters[0].desc.depthStencilAttachment.view.texture;
    assert.ok(textureResources(receiverDraws(f.device)[0]).some(view => view.texture === depth));
    assert.equal(f.renderer.info.f3d.presentedRenders, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
    assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getActiveCubeFace(), 0);
  });
}

test('failed first renderAsync preserves accepted point faces before rejecting its parent and later work', async t => {
  const f = pointFixture(t), failure = new Error('deferred point face failed'), faces = [];
  f.receiver.renderOrder = -1;
  f.caster.onBeforeShadow = () => {
    const face = f.renderer.getActiveCubeFace(); faces.push(face);
    if (face === 2) throw failure;
  };
  const first = f.renderer.renderAsync(f.scene, f.camera);
  const later = f.renderer.renderAsync(new T.Scene(), f.camera);
  await Promise.all([assert.rejects(first, error => error === failure), assert.rejects(later, error => error === failure)]);
  const accepted = targetPasses(f.device, 16, 16).filter(pass => pass.draws.length);
  assert.deepEqual(accepted.map(pass => pass.desc.depthStencilAttachment.view.baseArrayLayer), [0, 1]);
  assert.deepEqual(faces, [0, 1, 2]);
  assert.equal(receiverDraws(f.device).length, 0); assert.equal(f.renderer.info.f3d.presentedRenders, 0);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getActiveCubeFace(), 0);
  assert.equal(f.scene.overrideMaterial, null); assert.equal(f.renderer.getRenderObjectFunction(), null);
});

test('point mapSize edits stay dormant on source update skips and resize at the next requested refresh', async t => {
  const f = pointFixture(t), faces = [];
  f.caster.onBeforeShadow = () => faces.push(f.renderer.getActiveCubeFace());
  await f.renderer.compileAsync(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const map = f.light.shadow.map;
  f.light.shadow.mapSize.set(32, 32);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(faces.length, 6, 'the source same-camera/frame marker skips a dormant desired-size edit');
  f.light.shadow.autoUpdate = false; f.light.shadow.needsUpdate = false;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(faces.length, 6, 'frozen shadows retain the original six faces after the frame advances');
  assert.equal(map.width, 16); assert.equal(map.height, 16);
  assert.equal(f.renderer.info.f3d.presentedRenders, 3);
  f.light.shadow.needsUpdate = true;
  f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(map.width, 32); assert.equal(faces.length, 12);
  assert.equal(map.texture.image[0].width, 16, 'r186 changes the viewport but retains its original cube face allocation extent');
  assert.ok(targetPasses(f.device, 16, 16).slice(-6).every(pass => pass.viewport?.[2] === 32));
  assert.equal(f.light.shadow.needsUpdate, false);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.light.shadow.mapSize.set(16, 16);
  f.light.shadow.needsUpdate = true; f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(faces.length, 18);
  assert.equal(f.light.shadow.needsUpdate, false);
});

test('queued point-shadow resizes preserve the six-face generation sampled by each parent draw', async t => {
  const f = pointFixture(t); await f.renderer.compileAsync(f.scene, f.camera);
  for (const size of [16, 32, 8]) {
    f.light.shadow.mapSize.set(size, size); f.renderer._nodes.nodeFrame.update();
    f.renderer.render(f.scene, f.camera);
  }
  await drain(f.renderer);
  const parents = receiverDraws(f.device), passes = targetPasses(f.device, 16, 16).filter(pass => pass.draws.length);
  assert.equal(parents.length, 3); assert.equal(passes.length, 18);
  const depths = [0, 6, 12].map(index => passes[index].desc.depthStencilAttachment.view.texture);
  assert.equal(new Set(depths).size, 3);
  for (let frame = 0; frame < 3; frame++) {
    const owned = passes.slice(frame * 6, frame * 6 + 6);
    assert.ok(owned.every(pass => pass.desc.depthStencilAttachment.view.texture === depths[frame]));
    assert.ok(textureResources(parents[frame]).some(view => view.texture === depths[frame]));
    assert.ok(owned.every(pass => (pass.viewport?.[2] ?? depths[frame].width) === [16, 32, 8][frame]));
  }
  assert.equal(f.light.shadow.map.width, 8); assert.equal(f.light.shadow.map.texture.image[0].width, 16);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const type of [T.BasicShadowMap, T.PCFShadowMap]) {
  test(`native point shadows submit six captured cube faces before their depth comparison (type ${type})`, async t => {
    const f = pointFixture(t), faces = [];
    f.renderer.shadowMap.type = type;
    f.caster.onBeforeShadow = () => faces.push(f.renderer.getActiveCubeFace());
    await f.renderer.compileAsync(f.scene, f.camera);
    assert.equal(f.device.submissions.length, 0);
    f.renderer.render(f.scene, f.camera);
    assert.deepEqual(faces, [0, 1, 2, 3, 4, 5], 'each source callback sees its current cube face synchronously');
    assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getActiveCubeFace(), 0);
    await drain(f.renderer);
    const passes = targetPasses(f.device, 16, 16), casters = passes.filter(pass => pass.draws.length);
    assert.equal(casters.length, 6);
    assert.equal(passes.filter(pass => pass.desc.depthStencilAttachment.depthLoadOp === 'clear').length, 6);
    assert.deepEqual(casters.map(pass => pass.desc.depthStencilAttachment.view.baseArrayLayer), [0, 1, 2, 3, 4, 5]);
    assert.ok(passes.every(pass => pass.desc.depthStencilAttachment.view.dimension === '2d' &&
      pass.desc.depthStencilAttachment.view.arrayLayerCount === 1));
    const depth = casters[0].desc.depthStencilAttachment.view.texture;
    assert.equal(depth.depthOrArrayLayers, 6);
    assert.ok(casters.every(pass => pass.desc.depthStencilAttachment.view.texture === depth));
    const capturedCasters = nativeDraws(f.device).filter(draw =>
      draw.pipeline.vertex.module.code.includes('nativeShadowDeformation')).slice(0, 6);
    assert.equal(new Set(capturedCasters.map(draw => JSON.stringify(uniformValues(draw)))).size, 6,
      'the shared source camera contributes six separately captured face transforms');
    assert.ok(casters.every(pass => pass.draws[0].pipeline.vertex.module.code.includes('nativeShadowDeformation')));
    const receiver = receiverDraws(f.device)[0];
    assert.match(shader(receiver), /texture_depth_cube/); assert.match(shader(receiver), /textureSampleCompare/);
    assert.ok(textureResources(receiver).some(view => view.texture === depth && view.dimension === 'cube' &&
      view.aspect === 'depth-only'));
    assert.ok([...receiver.groups.values()].some(({group}) => group.layout.entries.some(entry =>
      entry.texture?.sampleType === 'depth' && entry.texture.viewDimension === 'cube')));
    const commands = f.device.submissions.flat(2);
    const receiverPass = commands.find(pass => pass.draws?.some(draw => shader(draw).includes('nativeShadowReceiver')));
    assert.ok(casters.every(pass => commands.indexOf(pass) < commands.indexOf(receiverPass)));
    assert.equal(f.light.shadow.map.texture.renderTarget, null, 'native admission does not rewrite source cube ownership');
    f.light.shadow.autoUpdate = false;
    f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
    assert.equal(targetPasses(f.device, 16, 16).filter(pass => pass.draws.length).length, 6, 'frozen point shadows retain all six faces');
    f.light.shadow.needsUpdate = true;
    f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
    assert.equal(targetPasses(f.device, 16, 16).filter(pass => pass.draws.length).length, 12);
    assert.equal(f.renderer._nodeCaptureBytes, 0); await f.renderer.waitForGPU();
  });
}

test('ordinary point-light materials use one native shadow owner and queued target faces retain their selection', async t => {
  const f = pointFixture(t);
  f.receiver.material = new T.MeshStandardMaterial({color: 0x706050});
  f.caster.material = new T.MeshBasicMaterial({color: 0xc07040});
  f.noncaster.material = new T.MeshBasicMaterial({color: 0x4070c0});
  const target = new T.CubeRenderTarget(16);
  target.depthTexture = new T.CubeDepthTexture(16);
  await f.renderer.init();
  for (const face of [4, 1, 5]) {
    f.renderer.setRenderTarget(target, face); f.renderer.render(f.scene, f.camera);
  }
  f.renderer.setRenderTarget(null); await drain(f.renderer);
  const sampled = nativeDraws(f.device).filter(draw => shader(draw).includes('texture_depth_cube'));
  assert.equal(sampled.length, 3, 'ordinary Standard materials compile into native cube-depth receivers');
  const parents = f.device.passes.filter(pass => pass.draws.some(draw => shader(draw).includes('texture_depth_cube')));
  assert.deepEqual(parents.map(pass => pass.desc.depthStencilAttachment.view.baseArrayLayer), [4, 1, 5]);
  assert.ok(parents.every(pass => pass.desc.colorAttachments[0].view.baseArrayLayer ===
    pass.desc.depthStencilAttachment.view.baseArrayLayer));
  assert.equal(f.renderer.getActiveCubeFace(), 0);
  for (const face of [-1, .5, 6]) assert.throws(() => f.renderer.setRenderTarget(target, face));
  assert.throws(() => f.renderer.setRenderTarget(new T.RenderTarget(16, 16), 1));
  assert.throws(() => f.renderer.setRenderTarget(null, 1));
  assert.throws(() => f.renderer.setRenderTarget(target, 0, 1));
});

test('throwing point-shadow callbacks restore the parent cube face and preserve earlier accepted faces', async t => {
  const f = pointFixture(t), target = new T.CubeRenderTarget(16);
  target.depthTexture = new T.CubeDepthTexture(16);
  await f.renderer.init(); f.renderer.setRenderTarget(target, 4);
  f.receiver.renderOrder = -1;
  f.scene.name = 'Point shadow parent';
  const before = {autoClear: f.renderer.autoClear, color: f.renderer.getClearColor(new T.Color()).clone(),
    alpha: f.renderer.getClearAlpha(), layers: f.light.shadow.camera.layers.mask};
  const failure = new Error('point face callback failed');
  const faces = [];
  f.caster.onBeforeShadow = () => {
    const face = f.renderer.getActiveCubeFace(); faces.push(face);
    if (face === 2) throw failure;
  };
  assert.throws(() => f.renderer.render(f.scene, f.camera), error => error === failure);
  assert.deepEqual(faces, [0, 1, 2]);
  assert.equal(f.renderer.getRenderTarget(), target); assert.equal(f.renderer.getActiveCubeFace(), 4);
  assert.equal(f.renderer.autoClear, before.autoClear); assert.equal(f.renderer.getClearAlpha(), before.alpha);
  assert.deepEqual(f.renderer.getClearColor(new T.Color()), before.color);
  assert.equal(f.scene.name, 'Point shadow parent'); assert.equal(f.scene.overrideMaterial, null);
  assert.equal(f.light.shadow.camera.layers.mask, before.layers);
  assert.equal(f.renderer.getRenderObjectFunction(), null);
  await drain(f.renderer);
  const accepted = targetPasses(f.device, 16, 16).filter(pass => pass.draws.length);
  assert.deepEqual(accepted.map(pass => pass.desc.depthStencilAttachment.view.baseArrayLayer), [0, 1],
    'source face draws accepted before the throw remain ordered; the parent was never accepted');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

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

test('pre-init computeAsync advances source frame one before its first frame-updated uniform', async t => {
  const f = fixture(t), clock = animationHost(f.renderer), updates = [];
  recordCompute(f.device);
  const attribute = new T.StorageBufferAttribute(1, 1), output = N.storage(attribute, 'float', 1);
  let kernel;
  const value = N.uniform(0).onFrameUpdate(frame => {
    assert.equal(frame.renderer, f.renderer); assert.equal(frame.compute, kernel);
    updates.push(frame.frameId); return frame.frameId + .125;
  });
  kernel = N.Fn(() => { output.element(0).assign(value); })().compute(1);
  t.after(() => kernel.dispose());
  assert.equal(f.renderer.initialized, false);
  await f.renderer.computeAsync(kernel); await drain(f.renderer);
  assert.equal(f.renderer.initialized, true);
  assert.equal(f.renderer.info.frame, 1); assert.equal(f.renderer._nodes.nodeFrame.frameId, 1);
  assert.deepEqual(updates, [1]); assert.equal(value.value, 1.125);
  assert.equal(f.device.computePasses.length, 1);
  const uniforms = [...f.device.computePasses[0].groups.values()].flatMap(group => group.entries)
    .filter(entry => entry.resource.buffer?.usage & 64)
    .flatMap(entry => [...new Float32Array(entry.resource.buffer.data)]);
  assert.ok(uniforms.includes(1.125), 'the first submitted kernel receives the updated source uniform');
  assert.equal(clock.pending, 1); assert.equal(f.renderer.getAnimationLoop(), null);
  f.renderer.compute(kernel);
  assert.deepEqual(updates, [1], 'another compute in the same frame does not repeat a frame update');
  clock.tick(16); f.renderer.compute(kernel);
  assert.equal(f.renderer.info.frame, 2); assert.deepEqual(updates, [1, 2]);
  assert.equal(f.device.computePasses.length, 3);
  f.renderer.dispose(); assert.equal(clock.pending, 0);
});

test('explicit compileComputeAsync keeps the source clock and replacement animation callbacks running', async t => {
  const f = fixture(t), clock = animationHost(f.renderer), callbacks = [], updates = [];
  recordCompute(f.device);
  let release, entered;
  const held = new Promise(resolve => { release = resolve; });
  const compiling = new Promise(resolve => { entered = resolve; });
  t.after(release);
  const createPipeline = f.device.createComputePipelineAsync;
  f.device.createComputePipelineAsync = async descriptor => {
    entered(); await held; return createPipeline(descriptor);
  };
  const attribute = new T.StorageBufferAttribute(1, 1), output = N.storage(attribute, 'float', 1);
  const value = N.uniform(0).onFrameUpdate(frame => { updates.push(frame.frameId); return frame.frameId; });
  const kernel = N.Fn(() => { output.element(0).assign(value); })().compute(1);
  t.after(() => kernel.dispose());
  await f.renderer.init();
  const compilation = f.renderer.compileComputeAsync(kernel); compilation.catch(() => {});
  await compiling;
  assert.ok(f.renderer._drain); assert.equal(f.device.computePasses.length, 0);
  clock.tick(16); assert.equal(f.renderer.info.frame, 2);
  await f.renderer.setAnimationLoop(time => { callbacks.push(['first', time]); });
  clock.tick(32);
  await f.renderer.setAnimationLoop(time => { callbacks.push(['replacement', time]); });
  clock.tick(48);
  await f.renderer.setAnimationLoop(null); clock.tick(64);
  assert.deepEqual(callbacks, [['first', 32], ['replacement', 48]]);
  assert.equal(f.renderer.info.frame, 5); assert.equal(clock.pending, 1);
  assert.deepEqual(updates, [1], 'host frames do not rerun the captured compilation updates');
  release(); await compilation; await drain(f.renderer);
  assert.equal(f.device.computePasses.length, 0, 'asynchronous compilation never dispatches the kernel');
  f.renderer.compute(kernel);
  assert.deepEqual(updates, [1, 5]); assert.equal(f.device.computePasses.length, 1);
  assert.equal(clock.pending, 1, 'settling compilation does not install a second host callback');
  f.renderer.dispose(); assert.equal(clock.pending, 0);
});
