/** Real facade/capture/resource/scene integration with recorded WebGPU commands.
 * Copies, clears and uniform bytes are observed; WGSL is not rasterized here. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';
const N = THREE.TSL;

async function fixture(t, options = {}, initialize = true) {
  const device = geometryDevice(), events = [], textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.features = new Set(['float32-filterable', 'float32-blendable']);
  device.createTexture = descriptor => {
    const width = descriptor.size.width ?? descriptor.size[0], height = descriptor.size.height ?? descriptor.size[1];
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor, width, height,
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(descriptor = {}) { return {texture: this, descriptor}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = descriptor => {
    const encoder = createEncoder(), commands = [];
    return {...encoder,
      beginRenderPass(descriptor) {
        const pass = encoder.beginRenderPass(descriptor), record = device.passes.at(-1);
        commands.push(record);
        return {...pass, setViewport(...value) { record.viewport = value; pass.setViewport(...value); },
          setScissorRect(...value) { record.scissor = value; pass.setScissorRect(...value); }};
      },
      copyTextureToTexture(source, destination, size) {
        assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
        assert.equal(source.texture.sampleCount, destination.texture.sampleCount);
        if (!source.texture.format.startsWith('depth')) assert.equal(source.texture.sampleCount, 1);
        assert.equal(source.texture.format, destination.texture.format);
        commands.push({copy: true, source, destination, size, draws: [], label: descriptor?.label});
      }, finish() { return commands; }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const list of commands) for (const command of list) events.push(command);
    submit(commands);
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { canvas.acquisitions = (canvas.acquisitions ?? 0) + 1; return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format, usage: configuration.usage}); }
  })};
  const Renderer = createWebGPURendererClass(THREE), renderer = new Renderer({canvas, device, ...options});
  t.after(() => renderer.dispose()); if (initialize) await renderer.init();
  const scene = new THREE.Scene(); scene.add(new THREE.AmbientLight(0xffffff, .5));
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const mesh = material => {
    const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.frustumCulled = false; scene.add(object); return object;
  };
  return {device, renderer, scene, camera, mesh, events, textures, canvas};
}
function physical(name, options = {}) {
  const material = new THREE.MeshPhysicalMaterial({transmission: .75, thickness: .375, roughness: .25, dispersion: .5, ...options});
  if (name) material.colorNode = N.uniform(new THREE.Color(.25, .5, .75)).setName(name);
  return material;
}
const code = draw => draw.pipeline.fragment?.module?.code ?? '';
const meshDraws = events => events.flatMap(event => event.draws).filter(draw => draw.indexed);
const copies = events => events.filter(event => event.copy);
const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries ?? [])
  .map(entry => entry.resource?.texture).filter(Boolean);
const uniformValues = draw => [...draw.contents.values()].flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);


test('cold public canvas copy stays between producer and sampled consumer', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial({color: 0x804020}));
  const image = new THREE.FramebufferTexture(8, 8);
  renderer.render(scene, camera);
  assert.equal(renderer.copyFramebufferToTexture(image, new THREE.Vector2(2.9, 1.9)), undefined);
  assert.equal(image.type, THREE.UnsignedByteType);
  const consumer = new THREE.Scene(), material = new THREE.MeshBasicNodeMaterial();
  material.colorNode = N.texture(image).rgb;
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material);
  object.frustumCulled = false; consumer.add(object);
  renderer.render(consumer, camera); await renderer.waitForGPU();
  assert.equal(image.type, THREE.UnsignedByteType, 'sampling preserves the initialized public type');
  const recorded = copies(f.events); assert.equal(recorded.length, 1);
  const copy = recorded[0], at = f.events.indexOf(copy);
  assert.equal(copy.source.texture.format, 'rgba16float');
  assert.deepEqual(copy.source.origin, {x: 2, y: 1, z: 0});
  assert.deepEqual(copy.size, {width: 8, height: 8, depthOrArrayLayers: 1});
  assert.equal(meshDraws(f.events.slice(0, at)).length, 1);
  const draws = meshDraws(f.events.slice(at + 1)); assert.equal(draws.length, 1);
  assert.ok(sampled(draws[0]).includes(copy.destination.texture));
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

for (const [type, format] of [[THREE.UnsignedByteType, 'rgba8unorm'], [THREE.HalfFloatType, 'rgba16float'], [THREE.FloatType, 'rgba32float']])
  test(`ordinary canvas retains ${format} scene color for a later public copy`, async t => {
    const f = await fixture(t, {outputBufferType: type}), {renderer, scene, camera} = f;
    f.mesh(new THREE.MeshBasicMaterial()); await renderer.renderAsync(scene, camera);
    const image = new THREE.FramebufferTexture(16, 16), calls = renderer.info.calls;
    assert.equal(renderer.copyFramebufferToTexture(image), undefined);
    const recorded = copies(f.events); assert.equal(recorded.length, 1, 'prepared copy submits synchronously');
    const copy = recorded[0], pass = f.events.find(event => event.draws.some(draw => draw.indexed));
    assert.equal(copy.source.texture.format, format); assert.equal(copy.destination.texture.format, format);
    assert.equal(copy.source.texture, pass.desc.colorAttachments[0].view.texture);
    assert.equal(renderer.info.calls, calls, 'copy does not render the scene again');
    await renderer.waitForGPU();
  });

test('public copies capture source target and source rectangle before later calls', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial());
  const first = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType});
  const second = new THREE.RenderTarget(8, 8, {type: THREE.HalfFloatType});
  t.after(() => { first.dispose(); second.dispose(); });
  const image = new THREE.FramebufferTexture(8, 8), rectangle = new THREE.Vector4(3.9, 2.9, 5.9, 4.9), reference = [];
  THREE.Renderer.prototype.copyFramebufferToTexture.call({
    _currentRenderContext: null, _renderTarget: first, _getFrameBufferTarget: () => null,
    _textures: {updateRenderTarget() {}, get: target => ({renderTarget: target}), updateTexture(texture, options) { reference.push(options.renderTarget); }},
    backend: {copyFramebufferToTexture(texture, context, rect) { reference.push([rect.x, rect.y, rect.z, rect.w]); }},
    _inspector: {copyFramebufferToTexture() {}},
  }, image, rectangle);
  renderer.setRenderTarget(first); renderer.render(scene, camera);
  renderer.copyFramebufferToTexture(image, rectangle); rectangle.set(0, 0, 1, 1);
  renderer.setRenderTarget(second); renderer.render(scene, camera); await renderer.waitForGPU();
  const recorded = copies(f.events); assert.equal(recorded.length, 1); const copy = recorded[0];
  assert.equal(reference[0], first);
  assert.deepEqual([copy.source.origin.x, copy.source.origin.y, copy.size.width, copy.size.height], reference[1]);
  assert.equal(copy.source.texture.width, 16); assert.equal(image.type, THREE.HalfFloatType);
  assert.equal(copy.source.mipLevel, 0); assert.equal(copy.source.origin.z, 0);
  const at = f.events.indexOf(copy);
  assert.equal(meshDraws(f.events.slice(0, at)).length, 1); assert.equal(meshDraws(f.events.slice(at + 1)).length, 1);
  assert.equal(renderer._targets.capture(first).references, 0);
});

test('public volume framebuffer copies preserve source layer-zero/mip-zero behavior despite the active render slice', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const volume = new THREE.RenderTarget3D(8, 8, 4, {type: THREE.HalfFloatType});
  volume.texture.mipmaps = [{}, {}, {}, {}];
  const image = new THREE.FramebufferTexture(5, 4), rectangle = new THREE.Vector4(2, 1, 5, 4);
  t.after(() => { volume.dispose(); image.dispose(); });
  f.mesh(new THREE.MeshBasicMaterial());
  renderer.setRenderTarget(volume, 1, 1); renderer.render(scene, camera);
  renderer.copyFramebufferToTexture(image, rectangle);
  renderer.setRenderTarget(volume, 3, 0); rectangle.set(0, 0, 1, 1);
  await renderer.waitForGPU();
  const [copy] = copies(f.events); assert.equal(copies(f.events).length, 1);
  const pass = f.events.find(event => event.draws.some(draw => draw.indexed));
  assert.equal(pass.desc.colorAttachments[0].depthSlice, 1);
  assert.equal(pass.desc.colorAttachments[0].view.descriptor.baseMipLevel, 1);
  assert.equal(copy.source.texture, pass.desc.colorAttachments[0].view.texture);
  assert.equal(copy.source.texture.dimension, '3d'); assert.equal(copy.destination.texture.dimension, '2d');
  assert.equal(renderer._targets.capture(volume).references, 0);

  // Exercise the retained copy implementation, not an inferred "current view"
  // policy. r186 copies the base image's first slice even though its preceding
  // render used slice 1/mip 1; this rectangle only fits that base image.
  const reference = [], backend = new THREE.WebGPUBackend();
  const command = (source, destination, size) => ({source: {mipLevel: source.mipLevel, origin: {...source.origin}},
    destination: {mipLevel: destination.mipLevel, origin: {...destination.origin}}, size: {...size}});
  backend.device = {queue: {submit() {}}, createCommandEncoder() { return {
    copyTextureToTexture(source, destination, size) { reference.push(command(source, destination, size)); }, finish() { return {}; },
  }; }};
  backend.get(volume.texture).texture = copy.source.texture;
  backend.get(image).texture = copy.destination.texture;
  const context = {renderTarget: volume, textures: volume.textures, depthTexture: null, camera,
    activeCubeFace: 1, activeMipmapLevel: 1};
  THREE.Renderer.prototype.copyFramebufferToTexture.call({
    _currentRenderContext: null, _renderTarget: volume, _activeCubeFace: 1, _activeMipmapLevel: 1,
    _textures: {updateRenderTarget() {}, get: () => context, updateTexture() {}},
    backend, _inspector: {copyFramebufferToTexture() {}},
  }, image, new THREE.Vector4(2, 1, 5, 4));
  assert.deepEqual(command(copy.source, copy.destination, copy.size), reference[0]);
  assert.equal(reference.length, 1);
  assert.deepEqual(reference[0].source, {mipLevel: 0, origin: {x: 2, y: 1, z: 0}});
});

test('queued public copies retain destination generations through framebuffer resize', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial()); const image = new THREE.FramebufferTexture(16, 16);
  renderer.render(scene, camera); renderer.copyFramebufferToTexture(image);
  image.image.width = 8; image.image.height = 8; image.needsUpdate = true;
  renderer.copyFramebufferToTexture(image); await renderer.waitForGPU();
  const recorded = copies(f.events); assert.equal(recorded.length, 2);
  assert.deepEqual(recorded.map(copy => copy.size.width), [16, 8]);
  assert.deepEqual(recorded.map(copy => copy.destination.texture.width), [16, 8]);
  assert.notEqual(recorded[0].destination.texture, recorded[1].destination.texture);
  assert.equal(recorded[0].source.texture, recorded[1].source.texture);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('public copy and destination mip generation share one encoder', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial()); await renderer.renderAsync(scene, camera);
  const image = new THREE.FramebufferTexture(16, 16); image.generateMipmaps = true;
  const before = f.device.submissions.length; renderer.copyFramebufferToTexture(image);
  assert.equal(f.device.submissions.length, before + 1);
  const submission = f.device.submissions.at(-1); assert.equal(submission.length, 1);
  assert.equal(submission[0].length, 5); assert.equal(submission[0][0].copy, true);
  assert.equal(submission[0][0].destination.texture.mipLevelCount, 5);
  await renderer.waitForGPU();
});

test('disposed queued framebuffer copies reject and release destination leases', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial()); const image = new THREE.FramebufferTexture(16, 16);
  renderer.render(scene, camera); renderer.copyFramebufferToTexture(image);
  const textures = renderer._getSampledTextureInputs(); image.dispose();
  await assert.rejects(renderer.waitForGPU(), error => error.code === 'THREE_FRAMEBUFFER_TEXTURE_STALE');
  assert.equal(copies(f.events).length, 0);
  assert.equal(textures.diagnostics.framebuffers.bindings, 0);
});

test('invalid public source rectangles fail before queueing copy leases', async t => {
  const f = await fixture(t), {renderer} = f, image = new THREE.FramebufferTexture(8, 8);
  assert.throws(() => renderer.copyFramebufferToTexture(image, new THREE.Vector4(12, 0, 8, 8)), error => error.code === 'F3D_RENDERER_COPY');
  assert.equal(renderer._pending.length, 0); assert.equal(copies(f.events).length, 0);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers?.bindings ?? 0, 0);
});

test('same-version framebuffer copies preserve source type until reallocation', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial());
  const target = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType}); t.after(() => target.dispose());
  const image = new THREE.FramebufferTexture(16, 16);
  renderer.setRenderTarget(target); renderer.render(scene, camera); renderer.copyFramebufferToTexture(image);
  assert.equal(image.type, THREE.HalfFloatType);
  renderer.setRenderTarget(null); renderer.render(scene, camera); renderer.copyFramebufferToTexture(image);
  assert.equal(image.type, THREE.HalfFloatType, 'initialized same-version texture skips type reassignment');
  image.needsUpdate = true; renderer.copyFramebufferToTexture(image);
  assert.equal(image.type, THREE.UnsignedByteType, 'new version uses the public canvas target');
  await renderer.waitForGPU();
  const recorded = copies(f.events); assert.equal(recorded.length, 3);
  assert.equal(recorded[0].destination.texture, recorded[1].destination.texture);
  assert.notEqual(recorded[1].destination.texture, recorded[2].destination.texture);
  assert.ok(recorded.every(copy => copy.destination.texture.format === 'rgba16float'));
});

test('public direct linear canvas copies submit without an output draw', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  f.mesh(new THREE.MeshBasicMaterial()); await renderer.renderAsync(scene, camera);
  assert.equal(renderer._hdr, false);
  const before = f.events.length, calls = renderer.info.calls;
  renderer.copyFramebufferToTexture(new THREE.FramebufferTexture(16, 16));
  const recorded = copies(f.events.slice(before)); assert.equal(recorded.length, 1);
  assert.equal(recorded[0].source.texture.format, renderer._preferredFormat ?? 'bgra8unorm');
  assert.equal(recorded[0].source.texture.sampleCount, 1);
  assert.equal(f.events.slice(before).flatMap(event => event.draws).length, 0);
  assert.equal(renderer.info.calls, calls); await renderer.waitForGPU();
});

test('public MSAA target copies read resolved primary color', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const target = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType, samples: 4});
  t.after(() => target.dispose()); renderer.setRenderTarget(target);
  f.mesh(new THREE.MeshBasicMaterial()); renderer.render(scene, camera);
  renderer.copyFramebufferToTexture(new THREE.FramebufferTexture(16, 16));
  await renderer.waitForGPU();
  const pass = f.events.find(event => event.draws.some(draw => draw.indexed));
  const [copy] = copies(f.events); assert.ok(copy);
  assert.equal(copy.source.texture, pass.desc.colorAttachments[0].resolveTarget.texture);
  assert.equal(copy.source.texture.sampleCount, 1);
  assert.equal(pass.desc.colorAttachments[0].view.texture.sampleCount, 4);
  assert.equal(renderer._targets.capture(target).references, 0);
});

test('disposing a queued public source rejects before copy and releases both leases', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const target = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType});
  f.mesh(new THREE.MeshBasicMaterial()); renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.copyFramebufferToTexture(new THREE.FramebufferTexture(16, 16));
  const source = renderer._targets.capture(target), textures = renderer._getSampledTextureInputs();
  assert.ok(source.references > 0); target.dispose();
  await assert.rejects(renderer.waitForGPU(), error => error.code === 'THREE_TARGET_STALE');
  assert.equal(copies(f.events).length, 0); assert.equal(source.references, 0);
  assert.equal(textures.diagnostics.framebuffers.bindings, 0);
});

test('invalid queued direct copies preserve the producer presentation', async t => {
  const f = await fixture(t), {renderer, scene, camera, canvas} = f;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  f.mesh(new THREE.MeshBasicMaterial()); renderer.render(scene, camera);
  const image = new THREE.FramebufferTexture(16, 16);
  renderer.copyFramebufferToTexture(image);
  const textures = renderer._getSampledTextureInputs(); image.dispose();
  await assert.rejects(renderer.waitForGPU(), error => error.code === 'THREE_FRAMEBUFFER_TEXTURE_STALE');
  assert.equal(canvas.acquisitions, 1, 'only the accepted producer acquires a presentation image');
  assert.equal(copies(f.events).length, 0);
  assert.equal(textures.diagnostics.framebuffers.bindings, 0);
});

for (const samples of [0, 4]) test(`public depth snapshots preserve private target storage and ${samples || 1} samples`, async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const target = new THREE.RenderTarget(16, 16, {samples}); t.after(() => target.dispose());
  renderer.setRenderTarget(target);
  f.mesh(new THREE.MeshBasicMaterial());
  renderer.render(scene, camera);
  const depth = new THREE.DepthTexture(16, 16);
  renderer.copyFramebufferToTexture(depth);
  const consumer = new THREE.Scene(), material = new THREE.MeshBasicNodeMaterial();
  material.colorNode = N.vec3(N.textureLoad(depth, N.ivec2(0, 0)).r);
  consumer.add(new THREE.Mesh(new THREE.PlaneGeometry(), material));
  renderer.setRenderTarget(null); renderer.render(consumer, camera); await renderer.waitForGPU();
  const recorded = copies(f.events); assert.equal(recorded.length, 1);
  const copy = recorded[0], at = f.events.indexOf(copy);
  const producer = f.events.slice(0, at).find(event => event.draws.some(draw => draw.indexed));
  assert.equal(target.depthTexture, null, 'copying private depth does not manufacture a public depth attachment');
  assert.equal(copy.source.texture, producer.desc.depthStencilAttachment.view.texture);
  assert.equal(copy.destination.texture.sampleCount, samples || 1);
  assert.equal(depth.type, THREE.UnsignedIntType, 'depth destination type remains authored');
  assert.ok(meshDraws(f.events.slice(at + 1)).some(draw => sampled(draw).includes(copy.destination.texture)));
  assert.equal(renderer._getSampledTextureInputs().diagnostics.depth.bindings, 0);
});

for (const options of [{}, {antialias: true}, {linear: true}])
  test(`viewport depth reads preceding draws and preserves later depth loads ${JSON.stringify(options)}`, async t => {
    const {linear, ...parameters} = options, f = await fixture(t, parameters), {renderer, scene, camera} = f;
    if (linear) renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    const prefix = f.mesh(new THREE.MeshBasicMaterial()); prefix.renderOrder = -1;
    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = N.vec3(N.viewportDepthTexture().r); material.depthWrite = false;
    f.mesh(material);
    await renderer.compileAsync(scene, camera);
    assert.equal(copies(f.events).length, 0, 'compilation does not snapshot depth');
    await renderer.renderAsync(scene, camera);
    const recorded = copies(f.events); assert.equal(recorded.length, 1);
    const copy = recorded[0], at = f.events.indexOf(copy);
    assert.equal(copy.source.texture.format, 'depth24plus');
    assert.equal(copy.source.texture.sampleCount, options.antialias ? 4 : 1);
    assert.equal(meshDraws(f.events.slice(0, at)).length, 1);
    const after = f.events.slice(at + 1).find(event => event.draws.some(draw => draw.indexed));
    assert.equal(after.desc.depthStencilAttachment.depthLoadOp, 'load');
    assert.equal(after.desc.depthStencilAttachment.view.texture, copy.source.texture);
    assert.ok(sampled(after.draws.find(draw => draw.indexed)).includes(copy.destination.texture));
    assert.equal(renderer._getSampledTextureInputs().diagnostics.depth.bindings, 0);
  });

test('public depth snapshots preserve matching depth/stencil and float target formats', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; f.mesh(new THREE.MeshBasicMaterial());
  for (const [type, format, stencilBuffer, native] of [[THREE.UnsignedInt248Type, THREE.DepthStencilFormat, true, 'depth24plus-stencil8'],
    [THREE.FloatType, THREE.DepthFormat, false, 'depth32float']]) {
    const attached = new THREE.DepthTexture(16, 16, type); attached.format = format;
    const target = new THREE.RenderTarget(16, 16, {depthTexture: attached, stencilBuffer}); t.after(() => target.dispose());
    renderer.setRenderTarget(target); renderer.render(scene, camera);
    const depth = new THREE.DepthTexture(16, 16, type); depth.format = format;
    renderer.copyFramebufferToTexture(depth); await renderer.waitForGPU();
    const copy = copies(f.events).at(-1);
    assert.equal(copy.source.texture.format, native); assert.equal(copy.destination.texture.format, native);
    assert.equal(copy.source.aspect, undefined); assert.equal(copy.destination.aspect, undefined);
    assert.equal(depth.type, type); assert.equal(depth.format, format);
  }
});

test('depth snapshots retain queued generations and reject disposed destinations', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; f.mesh(new THREE.MeshBasicMaterial());
  const depth = new THREE.DepthTexture(8, 8);
  for (const size of [8, 16]) {
    const target = new THREE.RenderTarget(size, size); t.after(() => target.dispose());
    renderer.setRenderTarget(target); renderer.render(scene, camera);
    if (size === 16) { depth.image.width = depth.image.height = size; depth.needsUpdate = true; }
    renderer.copyFramebufferToTexture(depth);
  }
  await renderer.waitForGPU();
  const recorded = copies(f.events);
  assert.deepEqual(recorded.map(copy => copy.destination.texture.width), [8, 16]);
  assert.notEqual(recorded[0].destination.texture, recorded[1].destination.texture);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.depth.bindings, 0);
  const second = await fixture(t), target = new THREE.RenderTarget(16, 16), disposed = new THREE.DepthTexture(16, 16);
  t.after(() => target.dispose()); second.renderer.setRenderTarget(target);
  second.mesh(new THREE.MeshBasicMaterial()); second.renderer.render(second.scene, second.camera);
  second.renderer.copyFramebufferToTexture(disposed);
  const owner = second.renderer._getSampledTextureInputs(); disposed.dispose();
  await assert.rejects(second.renderer.waitForGPU(), {code: 'THREE_DEPTH_TEXTURE_STALE'});
  assert.equal(copies(second.events).length, 0);
  assert.equal(owner.diagnostics.depth.bindings, 0);
});

for (const acceptedMip of [0, 1]) test(`deferred depth capture retains accepted mip ${acceptedMip} and restores the public selector`, async t => {
  const f = await fixture(t, {}, false), {renderer, scene, camera} = f;
  const target = new THREE.RenderTarget(16, 16); target.texture.generateMipmaps = true;
  t.after(() => target.dispose()); renderer.setRenderTarget(target, 0, acceptedMip);
  const material = new THREE.MeshBasicNodeMaterial(); material.colorNode = N.vec3(N.viewportDepthTexture().r);
  f.mesh(material);
  const pending = renderer.renderAsync(scene, camera);
  renderer.setRenderTarget(target, 0, 1 - acceptedMip);
  if (acceptedMip === 0) {
    await pending; await renderer.waitForGPU();
    assert.equal(copies(f.events).length, 1);
  } else await assert.rejects(pending, {code: 'F3D_RENDERER_COPY'});
  assert.equal(renderer.getActiveMipmapLevel(), 1 - acceptedMip);
});
