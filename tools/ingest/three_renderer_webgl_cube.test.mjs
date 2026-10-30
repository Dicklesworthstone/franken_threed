/** WebGL-authored cube captures use the existing native target owner. Recorded
 * calls prove source order/lifetimes; browser parity checks the actual pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {createThreeProgramPMREM} from './three_program_pmrem.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, options = {}) {
  const device = geometryDevice(); device.features = new Set(); device.textures = []; device.textureWrites = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => {
    const size = descriptor.size;
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: size.width ?? size[0], height: size.height ?? size[1],
      depthOrArrayLayers: size.depthOrArrayLayers ?? size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.queue.writeTexture = (...args) => device.textureWrites.push(args);
  const owner = createThreeRenderTargets(T, {getDevice: () => device, ...options});
  t.after(() => owner.dispose()); return {device, owner};
}
function cube(t, size = 8, options = {}) {
  const target = new T.WebGLCubeRenderTarget(size, {type: T.HalfFloatType, depthBuffer: false,
    generateMipmaps: true, minFilter: T.LinearMipmapLinearFilter, ...options});
  t.after(() => target.dispose()); return target;
}
async function rendererFixture(t) {
  const f = fixture(t); let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture: () => f.device.createTexture({size: [16, 16], format: configuration.format, usage: 16}),
  })};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device: f.device});
  t.after(() => renderer.dispose()); await renderer.init(); return {...f, renderer};
}
function producer(t) {
  const material = new T.ShaderMaterial({side: T.BackSide,
    vertexShader: 'varying vec3 direction; void main(){direction=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader: 'varying vec3 direction; void main(){gl_FragColor=vec4(normalize(direction)*.4+.5,1.0);}'});
  const geometry = new T.BoxGeometry(4, 4, 4), mesh = new T.Mesh(geometry, material);
  t.after(() => { geometry.dispose(); material.dispose(); }); return mesh;
}

test('WebGL cube resize follows target extents while preserving source face descriptors', t => {
  const {owner, device} = fixture(t), target = cube(t), images = target.texture.image;
  const first = owner.capture(target);
  assert.equal(first.shape.webglCube, true); assert.equal(first.shape.layers, 6);
  assert.equal(first.shape.mipLevelCount, 4); assert.equal(first.shape.colorBytes, 4080);
  assert.equal(device.textures.length, 0); assert.equal(target.texture.renderTarget, null);
  assert.throws(() => owner.render(first, () => {}, {mipLevel: 1}), {code: 'THREE_TARGET_MIP'});
  owner.render(first, () => {}, {face: 3}); target.setSize(16, 16);
  const next = owner.capture(target);
  assert.equal(first.alive, false); assert.notEqual(next, first);
  assert.equal(next.shape.width, 16); assert.equal(next.shape.mipLevelCount, 5);
  assert.equal(target.texture.image, images); assert.ok(images.every(image => image.width === 8));
  target.texture.mipmaps = [{}, {}];
  assert.throws(() => inspectThreeRenderTarget(target, T), {code: 'THREE_TARGET_UNSUPPORTED'});
});

test('unallocated source cube mip levels cannot expose the reserved native pyramid', t => {
  const {owner, device} = fixture(t), target = cube(t, 8, {generateMipmaps: false}), entry = owner.capture(target);
  assert.throws(() => owner.bindingsFor().get(target.texture), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.throws(() => owner.render(entry, () => {}, {face: 0, mipLevel: 1}), {code: 'THREE_TARGET_MIP'});
  assert.equal(device.textures.length, 0);
  target.texture.minFilter = T.LinearFilter;
  assert.equal(owner.bindingsFor().get(target.texture).sampler.lodMaxClamp, 0);
});

test('complete authored WebGL cube chains include base and sample every allocated face and mip', async t => {
  const {owner, device} = fixture(t), target = cube(t, 8, {generateMipmaps: false});
  target.texture.mipmaps = [{}, {}, {}, {}];
  const entry = owner.capture(target);
  assert.equal(entry.shape.authoredMipLevelCount, 4); assert.equal(entry.shape.mipLevelCount, 4);
  assert.equal(entry.shape.colorBytes, 4080); assert.equal(device.textures.length, 0);
  owner.render(entry, attachments => {
    assert.equal(attachments.colorView.baseMipLevel, 2); assert.equal(attachments.colorView.baseArrayLayer, 5);
    assert.equal(attachments.depthView, undefined);
  }, {face: 5, mipLevel: 2});
  const binding = owner.bindingsFor().get(target.texture);
  assert.equal(binding.viewDimension, 'cube'); assert.equal(binding.sampler.lodMaxClamp, 3);
  assert.deepEqual(device.passes.map(pass => [pass.desc.colorAttachments[0].view.baseMipLevel,
    pass.desc.colorAttachments[0].view.baseArrayLayer]),
  [0, 1, 2, 3].flatMap(level => [0, 1, 2, 3, 4, 5].map(face => [level, face])));
  assert.equal(entry.storage.diagnostics.mipGenerations, 0); assert.equal(device.textureWrites.length, 0);
  target.texture.generateMipmaps = true;
  assert.equal(owner.capture(target), entry, 'generation toggles preserve the authored allocation');
  target.texture.minFilter = T.LinearFilter;
  assert.equal(owner.bindingsFor().get(target.texture).sampler.lodMaxClamp, 0);
  await owner.whenIdle();
});

test('CubeCamera queues authored mip faces with physical viewports and captured generation flags', async t => {
  const {renderer, device} = await rendererFixture(t), target = cube(t, 8, {generateMipmaps: false}), mesh = producer(t);
  target.texture.mipmaps = [{}, {}, {}, {}];
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = (...args) => {
    const encoder = createEncoder(...args), begin = encoder.beginRenderPass;
    encoder.beginRenderPass = descriptor => {
      const pass = begin(descriptor), record = device.passes.at(-1);
      return {...pass, setViewport(...viewport) { record.viewport = viewport; pass.setViewport(...viewport); }};
    };
    return encoder;
  };
  const camera = new T.CubeCamera(.1, 10, target), previous = new T.WebGLRenderTarget(4, 4);
  t.after(() => previous.dispose()); renderer.setRenderTarget(previous); renderer.xr.enabled = true;
  target.viewport.set(0, 0, 4, 4); camera.activeMipmapLevel = 1; camera.update(renderer, mesh);
  const entry = renderer._targets.capture(target);
  target.viewport.set(0, 0, 2, 2); camera.activeMipmapLevel = 2; camera.update(renderer, mesh);
  assert.equal(renderer.getRenderTarget(), previous); assert.equal(renderer.getActiveMipmapLevel(), 0);
  assert.equal(renderer.xr.enabled, true); assert.equal(target.texture.generateMipmaps, false);
  target.texture.generateMipmaps = true;
  assert.throws(() => renderer.setRenderTarget(target, 0, 1), {code: 'THREE_TARGET_MIP'});
  assert.equal(renderer.getRenderTarget(), previous);
  await renderer.waitForGPU();
  assert.equal(renderer._targets.capture(target), entry, 'later flags cannot replace queued face storage');
  assert.deepEqual(device.passes.map(pass => [pass.desc.colorAttachments[0].view.baseMipLevel,
    pass.desc.colorAttachments[0].view.baseArrayLayer]),
  [1, 2].flatMap(level => [0, 1, 2, 3, 4, 5].map(face => [level, face])));
  assert.ok(device.passes.every(pass => pass.desc.depthStencilAttachment === undefined));
  assert.ok(device.passes.every(pass => pass.viewport === undefined), 'physical viewport already fills each selected mip');
  assert.equal(entry.storage.diagnostics.mipGenerations, 0); assert.equal(renderer.info.f3d.offscreenRenders, 12);
  assert.equal(device.textureWrites.length, 0);
});

test('cold CubeCamera renders all source faces then generates the full WebGL mip chain', async t => {
  const {renderer, device} = await rendererFixture(t), target = cube(t), mesh = producer(t);
  const camera = new T.CubeCamera(.1, 10, target);
  const previous = new T.WebGLRenderTarget(4, 4); t.after(() => previous.dispose());
  renderer.setRenderTarget(previous); renderer.xr.enabled = true;
  camera.update(renderer, mesh);
  assert.equal(camera.coordinateSystem, T.WebGLCoordinateSystem);
  assert.deepEqual(camera.children.map(child => child.up.toArray()),
    [[0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [0, 1, 0]]);
  assert.equal(renderer.getRenderTarget(), previous); assert.equal(renderer.xr.enabled, true);
  assert.equal(target.texture.pmremVersion, 1); assert.equal(target.texture.generateMipmaps, true);
  target.texture.generateMipmaps = false;
  await renderer.waitForGPU();
  const entry = renderer._targets.capture(target);
  assert.equal(entry.storage.diagnostics.mipLevelCount, 4);
  assert.equal(entry.storage.diagnostics.mipGenerations, 1, 'queued generation uses the source call-time flag');
  const levels = device.passes.map(pass => pass.desc.colorAttachments[0].view.baseMipLevel ?? 0);
  assert.deepEqual(levels, [0, 1, 2, 3].flatMap(level => Array(6).fill(level)));
  assert.deepEqual(device.passes.slice(0, 6).map(pass => pass.desc.colorAttachments[0].view.baseArrayLayer), [0, 1, 2, 3, 4, 5]);
  assert.equal(device.textureWrites.length, 0); assert.equal(renderer.info.f3d.offscreenRenders, 6);
});

test('cube sampling borrows one GPU replica with every source-oriented face and mip', async t => {
  const {owner, device} = fixture(t), target = cube(t), entry = owner.capture(target);
  for (let face = 0; face < 6; face++) owner.render(entry, () => {}, {face, generateMipmaps: face === 5});
  const before = device.passes.length, bindings = owner.bindingsFor(), binding = bindings.get(target.texture);
  assert.equal(binding.viewDimension, 'cube'); assert.equal(binding.view.dimension, 'cube');
  assert.equal(binding.view.texture.depthOrArrayLayers, 6); assert.equal(binding.sampler.lodMaxClamp, 3);
  const copies = device.passes.slice(before);
  assert.equal(copies.length, 24);
  assert.deepEqual(copies.map(pass => [pass.desc.colorAttachments[0].view.baseMipLevel, pass.desc.colorAttachments[0].view.baseArrayLayer]),
    [0, 1, 2, 3].flatMap(level => [0, 1, 2, 3, 4, 5].map(face => [level, face])));
  assert.ok(copies.every(pass => pass.desc.colorAttachments[0].view.arrayLayerCount === 1));
  assert.equal(bindings.get(target.texture).view, binding.view); assert.equal(device.passes.length, before + 24);
  target.texture.minFilter = T.LinearFilter;
  assert.equal(bindings.get(target.texture).sampler.lodMaxClamp, 0, 'GL nonmip filters clamp explicit LOD access');
  assert.throws(() => owner.bindingsFor(target).get(target.texture), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(device.textureWrites.length, 0); assert.equal(target.texture.renderTarget, null);
  await owner.whenIdle();
});

test('borrowed cube sampling generations survive disposal until the program lease ends', async t => {
  const {owner, device} = fixture(t), target = cube(t); owner.capture(target);
  const binding = owner.bindingsFor().get(target.texture), release = binding.retain();
  target.dispose(); assert.throws(binding.validate, {code: 'THREE_TARGET_STALE'});
  assert.ok(device.textures.every(texture => !texture.destroyed));
  release(); await Promise.resolve(); await Promise.resolve();
  assert.ok(device.textures.every(texture => texture.destroyed));
});

test('GPU cube capture feeds source PMREM and regenerates the same atlas without uploads', async t => {
  const {owner, device} = fixture(t), target = cube(t, 32), entry = owner.capture(target);
  const bindings = owner.bindingsFor(), pmrem = createThreeProgramPMREM({three: T, device, bindingOf: texture => bindings.get(texture)});
  t.after(() => pmrem.dispose());
  for (let face = 0; face < 6; face++) owner.render(entry, () => {}, {face, generateMipmaps: face === 5});
  target.texture.needsPMREMUpdate = true;
  const atlas = await pmrem.generate(target.texture), binding = pmrem.binding(atlas);
  assert.equal(atlas.pmremVersion, 1); assert.equal(atlas.mapping, T.CubeUVReflectionMapping);
  owner.render(entry, () => {}, {face: 4, generateMipmaps: true}); target.texture.needsPMREMUpdate = true;
  assert.equal(await pmrem.generate(target.texture), atlas); assert.equal(pmrem.binding(atlas), binding);
  assert.equal(atlas.pmremVersion, 2); assert.equal(device.textureWrites.length, 0);
});
