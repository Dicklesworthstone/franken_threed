/** Compressed source textures through native graphics and the shared upload owner.
 * The recording device checks source WGSL, bindings and captured mip histories;
 * it does not execute WGSL or establish browser pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, features = ['texture-compression-bc']) {
  const device = geometryDevice(), textures = [], writes = [], captures = [];
  device.features = new Set(features);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const value = {dimension: '2d', sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(value); return value;
  };
  device.createSampler = desc => ({...desc});
  device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice(),
    layout: {...layout}, size: [...size]});
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(); scene.add(object);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); });
  return {renderer, device, material, object, scene, camera, textures, writes, captures};
}
function compressed(t, {kind = '2d', width = 8, height = 8, layers = 3, levels = 3,
  format = T.RGBA_S3TC_DXT1_Format, type = T.UnsignedByteType, blockWidth = 4, blockHeight = 4,
  blockBytes = 8, colorSpace = T.NoColorSpace} = {}) {
  const chain = face => Array.from({length: levels}, (_, level) => {
    const w = Math.max(1, width >> level), h = Math.max(1, height >> level);
    const length = Math.ceil(w / blockWidth) * Math.ceil(h / blockHeight) * blockBytes * (kind === 'array' ? layers : 1);
    const data = new Uint8Array(length + 8).subarray(4, length + 4); data.fill(10 + face + level);
    return {width: w, height: h, data};
  });
  const value = kind === 'cube'
    ? new T.CompressedCubeTexture(Array.from({length: 6}, (_, face) => ({width, height, mipmaps: chain(face)})), format, type)
    : kind === 'array' ? new T.CompressedArrayTexture(chain(0), width, height, layers, format, type)
      : new T.CompressedTexture(chain(0), width, height, format, type);
  value.colorSpace = colorSpace; value.needsUpdate = true; t.after(() => value.dispose()); return value;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const sampledView = draw => entries(draw).find(entry => entry.resource?.texture).resource;
const layouts = draw => draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries);
const uploaded = (f, texture) => f.writes.filter(write => write.destination.texture === texture);
function sample(f, texture) {
  const node = texture.isCubeTexture ? N.cubeTexture(texture, N.vec3(1, .5, .25))
    : texture.isCompressedArrayTexture ? N.texture(texture, N.vec2(.25, .75)).depth(1)
      : N.texture(texture, N.vec2(.25, .75));
  f.material.colorNode = node.rgb; return node;
}
function assertSource(f, descriptor) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(builder.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(builder.fragmentShader));
}

for (const profile of [
  {name: 'BC1 sRGB 2D', colorSpace: T.SRGBColorSpace, native: 'bc1-rgba-unorm'},
  {name: 'ETC2 array', kind: 'array', format: T.RGBA_ETC2_EAC_Format, blockBytes: 16,
    features: ['texture-compression-etc2'], native: 'etc2-rgba8unorm'},
  {name: 'ASTC cube', kind: 'cube', format: T.RGBA_ASTC_4x4_Format, blockBytes: 16,
    features: ['texture-compression-astc'], native: 'astc-4x4-unorm'},
  {name: 'BC6H half HDR', format: T.RGB_BPTC_UNSIGNED_Format, type: T.HalfFloatType, blockBytes: 16, native: 'bc6h-rgb-ufloat'},
  {name: 'BC6H float HDR', format: T.RGB_BPTC_SIGNED_Format, type: T.FloatType, blockBytes: 16, native: 'bc6h-rgb-float'},
  {name: 'RGBA8 loader fallback', format: T.RGBAFormat, blockWidth: 1, blockHeight: 1, blockBytes: 4,
    features: [], native: 'rgba8unorm'},
]) test(`native ${profile.name} preserves source WGSL, authored mips and shared layout`, async t => {
  const f = fixture(t, profile.features), texture = compressed(t, profile); sample(f, texture);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0], view = sampledView(draw), dimension = profile.kind === 'array' ? '2d-array' : profile.kind ?? '2d';
  assertSource(f, f.captures[0].draws[0].binding.nodeProgram);
  assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
    [{sampleType: 'float', viewDimension: dimension, multisampled: false}]);
  assert.equal(view.texture.format, profile.native);
  assert.equal(view.format, profile.native + (profile.colorSpace === T.SRGBColorSpace ? '-srgb' : ''));
  assert.equal(view.texture.usage & 7, 7); assert.equal(view.texture.mipLevelCount, 3);
  const chains = texture.isCubeTexture ? texture.image.map(face => face.mipmaps) : [texture.mipmaps];
  const writes = uploaded(f, view.texture); assert.equal(writes.length, chains.length * 3);
  for (const [face, chain] of chains.entries()) for (const [level, mip] of chain.entries()) {
    const write = writes.find(value => value.destination.mipLevel === level && value.destination.origin[2] === face);
    assert.deepEqual(write.bytes, mip.data);
    assert.deepEqual(write.size, [Math.ceil(mip.width / (profile.blockWidth ?? 4)) * (profile.blockWidth ?? 4),
      Math.ceil(mip.height / (profile.blockHeight ?? 4)) * (profile.blockHeight ?? 4), texture.isCompressedArrayTexture ? 3 : 1]);
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('compressed material maps with authored opacity retain source UV and color conversion', async t => {
  const f = fixture(t), texture = compressed(t, {colorSpace: T.SRGBColorSpace}); f.material.map = texture;
  f.material.opacityNode = N.uniform(.5);
  texture.offset.set(.125, .25); texture.repeat.set(2, 3);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0]; assertSource(f, f.captures[0].draws[0].binding.nodeProgram);
  assert.equal(sampledView(draw).format, 'bc1-rgba-unorm-srgb');
  assert.match(draw.pipeline.fragment.module.code, /textureSample/);
});

test('queued compressed mip and sampler captures preserve acknowledged bytes and source identity', async t => {
  const f = fixture(t), texture = compressed(t), updates = [], image = texture.image, source = texture.source, mips = texture.mipmaps;
  texture.onUpdate = value => updates.push(value); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  for (const mip of mips) mip.data.fill(53);
  f.renderer.render(f.scene, f.camera);
  texture.wrapS = T.RepeatWrapping;
  f.renderer.render(f.scene, f.camera);
  for (const mip of mips) mip.data.fill(91); texture.needsUpdate = true; texture.wrapS = T.MirroredRepeatWrapping;
  f.renderer.render(f.scene, f.camera);
  for (const mip of mips) mip.data.fill(127); texture.wrapS = T.ClampToEdgeWrapping;
  await f.renderer.waitForGPU();
  const rendered = draws(f), natives = rendered.map(draw => sampledView(draw).texture);
  assert.equal(rendered.length, 4); assert.equal(natives[0], natives[1]); assert.notEqual(natives[1], natives[2]);
  for (let i = 0; i < 4; i++) for (const write of uploaded(f, natives[i]))
    assert.ok(write.bytes.every(value => value === (i === 3 ? 91 : i === 2 ? 53 : 10 + write.destination.mipLevel)));
  assert.deepEqual(rendered.map(draw => entries(draw).find(entry => entry.resource?.addressModeU).resource.addressModeU),
    ['clamp-to-edge', 'clamp-to-edge', 'repeat', 'mirror-repeat']);
  assert.deepEqual(updates, [texture, texture, texture]);
  assert.equal(texture.source, source); assert.equal(texture.image, image); assert.equal(texture.mipmaps, mips);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued compressed cube faces retain every captured mip without a top-level mipmaps array', async t => {
  const f = fixture(t), texture = compressed(t, {kind: 'cube'}), image = texture.image;
  sample(f, texture); assert.equal(texture.mipmaps, undefined); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  for (const face of image) for (const mip of face.mipmaps) mip.data.fill(73);
  texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  for (const face of image) for (const mip of face.mipmaps) mip.data.fill(99);
  await f.renderer.waitForGPU();
  const [a, b] = draws(f).map(draw => sampledView(draw).texture); assert.notEqual(a, b);
  for (const write of uploaded(f, a))
    assert.ok(write.bytes.every(value => value === 10 + write.destination.origin[2] + write.destination.mipLevel));
  assert.equal(uploaded(f, a).length, 18); assert.equal(uploaded(f, b).length, 18);
  assert.ok(uploaded(f, b).every(write => write.bytes.every(value => value === 73)));
  assert.equal(texture.image, image); assert.equal(texture.mipmaps, undefined); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('compressed HDR type changes rebuild source filtering code while retaining the float GPU layout', async t => {
  const f = fixture(t), texture = compressed(t, {format: T.RGB_BPTC_UNSIGNED_Format, type: T.FloatType, blockBytes: 16});
  sample(f, texture); await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  texture.type = T.HalfFloatType; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const [a, b] = f.captures.map(frame => frame.draws[0].binding.nodeProgram); assert.notEqual(a, b);
  assert.match(a.fragment, /textureLoad/); assert.match(b.fragment, /textureSample/);
  assert.ok(draws(f).every(draw => layouts(draw).filter(entry => entry.texture).every(entry => entry.texture.sampleType === 'float')));
  assertSource(f, b);
});

test('queued compressed array updates replace marked layers in every mip without exposing stale CPU edits', async t => {
  const f = fixture(t), texture = compressed(t, {kind: 'array'}); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  for (const mip of texture.mipmaps) mip.data.fill(71);
  texture.addLayerUpdate(1); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  for (const mip of texture.mipmaps) mip.data.fill(99);
  await f.renderer.waitForGPU();
  const rendered = draws(f), first = sampledView(rendered[0]).texture, second = sampledView(rendered[1]).texture;
  assert.notEqual(first, second);
  for (const write of uploaded(f, second)) {
    const slice = write.bytes.length / 3, old = 10 + write.destination.mipLevel;
    assert.deepEqual([...write.bytes], [...Array(slice).fill(old), ...Array(slice).fill(71), ...Array(slice).fill(old)]);
  }
  assert.equal(texture.layerUpdates.size, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('compatible compressed replacements share programs while array family changes rebuild the view', async t => {
  const f = fixture(t), a = compressed(t), b = compressed(t, {width: 16});
  const node = sample(f, a); await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  node.value = b; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.captures[0].draws[0].binding.nodeProgram, f.captures[1].draws[0].binding.nodeProgram);
  assert.deepEqual(draws(f).map(draw => sampledView(draw).texture.width), [8, 16]);
  const array = compressed(t, {kind: 'array'}); node.value = array; node.depthNode = N.int(1);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.notEqual(f.captures[1].draws[0].binding.nodeProgram, f.captures[2].draws[0].binding.nodeProgram);
  assert.equal(sampledView(draws(f)[2]).dimension, '2d-array');
  assert.match(draws(f)[2].pipeline.fragment.module.code, /texture_2d_array<f32>/);
});

for (const [name, change, code] of [
  ['missing compression feature', () => {}, 'THREE_TEXTURE_FEATURE'],
  ['invalid mip block bytes', texture => { texture.mipmaps[1].data = new Uint8Array(1); }, 'THREE_TEXTURE_STORAGE'],
  ['unsupported HDR color space', texture => { texture.format = T.RGB_BPTC_UNSIGNED_Format; texture.colorSpace = T.SRGBColorSpace; }, 'THREE_TEXTURE_COLOR'],
  ['forged compressed array class', texture => { texture.isCompressedArrayTexture = true; }, 'THREE_NODE_PROGRAM_TEXTURE'],
]) test(`native compressed admission refuses ${name} before texture allocation`, async t => {
  const f = fixture(t, name === 'missing compression feature' ? [] : ['texture-compression-bc']), texture = compressed(t);
  change(texture); sample(f, texture); await f.renderer.init(); const allocations = f.textures.length;
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code});
  assert.equal(f.textures.length, allocations); assert.equal(f.writes.length, 0); assert.equal(draws(f).length, 0);
});
