/** Native graphics use the same versioned array-texture residency as compute. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {inspectThreeTexture} from './three_textures.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function arrayTexture(type = T.UnsignedByteType, value = 1, layers = 3) {
  const Scalar = type === T.FloatType ? Float32Array : type === T.HalfFloatType ? Uint16Array : Uint8Array;
  const texture = new T.DataArrayTexture(new Scalar(2 * 2 * layers * 4).fill(value), 2, 2, layers);
  texture.type = type; texture.needsUpdate = true; return texture;
}
function compiler(t, filterable = false) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    coordinateSystem: T.WebGPUCoordinateSystem, debug: {diagnostics: {keywords: false}},
    depth: true, lighting: {enabled: true}, currentSamples: 1, info: {calls: 1},
    backend: {hasFeature: () => false,
      utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: name => name === 'float32-filterable' && filterable,
    getRenderTarget: () => null, getMRT: () => null, _nodes: {nodeFrame: new T.NodeFrame()},
    _getSampledTextureInputs: () => ({layoutOf({texture}) {
      const shape = inspectThreeTexture(texture, T, {float32Filterable: filterable});
      return {texture: {sampleType: shape.sampleType, viewDimension: shape.viewDimension, multisampled: false}};
    }})};
  const owner = createThreeNodePrograms({three: T, renderer}); t.after(() => owner.dispose());
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  scene.add(object); scene.updateMatrixWorld(); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, renderId: 1, frameId: 1};
  return {renderer, owner, material, object, scene, camera, context,
    compile: () => owner.compile(material, object, context)};
}
const textureLayouts = descriptor => descriptor.groups.flatMap(group => group.entries).filter(entry => entry.texture);

for (const [type, filterable, sampleType] of [[T.UnsignedByteType, false, 'float'],
  [T.HalfFloatType, false, 'float'], [T.FloatType, false, 'unfilterable-float'], [T.FloatType, true, 'float']]) {
  test(`native array sampling retains source WGSL and ${sampleType} bindings for type ${type}, feature ${filterable}`, t => {
    const f = compiler(t, filterable), source = arrayTexture(type);
    f.material.colorNode = N.texture(source, N.vec2(.25, .75)).depth(N.int(2)).rgb;
    const descriptor = f.compile();
    assert.match(descriptor.fragment, /texture_2d_array<f32>/);
    const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
    builder.scene = f.scene; builder.camera = f.camera; builder.build();
    assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
    assert.deepEqual(textureLayouts(descriptor).map(entry => entry.texture),
      [{sampleType, viewDimension: '2d-array', multisampled: false}]);
    const draw = f.owner.captureDraw(descriptor, f.context);
    assert.equal(draw.textures.find(binding => binding.isSampledTexture).texture, source);
  });
}

test('native vertex array loads do not require float32 filtering or sampler bindings', t => {
  const f = compiler(t), source = arrayTexture(T.FloatType, .125);
  f.material.positionNode = N.positionLocal.add(N.textureLoad(source, N.ivec2(0)).depth(N.int(1)).xyz);
  f.material.colorNode = N.vec3(.5);
  const descriptor = f.compile();
  assert.match(descriptor.vertex, /texture_2d_array<f32>/); assert.match(descriptor.vertex, /textureLoad/);
  assert.equal(textureLayouts(descriptor)[0].texture.sampleType, 'unfilterable-float');
  assert.equal(f.owner.captureDraw(descriptor, f.context).textures.some(binding => binding.isSampler && !binding.isSampledTexture), false);
});

test('array replacement reuses compatible programs but a changed view invalidates captured layout', t => {
  const f = compiler(t), source = arrayTexture(), node = N.texture(source, N.vec2(.5)).depth(N.int(1));
  f.material.colorNode = node.rgb;
  const first = f.compile(), replacement = arrayTexture(T.UnsignedByteType, 7, 4);
  node.value = replacement;
  assert.equal(f.compile(), first);
  assert.equal(f.owner.captureDraw(first, f.context).textures.find(binding => binding.isSampledTexture).texture, replacement);
  const flat = new T.DataTexture(new Uint8Array(16), 2, 2); flat.needsUpdate = true; node.value = flat;
  assert.throws(() => f.owner.captureDraw(first, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});

test('array admission keeps actual source classes, nearest float32 sampling and layer-view agreement', t => {
  const f = compiler(t), source = arrayTexture(T.FloatType);
  source.minFilter = source.magFilter = T.LinearFilter;
  f.material.colorNode = N.texture(source).depth(N.int(0)).rgb;
  assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  const forged = new T.DataTexture(new Uint8Array(16), 2, 2); forged.isDataArrayTexture = true;
  f.material.colorNode = N.texture(forged).depth(N.int(0)).rgb;
  assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});

function recording(t) {
  const device = geometryDevice(); device.features = new Set(); device.textureWrites = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data, layout, size) => device.textureWrites.push({
    texture: destination.texture, origin: destination.origin, size, layout,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const material = new T.MeshBasicNodeMaterial(), scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.z = 4; scene.add(new T.Mesh(new T.PlaneGeometry(), material));
  return {device, renderer, material, scene, camera};
}
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);

test('queued native array draws preserve partial layer histories instead of unmarked CPU writes', async t => {
  const f = recording(t); await f.renderer.init();
  const source = arrayTexture();
  f.material.colorNode = N.texture(source, N.vec2(.5)).depth(N.int(1)).rgb.mul(N.uniform(1).setName('nativeArrayHistory'));
  f.renderer.render(f.scene, f.camera);
  source.image.data.fill(9); source.addLayerUpdate(1); source.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  source.image.data.fill(99);
  while (f.renderer._drain) await f.renderer._drain;
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.fragment.module.code.includes('nativeArrayHistory'));
  assert.equal(draws.length, 2);
  const first = views(draws[0]).find(view => view.dimension === '2d-array');
  const second = views(draws[1]).find(view => view.dimension === '2d-array');
  assert.ok(first); assert.ok(second); assert.notEqual(first.texture, second.texture);
  assert.equal(first.texture.depthOrArrayLayers, 3);
  const a = f.device.textureWrites.find(write => write.texture === first.texture).bytes;
  const b = f.device.textureWrites.find(write => write.texture === second.texture).bytes;
  assert.deepEqual([...a], Array(48).fill(1));
  assert.deepEqual([...b], [...Array(16).fill(1), ...Array(16).fill(9), ...Array(16).fill(1)]);
  assert.equal(source.layerUpdates.size, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('actual native graphics bind nearest float32 arrays without the optional filtering feature', async t => {
  const f = recording(t); await f.renderer.init();
  const source = arrayTexture(T.FloatType, .125);
  f.material.colorNode = N.textureLoad(source, N.ivec2(0)).depth(N.int(2)).rgb.mul(N.uniform(1).setName('nativeFloatArray'));
  f.renderer.render(f.scene, f.camera);
  while (f.renderer._drain) await f.renderer._drain;
  const draw = f.device.snapshots.flat().find(value => value.pipeline.fragment.module.code.includes('nativeFloatArray'));
  assert.ok(draw);
  assert.ok([...draw.groups.values()].some(({group}) => group.layout.entries.some(entry =>
    entry.texture?.sampleType === 'unfilterable-float' && entry.texture.viewDimension === '2d-array')));
  const view = views(draw).find(value => value.dimension === '2d-array');
  assert.equal(view.texture.format, 'rgba32float');
  const bytes = f.device.textureWrites.find(write => write.texture === view.texture).bytes;
  assert.ok([...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)].every(value => value === .125));
});
