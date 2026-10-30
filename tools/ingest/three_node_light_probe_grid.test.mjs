/** Actual pinned LightProbeGrid construction and retained GPU bake submission.
 * The recording device checks ownership and ordering, not rendered pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {LightProbeGrid, LightProbeGridNode} from './fixtures/three_light_probe_grid.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL;
function graph(t, Material = T.MeshStandardMaterial) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, sortObjects: true,
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1},
    _getSampledTextureInputs: () => ({layoutOf: ({texture}) => ({texture: {
      sampleType: 'float', viewDimension: texture.isData3DTexture ? '3d' : '2d', multisampled: false}})})};
  renderer.library.addLight(LightProbeGridNode, LightProbeGrid);
  const programs = createThreeNodePrograms({three: T, renderer}); t.after(() => programs.dispose());
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  const grid = new LightProbeGrid(4, 6, 8, 2, 3, 4); grid.updateBoundingBox();
  const texture = new T.Data3DTexture(new Uint16Array(2 * 3 * 42 * 4), 2, 3, 42);
  texture.type = T.HalfFloatType; texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.needsUpdate = true; grid.texture = texture; t.after(() => texture.dispose());
  grid.intensity = .75; grid.falloff = 1.25; t.after(() => grid.dispose());
  const scene = new T.Scene(); scene.add(object, grid);
  const camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, lights: [grid], frameId: 1, renderId: 1};
  return {renderer, programs, material, object, grid, scene, camera, context,
    compile: () => programs.compile(material, object, context)};
}
const shader = text => text.replace(/(\/\/ codes\n)([\s\S]*?)(?=@fragment)/, (_, prefix, functions) =>
  prefix + functions.split(/(?=^fn )/m).map(part => part.trim()).filter(Boolean).sort().join('\n\n') + '\n\n');
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture && binding.texture.isData3DTexture).map(binding => binding.texture);
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);

for (const Material of [T.MeshToonMaterial, T.MeshLambertMaterial, T.MeshPhongMaterial, T.MeshStandardMaterial, T.MeshPhysicalMaterial,
  T.MeshToonNodeMaterial, T.MeshLambertNodeMaterial, T.MeshPhongNodeMaterial, T.MeshStandardNodeMaterial, T.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} grid irradiance retains source SH atlas shader and live FRAME uniforms`, t => {
    const f = graph(t, Material), descriptor = f.compile();
    const source = new T.WGSLNodeBuilder(f.object, f.renderer);
    source.material = f.material; source.context.material = f.material;
    source.scene = f.scene; source.camera = f.camera; source.lightsNode = N.lights([f.grid]); source.build();
    assert.equal(descriptor.vertex, source.vertexShader); assert.equal(shader(descriptor.fragment), shader(source.fragmentShader));
    assert.match(descriptor.fragment, /texture_3d<f32>/); assert.match(descriptor.fragment, /smoothstep/);
    const first = f.programs.captureDraw(descriptor, f.context), saved = first.uniforms.map(bytes => bytes.slice());
    assert.deepEqual(sampled(first), [f.grid.texture]); assert.ok(values(first).includes(.75)); assert.ok(values(first).includes(1.25));
    f.grid.intensity = .375; f.grid.falloff = 2.5; f.grid.position.set(1, 2, 3); f.grid.updateBoundingBox();
    const sameFrame = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
    assert.deepEqual(sameFrame.uniforms, saved, 'cached analytic nodes retain source once-per-frame updates');
    const next = f.programs.captureDraw(descriptor, {...f.context, frameId: 2, renderId: 3});
    assert.ok(values(next).includes(.375)); assert.ok(values(next).includes(2.5));
    assert.notDeepEqual(next.uniforms, saved); assert.deepEqual(first.uniforms, saved);
    assert.equal(f.compile(), descriptor, 'live probe bounds, radiance and positive falloff do not rebuild the shader');
  });
}

test('grid texture creation, replacement and falloff branching invalidate source setup safely', t => {
  const f = graph(t), original = f.grid.texture; f.grid.texture = null;
  const empty = f.compile(); assert.doesNotMatch(empty.fragment, /texture_3d/);
  f.grid.texture = original;
  assert.throws(() => f.programs.captureDraw(empty, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const baked = f.compile(); assert.match(baked.fragment, /texture_3d/);
  f.grid.falloff = 0;
  assert.throws(() => f.programs.captureDraw(baked, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const clamped = f.compile(); assert.doesNotMatch(clamped.fragment, /smoothstep/);
  const texture = f.grid.texture.clone(); t.after(() => texture.dispose()); f.grid.texture = texture;
  assert.throws(() => f.programs.captureDraw(clamped, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const replaced = f.compile(); assert.deepEqual(sampled(f.programs.captureDraw(replaced, f.context)), [texture]);
});

function recording(t, classOptions = {}) {
  const device = geometryDevice(), copies = [], targets = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, dimension: texture.dimension, ...options}; }, destroy() { this.destroyed = true; }};
    targets.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const commandEncoder = device.createCommandEncoder;
  device.createCommandEncoder = () => ({...commandEncoder(), copyTextureToTexture(source, destination, size) {
    copies.push({source, destination, size});
  }});
  let configured;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configured.format, usage: 16}); }
  })};
  const Renderer = createWebGPURendererClass(T, classOptions), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose());
  const material = new T.MeshStandardMaterial(), mesh = new T.Mesh(new T.BoxGeometry(), material);
  mesh.frustumCulled = false;
  const scene = new T.Scene(); scene.add(mesh, new T.AmbientLight(0xffffff, .5));
  const grid = new LightProbeGrid(4, 4, 4, 2, 2, 2); scene.add(grid); t.after(() => grid.dispose());
  const camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  return {renderer, device, copies, targets, material, mesh, scene, grid, camera};
}
const draws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const bindings = draw => [...draw.groups.values()].flatMap(({group}) => group.entries).map(entry => entry.resource);

test('retained source bake renders all cube faces, projects SH and packs padded 3D atlas slices before ordinary shading', async t => {
  const f = recording(t); await f.renderer.init();
  const target = new T.RenderTarget(3, 3); t.after(() => target.dispose());
  f.renderer.setRenderTarget(target); f.renderer.autoClear = false; f.renderer.xr.enabled = true;
  const oldMatrix = f.scene.matrixWorldAutoUpdate, oldVisible = f.grid.visible;
  f.grid.bake(f.renderer, f.scene, {cubemapSize: 2, sampleCount: 8});
  assert.equal(f.renderer.getRenderTarget(), target); assert.equal(f.renderer.autoClear, false);
  assert.equal(f.renderer.xr.enabled, true); assert.equal(f.scene.matrixWorldAutoUpdate, oldMatrix);
  assert.equal(f.grid.visible, oldVisible); assert.equal(f.renderer.library.getLightNodeClass(LightProbeGrid), LightProbeGridNode);
  f.renderer.xr.enabled = false; f.renderer.autoClear = true; f.renderer.setRenderTarget(null);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const passes = f.device.passes.filter(pass => pass.draws.length);
  const atlas = passes.filter(pass => pass.desc.colorAttachments[0].view.texture.dimension === '3d');
  assert.equal(atlas.length, 28); assert.deepEqual(atlas.map(pass => pass.desc.colorAttachments[0].depthSlice).sort((a, b) => a - b),
    Array.from({length: 28}, (_, index) => index));
  const texture = atlas[0].desc.colorAttachments[0].view.texture;
  assert.deepEqual([texture.width, texture.height, texture.depthOrArrayLayers, texture.format], [2, 2, 28, 'rgba16float']);
  const cubes = passes.filter(pass => pass.desc.colorAttachments[0].view.texture.depthOrArrayLayers === 6);
  assert.equal(cubes.length, 48);
  assert.equal(passes.filter(pass => pass.desc.colorAttachments[0].view.texture.format === 'rgba32float').length, 8);
  const final = draws(f.device).at(-1); assert.match(final.pipeline.fragment.module.code, /texture_3d<f32>/);
  assert.ok(bindings(final).some(binding => binding?.texture === texture && binding.dimension === '3d'));
  assert.equal(f.grid.texture.image.data, null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('ranged indirect baking snapshots the previous atlas before reusing the live volume', async t => {
  const f = recording(t); await f.renderer.init();
  f.grid.bake(f.renderer, f.scene, {cubemapSize: 2, sampleCount: 8});
  f.grid.bake(f.renderer, f.scene, {cubemapSize: 2, sampleCount: 8, pass: 1, start: 0, count: 3});
  f.grid.bake(f.renderer, f.scene, {cubemapSize: 2, sampleCount: 8, pass: 1, start: 3, count: 5});
  await f.renderer.waitForGPU();
  assert.equal(f.copies.length, 1); const copy = f.copies[0];
  assert.notEqual(copy.source.texture, copy.destination.texture);
  assert.equal(copy.source.texture.dimension, '3d'); assert.equal(copy.destination.texture.dimension, '3d');
  assert.deepEqual(copy.size, [2, 2, 28]);
  assert.equal(f.grid._bounceGrid.parent, null); assert.equal(f.grid.visible, true);
  assert.ok(draws(f.device).some(draw => bindings(draw).some(binding => binding?.texture === copy.destination.texture)));
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('inactive and unlit unbaked grids do not require registration and retain ordinary children', async t => {
  const f = recording(t); await f.renderer.init();
  const child = new T.Mesh(new T.BoxGeometry(.2, .2, .2), new T.MeshBasicMaterial());
  f.grid.add(child); f.grid.layers.set(1); f.renderer.render(f.scene, f.camera);
  f.grid.layers.set(0); f.renderer.lighting.enabled = false; f.renderer.render(f.scene, f.camera);
  f.renderer.lighting.enabled = true; f.mesh.material = new T.MeshBasicMaterial(); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.equal(f.renderer.library.getLightNodeClass(LightProbeGrid), null);
  const frames = f.device.snapshots.map(frame => frame.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main'));
  assert.deepEqual(frames.filter(frame => frame.length).map(frame => frame.length), [2, 2, 2]);
});

test('explicit pending-render limits reject an oversized source bake and restore captured renderer state', async t => {
  for (const maxPendingRenders of [0, -1, Infinity, 1.5])
    assert.throws(() => createWebGPURendererClass(T, {maxPendingRenders}), {code: 'F3D_RENDERER_OPTIONS'});
  const f = recording(t, {maxPendingRenders: 64}); await f.renderer.init();
  assert.throws(() => f.grid.bake(f.renderer, f.scene, {cubemapSize: 2, sampleCount: 8}), {code: 'F3D_RENDERER_LIMIT'});
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.grid.visible, true);
  assert.equal(f.scene.matrixWorldAutoUpdate, true); assert.equal(f.renderer._pending.length, 64);
  await f.renderer.waitForGPU(); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
