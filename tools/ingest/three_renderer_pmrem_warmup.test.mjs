/** Source PMREM warmup compiles without a texture or drawable geometry, and its
 * validated device modules must be the ones real conversion passes consume. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as GL from '../../upstream/three.js/build/three.module.js';
import * as NG from '../../upstream/three.js/build/three.webgpu.js';
import {createPMREMGeneratorClass, createWebGLRendererClass, createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, {native = false, deferred = false} = {}) {
  const T = native ? NG : GL, device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.modules = []; device.textures = []; device.textureWrites = [];
  device.createShaderModule = descriptor => {
    const info = device.compilationInfo?.(descriptor) ?? {messages: []};
    const module = {...descriptor, getCompilationInfo: () => Promise.resolve(info)};
    device.modules.push(module); return module;
  };
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => { device.textureWrites.push({destination, data}); };
  device.destroy = () => { device.destroyed = true; };
  let configuration, release;
  const gpu = {requestAdapter: () => new Promise(resolve => {
    release = () => resolve({features: new Set(), requestDevice: async () => device});
  })};
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }
  })};
  const Renderer = (native ? createWebGPURendererClass : createWebGLRendererClass)(T,
    {maxDraws: 64, maxBindings: 64, ...(deferred ? {gpu} : {})});
  const renderer = new Renderer({canvas, ...(deferred ? {} : {device})});
  const Generator = createPMREMGeneratorClass(T);
  t.after(async () => { renderer.dispose(); await renderer._disposePromise; });
  return {T, renderer, device, Generator, release: () => release()};
}
function panorama() {
  const texture = new GL.DataTexture(new Uint16Array(64 * 32 * 4).fill(0x3800), 64, 32, GL.RGBAFormat, GL.HalfFloatType);
  texture.mapping = GL.EquirectangularReflectionMapping; texture.minFilter = texture.magFilter = GL.LinearFilter;
  texture.needsUpdate = true; return texture;
}
function cube() {
  const texture = new GL.CubeTexture(Array.from({length: 6}, () =>
    new GL.DataTexture(new Uint8Array(16 * 16 * 4).fill(128), 16, 16)));
  texture.needsUpdate = true; return texture;
}
const warmed = device => device.modules.filter(module => module.label?.startsWith('PMREM/warmup/'));

test('both retained conversion shaders warm once without geometry, uploads or submissions', async t => {
  const {renderer, device, Generator} = fixture(t); await renderer.init();
  const before = [device.buffers.length, device.textures.length, device.textureWrites.length, device.submissions.length, device.pipelines.length];
  const generator = new Generator(renderer); t.after(() => generator.dispose());
  assert.equal(generator.compileCubemapShader(), undefined);
  assert.equal(generator.compileEquirectangularShader(), undefined);
  assert.equal(generator.compileCubemapShader(), undefined);
  await renderer.waitForGPU();
  assert.equal(warmed(device).length, 4, 'each retained conversion program compiles only once');
  assert.deepEqual([device.buffers.length, device.textures.length, device.textureWrites.length, device.submissions.length, device.pipelines.length], before);
  assert.deepEqual(device.scopes, []);
});

test('real cube and panorama PMREM draws reuse their warmed GPU module identities', async t => {
  const {renderer, device, Generator} = fixture(t); await renderer.init();
  const generator = new Generator(renderer);
  generator.compileCubemapShader(); generator.compileEquirectangularShader(); await renderer.waitForGPU();
  const modules = new Set(warmed(device)), fragmentModules = [...modules].filter(module => module.label.endsWith('/fragment'));
  const inputs = [cube(), panorama()], outputs = [generator.fromCubemap(inputs[0]), generator.fromEquirectangular(inputs[1])];
  generator.dispose(); await renderer.waitForGPU();
  const conversions = device.snapshots.flat().filter(draw => fragmentModules.includes(draw.pipeline.fragment.module));
  assert.equal(conversions.length, 2);
  for (const draw of conversions) assert.ok(modules.has(draw.pipeline.vertex.module));
  for (const module of modules) assert.equal(device.modules.filter(other => other.code === module.code).length, 1);
  for (const resource of [...inputs, ...outputs]) resource.dispose();
});

test('pre-initialization warmup captures source shaders before generator mutation and disposal', async t => {
  const {renderer, device, Generator, release} = fixture(t, {deferred: true}), generator = new Generator(renderer);
  assert.equal(renderer.hasInitialized(), false);
  generator.compileEquirectangularShader();
  generator._generator._equirectMaterial.fragmentShader = 'invalid later mutation';
  generator.dispose(); release(); await renderer.waitForGPU();
  assert.equal(warmed(device).length, 2);
  assert.ok(warmed(device).every(module => !module.code.includes('invalid later mutation')));
  assert.equal(device.submissions.length, 0); assert.equal(device.textureWrites.length, 0);
});

test('warmup validation errors reach the renderer boundary and failed modules can retry', async t => {
  const {renderer, device, Generator} = fixture(t); await renderer.init();
  device.compilationInfo = descriptor => ({messages: descriptor.label === 'PMREM/warmup/fragment'
    ? [{type: 'error', message: 'conversion compilation failed'}] : []});
  const first = new Generator(renderer); first.compileEquirectangularShader();
  await assert.rejects(renderer.waitForGPU(), /conversion compilation failed/);
  assert.throws(() => renderer.render(new GL.Scene(), new GL.PerspectiveCamera()), /conversion compilation failed/);
  first.dispose(); device.compilationInfo = null;
  const second = new Generator(renderer); second.compileEquirectangularShader(); await renderer.waitForGPU(); second.dispose();
  assert.equal(warmed(device).filter(module => module.label.endsWith('/vertex')).length, 1);
  assert.equal(warmed(device).filter(module => module.label.endsWith('/fragment')).length, 2);
});

test('renderer disposal settles a warmup blocked on GPU compilation', {timeout: 1500}, async t => {
  const {renderer, device, Generator} = fixture(t); await renderer.init();
  let started;
  const compiling = new Promise(resolve => { started = resolve; });
  device.compilationInfo = descriptor => {
    if (!descriptor.label?.startsWith('PMREM/warmup/')) return {messages: []};
    started(); return new Promise(() => {});
  };
  const generator = new Generator(renderer); generator.compileCubemapShader(); await compiling;
  generator.dispose(); renderer.dispose(); await renderer._disposePromise;
  assert.equal(renderer._preparing, null);
});

test('native compile is the source compileAsync alias and prepares an ordinary scene', async t => {
  const {T, renderer, device} = fixture(t, {native: true});
  assert.equal(renderer.compile, renderer.compileAsync);
  const material = new T.NodeMaterial(); material.lights = false; material.fragmentNode = T.TSL.vec4(.2, .4, .8, 1);
  const scene = new T.Scene(); scene.add(new T.Mesh(new T.PlaneGeometry(), material));
  const camera = new T.PerspectiveCamera(); camera.position.z = 3;
  const pending = renderer.compile(scene, camera); assert.equal(typeof pending.then, 'function'); await pending;
  assert.ok(device.pipelines.length > 0); assert.equal(device.submissions.length, 0);
  renderer.render(scene, camera); await renderer.waitForGPU(); assert.ok(device.snapshots.flat().length > 0);
  material.dispose(); scene.children[0].geometry.dispose();
});
