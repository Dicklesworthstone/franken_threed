/** Source PMREM construction with GPU render-target inputs; command recording
 * proves ownership/order. Browser parity is a separate software-GPU check. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLEnvironments} from '../../upstream/three.js/src/renderers/webgl/WebGLEnvironments.js';
import {createThreeProgramPMREM} from './three_program_pmrem.mjs';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function device() {
  const d = geometryDevice(); d.textures = []; d.textureWrites = [];
  d.limits.maxTextureDimension2D = 8192;
  d.createTexture = descriptor => {
    const size = descriptor.size;
    const texture = {...descriptor, width: size.width ?? size[0], height: size.height ?? size[1],
      destroyed: false, createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    d.textures.push(texture); return texture;
  };
  d.createSampler = descriptor => ({...descriptor});
  d.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  d.queue.writeTexture = (...args) => d.textureWrites.push(args);
  return d;
}
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
function fixture(t) {
  const d = device(), targets = createThreeRenderTargets(T, {getDevice: () => d});
  const source = new T.WebGLRenderTarget(64, 32, {type: T.HalfFloatType, depthBuffer: false});
  source.texture.mapping = T.EquirectangularReflectionMapping;
  const bindings = targets.bindingsFor(), calls = [];
  const owner = createThreeProgramPMREM({three: T, device: d, bindingOf(texture) { calls.push(texture); return bindings.get(texture); }});
  t.after(() => { owner.dispose(); source.dispose(); targets.dispose(); });
  return {d, targets, source, owner, calls};
}
const pmremPasses = d => d.passes.filter(p => p.desc.colorAttachments[0]?.view.texture.label?.startsWith('f3d-pmrem/'));

test('GPU equirectangular sources filter without CPU uploads and regenerate into the same atlas', async t => {
  const {d, source, owner, calls} = fixture(t);
  source.texture.needsPMREMUpdate = true;
  assert.deepEqual(owner.lookup(source.texture), {state: 'needed'});
  const first = await owner.generate(source.texture), binding = owner.binding(first);
  assert.equal(first.pmremVersion, 1); assert.equal(first.mapping, T.CubeUVReflectionMapping);
  assert.deepEqual(owner.lookup(source.texture), {state: 'ready', texture: first});
  assert.equal(calls.length, 1, 'one borrowed source generation is shared by the conversion draws');
  assert.equal(d.textureWrites.length, 0, 'GPU source pixels never pass through writeTexture');
  const before = pmremPasses(d).length, allocations = d.textures.length;
  source.texture.needsUpdate = true;
  assert.deepEqual(owner.lookup(source.texture), {state: 'ready', texture: first}, 'upload versions do not invalidate PMREM');
  source.texture.needsPMREMUpdate = true;
  assert.deepEqual(owner.lookup(source.texture), {state: 'needed'});
  assert.equal(await owner.generate(source.texture), first);
  assert.equal(owner.binding(first), binding); assert.equal(first.pmremVersion, 2);
  assert.equal(pmremPasses(d).length, before * 2); assert.equal(d.textures.length, allocations);
  assert.equal(d.textureWrites.length, 0);
});

test('ordinary source textures retain their initial bake when their versions change', async t => {
  const d = device(), source = new T.DataTexture(new Uint16Array(64 * 32 * 4), 64, 32, T.RGBAFormat, T.HalfFloatType);
  source.mapping = T.EquirectangularReflectionMapping; source.needsUpdate = true;
  // Supplying an acknowledged GPU binding isolates the cache contract.
  const texture = d.createTexture({size: [64, 32], format: 'rgba16float', usage: 4});
  const plain = createThreeProgramPMREM({three: T, device: d,
    bindingOf: () => ({view: texture.createView(), sampler: d.createSampler({})})});
  t.after(() => plain.dispose());
  const baked = await plain.generate(source);
  source.needsUpdate = true; source.needsPMREMUpdate = true;
  assert.deepEqual(plain.lookup(source), {state: 'ready', texture: baked});
});

for (const initialVersion of [0, 1]) test(`source-disposal cache behavior follows the first PMREM branch (version ${initialVersion})`, async t => {
  const {source, owner} = fixture(t);
  let target = null;
  const reference = WebGLEnvironments({isWebGLRenderer: true, autoClear: true, toneMapping: T.NoToneMapping,
    xr: {enabled: false}, getRenderTarget: () => target, getActiveCubeFace: () => 0, getActiveMipmapLevel: () => 0,
    setRenderTarget(value) { target = value; }, render() {}, compile() {}});
  t.after(() => reference.dispose());
  source.texture.pmremVersion = initialVersion;
  const expected = reference.get(source.texture, true), actual = await owner.generate(source.texture);
  source.texture.dispose();
  const after = reference.get(source.texture, true);
  if (after === expected) {
    assert.equal(initialVersion, 1);
    assert.deepEqual(owner.lookup(source.texture), {state: 'ready', texture: actual});
  } else {
    assert.equal(initialVersion, 0);
    assert.deepEqual(owner.lookup(source.texture), {state: 'needed'});
  }
});

test('a version change during pipeline preparation does not publish the later version', async t => {
  const {d, source, owner} = fixture(t), gate = deferred(), started = deferred();
  const compile = d.createRenderPipelineAsync;
  let blocked = false;
  d.createRenderPipelineAsync = async descriptor => {
    if (!blocked) { blocked = true; started.resolve(); await gate.promise; }
    return compile(descriptor);
  };
  source.texture.needsPMREMUpdate = true;
  const work = owner.generate(source.texture); await started.promise;
  source.texture.needsPMREMUpdate = true; gate.resolve();
  const baked = await work;
  assert.equal(baked.pmremVersion, 1); assert.deepEqual(owner.lookup(source.texture), {state: 'needed'});
  await owner.generate(source.texture); assert.equal(baked.pmremVersion, 2);
});

test('GPU input leases survive asynchronous compilation and reject disposed sources before submission', async t => {
  const {d, source, owner} = fixture(t), gate = deferred(), started = deferred();
  const compile = d.createRenderPipelineAsync;
  let blocked = false;
  d.createRenderPipelineAsync = async descriptor => {
    if (!blocked) { blocked = true; started.resolve(); await gate.promise; }
    return compile(descriptor);
  };
  const work = owner.generate(source.texture); await started.promise;
  const inputs = d.textures.filter(texture => !texture.label?.startsWith('f3d-pmrem/'));
  source.dispose(); assert.ok(inputs.length > 0); assert.ok(inputs.every(texture => !texture.destroyed));
  gate.resolve(); await assert.rejects(work, {code: 'THREE_TARGET_STALE'});
  assert.equal(d.submissions.length, 1, 'only the earlier UV-orientation copy was submitted');
  assert.ok(inputs.every(texture => texture.destroyed));
});

test('submitted PMREM reads retain GPU source storage until their queue fence settles', async t => {
  const {d, source, owner} = fixture(t), completion = deferred();
  d.completion = completion.promise;
  await owner.generate(source.texture);
  const inputs = d.textures.filter(texture => !texture.label?.startsWith('f3d-pmrem/'));
  source.dispose(); assert.ok(inputs.every(texture => !texture.destroyed));
  completion.resolve(); await Promise.resolve(); await Promise.resolve();
  assert.ok(inputs.every(texture => texture.destroyed));
});

test('disposing a GPU source during validation cannot publish a fresh filtered cache entry', async t => {
  const {d, source, owner} = fixture(t), validation = deferred(), started = deferred();
  const pop = d.popErrorScope, submit = d.queue.submit; let filteringSubmitted = false;
  d.queue.submit = commands => { submit(commands); if (commands[0].length > 1) filteringSubmitted = true; };
  d.popErrorScope = () => {
    const result = pop();
    if (filteringSubmitted) { started.resolve(); return validation.promise.then(() => result); }
    return result;
  };
  const work = owner.generate(source.texture); await started.promise;
  source.dispose(); validation.resolve();
  await assert.rejects(work, {code: 'THREE_TARGET_STALE'});
  assert.deepEqual(owner.lookup(source.texture), {state: 'needed'});
});

test('the public WebGL-authored renderer regenerates GPU environments and accepts authored cube-UV atlases', async t => {
  const d = device(); let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configuration = value; },
    unconfigure() {}, getCurrentTexture: () => d.createTexture({size: [32, 32], format: configuration.format, usage: 16})})};
  const Renderer = createWebGLRendererClass(T, {maxDraws: 32, maxBindings: 32}), renderer = new Renderer({canvas, device: d});
  const source = new T.WebGLRenderTarget(64, 32, {type: T.HalfFloatType, depthBuffer: false});
  source.texture.mapping = T.EquirectangularReflectionMapping;
  t.after(() => { source.dispose(); renderer.dispose(); }); await renderer.init();
  const scene = new T.Scene(), material = new T.MeshStandardMaterial({envMap: source.texture, metalness: 1, roughness: .4});
  scene.add(new T.Mesh(new T.SphereGeometry(1, 8, 6), material));
  const camera = new T.PerspectiveCamera(45, 1, .1, 10); camera.position.z = 4;
  renderer.render(scene, camera); await renderer.waitForGPU();
  const first = pmremPasses(d).length; assert.ok(first > 0);
  renderer.render(scene, camera); await renderer.waitForGPU(); assert.equal(pmremPasses(d).length, first);
  source.texture.needsPMREMUpdate = true;
  renderer.render(scene, camera); await renderer.waitForGPU(); assert.equal(pmremPasses(d).length, first * 2);
  const atlas = new T.DataTexture(new Uint16Array(336 * 64 * 4).fill(0x3800), 336, 64, T.RGBAFormat, T.HalfFloatType);
  atlas.mapping = T.CubeUVReflectionMapping; atlas.minFilter = atlas.magFilter = T.LinearFilter; atlas.needsUpdate = true;
  t.after(() => atlas.dispose()); material.envMap = atlas;
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(pmremPasses(d).length, first * 2, 'authored atlases are sampled directly');
  assert.equal(renderer.info.f3d.presentedRenders, 4); assert.equal(renderer.info.f3d.route, 'general-webgpu');
});
