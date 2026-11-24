/** Public shader inspection reads the native owner's actual compiled modules.
 * The recording device checks compilation and submission, not rendered pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T);
function fixture(t) {
  const device = geometryDevice(), shaders = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createShaderModule = descriptor => { shaders.push(descriptor.code); return descriptor; };
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configured.format, usage: 16})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  mesh.frustumCulled = false; scene.add(mesh);
  t.after(async () => { await renderer.dispose(); mesh.geometry.dispose(); material.dispose(); });
  return {renderer, device, shaders, scene, camera, material, mesh, canvas};
}
const sourceShader = (f, lights = []) => {
  const builder = new T.WGSLNodeBuilder(f.mesh, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = T.TSL.lights(lights);
  builder.build();
  return {fragmentShader: builder.fragmentShader, vertexShader: builder.vertexShader};
};

test('authored shader inspection initializes, compiles with scene context and returns the exact submitted module sources', async t => {
  const f = fixture(t), calls = [], compile = f.renderer.compileAsync;
  f.material.colorNode = T.TSL.vec3(.125, .25, .5);
  f.renderer.compileAsync = function (...args) { calls.push(args); return compile.apply(this, args); };
  const inspect = f.renderer.debug.getShaderAsync;
  const result = await inspect(f.scene, f.camera, f.mesh);
  assert.deepEqual(calls, [[f.mesh, f.camera, f.scene]]);
  assert.deepEqual(Object.keys(result).sort(), ['fragmentShader', 'vertexShader']);
  assert.deepEqual(result, sourceShader(f));
  assert.ok(f.shaders.includes(result.vertexShader)); assert.ok(f.shaders.includes(result.fragmentShader));
  assert.equal(f.renderer.hasInitialized(), true);
  assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draw = f.device.snapshots.flat().find(draw => draw.pipeline.vertex.entryPoint === 'main');
  assert.equal(draw.pipeline.vertex.module.code, result.vertexShader);
  assert.equal(draw.pipeline.fragment.module.code, result.fragmentShader);
});

test('fixed H1 toon and ordinary materials expose their actual compiled WGSL without a second shader construction', async t => {
  for (const Material of [T.MeshBasicMaterial, T.MeshToonNodeMaterial]) {
    const f = fixture(t); f.mesh.material = new Material(); t.after(() => f.mesh.material.dispose());
    if (Material === T.MeshToonNodeMaterial) f.scene.add(new T.DirectionalLight());
    const result = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
    assert.ok(f.shaders.includes(result.vertexShader)); assert.ok(f.shaders.includes(result.fragmentShader));
    assert.match(result.vertexShader, /@vertex/); assert.match(result.fragmentShader, /@fragment/);
    const count = f.shaders.length;
    assert.deepEqual(await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh), result);
    assert.equal(f.shaders.length, count, 'warm inspection reuses compiled modules');
    assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
  }
});

test('inspection selects the scene override and current target while preserving the public target', async t => {
  const f = fixture(t), override = new T.MeshBasicNodeMaterial(), target = new T.RenderTarget(16, 8);
  t.after(() => { override.dispose(); target.dispose(); });
  f.material.colorNode = T.TSL.vec3(.125, .25, .5);
  override.colorNode = T.TSL.vec3(.75, .5, .25);
  f.scene.overrideMaterial = override; f.renderer.setRenderTarget(target);
  const result = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.equal(f.renderer.getRenderTarget(), target);
  assert.match(result.fragmentShader, /vec3<f32>\( 0\.75, 0\.5, 0\.25 \)/);
  assert.ok(f.shaders.includes(result.vertexShader)); assert.ok(f.shaders.includes(result.fragmentShader));
  assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
  f.scene.overrideMaterial = null;
  const ordinary = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.notEqual(ordinary.fragmentShader, result.fragmentShader);
  assert.equal(f.renderer.getRenderTarget(), target);
});

test('converted canvas inspection returns the object shader rather than the presentation quad', async t => {
  const f = fixture(t); f.renderer.outputColorSpace = T.SRGBColorSpace;
  f.material.colorNode = T.TSL.vec3(.125, .25, .5);
  const result = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.match(result.fragmentShader, /vec3<f32>\( 0\.125, 0\.25, 0\.5 \)/);
  assert.ok(f.shaders.includes(result.vertexShader)); assert.ok(f.shaders.includes(result.fragmentShader));
  assert.equal(f.renderer.getRenderTarget(), null);
  assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
});

test('inspection follows graph invalidation and material disposal without retaining stale shader results', async t => {
  const f = fixture(t);
  f.material.colorNode = T.TSL.vec3(.125, .25, .5);
  const first = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh), count = f.shaders.length;
  first.fragmentShader = 'caller-owned result';
  const warm = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.notEqual(warm.fragmentShader, first.fragmentShader);
  assert.equal(f.shaders.length, count);
  f.material.colorNode = T.TSL.vec3(.75, .5, .25); f.material.needsUpdate = true;
  const changed = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.notEqual(changed.fragmentShader, warm.fragmentShader);
  f.material.dispose();
  const rebuilt = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.deepEqual(rebuilt, changed);
  assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
  await f.renderer.dispose();
  await assert.rejects(f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh), {code: 'F3D_RENDERER_DISPOSED'});
});

test('inspection retains the source public compilation call and propagates its original rejection', async t => {
  const f = fixture(t), failure = new Error('application compilation failure'), calls = [];
  // Read the actual pinned debug closure to establish its public call contract.
  const reference = new T.Renderer({getDomElement: () => f.canvas});
  for (const renderer of [reference, f.renderer]) {
    renderer.compileAsync = async (...args) => { calls.push(args); throw failure; };
    const inspect = renderer.debug.getShaderAsync;
    await assert.rejects(inspect(f.scene, f.camera, f.mesh), error => error === failure);
  }
  assert.deepEqual(calls, [[f.mesh, f.camera, f.scene], [f.mesh, f.camera, f.scene]]);
  assert.equal(f.shaders.length, 0); assert.equal(f.device.submissions.length, 0);
});

test('culled and hidden objects retain shader inspection without changing their render visibility', async t => {
  for (const mode of ['hidden', 'frustum', 'layers']) {
    const f = fixture(t); f.material.colorNode = T.TSL.vec3(.125, .25, .5);
    if (mode === 'hidden') f.mesh.visible = false;
    if (mode === 'frustum') { f.mesh.frustumCulled = true; f.mesh.position.x = 1000; }
    if (mode === 'layers') f.mesh.layers.set(1);
    const state = {visible: f.mesh.visible, frustumCulled: f.mesh.frustumCulled, layers: f.mesh.layers.mask};
    const result = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
    assert.deepEqual(result, sourceShader(f));
    assert.deepEqual({visible: f.mesh.visible, frustumCulled: f.mesh.frustumCulled, layers: f.mesh.layers.mask}, state);
    assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
  }
});

test('transparent double-sided inspection keeps the source default pass without changing material side', async t => {
  for (const visible of [true, false]) {
    const f = fixture(t); f.mesh.visible = visible;
    f.material.colorNode = T.TSL.normalView.mul(.5).add(.5);
    f.material.transparent = true; f.material.side = T.DoubleSide;
    const result = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
    assert.deepEqual(result, sourceShader(f));
    assert.equal(f.material.side, T.DoubleSide);
    assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
  }
});

test('inspection collects each light once when the object is also the scene root', async t => {
  const f = fixture(t), light = new T.DirectionalLight(), material = new T.MeshLambertNodeMaterial();
  t.after(() => material.dispose());
  material.colorNode = T.TSL.vec3(.125, .25, .5); material.visible = false;
  f.mesh.material = material; f.mesh.removeFromParent(); f.mesh.add(light); f.scene = f.mesh;
  const result = await f.renderer.debug.getShaderAsync(f.scene, f.camera, f.mesh);
  assert.deepEqual(result, sourceShader(f, [light]));
  assert.equal(f.device.encodedDrawCalls, 0); assert.equal(f.device.submissions.length, 0);
});

test('inspection retains validated sources when descriptors change during deferred pipeline creation', async t => {
  for (const native of [false, true]) {
    const device = geometryDevice(); device.createShaderModule = descriptor => descriptor;
    const owner = await createGpuAnimationRenderer(device, {maxDraws: 4});
    const geometry = new T.PlaneGeometry(), gpu = createGpuProgramGeometry(device, geometry,
      [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}]);
    t.after(() => { owner.dispose(); gpu.dispose(); geometry.dispose(); });
    let release, started;
    const gate = new Promise(resolve => { release = resolve; });
    const issued = new Promise(resolve => { started = resolve; }), pipelines = [];
    device.createRenderPipelineAsync = descriptor => {
      pipelines.push(descriptor); started(); return gate.then(() => descriptor);
    };
    const vertex = '// captured vertex', fragment = '// captured fragment';
    const program = native ? {nodeProgram: true, vertex, fragment, groups: []}
      : {vertex, fragment, reflection: {textures: [], outputs: [{location: 0}], uniformBufferSize: 16}};
    const key = native ? 'nodeProgram' : 'program';
    const options = {[key]: program, raster: {frontFace: 'ccw', cullMode: 'none', writeMask: 15,
      depthWriteEnabled: true, depthCompare: 'less-equal', blend: null}};
    const pending = owner.addMesh(gpu, options);
    await issued;
    program.vertex = '// changed vertex'; program.fragment = '// changed fragment';
    options[key] = {...program, vertex: '// replaced descriptor'};
    release(); const mesh = await pending;
    assert.deepEqual(owner.getShader(mesh), {fragmentShader: fragment, vertexShader: vertex});
    for (const pipeline of pipelines) {
      assert.equal(pipeline.vertex.module.code, vertex); assert.equal(pipeline.fragment.module.code, fragment);
    }
    assert.equal(device.encodedDrawCalls, 0); assert.equal(device.submissions.length, 0);
  }
});
