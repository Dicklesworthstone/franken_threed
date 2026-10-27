/** Native depth conventions through actual source cameras, WGSL and commands. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import WebGPUPipelineUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUPipelineUtils.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeSceneRaster} from './three_scene_raster.mjs';
import {captureThreeCamera, captureThreeArrayCamera, updateThreeCameraProjection} from './three_pass_snapshot.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {threeBackgroundFrame} from './three_background.mjs';

const Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
function fixture(t, parameters) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture() { return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); }})};
  const renderer = new Renderer({canvas, device, ...parameters});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 1, .125, 1024);
  camera.position.z = 3;
  const mesh = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial({color: 0x6699cc})); scene.add(mesh);
  t.after(() => { mesh.geometry.dispose(); mesh.material.dispose(); });
  return {renderer, device, scene, camera, mesh};
}
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');

test('projection transitions and captured camera views match pinned Renderer._updateCamera', () => {
  for (const reversedDepthBuffer of [false, true]) for (const kind of ['perspective', 'orthographic', 'array']) {
    const make = () => kind === 'orthographic' ? new T.OrthographicCamera(-2, 2, 1, -1, .25, 4096)
      : kind === 'array' ? new T.ArrayCamera([new T.PerspectiveCamera(39, 2, .125, 1024),
        new T.OrthographicCamera(-2, 2, 1, -1, .25, 4096)]) : new T.PerspectiveCamera(39, 2, .125, 1024);
    const actual = make(), expected = make();
    const context = {reversedDepthBuffer, coordinateSystem: T.WebGPUCoordinateSystem, xr: {isPresenting: false}};
    updateThreeCameraProjection(T, context, actual);
    T.Renderer.prototype._updateCamera.call(context, expected, false);
    const views = camera => [camera, ...(camera.cameras ?? [])];
    const state = camera => views(camera).map(value => ({reversed: value.reversedDepth,
      coordinateSystem: value.coordinateSystem, projection: value.projectionMatrix.toArray(), inverse: value.projectionMatrixInverse.toArray()}));
    assert.deepEqual(state(actual), state(expected), `${kind}: reversed=${reversedDepthBuffer}`);
    const copy = kind === 'array' ? captureThreeArrayCamera(T, actual) : captureThreeCamera(T, actual);
    assert.deepEqual(state(copy), state(actual));
    actual.near = 10; actual.far = 20; actual.updateProjectionMatrix();
    assert.deepEqual(state(copy), state(expected), 'accepted projection must survive later source edits');
    context.reversedDepthBuffer = false;
    updateThreeCameraProjection(T, context, actual);
    assert.equal(actual.reversedDepth, reversedDepthBuffer, 'source reversal persists after disabling the renderer flag');
  }
});

test('all native depth comparisons retain the pinned reversed mapping and disabled-depth behavior', () => {
  const material = new T.MeshBasicMaterial(), raster = createThreeSceneRaster(T, {fail(code, message) { throw Error(`${code}: ${message}`); }}).program;
  for (const reversedDepthBuffer of [false, true]) {
    const reference = new WebGPUPipelineUtils({parameters: {reversedDepthBuffer}});
    for (const depthTest of [false, true]) for (let depthFunc = 0; depthFunc < 8; depthFunc++) {
      material.depthTest = depthTest; material.depthFunc = depthFunc;
      assert.equal(raster(material, {reversedDepthBuffer}).depthCompare, reference._getDepthCompare(material));
    }
  }
  material.dispose();
});

for (const logarithmicDepthBuffer of [false, true]) for (const reversedDepthBuffer of [false, true]) {
  if (!logarithmicDepthBuffer && !reversedDepthBuffer) continue;
  test(`native meshes execute source depth mode log=${logarithmicDepthBuffer}, reversed=${reversedDepthBuffer}`, async t => {
    const f = fixture(t, {logarithmicDepthBuffer, reversedDepthBuffer}), r = f.renderer;
    await r.init();
    r.render(f.scene, f.camera);
    r.setClearDepth(.25); f.camera.near = .5; f.camera.far = 8192; f.camera.updateProjectionMatrix();
    r.render(f.scene, f.camera);
    r.setClearDepth(.875); await drain(r);
    const draws = nativeDraws(f.device);
    assert.equal(draws.length, 2);
    assert.ok(draws.every(draw => draw.pipeline.depthStencil.depthCompare === (reversedDepthBuffer ? 'greater-equal' : 'less-equal')));
    const fragment = draws[0].pipeline.fragment.module.code;
    assert.equal(/@builtin\(\s*frag_depth\s*\)/.test(fragment), logarithmicDepthBuffer);
    assert.equal(f.camera.reversedDepth, reversedDepthBuffer);
    assert.equal(f.camera.coordinateSystem, T.WebGPUCoordinateSystem);
    assert.deepEqual(f.device.passes.filter(pass => pass.draws.length).map(pass => pass.desc.depthStencilAttachment.depthClearValue),
      reversedDepthBuffer ? [0, .75] : [1, .25]);
    assert.equal(r.getClearDepth(), T.Renderer.prototype.getClearDepth.call(r));
    assert.equal(r._nodeCaptureBytes, 0);
    assert.equal(r._pending.length, 0);
  });
}

test('logarithmic orthographic draws use source fragment depth and respect material depthNode overrides', async t => {
  const f = fixture(t, {logarithmicDepthBuffer: true});
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .25, 100); camera.position.z = 3;
  await f.renderer.init(); f.renderer.render(f.scene, camera); await drain(f.renderer);
  const first = nativeDraws(f.device)[0].pipeline.fragment.module.code;
  assert.match(first, /@builtin\(\s*frag_depth\s*\)/);
  assert.doesNotMatch(first, /log2\(/);
  const material = new T.MeshBasicNodeMaterial(); material.depthNode = T.TSL.float(.375); f.mesh.material = material;
  t.after(() => material.dispose());
  f.renderer.render(f.scene, camera); await drain(f.renderer);
  const second = nativeDraws(f.device).at(-1).pipeline.fragment.module.code;
  assert.match(second, /frag_depth/); assert.match(second, /0\.375/);
});

test('ArrayCamera node draws keep their explicit admission boundary without submitting a wrong depth mode', async t => {
  const f = fixture(t, {reversedDepthBuffer: true}), a = f.camera;
  a.viewport = new T.Vector4(0, 0, 32, 64);
  const b = new T.PerspectiveCamera(45, .5, .25, 2048); b.position.z = 3; b.viewport = new T.Vector4(32, 0, 32, 64);
  const camera = new T.ArrayCamera([a, b]);
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, camera), {code: 'THREE_NODE_PROGRAM_CAMERA'});
  assert.ok([camera, a, b].every(value => value.reversedDepth && value.coordinateSystem === T.WebGPUCoordinateSystem));
  assert.equal(nativeDraws(f.device).length, 0);
  assert.equal(f.renderer._pending.length, 0);
});

test('source scene passes sample reversed depth through the native output graph', async t => {
  const f = fixture(t, {reversedDepthBuffer: true});
  const pass = T.TSL.pass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer, pass.getLinearDepthNode());
  await f.renderer.init(); pipeline.render(); await drain(f.renderer);
  assert.equal(f.camera.reversedDepth, true);
  assert.ok(nativeDraws(f.device).length >= 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  pipeline.dispose(); pass.dispose();
});

test('reversed projection retains normalized panorama rays and renders an existing directional background', async t => {
  const f = fixture(t, {reversedDepthBuffer: true});
  const pixels = new Float32Array(8 * 4 * 4).fill(.25);
  for (let i = 3; i < pixels.length; i += 4) pixels[i] = 1;
  const panorama = new T.DataTexture(pixels, 8, 4, T.RGBAFormat, T.FloatType);
  panorama.mapping = T.EquirectangularReflectionMapping; panorama.needsUpdate = true;
  f.scene.background = panorama; f.scene.backgroundRotation.set(.25, -.5, .375);
  t.after(() => panorama.dispose());
  f.camera.rotation.set(.125, .25, 0); f.camera.updateMatrixWorld();
  const ordinary = threeBackgroundFrame(f.scene, f.camera, T);
  updateThreeCameraProjection(T, f.renderer, f.camera);
  const reversed = threeBackgroundFrame(f.scene, f.camera, T);
  const ray = (frame, x, y) => new T.Vector4(x, y, .5, 1)
    .applyMatrix4(new T.Matrix4().fromArray(frame.directionFromClip));
  for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) {
    const a = ray(ordinary, x, y), b = ray(reversed, x, y);
    const av = new T.Vector3(a.x, a.y, a.z).normalize(), bv = new T.Vector3(b.x, b.y, b.z).normalize();
    assert.ok(av.distanceTo(bv) < 1e-12);
  }
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length, 1);
  assert.ok(f.device.snapshots.flat().length > 1, 'the directional background has its own draw');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
