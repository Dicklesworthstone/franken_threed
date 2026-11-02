/** Source clip-control depth semantics, actual facade packets and camera effects. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {ReversedDepthFuncs} from '../../upstream/three.js/src/utils.js';
import {createThreeProgramSupport, threeProgramRaster} from './three_program.mjs';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {captureThreeCamera} from './three_pass_snapshot.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const supportFor = state => createThreeProgramSupport({three: T,
  state: () => ({toneMapping: T.NoToneMapping, outputColorSpace: T.LinearSRGBColorSpace, ...state})});
const shader = raw => new (raw ? T.RawShaderMaterial : T.ShaderMaterial)({
  vertexShader: (raw ? 'precision highp float; attribute vec3 position; uniform mat4 projectionMatrix, modelViewMatrix;\n' : '') +
    'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader: (raw ? 'precision highp float;\n' : '') + 'void main(){gl_FragColor=vec4(1.0);}',
});

test('reversed raster comparisons follow every pinned ReversedDepthFuncs entry', () => {
  const material = new T.MeshBasicMaterial();
  for (const [depth, reversed] of Object.entries(ReversedDepthFuncs)) {
    material.depthFunc = Number(depth);
    const actual = threeProgramRaster(T, material, {reversedDepthBuffer: true});
    material.depthFunc = reversed;
    assert.equal(actual.depthCompare, threeProgramRaster(T, material).depthCompare, depth);
  }
  material.depthTest = false; material.depthWrite = true;
  const disabled = threeProgramRaster(T, material, {reversedDepthBuffer: true});
  assert.equal(disabled.depthCompare, 'always'); assert.equal(disabled.depthWriteEnabled, false);
  material.polygonOffset = true; material.polygonOffsetFactor = 2; material.polygonOffsetUnits = 3;
  for (const reversedDepthBuffer of [false, true]) {
    const raster = threeProgramRaster(T, material, {reversedDepthBuffer});
    assert.equal(raster.depthBiasSlopeScale, reversedDepthBuffer ? -2 : 2);
    assert.equal(raster.depthBias, 3, 'WebGLState reverses only the slope factor');
  }
  material.dispose();
});

test('raw and built-in programs distinguish GL and zero-to-one clip conventions in their cache', () => {
  const state = {reversedDepthBuffer: false}, support = supportFor(state);
  for (const material of [new T.MeshBasicMaterial(), shader(false), shader(true)]) {
    const mesh = new T.Mesh(new T.PlaneGeometry(), material);
    const normal = support.compile(material, mesh);
    assert.match(normal.program.vertex, /\(f3d_Position.z \+ f3d_Position.w\) \* 0.5/);
    state.reversedDepthBuffer = true;
    const reversed = support.compile(material, mesh);
    assert.notEqual(normal, reversed);
    assert.match(reversed.program.vertex, /out.f3d_position = f3d_Position;/);
    assert.equal(support.compile(material, mesh), reversed);
    state.reversedDepthBuffer = false;
    assert.equal(support.compile(material, mesh), normal);
    mesh.geometry.dispose(); material.dispose();
  }
});

test('point centers are clipped against the source zero-to-one near plane', () => {
  const support = supportFor({reversedDepthBuffer: true});
  const points = new T.Points(new T.BufferGeometry(), new T.PointsMaterial());
  points.geometry.setAttribute('position', new T.Float32BufferAttribute([0, 0, 0], 3));
  assert.match(support.compile(points.material, points).program.vertex, /f3d_p.z >= 0.0 && f3d_p.z <= f3d_p.w/);
  points.geometry.dispose(); points.material.dispose();
});

function deviceAndCanvas() {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); }})};
  return {device, canvas};
}

test('actual canvas and target draws retain first-draw camera transitions and reversed depth packets', async t => {
  const {device, canvas} = deviceAndCanvas(), Renderer = createWebGLRendererClass(T);
  const renderer = new Renderer({canvas, device, reversedDepthBuffer: true});
  t.after(() => renderer.dispose()); await renderer.init();
  assert.equal(renderer.capabilities.reversedDepthBuffer, true);
  assert.equal(renderer.state.buffers.depth.getReversed(), true);
  const target = new T.WebGLRenderTarget(32, 32), scene = new T.Scene();
  const mesh = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial()); mesh.position.z = -3; scene.add(mesh);
  t.after(() => { target.dispose(); mesh.geometry.dispose(); mesh.material.dispose(); });
  for (const destination of [null, target]) {
    const camera = new T.PerspectiveCamera(45, 1, .1, 1e9), trace = [];
    mesh.onBeforeRender = () => trace.push(['before', camera.reversedDepth]);
    mesh.onAfterRender = () => trace.push(['after', camera.reversedDepth]);
    renderer.setRenderTarget(destination); await renderer.renderAsync(scene, camera);
    assert.deepEqual(trace, [['before', false], ['after', true]]);
    assert.equal(camera.coordinateSystem, T.WebGLCoordinateSystem);
    assert.ok(camera.projectionMatrix.equals(new T.PerspectiveCamera(45, 1, .1, 1e9).projectionMatrix) === false);
  }
  assert.equal(device.snapshots.flat().length, 2);
  assert.ok(device.snapshots.flat().every(draw => draw.pipeline.depthStencil.depthCompare === 'greater-equal'));
  assert.ok(device.passes.every(pass => !pass.desc.depthStencilAttachment || pass.desc.depthStencilAttachment.depthClearValue === 0));
});

test('fullscreen snapshots publish reversal to the source camera on both cold and warm draws', async t => {
  const {device, canvas} = deviceAndCanvas(), renderer = new (createWebGLRendererClass(T))({canvas, device, reversedDepthBuffer: true});
  t.after(() => renderer.dispose()); await renderer.init();
  const mesh = new T.Mesh(new T.PlaneGeometry(2, 2), shader(false)); mesh.frustumCulled = false;
  t.after(() => { mesh.geometry.dispose(); mesh.material.dispose(); });
  for (let i = 0; i < 2; i++) {
    const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10); camera.position.z = 2;
    await renderer.renderAsync(mesh, camera);
    assert.equal(camera.reversedDepth, true);
  }
});

test('queued camera projections retain lens, frustum and view-offset inputs without cloning source cameras', () => {
  for (const source of [new T.PerspectiveCamera(67, 1.3, .2, 80), new T.OrthographicCamera(-3, 4, 2, -1, .2, 80)]) {
    source.setViewOffset(400, 300, 50, 20, 250, 180);
    source.zoom = 1.5;
    if (source.isPerspectiveCamera) { source.filmGauge = 42; source.filmOffset = 5; }
    source.updateProjectionMatrix();
    const expected = source.clone(); expected._reversedDepth = true; expected.updateProjectionMatrix();
    source.clone = source.copy = () => assert.fail('projection capture cannot call source clone/copy hooks');
    const captured = captureThreeCamera(T, source);
    source.view.offsetX = 99; source.zoom = 3; source.left = -20; source.filmOffset = 20;
    captured._reversedDepth = true; captured.updateProjectionMatrix();
    assert.ok(captured.projectionMatrix.elements.every(Number.isFinite));
    assert.deepEqual(captured.projectionMatrix.elements, expected.projectionMatrix.elements);
    assert.deepEqual(captured.projectionMatrixInverse.elements, expected.projectionMatrixInverse.elements);
    assert.notEqual(captured.view, source.view);
  }
});

test('cold and warm six-face cube captures publish reversed source cameras and finite captured projections', async t => {
  const {device, canvas} = deviceAndCanvas(), renderer = new (createWebGLRendererClass(T))({canvas, device, reversedDepthBuffer: true});
  t.after(() => renderer.dispose()); await renderer.init();
  const target = new T.WebGLCubeRenderTarget(16, {generateMipmaps: false, minFilter: T.LinearFilter});
  const scene = new T.Scene(), material = new T.MeshBasicMaterial({side: T.BackSide}), geometry = new T.BoxGeometry(4, 4, 4);
  scene.add(new T.Mesh(geometry, material));
  t.after(() => { target.dispose(); material.dispose(); geometry.dispose(); });
  const camera = new T.CubeCamera(.1, 10, target);
  for (let iteration = 0; iteration < 2; iteration++) {
    camera.position.x = iteration * .25;
    camera.update(renderer, scene); await renderer.waitForGPU();
    assert.equal(renderer.getRenderTarget(), null);
    for (const child of camera.children) {
      assert.equal(child.reversedDepth, true);
      assert.equal(child.coordinateSystem, T.WebGLCoordinateSystem);
      assert.ok(child.projectionMatrix.elements.every(Number.isFinite));
      assert.ok(child.projectionMatrixInverse.elements.every(Number.isFinite));
    }
  }
  assert.deepEqual(device.passes.map(pass => pass.desc.colorAttachments[0].view.baseArrayLayer),
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);
  const draws = device.snapshots.flat();
  assert.equal(draws.length, 12);
  assert.ok(draws.every(draw => draw.pipeline.depthStencil.depthCompare === 'greater-equal'));
  assert.ok(device.passes.every(pass => pass.desc.depthStencilAttachment.depthClearValue === 0));
});
