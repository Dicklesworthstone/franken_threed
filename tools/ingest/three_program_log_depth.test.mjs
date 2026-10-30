/** Source logarithmic-depth shaders and camera uniforms on the WebGL facade. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createThreeProgramSupport} from './three_program.mjs';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const logarithmicShader = () => new T.ShaderMaterial({
  vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\nvoid main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);\n#include <logdepthbuf_vertex>\n}',
  fragmentShader: '#include <logdepthbuf_pars_fragment>\nvoid main(){\n#include <logdepthbuf_fragment>\ngl_FragColor=vec4(1.0);}',
});
const makeSupport = state => createThreeProgramSupport({three: T,
  state: () => ({toneMapping: T.NoToneMapping, outputColorSpace: T.SRGBColorSpace, ...state})});

test('source logarithmic chunks produce native fragment depth for built-in and authored programs', () => {
  const state = {logarithmicDepthBuffer: true}, support = makeSupport(state);
  for (const material of [new T.MeshBasicMaterial(), new T.MeshPhongMaterial(), new T.MeshNormalMaterial(),
    new T.MeshDepthMaterial(), logarithmicShader()]) {
    const mesh = new T.Mesh(new T.PlaneGeometry(), material);
    const compiled = support.compile(material, mesh);
    assert.equal(compiled.program.reflection.writesFragDepth, true, material.type);
    assert.ok(compiled.program.reflection.uniforms.some(u => u.name === 'logDepthBufFC'));
    assert.match(compiled.program.fragment, /@builtin\(frag_depth\)/);
    assert.match(compiled.program.fragment, /log2\(/);
    assert.equal(support.compile(material, mesh), compiled, 'unchanged source program is reused');
    const target = support.compile(material, mesh, {renderTarget: true});
    assert.equal(target.program.reflection.writesFragDepth, true, 'offscreen depth uses the same source flag');
    state.logarithmicDepthBuffer = false;
    assert.equal(support.compile(material, mesh).program.reflection.writesFragDepth, false);
    state.logarithmicDepthBuffer = true;
    mesh.geometry.dispose(); material.dispose();
  }
});

test('camera far updates retain source logDepthBufFC arithmetic without recompiling the program', () => {
  const support = makeSupport({logarithmicDepthBuffer: true}), material = new T.MeshBasicMaterial();
  const mesh = new T.Mesh(new T.PlaneGeometry(), material), compiled = support.compile(material, mesh);
  const reflection = compiled.program.reflection, uniform = reflection.uniforms.find(u => u.name === 'logDepthBufFC');
  const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
  for (const camera of [new T.PerspectiveCamera(), new T.OrthographicCamera()]) {
    camera.updateMatrixWorld(); mesh.updateMatrixWorld();
    for (const far of [16, 1024, 1e9, 1e12]) {
      camera.far = far; camera.updateProjectionMatrix();
      mesh.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
      support.pack(reflection, support.refresh(material), mesh, camera, bytes, {targetSize: [64, 32, 64, 32]});
      assert.equal(view.getFloat32(uniform.offset, true), Math.fround(2 / (Math.log(far + 1) / Math.LN2)));
      assert.equal(support.compile(material, mesh), compiled);
    }
  }
  mesh.geometry.dispose(); material.dispose();
});

test('authored logDepthBufFC uniforms override the source camera upload and remain valid with linear depth', () => {
  const material = logarithmicShader(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  const camera = new T.PerspectiveCamera(45, 1, .1, 1e7), state = {logarithmicDepthBuffer: true};
  camera.updateMatrixWorld(); mesh.updateMatrixWorld();
  material.uniforms.logDepthBufFC = {value: .375};
  const support = makeSupport(state), reflection = support.compile(material, mesh).program.reflection;
  const bytes = new Uint8Array(reflection.uniformBufferSize), offset = reflection.uniforms.find(u => u.name === 'logDepthBufFC').offset;
  for (const enabled of [true, false]) {
    state.logarithmicDepthBuffer = enabled;
    support.pack(reflection, material.uniforms, mesh, camera, bytes, {targetSize: [16, 16, 16, 16]});
    assert.equal(new DataView(bytes.buffer).getFloat32(offset, true), .375);
  }
  state.logarithmicDepthBuffer = true;
  material.uniforms.logDepthBufFC = {needsUpdate: false, get value() { throw new Error('A skipped source upload read its value'); }};
  support.pack(reflection, material.uniforms, mesh, camera, bytes, {targetSize: [16, 16, 16, 16]});
  assert.equal(new DataView(bytes.buffer).getFloat32(offset, true), Math.fround(2 / (Math.log(camera.far + 1) / Math.LN2)),
    'WebGLUniforms.upload skips needsUpdate=false after the source camera upload');
  mesh.geometry.dispose(); material.dispose();
});

test('WebGL facade admits logarithmic depth and passes it to actual canvas and target programs', async t => {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); }})};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device, logarithmicDepthBuffer: true});
  t.after(() => renderer.dispose()); await renderer.init();
  assert.equal(renderer.capabilities.logarithmicDepthBuffer, true);
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 1, .1, 1e9);
  const mesh = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial()); mesh.position.z = -3; scene.add(mesh);
  const target = new T.WebGLRenderTarget(32, 32);
  t.after(() => { target.dispose(); mesh.geometry.dispose(); mesh.material.dispose(); });
  for (const destination of [null, target]) {
    renderer.setRenderTarget(destination); await renderer.renderAsync(scene, camera);
    const support = destination ? renderer._targets.capture(target)._programSupport : renderer._programSupport;
    assert.equal(support.state().logarithmicDepthBuffer, true);
  }
  const draws = device.snapshots.flat();
  assert.equal(draws.length, 2);
  assert.ok(draws.every(draw => /@builtin\(frag_depth\)/.test(draw.pipeline.fragment.module.code)));
  assert.ok(draws.every(draw => draw.pipeline.depthStencil.depthCompare === 'less-equal'));
  assert.equal(camera.coordinateSystem, T.WebGLCoordinateSystem);
  assert.equal(camera.reversedDepth, false);
});
