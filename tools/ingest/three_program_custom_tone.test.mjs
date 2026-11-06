/** Application-defined WebGL tone operators: source compiler and queue behavior.
 * The recording device observes shader modules and submitted uniform bytes,
 * not GPU shader execution, rasterized pixels or performance.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLProgram} from '../../upstream/three.js/src/renderers/webgl/WebGLProgram.js';
import {ShaderChunk as sourceChunks} from '../../upstream/three.js/src/renderers/shaders/ShaderChunk.js';
import {compileEsslProgram} from './essl_wgsl.mjs';
import {createThreeProgramSupport, threeProgramSources} from './three_program.mjs';
import {createWebGLRendererClass, createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const customCurve = 'vec3(1.0) - exp(-color * toneMappingExposure)';
function customChunk(t, expression = customCurve) {
  const original = T.ShaderChunk.tonemapping_pars_fragment, sourceOriginal = sourceChunks.tonemapping_pars_fragment;
  const placeholder = 'vec3 CustomToneMapping( vec3 color ) { return color; }';
  assert.ok(original.includes(placeholder), 'pinned source custom tone hook');
  t.after(() => { T.ShaderChunk.tonemapping_pars_fragment = original; sourceChunks.tonemapping_pars_fragment = sourceOriginal; });
  const replace = value => {
    const chunk = original.replace(placeholder, `vec3 CustomToneMapping( vec3 color ) { return ${value}; }`);
    T.ShaderChunk.tonemapping_pars_fragment = sourceChunks.tonemapping_pars_fragment = chunk;
  };
  replace(expression); return replace;
}
const makeSupport = state => createThreeProgramSupport({three: T,
  state: () => ({toneMapping: T.CustomToneMapping, outputColorSpace: T.SRGBColorSpace, ...state})});
const shaderMaterial = () => new T.ShaderMaterial({
  vertexShader: 'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader: 'void main(){gl_FragColor=vec4(2.0,0.5,0.125,1.0);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}',
});
function sourceProgram(parameters) {
  const sources = {};
  const gl = {VERTEX_SHADER: 'vertex', FRAGMENT_SHADER: 'fragment', createProgram: () => ({}), createShader: type => ({type}),
    shaderSource(shader, code) { sources[shader.type] = code; }, compileShader() {}, attachShader() {}, bindAttribLocation() {}, linkProgram() {}};
  new WebGLProgram({getContext: () => gl}, 'custom-tone-source', parameters, {});
  return compileEsslProgram(sources.vertex, sources.fragment);
}
function fixture(t, native = false) {
  const device = geometryDevice(), canvasCalls = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; canvasCalls.push('configure'); }, unconfigure() { canvasCalls.push('unconfigure'); },
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const Renderer = (native ? createWebGPURendererClass : createWebGLRendererClass)(T);
  const renderer = new Renderer({canvas, device}), scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 1, .1, 100);
  const material = new T.MeshBasicMaterial(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  mesh.position.z = -3; scene.add(mesh);
  t.after(() => { renderer.dispose(); material.dispose(); mesh.geometry.dispose(); });
  return {renderer, scene, camera, mesh, material, device, canvasCalls};
}
const fragment = draw => draw.pipeline.fragment.module.code;
const customCall = /fn toneMapping\([\s\S]*?return CustomToneMapping\(color\);/;
function exposure(draw, reflection) {
  const {group, offsets} = draw.groups.get(0), {resource} = group.entries.find(entry => entry.binding === 0);
  const bytes = draw.contents.get(resource.buffer), uniform = reflection.uniforms.find(u => u.name === 'toneMappingExposure');
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    .getFloat32((resource.offset ?? 0) + (offsets[0] ?? 0) + uniform.offset, true);
}

test('custom tone operators compile from the pinned WebGLProgram source for built-in and authored materials', t => {
  customChunk(t);
  const support = makeSupport({});
  for (const material of [new T.MeshBasicMaterial(), new T.MeshPhongMaterial(), shaderMaterial()]) {
    const mesh = new T.Mesh(new T.PlaneGeometry(), material);
    t.after(() => { material.dispose(); mesh.geometry.dispose(); });
    const sources = threeProgramSources(T, material, mesh, {toneMapping: T.CustomToneMapping, outputColorSpace: T.SRGBColorSpace});
    const compiled = support.compile(material, mesh), reference = sourceProgram(sources.parameters);
    assert.equal(compiled.program.vertex, reference.vertex, material.type);
    assert.equal(compiled.program.fragment, reference.fragment, material.type);
    assert.deepEqual(compiled.program.reflection, reference.reflection, material.type);
    assert.match(compiled.program.fragment, customCall);
    assert.match(compiled.program.fragment, /exp\(\(\(-color\) \* f3d_u.toneMappingExposure\)\)/);
    assert.equal(support.compile(material, mesh), compiled, 'unchanged sources reuse the program');
  }
});

test('custom tone recompilation preserves live ShaderChunk replacement and material shader hooks', async t => {
  const replace = customChunk(t), f = fixture(t);
  f.material.onBeforeCompile = shader => {
    shader.uniforms.customOffset = {value: .125};
    shader.fragmentShader = 'uniform float customOffset;\n' + shader.fragmentShader.replace('#include <tonemapping_fragment>',
      '#include <tonemapping_fragment>\ngl_FragColor.rgb += vec3(customOffset);');
  };
  f.renderer.toneMapping = T.CustomToneMapping;
  await f.renderer.renderAsync(f.scene, f.camera);
  const first = f.device.snapshots.flat().at(-1);
  assert.match(fragment(first), /f3d_u.customOffset/);
  assert.match(fragment(first), customCall);
  replace('sqrt(color * toneMappingExposure)'); f.material.needsUpdate = true;
  await f.renderer.renderAsync(f.scene, f.camera);
  const second = f.device.snapshots.flat().at(-1);
  assert.notEqual(second.pipeline, first.pipeline, 'changed runtime source produces a new program');
  assert.match(fragment(second), /sqrt\(\(color \* f3d_u.toneMappingExposure\)\)/);
  assert.match(fragment(second), /f3d_u.customOffset/);
  assert.equal(f.renderer._hdr, false);
});

test('canvas custom mapping honors toneMapped=false and render-target tone bypass', async t => {
  customChunk(t);
  const f = fixture(t), target = new T.WebGLRenderTarget(32, 32);
  t.after(() => target.dispose());
  // Keep the offscreen draw on its source-program route as well.
  f.material.onBeforeCompile = () => {};
  f.renderer.toneMapping = T.CustomToneMapping;
  await f.renderer.renderAsync(f.scene, f.camera);
  f.material.toneMapped = false; f.material.needsUpdate = true;
  await f.renderer.renderAsync(f.scene, f.camera);
  f.material.toneMapped = true; f.material.needsUpdate = true;
  f.renderer.setRenderTarget(target);
  await f.renderer.renderAsync(f.scene, f.camera);
  const draws = f.device.snapshots.flat();
  assert.equal(draws.length, 3);
  assert.match(fragment(draws[0]), customCall);
  assert.doesNotMatch(fragment(draws[1]), /CustomToneMapping|toneMappingExposure/);
  assert.doesNotMatch(fragment(draws[2]), /CustomToneMapping|toneMappingExposure/);
  assert.equal(f.renderer._targets.capture(target)._programSupport.state().toneMapping, T.NoToneMapping);
  assert.equal(f.renderer._hdr, false);
});

test('queued custom and built-in operators keep their accepted exposure on one canvas session', async t => {
  customChunk(t);
  const f = fixture(t); await f.renderer.init();
  const session = f.renderer._session, requests = [[T.CustomToneMapping, .375], [T.LinearToneMapping, 1.75], [T.CustomToneMapping, 2.5]];
  for (const [tone, value] of requests) {
    f.renderer.toneMapping = tone; f.renderer.toneMappingExposure = value;
    f.renderer.render(f.scene, f.camera);
  }
  f.renderer.toneMapping = T.NoToneMapping; f.renderer.toneMappingExposure = 99;
  await f.renderer.waitForGPU();
  assert.equal(f.renderer._deferredError, null);
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, requests.length);
  for (let i = 0; i < requests.length; i++) {
    const [tone, value] = requests[i], support = makeSupport({toneMapping: tone});
    const {reflection} = support.compile(f.material, f.mesh).program;
    assert.equal(exposure(draws[i], reflection), value);
    assert.match(fragment(draws[i]), tone === T.CustomToneMapping ? customCall : /return LinearToneMapping\(color\);/);
  }
  assert.equal(f.renderer._session, session);
  assert.equal(f.renderer._hdr, false);
});

test('callback-free queued custom curves retain A, B and A without replacing global shader source', async t => {
  const replace = customChunk(t), f = fixture(t); await f.renderer.init();
  f.renderer.toneMapping = T.CustomToneMapping; f.renderer.toneMappingExposure = .75;
  for (const curve of [customCurve, 'sqrt(color * toneMappingExposure)', customCurve]) {
    replace(curve); f.material.needsUpdate = true;
    f.renderer.render(f.scene, f.camera);
  }
  replace('color * 0.25');
  const live = T.ShaderChunk.tonemapping_pars_fragment;
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, 3);
  assert.match(fragment(draws[0]), /exp\(\(\(-color\) \* f3d_u.toneMappingExposure\)\)/);
  assert.match(fragment(draws[1]), /sqrt\(\(color \* f3d_u.toneMappingExposure\)\)/);
  assert.equal(fragment(draws[2]), fragment(draws[0]));
  assert.equal(T.ShaderChunk.tonemapping_pars_fragment, live, 'private source overrides never replace application globals');
});

test('captured custom source keys both the shader-hook cache and final compiler program', t => {
  const replace = customChunk(t), firstChunk = T.ShaderChunk.tonemapping_pars_fragment;
  replace('sqrt(color * toneMappingExposure)'); const secondChunk = T.ShaderChunk.tonemapping_pars_fragment;
  const state = {toneMappingChunk: firstChunk}, support = makeSupport(state);
  const material = new T.MeshBasicMaterial(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  t.after(() => { material.dispose(); mesh.geometry.dispose(); });
  let calls = 0; material.onBeforeCompile = () => { calls++; };
  const first = support.compile(material, mesh);
  replace('color * 0.25');
  assert.equal(support.compile(material, mesh), first); assert.equal(calls, 1);
  state.toneMappingChunk = secondChunk;
  const second = support.compile(material, mesh);
  assert.notEqual(second, first); assert.equal(calls, 2);
  assert.match(first.program.fragment, /exp\(\(\(-color\) \* f3d_u.toneMappingExposure\)\)/);
  assert.match(second.program.fragment, /sqrt\(\(color \* f3d_u.toneMappingExposure\)\)/);
  state.toneMappingChunk = firstChunk;
  assert.equal(support.compile(material, mesh), first); assert.equal(calls, 2);
});

test('accessor-backed custom chunks keep their live read boundary during queued preparation', async t => {
  const descriptor = Object.getOwnPropertyDescriptor(T.ShaderChunk, 'tonemapping_pars_fragment');
  t.after(() => Object.defineProperty(T.ShaderChunk, 'tonemapping_pars_fragment', descriptor));
  let reads = 0, live = descriptor.value;
  const get = () => { reads++; return live; };
  Object.defineProperty(T.ShaderChunk, 'tonemapping_pars_fragment', {get, enumerable: true, configurable: true});
  const f = fixture(t); await f.renderer.init(); f.renderer.toneMapping = T.CustomToneMapping;
  f.renderer.render(f.scene, f.camera);
  assert.equal(reads, 0, 'capture must not invoke an application shader-source getter');
  live = live.replace('return color; }', 'return sqrt(color * toneMappingExposure); }');
  f.material.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  assert.equal(reads, 0);
  await f.renderer.waitForGPU();
  assert.ok(reads > 0);
  assert.ok(f.device.snapshots.flat().every(draw => /sqrt\(\(color \* f3d_u.toneMappingExposure\)\)/.test(fragment(draw))));
  assert.equal(Object.getOwnPropertyDescriptor(T.ShaderChunk, 'tonemapping_pars_fragment').get, get);
});

test('shader hooks can still change the live custom tone chunk before source compilation', async t => {
  const replace = customChunk(t), f = fixture(t);
  f.material.onBeforeCompile = () => replace('sqrt(color * toneMappingExposure)');
  f.renderer.toneMapping = T.CustomToneMapping;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.match(fragment(f.device.snapshots.flat().at(-1)), /sqrt\(\(color \* f3d_u.toneMappingExposure\)\)/);
});

test('hooks installed after a queued custom draw restore its live shader-source behavior', async t => {
  const replace = customChunk(t), f = fixture(t); await f.renderer.init();
  f.renderer.toneMapping = T.CustomToneMapping;
  f.renderer.render(f.scene, f.camera);
  f.material.onBeforeCompile = () => replace('sqrt(color * toneMappingExposure)');
  f.material.needsUpdate = true;
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, 1);
  assert.match(fragment(draws[0]), /sqrt\(\(color \* f3d_u.toneMappingExposure\)\)/);
});

test('invalid custom GLSL fails without replacing the canvas owner or selecting a fixed output pass', async t => {
  const replace = customChunk(t), f = fixture(t);
  f.renderer.toneMapping = T.CustomToneMapping;
  await f.renderer.renderAsync(f.scene, f.camera);
  const session = f.renderer._session, calls = f.canvasCalls.slice(), draws = f.device.snapshots.length;
  replace('missingCustomToneFunction(color)'); f.material.needsUpdate = true;
  await assert.rejects(f.renderer.compileAsync(f.scene, f.camera), error => error.code === 'THREE_SCENE_TONE_MAPPING' && /missingCustomToneFunction/.test(error.message));
  assert.equal(f.renderer._session, session); assert.equal(f.renderer._hdr, false);
  assert.notEqual(f.renderer._hdrFallback, true);
  assert.deepEqual(f.canvasCalls, calls); assert.equal(f.device.snapshots.length, draws);
});

test('queued built-in HDR fallback retains captured output settings and rejects a later custom transition', async t => {
  customChunk(t);
  const f = fixture(t); await f.renderer.init();
  f.renderer._dispatcher.prepare = async () => {
    throw Object.assign(new Error('source program unavailable'), {code: 'THREE_SCENE_TONE_MAPPING'});
  };
  f.renderer.toneMapping = T.ReinhardToneMapping; f.renderer.toneMappingExposure = .625;
  f.renderer.render(f.scene, f.camera);
  f.renderer.toneMapping = T.CustomToneMapping; f.renderer.toneMappingExposure = 9;
  f.renderer.render(f.scene, f.camera);
  await assert.rejects(f.renderer.waitForGPU(), /Custom tone mapping cannot replace whole-image HDR fallback/);
  assert.equal(f.renderer.info.f3d.presentedRenders, 1, 'the preceding accepted built-in frame still presents');
  assert.equal(f.renderer._hdr, true); assert.equal(f.renderer._hdrFallback, true);
  const defaults = f.renderer._session.diagnostics.renderer.defaults;
  assert.equal(defaults.toneMapping, 'reinhard'); assert.equal(defaults.exposure, .625);
  const output = f.device.snapshots.flat().find(draw => draw.pipeline.label?.startsWith('Animation HDR output'));
  assert.ok(output);
  const buffer = output.groups.get(0).group.entries.find(entry => entry.resource.buffer?.label === 'Animation output parameters').resource.buffer;
  assert.equal(new DataView(output.contents.get(buffer).buffer).getFloat32(0, true), .625);
  assert.match(f.renderer._deferredError?.message ?? '', /Custom tone mapping cannot replace whole-image HDR fallback/);
  const session = f.renderer._session, calls = f.canvasCalls.slice();
  assert.throws(() => f.renderer.render(f.scene, f.camera), /Custom tone mapping cannot replace whole-image HDR fallback/);
  assert.equal(f.renderer._session, session); assert.deepEqual(f.canvasCalls, calls);
});

test('native output still rejects CustomToneMapping and unknown WebGL operators remain explicit errors', async t => {
  const native = fixture(t, true); native.renderer.toneMapping = T.CustomToneMapping;
  await assert.rejects(native.renderer.init(), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(native.renderer._session, null);
  const webgl = fixture(t); await webgl.renderer.init(); webgl.renderer.toneMapping = 999;
  assert.throws(() => webgl.renderer.render(webgl.scene, webgl.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
});
