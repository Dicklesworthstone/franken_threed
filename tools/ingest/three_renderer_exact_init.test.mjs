/**
 * Runtime backend selection for source addons that override prototype.init.
 * Uses the unmodified r186 Inspector forceWebGL function and real source
 * renderer/backend constructors. Device initialization is intercepted at the
 * retained init boundary: these are ownership tests, not WebGL pixel tests.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createWebGPURendererClass} from './three_renderer.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
const settings = fs.readFileSync(path.join(root, 'examples/jsm/inspector/tabs/Settings.js'), 'utf8');
const begin = settings.indexOf('const _init = WebGPURenderer.prototype.init;');
const end = settings.indexOf('let _state = null;', begin);
assert.ok(begin >= 0 && end > begin, 'pinned Inspector initialization contract');
const installSettings = new Function('WebGPURenderer', 'WebGLBackend',
  settings.slice(begin, end) + '\nreturn forceWebGL;');

function fixture() {
  const calls = [], canvas = {width: 8, height: 8, style: {}, getContext() { throw new Error('Unexpected canvas context acquisition'); }};
  class ExactRenderer extends T.WebGPURenderer {
    init() { calls.push({receiver: this, backend: this.backend}); return Promise.resolve(this); }
  }
  const Routed = createWebGPURendererClass(T, {exactBackend: ExactRenderer});
  const original = Routed.prototype.init;
  return {calls, canvas, ExactRenderer, Routed, original, forceWebGL: installSettings(Routed, T.WebGLBackend)};
}

test('persisted source Inspector forceWebGL constructs the real retained renderer before native ownership', async () => {
  const f = fixture(), parameters = {canvas: f.canvas, antialias: true};
  f.forceWebGL(true);
  const renderer = new f.Routed(parameters);
  assert.ok(renderer instanceof f.ExactRenderer);
  assert.ok(renderer instanceof T.WebGPURenderer);
  assert.equal(renderer.isF3DRenderer, undefined);
  assert.equal(renderer.backend.isWebGPUBackend, true, 'original constructor selection before source init wrapper');
  assert.equal(renderer.backend.parameters.canvas, parameters.canvas);
  assert.equal(renderer.backend.parameters.antialias, true);
  const backendParameters = {...renderer.backend.parameters};
  assert.equal(renderer.domElement, f.canvas);
  assert.equal(await renderer.init(), renderer);
  assert.ok(renderer.backend instanceof T.WebGLBackend);
  assert.deepEqual(renderer.backend.parameters, backendParameters);
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].receiver, renderer);
  assert.equal(f.calls[0].backend, renderer.backend);
  assert.equal(Object.hasOwn(renderer, '_session'), false);
  assert.equal(Object.hasOwn(renderer, '_storageTextures'), false, 'native mixins leave genuine source return values untouched');
});

test('Settings restoration and subsequent activation stay live on already constructed retained renderers', async () => {
  const f = fixture();
  f.forceWebGL(true);
  const renderer = new f.Routed({canvas: f.canvas});
  const originalBackend = renderer.backend;
  assert.equal(Object.hasOwn(renderer, 'init'), false, 'source init remains inherited');
  assert.equal(delete renderer.init, true);
  assert.equal(renderer.init, f.Routed.prototype.init, 'deleting a non-own method does not remove live Settings dispatch');
  f.forceWebGL(false);
  assert.equal(renderer.init, f.original);
  await renderer.init();
  assert.equal(renderer.backend, originalBackend, 'restoring source init cancels the pending WebGL swap');
  f.forceWebGL(true);
  assert.equal(renderer.init, f.Routed.prototype.init);
  await renderer.init();
  const gl = renderer.backend;
  assert.ok(gl instanceof T.WebGLBackend);
  await renderer.init();
  assert.equal(renderer.backend, gl, 'the original source wrapper does not recreate an existing GL backend');
  assert.equal(f.calls.length, 3);
  f.forceWebGL(false);
  const native = new f.Routed({canvas: f.canvas});
  assert.equal(native.isF3DRenderer, true, 'new constructions resume native ownership once the source override is removed');
  native.dispose();
});

test('runtime forceWebGL still uses the unchanged exact constructor, with and without Inspector', async () => {
  const f = fixture(), parameters = {canvas: f.canvas, forceWebGL: true};
  const direct = new f.Routed(parameters);
  assert.ok(direct instanceof f.ExactRenderer);
  assert.ok(direct.backend instanceof T.WebGLBackend);
  assert.equal(Object.hasOwn(direct, 'init'), false, 'ordinary explicit fallback remains unchanged');
  f.forceWebGL(true);
  const inspected = new f.Routed(parameters);
  const backend = inspected.backend;
  await inspected.init();
  assert.equal(inspected.backend, backend);
  assert.equal(f.calls.length, 1);
});

test('an own init assignment follows ordinary inherited method assignment semantics', async () => {
  const f = fixture(); f.forceWebGL(true);
  const renderer = new f.Routed({canvas: f.canvas});
  let calls = 0;
  const override = function () { calls++; return f.original.call(this); };
  renderer.init = override;
  assert.deepEqual(Object.getOwnPropertyDescriptor(renderer, 'init'),
    {value: override, writable: true, enumerable: true, configurable: true});
  f.forceWebGL(false);
  await renderer.init();
  assert.equal(calls, 1);
  assert.equal(f.calls.length, 1);
  delete renderer.init;
  assert.equal(renderer.init, f.original);
  await renderer.init();
  assert.equal(calls, 1);
  assert.equal(f.calls.length, 2);
});

test('accessor init overrides are not evaluated during constructor route selection', async () => {
  const f = fixture(); let reads = 0, receiver;
  Object.defineProperty(f.Routed.prototype, 'init', {
    configurable: true,
    get() { reads++; receiver = this; return f.original; },
  });
  const renderer = new f.Routed({canvas: f.canvas});
  assert.equal(reads, 0);
  await renderer.init();
  assert.equal(reads, 1);
  assert.equal(receiver, renderer);
  assert.equal(f.calls.length, 1);
});

test('a retained constructor own init method takes precedence over the prototype override', async () => {
  let ownCalls = 0;
  class Exact {
    constructor(parameters) { this.parameters = parameters; this.init = () => { ownCalls++; }; }
    init() { throw new Error('Own source method must win'); }
  }
  const Routed = createWebGPURendererClass(T, {exactBackend: Exact});
  Routed.prototype.init = function () { throw new Error('Inherited method must stay shadowed'); };
  const parameters = {};
  const renderer = new Routed(parameters);
  assert.equal(renderer.parameters, parameters);
  renderer.init();
  assert.equal(ownCalls, 1);
});

test('retained init dispatch cannot be acquired by copying public renderer flags', () => {
  const f = fixture();
  const foreign = {isWebGPURenderer: true, isF3DRenderer: false, backend: new T.WebGLBackend({})};
  assert.throws(() => f.original.call(foreign));
  assert.equal(f.calls.length, 0);
});

test('subclass init wrappers keep their receiver and live prototype lookup on the retained route', async () => {
  const f = fixture(); let wrapperCalls = 0;
  class Derived extends f.Routed {
    init() { wrapperCalls++; return super.init(); }
  }
  const renderer = new Derived({canvas: f.canvas});
  assert.ok(renderer instanceof f.ExactRenderer);
  await renderer.init();
  assert.equal(wrapperCalls, 1);
  assert.equal(f.calls.length, 1);
  f.forceWebGL(true);
  await renderer.init();
  assert.equal(wrapperCalls, 2);
  assert.ok(renderer.backend instanceof T.WebGLBackend);
});
