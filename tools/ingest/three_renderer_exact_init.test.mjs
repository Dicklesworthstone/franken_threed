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
import {createWebGPURendererClass, createWebGLRendererClass, sourceThreeRendererConstructor} from './three_renderer.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
const GL = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
const settings = fs.readFileSync(path.join(root, 'examples/jsm/inspector/tabs/Settings.js'), 'utf8');
const begin = settings.indexOf('const _init = WebGPURenderer.prototype.init;');
const end = settings.indexOf('let _state = null;', begin);
assert.ok(begin >= 0 && end > begin, 'pinned Inspector initialization contract');
const installSettings = new Function('WebGPURenderer', 'WebGLBackend',
  settings.slice(begin, end) + '\nreturn forceWebGL;');

const capableGpu = {requestAdapter() { throw new Error('Unexpected native device initialization'); }};

function fixture({gpu = capableGpu} = {}) {
  const calls = [], canvas = {width: 8, height: 8, style: {}, getContext() { throw new Error('Unexpected canvas context acquisition'); }};
  class ExactRenderer extends T.WebGPURenderer {
    init() { calls.push({receiver: this, backend: this.backend}); return Promise.resolve(this); }
  }
  const Routed = createWebGPURendererClass(T, {exactBackend: ExactRenderer, gpu});
  const original = Routed.prototype.init;
  return {calls, canvas, ExactRenderer, Routed, original, forceWebGL: installSettings(Routed, T.WebGLBackend)};
}

test('compiler-selected source construction preserves exact options, prototype and exported instanceof', () => {
  const f = fixture(), events = [];
  const options = () => ({canvas: f.canvas, forceWebGL: true,
    get alpha() { events.push(['alpha', this]); return true; },
    get customOption() { events.push(['customOption', this]); return 7; }});
  const baseline = options(), direct = new f.ExactRenderer(baseline);
  const expected = events.map(([key]) => key); events.length = 0;
  const parameters = options(), Source = sourceThreeRendererConstructor(f.Routed);
  assert.equal(Source, f.ExactRenderer);
  const selected = new Source(parameters);
  assert.deepEqual(events.map(([key]) => key), expected);
  assert.ok(events.every(([, receiver]) => receiver === parameters));
  assert.equal(Object.getPrototypeOf(selected), Object.getPrototypeOf(direct));
  assert.equal(selected.constructor, f.ExactRenderer);
  assert.equal(selected instanceof f.Routed, true);
  assert.equal(selected instanceof T.WebGPURenderer, true);
  assert.ok(selected.backend instanceof T.WebGLBackend);
  assert.equal(selected.isF3DRenderer, undefined);
  assert.equal({isF3DRenderer: true, isWebGPURenderer: true} instanceof f.Routed, false);
  class Child extends f.Routed {}
  assert.equal(selected instanceof Child, false, 'inherited instance checks do not broaden arbitrary subclasses');
  assert.equal(sourceThreeRendererConstructor(Child), Child, 'unregistered subclasses keep their original constructor');
  assert.equal(sourceThreeRendererConstructor(T.WebGPURenderer), T.WebGPURenderer);
});

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

async function withHostGPU(gpu, run) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {gpu}});
  try { return await run(); }
  finally {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  }
}

test('an unavailable WebGPU host keeps source constructor options, prototypes and device policy untouched', async () => {
  await withHostGPU(undefined, () => {
    const calls = [];
    const canvas = {width: 8, height: 8, style: {}, getContext() { assert.fail('No context before source init'); }};
    const device = {};
    const parameters = forceWebGL => new Proxy({canvas, device, forceWebGL, alpha: true, customSourceOption: 19}, {
      get(target, key, receiver) { calls.push(['get', String(key)]); return Reflect.get(target, key, receiver); },
      set(target, key, value, receiver) { calls.push(['set', String(key)]); return Reflect.set(target, key, value, receiver); },
      ownKeys(target) { calls.push(['ownKeys']); return Reflect.ownKeys(target); },
      getOwnPropertyDescriptor(target, key) { calls.push(['descriptor', String(key)]); return Reflect.getOwnPropertyDescriptor(target, key); },
    });
    const Routed = createWebGPURendererClass(T, {exactBackend: T.WebGPURenderer});
    for (const forced of [false, true]) {
      const baseline = new T.WebGPURenderer(parameters(forced));
      const expected = calls.splice(0);
      const sourceOptions = parameters(forced), renderer = new Routed(sourceOptions);
      assert.deepEqual(calls.splice(0), expected, 'host selection never inspects, enumerates or copies source options');
      assert.equal(Object.getPrototypeOf(renderer), Object.getPrototypeOf(baseline));
      assert.equal(renderer.constructor, T.WebGPURenderer);
      assert.equal(renderer instanceof Routed, true);
      assert.equal(renderer.isF3DRenderer, undefined);
      assert.equal(renderer.domElement, canvas);
      assert.equal(renderer.backend.parameters.canvas, canvas);
      assert.equal(renderer.backend.parameters.device, device, 'the retained source still receives its supplied device');
      assert.equal(renderer.backend.isWebGLBackend === true, forced);
      assert.equal(renderer.backend.isWebGPUBackend === true, !forced);
      assert.equal(typeof sourceOptions.getFallback, forced ? 'undefined' : 'function');
      calls.length = 0;
      assert.equal(Object.hasOwn(renderer, '_session'), false);
      assert.equal(Object.hasOwn(renderer, '_storageTextures'), false);
    }
    assert.throws(() => new Routed(null), TypeError, 'the unchanged source owns invalid-input errors');
    assert.equal({isWebGPURenderer: true} instanceof Routed, false);
    class Child extends Routed {}
    assert.equal(new Routed({canvas}) instanceof Child, false);
  });
});

test('a no-WebGPU WebGL constructor reaches the actual source GL boundary with original option effects', async t => {
  t.mock.method(console, 'error', () => {});
  await withHostGPU(undefined, () => {
    const calls = [], marker = new Error('source WebGL context boundary');
    const canvas = {width: 8, height: 8, style: {}, addEventListener() {}, removeEventListener() {},
      setAttribute() {}, getContext(...args) { calls.push(['context', ...args]); throw marker; }};
    const parameters = () => ({
      get canvas() { calls.push('canvas'); return canvas; },
      get alpha() { calls.push('alpha'); return true; },
      get antialias() { calls.push('antialias'); return false; },
      get ignoredSourceOption() { assert.fail('Source ignores unknown options without reading them'); },
    });
    assert.throws(() => new GL.WebGLRenderer(parameters()), error => error === marker);
    const expected = calls.splice(0);
    const Routed = createWebGLRendererClass(GL, {exactBackend: GL.WebGLRenderer});
    assert.throws(() => new Routed(parameters()), error => error === marker);
    assert.deepEqual(calls, expected);
    assert.ok(calls.some(call => Array.isArray(call) && call[0] === 'context' && call[1] === 'webgl2'));
    assert.equal(sourceThreeRendererConstructor(Routed), GL.WebGLRenderer);
    assert.equal({isWebGLRenderer: true, isF3DRenderer: true} instanceof Routed, false);
  });
});

test('host selection happens per construction and native initialization failures never replace the exposed renderer', async () => {
  await withHostGPU(undefined, async () => {
    const canvas = {width: 8, height: 8, style: {}, getContext() { assert.fail('No context before an adapter exists'); }};
    const Routed = createWebGPURendererClass(T, {exactBackend: T.WebGPURenderer});
    const retained = new Routed({canvas});
    assert.equal(retained.isF3DRenderer, undefined);
    let requests = 0;
    globalThis.navigator.gpu = {async requestAdapter() { requests++; return null; }};
    const native = new Routed({canvas});
    assert.equal(native.isF3DRenderer, true);
    await assert.rejects(native.init(), {code: 'F3D_RENDERER_UNAVAILABLE'});
    assert.equal(requests, 1);
    assert.equal(native.isF3DRenderer, true);
    assert.equal(native.domElement, canvas);
    assert.equal(native instanceof T.WebGPURenderer, false);
    native.dispose();
    delete globalThis.navigator.gpu;
    assert.equal(new Routed({canvas}).isF3DRenderer, undefined);
    const Injected = createWebGPURendererClass(T, {exactBackend: T.WebGPURenderer, gpu: capableGpu});
    const injected = new Injected({canvas});
    assert.equal(injected.isF3DRenderer, true, 'an explicitly supplied host selects the native route');
    injected.dispose();
  });
});

test('host fallback preserves the live source Inspector init override', async () => {
  await withHostGPU(undefined, async () => {
    const f = fixture({gpu: null});
    f.forceWebGL(true);
    const renderer = new f.Routed({canvas: f.canvas});
    await renderer.init();
    assert.equal(renderer.isF3DRenderer, undefined);
    assert.ok(renderer.backend instanceof T.WebGLBackend);
    assert.equal(f.calls.length, 1);
    f.forceWebGL(false);
    assert.equal(renderer.init, f.original);
    await renderer.init();
    assert.equal(f.calls.length, 2);
  });
});
