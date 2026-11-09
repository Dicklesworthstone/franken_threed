/** Build-time WebGPURenderer route substitution over the real module graph,
 * route decider and Rollup bundle. The emitted bundle is imported in Node to
 * prove which constructor the application actually receives; no GPU work runs.
 */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
import { buildApplication as buildSourceApplication, readsGlContextState } from "./build_application.mjs";
import { decideRendererRoute } from "../compat/route_decider.mjs";
import { geometryDevice } from "./fixtures/gpu_geometry_device.mjs";

const upstreamBuild = path.resolve(process.env.F3D_THREE_ROOT ?? "upstream/three.js", "build");
const buildApplication = (entry, out, options = {}) => buildSourceApplication(entry, out, {
  packageRootUrl: pathToFileURL(path.resolve(upstreamBuild, '..') + path.sep).href, ...options,
});

test.beforeEach(t => {
  // Native-route tests supply a capable host; no GPU work occurs here.
  const navigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {configurable: true,
    value: {gpu: {requestAdapter: async () => null}}});
  t.after(() => {
    if (navigator) Object.defineProperty(globalThis, 'navigator', navigator);
    else delete globalThis.navigator;
  });
});

function app(body, { html = true } = {}) {
  const dir = fs.mkdtempSync(path.join(tmpdir(), "f3d_route_build_"));
  const rel = path.relative(dir, upstreamBuild);
  fs.writeFileSync(path.join(dir, "main.js"), body);
  if (html)
    fs.writeFileSync(
      path.join(dir, "index.html"),
      `<!DOCTYPE html><script type="importmap">{"imports":{"three":"${rel}/three.webgpu.js","three/webgpu":"${rel}/three.webgpu.js","three/legacy":"${rel}/three.module.js"}}</script>` +
        `<script type="module" src="./main.js"></script>`,
    );
  return { dir, entry: path.join(dir, "index.html"), out: path.join(dir, "out") };
}
const PROGRAM = `import * as THREE from 'three/webgpu';
const renderer = new THREE.WebGPURenderer({ antialias: true });
export const isF3DRenderer = renderer.isF3DRenderer;
export const isWebGPURenderer = renderer.isWebGPURenderer;
export const route = renderer.info.f3d?.route;
export const scene = new THREE.Scene();
export const Scene = THREE.Scene;
export const MeshStandardNodeMaterial = THREE.MeshStandardNodeMaterial;
export const REVISION = THREE.REVISION;
`;

test("decider admits the general route only when the implementation is supplied", () => {
  const input = { constructorName: "WebGPURenderer", hostCapabilities: { hasWebGPU: true, hasWebGL: true } };
  assert.equal(decideRendererRoute(input).route, "retained-upstream");
  const general = decideRendererRoute({ ...input, generalWebGPUAvailable: true });
  assert.equal(general.route, "general-webgpu");
  assert.ok(general.reasons.includes("general-webgpu-admitted"));
  assert.equal(decideRendererRoute({ ...input, generalWebGPUAvailable: true, options: { forceWebGL: true } }).route, "exact-backend");
});

test("an ordinary WebGPURenderer application is routed and keeps upstream class identity", async () => {
  const a = app(PROGRAM);
  const result = await buildApplication(a.entry, a.out, { routeWebGPURenderer: true });
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.reason, "routed");
  assert.deepEqual(result.rendererRoute.substitutedModules, [pathToFileURL(path.join(upstreamBuild, "three.webgpu.js")).href]);
  assert.equal(result.rendererRoute.decisions[0].route, "general-webgpu");
  // Import the emitted bundle: the application receives the new-backend class
  // while every other export is the bundled upstream implementation.
  globalThis.document ??= { createElementNS: () => ({ width: 300, height: 150, style: {} }) };
  const emitted = await import(pathToFileURL(path.join(a.out, "main.js")).href);
  assert.equal(emitted.isF3DRenderer, true);
  assert.equal(emitted.isWebGPURenderer, true);
  assert.equal(emitted.route, "general-webgpu");
  assert.ok(emitted.scene instanceof emitted.Scene);
  assert.equal(typeof emitted.MeshStandardNodeMaterial, "function");
  assert.equal(emitted.REVISION, "186");
});

test('local renderer factories route through the production builder and preserve construction effects', async () => {
  const a = app(`import * as THREE from 'three/webgpu';
const events=[];
function create(canvas) {
  events.push('create');
  const renderer = new THREE.WebGPURenderer({canvas, get antialias(){events.push('antialias');return false;}});
  renderer.setSize(40,24,false); return renderer;
}
function outer(canvas) { events.push('outer'); return create(canvas); }
const canvas=document.createElement('canvas');
const renderer=outer(canvas);
export const observed=events;
export const flags=[renderer.isF3DRenderer,renderer.isWebGPURenderer,renderer instanceof THREE.WebGPURenderer,renderer.domElement===canvas];
export const size=renderer.getSize(new THREE.Vector2()).toArray();
export const dispose=()=>renderer.dispose();
`);
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer:true});
  const referenceOut = path.join(a.dir, 'reference');
  const reference = await buildApplication(a.entry, referenceOut);
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.decisions[0].route, 'general-webgpu');
  const previous = globalThis.document;
  globalThis.document = {createElement: () => ({width: 300, height: 150, style: {}})};
  let emitted, original;
  try {
    original = await import(pathToFileURL(path.join(referenceOut, reference.entryFiles[0])).href);
    emitted = await import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href);
    assert.deepEqual(emitted.flags, [true,true,true,true]);
    assert.deepEqual(emitted.observed, original.observed);
    assert.deepEqual(emitted.observed.slice(0,2), ['outer','create']);
    assert.deepEqual(emitted.size, [40,24]);
  } finally { emitted?.dispose(); globalThis.document=previous; }
});

test('private renderer toolkit and state records route through an emitted application', async () => {
  const a=app(`import * as THREE from 'three/webgpu';
const toolkit={Renderer:THREE.WebGPURenderer};
const alias=toolkit;
const {Renderer}=alias;
const state={renderer:new Renderer({antialias:false})};
const view=state;
const events=[];
function width(){events.push('width');return 40;}
view.renderer.setSize(width(),24,false);
export const native=view.renderer.isF3DRenderer===true;
export const flags=[view.renderer instanceof Renderer,view.renderer.isWebGPURenderer];
export const size=view.renderer.getSize(new THREE.Vector2()).toArray();
export const observed=events;
export const dispose=()=>view.renderer.dispose();
`);
  const result=await buildApplication(a.entry,a.out,{routeWebGPURenderer:true});
  const referenceOut=path.join(a.dir,'reference');
  const reference=await buildApplication(a.entry,referenceOut);
  assert.equal(result.rendererRoute.routed,true);
  assert.deepEqual(result.rendererRoute.decisions.map(site=>site.route),['general-webgpu']);
  assert.ok(result.chunks.some(chunk=>chunk.fileName.startsWith('f3d-renderer-runtime-')));
  const previous=globalThis.document;
  globalThis.document={createElementNS:()=>({width:300,height:150,style:{}})};
  let emitted;
  try {
    const original=await import(pathToFileURL(path.join(referenceOut,reference.entryFiles[0])).href);
    emitted=await import(pathToFileURL(path.join(a.out,result.entryFiles[0])).href);
    assert.equal(original.native,false); assert.equal(emitted.native,true);
    assert.deepEqual(emitted.flags,original.flags);
    assert.deepEqual(emitted.flags,[true,true]);
    assert.deepEqual(emitted.size,original.size); assert.deepEqual(emitted.size,[40,24]);
    assert.deepEqual(emitted.observed,original.observed); assert.deepEqual(emitted.observed,['width']);
  } finally { await emitted?.dispose(); globalThis.document=previous; }
});

test('publishing a renderer state record keeps its genuine source owner in the built application', async () => {
  const a=app(`import * as THREE from 'three/webgpu';
const state={renderer:new THREE.WebGPURenderer()};
const alias=state;
export const pluginState=alias;
`);
  const result=await buildApplication(a.entry,a.out,{routeWebGPURenderer:true});
  assert.equal(result.rendererRoute.routed,false);
  assert.ok(result.rendererRoute.decisions.some(site=>site.route==='exact-backend'));
  assert.equal(result.chunks.some(chunk=>chunk.fileName.startsWith('f3d-renderer-runtime-')),false);
});

for (const inline of [false, true]) test(`mixed source-selected WebGL and native construction preserve option effects with inline=${inline}`, async () => {
  const body = `import * as THREE from 'three/webgpu';
import {WebGPURenderer as Renderer} from 'three/webgpu';
const events = globalThis.__f3d_mixed_options;
const native = new THREE.WebGPURenderer({antialias: false});
const exact = new Renderer({
  forceWebGL: true,
  get alpha() { events.push('alpha'); return true; },
  get antialias() { events.push('antialias'); return false; },
  get sourceOption() { events.push('sourceOption'); return 17; }
}, events.push('extra argument'));
export const nativeFlags = [native.isF3DRenderer === true, exact.isF3DRenderer === true];
export const instances = [native instanceof THREE.WebGPURenderer, exact instanceof Renderer];
export const sourceFlags = [native.isWebGPURenderer, exact.isWebGPURenderer];
export const dispose = () => native.dispose();
`;
  const a = app(body), reference = app(body);
  if (inline) fs.writeFileSync(a.entry, fs.readFileSync(a.entry, 'utf8').replace(
    '<script type="module" src="./main.js"></script>', '<script type="module">\n' + body + '</script>'));
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  const source = await buildApplication(reference.entry, reference.out);
  assert.equal(result.rendererRoute.routed, true);
  assert.deepEqual(result.rendererRoute.decisions.map(value => value.route), ['general-webgpu', 'exact-backend']);
  assert.deepEqual(result.rendererRoute.decisions[1].reasons, ['explicit-source-selection']);
  assert.equal(result.rendererRoute.exactConstructionSites.length, 1);
  assert.ok(result.rendererRoute.exactConstructionSites[0].sourceSpan.start.offset > 0);
  const document = globalThis.document, previousEvents = globalThis.__f3d_mixed_options;
  globalThis.document = {createElementNS: () => ({width: 32, height: 24, style: {}})};
  try {
    globalThis.__f3d_mixed_options = [];
    const expected = await import(pathToFileURL(path.join(reference.out, source.entryFiles[0])).href);
    const observed = globalThis.__f3d_mixed_options.slice();
    globalThis.__f3d_mixed_options = [];
    const emitted = await import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href);
    assert.deepEqual(emitted.nativeFlags, [true, false]);
    assert.deepEqual(expected.nativeFlags, [false, false]);
    assert.deepEqual(emitted.instances, [true, true]);
    assert.deepEqual(emitted.sourceFlags, [true, true]);
    assert.deepEqual(globalThis.__f3d_mixed_options, observed,
      'source-selected option getters and extra arguments keep the exact source evaluation order');
    assert.equal(observed[0], 'extra argument');
    assert.ok(observed.includes('sourceOption'), 'source options are not rejected by native validation');
    emitted.dispose();
  } finally {
    globalThis.document = document; globalThis.__f3d_mixed_options = previousEvents;
  }
});

test('source-selected constructions in lazy modules retain the same routed export identity', async () => {
  const a = app(`import * as THREE from 'three/webgpu';
const native = new THREE.WebGPURenderer();
export const nativeFlag = native.isF3DRenderer;
export const load = () => import('./exact.js');
export const dispose = () => native.dispose();`);
  fs.writeFileSync(path.join(a.dir, 'exact.js'), `import {WebGPURenderer as Renderer} from 'three/webgpu';
const exact = new Renderer({forceWebGL: true});
export const nativeFlag = exact.isF3DRenderer === true;
export const sameClass = exact instanceof Renderer;`);
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.exactConstructionSites[0].moduleId, pathToFileURL(path.join(a.dir, 'exact.js')).href);
  const previous = globalThis.document;
  globalThis.document = {createElementNS: () => ({width: 32, height: 24, style: {}})};
  try {
    const emitted = await import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href);
    assert.equal(emitted.nativeFlag, true);
    const exact = await emitted.load();
    assert.equal(exact.nativeFlag, false); assert.equal(exact.sameClass, true);
    emitted.dispose();
  } finally { globalThis.document = previous; }
});

test('mixed source selection does not exempt opaque renderer state or unknown code', async () => {
  for (const extra of ['const backend = exact.backend;', "import 'https://example.invalid/renderer-plugin.js';"]) {
    const a = app(`import * as THREE from 'three/webgpu';
const native = new THREE.WebGPURenderer();
const exact = new THREE.WebGPURenderer({forceWebGL: true});\n` + extra);
    const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
    assert.equal(result.rendererRoute.routed, false);
    assert.deepEqual(result.rendererRoute.exactConstructionSites, []);
    assert.equal(result.chunks.some(chunk => chunk.fileName.startsWith('f3d-renderer-runtime-')), false);
  }
});

test("top-level native compilation and rendering complete across deferred runtime imports", async () => {
  const a = app(`import './before.js';
import * as THREE from 'three/webgpu';
const host = globalThis.__f3d_top_level_host;
const renderer = new THREE.WebGPURenderer({device: host.device});
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x112233, 1, 10);
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({map: new THREE.StorageTexture(2, 2)})));
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10); camera.position.z = 2;
host.order.push('compile');
await renderer.compileAsync(scene, camera);
await renderer.renderAsync(scene, camera);
host.order.push('rendered');
export const isF3DRenderer = renderer.isF3DRenderer;
export const drawCalls = renderer.info.render.drawCalls;
export const dispose = () => renderer.dispose();
export const load = () => import('./lazy.js');
`);
  fs.writeFileSync(path.join(a.dir, "before.js"), "globalThis.__f3d_top_level_host.order.push('before');\n");
  fs.writeFileSync(path.join(a.dir, "lazy.js"), "globalThis.__f3d_top_level_host.order.push('lazy'); export const value = 42;\n");
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  const runtime = result.chunks.find(chunk => chunk.fileName.startsWith("f3d-renderer-runtime-"));
  assert.ok(runtime, "renderer runtime is separate from the application entry");
  assert.ok(!runtime.modules.includes(pathToFileURL(path.join(a.dir, "main.js")).href));
  assert.ok(result.chunks.some(chunk => chunk.isDynamicEntry && chunk.modules.some(id => id.endsWith("/three_scene_uv.mjs"))));

  // Only the native GPU boundary is recorded. The emitted application's TLA,
  // route facade, scene preparation and deferred modules execute unchanged.
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.features = new Set();
  device.createSampler = descriptor => descriptor;
  device.createTexture = descriptor => {
    const size = descriptor.size;
    const texture = {...descriptor, dimension: "2d", depthOrArrayLayers: 1, sampleCount: 1,
      width: size.width ?? size[0], height: size.height ?? size[1], destroy() {},
      createView(options = {}) {return {texture, ...options};}};
    return texture;
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) {configuration = value;}, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [16, 16, 1], format: configuration.format, usage: 16}),
  })};
  const host = globalThis.__f3d_top_level_host = {device, order: []};
  const previousDocument = globalThis.document;
  globalThis.document = {createElementNS: () => canvas};
  let timer, emitted;
  try {
    emitted = await Promise.race([
      import(pathToFileURL(path.join(a.out, "main.js")).href),
      new Promise((_, reject) => {timer = setTimeout(() => reject(new Error("Top-level renderer preparation deadlocked")), 5000);}),
    ]);
    assert.equal(emitted.isF3DRenderer, true);
    assert.equal(emitted.drawCalls, 1);
    assert.deepEqual(host.order, ["before", "compile", "rendered"], "application effects keep their evaluation order");
    assert.equal((await emitted.load()).value, 42);
    assert.deepEqual(host.order, ["before", "compile", "rendered", "lazy"], "application dynamic imports remain deferred");
  } finally {
    clearTimeout(timer);
    emitted?.dispose();
    globalThis.document = previousDocument;
    delete globalThis.__f3d_top_level_host;
  }
});

test("default builds are unchanged and exact-backend applications are never substituted", async () => {
  const plain = app(PROGRAM);
  const unrouted = await buildApplication(plain.entry, plain.out);
  assert.equal(unrouted.rendererRoute, undefined);
  assert.ok(!fs.readFileSync(path.join(plain.out, "main.js"), "utf8").includes("F3D_RENDERER_"));

  const forced = app(`import * as THREE from 'three/webgpu';
export const renderer = new THREE.WebGPURenderer({ forceWebGL: true });`);
  const f = await buildApplication(forced.entry, forced.out, { routeWebGPURenderer: true });
  assert.equal(f.rendererRoute.routed, false);
  assert.equal(f.rendererRoute.reason, "construction-site-not-general-webgpu");
  assert.ok(!fs.readFileSync(path.join(forced.out, "main.js"), "utf8").includes("F3D_RENDERER_"));

  const escaped = app(`import * as THREE from 'three/webgpu';
export const renderer = new THREE.WebGPURenderer();
export const gl = document.createElement('canvas').getContext('webgl2');`);
  const e = await buildApplication(escaped.entry, escaped.out, { routeWebGPURenderer: true });
  assert.equal(e.rendererRoute.routed, false);

  const none = app(`import * as THREE from 'three/webgpu'; export const s = new THREE.Scene();`);
  const n = await buildApplication(none.entry, none.out, { routeWebGPURenderer: true });
  assert.equal(n.rendererRoute.reason, "no-webgpu-renderer-construction");
});

test('an emitted application attaches the actual retained RendererInspector and records native draws', async () => {
  const inspectorUrl = pathToFileURL(path.resolve(upstreamBuild, '../examples/jsm/inspector/RendererInspector.js')).href;
  const a = app(`import * as THREE from 'three/webgpu';
import {RendererInspector as Metrics} from ${JSON.stringify(inspectorUrl)};
const renderer=new THREE.WebGPURenderer({device:globalThis.__f3d_inspector_host.device});
globalThis.__f3d_inspector_host.dispose=()=>renderer.dispose();
const scene=new THREE.Scene();
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(),new THREE.MeshBasicNodeMaterial()));
const camera=new THREE.PerspectiveCamera(60,1,.1,10); camera.position.z=2;
await renderer.compileAsync(scene,camera);
const inspector=new Metrics(); renderer.inspector=inspector; inspector.maxFrames=2;
inspector.begin(); const rendering=renderer.renderAsync(scene,camera); inspector.finish();
// Observe this completed source frame before an asynchronous host tick starts
// the next frame. GPU submission and timestamp resolution still finish below.
const frame=inspector.getFrame();
export const observed={native:renderer.isF3DRenderer===true, route:renderer.info.f3d?.route,
  attached:renderer.inspector===inspector, owner:inspector.getRenderer()===renderer,
  type:inspector instanceof Metrics, renders:frame.renders.length,
  scene:frame.renders[0].scene===scene, camera:frame.renders[0].camera===camera,
  frames:inspector.frames.length, running:inspector.isRunning};
await rendering; await inspector.resolveTimestamp();
export const dispose=()=>renderer.dispose();
`);
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.decisions[0].route, 'general-webgpu');
  const device = geometryDevice(); device.features = new Set(); device.limits.maxTextureDimension2D = 8192;
  device.createSampler = descriptor => descriptor;
  device.createTexture = descriptor => ({...descriptor, dimension: '2d', sampleCount: 1, depthOrArrayLayers: 1,
    width: descriptor.size.width ?? descriptor.size[0],
    height: descriptor.size.height ?? descriptor.size[1], createView(options = {}) {return {texture: this, ...options};}, destroy() {}});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) {configuration = value;},
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [16, 16], format: configuration.format, usage: 16})})};
  const previous = {document: globalThis.document, request: globalThis.requestAnimationFrame,
    cancel: globalThis.cancelAnimationFrame, host: globalThis.__f3d_inspector_host};
  globalThis.document = {createElementNS: () => canvas}; globalThis.__f3d_inspector_host = {device};
  globalThis.requestAnimationFrame = callback => setTimeout(() => callback(performance.now()), 0);
  globalThis.cancelAnimationFrame = clearTimeout;
  let emitted;
  try {
    emitted = await import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href);
    assert.deepEqual(emitted.observed, {native: true, route: 'general-webgpu', attached: true, owner: true,
      type: true, renders: 1, scene: true, camera: true, frames: 1, running: false});
    assert.ok(device.encodedDrawCalls > 0); assert.ok(device.submissions.length > 0);
  } finally {
    await globalThis.__f3d_inspector_host.dispose?.(); globalThis.document = previous.document;
    globalThis.requestAnimationFrame = previous.request; globalThis.cancelAnimationFrame = previous.cancel;
    if (previous.host === undefined) delete globalThis.__f3d_inspector_host; else globalThis.__f3d_inspector_host = previous.host;
  }
});

test("H1 retains its upstream owner for the Inspector's native backend prototype mutation", async () => {
  const out = fs.mkdtempSync(path.join(tmpdir(), "f3d_route_h1_"));
  const entry = path.resolve(upstreamBuild, "../examples/webgpu_performance_renderbundle.html");
  const result = await buildApplication(entry, path.join(out, "out"), { routeWebGPURenderer: true });
  assert.equal(result.rendererRoute.routed, false);
  assert.ok(result.rendererRoute.decisions[0].reasons.includes("unresolved-force-webgl"));
  assert.ok(result.rendererRoute.decisions[0].reasons.includes("opaque-gl-escape"));
  assert.ok(!result.chunks.some(chunk => chunk.fileName.startsWith("f3d-renderer-runtime-")));
});

const GL_PROGRAM = `import * as THREE from 'three/legacy';
const renderer = new THREE.WebGLRenderer({ antialias: false });
export const isF3DRenderer = renderer.isF3DRenderer;
export const isWebGLRenderer = renderer.isWebGLRenderer;
export const anisotropy = renderer.capabilities.getMaxAnisotropy();
`;

function readsSourceGlState(source) {
  const id = 'file:///application/main.mjs';
  return readsGlContextState({modules: {[id]: {}}}, '/application/main.mjs', undefined,
    {read(url) { assert.equal(url, id); return {code: source}; }});
}

test('computed GL-context properties retain source ownership before renderer construction', () => {
  for (const expression of [
    "renderer.capabilities.maxTextures",
    "renderer['capabilities'].maxTextures",
    'renderer[`capabilities`].maxTextures',
    "renderer['cap' + 'abilities'].maxTextures",
    "renderer[flag ? 'capabilities' : 'extensions'].maxTextures",
    "renderer[(effect(), 'capabilities')].maxTextures",
    "renderer?.['capabilities']?.['maxTextures']",
    "renderer['capabilities'][capability]",
    "renderer['capabilities'][flag ? 'precision' : 'maxTextures']",
    "renderer['extensions']",
    "renderer['getContextAttributes']()",
    "renderer[`forceContextLoss`]()",
    "renderer['forceContextRestore']()",
  ]) assert.equal(readsSourceGlState(`const value = ${expression};`), true, expression);
  assert.equal(readsSourceGlState("const capabilities = renderer['capabilities']; capabilities.maxTextures;"), true);
});

test('computed served capabilities stay on the new renderer route', () => {
  for (const expression of [
    "renderer.capabilities.precision",
    "renderer['capabilities']['getMaxAnisotropy']()",
    'renderer[`capabilities`][`isWebGL2`]',
    "renderer['cap' + 'abilities']['getMax' + 'Precision']('highp')",
    "renderer?.['capabilities']?.['logarithmicDepthBuffer']",
    "renderer['capabilities'][flag ? 'precision' : 'reversedDepthBuffer']",
    "renderer[(effect(), 'capabilities')][(effect(), 'precision')]",
    "objects[index].visible",
  ]) assert.equal(readsSourceGlState(`const value = ${expression};`), false, expression);
});

test('computed capability analysis never evaluates source key expressions', () => {
  const key = '__f3d_capability_key_evaluated';
  const previous = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, {configurable: true, get() { assert.fail('source key evaluated during routing'); }});
  try {
    assert.equal(readsSourceGlState(`renderer[(globalThis.${key}, 'capabilities')].maxTextures;`), true);
    assert.equal(readsSourceGlState(`renderer['capabilities'][(globalThis.${key}, 'precision')];`), false);
  } finally {
    if (previous) Object.defineProperty(globalThis, key, previous);
    else delete globalThis[key];
  }
});

test('computed GL capability application emits the genuine synchronous source constructor', async t => {
  const a = app(GL_PROGRAM + "export const units = renderer['capabilities'].maxTextures;");
  const result = await buildApplication(a.entry, a.out, {routeWebGLRenderer: true});
  assert.equal(result.rendererRoute.routed, false);
  assert.equal(result.rendererRoute.reason, 'gl-context-state-read');
  assert.deepEqual(result.rendererRoute.substitutedModules, []);
  assert.ok(!result.chunks.some(chunk => chunk.fileName.startsWith('f3d-renderer-runtime-')));
  const marker = new Error('genuine source WebGL context request'), contexts = [];
  const canvas = {width: 32, height: 32, style: {}, addEventListener() {}, removeEventListener() {},
    setAttribute() {}, getContext(type) { contexts.push(type); throw marker; }};
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', {configurable: true, value: {createElementNS: () => canvas}});
  t.mock.method(console, 'error', () => {});
  try {
    await assert.rejects(import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href), error => error === marker);
    assert.deepEqual(contexts, ['webgl2']);
  } finally {
    if (previous) Object.defineProperty(globalThis, 'document', previous);
    else delete globalThis.document;
  }
});

test("WebGLRenderer apps route only when opted in and free of GL escapes or GL-state reads", async () => {
  const plain = app(GL_PROGRAM);
  const off = await buildApplication(plain.entry, plain.out, { routeWebGPURenderer: true });
  assert.equal(off.rendererRoute.routed, false);
  const on = app(GL_PROGRAM);
  const r = await buildApplication(on.entry, on.out, { routeWebGLRenderer: true });
  assert.equal(r.rendererRoute.routed, true);
  assert.deepEqual(r.rendererRoute.substitutedModules, [pathToFileURL(path.join(upstreamBuild, "three.module.js")).href]);
  assert.ok(r.rendererRoute.decisions[0].reasons.includes("general-webgl-surface-admitted"));
  globalThis.document ??= { createElementNS: () => ({ width: 300, height: 150, style: {} }) };
  const emitted = await import(pathToFileURL(path.join(on.out, "main.js")).href);
  assert.equal(emitted.isWebGLRenderer, true);
  assert.equal(emitted.isF3DRenderer, true);
  assert.equal(emitted.anisotropy, 16, "served: this route's sampler ceiling");

  for (const [body, reason] of [
    [GL_PROGRAM + "export const units = renderer.capabilities.maxTextures;", "gl-context-state-read"],
    [GL_PROGRAM + "export const gl = renderer.getContext();", "construction-site-not-general-webgpu"],
    [GL_PROGRAM + "export const ext = document.createElement('canvas').getContext('webgl2').getExtension('x');", "construction-site-not-general-webgpu"],
  ]) {
    const a = app(body);
    const result = await buildApplication(a.entry, a.out, { routeWebGLRenderer: true });
    assert.equal(result.rendererRoute.routed, false, body);
    assert.equal(result.rendererRoute.reason, reason);
  }
});

test("served capability reads (getMaxAnisotropy) do not veto the WebGL route", async () => {
  const a = app(GL_PROGRAM + "export const aniso = renderer.capabilities.getMaxAnisotropy();");
  const r = await buildApplication(a.entry, a.out, { routeWebGLRenderer: true });
  assert.equal(r.rendererRoute.routed, true);
});

test("H2 routes through the new backend after its verified controls ownership edge closes", async () => {
  const out = fs.mkdtempSync(path.join(tmpdir(), "f3d_route_h2_"));
  const entry = path.resolve(upstreamBuild, "../examples/webgl_marchingcubes.html");
  const result = await buildApplication(entry, path.join(out, "out"), { routeWebGLRenderer: true });
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.decisions[0].route, "general-webgpu");
  assert.deepEqual(result.rendererRoute.substitutedModules, [pathToFileURL(path.join(upstreamBuild, "three.module.js")).href]);
  assert.ok(result.chunks.some(chunk => chunk.fileName.startsWith("f3d-renderer-runtime-")));
});

test('a bundled standard-pass composer retains class identities and target sizing with the new renderer', async () => {
  const addon = name => pathToFileURL(path.resolve(upstreamBuild, '../examples/jsm/postprocessing/', name + '.js')).href;
  const a = app(`import * as THREE from 'three/legacy';
import {EffectComposer} from ${JSON.stringify(addon('EffectComposer'))};
import {MaskPass, ClearMaskPass} from ${JSON.stringify(addon('MaskPass'))};
import {RenderPass} from ${JSON.stringify(addon('RenderPass'))};
const renderer = new THREE.WebGLRenderer();
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
const composer = new EffectComposer(renderer);
const mask = new MaskPass(scene, camera), unmask = new ClearMaskPass();
composer.addPass(new RenderPass(scene, camera)); composer.addPass(mask); composer.addPass(unmask);
composer.renderToScreen = false; composer.setSize(32, 24);
export const isF3DRenderer = renderer.isF3DRenderer;
export const sourceClasses = [composer instanceof EffectComposer, mask instanceof MaskPass, unmask instanceof ClearMaskPass];
export const sizes = [composer.readBuffer.width, composer.readBuffer.height, composer.writeBuffer.width, composer.writeBuffer.height];
export const dispose = () => { composer.dispose(); renderer.dispose(); };
`);
  // Legacy passes import UniformsUtils from the same WebGL build as H6.
  fs.writeFileSync(a.entry, fs.readFileSync(a.entry, 'utf8').replace('/three.webgpu.js"', '/three.module.js"'));
  const result = await buildApplication(a.entry, a.out, {routeWebGLRenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.decisions[0].route, 'general-webgpu');
  assert.ok(result.chunks.some(chunk => chunk.modules.some(id => id.endsWith('/three_pass_state.mjs'))),
    'the source-verified stencil helpers are actually included in the emitted bundle');
  const previous = globalThis.document;
  globalThis.document = {createElementNS: () => ({width: 32, height: 24, style: {}})};
  let emitted;
  try {
    emitted = await import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href);
    assert.equal(emitted.isF3DRenderer, true);
    assert.deepEqual(emitted.sourceClasses, [true, true, true]);
    assert.deepEqual(emitted.sizes, [32, 24, 32, 24]);
  } finally { emitted?.dispose(); globalThis.document = previous; }
});

test('the original H4 application routes with its unchanged FirstPersonControls', async () => {
  const entry = path.resolve(upstreamBuild, '../examples/webgl_shadowmap_performance.html');
  const out = fs.mkdtempSync(path.join(tmpdir(), 'f3d-h4-first-person-'));
  const result = await buildApplication(entry, out, {routeWebGLRenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.deepEqual(result.rendererRoute.decisions.map(value => value.route), ['general-webgpu']);
  const control = pathToFileURL(path.resolve(upstreamBuild, '../examples/jsm/controls/FirstPersonControls.js')).href;
  assert.ok(result.chunks.some(chunk => chunk.modules.includes(control)), 'the actual pinned controls remain in the application');
  assert.ok(result.entryFiles.every(file => fs.existsSync(path.join(out, file))));
});

test("a bundled closed-controls application executes retained camera controls with the new renderer", async () => {
  const controlsUrl = pathToFileURL(path.resolve(upstreamBuild, "../examples/jsm/controls/OrbitControls.js")).href;
  const a = app(`import * as THREE from 'three/webgpu';
import {OrbitControls as Navigation} from ${JSON.stringify(controlsUrl)};
const renderer = new THREE.WebGPURenderer();
const camera = new THREE.PerspectiveCamera(); camera.position.z = 5;
const controls = new Navigation(camera, renderer.domElement);
controls.rotateLeft(.25);
export const isF3DRenderer = renderer.isF3DRenderer;
export const sourceControls = controls instanceof Navigation;
export const sameCamera = controls.object === camera;
export const cameraX = camera.position.x;
export const dispose = () => { controls.dispose(); renderer.dispose(); };
`);
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  const listeners = new Map();
  const document = {addEventListener() {}, removeEventListener() {}};
  const canvas = {width: 320, height: 160, clientWidth: 320, clientHeight: 160, style: {}, ownerDocument: document,
    getRootNode: () => document,
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name, callback) { if (listeners.get(name) === callback) listeners.delete(name); },
  };
  document.createElementNS = () => canvas;
  const previous = globalThis.document; globalThis.document = document;
  let emitted;
  try {
    emitted = await import(pathToFileURL(path.join(a.out, "main.js")).href);
    assert.equal(emitted.isF3DRenderer, true);
    assert.equal(emitted.sourceControls, true);
    assert.equal(emitted.sameCamera, true);
    assert.ok(Math.abs(emitted.cameraX) > .5, "the actual retained controls changed the source camera");
    assert.ok(listeners.has("pointerdown"));
  } finally {
    emitted?.dispose(); globalThis.document = previous;
  }
  assert.equal(listeners.size, 0, "the retained controls disconnect their native listeners");
});

test("opaque renderer consumers receive the actual retained upstream object", async () => {
  const a = app(`import * as THREE from 'three/webgpu';
const renderer = new THREE.WebGPURenderer();
const alias = renderer;
globalThis.__f3d_consume_renderer(alias);
export const native = renderer instanceof THREE.WebGPURenderer;
`);
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, false);
  assert.equal(result.rendererRoute.decisions[0].route, "exact-backend");
  assert.ok(result.rendererRoute.decisions[0].reasons.includes("opaque-gl-escape"));
  let received;
  const previousDocument = globalThis.document;
  globalThis.document ??= {createElementNS: () => ({width: 300, height: 150, style: {}})};
  globalThis.__f3d_consume_renderer = value => { received = value; };
  try {
    const emitted = await import(pathToFileURL(path.join(a.out, "main.js")).href);
    assert.equal(emitted.native, true);
    assert.equal(received.isWebGPURenderer, true);
    assert.equal(received.isF3DRenderer, undefined);
    assert.ok(received.backend, "the opaque consumer keeps the actual upstream backend");
  } finally {
    delete globalThis.__f3d_consume_renderer;
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

test('an emitted native application preserves the source constructor and synchronous APIs on a WebGL-only host', async () => {
  const a = app(`import * as THREE from 'three/webgpu';
const renderer = new THREE.WebGPURenderer({alpha: false});
renderer.setSize(40, 24);
renderer.setClearColor(0x336699, .5);
export const flags = [renderer.isF3DRenderer === true, renderer.isWebGPURenderer, renderer instanceof THREE.WebGPURenderer];
export const size = renderer.getSize(new THREE.Vector2()).toArray();
export const clearAlpha = renderer.getClearAlpha();
export const clearColor = renderer.getClearColor(new THREE.Color()).getHex();
`);
  const result = await buildApplication(a.entry, a.out, {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  delete globalThis.navigator.gpu;
  let acquired = 0;
  const previous = globalThis.document;
  globalThis.document = {createElementNS: () => ({width: 16, height: 16, style: {},
    getContext() { acquired++; assert.fail('Construction must preserve deferred source initialization'); }})};
  try {
    const emitted = await import(pathToFileURL(path.join(a.out, result.entryFiles[0])).href);
    assert.deepEqual(emitted.flags, [false, true, true]);
    assert.deepEqual(emitted.size, [40, 24]);
    assert.equal(emitted.clearAlpha, .5);
    assert.equal(emitted.clearColor, 0x336699);
    assert.equal(acquired, 0);
  } finally { globalThis.document = previous; }
});

test('an emitted WebGL application reaches the unchanged synchronous source context boundary without WebGPU', async t => {
  t.mock.method(console, 'error', () => {});
  const body = `import * as THREE from 'three/legacy';
const renderer = new THREE.WebGLRenderer({antialias: false, alpha: true});
export const isF3DRenderer = renderer.isF3DRenderer;
`;
  const original = app(body), candidate = app(body);
  const source = await buildApplication(original.entry, original.out);
  const result = await buildApplication(candidate.entry, candidate.out, {routeWebGLRenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  delete globalThis.navigator.gpu;
  const calls = [], marker = new Error('actual source WebGL context boundary');
  const previous = globalThis.document;
  globalThis.document = {createElementNS: () => ({width: 16, height: 16, style: {},
    addEventListener(name) { calls.push(['add', name]); },
    removeEventListener(name) { calls.push(['remove', name]); },
    setAttribute(...args) { calls.push(['attribute', ...args]); },
    getContext(...args) { calls.push(['context', ...args]); throw marker; }})};
  try {
    await assert.rejects(import(pathToFileURL(path.join(original.out, source.entryFiles[0])).href), error => error === marker);
    const expected = calls.splice(0);
    await assert.rejects(import(pathToFileURL(path.join(candidate.out, result.entryFiles[0])).href), error => error === marker);
    assert.deepEqual(calls, expected);
    assert.ok(calls.some(call => call[0] === 'context' && call[1] === 'webgl2'));
  } finally { globalThis.document = previous; }
});
