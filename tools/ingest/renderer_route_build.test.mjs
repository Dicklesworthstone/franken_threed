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
import { buildApplication } from "./build_application.mjs";
import { decideRendererRoute } from "../compat/route_decider.mjs";
import { geometryDevice } from "./fixtures/gpu_geometry_device.mjs";

const upstreamBuild = path.resolve(process.env.F3D_THREE_ROOT ?? "upstream/three.js", "build");

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
