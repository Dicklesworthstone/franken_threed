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
export const renderer = new THREE.WebGPURenderer({ antialias: true });
export const scene = new THREE.Scene();
export { THREE };
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
  assert.equal(emitted.renderer.isF3DRenderer, true);
  assert.equal(emitted.renderer.isWebGPURenderer, true);
  assert.equal(emitted.renderer.info.f3d.route, "general-webgpu");
  assert.ok(emitted.scene instanceof emitted.THREE.Scene);
  assert.equal(typeof emitted.THREE.MeshStandardNodeMaterial, "function");
  assert.equal(emitted.THREE.REVISION, "186");
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

test("the unmodified H1 example (runtime forceWebGL choice) routes to the new backend", async () => {
  const out = fs.mkdtempSync(path.join(tmpdir(), "f3d_route_h1_"));
  const entry = path.resolve(upstreamBuild, "../examples/webgpu_performance_renderbundle.html");
  const result = await buildApplication(entry, path.join(out, "out"), { routeWebGPURenderer: true });
  assert.equal(result.rendererRoute.routed, true);
  assert.ok(result.rendererRoute.decisions[0].reasons.includes("unresolved-force-webgl"));
  // A truthy runtime forceWebGL constructs the unchanged upstream renderer.
  const code = fs.readFileSync(path.join(out, "out", "inline_0.js"), "utf8");
  assert.ok(code.includes("exactBackend"));
});

const GL_PROGRAM = `import * as THREE from 'three/legacy';
export const renderer = new THREE.WebGLRenderer({ antialias: false });
export { THREE };
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
  assert.equal(emitted.renderer.isWebGLRenderer, true);
  assert.equal(emitted.renderer.isF3DRenderer, true);
  assert.throws(() => emitted.renderer.capabilities, { code: "F3D_RENDERER_UNSUPPORTED" });
  assert.throws(() => emitted.renderer.getContext(), { code: "F3D_RENDERER_UNSUPPORTED" });

  for (const [body, reason] of [
    [GL_PROGRAM + "export const aniso = renderer.capabilities.getMaxAnisotropy();", "gl-context-state-read"],
    [GL_PROGRAM + "export const gl = renderer.getContext();", "construction-site-not-general-webgpu"],
    [GL_PROGRAM + "export const ext = document.createElement('canvas').getContext('webgl2').getExtension('x');", "construction-site-not-general-webgpu"],
  ]) {
    const a = app(body);
    const result = await buildApplication(a.entry, a.out, { routeWebGLRenderer: true });
    assert.equal(result.rendererRoute.routed, false, body);
    assert.equal(result.rendererRoute.reason, reason);
  }
});

test("the unmodified H2 example routes its WebGLRenderer to the new backend", async () => {
  const out = fs.mkdtempSync(path.join(tmpdir(), "f3d_route_h2_"));
  const entry = path.resolve(upstreamBuild, "../examples/webgl_marchingcubes.html");
  const result = await buildApplication(entry, path.join(out, "out"), { routeWebGLRenderer: true });
  assert.equal(result.rendererRoute.routed, true);
});
