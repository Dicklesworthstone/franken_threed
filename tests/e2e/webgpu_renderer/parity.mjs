// Differential pixel check: the routed general-webgpu WebGPURenderer build vs the
// pinned upstream r186 WebGPURenderer (the same page built without routing) on
// deterministic source scenes, in the same Chromium.
// Usage: node tests/e2e/webgpu_renderer/parity.mjs [--reference webgl] [scenario...]
//
// This is a correctness oracle for the new backend, not a performance lane. A
// scenario fails when its candidate errors or its pixels diverge beyond the
// stated budget. `--reference webgl` compares against upstream WebGLRenderer
// instead; that renderer blends in sRGB-encoded space, so translucent scenes
// legitimately differ from both WebGPU renderers.
//
// The installed Chromium predates r186's string form of the identity texture
// view `swizzle: 'rgba'`; an init script removes only that identity value from
// view descriptors in the reference page. The upstream oracle is not edited.
// Headless Chromium here uses SwiftShader (software Vulkan); MSAA is disabled
// in both pages because this SwiftShader build drops sRGB encoding on MSAA
// resolves into reinterpreted canvas views even for raw WebGPU code.
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { buildApplication } from "../../../tools/ingest/build_application.mjs";
import { scenarios } from "./scenarios.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? "/opt/node-tools/node_modules/");
const { chromium } = require("playwright");
const CHANNEL_TOLERANCE = 12; // 8-bit levels per channel
const PIXEL_BUDGET = 0.02; // fraction of pixels allowed beyond the channel tolerance

const argv = process.argv.slice(2);
const refIndex = argv.indexOf("--reference");
const surfaceIndex = argv.indexOf("--surface");
// --surface webgl: routed WebGLRenderer page vs the unchanged upstream WebGLRenderer page.
const surface = surfaceIndex >= 0 ? argv.splice(surfaceIndex, 2)[1] : "webgpu";
if (!["webgpu", "webgl"].includes(surface)) throw new Error("--surface must be webgpu or webgl");
const referenceKind = refIndex >= 0 ? argv.splice(argv.indexOf("--reference"), 2)[1] : surface;
if (!["webgpu", "webgl"].includes(referenceKind)) throw new Error("--reference must be webgpu or webgl");
const out = fs.mkdtempSync(path.join(os.tmpdir(), "f3d_webgpu_parity_"));
const routed = await buildApplication(path.join(here, `${surface}.html`), path.join(out, "candidate"),
  surface === "webgl" ? { routeWebGLRenderer: true } : { routeWebGPURenderer: true });
if (!routed.rendererRoute?.routed) throw new Error(`candidate was not routed: ${routed.rendererRoute?.reason}`);
await buildApplication(path.join(here, `${referenceKind}.html`), path.join(out, "reference"));

const types = { ".js": "text/javascript", ".html": "text/html" };
const server = http.createServer((req, res) => {
  const file = path.join(out, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!file.startsWith(out) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": types[path.extname(file)] ?? "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
}).listen(0);
const base = `http://localhost:${server.address().port}`;
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/opt/pw-browsers/chromium",
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});

const identitySwizzleShim = () => {
  const createView = GPUTexture.prototype.createView;
  GPUTexture.prototype.createView = function (descriptor) {
    if (descriptor && descriptor.swizzle === "rgba") { descriptor = { ...descriptor }; delete descriptor.swizzle; }
    return createView.call(this, descriptor);
  };
};
async function capture(kind, name) {
  const page = await browser.newPage({ viewport: { width: 320, height: 240 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message.split("\n")[0]));
  if (kind === "reference" && referenceKind === "webgpu") await page.addInitScript(identitySwizzleShim);
  // Program-route scenarios must not pass through the whole-image HDR fallback.
  if (kind === "candidate" && WEBGL_ONLY.has(name)) await page.addInitScript(() => { globalThis.F3D_NO_FALLBACK = true; });
  const html = kind === "candidate" ? `${surface}.html` : `${referenceKind}.html`;
  await page.goto(`${base}/${kind}/${html}#${name}`);
  const state = await page
    .waitForFunction(() => window.__f3d && (window.__f3d.pixels || window.__f3d.error) && window.__f3d, null, { timeout: 20000 })
    .then((h) => h.jsonValue())
    .catch(() => page.evaluate(() => window.__f3d));
  await page.close();
  return { pixels: state?.pixels ?? null, error: state?.error ?? errors[0] ?? (state?.pixels ? null : "timeout") };
}

// Scenarios using WebGPU-build-only classes (node materials, BundleGroup).
const WEBGPU_ONLY = new Set(["node_materials"]);
// WebGLRenderer-only features (r186 WebGPURenderer does not run ShaderMaterial).
const WEBGL_ONLY = new Set(["shader_material", "shader_points", "shaderlib_instanced_morph", "shaderlib_wireframe_alphahash", "shader_uniform_blocks", "shader_integer_attributes", "shaderlib_maps", "shaderlib_physical", "shaderlib_transmission", "shaderlib_pmrem", "shaderlib_shadows", "shaderlib_vsm_shadows", "shaderlib_clipping", "shaderlib_background_cube", "shaderlib_background_texture", "shaderlib_tone_mapping", "shaderlib_equirect_cube", "shaderlib_skinning", "shaderlib_morph", "shaderlib_area_lights", "shaderlib_before_compile", "shaderlib_sprites", "shader_volume_textures", "shaderlib_indexed_points"]);
const selected = argv.length ? argv : Object.keys(scenarios).filter((n) => surface === "webgpu" ? !WEBGL_ONLY.has(n) : !WEBGPU_ONLY.has(n));
const results = [];
for (const name of selected) {
  const [candidate, reference] = [await capture("candidate", name), await capture("reference", name)];
  const row = { scenario: name, status: "fail" };
  if (reference.error) Object.assign(row, { status: "reference-error", detail: reference.error });
  else if (candidate.error) Object.assign(row, { detail: candidate.error });
  else {
    let over = 0, sum = 0;
    for (let i = 0; i < candidate.pixels.length; i += 4) {
      let worst = 0;
      for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(candidate.pixels[i + c] - reference.pixels[i + c]));
      sum += worst; if (worst > CHANNEL_TOLERANCE) over++;
    }
    const n = candidate.pixels.length / 4;
    Object.assign(row, { mismatched: +(over / n).toFixed(4), meanDiff: +(sum / n).toFixed(2) });
    row.status = over / n <= PIXEL_BUDGET ? "pass" : "fail";
    if (process.env.F3D_PARITY_DUMP) fs.writeFileSync(path.join(process.env.F3D_PARITY_DUMP, `${name}.json`), JSON.stringify({ candidate: candidate.pixels, reference: reference.pixels }));
  }
  results.push(row);
  console.log(JSON.stringify(row));
}
await browser.close();
server.close();
const failed = results.filter((r) => r.status !== "pass").length;
console.log(`${results.length - failed}/${results.length} scenarios within budget vs upstream ${referenceKind} (channel tolerance ${CHANNEL_TOLERANCE}, pixel budget ${PIXEL_BUDGET * 100}%)`);
process.exitCode = failed ? 1 : 0;
