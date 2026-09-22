/** Real pinned-Rollup application tests; no mock resolver, linker or Wasm VM. */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { rollup } from "rollup";
import { buildApplication } from "./build_application.mjs";
import { bundleWithRollup } from "./bundler.mjs";
import { numericKernelRollupPlugin } from "./numeric_rollup.mjs";

const UPDATE = `export function integrate(a, v, dt) {
  for (let i = 0; i < a.length; i++) a[i] += v[i] * dt;
}`;
const ENTRY = `import { integrate } from './update.mjs';
export function tick(a, v, dt) { return integrate(a, v, dt); }
export { integrate };`;
function fixture(files = { "update.mjs": UPDATE, "entry.mjs": ENTRY }) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-numeric-app-"));
  fs.writeFileSync(path.join(root, "package.json"), '{"type":"module"}');
  const source = path.join(root, "src");
  fs.mkdirSync(source);
  for (const [name, code] of Object.entries(files)) {
    const dest = path.join(source, name);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, code);
  }
  return { root, source, entry: path.join(source, "entry.mjs"), out: path.join(root, "dist") };
}
function observeWasm(t) {
  const NativeInstance = WebAssembly.Instance;
  const counts = { instances: 0, calls: 0 };
  t.after(() => {
    WebAssembly.Instance = NativeInstance;
  });
  WebAssembly.Instance = (...args) => {
    const instance = Reflect.construct(NativeInstance, args);
    counts.instances++;
    return {
      exports: {
        memory: instance.exports.memory,
        run(...params) {
          counts.calls++;
          return instance.exports.run(...params);
        },
      },
    };
  };
  return counts;
}
function reportFrom(result) {
  return JSON.parse(
    fs.readFileSync(path.join(result.outDir, "f3d-numeric-specialization.json"), "utf8"),
  );
}
async function importEntry(result) {
  return import(pathToFileURL(path.join(result.outDir, result.entryFiles[0])));
}

test("normal application build specializes calls across linked source modules with no source API changes", async (t) => {
  const counts = observeWasm(t),
    app = fixture();
  const result = await buildApplication(app.entry, app.out, { specializeNumeric: true });
  const report = reportFrom(result);
  assert.deepEqual(report, result.numericSpecialization);
  assert.equal(report.compiledKernels, 1);
  assert.equal(report.rewrittenCalls, 1);
  assert.equal(report.runtimeAssets.length, 2);
  assert.equal(report.accelerated, false);
  assert.ok(report.units.some((unit) => unit.moduleIds.some((id) => id.endsWith("/update.mjs"))));
  const module = await importEntry(result);
  assert.deepEqual(Object.keys(module).sort(), ["integrate", "tick"]);
  assert.equal(counts.instances, 0);
  const actual = new Float64Array([0, -0, NaN, Infinity, -Infinity, 1]),
    expected = actual.slice();
  const velocity = new Float64Array([1, -1, 2, 3, -3, 4]);
  for (let frame = 0; frame < 240; frame++) {
    module.tick(actual, velocity, 1 / 60);
    module.integrate(expected, velocity, 1 / 60);
    assert.deepEqual(actual, expected);
  }
  assert.deepEqual(counts, { instances: 1, calls: 240 });
});

test("specialization is opt-in and standalone bundle API reports the same assets", async () => {
  const app = fixture();
  const ordinary = await buildApplication(app.entry, app.out);
  assert.equal(ordinary.numericSpecialization, undefined);
  assert.ok(!ordinary.emittedFiles.some((name) => name.includes("numeric-")));
  const result = await bundleWithRollup(app.entry, { specializeNumeric: true });
  assert.equal(result.numericSpecialization.compiledKernels, 1);
  for (const asset of result.numericSpecialization.runtimeAssets)
    assert.equal(typeof result.files[asset.fileName], "string");
});

test("HTML document, assets and SRI use the specialized final bytes", async (t) => {
  const counts = observeWasm(t);
  const app = fixture({
    "update.mjs": UPDATE,
    "entry.mjs": `import { integrate } from './update.mjs';
      const a = new Float64Array([1]); integrate(a, new Float64Array([2]), 3);
      globalThis.__f3d_app_test_value = a[0];`,
    "index.html": `<!doctype html><title>numeric app</title><link rel="stylesheet" href="./app.css">
      <script>globalThis.classicAppBehavior = true;</script>
      <link rel="modulepreload" href="./entry.mjs" integrity="sha256-old">
      <script type="module" src="./entry.mjs" integrity="sha256-old" data-custom="retained"></script>`,
    "app.css": "body { font-size: 16px; }",
  });
  t.after(() => {
    delete globalThis.__f3d_app_test_value;
  });
  const result = await buildApplication(path.join(app.source, "index.html"), app.out, {
    specializeNumeric: true,
  });
  const html = fs.readFileSync(path.join(app.out, "index.html"), "utf8");
  assert.ok(html.includes("<script>globalThis.classicAppBehavior = true;</script>"));
  assert.ok(html.includes('data-custom="retained"'));
  assert.equal(fs.readFileSync(path.join(app.out, "app.css"), "utf8"), "body { font-size: 16px; }");
  const code = fs.readFileSync(path.join(app.out, result.entryFiles[0]));
  const integrity = "sha256-" + crypto.createHash("sha256").update(code).digest("base64");
  assert.equal(html.split(integrity).length - 1, 2);
  await importEntry(result);
  assert.equal(globalThis.__f3d_app_test_value, 7);
  assert.equal(counts.calls, 1);
});

test("dynamic chunks specialize their local calls and remain loadable after relocation", async (t) => {
  const counts = observeWasm(t);
  const app = fixture({
    "entry.mjs": `export const load = () => import('./lazy.mjs');`,
    "lazy.mjs": `${UPDATE}\nexport function tick(a, v, dt) { integrate(a, v, dt); }`,
  });
  const result = await buildApplication(app.entry, app.out, { specializeNumeric: true });
  assert.equal(result.numericSpecialization.compiledKernels, 1);
  assert.ok(result.chunks.some((chunk) => chunk.isDynamicEntry));
  assert.ok(
    result.numericSpecialization.units.every((unit) =>
      fs.existsSync(path.join(app.out, unit.fileName)),
    ),
  );
  const relocated = path.join(app.root, "relocated");
  fs.cpSync(app.out, relocated, { recursive: true });
  const entry = await import(pathToFileURL(path.join(relocated, result.entryFiles[0])));
  const module = await entry.load();
  const a = new Float64Array([1]);
  module.tick(a, new Float64Array([2]), 3);
  assert.equal(a[0], 7);
  assert.equal(counts.calls, 1);
});

test("unsupported loops, wrong typed arrays and unavailable Wasm retain working application behavior", async (t) => {
  const app = fixture();
  const result = await buildApplication(app.entry, app.out, { specializeNumeric: true });
  const native = globalThis.WebAssembly;
  t.after(() => {
    globalThis.WebAssembly = native;
  });
  globalThis.WebAssembly = undefined;
  const module = await importEntry(result);
  for (const a of [[1, 2], new Float32Array([1, 2]), new Float64Array([1, 2])]) {
    module.tick(a, [2, 3], 2);
    assert.deepEqual([...a], [5, 8]);
  }
  globalThis.WebAssembly = native;
  const refused = fixture({
    "entry.mjs": `function sine(a) { for (let i = 0; i < a.length; i++) a[i] = Math.sin(a[i]); }
    export const tick = a => sine(a);`,
  });
  const retained = await buildApplication(refused.entry, refused.out, { specializeNumeric: true });
  assert.equal(retained.numericSpecialization.compiledKernels, 0);
  assert.deepEqual(retained.numericSpecialization.runtimeAssets, []);
  const a = [1];
  (await importEntry(retained)).tick(a);
  assert.equal(a[0], Math.sin(1));
});

test("CLI --build-app --specialize-numeric emits executable output and a build manifest", async () => {
  const app = fixture();
  const result = spawnSync(
    process.execPath,
    [
      fileURLToPath(new URL("./cli.mjs", import.meta.url)),
      "--entry",
      app.entry,
      "--build-app",
      app.out,
      "--specialize-numeric",
      "--max-memory-pages",
      "4",
      "--output",
      path.join(app.root, "build.json"),
    ],
    { encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /1 kernels, 1 guarded call sites/);
  const manifest = JSON.parse(fs.readFileSync(path.join(app.root, "build.json")));
  assert.equal(manifest.numericSpecialization.compiledKernels, 1);
  const a = new Float64Array([1]);
  (await importEntry(manifest)).tick(a, new Float64Array([2]), 3);
  assert.equal(a[0], 7);
});

test("invalid specialization configuration and output collisions fail without replacing files", async () => {
  const app = fixture();
  await assert.rejects(
    buildApplication(app.entry, app.out, { specializeNumeric: { maxKernels: 0 } }),
    /maxKernels/,
  );
  assert.equal(fs.existsSync(app.out), false);
  fs.mkdirSync(app.out);
  fs.writeFileSync(path.join(app.out, "entry.js"), "original data");
  await assert.rejects(
    buildApplication(app.entry, app.out, { specializeNumeric: true }),
    /overwrite/,
  );
  assert.equal(fs.readFileSync(path.join(app.out, "entry.js"), "utf8"), "original data");
});

test("actual Rollup plugin supports repeated generation and preserves requested source-map output", async () => {
  const app = fixture();
  const plugin = numericKernelRollupPlugin();
  const bundle = await rollup({ input: app.entry, plugins: [plugin] });
  try {
    const first = await bundle.generate({ format: "es" });
    assert.equal(plugin.api.getReport().compiledKernels, 1);
    const second = await bundle.generate({ format: "es" });
    assert.equal(plugin.api.getReport().compiledKernels, 1);
    assert.deepEqual(
      first.output.map((item) => item.fileName),
      second.output.map((item) => item.fileName),
    );
    const mapped = await bundle.generate({ format: "es", sourcemap: true });
    assert.equal(plugin.api.getReport().compiledKernels, 0);
    assert.deepEqual(plugin.api.getReport().runtimeAssets, []);
    assert.ok(mapped.output.find((item) => item.type === "chunk").map);
  } finally {
    await bundle.close();
  }
});
