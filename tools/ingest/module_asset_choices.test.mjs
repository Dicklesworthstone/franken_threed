import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { analyzeModuleAst } from "./ast_analyzer.mjs";
import { buildApplication, extractJsModuleDependencies } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-module-assets-"));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(root, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
  }
  return root;
}

const source = `
  export const events = [];
  const directory = './assets/', stable = directory + 'side.bin';
  export const stableURL = new URL(stable, import.meta['url']);
  export function selected(flag) {
    const name = (events.push('choose'), flag) ? 'blue image.bin' : 'red%23image.bin';
    const suffix = flag ? '?revision=2#blue' : '?revision=1#red';
    const alias = directory + name + suffix;
    return new URL((events.push('resolve'), alias), import.meta.url);
  }
  export function nested(flag) {
    return new URL((events.push(new URL(stable, import.meta.url).href),
      flag ? './assets/side.bin' : './assets/blue image.bin'), import.meta.url);
  }
  export function beforeInitialization() {
    const url = new URL(later, import.meta.url);
    const later = './assets/side.bin';
    return url;
  }
  export function thrownSelector() {
    return new URL(((() => { throw new RangeError('selector failed'); })(), stable), import.meta.url);
  }
`;
const files = () => ({
  "index.html": '<script type="module" src="./main.mjs"></script>',
  "main.mjs": source,
  "assets/blue image.bin": Buffer.from([1, 2, 3]),
  "assets/red#image.bin": Buffer.from([4, 5, 6]),
  "assets/side.bin": Buffer.from([7, 8, 9]),
});

test("module asset analysis closes lexical aliases and bounded string choices with selector spans", () => {
  const code = `const base = './assets/';
    const name = pick() ? 'blue' : 'red', alias = name;
    export const asset = new URL((observe(), base + alias + '.bin'), import.meta['url']);`;
  const [reference] = analyzeModuleAst(code, "file:///app/main.mjs").assetReferences;
  assert.ok(reference);
  assert.deepEqual(reference.candidates, ["./assets/blue.bin", "./assets/red.bin"]);
  assert.equal(
    code.slice(reference.argument_span.start.offset, reference.argument_span.end.offset),
    "observe(), base + alias + '.bin'",
  );
  assert.deepEqual(extractJsModuleDependencies(code).assetSpecifiers, reference.candidates);
});

test("finite module assets execute after relocation with native URLs, suffixes, effects and TDZ errors", async () => {
  const root = fixture(files());
  const built = await buildApplication(path.join(root, "main.mjs"), path.join(root, "dist"));
  const deployment = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-module-assets-deployed-"));
  const deployed = path.join(deployment, "app");
  fs.renameSync(built.outDir, deployed);
  fs.renameSync(root, root + "-source-unavailable");
  const module = await import(pathToFileURL(path.join(deployed, built.entryFiles[0])).href);
  assert.ok(module.stableURL instanceof URL);
  assert.deepEqual(fs.readFileSync(module.stableURL), Buffer.from([7, 8, 9]));
  for (const [flag, bytes, query, hash] of [
    [true, [1, 2, 3], "?revision=2", "#blue"],
    [false, [4, 5, 6], "?revision=1", "#red"],
  ]) {
    module.events.length = 0;
    const url = module.selected(flag);
    assert.ok(url instanceof URL);
    assert.equal(url.search, query);
    assert.equal(url.hash, hash);
    assert.deepEqual(fs.readFileSync(url), Buffer.from(bytes));
    assert.deepEqual(module.events, ["choose", "resolve"]);
  }
  module.events.length = 0;
  assert.deepEqual(fs.readFileSync(module.nested(false)), Buffer.from([1, 2, 3]));
  assert.equal(module.events.length, 1);
  assert.deepEqual(fs.readFileSync(new URL(module.events[0])), Buffer.from([7, 8, 9]));
  assert.throws(() => module.beforeInitialization(), ReferenceError);
  assert.throws(() => module.thrownSelector(), /selector failed/);
});

test("finite asset selectors pack directly and after a build with real data fetches and original effects", async () => {
  for (const buildFirst of [false, true]) {
    const root = fixture(files());
    const entry = path.join(root, "index.html");
    const built = buildFirst ? await buildApplication(entry, path.join(root, "dist")) : null;
    const packed = path.join(root, "packed.html");
    packHtml(built ? path.join(built.outDir, built.htmlFile) : entry, packed);
    const html = fs.readFileSync(packed, "utf8");
    const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
    assert.equal(Object.keys(imports).length, 1);
    fs.renameSync(root, root + "-source-unavailable");
    const module = await import(Object.values(imports)[0]);
    for (const [flag, bytes, hash] of [[true, [1, 2, 3], "#blue"], [false, [4, 5, 6], "#red"]]) {
      module.events.length = 0;
      const url = module.selected(flag);
      assert.ok(url instanceof URL);
      assert.equal(url.protocol, "data:");
      assert.equal(url.hash, hash);
      assert.deepEqual(new Uint8Array(await (await fetch(url)).arrayBuffer()), new Uint8Array(bytes));
      assert.deepEqual(module.events, ["choose", "resolve"]);
    }
    module.events.length = 0;
    assert.equal((await fetch(module.nested(true))).status, 200);
    assert.equal(module.events.length, 1);
    assert.equal((await fetch(module.events[0])).status, 200);
    assert.throws(() => module.beforeInitialization(), ReferenceError);
    assert.throws(() => module.thrownSelector(), /selector failed/);
  }
});

test("retained modules close constant asset choices without bundling their source", async () => {
  const root = fixture({
    "index.html": "<script>globalThis.load = () => import('./retained.mjs');</script>",
    "retained.mjs": "const asset = './payload.bin'; export const url = new URL(asset, import.meta.url);",
    "payload.bin": Buffer.from([10, 20, 30]),
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.deepEqual(built.entryFiles, []);
  assert.ok(built.emittedFiles.includes("payload.bin"));
  const deployment = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-retained-assets-"));
  const deployed = path.join(deployment, "app");
  fs.renameSync(built.outDir, deployed);
  fs.renameSync(root, root + "-source-unavailable");
  const module = await import(pathToFileURL(path.join(deployed, "retained.mjs")).href);
  assert.deepEqual(fs.readFileSync(module.url), Buffer.from([10, 20, 30]));
});

test("constant glTF module assets retain their existing model and sidecar closure", async () => {
  const root = fixture({
    "main.mjs": "const asset = './models/model.gltf'; export const url = new URL(asset, import.meta.url);",
    "models/model.gltf": JSON.stringify({asset: {version: "2.0"}, buffers: [{uri: "mesh.bin", byteLength: 3}]}),
    "models/mesh.bin": Buffer.from([31, 32, 33]),
  });
  const built = await buildApplication(path.join(root, "main.mjs"), path.join(root, "dist"));
  const module = await import(pathToFileURL(path.join(built.outDir, built.entryFiles[0])).href);
  assert.ok(module.url.pathname.includes("/assets/gltf-"));
  const model = JSON.parse(fs.readFileSync(module.url, "utf8"));
  assert.deepEqual(fs.readFileSync(new URL(model.buffers[0].uri, module.url)), Buffer.from([31, 32, 33]));
});

test("a missing finite asset candidate fails before any artifact is published", async () => {
  const root = fixture({
    "index.html": '<script type="module" src="./main.mjs"></script>',
    "main.mjs": "const asset = './missing.bin'; export const url = new URL(asset, import.meta.url);",
  });
  const out = path.join(root, "dist");
  await assert.rejects(buildApplication(path.join(root, "main.mjs"), out), /Unresolved module asset/);
  assert.equal(fs.existsSync(out), false);
  const packed = path.join(root, "packed.html");
  assert.throws(() => packHtml(path.join(root, "index.html"), packed), /missing\.bin/);
  assert.equal(fs.existsSync(packed), false);
});

test("mutable, unknown, directory and shadowed URL selectors retain their original build route", async () => {
  const code = `let asset = './not-present.bin';
    export function mutable(value) { asset = value; return new URL(asset, import.meta.url); }
    export function unknown(value) { return new URL(value, import.meta.url); }
    export function custom(URL) { const file = './not-present.bin'; return new URL(file, import.meta.url); }
    const directory = './assets/'; export const base = new URL(directory, import.meta.url);`;
  assert.deepEqual(analyzeModuleAst(code, "file:///app/main.mjs").assetReferences, []);
  const root = fixture({"main.mjs": code});
  const built = await buildApplication(path.join(root, "main.mjs"), path.join(root, "dist"));
  assert.equal(built.emittedFiles.some(file => file.endsWith(".bin")), false);
  const module = await import(pathToFileURL(path.join(built.outDir, built.entryFiles[0])).href);
  const data = "data:application/octet-stream;base64,AQI=";
  assert.equal(module.mutable(data).href, data);
  assert.equal(module.unknown(data).href, data);
  assert.equal(module.custom(class {constructor(value) {this.value = value;}}).value, "./not-present.bin");

  for (const [body, error] of [
    ["let asset = './unresolved.bin'; export const url = new URL(asset, import.meta.url);", /DYNAMIC_ASSET/],
    ["export const url = value => new URL(value, import.meta.url);", /DYNAMIC_ASSET/],
    ["export const url = new URL((s => (s => s)())('./unresolved.bin'), import.meta.url);", /DYNAMIC_ASSET/],
    ["export const url = new URL((s => getAsset())(), import.meta.url);", /DYNAMIC_ASSET/],
    ["export function custom(URL) {const asset='./unresolved.bin'; return new URL(asset, import.meta.url);}", /WORKER_BINDING/],
  ]) {
    const fixtureRoot = fixture({"index.html": '<script type="module" src="main.mjs"></script>', "main.mjs": body});
    assert.throws(() => packHtml(path.join(fixtureRoot, "index.html"), path.join(fixtureRoot, "packed.html")), error);
  }
});
