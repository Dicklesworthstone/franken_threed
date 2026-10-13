import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { resolveInstalledPackage } from "../../tools/ingest/node_package_resolver.mjs";
import { IngestionResolutionError } from "../../tools/ingest/types.mjs";

const source = "export let value = 41; export function increment() { ++value; }\n";
const server = "throw new Error('server entry must never execute');\n";
function fixture(manifest = {}, files = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-legacy-entry-"));
  const write = (name, content) => {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, content);
    return filename;
  };
  write("package.json", JSON.stringify({ type: "module", imports: { "#dep": "dep" } }));
  const referrer = pathToFileURL(write("app.mjs", "")).href;
  write("node_modules/dep/package.json", JSON.stringify({ name: "dep", ...manifest }));
  for (const [name, text] of Object.entries(files)) write(`node_modules/dep/${name}`, text);
  const url = name => pathToFileURL(fs.realpathSync(path.join(root, "node_modules/dep", name))).href;
  return { root, write, referrer, url, resolve: name => resolveInstalledPackage(name, referrer) };
}
const codeIs = code => error => error instanceof IngestionResolutionError && error.code === code;

for (const [name, manifest, file] of [
  ["legacy module outranks CommonJS main", { main: "server.cjs", module: "client.mjs" }, "client.mjs"],
  ["legacy module is explicit ESM without type", { main: "server.cjs", module: "client.js" }, "client.js"],
  ["legacy module remains ESM beside type commonjs main", { type: "commonjs", main: "server.cjs", module: "client.js" }, "client.js"],
  ["module accepts an explicit relative prefix", { module: "./client.mjs" }, "client.mjs"],
  ["browser entry outranks module and main", { browser: "browser.mjs", module: "client.mjs", main: "server.cjs" }, "browser.mjs"],
  ["browser js in an ESM package", { type: "module", browser: "client.js", main: "server.cjs" }, "client.js"],
  ["browser path identical to declared module retains ESM format", { browser: "./client.js", module: "client.js" }, "client.js"],
  ["non-entry browser object does not shadow module", { browser: { fs: false }, module: "client.mjs" }, "client.mjs"],
  ["absent module preserves legacy main", { main: "client.mjs" }, "client.mjs"],
  ["non-string module is not an entrypoint", { module: {}, main: "client.mjs" }, "client.mjs"],
  ["empty module is not an entrypoint", { module: "", main: "client.mjs" }, "client.mjs"],
  ["default index remains available", { type: "module" }, "index.js"],
  ["exports outrank all legacy fields", { exports: "./export.mjs", browser: "browser.mjs", module: "client.mjs" }, "export.mjs"],
  ["exports browser condition retains priority", { exports: { browser: "./export.mjs", import: "./client.mjs" }, browser: "browser.mjs" }, "export.mjs"],
]) test(name, () => {
  const f = fixture(manifest, { [file]: source, "server.cjs": server });
  assert.equal(f.resolve("dep"), f.url(file));
});

for (const field of ["module", "browser"]) {
  test(`${field} selection cannot fall back when missing`, () => {
    const f = fixture({ [field]: "missing.mjs", main: "main.mjs" }, { "main.mjs": source });
    assert.throws(() => f.resolve("dep"), codeIs("ERR_MODULE_NOT_FOUND"));
  });
  for (const target of ["../outside.mjs", "lib/%2e%2e/outside.mjs", "node_modules/other.mjs", "\t../outside.mjs"]) {
    test(`${field} rejects unsafe target ${JSON.stringify(target)}`, () => {
      const f = fixture({ [field]: target, main: "main.mjs" }, { "main.mjs": source });
      assert.throws(() => f.resolve("dep"), codeIs("ERR_INVALID_PACKAGE_TARGET"));
    });
  }
  test(`${field} does not guess extensions`, () => {
    const f = fixture({ [field]: "client", main: "main.mjs" }, { "client.js": source, "main.mjs": source });
    assert.throws(() => f.resolve("dep"), codeIs("ERR_MODULE_NOT_FOUND"));
  });
  test(`${field} does not select a directory index`, () => {
    const f = fixture({ [field]: "client" }, { "client/index.mjs": source });
    assert.throws(() => f.resolve("dep"), codeIs("ERR_UNSUPPORTED_DIR_IMPORT"));
  });
}
for (const [name, manifest, files] of [
  ["CommonJS module target", { module: "client.cjs" }, { "client.cjs": server }],
  ["non-JavaScript module target", { module: "client.json" }, { "client.json": "{}" }],
  ["ambiguous browser js", { browser: "client.js" }, { "client.js": source }],
  ["CommonJS browser js", { type: "commonjs", browser: "client.js", module: "other.js" }, { "client.js": server }],
]) test(`${name} is not an invented translation route`, () => {
  const f = fixture(manifest, files);
  assert.throws(() => f.resolve("dep"), codeIs("F3D_PACKAGE_FORMAT"));
});

test("legacy entry fields never override export encapsulation", () => {
  const f = fixture({ exports: { ".": null }, module: "client.mjs", browser: "client.mjs" }, { "client.mjs": source });
  assert.throws(() => f.resolve("dep"), codeIs("ERR_PACKAGE_PATH_NOT_EXPORTED"));
  assert.throws(() => f.resolve("dep/client.mjs"), codeIs("ERR_PACKAGE_PATH_NOT_EXPORTED"));
});
test("an export missing its selected file never falls back to legacy fields", () => {
  const f = fixture({ exports: "./missing.mjs", module: "client.mjs", browser: "client.mjs" }, { "client.mjs": source });
  assert.throws(() => f.resolve("dep"), codeIs("ERR_MODULE_NOT_FOUND"));
});
test("legacy fields do not replace explicit subpath requests", () => {
  const f = fixture({ module: "missing.mjs", browser: "also-missing.mjs" }, { "feature.mjs": source });
  assert.equal(f.resolve("dep/feature.mjs"), f.url("feature.mjs"));
});
test("module metadata does not claim arbitrary CommonJS subpaths are ESM", () => {
  const f = fixture({ module: "client.js", type: "commonjs" }, { "client.js": source, "private.js": server });
  assert.throws(() => f.resolve("dep/private.js"), codeIs("F3D_PACKAGE_FORMAT"));
});
test("query and fragment identity survive legacy entry selection", () => {
  const f = fixture({ module: "client.mjs?profile=web#one" }, { "client.mjs": source });
  assert.equal(f.resolve("dep"), f.url("client.mjs") + "?profile=web#one");
});
test("package-private dependency aliases use the same entry policy", () => {
  const f = fixture({ module: "client.mjs", main: "server.cjs" }, { "client.mjs": source, "server.cjs": server });
  assert.equal(f.resolve("#dep"), f.resolve("dep"));
});
test("workspace aliases share the selected module identity", () => {
  const f = fixture({ module: "client.mjs" }, { "client.mjs": source });
  fs.symlinkSync(path.join(f.root, "node_modules/dep"), path.join(f.root, "node_modules/alias"), "junction");
  assert.equal(f.resolve("alias"), f.resolve("dep"));
});
test("errors retain original request and source span", () => {
  const f = fixture({ module: "missing.mjs" });
  const span = { line: 9, column: 4, offset: 99 };
  assert.throws(() => resolveInstalledPackage("dep", f.referrer, { span }), error => {
    assert.ok(codeIs("ERR_MODULE_NOT_FOUND")(error));
    assert.equal(error.specifier, "dep"); assert.equal(error.referrerUrl, f.referrer); assert.equal(error.span, span);
    return true;
  });
});
test("metadata changes are visible on the next resolution", () => {
  const f = fixture({ module: "a.mjs" }, { "a.mjs": source, "b.mjs": source });
  assert.equal(f.resolve("dep"), f.url("a.mjs"));
  f.write("node_modules/dep/package.json", JSON.stringify({ module: "b.mjs" }));
  assert.equal(f.resolve("dep"), f.url("b.mjs"));
});

test("native ESM linking executes a dual-package module and shares lazy live bindings", () => {
  const f = fixture({ type: "commonjs", main: "server.cjs", module: "client.js" }, { "client.js": source, "server.cjs": server });
  const resolver = new URL("../../tools/ingest/node_package_resolver.mjs", import.meta.url).href;
  // Use Node's actual ESM linker with the compiler's explicit ESM entry choice.
  // No source transformations, mocked package resolution, or CommonJS execution.
  f.write("loader.mjs", `import { resolveInstalledPackage } from ${JSON.stringify(resolver)};
export function resolve(name, context, next) {
  if (name === 'dep' || name === '#dep') return { url: resolveInstalledPackage(name, context.parentURL), format: 'module', shortCircuit: true };
  return next(name, context);
}`);
  f.write("run.mjs", `import { register } from 'node:module';
register(new URL('./loader.mjs', import.meta.url));
const a = await import('dep'); const b = await import('#dep'); a.increment();
const c = await import('dep'); console.log(JSON.stringify([a.value, b.value, c.value, a === b && b === c]));`);
  const child = spawnSync(process.execPath, [path.join(f.root, "run.mjs")], { encoding: "utf8" });
  assert.equal(child.status, 0, child.stderr);
  assert.deepEqual(JSON.parse(child.stdout), [42, 42, 42, true]);
});
