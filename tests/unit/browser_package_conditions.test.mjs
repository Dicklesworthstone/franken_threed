import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { resolveModuleSpecifier, resolveExportTargetValue, clearPackageJsonCache } from "../../tools/ingest/resolver.mjs";

function fixture(t, fields = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-browser-packages-"));
  t.after(() => { clearPackageJsonCache(); fs.rmSync(root, { recursive: true, force: true }); });
  const url = relative => pathToFileURL(path.join(root, relative)).href;
  function write(relative, text = "export const value = 42;\n") {
    const filename = path.join(root, relative);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, text);
    return url(relative);
  }
  const manifest = (relative, value) => write(relative, JSON.stringify(value));
  manifest("package.json", { name: "application", type: "module", ...fields });
  const entry = write("main.mjs");
  return { root, url, write, manifest, entry,
    resolve: (specifier, from = entry, map = {}, options = {}) => resolveModuleSpecifier(specifier, from, map, options) };
}

const choices = [
  ["browser-only entry", { browser: "./browser.mjs" }, "browser.mjs"],
  ["browser before generic import", { browser: "./browser.mjs", import: "./generic.mjs" }, "browser.mjs"],
  ["nested browser/import", { browser: { import: "./browser.mjs", require: "./wrong.cjs" }, default: "./generic.mjs" }, "browser.mjs"],
  ["nested import/browser", { import: { browser: "./browser.mjs", default: "./generic.mjs" } }, "browser.mjs"],
  ["build host node condition is inactive", { node: "./wrong.cjs", browser: "./browser.mjs" }, "browser.mjs"],
  ["require branch is inactive", { require: "./wrong.cjs", browser: "./browser.mjs" }, "browser.mjs"],
  ["default still obeys manifest order", { default: "./generic.mjs", browser: "./browser.mjs" }, "generic.mjs"],
  ["import before browser obeys manifest order", { import: "./generic.mjs", browser: "./browser.mjs" }, "generic.mjs"],
  ["unmatched browser subcondition continues", { browser: { development: "./wrong.mjs" }, import: "./generic.mjs" }, "generic.mjs"],
  ["environment mode is not invented", { development: "./wrong.mjs", production: "./wrong.mjs", browser: "./browser.mjs" }, "browser.mjs"],
  ["array chooses browser before later fallback", [{ browser: "./browser.mjs" }, "./generic.mjs"], "browser.mjs"],
];
for (const [name, target, expected] of choices) {
  test(`exports: ${name}`, async t => {
    const f = fixture(t);
    f.manifest("node_modules/engine/package.json", { name: "engine", type: "module", exports: target });
    f.write("node_modules/engine/browser.mjs"); f.write("node_modules/engine/generic.mjs");
    const resolved = f.resolve("engine");
    assert.equal(resolved, f.url(`node_modules/engine/${expected}`));
    assert.equal((await import(resolved)).value, 42);
  });
  test(`private imports: ${name}`, t => {
    const f = fixture(t, { imports: { "#engine": target } });
    f.write("browser.mjs"); f.write("generic.mjs");
    assert.equal(f.resolve("#engine"), f.url(expected));
  });
}

test("a browser null blocks default for both exports and private imports", t => {
  const f = fixture(t, { imports: { "#engine": { browser: null, default: "./generic.mjs" } } });
  f.write("generic.mjs");
  f.manifest("node_modules/engine/package.json", { exports: { browser: null, default: "./generic.mjs" } });
  f.write("node_modules/engine/generic.mjs");
  assert.throws(() => f.resolve("engine"), { code: "ERR_PACKAGE_PATH_NOT_EXPORTED" });
  assert.throws(() => f.resolve("#engine"), { code: "ERR_PACKAGE_IMPORT_NOT_DEFINED" });
});
test("a missing browser entry cannot switch to a generic file", t => {
  const f = fixture(t);
  f.manifest("node_modules/engine/package.json", { exports: [{ browser: "./missing.mjs" }, "./generic.mjs"] });
  f.write("node_modules/engine/generic.mjs");
  assert.throws(() => f.resolve("engine"), { code: "ERR_MODULE_NOT_FOUND" });
});
test("wildcard subpath exports use browser conditions", t => {
  const f = fixture(t);
  f.manifest("node_modules/engine/package.json", { type: "module", exports: { "./*": { browser: "./web/*.mjs", default: "./node/*.cjs" } } });
  const expected = f.write("node_modules/engine/web/math.mjs");
  assert.equal(f.resolve("engine/math"), expected);
});
test("external imports targets use browser conditions again in the dependency", t => {
  const f = fixture(t, { imports: { "#engine": { browser: "engine/math", default: "node-only" } } });
  f.manifest("node_modules/engine/package.json", { exports: { "./math": { browser: "./web.mjs", default: "./node.cjs" } } });
  const expected = f.write("node_modules/engine/web.mjs");
  assert.equal(f.resolve("#engine"), expected);
});
test("pinned Three.js uses the same browser condition policy", t => {
  const f = fixture(t);
  f.manifest("pinned/package.json", { exports: { browser: "./browser.mjs", default: "./wrong.cjs" } });
  const expected = f.write("pinned/browser.mjs");
  assert.equal(f.resolve("three", f.entry, {}, { packageRootUrl: path.join(f.root, "pinned") }), expected);
});
test("import maps still override a browser-specific package export", t => {
  const f = fixture(t), adapter = f.write("adapter.mjs");
  f.manifest("node_modules/engine/package.json", { exports: { browser: "./missing.mjs" } });
  assert.equal(f.resolve("engine", f.entry, { imports: { engine: adapter } }), adapter);
});
test("explicit low-level conditions remain independent of the browser default", () => {
  const target = { node: "./node.mjs", browser: "./web.mjs", default: "./generic.mjs" };
  assert.equal(resolveExportTargetValue(target), "./web.mjs");
  assert.equal(resolveExportTargetValue(target, ["node"]), "./node.mjs");
  assert.equal(resolveExportTargetValue(target, []), "./generic.mjs");
});

test("the real resolver links a browser dependency graph with live bindings and lazy imports", t => {
  const f = fixture(t, { imports: { "#engine": { browser: "engine", default: "./server.cjs" } } });
  f.manifest("node_modules/engine/package.json", {
    name: "engine", type: "module",
    exports: { ".": { browser: "./browser.mjs", default: "./server.cjs" }, "./lazy": { browser: "./lazy.mjs" } },
    imports: { "#state": { browser: "./private/state.mjs", default: "./server.cjs" } },
  });
  f.write("node_modules/engine/browser.mjs", "export { value, bump } from '#state';");
  f.write("node_modules/engine/private/state.mjs", "export let value = 42; export function bump() { value++; }");
  f.write("node_modules/engine/lazy.mjs", "import { value } from 'engine'; export const read = () => value;");
  f.write("node_modules/engine/server.cjs", "throw new Error('server-only code evaluated');");
  f.write("main.mjs", `import { value, bump } from 'engine';
import { value as alias } from '#engine';
const lazy = await import('engine/lazy');
bump();
console.log(JSON.stringify([value, alias, lazy.read()]));`);
  // Native Node ESM linking/evaluation with this production resolver at every
  // application edge; no mocked package selection. This is not a Rollup build
  // or browser-rendering claim. The hook is confined to the temporary fixture.
  const resolver = new URL("../../tools/ingest/resolver.mjs", import.meta.url).href;
  const loader = f.write("loader.mjs", `import { resolveModuleSpecifier } from ${JSON.stringify(resolver)};
export function resolve(specifier, context, next) {
  if (!context.parentURL) return next(specifier, context);
  return { url: resolveModuleSpecifier(specifier, context.parentURL), shortCircuit: true };
}`);
  const child = spawnSync(process.execPath, ["--no-warnings", "--loader", loader, path.join(f.root, "main.mjs")],
    { cwd: f.root, encoding: "utf8" });
  assert.equal(child.status, 0, child.stderr);
  assert.deepEqual(JSON.parse(child.stdout), [43, 43, 43]);
});
