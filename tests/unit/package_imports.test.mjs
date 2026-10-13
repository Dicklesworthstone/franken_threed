import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { resolveModuleSpecifier, clearPackageJsonCache } from "../../tools/ingest/resolver.mjs";
import { IngestionResolutionError } from "../../tools/ingest/types.mjs";

const moduleText = "export const value = 42;\n";
function fixture(t, imports = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-private-imports-"));
  t.after(() => { clearPackageJsonCache(); fs.rmSync(root, { recursive: true, force: true }); });
  function write(relative, text = moduleText) {
    const filename = path.join(root, relative);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, text);
    return pathToFileURL(filename).href;
  }
  const manifest = (relative, fields) => write(relative, JSON.stringify(fields));
  const referrer = write("src/app.mjs");
  manifest("package.json", { name: "application", type: "module", imports });
  return {
    root, referrer, write, manifest,
    url: relative => pathToFileURL(path.join(root, relative)).href,
    resolve: (key, from = referrer, map = {}, options = {}) => resolveModuleSpecifier(key, from, map, options),
    oracle(key, from = referrer) {
      const child = spawnSync(process.execPath, ["--input-type=module", "--eval",
        `console.log(import.meta.resolve(${JSON.stringify(key)}))`],
      { cwd: path.dirname(fileURLToPath(from)), encoding: "utf8" });
      assert.equal(child.status, 0, child.stderr);
      return child.stdout.trim();
    },
  };
}
const codeIs = code => error => error instanceof IngestionResolutionError && error.code === code;

for (const [name, imports, key, target] of [
  ["exact private path", { "#math": "./lib/math.mjs" }, "#math", "lib/math.mjs"],
  ["conditional import", { "#math": { require: "./wrong.cjs", import: "./lib/math.mjs" } }, "#math", "lib/math.mjs"],
  ["nested unmatched condition", { "#math": { import: { development: "./wrong.mjs" }, default: "./math.mjs" } }, "#math", "math.mjs"],
  ["declaration-ordered default", { "#math": { default: "./first.mjs", import: "./later.mjs" } }, "#math", "first.mjs"],
  ["fallback arrays", { "#math": [[42, "../invalid.mjs"], null, { require: "./wrong.cjs" }, "./math.mjs"] }, "#math", "math.mjs"],
  ["exact wins over wildcard", { "#*": "./wrong/*.mjs", "#math": "./math.mjs" }, "#math", "math.mjs"],
  ["wildcard directory capture", { "#lib/*": "./private/*.mjs" }, "#lib/math/vector", "private/math/vector.mjs"],
  ["longest trailer", { "#*": "./wrong/*.mjs", "#*.js": "./lib/*.mjs" }, "#math.js", "lib/math.mjs"],
  ["longest prefix", { "#*long.js": "./wrong/*.mjs", "#lib/*": "./private/*" }, "#lib/long.js", "private/long.js"],
  ["all target stars are literal substitutions", { "#*": "./*/copy-*.mjs" }, "#$&", "$&/copy-$&.mjs"],
  ["query and fragment identity", { "#math": "./math.mjs?variant=1#one" }, "#math", "math.mjs?variant=1#one"],
]) test(name, async t => {
  const f = fixture(t, imports);
  f.write(target.split(/[?#]/)[0]);
  const resolved = f.resolve(key);
  assert.equal(resolved, f.url(target.split(/[?#]/)[0]) + target.slice(target.split(/[?#]/)[0].length));
  assert.equal(resolved, f.oracle(key));
  assert.equal((await import(resolved)).value, 42);
});

test("private imports work inside installed ESM dependencies", async t => {
  const f = fixture(t, { "#math": "./wrong.mjs" });
  f.manifest("node_modules/engine/package.json", { name: "engine", type: "module", exports: "./index.mjs", imports: { "#math": "./private/math.mjs" } });
  const entry = f.write("node_modules/engine/index.mjs", "export { value } from '#math';");
  const math = f.write("node_modules/engine/private/math.mjs");
  assert.equal(f.resolve("engine"), entry);
  assert.equal(f.resolve("#math", entry), math);
  assert.equal((await import(f.resolve("engine"))).value, 42);
});
test("nearest package scope shadows its parent's mappings", t => {
  const f = fixture(t, { "#math": "./outer.mjs" });
  f.write("outer.mjs");
  f.manifest("src/nested/package.json", { type: "module", imports: { "#math": "./inner.mjs" } });
  const from = f.write("src/nested/entry.mjs"), expected = f.write("src/nested/inner.mjs");
  assert.equal(f.resolve("#math", from), expected);
});
test("a nearest scope with no mapping cannot inherit an ancestor's imports", t => {
  const f = fixture(t, { "#math": "./outer.mjs" }); f.write("outer.mjs");
  f.manifest("src/nested/package.json", { type: "module" });
  assert.throws(() => f.resolve("#math", f.write("src/nested/entry.mjs")), codeIs("ERR_PACKAGE_IMPORT_NOT_DEFINED"));
});
test("scope lookup stops at node_modules even for a package with no manifest", t => {
  const f = fixture(t, { "#math": "./outer.mjs" }); f.write("outer.mjs");
  const from = f.write("node_modules/unowned/entry.mjs");
  assert.throws(() => f.resolve("#math", from), codeIs("ERR_PACKAGE_IMPORT_NOT_DEFINED"));
});
test("workspace symlinks resolve private imports from the real package scope", t => {
  const f = fixture(t);
  f.manifest("workspace/package.json", { name: "linked", type: "module", imports: { "#math": "./math.mjs" } });
  f.write("workspace/index.mjs"); const expected = f.write("workspace/math.mjs");
  fs.mkdirSync(path.join(f.root, "node_modules"));
  fs.symlinkSync(path.join(f.root, "workspace"), path.join(f.root, "node_modules/linked"), "junction");
  assert.equal(f.resolve("#math", f.url("node_modules/linked/index.mjs")), expected);
});
test("imports metadata is reread between builds", t => {
  const f = fixture(t, { "#math": "./a.mjs" }), a = f.write("a.mjs"), b = f.write("b.mjs");
  assert.equal(f.resolve("#math"), a);
  f.manifest("package.json", { type: "module", imports: { "#math": "./b.mjs" } });
  assert.equal(f.resolve("#math"), b);
});
test("two aliases of one file share identity, query variants do not", t => {
  const f = fixture(t, { "#a": "./math.mjs", "#b": "./math.mjs", "#v": "./math.mjs?v=2" }); f.write("math.mjs");
  assert.equal(f.resolve("#a"), f.resolve("#b"));
  assert.notEqual(f.resolve("#a"), f.resolve("#v"));
});
for (const [name, imports, key, code] of [
  ["undefined import", {}, "#missing", "ERR_PACKAGE_IMPORT_NOT_DEFINED"],
  ["exact null blocks broad wildcard", { "#*": "./*.mjs", "#math": null }, "#math", "ERR_PACKAGE_IMPORT_NOT_DEFINED"],
  ["specific null pattern blocks broad wildcard", { "#*": "./*.mjs", "#private/*": null }, "#private/math", "ERR_PACKAGE_IMPORT_NOT_DEFINED"],
  ["unmatched condition cannot use broader wildcard", { "#*": "./*.mjs", "#math": { require: "./math.cjs" } }, "#math", "ERR_PACKAGE_IMPORT_NOT_DEFINED"],
  ["empty fallback blocks", { "#math": [] }, "#math", "ERR_PACKAGE_IMPORT_NOT_DEFINED"],
  ["missing selected local file cannot fall back", { "#math": ["./missing.mjs", "./math.mjs"] }, "#math", "ERR_MODULE_NOT_FOUND"],
  ["parent target escape", { "#math": "../math.mjs" }, "#math", "ERR_INVALID_PACKAGE_TARGET"],
  ["encoded target escape", { "#math": "./lib/%2e%2e/math.mjs" }, "#math", "ERR_INVALID_PACKAGE_TARGET"],
  ["wildcard escape", { "#*": "./lib/*.mjs" }, "#../math", "ERR_INVALID_MODULE_SPECIFIER"],
  ["encoded wildcard separator", { "#*": "./lib/*.mjs" }, "#a%2fb", "ERR_INVALID_MODULE_SPECIFIER"],
  ["numeric conditional keys", { "#math": { 0: "./wrong.mjs", import: "./math.mjs" } }, "#math", "ERR_INVALID_PACKAGE_CONFIG"],
  ["invalid imports field", ["./math.mjs"], "#math", "ERR_INVALID_PACKAGE_CONFIG"],
  ["absolute external URL", { "#math": "https://example.org/math.mjs" }, "#math", "ERR_INVALID_PACKAGE_TARGET"],
  ["recursive aliases are not invented", { "#math": "#math" }, "#math", "ERR_INVALID_MODULE_SPECIFIER"],
]) test(name, t => {
  const f = fixture(t, imports); f.write("math.mjs");
  assert.throws(() => f.resolve(key), codeIs(code));
});
test("invalid private names fail even without a package scope", t => {
  const f = fixture(t), from = f.write("node_modules/unowned/index.mjs");
  for (const key of ["#", "#/x", "#math/"]) assert.throws(() => f.resolve(key, from), codeIs("ERR_INVALID_MODULE_SPECIFIER"));
});
test("external dependencies resolve from the mapping's package, not the source directory", t => {
  const f = fixture(t, { "#dep": "dep" });
  f.manifest("node_modules/dep/package.json", { type: "module", exports: "./index.mjs" });
  const expected = f.write("node_modules/dep/index.mjs");
  f.manifest("src/node_modules/dep/package.json", { type: "module", exports: "./wrong.mjs" });
  f.write("src/node_modules/dep/wrong.mjs");
  assert.equal(f.resolve("#dep"), expected);
  assert.equal(f.oracle("#dep"), expected);
});
test("external scoped-package patterns honor that dependency's export map", t => {
  const f = fixture(t, { "#math/*": "@scope/math/features/*" });
  f.manifest("node_modules/@scope/math/package.json", { type: "module", exports: { "./features/*": "./lib/*.mjs" } });
  const expected = f.write("node_modules/@scope/math/lib/vector.mjs");
  assert.equal(f.resolve("#math/vector"), expected);
  assert.equal(f.oracle("#math/vector"), expected);
});
test("external self references respect the owning package's exports", t => {
  const f = fixture(t);
  f.manifest("package.json", { name: "application", type: "module", exports: { "./math": "./math.mjs" }, imports: { "#math": "application/math" } });
  const expected = f.write("math.mjs");
  assert.equal(f.resolve("#math"), expected);
  assert.equal(f.oracle("#math"), expected);
});
test("private aliases to Three.js preserve the pin and addon subpaths", t => {
  const f = fixture(t, { "#three": "three", "#addon/*": "three/addons/*" });
  f.manifest("node_modules/three/package.json", { exports: "./wrong.mjs" }); f.write("node_modules/three/wrong.mjs");
  f.manifest("pinned/package.json", { exports: { ".": "./index.mjs", "./addons/*": "./examples/*" } });
  const expected = f.write("pinned/index.mjs"), addon = f.write("pinned/examples/Controls.mjs");
  const options = { packageRootUrl: path.join(f.root, "pinned") };
  assert.equal(f.resolve("#three", f.referrer, {}, options), expected);
  assert.equal(f.resolve("#addon/Controls.mjs", f.referrer, {}, options), addon);
});
test("HTML import maps override private aliases, including null blocks", t => {
  const f = fixture(t, { "#math": "./math.mjs" }); f.write("math.mjs");
  assert.equal(f.resolve("#math", f.referrer, { imports: { "#math": f.referrer } }), f.referrer);
  assert.throws(() => f.resolve("#math", f.referrer, { imports: { "#math": null } }), /blocked import specifier/);
});
test("external mappings retain the HTML map base and owning-package scope", t => {
  const f = fixture(t);
  f.manifest("node_modules/engine/package.json", { type: "module", imports: { "#dep": "dep" } });
  const from = f.write("node_modules/engine/src/index.mjs"), adapter = f.write("adapter.mjs");
  const map = { imports: { dep: "./wrong.mjs" }, scopes: { "./node_modules/engine/": { dep: "./adapter.mjs" } } };
  assert.equal(f.resolve("#dep", from, map, { mapBaseUrl: f.url("index.html") }), adapter);
});
for (const [name, exports, files, code] of [
  ["missing external file", "./missing.mjs", {}, "ERR_MODULE_NOT_FOUND"],
  ["blocked external export", { ".": null }, { "index.mjs": moduleText }, "ERR_PACKAGE_PATH_NOT_EXPORTED"],
  ["CommonJS external route", "./index.cjs", { "index.cjs": "module.exports = 42;" }, "F3D_PACKAGE_FORMAT"],
]) test(`${name} cannot silently choose an imports-array fallback`, t => {
  const f = fixture(t, { "#dep": ["dep", "./math.mjs"] }); f.write("math.mjs");
  f.manifest("node_modules/dep/package.json", { type: "module", exports });
  for (const [file, text] of Object.entries(files)) f.write(`node_modules/dep/${file}`, text);
  assert.throws(() => f.resolve("#dep"), codeIs(code));
});
test("missing external package fails at the original alias and source span", t => {
  const f = fixture(t, { "#dep": ["not-installed", "./math.mjs"] }); f.write("math.mjs");
  const span = { line: 4, column: 8, offset: 73 };
  assert.throws(() => f.resolve("#dep", f.referrer, {}, { span }), error => {
    assert.ok(codeIs("ERR_MODULE_NOT_FOUND")(error));
    assert.equal(error.specifier, "#dep"); assert.equal(error.referrerUrl, f.referrer); assert.equal(error.span, span);
    assert.match(error.message, /not-installed/); assert.match(error.message, /package\.json/);
    return true;
  });
});
test("blocked external import-map mappings cannot fall back", t => {
  const f = fixture(t, { "#dep": ["dep", "./math.mjs"] }); f.write("math.mjs");
  assert.throws(() => f.resolve("#dep", f.referrer, { imports: { dep: null } }), /blocked import specifier/);
});
test("builtin imports dependencies need explicit host adapters", t => {
  const f = fixture(t, { "#fs": "fs" }), adapter = f.write("adapter.mjs");
  assert.throws(() => f.resolve("#fs"), codeIs("F3D_NODE_BUILTIN"));
  assert.equal(f.resolve("#fs", f.referrer, { imports: { fs: adapter } }), adapter);
});
test("a remote referrer cannot capture local private imports", t => {
  const f = fixture(t, { "#math": "./math.mjs" }); f.write("math.mjs");
  assert.throws(() => f.resolve("#math", "https://example.org/app.mjs"), codeIs("F3D_PACKAGE_REFERRER"));
});
