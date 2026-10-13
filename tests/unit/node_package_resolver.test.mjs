import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { resolveModuleSpecifier, clearPackageJsonCache } from "../../tools/ingest/resolver.mjs";
import { IngestionResolutionError } from "../../tools/ingest/types.mjs";

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-installed-esm-"));
  t.after(() => { clearPackageJsonCache(); fs.rmSync(root, { recursive: true, force: true }); });
  function write(filename, content) {
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, content);
  }
  write(path.join(root, "package.json"), JSON.stringify({ name: "application", type: "module" }));
  const entry = path.join(root, "src", "app.mjs");
  write(entry, "export const value = 1;\n");
  const referrer = pathToFileURL(entry).href;
  function add(name, manifest, files, parent = root) {
    const directory = path.join(parent, "node_modules", name);
    write(path.join(directory, "package.json"), JSON.stringify({ name, type: "module", ...manifest }));
    for (const [filename, content] of Object.entries(files)) write(path.join(directory, filename), content);
    return directory;
  }
  return { root, entry, referrer, add, write, resolve: (specifier, from = referrer, map = {}, options = {}) => resolveModuleSpecifier(specifier, from, map, options) };
}
const moduleText = "export const value = 42;\n";
const fileUrl = filename => pathToFileURL(fs.realpathSync(filename)).href;
const codeIs = code => error => error instanceof IngestionResolutionError && error.code === code;

for (const [name, manifest, specifier, filename] of [
  ["unscoped string export", { exports: "./index.mjs" }, "dep", "index.mjs"],
  ["conditional import export", { exports: { require: "./wrong.cjs", import: "./index.mjs" } }, "dep", "index.mjs"],
  ["array fallback export", { exports: ["../invalid.mjs", null, "./index.mjs"] }, "dep", "index.mjs"],
  ["explicit legacy main", { main: "src/index.mjs" }, "dep", "src/index.mjs"],
  ["type-module index default", {}, "dep", "index.js"],
  ["exact legacy ESM subpath", {}, "dep/src/index.mjs", "src/index.mjs"],
  ["scoped wildcard export", { exports: { "./*": "./src/*.mjs" } }, "@scope/dep/math", "src/math.mjs"],
  ["encoded legacy subpath", {}, "dep/file%20name.mjs", "file name.mjs"],
]) {
  test(name, async t => {
    const f = fixture(t), packageName = specifier.startsWith("@") ? "@scope/dep" : "dep";
    const directory = f.add(packageName, manifest, { [filename]: moduleText });
    const resolved = f.resolve(specifier);
    assert.equal(resolved, fileUrl(path.join(directory, filename)));
    assert.equal((await import(resolved)).value, 42);
    const child = spawnSync(process.execPath, ["--input-type=module", "--eval", `console.log(import.meta.resolve(${JSON.stringify(specifier)}))`], { cwd: path.dirname(f.entry), encoding: "utf8" });
    assert.equal(child.status, 0, child.stderr);
    assert.equal(child.stdout.trim(), resolved);
  });
}

test("nearest nested dependency wins over project dependency", t => {
  const f = fixture(t);
  f.add("dep", { exports: "./index.mjs" }, { "index.mjs": "export const version = 'outer';" });
  const parent = f.add("parent", { exports: "./index.mjs" }, { "index.mjs": moduleText });
  const child = f.add("dep", { exports: "./index.mjs" }, { "index.mjs": moduleText }, parent);
  assert.equal(f.resolve("dep", pathToFileURL(path.join(parent, "index.mjs")).href), fileUrl(path.join(child, "index.mjs")));
});
test("package self references use their own export map", t => {
  const f = fixture(t);
  f.write(path.join(f.root, "package.json"), JSON.stringify({ name: "application", type: "module", exports: { "./api": "./src/app.mjs" } }));
  f.add("application", { exports: { "./api": "./wrong.mjs" } }, { "wrong.mjs": moduleText });
  assert.equal(f.resolve("application/api"), fileUrl(f.entry));
});
test("a blocked self reference cannot escape to node_modules", t => {
  const f = fixture(t);
  f.write(path.join(f.root, "package.json"), JSON.stringify({ name: "application", type: "module", exports: { ".": null } }));
  f.add("application", { exports: "./index.mjs" }, { "index.mjs": moduleText });
  assert.throws(() => f.resolve("application"), codeIs("ERR_PACKAGE_PATH_NOT_EXPORTED"));
});
test("import maps override installed packages and null mappings still block", t => {
  const f = fixture(t);
  f.add("dep", { exports: "./index.mjs" }, { "index.mjs": moduleText });
  assert.equal(f.resolve("dep", f.referrer, { imports: { dep: f.referrer } }), f.referrer);
  assert.throws(() => f.resolve("dep", f.referrer, { imports: { dep: null } }), /blocked import specifier/);
});
test("scoped import maps still outrank dependency fallback", t => {
  const f = fixture(t);
  f.add("dep", { exports: "./index.mjs" }, { "index.mjs": moduleText });
  const scope = pathToFileURL(path.dirname(f.entry) + path.sep).href;
  assert.equal(f.resolve("dep", f.referrer, { scopes: { [scope]: { dep: f.referrer } } }), f.referrer);
});
test("installed three never replaces the explicit pinned Three.js package", t => {
  const f = fixture(t);
  f.add("three", { exports: "./wrong.mjs" }, { "wrong.mjs": moduleText });
  const pinned = path.join(f.root, "pinned-three");
  f.write(path.join(pinned, "package.json"), JSON.stringify({ exports: { ".": "./index.mjs", "./addons/*": "./examples/*" } }));
  f.write(path.join(pinned, "index.mjs"), moduleText);
  f.write(path.join(pinned, "examples", "Controls.mjs"), moduleText);
  assert.equal(f.resolve("three", f.referrer, {}, { packageRootUrl: pinned }), pathToFileURL(path.join(pinned, "index.mjs")).href);
  assert.equal(f.resolve("three/addons/Controls.mjs", f.referrer, {}, { packageRootUrl: pinned }), pathToFileURL(path.join(pinned, "examples", "Controls.mjs")).href);
});
test("a missing selected export does not try later fallback files", t => {
  const f = fixture(t);
  f.add("dep", { exports: ["./missing.mjs", "./index.mjs"] }, { "index.mjs": moduleText });
  assert.throws(() => f.resolve("dep"), codeIs("ERR_MODULE_NOT_FOUND"));
});
test("a broken nearest package cannot fall back to an ancestor version", t => {
  const f = fixture(t);
  f.add("dep", { exports: "./index.mjs" }, { "index.mjs": moduleText });
  const parent = f.add("parent", {}, { "index.js": moduleText });
  f.add("dep", { exports: "./missing.mjs" }, {}, parent);
  assert.throws(() => f.resolve("dep", pathToFileURL(path.join(parent, "index.js")).href), codeIs("ERR_MODULE_NOT_FOUND"));
});
test("an existing private file is not exported", t => {
  const f = fixture(t);
  f.add("dep", { exports: { ".": "./index.mjs", "./private/*": null } }, { "index.mjs": moduleText, "private/secret.mjs": moduleText });
  assert.throws(() => f.resolve("dep/private/secret.mjs"), codeIs("ERR_PACKAGE_PATH_NOT_EXPORTED"));
});
test("malformed installed manifests fail with the original source span", t => {
  const f = fixture(t), sourceSpan = { line: 3, column: 8, offset: 50 };
  const directory = f.add("dep", {}, { "index.js": moduleText });
  f.write(path.join(directory, "package.json"), "{ broken");
  assert.throws(() => f.resolve("dep", f.referrer, {}, { span: sourceSpan }), error => {
    assert.ok(codeIs("ERR_INVALID_PACKAGE_CONFIG")(error));
    assert.equal(error.span, sourceSpan); assert.equal(error.specifier, "dep");
    assert.match(error.message, /package\.json/); return true;
  });
});
test("package metadata changes are visible between builds", t => {
  const f = fixture(t), directory = f.add("dep", { exports: "./a.mjs" }, { "a.mjs": moduleText, "b.mjs": moduleText });
  assert.equal(f.resolve("dep"), fileUrl(path.join(directory, "a.mjs")));
  f.write(path.join(directory, "package.json"), JSON.stringify({ name: "dep", type: "module", exports: "./b.mjs" }));
  assert.equal(f.resolve("dep"), fileUrl(path.join(directory, "b.mjs")));
});
test("query and fragment identities are preserved after realpath", t => {
  const f = fixture(t), directory = f.add("dep", { exports: { "./a": "./index.mjs?version=1#one", "./b": "./index.mjs?version=2#two" } }, { "index.mjs": moduleText });
  const canonical = fileUrl(path.join(directory, "index.mjs"));
  assert.equal(f.resolve("dep/a"), canonical + "?version=1#one");
  assert.equal(f.resolve("dep/b"), canonical + "?version=2#two");
});
test("workspace links resolve from their real package and keep one identity", t => {
  const f = fixture(t), workspace = path.join(f.root, "workspace");
  f.write(path.join(workspace, "package.json"), JSON.stringify({ name: "linked", type: "module", exports: "./index.mjs" }));
  f.write(path.join(workspace, "index.mjs"), moduleText);
  fs.mkdirSync(path.join(f.root, "node_modules"), { recursive: true });
  fs.symlinkSync(workspace, path.join(f.root, "node_modules", "linked"), "junction");
  fs.symlinkSync(workspace, path.join(f.root, "node_modules", "alias"), "junction");
  const dependency = f.add("dep", { exports: "./index.mjs" }, { "index.mjs": moduleText }, workspace);
  assert.equal(f.resolve("linked"), fileUrl(path.join(workspace, "index.mjs")));
  assert.equal(f.resolve("alias"), f.resolve("linked"));
  assert.equal(f.resolve("dep", f.resolve("linked")), fileUrl(path.join(dependency, "index.mjs")));
});
test("nested package type overrides are respected", t => {
  const f = fixture(t), directory = f.add("dep", { type: "commonjs", exports: "./esm/index.js" }, { "esm/package.json": '{"type":"module"}', "esm/index.js": moduleText });
  assert.equal(f.resolve("dep"), fileUrl(path.join(directory, "esm", "index.js")));
  f.write(path.join(directory, "esm", "package.json"), '{"type":"commonjs"}');
  assert.throws(() => f.resolve("dep"), codeIs("F3D_PACKAGE_FORMAT"));
});
for (const [name, manifest, filename, code] of [
  ["CommonJS", { type: "commonjs", main: "index.cjs" }, "index.cjs", "F3D_PACKAGE_FORMAT"],
  ["ambiguous js", { type: undefined, main: "index.js" }, "index.js", "F3D_PACKAGE_FORMAT"],
  ["JSON loader", { exports: "./data.json" }, "data.json", "F3D_PACKAGE_FORMAT"],
  ["directory target", { exports: "./folder" }, "folder/index.mjs", "ERR_UNSUPPORTED_DIR_IMPORT"],
]) test(`${name} is refused instead of silently inventing a runtime route`, t => {
  const f = fixture(t); f.add("dep", manifest, { [filename]: moduleText });
  assert.throws(() => f.resolve("dep"), codeIs(code));
});
test("local packages are never looked up for remote referrers", t => {
  const f = fixture(t); f.add("dep", { exports: "./index.mjs" }, { "index.mjs": moduleText });
  assert.throws(() => f.resolve("dep", "https://example.org/main.mjs"), codeIs("F3D_PACKAGE_REFERRER"));
});
test("Node builtins require explicit host adapters", t => {
  const f = fixture(t);
  for (const specifier of ["fs", "node:fs", "fs/promises"]) assert.throws(() => f.resolve(specifier), codeIs("F3D_NODE_BUILTIN"));
  assert.equal(f.resolve("node:fs", f.referrer, { imports: { "node:fs": f.referrer } }), f.referrer);
});
test("missing packages fail with an actionable resolution error", t => {
  const f = fixture(t);
  assert.throws(() => f.resolve("not-installed"), codeIs("ERR_MODULE_NOT_FOUND"));
});
test("malformed package names and private-path traversal are rejected", t => {
  const f = fixture(t); f.add("dep", {}, { "index.js": moduleText });
  for (const name of ["@scope", "@/dep", "@scope/", "de%70", "dep\\secret", "dep/../escape.mjs", "dep/%2e%2e/escape.mjs", "dep/node_modules/other/index.mjs"])
    assert.throws(() => f.resolve(name), IngestionResolutionError);
});
test("URL control characters cannot turn an export into a package escape", t => {
  const f = fixture(t);
  f.add("dep", { exports: "./\t../outside.mjs" }, {});
  f.write(path.join(f.root, "node_modules", "outside.mjs"), moduleText);
  assert.throws(() => f.resolve("dep"), codeIs("ERR_INVALID_PACKAGE_TARGET"));
});
test("legacy subpath resolution does not strip query and fragment identity", t => {
  const f = fixture(t), directory = f.add("dep", {}, { "index.mjs": moduleText });
  const resolved = f.resolve("dep/index.mjs?variant=one#fragment");
  assert.equal(resolved, fileUrl(path.join(directory, "index.mjs")) + "?variant=one#fragment");
  assert.equal(fs.readFileSync(fileURLToPath(resolved), "utf8"), moduleText);
});
