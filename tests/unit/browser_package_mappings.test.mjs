import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { resolveModuleSpecifier } from "../../tools/ingest/resolver.mjs";
import { IngestionResolutionError } from "../../tools/ingest/types.mjs";

const client = "export let value = 41; export function increment() { ++value; }\n";
const server = "throw new Error('server implementation executed');\n";
const codeIs = code => error => error instanceof IngestionResolutionError && error.code === code;
function fixture(browser = {}, extra = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-browser-map-"));
  function write(name, text) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true }); fs.writeFileSync(filename, text);
    return pathToFileURL(filename).href;
  }
  const pkg = "node_modules/dep/";
  const metadata = { name: "dep", type: "module", main: "src/main.mjs", browser, ...extra };
  const manifest = value => write(pkg + "package.json", JSON.stringify(value));
  write("package.json", '{"type":"module"}');
  const app = write("app.mjs", ""); manifest(metadata);
  const referrer = write(pkg + "src/main.mjs", "export {};\n");
  const browserUrl = write(pkg + "browser.mjs", client);
  const url = name => new URL(name, pathToFileURL(path.join(root, pkg) + path.sep)).href;
  const resolve = (name, from = referrer, map = {}, options = {}) => resolveModuleSpecifier(name, from, map, options);
  return { root, pkg, metadata, manifest, write, app, referrer, browserUrl, url, resolve };
}

for (const [name, browser, request] of [
  ["package-relative file", { "./server.mjs": "./browser.mjs" }, "../server.mjs"],
  ["missing dependency", { unavailable: "./browser.mjs" }, "unavailable"],
  ["Node builtin", { fs: "./browser.mjs" }, "fs"],
  ["node-scheme builtin", { "node:fs": "./browser.mjs" }, "node:fs"],
  ["private alias", { "#host": "./browser.mjs" }, "#host"],
  ["own constructor property", { constructor: "./browser.mjs" }, "constructor"],
  ["own __proto__ property", JSON.parse('{"__proto__":"./browser.mjs"}'), "__proto__"],
]) test(`${name} is replaced without requiring the server target to exist`, () => {
  const f = fixture(browser); assert.equal(f.resolve(request), f.browserUrl);
});

test("file mappings apply from nested source directories and absolute file URLs", () => {
  const f = fixture({ "./server.mjs": "./browser.mjs" });
  const nested = f.write(f.pkg + "src/nested/entry.mjs", "");
  assert.equal(f.resolve("../../server.mjs", nested), f.browserUrl);
  assert.equal(f.resolve(f.url("server.mjs")), f.browserUrl);
});
for (const request of ["fs", "node:fs", "unused", "../server.mjs"]) {
  test(`explicit false suppresses ${request} without inventing named exports`, async () => {
    const key = request.startsWith("../") ? "./server.mjs" : request;
    const f = fixture({ [key]: false }); f.write(f.pkg + "server.mjs", server);
    const namespace = await import(f.resolve(request));
    assert.deepEqual(Object.keys(namespace), ["default"]); assert.deepEqual(namespace.default, {});
  });
}
test("legacy main and exact public subpaths use the destination package's file replacements", () => {
  const f = fixture({ "./server.cjs": "./browser.mjs" }, { main: "server.cjs" });
  assert.equal(f.resolve("dep", f.app), f.browserUrl);
  assert.equal(f.resolve("dep/server.cjs", f.app), f.browserUrl);
});
test("browser objects can replace the declared module entry", () => {
  const f = fixture({ "./server.js": "./browser.mjs" }, { module: "server.js", main: "server.cjs" });
  assert.equal(f.resolve("dep", f.app), f.browserUrl);
});
test("exports remain authoritative even when browser objects replace internal files", () => {
  const f = fixture({ "./src/main.mjs": "./browser.mjs" }, { exports: { ".": "./src/main.mjs" } });
  assert.equal(f.resolve("dep", f.app), f.referrer);
  assert.throws(() => f.resolve("dep/browser.mjs", f.app), codeIs("ERR_PACKAGE_PATH_NOT_EXPORTED"));
  assert.equal(f.resolve("./main.mjs"), f.browserUrl);
});
test("a missing exported file cannot fall back to browser mappings", () => {
  const f = fixture({ "./missing.mjs": "./browser.mjs" }, { exports: "./missing.mjs" });
  assert.throws(() => f.resolve("dep", f.app), codeIs("ERR_MODULE_NOT_FOUND"));
});
test("relative #imports and direct imports share the replaced file", () => {
  const f = fixture({ "./server.mjs": "./browser.mjs" }, { imports: { "#host": "./server.mjs" } });
  assert.equal(f.resolve("#host"), f.resolve("../server.mjs"));
});
test("replacement dependencies resolve from the owner and do not reapply its source map", () => {
  const f = fixture({ original: "replacement", replacement: "./wrong.mjs" });
  f.write("node_modules/replacement/package.json", '{"type":"module","main":"wrong.mjs"}');
  f.write("node_modules/replacement/wrong.mjs", server);
  f.write(f.pkg + "node_modules/replacement/package.json", '{"type":"module","main":"index.mjs"}');
  const expected = f.write(f.pkg + "node_modules/replacement/index.mjs", client);
  assert.equal(f.resolve("original"), expected);
});
test("HTML import-map precedence and null blocking are preserved", () => {
  const f = fixture({ fs: "./browser.mjs" });
  assert.equal(f.resolve("fs", f.referrer, { imports: { fs: f.app } }), f.app);
  assert.throws(() => f.resolve("fs", f.referrer, { imports: { fs: null } }), /blocked import specifier/);
});
test("replacement dependencies retain the document map base and scoped adapters", () => {
  const f = fixture({ original: "replacement" });
  const document = f.write("site/index.html", "");
  const mapped = f.write("site/adapter.mjs", client);
  const map = { scopes: { "../node_modules/dep/": { replacement: "./adapter.mjs" } } };
  assert.equal(f.resolve("original", f.referrer, map, { mapBaseUrl: document }), mapped);
});
test("blocked replacement dependency cannot execute the original dependency", () => {
  const f = fixture({ original: "replacement" });
  assert.throws(() => f.resolve("original", f.referrer, { imports: { replacement: null } }), /blocked import specifier/);
});
test("browser aliases targeting Three.js keep the pin and cannot override direct Three imports", () => {
  const f = fixture({ graphics: "three", three: "./browser.mjs" });
  f.write("pinned/package.json", '{"exports":"./index.mjs"}');
  const pinned = f.write("pinned/index.mjs", client);
  const options = { packageRootUrl: path.join(f.root, "pinned") };
  assert.equal(f.resolve("graphics", f.referrer, {}, options), pinned);
  assert.equal(f.resolve("three", f.referrer, {}, options), pinned);
});
for (const target of [null, true, 17, [], {}, "", "../outside.mjs", "/outside.mjs", "https://example.org/module.mjs", "./%2e%2e/outside.mjs", "./node_modules/other/index.mjs", "./\t../outside.mjs"]) {
  test(`invalid browser target ${JSON.stringify(target)} fails explicitly`, () => {
    const f = fixture({ fs: target });
    assert.throws(() => f.resolve("fs"), codeIs("ERR_INVALID_PACKAGE_TARGET"));
  });
}
for (const [target, file, code] of [
  ["./missing.mjs", null, "ERR_MODULE_NOT_FOUND"],
  ["./directory", "directory/index.mjs", "ERR_UNSUPPORTED_DIR_IMPORT"],
  ["./server.cjs", "server.cjs", "F3D_PACKAGE_FORMAT"],
  ["./data.json", "data.json", "F3D_PACKAGE_FORMAT"],
]) test(`selected ${target} never falls back to the original module`, () => {
  const f = fixture({ "./original.mjs": target });
  f.write(f.pkg + "original.mjs", client); if (file) f.write(f.pkg + file, client);
  assert.throws(() => f.resolve("../original.mjs"), codeIs(code));
});
test("unmapped Node builtins and inherited property names remain unavailable", () => {
  const f = fixture({});
  assert.throws(() => f.resolve("fs"), codeIs("F3D_NODE_BUILTIN"));
  assert.throws(() => f.resolve("toString"), codeIs("ERR_MODULE_NOT_FOUND"));
});
test("nearest package scope prevents parent and application mappings from leaking", () => {
  const f = fixture({ fs: false });
  const nested = f.write(f.pkg + "src/nested/entry.mjs", "");
  f.write(f.pkg + "src/nested/package.json", '{"type":"module"}');
  assert.throws(() => f.resolve("fs", nested), codeIs("F3D_NODE_BUILTIN"));
  f.manifest({ ...f.metadata, browser: {} });
  f.write("package.json", '{"type":"module","browser":{"fs":false}}');
  assert.throws(() => f.resolve("fs"), codeIs("F3D_NODE_BUILTIN"));
});
test("workspace referrers use their real package's browser map", () => {
  const f = fixture({ "./server.mjs": "./browser.mjs" });
  fs.symlinkSync(path.join(f.root, f.pkg), path.join(f.root, "node_modules/alias"), "junction");
  const linked = pathToFileURL(path.join(f.root, "node_modules/alias/src/main.mjs")).href;
  assert.equal(f.resolve("../server.mjs", linked), f.browserUrl);
});
test("pathname mappings preserve query/fragment identities, and exact keys win", () => {
  const f = fixture({ "./server.mjs": "./browser.mjs", "./server.mjs?v=2#two": "./exact.mjs" });
  const exact = f.write(f.pkg + "exact.mjs", client);
  assert.equal(f.resolve("../server.mjs?v=1#one"), f.browserUrl + "?v=1#one");
  assert.equal(f.resolve("../server.mjs?v=2#two"), exact);
});
test("replacement suffixes override inherited suffixes", () => {
  const f = fixture({ "./server.mjs": "./browser.mjs?fixed#fixed" });
  assert.equal(f.resolve("../server.mjs?original#original"), f.browserUrl + "?fixed#fixed");
});
test("rebuilds see changed browser mappings without a stale global cache", () => {
  const f = fixture({ fs: "./browser.mjs" }); assert.equal(f.resolve("fs"), f.browserUrl);
  f.manifest({ ...f.metadata, browser: { fs: false } }); assert.notEqual(f.resolve("fs"), f.browserUrl);
});
test("errors keep the original request, referrer and source span", () => {
  const f = fixture({ fs: "unavailable" }), span = { line: 2, column: 9, offset: 34 };
  assert.throws(() => f.resolve("fs", f.referrer, {}, { span }), error => {
    assert.ok(codeIs("ERR_MODULE_NOT_FOUND")(error)); assert.equal(error.specifier, "fs");
    assert.equal(error.referrerUrl, f.referrer); assert.equal(error.span, span); return true;
  });
});
test("cyclic cross-package entry replacements stop with a resolution error", () => {
  const f = fixture({ "./src/main.mjs": "other" });
  f.write("node_modules/other/package.json", '{"type":"module","main":"index.mjs","browser":{"./index.mjs":"dep"}}');
  assert.throws(() => f.resolve("dep", f.app), codeIs("F3D_BROWSER_MAPPING_CYCLE"));
});
test("remote referrers do not consume local package browser maps", () => {
  const f = fixture({ fs: false });
  assert.equal(f.resolve("./remote.mjs", "https://example.org/main.mjs"), "https://example.org/remote.mjs");
  assert.throws(() => f.resolve("local-dep", "https://example.org/main.mjs"), codeIs("F3D_PACKAGE_REFERRER"));
});

test("real ESM linking shares browser replacements across static, private and lazy imports", () => {
  const f = fixture({ "./server.mjs": "./browser.mjs", fs: false, alias: "replacement" }, { imports: { "#host": "./server.mjs" } });
  f.write(f.pkg + "server.mjs", server);
  f.write(f.pkg + "node_modules/replacement/package.json", '{"type":"module","module":"client.mjs","main":"server.cjs"}');
  f.write(f.pkg + "node_modules/replacement/client.mjs", "export const extra = 1;");
  f.write(f.pkg + "node_modules/replacement/server.cjs", server);
  f.write(f.pkg + "src/main.mjs", `import * as host from '../server.mjs'; import * as privateHost from '#host';
import disabled from 'fs'; import { extra } from 'alias'; host.increment();
const lazy = await import('../server.mjs');
export default [host.value, privateHost.value, lazy.value, host === privateHost && host === lazy, Object.keys(disabled).length, extra];`);
  const resolver = new URL("../../tools/ingest/resolver.mjs", import.meta.url).href;
  f.write("loader.mjs", `import { resolveModuleSpecifier } from ${JSON.stringify(resolver)};
export function resolve(name, context, next) {
  if (!context.parentURL || name === 'node:module') return next(name, context);
  return { url: resolveModuleSpecifier(name, context.parentURL), format: 'module', shortCircuit: true };
}`);
  f.write("run.mjs", `import { register } from 'node:module'; register(new URL('./loader.mjs', import.meta.url));
console.log(JSON.stringify((await import('dep')).default));`);
  const child = spawnSync(process.execPath, [path.join(f.root, "run.mjs")], { encoding: "utf8" });
  assert.equal(child.status, 0, child.stderr); assert.deepEqual(JSON.parse(child.stdout), [42, 42, 42, true, 0, 1]);
});

test("standalone package resolution retains cycle detection through private dependency aliases", async () => {
  const { resolveInstalledPackage } = await import("../../tools/ingest/node_package_resolver.mjs");
  const f = fixture({ "./src/main.mjs": "#next" }, { imports: { "#next": "other" } });
  f.write("node_modules/other/package.json", JSON.stringify({ type: "module", main: "index.mjs", browser: { "./index.mjs": "#next" }, imports: { "#next": "dep" } }));
  assert.throws(() => resolveInstalledPackage("dep", f.app), codeIs("F3D_BROWSER_MAPPING_CYCLE"));
});
test("absolute modules still resolve when no valid local package referrer is available", () => {
  assert.equal(resolveModuleSpecifier("data:text/javascript,export{}", "not-a-url"), "data:text/javascript,export{}");
});
test("browser lookup does not change ordinary local URL backslash normalization", () => {
  const f = fixture({}); const target = f.write(f.pkg + "src/nested/module.mjs", client);
  assert.equal(f.resolve("./nested\\module.mjs"), target);
});
test("browser mappings do not implicitly translate ambiguous js", () => {
  const f = fixture({ fs: "./browser.js" }, { type: "commonjs" });
  f.write(f.pkg + "browser.js", client);
  assert.throws(() => f.resolve("fs"), codeIs("F3D_PACKAGE_FORMAT"));
});
test("different inherited URL queries create distinct actual ESM instances", async () => {
  const f = fixture({ "./server.mjs": "./browser.mjs" });
  const a = await import(f.resolve("../server.mjs?one"));
  const b = await import(f.resolve("../server.mjs?two"));
  a.increment(); assert.equal(a.value, 42); assert.equal(b.value, 41); assert.notEqual(a, b);
});
