import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { buildModuleGraph } from "../../tools/ingest/module_graph.mjs";

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-reexports-"));
  for (const [name, content] of Object.entries(files)) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, content);
  }
  const url = name => pathToFileURL(path.join(root, name)).href;
  return { root, url, graph: (entry = "app.mjs", options) => buildModuleGraph(path.join(root, entry), options) };
}

for (const declaration of [
  "export * from './leaf.mjs';",
  "export * as values from './leaf.mjs';",
  "export { value as answer } from './leaf.mjs';",
  "export { default } from './leaf.mjs';",
  "export {} from './leaf.mjs';",
]) test(`traverses the static request in ${declaration}`, async () => {
  const f = fixture({ "app.mjs": declaration, "leaf.mjs": "export const value = 42; export default value;" });
  const g = await f.graph(), node = g.modules[f.url("app.mjs")];
  assert.equal(g.summary.total_modules, 2);
  assert.equal(node.static_imports.length, 1);
  assert.equal(node.static_imports[0].resolved_id, f.url("leaf.mjs"));
  assert.equal(node.static_imports[0].is_reexport, true);
  assert.deepEqual(node.static_imports[0].imported_bindings, [], "reexports do not create local bindings");
  assert.equal(node.static_exports[0].resolved_id, f.url("leaf.mjs"));
});

test("a barrel cannot hide native-context escapes or renderer construction from graph routing", async () => {
  const f = fixture({
    "app.mjs": "import { create } from './index.mjs'; export { create };",
    "index.mjs": "export { create } from './graphics/renderer.mjs';",
    "graphics/renderer.mjs": `export function create(canvas, kind, force) {
      canvas.getContext(kind); return new WebGPURenderer({forceWebGL: force});
    }`,
  });
  const g = await f.graph();
  assert.equal(g.summary.total_modules, 3);
  assert.equal(g.summary.total_renderer_construction_sites, 1);
  assert.equal(g.summary.total_unresolved_native_context_access, 1);
  assert.equal(g.summary.total_unresolved_force_webgl, 1);
  assert.equal(g.modules[f.url("graphics/renderer.mjs")].routing_facts.has_native_context_access, true);
});

for (const external of ["https://example.invalid/remote.mjs", "data:text/javascript,export const x=1"]) {
  test(`external reexports invalidate lazy closure transitively: ${external}`, async () => {
    const f = fixture({
      "app.mjs": "export const load = () => import('./index.mjs');",
      "index.mjs": "export * from './barrel.mjs';",
      "barrel.mjs": `export * from ${JSON.stringify(external)};`,
    });
    const g = await f.graph();
    assert.deepEqual(g.external_modules, [external]);
    assert.equal(g.modules[f.url("app.mjs")].dynamic_imports[0].claims_closure, false);
    assert.equal(g.modules[f.url("app.mjs")].dynamic_imports[0].unresolved, false);
  });
}

test("unknown lazy code behind a reexport invalidates every enclosing finite import choice", async () => {
  const f = fixture({
    "app.mjs": "export const load = flag => import(flag ? './index.mjs' : './safe.mjs');",
    "index.mjs": "export * from './hidden.mjs';",
    "hidden.mjs": "export const load = name => import(name);",
    "safe.mjs": "export const value = 42;",
  });
  const g = await f.graph();
  assert.equal(g.summary.unresolved_dynamic_imports, 1);
  assert.equal(g.modules[f.url("app.mjs")].dynamic_imports[0].claims_closure, false);
});

test("missing transitive reexports fail with the original declaration's source span", async () => {
  const f = fixture({ "app.mjs": "// header\nexport * from './missing.mjs';" });
  await assert.rejects(f.graph(), error => {
    assert.equal(error.name, "IngestionResolutionError");
    assert.equal(error.specifier, "./missing.mjs");
    assert.equal(error.referrerUrl, f.url("app.mjs"));
    assert.equal(error.span.start.line, 2);
    assert.equal(error.span.start.offset, 10);
    return true;
  });
});

test("reexport cycles close by canonical identity and preserve native live bindings", async () => {
  const f = fixture({
    "app.mjs": "export { value, bump } from './a.mjs';",
    "a.mjs": "export { value, bump } from './b.mjs'; export * as root from './app.mjs';",
    "b.mjs": "export let value = 41; export function bump() { value++; }",
  });
  const g = await f.graph();
  assert.equal(g.summary.total_modules, 3);
  assert.ok(g.cycles.some(c => c[0] === f.url("app.mjs") && c.at(-1) === f.url("app.mjs")));
  assert.equal(g.modules[f.url("b.mjs")].has_live_bindings, true);
  const app = await import(f.url("app.mjs"));
  app.bump(); assert.equal(app.value, 42);
});

test("identical-byte and query/fragment reexports do not collapse module instances", async () => {
  const code = "export const identity = {};";
  const f = fixture({
    "app.mjs": "export * as a from './leaf.mjs?one#x'; export * as b from './leaf.mjs?two#y'; export * as c from './copy.mjs';",
    "leaf.mjs": code, "copy.mjs": code,
  });
  const g = await f.graph();
  assert.equal(g.summary.total_modules, 4);
  assert.equal(g.summary.identical_content_pairs, 3);
  const app = await import(f.url("app.mjs"));
  assert.notEqual(app.a.identity, app.b.identity); assert.notEqual(app.a.identity, app.c.identity);
});

test("static request edges preserve source order and ordinary import binding metadata", async () => {
  const f = fixture({
    "app.mjs": "export * from './a.mjs'; import value from './b.mjs'; export {value}; export {} from './c.mjs'; import './a.mjs';",
    "a.mjs": "export {};", "b.mjs": "export default 1;", "c.mjs": "throw new Error('graph construction must not execute source');",
  });
  const g = await f.graph(), edges = g.modules[f.url("app.mjs")].static_imports;
  assert.deepEqual(edges.map(e => e.specifier), ["./a.mjs", "./b.mjs", "./c.mjs", "./a.mjs"]);
  assert.deepEqual(edges[1].imported_bindings, [{ local: "value", imported: "default", type: "default" }]);
  assert.equal(g.summary.total_modules, 4);
  assert.equal(g.summary.total_static_imports, 4);
  assert.equal(g.modules[f.url("app.mjs")].static_exports[1].specifier, null);
});

test("HTML scopes apply at the reexporting module, not the importing document", async () => {
  const f = fixture({
    "index.html": `<script type="importmap">{"imports":{"selected":"./wrong.mjs"},"scopes":{"./nested/":{"selected":"./chosen.mjs"}}}</script><script type="module" src="app.mjs"></script>`,
    "app.mjs": "export * from './nested/index.mjs';",
    "nested/index.mjs": "export * from 'selected';",
    "chosen.mjs": "export const value = 42;", "wrong.mjs": "export const value = -1;",
  });
  const g = await f.graph("index.html");
  assert.ok(g.modules[f.url("chosen.mjs")]); assert.equal(g.modules[f.url("wrong.mjs")], undefined);
});

test("blocked reexports honor the source import map", async () => {
  const f = fixture({
    "index.html": `<script type="importmap">{"imports":{"private":null}}</script><script type="module">export * from 'private';</script>`,
  });
  await assert.rejects(f.graph("index.html"), /blocked import specifier/);
});

test("installed package and package-private reexports traverse their actual implementation", async () => {
  const f = fixture({
    "app.mjs": "export * from 'dependency';",
    "node_modules/dependency/package.json": JSON.stringify({ name: "dependency", type: "module", exports: "./index.mjs", imports: { "#actual": "./actual.mjs" } }),
    "node_modules/dependency/index.mjs": "export { answer } from '#actual';",
    "node_modules/dependency/actual.mjs": "export const answer = 42;",
  });
  const g = await f.graph();
  assert.equal(g.summary.total_modules, 3);
  assert.ok(g.modules[f.url("node_modules/dependency/actual.mjs")]);
});

test("reexports retain the pinned Three.js package rather than an installed replacement", async () => {
  const f = fixture({
    "app.mjs": "export * from 'three';",
    "pinned/package.json": '{"exports":"./index.mjs"}', "pinned/index.mjs": "export const REVISION = 'pinned';",
    "node_modules/three/package.json": '{"type":"module","exports":"./wrong.mjs"}', "node_modules/three/wrong.mjs": "export const REVISION = 'wrong';",
  });
  const g = await f.graph("app.mjs", { packageRootUrl: path.join(f.root, "pinned") });
  assert.ok(g.modules[f.url("pinned/index.mjs")]);
  assert.equal(g.modules[f.url("node_modules/three/wrong.mjs")], undefined);
});

test("local exports do not invent dependency requests", async () => {
  const f = fixture({ "app.mjs": "const value = 42; export { value }; export default value;" });
  const g = await f.graph(); assert.equal(g.summary.total_modules, 1);
  assert.deepEqual(g.modules[f.url("app.mjs")].static_imports, []);
});
