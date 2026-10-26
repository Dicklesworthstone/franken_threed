import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import * as acorn from "acorn";
import { analyzeModuleAst, createDynamicImportClassifier } from "./ast_analyzer.mjs";
import { buildModuleGraph } from "./module_graph.mjs";

const imports = source => analyzeModuleAst(source, "file:///app/main.mjs").dynamicImports;

test("immutable import selectors close through lexical aliases and finite string expressions", () => {
  const sites = imports(`
    export const directory = './features/';
    const name = choose() ? 'viewer' : 'editor', alias = name;
    export async function load() {
      const entry = (observe(), directory + alias + '.mjs');
      return import(entry);
    }
    export const alternate = () => import(\`\${directory}\${name}.mjs\`);
  `);
  assert.equal(sites.length, 2);
  for (const site of sites) {
    assert.equal(site.classification, "finite_set");
    assert.deepEqual(site.candidates, ["./features/viewer.mjs", "./features/editor.mjs"]);
    assert.equal(site.unresolved, false);
    assert.ok(site.source_span.start.offset < site.source_span.end.offset);
  }
});

test("the nearest immutable declaration wins across nested functions, blocks and loop scopes", () => {
  const sites = imports(`
    const entry = './outer.mjs';
    function outer() {
      const entry = './function.mjs';
      { const entry = './block.mjs'; import(entry); }
      return () => import(entry);
    }
    for (const entry = './loop.mjs'; condition();) import(entry);
    import(entry);
  `);
  assert.deepEqual(sites.map(site => site.candidates),
    [["./block.mjs"], ["./function.mjs"], ["./loop.mjs"], ["./outer.mjs"]]);
});

test("default parameters and switch discriminants use their actual outer scope", () => {
  const sites = imports(`
    const entry = './outer.mjs';
    function load(value = import(entry)) { if (false) { var entry; } return import(entry); }
    switch (import(entry)) { case 0: const entry = './case.mjs'; import(entry); }
  `);
  assert.deepEqual(sites.map(site => site.candidates),
    [["./outer.mjs"], undefined, ["./outer.mjs"], ["./case.mjs"]]);
});

test("mutable, destructured, imported and shadowed selectors never borrow an outer constant", () => {
  const bodies = [
    "let entry = './mutable.mjs'; import(entry);",
    "var entry = './mutable.mjs'; import(entry);",
    "const {entry} = object; import(entry);",
    "function load(entry) { return import(entry); }",
    "function load({entry}) { return import(entry); }",
    "function load() { import(entry); if (false) { var entry; } }",
    "try {} catch (entry) { import(entry); }",
    "for (const entry of list) import(entry);",
    "for (let entry in object) import(entry);",
    "function load() { import(entry); function entry() {} }",
    "const load = function entry() { return import(entry); };",
    "const load = class entry { static value = import(entry); };",
    "class Load { static { if (false) { var entry; } import(entry); } }",
    "const entry = getPath(); import(entry);",
    "const entry = object.path; import(entry);",
  ];
  for (const body of bodies) {
    const sites = imports(`const entry = './outer.mjs'; function scope() { ${body} }`);
    assert.equal(sites.length, 1, body);
    assert.equal(sites[0].classification, "nonliteral", body);
  }
  assert.equal(imports("import entry from './config.mjs'; import(entry);")[0].classification, "nonliteral");
});

test("alias cycles, deep chains and exponential expansions stay bounded", () => {
  assert.equal(imports("const a=b,b=a; import(a);")[0].classification, "nonliteral");
  const deep = ["const v0='./a.mjs';", ...Array.from({length: 70}, (_, index) =>
    `const v${index + 1}=v${index};`), "import(v70);"].join("\n");
  assert.equal(imports(deep)[0].classification, "nonliteral");
  const doubled = ["const v0='x';", ...Array.from({length: 20}, (_, index) =>
    `const v${index + 1}=v${index}+v${index};`), "import(v20);"].join("\n");
  assert.equal(imports(doubled)[0].classification, "nonliteral");
  const choices = Array.from({length: 8}, (_, index) =>
    `const v${index}=choose()?'a':'b';`).join("\n");
  assert.equal(imports(choices + "import(v0+v1+v2+v3+v4+v5+v6+v7);")[0].classification, "nonliteral");
});

test("sloppy-script eval and with cannot invent a closed constant scope", () => {
  const ast = acorn.parse("const entry='./a.mjs'; with(object) { import(entry); }", {
    ecmaVersion: "latest", sourceType: "script",
  });
  const classify = createDynamicImportClassifier(ast);
  assert.equal(classify(ast.body[1].body.body[0].expression.source).classification, "nonliteral");
});

test("module graph closes lazy const routes while retaining URL identity and external taint", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-constant-imports-"));
  const files = {
    "main.mjs": `const entry=flag?'./lazy.mjs':'./lazy.mjs?instance#view';
      const remote='./remote.mjs'; export const load=()=>import(entry);
      export const external=()=>import(remote);`,
    "lazy.mjs": "export {value} from './shared.mjs';",
    "shared.mjs": "export let value=1; export const update=()=>++value;",
    "remote.mjs": "import 'https://example.invalid/live.mjs';",
  };
  for (const [name, source] of Object.entries(files)) fs.writeFileSync(path.join(root, name), source);
  const graph = await buildModuleGraph(path.join(root, "main.mjs"));
  const node = graph.modules[graph.root_entries[0]];
  assert.equal(graph.summary.total_modules, 5);
  assert.equal(node.dynamic_imports[0].claims_closure, true);
  assert.equal(node.dynamic_imports[0].resolved_targets.length, 2);
  assert.notEqual(...node.dynamic_imports[0].resolved_targets.map(value => value.resolved_id));
  assert.equal(node.dynamic_imports[1].unresolved, false);
  assert.equal(node.dynamic_imports[1].claims_closure, false);
  assert.equal(graph.modules[pathToFileURL(path.join(root, "shared.mjs")).href].has_live_bindings, true);
});
