import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import * as acorn from "acorn";
import * as walk from "acorn-walk";
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

const selectors = (source, sourceType = "module") => {
  const ast = acorn.parse(source, {ecmaVersion: "latest", sourceType});
  const before = JSON.stringify(ast), classify = createDynamicImportClassifier(ast), result = [];
  walk.simple(ast, {
    ImportExpression(node) { result.push(classify(node.source)); },
    NewExpression(node) {
      if (node.callee.name === "URL") result.push(classify(node.arguments[0]));
    },
  });
  assert.equal(JSON.stringify(ast), before, "classification must not rewrite runtime expressions");
  return result;
};

test("private records and arrays close finite module and asset selectors through const aliases", () => {
  const sites = selectors([
    "const viewer='./viewer.mjs', routes={viewer, editor:(observe(), './editor.mjs')};",
    "const alias=routes, choice=select()?'viewer':'editor';",
    "const entries=['./first.mjs','./second.mjs'], index=select()?0:1;",
    "import(alias[choice]);",
    "import(entries[index]);",
    "function nested(){const routes={viewer:'./nested.mjs'};return import(routes.viewer);}",
    "const images={day:'./day.png?cache#view',night:'./night.png'};",
    "const image=images[(observe(), select()?'day':'night')];",
    "const asset=new URL(image,import.meta.url);",
  ].join("\n"));
  assert.deepEqual(sites.map(site => site.candidates), [
    ["./viewer.mjs", "./editor.mjs"], ["./first.mjs", "./second.mjs"],
    ["./nested.mjs"], ["./day.png?cache#view", "./night.png"],
  ]);
  assert.ok(sites.every(site => site.classification === "finite_set"));
});

test("record selections agree with native keys, duplicate data fields and selector effects", () => {
  const cases = [
    ["const routes={entry:'./old.mjs',entry:'./last.mjs'};", "routes.entry"],
    ["const routes={'view-mode':'./viewer.mjs',editor:'./editor.mjs'}, prefix='view';",
      "routes[flag?prefix+'-mode':'editor']"],
    ["const routes=['./first.mjs','./second.mjs'], index=flag?0:1;", "routes[index]"],
    ["const routes={viewer:(log.push('value'),'./viewer.mjs'),editor:'./editor.mjs'};",
      "routes[(log.push('key'),flag?'viewer':'editor')]"],
    ["const routes={viewer:'./viewer.mjs'}, alias=routes, again=alias;", "again.viewer"],
  ];
  for (const [setup, expression] of cases) {
    const site = selectors(setup + "import(" + expression + ");")[0];
    assert.equal(site.classification, "finite_set", expression);
    const observed = [true, false].map(flag => {
      const log = [];
      const value = new Function("flag", "log", setup + "return " + expression + ";")(flag, log);
      if (expression.includes("log.push")) assert.deepEqual(log, ["value", "key"]);
      return value;
    });
    assert.deepEqual([...new Set(observed)], site.candidates, expression);
  }
});

test("record closure requires every alias to remain private and every use to read an own field", () => {
  const tails = [
    "routes.viewer='./changed.mjs';", "routes['viewer']+='?changed';", "routes.viewer++;",
    "delete routes.viewer;", "({value:routes.viewer}=unknown);", "[routes.viewer]=unknown;",
    "({value:routes.viewer='fallback'}=unknown);",
    "for(routes.viewer of unknown){}", "for(routes.viewer in unknown){}",
    "const alias=routes;alias.viewer='./changed.mjs';", "let alias=routes;",
    "consume(routes);", "globalThis.routes=routes;", "const holder={routes};",
    "const {viewer}=routes;", "const values=[...routes];", "const clone={...routes};",
    "Object.defineProperty(routes,'viewer',{get:opaque});", "Object.setPrototypeOf(routes,unknown);",
    "routes[unknown];", "routes.missing;", "routes.toString();", "routes.valueOf();",
    "routes.viewer();", "routes.viewer?.();", "new routes.viewer();",
    "routes.viewer\x60template\x60;", "routes?.viewer;", "const expose=()=>routes;",
    "export {routes};", "export {routes as default};", "export default routes;",
    "export const alias=routes;", "eval(code);", "function later(){eval(code)}",
  ];
  for (const tail of tails) {
    assert.equal(selectors("const routes={viewer:'./viewer.mjs'};import(routes.viewer);" + tail)[0].classification,
      "nonliteral", tail);
  }
  assert.equal(selectors("export const routes={viewer:'./viewer.mjs'};import(routes.viewer);")[0].classification,
    "nonliteral");
  for (const tail of ["routes.push('./changed.mjs');", "routes.sort();", "routes[0]='./changed.mjs';",
    "routes.length=0;", "routes[Symbol.iterator]();", "for(const value of routes){}"]) {
    assert.equal(selectors("const routes=['./viewer.mjs'];import(routes[0]);" + tail)[0].classification,
      "nonliteral", tail);
  }
});

test("record selectors never infer prototype values, holes, coercions or accessor results", () => {
  const sources = [
    "const routes={viewer:'./a.mjs'};import(routes[flag?'viewer':'missing']);",
    "const routes={viewer:'./a.mjs'};import(routes[key]);",
    "const routes={get viewer(){return './a.mjs'}};import(routes.viewer);",
    "const routes={viewer(){return './a.mjs'}};import(routes.viewer);",
    "const routes={...unknown,viewer:'./a.mjs'};import(routes.viewer);",
    "const routes={['view'+'er']:'./a.mjs'};import(routes.viewer);",
    "const routes={__proto__:{viewer:'./a.mjs'}};import(routes.viewer);",
    "const routes={['__proto__']:'./a.mjs'};import(routes['__proto__']);",
    "const routes={viewer:unknown};import(routes.viewer);",
    "const routes=['./a.mjs',,'./b.mjs'];import(routes[0]);",
    "const routes=['./a.mjs',...unknown];import(routes[0]);",
    "const routes=['./a.mjs'];import(routes[flag?0:1]);",
    "const routes={2:'./actual.mjs',11:'./wrong.mjs'}, one=1;import(routes[one+one]);",
    "const routes={viewer:'./a.mjs'}, key=routes.viewer;import(routes[key]);",
    "const routes={viewer:'./a.mjs'};function run(routes){return import(routes.viewer)}",
    "const routes={viewer:'./a.mjs'};function run(){import(routes.viewer);var routes;}",
    "const routes={viewer:'./a.mjs'};try{}catch(routes){import(routes.viewer)}",
  ];
  for (const source of sources) assert.equal(selectors(source)[0].classification, "nonliteral", source);
  assert.equal(selectors("const routes={viewer:'./a.mjs'};import(routes.viewer);", "script")[0].classification,
    "nonliteral", "sloppy script bindings stay conservative");
});

test("record selection preserves lexical scope and does not evaluate temporal-dead-zone reads", () => {
  const sites = selectors([
    "const routes={viewer:'./outer.mjs'};",
    "function run(value=import(routes.viewer)){var routes;return import(routes.viewer)}",
    "switch(import(routes.viewer)){case 0:const routes={viewer:'./case.mjs'};import(routes.viewer)}",
    "function pending(){const result=import(table.viewer);const table={viewer:'./later.mjs'};return result}",
  ].join("\n"));
  assert.deepEqual(sites.map(site => site.candidates),
    [["./outer.mjs"], undefined, ["./outer.mjs"], ["./case.mjs"], ["./later.mjs"]]);
  assert.throws(() => new Function("const result=table.viewer;const table={viewer:'./later.mjs'};return result")(),
    ReferenceError, "the unchanged source lookup still fails before initialization");
});

test("private record selection keeps alias cycles, depth and candidate expansion bounded", () => {
  assert.equal(selectors("const a={path:b.path},b={path:a.path};import(a.path);")[0].classification, "nonliteral");
  const chain = ["const r0={path:'./a.mjs'};", ...Array.from({length:70}, (_, i) =>
    "const r" + (i+1) + "=r" + i + ";"), "import(r70.path);"].join("\n");
  assert.equal(selectors(chain)[0].classification, "nonliteral");
  const fields = Array.from({length:129}, (_, i) => "v" + i + ":'./m" + i + ".mjs'").join(",");
  assert.equal(selectors("const routes={" + fields + "};import(routes.v0);")[0].classification, "nonliteral");
  const exponential = ["const r0={path:'x'};", ...Array.from({length:20}, (_, i) =>
    "const r" + (i+1) + "={path:r" + i + ".path+r" + i + ".path};"), "import(r20.path);"].join("\n");
  assert.equal(selectors(exponential)[0].classification, "nonliteral");
});
