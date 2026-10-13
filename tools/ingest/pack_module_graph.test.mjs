import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";
import * as acorn from "acorn";
import { packHtml } from "./pack_html.mjs";

const data = (source) => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const decode = (url) => Buffer.from(url.slice(url.indexOf(",") + 1).split("#")[0], "base64");
// Node supplies native ESM/JSON parsing, linking, import attributes and separate
// worker realms. The resolve hook models ONLY the emitted document import map;
// it does not rewrite source, intercept errors or supply a replacement loader.
const runner = new URL(data(`
  import {parentPort,workerData} from 'node:worker_threads';
  import {registerHooks} from 'node:module';
  registerHooks({resolve(specifier,context,next){
    return next(workerData.imports?.[specifier] ?? specifier,context);
  }});
  try {
    const m=await import(workerData.url);
    parentPort.postMessage({ok:true,value:await m.run()});
  } catch(e) { parentPort.postMessage({ok:false,name:e.name,code:e.code}); }
`));
async function execute(entry) {
  const thread = new Worker(runner, { workerData: entry });
  let timer;
  try {
    return await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error("Module execution timed out")), 10000);
      thread.once("message", resolve);
      thread.once("error", reject);
      thread.once("exit", (code) => reject(new Error(`Exited without a result: ${code}`)));
    });
  } finally {
    clearTimeout(timer);
    await thread.terminate();
  }
}
function fixture(t, lane, files, options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-packed-modules-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, "source"), output = path.join(root, "packed.html");
  const inputs = {
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": lane === "document" ? files["main.mjs"] :
      `export const start=()=>new Worker(new URL('./main.mjs',import.meta.url),{type:'module'});`,
    "settings.json": '{"value":42}',
    ...files,
  };
  for (const [name, bytes] of Object.entries(inputs)) {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
  }
  return {
    root, source, output, inputs,
    original: { url: pathToFileURL(path.join(source, lane === "document" ? "app.mjs" : "main.mjs")).href },
    pack() { return packHtml(path.join(source, "index.html"), output, options); },
    async emitted() {
      const html = fs.readFileSync(output, "utf8");
      const imports = JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)[1]).imports;
      const url = /<script type="module" src="([^"]+)"/.exec(html)[1];
      if (lane === "document") return { url, imports };
      const ast = acorn.parse(decode(url).toString(), { ecmaVersion: "latest", sourceType: "module" });
      const factoryURL = ast.body.find((n) => n.type === "ImportDeclaration").source.value;
      const factory = await import(factoryURL);
      const blob = factory.default();
      assert.equal(factory.default(), blob);
      assert.match(blob, /^blob:/);
      return { url: data(await (await fetch(blob)).text()) };
    },
  };
}
async function equivalent(t, lane, files, expected) {
  const f = fixture(t, lane, files);
  const original = await execute(f.original);
  if (expected !== undefined) assert.deepEqual(original, expected, "reference result");
  f.pack();
  const emitted = await f.emitted();
  for (const [name, bytes] of Object.entries(f.inputs))
    assert.deepEqual(fs.readFileSync(path.join(f.source, name)), Buffer.from(bytes));
  fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
  assert.deepEqual(await execute(emitted), original, "native emitted result without source files");
  return { f, emitted };
}

for (const lane of ["document", "worker"]) {
  test(`${lane}: JSON static/reexport/lazy identity, BOM, duplicate keys and signed zero`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `import config from './settings.json' with {type:'json'};
        import {config as relay} from './relay.mjs';
        export async function run(){
          const lazy=(await import('./settings.json',{with:{type:'json'}})).default;
          const variant=(await import('./settings.json?other#view',{with:{type:'json'}})).default;
          config.added=7;
          return [config===relay,config===lazy,variant!==config,lazy.added,
            variant.added,Object.is(config.zero,-0),config.value,Object.hasOwn(config,'__proto__')];
        }`,
      "relay.mjs": `export {default as config} from './settings.json' with {type:'json'};`,
      "settings.json": Buffer.from('\ufeff{"zero":-0,"value":1,"value":42,"__proto__":{"x":1}}'),
    }, { ok: true, value: [true, true, true, 7, undefined, true, 42, true] });
  });

  test(`${lane}: malformed lazy JSON stays unparsed until the selected import`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){const log=[];
        const load=flag=>import((log.push(flag),flag?'./bad.json':'./settings.json'),{with:{type:'json'}});
        const good=await load(false);let name;try{await load(true);}catch(e){name=e.name;}
        return [good.default.value,log,name];}`, "bad.json": '{"incomplete":',
    }, { ok: true, value: [42, [false, true], "SyntaxError"] });
  });

  test(`${lane}: invalid UTF-8 JSON is left to the native decoder`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){try{await import('./bad.json',{with:{type:'json'}});}
        catch(e){return e.name;}}`, "bad.json": Buffer.from([0xff]),
    }, { ok: true, value: "SyntaxError" });
  });

  test(`${lane}: preserves source/options/attribute side effects exactly once`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){const log=[];
        const m=await import((log.push('source'),'./settings.json'),
          (log.push('options'),{with:(log.push('with'),{type:(log.push('type'),'json')})}));
        return [m.default.value,log];}`,
    }, { ok: true, value: [42, ["source", "options", "with", "type"]] });
  });

  test(`${lane}: preserves throwing import options without eager evaluation`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){const log=[];const stop=()=>{throw new Error('stop')};
        try{await import((log.push('source'),'./settings.json'),
          (log.push('options'),stop(),{with:{type:'json'}}));}catch(e){log.push(e.message);}return log;}`,
    }, { ok: true, value: ["source", "options", "stop"] });
  });

  for (const [label, expression, code] of [
    ["missing JSON attribute", `import('./settings.json')`, "ERR_IMPORT_ATTRIBUTE_MISSING"],
    ["JSON type with JavaScript MIME", `import('./code.mjs',{with:{type:'json'}})`, "ERR_IMPORT_ATTRIBUTE_TYPE_INCOMPATIBLE"],
    ["unsupported extra attribute", `import('./settings.json',{with:{type:'json',extra:'no'}})`, "ERR_IMPORT_ATTRIBUTE_UNSUPPORTED"],
  ]) test(`${lane}: native rejection for ${label}`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){try{await ${expression};return 'incorrect success';}
        catch(e){return [e.name,e.code];}}`,
      // A MIME/type mismatch must fail before attempting this nonexistent edge.
      "code.mjs": `import './not-loaded.mjs';export const value=3;`,
    }, { ok: true, value: ["TypeError", code] });
  });

  test(`${lane}: named JSON exports retain native link errors`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `import {value} from './settings.json' with {type:'json'};export const run=()=>value;`,
    }, { ok: false, name: "SyntaxError", code: undefined });
  });

  test(`${lane}: URL-object imports close executable graphs, not raw file copies`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){const m=await import(new URL('lazy.mjs',import.meta.url));return m.value;}`,
      "lazy.mjs": `import {value} from './state.mjs';export {value};`,
      "state.mjs": `export const value=42;`,
    }, { ok: true, value: 42 });
  });

  test(`${lane}: conditional typed URL imports retain selection and options ordering`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){const log=[];const flag=false;
        const m=await import(new URL((log.push('url'),flag?'bad.json':'settings.json'),import.meta.url),
          (log.push('options'),{with:{type:'json'}}));return [m.default.value,log];}`,
      "bad.json": 'invalid unselected json',
    }, { ok: true, value: [42, ["url", "options"]] });
  });

  test(`${lane}: nested imports in selector side effects remain executable`, async (t) => {
    await equivalent(t, lane, {
      "main.mjs": `export async function run(){const log=[];
        const m=await import((log.push((await import(new URL('./effect.mjs',import.meta.url))).value),
          new URL('./lazy.mjs',import.meta.url)));return [log,m.value];}`,
      "effect.mjs": `export const value='effect';`,
      "lazy.mjs": `export const value=42;`,
    }, { ok: true, value: [["effect"], 42] });
  });

  for (const options of [
    "options", "{...options}", "{with:attrs}", "{with:{...attrs}}",
    "{with:{[key]:'json'}}", "{get with(){return {type:'json'}}}",
    "{with:{type:kind}}", "{with:{type:'css'}}", "{with:{__proto__:attrs}}",
    "{with:{type:'json'},with:{type:'json'}}",
  ]) test(`${lane}: rejects unproved import attributes: ${options}`, (t) => {
    const f = fixture(t, lane, { "main.mjs": `export const run=()=>import('./settings.json',${options});` });
    assert.throws(() => f.pack(), { code: "IMPORT_ATTRIBUTES" });
    assert.equal(fs.existsSync(f.output), false);
  });
}

test("document: URL imports use URL semantics rather than source bare import-map rules", async (t) => {
  const f = fixture(t, "document", {
    "index.html": `<script type="importmap">{"imports":{"lazy.mjs":"./wrong.mjs"}}</script><script type="module" src="app.mjs"></script>`,
    "main.mjs": `export async function run(){return (await import(new URL('lazy.mjs',import.meta.url))).value;}`,
    "lazy.mjs": "export const value=42;", "wrong.mjs": "throw new Error('wrong resolution');",
  });
  f.pack();
  const emitted = await f.emitted();
  assert.ok(Object.keys(emitted.imports).some((key) => key.startsWith("f3d-packed:")));
  fs.renameSync(f.source, path.join(f.root, "unavailable"));
  assert.deepEqual(await execute(emitted), { ok: true, value: 42 });
});

test("document: URL aliases preserve native cyclic module identity", async (t) => {
  await equivalent(t, "document", {
    "main.mjs": `export const token={};export async function run(){
      const m=await import(new URL('./cycle.mjs',import.meta.url));return m.token===token;}`,
    "cycle.mjs": `import {token} from './app.mjs';export {token};`,
  }, { ok: true, value: true });
});

test("worker: cycles still fail explicitly without a production linker shim", (t) => {
  const f = fixture(t, "worker", {
    "main.mjs": `export const run=()=>import(new URL('./cycle.mjs',import.meta.url));`,
    "cycle.mjs": `import './main.mjs';`,
  });
  assert.throws(() => f.pack(), { code: "WORKER_MODULE_CYCLE" });
  assert.equal(fs.existsSync(f.output), false);
});

test("typed resources retain identity/file/byte limits and atomic output", (t) => {
  const files = {
    "main.mjs": `export const run=()=>import('./large.json',{with:{type:'json'}});`,
    "large.json": JSON.stringify({ text: "x".repeat(4096) }),
  };
  for (const options of [{ maxBytes: 1024 }, { maxFiles: 2 }]) {
    const f = fixture(t, "worker", files, options);
    assert.throws(() => f.pack(), { code: "PACK_LIMIT" });
    assert.equal(fs.existsSync(f.output), false);
  }
});
