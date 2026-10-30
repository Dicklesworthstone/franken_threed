import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { Worker as Thread } from "node:worker_threads";
import { buildApplication } from "./build_application.mjs";
import { parseTagAttributes } from "./html_parser.mjs";
import { packHtml } from "./pack_html.mjs";

// Test host only: real native worker threads, ESM, classic script evaluation,
// Wasm compilation and filesystem-backed Response bodies. Production worker
// files execute unchanged. Browser verification additionally exercises native
// Worker/SharedWorker URL resolution and HTTP fetching without this adapter.
const bootstrap = new URL("data:text/javascript;base64," + Buffer.from(`
  import fs from 'node:fs'; import vm from 'node:vm';
  import {parentPort,workerData} from 'node:worker_threads';
  const NativeURL=URL,nativeFetch=fetch;
  const bytes=url=>String(url).startsWith('data:')
    ?Buffer.from(String(url).slice(String(url).indexOf(',')+1).split('#')[0],'base64'):fs.readFileSync(url);
  globalThis.self=globalThis; globalThis.location=new NativeURL(workerData.url);
  globalThis.postMessage=(value,transfer=[])=>parentPort.postMessage(value,transfer);
  globalThis.fetch=(input,options)=>{
    const url=new NativeURL(String(input),workerData.url);
    return url.protocol==='file:'?Promise.resolve(new Response(fs.readFileSync(url),{headers:{
      'content-type':url.pathname.endsWith('.wasm')?'application/wasm':'application/octet-stream'
    }})):nativeFetch(url,options);
  };
  globalThis.importScripts=(...urls)=>{
    if(workerData.kind!=='classic')throw new TypeError('Module workers cannot importScripts');
    for(const input of urls){const url=new NativeURL(input,workerData.url);
      vm.runInThisContext(bytes(url).toString(),{filename:url.href});}
  };
  if(workerData.kind==='classic')vm.runInThisContext(bytes(new NativeURL(workerData.url)).toString(),{filename:workerData.url});
  else await import(workerData.url);
`).toString("base64"));

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-built-workers-"));
  for (const [name, bytes] of Object.entries({
    "package.json": '{"type":"module"}',
    "index.html": '<base href="site/base.html"><script type="module" src="../modules/app.mjs"></script>',
    ...files,
  })) {
    fs.mkdirSync(path.dirname(path.join(root, name)), { recursive: true });
    fs.writeFileSync(path.join(root, name), bytes);
  }
  return root;
}

async function build(root) {
  const result = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  return { ...result, application: await import(pathToFileURL(path.join(result.outDir, result.entryFiles[0])).href) };
}

function workerHost(t, documentBase) {
  const original = globalThis.Worker, workers = [];
  globalThis.Worker = class {
    constructor(input, options = {}) {
      this.url = new URL(String(input), documentBase);
      this.options = options;
      workers.push(this);
      this.ready = (async () => {
        const url = this.url.protocol === "blob:" ? "data:text/javascript;base64," +
          Buffer.from(await (await fetch(this.url)).text()).toString("base64") : this.url.href;
        return this.thread = new Thread(bootstrap, {workerData: {url,kind:options?.type ?? "classic"}});
      })();
      this.result = this.ready.then(() => new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("Worker did not return a result")), 10000);
        this.thread.once("message", (value) => { clearTimeout(timer); resolve(value); });
        this.thread.once("error", (error) => { clearTimeout(timer); reject(error); });
        this.thread.once("exit", (code) => { clearTimeout(timer); if (code !== 0) reject(new Error(`Worker exit ${code}`)); });
      }));
    }
  };
  t.after(async () => {
    if (original === undefined) delete globalThis.Worker;
    else globalThis.Worker = original;
    await Promise.all(workers.map(async worker => (await worker.ready).terminate()));
  });
}

function classicDocument(html, documentBase) {
  const context = vm.createContext({Worker:globalThis.Worker,URL,Blob});
  for (const tag of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
    const attrs = parseTagAttributes(tag[1]);
    const source = attrs.src?.startsWith("data:")
      ? Buffer.from(attrs.src.slice(attrs.src.indexOf(",")+1).split("#")[0],"base64").toString()
      : attrs.src ? fs.readFileSync(new URL(attrs.src,documentBase),"utf8") : tag[2];
    vm.runInContext(source,context);
  }
  return context;
}

test("classic document worker builds execute complete graphs from the document base with native script globals", async (t) => {
  const source = `"use strict";{function Worker(){throw new Error('block-local');}}var runs=(globalThis.runs||0)+1;
    var start=flag=>new Worker((log.push('url'),flag?'./selected.js':'./other.js'),
      {name:(log.push('options'),'decoder')});
    var task=()=>new Worker('./task.mjs',{type:'module'});`;
  const sri = "sha384-" + createHash("sha384").update(source).digest("base64");
  const root = fixture({
    "index.html": `<base href="site/base.html"><script>const seed=21;var log=[];
      function localShadow(Worker){return Worker;}</script>
      <script src="../scripts/app.js" integrity="${sri}" defer data-original="yes"></script>
      <script src="../scripts/app.js" async></script>
      <script>globalThis.run=async()=>[await start(true).result,await task().result,log,runs,seed,this===globalThis];</script>`,
    "scripts/app.js": source,
    "site/selected.js": `importScripts('./lib/value.js');fetch('./settings.json').then(r=>r.json()).then(v=>postMessage([value,v.answer]));`,
    "site/lib/value.js": "self.value=42;",
    "site/settings.json": '{"answer":7}',
    "site/other.js": "postMessage(0);",
    "site/task.mjs": "import data from './settings.json' with {type:'json'};postMessage(data.answer);",
  });
  const built = await buildApplication(path.join(root,"index.html"),path.join(root,"dist"));
  assert.deepEqual(built.entryFiles, []);
  assert.ok(built.emittedFiles.some(name => name.startsWith("f3d-workers/")));
  const html = fs.readFileSync(path.join(built.outDir,"index.html"),"utf8");
  const output = fs.readFileSync(path.join(built.outDir,"scripts/app.js"),"utf8");
  assert.match(html, /defer data-original="yes"/);
  assert.match(html, /src="\.\.\/scripts\/app.js" async/);
  const updated = "sha384-" + createHash("sha384").update(output).digest("base64");
  assert.notEqual(updated,sri);
  assert.ok(html.includes(`integrity="${updated}"`));
  const documentBase = pathToFileURL(path.join(built.outDir,"site/base.html"));
  workerHost(t,documentBase);
  fs.renameSync(path.join(root,"site"),path.join(root,"source-site-unavailable"));
  fs.renameSync(path.join(root,"scripts"),path.join(root,"source-scripts-unavailable"));
  const context = classicDocument(html,documentBase);
  assert.deepEqual(JSON.parse(JSON.stringify(await context.run())),[[42,7],7,["url","options"],2,21,true]);
});

test("inline classic document workers build and pack into executable independent classic and module realms", async (t) => {
  const root = fixture({
    "index.html": `<script>"use strict";{function Worker(){throw new Error('block-local');}}const answer=42;var log=[];globalThis.run=async()=>{
      var a=new Worker((log.push('classic'),'./classic.js'));
      var b=new Worker('./module.mjs',{type:'module',name:(log.push('module'),'json')});
      return [await a.result,await b.result,log,answer,this===globalThis];};</script>`,
    "classic.js": "importScripts('./lib.js','./lib.js');postMessage([count,typeof document]);",
    "lib.js": "self.count=(self.count||0)+1;",
    "module.mjs": "import config from './config.json' with {type:'json'};postMessage([config.answer,typeof document]);",
    "config.json": '{"answer":42}',
  });
  const built = await buildApplication(path.join(root,"index.html"),path.join(root,"dist"));
  const packed = path.join(root,"packed.html"), entry = path.join(built.outDir,"index.html");
  assert.equal(packHtml(entry,packed).workerCount,2);
  const base = pathToFileURL(entry);
  workerHost(t,base);
  for (const file of ["classic.js","lib.js","module.mjs","config.json"])
    fs.renameSync(path.join(root,file),path.join(root,"source-unavailable-"+file));
  const expected = [[2,"undefined"],[42,"undefined"],["classic","module"],42,true];
  const ordinary = classicDocument(fs.readFileSync(entry,"utf8"),base);
  assert.deepEqual(JSON.parse(JSON.stringify(await ordinary.run())),expected);
  fs.renameSync(built.outDir,path.join(root,"build-unavailable"));
  const standalone = classicDocument(fs.readFileSync(packed,"utf8"),pathToFileURL(packed));
  assert.deepEqual(JSON.parse(JSON.stringify(await standalone.run())),expected);
});

test("classic document worker builds retain failed integrity and unclosed host routes", async () => {
  for (const [attributes, setup, worker, source = "globalThis.start=()=>new Worker('./worker.js');"] of [
    ['integrity="sha384-not-the-source"', "", "postMessage(42);"],
    ["", "const Worker=CustomWorker;", "postMessage(42);"],
    ["", "", "onmessage=e=>fetch(e.data).then(r=>postMessage(r.status));"],
    ["", "", "postMessage(42);", "with({Worker:class{constructor(url){this.url=url;}}}){new Worker('./worker.js');}"],
    ["", "", "postMessage(42);", "{function Worker(url){this.url=url;}}new Worker('./worker.js');"],
  ]) {
    const html = `<script>${setup}</script><script src="app.js" ${attributes}></script>`;
    const root = fixture({"index.html":html,"app.js":source,"worker.js":worker});
    const built = await buildApplication(path.join(root,"index.html"),path.join(root,"dist"));
    assert.equal(fs.readFileSync(path.join(built.outDir,"app.js"),"utf8"),source,attributes || setup || worker);
    assert.equal(fs.readFileSync(path.join(built.outDir,"index.html"),"utf8"),html);
    assert.ok(!built.emittedFiles.some(name => name.startsWith("f3d-workers/")));
  }
});

test("classic document host bindings also protect worker calls in bundled module scripts", async () => {
  const root = fixture({"index.html": `<script>{function Worker(url){this.url=url;}}</script>
    <script type="module" src="app.mjs"></script>`,
    "app.mjs": "export const start=()=>new Worker('./worker.js');",
    "worker.js": "postMessage(42);"});
  const built = await buildApplication(path.join(root,"index.html"),path.join(root,"dist"));
  assert.ok(!built.emittedFiles.some(name => name.startsWith("f3d-workers/")));
  assert.match(fs.readFileSync(path.join(built.outDir,built.entryFiles[0]),"utf8"),/new Worker\('\.\/worker\.js'\)/);
});

test("document worker strings preserve base, finite branch effects, options and real classic/Wasm dependencies", async (t) => {
  const root = fixture({
    "modules/app.mjs": `export function start(flag,log){
      const URL=null;
      return new Worker((log.push('url'),flag?'./selected.js?choice=1#worker':'./other.js'),
        {name:(log.push('options'),'decoder')});
    }`,
    "site/selected.js": `importScripts('./helper.js');
      WebAssembly.instantiateStreaming(fetch('./decoder%20%23.wasm?codec=1')).then(({instance})=>
        postMessage({value:instance.exports.run()+self.factor,search:location.search,hash:location.hash}));`,
    "site/helper.js": "globalThis.factor=8;",
    "site/decoder #.wasm": Buffer.from("0061736d010000000105016000017f030201000707010372756e00000a06010400412a0b", "hex"),
    "site/other.js": "postMessage({value:17});",
    "modules/selected.js": "postMessage({value:'wrong module base'});",
  });
  const built = await build(root);
  workerHost(t, pathToFileURL(path.join(built.outDir, "site/base.html")));
  const log = [], selected = built.application.start(true, log), other = built.application.start(false, log);
  assert.deepEqual(log, ["url", "options", "url", "options"]);
  assert.equal(selected.options.name, "decoder");
  assert.ok(selected.url.pathname.startsWith(path.join(built.outDir, "f3d-workers")));
  assert.deepEqual(await selected.result, { value: 50, search: "?choice=1", hash: "#worker" });
  assert.deepEqual(await other.result, { value: 17 });
});

test("document and import.meta worker forms keep distinct bases and execute native module dependencies", async (t) => {
  const root = fixture({
    "index.html": '<base href="site/"><script type="importmap">{"imports":{"./dependency.mjs":"./poison.mjs"}}</script><script type="module" src="../modules/app.mjs"></script>',
    "modules/app.mjs": `export const documentWorker=()=>new Worker('./worker.mjs?kind=document#entry',{type:'module'});
      export const moduleWorker=()=>new Worker(new URL('./worker.mjs?kind=module#entry',import.meta.url),{type:'module'});`,
    "site/worker.mjs": "import {value} from './dependency.mjs'; await Promise.resolve(); postMessage({value,search:location.search,hash:location.hash});",
    "site/dependency.mjs": "export const value='document dependency';",
    "site/poison.mjs": "export const value='document import map must not apply to workers';",
    "modules/worker.mjs": "import {value} from './dependency.mjs'; await Promise.resolve(); postMessage({value,search:location.search,hash:location.hash});",
    "modules/dependency.mjs": "export const value='module dependency';",
  });
  const built = await build(root);
  workerHost(t, pathToFileURL(path.join(built.outDir, "site/base.html")));
  const documentWorker = built.application.documentWorker(), moduleWorker = built.application.moduleWorker();
  assert.deepEqual(await documentWorker.result, { value: "document dependency", search: "?kind=document", hash: "#entry" });
  assert.deepEqual(await moduleWorker.result, { value: "module dependency", search: "?kind=module", hash: "#entry" });
  assert.notEqual(documentWorker.url.href, moduleWorker.url.href);
});

test("equivalent SharedWorker URL forms retain one output identity and their native name overloads", async (t) => {
  const root = fixture({
    "modules/app.mjs": `export function localURL(){const URL=null;return URL;}
      export const first=()=>new SharedWorker('./shared.js?realm=one#entry','decoder');
      export const second=()=>new SharedWorker('./shared.js?realm=one#entry',{name:'decoder'});
      export const moduleForm=()=>new SharedWorker(new URL('../site/shared.js?realm=one#entry',import.meta.url),'decoder');
      export const distinct=()=>new SharedWorker('./shared.js?realm=two#entry','decoder');`,
    "site/shared.js": "let count=0; onconnect=e=>e.ports[0].postMessage(++count);",
  });
  const built = await build(root), original = globalThis.SharedWorker;
  const documentBase = pathToFileURL(path.join(built.outDir, "site/base.html"));
  globalThis.SharedWorker = class {
    constructor(input, options) { this.url = new URL(String(input), documentBase); this.options = options; }
  };
  t.after(() => { if (original === undefined) delete globalThis.SharedWorker; else globalThis.SharedWorker = original; });
  const first = built.application.first(), second = built.application.second(), moduleForm = built.application.moduleForm();
  assert.equal(first.url.href, second.url.href);
  assert.equal(first.url.href, moduleForm.url.href);
  assert.equal(first.options, "decoder");
  assert.deepEqual(second.options, { name: "decoder" });
  assert.notEqual(first.url.href, built.application.distinct().url.href);
  assert.match(fs.readFileSync(first.url, "utf8"), /onconnect/);
});

test("open worker graphs and host-provided URLs retain original document routes without partial relocation", async () => {
  const root = fixture({
    "modules/app.mjs": `export const dynamic=()=>new Worker('./dynamic.js');
      export const missing=()=>new Worker('./server-generated.js');
      export const external=()=>new Worker('https://example.invalid/worker.js');
      export const originRoot=()=>new Worker('/host/worker.js');
      export const nested=()=>new Worker('./nested.js');
      export const xhr=()=>new Worker('./xhr.js');
      export const aliased=()=>new Worker('./aliased.js');
      export const escaped=()=>new Worker('./escaped.js');
      export const remote=()=>new Worker('./remote.mjs',{type:'module'});`,
    "site/dynamic.js": "const name='./runtime.json'; fetch(name).then(response=>response.json()).then(postMessage);",
    "site/nested.js": "onmessage=e=>new Worker(e.data);",
    "site/xhr.js": "const xhr=new XMLHttpRequest();xhr.open('GET','./data.bin');xhr.onload=()=>postMessage(xhr.responseText);xhr.send();",
    "site/aliased.js": "const get=fetch;get('./data.bin').then(response=>response.text()).then(postMessage);",
    "site/escaped.js": "const host=self;host.fetch('./data.bin').then(response=>response.text()).then(postMessage);",
    "site/remote.mjs": "import 'https://example.invalid/runtime.mjs';",
    "site/data.bin": "runtime server response",
  });
  const built = await build(root), code = fs.readFileSync(path.join(built.outDir, built.entryFiles[0]), "utf8");
  for (const url of ["./dynamic.js", "./server-generated.js", "https://example.invalid/worker.js", "/host/worker.js", "./nested.js", "./xhr.js", "./aliased.js", "./escaped.js", "./remote.mjs"]) assert.ok(code.includes(url));
  assert.equal(built.emittedFiles.some((file) => file.startsWith("f3d-workers/")), false);
});

test("a previously emitted open module worker does not admit the equivalent document-string route", async () => {
  const root = fixture({
    "modules/app.mjs": `export const moduleForm=()=>new Worker(new URL('../site/open.js',import.meta.url));
      export const documentForm=()=>new Worker('./open.js');`,
    "site/open.js": "const resource='./runtime.json'; fetch(resource).then(postMessage);",
  });
  const built = await build(root), code = fs.readFileSync(path.join(built.outDir, built.entryFiles[0]), "utf8");
  assert.ok(code.includes("'./open.js'") || code.includes('"./open.js"'));
  assert.ok(built.emittedFiles.some((file) => file.startsWith("f3d-workers/")));
});

test("library hosts and user-defined Worker constructors keep source string semantics", async () => {
  const root = fixture({
    "modules/app.mjs": "export class Worker {constructor(url){this.url=url;}} export const create=()=>new Worker('./custom-resource');",
    "library.mjs": "export const start=()=>new Worker('./host-worker.js');",
    "host-worker.js": "postMessage('one possible host');",
  });
  const built = await build(root);
  assert.equal(built.application.create().url, "./custom-resource");
  assert.equal(built.emittedFiles.some((file) => file.startsWith("f3d-workers/")), false);
  const library = await buildApplication(path.join(root, "library.mjs"), path.join(root, "library-dist"));
  assert.equal(library.emittedFiles.some((file) => file.startsWith("f3d-workers/")), false);
});
