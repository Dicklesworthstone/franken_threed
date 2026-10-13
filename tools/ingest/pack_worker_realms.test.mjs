import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { registerHooks } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Worker as Thread, isMainThread, parentPort, workerData } from "node:worker_threads";
import { packHtml } from "./pack_html.mjs";

const dataURL = (source) => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const decode = (url) => Buffer.from(url.slice(url.indexOf(",") + 1).split("#")[0], "base64").toString();
const digest = (value) => createHash("sha256").update(value).digest("hex");

// TEST HOST ONLY: real Node threads, native ESM/JSON/Wasm and transfer lists.
// The adapter supplies browser-shaped events and synchronous file/data scripts.
// Blob URLs are read in their actual creating realm before sending source to a
// child thread, since Node's Blob registry is thread-local. Production output
// is executed verbatim; no Worker or loader shim is added to the packed HTML.
if (!isMainThread) {
  const nativeFetch = globalThis.fetch, children = new Set(), listeners = new Set(), inbox = [];
  let ready = false, start;
  const started = new Promise((resolve) => { start = resolve; });
  function dispatch(value) {
    const event = { data: value };
    globalThis.onmessage?.(event);
    for (const listener of listeners) listener(event);
  }
  globalThis.self = globalThis;
  globalThis.onmessage = null;
  globalThis.addEventListener = (name, listener) => { if (name === "message") listeners.add(listener); };
  globalThis.removeEventListener = (name, listener) => { if (name === "message") listeners.delete(listener); };
  globalThis.postMessage = (value, transfer = []) => parentPort.postMessage({ event: "message", value }, transfer);
  globalThis.fetch = async (input, options) => {
    const url = new URL(String(input), workerData.entryUrl);
    if (url.protocol !== "file:") return nativeFetch(url, options);
    const mime = url.pathname.endsWith(".wasm") ? "application/wasm" :
      url.pathname.endsWith(".json") ? "application/json" : "application/octet-stream";
    return new Response(fs.readFileSync(url), { headers: { "content-type": mime } });
  };
  function script(source, url) {
    vm.runInThisContext(source, { filename: url,
      importModuleDynamically: vm.constants.USE_MAIN_CONTEXT_DEFAULT_LOADER });
  }
  globalThis.importScripts = (...inputs) => {
    const urls = inputs.map(String);
    if (workerData.kind !== "classic") throw new TypeError("Module workers cannot importScripts");
    for (const value of urls) {
      const url = new URL(value, workerData.entryUrl);
      script(url.protocol === "data:" ? decode(url.href) : fs.readFileSync(url, "utf8"), url.href);
    }
  };
  class WorkerHost {
    constructor(input, options = {}) {
      this.url = new URL(String(input), workerData.entryUrl);
      this.options = options;
      this.listeners = new Set();
      let settled;
      this.ready = new Promise((resolve) => { settled = resolve; });
      this.thread = new Thread(new URL(import.meta.url), { workerData: {
        entryUrl: this.url.href, kind: options?.type ?? "classic",
      } });
      children.add(this);
      this.thread.on("message", ({ event, value }) => {
        if (event === "message") {
          const message = { data: value };
          this.onmessage?.(message);
          for (const listener of this.listeners) listener(message);
        } else if (event === "ready") settled();
        else if (event === "failure") { settled(); this.onerror?.(new Error(value)); }
      });
      this.thread.on("error", (error) => { settled(); this.onerror?.(error); });
      this.thread.on("exit", settled);
      const load = async () => {
        if (this.url.protocol === "blob:")
          return { source: await (await nativeFetch(this.url)).text() };
        return { url: this.url.href };
      };
      load().then((entry) => this.thread.postMessage({ event: "start", ...entry }),
        (error) => this.onerror?.(error));
    }
    addEventListener(name, listener) { if (name === "message") this.listeners.add(listener); }
    postMessage(value, transfer = []) { this.thread.postMessage({ event: "message", value }, transfer); }
    terminate() {
      // Avoid a Node/V8 isolate-disposal race with native module fulfillment.
      // Tests still transfer/post synchronously; only host teardown waits for
      // the child harness to finish its initial module evaluation.
      return this.ready.then(() => this.thread.terminate()).finally(() => children.delete(this));
    }
  }
  globalThis.Worker = WorkerHost;
  parentPort.on("message", (message) => {
    if (message.event === "start") start(message);
    else if (message.event === "message") {
      if (ready) dispatch(message.value); else inbox.push(message.value);
    }
  });
  try {
    const entry = workerData.driver ? workerData : await started;
    if (workerData.driver) {
      registerHooks({ resolve(specifier, context, next) {
        return next(workerData.imports?.[specifier] ?? specifier, context);
      } });
      const main = await import(entry.url);
      parentPort.postMessage({ event: "result", value: await main.run() });
      await Promise.all([...children].map((child) => child.terminate()));
      parentPort.close();
    } else {
      if (workerData.kind === "classic") {
        const source = entry.source ?? (entry.url.startsWith("data:") ? decode(entry.url) : fs.readFileSync(new URL(entry.url), "utf8"));
        script(source, entry.url ?? workerData.entryUrl);
      } else await import(entry.url ?? dataURL(entry.source));
      ready = true;
      for (const value of inbox) dispatch(value);
      setImmediate(() => parentPort.postMessage({ event: "ready" }));
    }
  } catch (error) {
    parentPort.postMessage({ event: "failure", value: error.stack });
    await Promise.all([...children].map((child) => child.terminate()));
    parentPort.close();
  }
} else {
  function fixture(t, files, options = {}) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-pack-realms-"));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const source = path.join(root, "source"), output = path.join(root, "packed.html");
    const inputs = { "index.html": '<script type="module" src="app.mjs"></script>', ...files };
    for (const [name, value] of Object.entries(inputs)) {
      const file = path.join(source, name);
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, value);
    }
    return { root, source, output, inputs,
      pack() { return packHtml(path.join(source, "index.html"), output, options); },
      original: { driver: true, kind: "document", entryUrl: pathToFileURL(path.join(source, "index.html")).href,
        url: pathToFileURL(path.join(source, "app.mjs")).href },
      emitted() {
        const html = fs.readFileSync(output, "utf8");
        return { driver: true, kind: "document", entryUrl: pathToFileURL(output).href,
          url: /<script type="module" src="([^"]+)"/.exec(html)[1],
          imports: JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)[1]).imports };
      },
    };
  }
  async function execute(entry) {
    const thread = new Thread(new URL(import.meta.url), { workerData: entry });
    let timer, result, received = false;
    try {
      return await new Promise((resolve, reject) => {
        timer = setTimeout(() => reject(new Error("Worker graph execution timed out")), 10000);
        thread.on("message", ({ event, value }) => {
          if (event === "result") { result = value; received = true; }
          else if (event === "failure") reject(new Error(value));
        });
        thread.once("error", reject);
        thread.once("exit", (code) => code === 0 && received ? resolve(result) :
          reject(new Error(`Driver exited without result: ${code}`)));
      });
    } finally { clearTimeout(timer); await thread.terminate(); }
  }
  async function equivalent(t, files, expected) {
    const f = fixture(t, files);
    assert.deepEqual(await execute(f.original), expected, "original application result");
    const report = f.pack(), emitted = f.emitted();
    for (const [name, bytes] of Object.entries(f.inputs))
      assert.deepEqual(fs.readFileSync(path.join(f.source, name)), Buffer.from(bytes));
    fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
    assert.deepEqual(await execute(emitted), expected, "packed result with source directory unavailable");
    return { f, report, emitted };
  }
  const driver = (body) => `export async function run(){${body}}`;
  const receive = `const receive=w=>new Promise((resolve,reject)=>{w.onmessage=e=>{w.terminate();resolve(e.data)};w.onerror=reject;});`;

  for (const outer of ["classic", "module"]) for (const inner of ["classic", "module"]) {
    test(`native threads: ${outer} parent constructs a ${inner} child and transfers buffers`, async (t) => {
      await equivalent(t, {
        "app.mjs": driver(`${receive}const w=new Worker(new URL('./parent.mjs',import.meta.url),{type:'${outer}'});
          const done=receive(w);const a=new Uint8Array([20,22]);w.postMessage(a.buffer,[a.buffer]);return [a.byteLength,await done];`),
        "parent.mjs": `onmessage=e=>{const a=new Uint8Array(e.data);const w=new Worker('./child.mjs',{type:'${inner}'});
          w.onmessage=e=>{w.terminate();postMessage([a.byteLength,e.data])};w.onerror=e=>{throw e};w.postMessage(a.buffer,[a.buffer]);};`,
        "child.mjs": `onmessage=e=>{const a=new Uint8Array(e.data);postMessage(a[0]+a[1]);};`,
      }, [0, [0, 42]]);
    });
  }

  test("native threads: three mixed levels retain independent worker globals", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./top.mjs',{type:'module'}));`),
      "top.mjs": `const results=[];for(let i=0;i<2;i++){const w=new Worker('./middle.js');w.onmessage=e=>{
        results.push(e.data);w.terminate();if(results.length===2)postMessage(results.sort());};}`,
      "middle.js": `self.count=(self.count||0)+1;const w=new Worker('./bottom.mjs',{type:'module'});
        w.onmessage=e=>{w.terminate();postMessage(count+e.data);};`,
      "bottom.mjs": `self.count=(self.count||0)+1;postMessage(count);`,
    }, [2, 2]);
  });

  test("shared ESM helper resolves fetch and nested worker strings from each worker entry", async (t) => {
    const { report } = await equivalent(t, {
      "app.mjs": driver(`${receive}return Promise.all([receive(new Worker('./a/main.mjs',{type:'module'})),receive(new Worker('./b/main.mjs',{type:'module'}))]);`),
      "a/main.mjs": `import '../shared/helper.mjs';`, "b/main.mjs": `import '../shared/helper.mjs';`,
      "shared/helper.mjs": `import config from './module.json' with {type:'json'};
        const value=await (await fetch('./config.json')).json();const w=new Worker('./child.mjs',{type:'module'});
        w.onmessage=e=>{w.terminate();postMessage([config.tag,value.tag,e.data])};`,
      "shared/module.json": '{"tag":"module"}',
      "a/config.json": '{"tag":"a"}', "b/config.json": '{"tag":"b"}',
      "a/child.mjs": `postMessage('a-child');`, "b/child.mjs": `postMessage('b-child');`,
    }, [["module", "a", "a-child"], ["module", "b", "b-child"]]);
    assert.equal(report.workerCount, 4);
  });

  test("conditional Worker URLs preserve selector, URL construction and option ordering", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}const log=[];const flag=false;
        const w=new Worker(new URL((log.push('url'),flag?'./never.mjs':'./worker.mjs'),import.meta.url),
          (log.push('options'),{type:(log.push('type'),'module'),name:(log.push('name'),'decoder')}));
        return [log,await receive(w)];`),
      "never.mjs": `throw new Error('unselected worker must not run');`,
      "worker.mjs": `postMessage(42);`,
    }, [["url", "options", "type", "name"], 42]);
  });

  test("classic root prelude preserves directives and repeated imported scripts", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./root.js'));`),
      "root.js": `"use strict";self.events=[];try{accidentalGlobal=1;}catch(e){events.push(e.name);}
        importScripts('./helpers/spawn.js');importScripts('./helpers/spawn.js');`,
      "helpers/spawn.js": `var n=(self.n||0)+1;var w=new Worker('./leaf.js');w.onmessage=function(e){
        events.push(e.data);if(events.length===3)postMessage([n,events,Object.keys(self).some(k=>k.startsWith('__f3d_worker_'))]);};`,
      "leaf.js": `postMessage(7);`,
    }, [2, ["ReferenceError", 7, 7], false]);
  });

  test("classic ESM helper keeps native importScripts in the shared script environment", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./worker.js'));`),
      "worker.js": `self.events=[];importScripts('./lib/start.js');`,
      "lib/start.js": `import('./helper.mjs');`,
      "lib/helper.mjs": `self.importScripts(new URL('../values.js',import.meta.url));
        const w=new Worker(new URL('./leaf.mjs',import.meta.url),{type:'module'});
        w.onmessage=e=>{w.terminate();postMessage([events,e.data])};`,
      "values.js": `var answer=42;events.push(answer);`, "lib/leaf.mjs": `postMessage('leaf');`,
    }, [[42], "leaf"]);
  });

  test("classic finite importScripts arguments keep selected execution and all argument effects", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./worker.js'));`),
      "worker.js": `self.events=[];const flag=false;importScripts((events.push('first'),flag?'./never.js':'./a.js'),
        (events.push('second'),'./b.js'));postMessage(events);`,
      "a.js": `events.push('a');`, "b.js": `events.push('b');`,
      "never.js": `throw new Error('unselected script');`,
    }, ["first", "second", "a", "b"]);
  });

  test("classic helper factories remain available when the original lexical scope is strict", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./root.js'));`),
      "root.js": `"use strict";importScripts('./helper.js');launch();`,
      "helper.js": `"use strict";function launch(){const w=new Worker('./leaf.js');
        w.onmessage=e=>{w.terminate();postMessage(e.data)};}`,
      "leaf.js": `postMessage(42);`,
    }, 42);
  });

  test("classic helper module URL fetch stays module-relative while string fetch stays entry-relative", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./root.js'));`),
      "root.js": `import('./nested/helper.mjs');`,
      "nested/helper.mjs": `const a=await (await fetch('./config.json')).json();
        const b=await (await globalThis.fetch(new URL('./config.json',import.meta.url))).json();
        postMessage([a.tag,b.tag]);`,
      "config.json": '{"tag":"entry"}', "nested/config.json": '{"tag":"module"}',
    }, ["entry", "module"]);
  });

  test("raw-copy negative control cannot run nested workers after source removal", async (t) => {
    const f = fixture(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./root.js'));`),
      "root.js": `new Worker('./leaf.js');`, "leaf.js": "postMessage(1);",
    });
    const copy = path.join(f.root, "raw-copy.mjs");
    fs.copyFileSync(path.join(f.source,"app.mjs"),copy);
    fs.renameSync(f.source,path.join(f.root,"source-unavailable"));
    await assert.rejects(execute({...f.original,url:pathToFileURL(copy).href,
      entryUrl:pathToFileURL(path.join(f.root,"index.html")).href}), /ENOENT/);
  });

  test("module worker importScripts retains TypeError and never reads missing string targets", async (t) => {
    await equivalent(t, {
      "app.mjs": driver(`${receive}return receive(new Worker('./worker.mjs',{type:'module'}));`),
      "worker.mjs": `const events=[];try{importScripts();}catch(e){events.push(e.name);}
        try{self.importScripts((events.push('a'),'./missing.js'),(events.push('b'),'./also-missing.js'));}
        catch(e){events.push(e.name);}postMessage(events);`,
    }, ["TypeError", "a", "b", "TypeError"]);
  });

  for (const kind of ["classic", "module"]) {
    test(`${kind}: conditional entry-relative GET and native streaming Wasm`, async (t) => {
      await equivalent(t, {
        "app.mjs": driver(`${receive}return receive(new Worker('./worker.mjs',{type:'${kind}'}));`),
        "worker.mjs": `(async()=>{const log=[];const flag=false;const r=await self.fetch(
          (log.push('url'),flag?'./other.wasm':'./empty.wasm'),
          (log.push('options'),{method:(log.push('method'),'get')}));
          const m=await WebAssembly.compileStreaming(r);postMessage([log,WebAssembly.Module.exports(m)]);})();`,
        "empty.wasm": Buffer.from([0,97,115,109,1,0,0,0]), "other.wasm": Buffer.from([0,97,115,109,1,0,0,0]),
      }, [["url", "options", "method"], []]);
    });
  }

  for (const options of ["", ",null", ",{}", ",{name:'decoder'}"]) {
    test(`classic Worker defaults remain native with options ${options || '(absent)'}`, async (t) => {
      await equivalent(t, {
        "app.mjs": driver(`${receive}return receive(new Worker('./worker.js'${options}));`),
        "worker.js": `postMessage(42);`,
      }, 42);
    });
  }

  for (const name of ["Worker", "URL", "Blob", "fetch", "importScripts"]) {
    test(`cross-script host rebinding is rejected before publication: ${name}`, (t) => {
      const f = fixture(t, {
        "app.mjs": `new Worker('./root.js');`,
        "root.js": `importScripts('./rebind.js');new Worker('./child.js');fetch('./config.json');`,
        "rebind.js": `self.${name}=function(){};`, "child.js": "postMessage(1)", "config.json": "{}",
      });
      assert.throws(() => f.pack(), { code: "WORKER_BINDING" });
      assert.equal(fs.existsSync(f.output), false);
    });
  }

  test("ESM helper writes cannot replace a classic realm's host constructors", (t) => {
    const f = fixture(t, {
      "app.mjs": `new Worker('./root.js');`, "root.js": `import('./helper.mjs');new Worker('./child.js');`,
      "helper.mjs": `self.Worker=class {};`, "child.js": "postMessage(1)",
    });
    assert.throws(() => f.pack(), {code:"WORKER_BINDING"});
    assert.equal(fs.existsSync(f.output),false);
  });

  test("classic generated lexical names never capture source bindings", (t) => {
    const key = "f3d-packed/" + digest("child.js"), name = "__f3d_worker_" + digest("classic:" + key);
    const f = fixture(t, {
      "app.mjs": `new Worker('./root.js');`,
      "root.js": `importScripts('./helper.js');new Worker('./child.js');`,
      "helper.js": `const ${name}=42;`, "child.js": "postMessage(1)",
    });
    assert.throws(() => f.pack(), { code: "WORKER_BINDING" });
    assert.equal(fs.existsSync(f.output), false);
  });

  for (const [label, files, code] of [
    ["recursive construction", {"root.js": `new Worker('./child.mjs',{type:'module'});`, "child.mjs": `new Worker('./root.js');`}, "WORKER_NESTING_CYCLE"],
    ["missing nested dependency", {"root.js": `new Worker('./missing.js');`}, "ENOENT"],
    ["open child URL", {"root.js": `new Worker(name);`}, "WORKER_URL"],
    ["open option type", {"root.js": `new Worker('./child.js',{type:name});`}, "WORKER_OPTIONS"],
    ["prototype options", {"root.js": `new Worker('./child.js',{__proto__:options});`}, "WORKER_OPTIONS"],
    ["mutable fetch method", {"root.js": `fetch('./config.json',{method:name});`}, "WORKER_FETCH"],
    ["POST fetch", {"root.js": `fetch('./config.json',{method:'POST'});`}, "WORKER_FETCH"],
    ["open importScripts branch", {"root.js": `importScripts(flag?'./child.js':name);`}, "WORKER_SCRIPT_URL"],
    ["SharedWorker", {"root.js": `new SharedWorker('./child.js');`}, "HOST_RESOURCE"],
  ]) test(`bounded standalone worker graph refusal: ${label}`, (t) => {
    const f = fixture(t, { "app.mjs": `new Worker('./root.js');`, "child.js": "postMessage(1)", "config.json": "{}", ...files });
    assert.throws(() => f.pack(), { code });
    assert.equal(fs.existsSync(f.output), false);
  });

  test("nested graph byte, file, depth and choice budgets refuse without partial output", (t) => {
    const files = { "app.mjs": `new Worker('./root.js');`, "root.js": `new Worker('./child.js');`, "child.js": "postMessage(42);" };
    for (const options of [{maxBytes:1000},{maxFiles:3}]) {
      const f = fixture(t, files, options);
      assert.throws(() => f.pack(), { code: "PACK_LIMIT" });
      assert.equal(fs.existsSync(f.output), false);
    }
    const chain = { "app.mjs": `new Worker('./c0.js');` };
    for (let i=0;i<70;i++) chain[`c${i}.js`]=i<69?`new Worker('./c${i+1}.js');`:"postMessage(1);";
    const f = fixture(t, chain);
    assert.throws(() => f.pack(), { code: "PACK_LIMIT" });
    assert.equal(fs.existsSync(f.output), false);
    let choice="'./child.js'";
    for(let i=0;i<66;i++) choice=`flag?'./child.js':(${choice})`;
    const g = fixture(t, {...files,"root.js":`new Worker(${choice});`});
    assert.throws(() => g.pack(), { code: "WORKER_URL" });
    assert.equal(fs.existsSync(g.output), false);
  });

  test("nested workers retain root containment, symlink and fresh-output protections", (t) => {
    const f = fixture(t, { "app.mjs": `new Worker('./root.js');`, "root.js": `new Worker('./link.js');` });
    const outside = path.join(f.root,"outside.js");fs.writeFileSync(outside,"postMessage(1)");
    fs.symlinkSync(outside,path.join(f.source,"link.js"));
    assert.throws(() => f.pack(), {code:"ROOT_ESCAPE"});assert.equal(fs.existsSync(f.output),false);
    const g = fixture(t, {"app.mjs":`new Worker('./root.js');`,"root.js":`new Worker('./leaf.js');`,"leaf.js":"postMessage(1)"});
    g.pack();const before=fs.readFileSync(g.output);
    assert.throws(()=>g.pack(),{code:"OUTPUT_EXISTS"});assert.deepEqual(fs.readFileSync(g.output),before);
  });

  test("the same mixed graph in different directories emits identical standalone bytes", (t) => {
    const files={"app.mjs":`new Worker('./root.js');`,
      "root.js":`importScripts('./helper.js');`,
      "helper.js":`new Worker('./leaf.mjs',{type:'module'});`,
      "leaf.mjs":`import config from './config.json' with {type:'json'};postMessage(config.value);`,
      "config.json":'{"value":42}'};
    const a=fixture(t,files),b=fixture(t,files);a.pack();b.pack();
    assert.deepEqual(fs.readFileSync(a.output),fs.readFileSync(b.output));
  });
}
