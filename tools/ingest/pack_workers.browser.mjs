/** Native Worker/offline packing integration. F3D_CHROMIUM=/path/to/chromium node this-file.
 * Reuses the CDP approach of pack_html.browser.mjs; no npm browser dependency.
 * Files are retained for diagnosis. No real-GPU or timing claim.
 */
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { packHtml } from "./pack_html.mjs";

const browser = process.env.F3D_CHROMIUM;
if (!browser) throw new Error("Set F3D_CHROMIUM to an installed Chromium executable");
const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-worker-browser-"));
const source = path.join(root, "source"),
  output = path.join(root, "export", "app.html");
const files = {
  "index.html":
    '<!doctype html><meta charset="utf-8"><title>Packed off-thread jobs</title><script type="module" src="app.mjs"></script>',
  "app.mjs": `const events=[];
    function create(){return new Worker(new URL('./jobs/worker.js',import.meta.url),{type:'classic',name:(events.push('name'),'geometry')});}
    async function job(w,id){
      const values=Float64Array.from({length:30000},(_,i)=>i/8);
      const channel=new MessageChannel();
      const portReply=new Promise(resolve=>{channel.port1.onmessage=e=>resolve(e.data);channel.port1.start();});
      const reply=new Promise((resolve,reject)=>{w.onmessage=e=>resolve(e.data);w.onerror=e=>{e.preventDefault();reject(new Error(e.message));};});
      w.postMessage({id,buffer:values.buffer,port:channel.port2},[values.buffer,channel.port2]);
      const detached=values.byteLength===0;
      const result=await reply;const port=await portReply;channel.port1.close();
      const actual=new Float64Array(result.buffer);
      const correct=actual.every((value,i)=>value===(i/8)*2+3);
      return {detached,correct,length:actual.length,port,...result,buffer:undefined};
    }
    try {
      const first=create(),second=create();
      const native=first instanceof Worker && second instanceof Worker;
      const results=[await job(first,1),await job(second,2),await job(first,3)];
      first.terminate();second.terminate();
      const third=create();const restarted=await job(third,4);third.terminate();
      const spinning=new Worker('./spin.js');
      const started=new Promise(resolve=>spinning.onmessage=e=>resolve(e.data));
      spinning.postMessage(1);await started;spinning.terminate();
      const broken=new Worker('./broken.js');
      const error=await new Promise(resolve=>{broken.onerror=e=>{e.preventDefault();resolve(e.message);};});broken.terminate();
      globalThis.result={native,events,results,restarted,terminated:true,error};
    } catch(error){globalThis.failure=String(error.stack||error);}`,
  "jobs/worker.js": `self.order=[];importScripts('./lib/math.js','./lib/decoder.js');importScripts('./lib/math.js');
    self.onmessage=async e=>{
      const {id,buffer,port}=e.data;
      const result=await process(buffer);
      port.postMessage({id,name:self.name});port.close();
      self.postMessage({buffer,order:self.order,loads:self.loads,name:self.name,...result},[buffer]);
    };`,
  "jobs/lib/math.js": `self.loads=(self.loads||0)+1;self.order.push('math');self.transform=x=>x*2+3;`,
  "jobs/lib/decoder.js": `self.order.push('decoder');self.importScripts('./constant.js');
    self.process=async buffer=>{
      const values=new Float64Array(buffer);for(let i=0;i<values.length;i++)values[i]=transform(values[i]);
      const {instance}=await WebAssembly.instantiateStreaming(fetch('./add.wasm'));
      const state=await import('./state.mjs');state.bump();
      const left=await import(true?'./left.mjs':'./right.mjs');
      const right=await import(false?'./left.mjs':'./right.mjs');
      const config=await import('./settings.json',{with:{type:'json'}});
      const model=await (await fetch('./model.gltf')).json();
      const bytes=await (await fetch(model.buffers[0].uri)).arrayBuffer();
      return {wasm:instance.exports.add(20,22),state:state.value,left:left.read(),right:right.value,
        config:config.default.value,geometry:[...new Float32Array(bytes)]};
    };`,
  "jobs/constant.js": `self.order.push('entry-relative');`,
  "jobs/lib/state.mjs": `export let value=0;export const bump=()=>value++;`,
  "jobs/lib/left.mjs": `import {value} from './state.mjs';export const read=()=>value;`,
  "jobs/lib/right.mjs": `export const value=9;`,
  "jobs/lib/settings.json": '{"value":42}',
  "jobs/add.wasm": Buffer.from([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 7, 1, 96, 2, 127, 127, 1, 127, 3, 2, 1, 0, 7, 7, 1, 3, 97, 100,
    100, 0, 0, 10, 9, 1, 7, 0, 32, 0, 32, 1, 106, 11,
  ]),
  "jobs/model.gltf":
    '{"asset":{"version":"2.0"},"buffers":[{"uri":"geometry.bin","byteLength":12}]}',
  "jobs/geometry.bin": Buffer.from(new Float32Array([1, 2, 3]).buffer),
  "spin.js": `self.onmessage=()=>{postMessage('started');while(true){}};`,
  "broken.js": `throw new Error('expected packed worker error');`,
};
for (const [name, data] of Object.entries(files)) {
  const dest = path.join(source, name);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, data);
}
const report = packHtml(path.join(source, "index.html"), output);
if (process.env.F3D_HTML_SAMPLE) fs.copyFileSync(output, process.env.F3D_HTML_SAMPLE);
fs.renameSync(source, path.join(root, "source-not-used-by-export"));
const child = spawn(browser, [
  "--headless=new",
  "--no-sandbox",
  "--disable-gpu",
  "--disable-dev-shm-usage",
  "--no-proxy-server",
  "--host-resolver-rules=MAP * ~NOTFOUND",
  "--disable-background-networking",
  "--user-data-dir=" + path.join(root, "profile"),
  "--remote-debugging-port=0",
  "about:blank",
]);
let socket, timer;
try {
  const address = await new Promise((resolve, reject) => {
    let log = "";
    timer = setTimeout(() => reject(new Error("Browser did not start: " + log)), 10000);
    child.on("error", reject);
    child.stderr.on("data", (bytes) => {
      log += bytes;
      const m = /DevTools listening on (ws:\/\/[^\s]+)/.exec(log);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
    child.on("exit", (code) => reject(new Error("Browser exited: " + code + "\n" + log)));
  });
  socket = new WebSocket(address);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  let sequence = 0,
    workerTargets = 0;
  const pending = new Map(),
    network = [],
    errors = [],
    attachments = [];
  function command(method, params = {}, sessionId) {
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
  socket.addEventListener("message", (event) => {
    const m = JSON.parse(event.data);
    if (m.id) {
      const request = pending.get(m.id);
      if (!request) return;
      pending.delete(m.id);
      if (m.error) request.reject(new Error(JSON.stringify(m.error)));
      else request.resolve(m.result);
    } else if (m.method === "Network.requestWillBeSent") network.push(m.params.request.url);
    else if (m.method === "Runtime.exceptionThrown") errors.push(m.params.exceptionDetails);
    else if (m.method === "Target.attachedToTarget" && m.params.targetInfo.type === "worker") {
      workerTargets++;
      const session = m.params.sessionId;
      attachments.push(
        (async () => {
          await command("Network.enable", {}, session);
          // Worker targets do not implement Network.emulateNetworkConditions.
          // DNS is disabled process-wide; observe ALL worker requests before run.
          await command("Runtime.runIfWaitingForDebugger", {}, session);
        })().catch((error) => errors.push(String(error))),
      );
    }
  });
  const { targetId } = await command("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await command("Target.attachToTarget", { targetId, flatten: true });
  await command("Page.enable", {}, sessionId);
  await command("Runtime.enable", {}, sessionId);
  await command("Network.enable", {}, sessionId);
  await command(
    "Target.setAutoAttach",
    { autoAttach: true, waitForDebuggerOnStart: true, flatten: true },
    sessionId,
  );
  await command(
    "Network.emulateNetworkConditions",
    { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 },
    sessionId,
  );
  const { frameTree } = await command("Page.getFrameTree", {}, sessionId);
  await command(
    "Page.setDocumentContent",
    { frameId: frameTree.frame.id, html: fs.readFileSync(output, "utf8") },
    sessionId,
  );
  let actual, failure;
  for (let tries = 0; tries < 200; tries++) {
    const response = await command(
      "Runtime.evaluate",
      {
        expression: "({result:globalThis.result,failure:globalThis.failure})",
        returnByValue: true,
      },
      sessionId,
    );
    actual = response.result.value?.result;
    failure = response.result.value?.failure;
    if (actual || failure || errors.length) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  await Promise.all(attachments);
  assert.equal(failure, undefined);
  assert.deepEqual(errors, []);
  assert.ok(actual, "Worker application did not complete");
  assert.equal(actual.native, true);
  assert.equal(actual.terminated, true);
  assert.match(actual.error, /expected packed worker error/);
  assert.deepEqual(actual.events, ["name", "name", "name"]);
  for (const [index, result] of [...actual.results, actual.restarted].entries()) {
    assert.equal(result.detached, true);
    assert.equal(result.correct, true);
    assert.equal(result.length, 30000);
    assert.equal(result.wasm, 42);
    assert.equal(result.config, 42);
    assert.equal(result.name, "geometry");
    assert.deepEqual(result.order, ["math", "decoder", "entry-relative", "math"]);
    assert.equal(result.loads, 2);
    assert.deepEqual(result.port, { id: index + 1, name: "geometry" });
    assert.deepEqual(result.geometry, [1, 2, 3]);
    assert.equal(result.state, index === 2 ? 2 : 1);
    assert.equal(result.left, index === 2 ? 2 : 1);
    assert.equal(result.right, 9);
  }
  assert.equal(workerTargets, 5);
  const externalRequests = network.filter(
    (url) => !url.startsWith("data:") && !url.startsWith("blob:") && url !== "about:blank",
  );
  assert.deepEqual(externalRequests, []);
  console.log(
    JSON.stringify(
      {
        browser: spawnSync(browser, ["--version"], { encoding: "utf8" }).stdout.trim(),
        transport:
          "document injection; page offline; DNS disabled; worker network observed; source relocated",
        workerTargets,
        externalRequests,
        result: actual,
        ...report,
      },
      null,
      2,
    ),
  );
  await command("Browser.close");
} finally {
  clearTimeout(timer);
  socket?.close();
  child.kill();
}
