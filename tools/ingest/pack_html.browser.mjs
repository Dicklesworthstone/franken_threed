/**
 * Real Chromium single-file integration check; no extra npm test dependency.
 * F3D_CHROMIUM=/path/to/chromium node tools/ingest/pack_html.browser.mjs
 * Uses a fresh profile and leaves generated files available for diagnosis.
 * A software/headless browser proves functionality, not GPU performance.
 */
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const browser = process.env.F3D_CHROMIUM;
if (!browser) throw new Error("Set F3D_CHROMIUM to an installed Chromium executable");
const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-single-html-browser-"));
const input = path.join(root, "source"),
  output = path.join(root, "export", "app.html");
fs.mkdirSync(input);
const files = {
  "index.html": `<!doctype html><html><head><title>Offline application</title>
    <link rel="stylesheet" href="app.css"><link rel="modulepreload" href="entry.mjs" id="warm" onload="globalThis.warmed=true">
    </head><body><h1 id="title">Packed application</h1><img id="pixel" src="pixel.png" srcset="pixel.png 1x, pixel2.png 2x">
    <script>globalThis.events=['classic'];</script>
    <script type="module" src="entry.mjs" data-source="original" data-literal="__f3d_module_url__ </script>" onload="globalThis.entryLoaded=true"></script>
    <script type="module" src="entry.mjs" id="repeat"></script>
    <script type="module">import {token} from './entry.mjs';globalThis.sameEntry=token===globalThis.entryToken;</script>
    </body></html>`,
  "entry.mjs": `import { bump, value, object } from './shared.mjs';
    import { read } from './cycle.mjs';
    import config from './config.json' with {type:'json'};
    import { token as a } from './identity.mjs?a';
    import { token as b } from './identity.mjs?b';
    export const token = {};globalThis.entryToken=token;
    globalThis.events.push('entry'); bump();
    const lazy = await import('./lazy.mjs');
    const choose = flag => import((s=>s==='./source-left.mjs'?'./left.mjs':s==='./source-right.mjs'?'./right.mjs':s)(flag?'./source-left.mjs':'./source-right.mjs'));
    const {instance} = await WebAssembly.instantiateStreaming(fetch(new URL('./add.wasm', import.meta.url)));
    const model=await (await fetch(new URL('./model.gltf',import.meta.url))).json();
    const geometry=await (await fetch(model.buffers[0].uri)).arrayBuffer();
    const texture=new Image();texture.src=model.images[0].uri;await texture.decode();
    await document.querySelector('#pixel').decode();
    const style = await new Promise(resolve => {
      const check=()=>{const c=getComputedStyle(document.querySelector('#title')).color;
        if(c==='rgb(12, 34, 56)')resolve(c);else setTimeout(check,10);};check();
    });
    globalThis.result={events:globalThis.events, live:value, cyclic:read(), shared:lazy.object===object,
      distinct:a!==b, lazy:lazy.answer, left:(await choose(true)).value, right:(await choose(false)).value,
      wasm:instance.exports.add(20,22), settings:config.answer, mesh:[...new Float32Array(geometry)], texture:texture.naturalWidth, pixel:document.querySelector('#pixel').naturalWidth, style};
    document.body.setAttribute('data-result',JSON.stringify(globalThis.result));`,
  "shared.mjs": `import {read} from './cycle.mjs'; export let value=1; export const object={}; export function bump(){value++;}`,
  "cycle.mjs": `import {value} from './shared.mjs'; export function read(){return value;}`,
  "identity.mjs": `export const token={};`,
  "config.json": '{"answer":42}',
  "lazy.mjs": `export {object} from './shared.mjs'; export const answer=7;`,
  "left.mjs": `export const value='left';`,
  "right.mjs": `export const value='right';`,
  "app.css": `@import './colors.css'; body{background-image:url('./pixel.png');} body::after{content:'url(not-a-resource.png)';}`,
  "colors.css": `#title{color:rgb(12,34,56)}`,
  "pixel.png": Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==",
    "base64",
  ),
  "add.wasm": Buffer.from([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 7, 1, 96, 2, 127, 127, 1, 127, 3, 2, 1, 0, 7, 7, 1, 3, 97, 100,
    100, 0, 0, 10, 9, 1, 7, 0, 32, 0, 32, 1, 106, 11,
  ]),
};
// These become blob-backed modules. The repeated entry, its preload and its
// importer must share one evaluation. Identical bytes at distinct query URLs
// still name different module instances.
const largeComment = "/*" + "x".repeat(512 * 1024) + "*/";
files["entry.mjs"] += largeComment;
files["identity.mjs"] += largeComment;
files["model.gltf"] = JSON.stringify({
  asset: { version: "2.0" },
  buffers: [{ uri: "geometry.bin", byteLength: 12 }],
  images: [{ uri: "pixel.png" }],
});
files["geometry.bin"] = Buffer.from(new Float32Array([1, 2, 3]).buffer);
files["pixel2.png"] = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8z8Dwn4GBgYGJAQoAHxcCAk+Uzr4AAAAASUVORK5CYII=",
  "base64",
);
// Exercise honest integrity rehashing for rewritten module and stylesheet bytes.
const sri = (bytes) => "sha384-" + createHash("sha384").update(bytes).digest("base64");
files["index.html"] = files["index.html"]
  .replace('href="app.css"', `href="app.css" integrity="${sri(files["app.css"])}"`)
  .replace('href="entry.mjs"', `href="entry.mjs" integrity="${sri(files["entry.mjs"])}"`)
  .replace(
    'src="entry.mjs" data-source',
    `src="entry.mjs" integrity="${sri(files["entry.mjs"])}" data-source`,
  );
for (const [name, bytes] of Object.entries(files)) fs.writeFileSync(path.join(input, name), bytes);
const reportPath = path.join(root, "export-report.json");
const packed = spawnSync(
  process.execPath,
  [
    fileURLToPath(new URL("./cli.mjs", import.meta.url)),
    "--entry",
    path.join(input, "index.html"),
    "--pack-html",
    output,
    "--output",
    reportPath,
  ],
  { encoding: "utf8" },
);
assert.equal(packed.status, 0, packed.stderr);
const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
if (process.env.F3D_HTML_SAMPLE) fs.copyFileSync(output, process.env.F3D_HTML_SAMPLE);
// Relocate the original source directory out of reach; no source deletion.
fs.renameSync(input, path.join(root, "source-not-used-by-export"));
// Navigate to the relocated file so module scripts and preloads execute in
// the browser's ordinary streaming HTML parser, with the network offline.
const documentUrl = pathToFileURL(output).href;
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
    timer = setTimeout(
      () => reject(new Error("Browser debugging endpoint did not start: " + log)),
      10000,
    );
    child.on("error", reject);
    child.stderr.on("data", (bytes) => {
      log += bytes;
      const match = /DevTools listening on (ws:\/\/[^\s]+)/.exec(log);
      if (match) {
        clearTimeout(timer);
        resolve(match[1]);
      }
    });
    child.on("exit", (code) =>
      reject(new Error("Browser exited before connecting: " + code + "\n" + log)),
    );
  });
  socket = new WebSocket(address);
  await new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error("Chromium debugging WebSocket did not connect")), 15000);
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  clearTimeout(timer);
  let sequence = 0;
  const pending = new Map(),
    errors = [],
    network = [];
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const request = pending.get(message.id);
      if (!request) return;
      pending.delete(message.id);
      clearTimeout(request.timeout);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown")
      errors.push(message.params.exceptionDetails);
    else if (message.method === "Network.requestWillBeSent")
      network.push(message.params.request.url);
  });
  function command(method, params = {}, sessionId) {
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`Chromium stopped responding to ${method}`));
      }, 60000);
      pending.set(id, { resolve, reject, timeout });
      socket.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
  const { targetInfos } = await command("Target.getTargets");
  const { targetId } = targetInfos.find(info => info.type === "page" && info.url === "about:blank");
  const { sessionId } = await command("Target.attachToTarget", { targetId, flatten: true });
  await command("Runtime.enable", {}, sessionId);
  await command("Network.enable", {}, sessionId);
  await command(
    "Network.emulateNetworkConditions",
    { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 },
    sessionId,
  );
  const navigation = await command("Page.navigate", { url: documentUrl }, sessionId);
  assert.equal(navigation.errorText, undefined, navigation.errorText);
  let actual, tags;
  for (let attempts = 0; attempts < 150; attempts++) {
    const result = await command(
      "Runtime.evaluate",
      { expression: `(() => {
        if (!globalThis.result || !globalThis.entryLoaded || !globalThis.warmed || globalThis.sameEntry === undefined) return;
        const entry = document.querySelector('script[data-source="original"]');
        const preload = document.getElementById('warm');
        return {result: globalThis.result, tags: {
          blob: entry.src.startsWith('blob:'), sharedUrl: entry.src === preload.href && entry.src === document.getElementById('repeat').src,
          rehashed: entry.integrity === preload.integrity && entry.integrity !== ${JSON.stringify(sri(files["entry.mjs"]))},
          integrity: entry.integrity, literal: entry.dataset.literal, sameEntry: globalThis.sameEntry,
        }};
      })()`, returnByValue: true },
      sessionId,
    );
    ({result: actual, tags} = result.result.value ?? {});
    if (actual || errors.length) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.deepEqual(errors, [], JSON.stringify(errors));
  assert.ok(tags, "The packed entry, preload or indirect importer did not finish");
  const {integrity, ...identity} = tags;
  assert.match(integrity, /^sha384-[A-Za-z0-9+/]{64}$/);
  assert.deepEqual(identity, {blob: true, sharedUrl: true, rehashed: true,
    literal: "__f3d_module_url__ </script>", sameEntry: true});
  assert.deepEqual(actual, {
    events: ["classic", "entry"],
    live: 2,
    cyclic: 2,
    shared: true,
    distinct: true,
    lazy: 7,
    left: "left",
    right: "right",
    wasm: 42,
    settings: 42,
    mesh: [1, 2, 3],
    texture: 1,
    pixel: 1,
    style: "rgb(12, 34, 56)",
  });
  const externalRequests = network.filter(
    (url) => !url.startsWith("data:") && !url.startsWith("blob:") && url !== "about:blank" && url !== documentUrl,
  );
  assert.equal(network.filter(url => url === documentUrl).length, 1);
  assert.deepEqual(externalRequests, []);
  console.log(
    JSON.stringify(
      {
        browser: spawnSync(browser, ["--version"], { encoding: "utf8" }).stdout.trim(),
        transport: "file navigation; browser offline; source tree relocated",
        result: actual,
        externalRequests,
        ...report,
      },
      null,
      2,
    ),
  );
} finally {
  clearTimeout(timer);
  socket?.close();
  if (child.pid && child.exitCode === null && child.signalCode === null) {
    const stopped = new Promise(resolve => child.once("exit", resolve));
    child.kill();
    const forceStop = setTimeout(() => child.kill("SIGKILL"), 5000);
    try { await stopped; } finally { clearTimeout(forceStop); }
  }
}
