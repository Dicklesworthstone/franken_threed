import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildApplication } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";
import { startApplicationServer } from "./serve_application.mjs";

const cli = fileURLToPath(new URL("./cli.mjs", import.meta.url));
const wasm = Buffer.from("0061736d010000000105016000017f03020100070a0106616e7377657200000a06010400412a0b", "hex");
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-serve-"));
  for (const [name, bytes] of Object.entries(files)) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, bytes);
  }
  return root;
}

test("emitted applications serve their exact HTML, chunks and streaming Wasm after source relocation", async t => {
  const root = fixture({
    "source/index.html": '<script type="module" src="./main.mjs"></script>',
    "source/main.mjs": 'export const wasm = new URL("./answer.wasm", import.meta.url); export const lazy = () => import("./lazy.mjs");',
    "source/lazy.mjs": "export const answer = 42;",
    "source/answer.wasm": wasm,
  });
  const built = await buildApplication(path.join(root, "source/index.html"), path.join(root, "dist"));
  fs.renameSync(path.join(root, "source"), path.join(root, "source-unavailable"));
  const running = await startApplicationServer(built.outDir, { port: 0 });
  t.after(() => running.close());
  for (const filename of built.emittedFiles) {
    const response = await fetch(new URL(filename, running.url));
    assert.equal(response.status, 200, filename);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), fs.readFileSync(path.join(built.outDir, filename)));
    assert.equal(response.headers.get("cross-origin-embedder-policy"), null, "ordinary serving preserves external integrations");
    if (filename.endsWith(".js")) assert.match(response.headers.get("content-type"), /^text\/javascript/);
  }
  const wasmUrl = new URL(built.emittedFiles.find(filename => filename.endsWith(".wasm")), running.url);
  const { instance } = await WebAssembly.instantiateStreaming(fetch(wasmUrl));
  assert.equal(instance.exports.answer(), 42);
  const head = await fetch(wasmUrl, { method: "HEAD" });
  assert.equal(head.headers.get("content-type"), "application/wasm");
  assert.equal((await head.arrayBuffer()).byteLength, 0);
  assert.equal((await fetch(new URL("missing-chunk.js", running.url))).status, 404);
});

test("named HTML entries retain navigation queries and encoded paths without changing bytes", async t => {
  const html = '<script>globalThis.mode = location.search;</script><img src="./image%20%23.png">';
  const root = fixture({ "scene #.html": html, "image #.png": Buffer.from([1, 2, 3]), "nested/index.html": "nested page" });
  const running = await startApplicationServer(root, { port: 0, isolate: true });
  t.after(() => running.close());
  const response = await fetch(new URL("/?backend=webgl&count=1000", running.url));
  assert.equal(response.url, new URL("./scene%20%23.html?backend=webgl&count=1000", running.url).href);
  assert.equal(await response.text(), html);
  assert.equal(response.headers.get("cross-origin-opener-policy"), "same-origin");
  assert.equal(response.headers.get("cross-origin-embedder-policy"), "require-corp");
  assert.deepEqual(new Uint8Array(await (await fetch(new URL("image%20%23.png?revision=2", running.url))).arrayBuffer()), new Uint8Array([1, 2, 3]));
  assert.equal(await (await fetch(new URL("nested/", running.url))).text(), "nested page");
  assert.equal((await fetch(running.url, { method: "POST", body: "must not write" })).status, 405);
  assert.equal(fs.readFileSync(path.join(root, "scene #.html"), "utf8"), html);
});

test("application serving stays within its output directory and survives malformed requests", async t => {
  const root = fixture({ "dist/index.html": "application", "private.txt": "outside" });
  fs.symlinkSync(path.join(root, "private.txt"), path.join(root, "dist/escape.txt"));
  const running = await startApplicationServer(path.join(root, "dist"), { port: 0 });
  t.after(() => running.close());
  const request = pathname => new Promise((resolve, reject) => {
    http.get({ hostname: "127.0.0.1", port: running.port, path: pathname }, response => {
      response.resume(); response.once("end", () => resolve(response.statusCode));
    }).once("error", reject);
  });
  assert.equal(await request("/escape.txt"), 403);
  assert.equal(await request("/%2e%2e%2fprivate.txt"), 403);
  assert.equal(await request("/%ff"), 400);
  assert.equal(await (await fetch(running.url)).text(), "application");
  await assert.rejects(startApplicationServer(path.join(root, "absent"), { port: 0 }), { code: "ENOENT" });
  await assert.rejects(startApplicationServer(root, { port: 65536 }), /Port must/);
  await assert.rejects(startApplicationServer(root, { port: running.port }), { code: "EADDRINUSE" });
});

test("public serve command launches a standalone file, reports its URL and closes on SIGTERM", async t => {
  const root = fixture({ "source/index.html": '<script type="module" src="./main.mjs"></script>', "source/main.mjs": "export const answer = 42;" });
  const packed = path.join(root, "standalone.html");
  packHtml(path.join(root, "source/index.html"), packed);
  const child = spawn(process.execPath, [cli, "serve", packed, "--port", "0", "--isolate"], { stdio: ["ignore", "pipe", "pipe"] });
  const exited = new Promise(resolve => child.once("exit", (code, signal) => resolve({ code, signal })));
  t.after(async () => { if (child.exitCode === null) child.kill("SIGTERM"); await exited; });
  let stdout = "", stderr = "";
  child.stderr.on("data", chunk => { stderr += chunk; });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Server did not start: ${stderr}`)), 10000);
    child.once("error", error => { clearTimeout(timer); reject(error); });
    child.once("exit", () => { clearTimeout(timer); reject(new Error(`Server exited: ${stderr}`)); });
    child.stdout.on("data", chunk => {
      stdout += chunk;
      const match = /Serving application at (http:\/\/\S+)/.exec(stdout);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
  assert.equal(await (await fetch(url)).text(), fs.readFileSync(packed, "utf8"));
  child.kill("SIGTERM");
  assert.deepEqual(await exited, { code: 0, signal: null });
  assert.equal(stderr, "");
  for (const args of [["serve"], ["serve", root, "--port", "no"], ["serve", root, "--port", "65536"], ["serve", root, "--unknown"]]) {
    const result = spawnSync(process.execPath, [cli, ...args], { encoding: "utf8" });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Serve failed:/);
  }
});
