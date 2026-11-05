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

async function startServeCommand(t, args) {
  const child = spawn(process.execPath, [cli, "serve", ...args], { stdio: ["ignore", "pipe", "pipe"] });
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
  return { url, child, exited, stderr: () => stderr };
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

test("explicit document roots serve nested selected entries and their origin-relative resources", async t => {
  const html = '<script type="module" src="/assets/main.mjs?revision=2"></script><link rel="stylesheet" href="/assets/style.css">';
  const root = fixture({
    "dist/index.html": "different root page",
    "dist/pages #/scene #.html": html,
    "dist/assets/main.mjs": "export const answer = 42;",
    "dist/assets/style.css": "body { color: green; }",
    "dist/assets/answer.wasm": wasm,
  });
  const documentRoot = path.join(root, "dist"), entry = path.join(documentRoot, "pages #/scene #.html");
  const ordinary = await startApplicationServer(entry, { port: 0 });
  t.after(() => ordinary.close());
  assert.equal(ordinary.root, path.dirname(entry));
  assert.equal((await fetch(new URL("/assets/main.mjs", ordinary.url))).status, 404);
  for (const target of [entry, path.dirname(entry)]) {
    const running = await startApplicationServer(target, { port: 0, documentRoot });
    t.after(() => running.close());
    assert.equal(running.root, documentRoot);
    assert.equal(new URL(running.url).pathname, "/pages%20%23/scene%20%23.html");
    assert.equal(await (await fetch(running.url)).text(), html);
    const navigation = await fetch(new URL("/?backend=webgl&count=1000", running.url));
    assert.equal(navigation.url, running.url + "?backend=webgl&count=1000");
    assert.equal(await navigation.text(), html);
    assert.equal(await (await fetch(new URL("/assets/main.mjs?revision=2", running.url))).text(), "export const answer = 42;");
    assert.equal(await (await fetch(new URL("/assets/style.css", running.url))).text(), "body { color: green; }");
    const { instance } = await WebAssembly.instantiateStreaming(fetch(new URL("/assets/answer.wasm", running.url)));
    assert.equal(instance.exports.answer(), 42);
    assert.equal(await (await fetch(new URL("/index.html", running.url))).text(), "different root page");
  }
});

test("declared document roots check selected entries and physical containment before listening", async t => {
  const root = fixture({ "dist/pages/demo.html": "selected page", "private.html": "outside", "outside/index.html": "outside directory" });
  const documentRoot = path.join(root, "dist"), entry = path.join(documentRoot, "pages/demo.html");
  fs.symlinkSync(path.join(root, "private.html"), path.join(documentRoot, "pages/escape.html"));
  fs.symlinkSync(path.join(root, "outside"), path.join(documentRoot, "escape"));
  for (const target of [path.join(root, "private.html"), path.join(documentRoot, "pages/escape.html"), path.join(documentRoot, "escape")]) {
    await assert.rejects(startApplicationServer(target, { port: 0, documentRoot }), { code: "F3D_DOCUMENT_ROOT_ESCAPE" });
  }
  await assert.rejects(startApplicationServer(entry, { port: 0, documentRoot: entry }), { code: "F3D_DOCUMENT_ROOT" });
  fs.symlinkSync(documentRoot, path.join(root, "alias"));
  const running = await startApplicationServer(path.join(root, "alias/pages/demo.html"), { port: 0, documentRoot: path.join(root, "alias") });
  t.after(() => running.close());
  assert.equal(running.root, documentRoot);
  assert.equal(new URL(running.url).pathname, "/pages/demo.html");
  assert.equal(await (await fetch(running.url)).text(), "selected page");
  assert.equal((await fetch(new URL("/pages/escape.html", running.url))).status, 403);
  assert.equal((await fetch(new URL("/escape/index.html", running.url))).status, 403);
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
  const { url, child, exited, stderr } = await startServeCommand(t, [packed, "--port", "0", "--isolate"]);
  assert.equal(await (await fetch(url)).text(), fs.readFileSync(packed, "utf8"));
  child.kill("SIGTERM");
  assert.deepEqual(await exited, { code: 0, signal: null });
  assert.equal(stderr(), "");
  for (const args of [["serve"], ["serve", root, "--port", "no"], ["serve", root, "--port", "65536"], ["serve", root, "--document-root"], ["serve", root, "--document-root", "--isolate"], ["serve", root, "--unknown"]]) {
    const result = spawnSync(process.execPath, [cli, ...args], { encoding: "utf8" });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Serve failed:/);
  }
});

test("public serve command accepts a document root for a nested HTML route", async t => {
  const root = fixture({ "pages/demo.html": '<script src="/assets/main.js"></script>', "assets/main.js": "globalThis.answer = 42;" });
  const { url, child, exited, stderr } = await startServeCommand(t, [path.join(root, "pages/demo.html"), "--document-root", root, "--port", "0"]);
  assert.equal(new URL(url).pathname, "/pages/demo.html");
  assert.equal(await (await fetch(url)).text(), '<script src="/assets/main.js"></script>');
  assert.equal(await (await fetch(new URL("/assets/main.js", url))).text(), "globalThis.answer = 42;");
  child.kill("SIGTERM");
  assert.deepEqual(await exited, { code: 0, signal: null });
  assert.equal(stderr(), "");
});
