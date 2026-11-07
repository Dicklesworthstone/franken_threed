import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { buildApplication, extractRelativeAssetUrls } from "./build_application.mjs";
import { parseHtmlEntries } from "./html_parser.mjs";
import { packHtml } from "./pack_html.mjs";

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-html-assets-"));
  const source = path.join(root, "source");
  for (const [name, content] of Object.entries(files)) {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, content);
  }
  return { root, source, entry: path.join(source, "index.html"), out: path.join(root, "dist") };
}

function embeddedResources(html) {
  const seen = new Set();
  const resources = [];
  function visit(text) {
    for (const match of text.matchAll(/data:[^\s"'<>]*?;base64,([a-zA-Z0-9+/=]+)/g)) {
      if (seen.has(match[0])) continue;
      seen.add(match[0]);
      const bytes = Buffer.from(match[1], "base64");
      resources.push(bytes);
      if (match[0].startsWith("data:text/css;")) visit(bytes.toString("utf8"));
    }
  }
  visit(html);
  return resources;
}

test("HTML asset discovery includes real inline CSS and image controls without reading raw-text decoys", () => {
  const html = String.raw`
    <!-- <style>body {background:url(commented.png)}</style> -->
    <script>const template = '<style>body {background:url(script.png)}</style>';</script>
    <textarea><span style="background:url(textarea.png)"></span></textarea>
    <title><span style="background:url(title.png)"></span></title>
    <div data-preview='<style>body {background:url(attribute.png)}</style>'></div>
    <style>
      @import "./theme.css" screen;
      @font-face { font-family: Demo; src: url('./font.woff2'); }
      body { background-image: url('./raw&amp;.svg'); }
      .label::before { content: '<img src="not-markup.png"> url(not-css.png)'; }
      /* background: url(comment.png); */
    </style>
    <STYLE TYPE="text/css">.icon { background: url(./icon.svg#paint); }</STYLE>
    <style type="text/plain">url(inactive-style.png)</style>
    <div style="background:url(&quot;./space\20 name.svg&quot;);mask:url(&#39;./mask&amp;grid.svg&#39;);content:&quot;url(string.png)&quot;"
         style="background:url(duplicate.png)"></div>
    <div style="background:url(./&#x62;anner.svg);cursor:url(./&#99;ursor.svg),auto"></div>
    <svg><image href="./logo&amp;mark.svg#icon" xlink:href="ignored.svg" />
      <image xlink:href="./fallback.svg" /><image href="#local" xlink:href="ignored-too.svg" /></svg>
    <input type="IMAGE" src="./button.svg" /><input type="text" src="inactive-input.svg" />
    <div style="background:url(https://example.invalid/live.svg),url(data:image/svg+xml;base64,AA==)"></div>
  `;
  assert.deepEqual(new Set(extractRelativeAssetUrls(html)), new Set([
    "./theme.css", "./font.woff2", "./raw&amp;.svg", "./icon.svg#paint",
    "./space name.svg", "./mask&grid.svg", "./banner.svg", "./cursor.svg",
    "./logo&mark.svg#icon", "./fallback.svg", "./button.svg",
  ]));
});

test("build and standalone export retain inline style, imported stylesheet, font, SVG and image-input bytes", async () => {
  const style = `<style media="screen">
    @import './styles/theme.css';
    @font-face {font-family: Demo; src: url('./fonts/main.woff2');}
    body {background-image:url('./images/raw&amp;.svg');}
  </style>`;
  const markup = `<main style="background-image:url(&quot;./images/panel&amp;grid.svg&quot;);mask-image:url(&#39;./images/mask.svg#paint&#39;)">
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
      <image href="./images/logo.svg#paint" xlink:href="ignored.svg" />
      <image xlink:href="./images/fallback.svg" />
    </svg>
    <input type="image" src="./images/button.svg" alt="Render scene" />
  </main>`;
  const html = `<!doctype html><title>Scene controls</title>${style}${markup}
    <script type="module" src="./main.mjs"></script>`;
  const svg = (color) => `<svg xmlns="http://www.w3.org/2000/svg"><rect id="paint" width="4" height="4" fill="${color}"/></svg>`;
  const resources = {
    "styles/theme.css": '@import "./nested.css"; .label { background:url("../images/theme.svg"); }',
    "styles/nested.css": '@font-face {font-family: Nested; src:url("../fonts/nested.woff2");}',
    "fonts/main.woff2": Buffer.from([0x77, 0x4f, 0x46, 0x32, 1, 2, 3]),
    "fonts/nested.woff2": Buffer.from([0x77, 0x4f, 0x46, 0x32, 4, 5, 6]),
    "images/raw&amp;.svg": svg("red"),
    "images/panel&grid.svg": svg("green"),
    "images/mask.svg": svg("blue"),
    "images/logo.svg": svg("yellow"),
    "images/fallback.svg": svg("purple"),
    "images/button.svg": svg("orange"),
    "images/theme.svg": svg("cyan"),
  };
  const f = fixture({ "index.html": html, "main.mjs": "export const ready = 42;", ...resources });
  const result = await buildApplication(f.entry, f.out);
  const emittedHtml = fs.readFileSync(path.join(f.out, "index.html"), "utf8");
  assert.ok(emittedHtml.includes(style), "Do not rewrite inline CSS while collecting its resources");
  assert.ok(emittedHtml.includes(markup), "Preserve source attributes, references, fragments and controls");
  assert.equal(fs.readFileSync(f.entry, "utf8"), html, "The source document remains intact");
  for (const [name, bytes] of Object.entries(resources)) {
    assert.ok(result.emittedFiles.includes(name), `${name} is part of the emitted application`);
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), Buffer.from(bytes), name);
    assert.deepEqual(fs.readFileSync(path.join(f.source, name)), Buffer.from(bytes), `${name}: source`);
  }

  // Packing must consume only the emitted application, without finding discarded
  // resources through the original source directory or an incidental HTTP server.
  fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
  const packedFile = path.join(f.root, "scene.html");
  const packedResult = packHtml(path.join(f.out, "index.html"), packedFile);
  assert.equal(packedResult.assetCount, Object.keys(resources).length);
  fs.renameSync(f.out, path.join(f.root, "dist-unavailable"));
  const packed = fs.readFileSync(packedFile, "utf8");
  const embedded = embeddedResources(packed);
  for (const [name, bytes] of Object.entries(resources)) {
    if (name.endsWith(".css")) continue; // The packer rewrites each stylesheet's child URLs.
    assert.ok(embedded.some((value) => value.equals(Buffer.from(bytes))), `${name} is embedded exactly`);
  }
  assert.ok(embedded.some((bytes) => bytes.toString("utf8").includes("font-family: Nested")), "Follow transitive @import rules");
  assert.ok(packed.includes("#paint"), "SVG fragment selection survives embedding");
  const moduleUrl = /<script type="module" src="([^"]+)"/.exec(packed)?.[1];
  assert.ok(moduleUrl?.startsWith("data:text/javascript;base64,"));
  assert.equal((await import(moduleUrl)).ready, 42, "The built module runs with both source and output trees unavailable");
});

test("inline document assets resolve against the first local base while source CSS and markup stay unchanged", async () => {
  const html = `<!doctype html><base href="./assets/"><base href="./ignored/">
    <style>@import './styles/theme.css'; body {background:url('./background.svg');}</style>
    <div style="background-image:url(&quot;./panel.svg&quot;)"></div>
    <svg><image xlink:href="./logo.svg" /></svg><input type="image" src="./button.svg">`;
  const resources = {
    "assets/styles/theme.css": '@font-face {font-family: Example; src:url("../font.woff2")}',
    "assets/font.woff2": Buffer.from([1, 4, 9, 16]),
    "assets/background.svg": "background",
    "assets/panel.svg": "panel",
    "assets/logo.svg": "logo",
    "assets/button.svg": "button",
  };
  const f = fixture({ "index.html": html, ...resources });
  const result = await buildApplication(f.entry, f.out);
  assert.equal(fs.readFileSync(path.join(f.out, "index.html"), "utf8"), html);
  for (const [name, bytes] of Object.entries(resources)) {
    assert.ok(result.emittedFiles.includes(name));
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), Buffer.from(bytes), name);
  }
  const packed = path.join(f.root, "standalone.html");
  packHtml(path.join(f.out, "index.html"), packed);
  const standalone = fs.readFileSync(packed, "utf8");
  assert.ok(standalone.startsWith('<!doctype html><base href="./assets/"><base href="./ignored/">'));
  assert.ok(embeddedResources(standalone).some(bytes => bytes.equals(resources['assets/font.woff2'])),
    'the imported stylesheet retains the original font bytes');
  assert.match(standalone, /data:image\/svg\+xml;f3d-resource=/);
});

test("inline CSS resources use existing missing-file and application-root checks before emission", async () => {
  for (const [reference, expected] of [
    ["missing.svg", /Unresolved relative resource/],
    ["../outside.svg", /escapes application root directory/],
  ]) {
    const html = `<div style="background:url(&quot;${reference}&quot;)"></div>`;
    const f = fixture({ "index.html": html });
    await assert.rejects(buildApplication(f.entry, f.out), expected);
    assert.equal(fs.existsSync(path.join(f.out, "index.html")), false, "Do not emit a broken application");
    assert.equal(fs.readFileSync(f.entry, "utf8"), html);
  }
});

test("declared document root keeps nested HTML, root assets and retained modules working after relocation", async t => {
  const classic = "globalThis.loadRootShared = () => import('/scripts/shared.mjs');";
  const integrity = "sha384-" + crypto.createHash("sha384").update(classic).digest("base64");
  const html = `<!doctype html><title>Rooted scene</title>
    <link rel="stylesheet" href="/styles/app.css?version=2">
    <link rel="modulepreload" href="/scripts/main.mjs" integrity="sha384-placeholder">
    <link rel="modulepreload" href="/scripts/preload-only.mjs">
    <style>@import '/styles/inline.css'; body {mask:url('/images/mask.svg#paint')}</style>
    <img src="/images/panel%20wide.svg?version=2#paint" srcset="/images/small.svg 1x, /images/large.svg 2x">
    <img src="./page.svg"><img src="https://example.invalid/live.svg">
    <script src="/scripts/classic.js" integrity="${integrity}"></script>
    <script type="module" src="/scripts/main.mjs"></script>`;
  const resources = {
    "styles/app.css": '@import "/styles/nested.css"; .panel {background:url("../images/panel%20wide.svg#paint")} .remote {background:url(//example.invalid/remote.svg)}',
    "styles/nested.css": '@font-face {font-family: Scene; src:url("/fonts/scene.woff2")}',
    "styles/inline.css": '.icon {background:url("/images/icon.svg")}',
    "fonts/scene.woff2": Buffer.from([0x77, 0x4f, 0x46, 0x32, 7, 8, 9]),
    "images/panel wide.svg": "wide panel", "images/mask.svg": "mask", "images/icon.svg": "icon",
    "images/small.svg": "small", "images/large.svg": "large", "pages/page.svg": "page-relative image",
    "scripts/classic.js": classic,
    "scripts/shared.mjs": "export const state = {count: 1};",
    "scripts/preload-only.mjs": "export const preloaded = true;",
  };
  const f = fixture({"pages/demo.html": html, ...resources,
    "scripts/main.mjs": "import {state} from '/scripts/shared.mjs'; export {state}; export const answer = 42; export const later = () => import('/scripts/lazy.mjs');",
    "scripts/lazy.mjs": "export {state} from '/scripts/shared.mjs';",
  });
  const result = await buildApplication(path.join(f.source, "pages/demo.html"), f.out, {documentRoot: f.source});
  assert.equal(result.htmlFile, "pages/demo.html");
  assert.ok(!result.emittedFiles.includes("scripts/main.mjs"), "A bundled preload must use its emitted chunk");
  const moved = path.join(f.root, "deployed-site");
  fs.renameSync(f.out, moved); fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
  const server = http.createServer((request, response) => {
    const name = decodeURIComponent(new URL(request.url, "http://localhost").pathname).slice(1);
    const file = path.join(moved, name);
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) { response.writeHead(404); response.end(); return; }
    const type = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript"}[path.extname(file)];
    response.setHeader("Content-Type", type ?? "application/octet-stream"); response.end(fs.readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(() => { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(origin + "/pages/demo.html"); assert.equal(response.status, 200);
  const emitted = await response.text(), parsed = parseHtmlEntries(emitted, origin + "/pages/demo.html");
  assert.ok(emitted.includes(`<script src="/scripts/classic.js" integrity="${integrity}"></script>`));
  assert.ok(emitted.includes('<img src="./page.svg"><img src="https://example.invalid/live.svg">'));
  for (const [name, bytes] of Object.entries(resources)) {
    assert.ok(result.emittedFiles.includes(name), name);
    const resource = await fetch(origin + "/" + name.split("/").map(encodeURIComponent).join("/"));
    assert.equal(resource.status, 200, name);
    assert.deepEqual(Buffer.from(await resource.arrayBuffer()), Buffer.from(bytes), name);
  }
  const scriptUrl = new URL(parsed.moduleScripts[0].src, parsed.baseUrl);
  assert.equal(scriptUrl.pathname, "/" + result.entryFiles[0], "Nested HTML must reach root-level chunks");
  const script = await fetch(scriptUrl); assert.equal(script.status, 200);
  const code = await script.text();
  const preload = /<link rel="modulepreload" href="([^"]+)" integrity="([^"]+)">/.exec(emitted);
  assert.equal(new URL(preload[1], parsed.baseUrl).href, scriptUrl.href);
  assert.equal(preload[2], "sha384-" + crypto.createHash("sha384").update(code).digest("base64"));
  const app = await import(pathToFileURL(path.join(moved, result.entryFiles[0])).href);
  const shared = await import(pathToFileURL(path.join(moved, "scripts/shared.mjs")).href);
  assert.equal(app.answer, 42); assert.equal(app.state, shared.state);
  app.state.count = 9; assert.equal((await app.later()).state, shared.state); assert.equal(shared.state.count, 9);
});

for (const base of ["/site/", "/site/current.html?edition=2#section"])
  test(`declared document root preserves ${base} as the effective base after relocation`, async () => {
    const f = fixture({
      "pages/demo.html": `<base href="${base}"><base href="/ignored/"><link rel="stylesheet" href="./theme.css"><img src="./image.svg"><script type="module" src="./main.mjs"></script>`,
      "site/theme.css": '.image {background:url("/shared/background.svg")}',
      "site/image.svg": "image", "shared/background.svg": "background",
      "site/main.mjs": "export const answer = 42;",
    });
    const entry = path.join(f.source, "pages/demo.html");
    await assert.rejects(buildApplication(entry, path.join(f.root, "without-root")), /Explicit rejection/);
    const result = await buildApplication(entry, f.out, {documentRoot: pathToFileURL(f.source)});
    const moved = path.join(f.root, "deployed-site");
    fs.renameSync(f.out, moved); fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
    const emitted = fs.readFileSync(path.join(moved, result.htmlFile), "utf8");
    const parsed = parseHtmlEntries(emitted, "https://example.test/pages/demo.html");
    assert.equal(parsed.baseUrl, "https://example.test" + base);
    assert.ok(emitted.startsWith(`<base href="${base}"><base href="/ignored/">`));
    assert.equal(new URL(parsed.moduleScripts[0].src, parsed.baseUrl).pathname, "/" + result.entryFiles[0]);
    for (const name of ["site/theme.css", "site/image.svg", "shared/background.svg"])
      assert.ok(fs.existsSync(path.join(moved, name)), name);
    assert.equal((await import(pathToFileURL(path.join(moved, result.entryFiles[0])).href)).answer, 42);
    const packed = path.join(f.root, "standalone.html");
    packHtml(path.join(moved, result.htmlFile), packed, {rootDir: moved, documentRoot: moved});
    const standalone = fs.readFileSync(packed, 'utf8');
    assert.ok(standalone.startsWith(`<base href="${base}"><base href="/ignored/">`));
    assert.match(standalone, /data:text\/javascript;base64,/);
    assert.ok(!standalone.includes(moved));
  });

test("declared document root rejects escaping CSS, symlink resources and nested HTML collisions before output", async () => {
  for (const cause of ["css", "symlink", "collision"]) {
    const f = fixture({"pages/demo.html": cause === "css" ? '<link rel="stylesheet" href="/styles/app.css">'
      : `<img src="${cause === "symlink" ? "/images/outside.svg" : "/pages/demo.html"}">`,
      "styles/app.css": 'body {background:url("../../outside.svg")}',
    });
    const outside = path.join(f.root, "outside.svg"); fs.writeFileSync(outside, "outside declared root");
    if (cause === "symlink") {
      fs.mkdirSync(path.join(f.source, "images")); fs.symlinkSync(outside, path.join(f.source, "images/outside.svg"));
    }
    await assert.rejects(buildApplication(path.join(f.source, "pages/demo.html"), f.out, {documentRoot: f.source}),
      cause === "collision" ? /Collision detected/ : {code: "F3D_DOCUMENT_ROOT_ESCAPE"});
    assert.equal(fs.existsSync(f.out), false);
    assert.equal(fs.readFileSync(outside, "utf8"), "outside declared root");
  }
});
