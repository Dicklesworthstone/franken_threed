import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { buildApplication, extractRelativeAssetUrls } from "./build_application.mjs";
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
