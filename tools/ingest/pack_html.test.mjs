import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { packHtml } from "./pack_html.mjs";

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-pack-html-"));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(root, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
  }
  return { root, entry: path.join(root, "index.html"), out: path.join(root, "packed.html") };
}
function build(files, options) {
  const f = fixture(files),
    result = packHtml(f.entry, f.out, options);
  return { ...f, result, html: fs.readFileSync(f.out, "utf8") };
}
const dataText = (url) => Buffer.from(url.split(",")[1].split("#")[0], "base64").toString("utf8");
const mapOf = (html) =>
  JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;

test("packs circular native ESM, live bindings, finite lazy imports and shared roots", () => {
  const f = build({
    "index.html":
      '<!doctype html><title>App</title><script type="module" src="./entry.mjs"></script><script type="module" src="./entry.mjs"></script>',
    "entry.mjs": `import { get } from './a.mjs'; export {get}; export const lazy=flag=>import(flag?'./a.mjs':'./b.mjs');`,
    "a.mjs": `import { value } from './b.mjs'; export function get(){return value;}`,
    "b.mjs": `import {get} from './a.mjs'; export let value=3; export const read=()=>get();`,
  });
  assert.equal(f.result.moduleCount, 3);
  const imports = mapOf(f.html);
  assert.equal(Object.keys(imports).length, 3);
  for (const url of Object.values(imports)) {
    assert.ok(url.startsWith("data:text/javascript;base64,"));
    assert.ok(!dataText(url).includes("'./"));
  }
  const roots = [...f.html.matchAll(/type="module" src="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(roots[0], roots[1]);
  assert.ok(f.html.startsWith("<!doctype html><title>App</title>"));
});

test("same-byte modules and query/fragment variants remain distinct identities", () => {
  const f = build({
    "index.html": '<script type="module" src="a.mjs"></script>',
    "a.mjs": `import './same.mjs?one'; import './same.mjs?two'; import './same.mjs#other'; import './copy.mjs';`,
    "same.mjs": "export const value={};",
    "copy.mjs": "export const value={};",
  });
  assert.equal(f.result.moduleCount, 5);
  assert.equal(new Set(Object.values(mapOf(f.html))).size, 5);
});

test("resolves source import-map scopes and aliases before assigning embedded identities", () => {
  const f = build({
    "index.html": `<script type="importmap">{"imports":{"shared":"./shared.mjs"},"scopes":{"./nested/":{"shared":"./special.mjs"}}}</script><script type="module" src="entry.mjs"></script>`,
    "entry.mjs": `import 'shared'; import './nested/child.mjs';`,
    "nested/child.mjs": `import 'shared';`,
    "shared.mjs": 'export const tag="normal";',
    "special.mjs": 'export const tag="special";',
  });
  assert.equal(f.result.moduleCount, 4);
  assert.equal((f.html.match(/type="importmap"/g) ?? []).length, 1);
  const sources = Object.values(mapOf(f.html)).map(dataText);
  assert.ok(sources.includes('export const tag="normal";'));
  assert.ok(sources.includes('export const tag="special";'));
});

test("explicit document roots close nested HTML, rooted import maps, stylesheets and resource URLs", () => {
  const f = fixture({
    "pages/demo/index.html": `<script type="importmap">${JSON.stringify({
      imports: {shared: "/modules/shared.mjs?one#view", "/modules/alias.mjs": "/modules/shared.mjs?one#view", "pkg/": "/modules/"},
      scopes: {"/modules/nested/": {shared: "/modules/scoped.mjs"}},
    })}</script><link rel="stylesheet" href="/styles/main.css"><link rel="modulepreload" href="/modules/main.mjs?entry#view">
      <img src="/assets/pixel.png?one#image" srcset="/assets/pixel.png?two 2x, /assets/pixel.png?three 3x">
      <script type="module" src="/modules/main.mjs?entry#view"></script>
      <script type="module">import '/modules/main.mjs?entry#view';</script>`,
    "modules/main.mjs": `import 'shared'; import '/modules/alias.mjs'; import 'pkg/nested/child.mjs';
      export const resource=new URL('/assets/mesh.bin?cache=1#mesh',import.meta.url);`,
    "modules/nested/child.mjs": "import 'shared';",
    "modules/shared.mjs": 'export const tag="ordinary";',
    "modules/scoped.mjs": 'export const tag="scoped";',
    "styles/main.css": `@import '/styles/other.css';body{background:url('/assets/pixel.png');}`,
    "styles/other.css": `@font-face{font-family:packed;src:url('/assets/font.woff2');}`,
    "assets/pixel.png": Buffer.from([1, 2, 3]),
    "assets/font.woff2": Buffer.from([4, 5, 6]),
    "assets/mesh.bin": Buffer.from([7, 8, 9]),
  });
  const entry = path.join(f.root, "pages/demo/index.html");
  assert.throws(() => packHtml(entry, f.out, {rootDir: f.root}), {code: "ROOT_ESCAPE"},
    'root-relative URLs do not acquire a guessed local origin by default');
  const result = packHtml(entry, f.out, {rootDir: f.root, documentRoot: f.root});
  const html = fs.readFileSync(f.out, 'utf8'), sources = Object.values(mapOf(html)).map(dataText);
  assert.equal(result.moduleCount, 5);
  assert.ok(sources.includes('export const tag="ordinary";'));
  assert.ok(sources.includes('export const tag="scoped";'));
  assert.match(sources.join('\n'), /data:application\/octet-stream;base64,BwgJ#mesh/);
  const stylesheet = dataText(/rel="stylesheet" href="([^"]+)"/.exec(html)[1]);
  const imported = dataText(/@import url\("([^"]+)"\)/.exec(stylesheet)[1]);
  assert.match(stylesheet, /data:image\/png;f3d-resource=[0-9a-f]+;base64,AQID/);
  assert.match(imported, /data:font\/woff2;base64,BAUG/);
  const image = /<img src="([^"]+)" srcset="([^"]+)"/.exec(html);
  assert.ok(image[1].endsWith('#image'));
  const candidates = image[2].split(', ');
  assert.ok(candidates[0].endsWith(' 2x'));
  assert.ok(candidates[1].endsWith(' 3x'));
  assert.notEqual(candidates[0].slice(0, -3), candidates[1].slice(0, -3), 'query variants keep resource identity');
  assert.ok(!html.includes(f.root));
});

test("document origin mapping does not widen the packer's separately declared read root", () => {
  const f = fixture({
    "pages/index.html": '<script type="module" src="/shared.mjs"></script>',
    "shared.mjs": 'export const value=42;',
  });
  const entry = path.join(f.root, 'pages/index.html');
  assert.throws(() => packHtml(entry, f.out, {documentRoot: f.root}), {code: 'ROOT_ESCAPE'});
  assert.equal(fs.existsSync(f.out), false);
  assert.equal(packHtml(entry, f.out, {rootDir: f.root, documentRoot: f.root}).moduleCount, 1);
});

test("rooted import-map prefix expansion cannot follow symlinks outside the declared document root", () => {
  for (const map of [{imports: {"pkg/": "/mapped/"}}, {scopes: {"/": {"pkg/": "/mapped/"}}}]) {
    const f = fixture({
      "document/index.html": `<script type="importmap">${JSON.stringify(map)}</script><script type="module">import 'pkg/escape/module.mjs?cache=1#view';</script>`,
      "document/mapped/inside.mjs": "export const inside=true;",
      "outside/module.mjs": "export const outside=true;",
    });
    const documentRoot = path.join(f.root, "document"), entry = path.join(documentRoot, "index.html");
    fs.symlinkSync(path.join(f.root, "outside"), path.join(documentRoot, "mapped/escape"));
    assert.throws(() => packHtml(entry, f.out, {rootDir: f.root, documentRoot}), {code: "F3D_DOCUMENT_ROOT_ESCAPE"});
    assert.equal(fs.existsSync(f.out), false);
    assert.equal(fs.readFileSync(path.join(f.root, "outside/module.mjs"), "utf8"), "export const outside=true;");
  }
});

test("explicit relative and file import-map targets keep the independent packer read-root scope", () => {
  const f = fixture({"document/index.html": "", "shared/module.mjs": "export const outside=42;"});
  const documentRoot = path.join(f.root, "document"), entry = path.join(documentRoot, "index.html");
  const targets = ["../shared/", pathToFileURL(path.join(f.root, "shared") + path.sep).href];
  for (const [index, target] of targets.entries()) {
    fs.writeFileSync(entry, `<script type="importmap">${JSON.stringify({imports: {"pkg/": target}})}</script><script type="module">import 'pkg/module.mjs';</script>`);
    const output = path.join(f.root, `packed-${index}.html`);
    assert.equal(packHtml(entry, output, {rootDir: f.root, documentRoot}).moduleCount, 2);
    assert.ok(Object.values(mapOf(fs.readFileSync(output, "utf8"))).map(dataText).includes("export const outside=42;"));
  }
});

test("a declared document root contains the HTML entry even when the read root is wider", () => {
  const f = fixture({"index.html": "<title>outside origin</title>", "document/inside.html": "<title>inside</title>"});
  const documentRoot = path.join(f.root, "document"), alias = path.join(documentRoot, "escape.html");
  fs.symlinkSync(f.entry, alias);
  for (const entry of [f.entry, alias]) {
    assert.throws(() => packHtml(entry, f.out, {rootDir: f.root, documentRoot}), {code: "F3D_DOCUMENT_ROOT_ESCAPE"});
    assert.equal(fs.existsSync(f.out), false);
  }
  assert.equal(packHtml(f.entry, f.out, {rootDir: f.root}).moduleCount, 0);
});

test("document roots do not reinterpret remote or protocol-relative resource addresses as local files", () => {
  for (const src of ['https://example.invalid/app.mjs', '//example.invalid/app.mjs', '/\\example.invalid/app.mjs']) {
    const f = fixture({'index.html': `<script type="module" src="${src}"></script>`});
    let original;
    assert.throws(() => packHtml(f.entry, f.out), error => { original = error.code; return true; });
    assert.throws(() => packHtml(f.entry, f.out, {documentRoot: f.root}), error => error.code === original);
    assert.equal(fs.existsSync(f.out), false);
  }
});

test("embeds transitive CSS, image assets, module-relative Wasm and original script attributes", () => {
  const hash = (bytes) => createHash("sha384").update(bytes).digest("base64");
  const js = `export const resource = new URL('./mesh.bin?cache=1#mesh', import.meta.url);`;
  const css = `@import './other.css' screen; body {background:url('./pixel.png');content:"url(fake.png)";}`;
  const f = build({
    "index.html": `<link rel="stylesheet" href="main.css" integrity="sha384-${hash(css)}"><img src="pixel.png"><script type="module" src="app.mjs" nonce="abc" data-label='a"b' integrity="sha384-${hash(js)}"></script>`,
    "main.css": css,
    "other.css": "p{color:rgb(10,20,30)}",
    "pixel.png": Buffer.from([1, 2, 3]),
    "mesh.bin": Buffer.from([0, 255, 3]),
    "app.mjs": js,
  });
  assert.equal(f.result.assetCount, 4);
  assert.ok(f.html.includes('nonce="abc"'));
  assert.ok(f.html.includes(`data-label='a"b'`));
  assert.ok(!f.html.includes(`sha384-${hash(js)}`));
  assert.ok(!f.html.includes(`sha384-${hash(css)}`));
  const source = Object.values(mapOf(f.html)).map(dataText)[0];
  assert.match(source, /new URL\("data:application\/octet-stream;base64,AP8D#mesh"\)/);
});

test("HTML raw text and comments are not scanned as resource tags; closing script text is encoded", () => {
  const f = build({
    "index.html": `<!doctype html><!-- <img src="missing.png"> --><textarea><img src="missing.png"></textarea><script>const fake='<img src="missing.png">';</script><script type="module" src="app.mjs"></script>`,
    "app.mjs": `export const text = '</script><script>window.bad=true</script>';`,
  });
  assert.ok(f.html.includes(`const fake='<img src="missing.png">';`));
  assert.equal((f.html.match(/window.bad/g) ?? []).length, 0);
});

test("inline module bodies, classic dynamic imports and comments are preserved without eval", () => {
  const f = build({
    "index.html": `<script>globalThis.run=()=>import('./lazy.mjs');</script><script type="module">import {value} from './lazy.mjs'; globalThis.value=value;</script>`,
    "lazy.mjs": "export const value=7;",
  });
  assert.equal(f.result.moduleCount, 2);
  assert.match(f.html, /globalThis.run=\(\)=>import\("f3d-packed\//);
  assert.match(f.html, /<script type="module" src="data:text\/javascript;base64,/);
});

test("percent-encoded filenames and HTML entities resolve to actual local bytes", () => {
  const f = build({
    "index.html":
      '<img src="hello%20world.png?q=a&amp;b=c"><script type="module" src="app.mjs"></script>',
    "hello world.png": Buffer.from([1, 2]),
    "app.mjs": "export const x=1;",
  });
  assert.match(f.html, /data:image\/png;f3d-resource=[0-9a-f]+;base64,AQI=/);
});

for (const [label, files, code] of [
  [
    "dynamic imports",
    { "app.mjs": `export const load=name=>import(name);` },
    "DYNAMIC_IMPORT_OPEN",
  ],
  [
    "module URL reflection",
    { "app.mjs": `export const source=import.meta.url;` },
    "MODULE_URL_OBSERVATION",
  ],
  [
    "remote modules",
    { "app.mjs": `import 'https://example.com/module.mjs';` },
    "EXTERNAL_RESOURCE",
  ],
  ["worker realm", { "app.mjs": `new Worker('./worker.mjs');` }, "HOST_RESOURCE"],
  ["runtime fetch", { "app.mjs": `fetch('./bytes.bin');` }, "FETCH_URL"],
  [
    "base URL",
    {
      "index.html": '<base href="./"><script type="module" src="app.mjs"></script>',
      "app.mjs": "",
    },
    "BASE_URL",
  ],
  [
    "CSP",
    {
      "index.html":
        '<meta http-equiv="Content-Security-Policy" content="script-src \'self\'"><script type="module" src="app.mjs"></script>',
      "app.mjs": "",
    },
    "CSP_POLICY",
  ],
  [
    "blocked import",
    {
      "index.html":
        '<script type="importmap">{"imports":{"blocked":null}}</script><script type="module" src="app.mjs"></script>',
      "app.mjs": `import 'blocked';`,
    },
    "BLOCKED_IMPORT",
  ],
  [
    "bad integrity",
    {
      "index.html": '<script type="module" src="app.mjs" integrity="sha384-wrong"></script>',
      "app.mjs": "export const x=1;",
    },
    "INTEGRITY",
  ],
  [
    "CSS cycle",
    {
      "index.html": '<link rel="stylesheet" href="a.css">',
      "a.css": '@import "b.css";',
      "b.css": '@import "a.css";',
    },
    "CSS_CYCLE",
  ],
])
  test(`${label}: refuses before creating output and leaves the normal application intact`, () => {
    const f = fixture({ "index.html": '<script type="module" src="app.mjs"></script>', ...files });
    const before = fs.readFileSync(f.entry);
    assert.throws(() => packHtml(f.entry, f.out), { code });
    assert.equal(fs.existsSync(f.out), false);
    assert.deepEqual(fs.readFileSync(f.entry), before);
  });

test("path and symlink escapes cannot read outside the application root", () => {
  const f = fixture({ "index.html": '<script type="module" src="../secret.mjs"></script>' });
  const outside = path.join(f.root, "..", "secret.mjs");
  // Use a unique outside path rather than overwriting an existing file.
  const other = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-pack-outside-"));
  fs.writeFileSync(path.join(other, "private.mjs"), 'throw new Error("not read");');
  fs.writeFileSync(
    f.entry,
    `<script type="module" src="${path.relative(f.root, path.join(other, "private.mjs"))}"></script>`,
  );
  assert.throws(() => packHtml(f.entry, f.out), { code: "ROOT_ESCAPE" });
  fs.symlinkSync(path.join(other, "private.mjs"), path.join(f.root, "link.mjs"));
  fs.writeFileSync(f.entry, '<script type="module" src="link.mjs"></script>');
  assert.throws(() => packHtml(f.entry, f.out), { code: "ROOT_ESCAPE" });
});

test("output collisions and packing limits never overwrite or partially publish", () => {
  const f = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": "export const x=1;",
  });
  assert.throws(() => packHtml(f.entry, f.out, { maxBytes: 2 }), { code: "PACK_LIMIT" });
  assert.equal(fs.existsSync(f.out), false);
  assert.throws(() => packHtml(f.entry, f.out, { maxFiles: 1 }), { code: "PACK_LIMIT" });
  fs.writeFileSync(f.out, "existing");
  assert.throws(() => packHtml(f.entry, f.out), { code: "OUTPUT_EXISTS" });
  assert.equal(fs.readFileSync(f.out, "utf8"), "existing");
});

test("same logical application in two directories has byte-identical output", () => {
  const files = {
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import './side.mjs';`,
    "side.mjs": "export const value=1;",
  };
  assert.equal(build(files).html, build(files).html);
});

test("the generated import map precedes module preloads as well as script execution", () => {
  const f = build({
    "index.html":
      '<head><link rel="modulepreload" href="a.mjs"></head><body><script type="module" src="a.mjs"></script></body>',
    "a.mjs": `import './b.mjs';`,
    "b.mjs": "export const x=1;",
  });
  assert.ok(f.html.indexOf('type="importmap"') < f.html.indexOf('rel="modulepreload"'));
});

test("unused external source import mappings do not create a false network dependency", () => {
  const f = build({
    "index.html":
      '<script type="importmap">{"imports":{"unused":"https://example.com/not-loaded.mjs"}}</script><script type="module" src="a.mjs"></script>',
    "a.mjs": "export const x=1;",
  });
  assert.equal(f.result.moduleCount, 1);
  assert.ok(!f.html.includes("example.com"));
});
