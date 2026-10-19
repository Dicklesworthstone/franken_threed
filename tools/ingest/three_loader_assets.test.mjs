import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { parse } from "acorn";
import { buildApplication } from "./build_application.mjs";
import { findThreeLoaderRequests } from "./three_loader_assets.mjs";

const rootUrl = new URL("../../upstream/three.js/", import.meta.url).href;
const coreUrl = rootUrl + "build/three.core.js";
const gltfUrl = rootUrl + "examples/jsm/loaders/GLTFLoader.js";
const discover = (code, options = {}) => findThreeLoaderRequests(
  parse(code, { ecmaVersion: "latest", sourceType: "module" }),
  { resolveModule: (specifier) => specifier === "three" ? coreUrl : specifier, ...options },
);

test("discovers H2-style const cube-face arrays and literal textures through pinned namespace identity", () => {
  const requests = discover(`import * as THREE from 'three';
    function generateMaterials() {
      const path = 'textures/cube/SwedishRoyalCastle/', format = '.jpg';
      const urls = [path+'px'+format,path+'nx'+format,path+'py'+format,path+'ny'+format,path+'pz'+format,path+'nz'+format];
      const loader = new THREE.CubeTextureLoader();
      loader.load(urls); loader.load(urls);
      new THREE.TextureLoader().load('textures/uv_grid_opengl.jpg');
    }`);
  assert.deepEqual([...new Set(requests.map((request) => request.url))], [
    ...["px", "nx", "py", "ny", "pz", "nz"].map((face) => `textures/cube/SwedishRoyalCastle/${face}.jpg`),
    "textures/uv_grid_opengl.jpg",
  ]);
});

test("follows named constructor aliases, finite string choices, templates and straight-line loader settings", () => {
  const requests = discover(`import { FileLoader as SourceLoader } from 'three';
    const Loader = SourceLoader;
    function load(which) {
      const loader = new Loader();
      loader.setPath('assets/');
      const extension = which ? '.bin' : '.dat';
      loader.loadAsync(\`file\${extension}\`);
      loader.setPath('discarded/').setCrossOrigin('anonymous').setPath('other/');
      loader.load('final.txt');
      new Loader().setPath('chained/').load('third.txt');
    }`);
  assert.deepEqual(requests.map((request) => request.url), [
    "assets/file.bin", "assets/file.dat", "other/final.txt", "chained/third.txt",
  ]);
});

test("does not infer resources from foreign names, shadowed imports, mutable arrays or escaping loaders", () => {
  const requests = discover(`import * as THREE from 'three';
    import { TextureLoader } from 'file:///unrelated/build/three.core.js';
    new TextureLoader().load('foreign.png');
    function shadow(THREE) { new THREE.TextureLoader().load('parameter.png'); }
    function hoisted() { new THREE.TextureLoader().load('var.png'); var THREE; }
    { const THREE = globalThis.other; new THREE.TextureLoader().load('block.png'); }
    const loader = new THREE.TextureLoader(); useElsewhere(loader); loader.load('escaped.png');
    const faces=['a','b','c','d','e','f']; faces.reverse(); new THREE.CubeTextureLoader().load(faces);
    const conditional = new THREE.TextureLoader();
    if (globalThis.enabled) conditional.setPath('maybe/'); conditional.load('conditional.png');
    const argument = new THREE.TextureLoader();
    argument.load(argument.setPath('args/') ? 'yes.png' : 'no.png');
    const nestedArgument = new THREE.TextureLoader();
    nestedArgument.setCrossOrigin(nestedArgument.setPath('nested-args/') ? 'anonymous' : '').load('nested.png');
    const escapedSetting = new THREE.TextureLoader();
    useElsewhere(escapedSetting.setCrossOrigin('anonymous')); escapedSetting.load('escaped-setting.png');
    const escapedPath = new THREE.TextureLoader();
    useElsewhere(escapedPath.setPath('escape/')); escapedPath.load('escaped-path.png');
    const repeated = new THREE.TextureLoader();
    for (let i=0;i<2;i++) { repeated.load('repeated.png'); repeated.setPath('next-iteration/'); }
    new THREE.TextureLoader(globalThis.manager).load('managed.png');
    new THREE.TextureLoader().load('real.png');`);
  assert.deepEqual(requests.map((request) => request.url), ["real.png"]);
});

test("fluent GLTFLoader path and resource settings retain their final values", () => {
  const requests = discover(`import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
    const loader = new GLTFLoader();
    loader.setPath('first/').setPath('models/').setResourcePath('old/').setResourcePath('textures/');
    loader.load('scene.gltf');`);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "models/scene.gltf");
  assert.equal(requests[0].resourcePath, "textures/");
});

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-loader-assets-"));
  for (const [name, content] of Object.entries({ "package.json": '{"type":"module"}', ...files })) {
    const target = path.join(root, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content);
  }
  return root;
}

test("built FileLoader and GLTFLoader fetch original document-base URLs and external geometry bytes", async () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const model = JSON.stringify({
    asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    buffers: [{ uri: "geometry%20%23.bin?cache=1#positions", byteLength: positions.byteLength }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
  });
  const root = fixture({
    "index.html": '<base href="./site/"><link rel="icon" href="resources/message%20%23.txt"><script type="module" src="../modules/app.mjs"></script>',
    "modules/app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)};
      import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
      export async function load() {
        const loader = new FileLoader(); loader.setPath('resources/');
        const text = await loader.loadAsync('message%20%23.txt?edition=2#part');
        const gltf = await new GLTFLoader().setPath('models/').setResourcePath('shared/').loadAsync('triangle.gltf?version=3#scene');
        return { text, positions: [...gltf.scene.children[0].geometry.attributes.position.array] };
      }`,
    "site/resources/message #.txt": "document-relative bytes",
    "modules/resources/message #.txt": "wrong module-relative bytes",
    "site/models/triangle.gltf": model,
    "site/shared/geometry #.bin": Buffer.from(positions.buffer),
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.equal(fs.readFileSync(path.join(built.outDir, "site/models/triangle.gltf"), "utf8"), model);
  const seen = [];
  const server = http.createServer((request, response) => {
    seen.push(request.url);
    try { response.end(fs.readFileSync(path.join(built.outDir, decodeURIComponent(new URL(request.url, "http://localhost").pathname)))); }
    catch { response.writeHead(404); response.end("missing resource"); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const documentBase = `http://127.0.0.1:${server.address().port}/site/`;
  const OriginalRequest = globalThis.Request, OriginalProgressEvent = globalThis.ProgressEvent;
  // Node's Request lacks the browser document-base resolution used by the real
  // loaders. Supply only that host primitive; all loading/parsing code is built
  // from the unmodified pinned upstream implementation and uses real HTTP.
  globalThis.Request = class extends OriginalRequest {
    constructor(input, options) { super(typeof input === "string" ? new URL(input, documentBase) : input, options); }
  };
  globalThis.ProgressEvent = class extends Event {
    constructor(type, details) { super(type); Object.assign(this, details); }
  };
  try {
    const application = await import(pathToFileURL(path.join(built.outDir, built.entryFiles[0])).href);
    assert.deepEqual(await application.load(), { text: "document-relative bytes", positions: [...positions] });
    assert.deepEqual(seen, [
      "/site/resources/message%20%23.txt?edition=2",
      "/site/models/triangle.gltf?version=3",
      "/site/shared/geometry%20%23.bin?cache=1",
    ]);
  } finally {
    globalThis.Request = OriginalRequest;
    if (OriginalProgressEvent === undefined) delete globalThis.ProgressEvent;
    else globalThis.ProgressEvent = OriginalProgressEvent;
    await new Promise((resolve) => server.close(resolve));
  }
});

test("retains host-owned URLs and custom manager routes while packaging local finite resources", async () => {
  const root = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)};
      export function load(manager) {
        new FileLoader().load('https://example.invalid/data.bin');
        new FileLoader().load('/host/data.bin');
        new FileLoader().load('data:text/plain,inline');
        new FileLoader(manager).load('logical-name');
        new FileLoader().load('runtime-generated.txt');
        new FileLoader().load('local.txt');
      }`,
    "local.txt": "local bytes",
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.equal(fs.readFileSync(path.join(built.outDir, "local.txt"), "utf8"), "local bytes");
  const code = fs.readFileSync(path.join(built.outDir, built.entryFiles[0]), "utf8");
  for (const url of ["https://example.invalid/data.bin", "/host/data.bin", "data:text/plain,inline", "logical-name", "runtime-generated.txt", "local.txt"]) assert.ok(code.includes(url));
});

test("a shared DefaultLoadingManager route in another module prevents false local asset reads and collisions", async () => {
  const root = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import './manager.mjs'; import { FileLoader } from ${JSON.stringify(coreUrl)};
      import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
      export function load() { new FileLoader().load('app.js'); new GLTFLoader().load('logical.gltf'); }`,
    "manager.mjs": `import { DefaultLoadingManager as manager } from ${JSON.stringify(coreUrl)};
      manager.setURLModifier(() => 'data:text/plain,managed');`,
    "app.js": "unrelated local bytes must not replace the bundle",
    "logical.gltf": "this is not the logical manager response",
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.equal(built.emittedFiles.includes("logical.gltf"), false);
  const code = fs.readFileSync(path.join(built.outDir, built.entryFiles[0]), "utf8");
  assert.ok(code.includes("data:text/plain,managed"));
  assert.ok(code.includes("logical.gltf"));
  assert.ok(!code.includes("unrelated local bytes"));
});

test("loader asset names cannot overwrite bundled JavaScript or the HTML entry", async () => {
  for (const url of ["app.js", "index.html"]) {
    const root = fixture({
      "index.html": '<script type="module" src="app.mjs"></script>',
      "app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)}; new FileLoader().load(${JSON.stringify(url)});`,
      "app.js": "authored asset bytes",
    });
    await assert.rejects(buildApplication(path.join(root, "index.html"), path.join(root, "dist")), /collides with (?:an emitted bundle file|the emitted HTML entry)/);
    assert.equal(fs.existsSync(path.join(root, "dist")), false);
  }
});

test("ES module libraries retain relative loader URLs until a document host is known", async () => {
  const root = fixture({
    "app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)}; export const load=()=>new FileLoader().loadAsync('host.txt');`,
    "host.txt": "belongs to one possible host",
  });
  const built = await buildApplication(path.join(root, "app.mjs"), path.join(root, "dist"));
  assert.equal(built.emittedFiles.some((file) => path.basename(file) === "host.txt"), false);
});
