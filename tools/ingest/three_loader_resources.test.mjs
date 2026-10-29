import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import { parse } from "acorn";
import { buildApplication } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";
import { createThreeLoaderResourceModule, isThreeLoaderResourceModule, threeLoaderResourceEdits } from "./three_loader_resources.mjs";
import { vertex, vertexExpected } from "../../tests/fixtures/meshopt/vectors.mjs";

const core = new URL("../../upstream/three.js/build/three.core.js", import.meta.url);
const gltf = new URL("../../upstream/three.js/examples/jsm/loaders/GLTFLoader.js", import.meta.url);
const data = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const pixel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==", "base64");
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-loader-resources-"));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(root, "source", name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
  }
  return { root, entry: path.join(root, "source/index.html"), out: path.join(root, "build"), packed: path.join(root, "packed.html") };
}

// Native ESM, fetch, Request, Response and streams execute the packed source.
// Only browser document-base/import-map resolution and image host primitives
// are supplied by this recording host; it is not an image-decoder parity test.
const runner = new URL(data(`
import {parentPort,workerData} from 'node:worker_threads';
import {registerHooks} from 'node:module';
registerHooks({resolve(specifier,context,next){return next(workerData.imports[specifier]??specifier,context);}});
const requests=[],images=[],events=[],nativeFetch=globalThis.fetch;
globalThis.document={URL:'https://packed.example/view/index.html',baseURI:'https://packed.example/view/',
  createElementNS(_namespace,tag){if(tag!=='img')throw new Error(tag);return new ImageHost();}};
globalThis.self=globalThis;
globalThis.ProgressEvent=class extends Event {constructor(type,details){super(type);Object.assign(this,details);}};
globalThis.fetch=async (input,options)=>{
  const url=input instanceof Request?input.url:String(input);
  if(!url.startsWith('data:'))throw new Error('Undeclared network request: '+url);
  const response=await nativeFetch(input,options);requests.push({url,mime:response.headers.get('Content-Type')});return response;
};
class ImageHost extends EventTarget {
  get src(){return this._src;}
  set src(url){this._src=url;images.push(url);Promise.resolve().then(async()=>{
    if(!url.startsWith('data:image/png;'))throw new Error('Unembedded image: '+url);
    const bytes=new Uint8Array(await (await nativeFetch(url)).arrayBuffer());
    if(bytes[0]!==137||bytes[1]!==80)throw new Error('Image bytes changed');
    this.complete=true;this.naturalWidth=1;this.naturalHeight=1;this.dispatchEvent(new Event('load'));
  }).catch(error=>{this.dispatchEvent(new Event('error'));});}
}
globalThis.HTMLImageElement=ImageHost;
globalThis.createImageBitmap=async blob=>({width:1,height:1,type:blob.type,byteLength:(await blob.arrayBuffer()).byteLength});
try {
  const app=await import(workerData.url), manager=app.THREE?.DefaultLoadingManager;
  if(manager){
    manager.onStart=url=>events.push(['start',url]);
    manager.onProgress=url=>events.push(['end',url]);
    manager.setURLModifier(url=>url.replace('?version=1','?version=2'));
  }
  parentPort.postMessage({ok:true,result:await app.run(),requests,images,events});
} catch(error){parentPort.postMessage({ok:false,error:error.stack});}
`));
async function execute(html) {
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const url = /<script type="module" src="([^"]+)"/.exec(html)[1];
  const worker = new Worker(runner, { workerData: { imports, url } });
  let timer, received;
  try {
    return await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error("Packed loader execution timed out")), 15000);
      worker.once("message", value => { received = value; });
      worker.once("error", reject);
      worker.once("exit", code => code === 0 && received ? resolve(received) : reject(new Error(`Worker exited ${code}`)));
    });
  } finally { clearTimeout(timer); await worker.terminate(); }
}

test("packed source loaders and Meshopt preserve callbacks, manager URLs and decoded data without files or HTTP", async () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const model = JSON.stringify({ asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }], meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    buffers: [{ uri: "geometry%20%23.bin?buffer=1#position", byteLength: positions.byteLength }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }] });
  const f = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import * as THREE from ${JSON.stringify(core.href)};
      import {GLTFLoader} from ${JSON.stringify(gltf.href)}; export {THREE};
      import {MeshoptDecoder} from ${JSON.stringify(new URL("../../upstream/three.js/examples/jsm/libs/meshopt_decoder.module.js", import.meta.url).href)};
      export async function run(){
        const sequence=[], loader=new THREE.FileLoader().setPath('resources/');
        const first=loader.loadAsync('message%20%23.txt?version=1#text').then(value=>{sequence.push('loaded');return value;});
        const second=loader.loadAsync('message%20%23.txt?version=1#text');sequence.push('returned');
        const declaredModifierTarget=()=>loader.loadAsync('message%20%23.txt?version=2#text');
        if(globalThis.__loadDeclaredVariant)await declaredModifierTarget();
        const text=await first,duplicate=await second;
        const cube=await new THREE.CubeTextureLoader().setPath('faces/').loadAsync(['px.png','nx.png','py.png','ny.png','pz.png','nz.png']);
        const bitmap=await new THREE.ImageBitmapLoader().loadAsync('faces/px.png');
        const loaded=await new GLTFLoader().setPath('models/').setResourcePath('shared/').loadAsync('triangle.gltf');
        const raw=await new THREE.FileLoader().loadAsync('models/triangle.gltf');
        const meshopt=await MeshoptDecoder.decodeGltfBufferAsync(4,12,new Uint8Array(${JSON.stringify([...vertex])}),'ATTRIBUTES');
        return {text,duplicate,sequence,cube:cube.images.length,distinctFaces:new Set(cube.images).size,
          bitmap:bitmap.byteLength,positions:[...loaded.scene.children[0].geometry.attributes.position.array],raw,meshopt:[...meshopt]};
      }`,
    "resources/message #.txt": "original file bytes",
    "models/triangle.gltf": model,
    "shared/geometry #.bin": Buffer.from(positions.buffer),
    ...Object.fromEntries(["px", "nx", "py", "ny", "pz", "nz"].map(face => [`faces/${face}.png`, pixel])),
  });
  const sourceBefore = fs.readFileSync(core);
  await buildApplication(f.entry, f.out, { prepareStandaloneLoaders: true });
  packHtml(path.join(f.out, "index.html"), f.packed);
  const html = fs.readFileSync(f.packed, "utf8");
  fs.renameSync(path.join(f.root, "source"), path.join(f.root, "source-unavailable"));
  fs.renameSync(f.out, path.join(f.root, "build-unavailable"));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  assert.deepEqual(result.result, { text: "original file bytes", duplicate: "original file bytes",
    sequence: ["returned", "loaded"], cube: 6, distinctFaces: 6, bitmap: pixel.length, positions: [...positions], raw: model,
    meshopt: [...vertexExpected] });
  assert.equal(result.requests.length, 5, "parallel FileLoader duplicates still issue only one request");
  assert.equal(result.images.length, 6);
  assert.equal(new Set(result.images).size, 6, "separate source resources retain distinct embedded identities");
  assert.ok(result.requests.some(request => request.mime.startsWith("model/gltf+json")));
  assert.ok(result.events.some(([kind, url]) => kind === "end" && url === "resources/message%20%23.txt?version=2#text"));
  assert.ok(result.events.some(([kind, url]) => kind === "end" && url === "shared/geometry%20%23.bin?buffer=1#position"));
  assert.ok(result.events.every(([, url]) => !url.startsWith("data:")), "manager sees logical resolved URLs, not embedded host URLs");
  assert.deepEqual(fs.readFileSync(core), sourceBefore, "the pinned source oracle is never edited");
});

test("ObjectLoader saved scenes retain external images, cube faces and edited JSON round trips offline", async () => {
  const faces = ["px", "nx", "py", "ny", "pz", "nz"];
  const model = JSON.stringify({metadata: {version: 4.6, type: "Object"},
    geometries: [{uuid: "box", type: "BoxGeometry", width: 1, height: 2, depth: 3}],
    images: [{uuid: "paint-image", url: "paint%20%23.png?edition=1#surface"},
      {uuid: "cube-image", url: faces.map(face => `faces/${face}.png`)},
      {uuid: "data-image", url: {data: [10, 20, 30, 255], width: 1, height: 1, type: "Uint8Array"}}],
    textures: [{uuid: "paint", image: "paint-image", repeat: [2, 3], colorSpace: "srgb"},
      {uuid: "cube", image: "cube-image", mapping: 301}, {uuid: "data", image: "data-image"}],
    materials: [{uuid: "material", type: "MeshBasicMaterial", map: "paint", envMap: "cube", alphaMap: "data"}],
    object: {uuid: "scene", type: "Scene", userData: {source: "saved scene"},
      children: [{uuid: "mesh", type: "Mesh", name: "saved mesh", geometry: "box", material: "material"}]},
  });
  const images = Object.fromEntries(["models/a/", "models/prefix/", "shared/", "pending-images/"].flatMap(base => [
    [`${base}paint #.png`, pixel], ...faces.map(face => [`${base}faces/${face}.png`, pixel]),
  ]));
  const f = fixture({"index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import * as THREE from ${JSON.stringify(core.href)}; export {THREE};
      export async function run() {
        const order=[], loader=new THREE.ObjectLoader();
        const pending=loader.loadAsync('models/a/scene.json'); order.push('returned');
        const first=await pending; order.push('loaded');
        const second=await loader.loadAsync('models/b/scene.json');
        const callback=await new Promise((resolve,reject)=>new THREE.ObjectLoader().setPath('models/prefix/')
          .load('nested/scene.json',resolve,undefined,reject));
        const explicit=await new THREE.ObjectLoader().setPath('models/override/').setResourcePath('shared/')
          .loadAsync('nested/scene.json');
        const changed=new THREE.ObjectLoader(), changing=changed.loadAsync('models/pending/scene.json');
        changed.setResourcePath('pending-images/');
        const late=await changing;
        const a=first.children[0], b=second.children[0];
        a.position.x=9; a.name='edited mesh'; a.updateMatrix();
        const restored=await new THREE.ObjectLoader().parseAsync(first.toJSON());
        const raw=await new THREE.FileLoader().loadAsync('models/a/scene.json');
        return {order,kind:first.type,vertices:a.geometry.attributes.position.count,
          independent:first!==second&&a.geometry!==b.geometry&&a.material!==b.material&&a.material.map!==b.material.map,
          stickyImage:a.material.map.image.src===b.material.map.image.src,
          cube:a.material.envMap.images.length,cubeFaces:new Set(a.material.envMap.images.map(image=>image.src)).size,
          data:[...a.material.alphaMap.image.data],repeat:a.material.map.repeat.toArray(),
          callback:callback.children[0].name,explicit:explicit.children[0].name,
          late:late.children[0].name,
          restored:{x:restored.children[0].position.x,name:restored.children[0].name,userData:restored.userData},
          unchangedX:b.position.x,raw};
      }`,
    "models/a/scene.json": model, "models/b/scene.json": model,
    "models/prefix/nested/scene.json": model, "models/override/nested/scene.json": model,
    "models/pending/scene.json": model, ...images,
  });
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(images)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, "index.html"), f.packed);
  const html = fs.readFileSync(f.packed, "utf8");
  fs.renameSync(path.join(f.root, "source"), path.join(f.root, "source-unavailable"));
  fs.renameSync(f.out, path.join(f.root, "build-unavailable"));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  assert.deepEqual(result.result, {order: ["returned", "loaded"], kind: "Scene", vertices: 24,
    independent: true, stickyImage: true, cube: 6, cubeFaces: 6, data: [10, 20, 30, 255], repeat: [2, 3],
    callback: "saved mesh", explicit: "saved mesh",
    late: "saved mesh",
    restored: {x: 9, name: "edited mesh", userData: {source: "saved scene"}}, unchangedX: 0, raw: model});
  for (const base of ["models/a/", "models/prefix/", "shared/", "pending-images/"]) {
    assert.ok(result.events.some(([kind, url]) => kind === "end" && url === base + "paint%20%23.png?edition=1#surface"));
  }
  assert.ok(result.events.some(([kind, url]) => kind === "end" && url.startsWith("data:image/png;")),
    "re-import retains the data-image URLs written by the source JSON exporter");
});

test("packed HDR/EXR environments, fonts and geometry execute retained addon parsers without network or source files", async () => {
  const loaderUrl = name => new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href;
  const hdr = Buffer.concat([Buffer.from('#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n'), Buffer.from([255, 128, 64, 128])]);
  // Existing native OpenEXR ZIP fixture: 8x4 RGBA HALF, red x/2,
  // green 1/2, blue 1/4, alpha 3/4. No candidate encoder makes these bytes.
  const exr = Buffer.from('di8xAQIAAABjaGFubmVscwBjaGxpc3QASQAAAEEAAQAAAAAAAAABAAAAAQAAAEIAAQAAAAAAAAABAAAAAQAAAEcAAQAAAAAAAAABAAAAAQAAAFIAAQAAAAAAAAABAAAAAQAAAABjb21wcmVzc2lvbgBjb21wcmVzc2lvbgABAAAAA2RhdGFXaW5kb3cAYm94MmkAEAAAAAAAAAAAAAAABwAAAAMAAABkaXNwbGF5V2luZG93AGJveDJpABAAAAAAAAAAAAAAAAcAAAADAAAAbGluZU9yZGVyAGxpbmVPcmRlcgABAAAAAHBpeGVsQXNwZWN0UmF0aW8AZmxvYXQABAAAAAAAgD9zY3JlZW5XaW5kb3dDZW50ZXIAdjJmAAgAAAAAAAAAAAAAAHNjcmVlbldpbmRvd1dpZHRoAGZsb2F0AAQAAAAAAIA/AFMBAAAAAAAAAAAAAB8AAAB4nGNoGFiwC0pXQekWKO2xo6WpqbGxsZzG8gDgUX/E', 'base64');
  const font = JSON.stringify({familyName: 'Packed font', resolution: 100,
    boundingBox: {yMin: 0, yMax: 100}, underlineThickness: 0,
    glyphs: {A: {ha: 100, o: 'm 0 0 l 100 0 l 0 100 l 0 0'}}});
  const positions = [0, 0, 0, 1, 0, 0, 0, 1, 0];
  const resources = {
    'env/sky #.hdr': hdr, 'env/sky.exr': exr, 'fonts/typeface.json': Buffer.from(font),
    'geometry/triangle.stl': Buffer.from('solid triangle\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid triangle\n'),
    'geometry/triangle.obj': Buffer.from('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'),
    'geometry/triangle.ply': Buffer.from('ply\nformat ascii 1.0\nelement vertex 3\nproperty float px\nproperty float y\nproperty float z\nproperty float alpha\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 0.25\n1 0 0 0.5\n0 1 0 1\n3 0 1 2\n'),
  };
  const names = ['HDRLoader', 'RGBELoader', 'EXRLoader', 'FontLoader', 'STLLoader', 'PLYLoader', 'OBJLoader'];
  const f = fixture({'index.html': '<script type="module" src="modules/app.mjs"></script>',
    'modules/app.mjs': `import {FloatType, RGBAFormat} from ${JSON.stringify(core.href)};
      ${names.map(name => `import {${name}} from ${JSON.stringify(loaderUrl(name))};`).join('\n')}
      export async function run() {
        const sequence=[], hdrLoader=new HDRLoader();
        hdrLoader.setDataType(FloatType).setPath('env/');
        const pending=hdrLoader.loadAsync('sky%20%23.hdr?version=1#radiance').then(value=>{sequence.push('loaded');return value;});
        sequence.push('returned'); const hdr=await pending;
        const rgbe=await new RGBELoader().setDataType(FloatType).setPath('env/').loadAsync('sky%20%23.hdr?version=1#radiance');
        const exr=await new EXRLoader().setPart(0).setOutputFormat(RGBAFormat).setDataType(FloatType).setPath('env/').loadAsync('sky.exr');
        const font=await new Promise((resolve,reject)=>new FontLoader().setPath('fonts/').load('typeface.json',resolve,undefined,reject));
        const stl=await new STLLoader().setPath('geometry/').loadAsync('triangle.stl');
        const plyLoader=new PLYLoader();
        plyLoader.setPropertyNameMapping({px:'x'}); plyLoader.setCustomPropertyNameMapping({opacity:['alpha']});
        const ply=await plyLoader.setPath('geometry/').loadAsync('triangle.ply');
        const obj=await new OBJLoader().setPath('geometry/').loadAsync('triangle.obj');
        return {sequence,hdr:{width:hdr.image.width,height:hdr.image.height,pixels:[...hdr.image.data]},
          rgbe:[...rgbe.image.data],exr:{width:exr.image.width,height:exr.image.height,pixels:[...exr.image.data]},
          font:{name:font.data.familyName,shapes:font.generateShapes('A',1).length},
          stl:[...stl.attributes.position.array],ply:[...ply.attributes.position.array],
          opacity:[...ply.attributes.opacity.array],obj:[...obj.children[0].geometry.attributes.position.array]};
      }`, ...resources});
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(resources)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html), radiance = [1, Math.fround(128 / 255), Math.fround(64 / 255), 1];
  assert.equal(result.ok, true, result.error?.split('\n')[0]);
  assert.deepEqual(result.result, {sequence: ['returned', 'loaded'],
    hdr: {width: 1, height: 1, pixels: radiance}, rgbe: radiance,
    exr: {width: 8, height: 4, pixels: Array.from({length: 4}, () =>
      Array.from({length: 8}, (_, x) => [x / 2, .5, .25, .75]).flat()).flat()},
    font: {name: 'Packed font', shapes: 1}, stl: positions, ply: positions,
    opacity: [.25, .5, 1], obj: positions});
  assert.equal(result.requests.length, 7, 'each retained load reaches the embedded host route');
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
});

test("resource lookup preserves unknown and manager-owned URLs and exact bytes with real MIME", async () => {
  const bytes = Buffer.from([0, 255, 123, 9]);
  const source = createThreeLoaderResourceModule(new Map([["site/a #.bin", { source: bytes }]]), {
    logicalURLs: new Map([["site/a #.bin", new Set(["site/a%20%23.bin?q=1#part", "site/a%20%23.bin?q=2#part"])]]),
  });
  const previous = globalThis.document;
  globalThis.document = { URL: "https://example.test/view/index.html", baseURI: "https://example.test/view/site/" };
  try {
    const { resolveThreeLoaderResource: resolve } = await import(data(source));
    const url = resolve("a%20%23.bin?q=1#part");
    assert.match(url, /^data:application\/octet-stream;f3d-id=/);
    assert.ok(url.endsWith("#part"));
    assert.deepEqual(Buffer.from(await (await fetch(url)).arrayBuffer()), bytes);
    assert.notEqual(resolve("a%20%23.bin?q=2#part"), url, "declared query identities remain distinct");
    for (const url of ["runtime.json", "a%20%23.bin?q=3#part", "a%20%23.bin?q=1#other",
      "https://other.test/site/a%20%23.bin", "data:text/plain,managed", "blob:https://example.test/id"])
      assert.equal(resolve(url), url);
    globalThis.document.URL = "https://example.test/other/index.html";
    globalThis.document.baseURI = "https://example.test/other/site/";
    assert.equal(resolve("a%20%23.bin?q=1#part"), "a%20%23.bin?q=1#part", "history changes do not rebase embedded identities");
    assert.throws(() => createThreeLoaderResourceModule(new Map([["large.bin", { source: bytes }]]), { maxBytes: 1 }), /RESOURCE_LIMIT/);
  } finally { if (previous === undefined) delete globalThis.document; else globalThis.document = previous; }
});

test("host-call transformation requires both pinned module identity and exact source bytes", () => {
  const source = fs.readFileSync(core, "utf8"), ast = parse(source, { ecmaVersion: "latest", sourceType: "module" });
  assert.equal(threeLoaderResourceEdits(source, ast, core.href).filter(edit => edit.type === "replace").length, 3);
  assert.deepEqual(threeLoaderResourceEdits(source, ast, "file:///other/three.core.js"), []);
  const modified = source.replace("const req = new Request( url, {", "const req = new Request( url /* custom */, {");
  const changed = threeLoaderResourceEdits(modified, parse(modified, { ecmaVersion: "latest", sourceType: "module" }), core.href);
  assert.deepEqual(changed, [], "a modified implementation keeps its authored host route");
  const shadowed = "const Request = globalThis.customRequest;\n" + source;
  assert.deepEqual(threeLoaderResourceEdits(shadowed, parse(shadowed, { ecmaVersion: "latest", sourceType: "module" }), core.href), []);
});

test("only exact pinned Draco, KTX2 and Meshopt source constructors gain the native worker host hook", () => {
  for (const [file, argument, original] of [
    ["loaders/DRACOLoader.js", "this.workerSourceURL", "new Worker( this.workerSourceURL )"],
    ["loaders/KTX2Loader.js", "this.workerSourceURL", "new Worker( this.workerSourceURL )"],
    ["libs/meshopt_decoder.module.js", "url", "new Worker(url)"],
  ]) {
    const url = new URL(`../../upstream/three.js/examples/jsm/${file}`, import.meta.url);
    const source = fs.readFileSync(url, "utf8"), ast = parse(source, { ecmaVersion: "latest", sourceType: "module" });
    const edits = threeLoaderResourceEdits(source, ast, url.href);
    const replacements = edits.filter(edit => edit.type === "replace");
    assert.equal(replacements.length, 1);
    assert.equal(source.slice(replacements[0].pos, replacements[0].end), original);
    assert.equal(replacements[0].text, `__f3dCreateLoaderWorker(Worker, ${argument})`);
    assert.match(edits.find(edit => edit.type === "insert").text, /createThreeLoaderWorker/);
    assert.deepEqual(threeLoaderResourceEdits(source, ast, `file:///unrelated/${file}`), []);
    for (const changed of [source + "\n// custom module\n", "const Worker = globalThis.CustomWorker;\n" + source]) {
      assert.deepEqual(threeLoaderResourceEdits(changed, parse(changed, { ecmaVersion: "latest", sourceType: "module" }), url.href), []);
    }
    assert.equal(fs.readFileSync(url, "utf8"), source, "the codec, Blob creation and worker lifecycle stay in pinned source");
  }
});

test("the generated codec-worker host hook forwards native construction, options, result identity and exceptions", async () => {
  const source = createThreeLoaderResourceModule(new Map());
  assert.equal(isThreeLoaderResourceModule(source, parse(source, { ecmaVersion: "latest", sourceType: "module" })), true);
  const previous = Object.getOwnPropertyDescriptor(globalThis, "Worker"), calls = [], rejected = {};
  class NativeWorkerHost {
    constructor(...args) {
      if (args[0] === rejected) throw new TypeError("native constructor failure");
      calls.push({ args, target: new.target, worker: this });
    }
  }
  Object.defineProperty(globalThis, "Worker", { configurable: true, writable: true, value: NativeWorkerHost });
  try {
    const { createThreeLoaderWorker } = await import(data(source));
    const url = new URL("blob:https://example.test/source-codec"), options = { name: "original", type: "classic" };
    const worker = createThreeLoaderWorker(globalThis.Worker, url, options);
    assert.ok(worker instanceof NativeWorkerHost);
    assert.equal(calls[0].worker, worker);
    assert.equal(calls[0].target, NativeWorkerHost);
    assert.equal(calls[0].args[0], url);
    assert.equal(calls[0].args[1], options);
    const changing = { get url() { globalThis.Worker = class Replacement {}; return url; } };
    assert.ok(createThreeLoaderWorker(globalThis.Worker, changing.url) instanceof NativeWorkerHost,
      "constructor lookup precedes URL getter side effects, as in the original new Worker expression");
    assert.throws(() => createThreeLoaderWorker(NativeWorkerHost, rejected), /native constructor failure/);
  } finally {
    if (previous) Object.defineProperty(globalThis, "Worker", previous);
    else delete globalThis.Worker;
  }
});

test("packing accepts the exact emitted worker runtime and refuses lookalikes or arbitrary application workers", () => {
  const source = createThreeLoaderResourceModule(new Map());
  const main = `import {createThreeLoaderWorker} from './f3d-loader-resources.mjs';export {createThreeLoaderWorker};`;
  const f = fixture({ "index.html": '<script type="module" src="app.mjs"></script>', "app.mjs": main,
    "f3d-loader-resources.mjs": source });
  packHtml(f.entry, f.packed);
  assert.ok(fs.readFileSync(f.packed, "utf8").includes("f3d-packed/"));
  for (const changed of [
    source.replace("new Worker(...args)", "new Worker(globalThis.target)"),
    "const Worker = globalThis.CustomWorker;\n" + source,
    source + "\nnew Worker(globalThis.target);\n",
  ]) {
    assert.equal(isThreeLoaderResourceModule(changed, parse(changed, { ecmaVersion: "latest", sourceType: "module" })), false);
    const f = fixture({ "index.html": '<script type="module" src="app.mjs"></script>', "app.mjs": main,
      "f3d-loader-resources.mjs": changed });
    assert.throws(() => packHtml(f.entry, f.packed), error => ["WORKER_URL", "WORKER_BINDING"].includes(error.code));
  }
  const arbitrary = fixture({ "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": main + "\nnew Worker(globalThis.target);", "f3d-loader-resources.mjs": source });
  assert.throws(() => packHtml(arbitrary.entry, arbitrary.packed), { code: "WORKER_URL" });
});

test("CLI build plus pack enables embedded loaders while ordinary application builds stay unchanged", async () => {
  const f = fixture({ "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import {FileLoader} from ${JSON.stringify(core.href)};export const run=()=>new FileLoader().loadAsync('message.txt');`,
    "message.txt": "from CLI" });
  const plain = await buildApplication(f.entry, path.join(f.root, "ordinary"));
  assert.ok(!plain.emittedFiles.includes("f3d-loader-resources.mjs"));
  const child = spawnSync(process.execPath, [fileURLToPath(new URL("./cli.mjs", import.meta.url)),
    f.entry, "--build-app", f.out, "--pack-html", f.packed], { encoding: "utf8", timeout: 30000 });
  assert.equal(child.status, 0, child.stderr);
  const helper = fs.readFileSync(path.join(f.out, "f3d-loader-resources.mjs"), "utf8");
  assert.ok(helper.includes(Buffer.from("from CLI").toString("base64")));
  const html = fs.readFileSync(f.packed, "utf8");
  assert.ok(html.includes("f3d-packed/"));
  const sources = Object.values(JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports)
    .map(url => Buffer.from(url.split(",")[1].split("#")[0], "base64").toString());
  assert.ok(sources.some(source => source.includes(Buffer.from("from CLI").toString("base64"))));
});
