import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";
import { parse } from "acorn";
import { buildApplication } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";
import { findThreeLoaderRequests } from "./three_loader_assets.mjs";

const upstream = new URL("../../upstream/three.js/", import.meta.url);
const core = new URL("build/three.core.js", upstream).href;
const discover = (source, options = {}) => findThreeLoaderRequests(parse(source, { ecmaVersion: "latest", sourceType: "module" }), {
  resolveModule: specifier => specifier === "three" ? core : specifier,
  ...options,
}).map(request => request.url);
const data = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");

test("local pinned managers retain finite core-loader assets and observation callbacks", () => {
  const sourceManager = new URL("src/loaders/LoadingManager.js", upstream).href;
  assert.deepEqual(discover(`import * as THREE from 'three';
    import {LoadingManager as SourceManager} from ${JSON.stringify(sourceManager)};
    import * as Source from ${JSON.stringify(sourceManager)};
    const Manager = THREE.LoadingManager, events = [];
    const progress = (url, done, total) => events.push([url, done, total]);
    const manager = new Manager(function(){ events.push(this === manager); }, progress,
      function(url){ events.push([this === manager, url]); });
    manager.onStart = function(url){ events.push([this === manager, url]); };
    manager['onProgress'] = progress;
    const files = new THREE.FileLoader(manager);
    files.setPath('assets/'); files.loadAsync('hello.txt');
    const images = new THREE.TextureLoader(manager).setPath('images/'); images.load('pixel.png');
    new THREE.CubeTextureLoader(manager).load(['px.png','nx.png','py.png','ny.png','pz.png','nz.png']);
    new THREE.ObjectLoader(manager).setPath('models/').load('scene.json');
    export function cancel(){ manager.abort(); }
    new THREE.BufferGeometryLoader(new SourceManager()).load('geometry.json');
    const other = new Source.LoadingManager(undefined, () => events.push('other'));
    new THREE.ImageLoader(other).load('other.png');
    new THREE.ImageBitmapLoader(other).load('bitmap.png');
    new THREE.AudioLoader(other).load('clip.ogg');`), [
    'assets/hello.txt', 'images/pixel.png', 'px.png', 'nx.png', 'py.png', 'ny.png', 'pz.png', 'nz.png',
    'models/scene.json', 'geometry.json', 'other.png', 'bitmap.png', 'clip.ogg',
  ]);
});

test("changed, escaping and addon-shared managers keep their authored host routes", () => {
  const imports = `import {LoadingManager,FileLoader,TextureLoader} from 'three';
    import {GLTFLoader} from ${JSON.stringify(new URL('examples/jsm/loaders/GLTFLoader.js', upstream).href)};
    import {MTLLoader} from ${JSON.stringify(new URL('examples/jsm/loaders/MTLLoader.js', upstream).href)};`;
  const cases = [
    "manager.setURLModifier(url => 'data:text/plain,changed');",
    "manager.addHandler(/png$/, new TextureLoader());",
    "manager.onProgress = globalThis.observe;",
    "manager.onProgress = function(){ this.setURLModifier(() => 'changed'); };",
    "manager.onLoad = function(){ const self = this; globalThis.customize(self); };",
    "manager.onError = () => manager.setURLModifier(() => 'changed');",
    "manager.onStart = function(){ return () => this; };",
    "manager.onStart = function(){ eval('this.setURLModifier(() => 1)'); };",
    "const alias = manager;",
    "export {manager};",
    "globalThis.manager = manager;",
    "export function expose(){ return manager; }",
    "const loader = new FileLoader(manager); globalThis.customize(loader);",
    "const loader = new FileLoader(manager); export {loader};",
    "export const loader = new FileLoader(manager);",
    "const loader = new FileLoader(manager); loader.manager.setURLModifier(() => 'changed');",
    "globalThis.customize(new FileLoader(manager).setPath('escape/'));",
    "globalThis.customize(manager.abort());",
    "new GLTFLoader(manager).load('scene.gltf');",
    "new MTLLoader(manager).load('scene.mtl');",
    "new globalThis.OtherLoader(manager);",
    "eval('manager.setURLModifier(() => 1)');",
  ];
  for (const effect of cases) assert.deepEqual(discover(`${imports}
    const manager = new LoadingManager(); ${effect}
    new FileLoader(manager).load('not-proven.txt');`), [], effect);
  for (const source of [
    "export const manager = new LoadingManager(); new FileLoader(manager).load('exported.txt');",
    "export const loader = new FileLoader(new LoadingManager()); loader.load('inline-export.txt');",
    "let manager = new LoadingManager(); new FileLoader(manager).load('mutable.txt');",
    "const manager = new LoadingManager(globalThis.observe); new FileLoader(manager).load('callback.txt');",
    "new FileLoader(new LoadingManager(function(){ globalThis.customize(this); })).load('receiver.txt');",
    "function local(LoadingManager){ new FileLoader(new LoadingManager()).load('shadowed.txt'); }",
    "const manager = new LoadingManager(); manager.onLoad = function(){ return this == manager; }; new FileLoader(manager).load('coercion.txt');",
  ]) assert.deepEqual(discover(imports + source), [], source);
  assert.deepEqual(discover(`import {LoadingManager} from 'file:///foreign/src/loaders/LoadingManager.js';
    import {FileLoader} from 'three'; new FileLoader(new LoadingManager()).load('foreign.txt');`), []);
});

test("constructor and namespace exposure disables only local-manager discovery", () => {
  for (const effect of [
    "Object.defineProperty(LoadingManager.prototype, 'onProgress', {set(value){ globalThis.observe(this); }});",
    "const Manager=LoadingManager; Object.assign(Manager.prototype,globalThis.methods);",
    "globalThis.customize(LoadingManager);",
    "export {LoadingManager};",
    "const Manager=LoadingManager; export {Manager};",
    "globalThis.customize(THREE.LoadingManager.prototype);",
    "globalThis.customize(THREE['Loading'+'Manager']);",
    "globalThis.customize(THREE);",
    "const Core=THREE;",
    "export {THREE};",
  ]) {
    let exposed = 0;
    const source = `import * as THREE from 'three'; import {LoadingManager,FileLoader} from 'three';
      ${effect}
      const manager=new LoadingManager(); new FileLoader(manager).load('managed.txt');
      new FileLoader().load('default.txt');`;
    assert.deepEqual(discover(source, { onManagerConstructorExposure: () => { exposed++; } }), ['default.txt'], effect);
    assert.equal(exposed, 1, effect);
  }
});

test("application re-exports expose constructors while pinned library forwarding does not", () => {
  for (const declaration of [
    `export {LoadingManager as Manager} from ${JSON.stringify(core)};`,
    `export * from ${JSON.stringify(core)};`,
    `export * as Core from ${JSON.stringify(core)};`,
    `export {LoadingManager} from ${JSON.stringify(new URL('src/loaders/LoadingManager.js', upstream).href)};`,
  ]) {
    let exposed = 0;
    assert.deepEqual(discover(declaration, { moduleUrl: 'file:///app/bridge.mjs',
      onManagerConstructorExposure: () => { exposed++; } }), []);
    assert.equal(exposed, 1, declaration);
    assert.deepEqual(discover(`import {LoadingManager,FileLoader} from 'three'; ${declaration}
      new FileLoader(new LoadingManager()).load('managed.txt');`), [], declaration);
    for (const file of ['build/three.module.js', 'src/Three.Core.js']) {
      let libraryExposure = 0;
      discover(declaration, { moduleUrl: new URL(file, upstream).href,
        onManagerConstructorExposure: () => { libraryExposure++; } });
      assert.equal(libraryExposure, 0, file);
    }
  }
  for (const declaration of [
    `export {Vector3} from ${JSON.stringify(core)};`,
    `export * from ${JSON.stringify(new URL('build/three.tsl.js', upstream).href)};`,
    "export * from 'file:///other/build/three.core.js';",
  ]) {
    let exposed = 0;
    discover(declaration, { onManagerConstructorExposure: () => { exposed++; } });
    assert.equal(exposed, 0, declaration);
  }
});

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-local-manager-"));
  for (const [name, bytes] of Object.entries(files)) {
    const filename = path.join(root, "source", name);
    fs.mkdirSync(path.dirname(filename), { recursive: true }); fs.writeFileSync(filename, bytes);
  }
  return { root, source: path.join(root, "source"), entry: path.join(root, "source/index.html") };
}

// Native source modules, promises, fetch data URLs and streams execute here.
// This small host supplies document-relative Request resolution and image load
// events; byte identity and loader behavior, not browser image decoding, are tested.
const runner = new URL(data(`
import {parentPort,workerData} from 'node:worker_threads';
import {registerHooks} from 'node:module';
import fs from 'node:fs'; import path from 'node:path';
registerHooks({resolve(specifier,context,next){return next(workerData.imports[specifier]??specifier,context);}});
const requests=[],nativeFetch=globalThis.fetch,NativeRequest=globalThis.Request;
globalThis.document={URL:'https://app.example/view/index.html',baseURI:'https://app.example/view/',
  createElementNS(_namespace,tag){if(tag!=='img')throw new Error(tag);return new ImageHost();}};
globalThis.Request=class extends NativeRequest {
  constructor(input,options){super(typeof input==='string'?new URL(input,document.baseURI):input,options);}
};
globalThis.self=globalThis;
globalThis.ProgressEvent=class extends Event {constructor(type,details){super(type);Object.assign(this,details);}};
globalThis.fetch=async (input,options)=>{
  const url=input instanceof NativeRequest?input.url:new URL(String(input),document.baseURI).href;
  let response;
  if(url.startsWith('data:'))response=await nativeFetch(input,options);
  else if(workerData.root&&url.startsWith(document.baseURI)){
    const name=decodeURIComponent(new URL(url).pathname.slice('/view/'.length));
    response=new Response(fs.readFileSync(path.join(workerData.root,name)));
  } else throw new Error('Undeclared request: '+url);
  requests.push({url,bytes:[...new Uint8Array(await response.clone().arrayBuffer())]});
  return response;
};
class ImageHost extends EventTarget {
  get src(){return this._src;}
  set src(url){this._src=url;Promise.resolve().then(async()=>{
    this.bytes=[...new Uint8Array(await (await fetch(url)).arrayBuffer())];
    if(this.bytes[0]!==137||this.bytes[1]!==80)throw new Error('Invalid PNG');
    this.complete=true;this.naturalWidth=1;this.naturalHeight=1;this.dispatchEvent(new Event('load'));
  }).catch(()=>this.dispatchEvent(new Event('error')));}
}
globalThis.HTMLImageElement=ImageHost;globalThis.Image=ImageHost;
try{const app=await import(workerData.url);parentPort.postMessage({ok:true,result:await app.run(),requests});}
catch(error){parentPort.postMessage({ok:false,error:error.stack});}
`));
async function execute(url, { root = null, imports = {} } = {}) {
  const worker = new Worker(runner, { workerData: { url, root, imports } });
  let timer, result;
  try {
    return await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error("Managed loader execution timed out")), 15000);
      worker.once("message", value => { result = value; });
      worker.once("error", reject);
      worker.once("exit", code => code === 0 && result ? resolve(result) : reject(new Error(`Worker exited ${code}`)));
    });
  } finally { clearTimeout(timer); await worker.terminate(); }
}
function packedModule(html) {
  return {
    url: /<script type="module" src="([^"]+)"/.exec(html)[1],
    imports: JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports,
  };
}

test("managed text, textures and ObjectLoader scenes survive relocation and packing with source lifecycle behavior", async () => {
  const pixel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==", "base64");
  const positions = [0, 0, 0, 1, 0, 0, 0, 1, 0];
  const model = JSON.stringify({ metadata: { version: 4.6, type: "Object" },
    geometries: [{ uuid: "geometry", type: "BufferGeometry", data: { attributes: {
      position: { itemSize: 3, type: "Float32Array", array: positions, normalized: false },
    } } }], materials: [{ uuid: "material", type: "MeshBasicMaterial", color: 0x5588cc }],
    object: { uuid: "scene", type: "Scene", children: [{ uuid: "mesh", type: "Mesh", name: "saved triangle", geometry: "geometry", material: "material" }] },
  });
  const assets = { "assets/message #.txt": "managed text bytes", "images/pixel #.png": pixel,
    "images/bad.png": "invalid image bytes", "models/scene.json": model };
  const f = fixture({ "index.html": '<script type="module" src="app.mjs"></script>', ...assets,
    "app.mjs": `import {LoadingManager,FileLoader,TextureLoader,ObjectLoader} from ${JSON.stringify(core)};
      export async function run(){
        const events=[]; let resolveIdle, progressCalls=0;
        const idle=()=>new Promise(resolve=>{resolveIdle=resolve;});
        const complete=function(){events.push(['complete',this===manager]);resolveIdle();};
        const progress=(url,done,total)=>{progressCalls++;events.push(['progress',url,done,total]);};
        const error=url=>events.push(['error',url]);
        const manager=new LoadingManager(complete,progress,error);
        manager.onStart=function(url,done,total){events.push(['start',this===manager,url,done,total]);};
        manager.onProgress=progress;
        const files=new FileLoader(manager).setPath('assets/');
        const textures=new TextureLoader(manager).setPath('images/');
        const scenes=new ObjectLoader(manager).setPath('models/');
        let done=idle(); const textTask=files.loadAsync('message%20%23.txt?version=1#text');
        events.push(['text-returned']); const text=await textTask; await done;
        done=idle(); let returned;
        const texture=await new Promise((resolve,reject)=>{
          returned=textures.load('pixel%20%23.png?version=2#image',value=>{
            events.push(['image-loaded',value===returned]);resolve(value);
          },undefined,reject);
          events.push(['image-returned',returned.isTexture]);
        }); await done;
        done=idle(); const failure=await new Promise(resolve=>{
          textures.load('bad.png',()=>resolve('unexpected'),undefined,event=>{
            events.push(['load-error',event.type]);resolve(event.type);
          });
        }); await done;
        done=idle(); const scene=await scenes.loadAsync('scene.json'); await done;
        return {text,pixel:returned.image.bytes,sameTexture:texture===returned,failure,progressCalls,events,
          name:scene.children[0].name,positions:[...scene.children[0].geometry.attributes.position.array]};
      }`,
  });
  const source = await execute(pathToFileURL(path.join(f.source, "app.mjs")).href, { root: f.source });
  assert.equal(source.ok, true, source.error);
  const normal = await buildApplication(f.entry, path.join(f.root, "build"));
  const prepared = await buildApplication(f.entry, path.join(f.root, "standalone"), { prepareStandaloneLoaders: true });
  const relocated = path.join(f.root, "relocated"), relocatedPrepared = path.join(f.root, "relocated-standalone");
  fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
  fs.renameSync(normal.outDir, relocated); fs.renameSync(prepared.outDir, relocatedPrepared);
  for (const [name, bytes] of Object.entries(assets)) {
    assert.deepEqual(fs.readFileSync(path.join(relocated, name)), Buffer.from(bytes), name);
    assert.deepEqual(fs.readFileSync(path.join(relocatedPrepared, name)), Buffer.from(bytes), name);
  }
  const built = await execute(pathToFileURL(path.join(relocated, normal.entryFiles[0])).href, { root: relocated });
  assert.equal(built.ok, true, built.error); assert.deepEqual(built.result, source.result);
  const output = path.join(f.root, "packed.html");
  packHtml(path.join(relocatedPrepared, "index.html"), output);
  const packed = packedModule(fs.readFileSync(output, "utf8"));
  fs.renameSync(relocatedPrepared, path.join(f.root, "standalone-unavailable"));
  const offline = await execute(packed.url, { imports: packed.imports });
  assert.equal(offline.ok, true, offline.error); assert.deepEqual(offline.result, source.result);
  assert.ok(offline.requests.every(request => request.url.startsWith("data:")));
  for (const bytes of Object.values(assets)) assert.ok(offline.requests.some(request => Buffer.from(request.bytes).equals(Buffer.from(bytes))));
  assert.deepEqual(offline.result.positions, positions); assert.deepEqual(offline.result.pixel, [...pixel]);
  assert.equal(offline.result.text, "managed text bytes"); assert.equal(offline.result.name, "saved triangle");
  assert.equal(offline.result.failure, "error"); assert.equal(offline.result.sameTexture, true);
  assert.equal(offline.result.progressCalls, 4);
  const events = offline.result.events, kinds = events.map(event => event[0]);
  assert.ok(kinds.indexOf("start") < kinds.indexOf("text-returned"));
  assert.ok(kinds.indexOf("text-returned") < kinds.indexOf("progress"));
  assert.ok(kinds.indexOf("image-returned") < kinds.indexOf("image-loaded"));
  assert.ok(kinds.indexOf("load-error") < kinds.indexOf("error"));
  assert.equal(events.filter(event => event[0] === "complete").length, 4);
  assert.ok(events.filter(event => ["start", "complete"].includes(event[0])).every(event => event[1] === true));
  assert.deepEqual(events.filter(event => event[0] === "progress").map(event => event.slice(1)), [
    ["assets/message%20%23.txt?version=1#text", 1, 1], ["images/pixel%20%23.png?version=2#image", 2, 2],
    ["images/bad.png", 3, 3], ["models/scene.json", 4, 4],
  ]);
});

test("a manager URL modifier still runs and unrelated local model bytes are not admitted", async () => {
  const f = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "logical.gltf": "not the resource selected by this manager",
    "app.mjs": `import {LoadingManager,FileLoader} from ${JSON.stringify(core)};
      export async function run(){
        const urls=[],manager=new LoadingManager();
        manager.setURLModifier(url=>{urls.push(url);return 'data:text/plain,selected%20by%20manager';});
        const text=await new FileLoader(manager).loadAsync('logical.gltf');return {text,urls};
      }`,
  });
  const result = await buildApplication(f.entry, path.join(f.root, "build"), { prepareStandaloneLoaders: true });
  assert.equal(result.emittedFiles.includes("logical.gltf"), false);
  const output = path.join(f.root, "packed.html"); packHtml(path.join(result.outDir, "index.html"), output);
  const packed = packedModule(fs.readFileSync(output, "utf8"));
  fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
  fs.renameSync(result.outDir, path.join(f.root, "build-unavailable"));
  const execution = await execute(packed.url, { imports: packed.imports });
  assert.equal(execution.ok, true, execution.error);
  assert.deepEqual(execution.result, { text: "selected by manager", urls: ["logical.gltf"] });
});

test("a constructor prototype mutation in another module retains live routing and default-loader assets", async () => {
  for (const [kind, declaration, receiver] of [
    ['direct', `import * as Core from ${JSON.stringify(core)};`, "Core['Loading'+'Manager']"],
    ['named', "import {Manager} from './bridge.mjs';", 'Manager'],
    ['star', "import {LoadingManager} from './bridge.mjs';", 'LoadingManager'],
  ]) {
    const configure = `${declaration}
    Object.defineProperty(${receiver}.prototype,'setURLModifier',{
      configurable:true,set(value){
        Object.defineProperty(this,'setURLModifier',{value,writable:true,configurable:true});
        value.call(this,()=> 'data:text/plain,configured%20resource');
      }
    });`;
    // The direct exposure module has no loader call or constructor-name token;
    // the star-only bridge has no import binding. Both still need to be scanned.
    if (kind === 'direct') {
      assert.equal(configure.includes("load"), false); assert.equal(configure.includes("LoadingManager"), false);
    }
    const f = fixture({
      "index.html": '<script type="module" src="app.mjs"></script>',
      "configure.mjs": configure,
      "bridge.mjs": kind === 'named' ? `export {LoadingManager as Manager} from ${JSON.stringify(core)};`
        : `export * from ${JSON.stringify(core)};`,
      "app.js": "unrelated local bytes must not collide with the application bundle",
      "default.txt": "unchanged default-manager resource",
      "app.mjs": `import './configure.mjs'; import {LoadingManager,FileLoader} from ${JSON.stringify(core)};
      export async function run(){
        const manager=new LoadingManager();
        const selected=await new FileLoader(manager).loadAsync('app.js');
        const original=await new FileLoader().loadAsync('default.txt'); return {selected,original};
      }`,
    });
    const source = await execute(pathToFileURL(path.join(f.source, "app.mjs")).href, { root: f.source });
    assert.equal(source.ok, true, source.error);
    const built = await buildApplication(f.entry, path.join(f.root, "build"));
    const relocated = path.join(f.root, "relocated");
    fs.renameSync(f.source, path.join(f.root, "source-unavailable")); fs.renameSync(built.outDir, relocated);
    assert.equal(fs.readFileSync(path.join(relocated, "default.txt"), "utf8"), "unchanged default-manager resource");
    const result = await execute(pathToFileURL(path.join(relocated, built.entryFiles[0])).href, { root: relocated });
    assert.equal(result.ok, true, result.error); assert.deepEqual(result.result, source.result);
    assert.deepEqual(result.result, { selected: "configured resource", original: "unchanged default-manager resource" });
  }
});

test("pinned re-exports and compiler renderer facades preserve managed core-loader resources", async () => {
  const f = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "asset.txt": "managed resource beside a routed renderer",
    "app.mjs": `import {LoadingManager,FileLoader,WebGLRenderer} from ${JSON.stringify(new URL('build/three.module.js', upstream).href)};
      const renderer=new WebGLRenderer(); renderer.setSize(32,24,false);
      export async function run(){
        const manager=new LoadingManager(); return new FileLoader(manager).loadAsync('asset.txt');
      }`,
  });
  const built = await buildApplication(f.entry, path.join(f.root, "build"), { routeWebGLRenderer: true });
  assert.equal(built.rendererRoute.routed, true);
  assert.equal(fs.readFileSync(path.join(built.outDir, 'asset.txt'), 'utf8'), "managed resource beside a routed renderer");
});
