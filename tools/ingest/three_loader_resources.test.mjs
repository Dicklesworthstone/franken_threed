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
import { fbxResources, fbxPixel } from "./fixtures/fbx_resources.mjs";
import { gzipSync, zipSync, strToU8 } from "../../upstream/three.js/examples/jsm/libs/fflate.module.js";

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
const requests=[],images=[],events=[],nativeFetch=globalThis.fetch,NativeRequest=globalThis.Request;
globalThis.document={URL:'https://packed.example/view/index.html',baseURI:'https://packed.example/view/',
  createElementNS(_namespace,tag){if(tag!=='img')throw new Error(tag);return new ImageHost();}};
// Native browsers resolve relative Request inputs against the document. Keep
// unknown resource requests on that host path so real loader retries can run.
globalThis.Request=class extends NativeRequest {
  constructor(input,options){super(typeof input==='string'?new URL(input,document.baseURI):input,options);}
};
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
    if(!url.startsWith('data:image/png;')&&!url.startsWith('blob:'))throw new Error('Unembedded image: '+url);
    const bytes=new Uint8Array(await (await nativeFetch(url)).arrayBuffer());
    if(bytes[0]!==137||bytes[1]!==80)throw new Error('Image bytes changed');
    this.complete=true;this.naturalWidth=1;this.naturalHeight=1;const event=new Event('load');
    this.dispatchEvent(event);this.onload?.(event);
  }).catch(error=>{const event=new Event('error');this.dispatchEvent(event);this.onerror?.(event);});}
}
globalThis.HTMLImageElement=ImageHost;
globalThis.Image=ImageHost;
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

test("resource-table loader choices execute offline with original effects, model sidecars and independent scenes", async () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const model = JSON.stringify({asset: {version: '2.0'}, scene: 0, scenes: [{nodes: [0]}],
    nodes: [{mesh: 0}], meshes: [{primitives: [{attributes: {POSITION: 0}}]}],
    buffers: [{uri: 'triangle%20%23.bin?version=1#positions', byteLength: positions.byteLength}],
    bufferViews: [{buffer: 0, byteOffset: 0, byteLength: positions.byteLength}],
    accessors: [{bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0]}]});
  const f = fixture({
    'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import { FileLoader } from ${JSON.stringify(core.href)};
      import { GLTFLoader } from ${JSON.stringify(gltf.href)};
      export async function run() {
        const effects=[], paths={files:'data/',models:'models/',buffers:'geometry/'}, base=paths;
        const names={day:(effects.push('initialize'),'day%20%23.txt?version=1#text'),night:'night.txt'}, alias=names;
        const loader=new FileLoader().setPath((effects.push('path'),base.files)), texts=[];
        for(const day of [true,false]) {
          const pending=loader.loadAsync((effects.push('select'),alias[day?'day':'night']));
          effects.push('returned'); texts.push(await pending); effects.push('loaded');
        }
        const models=['triangle.gltf','copy.gltf'], scenes=[];
        const modelsLoader=new GLTFLoader().setPath(paths.models).setResourcePath(paths.buffers);
        for(const first of [true,false]) scenes.push((await modelsLoader.loadAsync(models[first?0:1])).scene);
        let tdz;
        try {
          const pending=new FileLoader().loadAsync(later.entry);
          const later={entry:'data/day%20%23.txt?version=1#text'};
          await pending;
        } catch(error) {tdz=error.name;}
        const first=scenes[0].children[0], second=scenes[1].children[0];
        first.position.x=7;
        return {texts,effects,tdz,logical:[alias.day,paths.models],positions:[...first.geometry.attributes.position.array],
          independent:scenes[0]!==scenes[1]&&first.geometry!==second.geometry&&first.material!==second.material,
          otherPosition:second.position.x};
      }`,
    'data/day #.txt': 'day source bytes',
    'data/night.txt': 'night source bytes',
    'models/triangle.gltf': model,
    'models/copy.gltf': model,
    'geometry/triangle #.bin': Buffer.from(positions.buffer),
  });
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const relative of ['data/day #.txt', 'data/night.txt', 'models/triangle.gltf', 'models/copy.gltf', 'geometry/triangle #.bin']) {
    assert.deepEqual(fs.readFileSync(path.join(f.out, relative)), fs.readFileSync(path.join(f.root, 'source', relative)),
      `normal output preserves original resource bytes: ${relative}`);
  }
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  assert.deepEqual(result.result, {
    texts: ['day source bytes', 'night source bytes'],
    effects: ['initialize', 'path', 'select', 'returned', 'loaded', 'select', 'returned', 'loaded'],
    tdz: 'ReferenceError', logical: ['day%20%23.txt?version=1#text', 'models/'], positions: [...positions],
    independent: true, otherPosition: 0,
  });
  assert.equal(result.requests.length, 6, 'two files, two models and each independent geometry request execute natively');
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
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

test("packed OBJ and MTL retain material textures, creator identity and pending resource paths offline", async () => {
  const mtl = `newmtl Paint
Kd 0.5 0.25 0.75
Ns 24
d 0.6
map_Kd -s 2 3 1 -o 0.25 0.5 0 paint%20%23.png?version=2#color
map_Ks specular.png
map_Ke glow.png
norm normal.png
map_bump -bm 0.4 bump.png
disp -mm -0.2 1.5 height.png
map_d alpha.png
`;
  const obj = `mtllib nested/library.mtl
o triangle
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
usemtl Paint
f 1/1/1 2/2/1 3/3/1
`;
  const imageNames = ['paint #.png', 'specular.png', 'glow.png', 'normal.png', 'bump.png', 'height.png', 'alpha.png'];
  const images = Object.fromEntries(['models/', 'shared/'].flatMap(base => imageNames.map(name => [base + name, pixel])));
  const loaderUrl = name => new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href;
  const f = fixture({'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import * as THREE from ${JSON.stringify(core.href)}; export {THREE};
      import {MTLLoader} from ${JSON.stringify(loaderUrl('MTLLoader'))};
      import {OBJLoader} from ${JSON.stringify(loaderUrl('OBJLoader'))};
      const mapNames=['map','specularMap','emissiveMap','normalMap','bumpMap','displacementMap','alphaMap'];
      async function ready(material){
        for(let attempt=0;attempt<100;attempt++){
          if(mapNames.every(name=>material[name]?.image?.complete))return;
          await new Promise(resolve=>setTimeout(resolve,10));
        }
        throw new Error('Unloaded MTL textures: '+mapNames.filter(name=>!material[name]?.image?.complete));
      }
      export async function run(){
        const order=[];
        const pending=new MTLLoader().setPath('models/').setMaterialOptions({side:THREE.DoubleSide})
          .loadAsync('nested/library.mtl'); order.push('returned');
        const materials=await pending; order.push('loaded'); materials.preload();
        const objects=new OBJLoader(); objects.setMaterials(materials); objects.setPath('models/');
        const first=await objects.loadAsync('triangle.obj');
        const second=await objects.loadAsync('triangle.obj');
        const independent=await new Promise((resolve,reject)=>new MTLLoader().setPath('models/')
          .load('nested/library.mtl',resolve,undefined,reject));
        const independentMesh=await new OBJLoader().setMaterials(independent).setPath('models/').loadAsync('triangle.obj');
        const changing=new MTLLoader(), lateLoad=changing.loadAsync('late/library.mtl');
        changing.setResourcePath('shared/'); const late=await lateLoad; late.preload();
        const a=first.children[0], b=second.children[0], c=independentMesh.children[0], material=a.material;
        await Promise.all([ready(material),ready(c.material),ready(late.create('Paint'))]);
        a.position.x=7;
        return {order,name:a.name,type:material.type,vertices:[...a.geometry.attributes.position.array],
          uv:[...a.geometry.attributes.uv.array],independentObjects:a!==b&&a.geometry!==b.geometry&&b.position.x===0,
          sharedCreator:material===b.material,independentCreator:material!==c.material&&material.map!==c.material.map,
          mapNames:mapNames.filter(name=>material[name].image.complete),repeat:material.map.repeat.toArray(),
          offset:material.map.offset.toArray(),side:material.side,opacity:material.opacity,transparent:material.transparent,
          shininess:material.shininess,bumpScale:material.bumpScale,displacementBias:material.displacementBias,
          displacementScale:material.displacementScale,colorSpaces:mapNames.map(name=>material[name].colorSpace),
          lateBase:late.baseUrl,lateDistinct:late.create('Paint').map.image.src!==material.map.image.src};
      }`,
    'models/nested/library.mtl': mtl, 'late/library.mtl': mtl, 'models/triangle.obj': obj, ...images,
  });
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders:true});
  assert.equal(fs.readFileSync(path.join(f.out,'models/nested/library.mtl'),'utf8'),mtl);
  assert.equal(fs.readFileSync(path.join(f.out,'models/triangle.obj'),'utf8'),obj);
  for (const [name, bytes] of Object.entries(images)) assert.deepEqual(fs.readFileSync(path.join(f.out,name)),bytes);
  packHtml(path.join(f.out,'index.html'),f.packed);
  const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  assert.deepEqual(result.result,{order:['returned','loaded'],name:'triangle',type:'MeshPhongMaterial',
    vertices:[0,0,0,1,0,0,0,1,0],uv:[0,0,1,0,0,1],independentObjects:true,sharedCreator:true,independentCreator:true,
    mapNames:['map','specularMap','emissiveMap','normalMap','bumpMap','displacementMap','alphaMap'],repeat:[2,3],
    offset:[0.25,0.5],side:2,opacity:0.6,transparent:true,shininess:24,bumpScale:0.4,displacementBias:-0.2,
    displacementScale:1.5,colorSpaces:['srgb','','srgb','','','',''],lateBase:'shared/',lateDistinct:true});
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='models/paint%20%23.png?version=2#color'));
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='shared/normal.png'));
  assert.ok(result.requests.every(({url})=>url.startsWith('data:')));
  assert.ok(result.images.every(url=>url.startsWith('data:image/png;')));
});

test("packed LDraw car retains its parts, construction steps and conditional edges without files or network", async () => {
  const model=fs.readFileSync(new URL('../../upstream/three.js/examples/models/ldraw/officialLibrary/models/car.ldr_Packed.mpd',import.meta.url));
  const loaderUrl=new URL('../../upstream/three.js/examples/jsm/loaders/LDrawLoader.js',import.meta.url).href;
  const materialUrl=new URL('../../upstream/three.js/examples/jsm/materials/LDrawConditionalLineMaterial.js',import.meta.url).href;
  const f=fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import * as THREE from ${JSON.stringify(core.href)};export {THREE};
      import {LDrawLoader} from ${JSON.stringify(loaderUrl)};
      import {LDrawConditionalLineMaterial} from ${JSON.stringify(materialUrl)};
      function inspect(group){
        const result={meshes:0,vertices:0,lines:0,conditional:0,steps:group.userData.numBuildingSteps};
        group.traverse(object=>{
          if(object.isMesh){result.meshes++;result.vertices+=object.geometry.attributes.position.count;}
          if(object.isLineSegments)result.lines++;
          if(object.isConditionalLine)result.conditional++;
        });return result;
      }
      export async function run(){
        const order=[],loader=new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial).setPath('models/');
        let returned;
        const first=await new Promise((resolve,reject)=>{
          returned=loader.load('car%20%23.mpd?version=1#model',group=>{order.push('loaded');resolve(group);},undefined,reject);
          order.push('returned');
        });
        const second=await new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial)
          .setPath('models/').loadAsync('car%20%23.mpd?version=2#model');
        let a,b;first.traverse(object=>{if(!a&&object.isMesh)a=object;});
        second.traverse(object=>{if(!b&&object.isMesh)b=object;});
        const before=b.geometry.attributes.position.getX(0);a.geometry.attributes.position.setX(0,before+100);
        a.position.x=123;
        return {order,undefinedReturn:returned===undefined,first:inspect(first),second:inspect(second),
          files:[first.userData.fileName,second.userData.fileName],
          independent:first!==second&&a!==b&&a.geometry!==b.geometry&&a.material!==b.material&&
            b.geometry.attributes.position.getX(0)===before&&b.position.x===0};
      }`, 'models/car #.mpd':model});
  await buildApplication(f.entry,f.out,{prepareStandaloneLoaders:true});
  assert.deepEqual(fs.readFileSync(path.join(f.out,'models/car #.mpd')),model);
  packHtml(path.join(f.out,'index.html'),f.packed);const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  const expected={meshes:62,vertices:77031,lines:124,conditional:62,steps:8};
  assert.deepEqual(result.result,{order:['returned','loaded'],undefinedReturn:true,first:expected,second:expected,
    files:['car%20%23.mpd?version=1#model','car%20%23.mpd?version=2#model'],independent:true});
  assert.equal(result.requests.length,2);
  assert.ok(result.requests.every(({url})=>url.startsWith('data:')));
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='models/car%20%23.mpd?version=2#model'));
  assert.ok(result.events.every(([,url])=>!url.startsWith('data:')));
});

test("LDraw palettes, live smoothing, runtime parsing and unavailable library errors retain source behavior", async () => {
  const palette='0 !COLOUR Custom_Red CODE 1000 VALUE #FF0000 EDGE #123456\n'+
    '0 !COLOUR Custom_Blue CODE 1001 VALUE #0000FF EDGE #654321 ALPHA 128';
  const model=`0 FILE scene.ldr
0 !LDRAW_ORG Model
0 !CATEGORY Vehicles
0 !KEYWORDS miniature, two steps
1 1000 0 0 0 1 0 0 0 1 0 0 0 1 Alias.dat
0 STEP
1 16 5 0 0 1 0 0 0 1 0 0 0 1 submodel.ldr
0 FILE Part.dat
0 !LDRAW_ORG Part
0 BFC CERTIFY CCW
3 16 0 0 0 1 0 0 0 1 0
3 16 1 0 0 0 0 0 0 -1 0.5
2 24 0 0 0 0 1 0
5 24 0 0 0 1 0 0 0 1 0 0 -1 0.5
0 FILE submodel.ldr
0 !LDRAW_ORG Model
1 1001 0 0 0 1 0 0 0 1 0 0 0 1 PART.DAT
`;
  const loaderUrl=new URL('../../upstream/three.js/examples/jsm/loaders/LDrawLoader.js',import.meta.url).href;
  const materialUrl=new URL('../../upstream/three.js/examples/jsm/materials/LDrawConditionalLineMaterial.js',import.meta.url).href;
  const resources={'palettes/colors #.ldr':palette,'models/scene.mpd':model,'models/broken.ldr':'6 invalid',
    'models/missing-parts.ldr':'1 16 0 0 0 1 0 0 0 1 0 0 0 1 Missing.DAT'};
  const f=fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import * as THREE from ${JSON.stringify(core.href)};export {THREE};
      import {LDrawLoader} from ${JSON.stringify(loaderUrl)};
      import {LDrawConditionalLineMaterial} from ${JSON.stringify(materialUrl)};
      function objects(group){const list=[];group.traverse(object=>{if(object.isMesh||object.isLineSegments)list.push(object);});return list;}
      const round=values=>values.map(value=>Number(value.toFixed(6)));
      export async function run(){
        const loader=new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial)
          .setPartsLibraryPath('library/').setFileMap({'Alias.dat':'Part.dat'}).setPath('palettes/');
        const preloaded=await loader.preloadMaterials('colors%20%23.ldr?version=1#palette');
        loader.setPath('models/');const pending=loader.loadAsync('scene.mpd');loader.smoothNormals=false;
        const flat=await pending;
        const smoothLoader=new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial)
          .setFileMap({'Alias.dat':'Part.dat'}).setPath('palettes/');
        await smoothLoader.preloadMaterials('colors%20%23.ldr?version=2#palette');
        smoothLoader.setPath('models/');const smooth=await smoothLoader.loadAsync('scene.mpd');
        const raw=await new THREE.FileLoader().loadAsync('models/scene.mpd');
        let parseReturn;const parseOrder=[];
        const parsed=await new Promise((resolve,reject)=>{
          parseReturn=loader.parse(raw,group=>{parseOrder.push('parsed');resolve(group);},reject);parseOrder.push('returned');
        });
        const failures=[];
        await new Promise(resolve=>new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial)
          .setPath('models/').load('broken.ldr',()=>{throw new Error('Invalid model accepted');},undefined,error=>{failures.push(error.message);resolve();}));
        await new Promise(resolve=>loader.parse('6 invalid',()=>{throw new Error('Invalid model accepted');},error=>{failures.push(error.message);resolve();}));
        try{await new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial).setPartsLibraryPath('library/')
          .setPath('models/').loadAsync('missing-parts.ldr');}catch(error){failures.push(error.message);}
        return {preloadUndefined:preloaded===undefined,parseUndefined:parseReturn===undefined,parseOrder,
          parsed:{objects:objects(parsed).length,file:parsed.userData.fileName},
          metadata:{steps:flat.userData.numBuildingSteps,category:flat.userData.category,keywords:flat.userData.keywords,
            children:flat.children.map(child=>({name:child.name,step:child.userData.buildingStep,position:child.position.toArray()}))},
          objects:objects(flat).map(object=>({type:object.type,conditional:!!object.isConditionalLine,name:object.material.name,
            color:object.material.color.getHexString(),opacity:object.material.opacity,vertices:object.geometry.attributes.position.count})),
          normals:[flat,smooth].map(group=>round([...objects(group)[0].geometry.attributes.normal.array.slice(0,3)])),failures};
      }`,...resources});
  await buildApplication(f.entry,f.out,{prepareStandaloneLoaders:true});
  for(const [name,bytes] of Object.entries(resources))assert.equal(fs.readFileSync(path.join(f.out,name),'utf8'),bytes);
  packHtml(path.join(f.out,'index.html'),f.packed);const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  const objects=[['Custom_Red','ff0000','123456',1],['Custom_Blue','0000ff','654321',128/255]].flatMap(([name,color,edge,opacity])=>[
    {type:'Mesh',conditional:false,name:' '+name,color,opacity,vertices:6},
    {type:'LineSegments',conditional:false,name:' '+name+' - Edge',color:edge,opacity,vertices:2},
    {type:'LineSegments',conditional:true,name:' '+name+' - Conditional Edge',color:edge,opacity,vertices:2},
  ]);
  assert.deepEqual(result.result,{preloadUndefined:true,parseUndefined:true,parseOrder:['returned','parsed'],
    parsed:{objects:6,file:''},metadata:{steps:2,category:'Vehicles',keywords:['miniature','two steps'],
      children:[{name:'Part.dat',step:0,position:[0,0,0]},{name:'submodel.ldr',step:1,position:[5,0,0]}]},
    objects,normals:[[0,0,1],[0,0.229753,0.973249]],failures:[
      'THREE.LDrawLoader: Unknown line type "6" at line 1.',
      'THREE.LDrawLoader: Unknown line type "6" at line 1.',
      'THREE.LDrawLoader: Subobject "missing.dat" could not be loaded.',
    ]});
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='palettes/colors%20%23.ldr?version=2#palette'));
  // These unresolved part paths retain the real source retries and failures;
  // packaging supplies neither substitute geometry nor synthetic responses.
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='library/parts/Missing.DAT'));
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='library/parts/missing.dat'));
  assert.ok(result.requests.every(({url})=>url.startsWith('data:')));
});

test("packed binary and ASCII FBX retain textured geometry, skeletal clips and animation without files or network", async () => {
  const sourceRoot=new URL('../../upstream/three.js/examples/models/fbx/',import.meta.url);
  const models=Object.fromEntries(['monkey.fbx','RotationTest.fbx','ball_anims_asc_2018.fbx']
    .map(name=>['models/'+name,fs.readFileSync(new URL(name,sourceRoot))]));
  const image=fs.readFileSync(new URL('monkey.fbm/UVTexture.png',sourceRoot));
  const f=fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import * as THREE from ${JSON.stringify(core.href)};export {THREE};
      import {FBXLoader} from ${JSON.stringify(new URL('../../upstream/three.js/examples/jsm/loaders/FBXLoader.js',import.meta.url).href)};
      async function ready(material){
        for(let i=0;i<100;i++){
          if(material.map?.image?.complete)return;
          await new Promise(resolve=>setTimeout(resolve,10));
        }
        throw new Error('FBX diffuse texture did not finish loading');
      }
      const round=values=>values.map(value=>Number(value.toFixed(9)));
      export async function run(){
        const order=[], pending=new FBXLoader().loadAsync('models/monkey.fbx');order.push('returned');
        const first=await pending;order.push('loaded');
        const second=await new Promise((resolve,reject)=>new FBXLoader().setPath('models/')
          .load('monkey.fbx',resolve,undefined,reject));
        const rotation=await new FBXLoader().setPath('models/').loadAsync('RotationTest.fbx');
        const skeletal=new FBXLoader();skeletal.trimAnimationClips=true;
        skeletal.setPath('models/');const ball=await skeletal.loadAsync('ball_anims_asc_2018.fbx');
        const changing=new FBXLoader().setPath('models/'), lateLoad=changing.loadAsync('monkey.fbx');
        changing.setResourcePath('shared/');const late=await lateLoad;
        const a=first.children[0],b=second.children[0],c=late.children[0],skin=ball.children[0];
        await Promise.all([ready(a.material),ready(b.material),ready(c.material)]);
        const rotate=new THREE.AnimationMixer(rotation);rotate.clipAction(rotation.animations[0]).play();rotate.update(0.5);
        const animate=new THREE.AnimationMixer(ball);animate.clipAction(ball.animations[0]).play();animate.update(0.2);
        ball.updateMatrixWorld(true);a.position.x=9;
        return {order,monkey:{name:a.name,vertices:a.geometry.attributes.position.count,material:a.material.type,
          independent:a!==b&&a.geometry!==b.geometry&&a.material!==b.material&&a.material.map!==b.material.map&&b.position.x===0,
          lateResource:c.material.map.image.src!==a.material.map.image.src},
          rotation:{vertices:rotation.children[0].geometry.attributes.position.count,clips:rotation.animations.length,
            duration:rotation.animations[0].duration,tracks:rotation.animations[0].tracks.length,
            quaternion:round(rotation.children[0].quaternion.toArray())},
          skeletal:{skinned:skin.isSkinnedMesh,vertices:skin.geometry.attributes.position.count,
            bones:skin.skeleton.bones.length,inverses:skin.skeleton.boneInverses.length,
            clips:ball.animations.map(clip=>({name:clip.name,duration:clip.duration,start:clip.tracks[0].times[0]})),
            animatedBone:round(skin.skeleton.bones[0].position.toArray())}};
      }`,...models,'models/monkey.fbm/UVTexture.png':image,'shared/monkey.fbm/UVTexture.png':image});
  await buildApplication(f.entry,f.out,{prepareStandaloneLoaders:true});
  for(const [name,bytes] of Object.entries(models))assert.deepEqual(fs.readFileSync(path.join(f.out,name)),bytes);
  for(const base of ['models/','shared/'])assert.deepEqual(fs.readFileSync(path.join(f.out,base+'monkey.fbm/UVTexture.png')),image);
  packHtml(path.join(f.out,'index.html'),f.packed);
  const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  assert.deepEqual(result.result,{order:['returned','loaded'],
    monkey:{name:'Suzanne',vertices:2910,material:'MeshPhongMaterial',independent:true,lateResource:true},
    rotation:{vertices:36,clips:1,duration:4.166666507720947,tracks:3,
      quaternion:[-0.705729324,0.044115921,0.044115723,0.705729205]},
    skeletal:{skinned:true,vertices:2280,bones:1,inverses:1,
      clips:[{name:'clip_vertical',duration:0.375,start:0},{name:'clip_horizontal',duration:0.3749999701976776,start:0}],
      animatedBone:[0,1.326013138,0]}});
  assert.equal(result.images.length,3);
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='models/monkey.fbm/UVTexture.png'));
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='shared/monkey.fbm/UVTexture.png'));
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

test('packed volume and voxel assets retain raw, gzip and ASCII samples, scenes and pending options offline', async () => {
  const loaderUrl = name => new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href;
  const values = [0, 1, 256, 65535, 2, 3, 1024, 4096], samples = Buffer.alloc(values.length * 2);
  values.forEach((value, index) => samples.writeUInt16LE(value, index * 2));
  const header = encoding => Buffer.from(`NRRD0005\ntype: ushort\ndimension: 3\nsizes: 2 2 2\nencoding: ${encoding}\nendian: little\nspace: left-posterior-superior\nspace directions: (1,0,0) (0,2,0) (0,0,3)\n# retained volume fixture\n\n`);
  const resources = {
    'volumes/raw #.nrrd': Buffer.concat([header('raw'), samples]),
    'volumes/gzip.nrrd': Buffer.concat([header('gzip'), gzipSync(samples)]),
    'volumes/ascii.nrrd': Buffer.concat([header('ascii'), Buffer.from(values.join(' ') + '\n')]),
    'volumes/bad.nrrd': Buffer.concat([header('bzip2'), samples]),
    'voxels/monument.vox': fs.readFileSync(new URL('../../upstream/three.js/examples/models/vox/monu10.vox', import.meta.url)),
    'voxels/bad.vox': Buffer.from('VOX '),
  };
  const fingerprint = view => {
    let hash = 2166136261;
    for (const byte of new Uint8Array(view.buffer, view.byteOffset, view.byteLength)) hash = Math.imul(hash ^ byte, 16777619);
    return [view.constructor.name, view.byteLength, hash >>> 0];
  };
  const describeVoxel = value => {
    const nodes = [];
    value.scene?.traverse(object => nodes.push({type: object.type, name: object.name, matrix: [...object.matrix.elements],
      attributes: object.isMesh ? Object.fromEntries(Object.entries(object.geometry.attributes).map(([name, attribute]) => [name, fingerprint(attribute.array)])) : null}));
    return {chunks: value.chunks.map(chunk => ({size: chunk.size, data: fingerprint(chunk.data), palette: chunk.palette})), nodes};
  };
  const describeVolume = volume => ({values: [...volume.data], type: volume.data.constructor.name,
    dimensions: volume.dimensions, spacing: volume.spacing, matrix: [...volume.matrix.elements],
    range: [volume.windowLow, volume.windowHigh], segmentation: volume.segmentation ?? null});
  const {VOXLoader} = await import(loaderUrl('VOXLoader'));
  const {NRRDLoader} = await import(loaderUrl('NRRDLoader'));
  const sourceBytes = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const sourceVoxel = describeVoxel(new VOXLoader().parse(sourceBytes(resources['voxels/monument.vox'])));
  const sourceVolume = describeVolume(new NRRDLoader().parse(sourceBytes(resources['volumes/raw #.nrrd'])));
  const failures = [];
  for (const [loader, bytes] of [[new NRRDLoader(), resources['volumes/bad.nrrd']], [new VOXLoader(), resources['voxels/bad.vox']]]) {
    try { loader.parse(sourceBytes(bytes)); } catch(error) { failures.push([error.name, error.message]); }
  }
  assert.equal(failures.length, 2);
  const f = fixture({'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import {NRRDLoader} from ${JSON.stringify(loaderUrl('NRRDLoader'))};
      import {VOXLoader,buildData3DTexture} from ${JSON.stringify(loaderUrl('VOXLoader'))};
      const fingerprint=${fingerprint.toString()}, describeVoxel=${describeVoxel.toString()}, describeVolume=${describeVolume.toString()};
      export async function run(){
        const order=[],loader=new NRRDLoader().setPath('volumes/');loader.setSegmentation(false);
        const pending=loader.loadAsync('raw%20%23.nrrd?edition=2#scan').then(value=>{order.push('loaded');return value;});
        order.push('returned');loader.segmentation=true;const raw=await pending;
        const gzip=await new NRRDLoader().loadAsync('volumes/gzip.nrrd');
        const ascii=await new NRRDLoader().loadAsync('volumes/ascii.nrrd');
        loader.setSegmentation(false);const again=await loader.loadAsync('raw%20%23.nrrd?edition=2#scan');
        const voxLoader=new VOXLoader().setPath('voxels/'),vox=await voxLoader.loadAsync('monument.vox');
        const copy=await voxLoader.loadAsync('monument.vox'),texture=buildData3DTexture(vox.chunks[0]);
        const volumes=[raw,gzip,ascii,again].map(describeVolume),voxel=describeVoxel(vox);
        raw.data[0]=42;vox.chunks[0].data[0]^=1;
        const failures=[];
        try{await new NRRDLoader().loadAsync('volumes/bad.nrrd');}catch(error){failures.push([error.name,error.message]);}
        try{await new VOXLoader().loadAsync('voxels/bad.vox');}catch(error){failures.push([error.name,error.message]);}
        return {order,volumes,voxel,failures,independent:raw!==again&&raw.data!==again.data&&again.data[0]===0&&
          vox.scene!==copy.scene&&vox.chunks[0].data!==copy.chunks[0].data&&vox.chunks[0].data[0]!==copy.chunks[0].data[0],
          texture:{dimensions:[texture.image.width,texture.image.height,texture.image.depth],data:fingerprint(texture.image.data)}};
      }`, ...resources});
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(resources)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  const {buildData3DTexture} = await import(loaderUrl('VOXLoader'));
  const texture = buildData3DTexture(new VOXLoader().parse(sourceBytes(resources['voxels/monument.vox'])).chunks[0]);
  assert.deepEqual(result.result, {order: ['returned', 'loaded'],
    volumes: [true, null, null, false].map(segmentation => ({...sourceVolume, segmentation})),
    voxel: sourceVoxel, failures, independent: true,
    texture: {dimensions: [texture.image.width, texture.image.height, texture.image.depth], data: fingerprint(texture.image.data)}});
  assert.deepEqual(sourceVolume.values, values);
  assert.ok(sourceVoxel.nodes.some(node => node.attributes?.position));
  assert.equal(result.requests.length, 8);
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
});

test('packed color-grading LUTs retain float and byte textures, domains, callbacks and parser errors offline', async () => {
  const loaderUrl = name => new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href;
  const colors = ['0 0 0', '1 0 0', '0 1 0', '1 1 0', '0 0 1', '1 0 1', '0 1 1', '1 1 1'];
  const cube = 'TITLE "Retained grade"\nLUT_3D_SIZE 2\nDOMAIN_MIN 0.1 0.2 0.3\nDOMAIN_MAX 0.8 0.9 1\n' + colors.join('\n') + '\n';
  const threeDl = '0 1023\n' + colors.map(line => line.split(' ').map(Number).map(value => value * 1024).join(' ')).join('\n') + '\n';
  const resources = {'luts/grade #.CUBE': Buffer.from(cube), 'luts/grade.3dl': Buffer.from(threeDl),
    'luts/bad.CUBE': Buffer.from('TITLE "Missing size"\n'), 'luts/bad.3dl': Buffer.from('0 1 3\n')};
  const describe = value => ({size: value.size, title: value.title ?? null,
    domain: value.domainMin ? [value.domainMin.toArray(), value.domainMax.toArray()] : null,
    type: value.texture3D.type, storage: value.texture3D.image.data.constructor.name, data: [...value.texture3D.image.data],
    dimensions: [value.texture3D.image.width, value.texture3D.image.height, value.texture3D.image.depth],
    sampler: [value.texture3D.minFilter,value.texture3D.magFilter,value.texture3D.wrapS,value.texture3D.wrapT,value.texture3D.wrapR],
    mipmaps: value.texture3D.generateMipmaps, version: value.texture3D.version});
  const {LUTCubeLoader} = await import(loaderUrl('LUTCubeLoader'));
  const {LUT3dlLoader} = await import(loaderUrl('LUT3dlLoader'));
  const {FloatType, UnsignedByteType} = await import(core.href);
  const expected = [describe(new LUTCubeLoader().setType(FloatType).parse(cube)), describe(new LUTCubeLoader().setType(UnsignedByteType).parse(cube)),
    describe(new LUT3dlLoader().setType(FloatType).parse(threeDl)), describe(new LUT3dlLoader().setType(UnsignedByteType).parse(threeDl))];
  const failures = [];
  for (const [loader, input] of [[new LUTCubeLoader(), resources['luts/bad.CUBE']], [new LUT3dlLoader(), resources['luts/bad.3dl']]]) {
    try { loader.parse(input.toString()); } catch(error) { failures.push([error.name,error.message]); }
  }
  assert.equal(failures.length, 2);
  const f = fixture({'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import {FloatType,UnsignedByteType} from ${JSON.stringify(core.href)};
      import {LUTCubeLoader} from ${JSON.stringify(loaderUrl('LUTCubeLoader'))};
      import {LUT3dlLoader} from ${JSON.stringify(loaderUrl('LUT3dlLoader'))};
      const describe=${describe.toString()};
      export async function run(){
        const order=[],cube=new LUTCubeLoader().setType(UnsignedByteType).setPath('luts/');
        const pending=cube.loadAsync('grade%20%23.CUBE?edition=2#lut').then(value=>{order.push('loaded');return value;});
        order.push('returned');cube.type=FloatType;const floatCube=await pending;
        cube.setType(UnsignedByteType);const byteCube=await cube.loadAsync('grade%20%23.CUBE?edition=2#lut');
        const film=new LUT3dlLoader().setPath('luts/');film.type=UnsignedByteType;
        const filmPending=film.loadAsync('grade.3dl');film.setType(FloatType);const floatFilm=await filmPending;
        film.type=UnsignedByteType;const byteFilm=await film.loadAsync('grade.3dl');
        const textures=[floatCube,byteCube,floatFilm,byteFilm].map(describe),failures=[];
        floatCube.texture3D.image.data[0]=.75;
        try{await new LUTCubeLoader().loadAsync('luts/bad.CUBE');}catch(error){failures.push([error.name,error.message]);}
        await new Promise(resolve=>new LUT3dlLoader().load('luts/bad.3dl',()=>{throw new Error('Invalid LUT accepted');},undefined,error=>{failures.push([error.name,error.message]);resolve();}));
        return {order,textures,failures,independent:floatCube.texture3D!==byteCube.texture3D&&byteCube.texture3D.image.data[0]===0&&
          floatFilm.texture3D!==byteFilm.texture3D&&floatFilm.texture3D.image.data!==byteFilm.texture3D.image.data};
      }`, ...resources});
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(resources)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  assert.deepEqual(result.result, {order: ['returned', 'loaded'], textures: expected, failures, independent: true});
  assert.equal(result.requests.length, 6);
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
});

test('packed PCD, XYZ and VTK retain ASCII and binary geometry, options and parse errors offline', async () => {
  const header = (fields, sizes, types, format) => Buffer.from(`VERSION 0.7\nFIELDS ${fields}\nSIZE ${sizes}\nTYPE ${types}\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ${format}\n`);
  const pointHeader = format => header('x y z', '4 4 4', 'F F F', format);
  const binaryPoints = Buffer.alloc(24), compressedPoints = Buffer.alloc(24);
  [1.25, -2.5, 3, -7.5, 8.25, -9].forEach((value, i) => binaryPoints.writeFloatBE(value, i * 4));
  // PCD's compressed encoding is planar; one LZF literal run preserves the
  // real source decompression path without adding a test encoder dependency.
  [1.25, -4, 2.5, 5, -3, 6].forEach((value, i) => compressedPoints.writeFloatLE(value, i * 4));
  const compressedSizes = Buffer.alloc(8); compressedSizes.writeUInt32LE(25); compressedSizes.writeUInt32LE(24, 4);
  const badSizes = Buffer.alloc(8); badSizes.writeUInt32LE(1); badSizes.writeUInt32LE(24, 4);
  const triangle = [0, 0, 0, 1, 0, 0, 0, 1, 0], normals = [0, 0, 1, 0, 0, 1, 0, 0, 1];
  const floats = values => { const bytes = Buffer.alloc(values.length * 4); values.forEach((v, i) => bytes.writeFloatBE(v, i * 4)); return bytes; };
  const indices = Buffer.alloc(16); [3, 0, 1, 2].forEach((v, i) => indices.writeUInt32BE(v, i * 4));
  // The source parser reads a 250-byte prefix even for small models.
  const vtkHeader = format => `# vtk DataFile Version 3.0\n${'Retained source geometry fixture. '.repeat(4)}\n${format}\nDATASET POLYDATA\n`;
  const resources = {
    'points/cloud #.pcd': Buffer.concat([header('x y z rgb normal_x normal_y normal_z intensity label',
      '4 4 4 4 4 4 4 4 4', 'F F F U F F F F I', 'ascii'),
      Buffer.from('1 2 3 16711680 0 1 0 0.5 17\n-1 0 4 65280 1 0 0 0.25 -8\n')]),
    'points/big-endian.pcd': Buffer.concat([pointHeader('binary'), binaryPoints]),
    'points/compressed.pcd': Buffer.concat([pointHeader('binary_compressed'), compressedSizes, Buffer.from([23]), compressedPoints]),
    'points/broken.pcd': Buffer.concat([pointHeader('binary_compressed'), badSizes, Buffer.from([10])]),
    'points/cloud.xyz': Buffer.from('# colored points\n1 2 3 255 0 0\n-1 0 4 0 255 0\n'),
    'surfaces/ascii.vtk': Buffer.from(vtkHeader('ASCII') + 'POINTS 3 float\n0 0 0 1 0 0 0 1 0\nPOLYGONS 1 4\n3 0 1 2\nPOINT_DATA 3\nNORMALS n float\n0 0 1 0 0 1 0 0 1\n'),
    'surfaces/binary.vtk': Buffer.concat([Buffer.from(vtkHeader('BINARY') + 'POINTS 3 float\n'), floats(triangle),
      Buffer.from('\nPOLYGONS 1 4\n'), indices, Buffer.from('\nPOINT_DATA 3\nNORMALS n float\n'), floats(normals), Buffer.from('\n')]),
    'surfaces/broken.vtk': Buffer.from(vtkHeader('ASCII').replace('POLYDATA', 'UNSTRUCTURED_GRID') + '\n'.repeat(100)),
  };
  const names = ['PCDLoader', 'VTKLoader', 'XYZLoader'];
  const f = fixture({'index.html': '<script type="module" src="modules/app.mjs"></script>',
    'modules/app.mjs': `import * as THREE from ${JSON.stringify(core.href)}; export {THREE};
      ${names.map(name => `import {${name}} from ${JSON.stringify(new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href)};`).join('\n')}
      export async function run() {
        const order=[], loader=new PCDLoader(); loader.setPath('points/');
        const pending=loader.loadAsync('cloud%20%23.pcd?edition=2#points').then(value=>{order.push('loaded');return value;});
        order.push('returned'); const ascii=await pending;
        const binaryLoader=new PCDLoader(); binaryLoader.littleEndian=false;
        binaryLoader.setPath('points/').setRequestHeader({'X-Fixture':'points'}).setWithCredentials(true);
        const binary=await binaryLoader.loadAsync('big-endian.pcd');
        const compressed=await new PCDLoader().setPath('points/').loadAsync('compressed.pcd');
        const xyz=await new XYZLoader().setPath('points/').loadAsync('cloud.xyz');
        const vtkAscii=await new VTKLoader().setPath('surfaces/').loadAsync('ascii.vtk');
        const vtkBinary=await new VTKLoader().setPath('surfaces/').loadAsync('binary.vtk');
        const failures=[];
        try { await new PCDLoader().setPath('points/').loadAsync('broken.pcd'); } catch(error) { failures.push([error.name,error.message]); }
        await new Promise(resolve=>new VTKLoader().setPath('surfaces/').load('broken.vtk',()=>{throw new Error('Invalid VTK accepted');},undefined,error=>{failures.push([error.name,error.message]);resolve();}));
        const attributes=geometry=>Object.fromEntries(Object.entries(geometry.attributes).map(([name,attribute])=>[name,[...attribute.array]]));
        return {order,ascii:attributes(ascii.geometry),binary:attributes(binary.geometry),compressed:attributes(compressed.geometry),
          xyz:attributes(xyz),surfaces:[vtkAscii,vtkBinary].map(geometry=>({attributes:attributes(geometry),indices:[...geometry.index.array]})),failures};
      }`, ...resources});
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(resources)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  const pointColors = {position: [1, 2, 3, -1, 0, 4], color: [1, 0, 0, 0, 1, 0]};
  assert.deepEqual(result.result, {order: ['returned', 'loaded'],
    ascii: {...pointColors, normal: [0, 1, 0, 1, 0, 0], intensity: [.5, .25], label: [17, -8]},
    binary: {position: [1.25, -2.5, 3, -7.5, 8.25, -9]}, compressed: {position: [1.25, 2.5, -3, -4, 5, 6]},
    xyz: pointColors, surfaces: Array.from({length: 2}, () => ({attributes: {position: triangle, normal: normals}, indices: [0, 1, 2]})),
    failures: [['Error', 'THREE.PCDLoader: Invalid compressed data'], ['Error', 'Unsupported DATASET type: UNSTRUCTURED_GRID']]});
  assert.equal(result.requests.length, Object.keys(resources).length);
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
  assert.ok(result.events.some(([kind, url]) => kind === 'end' && url === 'points/cloud%20%23.pcd?edition=2#points'));
});

test('packed BVH, GCode and PDB preserve animation switches, pending options, toolpaths and bonds offline', async () => {
  const atom = (id, element, xyz) => {
    const line = Array(78).fill(' '), put = (offset, value) => [...value].forEach((c, i) => { line[offset + i] = c; });
    put(0, 'HETATM'); put(6, String(id).padStart(5)); put(76, element.padStart(2));
    xyz.forEach((value, i) => put(30 + 8 * i, value.toFixed(3).padStart(8)));
    return line.join('');
  };
  const resources = {
    'motion/walk.bvh': Buffer.from('HIERARCHY\nROOT Hips\n{\nOFFSET 0 1 0\nCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\nEnd Site\n{\nOFFSET 0 2 0\n}\n}\nMOTION\nFrames: 2\nFrame Time: 0.5\n0 0 0 0 0 0\n1 2 3 90 0 0\n'),
    'paths/print.gcode': Buffer.from('G90\nM82\nG0 X0 Y0 Z0\nG1 X1 Y0 Z0 E1\nG0 X1 Y1 Z1\nG1 X2 Y1 Z1 E2\n'),
    'molecules/water.pdb': Buffer.from([atom(1, 'H', [0, 0, 0]), atom(2, 'O', [1.25, -2.5, 3.75]), 'CONECT    1    2', 'CONECT    2    1', ''].join('\n')),
  };
  const names = ['BVHLoader', 'GCodeLoader', 'PDBLoader'];
  const f = fixture({'index.html': '<script type="module" src="modules/app.mjs"></script>',
    'modules/app.mjs': `${names.map(name => `import {${name}} from ${JSON.stringify(new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href)};`).join('\n')}
      export async function run() {
        const motion=new BVHLoader(); motion.setPath('motion/');
        const full=await motion.loadAsync('walk.bvh');
        motion.animateBonePositions=false;
        const rotations=await motion.loadAsync('walk.bvh');
        motion.animateBonePositions=true; motion.animateBoneRotations=false;
        const positions=await motion.loadAsync('walk.bvh');
        const machine=new GCodeLoader(); machine.setPath('paths/');
        const pending=machine.loadAsync('print.gcode'); machine.splitLayer=true;
        const split=await pending;
        machine.splitLayer=false; const merged=await machine.loadAsync('print.gcode');
        const molecule=await new PDBLoader().setPath('molecules/').loadAsync('water.pdb');
        const tracks=clip=>clip.tracks.map(track=>({name:track.name,times:[...track.times],values:[...track.values]}));
        const lines=group=>group.children.map(line=>({name:line.name,material:line.material.name,positions:[...line.geometry.attributes.position.array]}));
        return {bones:full.skeleton.bones.map(bone=>({name:bone.name,position:bone.position.toArray()})),duration:full.clip.duration,
          full:tracks(full.clip),rotations:tracks(rotations.clip),positions:tracks(positions.clip),split:lines(split),merged:lines(merged),
          rotation:split.rotation.x,atoms:[...molecule.geometryAtoms.attributes.position.array],
          bonds:[...molecule.geometryBonds.attributes.position.array],json:molecule.json};
      }`, ...resources});
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(resources)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error);
  const position = {name: 'Hips.position', times: [0, .5], values: [0, 1, 0, 1, 3, 3]};
  const rotation = {name: 'Hips.quaternion', times: [0, .5], values: [0, 0, 0, 1, 0, 0, Math.fround(Math.SQRT1_2), Math.fround(Math.SQRT1_2)]};
  assert.deepEqual(result.result, {bones: [{name: 'Hips', position: [0, 1, 0]}, {name: 'ENDSITE', position: [0, 2, 0]}], duration: .5,
    full: [position, rotation], rotations: [rotation], positions: [position],
    split: [{name: 'layer0', material: 'extruded', positions: [0, 0, 0, 1, 0, 0]},
      {name: 'layer0', material: 'path', positions: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]},
      {name: 'layer1', material: 'extruded', positions: [1, 1, 1, 2, 1, 1]}, {name: 'layer1', material: 'path', positions: []}],
    merged: [{name: 'layer2', material: 'extruded', positions: [0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 1]},
      {name: 'layer2', material: 'path', positions: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]}], rotation: -Math.PI / 2,
    atoms: [0, 0, 0, 1.25, -2.5, 3.75], bonds: [0, 0, 0, 1.25, -2.5, 3.75],
    json: {atoms: [[0, 0, 0, [255, 255, 255], 'H'], [1.25, -2.5, 3.75, [255, 13, 13], 'O']]}});
  assert.equal(result.requests.length, 6, 'repeated loads still execute each retained parser with its current options');
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
});

test('packed ASCII and binary FBX retain textured geometry, embedded images and pending resource bases offline',async()=>{
  const fbxUrl=new URL('../../upstream/three.js/examples/jsm/loaders/FBXLoader.js',import.meta.url).href;
  const models={
    'models/nested/ascii.fbx':fbxResources([{relative:'C:\\exports\\paint%20%23.png?version=2#map'}],{triangle:true}),
    'models/binary.fbx':fbxResources([{relative:'paint.png'}],{binary:true,triangle:true}),
    'models/embedded.fbx':fbxResources([{relative:'embedded.png',content:fbxPixel}],{binary:true,version:7500,triangle:true}),
    'late/scene.fbx':fbxResources([{relative:'paint.png'}],{triangle:true}),
  };
  const f=fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import * as THREE from ${JSON.stringify(core.href)};export {THREE};
      import {FBXLoader} from ${JSON.stringify(fbxUrl)};
      async function ready(scene){const mesh=scene.children[0];for(let i=0;i<100;i++){
        if(mesh.material.map.image?.complete)return mesh;await new Promise(resolve=>setTimeout(resolve,10));
      }throw new Error('FBX image failed to load');}
      export async function run(){
        const order=[],loader=new FBXLoader().setPath('models/');
        const pending=loader.loadAsync('nested/ascii.fbx');order.push('returned');
        const first=await ready(await pending);order.push('loaded');
        const second=await ready(await loader.loadAsync('nested/ascii.fbx'));
        const binary=await ready(await new FBXLoader().setPath('models/').setResourcePath('shared/').loadAsync('binary.fbx'));
        const embedded=await ready(await new FBXLoader().setPath('models/').loadAsync('embedded.fbx'));
        const changing=new FBXLoader(),latePending=changing.loadAsync('late/scene.fbx');changing.setResourcePath('shared/');
        const late=await ready(await latePending);first.position.x=7;
        return {order,meshes:[first,binary,embedded,late].map(mesh=>({name:mesh.name,type:mesh.type,
          material:mesh.material.type,positions:[...mesh.geometry.attributes.position.array],
          uv:[...mesh.geometry.attributes.uv.array],colorSpace:mesh.material.map.colorSpace,complete:mesh.material.map.image.complete})),
          independent:first!==second&&first.geometry!==second.geometry&&first.material!==second.material&&second.position.x===0,
          embedded:embedded.material.map.image.src.startsWith('blob:')};
      }`,...models,'models/paint #.png':fbxPixel,'shared/paint.png':fbxPixel});
  await buildApplication(f.entry,f.out,{prepareStandaloneLoaders:true});
  for(const [name,bytes]of Object.entries(models))assert.deepEqual(fs.readFileSync(path.join(f.out,name)),bytes);
  packHtml(path.join(f.out,'index.html'),f.packed);const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  assert.deepEqual(result.result,{order:['returned','loaded'],meshes:Array.from({length:4},()=>({
    name:'Triangle',type:'Mesh',material:'MeshPhongMaterial',positions:[0,0,0,1,0,0,0,1,0],
    uv:[0,0,1,0,0,1],colorSpace:'srgb',complete:true})),independent:true,embedded:true});
  assert.ok(result.requests.every(({url})=>url.startsWith('data:')));
  assert.equal(result.images.length,5);
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='models/paint%20%23.png?version=2#map'));
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='shared/paint.png'));
});

function usdArchive(texturePath = 'textures/paint.png', {image = true, missingLayer = false, directory = 'nested/'} = {}) {
  const root = `#usda 1.0
(
  defaultPrim = "Triangle"
  upAxis = "Y"
  metersPerUnit = 1
)
def Xform "Triangle" (
  prepend references = @./geometry.usda@</Geometry>
)
{
  double3 xformOp:translate = (1, 2, 3)
  uniform token[] xformOpOrder = ["xformOp:translate"]
  rel material:binding = </Materials/Paint>
}
${missingLayer ? 'def Xform "Missing" (\n  prepend references = @https://outside.example/missing.usda@</Geometry>\n)\n{\n}\n' : ''}
def Scope "Materials" {
  def Material "Paint" {
    token outputs:surface.connect = </Materials/Paint/Surface.outputs:surface>
    def Shader "Surface" {
      uniform token info:id = "UsdPreviewSurface"
      color3f inputs:diffuseColor.connect = </Materials/Paint/Image.outputs:rgb>
      float inputs:roughness = 0.25
      float inputs:metallic = 0.5
      token outputs:surface
    }
    def Shader "Image" {
      uniform token info:id = "UsdUVTexture"
      asset inputs:file = @${texturePath}@
      token inputs:sourceColorSpace = "sRGB"
      token inputs:wrapS = "clamp"
      token inputs:wrapT = "mirror"
      float2 inputs:st.connect = </Materials/Paint/Transform.outputs:result>
      float3 outputs:rgb
    }
    def Shader "Transform" {
      uniform token info:id = "UsdTransform2d"
      float2 inputs:scale = (2, 3)
      float2 inputs:translation = (0.25, 0.5)
      float inputs:rotation = 30
      float2 outputs:result
    }
  }
}`;
  const geometry = `#usda 1.0
def "Geometry" {
  def Mesh "Geometry" {
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)] (
      interpolation = "vertex"
    )
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
      interpolation = "vertex"
    )
    uniform token subdivisionScheme = "none"
  }
}`;
  return Buffer.from(zipSync({[directory+'root.usda']: strToU8(root), [directory+'geometry.usda']: strToU8(geometry),
    ...(image ? {[directory+'textures/paint.png']: pixel} : {})}, {level: 0}));
}

test("packed USD archives retain layers, embedded textures, callbacks and edited USDZ export round trips offline", async () => {
  const archive = usdArchive(), malformed = Buffer.from(zipSync({'readme.txt': strToU8('not a USD root')}));
  const realArchive = fs.readFileSync(new URL('../../upstream/three.js/examples/models/usdz/saeukkang.usdz', import.meta.url));
  const usd = new URL('../../upstream/three.js/examples/jsm/loaders/USDLoader.js', import.meta.url).href;
  const legacy = new URL('../../upstream/three.js/examples/jsm/loaders/USDZLoader.js', import.meta.url).href;
  const exporter = new URL('../../upstream/three.js/examples/jsm/exporters/USDZExporter.js', import.meta.url).href;
  const f = fixture({'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import * as THREE from ${JSON.stringify(core.href)}; export {THREE};
      import {USDLoader} from ${JSON.stringify(usd)};
      import {USDZLoader} from ${JSON.stringify(legacy)};
      import {USDZExporter} from ${JSON.stringify(exporter)};
      const meshes=group=>{const result=[];group.traverse(object=>{if(object.isMesh)result.push(object);});return result;};
      export async function run(){
        const order=[], loader=new USDLoader().setPath('models/'); let returned;
        const first=await new Promise((resolve,reject)=>{
          returned=loader.load('scene%20%23.usdz?edition=2#root',group=>{
            order.push('loaded');resolve(group);
          },undefined,reject);order.push('returned');
        });
        const second=await loader.loadAsync('scene%20%23.usdz?edition=2#root');
        const legacy=await new USDZLoader().setPath('models/').loadAsync('scene%20%23.usdz?edition=2#root');
        const real=meshes(await new USDLoader().setPath('models/').loadAsync('real.usdz'));
        const raw=await new THREE.FileLoader().setResponseType('arraybuffer').loadAsync('models/scene%20%23.usdz?edition=2#root');
        let immediate,callbackIdentity=false,readyAtCallback=false;
        const parsed=await new Promise((resolve,reject)=>{
          immediate=loader.parse(raw,'ignored/external/',group=>{
            callbackIdentity=group===immediate;readyAtCallback=meshes(group)[0].material.map.image.complete;resolve(group);
          },reject);
        });
        const a=meshes(first)[0],b=meshes(second)[0],texture=a.material.map;
        const before={name:a.name,positions:[...a.geometry.attributes.position.array],uv:[...a.geometry.attributes.uv.array],
          position:a.position.toArray(),material:a.material.type,roughness:a.material.roughness,metalness:a.material.metalness,
          image:texture.image.complete,blob:texture.image.src.startsWith('blob:'),colorSpace:texture.colorSpace,
          wrap:[texture.wrapS,texture.wrapT],repeat:texture.repeat.toArray(),offset:texture.offset.toArray(),rotation:texture.rotation};
        const independent=first!==second&&a.geometry!==b.geometry&&a.material!==b.material&&texture!==b.material.map;
        a.name='Edited';a.position.set(7,8,9);a.updateMatrix();
        a.geometry.attributes.position.setXYZ(0,4,5,6);a.geometry.attributes.position.needsUpdate=true;
        a.material.map=null;a.material.color.setRGB(0.25,0.5,0.75);a.material.roughness=0.625;
        const exported=await new USDZExporter().parseAsync(first);
        const restored=meshes(await new Promise((resolve,reject)=>loader.parse(exported,'',resolve,reject)))[0];
        const failures=[];
        try{await new USDLoader().setPath('models/').loadAsync('invalid.usdz');}catch(error){failures.push(error.message);}
        const bad=await new THREE.FileLoader().setResponseType('arraybuffer').loadAsync('models/invalid.usdz');
        try{loader.parse(bad);}catch(error){failures.push(error.message);}
        return {order,undefinedReturn:returned===undefined,before,independent,callbackIdentity,readyAtCallback,
          parsedMeshes:meshes(parsed).length,legacyMeshes:meshes(legacy).length,
          real:real.map(mesh=>({name:mesh.name,vertices:mesh.geometry.attributes.position.count,image:mesh.material.map.image.complete})),
          untouched:{name:b.name,x:b.position.x,first:b.geometry.attributes.position.getX(0),map:!!b.material.map},
          restored:{name:restored.name,position:restored.position.toArray(),first:[...restored.geometry.attributes.position.array.slice(0,3)],
            color:restored.material.color.toArray(),roughness:restored.material.roughness,map:restored.material.map},
          zip:[...exported.slice(0,2)],failures};
      }`,
    'models/scene #.usdz': archive, 'models/invalid.usdz': malformed, 'models/real.usdz': realArchive,
  });
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for(const [name,bytes] of [['scene #.usdz',archive],['invalid.usdz',malformed],['real.usdz',realArchive]])
    assert.deepEqual(fs.readFileSync(path.join(f.out,'models',name)),bytes);
  packHtml(path.join(f.out,'index.html'),f.packed);const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  const error='THREE.USDLoader: Invalid USDZ package. The first ZIP entry must be a USD layer (.usd/.usda/.usdc).';
  assert.deepEqual(result.result,{order:['returned','loaded'],undefinedReturn:true,
    before:{name:'Triangle',positions:[0,0,0,1,0,0,0,1,0],uv:[0,0,1,0,0,1],position:[1,2,3],
      material:'MeshPhysicalMaterial',roughness:0.25,metalness:0.5,image:true,blob:true,colorSpace:'srgb',
      wrap:[1001,1002],repeat:[2,3],offset:[0.25,0.5],rotation:Math.PI/6},
    independent:true,callbackIdentity:true,readyAtCallback:true,parsedMeshes:1,legacyMeshes:1,
    real:[{name:'g0',vertices:75000,image:true}],untouched:{name:'Triangle',x:1,first:0,map:true},
    // The pinned importer reads scalar diffuse colors as sRGB, including the
    // original exporter's values; retain that conversion rather than fixing it.
    restored:{name:'Edited',position:[7,8,9],first:[4,5,6],
      color:[0.050876088164650994,0.2140411404715882,0.5225215539594343],roughness:0.625,map:null},
    zip:[80,75],failures:[error,error]});
  assert.equal(result.images.length,5);
  assert.ok(result.images.every(url=>url.startsWith('blob:')));
  assert.ok(result.requests.every(({url})=>url.startsWith('data:')));
  assert.ok(result.events.some(([kind,url])=>kind==='end'&&url==='models/scene%20%23.usdz?edition=2#root'));
});

test("USD archive texture fallbacks use only declared resources and retain source missing-reference behavior", async () => {
  const usd = new URL('../../upstream/three.js/examples/jsm/loaders/USDLoader.js', import.meta.url).href;
  const managedImage = 'data:image/png;base64,'+pixel.toString('base64');
  const f = fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import * as THREE from ${JSON.stringify(core.href)};
      import {USDLoader} from ${JSON.stringify(usd)};
      // An explicit finite resource also services the source composer's image
      // fallback. Unknown companion images are not guessed from archive paths.
      export const declaredImage=()=>new THREE.TextureLoader().load('nested/declared%20%23.png?edition=2');
      export const declaredModel=()=>new USDLoader().load('models/managed.usdz');
      const mesh=group=>{let result;group.traverse(object=>{if(object.isMesh)result=object;});return result;};
      export async function run(){
        const known=mesh(await new USDLoader().loadAsync('models/known.usdz'));
        const sequence=[],warnings=[],oldWarn=console.warn;console.warn=(...args)=>warnings.push(args);
        let missing,remote;
        try{
          missing=await new Promise((resolve,reject)=>new USDLoader().load('models/missing.usdz',group=>{
            sequence.push('loaded');resolve(group);
          },undefined,error=>{sequence.push('error');reject(error);}));
          remote=await new USDLoader().loadAsync('models/remote.usdz');
        }finally{console.warn=oldWarn;}
        const raw=await new THREE.FileLoader().setResponseType('arraybuffer').loadAsync('models/managed.usdz');
        const calls=[],manager=new THREE.LoadingManager();
        manager.onStart=url=>calls.push(['start',url]);manager.onProgress=url=>calls.push(['end',url]);
        manager.setURLModifier(url=>{calls.push(['resolve',url]);return ${JSON.stringify(managedImage)};});
        const managed=mesh(await new Promise((resolve,reject)=>new USDLoader(manager).parse(raw,'',resolve,reject)));
        return {known:known.material.map.image.complete,knownURL:known.material.map.image.src,sequence,
          missingImage:mesh(missing).material.map.image,remoteImage:mesh(remote).material.map.image,
          missingLayer:missing.getObjectByName('Missing').children.length,warnings,calls,
          managedURL:managed.material.map.image.src,managedReady:managed.material.map.image.complete};
      }`,
    'models/known.usdz':usdArchive('declared%20%23.png?edition=2',{image:false}),
    'models/missing.usdz':usdArchive('missing.png',{image:false,missingLayer:true}),
    'models/remote.usdz':usdArchive('https://outside.example/missing.png',{image:false}),
    'models/managed.usdz':usdArchive('managed.png',{image:false,directory:''}),
    'nested/declared #.png':pixel,
  });
  await buildApplication(f.entry,f.out,{prepareStandaloneLoaders:true});
  packHtml(path.join(f.out,'index.html'),f.packed);const html=fs.readFileSync(f.packed,'utf8');
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-unavailable'));
  fs.renameSync(f.out,path.join(f.root,'build-unavailable'));
  const result=await execute(html);
  assert.equal(result.ok,true,result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g,'<packed module>'));
  assert.equal(result.result.known,true);assert.match(result.result.knownURL,/^data:image\/png;f3d-id=/);
  assert.deepEqual({...result.result,known:undefined,knownURL:undefined},{known:undefined,knownURL:undefined,
    sequence:['loaded'],missingImage:null,remoteImage:null,missingLayer:0,
    warnings:[['USDLoader: Failed to load texture:','nested/missing.png'],
      ['USDLoader: Failed to load texture:','https://outside.example/missing.png']],
    calls:[['resolve','managed.png']],managedURL:managedImage,managedReady:true});
  assert.deepEqual(result.images,[result.result.knownURL,'nested/missing.png','https://outside.example/missing.png',managedImage]);
  assert.equal(result.requests.length,4,'missing USD references never produce layer requests');
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

test("only the exact pinned USD image assignment gains the resource host hook", () => {
  const url=new URL('../../upstream/three.js/examples/jsm/loaders/usd/USDComposer.js',import.meta.url);
  const source=fs.readFileSync(url,'utf8'),ast=parse(source,{ecmaVersion:'latest',sourceType:'module'});
  const edits=threeLoaderResourceEdits(source,ast,url.href),replacements=edits.filter(edit=>edit.type==='replace');
  assert.equal(replacements.length,1);
  assert.equal(source.slice(replacements[0].pos,replacements[0].end),'url');
  assert.equal(replacements[0].text,'__f3dResolveLoaderResource(url)');
  assert.deepEqual(threeLoaderResourceEdits(source,ast,'file:///other/USDComposer.js'),[]);
  for(const changed of [source+'\n// modified module\n','const Image=globalThis.CustomImage;\n'+source])
    assert.deepEqual(threeLoaderResourceEdits(changed,parse(changed,{ecmaVersion:'latest',sourceType:'module'}),url.href),[]);
  assert.equal(fs.readFileSync(url,'utf8'),source);
});

test("only exact pinned Draco, KTX2, Rhino3DM and Meshopt source constructors gain the native worker host hook", () => {
  for (const [file, argument, original] of [
    ["loaders/DRACOLoader.js", "this.workerSourceURL", "new Worker( this.workerSourceURL )"],
    ["loaders/KTX2Loader.js", "this.workerSourceURL", "new Worker( this.workerSourceURL )"],
    ["loaders/3DMLoader.js", "this.workerSourceURL", "new Worker( this.workerSourceURL )"],
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

test("packed DDS, KTX and TGA retain compressed mips, cube faces and decoded pixels without source files", async () => {
  // Small authored containers exercise the retained parsers, not a replacement
  // image codec: two DXT1 mip blocks, a six-face DDS and raw/RLE TGA pixels.
  const dds = (seed, cube = false) => {
    const bytes = Buffer.alloc(128 + (cube ? 6 : 1) * 16);
    const header = {0: 0x20534444, 1: 124, 2: 0x21007, 3: 4, 4: 4,
      7: 2, 19: 32, 20: 4, 21: 0x31545844, 27: 0x401008, 28: cube ? 0xfe00 : 0};
    for (const [word, value] of Object.entries(header)) bytes.writeUInt32LE(value, Number(word) * 4);
    for (let i = 128; i < bytes.length; i++) bytes[i] = (seed + i - 128) % 256;
    return bytes;
  };
  const ktx = Buffer.alloc(88);
  ktx.set([0xab, 0x4b, 0x54, 0x58, 0x20, 0x31, 0x31, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ktxHeader = [0x04030201, 0, 1, 0, 0x83f0, 0x1907, 4, 4, 0, 0, 1, 2, 0];
  ktxHeader.forEach((value, index) => ktx.writeUInt32LE(value, 12 + index * 4));
  ktx.writeUInt32LE(8, 64); ktx.set([40, 41, 42, 43, 44, 45, 46, 47], 68);
  ktx.writeUInt32LE(8, 76); ktx.set([48, 49, 50, 51, 52, 53, 54, 55], 80);
  const tga = Buffer.alloc(24); tga[2] = 2; tga.writeUInt16LE(2, 12); tga.writeUInt16LE(1, 14);
  tga[16] = 24; tga[17] = 0x20; tga.set([0, 0, 255, 0, 255, 0], 18);
  const rle = Buffer.from(tga.subarray(0, 18)); rle[2] = 10;
  const bad = Buffer.from(tga); bad[2] = 0;
  const faces = ['px', 'nx', 'py', 'ny', 'pz', 'nz'];
  const resources = {'textures/surface #.dds': dds(10), 'textures/cube.dds': dds(60, true),
    'textures/surface.ktx': ktx, 'textures/paint.tga': tga,
    'textures/repeated.tga': Buffer.concat([rle, Buffer.from([0x81, 255, 0, 0])]), 'textures/bad.tga': bad,
    ...Object.fromEntries(faces.map((face, index) => [`textures/${face}.dds`, dds(index * 16)])),
    ...Object.fromEntries(faces.map(face => [`textures/${face}.ktx`, ktx]))};
  const imports = ['DDSLoader', 'KTXLoader', 'TGALoader'].map(name =>
    `import {${name}} from ${JSON.stringify(new URL(`../../upstream/three.js/examples/jsm/loaders/${name}.js`, import.meta.url).href)};`).join('\n');
  const f = fixture({'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `${imports}
      export async function run() {
        const order=[], loader=new DDSLoader().setPath('textures/');
        let returned;
        const loaded=new Promise((resolve,reject)=>{
          returned=loader.load('surface%20%23.dds?v=2#mip',texture=>{order.push('loaded');resolve(texture);},undefined,reject);
        }); order.push('returned');
        const surface=await loaded, second=await loader.loadAsync('surface%20%23.dds?v=2#mip');
        const cube=await loader.loadAsync('cube.dds');
        const faces=['px.dds','nx.dds','py.dds','ny.dds','pz.dds','nz.dds'];
        const separate=await loader.loadAsync(faces);
        const ktx=await new KTXLoader().setPath('textures/').loadAsync('surface.ktx');
        const ktxCube=await new KTXLoader().setPath('textures/').loadAsync(['px.ktx','nx.ktx','py.ktx','ny.ktx','pz.ktx','nz.ktx']);
        const image=await new TGALoader().setPath('textures/').loadAsync('paint.tga');
        const repeated=await new TGALoader().setPath('textures/').loadAsync('repeated.tga');
        let error;try{await new TGALoader().setPath('textures/').loadAsync('bad.tga');}catch(e){error=e.message;}
        const mips=texture=>texture.mipmaps.map(mip=>({width:mip.width,height:mip.height,bytes:[...mip.data]}));
        const original=second.mipmaps[0].data[0]; surface.mipmaps[0].data[0]=255;
        return {order,returnedIdentity:surface===returned,independent:surface!==second&&second.mipmaps[0].data[0]===original,
          surface:{width:second.image.width,height:second.image.height,format:second.format,mips:mips(second)},
          cube:cube.image.map(mips),separate:separate.image.map(mips),
          ktx:{width:ktx.image.width,height:ktx.image.height,format:ktx.format,mips:mips(ktx)},
          ktxCube:ktxCube.image.map(mips),
          image:{width:image.image.width,height:image.image.height,pixels:[...image.image.data],flipY:image.flipY},
          repeated:[...repeated.image.data],error};
      }`, ...resources});
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const [name, bytes] of Object.entries(resources)) assert.deepEqual(fs.readFileSync(path.join(f.out, name)), bytes);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  fs.renameSync(path.join(f.root, 'source'), path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const result = await execute(html);
  assert.equal(result.ok, true, result.error?.split('\n')[0]);
  const mips = seed => [4, 2].map((size, level) => ({width: size, height: size,
    bytes: Array.from({length: 8}, (_, i) => seed + level * 8 + i)}));
  assert.deepEqual(result.result, {order: ['returned', 'loaded'], returnedIdentity: true, independent: true,
    surface: {width: 4, height: 4, format: 33776, mips: mips(10)},
    cube: faces.map((_, i) => mips(60 + i * 16)), separate: faces.map((_, i) => mips(i * 16)),
    ktx: {width: 4, height: 4, format: 33776, mips: mips(40)}, ktxCube: faces.map(() => mips(40)),
    image: {width: 2, height: 1, pixels: [255, 0, 0, 255, 0, 255, 0, 255], flipY: true},
    repeated: [0, 0, 255, 255, 0, 0, 255, 255], error: 'THREE.TGALoader: No data.'});
  assert.equal(result.requests.length, 19, 'each file and the independent DDS reload use the embedded resource host');
  assert.ok(result.requests.every(({url}) => url.startsWith('data:')));
});

test("packed standalone USDA retains sidecar images, captured bases and native callback completion", async () => {
  const model = Buffer.from("#usda 1.0\n(\n  upAxis = \"Y\"\n  metersPerUnit = 1\n)\ndef Mesh \"Triangle\" {\n  int[] faceVertexCounts = [3]\n  int[] faceVertexIndices = [0, 1, 2]\n  point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n  texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (\n    interpolation = \"vertex\"\n  )\n  rel material:binding = </Materials/Paint>\n}\ndef Scope \"Materials\" {\n  def Material \"Paint\" {\n    token outputs:surface.connect = </Materials/Paint/Surface.outputs:surface>\n    asset inputs:image = @\"./paint%20%23.png?edition=2#surface\"@\n    def Shader \"Surface\" {\n      uniform token info:id = \"UsdPreviewSurface\"\n      color3f inputs:diffuseColor.connect = </Materials/Paint/Image.outputs:rgb>\n      float inputs:roughness = 0.25\n      token outputs:surface\n    }\n    def Shader \"Image\" {\n      uniform token info:id = \"UsdUVTexture\"\n      asset inputs:file.connect = </Materials/Paint.inputs:image>\n      token inputs:wrapS = \"clamp\"\n      token inputs:wrapT = \"mirror\"\n      float3 outputs:rgb\n    }\n  }\n}\n");
  const usd = new URL("../../upstream/three.js/examples/jsm/loaders/USDLoader.js", import.meta.url).href;
  const legacy = new URL("../../upstream/three.js/examples/jsm/loaders/USDZLoader.js", import.meta.url).href;
  const f = fixture({"index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import * as THREE from ${JSON.stringify(core.href)}; export {THREE};
      import {USDLoader} from ${JSON.stringify(usd)};
      import {USDZLoader} from ${JSON.stringify(legacy)};
      export async function run() {
        const sequence=[], loader=new USDLoader().setPath('models/').setResourcePath('ignored/');
        let returned;
        const pending=new Promise((resolve,reject)=>{
          returned=loader.load('nested/scene%20%23.usda?edition=2#root',group=>{
            sequence.push('loaded'); resolve(group);
          },undefined,reject);
          sequence.push('returned');
        });
        loader.setPath('later/');
        const first=await pending;
        const second=await new USDLoader().loadAsync('models/nested/scene%20%23.usda?edition=2#root');
        const compatibility=await new USDZLoader().setPath('models/').loadAsync('nested/scene%20%23.usda?edition=2#root');
        const noSlash=await new USDLoader().setPath('flat').loadAsync('scene.usda');
        const raw=await new THREE.FileLoader().setResponseType('arraybuffer').loadAsync('models/nested/scene%20%23.usda?edition=2#root');
        let immediate,callbackIdentity=false,readyAtCallback=false;
        const parsed=await new Promise((resolve,reject)=>{
          immediate=loader.parse(raw,'models/',group=>{
            callbackIdentity=immediate===group;readyAtCallback=group.getObjectByName('Triangle').material.map.image.complete;
            resolve(group);
          },reject);
        });
        const a=first.getObjectByName('Triangle'),b=second.getObjectByName('Triangle');
        const states=[first,second,compatibility,noSlash,parsed].map(group=>{
          const mesh=group.getObjectByName('Triangle'),texture=mesh.material.map;
          return {vertices:mesh.geometry.attributes.position.count,complete:texture.image.complete,
            embedded:texture.image.src.startsWith('data:image/png;'),wrap:[texture.wrapS,texture.wrapT]};
        });
        return {sequence,undefinedReturn:returned===undefined,callbackIdentity,readyAtCallback,states,
          independent:first!==second&&a.geometry!==b.geometry&&a.material!==b.material&&a.material.map!==b.material.map};
      }`,
    "models/nested/scene #.usda": model, "flatscene.usda": model,
    "models/paint #.png": pixel, "models/nested/paint #.png": pixel, "flat/paint #.png": pixel,
  });
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders: true});
  for (const name of ["models/paint #.png", "models/nested/paint #.png", "flat/paint #.png"])
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), pixel);
  assert.equal(fs.existsSync(path.join(f.out, "ignored")), false);
  assert.deepEqual(fs.readFileSync(path.join(f.out, "models/nested/scene #.usda")), model);
  packHtml(path.join(f.out, "index.html"), f.packed);
  const html=fs.readFileSync(f.packed, "utf8");
  fs.renameSync(path.join(f.root, "source"), path.join(f.root, "source-unavailable"));
  fs.renameSync(f.out, path.join(f.root, "build-unavailable"));
  const result=await execute(html);
  assert.equal(result.ok, true, result.error?.replace(/data:text\/javascript;base64,[A-Za-z0-9+/=]+(?:#[a-f0-9]+)?/g, "<packed module>"));
  assert.deepEqual(result.result, {sequence:["returned","loaded"],undefinedReturn:true,callbackIdentity:true,readyAtCallback:true,
    independent:true,states:Array.from({length:5},()=>({vertices:3,complete:true,embedded:true,wrap:[1001,1002]}))});
  assert.equal(result.images.length,5);
  assert.ok(result.images.every(url=>url.startsWith("data:image/png;")));
  assert.ok(result.requests.every(({url})=>url.startsWith("data:")));
});
