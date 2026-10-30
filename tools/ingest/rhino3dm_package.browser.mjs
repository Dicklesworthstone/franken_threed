// Retained CAD parser, native Wasm/Blob workers and image decoding, compared
// with pinned source over HTTP and then packed offline. No renderer timing.
// RHINO3DM_DIR=/path/to/rhino3dm-8.32.1/package CHROMIUM_PATH=/path/to/chrome \
// PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ node tools/ingest/rhino3dm_package.browser.mjs
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const codec = process.env.RHINO3DM_DIR;
assert.ok(codec, 'RHINO3DM_DIR must contain the unmodified rhino3dm 8.32.1 npm package');
const hashes = {
  'rhino3dm.js': '7f3b804afda0cafbf456d5729f2980a6011cf90e882b8156c674e78696e2005a',
  'rhino3dm.wasm': '75c3965576d6e7ea1e1451dafd11300316c4783f0cb915a5b612d9ebabac7210',
};
for (const [name, hash] of Object.entries(hashes))
  assert.equal(createHash('sha256').update(fs.readFileSync(path.join(codec, name))).digest('hex'), hash, name);
const upstream = fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));
const loader = new URL('../../upstream/three.js/examples/jsm/loaders/3DMLoader.js', import.meta.url);
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-rhino3dm-package-'));
const source = path.join(root, 'source'), output = path.join(root, 'build'), packed = path.join(root, 'packed.html');
for (const directory of ['models', 'codecs']) fs.mkdirSync(path.join(source, directory), {recursive: true});
fs.copyFileSync(path.join(upstream, 'examples/models/3dm/Rhino_Logo.3dm'), path.join(source, 'models/cad #.3dm'));
for (const name of Object.keys(hashes)) fs.copyFileSync(path.join(codec, name), path.join(source, 'codecs', name));

function application(specifier) {
  return `import {Rhino3dmLoader} from ${JSON.stringify(specifier)};
function hash(array){let value=2166136261;
  for(const byte of new Uint8Array(array.buffer,array.byteOffset,array.byteLength))value=Math.imul(value^byte,16777619)>>>0;
  return value;}
function firstMesh(scene){let result;scene.traverse(object=>{if(!result&&object.isMesh)result=object});return result;}
async function summarize(scene){
  const nodes=[],images=new Map();let meshes=0,lines=0,vertices=0,triangles=0;
  scene.updateMatrixWorld(true);
  const materials=[];scene.traverse(object=>{if(object.material)materials.push(...[object.material].flat())});
  for(const material of materials){const image=material.map?.image;if(!image||images.has(image))continue;
    if(image instanceof HTMLImageElement)await image.decode();
    const canvas=document.createElement('canvas');canvas.width=image.naturalWidth??image.width;canvas.height=image.naturalHeight??image.height;
    const context=canvas.getContext('2d');context.drawImage(image,0,0);
    const pixels=context.getImageData(0,0,canvas.width,canvas.height).data;
    images.set(image,{width:canvas.width,height:canvas.height,hash:hash(pixels),firstPixel:[...pixels.slice(0,4)]});
  }
  scene.traverse(object=>{
    const node={type:object.type,name:object.name,visible:object.visible,children:object.children.length,
      matrix:[...object.matrixWorld.elements],userData:object===scene?undefined:object.userData};
    if(object.geometry){const geometry=object.geometry;vertices+=geometry.attributes.position.count;
      if(object.isMesh){meshes++;triangles+=(geometry.index?.count??geometry.attributes.position.count)/3;}
      if(object.isLine)lines++;
      node.attributes=Object.fromEntries(Object.entries(geometry.attributes).map(([name,a])=>[name,
        {count:a.count,size:a.itemSize,normalized:a.normalized,type:a.array.constructor.name,hash:hash(a.array)}]));
      node.index=geometry.index?{count:geometry.index.count,hash:hash(geometry.index.array)}:null;
      node.materials=[object.material].flat().map(m=>({type:m.type,name:m.name,color:m.color?.toArray(),opacity:m.opacity,
        vertexColors:m.vertexColors,metalness:m.metalness,roughness:m.roughness,
        map:m.map?{image:images.get(m.map.image),wrap:[m.map.wrapS,m.map.wrapT],repeat:m.map.repeat.toArray(),colorSpace:m.map.colorSpace}:null}));
    }nodes.push(node);
  });
  return JSON.parse(JSON.stringify({nodes,meshes,lines,vertices,triangles,images:[...images.values()],metadata:scene.userData}));
}
globalThis.__f3dRhinoRun=async(runtimeBytes)=>{
  const sequence=[],loader=new Rhino3dmLoader().setLibraryPath('before/').setWorkerLimit(1).setSubdivisionLevel(2);
  const pending=loader.setPath('models/').loadAsync('cad%20%23.3dm?version=2#assembly');
  loader.setLibraryPath('codecs/');sequence.push('returned');
  const cad=await pending;sequence.push('loaded');
  const raw=Uint8Array.from(runtimeBytes).buffer;
  const parsed=await new Promise((resolve,reject)=>loader.parse(raw,value=>{sequence.push('parsed');resolve(value)},reject));
  const independentLoader=new Rhino3dmLoader().setLibraryPath('codecs/').setWorkerLimit(1).setSubdivisionLevel(2);
  const other=await independentLoader.parseAsync(Uint8Array.from(runtimeBytes).buffer);
  const first=firstMesh(parsed),second=firstMesh(other);
  const independent=parsed!==other&&first.geometry!==second.geometry&&first.material!==second.material&&first.material.map!==second.material.map;
  independentLoader.dispose();
  let invalidError;try{await loader.parseAsync(new Uint8Array([1,2,3,4]).buffer)}catch(error){invalidError={type:error.type,id:error.id,message:error.error.message};}
  loader.dispose();
  const reopened=await loader.decodeObjects(Uint8Array.from(runtimeBytes).buffer,'runtime.3dm');
  const result={cad:await summarize(cad),parsed:await summarize(parsed),other:await summarize(other),reopened:await summarize(reopened),
    sequence,independent,inputTransferred:raw.byteLength===0,invalidError};
  loader.dispose();return result;
};`;
}
fs.writeFileSync(path.join(source, 'index.html'), '<script type="module" src="app.mjs"></script>', {flag: 'wx'});
fs.writeFileSync(path.join(source, 'app.mjs'), application(loader.href), {flag: 'wx'});
await buildApplication(path.join(source, 'index.html'), output, {prepareStandaloneLoaders: true});
for (const name of ['models/cad #.3dm', ...Object.keys(hashes).map(name => 'codecs/' + name)])
  assert.deepEqual(fs.readFileSync(path.join(output, name)), fs.readFileSync(path.join(source, name)));
packHtml(path.join(output, 'index.html'), packed);

// This model is created after packing and supplied as runtime input. Its mesh,
// curve, metadata and embedded texture cannot have been predecoded by the build.
const require = createRequire(import.meta.url), rhino = await require(path.join(path.resolve(codec), 'rhino3dm.js'))();
const document = new rhino.File3dm(), material = new rhino.Material(), layer = new rhino.Layer();
material.name = 'runtime paint';material.diffuseColor = {r:255,g:255,b:255,a:255};material.setBitmapTextureFilename('runtime.png');
layer.name = 'runtime layer';layer.renderMaterialIndex = document.materials().add(material);
const layerIndex = document.layers().add(layer), attributes = new rhino.ObjectAttributes(), mesh = new rhino.Mesh();
attributes.name = 'unseen mesh';attributes.layerIndex = layerIndex;attributes.setUserString('origin', 'runtime input');
for (const vertex of [[0,0,0],[1,0,0],[0,1,0]]) mesh.vertices().add(...vertex);
mesh.faces().addTriFace(0,1,2);mesh.normals().computeNormals();
for (const uv of [[0,0],[1,0],[0,1]]) mesh.textureCoordinates().add(...uv);
for (const color of [[255,0,0],[0,255,0],[0,0,255]]) mesh.vertexColors().add(...color);
document.objects().addMesh(mesh, attributes);
const curve = new rhino.LineCurve([0,0,0], [2,1,0]), curveAttributes = new rhino.ObjectAttributes();
curveAttributes.name = 'runtime line';curveAttributes.layerIndex = layerIndex;document.objects().addCurve(curve, curveAttributes);
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==', 'base64');
const embedded = rhino.EmbeddedFile.fromByteArray(png);embedded.fileName = 'runtime.png';document.embeddedFiles().add(embedded);
const runtimeBytes = [...document.toByteArray()];
for (const object of [embedded, curveAttributes, curve, mesh, attributes, layer, material, document]) object.delete();

const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const server = http.createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<script type="importmap">{"imports":{"three":"/upstream/build/three.module.js"}}</script><script type="module" src="/app.mjs"></script>');return;
    }
    if (pathname === '/app.mjs') {response.setHeader('Content-Type', 'text/javascript');response.end(application('/upstream/examples/jsm/loaders/3DMLoader.js'));return;}
    if (pathname === '/favicon.ico') {response.writeHead(204).end();return;}
    const base = pathname.startsWith('/upstream/') ? upstream : source;
    const relative = pathname.startsWith('/upstream/') ? pathname.slice('/upstream/'.length) : pathname.slice(1);
    const filename = path.resolve(base, relative);
    if (!filename.startsWith(path.resolve(base) + path.sep)) {response.writeHead(403).end();return;}
    response.setHeader('Content-Type', filename.endsWith('.js') ? 'text/javascript' : filename.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    response.end(fs.readFileSync(filename));
  } catch {response.writeHead(404).end('missing');}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let serverOpen = true;
async function capture(url, offline) {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH, headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--no-zygote', '--single-process', '--disable-dev-shm-usage']});
  try {
    const context = await browser.newContext({offline, userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'});
    const page = await context.newPage(), errors = [], external = [], workers = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    page.on('request', request => {if (offline && /^https?:/.test(request.url())) external.push(request.url());});
    page.on('worker', worker => workers.push({url:worker.url(), closed:new Promise(resolve => worker.once('close', resolve))}));
    await page.goto(url);await page.waitForFunction(() => typeof globalThis.__f3dRhinoRun === 'function');
    const result = await page.evaluate(bytes => globalThis.__f3dRhinoRun(bytes), runtimeBytes);
    let timer;
    try {await Promise.race([Promise.all(workers.map(worker => worker.closed)), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Source dispose did not terminate its workers')), 5000);
    })]);} finally {clearTimeout(timer);}
    assert.equal(workers.length, 3, 'one reused worker, one independent decoder and one recreated after disposal');
    assert.ok(workers.every(worker => worker.url.startsWith('blob:')));
    assert.deepEqual(errors, [], 'native browser errors');assert.deepEqual(external, [], 'offline external requests');
    return {result, workers:workers.length, browserVersion:browser.version()};
  } finally {await browser.close();}
}
try {
  const reference = await capture(`http://127.0.0.1:${server.address().port}/`, false);
  await new Promise(resolve => server.close(resolve));serverOpen = false;
  fs.renameSync(source, path.join(root, 'source-unavailable'));fs.renameSync(output, path.join(root, 'build-unavailable'));
  const candidate = await capture(pathToFileURL(packed).href, true), result = candidate.result;
  assert.deepEqual(result, reference.result);
  assert.deepEqual(result.sequence, ['returned', 'loaded', 'parsed']);assert.equal(result.independent, true);
  assert.equal(result.inputTransferred, true);assert.equal(result.invalidError.type, 'error');assert.ok(result.invalidError.message.length > 0);
  assert.ok(result.cad.meshes > 10 && result.cad.vertices > 10000);assert.equal(result.cad.metadata.layers.length, 6);
  assert.equal(result.parsed.meshes, 1);assert.equal(result.parsed.lines, 1);assert.equal(result.parsed.triangles, 1);
  assert.deepEqual(result.parsed.images.map(image => image.firstPixel), [[255,0,0,255]]);
  // Warnings accumulate on each source loader; independent instances need not
  // share the first loader's earlier CAD-model warnings.
  assert.deepEqual(result.parsed.nodes, result.other.nodes);
  assert.deepEqual(result.parsed.nodes, result.reopened.nodes);
  console.log(JSON.stringify({status:'pass', browser:candidate.browserVersion, packed,
    meshes:result.cad.meshes, vertices:result.cad.vertices, triangles:result.cad.triangles, layers:result.cad.metadata.layers.length,
    runtimeBytes:runtimeBytes.length, runtimeImages:result.parsed.images, codecVersion:'8.32.1',
    nativeWorkers:candidate.workers, disposedWorkers:candidate.workers, offline:true, externalRequests:0, browserErrors:0, exactSceneComparison:true}));
} finally {if (serverOpen) await new Promise(resolve => server.close(resolve));}
