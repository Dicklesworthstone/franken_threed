// Retained source 3MF parsing versus packed offline execution with native DOM,
// ZIP/image decoding and Blob lifetime. This is not GPU or performance evidence.
// CHROMIUM_PATH=/path/to/chrome PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
//   node tools/ingest/three_mf_package.browser.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const upstream = fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));
const loader = new URL('../../upstream/three.js/examples/jsm/loaders/3MFLoader.js', import.meta.url);
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-3mf-package-'));
const source = path.join(root, 'source'), output = path.join(root, 'build'), packed = path.join(root, 'packed.html');
fs.mkdirSync(path.join(source, 'models'), {recursive: true});
for (const [from, to] of [['truck.3mf', 'truck #.3mf'], ['facecolors.3mf', 'colors.3mf']])
  fs.copyFileSync(path.join(upstream, 'examples/models/3mf', from), path.join(source, 'models', to));

function application(specifier) {
  return `import {ThreeMFLoader} from ${JSON.stringify(specifier)};
function hash(array){let value=2166136261;
  for(const byte of new Uint8Array(array.buffer,array.byteOffset,array.byteLength))value=Math.imul(value^byte,16777619)>>>0;
  return value;}
function meshes(scene){const result=[];scene.traverse(object=>{if(object.isMesh)result.push(object)});return result;}
async function summarize(scene){
  const imageData=new Map(),geometries=new Set(),nodes=[];let vertices=0,triangles=0;
  const all=meshes(scene);
  for(const mesh of all)for(const material of [mesh.material].flat()){
    const image=material.map?.image;if(!image||imageData.has(image))continue;
    await image.decode();const canvas=document.createElement('canvas');
    canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
    imageData.set(image,{width:canvas.width,height:canvas.height,pixels:hash(ctx.getImageData(0,0,canvas.width,canvas.height).data)});
  }
  scene.updateMatrixWorld(true);
  scene.traverse(object=>{
    const node={type:object.type,name:object.name,children:object.children.length,matrix:[...object.matrixWorld.elements]};
    if(object.isMesh){const geometry=object.geometry;geometries.add(geometry);
      vertices+=geometry.attributes.position.count;triangles+=(geometry.index?.count??geometry.attributes.position.count)/3;
      node.attributes=Object.fromEntries(Object.entries(geometry.attributes).map(([name,a])=>[name,{count:a.count,size:a.itemSize,normalized:a.normalized,hash:hash(a.array)}]));
      node.index=geometry.index?{count:geometry.index.count,hash:hash(geometry.index.array)}:null;
      node.materials=[object.material].flat().map(m=>({type:m.type,name:m.name,color:m.color?.toArray(),opacity:m.opacity,
        vertexColors:m.vertexColors,metalness:m.metalness,roughness:m.roughness,
        map:m.map?{image:imageData.get(m.map.image),wrap:[m.map.wrapS,m.map.wrapT],colorSpace:m.map.colorSpace}:null}));
    }nodes.push(node);
  });
  return {nodes,meshes:all.length,geometries:geometries.size,vertices,triangles,images:[...imageData.values()]};
}
globalThis.__f3dThreeMFRun=async()=>{
  const truck=await new ThreeMFLoader().setPath('models/').loadAsync('truck%20%23.3mf?edition=2#components');
  const sequence=[],loader=new ThreeMFLoader().setPath('models/');
  const pending=new Promise((resolve,reject)=>loader.load('colors.3mf',value=>{sequence.push('loaded');resolve(value)},undefined,reject));
  sequence.push('returned');const colors=await pending;
  const second=await new ThreeMFLoader().setPath('models/').loadAsync('colors.3mf');
  const a=meshes(colors)[0],b=meshes(second)[0];
  return {truck:await summarize(truck),colors:await summarize(colors),second:await summarize(second),sequence,
    independent:colors!==second&&a.geometry!==b.geometry&&a.material!==b.material};
};`;
}
fs.writeFileSync(path.join(source, 'index.html'), '<script type="module" src="app.mjs"></script>', {flag: 'wx'});
fs.writeFileSync(path.join(source, 'app.mjs'), application(loader.href), {flag: 'wx'});
await buildApplication(path.join(source, 'index.html'), output, {prepareStandaloneLoaders: true});
for (const name of ['truck #.3mf', 'colors.3mf'])
  assert.deepEqual(fs.readFileSync(path.join(output, 'models', name)), fs.readFileSync(path.join(source, 'models', name)));
packHtml(path.join(output, 'index.html'), packed);

const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const server = http.createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<script type="importmap">{"imports":{"three":"/upstream/build/three.module.js"}}</script><script type="module" src="/app.mjs"></script>'); return;
    }
    if (pathname === '/app.mjs') {
      response.setHeader('Content-Type', 'text/javascript');
      response.end(application('/upstream/examples/jsm/loaders/3MFLoader.js')); return;
    }
    if (pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
    const base = pathname.startsWith('/upstream/') ? upstream : source;
    const relative = pathname.startsWith('/upstream/') ? pathname.slice('/upstream/'.length) : pathname.slice(1);
    const filename = path.resolve(base, relative);
    if (!filename.startsWith(path.resolve(base) + path.sep)) { response.writeHead(403).end(); return; }
    response.setHeader('Content-Type', filename.endsWith('.js') ? 'text/javascript' : 'application/octet-stream');
    response.end(fs.readFileSync(filename));
  } catch { response.writeHead(404).end('missing'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let serverOpen = true;
async function capture(url, offline) {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH, headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--no-zygote', '--single-process', '--disable-dev-shm-usage']});
  try {
    const context = await browser.newContext({offline, userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'});
    const page = await context.newPage(), errors = [], external = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    page.on('request', request => {if (offline && /^https?:/.test(request.url())) external.push(request.url());});
    await page.goto(url); await page.waitForFunction(() => typeof globalThis.__f3dThreeMFRun === 'function');
    const result = await page.evaluate(() => globalThis.__f3dThreeMFRun());
    assert.deepEqual(errors, [], 'native browser errors');
    assert.deepEqual(external, [], 'offline execution made an external request');
    return {result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const reference = await capture(`http://127.0.0.1:${server.address().port}/`, false);
  await new Promise(resolve => server.close(resolve)); serverOpen = false;
  fs.renameSync(source, path.join(root, 'source-unavailable'));
  fs.renameSync(output, path.join(root, 'build-unavailable'));
  const candidate = await capture(pathToFileURL(packed).href, true);
  assert.deepEqual(candidate.result, reference.result);
  const result = candidate.result;
  assert.deepEqual(result.sequence, ['returned', 'loaded']); assert.equal(result.independent, true);
  assert.ok(result.truck.meshes > 10 && result.truck.vertices > 1000);
  // The archive's two build items contain 10 and 24 component wrappers; the
  // wrappers themselves each reference one mesh (68 component edges total).
  assert.equal(result.truck.nodes[0].children, 2);
  for (const count of [10, 24]) assert.ok(result.truck.nodes.some(node => node.type === 'Group' && node.children === count));
  assert.ok(result.truck.images.length > 0 && result.truck.images.every(image => image.width > 0 && image.height > 0));
  assert.deepEqual(result.colors, result.second);
  console.log(JSON.stringify({status: 'pass', browser: candidate.browserVersion, packed,
    archiveBytes: fs.statSync(path.join(root, 'source-unavailable/models/truck #.3mf')).size,
    meshes: result.truck.meshes, geometries: result.truck.geometries, vertices: result.truck.vertices,
    triangles: result.truck.triangles, images: result.truck.images,
    offline: true, externalRequests: 0, browserErrors: 0, exactSceneComparison: true}));
} finally { if (serverOpen) await new Promise(resolve => server.close(resolve)); }
