// Production application routing through local renderer/control factories.
// Source WebGPU vs the emitted native route; software-GPU correctness only.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from '../../../tools/ingest/build_application.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const upstream = path.join(repo, 'upstream/three.js');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-renderer-factories-'));
const imports = {
  'three/webgpu': pathToFileURL(path.join(upstream, 'build/three.webgpu.js')).href,
  'three': pathToFileURL(path.join(upstream, 'build/three.module.js')).href,
  'controls': pathToFileURL(path.join(upstream, 'examples/jsm/controls/OrbitControls.js')).href,
};
const entry = path.join(root, 'index.html');
fs.writeFileSync(entry, '<!doctype html><script type="importmap">' + JSON.stringify({imports}) +
  '</script><script type="module" src="./main.js"></script>');
fs.writeFileSync(path.join(root, 'main.js'), `
import * as THREE from 'three/webgpu';
import {OrbitControls} from 'controls';
const events=[];
function makeCanvas() { events.push('canvas'); return document.createElement('canvas'); }
function makeRenderer(canvas) {
  events.push('renderer'); return new THREE.WebGPURenderer({canvas,antialias:false});
}
function makeView(canvas) { events.push('view'); return makeRenderer(canvas); }
function makeControls(camera,canvas) { return new OrbitControls(camera,canvas); }
async function main() {
  const renderer=makeView(makeCanvas()); renderer.setSize(128,96,false);
  document.body.appendChild(renderer.domElement);
  renderer.outputColorSpace=THREE.LinearSRGBColorSpace; renderer.toneMapping=THREE.NoToneMapping;
  const scene=new THREE.Scene(); scene.background=new THREE.Color(.1,.2,.3);
  const camera=new THREE.PerspectiveCamera(45,4/3,.1,20); camera.position.z=3;
  const controls=makeControls(camera,renderer.domElement); controls.rotateLeft(.2); controls.update();
  const geometry=new THREE.BoxGeometry(1,1,.5), material=new THREE.MeshBasicMaterial({color:0x33aa77});
  scene.add(new THREE.Mesh(geometry,material));
  await renderer.init();
  // Whole 256-byte rows make the source and candidate readback layout equal.
  const target=new THREE.RenderTarget(128,96,{type:THREE.UnsignedByteType});
  target.texture.colorSpace=THREE.LinearSRGBColorSpace; renderer.setRenderTarget(target);
  await renderer.compileAsync(scene,camera); renderer.render(scene,camera);
  if(renderer.isF3DRenderer===true) await renderer.waitForGPU();
  const pixels=await renderer.readRenderTargetPixelsAsync(target,0,0,128,96);
  globalThis.__result={pixels:Array.from(pixels),events,sourceControls:controls instanceof OrbitControls,
    sameCamera:controls.object===camera,camera:camera.position.toArray(),native:renderer.isF3DRenderer===true,
    route:renderer.info.f3d?.route,draws:renderer.info.f3d?.nativeDrawCalls??0};
  controls.dispose(); target.dispose(); geometry.dispose(); material.dispose(); renderer.dispose();
}
main().catch(error=>{globalThis.__result={error:String(error.stack??error)};});
`);
const options = {packageRootUrl:pathToFileURL(upstream + path.sep).href};
const candidate = await buildApplication(entry, path.join(root, 'candidate'), {...options, routeWebGPURenderer:true});
assert.equal(candidate.rendererRoute.routed, true);
assert.equal(candidate.rendererRoute.decisions[0].route, 'general-webgpu');
await buildApplication(entry, path.join(root, 'source'), options);
const server = http.createServer((request, response) => {
  if (new URL(request.url, 'http://localhost').pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
  const file = path.resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    response.writeHead(404).end(); return;
  }
  response.setHeader('Content-Type', file.endsWith('.html') ? 'text/html' : 'text/javascript');
  fs.createReadStream(file).pipe(response);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-vulkan=swiftshader',
      '--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const capture = async mode => {
    const page = await browser.newPage({userAgent:'OpenAI File Downloader, XaiImageApiFetch/1.0'}), errors=[];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => { if(message.type()==='error') errors.push(message.text()); });
    await page.addInitScript(() => {
      const createView=GPUTexture.prototype.createView;
      GPUTexture.prototype.createView=function(descriptor) {
        if(descriptor?.swizzle==='rgba') { descriptor={...descriptor}; delete descriptor.swizzle; }
        return createView.call(this,descriptor);
      };
      globalThis.__gpuErrors=[];
      const requestDevice=GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice=async function(...args) {
        const device=await requestDevice.apply(this,args);
        device.addEventListener('uncapturederror', event=>globalThis.__gpuErrors.push(event.error.message));
        return device;
      };
    });
    await page.goto('http://127.0.0.1:' + server.address().port + '/' + mode + '/index.html');
    await page.waitForFunction(() => globalThis.__result, null, {timeout:30000});
    const result=await page.evaluate(() => ({...globalThis.__result,gpuErrors:globalThis.__gpuErrors}));
    await page.close();
    assert.deepEqual(errors, [], mode+' browser errors'); assert.deepEqual(result.gpuErrors, [], mode+' GPU errors');
    assert.equal(result.error, undefined); return result;
  };
  const source=await capture('source'), native=await capture('candidate');
  assert.equal(source.native,false); assert.equal(native.native,true); assert.equal(native.route,'general-webgpu');
  assert.ok(native.draws>0); assert.equal(native.sourceControls,true); assert.equal(native.sameCamera,true);
  assert.deepEqual(native.events,source.events); assert.deepEqual(native.camera,source.camera);
  assert.equal(native.pixels.length,128*96*4); assert.equal(source.pixels.length,native.pixels.length);
  const mismatch=native.pixels.findIndex((value,index)=>value!==source.pixels[index]);
  assert.equal(mismatch,-1,'first differing RGBA byte: '+mismatch);
  assert.ok(native.pixels.filter((value,index)=>index%4===0&&value!==native.pixels[0]).length>100,'geometry must visibly render');
  console.log(JSON.stringify({case:'renderer_factories',status:'pass',automaticRouting:true,route:native.route,
    nativeDrawCalls:native.draws,rgbaBytes:native.pixels.length,exactComparison:true,
    browserErrors:0,gpuErrors:0,browserVersion:browser.version()}));
} finally {
  await browser?.close(); await new Promise(resolve => server.close(resolve));
}
