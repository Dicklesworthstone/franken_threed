// Pinned retained FirstPersonControls with real DOM input, source WebGL vs
// automatic production routing. SwiftShader correctness, not hardware timing.
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
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-first-person-controls-'));
const imports = {
  three: pathToFileURL(path.join(upstream, 'build/three.module.js')).href,
  controls: pathToFileURL(path.join(upstream, 'examples/jsm/controls/FirstPersonControls.js')).href,
};
const entry = path.join(root, 'index.html');
fs.writeFileSync(entry, '<!doctype html><style>body{margin:0}canvas{display:inline-block;width:128px;height:96px}</style>' +
  '<script type="importmap">' + JSON.stringify({imports}) + '</script><script type="module" src="./main.js"></script>');
fs.writeFileSync(path.join(root, 'main.js'), `
import * as THREE from 'three';
import {FirstPersonControls} from 'controls';
function navigation(camera,canvas) { return new FirstPersonControls(camera,canvas).lookAt(0,0,0); }
async function main() {
  const renderer=new THREE.WebGLRenderer({antialias:false}); renderer.setSize(128,96,false);
  renderer.domElement.id='view'; renderer.domElement.tabIndex=0; document.body.appendChild(renderer.domElement);
  const second=document.createElement('canvas'); second.id='second'; second.width=128; second.height=96; second.tabIndex=0;
  document.body.appendChild(second);
  renderer.outputColorSpace=THREE.LinearSRGBColorSpace; renderer.toneMapping=THREE.NoToneMapping;
  const scene=new THREE.Scene(); scene.background=new THREE.Color(.1,.2,.3);
  const camera=new THREE.PerspectiveCamera(45,4/3,.1,20); camera.position.set(0,0,5);
  const controls=navigation(camera,renderer.domElement);
  controls.movementSpeed=1.5; controls.lookSpeed=.25; controls.dampingFactor=1;
  const geometry=new THREE.BoxGeometry(1,1,.6);
  const material=new THREE.MeshBasicMaterial({color:0x33aa77});
  const mesh=new THREE.Mesh(geometry,material); mesh.rotation.y=.2; scene.add(mesh);
  const otherMaterial=new THREE.MeshBasicMaterial({color:0xdd6655});
  const other=new THREE.Mesh(geometry,otherMaterial); other.position.set(1.4,.3,-1); scene.add(other);
  const target=new THREE.WebGLRenderTarget(128,96,{type:THREE.UnsignedByteType});
  target.texture.colorSpace=THREE.LinearSRGBColorSpace;
  if(renderer.isF3DRenderer===true) await renderer.init();
  renderer.setRenderTarget(target); await renderer.compileAsync(scene,camera);
  globalThis.__step=async (action,delta=0) => {
    if(action==='update') controls.update(delta);
    if(action==='disconnect') controls.disconnect();
    if(action==='connect') controls.connect(second);
    if(action==='dispose') controls.dispose();
    renderer.render(scene,camera);
    if(renderer.isF3DRenderer===true) await renderer.waitForGPU();
    const pixels=new Uint8Array(128*96*4);
    await renderer.readRenderTargetPixelsAsync(target,0,0,128,96,pixels);
    return {pixels:Array.from(pixels),position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),
      sourceControls:controls instanceof FirstPersonControls,sameCamera:controls.object===camera,
      secondCanvas:controls.domElement===second,dragging:controls.mouseDragOn,
      native:renderer.isF3DRenderer===true,route:renderer.info.f3d?.route,draws:renderer.info.f3d?.nativeDrawCalls??0};
  };
  globalThis.__finish=()=>{controls.dispose();target.dispose();geometry.dispose();material.dispose();otherMaterial.dispose();renderer.dispose();};
  globalThis.__ready=true;
}
main().catch(error=>{globalThis.__error=String(error.stack??error);});
`);
const options = {packageRootUrl:pathToFileURL(upstream + path.sep).href};
const candidate = await buildApplication(entry, path.join(root, 'candidate'), {...options, routeWebGLRenderer:true});
assert.equal(candidate.rendererRoute.routed, true);
assert.equal(candidate.rendererRoute.decisions[0].route, 'general-webgpu');
await buildApplication(entry, path.join(root, 'source'), options);
const server = http.createServer((request, response) => {
  if (new URL(request.url, 'http://localhost').pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
  const file = path.resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { response.writeHead(404).end(); return; }
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
      globalThis.__gpuErrors=[];
      const requestDevice=GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice=async function(...args) {
        const device=await requestDevice.apply(this,args);
        device.addEventListener('uncapturederror', event=>globalThis.__gpuErrors.push(event.error.message));
        return device;
      };
    });
    await page.goto('http://127.0.0.1:' + server.address().port + '/' + mode + '/index.html');
    await page.waitForFunction(() => globalThis.__ready || globalThis.__error, null, {timeout:30000});
    assert.equal(await page.evaluate(() => globalThis.__error), undefined, mode+' startup');
    const frames=[];
    const step=async (name,action='capture',delta=0) => {
      const value=await page.evaluate(([action,delta]) => globalThis.__step(action,delta), [action,delta]);
      frames.push({name,...value}); return value;
    };
    const first=await page.locator('#view').boundingBox(), second=await page.locator('#second').boundingBox();
    const move=async (box,dx=0,dy=0) => page.mouse.move(box.x+box.width/2+dx,box.y+box.height/2+dy);
    await step('initial');
    await page.keyboard.down('w'); await step('keyboard','update',.15); await page.keyboard.up('w');
    await move(first); await page.mouse.down(); await move(first,20,-10);
    await step('pointer','update',.12); await page.mouse.up();
    await page.keyboard.down('w'); await move(first); await page.mouse.down({button:'right'}); await move(first,-12,8);
    await step('combined','update',.1); await page.mouse.up({button:'right'}); await page.keyboard.up('w');
    await step('disconnect','disconnect');
    await page.keyboard.down('w'); await move(first); await page.mouse.down(); await move(first,15,8);
    await step('disconnected-input','update',.25); await page.mouse.up(); await page.keyboard.up('w');
    await step('reconnect','connect');
    await move(first); await page.mouse.down(); await move(first,-15,-8);
    await step('old-canvas-input','update',.25); await page.mouse.up();
    await move(second); await page.mouse.down(); await move(second,15,8);
    await step('new-canvas-input','update',.15); await page.mouse.up();
    await step('dispose','dispose');
    await page.keyboard.down('w'); await move(second); await page.mouse.down(); await move(second,-15,-8);
    await step('disposed-input','update',.25); await page.mouse.up(); await page.keyboard.up('w');
    await page.evaluate(() => globalThis.__finish());
    const gpuErrors=await page.evaluate(() => globalThis.__gpuErrors);
    await page.close(); assert.deepEqual(errors,[],mode+' browser errors'); assert.deepEqual(gpuErrors,[],mode+' GPU errors');
    assert.notDeepEqual(frames[1].position,frames[0].position,mode+' keyboard movement');
    assert.notDeepEqual(frames[2].quaternion,frames[1].quaternion,mode+' pointer rotation');
    for(const [before,after] of [[4,5],[6,7],[9,10]]) {
      assert.deepEqual(frames[after].position,frames[before].position,mode+' removed listeners: '+frames[after].name);
      assert.deepEqual(frames[after].quaternion,frames[before].quaternion,mode+' removed look listeners: '+frames[after].name);
    }
    assert.notDeepEqual(frames[8].position,frames[7].position,mode+' reconnected pointer movement');
    return frames;
  };
  const source=await capture('source'), native=await capture('candidate');
  assert.equal(native.length,11); assert.equal(source.length,native.length);
  let rgbaBytes=0, worstBadPixelFraction=0;
  for(let index=0;index<native.length;index++) {
    const expected=source[index], actual=native[index];
    assert.equal(expected.native,false); assert.equal(actual.native,true); assert.equal(actual.route,'general-webgpu');
    assert.ok(actual.draws>0); assert.equal(actual.sourceControls,true); assert.equal(actual.sameCamera,true);
    assert.equal(actual.secondCanvas,index>=6); assert.equal(actual.dragging,expected.dragging);
    assert.deepEqual(actual.position,expected.position,actual.name+' source camera position');
    assert.deepEqual(actual.quaternion,expected.quaternion,actual.name+' source camera orientation');
    assert.equal(actual.pixels.length,128*96*4); assert.equal(expected.pixels.length,actual.pixels.length);
    let bad=0;
    for(let offset=0;offset<actual.pixels.length;offset+=4)
      if([0,1,2,3].some(channel=>Math.abs(actual.pixels[offset+channel]-expected.pixels[offset+channel])>12)) bad++;
    const fraction=bad/(128*96); worstBadPixelFraction=Math.max(worstBadPixelFraction,fraction);
    assert.ok(fraction<=.02,actual.name+' exceeds the existing WebGL 2% pixel tolerance: '+fraction);
    assert.ok(actual.pixels.filter((value,offset)=>offset%4===0&&value!==actual.pixels[0]).length>100,actual.name+' visible geometry');
    rgbaBytes+=actual.pixels.length;
  }
  console.log(JSON.stringify({case:'first_person_controls',status:'pass',automaticRouting:true,route:'general-webgpu',
    checkpoints:native.length,rgbaBytes,worstBadPixelFraction,exactCameraStates:true,retainedSourceControls:true,
    nativeDrawCalls:native.at(-1).draws,browserErrors:0,gpuErrors:0,browserVersion:browser.version()}));
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
