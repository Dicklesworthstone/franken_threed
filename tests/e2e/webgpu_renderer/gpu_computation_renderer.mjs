// Unmodified pinned GPUComputationRenderer, automatically built to the native
// backend. Texture-compute state is compared with source WebGL, including
// independent variable histories, queued uniforms and vertex texture inputs.
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
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-gpu-computation-'));
const entry = path.join(root, 'index.html');
const imports = {three: pathToFileURL(path.join(upstream, 'build/three.module.js')).href,
  compute: pathToFileURL(path.join(upstream, 'examples/jsm/misc/GPUComputationRenderer.js')).href};
fs.writeFileSync(entry, '<!doctype html><script type="importmap">' + JSON.stringify({imports}) +
  '</script><script type="module" src="./main.js"></script>');
fs.writeFileSync(path.join(root, 'main.js'), `
import * as THREE from 'three';
import {GPUComputationRenderer} from 'compute';
async function main() {
  const half = new URLSearchParams(location.search).get('half') === '1';
  const renderer = new THREE.WebGLRenderer({antialias:false});
  renderer.setSize(64,48,false); document.body.appendChild(renderer.domElement);
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping;
  if(renderer.isF3DRenderer) await renderer.init();
  const gpu = new GPUComputationRenderer(8,4,renderer).setDataType(half ? THREE.HalfFloatType : THREE.FloatType);
  const initialPosition = gpu.createTexture(), initialVelocity = gpu.createTexture();
  for(let i=0;i<32;i++) {
    initialPosition.image.data.set([(i%8)/16,Math.floor(i/8)/8,.125,1],i*4);
    initialVelocity.image.data.set([.0625,.03125,(i%3)/32,.125],i*4);
  }
  const position = gpu.addVariable('positionTexture',
    'uniform float delta; void main(){vec2 uv=gl_FragCoord.xy/resolution.xy;'+
    'gl_FragColor=texture2D(positionTexture,uv)+texture2D(velocityTexture,uv)*delta;}', initialPosition);
  const velocity = gpu.addVariable('velocityTexture',
    'uniform float force; void main(){vec2 uv=gl_FragCoord.xy/resolution.xy;'+
    'gl_FragColor=texture2D(velocityTexture,uv)+texture2D(positionTexture,uv)*force;}', initialVelocity);
  gpu.setVariableDependencies(position,[position,velocity]);
  gpu.setVariableDependencies(velocity,[position,velocity]);
  position.material.uniforms.delta = {value:1}; velocity.material.uniforms.force = {value:.0625};
  const result = gpu.init();
  if(result !== null) throw Error('GPUComputationRenderer.init: '+result);
  const output = gpu.createRenderTarget();
  const filter = gpu.createShaderMaterial('uniform sampler2D inputTexture; void main(){'+
    'vec2 uv=gl_FragCoord.xy/resolution.xy; gl_FragColor=texture2D(inputTexture,vec2(1.0-uv.x,uv.y));}',
    {inputTexture:{value:null}});
  const frame = new THREE.WebGLRenderTarget(64,48,{type:THREE.UnsignedByteType});
  frame.texture.colorSpace=THREE.LinearSRGBColorSpace;
  const material = new THREE.ShaderMaterial({depthTest:false,depthWrite:false,
    uniforms:{inputTexture:{value:output.texture}},
    vertexShader:'varying vec2 vUv; uniform sampler2D inputTexture; void main(){vUv=uv;'+
      'vec2 offset=texture2D(inputTexture,uv).xy*.125; gl_Position=vec4(position.xy*.75+offset,0.,1.);}',
    fragmentShader:'varying vec2 vUv; uniform sampler2D inputTexture; void main(){'+
      'gl_FragColor=vec4(texture2D(inputTexture,vUv).rgb,1.);}' });
  const scene=new THREE.Scene(), camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,10);
  camera.position.z=2;
  const geometry=new THREE.PlaneGeometry(2,2,8,4); const mesh=new THREE.Mesh(geometry,material);
  mesh.frustumCulled=false; scene.add(mesh);
  async function read(target) {
    const values=half?new Uint16Array(128):new Float32Array(128);
    await renderer.readRenderTargetPixelsAsync(target,0,0,8,4,values);
    return half?Array.from(values,THREE.DataUtils.fromHalfFloat):Array.from(values);
  }
  async function capture(name) {
    const current=gpu.getCurrentRenderTarget(position), previous=gpu.getAlternateRenderTarget(position);
    filter.uniforms.inputTexture.value=current.texture;
    gpu.doRenderTarget(filter,output);
    renderer.setRenderTarget(frame); renderer.render(scene,camera); renderer.setRenderTarget(null);
    if(renderer.isF3DRenderer) await renderer.waitForGPU();
    const pixels=new Uint8Array(64*48*4);
    await renderer.readRenderTargetPixelsAsync(frame,0,0,64,48,pixels);
    return {name,index:gpu.currentTextureIndex,position:await read(current),previous:await read(previous),
      velocity:await read(gpu.getCurrentRenderTarget(velocity)),filtered:await read(output),pixels:Array.from(pixels),
      sourceInstance:gpu instanceof GPUComputationRenderer,variableIdentity:gpu.variables[0]===position,
      native:renderer.isF3DRenderer===true,route:renderer.info.f3d?.route,draws:renderer.info.f3d?.nativeDrawCalls??0};
  }
  globalThis.__step=async action => {
    if(action==='queued') {
      position.material.uniforms.delta.value=1; velocity.material.uniforms.force.value=.0625; gpu.compute();
      position.material.uniforms.delta.value=.5; velocity.material.uniforms.force.value=.125; gpu.compute();
    } else if(action==='third') {
      position.material.uniforms.delta.value=.25; velocity.material.uniforms.force.value=.25; gpu.compute();
    } else if(action==='reset') {
      gpu.renderTexture(initialPosition,gpu.getCurrentRenderTarget(position));
      gpu.renderTexture(initialVelocity,gpu.getCurrentRenderTarget(velocity));
    } else if(action==='restart') {
      position.material.uniforms.delta.value=2; velocity.material.uniforms.force.value=.125; gpu.compute();
    }
    return capture(action);
  };
  globalThis.__finish=()=>{gpu.dispose();output.dispose();filter.dispose();frame.dispose();material.dispose();geometry.dispose();renderer.dispose();};
  globalThis.__ready=true;
}
main().catch(error=>{globalThis.__error=String(error.stack??error);});
`);
const options = {packageRootUrl:pathToFileURL(upstream+path.sep).href};
const built = await buildApplication(entry,path.join(root,'candidate'),{...options,routeWebGLRenderer:true});
assert.equal(built.rendererRoute.routed,true);
assert.equal(built.rendererRoute.decisions[0].route,'general-webgpu');
await buildApplication(entry,path.join(root,'source'),options);
const server = http.createServer((request,response)=>{
  const url=new URL(request.url,'http://localhost');
  if(url.pathname==='/favicon.ico'){response.writeHead(204).end();return;}
  const file=path.resolve(root,'.'+url.pathname);
  if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){
    response.writeHead(404).end();return;
  }
  response.setHeader('Content-Type',file.endsWith('.html')?'text/html':'text/javascript');
  fs.createReadStream(file).pipe(response);
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try {
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH??'/opt/pw-browsers/chromium',
    args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-vulkan=swiftshader',
      '--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const capture=async(mode,half)=>{
    const page=await browser.newPage({userAgent:'OpenAI File Downloader, XaiImageApiFetch/1.0'}),errors=[];
    page.on('pageerror',error=>errors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    await page.addInitScript(()=>{
      globalThis.__gpuErrors=[];
      const requestDevice=GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice=async function(...args){
        const device=await requestDevice.apply(this,args);
        device.addEventListener('uncapturederror',event=>globalThis.__gpuErrors.push(event.error.message));
        return device;
      };
    });
    await page.goto('http://127.0.0.1:'+server.address().port+'/'+mode+'/index.html?half='+(half?1:0));
    await page.waitForFunction(()=>globalThis.__ready||globalThis.__error,null,{timeout:30000});
    assert.equal(await page.evaluate(()=>globalThis.__error),undefined,mode+' startup');
    const frames=[];
    for(const action of ['initial','queued','third','reset','restart'])
      frames.push(await page.evaluate(action=>globalThis.__step(action),action));
    await page.evaluate(()=>globalThis.__finish());
    const gpuErrors=await page.evaluate(()=>globalThis.__gpuErrors); await page.close();
    assert.deepEqual(errors,[],mode+' browser errors'); assert.deepEqual(gpuErrors,[],mode+' GPU errors');
    return frames;
  };
  let rgbaBytes=0, floatComponents=0, maxChannelError=0, worstBadPixelFraction=0;
  for(const half of [false,true]) {
    const source=await capture('source',half),candidate=await capture('candidate',half);
    assert.equal(source.length,5); assert.equal(candidate.length,5);
    for(let index=0;index<candidate.length;index++) {
      const expected=source[index],actual=candidate[index],label=(half?'half ':'float ')+actual.name;
      assert.equal(actual.native,true); assert.equal(expected.native,false); assert.equal(actual.route,'general-webgpu');
      assert.equal(actual.sourceInstance,true); assert.equal(actual.variableIdentity,true); assert.ok(actual.draws>0);
      assert.equal(actual.index,expected.index);
      for(const field of ['position','previous','velocity','filtered']) {
        assert.deepEqual(actual[field],expected[field],label+' '+field); floatComponents+=actual[field].length;
      }
      assert.equal(actual.pixels.length,expected.pixels.length);
      let badPixels=0;
      for(let offset=0;offset<actual.pixels.length;offset+=4) {
        const error=Math.max(...[0,1,2,3].map(channel=>Math.abs(actual.pixels[offset+channel]-expected.pixels[offset+channel])));
        maxChannelError=Math.max(maxChannelError,error); if(error>12) badPixels++;
      }
      const fraction=badPixels/(64*48); worstBadPixelFraction=Math.max(worstBadPixelFraction,fraction);
      assert.ok(fraction<=.02,label+' exceeds the existing WebGL 2% pixel tolerance: '+fraction);
      rgbaBytes+=actual.pixels.length;
      assert.ok(new Set(actual.pixels).size>8,label+' visible computation output');
      for(let y=0;y<4;y++)for(let x=0;x<8;x++)for(let c=0;c<4;c++)
        assert.equal(actual.filtered[(y*8+x)*4+c],actual.position[(y*8+7-x)*4+c],label+' manual filter orientation');
    }
    assert.notDeepEqual(candidate[1].position,candidate[0].position,'queued kernels change the state');
    assert.notDeepEqual(candidate[1].previous,candidate[1].position,'alternate target preserves the prior step');
    assert.deepEqual(candidate[3].position,candidate[0].position,'renderTexture restores the selected current image');
    assert.deepEqual(candidate[3].velocity,candidate[0].velocity,'reset restores each independent variable');
    assert.notDeepEqual(candidate[4].position,candidate[0].position,'simulation resumes after reset');
  }
  console.log(JSON.stringify({case:'gpu_computation_renderer',status:'pass',automaticRouting:true,
    route:'general-webgpu',retainedSourceAddon:true,types:['float32','float16'],checkpoints:10,
    floatComponents,rgbaBytes,exactState:true,exactPixels:maxChannelError===0,maxChannelError,worstBadPixelFraction,
    gpuErrors:0,browserErrors:0,browserVersion:browser.version()}));
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
