// Pinned WebGL EXT_clip_control vs owned WebGPU reversed-depth submission.
// Depth32float samples, authored shaders, callbacks and shadows; no timing claim.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd:path.join(repo,'upstream/three.js'),maxBuffer:16*1024*1024});
  const hash = value => createHash('sha256').update(value).digest('hex');
  assert.equal(hash(await readFile(path.join(repo,'upstream/three.js',file))),hash(expected));
}
const bundle = await rollup({input:path.join(repo,'tools/ingest/three_renderer.mjs'),treeshake:false,
  onwarn(warning,warn) { if(warning.code!=='CIRCULAR_DEPENDENCY') warn(warning); }});
let rendererSource;
try { rendererSource=(await bundle.generate({format:'es',inlineDynamicImports:true})).output[0].code; }
finally { await bundle.close(); }
const server=http.createServer(async (request,response)=>{
  try {
    const pathname=new URL(request.url,'http://localhost').pathname;
    if(pathname==='/') { response.setHeader('Content-Type','text/html');response.end('<!doctype html>');return; }
    if(pathname==='/favicon.ico') { response.writeHead(204).end();return; }
    response.setHeader('Content-Type','text/javascript');
    if(pathname==='/renderer.mjs') { response.end(rendererSource);return; }
    const filename=path.resolve(repo,'.'+decodeURIComponent(pathname));
    if(!filename.startsWith(repo+path.sep)) { response.writeHead(403).end();return; }
    response.end(await readFile(filename));
  } catch(error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));

async function capture(mode) {
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH??'/opt/pw-browsers/chromium',
    args:['--no-sandbox','--no-zygote','--renderer-process-limit=1','--enable-unsafe-webgpu','--enable-features=Vulkan',
      '--use-vulkan=swiftshader','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try {
    const page=await browser.newPage({userAgent:'OpenAI File Downloader, XaiImageApiFetch/1.0'}),browserErrors=[];
    page.on('pageerror',error=>browserErrors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error') browserErrors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result=await page.evaluate(async mode=>{
      const T=await import('/upstream/three.js/build/three.module.js');
      const Renderer=mode==='source'?T.WebGLRenderer:(await import('/renderer.mjs')).createWebGLRendererClass(T);
      const profiles=[],errors=[],width=96,height=64;
      for(const reversedDepthBuffer of [false,true]) {
        const canvas=document.createElement('canvas');document.body.appendChild(canvas);
        const renderer=new Renderer({canvas,antialias:false,reversedDepthBuffer}),resources=[],frames=[],trace=[],cubeCameras=[];
        const own=value=>(resources.push(value),value);
        const output=own(new T.WebGLRenderTarget(width,height));
        output.depthTexture=own(new T.DepthTexture(width,height,T.FloatType));
        output.depthTexture.minFilter=output.depthTexture.magFilter=T.NearestFilter;
        const depthOutput=own(new T.WebGLRenderTarget(width,height,{type:T.FloatType,depthBuffer:false}));
        const geometry=own(new T.PlaneGeometry(1,1)),scene=new T.Scene();
        const nearMaterial=own(new T.MeshBasicMaterial({color:0x00ff00})),farMaterial=own(new T.MeshBasicMaterial({color:0xff0000}));
        const near=new T.Mesh(geometry,nearMaterial),far=new T.Mesh(geometry,farMaterial);scene.add(near,far);
        near.renderOrder=0;far.renderOrder=1;
        const perspective=new T.PerspectiveCamera(90,width/height,.1,2**40-1);
        const ortho=new T.OrthographicCamera(-1.5,1.5,1,-1,.1,64);
        const observe=(event,camera)=>trace.push([event,camera.reversedDepth,camera.coordinateSystem,
          camera.projectionMatrix.elements[10],camera.projectionMatrix.elements[14],
          camera===perspective?'perspective':camera===ortho?'orthographic':'shadow']);
        for(const mesh of [near,far]) {
          mesh.onBeforeRender=(_,__,camera)=>observe(mesh===near?'near-before':'far-before',camera);
          mesh.onAfterRender=(_,__,camera)=>observe(mesh===near?'near-after':'far-after',camera);
        }
        const sampled=own(new T.ShaderMaterial({uniforms:{map:{value:output.depthTexture}},depthTest:false,depthWrite:false,
          vertexShader:'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
          fragmentShader:'varying vec2 vUv;uniform sampler2D map;void main(){float d=texture2D(map,vUv).r;gl_FragColor=vec4(d,d,d,1.0);}'}));
        const quad=new T.Mesh(own(new T.PlaneGeometry(2,2)),sampled);quad.frustumCulled=false;
        const custom=own(new T.ShaderMaterial({vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
          fragmentShader:'void main(){gl_FragColor=vec4(0.0,1.0,0.0,1.0);}'}));
        const raw=own(new T.RawShaderMaterial({glslVersion:T.GLSL3,
          vertexShader:'precision highp float;uniform mat4 projectionMatrix;uniform mat4 modelViewMatrix;in vec3 position;void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
          fragmentShader:'precision highp float;out vec4 color;void main(){color=vec4(0.0,1.0,0.0,1.0);}'}));
        const arrange=(distance,difference,projected)=>{
          far.position.z=-distance;near.position.z=-(distance-difference);
          far.scale.set(3*(projected?distance:1),2*(projected?distance:1),1);
          near.scale.set(1.5*(projected?distance-difference:1),projected?distance-difference:1,1);
        };
        const read=async(name,target,start,float=false,face)=>{
          if(mode==='candidate') await renderer.waitForGPU();
          const w=target.width,h=target.height,pixels=float?new Float32Array(w*h*4):new Uint8Array(w*h*4);
          await renderer.readRenderTargetPixelsAsync(target,0,0,w,h,pixels,face);
          frames.push({name,float,width:w,height:h,pixels:Array.from(pixels),trace:trace.slice(start)});
        };
        const draw=async(name,root,camera,sampleDepth=false)=>{
          const start=trace.length;
          try {
            renderer.setRenderTarget(output);renderer.render(root,camera);await read(name,output,start);
            if(sampleDepth) { renderer.setRenderTarget(depthOutput);renderer.render(quad,ortho);await read(name+'-depth',depthOutput,trace.length,true); }
          } catch(error) { error.message=`reversed=${reversedDepthBuffer}/${name}: ${error.message}`;throw error; }
        };
        try {
          if(mode==='candidate') { await renderer.init();renderer._device.addEventListener('uncapturederror',event=>errors.push(event.error.message)); }
          if(mode==='source'&&!renderer.getContext().getExtension('EXT_clip_control')) throw Error('Source host lacks EXT_clip_control');
          renderer.setSize(width,height,false);renderer.outputColorSpace=T.LinearSRGBColorSpace;renderer.toneMapping=T.NoToneMapping;
          arrange(2**20-1,128,true);await draw('large-range',scene,perspective,true);
          perspective.far=2**30-1;perspective.updateProjectionMatrix();await draw('updated-far',scene,perspective,true);
          near.material=custom;await draw('shader-material',scene,perspective);
          near.material=raw;await draw('raw-material',scene,perspective);
          near.material=nearMaterial;arrange(31,1,false);await draw('orthographic',scene,ortho,true);
          far.renderOrder=0;near.renderOrder=1;arrange(4,2,false);
          for(const name of ['NeverDepth','AlwaysDepth','LessDepth','LessEqualDepth','EqualDepth','GreaterEqualDepth','GreaterDepth','NotEqualDepth']) {
            nearMaterial.depthFunc=T[name];await draw('depth-func-'+name,scene,ortho);
          }
          near.position.z=far.position.z;nearMaterial.depthFunc=T.EqualDepth;await draw('equal-coincident',scene,ortho);
          const pointGeometry=own(new T.BufferGeometry());
          pointGeometry.setAttribute('position',new T.Float32BufferAttribute([-.6,0,-.25,0,0,0,.6,0,.5],3));
          const pointMaterial=own(new T.RawShaderMaterial({glslVersion:T.GLSL3,depthTest:false,depthWrite:false,
            vertexShader:'precision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.0);gl_PointSize=8.0;}',
            fragmentShader:'precision highp float;out vec4 color;void main(){color=vec4(1.0,1.0,0.0,1.0);}'}));
          const points=new T.Points(pointGeometry,pointMaterial);points.frustumCulled=false;
          await draw('raw-point-clipping',points,ortho);
          const offsetScene=new T.Scene(),offsetGeometry=own(new T.PlaneGeometry(2.5,1.5));
          const offsetMaterial=own(new T.MeshBasicMaterial({color:0x00ff00,polygonOffset:true,polygonOffsetUnits:0}));
          const offsetBase=new T.Mesh(offsetGeometry,own(new T.MeshBasicMaterial({color:0xff0000})));
          const offsetOverlay=new T.Mesh(offsetGeometry,offsetMaterial);
          for(const mesh of [offsetBase,offsetOverlay]) { mesh.position.z=-3;mesh.rotation.y=Math.atan(.75);mesh.frustumCulled=false;offsetScene.add(mesh); }
          offsetBase.renderOrder=0;offsetOverlay.renderOrder=1;
          for(const [name,factor] of [['positive',1],['negative',-1]]) {
            offsetMaterial.polygonOffsetFactor=factor;await draw('polygon-offset-'+name,offsetScene,ortho);
          }
          const cubeTarget=own(new T.WebGLCubeRenderTarget(32,{generateMipmaps:false,minFilter:T.NearestFilter,magFilter:T.NearestFilter}));
          const cubeScene=new T.Scene(),cubeCamera=new T.CubeCamera(.1,12,cubeTarget);
          const cubeColors=[0xd32f2f,0x388e3c,0x1976d2,0xfbc02d,0x7b1fa2,0x0097a7];
          const cubeMaterials=cubeColors.map(color=>own(new T.MeshBasicMaterial({color,side:T.BackSide})));
          const cubeBox=new T.Mesh(own(new T.BoxGeometry(6,6,6)),cubeMaterials);cubeBox.frustumCulled=false;cubeScene.add(cubeBox);
          for(const phase of ['cold','warm']) {
            if(phase==='warm') {
              for(const [i,child] of cubeCamera.children.entries()) {
                child.near=.125+i/64;child.far=8+i;child.zoom=1+i/16;child.updateProjectionMatrix();
                cubeMaterials[i].color.setHex(cubeColors[(i+1)%6]);
              }
            }
            cubeCamera.update(renderer,cubeScene);
            if(mode==='candidate') await renderer.waitForGPU();
            cubeCameras.push({phase,children:cubeCamera.children.map(child=>({reversed:child.reversedDepth,
              system:child.coordinateSystem,projection:child.projectionMatrix.toArray(),inverse:child.projectionMatrixInverse.toArray()}))});
            for(let face=0;face<6;face++) await read(`cube-${phase}-${face}`,cubeTarget,trace.length,false,face);
          }
          renderer.shadowMap.enabled=true;
          for(const [kind,type] of [['directional','BasicShadowMap'],['directional','PCFShadowMap'],['directional','VSMShadowMap'],['point','BasicShadowMap'],['point','PCFShadowMap']]) {
            renderer.shadowMap.type=T[type];
            const shadowScene=new T.Scene(),camera=new T.PerspectiveCamera(45,1.5,.1,30);camera.position.set(0,2.2,4.5);camera.lookAt(0,0,0);
            const caster=new T.Mesh(own(new T.BoxGeometry(.75,.8,.75)),own(new T.MeshLambertMaterial({color:0xe5a362})));
            caster.position.y=.05;caster.castShadow=true;caster.frustumCulled=false;
            caster.onBeforeShadow=(_,__,___,child)=>{if(child!==light.shadow.camera) throw Error('Changed source shadow camera identity');observe('shadow-before',child);};
            caster.onAfterShadow=(_,__,___,child)=>{if(child!==light.shadow.camera) throw Error('Changed source shadow camera identity');observe('shadow-after',child);};
            const floor=new T.Mesh(own(new T.PlaneGeometry(6,6)),own(new T.MeshLambertMaterial({color:0xd2dce5})));
            floor.rotation.x=-Math.PI/2;floor.position.y=-.4;floor.receiveShadow=true;
            const light=kind==='point'?new T.PointLight(0xffffff,20,0,2):new T.DirectionalLight(0xffffff,2);
            light.position.set(1.5,2.5,3);light.castShadow=true;light.shadow.mapSize.set(128,128);light.shadow.bias=-.0001;
            if(kind==='directional') { light.shadow.camera.left=light.shadow.camera.bottom=-3;light.shadow.camera.right=light.shadow.camera.top=3; }
            shadowScene.add(caster,floor,light,new T.AmbientLight(0xffffff,.15));
            try {
              await draw('shadow-'+kind+'-'+type,shadowScene,camera);
              if(kind==='directional'&&type==='BasicShadowMap') { caster.castShadow=false;await draw('shadow-no-caster',shadowScene,camera); }
            } finally { light.shadow.dispose(); }
          }
          const gl=mode==='source'?renderer.getContext():null;
          if(gl&&gl.getError()!==gl.NO_ERROR) throw Error('Source WebGL error');
          profiles.push({reversedDepthBuffer,reported:renderer.capabilities.reversedDepthBuffer,frames,cubeCameras,
            cameraReversed:perspective.reversedDepth,cameraSystem:perspective.coordinateSystem,
            expectedSystem:T.WebGLCoordinateSystem,route:renderer.info.f3d?.route,nativeDrawCalls:renderer.info.f3d?.nativeDrawCalls});
        } finally { for(const resource of resources) resource.dispose();renderer.dispose();canvas.remove(); }
      }
      return {profiles,errors};
    },mode);
    assert.deepEqual(browserErrors,[],mode+' browser errors');assert.deepEqual(result.errors,[],mode+' GPU errors');
    return {...result,browserVersion:browser.version()};
  } finally { await browser.close(); }
}
const pixel=(frame,x=frame.width/2,y=frame.height/2)=>frame.pixels.slice((y*frame.width+x)*4,(y*frame.width+x+1)*4);
try {
  const source=await capture('source'),candidate=await capture('candidate'),results=[];
  let rgbaBytes=0,depthSamples=0;
  for(const [index,expected] of source.profiles.entries()) {
    const actual=candidate.profiles[index],byName=Object.fromEntries(expected.frames.map(frame=>[frame.name,frame]));
    const candidateFrames=Object.fromEntries(actual.frames.map(frame=>[frame.name,frame]));
    assert.equal(expected.reported,expected.reversedDepthBuffer,'source extension must actually activate');
    assert.equal(actual.reported,expected.reported);assert.equal(actual.cameraReversed,expected.cameraReversed);
    assert.equal(actual.cameraSystem,actual.expectedSystem);assert.equal(actual.route,'general-webgpu');assert.ok(actual.nativeDrawCalls>0);
    assert.deepEqual(pixel(byName['large-range']),expected.reversedDepthBuffer?[0,255,0,255]:[255,0,0,255]);
    assert.equal(byName['large-range'].trace[0][1],false,'the first before-render hook observes the source projection');
    assert.equal(byName['large-range'].trace[1][1],expected.reversedDepthBuffer,'the source draw publishes reversed camera state');
    if(expected.reversedDepthBuffer) {
      const depth=pixel(byName['large-range-depth'])[0];assert.ok(depth>0&&depth<1e-6,'sampled depth32float retains distant precision');
      assert.notDeepEqual(byName['large-range-depth'].pixels,byName['updated-far-depth'].pixels,'updated far plane changes sampled source depths');
      assert.notDeepEqual(candidateFrames['large-range-depth'].pixels,candidateFrames['updated-far-depth'].pixels,'updated far plane reaches the candidate GPU projection');
    }
    assert.deepEqual(pixel(byName['raw-point-clipping'],19),expected.reversedDepthBuffer?[0,0,0,255]:[255,255,0,255]);
    assert.deepEqual(pixel(byName['raw-point-clipping'],48),[255,255,0,255]);
    assert.deepEqual(pixel(byName['polygon-offset-positive']),[255,0,0,255],'positive slope offset moves the coplanar overlay away');
    assert.deepEqual(pixel(byName['polygon-offset-negative']),[0,255,0,255],'negative slope offset moves the coplanar overlay toward the camera');
    assert.deepEqual(actual.cubeCameras,expected.cubeCameras,'completed cube draws publish every public child projection');
    for(const state of expected.cubeCameras) {
      assert.equal(state.children.length,6);
      for(const child of state.children) {
        assert.equal(child.reversed,expected.reversedDepthBuffer);assert.equal(child.system,expected.expectedSystem);
        assert.ok(child.projection.every(Number.isFinite));assert.ok(child.inverse.every(Number.isFinite));
      }
      assert.equal(new Set(Array.from({length:6},(_,face)=>pixel(byName[`cube-${state.phase}-${face}`]).join(','))).size,6,'each cube face retains its distinct source image');
      for(let face=0;face<6;face++) {
        const frame=byName[`cube-${state.phase}-${face}`];
        assert.ok(frame.pixels.some((value,i)=>i%4!==3&&value!==0),`${frame.name} has visible source color`);
        assert.ok(candidateFrames[frame.name].pixels.some((value,i)=>i%4!==3&&value!==0),`${frame.name} has visible candidate color`);
      }
    }
    for(let face=0;face<6;face++) assert.notDeepEqual(byName[`cube-cold-${face}`].pixels,byName[`cube-warm-${face}`].pixels,'warm cube draw replaces each source face');
    assert.notDeepEqual(byName['shadow-directional-BasicShadowMap'].pixels,byName['shadow-no-caster'].pixels,'visible source shadow');
    assert.equal(actual.frames.length,expected.frames.length);
    for(const [i,reference] of expected.frames.entries()) {
      const frame=actual.frames[i];assert.equal(frame.name,reference.name);
      assert.equal(frame.pixels.length,reference.pixels.length);
      assert.deepEqual(frame.trace,reference.trace,`${expected.reversedDepthBuffer}/${reference.name} camera callbacks`);
      let maxError=0,differingBytes=0,bad=0;
      for(let offset=0;offset<reference.pixels.length;offset+=4) {
        let over=false;
        for(let channel=0;channel<4;channel++) {
          const value=reference.pixels[offset+channel],error=Math.abs(frame.pixels[offset+channel]-value);
          assert.ok(Number.isFinite(error),reference.name+' finite depth/color sample');
          maxError=Math.max(maxError,error);if(error) differingBytes++;
          if(reference.float?error>Math.max(1e-12,Math.abs(value)*2e-6):error>(reference.name.startsWith('shadow-')?12:0)) over=true;
        }
        if(over) bad++;
      }
      const allowed=reference.name.startsWith('shadow-')?.02:0;
      results.push({reversedDepthBuffer:expected.reversedDepthBuffer,checkpoint:reference.name,maxError,differingBytes,badPixels:bad,pixelCount:reference.pixels.length/4,allowedFraction:allowed});
      if(reference.float) depthSamples+=reference.pixels.length/4;else rgbaBytes+=reference.pixels.length;
    }
  }
  const passed=results.every(result=>result.badPixels/result.pixelCount<=result.allowedFraction);
  console.log(JSON.stringify({case:'program_reversed_depth',status:passed?'pass':'fail',results,rgbaBytes,depthSamples,gpuErrors:0,browserErrors:0,
    browserVersion:candidate.browserVersion,sourcePin:pin}));
  assert.ok(passed,'source/candidate mismatch; see per-checkpoint results above');
} finally { await new Promise(resolve=>server.close(resolve)); }
