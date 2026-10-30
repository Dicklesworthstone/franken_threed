// Full retained Soldier applications with source-derived palette Wasm enabled.
// Both lanes use the pinned WebGL renderer; this tests CPU island equivalence,
// not new-renderer ownership or hardware acceleration.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';

const repo=path.resolve(fileURLToPath(new URL('../../',import.meta.url))),upstream=path.join(repo,'upstream/three.js');
const {chromium}=createRequire(process.env.PLAYWRIGHT_MODULE_DIR??'/opt/node-tools/node_modules/')('playwright');
const pin='148ef33ecb6d2502ff796d4554abd1549c95d519';
const digest=value=>createHash('sha256').update(value).digest('hex');
for(const file of ['build/three.core.js','build/three.module.js','examples/jsm/loaders/GLTFLoader.js','examples/models/gltf/Soldier.glb']) {
  const original=execFileSync('git',['show',`${pin}:${file}`],{cwd:upstream,maxBuffer:16*1024*1024});
  assert.equal(digest(await readFile(path.join(upstream,file))),digest(original));
}

async function run() {
  const canvas=document.createElement('canvas');document.body.appendChild(canvas);
  const renderer=new THREE.WebGLRenderer({canvas,antialias:false}),width=128,height=96;
  renderer.setSize(width,height,false);renderer.outputColorSpace=THREE.SRGBColorSpace;
  const target=new THREE.WebGLRenderTarget(width,height),scene=new THREE.Scene();
  scene.background=new THREE.Color(0x182738);
  scene.add(new THREE.HemisphereLight(0xffffff,0x444444,3));
  const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(3,5,4);scene.add(light);
  const camera=new THREE.PerspectiveCamera(40,width/height,.1,100);camera.position.set(4,2.4,6);camera.lookAt(0,1,0);
  const loader=new GLTFLoader(),models=[],mixers=[],actions=[],skins=[],events=[],traces=[],frames=[];
  for(let instance=0;instance<2;instance++) {
    const model=await loader.loadAsync('/upstream/three.js/examples/models/gltf/Soldier.glb');
    model.scene.position.x=instance===0?-1:1;scene.add(model.scene);models.push(model);
    const mixer=new THREE.AnimationMixer(model.scene);mixers.push(mixer);
    const set=model.animations.map(clip=>mixer.clipAction(clip));actions.push(set);
    const idle=set.find(action=>action.getClip().name==='Idle'),walk=set.find(action=>action.getClip().name==='Walk');
    const run=set.find(action=>action.getClip().name==='Run');
    (instance===0?walk:run).play();idle.play().setEffectiveWeight(.2);
    mixer.addEventListener('loop',event=>events.push([instance,event.action.getClip().name,event.loopDelta,mixer.time]));
    model.scene.traverse(object=>{if(object.isSkinnedMesh) {
      skins.push(object);object.onBeforeRender=()=>traces.push([instance,object.name,Array.from(object.skeleton.boneMatrices.slice(0,16))]);
    }});
  }
  const references=skins.map(mesh=>mesh.skeleton.boneMatrices);
  try {
    for(let frame=0;frame<12;frame++) {
      if(frame===3)for(let i=0;i<2;i++) {
        const walk=actions[i].find(action=>action.getClip().name==='Walk'),run=actions[i].find(action=>action.getClip().name==='Run');
        run.reset().play();walk.crossFadeTo(run,.4,false);
      }
      if(frame===6)for(const mesh of skins)mesh.skeleton.boneInverses[Math.min(4,mesh.skeleton.boneInverses.length-1)].elements[12]+=.07;
      if(frame===8)for(const mesh of skins){mesh.skeleton.dispose();mesh.skeleton.computeBoneTexture();}
      for(let i=0;i<2;i++)mixers[i].update((i===0?.13:.17)*(frame===10?-1:1));
      if(frame===9)skins[0].skeleton.bones[Math.min(7,skins[0].skeleton.bones.length-1)].rotation.z+=.2;
      scene.updateMatrixWorld(true);renderer.setRenderTarget(target);renderer.render(scene,camera);
      const pixels=new Uint8Array(width*height*4);renderer.readRenderTargetPixels(target,0,0,width,height,pixels);
      const state=skins.map((mesh,index)=>{
        const points=[];
        const count=mesh.geometry.attributes.position.count;
        for(const vertex of [0,11,101,501])points.push(mesh.getVertexPosition(Math.min(vertex,count-1),new THREE.Vector3()).toArray());
        return {name:mesh.name,palette:Array.from(new Uint8Array(mesh.skeleton.boneMatrices.buffer)),
          textureVersion:mesh.skeleton.boneTexture.version,sourceVersion:mesh.skeleton.boneTexture.source.version,
          stable:frame===0||frame>=8?null:mesh.skeleton.boneMatrices===references[index],points,
          bones:mesh.skeleton.bones.map(bone=>bone.matrixWorld.toArray())};
      });
      if(frame===0)for(let i=0;i<skins.length;i++)references[i]=skins[i].skeleton.boneMatrices;
      frames.push({frame,pixels:Array.from(pixels),state,mixers:mixers.map(mixer=>mixer.time)});
    }
    const gl=renderer.getContext(),error=gl.getError();if(error!==gl.NO_ERROR)throw new Error('WebGL error '+error);
    return {frames,events,traces,skins:skins.length,bones:skins.map(mesh=>mesh.skeleton.bones.length),
      independent:skins.length>=2&&skins[0].skeleton!==skins.at(-1).skeleton,
      diagnostics:diagnostics?diagnostics(THREE.Skeleton):null};
  }finally{renderer.dispose();target.dispose();for(const skin of skins)skin.skeleton.dispose();}
}

const modules=new Map();let report;
for(const candidate of [false,true]) {
  const entry='\0skeleton-browser',plugin=marchingCubesRollupPlugin();
  const source=`import * as THREE from ${JSON.stringify(path.join(upstream,'build/three.module.js'))};
    import {GLTFLoader} from ${JSON.stringify(path.join(upstream,'examples/jsm/loaders/GLTFLoader.js'))};`+
    (candidate?`import {skeletonUpdateDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-skeleton-update-adapter')};`:'const diagnostics=null;')+
    '\nexport '+run.toString();
  const bundle=await rollup({input:entry,onwarn(warning,warn){if(warning.code!=='CIRCULAR_DEPENDENCY')warn(warning);},
    plugins:[{name:'skeleton-browser',resolveId:id=>id===entry?id:id==='three'?path.join(upstream,'build/three.module.js'):null,
      load:id=>id===entry?source:null},...(candidate?[plugin]:[])]});
  try{modules.set(candidate?'/candidate.mjs':'/source.mjs',(await bundle.generate({format:'es'})).output[0].code);}
  finally{await bundle.close();}
  if(candidate)report=plugin.api.getSkeletonUpdatesReport();
}
assert.equal(report.compiledModules,1);
const server=http.createServer(async(request,response)=>{
  try {
    const pathname=new URL(request.url,'http://localhost').pathname;
    if(pathname==='/'){response.setHeader('Content-Type','text/html');response.end('<!doctype html>');return;}
    if(pathname==='/favicon.ico'){response.writeHead(204).end();return;}
    if(modules.has(pathname)){response.setHeader('Content-Type','text/javascript');response.end(modules.get(pathname));return;}
    const filename=path.resolve(repo,'.'+decodeURIComponent(pathname));
    if(!filename.startsWith(repo+path.sep)){response.writeHead(403).end();return;}
    response.end(await readFile(filename));
  }catch(error){response.writeHead(404).end(String(error));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
async function capture(candidate) {
  const browser=await chromium.launch({headless:true,timeout:30000,
    executablePath:process.env.CHROMIUM_PATH??'/opt/pw-browsers/chromium',
    args:['--no-sandbox','--no-zygote','--renderer-process-limit=1','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try {
    const page=await browser.newPage({userAgent:'OpenAI File Downloader, XaiImageApiFetch/1.0'}),errors=[];
    page.on('pageerror',error=>errors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`,{timeout:30000});
    const result=await page.evaluate(async path=>(await import(path)).run(),candidate?'/candidate.mjs':'/source.mjs');
    assert.deepEqual(errors,[]);return {...result,browserVersion:browser.version()};
  }finally{await browser.close();}
}
try {
  const source=await capture(false),candidate=await capture(true);
  assert.deepEqual(candidate.frames,source.frames);assert.deepEqual(candidate.events,source.events);assert.deepEqual(candidate.traces,source.traces);
  assert.deepEqual(candidate.bones,source.bones);assert.ok(candidate.independent);assert.ok(candidate.diagnostics.wasmCalls>=24);
  assert.equal(candidate.diagnostics.fallbackCalls,0);assert.ok(candidate.frames.every(frame=>frame.state.every(state=>state.stable!==false)));
  assert.notDeepEqual(candidate.frames[0].pixels,candidate.frames[7].pixels);
  assert.ok(candidate.frames.some(frame=>frame.pixels.some((value,index)=>index%4!==3&&value>100)));
  console.log(JSON.stringify({case:'skeleton_update',status:'pass',checkpoints:candidate.frames.length,
    rgbaBytes:candidate.frames.reduce((sum,frame)=>sum+frame.pixels.length,0),skins:candidate.skins,bones:candidate.bones,
    diagnostics:candidate.diagnostics,wasmBytes:report.modules[0].wasmBytes,browserErrors:0,
    browserVersion:candidate.browserVersion,sourcePin:pin,rendererOwner:'retained-WebGL',accelerationClaim:false}));
}finally{await new Promise(resolve=>server.close(resolve));}
