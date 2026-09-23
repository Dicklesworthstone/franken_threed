import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';
import {buildAnimation} from './build_animation.mjs';
import {buildApplication} from './build_application.mjs';
import {animationFixture} from './fixtures/animation/gltf_fixture.mjs';
const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const T=await import(pathToFileURL(path.join(root,'build/three.core.js')));
const data=(w=2,h=2,format=T.RGBAFormat)=>{
  const channels=format===T.RedFormat?1:4;
  const t=new T.DataTexture(new Uint8Array(w*h*channels).fill(90),w,h,format);t.needsUpdate=true;return t;
};
function fixture(material=new T.MeshBasicMaterial()){
  const d=textureDevice(),scene=new T.Scene(),camera=new T.OrthographicCamera(-2,2,2,-2,.1,10);
  camera.position.z=3;
  const geometry=new T.PlaneGeometry(),mesh=new T.Mesh(geometry,material);scene.add(mesh);
  const light=new T.AmbientLight(0xffffff,1);scene.add(light);
  return {d,scene,camera,geometry,mesh,material};
}
const create=(f,options={})=>createGpuThreeScene(f.d,f.scene,{three:T,...options});
const frame=()=>({colorView:{},depthView:{}});
const marks=d=>[d.buffers.length,d.textures.length,d.textureWrites.length,d.writes.length,d.submissions.length];

for(const instancing of [false,true])for(const renderBundles of [false,true])
test(`ordinary source map uploads live versions with stable bindings (${instancing}/${renderBundles})`,async()=>{
  const f=fixture(),t=data();t.colorSpace=T.SRGBColorSpace;f.material.map=t;
  const source=t.image,b=await create(f,{renderer:{instancing,renderBundles}});
  b.render(f.camera,frame());const first=f.d.snapshots.at(-1)[0],texture=f.d.textures[0];
  assert.equal(first.groups.get(1).group.entries[1].resource.texture,texture);
  assert.equal(first.groups.get(1).group.entries[1].resource.format,'rgba8unorm-srgb');
  assert.equal(b.diagnostics.textures.uploads,1);t.image.data.fill(120);b.render(f.camera,frame());
  assert.ok(f.d.snapshots.at(-1)[0].textureContents.get(texture)[0].every(x=>x===90));
  t.needsUpdate=true;t.offset.x=.25;b.render(f.camera,frame());
  assert.ok(f.d.snapshots.at(-1)[0].textureContents.get(texture)[0].every(x=>x===120));
  assert.ok(first.textureContents.get(texture)[0].every(x=>x===90));
  assert.equal(f.d.textures.length,1);assert.equal(t.image,source);assert.equal(f.material.map,t);
  assert.equal(b.diagnostics.textures.uploads,2);if(renderBundles)assert.equal(b.diagnostics.bundles.reuses,2);
  await b.whenIdle();b.dispose();assert.ok(texture.destroyed);assert.equal(t.image,source);
});

test('source clones share one native image and sampler across multiple source meshes',async()=>{
  const f=fixture(),a=data(),b=a.clone();f.material.map=a;
  const mesh=new T.Mesh(f.geometry,new T.MeshBasicMaterial({map:b}));f.scene.add(mesh);
  const bridge=await create(f);bridge.render(f.camera,frame());
  assert.equal(f.d.textures.length,1);assert.equal(bridge.diagnostics.textures.textures,2);
  assert.equal(bridge.diagnostics.textures.resources,1);assert.equal(bridge.diagnostics.textures.uploads,1);
  b.image.data.fill(155);b.needsUpdate=true;bridge.render(f.camera,frame());
  assert.ok(f.d.textures[0].levels[0].every(x=>x===155));bridge.dispose();
});

test('source disposal and sampler changes require prepare and invalidate old schedules',async()=>{
  const f=fixture(),t=data();f.material.map=t;const b=await create(f,{renderer:{renderBundles:true}});
  b.render(f.camera,frame());const old=f.d.textures[0];t.dispose();const before=marks(f.d);
  assert.throws(()=>b.render(f.camera,frame()),{code:'THREE_TEXTURE_PREPARE'});assert.deepEqual(marks(f.d),before);
  await b.prepare();b.render(f.camera,frame());assert.ok(old.destroyed);assert.equal(f.d.textures.length,2);
  t.wrapS=T.MirroredRepeatWrapping;t.needsUpdate=true;
  assert.throws(()=>b.render(f.camera,frame()),{code:'THREE_TEXTURE_PREPARE'});
  await b.prepare();b.render(f.camera,frame());assert.equal(f.d.snapshots.at(-1)[0].groups.get(1).group.entries[0].resource.addressModeU,'mirror-repeat');
  assert.equal(b.diagnostics.bundles.builds,3);await b.whenIdle();b.dispose();
});

test('borrowed overrides retain their acknowledgment and ownership contract alongside automatic textures',async()=>{
  const f=fixture(new T.MeshPhongMaterial()),t=data(),borrowed=new T.DataTexture(new Uint8Array(4),1,1);
  f.material.map=t;f.material.specularMap=borrowed;
  const binding={view:{},sampler:{},version:borrowed.version,sourceVersion:borrowed.source.version},textures=new Map([[borrowed,binding]]);
  const b=await create(f,{textures});b.render(f.camera,frame());assert.equal(f.d.textures.length,1);
  const group=f.d.snapshots.at(-1)[0].groups.get(1).group;assert.equal(group.entries[3].resource,binding.view);
  borrowed.needsUpdate=true;const before=marks(f.d);t.needsUpdate=true;
  assert.throws(()=>b.render(f.camera,frame()),{code:'THREE_SCENE_TEXTURE'});assert.deepEqual(marks(f.d),before);
  binding.version=borrowed.version;binding.sourceVersion=borrowed.source.version;b.render(f.camera,frame());
  await b.whenIdle();b.dispose();assert.deepEqual(binding.view,{});assert.equal(textures.size,1);
  const other=fixture();other.material.map=data();await assert.rejects(create(other,{autoTextures:false}),{code:'THREE_SCENE_TEXTURE'});
});

test('source material coverage includes color, normal, emission, occlusion, specular and UV-free toon ramps',async()=>{
  for(const type of ['phong','standard','toon']){
    const f=fixture(type==='phong'?new T.MeshPhongMaterial():type==='standard'?new T.MeshStandardMaterial():new T.MeshToonMaterial());
    if(type==='toon'){f.geometry.deleteAttribute('uv');f.material.gradientMap=data(4,1,T.RedFormat);}
    else{
      for(const name of ['map','normalMap','emissiveMap','aoMap'])f.material[name]=data();
      if(type==='phong')f.material.specularMap=data();else f.material.roughnessMap=f.material.metalnessMap=data();
    }
    const b=await create(f);b.render(f.camera,frame());
    assert.equal(b.diagnostics.textures.resources,type==='toon'?1:5);assert.equal(b.diagnostics.sourceDraws,1);
    if(type==='toon')assert.equal(f.d.textures[0].format,'r8unorm');await b.whenIdle();b.dispose();
  }
});

test('bad final material/ranges cannot partly upload earlier dirty textures',async()=>{
  const f=fixture(),a=data(),b=data();f.material.map=a;
  const material=new T.MeshBasicMaterial({map:b}),second=new T.Mesh(f.geometry,material);f.scene.add(second);
  const bridge=await create(f);a.needsUpdate=true;a.image.data.fill(55);material.wireframe=true;let before=marks(f.d);
  assert.throws(()=>bridge.render(f.camera,frame()),{code:'THREE_SCENE_MATERIAL'});assert.deepEqual(marks(f.d),before);
  material.wireframe=false;b.addUpdateRange(7,8);b.needsUpdate=true;
  assert.throws(()=>bridge.render(f.camera,frame()),{code:'THREE_TEXTURE_RANGE'});assert.deepEqual(marks(f.d),before);
  b.clearUpdateRanges();bridge.render(f.camera,frame());assert.ok(f.d.textures[0].levels[0].every(x=>x===55));bridge.dispose();
});

test('ownership rejects effectful source upload callbacks before any allocations',async()=>{
  const f=fixture(),t=data();f.material.map=t;let calls=0;t.onUpdate=()=>calls++;
  await assert.rejects(create(f),{code:'THREE_SCENE_HOOK'});assert.equal(calls,0);assert.equal(f.d.buffers.length,0);assert.equal(f.d.textures.length,0);
});

test('texture budgets are separate, aggregate and reclaim unused source maps on prepare',async()=>{
  const f=fixture(),a=data(),b=data();f.material.map=a;
  const bridge=await create(f,{texture:{maxTextureBytes:32,maxTextures:2}});assert.equal(bridge.diagnostics.textures.textureBytes,16);
  f.material.map=b;await bridge.prepare();assert.ok(f.d.textures[0].destroyed);assert.equal(bridge.diagnostics.textures.textureBytes,16);
  f.material.map=data(4,2);await assert.rejects(bridge.prepare(),{code:'THREE_TEXTURE_LIMIT'});
  assert.equal(bridge.diagnostics.textures.textureBytes,16);f.material.map=null;await bridge.prepare();assert.equal(bridge.diagnostics.textures.textureBytes,0);
  f.material.map=data(4,2);await bridge.prepare();bridge.render(f.camera,frame());assert.equal(bridge.diagnostics.textures.textureBytes,32);bridge.dispose();
});

test('texture storage changing during a pipeline wait is not published and can be prepared again',async()=>{
  const f=fixture(),bridge=await create(f),t=data();f.material.map=t;
  const original=f.d.createRenderPipelineAsync;let resolve;const waiting=new Promise(r=>resolve=r);
  f.d.createRenderPipelineAsync=async x=>{await waiting;return original(x);};
  const preparing=bridge.prepare();t.wrapS=T.RepeatWrapping;t.needsUpdate=true;resolve();
  await assert.rejects(preparing,{code:'THREE_TEXTURE_PREPARE'});assert.equal(bridge.failed,false);
  assert.equal(bridge.diagnostics.textures.textureBytes,0);f.d.createRenderPipelineAsync=original;
  await bridge.prepare();bridge.render(f.camera,frame());await bridge.whenIdle();bridge.dispose();
});

test('driver failures in automatic uploads terminate the bridge and release all owned resources',async()=>{
  const f=fixture(),t=data();f.material.map=t;const b=await create(f);const error=new Error('upload failed');
  f.d.textureWriteError=error;t.needsUpdate=true;
  assert.throws(()=>b.render(f.camera,frame()),e=>e===error);assert.equal(b.failed,true);
  assert.ok(f.d.textures.every(t=>t.destroyed));assert.ok(f.d.buffers.every(b=>b.destroyed));await assert.rejects(b.whenIdle());b.dispose();
});

test('generated mips are rebuilt before a textured scene submission without replacing its material',async()=>{
  const f=fixture(),t=data(4,4);t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;t.colorSpace=T.SRGBColorSpace;f.material.map=t;
  const b=await create(f,{renderer:{renderBundles:true}});b.render(f.camera,frame());const count=f.d.mipPasses.length;
  t.needsUpdate=true;b.render(f.camera,frame());assert.equal(f.d.mipPasses.length,count+2);
  assert.equal(b.diagnostics.bundles.reuses,1);assert.equal(f.d.textures.length,1);await b.whenIdle();b.dispose();
});

test('ordinary application builds execute compiled marching cubes with automatic Phong and toon textures',async t=>{
  const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-textured-marching-'));t.after(()=>fs.rm(tmp,{recursive:true,force:true}));
  const entry=path.join(tmp,'main.mjs');await fs.writeFile(entry,`export * as THREE from 'three';\nexport {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';\nexport {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};`);
  const built=await buildApplication(entry,path.join(tmp,'out'),{packageRootUrl:pathToFileURL(path.resolve(root)+path.sep).href,specializeNumeric:true});
  await fs.writeFile(path.join(built.outDir,'package.json'),'{"type":"module"}');
  const api=await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0]))),S=api.THREE,d=textureDevice();
  const source=new S.Scene(),camera=new S.PerspectiveCamera(45,1,.1,100);camera.position.z=3;source.add(new S.AmbientLight(0xffffff,1));
  const map=new S.DataTexture(new Uint8Array(16).fill(180),2,2);map.needsUpdate=true;
  const effect=new api.MarchingCubes(8,new S.MeshPhongMaterial({map}),true,true,1000);source.add(effect);
  const b=await createGpuThreeScene(d,source,{three:S,renderer:{renderBundles:true}});
  for(let i=0;i<4;i++){effect.reset();effect.addBall(.5,.5,.5,1.2,12);effect.update();map.image.data.fill(100+i);map.needsUpdate=true;b.render(camera,frame());}
  assert.equal(api.diagnostics(effect).wasmCalls,4);assert.equal(api.diagnostics(effect).fieldKernels.addBall.wasmCalls,4);
  assert.equal(d.snapshots.at(-1)[0].args[0],effect.count);assert.equal(d.textures.length,1);assert.equal(b.diagnostics.bundles.reuses,3);
  const ramp=new S.DataTexture(new Uint8Array([30,100,180,255]),4,1,S.RedFormat);ramp.needsUpdate=true;
  effect.material=new S.MeshToonMaterial({gradientMap:ramp});await b.prepare();effect.update();b.render(camera,frame());
  assert.equal(b.diagnostics.textures.resources,1);assert.equal(d.textures.at(-1).format,'r8unorm');assert.ok(d.textures[0].destroyed);
  await b.whenIdle();b.dispose();
});

test('optional relocated source-scene package executes automatic textures without original toolkit paths',async t=>{
  const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-owned-texture-package-'));t.after(()=>fs.rm(tmp,{recursive:true,force:true}));
  const f=animationFixture(),entry=path.join(tmp,'model.gltf'),out=path.join(tmp,'out');
  await fs.writeFile(entry,JSON.stringify(f.model));await fs.writeFile(path.join(tmp,'clip data.bin'),f.bytes);
  const built=buildAnimation(entry,out,{webgpu:true,threeScene:true}),moved=path.join(tmp,'relocated');await fs.rename(out,moved);
  const api=await import(pathToFileURL(path.join(moved,built.gpuEntry)));assert.equal(typeof api.createGpuThreeTextures,'function');
  const scene=fixture();scene.material.map=data();const b=await api.createGpuThreeScene(scene.d,scene.scene,{three:T});b.render(scene.camera,frame());
  assert.equal(b.diagnostics.textures.uploads,1);await b.whenIdle();b.dispose();
  const plain=buildAnimation(entry,path.join(tmp,'plain'),{webgpu:true});assert.ok(!plain.emittedFiles.includes('three_textures.mjs'));
  assert.ok(built.artifacts.some(a=>a.file==='three_textures.mjs'));
});
