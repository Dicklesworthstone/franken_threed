/** Actual pinned Three.js objects, scene traversal, GLSL compilation and native
 * renderer. Only the GPU is a recording boundary; WGSL/pixels are not executed. */
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from '../three_scene.mjs';
import {createThreeProgramSupport} from '../three_program.mjs';
import {createGpuThreeTextures} from '../three_textures.mjs';
import {geometryDevice} from '../fixtures/gpu_geometry_device.mjs';
import {device as volumeDevice} from './volume_texture_test_fixture.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.module.js')));
function gpu(){
  const d=geometryDevice(),v=volumeDevice();
  Object.assign(d.limits,v.limits);
  Object.assign(d,{features:v.features,textures:v.textures,textureWrites:v.writes,
    createTexture:v.createTexture,createSampler:v.createSampler,createRenderPipeline:v.createRenderPipeline});
  d.queue.writeTexture=v.queue.writeTexture;
  const submit=d.queue.submit;
  d.queue.submit=commands=>{
    submit(commands);
    for(const draw of d.snapshots.at(-1)){
      draw.textureContents=new Map();
      for(const {group} of draw.groups.values())for(const entry of group.entries??[]){
        const t=entry.resource?.texture;
        if(t?.levels)draw.textureContents.set(t,t.levels.map(level=>level.slice()));
      }
    }
  };
  return d;
}
function source(kind='volume',integer=false){
  const C=kind==='volume'?T.Data3DTexture:kind==='array'?T.DataArrayTexture:T.DataTexture;
  const t=new C(integer?new (integer==='signed'?Int32Array:Uint32Array)(2*2*(kind==='2d'?1:2)*4).fill(0x80000001):new Uint8Array(2*2*(kind==='2d'?1:2)*4).fill(7),2,2,2);
  t.format=integer?T.RGBAIntegerFormat:T.RGBAFormat;
  if(integer)t.type=integer==='signed'?T.IntType:T.UnsignedIntType;
  t.needsUpdate=true;return t;
}
function shader(t,kind='volume',integer=false){
  const sampler=(integer==='signed'?'i':integer?'u':'')+(kind==='volume'?'sampler3D':kind==='array'?'sampler2DArray':'sampler2D');
  const coord=kind==='2d'?'ivec2(0)':'ivec3(0)';
  return new T.ShaderMaterial({glslVersion:T.GLSL3,uniforms:{dataMap:{value:t}},
    vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader:`precision highp ${sampler}; uniform ${sampler} dataMap; out vec4 outputColor; void main(){outputColor=vec4(texelFetch(dataMap,${coord},0));}`});
}
async function scene(t,kind='volume',integer=false,options={}){
  const {device:d=gpu(),...limits}=options;
  const root=new T.Scene(),material=shader(t,kind,integer),object=new T.Mesh(new T.PlaneGeometry(),material);
  root.add(object);const camera=new T.PerspectiveCamera(45,1,.1,10);camera.position.z=3;
  const program=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const bridge=await createGpuThreeScene(d,root,{three:T,program,...limits});
  return {d,root,material,object,camera,bridge,render(){bridge.render(camera,{colorView:{},depthView:{}});}};
}
for(const kind of ['2d','array','volume'])for(const integral of [false,true,'signed'])
 test(`${kind}/${integral==='signed'?'sint':integral?'uint':'unorm'}: scene uses shared ownership, versions and disposal`,async()=>{
  const t=source(kind,integral),s=await scene(t,kind,integral),{d,bridge}=s;
  try{
    assert.equal(bridge.diagnostics.textures?.resources,1);
    assert.equal(bridge.diagnostics.textures?.textureBytes,t.image.data.byteLength);
    s.render();const native=d.textures[0],before=native.levels[0].slice();
    const bytes=new Uint8Array(t.image.data.buffer);bytes.fill(23);s.render();
    assert.deepEqual(native.levels[0],before,'unrequested CPU mutations stay stale');
    t.needsUpdate=true;s.render();assert.ok(native.levels[0].every(x=>x===23));
    assert.equal(d.textureWrites.length,2);assert.equal(d.textures.length,1);
    assert.ok(d.snapshots.at(-1)[0].textureContents.has(native));
    await bridge.whenIdle();
  }finally{bridge.dispose();}
  assert.ok(d.textures.every(t=>t.destroyed),'scene disposal retires every texture');
 });

test('BatchedMesh allocates and uploads its first indirect ids through shared residency',async()=>{
  const d=gpu(),root=new T.Scene(),material=new T.MeshBasicMaterial();
  const object=new T.BatchedMesh(4,8,12,material),id=object.addGeometry(new T.PlaneGeometry());
  object.addInstance(id);object.addInstance(id);root.add(object);
  object.perObjectFrustumCulled=false;object.sortObjects=false;
  const camera=new T.PerspectiveCamera(45,1,.1,10);camera.position.z=3;
  const program=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const bridge=await createGpuThreeScene(d,root,{three:T,program});
  try{
    assert.equal(object._indirectTexture.version,0,'preparation must not invoke onBeforeRender or mark the source dirty');
    bridge.render(camera,{colorView:{},depthView:{}});
    assert.equal(bridge.diagnostics.textures.resources,2);
    const ids=d.textures.find(t=>t.desc.format==='r32uint');
    assert.deepEqual([...new Uint32Array(ids.levels[0].buffer)].slice(0,2),[0,1]);
    assert.equal(d.snapshots.at(-1).length,2);
    assert.deepEqual(d.snapshots.at(-1)[0].textureContents.get(ids)[0],ids.levels[0]);
    object.setVisibleAt(0,false);bridge.render(camera,{colorView:{},depthView:{}});
    assert.equal(new Uint32Array(ids.levels[0].buffer)[0],1);
    assert.equal(d.snapshots.at(-1).length,1);
  }finally{bridge.dispose();}
  assert.ok(d.textures.every(t=>t.destroyed));
});

test('program-generated morph arrays enter the scene owner and retire on scene disposal',async()=>{
  const d=gpu(),root=new T.Scene(),geometry=new T.PlaneGeometry();
  geometry.morphAttributes.position=[geometry.attributes.position.clone()];
  const material=shader(null,'array');material.uniforms={};
  material.fragmentShader=material.fragmentShader.replaceAll('dataMap','morphTargetsTexture');
  const object=new T.Mesh(geometry,material);root.add(object);
  const camera=new T.PerspectiveCamera(45,1,.1,10);camera.position.z=3;
  const program=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const bridge=await createGpuThreeScene(d,root,{three:T,program});
  try{
    assert.equal(bridge.diagnostics.textures.resources,1);
    assert.equal(d.textures[0].desc.format,'rgba32float');
    assert.equal(d.textures[0].desc.size[2],1);
    bridge.render(camera,{colorView:{},depthView:{}});assert.equal(d.textureWrites.length,1);
    assert.ok(d.snapshots.at(-1)[0].textureContents.has(d.textures[0]));
  }finally{bridge.dispose();}
  assert.ok(d.textures.every(t=>t.destroyed));
});

for(const kind of ['array','volume'])for(const integer of [false,true,'signed'])
 test(`${kind}/${integer}: typed loading defaults rebind after source arrival`,async()=>{
  const t=source(kind,integer);t.image.data=null;t.version=0;
  const s=await scene(t,kind,integer),{d,bridge}=s;
  try{
    s.render();assert.equal(d.textureWrites.length,0);
    assert.equal(d.textures[0].desc.format,integer==='signed'?'rgba8sint':integer?'rgba8uint':'rgba8unorm');
    assert.equal(d.snapshots.at(-1)[0].groups.get(1).group.entries[0].resource.dimension,kind==='array'?'2d-array':'3d');
    t.image.data=source(kind,integer).image.data;t.needsUpdate=true;
    assert.throws(()=>s.render(),{code:'THREE_SCENE_PREPARE'});
    await bridge.prepare();s.render();assert.equal(bridge.diagnostics.textures.resources,1);
    assert.equal(d.textureWrites.length,1);
  }finally{bridge.dispose();}
  assert.ok(d.textures.every(t=>t.destroyed));
 });

for(const kind of ['2d','array','volume'])test(`${kind}: a null integer sampler gets matching storage without creating a texture owner`,async()=>{
  const s=await scene(null,kind,true);
  try{
    s.render();assert.equal(s.bridge.diagnostics.textures,null);
    assert.equal(s.d.textures[0].desc.format,'rgba8uint');assert.equal(s.d.textureWrites.length,0);
    const group=s.d.snapshots.at(-1)[0].groups.get(1).group;
    const layout=group.layout.entries.find(e=>e.texture).texture;
    assert.equal(layout.viewDimension,kind==='array'?'2d-array':kind==='volume'?'3d':'2d');
    assert.equal(layout.sampleType,'uint');
  }finally{s.bridge.dispose();}
  assert.ok(s.d.textures.every(t=>t.destroyed));
});

test('scene sampling changes require preparation and old-plus-new texture capacity',async()=>{
  const t=source('array'),s=await scene(t,'array',false,{texture:{maxTextureBytes:128}});
  try{
    s.render();const old=s.d.textures[0];t.wrapR=T.RepeatWrapping;
    assert.throws(()=>s.render(),{code:'THREE_SCENE_PREPARE'});
    await s.bridge.prepare();s.render();assert.ok(old.destroyed);
    assert.equal(s.bridge.diagnostics.textures.textureBytes,32);
    t.dispose();assert.throws(()=>s.render(),{code:'THREE_SCENE_PREPARE'});
    await s.bridge.prepare();s.render();
  }finally{s.bridge.dispose();}
});

test('scene array updates preserve unselected layers and source aliases share one allocation',async()=>{
  const t=source('array',true),s=await scene(t,'array',true);
  try{
    const clone=t.clone(),m=shader(clone,'array',true);s.root.add(new T.Mesh(s.object.geometry,m));
    await s.bridge.prepare();s.render();assert.equal(s.d.textures.length,1);
    assert.equal(s.bridge.diagnostics.textures.textures,2);
    const n=s.d.textures[0],before=n.levels[0].slice(),half=before.length/2,writes=s.d.textureWrites.length;
    t.image.data.fill(0xffffffff);t.addLayerUpdate(1);t.needsUpdate=true;s.render();
    assert.deepEqual(n.levels[0].slice(0,half),before.slice(0,half));
    assert.ok(n.levels[0].slice(half).every(v=>v===255));assert.equal(t.layerUpdates.size,0);
    assert.equal(s.d.textureWrites.length,writes+1);
    s.root.remove(s.object);await s.bridge.prepare();assert.equal(n.destroyed,false);
    assert.equal(s.bridge.diagnostics.textures.textures,1);
    s.root.clear();await s.bridge.prepare();assert.ok(n.destroyed);
  }finally{s.bridge.dispose();}
});

for(const kind of ['array','volume'])test(`${kind}: scene-generated mip chains submit before their consuming draw`,async()=>{
  const t=source(kind);t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;
  const s=await scene(t,kind);
  try{
    assert.equal(s.d.textures[0].desc.mipLevelCount,2);assert.ok(s.bridge.diagnostics.textures.mipPasses>0);
    t.needsUpdate=true;s.render();const submissions=s.d.snapshots;
    assert.equal(submissions.at(-1).length,1);
    assert.equal(submissions.at(-2).length,kind==='array'?2:1);
    assert.equal(submissions.at(-1)[0].groups.get(1).group.entries[0].resource.texture,s.d.textures[0]);
  }finally{s.bridge.dispose();}
});

test('integer sample-type and dimensional mismatches fail before source allocations',async()=>{
  for(const [textureKind,samplerKind,textureInteger,samplerInteger] of [
    ['array','volume',false,false],['volume','array',false,false],['2d','array',true,true],
    ['array','array',true,false],['array','array',false,true],['array','array','signed',true]]){
    const d=gpu();await assert.rejects(scene(source(textureKind,textureInteger),samplerKind,samplerInteger,{device:d}),{code:'THREE_SCENE_TEXTURE'});
    assert.equal(d.textures.length,0);assert.equal(d.buffers.length,0);
  }
});

test('automatic integer and volume allocation obey scene budgets and autoTextures',async()=>{
  for(const kind of ['array','volume'])for(const opts of [{autoTextures:false},{texture:{maxTextureBytes:127}}]){
    const d=gpu();await assert.rejects(scene(source(kind,true),kind,true,{device:d,...opts}));
    assert.equal(d.textures.length,0);assert.ok(d.buffers.every(b=>b.destroyed));
  }
});

test('source-version-zero initialization reserves storage without uploading or changing source versions',async()=>{
  const d=gpu(),p=createGpuThreeTextures(d,{three:T}),t=new T.DataTexture(new Uint32Array([0xffffffff]),1,1,T.RedIntegerFormat,T.UnsignedIntType);
  try{
    assert.throws(()=>p.prepare([t]),{code:'THREE_TEXTURE_NOT_READY'});
    p.prepare([t],{initialize:true});assert.equal(p.binding(t).version,0);assert.equal(p.binding(t).sourceVersion,-1);
    assert.equal(t.version,0);assert.equal(t.source.version,0);assert.equal(d.textureWrites.length,0);
    assert.ok(p.nativeTexture(t).levels[0].every(v=>v===0));p.update([t]);assert.equal(d.textureWrites.length,0);
    t.needsUpdate=true;p.update([t]);assert.equal(d.textureWrites.length,1);
    assert.ok(p.nativeTexture(t).levels[0].every(v=>v===255));await p.whenIdle();
  }finally{p.dispose();}
});

test('native texture failures release the scene and its generated/shared resources',async()=>{
  const s=await scene(source('array',true),'array',true);
  s.d.lose();await assert.rejects(s.bridge.whenIdle());assert.equal(s.bridge.failed,true);
  assert.ok(s.d.textures.every(t=>t.destroyed));assert.ok(s.d.buffers.every(b=>b.destroyed));s.bridge.dispose();
});

test('ordinary complete version-zero textures stay unrequested, unlike explicitly initialized batching ids',async()=>{
  const t=new T.DataArrayTexture(new Uint32Array(32).fill(0xffffffff),2,2,2);
  t.type=T.UnsignedIntType;t.format=T.RGBAIntegerFormat;
  const s=await scene(t,'array',true);
  try{
    s.render();assert.equal(s.d.textureWrites.length,0);assert.equal(t.version,0);
    assert.equal(s.bridge.diagnostics.textures.resources,0);
    assert.ok(s.d.textures[0].levels[0].every(v=>v===0));
    t.needsUpdate=true;assert.throws(()=>s.render(),{code:'THREE_SCENE_PREPARE'});
    await s.bridge.prepare();s.render();assert.equal(s.d.textureWrites.length,1);
  }finally{s.bridge.dispose();}
});

test('a storage change during pipeline preparation does not publish stale scene bindings',async()=>{
  const s=await scene(source('array'),'array');
  try{
    const t=source('array');s.material.uniforms.dataMap.value=t;
    const create=s.d.createRenderPipelineAsync;let resume;
    const waiting=new Promise(resolve=>resume=resolve);
    s.d.createRenderPipelineAsync=async p=>{await waiting;return create(p);};
    const preparing=s.bridge.prepare();t.wrapS=T.RepeatWrapping;resume();
    await assert.rejects(preparing,{code:'THREE_SCENE_PREPARE'});
    s.d.createRenderPipelineAsync=create;await s.bridge.prepare();s.render();
    assert.equal(s.bridge.failed,false);
  }finally{s.bridge.dispose();}
});
