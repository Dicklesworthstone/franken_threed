import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {decodeGltfAnimationModel,prepareGltfAnimationModel,createCpuGltfAnimationModel} from './animation_model.mjs';
import {createAnimationPlayer} from './animation_runtime.mjs';
const EXT='EXT_mesh_gpu_instancing',UV=[0,0,1,0,0,1];
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-5,`${x} != ${b[i]}`));};
const texture=()=>({view:{},sampler:{}});
function fixture({morph=false,textured=false}={}) {
  const model={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0,translation:[10,0,0],scale:[2,3,1]}],
    meshes:[{primitives:[{attributes:{},material:0}]}],materials:[{}],accessors:[],bufferViews:[],buffers:[],
    extensionsUsed:[EXT],extensionsRequired:[EXT]},buffers=[];
  function attr(values,type='VEC3') {
    const a=new Float32Array(values),buffer=buffers.push(a)-1;model.buffers.push({byteLength:a.byteLength});
    const bufferView=model.bufferViews.push({buffer,byteLength:a.byteLength})-1;
    return model.accessors.push({bufferView,componentType:5126,type,count:values.length/({VEC3:3,VEC4:4,VEC2:2,SCALAR:1}[type])})-1;
  }
  const p=model.meshes[0].primitives[0];p.attributes.POSITION=attr([0,0,0,1,0,0,0,1,0]);Object.assign(model.accessors[0],{min:[0,0,0],max:[1,1,0]});
  p.attributes.NORMAL=attr([0,0,1,0,0,1,0,0,1]);p.attributes.TEXCOORD_0=attr(UV,'VEC2');
  const translation=attr([1,0,0,0,2,0]);
  model.nodes[0].extensions={[EXT]:{attributes:{TRANSLATION:translation}}};
  const times=attr([0,1],'SCALAR');Object.assign(model.accessors[times],{min:[0],max:[1]});
  model.animations=[{samplers:[{input:times,output:attr([10,0,0,20,0,0])}],channels:[{sampler:0,target:{node:0,path:'translation'}}]}];
  if(morph){p.targets=[{POSITION:attr([0,0,2,0,0,2,0,0,2])}];model.nodes[0].weights=[0.25];
    model.animations[0].samplers.push({input:times,output:attr([0,1],'SCALAR')});model.animations[0].channels.push({sampler:1,target:{node:0,path:'weights'}});}
  if(textured){model.textures=[{source:0}];model.images=[{uri:'orm.png'}];model.materials[0]={
    pbrMetallicRoughness:{baseColorTexture:{index:0},metallicRoughnessTexture:{index:0}},
    occlusionTexture:{index:0,strength:0.25},emissiveFactor:[0.5,0.25,0.1],extensions:{KHR_materials_emissive_strength:{emissiveStrength:4}}};}
  return {model,buffers,attr,p,translation};
}
const decode=(f,options={})=>decodeGltfAnimationModel(f.model,f.buffers,options);
const translation=(pose,node)=>Array.from(pose.worldMatrices.subarray(node*16+12,node*16+15));

test('ordinary model entry expands required TRS instances into drawable nodes and explicit origin metadata',()=>{
  const f=fixture(),before=structuredClone(f.model),d=decode(f),pose=createAnimationPlayer(d.definition);
  assert.deepEqual(f.model,before);assert.equal(d.drawables.length,2);assert.equal(pose.nodeCount,3);
  assert.deepEqual(d.source.map(s=>s.node),[1,2]);assert.deepEqual(d.instanceOrigins,{1:{node:0,instance:0},2:{node:0,instance:1}});
  close(translation(pose,1),[12,0,0]);close(translation(pose,2),[10,6,0]);
  assert.ok(d.diagnostics.some(x=>x.reason==='EXPANDED_INSTANCE_DRAWS_NOT_GPU_INSTANCING'));assert.equal(d.accelerationClaim,false);pose.dispose();
});
test('animated parent transform, instance transform and morph deformation execute together on real CPU meshes',()=>{
  const f=fixture({morph:true}),m=createCpuGltfAnimationModel(f.model,f.buffers),a=m.deformers[0],b=m.deformers[1];
  close(a.positions,[0,0,0.5,1,0,0.5,0,1,0.5]);assert.notEqual(a.positions,b.positions);
  assert.equal(m.instanceOrigins[a.node].instance,0);m.sample(0.5);
  close(a.positions,[0,0,1,1,0,1,0,1,1]);close(b.positions,a.positions);
  close(a.worldMatrix.slice(12,15),[17,0,0]);close(b.worldMatrix.slice(12,15),[15,6,0]);
  assert.equal(a.poseVersion,b.poseVersion);assert.equal(a.poseVersion,m.pose.version);
  m.reset();close(a.positions,[0,0,0.5,1,0,0.5,0,1,0.5]);m.dispose();assert.ok(a.disposed&&b.disposed&&m.pose.disposed);
});
test('CPU instance data is snapshotted before texture callbacks can mutate JSON and binary arrays',()=>{
  const f=fixture({morph:true,textured:true}),m=createCpuGltfAnimationModel(f.model,f.buffers,{resolveTexture:()=>{
    f.buffers.forEach(b=>b.fill(0));f.model.nodes[0].translation.fill(999);return texture();
  }});
  m.sample(1);close(m.deformers[0].worldMatrix.slice(12,15),[22,0,0]);close(m.deformers[1].positions,[0,0,2,1,0,2,0,1,2]);m.dispose();
});
test('instances share texture resolution, but own geometry, material arrays and per-instance UV storage',()=>{
  const f=fixture({textured:true}),requests=[],d=decode(f,{resolveTexture:r=>{requests.push(r);return texture();}});
  assert.deepEqual(requests.map(r=>r.colorSpace),['srgb','linear']);
  const [a,b]=d.drawables;assert.equal(a.baseColorTexture,b.baseColorTexture);assert.equal(a.occlusionTexture,b.metallicRoughnessTexture);
  assert.notEqual(a.geometry.positions,b.geometry.positions);assert.notEqual(a.texCoords,b.texCoords);assert.notEqual(a.baseColor,b.baseColor);
  assert.equal(a.occlusionStrength,0.25);close(a.emissiveFactor,[2,1,0.4]);
});
test('prepared instance plans survive a delayed texture stage with no re-decode or origin mutation',()=>{
  const f=fixture({textured:true}),p=prepareGltfAnimationModel(f.model,f.buffers);const origins=p.instanceOrigins;
  f.buffers.forEach(b=>b.fill(99));f.model.nodes[0].extensions[EXT].attributes.TRANSLATION=999;
  assert.throws(()=>{origins[1].instance=999;},TypeError);
  const d=p.resolveTextures(texture);assert.equal(d.instanceOrigins,origins);assert.equal(d.drawables.length,2);
  const pose=createAnimationPlayer(d.definition);close(translation(pose,2),[10,6,0]);pose.dispose();
});
test('instancing, geometry and animation share one buffer-provider cache',()=>{
  const f=fixture({morph:true}),reads=new Map();
  decodeGltfAnimationModel(f.model,i=>{reads.set(i,(reads.get(i)??0)+1);return f.buffers[i];});
  assert.ok(reads.size>0);assert.ok([...reads.values()].every(n=>n===1));
});
test('one camera and light on the batch remain one instance of each, tracking only the original parent',()=>{
  const f=fixture();f.model.nodes[0].camera=0;f.model.cameras=[{type:'perspective',perspective:{yfov:1,znear:0.1}}];
  f.model.nodes[0].extensions.KHR_lights_punctual={light:0};f.model.extensions={KHR_lights_punctual:{lights:[{type:'point'}]}};
  const m=createCpuGltfAnimationModel(f.model,f.buffers);assert.equal(m.cameras.length,1);assert.equal(m.lights.length,1);
  m.sample(0.5);const view=m.view.sample({aspectRatio:1});close(view.cameraPosition,[15,0,0]);close(view.lighting.lights[0].position,[15,0,0]);
  assert.equal(m.cameras[0].node,0);assert.equal(m.lights[0].node,0);m.dispose();
});
test('other scenes stay unrendered and ordinary sibling meshes are not lost or duplicated',()=>{
  const f=fixture();f.model.nodes.push({mesh:0,translation:[100,0,0]},{mesh:0});f.model.nodes[0].children=[1];f.model.scenes.push({nodes:[2]});
  const a=decode(f);assert.deepEqual(a.source.map(s=>s.node),[1,3,4]);
  const b=decode(f,{scene:1});assert.deepEqual(b.source.map(s=>s.node),[2]);
});
for(const [name,options] of [['instance',{maxInstances:1}],['primitive',{maxPrimitives:1}],['geometry component',{maxComponents:53}]])test(`aggregate ${name} budget fails before texture resolution`,()=>{
  const f=fixture({textured:true});let calls=0;
  assert.throws(()=>decode(f,{...options,resolveTexture:()=>{calls++;return texture();}}));assert.equal(calls,0);
});
test('invalid later instance material aborts the complete model before any texture callbacks',()=>{
  const f=fixture({textured:true});f.model.meshes[0].primitives.push({...f.p,material:1});f.model.materials.push({alphaMode:'INVALID'});let calls=0;
  assert.throws(()=>decode(f,{resolveTexture:()=>{calls++;return texture();}}));assert.equal(calls,0);
});
test('optional instancing is used too; unknown required and custom-instance behavior is not silently dropped',()=>{
  const f=fixture();delete f.model.extensionsRequired;assert.equal(decode(f).drawables.length,2);
  f.model.extensionsRequired=['EXT_unknown'];assert.throws(()=>decode(f),{code:'GLTF_MODEL_UNSUPPORTED'});
  f.model.extensionsRequired=[EXT];f.model.nodes[0].extensions[EXT].attributes._ID=f.translation;
  assert.throws(()=>decode(f),{code:'GLTF_INSTANCING_UNSUPPORTED'});
});
test('ordinary models keep their existing decoded shape without new metadata or diagnostics',()=>{
  const f=fixture();delete f.model.nodes[0].extensions;delete f.model.extensionsRequired;
  const d=decode(f);assert.equal(d.drawables.length,1);assert.equal(Object.hasOwn(d,'instanceOrigins'),false);
  assert.ok(!d.diagnostics.some(x=>x.reason.includes('INSTANCE')));
});

// Only replace GPU scene execution: model decoding, pose and factory are real.
// This is not a shader, device-limit, pixel or GPU-throughput test.
const encoded=s=>'data:text/javascript;base64,'+Buffer.from(s).toString('base64');
let gpuSource=readFileSync(new URL('./animation_model_gpu.mjs',import.meta.url),'utf8');
for(const name of ['animation_model.mjs','animation_runtime.mjs','animation_scene.mjs'])gpuSource=gpuSource.replace("'./"+name+"'",JSON.stringify(name==='animation_scene.mjs'?
  encoded('export async function createGpuAnimationScene(device,pose,drawables,options){return device.createScene(pose,drawables,options);}'):
  new URL('./'+name,import.meta.url).href));
const {createGpuGltfAnimationScene}=await import(encoded(gpuSource));
function device() {
  let pose,drawables,frames=[];
  const scene={controller:{},draws:[],deformers:[],poseVersion:0,bufferBytes:0,disposed:false,failed:false,
    update(t){pose.sample(t);scene.poseVersion=pose.version;},upload(){scene.poseVersion=pose.version;},render(f){frames.push(f);},whenIdle:async()=>{},dispose(){scene.disposed=true;}};
  return {scene,get pose(){return pose;},get drawables(){return drawables;},frames,
    createScene(p,d){pose=p;drawables=d;scene.draws=d.map(()=>({}));return scene;}};
}
test('GPU factory consumes expanded drawables and exposes source-instance lookup without owning borrowed textures',async()=>{
  const f=fixture({textured:true}),g=device(),borrowed=texture();
  const m=await createGpuGltfAnimationScene(g,f.model,f.buffers,{decode:{resolveTexture:()=>borrowed}});
  assert.equal(g.drawables.length,2);assert.equal(m.draws.length,2);assert.deepEqual(m.instanceOrigins,{1:{node:0,instance:0},2:{node:0,instance:1}});
  m.update(0.5);close(translation(m.pose,1),[17,0,0]);m.render({colorView:{}});assert.equal(g.frames.length,1);
  await m.whenIdle();m.dispose();assert.ok(g.scene.disposed&&g.pose.disposed);assert.ok(borrowed.view);
});
test('GPU instance budget errors cannot start scene allocation',async()=>{
  const f=fixture();let calls=0;
  await assert.rejects(createGpuGltfAnimationScene({createScene(){calls++;}},f.model,f.buffers,{decode:{maxInstances:1}}),{code:'GLTF_INSTANCING_LIMIT'});assert.equal(calls,0);
});

// Owning-loader orchestration uses real model preparation and the real GPU
// factory above. Replace only asset delivery, texture upload, and scene GPU work.
let loaderSource=readFileSync(new URL('./gltf_scene_loader.mjs',import.meta.url),'utf8');
for(const [name,replacement] of [
  ['gltf_asset.mjs',encoded('export class GltfAssetError extends Error {} export async function loadGltfAsset(source){return source;}')],
  ['gltf_textures.mjs',encoded('export class GltfTextureError extends Error {} export async function createGltfTextureResources(device,requests){return device.createTextures(requests);}')],
  ['animation_model.mjs',new URL('./animation_model.mjs',import.meta.url).href],
  ['animation_model_gpu.mjs',encoded(gpuSource)],
])loaderSource=loaderSource.replace("'./"+name+"'",JSON.stringify(replacement));
const {loadGpuGltfAnimationScene}=await import(encoded(loaderSource));
test('owning loader retains instance origins across texture preparation and model disposal',async()=>{
  const f=fixture({textured:true}),g=device();let requests,disposed=false;
  g.createTextures=r=>{requests=r;const maps=r.map(texture);return {textureBytes:8,failed:false,
    resolveTexture:request=>maps[r.findIndex(x=>x.textureIndex===request.textureIndex&&x.colorSpace===request.colorSpace)],dispose(){disposed=true;}};};
  const m=await loadGpuGltfAnimationScene(g,{json:f.model,buffers:f.buffers,bytesLoaded:42});
  assert.equal(requests.length,2);assert.equal(m.assetBytes,42);assert.equal(m.textureBytes,8);
  assert.deepEqual(m.instanceOrigins,{1:{node:0,instance:0},2:{node:0,instance:1}});m.update(0.5);
  close(translation(m.pose,2),[15,6,0]);m.dispose();assert.ok(disposed&&m.disposed&&m.pose.disposed);
});
test('owning loader forwards instance budget before texture creation or scene initialization',async()=>{
  const f=fixture({textured:true}),g=device();let calls=0;g.createTextures=()=>{calls++;};
  await assert.rejects(loadGpuGltfAnimationScene(g,{json:f.model,buffers:f.buffers},{decode:{maxInstances:1}}),{code:'GLTF_INSTANCING_LIMIT'});
  assert.equal(calls,0);assert.equal(g.pose,undefined);
});
