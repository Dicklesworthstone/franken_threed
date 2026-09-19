import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {decodeGltfAnimationModel,createCpuGltfAnimationModel} from './animation_model.mjs';
const code = expected => error => error.code===expected;
const close = (a,b) => {assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-5,`${i}: ${v} != ${b[i]}`));};
function fixture({skin=false,morph=false}={}) {
  const model={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0,1,2]}],nodes:[{mesh:0},{translation:[2,0,0]},{translation:[0,4,0]}],
    meshes:[{primitives:[{attributes:{},material:0}]}],materials:[{pbrMetallicRoughness:{baseColorFactor:[0.2,0.3,0.4,0.5],metallicFactor:0.25,roughnessFactor:0.75},emissiveFactor:[0.1,0.2,0.3],alphaMode:'BLEND',doubleSided:true}],
    buffers:[],bufferViews:[],accessors:[],animations:[]},buffers=[];
  function attr(values,type='VEC3') {
    const width={SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[type],data=new Float32Array(values),buffer=buffers.push(data)-1;
    model.buffers.push({byteLength:data.byteLength});const view=model.bufferViews.push({buffer,byteLength:data.byteLength})-1;
    return model.accessors.push({bufferView:view,componentType:5126,type,count:values.length/width})-1;
  }
  const p=model.meshes[0].primitives[0];
  p.attributes.POSITION=attr([0,0,0,1,0,0,0,1,0]);model.accessors[0].min=[0,0,0];model.accessors[0].max=[1,1,0];
  p.attributes.NORMAL=attr([0,0,1,0,0,1,0,0,1]);p.attributes.TANGENT=attr([1,0,0,-1,1,0,0,-1,1,0,0,-1],'VEC4');
  p.attributes.TEXCOORD_0=attr([0,0,1,0,0,1],'VEC2');p.attributes.TEXCOORD_1=attr([0.25,0.25,0.75,0.25,0.25,0.75],'VEC2');
  p.attributes.COLOR_0=attr([1,0.5,0.25,1,0.5,0.25,1,0.5,0.25]);
  const time=attr([0,1],'SCALAR');model.accessors[time].min=[0];model.accessors[time].max=[1];
  const translations=attr([2,0,0,6,0,0]);
  model.animations=[{samplers:[{input:time,output:translations}],channels:[{sampler:0,target:{node:skin?1:0,path:'translation'}}]}];
  if(morph) {
    p.targets=[{POSITION:attr([0,0,2,0,0,2,0,0,2])}];model.meshes[0].weights=[0.25];
    model.animations[0].samplers.push({input:time,output:attr([0,1],'SCALAR')});
    model.animations[0].channels.push({sampler:1,target:{node:0,path:'weights'}});
  }
  if(skin) {
    model.nodes[0].skin=0;model.skins=[{joints:[1,2]}];
    for(let set=0;set<2;set++) {
      const values=new Uint8Array([set,0,0,0,set,0,0,0,set,0,0,0]),buffer=buffers.push(values)-1;
      model.buffers.push({byteLength:values.byteLength});const bufferView=model.bufferViews.push({buffer,byteLength:values.byteLength})-1;
      p.attributes['JOINTS_'+set]=model.accessors.push({bufferView,componentType:5121,type:'VEC4',count:3})-1;
      p.attributes['WEIGHTS_'+set]=attr([0.5,0,0,0,0.5,0,0,0,0.5,0,0,0],'VEC4');
    }
  }
  return {model,buffers,p,material:model.materials[0],attr};
}
function maps(f) {
  f.model.images=[{uri:'model.png'}];f.model.samplers=[{wrapS:33071,wrapT:33648,magFilter:9728,minFilter:9987}];
  f.model.textures=[{source:0,sampler:0}];
  f.material.pbrMetallicRoughness.baseColorTexture={index:0};f.material.pbrMetallicRoughness.metallicRoughnessTexture={index:0};
  f.material.normalTexture={index:0,scale:0.25};f.material.emissiveTexture={index:0};return f;
}
const resources=()=>({view:{},sampler:{}});

test('decoded core materials use PBR defaults, alpha, colors and original source IDs',()=>{
  const f=fixture(),r=decodeGltfAnimationModel(f.model,f.buffers),a=r.drawables[0];
  assert.equal(a.shading,'metallic-roughness');assert.deepEqual(a.baseColor,[0.2,0.3,0.4,0.5]);assert.equal(a.metallicFactor,0.25);assert.equal(a.roughnessFactor,0.75);
  assert.equal(a.alphaMode,'BLEND');assert.equal(a.doubleSided,true);close([...a.vertexColors],[1,0.5,0.25,1,0.5,0.25,1,0.5,0.25]);
  assert.deepEqual(r.source,[{node:0,mesh:0,primitive:0,material:0}]);
  delete f.p.material;const defaults=decodeGltfAnimationModel(f.model,f.buffers).drawables[0];
  assert.equal(defaults.metallicFactor,1);assert.equal(defaults.roughnessFactor,1);assert.equal(defaults.alphaMode,'OPAQUE');
});

test('real CPU pose -> morph -> eight-influence skinning produces independently known vertices',()=>{
  const f=fixture({skin:true,morph:true}),model=createCpuGltfAnimationModel(f.model,f.buffers),mesh=model.deformers[0];
  close([...mesh.positions],[1,2,0.5,2,2,0.5,1,3,0.5]);
  const positions=mesh.positions,world=mesh.worldMatrix;
  assert.equal(model.sample(0.5),model);close([...mesh.positions],[2,2,1,3,2,1,2,3,1]);
  assert.equal(mesh.positions,positions);assert.equal(mesh.worldMatrix,world);assert.equal(mesh.poseVersion,model.pose.version);
  assert.equal(mesh.tangents[3],-1);model.reset();close([...mesh.positions],[1,2,0.5,2,2,0.5,1,3,0.5]);
  model.dispose();assert.ok(model.pose.disposed);assert.ok(mesh.disposed);
});

test('the CPU model snapshots source JSON and binary data before playback',()=>{
  const f=fixture({skin:true,morph:true}),model=createCpuGltfAnimationModel(f.model,f.buffers);
  f.buffers.forEach(a=>a.fill(0));f.model.nodes[1].translation[0]=999;f.material.pbrMetallicRoughness.baseColorFactor.fill(99);
  model.sample(1);close([...model.deformers[0].positions],[3,2,2,4,2,2,3,3,2]);
  assert.deepEqual(model.drawables[0].baseColor,[0.2,0.3,0.4,0.5]);model.dispose();
});

test('shared model meshes have distinct world matrices and synchronize through one pose',()=>{
  const f=fixture();f.model.nodes[1]={mesh:0,translation:[10,0,0]};
  const m=createCpuGltfAnimationModel(f.model,f.buffers);assert.equal(m.deformers.length,2);
  m.sample(0.5);assert.equal(m.deformers[0].worldMatrix[12],4);assert.equal(m.deformers[1].worldMatrix[12],10);
  assert.equal(m.deformers[0].poseVersion,m.deformers[1].poseVersion);
  m.pose.sample(1);m.update();assert.equal(m.deformers[0].worldMatrix[12],6);m.dispose();
});

test('invalid pose sample preserves model outputs; a deformer failure terminates every child',()=>{
  const f=fixture(),m=createCpuGltfAnimationModel(f.model,f.buffers),before=m.deformers[0].positions.slice();
  assert.throws(()=>m.sample(NaN));assert.equal(m.failed,false);assert.deepEqual(m.deformers[0].positions,before);m.sample(0.5);
  structuredClone(m.deformers[0].positions.buffer,{transfer:[m.deformers[0].positions.buffer]});
  assert.throws(()=>m.update(),code('ANIMATION_DEFORM_STORAGE'));assert.equal(m.failed,true);assert.equal(m.pose.disposed,true);
  assert.throws(()=>m.sample(0.5));m.dispose();m.dispose();assert.equal(m.disposed,true);
});

test('reentrant pose-option getters cannot dispose an updating CPU model',()=>{
  const f=fixture(),m=createCpuGltfAnimationModel(f.model,f.buffers);
  assert.throws(()=>m.sample(0,{get rootMatrix(){m.dispose();return null;}}),code('GLTF_MODEL_REENTRANT'));
  assert.equal(m.disposed,false);assert.equal(m.pose.disposed,false);m.sample(0.5);m.dispose();
});

test('four material maps resolve by texture AND color space and preserve sampler/image metadata',()=>{
  const f=maps(fixture()),requests=[],borrowed=[];
  const r=decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:request=>{requests.push(request);const b=resources();borrowed.push(b);return b;}}),d=r.drawables[0];
  assert.equal(requests.length,2);assert.deepEqual(requests.map(r=>r.colorSpace),['srgb','linear']);
  assert.equal(d.baseColorTexture,d.emissiveTexture);assert.equal(d.normalTexture,d.metallicRoughnessTexture);assert.notEqual(d.baseColorTexture,d.normalTexture);
  assert.equal(d.baseColorTexture.view,borrowed[0].view);assert.equal(d.normalScale,0.25);
  assert.deepEqual(requests[0].image,{uri:'model.png'});
  assert.deepEqual(requests[0].sampler,{wrapS:33071,wrapT:33648,magFilter:9728,minFilter:9987});
  assert.ok(Object.isFrozen(requests[0])&&Object.isFrozen(requests[0].image)&&Object.isFrozen(requests[0].sampler));
});

test('distinct glTF texture objects sharing one image retain distinct sampler resolutions',()=>{
  const f=maps(fixture());f.model.samplers.push({minFilter:9728});f.model.textures.push({source:0,sampler:1});f.material.emissiveTexture.index=1;
  const requests=[];decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:r=>{requests.push(r);return resources();}});
  assert.deepEqual(requests.map(r=>[r.textureIndex,r.colorSpace]),[[0,'srgb'],[0,'linear'],[1,'srgb']]);
  assert.equal(requests[2].sampler.minFilter,9728);
});

test('all material metadata is snapshotted before a texture resolver can mutate the JSON',()=>{
  const f=maps(fixture());f.model.meshes[0].primitives.push(structuredClone(f.p));
  const r=decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:request=>{
    f.material.pbrMetallicRoughness.baseColorFactor.fill(99);f.material.normalTexture.scale=99;f.model.images[0].uri='changed.png';
    assert.equal(request.image.uri,'model.png');return resources();
  }});
  for(const d of r.drawables){assert.deepEqual(d.baseColor,[0.2,0.3,0.4,0.5]);assert.equal(d.normalScale,0.25);}
});

test('KHR_texture_transform uses T*R*S and the extension UV set override',()=>{
  const f=maps(fixture()),transform={offset:[0.25,0.5],rotation:Math.PI/2,scale:[2,3],texCoord:1};
  for(const info of [f.material.pbrMetallicRoughness.baseColorTexture,f.material.pbrMetallicRoughness.metallicRoughnessTexture,f.material.normalTexture,f.material.emissiveTexture])info.extensions={KHR_texture_transform:structuredClone(transform)};
  const d=decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:resources}).drawables[0];
  close(d.uvTransform,[0,2,-3,0,0.25,0.5]);close([...d.texCoords],[0.25,0.25,0.75,0.25,0.25,0.75]);
});

test('unlit extension uses base color and alpha without loading ignored PBR fallback maps',()=>{
  const f=maps(fixture());f.material.extensions={KHR_materials_unlit:{}};f.material.occlusionTexture={index:999};
  f.material.normalTexture.index=999;f.material.emissiveTexture.index=999;f.material.pbrMetallicRoughness.metallicRoughnessTexture.index=999;
  const requests=[],d=decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:r=>{requests.push(r);return resources();}}).drawables[0];
  assert.equal(d.shading,'unlit');assert.equal(requests.length,1);assert.equal(requests[0].colorSpace,'srgb');
  assert.equal(d.normalTexture,undefined);assert.equal(d.emissiveFactor,undefined);assert.equal(d.metallicFactor,undefined);
  assert.equal(d.alphaMode,'BLEND');assert.equal(d.doubleSided,true);
});

test('embedded image requests carry validated bufferView offsets, length and MIME without fetching',()=>{
  const f=maps(fixture());f.model.images[0]={bufferView:0,mimeType:'image/png'};
  const requests=[];decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:r=>{requests.push(r);return resources();}});
  assert.deepEqual(requests[0].image,{bufferView:0,buffer:0,byteOffset:0,byteLength:36,mimeType:'image/png'});
});

for(const [name,mutate]of [
  ['missing alternate UV',f=>{f.material.normalTexture.texCoord=9;}],
  ['invalid UV transform',f=>{f.material.emissiveTexture.extensions={KHR_texture_transform:{offset:[Infinity,0]}};}],
  ['missing UV',f=>{delete f.p.attributes.TEXCOORD_0;}],
  ['invalid tangent',f=>{f.model.accessors[f.p.attributes.TANGENT].type='VEC3';}],
  ['invalid occlusion strength',f=>{f.material.occlusionTexture={index:0,strength:2};}],
  ['material extension',f=>{f.material.extensions={KHR_materials_clearcoat:{}};}],
  ['texture codec without fallback',f=>{delete f.model.textures[0].source;f.model.textures[0].extensions={KHR_texture_basisu:{source:0}};}],
  ['required extension',f=>{f.model.extensionsRequired=['EXT_unknown'];}],
  ['sampler',f=>{f.model.samplers[0].wrapS=123;}],
  ['bad later material',f=>{f.model.materials.push({alphaMode:'INVALID'});f.model.meshes[0].primitives.push({...f.p,material:1});}],
])test(`${name} fails before any resolver or GPU work`,()=>{
  const f=maps(fixture());mutate(f);let calls=0;
  assert.throws(()=>decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:()=>{calls++;return resources();}}));assert.equal(calls,0);
});

test('missing, async or malformed texture resolutions never become placeholder materials',()=>{
  const f=maps(fixture());assert.throws(()=>decodeGltfAnimationModel(f.model,f.buffers),code('GLTF_MODEL_TEXTURE'));
  for(const resolveTexture of [()=>({then(){}}),()=>null,()=>({view:{}}),()=>({view:{},sampler:{},flipY:true})])assert.throws(()=>decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture}));
});

test('buffer provider is called once per buffer across animation and geometry decode',()=>{
  const f=fixture(),calls=new Map();decodeGltfAnimationModel(f.model,i=>{calls.set(i,(calls.get(i)??0)+1);return f.buffers[i];});
  assert.ok([...calls.values()].every(v=>v===1));assert.ok(calls.size>0);
});

// Only replace the GPU-scene boundary. JSON/accessor/material decoding and pose
// construction below are production code. This verifies integration/ownership,
// not shader execution, rendering, device limits or image equivalence.
const encoded=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const gpuBoundary=encoded('export async function createGpuAnimationScene(device,pose,drawables,options){return device.createScene(pose,drawables,options);}');
let gpuSource=readFileSync(new URL('./animation_model_gpu.mjs',import.meta.url),'utf8');
for(const name of ['animation_model.mjs','animation_runtime.mjs','animation_scene.mjs'])gpuSource=gpuSource.replace("'./"+name+"'",JSON.stringify(name==='animation_scene.mjs'?gpuBoundary:new URL('./'+name,import.meta.url).href));
const {createGpuGltfAnimationScene}=await import(encoded(gpuSource));
function sceneBoundary() {
  const calls=[];let capturedPose,drawables;
  const scene={controller:{},draws:[{}],deformers:[],poseVersion:0,bufferBytes:100,disposed:false,failed:false,
    update(...args){calls.push(['update',...args]);},upload(){calls.push(['upload']);},render(frame){calls.push(['render',frame]);},
    whenIdle:async()=>{},dispose(){scene.disposed=true;}};
  return {scene,calls,get pose(){return capturedPose;},get drawables(){return drawables;},
    async createScene(pose,input,options){capturedPose=pose;drawables=input;calls.push(['create',options]);return scene;}};
}

test('GPU factory forwards decoded model and scene options while owning its new pose',async()=>{
  const f=fixture({skin:true,morph:true}),device=sceneBoundary(),m=await createGpuGltfAnimationScene(device,f.model,f.buffers,{scene:{maxBytes:4096}});
  assert.equal(m.pose,device.pose);assert.equal(m.controller,device.scene.controller);assert.equal(m.draws,device.scene.draws);
  assert.equal(device.drawables[0].geometry.influences,8);assert.deepEqual(device.calls[0],['create',{maxBytes:4096}]);
  const frame={viewProjection:[]};assert.equal(m.update(0.5),m);assert.equal(m.upload(),m);assert.equal(m.render(frame),m);assert.equal(await m.whenIdle(),m);
  assert.deepEqual(device.calls.slice(1),[['update',0.5,undefined],['upload'],['render',frame]]);
  m.dispose();assert.ok(device.scene.disposed);assert.ok(device.pose.disposed);
});

test('GPU initialization failure disposes the factory-owned pose, not external textures',async()=>{
  const f=maps(fixture()),texture=resources();let pose;
  await assert.rejects(createGpuGltfAnimationScene({async createScene(p){pose=p;throw Error('GPU initialization failed');}},f.model,f.buffers,{decode:{resolveTexture:()=>texture}}),/GPU initialization failed/);
  assert.equal(pose.disposed,true);assert.deepEqual(texture,resources());
});

test('recoverable frame errors retain pose; terminal completion errors dispose it',async()=>{
  const f=fixture(),device=sceneBoundary(),m=await createGpuGltfAnimationScene(device,f.model,f.buffers);
  device.scene.render=()=>{throw Error('bad frame');};assert.throws(()=>m.render({}),/bad frame/);assert.equal(m.pose.disposed,false);
  device.scene.whenIdle=async()=>{device.scene.failed=true;throw Error('device lost');};
  await assert.rejects(m.whenIdle(),/device lost/);assert.equal(m.pose.disposed,true);m.dispose();
});

test('GPU boundary rejection of reentrant disposal does not prematurely destroy the pose',async()=>{
  const f=fixture(),device=sceneBoundary(),m=await createGpuGltfAnimationScene(device,f.model,f.buffers),dispose=device.scene.dispose;
  device.scene.dispose=()=>{throw Error('busy scene');};assert.throws(()=>m.dispose(),/busy scene/);assert.equal(m.pose.disposed,false);
  device.scene.dispose=dispose;m.dispose();assert.equal(m.pose.disposed,true);
});

test('all 32 influences execute through the real CPU skinning path without truncation',()=>{
  const f=fixture({skin:true});f.model.nodes=[{mesh:0,skin:0},...Array.from({length:32},(_,i)=>({translation:[i,0,0]}))];
  f.model.skins=[{joints:Array.from({length:32},(_,i)=>i+1)}];f.model.scenes[0].nodes=Array.from({length:33},(_,i)=>i);f.model.animations=[];
  for(let set=0;set<8;set++){
    const values=new Uint8Array(Array.from({length:3},()=>[set*4,set*4+1,set*4+2,set*4+3]).flat()),buffer=f.buffers.push(values)-1;
    f.model.buffers.push({byteLength:values.byteLength});const bufferView=f.model.bufferViews.push({buffer,byteLength:values.byteLength})-1;
    f.p.attributes['JOINTS_'+set]=f.model.accessors.push({bufferView,componentType:5121,type:'VEC4',count:3})-1;
    f.p.attributes['WEIGHTS_'+set]=f.attr(Array(12).fill(1/32),'VEC4');
  }
  const m=createCpuGltfAnimationModel(f.model,f.buffers);
  assert.equal(m.drawables[0].geometry.influences,32);close([...m.deformers[0].positions],[15.5,0,0,16.5,0,0,15.5,1,0]);m.dispose();
});

test('mistaken async texture resolver rejection is observed without starting GPU construction',async()=>{
  const f=maps(fixture());let creates=0;
  await assert.rejects(createGpuGltfAnimationScene({createScene(){creates++;}},f.model,f.buffers,{
    decode:{resolveTexture:async()=>{throw Error('preload failed');}},
  }),code('GLTF_MODEL_TEXTURE'));
  await new Promise(resolve=>setImmediate(resolve));assert.equal(creates,0);
});

test('real decoded skin/morph model preserves independent material UVs without authored tangents',()=>{
  const f=maps(fixture({skin:true,morph:true}));delete f.p.attributes.TANGENT;
  f.material.normalTexture.texCoord=1;
  f.material.emissiveTexture.extensions={KHR_texture_transform:{offset:[0.25,0.5],scale:[2,3]}};
  const requests=[],m=createCpuGltfAnimationModel(f.model,f.buffers,{resolveTexture:r=>{requests.push(r);return resources();}}),d=m.drawables[0];
  assert.equal(requests.length,2);assert.equal(d.baseColorTexture,d.emissiveTexture);assert.equal(d.normalTexture,d.metallicRoughnessTexture);
  close(d.uvTransform,[1,0,0,1,0,0]);close(d.mapCoordinates.normalTexture.texCoords,[0.25,0.25,0.75,0.25,0.25,0.75]);
  close(d.mapCoordinates.emissiveTexture.uvTransform,[2,0,0,3,0.25,0.5]);
  close(d.texCoords,[0,0,1,0,0,1]);assert.equal(d.geometry.tangents,undefined);
  assert.ok(m.diagnostics.some(item=>item.reason==='DERIVATIVE_NORMAL_FRAME_NOT_MIKKTSPACE'));
  m.sample(0.5);close(m.deformers[0].positions,[2,2,1,3,2,1,2,3,1]);m.dispose();
});
