import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {prepareGltfAnimationModel,decodeGltfAnimationModel,createCpuGltfAnimationModel} from './animation_model.mjs';
import {createAnimationDeformer} from './animation_deformer.mjs';

// The GPU scene boundary is replaced; source/accessor/material decoding, camera
// evaluation, animation poses and CPU deformation below are production code.
const encoded=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const sceneBoundary=encoded('export async function createGpuAnimationScene(device,pose,drawables,options){return device.createScene(pose,drawables,options);}');
let gpuSource=readFileSync(new URL('./animation_model_gpu.mjs',import.meta.url),'utf8');
for(const name of ['animation_model.mjs','animation_runtime.mjs','animation_scene.mjs','gltf_scene_view.mjs']) {
  gpuSource=gpuSource.replace("'./"+name+"'",JSON.stringify(name==='animation_scene.mjs'?sceneBoundary:new URL('./'+name,import.meta.url).href));
}
const gpuModule=encoded(gpuSource);
const {createGpuGltfAnimationScene,createGpuDecodedAnimationScene}=await import(gpuModule);
const near=(a,b)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<1e-6,`${i}: ${a[i]} != ${b[i]}`);};
function fixture({orthographic=false}={}) {
  const json={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0,3]}],
    nodes:[{name:'animated rig',children:[1,2],translation:[10,0,0],scale:[2,2,2]},
      {name:'camera node',camera:0,translation:[0,0,5]},
      {name:'spot node',translation:[1,2,0],extensions:{KHR_lights_punctual:{light:0}}},{mesh:0}],
    cameras:[orthographic?{name:'ortho',type:'orthographic',orthographic:{xmag:2,ymag:1,znear:0,zfar:100}}:
      {name:'perspective',type:'perspective',perspective:{yfov:Math.PI/2,znear:1,zfar:100}}],
    extensionsUsed:['KHR_lights_punctual'],extensionsRequired:['KHR_lights_punctual'],
    extensions:{KHR_lights_punctual:{lights:[{name:'key',type:'spot',color:[0.8,0.6,0.4],intensity:12,range:8,spot:{innerConeAngle:0.2,outerConeAngle:0.6}}]}},
    meshes:[{primitives:[{attributes:{},material:0}]}],materials:[{pbrMetallicRoughness:{metallicFactor:0,roughnessFactor:1}}],
    buffers:[],bufferViews:[],accessors:[],animations:[]};
  const buffers=[];
  function attr(values,type) {
    const data=new Float32Array(values),buffer=buffers.push(data)-1,bufferView=json.bufferViews.push({buffer,byteLength:data.byteLength})-1;
    json.buffers.push({byteLength:data.byteLength});
    return json.accessors.push({bufferView,componentType:5126,type,count:values.length/({SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[type])})-1;
  }
  const p=json.meshes[0].primitives[0];
  p.attributes.POSITION=attr([0,0,0,1,0,0,0,1,0],'VEC3');json.accessors[0].min=[0,0,0];json.accessors[0].max=[1,1,0];
  p.attributes.NORMAL=attr([0,0,1,0,0,1,0,0,1],'VEC3');p.attributes.TEXCOORD_0=attr([0,0,1,0,0,1],'VEC2');
  const times=attr([0,1],'SCALAR');json.accessors[times].min=[0];json.accessors[times].max=[1];
  const values=[attr([10,0,0,14,0,0],'VEC3'),attr([0,0,5,0,0,7],'VEC3'),attr([0,0,0,1,0,Math.SQRT1_2,0,Math.SQRT1_2],'VEC4')];
  json.animations=[{samplers:values.map(output=>({input:times,output})),channels:[{sampler:0,target:{node:0,path:'translation'}},
    {sampler:1,target:{node:1,path:'translation'}},{sampler:2,target:{node:2,path:'rotation'}}]}];
  return {json,buffers};
}
function gpu() {
  let pose;const frames=[];
  const scene={controller:{},draws:[{}],deformers:[],poseVersion:0,bufferBytes:128,disposed:false,failed:false,
    update(time){pose.sample(time);this.upload();},upload(){for(const d of this.deformers)d.update();this.poseVersion=pose.version;},
    render(frame){if(this.disposed)throw Error('disposed scene');frames.push(frame);},async whenIdle(){},dispose(){this.disposed=true;for(const d of this.deformers)d.dispose();}};
  return {scene,frames,get pose(){return pose;},async createScene(p,drawables){pose=p;scene.deformers=drawables.map(d=>createAnimationDeformer(p,d.geometry));scene.poseVersion=p.version;return scene;}};
}

test('required punctual lights, camera instances and geometry are decoded together before texture resolution',()=>{
  const f=fixture(),plan=prepareGltfAnimationModel(f.json,f.buffers);
  assert.equal(plan.sceneView.cameras[0].node,1);assert.equal(plan.sceneView.lights[0].node,2);
  assert.ok(Object.isFrozen(plan.sceneView));assert.equal(plan.textureRequests.length,0);
  const decoded=plan.resolveTextures();assert.equal(decoded.sceneView,plan.sceneView);assert.equal(decoded.drawables.length,1);
  assert.equal(decoded.drawables[0].shading,'metallic-roughness');assert.equal(decoded.drawables[0].geometry.node,3);
});
test('real CPU animation moves the camera and parent rig while preserving photometric light properties',()=>{
  const f=fixture(),model=createCpuGltfAnimationModel(f.json,f.buffers),rest=model.view.sample({aspectRatio:1});
  near(rest.cameraPosition,[10,0,10]);near(rest.lighting.lights[0].position,[12,4,0]);
  assert.equal(model.cameras,model.view.cameras);assert.equal(model.lights,model.view.lights);
  model.sample(0.5);const frame=model.view.sample({aspectRatio:1});
  near(frame.cameraPosition,[12,0,12]);near(frame.lighting.lights[0].position,[14,4,0]);
  near(frame.lighting.lights[0].direction,[-Math.SQRT1_2,0,-Math.SQRT1_2]);
  assert.equal(frame.lighting.lights[0].range,8);assert.equal(frame.lighting.lights[0].intensity,12);
  assert.equal(frame.poseVersion,model.pose.version);assert.equal(model.deformers[0].poseVersion,frame.poseVersion);
  model.reset();near(model.view.sample({aspectRatio:1}).cameraPosition,[10,0,10]);
  model.dispose();assert.throws(()=>model.view.sample({aspectRatio:1}),{code:'GLTF_VIEW_POSE'});
});
test('root transforms and blended poses feed the same camera/light evaluation without a second clock',()=>{
  const f=fixture(),model=createCpuGltfAnimationModel(f.json,f.buffers),root=[1,0,0,0,0,1,0,0,0,0,1,0,100,200,300,1];
  model.pose.blend([{clip:0,time:1,weight:0.5}],{rootMatrix:root});model.update();
  const frame=model.view.sample({aspectRatio:1});near(frame.cameraPosition,[112,200,312]);near(frame.lighting.lights[0].position,[114,204,300]);
  const version=model.pose.version;model.view.sampleLights();model.view.sample({aspectRatio:1});assert.equal(model.pose.version,version);model.dispose();
});
test('source camera/light JSON mutations during texture resolution cannot change prepared metadata',()=>{
  const f=fixture();f.json.textures=[{source:0}];f.json.images=[{uri:'map.png'}];f.json.materials[0].pbrMetallicRoughness.baseColorTexture={index:0};
  const model=createCpuGltfAnimationModel(f.json,f.buffers,{resolveTexture(){
    f.json.cameras[0].perspective.yfov=0;f.json.extensions.KHR_lights_punctual.lights[0].intensity=99;return {view:{},sampler:{}};
  }});
  assert.equal(model.view.sample({aspectRatio:1}).lighting.lights[0].intensity,12);assert.equal(model.cameras[0].projection.yfov,Math.PI/2);model.dispose();
});
for(const [label,mutate] of [
  ['invalid camera',j=>{j.cameras[0].perspective.znear=0;}],
  ['invalid light',j=>{j.extensions.KHR_lights_punctual.lights[0].intensity=-1;}],
  ['excess lights',j=>{for(let i=0;i<8;i++){j.nodes[0].children.push(j.nodes.length);j.nodes.push({extensions:{KHR_lights_punctual:{light:0}}});}}],
])test(`${label} fails before any buffer provider, texture resolver or GPU call`,async()=>{
  const f=fixture();mutate(f.json);let buffers=0,resolutions=0,creates=0;
  const options={resolveTexture(){resolutions++;return {view:{},sampler:{}};}};
  assert.throws(()=>decodeGltfAnimationModel(f.json,()=>{buffers++;},options));
  await assert.rejects(createGpuGltfAnimationScene({createScene(){creates++;}},f.json,()=>{buffers++;},{decode:options}));
  assert.equal(buffers,0);assert.equal(resolutions,0);assert.equal(creates,0);
});
test('GPU renderCamera submits current source view and lighting using the existing frame/attachment route',async()=>{
  const f=fixture(),device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers);
  const colorView={},depthView={},draws=m.draws;assert.equal(m.update(0.5),m);
  assert.equal(m.renderCamera({colorView,depthView,draws},{cameraNode:1,aspectRatio:2}),m);
  const frame=device.frames[0];assert.equal(frame.colorView,colorView);assert.equal(frame.depthView,depthView);assert.equal(frame.draws,draws);
  near(frame.lighting.cameraPosition,[12,0,12]);near(frame.lighting.lights[0].position,[14,4,0]);
  near(frame.viewProjection,m.view.sample({cameraNode:1,aspectRatio:2}).viewProjection);assert.equal(m.poseVersion,m.pose.version);
  m.dispose();assert.equal(m.pose.disposed,true);
});
test('orthographic camera rendering uses constant view direction rather than positional view lighting',async()=>{
  const f=fixture({orthographic:true}),device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers);
  m.renderCamera({colorView:{},depthView:{}});assert.equal(device.frames[0].lighting.cameraPosition,undefined);
  near(device.frames[0].lighting.viewDirection,[0,0,1]);m.dispose();
});
test('direct pose sampling must be uploaded before camera rendering; recovery retains the model',async()=>{
  const f=fixture(),device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers);
  m.pose.sample(0.5);assert.throws(()=>m.renderCamera({},{aspectRatio:1}),{code:'GLTF_VIEW_STALE'});assert.equal(device.frames.length,0);assert.equal(m.failed,false);
  m.upload();m.renderCamera({},{aspectRatio:1});assert.equal(device.frames.length,1);m.dispose();
});
test('camera option or attachment getters cannot reenter model operations or dispose resources',async()=>{
  const f=fixture(),device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers);
  for(const operation of [()=>m.dispose(),()=>m.update(0.5),()=>m.render({})]) {
    assert.throws(()=>m.renderCamera({get colorView(){operation();return {};}},{aspectRatio:1}),{code:'GLTF_VIEW_REENTRANT'});
    assert.throws(()=>m.renderCamera({},{get aspectRatio(){operation();return 1;}}),{code:'GLTF_VIEW_REENTRANT'});
  }
  assert.equal(device.frames.length,0);assert.equal(m.disposed,false);assert.equal(m.pose.disposed,false);
  m.renderCamera({},{aspectRatio:1});m.dispose();
});
test('camera errors and conflicting external frame fields produce no draw and remain recoverable',async()=>{
  const f=fixture(),device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers);
  for(const frame of [{viewProjection:[]},{lighting:{}},null,[]])assert.throws(()=>m.renderCamera(frame,{aspectRatio:1}),{code:'GLTF_VIEW_FRAME'});
  assert.throws(()=>m.renderCamera({}),{code:'GLTF_VIEW_ASPECT'});
  assert.throws(()=>m.renderCamera({},{cameraNode:999,aspectRatio:1}),{code:'GLTF_VIEW_CAMERA'});
  assert.equal(device.frames.length,0);assert.equal(m.failed,false);m.renderCamera({},{aspectRatio:1});m.dispose();
});
test('explicit external rendering remains unchanged, including models without authored views',async()=>{
  const f=fixture();delete f.json.nodes[1].camera;delete f.json.nodes[2].extensions;
  const device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers),frame={viewProjection:[1],lighting:{lights:[]},colorView:{}};
  assert.equal(m.cameras.length,0);assert.equal(m.lights.length,0);assert.equal(m.render(frame),m);assert.equal(device.frames[0],frame);
  assert.throws(()=>m.renderCamera({}),{code:'GLTF_VIEW_CAMERA'});m.dispose();
});
test('selected scene camera/light membership matches the geometry scene selection',()=>{
  const f=fixture();f.json.scenes.push({nodes:[3]});const m=createCpuGltfAnimationModel(f.json,f.buffers,{scene:1});
  assert.equal(m.deformers.length,1);assert.equal(m.cameras.length,0);assert.equal(m.lights.length,0);m.dispose();
});
test('unknown required extensions still fail and no camera/light property animation is falsely accepted',()=>{
  const f=fixture();f.json.extensionsRequired.push('EXT_unknown');assert.throws(()=>decodeGltfAnimationModel(f.json,f.buffers),{code:'GLTF_MODEL_UNSUPPORTED'});
  f.json.extensionsRequired.pop();f.json.animations[0].channels[0].target.extensions={KHR_animation_pointer:{pointer:'/cameras/0/perspective/yfov'}};
  assert.throws(()=>decodeGltfAnimationModel(f.json,f.buffers),{code:'GLTF_ANIMATION_EXTENSION'});
});
test('terminal render errors dispose the owned pose while recoverable errors retain it',async()=>{
  const f=fixture(),device=gpu(),m=await createGpuGltfAnimationScene(device,f.json,f.buffers);
  device.scene.render=()=>{throw Error('bad attachment');};assert.throws(()=>m.renderCamera({},{aspectRatio:1}),/bad attachment/);assert.equal(m.pose.disposed,false);
  device.scene.render=()=>{device.scene.failed=true;throw Error('device lost');};assert.throws(()=>m.renderCamera({},{aspectRatio:1}),/device lost/);assert.equal(m.pose.disposed,true);m.dispose();
});
test('lower-level decoded-model entry preserves legacy geometry-only payloads',async()=>{
  const f=fixture(),prepared=decodeGltfAnimationModel(f.json,f.buffers);delete prepared.sceneView;
  const device=gpu(),m=await createGpuDecodedAnimationScene(device,prepared);assert.equal(m.cameras.length,0);m.render({});m.dispose();
});

// This final seam verifies the owning loader's forwarding and preflight order.
// Asset transport and native texture preparation are supplied test boundaries;
// they are unchanged by this feature and are exercised in their own suites.
const assetBoundary=encoded("export class GltfAssetError extends Error {}\nexport async function loadGltfAsset(source,options){return options.fetch(source);}");
const textureBoundary=encoded("export class GltfTextureError extends Error {}\nexport async function createGltfTextureResources(device,requests,readImage,options){return device.textureResources(requests,readImage,options);}");
let loaderSource=readFileSync(new URL('./gltf_scene_loader.mjs',import.meta.url),'utf8');
for(const [name,url]of [['gltf_asset.mjs',assetBoundary],['gltf_textures.mjs',textureBoundary],['animation_model_gpu.mjs',gpuModule],['animation_model.mjs',new URL('./animation_model.mjs',import.meta.url).href]])loaderSource=loaderSource.replace("'./"+name+"'",JSON.stringify(url));
const {loadGpuGltfAnimationScene}=await import(encoded(loaderSource));
test('owning scene loader exposes authored views and forwards camera rendering without changing ownership',async()=>{
  const f=fixture(),device=gpu();let prepared=0,disposed=0;
  device.textureResources=async()=>{prepared++;return {textureBytes:0,failed:false,resolveTexture:()=>({view:{},sampler:{}}),dispose(){disposed++;}};};
  const m=await loadGpuGltfAnimationScene(device,'model.glb',{assets:{fetch:async()=>({json:f.json,buffers:f.buffers,bytesLoaded:123,readImage:()=>assert.fail('No images')})}});
  assert.equal(prepared,1);assert.equal(m.assetBytes,123);assert.equal(m.cameras,m.view.cameras);assert.equal(m.lights,m.view.lights);
  m.update(0.5);assert.equal(m.renderCamera({},{aspectRatio:1}),m);near(device.frames[0].lighting.cameraPosition,[12,0,12]);
  m.dispose();assert.equal(disposed,1);assert.equal(m.pose.disposed,true);
});
test('owning loader refuses malformed authored cameras before texture preparation',async()=>{
  const f=fixture();f.json.cameras[0].perspective.znear=0;
  const device=gpu();device.textureResources=async()=>assert.fail('No textures before camera preflight');
  await assert.rejects(loadGpuGltfAnimationScene(device,'model.glb',{assets:{fetch:async()=>({json:f.json,buffers:f.buffers})}}),{code:'GLTF_VIEW_VALUE'});
  assert.equal(device.pose,undefined);
});
