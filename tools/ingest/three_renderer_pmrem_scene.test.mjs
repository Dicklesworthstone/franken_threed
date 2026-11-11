/** Retained r186 public PMREM on the WebGL-authored owned renderer. Recording
 * checks accepted bytes, source pass order and lifetimes; browser tests rasterize. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass, createPMREMGeneratorClass} from './three_renderer.mjs';
import {createThreePassSnapshots} from './three_pass_snapshot.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const Renderer=createWebGLRendererClass(T,{maxDraws:64,maxBindings:64}), Generator=createPMREMGeneratorClass(T);
function fixture(t,{deferred=false}={}){
  const device=geometryDevice();Object.assign(device.limits,{maxTextureDimension2D:8192,maxBindingsPerBindGroup:1000});
  device.textures=[];device.textureWrites=[];
  device.createTexture=descriptor=>{const texture={dimension:'2d',depthOrArrayLayers:1,sampleCount:1,...descriptor,
    width:descriptor.size.width??descriptor.size[0],height:descriptor.size.height??descriptor.size[1],
    createView(options={}){return {texture,...options};},destroy(){this.destroyed=true;}};device.textures.push(texture);return texture;};
  device.createRenderPipeline=descriptor=>({...descriptor,getBindGroupLayout:()=>({})});device.createSampler=descriptor=>({...descriptor});
  device.queue.writeTexture=(destination,data)=>{const bytes=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength).slice();
    destination.texture.pixels=bytes;device.textureWrites.push({texture:destination.texture,bytes});};
  const submit=device.queue.submit;
  device.queue.submit=commands=>{for(const passes of commands)for(const pass of passes)for(const draw of pass.draws)
    for(const {group} of draw.groups.values())for(const entry of group.entries??[])if(entry.resource?.texture)
      assert.notEqual(entry.resource.texture.destroyed,true,'submitted sampled view must remain live');submit(commands);};
  let configuration;
  const canvas={width:32,height:32,style:{},getContext:()=>({configure(value){configuration=value;},unconfigure(){},
    getCurrentTexture(){return device.createTexture({size:[32,32],format:configuration.format,usage:16});}})};
  let release;
  const gpu={requestAdapter:()=>new Promise(resolve=>{release=()=>resolve({features:new Set(),requestDevice:async()=>device});})};
  const SelectedRenderer=deferred?createWebGLRendererClass(T,{gpu,maxDraws:64,maxBindings:64}):Renderer;
  const renderer=new SelectedRenderer({canvas,...(deferred?{}:{device})});
  device.destroy=()=>{device.destroyed=true;};
  t.after(async()=>{renderer.dispose();await renderer._disposePromise;});
  return {renderer,device,release:()=>release()};
}
const drain=async renderer=>{await renderer.waitForGPU();await Promise.resolve();};


import {RoomEnvironment} from '../../upstream/three.js/examples/jsm/environments/RoomEnvironment.js';
import {Sky} from '../../upstream/three.js/examples/jsm/objects/Sky.js';

test('initialized fromScene returns a real owned target and captures the actual RoomEnvironment',async t=>{
 const {renderer,device}=fixture(t);await renderer.init();
 const environment=new RoomEnvironment(),generator=new Generator(renderer),selected=new T.WebGLRenderTarget(8,8);
 const instance=environment.children.find(object=>object.isInstancedMesh);assert.equal(instance.count,6);
 renderer.setRenderTarget(selected);renderer.setClearDepth(.25);renderer.autoClear=true;renderer.toneMapping=T.ACESFilmicToneMapping;
 const background=new T.Color(0.25,0.125,0.5);environment.background=background;renderer.xr.enabled=true;
 const output=generator.fromScene(environment,0.04,0.1,100,{size:16,renderTarget:selected});
 assert.equal(output.isWebGLRenderTarget,true);assert.notEqual(output,selected);
 assert.deepEqual([output.width,output.height],[336,64]);assert.equal(output.texture.mapping,T.CubeUVReflectionMapping);
 assert.equal(renderer.getRenderTarget(),selected);assert.equal(renderer.getClearDepth(),.25);assert.equal(renderer.xr.enabled,true);
 assert.equal(renderer.toneMapping,T.ACESFilmicToneMapping);assert.equal(renderer.autoClear,true);assert.equal(environment.background,background);
 renderer.xr.enabled=false;instance.count=0;instance.instanceMatrix.array.fill(0);environment.dispose();instance.dispose();environment.clear();
 generator.dispose();await drain(renderer);
 assert.equal(device.passes[0].desc.depthStencilAttachment.depthClearValue,1,'fresh normal depth is initialized independently of renderer clear depth');
 const draws=device.snapshots.flat();assert.ok(draws.some(draw=>draw.args[1]===6),'captured six CPU instances survive mutation');
 assert.ok(draws.some(draw=>draw.pipeline.fragment.module.code.includes('PMREM')) || draws.length>30);
 assert.equal(renderer._passSnapshots.rootCount,0);assert.equal(renderer._passSnapshots.allocatedBytes,0);output.dispose();selected.dispose();
});

test('Sky uniforms and scene inputs are captured before queued source filtering',async t=>{
 const {renderer,device}=fixture(t);await renderer.init();const sky=new Sky(),generator=new Generator(renderer);
 sky.scale.setScalar(100);sky.material.uniforms.sunPosition.value.set(1,1,0);
 const output=generator.fromScene(sky,0.08,0.1,200,{size:16});
 sky.material.uniforms.turbidity.value=999;sky.geometry.dispose();sky.material.dispose();generator.dispose();
 await drain(renderer);assert.ok(device.snapshots.flat().some(draw=>draw.pipeline.fragment.module.code.includes('turbidity')));
 assert.equal(renderer._passSnapshots.rootCount,0);output.dispose();
});

test('retired PMREM instance views do not accumulate public instance upload history',async t=>{
 const {renderer}=fixture(t);await renderer.init();
 for(let iteration=0;iteration<3;iteration++){
  const environment=new RoomEnvironment(),generator=new Generator(renderer);
  const instance=environment.children.find(object=>object.isInstancedMesh);
  const output=generator.fromScene(environment,0,.1,100,{size:16});
  t.after(()=>output.dispose());
  environment.dispose();instance.dispose();environment.clear();generator.dispose();
  await drain(renderer);
  assert.equal(renderer._passInstances.history.bytes,0,`retired generator ${iteration+1} leaves no public instance history`);
  assert.equal(renderer._passSnapshots.rootCount,0);assert.equal(renderer._passSnapshots.allocatedBytes,0);
 }
});

test('fromScene before device acquisition captures real targets, sigma and independently queued scenes',async t=>{
 const {renderer,device,release}=fixture(t,{deferred:true}),room=new RoomEnvironment(),generator=new Generator(renderer);
 const instances=room.children.find(object=>object.isInstancedMesh),selected=new T.WebGLRenderTarget(8,8);
 renderer.setRenderTarget(selected);renderer.setClearDepth(.25);renderer.toneMapping=T.ACESFilmicToneMapping;
 const first=generator.fromScene(room,0,.1,100,{size:16,renderTarget:selected});
 assert.equal(renderer.initialized,false);assert.equal(first.isWebGLRenderTarget,true);assert.notEqual(first,selected);
 assert.equal(first.texture.isF3DSceneEnvironment,undefined);
 instances.count=2;instances.instanceMatrix.array[12]+=2;instances.instanceMatrix.needsUpdate=true;
 const second=generator.fromScene(room,.2,.1,100,{size:16,position:new T.Vector3(1,2,3)});
 assert.equal(second.isWebGLRenderTarget,true);assert.notEqual(second,first);
 assert.equal(renderer.getRenderTarget(),selected);assert.equal(renderer.getClearDepth(),.25);
 assert.equal(renderer.toneMapping,T.ACESFilmicToneMapping);assert.equal(device.submissions.length,0);
 room.dispose();instances.dispose();room.clear();generator.dispose();
 release();await drain(renderer);
 const draws=device.snapshots.flat();
 assert.ok(draws.some(draw=>draw.args[1]===6),'first accepted instance count survives later edits');
 assert.ok(draws.some(draw=>draw.args[1]===2),'second accepted instance count survives disposal');
 assert.ok(draws.some(draw=>draw.pipeline.fragment.module.code.includes('thetaMax')),'sigma filtering is queued before initialization');
 assert.equal(renderer._passSnapshots.rootCount,0);assert.equal(renderer._passSnapshots.allocatedBytes,0);
 first.dispose();second.dispose();selected.dispose();
});

test('disposing a renderer during pre-init scene baking cancels pending resources',async t=>{
 const {renderer,device,release}=fixture(t,{deferred:true}),room=new RoomEnvironment(),generator=new Generator(renderer);
 const output=generator.fromScene(room,.04,.1,100,{size:16});
 generator.dispose();renderer.dispose();release();await renderer._disposePromise;await Promise.resolve();
 assert.equal(device.submissions.length,0);assert.equal(renderer._passSnapshots.rootCount,0);
 assert.equal(renderer._passSnapshots.allocatedBytes,0);assert.equal(renderer._device,null,'cancellation before adapter readiness never acquires a device');
 output.dispose();room.dispose();room.children.find(object=>object.isInstancedMesh).dispose();
});

import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';
import {WebGLAttributes} from '../../upstream/three.js/src/renderers/webgl/WebGLAttributes.js';
import {WebGLMorphtargets} from '../../upstream/three.js/src/renderers/webgl/WebGLMorphtargets.js';
import {threeMorphTargets, createThreeProgramSupport} from './three_program.mjs';
function uploadOracle(){
 let bound;return WebGLAttributes({
  createBuffer:()=>({data:null}),bindBuffer(_target,buffer){bound=buffer;},
  bufferData(_target,array){bound.data=new array.constructor(array);},
  bufferSubData(_target,offset,array,start=0,count=0){bound.data.set(array.subarray(start,count?start+count:array.length),offset/array.BYTES_PER_ELEMENT);},
  deleteBuffer(){}
 });
}
test('scene histories acknowledge only source visible/layer/frustum selection, with exact partial-range bytes',()=>{
 const device=geometryDevice(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(60,1,.1,10);
 const scene=new T.Scene(),parent=new T.Group(),geometry=new T.BufferGeometry();
 const attribute=new T.BufferAttribute(new Float32Array([0,0,0,1,0,0,0,1,0]),3),reference=attribute.clone(),gl=uploadOracle();
 geometry.setAttribute('position',attribute);const mesh=new T.Mesh(geometry,new T.MeshBasicMaterial());mesh.position.z=-2;parent.add(mesh);scene.add(parent);
 const take=()=>{const capture=owner.capturePMREMScene(scene,camera,device,{cube:true,position:new T.Vector3()});capture.install();
  const values=Array.from(capture.root.children[0].children[0].geometry.attributes.position.array);capture.release();return values;};
 const same=()=>assert.deepEqual(take(),Array.from(gl.get(reference).buffer.data));
 parent.visible=false;take();assert.equal(geometry.boundingSphere,null,'invisible ancestry does not compute source bounds');
 attribute.array[0]=reference.array[0]=.25;attribute.addUpdateRange(2,1);reference.addUpdateRange(2,1);
 parent.visible=true;gl.update(reference,0);same();assert.equal(attribute.updateRanges.length,1,'initial upload preserves ranges');
 parent.visible=false;attribute.array.fill(17);reference.array.fill(17);
 for(const item of [attribute,reference]){item.clearUpdateRanges();item.addUpdateRange(4,1);item.addUpdateRange(2,1);item.needsUpdate=true;}
 const merged=attribute.updateRanges[1];take();assert.equal(attribute.updateRanges.length,2,'hidden pending ranges remain pending');
 parent.visible=true;mesh.layers.set(1);take();assert.equal(attribute.updateRanges.length,2,'excluded layers do not upload');
 mesh.layers.set(0);mesh.position.z=-100;take();assert.equal(attribute.updateRanges.length,2,'all six frusta exclude the distant object');
 mesh.position.z=-2;gl.update(reference,0);same();assert.equal(attribute.updateRanges.length,0);assert.equal(merged.count,3);
 assert.equal(gl.get(reference).buffer.data[0],.25);assert.equal(gl.get(reference).buffer.data[3],17);
 geometry.dispose();attribute.array[0]=23;reference.array[0]=23;
 const reset=take();assert.equal(reset[0],23,'source disposal starts a new upload lifetime without a version bump');
 owner.dispose();geometry.dispose();mesh.material.dispose();
});
test('PMREM and ordinary WebGL geometry owners require disposal to transfer history',()=>{
 const device=geometryDevice(),geometry=new T.BoxGeometry(),mesh=new T.Mesh(geometry,new T.MeshBasicMaterial()),camera=new T.PerspectiveCamera();
 const owner=createThreePassSnapshots(T),resident=createGpuBufferGeometry(device,geometry);
 assert.throws(()=>owner.capturePMREMScene(mesh,camera,device),/separate WebGL geometry owner/);
 geometry.dispose();const captured=owner.capturePMREMScene(mesh,camera,device);captured.release();
 assert.throws(()=>createGpuBufferGeometry(device,geometry),/PMREM-captured attributes require disposal/);
 geometry.dispose();const next=createGpuBufferGeometry(device,geometry);next.dispose();resident.dispose();owner.dispose();
});
test('deferred device binding claims retained histories and rolls back a conflicting batch',()=>{
 const device=geometryDevice(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(),scene=new T.Scene();
 const first=new T.Mesh(new T.BoxGeometry(),new T.MeshBasicMaterial()),second=new T.Mesh(new T.BoxGeometry(),new T.MeshBasicMaterial());
 first.frustumCulled=second.frustumCulled=false;scene.add(first,second);
 const snapshot=owner.capturePMREMScene(scene,camera,null),resident=createGpuBufferGeometry(device,second.geometry);
 assert.throws(()=>owner.bindDevice(device),/separate WebGL geometry owner/);
 const rolledBack=createGpuBufferGeometry(device,first.geometry);
 first.geometry.dispose();second.geometry.dispose();rolledBack.dispose();resident.dispose();
 owner.bindDevice(device);snapshot.install();snapshot.release();
 const accepted=owner.capturePMREMScene(scene,camera,device);accepted.release();
 assert.throws(()=>createGpuBufferGeometry(device,first.geometry),/PMREM-captured attributes require disposal/);
 assert.throws(()=>owner.bindDevice(geometryDevice()),/one renderer device/);
 owner.dispose();first.geometry.dispose();second.geometry.dispose();first.material.dispose();second.material.dispose();
});
test('oversized scene arrays are rejected before retaining a history owner',()=>{
 const device=geometryDevice(),geometry=new T.BufferGeometry(),array=new Float32Array(3000);
 geometry.setAttribute('position',new T.BufferAttribute(array,3));const mesh=new T.Mesh(geometry,new T.MeshBasicMaterial());mesh.frustumCulled=false;
 const owner=createThreePassSnapshots(T,{maxCaptureBytes:1024,maxBytes:2048}),camera=new T.PerspectiveCamera();
 assert.throws(()=>owner.capturePMREMScene(mesh,camera,device),/BUDGET/);assert.equal(owner.allocatedBytes,0);
 const resident=createGpuBufferGeometry(device,geometry);resident.dispose();owner.dispose();geometry.dispose();
});

test('concurrent GPU waits keep private PMREM resources through size change and generator disposal',async t=>{
 const {renderer,device}=fixture(t);await renderer.init();
 const scene=new T.Scene(),mesh=new T.Mesh(new T.BoxGeometry(),new T.MeshBasicMaterial({side:T.BackSide}));scene.add(mesh);
 const generator=new Generator(renderer),first=generator.fromScene(scene,0,.1,10,{size:16});await drain(renderer);
 const oldPingPong=generator._generator._pingPongRenderTarget;let oldDisposals=0,newDisposals=0;
 oldPingPong.addEventListener('dispose',()=>oldDisposals++);
 let finish;device.completion=new Promise(resolve=>{finish=resolve;});
 const waiting=[renderer.waitForGPU(),renderer.waitForGPU()];
 const second=generator.fromScene(scene,.05,.1,10,{size:32});
 generator._generator._pingPongRenderTarget.addEventListener('dispose',()=>newDisposals++);
 generator.dispose();assert.equal(oldDisposals,0);assert.equal(newDisposals,0);
 waiting.push(renderer.waitForGPU());finish();await Promise.all(waiting);await Promise.resolve();
 assert.equal(oldDisposals,1);assert.equal(newDisposals,1);
 assert.equal(renderer._targets.capture(first).alive,true);assert.equal(renderer._targets.capture(second).alive,true);
 first.dispose();second.dispose();mesh.geometry.dispose();mesh.material.dispose();
});

test('a rejected GPU wait releases queued private retirement without masking the original failure',async t=>{
 const {renderer,device}=fixture(t);await renderer.init();
 const root=new T.Mesh(new T.BoxGeometry(),new T.MeshBasicMaterial({side:T.BackSide})),generator=new Generator(renderer);
 const output=generator.fromScene(root,0,.1,10,{size:16});await drain(renderer);
 let disposed=0;generator._generator._pingPongRenderTarget.addEventListener('dispose',()=>disposed++);
 device.completion=Promise.reject(new Error('intentional GPU completion failure'));device.completion.catch(()=>{});
 const waiting=renderer.waitForGPU();generator.dispose();assert.equal(disposed,0);
 await assert.rejects(waiting,/intentional GPU completion failure/);
 assert.equal(disposed,1);assert.equal(renderer._gpuWaiters,0);assert.equal(renderer._pmremIdleReleases,null);
 output.dispose();root.geometry.dispose();root.material.dispose();
});

function skinnedEnvironment() {
 const scene=new T.Scene(),geometry=new T.BoxGeometry(1,2,1),indices=[],weights=[];
 const position=geometry.attributes.position;
 for(let i=0;i<position.count;i++){
  const weight=(position.getY(i)+1)/2;
  indices.push(0,1,0,0);weights.push(1-weight,weight,0,0);
 }
 geometry.setAttribute('skinIndex',new T.Uint16BufferAttribute(indices,4));
 geometry.setAttribute('skinWeight',new T.Float32BufferAttribute(weights,4));
 const mesh=new T.SkinnedMesh(geometry,new T.MeshNormalMaterial({side:T.DoubleSide}));
 const root=new T.Bone(),tip=new T.Bone();root.position.y=-1;tip.position.y=2;root.add(tip);
 mesh.add(root);mesh.bind(new T.Skeleton([root,tip]));mesh.frustumCulled=false;scene.add(mesh);
 return {scene,mesh,root,tip,skeleton:mesh.skeleton};
}

test('PMREM skinning captures independent poses and preserves source palette updates once per face',()=>{
 const f=skinnedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(90,1,.1,20);
 const sibling=new T.SkinnedMesh(f.mesh.geometry,f.mesh.material);sibling.bind(f.skeleton,new T.Matrix4());
 sibling.bindMode=T.DetachedBindMode;sibling.frustumCulled=false;f.scene.add(sibling);
 const capture=()=>owner.capturePMREMScene(f.scene,camera,null,{cube:true,position:new T.Vector3()});
 try{
  f.tip.rotation.z=.4;const first=capture(),firstPalette=f.skeleton.boneMatrices.slice();
  assert.equal(f.skeleton.boneTexture.version,6,'one creation plus five selected-frame updates, shared by both meshes');
  f.tip.rotation.z=-.8;const second=capture(),secondPalette=f.skeleton.boneMatrices.slice();
  assert.equal(f.skeleton.boneTexture.version,12);assert.notDeepEqual(firstPalette,secondPalette);
  f.tip.rotation.z=2;f.skeleton.dispose();f.scene.clear();
  first.install();const a=first.root.children[0],shared=first.root.children[1];
  assert.notEqual(a.skeleton,f.skeleton);assert.equal(a.skeleton,shared.skeleton);
  assert.notEqual(a.skeleton.bones[1],f.tip);a.skeleton.update();a.skeleton.computeBoneTexture();
  const privateTexture=a.skeleton.boneTexture;
  assert.deepEqual(a.skeleton.boneMatrices,firstPalette);assert.equal(a.skeleton.bones[1].rotation.z,.4);
  second.install();const b=second.root.children[0];b.skeleton.update();
  assert.equal(b.skeleton.boneTexture,privateTexture,'later snapshots reuse the private palette residency');
  assert.equal(privateTexture.image.data,b.skeleton.boneMatrices);
  assert.deepEqual(b.skeleton.boneMatrices,secondPalette);assert.equal(b.skeleton.bones[1].rotation.z,-.8);
  first.release();second.release();
 }finally{owner.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.skeleton.dispose();}
});

test('PMREM detached skeletons freeze external bone worlds without capturing unrelated descendants',()=>{
 const f=skinnedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera();
 f.mesh.remove(f.root);f.mesh.bindMode=T.DetachedBindMode;f.tip.position.x=3;f.root.updateMatrixWorld(true);
 const unrelated=new T.Mesh(new T.BoxGeometry(),new T.MeshBasicMaterial());
 unrelated.onBeforeRender=()=>{throw Error('unrelated callback');};f.tip.add(unrelated);
 try{
  const snapshot=owner.capturePMREMScene(f.scene,camera,null),expected=f.skeleton.boneMatrices.slice();
  f.tip.position.x=99;f.root.updateMatrixWorld(true);snapshot.install();
  const skeleton=snapshot.root.children[0].skeleton;skeleton.update();
  assert.deepEqual(skeleton.boneMatrices.subarray(0,expected.length),expected);
  assert.deepEqual(skeleton.bones[1].children,[]);snapshot.release();
 }finally{owner.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.skeleton.dispose();unrelated.geometry.dispose();unrelated.material.dispose();}
});

test('PMREM skeleton acknowledgement follows source visibility, layers and material selection',()=>{
 const f=skinnedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(90,1,.1,20);
 const take=()=>{const snapshot=owner.capturePMREMScene(f.scene,camera,null,{cube:true,position:new T.Vector3()});snapshot.release();};
 try{
  f.tip.rotation.z=.6;f.mesh.visible=false;take();assert.equal(f.skeleton.boneTexture,null);
  assert.ok(f.skeleton.boneMatrices.every(value=>value===0));
  f.mesh.visible=true;f.mesh.layers.set(1);take();assert.ok(f.skeleton.boneMatrices.every(value=>value===0));
  f.mesh.layers.set(0);f.mesh.material.visible=false;take();
  assert.equal(f.skeleton.boneTexture,null,'invisible materials do not invoke setProgram');
  assert.ok(f.skeleton.boneMatrices.some(value=>value!==0),'projected objects still update their skeleton');
  f.mesh.material.visible=true;take();assert.equal(f.skeleton.boneTexture.version,6);
 }finally{owner.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.skeleton.dispose();}
});

test('PMREM refuses custom skeleton execution before queuing or advancing accepted palettes',()=>{
 const f=skinnedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera();
 let calls=0;f.skeleton.update=()=>{calls++;};
 try{
  assert.throws(()=>owner.capturePMREMScene(f.scene,camera,null),/ordinary bound source skeleton/);
  assert.equal(calls,0);assert.equal(owner.allocatedBytes,0);assert.equal(f.skeleton.boneTexture,null);
 }finally{owner.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.skeleton.dispose();}
});

test('PMREM rejects palette uniform aliases and altered bone texture uploads before source acknowledgement',()=>{
 for(const change of ['uniform','uniform-view','attribute','map','callback','flipY','dataReady','ranges','internalFormat']){
  const f=skinnedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera();
  f.skeleton.update();f.skeleton.computeBoneTexture();const texture=f.skeleton.boneTexture;
  let calls=0,shader;
  if(change==='uniform'||change==='uniform-view'){
   shader=new T.ShaderMaterial({uniforms:{palette:{value:change==='uniform'?f.skeleton.boneMatrices:
    new Float32Array(f.skeleton.boneMatrices.buffer)}}});f.mesh.material=shader;
  }else if(change==='attribute')f.mesh.geometry.setAttribute('palette',new T.BufferAttribute(f.skeleton.boneMatrices,4));
  else if(change==='map')f.mesh.material.map=texture;
  else if(change==='callback')texture.onUpdate=()=>{calls++;};
  else if(change==='flipY')texture.flipY=true;
  else if(change==='ranges')texture.addUpdateRange(0,4);
  else if(change==='internalFormat')texture.internalFormat='RGBA16F';
  else texture.source.dataReady=false;
  const before=f.skeleton.boneMatrices.slice(),version=texture.version;f.tip.rotation.z=.9;
  try{
   assert.throws(()=>owner.capturePMREMScene(f.scene,camera,null),error=>error.code==='THREE_PASS_SCENE',change);
   assert.equal(calls,0);assert.deepEqual(f.skeleton.boneMatrices,before);assert.equal(texture.version,version);
   assert.equal(owner.allocatedBytes,0);
  }finally{owner.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();shader?.dispose();f.skeleton.dispose();}
 }
});

test('public PMREM queues skinned scene poses before initialization and retains them after source disposal',async t=>{
 const {renderer,device,release}=fixture(t,{deferred:true}),f=skinnedEnvironment(),generator=new Generator(renderer);
 f.mesh.position.z=-3;f.tip.rotation.z=.35;
 const first=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.tip.rotation.z=-.7;const second=generator.fromScene(f.scene,.1,.1,20,{size:16});
 f.skeleton.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.scene.clear();generator.dispose();
 release();await drain(renderer);
 const skinWrites=device.textureWrites.filter(write=>write.texture.width===4&&write.texture.height===4&&write.texture.format==='rgba32float');
 assert.ok(skinWrites.length>=2,'both source palettes reach the owned ShaderLib draws');
 assert.ok(skinWrites.some(write=>new Float32Array(write.bytes.buffer)[16]===Math.fround(Math.cos(.35))));
 assert.ok(skinWrites.some(write=>new Float32Array(write.bytes.buffer)[16]===Math.fround(Math.cos(-.7))));
 assert.equal(renderer._passSnapshots.rootCount,0);assert.equal(renderer._passSnapshots.allocatedBytes,0);
 assert.equal(renderer.info.f3d.route,'general-webgpu');first.dispose();second.dispose();
});

function recordDeformations(device) {
 const records=[];
 Object.assign(device.limits,{maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535});
 device.createComputePipelineAsync=async descriptor=>({...descriptor,getBindGroupLayout:()=>({})});
 const create=device.createCommandEncoder.bind(device),submit=device.queue.submit;
 device.createCommandEncoder=()=>{
  const encoder=create(),passes=[];
  return {...encoder,beginComputePass(descriptor){
   const pass={descriptor,draws:[]};passes.push(pass);
   return {setPipeline(pipeline){pass.pipeline=pipeline;},setBindGroup(_index,group){pass.group=group;},dispatchWorkgroups(){},end(){}};
  },finish(){return [...encoder.finish(),...passes];}};
 };
 device.queue.submit=commands=>{
  for(const passes of commands)for(const pass of passes)if(pass.pipeline?.compute?.entryPoint==='deform')
   records.push({vertices:new Float32Array(pass.group.entries[0].resource.buffer.data).slice(),
    palette:new Float32Array(pass.group.entries[3].resource.buffer.data).slice(),
    morphs:new Float32Array(pass.group.entries[1].resource.buffer.data).slice(),
    weights:new Float32Array(pass.group.entries[4].resource.buffer.data).slice()});
  submit(commands);
 };
 return records;
}

test('ordinary Basic skin captures feed distinct accepted palettes to the existing GPU deformer',async t=>{
 const {renderer,device}=fixture(t),f=skinnedEnvironment(),generator=new Generator(renderer),records=recordDeformations(device);
 f.mesh.material.dispose();f.mesh.material=new T.MeshBasicMaterial({color:0x80c0ff,side:T.DoubleSide});
 await renderer.init();f.tip.rotation.z=.3;const first=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.tip.rotation.z=-.6;const second=generator.fromScene(f.scene,.1,.1,20,{size:16});
 f.skeleton.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.scene.clear();generator.dispose();
 await drain(renderer);
 assert.ok(records.some(({palette})=>palette[16]===Math.fround(Math.cos(.3))));
 assert.ok(records.some(({palette})=>palette[16]===Math.fround(Math.cos(-.6))));
 assert.equal(renderer._passSnapshots.allocatedBytes,0);first.dispose();second.dispose();
});


test('ordinary GPU skinning cannot silently replace PMREM upload history with unmarked CPU edits',async t=>{
 const {renderer,device}=fixture(t),f=skinnedEnvironment(),generator=new Generator(renderer),records=recordDeformations(device);
 f.mesh.material.dispose();f.mesh.material=new T.MeshBasicMaterial({side:T.DoubleSide});
 const camera=new T.PerspectiveCamera(60,1,.1,20);camera.position.z=5;
 t.after(()=>{generator.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.skeleton.dispose();});
 await renderer.init();renderer.render(f.scene,camera);await drain(renderer);
 assert.ok(records.length>0);assert.ok(records.every(({vertices})=>vertices[0]===.5));
 const originalRecords=records.length;
 const position=f.mesh.geometry.attributes.position,version=position.version;
 position.array[0]=23;
 const submissions=device.submissions.length,paletteVersion=f.skeleton.boneTexture?.version;
 assert.throws(()=>generator.fromScene(f.scene,0,.1,20,{size:16}),/separate WebGL geometry owner/);
 assert.equal(device.submissions.length,submissions);assert.equal(renderer._passSnapshots.allocatedBytes,0);
 assert.equal(f.skeleton.boneTexture?.version,paletteVersion,'rejected capture does not advance the source palette');
 f.mesh.geometry.dispose();const output=generator.fromScene(f.scene,0,.1,20,{size:16});
 t.after(()=>output.dispose());position.array[0]=42;f.mesh.geometry.dispose();generator.dispose();await drain(renderer);
 assert.equal(position.version,version,'disposal alone starts a fresh source upload lifetime');
 assert.ok(records.length>originalRecords);assert.ok(records.slice(originalRecords).every(({vertices})=>vertices[0]===23));
 assert.equal(renderer._passSnapshots.allocatedBytes,0);
});

test('PMREM skinning history blocks a separate ordinary GPU owner until source geometry disposal',async t=>{
 const {renderer,device}=fixture(t),f=skinnedEnvironment(),generator=new Generator(renderer),records=recordDeformations(device);
 f.mesh.material.dispose();f.mesh.material=new T.MeshBasicMaterial({side:T.DoubleSide});
 const camera=new T.PerspectiveCamera(60,1,.1,20);camera.position.z=5;
 t.after(()=>{generator.dispose();f.mesh.geometry.dispose();f.mesh.material.dispose();f.skeleton.dispose();});
 await renderer.init();const output=generator.fromScene(f.scene,0,.1,20,{size:16});t.after(()=>output.dispose());
 await drain(renderer);assert.ok(records.length>0);assert.ok(records.every(({vertices})=>vertices[0]===.5));
 const position=f.mesh.geometry.attributes.position,version=position.version,submitted=records.length;
 position.array[0]=23;
 await assert.rejects(renderer.compileAsync(f.scene,camera),/PMREM-captured attributes require disposal/);
 assert.equal(records.length,submitted,'a conflicting owner submits no deformation work');
 f.mesh.geometry.dispose();renderer.render(f.scene,camera);await drain(renderer);
 assert.ok(records.length>submitted);assert.ok(records.slice(submitted).every(({vertices})=>vertices[0]===23));
 assert.equal(position.version,version);
});

function morphedEnvironment({relative=false,skinned=false,colors=false}={}) {
 const f=skinned?skinnedEnvironment():{scene:new T.Scene(),mesh:new T.Mesh(new T.BoxGeometry(1,2,1),new T.MeshNormalMaterial({side:T.DoubleSide}))};
 const geometry=f.mesh.geometry,position=geometry.attributes.position;
 geometry.morphTargetsRelative=relative;
 const target=position.clone();
 for(let i=0;i<target.count;i++)target.setXYZ(i,relative?.3:position.getX(i)+.3,relative?.1:position.getY(i)+.1,relative?0:position.getZ(i));
 geometry.morphAttributes.position=[target];
 geometry.morphAttributes.normal=[geometry.attributes.normal.clone()];
 if(colors){
  geometry.setAttribute('color',new T.Float32BufferAttribute(new Float32Array(position.count*3).fill(.5),3));
  geometry.morphAttributes.color=[new T.Uint8BufferAttribute(new Uint8Array(position.count*4).fill(128),4,true)];
 }
 f.mesh.updateMorphTargets();f.mesh.morphTargetInfluences[0]=.25;f.mesh.frustumCulled=false;
 if(!skinned)f.scene.add(f.mesh);
 return {...f,geometry,target};
}
function morphOracle() {
 const values=new Map(),owner=WebGLMorphtargets({}, {maxTextureSize:8192}, {});
 return object=>{values.clear();owner.update(object,object.geometry,{getUniforms:()=>({setValue(_gl,name,value){values.set(name,value);}})});
  return {pixels:values.get('morphTargetsTexture').image.data.slice(),base:values.get('morphTargetBaseInfluence'),
   influences:[...values.get('morphTargetInfluences')]};};
}
function assertMorphSnapshot(snapshot,expected) {
 snapshot.install();const mesh=snapshot.root.children[0];
 assert.deepEqual(threeMorphTargets(T,mesh.geometry).texture.image.data,expected.pixels);
 assert.deepEqual(mesh.morphTargetInfluences,expected.influences);
 const base=mesh.geometry.morphTargetsRelative?1:1-mesh.morphTargetInfluences.reduce((sum,value)=>sum+value,0);
 assert.equal(base,expected.base);return mesh;
}

for(const relative of [false,true])test(`PMREM ${relative?'relative':'absolute'} morph captures preserve pinned first-use, count-change and disposal histories`,()=>{
 const f=morphedEnvironment({relative,colors:true}),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(),oracle=morphOracle();
 const capture=()=>owner.capturePMREMScene(f.scene,camera,null);
 const influences=f.mesh.morphTargetInfluences,targets=f.geometry.morphAttributes.position;
 const snapshots=[];
 try{
  const first=capture(),a=oracle(f.mesh);snapshots.push(first);
  f.target.array[0]=77;f.target.needsUpdate=true;influences[0]=.75;
  const second=capture(),b=oracle(f.mesh);snapshots.push(second);
  assert.deepEqual(a.pixels,b.pixels,'source morph atlases ignore attribute needsUpdate at an unchanged count');
  f.geometry.morphAttributes.position=[f.target.clone()];f.geometry.morphAttributes.position[0].array[0]=88;
  const replaced=capture(),c=oracle(f.mesh);snapshots.push(replaced);
  assert.deepEqual(c.pixels,a.pixels,'replacing a same-count target does not refresh the source atlas');
  for(const list of Object.values(f.geometry.morphAttributes))list.push(list[0].clone());
  f.mesh.updateMorphTargets();f.mesh.morphTargetInfluences.splice(0,2,.125,.875);
  const expanded=capture(),d=oracle(f.mesh);snapshots.push(expanded);
  assert.equal(d.pixels[0],88);assert.equal(d.pixels.length,2*a.pixels.length);
  f.geometry.dispose();f.geometry.morphAttributes.position[0].array[0]=99;
  const reused=capture(),e=oracle(f.mesh);snapshots.push(reused);
  assert.equal(e.pixels[0],99);
  f.geometry.dispose();f.mesh.material.dispose();f.scene.clear();
  const view=assertMorphSnapshot(first,a);
  assert.equal(assertMorphSnapshot(second,b),view,'per-mesh identity survives accepted influence changes');
  assert.equal(assertMorphSnapshot(replaced,c).geometry,view.geometry);
  assertMorphSnapshot(expanded,d);assertMorphSnapshot(reused,e);
  assert.equal(influences[0],.75);assert.equal(targets[0],f.target,'public input identities were never replaced');
 }finally{for(const snapshot of snapshots)snapshot.release();owner.dispose();f.geometry.dispose();f.mesh.material.dispose();}
 assert.equal(owner.allocatedBytes,0);
});

test('PMREM morph inputs use source component decoding and keep base-attribute aliases independent',()=>{
 const f=morphedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(),oracle=morphOracle();
 const source=f.geometry.attributes.position;
 f.geometry.morphAttributes.position=[source];
 const normal=new Int16Array(source.count*4);for(let i=0;i<source.count;i++)normal.set([32767,-32768,0,1234],i*4);
 f.geometry.morphAttributes.normal=[new T.InterleavedBufferAttribute(new T.InterleavedBuffer(normal,4),3,0,true)];
 const take=()=>owner.capturePMREMScene(f.scene,camera,null),before=new T.BufferAttribute(new Float32Array(),3).id;
 const first=take(),a=oracle(f.mesh),after=new T.BufferAttribute(new Float32Array(),3).id;
 assert.equal(after,before+1,'private morph inputs do not consume source BufferAttribute IDs');
 source.array[0]=3;source.needsUpdate=true;const second=take(),b=oracle(f.mesh);
 try{
  assertMorphSnapshot(first,a);const mesh=assertMorphSnapshot(second,b);
  assert.equal(mesh.geometry.attributes.position.array[0],3,'ordinary vertex history follows its upload request');
  assert.equal(mesh.geometry.morphAttributes.position[0].array[0],.5,'the aliased morph target retains its first atlas');
  assert.equal(mesh.geometry.morphAttributes.normal[0].getY(0),-1,'normalized source components decode before the f32 boundary');
  assert.equal(f.geometry.morphAttributes.position[0],source);
 }finally{first.release();second.release();owner.dispose();f.geometry.dispose();f.mesh.material.dispose();}
});

test('PMREM hidden and rejected scenes do not establish a morph atlas or retain its bytes',()=>{
 const f=morphedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera(60,1,.1,20),oracle=morphOracle();
 const take=()=>owner.capturePMREMScene(f.scene,camera,null);
 try{
  for(const state of ['hidden','layer','material']){
   f.mesh.visible=state!=='hidden';f.mesh.layers.set(state==='layer'?1:0);f.mesh.material.visible=state!=='material';
   const ignored=take();ignored.install();assert.deepEqual(ignored.root.children[0].geometry.morphAttributes,{});ignored.release();
   f.target.array[0]+=1;
  }
  f.mesh.visible=true;f.mesh.layers.set(0);f.mesh.material.visible=true;
  // The invalid value is encountered after geometry capture. Its failure must
  // roll back a provisional first-use morph history, just like vertex uploads.
  f.mesh.material.unowned=new Map();
  assert.throws(take,/Unsupported mutable PMREM scene field/);delete f.mesh.material.unowned;
  f.target.array[0]=31;
  const accepted=take(),expected=oracle(f.mesh);assert.equal(expected.pixels[0],31);
  assertMorphSnapshot(accepted,expected);accepted.release();
 }finally{owner.dispose();f.geometry.dispose();f.mesh.material.dispose();}
 assert.equal(owner.allocatedBytes,0);
});

test('PMREM rejects custom morph execution and renderer-uniform collisions before retaining a generation',()=>{
 for(const kind of ['component reader','influences','renderer uniform','instanced morph']){
  const f=morphedEnvironment(),owner=createThreePassSnapshots(T),camera=new T.PerspectiveCamera();let calls=0;
  if(kind==='component reader')f.target.getX=()=>{calls++;return 0;};
  else if(kind==='influences')f.mesh.morphTargetInfluences=[{}];
  else if(kind==='renderer uniform'){
   f.mesh.material.dispose();f.mesh.material=new T.ShaderMaterial({uniforms:{morphTargetInfluences:{value:[1]}}});
  }else{
   f.scene.remove(f.mesh);const instanced=new T.InstancedMesh(f.geometry,f.mesh.material,1);
   instanced.setMorphAt(0,f.mesh);f.scene.add(instanced);
  }
  try{
   assert.throws(()=>owner.capturePMREMScene(f.scene,camera,null),error=>error.code==='THREE_PASS_SCENE',kind);
   assert.equal(calls,0);assert.equal(owner.allocatedBytes,0);
  }finally{owner.dispose();f.geometry.dispose();f.mesh.material.dispose();f.scene.children[0].morphTexture?.dispose();}
 }
});

for(const skinned of [false,true])test(`public PMREM queues ${skinned?'skinned ':''}morph poses before initialization and preserves submitted weights after disposal`,async t=>{
 const {renderer,device,release}=fixture(t,{deferred:true}),f=morphedEnvironment({skinned}),generator=new Generator(renderer);
 const program=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
 const reflection=program.compile(f.mesh.material,f.mesh).program.reflection;
 const weights=reflection.uniforms.find(uniform=>uniform.name==='morphTargetInfluences');assert.ok(weights);
 f.mesh.position.z=-3;f.mesh.morphTargetInfluences[0]=.125;
 if(skinned)f.tip.rotation.z=.35;
 const first=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.target.array[0]=99;f.target.needsUpdate=true;f.mesh.morphTargetInfluences[0]=.875;
 if(skinned)f.tip.rotation.z=-.7;
 const second=generator.fromScene(f.scene,.1,.1,20,{size:16});
 f.geometry.dispose();f.mesh.material.dispose();f.skeleton?.dispose();f.scene.clear();generator.dispose();program.dispose();
 release();await drain(renderer);
 const draws=device.snapshots.flat().filter(draw=>draw.pipeline.vertex.module.code.includes('morphTargetInfluences'));
 assert.ok(draws.length>=12,'both accepted generations execute their six face draws');
 const values=draws.map(draw=>{const {group,offsets}=draw.groups.get(0),resource=group.entries[0].resource,bytes=draw.contents.get(resource.buffer);
  return new DataView(bytes.buffer,bytes.byteOffset).getFloat32((offsets[0]??resource.offset??0)+weights.offset,true);});
 assert.ok(values.includes(.125));assert.ok(values.includes(.875));
 const atlases=device.textureWrites.filter(write=>write.texture.format==='rgba32float'&&write.texture.width===f.target.count*2);
 assert.ok(atlases.length>0);assert.ok(atlases.every(write=>new Float32Array(write.bytes.buffer)[0]===Math.fround(.8)),
  'later needsUpdate cannot change an earlier first-use morph atlas');
 assert.equal(renderer.info.f3d.route,'general-webgpu');assert.equal(renderer._passSnapshots.allocatedBytes,0);
 first.dispose();second.dispose();
});

test('ordinary Basic morph captures feed accepted target histories and live weights to the GPU deformer',async t=>{
 const {renderer,device}=fixture(t),f=morphedEnvironment(),generator=new Generator(renderer),records=recordDeformations(device);
 f.mesh.material.dispose();f.mesh.material=new T.MeshBasicMaterial({side:T.DoubleSide});
 await renderer.init();f.mesh.morphTargetInfluences[0]=.125;
 const first=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.target.array[0]=99;f.target.needsUpdate=true;f.mesh.morphTargetInfluences[0]=.875;
 const second=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.geometry.dispose();f.mesh.morphTargetInfluences[0]=.5;
 const reused=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.geometry.dispose();f.mesh.material.dispose();generator.dispose();await drain(renderer);
 for(const weight of [.125,.875,.5])assert.ok(records.some(record=>record.weights[0]===weight),`accepted morph weight ${weight} reaches compute`);
 assert.ok(records.filter(record=>record.weights[0]!==.5).every(record=>record.morphs[0]===Math.fround(.8)-.5),
  'same-count target updates remain stale through both queued deformations');
 assert.ok(records.filter(record=>record.weights[0]===.5).every(record=>record.morphs[0]===98.5),
  'geometry disposal establishes a fresh same-count morph generation');
 assert.equal(renderer._passSnapshots.allocatedBytes,0);first.dispose();second.dispose();reused.dispose();
});

test('ordinary program morph residency cannot be silently replaced by a first PMREM capture',async t=>{
 const {renderer,device}=fixture(t),f=morphedEnvironment(),generator=new Generator(renderer);
 const camera=new T.PerspectiveCamera(60,1,.1,20);camera.position.z=5;
 t.after(()=>{generator.dispose();f.geometry.dispose();f.mesh.material.dispose();});
 await renderer.init();renderer.render(f.scene,camera);await drain(renderer);
 assert.ok(device.textureWrites.some(write=>write.texture.format==='rgba32float'&&write.texture.width===f.target.count*2));
 f.target.array[0]=99;f.target.needsUpdate=true;const submitted=device.submissions.length;
 assert.throws(()=>generator.fromScene(f.scene,0,.1,20,{size:16}),/separate WebGL geometry owner/);
 assert.equal(device.submissions.length,submitted);assert.equal(renderer._passSnapshots.allocatedBytes,0);
 f.geometry.dispose();const previous=device.textureWrites.length,output=generator.fromScene(f.scene,0,.1,20,{size:16});
 f.geometry.dispose();generator.dispose();await drain(renderer);
 const atlases=device.textureWrites.slice(previous).filter(write=>write.texture.format==='rgba32float'&&write.texture.width===f.target.count*2);
 assert.ok(atlases.length>0);assert.ok(atlases.every(write=>new Float32Array(write.bytes.buffer)[0]===99));
 assert.equal(renderer._passSnapshots.allocatedBytes,0);output.dispose();
});
