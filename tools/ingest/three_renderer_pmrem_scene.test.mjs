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
