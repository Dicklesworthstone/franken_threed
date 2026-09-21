/** Real scene orchestration, GPU deformer and CPU admission. Packed pose fixtures
 * and recording WebGPU calls; controller, renderer, bounds, LOD and shadow seams
 * below are explicit doubles. No browser, WGSL, pixel or glTF-decoding claim. */
import test,{after} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,copyFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {recordingDevice,packedPose,geometry,identity} from './fixtures/animation/gpu_batch_fixture.mjs';
const dir=await mkdtemp(join(tmpdir(),'f3d-scene-batch-'));after(()=>rm(dir,{recursive:true,force:true}));
for(const file of ['animation_scene.mjs','animation_webgpu.mjs','animation_deformer.mjs'])await copyFile(new URL(file,import.meta.url),join(dir,file));
await writeFile(join(dir,'animation_runtime.mjs'),`export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code;}}`);
await writeFile(join(dir,'animation_controller.mjs'),`
export function createAnimationController(pose){return {disposed:false,time:0,
  update(dt){if(!Number.isFinite(dt)||dt<0)throw Error('invalid delta');this.time+=dt;pose.advance?.(dt);},dispose(){this.disposed=true;}};}`);
await writeFile(join(dir,'animation_render.mjs'),`
export class AnimationRenderError extends Error{constructor(code,message){super(message);this.code=code;}}
export async function createGpuAnimationRenderer(device){const meshes=[];return {allocatedBytes:0,failed:false,
  async addMesh(deformer,material){const mesh={deformer,material,disposed:false,dispose(){this.disposed=true;}};meshes.push(mesh);return mesh;},
  render(frame){device.onRender?.();device.events.push('color');device.frames??=[];
    device.frames.push(frame.draws.map(mesh=>({node:mesh.deformer.node,poseVersion:mesh.deformer.poseVersion,world:[...mesh.deformer.worldMatrix]})));},
  async whenIdle(){},dispose(){for(const m of meshes)m.dispose();}};}`);
await writeFile(join(dir,'animation_draw_order.mjs'),`
export function createAnimationDrawOrder(input,{pose}){const entries=input.slice();return {
  boundsBytes:0,viewProjection:null,lastCulling:null,
  updateBounds(){if(entries.some(e=>e.deformer.poseVersion!==pose.version))throw Error('stale bounds inputs');pose.boundsUpdates=(pose.boundsUpdates??0)+1;},
  order(projection,selection){this.viewProjection=projection;const chosen=entries.filter((e,i)=>selection===null||selection.includes(i));
    const visible=chosen.filter(e=>!(pose.culled??[]).includes(e.deformer.node));
    this.lastCulling={poseVersion:pose.version,testedMeshes:chosen.length,culledMeshes:chosen.length-visible.length,submittedDraws:visible.length};
    return visible.map(e=>e.mesh);},dispose(){}};}`);
await writeFile(join(dir,'animation_lod.mjs'),`
export function createAnimationLod(pose,count,options){return {cameraCount:0,
  prepare(camera){const level=camera.position[2]>=10?1:0;const grouped=options.groups[0].levels.flatMap(x=>x.drawIndices);
    const selected=options.groups[0].levels[level].drawIndices;
    return Object.freeze({poseVersion:pose.version,cameraKey:camera.key,drawIndices:Object.freeze(Array.from({length:count},(_,i)=>i).filter(i=>!grouped.includes(i)||selected.includes(i)))});},
  commit(){this.cameraCount++;},resetCamera(){},dispose(){}};}`);
await writeFile(join(dir,'animation_scene_shadow.mjs'),`
export function prepareAnimationSceneShadows(pose){let device,deformers;return {allocatedBytes:0,boundsBytes:0,failed:false,
  async initialize(d,g){device=d;deformers=g;},render(lighting,selection){
    const active=deformers.filter((g,i)=>selection===null||selection.includes(i));
    if(active.some(g=>g.poseVersion!==pose.version))throw Error('stale shadow inputs');
    device.events.push('shadow');device.shadowNodes=active.map(g=>g.node);return {lighting,shadow:{},stats:{poseVersion:pose.version}};},
  async whenIdle(){},dispose(){}};}`);
await writeFile(join(dir,'animation_rigid_geometry.mjs'),`
export const canUseRigidAnimationGeometry=(pose,geometry)=>geometry.node%2===0;
export function createGpuRigidGeometryPool(device,pose){return {bufferBytes:0,meshCount:0,uniqueGeometries:0,failed:false,
  async addMesh(geometry){this.meshCount++;const node=geometry.node,worldMatrix=new Float64Array(16);
    const result={node,worldMatrix,poseVersion:pose.version,disposed:false,failed:false,
      update(){worldMatrix.set(pose.worldMatrices.subarray(node*16,node*16+16));result.poseVersion=pose.version;},
      dispose(){result.disposed=true;}};result.update();return result;},async whenIdle(){},dispose(){}};}`);
const {createGpuAnimationScene}=await import(pathToFileURL(join(dir,'animation_scene.mjs')));
async function fixture(t,count=16,options={}){
  const device=recordingDevice(),pose=packedPose(count),drawables=Array.from({length:count},(_,i)=>({geometry:geometry(i)}));
  const scene=await createGpuAnimationScene(device,pose,drawables,{deformationBatch:true,...options});t.after(()=>scene.dispose());
  device.clear();return {device,pose,scene};
}
const frame=(extra={})=>({colorView:{},depthView:{},viewProjection:identity(),...extra});

test('scene upload batches only changed meshes and publishes current transforms before render',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t,64);assert.equal(s.deformationBatchEnabled,true);assert.equal(s.deformationStats,null);
  const bytes=s.bufferBytes;assert.equal(s.deformationInputCacheBytes,64*68);p.version++;p.morphWeights[7]=0.5;p.jointMatrices[51*16+14]=2;
  assert.equal(s.upload(),s);assert.equal(s.deformationStats.dispatches,2);assert.equal(s.deformationStats.skippedMeshes,62);
  assert.equal(s.deformationStats.poseVersion,1);assert.equal(s.deformationStats.rigidMeshes,0);assert.equal(d.submissions.length,1);
  assert.equal(s.render(frame()),s);assert.ok(d.frames[0].every(x=>x.poseVersion===1));assert.deepEqual(d.events.slice(-2),['submit','color']);
  assert.equal(s.bufferBytes,bytes);assert.ok(Object.isFrozen(s.deformationStats));await s.whenIdle();
});

test('controller update retains its boundary while a transform-only pose performs no compute',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t);
  p.advance=dt=>{p.version++;for(let i=0;i<p.nodeCount;i++)p.worldMatrices[i*16+12]+=dt;};
  assert.equal(s.update(0.5),s);assert.equal(s.controller.time,0.5);assert.equal(s.deformationStats.submissions,0);
  assert.equal(d.encoders,0);assert.equal(d.writes.length,0);s.render(frame());assert.equal(d.frames[0][3].world[12],12.5);
  assert.ok(s.deformers.every(g=>g.poseVersion===p.version));await s.whenIdle();
});

test('legacy individual submission mode remains available and is still the default',async t=>{
  for(const deformationBatch of [false,undefined]){
    const {device:d,scene:s}=await fixture(t,8,{deformationBatch});s.upload();
    assert.equal(s.deformationBatchEnabled,false);assert.equal(d.submissions.length,8);assert.equal(s.deformationStats,null);await s.whenIdle();
  }
});

test('direct pose changes still reject render until upload, even when shader inputs are unchanged',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t);p.version++;
  assert.throws(()=>s.render(frame()),{code:'ANIMATION_SCENE_STALE'});assert.equal(d.submissions.length,0);
  s.upload();s.render(frame());assert.equal(s.poseVersion,1);assert.equal(s.deformationStats.dispatches,0);await s.whenIdle();
});

test('changed-input culling and shadow consumers receive current versions and independent subsets',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t,4,{frustumCulling:true,shadow:{lightIndex:0}});
  p.version++;p.culled=[2,3];p.morphWeights[2]=0.75;s.upload();assert.equal(p.boundsUpdates,1);
  s.render(frame({lighting:{}}));assert.deepEqual(d.shadowNodes,[0,1,2,3]);assert.deepEqual(d.frames[0].map(x=>x.node),[0,1]);
  assert.deepEqual(d.events.slice(-3),['submit','shadow','color']);assert.equal(s.shadowStats.poseVersion,1);assert.equal(s.cullingStats.culledMeshes,2);await s.whenIdle();
});

test('two same-pose cameras can select different LOD members without render-time deformation',async t=>{
  const lod={groups:[{node:0,levels:[{distance:0,drawIndices:[0,1]},{distance:10,drawIndices:[2]}]}]};
  const {device:d,pose:p,scene:s}=await fixture(t,4,{lod,shadow:{lightIndex:0}});
  p.version++;p.morphWeights[2]=1;s.upload();const submits=d.submissions.length;
  s.render(frame({lodCamera:{position:[0,0,1],key:'near'}}));assert.deepEqual(d.frames[0].map(x=>x.node),[0,1,3]);
  s.render(frame({lodCamera:{position:[0,0,50],key:'far'}}));assert.deepEqual(d.frames[1].map(x=>x.node),[2,3]);assert.deepEqual(d.shadowNodes,[2,3]);
  assert.equal(d.submissions.length,submits);assert.equal(s.lodStats.cameraKey,'far');assert.ok(s.deformers.every(g=>g.poseVersion===p.version));await s.whenIdle();
});

test('rigid handles stay outside the compute cohort and still update their transforms',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t,8,{rigidGeometry:true});
  p.version++;for(let i=0;i<8;i++){p.worldMatrices[i*16+12]+=7;p.morphWeights[i]=0.5;}
  s.upload();assert.equal(s.deformationStats.rigidMeshes,4);assert.equal(s.deformationStats.meshes,4);assert.equal(s.deformationStats.dispatches,4);
  assert.equal(d.submissions.length,1);s.render(frame());assert.equal(d.frames[0][0].world[12],7);assert.ok(s.deformers.every(g=>g.poseVersion===1));await s.whenIdle();
});

test('an all-rigid batch requires no compute registry members or queue submission',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t,1,{rigidGeometry:true});p.version++;p.worldMatrices[12]=42;s.upload();
  assert.equal(s.deformationStats.meshes,0);assert.equal(s.deformationStats.rigidMeshes,1);assert.equal(d.encoders,0);s.render(frame());assert.equal(d.frames[0][0].world[12],42);await s.whenIdle();
});

test('bad CPU advance preserves the last successful upload but invalid GPU inputs terminate the scene',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t);s.upload();const previous=s.deformationStats;
  assert.throws(()=>s.update(NaN));assert.equal(s.deformationStats,previous);assert.equal(s.failed,false);
  p.version++;p.morphWeights[15]=NaN;assert.throws(()=>s.upload(),{code:'ANIMATION_GPU_VALUE'});
  assert.equal(s.failed,true);assert.equal(s.deformationStats,null);assert.equal(s.bufferBytes,0);assert.equal(s.deformationInputCacheBytes,0);
  assert.ok(d.buffers.every(b=>b.destroyed===1));assert.equal(p.disposed,false);
});

test('submission failure prevents a partial scene frame and releases all compute resources',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t),error=Error('submit failed');p.version++;p.morphWeights.fill(1);d.onSubmit=()=>{throw error;};
  assert.throws(()=>s.upload(),e=>e===error);assert.throws(()=>s.render(frame()),e=>e===error);
  assert.equal(s.failed,true);assert.equal(s.deformationStats,null);assert.equal(s.controller.disposed,true);assert.ok(d.buffers.every(b=>b.destroyed===1));assert.equal(p.disposed,false);
});

test('async batch errors surface through scene completion and clear statistics on terminal release',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t);p.version++;p.morphWeights[0]=1;d.scopeResult={message:'bad compute'};
  s.upload();assert.equal(s.deformationStats.dispatches,1);await assert.rejects(s.whenIdle(),{code:'ANIMATION_GPU_DEVICE'});
  assert.equal(s.deformationStats,null);assert.equal(s.failed,true);assert.ok(d.buffers.every(b=>b.destroyed===1));
});

test('rejected color frames preserve successful upload stats and do not repeat compute',async t=>{
  const {device:d,pose:p,scene:s}=await fixture(t);p.version++;p.morphWeights[0]=1;s.upload();const previous=s.deformationStats,submits=d.submissions.length;
  d.onRender=()=>{throw Error('bad frame');};assert.throws(()=>s.render(frame()),/bad frame/);assert.equal(s.failed,false);assert.equal(s.deformationStats,previous);
  d.onRender=null;s.render(frame());assert.equal(d.submissions.length,submits);await s.whenIdle();s.dispose();assert.equal(s.deformationStats,null);
});

test('invalid scene batch policy fails before controller or GPU resource creation',async()=>{
  for(const value of [null,1,'true']){const d=recordingDevice(),p=packedPose(1);await assert.rejects(createGpuAnimationScene(d,p,[{geometry:geometry(0)}],{deformationBatch:value}),{code:'ANIMATION_SCENE_BATCH'});assert.equal(d.buffers.length,0);}
});
