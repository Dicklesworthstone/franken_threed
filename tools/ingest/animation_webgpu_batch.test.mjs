/** Real GPU deformer and CPU admission; packed pose fixtures, recording device,
 * and error-class-only runtime double. No WGSL execution or browser claim. */
import test,{after} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,copyFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {recordingDevice,packedPose,geometry,deferred} from './fixtures/animation/gpu_batch_fixture.mjs';
const dir=await mkdtemp(join(tmpdir(),'f3d-gpu-batch-'));
after(()=>rm(dir,{recursive:true,force:true}));
for(const file of ['animation_webgpu.mjs','animation_deformer.mjs'])await copyFile(new URL(file,import.meta.url),join(dir,file));
await writeFile(join(dir,'animation_runtime.mjs'),`export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code;}}`);
const {createGpuAnimationDeformer,updateGpuAnimationDeformers:batch}=await import(pathToFileURL(join(dir,'animation_webgpu.mjs')));
async function fixture(t,count=4,{device=recordingDevice(),pose=packedPose(count),flat=false}={}){
  const meshes=[];t.after(()=>{for(const mesh of meshes)mesh.dispose();});
  for(let i=0;i<count;i++)meshes.push(await createGpuAnimationDeformer(device,pose,geometry(i,{flat:flat&&i%2===0}),{label:`mesh-${i}`}));
  device.clear();return {device,pose,meshes};
}
const drain=meshes=>Promise.all(meshes.map(m=>m.whenIdle()));

test('64 deformations use one encoder, submit and acknowledgement with per-mesh inputs',async t=>{
  const {device:d,pose:p,meshes}=await fixture(t,64),buffers=d.buffers.length;
  p.version++;
  for(let i=0;i<64;i++){p.jointMatrices[i*16+14]=i/2;p.morphWeights[i]=i/64;}
  d.onSubmit=()=>assert.ok(meshes.every(m=>m.poseVersion===0),'publication follows submit');
  const stats=batch(meshes);assert.deepEqual(stats,{meshes:64,submissions:1,dispatches:64,bufferWrites:128,uploadedBytes:4352});
  assert.equal(d.submissions.length,1);assert.equal(d.submissions[0].length,64);assert.equal(d.encoders,1);assert.equal(d.acks,1);
  for(let i=0;i<64;i++){
    const command=d.submissions[0][i];assert.equal(new Float32Array(command.palette.buffer)[14],i/2);
    assert.equal(new Float32Array(command.weights.buffer)[0],i/64);assert.equal(meshes[i].poseVersion,1);assert.equal(meshes[i].version,1);
  }
  assert.equal(d.events.at(-1),'submit');assert.equal(d.buffers.length,buffers);assert.ok(Object.isFrozen(stats));await drain(meshes);
});

test('flat-normal passes remain ordered after their own deformation inside one command buffer',async t=>{
  const {device:d,meshes}=await fixture(t,4,{flat:true});
  const stats=batch(meshes);assert.equal(stats.dispatches,6);
  assert.deepEqual(d.submissions[0].map(p=>[p.label,p.entryPoint]),[
    ['mesh-0','deform'],['mesh-0/flat-normals','flat_normals'],['mesh-1','deform'],
    ['mesh-2','deform'],['mesh-2/flat-normals','flat_normals'],['mesh-3','deform']]);await drain(meshes);
});

test('120 successive batched poses match individual queue input snapshots exactly',async t=>{
  const a=await fixture(t,8),b=await fixture(t,8);let seed=90210;
  const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
  for(let round=0;round<120;round++){
    a.pose.version++;b.pose.version++;
    for(let i=0;i<8;i++){
      const z=(random()-0.5)*10,w=(random()-0.5)*4,x=(random()-0.5)*100;
      for(const p of [a.pose,b.pose]){p.jointMatrices[i*16+14]=z;p.morphWeights[i]=w;p.worldMatrices[i*16+12]=x;}
    }
    a.device.clear();b.device.clear();for(const m of a.meshes)assert.equal(m.update(),m);batch(b.meshes);
    assert.deepEqual(b.device.submissions.flat(),a.device.submissions.flat());assert.deepEqual(b.device.writes,a.device.writes);
    for(let i=0;i<8;i++){assert.deepEqual(b.meshes[i].worldMatrix,a.meshes[i].worldMatrix);assert.equal(b.meshes[i].version,a.meshes[i].version);}
  }
  await drain([...a.meshes,...b.meshes]);
});

test('invalid later pose inputs fail before all queue work and remain retryable',async t=>{
  const {device:d,pose:p,meshes}=await fixture(t);p.version++;p.morphWeights[3]=NaN;
  assert.throws(()=>batch(meshes),{code:'ANIMATION_GPU_VALUE'});assert.equal(d.encoders,0);assert.equal(d.writes.length,0);
  assert.ok(meshes.every(m=>m.poseVersion===0&&!m.failed));assert.equal(d.scopes.length,0);
  p.morphWeights[3]=0.5;batch(meshes);assert.equal(d.submissions.length,1);await drain(meshes);
});

test('duplicate, foreign, mixed-device and oversized batches are refused before acquisition',async t=>{
  const a=await fixture(t,1),b=await fixture(t,1),m=a.meshes[0];
  for(const list of [[m,m],[{}],[m,{...m}],new Array(4097).fill(m),[m,b.meshes[0]],[m,undefined],null])
    assert.throws(()=>batch(list),{code:'ANIMATION_GPU_BATCH'});
  assert.equal(a.device.encoders,0);assert.equal(b.device.encoders,0);m.update();await m.whenIdle();
});

test('empty batches do not allocate, enqueue or require a device',()=>{
  assert.deepEqual(batch([]),{meshes:0,submissions:0,dispatches:0,bufferWrites:0,uploadedBytes:0});
});

test('independent poses on the same device keep independent version stamps',async t=>{
  const a=await fixture(t,1),b=await fixture(t,1,{device:a.device});a.pose.version=4;b.pose.version=9;
  a.pose.worldMatrices[12]=11;b.pose.worldMatrices[12]=22;
  batch([...a.meshes,...b.meshes]);assert.equal(a.meshes[0].poseVersion,4);assert.equal(b.meshes[0].poseVersion,9);
  assert.equal(a.meshes[0].worldMatrix[12],11);assert.equal(b.meshes[0].worldMatrix[12],22);await drain([...a.meshes,...b.meshes]);
});

test('a later pose getter cannot silently invalidate an earlier prepared pose',async t=>{
  const a=await fixture(t,1),b=await fixture(t,1,{device:a.device});
  Object.defineProperty(b.pose,'version',{configurable:true,get(){a.pose.version++;return 1;}});
  assert.throws(()=>batch([...a.meshes,...b.meshes]),{code:'ANIMATION_GPU_CHANGED'});assert.equal(a.device.encoders,0);
  Object.defineProperty(b.pose,'version',{value:1,writable:true});batch([...a.meshes,...b.meshes]);await drain([...a.meshes,...b.meshes]);
});

test('all batch members are locked against update/disposal reentry, then released',async t=>{
  const {device:d,meshes}=await fixture(t);
  d.onEncoder=()=>{for(const m of meshes)for(const op of [()=>m.update(),()=>m.dispose()])assert.throws(op,{code:'ANIMATION_REENTRANT'});};
  batch(meshes);d.onEncoder=null;meshes[2].update();await drain(meshes);
});

test('failure acquiring a disposed later member unlocks previously acquired live owners',async t=>{
  const {meshes}=await fixture(t,2);meshes[1].dispose();
  assert.throws(()=>batch(meshes),{code:'ANIMATION_GPU_DISPOSED'});assert.equal(meshes[0].failed,false);
  meshes[0].update();await meshes[0].whenIdle();
});

for(const point of ['onPass','onFinish','onWrite','onSubmit'])test(`${point} failure is terminal for the entire attempted batch`,async t=>{
  const {device:d,pose:p,meshes}=await fixture(t),error=Error(point);p.version++;
  d[point]=()=>{throw error;};assert.throws(()=>batch(meshes),e=>e===error);
  assert.ok(meshes.every(m=>m.failed&&m.poseVersion===0));assert.ok(d.buffers.every(b=>b.destroyed===1));
  assert.equal(p.disposed,false);assert.equal(d.scopes.length,0);assert.equal(d.submissions.length,0);
  if(point==='onPass'||point==='onFinish')assert.equal(d.writes.length,0);
  assert.throws(()=>meshes[0].update(),e=>e===error);
});

test('asynchronous scope rejection reaches every participant through whenIdle',async t=>{
  const {device:d,meshes}=await fixture(t);d.scopeResult={message:'invalid batch'};
  batch(meshes);for(const m of meshes)await assert.rejects(m.whenIdle(),{code:'ANIMATION_GPU_DEVICE'});
  assert.ok(meshes.every(m=>m.failed));assert.ok(d.buffers.every(b=>b.destroyed===1));
});

test('a later successful batch cannot hide an earlier unresolved individual error scope',async t=>{
  const {device:d,meshes}=await fixture(t,2),gate=deferred();d.scopeResult=gate.promise;
  meshes[0].update();batch(meshes);gate.resolve({message:'earlier invalid update'});
  await assert.rejects(meshes[0].whenIdle(),{code:'ANIMATION_GPU_DEVICE'});
  await meshes[1].whenIdle();assert.equal(meshes[1].failed,false);
});

test('submission completion failure poisons all participants, never the borrowed pose',async t=>{
  const {device:d,pose:p,meshes}=await fixture(t),gate=deferred(),error=Error('queue failed');d.ackResult=gate.promise;
  batch(meshes);gate.reject(error);for(const m of meshes)await assert.rejects(m.whenIdle(),e=>e===error);
  assert.ok(meshes.every(m=>m.failed));assert.equal(p.disposed,false);
});

test('device loss while a batch is pending terminates all live owners and releases storage',async t=>{
  const {device:d,meshes}=await fixture(t);d.ackResult=deferred().promise;batch(meshes);
  d.loss.resolve({message:'lost'});for(const m of meshes)await assert.rejects(m.whenIdle(),{code:'ANIMATION_GPU_LOST'});
  assert.ok(d.buffers.every(b=>b.destroyed===1));
});


test('publication does not invoke an overridden setter on an exposed world view',async t=>{
  const {pose:p,meshes}=await fixture(t,2);p.version++;p.worldMatrices[12]=42;
  for(const m of meshes)m.worldMatrix.set=()=>{throw Error('caller setter invoked');};
  batch(meshes);assert.ok(meshes.every(m=>m.poseVersion===1));assert.equal(meshes[0].worldMatrix[12],42);await drain(meshes);
});
