import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuInstanceAttributes,instanceAttributesSnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const T=await import(pathToFileURL(path.join(root,'build/three.core.js')));
const {WebGLAttributes}=await import(pathToFileURL(path.join(root,'src/renderers/webgl/WebGLAttributes.js')));
const source=(n=4)=>new T.InstancedMesh(new T.BufferGeometry(),new T.MeshBasicMaterial(),n);
function oracle(){
  let bound;
  return WebGLAttributes({createBuffer:()=>({}),bindBuffer(_t,b){bound=b;},
    bufferData(_t,a){bound.bytes=new Uint8Array(a.buffer,a.byteOffset,a.byteLength).slice();},
    bufferSubData(_t,offset,a,start=0,count=0){
      bound.bytes.set(new Uint8Array(a.buffer,a.byteOffset+4*start,4*(count||a.length-start)),offset);
    },deleteBuffer(b){b.deleted=true;}});
}
const contents=(owner,d)=>instanceAttributesSnapshot(owner,d).vertexBuffers.map(b=>new Uint8Array(b.data));

test('instance matrices and colors follow the complete pinned WebGLAttributes uploader over 80 frames',async()=>{
  const a=source(),b=source(),d=geometryDevice(),gl=oracle();
  a.setColorAt(0,new T.Color());b.setColorAt(0,new T.Color());
  const events=[[],[]];
  for(const [i,s] of [a,b].entries())for(const field of ['instanceMatrix','instanceColor']){
    s[field].addUpdateRange(0,1);
    s[field].onUpload(function(){events[i].push([field,this.version,this.updateRanges.map(r=>({...r}))]);});
  }
  const owner=createGpuInstanceAttributes(d,a),buffers=instanceAttributesSnapshot(owner,d).vertexBuffers;
  for(const field of ['instanceMatrix','instanceColor'])gl.update(b[field],0);
  let state=8128;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state;};
  for(let frame=0;frame<80;frame++){
    for(const field of ['instanceMatrix','instanceColor']){
      const x=a[field],y=b[field];
      for(let i=0;i<x.array.length;i++)x.array[i]=y.array[i]=(random()%1000)/100;
      x.clearUpdateRanges();y.clearUpdateRanges();
      if(frame%4){
        // Initial full upload, source +1 gap merging, zero-length tail uploads,
        // unrelated CPU changes and no-upload frames all retain their histories.
        if(frame%3)for(const start of [5,3,2])for(const z of [x,y])z.addUpdateRange(start,frame%5?1:0);
        x.needsUpdate=true;y.needsUpdate=true;
      }
    }
    a.count=frame%5;owner.update();
    for(const [i,field] of ['instanceMatrix','instanceColor'].entries()){
      gl.update(b[field],0);
      assert.deepEqual(contents(owner,d)[i],gl.get(b[field]).buffer.bytes,`${frame}/${field}`);
      assert.deepEqual(a[field].updateRanges,b[field].updateRanges);
    }
    assert.deepEqual(events[0],events[1]);
    assert.equal(instanceAttributesSnapshot(owner,d).instanceCount,a.count);
    assert.deepEqual(instanceAttributesSnapshot(owner,d).vertexBuffers,buffers);
  }
  assert.equal(owner.bufferBytes,4*(64+12));assert.equal(owner.shadowBytes,owner.bufferBytes);
  await owner.whenIdle();owner.dispose();
});

test('instances are branded, device-bound and independent from shared source geometry lifetime',()=>{
  const d=geometryDevice(),s=source(),owner=createGpuInstanceAttributes(d,s),first=owner.vertexBuffer;
  assert.throws(()=>instanceAttributesSnapshot({},d),{code:'GEOMETRY_GPU_SHAPE'});
  assert.throws(()=>instanceAttributesSnapshot(owner,geometryDevice()),{code:'GEOMETRY_GPU_DEVICE'});
  s.geometry.dispose();assert.equal(first.destroyed,false);
  s.dispose();assert.equal(first.destroyed,true);assert.equal(owner.bufferBytes,0);
  assert.throws(()=>instanceAttributesSnapshot(owner,d),{code:'GEOMETRY_GPU_RELEASED'});
  s.instanceMatrix.array[12]=7;owner.update();assert.notEqual(owner.vertexBuffer,first);
  assert.equal(new Float32Array(owner.vertexBuffer.data)[12],7);assert.equal(owner.disposed,false);
  owner.dispose();assert.equal(s.instanceMatrix.array[12],7);
});

test('live counts are bounded and do not upload, resize or repack instance storage',()=>{
  const d=geometryDevice(),s=source(),owner=createGpuInstanceAttributes(d,s),count=d.writes.length;
  for(const value of [0,4,1,3]){s.count=value;assert.equal(instanceAttributesSnapshot(owner,d).instanceCount,value);}
  assert.equal(d.writes.length,count);
  for(const value of [-1,5,Infinity,1.5]){
    s.count=value;assert.throws(()=>instanceAttributesSnapshot(owner,d),{code:'GEOMETRY_GPU_SHAPE'});
    assert.throws(()=>owner.update(),{code:'GEOMETRY_GPU_SHAPE'});assert.equal(d.writes.length,count);
  }
  s.count=0;owner.dispose();
});

test('matrix and color admission, budget and resize failures happen before either stream uploads',()=>{
  const d=geometryDevice(),s=source(),owner=createGpuInstanceAttributes(d,s,{maxBytes:304,maxAttributes:2});
  s.instanceMatrix.array[12]=8;s.instanceMatrix.needsUpdate=true;
  s.instanceColor=new T.InstancedBufferAttribute(new Float32Array(9),3);
  let n=d.writes.length;
  assert.throws(()=>owner.update(),{code:'GEOMETRY_GPU_SHAPE'});assert.equal(d.writes.length,n);
  s.instanceColor=new T.InstancedBufferAttribute(new Float32Array(12).fill(1),3);owner.update();
  assert.equal(new Float32Array(owner.vertexBuffer.data)[12],8);assert.equal(owner.bufferBytes,304);
  s.instanceMatrix=new T.InstancedBufferAttribute(new Float32Array(64),16);
  n=d.writes.length;assert.throws(()=>owner.update(),{code:'GEOMETRY_GPU_LIMIT'});assert.equal(d.writes.length,n);
  assert.equal(owner.failed,false);owner.dispose();
});

test('instance shape refuses silent divisors, quantization, per-instance morphs and malformed storage',()=>{
  for(const mutate of [s=>s.instanceMatrix.meshPerAttribute=2,s=>s.instanceMatrix.normalized=true,
      s=>s.instanceMatrix.array=new Uint32Array(64),s=>s.instanceMatrix.count=3,
      s=>s.morphTexture={},s=>s.instanceMatrix=new T.BufferAttribute(new Float32Array(64),16)]){
    const d=geometryDevice(),s=source();mutate(s);
    assert.throws(()=>createGpuInstanceAttributes(d,s));assert.equal(d.writes.length,0);assert.equal(d.buffers.length,0);
  }
});

test('instance upload callback exceptions retain successful writes and source version ordering',()=>{
  const s=source(),d=geometryDevice(),owner=createGpuInstanceAttributes(d,s),a=s.instanceMatrix;
  const error=new Error('callback');a.array[12]=9;a.addUpdateRange(12,1);a.needsUpdate=true;
  a.onUpload(()=>{throw error;});assert.throws(()=>owner.update(),e=>e===error);
  assert.equal(new Float32Array(owner.vertexBuffer.data)[12],9);assert.equal(a.updateRanges.length,0);
  a.array[13]=8;a.onUpload(()=>{});owner.update();assert.equal(new Float32Array(owner.vertexBuffer.data)[13],8);
  assert.equal(owner.failed,false);owner.dispose();
});

test('instance ownership observes queue failure, source disposal and device loss',async()=>{
  for(const mode of ['write','scope','completion','loss']){
    const d=geometryDevice(),s=source(),owner=createGpuInstanceAttributes(d,s);await owner.whenIdle();
    if(mode==='loss')d.lose();
    else if(mode==='completion'){d.completion=Promise.reject(new Error('queue'));d.completion.catch(()=>{});}
    else {s.instanceMatrix.needsUpdate=true;
      if(mode==='write'){d.writeError=new Error('write');assert.throws(()=>owner.update(),/write/);}
      else {d.scopeError={message:'validation'};owner.update();}}
    await assert.rejects(owner.whenIdle());assert.equal(owner.failed,true);assert.equal(owner.bufferBytes,0);
    assert.ok(d.buffers.every(b=>b.destroyed));owner.dispose();
  }
});

test('disposal settles an instance wait even if the borrowed queue never completes',async()=>{
  const d=geometryDevice(),s=source(),owner=createGpuInstanceAttributes(d,s);
  d.completion=new Promise(()=>{});const waiting=owner.whenIdle();owner.dispose();
  await assert.rejects(waiting,{code:'GEOMETRY_GPU_DISPOSED'});
});
