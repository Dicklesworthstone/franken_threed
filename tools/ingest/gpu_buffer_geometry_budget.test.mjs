import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuBufferGeometry,bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.core.js')));
const geometry=()=>new T.BufferGeometry().setAttribute('position',new T.Float32BufferAttribute([0,0,0,1,0,0,0,1,0],3));

test('initial additional-byte admission precedes buffers, writes and upload callbacks',()=>{
  for(const allowance of [0,35,-1,1.5,Infinity,NaN]){
    const d=geometryDevice(),g=geometry();let calls=0;g.attributes.position.onUpload(()=>calls++);
    assert.throws(()=>createGpuBufferGeometry(d,g,{maxInitialBytes:allowance}));
    assert.equal(d.buffers.length,0);assert.equal(d.writes.length,0);assert.equal(calls,0);
  }
});

test('zero growth budget admits existing uploads but rejects a later stream before any effects',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g,{maxBytes:108,maxInitialBytes:36});
  const p=g.attributes.position;let calls=0;p.onUpload(()=>calls++);
  p.array[0]=.25;p.addUpdateRange(0,1);p.needsUpdate=true;
  gpu.update({maxAdditionalBytes:0});assert.equal(calls,1);assert.equal(gpu.bufferBytes,36);
  p.array[0]=.5;p.addUpdateRange(0,1);p.needsUpdate=true;
  g.setAttribute('normal',new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));
  const writes=d.writes.length;
  assert.throws(()=>gpu.update({maxAdditionalBytes:35}),{code:'GEOMETRY_GPU_LIMIT'});
  assert.equal(d.writes.length,writes);assert.equal(d.buffers.length,1);assert.equal(calls,1);
  assert.deepEqual(p.updateRanges,[{start:0,count:1}]);assert.equal(gpu.failed,false);
  gpu.update({maxAdditionalBytes:36});assert.equal(gpu.bufferBytes,72);assert.equal(calls,2);
  assert.equal(new Float32Array(d.buffers[0].data)[0],.5);assert.equal(p.updateRanges.length,0);
  await gpu.whenIdle();gpu.dispose();
});

test('per-call growth allowance resets, counts retained replacements and charges recreation after disposal',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g,{maxBytes:108,maxInitialBytes:36});
  const old=bufferGeometrySnapshot(gpu,d).vertexBuffers[0];
  g.setAttribute('position',new T.Float32BufferAttribute([0,0,0,2,0,0,0,2,0],3));
  assert.throws(()=>gpu.update({maxAdditionalBytes:0}),{code:'GEOMETRY_GPU_LIMIT'});
  gpu.update();assert.equal(gpu.bufferBytes,72,'source identity history remains charged');
  assert.equal(old.destroyed,false,'replacement does not pretend old residency is free');
  g.dispose();assert.equal(gpu.bufferBytes,0);assert.equal(old.destroyed,true);
  const writes=d.writes.length;
  assert.throws(()=>gpu.update({maxAdditionalBytes:0}),{code:'GEOMETRY_GPU_LIMIT'});
  assert.equal(d.writes.length,writes);gpu.update({maxAdditionalBytes:36});assert.equal(gpu.bufferBytes,36);
  await gpu.whenIdle();gpu.dispose();
});
