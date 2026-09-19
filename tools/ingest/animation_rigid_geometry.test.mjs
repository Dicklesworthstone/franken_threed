import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuRigidGeometryPool,canUseRigidAnimationGeometry} from './animation_rigid_geometry.mjs';
// Real packing, sharing, ownership and transform publication; only WebGPU's
// device boundary records operations. No native pixel or throughput assertion.
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function pose(n=2) {
  const p={nodeCount:n,instances:[],version:0,disposed:false,worldMatrices:new Float64Array(n*16),
    morphOffsets:new Uint32Array(n+1),morphWeights:new Float64Array(0)};
  for(let i=0;i<n;i++){p.worldMatrices.set(I(),i*16);p.worldMatrices[i*16+12]=i*10;}return p;
}
const geometry=(node=0)=>({node,positions:new Float64Array([0,0,0,1,0,0,0,1,0]),normals:new Float64Array([0,0,1,0,0,1,0,0,1]),morphTargets:[]});
function device() {
  const buffers=[],pops=[];let depth=0,lose,failScope=null,waitScopes=null,complete=async()=>{};
  const d={limits:{maxBufferSize:2**26},lost:new Promise(r=>{lose=r;}),
    pushErrorScope(){depth++;},popErrorScope(){depth--;pops.push(depth);return waitScopes?waitScopes():Promise.resolve(failScope);},
    createBuffer(desc){const data=new ArrayBuffer(desc.size);const b={...desc,data,destroyed:0,mapped:true,
      getMappedRange:()=>data,unmap(){this.mapped=false;},destroy(){this.destroyed++;}};buffers.push(b);return b;},
    queue:{onSubmittedWorkDone:()=>complete(),writeBuffer(){assert.fail('rigid updates must not upload');},submit(){assert.fail('rigid registration/update must not submit compute');}},
    createComputePipelineAsync(){assert.fail('rigid geometry has no compute pipeline');},
  };
  return {d,buffers,pops,lose,get depth(){return depth;},scopeError(e){failScope=e;},holdScopes(fn){waitScopes=fn;},completion(fn){complete=fn;}};
}
const code=c=>({code:'ANIMATION_RIGID_'+c});
test('uploads the existing 40-byte position/normal/tangent ABI without compute resources',async()=>{
  const g=device(),p=pose(),pool=createGpuRigidGeometryPool(g.d,p),input=geometry();
  input.tangents=new Float64Array([1,0,0,-1,1,0,0,-1,1,0,0,-1]);
  const a=await pool.addMesh(input);assert.equal(g.buffers.length,1);assert.equal(pool.bufferBytes,120);
  assert.deepEqual([...new Float32Array(g.buffers[0].data)],[0,0,0,0,0,1,1,0,0,-1,1,0,0,0,0,1,1,0,0,-1,0,1,0,0,0,1,1,0,0,-1]);
  assert.equal(a.vertexLayout.arrayStride,40);assert.deepEqual(a.vertexLayout.attributes.map(a=>a.shaderLocation),[0,1,2]);
  assert.equal(a.bufferBytes,0);assert.equal(a.sharedBufferBytes,120);assert.equal(g.buffers[0].usage,36);assert.equal(g.buffers[0].mapped,false);
  assert.equal(g.depth,0);await a.whenIdle();pool.dispose();assert.equal(g.buffers[0].destroyed,1);
});
test('one thousand rigid instances share one immutable GPU allocation and update distinct worlds',async()=>{
  const g=device(),p=pose(1000),pool=createGpuRigidGeometryPool(g.d,p,{maxBytes:120,maxMeshes:1000}),handles=[];
  for(let node=0;node<1000;node++)handles.push(await pool.addMesh(geometry(node),{maxAdditionalBytes:node?0:120}));
  assert.equal(g.buffers.length,1);assert.equal(pool.uniqueGeometries,1);assert.equal(pool.meshCount,1000);assert.equal(pool.bufferBytes,120);
  assert.ok(handles.every(a=>a.vertexBuffer===handles[0].vertexBuffer));assert.notEqual(handles[0].worldMatrix,handles[1].worldMatrix);
  p.version++;for(let node=0;node<1000;node++){p.worldMatrices[node*16+12]+=3;handles[node].update();assert.equal(handles[node].worldMatrix[12],node*10+3);assert.equal(handles[node].poseVersion,1);}
  assert.equal(g.buffers.length,1);pool.dispose();assert.equal(g.buffers[0].destroyed,1);assert.equal(pool.bufferBytes,0);assert.ok(handles.every(a=>a.disposed));
});
test('exact byte comparison distinguishes colliding geometry hashes',async()=>{
  const g=device(),p=pose(),pool=createGpuRigidGeometryPool(g.d,p),a=geometry(),b=geometry(1);
  // FNV word hashes of [0,0] and the f32 bits [2,-2] collide at the
  // same initial seed. Every remaining packed word is identical.
  b.positions[0]=2;b.positions[1]=-2;
  const first=await pool.addMesh(a),second=await pool.addMesh(b);
  assert.notEqual(first.vertexBuffer,second.vertexBuffer);assert.equal(pool.uniqueGeometries,2);pool.dispose();
});
test('sharing uses rounded GPU bytes, not f64 identity, and retains attribute layout distinctions',async()=>{
  const g=device(),pool=createGpuRigidGeometryPool(g.d,pose()),a=geometry(),b=geometry(1);a.positions[3]=1+1e-10;
  const first=await pool.addMesh(a),second=await pool.addMesh(b);assert.equal(first.vertexBuffer,second.vertexBuffer);
  const noNormals=geometry();delete noNormals.normals;const third=await pool.addMesh(noNormals);
  assert.notEqual(third.vertexBuffer,first.vertexBuffer);assert.deepEqual(third.vertexLayout.attributes.map(a=>a.shaderLocation),[0]);pool.dispose();
});
test('signed zero bits remain distinct and source edits cannot mutate retained comparison data',async()=>{
  const g=device(),pool=createGpuRigidGeometryPool(g.d,pose()),a=geometry(),first=await pool.addMesh(a);
  a.positions.fill(99);const second=await pool.addMesh(geometry(1));assert.equal(first.vertexBuffer,second.vertexBuffer);
  const negative=geometry();negative.positions[0]=-0;const third=await pool.addMesh(negative);assert.notEqual(third.vertexBuffer,first.vertexBuffer);pool.dispose();
});
test('disposing a borrower preserves siblings; last borrower releases the buffer exactly once',async()=>{
  const g=device(),pool=createGpuRigidGeometryPool(g.d,pose()),a=await pool.addMesh(geometry()),b=await pool.addMesh(geometry(1));
  a.dispose();a.dispose();assert.equal(g.buffers[0].destroyed,0);assert.equal(pool.meshCount,1);b.update();
  b.dispose();assert.equal(pool.bufferBytes,0);assert.equal(pool.uniqueGeometries,0);assert.equal(g.buffers[0].destroyed,1);
  const c=await pool.addMesh(geometry());assert.notEqual(c.vertexBuffer,a.vertexBuffer);pool.dispose();c.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});
test('world publication is atomic, versioned, and independent of vertex allocation',async()=>{
  const g=device(),p=pose(),pool=createGpuRigidGeometryPool(g.d,p),a=await pool.addMesh(geometry());const identity=a.worldMatrix,before=identity.slice();
  p.worldMatrices[0]=Infinity;p.version=1;assert.throws(()=>a.update(),code('VALUE'));assert.deepEqual(a.worldMatrix,before);assert.equal(a.poseVersion,0);assert.equal(a.version,0);
  p.worldMatrices.set([-2,0,0,0,0,3,0,0,1,0,1,0,5,6,7,1]);a.update();assert.equal(a.worldMatrix,identity);assert.equal(a.worldMatrix[0],-2);assert.equal(a.worldMatrix[12],5);
  assert.equal(a.poseVersion,1);assert.equal(a.version,1);assert.equal(g.buffers.length,1);pool.dispose();
});
for(const change of [g=>{g.morphTargets=[{positions:new Float64Array(9)}];},g=>{g.joints=[];},g=>{g.flatNormals=true;},g=>{g.customDisplacement=true;}])
  test('dynamic or unsupported geometry is not silently put on the rigid path',async()=>{
    const d=device(),p=pose(),pool=createGpuRigidGeometryPool(d.d,p),g=geometry();change(g);
    assert.equal(canUseRigidAnimationGeometry(p,g),false);await assert.rejects(pool.addMesh(g),code('DYNAMIC'));assert.equal(d.buffers.length,0);pool.dispose();
  });
test('skinned nodes and nodes with morph bindings retain ordinary deformation',async()=>{
  const d=device(),p=pose(),pool=createGpuRigidGeometryPool(d.d,p);p.instances.push({node:0});
  await assert.rejects(pool.addMesh(geometry()),code('DYNAMIC'));p.instances=[];p.morphOffsets[1]=1;
  await assert.rejects(pool.addMesh(geometry()),code('DYNAMIC'));assert.equal(d.buffers.length,0);pool.dispose();
});
for(const change of [g=>{g.positions[0]=NaN;},g=>{g.positions[0]=1e100;},g=>{g.normals=[0,0];},g=>{g.tangents=Array(12).fill(0);},
  g=>{g.positions=new Float32Array(new SharedArrayBuffer(36));},g=>{structuredClone(g.positions.buffer,{transfer:[g.positions.buffer]});}])
  test('bad vertex data is rejected before GPU allocation',async()=>{
    const d=device(),pool=createGpuRigidGeometryPool(d.d,pose()),g=geometry();change(g);await assert.rejects(pool.addMesh(g));assert.equal(d.buffers.length,0);assert.equal(pool.failed,false);pool.dispose();
  });
test('aggregate unique bytes, incoming bytes, component counts and handle counts are bounded',async()=>{
  const d=device(),p=pose(),pool=createGpuRigidGeometryPool(d.d,p,{maxBytes:120,maxMeshes:2,maxComponents:18});
  await assert.rejects(pool.addMesh(geometry(),{maxAdditionalBytes:119}),code('LIMIT'));assert.equal(d.buffers.length,0);
  const a=await pool.addMesh(geometry(),{maxAdditionalBytes:120}),b=await pool.addMesh(geometry(1),{maxAdditionalBytes:0});
  await assert.rejects(pool.addMesh(geometry()),code('LIMIT'));b.dispose();const other=geometry();other.positions[0]=9;
  await assert.rejects(pool.addMesh(other),code('LIMIT'));assert.equal(d.buffers.length,1);a.dispose();pool.dispose();
  for(const config of [{maxBytes:119},{maxComponents:17}]){const q=createGpuRigidGeometryPool(d.d,p,config);await assert.rejects(q.addMesh(geometry()),code('LIMIT'));q.dispose();}
});
test('device maximum buffer size is checked before allocation',async()=>{
  const d=device();d.d.limits.maxBufferSize=119;const pool=createGpuRigidGeometryPool(d.d,pose());await assert.rejects(pool.addMesh(geometry()),code('LIMIT'));assert.equal(d.buffers.length,0);pool.dispose();
});
test('source arrays are captured before waiting for native validation',async()=>{
  const d=device(),p=pose(),pool=createGpuRigidGeometryPool(d.d,p),resolves=[];d.holdScopes(()=>new Promise(r=>resolves.push(r)));
  const g=geometry(),pending=pool.addMesh(g);g.positions.fill(99);assert.equal(d.depth,0);resolves.forEach(r=>r(null));
  const a=await pending;assert.equal(new Float32Array(a.vertexBuffer.data)[0],0);pool.dispose();
});
test('pose changes during registration retire an unpublished buffer without poisoning the pool',async()=>{
  const d=device(),p=pose(),pool=createGpuRigidGeometryPool(d.d,p),resolves=[];d.holdScopes(()=>new Promise(r=>resolves.push(r)));
  const pending=pool.addMesh(geometry());p.version++;resolves.forEach(r=>r(null));await assert.rejects(pending,code('CHANGED'));
  assert.equal(pool.failed,false);assert.equal(pool.bufferBytes,0);assert.equal(d.buffers[0].destroyed,1);d.holdScopes(null);
  const a=await pool.addMesh(geometry());assert.equal(a.poseVersion,1);pool.dispose();
});
test('disposal interrupts pending allocation and does not leak late native results',async()=>{
  const d=device(),pool=createGpuRigidGeometryPool(d.d,pose()),resolves=[];d.holdScopes(()=>new Promise(r=>resolves.push(r)));
  const pending=pool.addMesh(geometry());pool.dispose();await assert.rejects(pending,code('DISPOSED'));resolves.forEach(r=>r(null));
  await new Promise(r=>setImmediate(r));assert.equal(d.buffers[0].destroyed,1);assert.equal(pool.bufferBytes,0);
});
for(const mode of ['validation','mapping','completion','loss'])test(`${mode} failure retires owned buffers and stays observable`,async()=>{
  const d=device(),p=pose(),pool=createGpuRigidGeometryPool(d.d,p);
  if(mode==='validation'){d.scopeError({message:'validation failed'});await assert.rejects(pool.addMesh(geometry()),/validation failed/);}
  else if(mode==='mapping'){const create=d.d.createBuffer;d.d.createBuffer=desc=>{const b=create(desc);b.getMappedRange=()=>{throw Error('mapping failed');};return b;};await assert.rejects(pool.addMesh(geometry()),/mapping failed/);}
  else {await pool.addMesh(geometry());if(mode==='completion')d.completion(async()=>{throw Error('completion failed');});else d.lose({message:'device lost'});
    await assert.rejects(pool.whenIdle());}
  assert.ok(pool.failed);assert.equal(pool.bufferBytes,0);assert.equal(d.depth,0);assert.ok(d.buffers.every(b=>b.destroyed===1));
  await assert.rejects(pool.addMesh(geometry()));pool.dispose();
});
test('loss races a never-settling allocation validation promise',async()=>{
  const d=device(),pool=createGpuRigidGeometryPool(d.d,pose());d.holdScopes(()=>new Promise(()=>{}));
  const pending=pool.addMesh(geometry());d.lose({});await assert.rejects(pending,code('LOST'));assert.equal(d.buffers[0].destroyed,1);
});
test('registration is sequential and synchronous input getters cannot dispose the pool',async()=>{
  const d=device(),pool=createGpuRigidGeometryPool(d.d,pose()),resolves=[];d.holdScopes(()=>new Promise(r=>resolves.push(r)));
  const pending=pool.addMesh(geometry());await assert.rejects(pool.addMesh(geometry()),code('REENTRANT'));resolves.forEach(r=>r(null));await pending;
  const g=geometry();Object.defineProperty(g,'positions',{get(){pool.dispose();return new Float32Array(9);}});
  await assert.rejects(pool.addMesh(g),code('REENTRANT'));assert.equal(pool.disposed,false);pool.dispose();
});
