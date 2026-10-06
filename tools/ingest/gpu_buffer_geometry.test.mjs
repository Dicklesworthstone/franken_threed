import assert from 'node:assert/strict';
import {test} from 'node:test';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createGpuBufferGeometry,bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const {BufferAttribute,BufferGeometry,InterleavedBuffer,InterleavedBufferAttribute,Float16BufferAttribute} = await import(pathToFileURL(path.join(root,'build/three.core.js')));
const {WebGLAttributes} = await import(pathToFileURL(path.join(root,'src/renderers/webgl/WebGLAttributes.js')));
const attribute=(values,width=3,C=Float32Array)=>new BufferAttribute(new C(values),width);
const geometry=()=>new BufferGeometry().setAttribute('position',attribute([0,0,0, 1,0,0, 0,1,0]));
const bytes=b=>[...new Uint8Array(b.data)];
function glReplica() {
  let bound;
  const gl={createBuffer:()=>({data:new ArrayBuffer(0)}),bindBuffer(_t,b){bound=b;},
    bufferData(_t,array){bound.data=array.buffer.slice(array.byteOffset,array.byteOffset+array.byteLength);},
    bufferSubData(_t,offset,array,start=0,count=0){
      const n=count===0?array.length-start:count;
      new Uint8Array(bound.data).set(new Uint8Array(array.buffer,array.byteOffset+start*array.BYTES_PER_ELEMENT,n*array.BYTES_PER_ELEMENT),offset);
    },deleteBuffer(b){b.deleted=true;}};
  return WebGLAttributes(gl);
}
function same(actual, expected) {
  const a=bytes(actual), b=bytes(expected); assert.deepEqual(a.slice(0,b.length),b);
  assert.ok(a.slice(b.length).every(v=>v===0),'alignment tail');
}

test('pinned upload oracle: stale CPU edits, source +1 coalescing, callbacks, empty-range remainder',async()=>{
  const d=geometryDevice(), source=geometry(), oracle=geometry(), gl=glReplica();
  const a=source.attributes.position, b=oracle.attributes.position;
  a.addUpdateRange(2,1); b.addUpdateRange(2,1);
  const seen=[[],[]];
  a.onUpload(function(){seen[0].push(this.updateRanges.map(r=>({...r})));});
  b.onUpload(function(){seen[1].push(this.updateRanges.map(r=>({...r})));});
  const gpu=createGpuBufferGeometry(d,source); gl.update(b,0);
  const resident=gpu.vertexBuffer; same(resident,gl.get(b).buffer);
  assert.equal(a.updateRanges.length,1,'first upload does not clear ranges');
  a.clearUpdateRanges(); b.clearUpdateRanges();
  for(let i=0;i<9;i++)a.array[i]=b.array[i]=10+i;
  gpu.update(); gl.update(b,0); same(resident,gl.get(b).buffer);
  assert.equal(d.writes.length,1,'no upload without version change');
  for(const attr of [a,b]) {attr.addUpdateRange(4,1);attr.addUpdateRange(2,1);attr.needsUpdate=true;}
  const oldRange=a.updateRanges[1];
  gpu.update();gl.update(b,0);same(resident,gl.get(b).buffer);
  assert.equal(oldRange.count,3,'pinned source includes the one-element gap');
  assert.equal(new Float32Array(resident.data)[3],13);
  assert.equal(new Float32Array(resident.data)[0],0,'unrequested CPU value remains stale');
  for(const attr of [a,b]){attr.addUpdateRange(7,0);attr.needsUpdate=true;}
  gpu.update();gl.update(b,0);same(resident,gl.get(b).buffer);
  assert.equal(new Float32Array(resident.data)[8],18);
  assert.deepEqual(seen[0],seen[1]); assert.equal(gpu.vertexBuffer,resident);
  await gpu.whenIdle();gpu.dispose();
});

test('Uint16 partial uploads fill alignment gaps from GPU history, not CPU arrays',async()=>{
  const d=geometryDevice(),g=geometry(),o=geometry(),gl=glReplica();
  g.setIndex(attribute([0,1,2,0,2,1],1,Uint16Array));o.setIndex(attribute([0,1,2,0,2,1],1,Uint16Array));
  const gpu=createGpuBufferGeometry(d,g);gl.update(o.index,0);
  const before=bufferGeometrySnapshot(gpu,d).indexBuffer;
  g.index.array.set([2,0,1,2,1,0]);o.index.array.set(g.index.array);
  for(const attr of [g.index,o.index]){attr.addUpdateRange(1,1);attr.needsUpdate=true;}
  gpu.update();gl.update(o.index,0);same(before,gl.get(o.index).buffer);
  assert.deepEqual([...new Uint16Array(before.data)],[0,0,2,0,2,1]);
  assert.equal(d.writes.at(-1).input.length,4);
  assert.equal(d.writes.at(-1).offset,0);
  await gpu.whenIdle();gpu.dispose();
});

test('interleaved positions and normals share one allocation, upload and callback',async()=>{
  const d=geometryDevice(),g=new BufferGeometry(),data=new InterleavedBuffer(new Float32Array(18).fill(1),6);
  g.setAttribute('normal',new InterleavedBufferAttribute(data,3,3));
  g.setAttribute('position',new InterleavedBufferAttribute(data,3,0));
  let calls=0;data.onUpload(()=>calls++);
  const gpu=createGpuBufferGeometry(d,g), snap=bufferGeometrySnapshot(gpu,d);
  assert.equal(snap.layouts.length,1);assert.equal(snap.layouts[0].arrayStride,24);
  assert.deepEqual(snap.layouts[0].attributes.map(a=>a.shaderLocation),[0,1]);
  assert.equal(d.buffers.length,1);assert.equal(calls,1);
  data.array[4]=7;data.addUpdateRange(4,1);data.needsUpdate=true;gpu.update();
  assert.equal(d.writes.length,2);assert.equal(calls,2);
  assert.equal(new Float32Array(gpu.vertexBuffer.data)[4],7);await gpu.whenIdle();gpu.dispose();
});

test('callbacks acknowledge their final version, never their post-upload CPU edits',()=>{
  const d=geometryDevice(),g=geometry(),o=geometry(),gl=glReplica();
  const change=function(){this.array[0]=77;this.needsUpdate=true;};
  g.attributes.position.onUpload(change);o.attributes.position.onUpload(change);
  const gpu=createGpuBufferGeometry(d,g);gl.update(o.attributes.position,0);
  gpu.update();gl.update(o.attributes.position,0);same(gpu.vertexBuffer,gl.get(o.attributes.position).buffer);
  assert.equal(d.writes.length,1);assert.equal(new Float32Array(gpu.vertexBuffer.data)[0],0);gpu.dispose();
});

test('throwing update callback preserves writes, cleared ranges and unacknowledged version',()=>{
  const d=geometryDevice(),g=geometry(),o=geometry(),gl=glReplica(),gpu=createGpuBufferGeometry(d,g);gl.update(o.attributes.position,0);
  const failure=new Error('callback');
  for(const a of [g.attributes.position,o.attributes.position]){a.array[0]=22;a.addUpdateRange(0,1);a.needsUpdate=true;a.onUpload(()=>{throw failure;});}
  assert.throws(()=>gpu.update(),e=>e===failure);assert.throws(()=>gl.update(o.attributes.position,0),e=>e===failure);
  assert.equal(gpu.failed,false);same(gpu.vertexBuffer,gl.get(o.attributes.position).buffer);
  assert.equal(g.attributes.position.updateRanges.length,0);
  for(const a of [g.attributes.position,o.attributes.position]){a.array[8]=88;a.onUpload(()=>{});}
  gpu.update();gl.update(o.attributes.position,0);same(gpu.vertexBuffer,gl.get(o.attributes.position).buffer);
  assert.equal(new Float32Array(gpu.vertexBuffer.data)[8],88);gpu.dispose();
});

test('independent replicas consume public ranges with separate GPU histories',()=>{
  const a=geometryDevice(),b=geometryDevice(),g=geometry(),x=createGpuBufferGeometry(a,g),y=createGpuBufferGeometry(b,g);
  const p=g.attributes.position;p.array.fill(9);p.addUpdateRange(0,1);p.needsUpdate=true;
  x.update();y.update();
  assert.deepEqual([...new Float32Array(x.vertexBuffer.data)],[9,0,0,1,0,0,0,1,0]);
  assert.deepEqual([...new Float32Array(y.vertexBuffer.data)],Array(9).fill(9));
  assert.throws(()=>bufferGeometrySnapshot(x,b),{code:'GEOMETRY_GPU_DEVICE'});x.dispose();y.dispose();
});

test('source disposal releases residency and later use recreates it without changing source identity',()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g),old=gpu.vertexBuffer,p=g.attributes.position;
  g.dispose();assert.equal(old.destroyed,true);assert.equal(gpu.disposed,false);assert.equal(gpu.bufferBytes,0);
  assert.throws(()=>bufferGeometrySnapshot(gpu,d),{code:'GEOMETRY_GPU_RELEASED'});
  p.array[0]=8;gpu.update();assert.notEqual(gpu.vertexBuffer,old);assert.equal(g.attributes.position,p);
  assert.equal(new Float32Array(gpu.vertexBuffer.data)[0],8);assert.equal(gpu.diagnostics.generation,1);
  gpu.dispose();gpu.dispose();assert.equal(gpu.disposed,true);assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_DISPOSED'});
});

test('replacement attributes can resize while resident source resizing is refused before writes',()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g),before=d.writes.length;
  const p=g.attributes.position;p.array=new Float32Array(18);p.count=6;p.needsUpdate=true;
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_RESIZE'});assert.equal(d.writes.length,before);assert.equal(gpu.failed,false);
  g.setAttribute('position',attribute(18));gpu.update();assert.equal(gpu.vertexCount,6);assert.equal(gpu.diagnostics.allocations,2);
  gpu.dispose();
});

test('admission budgets and invalid late attributes have no upload side effects',()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g,{maxBytes:36,maxAttributes:1});
  const p=g.attributes.position;p.array[0]=3;p.needsUpdate=true;
  g.setAttribute('normal',attribute(9));const count=d.writes.length;
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_LIMIT'});assert.equal(d.writes.length,count);
  g.deleteAttribute('normal');gpu.update();assert.equal(new Float32Array(gpu.vertexBuffer.data)[0],3);
  p.addUpdateRange(8,2);p.needsUpdate=true;
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_SHAPE'});assert.equal(gpu.failed,false);gpu.dispose();
});

test('draw ranges are read at consumption time without uploading CPU arrays',()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g);
  g.setDrawRange(1,1);assert.deepEqual(bufferGeometrySnapshot(gpu,d).drawRange,{first:1,count:1});
  g.setDrawRange(7,Infinity);assert.deepEqual(bufferGeometrySnapshot(gpu,d).drawRange,{first:3,count:0});
  assert.equal(d.writes.length,1);gpu.dispose();
});

test('native failures and device loss terminate and retire allocations',async()=>{
  for(const failure of ['write','scope','loss']){
    const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g);await gpu.whenIdle();
    if(failure==='loss')d.lose();
    else {g.attributes.position.needsUpdate=true;if(failure==='write')d.writeError=new Error('queue');else d.scopeError={message:'validation'};
      if(failure==='write')assert.throws(()=>gpu.update(),/queue/);else gpu.update();}
    await assert.rejects(gpu.whenIdle());assert.equal(gpu.failed,true);assert.equal(gpu.bufferBytes,0);
    assert.ok(d.buffers.every(b=>b.destroyed));gpu.dispose();
  }
});

test('empty geometry owns legal padded storage and a zero draw range',()=>{
  const d=geometryDevice(),g=new BufferGeometry().setAttribute('position',attribute(0)),gpu=createGpuBufferGeometry(d,g);
  assert.equal(gpu.vertexCount,0);assert.equal(gpu.bufferBytes,4);assert.equal(d.writes.length,0);
  assert.deepEqual(bufferGeometrySnapshot(gpu,d).drawRange,{first:0,count:0});gpu.dispose();
});

test('an earlier upload callback replaces a later same-sized view at source ordering',()=>{
  const d=geometryDevice(),g=geometry();g.setAttribute('normal',attribute(9));
  const gpu=createGpuBufferGeometry(d,g);
  g.attributes.position.onUpload(()=>{g.attributes.normal.array=new Float32Array(9).fill(8);g.attributes.normal.needsUpdate=true;});
  g.attributes.position.needsUpdate=true;gpu.update();
  const normal=bufferGeometrySnapshot(gpu,d).vertexBuffers[1];
  assert.deepEqual([...new Float32Array(normal.data)],Array(9).fill(8));gpu.dispose();
});

test('queue completion failures are terminal rather than a successful reusable residency',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g);
  const error=new Error('completion');d.completion=Promise.reject(error);d.completion.catch(()=>{});
  await assert.rejects(gpu.whenIdle(),e=>e===error);assert.equal(gpu.failed,true);assert.equal(gpu.bufferBytes,0);gpu.dispose();
});


test('rejected error scopes terminate residency even before whenIdle is called',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g);await gpu.whenIdle();
  const error=new Error('scope rejected');
  d.popErrorScope=()=>{d.scopes.pop();return Promise.reject(error);};
  g.attributes.position.needsUpdate=true;gpu.update();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(gpu.failed,true);assert.equal(gpu.bufferBytes,0);
  assert.throws(()=>gpu.update(),e=>e===error);gpu.dispose();
});

test('source counts cannot outgrow stale residency and unsupported deformation inputs are explicit',()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g),p=g.attributes.position;
  p.array=new Float32Array(18);p.count=6;
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_SHAPE'});
  assert.equal(gpu.vertexCount,3);assert.equal(gpu.failed,false);
  g.setAttribute('position',attribute(18));gpu.update();assert.equal(gpu.vertexCount,6);
  g.morphAttributes={position:[attribute(18)]};
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_SHAPE'});
  g.morphAttributes={};g.isInstancedBufferGeometry=true;
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_SHAPE'});gpu.dispose();
});

test('normalized/integer/half sources upload the GL float-attribute conversion; ranges stay element-exact',()=>{
  const d=geometryDevice(), g=geometry(), f32=b=>[...new Float32Array(b.data)];
  const color=new BufferAttribute(new Uint8Array([255,0,128,255, 0,51,255,0, 1,2,3,4]),4,true);
  const uv=new BufferAttribute(new Int16Array([-32768,32767, 0,-1, 16384,-16384]),2,true);
  const normal=new BufferAttribute(new Int8Array([1,-2,3, 4,5,-6, 7,8,9]),3); // not normalized: values as floats
  g.setAttribute('color',color).setAttribute('uv',uv).setAttribute('normal',normal);
  const gpu=createGpuBufferGeometry(d,g),snap=bufferGeometrySnapshot(gpu,d);
  const buffer=name=>snap.vertexBuffers[snap.layouts.findIndex(l=>l.attributes.some(a=>a.shaderLocation===({normal:1,uv:3,color:4})[name]))];
  assert.deepEqual(f32(buffer('color')),[1,0,128/255,1, 0,0.2,1,0, 1/255,2/255,3/255,4/255].map(Math.fround));
  assert.deepEqual(f32(buffer('uv')),[-1,1, 0,-1/32767, 16384/32767,-16384/32767].map(Math.fround),'signed: max(c/32767,-1)');
  assert.deepEqual(f32(buffer('normal')),[1,-2,3,4,5,-6,7,8,9]);
  for(const l of snap.layouts)for(const a of l.attributes)assert.match(a.format,/^float32x[234]$/);
  // A source range uploads only its converted elements; GPU-stale edits stay stale.
  color.array[0]=0;color.array[5]=255;color.addUpdateRange(5,1);color.needsUpdate=true;gpu.update();
  const c=f32(buffer('color'));assert.equal(c[5],1);assert.equal(c[0],1,'outside the range stays GPU-stale');
  // Halves decode exactly, including subnormals, infinities and signed zero.
  const half=new Float16BufferAttribute(new Uint16Array(9),3);
  half.setXYZ(0,0.5,-2,65504);half.setXYZ(1,2**-24,-0,Infinity);half.setXYZ(2,1/3,0,0);half.array[5]=0x7c00; // toHalfFloat clamps Infinity; store the raw bits
  const h=new BufferGeometry().setAttribute('position',half);
  const hs=bufferGeometrySnapshot(createGpuBufferGeometry(d,h),d);
  const hv=f32(hs.vertexBuffers[0]);
  assert.deepEqual(hv.slice(0,6),[0.5,-2,65504,2**-24,-0,Infinity]);assert.ok(Object.is(hv[4],-0));
  assert.equal(hv[6],Math.fround(0.333251953125));
  // Changing normalization of a resident stream is a new layout, not an in-place patch.
  color.normalized=false;color.needsUpdate=true;assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_FORMAT'});
});

test('onUpload(disposeArray): released arrays stay drawable until a newer version needs data',()=>{
  const d=geometryDevice(),g=geometry(),position=g.attributes.position;
  position.onUpload(function(){this.array=null;});
  const gpu=createGpuBufferGeometry(d,g);
  assert.equal(position.array,null,'callback ran after the upload');
  gpu.update();gpu.update();
  assert.equal(bufferGeometrySnapshot(gpu,d).vertexCount,3);
  position.needsUpdate=true;
  assert.throws(()=>gpu.update(),{code:'GEOMETRY_GPU_STORAGE'});
});
