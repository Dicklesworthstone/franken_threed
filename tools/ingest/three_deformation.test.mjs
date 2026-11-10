/** Runs actual source binding/lifetime code. The core GPU factory and dispatch
 * boundary are recorded here; these tests do not execute WGSL or retained Three.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import {fixture, THREE, deferred} from './fixtures/animation/three_deformation_fixture.mjs';
import {claimWebGLGeometrySnapshot, retainWebGLGeometryAttributes} from './gpu_buffer_geometry.mjs';
const key = '__f3d_source_deformation_test__';
const stub = 'data:text/javascript,' + encodeURIComponent(`export const createGpuAnimationDeformer=(...a)=>globalThis.${key}.create(...a); export const updateGpuAnimationDeformers=a=>globalThis.${key}.update(a); export const createGpuAnimationDeformerCache=()=>({bytes:0});`);
const source = (await fs.readFile(new URL('./three_deformation.mjs', import.meta.url), 'utf8'))
  .replace("'./animation_webgpu.mjs'", JSON.stringify(stub))
  .replaceAll("'./three_deformation_binding.mjs'", JSON.stringify(new URL('./three_deformation_binding.mjs',import.meta.url).href))
  .replaceAll("'./gpu_buffer_geometry.mjs'", JSON.stringify(new URL('./gpu_buffer_geometry.mjs',import.meta.url).href));
const {createGpuThreeDeformation,updateGpuThreeDeformations} = await import('data:text/javascript,'+encodeURIComponent(source));
const code = name => ({code:`THREE_DEFORMATION_${name}`});
function setup() {
  const calls=[], controls={ready:null,idle:null,fail:null}, device={};
  globalThis[key]={
    async create(d,pose,geometry,options) {
      const gpu={pose,geometry,options,disposed:false,failed:false,bufferBytes:2048,
        whenIdle(){return controls.idle?.promise??Promise.resolve();},
        dispose(){calls.push(['dispose',gpu]);gpu.disposed=true;}};
      calls.push(['create',gpu,d]); if(controls.ready)await controls.ready.promise;
      if(controls.fail)throw controls.fail; return gpu;
    },
    update(gpus){calls.push(['update',gpus]);for(const g of gpus)assert.equal(g.pose.disposed,false);},
  };
  return {calls,controls,device};
}
const create=(h,f,more={})=>createGpuThreeDeformation(h.device,f.mesh,{three:THREE,...more});

test('returns the exact registered core owner with decoded surfaces and bounded options',async()=>{
  const h=setup(),f=fixture({skin:true,morph:true}),o=await create(h,f,{maxBytes:4096,maxComponents:200});
  const native=h.calls[0][1];assert.equal(o.deformer,native);assert.equal(native.options.maxBytes,4096);
  assert.equal(native.options.maxComponents,200);assert.equal(o.bufferBytes,2048);assert.equal(o.vertexCount,3);
  o.dispose();assert.equal(h.calls.filter(c=>c[0]==='dispose').length,1);assert.equal(o.bufferBytes,0);
});
test('batch captures separate mesh palettes and morph weights before one core dispatch',async()=>{
  const h=setup(),a=fixture({skin:true,morph:true}),b=fixture({skin:true,morph:true});
  const x=await create(h,a),y=await create(h,b);a.mesh.morphTargetInfluences[0]=2;b.mesh.morphTargetInfluences[0]=-1;
  a.mesh.skeleton.bones[0].matrixWorld.elements[12]=3;b.mesh.skeleton.bones[0].matrixWorld.elements[12]=7;
  updateGpuThreeDeformations([x,y]);const updates=h.calls.filter(c=>c[0]==='update');assert.equal(updates.length,1);
  assert.deepEqual(updates[0][1].map(g=>g.pose.morphWeights[0]),[2,-1]);assert.deepEqual(updates[0][1].map(g=>g.pose.jointMatrices[12]),[3,7]);
  x.dispose();y.dispose();
});
test('invalid later source causes no GPU dispatch and remains retryable',async()=>{
  const h=setup(),a=fixture({morph:true}),b=fixture({morph:true}),x=await create(h,a),y=await create(h,b);
  a.mesh.morphTargetInfluences[0]=1;b.mesh.morphTargetInfluences[0]=NaN;
  assert.throws(()=>updateGpuThreeDeformations([x,y]),code('VALUE'));assert.equal(h.calls.filter(c=>c[0]==='update').length,0);
  b.mesh.morphTargetInfluences[0]=0.25;updateGpuThreeDeformations([x,y]);assert.equal(x.failed,false);x.dispose();y.dispose();
});
test('static content changes require preparation instead of silently drawing stale GPU data',async()=>{
  const h=setup(),f=fixture({morph:true}),o=await create(h,f);f.geometry.attributes.position.needsUpdate=true;
  assert.equal(o.matches(),false);assert.throws(()=>o.update(),code('PREPARE'));assert.equal(h.calls.filter(c=>c[0]==='update').length,0);o.dispose();
});
test('source edits during compilation reject publication and release the late core owner',async()=>{
  const h=setup(),f=fixture({morph:true});h.controls.ready=deferred();const pending=create(h,f);
  f.geometry.attributes.position.needsUpdate=true;h.controls.ready.resolve();await assert.rejects(pending,code('PREPARE'));
  assert.equal(h.calls.filter(c=>c[0]==='dispose').length,1);assert.equal(f.geometry.listeners.get('dispose').size,0);
});
test('pre-aborted initialization performs no core allocation',async()=>{
  const h=setup(),f=fixture({skin:true}),abort=new AbortController();abort.abort();
  await assert.rejects(create(h,f,{signal:abort.signal}),code('ABORTED'));assert.equal(h.calls.length,0);
});
test('abort ends stalled construction and disposes its late owner exactly once',async()=>{
  const h=setup(),f=fixture({skin:true}),abort=new AbortController();h.controls.ready=deferred();
  const pending=create(h,f,{signal:abort.signal});abort.abort();await assert.rejects(pending,code('ABORTED'));
  assert.equal(f.geometry.listeners.get('dispose').size,0);h.controls.ready.resolve();await new Promise(r=>setImmediate(r));
  assert.equal(h.calls.filter(c=>c[0]==='dispose').length,1);
});
test('dispose and lifetime abort both end a stalled idle wait without owning the device',async()=>{
  for(const cancel of ['dispose','abort']){
    const h=setup(),f=fixture({skin:true}),abort=new AbortController(),o=await create(h,f,{signal:abort.signal});
    h.controls.idle=deferred();const pending=o.whenIdle();if(cancel==='dispose')o.dispose();else abort.abort();
    await assert.rejects(pending,code(cancel==='dispose'?'DISPOSED':'ABORTED'));o.dispose();
    assert.equal(h.calls.filter(c=>c[0]==='dispose').length,1);
  }
});
test('duplicates, foreign handles and mixed devices are refused before core dispatch',async()=>{
  const h=setup(),f=fixture({morph:true}),x=await create(h,f),y=await createGpuThreeDeformation({},fixture({morph:true}).mesh,{three:THREE});
  for(const batch of [[x,x],[{}],[x,y]])assert.throws(()=>updateGpuThreeDeformations(batch),e=>e.code?.startsWith('THREE_DEFORMATION_'));
  assert.equal(h.calls.filter(c=>c[0]==='update').length,0);x.dispose();y.dispose();
});
test('core construction failures release source listeners and preserve the original error',async()=>{
  const h=setup(),f=fixture({skin:true}),error=new Error('compile rejected');h.controls.fail=error;
  await assert.rejects(create(h,f),e=>e===error);assert.equal(f.geometry.listeners.get('dispose').size,0);
});
test('native terminal failures release owned resources and become observable',async()=>{
  const h=setup(),f=fixture({skin:true}),o=await create(h,f);o.deformer.failed=true;
  assert.throws(()=>o.update(),code('GPU'));assert.equal(o.failed,true);assert.equal(h.calls.filter(c=>c[0]==='dispose').length,1);o.dispose();
});

test('WebGL deformation leases all original inputs until geometry or owner disposal',async()=>{
  const h=setup(),f=fixture({skin:true,morph:true});
  f.geometry.index=new THREE.BufferAttribute(new Uint16Array([0,1,2]),1);
  const attributes=[...Object.values(f.geometry.attributes),f.geometry.index,...f.geometry.morphAttributes.position];
  const owner=await create(h,f);
  for(const attribute of attributes)assert.throws(()=>claimWebGLGeometrySnapshot(h.device,attribute,{}),{code:'GEOMETRY_GPU_HISTORY'});
  f.geometry.dispose();assert.equal(owner.matches(),false);
  const releases=attributes.map(attribute=>claimWebGLGeometrySnapshot(h.device,attribute,{}));
  owner.dispose();
  for(const release of releases)release();
});
test('WebGL deformation conflicts roll back the whole input set while native inputs remain independent',async()=>{
  const h=setup(),f=fixture({skin:true,morph:true});
  const release=claimWebGLGeometrySnapshot(h.device,f.geometry.morphAttributes.position[0],{});
  await assert.rejects(create(h,f),{code:'GEOMETRY_GPU_HISTORY'});
  assert.equal(h.calls.length,0);
  for(const attribute of Object.values(f.geometry.attributes))claimWebGLGeometrySnapshot(h.device,attribute,{})();
  const native=await create(h,f,{sourceAttributeCallbacks:false});
  native.dispose();release();
  claimWebGLGeometrySnapshot(h.device,f.geometry.morphAttributes.position[0],{})();
});
test('interleaved and shared deformation leases release each counted owner exactly once',()=>{
  const h=setup(),data=new THREE.InterleavedBuffer(new Float32Array(18),6);
  const position=new THREE.InterleavedBufferAttribute(data,3,0),normal=new THREE.InterleavedBufferAttribute(data,3,3);
  const first=retainWebGLGeometryAttributes(h.device,[position,normal,position]);
  const second=retainWebGLGeometryAttributes(h.device,[normal]);
  first();first();assert.throws(()=>claimWebGLGeometrySnapshot(h.device,data,{}),{code:'GEOMETRY_GPU_HISTORY'});
  second();claimWebGLGeometrySnapshot(h.device,data,{})();
});
test('old asynchronous preparation cleanup cannot release a replacement deformation lease',async()=>{
  const h=setup(),f=fixture({skin:true});h.controls.ready=deferred();
  const pending=create(h,f),rejected=assert.rejects(pending,code('PREPARE'));
  assert.throws(()=>claimWebGLGeometrySnapshot(h.device,f.geometry.attributes.position,{}),{code:'GEOMETRY_GPU_HISTORY'});
  f.geometry.dispose();
  const replacement=create(h,f);h.controls.ready.resolve();
  await rejected;const owner=await replacement;
  assert.throws(()=>claimWebGLGeometrySnapshot(h.device,f.geometry.attributes.position,{}),{code:'GEOMETRY_GPU_HISTORY'});
  owner.dispose();claimWebGLGeometrySnapshot(h.device,f.geometry.attributes.position,{})();
  assert.equal(f.geometry.listeners.get('dispose').size,0);
});
test('source copying and GPU factory failures release every original deformation lease',async()=>{
  for(const failure of ['copy','gpu']){
    const h=setup(),f=fixture({skin:true,morph:true}),marker=new Error('factory rejected');
    if(failure==='copy')f.geometry.attributes.position.array[0]=NaN;else h.controls.fail=marker;
    await assert.rejects(create(h,f),failure==='copy'?code('VALUE'):error=>error===marker);
    for(const attribute of [...Object.values(f.geometry.attributes),...f.geometry.morphAttributes.position])
      claimWebGLGeometrySnapshot(h.device,attribute,{})();
    assert.equal(f.geometry.listeners.get('dispose').size,0);
  }
});
