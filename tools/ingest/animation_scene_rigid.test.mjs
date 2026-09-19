import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
// Production scene and rigid pool. Clock, draw-order, material rendering and
// ordinary compute are explicit boundaries. Native WebGPU is not claimed.
const encode=s=>'data:text/javascript;base64,'+Buffer.from(s).toString('base64');
let source=readFileSync(new URL('./animation_scene.mjs',import.meta.url),'utf8');
for(const [file,code] of [
  ['animation_controller.mjs','export function createAnimationController(p){return p.controller;}'],
  ['animation_draw_order.mjs','export function createAnimationDrawOrder(entries,options){return options.pose.order(entries,options);}'],
  ['animation_webgpu.mjs','export function createGpuAnimationDeformer(d,p,g,o){return d.deform(p,g,o);}'],
  ['animation_render.mjs','export class AnimationRenderError extends Error{constructor(code,text){super(text);this.code=code;}} export function createGpuAnimationRenderer(d,o){return d.renderer(o);}'],
  ['animation_scene_shadow.mjs','export function prepareAnimationSceneShadows(p,i,o){return p.shadow(i,o);}'],
])source=source.replace("'./"+file+"'",JSON.stringify(encode(code)));
source=source.replace("'./animation_rigid_geometry.mjs'",JSON.stringify(new URL('./animation_rigid_geometry.mjs',import.meta.url).href));
const {createGpuAnimationScene}=await import(encode(source));
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function fixture(n=2) {
  const p={nodeCount:n,version:0,disposed:false,instances:[],worldMatrices:new Float64Array(n*16),morphOffsets:new Uint32Array(n+1)};
  for(let i=0;i<n;i++){p.worldMatrices.set(I,i*16);p.worldMatrices[i*16+12]=i*10;}
  let controllerDisposed=false;
  p.controller={update(dt){p.version++;for(let i=0;i<n;i++)p.worldMatrices[i*16+12]+=dt;},dispose(){controllerDisposed=true;}};
  const inputs=Array.from({length:n},(_,node)=>({geometry:{node,positions:new Float64Array([0,0,0,1,0,0,0,1,0]),morphTargets:[]}}));
  const buffers=[],computed=[],materials=[],frames=[],events=[];let lose,allocated=0,renderOptions,queueDrains=0;
  let renderFailure=null,computeFailure=null,completionFailure=null;
  const renderer={get allocatedBytes(){return allocated;},failed:false,
    async addMesh(gpu,material){if(renderFailure)throw renderFailure;const item={gpu,material,disposed:false,dispose(){this.disposed=true;}};materials.push(item);return item;},
    render(frame){events.push('color');frames.push(frame);},whenIdle:async()=>{},dispose(){allocated=0;},
  };
  const d={limits:{maxBufferSize:2**26},lost:new Promise(r=>{lose=r;}),pushErrorScope(){},popErrorScope:async()=>null,
    queue:{onSubmittedWorkDone:async()=>{queueDrains++;if(completionFailure)throw completionFailure;}},
    createBuffer(desc){const bytes=new ArrayBuffer(desc.size);const b={...desc,bytes,destroyed:0,getMappedRange:()=>bytes,unmap(){},destroy(){this.destroyed++;}};buffers.push(b);return b;},
    async renderer(o){renderOptions=o;allocated=256*o.maxDraws;return renderer;},
    async deform(p,g,options){if(computeFailure)throw computeFailure;const gpu={node:g.node,vertexCount:3,worldMatrix:p.worldMatrices.slice(g.node*16,g.node*16+16),
      bufferBytes:336,poseVersion:p.version,version:0,vertexBuffer:{},disposed:false,failed:false,updates:0,
      update(){this.worldMatrix.set(p.worldMatrices.subarray(g.node*16,g.node*16+16));this.poseVersion=p.version;this.version++;this.updates++;},
      whenIdle:async()=>{},dispose(){this.disposed=true;}};computed.push(gpu);return gpu;},
  };
  return {p,d,inputs,buffers,computed,materials,frames,events,lose,renderer,
    failRender(error){renderFailure=error;},failCompute(error){computeFailure=error;},failCompletion(error){completionFailure=error;},
    get queueDrains(){return queueDrains;},get controllerDisposed(){return controllerDisposed;},get renderOptions(){return renderOptions;}};
}
const make=(f,options={})=>createGpuAnimationScene(f.d,f.p,f.inputs,{sortObjects:false,rigidGeometry:true,...options});
test('scene shares rigid buffers and keeps independent material/draw identities and transforms',async()=>{
  const f=fixture(64);f.inputs.forEach((d,i)=>{d.baseColor=[i/64,0,0,1];});const scene=await make(f,{maxMeshes:64});
  assert.equal(f.buffers.length,1);assert.equal(f.computed.length,0);assert.equal(scene.draws.length,64);
  assert.equal(scene.bufferBytes,64*256+120);assert.deepEqual(scene.rigidGeometryStats,{meshes:64,uniqueGeometries:1,bufferBytes:120,computeMeshes:0});
  scene.update(0.5);scene.render({viewProjection:I,colorView:{}});
  assert.equal(f.frames[0].draws.length,64);assert.notEqual(scene.draws[0],scene.draws[1]);assert.notEqual(f.materials[0].material,f.materials[1].material);
  assert.equal(scene.deformers[63].worldMatrix[12],630.5);assert.equal(scene.deformers[0].vertexBuffer,scene.deformers[63].vertexBuffer);
  await scene.whenIdle();assert.equal(f.queueDrains,1);scene.dispose();assert.equal(f.buffers[0].destroyed,1);assert.equal(f.p.disposed,false);
});
test('default scene path is unchanged and continues to create one compute deformer per mesh',async()=>{
  const f=fixture(),scene=await make(f,{rigidGeometry:false});assert.equal(f.buffers.length,0);assert.equal(f.computed.length,2);
  scene.update(1);assert.ok(f.computed.every(g=>g.updates===1));assert.equal(scene.bufferBytes,512+672);scene.dispose();
});
test('mixed rigid and morphed nodes retain their appropriate execution path',async()=>{
  const f=fixture(3);f.inputs[2].geometry.morphTargets=[{positions:new Float64Array(9)}];f.p.morphOffsets[3]=1;
  const scene=await make(f);assert.equal(f.computed.length,1);assert.equal(f.computed[0].node,2);assert.equal(f.buffers.length,1);
  scene.update(1);assert.equal(f.computed[0].updates,1);assert.ok(scene.deformers.every(g=>g.poseVersion===1));
  assert.equal(scene.bufferBytes,768+336+120);scene.dispose();assert.ok(f.computed[0].disposed);assert.equal(f.buffers[0].destroyed,1);
});
for(const mode of ['skin','flat'])test(`${mode} geometry stays on the existing compute path`,async()=>{
  const f=fixture();if(mode==='skin')f.p.instances=[{node:1}];else f.inputs[1].geometry.flatNormals=true;
  const scene=await make(f);assert.equal(f.computed.length,1);assert.equal(f.computed[0].node,1);scene.dispose();
});
test('identical vertices with different surfaces still forward all five material maps independently',async()=>{
  const f=fixture();for(const [i,d] of f.inputs.entries()){
    d.shading='metallic-roughness';d.occlusionStrength=i*0.5;
    for(const key of ['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture','occlusionTexture'])d[key]={view:{key,i},sampler:{}};
    d.texCoords=[0,0,1,0,0,1];d.mapCoordinates={occlusionTexture:{texCoords:[0.1,0.2,0.3,0.4,0.5,0.6]}};
  }
  const scene=await make(f);assert.equal(f.buffers.length,1);assert.equal(f.materials[0].material.occlusionStrength,0);assert.equal(f.materials[1].material.occlusionStrength,0.5);
  assert.notEqual(f.materials[0].material.occlusionTexture.view,f.materials[1].material.occlusionTexture.view);scene.dispose();
});
test('exact total budget admits a shared second mesh with zero additional geometry bytes',async()=>{
  const f=fixture(),scene=await make(f,{maxBytes:512+120});assert.equal(scene.bufferBytes,632);scene.dispose();
  const short=fixture();await assert.rejects(make(short,{maxBytes:631}),{code:'ANIMATION_RIGID_LIMIT'});assert.equal(short.buffers.length,0);
});
test('unique geometry is charged once each and cannot exceed the remaining scene budget',async()=>{
  const f=fixture();f.inputs[1].geometry.positions[0]=99;await assert.rejects(make(f,{maxBytes:632}),{code:'ANIMATION_RIGID_LIMIT'});
  assert.equal(f.buffers.length,1);assert.equal(f.buffers[0].destroyed,1);assert.ok(f.controllerDisposed);
});
test('per-deformer byte budget remains enforced by the rigid path',async()=>{
  const f=fixture();await assert.rejects(make(f,{deformer:{maxBytes:119}}),{code:'ANIMATION_RIGID_LIMIT'});assert.equal(f.buffers.length,0);
});
test('stale or externally disposed mesh handles cannot submit a new scene frame',async()=>{
  const f=fixture(),scene=await make(f);f.p.version++;assert.throws(()=>scene.render({}),{code:'ANIMATION_SCENE_STALE'});assert.equal(f.frames.length,0);
  scene.upload();scene.deformers[0].dispose();assert.throws(()=>scene.render({}),{code:'ANIMATION_SCENE_STALE'});assert.equal(f.buffers[0].destroyed,0);scene.dispose();assert.equal(f.buffers[0].destroyed,1);
});
test('transform failure terminates the group without destroying the borrowed pose',async()=>{
  const f=fixture(),scene=await make(f);f.p.worldMatrices[16]=Infinity;f.p.version++;
  assert.throws(()=>scene.upload(),{code:'ANIMATION_RIGID_VALUE'});assert.ok(scene.failed);assert.equal(f.buffers[0].destroyed,1);assert.equal(f.p.disposed,false);scene.dispose();
});
for(const phase of ['material','compute','completion','loss'])test(`${phase} failure cleans the pool and all scene children`,async()=>{
  const f=fixture(3);if(phase==='compute')f.inputs[2].geometry.flatNormals=true;
  if(phase==='material')f.failRender(Error('material failed'));if(phase==='compute')f.failCompute(Error('compute failed'));
  if(phase==='material'||phase==='compute')await assert.rejects(make(f));
  else {const scene=await make(f);if(phase==='completion')f.failCompletion(Error('completion failed'));else f.lose({message:'lost'});
    await assert.rejects(scene.whenIdle());assert.ok(scene.failed);scene.dispose();}
  assert.ok(f.buffers.every(b=>b.destroyed===1));assert.ok(f.materials.every(m=>m.disposed));assert.equal(f.p.disposed,false);
});
test('sorting/culling and automatic shadows receive every independent rigid transform',async()=>{
  const f=fixture();let orderEntries,shadowDeformers,boundsUpdated=0;
  f.p.order=(entries)=>{entries=entries.slice();orderEntries=entries;return {order(){return entries.map(e=>e.mesh).reverse();},updateBounds(){boundsUpdated++;},dispose(){},viewProjection:I,lastCulling:{submittedDraws:2}};};
  f.p.shadow=inputs=>({initialize:async(d,deformers)=>{assert.equal(inputs.length,2);shadowDeformers=deformers;},
    render(lighting){f.events.push('depth');return {lighting,shadow:{map:{}},stats:{casters:2}};},whenIdle:async()=>{},dispose(){}});
  const scene=await make(f,{sortObjects:true,frustumCulling:true,shadow:{lightIndex:0}});scene.update(0.5);
  scene.render({viewProjection:I,lighting:{lights:[]}});assert.deepEqual(f.events,['depth','color']);assert.equal(boundsUpdated,1);
  assert.equal(orderEntries[0].deformer,scene.deformers[0]);assert.equal(shadowDeformers[1],scene.deformers[1]);assert.notEqual(shadowDeformers[0].worldMatrix,shadowDeformers[1].worldMatrix);
  assert.equal(f.frames[0].draws[0],scene.draws[1]);assert.equal(scene.shadowStats.casters,2);scene.dispose();
});
test('explicit draw order and override transforms are not merged by geometry sharing',async()=>{
  const f=fixture(),scene=await make(f),draws=[{mesh:scene.draws[1],worldMatrix:I},scene.draws[0],scene.draws[1]];
  scene.render({viewProjection:I,draws});assert.equal(f.frames[0].draws,draws);scene.dispose();
});
test('invalid optimization flags reject before constructing a controller or renderer',async()=>{
  const f=fixture();await assert.rejects(make(f,{rigidGeometry:'yes'}),{code:'ANIMATION_SCENE_RIGID'});assert.equal(f.renderOptions,undefined);
});
