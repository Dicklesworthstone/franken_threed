/** Actual scene traversal, source deformation and ownership composition. Native
 * GPU calls and source constructors are recorded, not executed Three/WGSL.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import {fixture,THREE,Scene,Camera,Mesh,MeshBasicMaterial,BufferAttribute,attr,deferred} from './fixtures/animation/three_deformation_fixture.mjs';
const key='__f3d_animated_scene_test__';
const data=s=>'data:text/javascript,'+encodeURIComponent(s);
const core=data(`export const createGpuAnimationDeformer=(...a)=>globalThis.${key}.create(...a); export const updateGpuAnimationDeformers=a=>globalThis.${key}.update(a);`);
const deformation=data((await fs.readFile(new URL('./three_deformation.mjs',import.meta.url),'utf8'))
  .replace("'./animation_webgpu.mjs'",JSON.stringify(core))
  .replaceAll("'./three_deformation_binding.mjs'",JSON.stringify(new URL('./three_deformation_binding.mjs',import.meta.url).href)));
const renderer=data(`export const createGpuAnimationRenderer=(...a)=>globalThis.${key}.renderer(...a);`);
const geometry=data(`export const createGpuBufferGeometry=(...a)=>globalThis.${key}.geometry(...a);export const bufferGeometrySnapshot=g=>g.shape;
export const createGpuInstanceAttributes=()=>{throw Error('Unexpected instance');};export const instanceAttributesSnapshot=()=>{};export const inspectInstanceAttributes=()=>{};`);
const textures=data(`export const createGpuThreeTextures=()=>{throw Error('Unexpected texture');};`);
const sceneModule=data((await fs.readFile(new URL('./three_scene.mjs',import.meta.url),'utf8'))
  .replace("'./animation_render.mjs'",JSON.stringify(renderer)).replace("'./gpu_buffer_geometry.mjs'",JSON.stringify(geometry))
  .replace("'./three_textures.mjs'",JSON.stringify(textures)).replace("'./three_deformation.mjs'",JSON.stringify(deformation)));
const {createGpuThreeScene}=await import(sceneModule);
function setup(list=[fixture({skin:true,morph:true})]) {
  const calls=[],controls={compile:null,register:null,idle:null,registrationError:null},device={};
  const scene=new Scene(),camera=new Camera();
  for(const f of list){f.mesh.material??=new MeshBasicMaterial();scene.add(f.mesh);}
  const resource={failed:false,disposed:false,drawCount:0,
    async addMesh(gpu,options){
      const m={gpu,options,disposed:false,dispose(){this.disposed=true;calls.push(['mesh-dispose',this]);}};
      calls.push(['register',m]);if(controls.register)await controls.register.promise;
      if(controls.registrationError)throw controls.registrationError;
      if(resource.disposed)m.dispose();return m;
    },
    render(frame){calls.push(['draw',frame]);resource.drawCount=frame.draws.length;},
    whenIdle(){calls.push(['renderer-idle']);return controls.idle?.promise??Promise.resolve();},
    dispose(){this.disposed=true;calls.push(['renderer-dispose']);},
  };
  globalThis[key]={
    async create(d,pose,geometry,options){
      if(options.maxBytes<1000)throw Object.assign(new Error('GPU budget'),{code:'ANIMATION_GPU_LIMIT'});
      const gpu={pose,geometry,options,bufferBytes:1000,failed:false,disposed:false,
        dispose(){this.disposed=true;calls.push(['deformer-dispose',this]);},
        whenIdle(){return controls.idle?.promise??Promise.resolve();}};
      calls.push(['deformer',gpu]);if(controls.compile)await controls.compile.promise;return gpu;
    },
    update(gpus){calls.push(['deform',gpus]);for(const gpu of gpus)assert.equal(gpu.disposed,false);},
    async renderer(){calls.push(['renderer']);return resource;},
    geometry(d,g){const gpu={bufferBytes:100,failed:false,shape:{signature:g,indexBuffer:g.index,indexCount:g.index?.count??0,vertexCount:g.attributes.position.count},
      update(){},whenIdle(){return Promise.resolve();},dispose(){calls.push(['geometry-dispose',gpu]);}};calls.push(['geometry',gpu]);return gpu;},
  };
  return {calls,controls,device,scene,camera,list,resource};
}
const create=(h,options={})=>createGpuThreeScene(h.device,h.scene,{three:THREE,sortObjects:false,...options});
const count=(h,name)=>h.calls.filter(c=>c[0]===name).length;
const last=(h,name)=>h.calls.filter(c=>c[0]===name).at(-1)?.[1];
const frame={colorView:{},depthView:{}};
const tick=()=>new Promise(r=>setImmediate(r));

test('source skin/morph scenes dispatch fused deformation before normal material drawing',async()=>{
  const h=setup(),r=await create(h),f=h.list[0];assert.equal(count(h,'geometry'),0);assert.equal(r.diagnostics.deformedMeshCount,1);
  f.mesh.morphTargetInfluences[0]=0.75;f.mesh.skeleton.bones[0].matrixWorld.elements[12]=4;r.render(h.camera,frame);
  assert.equal(count(h,'deform'),1);assert.equal(last(h,'deform')[0].pose.jointMatrices[12],4);
  assert.equal(last(h,'deform')[0].pose.morphWeights[0],0.75);
  assert.ok(h.calls.findIndex(c=>c[0]==='deform')<h.calls.findIndex(c=>c[0]==='draw'));
  assert.equal(last(h,'register').gpu,last(h,'deform')[0]);assert.equal(last(h,'register').options.vertexColors,null);
  assert.equal(last(h,'draw').draws.length,1);r.dispose();assert.equal(count(h,'deformer-dispose'),1);
});
test('two source meshes sharing geometry keep distinct GPU morph state',async()=>{
  const a=fixture({morph:true}),b={geometry:a.geometry,mesh:new Mesh(a.geometry)};b.mesh.morphTargetInfluences=[-0.5];
  const h=setup([a,b]),r=await create(h);r.render(h.camera,frame);
  assert.equal(count(h,'deformer'),2);assert.equal(last(h,'deform').length,2);
  assert.deepEqual(last(h,'deform').map(g=>g.pose.morphWeights[0]),[0.5,-0.5]);r.dispose();
});
test('material groups and double-sided transparency share one deformer without dropping passes',async()=>{
  const f=fixture({morph:true}),m=new MeshBasicMaterial();m.transparent=true;m.side=THREE.DoubleSide;
  f.mesh.material=[m,m];f.geometry.index=new BufferAttribute(new Uint16Array([0,1,2,0,1,2]),1);
  f.geometry.groups=[{start:0,count:3,materialIndex:0},{start:3,count:3,materialIndex:1}];
  const h=setup([f]),r=await create(h);r.render(h.camera,frame);
  assert.equal(count(h,'deformer'),1);assert.equal(count(h,'register'),2);assert.equal(last(h,'deform').length,1);
  assert.deepEqual(last(h,'draw').draws.map(d=>[d.first,d.count]),[[0,3],[0,3],[3,3],[3,3]]);
  assert.equal(m.side,THREE.DoubleSide);r.dispose();
});
test('animated indices, UVs and RGBA colors reach the core immutable draw path',async()=>{
  const f=fixture({morph:true});f.geometry.index=new BufferAttribute(new Uint16Array([2,1,0]),1);
  f.geometry.attributes.uv=attr([0,0,1,0,0,1],2);f.geometry.attributes.color=attr([1,0,0,0,1,0,0,0,1]);
  f.mesh.material=new MeshBasicMaterial();f.mesh.material.vertexColors=true;
  const h=setup([f]),r=await create(h),options=last(h,'register').options;
  assert.deepEqual([...options.indices],[2,1,0]);assert.equal(options.texCoords.length,6);assert.equal(options.vertexColors.length,12);r.dispose();
});
test('dynamic drawRange intersects animated material groups without re-registering meshes',async()=>{
  const f=fixture({morph:true});f.geometry.index=new BufferAttribute(new Uint16Array([0,1,2,0,1,2,0,1,2]),1);
  f.mesh.material=[new MeshBasicMaterial()];f.geometry.groups=[{start:3,count:6,materialIndex:0}];
  f.geometry.drawRange={start:0,count:6};const h=setup([f]),r=await create(h);r.render(h.camera,frame);
  assert.deepEqual(last(h,'draw').draws.map(d=>[d.first,d.count]),[[3,3]]);
  f.geometry.drawRange={start:6,count:3};r.render(h.camera,frame);assert.deepEqual(last(h,'draw').draws.map(d=>[d.first,d.count]),[[6,3]]);
  assert.equal(count(h,'register'),1);r.dispose();
});
test('source static edits reject drawing until prepare replaces the deformer',async()=>{
  const h=setup(),r=await create(h),old=last(h,'deformer');h.list[0].geometry.attributes.position.needsUpdate=true;
  assert.throws(()=>r.render(h.camera,frame),{code:'THREE_DEFORMATION_PREPARE'});assert.equal(count(h,'draw'),0);
  await r.prepare();assert.equal(old.disposed,true);assert.notEqual(last(h,'deformer'),old);r.render(h.camera,frame);r.dispose();
});
test('failed replacement respects old-plus-new memory and leaves the old owner intact',async()=>{
  const h=setup(),r=await create(h,{maxDeformationBytes:1500}),old=last(h,'deformer');
  h.list[0].geometry.attributes.position.needsUpdate=true;
  await assert.rejects(r.prepare(),{code:'ANIMATION_GPU_LIMIT'});assert.equal(old.disposed,false);
  assert.equal(r.failed,false);assert.equal(r.diagnostics.deformationBytes,1000);r.dispose();
});
test('retired deformers remain alive until preceding submissions finish',async()=>{
  const h=setup(),r=await create(h),old=last(h,'deformer');r.render(h.camera,frame);
  h.controls.idle=deferred();h.scene.children=[];const pending=r.prepare();await tick();assert.equal(old.disposed,false);
  h.controls.idle.resolve();await pending;assert.equal(old.disposed,true);assert.equal(r.diagnostics.deformedMeshCount,0);r.dispose();
});
test('source edits during registration reject stale publication and clean all new owners',async()=>{
  const h=setup();h.controls.register=deferred();const pending=create(h);await tick();
  h.list[0].geometry.attributes.position.needsUpdate=true;h.controls.register.resolve();
  await assert.rejects(pending,{code:'THREE_DEFORMATION_PREPARE'});assert.equal(count(h,'deformer-dispose'),1);
  assert.equal(h.list[0].geometry.listeners.get('dispose').size,0);
});
test('disposal cancels an in-progress replacement and releases late core construction',async()=>{
  const h=setup(),r=await create(h),old=last(h,'deformer');h.list[0].geometry.attributes.position.needsUpdate=true;
  h.controls.compile=deferred();const pending=r.prepare();await tick();r.dispose();
  await assert.rejects(pending,e=>['THREE_DEFORMATION_ABORTED','THREE_SCENE_DISPOSED'].includes(e.code));
  assert.equal(old.disposed,true);h.controls.compile.resolve();await tick();assert.equal(count(h,'deformer-dispose'),2);
});
test('invalid later morph weights prevent the whole animated batch and final draw',async()=>{
  const h=setup([fixture({morph:true}),fixture({morph:true})]),r=await create(h);
  h.list[1].mesh.morphTargetInfluences[0]=NaN;assert.throws(()=>r.render(h.camera,frame),{code:'THREE_DEFORMATION_VALUE'});
  assert.equal(count(h,'deform'),0);assert.equal(count(h,'draw'),0);assert.equal(r.failed,false);
  h.list[1].mesh.morphTargetInfluences[0]=1;r.render(h.camera,frame);assert.equal(count(h,'deform'),1);r.dispose();
});
test('rigid and animated geometry coexist while hidden animation is not dispatched',async()=>{
  const h=setup([fixture(),fixture({morph:true})]),r=await create(h);r.render(h.camera,frame);
  assert.equal(count(h,'geometry'),1);assert.equal(last(h,'draw').draws.length,2);assert.equal(last(h,'deform').length,1);
  h.list[1].mesh.visible=false;r.render(h.camera,frame);assert.equal(count(h,'deform'),1);assert.equal(last(h,'draw').draws.length,1);r.dispose();
});
test('source world updates occur before palette capture',async()=>{
  const h=setup(),r=await create(h);h.scene.updateMatrixWorld=()=>{h.list[0].mesh.skeleton.bones[0].matrixWorld.elements[12]=12;};
  r.render(h.camera,frame);assert.equal(last(h,'deform')[0].pose.jointMatrices[12],12);r.dispose();
});
test('unsupported morph channels fail before the renderer is constructed',async()=>{
  const f=fixture({morph:true});f.geometry.morphAttributes.color=[attr(Array(9).fill(0))];const h=setup([f]);
  await assert.rejects(create(h),{code:'THREE_DEFORMATION_GEOMETRY'});assert.equal(count(h,'renderer'),0);
});
test('per-scene deformation capacity fails before any core allocation',async()=>{
  const h=setup([fixture({morph:true}),fixture({morph:true})]);await assert.rejects(create(h,{maxDeformedMeshes:1}),{code:'THREE_SCENE_LIMIT'});
  assert.equal(count(h,'renderer'),0);assert.equal(count(h,'deformer'),0);
});
test('terminal core GPU failures release the scene instead of replaying it',async()=>{
  const h=setup(),r=await create(h);last(h,'deformer').failed=true;
  assert.throws(()=>r.render(h.camera,frame),{code:'THREE_SCENE_DEVICE'});assert.equal(r.failed,true);assert.equal(count(h,'draw'),0);r.dispose();
});
test('pre-aborted source creation touches no renderer or deformation allocation',async()=>{
  const h=setup(),abort=new AbortController();abort.abort();await assert.rejects(create(h,{signal:abort.signal}),{code:'THREE_SCENE_ABORTED'});
  assert.equal(count(h,'renderer'),0);assert.equal(count(h,'deformer'),0);
});
test('lifetime signal cancels source creation while GPU deformation compilation is stalled',async()=>{
  const h=setup(),abort=new AbortController();h.controls.compile=deferred();const pending=create(h,{signal:abort.signal});await tick();
  abort.abort();await assert.rejects(pending,e=>['THREE_SCENE_ABORTED','THREE_DEFORMATION_ABORTED'].includes(e.code));
  assert.equal(h.resource.disposed,true);h.controls.compile.resolve();await tick();assert.equal(count(h,'deformer-dispose'),1);
});
