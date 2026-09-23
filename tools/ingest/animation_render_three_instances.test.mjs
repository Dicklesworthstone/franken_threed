import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {inspectInstanceAttributes} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';
const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const T=await import(pathToFileURL(path.join(root,'build/three.core.js')));
function geometry(){return new T.BufferGeometry()
  .setAttribute('position',new T.Float32BufferAttribute([-.25,-.25,0,.25,-.25,0,0,.25,0],3))
  .setAttribute('normal',new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));}
function fixture(n=3,material=new T.MeshBasicMaterial(),d=geometryDevice()){
  const scene=new T.Scene(),g=geometry(),mesh=new T.InstancedMesh(g,material,n),camera=new T.PerspectiveCamera(60,1,.1,100);
  camera.position.z=3;mesh.frustumCulled=false;scene.add(mesh,new T.AmbientLight(0xffffff));
  return {d,scene,g,mesh,material,camera};
}
const frame=()=>({colorView:{},depthView:{}});
const create=(f,options={})=>createGpuThreeScene(f.d,f.scene,{three:T,sortObjects:false,...options});
function attribute(draw,location){
  const slot=draw.pipeline.vertex.buffers.findIndex(layout=>layout.attributes.some(a=>a.shaderLocation===location));
  assert.ok(slot>=0,`missing source attribute ${location}`);return draw.streams.get(slot);
}
const words=(draw,buffer)=>new Float32Array(draw.contents.get(buffer).buffer);
const marks=d=>[d.buffers.length,d.writes.length,d.submissions.length];
const tick=()=>new Promise(resolve=>setImmediate(resolve));

for(const instancing of [false,true])for(const renderBundles of [false,true])for(const indexed of [false,true])
test(`live InstancedMesh scene submits one packet for 1024 source instances (${instancing}/${renderBundles}/${indexed})`,async()=>{
  const f=fixture(1024);if(indexed)f.g.setIndex([0,1,2]);
  f.mesh.setColorAt(0,new T.Color(.2,.4,.8));
  const sourceMatrix=f.mesh.instanceMatrix,sourceColor=f.mesh.instanceColor,b=await create(f,{renderer:{instancing,renderBundles,maxDraws:1}});
  const allocations=f.d.buffers.length;
  f.mesh.getMatrixAt=()=>{throw new Error('No per-instance CPU render traversal is allowed here');};
  f.mesh.getColorAt=()=>{throw new Error('No instance color repacking');};
  for(let i=0;i<4;i++){
    f.mesh.position.x=.01*i;f.mesh.count=1024-i;
    sourceMatrix.array[12]=i*.125;sourceMatrix.needsUpdate=true;
    sourceColor.array[1]=i*.125;sourceColor.needsUpdate=true;
    b.render(f.camera,frame());const draw=f.d.snapshots.at(-1)[0];
    assert.deepEqual(draw.args,indexed?[3,1024-i,0,0,0]:[3,1024-i,0,0]);
    assert.equal(words(draw,attribute(draw,5))[12],i*.125);assert.equal(words(draw,attribute(draw,9))[1],i*.125);
    assert.equal(draw.groups.get(0).group.layout.entries[0].buffer.type,'uniform');
  }
  assert.equal(f.d.buffers.length,allocations);assert.equal(b.diagnostics.geometryCount,1);
  assert.equal(b.diagnostics.instanceMeshCount,1);assert.equal(b.diagnostics.instanceBytes,1024*76);
  assert.equal(b.diagnostics.materialBindings,1);assert.equal(b.diagnostics.logicalDraws,1);assert.equal(b.diagnostics.drawCalls,1);
  assert.equal(f.mesh.instanceMatrix,sourceMatrix);assert.equal(f.mesh.instanceColor,sourceColor);assert.equal(f.mesh.geometry,f.g);
  if(renderBundles)assert.equal(b.diagnostics.bundles.builds,4,'count changes invalidate only the schedule');
  await b.whenIdle();b.dispose();assert.ok(f.d.buffers.every(buffer=>buffer.destroyed));
  assert.equal(f.mesh.instanceMatrix,sourceMatrix);assert.equal(b.diagnostics.instanceBytes,0);
});

test('shared geometry/material does not alias different native sources or ordinary mesh batches',async()=>{
  const f=fixture(2),other=new T.InstancedMesh(f.g,f.material,5);other.frustumCulled=false;
  other.setMatrixAt(0,new T.Matrix4().makeTranslation(.5,0,0));f.mesh.setMatrixAt(0,new T.Matrix4().makeTranslation(-.5,0,0));
  const a=new T.Mesh(f.g,f.material),c=new T.Mesh(f.g,f.material);a.frustumCulled=c.frustumCulled=false;
  f.scene.add(other,a,c);const b=await create(f,{renderer:{instancing:true,renderBundles:true,maxDraws:4}});
  b.render(f.camera,frame());const draws=f.d.snapshots.at(-1);
  assert.deepEqual(draws.map(x=>x.args),[[3,2,0,0],[3,5,0,0],[3,2,0,2]]);
  assert.equal(attribute(draws[0],0),attribute(draws[1],0));assert.equal(attribute(draws[1],0),attribute(draws[2],0));
  assert.notEqual(attribute(draws[0],5),attribute(draws[1],5));
  assert.equal(words(draws[0],attribute(draws[0],5))[12],-.5);assert.equal(words(draws[1],attribute(draws[1],5))[12],.5);
  assert.equal(b.diagnostics.geometryCount,1);assert.equal(b.diagnostics.materialBindings,3);assert.equal(b.diagnostics.instanceMeshCount,2);
  assert.equal(b.diagnostics.logicalDraws,4);assert.equal(b.diagnostics.drawCalls,3);
  b.render(f.camera,frame());assert.equal(b.diagnostics.bundles.reuses,1);await b.whenIdle();b.dispose();
});

test('automatic instance uploads honor needsUpdate, partial ranges and earlier submitted byte history',async()=>{
  const f=fixture(),b=await create(f,{renderer:{renderBundles:true}}),a=f.mesh.instanceMatrix;
  b.render(f.camera,frame());const first=f.d.snapshots[0][0],buffer=attribute(first,5);
  a.array[12]=.75;a.array[13]=.5;b.render(f.camera,frame());assert.equal(words(f.d.snapshots[1][0],buffer)[12],0);
  a.addUpdateRange(12,1);a.needsUpdate=true;b.render(f.camera,frame());
  assert.equal(words(f.d.snapshots[2][0],buffer)[12],.75);assert.equal(words(f.d.snapshots[2][0],buffer)[13],0);
  assert.equal(words(first,buffer)[12],0);assert.equal(a.updateRanges.length,0);
  assert.equal(b.diagnostics.bundles.builds,1);assert.equal(b.diagnostics.bundles.reuses,2);
  f.mesh.count=0;b.render(f.camera,frame());assert.equal(f.d.snapshots.at(-1)[0].args[1],0);
  assert.equal(b.diagnostics.bundles.builds,2);await b.whenIdle();b.dispose();
});

test('adding/removing native RGB is a prepare boundary, but live RGB changes keep the binding',async()=>{
  const f=fixture(),b=await create(f,{renderer:{renderBundles:true}});b.render(f.camera,frame());
  f.mesh.setColorAt(0,new T.Color(.5,.25,1));const before=marks(f.d);
  assert.throws(()=>b.render(f.camera,frame()),{code:'THREE_SCENE_PREPARE'});assert.deepEqual(marks(f.d),before);
  await b.prepare();b.render(f.camera,frame());const color=attribute(f.d.snapshots.at(-1)[0],9),allocations=f.d.buffers.length;
  f.mesh.instanceColor.array[0]=.125;f.mesh.instanceColor.needsUpdate=true;b.render(f.camera,frame());
  assert.equal(attribute(f.d.snapshots.at(-1)[0],9),color);assert.equal(words(f.d.snapshots.at(-1)[0],color)[0],.125);
  assert.equal(f.d.buffers.length,allocations);assert.equal(b.diagnostics.bundles.reuses,1);
  f.mesh.instanceColor=null;assert.throws(()=>b.render(f.camera,frame()),{code:'THREE_SCENE_PREPARE'});
  await b.prepare();b.render(f.camera,frame());assert.doesNotMatch(f.d.snapshots.at(-1)[0].pipeline.vertex.module.code,/instance_color/);
  assert.equal(f.material.vertexColors,false,'native colors are independent of vertexColors');await b.whenIdle();b.dispose();
});

test('native groups and drawRange use one instance owner for every material draw',async()=>{
  const f=fixture(4),green=new T.MeshBasicMaterial({color:0x00ff00});
  f.g.setIndex([0,1,2,0,1,2]);f.g.addGroup(0,3,0);f.g.addGroup(3,3,1);f.g.addGroup(0,6,9);
  f.g.setDrawRange(1,4);f.mesh.material=[f.material,green];const b=await create(f);
  b.render(f.camera,frame());const draws=f.d.snapshots.at(-1);
  assert.deepEqual(draws.map(x=>x.args),[[2,4,1,0,0],[2,4,3,0,0]]);
  assert.equal(attribute(draws[0],5),attribute(draws[1],5));assert.equal(b.diagnostics.instanceMeshCount,1);
  assert.equal(b.diagnostics.materialBindings,2);await b.whenIdle();b.dispose();
});

test('double-sided transparent native draws keep source order and upload matrices once',async()=>{
  const f=fixture(3,new T.MeshPhongMaterial({transparent:true,opacity:.5,side:T.DoubleSide}));
  const b=await create(f,{renderer:{instancing:true,renderBundles:true}}),version=f.material.version;
  f.mesh.instanceMatrix.needsUpdate=true;const before=f.d.writes.length;b.render(f.camera,frame());const draws=f.d.snapshots.at(-1);
  assert.deepEqual(draws.map(x=>x.args[1]),[3,3]);assert.deepEqual(draws.map(x=>x.pipeline.primitive.cullMode),['front','back']);
  const matrix=attribute(draws[0],5);assert.equal(f.d.writes.slice(before).filter(x=>x.buffer===matrix).length,1);
  assert.equal(f.material.version,version+2);assert.equal(f.material.side,T.DoubleSide);
  assert.equal(b.diagnostics.sourceDraws,1);assert.equal(b.diagnostics.logicalDraws,2);await b.whenIdle();b.dispose();
});

test('source instance bounds, visibility and layers govern culling without implicit recomputation',async()=>{
  const f=fixture(1),ordinary=new T.Mesh(f.g,f.material);f.mesh.frustumCulled=true;f.scene.add(ordinary);
  f.mesh.setMatrixAt(0,new T.Matrix4().makeTranslation(100,0,0));const b=await create(f,{sortObjects:true});
  b.render(f.camera,frame());assert.equal(b.diagnostics.sourceDraws,1);
  assert.equal(f.mesh.boundingSphere.center.x,100);assert.equal(f.g.boundingSphere.center.x,0);
  f.mesh.setMatrixAt(0,new T.Matrix4());f.mesh.instanceMatrix.needsUpdate=true;
  b.render(f.camera,frame());assert.equal(b.diagnostics.sourceDraws,1,'source cached bounds stay authoritative');
  f.mesh.computeBoundingSphere();b.render(f.camera,frame());assert.equal(b.diagnostics.sourceDraws,2);
  f.mesh.layers.set(1);b.render(f.camera,frame());assert.equal(b.diagnostics.sourceDraws,1);
  f.camera.layers.enable(1);b.render(f.camera,frame());assert.equal(b.diagnostics.sourceDraws,2);
  f.mesh.visible=false;b.render(f.camera,frame());assert.equal(b.diagnostics.sourceDraws,1);await b.whenIdle();b.dispose();
});

test('native source disposal and graph pruning retire only owned streams, not shared geometry',async()=>{
  const f=fixture(),ordinary=new T.Mesh(f.g,f.material);ordinary.frustumCulled=false;f.scene.add(ordinary);
  const b=await create(f,{renderer:{renderBundles:true}});b.render(f.camera,frame());const draw=f.d.snapshots[0][0],matrix=attribute(draw,5),vertices=attribute(draw,0);
  f.mesh.dispose();assert.ok(matrix.destroyed);assert.equal(vertices.destroyed,false);assert.equal(b.diagnostics.instanceBytes,0);
  b.render(f.camera,frame());assert.notEqual(attribute(f.d.snapshots.at(-1)[0],5),matrix);assert.equal(b.diagnostics.bundles.builds,2);
  f.scene.remove(f.mesh);await b.prepare();assert.equal(b.diagnostics.instanceMeshCount,0);assert.equal(b.diagnostics.instanceBytes,0);
  assert.equal(b.diagnostics.geometryCount,1);assert.equal(vertices.destroyed,false);b.render(f.camera,frame());
  assert.equal(b.diagnostics.logicalDraws,1);await b.whenIdle();b.dispose();assert.equal(f.mesh.geometry,f.g);
});

test('geometry replacement is not hidden by an unchanged native source identity',async()=>{
  const f=fixture(),b=await create(f);b.render(f.camera,frame());const matrix=attribute(f.d.snapshots[0][0],5),old=attribute(f.d.snapshots[0][0],0);
  f.mesh.geometry=geometry();f.mesh.geometry.attributes.position.array[0]=-.5;const before=marks(f.d);
  assert.throws(()=>b.render(f.camera,frame()),{code:'THREE_SCENE_PREPARE'});assert.deepEqual(marks(f.d),before);
  await b.prepare();b.render(f.camera,frame());const draw=f.d.snapshots.at(-1)[0];assert.equal(attribute(draw,5),matrix);
  assert.notEqual(attribute(draw,0),old);assert.equal(old.destroyed,true);assert.equal(b.diagnostics.geometryCount,1);await b.whenIdle();b.dispose();
});

test('owner pruning waits for submitted dependencies and does not invalidate earlier completion',async()=>{
  const f=fixture(),ordinary=new T.Mesh(f.g,f.material);ordinary.frustumCulled=false;f.scene.add(ordinary);
  const b=await create(f);await b.whenIdle();let finish;
  f.d.completion=new Promise(resolve=>finish=resolve);b.render(f.camera,frame());
  const matrix=attribute(f.d.snapshots.at(-1)[0],5),previous=b.whenIdle();f.scene.remove(f.mesh);
  let done=false;const preparing=b.prepare().then(()=>done=true);await tick();
  assert.equal(done,false);assert.equal(matrix.destroyed,false,'previous submitted use retains its owner');
  finish();await previous;await preparing;assert.equal(matrix.destroyed,true);assert.equal(b.failed,false);
  b.render(f.camera,frame());await b.whenIdle();b.dispose();
});

test('disposing during retirement drain ends preparation without awaiting the device forever',async()=>{
  const f=fixture(),b=await create(f);await b.whenIdle();f.d.completion=new Promise(()=>{});
  b.render(f.camera,frame());f.scene.remove(f.mesh);const preparing=b.prepare();await tick();
  b.dispose();await assert.rejects(preparing,{code:'THREE_SCENE_DISPOSED'});
  assert.ok(f.d.buffers.every(buffer=>buffer.destroyed));assert.equal(b.diagnostics.instanceMeshCount,0);
});

test('instance allocations have an independent exact aggregate and old-plus-new peak budget',async()=>{
  const f=fixture(2),b=await create(f,{maxGeometryBytes:72,maxInstanceBytes:128});
  assert.equal(b.diagnostics.geometryBytes,72);assert.equal(b.diagnostics.instanceBytes,128);
  const second=new T.InstancedMesh(f.g,f.material,2);f.scene.add(second);let before=marks(f.d);
  await assert.rejects(b.prepare(),/budget/);assert.deepEqual(marks(f.d),before);assert.equal(b.diagnostics.instanceMeshCount,1);f.scene.remove(second);
  f.mesh.instanceMatrix=new T.InstancedBufferAttribute(f.mesh.instanceMatrix.array.slice(),16);before=marks(f.d);
  await assert.rejects(b.prepare(),/budget/);assert.deepEqual(marks(f.d),before);
  f.mesh.dispose();await b.prepare();b.render(f.camera,frame());assert.equal(b.diagnostics.instanceBytes,128);
  assert.equal(b.diagnostics.geometryBytes,72);await b.whenIdle();b.dispose();
});

test('native count and owner-count bounds fail before frame publication and remain recoverable',async()=>{
  const f=fixture(2),b=await create(f,{maxInstanceMeshes:1});f.mesh.count=3;const before=marks(f.d);
  assert.throws(()=>b.render(f.camera,frame()),{code:'GEOMETRY_GPU_SHAPE'});assert.deepEqual(marks(f.d),before);f.mesh.count=2;
  f.scene.add(new T.InstancedMesh(f.g,f.material,2));await assert.rejects(b.prepare(),{code:'THREE_SCENE_LIMIT'});assert.deepEqual(marks(f.d),before);
  f.scene.remove(f.scene.children.at(-1));await b.prepare();b.render(f.camera,frame());assert.equal(b.failed,false);await b.whenIdle();b.dispose();
});

test('zero-capacity source instance buffers are valid and inspector is not a GPU handle',async()=>{
  const f=fixture(0),description=inspectInstanceAttributes(f.mesh);
  assert.equal(description.capacity,0);assert.equal(description.instanceCount,0);assert.equal(f.d.buffers.length,0);
  const b=await create(f);b.render(f.camera,frame());assert.equal(f.d.snapshots.at(-1)[0].args[1],0);
  assert.equal(b.diagnostics.instanceBytes,4);await b.whenIdle();b.dispose();
});

test('unsupported native morph/divisor/storage/callback inputs fail before any GPU allocation',async()=>{
  for(const edit of [
    f=>{f.mesh.morphTexture={};},f=>{f.mesh.instanceMatrix.meshPerAttribute=2;},
    f=>{f.mesh.instanceMatrix.normalized=true;},f=>{f.mesh.instanceMatrix.itemSize=4;},
    f=>{f.mesh.instanceMatrix.array=new Float64Array(48);},f=>{f.mesh.count=-1;},
    f=>{f.mesh.instanceMatrix.onUpload(()=>{throw new Error('must never run');});},
    f=>{f.mesh.visible=false;f.mesh.morphTexture={};},
  ]){const f=fixture();edit(f);await assert.rejects(create(f));assert.equal(f.d.buffers.length,0);assert.equal(f.d.writes.length,0);}
});

test('native layouts changing during pipeline waits cannot publish stale registrations',async()=>{
  const f=fixture(),b=await create(f),original=f.d.createRenderPipelineAsync,waiting=[];
  f.mesh.setColorAt(0,new T.Color());f.d.createRenderPipelineAsync=desc=>new Promise(resolve=>waiting.push(()=>resolve(original(desc))));
  const pending=b.prepare();assert.ok(waiting.length);f.mesh.instanceColor=null;waiting.forEach(resolve=>resolve());
  await assert.rejects(pending,{code:'THREE_SCENE_CHANGED'});assert.equal(b.diagnostics.prepareVersion,1);
  f.d.createRenderPipelineAsync=original;await b.prepare();b.render(f.camera,frame());assert.doesNotMatch(f.d.snapshots.at(-1)[0].pipeline.vertex.module.code,/instance_color/);
  await b.whenIdle();b.dispose();
});

test('native data/count changes while pipelines wait publish current data without new logical draws',async()=>{
  const f=fixture(),b=await create(f),original=f.d.createRenderPipelineAsync,waiting=[];
  f.mesh.setColorAt(0,new T.Color());f.d.createRenderPipelineAsync=desc=>new Promise(resolve=>waiting.push(()=>resolve(original(desc))));
  const pending=b.prepare();assert.ok(waiting.length);
  f.mesh.instanceMatrix.array[12]=.5;f.mesh.instanceMatrix.needsUpdate=true;f.mesh.count=2;waiting.forEach(resolve=>resolve());
  await pending;b.render(f.camera,frame());const draw=f.d.snapshots.at(-1)[0];assert.equal(words(draw,attribute(draw,5))[12],.5);assert.equal(draw.args[1],2);
  await b.whenIdle();b.dispose();
});

test('disposal during native preparation blocks late publication and frees all owned residency',async()=>{
  const f=fixture(),b=await create(f),waiting=[];
  f.mesh.setColorAt(0,new T.Color());f.d.createRenderPipelineAsync=desc=>new Promise(resolve=>waiting.push(()=>resolve(desc)));
  const pending=b.prepare();assert.ok(waiting.length);b.dispose();await assert.rejects(pending,{code:'THREE_SCENE_DISPOSED'});
  assert.ok(f.d.buffers.every(x=>x.destroyed));waiting.forEach(resolve=>resolve());await tick();assert.equal(b.diagnostics.materialBindings,0);
  assert.equal(b.diagnostics.instanceBytes,0);assert.equal(f.mesh.instanceMatrix.count,3);
});

test('a native upload failure terminates scene residency without disposing source objects',async()=>{
  const f=fixture(),b=await create(f);f.mesh.instanceMatrix.needsUpdate=true;f.d.writeError=new Error('native instance write failure');
  assert.throws(()=>b.render(f.camera,frame()),/native instance write failure/);assert.equal(b.failed,true);
  assert.ok(f.d.buffers.every(x=>x.destroyed));assert.equal(f.mesh.geometry,f.g);assert.equal(f.mesh.instanceMatrix.count,3);b.dispose();await tick();
});

test('disposal ends stalled native queue waits without claiming to cancel submitted work',async()=>{
  const f=fixture(),b=await create(f);let complete;f.d.completion=new Promise(resolve=>{complete=resolve;});
  b.render(f.camera,frame());const pending=b.whenIdle();b.dispose();await assert.rejects(pending,{code:'THREE_SCENE_DISPOSED'});
  assert.ok(f.d.buffers.every(x=>x.destroyed));complete();await tick();assert.equal(f.mesh.instanceMatrix.count,3);
});

test('automatic source textures and native instance updates share stable material/bundle bindings',async()=>{
  const f=fixture(2,new T.MeshPhongMaterial(),textureDevice());
  f.g.setAttribute('uv',new T.Float32BufferAttribute([0,0,1,0,.5,1],2));
  const texture=new T.DataTexture(new Uint8Array(16).fill(64),2,2);texture.needsUpdate=true;f.material.map=texture;
  f.mesh.setColorAt(0,new T.Color(.5,1,.25));const b=await create(f,{renderer:{instancing:true,renderBundles:true}});
  b.render(f.camera,frame());const first=f.d.snapshots.at(-1)[0],nativeTexture=f.d.textures[0];
  texture.image.data.fill(192);texture.needsUpdate=true;f.mesh.instanceMatrix.array[12]=.25;f.mesh.instanceMatrix.needsUpdate=true;
  b.render(f.camera,frame());const second=f.d.snapshots.at(-1)[0];
  assert.equal(first.groups.get(1).group,second.groups.get(1).group);assert.ok(second.textureContents.get(nativeTexture)[0].every(x=>x===192));
  assert.ok(first.textureContents.get(nativeTexture)[0].every(x=>x===64));assert.equal(words(second,attribute(second,5))[12],.25);
  assert.equal(b.diagnostics.bundles.reuses,1);assert.equal(b.diagnostics.textures.resources,1);await b.whenIdle();b.dispose();assert.ok(nativeTexture.destroyed);
});

test('compiled pinned Wasm MarchingCubes geometry reaches native source instances without expansion',async()=>{
  const {buildApplication}=await import('./build_application.mjs');
  const {readMarchingCubesOracle}=await import('./fixtures/marching_cubes_oracle.mjs');readMarchingCubesOracle();
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-native-cubes-')),entry=path.join(dir,'entry.mjs');
  await fs.writeFile(entry,`export * from 'three';\nexport {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';\nexport {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};\n`);
  const built=await buildApplication(entry,path.join(dir,'built'),{packageRootUrl:pathToFileURL(path.resolve(root)+path.sep).href,specializeNumeric:true});
  await fs.writeFile(path.join(built.outDir,'package.json'),' {"type":"module"} ',{flag:'wx'});
  const S=await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0]))),material=new S.MeshPhongMaterial();
  const effect=new S.MarchingCubes(8,material,false,false,2000),scene=new S.Scene(),camera=new S.PerspectiveCamera(60,1,.1,100);camera.position.z=3;
  effect.addBall(.5,.5,.5,1.2,8);effect.update();const mesh=new S.InstancedMesh(effect.geometry,material,64);mesh.frustumCulled=false;
  scene.add(mesh,new S.AmbientLight());const d=geometryDevice(),b=await createGpuThreeScene(d,scene,{three:S,sortObjects:false,renderer:{maxDraws:1,renderBundles:true}});
  for(let i=0;i<3;i++){
    if(i){effect.reset();effect.addBall(.5,.5,.5,1.2,8);effect.update();}
    mesh.instanceMatrix.array[12]=i*.125;mesh.instanceMatrix.needsUpdate=true;b.render(camera,frame());const draw=d.snapshots.at(-1)[0];
    assert.deepEqual(draw.args,[effect.count,64,0,0]);assert.deepEqual(words(draw,attribute(draw,0)),effect.positionArray);
    assert.equal(words(draw,attribute(draw,5))[12],i*.125);
  }
  assert.equal(S.diagnostics(effect).wasmCalls,3);assert.equal(S.diagnostics(effect).fieldKernels.addBall.wasmCalls,3);assert.equal(b.diagnostics.geometryCount,1);assert.equal(b.diagnostics.logicalDraws,1);
  assert.equal(b.diagnostics.bundles.reuses,2);await b.whenIdle();b.dispose();assert.equal(mesh.geometry,effect.geometry);
});

test('relocated optional scene packages expose native instance ownership and execute it',async()=>{
  const {buildAnimation}=await import('./build_animation.mjs'),{animationFixture}=await import('./fixtures/animation/gltf_fixture.mjs');
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-native-package-'));
  const input=path.join(dir,'scene.gltf'),asset=animationFixture();await fs.writeFile(input,JSON.stringify(asset.model));
  await fs.writeFile(path.join(dir,'clip data.bin'),asset.bytes);
  const out=path.join(dir,'generated');await buildAnimation(input,out,{webgpu:true,threeScene:true});
  const relocated=path.join(dir,'moved');await fs.rename(out,relocated);await fs.writeFile(path.join(relocated,'package.json'),'{"type":"module"}');
  const api=await import(pathToFileURL(path.join(relocated,'gpu_playback.mjs')));
  assert.equal(typeof api.createGpuInstanceAttributes,'function');const f=fixture(7);
  const b=await api.createGpuThreeScene(f.d,f.scene,{three:T,sortObjects:false,renderer:{maxDraws:1,renderBundles:true}});
  b.render(f.camera,frame());assert.deepEqual(f.d.snapshots.at(-1)[0].args,[3,7,0,0]);await b.whenIdle();b.dispose();
});
