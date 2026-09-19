import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {buildAnimation} from './build_animation.mjs';
import {decodeGltfAnimation} from './animation_gltf.mjs';
import {expandGltfInstances} from './gltf_instancing.mjs';
const EXT='EXT_mesh_gpu_instancing';
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${v} != ${b[i]}`));};
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const temporary=()=>fs.mkdtempSync(path.join(os.tmpdir(),'f3d-instance-package-'));
const importFile=file=>import(pathToFileURL(file).href);
function fixture(root,{embedded=false,instanced=true,morph=true}={}) {
  const dir=path.join(root,'input');fs.mkdirSync(dir);
  const model={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0,translation:[10,0,0]}],
    meshes:[{primitives:[{attributes:{POSITION:0}}]}],buffers:[],bufferViews:[],accessors:[]},buffers=[];
  function attr(values,type='SCALAR',name='data') {
    const data=new Float32Array(values),index=buffers.push(data)-1;
    const uri=embedded?'data:application/octet-stream;base64,'+Buffer.from(data.buffer).toString('base64'):name+'.bin';
    model.buffers.push({uri,byteLength:data.byteLength});model.bufferViews.push({buffer:index,byteLength:data.byteLength});
    const at=model.accessors.push({bufferView:index,componentType:5126,type,count:values.length/({SCALAR:1,VEC3:3,VEC4:4}[type])})-1;
    if(!embedded)fs.writeFileSync(path.join(dir,uri),new Uint8Array(data.buffer));return at;
  }
  attr([0,0,0,1,0,0,0,1,0],'VEC3','geometry');
  const transform=attr([1,0,0,0,2,0],'VEC3','instances');
  if(instanced){model.nodes[0].extensions={[EXT]:{attributes:{TRANSLATION:transform}}};model.extensionsUsed=[EXT];model.extensionsRequired=[EXT];}
  const times=attr([0,1],'SCALAR','times');Object.assign(model.accessors[times],{min:[0],max:[1]});
  const translations=attr([10,0,0,20,0,0],'VEC3','movement');
  model.animations=[{name:'move',samplers:[{input:times,output:translations}],channels:[{sampler:0,target:{node:0,path:'translation'}}]}];
  if(morph){model.meshes[0].primitives[0].targets=[{}];model.nodes[0].weights=[0.25];
    model.animations[0].samplers.push({input:times,output:attr([0,1],'SCALAR','weights')});
    model.animations[0].channels.push({sampler:1,target:{node:0,path:'weights'}});}
  const entry=path.join(dir,'model.gltf');fs.writeFileSync(entry,JSON.stringify(model));
  return {model,buffers,entry,dir,write(){fs.writeFileSync(entry,JSON.stringify(model));}};
}
// An isolated copy of the actual toolkit makes its ORIGINAL import path absent
// after building. No mocks for the builder, pose, controller or CPU deformer.
// Optional GPU modules below are explicit non-executing packaging boundaries;
// only the GPU entry's re-exports and bytes are checked, never GPU execution.
async function toolkit(root,gpu=false) {
  const directory=path.join(root,'toolkit');fs.mkdirSync(directory);
  for(const name of ['build_animation.mjs','gltf_instancing.mjs','animation_gltf.mjs','animation_runtime.mjs','animation_controller.mjs','animation_deformer.mjs'])
    fs.copyFileSync(new URL('./'+name,import.meta.url),path.join(directory,name));
  if(gpu)for(const [name,exports] of [
    ['animation_webgpu.mjs',['createGpuAnimationDeformer']],['animation_render.mjs',['createGpuAnimationRenderer']],
    ['animation_scene.mjs',['createGpuAnimationScene']],['animation_draw_order.mjs',[]],['animation_bounds.mjs',[]],
    ['animation_shadow.mjs',['createGpuAnimationShadowMap']],['animation_shadow_receiver.mjs',[]],['animation_scene_shadow.mjs',[]],
    ['animation_shadow_view.mjs',['fitAnimationShadowView','animationShadowWorldBounds']],
  ])fs.writeFileSync(path.join(directory,name),exports.map(n=>`export function ${n}(){throw Error('GPU execution is not under test');}\n`).join('')+'export {};\n');
  return {directory,build:(await importFile(path.join(directory,'build_animation.mjs'))).buildAnimation};
}
function contents(directory) {
  return Object.fromEntries(fs.readdirSync(directory).sort().map(name=>[name,fs.readFileSync(path.join(directory,name))]));
}

test('external instanced assets build with matching expanded pose IDs and complete buffer dependencies',()=>{
  const root=temporary(),f=fixture(root),out=path.join(root,'player'),result=buildAnimation(f.entry,out);
  const expected=decodeGltfAnimation(expandGltfInstances(f.model,f.buffers).json,f.buffers);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out,'animation.json'),'utf8')),expected);
  assert.equal(result.nodeCount,3);assert.equal(result.meshInstanceCount,2);assert.equal(result.instanceExecution,'expanded-node-mesh');
  assert.deepEqual(result.instanceOrigins,{1:{node:0,instance:0},2:{node:0,instance:1}});
  assert.deepEqual(result.dependencies.map(d=>d.uri),['instances.bin','times.bin','movement.bin','weights.bin']);
  for(const dep of result.dependencies)assert.equal(dep.sha256,sha(fs.readFileSync(path.join(f.dir,dep.uri))));
  assert.equal(result.morphWeightCount,2);assert.deepEqual(result.instances,[]);assert.equal(result.accelerationClaim,false);
  assert.deepEqual(JSON.parse(fs.readFileSync(f.entry,'utf8')),f.model);
});
for(const embedded of [false,true])test(`relocated ${embedded?'embedded':'external'} package runs real pose, action control and instance morphs without its original source`,async()=>{
  const root=temporary(),f=fixture(root,{embedded}),t=await toolkit(root),out=path.join(root,'built');
  t.build(f.entry,out);const destination=path.join(root,'relocated');fs.renameSync(out,destination);
  fs.renameSync(t.directory,t.directory+'-retained');fs.renameSync(f.dir,f.dir+'-retained');
  assert.equal(fs.existsSync(t.directory),false);assert.equal(fs.existsSync(f.dir),false);
  const module=await importFile(path.join(destination,'playback.mjs'));
  const p=module.createPlayer(),c=module.createAnimationController(p),meshes=[1,2].map(node=>module.createAnimationDeformer(p,{
    node,positions:[0,0,0,1,0,0,0,1,0],morphTargets:[{positions:[0,0,2,0,0,2,0,0,2]}],
  }));
  assert.deepEqual(module.instanceOrigins,{1:{node:0,instance:0},2:{node:0,instance:1}});
  assert.ok(Object.isFrozen(module.instanceOrigins)&&Object.isFrozen(module.instanceOrigins[1]));
  c.createAction(0,{loop:'once',clampWhenFinished:true}).play();c.update(0.5);meshes.forEach(m=>m.update());
  close(meshes[0].worldMatrix.slice(12,15),[16,0,0]);close(meshes[1].worldMatrix.slice(12,15),[15,2,0]);
  close(meshes[0].positions,[0,0,1,1,0,1,0,1,1]);close(meshes[1].positions,meshes[0].positions);
  c.update(0.5);meshes.forEach(m=>m.update());assert.equal(c.events[0].type,'finished');
  close(meshes[1].positions,[0,0,2,1,0,2,0,1,2]);assert.equal(p.morphWeights.length,2);
  meshes.forEach(m=>m.dispose());c.dispose();p.dispose();
});

test('the generated GPU entry re-exports the same immutable instance origins without evaluating GPU work',async()=>{
  const root=temporary(),f=fixture(root),t=await toolkit(root,true),out=path.join(root,'gpu');
  t.build(f.entry,out,{webgpu:true});fs.renameSync(t.directory,t.directory+'-retained');
  const a=await importFile(path.join(out,'animation.mjs')),gpu=await importFile(path.join(out,'gpu_playback.mjs'));
  assert.equal(a.instanceOrigins,gpu.instanceOrigins);const p=gpu.createPlayer();p.sample(0.5);
  close(p.worldMatrices.slice(28,31),[16,0,0]);p.dispose();
});

test('geometry-only buffers are not read, while instance transforms are required even for a static asset',async()=>{
  const root=temporary(),f=fixture(root,{morph:false});f.model.buffers[0].uri='not-present-geometry.bin';f.model.animations=[];f.write();
  const out=path.join(root,'static'),result=buildAnimation(f.entry,out);assert.deepEqual(result.dependencies.map(d=>d.uri),['instances.bin']);
  const {createPlayer}=await importFile(path.join(out,'animation.mjs')),p=createPlayer();close(p.worldMatrices.slice(28,31),[11,0,0]);p.dispose();
  f.model.buffers[1].uri='missing-instance-transforms.bin';f.write();
  const bad=path.join(root,'missing');assert.throws(()=>buildAnimation(f.entry,bad));assert.equal(fs.existsSync(bad),false);
});
for(const options of [{maxInstances:1},{maxComponents:29},{maxInstances:0},{maxInstances:65537}])test(`expansion limits reject before output creation: ${JSON.stringify(options)}`,()=>{
  const root=temporary(),f=fixture(root),out=path.join(root,'out');
  assert.throws(()=>buildAnimation(f.entry,out,options),{code:'GLTF_INSTANCING_LIMIT'});assert.equal(fs.existsSync(out),false);
});
for(const gpu of [false,true])test(`exact ${gpu?'GPU':'CPU'} output budget includes exported origin metadata and rejects one byte short`,async()=>{
  const root=temporary(),f=fixture(root),t=await toolkit(root,gpu),first=path.join(root,'first');
  const initial=t.build(f.entry,first,{webgpu:gpu}),out=path.join(root,'exact');
  const exact=t.build(f.entry,out,{webgpu:gpu,maxBytes:initial.outputBytes});assert.equal(exact.outputBytes,initial.outputBytes);
  assert.deepEqual(contents(first),contents(out));
  const failed=path.join(root,'short');assert.throws(()=>t.build(f.entry,failed,{webgpu:gpu,maxBytes:initial.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  assert.equal(fs.existsSync(failed),false);
});

test('instance buffers follow the existing root and no-network policy',()=>{
  for(const uri of ['../outside.bin','https://example.invalid/instances.bin']) {
    const root=temporary(),f=fixture(root),out=path.join(root,'out');fs.writeFileSync(path.join(root,'outside.bin'),new Uint8Array(f.buffers[1].buffer));
    f.model.buffers[1].uri=uri;f.write();
    assert.throws(()=>buildAnimation(f.entry,out),{code:uri.startsWith('https:')?'GLTF_ANIMATION_NETWORK':'GLTF_ANIMATION_ROOT'});
    assert.equal(fs.existsSync(out),false);
  }
});

test('ordinary packages retain the existing public exports and omit instance metadata',async()=>{
  const root=temporary(),f=fixture(root,{instanced:false}),out=path.join(root,'ordinary'),r=buildAnimation(f.entry,out);
  assert.equal(Object.hasOwn(r,'instanceOrigins'),false);assert.equal(Object.hasOwn(r,'meshInstanceCount'),false);
  const m=await importFile(path.join(out,'playback.mjs'));assert.deepEqual(Object.keys(m).sort(),['createAnimationController','createAnimationDeformer','createPlayer']);
  assert.ok(!r.dependencies.some(d=>d.uri==='instances.bin'));assert.equal(r.nodeCount,1);
});

test('GLB instance/accessor buffers use the BIN chunk without external dependencies',async()=>{
  const root=temporary(),f=fixture(root),model=structuredClone(f.model);
  let offset=0;const parts=[];
  for(const [i,b] of f.buffers.entries()){
    model.bufferViews[i].buffer=0;model.bufferViews[i].byteOffset=offset;
    const data=Buffer.from(b.buffer);parts.push(data);offset+=data.length;
  }
  model.buffers=[{byteLength:offset}];const bin=Buffer.concat(parts),text=Buffer.from(JSON.stringify(model));
  const size=Math.ceil(text.length/4)*4,bytes=Buffer.alloc(28+size+bin.length);
  bytes.writeUInt32LE(0x46546c67,0);bytes.writeUInt32LE(2,4);bytes.writeUInt32LE(bytes.length,8);
  bytes.writeUInt32LE(size,12);bytes.writeUInt32LE(0x4e4f534a,16);bytes.fill(32,20,20+size);text.copy(bytes,20);
  bytes.writeUInt32LE(bin.length,20+size);bytes.writeUInt32LE(0x004e4942,24+size);bin.copy(bytes,28+size);
  const entry=path.join(f.dir,'model.glb');fs.writeFileSync(entry,bytes);
  const out=path.join(root,'player'),result=buildAnimation(entry,out);
  assert.deepEqual(result.dependencies.map(d=>d.uri),['#BIN']);assert.equal(result.meshInstanceCount,2);
  const module=await importFile(path.join(out,'animation.mjs')),p=module.createPlayer();p.sample(0.5);
  close(p.worldMatrices.slice(28,31),[16,0,0]);close(p.morphWeights,[0.5,0.5]);p.dispose();
});
