/** Production builder, source scene and fog modules after relocation. Decoder,
 * player, native GPU services and unrelated optional factories are fixtures;
 * this proves deployable host integration, not asset or native pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {copyFogSources,writeServiceFixtures,sourceModule,deviceFixture} from './three_scene_fog_test_support.mjs';
const bytes=name=>fs.readFileSync(new URL('./'+name,import.meta.url));
const hash=data=>createHash('sha256').update(data).digest('hex');
async function fixture(t) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-source-fog-package-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const kit=path.join(root,'kit');fs.mkdirSync(kit);
  const code=bytes('build_animation.mjs').toString(),modules=new Map();
  for(const [,name] of code.matchAll(/["']([a-z_]+\.mjs)["']/g))modules.set(name,new Set());
  for(const [,exports,name] of code.matchAll(/export \{([^}]+)\} from '\.\/([a-z_]+\.mjs)'/g)) {
    const names=modules.get(name)??new Set();modules.set(name,names);
    for(const item of exports.split(',').map(x=>x.trim()))if(/^[A-Za-z]\w*$/.test(item))names.add(item);
  }
  for(const [name,names] of modules)fs.writeFileSync(path.join(kit,name),[...names].map(n=>
    `export function ${n}(){throw new Error('Unexercised fixture: ${n}');}\n`).join(''));
  fs.writeFileSync(path.join(kit,'animation_gltf.mjs'),`export function decodeGltfAnimation(model){return {nodes:model.nodes??[],ignoredChannels:[]};}`);
  fs.writeFileSync(path.join(kit,'gltf_instancing.mjs'),`export function expandGltfInstances(json){return {json,instanceCount:0};}`);
  fs.writeFileSync(path.join(kit,'animation_runtime.mjs'),
    `export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}\n`+
    `export function createAnimationPlayer(d){return {nodeCount:d.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}};}`);
  writeServiceFixtures(kit);copyFogSources(kit);
  // Public deformation exports not involved in fog still have explicit traps.
  fs.appendFileSync(path.join(kit,'three_deformation.mjs'),
    `\nexport class ThreeDeformationError extends Error{}\nexport function createThreeDeformationBinding(){throw new Error('Unexercised binding fixture');}\n`);
  fs.writeFileSync(path.join(kit,'build_animation.mjs'),code);
  const input=path.join(root,'actor.gltf');fs.writeFileSync(input,JSON.stringify({asset:{version:'2.0'},nodes:[{}]}));
  const {buildAnimation}=await import(pathToFileURL(path.join(kit,'build_animation.mjs')));
  let index=0;
  return {root,kit,input,buildAnimation,build:options=>buildAnimation(input,path.join(root,'out-'+index++),options)};
}
function artifacts(result) {
  assert.equal(new Set(result.emittedFiles).size,result.emittedFiles.length);
  for(const item of result.artifacts) {
    const data=fs.readFileSync(path.join(result.outDir,item.file));assert.equal(item.bytes,data.length);assert.equal(item.sha256,hash(data));
  }
  assert.equal(result.outputBytes,fs.readdirSync(result.outDir).reduce((n,p)=>n+fs.statSync(path.join(result.outDir,p)).size,0));
}
function source() {
  const three=sourceModule(),scene=new three.Scene(),camera=new three.Camera(),device=deviceFixture();
  scene.fog=new three.Fog();
  for(const flag of [true,false,true]){const m=new three.MeshBasicMaterial();m.fog=flag;scene.add(new three.Mesh(undefined,m));}
  return {three,scene,camera,device};
}
const frame=()=>({colorView:{},depthView:{},loadOp:'clear',depthLoadOp:'clear'});

test('relocated package executes the production source bridge, lazy fog adapter and real native packing',async t=>{
  const f=await fixture(t),result=f.build({webgpu:true,threeScene:true,fog:true});artifacts(result);
  assert.match(result.gpuThreeFog,/live material receivers/);
  const moved=path.join(f.root,'deployment');fs.renameSync(result.outDir,moved);
  fs.renameSync(f.kit,path.join(f.root,'retired-kit'));fs.renameSync(f.input,path.join(f.root,'retired-input'));
  const s=source(),api=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')));
  assert.deepEqual(s.device.events,[]);assert.equal(typeof api.threeFogDescriptor,'function');
  const bridge=await api.createGpuThreeScene(s.device,s.scene,{three:s.three,fog:{},autoTextures:false,sortObjects:false});
  t.after(()=>bridge.dispose());bridge.render(s.camera,frame());assert.deepEqual(s.device.frames.map(x=>x.packed[11]),[1,0,1]);
  s.scene.fog=new s.three.FogExp2();bridge.render(s.camera,frame());assert.deepEqual(s.device.frames.slice(-3).map(x=>x.packed[11]),[2,0,2]);
  s.scene.fog=null;bridge.render(s.camera,frame());assert.equal(s.device.frames.at(-1).packed[11],0);
  assert.equal(s.device.registrations.length,3);
});

test('source package without fog stays independent and fails rather than reaching back into the toolkit',async t=>{
  const f=await fixture(t),result=f.build({webgpu:true,threeScene:true});artifacts(result);
  assert.ok(!result.emittedFiles.some(n=>n.includes('fog')));assert.equal(result.gpuThreeFog,undefined);
  const api=await import(pathToFileURL(path.join(result.outDir,'gpu_playback.mjs'))),s=source();s.scene.fog=null;
  assert.equal(api.threeFogDescriptor,undefined);
  const bridge=await api.createGpuThreeScene(s.device,s.scene,{three:s.three,autoTextures:false,sortObjects:false});
  t.after(()=>bridge.dispose());bridge.render(s.camera,frame());assert.equal(s.device.options[0].fog,undefined);
  const d=deviceFixture();await assert.rejects(api.createGpuThreeScene(d,s.scene,{three:s.three,fog:{}}),{code:'ERR_MODULE_NOT_FOUND'});
  assert.deepEqual(d.events,[]);
});

test('explicit GPU fog does not acquire a Three dependency or source-only exports',async t=>{
  const f=await fixture(t),result=f.build({webgpu:true,fog:true});artifacts(result);
  assert.ok(result.emittedFiles.includes('animation_fog.mjs'));assert.ok(!result.emittedFiles.includes('three_fog.mjs'));
  const api=await import(pathToFileURL(path.join(result.outDir,'gpu_playback.mjs')));
  assert.equal(api.threeFogDescriptor,undefined);assert.equal(api.createGpuThreeScene,undefined);
  assert.equal(typeof api.snapshotAnimationCameraFog,'function');assert.equal(result.gpuThreeFog,undefined);
});

test('default/disabled fog preserves CPU, ordinary GPU and source-package entry/file selection',async t=>{
  const f=await fixture(t);
  for(const options of [{},{webgpu:true},{webgpu:true,threeScene:true}]) {
    const a=f.build(options),b=f.build({...options,fog:false});assert.deepEqual(a.emittedFiles,b.emittedFiles);
    for(const name of a.emittedFiles)assert.deepEqual(fs.readFileSync(path.join(a.outDir,name)),fs.readFileSync(path.join(b.outDir,name)));
    assert.ok(!a.emittedFiles.some(name=>name.includes('fog')));
  }
  const a=f.build({webgpu:true,threeScene:true}),b=f.build({webgpu:true,threeScene:true,fog:true});
  for(const name of ['animation.mjs','animation.json','playback.mjs'])assert.deepEqual(fs.readFileSync(path.join(a.outDir,name)),fs.readFileSync(path.join(b.outDir,name)));
});

test('source fog is hashed and charged with the manifest at exact and one-byte-short output limits',async t=>{
  const f=await fixture(t),options={webgpu:true,threeScene:true,fog:true},initial=f.build(options);
  const exact=f.build({...options,maxBytes:initial.outputBytes});artifacts(exact);assert.equal(exact.outputBytes,initial.outputBytes);
  for(const name of ['three_fog.mjs','animation_fog.mjs','animation_fog_camera.mjs','three_scene.mjs'])
    assert.deepEqual(fs.readFileSync(path.join(initial.outDir,name)),bytes(name));
  const short=path.join(f.root,'short');
  assert.throws(()=>f.buildAnimation(f.input,short,{...options,maxBytes:initial.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  assert.equal(fs.existsSync(short),false);
});

test('source fog packaging coexists with IBL/HDR, backgrounds, recovery, rigid geometry and IK exports',async t=>{
  const f=await fixture(t);
  for(const options of [{environment:true,hdr:true},{background:true},{canvasRecovery:true},
    {environment:true,hdr:true,background:true,canvasRecovery:true,rigidGeometry:true,inverseKinematics:true}]) {
    const result=f.build({webgpu:true,threeScene:true,fog:true,...options});artifacts(result);
    const api=await import(pathToFileURL(path.join(result.outDir,'gpu_playback.mjs'))),s=source();
    const bridge=await api.createGpuThreeScene(s.device,s.scene,{three:s.three,fog:{},autoTextures:false,sortObjects:false});
    bridge.render(s.camera,frame());assert.deepEqual(s.device.frames.map(x=>x.packed[11]),[1,0,1]);bridge.dispose();
    if(options.canvasRecovery)assert.equal(typeof api.createRecoverableGpuThreeCanvas,'function');
    if(options.inverseKinematics)assert.equal(typeof api.solveAnimationLimbIK,'function');
  }
});
