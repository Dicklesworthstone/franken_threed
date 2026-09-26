/** Executes the real builder and emitted fog modules. Asset decoding, the pose
 * player and unrelated renderer/controller factories are explicit fixtures:
 * this suite proves relocation, lazy dependency closure and package accounting,
 * not asset decoding, shader compilation, GPU submission or source-scene parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const source = new URL('./build_animation.mjs',import.meta.url);
const bytes = name => fs.readFileSync(new URL('./'+name,import.meta.url));
const hash = data => createHash('sha256').update(data).digest('hex');
const I = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
async function fixture(t) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-fog-package-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const kit=path.join(root,'kit'); fs.mkdirSync(kit);
  const code=fs.readFileSync(source,'utf8');
  // Unexercised modules export traps, not implementations claiming coverage.
  const modules=new Map();
  for(const [,name] of code.matchAll(/["']([a-z_]+\.mjs)["']/g)) modules.set(name,new Set());
  for(const [,exports,name] of code.matchAll(/export \{([^}]+)\} from '\.\/([a-z_]+\.mjs)'/g)) {
    const names=modules.get(name)??new Set(); modules.set(name,names);
    for(const item of exports.split(',')) if(/^[A-Za-z]\w*$/.test(item)) names.add(item);
  }
  for(const [name,names] of modules) fs.writeFileSync(path.join(kit,name),
    [...names].map(n=>`export function ${n}(){throw new Error('Unexercised fixture: ${n}');}\n`).join(''));
  fs.writeFileSync(path.join(kit,'animation_gltf.mjs'),
    `export function decodeGltfAnimation(model){return {format:'fixture-pose',nodes:model.nodes??[],ignoredChannels:[]};}\n`);
  fs.writeFileSync(path.join(kit,'gltf_instancing.mjs'),
    `export function expandGltfInstances(json){return {json,instanceCount:0};}\n`);
  fs.writeFileSync(path.join(kit,'animation_runtime.mjs'),
    `export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code;}}\n`+
    `export function createAnimationPlayer(d){return {nodeCount:d.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}};}\n`);
  // A narrow fixture for the production renderer's optional import boundary.
  // The imported packer and camera adapter below are the production modules.
  fs.writeFileSync(path.join(kit,'animation_render.mjs'),
    `export async function createGpuAnimationRenderer(device,{fog=false}={}){\n`+
    `const api=fog?await import('./animation_fog.mjs'):null;\n`+
    `return {render(frame){return api?api.packAnimationFog(frame.fog):null;}};}\n`);
  for(const name of ['build_animation.mjs','animation_fog.mjs','animation_fog_camera.mjs'])
    fs.writeFileSync(path.join(kit,name),bytes(name));
  const input=path.join(root,'actor.gltf');
  fs.writeFileSync(input,JSON.stringify({asset:{version:'2.0'},nodes:[{}]}));
  const {buildAnimation}=await import(pathToFileURL(path.join(kit,'build_animation.mjs')));
  let counter=0;
  const build=(options={})=>buildAnimation(input,path.join(root,'out-'+counter++),options);
  return {root,kit,input,build,buildAnimation};
}
function checkArtifacts(result) {
  for(const artifact of result.artifacts) {
    const data=fs.readFileSync(path.join(result.outDir,artifact.file));
    assert.equal(data.length,artifact.bytes);
    assert.equal(hash(data),artifact.sha256);
  }
  assert.equal(result.outputBytes,fs.readdirSync(result.outDir).reduce((n,f)=>n+fs.statSync(path.join(result.outDir,f)).size,0));
  assert.equal(new Set(result.emittedFiles).size,result.emittedFiles.length);
}

test('relocated fog-enabled package resolves the lazy import and executes the real camera adapter/packer',async t=>{
  const f=await fixture(t), result=f.build({webgpu:true,fog:true}); checkArtifacts(result);
  assert.ok(result.gpuFog.includes('native-view-depth'));
  const moved=path.join(f.root,'deployment'); fs.renameSync(result.outDir,moved);
  // Remove access without deleting fixtures: a relocated package cannot borrow
  // the builder directory or original asset through relative/absolute imports.
  fs.renameSync(f.kit,path.join(f.root,'retired-kit')); fs.renameSync(f.input,path.join(f.root,'retired-input'));
  const entry=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')));
  const renderer=await entry.createGpuAnimationRenderer(null,{fog:true});
  const sourceFog={type:'linear',color:[.1,.2,.3],near:2,far:10};
  const first=renderer.render({fog:entry.snapshotAnimationCameraFog(sourceFog,I)});
  sourceFog.near=3; sourceFog.color[0]=.5;
  const second=renderer.render({fog:entry.snapshotAnimationCameraFog(sourceFog,I)});
  assert.equal(first[8],2); assert.equal(second[8],3); assert.equal(second[4],.5);
  assert.deepEqual([...first.slice(0,4)],[0,0,-1,0]);
  const exp=renderer.render({fog:entry.snapshotAnimationCameraFog({type:'exp2',color:[1,2,3],density:.25},I)});
  assert.equal(exp[10],.25); assert.equal(exp[11],2);
  assert.deepEqual([...renderer.render({fog:null})],Array(12).fill(0));
});

test('ordinary GPU package remains isolated; its unbundled optional fog import reproduces the original failure',async t=>{
  const f=await fixture(t), r=f.build({webgpu:true});
  const entry=await import(pathToFileURL(path.join(r.outDir,'gpu_playback.mjs')));
  assert.ok(await entry.createGpuAnimationRenderer(null));
  await assert.rejects(entry.createGpuAnimationRenderer(null,{fog:true}),{code:'ERR_MODULE_NOT_FOUND'});
  assert.equal(entry.snapshotAnimationCameraFog,undefined);
  assert.ok(!r.emittedFiles.includes('animation_fog.mjs'));
});

test('disabled fog leaves CPU/GPU output bytes and all existing entries unchanged',async t=>{
  const f=await fixture(t);
  for(const webgpu of [false,true]) {
    const a=f.build({webgpu}), b=f.build({webgpu,fog:false});
    assert.deepEqual(a.emittedFiles,b.emittedFiles);
    for(const name of a.emittedFiles) assert.deepEqual(fs.readFileSync(path.join(a.outDir,name)),fs.readFileSync(path.join(b.outDir,name)));
    assert.ok(!a.emittedFiles.some(name=>name.includes('fog')));
  }
  const off=f.build({webgpu:true}), on=f.build({webgpu:true,fog:true});
  for(const name of ['animation.mjs','playback.mjs','animation.json'])
    assert.deepEqual(fs.readFileSync(path.join(off.outDir,name)),fs.readFileSync(path.join(on.outDir,name)));
});

test('invalid fog configuration is rejected before an output directory is created',async t=>{
  const f=await fixture(t);
  for(const options of [{fog:true},{webgpu:true,fog:1},{webgpu:true,fog:null},{webgpu:true,fog:'true'}]) {
    const out=path.join(f.root,'invalid');
    assert.throws(()=>f.buildAnimation(f.input,out,options),/fog must be boolean and requires webgpu:true/);
    assert.equal(fs.existsSync(out),false);
  }
});

test('every emitted fog byte and manifest byte participates in exact/one-byte-short output budgets',async t=>{
  const f=await fixture(t), initial=f.build({webgpu:true,fog:true});
  const exact=f.build({webgpu:true,fog:true,maxBytes:initial.outputBytes}); checkArtifacts(exact);
  assert.equal(exact.outputBytes,initial.outputBytes);
  const short=path.join(f.root,'short');
  assert.throws(()=>f.buildAnimation(f.input,short,{webgpu:true,fog:true,maxBytes:initial.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  assert.equal(fs.existsSync(short),false);
  for(const name of ['animation_fog.mjs','animation_fog_camera.mjs'])
    assert.deepEqual(fs.readFileSync(path.join(initial.outDir,name)),bytes(name));
});

test('fog composes with source scene, environment, background, recovery, rigid geometry and IK packaging',async t=>{
  const f=await fixture(t);
  for(const options of [
    {threeScene:true}, {environment:true,hdr:true}, {background:true}, {canvasRecovery:true},
    {threeScene:true,environment:true,background:true,canvasRecovery:true,inverseKinematics:true,rigidGeometry:true},
  ]) {
    const result=f.build({webgpu:true,fog:true,...options}); checkArtifacts(result);
    const api=await import(pathToFileURL(path.join(result.outDir,'gpu_playback.mjs')));
    assert.deepEqual(api.animationFogDepthFromProjection(I),[0,0,-1,0]);
  }
});
