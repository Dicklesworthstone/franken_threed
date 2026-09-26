import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

// Real builder, emitted pose runtime and all IK math/editor/palette operations.
// Asset decoding and unexercised renderer/controller factories are explicit
// boundaries. Throwing stubs ensure their accidental use cannot report success.
const production=['animation_runtime.mjs','animation_ik.mjs'];
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const near=(a,b,eps=1e-6)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<=eps,`${a} != ${b}`));
const point=(p,n)=>Array.from(p.worldMatrices.subarray(n*16+12,n*16+15));
function definition() {
  return {format:'f3d-animation-v1',nodes:[{weights:[0.3]},
    {parent:0,translation:[-1,0,0]},{parent:1,translation:[0,-1,0]},{parent:2,translation:[0,-1,0]},
    {parent:0,translation:[1,0,0]},{parent:4,translation:[0,-1,0]},{parent:5,translation:[0,-1,0]},
    {}, {translation:[5,0,0]}],skins:[{joints:[1,2,3,4,5,6]}],instances:[{node:7,skin:0},{node:8,skin:0}],
    clips:[{name:'retained-time',channels:[{node:0,path:'weights',times:[0,2],values:[0.3,0.7]}]}],ignoredChannels:[]};
}
const requests=()=>[
  {root:1,joint:2,effector:3,target:[-1,-1.5,0],pole:[-1,0,1],endRotation:[0,0,0,1]},
  {root:4,joint:5,effector:6,target:[1,-1.2,0],pole:[1,0,1],endRotation:[0,0,0,1]},
];
async function toolkit(t, instances=false) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-limb-package-'));
  // Keep test artifacts inspectable; never remove caller/source directories.
  t.diagnostic('Generated package fixture: '+root);
  const tool=path.join(root,'toolkit');fs.mkdirSync(tool);
  const source=fs.readFileSync(new URL('./build_animation.mjs',import.meta.url),'utf8'),names=new Map();
  for(const match of source.matchAll(/["'](?:\.\/)?([a-z][\w.]*\.mjs)["']/g))names.set(match[1],new Set());
  for(const match of source.matchAll(/export \{([^{}]+)\} from '\.\/([^']+)'/g)){
    if(!names.has(match[2]))names.set(match[2],new Set());
    for(const name of match[1].split(','))if(/^[A-Za-z]\w*$/.test(name.trim()))names.get(match[2]).add(name.trim());
  }
  for(const [file,exports] of names)fs.writeFileSync(path.join(tool,file),[...exports].map(name=>
    `export function ${name}(){throw new Error('Unexercised package boundary: ${file}:${name}');}`).join('\n')+'\n');
  fs.writeFileSync(path.join(tool,'animation_gltf.mjs'),
    'export function decodeGltfAnimation(model){return model.fixtureDefinition;}\n');
  fs.writeFileSync(path.join(tool,'gltf_instancing.mjs'),
    'export function expandGltfInstances(json){return {json,instanceCount:json.fixtureInstances?1:0,instanceOrigins:{8:{node:7,instance:0}}};}\n');
  for(const file of production)fs.copyFileSync(new URL('./'+file,import.meta.url),path.join(tool,file));
  fs.writeFileSync(path.join(tool,'build_animation.mjs'),source);
  const entry=path.join(root,'actor.gltf');fs.writeFileSync(entry,JSON.stringify({asset:{version:'2.0'},fixtureDefinition:definition(),fixtureInstances:instances}));
  const {buildAnimation}=await import(pathToFileURL(path.join(tool,'build_animation.mjs')));
  return {root,tool,entry,buildAnimation};
}
function verify(result,out) {
  assert.equal(new Set(result.emittedFiles).size,result.emittedFiles.length);
  assert.equal(result.outputBytes,result.emittedFiles.reduce((n,file)=>n+fs.statSync(path.join(out,file)).size,0));
  for(const artifact of result.artifacts){const bytes=fs.readFileSync(path.join(out,artifact.file));assert.equal(bytes.length,artifact.bytes);assert.equal(hash(bytes),artifact.sha256);}
  for(const file of production)if(result.emittedFiles.includes(file))
    assert.deepEqual(fs.readFileSync(path.join(out,file)),fs.readFileSync(new URL('./'+file,import.meta.url)));
}
for(const webgpu of [false,true])test(`relocated ${webgpu?'GPU':'CPU'} entry solves two animated feet and updates both skin palettes`,async t=>{
  const kit=await toolkit(t),out=path.join(kit.root,'built');
  const result=kit.buildAnimation(kit.entry,out,{webgpu,inverseKinematics:true});verify(result,out);
  assert.match(result.animationIK,/analytic-pole-limbs/);assert.equal(result.accelerationClaim,false);
  const relocated=path.join(kit.root,'deployment');fs.renameSync(out,relocated);fs.renameSync(kit.tool,kit.tool+'-unavailable');fs.renameSync(kit.entry,kit.entry+'-unavailable');
  const api=await import(pathToFileURL(path.join(relocated,webgpu?'gpu_playback.mjs':'playback.mjs')));
  for(const name of ['solveAnimationIK','solveAnimationTwoBoneIK','solveAnimationLimbIK'])assert.equal(typeof api[name],'function');
  const pose=api.createPlayer();pose.sample(0.75);const time=pose.time,clip=pose.clip,weights=pose.morphWeights.slice(),before=pose.version,palette=pose.jointMatrices;
  const r=api.solveAnimationLimbIK(pose,requests());assert.ok(r.converged);assert.equal(pose.version,before+1);assert.equal(pose.time,time);assert.equal(pose.clip,clip);
  assert.equal(pose.jointMatrices,palette);assert.deepEqual(pose.morphWeights,weights);
  near(point(pose,3),[-1,-1.5,0]);near(point(pose,6),[1,-1.2,0]);
  near(Array.from(palette.subarray(44,47)),[-1,-1.5,0]);near(Array.from(palette.subarray(188,191)),[-4,-1.2,0]);
  pose.sample(0.75);near(point(pose,3),[-1,-2,0]);near(point(pose,6),[1,-2,0]);
  assert.ok(api.solveAnimationTwoBoneIK(pose,requests()[0]).converged);
  const other=api.createPlayer();near(point(other,3),[-1,-2,0]);assert.equal(other.version,0);
  assert.ok(api.solveAnimationIK(other,{effector:3,target:[-0.5,-1,0],links:[{node:2},{node:1}],iterations:64}).converged);
});

test('disabled IK adds no modules, exports or bytes, and the sampling entry never gains IK imports',async t=>{
  const kit=await toolkit(t);
  for(const webgpu of [false,true]){
    const a=path.join(kit.root,'default-'+webgpu),b=path.join(kit.root,'disabled-'+webgpu);
    const left=kit.buildAnimation(kit.entry,a,{webgpu}),right=kit.buildAnimation(kit.entry,b,{webgpu,inverseKinematics:false});
    assert.deepEqual(left.emittedFiles,right.emittedFiles);assert.equal(left.outputBytes,right.outputBytes);assert.equal(left.animationIK,undefined);
    assert.ok(!left.emittedFiles.includes('animation_ik.mjs'));
    for(const file of left.emittedFiles)assert.deepEqual(fs.readFileSync(path.join(a,file)),fs.readFileSync(path.join(b,file)));
    const api=await import(pathToFileURL(path.join(a,webgpu?'gpu_playback.mjs':'playback.mjs')));assert.equal(api.solveAnimationLimbIK,undefined);
    const enabled=path.join(kit.root,'enabled-'+webgpu),on=kit.buildAnimation(kit.entry,enabled,{webgpu,inverseKinematics:true});
    assert.deepEqual(fs.readFileSync(path.join(a,'animation.mjs')),fs.readFileSync(path.join(enabled,'animation.mjs')));
    const added=on.emittedFiles.filter(file=>!left.emittedFiles.includes(file));assert.deepEqual(added,['animation_ik.mjs']);
  }
});

test('IK packaging rejects bad switches before input access and enforces exact emitted-byte budgets',async t=>{
  const kit=await toolkit(t),missing=path.join(kit.root,'missing.gltf');
  for(const inverseKinematics of [null,0,1,'true',{},[]])
    assert.throws(()=>kit.buildAnimation(missing,path.join(kit.root,'never'),{inverseKinematics}),/inverseKinematics must be boolean/);
  assert.equal(fs.existsSync(path.join(kit.root,'never')),false);
  const out=path.join(kit.root,'reference'),result=kit.buildAnimation(kit.entry,out,{inverseKinematics:true});verify(result,out);
  const refused=path.join(kit.root,'too-small');
  assert.throws(()=>kit.buildAnimation(kit.entry,refused,{inverseKinematics:true,maxBytes:result.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  assert.equal(fs.existsSync(refused),false);
  const exact=kit.buildAnimation(kit.entry,path.join(kit.root,'exact'),{inverseKinematics:true,maxBytes:result.outputBytes});assert.equal(exact.outputBytes,result.outputBytes);
});

test('IK and expanded-instance provenance coexist in both playback entries',async t=>{
  const kit=await toolkit(t,true),out=path.join(kit.root,'instances');
  const result=kit.buildAnimation(kit.entry,out,{webgpu:true,inverseKinematics:true});verify(result,out);assert.equal(result.meshInstanceCount,1);
  for(const entry of ['playback.mjs','gpu_playback.mjs']){
    const api=await import(pathToFileURL(path.join(out,entry)));assert.deepEqual(api.instanceOrigins[8],{node:7,instance:0});
    assert.ok(Object.isFrozen(api.instanceOrigins[8]));assert.ok(api.solveAnimationLimbIK(api.createPlayer(),requests()).converged);
  }
});

test('combined source, HDR, environment and recovery packages include IK once without changing ownership flags',async t=>{
  const kit=await toolkit(t),out=path.join(kit.root,'combined');
  const result=kit.buildAnimation(kit.entry,out,{webgpu:true,threeScene:true,canvasRecovery:true,environment:true,background:true,hdr:true,rigidGeometry:true,inverseKinematics:true});
  verify(result,out);assert.equal(result.artifacts.filter(a=>a.file==='animation_ik.mjs').length,1);
  assert.ok(result.gpuCanvasRecovery);assert.ok(result.gpuEnvironment);assert.ok(result.gpuThreeScene);assert.ok(result.gpuBackground);
  const api=await import(pathToFileURL(path.join(out,'gpu_playback.mjs')));assert.ok(api.solveAnimationLimbIK(api.createPlayer(),requests()).converged);
  assert.equal(typeof api.createRecoverableGpuThreeHdrCanvas,'function');
});
