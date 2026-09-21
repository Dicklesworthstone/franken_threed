/** Real model picker, Float32 morph/skin deformation and two-level raycaster.
 * Only the packed pose/camera fixtures and runtime error-class dependency are
 * doubles. This does not execute the pose sampler, glTF decoder, GPU or browser. */
import test,{after} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,copyFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';

const dir=await mkdtemp(join(tmpdir(),'f3d-pick-lazy-'));
after(()=>rm(dir,{recursive:true,force:true}));
for(const name of ['animation_model_pick','animation_raycast','animation_deformer'])
  await copyFile(new URL(`./${name}.mjs`,import.meta.url),join(dir,`${name}.mjs`));
await writeFile(join(dir,'animation_runtime.mjs'),
  `export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code;}}\n`);
const {createAnimationModelPicker}=await import(pathToFileURL(join(dir,'animation_model_pick.mjs')).href);
const {createAnimationDeformer}=await import(pathToFileURL(join(dir,'animation_deformer.mjs')).href);
const identity=()=>new Float64Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
const ray=(x=0,z=5)=>({origin:[x,0,z],direction:[0,0,-1]});
function fixture(t,count=32,{skin=false,morph=false,picking=true}={}) {
  const pose={nodeCount:count,version:0,disposed:false,instances:[],sample(){throw Error('picking advanced the pose');},
    worldMatrices:new Float64Array(count*16),jointMatrices:new Float64Array(skin?count*16:0),
    morphWeights:new Float64Array(morph?count:0).fill(0.5),morphOffsets:new Uint32Array(count+1)};
  for(let i=0;i<count;i++) {
    pose.worldMatrices.set(identity(),i*16);pose.worldMatrices[i*16+12]=i*4;
    if(morph)pose.morphOffsets[i+1]=i+1;
    if(skin){pose.instances.push({node:i,offset:i*16,jointCount:1});pose.jointMatrices.set(identity(),i*16);pose.jointMatrices[i*16+14]=1;}
  }
  const drawables=Array.from({length:count},(_,node)=>({geometry:{node,
    positions:new Float32Array([-1,-1,0,1,-1,0,0,1,0]),
    ...(morph?{morphTargets:[{positions:new Float32Array([0,0,2,0,0,2,0,0,2])}]}:{}),
    ...(skin?{influences:1,joints:new Uint8Array([0,0,0]),weights:new Float32Array([1,1,1])}:{})},
    indices:new Uint16Array([0,1,2]),texCoords:new Float32Array([0,0,1,0,0.5,1])}));
  const source=drawables.map((_,node)=>({node,mesh:node+100,primitive:7,material:3}));
  const view={sample({x=0,z=5,type='perspective'}={}){
    const v=identity();v[12]=-x;v[14]=-z;
    const p=type==='orthographic'?identity():new Float64Array([1,0,0,0,0,1,0,0,0,0,-1,-1,0,0,-1,0]);
    return {poseVersion:pose.version,type,viewMatrix:v,projectionMatrix:p,cameraPosition:[x,0,z]};
  }};
  const picker=createAnimationModelPicker(pose,view,drawables,source,picking);t.after(()=>picker.dispose());
  return {pose,drawables,source,view,picker};
}

test('one selected draw materializes one CPU mesh out of 256 and preserves original IDs',t=>{
  const {picker}=fixture(t,256);
  const hit=picker.raycast(ray(173*4),{drawIndices:[173]})[0];
  assert.equal(hit.drawIndex,173);assert.equal(hit.source.mesh,273);assert.equal(hit.source.primitive,7);
  assert.deepEqual(hit.uv,[0.5,0.5]);assert.equal(picker.lastQuery.materializedMeshes,1);
  assert.equal(picker.lastQuery.updatedMeshes,1);assert.equal(picker.lastQuery.refittedMeshes,1);
  picker.raycast(ray(173*4),{drawIndices:[173]});assert.equal(picker.lastQuery.updatedMeshes,0);
  picker.raycast(ray(2*4),{drawIndices:[2]});assert.equal(picker.lastQuery.materializedMeshes,2);
});

test('unselected malformed pose data is not evaluated and inactive cached meshes are not updated',t=>{
  const {picker,pose}=fixture(t);
  picker.raycast(ray(),{drawIndices:[0,1]});assert.equal(picker.lastQuery.materializedMeshes,2);
  pose.version++;pose.worldMatrices[16]=NaN;pose.worldMatrices[14]=1;
  const hit=picker.raycast(ray(),{drawIndices:[0]})[0];assert.equal(hit.distance,4);
  assert.equal(picker.lastQuery.updatedMeshes,1);assert.equal(picker.lastQuery.materializedMeshes,2);
  const previous=picker.lastQuery;
  assert.throws(()=>picker.raycast(ray(4),{drawIndices:[1]}),{code:'ANIMATION_DEFORM_VALUE'});
  assert.equal(picker.lastQuery,previous);
  pose.worldMatrices[16]=1;
  assert.equal(picker.raycast(ray(4),{drawIndices:[1]})[0].drawIndex,1);
  assert.equal(picker.lastQuery.updatedMeshes,1);
});

test('empty selections and rejected query input never materialize deformation',t=>{
  const {picker,pose}=fixture(t);
  pose.worldMatrices[0]=NaN;
  for(const [input,code] of [[{origin:[0,0,5],direction:[0,0,0]},'ANIMATION_PICK_RAY'],
    [{origin:[Infinity,0,0],direction:[0,0,-1]},'ANIMATION_PICK_VALUE']])
    assert.throws(()=>picker.raycast(input),{code});
  for(const settings of [{drawIndices:[99]},{drawIndices:[0,0]},{maxHits:0},{unknown:true}])
    assert.throws(()=>picker.raycast(ray(),settings),{code:'ANIMATION_PICK_OPTION'});
  assert.deepEqual(picker.raycast(ray(),{drawIndices:[]}),[]);
  assert.equal(picker.lastQuery.materializedMeshes,0);assert.equal(picker.lastQuery.updatedMeshes,0);
  assert.equal(picker.raycast(ray(4),{drawIndices:[1]})[0].drawIndex,1);
  assert.equal(picker.lastQuery.materializedMeshes,1);
});

test('morph-then-skin picking agrees with eager real deformation across partial pose updates',t=>{
  const {picker,pose,view,drawables,source}=fixture(t,16,{skin:true,morph:true});
  const deformers=drawables.map(d=>createAnimationDeformer(pose,d.geometry));
  const eager=createAnimationModelPicker(pose,view,drawables,source,true,deformers);
  t.after(()=>{eager.dispose();for(const d of deformers)d.dispose();});
  const compare=(i)=>assert.deepEqual(picker.raycast(ray(i*4),{drawIndices:[i]}),eager.raycast(ray(i*4),{drawIndices:[i]}));
  compare(11);assert.equal(picker.lastQuery.materializedMeshes,1);
  assert.equal(picker.raycast(ray(44),{drawIndices:[11]})[0].distance,3);
  pose.version++;pose.morphWeights[11]=0.25;pose.jointMatrices[11*16+14]=2;
  deformers[11].update();compare(11);
  assert.equal(picker.raycast(ray(44),{drawIndices:[11]})[0].distance,2.5);
  pose.morphWeights[5]=0.75;deformers[5].update();compare(5);
  assert.equal(picker.lastQuery.materializedMeshes,2);assert.equal(pose.version,1);
});

test('source geometry, morphs, joints, indices, UVs and source IDs are snapshotted before lazy creation',t=>{
  const {picker,drawables,source}=fixture(t,12,{skin:true,morph:true});
  const input=drawables[7];input.geometry.positions.fill(99);input.geometry.morphTargets[0].positions.fill(NaN);
  input.geometry.weights.fill(0);input.geometry.joints.fill(99);input.indices.fill(0);input.texCoords.fill(99);source[7].mesh=999;
  const hit=picker.raycast(ray(28),{drawIndices:[7]})[0];
  assert.equal(hit.distance,3);assert.equal(hit.source.mesh,107);assert.deepEqual(hit.uv,[0.5,0.5]);
});

test('querying all draws still prepares the complete scene and uses the new top-level BVH',t=>{
  const {picker}=fixture(t,128);
  assert.equal(picker.raycast(ray())[0].drawIndex,0);
  assert.equal(picker.lastQuery.materializedMeshes,128);assert.equal(picker.lastQuery.updatedMeshes,128);
  assert.ok(picker.lastQuery.sceneBoxesTested>0);assert.ok(picker.lastQuery.meshesTested<=4);
  picker.raycast(ray(4));assert.equal(picker.lastQuery.updatedMeshes,0);assert.equal(picker.lastQuery.sceneRefits,0);
  const linear=fixture(t,128,{picking:{sceneBvh:false}}).picker;linear.raycast(ray());
  assert.equal(linear.lastQuery.sceneBoxesTested,0);assert.equal(linear.lastQuery.meshesTested,128);
});

test('borrowed CPU deformation is never updated or disposed and only selected versions are required',t=>{
  const {pose,view,drawables,source}=fixture(t,2);
  const deformers=drawables.map(d=>createAnimationDeformer(pose,d.geometry));
  t.after(()=>{for(const d of deformers)d.dispose();});
  const picker=createAnimationModelPicker(pose,view,drawables,source,true,deformers);t.after(()=>picker.dispose());
  pose.version++;pose.worldMatrices[14]=1;deformers[0].update();
  assert.equal(picker.raycast(ray(),{drawIndices:[0]})[0].distance,4);
  assert.equal(picker.lastQuery.updatedMeshes,0);assert.equal(picker.lastQuery.materializedMeshes,0);
  assert.throws(()=>picker.raycast(ray(4),{drawIndices:[1]}),{code:'ANIMATION_PICK_STALE'});
  picker.dispose();assert.ok(deformers.every(d=>!d.disposed));assert.equal(pose.disposed,false);
});

test('submitted LOD selections filter geometry without renumbering or advancing hysteresis',t=>{
  const {picker}=fixture(t);
  const lodSelection=Object.freeze({poseVersion:0,cameraKey:'main',drawIndices:Object.freeze([12])});
  assert.equal(picker.raycast(ray(48),{lodSelection})[0].drawIndex,12);
  assert.equal(picker.lastQuery.materializedMeshes,1);
  assert.deepEqual(picker.raycast(ray(),{lodSelection}),[]);
  assert.deepEqual(lodSelection.drawIndices,[12]);
});

test('stale or absent LOD snapshots and conflicting explicit subsets fail before preparation',t=>{
  const {picker,pose}=fixture(t);
  const lodSelection={poseVersion:0,drawIndices:[0]};pose.version++;
  assert.throws(()=>picker.raycast(ray(),{lodSelection}),{code:'ANIMATION_PICK_STALE'});
  for(const options of [{lodSelection:null},{lodSelection:undefined},{lodSelection:{poseVersion:'1',drawIndices:[0]}},
    {lodSelection:{poseVersion:1,drawIndices:[0,0]}},{lodSelection:{poseVersion:1,drawIndices:[0]},drawIndices:[0]}])
    assert.throws(()=>picker.raycast(ray(),options),{code:'ANIMATION_PICK_OPTION'});
  assert.deepEqual(picker.raycast(ray(),{drawIndices:[]}),[]);assert.equal(picker.lastQuery.materializedMeshes,0);
});

test('LOD index arrays are captured before ray getters run',t=>{
  const {picker}=fixture(t);
  const indices=[0],lodSelection={poseVersion:0,drawIndices:indices};
  const input={get origin(){indices[0]=1;return [0,0,5];},direction:[0,0,-1]};
  assert.equal(picker.raycast(input,{lodSelection})[0].drawIndex,0);
  assert.equal(picker.lastQuery.materializedMeshes,1);
});

test('perspective and orthographic camera queries use the same lazy subset route',t=>{
  const {picker}=fixture(t);
  const lodSelection={poseVersion:0,drawIndices:[9]};
  for(const type of ['perspective','orthographic']) {
    const hit=picker.pick([0,0],{x:36,type},{lodSelection})[0];assert.equal(hit.drawIndex,9);
  }
  assert.equal(picker.lastQuery.materializedMeshes,1);assert.equal(picker.lastQuery.updatedMeshes,0);
});

test('failed materialization is retryable and does not replace prior successful query statistics',t=>{
  const {picker,pose}=fixture(t);
  picker.raycast(ray(),{drawIndices:[0]});const previous=picker.lastQuery;
  pose.worldMatrices[16]=NaN;
  assert.throws(()=>picker.raycast(ray(4),{drawIndices:[1]}),{code:'ANIMATION_DEFORM_VALUE'});
  assert.equal(picker.lastQuery,previous);pose.worldMatrices[16]=1;
  assert.equal(picker.raycast(ray(4),{drawIndices:[1]})[0].drawIndex,1);
  assert.equal(picker.lastQuery.materializedMeshes,2);
});

test('option getters cannot reenter, dispose or silently move picking onto a new pose',t=>{
  const {picker,pose}=fixture(t);
  for(const nested of [()=>picker.raycast(ray()),()=>picker.dispose()])
    assert.throws(()=>picker.raycast(ray(),{get lodSelection(){nested();return null;}}),{code:'ANIMATION_PICK_REENTRANT'});
  assert.throws(()=>picker.raycast(ray(),{get firstHitOnly(){pose.version++;return true;}}),{code:'ANIMATION_PICK_CHANGED'});
  picker.raycast(ray(),{drawIndices:[]});assert.equal(picker.lastQuery.materializedMeshes,0);
  picker.dispose();assert.equal(picker.lastQuery,null);assert.equal(pose.disposed,false);
  assert.throws(()=>picker.raycast(ray()),{code:'ANIMATION_PICK_DISPOSED'});
});

test('snapshot component and topology limits remain aggregate; invalid BVH options are rejected',t=>{
  for(const picking of [{maxComponents:1},{maxTriangles:1},{sceneBvh:1},{sceneBvh:null}])
    assert.throws(()=>fixture(t,16,{picking}));
  const disabled=fixture(t,16,{picking:false});
  assert.equal(disabled.picker.enabled,false);assert.throws(()=>disabled.picker.raycast(ray()),{code:'ANIMATION_PICK_DISABLED'});
});
