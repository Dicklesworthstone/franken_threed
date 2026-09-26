import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {solveAnimationTwoBoneIK, solveAnimationLimbIK as solve} from './animation_ik.mjs';
const near=(a,b,eps=1e-9)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<=eps,`${a} != ${b}`));
const point=(p,n)=>Array.from(p.worldMatrices.subarray(16*n+12,16*n+15));
const state=p=>({version:p.version,locals:p.snapshotLocalPose(),world:Array.from(p.worldMatrices),palette:Array.from(p.jointMatrices)});
const code=name=>({code:'ANIMATION_IK_'+name});
function biped(extra={}) {
  return createAnimationPlayer({format:'f3d-animation-v1',nodes:[{weights:[0.3]},
    {parent:0,translation:[-1,0,0]},{parent:1,translation:[0,-1,0]},{parent:2,translation:[0,-1,0]},
    {parent:0,translation:[1,0,0]},{parent:4,translation:[0,-1,0]},{parent:5,translation:[0,-1,0]},
    {}, {translation:[5,0,0]}],skins:[{joints:[1,2,3,4,5,6]}],instances:[{node:7,skin:0},{node:8,skin:0}],...extra});
}
const requests=()=>[
  {root:1,joint:2,effector:3,target:[-1,-1.5,0],pole:[-1,0,1],endRotation:[0,0,0,1]},
  {root:4,joint:5,effector:6,target:[1,-1.2,0],pole:[1,0,1],endRotation:[0,0,0,1]},
];

test('two feet use one real pose snapshot and one publication with both skin palettes',()=>{
  const p=biped(), before=p.version, palettes=p.jointMatrices, world=p.worldMatrices; let snapshots=0, edits=0;
  const borrowed={snapshotLocalPose(){snapshots++;return p.snapshotLocalPose();},edit(changes){edits++;p.edit(changes);},
    get version(){return p.version;},get disposed(){return p.disposed;}};
  const r=solve(borrowed,requests()); assert.equal(snapshots,1);assert.equal(edits,1);assert.equal(p.version,before+1);
  assert.ok(r.converged);assert.equal(r.limbs.length,2);assert.equal(p.jointMatrices,palettes);assert.equal(p.worldMatrices,world);
  near(point(p,3),[-1,-1.5,0]);near(point(p,6),[1,-1.2,0]);
  near(Array.from(palettes.subarray(44,47)),[-1,-1.5,0],1e-6);
  near(Array.from(palettes.subarray(92,95)),[1,-1.2,0],1e-6);
  near(Array.from(palettes.subarray(140,143)),[-6,-1.5,0],1e-6);
  near(Array.from(palettes.subarray(188,191)),[-4,-1.2,0],1e-6);
  assert.ok(Object.isFrozen(r));assert.ok(Object.isFrozen(r.limbs));assert.ok(Object.isFrozen(r.changedNodes));
  for(const limb of r.limbs){assert.equal(limb.poseVersion,r.poseVersion);assert.ok(Object.isFrozen(limb));assert.ok(Object.isFrozen(limb.nodes));}
  assert.deepEqual(Array.from(p.morphWeights),[0.3]);
});

test('independent batch order does not change the final pose',()=>{
  const a=biped(),b=biped(),left=requests(),right=requests().reverse(); solve(a,left);solve(b,right);
  assert.deepEqual(state(a),state(b));
});

test('batch and sequential single-limb solves produce identical pose arrays, not identical version counts',()=>{
  const a=biped(),b=biped(); solve(a,requests());for(const request of requests())solveAnimationTwoBoneIK(b,request);
  for(const field of ['translations','rotations','scales','morphWeights','worldMatrices','jointMatrices'])assert.deepEqual(a[field],b[field]);
  assert.equal(a.version,1);assert.equal(b.version,2);
});

test('bad second limb cannot leave the first leg partially solved',()=>{
  const p=biped(),before=state(p),inputs=requests();inputs[1].pole=[Infinity,0,0];
  assert.throws(()=>solve(p,inputs),code('VALUE'));assert.deepEqual(state(p),before);
  assert.ok(solve(p,requests()).converged);
});

test('batch detects a later getter changing the pose and does not overwrite that external edit',()=>{
  const p=biped(),inputs=requests();let edited;
  Object.defineProperty(inputs[1],'target',{enumerable:true,get(){p.edit([{node:0,weights:[0.9]}]);edited=state(p);return [1,-1.2,0];}});
  assert.throws(()=>solve(p,inputs),code('STALE'));assert.deepEqual(state(p),edited);
});

test('duplicate and overlapping limbs are rejected regardless of zero influence',()=>{
  const p=biped(),before=state(p),first=requests()[0];
  for(const pair of [[first,first],[{...first,weight:0},{...first,weight:0}]]){
    assert.throws(()=>solve(p,pair),code('OVERLAP'));assert.deepEqual(state(p),before);
  }
});

test('disjoint node triples with a cross-limb ancestor dependency are refused in either order',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},...Array.from({length:6},(_,i)=>({parent:i,translation:[1,0,0]}))]});
  const first={root:0,joint:1,effector:2,target:[1,0,0],pole:[0,1,0]},
    second={root:4,joint:5,effector:6,target:[5,1,0],pole:[5,0,1]};
  const before=state(p);
  for(const inputs of [[first,second],[second,first]]){
    assert.throws(()=>solve(p,inputs),code('DEPENDENCY'));assert.deepEqual(state(p),before);
  }
});

test('independent weights and orientation targets retain honest per-limb convergence',()=>{
  const p=biped(),inputs=requests();inputs[1].weight=0.25;inputs[1].orientationWeight=0;
  const r=solve(p,inputs);assert.equal(r.converged,false);assert.ok(r.limbs[0].converged);
  assert.equal(r.limbs[1].positionConverged,false);near(point(p,3),[-1,-1.5,0]);assert.ok(point(p,6)[1]<-1.2);
});

test('unreachable limb is a measured result and does not discard another valid solve',()=>{
  const p=biped(),inputs=requests();inputs[1].target=[1,-5,0];
  const r=solve(p,inputs);assert.ok(r.limbs[0].converged);assert.equal(r.limbs[1].reachable,false);
  assert.equal(r.converged,false);near(point(p,3),[-1,-1.5,0]);near(point(p,6),[1,-2,0]);
});

test('empty/all-zero batches retain version and arrays; empty still requires a live player',()=>{
  const p=biped(),before=state(p),r=solve(p,[]);assert.equal(r.converged,true);assert.deepEqual(r.limbs,[]);assert.deepEqual(state(p),before);
  solve(p,requests().map(x=>({...x,weight:0,orientationWeight:0})));assert.deepEqual(state(p),before);
  p.dispose();assert.throws(()=>solve(p,[]),{code:'ANIMATION_DISPOSED'});
});

test('batch work and ancestry are bounded; custom list iterators are never consumed',()=>{
  const p=biped(),inputs=requests();inputs[Symbol.iterator]=()=>{throw new Error('unbounded iterator');};assert.ok(solve(p,inputs).converged);
  for(const input of [null,{},new Uint8Array(2),Array(33).fill(requests()[0])])assert.throws(()=>solve(p,input),code('LIMIT'));
  const nodes=[{},...Array.from({length:256},(_,i)=>({parent:i,translation:[1,0,0]}))];
  const deep=createAnimationPlayer({format:'f3d-animation-v1',nodes}),before=state(deep);
  assert.throws(()=>solve(deep,[{root:254,joint:255,effector:256,target:[256,1,0],pole:[0,1,0]}]),code('CHAIN'));
  assert.deepEqual(state(deep),before);
});

test('32 independent limbs can publish once at the admitted boundary',()=>{
  const nodes=[{}],inputs=[];
  for(let i=0;i<32;i++){
    const root=nodes.length;nodes.push({parent:0,translation:[0,3*i,0]},
      {parent:root,translation:[1,0,0]},{parent:root+1,translation:[1,0,0]});
    inputs.push({root,joint:root+1,effector:root+2,target:[1,3*i,0],pole:[0,3*i+1,0]});
  }
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes}),r=solve(p,inputs);
  assert.equal(p.version,1);assert.ok(r.converged);assert.equal(r.limbs.length,32);
  for(const input of inputs)near(point(p,input.effector),input.target);
});

test('single and batch entry points share the same reentrancy guard',()=>{
  const p=biped(),inputs=requests();Object.defineProperty(inputs[1],'pole',{enumerable:true,get(){return solveAnimationTwoBoneIK(p,requests()[0]);}});
  assert.throws(()=>solve(p,inputs),code('REENTRANT'));assert.equal(p.version,0);
  assert.throws(()=>solveAnimationTwoBoneIK(p,{...requests()[0],get pole(){return solve(p,requests());}}),code('REENTRANT'));
  assert.ok(solve(p,requests()).converged);
});

test('final f32 skin-palette rejection rolls back every staged limb',()=>{
  const ibm=[1,0,0,0,0,1,0,0,0,0,1,0,0,1e30,0,1];
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},
    {parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]}, {},
    {parent:3,translation:[1,0,0]},{parent:4,translation:[1,0,0]}, {scale:[1e-10,1,1]}],
    skins:[{joints:[3],inverseBindMatrices:ibm}],instances:[{node:6,skin:0}]});
  const before=state(p);assert.throws(()=>solve(p,[
    {root:0,joint:1,effector:2,target:[1,0,0],pole:[0,1,0]},
    {root:3,joint:4,effector:5,target:[1,0,0],pole:[0,1,0]},
  ]),{code:'ANIMATION_VALUE'});assert.deepEqual(state(p),before);
});
