import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {solveAnimationIK} from './animation_ik.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const T=(x,y=0,z=0)=>{const m=I();m[12]=x;m[13]=y;m[14]=z;return m;};
const Q=(angle,axis=[0,0,1])=>[...axis.map(v=>v*Math.sin(angle/2)),Math.cos(angle/2)];
const position=(p,n)=>[...p.worldMatrices.subarray(n*16+12,n*16+15)];
const near=(a,b,eps=1e-5)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<=eps,`${a} != ${b}`);};
const state=p=>({version:p.version,locals:p.snapshotLocalPose(),world:[...p.worldMatrices],palettes:[...p.jointMatrices]});
function rig(extra={}) {
  return createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},{parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]},
    {translation:[3,0,0],weights:[.2]},{translation:[-2,0,0],weights:[.4]}],
    skins:[{joints:[0,1,2],inverseBindMatrices:[...I(),...T(-1),...T(-2)]}],instances:[{node:3,skin:0},{node:4,skin:0}],...extra});
}
const options=(extra={})=>({effector:2,target:[1,1,0],links:[{node:1},{node:0}],...extra});
test('two-bone solve publishes once, updates descendants and both independent skin palettes',()=>{
  const p=rig(),arrays=[p.worldMatrices,p.jointMatrices,p.rotations],before=p.version;
  const r=solveAnimationIK(p,options());assert.equal(r.converged,true);assert.equal(r.iterations,1);assert.equal(p.version,before+1);
  assert.equal(r.poseVersion,p.version);assert.ok(Object.isFrozen(r));assert.ok(Object.isFrozen(r.changedNodes));
  near(position(p,2),[1,1,0]);near(p.morphWeights,[.2,.4]);
  assert.equal(p.worldMatrices,arrays[0]);assert.equal(p.jointMatrices,arrays[1]);assert.equal(p.rotations,arrays[2]);
  // End-bone inverse-bind origin is (2,0,0); q=90deg yields palette translation.
  near(p.jointMatrices.subarray(44,47),[-2,-1,0]);near(p.jointMatrices.subarray(92,95),[3,-1,0]);
});
for(const target of [[.8,1.1,.4],[-.6,.7,-.3],[.5,-1.5,.2]])test(`multiple CCD iterations reach a 3D goal ${target}`,()=>{
  const p=rig(),r=solveAnimationIK(p,options({target,iterations:64,tolerance:1e-7}));
  assert.equal(r.converged,true);assert.ok(r.distance<1e-7);assert.ok(r.distance<r.initialDistance);near(position(p,2),target,1e-7);
});
test('root translation/rotation/uniform scaling defines world target coordinates',()=>{
  const p=rig(),root=[0,2,0,0,-2,0,0,0,0,0,2,0,10,20,30,1];p.sample(0,{clip:-1,rootMatrix:root});
  const r=solveAnimationIK(p,options({target:[8,22,30]}));assert.equal(r.converged,true);near(position(p,2),[8,22,30]);
  assert.deepEqual([...p.snapshotLocalPose().rootMatrix],root);
});
test('matrix-authored ancestors are preserved, and non-joint intermediate nodes may be skipped',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{parent:3,translation:[1,0,0]},{matrix:T(7)},{parent:1},{parent:2,translation:[1,0,0]}]});
  const r=solveAnimationIK(p,{effector:0,target:[8,1,0],links:[{node:3},{node:2}]});
  assert.equal(r.converged,true);near(position(p,0),[8,1,0]);assert.equal(p.snapshotLocalPose().matrices[0].matrix[12],7);
});
test('antiparallel direction has a finite deterministic axis',()=>{
  const p=rig(),r=solveAnimationIK(p,options({effector:1,target:[-1,0,0],links:[{node:0}],maxAngle:Math.PI}));
  assert.equal(r.converged,true);near(position(p,1),[-1,0,0]);assert.ok(p.rotations.every(Number.isFinite));
  const q=rig();solveAnimationIK(q,options({effector:1,target:[-1,0,0],links:[{node:0}],maxAngle:Math.PI}));assert.deepEqual(p.rotations,q.rotations);
});
test('unreachable goals return the achieved distance without stretching bones or claiming convergence',()=>{
  const p=rig(),r=solveAnimationIK(p,options({target:[0,8,0],iterations:64,tolerance:1e-6}));
  assert.equal(r.converged,false);assert.ok(r.distance>=6-1e-8);assert.ok(r.distance<r.initialDistance);
  const a=position(p,0),b=position(p,1),c=position(p,2);near([Math.hypot(...b.map((v,i)=>v-a[i])),Math.hypot(...c.map((v,i)=>v-b[i]))],[1,1]);
  assert.ok(r.iterations<=64);
});
test('already reached and zero-weight solves do not advance pose version',()=>{
  const p=rig(),before=state(p);
  const reached=solveAnimationIK(p,options({target:[2,0,0]}));assert.equal(reached.converged,true);assert.equal(reached.iterations,0);
  const zero=solveAnimationIK(p,options({weight:0}));assert.equal(zero.converged,false);assert.equal(zero.iterations,0);
  assert.deepEqual(reached.changedNodes,[]);assert.deepEqual(zero.changedNodes,[]);assert.deepEqual(state(p),before);
});
test('maxAngle bounds individual steps and partial weight reports the blended result',()=>{
  const p=rig(),r=solveAnimationIK(p,options({effector:1,target:[0,1,0],links:[{node:0}],iterations:1,maxAngle:.2}));
  near(p.rotations.subarray(0,4),Q(.2));assert.equal(r.converged,false);
  const q=rig(),blend=solveAnimationIK(q,options({weight:.5}));near(position(q,2),[1+Math.SQRT1_2,Math.SQRT1_2,0]);
  assert.equal(blend.converged,false);assert.ok(Math.abs(blend.distance-Math.hypot(Math.SQRT1_2,Math.SQRT1_2-1))<1e-12);
});
test('disabled link remains fixed while enabled ancestors may move',()=>{
  const p=rig(),r=solveAnimationIK(p,options({target:[0,2,0],links:[{node:1,enabled:false},{node:0}]}));
  assert.equal(r.converged,true);near(p.rotations.subarray(4,8),[0,0,0,1]);near(position(p,2),[0,2,0]);
});
test('rest-relative hinge limits constrain motion without Euler wraparound',()=>{
  const p=rig(),r=solveAnimationIK(p,options({effector:1,target:[0,1,0],links:[{node:0,hinge:{axis:[0,0,5],min:-.2,max:.3}}]}));
  assert.equal(r.converged,false);near(p.rotations.subarray(0,4),Q(.3));
  const v=p.version;solveAnimationIK(p,options({effector:1,target:[0,1,0],links:[{node:0,hinge:{axis:[0,0,1],min:-.2,max:.3}}]}));
  assert.equal(p.version,v); // Already at the same persistent rest-relative limit.
});
test('rotated rest poses and explicit reference rotations define hinge frames',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{rotation:Q(Math.PI/2)},{parent:0,translation:[1,0,0]}]});
  const r=solveAnimationIK(p,{effector:1,target:[-1,0,0],links:[{node:0,hinge:{axis:[0,0,1],min:0,max:Math.PI/2}}]});
  assert.equal(r.converged,true);near(position(p,1),[-1,0,0]);
  const q=rig();q.edit([{node:0,rotation:Q(.4)}]);
  solveAnimationIK(q,options({effector:1,target:[0,1,0],links:[{node:0,hinge:{axis:[0,0,1],referenceRotation:Q(.4),min:0,max:.2}}]}));
  near(q.rotations.subarray(0,4),Q(.6));
});
test('hinge blending stays inside a non-wrapping interval instead of taking the forbidden short arc',()=>{
  const p=rig();p.edit([{node:0,rotation:Q(-2.9)}]);
  const r=solveAnimationIK(p,options({effector:1,target:[Math.cos(2.9),Math.sin(2.9),0],links:[{node:0,hinge:{axis:[0,0,1],min:-3,max:3}}],weight:.5}));
  const angle=2*Math.atan2(p.rotations[2],p.rotations[3]);assert.ok(angle>=-3&&angle<=3);assert.equal(r.converged,false);
});
test('zero-length bones and a target on a joint produce bounded finite results',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},{parent:0},{parent:1}]});
  const r=solveAnimationIK(p,options());assert.equal(r.converged,false);assert.equal(r.iterations,1);assert.equal(p.version,0);
  const q=rig(),s=solveAnimationIK(q,options({effector:1,target:[0,0,0],links:[{node:0}]}));assert.equal(s.converged,false);near(position(q,1),[1,0,0]);
});
const invalid=[null,{},options({iterations:0}),options({iterations:65}),options({iterations:1.2}),options({tolerance:-1}),
  options({maxAngle:0}),options({maxAngle:4}),options({weight:-1}),options({weight:1.1}),options({target:[NaN,0,0]}),
  options({target:[0,0]}),options({effector:9}),options({effector:1.5}),options({links:[]}),options({links:Array(33).fill({node:0})}),
  options({links:[{node:0},{node:1}]}),options({links:[{node:1},{node:1}]}),options({links:[{node:3}]}),options({links:[{node:2}]}),
  options({links:[{node:0,enabled:1}]}),options({links:[{node:0,extra:1}]}),options({links:[{node:0,hinge:{axis:[0,0,0]}}]}),
  options({links:[{node:0,hinge:{axis:[0,0,1],min:1,max:2}}]}),options({links:[{node:0,hinge:{axis:[0,0,1],min:-4}}]}),
  options({links:[{node:0,hinge:{axis:[0,0,1],referenceRotation:[0,0,0,0]}}]}),options({unknown:1}),
];
for(let i=0;i<invalid.length;i++)test(`invalid solve ${i} is atomic`,()=>{
  const p=rig(),before=state(p);assert.throws(()=>solveAnimationIK(p,invalid[i]));assert.deepEqual(state(p),before);
});
for(const scale of [[2,1,1],[-1,-1,1],[0,0,0]])test(`inadmissible active link scale ${scale} is refused`,()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{scale},{parent:0,translation:[1,0,0]}]}),before=state(p);
  assert.throws(()=>solveAnimationIK(p,{effector:1,target:[0,1,0],links:[{node:0}]}),{code:'ANIMATION_IK_TRANSFORM'});assert.deepEqual(state(p),before);
});
test('sheared or reflected world frames refuse without partial edits',()=>{
  for(const root of [[1,0,0,0,.5,1,0,0,0,0,1,0,0,0,0,1],[-1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]]) {
    const p=rig();p.sample(0,{clip:-1,rootMatrix:root});const before=state(p);
    assert.throws(()=>solveAnimationIK(p,options()),{code:'ANIMATION_IK_TRANSFORM'});assert.deepEqual(state(p),before);
  }
});
test('matrix-authored active link refuses rather than losing its transform representation',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{matrix:I()},{parent:0,translation:[1,0,0]}]});
  assert.throws(()=>solveAnimationIK(p,{effector:1,target:[0,1,0],links:[{node:0}]}),{code:'ANIMATION_IK_TRANSFORM'});assert.equal(p.version,0);
});
test('hinge rejects entry swing and out-of-range twist without modifying the pose',()=>{
  const p=rig();p.edit([{node:0,rotation:Q(.2,[1,0,0])}]);const before=state(p);
  assert.throws(()=>solveAnimationIK(p,options({links:[{node:0,hinge:{axis:[0,0,1]}}]})),{code:'ANIMATION_IK_HINGE'});assert.deepEqual(state(p),before);
});
test('private committed pose is used even when public outputs were modified',()=>{
  const p=rig();p.worldMatrices.fill(99);p.rotations.fill(33);p.translations.fill(88);
  const r=solveAnimationIK(p,options());assert.equal(r.converged,true);near(position(p,2),[1,1,0]);
});
test('edits after animation blending preserve non-IK bindings and current sample provenance',()=>{
  const p=rig({clips:[{channels:[{node:0,path:'translation',times:[0,1],values:[0,0,0,2,0,0]},
    {node:3,path:'weights',times:[0,1],values:[.2,.8]}]}]});
  p.sample(.5);solveAnimationIK(p,options({target:[2,1,0]}));near(position(p,2),[2,1,0]);near(p.morphWeights,[.5,.4]);assert.equal(p.time,.5);assert.equal(p.clip,0);
  p.sample(.5);near(position(p,2),[3,0,0]); // Procedural rotations do not leak into the next sample.
});
test('changed input getters invalidate the snapshot rather than overwriting a newer pose',()=>{
  const p=rig();const o=options();Object.defineProperty(o,'target',{get(){p.edit([{node:0,translation:[8,0,0]}]);return [1,1,0];}});
  assert.throws(()=>solveAnimationIK(p,o),{code:'ANIMATION_IK_STALE'});assert.equal(p.version,1);near(position(p,2),[10,0,0]);
});
test('a final f32 palette overflow does not publish intermediate solver rotations',()=>{
  const X=1e39,p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{translation:[X,0,0]},{parent:0,translation:[X,0,0]},{}],
    skins:[{joints:[0],inverseBindMatrices:T(-X)}],instances:[{node:2,skin:0}]}),before=state(p);
  assert.throws(()=>solveAnimationIK(p,{effector:1,target:[X,X,0],links:[{node:0}]}),{code:'ANIMATION_VALUE'});assert.deepEqual(state(p),before);
});
test('excessive depth, detached output and disposed pose fail before solving',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:Array.from({length:257},(_,i)=>({parent:i?i-1:-1,translation:[1,0,0]}))});
  assert.throws(()=>solveAnimationIK(p,{effector:256,target:[0,0,0],links:[{node:0}]}),{code:'ANIMATION_IK_CHAIN'});
  const q=rig();structuredClone(q.jointMatrices.buffer,{transfer:[q.jointMatrices.buffer]});assert.throws(()=>solveAnimationIK(q,options()),{code:'ANIMATION_OUTPUT_STORAGE'});
  const r=rig();r.dispose();assert.throws(()=>solveAnimationIK(r,options()),{code:'ANIMATION_DISPOSED'});
});
test('no import-time renderer, clock or WebAssembly dependencies are required',async()=>{
  const {readFileSync}=await import('node:fs');const source=readFileSync(new URL('./animation_ik.mjs',import.meta.url),'utf8');
  assert.doesNotMatch(source,/\b(?:fetch|requestAnimationFrame|setInterval|setTimeout|WebAssembly)\s*[.(]/);
});

test('IK and live morph editing flow through the unchanged CPU deformer, bounds and normal streams',async()=>{
  const {createAnimationDeformer}=await import('./animation_deformer.mjs');
  const p=rig(),geometry={positions:[2,0,0],normals:[1,0,0],tangents:[1,0,0,1],
    morphTargets:[{positions:[0,0,1]}],joints:[2],weights:[1],influences:1};
  const a=createAnimationDeformer(p,{...geometry,node:3}),b=createAnimationDeformer(p,{...geometry,node:4});
  const arrays=[a.positions,a.normals,a.tangents,a.worldMatrix],aVersion=a.version,bVersion=b.version;
  p.edit([{node:3,weights:[.75]}]);const solved=solveAnimationIK(p,options());
  assert.equal(solved.converged,true);assert.equal(a.version,aVersion);assert.equal(b.version,bVersion);
  assert.notEqual(a.poseVersion,p.version);a.update();b.update();
  near(a.positions,[-2,1,.75]);near(b.positions,[3,1,.4]);
  near(a.normals,[0,1,0]);near(a.tangents,[0,1,0,1]);near(a.bounds.min,a.positions);near(a.bounds.max,a.positions);
  const world=d=>[0,1,2].map(i=>d.worldMatrix[i]*d.positions[0]+d.worldMatrix[i+4]*d.positions[1]+
    d.worldMatrix[i+8]*d.positions[2]+d.worldMatrix[i+12]);
  near(world(a),[1,1,.75]);near(world(b),[1,1,.4]);
  [a.positions,a.normals,a.tangents,a.worldMatrix].forEach((value,i)=>assert.equal(value,arrays[i]));
  assert.equal(a.poseVersion,p.version);assert.equal(b.poseVersion,p.version);
  p.reset();a.update();b.update();near(world(a),[2,0,.2]);near(world(b),[2,0,.4]);
  a.dispose();b.dispose();assert.equal(p.disposed,false);p.dispose();
});
