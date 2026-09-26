import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {solveAnimationIK, solveAnimationTwoBoneIK as solve} from './animation_ik.mjs';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b} (+/- ${eps})`);
const vectorClose = (a, b, eps) => { assert.equal(a.length, b.length); a.forEach((x, i) => close(x, b[i], eps)); };
const point = (p, n) => Array.from(p.worldMatrices.subarray(n * 16 + 12, n * 16 + 15));
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const fields = ['translations','rotations','scales','morphWeights','worldMatrices','jointMatrices'];
const state = p => ({version:p.version, time:p.time, clip:p.clip, mode:p.mode,
  ...Object.fromEntries(fields.map(k=>[k,Array.from(p[k])]))});
const q = (axis, angle) => [...axis.map(x=>x*Math.sin(angle/2)), Math.cos(angle/2)];
const matrix = (rotation = [0,0,0,1], scale = 1, translation = [0,0,0]) => {
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{rotation,scale:[scale,scale,scale],translation}]});
  return Array.from(p.worldMatrices);
};
const transform = (m, v) => [0,1,2].map(i=>m[i]*v[0]+m[4+i]*v[1]+m[8+i]*v[2]+m[12+i]);
const limb = (a = 1, b = 1, extra = {}) => createAnimationPlayer({format:'f3d-animation-v1',
  nodes:[{}, {parent:0,translation:[a,0,0]}, {parent:1,translation:[b,0,0]}], ...extra});
const options = (target = [1,0,0], pole = [0,1,0]) => ({root:0,joint:1,effector:2,target,pole});
const code = name => ({code:'ANIMATION_IK_'+name});

test('analytic pole solve reaches an inward target where straight-chain CCD stalls',()=>{
  const p=limb(), original=state(p), refs=fields.map(k=>p[k]);
  const ccd=solveAnimationIK(p,{effector:2,target:[1,0,0],links:[{node:1},{node:0}]});
  assert.equal(ccd.converged,false); close(ccd.distance,1); assert.deepEqual(state(p),original);
  const r=solve(p,options()); assert.equal(r.converged,true); assert.equal(r.reachable,true);
  vectorClose(point(p,1),[0.5,Math.sqrt(3)/2,0]); vectorClose(point(p,2),[1,0,0]);
  assert.deepEqual(r.changedNodes,[0,1]); assert.equal(p.version,1);
  close(r.distance,distance(point(p,2),[1,0,0])); close(r.initialDistance,1);
  assert.equal(r.poleFallback,'none'); assert.ok(Object.isFrozen(r)); assert.ok(Object.isFrozen(r.changedNodes));
  fields.forEach((k,i)=>assert.equal(p[k],refs[i]));
  assert.deepEqual(Array.from(p.translations),original.translations); assert.deepEqual(Array.from(p.scales),original.scales);
});

test('opposite poles select opposite knees without changing the target or bone lengths',()=>{
  for(const sign of [-1,1]){
    const p=limb(); solve(p,options([1,0,0],[0,sign,0]));
    vectorClose(point(p,1),[0.5,sign*Math.sqrt(3)/2,0]); vectorClose(point(p,2),[1,0,0]);
    close(distance(point(p,0),point(p,1)),1); close(distance(point(p,1),point(p,2)),1);
  }
});

for(const [a,b,target,expected] of [[2,1,[10,0,0],[3,0,0]], [2,1,[0.2,0,0],[1,0,0]],
    [1,2,[0.2,0,0],[1,0,0]], [1,2,[0,0,0],[1,0,0]]]){
  test(`unreachable ${a}:${b} limb clamps ${target} without stretching`,()=>{
    const p=limb(a,b), r=solve(p,options(target));
    assert.equal(r.reachable,false); assert.equal(r.converged,false);
    vectorClose(point(p,2),expected,1e-8); close(r.distance,distance(expected,target),1e-8);
    close(distance(point(p,0),point(p,1)),a); close(distance(point(p,1),point(p,2)),b);
  });
}

test('equal bones fold exactly to the root while the pole chooses the joint direction',()=>{
  const p=limb(), r=solve(p,options([0,0,0],[0,0,1]));
  assert.equal(r.converged,true); assert.equal(r.reachable,true); close(r.solvedDistance,0);
  vectorClose(point(p,1),[0,0,1]); vectorClose(point(p,2),[0,0,0]);
});

test('an already folded limb opens from a deterministic axis with no division by zero',()=>{
  const p=limb(); p.edit([{node:1,rotation:q([0,0,1],Math.PI)}]);
  const r=solve(p,options([0.8,0,0],[0,1,0]));
  assert.equal(r.converged,true); vectorClose(point(p,2),[0.8,0,0]); assert.ok(point(p,1)[1]>0);
});

test('antiparallel reach and collinear pole choose a repeatable finite bend',()=>{
  const a=limb(),b=limb(), ra=solve(a,options([-1,0,0],[-2,0,0])), rb=solve(b,options([-1,0,0],[-2,0,0]));
  assert.equal(ra.converged,true); assert.equal(ra.poleFallback,'axis'); assert.deepEqual(state(a),state(b));
  assert.deepEqual(ra,rb); vectorClose(point(a,2),[-1,0,0]);
});

test('collinear poles preserve an existing bend before choosing a fallback axis',()=>{
  const p=limb(); solve(p,options([1,0,0],[0,0,-1]));
  const r=solve(p,options([0.75,0,0],[5,0,0]));
  assert.equal(r.poleFallback,'current'); assert.ok(point(p,1)[2]<0); vectorClose(point(p,2),[0.75,0,0]);
});

test('a pole at the root is explicit degeneracy, not NaN or an exception',()=>{
  const p=limb(), r=solve(p,options([1,0,0],[0,0,0]));
  assert.equal(r.poleFallback,'axis'); assert.ok(r.converged); p.rotations.forEach(v=>assert.ok(Number.isFinite(v)));
});

test('pole coordinates are world points, not direction vectors relative to the root',()=>{
  const p=limb(); p.edit([{node:0,translation:[10,20,30]}]);
  const r=solve(p,options([11,20,30],[10,21,30]));
  assert.ok(r.converged); vectorClose(point(p,1),[10.5,20+Math.sqrt(3)/2,30]);
});

test('rootMatrix, rotated matrix ancestors and non-topological indices preserve world-space targeting',()=>{
  const parent=matrix(q([0,0,1],0.6),2,[3,-2,1]), rootMatrix=matrix(q([0,1,0],0.9),3,[5,2,-4]);
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{parent:2,translation:[1,0,0]},
    {parent:3}, {parent:1,translation:[1,0,0]}, {matrix:parent}]});
  p.edit([],{rootMatrix});
  const origin=point(p,1), target=transform(rootMatrix,transform(parent,[1,0,0])), pole=transform(rootMatrix,transform(parent,[0,1,0]));
  const r=solve(p,{root:1,joint:2,effector:0,target,pole});
  assert.ok(r.converged); vectorClose(point(p,0),target); vectorClose(point(p,1),origin);
  close(distance(point(p,1),point(p,2)),6); close(distance(point(p,2),point(p,0)),6);
  assert.deepEqual(p.snapshotLocalPose().rootMatrix,new Float64Array(rootMatrix));
});

test('weights blend local rotations after the full analytic solve and preserve zero-weight state',()=>{
  const full=limb(),half=limb(),zero=limb(); const before=state(zero);
  solve(full,options()); const r=solve(half,{...options(),weight:0.5}); solve(zero,{...options(),weight:0});
  assert.deepEqual(state(zero),before); assert.equal(r.positionConverged,false);
  for(const node of [0,1]){
    const angleFull=2*Math.atan2(Math.hypot(...full.rotations.subarray(node*4,node*4+3)),Math.abs(full.rotations[node*4+3]));
    const angleHalf=2*Math.atan2(Math.hypot(...half.rotations.subarray(node*4,node*4+3)),Math.abs(half.rotations[node*4+3]));
    close(angleHalf,angleFull/2);
  }
  close(distance(point(half,0),point(half,1)),1); close(distance(point(half,1),point(half,2)),1);
});

for(const desired of [[0,0,0,1], q([1,0,0],Math.PI),q([0,1,0],Math.PI),q([0,0,1],Math.PI),q([0,0,1],0.7)]){
  test(`world end orientation ${desired} is independent of parent rotations`,()=>{
    const p=limb(); p.edit([{node:0,rotation:q([0,1,0],0.3)}]);
    const r=solve(p,{...options([1,0.3,0.2]),endRotation:desired});
    assert.ok(r.converged); assert.ok(r.orientationConverged); assert.ok(r.orientationError<1e-9);
    const expected=matrix(desired); for(const i of [0,1,2,4,5,6,8,9,10])close(p.worldMatrices[32+i],expected[i]);
  });
}

test('world orientation influence is continuous from the post-position end orientation',()=>{
  const base=limb(), half=limb(); solve(base,options());
  const fullError=solve(base,{...options(),weight:0,endRotation:[0,0,0,1],orientationWeight:0}).orientationError;
  const r=solve(half,{...options(),endRotation:[0,0,0,1],orientationWeight:0.5});
  close(r.orientationError,fullError/2); assert.equal(r.positionConverged,true); assert.equal(r.converged,false);
});

test('orientation-only edits leave limb positions and non-target local rotations untouched',()=>{
  const p=limb(), before=state(p), r=solve(p,{...options([2,0,0]),weight:0,orientationWeight:1,endRotation:q([0,1,0],1)});
  assert.deepEqual(r.changedNodes,[2]); vectorClose(point(p,2),[2,0,0]);
  assert.deepEqual(Array.from(p.rotations.subarray(0,8)),before.rotations.slice(0,8)); assert.ok(r.converged);
});

test('sampler time, clip state, morphs and output identities survive a solve; sampling later replaces it',()=>{
  const p=limb(1,1,{nodes:[{weights:[0.2]},{parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]}],
    clips:[{name:'morph',channels:[{node:0,path:'weights',times:[0,2],values:[0.2,0.8]}]}]});
  p.sample(0.75); const time=p.time,clip=p.clip,weights=p.morphWeights.slice(),version=p.version,refs=fields.map(k=>p[k]);
  solve(p,options()); assert.equal(p.time,time); assert.equal(p.clip,clip); assert.equal(p.version,version+1);
  assert.deepEqual(p.morphWeights,weights); fields.forEach((k,i)=>assert.equal(p[k],refs[i]));
  p.sample(0.75); vectorClose(point(p,2),[2,0,0]); assert.deepEqual(p.morphWeights,weights);
});

test('the real runtime publishes both mesh-local skin palettes with the solved pose once',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},
    {parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]}, {},{translation:[4,0,0]}],
    skins:[{joints:[0,1,2]}],instances:[{node:3,skin:0},{node:4,skin:0}]});
  const palette=p.jointMatrices, r=solve(p,options()); assert.equal(p.jointMatrices,palette); assert.equal(r.poseVersion,1);
  vectorClose(Array.from(palette.subarray(44,47)),[1,0,0],1e-6);
  vectorClose(Array.from(palette.subarray(92,95)),[-3,0,0],1e-6);
});

test('published-array corruption cannot become solver input',()=>{
  const a=limb(), b=limb();
  a.translations.fill(999); a.rotations.fill(NaN); a.worldMatrices.fill(-42);
  solve(a,options()); solve(b,options()); assert.deepEqual(state(a),state(b));
});

for(const scale of [1e-150,1e-75,1e75,1e150]){
  test(`normalized law of cosines remains finite at world length ${scale}`,()=>{
    const p=limb(2*scale,scale), target=[2*scale,0,0];
    const r=solve(p,{...options(target,[0,scale,0]),tolerance:scale*1e-8});
    assert.ok(r.converged); vectorClose(point(p,2).map(v=>v/scale),[2,0,0],1e-8);
    close(distance(point(p,0),point(p,1))/scale,2,1e-8); close(distance(point(p,1),point(p,2))/scale,1,1e-8);
  });
}

test('bad vectors, indices, limits and topology fail without changing the live pose',()=>{
  const p=limb(),before=state(p);
  const invalid=[null,[],{}, {...options(),root:3}, {...options(),joint:0}, {...options(),effector:-1},
    {...options(),target:[NaN,0,0]}, {...options(),pole:[0,0]}, {...options(),weight:-1},
    {...options(),weight:2},{...options(),tolerance:-1},{...options(),tolerance:Infinity},
    {...options(),orientationWeight:1}, {...options(),endRotation:[0,0,0,0]},
    {...options(),endRotation:[0,0,0,1],orientationWeight:2}, {...options(),orientationTolerance:4}, {...options(),stretch:true}];
  for(const input of invalid){ assert.throws(()=>solve(p,input)); assert.deepEqual(state(p),before); }
  const indirect=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},{parent:0},{parent:1},{parent:2,translation:[1,0,0]}]});
  assert.throws(()=>solve(indirect,{...options(),joint:2,effector:3}),code('CHAIN'));
});

test('collapsed or numerically disproportionate bones are refused transactionally',()=>{
  for(const [a,b] of [[0,1],[1,0],[1e-14,1],[1,1e-14]]){
    const p=limb(a,b),before=state(p); assert.throws(()=>solve(p,options()),code('LENGTH')); assert.deepEqual(state(p),before);
  }
});

test('unsupported anisotropy, reflection, matrix links and shear are refused, not decomposed',()=>{
  const cases=[
    [{scale:[1,2,1]},{parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]}],
    [{scale:[-1,1,1]},{parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]}],
    [{matrix:matrix()},{parent:0,translation:[1,0,0]},{parent:1,translation:[1,0,0]}],
  ];
  for(const nodes of cases){const p=limb(1,1,{nodes}),before=state(p);assert.throws(()=>solve(p,options()),code('TRANSFORM'));assert.deepEqual(state(p),before);}
  const p=limb(), shear=matrix(); shear[4]=0.1; p.edit([],{rootMatrix:shear});
  const before=state(p); assert.throws(()=>solve(p,options()),code('TRANSFORM'));assert.deepEqual(state(p),before);
});

test('matrix effector position can be solved, but its world orientation cannot be silently rewritten',()=>{
  const p=limb(1,1,{nodes:[{},{parent:0,translation:[1,0,0]},{parent:1,matrix:matrix(undefined,1,[1,0,0])}]});
  assert.ok(solve(p,options()).converged); const before=state(p);
  assert.throws(()=>solve(p,{...options(),endRotation:[0,0,0,1]}),code('TRANSFORM')); assert.deepEqual(state(p),before);
});

test('input getters that change or dispose the pose are detected before publication',()=>{
  const p=limb(); let after;
  const bad={...options(),get pole(){p.edit([{node:0,translation:[3,0,0]}]);after=state(p);return [0,1,0];}};
  assert.throws(()=>solve(p,bad),code('STALE')); assert.deepEqual(state(p),after);
  const d=limb(),old=d.version; assert.throws(()=>solve(d,{...options(),get target(){d.dispose();return [1,0,0];}}),code('STALE'));
  assert.equal(d.version,old);
});

test('same-pose recursive limb solving is blocked and the guard is released after failure',()=>{
  const p=limb(); assert.throws(()=>solve(p,{...options(),get target(){return solve(p,options());}}),code('REENTRANT'));
  assert.equal(p.version,0); assert.ok(solve(p,options()).converged);
});

test('vector reads are bounded even with a hostile custom iterator',()=>{
  const target=[1,0,0]; target[Symbol.iterator]=()=>{throw new Error('unexpected iterator');};
  assert.ok(solve(limb(),options(target)).converged);
});

test('detached pose storage and final palette failure never publish half a limb',()=>{
  const detached=limb(); structuredClone(detached.worldMatrices.buffer,{transfer:[detached.worldMatrices.buffer]});
  assert.throws(()=>solve(detached,options()),{code:'ANIMATION_OUTPUT_STORAGE'}); assert.equal(detached.version,0);
  // Initial palette is finite. Rotation sends a huge inverse-bind translation
  // into a huge nonuniform mesh inverse, overflowing only the FINAL f32 palette.
  const ibm=matrix(undefined,1,[0,1e30,0]);
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},
    {parent:0,translation:[1,0,0]}, {parent:1,translation:[1,0,0]}, {scale:[1e-10,1,1]}],
    skins:[{joints:[0],inverseBindMatrices:ibm}],instances:[{node:3,skin:0}]});
  const before=state(p); assert.throws(()=>solve(p,options()),{code:'ANIMATION_VALUE'}); assert.deepEqual(state(p),before);
});

test('300 deterministic randomized limbs reach valid targets and preserve lengths and pole half-plane',()=>{
  let seed=0x762abd13; const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
  for(let i=0;i<300;i++){
    const a=0.25+random()*3,b=0.25+random()*3,axis=[random()-.5,random()-.5,random()-.5],len=Math.hypot(...axis);
    const rotation=q(axis.map(x=>x/len),random()*6),s=0.2+random()*4,t=[random()*8,random()*8,random()*8];
    const p=limb(a,b),m=matrix(rotation,s,t); p.edit([],{rootMatrix:m});
    const d=Math.abs(a-b)+(a+b-Math.abs(a-b))*(0.01+random()*0.98), target=transform(m,[d,0,0]),pole=transform(m,[0,1,0]);
    const r=solve(p,{...options(target,pole),tolerance:1e-8});
    assert.ok(r.converged,JSON.stringify({i,a,b,d,r})); vectorClose(point(p,2),target,1e-8);
    close(distance(point(p,0),point(p,1)),a*s,1e-8);close(distance(point(p,1),point(p,2)),b*s,1e-8);
    const x=(d*d+a*a-b*b)/(2*d),h=Math.sqrt(Math.max(0,a*a-x*x));vectorClose(point(p,1),transform(m,[x,h,0]),1e-8);
  }
});
