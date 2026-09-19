import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const T=(x,y=0,z=0)=>{const m=I();m[12]=x;m[13]=y;m[14]=z;return m;};
const Q=a=>[0,0,Math.sin(a/2),Math.cos(a/2)];
const fields=['translations','rotations','scales','morphWeights','worldMatrices','jointMatrices'];
const state=p=>({version:p.version,time:p.time,clip:p.clip,mode:p.mode,...Object.fromEntries(fields.map(k=>[k,[...p[k]]]))});
const near=(a,b,eps=1e-6)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<eps,`${i}: ${a[i]} != ${b[i]}`);};
const position=(p,n)=>[...p.worldMatrices.subarray(n*16+12,n*16+15)];
function fixture() {
  return {format:'f3d-animation-v1',nodes:[
    {translation:[1,0,0]}, {parent:0,translation:[0,2,0]},
    {translation:[4,0,0],weights:[0,0]}, {translation:[-2,0,0],weights:[0,0]},
    {matrix:T(5)}, {parent:4,translation:[1,0,0]},
  ],skins:[{joints:[0,1],inverseBindMatrices:[...T(-1),...T(-1,-2)]}],instances:[{node:2,skin:0},{node:3,skin:0}],clips:[
    {channels:[{node:0,path:'translation',times:[0,1],values:[1,0,0,3,0,0]},
      {node:2,path:'weights',times:[0,1],values:[0,0,.8,.4]}]},
    {channels:[{node:2,path:'scale',times:[0,1],values:[1,1,1,0,0,0]}]},
  ]};
}
test('batch TRS, morph and matrix edits publish one coherent pose without changing buffer identities',()=>{
  const p=createAnimationPlayer(fixture()),arrays=fields.map(f=>p[f]),root=T(10,20);
  p.sample(.5,{rootMatrix:root});root.fill(999);const version=p.version;
  assert.equal(p.edit([{node:0,rotation:Q(Math.PI/2)},{node:1,translation:[0,4,0]},
    {node:2,scale:[2,2,2],weights:[.1,.9]},{node:4,matrix:T(-5)}]),p);
  assert.equal(p.version,version+1);assert.equal(p.mode,'edit');assert.equal(p.time,.5);assert.equal(p.clip,0);
  near(position(p,0),[12,20,0]);near(position(p,1),[8,20,0]);near(position(p,2),[14,20,0]);near(position(p,5),[6,20,0]);
  near(p.morphWeights,[.1,.9,0,0]);
  // inverse(meshWorld)*jointWorld*inverseBind, independently evaluated here.
  near(p.jointMatrices.subarray(12,15),[-1,-.5,0]);
  near(p.jointMatrices.subarray(28,31),[-2,-.5,0]);
  near(p.jointMatrices.subarray(44,47),[4,-1,0]);
  near(p.jointMatrices.subarray(60,63),[2,-1,0]);
  fields.forEach((f,i)=>assert.equal(p[f],arrays[i]));
});
test('omitted properties and root preserve committed values; explicit null clears root',()=>{
  const p=createAnimationPlayer(fixture());p.sample(.25,{rootMatrix:T(10)});
  p.edit([{node:0,translation:[7,8,9]}]);p.edit([{node:0,scale:[2,3,4]}]);
  near(position(p,1),[17,14,9]);near(p.rotations.subarray(0,4),[0,0,0,1]);
  p.edit([],{rootMatrix:null});near(position(p,1),[7,14,9]);
  p.edit([],{rootMatrix:T(-10)});near(position(p,1),[-3,14,9]);
});
test('sample/blend/reset still replace edits with sampled or imported values',()=>{
  const p=createAnimationPlayer(fixture());
  for(const next of [()=>p.sample(.5),()=>p.blend([{clip:0,time:.5}])]) {
    p.edit([{node:0,translation:[100,0,0]},{node:2,weights:[5,7]},{node:4,matrix:T(99)}],{rootMatrix:T(500)});
    next();near(position(p,0),[2,0,0]);near(position(p,5),[6,0,0]);near(p.morphWeights,[.4,.2,0,0]);
    p.edit([{node:1,translation:[0,3,0]}]);near(position(p,1),[2,3,0]);
  }
  p.reset();near(position(p,0),[1,0,0]);near(position(p,1),[1,2,0]);near(p.morphWeights,[0,0,0,0]);
});
test('private baseline is not poisoned by failed sampling, blending or edits',()=>{
  const p=createAnimationPlayer(fixture());p.sample(.5,{rootMatrix:T(10)});p.edit([{node:4,matrix:T(8)}]);
  const before=state(p),snapshot=p.snapshotLocalPose();
  for(const bad of [()=>p.sample(1,{clip:1}),()=>p.blend([{clip:1,time:1}]),
    ()=>p.edit([{node:0,translation:[30,0,0]},{node:2,scale:[0,0,0]},{node:4,matrix:T(60)}],{rootMatrix:T(900)}),
    ()=>p.edit([{node:0,scale:[1e308,1e308,1e308]},{node:1,scale:[1e308,1e308,1e308]}])]) {
    assert.throws(bad);assert.deepEqual(state(p),before);assert.deepEqual(p.snapshotLocalPose(),{...snapshot,version:p.version});
    p.edit([]);const after=state(p);delete after.version;const expected={...before,mode:'edit'};delete expected.version;assert.deepEqual(after,expected);
    before.version=p.version;
  }
});
test('snapshots own their bytes and use private state rather than edited output arrays',()=>{
  const d=fixture(),p=createAnimationPlayer(d);p.sample(.5,{rootMatrix:T(10)});
  const s=p.snapshotLocalPose(),version=p.version;
  assert.equal(s.format,'f3d-local-pose-v1');assert.equal(s.version,version);assert.equal(s.nodeCount,6);
  assert.ok(Object.isFrozen(s));assert.ok(Object.isFrozen(s.matrices));
  near(s.parents,[-1,0,-1,-1,-1,4]);near(s.restRotations.subarray(0,4),[0,0,0,1]);
  for(const f of fields)p[f].fill(777);p.morphOffsets.fill(99);d.nodes[0].translation.fill(33);
  p.edit([{node:1,translation:[0,3,0]}]);near(position(p,0),[12,0,0]);near(position(p,1),[12,3,0]);
  for(const f of ['translations','rotations','scales','morphWeights','parents','morphOffsets','rootMatrix','restRotations'])s[f].fill(999);
  s.matrices[0].matrix.fill(999);p.edit([]);near(position(p,5),[16,0,0]);
  assert.notEqual(p.snapshotLocalPose().translations.buffer,s.translations.buffer);
});
test('explicit output subarray inputs are snapshotted and are not retained',()=>{
  const p=createAnimationPlayer(fixture());p.sample(.5);
  const input=p.translations.subarray(0,3);p.edit([{node:1,translation:input},{node:0,translation:[9,0,0]}]);
  near(position(p,1),[11,0,0]);input.fill(400);p.edit([]);near(position(p,1),[11,0,0]);
});
test('edited quaternion is normalized within the accepted unit tolerance',()=>{
  const p=createAnimationPlayer(fixture());p.edit([{node:0,rotation:Q(Math.PI/2).map(x=>x*1.0001)}]);
  near(position(p,1),[-1,0,0]);assert.ok(Math.abs(Math.hypot(...p.rotations.subarray(0,4))-1)<1e-15);
});
const invalid=[
  [null],[{}],[new Float32Array(0)],[[{node:0}]],[[{node:0,scale:[1,1,1],bogus:true}]],
  [[{node:-1,translation:[1,2,3]}]],[[{node:6,translation:[1,2,3]}]],[[{node:1.5,translation:[1,2,3]}]],
  [[{node:0,rotation:[0,0,0,0]}]],[[{node:0,rotation:[0,0,0,2]}]],[[{node:0,translation:[Infinity,0,0]}]],
  [[{node:0,translation:[1,2]}]],[[{node:0,scale:['1',1,1]}]],[[{node:0,weights:[]}]],[[{node:2,weights:[1]}]],
  [[{node:0,matrix:I()}]],[[{node:4,translation:[1,2,3]}]],[[{node:4,matrix:Array(16).fill(0)}]],
  [[{node:0,translation:[1,0,0]},{node:0,scale:[1,1,1]}]],[[],null],[[],{unknown:1}],[[],{rootMatrix:[1]}],
];
for(let i=0;i<invalid.length;i++)test(`invalid edit ${i} has no partial effects`,()=>{
  const p=createAnimationPlayer(fixture()),s=state(p);assert.throws(()=>p.edit(...invalid[i]));assert.deepEqual(state(p),s);
});
test('getter reentry is rejected before publication and the instance remains usable',()=>{
  const p=createAnimationPlayer(fixture()),s=state(p);
  for(const reenter of [()=>p.edit([]),()=>p.sample(0),()=>p.snapshotLocalPose(),()=>p.dispose()]) {
    assert.throws(()=>p.edit([{node:0,get translation(){reenter();return [9,0,0];}}]),{code:'ANIMATION_REENTRANT'});
    assert.deepEqual(state(p),s);
  }
  p.edit([{node:0,translation:[7,0,0]}]);near(position(p,1),[7,2,0]);
});
test('detached outputs and disposed players reject edits/snapshots',()=>{
  const p=createAnimationPlayer(fixture());structuredClone(p.worldMatrices.buffer,{transfer:[p.worldMatrices.buffer]});
  for(const operation of [()=>p.edit([]),()=>p.snapshotLocalPose()])assert.throws(operation,{code:'ANIMATION_OUTPUT_STORAGE'});
  const q=createAnimationPlayer(fixture());q.dispose();
  for(const operation of [()=>q.edit([]),()=>q.snapshotLocalPose()])assert.throws(operation,{code:'ANIMATION_DISPOSED'});
});
test('empty pose supports atomic root-only edits and independent snapshots',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[]});p.edit([],{rootMatrix:T(8)});
  assert.equal(p.version,1);assert.equal(p.snapshotLocalPose().rootMatrix[12],8);assert.equal(p.translations.length,0);
});
test('edits tolerate unsorted node IDs and preserve matrix-node shear exactly',()=>{
  const m=I();m[4]=.25;
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{parent:2,translation:[0,2,0]},{matrix:m},{parent:1,translation:[1,0,0]}]});
  const shear=m.slice();shear[4]=2;p.edit([{node:0,translation:[0,3,0]},{node:1,matrix:shear}]);
  near(position(p,0),[7,3,0]);assert.equal(p.snapshotLocalPose().matrices[0].matrix[4],2);
});
test('STEP, LINEAR rotation, CUBICSPLINE and additive blending still feed editable poses',()=>{
  const p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{}],clips:[
    {channels:[{node:0,path:'translation',interpolation:'STEP',times:[0,1],values:[1,2,3,4,5,6]}]},
    {channels:[{node:0,path:'rotation',times:[0,1],values:[...Q(0),...Q(Math.PI)]}]},
    {channels:[{node:0,path:'translation',interpolation:'CUBICSPLINE',times:[0,1],values:[0,0,0,0,0,0,2,0,0,2,0,0,2,0,0,0,0,0]}]},
  ]});
  p.sample(.5,{clip:0});p.edit([{node:0,scale:[2,2,2]}]);near(p.translations,[1,2,3]);
  p.sample(.5,{clip:1});p.edit([{node:0,translation:[3,0,0]}]);near(p.rotations,Q(Math.PI/2));
  p.sample(.5,{clip:2});p.edit([{node:0,scale:[2,2,2]}]);near(p.translations,[1,0,0]);
  p.blend([{clip:0,time:1,weight:.5,mode:'additive'}]);p.edit([{node:0,scale:[3,3,3]}]);near(p.translations,[2,2.5,3]);
});

test('publishing uses intrinsic typed operations, not user-overridden output methods or getters',()=>{
  const p=createAnimationPlayer(fixture());
  for(const f of fields) {
    p[f].set=()=>assert.fail('public set must not run');
    Object.defineProperty(p[f],'length',{get(){assert.fail('public length must not run');}});
    Object.defineProperty(p[f],'buffer',{get(){assert.fail('public buffer must not run');}});
  }
  p.edit([{node:0,translation:[5,0,0]}]);assert.equal(p.worldMatrices[12],5);assert.equal(p.worldMatrices[28],5);
  p.reset();assert.equal(p.worldMatrices[12],1);
});
