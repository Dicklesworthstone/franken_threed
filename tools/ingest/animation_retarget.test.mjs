import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createAnimationRetargeter} from './animation_retarget.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const q=(axis,angle)=>[...axis.map(v=>v*Math.sin(angle/2)),Math.cos(angle/2)];
const z=angle=>q([0,0,1],angle),x=angle=>q([1,0,0],angle);
const close=(a,b,e=1e-9)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<e,`${i}: ${v} != ${b[i]}`));};
const definition=(nodes,extra={})=>({format:'f3d-animation-v1',nodes,...extra});
const pose=(nodes,extra)=>createAnimationPlayer(definition(nodes,extra));
const pair=()=>[pose([{}, {parent:0,translation:[1,0,0]}, {parent:1,translation:[1,0,0]}]),
  pose([{}, {parent:0,translation:[2,0,0]}, {parent:1,translation:[2,0,0]}])];
const mapping=[{source:0,target:0},{source:1,target:1}];
const rotation=(p,n)=>[...p.snapshotLocalPose().rotations.slice(n*4,n*4+4)];
const position=(p,n)=>[...p.worldMatrices.slice(n*16+12,n*16+15)];
const state=p=>({version:p.version,local:p.snapshotLocalPose(),world:p.worldMatrices.slice(),palette:p.jointMatrices.slice()});
const unchanged=(p,before)=>{assert.equal(p.version,before.version);assert.deepEqual(p.snapshotLocalPose(),before.local);assert.deepEqual(p.worldMatrices,before.world);assert.deepEqual(p.jointMatrices,before.palette);};
const code=s=>({code:'ANIMATION_RETARGET_'+s});
// Independent 3x3 matrix oracle: world rotation transfer must equal
// currentSource * transpose(referenceSource) * referenceTarget.
const matrix=(p,n)=>{const m=p.worldMatrices.subarray(n*16,n*16+16),s=Math.hypot(m[0],m[1],m[2]);return [m[0]/s,m[4]/s,m[8]/s,m[1]/s,m[5]/s,m[9]/s,m[2]/s,m[6]/s,m[10]/s];};
const transpose=a=>[a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]];
const mul=(a,b)=>Array.from({length:9},(_,i)=>{const row=Math.floor(i/3),col=i%3;return a[row*3]*b[col]+a[row*3+1]*b[col+3]+a[row*3+2]*b[col+6];});

test('motion transfers without copying source bone lengths or modifying its pose',()=>{
  const [s,t]=pair(),retarget=createAnimationRetargeter(s,t,{mapping}),translations=t.translations,world=t.worldMatrices;
  s.edit([{node:0,rotation:z(Math.PI/2)},{node:1,rotation:z(-Math.PI/2)}]);const before=state(s),version=t.version;
  const result=retarget.apply();close(position(t,1),[0,2,0]);close(position(t,2),[2,2,0]);
  close(rotation(t,0),z(Math.PI/2));close(rotation(t,1),z(-Math.PI/2));assert.equal(t.version,version+1);
  assert.deepEqual(result.updatedNodes,[0,1]);assert.equal(result.sourceVersion,s.version);assert.equal(result.targetVersion,t.version);
  assert.ok(Object.isFrozen(result)&&Object.isFrozen(result.updatedNodes));assert.equal(t.translations,translations);assert.equal(t.worldMatrices,world);unchanged(s,before);
  retarget.dispose();assert.equal(s.disposed,false);assert.equal(t.disposed,false);
});

test('different reference axes and hierarchy helper nodes obey the independent world-matrix oracle',()=>{
  const s=pose([{rotation:x(.4)},{parent:0,translation:[1,0,0],rotation:z(.6)},{parent:1,translation:[1,0,0]}]);
  // Destination parent index need not be smaller than child, and mappings need
  // not follow hierarchy order. An unmapped helper lies between mapped bones.
  const t=pose([{parent:2,translation:[2,0,0],rotation:x(-.3)},{rotation:z(-.2)},
    {parent:1,rotation:z(.8),translation:[0,1,0]},{parent:0,translation:[3,0,0]}]);
  const sr=[matrix(s,0),matrix(s,1)],tr=[matrix(t,1),matrix(t,0)];
  const r=createAnimationRetargeter(s,t,{mapping:[{source:1,target:0},{source:0,target:1}]});
  s.edit([{node:0,rotation:z(1.1)},{node:1,rotation:x(-.9)}]);r.apply();
  close(matrix(t,1),mul(mul(matrix(s,0),transpose(sr[0])),tr[0]));
  close(matrix(t,0),mul(mul(matrix(s,1),transpose(sr[1])),tr[1]));close(rotation(t,2),z(.8));
  close(t.translations.slice(0,3),[2,0,0]);close(t.translations.slice(9,12),[3,0,0]);
});

test('explicit reference snapshots can initialize a binding while both rigs are already animated',()=>{
  const [s,t]=pair(),sourceReference=s.snapshotLocalPose(),targetReference=t.snapshotLocalPose();
  s.edit([{node:0,rotation:z(.7)}]);t.edit([{node:0,rotation:x(.2)}]);
  const r=createAnimationRetargeter(s,t,{mapping,sourceReference,targetReference});
  sourceReference.rotations.fill(9);targetReference.rotations.fill(9);r.apply();close(rotation(t,0),z(.7));
});

test('references are the captured stance, not guessed inverse-bind or first-animation-frame values',()=>{
  const [s,t]=pair();s.edit([{node:0,rotation:z(.3)}]);t.edit([{node:0,rotation:x(.5)}]);
  const before=matrix(t,0),r=createAnimationRetargeter(s,t,{mapping});r.apply();close(matrix(t,0),before);
});

test('root motion supports alignment, unit conversion and different rig placement',()=>{
  const s=pose([{translation:[1,2,3]}]),t=pose([{translation:[10,20,30]}]);
  const r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:0}],alignment:z(Math.PI/2),rootMotion:{source:0,target:0,scale:2}});
  const rootS=I(),rootT=I();rootS[12]=1000;rootT[13]=100;
  s.edit([{node:0,translation:[4,2,3],rotation:x(.5)}],{rootMatrix:rootS});
  t.edit([],{rootMatrix:rootT});r.apply();close(position(t,0),[10,126,30]);
  close(t.translations,[10,26,30]);close(rotation(t,0),q([0,1,0],.5));assert.deepEqual([...t.snapshotLocalPose().rootMatrix],rootT);
});

test('root motion inverts a rotated/scaled destination parent, while preserving its current placement',()=>{
  const s=pose([{translation:[1,0,0]}]),t=pose([{rotation:z(Math.PI/2),scale:[2,2,2]},{parent:0,translation:[1,0,0]}]);
  const r=createAnimationRetargeter(s,t,{mapping:[],rootMotion:{source:0,target:1,scale:1}});
  s.edit([{node:0,translation:[5,0,0]}]);r.apply();close(position(t,1),[4,2,0]);close(t.translations.slice(3,6),[1,-2,0]);close(rotation(t,0),z(Math.PI/2));
});

test('partial transfer blends current destination local rotations and root translation',()=>{
  const [s,t]=pair(),r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:0}],rootMotion:{source:0,target:0,scale:2}});
  s.edit([{node:0,rotation:z(Math.PI/2),translation:[4,0,0]}]);r.apply({weight:.5});close(rotation(t,0),z(Math.PI/4));close(t.translations.slice(0,3),[4,0,0]);
  r.apply({weight:.5});close(rotation(t,0),z(Math.PI*3/8));close(t.translations.slice(0,3),[6,0,0]);
});

test('zero weight does not publish; mapping weights preserve unmapped local animation and all morphs/scales',()=>{
  const s=pose([{}, {parent:0},{weights:[0,1]}]),t=pose([{}, {parent:0}, {weights:[.2,.8]}]);
  const r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:0,weight:.5},{source:1,target:1,weight:0}]});
  s.edit([{node:0,rotation:z(1)}]);t.edit([{node:1,rotation:x(.6),scale:[2,2,2]},{node:2,weights:[.4,.6]}]);
  const before=state(t);assert.deepEqual(r.apply({weight:0}).updatedNodes,[]);unchanged(t,before);
  r.apply();close(rotation(t,0),z(.5));close(rotation(t,1),x(.6));close(t.scales.slice(3,6),[2,2,2]);close(t.morphWeights,[.4,.6]);
});

test('source public output edits never poison reference or current transfer truth',()=>{
  const [s,t]=pair(),r=createAnimationRetargeter(s,t,{mapping});s.edit([{node:0,rotation:z(.5)}]);
  s.rotations.fill(NaN);s.worldMatrices.fill(NaN);t.rotations.fill(NaN);r.apply();close(rotation(t,0),z(.5));assert.ok(t.worldMatrices.every(Number.isFinite));
});

for(const angle of [0,Math.PI/2,Math.PI,-Math.PI/2])test(`positive uniform matrix ancestors support rotation ${angle}`,()=>{
  const c=Math.cos(angle),sine=Math.sin(angle),m=[2*c,2*sine,0,0,-2*sine,2*c,0,0,0,0,2,0,3,4,5,1];
  const s=pose([{matrix:m},{parent:0,translation:[1,0,0]}]),t=pose([{matrix:m},{parent:0,translation:[2,0,0]}]);
  const r=createAnimationRetargeter(s,t,{mapping:[{source:1,target:1}]});s.edit([{node:1,rotation:x(.7)}]);r.apply();close(rotation(t,1),x(.7));
});

test('source matrix nodes can drive mapped TRS targets without changing their representation',()=>{
  const s=pose([{matrix:I()}]),t=pose([{}]),r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:0}]});
  const m=[0,1,0,0,-1,0,0,0,0,0,1,0,0,0,0,1];s.edit([{node:0,matrix:m}]);r.apply();close(rotation(t,0),z(Math.PI/2));
});

test('unrelated sheared/nonuniform nodes do not block a valid rig chain',()=>{
  const shear=I();shear[4]=.2;const s=pose([{}, {matrix:shear}]),t=pose([{}, {scale:[1,2,3]}]);
  const r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:0}]});s.edit([{node:0,rotation:x(.3)}]);r.apply();close(rotation(t,0),x(.3));close(t.scales.slice(3,6),[1,2,3]);
});

for(const scale of [[-1,-1,-1],[1,2,1],[0,0,0]])test(`rejects inadmissible active scales ${scale} without publication`,()=>{
  const [s,t]=pair(),r=createAnimationRetargeter(s,t,{mapping}),before=state(t);s.edit([{node:0,scale}]);assert.throws(()=>r.apply(),code('TRANSFORM'));unchanged(t,before);
  s.edit([{node:0,scale:[1,1,1]}]);r.apply();assert.equal(t.version,before.version+1);
});

test('reflected and sheared matrix ancestors are refused, never silently decomposed',()=>{
  for(const mutate of [m=>{m[0]=-1;},m=>{m[4]=.5;},m=>{m[5]=2;}]){
    const m=I();mutate(m);const s=pose([{matrix:m},{parent:0}]),t=pose([{}]),before=state(t);
    assert.throws(()=>createAnimationRetargeter(s,t,{mapping:[{source:1,target:0}]}),code('TRANSFORM'));unchanged(t,before);
  }
});

test('both independent mesh-local skin palettes update together with bone motion',()=>{
  const s=pose([{}]),t=pose([{translation:[10,0,0]},{translation:[20,0,0]},{}],{
    skins:[{joints:[2]}],instances:[{node:0,skin:0},{node:1,skin:0}]});
  const r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:2}],rootMotion:{source:0,target:2}}),palette=t.jointMatrices;
  s.edit([{node:0,translation:[4,0,0],rotation:z(Math.PI/2)}]);r.apply();
  assert.equal(t.jointMatrices,palette);assert.equal(palette[12],-6);assert.equal(palette[28],-16);
  close(palette.slice(0,4),[0,1,0,0],1e-6);close(palette.slice(16,20),[0,1,0,0],1e-6);
});

test('final Float32 skin-palette overflow rolls back the whole destination and retargeter remains usable',()=>{
  const s=pose([{}]),t=pose([{},{}],{skins:[{joints:[1]}],instances:[{node:0,skin:0}]});
  const r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:1}],rootMotion:{source:0,target:1}}),before=state(t);
  s.edit([{node:0,translation:[1e40,0,0],rotation:z(.3)}]);assert.throws(()=>r.apply(),{code:'ANIMATION_VALUE'});unchanged(t,before);
  s.edit([{node:0,translation:[2,0,0]}]);r.apply();assert.equal(t.jointMatrices[12],2);assert.equal(t.version,before.version+1);
});

test('pose sampling and blending remain usable after procedural transfer',()=>{
  const channels=[{node:0,path:'rotation',times:[0,1],values:[...z(0),...z(1)]}],s=pose([{}],{clips:[{channels}]}),t=pose([{}],{clips:[{channels}]});
  const r=createAnimationRetargeter(s,t,{mapping:[{source:0,target:0}]});s.sample(.5);r.apply();close(rotation(t,0),z(.5));
  t.sample(.1);close(rotation(t,0),z(.1));t.blend([{clip:0,time:1,weight:.4}]);close(rotation(t,0),z(.4));r.apply();close(rotation(t,0),z(.5));
});

for(const options of [{mapping:[]},{mapping:[{source:0,target:0},{source:1,target:0}]},{mapping:[{source:99,target:0}]},
  {mapping:[{source:0,target:0,weight:2}]},{mapping,alignment:[0,0,0,0]},{mapping,rootMotion:{source:0,target:0,scale:-1}},
  {mapping,maxNodes:5},{mapping,maxMappings:1},{mapping,unknown:true}])test('invalid configuration is rejected without destination effects '+JSON.stringify(options),()=>{
  const [s,t]=pair(),before=state(t);assert.throws(()=>createAnimationRetargeter(s,t,options));unchanged(t,before);
});

test('a destination matrix node cannot be retargeted as TRS',()=>{
  assert.throws(()=>createAnimationRetargeter(pose([{}]),pose([{matrix:I()}]),{mapping:[{source:0,target:0}]}),code('MAPPING'));
});

test('configuration is snapshotted and exposed mapping records cannot be rewritten',()=>{
  const [s,t]=pair(),bindings=[{source:0,target:0}],root={source:0,target:0,scale:2};
  const r=createAnimationRetargeter(s,t,{mapping:bindings,rootMotion:root});bindings[0].source=99;root.scale=99;
  assert.throws(()=>{r.mapping[0].target=2;},TypeError);s.edit([{node:0,translation:[1,0,0]}]);r.apply();close(t.translations.slice(0,3),[2,0,0]);
});

test('disposed/detached source and destination poses cannot publish a transfer',()=>{
  for(const which of [0,1])for(const detach of [false,true]){
    const [s,t]=pair(),r=createAnimationRetargeter(s,t,{mapping}),version=t.version;
    const p=[s,t][which];if(detach)structuredClone(p.worldMatrices.buffer,{transfer:[p.worldMatrices.buffer]});else p.dispose();
    assert.throws(()=>r.apply());assert.equal(t.version,version);
  }
});

test('stale source snapshots are detected before any destination edit',()=>{
  const [s,t]=pair();let mutate=false;const proxy={get version(){return s.version;},nodeCount:s.nodeCount,get disposed(){return s.disposed;},
    snapshotLocalPose(){const snapshot=s.snapshotLocalPose();if(mutate)s.edit([{node:0,translation:[1,0,0]}]);return snapshot;}};
  const r=createAnimationRetargeter(proxy,t,{mapping}),before=state(t);mutate=true;assert.throws(()=>r.apply(),code('STALE'));unchanged(t,before);
});

test('retargeter locks option getters against reentry and disposal, then recovers',()=>{
  const [s,t]=pair(),r=createAnimationRetargeter(s,t,{mapping}),before=state(t);
  assert.throws(()=>r.apply({get weight(){r.dispose();return 1;}}),code('REENTRANT'));unchanged(t,before);
  assert.throws(()=>r.apply({get weight(){r.apply();return 1;}}),code('REENTRANT'));unchanged(t,before);
  r.apply();r.dispose();r.dispose();assert.throws(()=>r.apply(),code('DISPOSED'));assert.equal(s.disposed,false);assert.equal(t.disposed,false);
});

test('large/deep hierarchy traversal is bounded and nonrecursive',()=>{
  const nodes=Array.from({length:3000},(_,i)=>i?{parent:i-1,translation:[1,0,0]}:{}),s=pose(nodes),t=pose(nodes);
  const r=createAnimationRetargeter(s,t,{mapping:[{source:2998,target:2998}],maxNodes:6000});s.edit([{node:2998,rotation:z(Math.PI/2)}]);r.apply();
  close(position(t,2999),[2998,1,0]);
});
