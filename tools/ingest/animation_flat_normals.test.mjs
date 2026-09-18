import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createAnimationDeformer} from './animation_deformer.mjs';

const triangle = () => [0,0,0,1,0,0,0,1,0];
const geometry = extras => ({node:0,positions:triangle(),flatNormals:true,...extras});
const pose = (nodes=[{}],extras={}) => createAnimationPlayer({format:'f3d-animation-v1',nodes,...extras});
const code = expected => error => error.code === expected;
function near(actual,expected,tolerance=2e-6) {
  assert.equal(actual.length,expected.length);
  for (let i=0;i<actual.length;i++) assert.ok(Math.abs(actual[i]-expected[i])<=tolerance,`${i}: ${actual[i]} != ${expected[i]}`);
}
const normals = value => [...value,...value,...value];
function animated() {
  const p = pose([{weights:[0]}, {}, {}],{skins:[{joints:[1,2]}],instances:[{node:0,skin:0}],clips:[{channels:[
    {node:0,path:'weights',times:[0,1],values:[0,1]},
    {node:2,path:'translation',times:[0,1],values:[0,0,0,0,0,2]},
  ]}]});
  const g = geometry({joints:[0,1,0],weights:[1,1,1],influences:1,
    morphTargets:[{positions:[0,0,0,0,0,0,0,0,1]}]});
  return {p,g};
}

test('flat normals are generated for independent triangles without normal placeholders', () => {
  const p=pose(),d=createAnimationDeformer(p,geometry());
  near(d.normals,normals([0,0,1])); assert.equal(d.tangents,null);
  assert.equal(d.poseVersion,p.version); assert.equal(d.vertexCount,3);
  d.dispose(); assert.equal(p.disposed,false); p.dispose();
});

test('morphing precedes skinning and normals follow the final triangle, not a blended rest direction', () => {
  const {p,g}=animated(),d=createAnimationDeformer(p,g),output=d.normals,positions=d.positions,bounds=d.bounds;
  p.sample(1); d.update();
  near(d.positions,[0,0,0,1,0,2,0,1,1]);
  near(d.normals,normals([-2/Math.sqrt(6),-1/Math.sqrt(6),1/Math.sqrt(6)]));
  p.sample(.5); d.update();
  near(d.normals,normals([-2/3,-1/3,2/3]));
  assert.equal(d.normals,output); assert.equal(d.positions,positions); assert.equal(d.bounds,bounds);
  near(d.bounds.min,[0,0,0]); near(d.bounds.max,[1,1,1]);
  assert.equal(d.poseVersion,p.version); d.dispose(); p.dispose();
});

test('CPU pose blending also recomputes flat normals after the blended pose', () => {
  const {p,g}=animated(),d=createAnimationDeformer(p,g);
  p.blend([{clip:0,time:1,weight:.5}]); d.update();
  near(d.positions,[0,0,0,1,0,1,0,1,.5]); near(d.normals,normals([-2/3,-1/3,2/3]));
  d.dispose(); p.dispose();
});

test('morph-only primitives recompute orientation and preserve independent instance outputs', () => {
  const p=pose([{weights:[0]},{weights:[1]}]);
  const g=geometry({morphTargets:[{positions:[0,0,0,0,0,1,0,0,0]}]});
  const a=createAnimationDeformer(p,g),b=createAnimationDeformer(p,{...g,node:1});
  near(a.normals,normals([0,0,1])); near(b.normals,normals([-Math.SQRT1_2,0,Math.SQRT1_2]));
  assert.notEqual(a.normals,b.normals); a.normals.fill(9); b.update();
  near(b.normals,normals([-Math.SQRT1_2,0,Math.SQRT1_2])); a.dispose(); b.dispose(); p.dispose();
});

test('world transforms are not baked twice into mesh-local face normals', () => {
  const p=pose([{translation:[10,20,30],scale:[2,3,4],weights:[1]}]);
  const d=createAnimationDeformer(p,geometry({morphTargets:[{positions:[0,0,0,0,0,1,0,0,0]}]}));
  near(d.positions,[0,0,0,1,0,1,0,1,0]); near(d.normals,normals([-Math.SQRT1_2,0,Math.SQRT1_2]));
  near(d.worldMatrix.slice(12,15),[10,20,30]); d.dispose(); p.dispose();
});

test('adjacent triangles retain separate flat directions and source winding', () => {
  const p=pose(),d=createAnimationDeformer(p,geometry({positions:[...triangle(),0,0,0,0,1,1,1,0,0]}));
  near(d.normals.slice(0,9),normals([0,0,1])); near(d.normals.slice(9),normals([0,Math.SQRT1_2,-Math.SQRT1_2]));
  d.dispose(); p.dispose();
});

test('collapsed triangles publish zero normals, then recover when animation unfolds them', () => {
  const p=pose([{weights:[0]}],{clips:[{channels:[{node:0,path:'weights',times:[0,1],values:[0,1]}]}]});
  const d=createAnimationDeformer(p,geometry({morphTargets:[{positions:[0,0,0,0,0,0,0,-1,0]}]}));
  p.sample(1); d.update(); near(d.normals,new Array(9).fill(0));
  p.reset(); d.update(); near(d.normals,normals([0,0,1])); d.dispose(); p.dispose();
});

for (const scale of [1e-38,1e-20,1,1e20,1e38]) test(`finite Float32 triangles at scale ${scale} have finite unit normals`, () => {
  const p=pose(),d=createAnimationDeformer(p,geometry({positions:triangle().map(x=>x*scale)}));
  near(d.normals,normals([0,0,1])); assert.ok(d.normals.every(Number.isFinite)); d.dispose(); p.dispose();
});

test('geometry arrays, the flat flag and optional ignored rest normals are snapshotted', () => {
  const p=pose(),g=geometry({normals:new Array(9).fill(7)}),d=createAnimationDeformer(p,g);
  g.flatNormals=false; g.positions.fill(0); g.normals.fill(9); d.update();
  near(d.positions,triangle()); near(d.normals,normals([0,0,1])); d.dispose(); p.dispose();
});

test('invalid flat layout, configuration and incompatible tangent data are rejected', () => {
  const p=pose();
  for (const invalid of [null,1,0,'true',{},[]]) assert.throws(()=>createAnimationDeformer(p,geometry({flatNormals:invalid})),code('ANIMATION_DEFORM_NORMAL'));
  for (const count of [1,2,4,5]) assert.throws(()=>createAnimationDeformer(p,geometry({positions:new Array(count*3).fill(0)})),code('ANIMATION_DEFORM_NORMAL'));
  assert.throws(()=>createAnimationDeformer(p,geometry({tangents:new Array(3).fill([1,0,0,1]).flat()})),code('ANIMATION_DEFORM_NORMAL'));
  p.dispose();
});

test('normal and tangent morph targets cannot silently replace the flat contract', () => {
  const p=pose([{weights:[0]}]);
  for (const field of ['normals','tangents']) assert.throws(()=>createAnimationDeformer(p,geometry({morphTargets:[{[field]:new Array(9).fill(0)}]})),code('ANIMATION_DEFORM_NORMAL'));
  p.dispose();
});

test('generated normals are charged before allocation and provided normals are not double-charged', () => {
  const p=pose();
  assert.throws(()=>createAnimationDeformer(p,geometry(),{maxComponents:17}),code('ANIMATION_DEFORM_LIMIT'));
  for (const g of [geometry(),geometry({normals:new Array(9).fill(0)})]) {
    const d=createAnimationDeformer(p,g,{maxComponents:18}); near(d.normals,normals([0,0,1])); d.dispose();
  }
  p.dispose();
});

test('a failed deformation preserves normals, positions, bounds, transforms and version together', () => {
  const {p,g}=animated(),d=createAnimationDeformer(p,g); p.sample(.5); d.update();
  const before={positions:[...d.positions],normals:[...d.normals],min:[...d.bounds.min],max:[...d.bounds.max],world:[...d.worldMatrix],version:d.version,poseVersion:d.poseVersion};
  p.jointMatrices[0]=NaN; assert.throws(()=>d.update(),code('ANIMATION_DEFORM_VALUE'));
  assert.deepEqual([...d.normals],before.normals); assert.deepEqual([...d.positions],before.positions);
  assert.deepEqual([...d.bounds.min],before.min); assert.deepEqual([...d.bounds.max],before.max);
  assert.deepEqual([...d.worldMatrix],before.world); assert.equal(d.version,before.version); assert.equal(d.poseVersion,before.poseVersion);
  p.reset(); d.update(); near(d.normals,normals([0,0,1])); d.dispose(); p.dispose();
});

test('overflowing morph output cannot publish a new normal or partially advance a deformer', () => {
  const {p,g}=animated(),d=createAnimationDeformer(p,g),before=[...d.normals],version=d.version;
  p.morphWeights[0]=1e100; assert.throws(()=>d.update(),code('ANIMATION_DEFORM_VALUE'));
  assert.deepEqual([...d.normals],before); assert.equal(d.version,version);
  p.reset(); d.update(); d.dispose(); p.dispose();
});

test('detached normal output is rejected before any new position publication', () => {
  const {p,g}=animated(),d=createAnimationDeformer(p,g),before=[...d.positions],version=d.version;
  structuredClone(d.normals.buffer,{transfer:[d.normals.buffer]}); p.sample(1);
  assert.throws(()=>d.update(),code('ANIMATION_DEFORM_STORAGE')); assert.deepEqual([...d.positions],before); assert.equal(d.version,version);
  d.dispose(); p.dispose();
});

test('ordinary authored normals keep their existing unnormalized deformation contract', () => {
  const p=pose();
  const a=createAnimationDeformer(p,{node:0,positions:triangle()}); assert.equal(a.normals,null);
  const b=createAnimationDeformer(p,geometry({flatNormals:false,normals:normals([0,0,3])}));
  near(b.normals,normals([0,0,3])); a.dispose(); b.dispose(); p.dispose();
});

test('deterministic animated triangles produce face-parallel unit normals over many poses', () => {
  let seed=341; const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32)*2-1;
  const p=pose([{weights:[0]}],{clips:[{channels:[{node:0,path:'weights',times:[0,1],values:[-1,2]}]}]});
  for(let trial=0;trial<40;trial++) {
    const d=createAnimationDeformer(p,geometry({positions:Array.from({length:9},random),morphTargets:[{positions:Array.from({length:9},random)}]}));
    for(const time of [0,.2,.5,.9,1]) {
      p.sample(time); d.update(); const v=d.positions,n=d.normals;
      const a=[v[3]-v[0],v[4]-v[1],v[5]-v[2]],b=[v[6]-v[0],v[7]-v[1],v[8]-v[2]];
      assert.ok(Math.abs(Math.hypot(...n.slice(0,3))-1)<2e-6);
      assert.ok(Math.abs(a.reduce((s,x,i)=>s+x*n[i],0))<2e-6);
      assert.ok(Math.abs(b.reduce((s,x,i)=>s+x*n[i],0))<2e-6);
      near(n.slice(0,3),n.slice(3,6),0); near(n.slice(0,3),n.slice(6,9),0);
    }
    d.dispose();
  }
  p.dispose();
});
