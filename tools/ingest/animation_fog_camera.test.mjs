import assert from 'node:assert/strict';
import test from 'node:test';
import {animationFogDepthFromProjection as depthRow, snapshotAnimationCameraFog as snapshot} from './animation_fog_camera.mjs';
import {packAnimationFog} from './animation_fog.mjs';
const I = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const mul = (m, v) => Array.from({length:4}, (_, r) => v.reduce((sum,x,c) => sum + m[c*4+r]*x, 0));
const dot = (a,b) => a.reduce((s,x,i) => s+x*b[i],0);
const near = (a,b,eps=1e-10) => assert.ok(Math.abs(a-b)<=eps*Math.max(1,Math.abs(a),Math.abs(b)), `${a} != ${b}`);
const perspective = (near=.1, far=100) => [1.7,0,0,0, 0,2.3,0,0, .2,-.15,-(far+near)/(far-near),-1, 0,0,-2*far*near/(far-near),0];
const native = p => p.map((x,i)=>i%4===2 ? .5*x+.5*p[i+1] : x);
const fog = () => ({type:'linear',color:[.1,.2,.3],near:2,far:12});

for(const [name, matrix, clipSpace] of [
  ['WebGL off-axis perspective',perspective(),'webgl'],
  ['WebGPU off-axis perspective',native(perspective()),'webgpu'],
  ['WebGL orthographic',[.3,0,0,0, 0,.4,0,0, 0,0,-.02,0, -.2,.3,-1.002,1],'webgl'],
  ['WebGPU orthographic',[.3,0,0,0, 0,.4,0,0, 0,0,-.01,0, -.2,.3,-.001,1],'webgpu'],
  ['infinite-far perspective',[1.5,0,0,0, 0,2,0,0, .1,.2,-1,-1, 0,0,-.2,0],'webgl'],
  ['infinite reverse-Z',[1.5,0,0,0, 0,2,0,0, .1,.2,0,-1, 0,0,.1,0],'webgpu'],
  ['oblique projection',[1.5,0,.07,0, 0,2,.13,0, .1,.2,-1.02,-1, 0,0,-.202,0],'webgl'],
  ['general invertible matrix',[2,.3,.2,.1, .4,3,.2,.3, .1,.4,4,.2, .2,.1,.3,5],'webgpu'],
]) test(`${name}: recover view depth before division without mutating projection`,()=>{
  const original=matrix.slice(), row=depthRow(matrix,{clipSpace});
  const p=clipSpace==='webgl'?native(matrix):matrix;
  for(const z of [-.1,-1,-5,-50,-1000]) for(const [x,y] of [[0,0],[3,-2],[-10,20]]) {
    const clip=mul(p,[x,y,z,1]);
    near(dot(row,clip),-z);
    // The renderer stores the row in f32; exercise its actual packing path too.
    const packed=packAnimationFog({...fog(),depthFromClip:row});
    near(dot([...packed.slice(0,4)],clip),-z,2e-5);
  }
  assert.deepEqual(matrix,original);
  assert.ok(Object.isFrozen(row));
});

test('perspective-correct interpolation recovers view-space fog depth; divided clip Z does not',()=>{
  const p=native(perspective()), row=depthRow(p);
  const vertices=[[0,0,-1,1],[1,0,-5,1],[0,1,-11,1]], weights=[.2,.3,.5];
  const clips=vertices.map(v=>mul(p,v));
  const denominator=weights.reduce((s,w,i)=>s+w/clips[i][3],0);
  const depth=weights.reduce((s,w,i)=>s+w*dot(row,clips[i])/clips[i][3],0)/denominator;
  const expected=weights.reduce((s,w,i)=>s-w*vertices[i][2]/clips[i][3],0)/denominator;
  near(depth,expected);
  assert.ok(Math.abs(depth-weights.reduce((s,w,i)=>s+w*clips[i][2]/clips[i][3],0))>1);
});

test('500 deterministic invertible matrices satisfy the independent row identity',()=>{
  let seed=1729;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
  for(let n=0;n<500;n++) {
    const p=Array.from({length:16},(_,i)=>(i%5===0?4:0)+random()-.5);
    const row=depthRow(p);
    for(let c=0;c<4;c++) near(dot(row,p.slice(c*4,c*4+4)),c===2?-1:0);
  }
});

test('tiny invertible projections are not misclassified by an absolute determinant threshold',()=>{
  const p=I().map(x=>x*1e-20), row=depthRow(p);
  near(row[2],-1e20); near(dot(row,mul(p,[0,0,-7,1])),7);
});

for(const [name,matrix,options] of [
  ['singular',new Array(16).fill(0)],
  ['dependent columns',[1,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,1]],
  ['non-finite',I().map((x,i)=>i===0?Infinity:x)],
  ['NaN',I().map((x,i)=>i===5?NaN:x)],
  ['string',I().map((x,i)=>i===3?'0':x)],
  ['missing element',I().slice(1)],
  ['extra element',[...I(),0]],
  ['data view',new DataView(new ArrayBuffer(128))],
  ['unknown clip space',I(),{clipSpace:'opengl'}],
  ['inverse exceeds f32',I().map(x=>x*1e-50)],
]) test(`reject ${name}`,()=>assert.throws(()=>depthRow(matrix,options), {code:'ANIMATION_FOG_CAMERA'}));

test('projection and color iterators cannot bypass fixed input extents',()=>{
  const p=I(), f=fog();
  p[Symbol.iterator]=()=>{throw new Error('must not iterate projection');};
  f.color[Symbol.iterator]=()=>{throw new Error('must not iterate color');};
  assert.deepEqual(snapshot(f,p).depthFromClip,[0,0,-1,0]);
});

test('linear and exp2 snapshots are independent and deeply frozen at public arrays',()=>{
  const p=perspective(), f=fog(), first=snapshot(f,p,{clipSpace:'webgl'});
  f.color[0]=9; f.near=3; p[10]=-.5;
  assert.equal(first.color[0],.1); assert.equal(first.near,2);
  assert.ok(Object.isFrozen(first)); assert.ok(Object.isFrozen(first.color));
  assert.ok(Object.isFrozen(first.depthFromClip));
  const second=snapshot(f,p,{clipSpace:'webgl'});
  assert.equal(second.near,3); assert.equal(second.color[0],9);
  assert.notDeepEqual(second.depthFromClip,first.depthFromClip);
  const exp=snapshot({type:'exp2',color:[2,0,1],density:0},I());
  const packed=packAnimationFog(exp);
  assert.equal(packed[10],0); assert.equal(packed[11],2);
});

test('null disables fog without accessing a camera or starting GPU work',()=>{
  assert.equal(snapshot(null,new Proxy({}, {get(){throw new Error('unused camera');}})),null);
  assert.deepEqual([...packAnimationFog(snapshot(null,null))],Array(12).fill(0));
});

for(const [name,input] of [
  ['unknown type',{...fog(),type:'height'}],
  ['unexpected field',{...fog(),depthFromClip:[0,0,1,0]}],
  ['color extent',{...fog(),color:[1,2,3,4]}],
  ['negative color',{...fog(),color:[-1,0,0]}],
  ['non-finite color',{...fog(),color:[NaN,0,0]}],
  ['equal f32 edges',{...fog(),near:1,far:1+Number.EPSILON}],
  ['negative density',{type:'exp2',color:[0,0,0],density:-1}],
  ['non-record',[]],
]) test(`camera snapshot refuses ${name} through the native admission contract`,()=>{
  assert.throws(()=>snapshot(input,I()),error=>error.code?.startsWith('ANIMATION_FOG_'));
});
