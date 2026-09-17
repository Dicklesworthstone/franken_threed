import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer,AnimationPoseError} from './animation_runtime.mjs';

const make=(nodes,clips,extra={})=>createAnimationPlayer({format:'f3d-animation-v1',nodes,clips,...extra});
const track=(node,path,value)=>({node,path,times:[0,1],values:[...value,...value]});
const clip=(...channels)=>({channels});
const close=(actual,expected,epsilon=1e-11)=>{
  assert.equal(actual.length,expected.length);
  for(let i=0;i<actual.length;i++)assert.ok(Math.abs(actual[i]-expected[i])<=epsilon,`${i}: ${actual[i]} != ${expected[i]}`);
};
const qz=angle=>[0,0,Math.sin(angle/2),Math.cos(angle/2)];
const snapshot=p=>({version:p.version,time:p.time,clip:p.clip,mode:p.mode,
  ...Object.fromEntries(['translations','rotations','scales','morphWeights','worldMatrices','jointMatrices'].map(k=>[k,[...p[k]]]))});

for(const [a,b,expected] of [[0,0,10],[0.25,0.25,20],[0.25,0.75,35],[1,3,35],[2,2,30]]) {
  test(`normal weights ${a}/${b} normalize per binding with rest residual`,()=>{
    const p=make([{translation:[10,0,0],scale:[2,2,2]}],[
      clip(track(0,'translation',[20,0,0])),clip(track(0,'translation',[40,0,0])),clip(track(0,'scale',[4,4,4])),
    ]);
    p.blend([{clip:0,time:0,weight:a},{clip:1,time:0,weight:b},{clip:2,time:0,weight:1}]);
    close(p.translations,[expected,0,0]);close(p.scales,[4,4,4]);
    assert.equal(p.mode,'blend');assert.equal(p.clip,-1);assert.equal(p.time,0);
  });
}

test('unrelated targets do not steal weight; untargeted values return to rest',()=>{
  const p=make([{translation:[2,3,4],weights:[0.2,0.8]},{}],[
    clip(track(0,'translation',[10,0,0])),clip(track(1,'translation',[0,6,0])),
  ]);
  p.blend([{clip:0,time:0},{clip:1,time:0}]);close(p.translations,[10,0,0,0,6,0]);
  p.blend([{clip:1,time:0}]);close(p.translations,[2,3,4,0,6,0]);close(p.morphWeights,[0.2,0.8]);
  p.blend([]);close(p.translations,[2,3,4,0,0,0]);
});

test('fractional node masks support independent upper/lower body contributions',()=>{
  const p=make([{},{}],[clip(track(0,'translation',[8,0,0]),track(1,'translation',[0,8,0]))]);
  const mask=new Float32Array([1,0.25]);
  p.blend([{clip:0,time:0,mask}]);close(p.translations,[8,0,0,0,2,0]);
  mask.set([0,1]);p.blend([{clip:0,time:0,mask}]);close(p.translations,[0,0,0,0,8,0]);
  // Removing a previously used mask must not retain its scratch values.
  p.blend([{clip:0,time:0}]);close(p.translations,[8,0,0,0,8,0]);
});

test('mask snapshots may alias published pose arrays',()=>{
  const p=make([{weights:[0.25,0.75]},{}],[clip(track(0,'translation',[8,0,0]),track(1,'translation',[8,0,0]))]);
  p.blend([{clip:0,time:0,mask:p.morphWeights}]);close(p.translations,[2,0,0,6,0,0]);
});

test('rotations blend through SLERP including rest weight and antipodal keys',()=>{
  const p=make([{}],[clip(track(0,'rotation',qz(Math.PI))),clip(track(0,'rotation',qz(Math.PI).map(x=>-x)))]);
  p.blend([{clip:0,time:0,weight:0.5}]);close(p.rotations,qz(Math.PI/2));
  p.blend([{clip:0,time:0,weight:0.5},{clip:1,time:0,weight:0.5}]);close(p.rotations,qz(Math.PI));
  assert.ok(Math.abs(Math.hypot(...p.rotations)-1)<1e-15);
});

test('local rotation mixing precedes hierarchy and skin evaluation, never matrix averaging',()=>{
  const p=make([{}, {parent:0,translation:[1,0,0]}, {}, {translation:[3,0,0]}],[
    clip(track(0,'rotation',qz(0))),clip(track(0,'rotation',qz(Math.PI))),
  ],{skins:[{joints:[1]}],instances:[{node:2,skin:0},{node:3,skin:0}]});
  p.blend([{clip:0,time:0,weight:0.5},{clip:1,time:0,weight:0.5}]);
  close(p.worldMatrices.subarray(28,31),[0,1,0]);
  close(p.jointMatrices.subarray(12,15),[0,1,0],1e-7);
  close(p.jointMatrices.subarray(28,31),[-3,1,0],1e-7);
  const m=p.jointMatrices;
  close([m[0]+m[12],m[1]+m[13],m[2]+m[14]],[0,2,0],1e-7);
});

test('each layer samples its own time, interpolation and loop state',()=>{
  const p=make([{}],[
    clip({node:0,path:'translation',times:[0,2],values:[0,0,0,2,4,6]}),
    clip({node:0,path:'translation',times:[0,2],values:[8,8,8,16,16,16],interpolation:'STEP'}),
  ]);
  p.blend([{clip:0,time:-0.5,loop:true,weight:0.5},{clip:1,time:2,weight:0.5}]);
  close(p.translations,[8.75,9.5,10.25]);
  p.blend([{clip:0,time:0.1,loop:true}]);close(p.translations,[0.1,0.2,0.3]);
});

test('cubic morph channels retain their widths while blending',()=>{
  const p=make([{weights:[0.2,0.8]}],[clip({node:0,path:'weights',times:[0,2],interpolation:'CUBICSPLINE',
    values:[0,0,0,1,0,0,0,0,1,0,0,0]})]);
  p.blend([{clip:0,time:1,weight:0.5}]);close(p.morphWeights,[0.35,0.65]);
});

test('additive channels are relative to imported rest, including scale and rotations',()=>{
  const p=make([{translation:[10,0,0],scale:[2,2,2],rotation:qz(Math.PI/2),weights:[0.2]}],[
    clip(track(0,'translation',[14,0,0]),track(0,'scale',[4,4,4]),track(0,'rotation',qz(Math.PI)),track(0,'weights',[0.6])),
  ]);
  p.blend([{clip:0,time:0,weight:0.5,mode:'additive'}]);
  close(p.translations,[12,0,0]);close(p.scales,[3,3,3]);close(p.morphWeights,[0.4]);close(p.rotations,qz(3*Math.PI/4));
  p.blend([{clip:0,time:0,weight:0,mode:'additive'}]);close(p.rotations,qz(Math.PI/2));
});

test('normal accumulation completes before additive layers regardless of interleaving',()=>{
  const p=make([{translation:[10,0,0]}],[clip(track(0,'translation',[14,0,0])),clip(track(0,'translation',[30,0,0]))]);
  p.blend([{clip:0,time:0,mode:'additive',weight:0.5},{clip:1,time:0}]);close(p.translations,[32,0,0]);
  p.blend([{clip:1,time:0},{clip:0,time:0,mode:'additive',weight:0.5}]);close(p.translations,[32,0,0]);
});

test('additive quaternion layers preserve noncommutative input order',()=>{
  const h=Math.SQRT1_2;
  const p=make([{}],[clip(track(0,'rotation',[h,0,0,h])),clip(track(0,'rotation',[0,h,0,h]))]);
  p.blend([{clip:0,time:0,mode:'additive'},{clip:1,time:0,mode:'additive'}]);close(p.rotations,[0.5,0.5,0.5,0.5]);
  p.blend([{clip:1,time:0,mode:'additive'},{clip:0,time:0,mode:'additive'}]);close(p.rotations,[0.5,0.5,-0.5,0.5]);
});

test('reused output storage, independent actors and switching back to sample/reset',()=>{
  const d={format:'f3d-animation-v1',nodes:[{}],clips:[clip(track(0,'translation',[4,0,0]))]};
  const p=createAnimationPlayer(d),other=createAnimationPlayer(d),out=p.worldMatrices;
  const layers=[{clip:0,time:0,weight:0}];
  for(let i=0;i<1000;i++){layers[0].weight=(i%101)/100;p.blend(layers);assert.equal(p.worldMatrices,out);}
  close(other.translations,[0,0,0]);p.sample(0);assert.equal(p.mode,'sample');close(p.translations,[4,0,0]);
  p.reset();assert.equal(p.mode,'rest');close(p.translations,[0,0,0]);
  p.dispose();assert.throws(()=>p.blend([]),{code:'ANIMATION_DISPOSED'});
});

test('root matrix applies once and cancels out of mesh-local skin palettes',()=>{
  const p=make([{},{}],[clip(track(1,'translation',[4,0,0]))],{skins:[{joints:[1]}],instances:[{node:0,skin:0}]});
  p.blend([{clip:0,time:0,weight:0.5}],{rootMatrix:[1,0,0,0,0,1,0,0,0,0,1,0,100,200,300,1]});
  close(p.worldMatrices.subarray(28,31),[102,200,300]);close(p.jointMatrices.subarray(12,15),[2,0,0]);
});

const invalidLayers=[null,{},new Array(257),[null],[{}],[{clip:99,time:0}],[{clip:0,time:NaN}],
  [{clip:0,time:0,weight:-1}],[{clip:0,time:0,weight:Infinity}],[{clip:0,time:0,mode:'bogus'}],
  [{clip:0,time:0,loop:1}],[{clip:0,time:0,mask:[]}],[{clip:0,time:0,mask:[2]}],
  [{clip:0,time:0,mask:[NaN]}],[{clip:0,time:0,mask:[-1]}],
  [{clip:0,time:0,weight:Number.MAX_VALUE},{clip:0,time:0,weight:Number.MAX_VALUE}],
];
for(const [i,layers]of invalidLayers.entries())test(`invalid blend ${i} leaves the entire published pose untouched`,()=>{
  const p=make([{}],[clip(track(0,'translation',[4,0,0]))]);p.sample(0.5);const before=snapshot(p);
  assert.throws(()=>p.blend(layers),AnimationPoseError);assert.deepEqual(snapshot(p),before);
  p.blend([{clip:0,time:0}]);close(p.translations,[4,0,0]);
});

test('a singular blended mesh or zero cubic rotation cannot publish a partial pose',()=>{
  const p=make([{},{}],[clip(track(0,'scale',[1,1,1])),clip(track(0,'scale',[-1,1,1]))],
    {skins:[{joints:[1]}],instances:[{node:0,skin:0}]});
  const before=snapshot(p);
  assert.throws(()=>p.blend([{clip:0,time:0,weight:0.5},{clip:1,time:0,weight:0.5}]),{code:'ANIMATION_SINGULAR_MESH'});
  assert.deepEqual(snapshot(p),before);
  const q=make([{}],[clip({node:0,path:'rotation',times:[0,2],interpolation:'CUBICSPLINE',
    values:[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0]})]);
  assert.throws(()=>q.blend([{clip:0,time:1}]),{code:'ANIMATION_QUATERNION'});
  q.blend([{clip:0,time:1,weight:0}]);close(q.rotations,[0,0,0,1]);
});

test('reentrant getters cannot change pose state during blending',()=>{
  const p=make([{}],[clip(track(0,'translation',[4,0,0]))]);const before=snapshot(p);
  assert.throws(()=>p.blend([{clip:0,get time(){p.sample(0);return 0;}}]),{code:'ANIMATION_REENTRANT'});
  assert.deepEqual(snapshot(p),before);
  assert.throws(()=>p.sample(0,{get clip(){p.dispose();return 0;}}),{code:'ANIMATION_REENTRANT'});
  assert.equal(p.disposed,false);p.blend([{clip:0,time:0}]);close(p.translations,[4,0,0]);
});

test('output storage detached by a layer getter is caught before publication',()=>{
  const p=make([{}],[clip(track(0,'translation',[4,0,0]))]);const t=p.translations.slice(),v=p.version;
  assert.throws(()=>p.blend([{clip:0,get time(){structuredClone(p.jointMatrices.buffer,{transfer:[p.jointMatrices.buffer]});return 0;}}]),
    {code:'ANIMATION_OUTPUT_STORAGE'});
  assert.deepEqual(p.translations,t);assert.equal(p.version,v);
});
