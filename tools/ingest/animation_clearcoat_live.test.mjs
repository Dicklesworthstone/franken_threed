import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createGpuAnimationRenderer, snapshotAnimationClearcoat} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const binding=()=>({view:{},sampler:{}});
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),lighting:{cameraPosition:[0,0,3],lights:[]},draws});
function geometry(d){return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),whenIdle:async()=>{},
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'}]}};}
async function setup(options={},material={}){
  const d=geometryDevice();d.limits.maxBindingsPerBindGroup=1000;
  const r=await createGpuAnimationRenderer(d,{maxDraws:8,...options}),g=geometry(d);
  const m=await r.addMesh(g,{shading:'metallic-roughness',mutableClearcoat:true,...material});
  return {d,r,g,m};
}
const coats=d=>d.buffers.filter(b=>b.label?.endsWith('/clearcoat'));
function packet(draw){const buffer=draw.groups.get(1).group.entries.find(e=>e.binding===16).resource.buffer;
  return [...new Float32Array(draw.contents.get(buffer).buffer)];}
const state=d=>[d.writes.length,d.submissions.length,d.buffers.length,d.pipelines.length];

test('private clearcoat snapshots preserve scalar compatibility and independent XY',()=>{
  assert.deepEqual([...snapshotAnimationClearcoat({},false)],[0,0,1,1]);
  const source=[-2,0],a=snapshotAnimationClearcoat({clearcoatFactor:.5,clearcoatNormalScale:source},true);
  source[0]=77;assert.deepEqual([...a],[.5,0,-2,0]);
  const b=snapshotAnimationClearcoat({clearcoatRoughnessFactor:.25},true,a);
  assert.deepEqual([...b],[.5,.25,-2,0]);assert.deepEqual([...a],[.5,0,-2,0]);
  assert.deepEqual([...snapshotAnimationClearcoat({clearcoatNormalScale:-3},true)],[0,0,-3,-3]);
  assert.deepEqual([...snapshotAnimationClearcoat({clearcoatNormalScale:new Float64Array([.5,-.25])},true)],[0,0,.5,-.25]);
});

test('clearcoat snapshot rejects invalid values, symbols, accessors and unsafe storage without invoking getters',()=>{
  for(const bad of [null,[],0,{unknown:1},{[Symbol()]:1}])assert.throws(()=>snapshotAnimationClearcoat(bad,true),{code:'ANIMATION_RENDER_OPTIONS'});
  for(const bad of [-1,1.01,NaN,Infinity,'0',null,1e100])for(const key of ['clearcoatFactor','clearcoatRoughnessFactor'])
    assert.throws(()=>snapshotAnimationClearcoat({[key]:bad},true),{code:'ANIMATION_RENDER_VALUE'});
  for(const bad of [[],[1],[1,2,3],[1,NaN],['1',1],null,{},new Uint8Array(2)])
    assert.throws(()=>snapshotAnimationClearcoat({clearcoatNormalScale:bad},true));
  const object={};Object.defineProperty(object,'clearcoatFactor',{get(){assert.fail('getter invoked');}});
  assert.throws(()=>snapshotAnimationClearcoat(object,true),{code:'ANIMATION_RENDER_OPTIONS'});
  const pair=[0,0];Object.defineProperty(pair,0,{get(){assert.fail('component getter invoked');}});
  assert.throws(()=>snapshotAnimationClearcoat({clearcoatNormalScale:pair},true),{code:'ANIMATION_RENDER_OPTIONS'});
  for(const buffer of [new SharedArrayBuffer(8),new ArrayBuffer(8,{maxByteLength:16})])
    assert.throws(()=>snapshotAnimationClearcoat({clearcoatNormalScale:new Float32Array(buffer)},true),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.throws(()=>snapshotAnimationClearcoat({clearcoatNormalScale:1},false),{code:'ANIMATION_RENDER_OPTIONS'});
});

test('mutable material uploads only its 16 bytes; previously submitted uses keep old values',async()=>{
  const {d,r,m}=await setup();const b=coats(d)[0];assert.equal(b.usage,64|8);
  r.render(frame([{mesh:m}]));const before=state(d),bytes=r.allocatedBytes;
  assert.equal(m.setClearcoat({clearcoatFactor:.75,clearcoatRoughnessFactor:.25}),m);
  assert.deepEqual(state(d),[before[0]+1,...before.slice(1)]);assert.equal(r.version,1);assert.equal(r.allocatedBytes,bytes);
  const write=d.writes.at(-1);assert.equal(write.buffer,b);assert.equal(write.input.length,16);
  r.render(frame([{mesh:m}]));assert.deepEqual(packet(d.snapshots[0][0]),[0,0,1,1]);
  assert.deepEqual(packet(d.snapshots[1][0]),[.75,.25,1,1]);await r.whenIdle();r.dispose();assert.equal(b.destroyed,true);
});

test('partial updates, normal pairs, signed zero and f32-equal no-ops are deterministic',async()=>{
  const {d,r,m}=await setup({}, {clearcoatNormalTexture:binding(),texCoords:[0,0,1,0,0,1]});
  const initial=state(d);m.setClearcoat({});m.setClearcoat({clearcoatFactor:undefined});assert.deepEqual(state(d),initial);
  m.setClearcoat({clearcoatFactor:.3,clearcoatNormalScale:[2,-1]});const writes=d.writes.length;
  m.setClearcoat({clearcoatFactor:Math.fround(.3)});assert.equal(d.writes.length,writes);
  m.setClearcoat({clearcoatRoughnessFactor:.5});m.setClearcoat({clearcoatNormalScale:[-0,1]});
  assert.ok(Object.is(new Float32Array(coats(d)[0].data)[2],-0));
  m.setClearcoat({clearcoatNormalScale:[0,1]});assert.equal(d.writes.length,writes+3);
  const code=d.pipelines.find(p=>p.fragment?.module.code.includes('clearcoat_info')).fragment.module.code;
  assert.match(code,/mapped.xy \* clearcoat_info.zw/);await r.whenIdle();r.dispose();
});

test('host-invalid live values have no native effects and corrected updates remain usable',async()=>{
  const {d,r,m}=await setup();const before=state(d);
  for(const bad of [null,{}, {clearcoatFactor:2},{clearcoatNormalScale:[1,1]},{baseColor:[1,1,1,1]}]){
    if(bad&&Object.keys(bad).length===0)continue;
    assert.throws(()=>m.setClearcoat(bad));assert.deepEqual(state(d),before);assert.equal(r.failed,false);
  }
  const getter={get clearcoatFactor(){assert.fail('getter invoked');}};
  assert.throws(()=>m.setClearcoat(getter));assert.deepEqual(state(d),before);
  m.setClearcoat({clearcoatFactor:.5});r.render(frame([{mesh:m}]));await r.whenIdle();r.dispose();
});

test('immutable handles stay immutable and registration rejects wrong profiles/options before allocations',async()=>{
  const {d,r,g,m}=await setup({}, {mutableClearcoat:false,clearcoatFactor:.5});
  assert.equal(coats(d)[0].usage,64);assert.throws(()=>m.setClearcoat({clearcoatFactor:0}),{code:'ANIMATION_RENDER_OPTIONS'});
  const before=state(d);
  for(const options of [{mutableClearcoat:1},{mutableClearcoat:null},{mutableClearcoat:true,shading:'unlit'}])
    await assert.rejects(r.addMesh(g,options),{code:'ANIMATION_RENDER_OPTIONS'});
  const options={shading:'metallic-roughness'};Object.defineProperty(options,'clearcoatFactor',{get(){assert.fail('registration getter invoked');}});
  await assert.rejects(r.addMesh(g,options),{code:'ANIMATION_RENDER_OPTIONS'});assert.deepEqual(state(d),before);r.dispose();
});

for(const concurrent of [false,true])test(`mutable coats never alias under generated instancing (concurrent=${concurrent})`,async()=>{
  const {d,r,g,m}=await setup({instancing:true});const options={shading:'metallic-roughness',mutableClearcoat:true};
  const [a,b]=concurrent?await Promise.all([r.addMesh(g,options),r.addMesh(g,options)]):[await r.addMesh(g,options),await r.addMesh(g,options)];
  assert.equal(coats(d).filter(b=>!b.destroyed).length,3);
  m.setClearcoat({clearcoatFactor:.25});a.setClearcoat({clearcoatFactor:.5});b.setClearcoat({clearcoatFactor:.75});
  r.render(frame([{mesh:m},{mesh:a},{mesh:b}]));assert.equal(r.drawCallCount,3);
  assert.deepEqual(d.snapshots.at(-1).map(s=>packet(s)[0]),[.25,.5,.75]);
  const bytes=r.allocatedBytes;a.dispose();assert.equal(r.allocatedBytes,bytes-16);
  b.setClearcoat({clearcoatFactor:1});r.render(frame([{mesh:b},{mesh:m}]));assert.deepEqual(d.snapshots.at(-1).map(s=>packet(s)[0]),[1,.25]);
  await r.whenIdle();r.dispose();assert.ok(coats(d).every(b=>b.destroyed));
});

test('identical immutable coats still deduplicate with instancing',async()=>{
  const options={shading:'metallic-roughness',clearcoatFactor:.5,mutableClearcoat:false};
  const {d,r,g,m}=await setup({instancing:true},options),a=await r.addMesh(g,options);
  assert.equal(coats(d).filter(b=>!b.destroyed).length,1);r.render(frame([{mesh:m},{mesh:a}]));assert.equal(r.drawCallCount,1);
  m.dispose();assert.equal(coats(d)[0].destroyed,false);a.dispose();assert.equal(coats(d)[0].destroyed,true);r.dispose();
});

test('cached render bundles consume live coat contents without rebuilds or extra geometry allocation',async()=>{
  const {d,r,m}=await setup({renderBundles:true});const f=frame([{mesh:m}]);r.render(f);
  const bundles=d.bundleEncoders.length,allocations=d.buffers.length;
  m.setClearcoat({clearcoatFactor:1});r.render(f);
  assert.equal(d.bundleEncoders.length,bundles);assert.equal(d.buffers.length,allocations);
  assert.deepEqual(packet(d.snapshots.at(-1)[0]),[1,0,1,1]);await r.whenIdle();r.dispose();
});

test('mutable coat bytes are charged separately and recovered on disposal',async()=>{
  const {r,g,m}=await setup({instancing:true});const used=r.allocatedBytes;
  const a=await r.addMesh(g,{shading:'metallic-roughness',mutableClearcoat:true});assert.equal(r.allocatedBytes,used+16);
  a.dispose();assert.equal(r.allocatedBytes,used);m.dispose();assert.equal(r.allocatedBytes,used-16);r.dispose();
});

test('disposed handles/renderers and reentrant updates fail without writes',async()=>{
  const {d,r,m}=await setup();let calls=0;const original=d.queue.writeBuffer;
  d.queue.writeBuffer=(...args)=>{calls++;assert.throws(()=>m.setClearcoat({clearcoatFactor:.1}),{code:'ANIMATION_RENDER_REENTRANT'});
    assert.throws(()=>m.dispose(),{code:'ANIMATION_RENDER_REENTRANT'});return original(...args);};
  m.setClearcoat({clearcoatFactor:.5});assert.equal(calls,1);await r.whenIdle();m.dispose();
  assert.throws(()=>m.setClearcoat({}),{code:'ANIMATION_RENDER_DISPOSED'});r.dispose();assert.throws(()=>m.setClearcoat({}),{code:'ANIMATION_RENDER_DISPOSED'});
});

test('whenIdle drains a material upload even with no following render',async()=>{
  const {d,r,m}=await setup();let finish;d.completion=new Promise(resolve=>{finish=resolve;});
  m.setClearcoat({clearcoatFactor:1});let done=false;const idle=r.whenIdle().then(()=>{done=true;});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(done,false);assert.equal(r.version,0);
  finish();await idle;assert.equal(done,true);r.dispose();
});

test('synchronous upload failure is terminal and visible from whenIdle',async()=>{
  const {d,r,m}=await setup({renderBundles:true});r.render(frame([{mesh:m}]));
  d.writeError=new Error('native write failed');assert.throws(()=>m.setClearcoat({clearcoatFactor:1}),/native write failed/);
  assert.equal(r.failed,true);assert.equal(d.scopes.length,0);await assert.rejects(r.whenIdle(),/native write failed/);r.dispose();
});

test('asynchronous scope failure is terminal, including a subsequent no-op update',async()=>{
  const {d,r,m}=await setup();d.scopeError={message:'invalid material upload'};m.setClearcoat({clearcoatFactor:1});
  await assert.rejects(r.whenIdle(),/invalid material upload/);assert.equal(r.failed,true);
  assert.throws(()=>m.setClearcoat({clearcoatFactor:1}),/invalid material upload/);r.dispose();
});

test('completion retrieval failure after writes is terminal',async()=>{
  const {d,r,m}=await setup();d.queue.onSubmittedWorkDone=()=>{throw new Error('completion failed');};
  assert.throws(()=>m.setClearcoat({clearcoatFactor:1}),/completion failed/);assert.equal(r.failed,true);
  await assert.rejects(r.whenIdle(),/completion failed/);r.dispose();
});

test('device loss interrupts pending uploads and retires owned coating buffers',async()=>{
  const {d,r,m}=await setup();d.completion=new Promise(()=>{});m.setClearcoat({clearcoatFactor:1});const idle=r.whenIdle();
  d.lose();await assert.rejects(idle,{code:'ANIMATION_RENDER_LOST'});assert.ok(coats(d).every(b=>b.destroyed));
  assert.throws(()=>m.setClearcoat({}),{code:'ANIMATION_RENDER_LOST'});r.dispose();
});
