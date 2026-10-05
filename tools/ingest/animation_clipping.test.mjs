import assert from 'node:assert/strict';
import {test} from 'node:test';
import {animationClippingBytes, snapshotAnimationClipping, packAnimationClipping} from './animation_clipping.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const I=()=>[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),draws});
function geometry(d) {
  return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),whenIdle:async()=>{},
    vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'}, {shaderLocation:1,offset:12,format:'float32x3'}]}};
}
function packet(snapshot,slot=0,stride=512) {
  const {group,offsets}=snapshot.groups.get(0),entry=group.entries.find(x=>x.binding===0);
  const start=(offsets[0]??slot*stride)+(entry.resource.offset??0);
  return new Float32Array(snapshot.contents.get(entry.resource.buffer).buffer,start,100);
}
const plane=[1,0,0,0], opposite=[-1,0,0,0];

test('plane snapshots preserve sign, own arrays, validate bounds and the f32 domain',()=>{
  const source=[new Float64Array([2,0,-0,3])],copy=snapshotAnimationClipping(source);
  source[0][0]=7;assert.deepEqual(copy,[[2,0,-0,3]]);
  for(const bad of [null,{},[[]],[[0,0,0,1]],[[1e-100,0,0,0]],[[NaN,0,0,0]],[[1,0,0,Infinity]],[[1,0,0,1e100]],[[1,0,0,'0']]])
    assert.throws(()=>snapshotAnimationClipping(bad),{code:'ANIMATION_CLIPPING_INPUT'});
  for(const capacity of [0,-1,1.5,65,Infinity])assert.throws(()=>animationClippingBytes(capacity),{code:'ANIMATION_CLIPPING_INPUT'});
  assert.equal(animationClippingBytes(8),144);assert.equal(animationClippingBytes(64),1040);
  assert.throws(()=>snapshotAnimationClipping([plane,opposite],1),{code:'ANIMATION_CLIPPING_INPUT'});
});

test('pack combines global union with local union/intersection and clears removed planes',()=>{
  const words=new Float32Array(40).fill(99);
  packAnimationClipping([plane],[opposite],true,words,4,3);
  assert.deepEqual([...words.slice(4,20)],[2,1,0,0,...plane,...opposite,0,0,0,0]);
  packAnimationClipping([plane],[opposite],false,words,4,3);assert.equal(words[5],2);
  packAnimationClipping([],[],true,words,4,3);assert.ok(words.slice(4,20).every(v=>v===0));
  assert.equal(words[3],99);assert.equal(words[20],99);
  assert.throws(()=>packAnimationClipping([plane],[opposite],false,words,0,1),{code:'ANIMATION_CLIPPING_INPUT'});
  assert.throws(()=>packAnimationClipping([],[],0,words,0,3),{code:'ANIMATION_CLIPPING_INPUT'});
});

for(const instancing of [false,true])for(const renderBundles of [false,true])
  test(`per-use planes and live bundle edits (${instancing}/${renderBundles})`,async()=>{
    const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{clipping:true,instancing,renderBundles,maxDraws:2});
    const m=await r.addMesh(geometry(d));
    const first={mesh:m,clippingPlanes:[plane.slice()]},second={mesh:m,clippingPlanes:[opposite.slice()]};
    r.render(frame([first,second]));
    const snapshots=d.snapshots[0];
    assert.deepEqual([...packet(snapshots[0]).slice(64,72)],[1,1,0,0,...plane]);
    assert.deepEqual([...packet(snapshots.at(-1),instancing?1:0).slice(64,72)],[1,1,0,0,...opposite]);
    assert.equal(r.drawCallCount,instancing?1:2);
    first.clippingPlanes[0][3]=.25;second.clippingPlanes=[];
    r.render({...frame([first,second]),clippingPlanes:[[0,1,0,0]]});
    assert.deepEqual([...packet(d.snapshots[1][0]).slice(64,76)],[2,2,0,0,0,1,0,0,1,0,0,.25]);
    assert.deepEqual([...packet(d.snapshots[1].at(-1),instancing?1:0).slice(64,76)],[1,1,0,0,0,1,0,0,0,0,0,0]);
    assert.equal(packet(snapshots[0])[71],0,'previous submission owns its plane data');
    if(renderBundles)assert.equal(r.bundleDiagnostics.reuses,1);
    await r.whenIdle();r.dispose();assert.ok(d.buffers.filter(b=>b.label==='f3d-animation-draw').every(b=>b.destroyed));
  });

for(const shading of ['unlit','lambert','phong','toon','metallic-roughness'])
  test(`${shading}: world-space clipping follows current object transform without changing the clip matrix`,async()=>{
    const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{clipping:true,maxDraws:1});
    const m=await r.addMesh(geometry(d),{shading});const world=I();world[0]=-2;world[12]=3;
    r.render({...frame([{mesh:m,worldMatrix:world,clippingPlanes:[plane]}]),lighting:{viewDirection:[0,0,1],lights:[{type:'directional'}]}});
    const snapshot=d.snapshots[0][0],words=packet(snapshot);
    assert.deepEqual([...words.slice(0,16)],world);assert.deepEqual([...words.slice(32,48)],world);
    const code=snapshot.pipeline.fragment.module.code;
    assert.match(code,/out.world = \(draw_info.world_from_local/);
    assert.ok(code.lastIndexOf('if (animation_clipped(input.world))')>code.lastIndexOf('let rgb =')||shading!=='unlit');
    const body=code.slice(code.indexOf('@fragment fn fragment_main'));
    assert.ok(body.lastIndexOf('animation_clipped(input.world)')>body.lastIndexOf('illuminate('));
    r.dispose();
  });

test('depth-only MASK rendering gets the same clipping packet and discard as color',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{format:null,clipping:true,maxDraws:1});
  const m=await r.addMesh(geometry(d),{alphaMode:'MASK',alphaCutoff:.5,baseColor:[1,1,1,.8]});
  r.render({depthView:{},viewProjection:I(),draws:[{mesh:m,clippingPlanes:[plane],clipIntersection:true}]});
  const s=d.snapshots[0][0];assert.deepEqual(s.pipeline.fragment.targets,[]);
  assert.match(s.pipeline.fragment.module.code,/if \(animation_clipped\(input.world\)\) \{ discard; \}/);
  assert.deepEqual([...packet(s).slice(64,72)],[1,0,0,0,...plane]);r.dispose();
});

test('invalid planes in a later draw cause no queue writes or partial submission; corrected input can retry',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{clipping:true,maxClippingPlanes:1,maxDraws:2});
  const m=await r.addMesh(geometry(d)),before=d.writes.length;
  for(const bad of [{clippingPlanes:[[NaN,0,0,0]]},{clippingPlanes:[plane,opposite]},{clipIntersection:3}]) {
    assert.throws(()=>r.render(frame([m,{mesh:m,...bad}])),{code:'ANIMATION_CLIPPING_INPUT'});
    assert.equal(d.writes.length,before);assert.equal(d.submissions.length,0);assert.equal(r.failed,false);
  }
  assert.throws(()=>r.render({...frame([{mesh:m,clippingPlanes:[opposite]}]),clippingPlanes:[plane]}),{code:'ANIMATION_CLIPPING_INPUT'});
  assert.equal(d.writes.length,before);r.render(frame([{mesh:m,clippingPlanes:[plane]}]));assert.equal(r.version,1);r.dispose();
});

test('default-off keeps the original arena and rejects clipping instead of silently ignoring it',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{maxDraws:2});assert.equal(r.allocatedBytes,512);
  const m=await r.addMesh(geometry(d));const before=d.writes.length;
  assert.throws(()=>r.render({...frame([m]),clippingPlanes:[plane]}),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.throws(()=>r.render(frame([{mesh:m,clippingPlanes:[plane]}])),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.equal(d.writes.length,before);assert.ok(!d.pipelines[0].vertex.module.code.includes('clipping_meta'));
  r.render(frame([m]));await r.whenIdle();r.dispose();
});

test('expanded packet capacity and device/budget constraints fail before allocation',async()=>{
  for(const options of [{clipping:1},{clipping:true,maxClippingPlanes:0},{clipping:true,maxBytes:511,maxDraws:1}]) {
    const d=geometryDevice();await assert.rejects(createGpuAnimationRenderer(d,options));assert.equal(d.buffers.length,0);
  }
  const d=geometryDevice();d.limits.maxUniformBufferBindingSize=256;
  await assert.rejects(createGpuAnimationRenderer(d,{clipping:true}),{code:'ANIMATION_RENDER_LIMIT'});assert.equal(d.buffers.length,0);
});
