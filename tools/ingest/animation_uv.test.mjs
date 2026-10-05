import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuBufferGeometry, createGpuInstanceAttributes, bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {snapshotAnimationUvMatrix, snapshotAnimationMapTransforms, animationMapChannelKey,
  packAnimationMapTransforms} from './animation_uv.mjs';
const root = process.env.F3D_THREE_ROOT ?? fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));
const entry = path.join(root, 'build/three.core.js');
const T = fs.existsSync(entry) ? await import(pathToFileURL(entry).href) : null;
const native = {skip: T ? false : 'Set F3D_THREE_ROOT to pinned r186'};
const I = [1,0,0,1,0,0], M = [2,3,4,5,.25,.5];
const world = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const texture = () => ({view:{}, sampler:{}});
function fixture(channels = [0,1,2,3]) {
  assert.equal(T.REVISION, '186');
  const d=geometryDevice(); d.limits.maxBindingsPerBindGroup=32;
  const g=new T.BufferGeometry();
  g.setAttribute('position', new T.Float32BufferAttribute([-1,-1,0,1,-1,0,0,1,0],3));
  g.setAttribute('normal', new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));
  for (const c of channels) g.setAttribute(c ? 'uv'+c : 'uv', new T.Float32BufferAttribute([0,0,1,0,0,1],2));
  const gpu=createGpuBufferGeometry(d,g); return {d,g,gpu};
}
const frame = draws => ({colorView:{},depthView:{},viewProjection:world,draws,
  lighting:{cameraPosition:[0,0,3],lights:[]}});
function words(draw, byteOffset=0, count=128) {
  const {group, offsets}=draw.groups.get(0), entry=group.entries.find(x=>x.binding===0);
  return new Float32Array(draw.contents.get(entry.resource.buffer).buffer,(entry.resource.offset??0)+(offsets[0]??0)+byteOffset,count);
}
const draws = d => d.snapshots.at(-1);
const matrixWords = m => [m[0],m[2],m[4],0,m[1],m[3],m[5],0];

test('UV matrices snapshot finite fixed arrays and never retain caller coefficient storage',()=>{
  const v=[...M],a=snapshotAnimationUvMatrix(v);v[0]=9;assert.equal(a[0],2);assert.ok(Object.isFrozen(a));
  for (const value of [null, [1,2], [1,0,0,1,NaN,0], [1,0,0,1,1e100,0], ['1',0,0,1,0,0], new Int32Array(6)])
    assert.throws(()=>snapshotAnimationUvMatrix(value),{code:'ANIMATION_UV_INPUT'});
  assert.throws(()=>snapshotAnimationUvMatrix(new Float32Array(new SharedArrayBuffer(24))),{code:'ANIMATION_UV_INPUT'});
});
test('map slots validate active maps, reject accessors, and preserve independent default/override packets',()=>{
  const fields=['baseColorTexture','metallicRoughnessTexture','normalTexture'];
  let calls=0;const accessor={get normalTexture(){calls++;return M;}};
  assert.throws(()=>snapshotAnimationMapTransforms(accessor,fields,5),{code:'ANIMATION_UV_INPUT'});assert.equal(calls,0);
  for (const input of [null,[],{unknown:M},{metallicRoughnessTexture:M}])
    assert.throws(()=>snapshotAnimationMapTransforms(input,fields,5),{code:'ANIMATION_UV_INPUT'});
  const base=snapshotAnimationMapTransforms({baseColorTexture:M},fields,5),override=snapshotAnimationMapTransforms({normalTexture:[...I.slice(0,4),.75,0]},fields,5);
  const out=new Float32Array(64).fill(99);packAnimationMapTransforms(base,override,I,out,0);
  assert.deepEqual([...out.slice(0,8)],matrixWords(M));assert.deepEqual([...out.slice(8,16)],matrixWords(I));assert.equal(out[18],.75);
  assert.throws(()=>packAnimationMapTransforms(base,override,I,out,1),{code:'ANIMATION_UV_INPUT'});
  assert.equal(animationMapChannelKey({baseColorTexture:1,normalTexture:3},fields,5,{uv1:true,uv3:true}),49);
  for (const c of [-1,4,1.5,'1']) assert.throws(()=>animationMapChannelKey({baseColorTexture:c},fields,1,{uv1:true}),{code:'ANIMATION_UV_INPUT'});
});
for (const instancing of [false,true]) for (const renderBundles of [false,true])
test(`per-use transforms remain independent and live (${instancing}/${renderBundles})`,native,async()=>{
  const h=fixture(), r=await createGpuAnimationRenderer(h.d,{textureTransforms:true,instancing,renderBundles,maxDraws:2});
  try {
    const registration=[...M],m=await r.addMesh(h.gpu,{baseColorTexture:texture(),mapTransforms:{baseColorTexture:registration}});
    registration[0]=99;
    const a={mesh:m,mapTransforms:{baseColorTexture:I}},b={mesh:m};
    r.render(frame([a,b]));const snapshot=draws(h.d),first=words(snapshot[0]);
    assert.deepEqual([...first.slice(64,72)],matrixWords(I));
    const last=instancing?words(snapshot[0],512):words(snapshot[1]);assert.deepEqual([...last.slice(64,72)],matrixWords(M));
    const builds=h.d.bundleEncoders.length,allocations=h.d.buffers.length,pipelines=h.d.pipelines.length;
    a.mapTransforms.baseColorTexture=[1,0,0,1,.75,0];r.render(frame([a,b]));
    assert.equal(words(draws(h.d)[0])[66],.75);assert.equal(first[66],0);
    assert.equal(h.d.buffers.length,allocations);assert.equal(h.d.pipelines.length,pipelines);
    if(renderBundles)assert.equal(h.d.bundleEncoders.length,builds);
    assert.match(draws(h.d)[0].pipeline.vertex.module.code,/out\.uv_0 = .*draw_info\.map_uv\[0\]/);
    await r.whenIdle();
  } finally {r.dispose();h.gpu.dispose();}
  assert.ok(h.d.buffers.every(b=>b.destroyed));
});
for(const channel of [0,1,2,3])
test(`selected uv${channel} streams bind natively without CPU repacking`,native,async()=>{
  const h=fixture([channel]),r=await createGpuAnimationRenderer(h.d,{textureTransforms:true,maxDraws:1});
  try {
    const m=await r.addMesh(h.gpu,{baseColorTexture:texture(),mapChannels:{baseColorTexture:channel}});r.render(frame([m]));
    const draw=draws(h.d)[0],location=channel?9+channel:3;
    const slot=draw.pipeline.vertex.buffers.findIndex(b=>b.attributes.some(a=>a.shaderLocation===location));assert.ok(slot>=0);
    assert.deepEqual([...new Float32Array(draw.contents.get(draw.streams.get(slot)).buffer)],[0,0,1,0,0,1]);
    assert.match(draw.pipeline.vertex.module.code,new RegExp(`vec3<f32>\\(uv${channel||''}, 1.0\\)`));
    const allocations=h.d.buffers.length,a=h.g.attributes[channel?'uv'+channel:'uv'];a.array[0]=.25;a.needsUpdate=true;h.gpu.update();r.render(frame([m]));
    assert.equal(h.d.buffers.length,allocations);
  } finally {r.dispose();h.gpu.dispose();}
});
test('different map channels form distinct pipelines and normal derivatives follow normal-map UVs',native,async()=>{
  const h=fixture(),r=await createGpuAnimationRenderer(h.d,{textureTransforms:true,maxDraws:2});
  try {
    const opts={shading:'lambert',baseColorTexture:texture(),normalTexture:texture(),occlusionTexture:texture()};
    const a=await r.addMesh(h.gpu,{...opts,mapChannels:{baseColorTexture:1,normalTexture:2,occlusionTexture:3}});
    const b=await r.addMesh(h.gpu,{...opts,mapChannels:{baseColorTexture:3,normalTexture:1,occlusionTexture:2}});
    r.render(frame([a,b]));const [x,y]=draws(h.d);assert.notEqual(x.pipeline,y.pipeline);
    assert.match(x.pipeline.vertex.module.code,/out\.uv_2 = .*vec3<f32>\(uv2, 1.0\)/);
    assert.match(x.pipeline.fragment.module.code,/let uv_dx = dpdx\(input\.uv_2\)/);
    assert.match(x.pipeline.fragment.module.code,/textureSample\(occlusion_texture, occlusion_sampler, input\.uv_4\)/);
  } finally {r.dispose();h.gpu.dispose();}
});
for(const instancing of [false,true])
test(`native instance locations do not collide with extra UV streams (${instancing})`,native,async()=>{
  const h=fixture(),object=new T.InstancedMesh(h.g,new T.MeshBasicMaterial(),2);
  object.setColorAt(0,new T.Color(1,1,1));const instances=createGpuInstanceAttributes(h.d,object);
  const r=await createGpuAnimationRenderer(h.d,{textureTransforms:true,instancing,maxDraws:1});
  try {
    const m=await r.addMesh(h.gpu,{instances,baseColorTexture:texture(),mapChannels:{baseColorTexture:3}});r.render(frame([m]));
    const draw=draws(h.d)[0],locations=draw.pipeline.vertex.buffers.flatMap(b=>b.attributes.map(a=>a.shaderLocation));
    assert.equal(new Set(locations).size,locations.length);assert.equal(draw.args[1],2);assert.ok(locations.includes(9)&&locations.includes(12));
    assert.equal(draw.groups.get(0).group.layout.entries[0].buffer.type,'uniform');
    assert.match(draw.pipeline.vertex.module.code,/@location\(12\) uv3: vec2<f32>/);
  } finally {r.dispose();instances.dispose();h.gpu.dispose();}
});
test('clipping and per-map transforms occupy separate bounded tails in depth-only draws',native,async()=>{
  const h=fixture(),r=await createGpuAnimationRenderer(h.d,{format:null,textureTransforms:true,clipping:true,maxClippingPlanes:2,maxDraws:1});
  try {
    const m=await r.addMesh(h.gpu,{baseColorTexture:texture(),alphaMode:'MASK',mapChannels:{baseColorTexture:1}});
    r.render({depthView:{},viewProjection:world,draws:[{mesh:m,clippingPlanes:[[1,0,0,.25]],mapTransforms:{baseColorTexture:M}}]});
    const draw=draws(h.d)[0],w=words(draw,0,140);
    assert.deepEqual([...w.slice(64,72)],[1,1,0,0,1,0,0,.25]);assert.deepEqual([...w.slice(76,84)],matrixWords(M));
    assert.equal(draw.pipeline.fragment.targets.length,0);
    const code=draw.pipeline.fragment.module.code;assert.ok(code.indexOf('textureSample(')<code.indexOf('discard;'));
    assert.match(code,/animation_clipped\(input.world\)/);
  } finally {r.dispose();h.gpu.dispose();}
});
test('invalid later transforms reject before queue effects and allow a corrected retry',native,async()=>{
  const h=fixture(),r=await createGpuAnimationRenderer(h.d,{textureTransforms:true,maxDraws:2});
  try {
    const m=await r.addMesh(h.gpu,{baseColorTexture:texture()}),a={mesh:m,mapTransforms:{baseColorTexture:I}},b={mesh:m,mapTransforms:{baseColorTexture:[1,0,0,1,NaN,0]}};
    const before=[h.d.writes.length,h.d.submissions.length];assert.throws(()=>r.render(frame([a,b])),{code:'ANIMATION_UV_INPUT'});
    assert.deepEqual([h.d.writes.length,h.d.submissions.length],before);assert.equal(r.failed,false);
    b.mapTransforms.baseColorTexture=M;r.render(frame([a,b]));assert.equal(r.version,1);
  } finally {r.dispose();h.gpu.dispose();}
});
test('missing channels and disabled profile refuse before registration publishes resources',native,async()=>{
  const h=fixture([0]);
  for(const textureTransforms of [false,true]) {
    const r=await createGpuAnimationRenderer(h.d,{textureTransforms,maxDraws:1}),count=h.d.buffers.length;
    await assert.rejects(r.addMesh(h.gpu,{baseColorTexture:texture(),mapChannels:{baseColorTexture:2}}));
    assert.equal(h.d.buffers.length,count);assert.equal(r.meshCount,0);r.dispose();
  }
  h.gpu.dispose();
});
test('expanded packet budget and GPU binding limits are enforced',native,async()=>{
  const h=fixture();h.d.limits.maxUniformBufferBindingSize=400;
  await assert.rejects(createGpuAnimationRenderer(h.d,{textureTransforms:true,maxDraws:1}),{code:'ANIMATION_RENDER_LIMIT'});
  h.d.limits.maxUniformBufferBindingSize=65536;
  await assert.rejects(createGpuAnimationRenderer(h.d,{textureTransforms:true,maxDraws:1,maxBytes:511}),{code:'ANIMATION_RENDER_LIMIT'});
  await assert.rejects(createGpuAnimationRenderer(h.d,{textureTransforms:1}),{code:'ANIMATION_RENDER_OPTIONS'});h.gpu.dispose();
});
test('interleaved secondary UV streams share one versioned native allocation',native,()=>{
  const h=fixture([]),data=new T.InterleavedBuffer(new Float32Array([0,0,1,1, 1,0,0,1, 0,1,1,0]),4);
  h.g.setAttribute('uv1',new T.InterleavedBufferAttribute(data,2,0));h.g.setAttribute('uv2',new T.InterleavedBufferAttribute(data,2,2));
  h.gpu.update();const shape=bufferGeometrySnapshot(h.gpu,h.d),layout=shape.layouts.find(b=>b.attributes.some(a=>a.shaderLocation===10));
  assert.deepEqual(layout.attributes.map(a=>a.shaderLocation),[10,11]);assert.equal(layout.arrayStride,16);assert.equal(shape.channels.uv1,true);h.gpu.dispose();
});
