import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {buildAnimation} from './build_animation.mjs';
import {animationFixture} from './fixtures/animation/gltf_fixture.mjs';

// Real renderer, residency and package builder; the device boundary records
// commands and queue-ordered bytes. These are not native shader/pixel tests.
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const attribute=(array,itemSize)=>({array,itemSize,count:array.length/itemSize,version:0,
  updateRanges:[],onUploadCallback(){},clearUpdateRanges(){this.updateRanges.length=0;}});
function geometry() {
  return {attributes:{
    position:attribute(new Float32Array([0,0,0,1,0,0,0,1,0]),3),
    normal:attribute(new Float32Array([0,0,1,0,0,1,0,0,1]),3),
    uv:attribute(new Float32Array([0,0,1,0,0,1]),2),
    color:attribute(new Float32Array([1,0,0,1,0,0,1,0,0]),3),
  },index:attribute(new Uint16Array([0,1,2]),1),drawRange:{start:0,count:3}};
}
const frame=(draws,extra={})=>({colorView:{},depthView:{},viewProjection:I(),draws,...extra});
const uniforms=draw=>new Float32Array(draw.contents.get(draw.groups.get(0).group.entries[0].resource.buffer).buffer);
async function setup(options={}) {
  const device=geometryDevice(),source=geometry(),gpu=createGpuBufferGeometry(device,source);
  const renderer=await createGpuAnimationRenderer(device,{renderBundles:true,maxDraws:4,...options});
  return {device,source,gpu,renderer};
}

for(const instancing of [false,true])
test(`animated buffers, cameras and material factors stay live in reused bundles (instancing=${instancing})`,async()=>{
  const {device:d,source:g,gpu,renderer:r}=await setup({instancing});
  const a=await r.addMesh(gpu),b=await r.addMesh(gpu),buffers=d.buffers.length;
  for(let i=0;i<6;i++){
    g.attributes.position.array[0]=i;g.attributes.position.version++;
    g.attributes.color.array[0]=i/8;g.attributes.color.version++;
    g.index.array[0]=i%3;g.index.version++;gpu.update();
    const camera=I();camera[12]=i;
    const world=I();world[13]=i+1;
    r.render(frame([{mesh:a,baseColor:[i/8,0,0,1]}, {mesh:b,worldMatrix:world,baseColor:[0,i/8,0,1]}],{viewProjection:camera}));
  }
  // No await between submissions: earlier snapshots must not read later writes.
  await r.whenIdle();
  assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,5);
  assert.equal(d.encodedDrawCalls,instancing?1:2);assert.equal(r.drawCallCount,instancing?1:2);
  assert.equal(r.drawCount,2);assert.equal(d.buffers.length,buffers);
  for(let i=0;i<6;i++){
    const draw=d.snapshots[i][0],words=uniforms(draw);
    assert.equal(words[12],i);assert.equal(words[16],i/8);
    assert.equal(words[64+13],i+1);assert.equal(words[64+17],i/8);
    assert.equal(new Float32Array(draw.contents.get(draw.streams.get(0)).buffer)[0],i);
    assert.equal(new Uint16Array(draw.contents.get(draw.index.buffer).buffer)[0],i%3);
  }
  r.dispose();assert.equal(gpu.disposed,false);gpu.dispose();
});

test('live ranges and same-layout storage replacement rebuild; returning to a cached range reuses',async()=>{
  const {device:d,source:g,gpu,renderer:r}=await setup();const mesh=await r.addMesh(gpu);
  r.render(frame([mesh]));const initial=d.passes.at(-1).bundles[0];
  g.drawRange.count=0;r.render(frame([mesh]));assert.equal(d.passes.at(-1).draws[0].args[0],0);
  g.drawRange.count=3;r.render(frame([mesh]));assert.equal(d.passes.at(-1).bundles[0],initial);
  const old=gpu.vertexBuffer;g.attributes.position=attribute(new Float32Array(9).fill(2),3);gpu.update();
  r.render(frame([mesh]));assert.notEqual(d.passes.at(-1).draws[0].streams.get(0),old);
  assert.equal(r.bundleDiagnostics.builds,3);assert.equal(r.bundleDiagnostics.reuses,1);
  gpu.release();gpu.update();r.render(frame([mesh]));assert.equal(r.bundleDiagnostics.builds,4);
  await r.whenIdle();r.dispose();gpu.dispose();
});

test('texture groups, transparent ordering and per-draw dynamic offsets are never stale',async()=>{
  const {device:d,gpu,renderer:r}=await setup({instancing:true});
  const a=await r.addMesh(gpu,{baseColorTexture:{view:{},sampler:{}},alphaMode:'BLEND'});
  const b=await r.addMesh(gpu,{baseColorTexture:{view:{},sampler:{}},alphaMode:'BLEND'});
  r.render(frame([a,b]));const first=d.passes.at(-1).draws;
  r.render(frame([b,a]));const reversed=d.passes.at(-1).draws;
  assert.equal(first.length,2);assert.equal(reversed[0].groups.get(1).group,first[1].groups.get(1).group);
  assert.equal(reversed[1].groups.get(1).group,first[0].groups.get(1).group);
  assert.deepEqual(reversed.map(x=>x.args[4]),[0,1]);
  r.render(frame([b,a]));assert.equal(r.bundleDiagnostics.reuses,1);
  await r.whenIdle();r.dispose();gpu.dispose();
});

test('light/camera uniforms reuse while shadow and environment resource bindings invalidate',async()=>{
  const {device:d,gpu,renderer:r}=await setup({environment:true,shadows:true});d.limits.maxTextureDimension2D=4096;
  const mesh=await r.addMesh(gpu,{shading:'lambert'});
  const shadow=(view={})=>{const snapshot=Object.freeze({view,sampler:{},viewProjection:I(),version:1,width:8,height:8});
    return {sample:()=>snapshot,whenIdle:async()=>{}};};
  const environment=()=>{const snapshot=Object.freeze({profile:'f3d-animation-environment-v1',version:1,
    diffuseView:{},specularView:{},brdfView:{},sampler:{},mipLevelCount:2});
    return {sample:()=>snapshot,whenIdle:async()=>{}};};
  const map=shadow(),env=environment(),lighting={cameraPosition:[0,0,3],lights:[{type:'directional'}]};
  r.render(frame([mesh],{lighting}));lighting.cameraPosition[2]=4;lighting.lights[0].intensity=2;
  r.render(frame([mesh],{lighting}));assert.equal(r.bundleDiagnostics.reuses,1);
  r.render(frame([mesh],{lighting,shadow:{map},environment:{map:env}}));
  const bound=d.passes.at(-1).draws[0].groups.get(1).group;
  r.render(frame([mesh],{lighting,shadow:{map,bias:0.1},environment:{map:env,intensity:2}}));
  assert.equal(d.passes.at(-1).draws[0].groups.get(1).group,bound);assert.equal(r.bundleDiagnostics.reuses,2);
  r.render(frame([mesh],{lighting,shadow:{map:shadow()},environment:{map:environment()}}));
  assert.notEqual(d.passes.at(-1).draws[0].groups.get(1).group,bound);assert.equal(r.bundleDiagnostics.builds,3);
  await r.whenIdle();r.dispose();gpu.dispose();
});

test('viewport, scissor, attachments and clear operations are pass state, not frozen bundle inputs',async()=>{
  const {device:d,gpu,renderer:r}=await setup();const mesh=await r.addMesh(gpu);
  r.render(frame([mesh]));const original=d.passes.at(-1).bundles[0];
  const colorView={},depthView={};
  r.render(frame([mesh],{colorView,depthView,loadOp:'load',depthLoadOp:'load',viewport:[1,2,32,32,0,1],scissor:[1,2,16,16]}));
  const pass=d.passes.at(-1);assert.equal(pass.bundles[0],original);assert.equal(pass.desc.colorAttachments[0].view,colorView);
  assert.equal(pass.desc.colorAttachments[0].loadOp,'load');assert.equal(pass.desc.depthStencilAttachment.view,depthView);
  r.render(frame([]));assert.equal(d.passes.at(-1).bundles.length,0);assert.equal(r.drawCallCount,0);
  assert.equal(r.bundleDiagnostics.executions,2);await r.whenIdle();r.dispose();gpu.dispose();
});

test('direct-frame toggle retains current data and later valid bundle reuse',async()=>{
  const {device:d,gpu,renderer:r}=await setup();const mesh=await r.addMesh(gpu);
  for(const use of [true,false,true])r.render(frame([mesh],{renderBundles:use}));
  assert.deepEqual(d.passes.map(p=>p.bundles.length),[1,0,1]);assert.equal(r.bundleDiagnostics.builds,1);
  assert.equal(r.bundleDiagnostics.reuses,1);assert.equal(d.encodedDrawCalls,2);
  r.clearRenderBundles();assert.equal(r.bundleDiagnostics.cachedBundles,0);r.render(frame([mesh]));
  assert.equal(r.bundleDiagnostics.builds,2);await r.whenIdle();mesh.dispose();
  assert.equal(r.bundleDiagnostics.cachedDraws,0);assert.equal(gpu.disposed,false);r.dispose();gpu.dispose();
});

test('cache capacity and unsubmitted validation failures do not corrupt the last valid schedule',async()=>{
  const {device:d,gpu,renderer:r}=await setup({maxRenderBundles:2});const mesh=await r.addMesh(gpu);
  for(const count of [0,1,2,3])r.render(frame([{mesh,count}]));
  assert.equal(r.bundleDiagnostics.cachedBundles,2);assert.equal(r.bundleDiagnostics.evictions,2);
  const before={...r.bundleDiagnostics},writes=d.writes.length,submissions=d.submissions.length;
  assert.throws(()=>r.render(frame([mesh,{mesh,count:99}])));assert.equal(d.writes.length,writes);
  assert.equal(d.submissions.length,submissions);assert.deepEqual(r.bundleDiagnostics,before);
  r.render(frame([{mesh,count:3}]));assert.equal(r.bundleDiagnostics.reuses,1);
  await r.whenIdle();r.dispose();gpu.dispose();
});

for(const kind of ['bundleError','bundleFinishError','bundleExecuteError','scope','loss'])
test(`native ${kind} is terminal and clears cache without a direct replay`,async()=>{
  const {device:d,gpu,renderer:r}=await setup();const mesh=await r.addMesh(gpu);
  if(kind==='loss'){
    r.render(frame([mesh]));await r.whenIdle();d.lose();
  } else if(kind==='scope'){
    d.scopeError={message:'invalid bundle'};r.render(frame([mesh]));
  } else {
    d[kind]=new Error(kind);assert.throws(()=>r.render(frame([mesh])),e=>e===d[kind]);
    assert.equal(d.submissions.length,0);assert.equal(r.version,0);
  }
  await assert.rejects(r.whenIdle());assert.equal(r.failed,true);assert.equal(r.bundleDiagnostics.cachedBundles,0);
  assert.throws(()=>r.render(frame([mesh])));r.dispose();gpu.dispose();
});

test('bundle opt-in and capacity are checked before native allocation; the default does not require bundle APIs',async()=>{
  for(const option of [{renderBundles:null},{renderBundles:1},{maxRenderBundles:0},{maxRenderBundles:65}]){
    const d=geometryDevice();await assert.rejects(createGpuAnimationRenderer(d,option));assert.equal(d.buffers.length,0);
  }
  const d=geometryDevice();d.createRenderBundleEncoder=undefined;
  await assert.rejects(createGpuAnimationRenderer(d,{renderBundles:true}),{code:'ANIMATION_RENDER_DEVICE'});
  assert.equal(d.buffers.length,0);
  const r=await createGpuAnimationRenderer(d),gpu=createGpuBufferGeometry(d,geometry()),mesh=await r.addMesh(gpu);
  assert.equal(r.renderBundles,false);assert.equal(r.bundleDiagnostics,null);r.render(frame([mesh]));
  const writes=d.writes.length;
  assert.throws(()=>r.render(frame([mesh],{renderBundles:true})),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.equal(d.writes.length,writes);await r.whenIdle();r.dispose();gpu.dispose();
});

test('relocated GPU packages execute reusable bundles without importing the original toolkit',async()=>{
  const temp=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-bundle-package-')),f=animationFixture(),entry=path.join(temp,'actor.gltf');
  await fs.writeFile(entry,JSON.stringify(f.model));await fs.writeFile(path.join(temp,'clip data.bin'),f.bytes);
  const out=path.join(temp,'built'),built=buildAnimation(entry,out,{webgpu:true});
  assert.ok(built.emittedFiles.includes('animation_render_bundles.mjs'));
  const moved=path.join(temp,'relocated');await fs.rename(out,moved);
  const api=await import(pathToFileURL(path.join(moved,built.gpuEntry)).href);
  const d=geometryDevice(),gpu=api.createGpuBufferGeometry(d,geometry()),r=await api.createGpuAnimationRenderer(d,{renderBundles:true});
  const mesh=await r.addMesh(gpu);r.render(frame([mesh]));r.render(frame([mesh]));await r.whenIdle();
  assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);r.dispose();gpu.dispose();
});

import {createAnimationPlayer} from './animation_runtime.mjs';
import {createGpuAnimationScene} from './animation_scene.mjs';
test('actual rigid animation scenes forward the option, update current poses, and expose cache control',async()=>{
  const d=geometryDevice(),pose=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},{}],clips:[{channels:[
    {node:0,path:'translation',times:[0,1],values:[0,0,0,1,0,0]},
  ]}]});
  const scene=await createGpuAnimationScene(d,pose,[0,1].map(node=>({geometry:{node,positions:[0,0,0.5,1,0,0.5,0,1,0.5]}})),{
    rigidGeometry:true,sortObjects:false,renderer:{renderBundles:true,instancing:true},
  });
  const input=frame([]);delete input.draws;
  for(const time of [0,0.5,1]){pose.sample(time);scene.upload();scene.render(input);}
  await scene.whenIdle();assert.equal(scene.renderBundlesEnabled,true);
  assert.equal(scene.renderBundleStats.builds,1);assert.equal(scene.renderBundleStats.reuses,2);
  assert.equal(d.encodedDrawCalls,1);assert.equal(d.snapshots[2][0].args[1],2);
  assert.equal(uniforms(d.snapshots[1][0])[12],0.5);assert.equal(uniforms(d.snapshots[2][0])[12],1);
  scene.clearRenderBundles();assert.equal(scene.renderBundleStats.cachedBundles,0);
  scene.render({...input,renderBundles:false});assert.equal(scene.renderBundleStats.builds,1);
  scene.dispose();assert.equal(pose.disposed,false);pose.dispose();
});

test('1000 changing logical draws keep one recorded schedule without instancing or frozen uniforms',async()=>{
  const {device:d,gpu,renderer:r}=await setup({maxDraws:1000});const mesh=await r.addMesh(gpu);
  const draws=Array.from({length:1000},()=>({mesh,worldMatrix:I()}));
  for(let frameIndex=0;frameIndex<4;frameIndex++){
    for(let i=0;i<draws.length;i++)draws[i].worldMatrix[12]=frameIndex+i/1024;
    r.render(frame(draws));
  }
  await r.whenIdle();assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,3);
  assert.equal(r.bundleDiagnostics.cachedDraws,1000);assert.equal(d.encodedDrawCalls,1000);
  assert.equal(r.drawCallCount,1000);assert.equal(r.drawCount,1000);
  for(let frameIndex=0;frameIndex<4;frameIndex++){
    const last=d.snapshots[frameIndex][999],words=uniforms(last);
    assert.equal(words[999*64+12],frameIndex+999/1024);
    assert.deepEqual(last.groups.get(0).offsets,[999*256]);
  }
  r.dispose();gpu.dispose();
});
