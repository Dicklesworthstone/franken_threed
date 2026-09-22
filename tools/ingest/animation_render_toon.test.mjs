import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuAnimationScene} from './animation_scene.mjs';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {buildAnimation} from './build_animation.mjs';
import {animationFixture} from './fixtures/animation/gltf_fixture.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const texture=()=>({view:{},sampler:{}});
const lighting={viewDirection:[0,0,1],lights:[{type:'directional',direction:[0,0,-1]}]};
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),lighting,draws});
function geometry(d) {
  return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),
    whenIdle:async()=>{},vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'},
      {shaderLocation:1,offset:12,format:'float32x3'},
      {shaderLocation:2,offset:24,format:'float32x4'},
    ]}};
}
function words(d,submission,draw=0,instance=0) {
  const s=d.snapshots[submission][draw],binding=s.groups.get(0),buffer=binding.group.entries[0].resource.buffer;
  const offset=(binding.offsets[0]??0)+instance*256;
  return new Float32Array(s.contents.get(buffer).slice(offset,offset+256).buffer);
}
const toonShader=d=>d.pipelines.find(p=>p.label.includes('toon-')).fragment.module.code;

// Production registration, packing and queue boundaries are exercised here.
// Shader arithmetic and pixels require runPhongToonChecks on a real GPU.
for (const instancing of [false,true]) for (const renderBundles of [false,true])
  test(`toon ramp needs no UVs; snapshots and draw reuse survive updates (${instancing}/${renderBundles})`,async()=>{
    const d=geometryDevice(),g=geometry(d),r=await createGpuAnimationRenderer(d,{instancing,renderBundles,maxDraws:2});
    const gradientTexture=texture(),originalView=gradientTexture.view;
    const pending=r.addMesh(g,{shading:'toon',gradientTexture});gradientTexture.view={replacement:true};
    const m=await pending;
    assert.equal(r.allocatedBytes,512+544,'no synthetic UVs, color or ramp buffer');
    r.render(frame([{mesh:m,baseColor:[1,0,0,1]},{mesh:m,baseColor:[0,1,0,1]}]));
    r.render(frame([{mesh:m,baseColor:[0,0,1,1]},{mesh:m,emissiveFactor:[.25,0,0]}]));
    assert.deepEqual([...words(d,0).slice(16,20)],[1,0,0,1]);
    assert.deepEqual([...words(d,0,instancing?0:1,instancing?1:0).slice(16,20)],[0,1,0,1]);
    assert.deepEqual([...words(d,1).slice(16,20)],[0,0,1,1]);
    assert.equal(words(d,1,instancing?0:1,instancing?1:0)[60],.25);
    assert.equal(words(d,0)[22],4);assert.equal(r.drawCallCount,instancing?1:2);
    const draw=d.snapshots[0][0];assert.equal(draw.pipeline.vertex.buffers.length,1);
    assert.equal(draw.groups.get(1).group.entries[1].resource,originalView);
    if(renderBundles){assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);}
    await r.whenIdle();r.dispose();assert.equal(g.vertexBuffer.destroyed,false);
  });

test('default ramp preserves the signed light angle and uses a finite zero-footprint step',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),m=await r.addMesh(geometry(d),{shading:'toon'});
  r.render(frame([m]));const s=toonShader(d),loop=s.slice(s.indexOf('for (var i = 0u;'),s.indexOf('return result;'));
  assert.match(loop,/dot\(normal, incoming\) \* 0.5 \+ 0.5/);
  assert.match(loop,/fwidth\(gradient_coordinate.x\) \* 0.5/);
  assert.match(loop,/select\(0.7, 1.0, gradient_coordinate.x >= 0.7\)/);
  assert.match(loop,/if \(width > 0.0\)/);assert.match(loop,/smoothstep\(0.7 - width, 0.7 \+ width/);
  assert.doesNotMatch(loop,/continue;|let nl = max|perceptual_roughness|half_vector/);
  assert.match(loop,/brdf \* \(nl \* attenuation\)/,'ramp contributes once, not times a second cosine');
  assert.ok(s.lastIndexOf('rgb = illuminate(')<s.indexOf('discard;'));
  assert.doesNotMatch(s,/textureSample\(|@location\(3\) uv:/);
  assert.equal(r.allocatedBytes,256*1024+544);await r.whenIdle();r.dispose();
});

test('gradient texture uses linear R at angle coordinates, independent of material UV transforms',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),m=await r.addMesh(geometry(d),{shading:'toon',
    gradientTexture:texture(),baseColorTexture:texture(),normalTexture:texture(),flatShading:true,
    alphaMode:'MASK',texCoords:[0,0,1,0,0,1],mapCoordinates:{baseColorTexture:{uvTransform:[2,0,0,1,.5,0]}}});
  r.render(frame([m]));const s=toonShader(d);
  assert.match(s,/textureSample\(gradient_texture, gradient_sampler, gradient_coordinate\).r/);
  assert.doesNotMatch(s,/gradient_texel|input.uv_1|fwidth\(gradient_coordinate/);
  assert.match(s,/textureSample\(color_texture, color_sampler, input.uv_0\)/);
  assert.match(s,/let position_dx = dpdx\(input.world\)/);
  assert.match(s,/flat_normal = unit_vector\(cross\(dpdy\(input.world\), dpdx\(input.world\)\)\)/);
  const fragment=s.slice(s.indexOf('@fragment'));
  assert.ok(fragment.indexOf('rgb = illuminate(')<fragment.indexOf('discard;'));
  assert.doesNotMatch(fragment,/normal \*= select|@location\(2\) tangent/);
  await r.whenIdle();r.dispose();
});

test('toon light attenuation, shadows and diffuse environment generate complete variants',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{shadows:true,environment:true});
  const m=await r.addMesh(geometry(d),{shading:'toon',gradientTexture:texture(),
    occlusionTexture:texture(),texCoords:[0,0,1,0,0,1]});
  const variants=d.pipelines.filter(p=>p.label.includes('toon-'));assert.equal(variants.length,32);
  for(const p of variants){const s=p.fragment.module.code;
    assert.match(s,/max\(distance \* distance, 0.01\)/);assert.match(s,/attenuation \*= window \* window/);
    assert.match(s,/angular \* angular \* \(3.0 - 2.0 \* angular\)/);
    assert.ok(s.lastIndexOf('rgb = illuminate(')<s.indexOf('discard;'));
    if(p.label.includes('environment-'))assert.match(s,/environment_lighting.*== 2.0\) \* occlusion/);
    if(p.label.includes('shadow-'))assert.match(s,/projected_shadow\(position, normal\)/);
  }
  r.render(frame([m]));await r.whenIdle();r.dispose();
});

test('inapplicable maps/parameters and late draw overrides fail before writes',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  for(const options of [{gradientTexture:{view:{}}},{gradientTexture:{...texture(),flipY:true}},
    {gradientTexture:texture(),mapCoordinates:{gradientTexture:{texCoords:[0,0,1,0,0,1]}}},
    {specularTexture:texture()},{specularColor:[1,1,1]},{shininess:30},
    {metallicRoughnessTexture:texture()},{metallicFactor:0},{roughnessFactor:.5},{clearcoatFactor:.5}]){
    const allocations=d.buffers.length,writes=d.writes.length;
    await assert.rejects(r.addMesh(g,{shading:'toon',...options}));
    assert.equal(d.buffers.length,allocations);assert.equal(d.writes.length,writes);
  }
  for(const shading of ['unlit','lambert','phong','metallic-roughness'])
    await assert.rejects(r.addMesh(g,{shading,gradientTexture:texture()}),{code:'ANIMATION_RENDER_OPTIONS'});
  const m=await r.addMesh(g,{shading:'toon'}),writes=d.writes.length;
  for(const options of [{shininess:30},{specularColor:[1,1,1]},{metallicFactor:0},{gradientTexture:texture()}])
    assert.throws(()=>r.render(frame([m,{mesh:m,...options}])));
  assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,0);assert.equal(r.failed,false);
  r.render(frame([m]));await r.whenIdle();r.dispose();
});

test('gradient descriptors are shared by identity without merging shading models or BLEND draws',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{instancing:true,renderBundles:true}),g=geometry(d),t=texture();
  const a=await r.addMesh(g,{shading:'toon',gradientTexture:t});
  const b=await r.addMesh(g,{shading:'toon',gradientTexture:t});
  const c=await r.addMesh(g,{shading:'toon',gradientTexture:texture()});
  const p=await r.addMesh(g,{shading:'phong'}),blend=await r.addMesh(g,{shading:'toon',gradientTexture:t,alphaMode:'BLEND'});
  r.render(frame([a,b,c,p,blend,blend,a]));assert.equal(r.drawCallCount,6);
  assert.deepEqual(d.snapshots.at(-1).map(s=>s.args[1]),[2,1,1,1,1,1]);
  assert.deepEqual(d.snapshots.at(-1).map(s=>s.args[3]),[0,2,3,4,5,6]);
  assert.equal(d.snapshots.at(-1)[3].pipeline.depthStencil.depthWriteEnabled,false);
  a.dispose();assert.equal(r.bundleDiagnostics.cachedBundles,0);
  r.render(frame([b]));await r.whenIdle();r.dispose();
});

test('live BufferGeometry supports UV-free ramps and reuses bundles across normal/position updates',async()=>{
  const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
  const {BufferGeometry,BufferAttribute}=await import(pathToFileURL(path.join(root,'build/three.core.js')));
  const g=new BufferGeometry().setAttribute('position',new BufferAttribute(new Float32Array([0,0,0,1,0,0,0,1,0]),3))
    .setAttribute('normal',new BufferAttribute(new Float32Array([0,0,1,0,0,1,0,0,1]),3));
  const d=geometryDevice(),gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d,{renderBundles:true});
  const m=await r.addMesh(gpu,{shading:'toon',gradientTexture:texture(),flatShading:true});
  r.render(frame([m]));const a=bufferGeometrySnapshot(gpu,d),count=d.buffers.length;
  g.attributes.position.array[2]=.25;g.attributes.position.needsUpdate=true;gpu.update();r.render(frame([m]));
  assert.equal(d.buffers.length,count);assert.equal(r.bundleDiagnostics.reuses,1);
  assert.deepEqual([...d.snapshots.at(-1)[0].streams.values()],a.vertexBuffers);
  await r.whenIdle();r.dispose();assert.equal(gpu.disposed,false);gpu.dispose();
});

function computeDevice(){
  const d=geometryDevice(),encoder=d.createCommandEncoder;
  Object.assign(d.limits,{maxBindingsPerBindGroup:1000,maxComputeInvocationsPerWorkgroup:256,
    maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535});
  d.createComputePipelineAsync=async x=>({...x,getBindGroupLayout:()=>({})});
  d.createCommandEncoder=()=>({...encoder(),beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}})});
  return d;
}
test('scene forwards angle maps without charging nonexistent UV buffers, at its exact budget',async()=>{
  async function create(maxBytes){
    const d=computeDevice(),p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{}],skins:[],instances:[],clips:[]});
    const t=texture(),original=t.view,pending=createGpuAnimationScene(d,p,[{geometry:{node:0,
      positions:[0,0,0,1,0,0,0,1,0],normals:[0,0,1,0,0,1,0,0,1]},shading:'toon',gradientTexture:t}],
      {sortObjects:false,renderer:{renderBundles:true},maxBytes});
    t.view={};
    try{return {d,p,s:await pending,original};}catch(e){p.dispose();throw e;}
  }
  const a=await create(8192),bytes=a.s.bufferBytes;a.s.dispose();a.p.dispose();
  const {d,p,s,original}=await create(bytes);
  s.render(frame(s.draws));s.update(.25);s.render(frame(s.draws));
  assert.equal(s.renderBundleStats.reuses,1);assert.equal(s.bufferBytes,bytes);
  assert.equal(d.snapshots.at(-1)[0].groups.get(1).group.entries[1].resource,original);
  await s.whenIdle();s.dispose();p.dispose();
  await assert.rejects(create(bytes-1),/budget|buffers/);
});

test('relocated GPU package executes the toon profile without extra runtime modules',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-toon-package-')),f=animationFixture();
  const entry=path.join(root,'model.gltf'),out=path.join(root,'built');
  fs.writeFileSync(entry,JSON.stringify(f.model));fs.writeFileSync(path.join(root,'clip data.bin'),f.bytes);
  const built=buildAnimation(entry,out,{webgpu:true}),moved=path.join(root,'relocated');fs.renameSync(out,moved);
  const api=await import(pathToFileURL(path.join(moved,built.gpuEntry))),d=geometryDevice();
  const r=await api.createGpuAnimationRenderer(d,{renderBundles:true}),m=await r.addMesh(geometry(d),{shading:'toon',gradientTexture:texture()});
  r.render(frame([m]));r.render(frame([m]));assert.equal(r.bundleDiagnostics.reuses,1);
  assert.match(toonShader(d),/gradient_coordinate/);await r.whenIdle();r.dispose();
});
