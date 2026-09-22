import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuAnimationScene} from './animation_scene.mjs';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {buildAnimation} from './build_animation.mjs';
import {animationFixture} from './fixtures/animation/gltf_fixture.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const uv=[0,0,1,0,0,1];
const texture=()=>({view:{},sampler:{}});
const lighting={viewDirection:[0,0,1],lights:[{type:'directional',direction:[0,0,-1]}]};
function geometry(d) {
  return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),
    whenIdle:async()=>{},vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'},
      {shaderLocation:1,offset:12,format:'float32x3'},
      {shaderLocation:2,offset:24,format:'float32x4'},
    ]}};
}
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),lighting,draws});
function words(d,submission=-1,draw=0,instance=0) {
  const s=d.snapshots.at(submission)[draw],binding=s.groups.get(0),buffer=binding.group.entries[0].resource.buffer;
  const offset=(binding.offsets[0]??0)+instance*256;
  return new Float32Array(s.contents.get(buffer).slice(offset,offset+256).buffer);
}
const specular=w=>[w[23],w[55],w[59]];
const f32=v=>v.map(Math.fround);
const shader=d=>d.pipelines.find(p=>p.label.includes('phong-')).fragment.module.code;

// This boundary observes production registration/encoding and submitted bytes;
// it does not execute WGSL. Native pixel comparisons are separate browser tests.
for (const instancing of [false,true]) for (const renderBundles of [false,true])
  test(`Phong values stay per-use and per-frame (instances=${instancing}, bundles=${renderBundles})`,async()=>{
    const d=geometryDevice(),g=geometry(d),r=await createGpuAnimationRenderer(d,{instancing,renderBundles,maxDraws:2});
    const m=await r.addMesh(g,{shading:'phong'});
    r.render(frame([{mesh:m,specularColor:[1,0,0],shininess:8},{mesh:m,specularColor:[0,1,0],shininess:64}]));
    r.render(frame([{mesh:m,specularColor:[0,0,1],shininess:0},{mesh:m,specularColor:[.1,.2,.3],shininess:120}]));
    assert.deepEqual(specular(words(d,0)),[1,0,0]);
    assert.equal(words(d,0)[63],8);
    assert.deepEqual(specular(words(d,0,instancing?0:1,instancing?1:0)),[0,1,0]);
    assert.equal(words(d,0,instancing?0:1,instancing?1:0)[63],64);
    assert.deepEqual(specular(words(d,1)),[0,0,1]);
    assert.equal(words(d,1)[63],Math.fround(1e-4));
    assert.deepEqual(specular(words(d,1,instancing?0:1,instancing?1:0)),f32([.1,.2,.3]));
    assert.equal(r.drawCallCount,instancing?1:2);
    if(renderBundles){assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);}
    assert.equal(r.allocatedBytes,2*256+544,'no extra Phong material buffer');
    await r.whenIdle();r.dispose();assert.ok(!g.vertexBuffer.destroyed,'borrowed geometry survives');
  });

test('Phong defaults, r186 lobe and punctual attenuation do not use GGX',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  const m=await r.addMesh(g,{shading:'phong'});r.render(frame([m]));
  assert.deepEqual(specular(words(d)),f32(Array(3).fill(.005605391621829107)));
  assert.equal(words(d)[63],30);assert.equal(words(d)[22],3);
  const source=shader(d);
  assert.match(source,/exp2\(\(-5\.55473 \* vh - 6\.98316\) \* vh\)/);
  assert.match(source,/\(roughness \* 0\.5 \+ 1\.0\) \* pow\(nh, roughness\)/);
  assert.match(source,/0\.25 \* distribution \* specular_strength/);
  assert.match(source,/max\(distance \* distance, 0\.01\)/);
  assert.match(source,/window \* window/);assert.match(source,/angular \* angular \* \(3\.0 - 2\.0 \* angular\)/);
  assert.doesNotMatch(source,/perceptual_roughness|a2|visibility = 0.5/);
  await r.whenIdle();r.dispose();
});

test('specular map uses linear R, independent coordinates and no clearcoat allocation',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d),map=texture();
  const m=await r.addMesh(g,{shading:'phong',texCoords:uv,specularTexture:map,
    mapCoordinates:{specularTexture:{uvTransform:[2,0,0,3,.25,.5]}}});
  r.render(frame([m]));const s=d.snapshots.at(-1)[0],group=s.groups.get(1).group;
  assert.deepEqual(group.entries.map(e=>e.binding),[2,3]);assert.equal(group.entries[1].resource,map.view);
  assert.match(shader(d),/specular_texel = textureSample\(specular_texture, specular_sampler, input.uv_1\)/);
  assert.match(shader(d),/emission, specular_texel.r\)/);
  assert.doesNotMatch(shader(d),/metallic_roughness_texel|clearcoat_info/);
  const surface=new Float32Array(s.contents.get(s.streams.get(1)).buffer);
  assert.deepEqual([...surface.slice(6,8)],[.25,.5]);
  await r.whenIdle();r.dispose();
});

test('occlusion strength and Phong RGB occupy independent words',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  const m=await r.addMesh(g,{shading:'phong',texCoords:uv,occlusionTexture:texture(),
    occlusionStrength:.25,specularColor:[.1,.2,.3],shininess:50});
  r.render(frame([m]));const w=words(d);
  assert.equal(w[51],.25);assert.deepEqual(specular(w),f32([.1,.2,.3]));
  assert.deepEqual([w[48],w[49],w[50],w[52],w[53],w[54],w[56],w[57],w[58]],[1,0,0,0,1,0,0,0,1]);
  await r.whenIdle();r.dispose();
});

test('flat Phong derives normals before discard and does not bind stale authored tangents',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  const m=await r.addMesh(g,{shading:'phong',flatShading:true,alphaMode:'MASK',normalTexture:texture(),texCoords:uv});
  r.render(frame([m]));const source=shader(d);
  assert.ok(source.indexOf('cross(dpdy(input.world), dpdx(input.world))')<source.indexOf('discard;'));
  assert.match(source,/var normal = flat_normal/);assert.doesNotMatch(source,/normal \*= select|@location\(2\) tangent/);
  assert.match(source,/let position_dx = dpdx\(input.world\)/);
  const p=d.snapshots.at(-1)[0].pipeline;
  assert.ok(!p.vertex.buffers[0].attributes.some(a=>a.shaderLocation===2));
  await r.whenIdle();r.dispose();
});

test('Phong and PBR cannot instance together; BLEND stays source ordered',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{instancing:true,renderBundles:true}),g=geometry(d);
  const a=await r.addMesh(g,{shading:'phong'}),b=await r.addMesh(g,{shading:'metallic-roughness'});
  const c=await r.addMesh(g,{shading:'phong',alphaMode:'BLEND'});
  r.render(frame([a,b,c,c,a]));assert.equal(r.drawCallCount,5);
  assert.deepEqual(d.snapshots.at(-1).map(s=>s.args[3]),[0,1,2,3,4]);
  assert.notEqual(d.snapshots.at(-1)[0].pipeline,d.snapshots.at(-1)[1].pipeline);
  assert.equal(d.snapshots.at(-1)[2].pipeline.depthStencil.depthWriteEnabled,false);
  await r.whenIdle();r.dispose();
});

test('invalid Phong registration and final-draw changes reject before GPU effects',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  const invalid=[{specularColor:[-1,0,0]},{specularColor:[1,0]},{specularColor:[1e99,0,0]},
    {shininess:-1},{shininess:Infinity},{shininess:1e99},{flatShading:'yes'},
    {metallicFactor:.5},{metallicRoughnessTexture:texture()},{clearcoatFactor:.5},
    {specularTexture:{view:{}}}];
  for(const value of invalid){const n=d.buffers.length,w=d.writes.length;
    await assert.rejects(r.addMesh(g,{shading:'phong',...value}));
    assert.equal(d.buffers.length,n);assert.equal(d.writes.length,w);}
  for(const shading of ['unlit','lambert','metallic-roughness'])
    for(const value of [{specularColor:[1,1,1]},{shininess:10},{specularTexture:texture()}])
      await assert.rejects(r.addMesh(g,{shading,...value}),{code:'ANIMATION_RENDER_OPTIONS'});
  const m=await r.addMesh(g,{shading:'phong'}),n=d.writes.length;
  for(const value of [{specularColor:[0,-1,0]},{shininess:NaN},{roughnessFactor:.5}])
    assert.throws(()=>r.render(frame([m,{mesh:m,...value}])));
  assert.equal(d.writes.length,n);assert.equal(d.submissions.length,0);assert.equal(r.failed,false);
  r.render(frame([m]));await r.whenIdle();r.dispose();
});

test('Phong registration snapshots mutable arrays before pipeline awaits',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d),specularColor=[.2,.3,.4];
  const options={shading:'phong',specularColor,shininess:19},pending=r.addMesh(g,options);
  specularColor.fill(1);options.shininess=1;
  const m=await pending;r.render(frame([m]));
  assert.deepEqual(specular(words(d)),f32([.2,.3,.4]));assert.equal(words(d)[63],19);
  await r.whenIdle();r.dispose();
});

function computeDevice(){
  const d=geometryDevice(),encoder=d.createCommandEncoder;
  Object.assign(d.limits,{maxBindingsPerBindGroup:1000,maxComputeInvocationsPerWorkgroup:256,
    maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535});
  d.createComputePipelineAsync=async x=>({...x,getBindGroupLayout:()=>({})});
  d.createCommandEncoder=()=>({...encoder(),beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}})});
  return d;
}
test('real scene reserves lighting, snapshots Phong fields and reuses bundles after pose updates',async()=>{
  const d=computeDevice(),p=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{}],skins:[],instances:[],clips:[]});
  const values=[.1,.2,.3],s=await createGpuAnimationScene(d,p,[{geometry:{node:0,
    positions:[0,0,0,1,0,0,0,1,0],normals:[0,0,1,0,0,1,0,0,1]},
    shading:'phong',flatShading:true,specularColor:values,shininess:17}],
    {sortObjects:false,renderer:{renderBundles:true},maxBytes:8192});
  values.fill(1);s.render(frame(s.draws));const first=d.snapshots.findIndex(x=>x.length);
  assert.deepEqual(specular(words(d,first)),f32([.1,.2,.3]));assert.equal(words(d,first)[63],17);
  s.update(.25);s.render(frame(s.draws));assert.equal(s.renderBundleStats.builds,1);assert.equal(s.renderBundleStats.reuses,1);
  await s.whenIdle();s.dispose();p.dispose();
});

test('relocated GPU package executes Phong without a new dependency or source directory',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-phong-package-')),f=animationFixture();
  const entry=path.join(root,'model.gltf'),out=path.join(root,'built');
  fs.writeFileSync(entry,JSON.stringify(f.model));fs.writeFileSync(path.join(root,'clip data.bin'),f.bytes);
  const built=buildAnimation(entry,out,{webgpu:true}),moved=path.join(root,'relocated');fs.renameSync(out,moved);
  const api=await import(pathToFileURL(path.join(moved,built.gpuEntry))),d=geometryDevice();
  const r=await api.createGpuAnimationRenderer(d,{renderBundles:true});
  const m=await r.addMesh(geometry(d),{shading:'phong',specularColor:[.4,.3,.2],shininess:11});
  r.render(frame([m]));r.render(frame([{mesh:m,shininess:33}]));
  assert.deepEqual(specular(words(d,0)),f32([.4,.3,.2]));assert.equal(words(d,0)[63],11);assert.equal(words(d,1)[63],33);
  assert.equal(r.bundleDiagnostics.reuses,1);await r.whenIdle();r.dispose();
});
