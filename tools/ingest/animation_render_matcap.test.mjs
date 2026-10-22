import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const texture=()=>({view:{},sampler:{}});
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),
  lighting:{cameraPosition:[0,0,5],viewMatrix:I(),lights:[]},draws});
function geometry(d) {
  return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),
    whenIdle:async()=>{},vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'},
      {shaderLocation:1,offset:12,format:'float32x3'},
    ]}};
}
const shader=d=>d.pipelines.find(p=>p.label.includes('matcap-')).fragment.module.code;
function words(d,submission,draw=0,instance=0) {
  const snapshot=d.snapshots[submission][draw],binding=snapshot.groups.get(0);
  const buffer=binding.group.entries[0].resource.buffer;
  // Native independent-map packets: 256 base + 13*32 UV + 32 physical bytes,
  // aligned to the device's 256-byte dynamic-uniform boundary.
  const offset=(binding.offsets[0]??0)+instance*768;
  return new Float32Array(snapshot.contents.get(buffer).slice(offset).buffer);
}

test('Matcap fallback uses camera-space normals without light, environment or alpha contributions',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{threeLights:true,shadows:true,environment:true,maxDraws:1});
  const m=await r.addMesh(geometry(d),{shading:'matcap',baseColor:[2,.5,.25,.7],alphaMode:'BLEND'});
  r.render(frame([m]));assert.equal(words(d,0)[22],6);assert.equal(words(d,0)[16],2);
  const variants=d.pipelines.filter(p=>p.label.includes('matcap-'));
  assert.ok(variants.length>1);
  for(const p of variants){const s=p.fragment.module.code;
    assert.match(s,/matcap_view = unit_vector\(-view_position\)/);
    assert.match(s,/lighting.camera.w > 0.0\) \{ matcap_view = vec3<f32>\(0.0, 0.0, 1.0\)/);
    assert.match(s,/dot\(matcap_x, matcap_normal\).*dot\(matcap_y, matcap_normal\).*\* 0.495/);
    assert.match(s,/vec3<f32>\(mix\(0.2, 0.8, matcap_uv.y\)\)/);
    assert.match(s,/let rgb = rgba.rgb \* matcap_color/);
    assert.doesNotMatch(s,/fn illuminate|f3d_dfg|environment_lighting|projected_shadow|textureSample\(/);
  }
  await r.whenIdle();r.dispose();
});

for(const instancing of [false,true])
  test(`generated Matcap UVs need no surface buffer and snapshot each live matrix (${instancing})`,async()=>{
    const d=geometryDevice(),g=geometry(d),r=await createGpuAnimationRenderer(d,
      {threeLights:true,textureTransforms:true,instancing,renderBundles:true,maxDraws:2});
    const matcapTexture=texture(),originalView=matcapTexture.view,t=[2,0,0,3,.25,.5];
    const pending=r.addMesh(g,{shading:'matcap',matcapTexture,mapTransforms:{matcapTexture:t}});
    matcapTexture.view={replacement:true};t[0]=9;
    const m=await pending,allocations=d.buffers.length,pipelines=d.pipelines.length;
    r.render(frame([m,{mesh:m,mapTransforms:{matcapTexture:[.5,0,0,.25,.75,.125]}}]));
    r.render(frame([{mesh:m,mapTransforms:{matcapTexture:[1,0,0,1,0,0]}},m]));
    const first=words(d,0),second=words(d,0,instancing?0:1,instancing?1:0);
    assert.deepEqual([...first.slice(72,80)],[2,0,.25,0,0,3,.5,0]);
    assert.deepEqual([...second.slice(72,80)],[.5,0,.75,0,0,.25,.125,0]);
    assert.deepEqual([...words(d,1).slice(72,80)],[1,0,0,0,0,1,0,0]);
    assert.deepEqual([...words(d,1,instancing?0:1,instancing?1:0).slice(72,80)],[2,0,.25,0,0,3,.5,0]);
    assert.equal(d.buffers.length,allocations);assert.equal(d.pipelines.length,pipelines);
    const draw=d.snapshots[0][0];assert.equal(draw.pipeline.vertex.buffers.length,1);
    assert.equal(draw.groups.get(1).group.entries.find(e=>e.binding===3).resource,originalView);
    const s=shader(d);
    assert.match(s,/dot\(draw_info.map_uv\[2\].xyz, vec3<f32>\(matcap_uv, 1.0\)\)/);
    assert.match(s,/textureSample\(matcap_texture, matcap_sampler, matcap_coordinate\).rgb/);
    assert.doesNotMatch(s,/@location\(3\) uv:|uv_1|matcap_texel|matcap_color.a/);
    assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);
    await r.whenIdle();r.dispose();assert.equal(g.vertexBuffer.destroyed,false);
  });

test('Matcap applies the shared matrix when independent transforms are disabled',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{threeLights:true,maxDraws:1});
  const m=await r.addMesh(geometry(d),{shading:'matcap',matcapTexture:texture(),uvTransform:[2,0,0,3,.25,.5]});
  r.render(frame([m]));
  assert.deepEqual([...words(d,0).slice(24,32)],[2,0,.25,0,0,3,.5,0]);
  assert.match(shader(d),/dot\(draw_info.uv_x.xyz, vec3<f32>\(matcap_uv, 1.0\)\)/);
  assert.doesNotMatch(shader(d),/map_uv\[/);
  await r.whenIdle();r.dispose();
});

test('normal/base/opacity maps run before generated Matcap sampling and every discard',async()=>{
  const d=geometryDevice();d.limits.maxBindingsPerBindGroup=32;
  const r=await createGpuAnimationRenderer(d,{threeLights:true,textureTransforms:true,alphaMaps:true,
    alphaTextureComponent:'r',clipping:true,maxDraws:1});
  const m=await r.addMesh(geometry(d),{shading:'matcap',matcapTexture:texture(),
    baseColorTexture:texture(),normalTexture:texture(),alphaTexture:texture(),
    normalScale:[.5,.75],texCoords:[0,0,1,0,0,1],alphaMode:'MASK'});
  r.render(frame([m]));const s=shader(d),fragment=s.slice(s.indexOf('@fragment'));
  const sampled=fragment.indexOf('textureSample(matcap_texture'),discard=fragment.indexOf('discard;');
  assert.ok(fragment.indexOf('normal = unit_vector(tangent * mapped.x')<sampled);
  assert.ok(fragment.indexOf('normal *= select(-1.0, 1.0, front)')<sampled);
  assert.ok(sampled>0&&sampled<discard);
  assert.ok(fragment.indexOf('dpdx(input.world)')<discard);
  assert.match(fragment,/rgba.a \*= alpha_texel.r/);
  assert.match(fragment,/draw_info.color \* input.color \* color_texel/);
  assert.doesNotMatch(fragment,/matcap_color.a|metallic_roughness_texel|let metallic|illuminate\(/);
  assert.equal(words(d,0)[27],.5);assert.equal(words(d,0)[55],.75);
  await r.whenIdle();r.dispose();
});

test('Matcap rejects unsupported map selectors and lighting parameters before writes',async()=>{
  const d=geometryDevice(),g=geometry(d),r=await createGpuAnimationRenderer(d,{threeLights:true,textureTransforms:true,maxDraws:2});
  for(const options of [{matcapTexture:{view:{}}},{matcapTexture:{...texture(),flipY:true}},
    {matcapTexture:texture(),mapChannels:{matcapTexture:0}},
    {matcapTexture:texture(),mapCoordinates:{matcapTexture:{texCoords:[0,0,1,0,0,1]}}},
    {baseColorTexture:texture()},{normalTexture:texture()},
    {gradientTexture:texture()},{specularTexture:texture()},{metallicRoughnessTexture:texture()},
    {emissiveTexture:texture()},{occlusionTexture:texture()},{emissiveFactor:[1,0,0]},
    {metallicFactor:0},{roughnessFactor:.5},{clearcoatFactor:.5}]){
    const allocations=d.buffers.length,writes=d.writes.length;
    await assert.rejects(r.addMesh(g,{shading:'matcap',...options}));
    assert.equal(d.buffers.length,allocations);assert.equal(d.writes.length,writes);
  }
  for(const shading of ['unlit','lambert','phong','toon','normal','metallic-roughness'])
    await assert.rejects(r.addMesh(g,{shading,matcapTexture:texture()}),{code:'ANIMATION_RENDER_OPTIONS'});
  const m=await r.addMesh(g,{shading:'matcap'}),writes=d.writes.length;
  assert.throws(()=>r.render(frame([m,{mesh:m,emissiveFactor:[1,0,0]}])),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,0);assert.equal(r.failed,false);
  r.render(frame([m]));await r.whenIdle();r.dispose();
  const generic=await createGpuAnimationRenderer(d,{maxDraws:1});
  await assert.rejects(generic.addMesh(g,{shading:'matcap'}),{code:'ANIMATION_RENDER_OPTIONS'});
  generic.dispose();
});
