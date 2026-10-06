/** Production shader/binding/queue integration. Recording device, not rasterization. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {animationUvBytes, animationUvFields, snapshotAnimationMapTransforms,
  animationMapChannelKey, packAnimationMapTransforms} from './animation_uv.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const UV=[1,0,0,1,0,0], A=[2,3,4,5,.25,.5];
const fields=['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture','occlusionTexture',
  'clearcoatTexture','clearcoatRoughnessTexture','clearcoatNormalTexture','alphaTexture'];
const binding=()=>({view:{},sampler:{}});
const frame=draws=>({colorView:{},depthView:{},viewProjection:I,draws,lighting:{cameraPosition:[0,0,3],lights:[]}});
function device(){const d=geometryDevice();d.limits.maxBindingsPerBindGroup=32;return d;}
function geometry(d){return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I,whenIdle:async()=>{},
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'}]}};}
const coords=[0,0,1,0,0,1];
function words(s,slot=0,stride=768){const {group,offsets}=s.groups.get(0),entry=group.entries.find(e=>e.binding===0);
  return new Float32Array(s.contents.get(entry.resource.buffer).buffer,(offsets[0]??slot*stride)+(entry.resource.offset??0),stride/4);}
const uvWords=t=>[t[0],t[2],t[4],0,t[1],t[3],t[5],0];
function alphaShader(s){const code=s.pipeline.fragment.module.code,body=code.slice(code.indexOf('@fragment fn fragment_main'));
  assert.match(code,/@binding\(17\) var alpha_sampler/);assert.match(code,/@binding\(18\) var alpha_texture/);
  assert.match(body,/rgba\.a \*= alpha_texel\.g;/);assert.doesNotMatch(body,/rgba\.rgb \*= alpha|alpha_texel\.a/);
  assert.ok(body.indexOf('textureSample(alpha_texture')<body.indexOf('rgba.a *= alpha_texel.g'));
  assert.ok(body.indexOf('rgba.a *= alpha_texel.g')<body.indexOf('rgba.a < draw_info.options.x'));
  return body;
}

test('ninth UV slot is explicit, bounded and independent of the legacy eight-slot ABI',()=>{
  assert.equal(animationUvBytes(),256);assert.equal(animationUvBytes(9),288);
  assert.equal(animationUvFields(9),', map_uv: array<vec4<f32>, 18>');
  for(const value of [0,7,10,'9',null])assert.throws(()=>animationUvBytes(value),{code:'ANIMATION_UV_INPUT'});
  assert.throws(()=>snapshotAnimationMapTransforms({alphaTexture:A},fields,256),{code:'ANIMATION_UV_INPUT'});
  const defaults=snapshotAnimationMapTransforms({alphaTexture:A},fields,256,0,9),overrides=snapshotAnimationMapTransforms({},fields,256,0,9);
  const out=new Float32Array(74).fill(99);packAnimationMapTransforms(defaults,overrides,UV,out,1);
  assert.deepEqual([...out.slice(65,73)],uvWords(A));assert.equal(out[0],99);assert.equal(out[73],99);
  assert.throws(()=>packAnimationMapTransforms(defaults,overrides,UV,new Float32Array(64),0),{code:'ANIMATION_UV_INPUT'});
  assert.throws(()=>packAnimationMapTransforms(defaults,Array(8),UV,out,0),{code:'ANIMATION_UV_INPUT'});
  assert.equal(animationMapChannelKey({alphaTexture:3},fields,256,{uv3:true},0,9),3*65536);
});

for(const shading of ['unlit','lambert','phong','toon','metallic-roughness'])
  test(`${shading}: independent alpha texture contributes green opacity before test/blend, never RGB`,async()=>{
    const d=device(),r=await createGpuAnimationRenderer(d,{alphaMaps:true,maxDraws:3}),g=geometry(d),alpha=binding();
    const materials=[];
    for(const alphaMode of ['OPAQUE','MASK','BLEND'])materials.push(await r.addMesh(g,{shading,alphaMode,texCoords:coords,
      alphaCutoff:.25,alphaTexture:alpha,baseColorTexture:binding(),baseColor:[.5,.6,.7,.8]}));
    r.render(frame(materials));
    for(let i=0;i<3;i++){
      const s=d.snapshots[0][i],body=alphaShader(s),data=words(s,0,256);
      assert.equal(data[20],i===1?.25:-1);assert.equal(data[21],i===2?1:0);
      assert.equal(s.groups.get(1).group.entries.find(e=>e.binding===18).resource,alpha.view);
      assert.match(body,/draw_info.color \* input.color \* color_texel/);
      if(shading==='toon')assert.ok(body.indexOf('illuminate(')<body.indexOf('rgba.a < draw_info.options.x'));
    }
    await r.whenIdle();r.dispose();assert.equal(g.vertexBuffer.destroyed,false);
  });

for(const instancing of [false,true])for(const renderBundles of [false,true])
  test(`alpha matrices are per-use and stay live inside reused schedules (${instancing}/${renderBundles})`,async()=>{
    const d=device(),r=await createGpuAnimationRenderer(d,{alphaMaps:true,textureTransforms:true,instancing,renderBundles,maxDraws:2});
    const m=await r.addMesh(geometry(d),{alphaTexture:binding(),texCoords:coords,alphaMode:'MASK'});
    const first={mesh:m,mapTransforms:{alphaTexture:[...A]}},second={mesh:m,mapTransforms:{alphaTexture:UV}};
    r.render(frame([first,second]));const snapshots=d.snapshots[0],before=[d.buffers.length,d.pipelines.length,d.bundleEncoders.length];
    assert.deepEqual([...words(snapshots[0]).slice(128,136)],uvWords(A));
    assert.deepEqual([...words(snapshots.at(-1),instancing?1:0).slice(128,136)],uvWords(UV));
    assert.equal(r.drawCallCount,instancing?1:2);
    if(instancing)assert.match(snapshots[0].pipeline.fragment.module.code,/@location\(15\) @interpolate\(flat\) draw_index/);
    first.mapTransforms.alphaTexture[4]=.75;r.render(frame([first,second]));
    assert.equal(words(d.snapshots[1][0])[130],.75);assert.equal(words(snapshots[0])[130],.25);
    assert.equal(d.buffers.length,before[0]);assert.equal(d.pipelines.length,before[1]);
    if(renderBundles)assert.equal(d.bundleEncoders.length,before[2]);
    await r.whenIdle();r.dispose();
  });

test('depth-only masks use the same alpha sample and independent baked coordinates',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{format:null,alphaMaps:true,textureTransforms:true,maxDraws:1});
  const m=await r.addMesh(geometry(d),{alphaTexture:binding(),alphaMode:'MASK',alphaCutoff:.6,
    mapCoordinates:{alphaTexture:{texCoords:coords,uvTransform:A}}});
  r.render({depthView:{},viewProjection:I,draws:[{mesh:m,alphaCutoff:.125,mapTransforms:{alphaTexture:UV}}]});
  const s=d.snapshots[0][0];alphaShader(s);assert.deepEqual(s.pipeline.fragment.targets,[]);
  assert.match(s.pipeline.vertex.module.code,/@location\(13\) uv_8/);
  assert.equal(words(s)[20],.125);r.dispose();
});

test('all nine maps coexist with clearcoat: independent binding indices and layout identities',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{alphaMaps:true,textureTransforms:true,instancing:true,fog:true,clipping:true,maxDraws:3});
  const g=geometry(d),textures=Object.fromEntries(fields.map(f=>[f,binding()]));
  const complete=await r.addMesh(g,{...textures,shading:'metallic-roughness',clearcoatFactor:.5,texCoords:coords});
  const alpha=await r.addMesh(g,{alphaTexture:binding(),texCoords:coords});
  const coat=await r.addMesh(g,{shading:'metallic-roughness',clearcoatFactor:.5});
  r.render(frame([complete,alpha,coat]));const s=d.snapshots[0][0],entries=s.groups.get(1).group.entries;
  assert.deepEqual(entries.map(e=>e.binding).sort((a,b)=>a-b),Array.from({length:19},(_,i)=>i));
  assert.equal(entries.find(e=>e.binding===16).resource.size,16);
  const code=s.pipeline.fragment.module.code;alphaShader(s);
  assert.match(code,/@location\(13\) uv_8/);assert.match(code,/@location\(14\) fog_depth/);assert.match(code,/@location\(15\).*draw_index/);
  assert.notEqual(d.snapshots[0][1].groups.get(1).group.layout,d.snapshots[0][2].groups.get(1).group.layout);
  assert.ok(code.indexOf('let coat_uv_dx = dpdx')<code.indexOf('rgba.a *= alpha_texel.g'));
  await r.whenIdle();r.dispose();
});

test('alpha testing can coexist with blending while retaining live thresholds and depth-write policy',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{alphaMaps:true,maxDraws:2}),g=geometry(d);
  const a=await r.addMesh(g,{alphaMode:'BLEND',alphaTest:true,alphaCutoff:.3,alphaTexture:binding(),texCoords:coords,depthWrite:true});
  const b=await r.addMesh(g,{alphaMode:'BLEND'});
  r.render(frame([{mesh:a,alphaCutoff:.75},b]));
  assert.equal(words(d.snapshots[0][0],0,256)[20],.75);assert.equal(words(d.snapshots[0][0],0,256)[21],1);
  assert.equal(d.snapshots[0][0].pipeline.depthStencil.depthWriteEnabled,true);
  assert.equal(words(d.snapshots[0][1],0,256)[20],-1);assert.equal(d.snapshots[0][1].pipeline.depthStencil.depthWriteEnabled,false);
  const before=d.writes.length;assert.throws(()=>r.render(frame([{mesh:b,alphaCutoff:.2}])),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.equal(d.writes.length,before);r.dispose();
});

test('invalid later alpha state leaves the previous frame and owner usable without queue effects',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{alphaMaps:true,textureTransforms:true,maxDraws:2});
  const m=await r.addMesh(geometry(d),{alphaTexture:binding(),texCoords:coords,alphaMode:'MASK'});
  r.render(frame([m]));const before=[d.writes.length,d.submissions.length];
  for(const value of [{alphaCutoff:NaN},{mapTransforms:{alphaTexture:[1,0,0,1,NaN,0]}}]){
    assert.throws(()=>r.render(frame([m,{mesh:m,...value}])));
    assert.deepEqual([d.writes.length,d.submissions.length],before);assert.equal(r.failed,false);
  }
  r.render(frame([m]));assert.equal(r.version,2);r.dispose();
});

test('alpha capacity respects binding indices, varying slots, packet alignment and total budgets',async()=>{
  for(const options of [{alphaMaps:1},{alphaMaps:true,textureTransforms:true,maxDraws:1,maxBytes:767}]){
    const d=device();await assert.rejects(createGpuAnimationRenderer(d,options));assert.equal(d.buffers.length,0);
  }
  for(const [name,value] of [['maxBindingsPerBindGroup',18],['maxInterStageShaderVariables',15],['maxSamplersPerShaderStage',0],['maxSampledTexturesPerShaderStage',0]]){
    const d=device(),r=await createGpuAnimationRenderer(d,{alphaMaps:true,textureTransforms:true,instancing:true,maxDraws:1}),g=geometry(d);
    d.limits[name]=value;const before=d.buffers.length;
    await assert.rejects(r.addMesh(g,{alphaTexture:binding(),texCoords:coords}),{code:'ANIMATION_RENDER_LIMIT'});
    assert.equal(d.buffers.length,before);assert.equal(d.submissions.length,0);r.dispose();
  }
  const d=device();d.limits.maxUniformBufferBindingSize=543;
  await assert.rejects(createGpuAnimationRenderer(d,{alphaMaps:true,textureTransforms:true,maxDraws:1}),{code:'ANIMATION_RENDER_LIMIT'});
  assert.equal(d.buffers.length,0);
});

test('default-off preserves legacy packet/resources and rejects opacity instead of ignoring it',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),g=geometry(d);
  assert.equal(r.allocatedBytes,256);
  await assert.rejects(r.addMesh(g,{alphaTexture:binding(),texCoords:coords}),{code:'ANIMATION_RENDER_OPTIONS'});
  for(const alphaTest of [false,1,null])await assert.rejects(r.addMesh(g,{alphaMode:'MASK',alphaTest}),{code:'ANIMATION_RENDER_OPTIONS'});
  const m=await r.addMesh(g);r.render(frame([m]));assert.equal(d.snapshots[0][0].groups.has(1),false);r.dispose();
});
