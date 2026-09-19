import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationRenderer} from './animation_render.mjs';
const I = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const UV = [0,0,1,0,0,1];
const maps = ['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture','occlusionTexture',
  'clearcoatTexture','clearcoatRoughnessTexture','clearcoatNormalTexture'];
const texture = () => ({view:{},sampler:{}});
// Production renderer and receivers; this device records their actual API
// calls/data. No WGSL execution, native pixels or throughput is claimed here.
function device(limits={}) {
  const buffers=[],pipelines=[],groups=[],passes=[],writes=[],submissions=[];let scopes=0,lose;
  const d={limits:{minUniformBufferOffsetAlignment:256,maxBufferSize:2**27,maxUniformBufferBindingSize:65536,
    maxStorageBuffersPerShaderStage:8,maxStorageBufferBindingSize:2**27,maxDynamicUniformBuffersPerPipelineLayout:8,
    maxBindGroups:4,maxUniformBuffersPerShaderStage:12,maxVertexBuffers:8,maxVertexAttributes:16,
    maxVertexBufferArrayStride:2048,maxInterStageShaderVariables:16,maxSamplersPerShaderStage:16,
    maxBindingsPerBindGroup:1000,maxSampledTexturesPerShaderStage:16,maxTextureDimension2D:4096,...limits},
    lost:new Promise(r=>{lose=r;}),pushErrorScope(){scopes++;},
    popErrorScope(){assert.ok(scopes>0);scopes--;return Promise.resolve(null);},
    createBuffer(desc){const data=new ArrayBuffer(desc.size),b={...desc,data,destroyed:0,
      getMappedRange:()=>data,unmap(){},destroy(){this.destroyed++;}};buffers.push(b);return b;},
    createBindGroupLayout:x=>x,createPipelineLayout:x=>x,createShaderModule:x=>x,
    createBindGroup(x){groups.push(x);return x;},async createRenderPipelineAsync(x){pipelines.push(x);return x;},
    createCommandEncoder(){const command={passes:[]};return {beginRenderPass(desc){
      const p={desc,draws:[]};passes.push(p);command.passes.push(p);let pipeline,index;const bindings=new Map(),vertices=new Map();
      const draw=(indexed,args)=>p.draws.push({pipeline,index,bindings:new Map(bindings),vertices:new Map(vertices),indexed,args});
      return {setPipeline(x){pipeline=x;},setBindGroup(i,g,offsets=[]){bindings.set(i,{group:g,offsets});},
        setVertexBuffer(i,b){vertices.set(i,b);},setIndexBuffer(b,format){index={buffer:b,format};},
        setViewport(){},setScissorRect(){},draw(...a){draw(false,a);},drawIndexed(...a){draw(true,a);},end(){}};
    },finish:()=>command};},
    queue:{writeBuffer(buffer,offset,value,start=0,length){
      const width=value.BYTES_PER_ELEMENT??1,storage=ArrayBuffer.isView(value)?value.buffer:value;
      const bytes=new Uint8Array(storage,(value.byteOffset??0)+start*width,(length??(value.byteLength/width-start))*width).slice();
      new Uint8Array(buffer.data,offset,bytes.length).set(bytes);writes.push({buffer,offset,bytes});
    },submit(commands){submissions.push(commands);},onSubmittedWorkDone:async()=>{}},
  };
  return {d,buffers,pipelines,groups,passes,writes,submissions,lose,get scopes(){return scopes;}};
}
const gpu=(vertexBuffer={},tangents=true)=>({vertexBuffer,vertexCount:3,worldMatrix:I(),disposed:false,failed:false,whenIdle:async()=>{},
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},
    {shaderLocation:1,offset:12,format:'float32x3'},...(tangents?[{shaderLocation:2,offset:24,format:'float32x4'}]:[])]}});
const frame=(draws,extra={})=>({colorView:{},depthView:{},viewProjection:I(),draws,
  lighting:{cameraPosition:[0,0,3],lights:[{type:'directional'}]},...extra});
const last=g=>g.passes.at(-1).draws;
const code=s=>({code:'ANIMATION_RENDER_'+s});
function environment(d) {const snapshot=Object.freeze({profile:'f3d-animation-environment-v1',version:1,
  diffuseView:{},specularView:{},brdfView:{},sampler:{},mipLevelCount:3});
  return {sample(device){assert.equal(device,d);return snapshot;},whenIdle:async()=>{}};}
function shadow(d){const snapshot=Object.freeze({view:{},sampler:{},version:1,width:8,height:8,viewProjection:I()});
  return {sample(device){assert.equal(device,d);return snapshot;},whenIdle:async()=>{}};}

for (const instancing of [false,true]) test(`factor-only coating has no placeholder textures/UVs or larger draw arena; instancing=${instancing}`,async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing,maxDraws:2,label:'draw'});
  const m=await r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatFactor:0.75,clearcoatRoughnessFactor:0.25});
  assert.equal(r.allocatedBytes,512+544+16);assert.equal(g.buffers.filter(b=>b.label.endsWith('/surface')).length,0);
  const c=g.buffers.find(b=>b.label.endsWith('/clearcoat'));assert.deepEqual([...new Float32Array(c.data)],[0.75,0.25,1,0]);
  r.render(frame([m,m]));const draw=last(g)[0];assert.equal(draw.bindings.get(1).group.entries.length,1);
  assert.equal(draw.bindings.get(1).group.entries[0].binding,16);assert.ok(draw.bindings.get(2));
  assert.equal(r.drawCallCount,instancing?1:2);assert.equal(r.drawCount,2);
  const shader=draw.pipeline.vertex.module.code;assert.match(shader,/clearcoat_info: vec4<f32>/);
  assert.doesNotMatch(shader,/var clearcoat_sampler/);assert.match(shader,/var coat_normal = normal/);
  await r.whenIdle();r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));assert.equal(g.scopes,0);
});

for(const tangents of [false,true])for(const instancing of [false,true])for(const shadows of [false,true])for(const ibl of [false,true])
  test(`all eight maps: tangents=${tangents}, instancing=${instancing}, shadow=${shadows}, environment=${ibl}`,async()=>{
    const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing,shadows,environment:ibl,maxDraws:2,label:'draw'});
    const material={shading:'metallic-roughness',clearcoatFactor:1,clearcoatRoughnessFactor:0.1,clearcoatNormalScale:-0.5,
      normalScale:2,occlusionStrength:0.2,texCoords:UV,mapCoordinates:{},alphaMode:'MASK'};
    maps.forEach((f,i)=>{material[f]=texture();material.mapCoordinates[f]={texCoords:UV,uvTransform:[i+1,0,0,i+1,0,0]};});
    const m=await r.addMesh(gpu({},tangents),material);
    r.render(frame([m,m],{...(shadows?{shadow:{map:shadow(g.d)}}:{}),...(ibl?{environment:{map:environment(g.d)}}:{})}));
    const draw=last(g)[0],shader=draw.pipeline.vertex.module.code;
    assert.equal(draw.bindings.get(1).group.entries.length,17);assert.equal(draw.pipeline.vertex.buffers[1].arrayStride,88);
    const locations=draw.pipeline.vertex.buffers.flatMap(b=>b.attributes.map(a=>a.shaderLocation));assert.equal(new Set(locations).size,locations.length);
    if(instancing)assert.match(shader,/@location\(13\) @interpolate\(flat\) draw_index/);
    assert.match(shader,/clearcoat_info.x \* clearcoat_texel.r/);assert.match(shader,/clearcoat_info.y \* clearcoat_roughness_texel.g/);
    assert.match(shader,/mapped.xy \* clearcoat_info.z/);
    assert.equal(shader.includes('coat_position_dx = dpdx'),!tangents);assert.equal(shader.includes('position_dx = dpdx(input.world)'),!tangents);
    if(!tangents){assert.ok(shader.indexOf('let coat_uv_dx = dpdx(input.uv_7)')<shader.indexOf('discard;'));
      assert.match(shader,/let uv_dx = dpdx\(input.uv_2\)/);}
    assert.ok(shader.indexOf('var coat_normal = normal')<shader.indexOf('var mapped = normal_map_texel'));
    assert.match(shader,/coat_normal \*= select\(-1.0, 1.0, front\)/);
    assert.equal(shader.includes('fn projected_shadow'),shadows);assert.equal(shader.includes('fn environment_lighting'),ibl);
    assert.match(shader,/illuminate\(vec3<f32>\(1.0\), input.world, coat_normal, 1.0, coat_roughness, vec3<f32>\(0.0\)/);
    if(ibl)assert.match(shader,/coat_roughness, vec3<f32>\(0.0\), occlusion\)/);
    const surface=draw.vertices.get(1),values=new Float32Array(surface.data);
    for(let i=0;i<8;i++)assert.equal(values[22+6+i*2],i+1);
    r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
  });

for(const tangents of [false,true])test(`coat-only normal map does not require a base normal map (${tangents?'authored':'derivative'})`,async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:1,label:'draw'}),d=gpu({},tangents);d.worldMatrix[0]=-1;
  const m=await r.addMesh(d,{shading:'metallic-roughness',clearcoatFactor:1,clearcoatNormalTexture:texture(),texCoords:UV});r.render(frame([m]));
  const shader=last(g)[0].pipeline.vertex.module.code;
  assert.doesNotMatch(shader,/var mapped = normal_map_texel/);
  assert.equal(shader.includes('let tangent = unit_vector(input.tangent'),tangents);
  if(!tangents)assert.doesNotMatch(shader,/let uv_dx = dpdx/);
  const upload=g.writes.find(w=>w.buffer.label==='draw');assert.equal(new Float32Array(upload.bytes.buffer)[31],-1);
  r.dispose();
});

test('emitted Fresnel mix uses the coating factor once, attenuates base emission, and preserves alpha',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{maxDraws:1}),m=await r.addMesh(gpu(),{
    shading:'metallic-roughness',clearcoatFactor:1,emissiveFactor:[10,20,30],alphaMode:'BLEND',baseColor:[1,1,1,0.25]});
  r.render(frame([m]));const shader=last(g)[0].pipeline.vertex.module.code;
  const expression=shader.match(/let weight = ([^;]+);/)[1],weight=new Function('coat_factor','edge','return '+expression);
  // Execute the scalar expression actually emitted to WGSL, not a second
  // implementation of the weighting rule. This is still NOT native WGSL.
  for(const [strength,edge,expected] of [[1,0,0.04],[1,1,1],[0,1,0],[0.5,0,0.02],[1,0.5,0.07]])assert.equal(weight(strength,edge),expected);
  assert.match(shader,/rgb = rgb \* \(1.0 - weight\) \+ coat_light \* weight/);
  assert.match(shader,/return vec4<f32>\(rgb, select\(1.0, rgba.a, draw_info.options.y > 0.0\)\)/);
  assert.match(shader,/if \(coat_factor > 0.0/);r.dispose();
});

for(const [field,value] of [['clearcoatFactor',-1],['clearcoatFactor',2],['clearcoatFactor',NaN],['clearcoatFactor',null],
  ['clearcoatRoughnessFactor',Infinity],['clearcoatRoughnessFactor',-0.1],['clearcoatNormalScale',1e100]])
  test(`invalid ${field}=${value} rejects before native material work`,async()=>{
    const g=device(),r=await createGpuAnimationRenderer(g.d,{maxDraws:1});const before=[g.buffers.length,g.groups.length,g.pipelines.length];
    await assert.rejects(r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatNormalTexture:texture(),texCoords:UV,[field]:value}),code('VALUE'));
    assert.deepEqual([g.buffers.length,g.groups.length,g.pipelines.length],before);r.dispose();
  });

for(const material of [{clearcoatFactor:1},{shading:'lambert',clearcoatTexture:texture()},
  {shading:'metallic-roughness',clearcoatNormalScale:1}])test('inapplicable coating parameters are never ignored: '+JSON.stringify(material),async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{maxDraws:1});await assert.rejects(r.addMesh(gpu(),material),code('OPTIONS'));r.dispose();
});

for(const limits of [{maxUniformBuffersPerShaderStage:2},{maxBindGroups:2},{maxBindingsPerBindGroup:0},
  {maxInterStageShaderVariables:13}])test('coating resource limits reject before allocating: '+JSON.stringify(limits),async()=>{
  const g=device(limits),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:1}),count=g.buffers.length;
  await assert.rejects(r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatFactor:1}),code('LIMIT'));assert.equal(g.buffers.length,count);r.dispose();
});

test('aggregate samplers include all coating maps plus projected shadows and IBL',async()=>{
  const g=device({maxSamplersPerShaderStage:9}),r=await createGpuAnimationRenderer(g.d,{shadows:true,environment:true,maxDraws:1});
  const input={shading:'metallic-roughness',texCoords:UV,clearcoatFactor:1};maps.forEach(f=>input[f]=texture());
  await assert.rejects(r.addMesh(gpu(),input),code('LIMIT'));assert.equal(g.buffers.length,1);r.dispose();
});

for(const instancing of [false,true])test(`exact coated material budget includes the 16-byte uniform (${instancing})`,async()=>{
  for(const maxBytes of [815,816]) {
    const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing,maxDraws:1,maxBytes});
    if(maxBytes===815){await assert.rejects(r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatFactor:1}),code('LIMIT'));assert.equal(g.buffers.length,1);}
    else {await r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatFactor:1});assert.equal(r.allocatedBytes,maxBytes);}
    r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
  }
});

test('identical coating materials share one uniform/bind group and instance; different factors split',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:4}),vertex={},meshes=[];
  for(const factor of [0.5,0.5,1,1])meshes.push(await r.addMesh(gpu(vertex),{shading:'metallic-roughness',clearcoatFactor:factor}));
  assert.equal(g.buffers.filter(b=>b.label.endsWith('/clearcoat')).length,2);
  r.render(frame(meshes));assert.deepEqual(last(g).map(d=>d.args),[[3,2,0,0],[3,2,0,2]]);
  const b=last(g)[0].bindings.get(1).group.entries[0].resource.buffer;
  meshes[0].dispose();assert.equal(b.destroyed,0);meshes[1].dispose();assert.equal(b.destroyed,1);
  r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

test('concurrent coating registrations merge only validated uniforms and retire private duplicates',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),vertex={},ready=[];
  g.d.createRenderPipelineAsync=x=>new Promise(resolve=>ready.push(()=>resolve(x)));
  const pending=[r.addMesh(gpu(vertex),{shading:'metallic-roughness',clearcoatFactor:1}),r.addMesh(gpu(vertex),{shading:'metallic-roughness',clearcoatFactor:1})];
  ready.forEach(f=>f());const meshes=await Promise.all(pending);
  const buffers=g.buffers.filter(b=>b.label.endsWith('/clearcoat'));assert.equal(buffers.length,2);assert.equal(buffers.filter(b=>b.destroyed===1).length,1);
  r.render(frame(meshes));assert.equal(r.drawCallCount,1);const resource=last(g)[0].bindings.get(1).group.entries[0].resource;
  assert.equal(resource.buffer.destroyed,0);r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

for(const failure of ['mapping','binding','compile'])test(`${failure} failure cannot leak a coating uniform or retire a sibling`,async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),m=await r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatFactor:0.5}),before=r.allocatedBytes;
  const create=g.d.createBuffer,bind=g.d.createBindGroup,compile=g.d.createRenderPipelineAsync;
  if(failure==='mapping')g.d.createBuffer=desc=>{const b=create(desc);if(desc.label.endsWith('/clearcoat'))b.getMappedRange=()=>{throw Error('mapping');};return b;};
  if(failure==='binding')g.d.createBindGroup=()=>{throw Error('binding');};
  if(failure==='compile')g.d.createRenderPipelineAsync=async()=>{throw Error('compile');};
  await assert.rejects(r.addMesh(gpu(),{shading:'metallic-roughness',clearcoatFactor:1,
    ...(failure==='compile'?{clearcoatTexture:texture(),texCoords:UV}:{})}),new RegExp(failure));
  assert.equal(r.allocatedBytes,before);assert.equal(r.meshCount,1);
  g.d.createBuffer=create;g.d.createBindGroup=bind;g.d.createRenderPipelineAsync=compile;
  r.render(frame([m]));r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

test('changing source coating parameters during async compilation does not alter the captured material',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{maxDraws:1}),ready=[];
  g.d.createRenderPipelineAsync=x=>new Promise(resolve=>ready.push(()=>resolve(x)));
  const material={shading:'metallic-roughness',clearcoatFactor:0.25},pending=r.addMesh(gpu(),material);
  material.clearcoatFactor=99;ready.forEach(f=>f());const m=await pending;r.render(frame([m]));
  assert.equal(new Float32Array(g.buffers.find(b=>b.label.endsWith('/clearcoat')).data)[0],0.25);r.dispose();
});
