import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationRenderer} from './animation_render.mjs';
const I = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const UV = [0,0,1,0,0,1];
const texture = () => ({view:{},sampler:{}});
const fields = ['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture','occlusionTexture'];
// Only the device boundary is recorded. The renderer and both lighting/shadow
// receivers are production modules; these assertions are not native GPU pixels.
function device(limits = {}) {
  const buffers=[],pipelines=[],groups=[],passes=[],writes=[],submissions=[];let depth=0,lose;
  const d={limits:{minUniformBufferOffsetAlignment:256,maxBufferSize:2**27,maxUniformBufferBindingSize:65536,
    maxStorageBuffersPerShaderStage:8,maxStorageBufferBindingSize:2**27,maxDynamicUniformBuffersPerPipelineLayout:8,
    maxBindGroups:4,maxUniformBuffersPerShaderStage:12,maxVertexBuffers:8,maxVertexAttributes:16,
    maxVertexBufferArrayStride:2048,maxInterStageShaderVariables:16,maxSamplersPerShaderStage:16,
    maxSampledTexturesPerShaderStage:16,maxTextureDimension2D:4096,...limits},lost:new Promise(r=>{lose=r;}),
    pushErrorScope(){depth++;},popErrorScope(){assert.ok(depth>0);depth--;return Promise.resolve(null);},
    createBuffer(desc){const data=new ArrayBuffer(desc.size),b={...desc,data,destroyed:0,
      getMappedRange:()=>data,unmap(){},destroy(){this.destroyed++;}};buffers.push(b);return b;},
    createBindGroupLayout:x=>x,createPipelineLayout:x=>x,createShaderModule:x=>x,
    createBindGroup(x){groups.push(x);return x;},async createRenderPipelineAsync(x){pipelines.push(x);return x;},
    createCommandEncoder(){const command={passes:[]};return {beginRenderPass(desc){
      const p={desc,draws:[]};passes.push(p);command.passes.push(p);let pipeline,index;const bindings=new Map(),vertices=new Map();
      const draw=(indexed,args)=>p.draws.push({pipeline,index,bindings:new Map(bindings),vertices:new Map(vertices),indexed,args});
      return {setPipeline(x){pipeline=x;},setBindGroup(i,g,offsets=[]){bindings.set(i,{group:g,offsets});},
        setVertexBuffer(i,b){vertices.set(i,b);},setIndexBuffer(b,format){index={buffer:b,format};},
        setViewport(){},setScissorRect(){},draw(...args){draw(false,args);},drawIndexed(...args){draw(true,args);},end(){}};
    },finish:()=>command};},
    queue:{writeBuffer(buffer,offset,value,start=0,length){
      const width=value.BYTES_PER_ELEMENT??1,storage=ArrayBuffer.isView(value)?value.buffer:value;
      const bytes=new Uint8Array(storage,(value.byteOffset??0)+start*width,(length??(value.byteLength/width-start))*width).slice();
      new Uint8Array(buffer.data,offset,bytes.length).set(bytes);writes.push({buffer,offset,bytes});
    },submit(commands){submissions.push(commands);},onSubmittedWorkDone:async()=>{}},
  };
  return {d,buffers,pipelines,groups,passes,writes,submissions,lose,
    counts:()=>[buffers.length,pipelines.length,groups.length,writes.length,submissions.length],get depth(){return depth;}};
}
const gpu=(vertexBuffer={},tangents=true)=>({vertexBuffer,vertexCount:3,worldMatrix:I(),disposed:false,failed:false,whenIdle:async()=>{},
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},
    {shaderLocation:1,offset:12,format:'float32x3'},...(tangents?[{shaderLocation:2,offset:24,format:'float32x4'}]:[])]}});
const frame=(draws,extra={})=>({colorView:{},depthView:{},viewProjection:I(),draws,
  lighting:{cameraPosition:[0,0,3],lights:[{type:'directional'}]},...extra});
const calls=g=>g.passes.at(-1).draws;
const words=g=>new Float32Array(g.writes.findLast(w=>w.buffer.label==='arena').bytes.buffer);
const code=name=>({code:'ANIMATION_RENDER_'+name});
function environment(d){const s=Object.freeze({profile:'f3d-animation-environment-v1',version:1,
  diffuseView:{},specularView:{},brdfView:{},sampler:{},mipLevelCount:3});
  return {sample(other){assert.equal(other,d);return s;},whenIdle:async()=>{}};}
function shadow(d){const s=Object.freeze({view:{},sampler:{},version:1,width:8,height:8,viewProjection:I()});
  return {sample(other){assert.equal(other,d);return s;},whenIdle:async()=>{}};}

test('1000 independently transformed meshes sharing vertices submit one native instanced draw',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:1000,label:'arena'}),buffer={},draws=[];
  for(let i=0;i<1000;i++){const d=gpu(buffer);d.worldMatrix[12]=i;draws.push(await r.addMesh(d,{baseColor:[i/1000,0.5,1,1]}));}
  r.render(frame(draws));assert.deepEqual(calls(g).map(x=>x.args),[[3,1000,0,0]]);
  assert.equal(r.drawCount,1000);assert.equal(r.drawCallCount,1);assert.equal(r.allocatedBytes,256000);
  const packed=words(g);for(let i=0;i<1000;i++) {assert.equal(packed[i*64+12],i);assert.equal(packed[i*64+16],Math.fround(i/1000));}
  const group=calls(g)[0].bindings.get(0);assert.deepEqual(group.offsets,[]);
  assert.equal(group.group.entries[0].resource.size,256000);assert.equal(g.buffers[0].usage,136);
  assert.equal(g.depth,0);await r.whenIdle();r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

for(const shading of ['unlit','lambert','metallic-roughness'])for(const indexed of [false,true])
  test(`${shading} ${indexed?'indexed':'unindexed'} repeated mesh keeps per-instance color, matrix and material packets`,async()=>{
    const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:3,label:'arena'});
    const mesh=await r.addMesh(gpu(),{shading,...(indexed?{indices:[0,1,2]}:{})});
    const draws=[0,1,2].map(i=>{const worldMatrix=I();worldMatrix[12]=i;worldMatrix[0]=i+1;return {mesh,worldMatrix,baseColor:[i/3,1,0.5,1],
      ...(shading==='unlit'?{}:{emissiveFactor:[i,i*2,i*3]}),...(shading==='metallic-roughness'?{metallicFactor:i/2,roughnessFactor:i/3}:{})};});
    r.render(frame(draws));assert.deepEqual(calls(g)[0].args,indexed?[3,3,0,0,0]:[3,3,0,0]);
    const packed=words(g);for(let i=0;i<3;i++) {
      assert.equal(packed[i*64],i+1);assert.equal(packed[i*64+12],i);assert.equal(packed[i*64+16],Math.fround(i/3));
      if(shading!=='unlit'){assert.equal(packed[i*64+48],Math.fround(1/(i+1)));assert.equal(packed[i*64+60],i);}
      if(shading==='metallic-roughness'){assert.equal(packed[i*64+23],i/2);assert.equal(packed[i*64+63],Math.fround(i/3));}
    }
    draws[0].worldMatrix.fill(99);assert.equal(words(g)[12],0);r.dispose();
  });

for(const tangents of [false,true])for(const shadows of [false,true])for(const ibl of [false,true])
  test(`five-map instanced PBR retains ${tangents?'authored':'derivative'} normals, shadow=${shadows}, environment=${ibl}`,async()=>{
    const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,shadows,environment:ibl,maxDraws:2,label:'arena'});
    const material={shading:'metallic-roughness',alphaMode:'MASK',texCoords:UV,mapCoordinates:{}};
    for(const name of fields){material[name]=texture();material.mapCoordinates[name]={texCoords:UV,uvTransform:[2,0,0,3,0.25,0.5]};}
    const mesh=await r.addMesh(gpu({},tangents),material);
    r.render(frame([{mesh,normalScale:0.25,occlusionStrength:0},{mesh,normalScale:2,occlusionStrength:0.75}],{
      ...(shadows?{shadow:{map:shadow(g.d)}}:{}),...(ibl?{environment:{map:environment(g.d)}}:{})}));
    assert.deepEqual(calls(g).map(x=>x.args),[[3,2,0,0]]);const packed=words(g);
    assert.equal(packed[27],0.25);assert.equal(packed[64+27],2);assert.equal(packed[51],0);assert.equal(packed[64+51],0.75);
    const shader=calls(g)[0].pipeline.vertex.module.code;
    assert.match(shader,/@builtin\(instance_index\) draw_index: u32/);
    assert.match(shader,/@location\(10\) @interpolate\(flat\) draw_index: u32/);
    assert.match(shader,/draw_info = instance_draws\[draw_index\]\.info/);
    assert.match(shader,/draw_info = instance_draws\[input.draw_index\]\.info/);
    assert.match(shader,/normal_from_local: OcclusionNormal/);assert.equal(shader.includes('dpdx(input.world)'),!tangents);
    assert.equal(shader.includes('fn projected_shadow'),shadows);assert.equal(shader.includes('fn environment_lighting'),ibl);
    assert.equal(calls(g)[0].bindings.get(1).group.entries.length,10);await r.whenIdle();r.dispose();
  });

test('batch boundaries preserve original order, geometry, draw ranges, winding and BLEND isolation',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:12,label:'arena'}),a=await r.addMesh(gpu()),b=await r.addMesh(gpu()),
    blend=await r.addMesh(gpu(),{alphaMode:'BLEND'}),reflected=I();reflected[0]=-1;
  const draws=[a,a,b,a,{mesh:a,worldMatrix:reflected},{mesh:a,worldMatrix:reflected},blend,blend,
    {mesh:a,first:1,count:2},{mesh:a,first:1,count:2},a];
  r.render(frame(draws));assert.deepEqual(calls(g).map(x=>x.args),[[3,2,0,0],[3,1,0,2],[3,1,0,3],[3,2,0,4],
    [3,1,0,6],[3,1,0,7],[2,2,1,8],[3,1,0,10]]);
  assert.equal(calls(g)[3].pipeline.primitive.frontFace,'cw');assert.equal(r.drawCount,11);assert.equal(r.drawCallCount,8);r.dispose();
});

test('incompatible sampler/view resources cannot be merged',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:4}),shared={},a=await r.addMesh(gpu(shared),{texCoords:UV,baseColorTexture:texture()}),
    b=await r.addMesh(gpu(shared),{texCoords:UV,baseColorTexture:texture()});
  r.render(frame([a,a,b,b]));assert.deepEqual(calls(g).map(x=>x.args),[[3,2,0,0],[3,2,0,2]]);r.dispose();
});

for(const mapped of [false,true])test(`depth-only ${mapped?'masked':'plain'} variants use instance packets without color output`,async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{format:null,instancing:true,maxDraws:2});
  const m=await r.addMesh(gpu(),mapped?{texCoords:UV,baseColorTexture:texture(),alphaMode:'MASK'}:{});
  r.render(frame([m,m],{colorView:undefined}));assert.deepEqual(calls(g)[0].args,[3,2,0,0]);
  assert.deepEqual(calls(g)[0].pipeline.fragment.targets,[]);assert.equal(g.passes[0].desc.colorAttachments.length,0);r.dispose();
});

test('storage packet stride honors device uniform alignment without indexing padding as a draw',async()=>{
  const g=device({minUniformBufferOffsetAlignment:512}),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2,label:'arena'});
  const mesh=await r.addMesh(gpu());r.render(frame([mesh,{mesh,baseColor:[0.25,0.5,0.75,1]}]));
  assert.equal(words(g)[128+16],0.25);assert.match(calls(g)[0].pipeline.vertex.module.code,/@size\(512\) info: DrawInfo/);
  assert.equal(r.allocatedBytes,1024);r.dispose();
});

for(const limits of [{maxStorageBuffersPerShaderStage:0},{maxStorageBufferBindingSize:511},{maxInterStageShaderVariables:10}])
  test(`insufficient instancing device limits reject before allocation: ${JSON.stringify(limits)}`,async()=>{
    const g=device(limits);await assert.rejects(createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),code('LIMIT'));
    assert.deepEqual(g.counts(),[0,0,0,0,0]);
    const ordinary=await createGpuAnimationRenderer(g.d,{maxDraws:2});assert.equal(ordinary.instancing,false);ordinary.dispose();
  });

test('logical draw and byte limits are not bypassed by a single batched call',async()=>{
  const g=device();await assert.rejects(createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2,maxBytes:511}),code('LIMIT'));
  const r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2,maxBytes:512}),m=await r.addMesh(gpu());
  const before=g.counts();assert.throws(()=>r.render(frame([m,m,m])),code('LIMIT'));assert.deepEqual(g.counts(),before);
  r.render(frame([m,m]));assert.equal(r.drawCallCount,1);r.dispose();
});

test('a late invalid draw publishes no partial batch and preserves prior submission statistics',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:3}),m=await r.addMesh(gpu());
  r.render(frame([m,m]));const before=g.counts(),version=r.version;
  assert.throws(()=>r.render(frame([m,{mesh:m,baseColor:[NaN,0,0,1]}])),code('VALUE'));
  assert.deepEqual(g.counts(),before);assert.equal(r.version,version);assert.equal(r.drawCount,2);assert.equal(r.drawCallCount,1);
  r.render(frame([]));assert.equal(r.drawCount,0);assert.equal(r.drawCallCount,0);r.dispose();
});

test('every coalesced deformer remains a completion dependency',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),shared={},a=gpu(shared),b=gpu(shared);let completed=0;
  a.whenIdle=async()=>{completed++;};b.whenIdle=async()=>{completed++;throw Error('second deformer failed');};
  const x=await r.addMesh(a),y=await r.addMesh(b);r.render(frame([x,y]));assert.equal(r.drawCallCount,1);
  await assert.rejects(r.whenIdle(),/second deformer failed/);assert.equal(completed,2);assert.equal(r.failed,true);r.dispose();
});

test('device loss destroys the instance arena once and remains observable',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),m=await r.addMesh(gpu());
  r.render(frame([m,m]));g.lose({message:'device gone'});await assert.rejects(r.whenIdle(),code('LOST'));
  assert.ok(g.buffers.every(b=>b.destroyed===1));r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

test('separate render submissions snapshot their own instance packets and do not defer queue consumption',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2,label:'arena'}),d=gpu(),m=await r.addMesh(d);
  r.render(frame([m,m]));d.worldMatrix[12]=7;r.render(frame([m,m]));
  const packets=g.writes.filter(w=>w.buffer.label==='arena');assert.equal(packets.length,2);
  assert.equal(new Float32Array(packets[0].bytes.buffer)[12],0);assert.equal(new Float32Array(packets[1].bytes.buffer)[12],7);
  assert.equal(g.submissions.length,2);await r.whenIdle();r.dispose();
});

test('uint32 indexed batches keep their index format and nonzero first-instance addressing',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:3}),d=gpu();d.vertexCount=65537;
  const m=await r.addMesh(d,{indices:[0,65536,1]}),other=await r.addMesh(gpu());
  r.render(frame([other,m,m]));assert.deepEqual(calls(g)[1].args,[3,2,0,0,1]);assert.equal(calls(g)[1].index.format,'uint32');r.dispose();
});

test('disabled instancing retains separate uniform bindings and ordinary shader modules',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{maxDraws:2}),m=await r.addMesh(gpu());r.render(frame([m,m]));
  assert.deepEqual(calls(g).map(x=>x.args),[[3,1,0,0],[3,1,0,0]]);
  assert.deepEqual(calls(g).map(x=>x.bindings.get(0).offsets),[[0],[256]]);
  assert.doesNotMatch(calls(g)[0].pipeline.vertex.module.code,/instance_index|instance_draws/);assert.equal(g.buffers[0].usage,72);
  assert.equal(r.drawCount,2);assert.equal(r.drawCallCount,2);r.dispose();
});
