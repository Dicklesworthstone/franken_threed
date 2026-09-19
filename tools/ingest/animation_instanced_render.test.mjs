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

// Separate registrations are how decoded glTF instances reach the renderer.
// Their immutable packed streams must converge too, not just repeated handles.
test('1000 separately registered indexed textured meshes fit one set of immutable material streams and one draw',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:1000,maxMeshes:1000,maxBytes:256080,label:'arena'});
  const vertex={},map=texture(),meshes=[];
  for(let i=0;i<1000;i++)meshes.push(await r.addMesh(gpu(vertex),{
    indices:[0,1,2],texCoords:[...UV],baseColorTexture:{...map},baseColor:[i/1000,1,1,1],
  }));
  assert.equal(r.allocatedBytes,256080);assert.equal(g.buffers.length,3);
  r.render(frame(meshes));assert.deepEqual(calls(g).map(x=>x.args),[[3,1000,0,0,0]]);
  assert.equal(g.groups.length,2);assert.equal(r.drawCount,1000);assert.equal(r.drawCallCount,1);
  meshes[0].dispose();assert.equal(r.allocatedBytes,256080);r.render(frame(meshes.slice(1)));
  assert.deepEqual(calls(g)[0].args,[3,999,0,0,0]);
  meshes.slice(1).forEach(m=>m.dispose());assert.equal(r.allocatedBytes,256000);r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

test('all five separately loaded maps and UV streams batch with independent PBR factors',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2,environment:true,label:'arena'}),vertex={},maps=fields.map(texture);
  const meshes=[];
  for(let n=0;n<2;n++) {
    const material={shading:'metallic-roughness',indices:[0,1,2],texCoords:[...UV],mapCoordinates:{},occlusionStrength:n,metallicFactor:n/2};
    fields.forEach((name,i)=>{material[name]={...maps[i]};material.mapCoordinates[name]={texCoords:[...UV],uvTransform:[i+1,0,0,1,0.25,0]};});
    meshes.push(await r.addMesh(gpu(vertex),material));
  }
  assert.equal(r.allocatedBytes,512+8+192+608);r.render(frame(meshes,{environment:{map:environment(g.d)}}));
  assert.equal(r.drawCallCount,1);assert.equal(words(g)[51],0);assert.equal(words(g)[64+51],1);
  assert.equal(words(g)[23],0);assert.equal(words(g)[64+23],0.5);r.dispose();
});

test('byte comparison distinguishes signed-zero UVs and different vertex colors',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:3}),vertex={},map=texture(),meshes=[];
  for(const [zero,color] of [[0,1],[-0,1],[0,0.5]])meshes.push(await r.addMesh(gpu(vertex),{
    indices:[0,1,2],texCoords:[zero,0,1,0,0,1],vertexColors:Array(9).fill(color),baseColorTexture:map,
  }));
  assert.equal(g.buffers.filter(b=>b.label.endsWith('/indices')).length,1);
  assert.equal(g.buffers.filter(b=>b.label.endsWith('/surface')).length,3);
  r.render(frame(meshes));assert.equal(r.drawCallCount,3);r.dispose();
});

test('index order and sampler identity split batches even when other streams agree',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:4}),vertex={},map=texture(),meshes=[];
  for(const [indices,sampler] of [[[0,1,2],map.sampler],[[0,2,1],map.sampler],[[0,1,2],{}],[[0,1,2],map.sampler]])
    meshes.push(await r.addMesh(gpu(vertex),{indices,texCoords:[...UV],baseColorTexture:{view:map.view,sampler}}));
  assert.equal(g.buffers.filter(b=>b.label.endsWith('/surface')).length,1);
  r.render(frame(meshes));assert.equal(r.drawCallCount,4);assert.deepEqual(calls(g).map(d=>d.args.at(-1)),[0,1,2,3]);r.dispose();
});

test('material array/descriptor mutation after registration cannot change canonical streams or bindings',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),vertex={},map=texture();
  const first={indices:[0,1,2],texCoords:[...UV],baseColorTexture:{...map}},a=await r.addMesh(gpu(vertex),first);
  first.indices.reverse();first.texCoords.fill(99);first.baseColorTexture.view={};
  const b=await r.addMesh(gpu(vertex),{indices:[0,1,2],texCoords:UV,baseColorTexture:map});
  r.render(frame([a,b]));assert.equal(r.drawCallCount,1);
  assert.equal(calls(g)[0].bindings.get(1).group.entries[1].resource,map.view);r.dispose();
});

test('failed texture-group creation releases only the failing registration, never its successful siblings',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),vertex={},material={indices:[0,1,2],texCoords:UV,baseColorTexture:texture()};
  const a=await r.addMesh(gpu(vertex),material),bytes=r.allocatedBytes,create=g.d.createBindGroup;
  g.d.createBindGroup=()=>{throw Error('binding failed');};
  await assert.rejects(r.addMesh(gpu(vertex),{...material,baseColorTexture:texture()}),/binding failed/);
  assert.equal(r.allocatedBytes,bytes);assert.equal(r.meshCount,1);assert.ok(g.buffers.every(b=>b.destroyed===0));
  g.d.createBindGroup=create;r.render(frame([a]));assert.equal(r.failed,false);a.dispose();assert.equal(r.allocatedBytes,512);r.dispose();
});

test('failed pipeline validation never installs a shared stream for later registrations',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),compile=g.d.createRenderPipelineAsync;
  g.d.createRenderPipelineAsync=async()=>{throw Error('pipeline failed');};
  const material={indices:[0,1,2],texCoords:UV,baseColorTexture:texture()};
  await assert.rejects(r.addMesh(gpu(),material),/pipeline failed/);assert.equal(r.meshCount,0);assert.equal(r.allocatedBytes,512);
  assert.ok(g.buffers.slice(1).every(b=>b.destroyed===1));g.d.createRenderPipelineAsync=compile;
  const a=await r.addMesh(gpu(),material);r.render(frame([a,a]));assert.equal(r.drawCallCount,1);r.dispose();
});

test('concurrent registrations merge only after validation, and retire duplicate private allocations',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),vertex={},material={indices:[0,1,2],texCoords:UV,baseColorTexture:texture()},resolvers=[];
  g.d.createRenderPipelineAsync=desc=>new Promise(resolve=>resolvers.push(()=>resolve(desc)));
  const pending=[r.addMesh(gpu(vertex),material),r.addMesh(gpu(vertex),material)];
  assert.equal(g.buffers.length,5);assert.equal(r.meshCount,0);
  resolvers.forEach(resolve=>resolve());const meshes=await Promise.all(pending);
  assert.equal(r.allocatedBytes,592);assert.equal(g.buffers.filter(b=>b.destroyed===1).length,2);
  r.render(frame(meshes));assert.equal(r.drawCallCount,1);r.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));
});

test('an outstanding registration retains a shared stream after its original mesh is disposed',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),vertex={},material={indices:[0,1,2],texCoords:UV},a=await r.addMesh(gpu(vertex),material),resolvers=[];
  g.d.createRenderPipelineAsync=desc=>new Promise(resolve=>resolvers.push(()=>resolve(desc)));
  const pending=r.addMesh(gpu(vertex),{...material,shading:'lambert'});a.dispose();
  assert.ok(g.buffers.filter(b=>b.label.endsWith('/surface')||b.label.endsWith('/indices')).every(b=>b.destroyed===0));
  resolvers.forEach(resolve=>resolve());const b=await pending;r.render(frame([b,b]));assert.equal(r.drawCallCount,1);r.dispose();
});

test('disposing the renderer during registration releases published and pending allocations exactly once',async()=>{
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),a=await r.addMesh(gpu(),{indices:[0,1,2]}),resolvers=[];
  g.d.createRenderPipelineAsync=desc=>new Promise(resolve=>resolvers.push(()=>resolve(desc)));
  const pending=r.addMesh(gpu(),{indices:[0,1,2],texCoords:UV});r.dispose();resolvers.forEach(resolve=>resolve());
  await assert.rejects(pending,code('DISPOSED'));assert.equal(r.allocatedBytes,0);a.dispose();
  assert.ok(g.buffers.every(b=>b.destroyed===1));
});

test('the exact unique-stream byte budget succeeds and one byte short fails before material allocation',async()=>{
  const material={indices:[0,1,2],texCoords:UV};
  for(const budget of [591,592]) {
    const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2,maxBytes:budget});
    if(budget===591){await assert.rejects(r.addMesh(gpu(),material),code('LIMIT'));assert.equal(g.buffers.length,1);}
    else {await r.addMesh(gpu(),material);await r.addMesh(gpu(),material);assert.equal(r.allocatedBytes,budget);}
    r.dispose();
  }
});

test('a real surface-stream hash collision cannot merge different packed UV coordinates',async()=>{
  // These two first-UV pairs have the same byte-wise FNV1a hash (4183942755)
  // after ordinary 72-byte UV/color packing; the remaining coordinates match.
  const pairs=[[1062715850,1060363805],[1060661827,1064948635]].map(words=>new Float32Array(new Uint32Array(words).buffer));
  const g=device(),r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:2}),shared={},map=texture(),meshes=[];
  for(const pair of pairs)meshes.push(await r.addMesh(gpu(shared),{texCoords:[...pair,1,0,0,1],baseColorTexture:map}));
  r.render(frame(meshes));assert.equal(r.drawCallCount,2);
  const surfaces=g.buffers.filter(b=>b.label.endsWith('/surface'));assert.equal(surfaces.length,2);
  const hash=data=>{let h=2166136261;for(const b of new Uint8Array(data))h=Math.imul(h^b,16777619)>>>0;return h;};
  assert.equal(hash(surfaces[0].data),4183942755);assert.equal(hash(surfaces[1].data),4183942755);
  assert.notDeepEqual(new Uint8Array(surfaces[0].data),new Uint8Array(surfaces[1].data));r.dispose();
});

test('the production rigid pool and material renderer batch separately registered textured geometry together',async()=>{
  const {createGpuRigidGeometryPool}=await import('./animation_rigid_geometry.mjs');
  const count=64,pose={nodeCount:count,version:0,disposed:false,instances:[],morphOffsets:new Uint32Array(count+1),worldMatrices:new Float64Array(count*16)};
  for(let i=0;i<count;i++){pose.worldMatrices.set(I(),i*16);pose.worldMatrices[i*16+12]=i;}
  const g=device(),pool=createGpuRigidGeometryPool(g.d,pose,{maxBytes:120,maxMeshes:count}),
    r=await createGpuAnimationRenderer(g.d,{instancing:true,maxDraws:count,label:'arena'}),map=texture(),handles=[],meshes=[];
  for(let i=0;i<count;i++){
    const handle=await pool.addMesh({node:i,positions:[0,0,0,1,0,0,0,1,0],normals:[0,0,1,0,0,1,0,0,1]});
    handles.push(handle);meshes.push(await r.addMesh(handle,{shading:'metallic-roughness',indices:[0,1,2],texCoords:UV,baseColorTexture:map}));
  }
  assert.equal(pool.uniqueGeometries,1);assert.equal(pool.bufferBytes,120);
  assert.equal(r.allocatedBytes,count*256+8+72+544);assert.equal(g.buffers.length,5);
  r.render(frame(meshes));await r.whenIdle();assert.deepEqual(calls(g).map(x=>x.args),[[3,count,0,0,0]]);
  const previous=words(g).slice(),before=g.counts();pose.version++;
  for(let i=0;i<count;i++){pose.worldMatrices[i*16+12]+=10;handles[i].update();}
  assert.deepEqual(g.counts(),before,'rigid pose updates require no vertex upload or submission');
  r.render(frame(meshes));await r.whenIdle();
  for(let i=0;i<count;i++){assert.equal(previous[i*64+12],i);assert.equal(words(g)[i*64+12],i+10);}
  // Each layer owns only its resources: retiring draw records leaves the shared
  // vertex buffer live until the last actual rigid geometry reference is released.
  meshes.forEach(m=>m.dispose());assert.equal(pool.bufferBytes,120);
  assert.equal(handles[0].vertexBuffer.destroyed,0);handles.forEach(h=>h.dispose());assert.equal(pool.bufferBytes,0);
  r.dispose();pool.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));assert.equal(pose.disposed,false);
});
