import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationRenderer} from './animation_render.mjs';

const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const uv = () => [0,0,1,0,0,1];
const code = expected => error => error.code === expected;
const fields = ['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture'];
const descriptor = name => ({view:{name:name+'/view'},sampler:{name:name+'/sampler'}});
const deferred = () => {let resolve; const promise = new Promise(r=>{resolve=r;});return {promise,resolve};};
function geometry(tangents = true) {
  return {vertexCount:3,vertexBuffer:{},worldMatrix:identity(),disposed:false,failed:false,whenIdle:async()=>{},
    vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'},
      {shaderLocation:1,offset:12,format:'float32x3'},
      ...(tangents?[{shaderLocation:2,offset:24,format:'float32x4'}]:[]),
    ]}};
}
// Records WebGPU API calls and immutable submission snapshots. This does not
// execute WGSL or emulate pixels; actual-device checks live in the browser runner.
function deviceSpy() {
  const loss=deferred(),buffers=[],pipelines=[],groups=[],writes=[],submissions=[],scopes=[];
  const d={buffers,pipelines,groups,writes,submissions,scopes,loss,lost:loss.promise,
    limits:{minUniformBufferOffsetAlignment:256,maxBufferSize:1<<28,maxUniformBufferBindingSize:65536,
      maxDynamicUniformBuffersPerPipelineLayout:8,maxBindGroups:4,maxUniformBuffersPerShaderStage:12,
      maxVertexBuffers:8,maxVertexAttributes:16,maxVertexBufferArrayStride:2048,
      maxSamplersPerShaderStage:16,maxSampledTexturesPerShaderStage:16},
    pushErrorScope(value){scopes.push(value);},popErrorScope(){assert.ok(scopes.pop());return Promise.resolve(null);},
    createBuffer(options){const data=new ArrayBuffer(options.size);const b={...options,data,destroyed:false,
      getMappedRange:()=>data,unmap(){},destroy(){b.destroyed=true;}};buffers.push(b);return b;},
    createBindGroupLayout:options=>options,createPipelineLayout:options=>options,createShaderModule:options=>options,
    createBindGroup(options){groups.push(options);return options;},
    createRenderPipelineAsync(options){pipelines.push(options);return Promise.resolve(options);},
    createCommandEncoder(){const draws=[];let pipeline;const bound=new Map(),vertices=new Map();
      return {beginRenderPass(){return {
        setPipeline(value){pipeline=value;},setBindGroup(slot,group,offset=[0]){bound.set(slot,{group,offset:offset[0]});},
        setVertexBuffer(slot,value){vertices.set(slot,value);},setIndexBuffer(){},setViewport(){},setScissorRect(){},
        draw(...args){draws.push({pipeline,bound:new Map(bound),vertices:new Map(vertices),args});},
        drawIndexed(...args){draws.push({pipeline,bound:new Map(bound),vertices:new Map(vertices),args,indexed:true});},end(){},
      };},finish:()=>draws};},
    queue:{writeBuffer(buffer,offset,data,start=0,size=data.length-start){
      const n=data.BYTES_PER_ELEMENT??1,bytes=new Uint8Array(data.buffer,data.byteOffset+start*n,size*n);
      new Uint8Array(buffer.data,offset,bytes.length).set(bytes);writes.push({buffer,offset,bytes:bytes.slice()});
    },submit(commands){for(const draws of commands)for(const draw of draws){const u=draw.bound.get(0);
      draw.uniforms=new Float32Array(u.group.entries[0].resource.buffer.data,u.offset,64).slice();submissions.push(draw);}},
    onSubmittedWorkDone:()=>Promise.resolve()},
  };return d;
}
const frame = draws => ({colorView:{},depthView:{},viewProjection:identity(),draws,
  lighting:{cameraPosition:[0,0,5],lights:[{type:'directional',direction:[0,0,-1]}]}});
const options = mask => ({shading:'metallic-roughness',...(mask?{texCoords:uv()}:{}),
  ...Object.fromEntries(fields.flatMap((field,slot)=>mask&(1<<slot)?[[field,descriptor(field)]]:[]))});

for(let mask=0;mask<16;mask++)test(`map combination ${mask}: only requested resource pairs and tangent inputs`,async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),material=options(mask),mesh=await r.addMesh(geometry(),material);
  r.render(frame([mesh]));await r.whenIdle();const draw=d.submissions.at(-1),p=draw.pipeline;
  const attrs=p.vertex.buffers[0].attributes;
  assert.equal(attrs.some(a=>a.shaderLocation===2),!!(mask&4));
  assert.equal(p.layout.bindGroupLayouts.length,mask?3:2);
  if(mask){
    const group=draw.bound.get(1).group,slots=[0,1,2,3].filter(slot=>mask&(1<<slot));
    assert.deepEqual(group.entries.map(e=>e.binding),slots.flatMap(slot=>[slot*2,slot*2+1]));
    assert.equal(group.layout,p.layout.bindGroupLayouts[1]);
    for(const slot of slots){assert.equal(group.entries.find(e=>e.binding===slot*2).resource,material[fields[slot]].sampler);
      assert.equal(group.entries.find(e=>e.binding===slot*2+1).resource,material[fields[slot]].view);}
  }
  const shader=p.fragment.module.code;
  assert.equal((shader.match(/textureSample\(/g)??[]).length,fields.filter((_,i)=>mask&(1<<i)).length);
  assert.ok(shader.lastIndexOf('textureSample(')<shader.indexOf('discard;'));
  assert.equal(shader.includes('* metallic_roughness_texel.b'),!!(mask&2));
  assert.equal(shader.includes('* metallic_roughness_texel.g'),!!(mask&2));
  assert.equal(shader.includes('* emissive_texel.rgb'),!!(mask&8));
  assert.equal(r.allocatedBytes,256+544+(mask?72:0));
  r.dispose();assert.ok(d.buffers.every(b=>b.destroyed));assert.ok(Object.values(material).every(v=>!v?.view?.destroyed));
});

test('normal scale, handedness, UV and material factors get separate per-draw snapshots',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:2}),mesh=await r.addMesh(geometry(),{...options(15),normalScale:0.25});
  const reflected=identity();reflected[0]=-2;reflected[5]=3;
  r.render(frame([{mesh,uvTransform:[2,0,0,3,4,5]},
    {mesh,normalScale:-2,worldMatrix:reflected,metallicFactor:0.2,roughnessFactor:0.3,emissiveFactor:[0.1,0.2,0.3]}]));
  const [a,b]=d.submissions;assert.equal(a.bound.get(0).offset,0);assert.equal(b.bound.get(0).offset,256);
  assert.equal(a.uniforms[27],0.25);assert.equal(a.uniforms[31],1);
  assert.deepEqual([...a.uniforms.slice(24,27)],[2,0,4]);assert.deepEqual([...a.uniforms.slice(28,31)],[0,3,5]);
  assert.equal(b.uniforms[27],-2);assert.equal(b.uniforms[31],-1);assert.equal(b.pipeline.primitive.frontFace,'cw');
  assert.equal(b.uniforms[23],Math.fround(0.2));assert.equal(b.uniforms[63],Math.fround(0.3));
  r.render(frame([{mesh,normalScale:0}]));assert.equal(d.submissions.at(-1).uniforms[27],0);
  assert.equal(a.uniforms[27],0.25,'A later write cannot overwrite a submitted use');r.dispose();
});

test('descriptors are snapshotted across async initialization, layouts and pipelines reused',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),wait=deferred(),compile=d.createRenderPipelineAsync;
  d.createRenderPipelineAsync=o=>{compile(o);return wait.promise.then(()=>o);};
  const material=options(14),views=fields.slice(1).map(f=>material[f].view);
  const pending=r.addMesh(geometry(),material);for(const f of fields.slice(1))material[f].view={mutated:true};
  wait.resolve();const mesh=await pending;r.render(frame([mesh]));
  const group=d.submissions[0].bound.get(1).group;
  assert.deepEqual(group.entries.filter(e=>e.binding%2).map(e=>e.resource),views);
  const count=d.pipelines.length,next=await r.addMesh(geometry(),options(14));assert.equal(d.pipelines.length,count);
  r.render(frame([next]));assert.equal(d.submissions.at(-1).bound.get(1).group.layout,group.layout);r.dispose();
});

test('all map feature/geometry/capability validation precedes owned GPU allocation',async()=>{
  for(const change of [
    (m,g,d)=>{m.shading='unlit';},(m,g,d)=>{m.shading='lambert';},(m,g,d)=>{delete m.texCoords;},
    (m,g,d)=>{g.vertexLayout.attributes.pop();},(m,g,d)=>{m.normalScale=Infinity;},
    (m,g,d)=>{m.normalScale=1e300;},(m,g,d)=>{m.normalTexture.sampler=null;},
    (m,g,d)=>{m.emissiveTexture.texCoord=1;},(m,g,d)=>{d.limits.maxSamplersPerShaderStage=3;},
    (m,g,d)=>{d.limits.maxSampledTexturesPerShaderStage=3;},(m,g,d)=>{d.limits.maxBindGroups=2;},
  ]){
    const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),g=geometry(),m=options(15);change(m,g,d);
    await assert.rejects(r.addMesh(g,m));assert.equal(d.buffers.length,1);assert.equal(d.pipelines.length,6);assert.equal(r.meshCount,0);r.dispose();
  }
});

test('normal mapping cannot silently ignore absent tangent attributes or extraneous scale',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d);
  await assert.rejects(r.addMesh(geometry(false),options(4)),code('ANIMATION_RENDER_NORMAL'));
  await assert.rejects(r.addMesh(geometry(),{...options(8),normalScale:0}),code('ANIMATION_RENDER_OPTIONS'));
  const mesh=await r.addMesh(geometry(),options(8));assert.throws(()=>r.render(frame([{mesh,normalScale:0}])),code('ANIMATION_RENDER_OPTIONS'));
  assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);r.render(frame([mesh]));r.dispose();
});

test('an invalid later mapped draw causes no partial frame and leaves renderer reusable',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(geometry(),options(4));
  for(const normalScale of [NaN,Infinity,1e100,{},'1']){
    assert.throws(()=>r.render(frame([mesh,{mesh,normalScale}])));assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);assert.equal(r.version,0);
  }
  r.render(frame([mesh]));await r.whenIdle();assert.equal(r.version,1);r.dispose();
});

test('Lambert composes normal and emissive maps without introducing a metallic map',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(geometry(),{...options(12),shading:'lambert'});
  r.render(frame([mesh]));await r.whenIdle();assert.equal(d.submissions[0].uniforms[22],1);
  assert.deepEqual(d.submissions[0].bound.get(1).group.entries.map(e=>e.binding),[4,5,6,7]);r.dispose();
});

test('base-color-only and untextured materials retain the original layout and budget',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:328});
  const mesh=await r.addMesh(geometry(),{texCoords:uv(),baseColorTexture:descriptor('color')});
  r.render(frame([mesh]));assert.equal(r.allocatedBytes,328);assert.equal(d.submissions[0].pipeline.layout.bindGroupLayouts.length,2);
  assert.equal(d.submissions[0].pipeline.vertex.buffers[0].attributes.length,1);mesh.dispose();
  const plain=await r.addMesh(geometry());r.render(frame([plain]));assert.equal(r.allocatedBytes,256);r.dispose();
});

test('normal and emissive maps need no new owned buffers beyond existing UV and light blocks',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:872});
  const mesh=await r.addMesh(geometry(),options(15));assert.equal(r.allocatedBytes,872);
  await assert.rejects(r.addMesh(geometry(),options(4)),code('ANIMATION_RENDER_LIMIT'));
  mesh.dispose();assert.equal(r.allocatedBytes,800);const next=await r.addMesh(geometry(),options(12));
  r.render(frame([next]));assert.equal(r.allocatedBytes,872);r.dispose();
});

test('a mapped pipeline rejection releases mesh buffers and allows a later retry',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),compile=d.createRenderPipelineAsync;
  d.createRenderPipelineAsync=()=>Promise.reject(new Error('mapped shader rejected'));
  await assert.rejects(r.addMesh(geometry(),options(15)),/mapped shader rejected/);
  assert.equal(r.meshCount,0);assert.equal(r.allocatedBytes,800);assert.ok(d.buffers.filter(b=>b.label.endsWith('/surface')).every(b=>b.destroyed));
  d.createRenderPipelineAsync=compile;const mesh=await r.addMesh(geometry(),options(15));r.render(frame([mesh]));await r.whenIdle();r.dispose();
});

test('device loss during pending map compilation releases only owned buffers',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),wait=deferred();d.createRenderPipelineAsync=()=>wait.promise;
  const g=geometry(),m=options(15),pending=r.addMesh(g,m);d.loss.resolve({message:'lost during maps'});
  await assert.rejects(pending,code('ANIMATION_RENDER_LOST'));assert.equal(r.failed,true);assert.ok(d.buffers.every(b=>b.destroyed));
  assert.equal(g.disposed,false);assert.equal(m.normalTexture.view.destroyed,undefined);wait.resolve({});r.dispose();
});
