import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationRenderer} from './animation_render.mjs';
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };
const code = expected => error => error.code === expected;
function gpu(overrides = {}) {
  return {vertexBuffer: {name: 'deformed-vertices'}, vertexCount: 3, worldMatrix: new Float64Array(identity()),
    vertexLayout: {arrayStride: 40, stepMode: 'vertex', attributes: [{shaderLocation: 0, offset: 0, format: 'float32x3'}]},
    version: 0, poseVersion: 0, disposed: false, failed: false, whenIdle: async () => {}, ...overrides};
}
function deviceSpy() {
  const loss = deferred(), buffers = [], pipelines = [], passes = [], writes = [], submissions = [], scopes = [];
  const device = {loss, buffers, pipelines, passes, writes, submissions, scopes, lost: loss.promise,
    limits: {minUniformBufferOffsetAlignment: 256, maxBufferSize: 64 * 1024 * 1024, maxUniformBufferBindingSize: 65536,
      maxDynamicUniformBuffersPerPipelineLayout: 8, maxBindGroups: 4, maxUniformBuffersPerShaderStage: 12,
      maxVertexBuffers: 8, maxVertexAttributes: 16, maxVertexBufferArrayStride: 2048,
      maxSamplersPerShaderStage: 16, maxSampledTexturesPerShaderStage: 16},
    pushErrorScope(filter) { scopes.push(filter); },
    popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createBuffer(descriptor) {
      const data = new ArrayBuffer(descriptor.size), buffer = {...descriptor, data, destroyed: false,
        getMappedRange() { assert.equal(this.mappedAtCreation,true); return data; },
        unmap() { this.mappedAtCreation=false; }, destroy() { this.destroyed=true; }};
      buffers.push(buffer); return buffer;
    },
    createBindGroupLayout: descriptor => descriptor,
    createPipelineLayout: descriptor => descriptor,
    createBindGroup: descriptor => descriptor,
    createShaderModule: descriptor => descriptor,
    createRenderPipelineAsync(descriptor) { assert.equal(scopes.length,2); pipelines.push(descriptor); return Promise.resolve(descriptor); },
    createCommandEncoder() {
      const encoded = [];
      return {beginRenderPass(descriptor) {
        const pass = {descriptor, draws: [], viewport: null, scissor: null}; let pipeline, uniform, vertex, index, surface, texture;
        passes.push(pass); encoded.push(pass);
        return {setPipeline(p) {pipeline=p;}, setBindGroup(slot, group, offsets=[]) {if(slot===0)uniform={group,offset:offsets[0]};else texture=group;},
          setVertexBuffer(slot,buffer) {if(slot===0)vertex=buffer;else surface=buffer;}, setIndexBuffer(buffer,format) {index={buffer,format};},
          draw(...args) {pass.draws.push({args,indexed:false,pipeline,uniform,vertex,surface,texture});},
          drawIndexed(...args) {pass.draws.push({args,indexed:true,pipeline,uniform,vertex,index,surface,texture});},
          setViewport(...values) {pass.viewport=values;}, setScissorRect(...values) {pass.scissor=values;}, end() {pass.ended=true;}};
      }, finish() {return encoded;}};
    },
    queue: {
      writeBuffer(buffer, offset, data, dataOffset=0, size=data.length-dataOffset) {
        const bytes = data.BYTES_PER_ELEMENT ?? 1;
        const input = new Uint8Array(data.buffer, data.byteOffset + dataOffset * bytes, size * bytes);
        writes.push({buffer, offset, bytes: input.slice()}); new Uint8Array(buffer.data,offset,input.length).set(input);
      },
      submit(commands) {
        for (const passes of commands) for (const pass of passes) for (const draw of pass.draws) {
          const buffer=draw.uniform.group.entries[0].resource.buffer;
          draw.snapshot=new Float32Array(buffer.data,draw.uniform.offset,32).slice();
        }
        submissions.push(commands);
      },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };
  return device;
}
const frame = (draws, overrides={}) => ({colorView:{color:true},depthView:{depth:true},viewProjection:identity(),draws,...overrides});

test('builds reusable pipelines and aligned bounded uniform storage without browser globals',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:3});
  assert.equal(d.buffers[0].size,768);assert.equal(d.buffers[0].usage,72);assert.equal(d.pipelines.length,6);
  assert.equal(d.scopes.length,0);assert.equal(r.version,0);assert.equal(r.meshCount,0);
  for(const p of d.pipelines) {
    assert.equal(p.vertex.buffers[0].arrayStride,40);assert.equal(p.vertex.buffers[0].attributes.length,1);
    assert.equal(p.depthStencil.depthWriteEnabled,!p.fragment.targets[0].blend);
    assert.equal(p.layout.bindGroupLayouts[0].entries[0].buffer.hasDynamicOffset,true);
    assert.equal(p.layout.bindGroupLayouts[0].entries[0].buffer.minBindingSize,128);
  }
  r.dispose();assert.ok(d.buffers.every(b=>b.destroyed));
});

test('per-draw matrix/color snapshots preserve red A and blue B in one submission',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),g=gpu(),mesh=await r.addMesh(g);
  const a=identity(),b=identity();a[12]=-0.5;b[12]=0.5;
  r.render(frame([{mesh,worldMatrix:a,baseColor:[1,0,0,1]},{mesh,worldMatrix:b,baseColor:[0,0,1,1]}]));
  await r.whenIdle();const draws=d.passes[0].draws;
  assert.deepEqual(draws.map(x=>x.uniform.offset),[0,256]);assert.equal(draws[0].snapshot[12],-0.5);assert.equal(draws[1].snapshot[12],0.5);
  assert.deepEqual([...draws[0].snapshot.slice(16,20)],[1,0,0,1]);assert.deepEqual([...draws[1].snapshot.slice(16,20)],[0,0,1,1]);
  assert.equal(d.writes.length,1);assert.equal(d.writes[0].bytes.length,384);assert.equal(r.drawCount,2);assert.equal(r.version,1);
  r.render(frame([{mesh,baseColor:[0,1,0,1]}]));assert.deepEqual([...draws[0].snapshot.slice(16,20)],[1,0,0,1]);
  assert.equal(d.buffers.length,1,'No buffers allocated during frames');r.dispose();assert.equal(g.disposed,false);
});

test('uploads/pads small index arrays and selects uint32 without truncation',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),source=new Uint8Array([0,1,2]);
  const mesh=await r.addMesh(gpu(),{indices:source});source.fill(2);
  assert.equal(mesh.indexCount,3);assert.equal(d.buffers[1].size,8);assert.equal(d.buffers[1].usage,16);
  assert.deepEqual([...new Uint16Array(d.buffers[1].data)],[0,1,2,0]);
  r.render(frame([{mesh,first:1,count:2}]));
  const draw=d.passes[0].draws[0];assert.equal(draw.index.format,'uint16');assert.deepEqual(draw.args,[2,1,1,0,0]);
  const big=await r.addMesh(gpu({vertexCount:70000}),{indices:[0,65536,69999]});r.render(frame([big]));
  assert.equal(d.passes[1].draws[0].index.format,'uint32');assert.deepEqual([...new Uint32Array(d.buffers[2].data)],[0,65536,69999]);r.dispose();
});

test('nonindexed ranges and reflected/double-sided winding are explicit',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),g=gpu({vertexCount:6});
  const mesh=await r.addMesh(g),two=await r.addMesh(g,{doubleSided:true}),reflected=identity();reflected[0]=-1;
  r.render(frame([{mesh,first:3,count:3},{mesh,worldMatrix:reflected},{mesh:two,worldMatrix:reflected}]));
  const draws=d.passes[0].draws;assert.deepEqual(draws[0].args,[3,1,3,0]);assert.equal(draws[0].indexed,false);
  assert.equal(draws[0].pipeline.primitive.frontFace,'ccw');assert.equal(draws[1].pipeline.primitive.frontFace,'cw');
  assert.equal(draws[2].pipeline.primitive.cullMode,'none');r.dispose();
});

test('opaque, alpha-mask and straight-alpha blend use correct uniforms and depth state',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),g=gpu();
  const opaque=await r.addMesh(g,{baseColor:[0.5,0.5,0.5,0.2]});
  const mask=await r.addMesh(g,{alphaMode:'MASK',alphaCutoff:0.25});const blend=await r.addMesh(g,{alphaMode:'BLEND'});
  r.render(frame([opaque,mask,blend]));const [a,b,c]=d.passes[0].draws;
  assert.equal(a.snapshot[20],-1);assert.equal(a.snapshot[21],0);assert.equal(a.pipeline.depthStencil.depthWriteEnabled,true);
  assert.equal(b.snapshot[20],0.25);assert.equal(b.snapshot[21],0);assert.equal(b.pipeline.depthStencil.depthWriteEnabled,true);
  assert.equal(c.snapshot[21],1);assert.equal(c.pipeline.depthStencil.depthWriteEnabled,false);
  assert.equal(c.pipeline.fragment.targets[0].blend.color.srcFactor,'src-alpha');assert.equal(c.pipeline.fragment.targets[0].blend.alpha.srcFactor,'one');r.dispose();
});

test('multi-sample render targets, load preservation, viewport and scissor stay caller owned',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{sampleCount:4,format:'bgra8unorm-srgb'}),mesh=await r.addMesh(gpu());
  const colorView={},depthView={},resolveTarget={};
  r.render(frame([mesh],{colorView,depthView,resolveTarget,loadOp:'load',depthLoadOp:'load',viewport:[1,2,30,40,0,1],scissor:[1,2,20,30]}));
  const p=d.passes[0];assert.equal(p.descriptor.colorAttachments[0].view,colorView);assert.equal(p.descriptor.colorAttachments[0].resolveTarget,resolveTarget);
  assert.equal(p.descriptor.colorAttachments[0].loadOp,'load');assert.equal(p.descriptor.depthStencilAttachment.depthLoadOp,'load');
  assert.deepEqual(p.viewport,[1,2,30,40,0,1]);assert.deepEqual(p.scissor,[1,2,20,30]);assert.equal(d.pipelines[0].multisample.count,4);
  r.dispose();assert.deepEqual(colorView,{});assert.deepEqual(depthView,{});
});

test('clear-only submissions and explicit no-depth mode require no geometry',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{depthFormat:null});r.render(frame([],{depthView:null,clearColor:[0.25,0.5,0.75,1]}));await r.whenIdle();
  assert.equal(d.writes.length,0);assert.equal(d.submissions.length,1);assert.equal(d.passes[0].descriptor.depthStencilAttachment,undefined);
  assert.deepEqual(d.passes[0].descriptor.colorAttachments[0].clearValue,{r:0.25,g:0.5,b:0.75,a:1});
  assert.throws(()=>r.render(frame([])),code('ANIMATION_RENDER_ATTACHMENT'));r.dispose();
});

test('bad final draw and invalid frame values cause no GPU effects or partial publication',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:2}),mesh=await r.addMesh(gpu());
  const bad=identity();bad[0]=NaN;const overflow=identity();overflow[0]=1e300;
  for(const input of [frame([mesh,{mesh,worldMatrix:bad}]),frame([{mesh,worldMatrix:overflow}]),frame([{mesh,count:4}]),
      frame([mesh],{clearColor:[0,0,0,2]}),frame([mesh],{resolveTarget:{}}),frame([mesh],{depthView:null}),frame([mesh,mesh,mesh]),
      frame([mesh],{viewport:[0,0,-1,1,0,1]}),frame([mesh],{scissor:[0,0,1.5,2]})]) {
    assert.throws(()=>r.render(input));assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);assert.equal(d.passes.length,0);assert.equal(r.version,0);
  }
  r.render(frame([mesh]));await r.whenIdle();assert.equal(r.version,1);r.dispose();
});

test('index validation, material admissions and allocation budgets reject before GPU allocations',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:264});const g=gpu();
  for(const options of [{indices:[0,1,3]},{indices:[0,-1,2]},{indices:[0,0.5,2]},{indices:[]},{baseColor:[1,1,1,2]},
      {map:{}},{metalness:1},{alphaMode:'bad'},{doubleSided:1},{indices:[0,1,2,0,1,2]}]) {
    await assert.rejects(r.addMesh(g,options));assert.equal(d.buffers.length,1);assert.equal(r.meshCount,0);
  }
  const a=await r.addMesh(g,{indices:[0,1,2]});await assert.rejects(r.addMesh(g,{indices:[0,1,2]}),code('ANIMATION_RENDER_LIMIT'));
  a.dispose();assert.equal(d.buffers[1].destroyed,true);const b=await r.addMesh(g,{indices:[0,1,2]});assert.equal(b.disposed,false);r.dispose();
});

test('material data is snapshotted and camera/world matrix multiplication is column-major',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),g=gpu(),rgba=[0.25,0.5,0.75,1],mesh=await r.addMesh(g,{baseColor:rgba});
  rgba.fill(0);g.worldMatrix[12]=2;g.worldMatrix[13]=3;const vp=identity();vp[0]=4;vp[5]=5;vp[14]=6;
  r.render(frame([mesh],{viewProjection:vp}));const u=d.passes[0].draws[0].snapshot;
  assert.deepEqual([...u.slice(12,16)],[8,15,6,1]);assert.deepEqual([...u.slice(16,20)],[0.25,0.5,0.75,1]);r.dispose();
});

test('foreign, disposed and failed borrowed meshes are not submitted',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),other=await createGpuAnimationRenderer(d),g=gpu();
  const a=await r.addMesh(g),b=await other.addMesh(g);assert.throws(()=>r.render(frame([{mesh:b}])),code('ANIMATION_RENDER_MESH'));
  g.failed=true;assert.throws(()=>r.render(frame([a])),code('ANIMATION_RENDER_GEOMETRY'));g.failed=false;
  a.dispose();a.dispose();assert.equal(r.meshCount,0);assert.equal(g.disposed,false);assert.throws(()=>r.render(frame([a])),code('ANIMATION_RENDER_MESH'));
  r.dispose();r.dispose();assert.throws(()=>r.render(frame([])),code('ANIMATION_RENDER_DISPOSED'));
  await assert.rejects(r.addMesh(g),code('ANIMATION_RENDER_DISPOSED'));other.dispose();
});

test('reentrant frame getters cannot dispose or submit during snapshotting',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(gpu());
  const draw={mesh,get worldMatrix(){r.dispose();return identity();}};
  assert.throws(()=>r.render(frame([draw])),code('ANIMATION_RENDER_REENTRANT'));assert.equal(r.disposed,false);assert.equal(d.submissions.length,0);
  r.render(frame([mesh]));r.dispose();
});

test('whenIdle includes earlier error scopes despite later successful queue completion',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(gpu()),early=deferred();const pop=d.popErrorScope.bind(d);let first=true;
  d.popErrorScope=()=>{const value=pop();if(first){first=false;return early.promise;}return value;};
  r.render(frame([mesh]));r.render(frame([mesh]));let settled=false;
  const pending=r.whenIdle().then(()=>{settled=true;},error=>{settled=true;return error;});
  await new Promise(resolve=>setImmediate(resolve));const premature=settled;early.resolve({message:'first draw invalid'});
  const error=await pending;assert.equal(premature,false);assert.equal(error.code,'ANIMATION_RENDER_DEVICE');assert.equal(r.failed,true);
  assert.throws(()=>r.render(frame([mesh])),code('ANIMATION_RENDER_DEVICE'));r.dispose();
});

test('deformation failure is not hidden by successful render queue acknowledgement',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),error=new Error('bad deformation'),g=gpu({whenIdle:()=>Promise.reject(error)}),mesh=await r.addMesh(g);
  r.render(frame([mesh]));await assert.rejects(r.whenIdle(),e=>e===error);assert.equal(r.failed,true);r.dispose();
});

test('synchronous submission failure is terminal and publishes no version',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(gpu()),error=new Error('submit failed');d.queue.submit=()=>{throw error;};
  assert.throws(()=>r.render(frame([mesh])),e=>e===error);assert.equal(r.version,0);assert.equal(r.failed,true);
  await assert.rejects(r.whenIdle(),e=>e===error);r.dispose();
});

test('pipeline rejection and device loss release all owned storage',async()=>{
  const d=deviceSpy();d.createRenderPipelineAsync=()=>Promise.reject(new Error('shader failed'));
  await assert.rejects(createGpuAnimationRenderer(d),/shader failed/);assert.ok(d.buffers.every(b=>b.destroyed));assert.equal(d.scopes.length,0);
  const next=deviceSpy(),r=await createGpuAnimationRenderer(next),g=gpu(),mesh=await r.addMesh(g,{indices:[0,1,2]});
  next.loss.resolve({message:'device disconnected'});await new Promise(resolve=>setImmediate(resolve));
  await assert.rejects(r.whenIdle(),code('ANIMATION_RENDER_LOST'));assert.ok(next.buffers.every(b=>b.destroyed));assert.equal(g.disposed,false);assert.equal(r.failed,true);r.dispose();assert.equal(mesh.disposed,true);
});

test('failed index registration is cleaned up and the renderer remains usable',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),pop=d.popErrorScope.bind(d);let first=true;
  d.popErrorScope=()=>{const value=pop();if(first){first=false;return Promise.resolve({message:'allocation failed'});}return value;};
  await assert.rejects(r.addMesh(gpu(),{indices:[0,1,2]}),code('ANIMATION_RENDER_DEVICE'));assert.equal(r.meshCount,0);assert.equal(d.buffers[1].destroyed,true);
  const mesh=await r.addMesh(gpu());r.render(frame([mesh]));await r.whenIdle();r.dispose();
});

test('pending registrations count against capacity and dispose safely during initialization',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxMeshes:1}),slow=deferred(),pop=d.popErrorScope.bind(d);let first=true;
  d.popErrorScope=()=>{const value=pop();if(first){first=false;return slow.promise;}return value;};
  const pending=r.addMesh(gpu(),{indices:[0,1,2]});await assert.rejects(r.addMesh(gpu()),code('ANIMATION_RENDER_LIMIT'));
  r.dispose();slow.resolve(null);await assert.rejects(pending,code('ANIMATION_RENDER_DISPOSED'));assert.ok(d.buffers.every(b=>b.destroyed));
});

test('capabilities and initial uniform budget fail before allocating',async()=>{
  for(const [options,change] of [[{maxDraws:0},()=>{}],[{maxBytes:100},()=>{}],[{format:'depth24plus'},()=>{}],
      [{},d=>{d.limits.maxUniformBufferBindingSize=64;}],[{},d=>{d.limits.maxDynamicUniformBuffersPerPipelineLayout=0;}]]) {
    const d=deviceSpy();change(d);await assert.rejects(createGpuAnimationRenderer(d,options));assert.equal(d.buffers.length,0);
  }
});

const uvTriangle = () => [0,0, 1,0, 0,1];
const texture = () => ({view: {}, sampler: {}});
test('textured draws snapshot UV/RGBA data once and bind each material without dummy textures',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),g=gpu();
  const tex=texture(),uv=uvTriangle(),colors=[1,0,0,0.5, 0,1,0,1, 0,0,1,0.25];
  const a=await r.addMesh(g,{texCoords:uv,vertexColors:colors,baseColorTexture:tex,alphaMode:'MASK'});
  const other=texture(),b=await r.addMesh(g,{texCoords:uvTriangle(),baseColorTexture:other});
  uv.fill(9);colors.fill(9);tex.view={replacement:true};
  r.render(frame([a,b]));await r.whenIdle();const [first,second]=d.passes[0].draws;
  assert.equal(first.vertex,g.vertexBuffer);assert.equal(first.surface.usage,32);
  assert.deepEqual([...new Float32Array(first.surface.data)],[0,0,1,0,0,0.5, 1,0,0,1,0,1, 0,1,0,0,1,0.25]);
  assert.notEqual(first.texture.entries[1].resource,tex.view);assert.equal(second.texture.entries[1].resource,other.view);
  assert.equal(first.pipeline,second.pipeline);assert.equal(d.pipelines.length,12,'one cached texture pipeline family');
  assert.deepEqual(first.pipeline.vertex.buffers[1].attributes.map(a=>a.shaderLocation),[3,4]);
  assert.equal(first.pipeline.layout.bindGroupLayouts[1].entries[1].texture.sampleType,'float');
  const code=first.pipeline.fragment.module.code;
  assert.ok(code.indexOf('textureSample(')<code.indexOf('discard;'));
  assert.match(code,/rgba\.a < draw_info\.options\.x/);assert.match(code,/draw_info\.color \* input\.color/);
  const count=d.buffers.length;r.render(frame([a,b]));assert.equal(d.buffers.length,count);r.dispose();
});

test('RGB vertex colors gain opaque alpha and require no sampler or texture capability',async()=>{
  const d=deviceSpy();d.limits.maxSamplersPerShaderStage=0;d.limits.maxSampledTexturesPerShaderStage=0;
  const r=await createGpuAnimationRenderer(d),colors=[1,0,0, 0,1,0, 0,0,1];
  const mesh=await r.addMesh(gpu(),{vertexColors:colors});r.render(frame([mesh]));
  const draw=d.passes[0].draws[0],packed=[...new Float32Array(draw.surface.data)];
  assert.deepEqual(packed,[0,0,1,0,0,1, 0,0,0,1,0,1, 0,0,0,0,1,1]);
  assert.doesNotMatch(draw.pipeline.fragment.module.code,/textureSample|@group\(1\)/);
  assert.equal(draw.texture,undefined);r.dispose();
});

test('independent affine UV transforms are snapshotted at each submitted use',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),transform=[2,3,4,5,6,7];
  const mesh=await r.addMesh(gpu(),{texCoords:uvTriangle(),baseColorTexture:texture(),uvTransform:transform});
  transform.fill(0);r.render(frame([mesh,{mesh,uvTransform:[0,1,-1,0,0.25,0.5]}]));
  const [a,b]=d.passes[0].draws;
  assert.deepEqual([...a.snapshot.slice(24)],[2,4,6,0,3,5,7,0]);
  assert.deepEqual([...b.snapshot.slice(24)],[0,-1,0.25,0,1,0,0.5,0]);
  r.render(frame([{mesh,uvTransform:[1,0,0,1,9,9]}]));
  assert.deepEqual([...a.snapshot.slice(24)],[2,4,6,0,3,5,7,0]);r.dispose();
});

test('invalid surface descriptors and capabilities fail before device allocation',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d);
  for(const options of [{baseColorTexture:texture()},{baseColorTexture:{view:{}},texCoords:uvTriangle()},
    {baseColorTexture:{...texture(),flipY:true},texCoords:uvTriangle()},{texCoords:[1,2]},
    {texCoords:[0,0,0,0,0,Infinity]},{texCoords:[0,0,0,0,0,1e300]},
    {vertexColors:[1,2,3]},{vertexColors:[1,1,1,2,1,1,1,1,1,1,1,1]},
    {texCoords:uvTriangle(),uvTransform:[1,2]}]) {
    await assert.rejects(r.addMesh(gpu(),options));assert.equal(d.buffers.length,1);assert.equal(r.meshCount,0);
  }
  d.limits.maxSamplersPerShaderStage=0;
  await assert.rejects(r.addMesh(gpu(),{texCoords:uvTriangle(),baseColorTexture:texture()}),code('ANIMATION_RENDER_LIMIT'));
  assert.equal(d.buffers.length,1);r.dispose();
});

test('invalid last UV transform leaves all prior frame work unsubmitted',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(gpu(),{vertexColors:Array(9).fill(1)});
  assert.throws(()=>r.render(frame([mesh,{mesh,uvTransform:[1,0,0,1,NaN,0]}])),code('ANIMATION_RENDER_VALUE'));
  assert.equal(d.writes.length,0);assert.equal(d.passes.length,0);assert.equal(r.version,0);
  r.render(frame([mesh]));await r.whenIdle();r.dispose();
});

test('surface plus index allocations are jointly bounded and reclaimed per mesh',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:336}),t=texture();
  const a=await r.addMesh(gpu(),{indices:[0,1,2],texCoords:uvTriangle(),baseColorTexture:t});
  assert.equal(r.allocatedBytes,336);
  await assert.rejects(r.addMesh(gpu(),{vertexColors:Array(9).fill(1)}),code('ANIMATION_RENDER_LIMIT'));
  a.dispose();assert.equal(r.allocatedBytes,256);assert.ok(d.buffers.slice(1).every(b=>b.destroyed));
  const b=await r.addMesh(gpu(),{vertexColors:Array(9).fill(1)});assert.equal(r.allocatedBytes,328);
  r.dispose();assert.equal(b.disposed,true);assert.deepEqual(t,{view:{},sampler:{}});assert.equal(r.allocatedBytes,0);
});

test('pending surface pipelines reserve mesh/buffer capacity and unwind on failure',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:328}),slow=deferred();
  d.createRenderPipelineAsync=()=>slow.promise;
  const pending=r.addMesh(gpu(),{vertexColors:Array(9).fill(1)});
  assert.equal(r.allocatedBytes,328);
  await assert.rejects(r.addMesh(gpu(),{vertexColors:Array(9).fill(1)}),code('ANIMATION_RENDER_LIMIT'));
  slow.reject(new Error('surface shader rejected'));await assert.rejects(pending,/surface shader rejected/);
  assert.equal(r.allocatedBytes,256);assert.equal(r.meshCount,0);assert.ok(d.buffers.slice(1).every(b=>b.destroyed));
  const plain=await r.addMesh(gpu());r.render(frame([plain]));await r.whenIdle();r.dispose();
});

test('surface bind-group validation errors clean up attributes without owning the borrowed texture',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),bind=d.createBindGroup,t=texture();
  d.createBindGroup=descriptor=>{if(descriptor.entries.length===2)throw new Error('invalid texture view');return bind(descriptor);};
  await assert.rejects(r.addMesh(gpu(),{texCoords:uvTriangle(),baseColorTexture:t}),/invalid texture view/);
  assert.equal(r.meshCount,0);assert.equal(r.allocatedBytes,262144);assert.ok(d.buffers.slice(1).every(b=>b.destroyed));
  assert.deepEqual(t,{view:{},sampler:{}});r.dispose();
});
