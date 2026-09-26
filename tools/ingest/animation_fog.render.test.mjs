import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer,gpuFixture,frame,events,linearFog,expFog,identity,deferred} from './animation_fog_render_fixture.mjs';
const code=s=>({code:'ANIMATION_RENDER_'+s});
const writes=(h)=>events(h,'write').filter(e=>e.value.buffer.label.endsWith('/fog')).map(e=>e.value);

test('fog allocates exactly 48 bytes beyond the unchanged 256-byte arena and binds both shader stages',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,maxDraws:1,maxBytes:304});
  assert.equal(r.allocatedBytes,304);assert.equal(r.fog,true);assert.deepEqual(h.state.buffers.map(b=>b.size),[48,256]);
  const group=h.state.groups[0];assert.equal(group.entries[1].resource.buffer,h.state.buffers[0]);
  assert.deepEqual(group.layout.entries[1],{binding:1,visibility:3,buffer:{type:'uniform',minBindingSize:48}});
  assert.equal(group.layout.entries[0].buffer.minBindingSize,256);r.dispose();assert.ok(h.state.buffers.every(b=>b.destroyed===1));
});
test('fog-off keeps the cheap plain shader, resource counts and old minimum limits',async()=>{
  const h=gpuFixture();h.device.limits.maxUniformBuffersPerShaderStage=1;delete h.device.limits.maxInterStageShaderVariables;delete h.device.limits.maxBindingsPerBindGroup;
  const r=await createGpuAnimationRenderer(h.device,{maxDraws:1,maxBytes:256});assert.equal(r.allocatedBytes,256);assert.equal(h.state.groups[0].entries.length,1);
  assert.doesNotMatch(h.state.shaders[0],/FogInfo|VertexOutput|fog_depth/);r.dispose();
});
for(const [name,options,limits] of [['one-byte-short',{maxBytes:303},{}],['uniform bindings',{}, {maxUniformBuffersPerShaderStage:1}],
  ['inter-stage locations',{}, {maxInterStageShaderVariables:14}],['group bindings',{}, {maxBindingsPerBindGroup:1}]])
  test('fog '+name+' limit rejects before allocating native buffers',async()=>{
    const h=gpuFixture();Object.assign(h.device.limits,limits);await assert.rejects(createGpuAnimationRenderer(h.device,{fog:true,maxDraws:1,...options}),code('LIMIT'));
    assert.equal(h.state.buffers.length,0);assert.equal(h.state.pipelines.length,0);
  });
test('invalid options and depth-only fog refuse before native allocation',async()=>{
  for(const options of [{fog:{}},{fog:1},{fog:true,format:null}]){const h=gpuFixture();await assert.rejects(createGpuAnimationRenderer(h.device,options),code('OPTIONS'));assert.equal(h.state.buffers.length,0);}
});
test('linear, exp2 and disabled frames write separate snapshots before consuming submissions',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,maxDraws:2}),mesh=await r.addMesh(h.deformer()),input=linearFog();
  r.render(frame([mesh],{fog:input}));input.color[0]=1;r.render(frame([mesh],{fog:expFog()}));r.render(frame([mesh]));await r.whenIdle();
  const out=writes(h);assert.equal(out.length,3);assert.equal(out[0].data[4],Math.fround(.2));assert.deepEqual(out.map(w=>w.data[11]),[1,2,0]);
  assert.deepEqual(out[2].data,Array(12).fill(0));assert.equal(r.drawCallCount,1);assert.equal(events(h,'pass').length,3);assert.equal(h.state.buffers.length,2);
  assert.deepEqual(h.state.events.filter(e=>e.type==='submit'||(e.type==='write'&&e.value.bytes===48)).map(e=>e.type),['write','submit','write','submit','write','submit']);r.dispose();
});
test('invalid fog or a later malformed draw cannot partly update the frame',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,maxDraws:2}),mesh=await r.addMesh(h.deformer());const before=h.state.events.length;
  assert.throws(()=>r.render(frame([mesh],{fog:linearFog({near:20,far:2})})),{code:'ANIMATION_FOG_VALUE'});
  assert.throws(()=>r.render(frame([mesh,{mesh:{}}],{fog:linearFog()})),code('MESH'));assert.equal(h.state.events.length,before);assert.equal(r.failed,false);
  r.render(frame([mesh],{fog:linearFog()}));r.dispose();
});
test('a non-enabled renderer refuses fog descriptors without writes',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device),mesh=await r.addMesh(h.deformer()),before=h.state.events.length;
  assert.throws(()=>r.render(frame([mesh],{fog:linearFog()})),code('OPTIONS'));assert.equal(h.state.events.length,before);r.dispose();
});
for(const shading of ['unlit','lambert','phong','toon','metallic-roughness'])for(const alphaMode of ['OPAQUE','MASK','BLEND'])
  test(`${shading} ${alphaMode}: fog after shading preserves alpha, coverage, draw count and fixed depth state`,async()=>{
    const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,maxDraws:1});const mesh=await r.addMesh(h.deformer(),{shading,alphaMode,baseColor:[.5,.5,.5,.4]});
    r.render(frame([mesh],{fog:linearFog(),...(shading==='unlit'?{}:{lighting:h.lighting})}));await r.whenIdle();
    const p=events(h,'set-pipeline').at(-1).value,s=p.vertex.module.code;
    assert.match(s,/@location\(14\) fog_depth: f32/);assert.match(s,/out.fog_depth = dot\(fog_info.depth_from_clip, out.position\)/);
    assert.match(s,/vec4<f32>\(apply_distance_fog\(rgb, input.fog_depth\), select\(1.0, rgba.a, draw_info.options.y > 0.0\)\)/);
    assert.ok(s.lastIndexOf('discard;')<s.lastIndexOf('apply_distance_fog(rgb,'));
    assert.equal(p.depthStencil.depthWriteEnabled,alphaMode!=='BLEND');assert.equal(!!p.fragment.targets[0].blend,alphaMode==='BLEND');assert.equal(r.drawCallCount,1);r.dispose();
  });
test('normal, AO, independent UV, clearcoat, IBL and shadows coexist without binding or varying collisions',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,environment:true,shadows:true,instancing:true,maxDraws:3});
  const names=['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture','occlusionTexture','clearcoatTexture','clearcoatRoughnessTexture','clearcoatNormalTexture'];
  const texCoords=Array(12).fill(.5),options={shading:'metallic-roughness',clearcoatFactor:.5,texCoords,mapCoordinates:{}};
  for(const name of names){options[name]=h.binding();options.mapCoordinates[name]={texCoords};}
  const mesh=await r.addMesh(h.deformer(),options);r.render(frame([mesh,mesh],{fog:expFog(),lighting:h.lighting,shadow:h.shadow,environment:h.environment}));await r.whenIdle();
  const s=events(h,'set-pipeline').at(-1).value.vertex.module.code;assert.match(s,/uv_7/);assert.match(s,/@location\(13\) @interpolate\(flat\) draw_index/);
  assert.ok(s.indexOf('rgb = rgb * (1.0 - weight)')<s.lastIndexOf('apply_distance_fog(rgb,'));assert.match(s,/projected_shadow\(/);assert.match(s,/environment_lighting\(/);
  assert.equal(r.drawCount,2);assert.equal(r.drawCallCount,1);assert.equal(writes(h)[0].bytes,48);r.dispose();
});
for(const instancing of [false,true])test('native source instances retain a fog binding with logical instancing '+instancing,async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,instancing,maxDraws:2}),g=h.geometry(),instances=h.instances(4);
  const mesh=await r.addMesh(g,{instances,shading:'phong'});r.render(frame([mesh],{fog:linearFog(),lighting:h.lighting}));await r.whenIdle();
  assert.deepEqual(events(h,'draw').at(-1).value,[6,4,0,0]);const group=events(h,'set-group').find(e=>e.value.index===0).value.group;
  assert.equal(group.layout.entries[0].buffer.type,'uniform');assert.equal(group.entries[1].resource.size,48);
  assert.match(events(h,'set-pipeline').at(-1).value.vertex.module.code,/instance_position/);r.dispose();
});
test('render bundle reuse retains live fog, including disabling it without building new schedules',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,renderBundles:true,maxDraws:2}),mesh=await r.addMesh(h.deformer());
  for(const fog of [linearFog(),expFog(),null])r.render(frame([mesh],{fog}));await r.whenIdle();
  assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,2);assert.equal(r.bundleDiagnostics.executions,3);
  assert.deepEqual(writes(h).map(w=>w.data[11]),[1,2,0]);assert.equal(r.drawCallCount,1);r.dispose();
});
test('fog does not alter viewport, scissor, indexed range, multisample resolve or attachment policy',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true,sampleCount:4}),mesh=await r.addMesh(h.deformer(),{indices:[0,1,2,2,3,0]});
  const f=frame([{mesh,first:3,count:3}],{fog:linearFog(),resolveTarget:{},loadOp:'load',depthLoadOp:'load',viewport:[1,2,16,32,0,1],scissor:[2,3,4,5]});
  r.render(f);assert.deepEqual(events(h,'draw-indexed').at(-1).value,[3,1,3,0,0]);
  assert.equal(events(h,'pass')[0].value.colorAttachments[0].resolveTarget,f.resolveTarget);assert.equal(events(h,'pass')[0].value.depthStencilAttachment.depthLoadOp,'load');
  assert.deepEqual(events(h,'viewport')[0].value,f.viewport);assert.deepEqual(events(h,'scissor')[0].value,f.scissor);r.dispose();
});
test('device loss during pending completion releases the fog buffer with the draw arena',async()=>{
  const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true}),mesh=await r.addMesh(h.deformer()),gate=deferred();h.state.queueGate=gate.promise;
  r.render(frame([mesh],{fog:expFog()}));const idle=r.whenIdle();h.loss.resolve({message:'lost'});await assert.rejects(idle,code('LOST'));assert.ok(h.state.buffers.every(b=>b.destroyed===1));
});
test('failed construction releases the private fog buffer and balances native scopes',async()=>{
  const h=gpuFixture();h.state.nextError={message:'invalid pipeline'};await assert.rejects(createGpuAnimationRenderer(h.device,{fog:true}),code('DEVICE'));
  assert.ok(h.state.buffers.every(b=>b.destroyed===1));assert.equal(h.state.scopes.length,0);
});
test('native fog write failure and async validation become terminal and are drained honestly',async()=>{
  for(const sync of [false,true]){const h=gpuFixture(),r=await createGpuAnimationRenderer(h.device,{fog:true}),mesh=await r.addMesh(h.deformer());
    if(sync)h.state.throwAt='write';else h.state.nextError={message:'bad draw'};
    if(sync)assert.throws(()=>r.render(frame([mesh],{fog:linearFog()})),/native write/);
    else {r.render(frame([mesh],{fog:linearFog()}));await assert.rejects(r.whenIdle(),code('DEVICE'));}
    assert.equal(r.failed,true);r.dispose();assert.ok(h.state.buffers.every(b=>b.destroyed===1));assert.equal(h.state.scopes.length,0);
  }
});
