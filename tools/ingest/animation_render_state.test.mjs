import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const frame=(draws,lighting)=>({colorView:{},depthView:{},viewProjection:I(),draws,lighting});
function geometry(d){return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),whenIdle:async()=>{},
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},
    {shaderLocation:1,offset:12,format:'float32x3'}]}};}
for(const shading of ['unlit','phong'])for(const renderBundles of [false,true])
  test(`native material state survives bundles and reflection (${shading}/${renderBundles})`,async()=>{
    const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{renderBundles,instancing:true}),g=geometry(d);
    const m=await r.addMesh(g,{shading,side:'back',depthWrite:false,depthCompare:'greater',colorWrite:false});
    const reflected=I();reflected[0]=-1;
    const lights={viewDirection:[0,0,1],lights:[{type:'directional'}]};
    r.render(frame([m,{mesh:m,worldMatrix:reflected}],lights));
    r.render(frame([m,{mesh:m,worldMatrix:reflected}],lights));
    const [a,b]=d.snapshots[0];
    assert.equal(a.pipeline.primitive.cullMode,'front');assert.equal(a.pipeline.primitive.frontFace,'ccw');
    assert.equal(b.pipeline.primitive.cullMode,'front');assert.equal(b.pipeline.primitive.frontFace,'cw');
    assert.equal(a.pipeline.depthStencil.depthCompare,'greater');assert.equal(a.pipeline.depthStencil.depthWriteEnabled,false);
    assert.equal(a.pipeline.fragment.targets[0].writeMask,0);assert.equal(r.drawCallCount,2);
    if(renderBundles)assert.equal(r.bundleDiagnostics.reuses,1);
    await r.whenIdle();r.dispose();
  });
test('transparent source materials can write depth; disabling the depth test suppresses writes',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  const a=await r.addMesh(g,{alphaMode:'BLEND',depthWrite:true});
  const b=await r.addMesh(g,{depthTest:false,depthWrite:true});
  const c=await r.addMesh(g,{side:'double'});
  r.render(frame([a,b,c]));const draws=d.snapshots[0];
  assert.equal(draws[0].pipeline.depthStencil.depthWriteEnabled,true);
  assert.ok(draws[0].pipeline.fragment.targets[0].blend);
  assert.equal(draws[1].pipeline.depthStencil.depthWriteEnabled,false);
  assert.equal(draws[1].pipeline.depthStencil.depthCompare,'always');
  assert.equal(draws[2].pipeline.primitive.cullMode,'none');await r.whenIdle();r.dispose();
});
test('render state admission rejects unsupported or conflicting values without allocation',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d),n=d.buffers.length;
  for(const options of [{side:'both'},{side:null},{side:'front',doubleSided:true},{depthWrite:1},{depthTest:null},
    {depthCompare:'nearer'},{colorWrite:[]},{side:'double',doubleSided:false}])await assert.rejects(r.addMesh(g,options));
  assert.equal(d.buffers.length,n);assert.equal(d.pipelines.length,6);r.dispose();
});
test('r186 light mode packs source decay, accepts zero penumbra, and keeps Lambert out of glTF falloff',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{threeLights:true}),m=await r.addMesh(geometry(d),{shading:'lambert'});
  r.render(frame([m],{cameraPosition:[0,0,5],lights:[{type:'point',position:[0,0,2],decay:0},
    {type:'spot',position:[0,0,2],decay:3,innerConeAngle:.5,outerConeAngle:.5}]}));
  const draw=d.snapshots[0][0],group=draw.groups.get(1).group,words=new Float32Array(draw.contents.get(group.entries[0].resource.buffer).buffer);
  assert.equal(words[22],0);assert.equal(words[38],3);assert.equal(words[36],words[37]);
  assert.match(draw.pipeline.fragment.module.code,/pow\(distance, light.cone.z\), 0.01/);
  assert.match(draw.pipeline.fragment.module.code,/window \* window/);
  const before=d.writes.length;
  for(const light of [{type:'ambient',decay:0},{type:'directional',decay:2},{type:'point',position:[0,0,1],decay:-1},
    {type:'spot',position:[0,0,1],decay:Infinity}])assert.throws(()=>r.render(frame([m],{cameraPosition:[0,0,5],lights:[light]})));
  assert.equal(d.writes.length,before);await r.whenIdle();r.dispose();
});
test('legacy light mode does not silently accept a requested source decay',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),m=await r.addMesh(geometry(d),{shading:'phong'});
  assert.throws(()=>r.render(frame([m],{cameraPosition:[0,0,5],lights:[{type:'point',position:[0,0,1],decay:2}]})));
  assert.equal(d.submissions.length,0);r.dispose();
});

test('comparison-only changes select all native depth functions without aliasing schedules',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{renderBundles:true}),g=geometry(d);
  const modes=['never','always','less','less-equal','equal','greater-equal','greater','not-equal'];
  const meshes=[];for(const depthCompare of modes)meshes.push(await r.addMesh(g,{depthCompare}));
  r.render(frame(meshes));r.render(frame(meshes));
  assert.deepEqual(d.snapshots[0].map(s=>s.pipeline.depthStencil.depthCompare),modes);
  assert.equal(new Set(d.snapshots[0].map(s=>s.pipeline)).size,8);
  assert.equal(r.bundleDiagnostics.reuses,1);await r.whenIdle();r.dispose();
});

test('volume slices reach every MRT attachment in clear and geometry passes without changing depth attachments',async()=>{
  const d=geometryDevice();Object.assign(d.limits,{maxColorAttachments:8,maxColorAttachmentBytesPerSample:32});
  d.createRenderPipeline=descriptor=>({...descriptor,getBindGroupLayout:()=>({})});
  const r=await createGpuAnimationRenderer(d,{format:'rgba8unorm',colorFormats:['rgba8unorm','rgba16float']});
  const colorViews=[{},{}],depthView={};
  for(const depthSlice of [0,3])r.render({...frame([]),colorView:colorViews[0],colorViews,depthView,depthSlice,scissor:[0,0,4,4]});
  assert.equal(d.passes.length,4);
  for(const [index,pass] of d.passes.entries()){
    assert.deepEqual(pass.desc.colorAttachments.map(attachment=>attachment.depthSlice),Array(2).fill(index<2?0:3));
    assert.equal(pass.desc.depthStencilAttachment.view,depthView);
    assert.equal(Object.hasOwn(pass.desc.depthStencilAttachment,'depthSlice'),false);
  }
  await r.whenIdle();r.dispose();
});

test('invalid volume slices and multisampled volume attachments reject before GPU effects',async()=>{
  for(const sampleCount of [1,4]){
    const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{sampleCount}),before=[d.buffers.length,d.writes.length,d.passes.length];
    for(const depthSlice of sampleCount===1?[-1,.5,NaN,0x100000000]:[0])
      assert.throws(()=>r.render({...frame([]),depthSlice}),{code:'ANIMATION_RENDER_ATTACHMENT'});
    assert.deepEqual([d.buffers.length,d.writes.length,d.passes.length],before);r.dispose();
  }
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d);r.render(frame([]));
  assert.equal(Object.hasOwn(d.passes[0].desc.colorAttachments[0],'depthSlice'),false);await r.whenIdle();r.dispose();
});
