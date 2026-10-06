/** Production command/pipeline tests. The recording device does not rasterize. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {snapshotAnimationRaster, ANIMATION_NORMAL_BLEND} from './animation_raster.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {withThreeFogReceivers} from './three_fog.mjs';
const I = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const constantBlend = () => ({color:{srcFactor:'constant',dstFactor:'one-minus-constant'},alpha:{srcFactor:'one',dstFactor:'zero'}});
const stencil = () => ({front:{compare:'equal',passOp:'replace',failOp:'increment-wrap',depthFailOp:'decrement-clamp'},
  back:{compare:'not-equal',passOp:'invert',failOp:'zero',depthFailOp:'increment-clamp'},readMask:0x7f,writeMask:0x3f});
function device() { const d = geometryDevice(); d.limits.maxBindingsPerBindGroup = 32; return d; }
function geometry(d) { return {vertexCount:3, vertexBuffer:d.createBuffer({size:120,usage:32}), worldMatrix:I, whenIdle:async()=>{},
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'}]}}; }
const frame = draws => ({draws,colorView:{},depthView:{},viewProjection:I,lighting:{cameraPosition:[0,0,3],lights:[]}});
function packet(draw, slot = 0) {
  const {group,offsets} = draw.groups.get(0), entry = group.entries.find(e=>e.binding===0);
  return new Float32Array(draw.contents.get(entry.resource.buffer).buffer,(entry.resource.offset??0)+(offsets[0]??slot*256),64);
}

test('blend equations, stencil faces/masks and depth bias are immutable pipeline state', async()=>{
  const d=device(), r=await createGpuAnimationRenderer(d,{depthFormat:'depth24plus-stencil8',maxDraws:2}),g=geometry(d);
  const options={blend:constantBlend(),blendConstant:[.1,.2,.3,.4],stencil:stencil(),stencilReference:17,
    depthBias:-7,depthBiasSlopeScale:-1.5,depthBiasClamp:2};
  const adding=r.addMesh(g,options);
  options.blend.color.srcFactor='zero';options.stencil.front.compare='never';options.blendConstant[0]=1;
  const m=await adding; r.render(frame([m]));
  const s=d.snapshots[0][0],depth=s.pipeline.depthStencil;
  assert.equal(s.pipeline.fragment.targets[0].blend.color.srcFactor,'constant');
  assert.equal(depth.stencilFront.compare,'equal');assert.equal(depth.stencilBack.compare,'not-equal');
  assert.equal(depth.stencilFront.failOp,'increment-wrap');assert.equal(depth.stencilWriteMask,0x3f);
  assert.equal(depth.depthBias,-7);assert.equal(depth.depthBiasSlopeScale,-1.5);assert.equal(depth.depthBiasClamp,2);
  assert.deepEqual(s.blendConstant,[.1,.2,.3,.4]);assert.equal(s.stencilReference,17);
  assert.equal(d.passes[0].desc.depthStencilAttachment.stencilLoadOp,'clear');
  assert.equal(d.passes[0].desc.depthStencilAttachment.stencilStoreOp,'store');
  await r.whenIdle();r.dispose();assert.equal(g.vertexBuffer.destroyed,false);
});

for (const renderBundles of [false,true]) for (const instancing of [false,true])
  test(`live per-use constants/references preserve source order and arena offsets (${renderBundles}/${instancing})`,async()=>{
    const d=device(),r=await createGpuAnimationRenderer(d,{depthFormat:'depth24plus-stencil8',renderBundles,instancing,maxDraws:4,maxRenderBundles:4});
    const m=await r.addMesh(geometry(d),{blend:constantBlend(),stencil:stencil()});
    const inputs=[{mesh:m,blendConstant:[.1,.2,.3,.4],stencilReference:7,baseColor:[1,0,0,.3]},
      {mesh:m,blendConstant:[.5,.6,.7,.8],stencilReference:9,baseColor:[0,1,0,.7]},
      {mesh:m,blendConstant:[.5,.6,.7,.8],stencilReference:9,baseColor:[0,0,1,.9]}];
    r.render({...frame(inputs),stencilLoadOp:'load',clearStencil:23});
    assert.equal(r.drawCallCount,3,'fixed-function blended draws never coalesce, even with OPAQUE shader alpha');
    for(let i=0;i<3;i++){
      const s=d.snapshots[0][i];assert.deepEqual(s.blendConstant,inputs[i].blendConstant);assert.equal(s.stencilReference,inputs[i].stencilReference);
      assert.deepEqual([...packet(s,i).slice(16,19)],inputs[i].baseColor.slice(0,3));
      if(instancing)assert.equal(s.args[3],i);else assert.equal(s.groups.get(0).offsets[0],i*256);
    }
    const before=[d.pipelines.length,d.buffers.length,d.bundleEncoders.length];
    inputs[0].blendConstant[0]=.9;inputs[0].stencilReference=31;
    inputs[1].blendConstant[1]=.4;inputs[2].blendConstant[1]=.4;
    r.render({...frame(inputs),clearStencil:12});
    assert.equal(d.snapshots[1][0].blendConstant[0],.9);assert.equal(d.snapshots[1][0].stencilReference,31);
    assert.equal(d.snapshots[0][0].blendConstant[0],.1);assert.equal(d.snapshots[0][0].stencilReference,7);
    assert.deepEqual([d.pipelines.length,d.buffers.length,d.bundleEncoders.length],before);
    if(renderBundles){assert.equal(r.bundleDiagnostics.builds,2);assert.equal(r.bundleDiagnostics.reuses,2);
      for(const b of d.bundleEncoders)assert.equal(b.desc.stencilReadOnly,false);
      const encoder=d.createRenderBundleEncoder({});assert.equal(encoder.setBlendConstant,undefined);assert.equal(encoder.setStencilReference,undefined);
    }
    await r.whenIdle();r.dispose();
  });

test('stencil-only instance runs split on reference and keep global firstInstance in bundles',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{depthFormat:'depth24plus-stencil8',instancing:true,renderBundles:true,maxDraws:4});
  const m=await r.addMesh(geometry(d),{stencil:stencil()});
  r.render(frame([1,1,2,2].map(stencilReference=>({mesh:m,stencilReference}))));
  assert.equal(r.drawCallCount,2);assert.deepEqual(d.snapshots[0].map(s=>s.args),[[3,2,0,0],[3,2,0,2]]);
  assert.deepEqual(d.snapshots[0].map(s=>s.stencilReference),[1,2]);
  r.render(frame([3,3,4,4].map(stencilReference=>({mesh:m,stencilReference}))));
  assert.equal(r.bundleDiagnostics.reuses,2);assert.deepEqual(d.snapshots[1].map(s=>s.stencilReference),[3,4]);
  // New equality boundaries change schedules, not the meaning of the old arena offsets.
  r.render(frame([3,3,3,3].map(stencilReference=>({mesh:m,stencilReference}))));
  assert.equal(r.drawCallCount,1);assert.deepEqual(d.snapshots[2][0].args,[3,4,0,0]);r.dispose();
});

for(const shading of ['unlit','lambert','phong','toon','metallic-roughness'])
  test(`${shading}: premultiplication follows lighting/fog/alpha masking without changing the draw packet`,async()=>{
    const d=device(),r=await createGpuAnimationRenderer(d,{fog:true,maxDraws:2});
    const m=await r.addMesh(geometry(d),{shading,premultipliedAlpha:true,alphaMode:'BLEND',alphaTest:true,alphaCutoff:.25});
    r.render(frame([{mesh:m,baseColor:[.1,.2,.3,.4]}]));const s=d.snapshots[0][0],code=s.pipeline.fragment.module.code;
    assert.match(code,/apply_distance_fog\(rgb, input.fog_depth\) \* output_alpha, output_alpha/);
    assert.ok(code.lastIndexOf('rgba.a < draw_info.options.x')<code.indexOf('let output_alpha ='));
    assert.equal(s.pipeline.fragment.targets[0].blend.color.srcFactor,'one');
    assert.equal(s.pipeline.depthStencil.depthWriteEnabled,false);assert.equal(packet(s)[20],.25);assert.equal(packet(s)[21],1);
    assert.equal(r.allocatedBytes,2*256+48+(shading==='unlit'?0:544));await r.whenIdle();r.dispose();
  });

test('explicit no-blend retains fractional shader output; normal blending remains the legacy default',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{maxDraws:3}),g=geometry(d);
  const no=await r.addMesh(g,{alphaMode:'BLEND',blend:null}),normal=await r.addMesh(g,{alphaMode:'BLEND'}),opaque=await r.addMesh(g);
  r.render(frame([no,normal,opaque]));
  assert.equal(d.snapshots[0][0].pipeline.fragment.targets[0].blend,undefined);assert.equal(packet(d.snapshots[0][0])[21],1);
  assert.deepEqual(d.snapshots[0][1].pipeline.fragment.targets[0].blend,ANIMATION_NORMAL_BLEND);
  assert.equal(d.snapshots[0][2].pipeline.fragment.targets[0].blend,undefined);r.dispose();
});

test('min/max normalize ignored factors while all five operations and portable factors reach pipelines',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),g=geometry(d);
  const factors=['zero','one','src','one-minus-src','src-alpha','one-minus-src-alpha','dst','one-minus-dst','dst-alpha','one-minus-dst-alpha','src-alpha-saturated','constant','one-minus-constant'];
  for(const operation of ['add','subtract','reverse-subtract','min','max'])for(const srcFactor of factors){
    const m=await r.addMesh(g,{blend:{color:{operation,srcFactor,dstFactor:'one-minus-src-alpha'},alpha:{operation}}});
    r.render(frame([m]));const b=d.snapshots.at(-1)[0].pipeline.fragment.targets[0].blend;
    assert.equal(b.color.operation,operation);assert.equal(b.color.srcFactor,['min','max'].includes(operation)?'one':srcFactor);
    if(['min','max'].includes(operation))assert.equal(b.alpha.dstFactor,'one');m.dispose();
  }
  await r.whenIdle();r.dispose();
});

test('different default constants/references reuse fixed pipelines without sharing live values',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{depthFormat:'depth24plus-stencil8',maxDraws:2}),g=geometry(d);
  const a=await r.addMesh(g,{blend:constantBlend(),stencil:stencil(),blendConstant:[.1,.2,.3,.4],stencilReference:3});
  const before=d.pipelines.length,b=await r.addMesh(g,{blend:constantBlend(),stencil:stencil(),blendConstant:[.4,.3,.2,.1],stencilReference:5});
  assert.equal(d.pipelines.length,before);r.render(frame([a,b]));
  assert.notDeepEqual(d.snapshots[0][0].blendConstant,d.snapshots[0][1].blendConstant);
  assert.deepEqual(d.snapshots[0].map(s=>s.stencilReference),[3,5]);r.dispose();
});

test('stencil works in colorless passes and disabled depth testing retains independent stencil operations',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{format:null,depthFormat:'depth24plus-stencil8',maxDraws:1});
  const m=await r.addMesh(geometry(d),{stencil:stencil(),depthTest:false,depthWrite:true,stencilReference:255});
  r.render({depthView:{},viewProjection:I,draws:[m],depthLoadOp:'load',stencilLoadOp:'clear',clearStencil:8});
  const s=d.snapshots[0][0];assert.deepEqual(s.pipeline.fragment.targets,[]);assert.equal(s.pipeline.depthStencil.depthWriteEnabled,false);
  assert.equal(s.pipeline.depthStencil.depthCompare,'always');assert.equal(s.pipeline.depthStencil.stencilFront.passOp,'replace');
  assert.equal(s.stencilReference,255);assert.equal(d.passes[0].desc.depthStencilAttachment.stencilClearValue,8);r.dispose();
});

test('unsupported format/features and invalid fixed state fail before new resources or commands',async()=>{
  const d=device();await assert.rejects(createGpuAnimationRenderer(d,{depthFormat:'depth32float-stencil8'}));assert.equal(d.buffers.length,0);
  d.features=new Set(['depth32float-stencil8']);const supported=await createGpuAnimationRenderer(d,{depthFormat:'depth32float-stencil8'});supported.dispose();
  const r=await createGpuAnimationRenderer(d,{maxDraws:1}),g=geometry(d);
  const invalid=[{stencil:stencil()},{stencilReference:1},{blendConstant:[0,0,0,0]},
    {blend:{color:{srcFactor:'src1'}}},{blend:{alpha:{operation:'wrong'}}},{blend:{unknown:1}},
    {premultipliedAlpha:null},{premultipliedAlpha:1},{depthBias:1.5},{depthBias:2**31},{depthBiasSlopeScale:NaN},{depthBiasClamp:Infinity}];
  for(const options of invalid){const before=[d.buffers.length,d.pipelines.length,d.writes.length];await assert.rejects(r.addMesh(g,options));
    assert.deepEqual([d.buffers.length,d.pipelines.length,d.writes.length],before);}
  const accessor={};Object.defineProperty(accessor,'color',{get(){throw Error('getter executed');}});
  await assert.rejects(r.addMesh(g,{blend:accessor}),{code:'ANIMATION_RASTER_INPUT'});r.dispose();
});

test('invalid late live state leaves prior submission untouched and permits corrected retry',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{depthFormat:'depth24plus-stencil8',maxDraws:2}),g=geometry(d);
  const m=await r.addMesh(g,{stencil:stencil(),blend:constantBlend()});r.render(frame([m]));
  for(const bad of [{stencilReference:-1},{stencilReference:256},{stencilReference:1.5},
    {blendConstant:[0,0,NaN,0]},{blendConstant:[1,2,3,4]},{blendConstant:[0,0,0]}]){
    const before=[d.writes.length,d.submissions.length,d.passes.length];assert.throws(()=>r.render(frame([m,{mesh:m,...bad}])));
    assert.deepEqual([d.writes.length,d.submissions.length,d.passes.length],before);assert.equal(r.failed,false);
  }
  for(const bad of [{stencilLoadOp:'discard'},{clearStencil:256},{clearStencil:NaN}]){
    const count=d.submissions.length;assert.throws(()=>r.render({...frame([m]),...bad}));assert.equal(d.submissions.length,count);
  }
  r.render(frame([{mesh:m,stencilReference:123,blendConstant:[1,0,1,0]}]));assert.equal(r.version,2);r.dispose();
});

test('fog receiver spans preserve stencil contents after the first explicit clear',async()=>{
  const d=device(),native=await createGpuAnimationRenderer(d,{depthFormat:'depth24plus-stencil8',fog:true,maxDraws:3}),r=withThreeFogReceivers(native);
  const m=await r.addMesh(geometry(d),{stencil:stencil()});
  r.render({...frame([true,false,true].map(receiveFog=>({mesh:m,receiveFog}))),stencilLoadOp:'clear',clearStencil:19,
    fog:{type:'linear',color:[.1,.2,.3],near:1,far:9,depthFromClip:[0,0,0,1]}});
  assert.deepEqual(d.passes.map(p=>p.desc.depthStencilAttachment.stencilLoadOp),['clear','load','load']);
  assert.equal(d.passes[0].desc.depthStencilAttachment.stencilClearValue,19);await r.whenIdle();r.dispose();
});
