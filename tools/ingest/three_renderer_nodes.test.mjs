/** Native TSL through the actual facade, scene bridge and GPU command owner.
 * The recording device proves per-call capture and queue behavior; browser
 * source comparisons separately execute the generated shaders. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';
import {SunLight, SunLightNode} from './fixtures/three_sun_light.mjs';
const Renderer=createWebGPURendererClass(T);
function fixture(t){
  const device=geometryDevice();device.limits.maxTextureDimension2D=8192;device.limits.maxBindingsPerBindGroup=1000;
  device.textureWrites=[];device.textures=[];
  device.createTexture=desc=>{const texture={dimension:'2d',depthOrArrayLayers:1,sampleCount:1,...desc,
    width:desc.size.width??desc.size[0],height:desc.size.height??desc.size[1],
    createView(options={}){return {texture,...options};},destroy(){this.destroyed=true;}};device.textures.push(texture);return texture;};
  device.createSampler=desc=>({...desc});device.queue.writeTexture=(destination,data)=>{
    const bytes=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength).slice();
    destination.texture.pixels=bytes;device.textureWrites.push({destination,bytes});
  };
  let configured,acquired=0;
  const canvas={width:64,height:64,style:{},getContext:()=>({configure(value){configured=value;},unconfigure(){},
    getCurrentTexture(){acquired++;return device.createTexture({size:[64,64],format:configured.format,usage:16});}})};
  const renderer=new Renderer({canvas,device});t.after(()=>renderer.dispose());
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(50,1,.1,30);camera.position.z=4;
  const tint=T.TSL.uniform(new T.Color(.125,.375,.625)).setName('queuedTint');
  const material=new T.MeshBasicNodeMaterial();material.colorNode=tint;
  const mesh=new T.Mesh(new T.PlaneGeometry(),material);scene.add(mesh);
  return {renderer,device,scene,camera,tint,material,mesh,acquired:()=>acquired};
}
const drain=async renderer=>{while(renderer._drain)await renderer._drain;};
const nativeDraws=device=>device.snapshots.flat().filter(draw=>draw.pipeline.vertex.entryPoint==='main');
function storageWords(draw){
  return storageResources(draw).map(({buffer,offset=0,size=buffer.size-offset})=>{
    const bytes=draw.contents.get(buffer);
    return [...new Float32Array(bytes.buffer,bytes.byteOffset+offset,size/4)];
  });
}
function containsColor(draw,color){
  for(const {group,offsets} of draw.groups.values()){
    const resource=group.entries.find(entry=>entry.resource?.buffer)?.resource;if(!resource)continue;
    const {buffer,size}=resource,bytes=draw.contents.get(buffer),offset=offsets[0]??0;
    const words=new Float32Array(bytes.buffer,bytes.byteOffset+offset,size/4);
    for(let i=0;i<=words.length-3;i++)if(color.every((v,j)=>words[i+j]===v))return true;
  }
  return false;
}

function uniformBlocks(draw){
  const result=[];
  for(const {group,offsets} of draw.groups.values()){
    const layout=group.layout.entries.filter(entry=>entry.buffer?.type==='uniform').sort((a,b)=>a.binding-b.binding);
    for(const [index,entry] of layout.entries()){
      const resource=group.entries.find(value=>value.binding===entry.binding).resource;
      const bytes=draw.contents.get(resource.buffer),offset=(resource.offset??0)+(offsets[index]??0);
      result.push([...new Float32Array(bytes.buffer,bytes.byteOffset+offset,resource.size/4)]);
    }
  }
  return result;
}

test('queued uniform-buffer arrays retain every binding and current values without update flags',async t=>{
  const f=fixture(t),N=T.TSL;
  const initial=new Float32Array([.125,.375,.625,1,.25,.5,.75,1]);
  const colors=N.buffer(initial,'vec4',2),gain=N.buffer(new Float32Array([.75,0,0,0]),'vec4',1);
  f.material.colorNode=colors.element(0).rgb.mul(gain.element(0).x).mul(f.tint);
  await f.renderer.init();
  assert.equal(f.renderer.backend.capabilities.getUniformBufferLimit(),f.device.limits.maxUniformBufferBindingSize);
  f.renderer.render(f.scene,f.camera);
  initial.set([.875,.625,.375,1]);gain.value[0]=.5;
  f.renderer.render(f.scene,f.camera);
  colors.value=new Float32Array([.25,.75,.5,1,.375,.625,.875,1]);gain.value[0]=.25;
  f.renderer.render(f.scene,f.camera);
  colors.value.fill(0);gain.value.fill(0);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===8)),[
    [.125,.375,.625,1,.25,.5,.75,1],
    [.875,.625,.375,1,.25,.5,.75,1],
    [.25,.75,.5,1,.375,.625,.875,1]]);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===4)?.[0]),[.75,.5,.25]);
  assert.equal(draws[0].pipeline,draws[1].pipeline);assert.equal(draws[1].pipeline,draws[2].pipeline);
  for(const draw of draws)assert.ok([...draw.groups.values()].some(({offsets})=>offsets.length>=3));
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('uniform-buffer partial updates retain unrequested bytes across queued render calls',async t=>{
  const f=fixture(t),N=T.TSL;
  const colors=N.buffer(new Float32Array([.125,.375,.625,1,.25,.5,.75,1]),'vec4',2);
  f.material.colorNode=colors.element(0).rgb.add(colors.element(1).rgb).mul(f.tint);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  colors.value.set([.875,.625,.375,1,.75,.25,.5,1]);colors.addUpdateRange(4,4);
  f.renderer.render(f.scene,f.camera);
  assert.equal(colors.updateRanges.length,0,'source upload ranges are consumed synchronously');
  f.renderer.render(f.scene,f.camera);
  colors.value.fill(0);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===8)),[
    [.125,.375,.625,1,.25,.5,.75,1],
    [.125,.375,.625,1,.75,.25,.5,1],
    [.875,.625,.375,1,.75,.25,.5,1]]);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('two unprepared render calls keep independent native uniforms until actual submission',async t=>{
  const f=fixture(t);await f.renderer.init();
  f.renderer.render(f.scene,f.camera);
  assert.equal(f.acquired(),0);
  f.tint.value.setRGB(.875,.625,.375);f.renderer.render(f.scene,f.camera);
  f.tint.value.setRGB(.2,.3,.4);
  assert.ok(f.renderer._nodeCaptureBytes>0);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(containsColor(draws[0],[.125,.375,.625]));assert.ok(containsColor(draws[1],[.875,.625,.375]));
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.renderer.info.f3d.presentedRenders,2);
  assert.equal(f.camera.coordinateSystem,T.WebGPUCoordinateSystem);
  assert.ok(draws.every(d=>d.pipeline.fragment.entryPoint==='main'));
});

test('transparent native DoubleSide renders submit ordered side pipelines with independent queued uniforms',async t=>{
  const f=fixture(t);f.material.transparent=true;f.material.side=T.DoubleSide;f.mesh.scale.x=-1;
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  f.tint.value.setRGB(.875,.625,.375);f.renderer.render(f.scene,f.camera);
  f.tint.value.setRGB(.25,.5,.75);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  assert.ok(draws.every(draw=>draw.pipeline.primitive.cullMode==='back'));
  assert.deepEqual(draws.map(draw=>draw.pipeline.primitive.frontFace),['ccw','cw','ccw','cw'],
    'source BackSide reverses winding; negative determinant flips both side pipelines');
  assert.ok(draws.slice(0,2).every(draw=>containsColor(draw,[.125,.375,.625])));
  assert.ok(draws.slice(2).every(draw=>containsColor(draw,[.875,.625,.375])));
  assert.equal(draws[0].pipeline,draws[2].pipeline);assert.equal(draws[1].pipeline,draws[3].pipeline);
  assert.equal(f.material.side,T.DoubleSide);assert.equal(f.renderer._nodeCaptureBytes,0);
  assert.equal(f.renderer.info.f3d.presentedRenders,2);
  const deferred=f.renderer.info.f3d.deferredRenders,pipelines=f.device.pipelines.length;
  f.renderer.render(f.scene,f.camera);
  assert.equal(f.renderer.info.f3d.deferredRenders,deferred,'both prepared side variants remain ready');
  assert.equal(f.device.pipelines.length,pipelines);
  assert.ok(nativeDraws(f.device).slice(-2).every(draw=>containsColor(draw,[.25,.5,.75])));
  f.material.forceSinglePass=true;f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length,7);
  assert.equal(nativeDraws(f.device).at(-1).pipeline.primitive.cullMode,'none');
});

test('compilation runs no node updates; logical source callbacks run once per visible draw before awaits',async t=>{
  const f=fixture(t),calls=[],frame=f.renderer._nodes.nodeFrame;
  const other=new T.Mesh(f.mesh.geometry,f.material);other.position.x=.5;f.scene.add(other);
  const hidden=new T.Mesh(f.mesh.geometry,f.material);hidden.visible=false;f.scene.add(hidden);
  f.material.colorNode=T.TSL.uniform(new T.Color()).onObjectUpdate(context=>{
    calls.push([context.object,context.renderId,context.frameId]);return context.object===f.mesh?new T.Color(.125,.375,.625):new T.Color(.875,.625,.375);
  });
  await f.renderer.compileAsync(f.scene,f.camera);assert.equal(calls.length,0);
  const original={object:frame.object,material:frame.material,camera:frame.camera,renderer:frame.renderer,renderId:frame.renderId};
  f.renderer.render(f.scene,f.camera);assert.equal(calls.length,2);
  await drain(f.renderer);assert.equal(calls.length,2);
  assert.deepEqual(new Set(calls.map(c=>c[0])),new Set([f.mesh,other]));
  assert.ok(calls.every(c=>c[1]===f.renderer.info.calls));
  assert.deepEqual({object:frame.object,material:frame.material,camera:frame.camera,renderer:frame.renderer,renderId:frame.renderId},original);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(containsColor(draws[0],[.125,.375,.625]));assert.ok(containsColor(draws[1],[.875,.625,.375]));
});

test('queued graph replacement preserves the earlier shader and later visibility cannot erase its draw',async t=>{
  const f=fixture(t);await f.renderer.init();
  f.renderer.render(f.scene,f.camera);
  f.material.colorNode=T.TSL.vec3(.8,.2,.6);f.renderer.render(f.scene,f.camera);
  f.mesh.visible=false;f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.match(draws[0].pipeline.fragment.module.code,/queuedTint/);
  assert.doesNotMatch(draws[1].pipeline.fragment.module.code,/queuedTint/);
  assert.equal(f.renderer.info.f3d.presentedRenders,3);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native and fixed meshes share native depth sorting even when material creation order disagrees',async t=>{
  const f=fixture(t);const fixed=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial({color:0xff0000}));
  // The native material was created first, but the fixed object is nearer.
  fixed.position.z=.5;f.scene.add(fixed);
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=f.device.snapshots.at(-1);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(d=>d.pipeline.vertex.entryPoint),['vertex_main','main']);
});

test('unsupported native graph resources fail before canvas acquisition or submission',async t=>{
  const f=fixture(t);await f.renderer.init();
  f.material.colorNode=T.TSL.texture(new T.DepthTexture(1,1));
  const writes=f.device.writes.length;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(f.acquired(),0);assert.equal(f.device.submissions.length,0);assert.equal(f.device.writes.length,writes);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('disposal and preparation failure release queued native capture bytes',async t=>{
  const f=fixture(t);await f.renderer.init();f.renderer.render(f.scene,f.camera);
  assert.ok(f.renderer._nodeCaptureBytes>0);f.renderer.dispose();
  await drain(f.renderer);assert.equal(f.renderer._nodeCaptureBytes,0);
  const g=fixture(t);await g.renderer.init();g.device.createRenderPipelineAsync=()=>Promise.reject(new Error('native pipeline rejected'));
  g.renderer.render(g.scene,g.camera);await assert.rejects(drain(g.renderer),/native pipeline rejected/);
  assert.equal(g.renderer._nodeCaptureBytes,0);assert.equal(g.device.submissions.length,0);
});

function dataTexture(r,g,b){
  const texture=new T.DataTexture(new Uint8Array([r,g,b,255]),1,1);
  texture.minFilter=texture.magFilter=T.LinearFilter;
  texture.needsUpdate=true;return texture;
}
const textureResources=draw=>[...draw.groups.values()].flatMap(({group})=>group.entries)
  .map(entry=>entry.resource).filter(resource=>resource?.texture);

test('queued native graphs retain texture identity, pixel versions and sampler state',async t=>{
  const f=fixture(t),a=dataTexture(240,30,10),b=dataTexture(20,60,230),sampled=T.TSL.texture(a);
  f.material.colorNode=sampled.rgb.mul(f.tint);await f.renderer.init();
  f.renderer.render(f.scene,f.camera);
  sampled.value=b;f.renderer.render(f.scene,f.camera);
  a.image.data.set([10,210,70,255]);a.needsUpdate=true;a.wrapS=T.RepeatWrapping;
  sampled.value=a;f.renderer.render(f.scene,f.camera);
  a.image.data.fill(0);b.image.data.fill(0);sampled.value=dataTexture(1,2,3);
  assert.equal(f.device.textureWrites.length,0,'capture does not upload during pending pipeline preparation');
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(draw=>[...textureResources(draw)[0].texture.pixels]),[
    [240,30,10,255],[20,60,230,255],[10,210,70,255]]);
  const samplers=draws.map(draw=>[...draw.groups.values()].flatMap(({group})=>group.entries)
    .map(entry=>entry.resource).find(resource=>resource?.addressModeU));
  assert.deepEqual(samplers.map(s=>s.addressModeU),['clamp-to-edge','clamp-to-edge','repeat']);
  assert.equal(draws[0].pipeline,draws[2].pipeline,'compatible texture edits retain the pipeline');
  assert.equal(f.renderer._nodeCaptureBytes,0);await f.renderer.waitForGPU();
});

test('textured compileAsync initializes without uploading pixels or replaying object updates',async t=>{
  const f=fixture(t),a=dataTexture(220,40,80),b=dataTexture(30,170,240),sampled=T.TSL.texture(a);
  const second=new T.Mesh(f.mesh.geometry,f.material);second.position.x=.4;f.scene.add(second);
  const seen=[];let uploads=0;a.onUpdate=b.onUpdate=()=>uploads++;
  const gain=T.TSL.uniform(1).onObjectUpdate(({object})=>{
    seen.push(object);sampled.value=object===f.mesh?a:b;return 1;
  });
  f.material.colorNode=sampled.rgb.mul(gain);
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(f.renderer._initialized,true);assert.equal(uploads,0);assert.equal(seen.length,0);
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(seen,[f.mesh,second]);assert.equal(uploads,2);
  assert.deepEqual(nativeDraws(f.device).map(draw=>[...textureResources(draw)[0].texture.pixels]),[
    [220,40,80,255],[30,170,240,255]]);
});

test('native texture graphs borrow rendered target storage and reject attachment feedback',async t=>{
  const f=fixture(t),target=new T.RenderTarget(16,16,{depthBuffer:false});await f.renderer.init();
  f.renderer.setRenderTarget(target);f.renderer.render(f.scene,f.camera);
  f.material.colorNode=T.TSL.texture(target.texture).rgb;
  f.renderer.setRenderTarget(null);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  const targetView=f.device.passes.find(pass=>pass.desc.colorAttachments[0].view.texture.width===16).desc.colorAttachments[0].view;
  assert.equal(textureResources(draws[1])[0].texture,targetView.texture,'native graph samples the actual target');
  assert.equal(f.device.textureWrites.length,0,'no readback or CPU upload for target sampling');
  const submissions=f.device.submissions.length;f.renderer.setRenderTarget(target);
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_FEEDBACK'});
  assert.equal(f.device.submissions.length,submissions);assert.equal(f.renderer._nodeCaptureBytes,0);
  await f.renderer.waitForGPU();
});

test('native depth comparisons preserve queued functions and sample the rendered depth attachment',async t=>{
  const f=fixture(t),depth=new T.DepthTexture(16,16,T.UnsignedIntType),target=new T.RenderTarget(16,16,{depthTexture:depth});
  depth.minFilter=depth.magFilter=T.LinearFilter;depth.compareFunction=T.LessEqualCompare;
  assert.throws(()=>f.renderer.hasCompatibility(T.Compatibility.TEXTURE_COMPARE),/before the backend is initialized/);
  await f.renderer.init();assert.equal(f.renderer.hasCompatibility(T.Compatibility.TEXTURE_COMPARE),true);
  assert.equal(f.renderer.hasCompatibility('unsupported-compatibility'),false);
  f.renderer.setRenderTarget(target);f.renderer.render(f.scene,f.camera);
  f.material.colorNode=T.TSL.vec3(T.TSL.texture(depth).compare(T.TSL.float(.5)));
  f.renderer.setRenderTarget(null);f.renderer.render(f.scene,f.camera);
  depth.compareFunction=T.GreaterCompare;f.renderer.render(f.scene,f.camera);
  depth.compareFunction=T.AlwaysCompare;
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  const view=f.device.passes.find(pass=>pass.desc.colorAttachments[0].view.texture.width===16).desc.depthStencilAttachment.view;
  const consumers=draws.slice(1);
  for(const draw of consumers){
    assert.match(draw.pipeline.fragment.module.code,/texture_depth_2d/);
    assert.match(draw.pipeline.fragment.module.code,/sampler_comparison/);
    assert.match(draw.pipeline.fragment.module.code,/textureSampleCompare/);
    assert.equal(textureResources(draw)[0].texture,view.texture);
    assert.equal(textureResources(draw)[0].aspect,'depth-only');
    assert.ok([...draw.groups.values()].some(({group})=>group.layout.entries.some(entry=>entry.sampler?.type==='comparison')));
  }
  const samplers=consumers.map(draw=>[...draw.groups.values()].flatMap(({group})=>group.entries)
    .map(entry=>entry.resource).find(resource=>resource?.compare));
  assert.deepEqual(samplers.map(sampler=>sampler.compare),['less-equal','greater']);
  assert.ok(samplers.every(sampler=>sampler.minFilter==='linear'&&sampler.magFilter==='linear'));
  assert.equal(f.device.textureWrites.length,0);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('ordinary native depth reads use the source nearest sampler and reject depth attachment feedback',async t=>{
  const f=fixture(t),depth=new T.DepthTexture(16,16,T.UnsignedIntType),target=new T.RenderTarget(16,16,{depthTexture:depth});
  depth.minFilter=depth.magFilter=T.LinearFilter;depth.compareFunction=T.LessEqualCompare;
  await f.renderer.init();f.renderer.setRenderTarget(target);f.renderer.render(f.scene,f.camera);
  f.material.colorNode=T.TSL.vec3(T.TSL.texture(depth));f.renderer.setRenderTarget(null);f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  const draw=nativeDraws(f.device).at(-1);
  assert.match(draw.pipeline.fragment.module.code,/texture_depth_2d/);
  assert.doesNotMatch(draw.pipeline.fragment.module.code,/sampler_comparison/);
  const sampler=[...draw.groups.values()].flatMap(({group})=>group.entries)
    .map(entry=>entry.resource).find(resource=>resource?.addressModeU);
  assert.equal(sampler.compare,undefined);assert.equal(sampler.minFilter,'nearest');assert.equal(sampler.magFilter,'nearest');
  assert.ok([...draw.groups.values()].some(({group})=>group.layout.entries.some(entry=>entry.sampler?.type==='non-filtering')));
  const submissions=f.device.submissions.length;f.renderer.setRenderTarget(target);
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_FEEDBACK'});
  assert.equal(f.device.submissions.length,submissions);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('depth compare graphs without a comparison function fail before application updates or submission',async t=>{
  const f=fixture(t),depth=new T.DepthTexture(16,16,T.UnsignedIntType),target=new T.RenderTarget(16,16,{depthTexture:depth});
  let updates=0;
  f.material.colorNode=T.TSL.vec3(T.TSL.texture(depth).compare(.5)).mul(T.TSL.uniform(1).onObjectUpdate(()=>{updates++;return 1;}));
  await f.renderer.init();
  assert.equal(target.depthTexture,depth);
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(updates,0);assert.equal(f.acquired(),0);assert.equal(f.device.submissions.length,0);
});

test('failed textured preparation releases queued histories and renderer disposal releases the shared owner',async t=>{
  const f=fixture(t),texture=dataTexture(190,30,70);f.material.colorNode=T.TSL.texture(texture);
  await f.renderer.init();f.device.createRenderPipelineAsync=()=>Promise.reject(new Error('textured pipeline rejected'));
  f.renderer.render(f.scene,f.camera);texture.image.data[0]=70;texture.needsUpdate=true;
  f.renderer.render(f.scene,f.camera);const owner=f.renderer._getSampledTextureInputs();
  assert.equal(owner.diagnostics.versions,2);
  await assert.rejects(drain(f.renderer),/textured pipeline rejected/);
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(owner.diagnostics.versions,1);
  assert.equal(f.device.submissions.length,0);f.renderer.dispose();
  assert.equal(owner.diagnostics.snapshotBytes,0);assert.equal(owner.diagnostics.versions,0);
});

test('queued lit graphs retain source FRAME light updates across objects and program replacements',async t=>{
  const f=fixture(t),lambert=new T.MeshLambertNodeMaterial(),phong=new T.MeshPhongNodeMaterial();
  lambert.colorNode=f.tint;phong.colorNode=T.TSL.vec3(.2,.4,.6);f.mesh.material=lambert;
  const second=new T.Mesh(f.mesh.geometry,phong);second.position.x=.5;f.scene.add(second);
  const light=new T.AmbientLight(new T.Color(.125,.25,.5),2);f.scene.add(light);
  await f.renderer.init();
  f.renderer.render(f.scene,f.camera);
  light.intensity=3;lambert.colorNode=T.TSL.vec3(.3,.5,.7);
  f.renderer.render(f.scene,f.camera);
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  light.intensity=7;
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,6);
  for(const draw of draws.slice(0,4))assert.ok(containsColor(draw,[.25,.5,1]));
  for(const draw of draws.slice(4))assert.ok(containsColor(draw,[.375,.75,1.5]));
  assert.notEqual(draws[0].pipeline,draws[2].pipeline,'graph replacement prepares a new program');
  assert.equal(f.renderer.info.f3d.presentedRenders,3);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('lit material arrays and scene overrides use the effective source material context',async t=>{
  const f=fixture(t),toon=new T.MeshToonNodeMaterial({emissive:new T.Color(.125,.25,.5)}),
    phong=new T.MeshPhongNodeMaterial({emissive:new T.Color(.625,.375,.125),shininess:23});
  toon.colorNode=f.tint;phong.colorNode=T.TSL.vec3(.1,.2,.3);
  f.mesh.geometry.clearGroups();f.mesh.geometry.addGroup(0,3,0);f.mesh.geometry.addGroup(3,3,1);
  f.mesh.material=[toon,phong];f.scene.add(new T.AmbientLight(0xffffff,.25));
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);
  const override=new T.MeshLambertNodeMaterial({emissive:new T.Color(.5,.25,.125)});
  override.colorNode=T.TSL.vec3(.6,.3,.2);f.scene.overrideMaterial=override;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  assert.ok(containsColor(draws[0],[.125,.25,.5]));assert.ok(containsColor(draws[1],[.625,.375,.125]));
  assert.ok(draws.slice(2).every(draw=>containsColor(draw,[.5,.25,.125])));
  assert.equal(f.mesh.material[0],toon);assert.equal(f.mesh.material[1],phong);
  assert.equal(f.scene.overrideMaterial,override);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native-only scenes ignore unused lights and reject unsupported active lighting before callbacks',async t=>{
  const f=fixture(t);f.scene.add(new T.RectAreaLight(),new T.LightProbe());
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length,1,'unlit authored graphs do not consume scene lights');
  const g=fixture(t),material=new T.MeshToonNodeMaterial(),area=new T.RectAreaLight();let updates=0;
  material.colorNode=T.TSL.uniform(new T.Color(.25,.5,.75)).onObjectUpdate(()=>{updates++;});
  g.mesh.material=material;g.scene.add(area);g.renderer.lighting.enabled=false;
  await g.renderer.compileAsync(g.scene,g.camera);assert.equal(updates,0);
  g.renderer.render(g.scene,g.camera);await drain(g.renderer);assert.equal(updates,1);
  const submissions=g.device.submissions.length,acquired=g.acquired();g.renderer.lighting.enabled=true;
  assert.throws(()=>g.renderer.render(g.scene,g.camera),{code:'THREE_NODE_PROGRAM_LIGHTS'});
  assert.equal(updates,1);assert.equal(g.device.submissions.length,submissions);assert.equal(g.acquired(),acquired);
  area.visible=false;g.scene.add(new T.AmbientLight(0xffffff,.5));
  g.renderer.render(g.scene,g.camera);await drain(g.renderer);
  assert.equal(updates,2);assert.equal(nativeDraws(g.device).length,2);
  assert.equal(g.renderer._nodeCaptureBytes,0);
});

test('native Standard and Physical graphs share the source half-float DFG texture through the facade',async t=>{
  const f=fixture(t),standard=new T.MeshStandardNodeMaterial({roughness:.4,metalness:.6,emissive:new T.Color(.125,.25,.375)}),
    physical=new T.MeshPhysicalNodeMaterial({roughness:.3,metalness:.2,clearcoat:.7,clearcoatRoughness:.15,
      sheen:.4,sheenColor:new T.Color(.2,.5,.7),iridescence:.5,anisotropy:.35,retroreflectivity:.2,
      emissive:new T.Color(.625,.375,.125)});
  standard.colorNode=f.tint;physical.colorNode=T.TSL.vec3(.7,.3,.2);
  f.mesh.material=standard;const second=new T.Mesh(f.mesh.geometry,physical);second.position.x=.4;f.scene.add(second);
  const light=new T.DirectionalLight(0xffffff,2);light.position.set(2,3,4);f.scene.add(light,new T.AmbientLight(0xffffff,.2));
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(f.device.textureWrites.length,0,'compilation captures no DFG upload');
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(containsColor(draws[0],[.125,.25,.375]));assert.ok(containsColor(draws[1],[.625,.375,.125]));
  const lookupViews=draws.flatMap(textureResources).filter(view=>view.texture.format==='rg16float');
  assert.ok(lookupViews.length>=2,'each PBR draw binds the actual source DFG lookup');
  assert.equal(new Set(lookupViews.map(view=>view.texture)).size,1,'source lookup aliases share one GPU allocation');
  assert.equal(lookupViews[0].texture.width,16);assert.equal(lookupViews[0].texture.height,16);
  assert.equal(lookupViews[0].texture.pixels.byteLength,16*16*2*2);
  assert.equal(f.device.textureWrites.filter(write=>write.destination.texture.format==='rg16float').length,1);
  assert.equal(f.renderer._nodeCaptureBytes,0);await f.renderer.waitForGPU();
});

test('queued Physical feature changes keep earlier programs and shared lookup storage',async t=>{
  const f=fixture(t),material=new T.MeshPhysicalNodeMaterial({roughness:.4});
  const roughness=T.TSL.uniform(.234375).setName('queuedPbrRoughness');
  material.colorNode=f.tint;material.roughnessNode=roughness;f.mesh.material=material;
  f.scene.add(new T.DirectionalLight(0xffffff,2));await f.renderer.init();
  f.renderer.render(f.scene,f.camera);
  material.clearcoat=.75;roughness.value=.703125;f.renderer.render(f.scene,f.camera);
  roughness.value=.421875;f.renderer.render(f.scene,f.camera);
  material.clearcoat=0;roughness.value=.9375;
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.notEqual(draws[0].pipeline,draws[1].pipeline,'enabling a source Physical feature selects its shader branch');
  assert.equal(draws[1].pipeline,draws[2].pipeline,'live roughness changes keep the shader');
  const scalarPresent=(draw,value)=>[...draw.groups.values()].some(({group,offsets})=>group.entries.some(({resource})=>{
    if(!resource?.buffer)return false;const bytes=draw.contents.get(resource.buffer);if(!bytes)return false;
    const words=new Float32Array(bytes.buffer,bytes.byteOffset+(offsets[0]??0),resource.size/4);return words.includes(value);
  }));
  assert.ok(scalarPresent(draws[0],.234375));assert.ok(scalarPresent(draws[1],.703125));assert.ok(scalarPresent(draws[2],.421875));
  assert.ok(draws.every(draw=>!scalarPresent(draw,.9375)),'later CPU edits cannot replace queued uniform values');
  assert.equal(new Set(draws.flatMap(textureResources).map(view=>view.texture)).size,1);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native storage captures queued upload versions and ignores CPU edits without needsUpdate',async t=>{
  const f=fixture(t),attribute=new T.StorageBufferAttribute(new Float32Array([.125,.375,.625,1]),4);
  f.material.colorNode=T.TSL.storage(attribute,'vec4',1).toReadOnly().element(0).rgb;
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  attribute.array.set([.875,.625,.375,1]);attribute.needsUpdate=true;
  f.renderer.render(f.scene,f.camera);
  attribute.array.fill(0);f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(storageWords),[[[.125,.375,.625,1]],[[.875,.625,.375,1]],[[.875,.625,.375,1]]]);
  const buffer=storageResources(draws[0])[0].buffer;
  assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,2,
    'the later unchanged version must not re-upload source CPU data');
  assert.equal(new Set(draws.flatMap(storageResources).map(resource=>resource.buffer)).size,1,
    'sequential source renders share the renderer-owned storage allocation');
  assert.equal(f.renderer._nodeCaptureBytes,0);await f.renderer.waitForGPU();
});

test('native storage partial uploads preserve GPU-visible bytes outside requested ranges',async t=>{
  const f=fixture(t),attribute=new T.StorageBufferAttribute(new Float32Array([.125,.25,.5,1,.75,.5,.25,1]),4);
  f.material.colorNode=T.TSL.storage(attribute,'vec4',2).toReadOnly().element(0).rgb;
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  attribute.array.fill(99);attribute.array.set([.25,.75,.5,1],4);
  attribute.addUpdateRange(4,4);attribute.needsUpdate=true;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(storageWords(nativeDraws(f.device).at(-1)),[[.125,.25,.5,1,.25,.75,.5,1]]);
  assert.equal(attribute.array[0],99,'public CPU contents stay independent of GPU visibility');
  assert.equal(attribute.updateRanges.length,0);
  const buffer=storageResources(nativeDraws(f.device).at(-1))[0].buffer;
  const writes=f.device.writes.filter(write=>write.buffer===buffer);
  assert.deepEqual(writes.map(write=>[write.offset,write.input.byteLength]),[[0,32],[16,16]]);
  await f.renderer.waitForGPU();
});

test('native storage compilation preserves source updates until the logical render',async t=>{
  const f=fixture(t),attribute=new T.StorageBufferAttribute(new Float32Array([.25,.5,.75,1]),4);
  let updates=0;
  const gain=T.TSL.uniform(1).onObjectUpdate(()=>{updates++;return 1;});
  f.material.colorNode=T.TSL.storage(attribute,'vec4',1).toReadOnly().element(0).rgb.mul(gain);
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(updates,0);
  assert.equal(f.device.writes.filter(write=>write.buffer.label==='f3d-compute-buffer').length,0);
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const buffer=storageResources(nativeDraws(f.device)[0])[0].buffer;
  assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,1);assert.equal(updates,1);
  assert.deepEqual(storageWords(nativeDraws(f.device)[0]),[[.25,.5,.75,1]]);
});

test('native graphics borrows compute storage without uploading stale CPU values or owning a second copy',async t=>{
  const f=fixture(t),attribute=new T.StorageBufferAttribute(new Float32Array([0,0,0,0]),4);
  f.device.createComputePipeline=descriptor=>({...descriptor,getBindGroupLayout:()=>({})});
  f.device.createComputePipelineAsync=async descriptor=>f.device.createComputePipeline(descriptor);
  const writable=T.TSL.storage(attribute,'vec4',1);
  const kernel=T.TSL.Fn(()=>{writable.element(0).assign(T.TSL.vec4(.25,.5,.75,1));})().compute(1);
  await f.renderer.compileComputeAsync(kernel);
  const buffer=f.device.buffers.find(value=>value.usage&128);
  assert.ok(buffer,'compute preparation publishes actual owned storage');
  // Model GPU-produced bytes at the recording boundary; the browser parity
  // scenario executes the actual kernel and checks its rendered result.
  f.device.queue.writeBuffer(buffer,0,new Float32Array([.25,.5,.75,1]));
  const storageWrites=f.device.writes.filter(write=>write.buffer===buffer).length;
  f.material.colorNode=T.TSL.storage(attribute,'vec4',1).toReadOnly().element(0).rgb;
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draw=nativeDraws(f.device)[0];assert.equal(storageResources(draw)[0].buffer,buffer);
  assert.deepEqual(storageWords(draw),[[.25,.5,.75,1]]);assert.deepEqual([...attribute.array],[0,0,0,0]);
  assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,storageWrites);
  assert.equal(f.device.buffers.filter(value=>value.label==='f3d-compute-buffer'&&(value.usage&128)).length,1);
  await f.renderer.waitForGPU();f.renderer.dispose();assert.equal(buffer.destroyed,true);
});

test('queued native instances preserve post-update Mesh.count and skip empty draws without skipping callbacks',async t=>{
  const f=fixture(t),requested=[3,0,2],updates=[],after=[];
  const tint=T.TSL.uniform(new T.Color(.25,.5,.75)).onObjectUpdate(frame=>{
    frame.object.count=requested.shift();updates.push(frame.object.count);return new T.Color(.25,.5,.75);
  });
  tint.updateAfterType='object';tint.updateAfter=frame=>{after.push(frame.object.count);frame.object.count=99;};
  f.material.colorNode=tint;
  f.material.positionNode=T.TSL.positionLocal.add(T.TSL.vec3(T.TSL.instanceIndex.toFloat().mul(.2),0,0));
  await f.renderer.init();
  for(let i=0;i<3;i++)f.renderer.render(f.scene,f.camera);
  assert.deepEqual(updates,[3,0,2]);assert.deepEqual(after,[3,0,2]);
  f.mesh.count=42;
  await drain(f.renderer);await f.renderer.waitForGPU();
  const draws=nativeDraws(f.device);
  assert.deepEqual(draws.map(draw=>draw.args[1]),[3,2]);
  assert.ok(draws.every(draw=>draw.args[4]===0),'native instance indices start at zero for each draw');
  assert.equal(draws[0].pipeline,draws[1].pipeline,'count updates retain the native shader');
  assert.equal(f.renderer.info.render.drawCalls,2);assert.equal(f.renderer._nodeCaptureBytes,0);
  assert.deepEqual(updates,[3,0,2]);assert.deepEqual(after,[3,0,2]);
});

function recordCompute(device){
  device.computePasses=[];
  device.createComputePipeline=descriptor=>({...descriptor,getBindGroupLayout:index=>({index})});
  device.createComputePipelineAsync=descriptor=>Promise.resolve(device.createComputePipeline(descriptor));
  const create=device.createCommandEncoder.bind(device);
  device.createCommandEncoder=descriptor=>{
    const encoder=create(descriptor),passes=[];
    return {...encoder,beginComputePass(desc){
      const pass={desc,draws:[],groups:new Map(),dispatches:[]};passes.push(pass);device.computePasses.push(pass);
      return {setPipeline(pipeline){pass.pipeline=pipeline;},setBindGroup(index,group){pass.groups.set(index,group);},
        dispatchWorkgroups(...size){pass.dispatches.push(size);},end(){}};
    },finish(){return [...encoder.finish(),...passes];}};
  };
}
function storageKernel(storage){
  return T.TSL.Fn(()=>{storage.element(T.TSL.instanceIndex).addAssign(T.TSL.vec4(0));})().compute(4);
}
const storageResources=draw=>[...draw.groups.values()].flatMap(({group})=>group.entries)
  .map(entry=>entry.resource).filter(resource=>resource?.buffer && (resource.buffer.usage&128));

test('native storage borrows queued compute versions in FIFO order without graphics uploads',async t=>{
  const f=fixture(t);recordCompute(f.device);
  const attribute=new T.StorageBufferAttribute(4,4),storage=T.TSL.storage(attribute,'vec4',4),kernel=storageKernel(storage);
  for(let i=0;i<4;i++)attribute.array.set([.125,.25,.375,1],i*4);
  f.material.colorNode=storage.element(0).xyz;
  f.material.positionNode=T.TSL.positionLocal.add(storage.element(T.TSL.vertexIndex).xyz.mul(.01));
  await f.renderer.init();
  const firstCompute=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  for(let i=0;i<4;i++)attribute.array.set([.625,.75,.875,1],i*4);
  attribute.needsUpdate=true;
  const secondCompute=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  attribute.array.fill(99);
  await Promise.all([firstCompute,secondCompute,drain(f.renderer)]);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);assert.equal(f.device.computePasses.length,2);
  const resources=draws.map(storageResources);
  assert.ok(resources.every(bindings=>bindings.length===2),'source vertex and fragment aliases retain both binding slots');
  const buffer=resources[0][0].buffer;
  assert.ok(resources.flat().every(resource=>resource.buffer===buffer));
  assert.deepEqual(draws.map(draw=>Array.from(new Float32Array(draw.contents.get(buffer).buffer)).slice(0,4)),
    [[.125,.25,.375,1],[.625,.75,.875,1]]);
  assert.equal(captureComputeStorage(f.device,attribute).resolve().buffer,buffer);
  assert.ok(f.device.computePasses.every(pass=>[...pass.groups.values()].some(group=>
    group.entries.some(entry=>entry.resource?.buffer===buffer))),'graphics uses the exact compute allocation');
  assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,2,'only the two compute source versions upload');
  assert.equal(f.device.buffers.some(candidate=>candidate.usage&1),false,'graphics creates no storage readback buffer');
  assert.equal(storage.access,'readWrite');
  assert.ok(f.device.computePasses.every(pass=>/var<storage,\s*read_write>/.test(pass.pipeline.compute.module.code)));
  assert.equal(f.renderer._nodeCaptureBytes,0);await f.renderer.waitForGPU();
});

test('native storage compilation precedes publication and compileComputeAsync supplies the sole buffer',async t=>{
  const f=fixture(t);recordCompute(f.device);
  const attribute=new T.StorageBufferAttribute(4,4),storage=T.TSL.storage(attribute,'vec4',4),kernel=storageKernel(storage);
  attribute.array.fill(.375);let updates=0;
  const gain=T.TSL.uniform(1).onObjectUpdate(()=>{updates++;return 1;});
  f.material.fragmentNode=storage.element(0).mul(gain);
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(updates,0);assert.throws(captureComputeStorage(f.device,attribute).resolve,/UNPREPARED/);
  await f.renderer.compileComputeAsync(kernel);
  assert.equal(f.device.computePasses.length,0,'compute compilation publishes upload storage without dispatching');
  const buffer=captureComputeStorage(f.device,attribute).resolve().buffer,writes=f.device.writes.filter(write=>write.buffer===buffer).length;
  assert.ok(buffer);assert.equal(writes,1);
  attribute.array.fill(99);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draw=nativeDraws(f.device)[0];assert.equal(updates,1);assert.equal(storageResources(draw)[0].buffer,buffer);
  assert.equal(new Float32Array(draw.contents.get(buffer).buffer)[0],.375);
  assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,writes);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('storage-only node callbacks cannot enqueue compute inside the draw capture transaction',async t=>{
  const f=fixture(t);recordCompute(f.device);
  const attribute=new T.StorageBufferAttribute(4,4),storage=T.TSL.storage(attribute,'vec4',4),kernel=storageKernel(storage);
  const gain=T.TSL.uniform(1).onObjectUpdate(()=>{f.renderer.compute(kernel);return 1;});
  f.material.fragmentNode=storage.element(0).mul(gain);
  await f.renderer.compileComputeAsync(kernel);
  const writes=f.device.writes.length,submissions=f.device.submissions.length,acquired=f.acquired();
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_COMPUTE_REENTRY'});
  assert.equal(f.device.writes.length,writes);assert.equal(f.device.submissions.length,submissions);
  assert.equal(f.acquired(),acquired);assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.device.computePasses.length,0);
});

function storageInstances(f,count=4){
  const mesh=new T.InstancedMesh(f.mesh.geometry,f.material,count);
  const matrices=new T.StorageInstancedBufferAttribute(count,16),colors=new T.StorageInstancedBufferAttribute(count,3);
  for(let i=0;i<count;i++){
    new T.Matrix4().makeScale(1+i*.25,.75,1).setPosition(i*.5,0,0).toArray(matrices.array,i*16);
    colors.array.set([.125+i*.125,.5,.75],i*3);
  }
  mesh.instanceMatrix=matrices;mesh.instanceColor=colors;mesh.frustumCulled=false;
  f.scene.remove(f.mesh);f.scene.add(mesh);
  return {mesh,matrices,colors};
}

function cpuInstances(f,count=3){
  const mesh=new T.InstancedMesh(f.mesh.geometry,f.material,count);
  for(let i=0;i<count;i++){
    mesh.setMatrixAt(i,new T.Matrix4().makeScale(1+i*.25,.75,1).setPosition(i*.5,0,0));
    mesh.setColorAt(i,new T.Color(.125+i*.125,.5,.75));
  }
  mesh.frustumCulled=false;f.scene.remove(f.mesh);f.scene.add(mesh);
  return {mesh,matrices:mesh.instanceMatrix,colors:mesh.instanceColor};
}
function instanceStreams(draw){
  return draw.pipeline.vertex.buffers.flatMap((layout,slot)=>{
    if(layout.stepMode!=='instance')return [];
    const buffer=draw.streams.get(slot),bytes=draw.contents.get(buffer);
    return [{layout,buffer,values:[...new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)]}];
  });
}

test('ordinary InstancedMesh captures unflagged matrix uniforms and versioned RGB streams before queued preparation',async t=>{
  const f=fixture(t),{mesh,matrices,colors}=cpuInstances(f);
  const geometryKeys=Object.keys(mesh.geometry.attributes),originalMatrices=matrices.array,originalColors=colors.array;
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  matrices.array[12]=1.5;colors.array[0]=.875;colors.needsUpdate=true;mesh.count=2;
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  mesh.count=0;f.renderer.render(f.scene,f.camera);
  mesh.count=99;matrices.array.fill(99);colors.array.fill(99);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.deepEqual(draws.map(draw=>draw.args[1]),[3,2]);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===48)?.[12]),[0,1.5]);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw).map(stream=>stream.values[0])),[[.125],[.875]]);
  assert.ok(draws.every(draw=>instanceStreams(draw)[0].layout.arrayStride===12));
  assert.equal(draws[0].pipeline,draws[1].pipeline);assert.equal(matrices.version,0);
  assert.equal(matrices.array,originalMatrices);assert.equal(colors.array,originalColors);
  assert.deepEqual(Object.keys(mesh.geometry.attributes),geometryKeys);
  assert.equal(matrices.array[12],99);assert.equal(colors.array[0],99);
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.renderer.info.f3d.presentedRenders,3);
});

test('large ordinary instance matrices use captured interleaved columns and preserve partial update history',async t=>{
  const f=fixture(t);f.device.limits.maxUniformBufferBindingSize=65536;
  const {mesh,matrices,colors}=cpuInstances(f,1025);mesh.count=3;
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  matrices.array[12]=2;matrices.array[28]=99;matrices.addUpdateRange(12,1);matrices.needsUpdate=true;
  colors.array[0]=.875;colors.array[3]=99;colors.addUpdateRange(0,3);colors.needsUpdate=true;mesh.count=2;
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  assert.equal(matrices.updateRanges.length,0);assert.equal(colors.updateRanges.length,0);
  matrices.array.fill(99);colors.array.fill(99);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.deepEqual(draws.map(draw=>draw.args[1]),[3,2]);
  const matrixStreams=draws.map(draw=>instanceStreams(draw).find(stream=>stream.layout.arrayStride===64));
  assert.ok(matrixStreams.every(Boolean));assert.ok(matrixStreams.every(stream=>stream.layout.attributes.length===4));
  assert.deepEqual(matrixStreams.map(stream=>[stream.values[12],stream.values[28]]),[[0,.5],[2,.5]]);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw).find(stream=>stream.layout.arrayStride===12).values.slice(0,4)),
    [[.125,.5,.75,.25],[.875,.5,.75,.25]]);
  assert.ok(draws.every(draw=>uniformBlocks(draw).every(words=>words.length<=16384)));
  assert.equal(draws[0].pipeline,draws[1].pipeline);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('ordinary instance colors upload before current node callbacks and retain history across graph replacement',async t=>{
  const f=fixture(t),{mesh,colors}=cpuInstances(f,2);let next=.875;
  const gain=T.TSL.uniform(1).onObjectUpdate(()=>{
    if(next!==null){colors.array[0]=next;colors.needsUpdate=true;}return 1;
  });
  f.material.colorNode=f.tint.mul(gain);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  next=null;f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  colors.array[0]=99;
  f.material.colorNode=T.TSL.vec3(.5,.75,1);f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw)[0].values[0]),[.125,.875,.875]);
  assert.notEqual(draws[1].pipeline,draws[2].pipeline);assert.equal(colors.array[0],99);
  assert.equal(mesh.instanceColor,colors);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('compileAsync preserves the initial instance-color upload without capturing a hidden instance',async t=>{
  const f=fixture(t),{mesh,colors}=cpuInstances(f,2);
  const hidden=new T.InstancedMesh(mesh.geometry,f.material,2);hidden.frustumCulled=false;hidden.visible=false;
  for(let index=0;index<2;index++){
    hidden.setMatrixAt(index,new T.Matrix4());hidden.setColorAt(index,new T.Color(.25,.5,.75));
  }
  f.scene.add(hidden);await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(nativeDraws(f.device).length,0);
  colors.array[0]=.875;hidden.instanceColor.array[0]=.625;hidden.visible=true;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw)[0].values[0]),[.125,.625]);
  assert.equal(colors.version,0);assert.equal(hidden.instanceColor.version,0);
  assert.equal(colors.array[0],.875);assert.equal(hidden.instanceColor.array[0],.625);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('graph replacement waits for submitted generated streams before retiring their GPU owner',async t=>{
  const f=fixture(t);cpuInstances(f,2);await f.renderer.compileAsync(f.scene,f.camera);
  let finish;f.device.completion=new Promise(resolve=>{finish=resolve;});
  f.renderer.render(f.scene,f.camera);
  const first=nativeDraws(f.device)[0],oldBuffer=instanceStreams(first)[0].buffer;
  f.material.colorNode=T.TSL.vec3(.5,.75,1);f.renderer.render(f.scene,f.camera);
  let settled=false;
  const pending=drain(f.renderer).then(()=>{settled=true;return null;},error=>{settled=true;return error;});
  try{
    // Let pipeline preparation reach its retirement boundary while the first
    // submission still holds the old geometry as an outstanding dependency.
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(settled,false);assert.equal(nativeDraws(f.device).length,1);
    assert.equal(oldBuffer.destroyed,false);
    finish();assert.equal(await pending,null);await f.renderer.waitForGPU();
    const draws=nativeDraws(f.device);assert.equal(draws.length,2);
    assert.notEqual(draws[0].pipeline,draws[1].pipeline);assert.equal(oldBuffer.destroyed,true);
    assert.deepEqual(draws.map(draw=>instanceStreams(draw)[0].values[0]),[.125,.125]);
    assert.equal(f.renderer._nodeCaptureBytes,0);
  }finally{finish();await pending;}
});

test('native InstancedMesh borrows compute matrices and colors while retaining queued counts and source normals',async t=>{
  const f=fixture(t);recordCompute(f.device);
  const {mesh,matrices,colors}=storageInstances(f),offsets=new T.StorageInstancedBufferAttribute(4,4);
  const matrixNode=T.TSL.storage(matrices,'mat4',4),colorNode=T.TSL.storage(colors,'vec3',4),
    offsetNode=T.TSL.storage(offsets,'vec4',4);
  const kernel=T.TSL.Fn(()=>{
    const index=T.TSL.instanceIndex;
    matrixNode.element(index).assign(matrixNode.element(index));
    colorNode.element(index).addAssign(T.TSL.vec3(0));
    offsetNode.element(index).addAssign(T.TSL.vec4(0));
  })().compute(4);
  const requested=[3,1,0],updates=[],after=[];
  const gain=T.TSL.uniform(1).onObjectUpdate(frame=>{
    frame.object.count=requested.shift();updates.push(frame.object.count);return 1;
  });
  gain.updateAfterType='object';gain.updateAfter=frame=>{after.push(frame.object.count);frame.object.count=4;};
  const material=new T.MeshPhongNodeMaterial();material.colorNode=T.TSL.vec3(.8,.6,.4).mul(gain);
  material.positionNode=T.TSL.positionLocal.add(offsetNode.element(T.TSL.instanceIndex).xyz);
  mesh.material=material;f.scene.add(new T.DirectionalLight(0xffffff,2));
  await f.renderer.compileAsync(f.scene,f.camera);assert.deepEqual(updates,[]);
  const first=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  matrices.array[12]=1.5;matrices.needsUpdate=true;
  colors.array[0]=.875;colors.needsUpdate=true;offsets.array[0]=.25;offsets.needsUpdate=true;
  const second=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  f.renderer.render(f.scene,f.camera);
  mesh.count=99;matrices.array.fill(99);colors.array.fill(99);offsets.array.fill(99);
  await Promise.all([first,second,drain(f.renderer)]);await f.renderer.waitForGPU();
  const draws=nativeDraws(f.device);assert.deepEqual(draws.map(draw=>draw.args[1]),[3,1]);
  assert.deepEqual(updates,[3,1,0]);assert.deepEqual(after,[3,1,0]);
  assert.equal(draws[0].pipeline,draws[1].pipeline,'count, upload version and vec3 normalization retain the program');
  const buffers=[matrices,colors,offsets].map(attribute=>captureComputeStorage(f.device,attribute).resolve().buffer);
  for(const draw of draws)assert.ok(buffers.every(buffer=>storageResources(draw).some(resource=>resource.buffer===buffer)));
  const words=(draw,buffer)=>new Float32Array(draw.contents.get(buffer).buffer);
  assert.deepEqual(draws.map(draw=>words(draw,buffers[0])[12]),[0,1.5]);
  assert.deepEqual(draws.map(draw=>words(draw,buffers[1])[0]),[.125,.875]);
  assert.deepEqual(draws.map(draw=>words(draw,buffers[2])[0]),[0,.25]);
  for(const buffer of buffers){
    assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,2,'only captured host versions upload');
    assert.ok(f.device.computePasses.every(pass=>[...pass.groups.values()].some(group=>
      group.entries.some(entry=>entry.resource?.buffer===buffer))));
  }
  assert.match(draws[0].pipeline.vertex.module.code,/@builtin\(\s*instance_index\s*\)/);
  assert.match(draws[0].pipeline.vertex.module.code,/array<\s*mat4x4<f32>/);
  assert.equal(f.renderer.info.render.drawCalls,2);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native InstancedMesh captures replacement storage identities before later queued changes',async t=>{
  const f=fixture(t),{mesh,matrices:a,colors:firstColor}=storageInstances(f,2);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  const b=new T.StorageInstancedBufferAttribute(a.array.slice(),16),secondColor=new T.StorageInstancedBufferAttribute(2,3);
  b.array[12]=2;secondColor.array.set([.875,.25,.5,.5,.75,.25]);
  mesh.instanceMatrix=b;mesh.instanceColor=secondColor;mesh.count=1;f.renderer.render(f.scene,f.camera);
  mesh.instanceMatrix=new T.StorageInstancedBufferAttribute(2,16);mesh.instanceColor=null;mesh.count=99;
  await drain(f.renderer);await f.renderer.waitForGPU();
  const draws=nativeDraws(f.device);assert.deepEqual(draws.map(draw=>draw.args[1]),[2,1]);
  for(const [index,attribute,color,translation] of [[0,a,firstColor,0],[1,b,secondColor,2]]){
    const buffer=captureComputeStorage(f.device,attribute).resolve().buffer;
    const colorBuffer=captureComputeStorage(f.device,color).resolve().buffer;
    assert.ok(storageResources(draws[index]).some(resource=>resource.buffer===buffer));
    assert.ok(storageResources(draws[index]).some(resource=>resource.buffer===colorBuffer));
    assert.equal(new Float32Array(draws[index].contents.get(buffer).buffer)[12],translation);
  }
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native InstancedMesh refuses post-update capacity overflow before any GPU submission',async t=>{
  const f=fixture(t),{mesh}=storageInstances(f,2);await f.renderer.init();
  f.material.colorNode=T.TSL.vec3(T.TSL.uniform(1).onObjectUpdate(frame=>{frame.object.count=3;return 1;}));
  const writes=f.device.writes.length,submissions=f.device.submissions.length;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_OBJECT'});
  assert.equal(mesh.count,3);assert.equal(f.acquired(),0);
  assert.equal(f.device.writes.length,writes);assert.equal(f.device.submissions.length,submissions);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('shared native material keeps object-specific instance storage and validates the smaller color capacity',async t=>{
  const f=fixture(t),left=storageInstances(f,4),right=storageInstances(f,2);
  left.mesh.renderOrder=-1;right.matrices.array[12]=3;
  left.mesh.userData.drawCount=2;right.mesh.userData.drawCount=1;
  left.mesh.count=right.mesh.count=99;
  const gain=T.TSL.uniform(1).onObjectUpdate(frame=>{frame.object.count=frame.object.userData.drawCount;return 1;});
  gain.updateAfterType='object';gain.updateAfter=frame=>{frame.object.count=99;};
  f.material.colorNode=T.TSL.vec3(.5).mul(gain);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.deepEqual(draws.map(draw=>draw.args[1]),[2,1]);
  const leftBuffer=captureComputeStorage(f.device,left.matrices).resolve().buffer,
    rightBuffer=captureComputeStorage(f.device,right.matrices).resolve().buffer;
  assert.notEqual(leftBuffer,rightBuffer);
  assert.ok(storageResources(draws[0]).some(resource=>resource.buffer===leftBuffer));
  assert.ok(storageResources(draws[1]).some(resource=>resource.buffer===rightBuffer));
  assert.ok(!storageResources(draws[0]).some(resource=>resource.buffer===rightBuffer));
  assert.ok(!storageResources(draws[1]).some(resource=>resource.buffer===leftBuffer));
  left.mesh.instanceColor=new T.StorageInstancedBufferAttribute(1,3);
  const writes=f.device.writes.length,submissions=f.device.submissions.length,acquired=f.acquired();
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_OBJECT'});
  assert.equal(f.device.writes.length,writes);assert.equal(f.device.submissions.length,submissions);
  assert.equal(f.acquired(),acquired);assert.equal(f.renderer._nodeCaptureBytes,0);
});


test('source Physical H7 material maps and compute-authored instance draws retain queued versions',async t=>{
  const f=fixture(t);recordCompute(f.device);
  const a=dataTexture(240,30,10),b=dataTexture(20,60,230),specular=dataTexture(120,180,220),metalRough=dataTexture(30,160,110);
  a.colorSpace=b.colorSpace=specular.colorSpace=T.SRGBColorSpace;
  const material=new T.MeshPhysicalMaterial({map:a,specularColorMap:specular,roughnessMap:metalRough,metalnessMap:metalRough,
    normalMap:dataTexture(128,128,255),metalness:.5,ior:1.45,side:T.DoubleSide});
  const vertexCount=f.mesh.geometry.attributes.position.count,attribute=new T.StorageBufferAttribute(2*vertexCount*2,4),
    storage=T.TSL.storage(attribute,'vec4',attribute.count),index=T.TSL.instanceIndex.mul(T.TSL.uint(vertexCount)).add(T.TSL.vertexIndex).mul(T.TSL.uint(2));
  attribute.array.fill(.125);let callbacks=0;
  const gain=T.TSL.uniform(1).onObjectUpdate(context=>{
    assert.equal(context.material,material);assert.equal(context.object,f.mesh);callbacks++;return 1;
  });
  material.positionNode=storage.element(index).xyz.mul(gain);
  material.normalNode=T.TSL.transformNormalToView(storage.element(index.add(T.TSL.uint(1))).xyz).toVarying();
  f.mesh.material=material;f.mesh.count=2;f.mesh.frustumCulled=false;
  f.scene.add(new T.DirectionalLight(0xffffff,2),new T.AmbientLight(0xffffff,.25));
  const kernel=T.TSL.Fn(()=>{storage.element(T.TSL.instanceIndex).addAssign(T.TSL.vec4(0));})().compute(attribute.count);
  await f.renderer.init();
  const firstCompute=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  material.map=b;attribute.array.fill(.625);attribute.needsUpdate=true;f.mesh.count=1;
  const secondCompute=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  a.image.data.fill(0);b.image.data.fill(0);attribute.array.fill(99);f.mesh.count=12;
  assert.equal(callbacks,2);assert.equal(f.device.textureWrites.length,0);
  await Promise.all([firstCompute,secondCompute,drain(f.renderer)]);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);assert.deepEqual(draws.map(draw=>draw.args[1]),[2,1]);
  assert.equal(draws[0].pipeline,draws[1].pipeline,'compatible material-map replacements keep the native program');
  assert.match(draws[0].pipeline.fragment.module.code,/fn D_GGX/);
  const textures=draws.map(textureResources);
  assert.ok(textures.every(views=>views.length===4),'map, specular color, shared metal/roughness map and DFG');
  const colorPixels=textures.map(views=>views.filter(view=>view.format==='rgba8unorm-srgb').map(view=>[...view.texture.pixels]));
  assert.deepEqual(colorPixels[0],[[240,30,10,255],[120,180,220,255]]);
  assert.deepEqual(colorPixels[1],[[20,60,230,255],[120,180,220,255]]);
  const buffer=storageResources(draws[0])[0].buffer;
  assert.equal(storageResources(draws[1])[0].buffer,buffer);
  assert.deepEqual(draws.map(draw=>new Float32Array(draw.contents.get(buffer).buffer)[0]),[.125,.625]);
  assert.equal(captureComputeStorage(f.device,attribute).resolve().buffer,buffer);
  assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,2,'graphics does not overwrite queued compute data');
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(callbacks,2);await f.renderer.waitForGPU();
});

test('source material arrays preserve per-group maps while ordinary materials retain fixed rendering',async t=>{
  const f=fixture(t),a=new T.MeshStandardMaterial({map:dataTexture(220,30,70)}),b=new T.MeshPhysicalMaterial({map:dataTexture(40,170,230)});
  a.positionNode=b.positionNode=T.TSL.positionLocal;
  f.mesh.geometry.clearGroups();f.mesh.geometry.addGroup(0,3,0);f.mesh.geometry.addGroup(3,3,1);f.mesh.material=[a,b];
  const ordinary=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial({color:0xff0000}));ordinary.position.z=.25;f.scene.add(ordinary);
  f.scene.add(new T.AmbientLight(0xffffff,.5));
  await f.renderer.compileAsync(f.scene,f.camera);assert.equal(f.device.textureWrites.length,0);
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const native=nativeDraws(f.device);assert.equal(native.length,2);
  assert.deepEqual(native.map(draw=>draw.args[0]),[3,3]);
  assert.deepEqual(native.map(draw=>[...textureResources(draw).find(view=>view.texture.format==='rgba8unorm').texture.pixels]),
    [[220,30,70,255],[40,170,230,255]]);
  assert.equal(f.device.snapshots.flat().filter(draw=>draw.pipeline.vertex.entryPoint==='vertex_main').length,1);
  assert.equal(f.mesh.material[0],a);assert.equal(f.mesh.material[1],b);assert.equal(f.renderer._nodeCaptureBytes,0);
});


test('registered SunLight captures source color, direction and FRAME frequency through native queued draws',async t=>{
  const f=fixture(t),material=new T.MeshPhysicalMaterial(),secondMaterial=new T.MeshLambertNodeMaterial();
  material.colorNode=f.tint;secondMaterial.colorNode=T.TSL.vec3(.5);f.mesh.material=material;
  const second=new T.Mesh(f.mesh.geometry,secondMaterial);second.position.x=.5;f.scene.add(second);
  const sun=new SunLight(new T.Color(.125,.25,.5),2),group=new T.Group();
  group.position.set(1,2,3);sun.position.set(-4,10,8);group.add(sun);f.scene.add(group);
  f.renderer.library.addLight(SunLightNode,SunLight);await f.renderer.init();
  f.renderer.render(f.scene,f.camera);
  sun.intensity=3;sun.position.set(-1,4,2);material.colorNode=T.TSL.vec3(.375);
  f.renderer.render(f.scene,f.camera);
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  sun.intensity=9;sun.position.set(90,90,90);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,6);
  for(const draw of draws.slice(0,4))assert.ok(containsColor(draw,[.25,.5,1]));
  for(const draw of draws.slice(4))assert.ok(containsColor(draw,[.375,.75,1.5]));
  for(const draw of draws.slice(0,2))assert.ok(containsColor(draw,[-3,12,11]));
  for(const draw of draws.slice(2))assert.ok(containsColor(draw,[0,6,5]));
  assert.notEqual(draws[0].pipeline,draws[2].pipeline,'graph recompilation keeps the source light update frequency');
  assert.equal(sun._listeners.dispose.length,1);assert.equal(sun.shadow.map,null);
  assert.equal(f.renderer._nodeCaptureBytes,0);await f.renderer.waitForGPU();
});

test('registered SunLight refuses unsupported shadow effects before updates and honors disabled shadow maps',async t=>{
  const f=fixture(t),material=new T.MeshStandardMaterial(),sun=new SunLight();let updates=0;
  material.colorNode=T.TSL.uniform(new T.Color(.25,.5,.75)).onObjectUpdate(()=>{updates++;});
  f.mesh.material=material;f.scene.add(sun);await f.renderer.init();
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_LIGHTS'});
  f.renderer.library.addLight(SunLightNode,SunLight);sun.castShadow=true;
  f.mesh.receiveShadow=true;f.renderer.shadowMap.enabled=true;f.renderer.shadowMap.type=T.VSMShadowMap;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_SHADOW'});
  f.renderer.shadowMap.type=T.PCFShadowMap;sun.shadow.filterNode=T.TSL.float(1);
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_SHADOW'});sun.shadow.filterNode=null;
  f.renderer.shadowMap.transmitted=true;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_SHADOW'});
  f.renderer.shadowMap.transmitted=false;
  assert.equal(updates,0);assert.equal(f.acquired(),0);assert.equal(f.device.submissions.length,0);
  assert.equal(f.renderer._nodeCaptureBytes,0);
  f.renderer.shadowMap.enabled=false;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);assert.equal(updates,1);assert.equal(nativeDraws(f.device).length,1);
  assert.equal(sun.shadow.map,null);
});
