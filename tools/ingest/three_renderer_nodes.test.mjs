/** Native TSL through the actual facade, scene bridge and GPU command owner.
 * The recording device proves per-call capture and queue behavior; browser
 * source comparisons separately execute the generated shaders. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
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
function containsColor(draw,color){
  for(const {group,offsets} of draw.groups.values()){
    const resource=group.entries.find(entry=>entry.resource?.buffer)?.resource;if(!resource)continue;
    const {buffer,size}=resource,bytes=draw.contents.get(buffer),offset=offsets[0]??0;
    const words=new Float32Array(bytes.buffer,bytes.byteOffset+offset,size/4);
    for(let i=0;i<=words.length-3;i++)if(color.every((v,j)=>words[i+j]===v))return true;
  }
  return false;
}

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
