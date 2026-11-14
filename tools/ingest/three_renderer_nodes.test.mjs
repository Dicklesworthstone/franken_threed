/** Native TSL through the actual facade, scene bridge and GPU command owner.
 * The recording device proves per-call capture and queue behavior; browser
 * source comparisons separately execute the generated shaders. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceWebGPUAttributeUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';
import {withThreeNativeComputePrerequisites} from './three_node_shadows.mjs';
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

for(const kind of ['fixed','authored','mixed','bundle','fixed instances','authored instances'])
test(`native ${kind} rendering keeps WebGL upload callbacks inert through admission and uploads`,async t=>{
  const f=fixture(t);let mesh=f.mesh;
  if(kind.startsWith('fixed'))mesh.material=new T.MeshBasicMaterial();
  if(kind.endsWith('instances')){
    const instanced=new T.InstancedMesh(mesh.geometry,mesh.material,2);
    instanced.setColorAt(0,new T.Color(.25,.5,.75));instanced.setColorAt(1,new T.Color(.5,.25,.75));
    f.scene.remove(mesh);f.scene.add(instanced);mesh=instanced;
  }
  mesh.frustumCulled=false;
  const geometries=[mesh.geometry];
  if(kind==='mixed'){
    const ordinary=new T.Mesh(mesh.geometry.clone(),new T.MeshBasicMaterial());ordinary.frustumCulled=false;f.scene.add(ordinary);
    geometries.push(ordinary.geometry);t.after(()=>ordinary.geometry.dispose());
  }
  if(kind==='bundle'){const group=new T.BundleGroup();group.static=false;group.add(mesh);f.scene.add(group);}
  const positions=geometries.map(geometry=>geometry.attributes.position);
  for(const position of positions)position.array.fill(.125);
  for(const attribute of [...positions,mesh.instanceMatrix,mesh.instanceColor].filter(Boolean))
    Object.defineProperty(attribute,'onUploadCallback',{get(){assert.fail('native uploads never read WebGL callbacks');}});
  for(const geometry of geometries)geometry.index.onUpload(()=>assert.fail('native index uploads never invoke WebGL callbacks'));
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  for(const position of positions){position.array[0]=.625;position.array[1]=99;position.addUpdateRange(0,1);position.needsUpdate=true;}
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.size>0),perRender=kind==='mixed'?2:1;
  assert.equal(draws.length,2*perRender);
  assert.deepEqual(draws.map(draw=>{
    const bytes=draw.contents.get(draw.streams.get(0));return [...new Float32Array(bytes.buffer,bytes.byteOffset,2)];
  }),[...Array.from({length:perRender},()=>[.125,.125]),...Array.from({length:perRender},()=>[.625,.125])]);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});
test('native inert upload callbacks share captured history across ordinary and authored draws',async t=>{
  const f=fixture(t),ordinary=new T.Mesh(f.mesh.geometry,new T.MeshBasicMaterial());
  ordinary.frustumCulled=f.mesh.frustumCulled=false;f.scene.add(ordinary);
  const position=f.mesh.geometry.attributes.position;position.array.fill(.125);
  Object.defineProperty(position,'onUploadCallback',{
    get(){assert.fail('ownership admission must not read a native-inert callback');},
  });
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  position.array.fill(99);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  assert.ok(draws.every(draw=>{
    const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4).every(value=>value===.125);
  }));
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native custom object dispatch captures queued prepasses and skips source items without replay',async t=>{
  const f=fixture(t),material=new T.MeshBasicMaterial(),prepass=new T.MeshBasicMaterial();
  f.mesh.material=material;f.mesh.frustumCulled=false;
  const skipped=new T.Mesh(new T.PlaneGeometry(),new T.ShaderMaterial());skipped.frustumCulled=false;
  skipped.renderOrder=1;f.scene.add(skipped);
  skipped.onBeforeRender=skipped.onAfterRender=()=>assert.fail('a skipped source item has no object lifecycle');
  const light=new T.AmbientLight(),unusedArea=new T.RectAreaLight();f.scene.add(light,unusedArea);
  const source=new T.WGSLNodeBuilder(f.mesh,f.renderer);source.scene=f.scene;source.camera=f.camera;
  source.lightsNode=new T.LightsNode().setLights([light,unusedArea]);
  // Even Basic's source lighting model constructs analytic nodes before
  // discarding their direct contribution, so its LTC prerequisite still holds.
  T.RectAreaLightNode.setLTC(null);
  assert.throws(()=>source.build(),error=>error instanceof TypeError&&/LTC_(HALF|FLOAT)_1/.test(error.message));
  const {RectAreaLightTexturesLib}=await import('./fixtures/three_rect_area_light.mjs');
  T.RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());t.after(()=>T.RectAreaLightNode.setLTC(null));
  const events=[],position=f.mesh.geometry.attributes.position;let phase=0;
  f.mesh.onBeforeRender=function(renderer,scene,camera,geometry,selected){
    events.push(['before',phase,selected]);position.array[0]=phase===0?.125:.25;position.needsUpdate=true;
  };
  f.mesh.onAfterRender=function(){events.push(['after',phase]);position.array[0]=99;};
  let sourceLights=null;
  const dispatch=function(object,scene,camera,geometry,selected,group,lights,clipping,passId){
    assert.equal(this,f.renderer);assert.equal(scene,f.scene);assert.equal(camera,f.camera);
    assert.equal(geometry,object.geometry);assert.equal(selected,object.material);assert.equal(group,null);
    assert.equal(clipping,null);assert.equal(passId,null);assert.ok(lights instanceof T.LightsNode);
    assert.equal(lights,f.renderer.lighting.getNode(scene));assert.deepEqual(lights.getLights(),[light,unusedArea]);
    sourceLights??=lights;assert.equal(lights,sourceLights);
    events.push(['dispatch',phase,object]);if(object===skipped)return;
    prepass.color.setRGB(phase===0?.125:.625,.375,.875);
    this.renderObject(object,scene,camera,geometry,prepass,group,lights,clipping,'prepass');
    selected.color.setRGB(phase===0?.25:.75,.375,.875);
    this.renderObject(object,scene,camera,geometry,selected,group,lights,clipping,passId);
  };
  f.renderer.setRenderObjectFunction(dispatch);await f.renderer.init();
  f.renderer.render(f.scene,f.camera);phase=1;f.renderer.render(f.scene,f.camera);
  prepass.color.setRGB(0,0,0);material.color.setRGB(0,0,0);position.array.fill(-99);
  const accepted=events.slice();assert.equal(f.acquired(),0);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  for(const [index,red]of[.125,.25,.625,.75].entries())assert.ok(containsColor(draws[index],[red,.375,.875]));
  const positions=draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];});
  assert.deepEqual(positions,[.125,.125,.25,.25]);assert.deepEqual(events,accepted);
  assert.deepEqual(events.filter(event=>event[0]==='dispatch').map(event=>event[2]),[f.mesh,skipped,f.mesh,skipped]);
  assert.deepEqual(events.filter(event=>event[0]==='before').map(event=>event[2]),[prepass,material,prepass,material]);
  assert.equal(f.device.textureWrites.length,0,'unlit material passes do not upload unused source LTC tables');
  assert.deepEqual(sourceLights.getLights(),[]);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native custom object dispatch fixes its function per render and retains history after reset',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const other=new T.Mesh(f.mesh.geometry,f.mesh.material);other.frustumCulled=false;other.renderOrder=1;f.scene.add(other);
  const events=[],position=f.mesh.geometry.attributes.position;
  const second=function(object){assert.equal(this._currentRenderObjectFunction,second);events.push(['second',object]);};
  const first=function(...args){
    assert.equal(this._currentRenderObjectFunction,first);events.push(['first',args[0]]);
    this.setRenderObjectFunction(second);position.array[0]=.25;
    if(args[0]===f.mesh)position.needsUpdate=true;this.renderObject(...args);
  };
  f.renderer.setRenderObjectFunction(first);await f.renderer.init();
  f.renderer.render(f.scene,f.camera);assert.equal(f.renderer.getRenderObjectFunction(),second);
  f.renderer.render(f.scene,f.camera);f.renderer.setRenderObjectFunction(null);
  position.array[0]=99;f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(events,[['first',f.mesh],['first',other],['second',f.mesh],['second',other]]);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  for(const draw of draws){const bytes=draw.contents.get(draw.streams.get(0));
    assert.equal(new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0],.25);}
  assert.equal(f.renderer.getRenderObjectFunction(),null);assert.equal(f.renderer._currentRenderObjectFunction,null);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native compileAsync ignores custom dispatch while preserving the source default lifecycle and uploads',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const original=f.mesh.geometry.attributes.position.array.slice(),events=[];let dispatched=0;
  f.mesh.onBeforeRender=()=>events.push('before');f.mesh.onAfterRender=()=>events.push('after');
  f.renderer.setRenderObjectFunction(()=>{dispatched++;});
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(dispatched,0);assert.deepEqual(events,['before','after']);
  f.mesh.geometry.attributes.position.array.fill(7);f.renderer.render(f.scene,f.camera);
  f.renderer.setRenderObjectFunction(function(...args){dispatched++;this.renderObject(...args);});
  f.renderer.render(f.scene,f.camera);f.mesh.geometry.attributes.position.array.fill(99);await drain(f.renderer);
  assert.equal(dispatched,2);assert.deepEqual(events,['before','after','before','after']);
  const draws=nativeDraws(f.device);assert.equal(draws.length,1);
  const bytes=draws[0].contents.get(draws[0].streams.get(0));
  assert.deepEqual(new Float32Array(bytes.buffer,bytes.byteOffset,original.length),original);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native custom object dispatch uses its live source lights for every material pass',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshLambertNodeMaterial();f.mesh.frustumCulled=false;
  const a=new T.AmbientLight(),b=new T.AmbientLight();a.color.setRGB(.125,.25,.375);b.color.setRGB(.625,.75,.875);
  f.scene.add(a);let supplied=null;
  f.renderer.setRenderObjectFunction(function(object,scene,camera,geometry,material,group,lights,clipping){
    supplied=lights;assert.equal(lights,this.lighting.getNode(scene));assert.deepEqual(lights.getLights(),[a]);
    this.renderObject(object,scene,camera,geometry,material,group,lights,clipping,'first');
    lights.setLights([b]);this.renderObject(object,scene,camera,geometry,material,group,lights,clipping,'second');
  });
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(containsColor(draws[0],[.125,.25,.375]));assert.ok(!containsColor(draws[0],[.625,.75,.875]));
  assert.ok(containsColor(draws[1],[.625,.75,.875]));assert.ok(!containsColor(draws[1],[.125,.25,.375]));
  assert.deepEqual(supplied.getLights(),[]);assert.equal(f.renderer.lighting._cache.length,0);
});

test('native custom dispatch exceptions retain source effects and release accepted captures',async t=>{
  const f=fixture(t),failure=new Error('dispatch stopped'),light=new T.AmbientLight(),events=[];
  f.mesh.onBeforeRender=()=>events.push('before');f.mesh.onAfterRender=()=>events.push('after');
  f.renderer.setRenderObjectFunction(function(...args){
    this.renderObject(...args);this.lighting.getNode(f.scene).setLights([light]);
    f.mesh.userData.completedPrepass=true;throw failure;
  });
  await f.renderer.init();assert.throws(()=>f.renderer.render(f.scene,f.camera),error=>error===failure);
  assert.deepEqual(events,['before','after']);assert.equal(f.mesh.userData.completedPrepass,true);
  assert.deepEqual(f.renderer.lighting.getNode(f.scene).getLights(),[light]);assert.equal(f.renderer.lighting._cache.length,0);
  assert.equal(f.renderer._handleObjectFunction,null);assert.equal(f.renderer._nodeCaptureBytes,0);
  assert.equal(f.device.submissions.length,0);assert.equal(f.acquired(),0);
  f.renderer.setRenderObjectFunction(null);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(events,['before','after','before','after']);assert.equal(nativeDraws(f.device).length,1);
});

test('native custom dispatch records repeated bundle passes without replaying the dispatcher on warm renders',async t=>{
  const f=fixture(t),group=new T.BundleGroup();f.scene.add(group);group.add(f.mesh);
  f.mesh.material=new T.MeshBasicMaterial({color:new T.Color(.125,.375,.625)});
  let dispatches=0,before=0;f.mesh.onBeforeRender=()=>{before++;};
  f.renderer.setRenderObjectFunction(function(...args){
    dispatches++;this.renderObject(...args);args[8]='repeat';this.renderObject(...args);
  });
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  f.mesh.material.color.setRGB(.875,.625,.375);
  f.renderer.setRenderObjectFunction(()=>{dispatches++;});
  f.renderer.render(f.scene,f.camera);group.needsUpdate=true;f.renderer.render(f.scene,f.camera);
  assert.equal(dispatches,2);assert.equal(before,2);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  assert.ok(draws.every(draw=>containsColor(draw,[.125,.375,.625])));
  assert.equal(dispatches,2);assert.equal(before,2);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native custom dispatch rejects replacement source contexts after one unreplayed before callback',async t=>{
  for(const index of [6,7]){
    const f=fixture(t);let before=0,after=0;
    f.mesh.onBeforeRender=()=>{before++;};f.mesh.onAfterRender=()=>{after++;};
    f.renderer.setRenderObjectFunction(function(...args){
      args[index]=index===6?new T.LightsNode():{};this.renderObject(...args);
    });
    await f.renderer.init();assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_HOOK'});
    assert.equal(before,1);assert.equal(after,0);assert.equal(f.device.submissions.length,0);
    assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.renderer.lighting._cache.length,0);
    f.renderer.setRenderObjectFunction(null);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
    assert.equal(before,2);assert.equal(after,1);assert.equal(nativeDraws(f.device).length,1);
  }
});

test('authored sprites retain queued billboard uniforms and render once per source object instead of per instance',async t=>{
  const f=fixture(t),N=T.TSL,material=new T.SpriteNodeMaterial();
  const rotation=N.uniform(.375).setName('queuedSpriteRotation'),scale=N.uniform(new T.Vector2(.625,.8125)).setName('queuedSpriteScale');
  material.colorNode=f.tint;material.rotationNode=rotation;material.scaleNode=scale;material.opacityNode=N.shapeCircle();
  const sprite=new T.Sprite(material);sprite.center.set(.125,.875);sprite.scale.x=-2;sprite.count=5;
  f.scene.remove(f.mesh);f.scene.add(sprite);let updates=0;
  material.positionNode=N.uniform(new T.Vector3()).onObjectUpdate(()=>{updates++;return new T.Vector3();});
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  rotation.value=.6875;scale.value.set(.9375,.4375);sprite.center.set(.3125,.5625);sprite.count=11;
  f.renderer.render(f.scene,f.camera);
  rotation.value=99;scale.value.set(99,99);sprite.center.set(99,99);sprite.count=NaN;
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);assert.equal(updates,2);
  assert.deepEqual(draws.map(draw=>draw.args.slice(0,2)),[[6,5],[6,11]]);
  for(const [draw,values] of [[draws[0],[.375,.625,.8125,.125,.875]],[draws[1],[.6875,.9375,.4375,.3125,.5625]]]){
    const words=uniformBlocks(draw).flat();for(const value of values)assert.ok(words.includes(value));
    assert.equal(draw.pipeline.primitive.frontFace,'ccw','source sprites do not reverse face winding for negative object scales');
  }
  assert.equal(draws[0].pipeline,draws[1].pipeline);assert.equal(f.renderer._nodeCaptureBytes,0);
});

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

test('native object callbacks retain queued colors, source side multiplicity and immutable geometry submissions',async t=>{
  const f=fixture(t),events=[],other=new T.Mesh(f.mesh.geometry,f.material);
  f.scene.add(other);f.material.transparent=true;f.material.side=T.DoubleSide;
  f.mesh.frustumCulled=other.frustumCulled=false;
  const position=f.mesh.geometry.attributes.position;let phase=0;
  for(const [object,index] of [[f.mesh,0],[other,1]]){
    object.onBeforeRender=function(renderer,scene,camera,geometry,material,group){
      assert.equal(this,object);assert.equal(renderer,f.renderer);assert.equal(scene,f.scene);assert.equal(camera,f.camera);
      assert.equal(geometry,f.mesh.geometry);assert.equal(material,f.material);assert.equal(group,null);
      events.push(['before',phase,index,material.side]);
      const red=phase===0?(index===0?.125:.625):(index===0?.25:.75);
      f.tint.value.setRGB(red,.375,.875);
      position.array[0]=red;position.needsUpdate=true;
    };
    object.onAfterRender=function(renderer,scene,camera,geometry,material){
      events.push(['after',phase,index,material.side]);f.tint.value.setRGB(0,0,0);position.array[0]=99;
    };
  }
  f.material.onBeforeRender=()=>assert.fail('material callback is inert on the pinned WebGPU route');
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  phase=1;f.renderer.render(f.scene,f.camera);
  assert.deepEqual(events,[
    ['before',0,0,T.DoubleSide],['after',0,0,T.DoubleSide],['before',0,1,T.DoubleSide],['after',0,1,T.DoubleSide],
    ['before',1,0,T.DoubleSide],['after',1,0,T.DoubleSide],['before',1,1,T.DoubleSide],['after',1,1,T.DoubleSide]]);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,8);
  for(const [index,red] of [.125,.125,.625,.625,.25,.25,.75,.75].entries())
    assert.ok(containsColor(draws[index],[red,.375,.875]),`captured color ${index}`);
  const firstPosition=draw=>{
    const buffer=draw.streams.get(0),bytes=draw.contents.get(buffer);
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];
  };
  assert.deepEqual(draws.map(firstPosition),[.125,.125,.125,.125,.25,.25,.25,.25],
    'the source uploads a shared geometry once per logical render, before its later callbacks');
  assert.equal(position.array[0],99);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native object callbacks preserve prior upload history when hooks are added after preparation',async t=>{
  const f=fixture(t);await f.renderer.compileAsync(f.scene,f.camera);
  const position=f.mesh.geometry.attributes.position,original=position.array.slice();let calls=0;
  position.array.fill(7);
  f.mesh.onBeforeRender=()=>{calls++;f.tint.value.setRGB(.25,.5,.75);};
  f.mesh.onAfterRender=()=>{position.array.fill(9);};
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draw=nativeDraws(f.device)[0],buffer=draw.streams.get(0),bytes=draw.contents.get(buffer);
  assert.deepEqual(new Float32Array(bytes.buffer,bytes.byteOffset,original.length),original);
  assert.equal(calls,1);assert.ok(containsColor(draw,[.25,.5,.75]));
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('object callbacks replace queued draw geometry while retaining projected hook arguments and shared upload history',async t=>{
  const f=fixture(t),projected=f.mesh.geometry,a=new T.PlaneGeometry(),b=new T.PlaneGeometry().toNonIndexed(),events=[];
  f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  a.setDrawRange(3,3);a.attributes.position.array.fill(.125);b.attributes.position.array.fill(.75);
  t.after(()=>{a.dispose();b.dispose();});let phase=0;
  f.mesh.onBeforeRender=function(renderer,scene,camera,geometry){
    events.push(['before',geometry]);this.geometry=phase===0?a:b;
  };
  f.mesh.onAfterRender=function(renderer,scene,camera,geometry){
    events.push(['after',geometry]);this.geometry.attributes.position.array.fill(99);
  };
  await f.renderer.init();f.renderer.render(f.scene,f.camera);phase=1;f.renderer.render(f.scene,f.camera);
  assert.deepEqual(events,[['before',projected],['after',projected],['before',a],['after',a]]);
  f.mesh.onBeforeRender=T.Object3D.prototype.onBeforeRender;f.mesh.onAfterRender=T.Object3D.prototype.onAfterRender;
  const otherScene=new T.Scene(),other=new T.Mesh(b,new T.MeshBasicMaterial());other.frustumCulled=false;otherScene.add(other);
  f.renderer.render(otherScene,f.camera);assert.equal(f.acquired(),0);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(draw=>[draw.indexed,draw.args[0],draw.args[2]]),[[true,3,3],[false,6,0],[false,6,0]]);
  assert.deepEqual(draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];}),[.125,.75,.75]);
  assert.equal(events.length,4);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom object dispatch selects replacement geometry independently for each forwarded pass',async t=>{
  const f=fixture(t),projected=f.mesh.geometry,replacement=new T.PlaneGeometry().toNonIndexed(),seen=[];
  f.mesh.frustumCulled=false;replacement.attributes.position.array.fill(.75);t.after(()=>replacement.dispose());
  f.mesh.onBeforeRender=function(renderer,scene,camera,geometry){seen.push([geometry,this.geometry]);};
  f.renderer.setRenderObjectFunction(function(...args){
    this.renderObject(...args);args[0].geometry=replacement;this.renderObject(...args);
  });
  await f.renderer.init();f.renderer.render(f.scene,f.camera);replacement.attributes.position.array.fill(99);await drain(f.renderer);
  assert.deepEqual(seen,[[projected,projected],[projected,replacement]]);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);assert.deepEqual(draws.map(draw=>draw.indexed),[true,false]);
  const bytes=draws[1].contents.get(draws[1].streams.get(0));
  assert.equal(new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0],.75);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('a completed DoubleSide pass can replace geometry for the following source pass',async t=>{
  const f=fixture(t),projected=f.mesh.geometry,replacement=new T.PlaneGeometry().toNonIndexed(),events=[];
  f.mesh.frustumCulled=false;f.material.transparent=true;f.material.side=T.DoubleSide;
  replacement.attributes.position.array.fill(.75);t.after(()=>replacement.dispose());
  f.mesh.onBeforeRender=(renderer,scene,camera,geometry)=>{events.push(['before',geometry]);};
  f.mesh.onAfterRender=(renderer,scene,camera,geometry)=>{events.push(['after',geometry]);};
  f.tint.updateAfterType='object';f.tint.updateAfter=()=>{
    events.push(['node-after',f.material.side]);if(f.material.side===T.BackSide)f.mesh.geometry=replacement;
  };
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(events,[['before',projected],['node-after',T.BackSide],['node-after',T.FrontSide],['after',projected]]);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(draws.every(draw=>draw.pipeline.primitive.cullMode==='back'));
  assert.deepEqual(draws.map(draw=>[draw.indexed,draw.pipeline.primitive.frontFace]),[[true,'cw'],[false,'ccw']]);
  const bytes=draws[1].contents.get(draws[1].streams.get(0));
  assert.equal(new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0],.75);
  assert.equal(f.material.side,T.DoubleSide);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('compileAsync selects replacement geometry after object after-hooks without uploading the intermediate selection',async t=>{
  const f=fixture(t),projected=f.mesh.geometry,before=new T.PlaneGeometry(),after=new T.PlaneGeometry().toNonIndexed(),events=[];
  f.mesh.frustumCulled=false;before.attributes.position.array.fill(.25);after.attributes.position.array.fill(.625);
  before.attributes.position.addUpdateRange(0,3);t.after(()=>{before.dispose();after.dispose();});
  f.mesh.onBeforeRender=function(renderer,scene,camera,geometry){events.push(['before',geometry]);this.geometry=before;};
  f.mesh.onAfterRender=function(renderer,scene,camera,geometry){events.push(['after',geometry]);this.geometry=after;};
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.deepEqual(events,[['before',projected],['after',projected]]);assert.equal(nativeDraws(f.device).length,0);
  assert.deepEqual(before.attributes.position.updateRanges,[{start:0,count:3}]);
  after.attributes.position.array.fill(99);f.mesh.onBeforeRender=T.Object3D.prototype.onBeforeRender;
  f.mesh.onAfterRender=T.Object3D.prototype.onAfterRender;f.renderer.render(f.scene,f.camera);
  f.mesh.geometry=before;before.attributes.position.array.fill(.375);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];}),[.625,.375]);
  assert.equal(events.length,2);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('callback replacement reuses a prior fixed geometry history across scenes',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const position=f.mesh.geometry.attributes.position;position.array.fill(.125);
  await f.renderer.compileAsync(f.scene,f.camera);
  const otherScene=new T.Scene(),fresh=new T.PlaneGeometry(),other=new T.Mesh(fresh,new T.MeshBasicMaterial());
  other.frustumCulled=false;otherScene.add(other);t.after(()=>fresh.dispose());let before=0,after=0;
  position.array.fill(99);
  other.onBeforeRender=()=>{before++;other.geometry=f.mesh.geometry;};other.onAfterRender=()=>{after++;};
  f.renderer.render(otherScene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,1);
  const bytes=draws[0].contents.get(draws[0].streams.get(0));
  assert.deepEqual([...new Float32Array(bytes.buffer,bytes.byteOffset,position.array.length)],Array(position.array.length).fill(.125));
  assert.equal(before,1);assert.equal(after,1);assert.equal(other.geometry,f.mesh.geometry);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

for(const firstUse of ['render','compileAsync'])test(`native selected inputs defer inactive color uploads after ${firstUse}`,async t=>{
  const f=fixture(t),position=f.mesh.geometry.attributes.position,original=position.array.slice();
  const color=new T.Uint8BufferAttribute(new Uint8Array(position.count*3).fill(64),3,true);
  f.mesh.geometry.setAttribute('color',color);f.mesh.frustumCulled=false;
  color.addUpdateRange(0,3);color.needsUpdate=true;
  await f.renderer.init();
  if(firstUse==='compileAsync')await f.renderer.compileAsync(f.scene,f.camera);
  else f.renderer.render(f.scene,f.camera);
  assert.deepEqual(color.updateRanges,[{start:0,count:3}]);
  color.array.fill(128);position.array.fill(99);
  f.material.vertexColors=true;f.material.needsUpdate=true;
  f.renderer.render(f.scene,f.camera);
  color.array.fill(0);position.array.fill(-99);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,firstUse==='render'?2:1);
  const draw=draws.at(-1),stream=name=>{
    const match=new RegExp('@location\\(\\s*(\\d+)\\s*\\)\\s+'+name+'\\s*:').exec(draw.pipeline.vertex.module.code);
    assert.ok(match,`source shader input ${name}`);
    const slot=draw.pipeline.vertex.buffers.findIndex(layout=>layout.attributes.some(a=>a.shaderLocation===Number(match[1])));
    return draw.contents.get(draw.streams.get(slot));
  };
  const bytes=stream('color');
  assert.deepEqual([...new Float32Array(bytes.buffer,bytes.byteOffset,color.count*3)],Array(color.count*3).fill(Math.fround(128/255)));
  const positions=stream('position');
  assert.deepEqual(new Float32Array(positions.buffer,positions.byteOffset,original.length),original);
  assert.deepEqual(color.updateRanges,[{start:0,count:3}],'first selected upload leaves its initial ranges pending');
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native vertex-index shaders with object hooks keep nonindexed draw counts without a position upload',async t=>{
  const f=fixture(t);f.mesh.geometry=f.mesh.geometry.toNonIndexed();f.mesh.frustumCulled=false;
  f.material.vertexNode=T.TSL.vec4(T.TSL.vertexIndex.toFloat().mul(.25),0,0,1);
  let callbacks=0;f.mesh.onBeforeRender=()=>{callbacks++;};
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draw=nativeDraws(f.device)[0];assert.equal(draw.indexed,false);assert.equal(draw.args[0],6);
  assert.equal(draw.streams.size,0);assert.equal(draw.pipeline.vertex.buffers.length,0);
  assert.equal(callbacks,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('ordinary object hooks capture queued draws and retain upload history after callbacks are removed',async t=>{
  const f=fixture(t),material=new T.MeshBasicMaterial(),companionMaterial=new T.MeshBasicMaterial({color:0xffffff});
  f.mesh.material=material;f.mesh.frustumCulled=false;
  const companion=new T.Mesh(new T.PlaneGeometry(),companionMaterial);companion.renderOrder=1;
  f.scene.add(companion);
  const position=f.mesh.geometry.attributes.position,events=[];let phase=0;
  f.mesh.onBeforeRender=function(renderer,scene,camera,geometry,sourceMaterial,group){
    assert.equal(this,f.mesh);assert.equal(renderer,f.renderer);assert.equal(scene,f.scene);assert.equal(camera,f.camera);
    assert.equal(geometry,f.mesh.geometry);assert.equal(sourceMaterial,material);assert.equal(group,null);
    const red=phase===0?.25:.5;material.color.setRGB(red,.375,.875);
    position.array[0]=red;position.needsUpdate=true;events.push(['before',phase]);
  };
  f.mesh.onAfterRender=function(){
    events.push(['after',phase]);position.array[0]=99;material.color.setRGB(0,0,0);
    companionMaterial.color.setRGB(phase===0?.125:.625,.75,.875);
  };
  material.onBeforeRender=()=>assert.fail('native source does not call material.onBeforeRender');
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  phase=1;f.renderer.render(f.scene,f.camera);
  f.mesh.onBeforeRender=T.Object3D.prototype.onBeforeRender;f.mesh.onAfterRender=T.Object3D.prototype.onAfterRender;
  material.color.setRGB(.75,.375,.875);companionMaterial.color.setRGB(.875,.75,.875);
  f.renderer.render(f.scene,f.camera);
  material.color.setRGB(0,0,0);companionMaterial.color.setRGB(0,0,0);position.array[0]=-99;
  assert.deepEqual(events,[['before',0],['after',0],['before',1],['after',1]]);
  assert.equal(f.acquired(),0,'all callback effects and draw snapshots precede asynchronous preparation');
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,6);
  for(const [index,color] of [[.25,.375,.875],[.125,.75,.875],[.5,.375,.875],[.625,.75,.875],
    [.75,.375,.875],[.875,.75,.875]].entries())assert.ok(containsColor(draws[index],color),`ordinary queued color ${index}`);
  const firstPosition=draw=>{
    const bytes=draw.contents.get(draw.streams.get(0));return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];
  };
  assert.deepEqual([draws[0],draws[2],draws[4]].map(firstPosition),[.25,.5,.5],
    'removing callbacks preserves the last uploaded bytes without a new source update request');
  assert.deepEqual(events,[['before',0],['after',0],['before',1],['after',1]],'preparation never replays source effects');
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.renderer.info.f3d.presentedRenders,3);
});

for(const firstUse of ['render','compileAsync'])test(`late ordinary hooks preserve fixed upload generations after ${firstUse}`,async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const position=f.mesh.geometry.attributes.position,events=[];position.array.fill(.125);
  await f.renderer.init();
  if(firstUse==='compileAsync')await f.renderer.compileAsync(f.scene,f.camera);
  else f.renderer.render(f.scene,f.camera);
  position.array.fill(99);let phase=0;
  f.mesh.onBeforeRender=()=>{
    events.push(['before',phase]);
    if(phase===1){position.array[0]=.25;position.addUpdateRange(0,1);position.needsUpdate=true;}
  };
  f.mesh.onAfterRender=()=>{events.push(['after',phase]);position.array.fill(-99);};
  f.renderer.render(f.scene,f.camera);phase=1;f.renderer.render(f.scene,f.camera);
  position.array.fill(99);
  assert.deepEqual(events,[['before',0],['after',0],['before',1],['after',1]]);
  assert.equal(f.acquired(),0,'accepted callback effects precede pending pipeline preparation');
  await drain(f.renderer);
  f.mesh.onBeforeRender=T.Object3D.prototype.onBeforeRender;f.mesh.onAfterRender=T.Object3D.prototype.onAfterRender;
  f.renderer.render(f.scene,f.camera);position.array.fill(-99);await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.size>0);
  assert.equal(draws.length,firstUse==='render'?4:3);
  const uploaded=draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return [...new Float32Array(bytes.buffer,bytes.byteOffset,position.array.length)];});
  assert.deepEqual(uploaded.map(words=>words[0]),firstUse==='render'?[.125,.125,.25,.25]:[.125,.25,.25]);
  for(const words of uploaded)assert.deepEqual(words.slice(1),Array(position.array.length-1).fill(.125));
  assert.deepEqual(position.updateRanges,[]);assert.equal(events.length,4);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('late hooks preserve shared normalized interleaved history and source disposal across scenes',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial({vertexColors:true});f.mesh.frustumCulled=false;
  const geometry=f.mesh.geometry,position=geometry.attributes.position;
  const colorOwner=new T.InterleavedBuffer(new Uint8Array(position.count*4).fill(64),4);
  geometry.setAttribute('color',new T.InterleavedBufferAttribute(colorOwner,3,0,true));position.array.fill(.125);
  await f.renderer.compileAsync(f.scene,f.camera);
  const otherGeometry=geometry.clone(),otherScene=new T.Scene(),other=new T.Mesh(otherGeometry,f.mesh.material);
  otherGeometry.setAttribute('position',position);otherGeometry.setAttribute('color',geometry.attributes.color);
  otherScene.add(other);other.frustumCulled=false;t.after(()=>otherGeometry.dispose());
  position.array.fill(99);colorOwner.array.fill(255);
  other.onBeforeRender=()=>{colorOwner.array[1]=128;colorOwner.addUpdateRange(1,1);colorOwner.needsUpdate=true;};
  f.renderer.render(otherScene,f.camera);
  other.onBeforeRender=T.Object3D.prototype.onBeforeRender;geometry.dispose();
  position.array.fill(.5);colorOwner.array.fill(192);f.renderer.render(otherScene,f.camera);
  position.array.fill(-99);colorOwner.array.fill(0);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  const words=(draw,name)=>{
    const match=new RegExp('@location\\(\\s*(\\d+)\\s*\\)\\s+'+name+'\\s*:').exec(draw.pipeline.vertex.module.code);
    assert.ok(match,`source input ${name}`);
    const slot=draw.pipeline.vertex.buffers.findIndex(layout=>layout.attributes.some(a=>a.shaderLocation===Number(match[1])));
    const bytes=draw.contents.get(draw.streams.get(slot));return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
  };
  assert.equal(words(draws[0],'position')[0],.125);assert.equal(words(draws[1],'position')[0],.5);
  assert.deepEqual([...words(draws[0],'color').slice(0,4)],[64,128,64,64].map(value=>Math.fround(value/255)));
  assert.deepEqual([...words(draws[1],'color').slice(0,4)],Array(4).fill(Math.fround(192/255)));
  assert.deepEqual(colorOwner.updateRanges,[]);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('late custom dispatch preserves an earlier queued fixed geometry and group selection',async t=>{
  const f=fixture(t);f.mesh.material=[new T.MeshBasicMaterial()];f.mesh.frustumCulled=false;
  const original=f.mesh.geometry,replacement=new T.PlaneGeometry().toNonIndexed();
  original.clearGroups();original.addGroup(0,3,0);original.attributes.position.array.fill(.125);
  replacement.clearGroups();replacement.addGroup(0,6,0);replacement.attributes.position.array.fill(.5);
  t.after(()=>replacement.dispose());await f.renderer.init();f.renderer.render(f.scene,f.camera);
  f.mesh.geometry=replacement;let dispatches=0;
  f.renderer.setRenderObjectFunction(function(...args){dispatches++;this.renderObject(...args);});
  f.renderer.render(f.scene,f.camera);
  original.attributes.position.array.fill(99);replacement.attributes.position.array.fill(-99);
  original.clearGroups();replacement.clearGroups();await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.size>0);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>[draw.indexed,draw.args[0]]),[[true,3],[false,6]]);
  assert.deepEqual(draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];}),[.125,.5]);
  assert.equal(dispatches,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('queued fixed geometry remains rigid after a later callback selects morph geometry',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const original=f.mesh.geometry,replacement=new T.PlaneGeometry().toNonIndexed();
  original.attributes.position.array.fill(.125);replacement.attributes.position.array.fill(.5);
  replacement.morphAttributes.position=[replacement.attributes.position.clone()];replacement.morphTargetsRelative=true;
  t.after(()=>replacement.dispose());await f.renderer.init();f.renderer.render(f.scene,f.camera);
  let callbacks=0;
  f.mesh.onBeforeRender=()=>{callbacks++;f.mesh.geometry=replacement;f.mesh.updateMorphTargets();};
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.size>0);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>[draw.indexed,draw.args[0]]),[[true,6],[false,6]]);
  assert.deepEqual(draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];}),[.125,.5]);
  assert.equal(callbacks,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('queued fixed geometry keeps projected depth when later calls change its source bounds',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const otherGeometry=new T.PlaneGeometry(),other=new T.Mesh(otherGeometry,f.mesh.material);
  other.frustumCulled=false;f.scene.add(other);t.after(()=>otherGeometry.dispose());
  f.mesh.geometry.translate(0,0,1);otherGeometry.translate(0,0,-1);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  const position=f.mesh.geometry.attributes.position;
  for(let i=2;i<position.array.length;i+=3)position.array[i]=-3;
  f.mesh.geometry.computeBoundingSphere();f.mesh.onBeforeRender=()=>{};
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.size>0);assert.equal(draws.length,4);
  assert.deepEqual(draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[2];}),[1,-1,-1,1]);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('fixed projected shadows retain off-camera casters and their guarded upload owner',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshPhongMaterial();f.mesh.receiveShadow=true;
  f.renderer.shadowMap.enabled=true;f.renderer.shadowMap.type=T.PCFShadowMap;
  const caster=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial());caster.position.x=3;caster.castShadow=true;
  const light=new T.DirectionalLight();light.position.z=5;light.castShadow=true;light.shadow.mapSize.set(16,16);
  Object.assign(light.shadow.camera,{left:-6,right:6,top:6,bottom:-6,near:.1,far:20});light.shadow.camera.updateProjectionMatrix();
  f.scene.add(caster,light,light.target);t.after(()=>caster.geometry.dispose());
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const projected=new T.Matrix4().multiplyMatrices(f.camera.projectionMatrix,f.camera.matrixWorldInverse);
  assert.equal(caster.intersectsFrustum(new T.Frustum().setFromProjectionMatrix(projected,f.camera.coordinateSystem)),false);
  assert.equal(caster.intersectsFrustum(light.shadow.getFrustum()),true);
  const depth=f.device.passes.filter(pass=>!pass.desc.colorAttachments?.length);
  assert.equal(depth.length,1);assert.equal(depth[0].draws.length,1);
  assert.equal(f.device.passes.filter(pass=>pass.desc.colorAttachments?.length).flatMap(pass=>pass.draws).filter(draw=>draw.streams.size).length,1);
  let callbacks=0;caster.onBeforeRender=()=>{callbacks++;};
  assert.throws(()=>f.renderer.render(f.scene,f.camera),error=>error.code==='THREE_NODE_PROGRAM_HOOK'&&/fixed owner/.test(error.message));
  assert.equal(callbacks,0);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('culled fixed geometry remains fresh when a later hook first selects it',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.position.x=10000;
  const position=f.mesh.geometry.attributes.position;await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(f.device.writes.length,0,'a culled fixed geometry never reaches the upload owner');
  f.mesh.position.x=0;f.mesh.frustumCulled=false;position.array.fill(.375);let callbacks=0;
  f.mesh.onBeforeRender=()=>{callbacks++;};f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draw=nativeDraws(f.device)[0],bytes=draw.contents.get(draw.streams.get(0));
  assert.equal(new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0],.375);
  assert.equal(callbacks,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('fixed dynamic attributes keep source render frequency when late hooks are installed',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const position=f.mesh.geometry.attributes.position;position.setUsage(T.DynamicDrawUsage);position.array.fill(.125);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  position.array.fill(.25);f.renderer.render(f.scene,f.camera);
  position.array.fill(.5);f.mesh.onBeforeRender=()=>{};f.renderer.render(f.scene,f.camera);
  position.array.fill(99);await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.size>0);
  assert.deepEqual(draws.map(draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];}),[.125,.25,.5]);
  assert.equal(position.version,0);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('late hooks follow ordinary instance geometry history across shared source owners',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const instanceGeometry=f.mesh.geometry.clone(),linkedGeometry=f.mesh.geometry.clone();
  instanceGeometry.setAttribute('position',f.mesh.geometry.attributes.position);
  linkedGeometry.setAttribute('normal',f.mesh.geometry.attributes.normal);
  const instances=new T.InstancedMesh(instanceGeometry,new T.MeshBasicMaterial(),1),linked=new T.Mesh(linkedGeometry,f.mesh.material);
  instances.frustumCulled=false;linked.frustumCulled=false;f.scene.add(linked,instances);
  linkedGeometry.attributes.position.array.fill(.25);
  t.after(()=>{instanceGeometry.dispose();linkedGeometry.dispose();});await f.renderer.compileAsync(f.scene,f.camera);
  linkedGeometry.attributes.position.array.fill(99);
  const otherScene=new T.Scene(),otherGeometry=new T.PlaneGeometry(),other=new T.Mesh(otherGeometry,f.mesh.material);
  otherGeometry.setAttribute('position',linkedGeometry.attributes.position);other.frustumCulled=false;otherScene.add(other);
  t.after(()=>otherGeometry.dispose());let callbacks=0;other.onBeforeRender=()=>{callbacks++;};
  f.renderer.render(otherScene,f.camera);assert.equal(callbacks,1);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,1);
  const bytes=draws[0].contents.get(draws[0].streams.get(0));
  assert.ok(new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4).every(value=>value===.25));
  assert.equal(callbacks,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('ordinary hook attribute history follows shared geometry into another scene after hook removal',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const position=f.mesh.geometry.attributes.position;
  f.mesh.onBeforeRender=()=>{position.array[0]=.25;position.needsUpdate=true;};
  await f.renderer.init();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  f.mesh.onBeforeRender=T.Object3D.prototype.onBeforeRender;position.array[0]=99;
  const otherScene=new T.Scene(),other=new T.Mesh(f.mesh.geometry.clone(),new T.MeshBasicMaterial());
  other.geometry.setAttribute('position',position);other.frustumCulled=false;otherScene.add(other);
  f.renderer.render(otherScene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  for(const draw of draws){
    const bytes=draw.contents.get(draw.streams.get(0));
    assert.equal(new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0],.25);
  }
  assert.equal(position.array[0],99);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native callback reentrant GPU rendering fails before a nested submission and releases the call scope',async t=>{
  const f=fixture(t);await f.renderer.init();
  const inner=new T.Scene(),innerMaterial=new T.MeshBasicNodeMaterial();innerMaterial.colorNode=T.TSL.vec3(.25);
  inner.add(new T.Mesh(new T.PlaneGeometry(),innerMaterial));
  let before=0,after=0;
  f.mesh.onBeforeRender=()=>{before++;f.renderer.render(inner,f.camera);};
  f.mesh.onAfterRender=()=>{after++;};
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_REENTRY'});
  assert.equal(before,1);assert.equal(after,0);assert.equal(f.device.submissions.length,0);assert.equal(f.renderer._nodeCaptureBytes,0);
  f.mesh.onBeforeRender=T.Object3D.prototype.onBeforeRender;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length,1);assert.equal(after,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('transparent native DoubleSide renders submit ordered side pipelines with independent queued uniforms',async t=>{
  const f=fixture(t);f.material.transparent=true;f.material.side=T.DoubleSide;f.mesh.scale.x=-1;
  f.material.colorNode=T.TSL.normalView.mul(.25).add(.35).mul(f.tint);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  f.tint.value.setRGB(.875,.625,.375);f.renderer.render(f.scene,f.camera);
  f.tint.value.setRGB(.25,.5,.75);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  assert.ok(draws.every(draw=>draw.pipeline.primitive.cullMode==='back'));
  assert.deepEqual(draws.map(draw=>draw.pipeline.primitive.frontFace),['ccw','cw','ccw','cw'],
    'source BackSide reverses winding; negative determinant flips both side pipelines');
  assert.ok(draws[0].pipeline.fragment.module.code.includes('normalView = ( normalViewGeometry * vec3<f32>( -1.0 ) );'),
    'the back-side shader reverses normals even though the pipeline regards its selected triangles as front-facing');
  assert.ok(draws[1].pipeline.fragment.module.code.includes('normalView = normalViewGeometry;'));
  assert.ok(draws.every(draw=>!draw.pipeline.fragment.module.code.includes('@builtin( front_facing )')),
    'separate side shaders must not retain the single-pass front-facing normal expression');
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
  assert.ok(nativeDraws(f.device).at(-1).pipeline.fragment.module.code.includes('@builtin( front_facing )'));
});

test('precompiled transparent DoubleSide normals survive queued side pipelines and material invalidation',async t=>{
  const f=fixture(t);f.material.transparent=true;f.material.side=T.DoubleSide;f.mesh.scale.x=-1;
  f.material.colorNode=T.TSL.normalView.mul(.25).add(.35).mul(f.tint);
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(nativeDraws(f.device).length,0);
  f.renderer.render(f.scene,f.camera);
  f.tint.value.setRGB(.875,.625,.375);f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,4);
  assert.ok(draws.every(draw=>draw.pipeline.fragment.module.code.includes('@builtin( front_facing )')),
    'source compileAsync builds after restoring DoubleSide and retains that shader in each pass cache');
  assert.ok(draws.every(draw=>draw.pipeline.primitive.cullMode==='back'));
  assert.deepEqual(draws.map(draw=>draw.pipeline.primitive.frontFace),['ccw','cw','ccw','cw']);
  assert.ok(draws.slice(0,2).every(draw=>containsColor(draw,[.125,.375,.625])));
  assert.ok(draws.slice(2).every(draw=>containsColor(draw,[.875,.625,.375])));
  assert.equal(draws[0].pipeline,draws[2].pipeline);assert.equal(draws[1].pipeline,draws[3].pipeline);
  f.material.needsUpdate=true;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const rebuilt=nativeDraws(f.device).slice(-2);
  assert.ok(rebuilt.every(draw=>!draw.pipeline.fragment.module.code.includes('@builtin( front_facing )')));
  await f.renderer.compileAsync(f.scene,f.camera);
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(nativeDraws(f.device).slice(-2).map(draw=>draw.pipeline),rebuilt.map(draw=>draw.pipeline),
    'later compilation cannot replace already constructed side-specific normals without invalidation');
  assert.equal(f.material.side,T.DoubleSide);assert.equal(f.renderer._nodeCaptureBytes,0);
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

test('native and ordinary meshes share captured depth sorting even when material creation order disagrees',async t=>{
  const f=fixture(t);const fixed=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial({color:new T.Color(.875,.625,.125)}));
  // The authored material was created first, but the ordinary object is nearer.
  fixed.position.z=.5;f.scene.add(fixed);
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(containsColor(draws[0],[.875,.625,.125]));
  assert.ok(containsColor(draws[1],[.125,.375,.625]));
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

for (const initialized of [false,true])
test(`effect compute prerequisites retain FIFO storage versions with initialized=${initialized}`,async t=>{
  const f=fixture(t);recordCompute(f.device);
  const attribute=new T.StorageBufferAttribute(4,4),storage=T.TSL.storage(attribute,'vec4',4),kernel=storageKernel(storage);
  const values=[.125,.625];let updates=0;
  f.tint.updateBeforeType='object';
  f.tint.updateBefore=()=>withThreeNativeComputePrerequisites(f.renderer,()=>{
    attribute.array.fill(values[updates++]);attribute.needsUpdate=true;
    f.renderer.compute(kernel);
  });
  f.material.fragmentNode=storage.element(0).mul(f.tint);
  if(initialized)await f.renderer.init();
  const first=f.renderer.renderAsync(f.scene,f.camera),second=f.renderer.renderAsync(f.scene,f.camera);
  await Promise.all([first,second,drain(f.renderer)]);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);assert.equal(updates,2);
  assert.equal(f.device.computePasses.length,2);
  assert.deepEqual(draws.map(draw=>storageWords(draw)[0][0]),values);
  const order=f.device.submissions.flat(2).flatMap(pass=>
    pass.dispatches?['compute']:pass.draws.some(draw=>draw.pipeline.vertex.entryPoint==='main')?['render']:[]);
  assert.deepEqual(order,['compute','render','compute','render']);
  assert.equal(f.renderer._nodeCaptureBytes,0);await f.renderer.waitForGPU();
});

for(const kind of ['native','fixed','cold bundle','warm bundle'])
test(`effect compute refuses a possible earlier ${kind} draw before writing storage`,async t=>{
  const f=fixture(t);recordCompute(f.device);
  const attribute=new T.StorageBufferAttribute(4,4),storage=T.TSL.storage(attribute,'vec4',4),kernel=storageKernel(storage);
  const material=kind==='fixed'?new T.MeshBasicMaterial():new T.MeshBasicNodeMaterial();
  const earlier=new T.Mesh(new T.PlaneGeometry(),material);earlier.renderOrder=-1;
  if(kind.endsWith('bundle')) { const group=new T.BundleGroup();group.add(earlier);f.scene.add(group); }
  else f.scene.add(earlier);
  let active=kind!=='warm bundle';
  f.tint.updateBeforeType='object';
  f.tint.updateBefore=()=>{if(active)withThreeNativeComputePrerequisites(f.renderer,()=>f.renderer.compute(kernel));};
  await f.renderer.init();
  if(!active) { f.renderer.render(f.scene,f.camera);await drain(f.renderer);active=true;f.renderer._nodes.nodeFrame.update(); }
  const writes=f.device.writes.length,submissions=f.device.submissions.length;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_COMPUTE_REENTRY'});
  assert.equal(f.device.computePasses.length,0);assert.equal(f.device.submissions.length,submissions);
  assert.equal(f.device.writes.length,writes);assert.equal(f.renderer._nodeCaptureBytes,0);
});

for(const initialized of [false,true])
test(`accepted effect compute survives a later failed parent with initialized=${initialized}`,async t=>{
  const f=fixture(t);recordCompute(f.device);
  const attribute=new T.StorageBufferAttribute(4,4),kernel=storageKernel(T.TSL.storage(attribute,'vec4',4));
  f.tint.updateBeforeType='object';
  f.tint.updateBefore=()=>withThreeNativeComputePrerequisites(f.renderer,()=>{
    f.renderer.compute(kernel);
    throw new Error('effect capture failed');
  });
  if(initialized)await f.renderer.init();
  await assert.rejects(f.renderer.renderAsync(f.scene,f.camera),/effect capture failed/);
  assert.equal(f.device.computePasses.length,1);assert.equal(nativeDraws(f.device).length,0);
  assert.equal(f.device.submissions.length,1);assert.equal(f.renderer._nodeCaptureBytes,0);
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

for(const registered of [false,true])
test(`custom named storage vertices retain compute allocations and queued replacements (registered=${registered})`,async t=>{
  const f=fixture(t);recordCompute(f.device);f.renderer.outputColorSpace=T.LinearSRGBColorSpace;
  let setups=0;
  class Extension extends T.MeshBasicNodeMaterial{
    constructor(){super();this.applicationGain=.3125;}
    setupOutput(builder,output){
      setups++;return super.setupOutput(builder,T.TSL.vec4(output.rgb
        .mul(T.TSL.materialReference('applicationGain','float')),output.a));
    }
  }
  class Application extends T.MeshBasicMaterial{
    constructor(){super();this.type='StorageVertexApplication';this.applicationGain=.3125;}
  }
  if(registered)f.renderer.library.addMaterial(Extension,'StorageVertexApplication');
  const material=new(registered?Application:Extension)(),other=new T.Mesh(f.mesh.geometry.clone(),material);
  f.mesh.material=material;f.mesh.frustumCulled=other.frustumCulled=false;
  f.mesh.renderOrder=-1;other.renderOrder=1;f.scene.add(other);
  t.after(()=>{other.geometry.dispose();material.dispose();});
  const original=f.mesh.geometry.attributes.position;
  const attributes=Array.from({length:3},()=>new T.StorageBufferAttribute(original.array.slice(),3));
  const [a,b,replacement]=attributes;
  const sourceDevice=geometryDevice(),sourceAttribute=new T.StorageBufferAttribute(original.array.slice(),3),sourceRecords=new WeakMap();
  const sourceUploader=new SourceWebGPUAttributeUtils({device:sourceDevice,get(attribute){
    let record=sourceRecords.get(attribute);if(!record)sourceRecords.set(attribute,record={});return record;
  }});
  sourceUploader.createAttribute(sourceAttribute,128|32|8|4);
  const sourceBuffer=sourceRecords.get(sourceAttribute).buffer;
  t.after(()=>sourceBuffer.destroy());
  f.mesh.geometry.setAttribute('position',a);other.geometry.setAttribute('position',b);
  const nodes=attributes.map(attribute=>T.TSL.storage(attribute,'vec3',attribute.count));
  const kernel=T.TSL.Fn(()=>{
    for(const node of nodes)node.element(T.TSL.instanceIndex).addAssign(T.TSL.vec3(0));
  })().compute(original.count);
  await f.renderer.compileComputeAsync(kernel);
  assert.ok(attributes.every(attribute=>attribute.itemSize===4),'compute prepares the source vec3 padding');
  const buffers=attributes.map(attribute=>captureComputeStorage(f.device,attribute).resolve().buffer);
  assert.equal(new Set(buffers).size,3,'separate geometry inputs own separate live storage');
  // The recorder does not execute WGSL. Model earlier GPU results while the
  // public CPU arrays stay stale, then inspect bytes at command submission.
  for(const [index,buffer]of buffers.entries())new Float32Array(buffer.data)[0]=[.625,1.125,2.125][index];
  new Float32Array(sourceBuffer.data)[0]=2.125;
  const initialWrites=buffers.map(buffer=>f.device.writes.filter(write=>write.buffer===buffer).length);
  const order=[],submit=f.device.queue.submit.bind(f.device.queue);
  f.device.queue.submit=commands=>{
    for(const passes of commands)for(const pass of passes){
      if(pass.dispatches?.length)order.push('compute');
      for(const draw of pass.draws)if(draw.streams.size)order.push('draw');
    }
    submit(commands);
  };
  const first=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  a.array[0]=b.array[0]=99;f.mesh.geometry.setAttribute('position',replacement);
  f.renderer.render(f.scene,f.camera);
  for(const attribute of [replacement,sourceAttribute]){
    attribute.array[0]=3.125;attribute.array[1]=attribute.array[4]=99;
    attribute.addUpdateRange(0,1);attribute.needsUpdate=true;
  }
  // r186 expands a packed vec3 range to the whole padded vertex. Compare the
  // actual source uploader, including untouched vertices outside that range.
  sourceUploader.updateAttribute(sourceAttribute);
  assert.deepEqual(sourceDevice.writes.map(write=>[write.offset,write.input.byteLength]),[[0,16]]);
  const second=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  assert.equal(setups,1,'equivalent geometry owners and named replacements share source construction');
  for(const attribute of attributes)attribute.array.fill(99);
  material.applicationGain=99;
  await Promise.all([first,second,drain(f.renderer)]);await f.renderer.waitForGPU();
  const draws=nativeDraws(f.device),selected=[buffers[0],buffers[1],buffers[2],buffers[1],buffers[2],buffers[1]];
  assert.equal(draws.length,6);
  assert.deepEqual(order,['compute','draw','draw','draw','draw','compute','draw','draw']);
  for(const [index,draw]of draws.entries()){
    assert.equal(draw.streams.get(0),selected[index],'each accepted draw retains its selected compute allocation');
    assert.equal(draw.pipeline.vertex.buffers[0].arrayStride,16);
    assert.equal(draw.pipeline.vertex.buffers[0].attributes[0].format,'float32x4');
    assert.equal(draw.pipeline,draws[0].pipeline);
    assert.ok(uniformBlocks(draw).flat().includes(.3125),'queued material uniforms keep their accepted value');
  }
  const words=(draw,buffer)=>new Float32Array(draw.contents.get(buffer).buffer);
  assert.deepEqual(draws.map((draw,index)=>words(draw,selected[index])[0]),[.625,1.125,2.125,1.125,3.125,1.125]);
  assert.deepEqual(draws[4].contents.get(buffers[2]),new Uint8Array(sourceBuffer.data));
  assert.equal(words(draws[4],buffers[2])[1],99,'the source padded range uploads the complete first vertex');
  assert.equal(words(draws[4],buffers[2])[4],original.array[3],'vertices outside the source range retain resident GPU bytes');
  assert.deepEqual(f.device.writes.filter(write=>write.buffer===buffers[2]).slice(initialWrites[2])
    .map(write=>[write.offset,write.input.byteLength]),sourceDevice.writes.map(write=>[write.offset,write.input.byteLength]));
  assert.equal(f.device.computePasses.length,2);
  for(const [index,buffer]of buffers.entries()){
    assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,initialWrites[index]+(index===2?1:0),
      'only the marked host range uploads; graphics never overwrites resident storage');
    assert.ok(f.device.computePasses.every(pass=>[...pass.groups.values()].some(group=>
      group.entries.some(entry=>entry.resource?.buffer===buffer))));
  }
  assert.equal(f.device.buffers.filter(buffer=>buffer.label==='f3d-compute-buffer'&&(buffer.usage&128)).length,3);
  assert.ok(f.device.buffers.every(buffer=>(buffer.usage&1)===0),'vertices borrow storage without readback');
  assert.equal(setups,1);assert.equal(f.renderer._nodeCaptureBytes,0);
});

for(const registered of [false,true])
test(`custom storage instances borrow compute allocations and preserve queued GPU history (registered=${registered})`,async t=>{
  const f=fixture(t);recordCompute(f.device);f.renderer.outputColorSpace=T.LinearSRGBColorSpace;
  const {mesh,matrices,colors}=storageInstances(f),updates=[],after=[];
  let requested=3,setups=0;
  const pulse=T.TSL.uniform(1).onObjectUpdate(frame=>{
    assert.equal(frame.object,mesh);assert.equal(frame.material,material);
    frame.object.count=requested;updates.push(requested);return 1;
  });
  pulse.updateAfterType='object';pulse.updateAfter=frame=>{after.push(frame.object.count);frame.object.count=99;};
  class Extension extends T.MeshBasicNodeMaterial{
    constructor(){super();this.applicationGain=.3125;}
    setupOutput(builder,output){
      setups++;return super.setupOutput(builder,T.TSL.vec4(output.rgb
        .mul(T.TSL.materialReference('applicationGain','float')).mul(pulse),output.a));
    }
  }
  class Application extends T.MeshBasicMaterial{
    constructor(){super();this.type='StorageInstanceApplication';this.applicationGain=.3125;}
  }
  if(registered)f.renderer.library.addMaterial(Extension,'StorageInstanceApplication');
  const material=new(registered?Application:Extension)();mesh.material=material;
  t.after(()=>{mesh.dispose();material.dispose();});
  const matrixNode=T.TSL.storage(matrices,'mat4',4),colorNode=T.TSL.storage(colors,'vec3',4);
  const kernel=T.TSL.Fn(()=>{
    const index=T.TSL.instanceIndex;
    matrixNode.element(index).assign(matrixNode.element(index));
    colorNode.element(index).addAssign(T.TSL.vec3(0));
  })().compute(4);
  await f.renderer.compileComputeAsync(kernel);
  const buffers=[matrices,colors].map(attribute=>captureComputeStorage(f.device,attribute).resolve().buffer);
  // Model prior GPU-produced bytes at the recording boundary. The public CPU
  // arrays stay stale; these identity kernels must retain the resident result.
  new Float32Array(buffers[0].data)[12]=.625;new Float32Array(buffers[1].data)[0]=.375;
  assert.equal(matrices.array[12],0);assert.equal(colors.array[0],.125);
  const initialWrites=buffers.map(buffer=>f.device.writes.filter(write=>write.buffer===buffer).length);
  const first=f.renderer.compute(kernel);f.renderer.render(f.scene,f.camera);
  matrices.array[12]=99;colors.array[0]=99;requested=1;material.applicationGain=.5625;
  f.renderer.render(f.scene,f.camera);
  matrices.array[12]=1.5;matrices.needsUpdate=true;colors.array[0]=.875;colors.needsUpdate=true;
  const second=f.renderer.compute(kernel);requested=2;material.applicationGain=.8125;
  f.renderer.render(f.scene,f.camera);requested=0;f.renderer.render(f.scene,f.camera);
  assert.equal(setups,1);assert.deepEqual(updates,[3,1,2,0]);assert.deepEqual(after,updates);
  matrices.array.fill(99);colors.array.fill(99);material.applicationGain=99;mesh.count=99;
  await Promise.all([first,second,drain(f.renderer)]);await f.renderer.waitForGPU();
  const draws=nativeDraws(f.device);
  assert.deepEqual(draws.map(draw=>draw.args[1]),[3,1,2]);
  // The pinned observer retains object uniforms on SHARED refresh: an
  // untracked application property and unmarked CPU edit do not invalidate
  // them. The marked storage version requests FULL on the third draw.
  for(const [index,draw] of draws.entries()){
    assert.ok(buffers.every(buffer=>storageResources(draw).some(resource=>resource.buffer===buffer)));
    assert.ok(uniformBlocks(draw).flat().includes([.3125,.3125,.8125][index]),
      `draw ${index} keeps its accepted gain: ${JSON.stringify(uniformBlocks(draw))}`);
    assert.equal(draw.pipeline,draws[0].pipeline);
  }
  const words=(draw,buffer)=>new Float32Array(draw.contents.get(buffer).buffer);
  assert.deepEqual(draws.map(draw=>words(draw,buffers[0])[12]),[.625,.625,1.5]);
  assert.deepEqual(draws.map(draw=>words(draw,buffers[1])[0]),[.375,.375,.875]);
  assert.equal(f.device.computePasses.length,2);
  for(const [index,buffer] of buffers.entries()){
    assert.equal(f.device.writes.filter(write=>write.buffer===buffer).length,initialWrites[index]+1,
      'graphics never uploads stale CPU data and only the marked host version uploads');
    assert.ok(f.device.computePasses.every(pass=>[...pass.groups.values()].some(group=>
      group.entries.some(entry=>entry.resource?.buffer===buffer))));
  }
  assert.equal(f.device.buffers.filter(buffer=>buffer.label==='f3d-compute-buffer'&&(buffer.usage&128)).length,2);
  assert.ok(f.device.buffers.every(buffer=>(buffer.usage&1)===0),'compute-to-render never allocates a readback buffer');
  assert.equal(setups,1);assert.equal(f.renderer._nodeCaptureBytes,0);
  const writes=f.device.writes.length,submissions=f.device.submissions.length,acquired=f.acquired();requested=5;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_OBJECT'});
  assert.equal(f.device.writes.length,writes);assert.equal(f.device.submissions.length,submissions);
  assert.equal(f.acquired(),acquired);assert.equal(f.renderer._nodeCaptureBytes,0);
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

test('source material arrays preserve per-group maps alongside captured ordinary materials',async t=>{
  const f=fixture(t),a=new T.MeshStandardMaterial({map:dataTexture(220,30,70)}),b=new T.MeshPhysicalMaterial({map:dataTexture(40,170,230)});
  a.positionNode=b.positionNode=T.TSL.positionLocal;
  f.mesh.geometry.clearGroups();f.mesh.geometry.addGroup(0,3,0);f.mesh.geometry.addGroup(3,3,1);f.mesh.material=[a,b];
  const ordinary=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial({color:0xff0000}));ordinary.position.z=.25;f.scene.add(ordinary);
  f.scene.add(new T.AmbientLight(0xffffff,.5));
  await f.renderer.compileAsync(f.scene,f.camera);assert.equal(f.device.textureWrites.length,0);
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const native=nativeDraws(f.device);assert.equal(native.length,3);
  assert.deepEqual(native.map(draw=>draw.args[0]),[6,3,3]);
  assert.deepEqual(native.slice(1).map(draw=>[...textureResources(draw).find(view=>view.texture.format==='rgba8unorm').texture.pixels]),
    [[220,30,70,255],[40,170,230,255]]);
  assert.equal(f.device.snapshots.flat().filter(draw=>draw.pipeline.vertex.entryPoint==='vertex_main'&&draw.streams.size).length,0);
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
  f.mesh.receiveShadow=true;f.renderer.shadowMap.enabled=true;f.renderer.shadowMap.type=T.PCFShadowMap;
  sun.shadow.filterNode=T.TSL.float(1);
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_SHADOW'});sun.shadow.filterNode=null;
  sun.shadow.shadowNode=T.TSL.float(1);
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_SHADOW'});
  sun.shadow.shadowNode=undefined;
  assert.equal(updates,0);assert.equal(f.acquired(),0);assert.equal(f.device.submissions.length,0);
  assert.equal(f.renderer._nodeCaptureBytes,0);
  f.renderer.shadowMap.enabled=false;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);assert.equal(updates,1);assert.equal(nativeDraws(f.device).length,1);
  assert.equal(sun.shadow.map,null);
});

for(const Material of [T.MeshPhysicalMaterial,T.MeshPhysicalNodeMaterial])
for(const [feature,shader] of [['sheen',/sheenSpecularDirect/],['iridescence',/fn evalIridescence/],
  ['anisotropy',/fn D_GGX_Anisotropic/],['retroreflectivity',/Retroreflectivity/]])
test(`ordinary ${Material.name} ${feature} selects native shading without authored nodes`,async t=>{
  const f=fixture(t),material=new Material({color:new T.Color(.125,.375,.625),roughness:.4});
  material[feature]=.5;f.mesh.material=material;
  const light=new T.DirectionalLight(0xffffff,2);light.position.set(1,2,3);f.scene.add(light);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  material.color.setRGB(.875,.625,.375);material[feature]=.75;f.renderer.render(f.scene,f.camera);
  material.color.setRGB(0,0,0);material[feature]=0;
  await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  assert.ok(draws.every(draw=>shader.test(draw.pipeline.fragment.module.code)));
  assert.ok(containsColor(draws[0],[.125,.375,.625]));assert.ok(containsColor(draws[1],[.875,.625,.375]));
  assert.equal(draws[0].pipeline,draws[1].pipeline,'live factors reuse the native variant');
  assert.equal(f.mesh.material,material);assert.equal(material.colorNode??null,null);
  assert.equal(f.renderer._nodeCaptureBytes,0);
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.presentedRenders,3,'disabling the lobe returns through the ordinary preparation boundary');
});

for(const slot of ['bumpMap','displacementMap'])
test(`ordinary built-in ${slot} reaches source native shaders and captures queued image versions`,async t=>{
  const f=fixture(t),data=new Uint8Array(2*2*4).fill(64),map=new T.DataTexture(data,2,2);
  map.generateMipmaps=false;map.needsUpdate=true;
  const material=new T.MeshPhongMaterial({color:0x4488cc,[slot]:map});
  f.mesh.material=material;f.scene.add(new T.DirectionalLight(0xffffff,2));
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  data.fill(192);map.needsUpdate=true;f.renderer.render(f.scene,f.camera);
  data.fill(0);await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,2);
  const code=draws[0].pipeline[slot==='displacementMap'?'vertex':'fragment'].module.code;
  assert.match(code,/textureLoad|textureSample/);
  assert.match(code,slot==='displacementMap'?/positionLocal = \( positionLocal \+/:/dpdx/);
  const uploads=f.device.textureWrites.filter(write=>write.bytes.length===data.byteLength);
  assert.ok(uploads.some(write=>write.bytes.every(value=>value===64)));
  assert.ok(uploads.some(write=>write.bytes.every(value=>value===192)));
  assert.equal(material.colorNode,undefined);assert.equal(material[slot],map);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('unprepared fixed-native-fixed material transitions preserve each accepted state and source disposal',async t=>{
  const f=fixture(t),material=new T.MeshPhysicalMaterial({color:new T.Color(.125,.375,.625),roughness:.4});
  f.mesh.material=material;f.scene.add(new T.DirectionalLight(0xffffff,2));
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  material.sheen=.5;material.color.setRGB(.875,.625,.375);f.renderer.render(f.scene,f.camera);
  material.sheen=0;material.color.setRGB(.25,.5,.75);f.renderer.render(f.scene,f.camera);
  material.color.setRGB(0,0,0);
  await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.has(0));
  assert.equal(draws.length,3);
  assert.deepEqual(draws.map(draw=>draw.pipeline.vertex.entryPoint),['vertex_main','main','vertex_main']);
  for(const [index,color] of [[.125,.375,.625],[.875,.625,.375],[.25,.5,.75]].entries())
    assert.ok(containsColor(draws[index],color),'accepted material color '+index+' must survive later edits');
  assert.equal(f.mesh.material,material);assert.equal(material.sheen,0);
  assert.deepEqual(material.color.toArray(),[0,0,0]);
  assert.equal(f.renderer.info.f3d.presentedRenders,3);assert.equal(f.renderer._pending.length,0);
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.renderer._materialCaptureValues,0);
  assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs,1);
  material.dispose();assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs,0,
    'the native cache must listen to the original material, not captured state');
  material.sheen=.75;material.color.setRGB(.375,.125,.625);
  await f.renderer.renderAsync(f.scene,f.camera);
  assert.equal(f.renderer.info.f3d.presentedRenders,4);
  assert.ok(containsColor(nativeDraws(f.device).at(-1),[.375,.125,.625]));
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('queued fixed material arrays retain group selection and independent material values',async t=>{
  const f=fixture(t),a=new T.MeshBasicMaterial({color:new T.Color(.125,.25,.375)}),
    b=new T.MeshBasicMaterial({color:new T.Color(.625,.75,.875)});
  f.mesh.geometry.clearGroups();f.mesh.geometry.addGroup(0,3,0);f.mesh.geometry.addGroup(3,3,1);
  const materials=[a,b];f.mesh.material=materials;
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  materials.reverse();a.color.setRGB(.25,.375,.5);b.color.setRGB(.5,.625,.75);
  f.renderer.render(f.scene,f.camera);
  f.mesh.material=new T.ShaderMaterial();a.color.setRGB(0,0,0);b.color.setRGB(0,0,0);
  await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.has(0));
  assert.equal(draws.length,4);assert.ok(draws.every(draw=>draw.pipeline.vertex.entryPoint==='vertex_main'));
  assert.deepEqual(draws.map(draw=>draw.args[0]),[3,3,3,3]);
  for(const [index,color] of [[.125,.25,.375],[.625,.75,.875],[.5,.625,.75],[.25,.375,.5]].entries())
    assert.ok(containsColor(draws[index],color),'queued group '+index+' lost its material selection');
  assert.equal(f.renderer.info.f3d.presentedRenders,2);assert.equal(f.renderer._nodeCaptureBytes,0);
  assert.equal(f.renderer._materialCaptureValues,0);
  assert.ok(f.mesh.material.isShaderMaterial,'request capture must not replace the application material');
});

test('captured fixed double-sided passes do not overwrite later source side or version',async t=>{
  const f=fixture(t),material=new T.MeshBasicMaterial({transparent:true,side:T.DoubleSide,opacity:.5});
  f.mesh.material=material;
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  material.side=T.FrontSide;const version=material.version;
  await drain(f.renderer);
  const draws=f.device.snapshots.flat().filter(draw=>draw.streams.has(0));
  assert.equal(draws.length,2);assert.equal(material.side,T.FrontSide);assert.equal(material.version,version);
  assert.equal(f.renderer.info.f3d.presentedRenders,1);assert.equal(f.renderer._materialCaptureValues,0);
});

test('fixed material capture budgets are released on disposal and preparation failure',async t=>{
  for(const failure of [false,true]){
    const f=fixture(t),material=new T.MeshBasicMaterial();f.mesh.material=material;
    if(failure)material.alphaHash=true;
    await f.renderer.init();f.renderer.render(f.scene,f.camera);
    assert.ok(f.renderer._materialCaptureValues>0);
    if(failure)await assert.rejects(drain(f.renderer),{code:'THREE_SCENE_MATERIAL'});
    else{f.renderer.dispose();await drain(f.renderer);}
    assert.equal(f.renderer._materialCaptureValues,0);assert.equal(f.renderer._pending.length,0);
    assert.equal(f.renderer._nodeCaptureBytes,0);
  }
});


test('fixed raw history rejects legacy shadows adoption without poisoning later draws',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const geometry=f.mesh.geometry,position=geometry.attributes.position;position.array.fill(.125);
  await f.renderer.compileAsync(f.scene,f.camera);position.array.fill(99);
  const writes=f.device.writes.length,submissions=f.device.submissions.length;
  f.renderer.shadowMap.enabled=true;
  assert.throws(()=>f.renderer.render(f.scene,f.camera),{code:'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(f.device.writes.length,writes);assert.equal(f.device.submissions.length,submissions);
  assert.equal(f.renderer._nodeCaptureBytes,0,'refused route retains no queued leases');
  f.renderer.shadowMap.enabled=false;f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=()=>f.device.snapshots.flat().filter(draw=>draw.streams.size>0);
  const firstWord=draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];};
  assert.equal(draws().length,1);assert.equal(firstWord(draws()[0]),.125,'refusal preserves the accepted history');
  geometry.dispose();f.renderer.shadowMap.enabled=true;f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const fresh=draws().slice(1);assert.equal(fresh.length,1);
  assert.ok(fresh.every(draw=>firstWord(draw)===99),'source disposal permits a fresh fixed upload');
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('ordinary instances share earlier fixed geometry generations and source disposal restarts both views',async t=>{
  const f=fixture(t);f.mesh.material=new T.MeshBasicMaterial();f.mesh.frustumCulled=false;
  const geometry=f.mesh.geometry,position=geometry.attributes.position;position.array.fill(.125);
  await f.renderer.compileAsync(f.scene,f.camera);position.array.fill(99);
  const aliasGeometry=geometry.clone();aliasGeometry.setAttribute('position',position);
  const alias=new T.InstancedMesh(aliasGeometry,new T.MeshBasicMaterial(),1);alias.frustumCulled=false;
  t.after(()=>{aliasGeometry.dispose();alias.material.dispose();});
  f.scene.add(alias);f.renderer.render(f.scene,f.camera);f.scene.remove(alias);f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  const firstWord=draw=>{const bytes=draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)[0];};
  assert.deepEqual(nativeDraws(f.device).map(firstWord),[.125,.125,.125]);
  geometry.dispose();f.scene.add(alias);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  assert.deepEqual(nativeDraws(f.device).slice(-2).map(firstWord),[99,99]);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

const customInstanceDraws=device=>nativeDraws(device).filter(draw=>instanceStreams(draw).length>0);

for(const registered of [false,true])
test(`custom ${registered?'registered':'inherited'} InstancedMesh materials retain queued uniforms, colors and zero counts`,async t=>{
  const f=fixture(t),setups=[],updates=[];
  class Extension extends T.MeshBasicNodeMaterial {
    constructor(){super();this.applicationGain=.375;}
    setupOutput(builder,output){
      setups.push(builder.object);
      return super.setupOutput(builder,T.TSL.vec4(
        output.rgb.mul(T.TSL.materialReference('applicationGain','float')),output.a));
    }
  }
  if(registered)f.renderer.library.materialNodes.set('MeshBasicMaterial',Extension);
  const material=registered?new T.MeshBasicMaterial():new Extension();
  material.applicationGain=.375;f.material=material;
  const {mesh,matrices,colors}=cpuInstances(f);
  mesh.onBeforeRender=()=>updates.push(mesh.count);
  await f.renderer.init();f.renderer.render(f.scene,f.camera);
  matrices.array[12]=1.5;colors.array[0]=.875;colors.needsUpdate=true;
  material.applicationGain=.625;mesh.count=2;
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  mesh.count=0;f.renderer.render(f.scene,f.camera);
  material.applicationGain=99;mesh.count=99;matrices.array.fill(99);colors.array.fill(99);
  await drain(f.renderer);
  const draws=customInstanceDraws(f.device);
  assert.deepEqual(draws.map(draw=>draw.args[1]),[3,2]);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===48)?.[12]),[0,1.5]);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw).map(stream=>stream.values[0])),[[.125],[.875]]);
  assert.ok(uniformBlocks(draws[0]).flat().includes(.375));
  assert.ok(uniformBlocks(draws[1]).flat().includes(.625));
  assert.equal(draws[0].pipeline,draws[1].pipeline);
  assert.deepEqual(setups,[mesh]);assert.deepEqual(updates,[3,2,0]);
  assert.equal(mesh.material,material);assert.equal(matrices.version,0);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom InstancedMesh materials keep source instance owners separate when a material is shared',async t=>{
  const f=fixture(t),setups=[];
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder,output){setups.push(builder.object);return super.setupOutput(builder,output);}
  }
  f.material=new Extension();
  const left=cpuInstances(f,2),right=cpuInstances(f,2);
  left.mesh.renderOrder=-1;right.mesh.renderOrder=1;
  left.matrices.array[12]=.25;right.matrices.array[12]=1.75;
  left.colors.array[0]=.125;right.colors.array[0]=.875;
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(customInstanceDraws(f.device).length,0);
  left.colors.array[0]=99;right.colors.array[0]=99;
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  let draws=customInstanceDraws(f.device);
  assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===32&&words[15]===1&&words[31]===1)?.[12]),[.25,1.75]);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw)[0].values[0]),[.125,.875]);
  assert.deepEqual(setups,[left.mesh,right.mesh]);
  left.colors.array[0]=.625;left.colors.needsUpdate=true;
  right.matrices.array[12]=2.25;right.matrices.needsUpdate=true;f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  draws=customInstanceDraws(f.device).slice(2);
  assert.deepEqual(draws.map(draw=>uniformBlocks(draw).find(words=>words.length===32&&words[15]===1&&words[31]===1)?.[12]),[.25,2.25]);
  assert.deepEqual(draws.map(draw=>instanceStreams(draw)[0].values[0]),[.625,.875]);
  assert.deepEqual(setups,[left.mesh,right.mesh]);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom InstancedMesh material disposal rebuilds construction without exposing stale CPU colors',async t=>{
  const f=fixture(t);let setups=0;
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder,output){setups++;return super.setupOutput(builder,output);}
  }
  f.material=new Extension();const {colors}=cpuInstances(f,2);
  await f.renderer.compileAsync(f.scene,f.camera);f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  colors.array[0]=99;f.material.dispose();
  f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  const draws=customInstanceDraws(f.device);
  assert.equal(draws.length,2);assert.deepEqual(draws.map(draw=>instanceStreams(draw)[0].values[0]),[.125,.125]);
  assert.equal(setups,2);assert.equal(f.renderer._nodeCaptureBytes,0);
});

function customSkinnedMesh(f,material,count=2){
  const geometry=new T.PlaneGeometry();
  geometry.setAttribute('skinIndex',new T.Uint16BufferAttribute(
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],4));
  geometry.setAttribute('skinWeight',new T.Float32BufferAttribute(
    [.25,.75,0,0,.25,.75,0,0,.25,.75,0,0,.25,.75,0,0],4));
  const mesh=new T.SkinnedMesh(geometry,material);
  const bones=Array.from({length:count},()=>new T.Bone());
  bones[0].add(...bones.slice(1));mesh.add(bones[0]);mesh.bind(new T.Skeleton(bones));
  mesh.frustumCulled=false;f.scene.remove(f.mesh);f.scene.add(mesh);
  return {mesh,bones,geometry};
}
const customSkinnedDraws=device=>nativeDraws(device).filter(draw=>
  /skinIndex/.test(draw.pipeline.vertex.module.code));
function skinPalette(draw){
  const code=draw.pipeline.vertex.module.code;
  const name=/struct\s+(NodeBuffer_\d+)Struct\s*\{\s*value\s*:\s*array<\s*mat4x4<f32>/.exec(code)?.[1];
  const declaration=[...code.matchAll(/((?:@(?:group|binding)\s*\(\s*\d+\s*\)\s*){2})var<uniform>\s+(NodeBuffer_\d+)\s*:/g)]
    .find(match=>match[2]===name);
  assert.ok(declaration,'source skinning exposes its uniform palette');
  const groupIndex=Number(/@group\s*\(\s*(\d+)/.exec(declaration[1])[1]);
  const binding=Number(/@binding\s*\(\s*(\d+)/.exec(declaration[1])[1]);
  const {group,offsets}=draw.groups.get(groupIndex);
  const entry=group.entries.find(value=>value.binding===binding),resource=entry.resource;
  const layout=group.layout.entries.filter(value=>value.buffer?.type==='uniform').sort((a,b)=>a.binding-b.binding);
  const offset=(resource.offset??0)+(offsets[layout.findIndex(value=>value.binding===binding)]??0);
  const bytes=draw.contents.get(resource.buffer);
  return [...new Float32Array(bytes.buffer,bytes.byteOffset+offset,resource.size/4)];
}

for(const registered of [false,true])
test(`custom ${registered?'registered':'inherited'} SkinnedMesh materials retain queued animation palettes and uniforms`,async t=>{
  const f=fixture(t),setups=[],updates=[];
  class Extension extends T.MeshBasicNodeMaterial {
    constructor(){super();this.applicationGain=.375;}
    setupOutput(builder,output){
      setups.push(builder.object);
      return super.setupOutput(builder,T.TSL.vec4(
        output.rgb.mul(T.TSL.materialReference('applicationGain','float')),output.a));
    }
  }
  if(registered)f.renderer.library.materialNodes.set('MeshBasicMaterial',Extension);
  const material=registered?new T.MeshBasicMaterial():new Extension();material.applicationGain=.375;
  const {mesh,bones,geometry}=customSkinnedMesh(f,material);
  t.after(()=>geometry.dispose());mesh.onBeforeRender=()=>updates.push(bones[1].position.x);
  await f.renderer.init();bones[1].position.x=.25;f.renderer.render(f.scene,f.camera);
  bones[1].position.x=.75;material.applicationGain=.625;f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene,f.camera);
  bones[1].position.x=99;material.applicationGain=99;mesh.skeleton.boneMatrices.fill(99);
  await drain(f.renderer);
  const draws=customSkinnedDraws(f.device);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>skinPalette(draw)[28]),[.25,.75]);
  assert.ok(uniformBlocks(draws[0]).flat().includes(.375));
  assert.ok(uniformBlocks(draws[1]).flat().includes(.625));
  assert.equal(draws[0].pipeline,draws[1].pipeline);
  assert.deepEqual(setups,[mesh]);assert.deepEqual(updates,[.25,.75]);
  assert.ok(geometry.attributes.skinIndex.array instanceof Uint16Array);
  assert.equal(geometry.attributes.skinIndex.gpuType,T.FloatType);
  assert.match(draws[0].pipeline.vertex.module.code,/skinIndex\s*:\s*vec4<u32>/);
  assert.equal(mesh.material,material);assert.equal(f.renderer._nodeCaptureBytes,0);
});

for(const paddedFirst of [false,true])
test(`custom SkinnedMesh materials share source construction with the ${paddedFirst?'first':'second'} palette padded`,async t=>{
  const f=fixture(t),setups=[];
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder,output){setups.push(builder.object);return super.setupOutput(builder,output);}
  }
  const material=new Extension(),left=customSkinnedMesh(f,material),right=customSkinnedMesh(f,material);
  t.after(()=>{left.geometry.dispose();right.geometry.dispose();});
  left.mesh.renderOrder=-1;right.mesh.renderOrder=1;
  (paddedFirst?left:right).mesh.skeleton.computeBoneTexture();
  assert.notEqual(left.mesh.skeleton.boneMatrices.length,right.mesh.skeleton.boneMatrices.length);
  left.bones[1].position.x=.25;right.bones[1].position.x=.75;
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(customSkinnedDraws(f.device).length,0);
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);await drain(f.renderer);
  let draws=customSkinnedDraws(f.device);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(draw=>skinPalette(draw)[28]),[.25,.75]);
  assert.equal(draws[0].pipeline.vertex.module.code,draws[1].pipeline.vertex.module.code);
  assert.equal(draws[0].pipeline.fragment.module.code,draws[1].pipeline.fragment.module.code);
  assert.deepEqual(setups,[left.mesh]);
  left.bones[1].position.x=.5;right.bones[1].position.x=.875;f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene,f.camera);
  left.mesh.skeleton.boneMatrices.fill(99);right.mesh.skeleton.boneMatrices.fill(99);
  await drain(f.renderer);draws=customSkinnedDraws(f.device).slice(2);
  assert.deepEqual(draws.map(draw=>skinPalette(draw)[28]),[.5,.875]);
  assert.deepEqual(setups,[left.mesh]);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom SkinnedMesh material replacement and disposal keep accepted skeleton snapshots',async t=>{
  const f=fixture(t);let setups=0;
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder,output){setups++;return super.setupOutput(builder,output);}
  }
  const material=new Extension(),{mesh,bones,geometry}=customSkinnedMesh(f,material);
  t.after(()=>geometry.dispose());await f.renderer.init();
  bones[1].position.y=.25;f.renderer.render(f.scene,f.camera);
  const previous=mesh.skeleton;
  mesh.skeleton=new T.Skeleton(bones,previous.boneInverses.map(matrix=>matrix.clone()));
  bones[1].position.y=.625;f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  await drain(f.renderer);
  let draws=customSkinnedDraws(f.device);
  assert.deepEqual(draws.map(draw=>skinPalette(draw)[29]),[.25,.625]);assert.equal(setups,1);
  material.dispose();bones[1].position.y=.875;f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene,f.camera);mesh.skeleton.boneMatrices.fill(99);await drain(f.renderer);
  draws=customSkinnedDraws(f.device);
  assert.deepEqual(draws.map(draw=>skinPalette(draw)[29]),[.25,.625,.875]);
  assert.equal(setups,2);assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom SkinnedMesh materials capture bone textures above the device uniform limit',async t=>{
  const f=fixture(t);
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder,output){return super.setupOutput(builder,output);}
  }
  const {mesh,bones,geometry}=customSkinnedMesh(f,new Extension(),1025);
  t.after(()=>geometry.dispose());await f.renderer.init();
  bones[1024].position.z=.25;f.renderer.render(f.scene,f.camera);
  bones[1024].position.z=.75;f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  bones[1024].position.z=99;mesh.skeleton.boneMatrices.fill(99);await drain(f.renderer);
  const draws=customSkinnedDraws(f.device);assert.equal(draws.length,2);
  assert.ok(draws.every(draw=>/textureLoad/.test(draw.pipeline.vertex.module.code)));
  const palettes=f.device.textureWrites.filter(write=>write.destination.texture.format==='rgba32float')
    .map(write=>new Float32Array(write.bytes.buffer,write.bytes.byteOffset,write.bytes.byteLength/4));
  assert.deepEqual(palettes.map(words=>words[1024*16+14]),[.25,.75]);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

function customMorphTargets(mesh,relative=false){
  const geometry=mesh.geometry;geometry.morphTargetsRelative=relative;
  for(const name of ['position','normal'])geometry.morphAttributes[name]=[0,1].map(target=>
    new T.Float32BufferAttribute(Array.from(geometry.attributes[name].array,(value,component)=>
      (relative?0:value)+(component%3===target?(name==='position'?.25:.125):0)),3));
  mesh.updateMorphTargets();mesh.morphTargetInfluences.splice(0,2,.25,.5);
  mesh.frustumCulled=false;return geometry;
}
const customMorphDraws=device=>nativeDraws(device).filter(draw=>
  /texture_2d_array<f32>/.test(draw.pipeline.vertex.module.code));
function capturedMorphWeights(draw){
  assert.match(draw.pipeline.vertex.module.code,/value : array< vec4<f32>, 2 >/);
  const words=uniformBlocks(draw).find(block=>block.length===8);
  assert.ok(words,'source morph weights retain their padded raw buffer');
  assert.deepEqual([words[1],words[2],words[3],words[5],words[6],words[7]],[0,0,0,0,0,0]);
  return [words[0],words[4]];
}

for(const registered of [false,true])for(const skinned of [false,true])
test(`custom ${registered?'registered':'inherited'} morph materials capture queued weights with skinning=${skinned}`,async t=>{
  const f=fixture(t),setups=[];f.device.limits.maxTextureArrayLayers=256;
  class Extension extends T.MeshBasicNodeMaterial {
    constructor(){super();this.applicationGain=.375;}
    setupOutput(builder,output){
      setups.push(builder.object);
      return super.setupOutput(builder,T.TSL.vec4(
        output.rgb.mul(T.TSL.materialReference('applicationGain','float')),output.a));
    }
  }
  if(registered)f.renderer.library.materialNodes.set('MeshBasicMaterial',Extension);
  const material=registered?new T.MeshBasicMaterial():new Extension();material.applicationGain=.375;
  const item=skinned?customSkinnedMesh(f,material):{mesh:f.mesh,geometry:f.mesh.geometry};
  item.mesh.material=material;customMorphTargets(item.mesh);t.after(()=>item.geometry.dispose());
  await f.renderer.init();
  if(skinned)item.bones[1].position.x=.125;
  f.renderer.render(f.scene,f.camera);
  item.mesh.morphTargetInfluences=[1.25,-.5];item.geometry.morphTargetsRelative=true;
  material.applicationGain=.625;if(skinned)item.bones[1].position.x=.875;
  f.renderer._nodes.nodeFrame.update();f.renderer.render(f.scene,f.camera);
  item.mesh.morphTargetInfluences.fill(99);material.applicationGain=99;
  if(skinned)item.mesh.skeleton.boneMatrices.fill(99);
  await drain(f.renderer);
  const draws=customMorphDraws(f.device);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(capturedMorphWeights),[[.25,.5],[1.25,-.5]]);
  assert.ok(uniformBlocks(draws[0]).flat().includes(.375));
  assert.ok(uniformBlocks(draws[1]).flat().includes(.625));
  if(skinned){
    assert.deepEqual(draws.map(draw=>skinPalette(draw)[28]),[.125,.875]);
    assert.ok(draws.every(draw=>draw.pipeline.vertex.module.code.indexOf('textureLoad')<
      draw.pipeline.vertex.module.code.indexOf('skinWeight.x')));
  }
  assert.equal(draws[0].pipeline,draws[1].pipeline);
  assert.deepEqual(setups,[item.mesh]);assert.equal(item.mesh.material,material);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom morph materials share the source atlas and construction while capturing independent object weights',async t=>{
  const f=fixture(t),setups=[];f.device.limits.maxTextureArrayLayers=256;
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder,output){setups.push(builder.object);return super.setupOutput(builder,output);}
  }
  const material=new Extension(),left=f.mesh;left.material=material;customMorphTargets(left);
  const right=new T.Mesh(new T.PlaneGeometry(),material);right.frustumCulled=false;
  right.geometry.morphAttributes={...left.geometry.morphAttributes};
  right.geometry.morphTargetsRelative=true;right.updateMorphTargets();right.morphTargetInfluences=[.75,-.25];
  left.renderOrder=-1;right.renderOrder=1;f.scene.add(right);
  t.after(()=>{left.geometry.dispose();right.geometry.dispose();});
  await f.renderer.compileAsync(f.scene,f.camera);
  assert.equal(customMorphDraws(f.device).length,0);
  f.renderer.render(f.scene,f.camera);left.morphTargetInfluences.fill(99);right.morphTargetInfluences.fill(99);
  await drain(f.renderer);
  const draws=customMorphDraws(f.device);assert.equal(draws.length,2);
  assert.deepEqual(draws.map(capturedMorphWeights),[[.25,.5],[.75,-.25]]);
  assert.deepEqual(setups,[left]);
  assert.equal(draws[0].pipeline.vertex.module.code,draws[1].pipeline.vertex.module.code);
  const atlases=new Set(f.device.textureWrites.filter(write=>write.destination.texture.format==='rgba32float')
    .map(write=>write.destination.texture));
  assert.equal(atlases.size,1,'shared target identities retain the first source geometry atlas');
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('custom morph materials retain source atlas lifetime and queued reconstruction',async t=>{
  const f=fixture(t);let setups=0;f.device.limits.maxTextureArrayLayers=256;
  class Extension extends T.MeshBasicNodeMaterial {
    constructor(){super();this.applicationVariant=0;}
    customProgramCacheKey(){return super.customProgramCacheKey()+':'+this.applicationVariant;}
    setupOutput(builder,output){setups++;return super.setupOutput(builder,output);}
  }
  const material=new Extension(),mesh=f.mesh;mesh.material=material;
  const geometry=customMorphTargets(mesh),target=geometry.morphAttributes.position[0];
  t.after(()=>geometry.dispose());await f.renderer.init();
  f.renderer.render(f.scene,f.camera);await f.renderer.waitForGPU();
  // Explicit texture disposal invalidates queued sampled uses; finish that draw
  // before geometry.dispose dispatches the source atlas texture's dispose event.
  target.array[0]=5;target.needsUpdate=true;geometry.dispose();material.needsUpdate=true;
  f.renderer.render(f.scene,f.camera);
  assert.equal(setups,1,'version-only invalidation retains the source graph and packed atlas');
  material.applicationVariant=1;material.needsUpdate=true;f.renderer.render(f.scene,f.camera);
  target.array[0]=99;mesh.morphTargetInfluences.fill(99);await drain(f.renderer);
  const draws=customMorphDraws(f.device);assert.equal(draws.length,3);
  assert.deepEqual(draws.map(capturedMorphWeights),[[.25,.5],[.25,.5],[.25,.5]]);
  const generations=new Map();
  for(const write of f.device.textureWrites)if(write.destination.texture.format==='rgba32float'&&
    !generations.has(write.destination.texture))
    generations.set(write.destination.texture,new Float32Array(write.bytes.buffer,write.bytes.byteOffset,write.bytes.byteLength/4)[0]);
  assert.deepEqual(draws.map(draw=>{
    const textures=[...draw.groups.values()].flatMap(({group})=>group.entries)
      .map(entry=>entry.resource?.texture).filter(texture=>texture?.format==='rgba32float');
    assert.equal(new Set(textures).size,1);return generations.get(textures[0]);
  }),[-.25,-.25,5]);
  assert.equal(setups,2);assert.equal(f.renderer._nodeCaptureBytes,0);
});
