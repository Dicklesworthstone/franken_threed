import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {inspectThreeLightingContext, inspectThreeNativePass} from './three_node_passes.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N=T.TSL;
function fixture(t,ordinary=false){
  const renderer={library:new T.StandardNodeLibrary(),contextNode:N.context(),
    debug:{diagnostics:{keywords:false}},depth:true,lighting:{enabled:true},
    backend:{hasFeature:()=>false,utils:{getTextureSampleData:()=>({samples:1,primarySamples:1,isMSAA:false})},
      capabilities:{getUniformBufferLimit:()=>65536}},
    hasFeature:()=>false,getRenderTarget:()=>null,getMRT:()=>null,currentSamples:1,
    _nodes:{nodeFrame:new T.NodeFrame()},info:{calls:1}};
  const texture=new T.DataTexture(new Uint8Array(8*8*4).fill(255),8,8);texture.name='context-input';
  texture.generateMipmaps=true;texture.minFilter=T.LinearMipmapLinearFilter;texture.needsUpdate=true;
  const material=ordinary?new T.MeshBasicMaterial({map:texture}):new T.MeshBasicNodeMaterial();
  if(!ordinary)material.colorNode=N.texture(texture);
  const object=new T.Mesh(new T.PlaneGeometry(),material),scene=new T.Scene();scene.add(object);
  const camera=new T.PerspectiveCamera(60,1,.1,100);camera.position.z=4;
  camera.coordinateSystem=T.WebGPUCoordinateSystem;camera.updateProjectionMatrix();
  scene.updateMatrixWorld();camera.updateMatrixWorld();
  const owner=createThreeNodePrograms({three:T,renderer});t.after(()=>{owner.dispose();object.geometry.dispose();material.dispose();texture.dispose();});
  const context={object,material,scene,camera,renderId:1,frameId:1,time:.2,deltaTime:.01};
  return {owner,renderer,texture,material,object,scene,camera,context};
}
function compareSource(f){
  const actual=f.owner.compile(f.material,f.object,f.context);
  const source=new T.WGSLNodeBuilder(f.object,f.renderer);source.scene=f.scene;source.camera=f.camera;source.build();
  assert.equal(actual.vertex,source.vertexShader);assert.equal(actual.fragment,source.fragmentShader);
  return actual;
}

for(const scope of ['renderer','material'])
test(`native ${scope} UV and mip-level contexts retain pinned construction and cache behavior`,t=>{
  const f=fixture(t,scope==='material'),calls=[];
  const hooks=N.context({
    getUV(texture,builder){calls.push(['uv',texture.value,builder.constructor]);return N.vec2(.125,.375);},
    getTextureLevel(texture){calls.push(['level',texture.value]);return N.float(1);}
  });
  f[scope].contextNode=hooks;
  assert.equal(calls.length,0);assert.ok(inspectThreeLightingContext(T,hooks));
  const descriptor=f.owner.compile(f.material,f.object,f.context),expected=calls.splice(0);
  assert.ok(expected.length>0);
  const source=new T.WGSLNodeBuilder(f.object,f.renderer);source.scene=f.scene;source.camera=f.camera;source.build();
  assert.deepEqual(calls,expected,'admission does not execute or repeat source construction hooks');
  assert.equal(descriptor.vertex,source.vertexShader);assert.equal(descriptor.fragment,source.fragmentShader);
  assert.match(descriptor.fragment,/textureSampleLevel/);
  calls.length=0;assert.equal(f.owner.compile(f.material,f.object,f.context),descriptor);
  f.owner.capture(descriptor,f.context);assert.deepEqual(calls,[],'warm capture does not reconstruct context hooks');
  if(scope==='renderer'){
    hooks.value.getUV=()=>N.vec2(.625,.875);hooks.needsUpdate=true;
    assert.notEqual(compareSource(f),descriptor,'source renderer context version invalidates construction');
  }
});

test('texture context hooks retain lazy Fn construction, original source objects and generated graph bindings',t=>{
  const f=fixture(t),calls=[],shift=N.uniform(new T.Vector2(.125,.25)).setName('lazyContextShift');
  f.material.contextNode=N.context({getUV:()=>N.Fn(({material,geometry,renderer})=>{
    calls.push([material,geometry,renderer]);return N.uv().add(shift);
  })(),getTextureLevel:()=>N.Fn(()=>N.float(1))()});
  const descriptor=f.owner.compile(f.material,f.object,f.context),candidate=calls.splice(0);
  assert.deepEqual(candidate,[[f.material,f.object.geometry,f.renderer]]);
  const source=new T.WGSLNodeBuilder(f.object,f.renderer);source.scene=f.scene;source.camera=f.camera;source.build();
  assert.deepEqual(calls,candidate);assert.equal(descriptor.vertex,source.vertexShader);assert.equal(descriptor.fragment,source.fragmentShader);
  assert.match(descriptor.fragment,/lazyContextShift/);assert.ok(descriptor.attributes.some(attribute=>attribute.name==='uv'));
});

test('material texture contexts override renderer hooks and null restores the source default UV',t=>{
  const f=fixture(t),calls=[];
  f.renderer.contextNode=N.context({getUV(){calls.push('renderer');return N.vec2(.125,.25);},
    getTextureLevel(){calls.push('renderer-level');return N.float(1);}});
  f.material.contextNode=N.context({getUV(){calls.push('material');return N.vec2(.75,.875);}});
  compareSource(f);assert.ok(calls.includes('material'));assert.ok(calls.includes('renderer-level'));assert.ok(!calls.includes('renderer'));
  calls.length=0;f.material.contextNode=N.context({getUV:null,getTextureLevel:null});f.material.needsUpdate=true;
  const next=compareSource(f);assert.deepEqual(calls,[]);assert.doesNotMatch(next.fragment,/textureSampleLevel/);
  assert.ok(next.attributes.some(attribute=>attribute.name==='uv'));
});

test('forceUVContext and explicit levels retain pinned hook calls and shader precedence',t=>{
  const f=fixture(t),calls=[];
  f.material.colorNode=N.texture(f.texture,N.vec2(.25,.5)).level(2);
  f.material.contextNode=N.context({getUV(){calls.push('uv');return N.vec2(.75,.875);},
    getTextureLevel(){calls.push('level');return N.float(1);},forceUVContext:false});
  compareSource(f);assert.ok(!calls.includes('uv'));
  // .level() also retains its reference texture node. That source node still
  // invokes the level hook even though the sampled clone has an explicit LOD.
  assert.deepEqual(calls,['level','level']);calls.length=0;
  f.material.contextNode=N.context({...f.material.contextNode.value,forceUVContext:true});f.material.needsUpdate=true;
  const actual=f.owner.compile(f.material,f.object,f.context),trace=calls.splice(0);
  const source=new T.WGSLNodeBuilder(f.object,f.renderer);source.scene=f.scene;source.camera=f.camera;source.build();
  assert.deepEqual(calls,trace);assert.ok(trace.includes('uv'));
  assert.equal(actual.vertex,source.vertexShader);assert.equal(actual.fragment,source.fragmentShader);
});

test('texture context chains retain source child precedence and reject unsupported context state without calling hooks',t=>{
  const f=fixture(t),child=N.context({getUV:()=>N.vec2(.75,.875),forceUVContext:true});
  const parent=N.context(child,{getUV:()=>N.vec2(.125,.25),forceUVContext:false});
  assert.equal(parent.getFlowContextData().getUV,child.value.getUV);assert.equal(parent.getFlowContextData().forceUVContext,true);
  f.material.contextNode=parent;compareSource(f);
  const key=child.getCacheKey();child.value.getUV=()=>N.vec2(.25,.5);child.needsUpdate=true;
  assert.equal(child.getCacheKey(),key,'source material node keys do not hash context hook identity or version');
  let calls=0;
  for(const values of [{outputColorSpace:T.SRGBColorSpace},{forceUVContext:1},{getUV:3},{getTextureLevel:undefined}]){
    const context=N.context(values);assert.throws(()=>inspectThreeLightingContext(T,context),{code:'THREE_NODE_PROGRAM_GRAPH'});
  }
  const data={};Object.defineProperty(data,'getUV',{enumerable:true,get(){calls++;return ()=>N.vec2(0);}});
  assert.throws(()=>inspectThreeLightingContext(T,N.context(data)),{code:'THREE_NODE_PROGRAM_GRAPH'});assert.equal(calls,0);
});

test('native PassNode admission retains its source texture context and versioned composition owner',t=>{
  const f=fixture(t),pass=N.pass(f.scene,f.camera),context=N.context({getUV:()=>N.vec2(.25,.75),getTextureLevel:()=>N.float(1),forceUVContext:true});
  pass.contextNode=context;t.after(()=>pass.dispose());
  assert.equal(inspectThreeNativePass(T,f.renderer,pass),true);assert.equal(pass.contextNode,context);
  const key=inspectThreeLightingContext(T,context);context.needsUpdate=true;
  assert.notEqual(inspectThreeLightingContext(T,context),key);
  pass.contextNode=N.context({getTextureLevel:1});
  assert.throws(()=>inspectThreeNativePass(T,f.renderer,pass),{code:'THREE_NODE_PROGRAM_PASS'});
});

// This existing recording seam proves captured bindings and queue order, not
// raster output. The builder cases above compare the complete pinned shaders.
function rendererFixture(t){
  const device=geometryDevice();Object.assign(device.limits,{maxTextureDimension2D:8192,maxBindingsPerBindGroup:1000,maxTextureArrayLayers:256});
  device.textures=[];
  device.createTexture=desc=>{const texture={dimension:'2d',depthOrArrayLayers:1,sampleCount:1,...desc,
    width:desc.size.width??desc.size[0],height:desc.size.height??desc.size[1],
    createView(options={}){return {texture,...options};},destroy(){this.destroyed=true;}};device.textures.push(texture);return texture;};
  device.createSampler=desc=>({...desc});device.queue.writeTexture=(destination,data)=>{
    destination.texture.pixels=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength).slice();
  };
  let configured;
  const canvas={width:64,height:64,style:{},getContext:()=>({configure(value){configured=value;},unconfigure(){},
    getCurrentTexture(){return device.createTexture({size:[64,64],format:configured.format,usage:16});}})};
  const Renderer=createWebGPURendererClass(T),renderer=new Renderer({canvas,device});
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(60,1,.1,100);camera.position.z=4;
  const texture=new T.DataTexture(new Uint8Array(8*8*4).fill(255),8,8);texture.needsUpdate=true;
  const material=new T.MeshBasicMaterial({map:texture}),object=new T.Mesh(new T.PlaneGeometry(),material);scene.add(object);
  t.after(()=>{renderer.dispose();object.geometry.dispose();material.dispose();texture.dispose();});
  return {renderer,device,scene,camera,texture,material,object};
}
const drain=async renderer=>{while(renderer._drain)await renderer._drain;};
const nativeDraws=device=>device.snapshots.flat().filter(draw=>draw.pipeline.vertex.entryPoint==='main');
function uniformValues(draw){
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
function containsUniform(draw,pair){
  return uniformValues(draw).some(words=>words.some((_,i)=>pair.every((value,j)=>words[i+j]===value)));
}
function sampledTextures(draw){
  return [...draw.groups.values()].flatMap(({group})=>group.entries.map(entry=>entry.resource?.texture).filter(Boolean));
}

for(const scope of ['renderer','material'])
test(`native ${scope} texture contexts capture live uniforms and queued context replacements`,async t=>{
  const f=rendererFixture(t),updates=[];
  if(scope==='material'){
    f.material=new T.MeshBasicNodeMaterial({map:f.texture});f.object.material=f.material;t.after(()=>f.material.dispose());
  }
  const first=N.uniform(new T.Vector2(.125,.375)).setName('firstContextUV');
  first.onObjectUpdate(frame=>{updates.push([frame.object,frame.material]);return first.value;});
  f[scope].contextNode=N.context({getUV:()=>first,getTextureLevel:()=>N.float(0)});
  await f.renderer.init();
  let release;const held=new Promise(resolve=>{release=resolve;}),create=f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync=async descriptor=>{await held;return create(descriptor);};
  f.renderer._advanceAnimationFrame();f.renderer.render(f.scene,f.camera);
  first.value.set(.625,.875);f.renderer._advanceAnimationFrame();f.renderer.render(f.scene,f.camera);
  const next=N.uniform(new T.Vector2(.25,.5)).setName('nextContextUV');
  f[scope].contextNode=N.context({getUV:()=>next,getTextureLevel:()=>N.float(0)});
  // Source NodeMaterial keys include the new context's identity. Ordinary
  // Material keys do not, so this replacement case uses a node material.
  if(scope==='material')f.material.needsUpdate=true;
  f.renderer._advanceAnimationFrame();f.renderer.render(f.scene,f.camera);
  first.value.set(99,99);next.value.set(99,99);release();await drain(f.renderer);
  const draws=nativeDraws(f.device);assert.equal(draws.length,3);
  for(const [index,pair] of [[.125,.375],[.625,.875],[.25,.5]].entries())assert.ok(containsUniform(draws[index],pair));
  assert.ok(draws.every(draw=>!containsUniform(draw,[99,99])));
  assert.deepEqual(updates,[[f.object,f.material],[f.object,f.material]]);
  assert.match(draws[0].pipeline.fragment.module.code,/firstContextUV/);
  assert.match(draws[2].pipeline.fragment.module.code,/nextContextUV/);
  const texture=sampledTextures(draws[0])[0];assert.ok(texture);
  assert.ok(draws.every(draw=>sampledTextures(draw).includes(texture)),'context changes retain the source texture residency');
  assert.equal(f.renderer._nodeCaptureBytes,0);
});

test('native PassNode composes renderer, pass and material texture contexts with source version invalidation',async t=>{
  const f=rendererFixture(t),global=N.context({getUV:()=>N.vec2(.125,.375),getTextureLevel:()=>N.float(0)});
  f.renderer.contextNode=global;
  const pass=N.pass(f.scene,f.camera);pass.contextNode=N.context({getUV:()=>N.vec2(.25,.5)});
  const outer=new T.Scene(),material=new T.MeshBasicNodeMaterial();material.colorNode=pass.getTextureNode();
  const object=new T.Mesh(new T.PlaneGeometry(),material);outer.add(object);
  t.after(()=>{pass.dispose();object.geometry.dispose();material.dispose();});
  await f.renderer.init();f.renderer._advanceAnimationFrame();f.renderer.render(outer,f.camera);await drain(f.renderer);
  assert.equal(f.renderer.contextNode,global);
  const cached=pass._contextNodeCache.context;
  assert.equal(cached.getFlowContextData().getUV,pass.contextNode.value.getUV);
  assert.equal(cached.getFlowContextData().getTextureLevel,global.value.getTextureLevel);
  assert.match(nativeDraws(f.device)[0].pipeline.fragment.module.code,/vec2<f32>\( 0\.25, 0\.5 \)/);
  pass.contextNode=N.context({getUV:()=>N.vec2(.625,.875)});
  f.renderer._advanceAnimationFrame();f.renderer.render(outer,f.camera);await drain(f.renderer);
  assert.equal(pass._contextNodeCache.context,cached,'source PassNode caches composition by its own version');
  assert.match(nativeDraws(f.device)[2].pipeline.fragment.module.code,/vec2<f32>\( 0\.25, 0\.5 \)/);
  pass.needsUpdate=true;f.material.contextNode=N.context({getUV:()=>N.vec2(.75,.25)});f.material.needsUpdate=true;
  f.renderer._advanceAnimationFrame();f.renderer.render(outer,f.camera);await drain(f.renderer);
  assert.notEqual(pass._contextNodeCache.context,cached);assert.equal(f.renderer.contextNode,global);
  assert.match(nativeDraws(f.device)[4].pipeline.fragment.module.code,/vec2<f32>\( 0\.75, 0\.25 \)/);
  assert.equal(f.renderer._nodeCaptureBytes,0);
});
