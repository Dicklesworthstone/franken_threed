/** Source forward-light graphs and queue-visible uniforms for many-light scenes.
 * The recording device checks programs/resources/bytes, not rasterized pixels.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';

const N=THREE.TSL;
const lightTypes=[THREE.AmbientLight,THREE.DirectionalLight,THREE.PointLight,THREE.SpotLight,
  THREE.HemisphereLight,THREE.RectAreaLight,THREE.LightProbe];
THREE.RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
function lightsFor(scene,count,{mixed=true}={}) {
  return Array.from({length:count},(_,index)=>{
    const light=new (mixed?lightTypes[index%lightTypes.length]:THREE.DirectionalLight)();
    light.color.setRGB((index+1)/64,(index+2)/128,(index+3)/256);light.intensity=.5;
    light.position.set(index/4+1,2,3);
    if(light.isPointLight||light.isSpotLight){light.distance=index%2?0:12;light.decay=index%3;}
    if(light.isSpotLight){light.angle=.5;light.penumbra=.125;}
    if(light.isHemisphereLight)light.groundColor.setRGB(.125,.25,.375);
    if(light.isRectAreaLight){light.width=2+index/8;light.height=1.5;}
    if(light.isLightProbe)light.sh.coefficients.forEach((value,slot)=>value.set((index+slot+1)/128,0,0));
    scene.add(light);return light;
  });
}
function sourceRig(t,{count=9,Material=THREE.MeshStandardMaterial,mixed=true,limits={},options={}}={}) {
  const frame=new THREE.NodeFrame();frame.frameId=1;
  const renderer={library:new THREE.StandardNodeLibrary(),contextNode:N.context(),debug:{diagnostics:{keywords:false}},
    depth:true,lighting:{enabled:true},sortObjects:true,_device:{limits},
    backend:{hasFeature:()=>false,utils:{getTextureSampleData:()=>({samples:1,primarySamples:1,isMSAA:false})},
      capabilities:{getUniformBufferLimit:()=>limits.maxUniformBufferBindingSize??65536}},
    hasFeature:()=>false,getRenderTarget:()=>null,getMRT:()=>null,currentSamples:1,_nodes:{nodeFrame:frame},info:{calls:1}};
  const programs=createThreeNodePrograms({three:THREE,renderer,...options});t.after(()=>programs.dispose());
  const scene=new THREE.Scene(),material=new Material(),object=new THREE.Mesh(new THREE.PlaneGeometry(),material);
  object.frustumCulled=false;scene.add(object);
  const lights=lightsFor(scene,count,{mixed}),camera=new THREE.PerspectiveCamera(50,1,.1,30);
  camera.position.z=4;camera.coordinateSystem=THREE.WebGPUCoordinateSystem;camera.updateProjectionMatrix();
  scene.updateMatrixWorld();camera.updateMatrixWorld();
  const context={object,material,scene,camera,lights,frameId:1,renderId:1};
  const compile=()=>programs.compile(material,object,context);
  const capture=()=>captureThreeNodeScene(THREE,scene,camera,{renderer,getPrograms:()=>programs,
    raster:value=>({side:value.side}),getTextures:()=>({captureBinding:value=>value,release(){},captureDispatch:run=>run(),validateDispatch(){}})});
  return {programs,renderer,frame,scene,material,object,lights,camera,context,compile,capture};
}
function sourceBuilder(f) {
  const builder=new THREE.WGSLNodeBuilder(f.object,f.renderer);
  builder.material=f.material;builder.context.material=f.material;builder.scene=f.scene;builder.camera=f.camera;
  builder.lightsNode=N.lights(f.lights.slice().reverse());builder.build();return builder;
}
function sourceUniforms(builder,descriptor,f) {
  const frame=new THREE.NodeFrame();Object.assign(frame,{renderer:f.renderer,scene:f.scene,object:f.object,
    material:f.material,camera:f.camera,frameId:1,renderId:1});
  for(const node of builder.updateBeforeNodes)frame.updateBeforeNode(node);
  for(const node of builder.updateNodes)frame.updateNode(node);
  return builder.getBindings().map((group,index)=>{
    const layout=descriptor.groups[index],packet=new Uint8Array(layout.byteLength);
    for(const member of layout.uniforms) {
      const binding=group.bindings[member.binding];binding.update();
      const array=binding.buffer;packet.set(new Uint8Array(array.buffer,array.byteOffset,array.byteLength),member.offset);
    }
    return packet;
  });
}
const floats=packets=>packets.flatMap(bytes=>[...new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)]);
// As in the existing area-light oracle, source Fn caching can reorder whole
// LTC helper declarations. Keep every body and the main accumulation verbatim.
const shaderOrder=shader=>shader.replace(/(\/\/ codes\n)([\s\S]*?)(?=@fragment)/,(_,prefix,functions)=>
  prefix+functions.split(/(?=^fn )/m).map(part=>part.trim()).filter(Boolean).sort().join('\n\n')+'\n\n');

for(const Material of [THREE.MeshLambertMaterial,THREE.MeshPhongMaterial,THREE.MeshToonMaterial,THREE.MeshStandardMaterial,THREE.MeshPhysicalMaterial])
test(`${Material.name} keeps every selected light in the exact pinned forward shader and uniforms`,t=>{
  for(const count of [9,32]) {
    const f=sourceRig(t,{count,Material}),descriptor=f.compile(),source=sourceBuilder(f);
    assert.equal(descriptor.vertex,source.vertexShader);assert.equal(shaderOrder(descriptor.fragment),shaderOrder(source.fragmentShader));
    const packet=f.programs.captureDraw(descriptor,f.context);
    // Replay the source builder's own updates before reading its buffers, so an
    // omitted native update cannot pass merely through shared default values.
    assert.deepEqual(packet.uniforms,sourceUniforms(source,descriptor,f));
    const selected=f.capture();assert.equal(selected.objects.has(f.object),true);
    assert.ok(f.lights.every(light=>selected.nativeLights.has(light)));
    assert.equal(selected.draws.length,1);assert.equal(f.material.colorNode,undefined);
    assert.ok(descriptor.groups.every(group=>group.uniforms.every(uniform=>uniform.byteLength<=65536)));
  }
});

test('many lights retain source frame updates and immutable accepted uniform packets',t=>{
  const f=sourceRig(t,{count:24,mixed:false}),descriptor=f.compile();
  const first=f.programs.captureDraw(descriptor,f.context),saved=first.uniforms.map(bytes=>bytes.slice());
  const light=f.lights.at(-1);light.color.setRGB(.8125,.625,.4375);light.intensity=2;
  const second=f.programs.captureDraw(descriptor,{...f.context,renderId:2});
  assert.deepEqual(second.uniforms,first.uniforms,'FRAME light colors do not change during another render in that frame');
  const third=f.programs.captureDraw(descriptor,{...f.context,frameId:2,renderId:3});
  for(const value of [1.625,1.25,.875])assert.ok(floats(third.uniforms).includes(value));
  assert.notDeepEqual(third.uniforms,first.uniforms);assert.deepEqual(first.uniforms,saved);
  assert.equal(f.compile(),descriptor,'ordinary color/intensity edits do not rebuild shaders');
});

test('many-light admission respects actual graph, uniform and shader budgets without truncation',t=>{
  for(const configuration of [
    {options:{maxNodes:100}}, {limits:{maxUniformBufferBindingSize:256}}, {options:{maxShaderBytes:1024}},
  ]) {
    const f=sourceRig(t,{count:9,mixed:false,...configuration});
    assert.throws(f.compile,{code:'THREE_NODE_PROGRAM_LIMIT'});
    assert.equal(f.programs.diagnostics.programs,0);
  }
  const f=sourceRig(t,{count:32,mixed:false});assert.doesNotThrow(f.compile);
});

test('unsupported mixed lit materials reject the whole native selection instead of losing their lights',t=>{
  class CustomStandard extends THREE.MeshStandardMaterial {}
  const f=sourceRig(t,{mixed:false}),other=new THREE.Mesh(f.object.geometry,new CustomStandard());
  f.scene.add(other);
  assert.throws(f.capture,{code:'THREE_NODE_PROGRAM_MATERIAL'});
  assert.equal(f.programs.diagnostics.programs,0,'selection is checked before any program or draw publication');
});

async function recording(t,{count=9,Material=THREE.MeshStandardMaterial}={}) {
  const device=geometryDevice();Object.assign(device.limits,{maxTextureDimension2D:8192,maxTextureArrayLayers:256,maxBindingsPerBindGroup:1000});
  device.features=new Set(['float32-filterable']);
  device.createTexture=descriptor=>({dimension:'2d',sampleCount:1,...descriptor,
    width:descriptor.size.width??descriptor.size[0],height:descriptor.size.height??descriptor.size[1],
    depthOrArrayLayers:descriptor.size.depthOrArrayLayers??descriptor.size[2]??1,
    createView(options={}){return {texture:this,...options};},destroy(){this.destroyed=true;}});
  device.createSampler=descriptor=>({...descriptor});device.queue.writeTexture=()=>{};
  device.createRenderPipeline=descriptor=>({...descriptor,getBindGroupLayout:()=>({})});
  let configuration;
  const canvas={width:32,height:32,style:{},getContext:()=>({configure(value){configuration=value;},unconfigure(){},
    getCurrentTexture(){return device.createTexture({size:[32,32],format:configuration.format,usage:configuration.usage});}})};
  const Renderer=createWebGPURendererClass(THREE),renderer=new Renderer({canvas,device});t.after(()=>renderer.dispose());await renderer.init();
  const scene=new THREE.Scene(),material=new Material(),object=new THREE.Mesh(new THREE.PlaneGeometry(),material);
  object.frustumCulled=false;scene.add(object);const lights=lightsFor(scene,count,{mixed:false});
  const camera=new THREE.PerspectiveCamera(50,1,.1,30);camera.position.z=4;
  return {device,renderer,scene,material,object,lights,camera};
}
const isColorDraw=draw=>!draw.pipeline.label?.startsWith('Animation HDR output ') && !!draw.pipeline.fragment;
const isNative=draw=>draw.pipeline.vertex.entryPoint!=='vertex_main';
const submittedFloats=draw=>[...draw.contents.values()].flatMap(bytes=>[...new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)]);

test('queued many-light renders retain all light versions and return to the small fixed path',async t=>{
  const f=await recording(t,{count:24}),last=f.lights.at(-1);
  last.color.setRGB(.75,.5,.25);last.intensity=2;f.renderer._advanceAnimationFrame();f.renderer.render(f.scene,f.camera);
  last.color.setRGB(.875,.625,.375);last.intensity=3;f.renderer._advanceAnimationFrame();f.renderer.render(f.scene,f.camera);
  for(const light of f.lights.slice(8))f.scene.remove(light);f.renderer.render(f.scene,f.camera);
  last.color.setRGB(99,99,99);last.intensity=99;
  await f.renderer.waitForGPU();const draws=f.device.snapshots.flat().filter(isColorDraw);
  assert.equal(draws.length,3);assert.deepEqual(draws.map(isNative),[true,true,false]);
  for(const value of [1.5,1,.5])assert.ok(submittedFloats(draws[0]).includes(value));
  for(const value of [2.625,1.875,1.125])assert.ok(submittedFloats(draws[1]).includes(value));
  assert.ok(!submittedFloats(draws[0]).includes(2.625));assert.ok(!submittedFloats(draws[1]).includes(9801));
  assert.equal(f.renderer._nodeCaptureBytes,0);assert.equal(f.renderer._materialCaptureValues,0);
});

test('many-light projection preserves layers, inherited visibility, disabled lighting and unlit companions',async t=>{
  const f=await recording(t),last=f.lights.at(-1),basic=new THREE.Mesh(f.object.geometry,new THREE.MeshBasicMaterial());
  basic.frustumCulled=false;last.add(basic);
  const hidden=new THREE.Group();hidden.visible=false;hidden.add(...lightsFor(new THREE.Scene(),12,{mixed:false}));f.scene.add(hidden);
  last.layers.set(1);f.renderer.render(f.scene,f.camera); // eight contributors; child still on layer zero
  last.layers.set(0);f.renderer.render(f.scene,f.camera);
  f.renderer.lighting.enabled=false;f.renderer.render(f.scene,f.camera);
  f.renderer.lighting.enabled=true;last.layers.set(2);
  await f.renderer.waitForGPU();
  const frames=f.device.snapshots.map(draws=>draws.filter(isColorDraw)).filter(draws=>draws.length);
  assert.deepEqual(frames.map(draws=>draws.length),[2,2,2]);
  assert.ok(frames[0].every(draw=>!isNative(draw)));
  assert.ok(frames[1].every(isNative));assert.ok(frames[2].every(isNative));
});

test('many-light native shadows retain late light identities without a second fixed shadow owner',async t=>{
  const f=await recording(t,{count:12});f.renderer.shadowMap.enabled=true;f.renderer.shadowMap.type=THREE.BasicShadowMap;
  f.object.receiveShadow=true;
  const caster=new THREE.Mesh(new THREE.BoxGeometry(.5,.5,.5),new THREE.MeshBasicMaterial());
  caster.castShadow=true;caster.frustumCulled=false;f.scene.add(caster);
  const shadowLights=[f.lights[2],f.lights[10]];
  for(const light of shadowLights){light.castShadow=true;light.shadow.mapSize.set(16,16);}
  f.renderer.render(f.scene,f.camera);await f.renderer.waitForGPU();
  const maps=shadowLights.map(light=>light.shadow.map);assert.ok(maps.every(Boolean));assert.notEqual(maps[0],maps[1]);
  const diagnostics=f.renderer._dispatcher.entry(f.scene).bridge.diagnostics;
  assert.equal(diagnostics.shadowBytes,0);assert.equal(diagnostics.shadowStats,null);
  const receivers=f.device.snapshots.flat().filter(draw=>isColorDraw(draw)&&/textureSampleCompare/.test(draw.pipeline.fragment.module.code));
  assert.equal(receivers.length,1);
  const depthViews=[...receivers[0].groups.values()].flatMap(({group})=>group.entries??[])
    .map(entry=>entry.resource).filter(resource=>resource?.texture?.format?.startsWith('depth'));
  assert.equal(new Set(depthViews.map(view=>view.texture)).size,2);
});
