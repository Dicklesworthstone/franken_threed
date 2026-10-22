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
  device.createTexture=desc=>{const texture={dimension:'2d',depthOrArrayLayers:1,sampleCount:1,...desc,
    width:desc.size.width??desc.size[0],height:desc.size.height??desc.size[1],
    createView(options={}){return {texture,...options};},destroy(){this.destroyed=true;}};return texture;};
  device.createSampler=desc=>({...desc});device.queue.writeTexture=()=>{};
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
    const {buffer,size}=group.entries[0].resource,bytes=draw.contents.get(buffer),offset=offsets[0]??0;
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
  f.material.colorNode=T.TSL.texture(new T.DataTexture(new Uint8Array([255,0,0,255]),1,1));
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
