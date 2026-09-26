/** Actual builder and relocated source environment/scene modules. Pose decoding,
 * GPU allocation/filtering and other runtime components are explicit fixtures.
 * This tests deployable dependency closure, not native IBL or GPU pixels.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
const hash=b=>createHash('sha256').update(b).digest('hex');
const exportsSource=names=>names.split(' ').map(n=>`export function ${n}(){throw Error('unexpected boundary: ${n}');}`).join('\n');
async function fixture(){
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-source-environment-package-'));
  const toolkit=path.join(root,'toolkit');await fs.mkdir(toolkit);
  const original=new Map();
  for(const name of ['build_animation.mjs','three_scene.mjs','three_environment.mjs','three_shadows.mjs']){
    const bytes=await fs.readFile(new URL('./'+name,import.meta.url));original.set(name,bytes);
    await fs.writeFile(path.join(toolkit,name),bytes);
  }
  const boundaries={
    'animation_gltf.mjs':'export const decodeGltfAnimation=model=>model;',
    'gltf_instancing.mjs':'export const expandGltfInstances=json=>({json,instanceCount:0});',
    'animation_runtime.mjs':`export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}
export const createAnimationPlayer=d=>({nodeCount:d.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}});`,
    'animation_controller.mjs':"import {animationMarkerEventLimit,createAnimationMarkerTrack} from './animation_markers.mjs';\n"+exportsSource('createAnimationController'),
    'animation_markers.mjs':exportsSource('animationMarkerEventLimit createAnimationMarkerTrack'),
    'animation_deformer.mjs':exportsSource('createAnimationDeformer'),
    'animation_webgpu.mjs':exportsSource('createGpuAnimationDeformer'),
    'animation_scene.mjs':exportsSource('createGpuAnimationScene'),
    'animation_render.mjs':`export async function createGpuAnimationRenderer(device){device.allocations++;
return {allocatedBytes:0,drawCount:0,drawCallCount:0,disposed:false,failed:false,whenIdle:async()=>{},dispose(){this.disposed=true;}};}`,
    'animation_render_bundles.mjs':'export {};',
    'gpu_buffer_geometry.mjs':exportsSource('createGpuBufferGeometry bufferGeometrySnapshot createGpuInstanceAttributes instanceAttributesSnapshot inspectInstanceAttributes'),
    'animation_shadow.mjs':exportsSource('createGpuAnimationShadowMap'),
    'animation_shadow_receiver.mjs':'export {};',
    'animation_shadow_view.mjs':exportsSource('fitAnimationShadowView animationShadowWorldBounds'),
    'animation_scene_shadow.mjs':'export {};',
    'animation_lod.mjs':'export {};','animation_draw_order.mjs':'export {};','animation_bounds.mjs':'export {};',
    'three_textures.mjs':exportsSource('createGpuThreeTextures'),
    'three_deformation.mjs':exportsSource('hasThreeDeformation inspectThreeDeformation createGpuThreeDeformation updateGpuThreeDeformations createThreeDeformationBinding ThreeDeformationError'),
    'three_deformation_binding.mjs':'export {};',
    'three_canvas.mjs':exportsSource('createGpuThreeCanvas createGpuThreeHdrCanvas'),
    'gpu_canvas.mjs':exportsSource('createGpuCanvasTarget GpuCanvasError'),
    'gpu_canvas_renderer.mjs':exportsSource('createGpuCanvasRenderer'),
    'gpu_hdr_canvas.mjs':exportsSource('createGpuHdrCanvasRenderer GpuHdrCanvasError'),
    'gpu_render_target.mjs':exportsSource('createGpuRenderTarget GpuRenderTargetError'),
    'animation_output.mjs':'export {};',
    'animation_environment_receiver.mjs':'export {};',
    'animation_environment.mjs':`export const planAnimationEnvironment=()=>({textureBytes:128,uniformBytes:256});
export async function createGpuAnimationEnvironment(device){device.filters++;
return {textureBytes:128,disposed:false,sample(d){if(d!==device)throw Error('foreign device');return {};},whenIdle:async()=>{},dispose(){this.disposed=true;}};}`,
  };
  for(const [name,text] of Object.entries(boundaries))await fs.writeFile(path.join(toolkit,name),text);
  const entry=path.join(root,'input.gltf');await fs.writeFile(entry,JSON.stringify({asset:{version:'2.0'},nodes:[],ignoredChannels:[]}));
  const {buildAnimation}=await import(pathToFileURL(path.join(toolkit,'build_animation.mjs')));
  return {root,toolkit,entry,original,buildAnimation};
}
function sourceModule(){
  class Object3D{children=[];onBeforeRender(){}onAfterRender(){}}
  class Scene extends Object3D{fog=null;environment=null;background=null;overrideMaterial=null;
    environmentIntensity=1;environmentRotation={isEuler:true,x:0,y:0,z:0,order:'XYZ'};}
  class Matrix4{makeRotationFromEuler(){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];return this;}}
  class DataTexture{
    constructor(){this.image={width:4,height:2,data:new Uint16Array(32).fill(0x3c00)};
      Object.assign(this,{type:1,format:3,internalFormat:null,mapping:4,colorSpace:'linear',flipY:true,premultiplyAlpha:false,
        unpackAlignment:1,onUpdate:null,updateRanges:[],mipmaps:[],version:1,source:{data:this.image,dataReady:true,version:1}});this.listeners=new Set();}
    addEventListener(k,f){this.listeners.add(f);}removeEventListener(k,f){this.listeners.delete(f);}
  }
  class Placeholder{}
  return {REVISION:'186',Object3D,Scene,DataTexture,HalfFloatType:1,FloatType:2,RGBAFormat:3,
    EquirectangularReflectionMapping:4,LinearSRGBColorSpace:'linear',NoColorSpace:'',Mesh:Placeholder,Matrix4,Frustum:Placeholder,Vector3:Placeholder,
    MeshBasicMaterial:Placeholder,MeshLambertMaterial:Placeholder,MeshPhongMaterial:Placeholder,
    MeshToonMaterial:Placeholder,MeshStandardMaterial:Placeholder};
}
test('combined source/IBL packages contain exact owner bytes, exports and artifact accounting',async()=>{
  const h=await fixture(),out=path.join(h.root,'out');
  const result=h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true,environment:true});
  for(const name of ['three_scene.mjs','three_environment.mjs','three_shadows.mjs']){
    const bytes=await fs.readFile(path.join(out,name));assert.deepEqual(bytes,h.original.get(name));
    const artifact=result.artifacts.find(a=>a.file===name);assert.equal(artifact.bytes,bytes.length);assert.equal(artifact.sha256,hash(bytes));
  }
  for(const name of ['animation_environment.mjs','animation_environment_receiver.mjs'])assert.ok(result.emittedFiles.includes(name));
  const entry=await fs.readFile(path.join(out,'gpu_playback.mjs'),'utf8');
  assert.match(entry,/export \{createGpuThreeEnvironment,ThreeEnvironmentError\} from '.\/three_environment.mjs'/);
  assert.ok(result.emittedFiles.includes('animation_markers.mjs'));
  assert.equal(result.accelerationClaim,false);
});
test('relocated GPU entry prepares real source HDR ownership without the original toolkit',async()=>{
  const h=await fixture(),out=path.join(h.root,'out'),moved=path.join(h.root,'deployed');
  h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true,environment:true});await fs.rename(out,moved);
  await fs.rename(h.toolkit,path.join(h.root,'toolkit-unavailable'));
  const module=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')));
  const three=sourceModule(),source=new three.Scene();source.environment=new three.DataTexture();
  const device={allocations:0,filters:0,uploads:0,limits:{maxTextureDimension2D:4096,minUniformBufferOffsetAlignment:256},
    lost:new Promise(()=>{}),pushErrorScope(){},async popErrorScope(){return null;},createTexture(){return {destroy(){}};},queue:{writeTexture(){device.uploads++;}}};
  assert.equal(device.allocations,0);assert.equal(typeof module.createGpuThreeEnvironment,'function');
  const scene=await module.createGpuThreeScene(device,source,{three,environment:{}});
  assert.equal(device.allocations,1);assert.equal(device.filters,1);assert.equal(device.uploads,1);
  assert.equal(scene.diagnostics.environmentBytes,128);await scene.whenIdle();scene.dispose();
  assert.equal(source.environment.listeners.size,0);assert.equal(source.environment.image.data.length,32);
});
test('CPU, ordinary GPU, source-only and IBL-only output omit the source environment dependency',async()=>{
  const h=await fixture();
  for(const [i,options] of [{},{webgpu:true},{webgpu:true,threeScene:true},{webgpu:true,environment:true}].entries()){
    const a=path.join(h.root,'a'+i),b=path.join(h.root,'b'+i);
    const implicit=h.buildAnimation(h.entry,a,options),explicit=h.buildAnimation(h.entry,b,{threeScene:false,environment:false,...options});
    assert.ok(!implicit.emittedFiles.includes('three_environment.mjs'));assert.deepEqual(implicit.artifacts,explicit.artifacts);
    for(const name of implicit.emittedFiles)assert.deepEqual(await fs.readFile(path.join(a,name)),await fs.readFile(path.join(b,name)));
  }
});
test('source environment bytes count toward output budget before any destination publication',async()=>{
  const h=await fixture(),options={webgpu:true,threeScene:true,environment:true};
  const full=h.buildAnimation(h.entry,path.join(h.root,'out'),options),short=path.join(h.root,'too-small');
  assert.throws(()=>h.buildAnimation(h.entry,short,{...options,maxBytes:full.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  await assert.rejects(fs.stat(short),{code:'ENOENT'});
});
