/** Actual package builder plus emitted source-scene/shadow modules. Decoder,
 * pose and native GPU boundaries are explicit fixtures; no GPU pixels claimed.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const exportsSource=names=>names.split(' ').map(n=>`export function ${n}(){throw Error('unexpected boundary: ${n}');}`).join('\n');
async function fixture(t){
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-source-shadow-package-'));
  t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const toolkit=path.join(root,'toolkit');await fs.mkdir(toolkit);
  const original=new Map();
  for(const name of ['build_animation.mjs','three_scene.mjs','three_shadows.mjs']){
    const bytes=await fs.readFile(new URL('./'+name,import.meta.url));original.set(name,bytes);
    await fs.writeFile(path.join(toolkit,name),bytes);
  }
  const boundaries={
    'animation_gltf.mjs':'export const decodeGltfAnimation=model=>model;',
    'gltf_instancing.mjs':'export const expandGltfInstances=json=>({json,instanceCount:0});',
    'animation_runtime.mjs':`export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}
export const createAnimationPlayer=d=>({nodeCount:d.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}});`,
    'animation_controller.mjs':exportsSource('createAnimationController'),
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
  };
  for(const [name,text] of Object.entries(boundaries))await fs.writeFile(path.join(toolkit,name),text);
  const entry=path.join(root,'input.gltf');await fs.writeFile(entry,JSON.stringify({asset:{version:'2.0'},nodes:[],ignoredChannels:[]}));
  const {buildAnimation}=await import(pathToFileURL(path.join(toolkit,'build_animation.mjs')));
  return {root,toolkit,entry,original,buildAnimation};
}
function sourceModule(){
  class Object3D{children=[];onBeforeRender(){}onAfterRender(){}}
  class Scene extends Object3D{fog=null;environment=null;background=null;overrideMaterial=null;}
  class Placeholder{}
  return {REVISION:'186',Object3D,Scene,Mesh:Placeholder,Matrix4:Placeholder,Frustum:Placeholder,Vector3:Placeholder,
    MeshBasicMaterial:Placeholder,MeshLambertMaterial:Placeholder,MeshPhongMaterial:Placeholder,
    MeshToonMaterial:Placeholder,MeshStandardMaterial:Placeholder};
}
test('source packages include exact shadow/scene bytes in the bounded artifact manifest',async t=>{
  const h=await fixture(t),out=path.join(h.root,'out');
  const result=h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true});
  for(const name of ['three_scene.mjs','three_shadows.mjs']){
    const bytes=await fs.readFile(path.join(out,name));assert.deepEqual(bytes,h.original.get(name));
    const artifact=result.artifacts.find(a=>a.file===name);assert.equal(artifact.bytes,bytes.length);assert.equal(artifact.sha256,hash(bytes));
  }
  assert.ok(result.emittedFiles.includes('animation_shadow.mjs'));
  assert.equal(result.accelerationClaim,false);
  const manifest=JSON.parse(await fs.readFile(path.join(out,'manifest.json'),'utf8'));
  assert.deepEqual(manifest.artifacts,result.artifacts);
});
test('relocated GPU entry activates the real lazy source-shadow module without toolkit access',async t=>{
  const h=await fixture(t),out=path.join(h.root,'out'),moved=path.join(h.root,'deployed');
  h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true});await fs.rename(out,moved);
  // Physically remove the original tool location from resolution, not just cwd.
  await fs.rename(h.toolkit,path.join(h.root,'toolkit-unavailable'));
  const module=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')));
  const three=sourceModule(),device={allocations:0};assert.equal(device.allocations,0);
  const scene=await module.createGpuThreeScene(device,new three.Scene(),{three,shadow:{}});
  assert.equal(device.allocations,1);assert.equal(scene.diagnostics.shadowBytes,0);
  assert.equal(scene.diagnostics.colorPasses,0);await scene.whenIdle();scene.dispose();
  // Exercise the actual relocated composition helper too, not a stub export.
  const {withThreeShadowReceivers}=await import(pathToFileURL(path.join(moved,'three_shadows.mjs')));
  const frames=[],renderer={disposed:false,drawCallCount:1,render:f=>frames.push(f)};
  withThreeShadowReceivers(renderer).render({draws:[{receiveShadow:true},{receiveShadow:false}],shadow:{map:{}},loadOp:'clear'});
  assert.equal(frames.length,2);assert.equal(frames[1].loadOp,'load');assert.equal(frames[1].depthLoadOp,'load');assert.equal(frames[1].shadow,null);
});
test('CPU and ordinary GPU packages remain independent of source-shadow modules',async t=>{
  const h=await fixture(t);
  for(const webgpu of [false,true]){
    const a=path.join(h.root,'a'+webgpu),b=path.join(h.root,'b'+webgpu);
    const implicit=h.buildAnimation(h.entry,a,{webgpu}),explicit=h.buildAnimation(h.entry,b,{webgpu,threeScene:false});
    assert.deepEqual(implicit.artifacts,explicit.artifacts);
    assert.ok(!implicit.emittedFiles.includes('three_shadows.mjs'));assert.ok(!implicit.emittedFiles.includes('three_scene.mjs'));
    for(const name of implicit.emittedFiles)assert.deepEqual(await fs.readFile(path.join(a,name)),await fs.readFile(path.join(b,name)));
  }
});
test('package output budget includes lazy source-shadow bytes before destination publication',async t=>{
  const h=await fixture(t),out=path.join(h.root,'out');
  const full=h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true});
  const short=path.join(h.root,'too-small');
  assert.throws(()=>h.buildAnimation(h.entry,short,{webgpu:true,threeScene:true,maxBytes:full.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  await assert.rejects(fs.stat(short),{code:'ENOENT'});
});
