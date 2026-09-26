/** Actual builder and relocated production background/scene owners. Pose/model
 * decoding, Three classes and lower GPU mesh/filter services are fixtures.
 * Temporary directories are retained for inspection; no source files removed. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {setup,THREE,events} from './three_background_test_fixture.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
const exportsSource=names=>names.split(' ').map(n=>`export function ${n}(){throw Error('unexpected boundary: ${n}');}`).join('\n');
async function fixture(){
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-background-package-')),toolkit=path.join(root,'toolkit');await fs.mkdir(toolkit);
  const original=new Map();
  for(const name of ['build_animation.mjs','three_scene.mjs','three_shadows.mjs','three_environment.mjs','three_background.mjs','animation_background.mjs']){
    const bytes=await fs.readFile(new URL('./'+name,import.meta.url));original.set(name,bytes);await fs.writeFile(path.join(toolkit,name),bytes);
  }
  const boundaries={
    'animation_gltf.mjs':'export const decodeGltfAnimation=model=>model;',
    'gltf_instancing.mjs':'export const expandGltfInstances=json=>({json,instanceCount:0});',
    'animation_runtime.mjs':`export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}
export const createAnimationPlayer=d=>({nodeCount:d.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}});`,
    'animation_controller.mjs':"import './animation_markers.mjs';\n"+exportsSource('createAnimationController'),
    'animation_markers.mjs':'export {};',
    'animation_deformer.mjs':exportsSource('createAnimationDeformer'),
    'animation_webgpu.mjs':exportsSource('createGpuAnimationDeformer'),
    'animation_scene.mjs':exportsSource('createGpuAnimationScene'),
    'animation_render.mjs':'export const createGpuAnimationRenderer=(d,o)=>d.testState.color(d,o);',
    'animation_render_bundles.mjs':'export {};',
    'gpu_buffer_geometry.mjs':exportsSource('createGpuBufferGeometry bufferGeometrySnapshot createGpuInstanceAttributes instanceAttributesSnapshot inspectInstanceAttributes'),
    'animation_shadow.mjs':'export const createGpuAnimationShadowMap=(d,o)=>d.testState.map(d,o);',
    'animation_shadow_receiver.mjs':'export {};',
    'animation_shadow_view.mjs':exportsSource('fitAnimationShadowView animationShadowWorldBounds'),
    'animation_scene_shadow.mjs':'export {};',
    'animation_lod.mjs':'export {};','animation_draw_order.mjs':'export {};','animation_bounds.mjs':'export {};',
    'three_textures.mjs':exportsSource('createGpuThreeTextures'),
    'three_deformation.mjs':exportsSource('hasThreeDeformation inspectThreeDeformation createGpuThreeDeformation createThreeDeformationBinding ThreeDeformationError')+'\nexport function updateGpuThreeDeformations(items){if(items.length)throw Error("unexpected deformation");}',
    'three_deformation_binding.mjs':'export {};',
    'three_canvas.mjs':exportsSource('createGpuThreeCanvas createGpuThreeHdrCanvas'),
    'gpu_canvas.mjs':exportsSource('createGpuCanvasTarget GpuCanvasError'),
    'gpu_canvas_renderer.mjs':exportsSource('createGpuCanvasRenderer'),
    'gpu_hdr_canvas.mjs':exportsSource('createGpuHdrCanvasRenderer GpuHdrCanvasError'),
    'gpu_render_target.mjs':exportsSource('createGpuRenderTarget GpuRenderTargetError'),
    'animation_output.mjs':'export {};',
    'animation_environment.mjs':'export const planAnimationEnvironment=()=>({textureBytes:256,uniformBytes:128});\nexport const createGpuAnimationEnvironment=(d,t,o)=>d.testState.filter(d,t,o);',
    'animation_environment_receiver.mjs':'export {};',
    'animation_hdr.mjs':exportsSource('decodeAnimationHdr'),
    'animation_environment_loader.mjs':exportsSource('loadGpuAnimationEnvironment'),
  };
  for(const [name,text] of Object.entries(boundaries))await fs.writeFile(path.join(toolkit,name),text);
  const entry=path.join(root,'input.gltf');await fs.writeFile(entry,JSON.stringify({asset:{version:'2.0'},nodes:[],ignoredChannels:[]}));
  const {buildAnimation}=await import(pathToFileURL(path.join(toolkit,'build_animation.mjs')));
  return {root,toolkit,entry,original,buildAnimation};
}
test('source background packages include exact production bytes and the complete static HDR helper dependency',async()=>{
  const h=await fixture(),out=path.join(h.root,'output'),m=h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true,background:true});
  for(const name of ['animation_background.mjs','three_background.mjs','three_environment.mjs','three_scene.mjs']){
    const bytes=await fs.readFile(path.join(out,name)),item=m.artifacts.find(a=>a.file===name);assert.deepEqual(bytes,h.original.get(name));
    assert.equal(item.bytes,bytes.length);assert.equal(item.sha256,hash(bytes));
  }
  assert.ok(m.emittedFiles.includes('animation_environment.mjs'));assert.equal(m.gpuEnvironment,undefined);
  assert.equal(m.emittedFiles.includes('animation_environment_receiver.mjs'),false);assert.equal(m.emittedFiles.includes('animation_hdr.mjs'),false);
  assert.ok(m.gpuBackground);assert.equal(m.accelerationClaim,false);
  const manifest=JSON.parse(await fs.readFile(path.join(out,'manifest.json'),'utf8'));assert.deepEqual(manifest.artifacts,m.artifacts);
});
test('relocated GPU entry renders the real source background without original toolkit or enabling environment lighting',async()=>{
  const h=await fixture(),out=path.join(h.root,'out'),moved=path.join(h.root,'deployed');
  h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true,background:true});await fs.rename(out,moved);await fs.rename(h.toolkit,path.join(h.root,'toolkit-unavailable'));
  const gpu=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs'))),s=setup();assert.equal(s.state.events.length,0);
  assert.equal(typeof gpu.createGpuThreeBackground,'function');assert.equal(typeof gpu.createGpuAnimationBackground,'function');
  assert.equal(gpu.createGpuThreeEnvironment,undefined);
  const owner=await gpu.createGpuThreeScene(s.device,s.scene,{three:THREE,background:{}});
  owner.render(s.camera,{colorView:{},depthView:{}});await owner.whenIdle();assert.equal(events(s,'background').length,1);
  assert.equal(events(s,'filter').length,0);assert.equal(owner.diagnostics.colorPasses,2);owner.dispose();
});
test('ordinary GPU background packages export the core without source or environment dependencies',async()=>{
  const h=await fixture(),out=path.join(h.root,'out'),m=h.buildAnimation(h.entry,out,{webgpu:true,background:true});
  assert.ok(m.emittedFiles.includes('animation_background.mjs'));for(const name of ['three_background.mjs','three_environment.mjs','animation_environment.mjs','three_scene.mjs'])assert.ok(!m.emittedFiles.includes(name));
  const gpu=await import(pathToFileURL(path.join(out,'gpu_playback.mjs'))),s=setup();
  const texture=s.device.createTexture({size:[4,4,6],dimension:'2d',format:'rgba16float',usage:4});
  const bg=await gpu.createGpuAnimationBackground(s.device,texture,{mapping:'cube'});
  bg.render({colorView:{},directionFromClip:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]});await bg.whenIdle();bg.dispose();assert.equal(texture.destroyed,0);
});
test('combined lighting and background package deduplicates shared files and keeps both exported features usable',async()=>{
  const h=await fixture(),out=path.join(h.root,'out'),m=h.buildAnimation(h.entry,out,{webgpu:true,threeScene:true,background:true,environment:true,hdr:true});
  assert.equal(new Set(m.emittedFiles).size,m.emittedFiles.length);assert.ok(m.gpuEnvironment);assert.ok(m.gpuBackground);
  const gpu=await import(pathToFileURL(path.join(out,'gpu_playback.mjs'))),s=setup();s.scene.environment=s.texture;
  const owner=await gpu.createGpuThreeScene(s.device,s.scene,{three:THREE,background:{},environment:{}});
  owner.render(s.camera,{colorView:{},depthView:{}});assert.equal(events(s,'filter').length,1);assert.equal(events(s,'background').length,1);await owner.whenIdle();owner.dispose();
  assert.equal(typeof gpu.createGpuThreeEnvironment,'function');assert.equal(typeof gpu.loadGpuAnimationEnvironment,'function');
});
test('omitted and explicit-disabled backgrounds preserve every emitted byte in existing package profiles',async()=>{
  const h=await fixture();let index=0;
  for(const profile of [{},{webgpu:true},{webgpu:true,threeScene:true},{webgpu:true,environment:true},{webgpu:true,threeScene:true,environment:true}]){
    const a=path.join(h.root,'a'+index),b=path.join(h.root,'b'+index++),implicit=h.buildAnimation(h.entry,a,profile),explicit=h.buildAnimation(h.entry,b,{...profile,background:false});
    assert.deepEqual(implicit.artifacts,explicit.artifacts);assert.equal(implicit.gpuBackground,undefined);
    assert.ok(!implicit.emittedFiles.includes('three_background.mjs'));assert.ok(!implicit.emittedFiles.includes('animation_background.mjs'));
    for(const name of implicit.emittedFiles)assert.deepEqual(await fs.readFile(path.join(a,name)),await fs.readFile(path.join(b,name)));
  }
});
test('background output budget includes every dependency before destination publication',async()=>{
  const h=await fixture(),out=path.join(h.root,'out'),profile={webgpu:true,threeScene:true,background:true},full=h.buildAnimation(h.entry,out,profile);
  const short=path.join(h.root,'too-small');assert.throws(()=>h.buildAnimation(h.entry,short,{...profile,maxBytes:full.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  await assert.rejects(fs.stat(short),{code:'ENOENT'});assert.ok(full.outputBytes>100000);
});
test('invalid background switches fail before reading source or creating output',async()=>{
  const h=await fixture();for(const profile of [{background:true},{background:{}},{webgpu:true,background:1}]){
    const out=path.join(h.root,'invalid');assert.throws(()=>h.buildAnimation(path.join(h.root,'missing'),out,profile),/background must be boolean and requires webgpu:true/);
    await assert.rejects(fs.stat(out),{code:'ENOENT'});
  }
});
