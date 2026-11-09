/** Retained r186 public PMREM on the WebGL-authored owned renderer. Recording
 * checks accepted bytes, source pass order and lifetimes; browser tests rasterize. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass, createPMREMGeneratorClass} from './three_renderer.mjs';
import {createGpuThreeTextures, hasThreeWebGLTextureResidency} from './three_textures.mjs';
import {createThreePassSnapshots} from './three_pass_snapshot.mjs';
import {threeProgramSources} from './three_program.mjs';
import {compileEsslProgram} from './essl_wgsl.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const Renderer=createWebGLRendererClass(T,{maxDraws:64,maxBindings:64}), Generator=createPMREMGeneratorClass(T);
function panorama(width=64,value=0x3800){
  const texture=new T.DataTexture(new Uint16Array(width*width*2).fill(value),width,width/2,T.RGBAFormat,T.HalfFloatType);
  texture.mapping=T.EquirectangularReflectionMapping;texture.minFilter=texture.magFilter=T.LinearFilter;texture.needsUpdate=true;return texture;
}
function fixture(t){
  const device=geometryDevice();Object.assign(device.limits,{maxTextureDimension2D:8192,maxBindingsPerBindGroup:1000});
  device.textures=[];device.textureWrites=[];
  device.createTexture=descriptor=>{const texture={dimension:'2d',depthOrArrayLayers:1,sampleCount:1,...descriptor,
    width:descriptor.size.width??descriptor.size[0],height:descriptor.size.height??descriptor.size[1],
    createView(options={}){return {texture,...options};},destroy(){this.destroyed=true;}};device.textures.push(texture);return texture;};
  device.createRenderPipeline=descriptor=>({...descriptor,getBindGroupLayout:()=>({})});device.createSampler=descriptor=>({...descriptor});
  device.queue.writeTexture=(destination,data)=>{const bytes=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength).slice();
    destination.texture.pixels=bytes;device.textureWrites.push({texture:destination.texture,bytes});};
  const submit=device.queue.submit;
  device.queue.submit=commands=>{for(const passes of commands)for(const pass of passes)for(const draw of pass.draws)
    for(const {group} of draw.groups.values())for(const entry of group.entries??[])if(entry.resource?.texture)
      assert.notEqual(entry.resource.texture.destroyed,true,'submitted sampled view must remain live');submit(commands);};
  let configuration;
  const canvas={width:32,height:32,style:{},getContext:()=>({configure(value){configuration=value;},unconfigure(){},
    getCurrentTexture(){return device.createTexture({size:[32,32],format:configuration.format,usage:16});}})};
  const renderer=new Renderer({canvas,device});t.after(()=>renderer.dispose());return {renderer,device};
}
function packet(draw){const {group,offsets}=draw.groups.get(0),resource=group.entries[0].resource,bytes=draw.contents.get(resource.buffer);
  return new DataView(bytes.buffer,bytes.byteOffset+(resource.offset??0)+(offsets[0]??0),resource.size);}
function oracle(texture){
  let target=null;const draws=[];let reflection;
  const host={autoClear:true,xr:{enabled:false},getRenderTarget:()=>target,getActiveCubeFace:()=>0,getActiveMipmapLevel:()=>0,
    setRenderTarget(value){target=value;},render(mesh){const m=mesh.material;if(m.name==='PMREMGGXConvolution'){
      if(!reflection){const source=threeProgramSources(T,m,mesh,{toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace});
        reflection=compileEsslProgram(source.vertex,source.fragment).reflection;}
      draws.push([Math.fround(m.uniforms.roughness.value),Math.fround(m.uniforms.mipInt.value)]);
    }}};
  const generator=new T.PMREMGenerator(host),result=generator.fromEquirectangular(texture);generator.dispose();result.dispose();
  return {draws,reflection};
}
const drain=async renderer=>{await renderer.waitForGPU();await Promise.resolve();};
function mapped(texture){const root=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial({map:texture}));const camera=new T.PerspectiveCamera();camera.position.z=2;return {root,camera};}

test('real CubeUV output and each queued GGX uniform match the retained source helper',async t=>{
  const {renderer,device}=fixture(t);await renderer.init();const input=panorama(),expected=oracle(input),generator=new Generator(renderer);
  const selected=new T.WebGLRenderTarget(8,8);renderer.setRenderTarget(selected);renderer.xr.enabled=true;
  const output=generator.fromEquirectangular(input);t.after(()=>{selected.dispose();output.dispose();input.dispose();generator.dispose();});
  assert.equal(output.isWebGLRenderTarget,true);assert.deepEqual([output.width,output.height],[336,64]);
  assert.equal(output.texture.mapping,T.CubeUVReflectionMapping);assert.equal(output.texture.isF3DPMREMSource,undefined);
  assert.equal(renderer.getRenderTarget(),selected);assert.equal(renderer.xr.enabled,true);renderer.xr.enabled=false;
  await drain(renderer);
  const fields=['roughness','mipInt'].map(name=>expected.reflection.uniforms.find(value=>value.name===name).offset);
  const actual=device.snapshots.flat().filter(draw=>draw.pipeline.fragment.module.code.includes('radicalInverse_VdC'))
    .map(draw=>fields.map(offset=>packet(draw).getFloat32(offset,true)));
  assert.deepEqual(actual,expected.draws);assert.equal(actual.length,12);
  assert.equal(generator.fromEquirectangular(input,output),output);await drain(renderer);
});

test('a generator made before init preserves early admission and later creates real targets',async t=>{
  const {renderer}=fixture(t),generator=new Generator(renderer),input=panorama();
  const early=generator.fromEquirectangular(input);assert.equal(early.isRenderTarget,true);assert.equal(early.isWebGLRenderTarget,undefined);
  await renderer.init();const output=generator.fromEquirectangular(input);assert.equal(output.isWebGLRenderTarget,true);
  await drain(renderer);generator.dispose();early.dispose();output.dispose();input.dispose();
});

test('queued bakes retain unmarked history and independent marked bytes after source and generator disposal',async t=>{
  const {renderer,device}=fixture(t);await renderer.init();const input=panorama(),generator=new Generator(renderer),outputs=[];
  outputs.push(generator.fromEquirectangular(input));input.image.data.fill(0x3c00);
  outputs.push(generator.fromEquirectangular(input));input.image.data.fill(0x3400);input.needsUpdate=true;
  outputs.push(generator.fromEquirectangular(input));input.image.data.fill(0);input.dispose();
  let callbacks=0;input.onUpdate=()=>{callbacks++;};generator.dispose();await drain(renderer);
  assert.deepEqual(device.textureWrites.map(write=>new Uint16Array(write.bytes.buffer)[0]),[0x3800,0x3400]);
  assert.equal(callbacks,0);assert.equal(renderer._sampledTextureInputs.diagnostics.snapshotBytes,0);
  assert.equal(renderer._sampledTextureInputs.diagnostics.versions,0);assert.equal(renderer._passSnapshots.rootCount,0);
  for(const output of outputs)output.dispose();
});

test('source disposal starts a fresh upload lifetime while an earlier detached bake stays queued',async t=>{
  const {renderer,device}=fixture(t);await renderer.init();const input=panorama(),generator=new Generator(renderer);
  const first=generator.fromEquirectangular(input);input.dispose();input.image.data.fill(0x3a00);
  const second=generator.fromEquirectangular(input);input.dispose();generator.dispose();await drain(renderer);
  assert.deepEqual(device.textureWrites.map(write=>new Uint16Array(write.bytes.buffer)[0]),[0x3800,0x3a00]);
  assert.equal(renderer._sampledTextureInputs.diagnostics.snapshotBytes,0);first.dispose();second.dispose();
});

test('a warm detached read acquires a live descriptor after the original texture is disposed',async t=>{
  const {renderer,device}=fixture(t);await renderer.init();const input=panorama(),generator=new Generator(renderer);
  const first=generator.fromEquirectangular(input);await drain(renderer);input.image.data.fill(0x3c00);
  const second=generator.fromEquirectangular(input);input.dispose();generator.dispose();await drain(renderer);
  assert.ok(device.textureWrites.every(write=>new Uint16Array(write.bytes.buffer)[0]===0x3800));
  assert.equal(renderer._sampledTextureInputs.diagnostics.versions,0);first.dispose();second.dispose();
});

test('ordinary maps and Source aliases keep the baked source upload history',async t=>{
  const {renderer,device}=fixture(t);await renderer.init();const input=panorama(),alias=input.clone(),generator=new Generator(renderer);
  const output=generator.fromEquirectangular(input);await drain(renderer);input.image.data.fill(0x3400);
  const {root,camera}=mapped(alias);renderer.render(root,camera);await drain(renderer);
  assert.ok(device.textureWrites.every(write=>new Uint16Array(write.bytes.buffer)[0]===0x3800));
  const oldWrites=device.textureWrites.length;
  const unused=input.clone();root.material.map=input;renderer.render(root,camera);await drain(renderer);
  assert.equal(device.textureWrites.length,oldWrites,'unconsumed alias version bump is not an upload request');
  alias.needsUpdate=true;root.material.map=alias;renderer.render(root,camera);await drain(renderer);
  assert.equal(device.textureWrites.length,oldWrites+1);root.material.map=input;renderer.render(root,camera);await drain(renderer);
  assert.equal(device.textureWrites.length,oldWrites+1,'a warm earlier Texture version sees its alias shared upload');
  assert.equal(new Uint16Array(device.textureWrites.at(-1).bytes.buffer)[0],0x3400);
  generator.dispose();output.dispose();input.dispose();alias.dispose();unused.dispose();
});

test('preexisting scene ownership and its cold Source aliases refuse before accepting a fresh history',async t=>{
  const {renderer,device}=fixture(t);await renderer.init();const input=panorama(),alias=input.clone();
  const owner=createGpuThreeTextures(device,{three:T,webglVersioning:true});owner.prepare([input,alias]);
  t.after(()=>owner.dispose());input.image.data.fill(0);
  const generator=new Generator(renderer);assert.equal(hasThreeWebGLTextureResidency(device,alias),true);
  assert.throws(()=>generator.fromEquirectangular(alias),/separate scene texture owner/);
  input.dispose();assert.equal(hasThreeWebGLTextureResidency(device,input),true,'remaining alias still owns the Source');
  alias.dispose();assert.equal(hasThreeWebGLTextureResidency(device,input),false);
  const output=generator.fromEquirectangular(input);await drain(renderer);generator.dispose();output.dispose();input.dispose();
});

test('generator disposal retires only its private filter roots and bridges while output remains usable',async t=>{
  const {renderer}=fixture(t);await renderer.init();const input=panorama();let output=null;
  for(let cycle=0;cycle<12;cycle++){
    const generator=new Generator(renderer),fresh=generator.fromEquirectangular(input);await drain(renderer);
    if(output){generator.fromEquirectangular(input,output);await drain(renderer);fresh.dispose();}else output=fresh;
    const destination=renderer._targets.capture(output);generator.dispose();await drain(renderer);
    assert.equal(renderer._passSnapshots.rootCount,0);assert.equal(destination._dispatcher.diagnostics.scenes,0);
    assert.equal(destination._wanted.size,0);assert.equal(destination.alive,true);
  }
  output.dispose();input.dispose();
});

test('36-vertex capture is limited to the source PMREM plane shape and explicit helper scope',()=>{
  const owner=createThreePassSnapshots(T),camera=new T.OrthographicCamera();
  const root=new T.Mesh(new T.BufferGeometry(),new T.ShaderMaterial());
  root.geometry.setAttribute('position',new T.BufferAttribute(new Float32Array(108),3));
  assert.equal(owner.capturePMREM(root,camera),null);
  root.geometry.setAttribute('outputDirection',new T.BufferAttribute(new Float32Array(108),3));
  assert.equal(owner.capture(root,camera),null);
  const captured=owner.capturePMREM(root,camera);assert.ok(captured);captured.release();owner.retire(root);assert.equal(owner.rootCount,0);owner.dispose();
});

test('detached callback freezing leaves later marked ordinary uploads live',async t=>{
  const {renderer}=fixture(t);await renderer.init();const owner=renderer._getSampledTextureInputs(),input=panorama();
  const detached=owner.captureDetachedBinding(input,{webglVersioning:true});let calls=0;input.onUpdate=()=>{calls++;};
  owner.resolve(detached);assert.equal(calls,0);owner.release(detached);
  input.image.data.fill(0x3400);input.needsUpdate=true;
  const next=owner.captureBinding({texture:input,isSampledTexture:true});owner.resolve(next);owner.release(next);
  assert.equal(calls,1);input.dispose();assert.equal(owner.diagnostics.snapshotBytes,0);
});
