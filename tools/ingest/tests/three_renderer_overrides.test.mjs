/** Actual pinned RenderPass/Three objects and production public renderer,
 * dispatcher, scene/compiler and target owner. Only GPU execution is recorded. */
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createWebGLRendererClass} from '../three_renderer.mjs';
import {geometryDevice} from '../fixtures/gpu_geometry_device.mjs';
import {device as textureDevice, deferred} from './volume_texture_test_fixture.mjs';
const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const T=await import(pathToFileURL(path.join(root,'build/three.module.js')));
const {RenderPass}=await import(pathToFileURL(path.join(root,'examples/jsm/postprocessing/RenderPass.js')));
const Renderer=createWebGLRendererClass(T);
const shader=name=>Object.assign(new T.ShaderMaterial({
  vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader:'void main(){gl_FragColor=vec4(1.0);}',
}),{name});
function fixture(){
  const d=geometryDevice(),t=textureDevice();Object.assign(d.limits,t.limits);
  Object.assign(d,{features:t.features,textures:t.textures,createSampler:t.createSampler,createRenderPipeline:t.createRenderPipeline});
  d.createTexture=desc=>{
    const size=Array.isArray(desc.size)?desc.size:[desc.size.width,desc.size.height,desc.size.depthOrArrayLayers??1];
    const texture=t.createTexture({...desc,size});
    return Object.assign(texture,{format:desc.format,usage:desc.usage,sampleCount:desc.sampleCount??1,depthOrArrayLayers:size[2]??1});
  };
  d.queue.writeTexture=t.queue.writeTexture;
  let config;
  const canvas={width:16,height:16,style:{},getContext:()=>context};
  const context={configure(value){config=value;},unconfigure(){},getCurrentTexture(){
    return {format:config.format,usage:16,width:16,height:16,dimension:'2d',depthOrArrayLayers:1,sampleCount:1,createView(){return {screen:true};}};
  }};
  const renderer=new Renderer({canvas,device:d}),scene=new T.Scene(),camera=new T.PerspectiveCamera(50,1,.1,20);camera.position.z=4;
  const original=shader('original'),mesh=new T.Mesh(new T.PlaneGeometry(),original),calls=[];scene.add(mesh);
  mesh.onBeforeRender=(r,s,c,g,m)=>calls.push({renderer:r,scene:s,camera:c,geometry:g,material:m,sourceOverride:scene.overrideMaterial});
  const a=new T.RenderTarget(8,8),b=new T.RenderTarget(8,8);
  return {d,renderer,scene,camera,original,mesh,calls,a,b};
}
const flush=async r=>{while(r._drain)await r._drain;};
const materials=f=>f.calls.map(c=>c.material);

for(const screen of [false,true])test(`queued overrides survive immediate source restoration (${screen?'canvas':'offscreen'})`,async()=>{
  const f=fixture(),{renderer:r,scene,camera,a,b}=f,first=shader('first'),second=shader('second');
  try{
    if(!screen)r.setRenderTarget(a);scene.overrideMaterial=first;const one=r.renderAsync(scene,camera);
    if(!screen)r.setRenderTarget(b);scene.overrideMaterial=second;const two=r.renderAsync(scene,camera);
    scene.overrideMaterial=null;const three=r.renderAsync(scene,camera);
    await Promise.all([one,two,three]);await flush(r);
    assert.deepEqual(materials(f),[first,second,f.original]);assert.equal(scene.overrideMaterial,null);
    assert.ok(f.calls.every(c=>c.scene===scene&&c.camera===camera&&c.geometry===f.mesh.geometry&&c.renderer===r));
    assert.ok(f.calls.every(c=>c.sourceOverride===null),'no temporary writes to source Scene');
    assert.equal(f.mesh.material,f.original);
  }finally{r.dispose();}
});

for(const kind of ['MeshDepthMaterial','MeshNormalMaterial','ShaderMaterial'])test(`unmodified pinned RenderPass selects ${kind} then restores the scene`,async()=>{
  const f=fixture(),{renderer:r,scene,camera,a,b}=f;
  const m=kind==='ShaderMaterial'?shader('pass'):new T[kind]();
  const pass=new RenderPass(scene,camera,m,0x112233,.5),beauty=new RenderPass(scene,camera);
  try{
    await r.init();r.setClearColor(0x010203,.75);const before=r.getClearColor(new T.Color()).clone();
    pass.render(r,b,a);assert.equal(scene.overrideMaterial,null);assert.equal(r.autoClear,true);
    assert.equal(r.getClearAlpha(),.75);assert.ok(r.getClearColor(new T.Color()).equals(before));
    beauty.render(r,a,b);await flush(r);
    assert.deepEqual(materials(f),[m,f.original]);assert.equal(r.getRenderTarget(),b);
    const draws=f.d.submissions.flatMap(commands=>commands.flat()).filter(p=>p.draws.length);
    assert.equal(draws.length,2);assert.notEqual(draws[0].desc.colorAttachments[0].view.texture,draws[1].desc.colorAttachments[0].view.texture);
    assert.equal(draws[0].desc.colorAttachments[0].loadOp,'load','explicit RenderPass clear precedes drawing');
  }finally{r.dispose();}
});

test('compileAsync captures the override before initialization and later reuse stays synchronous',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,m=shader('prepared');
  try{
    r.setRenderTarget(a);scene.overrideMaterial=m;const prepared=r.compileAsync(scene,camera);scene.overrideMaterial=null;
    await prepared;assert.equal(f.calls.length,0);
    scene.overrideMaterial=m;const before=r.info.f3d.deferredRenders;r.render(scene,camera);scene.overrideMaterial=null;
    assert.deepEqual(materials(f),[m]);assert.equal(r.info.f3d.deferredRenders,before);await flush(r);
  }finally{r.dispose();}
});

test('pipeline awaits do not expose captured overrides on the application scene',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,gate=deferred(),entered=deferred(),one=shader('one'),two=shader('two');
  const compile=f.d.createRenderPipelineAsync;
  try{
    await r.init();r.setRenderTarget(a);
    f.d.createRenderPipelineAsync=async desc=>{entered.resolve();await gate.promise;return compile(desc);};
    scene.overrideMaterial=one;r.render(scene,camera);scene.overrideMaterial=two;
    await entered.promise;assert.equal(scene.overrideMaterial,two);assert.equal(f.calls.length,0);
    r.render(scene,camera);scene.overrideMaterial=null;gate.resolve();await flush(r);
    assert.deepEqual(materials(f),[one,two]);assert.equal(scene.overrideMaterial,null);
  }finally{gate.resolve();r.dispose();}
});

test('clipping admission reads the captured override even after the pass restores the property',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,m=new T.MeshDepthMaterial();
  m.clippingPlanes=[new T.Plane(new T.Vector3(1,0,0),.5)];
  try{
    await r.init();r.localClippingEnabled=true;r.setRenderTarget(a);
    const p=new RenderPass(scene,camera,m);p.render(r,null,a);await flush(r);
    const entry=r._targets.capture(a)._dispatcher.entry(scene);
    assert.equal(entry.clipping,true);assert.deepEqual(materials(f),[m]);assert.equal(scene.overrideMaterial,null);
  }finally{r.dispose();}
});

test('adding clipping planes to a prepared override triggers preparation instead of dropping clipping',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,m=new T.MeshDepthMaterial();
  try{
    await r.init();r.localClippingEnabled=true;r.setRenderTarget(a);scene.overrideMaterial=m;await r.compileAsync(scene,camera);
    m.clippingPlanes=[new T.Plane(new T.Vector3(1,0,0),.5)];r.render(scene,camera);scene.overrideMaterial=null;await flush(r);
    assert.equal(r._targets.capture(a)._dispatcher.entry(scene).clipping,true);assert.deepEqual(materials(f),[m]);
  }finally{r.dispose();}
});

test('a rejected later override does not replace the preceding submitted pass',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,m=shader('valid'),bad=shader('invalid');bad.fragmentShader='invalid GLSL';
  try{
    await r.init();r.setRenderTarget(a);scene.overrideMaterial=m;r.render(scene,camera);
    scene.overrideMaterial=bad;r.render(scene,camera);scene.overrideMaterial=null;
    await assert.rejects(flush(r));assert.deepEqual(materials(f),[m]);assert.equal(scene.overrideMaterial,null);
  }finally{r.dispose();}
});

test('queued override material selection honors allowOverride false',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,m=shader('pass');f.original.allowOverride=false;
  try{
    r.setRenderTarget(a);scene.overrideMaterial=m;const ready=r.renderAsync(scene,camera);scene.overrideMaterial=null;
    await ready;assert.deepEqual(materials(f),[f.original]);assert.equal(f.mesh.material,f.original);
  }finally{r.dispose();}
});

test('concurrent compileAsync requests keep independent pass selections and release their scope',async()=>{
  const f=fixture(),{renderer:r,scene,camera,a}=f,one=shader('one'),two=shader('two');
  try{
    r.setRenderTarget(a);scene.overrideMaterial=one;const first=r.compileAsync(scene,camera);
    scene.overrideMaterial=two;const second=r.compileAsync(scene,camera);scene.overrideMaterial=null;
    await Promise.all([first,second]);
    scene.overrideMaterial=two;r.render(scene,camera);scene.overrideMaterial=null;
    assert.deepEqual(materials(f),[two]);assert.equal(r._targets.capture(a)._preparationOptions,undefined);
  }finally{r.dispose();}
});
