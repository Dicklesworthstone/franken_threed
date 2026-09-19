import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
// Real owning-loader source, presentation, output shader generation and GPU
// submission code execute. Asset/decode/texture/model construction and GPU calls
// are explicit boundary doubles. This does not execute codecs, poses or WGSL.
if(!process.execArgv.includes('--experimental-vm-modules')){
  test('owning glTF HDR presentation integration',()=>{
    const env={...process.env};delete env.NODE_TEST_CONTEXT;
    const text=execFileSync(process.execPath,['--experimental-vm-modules','--test','--test-reporter=tap',fileURLToPath(import.meta.url)],{env,encoding:'utf8',timeout:20000,stdio:['ignore','pipe','pipe']});
    assert.match(text,/# fail 0/);assert.match(text,/# pass [1-9]\d*/);
  });
}else{
  const vm=await import('node:vm');
  const root=new URL('./gltf_scene_loader.mjs',import.meta.url),source=await readFile(root,'utf8');
  const deferred=()=>{let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j;});return {promise,resolve,reject};};
  async function setup({assetError,textureError,sceneError,compile,assetWait}={}){
    const calls={assets:0,model:[],prepare:[],textures:[],render:[],events:[],camera:[],updates:[],exports:[]},buffers=[],owned=[],scopes=[],lost=deferred();
    let sceneDisposed=false,textureDisposed=false;
    const model={pose:{version:1},view:{},controller:{},draws:[],deformers:[],source:[],diagnostics:[],failed:false,bufferBytes:48,
      get disposed(){return sceneDisposed;},dispose(){sceneDisposed=true;},render(f){calls.render.push(f);calls.events.push('scene');return this;},
      renderCamera(f,o){calls.camera.push(o);return this.render(f);},update(dt){calls.updates.push(dt);return this;},upload(){return this;},whenIdle:async()=>{},
      pick:()=>['pick'],raycast:()=>['ray'],pickingEnabled:true,exportingEnabled:true,exportPoseGLB(o){calls.exports.push(o);return Promise.resolve(new ArrayBuffer(4));}};
    const resources={failed:false,textureBytes:7,dispose(){textureDisposed=true;},resolveTexture:()=>({view:{},sampler:{}})};
    const device={limits:{maxTextureDimension2D:4096},lost:lost.promise,queue:{writeBuffer(b,o,v){calls.events.push('parameters');},submit(){calls.events.push('output');},onSubmittedWorkDone:async()=>{}},
      createShaderModule:x=>x,createBindGroupLayout:x=>x,createPipelineLayout:x=>x,createRenderPipelineAsync:x=>compile?.promise ?? Promise.resolve(x),createBindGroup:x=>x,
      createBuffer(x){const b={...x,destroyed:0,destroy(){this.destroyed++;}};buffers.push(b);return b;},
      createTexture(x){const t=target(...x.size,x.format,x.sampleCount,x.usage);owned.push(t);return t;},pushErrorScope(){},popErrorScope:()=>scopes.shift() ?? Promise.resolve(null),
      createCommandEncoder:()=>({beginRenderPass:()=>({setPipeline(){},setBindGroup(){},draw(){},end(){}}),finish:()=>({})})};
    function target(width=8,height=4,depthOrArrayLayers=1,format='bgra8unorm',sampleCount=1,usage=16){return {width,height,depthOrArrayLayers,format,sampleCount,usage,dimension:'2d',destroyed:0,createView: function(){return {texture:this};},destroy(){this.destroyed++;}};}
    class AssetError extends Error{constructor(code,message){super(message);this.code=code;}}
    const imports={
      './gltf_asset.mjs':{GltfAssetError:AssetError,loadGltfAsset:async()=>{calls.assets++;if(assetWait)await assetWait.promise;if(assetError)throw assetError;return {json:{},buffers:[],bytesLoaded:111,readImage:async()=>assert.fail('No images in fixture')};}},
      './animation_model.mjs':{prepareGltfAnimationModel(j,b,o){calls.prepare.push(o);return {textureRequests:[],resolveTextures:()=>({})};}},
      './gltf_textures.mjs':{GltfTextureError:AssetError,createGltfTextureResources:async(d,r,reader,o)=>{calls.textures.push(o);if(textureError)throw textureError;return resources;}},
      './animation_model_gpu.mjs':{createGpuDecodedAnimationScene:async(d,p,o)=>{calls.model.push(o);if(sceneError)throw sceneError;return model;}},
    };
    const module=new vm.SourceTextModule(source,{identifier:root.href,importModuleDynamically:specifier=>import(new URL(specifier,root).href)});
    await module.link(specifier=>{const values=imports[specifier];assert.ok(values,'Unexpected static dependency '+specifier);return new vm.SyntheticModule(Object.keys(values),function(){for(const [k,v]of Object.entries(values))this.setExport(k,v);});});await module.evaluate();
    return {load:options=>module.namespace.loadGpuGltfAnimationScene(device,new Uint8Array(),options),calls,device,buffers,owned,scopes,lost,target,model,resources,
      get sceneDisposed(){return sceneDisposed;},get textureDisposed(){return textureDisposed;}};
  }
  test('default loader keeps the original attachment route and allocates no presentation resources',async()=>{
    const f=await setup(),loaded=await f.load({scene:{renderer:{format:'rgba8unorm',sampleCount:4}}}),frame={colorView:{},depthView:{},viewProjection:[]};
    loaded.render(frame);assert.equal(f.calls.render[0],frame);assert.equal(f.calls.model[0].renderer.format,'rgba8unorm');
    assert.equal(loaded.outputEnabled,false);assert.equal(loaded.outputBufferBytes,0);assert.equal(loaded.outputTextureBytes,0);assert.equal(f.buffers.length,0);
    await loaded.whenIdle();loaded.dispose();assert.equal(f.sceneDisposed,true);assert.equal(f.textureDisposed,true);
  });
  test('owning loader renders through rgba16float, preserves 4x MSAA/depth and applies one final output pass',async()=>{
    const f=await setup(),loaded=await f.load({scene:{renderer:{format:'rgba8unorm',sampleCount:4,depthFormat:'depth16unorm',maxDraws:9}},output:{toneMapping:'aces-filmic',exposure:2}});
    assert.deepEqual({...f.calls.model[0].renderer},{format:'rgba16float',sampleCount:4,depthFormat:'depth16unorm',maxDraws:9});
    const target=f.target();loaded.render({target,viewProjection:[],draws:[]});await loaded.whenIdle();
    assert.deepEqual(f.calls.events,['scene','parameters','output']);assert.equal(f.calls.render[0].resolveTarget.texture.format,'rgba16float');
    assert.equal(loaded.outputTextureBytes,8*4*(40+8));assert.equal(loaded.outputBufferBytes,16);assert.equal(loaded.textureBytes,7);assert.equal(loaded.bufferBytes,48);
    loaded.dispose();assert.ok(f.owned.every(t=>t.destroyed===1));assert.equal(f.buffers[0].destroyed,1);assert.equal(target.destroyed,0);
  });
  test('authored-camera rendering, current-pose queries and export still use the owning model',async()=>{
    const f=await setup(),loaded=await f.load({output:{},picking:true,exporting:true}),camera={cameraNode:4,aspectRatio:2};
    loaded.update(0.25);loaded.renderCamera({target:f.target(),output:{exposure:3}},camera);
    assert.equal(f.calls.camera[0],camera);assert.deepEqual(f.calls.updates,[0.25]);assert.deepEqual(loaded.pick([0,0]),['pick']);assert.deepEqual(loaded.raycast({}),['ray']);
    await loaded.exportPoseGLB();assert.equal(f.calls.exports[0].sceneView,f.model.view);assert.equal(f.calls.model[0].picking,true);assert.equal(f.calls.model[0].exporting,true);loaded.dispose();
  });
  test('output/renderer/decoder settings are snapshotted before the first await',async()=>{
    const wait=deferred(),f=await setup({assetWait:wait}),loader={parse(){}},options={scene:{renderer:{sampleCount:4,depthFormat:'depth16unorm'}},output:{toneMapping:'agx'},textures:{ktx2Loader:loader}};
    const pending=f.load(options);options.output.toneMapping='bad';options.scene.renderer.sampleCount=2;options.textures.ktx2Loader=null;
    wait.resolve();const loaded=await pending;assert.equal(f.calls.model[0].renderer.sampleCount,4);assert.equal(f.calls.prepare[0].basisu,true);assert.equal(f.calls.textures[0].ktx2Loader,loader);
    loaded.render({target:f.target()});await loaded.whenIdle();loaded.dispose();
  });
  for(const field of ['assetError','textureError','sceneError'])test(`${field}: downstream initialization failure releases presentation resources`,async()=>{
    const error=new Error(field),f=await setup({[field]:error});await assert.rejects(f.load({output:{}}),e=>e===error);
    assert.equal(f.buffers.length,1);assert.equal(f.buffers[0].destroyed,1);assert.equal(f.owned.length,0);if(field==='sceneError')assert.equal(f.textureDisposed,true);
  });
  for(const config of [{output:false},{output:{exposure:-1}},{output:{sampleCount:2}},
    {output:{sampleCount:4},scene:{renderer:{sampleCount:1}}},{output:{depthFormat:null},scene:{renderer:{depthFormat:'depth24plus'}}}])
    test('invalid output settings fail before asset fetch: '+JSON.stringify(config),async()=>{
      const f=await setup();await assert.rejects(f.load(config));assert.equal(f.calls.assets,0);assert.equal(f.buffers.length,0);
    });
  test('construction cancellation rejects pending output compilation before asset I/O',async()=>{
    const compile=deferred(),f=await setup({compile}),c=new AbortController();const pending=f.load({output:{},signal:c.signal});
    await new Promise(r=>setImmediate(r));c.abort();await assert.rejects(pending,{name:'AbortError'});compile.resolve({});assert.equal(f.calls.assets,0);assert.equal(f.buffers.length,0);
  });
  test('mismatched construction signals fail without starting output or asset work',async()=>{
    const f=await setup(),a=new AbortController(),b=new AbortController();await assert.rejects(f.load({signal:a.signal,output:{signal:b.signal}}),{code:'GLTF_MODEL_LOAD_OPTIONS'});assert.equal(f.calls.assets,0);assert.equal(f.buffers.length,0);
  });
  test('presentation validation errors leave the loaded model usable',async()=>{
    const f=await setup(),loaded=await f.load({output:{}});assert.throws(()=>loaded.render({target:f.target(),loadOp:'load'}),{code:'ANIMATION_PRESENTATION_HISTORY'});
    assert.equal(loaded.disposed,false);assert.equal(loaded.failed,false);loaded.render({target:f.target()});await loaded.whenIdle();loaded.dispose();
  });
  test('output driver failure drains and disposes the whole owning scene',async()=>{
    const f=await setup(),loaded=await f.load({output:{}});f.scopes.push(Promise.resolve({message:'allocation validation'}),Promise.resolve(null));
    loaded.render({target:f.target()});await assert.rejects(loaded.whenIdle());assert.equal(loaded.disposed,true);assert.equal(loaded.failed,true);
    assert.equal(f.textureDisposed,true);assert.ok(f.owned.every(t=>t.destroyed===1));assert.equal(f.buffers[0].destroyed,1);
  });
}
