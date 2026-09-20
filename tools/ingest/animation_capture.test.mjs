import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createGpuAnimationPresentation} from './animation_presentation.mjs';
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
// Execute the production presentation, output-pass orchestration and readback.
// Only native WebGPU and the caller's scene are doubles: no WGSL/driver/pixel
// rendering is claimed. HDR and display bytes below are explicit test fixtures.
function environment({holdMap=false,holdQueue=false}={}){
  const buffers=[],textures=[],copies=[],frames=[],events=[],scopes=[],lost=deferred(),queueGate=deferred();
  let invalid=null;
  function texture(width,height,format='bgra8unorm',usage=17,sampleCount=1){
    const bpp=format==='rgba16float'?8:4,bytes=new Uint8Array(width*height*bpp);
    const t={width,height,format,usage,sampleCount,bpp,bytes,dimension:'2d',depthOrArrayLayers:1,mipLevelCount:1,destroyed:0,
      createView(options){return {texture:this,options};},destroy(){this.destroyed++;}};
    return t;
  }
  const device={limits:{maxTextureDimension2D:8192,maxBufferSize:1024*1024},lost:lost.promise,
    destroy(){assert.fail('Borrowed device destroyed');},
    pushErrorScope(filter){scopes.push(filter);},
    popErrorScope(){assert.ok(scopes.length);const filter=scopes.pop(),error=filter==='validation'?invalid:null;if(error)invalid=null;return Promise.resolve(error);},
    createShaderModule(descriptor){events.push('shader');assert.match(descriptor.code,/@fragment/);return descriptor;},
    createBindGroupLayout(descriptor){return descriptor;},createPipelineLayout(descriptor){return descriptor;},
    async createRenderPipelineAsync(descriptor){return descriptor;},createBindGroup(descriptor){return descriptor;},
    createTexture(d){const t=texture(d.size[0],d.size[1],d.format,d.usage,d.sampleCount);textures.push(t);events.push('texture');return t;},
    createBuffer(d){const gate=deferred();let mapped=false;const b={d,gate,storage:new ArrayBuffer(d.size),destroyed:0,
      mapAsync(){events.push('map');return (holdMap?gate.promise:Promise.resolve()).then(()=>{mapped=true;});},
      getMappedRange(){assert.ok(mapped);return this.storage;},
      destroy(){this.destroyed++;structuredClone(this.storage,{transfer:[this.storage]});},
    };buffers.push(b);events.push(d.usage===9?'staging':'uniform');return b;},
    createCommandEncoder(){const commands=[];return {
      beginRenderPass(d){commands.push({output:d});return {setPipeline(){},setBindGroup(){},draw(n){assert.equal(n,3);},end(){}};},
      copyTextureToBuffer(source,destination,size){commands.push({source,destination,size});copies.push(commands.at(-1));},
      finish(){return commands;},
    };},
    queue:{writeBuffer(buffer,offset,bytes){new Uint8Array(buffer.storage).set(new Uint8Array(bytes),offset);},
      submit(list){for(const commands of list)for(const c of commands){
        if(c.output){events.push('output-submit');continue;}
        events.push('copy-submit');const {source:s,destination:d,size}=c,t=s.texture;
        assert.equal(t.sampleCount,1);assert.ok(t.usage&1);assert.equal(d.bytesPerRow%256,0);
        const dest=new Uint8Array(d.buffer.storage);
        for(let y=0;y<size[1];y++){
          const at=((s.origin[1]+y)*t.width+s.origin[0])*t.bpp;
          dest.set(t.bytes.subarray(at,at+size[0]*t.bpp),y*d.bytesPerRow);
        }
      }},
      onSubmittedWorkDone(){events.push('fence');return holdQueue?queueGate.promise:Promise.resolve();},
    },
  };
  const scene={disposed:false,failed:false,word:0x3c00,pose:{},view:{},cameras:[],lights:[],controller:{},draws:[],deformers:[],source:[],diagnostics:[],poseVersion:0,bufferBytes:123,
    render(frame){events.push('scene-submit');frames.push(frame);const t=(frame.resolveTarget??frame.colorView).texture;
      if(t.format==='rgba16float'){const v=new DataView(t.bytes.buffer);for(let i=0;i<t.width*t.height;i++)[this.word,0x4000,0x4200,0x3800].forEach((w,k)=>v.setUint16(i*8+k*2,w,true));}
      return this;
    },
    renderCamera(frame,settings){this.cameraSettings=settings;return this.render(frame);},
    update(){this.poseVersion++;return this;},upload(){return this;},
    async whenIdle(){events.push('scene-validated');},dispose(){this.disposed=true;this.pose.disposed=true;},
  };
  return {device,scene,buffers,textures,copies,frames,events,scopes,lost,queueGate,texture,
    invalidate(message){invalid={message};},get staging(){return buffers.filter(b=>b.d.usage===9);}};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function clean(g,p){p.dispose();assert.equal(g.scopes.length,0);for(const b of g.buffers)assert.equal(b.destroyed,1);for(const t of g.textures)assert.equal(t.destroyed,1);}

test('default presentation allocation/usage is unchanged and readback is disabled',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device),target=g.texture(2,1);p.render(g.scene,{target});
  assert.equal(p.readbackEnabled,false);assert.equal(p.bufferBytes,16);assert.equal(p.readbackBufferBytes,0);assert.equal(g.staging.length,0);
  assert.equal(g.textures[0].usage,20);await assert.rejects(p.readPixels(),{code:'ANIMATION_PRESENTATION_READBACK_DISABLED'});
  clean(g,p);assert.equal(target.destroyed,0);
});
test('opt-in is lazy, and captures resolved MSAA HDR with version/alpha/color metadata',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true,sampleCount:4});
  assert.equal(p.readbackEnabled,true);assert.equal(g.textures.length,0);assert.equal(g.staging.length,0);
  await assert.rejects(p.readPixels(),{code:'ANIMATION_PRESENTATION_READBACK_EMPTY'});
  p.render(g.scene,{target:g.texture(2,1)});const result=await p.readPixels();
  assert.equal(g.textures[0].usage,21);assert.equal(g.textures[1].usage,16);assert.equal(g.textures[1].sampleCount,4);
  assert.equal(g.copies[0].source.texture,g.frames[0].resolveTarget.texture);
  assert.deepEqual([...result.data],[1,2,3,.5,1,2,3,.5]);assert.equal(result.source,'hdr');assert.equal(result.colorSpace,'linear');assert.equal(result.alpha,'premultiplied');assert.equal(result.presentationVersion,1);
  assert.equal(p.readbackPending,0);assert.equal(p.bufferBytes,16);clean(g,p);
});
test('display capture selects native target bytes, swaps BGRA, and retains output conventions',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true,outputColorSpace:'srgb'}),target=g.texture(1,1);
  target.bytes.set([4,8,16,64]);p.render(g.scene,{target,output:{outputAlpha:'straight'}});
  const result=await p.readPixels({source:'output'});
  assert.equal(g.copies[0].source.texture,target);assert.deepEqual([...result.data],[16,8,4,64]);assert.equal(result.alpha,'straight');assert.equal(result.colorSpace,'srgb');assert.equal(result.srgb,false);
  assert.ok(g.events.indexOf('output-submit')<g.events.indexOf('copy-submit'));clean(g,p);assert.equal(target.destroyed,0);
});
test('display capture without COPY_SRC fails without allocations or disabling HDR capture',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true}),target=g.texture(1,1,'bgra8unorm',16);p.render(g.scene,{target});
  const events=g.events.length;await assert.rejects(p.readPixels({source:'output'}),{code:'ANIMATION_READBACK_TEXTURE'});
  assert.equal(g.events.length,events);assert.equal(p.failed,false);assert.deepEqual([...(await p.readPixels()).data],[1,2,3,.5]);clean(g,p);
});
test('capture snapshots queue and version before a following animated frame',async()=>{
  const g=environment({holdMap:true}),p=await createGpuAnimationPresentation(g.device,{readback:true}),target=g.texture(1,1);
  p.render(g.scene,{target});const first=p.readPixels();assert.equal(g.copies.length,1);assert.equal(p.readbackPending,1);assert.equal(p.readbackReservedBytes,272);
  g.scene.word=0x4500;p.render(g.scene,{target});g.staging[0].gate.resolve();const old=await first;
  assert.equal(old.data[0],1);assert.equal(old.presentationVersion,1);assert.equal(p.version,2);
  const second=p.readPixels();g.staging[1].gate.resolve();assert.equal((await second).data[0],5);clean(g,p);
});
test('capture survives resize retirement of its source and keeps the old extent',async()=>{
  const g=environment({holdMap:true}),p=await createGpuAnimationPresentation(g.device,{readback:true});
  p.render(g.scene,{target:g.texture(2,1)});const pending=p.readPixels(),old=g.textures[0];
  p.render(g.scene,{target:g.texture(4,2)});assert.equal(old.destroyed,1);g.staging[0].gate.resolve();const result=await pending;
  assert.equal(result.width,2);assert.equal(result.height,1);assert.equal(result.data.length,8);assert.equal(p.width,4);assert.equal(p.height,2);clean(g,p);
});
test('camera route captures the same current resolved image',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true}),settings={camera:2,aspect:1.5};
  p.renderCamera(g.scene,{target:g.texture(3,1)},settings);assert.equal(g.scene.cameraSettings,settings);
  const a=await p.readPixels({x:1,width:2});assert.equal(a.width,2);assert.deepEqual(a.origin,[1,0,0]);assert.equal(a.presentationVersion,1);clean(g,p);
});
test('capture budget covers producer validation even after mapping completes',async()=>{
  const g=environment({holdQueue:true}),p=await createGpuAnimationPresentation(g.device,{readback:{maxPending:1,maxBytes:272}});p.render(g.scene,{target:g.texture(1,1)});
  const c=new AbortController(),pending=p.readPixels({signal:c.signal});await tick();assert.equal(p.readbackPending,1);assert.equal(p.readbackReservedBytes,272);
  await assert.rejects(p.readPixels(),{code:'ANIMATION_READBACK_LIMIT'});assert.equal(g.staging.length,1);
  c.abort();await assert.rejects(pending,{name:'AbortError'});assert.equal(p.readbackPending,0);assert.equal(p.failed,false);
  g.queueGate.resolve();await p.readPixels();clean(g,p);
});
test('presentation whenIdle includes pending captures',async()=>{
  const g=environment({holdMap:true}),p=await createGpuAnimationPresentation(g.device,{readback:true});p.render(g.scene,{target:g.texture(1,1)});
  const read=p.readPixels();let done=false;const idle=p.whenIdle().then(()=>{done=true;});await tick();assert.equal(done,false);
  g.staging[0].gate.resolve();await read;await idle;assert.equal(done,true);clean(g,p);
});
test('dispose aborts capture and releases staging and owned attachments exactly once',async()=>{
  const g=environment({holdMap:true}),p=await createGpuAnimationPresentation(g.device,{readback:true}),target=g.texture(1,1);p.render(g.scene,{target});
  const read=p.readPixels();p.dispose();p.dispose();await assert.rejects(read,/DISPOSED/);assert.equal(p.readbackPending,0);clean(g,p);assert.equal(target.destroyed,0);assert.equal(g.scene.disposed,false);
});
test('completed snapshots outlive presentation disposal',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true});p.render(g.scene,{target:g.texture(1,1)});const a=await p.readPixels();clean(g,p);assert.deepEqual([...a.data],[1,2,3,.5]);
});
test('device loss rejects capture and makes the presentation terminal',async()=>{
  const g=environment({holdMap:true}),p=await createGpuAnimationPresentation(g.device,{readback:true});p.render(g.scene,{target:g.texture(1,1)});const read=p.readPixels();
  g.lost.resolve({message:'lost'});await assert.rejects(read,/DEVICE_LOST/);assert.equal(p.failed,true);await assert.rejects(p.readPixels(),/DEVICE_LOST/);clean(g,p);
});
test('failed draw validation cannot publish apparently valid copied pixels',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true});g.scene.whenIdle=async()=>{throw Error('draw invalid');};p.render(g.scene,{target:g.texture(1,1)});
  await assert.rejects(p.readPixels(),/draw invalid|DISPOSED/);assert.equal(p.failed,true);clean(g,p);
});
test('failed attachment validation cannot publish a capture',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true});g.invalidate('bad HDR');p.render(g.scene,{target:g.texture(1,1)});
  await assert.rejects(p.readPixels(),/bad HDR|DISPOSED/);assert.equal(p.failed,true);clean(g,p);
});
for(const settings of [{source:'depth'},{width:0},{x:-1},{completion:Promise.resolve()},{bogus:1},{signal:{}}])test('invalid capture settings have no submission effects '+Object.keys(settings),async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true});p.render(g.scene,{target:g.texture(1,1)});
  const n=g.events.length;await assert.rejects(p.readPixels(settings));assert.equal(g.events.length,n);assert.equal(p.failed,false);clean(g,p);
});
for(const readback of [null,42,[],{maxBytes:0},{maxPending:65},{unknown:1}])test('invalid readback construction '+JSON.stringify(readback)+' precedes GPU allocations',async()=>{
  const g=environment();await assert.rejects(createGpuAnimationPresentation(g.device,{readback}));assert.equal(g.events.length,0);
});
test('readback limits are snapshotted before asynchronous output construction',async()=>{
  const g=environment({holdMap:true}),options={readback:{maxPending:1}},pending=createGpuAnimationPresentation(g.device,options);options.readback.maxPending=0;const p=await pending;
  p.render(g.scene,{target:g.texture(1,1)});const read=p.readPixels();await assert.rejects(p.readPixels(),{code:'ANIMATION_READBACK_LIMIT'});g.staging[0].gate.resolve();await read;clean(g,p);
});
test('capture submission rejects render reentry from a producer validation hook',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true}),frame={target:g.texture(1,1)};p.render(g.scene,frame);
  g.scene.whenIdle=()=>{assert.throws(()=>p.render(g.scene,frame),{code:'ANIMATION_PRESENTATION_REENTRANT'});};await p.readPixels();assert.equal(p.version,1);clean(g,p);
});

// Owning-loader tests replace only asset/geometry/texture/model creation stages.
// The production loader, presentation, output pass and readback still execute.
const encoded=s=>'data:text/javascript;base64,'+Buffer.from(s).toString('base64');
const boundary=encoded(`
export class GltfAssetError extends Error {constructor(code,message){super(message);this.code=code;}}
export class GltfTextureError extends GltfAssetError {}
export async function loadGltfAsset(source,options){return source.load(options);}
export function prepareGltfAnimationModel(json,buffers,options){return {textureRequests:[],resolveTextures(){return {};}};}
export async function createGltfTextureResources(device){return device.resources;}
export async function createGpuDecodedAnimationScene(device,prepared,options){device.settings=options;return device.scene;}
`);
let code=readFileSync(new URL('./gltf_scene_loader.mjs',import.meta.url),'utf8');
for(const file of ['gltf_asset.mjs','animation_model.mjs','gltf_textures.mjs','animation_model_gpu.mjs'])code=code.replace("'./"+file+"'",JSON.stringify(boundary));
code=code.replace("'./animation_presentation.mjs'",JSON.stringify(new URL('./animation_presentation.mjs',import.meta.url).href));
const {loadGpuGltfAnimationScene}=await import(encoded(code));
async function load(g,output={readback:true}){
  g.device.scene=g.scene;g.device.resources={failed:false,textureBytes:42,resolveTexture(){},disposed:0,dispose(){this.disposed++;}};
  const source={async load(){return {json:{},buffers:[],bytesLoaded:7};}};
  return loadGpuGltfAnimationScene(g.device,source,{output});
}
test('owning loader exposes end-to-end capture, accounting and camera route',async()=>{
  const g=environment({holdMap:true}),m=await load(g),target=g.texture(1,1);
  assert.equal(m.readbackEnabled,true);assert.equal(g.device.settings.renderer.format,'rgba16float');
  const settings={camera:0};m.renderCamera({target},settings);const p=m.readPixels();assert.equal(m.readbackPending,1);assert.equal(m.readbackBufferBytes,256);assert.equal(m.outputBufferBytes,272);assert.equal(m.readbackReservedBytes,272);
  g.staging[0].gate.resolve();const a=await p;assert.equal(a.presentationVersion,1);assert.equal(g.scene.cameraSettings,settings);assert.equal(m.readbackPending,0);assert.equal(m.sourceExportEnabled,false);
  m.dispose();assert.equal(g.scene.disposed,true);assert.deepEqual([...a.data],[1,2,3,.5]);assert.equal(target.destroyed,0);
});
test('owning loader without output does not acquire capture allocations',async()=>{
  const g=environment(),m=await load(g,null);assert.equal(m.readbackEnabled,false);assert.equal(m.readbackPending,0);assert.equal(g.buffers.length,0);
  await assert.rejects(m.readPixels(),{code:'GLTF_MODEL_READBACK_DISABLED'});m.dispose();
});
test('owning loader with ordinary output still refuses capture',async()=>{
  const g=environment(),m=await load(g,{});assert.equal(m.readbackEnabled,false);m.render({target:g.texture(1,1)});
  await assert.rejects(m.readPixels(),{code:'ANIMATION_PRESENTATION_READBACK_DISABLED'});assert.equal(g.staging.length,0);m.dispose();
});
test('owning loader cancellation does not dispose a usable scene',async()=>{
  const g=environment({holdMap:true}),m=await load(g),c=new AbortController();m.render({target:g.texture(1,1)});const p=m.readPixels({signal:c.signal});c.abort();
  await assert.rejects(p,{name:'AbortError'});assert.equal(m.disposed,false);assert.equal(m.failed,false);assert.equal(m.readbackPending,0);m.dispose();
});
test('owning loader disposal cancels capture but never destroys borrowed display targets',async()=>{
  const g=environment({holdMap:true}),m=await load(g),target=g.texture(1,1);m.render({target});const p=m.readPixels({source:'output'});m.dispose();
  await assert.rejects(p,/DISPOSED/);await assert.rejects(m.readPixels(),{code:'GLTF_MODEL_DISPOSED'});assert.equal(target.destroyed,0);for(const b of g.buffers)assert.equal(b.destroyed,1);
});
test('owning loader terminal capture failure releases model and texture ownership',async()=>{
  const g=environment(),m=await load(g);g.scene.whenIdle=async()=>{throw Error('render failure');};m.render({target:g.texture(1,1)});
  await assert.rejects(m.readPixels(),/render failure|DISPOSED/);assert.equal(g.scene.disposed,true);assert.ok(g.device.resources.disposed>0);for(const b of g.buffers)assert.equal(b.destroyed,1);
});
test('capture refuses unrendered target mips even when standalone texture readback supports them',async()=>{
  const g=environment(),p=await createGpuAnimationPresentation(g.device,{readback:true}),target=g.texture(4,4);target.mipLevelCount=3;p.render(g.scene,{target});
  await assert.rejects(p.readPixels({source:'output',mipLevel:1}),{code:'ANIMATION_PRESENTATION_OPTIONS'});assert.equal(g.staging.length,0);clean(g,p);
});
