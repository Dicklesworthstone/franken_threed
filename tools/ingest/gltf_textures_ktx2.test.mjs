import test from 'node:test';
import assert from 'node:assert/strict';
import {createGltfTextureResources} from './gltf_textures.mjs';
import {syntheticKtx2,retainedDecoderDouble} from './fixtures/animation/ktx2_upload_fixture.mjs';

// Independent layout assertions at the WebGPU boundary, not a native GPU mock
// presented as a conformance oracle. Production texture manager + adapter run.
function gpu({features=['texture-compression-bc'],errorAt=0,throwAt=0}={}) {
  const textures=[],writes=[],samplers=[],scopes=[],external=[],passes=[];let lose,pops=0,done=0;
  const device={features:new Set(features),limits:{maxTextureDimension2D:8192},lost:new Promise(resolve=>{lose=resolve;}),
    destroy(){assert.fail('Borrowed device');},pushErrorScope(s){scopes.push(s);},
    popErrorScope(){assert.ok(scopes.pop());return Promise.resolve(++pops===errorAt?{message:'GPU rejected upload'}:null);},
    createTexture(descriptor){assert.equal(scopes.length,2);const texture={descriptor,destroyed:0,destroy(){this.destroyed++;},createView(options){return {texture:this,options};}};textures.push(texture);return texture;},
    createSampler(descriptor){const sampler={descriptor};samplers.push(sampler);return sampler;},
    createShaderModule(d){return d;},async createRenderPipelineAsync(d){return {d,getBindGroupLayout(){return {};}};},
    createBindGroup(d){return d;},createCommandEncoder(){return {beginRenderPass(d){passes.push(d);return {setPipeline(){},setBindGroup(){},draw(){},end(){}};},finish(){return {};}};},
    queue:{writeTexture(destination,data,layout,extent){
      assert.equal(scopes.length,2);const d=destination.texture.descriptor;
      const block=d.format.startsWith('rgba8')?1:4,bpp=/bc1|etc2-rgb8/.test(d.format)?8:block===1?4:16;
      const w=Math.max(1,Math.floor(d.size[0]/2**destination.mipLevel)),h=Math.max(1,Math.floor(d.size[1]/2**destination.mipLevel));
      assert.deepEqual(extent,[Math.ceil(w/block)*block,Math.ceil(h/block)*block,1]);
      assert.equal(layout.bytesPerRow,Math.ceil(w/block)*bpp);assert.equal(data.byteLength,layout.bytesPerRow*Math.ceil(h/block));
      assert.equal(d.usage,6);assert.ok(destination.mipLevel<d.mipLevelCount);
      writes.push({destination,layout,extent,data:data.slice()});if(writes.length===throwAt)throw new Error('write failure');
    },async onSubmittedWorkDone(){done++;},submit(){},
    copyExternalImageToTexture(...args){external.push(args);}},
  };
  return {device,textures,writes,samplers,scopes,external,passes,lose,get done(){return done;}};
}
const request=(textureIndex=0,imageIndex=0,colorSpace='srgb',sampler={})=>({textureIndex,imageIndex,colorSpace,sampler});
const image=bytes=>({bytes,mimeType:'image/ktx2'});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==','base64'));

for(const [format,block,blockBytes,feature,gpuFormat]of [
  [33777,4,8,'texture-compression-bc','bc1-rgba-unorm'],[33778,4,16,'texture-compression-bc','bc2-rgba-unorm'],
  [33779,4,16,'texture-compression-bc','bc3-rgba-unorm'],[36492,4,16,'texture-compression-bc','bc7-rgba-unorm'],
  [36196,4,8,'texture-compression-etc2','etc2-rgb8unorm'],[37492,4,8,'texture-compression-etc2','etc2-rgb8unorm'],
  [37496,4,16,'texture-compression-etc2','etc2-rgba8unorm'],[37808,4,16,'texture-compression-astc','astc-4x4-unorm'],
  [1023,1,4,null,'rgba8unorm'],
])test(`uploads retained format ${format} with exact mip/block extents and private bytes`,async()=>{
  const g=gpu({features:feature?[feature]:[]}),bytes=syntheticKtx2({width:12,height:8}),copy=bytes.slice();
  delete g.device.queue.copyExternalImageToTexture;
  const decoder=retainedDecoderDouble({format,block,blockBytes,onParse({buffer,texture,onLoad}){
    assert.deepEqual(g.scopes,[]);assert.notEqual(buffer,bytes.buffer);structuredClone(buffer,{transfer:[buffer]});onLoad(texture);
  }});
  const resources=await createGltfTextureResources(g.device,[request()],async()=>image(bytes),{ktx2Loader:decoder,createImageBitmap:null});
  assert.equal(g.textures[0].descriptor.format,gpuFormat+'-srgb');assert.equal(g.textures[0].descriptor.mipLevelCount,4);
  assert.deepEqual(bytes,copy);assert.equal(decoder.textures[0].disposed,1);assert.equal(g.done,1);assert.equal(g.passes.length,0);
  assert.equal(resources.textureBytes,g.writes.reduce((sum,w)=>sum+w.data.length,0));assert.equal(resources.textureCount,1);
  for(let i=0;i<g.writes.length;i++)assert.ok(g.writes[i].data.every(n=>n===17+i));
  assert.equal(resources.resolveTexture(request()).view.texture,g.textures[0]);resources.dispose();resources.dispose();
  assert.equal(g.textures[0].destroyed,1);assert.equal(resources.textureBytes,0);assert.deepEqual(g.scopes,[]);
});
test('linear maps use a linear GPU format and one image/transcode shared by distinct samplers',async()=>{
  const g=gpu(),decoder=retainedDecoderDouble(),bytes=syntheticKtx2({colorSpace:'linear'});let reads=0;
  const a=request(0,0,'linear',{minFilter:9728,magFilter:9728,wrapS:33071,wrapT:33648}),b=request(1,0,'linear',{minFilter:9987});
  const r=await createGltfTextureResources(g.device,[a,b,b],async()=>{reads++;return image(bytes);},{ktx2Loader:decoder});
  assert.equal(reads,1);assert.equal(decoder.calls.length,1);assert.equal(r.textureCount,1);assert.equal(g.writes.length,4);
  const ar=r.resolveTexture(a),br=r.resolveTexture(b);assert.equal(ar.view,br.view);assert.notEqual(ar.sampler,br.sampler);
  assert.deepEqual(ar.sampler.descriptor,{addressModeU:'clamp-to-edge',addressModeV:'mirror-repeat',magFilter:'nearest',minFilter:'nearest',mipmapFilter:'nearest',lodMaxClamp:0});
  assert.equal(br.sampler.descriptor.lodMaxClamp,undefined);assert.equal(g.textures[0].descriptor.format,'bc7-rgba-unorm');r.dispose();
});
for(const levels of [1,2,4])test(`preserves ${levels} authored levels without regenerating or rejecting a partial pyramid`,async()=>{
  const g=gpu(),decoder=retainedDecoderDouble(),bytes=syntheticKtx2({levels});
  const r=await createGltfTextureResources(g.device,[request()],async()=>image(bytes),{ktx2Loader:decoder});
  assert.equal(g.writes.length,levels);assert.equal(g.textures[0].descriptor.mipLevelCount,levels);assert.equal(g.passes.length,0);r.dispose();
});
test('compressed storage and worst-case expansion have separate budgets',async()=>{
  const bytes=syntheticKtx2(),decoder=retainedDecoderDouble(),g=gpu();
  // 8x8 + 4x4 + 2x2 + 1x1: RGBA expansion 340 bytes, BC7 GPU storage 112.
  const r=await createGltfTextureResources(g.device,[request()],async()=>image(bytes),{ktx2Loader:decoder,maxTextureBytes:112,maxTranscodeBytes:340});
  assert.equal(r.textureBytes,112);r.dispose();
  const before=retainedDecoderDouble();await assert.rejects(createGltfTextureResources(gpu().device,[request()],async()=>image(bytes),{ktx2Loader:before,maxTranscodeBytes:339}),{code:'GLTF_KTX2_LIMIT'});assert.equal(before.calls.length,0);
  const after=retainedDecoderDouble(),h=gpu();await assert.rejects(createGltfTextureResources(h.device,[request()],async()=>image(bytes),{ktx2Loader:after,maxTextureBytes:111}),{code:'GLTF_TEXTURE_LIMIT'});
  assert.equal(after.textures[0].disposed,1);assert.equal(h.textures.length,0);
});
test('aggregate GPU budget includes distinct KTX2 and core images and rolls prior allocations back',async()=>{
  for(const coreFirst of [false,true]) {
    const g=gpu(),decoder=retainedDecoderDouble();let closed=0;
    const provider=async i=>(i===0)===coreFirst?{bytes:png,mimeType:'image/png'}:image(syntheticKtx2());
    await assert.rejects(createGltfTextureResources(g.device,[request(),request(1,1)],provider,{ktx2Loader:decoder,maxTextureBytes:115,
      createImageBitmap:async()=>({width:1,height:1,close(){closed++;}})}),{code:'GLTF_TEXTURE_LIMIT'});
    assert.equal(g.textures.length,1);assert.equal(g.textures[0].destroyed,1);assert.equal(decoder.textures[0].disposed,1);assert.equal(closed,coreFirst?1:0);
  }
});
test('successful mixed PNG/KTX2 upload keeps each transfer path and shares identical samplers',async()=>{
  const g=gpu(),decoder=retainedDecoderDouble();let closed=0;
  const r=await createGltfTextureResources(g.device,[request(),request(1,1)],async i=>i?image(syntheticKtx2()):{bytes:png,mimeType:'image/png'},
    {ktx2Loader:decoder,createImageBitmap:async()=>({width:1,height:1,close(){closed++;}})});
  assert.equal(closed,1);assert.equal(g.external.length,1);assert.equal(g.writes.length,4);assert.equal(r.textureBytes,116);
  assert.equal(r.resolveTexture(request()).sampler,r.resolveTexture(request(1,1)).sampler);r.dispose();
});
test('color-space conflicts, invalid primaries, dimensions and unsupported features reject without GPU effects',async()=>{
  for(const settings of ['conflict','primaries','dimensions','features']) {
    const g=gpu(),decoder=retainedDecoderDouble(),bytes=syntheticKtx2({colorSpace:settings==='primaries'?'linear':'srgb'});
    let requests=[request()],code='GLTF_TEXTURE_COLOR';
    if(settings==='conflict')requests.push(request(1,0,'linear'));
    if(settings==='primaries'){bytes[new DataView(bytes.buffer).getUint32(48,true)+13]=1;requests=[request(0,0,'linear')];}
    if(settings==='dimensions'){g.device.limits.maxTextureDimension2D=4;code='GLTF_KTX2_LIMIT';}
    if(settings==='features'){g.device.features.clear();code='GLTF_KTX2_FEATURE';}
    await assert.rejects(createGltfTextureResources(g.device,requests,async()=>image(bytes),{ktx2Loader:decoder}),{code});assert.equal(g.textures.length,0);
    assert.equal(decoder.calls.length,settings==='features'?1:0);
  }
});
test('missing decoder, missing write API and malformed KTX2 fail rather than interpreting bytes as pixels',async()=>{
  const bytes=syntheticKtx2();
  await assert.rejects(createGltfTextureResources(gpu().device,[request()],async()=>image(bytes)),{code:'GLTF_KTX2_DECODER'});
  const g=gpu();delete g.device.queue.writeTexture;
  await assert.rejects(createGltfTextureResources(g.device,[request()],async()=>image(bytes),{ktx2Loader:retainedDecoderDouble()}),{code:'GLTF_TEXTURE_DEVICE'});
  await assert.rejects(createGltfTextureResources(gpu().device,[request()],async()=>image(png),{ktx2Loader:retainedDecoderDouble()}),{code:'GLTF_KTX2_HEADER'});
});
test('GPU validation, write failures, view failures and queue rejection release all owned allocations',async()=>{
  for(const mode of ['validation','write','view','queue']) {
    const g=gpu({errorAt:mode==='validation'?1:0,throwAt:mode==='write'?2:0}),decoder=retainedDecoderDouble();
    if(mode==='view'){const create=g.device.createTexture;g.device.createTexture=d=>{const t=create(d);t.createView=()=>{throw new Error('view failed');};return t;};}
    if(mode==='queue')g.device.queue.onSubmittedWorkDone=async()=>{throw new Error('queue failed');};
    await assert.rejects(createGltfTextureResources(g.device,[request()],async()=>image(syntheticKtx2()),{ktx2Loader:decoder}));
    assert.equal(g.textures[0].destroyed,1);assert.equal(decoder.textures[0].disposed,1);assert.deepEqual(g.scopes,[]);
  }
});
test('abort during retained decode rejects promptly and disposes late results without uploading',async()=>{
  const g=gpu(),c=new AbortController();let deliver;
  const decoder=retainedDecoderDouble({onParse({texture,onLoad}){deliver=()=>onLoad(texture);}});
  const pending=createGltfTextureResources(g.device,[request()],async()=>image(syntheticKtx2()),{ktx2Loader:decoder,signal:c.signal});
  await tick();c.abort();await assert.rejects(pending,{name:'AbortError'});assert.equal(g.textures.length,0);
  deliver();assert.equal(decoder.textures[0].disposed,1);assert.equal(g.writes.length,0);
});
test('device loss during retained decode rejects promptly and frees previous textures plus a late native result',async()=>{
  const g=gpu();let deliver;
  const decoder=retainedDecoderDouble({onParse({texture,onLoad}){if(deliver===undefined){deliver=null;onLoad(texture);}else deliver=()=>onLoad(texture);}});
  const pending=createGltfTextureResources(g.device,[request(),request(1,1)],async()=>image(syntheticKtx2()),{ktx2Loader:decoder});
  await tick();assert.equal(typeof deliver,'function');g.lose({message:'device removed'});
  await assert.rejects(pending,{code:'GLTF_TEXTURE_DEVICE_LOST'});assert.equal(g.textures[0].destroyed,1);
  deliver();assert.equal(decoder.textures[1].disposed,1);assert.deepEqual(g.scopes,[]);
});
test('device loss after publication invalidates resolution and disposal stays idempotent',async()=>{
  const g=gpu(),r=await createGltfTextureResources(g.device,[request()],async()=>image(syntheticKtx2()),{ktx2Loader:retainedDecoderDouble()});
  g.lose({message:'lost'});await tick();assert.equal(r.failed,true);assert.equal(r.textureBytes,0);
  assert.throws(()=>r.resolveTexture(request()),{code:'GLTF_TEXTURE_DEVICE_LOST'});r.dispose();assert.equal(g.textures[0].destroyed,1);
});
test('pre-abort and invalid options have no provider or decoder side effects',async()=>{
  const c=new AbortController();c.abort();
  await assert.rejects(createGltfTextureResources(null,[request()],()=>assert.fail('No read'),{signal:c.signal}),{name:'AbortError'});
  await assert.rejects(createGltfTextureResources(null,[],()=>assert.fail('No read'),{maxTranscodeBytes:0}),{code:'GLTF_TEXTURE_LIMIT'});
});
test('a retained decoder promise rejection is observed and closes prior images',async()=>{
  const g=gpu();let calls=0;
  const decoder=retainedDecoderDouble({onParse({texture,onLoad}){if(calls++)return Promise.reject(new Error('worker rejected'));onLoad(texture);}});
  await assert.rejects(createGltfTextureResources(g.device,[request(),request(1,1)],async()=>image(syntheticKtx2()),{ktx2Loader:decoder}),/worker rejected/);
  assert.equal(g.textures[0].destroyed,1);
});
