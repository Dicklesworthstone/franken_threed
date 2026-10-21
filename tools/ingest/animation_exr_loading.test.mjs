import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import * as exr from './animation_exr.mjs';

// Execute unchanged production loader source with its actual EXR decoder.
// GPU calls, the pre-existing IBL filter and RGBE decoder are explicit boundaries.
// The package test executes the production AST closure with the real Acorn parser.
if(!process.execArgv.includes('--experimental-vm-modules')) {
  test('EXR environment loading and relocated module closure',()=>{
    const env={...process.env};delete env.NODE_TEST_CONTEXT;
    const text=execFileSync(process.execPath,['--experimental-vm-modules','--test','--test-reporter=tap',fileURLToPath(import.meta.url)],
      {env,encoding:'utf8',timeout:20000,stdio:['ignore','pipe','pipe']});
    assert.match(text,/# fail 0/);assert.match(text,/# pass [1-9]\d*/);
  });
} else {
  const vm=await import('node:vm');
  const loaderUrl=new URL('./animation_environment_loader.mjs',import.meta.url),source=fs.readFileSync(loaderUrl,'utf8');
  const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
  class StageError extends Error{constructor(code,message){super(message);this.code=code;}}
  const synthetic=values=>new vm.SyntheticModule(Object.keys(values),function(){for(const [k,v]of Object.entries(values))this.setExport(k,v);});
  const u32=n=>{const b=Buffer.alloc(4);b.writeUInt32LE(n);return b;};
  const f32=n=>{const b=Buffer.alloc(4);b.writeFloatLE(n);return b;};
  const str=s=>Buffer.from(s+'\0');
  const attribute=(n,t,d)=>Buffer.concat([str(n),str(t),u32(d.length),d]);
  function panorama({w=4,h=2,crop=false,cube=false,type=1,overflow=false}={}) {
    const names=['B','G','R'],channels=Buffer.concat([...names.map(n=>Buffer.concat([str(n),u32(type),Buffer.alloc(4),u32(1),u32(1)])),Buffer.of(0)]);
    const header=Buffer.concat([u32(20000630),u32(2),attribute('channels','chlist',channels),attribute('compression','compression',Buffer.of(0)),
      attribute('dataWindow','box2i',Buffer.concat([0,0,w-1,h-1].map(u32))),
      attribute('displayWindow','box2i',Buffer.concat([0,0,w-1+(crop?2:0),h-1].map(u32))),
      attribute('lineOrder','lineOrder',Buffer.of(0)),attribute('pixelAspectRatio','float',f32(1)),
      attribute('screenWindowCenter','v2f',Buffer.alloc(8)),attribute('screenWindowWidth','float',f32(1)),
      ...(cube?[attribute('envmap','envmap',Buffer.of(1))]:[]),Buffer.of(0)]);
    const table=Buffer.alloc(h*8),blocks=[];let offset=header.length+table.length;
    for(let y=0;y<h;y++){
      const data=Buffer.alloc(w*3*(type===1?2:4));let at=0;
      for(const c of names)for(let x=0;x<w;x++){
        if(type===1){data.writeUInt16LE({R:0x3c00,G:0x3800,B:0x3400}[c],at);at+=2;}
        else {data.writeFloatLE(overflow?70000:{R:1,G:.5,B:.25}[c],at);at+=4;}
      }
      table.writeBigUInt64LE(BigInt(offset),y*8);const block=Buffer.concat([u32(y),u32(data.length),data]);blocks.push(block);offset+=block.length;
    }
    return Buffer.concat([header,table,...blocks]);
  }
  async function setup({pendingFilter,filterError,uploadError,validationError,decoder=exr}={}) {
    const calls={hdr:0,filter:[],fetch:[]},textures=[],uploads=[],lost=deferred();let depth=0;
    let disposed=false;
    // Match the real factory's closure-owned lifetime, not a this-bound class.
    const map={get disposed(){return disposed;},failed:false,textureBytes:128,
      sample(device){assert.equal(device,d);if(disposed)throw Error('disposed');return map;},
      async whenIdle(){return map;},dispose(){disposed=true;}};
    const d={limits:{maxTextureDimension2D:4096,minUniformBufferOffsetAlignment:256,maxBufferSize:65536},lost:lost.promise,
      queue:{writeTexture(target,pixels,layout,size){uploads.push({target,pixels:pixels.slice(),layout,size});if(uploadError)throw uploadError;},writeBuffer(){},submit(){},onSubmittedWorkDone:async()=>{}},
      createTexture(desc){const t={...desc,destroyed:0,createView(){return {texture:this};},destroy(){this.destroyed++;}};textures.push(t);return t;},
      pushErrorScope(){depth++;},popErrorScope(){depth--;return Promise.resolve(validationError ?? null);}};
    for(const key of ['createBuffer','createSampler','createBindGroupLayout','createPipelineLayout','createShaderModule','createBindGroup','createCommandEncoder'])d[key]=()=>({});
    d.createRenderPipelineAsync=async()=>({});
    const deps={
      './animation_exr.mjs':decoder,
      './animation_hdr.mjs':{AnimationHdrError:StageError,decodeAnimationHdr(input){calls.hdr++;if(input[0]!==35)throw new StageError('RGBE_SIGNATURE','Expected HDR');
        return {width:4,height:2,byteLength:64,data:new Uint16Array(32),exposure:1,clampedComponents:0};}},
      './animation_environment.mjs':{AnimationEnvironmentError:StageError,planAnimationEnvironment:()=>({uniformBytes:256,textureBytes:128}),
        createGpuAnimationEnvironment:async(device,texture,options)=>{assert.equal(depth,0);assert.equal(texture.destroyed,0);calls.filter.push({texture,options});
          if(filterError)throw filterError;if(pendingFilter)return pendingFilter.promise;return map;}},
    };
    const module=new vm.SourceTextModule(source,{identifier:loaderUrl.href});await module.link(specifier=>{assert.ok(deps[specifier],specifier);return synthetic(deps[specifier]);});await module.evaluate();
    return {load:(input,options)=>module.namespace.loadGpuAnimationEnvironment(d,input,options),d,calls,textures,uploads,lost,map,get depth(){return depth;}};
  }
  test('real EXR bytes upload exact RGBA16F, prepare IBL and release only the temporary panorama',async()=>{
    const f=await setup(),bytes=panorama(),result=await f.load(bytes,{maxTextureBytes:192});
    assert.equal(f.calls.hdr,0);assert.equal(f.calls.filter.length,1);assert.equal(f.uploads.length,1);assert.equal(f.depth,0);
    assert.deepEqual([...f.uploads[0].pixels],Array.from({length:8},()=>[0x3c00,0x3800,0x3400,0x3c00]).flat());
    assert.deepEqual(f.uploads[0].layout,{offset:0,bytesPerRow:32,rowsPerImage:2});assert.deepEqual([...f.uploads[0].size],[4,2,1]);
    assert.equal(f.calls.filter[0].options.maxTextureBytes,128);assert.equal(f.textures[0].destroyed,1);
    assert.equal(result.sourceInfo.format,'exr');assert.equal(result.sourceInfo.compression,'none');assert.equal(result.sourceInfo.inputBytes,bytes.length);
    assert.equal(result.sourceInfo.peakTextureBytes,192);assert.equal(result.textureBytes,128);assert.equal(result.sample(f.d),f.map);
    assert.equal(await result.whenIdle(),result);result.dispose();assert.equal(result.disposed,true);
  });
  // Complete 8x4 RGBA HALF/ZIP file produced by the native OpenEXR codec,
  // not by this test's raw fixture writer. Every output row is independently known.
  const nativeZip=()=>Uint8Array.from(Buffer.from('di8xAQIAAABjaGFubmVscwBjaGxpc3QASQAAAEEAAQAAAAAAAAABAAAAAQAAAEIAAQAAAAAAAAABAAAAAQAAAEcAAQAAAAAAAAABAAAAAQAAAFIAAQAAAAAAAAABAAAAAQAAAABjb21wcmVzc2lvbgBjb21wcmVzc2lvbgABAAAAA2RhdGFXaW5kb3cAYm94MmkAEAAAAAAAAAAAAAAABwAAAAMAAABkaXNwbGF5V2luZG93AGJveDJpABAAAAAAAAAAAAAAAAcAAAADAAAAbGluZU9yZGVyAGxpbmVPcmRlcgABAAAAAHBpeGVsQXNwZWN0UmF0aW8AZmxvYXQABAAAAAAAgD9zY3JlZW5XaW5kb3dDZW50ZXIAdjJmAAgAAAAAAAAAAAAAAHNjcmVlbldpbmRvd1dpZHRoAGZsb2F0AAQAAAAAAIA/AFMBAAAAAAAAAAAAAB8AAAB4nGNoGFiwC0pXQekWKO2xo6WpqbGxsZzG8gDgUX/E','base64'));
  test('native ZIP panorama survives immediate caller transfer and uploads every exact half component',async()=>{
    const f=await setup(),bytes=nativeZip(),length=bytes.length,pending=f.load(bytes.buffer);
    structuredClone(bytes.buffer,{transfer:[bytes.buffer]});assert.equal(bytes.byteLength,0);
    const result=await pending,red=[0,0x3800,0x3c00,0x3e00,0x4000,0x4100,0x4200,0x4300];
    assert.equal(result.sourceInfo.inputBytes,length);assert.equal(result.sourceInfo.compression,'zip');
    const expected=Array.from({length:4},()=>red.flatMap(r=>[r,0x3800,0x3400,0x3a00])).flat();
    assert.deepEqual([...f.uploads[0].pixels],expected);assert.deepEqual([...f.uploads[0].size],[8,4,1]);
    assert.equal(f.calls.hdr,0);assert.equal(f.textures[0].destroyed,1);result.dispose();
  });
  test('caller abort interrupts real ZIP decompression before upload without switching decoders',async()=>{
    const f=await setup(),c=new AbortController(),reason=new Error('cancel ZIP'),pending=f.load(nativeZip(),{signal:c.signal});
    c.abort(reason);await assert.rejects(pending,e=>e===reason);
    assert.equal(f.calls.hdr,0);assert.equal(f.textures.length,0);assert.equal(f.calls.filter.length,0);
  });
  test('corrupt compressed EXR rejects rather than falling through to RGBE or creating a placeholder',async()=>{
    const f=await setup(),data=nativeZip();data[data.length-1]^=255;
    await assert.rejects(f.load(data),{code:'ANIMATION_EXR_DECOMPRESS'});
    assert.equal(f.calls.hdr,0);assert.equal(f.textures.length,0);
  });
  test('format detection follows bytes, not a misleading URL extension or content type',async()=>{
    const f=await setup(),bytes=panorama(),requests=[];
    const result=await f.load('./misnamed.hdr#view',{baseURL:'https://assets.test/a/',fetch:async(url,settings)=>{
      requests.push({url,settings});return new Response(new ReadableStream({start(c){for(let i=0;i<bytes.length;i+=7)c.enqueue(bytes.subarray(i,i+7));c.close();}}),{headers:{'content-type':'image/png'}});}});
    assert.equal(result.sourceInfo.format,'exr');assert.equal(requests[0].url,'https://assets.test/a/misnamed.hdr');
    assert.equal(requests[0].settings.credentials,'omit');assert.equal(requests[0].settings.redirect,'error');assert.equal(f.calls.hdr,0);result.dispose();
  });
  test('explicit format selection is strict and invalid options fail before fetching',async()=>{
    const f=await setup();await assert.rejects(f.load(panorama(),{format:'hdr'}),{code:'RGBE_SIGNATURE'});
    await assert.rejects(f.load(Uint8Array.of(35,63,82),{format:'exr'}),{code:'ANIMATION_EXR_HEADER'});
    let calls=0;await assert.rejects(f.load('https://test/a',{format:'guess',fetch:()=>{calls++;}}),{code:'ANIMATION_HDR_OPTIONS'});assert.equal(calls,0);
    assert.equal(f.uploads.length,0);
  });
  test('the existing Radiance route and its sourceInfo shape remain unchanged',async()=>{
    const f=await setup(),result=await f.load(Uint8Array.of(35,63,82));assert.equal(f.calls.hdr,1);
    assert.deepEqual({...result.sourceInfo},{width:4,height:2,inputBytes:3,exposure:1,clampedComponents:0,uploadTextureBytes:64,peakTextureBytes:192});result.dispose();
  });
  for(const [name,settings]of [['nonpanorama',{w:3}],['cropped',{crop:true}],['cubemap',{cube:true}]])
    test(`${name} EXR rejects before decoding pixel output or GPU allocation`,async()=>{
      let decodes=0;const f=await setup({decoder:{...exr,decodeAnimationExr(...args){decodes++;return exr.decodeAnimationExr(...args);}}});
      await assert.rejects(f.load(panorama(settings)),{code:'ANIMATION_HDR_SOURCE'});assert.equal(decodes,0);assert.equal(f.textures.length,0);assert.equal(f.calls.hdr,0);
    });
  test('input, decoded and peak GPU budgets reject without decoder fallback or allocation',async()=>{
    const bytes=panorama(),meta=exr.inspectAnimationExr(bytes);
    for(const settings of [{maxInputBytes:bytes.length-1},{maxDecodedBytes:meta.peakDecodedBytes-1},{maxHeaderBytes:32},{maxTextureBytes:191}]){
      const f=await setup();await assert.rejects(f.load(bytes,settings));assert.equal(f.textures.length,0);assert.equal(f.calls.hdr,0);
    }
  });
  test('explicit clamping reports affected source radiance without exposure compensation',async()=>{
    const f=await setup(),bytes=panorama({type:2,overflow:true});await assert.rejects(f.load(bytes),{code:'ANIMATION_EXR_VALUE'});
    const result=await f.load(bytes,{overflow:'clamp'});assert.equal(result.sourceInfo.clampedComponents,24);assert.equal(f.uploads[0].pixels[0],0x7bff);result.dispose();
  });
  for(const kind of ['uploadError','validationError','filterError'])test(`${kind} releases every allocated temporary texture`,async()=>{
    const error=new Error(kind),f=await setup({[kind]:error});await assert.rejects(f.load(panorama()));
    assert.equal(f.textures.length,1);assert.equal(f.textures[0].destroyed,1);assert.equal(f.depth,0);assert.equal(f.map.disposed,false);
  });
  test('abort or device loss during decompression rejects with original reason and no GPU work',async()=>{
    for(const loss of [false,true]){
      const pending=deferred();let signal;const f=await setup({decoder:{...exr,decodeAnimationExr(input,options){signal=options.signal;return pending.promise;}}});
      const c=new AbortController(),reason=new Error('cancel'),promise=f.load(panorama(),{signal:c.signal});
      if(loss)f.lost.resolve({message:'lost'});else c.abort(reason);
      await assert.rejects(promise,e=>loss?e.code==='ANIMATION_ENVIRONMENT_LOST':e===reason);assert.equal(signal.aborted,true);assert.equal(f.textures.length,0);
      pending.resolve(await exr.decodeAnimationExr(panorama()));
    }
  });
  test('cancellation during filtering releases upload and disposes a late map',async()=>{
    const filter=deferred(),f=await setup({pendingFilter:filter}),c=new AbortController();const pending=f.load(panorama(),{signal:c.signal});
    while(!f.calls.filter.length)await new Promise(r=>setImmediate(r));c.abort();await assert.rejects(pending,{name:'AbortError'});
    assert.equal(f.textures[0].destroyed,1);filter.resolve(f.map);await new Promise(r=>setImmediate(r));assert.equal(f.map.disposed,true);
  });
  test('construction signal does not revoke a successfully returned environment',async()=>{
    const f=await setup(),c=new AbortController(),result=await f.load(panorama(),{signal:c.signal});c.abort();
    assert.equal(result.disposed,false);await result.whenIdle();result.dispose();
  });
  test('stream limits and abort cancel outstanding body reads',async()=>{
    for(const aborting of [false,true]){
      const f=await setup(),c=new AbortController();let cancelled=false;
      const pending=f.load('https://test/a.exr',{signal:c.signal,maxInputBytes:400,fetch:async()=>new Response(new ReadableStream({
        start(control){if(!aborting)control.enqueue(new Uint8Array(401));},cancel(){cancelled=true;}}))});
      if(aborting){await new Promise(r=>setImmediate(r));c.abort();}
      await assert.rejects(pending);assert.equal(cancelled,true);assert.equal(f.textures.length,0);
    }
  });
  test('packaged static closure includes EXR and executes after relocation without toolkit imports',async()=>{
    const {parse}=await import('acorn'),helperSource=fs.readFileSync(new URL('./animation_package_modules.mjs',import.meta.url),'utf8');
    const helper=new vm.SourceTextModule(helperSource);await helper.link(name=>synthetic(name==='node:fs'?{default:fs}:name==='acorn'?{parse}:{AnimationPoseError:StageError}));await helper.evaluate();
    const outputs=new Map([['animation_environment_loader.mjs',source]]);
    const runtime=new Map([
      ['animation_exr.mjs',fs.readFileSync(new URL('./animation_exr.mjs',import.meta.url),'utf8')],
      ['animation_hdr.mjs',`export class AnimationHdrError extends Error{constructor(code,message){super(message);this.code=code;}}\nexport function decodeAnimationHdr(){throw Error('EXR must not use the RGBE boundary');}`],
      ['animation_environment.mjs',`export class AnimationEnvironmentError extends Error{}\nexport function planAnimationEnvironment(){return {uniformBytes:256,textureBytes:128};}\nexport async function createGpuAnimationEnvironment(){let disposed=false;return {get disposed(){return disposed;},failed:false,textureBytes:128,dispose(){disposed=true;},async whenIdle(){}};}`],
    ]);
    helper.namespace.completeAnimationRuntimeModules(outputs,name=>{assert.ok(runtime.has(name),name);return runtime.get(name);});
    assert.ok(outputs.has('animation_exr.mjs'));assert.equal(outputs.size,4);
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-exr-closure-')),out=path.join(root,'package'),deployed=path.join(root,'deployed');fs.mkdirSync(out);
    for(const [name,text]of outputs)fs.writeFileSync(path.join(out,name),text);fs.renameSync(out,deployed);
    const {loadGpuAnimationEnvironment}=await import(pathToFileURL(path.join(deployed,'animation_environment_loader.mjs'))),f=await setup();
    const result=await loadGpuAnimationEnvironment(f.d,panorama());assert.equal(result.sourceInfo.format,'exr');assert.equal(f.uploads[0].pixels[0],0x3c00);result.dispose();
  });
}
