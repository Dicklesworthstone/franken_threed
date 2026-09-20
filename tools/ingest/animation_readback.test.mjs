import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationReadback} from './animation_readback.mjs';
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
// Native WebGPU boundary double. Production copy descriptors and unpacking run.
// This is not evidence of a browser, driver, WGSL execution or rendered pixels.
function gpu({hold=false,scopeHold=false,invalid=null,failAt=null}={}){
  const buffers=[],copies=[],submissions=[],scopes=[],events=[],lost=deferred(),scopeGate=deferred();
  let fail=failAt;
  const raise=name=>{events.push(name);if(fail===name)throw Error('failure '+name);};
  const device={limits:{maxBufferSize:1024*1024},lost:lost.promise,
    destroy(){assert.fail('Borrowed device destroyed');},
    pushErrorScope(type){raise('push');scopes.push(type);},
    popErrorScope(){raise('pop');assert.ok(scopes.length);const type=scopes.pop();return scopeHold?scopeGate.promise:Promise.resolve(type==='validation'?invalid:null);},
    createBuffer(desc){raise('allocate');const gate=deferred();let storage=new ArrayBuffer(desc.size),mapped=false;
      const b={desc,gate,destroyed:0,get storage(){return storage;},
        mapAsync(mode,offset,size){raise('map');assert.equal(mode,1);assert.equal(offset,0);assert.equal(size,desc.size);return (hold?gate.promise:Promise.resolve()).then(()=>{mapped=true;});},
        getMappedRange(offset,size){raise('range');assert.ok(mapped);assert.equal(offset,0);assert.equal(size,desc.size);return storage;},
        destroy(){this.destroyed++;mapped=false;structuredClone(storage,{transfer:[storage]});},
      };buffers.push(b);return b;},
    createCommandEncoder(){raise('encoder');const commands=[];return {
      copyTextureToBuffer(source,destination,size){raise('copy');assert.equal(destination.bytesPerRow%256,0);assert.equal(destination.offset,0);assert.equal(source.aspect,'all');commands.push({source,destination,size});copies.push(commands.at(-1));},
      finish(){raise('finish');return commands;},
    };},
    queue:{submit(batch){raise('submit');assert.ok(scopes.length);submissions.push(batch);
      for(const commands of batch)for(const {source:s,destination:d,size}of commands){
        const bpp=s.texture.bpp,width=Math.max(1,Math.floor(s.texture.width/2**s.mipLevel));
        const data=s.texture.getBytes?.(s.mipLevel,s.origin[2]) ?? s.texture.bytes;
        const dest=new Uint8Array(d.buffer.storage);dest.fill(0xed);
        for(let y=0;y<size[1];y++){
          const start=((s.origin[1]+y)*width+s.origin[0])*bpp;
          dest.set(data.subarray(start,start+size[0]*bpp),y*d.bytesPerRow);
        }
      }
    },onSubmittedWorkDone(){assert.fail('Must not drain unrelated queue work');}},
  };
  return {device,buffers,copies,submissions,scopes,events,lost,scopeGate,setFailure(value){fail=value;}};
}
function texture(width=3,height=2,format='rgba8unorm'){
  const bpp=format==='rgba16float'?8:format==='rgba32float'?16:4;
  return {width,height,format,bpp,dimension:'2d',sampleCount:1,depthOrArrayLayers:1,mipLevelCount:1,usage:1|16,
    bytes:Uint8Array.from({length:width*height*bpp},(_,i)=>i%251),destroy(){assert.fail('Borrowed texture destroyed');}};
}
function clean(g,r){assert.equal(r.pending,0);assert.equal(r.bufferBytes,0);assert.equal(r.reservedBytes,0);assert.equal(g.scopes.length,0);for(const b of g.buffers)assert.equal(b.destroyed,1);}

test('copy is submitted synchronously; padded mapped storage becomes independent tight RGBA',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device),t=texture();
  const expected=t.bytes.slice(),pending=r.readPixels(t);
  assert.equal(g.submissions.length,1);assert.equal(g.scopes.length,0);assert.equal(r.pending,1);
  assert.equal(r.bufferBytes,512);assert.equal(r.reservedBytes,536);assert.equal(g.buffers[0].desc.usage,9);
  t.bytes.fill(99);g.buffers[0].gate.resolve();const result=await pending;
  assert.deepEqual(result.data,expected);assert.equal(result.bytesPerRow,12);assert.equal(result.channels,'rgba');assert.equal(result.srgb,false);
  assert.equal(result.componentType,'unorm8');assert.ok(Object.isFrozen(result));assert.ok(Object.isFrozen(result.origin));
  assert.equal(g.buffers[0].storage.byteLength,0);assert.equal(result.data.byteLength,24);clean(g,r);r.dispose();
});
for(const format of ['rgba8unorm','rgba8unorm-srgb','bgra8unorm','bgra8unorm-srgb'])test(format+' gives RGBA without gamma/alpha conversion',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),t=texture(1,1,format);t.bytes.set([2,13,251,47]);
  const a=await r.readPixels(t);assert.deepEqual([...a.data],format.startsWith('bgra')?[251,13,2,47]:[2,13,251,47]);
  assert.equal(a.srgb,format.endsWith('-srgb'));assert.equal(a.sourceFormat,format);clean(g,r);
});
test('subrect defaults, selected mip/layer and flipped row order',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),t=texture(10,6);t.mipLevelCount=3;t.depthOrArrayLayers=4;
  const bytes=Uint8Array.from({length:5*3*4},(_,i)=>i);t.getBytes=(mip,layer)=>{assert.equal(mip,1);assert.equal(layer,3);return bytes;};
  const opts={x:2,y:1,mipLevel:1,layer:3,flipY:true};const p=r.readPixels(t,opts);opts.x=9;
  const result=await p;assert.deepEqual([...result.data],[...bytes.slice(48,60),...bytes.slice(28,40)]);
  assert.equal(result.width,3);assert.equal(result.height,2);assert.deepEqual(result.origin,[2,1,3]);
  assert.deepEqual(g.copies[0].size,[3,2,1]);assert.equal(result.mipLevel,1);clean(g,r);
});
test('half-float HDR conversion preserves signed zero, subnormals, infinities and NaNs',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),t=texture(3,1,'rgba16float');
  const words=[0,0x8000,0x3c00,0xc000,1,0x3ff,0x400,0x7bff,0x7c00,0xfc00,0x7e00,0x3800];
  const v=new DataView(t.bytes.buffer);words.forEach((x,i)=>v.setUint16(i*2,x,true));
  const a=await r.readPixels(t);assert.ok(a.data instanceof Float32Array);assert.equal(a.bytesPerRow,48);
  const expected=[0,-0,1,-2,2**-24,1023*2**-24,2**-14,65504,Infinity,-Infinity,NaN,.5];
  expected.forEach((x,i)=>assert.ok(Object.is(a.data[i],x),`${i}: ${a.data[i]}`));clean(g,r);
});
test('float32 HDR uses little-endian source data, without clipping or normalization',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),t=texture(1,2,'rgba32float');
  const values=[-.25,123456,Infinity,0,NaN,-0,1,-Infinity],v=new DataView(t.bytes.buffer);
  values.forEach((x,i)=>v.setFloat32(i*4,x,true));const a=await r.readPixels(t,{flipY:true});
  assert.deepEqual([...a.data],[...values.slice(4),...values.slice(0,4)]);clean(g,r);
});
for(const [name,patch,options,code]of [
  ['depth',{format:'depth32float'},{},'FORMAT'],['compressed',{format:'bc1-rgba-unorm'},{},'FORMAT'],
  ['multisample',{sampleCount:4},{},'TEXTURE'],['missing copy usage',{usage:16},{},'TEXTURE'],['3d',{dimension:'3d'},{},'TEXTURE'],
  ['negative x',{}, {x:-1},'RANGE'],['fractional width',{}, {width:1.5},'RANGE'],['zero width',{}, {width:0},'RANGE'],
  ['beyond right',{}, {x:2,width:2},'RANGE'],['beyond bottom',{}, {y:1,height:2},'RANGE'],
  ['bad mip',{}, {mipLevel:1},'RANGE'],['bad layer',{}, {layer:1},'RANGE'],
  ['unknown option',{}, {pretend:true},'OPTIONS'],['bad signal',{}, {signal:{}},'OPTIONS'],['flip string',{}, {flipY:'true'},'OPTIONS'],
])test('rejects '+name+' before allocation/submission',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device);await assert.rejects(r.readPixels(Object.assign(texture(),patch),options),{code:'ANIMATION_READBACK_'+code});
  assert.equal(g.events.length,0);clean(g,r);
});
test('per-device staging and aggregate transient-memory limits precede effects',async()=>{
  const g=gpu();g.device.limits.maxBufferSize=256;const r=createGpuAnimationReadback(g.device);
  await assert.rejects(r.readPixels(texture()),{code:'ANIMATION_READBACK_LIMIT'});assert.equal(g.events.length,0);
  const g2=gpu({hold:true}),r2=createGpuAnimationReadback(g2.device,{maxBytes:1071}),t=texture();
  const first=r2.readPixels(t);await assert.rejects(r2.readPixels(t),{code:'ANIMATION_READBACK_LIMIT'});
  assert.equal(g2.buffers.length,1);g2.buffers[0].gate.resolve();await first;
  const second=r2.readPixels(t);g2.buffers[1].gate.resolve();await second;clean(g2,r2);
});
test('bounded simultaneous requests retain their own frame versions and finish out of order',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device,{maxPending:2}),t=texture(1,1);
  t.bytes.fill(17);const a=r.readPixels(t);t.bytes.fill(29);const b=r.readPixels(t);t.bytes.fill(43);
  await assert.rejects(r.readPixels(t),{code:'ANIMATION_READBACK_LIMIT'});
  g.buffers[1].gate.resolve();assert.deepEqual([...(await b).data],[29,29,29,29]);assert.equal(r.pending,1);
  g.buffers[0].gate.resolve();assert.deepEqual([...(await a).data],[17,17,17,17]);clean(g,r);
});
test('already aborted requests have no effects',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),c=new AbortController(),reason=Error('stop');c.abort(reason);
  await assert.rejects(r.readPixels(texture(),{signal:c.signal}),e=>e===reason);assert.equal(g.events.length,0);clean(g,r);
});
for(const scopeHold of [false,true])test('abort promptly destroys owned staging even with unresolved '+(scopeHold?'validation':'mapping'),async()=>{
  const g=gpu({hold:true,scopeHold}),r=createGpuAnimationReadback(g.device),c=new AbortController(),reason=Error('cancel');
  const p=r.readPixels(texture(),{signal:c.signal});c.abort(reason);await assert.rejects(p,e=>e===reason);clean(g,r);
  g.buffers[0].gate.reject(Error('late map'));g.scopeGate.reject(Error('late scope'));g.scopeGate.promise.catch(()=>{});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(r.failed,false);
});
test('dispose cancels all pending reads and does not invalidate completed snapshots',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device);const a=r.readPixels(texture());g.buffers[0].gate.resolve();const snapshot=await a;
  const b=r.readPixels(texture()),c=r.readPixels(texture());r.dispose();r.dispose();
  await assert.rejects(b,{code:'ANIMATION_READBACK_DISPOSED'});await assert.rejects(c,{code:'ANIMATION_READBACK_DISPOSED'});
  assert.equal(snapshot.data.length,24);assert.equal(r.disposed,true);clean(g,r);
  await assert.rejects(r.readPixels(texture()),{code:'ANIMATION_READBACK_DISPOSED'});
});
test('device loss rejects all in-flight and subsequent reads without destroying borrowed resources',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device),a=r.readPixels(texture()),b=r.readPixels(texture());
  g.lost.resolve({message:'test loss'});await assert.rejects(a,{code:'ANIMATION_READBACK_DEVICE_LOST'});await assert.rejects(b,{code:'ANIMATION_READBACK_DEVICE_LOST'});
  assert.equal(r.failed,true);await assert.rejects(r.readPixels(texture()),{code:'ANIMATION_READBACK_DEVICE_LOST'});clean(g,r);
});
for(const failAt of ['allocate','encoder','copy','finish','submit','map','range'])test(failAt+' failure releases staging, closes scopes, and allows retry',async()=>{
  const g=gpu({failAt}),r=createGpuAnimationReadback(g.device);await assert.rejects(r.readPixels(texture()),new RegExp('failure '+failAt));clean(g,r);
  g.setFailure(null);await r.readPixels(texture());clean(g,r);assert.equal(r.failed,false);
});
test('native map rejection is observed and destroys staging',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device),p=r.readPixels(texture());g.buffers[0].gate.reject(Error('mapping rejected'));
  await assert.rejects(p,/mapping rejected/);clean(g,r);
});
test('validation failure rejects rather than reporting successful zero-filled pixels',async()=>{
  const g=gpu({invalid:{message:'foreign device texture'}}),r=createGpuAnimationReadback(g.device);
  await assert.rejects(r.readPixels(texture()),{code:'ANIMATION_READBACK_GPU'});clean(g,r);assert.equal(g.events.includes('range'),false);
});
test('whenIdle drains only requests present at call and tolerates their cancellation',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device),c=new AbortController(),a=r.readPixels(texture(),{signal:c.signal});
  let drained=false;const idle=r.whenIdle().then(value=>{drained=true;assert.equal(value,r);});
  const b=r.readPixels(texture());c.abort();await assert.rejects(a,{name:'AbortError'});await idle;
  assert.equal(drained,true);assert.equal(r.pending,1);g.buffers[0].gate.resolve();g.buffers[1].gate.resolve();await b;clean(g,r);
});
test('validation is fully observed before a mapped result can escape',async()=>{
  const g=gpu({scopeHold:true}),r=createGpuAnimationReadback(g.device);let done=false;
  const pending=r.readPixels(texture()).then(v=>{done=true;return v;});await Promise.resolve();await Promise.resolve();assert.equal(done,false);
  g.scopeGate.resolve(null);await pending;clean(g,r);
});
for(const options of [{maxBytes:0},{maxPending:0},{maxPending:65},{label:42},{unknown:1}])test('invalid service options '+JSON.stringify(options),()=>{
  const g=gpu();assert.throws(()=>createGpuAnimationReadback(g.device,options));assert.equal(g.events.length,0);
});
test('pure preflight has no effects and repeats current capacity checks',async()=>{
  const g=gpu({hold:true}),r=createGpuAnimationReadback(g.device,{maxPending:1}),t=texture();
  assert.equal(r.validate(t),undefined);assert.equal(g.events.length,0);assert.equal(r.pending,0);
  const p=r.readPixels(t);assert.throws(()=>r.validate(t),{code:'ANIMATION_READBACK_LIMIT'});
  g.buffers[0].gate.resolve();await p;r.validate(t);clean(g,r);
});
test('producer validation retains the admission reservation after native mapping completes',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device,{maxPending:1}),completion=deferred();let done=false;
  const p=r.readPixels(texture(),{completion:completion.promise}).then(v=>{done=true;return v;});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(done,false);assert.equal(r.pending,1);assert.equal(r.reservedBytes,536);
  await assert.rejects(r.readPixels(texture()),{code:'ANIMATION_READBACK_LIMIT'});completion.resolve();await p;clean(g,r);
});
test('abort still releases a mapped request waiting on producer validation',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),completion=deferred(),c=new AbortController();
  const p=r.readPixels(texture(),{completion:completion.promise,signal:c.signal});await new Promise(resolve=>setImmediate(resolve));
  c.abort();await assert.rejects(p,{name:'AbortError'});clean(g,r);completion.reject(Error('late producer failure'));
});
test('failed producer validation does not publish pixels',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device),completion=deferred();
  const p=r.readPixels(texture(),{completion:completion.promise});completion.reject(Error('render invalid'));
  await assert.rejects(p,/render invalid/);assert.equal(g.events.includes('range'),false);clean(g,r);
});
test('invalid producer validation is refused before GPU effects',async()=>{
  const g=gpu(),r=createGpuAnimationReadback(g.device);await assert.rejects(r.readPixels(texture(),{completion:123}),{code:'ANIMATION_READBACK_OPTIONS'});assert.equal(g.events.length,0);
});
