/** Explicit WebGPU color-texture readback. No device creation, frame clock,
 * canvas, renderer substitution or global queue drain. Copies are submitted
 * before readPixels returns; a later render cannot change that snapshot.
 * WebGPU: texture-to-buffer row pitch is 256-byte aligned. The returned RGBA
 * array is tightly packed and owns its memory after staging is unmapped.
 */
export class AnimationReadbackError extends Error {
  constructor(code,message,options){super(`${code}: ${message}`,options);this.name='AnimationReadbackError';this.code=code;}
}
const error=(code,message,options)=>new AnimationReadbackError('ANIMATION_READBACK_'+code,message,options);
const fail=(code,message)=>{throw error(code,message);};
const formats=new Map([
  ['rgba8unorm',4],['rgba8unorm-srgb',4],['bgra8unorm',4],['bgra8unorm-srgb',4],
  ['rgba16float',8],['rgba32float',16],
]);
const integer=(v,label,min=0)=>{if(!Number.isSafeInteger(v)||v<min)fail('RANGE',`Invalid ${label}`);return v;};
function fields(value,keys,label){
  if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).some(key=>!keys.includes(key)))fail('OPTIONS',`Invalid ${label}`);
}
const abortReason=signal=>signal.reason ?? new DOMException('Aborted','AbortError');
function checkSignal(signal){
  if(signal!==undefined&&(signal===null||typeof signal.aborted!=='boolean'||typeof signal.addEventListener!=='function'||typeof signal.removeEventListener!=='function'))fail('OPTIONS','Expected an AbortSignal');
  if(signal?.aborted)throw abortReason(signal);
}
function half(value){
  const sign=value&0x8000?-1:1,exponent=(value>>>10)&31,fraction=value&1023;
  return exponent===31?(fraction?NaN:sign*Infinity):exponent===0?sign*fraction*2**-24:sign*(1024+fraction)*2**(exponent-25);
}
function plan(texture,options,maxBufferSize){
  fields(options,['x','y','layer','mipLevel','width','height','flipY','signal','completion'],'read options');
  checkSignal(options.signal);
  if(options.completion!==undefined&&typeof options.completion?.then!=='function')fail('OPTIONS','completion must be a producer-validation promise');
  const format=texture?.format,texelBytes=formats.get(format);
  if(!texelBytes)fail('FORMAT','Only RGBA/BGRA unorm8 and RGBA float16/float32 textures are supported');
  if(texture.dimension!=='2d'||texture.sampleCount!==1||!Number.isInteger(texture.usage)||!(texture.usage&1))fail('TEXTURE','Expected a single-sample 2D texture with COPY_SRC usage');
  const tw=integer(texture.width,'texture width',1),th=integer(texture.height,'texture height',1);
  const layers=integer(texture.depthOrArrayLayers,'texture layers',1),levels=integer(texture.mipLevelCount,'mip levels',1);
  const mipLevel=integer(options.mipLevel ?? 0,'mip level');
  if(mipLevel>=levels||mipLevel>31)fail('RANGE','Mip level is outside the texture');
  const mw=Math.max(1,Math.floor(tw/2**mipLevel)),mh=Math.max(1,Math.floor(th/2**mipLevel));
  const x=integer(options.x ?? 0,'x'),y=integer(options.y ?? 0,'y'),layer=integer(options.layer ?? 0,'layer');
  const width=integer(options.width ?? mw-x,'width',1),height=integer(options.height ?? mh-y,'height',1);
  if(x>=mw||y>=mh||width>mw-x||height>mh-y||layer>=layers)fail('RANGE','Read rectangle is outside the selected mip/layer');
  const flipY=options.flipY ?? false;if(typeof flipY!=='boolean')fail('OPTIONS','flipY must be boolean');
  const rowBytes=width*texelBytes,pitch=Math.ceil(rowBytes/256)*256,size=pitch*height;
  const floating=texelBytes>4,outputBytes=width*height*(floating?16:4),reserved=size+outputBytes;
  if(!Number.isSafeInteger(reserved)||size>maxBufferSize)fail('LIMIT','Readback exceeds the device buffer limit');
  return {format,texelBytes,x,y,layer,mipLevel,width,height,flipY,rowBytes,pitch,size,floating,outputBytes,reserved};
}
function unpack(buffer,p){
  const view=new DataView(buffer),source=new Uint8Array(buffer),data=p.floating?new Float32Array(p.width*p.height*4):new Uint8Array(p.outputBytes);
  const bgra=p.format.startsWith('bgra');
  for(let y=0;y<p.height;y++){
    const from=y*p.pitch,to=(p.flipY?p.height-1-y:y)*p.width*4;
    if(!p.floating){
      data.set(source.subarray(from,from+p.rowBytes),to);
      if(bgra)for(let x=0;x<p.width;x++){const i=to+x*4,b=data[i];data[i]=data[i+2];data[i+2]=b;}
    }else for(let c=0;c<p.width*4;c++)data[to+c]=p.texelBytes===8?half(view.getUint16(from+c*2,true)):view.getFloat32(from+c*4,true);
  }
  return Object.freeze({data,width:p.width,height:p.height,sourceFormat:p.format,channels:'rgba',
    componentType:p.floating?'float32':'unorm8',srgb:p.format.endsWith('-srgb'),
    bytesPerRow:p.width*(p.floating?16:4),flipY:p.flipY,
    origin:Object.freeze([p.x,p.y,p.layer]),mipLevel:p.mipLevel});
}

/** maxBytes bounds the sum of staging + prospective returned arrays for every
 * pending request. maxPending is an admission limit, not an unbounded work queue.
 * Only the staging buffers are owned. Textures and the device remain borrowed.
 * sRGB bytes and alpha are NOT transformed; float values are not clamped.
 */
export function createGpuAnimationReadback(device,options={}){
  fields(options,['maxBytes','maxPending','label'],'readback options');
  const maxBytes=integer(options.maxBytes ?? 64*1024*1024,'byte budget',1),maxPending=integer(options.maxPending ?? 4,'pending limit',1);
  const maxBufferSize=integer(device?.limits?.maxBufferSize,'device buffer limit',1),label=options.label ?? 'f3d-animation-readback';
  if(maxPending>64||typeof label!=='string')fail('OPTIONS','Invalid pending limit or label');
  for(const name of ['createBuffer','createCommandEncoder','pushErrorScope','popErrorScope'])if(typeof device?.[name]!=='function')fail('DEVICE',`Missing device.${name}`);
  if(typeof device.queue?.submit!=='function')fail('DEVICE','Expected a WebGPU queue');
  let disposed=false,terminal=null,reservedBytes=0,bufferBytes=0,issuing=false;
  const pending=new Set();
  function live(){if(terminal)throw terminal;if(disposed)fail('DISPOSED','Readback service is disposed');}
  function stop(cause){if(terminal||disposed)return;terminal=cause;for(const r of pending)r.cancel(cause);}
  if(device.lost&&typeof device.lost.then==='function')device.lost.then(
    info=>stop(error('DEVICE_LOST',info?.message ?? 'Device lost')),
    cause=>stop(error('DEVICE_LOST','Device loss notification failed',{cause})),
  );
  function admit(texture,options){
    live();if(issuing)fail('REENTRANT','Readback submission cannot be reentered');
    const p=plan(texture,options,maxBufferSize);
    if(pending.size>=maxPending||p.reserved>maxBytes-reservedBytes)fail('LIMIT','Pending readbacks exceed the configured budget');
    return p;
  }
  async function readPixels(texture,options={}){
    const p=admit(texture,options),signal=options.signal;
    const completion=Promise.resolve(options.completion);completion.catch(()=>{});
    let buffer,cleaned=false,cancelled=false,cancellation,resolveDone,rejectCancel,onAbort;
    const cancelledPromise=new Promise((_,reject)=>{rejectCancel=reject;});cancelledPromise.catch(()=>{});
    const done=new Promise(resolve=>{resolveDone=resolve;});
    const destroy=()=>{if(buffer){const owned=buffer;buffer=null;try{owned.destroy();}catch{}}};
    const r={done,cancel(cause){if(cancelled||cleaned)return;cancelled=true;cancellation=cause;destroy();rejectCancel(cause);}};
    const check=()=>{if(cancelled)throw cancellation;live();checkSignal(signal);};
    pending.add(r);reservedBytes+=p.reserved;bufferBytes+=p.size;
    try{
      if(signal){onAbort=()=>r.cancel(abortReason(signal));signal.addEventListener('abort',onAbort,{once:true});}
      check();issuing=true;
      let depth=0,issueError,mapping;const scopes=[];
      try{
        // Scopes are popped in this synchronous turn, never across mapAsync.
        for(const filter of ['out-of-memory','validation']){device.pushErrorScope(filter);depth++;}
        buffer=device.createBuffer({label,size:p.size,usage:1|8});check();
        const encoder=device.createCommandEncoder({label});
        encoder.copyTextureToBuffer({texture,mipLevel:p.mipLevel,origin:[p.x,p.y,p.layer],aspect:'all'},
          {buffer,offset:0,bytesPerRow:p.pitch,rowsPerImage:p.height},[p.width,p.height,1]);
        const commands=encoder.finish();check();device.queue.submit([commands]);check();
        mapping=Promise.resolve(buffer.mapAsync(1,0,p.size));mapping.catch(()=>{});
      }catch(cause){issueError=cause;}
      finally{
        while(depth-- > 0){try{const scope=Promise.resolve(device.popErrorScope());scope.catch(()=>{});scopes.push(scope);}catch(cause){issueError ??= cause;}}
        issuing=false;
      }
      const validation=Promise.all(scopes).then(results=>{const invalid=results.find(Boolean);if(invalid)throw error('GPU',invalid.message ?? 'GPU readback validation failed');});
      validation.catch(()=>{});
      if(issueError!==undefined)throw issueError;
      await Promise.race([Promise.all([mapping,validation,completion]),cancelledPromise]);check();
      // Copy before unmap/destroy detaches mapped memory. No borrowed output.
      const mapped=buffer.getMappedRange(0,p.size);
      if(!(mapped instanceof ArrayBuffer)||mapped.byteLength<p.size)fail('MAP','Mapped staging extent is truncated');
      return unpack(mapped,p);
    }finally{
      issuing=false;cleaned=true;
      if(onAbort)signal.removeEventListener('abort',onAbort);
      destroy();pending.delete(r);reservedBytes-=p.reserved;bufferBytes-=p.size;resolveDone();
    }
  }
  const result=Object.freeze({readPixels,
    // Pure preflight for composing capture with other validation stages. Does
    // not reserve capacity; readPixels repeats admission at actual submission.
    validate(texture,options={}){admit(texture,options);},
    get pending(){return pending.size;},get bufferBytes(){return bufferBytes;},get reservedBytes(){return reservedBytes;},
    get disposed(){return disposed;},get failed(){return terminal!==null;},
    // Await the CPU requests present at the call, not unrelated GPU queue work.
    async whenIdle(){live();await Promise.all([...pending].map(r=>r.done));live();return result;},
    dispose(){if(disposed)return;disposed=true;for(const r of pending)r.cancel(error('DISPOSED','Readback service disposed'));},
  });
  return result;
}
