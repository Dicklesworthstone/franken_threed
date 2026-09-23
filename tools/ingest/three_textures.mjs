/** Versioned source Texture residency for the explicit r186 scene bridge.
 * Owns only GPU textures/samplers; never decodes URLs, replaces source images,
 * sets needsUpdate or closes caller images. prepare() admits a whole set before
 * allocations; update() uploads only requested texture/source versions. Native
 * failures are terminal, callback exceptions preserve their source upload history.
 *
 * Byte DataTextures: R/RG/RGBA, manual/generated mipmaps and RGBA row ranges.
 * Decoded image/canvas/ImageData: browser external-copy sRGB profile. ImageBitmap
 * decode flags, video, compressed/depth/array/cube/float textures remain explicit
 * errors; applications can keep borrowing bindings for those sources.
 * See THREE_TEXTURES.md for source-state, color, bounds and completion contracts.
 */
export class ThreeTextureError extends Error {
  constructor(code,message){super(`THREE_TEXTURE_${code}: ${message}`);this.name='ThreeTextureError';this.code='THREE_TEXTURE_'+code;}
}
const fail=(code,message)=>{throw new ThreeTextureError(code,message);};
const integer=(v,min,max,label)=>{if(!Number.isSafeInteger(v)||v<min||v>max)fail('VALUE',`Invalid ${label}`);return v;};
const instance=(value,name)=>typeof globalThis[name]==='function'&&value instanceof globalThis[name];
// Same native downsample profile as gltf_textures.mjs. The source and destination
// views select linear vs sRGB filtering; no gamma approximation or CPU mip bake.
const MIP_SHADER=`
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32) -> @builtin(position) vec4<f32> {
  let p=vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p*2.0-1.0,0.0,1.0);
}
@fragment fn fs(@builtin(position) p:vec4<f32>) -> @location(0) vec4<f32> {
  let size=max(textureDimensions(source)/2u,vec2<u32>(1u));
  return textureSampleLevel(source,linearSampler,p.xy/vec2<f32>(size),0.0);
}`;

export function createGpuThreeTextures(device,{
  three:T,maxTextureBytes=128*1024*1024,maxTextures=256,maxPixels=16*1024*1024,label='f3d-three-textures',
}={}) {
  if(T?.REVISION!=='186'||typeof T.Texture!=='function')fail('SOURCE','Supply the pinned r186 module');
  integer(maxTextureBytes,1,Number.MAX_SAFE_INTEGER,'texture budget');integer(maxTextures,1,65536,'texture capacity');
  integer(maxPixels,1,Number.MAX_SAFE_INTEGER,'pixel capacity');if(typeof label!=='string')fail('VALUE','Expected a label');
  const wraps=new Map([[T.ClampToEdgeWrapping,'clamp-to-edge'],[T.RepeatWrapping,'repeat'],[T.MirroredRepeatWrapping,'mirror-repeat']]);
  const filters=new Map([[T.NearestFilter,['nearest','nearest',false]],[T.LinearFilter,['linear','nearest',false]],
    [T.NearestMipmapNearestFilter,['nearest','nearest',true]],[T.NearestMipmapLinearFilter,['nearest','linear',true]],
    [T.LinearMipmapNearestFilter,['linear','nearest',true]],[T.LinearMipmapLinearFilter,['linear','linear',true]]]);
  const records=new Map(),sources=new Map(),pipelines=new Map();
  let textureBytes=0,disposed=false,terminal=null,busy=false,pending=Promise.resolve(),mipSampler=null,rejectStop;
  const stopped=new Promise((_,reject)=>{rejectStop=reject;});stopped.catch(()=>{});
  const stats={allocations:0,uploads:0,writeCalls:0,externalCopies:0,mipPasses:0};
  function live(){if(disposed)fail('DISPOSED','Texture owner is disposed');if(terminal)throw terminal;}
  function drop(t){const r=records.get(t);if(!r)return;t.removeEventListener('dispose',r.listener);records.delete(t);
    if(--r.resource.refs===0){const a=r.resource;a.texture.destroy();textureBytes-=a.bytes;sources.get(a.source)?.delete(a.key);
      if(sources.get(a.source)?.size===0)sources.delete(a.source);}}
  function release(){for(const t of [...records.keys()])drop(t);
    for(const variants of sources.values())for(const r of variants.values())r.texture.destroy();
    sources.clear();textureBytes=0;pipelines.clear();mipSampler=null;}
  function stop(error){if(!terminal){terminal=error;release();rejectStop(error);}return terminal;}
  function native(fn){try{return fn();}catch(error){throw stop(error);}}
  function checked(fn){
    native(()=>device.pushErrorScope('validation'));native(()=>device.pushErrorScope('out-of-memory'));
    try{return fn();}finally{
      const scopes=native(()=>[device.popErrorScope(),device.popErrorScope()]);
      const checked=Promise.all(scopes).then(errors=>{const e=errors.find(Boolean);if(e)throw new ThreeTextureError('DEVICE',e.message||'Texture operation failed');}).catch(e=>{throw stop(e);});
      pending=Promise.all([pending,checked]).then(()=>{});pending.catch(()=>{});
    }
  }
  function dimensions(image,data){
    if(!image||image.complete===false)fail('NOT_READY','Source image has not loaded');
    const width=integer(data?image.width:image.naturalWidth??image.width,1,device.limits.maxTextureDimension2D,'image width');
    const height=integer(data?image.height:image.naturalHeight??image.height,1,device.limits.maxTextureDimension2D,'image height');
    if(width*height>maxPixels)fail('LIMIT','Source exceeds pixel budget');return {width,height};
  }
  function describe(t){
    if(!(t instanceof T.Texture)||t.isCubeTexture||t.isVideoTexture||t.isCompressedTexture||t.isDepthTexture||
      t.isData3DTexture||t.isDataArrayTexture||t.isFramebufferTexture||t.isRenderTargetTexture||t.isExternalTexture)
      fail('SOURCE','Expected an ordinary 2D source texture');
    if(t.type!==T.UnsignedByteType||t.internalFormat!==null||t.compareFunction!=null)
      fail('FORMAT','This source path requires unsigned-byte, non-depth storage without internal overrides');
    const channels=new Map([[T.RedFormat,1],[T.RGFormat,2],[T.RGBAFormat,4]]).get(t.format);
    if(!channels)fail('FORMAT','Expected R, RG or RGBA byte storage');
    if(![T.NoColorSpace,T.LinearSRGBColorSpace,T.SRGBColorSpace].includes(t.colorSpace)||
        (t.colorSpace===T.SRGBColorSpace&&channels!==4))fail('COLOR','Unsupported transfer function or channel format');
    if(!Number.isSafeInteger(t.version)||t.version<1||!Number.isSafeInteger(t.source?.version)||t.source.version<0||t.source.dataReady!==true)
      fail('NOT_READY','Set needsUpdate after providing ready source data');
    if(t.onUpdate!==null&&typeof t.onUpdate!=='function')fail('SOURCE','Invalid upload callback');
    const filter=filters.get(t.minFilter),mag=filters.get(t.magFilter);
    if(!filter||!mag||mag[2]||!wraps.has(t.wrapS)||!wraps.has(t.wrapT))fail('SAMPLER','Unsupported texture sampling');
    for(const name of ['flipY','premultiplyAlpha','generateMipmaps'])if(typeof t[name]!=='boolean')fail('SOURCE',`Expected boolean ${name}`);
    if(![1,2,4,8].includes(t.unpackAlignment))fail('SOURCE','Invalid unpack alignment');
    const anisotropy=integer(t.anisotropy,1,16,'anisotropy');
    // WebGL also allows anisotropic nearest-mipmap-linear, but WebGPU requires
    // linear minification too. Refuse that combination instead of changing it.
    if(anisotropy>1&&mag[0]==='linear'&&t.minFilter===T.NearestMipmapLinearFilter)
      fail('SAMPLER','Anisotropic nearest minification requires an explicit sampler profile');
    const maxAnisotropy=mag[0]==='linear'&&t.minFilter===T.LinearMipmapLinearFilter?anisotropy:1;
    const data=t.isDataTexture===true;
    if(!data){
      if(!(instance(t.image,'HTMLImageElement')||instance(t.image,'HTMLCanvasElement')||instance(t.image,'OffscreenCanvas')||instance(t.image,'ImageData')))
        fail('SOURCE','Use a decoded image, canvas or ImageData; ImageBitmap decode policy requires a borrowed binding');
      if(channels!==4)fail('FORMAT','External images require RGBA storage');
      if(instance(t.image,'ImageData')&&t.image.colorSpace&&t.image.colorSpace!=='srgb')fail('COLOR','ImageData requires the sRGB copy profile');
    }else if(t.premultiplyAlpha)fail('FORMAT','Premultiplied byte data requires an explicit upload profile');
    const {width,height}=dimensions(t.image,data),full=1+Math.floor(Math.log2(Math.max(width,height)));
    if(!Array.isArray(t.mipmaps))fail('SOURCE','Expected source mipmaps');
    const manual=t.mipmaps.length>0;
    if(manual&&(!data||t.generateMipmaps))fail('MIPS','Authored mips require a DataTexture with generation disabled');
    const levels=manual?t.mipmaps.length:t.generateMipmaps?full:1;
    if(levels>full||(filter[2]&&levels!==full))fail('MIPS','Mipmapped filtering requires a complete source pyramid');
    const images=manual?t.mipmaps:[t.image];let bytes=0;
    for(let level=0;level<levels;level++)bytes+=Math.max(1,width>>level)*Math.max(1,height>>level)*channels;
    if(bytes>maxTextureBytes)fail('LIMIT','Texture exceeds byte budget');
    if(data)for(const [level,image] of images.entries()){
      if(image.width!==Math.max(1,width>>level)||image.height!==Math.max(1,height>>level))fail('MIPS','Invalid authored mip dimensions');
      const a=image.data;
      if(!(a instanceof Uint8Array)||!(a.buffer instanceof ArrayBuffer)||a.buffer.resizable)fail('STORAGE','Expected fixed, unshared Uint8Array source');
      try{new Uint8Array(a.buffer,0,0);}catch{fail('STORAGE','Source pixels are detached');}
      const row=image.width*channels,pitch=Math.ceil(row/t.unpackAlignment)*t.unpackAlignment;
      if(a.byteLength<(image.height-1)*pitch+row)fail('STORAGE','Source pixels do not fill the unpacked image');
    }
    if(!Array.isArray(t.updateRanges))fail('RANGE','Expected source update ranges');
    if(t.updateRanges.length){
      if(!data||manual||channels!==4||t.flipY)fail('RANGE','Partial source uploads require unflipped RGBA base pixels');
      for(const r of t.updateRanges){
        integer(r.start,0,width*height*4,'range start');integer(r.count,0,width*height*4-r.start,'range count');
        const x=Math.floor(r.start/4)%width,count=Math.ceil(r.count/4);
        if(x+count>width)fail('RANGE','Source partial updates must fit one pixel row');
      }
    }
    const format=channels===4?'rgba8unorm':channels===2?'rg8unorm':'r8unorm',viewFormat=t.colorSpace===T.SRGBColorSpace?'rgba8unorm-srgb':format;
    const sampler={addressModeU:wraps.get(t.wrapS),addressModeV:wraps.get(t.wrapT),magFilter:mag[0],minFilter:filter[0],
      mipmapFilter:filter[1],lodMinClamp:0,lodMaxClamp:filter[2]?levels-1:0,maxAnisotropy};
    const key=JSON.stringify([width,height,levels,format,viewFormat,sampler,t.flipY,t.premultiplyAlpha,t.unpackAlignment,t.generateMipmaps,manual,data]);
    return {source:t.source,width,height,levels,bytes,format,viewFormat,sampler,key,data,images,manual,channels,
      flipY:t.flipY,premultiplyAlpha:t.premultiplyAlpha,alignment:t.unpackAlignment};
  }
  function plans(input){
    live();const list=[];for(const t of input){if(list.length>=maxTextures)fail('LIMIT','Requested texture count exceeds capacity');list.push(t);}
    return [...new Set(list)].map(t=>({t,d:describe(t)}));
  }
  function match(t,d){const r=records.get(t);return r&&r.resource.source===d.source&&r.resource.key===d.key?r:null;}
  function ranges(t,width){
    const rs=t.updateRanges;rs.sort((a,b)=>a.start-b.start);let last=0;
    const row=i=>Math.floor(Math.floor(i/4)/width);
    for(let i=1;i<rs.length;i++){const a=rs[last],b=rs[i];
      if(b.start<=a.start+a.count+1&&row(b.start)===row(a.start)&&row(b.start+b.count-1)===row(b.start))a.count=Math.max(a.count,b.start+b.count-a.start);
      else rs[++last]=b;}
    rs.length=last+1;return rs;
  }
  function downsample(resource,d){
    if(d.manual||d.levels===1)return;
    let pipeline=pipelines.get(d.viewFormat);
    if(!pipeline){pipeline=native(()=>device.createRenderPipeline({label:label+'/mips',layout:'auto',
      vertex:{module:device.createShaderModule({code:MIP_SHADER}),entryPoint:'vs'},
      fragment:{module:device.createShaderModule({code:MIP_SHADER}),entryPoint:'fs',targets:[{format:d.viewFormat}]},primitive:{topology:'triangle-list'}}));
      pipelines.set(d.viewFormat,pipeline);}
    mipSampler??=native(()=>device.createSampler({minFilter:'linear',magFilter:'linear'}));
    const encoder=native(()=>device.createCommandEncoder({label:label+'/mips'}));
    for(let level=1;level<d.levels;level++)native(()=>{
      const view=i=>resource.texture.createView({format:d.viewFormat,baseMipLevel:i,mipLevelCount:1});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:view(level-1)},{binding:1,resource:mipSampler}]});
      const pass=encoder.beginRenderPass({colorAttachments:[{view:view(level),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();stats.mipPasses++;
    });
    native(()=>device.queue.submit([encoder.finish()]));
  }
  function upload(t,r,d){
    if(r.version===t.version)return;
    const a=r.resource;
    if(a.sourceVersion!==t.source.version){
      if(d.data){
        if(t.updateRanges.length){
          const rowBytes=d.width*4,pitch=Math.ceil(rowBytes/d.alignment)*d.alignment;
          for(const range of ranges(t,d.width)){
            const pixel=Math.floor(range.start/4),x=pixel%d.width,y=Math.floor(pixel/d.width),width=Math.ceil(range.count/4);
            if(width)native(()=>device.queue.writeTexture({texture:a.texture,origin:[x,y,0]},t.image.data.subarray(y*pitch+x*4),{bytesPerRow:pitch},[width,1,1]));
            if(width)stats.writeCalls++;
          }
          t.clearUpdateRanges();
        }else for(const [level,image] of d.images.entries()){
          const row=image.width*d.channels,pitch=Math.ceil(row/d.alignment)*d.alignment;
          let data=image.data;
          if(d.flipY){data=new Uint8Array(row*image.height);for(let y=0;y<image.height;y++)data.set(image.data.subarray((image.height-1-y)*pitch,(image.height-1-y)*pitch+row),y*row);}
          native(()=>device.queue.writeTexture({texture:a.texture,mipLevel:level},data,{bytesPerRow:d.flipY?row:pitch},[image.width,image.height,1]));stats.writeCalls++;
        }
      }else{
        native(()=>device.queue.copyExternalImageToTexture({source:t.image,flipY:d.flipY},
          {texture:a.texture,colorSpace:'srgb',premultipliedAlpha:d.premultiplyAlpha},[d.width,d.height]));stats.externalCopies++;
      }
      downsample(a,d);stats.uploads++;a.sourceVersion=t.source.version;
      // Upstream acknowledges the source BEFORE onUpdate, and the texture AFTER.
      // A callback exception is not a driver error and is not replayed on retry.
      if(t.onUpdate)t.onUpdate(t);
    }
    r.version=t.version;
  }
  function prepare(input){
    const list=plans(input);if(busy)fail('REENTRANT','Cannot prepare during a texture operation');busy=true;
    try{
      let addedBytes=0,addedCount=0;const newKeys=new Map();
      for(const {t,d} of list){
        if(!records.has(t))addedCount++;
        if(!sources.get(d.source)?.has(d.key)){
          let keys=newKeys.get(d.source);if(!keys)newKeys.set(d.source,keys=new Set());
          if(!keys.has(d.key)){keys.add(d.key);addedBytes+=d.bytes;}
        }
      }
      if(records.size+addedCount>maxTextures||textureBytes+addedBytes>maxTextureBytes)fail('LIMIT','Aggregate texture residency budget exceeded');
      if(list.length)checked(()=>{for(const {t,d} of list){
        let r=match(t,d);
        if(!r){
          let resource=sources.get(d.source)?.get(d.key);
          if(!resource){
            const texture=native(()=>device.createTexture({label,size:[d.width,d.height,1],format:d.format,
              viewFormats:d.format===d.viewFormat?[]:[d.viewFormat],mipLevelCount:d.levels,usage:2|4|16}));
            resource={texture,source:d.source,key:d.key,bytes:d.bytes,sourceVersion:-1,refs:0};
            // Own immediately so a subsequent native view/sampler failure retires it.
            let keys=sources.get(d.source);if(!keys)sources.set(d.source,keys=new Map());keys.set(d.key,resource);textureBytes+=d.bytes;
            try{resource.view=native(()=>texture.createView({format:d.viewFormat}));resource.sampler=native(()=>device.createSampler(d.sampler));}
            catch(e){texture.destroy();keys.delete(d.key);textureBytes=Math.max(0,textureBytes-d.bytes);throw e;}
            stats.allocations++;
          }
          resource.refs++;drop(t);
          r={resource,version:-1,listener:()=>{if(busy)fail('REENTRANT','Cannot dispose source during upload');drop(t);}};
          records.set(t,r);t.addEventListener('dispose',r.listener);
        }
        upload(t,r,d);
      }});
      return api;
    }finally{busy=false;}
  }
  function check(t){live();const d=describe(t),r=match(t,d);if(!r)fail('PREPARE','Prepare the current texture storage and sampler before drawing');return {r,d};}
  function binding(t){const {r}=check(t);return Object.freeze({view:r.resource.view,sampler:r.resource.sampler,version:r.version,sourceVersion:r.resource.sourceVersion});}
  function update(input){
    const list=plans(input);if(busy)fail('REENTRANT','Cannot update during a texture operation');
    for(const {t,d} of list)if(!match(t,d))fail('PREPARE','Texture structure changed; call prepare');
    busy=true;try{if(list.length)checked(()=>{for(const {t,d} of list)upload(t,match(t,d),d);});return api;}finally{busy=false;}
  }
  const api=Object.freeze({prepare,update,binding,inspect:describe,
    retain(input){live();if(busy)fail('REENTRANT','Cannot evict during upload');const keep=new Set(input);for(const t of [...records.keys()])if(!keep.has(t))drop(t);},
    get disposed(){return disposed;},get failed(){return terminal!==null;},
    get diagnostics(){return Object.freeze({...stats,textureBytes,textures:records.size,resources:[...sources.values()].reduce((n,m)=>n+m.size,0)});},
    async whenIdle(){live();try{await Promise.race([Promise.all([pending,device.queue.onSubmittedWorkDone()]),stopped]);live();return api;}catch(e){if(!disposed)stop(e);throw e;}},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during upload');if(!disposed){disposed=true;release();rejectStop(new ThreeTextureError('DISPOSED','Texture owner is disposed'));}},
  });
  if(!device?.limits||!Number.isSafeInteger(device.limits.maxTextureDimension2D)||typeof device.createTexture!=='function'||
    typeof device.createSampler!=='function'||typeof device.queue?.writeTexture!=='function'||typeof device.queue.onSubmittedWorkDone!=='function'||
    typeof device.pushErrorScope!=='function'||typeof device.popErrorScope!=='function'||typeof device.lost?.then!=='function')fail('DEVICE','Supply a WebGPU device with texture uploads');
  device.lost.then(info=>{if(!disposed)stop(new ThreeTextureError('LOST',info?.message||'Device lost'));},e=>{if(!disposed)stop(e);});
  return api;
}
