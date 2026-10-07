/** Versioned source Texture residency for the explicit r186 scene bridge.
 * Owns only GPU textures/samplers; never decodes URLs, replaces source images,
 * closes caller images or controls playback. prepare() admits a whole set before
 * allocations; update() uploads only requested texture/source versions. Native
 * failures are terminal, callback exceptions preserve their source upload history.
 *
 * Byte DataTextures: R/RG/RGBA, manual/generated mipmaps and RGBA row ranges.
 * Decoded image/canvas/ImageData: browser external-copy sRGB profile.
 * CubeTextures admit six matching byte/image faces with authored or generated
 * mips. VideoTextures copy ready HTMLVideoElement frames into stable 2D residency,
 * using the source's frame versions and built-in update() fallback. No frame
 * callbacks, playback controls or frame loop are installed here.
 * ImageBitmaps are admitted with flipY=false/premultiplyAlpha=false (the decode-time
 * profile both r186 renderers agree on, e.g. GLTFLoader's ImageBitmapLoader path).
 * Other ImageBitmap flags and compressed/depth/array/float textures remain explicit
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

/** Metadata-only admission, shared with source backgrounds. No allocation,
 * pixel copying, upload acknowledgement or GPU services. maxPixels bounds all
 * six base faces together; maxTextureBytes includes every face and mip level.
 */
export function inspectThreeTexture(texture, three, limits={}) {
  return textureInspector(three,limits)(texture);
}
function textureInspector(T,{
  maxTextureBytes=128*1024*1024,maxPixels=16*1024*1024,maxDimension=32768,maxLayers=6,float32Filterable=false,
}={}) {
  if(T?.REVISION!=='186'||typeof T.Texture!=='function')fail('SOURCE','Supply the pinned r186 module');
  integer(maxTextureBytes,1,Number.MAX_SAFE_INTEGER,'texture budget');
  integer(maxPixels,1,Number.MAX_SAFE_INTEGER,'pixel capacity');integer(maxDimension,1,32768,'texture dimension');
  integer(maxLayers,1,Number.MAX_SAFE_INTEGER,'texture layers');
  const wraps=new Map([[T.ClampToEdgeWrapping,'clamp-to-edge'],[T.RepeatWrapping,'repeat'],[T.MirroredRepeatWrapping,'mirror-repeat']]);
  const filters=new Map([[T.NearestFilter,['nearest','nearest',false]],[T.LinearFilter,['linear','nearest',false]],
    [T.NearestMipmapNearestFilter,['nearest','nearest',true]],[T.NearestMipmapLinearFilter,['nearest','linear',true]],
    [T.LinearMipmapNearestFilter,['linear','nearest',true]],[T.LinearMipmapLinearFilter,['linear','linear',true]]]);
  function dimensions(image,data,video=false){
    if(!image||image.complete===false)fail('NOT_READY','Source image has not loaded');
    if(video&&(!(image.videoWidth>0)||!(image.videoHeight>0)))fail('NOT_READY','Video has no decoded frame dimensions');
    const width=integer(video?image.videoWidth:data?image.width:image.naturalWidth??image.width,1,maxDimension,'image width');
    const height=integer(video?image.videoHeight:data?image.height:image.naturalHeight??image.height,1,maxDimension,'image height');
    if(width*height>maxPixels)fail('LIMIT','Source exceeds pixel budget');return {width,height};
  }
  function describe(t){
    if(!(t instanceof T.Texture)||t.isCompressedTexture||t.isDepthTexture||
      t.isData3DTexture||t.isDataArrayTexture||t.isFramebufferTexture||t.isRenderTargetTexture||t.isExternalTexture)
      fail('SOURCE','Expected a byte 2D texture, VideoTexture or CubeTexture');
    const video=t.isVideoTexture===true,cube=t.isCubeTexture===true,layers=cube?6:1;
    if(video&&(typeof T.VideoTexture!=='function'||!(t instanceof T.VideoTexture)||
        !instance(t.image,'HTMLVideoElement')||cube||t.isDataTexture||t.isVideoFrameTexture||
        typeof T.VideoTexture.prototype.update!=='function'||t.update!==T.VideoTexture.prototype.update))
      fail('SOURCE','Video requires a source VideoTexture, HTMLVideoElement and built-in update hook');
    // HAVE_CURRENT_DATA is the platform value 2. A resident, temporarily starved
    // stream retains its last submitted pixels without acknowledging a new frame.
    const videoReady=video&&integer(t.image.readyState,0,4,'video readyState')>=2;
    if(cube&&(typeof T.CubeTexture!=='function'||!(t instanceof T.CubeTexture)||!Array.isArray(t.image)||t.image.length!==6))
      fail('SOURCE','CubeTexture requires six source faces');
    if(layers>maxLayers)fail('LIMIT','Native texture array-layer limit is too small');
    // Element storage: bytes, half floats (Uint16 bit patterns) or 32-bit floats.
    const bpc=t.type===T.UnsignedByteType?1:t.type===T.HalfFloatType?2:t.type===T.FloatType?4:0;
    if(!bpc||t.internalFormat!==null||t.compareFunction!=null)
      fail('FORMAT','This source path requires byte, half-float or float non-depth storage without internal overrides');
    const channels=new Map([[T.RedFormat,1],[T.RGFormat,2],[T.RGBAFormat,4]]).get(t.format);
    if(!channels)fail('FORMAT','Expected R, RG or RGBA storage');
    if(![T.NoColorSpace,T.LinearSRGBColorSpace,T.SRGBColorSpace].includes(t.colorSpace)||
        (t.colorSpace===T.SRGBColorSpace&&(channels!==4||bpc!==1)))fail('COLOR','Unsupported transfer function or channel format');
    if(!Number.isSafeInteger(t.version)||t.version<(video?0:1)||!Number.isSafeInteger(t.source?.version)||t.source.version<0||t.source.dataReady!==true)
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
    const data=cube?t.image[0]?.isDataTexture===true:t.isDataTexture===true;
    if(bpc!==1&&(!data||video))fail('FORMAT','Half-float and float storage is admitted for DataTextures');
    // generateMipmap on half-float/float storage: render-pass box filtering, as for
    // bytes; float32 needs float32-filterable (WebGL: OES_texture_float_linear).
    if(bpc===4&&t.generateMipmaps&&!float32Filterable)fail('MIPS','Generated float32 mipmaps need the float32-filterable device feature');
    // Float32 linear filtering needs the device's float32-filterable feature (WebGL
    // OES_texture_float_linear); never silently degrade to nearest.
    const filterable=bpc!==4||float32Filterable;
    if(!filterable&&(mag[0]==='linear'||filter[0]==='linear'||filter[1]==='linear'))
      fail('SAMPLER','Linear filtering of float32 textures needs the float32-filterable device feature');
    const unwrap=image=>{
      if(!cube)return image;
      if(data){
        if(typeof T.DataTexture!=='function'||!(image instanceof T.DataTexture)||image.isDataTexture!==true)
          fail('SOURCE','Cube byte faces must all be source DataTextures');
        return image.image;
      }
      return image;
    };
    const base=cube?t.image.map(unwrap):[t.image];
    const {width,height}=dimensions(base[0],data,video),full=1+Math.floor(Math.log2(Math.max(width,height)));
    if(cube&&width!==height)fail('SOURCE','Cube faces must be square');
    if(width*height*layers>maxPixels)fail('LIMIT','All cube faces together exceed the pixel budget');
    if(!Array.isArray(t.mipmaps))fail('SOURCE','Expected source mipmaps');
    const manual=t.mipmaps.length>0;
    if(video&&(manual||t.generateMipmaps||filter[2]))
      fail('MIPS','Live video uses the source single-level, non-mipmapped sampling profile');
    if(manual&&((!cube&&!data)||t.generateMipmaps))fail('MIPS','Authored mips require byte data or cube faces with generation disabled');
    // r186 uncompressed cubes list ADDITIONAL mip levels; ordinary DataTextures
    // include the base level in mipmaps. Do not drop or duplicate either base.
    const levels=manual?t.mipmaps.length+Number(cube):t.generateMipmaps?full:1;
    if(levels>full||(filter[2]&&levels!==full))fail('MIPS','Mipmapped filtering requires a complete source pyramid');
    const uploads=[];
    if(cube){
      for(let layer=0;layer<6;layer++)uploads.push({image:base[layer],level:0,layer});
      for(let mip=0;mip<t.mipmaps.length;mip++){
        const faces=t.mipmaps[mip]?.image;
        if(!Array.isArray(faces)||faces.length!==6)fail('MIPS','Each authored cube mip needs six faces');
        for(let layer=0;layer<6;layer++)uploads.push({image:unwrap(faces[layer]),level:mip+1,layer});
      }
    }else for(const [level,image] of (manual?t.mipmaps:base).entries())uploads.push({image,level,layer:0});
    let bytes=0;
    for(let level=0;level<levels;level++)bytes+=Math.max(1,width>>level)*Math.max(1,height>>level)*channels*bpc*layers;
    if(bytes>maxTextureBytes)fail('LIMIT','Texture exceeds byte budget');
    if(data&&t.premultiplyAlpha)fail('FORMAT','Premultiplied byte data requires an explicit upload profile');
    for(const {image,level} of uploads){
      if(!data){
        const bitmap=instance(image,'ImageBitmap');
        if(!(video||bitmap||instance(image,'HTMLImageElement')||instance(image,'HTMLCanvasElement')||instance(image,'OffscreenCanvas')||instance(image,'ImageData')))
          fail('SOURCE','Use decoded images, ImageBitmaps, canvases or ImageData');
        // r186 WebGLTextures skips UNPACK_FLIP_Y/PREMULTIPLY for ImageBitmaps (their
        // decode options already decided both) while WebGPUTextureUtils applies them.
        // Only the profile where both source renderers agree is admitted.
        if(bitmap&&(t.flipY||t.premultiplyAlpha))
          fail('SOURCE','ImageBitmap sources require flipY=false and premultiplyAlpha=false (orientation and alpha are decided at decode)');
        if(channels!==4)fail('FORMAT','External images require RGBA storage');
        if(instance(image,'ImageData')&&image.colorSpace&&image.colorSpace!=='srgb')fail('COLOR','ImageData requires the sRGB copy profile');
      }
      const extent=dimensions(image,data,video);
      if(extent.width!==Math.max(1,width>>level)||extent.height!==Math.max(1,height>>level))fail('MIPS','Invalid face or authored mip dimensions');
      if(data){
        const a=image.data,Kind=bpc===1?Uint8Array:bpc===2?Uint16Array:Float32Array;
        if(!(a instanceof Kind)||!(a.buffer instanceof ArrayBuffer)||a.buffer.resizable)fail('STORAGE',`Expected fixed, unshared ${Kind.name} source`);
        try{new Uint8Array(a.buffer,0,0);}catch{fail('STORAGE','Source pixels are detached');}
        const row=image.width*channels*bpc,pitch=Math.ceil(row/t.unpackAlignment)*t.unpackAlignment;
        if(a.byteLength<(image.height-1)*pitch+row)fail('STORAGE','Source pixels do not fill the unpacked image');
      }
    }
    const images=uploads.map(entry=>entry.image);
    if(!Array.isArray(t.updateRanges))fail('RANGE','Expected source update ranges');
    if(t.updateRanges.length){
      if(cube||!data||manual||channels!==4||t.flipY||bpc!==1)fail('RANGE','Partial source uploads require unflipped RGBA byte base pixels');
      for(const r of t.updateRanges){
        integer(r.start,0,width*height*4,'range start');integer(r.count,0,width*height*4-r.start,'range count');
        const x=Math.floor(r.start/4)%width,count=Math.ceil(r.count/4);
        if(x+count>width)fail('RANGE','Source partial updates must fit one pixel row');
      }
    }
    const prefix=channels===4?'rgba':channels===2?'rg':'r';
    const format=bpc===1?prefix+'8unorm':bpc===2?prefix+'16float':prefix+'32float',viewFormat=t.colorSpace===T.SRGBColorSpace?'rgba8unorm-srgb':format;
    const sampler={addressModeU:wraps.get(t.wrapS),addressModeV:wraps.get(t.wrapT),magFilter:mag[0],minFilter:filter[0],
      mipmapFilter:filter[1],lodMinClamp:0,lodMaxClamp:filter[2]?levels-1:0,maxAnisotropy};
    const key=JSON.stringify([width,height,levels,format,viewFormat,sampler,t.flipY,t.premultiplyAlpha,t.unpackAlignment,t.generateMipmaps,manual,data,...(cube?['cube']:video?['video']:[])]);
    return {source:t.source,width,height,levels,layers,cube,video,videoReady,bytes,format,viewFormat,sampler,key,data,images,uploads,manual,channels,bpc,
      sampleType:filterable?'float':'unfilterable-float',
      flipY:t.flipY,premultiplyAlpha:t.premultiplyAlpha,alignment:t.unpackAlignment};
  }
  return describe;
}

export function createGpuThreeTextures(device,{
  three:T,maxTextureBytes=128*1024*1024,maxTextures=256,maxPixels=16*1024*1024,label='f3d-three-textures',
}={}) {
  if(T?.REVISION!=='186'||typeof T.Texture!=='function')fail('SOURCE','Supply the pinned r186 module');
  integer(maxTextureBytes,1,Number.MAX_SAFE_INTEGER,'texture budget');integer(maxTextures,1,65536,'texture capacity');
  integer(maxPixels,1,Number.MAX_SAFE_INTEGER,'pixel capacity');if(typeof label!=='string')fail('VALUE','Expected a label');
  const describe=textureInspector(T,{maxTextureBytes,maxPixels,float32Filterable:device?.features?.has?.('float32-filterable')===true,
    maxDimension:device?.limits?.maxTextureDimension2D??32768,maxLayers:device?.limits?.maxTextureArrayLayers??1});
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
    for(let layer=0;layer<d.layers;layer++)for(let level=1;level<d.levels;level++)native(()=>{
      const view=i=>resource.texture.createView({format:d.viewFormat,baseMipLevel:i,mipLevelCount:1,
        ...(d.cube?{dimension:'2d',baseArrayLayer:layer,arrayLayerCount:1}:{})});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:view(level-1)},{binding:1,resource:mipSampler}]});
      const pass=encoder.beginRenderPass({colorAttachments:[{view:view(level),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();stats.mipPasses++;
    });
    native(()=>device.queue.submit([encoder.finish()]));
  }
  function upload(t,r,d){
    if(d.video){
      if(!d.videoReady)return;
      // The pinned hook is a no-op with requestVideoFrameCallback. On older
      // hosts it requests one upload per consuming update, after set admission.
      // Never subscribe, seek, play, pause or dispose the application's video.
      t.update();
    }
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
        }else for(const {level,layer,image} of d.uploads){
          // Rows in bytes: GL UNPACK_ALIGNMENT pads each source row to the alignment.
          const row=image.width*d.channels*d.bpc,pitch=Math.ceil(row/d.alignment)*d.alignment;
          const src=new Uint8Array(image.data.buffer,image.data.byteOffset,image.data.byteLength);
          let data=src;
          if(d.flipY){data=new Uint8Array(row*image.height);for(let y=0;y<image.height;y++)data.set(src.subarray((image.height-1-y)*pitch,(image.height-1-y)*pitch+row),y*row);}
          native(()=>device.queue.writeTexture({texture:a.texture,mipLevel:level,...(d.cube?{origin:[0,0,layer]}:{})},data,{bytesPerRow:d.flipY?row:pitch},[image.width,image.height,1]));stats.writeCalls++;
        }
      }else for(const {image,level,layer} of d.uploads){
        native(()=>device.queue.copyExternalImageToTexture({source:image,flipY:d.flipY},
          {texture:a.texture,colorSpace:'srgb',premultipliedAlpha:d.premultiplyAlpha,
            ...(d.cube?{mipLevel:level,origin:[0,0,layer]}:{})},
          [Math.max(1,d.width>>level),Math.max(1,d.height>>level)]));stats.externalCopies++;
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
        if(d.video&&!d.videoReady&&!match(t,d))fail('NOT_READY','A new video residency requires a current decoded frame');
        if(!d.data&&typeof device.queue.copyExternalImageToTexture!=='function')
          fail('DEVICE','External source uploads require GPUQueue.copyExternalImageToTexture');
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
            const texture=native(()=>device.createTexture({label,size:[d.width,d.height,d.layers],format:d.format,
              viewFormats:d.format===d.viewFormat?[]:[d.viewFormat],mipLevelCount:d.levels,usage:2|4|16}));
            resource={texture,source:d.source,key:d.key,bytes:d.bytes,sourceVersion:-1,refs:0};
            // Own immediately so a subsequent native view/sampler failure retires it.
            let keys=sources.get(d.source);if(!keys)sources.set(d.source,keys=new Map());keys.set(d.key,resource);textureBytes+=d.bytes;
            try{resource.view=native(()=>texture.createView({format:d.viewFormat,...(d.cube?{dimension:'cube'}:{})}));resource.sampler=native(()=>device.createSampler(d.sampler));}
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
  function binding(t){const {r,d}=check(t);return Object.freeze({view:r.resource.view,sampler:r.resource.sampler,version:r.version,sourceVersion:r.resource.sourceVersion,
    sampleType:d?.sampleType??'float',filtering:(d?.sampleType??'float')==='float'});}
  function update(input){
    const list=plans(input);if(busy)fail('REENTRANT','Cannot update during a texture operation');
    for(const {t,d} of list)if(!match(t,d))fail('PREPARE','Texture structure changed; call prepare');
    busy=true;try{if(list.length)checked(()=>{for(const {t,d} of list)upload(t,match(t,d),d);});return api;}finally{busy=false;}
  }
  const api=Object.freeze({prepare,update,binding,inspect:describe,
    // Borrowed native storage, for consumers such as the background renderer.
    // The pool remains its sole owner; callers must not destroy this texture.
    nativeTexture(t){return check(t).r.resource.texture;},
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
