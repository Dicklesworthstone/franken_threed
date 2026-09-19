/** Owned PNG/JPEG and BasisU KTX2 glTF image uploads and sampler/mipmap realization for WebGPU.
 * Requests come from prepareGltfAnimationModel(). The supplied device is borrowed.
 * A texture is shared by image AND color space, never across incompatible mip
 * filtering domains. Samplers remain distinct when source sampling differs.
 * Images retain glTF orientation, straight alpha and encoded numeric channels.
 * https://gpuweb.github.io/gpuweb/#dom-gpuqueue-copyexternalimagetotexture
 */
import {inspectGltfKtx2,transcodeGltfKtx2} from './gltf_ktx2.mjs';
export class GltfTextureError extends Error {
  constructor(code,message){super(`${code}: ${message}`);this.name='GltfTextureError';this.code=code;}
}
const fail=(code,message)=>{throw new GltfTextureError('GLTF_TEXTURE_'+code,message);};
const positive=(n,label)=>{if(!Number.isSafeInteger(n)||n<1)fail('LIMIT',`Invalid ${label}`);return n;};
const index=(n,label)=>{if(!Number.isSafeInteger(n)||n<0)fail('REQUEST',`Invalid ${label}`);return n;};
const wraps=new Map([[33071,'clamp-to-edge'],[33648,'mirror-repeat'],[10497,'repeat']]);
const filters=new Map([[9728,['nearest','nearest',false]],[9729,['linear','nearest',false]],
  [9984,['nearest','nearest',true]],[9985,['linear','nearest',true]],
  [9986,['nearest','linear',true]],[9987,['linear','linear',true]]]);

/** Read dimensions BEFORE native image decoding, bounding decompression size.
 * This is a header check, not a PNG/JPEG decoder or complete file validator.
 */
export function gltfImageDimensions(bytes,mimeType) {
  if(!(bytes instanceof Uint8Array)||!(bytes.buffer instanceof ArrayBuffer)||bytes.buffer.resizable)fail('IMAGE','Expected fixed image bytes');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  if(mimeType==='image/png') {
    if(bytes.length<33||![137,80,78,71,13,10,26,10].every((v,i)=>bytes[i]===v)||
       view.getUint32(8)!==13||view.getUint32(12)!==0x49484452)fail('IMAGE','Invalid PNG IHDR');
    return {width:positive(view.getUint32(16),'PNG width'),height:positive(view.getUint32(20),'PNG height')};
  }
  if(mimeType==='image/jpeg'&&bytes.length>=4&&bytes[0]===255&&bytes[1]===216) {
    let at=2;
    while(at<bytes.length) {
      if(bytes[at++]!==255)fail('IMAGE','Invalid JPEG marker');
      while(at<bytes.length&&bytes[at]===255)at++;
      const marker=bytes[at++];
      if(marker===0xda||marker===0xd9)break;
      if(marker===1||(marker>=0xd0&&marker<=0xd7))continue;
      if(at+2>bytes.length)break;
      const length=view.getUint16(at);
      if(length<2||length>bytes.length-at)fail('IMAGE','Invalid JPEG segment extent');
      if([0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf].includes(marker)) {
        if(length<8)fail('IMAGE','Invalid JPEG frame');
        return {width:positive(view.getUint16(at+5),'JPEG width'),height:positive(view.getUint16(at+3),'JPEG height')};
      }
      at+=length;
    }
  }
  fail('IMAGE','Missing supported PNG/JPEG dimensions');
}
const mipShader=`
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

/** Create a synchronous resolveTexture callback for the model decoder.
 * readImage(index) -> Promise<{bytes: Uint8Array, mimeType}> (loadGltfAsset API).
 * Defaults for missing glTF filters are an explicit loader policy: linear mag,
 * trilinear min. Nonmip filters clamp LOD to zero even on a shared mip chain.
 * maxTextureBytes bounds actual texel/block storage, including authored mips,
 * not driver overhead. maxTranscodeBytes bounds worst-case RGBA8 expansion per
 * KTX2 image, separately from its possibly much smaller compressed GPU storage.
 * ktx2Loader is a caller-owned, configured retained Three.js r186 KTX2Loader.
 * KTX2 uploads keep all authored mip levels, even a partial pyramid; sampling
 * cannot access levels outside the texture. No compressed mip is regenerated.
 * No ImageBitmap or external-image-copy API is needed for KTX2-only requests.
 * Native decode is cooperative: cancellation closes any eventual ImageBitmap.
 */
export async function createGltfTextureResources(device,requests,readImage,{
  createImageBitmap:decodeImage=globalThis.createImageBitmap,ktx2Loader=null,signal,
  maxTextureBytes=256*1024*1024,maxImagePixels=16*1024*1024,maxTextures=4096,
  maxTranscodeBytes=256*1024*1024,defaultMagFilter=9729,defaultMinFilter=9987,
}={}) {
  positive(maxTranscodeBytes,'transcode byte limit');positive(maxTextureBytes,'texture byte limit');positive(maxImagePixels,'image pixel limit');positive(maxTextures,'texture count');
  if(!Array.isArray(requests)||requests.length>maxTextures||typeof readImage!=='function')fail('REQUEST','Invalid texture requests or image provider');
  if(![9728,9729].includes(defaultMagFilter)||!filters.has(defaultMinFilter))fail('SAMPLER','Invalid default filters');
  // Snapshot and validate every request before fetching or allocating anything.
  const entries=new Map(),images=new Map();
  for(const request of requests) {
    const textureIndex=index(request?.textureIndex,'texture index'),imageIndex=index(request?.imageIndex,'image index');
    const colorSpace=request.colorSpace;
    if(colorSpace!=='srgb'&&colorSpace!=='linear')fail('REQUEST','Invalid texture color space');
    const input=request.sampler ?? {},mag=input.magFilter ?? defaultMagFilter,min=input.minFilter ?? defaultMinFilter;
    const u=wraps.get(input.wrapS ?? 10497),v=wraps.get(input.wrapT ?? 10497),filter=filters.get(min);
    if(!u||!v||!filter||![9728,9729].includes(mag))fail('SAMPLER','Unsupported glTF sampler');
    const sampler={addressModeU:u,addressModeV:v,magFilter:mag===9728?'nearest':'linear',minFilter:filter[0],mipmapFilter:filter[1],
      ...(filter[2]?{}:{lodMaxClamp:0})};
    const key=textureIndex+':'+colorSpace,signature=JSON.stringify([imageIndex,sampler]);
    if(entries.has(key)) {if(entries.get(key).signature!==signature)fail('REQUEST','Conflicting texture identity');continue;}
    const entry={key,imageIndex,colorSpace,sampler,signature,mips:filter[2]};entries.set(key,entry);
    if(!images.has(imageIndex))images.set(imageIndex,new Map());
    const groups=images.get(imageIndex);
    if(!groups.has(colorSpace))groups.set(colorSpace,{mips:false,entries:[]});
    const group=groups.get(colorSpace);group.mips||=entry.mips;group.entries.push(entry);
  }
  const owned=new Set(),resolved=new Map(),pipelines=new Map(),samplers=new Map();
  let disposed=false,terminal=null,textureBytes=0,transcodeCancel=null;
  function release(){for(const texture of owned)texture.destroy();owned.clear();resolved.clear();pipelines.clear();samplers.clear();}
  function live(){
    if(terminal)throw terminal;
    if(disposed)fail('DISPOSED','Texture resources have been disposed');
    if(signal?.aborted)throw signal.reason ?? new DOMException('Aborted','AbortError');
  }
  live();
  if(entries.size) {
    if(!device?.queue||typeof device.createTexture!=='function'||typeof device.pushErrorScope!=='function'||typeof device.popErrorScope!=='function'||
       typeof device.queue.onSubmittedWorkDone!=='function')fail('DEVICE','WebGPU texture upload is required');
    positive(device.limits?.maxTextureDimension2D,'device texture dimension limit');
    if(device.lost&&typeof device.lost.then==='function')device.lost.then(info=>{
      terminal=new GltfTextureError('GLTF_TEXTURE_DEVICE_LOST',info?.message ?? 'WebGPU device lost');release();transcodeCancel?.abort(terminal);
    },error=>{terminal=error;release();transcodeCancel?.abort(error);});
  }
  // Pop error scopes synchronously, before awaiting. Never leave a scope around
  // application/network/image-decoder awaits on a borrowed, shared GPUDevice.
  async function checked(operation) {
    live();device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
    let value,error;
    try{value=operation();}catch(cause){error=cause;}
    const validation=device.popErrorScope(),memory=device.popErrorScope();
    const [result,v,m]=await Promise.all([value,validation,memory]);
    if(error)throw error;if(v||m)fail('GPU',(v||m).message ?? 'WebGPU upload failed');
    live();return result;
  }
  async function pipeline(format) {
    if(!pipelines.has(format))pipelines.set(format,await checked(()=>{
      const module=device.createShaderModule({label:'glTF mipmap',code:mipShader});
      return device.createRenderPipelineAsync({label:'glTF mipmap '+format,layout:'auto',
        vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
    }));
    return pipelines.get(format);
  }
  function resolveGroup(view,group) {
    for(const entry of group.entries) {
      const key=JSON.stringify(entry.sampler);
      if(!samplers.has(key))samplers.set(key,device.createSampler(entry.sampler));
      resolved.set(entry.key,Object.freeze({view,sampler:samplers.get(key)}));
    }
  }
  async function uploadKtx2(image,imageIndex,groups) {
    if(typeof device.queue.writeTexture!=='function')fail('DEVICE','KTX2 uploads require queue.writeTexture');
    const limits={maxImagePixels,maxDimension:device.limits.maxTextureDimension2D};
    const header=inspectGltfKtx2(image.bytes,limits);
    const dfd=new DataView(image.bytes.buffer,image.bytes.byteOffset,image.bytes.byteLength).getUint32(48,true);
    if([...groups.keys()].some(space=>space!==header.colorSpace)||image.bytes[dfd+13]!== (header.colorSpace==='srgb'?1:0)) {
      fail('COLOR','KTX2 transfer/primaries must match every material use');
    }
    // Propagate device loss as well as caller cancellation through the retained
    // callback adapter. This does not terminate the borrowed loader or workers;
    // the adapter closes a late native result after rejecting construction.
    const controller=new AbortController();transcodeCancel=controller;
    const cancel=()=>controller.abort(signal.reason);
    signal?.addEventListener('abort',cancel,{once:true});
    let decoded;
    try {
      live();
      decoded=await transcodeGltfKtx2(image.bytes,ktx2Loader,{...limits,colorSpace:header.colorSpace,
        features:device.features ?? new Set(),maxDecodedBytes:maxTranscodeBytes,signal:controller.signal});
    } finally {signal?.removeEventListener('abort',cancel);transcodeCancel=null;}
    live();
    if(decoded.byteLength>maxTextureBytes-textureBytes)fail('LIMIT','KTX2 mip storage exceeds texture byte budget');
    textureBytes+=decoded.byteLength;
    await checked(()=>{
      // Compressed textures cannot be render attachments. Typed-byte uploads use
      // the final transfer format directly, unlike external PNG/JPEG image copies.
      const texture=device.createTexture({label:`glTF KTX2 image ${imageIndex}`,size:[decoded.width,decoded.height,1],
        format:decoded.format,mipLevelCount:decoded.levelCount,usage:2|4});
      owned.add(texture);
      for(const [mipLevel,mip]of decoded.mipmaps.entries()) {
        device.queue.writeTexture({texture,mipLevel},mip.data,{bytesPerRow:mip.bytesPerRow},[mip.copyWidth,mip.copyHeight,1]);
      }
      resolveGroup(texture.createView({format:decoded.format}),groups.get(header.colorSpace));
    });
  }
  try {
    for(const [imageIndex,groups]of images) {
      live();const image=await readImage(imageIndex);live();
      if(image?.mimeType==='image/ktx2') {await uploadKtx2(image,imageIndex,groups);continue;}
      if(typeof decodeImage!=='function'||typeof device.queue.copyExternalImageToTexture!=='function')fail('DEVICE','PNG/JPEG uploads require ImageBitmap and external image copy');
      const {width,height}=gltfImageDimensions(image?.bytes,image?.mimeType);
      if(width>device.limits.maxTextureDimension2D||height>device.limits.maxTextureDimension2D||width>maxImagePixels/height)fail('LIMIT','Image dimensions exceed decode/device budget');
      // Charge all color-space copies of this image before decoding it.
      for(const group of groups.values()) {
        group.levels=group.mips?1+Math.floor(Math.log2(Math.max(width,height))):1;
        let bytes=0,w=width,h=height;
        for(let level=0;level<group.levels;level++){bytes+=w*h*4;w=Math.max(1,Math.floor(w/2));h=Math.max(1,Math.floor(h/2));}
        if(bytes>maxTextureBytes-textureBytes)fail('LIMIT','Texture mip storage exceeds byte budget');textureBytes+=bytes;
      }
      const bitmap=await decodeImage(new Blob([image.bytes],{type:image.mimeType}),{
        imageOrientation:'none',premultiplyAlpha:'none',colorSpaceConversion:'none',
      });
      try {
        live();if(!bitmap||bitmap.width!==width||bitmap.height!==height||typeof bitmap.close!=='function')fail('IMAGE','Decoded image dimensions disagree with header');
        for(const [colorSpace,group]of groups) {
          const format=colorSpace==='srgb'?'rgba8unorm-srgb':'rgba8unorm';
          const mipPipeline=group.levels>1?await pipeline(format):null;
          await checked(()=>{
            // Upload through an unorm base texture; the material/mip views alone
            // select sRGB transfer decoding. Normal/metallic channels stay linear.
            const texture=device.createTexture({label:`glTF image ${imageIndex} ${colorSpace}`,size:[width,height,1],format:'rgba8unorm',
              viewFormats:colorSpace==='srgb'?['rgba8unorm-srgb']:[],mipLevelCount:group.levels,usage:2|4|16});
            owned.add(texture);
            device.queue.copyExternalImageToTexture({source:bitmap,flipY:false},{texture,colorSpace:'srgb',premultipliedAlpha:false},[width,height]);
            if(mipPipeline) {
              const encoder=device.createCommandEncoder({label:'glTF mip chain'}),sampler=device.createSampler({minFilter:'linear',magFilter:'linear'});
              for(let level=1;level<group.levels;level++) {
                const source=texture.createView({format,baseMipLevel:level-1,mipLevelCount:1});
                const target=texture.createView({format,baseMipLevel:level,mipLevelCount:1});
                const bindGroup=device.createBindGroup({layout:mipPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:source},{binding:1,resource:sampler}]});
                const pass=encoder.beginRenderPass({colorAttachments:[{view:target,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,0]}]});
                pass.setPipeline(mipPipeline);pass.setBindGroup(0,bindGroup);pass.draw(3);pass.end();
              }
              device.queue.submit([encoder.finish()]);
            }
            resolveGroup(texture.createView({format}),group);
          });
        }
      } finally {if(bitmap&&typeof bitmap.close==='function')bitmap.close();}
    }
    if(entries.size)await device.queue.onSubmittedWorkDone();live();
    return Object.freeze({
      get textureBytes(){return owned.size?textureBytes:0;},get textureCount(){return owned.size;},
      get disposed(){return disposed;},get failed(){return terminal!==null;},
      resolveTexture(request){live();const entry=entries.get(request.textureIndex+':'+request.colorSpace);
        if(!entry||entry.imageIndex!==request.imageIndex)fail('REQUEST','Texture was not prepared');return resolved.get(entry.key);},
      dispose(){if(!disposed){disposed=true;release();}},
    });
  } catch(error) {release();throw error;}
}
