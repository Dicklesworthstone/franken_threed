/** Managed linear HDR color/MSAA/depth attachments plus the explicit output pass.
 * Borrow the device, scene/renderer, and final target. Own only intermediate
 * textures, optional readback staging and the output pass. No clock, traversal or renderer
 * substitution. Use rendererOptions when constructing the borrowed draw path.
 * render(scene,{target,...frame,output:{toneMapping,exposure,outputAlpha}}) first
 * renders linear radiance, then submits display conversion. target is a native
 * single-sample GPUTexture, e.g. context.getCurrentTexture(), not a texture view.
 * Different target extents resize the HDR attachments; this is never downscaling.
 * On first use and after resize, color/depth loads must clear. Later load keeps
 * the HDR history (not the previous canvas bytes). clearColor is straight linear
 * RGBA and is premultiplied for the renderer's existing blend equation.
 * This whole-image effect does not implement per-material toneMapped exclusions.
 */
import {createGpuAnimationOutput,animationOutputSettings,AnimationOutputError} from './animation_output.mjs';
const fail=(code,message)=>{throw new AnimationOutputError('ANIMATION_PRESENTATION_'+code,message);};
const object=(v,label)=>{if(!v||typeof v!=='object'||Array.isArray(v))fail('OPTIONS',`Expected ${label}`);return v;};
const positive=(n,label)=>{if(!Number.isSafeInteger(n)||n<1)fail('LIMIT',`Invalid ${label}`);return n;};
const depthBytes=new Map([[null,0],['depth16unorm',2],['depth24plus',4],['depth32float',4]]);
export async function createGpuAnimationPresentation(device,options={}) {
  object(options,'presentation options');
  const {sampleCount=1,depthFormat='depth24plus',maxTextureBytes=256*1024*1024,readback=false,...outputOptions}={...options};
  if(![1,4].includes(sampleCount)||!depthBytes.has(depthFormat))fail('FORMAT','Unsupported sample count or depth format');
  positive(maxTextureBytes,'intermediate texture budget');
  const maxDimension=positive(device?.limits?.maxTextureDimension2D,'device dimension limit');
  if(typeof device?.createTexture!=='function')fail('DEVICE','Texture creation is required');
  if(outputOptions.inputAlpha!==undefined&&outputOptions.inputAlpha!=='premultiplied')fail('ALPHA','Managed scene output is premultiplied linear color');
  outputOptions.inputAlpha='premultiplied';
  if(readback!==false&&readback!==true&&(!readback||typeof readback!=='object'||Array.isArray(readback)))fail('OPTIONS','readback must be false, true or a limits object');
  const readbackOptions=readback===false?null:readback===true?{}:{...readback};
  let reader,output;
  const checkAbort=()=>{if(outputOptions.signal?.aborted)throw outputOptions.signal.reason ?? new DOMException('Aborted','AbortError');};
  try{
    checkAbort();
    if(readbackOptions){
      const {createGpuAnimationReadback}=await import('./animation_readback.mjs');checkAbort();
      reader=createGpuAnimationReadback(device,readbackOptions);
    }
    output=await createGpuAnimationOutput(device,outputOptions);checkAbort();
  }catch(error){reader?.dispose();output?.dispose();throw error;}
  const rendererOptions=Object.freeze({format:'rgba16float',depthFormat,sampleCount});
  let current=null,lastTarget=null,lastScene=null,lastOutput=null,disposed=false,terminal=null,busy=false,version=0,validation=Promise.resolve();
  let rejectStop;const stopped=new Promise((_,reject)=>{rejectStop=reject;});stopped.catch(()=>{});
  function release(){reader?.dispose();current?.dispose();current=null;lastTarget=lastScene=lastOutput=null;output.dispose();}
  function stop(error){if(!terminal&&!disposed){terminal=error;release();rejectStop(error);}}
  function live(){if(terminal)throw terminal;if(disposed)fail('DISPOSED','Presentation is disposed');if(output.failed||reader?.failed)fail('GPU','Output or readback failed');}
  if(device.lost&&typeof device.lost.then==='function')device.lost.then(info=>stop(new AnimationOutputError('ANIMATION_PRESENTATION_DEVICE_LOST',info?.message ?? 'Device lost')),stop);
  function targetInfo(target){
    object(target,'output target');const width=positive(target.width,'target width'),height=positive(target.height,'target height');
    if(width>maxDimension||height>maxDimension)fail('LIMIT','Target exceeds the device dimension limit');
    if(target.dimension!=='2d'||target.depthOrArrayLayers!==1||target.sampleCount!==1||!Number.isInteger(target.usage)||!(target.usage&16)||typeof target.createView!=='function')fail('TARGET','Target requires a single-sample 2D render attachment');
    if(target.format!==output.format&&!(output.format.endsWith('-srgb')&&target.format===output.format.slice(0,-5)))fail('FORMAT','Target format differs from output format');
    const bytes=width*height*(8*(sampleCount===4?5:1)+depthBytes.get(depthFormat)*sampleCount);
    if(!Number.isSafeInteger(bytes)||bytes>maxTextureBytes)fail('LIMIT','HDR attachments exceed the byte budget');
    return {target,width,height,bytes};
  }
  function allocate({width,height,bytes}){
    // The bound includes old+new application-owned allocations during resize;
    // format estimates exclude driver metadata/deferred physical retirement.
    if(bytes+(current?.bytes ?? 0)>maxTextureBytes)fail('LIMIT','Resize peak exceeds the intermediate byte budget');
    const owned=[];let complete=false;
    const make=(format,count,usage)=>{const texture=device.createTexture({label:'Animation HDR '+format,size:[width,height,1],format,sampleCount:count,usage});owned.push(texture);return texture;};
    let result,error;
    device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
    try{
      const color=make('rgba16float',1,4|16|(reader?1:0)),colorView=color.createView();
      const multisample=sampleCount===4?make('rgba16float',4,16):null;
      const depth=depthFormat?make(depthFormat,sampleCount,16):null;
      result={width,height,bytes,color,colorView:multisample?multisample.createView():colorView,
        ...(multisample?{resolveTarget:colorView}:{}),...(depth?{depthView:depth.createView()}:{}),
        dispose(){if(!complete){complete=true;for(const t of owned)t.destroy();}}};
    }catch(cause){error=cause;}
    const check=Promise.all([device.popErrorScope(),device.popErrorScope()]).then(([invalid,oom])=>{
      if(invalid||oom)fail('GPU',(invalid||oom).message ?? 'HDR allocation failed');
    });
    validation=Promise.all([validation,check]);validation.catch(stop);
    if(error){for(const t of owned)t.destroy();stop(error);throw error;}
    return result;
  }
  function submit(scene,input,camera,cameraSettings){
    live();if(busy)fail('REENTRANT','Presentation operation cannot be reentered');busy=true;
    try{
      object(input,'frame');object(scene,'borrowed scene/renderer');
      const method=camera?'renderCamera':'render';
      if(typeof scene[method]!=='function'||scene.disposed||scene.failed)fail('SCENE','A live scene/renderer is required');
      for(const [key,value]of Object.entries(rendererOptions))if(scene[key]!==undefined&&scene[key]!==value)fail('FORMAT','Borrowed renderer configuration differs from HDR attachments');
      const {target,output:overrides={},...frame}={...input};object(overrides,'output overrides');
      for(const key of Object.keys(overrides))if(!['toneMapping','exposure','outputAlpha'].includes(key))fail('OPTIONS',`Unknown output override: ${key}`);
      for(const key of ['colorView','depthView','resolveTarget'])if(Object.hasOwn(frame,key))fail('TARGET','Managed presentation supplies color/depth/resolve views');
      const selected=animationOutputSettings({...overrides,inputAlpha:'premultiplied'},output.defaults),info=targetInfo(target);
      const resize=!current||current.width!==info.width||current.height!==info.height;
      if(resize&&(frame.loadOp==='load'||(depthFormat&&frame.depthLoadOp==='load')))fail('HISTORY','Clear attachments on first use or resize');
      if(version===Number.MAX_SAFE_INTEGER)fail('VERSION','Submission counter exhausted');
      if(frame.clearColor!==undefined){
        const color=frame.clearColor;
        if((!Array.isArray(color)&&!ArrayBuffer.isView(color))||color.length!==4)fail('COLOR','Expected linear RGBA clear color');
        const values=Array.from(color);
        if(values.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0||v>65504)||values[3]>1)fail('COLOR','Invalid half-float clear color or alpha');
        frame.clearColor=[values[0]*values[3],values[1]*values[3],values[2]*values[3],values[3]];
      }
      // Keep old history alive until a resized frame actually submits. A host
      // camera/draw validation failure may be corrected without losing it.
      const next=resize?allocate(info):current;
      let submitted=false;
      try{
        const prepared={...frame,colorView:next.colorView,...(depthFormat?{depthView:next.depthView}:{}),...(sampleCount===4?{resolveTarget:next.resolveTarget}:{})};
        const completion=camera?scene.renderCamera(prepared,cameraSettings):scene.render(prepared);
        if(completion&&typeof completion.then==='function'){
          try{Promise.prototype.then.call(completion,undefined,()=>{});}catch{}
          submitted=true;fail('ASYNC','Scene rendering must submit synchronously');
        }
        submitted=true;
        output.render({source:next.color,target,...selected});
      }catch(error){
        if(next!==current)next.dispose();
        if(submitted||scene.failed||output.failed)stop(error);
        throw error;
      }
      if(next!==current){const old=current;current=next;old?.dispose();}
      if(reader){lastTarget=target;lastScene=scene;lastOutput=selected;}
      version++;return presentation;
    }finally{busy=false;}
  }
  async function readPixels(settings={}){
    live();if(busy)fail('REENTRANT','Cannot capture during render submission');
    if(!reader)fail('READBACK_DISABLED','Enable readback at presentation construction');
    if(!current)fail('READBACK_EMPTY','Render a frame before capturing');
    object(settings,'capture settings');
    const {source='hdr',...options}={...settings};
    if(source!=='hdr'&&source!=='output')fail('OPTIONS','Capture source must be hdr or output');
    if((options.mipLevel!==undefined&&options.mipLevel!==0)||(options.layer!==undefined&&options.layer!==0))fail('OPTIONS','Capture reads the rendered base mip/layer only');
    const texture=source==='hdr'?current.color:lastTarget;
    reader.validate(texture,options); // Reject bad rectangles/usage before fences.
    if(Object.hasOwn(options,'completion'))fail('OPTIONS','Presentation supplies producer validation');
    const observedVersion=version,observedScene=lastScene;
    const metadata={source,presentationVersion:observedVersion,
      colorSpace:source==='hdr'?'linear':output.outputColorSpace,
      alpha:source==='hdr'?'premultiplied':lastOutput.outputAlpha};
    let resolveCompletion,rejectCompletion,pixels;
    const completion=new Promise((resolve,reject)=>{resolveCompletion=resolve;rejectCompletion=reject;});completion.catch(()=>{});
    busy=true;
    try{
      // Queue the copy NOW. Keep its reservation until the originating draw
      // and output validations settle, not only until mapping completes.
      pixels=reader.readPixels(texture,{...options,completion});pixels.catch(()=>{});
      const verified=Promise.all([validation,output.whenIdle(),observedScene.whenIdle?.()]);
      verified.then(resolveCompletion,error=>{rejectCompletion(error);stop(error);});
    }catch(error){rejectCompletion(error);stop(error);throw error;}
    finally{busy=false;}
    const result=await Promise.race([pixels,stopped]);live();return Object.freeze({...result,...metadata});
  }

  const presentation=Object.freeze({rendererOptions,format:output.format,readPixels,
    get readbackEnabled(){return reader!==undefined;},get readbackPending(){return reader?.pending ?? 0;},
    get readbackBufferBytes(){return reader?.bufferBytes ?? 0;},get readbackReservedBytes(){return reader?.reservedBytes ?? 0;},
    get version(){return version;},get textureBytes(){return current?.bytes ?? 0;},get bufferBytes(){return output.allocatedBytes+(reader?.bufferBytes ?? 0);},
    get width(){return current?.width ?? 0;},get height(){return current?.height ?? 0;},
    get disposed(){return disposed;},get failed(){return terminal!==null||output.failed||Boolean(reader?.failed);},
    render(scene,frame){return submit(scene,frame,false);},renderCamera(scene,frame,settings){return submit(scene,frame,true,settings);},
    async whenIdle(){live();try{await Promise.race([Promise.all([validation,output.whenIdle(),reader?.whenIdle()]),stopped]);live();return presentation;}catch(error){stop(error);throw error;}},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during a render');if(!disposed){disposed=true;release();rejectStop(new AnimationOutputError('ANIMATION_PRESENTATION_DISPOSED','Presentation disposed'));}},
  });
  return presentation;
}
