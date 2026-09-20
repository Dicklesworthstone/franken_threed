/** URL/GLB -> owned textures + the existing animated WebGPU scene. No second
 * renderer, clock, scheduler or placeholder drawables. Attachments and the frame
 * loop remain caller-owned; authored cameras/lights are available explicitly.
 * picking:true opts in to synchronous current-pose geometric selection.
 * exporting:true enables self-contained posed GLB export with cached source images
 * and current-pose authored cameras/lights. Export sceneView:null for meshes only.
 * sourceExport:true separately retains an offline, rig/clip-preserving GLB of the
 * loaded asset. This captures authored state, not subsequent live pose edits.
 */
import * as gltfAssets from './gltf_asset.mjs';
const {loadGltfAsset,GltfAssetError}=gltfAssets;
import {prepareGltfAnimationModel} from './animation_model.mjs';
import {createGltfTextureResources,GltfTextureError} from './gltf_textures.mjs';
import {createGpuDecodedAnimationScene} from './animation_model_gpu.mjs';
const abort=signal=>{if(signal?.aborted)throw signal.reason ?? new DOMException('Aborted','AbortError');};

/** Nested assets/decode/textures/scene options belong to their existing stages.
 * signal cancels construction cooperatively; a late native decoder/GPU result is
 * disposed before rejection. A returned scene is owned until explicit dispose().
 * The device is never destroyed. Custom borrowed textures use the original
 * createGpuGltfAnimationScene API, not this owning loader.
 * textures.ktx2Loader lends a configured retained loader and enables BasisU
 * source selection. decode.basisu:false explicitly chooses optional core
 * fallbacks; required BasisU still fails. A selected transcode failure never
 * retries another source. The loader/device and retained worker pool are borrowed.
 * output:{format,toneMapping,exposure,...} opts into managed HDR presentation.
 * The scene renderer then uses rgba16float; output.format is the display target.
 * render()/renderCamera() take {target:context.getCurrentTexture(),...} instead
 * of attachment views. Output targets resize to the actual target extent, keep
 * the requested sampleCount/depthFormat, and are owned until scene disposal.
 * outputTextureBytes/outputBufferBytes are separate from asset/geometry budgets.
 * output.readback:true (or {maxBytes,maxPending,label}) enables readPixels().
 * Its source defaults to resolved linear HDR; source:'output' captures the last
 * display target, which must have COPY_SRC usage and still be valid this turn.
 * Omit output to retain the original attachment API and allocation behavior.
 * sourceExport:true (or {maxBytes,maxJsonBytes,maxResources}) closes ALL source
 * scenes/images at construction, before GPU texture/scene allocation. This can
 * fetch images outside the selected render scene under the existing asset policy.
 * exportSourceGLB({signal}) returns an independent ArrayBuffer without later I/O,
 * encoding or GPU readback. sourceExportBytes reports retained file bytes under
 * a separate default 128 MiB budget; disposal releases that retained snapshot.
 * Omission neither imports the packer nor loads/retains additional resources.
 * exportAnimationGLB(clips,settings) repacks that snapshot with explicit packed
 * clip records. It never samples the live pose, modifies its clips, or performs
 * I/O/GPU work. Targets use original glTF node IDs, not synthetic instance IDs.
 * settings accepts animationMode (append/replace), signal and tighter export
 * limits; construction limits remain ceilings. A parse copy of the retained
 * snapshot and binary/output staging may coexist; budgets are per stage.
 * exporting and sourceExport are independent options: posed vs authored export.
 */
export async function loadGpuGltfAnimationScene(device,source,{
  assets={},decode={},textures={},scene={},output=null,picking=false,exporting=false,sourceExport=false,signal=assets.signal ?? textures.signal ?? output?.signal,
}={}) {
  if(decode.resolveTexture!=null)throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','The owning loader supplies resolveTexture');
  for(const nested of [assets.signal,textures.signal,output?.signal])if(nested!==undefined&&nested!==signal)throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','Use one construction AbortSignal');
  if(sourceExport!==false&&sourceExport!==true&&(!sourceExport||typeof sourceExport!=='object'||Array.isArray(sourceExport)))
    throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','sourceExport must be false, true or a limits object');
  const sourceOptions=sourceExport===false?null:sourceExport===true?{}:{...sourceExport};
  if(sourceOptions) {
    for(const key of Object.keys(sourceOptions))if(!['maxBytes','maxJsonBytes','maxResources'].includes(key))
      throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS',`Unsupported sourceExport option: ${key}`);
    for(const [key,min,max]of [['maxBytes',20,0xffffffff],['maxJsonBytes',1,sourceOptions.maxBytes??128*1024*1024],['maxResources',1,65536]])
      if(sourceOptions[key]!==undefined&&(!Number.isSafeInteger(sourceOptions[key])||sourceOptions[key]<min||sourceOptions[key]>max))
        throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS',`Invalid sourceExport ${key}`);
  }
  // Freeze the route choice and stage settings before the first I/O await.
  // A caller editing its options while fetching cannot select one image and
  // accidentally supply another decoder policy when uploads begin.
  const decodeOptions={...decode},textureOptions={...textures,signal},assetOptions={...assets,signal};
  const sceneOptions={...scene,renderer:{...scene.renderer}};
  if(output!==null&&(!output||typeof output!=='object'||Array.isArray(output)))
    throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','output must be a presentation options object or null');
  const outputOptions=output===null?null:{...output,signal};
  if(outputOptions){
    if(outputOptions.readback&&typeof outputOptions.readback==='object'&&!Array.isArray(outputOptions.readback))outputOptions.readback={...outputOptions.readback};
    for(const key of ['sampleCount','depthFormat']){
      if(outputOptions[key]!==undefined&&sceneOptions.renderer[key]!==undefined&&outputOptions[key]!==sceneOptions.renderer[key])
        throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','Output and scene attachment settings conflict');
      if(outputOptions[key]===undefined&&sceneOptions.renderer[key]!==undefined)outputOptions[key]=sceneOptions.renderer[key];
    }
  }
  const basisu=decodeOptions.basisu===undefined ? textureOptions.ktx2Loader!=null : decodeOptions.basisu;
  if(typeof basisu!=='boolean'||(basisu&&typeof textureOptions.ktx2Loader?.parse!=='function'))
    throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','BasisU selection requires a configured textures.ktx2Loader; decode.basisu must be boolean');
  decodeOptions.basisu=basisu;
  abort(signal);
  let asset,resources,model,presentation,sourceSnapshot=null,sourceWriter=null,exportSources=[];
  try {
    if(outputOptions){
      // Disabled output neither imports its implementation nor allocates a pass.
      const {createGpuAnimationPresentation}=await import('./animation_presentation.mjs');abort(signal);
      presentation=await createGpuAnimationPresentation(device,outputOptions);abort(signal);
      Object.assign(sceneOptions.renderer,presentation.rendererOptions);
    }
    asset=await loadGltfAsset(source,assetOptions);abort(signal);
    if(sourceOptions) {
      const {exportGltfAssetGLB}=await import('./gltf_asset_export.mjs');abort(signal);
      sourceSnapshot=await exportGltfAssetGLB(asset,{...sourceOptions,signal});abort(signal);
      sourceWriter=exportGltfAssetGLB;
    }
    const prepared=prepareGltfAnimationModel(asset.json,asset.buffers,decodeOptions);
    resources=await createGltfTextureResources(device,prepared.textureRequests,asset.readImage,textureOptions);
    abort(signal);
    if(exporting!==false) {
      // These images were already loaded for uploads. Keep encoded bytes only
      // for opt-in export, so saving later needs neither network nor GPU readback.
      const images=new Map();
      for(const request of prepared.textureRequests) {
        if(!images.has(request.imageIndex))images.set(request.imageIndex,await asset.readImage(request.imageIndex));
        const image=images.get(request.imageIndex),resource=resources.resolveTexture(request);
        const sampler={...request.sampler,
          magFilter:request.sampler.magFilter ?? textureOptions.defaultMagFilter ?? 9729,
          minFilter:request.sampler.minFilter ?? textureOptions.defaultMinFilter ?? 9987};
        exportSources.push({resource,encoded:{bytes:image.bytes,mimeType:image.mimeType,sampler}});
      }
    }
    model=await createGpuDecodedAnimationScene(device,prepared.resolveTextures(resources.resolveTexture),{...sceneOptions,picking,exporting});
    abort(signal);
  } catch(error) {
    try{presentation?.dispose();}finally{try{model?.dispose();}finally{resources?.dispose();}}
    throw error;
  }
  function release(){try{presentation?.dispose();}finally{try{model.dispose();}finally{resources.dispose();exportSources=[];sourceSnapshot=null;sourceWriter=null;}}}
  function checkTextures(){
    if(resources.failed){release();throw new GltfTextureError('GLTF_TEXTURE_DEVICE_LOST','Model texture device was lost');}
    if(presentation?.failed){release();throw new GltfAssetError('GLTF_MODEL_OUTPUT_FAILED','Model presentation failed');}
  }
  function query(operation) {
    checkTextures();
    try{return operation();}
    catch(error){if(model.failed||resources.failed||Boolean(presentation?.failed))release();throw error;}
  }
  function invoke(operation){query(operation);return result;}
  const result=Object.freeze({pose:model.pose,view:model.view,cameras:model.cameras,lights:model.lights,
    controller:model.controller,draws:model.draws,deformers:model.deformers,
    source:model.source,diagnostics:model.diagnostics,assetBytes:asset.bytesLoaded,
    ...(model.instanceOrigins ? {instanceOrigins:model.instanceOrigins} : {}),
    get poseVersion(){return model.poseVersion;},get bufferBytes(){return model.bufferBytes;},
    get textureBytes(){return resources.textureBytes;},get disposed(){return model.disposed;},
    get failed(){return model.failed||resources.failed||Boolean(presentation?.failed);},
    get outputEnabled(){return presentation!==undefined;},
    get outputTextureBytes(){return presentation?.textureBytes ?? 0;},get outputBufferBytes(){return presentation?.bufferBytes ?? 0;},
    get readbackEnabled(){return presentation?.readbackEnabled ?? false;},
    get readbackPending(){return presentation?.readbackPending ?? 0;},
    get readbackBufferBytes(){return presentation?.readbackBufferBytes ?? 0;},
    get readbackReservedBytes(){return presentation?.readbackReservedBytes ?? 0;},
    async readPixels(settings={}){
      checkTextures();
      if(model.disposed)throw new GltfAssetError('GLTF_MODEL_DISPOSED','Model has been disposed');
      if(!presentation)throw new GltfAssetError('GLTF_MODEL_READBACK_DISABLED','Enable output.readback at model construction');
      try{const pixels=await presentation.readPixels(settings);checkTextures();return pixels;}
      catch(error){if(model.failed||resources.failed||Boolean(presentation.failed))release();throw error;}
    },
    get sourceExportEnabled(){return sourceOptions!==null;},
    get sourceExportBytes(){return sourceSnapshot?.byteLength??0;},
    async exportSourceGLB(settings={}){return query(()=>{
      if(model.disposed)throw new GltfAssetError('GLTF_EXPORT_DISPOSED','Model has been disposed');
      if(!sourceSnapshot)throw new GltfAssetError('GLTF_EXPORT_DISABLED','Enable sourceExport at model construction');
      if(!settings||typeof settings!=='object'||Array.isArray(settings)||Object.keys(settings).some(key=>key!=='signal'))
        throw new GltfAssetError('GLTF_EXPORT_OPTIONS','Source export accepts only an optional AbortSignal');
      abort(settings.signal);
      if(model.disposed||!sourceSnapshot)throw new GltfAssetError('GLTF_EXPORT_DISPOSED','Model was disposed during source export');
      return sourceSnapshot.slice(0);
    });},
    async exportAnimationGLB(clips,settings={}){return query(()=>{
      if(model.disposed)throw new GltfAssetError('GLTF_EXPORT_DISPOSED','Model has been disposed');
      if(!sourceSnapshot)throw new GltfAssetError('GLTF_EXPORT_DISABLED','Enable sourceExport at model construction');
      if(!settings||typeof settings!=='object'||Array.isArray(settings))
        throw new GltfAssetError('GLTF_EXPORT_OPTIONS','Invalid animation export settings');
      const options={};
      for(const key of Object.keys(settings)) {
        if(!['signal','animationMode','maxBytes','maxJsonBytes','maxResources'].includes(key))
          throw new GltfAssetError('GLTF_EXPORT_OPTIONS',`Unsupported animation export option: ${key}`);
        const property=Object.getOwnPropertyDescriptor(settings,key);
        if(!property||!Object.hasOwn(property,'value'))
          throw new GltfAssetError('GLTF_EXPORT_OPTIONS','Animation export settings must be data properties');
        options[key]=property.value;
      }
      abort(options.signal);
      if(!Array.isArray(clips))throw new GltfAssetError('GLTF_EXPORT_ANIMATION','Supply an explicit clip array');
      if(options.animationMode!==undefined&&!['append','replace'].includes(options.animationMode))
        throw new GltfAssetError('GLTF_EXPORT_OPTIONS','animationMode must be append or replace');
      const byteCeiling=sourceOptions.maxBytes??128*1024*1024;
      const ceilings={maxBytes:byteCeiling,maxJsonBytes:sourceOptions.maxJsonBytes??Math.min(byteCeiling,16*1024*1024),
        maxResources:sourceOptions.maxResources??4096};
      for(const [key,ceiling]of Object.entries(ceilings)) {
        if(options[key]===undefined)options[key]=key==='maxJsonBytes'?Math.min(ceiling,options.maxBytes):ceiling;
        if(!Number.isSafeInteger(options[key])||options[key]<(key==='maxBytes'?20:1)||options[key]>ceiling)
          throw new GltfAssetError('GLTF_EXPORT_LIMIT',`Invalid or increased animation export limit: ${key}`);
      }
      if(options.maxJsonBytes>options.maxBytes)throw new GltfAssetError('GLTF_EXPORT_LIMIT','JSON limit exceeds GLB limit');
      if(model.disposed||!sourceSnapshot)throw new GltfAssetError('GLTF_EXPORT_DISPOSED','Model was disposed during animation export');
      // Parse the private offline snapshot, not mutable caller JSON or posed
      // geometry. Keep the packer from construction so there is no import await
      // before it captures all supplied clips. No live-pose state is accessed.
      const {json,bin}=gltfAssets.parseGltfAsset(sourceSnapshot,{maxBytes:byteCeiling});
      return sourceWriter({json,buffers:bin===null?[]:[bin]},{...options,clips});
    });},
    get exportingEnabled(){return model.exportingEnabled;},
    exportPoseGLB(settings={}){return query(()=>{
      if(!settings||typeof settings!=='object'||Array.isArray(settings))throw new GltfAssetError('GLTF_EXPORT_OPTIONS','Invalid export settings');
      // An in-flight snapshot retains its own encoded resources even if the
      // model is disposed while an explicit caller resolver is awaiting I/O.
      const sources=exportSources.slice(),options={...settings};
      // Borrow the model view, not the decoded preparation plan and its arrays.
      // An explicit null opts out; an explicit view can choose other instances.
      if(options.sceneView===undefined)options.sceneView=model.view;
      options.resolveTexture ??= resource=>{
        const found=sources.find(s=>s.resource.view===resource.view&&s.resource.sampler===resource.sampler);
        if(!found)throw new GltfAssetError('GLTF_EXPORT_TEXTURE','Texture was not loaded by this model');
        return found.encoded;
      };
      return model.exportPoseGLB(options);
    });},
    get pickingEnabled(){return model.pickingEnabled;},get pickingStats(){return model.pickingStats;},
    raycast(ray,settings){return query(()=>model.raycast(ray,settings));},
    pick(ndc,cameraSettings,querySettings){return query(()=>model.pick(ndc,cameraSettings,querySettings));},
    update(dt,settings){return invoke(()=>model.update(dt,settings));},
    upload(){return invoke(()=>model.upload());},render(frame){return invoke(()=>presentation?presentation.render(model,frame):model.render(frame));},
    renderCamera(frame,settings){return invoke(()=>presentation?presentation.renderCamera(model,frame,settings):model.renderCamera(frame,settings));},
    async whenIdle(){
      checkTextures();
      try{await Promise.all([model.whenIdle(),presentation?.whenIdle()]);checkTextures();return result;}
      catch(error){if(model.failed||resources.failed||Boolean(presentation?.failed))release();throw error;}
    },
    dispose:release,
  });
  asset=null; // Only explicitly retained export snapshots/images survive construction.
  return result;
}
