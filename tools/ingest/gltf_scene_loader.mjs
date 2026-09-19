/** URL/GLB -> owned textures + the existing animated WebGPU scene. No second
 * renderer, clock, scheduler or placeholder drawables. Attachments and the frame
 * loop remain caller-owned; authored cameras/lights are available explicitly.
 * picking:true opts in to synchronous current-pose geometric selection.
 * exporting:true enables self-contained posed GLB export with cached source images
 * and current-pose authored cameras/lights. Export sceneView:null for meshes only.
 */
import {loadGltfAsset,GltfAssetError} from './gltf_asset.mjs';
import {prepareGltfAnimationModel} from './animation_model.mjs';
import {createGltfTextureResources,GltfTextureError} from './gltf_textures.mjs';
import {createGpuDecodedAnimationScene} from './animation_model_gpu.mjs';
const abort=signal=>{if(signal?.aborted)throw signal.reason ?? new DOMException('Aborted','AbortError');};

/** Nested assets/decode/textures/scene options belong to their existing stages.
 * signal cancels construction cooperatively; a late native decoder/GPU result is
 * disposed before rejection. A returned scene is owned until explicit dispose().
 * The device is never destroyed. Custom borrowed textures use the original
 * createGpuGltfAnimationScene API, not this owning loader.
 */
export async function loadGpuGltfAnimationScene(device,source,{
  assets={},decode={},textures={},scene={},picking=false,exporting=false,signal=assets.signal ?? textures.signal,
}={}) {
  if(decode.resolveTexture!=null)throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','The owning loader supplies resolveTexture');
  for(const nested of [assets.signal,textures.signal])if(nested!==undefined&&nested!==signal)throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','Use one construction AbortSignal');
  abort(signal);
  const asset=await loadGltfAsset(source,{...assets,signal});abort(signal);
  const prepared=prepareGltfAnimationModel(asset.json,asset.buffers,decode);
  let resources,model,exportSources=[];
  const textureOptions={...textures,signal};
  try {
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
    model=await createGpuDecodedAnimationScene(device,prepared.resolveTextures(resources.resolveTexture),{...scene,picking,exporting});
    abort(signal);
  } catch(error) {
    try{model?.dispose();}finally{resources?.dispose();}
    throw error;
  }
  function release(){model.dispose();resources.dispose();exportSources=[];}
  function checkTextures(){
    if(resources.failed){release();throw new GltfTextureError('GLTF_TEXTURE_DEVICE_LOST','Model texture device was lost');}
  }
  function query(operation) {
    checkTextures();
    try{return operation();}
    catch(error){if(model.failed||resources.failed)release();throw error;}
  }
  function invoke(operation){query(operation);return result;}
  const result=Object.freeze({pose:model.pose,view:model.view,cameras:model.cameras,lights:model.lights,
    controller:model.controller,draws:model.draws,deformers:model.deformers,
    source:model.source,diagnostics:model.diagnostics,assetBytes:asset.bytesLoaded,
    get poseVersion(){return model.poseVersion;},get bufferBytes(){return model.bufferBytes;},
    get textureBytes(){return resources.textureBytes;},get disposed(){return model.disposed;},
    get failed(){return model.failed||resources.failed;},
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
    upload(){return invoke(()=>model.upload());},render(frame){return invoke(()=>model.render(frame));},
    renderCamera(frame,settings){return invoke(()=>model.renderCamera(frame,settings));},
    async whenIdle(){
      checkTextures();
      try{await model.whenIdle();checkTextures();return result;}
      catch(error){if(model.failed||resources.failed)release();throw error;}
    },
    dispose:release,
  });
  return result;
}
