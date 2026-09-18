/** URL/GLB -> owned textures + the existing animated WebGPU scene. No second
 * renderer, clock, scheduler or placeholder drawables. Attachments and the frame
 * loop remain caller-owned; authored cameras/lights are available explicitly.
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
  assets={},decode={},textures={},scene={},signal=assets.signal ?? textures.signal,
}={}) {
  if(decode.resolveTexture!=null)throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','The owning loader supplies resolveTexture');
  for(const nested of [assets.signal,textures.signal])if(nested!==undefined&&nested!==signal)throw new GltfAssetError('GLTF_MODEL_LOAD_OPTIONS','Use one construction AbortSignal');
  abort(signal);
  const asset=await loadGltfAsset(source,{...assets,signal});abort(signal);
  const prepared=prepareGltfAnimationModel(asset.json,asset.buffers,decode);
  let resources,model;
  try {
    resources=await createGltfTextureResources(device,prepared.textureRequests,asset.readImage,{...textures,signal});
    abort(signal);
    model=await createGpuDecodedAnimationScene(device,prepared.resolveTextures(resources.resolveTexture),scene);
    abort(signal);
  } catch(error) {
    try{model?.dispose();}finally{resources?.dispose();}
    throw error;
  }
  function release(){model.dispose();resources.dispose();}
  function checkTextures(){
    if(resources.failed){release();throw new GltfTextureError('GLTF_TEXTURE_DEVICE_LOST','Model texture device was lost');}
  }
  function invoke(operation) {
    checkTextures();
    try{operation();return result;}
    catch(error){if(model.failed||resources.failed)release();throw error;}
  }
  const result=Object.freeze({pose:model.pose,view:model.view,cameras:model.cameras,lights:model.lights,
    controller:model.controller,draws:model.draws,deformers:model.deformers,
    source:model.source,diagnostics:model.diagnostics,assetBytes:asset.bytesLoaded,
    get poseVersion(){return model.poseVersion;},get bufferBytes(){return model.bufferBytes;},
    get textureBytes(){return resources.textureBytes;},get disposed(){return model.disposed;},
    get failed(){return model.failed||resources.failed;},
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
