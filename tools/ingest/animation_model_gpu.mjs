/** glTF JSON + buffers -> the existing animated WebGPU scene.
 * Device, image/sampler uploads and attachments remain caller supplied.
 * Authored cameras/lights are optional; explicit external frames keep their API.
 * No new renderer, scheduler, async asset pool or animation clock.
 * This factory owns the new pose, unlike createGpuAnimationScene's borrowed pose.
 * All decode/material validation and texture resolution finish before GPU work.
 */
import {decodeGltfAnimationModel,createGltfSceneView,GltfSceneViewError} from './animation_model.mjs';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createGpuAnimationScene} from './animation_scene.mjs';
const fail=(code,message)=>{throw new GltfSceneViewError('GLTF_VIEW_'+code,message);};

/**
 * const model=await createGpuGltfAnimationScene(device,json,buffers,{
 *   decode:{scene:0,resolveTexture},scene:{renderer:{format:'rgba8unorm-srgb'}}
 * });
 * model.controller.createAction(0).play();
 * model.update(dt);model.render({colorView,depthView,viewProjection,lighting});
 * // Or use an authored view and all selected-scene punctual lights:
 * model.renderCamera({colorView,depthView},{cameraNode,aspectRatio:width/height});
 * await model.whenIdle();model.dispose();
 * CPU pose arithmetic and GPU f32 deformation retain their existing profiles.
 * The API is a core triangle/direct-light material route, not full glTF parity.
 */
export async function createGpuGltfAnimationScene(device, json, buffers, {decode={},scene:options={}}={}) {
  return createGpuDecodedAnimationScene(device,decodeGltfAnimationModel(json,buffers,decode),options);
}

/** Consume the real decoded-model output, avoiding a second geometry decode
 * after asynchronous texture preparation. Texture resources remain borrowed.
 */
export async function createGpuDecodedAnimationScene(device,prepared,options={}) {
  const pose=createAnimationPlayer(prepared.definition);
  let scene,view,busy=false;
  try {
    view=createGltfSceneView(pose,prepared.sceneView ?? {
      format:'f3d-gltf-scene-view-v1',nodeCount:pose.nodeCount,cameras:[],lights:[],
    });
    scene=await createGpuAnimationScene(device,pose,prepared.drawables,options);
  } catch(error){pose.dispose();throw error;}
  // Guard the new camera/frame preparation as well as the underlying draw call:
  // source getters cannot reenter/dispose the model before scene.render locks.
  function invoke(operation) {
    if(busy)fail('REENTRANT','Model operation cannot be reentered');
    busy=true;
    try {operation();return result;}
    catch(error){if(scene.failed)pose.dispose();throw error;}
    finally {busy=false;}
  }
  const result=Object.freeze({pose,view,cameras:view.cameras,lights:view.lights,
    controller:scene.controller,draws:scene.draws,deformers:scene.deformers,
    source:Object.freeze(prepared.source),diagnostics:Object.freeze(prepared.diagnostics),
    get poseVersion(){return scene.poseVersion;},get bufferBytes(){return scene.bufferBytes;},
    get disposed(){return scene.disposed;},get failed(){return scene.failed;},
    update(dt,settings){return invoke(()=>scene.update(dt,settings));},
    upload(){return invoke(()=>scene.upload());},render(frame){return invoke(()=>scene.render(frame));},
    renderCamera(frame,settings){return invoke(()=>{
      if(!frame||typeof frame!=='object'||Array.isArray(frame))fail('FRAME','Expected render attachments/options');
      const input={...frame};
      if(Object.hasOwn(input,'viewProjection')||Object.hasOwn(input,'lighting'))fail('FRAME','renderCamera supplies viewProjection and lighting; use render for external frames');
      const sample=view.sample(settings);
      if(sample.poseVersion!==scene.poseVersion)fail('STALE','Upload the current pose before camera rendering');
      scene.render({...input,viewProjection:sample.viewProjection,lighting:sample.lighting});
    });},
    async whenIdle(){try{await scene.whenIdle();return result;}catch(error){if(scene.failed)pose.dispose();throw error;}},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during model operation');scene.dispose();pose.dispose();},
  });
  return result;
}
