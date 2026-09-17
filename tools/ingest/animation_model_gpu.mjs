/** glTF JSON + buffers -> the existing animated WebGPU scene.
 * Device, image/sampler uploads, attachments, camera and lighting remain caller
 * supplied. No new renderer, scheduler, async asset pool or animation clock.
 * This factory owns the new pose, unlike createGpuAnimationScene's borrowed pose.
 * All decode/material validation and texture resolution finish before GPU work.
 */
import {decodeGltfAnimationModel} from './animation_model.mjs';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createGpuAnimationScene} from './animation_scene.mjs';

/**
 * const model=await createGpuGltfAnimationScene(device,json,buffers,{
 *   decode:{scene:0,resolveTexture},scene:{renderer:{format:'rgba8unorm-srgb'}}
 * });
 * model.controller.createAction(0).play();
 * model.update(dt);model.render({colorView,depthView,viewProjection,lighting});
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
  let scene;
  try {scene=await createGpuAnimationScene(device,pose,prepared.drawables,options);}
  catch(error){pose.dispose();throw error;}
  // Keep source IDs for selection/UI without adding invalid renderer fields to
  // drawable descriptors. Forward through the existing scene's reentry guards.
  function invoke(operation) {
    try {operation();return result;}
    catch(error){if(scene.failed)pose.dispose();throw error;}
  }
  const result=Object.freeze({pose,controller:scene.controller,draws:scene.draws,deformers:scene.deformers,
    source:Object.freeze(prepared.source),diagnostics:Object.freeze(prepared.diagnostics),
    get poseVersion(){return scene.poseVersion;},get bufferBytes(){return scene.bufferBytes;},
    get disposed(){return scene.disposed;},get failed(){return scene.failed;},
    update(dt,settings){return invoke(()=>scene.update(dt,settings));},
    upload(){return invoke(()=>scene.upload());},render(frame){return invoke(()=>scene.render(frame));},
    async whenIdle(){try{await scene.whenIdle();return result;}catch(error){if(scene.failed)pose.dispose();throw error;}},
    dispose(){scene.dispose();pose.dispose();},
  });
  return result;
}
