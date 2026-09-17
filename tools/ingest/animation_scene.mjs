/**
 * Multi-mesh playback through the existing pose controller, GPU deformation and
 * explicit unlit draw pass. No new frame loop, scheduler, loader or renderer
 * routing. The caller supplies decoded geometry, a pose, device and attachments.
 *
 * const scene = await createGpuAnimationScene(device, pose, [
 *   {geometry, indices, baseColor: [1,0,0,1]}, ...
 * ], {renderer: {format: 'rgba8unorm-srgb'}});
 * scene.controller.createAction(0).play();
 * scene.update(deltaSeconds); // one controller advance, all mesh deformations
 * scene.render({colorView, depthView, viewProjection});
 * await scene.whenIdle(); // GPU completion, not just successful submission
 *
 * update and render are separate source-observable boundaries. A failed CPU
 * sample leaves the previous pose usable. A GPU upload/submission failure is
 * terminal: owned resources are released and no partial new frame is drawn.
 * CPU time already advanced before such a GPU failure is NOT rolled back.
 * After sampling the borrowed pose directly, call upload() before render();
 * stale CPU/GPU versions are rejected, never silently mixed.
 *
 * Keep pose and geometry stable until initialization resolves. An observed pose
 * version change during initialization cancels the construction and releases
 * everything created so far. Scene disposal owns its controller, draw resources
 * and deformers, never the borrowed pose/device/attachments. maxBytes bounds
 * owned GPU buffers (not driver pipelines, CPU pose arrays or caller textures).
 * Geometry/material support is exactly the explicitly selected unlit slice.
 */
import {createAnimationController} from './animation_controller.mjs';
import {createGpuAnimationDeformer} from './animation_webgpu.mjs';
import {createGpuAnimationRenderer, AnimationRenderError} from './animation_render.mjs';
const fail = (code, message) => { throw new AnimationRenderError(code, message); };

export async function createGpuAnimationScene(device, pose, drawables, {
  renderer: renderOptions = {}, deformer: deformOptions = {}, maxMeshes = 256, maxBytes = 256 * 1024 * 1024,
} = {}) {
  if (!Number.isSafeInteger(maxMeshes) || maxMeshes < 1 || maxMeshes > 4096 ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 1 || !Array.isArray(drawables) ||
      !drawables.length || drawables.length > maxMeshes) fail('ANIMATION_SCENE_LIMIT', 'Invalid mesh count or GPU buffer budget');
  const controller = createAnimationController(pose), initialVersion = pose.version;
  const deformers = [], meshes = [];
  let renderer, deformationBytes = 0, poseVersion = initialVersion, disposed = false, terminal = null, busy = false;
  function release() {
    for (const mesh of meshes) mesh.dispose();
    for (const gpu of deformers) gpu.dispose();
    renderer?.dispose(); controller.dispose(); deformationBytes = 0;
  }
  function failGroup(error) { terminal ??= error; release(); throw terminal; }
  function live() {
    if (disposed) fail('ANIMATION_SCENE_DISPOSED', 'GPU animation scene has been disposed');
    if (terminal) throw terminal;
    if (pose.disposed) fail('ANIMATION_SCENE_POSE', 'Borrowed pose has been disposed');
  }
  function unchanged() {
    if (pose.disposed || pose.version !== initialVersion) fail('ANIMATION_SCENE_CHANGED', 'Pose changed during GPU scene initialization');
  }
  try {
    // Snapshot material/index descriptors before the first await. Deformation
    // performs its existing bounded geometry snapshot during each creation.
    const inputs = drawables.map(input => {
      if (!input || typeof input !== 'object') fail('ANIMATION_SCENE_GEOMETRY', 'Expected a drawable descriptor');
      for (const key of Object.keys(input)) if (!['geometry', 'indices', 'baseColor', 'doubleSided', 'alphaMode', 'alphaCutoff'].includes(key)) {
        fail('ANIMATION_SCENE_GEOMETRY', `Unsupported drawable field: ${key}`);
      }
      const {geometry, indices = null, baseColor = [1,1,1,1], doubleSided = false, alphaMode = 'OPAQUE', alphaCutoff = 0.5} = input;
      if (indices !== null && ((!Array.isArray(indices) && !ArrayBuffer.isView(indices)) ||
          !Number.isSafeInteger(indices.length) || indices.length > Math.floor(maxBytes / 4))) fail('ANIMATION_SCENE_LIMIT', 'Index storage exceeds budget');
      if ((!Array.isArray(baseColor) && !ArrayBuffer.isView(baseColor)) || baseColor.length !== 4) fail('ANIMATION_SCENE_GEOMETRY', 'Expected RGBA material color');
      return {geometry, material: {indices: indices === null ? null : Array.from(indices), baseColor: Array.from(baseColor), doubleSided, alphaMode, alphaCutoff}};
    });
    if ((renderOptions.maxDraws ?? drawables.length) < drawables.length || (renderOptions.maxMeshes ?? maxMeshes) < drawables.length) {
      fail('ANIMATION_SCENE_LIMIT', 'Renderer capacity cannot hold the scene');
    }
    renderer = await createGpuAnimationRenderer(device, {...renderOptions, maxDraws: renderOptions.maxDraws ?? drawables.length,
      maxMeshes: renderOptions.maxMeshes ?? maxMeshes, maxBytes: Math.min(maxBytes, renderOptions.maxBytes ?? maxBytes)});
    unchanged();
    for (const {geometry, material} of inputs) {
      // Reserve a conservative uint32 index allocation before allocating the
      // next deformer. The renderer may use less (padded uint16) in practice.
      const indexReserve = (material.indices?.length ?? 0) * 4;
      const remaining = maxBytes - renderer.allocatedBytes - deformationBytes - indexReserve;
      if (remaining < 1) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exhausted');
      const gpu = await createGpuAnimationDeformer(device, pose, geometry, {...deformOptions,
        maxBytes: Math.min(remaining, deformOptions.maxBytes ?? 128 * 1024 * 1024)});
      deformers.push(gpu); deformationBytes += gpu.bufferBytes; unchanged();
      meshes.push(await renderer.addMesh(gpu, material)); unchanged();
    }
  } catch (error) { release(); throw error; }
  function exclusive(operation) {
    live(); if (busy) fail('ANIMATION_SCENE_REENTRANT', 'GPU scene operation cannot be reentered');
    busy = true; try { return operation(); } finally { busy = false; }
  }
  function upload() {
    try {
      const nextVersion = pose.version;
      for (const gpu of deformers) gpu.update();
      if (deformers.some(gpu => gpu.poseVersion !== nextVersion) || pose.version !== nextVersion) {
        fail('ANIMATION_SCENE_CHANGED', 'Pose changed during GPU upload');
      }
      poseVersion = nextVersion;
    } catch (error) { failGroup(error); }
    return scene;
  }
  function synchronized() {
    if (pose.version !== poseVersion || deformers.some(gpu => gpu.poseVersion !== poseVersion || gpu.disposed || gpu.failed)) {
      fail('ANIMATION_SCENE_STALE', 'Upload a complete current pose before rendering');
    }
  }
  const scene = Object.freeze({pose, controller, draws: Object.freeze(meshes), deformers: Object.freeze(deformers),
    get poseVersion() { return poseVersion; },
    get bufferBytes() { return renderer.allocatedBytes + deformationBytes; },
    get disposed() { return disposed; }, get failed() { return terminal !== null || renderer.failed || deformers.some(gpu => gpu.failed); },
    update(delta, options) { return exclusive(() => { controller.update(delta, options); return upload(); }); },
    upload() { return exclusive(upload); },
    render(frame) { return exclusive(() => {
      synchronized();
      try { renderer.render({...frame, draws: frame?.draws ?? meshes}); }
      catch (error) { if (renderer.failed) failGroup(error); throw error; }
      return scene;
    }); },
    async whenIdle() {
      live();
      try { await Promise.all([renderer.whenIdle(), ...deformers.map(gpu => gpu.whenIdle())]); }
      catch (error) { failGroup(error); }
      live(); return scene;
    },
    dispose() {
      if (busy) fail('ANIMATION_SCENE_REENTRANT', 'Cannot dispose during a scene operation');
      if (!disposed) { disposed = true; release(); }
    },
  });
  return scene;
}
