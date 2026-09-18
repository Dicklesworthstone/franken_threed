/**
 * Multi-mesh playback through the existing pose controller, GPU deformation and
 * explicit material/draw pass. No new frame loop, scheduler, loader or renderer
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
 * Drawable material fields match renderer.addMesh: texCoords, vertexColors,
 * baseColorTexture, metallicRoughnessTexture, normalTexture, emissiveTexture,
 * uvTransform, shading, metallicFactor, roughnessFactor, normalScale and
 * emissiveFactor are optional. Normal maps use authored tangents or the renderer's
 * derivative frame. mapCoordinates supplies per-map UVs/local transforms; the
 * shared uvTransform applies afterwards. Lit scenes pass lighting to render(). Textures
 * stay caller-owned; material arrays/descriptors are snapshotted before awaits.
 */
import {createAnimationController} from './animation_controller.mjs';
import {createGpuAnimationDeformer} from './animation_webgpu.mjs';
import {createGpuAnimationRenderer, AnimationRenderError} from './animation_render.mjs';
const fail = (code, message) => { throw new AnimationRenderError(code, message); };
const TEXTURE_FIELDS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture'];

export async function createGpuAnimationScene(device, pose, drawables, {
  renderer: renderOptions = {}, deformer: deformOptions = {}, maxMeshes = 256, maxBytes = 256 * 1024 * 1024,
} = {}) {
  if (!Number.isSafeInteger(maxMeshes) || maxMeshes < 1 || maxMeshes > 4096 ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 1 || !Array.isArray(drawables) ||
      !drawables.length || drawables.length > maxMeshes) fail('ANIMATION_SCENE_LIMIT', 'Invalid mesh count or GPU buffer budget');
  const controller = createAnimationController(pose), initialVersion = pose.version;
  const deformers = [], meshes = [];
  let renderer, deformationBytes = 0, poseVersion = initialVersion, disposed = false, terminal = null, busy = false;
  let lightingAllocated = false, materialComponents = 0;
  function copyMaterialArray(value, key) {
    if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || !Number.isSafeInteger(value.length) ||
        (materialComponents += value.length) > Math.floor(maxBytes / 4)) fail('ANIMATION_SCENE_LIMIT', `Invalid or excessive ${key} storage`);
    if (ArrayBuffer.isView(value)) {
      if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('ANIMATION_SCENE_GEOMETRY', `${key} must have fixed unshared storage`);
      try { new Uint8Array(value.buffer, 0, 0); } catch { fail('ANIMATION_SCENE_GEOMETRY', `${key} is detached`); }
    }
    return Array.from(value);
  }
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
      const allowed = ['geometry', 'indices', 'baseColor', 'doubleSided', 'alphaMode', 'alphaCutoff',
        'texCoords', 'vertexColors', 'mapCoordinates', ...TEXTURE_FIELDS, 'uvTransform', 'shading', 'metallicFactor', 'roughnessFactor', 'emissiveFactor', 'normalScale'];
      for (const key of Object.keys(input)) if (!allowed.includes(key)) fail('ANIMATION_SCENE_GEOMETRY', `Unsupported drawable field: ${key}`);
      const {geometry, indices = null, baseColor = [1,1,1,1], doubleSided = false, alphaMode = 'OPAQUE', alphaCutoff = 0.5} = input;
      if ((!Array.isArray(baseColor) && !ArrayBuffer.isView(baseColor)) || baseColor.length !== 4) fail('ANIMATION_SCENE_GEOMETRY', 'Expected RGBA material color');
      const material = {indices: indices === null ? null : copyMaterialArray(indices, 'indices'),
        baseColor: copyMaterialArray(baseColor, 'baseColor'), doubleSided, alphaMode, alphaCutoff};
      for (const key of ['texCoords', 'vertexColors', 'uvTransform', 'emissiveFactor']) {
        const value = input[key];
        if (value !== undefined) material[key] = value === null ? null : copyMaterialArray(value, key);
      }
      for (const key of ['shading', 'metallicFactor', 'roughnessFactor', 'normalScale']) {
        const value = input[key]; if (value !== undefined) material[key] = value;
      }
      for (const key of TEXTURE_FIELDS) {
        const texture = input[key];
        if (texture === undefined) continue;
        if (texture !== null && (typeof texture !== 'object' || Array.isArray(texture))) fail('ANIMATION_SCENE_GEOMETRY', 'Expected a texture descriptor');
        // Preserve unknown descriptor keys so the renderer rejects them; never
        // silently drop a requested flipY, color conversion or unsupported map.
        material[key] = texture === null ? null : {...texture};
      }
      const mapCoordinates = input.mapCoordinates;
      if (mapCoordinates != null) {
        if (typeof mapCoordinates !== 'object' || Array.isArray(mapCoordinates)) fail('ANIMATION_SCENE_GEOMETRY', 'Expected map coordinates');
        material.mapCoordinates = {};
        for (const [field, coordinate] of Object.entries(mapCoordinates)) {
          if (!TEXTURE_FIELDS.includes(field) || material[field] == null || !coordinate || typeof coordinate !== 'object' || Array.isArray(coordinate)) {
            fail('ANIMATION_SCENE_GEOMETRY', 'Coordinates require an existing material map');
          }
          const copy = {};
          for (const key of Object.keys(coordinate)) {
            if (!['texCoords', 'uvTransform'].includes(key)) fail('ANIMATION_SCENE_GEOMETRY', `Unsupported coordinate field: ${key}`);
            if (coordinate[key] !== undefined) copy[key] = copyMaterialArray(coordinate[key], field + ' ' + key);
          }
          material.mapCoordinates[field] = copy;
        }
      }
      return {geometry, material};
    });
    if ((renderOptions.maxDraws ?? drawables.length) < drawables.length || (renderOptions.maxMeshes ?? maxMeshes) < drawables.length) {
      fail('ANIMATION_SCENE_LIMIT', 'Renderer capacity cannot hold the scene');
    }
    renderer = await createGpuAnimationRenderer(device, {...renderOptions, maxDraws: renderOptions.maxDraws ?? drawables.length,
      maxMeshes: renderOptions.maxMeshes ?? maxMeshes, maxBytes: Math.min(maxBytes, renderOptions.maxBytes ?? maxBytes)});
    unchanged();
    for (const {geometry, material} of inputs) {
      // Reserve the renderer's pending auxiliary buffers before allocating the
      // next deformer. uint32 indices conservatively bound padded uint16 data.
      const vertices = geometry?.positions?.length / 3;
      const surface = material.texCoords != null || material.vertexColors != null || TEXTURE_FIELDS.some(key => material[key] != null);
      if (surface && (!Number.isSafeInteger(vertices) || vertices < 1)) fail('ANIMATION_SCENE_GEOMETRY', 'Surface attributes require XYZ geometry');
      const lit = material.shading === 'lambert' || material.shading === 'metallic-roughness';
      const surfaceStride = 24 + Object.keys(material.mapCoordinates ?? {}).length * 8;
      const reserve = (material.indices?.length ?? 0) * 4 + (surface ? vertices * surfaceStride : 0) + (lit && !lightingAllocated ? 544 : 0);
      const remaining = maxBytes - renderer.allocatedBytes - deformationBytes - reserve;
      if (remaining < 1) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exhausted');
      const gpu = await createGpuAnimationDeformer(device, pose, geometry, {...deformOptions,
        maxBytes: Math.min(remaining, deformOptions.maxBytes ?? 128 * 1024 * 1024)});
      deformers.push(gpu); deformationBytes += gpu.bufferBytes; unchanged();
      meshes.push(await renderer.addMesh(gpu, material)); lightingAllocated ||= lit; unchanged();
      if (renderer.allocatedBytes + deformationBytes > maxBytes) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exceeded');
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
