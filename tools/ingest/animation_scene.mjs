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
 * baseColorTexture, metallicRoughnessTexture, normalTexture, emissiveTexture, occlusionTexture,
 * uvTransform, shading, metallicFactor, roughnessFactor, normalScale and
 * emissiveFactor and occlusionStrength are optional. Normal maps use authored tangents or the renderer's
 * derivative frame. mapCoordinates supplies per-map UVs/local transforms; the
 * shared uvTransform applies afterwards. Lit scenes pass lighting to render(). Textures
 * stay caller-owned; material arrays/descriptors are snapshotted before awaits.
 * renderer.environment:true receives a borrowed frame.environment:{map,...}.
 * Its extra 64-byte receiver uniform is reserved in the scene GPU budget;
 * environment textures remain caller-owned. See ANIMATION_ENVIRONMENT.md.
 *
 * Implicit scene draws render opaque/masked meshes first, preserving their
 * relative order, then blended meshes back-to-front by projected node origin.
 * Sorting is recomputed for each camera and current uploaded pose. Ties retain
 * source order. An explicit frame.draws list always preserves caller order;
 * sortObjects:false also preserves the original implicit list. Intersecting or
 * unusually offset transparent geometry may need an explicit draw list.
 *
 * frustumCulling:true opts into conservative current-pose bounds for implicit
 * draws. Explicit frame.draws bypasses culling, including custom world matrices.
 * maxBoundsBytes/maxBoundsComponents separately bound CPU summaries/source scans.
 * Cull decisions do not skip GPU deformation or change update/submit timing.
 * cullingStats acknowledges the latest successful render submission, not GPU
 * completion. Overflowed bounds remain visible. No custom shader displacement
 * or occlusion is assumed; native backend pixel equivalence is not certified.
 *
 * shadow:{lightIndex:0} owns a fitted directional/spot depth map, registers
 * OPAQUE/MASK materials against these same deformers, and submits depth before
 * each implicit color frame. BLEND requires blend:'skip' or casters exclusions.
 * Fitting uses all current-pose bounds, including off-camera casters. Map GPU
 * storage and CPU summaries have separate shadow.maxBytes/maxBoundsBytes budgets;
 * shadowBytes/shadowBoundsBytes report them. The color receiver's extra 96-byte
 * uniform is charged to this scene's maxBytes. shadowStats reports the last
 * successful color submission, not GPU completion. frame.shadow:null opts out;
 * an explicit map overrides automatic shadows. Custom frame.draws require one
 * of those explicit choices rather than guessing matching caster transforms.
 * The depth pass may have submitted before a recoverable color validation error;
 * neither pose advancement nor GPU submissions are rolled back. See ANIMATION_SHADOWS.md.
 */
import {createAnimationDrawOrder} from './animation_draw_order.mjs';
import {createAnimationController} from './animation_controller.mjs';
import {createGpuAnimationDeformer} from './animation_webgpu.mjs';
import {createGpuAnimationRenderer, AnimationRenderError} from './animation_render.mjs';
const fail = (code, message) => { throw new AnimationRenderError(code, message); };
const TEXTURE_FIELDS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture', 'occlusionTexture'];

export async function createGpuAnimationScene(device, pose, drawables, {
  shadow = null, sortObjects = true, frustumCulling = false, maxBoundsBytes = 16*1024*1024, maxBoundsComponents = 16777216, renderer: renderOptions = {}, deformer: deformOptions = {}, maxMeshes = 256, maxBytes = 256 * 1024 * 1024,
} = {}) {
  if (!Number.isSafeInteger(maxMeshes) || maxMeshes < 1 || maxMeshes > 4096 ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 1 || !Array.isArray(drawables) ||
      !drawables.length || drawables.length > maxMeshes) fail('ANIMATION_SCENE_LIMIT', 'Invalid mesh count or GPU buffer budget');
  if (typeof sortObjects !== 'boolean') fail('ANIMATION_SCENE_SORT', 'sortObjects must be boolean');
  if (typeof frustumCulling !== 'boolean' || (frustumCulling && (!Number.isSafeInteger(maxBoundsBytes) || maxBoundsBytes < 1 || !Number.isSafeInteger(maxBoundsComponents) || maxBoundsComponents < 1))) fail('ANIMATION_SCENE_CULL', 'Invalid frustum culling options');
  const controller = createAnimationController(pose), initialVersion = pose.version;
  const deformers = [], meshes = [], ordering = [];
  let drawOrder, sceneShadows, shadowStats = null, cullingStats = null;
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
    sceneShadows?.dispose(); drawOrder?.dispose();
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
    if (shadow !== null && (typeof shadow !== 'object' || Array.isArray(shadow))) fail('ANIMATION_SCENE_SHADOW', 'Expected shadow options or null');
    // All scalar options and the only array option are captured before awaits.
    const shadowOptions = shadow === null ? null : {...shadow};
    if (shadowOptions?.casters !== undefined) {
      if (!Array.isArray(shadowOptions.casters) || shadowOptions.casters.length !== drawables.length) fail('ANIMATION_SCENE_SHADOW', 'Expected one caster boolean per drawable');
      shadowOptions.casters = Array.from(shadowOptions.casters);
    }
    renderOptions = {...renderOptions};
    if (shadowOptions) {
      if (renderOptions.shadows === false || renderOptions.format === null) fail('ANIMATION_SCENE_SHADOW', 'Automatic shadows require a shadow-enabled color renderer');
      renderOptions.shadows = true;
    }
    // Snapshot material/index descriptors before the first await. Deformation
    // performs its existing bounded geometry snapshot during each creation.
    const inputs = drawables.map(input => {
      if (!input || typeof input !== 'object') fail('ANIMATION_SCENE_GEOMETRY', 'Expected a drawable descriptor');
      const allowed = ['geometry', 'indices', 'baseColor', 'doubleSided', 'alphaMode', 'alphaCutoff',
        'texCoords', 'vertexColors', 'mapCoordinates', ...TEXTURE_FIELDS, 'uvTransform', 'shading', 'metallicFactor', 'roughnessFactor', 'emissiveFactor', 'normalScale', 'occlusionStrength'];
      for (const key of Object.keys(input)) if (!allowed.includes(key)) fail('ANIMATION_SCENE_GEOMETRY', `Unsupported drawable field: ${key}`);
      const {geometry, indices = null, baseColor = [1,1,1,1], doubleSided = false, alphaMode = 'OPAQUE', alphaCutoff = 0.5} = input;
      if ((!Array.isArray(baseColor) && !ArrayBuffer.isView(baseColor)) || baseColor.length !== 4) fail('ANIMATION_SCENE_GEOMETRY', 'Expected RGBA material color');
      const material = {indices: indices === null ? null : copyMaterialArray(indices, 'indices'),
        baseColor: copyMaterialArray(baseColor, 'baseColor'), doubleSided, alphaMode, alphaCutoff};
      for (const key of ['texCoords', 'vertexColors', 'uvTransform', 'emissiveFactor']) {
        const value = input[key];
        if (value !== undefined) material[key] = value === null ? null : copyMaterialArray(value, key);
      }
      for (const key of ['shading', 'metallicFactor', 'roughnessFactor', 'normalScale', 'occlusionStrength']) {
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
    if (shadowOptions) {
      const {prepareAnimationSceneShadows} = await import('./animation_scene_shadow.mjs');
      unchanged();
      sceneShadows = prepareAnimationSceneShadows(pose, inputs, shadowOptions);
      unchanged();
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
      const reserve = (material.indices?.length ?? 0) * 4 + (surface ? vertices * surfaceStride : 0) + (lit && !lightingAllocated ? 544 + (renderOptions.shadows ? 96 : 0) + (renderOptions.environment ? 64 : 0) : 0);
      const remaining = maxBytes - renderer.allocatedBytes - deformationBytes - reserve;
      if (remaining < 1) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exhausted');
      const gpu = await createGpuAnimationDeformer(device, pose, geometry, {...deformOptions,
        maxBytes: Math.min(remaining, deformOptions.maxBytes ?? 128 * 1024 * 1024)});
      deformers.push(gpu); deformationBytes += gpu.bufferBytes; unchanged();
      meshes.push(await renderer.addMesh(gpu, material)); lightingAllocated ||= lit; unchanged();
      ordering.push({mesh: meshes.at(-1), deformer: gpu, alphaMode: material.alphaMode, ...(frustumCulling ? {geometry} : {})});
      if (renderer.allocatedBytes + deformationBytes > maxBytes) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exceeded');
    }
    if (sortObjects || frustumCulling) drawOrder = createAnimationDrawOrder(ordering, {pose, sortObjects, frustumCulling, maxBoundsBytes, maxBoundsComponents});
    if (sceneShadows) { await sceneShadows.initialize(device, deformers); unchanged(); }
    ordering.length = 0;
  } catch (error) { release(); throw error; }
  function exclusive(operation) {
    live(); if (busy) fail('ANIMATION_SCENE_REENTRANT', 'GPU scene operation cannot be reentered');
    busy = true; try { return operation(); } finally { busy = false; }
  }
  function upload() {
    try {
      const nextVersion = pose.version;
      for (const gpu of deformers) gpu.update();
      if (frustumCulling) drawOrder.updateBounds();
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
    get boundsBytes() { return drawOrder?.boundsBytes ?? 0; },
    get cullingStats() { return cullingStats; },
    get shadowEnabled() { return sceneShadows !== undefined; },
    get shadowBytes() { return sceneShadows?.allocatedBytes ?? 0; },
    get shadowBoundsBytes() { return sceneShadows?.boundsBytes ?? 0; },
    get shadowStats() { return shadowStats; },
    get disposed() { return disposed; }, get failed() { return terminal !== null || renderer.failed || !!sceneShadows?.failed || deformers.some(gpu => gpu.failed); },
    update(delta, options) { return exclusive(() => { controller.update(delta, options); return upload(); }); },
    upload() { return exclusive(upload); },
    render(frame) { return exclusive(() => {
      synchronized();
      try {
        const prepared = {...frame}, implicit = prepared.draws == null;
        prepared.draws ??= drawOrder ? drawOrder.order(prepared.viewProjection) : meshes;
        // Use the exact camera snapshot tested by culling for shader packing too.
        if (frustumCulling && implicit) prepared.viewProjection = drawOrder.viewProjection;
        let automatic = null;
        if (sceneShadows && prepared.shadow === undefined) {
          if (!implicit) fail('ANIMATION_SCENE_SHADOW', 'Explicit draws require an explicit shadow map or shadow:null');
          synchronized();
          automatic = sceneShadows.render(prepared.lighting);
          prepared.lighting = automatic.lighting; prepared.shadow = automatic.shadow;
        }
        renderer.render(prepared);
        if (sceneShadows) shadowStats = automatic?.stats ?? null;
        if (frustumCulling) cullingStats = implicit ? drawOrder.lastCulling : Object.freeze({
          poseVersion, testedMeshes: 0, culledMeshes: 0, submittedDraws: prepared.draws.length,
        });
      }
      catch (error) { if (renderer.failed || sceneShadows?.failed) failGroup(error); throw error; }
      return scene;
    }); },
    async whenIdle() {
      live();
      try { await Promise.all([renderer.whenIdle(), sceneShadows?.whenIdle(), ...deformers.map(gpu => gpu.whenIdle())]); }
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
