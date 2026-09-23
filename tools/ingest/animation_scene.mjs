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
 * Clearcoat factors and three independent linear maps are forwarded unchanged;
 * its 16-byte material uniform is reserved before each geometry allocation.
 * rigidGeometry:true opts into shared immutable vertex storage for rigid nodes.
 * Transform-only updates then issue no deformation dispatch. Skinned, morphed
 * and dynamic-flat meshes retain the existing compute path; materials, draw
 * order, culling and shadow registration remain per mesh. No draw batching.
 * Shared GPU bytes count once in maxBytes. The pool retains the same number of
 * CPU comparison bytes plus one bounded incoming snapshot; rigidGeometryStats
 * reports unique buffers and mesh handles. Omission preserves the compute path.
 * deformationBatch:true validates compute inputs as a group, omits dispatches
 * with unchanged submitted f32 palette/morph words, and submits changed meshes
 * once at the existing update/upload boundary. All transforms and pose versions
 * remain current, including inactive LOD levels and off-camera shadow casters.
 * No render-time deformation, implicit asynchronous work, or GPU readback.
 * deformationStats counts the latest successful batch upload, excluding initial
 * construction. deformationInputCacheBytes reports CPU input shadows separately
 * from the unchanged GPU buffer budget. Omission retains individual submissions.
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
 * lod:{groups:[{node,levels:[{distance:0,drawIndices:[...]},...]}]} selects
 * whole drawable levels before sorting/culling. Implicit frames then require
 * lodCamera:{position:[x,y,z],key:'main',zoom:1}; keys isolate bounded hysteresis
 * histories. lodStats publishes the selection only after successful color
 * submission. Explicit frame.draws bypasses LOD; resetLodCamera(key) releases a
 * history slot. All geometry stays resident and deformation still updates every
 * mesh. This is distance LOD, not screen-error selection or asset streaming.
 *
 * shadow:{lightIndex:0} owns a fitted directional/spot depth map, registers
 * OPAQUE/MASK materials against these same deformers, and submits depth before
 * each implicit color frame. BLEND requires blend:'skip' or casters exclusions.
 * Fitting uses active-LOD current-pose bounds, including off-camera casters. Map GPU
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
import * as gpuDeformation from './animation_webgpu.mjs';
const {createGpuAnimationDeformer, updateGpuAnimationDeformers} = gpuDeformation;
import {createGpuAnimationRenderer, AnimationRenderError} from './animation_render.mjs';
const fail = (code, message) => { throw new AnimationRenderError(code, message); };
const TEXTURE_FIELDS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture', 'occlusionTexture',
  'clearcoatTexture', 'clearcoatRoughnessTexture', 'clearcoatNormalTexture', 'specularTexture', 'gradientTexture'];
const COAT_FIELDS = ['clearcoatFactor', 'clearcoatRoughnessFactor', 'clearcoatNormalScale'];

export async function createGpuAnimationScene(device, pose, drawables, {
  deformationBatch = false, rigidGeometry = false, shadow = null, lod = null, sortObjects = true, frustumCulling = false, maxBoundsBytes = 16*1024*1024, maxBoundsComponents = 16777216, renderer: renderOptions = {}, deformer: deformOptions = {}, maxMeshes = 256, maxBytes = 256 * 1024 * 1024,
} = {}) {
  if (!Number.isSafeInteger(maxMeshes) || maxMeshes < 1 || maxMeshes > 4096 ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 1 || !Array.isArray(drawables) ||
      !drawables.length || drawables.length > maxMeshes) fail('ANIMATION_SCENE_LIMIT', 'Invalid mesh count or GPU buffer budget');
  if (typeof deformationBatch !== 'boolean') fail('ANIMATION_SCENE_BATCH', 'deformationBatch must be boolean');
  if (typeof rigidGeometry !== 'boolean') fail('ANIMATION_SCENE_RIGID', 'rigidGeometry must be boolean');
  if (typeof sortObjects !== 'boolean') fail('ANIMATION_SCENE_SORT', 'sortObjects must be boolean');
  if (typeof frustumCulling !== 'boolean' || (frustumCulling && (!Number.isSafeInteger(maxBoundsBytes) || maxBoundsBytes < 1 || !Number.isSafeInteger(maxBoundsComponents) || maxBoundsComponents < 1))) fail('ANIMATION_SCENE_CULL', 'Invalid frustum culling options');
  const controller = createAnimationController(pose), initialVersion = pose.version;
  const deformers = [], computeDeformers = [], rigidDeformers = [], meshes = [], ordering = [];
  let rigidApi, rigidPool, deformationStats = null;
  const deformationTotal = () => deformationBytes + (rigidPool?.bufferBytes ?? 0);
  let drawOrder, sceneShadows, lodState, shadowStats = null, cullingStats = null, lodStats = null;
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
    sceneShadows?.dispose(); drawOrder?.dispose(); lodState?.dispose(); lodStats = null; deformationStats = null;
    for (const mesh of meshes) mesh.dispose();
    for (const gpu of deformers) gpu.dispose();
    rigidPool?.dispose(); renderer?.dispose(); controller.dispose(); deformationBytes = 0;
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
    // Bound and copy nested LOD options before any asynchronous construction.
    // The selector validates values and unknown fields after its lazy import.
    let lodOptions = null;
    if (lod !== null) {
      if (!lod || typeof lod !== 'object' || Array.isArray(lod)) fail('ANIMATION_SCENE_LOD', 'Expected LOD options or null');
      lodOptions = {...lod};
      if (!Array.isArray(lodOptions.groups) || !lodOptions.groups.length || lodOptions.groups.length > drawables.length) fail('ANIMATION_SCENE_LOD', 'Invalid LOD groups');
      let levels = 0, indices = 0;
      lodOptions.groups = lodOptions.groups.map(group => {
        if (!group || typeof group !== 'object' || Array.isArray(group) || !Array.isArray(group.levels) || !group.levels.length ||
            (levels += group.levels.length) > drawables.length) fail('ANIMATION_SCENE_LOD', 'Invalid aggregate LOD levels');
        return {...group, levels: group.levels.map(level => {
          if (!level || typeof level !== 'object' || Array.isArray(level) || !Array.isArray(level.drawIndices) || !level.drawIndices.length ||
              (indices += level.drawIndices.length) > drawables.length) fail('ANIMATION_SCENE_LOD', 'Invalid aggregate LOD indices');
          return {...level, drawIndices: Array.from(level.drawIndices)};
        })};
      });
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
      const allowed = ['geometry', 'indices', 'baseColor', 'doubleSided', 'side', 'depthTest', 'depthWrite', 'depthCompare', 'colorWrite', 'alphaMode', 'alphaCutoff',
        'texCoords', 'vertexColors', 'mapCoordinates', ...TEXTURE_FIELDS, ...COAT_FIELDS, 'uvTransform', 'shading', 'flatShading', 'specularColor', 'shininess', 'metallicFactor', 'roughnessFactor', 'emissiveFactor', 'normalScale', 'occlusionStrength'];
      for (const key of Object.keys(input)) if (!allowed.includes(key)) fail('ANIMATION_SCENE_GEOMETRY', `Unsupported drawable field: ${key}`);
      const {geometry, indices = null, baseColor = [1,1,1,1], doubleSided = false, alphaMode = 'OPAQUE', alphaCutoff = 0.5} = input;
      if ((!Array.isArray(baseColor) && !ArrayBuffer.isView(baseColor)) || baseColor.length !== 4) fail('ANIMATION_SCENE_GEOMETRY', 'Expected RGBA material color');
      const material = {indices: indices === null ? null : copyMaterialArray(indices, 'indices'),
        baseColor: copyMaterialArray(baseColor, 'baseColor'),
        ...(input.side === undefined ? {doubleSided} : {side: input.side}), alphaMode, alphaCutoff};
      if (input.side !== undefined && input.doubleSided !== undefined)
        fail('ANIMATION_SCENE_GEOMETRY', 'Choose side or doubleSided, not both');
      for (const key of ['texCoords', 'vertexColors', 'uvTransform', 'emissiveFactor', 'specularColor']) {
        const value = input[key];
        if (value !== undefined) material[key] = value === null ? null : copyMaterialArray(value, key);
      }
      for (const key of ['shading', 'flatShading', 'shininess', 'metallicFactor', 'roughnessFactor', 'normalScale', 'occlusionStrength', 'depthTest', 'depthWrite', 'depthCompare', 'colorWrite', ...COAT_FIELDS]) {
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
    if (lodOptions) {
      const {createAnimationLod} = await import('./animation_lod.mjs');
      unchanged();
      lodState = createAnimationLod(pose, inputs.length, lodOptions);
    }
    if (shadowOptions) {
      const {prepareAnimationSceneShadows} = await import('./animation_scene_shadow.mjs');
      unchanged();
      sceneShadows = prepareAnimationSceneShadows(pose, inputs, shadowOptions);
      unchanged();
    }
    if (rigidGeometry) { rigidApi = await import('./animation_rigid_geometry.mjs'); unchanged(); }
    renderer = await createGpuAnimationRenderer(device, {...renderOptions, maxDraws: renderOptions.maxDraws ?? drawables.length,
      maxMeshes: renderOptions.maxMeshes ?? maxMeshes, maxBytes: Math.min(maxBytes, renderOptions.maxBytes ?? maxBytes)});
    unchanged();
    for (const {geometry, material} of inputs) {
      // Reserve the renderer's pending auxiliary buffers before allocating the
      // next deformer. uint32 indices conservatively bound padded uint16 data.
      const vertices = geometry?.positions?.length / 3;
      const surface = material.texCoords != null || material.vertexColors != null || TEXTURE_FIELDS.some(key => key !== 'gradientTexture' && material[key] != null);
      if (surface && (!Number.isSafeInteger(vertices) || vertices < 1)) fail('ANIMATION_SCENE_GEOMETRY', 'Surface attributes require XYZ geometry');
      const lit = ['lambert', 'metallic-roughness', 'phong', 'toon'].includes(material.shading);
      const surfaceStride = 24 + Object.keys(material.mapCoordinates ?? {}).length * 8;
      const coated = COAT_FIELDS.some(key => material[key] !== undefined) || TEXTURE_FIELDS.slice(5, 8).some(key => material[key] != null);
      const reserve = (coated ? 16 : 0) + (material.indices?.length ?? 0) * 4 + (surface ? vertices * surfaceStride : 0) + (lit && !lightingAllocated ? 544 + (renderOptions.shadows ? 96 : 0) + (renderOptions.environment ? 64 : 0) : 0);
      const remaining = maxBytes - renderer.allocatedBytes - deformationTotal() - reserve;
      let gpu;
      if (rigidApi?.canUseRigidAnimationGeometry(pose, geometry)) {
        if (remaining < 0) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exhausted');
        rigidPool ??= rigidApi.createGpuRigidGeometryPool(device, pose, {...deformOptions, maxBytes, maxMeshes});
        gpu = await rigidPool.addMesh(geometry, {maxAdditionalBytes: Math.min(remaining, deformOptions.maxBytes ?? 128 * 1024 * 1024)});
        rigidDeformers.push(gpu);
      } else {
        if (remaining < 1) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exhausted');
        gpu = await createGpuAnimationDeformer(device, pose, geometry, {...deformOptions,
          maxBytes: Math.min(remaining, deformOptions.maxBytes ?? 128 * 1024 * 1024)});
        computeDeformers.push(gpu); deformationBytes += gpu.bufferBytes;
      }
      deformers.push(gpu); unchanged();
      meshes.push(await renderer.addMesh(gpu, material)); lightingAllocated ||= lit; unchanged();
      ordering.push({mesh: meshes.at(-1), deformer: gpu, alphaMode: material.alphaMode, ...(frustumCulling ? {geometry} : {})});
      if (renderer.allocatedBytes + deformationTotal() > maxBytes) fail('ANIMATION_SCENE_LIMIT', 'Scene GPU buffer budget exceeded');
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
      let batchStats = null;
      if (deformationBatch) {
        // Rigid handles do not belong to the compute batch. They still publish
        // their current transforms before culling or any color/shadow draw.
        for (const gpu of rigidDeformers) gpu.update();
        batchStats = updateGpuAnimationDeformers(computeDeformers, {skipUnchanged: true});
      } else for (const gpu of deformers) gpu.update();
      if (frustumCulling) drawOrder.updateBounds();
      if (deformers.some(gpu => gpu.poseVersion !== nextVersion) || pose.version !== nextVersion) {
        fail('ANIMATION_SCENE_CHANGED', 'Pose changed during GPU upload');
      }
      poseVersion = nextVersion;
      if (batchStats) deformationStats = Object.freeze({...batchStats, poseVersion,
        rigidMeshes: rigidDeformers.length});
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
    get renderBundlesEnabled() { return renderer.renderBundles; },
    get renderBundleStats() { return renderer.bundleDiagnostics; },
    clearRenderBundles() { return exclusive(() => { renderer.clearRenderBundles(); return scene; }); },
    get deformationBatchEnabled() { return deformationBatch; },
    get deformationStats() { return deformationStats; },
    get deformationInputCacheBytes() { return disposed || terminal ? 0 : computeDeformers.reduce((n, gpu) => n + (gpu.inputCacheBytes ?? 0), 0); },
    get bufferBytes() { return renderer.allocatedBytes + deformationTotal(); },
    get rigidGeometryStats() { return Object.freeze({meshes: rigidPool?.meshCount ?? 0, uniqueGeometries: rigidPool?.uniqueGeometries ?? 0, bufferBytes: rigidPool?.bufferBytes ?? 0, computeMeshes: computeDeformers.length}); },
    get boundsBytes() { return drawOrder?.boundsBytes ?? 0; },
    get cullingStats() { return cullingStats; },
    get lodEnabled() { return lodState !== undefined; },
    get lodCameraCount() { return lodState?.cameraCount ?? 0; },
    get lodStats() { return lodStats; },
    resetLodCamera(key = 'default') { return exclusive(() => {
      if (!lodState) fail('ANIMATION_SCENE_LOD', 'This scene has no LOD selection');
      lodState.resetCamera(key);
      if (lodStats?.cameraKey === key) lodStats = null;
      return scene;
    }); },
    get shadowEnabled() { return sceneShadows !== undefined; },
    get shadowBytes() { return sceneShadows?.allocatedBytes ?? 0; },
    get shadowBoundsBytes() { return sceneShadows?.boundsBytes ?? 0; },
    get shadowStats() { return shadowStats; },
    get disposed() { return disposed; }, get failed() { return terminal !== null || renderer.failed || !!rigidPool?.failed || !!sceneShadows?.failed || deformers.some(gpu => gpu.failed); },
    update(delta, options) { return exclusive(() => { controller.update(delta, options); return upload(); }); },
    upload() { return exclusive(upload); },
    render(frame) { return exclusive(() => {
      synchronized();
      try {
        const prepared = {...frame}, implicit = prepared.draws == null;
        const selection = implicit && lodState ? lodState.prepare(prepared.lodCamera) : null;
        // This is scene metadata, not an extra renderer option or light field.
        delete prepared.lodCamera;
        prepared.draws ??= drawOrder ? drawOrder.order(prepared.viewProjection, selection?.drawIndices ?? null)
          : selection ? selection.drawIndices.map(index => meshes[index]) : meshes;
        // Use the exact camera snapshot tested by culling for shader packing too.
        if (frustumCulling && implicit) prepared.viewProjection = drawOrder.viewProjection;
        let automatic = null;
        if (sceneShadows && prepared.shadow === undefined) {
          if (!implicit) fail('ANIMATION_SCENE_SHADOW', 'Explicit draws require an explicit shadow map or shadow:null');
          synchronized();
          automatic = sceneShadows.render(prepared.lighting, selection?.drawIndices ?? null);
          prepared.lighting = automatic.lighting; prepared.shadow = automatic.shadow;
        }
        renderer.render(prepared);
        // No hysteresis publication on a rejected color frame, even if its
        // automatic depth pass already submitted. GPU work is not rolled back.
        if (selection) lodState.commit(selection);
        lodStats = selection;
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
      try { await Promise.all([renderer.whenIdle(), sceneShadows?.whenIdle(), rigidPool?.whenIdle(), ...computeDeformers.map(gpu => gpu.whenIdle())]); }
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
