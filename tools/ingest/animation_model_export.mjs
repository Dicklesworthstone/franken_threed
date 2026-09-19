/** Opt-in posed export for model factories. CPU models reuse current deformer
 * outputs. GPU models retain source data only when enabled, and evaluate the
 * existing CPU reference deformer only on export calls. No GPU readback and no
 * work is added to animation update/render. Not GPU-f32/pixel equivalence.
 */
import {createAnimationDeformer} from './animation_deformer.mjs';
import {exportAnimationPoseGLB, AnimationExportError} from './animation_pose_export.mjs';
export {AnimationExportError} from './animation_pose_export.mjs';
const fail = (code, text) => { throw new AnimationExportError('ANIMATION_EXPORT_' + code, text); };
const MAPS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture', 'occlusionTexture',
  'clearcoatTexture', 'clearcoatRoughnessTexture', 'clearcoatNormalTexture'];

export function createAnimationModelExporter(pose, drawables, source, exporting = false, deformers = null, copyright) {
  if (exporting === false) return Object.freeze({enabled: false, exportPoseGLB() { fail('DISABLED', 'Enable exporting at model construction'); }, dispose() {}});
  const config = exporting === true ? {} : exporting;
  if (!config || typeof config !== 'object' || Array.isArray(config)) fail('OPTION', 'exporting must be false, true or a limits object');
  for (const key of Object.keys(config)) if (!['maxComponents', 'maxBytes', 'maxVertices', 'copyright'].includes(key)) fail('OPTION', `Unknown exporting option: ${key}`);
  const {maxComponents = 16777216, ...limits} = config;
  if (limits.copyright === undefined && copyright !== undefined) limits.copyright = copyright;
  if (!Number.isSafeInteger(maxComponents) || maxComponents < 1) fail('LIMIT', 'Invalid retained component limit');
  for (const [key, min, max] of [['maxBytes',64,0xffffffff],['maxVertices',1,0xfffffffe]]) if (limits[key] !== undefined &&
    (!Number.isSafeInteger(limits[key]) || limits[key] < min || limits[key] > max)) fail('LIMIT', `Invalid ${key}`);
  if (limits.copyright !== undefined && typeof limits.copyright !== 'string') fail('OPTION', 'copyright must be text');
  if (!pose || pose.disposed || !Array.isArray(drawables) || !drawables.length || drawables.length > 4096 ||
    !Array.isArray(source) || source.length !== drawables.length) fail('SHAPE', 'Expected a decoded model');
  if (deformers !== null && (!Array.isArray(deformers) || deformers.length !== drawables.length)) fail('SHAPE', 'CPU deformers must match the model');
  const initialVersion = pose.version;
  let disposed = false, busy = false, owned = [], records, copied = 0;
  function copy(value, label) {
    if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value instanceof DataView || !Number.isSafeInteger(value.length)) fail('SHAPE', `Invalid ${label}`);
    copied += value.length; if (copied > maxComponents) fail('LIMIT', 'Export source component budget exceeded');
    if (ArrayBuffer.isView(value)) {
      if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('STORAGE', 'Export sources require fixed unshared arrays');
      try { new Uint8Array(value.buffer,0,0); } catch { fail('STORAGE', 'Detached export source'); }
    }
    return Float64Array.from(value, n => { if (typeof n !== 'number' || !Number.isFinite(n)) fail('VALUE', `Invalid ${label}`); return n; });
  }
  if (deformers) records = drawables.map((drawable, i) => ({drawable, source: {...source[i]}, deformer: deformers[i]}));
  else records = drawables.map((input, i) => {
    const g = input.geometry;
    if (!g || !Number.isSafeInteger(g.node)) fail('SHAPE', 'Expected decoded geometry');
    const geometry = {...g};
    for (const key of ['positions','normals','tangents','joints','weights']) if (g[key] !== undefined) geometry[key] = copy(g[key], key);
    if (!Array.isArray(g.morphTargets ?? []) || (g.morphTargets?.length ?? 0) > 4096) fail('LIMIT', 'Invalid morph target count');
    geometry.morphTargets = (g.morphTargets ?? []).map(target => {
      if (!target || typeof target !== 'object' || Array.isArray(target)) fail('SHAPE', 'Invalid morph target');
      return Object.fromEntries(Object.entries(target).map(([key, value]) => [key, copy(value, 'morph ' + key)]));
    });
    const drawable = {...input, geometry};
    for (const key of ['indices','baseColor','emissiveFactor','texCoords','vertexColors','uvTransform']) if (input[key] != null) drawable[key] = copy(input[key], key);
    for (const key of MAPS) if (input[key] != null) drawable[key] = {...input[key]};
    if (input.mapCoordinates != null) {
      if (typeof input.mapCoordinates !== 'object' || Array.isArray(input.mapCoordinates)) fail('SHAPE', 'Invalid per-map coordinates');
      drawable.mapCoordinates = {};
      for (const [field, coordinate] of Object.entries(input.mapCoordinates)) {
        if (!MAPS.includes(field) || !coordinate || typeof coordinate !== 'object' || Array.isArray(coordinate)) fail('SHAPE', 'Invalid map coordinate');
        const c = {...coordinate};
        for (const key of ['texCoords','uvTransform']) if (coordinate[key] !== undefined) c[key] = copy(coordinate[key], field + ' ' + key);
        drawable.mapCoordinates[field] = c;
      }
    }
    return {drawable, source: {...source[i]}, deformer: null};
  });
  if (pose.disposed || pose.version !== initialVersion) fail('STALE', 'Pose changed during export setup');
  function live() { if (disposed || pose.disposed) fail('DISPOSED', 'Model exporter has been disposed'); }
  function initialize() {
    if (deformers || owned.length) return;
    const created = [];
    try { for (const record of records) created.push(createAnimationDeformer(pose, record.drawable.geometry, {maxComponents})); }
    catch (error) { for (const d of created) d.dispose(); throw error; }
    owned = created;
    for (let i = 0; i < records.length; i++) {
      records[i].deformer = owned[i];
      // Deformers own their own source copies now. The writer never reads raw
      // geometry, only their current published outputs and material metadata.
      records[i].drawable.geometry = null;
    }
  }
  return Object.freeze({enabled: true,
    exportPoseGLB(options = {}) {
      live(); if (busy) fail('REENTRANT', 'Export capture cannot be reentered'); busy = true;
      try {
        if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTION', 'Invalid export options');
        const settings = {...limits, ...options};
        if (limits.maxBytes !== undefined) settings.maxBytes = Math.min(limits.maxBytes, settings.maxBytes);
        if (limits.maxVertices !== undefined) settings.maxVertices = Math.min(limits.maxVertices, settings.maxVertices);
        if (settings.signal?.aborted) throw settings.signal.reason ?? new DOMException('Aborted','AbortError');
        initialize();
        for (const d of owned) if (d.poseVersion !== pose.version) d.update();
        // The async writer captures all geometry before its first await. Release
        // the model lock then: awaiting images must not stop live animation.
        return exportAnimationPoseGLB(pose, records, settings);
      } finally { busy = false; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose while capturing export');
      if (!disposed) { disposed = true; for (const d of owned) d.dispose(); owned = []; records = null; }
    },
  });
}
