/**
 * Scene-level alpha ordering, not geometry sorting or a Three.js RenderList.
 * Opaque/masked draws retain source order (including equal-depth surfaces).
 * Blended draws use descending projected world-space node-origin depth and
 * stable source-order ties. Recompute for every view and uploaded pose.
 * Intersecting transparent geometry still requires application-supplied draws.
 */
export class AnimationDrawOrderError extends Error {
  constructor(message) {
    super(`ANIMATION_SCENE_SORT: ${message}`);
    this.name = 'AnimationDrawOrderError';
    this.code = 'ANIMATION_SCENE_SORT';
  }
}
const fail = message => { throw new AnimationDrawOrderError(message); };
function matrix(value) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== 16) {
    fail('Expected a 16-component matrix');
  }
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('Matrices require fixed unshared storage');
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail('Detached matrix storage'); }
  }
  const result = new Float64Array(16);
  for (let i = 0; i < 16; i++) {
    const item = value[i];
    if (typeof item !== 'number' || !Number.isFinite(item)) fail('Matrix values must be finite numbers');
    result[i] = item;
  }
  return result;
}

/** Entries borrow mesh/deformer identity; alpha mode and insertion order freeze. */
export function createAnimationDrawOrder(entries) {
  if (!Array.isArray(entries) || entries.length > 4096) fail('Invalid scene draw count');
  const records = entries.map((entry, index) => {
    if (!entry?.mesh || !entry.deformer || !['OPAQUE', 'MASK', 'BLEND'].includes(entry.alphaMode)) {
      fail('Expected a mesh, deformer and alpha mode');
    }
    return {mesh: entry.mesh, deformer: entry.deformer, alphaMode: entry.alphaMode, index};
  });
  const opaque = records.filter(record => record.alphaMode !== 'BLEND').map(record => record.mesh);
  const blended = records.filter(record => record.alphaMode === 'BLEND');
  return Object.freeze({
    order(viewProjection) {
      if (!blended.length) return opaque.slice();
      const camera = matrix(viewProjection);
      const sorted = blended.map(record => {
        const world = matrix(record.deformer.worldMatrix);
        if (world[3] !== 0 || world[7] !== 0 || world[11] !== 0 || world[15] !== 1) fail('World matrix must be affine');
        const x = world[12], y = world[13], z = world[14];
        const clipZ = camera[2] * x + camera[6] * y + camera[10] * z + camera[14];
        const clipW = camera[3] * x + camera[7] * y + camera[11] * z + camera[15];
        if (!Number.isFinite(clipZ) || !Number.isFinite(clipW)) fail('Projected sort position overflowed');
        // A center on the eye plane does not make its geometry invalid. Keep
        // it drawable, with a deterministic signed-infinity key (0/0 -> 0).
        const depth = clipW === 0 ? Math.sign(clipZ) * Infinity : clipZ / clipW;
        return {...record, depth: Number.isNaN(depth) ? 0 : depth};
      });
      sorted.sort((a, b) => a.depth === b.depth ? a.index - b.index : a.depth > b.depth ? -1 : 1);
      return opaque.concat(sorted.map(record => record.mesh));
    },
  });
}
