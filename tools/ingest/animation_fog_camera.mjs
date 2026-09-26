/** Camera-space fog descriptors for the explicit renderer. No Three.js import,
 * camera mutation, world-matrix update, GPU work or clock is performed here.
 * Input projection matrices are column-major. The result consumes the native
 * WebGPU homogeneous clip position BEFORE division, not fragment depth or W.
 */
import {AnimationFogError, packAnimationFog} from './animation_fog.mjs';
const fail = message => { throw new AnimationFogError('CAMERA', message); };
const finite = (x, label) => {
  if (typeof x !== 'number' || !Number.isFinite(x)) fail(`${label} must be finite`);
  return x;
};

/** Return the row mapping native clip coordinates to -viewPosition.z.
 * clipSpace names the INPUT projection convention. A WebGL projection receives
 * exactly the z'=(z+w)/2 conversion used before WebGPU rasterization. The source
 * matrix is never rewritten. Perspective, orthographic, off-axis, reverse-Z and
 * infinite-far projections use the same solve; no standard-camera coefficients
 * or depth-buffer reconstruction are guessed.
 */
export function animationFogDepthFromProjection(projection, {clipSpace = 'webgpu'} = {}) {
  if (clipSpace !== 'webgpu' && clipSpace !== 'webgl') fail('Unknown projection clip space');
  if ((!Array.isArray(projection) && !ArrayBuffer.isView(projection)) || projection.length !== 16)
    fail('Expected a column-major projection matrix with 16 elements');
  // Indexed reads have a fixed bound, including arrays with hostile iterators.
  const p = Array.from({length: 16}, (_, i) => finite(projection[i], 'Projection element'));
  if (clipSpace === 'webgl')
    for (let c = 0; c < 4; c++) p[c * 4 + 2] = p[c * 4 + 2] * 0.5 + p[c * 4 + 3] * 0.5;
  // Solve transpose(Pnative) * row = [0,0,-1,0]. Row scaling bounds
  // elimination and avoids rejecting merely small, but invertible, projections.
  const a = Array.from({length: 4}, (_, r) => {
    const row = p.slice(r * 4, r * 4 + 4);
    const scale = Math.max(...row.map(Math.abs));
    if (!scale) fail('Projection is singular');
    return [...row.map(x => x / scale), (r === 2 ? -1 : 0) / scale];
  });
  for (let c = 0; c < 4; c++) {
    let pivot = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    if (a[pivot][c] === 0) fail('Projection is singular');
    [a[c], a[pivot]] = [a[pivot], a[c]];
    const divisor = a[c][c];
    for (let k = c; k < 5; k++) a[c][k] /= divisor;
    for (let r = 0; r < 4; r++) if (r !== c) {
      const factor = a[r][c];
      if (factor === 0) continue;
      for (let k = c; k < 5; k++) a[r][k] -= factor * a[c][k];
    }
  }
  const row = a.map(r => r[4]);
  if (row.some(x => !Number.isFinite(Math.fround(x)))) fail('Clip-depth row does not fit finite f32');
  return Object.freeze(row);
}

/** Snapshot a live linear/exp2 fog definition and a camera projection for one
 * renderer.render({..., fog: descriptor}) call. No pipeline recompilation or
 * per-mesh registration is needed for scalar/color/camera changes. Null resets
 * the existing frame uniform. Color is already in the renderer's linear space.
 */
export function snapshotAnimationCameraFog(fog, projection, options) {
  if (fog === null) return null;
  if (!fog || typeof fog !== 'object' || Array.isArray(fog)) fail('Expected fog or null');
  const type = fog.type;
  const fields = type === 'linear' ? ['type', 'color', 'near', 'far'] :
    type === 'exp2' ? ['type', 'color', 'density'] : null;
  if (!fields || Object.keys(fog).some(key => !fields.includes(key))) fail('Invalid camera fog profile');
  const sourceColor = fog.color;
  if ((!Array.isArray(sourceColor) && !ArrayBuffer.isView(sourceColor)) || sourceColor.length !== 3)
    fail('Expected three linear fog color components');
  const descriptor = {
    type,
    color: Object.freeze(Array.from({length: 3}, (_, i) => sourceColor[i])),
    depthFromClip: animationFogDepthFromProjection(projection, options),
    ...(type === 'linear' ? {near: fog.near, far: fog.far} : {density: fog.density}),
  };
  // Use the renderer's exact admission/quantization contract before publication.
  packAnimationFog(descriptor);
  return Object.freeze(descriptor);
}
