/** Color-attachment admission shared by built-in and compiled-program draws.
 *
 * Float32 rendering is independent of optional float32 filtering and blending.
 * Keep the single-sample profile of gpu_render_target.mjs; never substitute a
 * lower-precision attachment or silently turn blending off. All checks are host
 * checks; these helpers allocate no GPU resources and enter no error scopes.
 */
const FLOAT32 = new Set(['r32float', 'rg32float', 'rgba32float']);
const FORMATS = new Set([null, 'rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm',
  'bgra8unorm-srgb', 'rg16float', 'rgba16float', ...FLOAT32]);
const UNBLENDED = Object.freeze([false]);
const BOTH = Object.freeze([false, true]);

/** reject(message) must throw the owning renderer's public input error. */
export function createAnimationColorTargetProfile(device, {format, sampleCount}, reject) {
  if (!FORMATS.has(format) || ![1, 4].includes(sampleCount))
    reject('Unsupported color attachment configuration');
  const float32 = FLOAT32.has(format);
  if (float32 && sampleCount !== 1)
    reject('Float32 color targets currently require one sample');
  const blendable = format !== null && (!float32 || device.features?.has?.('float32-blendable') === true);
  function assertBlend(blend) {
    if (blend && !blendable)
      reject(format === null ? 'Color blending requires a color attachment'
        : `${format} blending requires float32-blendable on the supplied device`);
  }
  return Object.freeze({
    format, blendable,
    assertBlend,
    /** These booleans select alpha-mode/depth-write pipeline keys, not merely
     * whether the native descriptor includes blending. Explicit blend:null
     * still needs BOTH keys: an alphaMode:BLEND draw preserves its alpha and
     * default disabled depth writes without enabling fixed-function blending.
     */
    blendModes(raster) {
      assertBlend(raster?.blend);
      if (format === null) return UNBLENDED;
      return blendable || raster?.blend === null ? BOTH : UNBLENDED;
    },
  });
}
