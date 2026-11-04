/** Color-attachment admission shared by built-in and compiled-program draws.
 *
 * Float32 rendering is independent of optional float32 filtering and blending.
 * Keep the single-sample profile of gpu_render_target.mjs; never substitute a
 * lower-precision attachment or silently turn blending off. All checks are host
 * checks; these helpers allocate no GPU resources and enter no error scopes.
 */
const FLOAT32 = new Set(['r32float', 'rg32float', 'rgba32float']);
const INTEGER32 = new Set(['r32sint', 'rg32sint', 'rgba32sint', 'r32uint', 'rg32uint', 'rgba32uint']);
const FORMATS = new Set([null, 'rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm',
  'bgra8unorm-srgb', 'r8unorm', 'rg11b10ufloat', 'r16float', 'rg16float', 'rgba16float', ...FLOAT32, ...INTEGER32]);
const UNBLENDED = Object.freeze([false]);
const BOTH = Object.freeze([false, true]);

/** reject(message) must throw the owning renderer's public input error. */
export function createAnimationColorTargetProfile(device, {format, sampleCount}, reject) {
  if (!FORMATS.has(format) || ![1, 4].includes(sampleCount))
    reject('Unsupported color attachment configuration');
  if (format === 'rg11b10ufloat' && device.features?.has?.('rg11b10ufloat-renderable') !== true)
    reject('rg11b10ufloat color targets require rg11b10ufloat-renderable on the supplied device');
  const float32 = FLOAT32.has(format);
  const integer = INTEGER32.has(format);
  const componentType = integer ? format.endsWith('sint') ? 'i32' : 'u32' : 'f32';
  if (float32 && sampleCount !== 1)
    reject('Float32 color targets currently require one sample');
  if (integer && sampleCount !== 1)
    reject('Integer32 color targets require one sample');
  const blendable = format !== null && !integer && (!float32 || device.features?.has?.('float32-blendable') === true);
  function assertBlend(blend) {
    if (blend && !blendable)
      reject(format === null ? 'Color blending requires a color attachment'
        : integer ? `${format} color targets do not support blending`
        : `${format} blending requires float32-blendable on the supplied device`);
  }
  function assertOutput(type) {
    const scalar = /^(?:uint|uvec[234])$/.test(type) ? 'u32'
      : /^(?:int|ivec[234])$/.test(type) ? 'i32'
      : /^(?:float|vec[234])$/.test(type) ? 'f32' : null;
    if (format !== null && scalar !== componentType)
      reject(`${format} color targets require ${componentType} fragment outputs`);
    const components = format?.startsWith('rgba') || format?.startsWith('bgra') ? 4
      : format === 'rg11b10ufloat' ? 3 : format?.startsWith('rg') ? 2 : 1;
    const outputComponents = Number(/vec([234])$/.exec(type)?.[1] ?? 1);
    if (format !== null && outputComponents < components)
      reject(`${format} color targets require at least ${components} fragment output components`);
  }
  return Object.freeze({
    format, blendable, integer, componentType,
    assertBlend, assertOutput,
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
