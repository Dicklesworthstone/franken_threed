/** Fixed-function material state for the direct renderer. Pipeline descriptors
 * and live pass state are deliberately separate: WebGPU render bundles cannot
 * record blend constants or stencil references. No source or GPU side effects.
 */
export class AnimationRasterError extends Error {
  constructor(message) { super(`ANIMATION_RASTER_INPUT: ${message}`); this.name = 'AnimationRasterError'; this.code = 'ANIMATION_RASTER_INPUT'; }
}
const fail = message => { throw new AnimationRasterError(message); };
const FACTORS = ['zero','one','src','one-minus-src','src-alpha','one-minus-src-alpha',
  'dst','one-minus-dst','dst-alpha','one-minus-dst-alpha','src-alpha-saturated','constant','one-minus-constant'];
const OPERATIONS = ['add','subtract','reverse-subtract','min','max'];
const COMPARES = ['never','less','equal','less-equal','greater','not-equal','greater-equal','always'];
const STENCIL_OPS = ['keep','zero','replace','invert','increment-clamp','decrement-clamp','increment-wrap','decrement-wrap'];
function fields(value, allowed, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`Expected ${label} descriptor`);
  for (const [key, d] of Object.entries(Object.getOwnPropertyDescriptors(value)))
    if (!allowed.includes(key) || !Object.hasOwn(d, 'value')) fail(`Unsupported or accessor-backed ${label} field: ${key}`);
  return value;
}
function choice(value, allowed, label) { if (!allowed.includes(value)) fail(`Invalid ${label}`); return value; }
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail(`Invalid ${label}`); return value;
}
function f32(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value))) fail(`Invalid ${label}`); return value;
}
function component(input = {}) {
  fields(input, ['operation','srcFactor','dstFactor'], 'blend component');
  const operation = choice(input.operation ?? 'add', OPERATIONS, 'blend operation');
  const srcFactor = choice(input.srcFactor ?? 'one', FACTORS, 'source blend factor');
  const dstFactor = choice(input.dstFactor ?? 'zero', FACTORS, 'destination blend factor');
  // Min/max ignore factors in GL. WebGPU requires both factors to be one.
  return Object.freeze({operation, srcFactor: ['min','max'].includes(operation) ? 'one' : srcFactor,
    dstFactor: ['min','max'].includes(operation) ? 'one' : dstFactor});
}
function face(input = {}) {
  fields(input, ['compare','failOp','depthFailOp','passOp'], 'stencil face');
  return Object.freeze({compare: choice(input.compare ?? 'always', COMPARES, 'stencil comparison'),
    failOp: choice(input.failOp ?? 'keep', STENCIL_OPS, 'stencil fail operation'),
    depthFailOp: choice(input.depthFailOp ?? 'keep', STENCIL_OPS, 'stencil depth-fail operation'),
    passOp: choice(input.passOp ?? 'keep', STENCIL_OPS, 'stencil pass operation')});
}
export const hasAnimationStencil = format => ['depth24plus-stencil8','depth32float-stencil8'].includes(format);
export function snapshotAnimationBlendConstant(input) {
  if (!(Array.isArray(input) || input instanceof Float32Array || input instanceof Float64Array) || input.length !== 4)
    fail('Blend constant must have four components');
  return Object.freeze(Array.from(input, value => {
    f32(value, 'blend constant'); if (value < 0 || value > 1) fail('Blend constant must be in [0,1]'); return value;
  }));
}
export function snapshotAnimationRaster(options, {format, depthFormat}) {
  let blend;
  if (options.blend === null) blend = null;
  else if (options.blend !== undefined) {
    fields(options.blend, ['color','alpha'], 'blend');
    blend = Object.freeze({color: component(options.blend.color), alpha: component(options.blend.alpha)});
  }
  const premultipliedAlpha = options.premultipliedAlpha ?? false;
  if (typeof premultipliedAlpha !== 'boolean' || options.premultipliedAlpha === null) fail('premultipliedAlpha must be boolean');
  const usesConstant = !!blend && [blend.color, blend.alpha].some(c => [c.srcFactor,c.dstFactor].some(f => f.includes('constant')));
  if (options.blendConstant !== undefined && !usesConstant) fail('Blend constants require an active constant factor');
  const blendConstant = usesConstant ? snapshotAnimationBlendConstant(options.blendConstant ?? [0,0,0,0]) : null;
  let stencil = null;
  if (options.stencil != null) {
    if (!hasAnimationStencil(depthFormat)) fail('Stencil requires a depth-stencil attachment format');
    fields(options.stencil, ['front','back','readMask','writeMask'], 'stencil');
    stencil = Object.freeze({stencilFront: face(options.stencil.front), stencilBack: face(options.stencil.back),
      stencilReadMask: integer(options.stencil.readMask ?? 0xff, 0, 0xffffffff, 'stencil read mask'),
      stencilWriteMask: integer(options.stencil.writeMask ?? 0xff, 0, 0xffffffff, 'stencil write mask')});
  }
  if (options.stencilReference !== undefined && !stencil) fail('Stencil references require stencil state');
  const stencilReference = stencil ? integer(options.stencilReference ?? 0, 0, 0xff, 'stencil reference') : null;
  const depthBias = integer(options.depthBias ?? 0, -0x80000000, 0x7fffffff, 'depth bias');
  const depthBiasSlopeScale = f32(options.depthBiasSlopeScale ?? 0, 'depth slope bias');
  const depthBiasClamp = f32(options.depthBiasClamp ?? 0, 'depth bias clamp');
  if (!depthFormat && (depthBias || depthBiasSlopeScale || depthBiasClamp)) fail('Depth bias requires a depth attachment');
  if (format === null && (blend != null || premultipliedAlpha)) fail('Color blending and premultiplication require a color attachment');
  const bias = Object.freeze({depthBias, depthBiasSlopeScale, depthBiasClamp});
  const ordinary = blend === undefined && !premultipliedAlpha && !stencil && !depthBias && !depthBiasSlopeScale && !depthBiasClamp;
  // The identity excludes live values. Constants/references never grow pipelines.
  const key = ordinary ? '' : JSON.stringify([blend === undefined ? 'legacy' : blend, premultipliedAlpha, stencil, bias]);
  return Object.freeze({key, blend, premultipliedAlpha, stencil, bias, usesConstant, blendConstant, stencilReference});
}
export function snapshotAnimationRasterUse(raster, input) {
  if (input.blendConstant !== undefined && !raster.usesConstant) fail('Draw blend constant requires constant-factor blending');
  if (input.stencilReference !== undefined && !raster.stencil) fail('Draw stencil reference requires stencil state');
  return {blendConstant: raster.usesConstant ? snapshotAnimationBlendConstant(input.blendConstant ?? raster.blendConstant) : null,
    stencilReference: raster.stencil ? integer(input.stencilReference ?? raster.stencilReference, 0, 0xff, 'stencil reference') : null};
}
export const ANIMATION_NORMAL_BLEND = Object.freeze({
  color: Object.freeze({operation:'add', srcFactor:'src-alpha', dstFactor:'one-minus-src-alpha'}),
  alpha: Object.freeze({operation:'add', srcFactor:'one', dstFactor:'one-minus-src-alpha'}),
});
