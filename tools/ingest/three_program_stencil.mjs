/** r186 WebGLState.setMaterial stencil state for compiled source programs.
 * Three uses stencilFunc/stencilOp, not their Separate variants, so both faces
 * get the same state even when side or world winding changes. GL references
 * undergo signed-long conversion and clamp to the native 8-bit stencil range;
 * masks undergo unsigned-long conversion. Do not wrap a reference to 8 bits.
 * The program owner omits this state on targets without a stencil attachment,
 * matching GL's disabled-test behavior there. No source mutation or GPU work.
 */
/** WebIDL [EnforceRange] unsigned long, used by the source WebGPU backend.
 * GL callers deliberately keep their signed-reference clamping instead. */
export function threeWebGPUStencilValue(value, fail) {
  const number = Math.trunc(+value);
  if (!Number.isFinite(number) || number < 0 || number > 0xffffffff)
    fail('MATERIAL', 'Stencil values must convert to unsigned 32-bit integers');
  return number;
}
export function threeProgramStencil(T, material, fail, {webgpu = false} = {}) {
  if (!material.stencilWrite) return {};
  const compares = new Map([[T.NeverStencilFunc, 'never'], [T.LessStencilFunc, 'less'],
    [T.EqualStencilFunc, 'equal'], [T.LessEqualStencilFunc, 'less-equal'],
    [T.GreaterStencilFunc, 'greater'], [T.NotEqualStencilFunc, 'not-equal'],
    [T.GreaterEqualStencilFunc, 'greater-equal'], [T.AlwaysStencilFunc, 'always']]);
  const operations = new Map([[T.ZeroStencilOp, 'zero'], [T.KeepStencilOp, 'keep'],
    [T.ReplaceStencilOp, 'replace'], [T.IncrementStencilOp, 'increment-clamp'],
    [T.DecrementStencilOp, 'decrement-clamp'], [T.IncrementWrapStencilOp, 'increment-wrap'],
    [T.DecrementWrapStencilOp, 'decrement-wrap'], [T.InvertStencilOp, 'invert']]);
  const select = (table, value, field) => {
    const result = table.get(value >>> 0);
    if (result === undefined) fail('MATERIAL', `Unsupported ${field}`);
    return result;
  };
  const face = Object.freeze({compare: select(compares, material.stencilFunc, 'stencil function'),
    failOp: select(operations, material.stencilFail, 'stencil fail operation'),
    depthFailOp: select(operations, material.stencilZFail, 'stencil depth-fail operation'),
    passOp: select(operations, material.stencilZPass, 'stencil pass operation')});
  return {
    stencil: Object.freeze({front: face, back: face,
      readMask: webgpu ? threeWebGPUStencilValue(material.stencilFuncMask, fail) : material.stencilFuncMask >>> 0,
      writeMask: webgpu ? threeWebGPUStencilValue(material.stencilWriteMask, fail) : material.stencilWriteMask >>> 0}),
    stencilReference: webgpu ? threeWebGPUStencilValue(material.stencilRef, fail) : Math.min(255, Math.max(0, material.stencilRef | 0)),
  };
}
