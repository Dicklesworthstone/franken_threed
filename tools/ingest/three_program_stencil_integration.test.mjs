import test from 'node:test';
import assert from 'node:assert/strict';
import {inspectThreeProgram, threeProgramRaster, ThreeProgramError} from './three_program.mjs';
import {snapshotAnimationRaster} from './animation_raster.mjs';

const T = {NoBlending: 0, NormalBlending: 1, FrontSide: 0, BackSide: 1, DoubleSide: 2,
  NeverDepth: 0, AlwaysDepth: 1, LessDepth: 2, LessEqualDepth: 3, EqualDepth: 4, GreaterEqualDepth: 5, GreaterDepth: 6, NotEqualDepth: 7,
  NeverStencilFunc: 512, LessStencilFunc: 513, EqualStencilFunc: 514, LessEqualStencilFunc: 515,
  GreaterStencilFunc: 516, NotEqualStencilFunc: 517, GreaterEqualStencilFunc: 518, AlwaysStencilFunc: 519,
  ZeroStencilOp: 0, KeepStencilOp: 7680, ReplaceStencilOp: 7681, IncrementStencilOp: 7682,
  DecrementStencilOp: 7683, IncrementWrapStencilOp: 34055, DecrementWrapStencilOp: 34056, InvertStencilOp: 5386};
function material(patch = {}) {
  return {type: 'ShaderMaterial', isShaderMaterial: true, vertexShader: 'void main(){}', fragmentShader: 'void main(){}',
    blending: T.NoBlending, transparent: false, colorWrite: true, depthTest: true, depthWrite: false, depthFunc: T.LessEqualDepth, side: T.FrontSide,
    stencilWrite: true, stencilRef: 37, stencilFunc: T.EqualStencilFunc, stencilFuncMask: 0x7f, stencilWriteMask: 0,
    stencilFail: T.KeepStencilOp, stencilZFail: T.KeepStencilOp, stencilZPass: T.ReplaceStencilOp, ...patch};
}
for (const patch of [{}, {type: 'RawShaderMaterial', isRawShaderMaterial: true}, {type: 'MeshBasicMaterial', isShaderMaterial: false}]) {
  test(`${patch.type ?? 'ShaderMaterial'} stencil admission reaches native raster mapping`, () => {
    const m = material(patch); inspectThreeProgram(T, m, {});
    const source = threeProgramRaster(T, m);
    const native = snapshotAnimationRaster({stencil: source.stencil, stencilReference: source.stencilReference},
      {format: 'rgba8unorm', depthFormat: 'depth24plus-stencil8'});
    assert.equal(native.stencil.stencilFront.compare, 'equal'); assert.equal(native.stencil.stencilWriteMask, 0);
    assert.equal(native.stencilReference, 37); assert.equal(source.writeMask, 15);
  });
}
test('back sides, reflected worlds, and non-triangle topologies keep source stencil state', () => {
  const m = material(), expected = threeProgramRaster(T, m).stencil;
  for (const side of [T.FrontSide, T.BackSide, T.DoubleSide]) for (const frontFaceCW of [false, true])
    for (const topology of ['triangles', 'lines', 'points'])
      assert.deepEqual(threeProgramRaster(T, m, {side, frontFaceCW, topology}).stencil, expected);
});
test('disabled stencil preserves the existing raster shape even with invalid dormant fields', () => {
  const r = threeProgramRaster(T, material({stencilWrite: false, stencilFunc: -999}));
  assert.equal(Object.hasOwn(r, 'stencil'), false); assert.equal(Object.hasOwn(r, 'stencilReference'), false);
  assert.equal(r.blend, null); assert.equal(r.depthCompare, 'less-equal'); assert.equal(r.depthWriteEnabled, false);
});
test('invalid stencil state retains the public ThreeProgramError type', () => {
  assert.throws(() => threeProgramRaster(T, material({stencilZPass: -999})), error =>
    error instanceof ThreeProgramError && error.code === 'THREE_PROGRAM_MATERIAL');
});
