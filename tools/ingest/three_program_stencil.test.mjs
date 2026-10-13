import test from 'node:test';
import assert from 'node:assert/strict';
import {threeProgramStencil} from './three_program_stencil.mjs';
import {snapshotAnimationRaster} from './animation_raster.mjs';

// r186 constants use the corresponding WebGL enum values.
const T = {NeverStencilFunc: 512, LessStencilFunc: 513, EqualStencilFunc: 514, LessEqualStencilFunc: 515,
  GreaterStencilFunc: 516, NotEqualStencilFunc: 517, GreaterEqualStencilFunc: 518, AlwaysStencilFunc: 519,
  ZeroStencilOp: 0, KeepStencilOp: 7680, ReplaceStencilOp: 7681, IncrementStencilOp: 7682,
  DecrementStencilOp: 7683, IncrementWrapStencilOp: 34055, DecrementWrapStencilOp: 34056, InvertStencilOp: 5386};
const defaults = {stencilWrite: true, stencilRef: 1, stencilFunc: T.AlwaysStencilFunc, stencilFuncMask: 255,
  stencilWriteMask: 255, stencilFail: T.KeepStencilOp, stencilZFail: T.KeepStencilOp, stencilZPass: T.ReplaceStencilOp};
const fail = (code, message) => { throw Object.assign(Error(message), {code: 'THREE_PROGRAM_' + code}); };
const stencil = overrides => threeProgramStencil(T, {...defaults, ...overrides}, fail);
for (const [field, expected] of [['Never', 'never'], ['Less', 'less'], ['Equal', 'equal'], ['LessEqual', 'less-equal'],
  ['Greater', 'greater'], ['NotEqual', 'not-equal'], ['GreaterEqual', 'greater-equal'], ['Always', 'always']]) {
  test(`source ${field}StencilFunc maps both faces without reversing comparison order`, () => {
    const result = stencil({stencilFunc: T[field + 'StencilFunc']});
    assert.equal(result.stencil.front.compare, expected); assert.deepEqual(result.stencil.front, result.stencil.back);
  });
}
for (const [name, expected] of [['Zero', 'zero'], ['Keep', 'keep'], ['Replace', 'replace'], ['Increment', 'increment-clamp'],
  ['Decrement', 'decrement-clamp'], ['IncrementWrap', 'increment-wrap'], ['DecrementWrap', 'decrement-wrap'], ['Invert', 'invert']]) {
  test(`source ${name}StencilOp retains fail/depth-fail/pass semantics`, () => {
    const result = stencil({stencilFail: T[name + 'StencilOp'], stencilZFail: T[name + 'StencilOp'], stencilZPass: T[name + 'StencilOp']});
    for (const face of [result.stencil.front, result.stencil.back]) {
      assert.equal(face.failOp, expected); assert.equal(face.depthFailOp, expected); assert.equal(face.passOp, expected);
    }
  });
}
for (const [reference, expected] of [[0, 0], [255, 255], [256, 255], [-1, 0], [7.9, 7], [-7.9, 0],
  [0x80000000, 0], [0xffffffff, 0], [0x100000007, 7], [NaN, 0], [Infinity, 0], ['23', 23]]) {
  test(`reference ${String(reference)} is signed-long converted then clamped, not wrapped`, () => {
    assert.equal(stencil({stencilRef: reference}).stencilReference, expected);
  });
}
test('read and write masks undergo unsigned-long conversion independently', () => {
  for (const [value, expected] of [[-1, 0xffffffff], [0x100000035, 0x35], [8.9, 8], [0, 0], [NaN, 0]]) {
    const result = stencil({stencilFuncMask: value, stencilWriteMask: value + 1});
    assert.equal(result.stencil.readMask, expected);
    assert.equal(result.stencil.writeMask, (value + 1) >>> 0);
  }
});
test('inactive stencil never reads or validates unrelated stencil fields', () => {
  const material = {stencilWrite: false};
  for (const field of Object.keys(defaults).filter(k => k !== 'stencilWrite'))
    Object.defineProperty(material, field, {get() { throw Error('inactive field read'); }});
  assert.deepEqual(threeProgramStencil(T, material, fail), {});
});
test('zero write mask leaves comparison active and input material unchanged', () => {
  const material = Object.freeze({...defaults, stencilWriteMask: 0, stencilFunc: T.NotEqualStencilFunc});
  const result = threeProgramStencil(T, material, fail);
  assert.equal(result.stencil.writeMask, 0); assert.equal(result.stencil.front.compare, 'not-equal');
  assert.ok(Object.isFrozen(result.stencil)); assert.ok(Object.isFrozen(result.stencil.front));
  const native = snapshotAnimationRaster(result, {format: 'rgba8unorm', depthFormat: 'depth24plus-stencil8'});
  assert.equal(native.stencil.stencilWriteMask, 0); assert.equal(native.stencilReference, 1);
});
test('invalid active source enum produces a source-material error', () => {
  for (const field of ['stencilFunc', 'stencilFail', 'stencilZFail', 'stencilZPass'])
    assert.throws(() => stencil({[field]: 0x123456}), {code: 'THREE_PROGRAM_MATERIAL'});
});
test('references and operations are fresh snapshots for later material uses', () => {
  const material = {...defaults}; const first = threeProgramStencil(T, material, fail);
  material.stencilRef = 19; material.stencilZPass = T.InvertStencilOp;
  const second = threeProgramStencil(T, material, fail);
  assert.equal(first.stencilReference, 1); assert.equal(first.stencil.front.passOp, 'replace');
  assert.equal(second.stencilReference, 19); assert.equal(second.stencil.front.passOp, 'invert');
  assert.notEqual(JSON.stringify(first), JSON.stringify(second));
});
