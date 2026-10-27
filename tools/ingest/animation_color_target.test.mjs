import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationColorTargetProfile} from './animation_color_target.mjs';
import {createProgramMeshes} from './animation_program_mesh.mjs';

const FLOAT32 = ['r32float', 'rg32float', 'rgba32float'];
const LEGACY = ['rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb', 'r16float', 'rg16float', 'rgba16float'];
const BLEND = Object.freeze({color: {srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha'},
  alpha: {srcFactor: 'one', dstFactor: 'one-minus-src-alpha'}});
function reject(message) { throw Object.assign(new Error(message), {code: 'ANIMATION_RENDER_OPTIONS'}); }
function profile(format, features = [], sampleCount = 1) {
  return createAnimationColorTargetProfile({features: new Set(features)}, {format, sampleCount}, reject);
}
for (const format of FLOAT32) {
  test(`${format}: unblended draws need neither optional float32 feature`, () => {
    const p = profile(format);
    assert.equal(p.format, format);
    assert.equal(p.blendable, false);
    assert.deepEqual(p.blendModes(), [false]);
    assert.deepEqual(p.blendModes({blend: undefined, premultipliedAlpha: true}), [false]);
    for (const mode of ['OPAQUE', 'MASK']) p.assertBlend(mode === 'BLEND');
    assert.throws(() => p.assertBlend(true), /float32-blendable/);
    assert.throws(() => p.blendModes({blend: BLEND}), /float32-blendable/);
  });
  test(`${format}: explicit no-blend retains the transparent alpha/depth key`, () => {
    const p = profile(format);
    p.assertBlend(null);
    assert.deepEqual(p.blendModes({blend: null}), [false, true]);
  });
  test(`${format}: filtering alone does not authorize blending`, () => {
    const p = profile(format, ['float32-filterable']);
    assert.equal(p.blendable, false);
    assert.throws(() => p.assertBlend(BLEND), /float32-blendable/);
    assert.deepEqual(p.blendModes(), [false]);
  });
  test(`${format}: enabled blending admits legacy and custom blend states`, () => {
    const p = profile(format, ['float32-blendable']);
    assert.equal(p.blendable, true);
    p.assertBlend(true); p.assertBlend(BLEND);
    assert.deepEqual(p.blendModes(), [false, true]);
    assert.deepEqual(p.blendModes({blend: BLEND}), [false, true]);
  });
  test(`${format}: MSAA is refused without a precision fallback`, () => {
    assert.throws(() => profile(format, ['float32-blendable', 'float32-filterable'], 4), /one sample/);
  });
  test(`${format}: compiled-program blend refusal precedes all GPU work`, async () => {
    let calls = 0;
    const device = {features: new Set(), limits: {maxUniformBufferBindingSize: 65536},
      createBuffer() { calls++; throw Error('unexpected allocation'); }};
    const owner = createProgramMeshes({device, format, depthFormat: 'depth24plus', sampleCount: 1,
      maxDraws: 4, label: 'float32-admission', fail: (_code, message) => reject(message),
      scoped() { calls++; throw Error('unexpected GPU scope'); }, lost: new Promise(() => {})});
    const program = {vertex: '', fragment: '', reflection: {textures: [], outputs: [{location: 0}], uniformBufferSize: 16}};
    // An invalid geometry handle is deliberate: blend validation must happen
    // before geometry inspection, arena allocation or pipeline preparation.
    await assert.rejects(owner.add(null, {program, raster: {blend: BLEND, writeMask: 0}}), /float32-blendable/);
    assert.equal(calls, 0);
    owner.dispose();
  });
}
for (const format of LEGACY) for (const sampleCount of [1, 4]) {
  test(`${format}/${sampleCount}: existing blend pipelines are unchanged`, () => {
    const p = profile(format, [], sampleCount);
    assert.equal(p.blendable, true);
    assert.deepEqual(p.blendModes(), [false, true]);
    assert.deepEqual(p.blendModes({blend: null}), [false, true]);
    assert.deepEqual(p.blendModes({blend: BLEND}), [false, true]);
  });
}
test('depth-only renderers keep a single unblended pipeline key', () => {
  const p = profile(null);
  assert.equal(p.blendable, false);
  assert.deepEqual(p.blendModes(), [false]);
  assert.deepEqual(p.blendModes({blend: null}), [false]);
  assert.throws(() => p.assertBlend(BLEND), /color attachment/);
});
test('unsupported formats and sample counts are rejected before feature access', () => {
  const device = {get features() { throw Error('feature access should not happen'); }};
  for (const format of ['', undefined, 'rgba32uint', 'depth32float'])
    assert.throws(() => createAnimationColorTargetProfile(device, {format, sampleCount: 1}, reject), /configuration/);
  for (const sampleCount of [0, 2, 8, '1', undefined, NaN])
    assert.throws(() => createAnimationColorTargetProfile(device, {format: 'rgba32float', sampleCount}, reject), /configuration/);
});
test('feature admission requires enabled device support, not a truthy substitute', () => {
  for (const features of [undefined, {has: () => 1}, {has: () => false}]) {
    const p = createAnimationColorTargetProfile({features}, {format: 'rgba32float', sampleCount: 1}, reject);
    assert.equal(p.blendable, false);
  }
});
test('profiles and shared pipeline-key arrays cannot be mutated', () => {
  const p = profile('rgba32float');
  assert.ok(Object.isFrozen(p));
  assert.ok(Object.isFrozen(p.blendModes()));
  assert.ok(Object.isFrozen(p.blendModes({blend: null})));
  assert.throws(() => p.blendModes().push(true), TypeError);
  assert.deepEqual(profile('r32float').blendModes(), [false]);
});
