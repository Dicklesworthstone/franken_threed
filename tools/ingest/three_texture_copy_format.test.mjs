import test from 'node:test';
import assert from 'node:assert/strict';
import {inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';

const shape = format => ({viewDimension: '2d', width: 8, height: 8, depth: 1,
  mipLevelCount: 2, format});

for (const prefix of ['rgba8unorm', 'bgra8unorm']) {
  for (const [source, destination] of [[prefix, `${prefix}-srgb`], [`${prefix}-srgb`, prefix]]) {
    test(`native texture copy preserves byte-compatible ${source} to ${destination}`, () => {
      const plan = inspectStorageTextureCopy(shape(source), shape(destination),
        {isBox2: true, min: {x: 0, y: 0}, max: {x: 4, y: 4}}, null, 0, 1);
      assert.deepEqual(plan.size, [4, 4, 1]);
      assert.equal(plan.destinationLevel, 1);
    });
  }
}

test('native copies still reject non-compatible formats and different texel widths', () => {
  for (const [source, destination] of [
    ['rgba8unorm', 'bgra8unorm'],
    ['rgba8unorm-srgb', 'bgra8unorm-srgb'],
    ['rgba8unorm', 'rgba16float'],
    ['rgba16float', 'rgba16float-srgb'],
  ]) {
    assert.throws(() => inspectStorageTextureCopy(shape(source), shape(destination)),
      {code: 'THREE_STORAGE_COPY_FORMAT'});
  }
});

test('compatible formats do not bypass same-subresource or mip bounds checks', () => {
  assert.throws(() => inspectStorageTextureCopy(shape('rgba8unorm'),
    shape('rgba8unorm-srgb'), null, null, 0, 0, true),
    {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  assert.throws(() => inspectStorageTextureCopy(shape('rgba8unorm'),
    shape('rgba8unorm-srgb'), null, null, 0, 2),
    {code: 'THREE_STORAGE_COPY_LEVEL'});
});
