/** A byte interpreter for recorded copies, not GPU execution or pixel parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {submitTextureCopy} from './three_texture_copy_lifetime.mjs';
import {targetCopyFixture, box, point, failure} from './three_target_copy_fixtures.mjs';

const pixelBytes = format => ({r8unorm: 1, rg11b10ufloat: 4, rgba8unorm: 4,
  'rgba8unorm-srgb': 4, rgba16float: 8, rgba32float: 16}[format]);
function bytesFixture(sourceOptions = {}, destinationOptions = sourceOptions) {
  const f = targetCopyFixture();
  const source = f.target({width: 8, height: 4, mips: 3, dimension: '2d-array', depth: 32, ...sourceOptions});
  const destination = f.target({width: 8, height: 4, mips: 3, dimension: '2d-array', depth: 32, ...destinationOptions});
  for (const color of destination.entry.shape.colors) color.generateMipmaps = false;
  function allocate(target, seed) {
    const shape = target.entry.shape, bytes = pixelBytes(shape.format);
    assert.ok(bytes);
    for (const [attachment, native] of target.entry.native.entries()) {
      native.levels = Array.from({length: shape.mipLevelCount}, (_, level) => {
        const width = Math.max(1, Math.floor(shape.width / 2 ** level));
        const height = Math.max(1, Math.floor(shape.height / 2 ** level));
        const depth = shape.volume ? Math.max(1, Math.floor(shape.layers / 2 ** level)) : shape.layers;
        return {width, height, depth, bytes, data: Uint8Array.from({length: width * height * depth * bytes},
          (_, i) => (seed + i * 17 + level * 43 + attachment * 71) & 255)};
      });
    }
  }
  allocate(source, 19); allocate(destination, 173);
  const xyz = value => Array.isArray(value) ? value : [value.x, value.y, value.z];
  function execute(command) {
    const a = command.from.texture.levels[command.from.mipLevel], b = command.to.texture.levels[command.to.mipLevel];
    const [sx, sy, sz] = xyz(command.from.origin), [dx, dy, dz] = xyz(command.to.origin);
    const [width, height, depth] = Array.isArray(command.size) ? command.size
      : [command.size.width, command.size.height, command.size.depthOrArrayLayers];
    assert.equal(a.bytes, b.bytes);
    assert.ok(sx + width <= a.width && sy + height <= a.height && sz + depth <= a.depth);
    assert.ok(dx + width <= b.width && dy + height <= b.height && dz + depth <= b.depth);
    for (let z = 0; z < depth; z++) for (let y = 0; y < height; y++) {
      const from = (((sz + z) * a.height + sy + y) * a.width + sx) * a.bytes;
      const to = (((dz + z) * b.height + dy + y) * b.width + dx) * b.bytes;
      b.data.set(a.data.subarray(from, from + width * a.bytes), to);
    }
  }
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => { submit(commands); commands.forEach(execute); };
  // Interpret the existing per-face recording owner as well, so the oracle
  // also models the pre-batching implementation rather than relying on a stub.
  const copyColor = f.targets.copyColor;
  f.targets.copyColor = (...args) => {
    const start = f.commands.length, result = copyColor(...args);
    f.commands.slice(start).forEach(execute); return result;
  };
  const copy = (...args) => submitTextureCopy(f.renderer,
    captureTargetColorCopy(f.renderer, source.texture, destination.texture, ...args), f.rejected);
  return {f, source, destination, copy};
}
function expectedRange(source, destination, from, to, size, sourceLevel = 0, destinationLevel = 0) {
  const a = source.entry.native[source.textures.indexOf(source.texture)].levels[sourceLevel];
  const target = destination.entry.native[destination.textures.indexOf(destination.texture)];
  const expected = target.levels.map(level => level.data.slice()), b = target.levels[destinationLevel];
  // Independent texel/component oracle, not the row-copy interpreter above.
  for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) for (let x = 0; x < size[0]; x++) {
    const sourcePixel = ((from[2] + z) * a.height + from[1] + y) * a.width + from[0] + x;
    const destinationPixel = ((to[2] + z) * b.height + to[1] + y) * b.width + to[0] + x;
    for (let byte = 0; byte < a.bytes; byte++)
      expected[destinationLevel][destinationPixel * b.bytes + byte] = a.data[sourcePixel * a.bytes + byte];
  }
  return expected;
}
const assertLevels = (destination, expected) => assert.deepEqual(
  destination.entry.native[destination.textures.indexOf(destination.texture)].levels.map(level => level.data), expected);

for (const dimension of ['2d-array', '3d']) {
  test(`${dimension} level-zero range uses one submission and preserves all destination bytes`, () => {
    const {f, source, destination, copy} = bytesFixture({dimension});
    const expected = expectedRange(source, destination, [0, 0, 0], [0, 0, 0], [8, 4, 32]);
    copy();
    assert.equal(f.commands.length, 1); assert.equal(f.events.filter(event => event === 'submit').length, 1);
    assert.equal(f.events.filter(event => event === 'source-borrow').length, 1);
    assertLevels(destination, expected); assert.equal(f.references, 0);
  });
}

test('all six cube faces copy into one non-cube range command', () => {
  const {f, source, destination, copy} = bytesFixture({dimension: 'cube', width: 4}, {depth: 8, width: 4});
  const expected = expectedRange(source, destination, [0, 0, 0], [0, 0, 0], [4, 4, 6]);
  copy(); assert.equal(f.commands.length, 1); assertLevels(destination, expected);
});

for (const [from, to] of [['3d', '2d-array'], ['2d-array', '3d']]) {
  test(`partial ${from} to ${to} ranges leave every untouched texel, layer and mip unchanged`, () => {
    const {f, source, destination, copy} = bytesFixture({dimension: from, depth: 8}, {dimension: to, depth: 8});
    const expected = expectedRange(source, destination, [1, 1, 2], [3, 0, 1], [3, 2, 3]);
    copy(box(1, 1, 2, 3, 2, 3), point(3, 0, 1));
    assert.equal(f.commands.length, 1); assertLevels(destination, expected); assert.equal(f.references, 0);
  });
}

for (const format of ['r8unorm', 'rg11b10ufloat', 'rgba16float', 'rgba32float']) {
  test(`${format} batched ranges preserve original component bits`, () => {
    const {f, source, destination, copy} = bytesFixture({format, depth: 4});
    const expected = expectedRange(source, destination, [0, 0, 1], [2, 0, 0], [4, 2, 3]);
    copy(box(0, 0, 1, 4, 2, 3), point(2, 0, 0));
    assert.equal(f.commands.length, 1); assertLevels(destination, expected);
  });
}

test('linear/sRGB compatible copies preserve bits rather than converting color', () => {
  const {f, source, destination, copy} = bytesFixture({format: 'rgba8unorm-srgb', depth: 4}, {depth: 4});
  const expected = expectedRange(source, destination, [0, 0, 0], [0, 0, 0], [8, 4, 4]);
  copy(); assert.equal(f.commands.length, 1); assertLevels(destination, expected);
});

test('queued copy observes preceding GPU contents and publishes before the next FIFO request', () => {
  const {f, source, destination, copy} = bytesFixture({depth: 4});
  let expected;
  f.renderer._pending.push({issue() {
    source.entry.native[0].levels[0].data.fill(93);
    expected = expectedRange(source, destination, [0, 0, 1], [1, 1, 0], [3, 2, 2]);
  }, resolve() {}, reject: assert.fail});
  const region = box(0, 0, 1, 3, 2, 2), position = point(1, 1, 0);
  copy(region, position); region.min.z = 3; position.y = 99;
  f.renderer._pending.push({issue() { assertLevels(destination, expected); }, resolve() {}, reject: assert.fail});
  assert.equal(f.renderer._pending.length, 3); assert.equal(f.references, 2);
  f.flush(); assert.equal(f.commands.length, 1); assert.deepEqual(f.errors, []); assert.equal(f.references, 0);
});

test('source and destination MRT selections survive native range batching', () => {
  const {f, source, destination, copy} = bytesFixture({depth: 4, count: 3}, {depth: 4, count: 2});
  source.texture = source.textures[2]; destination.texture = destination.textures[1];
  const primary = destination.entry.native[0].levels.map(level => level.data.slice());
  const expected = expectedRange(source, destination, [0, 0, 0], [0, 0, 0], [8, 4, 4]);
  copy(); assert.equal(f.commands.length, 1); assertLevels(destination, expected);
  assert.deepEqual(destination.entry.native[0].levels.map(level => level.data), primary);
});

test('resolved MSAA array colors retain one-copy selection at the destination', () => {
  const {f, source, destination, copy} = bytesFixture({samples: 4, depth: 8}, {depth: 8});
  const expected = expectedRange(source, destination, [0, 0, 0], [0, 0, 0], [8, 4, 8]);
  copy(); assert.equal(f.commands.length, 1); assertLevels(destination, expected);
});

test('native submission failure cannot leave a partially copied layer prefix', () => {
  const {f, destination, copy} = bytesFixture({depth: 32});
  const original = destination.entry.native[0].levels.map(level => level.data.slice());
  f.device.onSubmit = () => { throw failure('GPU_SUBMIT'); };
  assert.throws(() => copy(), {code: 'GPU_SUBMIT'});
  assert.equal(f.commands.length, 0); assertLevels(destination, original); assert.equal(f.references, 0);
});

test('level-zero batches propagate producer validation without awaiting the GPU in the FIFO', async () => {
  const {f, source, copy} = bytesFixture({depth: 8});
  const error = failure('SOURCE_GPU'); source.entry.storage.validation = Promise.reject(error);
  source.entry.storage.validation.catch(() => {});
  copy(); assert.equal(f.commands.length, 1);
  await assert.rejects(Promise.all(f.validations), error); assert.equal(f.references, 0);
});
