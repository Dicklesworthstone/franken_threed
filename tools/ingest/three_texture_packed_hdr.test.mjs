/** Production packed-HDR residency with a byte-recording GPU boundary.
 * These checks preserve source words and lifecycle; they do not execute WGSL.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuThreeTextures, inspectThreeTexture} from './three_textures.mjs';
import {T as source, device} from './tests/volume_texture_test_fixture.mjs';

const T = {...source, RGBFormat: 1022, UnsignedInt5999Type: 35902, UnsignedInt101111Type: 35899};
const profiles = [['RGB9E5', T.UnsignedInt5999Type, 'rgb9e5ufloat'],
  ['RGB11F', T.UnsignedInt101111Type, 'rg11b10ufloat']];
const pool = (d, options = {}) => createGpuThreeTextures(d, {three: T, ...options});
const words = texture => new Uint32Array(texture.levels[0].buffer);
function texture(family = 'DataTexture', type = profiles[0][1], width = 3, height = 2, depth = 3) {
  const count = width * height * (family === 'DataTexture' ? 1 : depth);
  const backing = new Uint32Array(count + 2), data = backing.subarray(1, count + 1);
  data.set(Array.from({length: count}, (_, i) => (0xf8000000 + i * 0x10203) >>> 0));
  const value = new T[family](data, width, height, depth);
  Object.assign(value, {format: T.RGBFormat, type, minFilter: T.LinearFilter, magFilter: T.LinearFilter});
  value.needsUpdate = true;
  return value;
}

for (const [name, type, format] of profiles)
for (const family of ['DataTexture', 'DataArrayTexture', 'Data3DTexture'])
test(`${name} ${family} uploads one Uint32 per texel without a filtering or renderability feature`, async t => {
  const d = device(), p = pool(d), input = texture(family, type), image = input.image;
  t.after(() => p.dispose());
  const before = input.image.data.slice();
  p.prepare([input]); await p.whenIdle();
  const native = p.nativeTexture(input), binding = p.binding(input);
  assert.equal(native.desc.format, format); assert.equal(native.desc.usage, 1 | 2 | 4);
  assert.equal(native.dimension, family === 'Data3DTexture' ? '3d' : '2d');
  assert.equal(binding.view.dimension, family === 'Data3DTexture' ? '3d' : family === 'DataArrayTexture' ? '2d-array' : '2d');
  assert.equal(binding.sampleType, 'float'); assert.equal(binding.filtering, true);
  assert.equal(binding.sampler.minFilter, 'linear'); assert.equal(binding.sampler.magFilter, 'linear');
  assert.deepEqual(words(native), before); assert.equal(input.image, image);
  assert.equal(d.writes[0].layout.bytesPerRow, 12);
  assert.equal(p.diagnostics.textureBytes, before.byteLength);
  input.image.data.fill(0xffffffff); p.update([input]);
  assert.deepEqual(words(native), before, 'unrequested CPU edits remain GPU-stale');
  input.needsUpdate = true; p.update([input]);
  assert.deepEqual(words(native), input.image.data); assert.equal(p.binding(input).view, binding.view);
});

for (const [name, type] of profiles) test(`${name} row padding, typed-array offset and flipY preserve packed words`, t => {
  const d = device(), p = pool(d), input = texture('DataTexture', type, 1, 2);
  t.after(() => p.dispose());
  const backing = Uint32Array.from([0xdeadbeef, 0xf8000001, 0xabcdef01, 0x84020100, 0xdeadbeef]);
  input.image.data = backing.subarray(1, 4); input.unpackAlignment = 8; input.flipY = true;
  p.prepare([input]); assert.deepEqual([...words(p.nativeTexture(input))], [0x84020100, 0xf8000001]);
  assert.deepEqual([...backing], [0xdeadbeef, 0xf8000001, 0xabcdef01, 0x84020100, 0xdeadbeef]);
  assert.equal(p.diagnostics.textureBytes, 8);
});

for (const [name, type] of profiles) test(`${name} array layer requests keep unrequested packed layers GPU-stale`, t => {
  const d = device(), p = pool(d), input = texture('DataArrayTexture', type, 2, 2, 3);
  t.after(() => p.dispose()); p.prepare([input]);
  const native = p.nativeTexture(input), before = words(native).slice();
  input.image.data.fill(0xffc00001); input.addLayerUpdate(1); p.update([input]);
  assert.deepEqual(words(native), before); assert.equal(input.layerUpdates.size, 1);
  input.needsUpdate = true; p.update([input]);
  const expected = before.slice(); expected.fill(0xffc00001, 4, 8);
  assert.deepEqual(words(native), expected); assert.equal(input.layerUpdates.size, 0);
  assert.deepEqual(d.writes.at(-1).dest.origin, [0, 0, 1]);
  assert.deepEqual(d.writes.at(-1).size, [2, 2, 1]);
});

for (const [name, type, format] of profiles) test(`${name} authored 2D and cube mip chains retain exact packed levels`, t => {
  const d = device(), p = pool(d); t.after(() => p.dispose());
  const input = texture('DataTexture', type, 4, 2);
  input.minFilter = T.LinearMipmapLinearFilter;
  input.mipmaps = [input.image, {width: 2, height: 1, data: Uint32Array.of(0xffffffff, 0x80000001)},
    {width: 1, height: 1, data: Uint32Array.of(0x81234567)}];
  p.prepare([input]);
  assert.equal(p.binding(input).sampler.lodMaxClamp, 2);
  assert.deepEqual([...new Uint32Array(p.nativeTexture(input).levels[1].buffer)], [0xffffffff, 0x80000001]);
  assert.equal(p.diagnostics.textureBytes, (8 + 2 + 1) * 4); assert.equal(d.passes.length, 0);
  const faces = Array.from({length: 6}, (_, i) => {
    const face = texture('DataTexture', type, 2, 2); face.image.data.fill(i + 1); return face;
  });
  const cube = new T.CubeTexture(faces);
  Object.assign(cube, {format: T.RGBFormat, type, generateMipmaps: false,
    minFilter: T.LinearMipmapLinearFilter, magFilter: T.LinearFilter,
    mipmaps: [{image: Array.from({length: 6}, (_, i) => {
      const face = texture('DataTexture', type, 1, 1); face.image.data[0] = i + 11; return face;
    })}]});
  cube.needsUpdate = true; p.prepare([cube]);
  const gpu = p.nativeTexture(cube);
  assert.equal(gpu.desc.format, format); assert.equal(p.binding(cube).view.dimension, 'cube');
  assert.deepEqual([...new Uint32Array(gpu.levels[1].buffer)], [11, 12, 13, 14, 15, 16]);
  assert.equal(p.diagnostics.textureBytes, (8 + 2 + 1 + 6 * (4 + 1)) * 4);
  assert.equal(d.passes.length, 0);
});

test('RGB11F generated 2D, array and volume mips require the enabled renderability feature', t => {
  for (const family of ['DataTexture', 'DataArrayTexture', 'Data3DTexture']) {
    const input = texture(family, T.UnsignedInt101111Type, 4, 2, 3);
    input.generateMipmaps = true; input.minFilter = T.LinearMipmapLinearFilter;
    const blocked = device(), p = pool(blocked); t.after(() => p.dispose());
    assert.throws(() => p.prepare([input]), {code: 'THREE_TEXTURE_MIPS'});
    assert.equal(blocked.textures.length, 0); assert.equal(blocked.writes.length, 0);
    const enabled = device({features: ['rg11b10ufloat-renderable']}), q = pool(enabled); t.after(() => q.dispose());
    q.prepare([input]); const native = q.nativeTexture(input);
    assert.equal(native.desc.usage, 1 | 2 | 4 | 16);
    assert.ok(enabled.passes.length > 0); assert.equal(enabled.pipelines.length, 1);
    assert.equal(enabled.pipelines[0].fragment.targets[0].format, 'rg11b10ufloat');
    assert.equal(q.diagnostics.textureBytes, family === 'DataTexture' ? 44 : family === 'DataArrayTexture' ? 132 : 108);
    assert.deepEqual(words(native), input.image.data);
  }
});

test('RGB9E5 never generates a renderable chain, including explicit writes to authored mip storage', t => {
  const d = device({features: ['rg11b10ufloat-renderable']}), p = pool(d); t.after(() => p.dispose());
  const input = texture('DataTexture', T.UnsignedInt5999Type, 2, 2);
  input.generateMipmaps = true; input.minFilter = T.LinearMipmapLinearFilter;
  assert.throws(() => p.prepare([input]), {code: 'THREE_TEXTURE_MIPS'});
  assert.equal(d.textures.length, 0);
  input.generateMipmaps = false;
  input.mipmaps = [input.image, {width: 1, height: 1, data: Uint32Array.of(0x12345678)}];
  p.prepare([input]); assert.equal(p.nativeTexture(input).desc.usage, 1 | 2 | 4);
  let wrote = false;
  assert.throws(() => p.writeTexture(input, () => { wrote = true; }, {generateMipmaps: true}), {code: 'THREE_TEXTURE_COPY'});
  assert.equal(wrote, false); assert.equal(d.passes.length, 0);
});

test('packed formats stay independent with shared Source and obey peak residency budgets', t => {
  const d = device(), p = pool(d, {maxTextureBytes: 48}); t.after(() => p.dispose());
  const first = texture(), alias = first.clone(), other = first.clone(); other.type = T.UnsignedInt101111Type;
  p.prepare([first, alias, other]);
  assert.equal(p.binding(first).view, p.binding(alias).view);
  assert.notEqual(p.binding(first).view, p.binding(other).view); assert.equal(p.diagnostics.textureBytes, 48);
  first.wrapS = T.RepeatWrapping;
  assert.throws(() => p.prepare([first]), {code: 'THREE_TEXTURE_LIMIT'});
  assert.equal(d.textures.length, 2); assert.ok(d.textures.every(texture => !texture.destroyed));
});

test('packed storage, color and partial-range refusals precede every allocation', t => {
  for (const change of [input => { input.image.data = new Float32Array(6); },
    input => { input.image.data = new Uint32Array(5); }, input => { input.type = T.FloatType; },
    input => { input.format = T.RGBAFormat; }, input => { input.colorSpace = T.SRGBColorSpace; },
    input => { input.image.data = new Uint32Array(new SharedArrayBuffer(24)); },
    input => { input.image.data = new Uint32Array(new ArrayBuffer(24, {maxByteLength: 48})); },
    input => { input.updateRanges.push({start: 0, count: 4}); }]) {
    const d = device(), p = pool(d), valid = texture(), invalid = texture(); t.after(() => p.dispose());
    change(invalid); assert.throws(() => p.prepare([valid, invalid]));
    assert.equal(d.textures.length, 0); assert.equal(d.writes.length, 0);
  }
});

test('native packed 2D requests upload the whole image and preserve source updateRanges', t => {
  for (const [, type] of profiles) {
    const d = device(), p = pool(d, {nativeMedia: true}), input = texture('DataTexture', type); t.after(() => p.dispose());
    input.updateRanges.push({start: 4, count: 4}); const ranges = input.updateRanges.map(value => ({...value}));
    p.prepare([input]); const native = p.nativeTexture(input);
    input.image.data.fill(0x9abcdef0); input.needsUpdate = true; p.update([input]);
    assert.deepEqual(words(native), input.image.data); assert.deepEqual(input.updateRanges, ranges);
    assert.deepEqual(d.writes.map(write => write.size), [[3, 2, 1], [3, 2, 1]]);
  }
});

test('packed metadata budgets count physical texels before reading source pixel values', () => {
  const input = texture('DataTexture', T.UnsignedInt5999Type, 4, 2);
  assert.equal(inspectThreeTexture(input, T).bytes, 32);
  assert.throws(() => inspectThreeTexture(input, T, {maxTextureBytes: 31}), {code: 'THREE_TEXTURE_LIMIT'});
});
