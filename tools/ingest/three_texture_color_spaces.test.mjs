/** Registered native sampled spaces use r186's hardware-transfer selection.
 * The byte recorder checks uploads and views, not shader execution or pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.core.js';
import {ColorManagement as sourceColorManagement} from '../../upstream/three.js/src/math/ColorManagement.js';
import {getFormat} from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import {DisplayP3ColorSpace, DisplayP3ColorSpaceImpl, LinearDisplayP3ColorSpace,
  LinearDisplayP3ColorSpaceImpl, LinearRec2020ColorSpace, LinearRec2020ColorSpaceImpl,
  ExtendedSRGBColorSpace, ExtendedSRGBColorSpaceImpl} from '../../upstream/three.js/examples/jsm/math/ColorSpaces.js';
import {createGpuThreeTextures, inspectThreeTexture} from './three_textures.mjs';
import {device as textureDevice} from './tests/volume_texture_test_fixture.mjs';

const definitions = {
  [DisplayP3ColorSpace]: DisplayP3ColorSpaceImpl,
  [LinearDisplayP3ColorSpace]: LinearDisplayP3ColorSpaceImpl,
  [LinearRec2020ColorSpace]: LinearRec2020ColorSpaceImpl,
  [ExtendedSRGBColorSpace]: ExtendedSRGBColorSpaceImpl,
};
const features = new Set(['texture-formats-tier1', 'float32-filterable',
  'texture-compression-bc', 'texture-compression-etc2', 'texture-compression-astc']);
const native = {nativeMedia: true, nativeCube: true, features, float32Filterable: true};
function register(t) {
  for (const management of [T.ColorManagement, sourceColorManagement]) {
    const old = new Map(Object.keys(definitions).map(space => [space, management.spaces[space]]));
    management.define(definitions);
    t.after(() => { for (const [space, value] of old) {
      if (value === undefined) delete management.spaces[space]; else management.spaces[space] = value;
    } });
  }
}
function data(t, space, {kind = '2d', format = T.RGBAFormat, type = T.UnsignedByteType,
  Kind = Uint8Array, channels = 4, normalized = false} = {}) {
  const pixels = () => Kind.from({length: 2 * 2 * (kind === 'array' || kind === '3d' ? 2 : 1) * channels}, (_, i) => i + 1);
  const texture = kind === 'cube' ? new T.CubeTexture(Array.from({length: 6}, () => new T.DataTexture(pixels(), 2, 2)))
    : kind === 'array' ? new T.DataArrayTexture(pixels(), 2, 2, 2)
      : kind === '3d' ? new T.Data3DTexture(pixels(), 2, 2, 2) : new T.DataTexture(pixels(), 2, 2);
  Object.assign(texture, {format, type, normalized, colorSpace: space, flipY: false,
    generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
function compressed(t, space, format, blockBytes = 16, type = T.UnsignedByteType) {
  const texture = new T.CompressedTexture([{width: 4, height: 4, data: new Uint8Array(blockBytes)}], 4, 4, format, type);
  texture.colorSpace = space; texture.minFilter = T.NearestFilter; texture.needsUpdate = true;
  t.after(() => texture.dispose()); return texture;
}

for (const space of Object.keys(definitions))
test(`native ${space} matches pinned texture-format selection`, t => {
  register(t);
  for (const profile of [
    {}, {format: T.RedFormat, channels: 1}, {format: T.RGFormat, channels: 2},
    {type: T.HalfFloatType, Kind: Uint16Array}, {type: T.FloatType, Kind: Float32Array},
    {type: T.ByteType, Kind: Int8Array},
    {type: T.ShortType, Kind: Int16Array, normalized: true},
    {type: T.UnsignedShortType, Kind: Uint16Array, normalized: true},
    {format: T.RGBFormat, type: T.UnsignedInt5999Type, Kind: Uint32Array, channels: 1},
  ]) {
    const texture = data(t, space, profile), description = inspectThreeTexture(texture, T, native);
    assert.equal(description.viewFormat, getFormat(texture, {features}));
    assert.equal(description.source, texture.source);
    assert.equal(texture.colorSpace, space);
  }
  for (const [format, bytes, type] of [
    [T.RGBA_S3TC_DXT1_Format, 8], [T.RGBA_ETC2_EAC_Format, 16], [T.RGBA_ASTC_4x4_Format, 16],
    [T.RGB_BPTC_UNSIGNED_Format, 16, T.HalfFloatType], [T.RED_RGTC1_Format, 8],
  ]) {
    const texture = compressed(t, space, format, bytes, type);
    assert.equal(inspectThreeTexture(texture, T, native).viewFormat, getFormat(texture, {features}));
  }
});

test('registered spaces retain exact bytes and source sharing across native dimensions', async t => {
  register(t);
  const device = textureDevice({features: [...features]});
  const pool = createGpuThreeTextures(device, {three: T, nativeMedia: true, nativeCube: true});
  t.after(() => pool.dispose());
  for (const kind of ['2d', 'array', '3d', 'cube']) {
    const texture = data(t, DisplayP3ColorSpace, {kind}), source = texture.source;
    const alias = texture.clone(); alias.colorSpace = ExtendedSRGBColorSpace;
    t.after(() => alias.dispose()); pool.prepare([texture, alias]);
    const binding = pool.binding(texture);
    assert.equal(binding.view, pool.binding(alias).view);
    assert.equal(binding.view.format, getFormat(texture, device));
    assert.equal(binding.view.dimension, kind === 'array' ? '2d-array' : kind);
    const faces = kind === 'cube' ? texture.image.map(face => face.image.data) : [texture.image.data];
    const expected = Uint8Array.from(faces.flatMap(face => [...new Uint8Array(face.buffer, face.byteOffset, face.byteLength)]));
    assert.deepEqual(binding.view.texture.levels[0], expected);
    assert.equal(texture.source, source); assert.equal(texture.colorSpace, DisplayP3ColorSpace);
    assert.equal(alias.source, source); assert.equal(alias.colorSpace, ExtendedSRGBColorSpace);
  }
  await pool.validation; assert.equal(pool.diagnostics.allocations, 4);
});

test('native loading and disabled color management retain source format rules', t => {
  register(t);
  const previous = T.ColorManagement.enabled; T.ColorManagement.enabled = false;
  t.after(() => { T.ColorManagement.enabled = previous; });
  const texture = new T.DataTexture(); texture.colorSpace = DisplayP3ColorSpace;
  t.after(() => texture.dispose());
  const description = inspectThreeTexture(texture, T, native);
  assert.equal(description.nativeAllocation, 'default');
  assert.equal(description.format, getFormat(texture, {features}));
  assert.equal(description.viewFormat, 'rgba8unorm-srgb');
  const cube = new T.CubeTexture(); cube.colorSpace = DisplayP3ColorSpace;
  t.after(() => cube.dispose());
  assert.equal(inspectThreeTexture(cube, T, native).viewFormat, 'rgba8unorm',
    'r186 constructs its default cube from an ordinary byte CubeTexture');
});

test('unsupported native spaces fail before any upload and legacy boundaries remain explicit', t => {
  register(t);
  const device = textureDevice(), pool = createGpuThreeTextures(device, {three: T, nativeMedia: true});
  t.after(() => pool.dispose());
  const valid = data(t, DisplayP3ColorSpace), invalid = data(t, 'unregistered-color-space');
  assert.throws(() => pool.prepare([valid, invalid]), {code: 'THREE_TEXTURE_COLOR'});
  assert.equal(device.textures.length, 0); assert.equal(device.writes.length, 0);
  assert.throws(() => inspectThreeTexture(valid, T), {code: 'THREE_TEXTURE_COLOR'});
  const hdr = data(t, T.SRGBColorSpace, {type: T.HalfFloatType, Kind: Uint16Array});
  assert.throws(() => inspectThreeTexture(hdr, T), {code: 'THREE_TEXTURE_COLOR'});
  assert.equal(inspectThreeTexture(hdr, T, native).viewFormat, getFormat(hdr, {features}));
  const blocks = compressed(t, T.SRGBColorSpace, T.RGB_BPTC_UNSIGNED_Format, 16, T.HalfFloatType);
  assert.throws(() => inspectThreeTexture(blocks, T, {features}), {code: 'THREE_TEXTURE_COLOR'});
  assert.equal(inspectThreeTexture(blocks, T, native).viewFormat, getFormat(blocks, {features}));
});
