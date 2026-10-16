/** CPU cubemap/source-alias capture contracts; recording source pool only.
 * Run: node --test tools/ingest/tests_three_compute_cube_inputs.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeComputeTextures} from './three_compute_textures.mjs';

let ids = 0;
class Texture {
  constructor(image) {
    this.id = ++ids; this.isTexture = true;
    this.source = {data: image, version: 1, dataReady: true}; this.version = 1;
    this.mipmaps = []; this.updateRanges = []; this.onUpdate = null; this.generateMipmaps = false;
  }
  get image() { return this.source.data; }
  addEventListener(type, listener) { ((this._listeners ??= {})[type] ??= new Set()).add(listener); }
  removeEventListener(type, listener) { this._listeners?.[type]?.delete(listener); }
  dispose() { for (const listener of [...this._listeners?.dispose ?? []]) listener(); }
  touch() { this.version++; this.source.version++; }
}
class DataTexture extends Texture {
  constructor(value, size = 1) {
    super({width: size, height: size, data: new Uint8Array(size * size * 4).fill(value)}); this.isDataTexture = true;
  }
}
class CubeTexture extends Texture {
  constructor(faces) { super(faces); this.isCubeTexture = true; }
}
class CompressedTexture extends Texture {
  constructor(value) {
    super({width: 4, height: 4}); this.isCompressedTexture = true;
    this.mipmaps = [{width: 4, height: 4, data: new Uint8Array(8).fill(value)}];
  }
}
const bind = texture => ({texture, isSampledTexture: true, isSampler: true});
const faces = (size = 1) => Array.from({length: 6}, (_, i) => new DataTexture(i + 1, size));
function rig(options = {}) {
  const bindings = new Map(), resources = new Map(), uploads = [];
  let disposed = false;
  const sourcePool = {
    inspect(t) {
      assert.ok(t instanceof Texture);
      const cube = t.isCubeTexture === true;
      const compressed = t.isCompressedTexture === true;
      const data = cube ? t.image.every(face => compressed ? Array.isArray(face.mipmaps) : face instanceof DataTexture) : !!t.isDataTexture;
      const width = cube ? (t.image[0].image ?? t.image[0]).width : t.image.width;
      return {cube, data, manual: compressed || t.mipmaps.length > 0, sampleType: 'float',
        viewDimension: cube ? 'cube' : '2d', key: JSON.stringify([cube, compressed, width, t.mipmaps.length])};
    },
    prepare(textures) {
      for (const t of textures) {
        let resource = resources.get(t.source);
        if (!resource) {
          const cube = t.isCubeTexture, compressed = t.isCompressedTexture;
          const base = cube ? compressed ? t.image.map(face => face.mipmaps[0]) : t.image.map(face => face.image) : [t.image];
          const mips = cube ? t.mipmaps.map(mip => mip.image.map(face => [...face.image.data])) : [];
          resource = {view: {base: base.map(image => [...image.data]), mips}, sampler: {}};
          resources.set(t.source, resource); uploads.push(t); t.onUpdate?.(t);
        }
        bindings.set(t, resource);
      }
    },
    binding(t) { return bindings.get(t); },
    retain(textures) {
      for (const t of bindings.keys()) if (!textures.includes(t)) bindings.delete(t);
      for (const source of resources.keys()) if (![...bindings.keys()].some(t => t.source === source)) resources.delete(source);
    },
    async whenIdle() {}, dispose() { disposed = true; resources.clear(); bindings.clear(); },
  };
  const storage = {captureBinding() { assert.fail('not a storage texture'); }, validateDispatch() {}, written() {}, async whenIdle() {}};
  const owner = createThreeComputeTextures({storage, poolFactory: () => sourcePool, ...options});
  return {owner, resources, uploads, get disposed() { return disposed; }};
}

test('six source DataTexture faces survive queued edits without source IDs or callbacks', () => {
  const r = rig(), texture = new CubeTexture(faces()), before = ids;
  let callbacks = 0; texture.onUpdate = function (value) { assert.equal(this, texture); assert.equal(value, texture); callbacks++; };
  for (const face of texture.image) face.onUpdate = () => assert.fail('cube uploads do not invoke face upload callbacks');
  const first = r.owner.captureBinding(bind(texture));
  for (const face of texture.image) face.image.data.fill(77);
  texture.touch(); const second = r.owner.captureBinding(bind(texture));
  for (const face of texture.image) face.image.data.fill(99);
  assert.deepEqual(r.owner.layout(first), {texture: {sampleType: 'float', viewDimension: 'cube', multisampled: false}});
  assert.deepEqual(r.owner.resolve(first).base.map(data => data[0]), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(r.owner.resolve(second).base.map(data => data[0]), [77, 77, 77, 77, 77, 77]);
  const captured = r.uploads[0];
  assert.ok(captured instanceof CubeTexture); assert.notEqual(captured.image, texture.image);
  for (const [i, face] of captured.image.entries()) {
    assert.ok(face instanceof DataTexture); assert.notEqual(face, texture.image[i]);
    assert.notEqual(face.source, texture.image[i].source); assert.notEqual(face.image.data, texture.image[i].image.data);
  }
  assert.equal(callbacks, 2); assert.equal(ids, before);
  r.owner.release(first); r.owner.release(second); r.owner.dispose();
});
test('authored cube mip faces are captured as additional levels, including Texture descriptors', () => {
  const r = rig(), texture = new CubeTexture(faces(2));
  const mip = new CubeTexture(faces()); texture.mipmaps = [mip];
  const before = ids, lease = r.owner.captureBinding(bind(texture));
  mip.image[0].image.data.fill(55);
  const result = r.owner.resolve(lease);
  assert.equal(result.base[0].length, 16); assert.equal(result.mips.length, 1);
  assert.equal(result.mips[0][0].length, 4); assert.equal(result.mips[0][0][0], 1);
  assert.ok(r.uploads[0].mipmaps[0] instanceof CubeTexture);
  assert.equal(r.owner.diagnostics.snapshotBytes, 6 * (16 + 4));
  assert.equal(ids, before); r.owner.release(lease); r.owner.dispose();
});
test('compressed cube face chains preserve immutable blocks for class and plain faces', () => {
  for (const plainFaces of [false, true]) {
    const cubeFaces = Array.from({length: 6}, (_, i) => new CompressedTexture(i + 1));
    const texture = new CubeTexture(plainFaces ? cubeFaces.map(face => ({...face.image, mipmaps: face.mipmaps})) : cubeFaces);
    texture.isCompressedTexture = true;
    const r = rig(), before = ids, lease = r.owner.captureBinding(bind(texture));
    for (const face of cubeFaces) face.mipmaps[0].data.fill(88);
    assert.deepEqual(r.owner.resolve(lease).base.map(data => data[0]), [1, 2, 3, 4, 5, 6]);
    assert.equal(r.owner.diagnostics.snapshotBytes, 48); assert.equal(ids, before);
    r.owner.release(lease); r.owner.dispose();
  }
});
test('cubemap budget includes every face/mip and failed capture leaves no listeners', () => {
  const texture = new CubeTexture(faces(2)); texture.mipmaps = [{image: faces()}];
  const r = rig({maxSnapshotBytes: 119});
  assert.throws(() => r.owner.captureBinding(bind(texture)), {code: 'THREE_COMPUTE_TEXTURE_BUDGET'});
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(r.uploads.length, 0);
  assert.equal(texture._listeners, undefined); r.owner.dispose();
});
test('shared faces do not multiply captured bytes', () => {
  const face = new DataTexture(6), texture = new CubeTexture(Array(6).fill(face));
  const r = rig({maxSnapshotBytes: 4}), lease = r.owner.captureBinding(bind(texture));
  assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  assert.deepEqual(r.owner.resolve(lease).base.map(data => data[0]), [6, 6, 6, 6, 6, 6]);
  r.owner.release(lease); r.owner.dispose();
});
test('textures sharing a Source reuse captured bytes/native residency and acknowledge only one upload', () => {
  const r = rig({maxSnapshotBytes: 4}), first = new DataTexture(3), second = new DataTexture(9);
  second.source = first.source;
  let firstUpdates = 0, secondUpdates = 0;
  first.onUpdate = () => firstUpdates++; second.onUpdate = () => secondUpdates++;
  const a = r.owner.captureBinding(bind(first)), b = r.owner.captureBinding(bind(second));
  assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  assert.equal(r.owner.resolve(a), r.owner.resolve(b));
  assert.equal(r.uploads.length, 1); assert.equal(firstUpdates, 1); assert.equal(secondUpdates, 0);
  first.dispose(); assert.equal(r.resources.size, 1); assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  assert.equal(r.owner.resolve(b).base[0][0], 3);
  second.dispose(); assert.equal(r.resources.size, 0); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
  r.owner.release(a); r.owner.release(b); r.owner.dispose();
});
test('shared-source queued versions remain isolated and retire only after the last alias', () => {
  const r = rig({maxSnapshotBytes: 8}), first = new DataTexture(3), second = new DataTexture(1); second.source = first.source;
  const a = r.owner.captureBinding(bind(first)), b = r.owner.captureBinding(bind(second));
  first.image.data.fill(7); first.touch(); second.version++;
  const c = r.owner.captureBinding(bind(first)), d = r.owner.captureBinding(bind(second));
  first.image.data.fill(99);
  assert.equal(r.owner.resolve(a).base[0][0], 3); assert.equal(r.owner.resolve(b).base[0][0], 3);
  assert.equal(r.owner.resolve(c).base[0][0], 7); assert.equal(r.owner.resolve(d).base[0][0], 7);
  assert.equal(r.uploads.length, 2); assert.equal(r.owner.diagnostics.snapshotBytes, 8);
  r.owner.release(a); assert.equal(r.owner.diagnostics.snapshotBytes, 8);
  r.owner.release(b); assert.equal(r.owner.diagnostics.snapshotBytes, 4); assert.equal(r.resources.size, 1);
  r.owner.release(c); r.owner.release(d); r.owner.dispose();
});
test('decoded-image and malformed cubes remain explicit errors', () => {
  const r = rig();
  const decoded = new CubeTexture(Array.from({length: 6}, () => ({width: 1, height: 1})));
  assert.throws(() => r.owner.captureBinding(bind(decoded)), {code: 'THREE_COMPUTE_TEXTURE_SOURCE'});
  assert.throws(() => r.owner.captureBinding(bind(new CubeTexture(faces().slice(1)))), {code: 'THREE_COMPUTE_TEXTURE_SOURCE'});
  assert.equal(r.uploads.length, 0); r.owner.dispose();
});
test('disposed queued leases release pixel descriptors before the FIFO consumes them', () => {
  const r = rig({maxSnapshotBytes: 4}), texture = new DataTexture(1), pending = [];
  for (let i = 0; i < 270; i++) {
    const lease = r.owner.captureBinding(bind(texture)); pending.push(lease);
    texture.dispose(); texture.touch();
    assert.equal(lease.record.texture, null); assert.equal(lease.record.pixels, null);
    assert.equal(r.owner.diagnostics.snapshotBytes, 0);
  }
  for (const lease of pending) {
    assert.throws(() => r.owner.resolve(lease), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
    r.owner.release(lease);
  }
  r.owner.dispose();
});
