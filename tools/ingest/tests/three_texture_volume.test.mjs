import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuThreeTextures, inspectThreeTexture} from '../three_textures.mjs';
import {T, texture, device, deferred} from './volume_texture_test_fixture.mjs';
const pool = (d, options = {}) => createGpuThreeTextures(d, {three: T, ...options});
const bytesOf = t => new Uint8Array(t.image.data.buffer, t.image.data.byteOffset, t.image.data.byteLength);
const mips = t => { t.generateMipmaps = true; t.minFilter = T.LinearMipmapLinearFilter; return t; };

for (const kind of ['array', 'volume']) for (const channels of [1, 2, 4]) for (const bytes of [1, 2, 4])
  test(`${kind} R${channels} storage preserves ${bytes * 8}-bit source bytes and native dimensions`, async () => {
    const d = device(), p = pool(d), t = texture(kind, 3, 2, 4, channels, bytes), input = bytesOf(t).slice();
    const image = t.image, data = image.data;
    p.prepare([t]); await p.whenIdle();
    const b = p.binding(t), native = p.nativeTexture(t);
    assert.equal(b.viewDimension, kind === 'array' ? '2d-array' : '3d');
    assert.equal(b.view.dimension, b.viewDimension);
    assert.equal(native.dimension, kind === 'array' ? '2d' : '3d');
    assert.deepEqual(native.levels[0], input);
    assert.deepEqual(d.writes[0].layout, {bytesPerRow: 3 * channels * bytes, rowsPerImage: 2});
    assert.deepEqual(d.writes[0].size, [3, 2, 4]);
    assert.equal(b.sampleType, bytes === 4 ? 'unfilterable-float' : 'float');
    assert.equal(b.filtering, bytes !== 4);
    assert.equal(p.diagnostics.textureBytes, input.byteLength);
    assert.equal(t.image, image); assert.equal(t.image.data, data);
    assert.equal(d.scopes.length, 0);
    p.dispose(); assert.equal(native.destroyed, true);
  });

for (const kind of ['array', 'volume']) test(`${kind} uploads padded rows across slices from a nonzero byte offset`, () => {
  const d = device(), p = pool(d), t = texture(kind, 3, 2, 2);
  const buffer = Uint8Array.from([99, 98, 1, 2, 3, 55, 4, 5, 6, 54, 7, 8, 9, 53, 10, 11, 12, 97]);
  t.image.data = buffer.subarray(2, 17); t.unpackAlignment = 4;
  p.prepare([t]); assert.deepEqual([...p.nativeTexture(t).levels[0]], [1,2,3,4,5,6,7,8,9,10,11,12]);
  assert.equal(d.writes[0].layout.bytesPerRow, 4); assert.equal(d.writes[0].layout.rowsPerImage, 2);
  assert.equal(buffer[0], 99); assert.equal(buffer.at(-1), 97); p.dispose();
});

for (const bytes of [1, 2, 4]) test(`array layerUpdates change only requested layers and preserve ${bytes * 8}-bit patterns`, () => {
  const d = device(), p = pool(d), t = texture('array', 3, 2, 4, 2, bytes);
  p.prepare([t]); const native = p.nativeTexture(t), before = native.levels[0].slice(), slice = 3 * 2 * 2 * bytes;
  bytesOf(t).fill(0xa5); t.addLayerUpdate(3); t.addLayerUpdate(1); t.addLayerUpdate(3);
  p.update([t]); assert.equal(d.writes.length, 1, 'unrequested source edits stay GPU-stale');
  assert.equal(t.layerUpdates.size, 2);
  t.needsUpdate = true; p.update([t]);
  assert.deepEqual(d.writes.slice(1).map(w => w.dest.origin), [[0,0,3], [0,0,1]]);
  for (let layer = 0; layer < 4; layer++) assert.deepEqual(native.levels[0].slice(layer * slice, (layer + 1) * slice),
    layer % 2 ? new Uint8Array(slice).fill(0xa5) : before.slice(layer * slice, (layer + 1) * slice));
  assert.equal(t.layerUpdates.size, 0); assert.equal(p.binding(t).version, t.version); p.dispose();
});

test('initial partial array upload leaves unselected layers at native zero', () => {
  const d = device(), p = pool(d), t = texture('array', 2, 2, 3); t.addLayerUpdate(1);
  p.prepare([t]); const expected = new Uint8Array(12); expected.set(bytesOf(t).subarray(4,8),4);
  assert.deepEqual(p.nativeTexture(t).levels[0], expected); assert.equal(d.writes.length, 1); p.dispose();
});

test('array mipmaps halve width/height, preserve every layer and isolate source/destination subresources', () => {
  const d = device(), p = pool(d), t = mips(texture('array', 5, 3, 3, 4)); t.colorSpace = T.SRGBColorSpace;
  p.prepare([t]); assert.equal(p.diagnostics.textureBytes, (15 + 2 + 1) * 3 * 4);
  assert.equal(d.passes.length, 6); assert.equal(d.pipelines.length, 1);
  for (const [i, pass] of d.passes.entries()) {
    const out = pass.desc.colorAttachments[0].view, input = pass.group.entries[0].resource;
    assert.equal(out.baseMipLevel, 1 + i % 2); assert.equal(input.baseMipLevel, out.baseMipLevel - 1);
    assert.equal(input.baseArrayLayer, Math.floor(i / 2)); assert.equal(out.baseArrayLayer, Math.floor(i / 2));
    assert.equal(input.dimension, '2d'); assert.equal(out.arrayLayerCount, 1);
    assert.equal(pass.pipeline.fragment.targets[0].format, 'rgba8unorm-srgb'); assert.ok(pass.ended);
  }
  t.addLayerUpdate(2); t.needsUpdate = true; p.update([t]);
  assert.equal(d.passes.length, 12); assert.equal(d.pipelines.length, 1); assert.equal(d.submissions, 2); p.dispose();
});

for (const extent of [[5,3,7], [1,1,9], [9,1,1]]) test(`3D mip pyramid ${extent.join('x')} shrinks depth and records every destination slice`, () => {
  const d = device(), p = pool(d), t = mips(texture('volume', ...extent)), shape = inspectThreeTexture(t, T);
  p.prepare([t]); const native = p.nativeTexture(t);
  let cost = 0, n = 0;
  for (let level = 0; level < shape.levels; level++) {
    const [w,h,z] = extent.map(x => Math.max(1, x >> level)); cost += w*h*z;
    if (!level) continue;
    for (let slice = 0; slice < z; slice++) {
      const pass = d.passes[n++], out = pass.desc.colorAttachments[0], input = pass.group.entries[0].resource;
      assert.equal(out.depthSlice, slice); assert.equal(out.view.baseMipLevel, level);
      assert.equal(out.view.dimension, '3d'); assert.equal(input.baseMipLevel, level-1);
      assert.equal(input.dimension, '3d'); assert.deepEqual(pass.drawArgs, [3,1,0,slice]);
      assert.equal(out.view.texture, native); assert.equal(input.texture, native);
    }
  }
  assert.equal(d.passes.length, n); assert.equal(p.diagnostics.textureBytes, cost);
  assert.match(d.pipelines[0].fragment.module.code, /texture_3d<f32>/);
  assert.match(d.pipelines[0].fragment.module.code, /f32\(v.slice\)\+0.5/); p.dispose();
});

test('2D, array and cube share 2D mip pipelines; 3D uses a distinct shader/profile', () => {
  const d = device(), p = pool(d), a = mips(texture('array',4,4,2,4)), v = mips(texture('volume',4,4,2,4));
  const t = new T.DataTexture(new Uint8Array(4*4*4),4,4); t.needsUpdate = true; mips(t);
  const c = new T.CubeTexture(Array.from({length:6}, () => t)); c.needsUpdate = true;
  p.prepare([a,v,t,c]); assert.equal(d.pipelines.length, 2);
  assert.equal(p.binding(t).view.dimension, '2d'); assert.equal(p.binding(c).view.dimension, 'cube'); p.dispose();
});

test('shared source clones share native storage; 2D, array and 3D domains never alias', () => {
  const d = device(), p = pool(d), a = texture('array',2,2,2), b = a.clone(), v = texture('volume',2,2,2);
  v.source = a.source;
  const t = new T.DataTexture(a.image.data,2,2); t.format = T.RedFormat; t.source = a.source; t.needsUpdate = true;
  p.prepare([a,b,v,t]); assert.equal(d.textures.length, 3); assert.equal(d.writes.length, 3);
  assert.equal(p.binding(a).view, p.binding(b).view); assert.notEqual(p.binding(a).view, p.binding(v).view);
  assert.notEqual(p.binding(a).view, p.binding(t).view);
  a.dispose(); assert.equal(d.textures[0].destroyed, false); b.dispose(); assert.equal(d.textures[0].destroyed, true);
  p.dispose(); assert.ok(d.textures.every(t => t.destroyed));
});

test('shared clones acknowledge one source upload with layer selection, not one upload per texture', () => {
  const d = device(), p = pool(d), a = texture('array'), b = a.clone(); p.prepare([a,b]);
  bytesOf(a).fill(42); a.addLayerUpdate(2); a.needsUpdate = true; b.needsUpdate = true;
  p.update([a,b]); assert.equal(d.writes.length, 2); assert.equal(p.binding(a).view, p.binding(b).view);
  assert.equal(p.binding(b).sourceVersion, a.source.version); p.dispose();
});

test('wrapR/filter/depth edits require preparation, preserve budgets, and do not reuse stale sampler/storage', () => {
  const d = device(), p = pool(d), t = texture('volume'); p.prepare([t]); let old = p.nativeTexture(t);
  t.wrapR = T.RepeatWrapping; assert.throws(() => p.update([t]), {code:'THREE_TEXTURE_PREPARE'});
  p.prepare([t]); assert.equal(p.binding(t).sampler.addressModeW, 'repeat'); assert.ok(old.destroyed);
  old = p.nativeTexture(t); t.image.depth = 2; t.needsUpdate = true;
  assert.throws(() => p.binding(t), {code:'THREE_TEXTURE_PREPARE'}); p.prepare([t]); assert.ok(old.destroyed);
  assert.deepEqual(p.nativeTexture(t).desc.size,[3,2,2]); p.dispose();
});

test('independent 3D dimension and array-layer limits use the native axis, not a 2D/cube proxy', () => {
  const d = device({limits:{maxTextureDimension2D:4,maxTextureDimension3D:8,maxTextureArrayLayers:2}}), p = pool(d);
  p.prepare([texture('volume',5,1,8)]);
  assert.throws(() => p.prepare([texture('array',1,1,3)]), {code:'THREE_TEXTURE_VALUE'});
  assert.throws(() => p.prepare([texture('volume',1,9,1)]), {code:'THREE_TEXTURE_VALUE'});
  assert.throws(() => p.prepare([texture('volume',1,1,9)]), {code:'THREE_TEXTURE_VALUE'});
  assert.throws(() => p.prepare([texture('array',5,1,1)]), {code:'THREE_TEXTURE_VALUE'});
  assert.equal(d.textures.length,1); p.dispose();
});

test('voxel and aggregate mip budgets admit whole sets before any allocation', () => {
  const d = device(), p = pool(d, {maxPixels:16,maxTextureBytes:31}), a = texture('array',2,2,4), v = texture('volume',2,2,4);
  assert.throws(() => p.prepare([texture('volume',2,2,5)]), {code:'THREE_TEXTURE_LIMIT'});
  assert.throws(() => p.prepare([a,v]), {code:'THREE_TEXTURE_LIMIT'}); assert.equal(d.textures.length,0);
  const m = mips(texture('array',2,2,4)); p.prepare([m]); assert.equal(p.diagnostics.textureBytes,20);
  assert.throws(() => p.prepare([a]), {code:'THREE_TEXTURE_LIMIT'}); assert.equal(d.textures.length,1); p.dispose();
});

for (const edit of [t => t.image.data = t.image.data.subarray(1), t => t.image.data = new Uint16Array(24),
  t => t.image.data = new Uint8Array(new SharedArrayBuffer(24)), t => t.image.data = new Uint8Array(new ArrayBuffer(24,{maxByteLength:48})),
  t => { structuredClone(t.image.data.buffer,{transfer:[t.image.data.buffer]}); },
  t => t.flipY = true, t => t.premultiplyAlpha = true, t => t.mipmaps.push(t.image),
  t => t.updateRanges.push({start:0,count:1}), t => t.layerUpdates.add(4), t => t.layerUpdates.add(-1),
  t => t.layerUpdates.add(0.5), t => t.layerUpdates = [], t => t.wrapR = -1])
  test(`invalid volume storage/profile is rejected before effects: ${edit.toString()}`, () => {
    const d = device(), p = pool(d), t = texture('array'), a = texture('volume'); edit(t);
    assert.throws(() => p.prepare([a,t])); assert.equal(d.textures.length,0); assert.equal(d.writes.length,0); p.dispose();
  });

test('padded partial-layer offsets are refused, not silently reinterpreted', () => {
  const d = device(), p = pool(d), t = texture('array',3,2,2); t.unpackAlignment = 4;
  t.image.data = new Uint8Array(15); t.addLayerUpdate(1);
  assert.throws(() => p.prepare([t]), {code:'THREE_TEXTURE_RANGE'}); assert.equal(d.textures.length,0); p.dispose();
});

test('source-class authenticity rejects forged or conflicting family flags', () => {
  for (const t of [Object.assign(texture('volume'),{isDataArrayTexture:true}),
    Object.assign(new T.Texture(texture().image),{isData3DTexture:true}),
    Object.assign(texture('array'),{isCubeTexture:true})]) assert.throws(() => inspectThreeTexture(t,T), {code:'THREE_TEXTURE_SOURCE'});
});

test('float32 generated/linear volume sampling needs a negotiated feature, never downgraded precision', () => {
  const d = device(), p = pool(d), t = mips(texture('volume',3,2,4,4,4));
  assert.throws(() => p.prepare([t]), {code:'THREE_TEXTURE_MIPS'}); assert.equal(d.textures.length,0); p.dispose();
  const enabled = device({features:['float32-filterable']}), q = pool(enabled); q.prepare([t]);
  assert.equal(q.nativeTexture(t).desc.format,'rgba32float'); assert.equal(q.binding(t).sampleType,'float');
  assert.equal(enabled.passes.length,3); q.dispose();
});

test('onUpdate sees cleared layers and acknowledged source; callback failures do not replay writes', () => {
  const d = device(), p = pool(d), t = texture('array'), sentinel = new Error('callback'); let calls = 0;
  t.addLayerUpdate(1); t.onUpdate = () => {
    calls++; assert.equal(t.layerUpdates.size,0); assert.equal(p.binding(t).sourceVersion,t.source.version); throw sentinel;
  };
  assert.throws(() => p.prepare([t]), e => e === sentinel); assert.equal(p.failed,false);
  p.update([t]); assert.equal(calls,1); assert.equal(d.writes.length,1); p.dispose();
});

test('an invalid layer in a later texture prevents every upload and preserves pending layers', () => {
  const d = device(), p = pool(d), a = texture('array'), b = texture('array'); p.prepare([a,b]);
  a.addLayerUpdate(1); a.needsUpdate = true; b.addLayerUpdate(100); b.needsUpdate = true;
  assert.throws(() => p.update([a,b])); assert.equal(d.writes.length,2); assert.equal(a.layerUpdates.size,1);
  b.clearLayerUpdates(); p.update([a,b]); assert.equal(d.writes.length,4); p.dispose();
});

test('retirement, retain and disposal remove source listeners and release every volume', () => {
  const d = device(), p = pool(d), a = texture('array'), b = texture('volume'); p.prepare([a,b]);
  p.retain([b]); assert.equal(a.listeners.get('dispose').size,0); assert.equal(d.textures[0].destroyed,true);
  p.dispose(); assert.equal(b.listeners.get('dispose').size,0); assert.equal(p.diagnostics.textureBytes,0);
});

for (const failure of ['write','validation','lost']) test(`${failure} failure releases layered storage and terminates ownership`, async () => {
  const d = device(), p = pool(d), t = texture('volume'); p.prepare([t]); await p.whenIdle();
  if (failure === 'write') { d.failWrite = true; t.needsUpdate = true; assert.throws(() => p.update([t]), /injected/); }
  if (failure === 'validation') { d.scopeError = {message:'injected validation'}; t.needsUpdate = true; p.update([t]); }
  if (failure === 'lost') d.loss.resolve({message:'injected loss'});
  await assert.rejects(p.whenIdle()); assert.equal(p.failed,true); assert.equal(p.diagnostics.textureBytes,0);
  assert.ok(d.textures.every(t => t.destroyed)); assert.equal(d.scopes.length,0);
});

test('whenIdle includes completion; disposal interrupts an unfinished native fence', async () => {
  const fence = deferred(), d = device({fence:fence.promise}), p = pool(d); p.prepare([texture('volume')]);
  let done = false; const idle = p.whenIdle().then(() => { done = true; });
  await Promise.resolve(); assert.equal(done,false); p.dispose(); await assert.rejects(idle,{code:'THREE_TEXTURE_DISPOSED'}); fence.resolve();
});

test('ordinary 2D binding shape, byte ranges and row flips remain unchanged', () => {
  const d = device(), p = pool(d), t = new T.DataTexture(Uint8Array.from({length:16},(_,i)=>i),2,2);
  t.needsUpdate = true; p.prepare([t]);
  assert.deepEqual(Object.keys(p.binding(t)),['view','sampler','version','sourceVersion','sampleType','filtering']);
  t.image.data.fill(99); t.updateRanges.push({start:4,count:4}); t.needsUpdate = true; p.update([t]);
  assert.deepEqual([...p.nativeTexture(t).levels[0]],[0,1,2,3,99,99,99,99,8,9,10,11,12,13,14,15]);
  const flipped = new T.DataTexture(new Uint8Array([1,7,7,7,2]),1,2);
  flipped.format = T.RedFormat; flipped.flipY = true; flipped.unpackAlignment = 4; flipped.needsUpdate = true;
  p.prepare([flipped]); assert.deepEqual([...p.nativeTexture(flipped).levels[0]],[2,1]); p.dispose();
});

test('authored 2D mip levels remain verbatim and disable automatic generation', () => {
  const d = device(), p = pool(d), t = new T.DataTexture(new Uint8Array(32).fill(1),4,2);
  t.mipmaps = [t.image,{width:2,height:1,data:new Uint8Array(8).fill(2)},{width:1,height:1,data:new Uint8Array(4).fill(3)}];
  t.minFilter = T.LinearMipmapLinearFilter; t.generateMipmaps = true; t.needsUpdate = true;
  p.prepare([t]); assert.equal(t.generateMipmaps,false); assert.equal(d.passes.length,0);
  assert.deepEqual(p.nativeTexture(t).levels.map(x=>[x.length,x[0]]),[[32,1],[8,2],[4,3]]); p.dispose();
});

test('cube mip generation retains the established layer-major submission order', () => {
  const d = device(), p = pool(d), face = new T.DataTexture(new Uint8Array(64),4,4);
  const t = new T.CubeTexture(Array.from({length:6},()=>face)); t.needsUpdate = true;
  p.prepare([t]); assert.equal(d.passes.length,12);
  for (const [i,pass] of d.passes.entries()) {
    const source = pass.group.entries[0].resource, output = pass.desc.colorAttachments[0].view;
    assert.equal(source.baseArrayLayer,Math.floor(i/2)); assert.equal(output.baseArrayLayer,source.baseArrayLayer);
    assert.equal(source.baseMipLevel,i%2); assert.equal(output.baseMipLevel,source.baseMipLevel+1);
  }
  p.dispose();
});
