/** Compressed copies through the actual native renderer and shared texture owner.
 * The device validates block geometry and executes byte copies, not compressed
 * decoding or WGSL. Source comparisons check the pinned backend's commands. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T);
const formats = new Map([
  ['bc1-rgba-unorm', [4, 4, 8]], ['bc3-rgba-unorm', [4, 4, 16]],
  ['bc6h-rgb-ufloat', [4, 4, 16]], ['etc2-rgba8unorm', [4, 4, 16]],
  ['astc-5x4-unorm', [5, 4, 16]],
]);
const origin = value => Array.isArray(value) ? [...value] : [value?.x ?? 0, value?.y ?? 0, value?.z ?? 0];
const extent = value => Array.isArray(value) ? [...value] : [value.width, value.height, value.depthOrArrayLayers ?? 1];
const raw = data => new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength);
const box = (x, y, width, height, z = 0, depth = 1) => new T.Box3(
  new T.Vector3(x, y, z), new T.Vector3(x + width, y + height, z + depth));
const point = (x = 0, y = 0, z = 0) => new T.Vector3(x, y, z);

async function fixture(t, {features = ['core-features-and-limits', 'texture-compression-bc',
  'texture-compression-etc2', 'texture-compression-astc']} = {}) {
  const device = geometryDevice(), textures = [], uploads = [], copies = [];
  device.features = new Set(features);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const size = extent(descriptor.size), [bw, bh, bytes] = formats.get(descriptor.format.replace(/-srgb$/, '')) ?? [1, 1, 4];
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: size[0], height: size[1], depthOrArrayLayers: size[2], bw, bh, blockBytes: bytes, destroyed: 0,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed++; }};
    texture.levels = Array.from({length: descriptor.mipLevelCount ?? 1}, (_, level) => {
      const width = Math.ceil(Math.max(1, size[0] >> level) / bw), height = Math.ceil(Math.max(1, size[1] >> level) / bh);
      const depth = texture.dimension === '3d' ? Math.max(1, size[2] >> level) : size[2];
      return {width, height, depth, data: new Uint8Array(width * height * depth * bytes)};
    });
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  function selected(info, size) {
    const texture = info.texture, level = texture.levels[info.mipLevel ?? 0], start = origin(info.origin);
    assert.ok(!texture.destroyed && level, 'copy uses live allocated storage');
    assert.ok([...start, ...size].every(n => Number.isSafeInteger(n) && n >= 0));
    assert.equal(start[0] % texture.bw, 0); assert.equal(start[1] % texture.bh, 0);
    assert.equal(size[0] % texture.bw, 0); assert.equal(size[1] % texture.bh, 0);
    const blockStart = [start[0] / texture.bw, start[1] / texture.bh, start[2]];
    const blockSize = [size[0] / texture.bw, size[1] / texture.bh, size[2]];
    assert.ok(blockStart.every((n, i) => n + blockSize[i] <= [level.width, level.height, level.depth][i]));
    const offset = (z, y) => (((blockStart[2] + z) * level.height + blockStart[1] + y) * level.width + blockStart[0]) * texture.blockBytes;
    return {level, blockSize, offset};
  }
  device.queue.writeTexture = (destination, data, layout, copySize) => {
    const size = extent(copySize), {level, blockSize, offset} = selected(destination, size), bytes = raw(data);
    const row = blockSize[0] * destination.texture.blockBytes, pitch = layout.bytesPerRow ?? row;
    const rows = layout.rowsPerImage ?? blockSize[1];
    for (let z = 0; z < blockSize[2]; z++) for (let y = 0; y < blockSize[1]; y++) {
      const from = (layout.offset ?? 0) + (z * rows + y) * pitch;
      assert.ok(from + row <= bytes.length); level.data.set(bytes.subarray(from, from + row), offset(z, y));
    }
    uploads.push({texture: destination.texture, mipLevel: destination.mipLevel ?? 0,
      origin: origin(destination.origin), size, bytes: bytes.slice()});
  };
  const makeEncoder = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = (descriptor = {}) => {
    const encoder = makeEncoder(), commands = [];
    return {...encoder, copyTextureToTexture(from, to, copySize) {
      const size = extent(copySize);
      assert.ok(from.texture.usage & 1); assert.ok(to.texture.usage & 2);
      assert.equal(from.texture.format, to.texture.format);
      selected(from, size); selected(to, size);
      commands.push({draws: [], copy: {label: descriptor.label,
        from: {...from, mipLevel: from.mipLevel ?? 0, origin: origin(from.origin)},
        to: {...to, mipLevel: to.mipLevel ?? 0, origin: origin(to.origin)}, size}});
    }, finish() { return [...encoder.finish(), ...commands]; }};
  };
  device.queue.submit = commands => {
    for (const list of commands) for (const command of list) if (command.copy) {
      const {from, to, size} = command.copy, a = selected(from, size), b = selected(to, size);
      const row = a.blockSize[0] * from.texture.blockBytes;
      for (let z = 0; z < a.blockSize[2]; z++) for (let y = 0; y < a.blockSize[1]; y++)
        b.level.data.set(a.level.data.slice(a.offset(z, y), a.offset(z, y) + row), b.offset(z, y));
      copies.push(command.copy);
    }
    submit(commands);
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({label: 'canvas', size: [32, 32, 1],
      format: configured.format, usage: configured.usage})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  object.frustumCulled = false; scene.add(object);
  t.after(() => { renderer.dispose(); material.dispose(); object.geometry.dispose(); });
  await renderer.init();
  const inputs = renderer._getSampledTextureInputs();
  const native = texture => {
    const lease = inputs.captureBinding({texture, isSampledTexture: true});
    try { return inputs.withTexture(lease, value => value); } finally { inputs.release(lease); }
  };
  return {renderer, device, textures, uploads, copies, inputs, native, material, object, scene, camera,
    initialTextures: textures.length};
}
function compressed(t, {width = 12, height = 8, layers = 1, levels = 4, format = T.RGBA_S3TC_DXT1_Format,
  type = T.UnsignedByteType, bw = 4, bh = 4, blockBytes = 8, value = 7, View = Uint8Array} = {}) {
  const mips = Array.from({length: Math.min(levels, 1 + Math.floor(Math.log2(Math.max(width, height))))}, (_, level) => {
    const w = Math.max(1, width >> level), h = Math.max(1, height >> level), slice = Math.ceil(w / bw) * Math.ceil(h / bh) * blockBytes;
    const bytes = new Uint8Array(slice * layers);
    for (let layer = 0; layer < layers; layer++) bytes.fill(value + level + layer * 10, layer * slice, (layer + 1) * slice);
    return {width: w, height: h, data: new View(bytes.buffer)};
  });
  const texture = layers > 1 ? new T.CompressedArrayTexture(mips, width, height, layers, format, type)
    : new T.CompressedTexture(mips, width, height, format, type);
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
function blockQueue(renderer) {
  let release;
  const promise = new Promise(resolve => { release = resolve; });
  renderer._enqueue({transfer: true, awaitSubmission: true, issue: () => promise, resolve() {}, reject() {}});
  return release;
}
const publicCopies = f => f.copies.filter(copy => copy.label === 'f3d-sampled-texture-copy');
const commandShape = copy => ({sourceLevel: copy.from.mipLevel, destinationLevel: copy.to.mipLevel,
  sourceOrigin: origin(copy.from.origin), destinationOrigin: origin(copy.to.origin), size: extent(copy.size)});
function sourceCommand(source, destination, region, position, sourceLevel, destinationLevel) {
  let recorded;
  const backend = {get: texture => ({texture}), textureUtils: {generateMipmaps() { assert.fail('Authored compressed mips must stay unchanged'); }},
    device: {createCommandEncoder() { return {copyTextureToTexture(from, to, size) {
      recorded = commandShape({from, to, size});
    }, finish() { return []; }}; }, queue: {submit() {}}}};
  T.WebGPUBackend.prototype.copyTextureToTexture.call(backend, source, destination, region, position, sourceLevel, destinationLevel);
  return recorded;
}

for (const profile of [
  {name: 'BC1', native: 'bc1-rgba-unorm'},
  {name: 'BC6H HDR block views', native: 'bc6h-rgb-ufloat', format: T.RGB_BPTC_UNSIGNED_Format,
    type: T.HalfFloatType, blockBytes: 16, View: Uint16Array},
  {name: 'ETC2', native: 'etc2-rgba8unorm', format: T.RGBA_ETC2_EAC_Format, blockBytes: 16},
  {name: 'non-square ASTC', native: 'astc-5x4-unorm', format: T.RGBA_ASTC_5x4_Format, bw: 5, blockBytes: 16},
  {name: 'RGBA8 loader fallback', native: 'rgba8unorm', format: T.RGBAFormat, bw: 1, bh: 1, blockBytes: 4, features: []},
]) test(`native ${profile.name} copies preserve blocks, aliases, authored mips and sampled residency`, async t => {
  const f = await fixture(t, profile), bw = profile.bw ?? 4, bh = profile.bh ?? 4, blockBytes = profile.blockBytes ?? 8;
  const options = {...profile, width: bw * 3, height: bh * 2};
  const source = compressed(t, {...options, value: 51}), destination = compressed(t, options), alias = destination.clone();
  t.after(() => alias.dispose());
  const original = f.native(destination); assert.equal(f.native(alias), original);
  const before = original.levels.map(level => level.data.slice()), cpu = destination.mipmaps.map(mip => raw(mip.data).slice());
  const region = box(0, 0, bw, bh), position = point(bw, bh);
  const expectedCommand = sourceCommand(source, destination, region, position, 0, 0);
  f.renderer.copyTextureToTexture(source, destination, region, position);
  const native = f.native(destination), expected = before[0].slice();
  expected.set(raw(source.mipmaps[0].data).subarray(0, blockBytes), (original.levels[0].width + 1) * blockBytes);
  assert.notEqual(native, original); assert.equal(native.format, profile.native);
  assert.deepEqual(native.levels[0].data, expected);
  assert.deepEqual(native.levels.slice(1).map(level => level.data), before.slice(1));
  assert.deepEqual(original.levels.map(level => level.data), before); assert.equal(f.native(alias), original);
  assert.deepEqual(destination.mipmaps.map(mip => raw(mip.data)), cpu);
  assert.deepEqual(commandShape(publicCopies(f)[0]), expectedCommand);
  const seeds = f.copies.filter(copy => copy.label?.endsWith('/preserve-copy'));
  assert.equal(seeds.length, destination.mipmaps.length, 'detach seeds every authored mip once');
  assert.deepEqual(seeds.map(copy => copy.size), native.levels.map(level => [level.width * bw, level.height * bh, 1]));
  const allocations = f.textures.length, writes = f.uploads.length;
  f.renderer.copyTextureToTexture(source, destination, region, point(0, 0));
  assert.equal(f.native(destination), native); assert.equal(f.textures.length, allocations); assert.equal(f.uploads.length, writes);
  f.material.map = destination; await f.renderer.renderAsync(f.scene, f.camera);
  const draw = f.device.snapshots.flat().find(draw => draw.pipeline.vertex.entryPoint === 'main');
  const view = [...draw.groups.values()].flatMap(({group}) => group.entries).find(entry => entry.resource?.texture).resource;
  assert.equal(view.texture, native, 'graphics samples the allocation patched by the public copy');
  assert.ok(f.device.buffers.every(buffer => !(buffer.usage & 1)), 'copying allocates no readback buffers');
  await f.renderer.waitForGPU();
});

test('compressed array copies preserve queued source versions and partial host layers across GPU seeds', async t => {
  const f = await fixture(t), source = compressed(t, {width: 4, height: 4, value: 31}),
    destination = compressed(t, {width: 8, height: 8, layers: 3});
  const original = f.native(destination), release = blockQueue(f.renderer), position = point();
  const before = original.levels.map(level => level.data.slice());
  f.renderer.copyTextureToTexture(source, destination, null, position);
  destination.mipmaps.forEach(mip => raw(mip.data).fill(99)); destination.addLayerUpdate(2); destination.needsUpdate = true;
  raw(source.mipmaps[0].data).fill(41); source.needsUpdate = true; position.set(4, 4, 1);
  f.renderer.copyTextureToTexture(source, destination, null, position);
  source.mipmaps.forEach(mip => raw(mip.data).fill(199)); destination.mipmaps.forEach(mip => raw(mip.data).fill(211)); position.z = 99;
  assert.equal(publicCopies(f).length, 0); release(); await f.renderer.waitForGPU();
  const commands = publicCopies(f), native = f.native(destination), expected = before.map(data => data.slice());
  expected[0].fill(31, 0, 8); expected[0].fill(41, 56, 64);
  for (const bytes of expected) bytes.fill(99, bytes.length * 2 / 3);
  assert.deepEqual(native.levels.map(level => level.data), expected);
  assert.equal(commands.length, 2); assert.notEqual(commands[0].from.texture, commands[1].from.texture);
  assert.notEqual(commands[0].to.texture, commands[1].to.texture);
  assert.deepEqual([...commands[0].from.texture.levels[0].data], Array(8).fill(31));
  assert.deepEqual([...commands[1].from.texture.levels[0].data], Array(8).fill(41));
  assert.deepEqual(commands[0].to.texture.levels[0].data.subarray(64), before[0].subarray(64));
  const updates = f.uploads.filter(upload => upload.texture === native);
  assert.equal(updates.length, destination.mipmaps.length); assert.ok(updates.every(upload => upload.origin[2] === 2 && upload.size[2] === 1));
  assert.equal(destination.layerUpdates.size, 0); assert.ok(destination.mipmaps.every(mip => raw(mip.data).every(byte => byte === 211)));
  f.renderer.dispose(); assert.ok(f.textures.every(texture => texture.destroyed === 1));
});

test('compressed tiny mip copies use physical blocks and preserve the pinned default extent', async t => {
  const f = await fixture(t), source = compressed(t, {width: 4, height: 4, value: 31}), destination = compressed(t, {width: 4, height: 4});
  f.renderer.copyTextureToTexture(source, destination, null, null, 2, 2);
  const command = publicCopies(f)[0], expected = sourceCommand(source, destination, null, null, 2, 2);
  assert.deepEqual(commandShape(command), expected); assert.deepEqual(command.size, [4, 4, 1]);
  assert.deepEqual(f.native(destination).levels[2].data, raw(source.mipmaps[2].data));
  const bigger = compressed(t, {width: 12, height: 8, value: 61});
  f.renderer.copyTextureToTexture(bigger, destination, box(0, 0, 4, 4), null, 3, 2);
  assert.deepEqual(commandShape(publicCopies(f)[1]), sourceCommand(bigger, destination, box(0, 0, 4, 4), null, 3, 2));
  assert.deepEqual(f.native(destination).levels[2].data, raw(bigger.mipmaps[3].data));
  assert.throws(() => f.renderer.copyTextureToTexture(bigger, destination, null, null, 3, 2), {code: 'THREE_STORAGE_COPY_BOUNDS'});
  assert.throws(() => f.renderer.copyTextureToTexture(source, destination, box(0, 0, 1, 1), null, 2, 2), {code: 'THREE_STORAGE_COPY_BLOCK'});
  assert.equal(publicCopies(f).length, 2); await f.renderer.waitForGPU();
});

test('compressed arrays retain source one-layer defaults and allow only disjoint self-copy subresources', async t => {
  const f = await fixture(t), source = compressed(t, {width: 8, height: 8, layers: 3, value: 41}),
    destination = compressed(t, {width: 8, height: 8, layers: 3});
  f.renderer.copyTextureToTexture(source, destination, null, point(0, 0, 1));
  const native = f.native(destination), before = native.levels.map(level => level.data.slice());
  assert.deepEqual(commandShape(publicCopies(f)[0]), sourceCommand(source, destination, null, point(0, 0, 1), 0, 0));
  assert.deepEqual(native.levels[0].data.subarray(32, 64), raw(source.mipmaps[0].data).subarray(0, 32));
  f.renderer.copyTextureToTexture(destination, destination, box(0, 0, 8, 8, 1), point(0, 0, 2));
  assert.equal(publicCopies(f)[1].from.texture, publicCopies(f)[1].to.texture);
  assert.deepEqual(native.levels[0].data.subarray(64), before[0].subarray(32, 64));
  f.renderer.copyTextureToTexture(destination, destination, box(0, 0, 4, 4), point(4, 4), 3, 0);
  assert.equal(publicCopies(f)[2].from.texture, native);
  assert.deepEqual(native.levels[0].data.subarray(24, 32), before[3].subarray(0, 8));
  assert.throws(() => f.renderer.copyTextureToTexture(destination, destination, box(0, 0, 4, 4), point(4, 4)),
    {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  assert.equal(publicCopies(f).length, 3); await f.renderer.waitForGPU();
});

for (const mode of ['source-origin', 'destination-origin', 'width', 'height', 'physical-bounds', 'format', 'compatibility'])
test(`compressed copy rejects ${mode} before uploads and rolls back destination ownership`, async t => {
  const f = await fixture(t, mode === 'compatibility' ? {features: ['texture-compression-bc']} : {});
  const source = compressed(t), destination = compressed(t, mode === 'format' ? {format: T.RGBA_S3TC_DXT5_Format, blockBytes: 16} : {});
  const region = box(mode === 'source-origin' ? 1 : 0, 0, mode === 'width' ? 2 : mode === 'physical-bounds' ? 16 : 4,
    mode === 'height' ? 2 : 4), position = point(mode === 'destination-origin' ? 1 : 0, 0);
  assert.throws(() => f.renderer.copyTextureToTexture(source, destination, region, position),
    {code: `THREE_STORAGE_COPY_${mode === 'compatibility' ? 'FEATURE' : mode === 'format' ? 'FORMAT' : mode === 'physical-bounds' ? 'BOUNDS' : 'BLOCK'}`});
  assert.equal(f.textures.length, f.initialTextures); assert.equal(f.uploads.length, 0); assert.equal(f.copies.length, 0);
  assert.equal(f.inputs.isCopyDestination(destination), false); assert.equal(f.inputs.diagnostics.snapshotBytes, 0);
});

test('existing unaligned compressed allocation support keeps public copy requests unrounded', async t => {
  const f = await fixture(t, {features: ['core-features-and-limits', 'texture-compression-bc', 'texture-compression-unaligned']});
  const source = compressed(t, {width: 6, height: 6, value: 31}), destination = compressed(t, {width: 6, height: 6});
  assert.throws(() => f.renderer.copyTextureToTexture(source, destination), {code: 'THREE_STORAGE_COPY_BLOCK'});
  assert.equal(f.textures.length, f.initialTextures);
  f.renderer.copyTextureToTexture(source, destination, box(0, 0, 8, 8));
  assert.deepEqual(commandShape(publicCopies(f)[0]), sourceCommand(source, destination, box(0, 0, 8, 8), null, 0, 0));
  assert.deepEqual(f.native(destination).levels[0].data, raw(source.mipmaps[0].data));
  await f.renderer.waitForGPU();
});

test('compressed cube destinations retain their boundary alongside ordinary cube admission', async t => {
  const f = await fixture(t), source = compressed(t, {width: 4, height: 4});
  const faces = Array.from({length: 6}, (_, face) => ({width: 4, height: 4,
    mipmaps: [{width: 4, height: 4, data: new Uint8Array(8).fill(face + 1)}]}));
  const destination = new T.CompressedCubeTexture(faces, T.RGBA_S3TC_DXT1_Format);
  destination.needsUpdate = true; t.after(() => destination.dispose());
  assert.throws(() => f.renderer.copyTextureToTexture(source, destination, box(0, 0, 4, 4), point(0, 0, 2)),
    {code: 'THREE_COMPUTE_TEXTURE_COPY'});
  assert.equal(f.textures.length, f.initialTextures); assert.equal(f.uploads.length, 0); assert.equal(f.copies.length, 0);
  assert.equal(f.inputs.isCopyDestination(destination), false); assert.equal(f.inputs.diagnostics.snapshotBytes, 0);
});

test('disposing a compressed destination cancels its pending copy without allocating or uploading', async t => {
  const f = await fixture(t), source = compressed(t), destination = compressed(t), release = blockQueue(f.renderer);
  let updates = 0; source.onUpdate = destination.onUpdate = () => updates++;
  f.renderer.copyTextureToTexture(source, destination, box(0, 0, 4, 4)); destination.dispose(); release();
  await assert.rejects(f.renderer.waitForGPU(), /disposed|STALE/i);
  assert.equal(f.textures.length, f.initialTextures); assert.equal(f.uploads.length, 0); assert.equal(f.copies.length, 0); assert.equal(updates, 0);
  f.renderer.dispose(); assert.equal(f.inputs.diagnostics.snapshotBytes, 0);
});
