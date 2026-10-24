/** Recording native calls with independent byte-array copies. No rasterization,
 * WGSL execution, hardware validation or browser pixel-parity claim. */
import assert from 'node:assert/strict';
import {gpuPostprocessFixture, deferred} from './animation/gpu_postprocess_fixture.mjs';
export {deferred};
export const layouts = new Map([
  ['rgba8unorm', [Uint8Array, 4]], ['rgba8unorm-srgb', [Uint8Array, 4]],
  ['bgra8unorm', [Uint8Array, 4]], ['bgra8unorm-srgb', [Uint8Array, 4]],
  ['rgba16float', [Uint16Array, 4]], ['rgba32float', [Float32Array, 4]],
  ['r32float', [Float32Array, 1]], ['rg32float', [Float32Array, 2]],
]);
export const bytes = data => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
export function mrtTransferDevice() {
  const f = gpuPostprocessFixture(), uploads = [], copies = [], reads = [];
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxColorAttachments: 8,
    maxColorAttachmentBytesPerSample: 64, maxBufferSize: 64 * 1024 * 1024});
  const createTexture = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const texture = createTexture(descriptor), layout = layouts.get(descriptor.format);
    Object.assign(texture, {sampleCount: descriptor.sampleCount, mipLevelCount: descriptor.mipLevelCount,
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers});
    texture.bpp = layout ? layout[0].BYTES_PER_ELEMENT * layout[1] : 4;
    texture.planes = Array.from({length: texture.mipLevelCount}, (_, mip) =>
      Array.from({length: texture.depthOrArrayLayers}, () => new Uint8Array(
        Math.max(1, texture.width >> mip) * Math.max(1, texture.height >> mip) * texture.bpp)));
    return texture;
  };
  f.device.createBuffer = descriptor => {
    const buffer = {descriptor, data: new ArrayBuffer(descriptor.size), destroyed: 0,
      mapAsync: () => f.controls.mapping?.promise ?? Promise.resolve(),
      getMappedRange() { assert.equal(this.destroyed, 0); return this.data; }, unmap() {}, destroy() { this.destroyed++; }};
    f.calls.buffers.push(buffer); return buffer;
  };
  const row = (copy, size, index) => {
    const {texture, origin, mipLevel = 0} = copy;
    assert.equal(texture.destroyed, 0); assert.equal(texture.sampleCount, 1);
    assert.equal(size.depthOrArrayLayers, 1);
    assert.ok(mipLevel >= 0 && mipLevel < texture.mipLevelCount);
    const width = Math.max(1, texture.width >> mipLevel), height = Math.max(1, texture.height >> mipLevel);
    assert.ok(origin.z >= 0 && origin.z < texture.depthOrArrayLayers);
    assert.ok(origin.x >= 0 && origin.y >= 0 && origin.x + size.width <= width && origin.y + size.height <= height);
    const start = ((origin.y + index) * width + origin.x) * texture.bpp;
    return texture.planes[mipLevel][origin.z].subarray(start, start + size.width * texture.bpp);
  };
  f.device.queue.writeTexture = (destination, data, layout, size) => {
    assert.ok(destination.texture.usage & 2);
    const snapshot = bytes(data).slice(); uploads.push({destination, layout, size, data: snapshot});
    for (let y = 0; y < size.height; y++) {
      const output = row(destination, size, y), offset = layout.offset + y * layout.bytesPerRow;
      output.set(snapshot.subarray(offset, offset + output.length));
    }
  };
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => { submit(commands); for (const command of commands) for (const run of command.commands) run(); };
  f.device.createCommandEncoder = () => {
    const commands = [];
    return {
      copyTextureToBuffer(source, destination, size) {
        assert.ok(source.texture.usage & 1); assert.equal(destination.bytesPerRow % 256, 0);
        reads.push({source, destination, size});
        commands.push(() => {
          for (let y = 0; y < size.height; y++) new Uint8Array(destination.buffer.data).set(row(source, size, y), y * destination.bytesPerRow);
        });
      },
      copyTextureToTexture(source, destination, size) {
        assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
        assert.equal(source.texture.format.replace(/-srgb$/, ''), destination.texture.format.replace(/-srgb$/, ''));
        copies.push({source, destination, size});
        commands.push(() => {
          for (let y = 0; y < size.height; y++) row(destination, size, y).set(row(source, size, y));
        });
      },
      finish() { return {commands}; },
    };
  };
  return {...f, uploads, copies, reads};
}
