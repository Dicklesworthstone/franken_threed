/** Byte-copy/descriptor recorder. Render passes are recorded, not rasterized;
 * this fixture does not execute WGSL, generate mip pixels or certify GPU parity.
 */
import assert from 'node:assert/strict';
import {targetDevice, targetThree} from './target_residency_fixture.mjs';
export const transferThree = {...targetThree, FloatType: 1015, RedFormat: 1028, RGFormat: 1030,
  NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008};
const bytesPerPixel = format => ({rgba16float: 8, rgba32float: 16, rg32float: 8}[format] ?? 4);
export function transferDevice() {
  const f = targetDevice(), createTexture = f.device.createTexture, createEncoder = f.device.createCommandEncoder;
  const submit = f.device.queue.submit;
  f.calls.uploads = []; f.calls.colorCopies = []; f.calls.buffers = [];
  f.device.createBindGroupLayout = d => d; f.device.createPipelineLayout = d => d;
  f.device.createTexture = d => {
    const t = createTexture(d); t.levels = [];
    for (let level = 0; level < d.mipLevelCount; level++) {
      const width = Math.max(1, Math.floor(t.width / 2 ** level)), height = Math.max(1, Math.floor(t.height / 2 ** level));
      t.levels.push({width, height, bytes: new Uint8Array(width * height * bytesPerPixel(t.format))});
    }
    t.createView = (descriptor = {}) => ({texture: t, descriptor});
    return t;
  };
  f.device.createBuffer = d => {
    const buffer = {bytes: new Uint8Array(d.size), destroyed: 0,
      mapAsync: () => f.controls.mapping ?? Promise.resolve(),
      getMappedRange() { return this.bytes.buffer; }, unmap() {}, destroy() { this.destroyed++; }};
    f.calls.buffers.push(buffer); return buffer;
  };
  f.device.queue.writeTexture = (destination, data, layout, size) => {
    f.controls.onUpload?.();
    assert.ok(destination.texture.usage & 2); assert.equal(destination.texture.sampleCount, 1);
    const snapshot = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    f.calls.uploads.push({destination, data: snapshot, layout, size});
    const level = destination.texture.levels[destination.mipLevel ?? 0], bpp = bytesPerPixel(destination.texture.format);
    for (let row = 0; row < size.height; row++) {
      const from = layout.offset + row * layout.bytesPerRow;
      level.bytes.set(snapshot.subarray(from, from + size.width * bpp),
        ((destination.origin.y + row) * level.width + destination.origin.x) * bpp);
    }
  };
  f.device.createCommandEncoder = () => {
    const encoder = createEncoder(), finish = encoder.finish, actions = [];
    encoder.copyTextureToTexture = (source, destination, size) => {
      f.controls.onEncodeCopy?.();
      assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
      assert.equal(source.texture.sampleCount, 1); assert.equal(destination.texture.sampleCount, 1);
      f.calls.colorCopies.push({source, destination, size});
      actions.push(() => {
        const a = source.texture.levels[source.mipLevel ?? 0], b = destination.texture.levels[destination.mipLevel ?? 0];
        const bpp = bytesPerPixel(source.texture.format);
        for (let row = 0; row < size.height; row++) {
          const offset = ((source.origin.y + row) * a.width + source.origin.x) * bpp;
          b.bytes.set(a.bytes.subarray(offset, offset + size.width * bpp),
            ((destination.origin.y + row) * b.width + destination.origin.x) * bpp);
        }
      });
    };
    encoder.copyTextureToBuffer = (source, destination, size) => {
      actions.push(() => {
        const level = source.texture.levels[source.mipLevel ?? 0], bpp = bytesPerPixel(source.texture.format);
        for (let row = 0; row < size.height; row++) {
          const offset = ((source.origin.y + row) * level.width + source.origin.x) * bpp;
          destination.buffer.bytes.set(level.bytes.subarray(offset, offset + size.width * bpp), row * destination.bytesPerRow);
        }
      });
    };
    encoder.finish = () => ({...finish(), actions});
    return encoder;
  };
  f.device.queue.submit = commands => { for (const command of commands) for (const run of command.actions ?? []) run(); submit(commands); };
  f.device.queue.onSubmittedWorkDone = () => f.controls.completion ?? Promise.resolve();
  return f;
}
export function deferred() {
  let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
