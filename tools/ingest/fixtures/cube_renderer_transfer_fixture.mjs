/** Real transfer controller, source target owner and native target implementation.
 * Only source descriptor identities, the renderer's queue driver and GPU calls
 * are doubles. Reference arrays check addressing, not WGSL/browser pixel parity.
 */
import assert from 'node:assert/strict';
import {targetThree} from './target_residency_fixture.mjs';
import {depthDevice, deferred, flush} from '../tests/depth_transfer_test_fixture.mjs';
import {createThreeRenderTargets} from '../three_render_targets.mjs';
import {createThreeRendererTransfers} from '../three_renderer_transfers.mjs';
export {deferred, flush};

class DepthTexture extends targetThree.Texture {
  constructor(width, height) {
    super(); Object.assign(this, {isDepthTexture: true, image: {width, height},
      format: 1026, type: 1015, flipY: false, minFilter: 1003, magFilter: 1003, compareFunction: null});
  }
}
const faces = size => Array.from({length: 6}, () => ({width: size, height: size, depth: 1}));
class CubeTexture extends targetThree.Texture {
  constructor(size) { super(); this.isCubeTexture = true; this.image = faces(size); this.renderTarget = null; }
}
class CubeDepthTexture extends DepthTexture {
  constructor(size) {
    super(size, size); Object.assign(this, {isCubeTexture: true, isCubeDepthTexture: true, image: faces(size), mapping: 301});
  }
}
class CubeRenderTarget extends targetThree.RenderTarget {
  constructor(size = 4) {
    super(size, size); this.isCubeRenderTarget = true;
    const texture = new CubeTexture(size); texture.isRenderTargetTexture = true; this.textures = [texture];
  }
}
export const T = {...targetThree, CubeTexture, CubeRenderTarget, DepthTexture, CubeDepthTexture,
  DepthFormat: 1026, DepthStencilFormat: 1027, FloatType: 1015, UnsignedIntType: 1014,
  UnsignedInt248Type: 1020, CubeReflectionMapping: 301};
export function target({cube = true, size = 4, depth = true, samples = 0} = {}) {
  const result = cube ? new T.CubeRenderTarget(size) : new T.RenderTarget(size, size);
  result.samples = samples;
  if (depth) {
    result.depthTexture = cube ? new T.CubeDepthTexture(size) : new T.DepthTexture(size, size);
    result.depthTexture.renderTarget = result;
  }
  return result;
}

function colorDevice() {
  const f = depthDevice(), uploads = [], reads = [], copies = [];
  const createTexture = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const texture = createTexture(descriptor);
    texture.pixels = Array.from({length: descriptor.mipLevelCount}, (_, mip) =>
      new Uint8Array(Math.max(1, texture.width >> mip) * Math.max(1, texture.height >> mip) * texture.depthOrArrayLayers * 4));
    return texture;
  };
  function region(source, size, run) {
    const {texture, origin, mipLevel = 0} = source;
    const width = Math.max(1, texture.width >> mipLevel), height = Math.max(1, texture.height >> mipLevel);
    assert.equal(texture.destroyed, false);
    assert.equal(size.depthOrArrayLayers, 1);
    assert.ok(origin.z >= 0 && origin.z < texture.depthOrArrayLayers);
    assert.ok(origin.x >= 0 && origin.x + size.width <= width && origin.y >= 0 && origin.y + size.height <= height);
    for (let row = 0; row < size.height; row++) {
      const start = ((origin.z * height + origin.y + row) * width + origin.x) * 4;
      run(texture.pixels[mipLevel], start, size.width * 4, row);
    }
  }
  f.device.queue.writeTexture = (destination, data, layout, size) => {
    if (f.control.uploadError) throw f.control.uploadError;
    assert.ok(destination.texture.usage & 2);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    uploads.push({destination, bytes, layout, size}); f.events.push('upload');
    region(destination, size, (plane, start, length, row) =>
      plane.set(bytes.subarray(layout.offset + row * layout.bytesPerRow, layout.offset + row * layout.bytesPerRow + length), start));
  };
  const createEncoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const encoder = createEncoder(), extras = [], finish = encoder.finish;
    encoder.copyTextureToBuffer = (source, destination, size) => extras.push({kind: 'color-read', source, destination, size});
    encoder.finish = () => [...finish(), ...extras];
    return encoder;
  };
  const submit = f.device.queue.submit;
  f.device.queue.submit = batches => {
    // The retained fixture models depth passes. Each production transfer submits
    // its own command buffer; color reads/copies use the independent byte model.
    submit(batches.map(batch => batch.filter(packet => packet.kind !== 'color-read')));
    for (const batch of batches) for (const packet of batch) {
      if (packet.kind === 'color-read') {
        reads.push(packet);
        assert.ok(packet.source.texture.usage & 1);
        region(packet.source, packet.size, (plane, start, length, row) =>
          new Uint8Array(packet.destination.buffer.bytes).set(plane.subarray(start, start + length), row * packet.destination.bytesPerRow));
      } else if (packet.kind === 'color-copy') {
        copies.push(packet);
        assert.ok(packet.source.texture.usage & 1); assert.ok(packet.destination.texture.usage & 2);
        const rows = [];
        region(packet.source, packet.size, (plane, start, length) => rows.push(plane.slice(start, start + length)));
        region(packet.destination, packet.size, (plane, start, length, row) => plane.set(rows[row], start));
      }
    }
  };
  return {...f, uploads, reads, copies};
}
export class RendererError extends Error {
  constructor(code, message) { super(message); this.code = `F3D_RENDERER_${code}`; }
}
export function fixture({initialized = true, flipY = false, ownerOptions = {}, transferOptions = {}} = {}) {
  const f = colorDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY, ...ownerOptions});
  const renderer = {_targets: owner, _targetLifetime: new AbortController(), _initialized: initialized,
    _bottomLeftOrigin: flipY, _pending: [], _drain: null, _preparing: null, _disposed: false,
    _enqueue(request) { this._pending.push(request); },
  };
  const transfers = createThreeRendererTransfers(renderer, RendererError, transferOptions);
  // These are the public renderer's existing delegations. The installed init,
  // depth-copy and extended-readback methods themselves are production methods.
  renderer.writeRenderTargetPixelsAsync = (...args) => transfers.writePixels(...args);
  renderer.copyRenderTargetPixelsAsync = (...args) => transfers.copyColor(...args);
  function drain() {
    renderer._initialized = true; renderer._drain = true;
    try {
      while (renderer._pending.length) {
        const request = renderer._pending[0];
        try { request.resolve(request.issue()); } catch (error) { request.reject(error); }
        renderer._pending.shift();
      }
    } finally { renderer._drain = null; }
  }
  function dispose() {
    renderer._disposed = true;
    const error = new RendererError('DISPOSED', 'Renderer disposed');
    renderer._targetLifetime.abort(error); owner.dispose();
    for (const request of renderer._pending.splice(0)) request.reject(error);
  }
  return {...f, owner, renderer, transfers, drain, dispose};
}
export function seedDepth(f, source) {
  const entry = f.owner.capture(source); f.owner.ensure(entry);
  entry.storage.withDepthTexture(texture => {
    texture.data.set(Array.from(texture.data, (_, i) => (i + 1) / (texture.data.length + 1)));
  });
  return entry;
}
