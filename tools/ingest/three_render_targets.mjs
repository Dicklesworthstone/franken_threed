/** Source RenderTarget residency shared by all scenes of one renderer.
 * Native attachments stay top-left for rendering/readback. A lazily allocated
 * GPU-only, vertically flipped sampling image preserves source texture UVs;
 * it is refreshed once per written version, never downloaded or re-uploaded.
 * This is general WebGPU execution, not Rust specialization or a speedup claim.
 */
import {createGpuRenderTarget} from './gpu_render_target.mjs';

export class ThreeRenderTargetError extends Error {
  constructor(code, message) {
    super(`THREE_TARGET_${code}: ${message}`);
    this.name = 'ThreeRenderTargetError'; this.code = `THREE_TARGET_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeRenderTargetError(code, message); };
const positive = n => Number.isSafeInteger(n) && n > 0;

export function inspectThreeRenderTarget(target, three) {
  if (!three?.RenderTarget || !(target instanceof three.RenderTarget)) fail('SOURCE', 'Expected a source RenderTarget');
  if (!positive(target.width) || !positive(target.height)) fail('SIZE', 'Target dimensions must be positive integers');
  if (target.isWebGLCubeRenderTarget || target.depth !== 1 || target.multiview || target.textures?.length !== 1)
    fail('UNSUPPORTED', 'Only single-color 2D render targets are admitted');
  const texture = target.texture;
  if (!(texture instanceof three.Texture) || texture.isCubeTexture || texture.isDataArrayTexture || texture.isData3DTexture ||
      texture.renderTarget !== target || texture.isRenderTargetTexture !== true)
    fail('SOURCE', 'Expected the target-owned 2D color texture');
  if (target.stencilBuffer || target.depthTexture) fail('UNSUPPORTED', 'Stencil and sampled depth targets are not admitted');
  if (![0, 1, 4].includes(target.samples)) fail('UNSUPPORTED', 'Target samples must be zero, one or four');
  if (target.samples === 4 && target.resolveColorBuffer === false) fail('UNSUPPORTED', 'MSAA targets must resolve their color');
  if (texture.format !== three.RGBAFormat || texture.internalFormat != null || texture.generateMipmaps)
    fail('UNSUPPORTED', 'Targets require RGBA, no internal format override and no generated mipmaps');
  const linear = texture.colorSpace === three.NoColorSpace || texture.colorSpace === three.LinearSRGBColorSpace;
  let format;
  if (texture.type === three.UnsignedByteType && (linear || texture.colorSpace === three.SRGBColorSpace))
    format = linear ? 'rgba8unorm' : 'rgba8unorm-srgb';
  else if (texture.type === three.HalfFloatType && linear) format = 'rgba16float';
  else fail('UNSUPPORTED', 'Target color must be RGBA8 or linear RGBA16F');
  const sampleCount = target.samples === 4 ? 4 : 1, depthFormat = target.depthBuffer ? 'depth24plus' : null;
  const colorBytes = target.width * target.height * (format === 'rgba16float' ? 8 : 4);
  const bytes = colorBytes * (sampleCount === 4 ? 5 : 1) + (depthFormat ? target.width * target.height * 4 * sampleCount : 0);
  if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Target byte size exceeds safe integer precision');
  return {width: target.width, height: target.height, format, depthFormat, sampleCount, bytes, colorBytes, texture,
    key: `${target.width}/${target.height}/${format}/${depthFormat}/${sampleCount}`};
}

function samplerDescriptor(texture, three) {
  const filters = new Map([[three.NearestFilter, 'nearest'], [three.LinearFilter, 'linear']]);
  const modes = new Map([[three.ClampToEdgeWrapping, 'clamp-to-edge'], [three.RepeatWrapping, 'repeat'], [three.MirroredRepeatWrapping, 'mirror-repeat']]);
  const magFilter = filters.get(texture.magFilter), minFilter = filters.get(texture.minFilter);
  const addressModeU = modes.get(texture.wrapS), addressModeV = modes.get(texture.wrapT);
  if (!magFilter || !minFilter || !addressModeU || !addressModeV || !Number.isFinite(texture.anisotropy) || texture.anisotropy < 1)
    fail('SAMPLER', 'Expected nearest/linear target filtering, source wrap modes and positive anisotropy');
  return {magFilter, minFilter, mipmapFilter: 'linear', addressModeU, addressModeV, lodMaxClamp: 0,
    maxAnisotropy: magFilter === 'linear' && minFilter === 'linear' ? Math.min(16, Math.floor(texture.anisotropy)) : 1};
}

const FLIP_WGSL = `
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let size = textureDimensions(source);
  return textureLoad(source, vec2i(i32(p.x), i32(size.y) - 1 - i32(p.y)), 0);
}`;

/** getDevice is lazy so selecting a target before renderer.init() performs no GPU work. */
export function createThreeRenderTargets(three, {getDevice, maxTargets = 256, maxBytes = 512 * 1024 * 1024} = {}) {
  if (typeof getDevice !== 'function' || !positive(maxTargets) || !positive(maxBytes)) fail('OPTIONS', 'Invalid target owner options');
  const entries = new Map(), pipelines = new Map();
  let disposed = false, allocatedBytes = 0, terminal = null;
  const live = () => { if (disposed) fail('DISPOSED', 'Render target owner is disposed'); if (terminal) throw terminal; };
  function release(entry) {
    if (entry.busy || entry.released) return;
    entry.released = true;
    try { entry._dispatcher?.dispose(); }
    finally {
      entry.storage?.dispose(); entry.sample?.destroy();
      allocatedBytes -= entry.charged; entry.charged = 0;
      entry.storage = entry.sample = entry.view = entry.group = null;
      entry._wanted.clear();
    }
  }
  function retire(entry) {
    if (!entry.alive) return;
    entry.alive = false;
    entry.target.removeEventListener('dispose', entry.onDispose);
    entry.shape.texture.removeEventListener('dispose', entry.onDispose);
    if (entries.get(entry.target) === entry) entries.delete(entry.target);
    release(entry);
  }
  function capture(target) {
    live();
    const shape = inspectThreeRenderTarget(target, three);
    let entry = entries.get(target);
    if (entry && (entry.shape.key !== shape.key || entry.shape.texture !== shape.texture)) { retire(entry); entry = null; }
    if (entry) return entry;
    if (entries.size >= maxTargets) fail('BUDGET', 'Render target count exceeds maxTargets');
    entry = {target, shape, alive: true, released: false, busy: false, charged: 0, storage: null, sample: null,
      dirty: true, copies: 0, error: null, validation: Promise.resolve(), _wanted: new Map(), _dispatcher: null};
    entry.onDispose = () => retire(entry);
    target.addEventListener('dispose', entry.onDispose);
    shape.texture.addEventListener('dispose', entry.onDispose);
    entries.set(target, entry);
    return entry;
  }
  function assertCurrent(entry) {
    live();
    if (!entry?.alive || entries.get(entry.target) !== entry) fail('STALE', 'Target was disposed or replaced before its queued use');
    const shape = inspectThreeRenderTarget(entry.target, three);
    if (shape.key !== entry.shape.key || shape.texture !== entry.shape.texture) fail('STALE', 'Target configuration changed before its queued use');
    if (entry.error) throw entry.error;
    if (entry.storage?.failed) fail('GPU', 'Native target storage failed');
    return entry;
  }
  function budget(bytes) {
    if (bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Render targets exceed their shared byte budget');
  }
  function ensure(entry) {
    assertCurrent(entry);
    if (entry.storage) return entry;
    budget(entry.shape.bytes);
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before acquiring target storage');
    const {width, height, format, depthFormat, sampleCount} = entry.shape;
    entry.storage = createGpuRenderTarget(device, {width, height, format, depthFormat, sampleCount, maxBytes});
    entry.charged += entry.shape.bytes; allocatedBytes += entry.shape.bytes;
    // The readback helper consumes this session-shaped object. It copies the
    // original attachment, not the UV-oriented sampling replica.
    entry.session = {device, get width() { return width; }, get height() { return height; },
      get disposed() { return !entry.alive; }, get failed() { return !!entry.error || !!entry.storage?.failed; },
      get diagnostics() { return {target: {format}}; },
      withTexture(callback) { assertCurrent(entry); return entry.storage.withTexture(callback); }};
    return entry;
  }
  function checked(entry, operation) {
    const device = getDevice();
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let result, error;
    try { result = operation(device); } catch (cause) { error = cause; }
    const invalid = device.popErrorScope(), oom = device.popErrorScope();
    entry.validation = Promise.all([entry.validation, invalid, oom]).then(([, a, b]) => {
      if (error) throw error;
      if (a || b) throw new ThreeRenderTargetError('GPU', (a || b).message ?? 'Target sampling failed');
    });
    entry.validation.catch(cause => { terminal ??= cause; entry.error = cause; retire(entry); });
    if (error) { entry.error = error; retire(entry); throw error; }
    return result;
  }
  function sampling(entry, texture) {
    ensure(entry);
    // Validate before allocating any additional native objects.
    const descriptor = samplerDescriptor(texture, three), samplerKey = JSON.stringify(descriptor);
    if (!entry.sample) {
      budget(entry.shape.colorBytes);
      checked(entry, device => {
        const {width, height, format} = entry.shape;
        entry.sample = device.createTexture({label: 'Source target UV sampling image',
          size: {width, height, depthOrArrayLayers: 1}, format, dimension: '2d', sampleCount: 1, mipLevelCount: 1, usage: 4 | 16});
        entry.charged += entry.shape.colorBytes; allocatedBytes += entry.shape.colorBytes;
        entry.view = entry.sample.createView();
        let pipeline = pipelines.get(format);
        if (!pipeline) {
          const module = device.createShaderModule({label: 'Source target UV orientation', code: FLIP_WGSL});
          pipeline = device.createRenderPipeline({label: 'Source target UV orientation', layout: 'auto',
            vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
            primitive: {topology: 'triangle-list'}});
          pipelines.set(format, pipeline);
        }
        entry.pipeline = pipeline;
        entry.storage.withTexture(source => {
          entry.group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: source.createView()}]});
        });
      });
    }
    if (entry.samplerKey !== samplerKey) {
      entry.sampler = checked(entry, device => device.createSampler(descriptor));
      entry.samplerKey = samplerKey;
    }
    if (entry.dirty) {
      checked(entry, device => entry.storage.withTexture(() => {
        const encoder = device.createCommandEncoder({label: 'Source target UV orientation'});
        const pass = encoder.beginRenderPass({colorAttachments: [{view: entry.view, loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
        pass.setPipeline(entry.pipeline); pass.setBindGroup(0, entry.group); pass.draw(3); pass.end();
        device.queue.submit([encoder.finish()]);
      }));
      entry.dirty = false; entry.copies++;
    }
    return {view: entry.view, sampler: entry.sampler, sampleType: 'float', viewDimension: '2d',
      version: texture.version, sourceVersion: texture.source.version};
  }
  // The existing scene bridge accepts a Map of externally owned bindings. Its
  // has/get calls always resolve the current target generation, so stale views
  // trigger the bridge's normal re-preparation boundary before submission.
  function bindingsFor(destination = null) {
    return new class extends Map {
      has(texture) { return texture?.isRenderTargetTexture === true && texture.renderTarget != null; }
      get(texture) {
        if (!this.has(texture)) return undefined;
        if (texture.renderTarget === destination) fail('FEEDBACK', 'A render pass cannot sample its own color target');
        const entry = capture(texture.renderTarget);
        if (entry.shape.texture !== texture) fail('STALE', 'Texture no longer belongs to its source target');
        return sampling(entry, texture);
      }
    }();
  }
  return {
    capture, ensure, assertCurrent, bindingsFor,
    get allocatedBytes() { return allocatedBytes; }, get size() { return entries.size; },
    render(entry, callback) {
      ensure(entry);
      if (entry.busy) fail('REENTRANT', 'Cannot reenter a target submission');
      entry.busy = true; entry.dirty = true;
      try { return entry.storage.withFrame((frame, texture) => { const result = callback(frame, texture); assertCurrent(entry); return result; }); }
      finally { entry.busy = false; if (!entry.alive) release(entry); }
    },
    async whenIdle() {
      live();
      try { await Promise.all([...entries.values()].flatMap(entry => [entry.validation, entry.storage?.whenIdle(), entry._dispatcher?.whenIdle()])); }
      catch (error) { throw terminal ?? error; }
      live();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of [...entries.values()]) retire(entry);
      pipelines.clear();
    },
  };
}
