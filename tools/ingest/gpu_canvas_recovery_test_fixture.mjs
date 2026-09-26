/** Recorded native GPU boundary. Production canvas/owner/recovery code is not
 * replaced. This models command/resource lifetime, not actual GPU pixels.
 */
export function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  promise.catch(() => {});
  return {promise, resolve, reject};
}
export async function turns(count = 12) { for (let i = 0; i < count; i++) await Promise.resolve(); }
export function recoveryFixture() {
  const log = [], adapters = [], devices = [], renderers = [];
  const canvas = {width: 16, height: 12, getContext(type) {
    if (type !== 'webgpu') throw new Error('Unexpected context route');
    return context;
  }};
  const context = {
    device: null, configurations: [], acquisitions: 0, unconfigurations: 0,
    configure(descriptor) {
      this.device = descriptor.device; this.configurations.push(descriptor);
      log.push(['configure', descriptor.device.name]);
    },
    unconfigure() { log.push(['unconfigure', this.device?.name]); this.unconfigurations++; this.device = null; },
    getCurrentTexture() {
      if (!this.device || this.device.isLost) throw new Error('No live canvas device');
      this.acquisitions++;
      return this.device.texture({size: [canvas.width, canvas.height], format: 'bgra8unorm', usage: 16}, true);
    },
  };
  const gpu = {
    requests: [], deviceRequests: [],
    getPreferredCanvasFormat() { return 'bgra8unorm'; },
    async requestAdapter(options) {
      this.requests.push(options); log.push(['adapter']);
      if (!adapters.length) return null;
      return adapters.shift();
    },
  };
  function device(name, options = {}) {
    const lost = deferred();
    const d = {
      name, features: new Set(options.features ?? []),
      limits: {maxTextureDimension2D: 2048, maxBufferSize: 1048576,
        minUniformBufferOffsetAlignment: 256, ...options.limits},
      lost: lost.promise, isLost: false, destroyCount: 0, textures: [], scopes: [], scopeResults: [],
      scopePending: [],
      queue: {fence: null, onSubmittedWorkDone() { log.push(['fence', name]); return this.fence ?? Promise.resolve(); }},
      pushErrorScope(kind) { this.scopes.push(kind); },
      popErrorScope() {
        if (!this.scopes.length) throw new Error('Unbalanced GPU scopes');
        this.scopes.pop();
        return this.scopeResults.shift() ?? Promise.resolve(null);
      },
      texture(descriptor, borrowed = false) {
        const size = descriptor.size, texture = {device: d, descriptor, borrowed,
          width: size[0], height: size[1], depthOrArrayLayers: size[2] ?? 1,
          format: descriptor.format, sampleCount: descriptor.sampleCount ?? 1,
          usage: descriptor.usage, dimension: '2d', destroyed: 0,
          createView(view = {}) { return {texture: this, descriptor: view}; },
          destroy() { this.destroyed++; log.push(['texture.destroy', name, borrowed]); },
        };
        this.textures.push(texture); log.push(['texture', name, borrowed]); return texture;
      },
      createTexture(descriptor) { return this.texture(descriptor); },
      destroy() { this.destroyCount++; log.push(['device.destroy', name]); this.lose({reason: 'destroyed', message: 'Explicit destroy'}); },
      lose(info = {reason: 'unknown', message: `Lost ${name}`}) { this.isLost = true; lost.resolve(info); },
      rejectLoss(error) { lost.reject(error); },
    };
    devices.push(d); return d;
  }
  function adapter(d, settings = {}) {
    return {features: d.features, limits: d.limits, ...settings,
      requestDevice(options) {
        gpu.deviceRequests.push(options); log.push(['requestDevice', d.name]);
        return settings.devicePromise ?? Promise.resolve(d);
      }};
  }
  function enqueue(d, settings) { adapters.push(adapter(d, settings)); return d; }
  function factory(d, attachments, lifetime) {
    const renderer = {device: d, attachments, lifetime, disposed: false, failed: false,
      disposeCount: 0, renders: [], preparations: 0, idleWait: null, prepareWait: null,
      error: null, renderHook: null,
      render(input, frame, texture) {
        if (this.disposed) throw new Error('Disposed renderer');
        this.renderHook?.();
        if (this.error) throw this.error;
        this.renders.push({input, frame, texture}); log.push(['render', d.name, input]);
      },
      prepare() { this.preparations++; return this.prepareWait ?? Promise.resolve(); },
      whenIdle() { if (this.error && this.failed) return Promise.reject(this.error); return this.idleWait ?? Promise.resolve(); },
      dispose() { if (this.disposed) return; this.disposeCount++; this.disposed = true; log.push(['renderer.dispose', d.name]); },
      get diagnostics() { return {name: d.name, renders: this.renders.length}; },
    };
    renderers.push(renderer); log.push(['factory', d.name, lifetime.generation]); return renderer;
  }
  return {canvas, context, gpu, device, adapter, enqueue, adapters, devices, renderers, factory, log};
}
