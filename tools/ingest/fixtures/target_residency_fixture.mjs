/** Recording boundary only: no WGSL execution or pixel-parity claim. */
export function targetDevice() {
  const calls = {textures: [], samplers: [], pipelines: [], groups: [], submissions: [], modules: [], scopes: [], destroyed: 0};
  const controls = {error: null};
  let lose;
  const device = {
    limits: {maxTextureDimension2D: 8192, maxBufferSize: 1024 * 1024}, features: new Set(),
    lost: new Promise(resolve => { lose = resolve; }),
    createTexture(d) {
      const t = {...d, width: d.size.width, height: d.size.height, destroyed: 0,
        createView() { return {texture: t}; }, destroy() { t.destroyed++; }};
      calls.textures.push(t); return t;
    },
    createSampler(d) { const s = {...d}; calls.samplers.push(s); return s; },
    createShaderModule(d) { calls.modules.push(d); return d; },
    createRenderPipeline(d) { const p = {...d, getBindGroupLayout() { return {}; }}; calls.pipelines.push(p); return p; },
    createBindGroup(d) { calls.groups.push(d); return d; },
    pushErrorScope(kind) { calls.scopes.push(kind); },
    popErrorScope() { calls.scopes.pop(); const error = controls.error; controls.error = null; return Promise.resolve(error); },
    createCommandEncoder() {
      const passes = [], copies = [];
      return {beginRenderPass(d) {
        const p = {descriptor: d}; passes.push(p);
        return {setPipeline(v) { p.pipeline = v; }, setBindGroup(i, v) { p.group = v; }, draw(n) { p.vertices = n; }, end() { p.ended = true; }};
      }, copyTextureToBuffer(source, target, size) { copies.push({source, target, size}); }, finish() { return {passes, copies}; }};
    },
    queue: {onSubmittedWorkDone: () => Promise.resolve(), submit(commands) { calls.submissions.push(...commands); }},
    destroy() { calls.destroyed++; },
  };
  return {device, calls, controls, lose};
}

// The owner tests exercise the event/identity/descriptor contract, not Three.js
// implementation equivalence. Renderer integration can also use the real r186
// module through F3D_THREE_ROOT in the repository's existing suites.
class Events {
  listeners = new Map();
  addEventListener(name, fn) { if (!this.listeners.has(name)) this.listeners.set(name, new Set()); this.listeners.get(name).add(fn); }
  removeEventListener(name, fn) { this.listeners.get(name)?.delete(fn); }
  dispose() { for (const fn of [...this.listeners.get('dispose') ?? []]) fn({target: this}); }
}
class Texture extends Events {
  constructor() {
    super(); Object.assign(this, {isTexture: true, version: 0, source: {version: 0}, format: 1023, type: 1009, colorSpace: '',
      internalFormat: null, generateMipmaps: false, minFilter: 1006, magFilter: 1006, wrapS: 1001, wrapT: 1001, anisotropy: 1});
  }
}
class Vector4 {
  isVector4 = true;
  constructor(x = 0, y = 0, z = 0, w = 0) { this.set(x, y, z, w); }
  set(x, y, z, w) { Object.assign(this, {x, y, z, w}); return this; }
  copy(v) { return this.set(v.x, v.y, v.z, v.w); }
  toArray() { return [this.x, this.y, this.z, this.w]; }
}
class RenderTarget extends Events {
  constructor(width = 4, height = 2, options = {}) {
    super(); Object.assign(this, {isRenderTarget: true, width, height, depth: 1, samples: 0, depthBuffer: true, stencilBuffer: false, depthTexture: null, resolveColorBuffer: true});
    const t = new Texture(); t.isRenderTargetTexture = true; t.renderTarget = this;
    this.textures = [t]; this.viewport = new Vector4(0, 0, width, height); this.scissor = new Vector4(0, 0, width, height); this.scissorTest = false;
    for (const [k, v] of Object.entries(options)) if (k in t) t[k] = v; else this[k] = v;
  }
  get texture() { return this.textures[0]; }
  setSize(width, height) { this.width = width; this.height = height; this.dispose(); this.viewport.set(0, 0, width, height); this.scissor.copy(this.viewport); }
}
export const targetThree = {REVISION: '186', Texture, RenderTarget, Vector4, RGBAFormat: 1023,
  UnsignedByteType: 1009, HalfFloatType: 1016, NoColorSpace: '', LinearSRGBColorSpace: 'srgb-linear', SRGBColorSpace: 'srgb',
  NearestFilter: 1003, LinearFilter: 1006, ClampToEdgeWrapping: 1001, RepeatWrapping: 1000, MirroredRepeatWrapping: 1002};
