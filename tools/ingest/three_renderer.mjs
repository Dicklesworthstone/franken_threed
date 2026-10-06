/** Drop-in Three.js r186 `WebGPURenderer` surface over the new WebGPU scene path.
 *
 * `createWebGPURendererClass(THREE)` returns a class with the public renderer
 * contract ordinary applications use -- `new WebGPURenderer(parameters)`,
 * `await init()`, `setAnimationLoop()`, `setSize()`/`setPixelRatio()`,
 * clear-color state, `render(scene, camera)`, `compileAsync()`, `info` and
 * `dispose()` -- and executes every admitted frame through the explicit source
 * scene bridge (`three_scene.mjs`) on an owned canvas session. It constructs no
 * upstream renderer and has no hidden retained-renderer submission fallback.
 *
 * Route/ownership: GENERAL new WebGPU execution in retained JavaScript command
 * preparation. Not a Rust/Wasm specialization claim and not an acceleration claim.
 *
 * Contract differences from the source renderer, all explicit:
 * - Structural scene edits (a new mesh/material binding, a material structure
 *   change, a new scene, enabling fog/environment/background/shadows) need the
 *   bridge's asynchronous preparation boundary. `render()` then returns without
 *   presenting; the canvas keeps its previously presented image (attachments are
 *   acquired lazily) and the pending render requests are submitted, in order,
 *   as soon as preparation completes. Application callbacks are never skipped
 *   or replayed. `info.f3d.deferredRenders` counts such requests. Await
 *   `compileAsync(scene, camera)` to prepare ahead of the first frame.
 * - Admission failures discovered during deferred preparation are thrown from
 *   the next `render()`/`compileAsync()` call instead of the original call.
 * - Unsupported source state (render targets, XR presentation, custom tone
 *   mapping, logarithmic/reversed depth, stencil buffers, non-sRGB output,
 *   transparent clears with tone mapping) throws `F3DRendererError` rather than
 *   silently rendering something else.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {createGpuThreeScene} from './three_scene.mjs';

export class F3DRendererError extends Error {
  constructor(code, message) {
    super(`F3D_RENDERER_${code}: ${message}`);
    this.name = 'F3DRendererError';
    this.code = `F3D_RENDERER_${code}`;
  }
}
const fail = (code, message) => { throw new F3DRendererError(code, message); };
const NOT_INITIALIZED = 'THREE.Renderer: .render() called before the backend is initialized. Use "await renderer.init();" before rendering.';
// r186 tone-mapping constants -> explicit output-pass names. CustomToneMapping (5)
// is a shader hook and has no admitted equivalent.
const TONE_MAPPINGS = new Map([[0, 'none'], [1, 'linear'], [2, 'reinhard'], [3, 'cineon'], [4, 'aces-filmic'], [6, 'agx'], [7, 'neutral']]);
const PARAMETERS = new Set(['canvas', 'antialias', 'alpha', 'depth', 'stencil', 'samples', 'forceWebGL',
  'logarithmicDepthBuffer', 'reversedDepthBuffer', 'powerPreference', 'requiredLimits', 'requiredFeatures',
  'device', 'outputBufferType', 'multiview', 'trackTimestamp', 'colorBufferType', 'getFallback', 'context']);
const MAX_PENDING = 64;

/** True when the error is a recoverable "call prepare()" boundary from the bridge. */
const isPrepareBoundary = error => error?.code === 'THREE_SCENE_PREPARE' || error?.code === 'F3D_RENDERER_PREPARE';

function inspectorStub() {
  return {isRunning: false, setRenderer() { return this; }, begin() { this.isRunning = true; }, finish() { this.isRunning = false; },
    beginRender() {}, finishRender() {}, beginCompute() {}, finishCompute() {}, dispose() {}};
}

export function createWebGPURendererClass(THREE, classOptions = {}) {
  if (THREE?.REVISION !== '186' || typeof THREE.Scene !== 'function' || typeof THREE.Color !== 'function' ||
      typeof THREE.Vector2 !== 'function') fail('SOURCE', 'Supply the pinned r186 three module namespace');
  if (!classOptions || typeof classOptions !== 'object' || Array.isArray(classOptions)) fail('OPTIONS', 'Expected class options');
  const {gpu: defaultGpu, maxDraws = 16384, maxBindings = 16384, scene: sceneLimits = {}, exactBackend = null} = classOptions;
  if (exactBackend !== null && typeof exactBackend !== 'function') fail('OPTIONS', 'exactBackend must be a renderer constructor');
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1 || !Number.isSafeInteger(maxBindings) || maxBindings < 1)
    fail('OPTIONS', 'Expected positive draw and binding capacities');
  const {NoToneMapping, SRGBColorSpace} = THREE;

  /** Cheap, per-frame profile of scene-level features that select bridge options. */
  function sceneKey(renderer, scene) {
    const background = scene.background !== null && scene.background !== undefined && !scene.background.isColor;
    return `${scene.fog ? 1 : 0}${scene.environment ? 1 : 0}${background ? 1 : 0}${renderer.shadowMap.enabled ? 1 : 0}`;
  }
  /** Preparation-time traversal for options that depend on materials. */
  function needsClipping(scene) {
    let clipping = false;
    scene.traverse(object => {
      const list = Array.isArray(object.material) ? object.material : object.material ? [object.material] : [];
      if (list.some(m => m?.clippingPlanes?.length)) clipping = true;
    });
    if (scene.overrideMaterial?.clippingPlanes?.length) clipping = true;
    return clipping;
  }
  // r186 WebGPURenderer applies material clipping planes without a renderer-level
  // opt-in; there are no renderer-global clipping planes on this path.
  const clippingControls = Object.freeze({planes: Object.freeze([]), localClippingEnabled: true});

  /** Multi-scene renderer handed to the owned canvas session. */
  function createDispatcher(owner) {
    return (device, attachments, {signal}) => {
      const entries = new Map();
      let disposed = false;
      const dispatcher = {
        get disposed() { return disposed; },
        get failed() { for (const e of entries.values()) if (e.bridge.failed) return true; return false; },
        get diagnostics() {
          return Object.freeze({scenes: entries.size,
            bridges: [...entries.values()].map(e => e.bridge.diagnostics)});
        },
        entry(scene) { return entries.get(scene) ?? null; },
        render({scene, camera}, frame) {
          const entry = entries.get(scene);
          if (!entry) fail('PREPARE', 'Scene has not been prepared');
          entry.bridge.render(camera, frame);
          entry.lastDiagnostics = entry.bridge.diagnostics;
          return entry;
        },
        async prepare() {
          for (const [scene, key] of owner._wanted) {
            const clipping = needsClipping(scene);
            const previous = entries.get(scene);
            if (previous && previous.key === key && (previous.clipping || !clipping) && !previous.bridge.failed) {
              await previous.bridge.prepare();
              continue;
            }
            // A changed feature profile selects different bridge pipelines.
            // Retire the old owner only after the replacement is published.
            const background = scene.background && !scene.background.isColor;
            const bridge = await createGpuThreeScene(device, scene, {
              ...sceneLimits, three: THREE, signal, maxBindings,
              renderer: {...attachments, instancing: true, renderBundles: true, maxDraws},
              textureTransforms: true, alphaMaps: true,
              fog: scene.fog ? {} : null, environment: scene.environment ? {} : null,
              background: background ? {} : null,
              shadow: owner.shadowMap.enabled ? {} : null,
              clipping: clipping || previous?.clipping ? clippingControls : null,
            });
            if (disposed) { bridge.dispose(); return; }
            entries.set(scene, {bridge, key, clipping: clipping || !!previous?.clipping, lastDiagnostics: null});
            previous?.bridge.dispose();
          }
        },
        async whenIdle() { await Promise.all([...entries.values()].map(e => e.bridge.whenIdle())); },
        dispose() {
          if (disposed) return;
          disposed = true;
          for (const e of entries.values()) e.bridge.dispose();
          entries.clear();
        },
      };
      owner._dispatcher = dispatcher;
      return dispatcher;
    };
  }

  class WebGPURenderer {
    constructor(parameters = {}) {
      if (!parameters || typeof parameters !== 'object') fail('OPTIONS', 'Expected renderer parameters');
      for (const key of Object.keys(parameters)) if (!PARAMETERS.has(key)) fail('OPTIONS', `Unknown renderer parameter: ${key}`);
      const {
        canvas = null, antialias = false, alpha = true, depth = true, stencil = false, samples = 0,
        forceWebGL = false, logarithmicDepthBuffer = false, reversedDepthBuffer = false,
      } = parameters;
      // Route selection precedes construction and any canvas binding: a WebGL
      // request (including a runtime value such as `forceWebGL: !api.webgpu`)
      // belongs to the exact-backend route. With the pinned upstream renderer
      // supplied, construct it unchanged; it is never this new backend.
      if (forceWebGL || parameters.context) {
        if (exactBackend) return new exactBackend(parameters);
        fail('ROUTE', 'forceWebGL or a supplied context selects the exact backend route, not this renderer');
      }
      this.isRenderer = true;
      this.isWebGPURenderer = true;
      this.isF3DRenderer = true;
      this.domElement = canvas ?? THREE.createCanvasElement();
      this.autoClear = true;
      this.autoClearColor = true;
      this.autoClearDepth = true;
      this.autoClearStencil = true;
      this.alpha = alpha;
      this.depth = depth;
      this.stencil = stencil;
      this.antialias = antialias;
      this.samples = samples;
      this.logarithmicDepthBuffer = logarithmicDepthBuffer;
      this.reversedDepthBuffer = reversedDepthBuffer;
      this.outputColorSpace = SRGBColorSpace;
      this.toneMapping = NoToneMapping;
      this.toneMappingExposure = 1;
      this.sortObjects = true;
      this.shadowMap = {enabled: false, transmitted: false, type: THREE.PCFShadowMap};
      this.xr = {enabled: false, isPresenting: false, getEnvironmentBlendMode: () => undefined};
      this.info = {
        autoReset: true, calls: 0, frame: 0,
        render: {calls: 0, frameCalls: 0, drawCalls: 0, triangles: 0, points: 0, lines: 0, timestamp: 0},
        compute: {calls: 0, frameCalls: 0, timestamp: 0},
        // Source Info shape. Filled from this backend's owned residency (geometry,
        // instance/deformation streams, draw/material buffers); fields this
        // backend does not track stay zero.
        memory: Object.fromEntries(['attributes', 'attributesSize', 'geometries', 'indexAttributes', 'indexAttributesSize',
          'indirectStorageAttributes', 'indirectStorageAttributesSize', 'programs', 'programsSize', 'readbackBuffers',
          'readbackBuffersSize', 'renderTargets', 'storageAttributes', 'storageAttributesSize', 'textures', 'texturesSize',
          'uniformBuffers', 'uniformBuffersSize', 'total'].map(key => [key, 0])),
        // F3D's own namespace: native submissions are not source draw counts.
        f3d: {route: 'general-webgpu', deferredRenders: 0, presentedRenders: 0, nativeDrawCalls: 0, preparations: 0},
        reset: () => { this.info.render.drawCalls = 0; this.info.render.frameCalls = 0; this.info.compute.frameCalls = 0; },
        dispose: () => {},
      };
      this._parameters = Object.freeze({...parameters});
      this._clearColor = new THREE.Color(0, 0, 0);
      this._clearAlpha = alpha === true ? 0 : 1;
      this._clearDepth = 1;
      this._clearStencil = 0;
      this._pixelRatio = 1;
      this._width = this.domElement.width;
      this._height = this.domElement.height;
      this._renderTarget = null;
      this._initialized = false;
      this._initPromise = null;
      this._session = null;
      this._dispatcher = null;
      this._hdr = false;
      this._wanted = new Map();
      this._pending = [];
      this._drain = null;
      this._deferredError = null;
      this._animationCallback = null;
      this._animationHandle = null;
      this._disposed = false;
      this._renderUid = 0;
      // Tooling surface (e.g. the r186 Inspector addon). This backend records no
      // GPU timestamps: hasTimestamp stays false and no timing is reported.
      this._backend = {isWebGPUBackend: true, isF3DBackend: true, device: null, trackTimestamp: false, hasTimestamp: false,
        hasTimestampQuery: () => false, getTimestampUID: () => null, delete() {}};
      // The source frame clock (Animation advances it once per loop tick).
      this._nodes = {nodeFrame: typeof THREE.NodeFrame === 'function' ? new THREE.NodeFrame() : {frameId: 0, time: 0, deltaTime: 0, update() { this.frameId++; }}};
      this._inspector = typeof THREE.InspectorBase === 'function' ? new THREE.InspectorBase() : inspectorStub();
      this._inspector.setRenderer(this);
    }
    set inspector(value) {
      this._inspector?.setRenderer(null);
      this._inspector = value;
      this._inspector.setRenderer(this);
    }
    get inspector() { return this._inspector; }
    hasFeature(name) { return !!this._session?.device?.features?.has(name); }
    async hasFeatureAsync(name) { await this.init(); return this.hasFeature(name); }

    get initialized() { return this._initialized; }
    hasInitialized() { return this._initialized; }
    get coordinateSystem() { return THREE.WebGPUCoordinateSystem; }
    get backend() { this._backend.device = this._session?.device ?? null; return this._backend; }

    init() {
      if (this._initPromise) return this._initPromise;
      this._initPromise = this._init();
      return this._initPromise;
    }
    async _init() {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this.logarithmicDepthBuffer || this.reversedDepthBuffer) fail('UNSUPPORTED', 'Logarithmic and reversed depth buffers are not admitted');
      if (this.stencil) fail('UNSUPPORTED', 'Stencil framebuffers are not admitted on the default canvas');
      const tone = this._toneMapping();
      const p = this._parameters;
      const sampleCount = this.antialias || this.samples > 0 ? 4 : 1;
      const depthFormat = this.depth ? 'depth24plus' : null;
      const common = {
        ...(p.device ? {device: p.device} : {}),
        ...(p.powerPreference ? {powerPreference: p.powerPreference} : {}),
        ...(p.requiredLimits ? {requiredLimits: p.requiredLimits} : {}),
        ...(p.requiredFeatures ? {requiredFeatures: p.requiredFeatures} : {}),
        ...(defaultGpu ? {gpu: defaultGpu} : {}),
      };
      const [width, height] = this._bufferSize();
      // Tone mapping needs the linear HDR target plus a whole-image output pass;
      // the direct path renders straight into the sRGB presentation view.
      this._hdr = tone !== 'none';
      this._session = this._hdr
        ? await createGpuHdrCanvasRenderer(this.domElement, createDispatcher(this), {...common,
          target: {width, height}, renderTarget: {depthFormat, sampleCount},
          output: {toneMapping: tone, exposure: this.toneMappingExposure}})
        : await createGpuCanvasRenderer(this.domElement, createDispatcher(this), {...common, lazyAttachments: true,
          target: {width, height, depthFormat, sampleCount, alphaMode: this.alpha ? 'premultiplied' : 'opaque'}});
      if (this._disposed) { this._session.dispose(); fail('DISPOSED', 'Renderer was disposed during initialization'); }
      this._initialized = true;
      return this;
    }

    _toneMapping() {
      const name = TONE_MAPPINGS.get(this.toneMapping);
      if (name === undefined) fail('UNSUPPORTED', `Tone mapping ${this.toneMapping} has no admitted output pass`);
      return name;
    }
    _bufferSize() {
      return [Math.floor(this._width * this._pixelRatio), Math.floor(this._height * this._pixelRatio)];
    }
    _resize() {
      const [w, h] = this._bufferSize();
      if (this._session) this._session.resize(w, h);
      else { this.domElement.width = w; this.domElement.height = h; }
    }

    // ---- size and clear state (source-equivalent semantics) ----
    getPixelRatio() { return this._pixelRatio; }
    setPixelRatio(value = 1) {
      if (this._pixelRatio === value) return;
      this._pixelRatio = value;
      this.setSize(this._width, this._height, false);
    }
    getSize(target) { return target.set(this._width, this._height); }
    getDrawingBufferSize(target) { return target.set(...this._bufferSize()); }
    setSize(width, height, updateStyle = true) {
      this._width = width;
      this._height = height;
      this._resize();
      if (updateStyle === true && this.domElement.style) {
        this.domElement.style.width = width + 'px';
        this.domElement.style.height = height + 'px';
      }
    }
    setDrawingBufferSize(width, height, pixelRatio) {
      this._width = width;
      this._height = height;
      this._pixelRatio = pixelRatio;
      this._resize();
    }
    getClearColor(target) { target.copy(this._clearColor); if ('a' in target) target.a = this._clearAlpha; return target; }
    setClearColor(color, alpha = 1) { this._clearColor.set(color); this._clearAlpha = alpha; }
    getClearAlpha() { return this._clearAlpha; }
    setClearAlpha(alpha) { this._clearAlpha = alpha; }
    getClearDepth() { return this._clearDepth; }
    setClearDepth(depth) { this._clearDepth = depth; }
    getClearStencil() { return this._clearStencil; }
    setClearStencil(stencil) { this._clearStencil = stencil; }
    getRenderTarget() { return this._renderTarget; }
    setRenderTarget(renderTarget) {
      if (renderTarget !== null) fail('UNSUPPORTED', 'Offscreen source render targets are not admitted by this renderer yet');
      this._renderTarget = null;
    }
    getContext() { return this._initialized ? this.domElement.getContext('webgpu') : null; }

    // ---- frames ----
    _frame(scene) {
      if (this._renderTarget !== null) fail('UNSUPPORTED', 'Offscreen source render targets are not admitted');
      if (this.xr.enabled && this.xr.isPresenting) fail('UNSUPPORTED', 'XR presentation is not admitted');
      if (this.outputColorSpace !== SRGBColorSpace) fail('UNSUPPORTED', 'Only sRGB canvas output is admitted');
      const tone = this._toneMapping();
      if (!this._hdr && tone !== 'none') fail('UNSUPPORTED', 'Enable tone mapping before init(); the session output path is fixed at initialization');
      const clear = this.autoClear === true;
      const color = this._clearColor, a = this._clearAlpha;
      // Source Background semantics: the renderer clear color is premultiplied
      // for alpha canvases; an opaque canvas ignores alpha.
      const premultiply = this.alpha === true;
      const frame = {
        loadOp: clear && this.autoClearColor ? 'clear' : 'load',
        depthLoadOp: clear && this.autoClearDepth ? 'clear' : 'load',
        clearColor: premultiply ? [color.r * a, color.g * a, color.b * a, a] : [color.r, color.g, color.b, 1],
        clearDepth: this._clearDepth,
      };
      if (this._hdr) {
        // The tone-mapped output pass presents an opaque canvas. Only a scene
        // background or an opaque clear makes that equal to the source result.
        if (this.alpha === true && a < 1 && frame.loadOp === 'clear' && !scene.background)
          fail('UNSUPPORTED', 'Transparent canvas clears are not admitted with tone mapping; set an opaque clear alpha or scene background');
        frame.clearColor[3] = 1;
        frame.output = {toneMapping: tone, exposure: this.toneMappingExposure};
      }
      return frame;
    }
    render(scene, camera) {
      if (this._initialized === false) throw new Error(NOT_INITIALIZED);
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (!(scene instanceof THREE.Object3D) || !(camera instanceof THREE.Camera)) fail('SOURCE', 'Expected a source scene and camera');
      if (!scene.isScene) fail('UNSUPPORTED', 'Rendering a non-Scene root object is not admitted');
      this.info.calls++;
      this.info.render.calls++;
      this.info.render.frameCalls++;
      // Source timestamp UID shape `<call id>:f<frame>` (Inspector groups by call id).
      this._inspector.beginRender(`render:${++this._renderUid}:f${this.info.frame}`, scene, camera, null);
      try { return this._renderFrame(scene, camera); } finally { this._inspector.finishRender(); }
    }
    _renderFrame(scene, camera) {
      const request = {scene, camera, frame: this._frame(scene), key: sceneKey(this, scene)};
      this._wanted.set(scene, request.key);
      const entry = this._dispatcher?.entry(scene);
      if (this._drain || !entry || entry.key !== request.key) return this._defer(request);
      try {
        this._submit(request);
      } catch (error) {
        if (isPrepareBoundary(error) || (error?.code === 'THREE_SCENE_MATERIAL' && !entry.clipping && needsClipping(scene)))
          return this._defer(request);
        throw error;
      }
    }
    _submit({scene, camera, frame}) {
      this._session.render({scene, camera}, frame);
      const d = this._dispatcher.entry(scene)?.lastDiagnostics;
      if (d) {
        this.info.render.drawCalls += d.sourceDraws;
        this.info.f3d.nativeDrawCalls = d.drawCalls;
        const memory = this.info.memory;
        memory.geometries = d.geometryCount;
        memory.attributesSize = d.geometryBytes + d.instanceBytes + d.deformationBytes;
        memory.uniformBuffersSize = d.rendererBytes;
        memory.total = memory.attributesSize + memory.uniformBuffersSize;
      }
      this.info.f3d.presentedRenders++;
    }
    _defer(request) {
      this.info.f3d.deferredRenders++;
      // A color clear hides every earlier request on this canvas.
      if (request.frame.loadOp === 'clear') this._pending = [request];
      else if (this._pending.length >= MAX_PENDING) fail('LIMIT', 'Too many layered render requests are waiting for preparation');
      else this._pending.push(request);
      this._drain ??= this._drainPending().finally(() => { this._drain = null; });
      this._drain.catch(error => { this._deferredError ??= error; });
    }
    async _drainPending() {
      for (let rounds = 0; this._pending.length; rounds++) {
        if (rounds > 8) fail('PREPARE', 'Source structure kept changing during preparation');
        await this._session.prepare();
        this.info.f3d.preparations++;
        if (this._disposed) return;
        const batch = this._pending;
        this._pending = [];
        for (let i = 0; i < batch.length; i++) {
          const request = batch[i];
          const entry = this._dispatcher.entry(request.scene);
          const key = sceneKey(this, request.scene);
          this._wanted.set(request.scene, key);
          try {
            if (!entry || entry.key !== key) fail('PREPARE', 'Scene profile changed during preparation');
            this._submit(request);
          } catch (error) {
            if (!isPrepareBoundary(error)) throw error;
            // Keep source order: this request and every later one wait again.
            this._pending = [...batch.slice(i), ...this._pending];
            break;
          }
        }
      }
    }
    async renderAsync(scene, camera) {
      await this.init();
      await this.compileAsync(scene, camera);
      this.render(scene, camera);
    }
    async compileAsync(scene, camera) {
      await this.init();
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (camera !== undefined && !(camera instanceof THREE.Camera)) fail('SOURCE', 'Expected a source camera');
      this._wanted.set(scene, sceneKey(this, scene));
      while (this._drain) await this._drain;
      await this._session.prepare();
      this.info.f3d.preparations++;
    }
    async waitForGPU() { if (this._session) await this._session.whenIdle(); }

    // ---- animation loop (source Animation semantics: callback(time, xrFrame)) ----
    async setAnimationLoop(callback) {
      if (this._initialized === false && callback !== null) await this.init();
      const host = this.domElement.ownerDocument?.defaultView ?? globalThis;
      if (this._animationHandle !== null) { host.cancelAnimationFrame?.(this._animationHandle); this._animationHandle = null; }
      if (this._inspector.isRunning) this._inspector.finish();
      this._animationCallback = callback;
      if (callback === null || this._disposed) return;
      const tick = time => {
        if (this._animationCallback !== callback || this._disposed) return;
        this._animationHandle = host.requestAnimationFrame(tick);
        // Source Animation order: finish the previous inspected frame, then begin.
        if (this._inspector.isRunning) this._inspector.finish();
        if (this.info.autoReset === true) this.info.reset();
        this._nodes.nodeFrame.update();
        this._renderUid = 0; // stable per-frame call ids, like source render contexts
        this.info.frame = this._nodes.nodeFrame.frameId;
        this._inspector.begin();
        callback(time, null);
      };
      this._animationHandle = host.requestAnimationFrame(tick);
    }
    getAnimationLoop() { return this._animationCallback; }

    dispose() {
      if (this._disposed) return;
      this.setAnimationLoop(null);
      this._disposed = true;
      this._pending = [];
      this._inspector.dispose?.();
      if (this._session && !this._drain) this._session.dispose();
      else if (this._drain) this._drain.finally(() => this._session?.dispose()).catch(() => {});
    }
  }
  return WebGPURenderer;
}
