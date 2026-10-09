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
 * - Unsupported source state (advanced render targets, XR presentation, custom tone
 *   mapping, logarithmic/reversed depth, stencil buffers, non-sRGB output,
 *   transparent clears with tone mapping) throws `F3DRendererError` rather than
 *   silently rendering something else.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {createThreePassSnapshots} from './three_pass_snapshot.mjs';
import {createThreeProgramSupport} from './three_program.mjs';
import {createThreeProgramPMREM} from './three_program_pmrem.mjs';
import {createThreeProgramShadows} from './three_program_shadows.mjs';
import {createThreeProgramTransmission} from './three_program_transmission.mjs';
// r186 src/renderers/webgpu/utils/WebGPUConstants.js GPUFeatureName values.
const R186_GPU_FEATURES = ['core-features-and-limits', 'depth-clip-control', 'depth32float-stencil8', 'texture-compression-bc',
  'texture-compression-bc-sliced-3d', 'texture-compression-etc2', 'texture-compression-astc', 'texture-compression-astc-sliced-3d',
  'timestamp-query', 'indirect-first-instance', 'shader-f16', 'rg11b10ufloat-renderable', 'bgra8unorm-storage', 'float32-filterable',
  'float32-blendable', 'clip-distances', 'dual-source-blending', 'subgroups', 'texture-formats-tier1', 'texture-formats-tier2',
  'texture-compression-s3tc', 'texture-compression-etc1'];

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
const PARAMETERS = new Set(['outputType', 'canvas', 'antialias', 'alpha', 'depth', 'stencil', 'samples', 'forceWebGL',
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
    const isScene = scene.isScene === true;
    const background = isScene && scene.background !== null && scene.background !== undefined && !scene.background.isColor;
    return `${isScene && scene.fog ? 1 : 0}${isScene && scene.environment ? 1 : 0}${background ? 1 : 0}${renderer.shadowMap.enabled ? 1 : 0}${renderer._needsGlobalClipping?.() ? 1 : 0}`;
  }
  /** Preparation-time traversal for options that depend on materials. */
  function needsClipping(scene) {
    let clipping = false;
    scene.traverse(object => {
      const list = Array.isArray(object.material) ? object.material : object.material ? [object.material] : [];
      if (list.some(m => m?.clippingPlanes?.length)) clipping = true;
    });
    if (scene.isScene === true && scene.overrideMaterial?.clippingPlanes?.length) clipping = true;
    return clipping;
  }
  // r186 WebGPURenderer applies material clipping planes without a renderer-level
  // opt-in; there are no renderer-global clipping planes on this path.
  const clippingControls = Object.freeze({planes: Object.freeze([]), localClippingEnabled: true});

  /** Multi-scene renderer handed to the owned canvas session. */
  function createDispatcher(owner, destination = owner) {
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
          // Only the current request is installed while preparation can await.
          // Other queued fullscreen roots may still hold an earlier pass state.
          const root = destination._preparationRoot;
          const wanted = root ? [[root, destination._wanted.get(root)]] : destination._wanted;
          for (const [scene, key] of wanted) {
            const clipping = needsClipping(scene) || owner._needsGlobalClipping?.() === true;
            const previous = entries.get(scene);
            if (previous && previous.key === key && (previous.clipping || !clipping) && !previous.bridge.failed) {
              await previous.bridge.prepare();
              continue;
            }
            // A changed feature profile selects different bridge pipelines.
            // Retire the old owner only after the replacement is published.
            const sourceScene = scene.isScene === true;
            const background = sourceScene && scene.background && !scene.background.isColor;
            // Options and the entry key describe the same live profile.
            const liveKey = sceneKey(owner, scene);
            let bridge;
            try { bridge = await createGpuThreeScene(device, scene, {
              // Mesh-count caps follow the binding capacity; the byte budgets
              // (deformation/instance) stay the scene owner's defaults and bound memory.
              maxDeformedMeshes: Math.min(maxBindings, 4096), maxInstanceMeshes: Math.min(maxBindings, 4096),
              // Source renderers have no texture budget: bound residency by a
              // device-scale default (1 GiB) rather than the bridge's 128 MiB.
              texture: {maxTextureBytes: 1024 * 1024 * 1024},
              ...sceneLimits, three: THREE, signal, maxBindings,
              textures: owner._targets.bindingsFor(destination === owner ? null : destination.target, sceneLimits.textures),
              renderer: {...attachments, instancing: true, renderBundles: true, maxDraws,
                ...(destination !== owner || owner._hdr || !owner._shaderEncodedOutput ? {} : {outputTransfer: 'srgb'})},
              textureTransforms: true, alphaMaps: true,
              // ShaderMaterial programs (WebGLRenderer semantics) on the WebGL surface.
              program: owner._shaderEncodedOutput && (destination !== owner || !owner._hdr) ? destination._programSupport ??= createThreeProgramSupport({three: THREE, pmrem: createThreeProgramPMREM, shadows: createThreeProgramShadows, transmission: createThreeProgramTransmission,
                state: () => ({toneMapping: owner.toneMapping, toneMappingExposure: owner.toneMappingExposure, outputColorSpace: owner.outputColorSpace,
                  pixelRatio: owner._pixelRatio, height: owner._height, shadowMap: owner.shadowMap, shadowMapType: owner.shadowMap.type, renderer: owner,
                  floatLinear: device.features?.has?.('float32-filterable') === true, ...owner._programFrame,
                  // Public targets keep top-down native storage (the target owner
                  // provides UV-oriented sampling). Do not use the compiler's
                  // renderTarget:true/GL-row profile reserved for shadow/PMREM maps.
                  // Native sRGB attachments perform their own transfer function.
                  // Keep source point-material size/scale uniforms tied to the
                  // renderer's pixel ratio and logical height, even offscreen.
                  ...(destination === owner ? {} : {toneMapping: NoToneMapping, outputColorSpace: THREE.LinearSRGBColorSpace})})}) : null,
              fog: sourceScene && scene.fog ? {} : null, environment: sourceScene && scene.environment ? {} : null,
              background: background ? {} : null,
              shadow: owner.shadowMap.enabled ? {} : null,
              clipping: clipping || previous?.clipping ? owner._clippingControls ?? clippingControls : null,
            }); } catch (error) {
              // The scene gained/lost fog, environment or background while the
              // bridge was being created: the next preparation round rebuilds it.
              if (error?.code === 'THREE_SCENE_SCENE' && sceneKey(owner, scene) !== liveKey) continue;
              throw error;
            }
            if (disposed) { bridge.dispose(); return; }
            entries.set(scene, {bridge, key: liveKey, clipping: clipping || !!previous?.clipping, lastDiagnostics: null});
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
      destination._dispatcher = dispatcher;
      return dispatcher;
    };
  }

  class WebGPURenderer {
    constructor(parameters = {}) {
      if (!parameters || typeof parameters !== 'object') fail('OPTIONS', 'Expected renderer parameters');
      for (const key of Object.keys(parameters)) if (!(this._parameterNames?.() ?? PARAMETERS).has(key)) fail('OPTIONS', `Unknown renderer parameter: ${key}`);
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
      // The source node library instance (registrations such as addLight are
      // recorded on the real object). The F3D bridge evaluates admitted light
      // and material classes itself; unadmitted classes still fail explicitly.
      if (typeof THREE.StandardNodeLibrary === 'function') this.library = new THREE.StandardNodeLibrary();
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
        f3d: {route: 'general-webgpu', deferredRenders: 0, presentedRenders: 0, offscreenRenders: 0, nativeDrawCalls: 0, preparations: 0},
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
      // Source CanvasTarget viewport/scissor: logical units, top-left origin on
      // WebGPU, scaled by the pixel ratio at render time.
      this._viewport = new THREE.Vector4(0, 0, this._width, this._height);
      this._viewportDepth = [0, 1];
      this._scissor = new THREE.Vector4(0, 0, this._width, this._height);
      this._scissorTest = false;
      this._clearScene = null;
      this._renderTarget = null;
      this._targetLifetime = new AbortController();
      this._targets = createThreeRenderTargets(THREE, {getDevice: () => this._device});
      this._passSnapshots = createThreePassSnapshots(THREE);
      this._initialized = false;
      this._initPromise = null;
      this._session = null;
      this._dispatcher = null;
      this._device = null;
      this._ownsDevice = false;
      this._rebuild = false;
      this._preparing = null;
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
      const p = this._parameters;
      if (p.outputType !== undefined && p.outputType !== THREE.UnsignedByteType)
        fail('UNSUPPORTED', 'Extended-range canvas output types are not admitted');
      // The renderer owns one device for its lifetime so the output path can be
      // rebuilt (tone mapping on/off) without losing it. A supplied device is borrowed.
      if (p.device) this._device = p.device;
      else {
        const gpu = defaultGpu ?? globalThis.navigator?.gpu;
        if (!gpu || typeof gpu.requestAdapter !== 'function') fail('UNAVAILABLE', 'WebGPU is unavailable in this host');
        const adapter = await gpu.requestAdapter(p.powerPreference ? {powerPreference: p.powerPreference} : {});
        if (!adapter) fail('UNAVAILABLE', 'No WebGPU adapter is available');
        // As r186 WebGPUBackend: request every GPUFeatureName the adapter supports.
        const requiredFeatures = [...new Set([...R186_GPU_FEATURES.filter(f => adapter.features?.has?.(f)), ...(p.requiredFeatures ?? [])])];
        this._device = await adapter.requestDevice({requiredFeatures, requiredLimits: p.requiredLimits ?? {}});
        this._ownsDevice = true;
      }
      this._preferredFormat = (defaultGpu ?? globalThis.navigator?.gpu)?.getPreferredCanvasFormat?.();
      await this._createSession(this._wantsHdr(this._toneMapping()));
      if (this._disposed) { this._release(); fail('DISPOSED', 'Renderer was disposed during initialization'); }
      this._initialized = true;
      return this;
    }
    /** Tone mapping needs the linear HDR target plus a whole-image output pass
     * (the source renders through a half-float framebuffer target in that case);
     * otherwise frames go straight into the sRGB presentation view. */
    async _createSession(hdr) {
      const sampleCount = this.antialias || this.samples > 0 ? 4 : 1;
      const depthFormat = this.depth ? 'depth24plus' : null;
      const [width, height] = this._bufferSize();
      const common = {device: this._device};
      const format = this._preferredFormat ? {format: this._preferredFormat} : {};
      this._hdr = hdr;
      this._dispatcher = null;
      this._session = hdr
        ? await createGpuHdrCanvasRenderer(this.domElement, createDispatcher(this), {...common,
          target: {width, height, ...format, alphaMode: this._compositedAlpha() ? 'premultiplied' : 'opaque'}, renderTarget: {depthFormat, sampleCount},
          output: {toneMapping: this._toneMapping(), exposure: this.toneMappingExposure}})
        : await createGpuCanvasRenderer(this.domElement, createDispatcher(this), {...common, lazyAttachments: true,
          target: {width, height, depthFormat, sampleCount, ...format, alphaMode: this._compositedAlpha() ? 'premultiplied' : 'opaque',
            ...(this._shaderEncodedOutput ? {srgbView: false} : {})}});
    }
    _release() {
      this._targets.dispose();
      this._session?.dispose();
      this._session = null;
      this._wanted.clear();
      this._passSnapshots.dispose();
      if (this._ownsDevice) { this._ownsDevice = false; this._device?.destroy(); }
    }

    /** The whole-image tone-mapping output pass. WebGLRenderer (shader-encoded
     * output) tone-maps inside each material's program instead, unless a scene
     * needed a draw the program route cannot take (sticky fallback). */
    _wantsHdr(tone) { return tone !== 'none' && (!this._shaderEncodedOutput || this._hdrFallback === true); }
    /** Whether the canvas composites with the page by alpha. WebGLRenderer (r186)
     * always creates an alpha:true context; its `alpha` option only sets the
     * default clear alpha, so shader-written alpha and setClearAlpha show. */
    _compositedAlpha() { return this.alpha === true || this._contextAlpha === true; }
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
      this.setViewport(0, 0, width, height);
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
      this.setViewport(0, 0, width, height);
    }
    getViewport(target) { return target.copy(this._viewport); }
    setViewport(x, y, width, height, minDepth = 0, maxDepth = 1) {
      if (x.isVector4) this._viewport.copy(x); else this._viewport.set(x, y, width, height);
      this._viewportDepth = [minDepth, maxDepth];
    }
    getScissor(target) { return target.copy(this._scissor); }
    setScissor(x, y, width, height) {
      if (x.isVector4) this._scissor.copy(x); else this._scissor.set(x, y, width, height);
    }
    getScissorTest() { return this._scissorTest; }
    setScissorTest(value) { this._scissorTest = value; }
    // WebGPU's maximum sampler anisotropy, as the source WebGPU backend reports.
    getMaxAnisotropy() { return 16; }
    getActiveCubeFace() { return 0; }
    getActiveMipmapLevel() { return 0; }
    /** Source Renderer.clear(): a clear-only pass on the canvas with the current
     * clear color/depth, honoring viewport/scissor state like other passes. */
    clear(color = true, depth = true) {
      if (this._initialized === false) return;
      this._clearScene ??= new THREE.Scene();
      this._clearCamera ??= new THREE.PerspectiveCamera();
      this._renderFrame(this._clearScene, this._clearCamera, {color, depth});
    }
    clearColor() { this.clear(true, false); }
    clearDepth() { this.clear(false, true); }
    clearStencil() {}
    async clearAsync(color = true, depth = true) { await this.init(); this.clear(color, depth); }
    async clearColorAsync() { await this.clearAsync(true, false); }
    async clearDepthAsync() { await this.clearAsync(false, true); }
    async clearStencilAsync() {}
    getClearColor(target) { target.copy(this._clearColor); if ('a' in target) target.a = this._clearAlpha; return target; }
    setClearColor(color, alpha = 1) { this._clearColor.set(color); this._clearAlpha = alpha; }
    getClearAlpha() { return this._clearAlpha; }
    setClearAlpha(alpha) { this._clearAlpha = alpha; }
    getClearDepth() { return this._clearDepth; }
    setClearDepth(depth) { this._clearDepth = depth; }
    getClearStencil() { return this._clearStencil; }
    setClearStencil(stencil) { this._clearStencil = stencil; }
    getRenderTarget() { return this._renderTarget; }
    setRenderTarget(renderTarget, activeCubeFace = 0, activeMipmapLevel = 0) {
      if (activeCubeFace !== 0 || activeMipmapLevel !== 0) fail('UNSUPPORTED', 'Cube faces and target mip levels are not admitted');
      if (renderTarget !== null) this._targets.capture(renderTarget);
      this._renderTarget = renderTarget;
    }
    readRenderTargetPixelsAsync(target, x, y, width, height, textureIndex = 0, faceIndex = 0) {
      if (textureIndex !== 0 || faceIndex !== 0) return Promise.reject(new F3DRendererError('UNSUPPORTED', 'Only the 2D color attachment is readable'));
      return this._readTargetPixels(target, {x, y, width, height});
    }
    _readTargetPixels(target, options) {
      try {
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        const destination = this._targets.capture(target);
        const issue = () => {
          this._targets.ensure(destination);
          return readGpuTargetPixels(destination.session, {...options, signal: this._targetLifetime.signal});
        };
        // The helper submits its copy before returning the mapping promise.
        if (this._initialized && !this._drain && !this._preparing) return issue();
        return new Promise((resolve, reject) => this._enqueue({readback: true, issue, resolve, reject}));
      } catch (error) { return Promise.reject(error); }
    }
    getContext() { return this._initialized ? this.domElement.getContext('webgpu') : null; }

    // ---- frames ----
    _frame(scene, clearOnly = null, target = this._renderTarget) {
      if (target) inspectThreeRenderTarget(target, THREE);
      if (this.xr.enabled && this.xr.isPresenting) fail('UNSUPPORTED', 'XR presentation is not admitted');
      if (!target && this.outputColorSpace !== SRGBColorSpace) fail('UNSUPPORTED', 'Only sRGB canvas output is admitted');
      const tone = this._toneMapping();
      // Switching between direct and tone-mapped output rebuilds the canvas
      // session (same device) at the next preparation boundary.
      if (!target) this._rebuild = this._wantsHdr(tone) !== this._hdr;
      const hdr = !target && this._wantsHdr(tone);
      const clear = this.autoClear === true;
      const color = this._clearColor, a = this._clearAlpha;
      // Source Background semantics: the renderer clear color is premultiplied
      // for alpha canvases; an opaque canvas ignores alpha.
      const premultiply = this._compositedAlpha() && (!target || !this._bottomLeftOrigin);
      const frame = {
        loadOp: (clearOnly ? clearOnly.color : clear && this.autoClearColor) ? 'clear' : 'load',
        depthLoadOp: (clearOnly ? clearOnly.depth : clear && this.autoClearDepth) ? 'clear' : 'load',
        clearColor: premultiply ? [color.r * a, color.g * a, color.b * a, a] : [color.r, color.g, color.b, target ? a : 1],
      };
      // Shader-encoded output stores sRGB values: clear with the encoded color
      // (source WebGL Background converts the clear color to the output space).
      if (!target && this._shaderEncodedOutput && !hdr) {
        const encoded = [color.r, color.g, color.b].map(c => c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 0.41666) - 0.055);
        frame.clearColor = premultiply ? [...encoded.map(c => c * a), a] : [...encoded, 1];
      }
      Object.assign(frame, {
        clearDepth: this._clearDepth,
      });
      const pr = target ? 1 : this._pixelRatio;
      const [bufferWidth, bufferHeight] = target ? [target.width, target.height] : this._bufferSize();
      const v = target ? target.viewport : this._viewport, px = this._pixelRound ?? Math.floor;
      const vx = px(v.x * pr), vw = px(v.z * pr), vh = px(v.w * pr);
      // Program gl_FragCoord (framebuffer) and point-sprite sizes (viewport).
      if (this._shaderEncodedOutput && !hdr) frame.targetSize = [bufferWidth, bufferHeight, vw, vh];
      // r186 _currentViewport size: the transmission target follows it.
      if (this._shaderEncodedOutput) frame.viewportSize = [vw, vh];
      // WebGL viewports/scissors use a bottom-left origin; WebGPU's is top-left.
      const vy = this._bottomLeftOrigin ? bufferHeight - px(v.y * pr) - vh : px(v.y * pr);
      const [minDepth, maxDepth] = target ? [0, 1] : this._viewportDepth;
      if (vx !== 0 || vy !== 0 || vw !== bufferWidth || vh !== bufferHeight || minDepth !== 0 || maxDepth !== 1)
        frame.viewport = [vx, vy, vw, vh, minDepth, maxDepth];
      if (target ? target.scissorTest : this._scissorTest) {
        // Source clamping: non-negative and inside the drawing buffer.
        const sc = target ? target.scissor : this._scissor, rawH = px(sc.w * pr);
        const rawY = this._bottomLeftOrigin ? bufferHeight - px(sc.y * pr) - rawH : px(sc.y * pr);
        const sx = Math.max(0, px(sc.x * pr)), sy = Math.max(0, rawY);
        // WebGPU source: clamp each component to >= 0, then fit inside the buffer.
        // WebGL: the GL scissor box is intersected with the buffer, so a negative
        // origin also shortens the box.
        const clip = this._bottomLeftOrigin;
        const sw = Math.max(0, Math.min(px(sc.z * pr) + (clip ? Math.min(0, px(sc.x * pr)) : 0), bufferWidth - sx)),
          sh = Math.max(0, Math.min(rawH + (clip ? Math.min(0, rawY) : 0), bufferHeight - sy));
        if (sx !== 0 || sy !== 0 || sw !== bufferWidth || sh !== bufferHeight) frame.scissor = [sx, sy, sw, sh];
      }
      if (hdr) {
        if (!this._compositedAlpha()) frame.clearColor[3] = 1;
        frame.output = {toneMapping: tone, exposure: this.toneMappingExposure};
      }
      return frame;
    }
    render(scene, camera) {
      if (this._initialized === false && !this._deferUntilInitialized) throw new Error(NOT_INITIALIZED);
      return this._renderCall(scene, camera);
    }
    _beforeRender() {}
    _renderCall(scene, camera) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (!(scene instanceof THREE.Object3D) || !(camera instanceof THREE.Camera)) fail('SOURCE', 'Expected a source scene and camera');
      this._beforeRender();
      this.info.calls++;
      this.info.render.calls++;
      this.info.render.frameCalls++;
      // Source timestamp UID shape `<call id>:f<frame>` (Inspector groups by call id).
      this._inspector.beginRender(`render:${++this._renderUid}:f${this.info.frame}`, scene, camera, this._renderTarget);
      try { return this._renderFrame(scene, camera); } finally { this._inspector.finishRender(); }
    }
    _makeRequest(scene, camera, clearOnly = null) {
      const target = this._renderTarget;
      const destination = target ? this._targets.capture(target) : this;
      const frame = this._frame(scene, clearOnly, target), key = sceneKey(this, scene);
      const c = this._clearColor, a = this._clearAlpha;
      // Small callback-free shader passes (FullScreenQuad) are private data views;
      // general scenes and hooked roots retain their established source path.
      const snapshot = this._shaderEncodedOutput ? this._passSnapshots.capture(scene, camera) : null;
      return {scene: snapshot?.root ?? scene, camera: snapshot?.camera ?? camera, snapshot, clearOnly, destination, frame,
        linearClearColor: this.alpha ? [c.r * a, c.g * a, c.b * a, a] : [c.r, c.g, c.b, 1],
        hdr: !target && this._wantsHdr(this._toneMapping()), key,
        programFrame: {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
          outputColorSpace: this.outputColorSpace, pixelRatio: this._pixelRatio, height: this._height}};
    }
    _renderFrame(scene, camera, clearOnly = null) {
      const request = this._makeRequest(scene, camera, clearOnly);
      return this._dispatchRequest(request);
    }
    _dispatchRequest(request) {
      const {scene, destination, snapshot} = request;
      destination._wanted.set(scene, request.key);
      const entry = destination._dispatcher?.entry(scene);
      if (this._drain || this._preparing || (destination === this && request.hdr !== this._hdr) || !entry || entry.key !== request.key)
        return this._defer(request);
      try { this._submit(request); snapshot?.release(); }
      catch (error) {
        if (isPrepareBoundary(error) || (error?.code === 'THREE_SCENE_MATERIAL' && !entry.clipping && needsClipping(scene)))
          return this._defer(request);
        snapshot?.release();
        throw error;
      }
    }
    _submit({scene, camera, frame, destination, programFrame, snapshot, deferred = false}) {
      snapshot?.install();
      const previous = this._programFrame;
      // Immediate source callbacks must still be able to change live program
      // state. Only deferred calls need the queued renderer-state snapshot.
      this._programFrame = deferred ? programFrame : undefined;
      try {
        if (destination === this) this._session.render({scene, camera}, frame);
        else this._targets.render(destination, attachments => destination._dispatcher.render({scene, camera}, {...frame, ...attachments}));
      } finally { this._programFrame = previous; }
      const d = destination._dispatcher.entry(scene)?.lastDiagnostics;
      if (d) {
        this.info.render.drawCalls += d.sourceDraws;
        this.info.f3d.nativeDrawCalls = d.drawCalls;
        const memory = this.info.memory;
        memory.geometries = d.geometryCount;
        memory.attributesSize = d.geometryBytes + d.instanceBytes + d.deformationBytes;
        memory.uniformBuffersSize = d.rendererBytes;
        memory.renderTargets = this._targets.size;
        memory.total = memory.attributesSize + memory.uniformBuffersSize + this._targets.allocatedBytes;
      }
      if (destination === this) this.info.f3d.presentedRenders++;
      else this.info.f3d.offscreenRenders++;
    }
    _defer(request) {
      request.deferred = true; this.info.f3d.deferredRenders++;
      try { this._enqueue(request); } catch (error) { request.snapshot?.release(); throw error; }
    }
    _enqueue(request) {
      // A later clear is NOT a license to discard an earlier target write,
      // source callback or readback. All destinations share this submission FIFO.
      if (this._pending.length >= MAX_PENDING) fail('LIMIT', 'Too many render operations are waiting for preparation');
      this._pending.push(request);
      this._startDrain();
    }
    _startDrain() {
      if (!this._drain) {
        this._drain = this._drainPending().catch(error => {
          for (const pending of this._pending) { pending.snapshot?.release(); pending.reject?.(error); }
          this._pending = [];
          this._deferredError ??= error;
          throw error;
        }).finally(() => {
          this._drain = null;
          if (this._pending.length && !this._disposed) this._startDrain();
        });
        this._drain.catch(() => {});
      }
    }
    async _drainPending() {
      await this.init();
      while (this._pending.length && !this._disposed) {
        const request = this._pending[0];
        if (request.readback) {
          // Queue the copy now but do not await mapping: subsequent draws may
          // submit once the copy has captured this version of the attachment.
          try { request.resolve(request.issue()); } catch (error) { request.reject(error); }
          this._pending.shift(); continue;
        }
        const {destination, scene} = request;
        for (let attempts = 0; ; attempts++) {
          if (attempts > 8) fail('PREPARE', 'Source structure kept changing during preparation');
          if (destination !== this) this._targets.assertCurrent(destination);
          destination._wanted.set(scene, sceneKey(this, scene));
          await this._prepare(destination, request);
          if (this._disposed) return;
          try {
            const entry = destination._dispatcher?.entry(scene);
            if (!entry || entry.key !== sceneKey(this, scene)) fail('PREPARE', 'Scene profile changed during preparation');
            this._submit(request); break;
          } catch (error) {
            if (!isPrepareBoundary(error)) throw error;
          }
        }
        request.snapshot?.release();
        this._pending.shift();
      }
    }
    async renderAsync(scene, camera) {
      // Enqueue at the call, before the first await: a following draw/readback
      // must not overtake this draw while initialization is still pending.
      this._renderCall(scene, camera);
      while (this._drain) await this._drain;
    }
    async compileAsync(scene, camera, targetScene = null) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (camera !== undefined && !(camera instanceof THREE.Camera)) fail('SOURCE', 'Expected a source camera');
      const compiled = targetScene ?? scene;
      if (!(compiled instanceof THREE.Object3D)) fail('SOURCE', 'Expected a source Object3D root');
      // Capture before initialization or another compile can yield. Preparation
      // and later draws address the same stable execution root.
      const request = this._makeRequest(compiled, camera);
      const {destination, snapshot} = request;
      try {
        await this.init();
        if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
        destination._wanted.set(request.scene, request.key);
        while (this._drain) await this._drain;
        await this._prepare(destination, request);
      } finally { snapshot?.release(); }
    }
    /** All scene preparations share one queue; targets never rebuild the canvas. */
    _prepare(destination = this, request = null) {
      const run = (this._preparing ?? Promise.resolve()).catch(() => {}).then(async () => {
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        const previous = this._programFrame, previousRoot = destination._preparationRoot;
        this._programFrame = request?.programFrame;
        destination._preparationRoot = request?.scene ?? null;
        try {
          request?.snapshot?.install();
          if (destination !== this) {
            this._targets.ensure(destination);
            if (!destination._dispatcher) createDispatcher(this, destination)(this._device, destination.storage.rendererOptions, {signal: destination.lifetime.signal});
            await destination._dispatcher.prepare();
            this._targets.assertCurrent(destination);
          } else {
            const tone = request ? TONE_MAPPINGS.get(request.programFrame.toneMapping) : this._toneMapping();
            const selectOutput = () => {
              const hdr = this._wantsHdr(tone);
              if (request) {
                request.hdr = hdr;
                // A preceding queued request may have activated sticky HDR
                // fallback. Keep this request's clear/exposure, not live state.
                if (hdr) {
                  request.frame.output = {toneMapping: tone, exposure: request.programFrame.toneMappingExposure};
                  request.frame.clearColor = request.linearClearColor.slice();
                  delete request.frame.targetSize;
                }
              }
              return hdr;
            };
            const hdr = selectOutput();
            if (hdr !== this._hdr) { this._session.dispose(); await this._createSession(hdr); }
            this._rebuild = false;
            try { await this._session.prepare(); }
            catch (error) {
              if (error?.code !== 'THREE_SCENE_TONE_MAPPING' || this._hdrFallback === true || globalThis.F3D_NO_FALLBACK === true) throw error;
              this._hdrFallback = true;
              this._session.dispose();
              await this._createSession(selectOutput());
              this._rebuild = false;
              await this._session.prepare();
            }
          }
          this.info.f3d.preparations++;
        } finally { this._programFrame = previous; destination._preparationRoot = previousRoot; }
      });
      this._preparing = run;
      run.finally(() => { if (this._preparing === run) this._preparing = null; }).catch(() => {});
      return run;
    }
    async waitForGPU() {
      while (this._drain) await this._drain;
      if (this._preparing) await this._preparing;
      if (this._session) await this._session.whenIdle();
      await this._targets.whenIdle();
    }

    // ---- animation loop (source Animation semantics: callback(time, xrFrame)) ----
    async setAnimationLoop(callback) {
      if (this._initialized === false && callback !== null) await this.init();
      this._setLoop(callback);
    }
    _setLoop(callback) {
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
      const stopped = new F3DRendererError('DISPOSED', 'Renderer is disposed');
      this._targetLifetime.abort(stopped);
      this._targets.dispose();
      for (const request of this._pending) { request.snapshot?.release(); request.reject?.(stopped); }
      this._pending = [];
      this._inspector.dispose?.();
      if (!this._drain) this._release();
      else this._drain.finally(() => this._release()).catch(() => {});
    }
  }
  return WebGPURenderer;
}

const WEBGL_PARAMETERS = new Set(['canvas', 'context', 'depth', 'stencil', 'alpha', 'antialias', 'premultipliedAlpha',
  'preserveDrawingBuffer', 'powerPreference', 'failIfMajorPerformanceCaveat', 'reversedDepthBuffer', 'outputBufferType',
  'precision', 'logarithmicDepthBuffer', 'device', 'requiredLimits', 'requiredFeatures']);

/** Drop-in r186 `WebGLRenderer` surface over the same new-WebGPU scene path.
 *
 * Route/ownership: general new WebGPU, retained-JS command preparation. The build
 * substitutes it only when the whole application has no GL escape (no context
 * access, extension/parameter queries or capability reads). It is not a GL
 * context and does not pretend to be one: getContext(), capabilities,
 * extensions, properties and state throw F3DRendererError.
 *
 * Contract differences from the source WebGL renderer, all explicit:
 * - Construction is synchronous, as in the source; device negotiation starts
 *   immediately. Frames requested before it completes (and structural edits)
 *   are deferred to the preparation boundary and submitted in order; the
 *   canvas keeps its previous image meanwhile. Callbacks are never skipped.
 * - Blending happens in linear space on a WebGPU sRGB view, whereas WebGL blends
 *   sRGB-encoded shader outputs: translucent pixels differ (opaque ones do not).
 * - Tone mapping is a whole-image output pass, not per-material in the shader;
 *   toneMapped:false materials and translucent pixels under tone mapping differ.
 * - preserveDrawingBuffer with autoClear:false accumulation is not admitted.
 */
export function createWebGLRendererClass(THREE, classOptions = {}) {
  const {exactBackend = null, ...options} = classOptions;
  if (exactBackend !== null && typeof exactBackend !== 'function') fail('OPTIONS', 'exactBackend must be a renderer constructor');
  const Base = createWebGPURendererClass(THREE, options);
  const glOnly = name => ({get() { fail('UNSUPPORTED', `WebGLRenderer.${name} describes a GL context; this route has none`); }, configurable: true});
  class WebGLRenderer extends Base {
    constructor(parameters = {}) {
      if (!parameters || typeof parameters !== 'object') fail('OPTIONS', 'Expected renderer parameters');
      for (const key of Object.keys(parameters)) if (!WEBGL_PARAMETERS.has(key)) fail('OPTIONS', `Unknown renderer parameter: ${key}`);
      // A supplied GL context is the exact backend by definition.
      if (parameters.context) {
        if (exactBackend) return new exactBackend(parameters);
        fail('ROUTE', 'A supplied rendering context selects the exact WebGL backend');
      }
      const {premultipliedAlpha = true, preserveDrawingBuffer = false, powerPreference = 'default',
        failIfMajorPerformanceCaveat, precision, outputBufferType, context, ...rest} = parameters;
      super({...rest, alpha: parameters.alpha ?? false,
        ...(powerPreference === 'default' ? {} : {powerPreference})});
      this.isWebGPURenderer = false;
      this.isWebGLRenderer = true;
      delete this.library; // WebGLRenderer has no node library.
      this.premultipliedAlpha = premultipliedAlpha;
      this.preserveDrawingBuffer = preserveDrawingBuffer;
      this._outputBufferType = outputBufferType;
      this.clippingPlanes = [];
      this.localClippingEnabled = false;
      this.shadowMap = {enabled: false, autoUpdate: true, needsUpdate: false, type: THREE.PCFShadowMap};
      this.transmissionResolutionScale = 1;
      this.debug = {checkShaderErrors: true, onShaderError: null};
      // Live per-frame controls read by the bridge as ordinary data properties.
      this._clippingControls = {planes: [], localClippingEnabled: false};
      this._deferUntilInitialized = true;
      this._bottomLeftOrigin = true;
      // WebGL writes sRGB-encoded fragments into an 8-bit framebuffer and blends
      // those encoded values; reproduce that with shader-side encoding.
      this._shaderEncodedOutput = true;
      this._contextAlpha = true;
      this._pixelRound = Math.round;
      // Source construction is synchronous: start device negotiation now.
      this.init().catch(error => { this._deferredError ??= error; });
    }
    _parameterNames() { return WEBGL_PARAMETERS; }
    async _init() {
      // r186 always creates an alpha context; straight alpha would need unpremultiplied compositing.
      if (this.premultipliedAlpha === false) fail('UNSUPPORTED', 'Straight-alpha canvas compositing is not admitted');
      if (this._outputBufferType !== undefined && this._outputBufferType !== THREE.UnsignedByteType)
        fail('UNSUPPORTED', 'Non-8-bit output buffers are not admitted');
      return super._init();
    }
    _needsGlobalClipping() { return this.clippingPlanes.length > 0; }
    _beforeRender() {
      if (!this._renderTarget && this.preserveDrawingBuffer && this.autoClear === false)
        fail('UNSUPPORTED', 'preserveDrawingBuffer accumulation across frames is not admitted');
      this._clippingControls.planes = this.clippingPlanes;
      this._clippingControls.localClippingEnabled = this.localClippingEnabled;
    }
    init() { return super.init(); }
    setAnimationLoop(callback) { this._setLoop(callback); }
    getContext() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    getContextAttributes() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    forceContextLoss() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    forceContextRestore() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    readRenderTargetPixelsAsync(target, x, y, width, height, buffer, activeCubeFace = 0, textureIndex = 0) {
      if (activeCubeFace !== 0 || textureIndex !== 0 || !ArrayBuffer.isView(buffer))
        return Promise.reject(new F3DRendererError('UNSUPPORTED', 'Supply an output array for the 2D color attachment'));
      return this._readTargetPixels(target, {x, y, width, height, output: buffer, flipY: true});
    }
    readRenderTargetPixels() { fail('ROUTE', 'Synchronous pixel readback requires the exact native backend'); }
    /** r186 compile(): returns the Set of the scene's materials synchronously, as
     * r186 does. Its GL programs are this route's asynchronously built
     * pipelines: preparation starts now (compileAsync) and a failure surfaces at
     * the next render()/compileAsync() call. Not a claim that pipelines exist on return. */
    compile(scene, camera, targetScene = null) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (!(scene instanceof THREE.Object3D)) fail('SOURCE', 'Expected a source Object3D root');
      const materials = new Set();
      scene.traverse(object => {
        if (!(object.isMesh || object.isPoints || object.isLine || object.isSprite)) return;
        const material = object.material;
        if (material) { if (Array.isArray(material)) for (const m of material) materials.add(m); else materials.add(material); }
      });
      this.compileAsync(scene, camera, targetScene).catch(error => { this._deferredError ??= error; });
      return materials;
    }
    getCurrentViewport(target) {
      return this._renderTarget ? target.copy(this._renderTarget.viewport).round() : target.copy(this._viewport).multiplyScalar(this._pixelRatio).round();
    }
  }
  for (const name of ['extensions', 'properties', 'renderLists'])
    Object.defineProperty(WebGLRenderer.prototype, name, glOnly(name));
  // WebGLState: only facts true of THIS renderer are served. Depth is never
  // reversed (capabilities.reversedDepthBuffer is false). setMask(true) is what
  // CubeCamera/Reflector/Water call so the next clear writes depth: this route's
  // clears always write depth, so it is that behavior, not a dropped call. Any
  // other GL state access (masks off, stencil, raw viewport) throws explicitly.
  const glState = (path, known) => new Proxy(known, {get(target, key) {
    if (typeof key === 'symbol' || Object.hasOwn(target, key)) return target[key];
    fail('UNSUPPORTED', `WebGLRenderer.state.${path}${String(key)} describes a GL context; this route has none`);
  }});
  const depthState = glState('buffers.depth.', {
    getReversed: () => false,
    setMask(mask) { if (mask !== true) fail('UNSUPPORTED', 'WebGLRenderer.state.buffers.depth.setMask(false) needs GL depth-mask state; this route has none'); },
  });
  const stateBuffers = glState('buffers.', {depth: depthState});
  const webglState = glState('', {buffers: stateBuffers});
  Object.defineProperty(WebGLRenderer.prototype, 'state', {configurable: true, get() { return webglState; }});
  // Only capabilities that describe THIS renderer's behavior are served: the
  // sampler anisotropy ceiling it applies (WebGPU clamps maxAnisotropy to 16, as
  // r186's WebGPU backend reports) and the depth/precision modes it admits.
  // GL-context limits (texture units, uniform vectors, ...) throw explicitly.
  Object.defineProperty(WebGLRenderer.prototype, 'capabilities', {configurable: true, get() {
    const known = {getMaxAnisotropy: () => 16, logarithmicDepthBuffer: false, reversedDepthBuffer: false, precision: 'highp',
      getMaxPrecision: () => 'highp', isWebGL2: true};
    return new Proxy(known, {get(target, key) {
      if (typeof key === 'symbol' || Object.hasOwn(target, key)) return target[key];
      fail('UNSUPPORTED', `WebGLRenderer.capabilities.${String(key)} describes a GL context; this route has none`);
    }});
  }});
  return WebGLRenderer;
}

/** `PMREMGenerator` for renderers on this route. Upstream PMREM renders into
 * CubeUV render targets with renderer internals this route does not have; here
 * the result's texture describes the source instead, and the scene bridge's
 * environment owner prepares it at its preparation boundary:
 * - fromScene(scene, sigma, near, far, {size, position}): captured into an
 *   rgba16float cubemap with the native scene bridge when first prepared (not at
 *   this call; later edits to the captured scene before preparation are seen),
 *   then GGX/DFG-filtered. sigma pre-blur is not reproduced.
 * - fromEquirectangular(texture): a clone sharing the source pixels (so the
 *   application may dispose its original), filtered from the panorama.
 * Neither is Three PMREM/CubeUV pixel equivalence. fromCubemap and explicit
 * render targets fail explicitly. Other renderers get the upstream class.
 */
export function createPMREMGeneratorClass(THREE, {exactBackend = null} = {}) {
  class PMREMGenerator {
    constructor(renderer) {
      if (!renderer?.isF3DRenderer) {
        if (exactBackend) return new exactBackend(renderer);
        fail('ROUTE', 'This PMREMGenerator serves the new-backend renderer route');
      }
      this._renderer = renderer;
    }
    fromScene(scene, sigma = 0, near = 0.1, far = 100, options = {}) {
      const {size = 256, position = new THREE.Vector3(), renderTarget = null} = options;
      if (renderTarget !== null) fail('UNSUPPORTED', 'Explicit PMREM render targets are not admitted');
      if (!(scene instanceof THREE.Object3D)) fail('SOURCE', 'Expected a source Object3D (r186 renders any root)');
      const texture = new THREE.CubeTexture();
      texture.name = 'PMREM.cubeUv';
      texture.mapping = THREE.CubeUVReflectionMapping;
      texture.colorSpace = THREE.LinearSRGBColorSpace;
      texture.isF3DSceneEnvironment = true;
      texture.f3dCapture = Object.freeze({scene, sigma, near, far, size, position: position.clone()});
      texture.needsUpdate = true;
      return {isRenderTarget: true, texture, dispose() { texture.dispose(); }};
    }
    async fromSceneAsync(...args) { return this.fromScene(...args); }
    fromEquirectangular(equirectangular, renderTarget = null) {
      if (renderTarget !== null) fail('UNSUPPORTED', 'Explicit PMREM render targets are not admitted');
      const texture = equirectangular.clone();
      texture.mapping = THREE.EquirectangularReflectionMapping;
      texture.needsUpdate = true;
      return {isRenderTarget: true, texture, dispose() { texture.dispose(); }};
    }
    async fromEquirectangularAsync(...args) { return this.fromEquirectangular(...args); }
    /** The cube-UV result is generated from the source at the scene bridge's
     * preparation boundary (WebGL surface programs run r186 PMREMGenerator
     * there); other routes reject the placeholder explicitly when preparing. */
    fromCubemap(cubemap, renderTarget = null) {
      if (renderTarget !== null) fail('UNSUPPORTED', 'Explicit PMREM render targets are not admitted');
      if (!cubemap?.isCubeTexture) fail('SOURCE', 'Expected a source CubeTexture');
      const texture = new THREE.Texture();
      texture.name = 'PMREM.cubeUv';
      texture.mapping = THREE.CubeUVReflectionMapping;
      texture.colorSpace = THREE.LinearSRGBColorSpace;
      texture.isF3DPMREMSource = true;
      texture.f3dPMREMSource = cubemap;
      return {isRenderTarget: true, texture, dispose() { texture.dispose(); }};
    }
    async fromCubemapAsync(...args) { return this.fromCubemap(...args); }
    compileCubemapShader() {}
    compileEquirectangularShader() {}
    dispose() {}
  }
  return PMREMGenerator;
}
