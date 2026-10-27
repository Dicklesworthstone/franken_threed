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
 *   Native animation callbacks wait for deferred command submission before
 *   starting another frame; this bounds cold-start work without waiting for
 *   GPU completion or discarding accepted render/compute requests.
 * - Admission failures discovered during deferred preparation are thrown from
 *   the next `render()`/`compileAsync()` call instead of the original call.
 * - Unsupported source state (advanced render targets, XR presentation, custom tone
 *   mapping, logarithmic/reversed depth, unsupported output color spaces,
 *   transparent clears with tone mapping) throws `F3DRendererError` rather than
 *   silently rendering something else.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {createThreeRenderTargets, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {createThreeRendererTransfers} from './three_renderer_transfers.mjs';
import {createThreePassSnapshots, captureThreeArrayCamera, captureThreeCamera, captureThreeBackground} from './three_pass_snapshot.mjs';
import {captureThreeMaterialState} from './three_material_snapshot.mjs';
import {threeWebGPUStencilValue} from './three_program_stencil.mjs';
import {createThreeProgramSupport} from './three_program.mjs';
import {createThreeProgramPMREM} from './three_program_pmrem.mjs';
import {createThreeProgramShadows} from './three_program_shadows.mjs';
import {createThreeProgramTransmission} from './three_program_transmission.mjs';
import {withThreeCompute} from './three_compute.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {inspectThreeMRT} from './three_node_passes.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {createThreeNodeBundles} from './three_node_bundles.mjs';
import {assertThreeNativeTargetWrite, captureThreeShadowMaterialState} from './three_node_shadows.mjs';
import {createThreeNodeBackgrounds} from './three_node_background.mjs';
import {createThreeSceneRaster} from './three_scene_raster.mjs';
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
const MAX_NODE_CAPTURE_BYTES = 64 * 1024 * 1024;
const MAX_MATERIAL_CAPTURE_VALUES = 1024 * 1024;

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
  const {gpu: defaultGpu, maxDraws = 16384, maxBindings = 16384, scene: sceneLimits = {}, exactBackend = null,
    maxTransferBytes = 128 * 1024 * 1024, maxTransfers = 64, targetTextureFlipY = false} = classOptions;
  if (![maxTransferBytes, maxTransfers].every(n => Number.isSafeInteger(n) && n > 0))
    fail('OPTIONS', 'Expected positive renderer transfer byte and request limits');
  if (exactBackend !== null && typeof exactBackend !== 'function') fail('OPTIONS', 'exactBackend must be a renderer constructor');
  if (typeof targetTextureFlipY !== 'boolean') fail('OPTIONS', 'Expected a target sampling orientation');
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1 || !Number.isSafeInteger(maxBindings) || maxBindings < 1)
    fail('OPTIONS', 'Expected positive draw and binding capacities');
  const {NoToneMapping, SRGBColorSpace} = THREE;
  // A retained addon may replace the exported renderer prototype's init before
  // construction (r186 Inspector Settings does this to select WebGL). Preserve
  // that method on a genuine source renderer, before allocating any native
  // state or acquiring a canvas context. Only instances constructed here may
  // dispatch the captured native init back into the retained implementation.
  const retainedInitReceivers = new WeakSet();
  const exactInit = exactBackend?.prototype?.init;
  const overriddenInitPrototype = receiver => {
    const prototype = Object.getPrototypeOf(receiver);
    for (let current = prototype; current !== null; current = Object.getPrototypeOf(current)) {
      const descriptor = Object.getOwnPropertyDescriptor(current, 'init');
      // Do not invoke an accessor early. Its ordinary getter must run only
      // when the application reads init on the constructed source instance.
      if (descriptor) return descriptor.value === nativeInit ? null : prototype;
    }
    return prototype;
  };

  /** Cheap, per-frame profile of scene-level features that select bridge options. */
  function sceneKey(renderer, scene, backgroundState) {
    const isScene = scene.isScene === true;
    const selected = backgroundState ? backgroundState.background : scene.background;
    const background = isScene && selected !== null && selected !== undefined && !selected.isColor;
    return `${isScene && scene.fog ? 1 : 0}${isScene && scene.environment ? 1 : 0}${background ? 1 : 0}${renderer.shadowMap.enabled ? 1 : 0}${renderer._needsGlobalClipping?.() ? 1 : 0}`;
  }
  /** Preparation-time traversal for options that depend on materials. */
  function needsClipping(scene, overrideMaterial = scene.isScene === true ? scene.overrideMaterial : null, nodeFrame = null, materialFrame = null) {
    const state = material => materialFrame?.materials.get(material) ?? material;
    let clipping = false;
    scene.traverse(object => {
      // A queued native render owns its visibility selection. Hidden source
      // objects must not force material admission during later preparation.
      if (nodeFrame?.excludedObjects?.has(object)) return;
      for (let ancestor = object; ancestor; ancestor = ancestor.parent)
        if (nodeFrame?.bundleObjects?.has(ancestor)) return;
      const material = materialFrame?.objects.has(object) ? materialFrame.objects.get(object) : object.material;
      const list = Array.isArray(material) ? material : material ? [material] : [];
      if (list.some(m => state(m)?.clippingPlanes?.length)) clipping = true;
    });
    if (state(overrideMaterial)?.clippingPlanes?.length) clipping = true;
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
        render({scene, camera, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame}, frame, framebuffer) {
          const entry = entries.get(scene);
          if (!entry) fail('PREPARE', 'Scene has not been prepared');
          entry.bridge.render(camera, frame, {overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, framebuffer});
          entry.lastDiagnostics = entry.bridge.diagnostics;
          return entry;
        },
        async prepare() {
          // Only the current request is installed while preparation can await.
          // Other queued fullscreen roots may still hold an earlier pass state.
          const root = destination._preparationRoot;
          const wanted = root ? [[root, destination._wanted.get(root)]] : destination._wanted;
          for (const [scene, key] of wanted) {
            const passOptions = root === scene ? destination._preparationOptions : undefined;
            const overrideMaterial = passOptions ? passOptions.overrideMaterial : scene.isScene === true ? scene.overrideMaterial : null;
            const clipping = needsClipping(scene, overrideMaterial, passOptions?.nodeFrame, passOptions?.materialFrame) || owner._needsGlobalClipping?.() === true;
            const previous = entries.get(scene);
            if (previous && previous.key === key && (previous.clipping || !clipping) && !previous.bridge.failed) {
              await previous.bridge.prepare(passOptions);
              continue;
            }
            // A changed feature profile selects different bridge pipelines.
            // Retire the old owner only after the replacement is published.
            const sourceScene = scene.isScene === true;
            const selectedBackground = passOptions?.backgroundState ? passOptions.backgroundState.background : scene.background;
            const background = sourceScene && selectedBackground && !selectedBackground.isColor;
            // The captured background and live structural options must select
            // the same profile during preparation and submission.
            const liveKey = sceneKey(owner, scene, passOptions?.backgroundState);
            let bridge;
            try { bridge = await createGpuThreeScene(device, scene, {
              // Mesh-count caps follow the binding capacity; the byte budgets
              // (deformation/instance) stay the scene owner's defaults and bound memory.
              maxDeformedMeshes: Math.min(maxBindings, 4096), maxInstanceMeshes: Math.min(maxBindings, 4096),
              // Source renderers have no texture budget: bound residency by a
              // device-scale default (1 GiB) rather than the bridge's 128 MiB.
              texture: {maxTextureBytes: 1024 * 1024 * 1024},
              ...sceneLimits, three: THREE, signal, maxBindings, overrideMaterial, nodeFrame: passOptions?.nodeFrame ?? null,
              backgroundState: passOptions?.backgroundState, materialFrame: passOptions?.materialFrame,
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
              if (error?.code === 'THREE_SCENE_SCENE' && sceneKey(owner, scene, passOptions?.backgroundState) !== liveKey) continue;
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
      const sourcePrototype = exactBackend && overriddenInitPrototype(this);
      if (sourcePrototype) {
        if (typeof exactInit !== 'function') fail('ROUTE', 'An overridden init requires an exact backend with its source init method');
        const retained = new exactBackend(parameters);
        retainedInitReceivers.add(retained);
        // Keep prototype changes live: Settings can restore the captured init
        // after constructing this instance. An application assignment still
        // creates its ordinary own method, just as inherited writable init did.
        if (!Object.hasOwn(retained, 'init')) {
          const bridge = Object.create(Object.getPrototypeOf(retained), {
            init: {
              configurable: true,
              get() { return Reflect.get(sourcePrototype, 'init', this); },
              set(value) { Object.defineProperty(this, 'init', {value, configurable: true, writable: true, enumerable: true}); },
            },
          });
          Object.setPrototypeOf(retained, bridge);
        }
        return retained;
      }
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
      this.samples = samples || (antialias === true ? 4 : 0);
      this.logarithmicDepthBuffer = logarithmicDepthBuffer;
      this.reversedDepthBuffer = reversedDepthBuffer;
      this.outputColorSpace = SRGBColorSpace;
      this.toneMapping = NoToneMapping;
      this.toneMappingExposure = 1;
      this.sortObjects = true;
      this.opaque = true;
      this.transparent = true;
      this.shadowMap = {enabled: false, transmitted: false, type: THREE.PCFShadowMap};
      this.lighting = {enabled: true};
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
      this._outputRenderTarget = null;
      this._outputQuads = new Map();
      this._targetDepthInitialized = new WeakSet();
      this._outputBufferType = parameters.outputBufferType ?? parameters.colorBufferType ?? THREE.HalfFloatType;
      this._activeCubeFace = 0;
      this._activeMipmapLevel = 0;
      this._targetLifetime = new AbortController();
      this._targets = createThreeRenderTargets(THREE, {getDevice: () => this._device, flipY: targetTextureFlipY});
      this._passSnapshots = createThreePassSnapshots(THREE);
      this._nodeProgramSupport = null;
      this._nodeBackgrounds = null;
      this._nodeBundles = null;
      this._nodeCaptureDepth = 0;
      this._nodeCaptureBytes = 0;
      this._materialCaptureValues = 0;
      this._nodeRaster = new Map();
      this._renderObjectFunction = null;
      this._handleObjectFunction = null;
      this._cacheShadowNodes = new WeakMap();
      this._currentSourceMaterial = null;
      this._mrt = null;
      this._initialized = false;
      this._initPromise = null;
      this._session = null;
      this._dispatcher = null;
      this._device = null;
      this._ownsDevice = false;
      this._rebuild = false;
      this._preparing = null;
      this._hdr = false;
      this._srgbView = true;
      this._wanted = new Map();
      this._pending = [];
      this._drain = null;
      this._deferredError = null;
      this._animationCallback = null;
      this._animationHandle = null;
      this._animationWaiting = null;
      this._animationStarted = false;
      this._animationHost = null;
      this._disposed = false;
      this._transfers = createThreeRendererTransfers(this, F3DRendererError, {maxBytes: maxTransferBytes, maxRequests: maxTransfers});
      Object.defineProperties(this.info.f3d, {
        pendingTransfers: {enumerable: true, get: () => this._transfers.pendingRequests},
        pendingTransferBytes: {enumerable: true, get: () => this._transfers.allocatedBytes},
      });
      this._renderUid = 0;
      // Tooling surface (e.g. the r186 Inspector addon). This backend records no
      // GPU timestamps: hasTimestamp stays false and no timing is reported.
      this._backend = {isWebGPUBackend: true, isF3DBackend: true, device: null, trackTimestamp: false, hasTimestamp: false,
        capabilities: {getUniformBufferLimit: () => this._device.limits.maxUniformBufferBindingSize},
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
    get isOutputTarget() { return this._renderTarget === null || this._renderTarget === this._outputRenderTarget; }
    get currentToneMapping() { return this.isOutputTarget ? this.toneMapping : NoToneMapping; }
    get currentColorSpace() { return this.isOutputTarget ? this.outputColorSpace : THREE.LinearSRGBColorSpace; }
    get needsFrameBufferTarget() { return this.currentToneMapping !== NoToneMapping || this.currentColorSpace !== THREE.LinearSRGBColorSpace; }
    get currentSamples() { return this._renderTarget ? this._renderTarget.samples : this.needsFrameBufferTarget ? 0 : this.samples; }
    get backend() { this._backend.device = this._session?.device ?? null; return this._backend; }

    init() {
      if (retainedInitReceivers.has(this)) return exactInit.call(this);
      if (this._initPromise) return this._initPromise;
      this._initPromise = this._init();
      return this._initPromise;
    }
    async _init() {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this.logarithmicDepthBuffer || this.reversedDepthBuffer) fail('UNSUPPORTED', 'Logarithmic and reversed depth buffers are not admitted');
      if (this.stencil && !this.depth) fail('UNSUPPORTED', 'Stencil framebuffers currently require a depth attachment');
      const p = this._parameters;
      if (p.outputType !== undefined && p.outputType !== THREE.UnsignedByteType)
        fail('UNSUPPORTED', 'Extended-range canvas output types are not admitted');
      if (this.isWebGPURenderer && ![THREE.UnsignedByteType, THREE.HalfFloatType, THREE.FloatType].includes(this._outputBufferType))
        fail('UNSUPPORTED', 'Output buffers require unsigned-byte, half-float or float texture storage');
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
      // Source Animation.start() advances once during initialization and keeps
      // the node clock running even without a user animation callback.
      if (this.isWebGPURenderer) this._setLoop(null);
      return this;
    }
    /** Native tone mapping and output color conversion use the source's linear
     * framebuffer target. Preserve working-linear scene color from the first
     * render, including when no tone mapping is selected. */
    async _createSession(hdr, srgbView = this._canvasSrgbView(this.outputColorSpace, hdr)) {
      const sampleCount = this.antialias || this.samples > 0 ? 4 : 1;
      const depthFormat = this.depth ? (this.stencil ? 'depth24plus-stencil8' : 'depth24plus') : null;
      const [width, height] = this._bufferSize();
      const common = {device: this._device};
      const format = this._preferredFormat ? {format: this._preferredFormat} : {};
      this._hdr = hdr;
      this._srgbView = srgbView;
      this._dispatcher = null;
      this._session = hdr
        ? await createGpuHdrCanvasRenderer(this.domElement, createDispatcher(this), {...common,
          target: {width, height, ...format, srgbView, alphaMode: this._compositedAlpha() ? 'premultiplied' : 'opaque'},
          renderTarget: {depthFormat, sampleCount, format: this._outputBufferFormat()},
          output: {toneMapping: this._toneMapping(), exposure: this.toneMappingExposure}})
        : await createGpuCanvasRenderer(this.domElement, createDispatcher(this), {...common, lazyAttachments: true,
          target: {width, height, depthFormat, sampleCount, ...format, srgbView,
            alphaMode: this._compositedAlpha() ? 'premultiplied' : 'opaque'}});
    }
    _release() {
      for (const cache of this._outputQuads.values()) cache.dispose();
      // The retained framebuffer-target listener releases its private target.
      // Source Renderer.dispose() also disposes custom output targets used as
      // framebuffer-cache keys; an unused output selector owns no such target.
      for (const target of this._frameBufferTargets?.keys() ?? [])
        if (target !== this._canvasTarget) target.dispose();
      this._targets.dispose();
      this._session?.dispose();
      this._session = null;
      this._wanted.clear();
      this._passSnapshots.dispose();
      this._nodeBackgrounds?.dispose(); this._nodeBackgrounds = null;
      this._nodeProgramSupport?.dispose();
      this._nodeProgramSupport = null;
      this._nodeBundles?.dispose(); this._nodeBundles = null;
      this._canvasTarget?.dispose();
      if (this._ownsDevice) { this._ownsDevice = false; this._device?.destroy(); }
    }

    /** The whole-image tone-mapping output pass. WebGLRenderer (shader-encoded
     * output) tone-maps inside each material's program instead, unless a scene
     * needed a draw the program route cannot take (sticky fallback). */
    _wantsHdr(tone, space = this.outputColorSpace) { return !this._shaderEncodedOutput &&
      (tone !== 'none' || space !== THREE.LinearSRGBColorSpace) ||
      tone !== 'none' && this._hdrFallback === true; }
    _outputBufferFormat() {
      return this._outputBufferType === THREE.UnsignedByteType ? 'rgba8unorm' :
        this._outputBufferType === THREE.FloatType ? 'rgba32float' : 'rgba16float';
    }
    // Source RenderPipeline writes its output transform in the graph and sets
    // working-linear output to suppress a second transfer. Each queued frame
    // captures the attachment view separately from the live renderer property.
    _canvasSrgbView(space = this.outputColorSpace, hdr = this._wantsHdr(this._toneMapping())) {
      return this._shaderEncodedOutput ? hdr : space === SRGBColorSpace;
    }
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
    getActiveCubeFace() { return this._activeCubeFace; }
    getActiveMipmapLevel() { return this._activeMipmapLevel; }
    /** Source Renderer.clear(): a clear-only pass on the canvas with the current
     * clear color/depth, honoring viewport/scissor state like other passes. */
    clear(color = true, depth = true, stencil = true) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (this._initialized === false && !this._deferUntilInitialized)
        throw new Error('THREE.Renderer: .clear() called before the backend is initialized. Use "await renderer.init();" before using this method.');
      const targetShape = this._renderTarget ? this._targets.capture(this._renderTarget).shape : null;
      const hasDepth = targetShape ? targetShape.depthAttachment && targetShape.depthFormat !== null : this.depth;
      const hasStencil = targetShape ? targetShape.depthAttachment && targetShape.depthFormat?.endsWith('-stencil8') : this.stencil;
      if (!color && !(depth && hasDepth) && !(stencil && hasStencil)) return;
      this._clearScene ??= new THREE.Scene();
      this._clearCamera ??= new THREE.PerspectiveCamera();
      this._renderFrame(this._clearScene, this._clearCamera, {color, depth, stencil});
    }
    clearColor() { this.clear(true, false, false); }
    clearDepth() { this.clear(false, true, false); }
    clearStencil() { this.clear(false, false, true); }
    async clearAsync(color = true, depth = true, stencil = true) {
      await this.init(); this.clear(color, depth, stencil);
      while (this._drain) await this._drain;
    }
    async clearColorAsync() { this.clear(true, false, false); while (this._drain) await this._drain; }
    async clearDepthAsync() { this.clear(false, true, false); while (this._drain) await this._drain; }
    async clearStencilAsync() { this.clear(false, false, true); while (this._drain) await this._drain; }
    getClearColor(target) { target.copy(this._clearColor); if ('a' in target) target.a = this._clearAlpha; return target; }
    setClearColor(color, alpha = 1) { this._clearColor.set(color); this._clearAlpha = alpha; }
    getClearAlpha() { return this._clearAlpha; }
    setClearAlpha(alpha) { this._clearAlpha = alpha; }
    getClearDepth() { return this._clearDepth; }
    setClearDepth(depth) { this._clearDepth = depth; }
    getClearStencil() { return this._clearStencil; }
    setClearStencil(stencil) { this._clearStencil = stencil; }
    getRenderTarget() { return this._renderTarget; }
    getCanvasTarget() {
      // ViewportTextureNode keys its source-owned images by this real, stable
      // CanvasTarget. Synchronize its read state without resizing the canvas
      // while a render request is being captured.
      const target = this._canvasTarget ??= new THREE.CanvasTarget(this.domElement);
      target._width = this._width; target._height = this._height; target._pixelRatio = this._pixelRatio;
      target._viewport.copy(this._viewport); target._scissor.copy(this._scissor); target._scissorTest = this._scissorTest;
      return target;
    }
    _framebufferFormat() {
      const internal = this._frameBufferTargets?.get(this._canvasTarget);
      if (internal && this._renderTarget === internal) this._nativeFramebufferOutput = true;
      if (this._renderTarget) return this._targets.capture(this._renderTarget).shape.format;
      // Refraction reads working-linear scene color before the final output
      // conversion. Sampling bytes copied from the sRGB presentation view
      // would apply that conversion twice. Select the existing linear scene
      // target at first framebuffer layout admission, before shader capture.
      if (!this._shaderEncodedOutput && this.needsFrameBufferTarget) this._nativeFramebufferOutput = true;
      return this._wantsHdr(this._toneMapping()) ? this._outputBufferFormat() : this._preferredFormat ?? 'bgra8unorm';
    }
    copyFramebufferToTexture(texture, rectangle = null) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (!this._initialized) fail('SOURCE', 'Initialize the renderer before copying its framebuffer');
      if (this._shaderEncodedOutput) fail('UNSUPPORTED', 'Public framebuffer copies require the native WebGPU route');
      if (!(texture instanceof THREE.FramebufferTexture) || texture.isDepthTexture)
        fail('SOURCE', 'Expected a source color FramebufferTexture');
      let region;
      if (rectangle === null) region = [0, 0, texture.image.width, texture.image.height];
      else if (rectangle.isVector2) region = [rectangle.x, rectangle.y, texture.image.width, texture.image.height].map(Math.floor);
      else if (rectangle.isVector4) region = [rectangle.x, rectangle.y, rectangle.z, rectangle.w].map(Math.floor);
      else { (THREE.error ?? console.error)('Renderer.copyFramebufferToTexture: Invalid rectangle.'); return; }
      const textures = this._getSampledTextureInputs();
      textures.prepareFramebuffer(texture, this._renderTarget?.texture.type ?? THREE.UnsignedByteType);
      const format = this._framebufferFormat();
      const source = !this._nodeFramebufferCopyRecorder && this._renderTarget
        ? this._targets.capture(this._renderTarget) : null;
      const [width, height] = source ? [source.shape.width, source.shape.height] : this._bufferSize();
      if (!this._nodeFramebufferCopyRecorder &&
          (region[0] + region[2] > width || region[1] + region[3] > height))
        fail('COPY', 'Framebuffer copy rectangle exceeds the selected source');
      const lease = textures.captureFramebuffer(texture, {format,
        rectangle: {x: region[0], y: region[1], z: region[2], w: region[3]}});
      if (this._nodeFramebufferCopyRecorder) {
        try { this._nodeFramebufferCopyRecorder(Object.freeze({lease})); }
        catch (error) { textures.release(lease); throw error; }
      } else {
        let releaseSource = null, released = false;
        const release = () => {
          if (released) return; released = true;
          textures.release(lease); releaseSource?.();
        };
        try {
          releaseSource = source ? this._targets.retainGeneration(source) : null;
          const hdr = !source && this._wantsHdr(this._toneMapping());
          const srgbView = this._canvasSrgbView(this.outputColorSpace, hdr);
          const copy = () => {
            if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
            textures.validateDispatch([lease]);
            const encode = native => {
              textures.validateDispatch([lease]);
              textures.withDispatch([lease], () => {
                const encoder = this._device.createCommandEncoder({label: 'f3d-public-framebuffer-copy'});
                textures.encodeFramebufferCopy(lease, encoder, native);
                this._device.queue.submit([encoder.finish()]);
              });
            };
            if (source) this._targets.withFramebufferTexture(source, encode);
            else this._session.withFramebufferTexture(encode, {width, height});
          };
          const request = {framebufferCopy: true, releaseFramebuffer: release, issue: () => {
            // Invalid destinations must not acquire or replace presentation.
            textures.validateDispatch([lease]);
            if (!source && (hdr !== this._hdr || srgbView !== this._srgbView)) {
              this._session.dispose();
              return this._createSession(hdr, srgbView).then(copy);
            }
            return copy();
          }};
          // Public void calls share the render FIFO. An already prepared copy
          // submits now; pending renders keep their original producer boundary.
          if (!this._drain && !this._preparing && !this._pending.length &&
              (source || hdr === this._hdr && srgbView === this._srgbView)) {
            try { request.issue(); } finally { this._releaseRequest(request); }
          } else this._enqueue(request);
        } catch (error) { release(); throw error; }
      }
      this._inspector.copyFramebufferToTexture?.(texture);
    }
    getOutputRenderTarget() { return this._outputRenderTarget; }
    setOutputRenderTarget(target) {
      if (target !== null) {
        if (this._shaderEncodedOutput) fail('UNSUPPORTED', 'Custom output targets require the native WebGPU route');
        const shape = this._targets.capture(target).shape;
        if (target.isCubeRenderTarget || shape.textures.length !== 1)
          fail('UNSUPPORTED', 'Custom output targets require one 2D color attachment');
      }
      this._outputRenderTarget = target;
    }
    getOutputBufferType() { return this._outputBufferType; }
    getMRT() { return this._mrt; }
    setMRT(value) { this._mrt = value; return this; }
    initRenderTarget(target) {
      if (this._initialized === false)
        throw new Error('THREE.Renderer: .initRenderTarget() called before the backend is initialized. Use "await renderer.init();" before using this method.');
      this._targets.ensure(this._targets.capture(target, {initialize: true}));
    }
    setRenderTarget(renderTarget, activeCubeFace = 0, activeMipmapLevel = 0) {
      if (activeMipmapLevel !== 0 && this._shaderEncodedOutput)
        fail('UNSUPPORTED', 'Nonzero target mips on the WebGL-authored route are not admitted');
      if (renderTarget !== null) inspectThreeRenderTargetMip(this._targets.capture(renderTarget).shape, activeMipmapLevel, activeCubeFace);
      else if (activeMipmapLevel !== 0 || activeCubeFace !== 0) fail('UNSUPPORTED', 'The canvas has only mip level and face zero');
      this._renderTarget = renderTarget;
      this._activeCubeFace = activeCubeFace;
      this._activeMipmapLevel = activeMipmapLevel;
    }
    /** F3D extensions: call-time upload snapshots and ordered GPU-only copies.
     * Regions use this surface's origin (WebGPU top-left, WebGL bottom-left),
     * overridable with options.flipY. Promises include validation/completion. */
    writeRenderTargetPixelsAsync(target, data, options = {}) {
      return this._transfers.writePixels(target, data, options);
    }
    copyRenderTargetPixelsAsync(source, destination, options = {}) {
      return this._transfers.copyColor(source, destination, options);
    }
    readRenderTargetPixelsAsync(target, x, y, width, height, textureIndex = 0, faceIndex = 0) {
      return this._readTargetPixels(target, {x, y, width, height, colorIndex: textureIndex, face: faceIndex});
    }
    _readTargetPixels(target, options) {
      // Source-shaped reads share target generation/cancellation, request
      // budgets and queue ordering with explicit mip/depth reads and transfers.
      // Mapping never holds the FIFO up after its texture copy was submitted.
      return this._transfers.readPixels(target, options);
    }
    getContext() { return this._initialized ? this.domElement.getContext('webgpu') : null; }

    // ---- frames ----
    _frame(scene, clearOnly = null, target = this._renderTarget, mipLevel = this._activeMipmapLevel) {
      const targetShape = target ? this._targets.capture(target).shape : null;
      const mipShape = targetShape ? inspectThreeRenderTargetMip(targetShape, mipLevel) : null;
      if (this.xr.enabled && this.xr.isPresenting) fail('UNSUPPORTED', 'XR presentation is not admitted');
      if (!target && this.outputColorSpace !== SRGBColorSpace &&
          (this._shaderEncodedOutput || this.outputColorSpace !== THREE.LinearSRGBColorSpace))
        fail('UNSUPPORTED', 'Native canvas output requires sRGB or linear-sRGB');
      const tone = this._toneMapping();
      // Switching between direct and tone-mapped output rebuilds the canvas
      // session (same device) at the next preparation boundary.
      if (!target) this._rebuild = this._wantsHdr(tone) !== this._hdr || this._canvasSrgbView() !== this._srgbView;
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
      // Use the actual depth attachment: a target-owned depth-only texture can
      // override stencilBuffer:true. A stencil clear never clears color/depth.
      const stencilAttachment = target ? targetShape.depthAttachment && targetShape.depthFormat?.endsWith('-stencil8') : this.stencil;
      if (stencilAttachment) {
        const clearStencil = clearOnly ? clearOnly.stencil : clear && this.autoClearStencil;
        frame.stencilLoadOp = clearStencil ? 'clear' : 'load';
        // Both native attachment formats have eight stencil bits. GL converts
        // a signed long and masks it; WebGPU first enforces an unsigned long.
        frame.clearStencil = clearStencil ? (this._bottomLeftOrigin ? this._clearStencil | 0
          : threeWebGPUStencilValue(this._clearStencil, fail)) & 0xff : 0;
      }
      const pr = target ? 1 : this._pixelRatio;
      const [bufferWidth, bufferHeight] = target ? [mipShape.width, mipShape.height] : this._bufferSize();
      const v = target ? target.viewport : this._viewport, px = this._pixelRound ?? Math.floor;
      // r186 WebGPU scales only width/height at a selected mip; origins remain
      // physical coordinates. Capture without mutating the source Vector4s.
      const divisor = target ? 2 ** mipLevel : 1;
      const vx = px(v.x * pr), vw = Math.floor(px(v.z * pr) / divisor), vh = Math.floor(px(v.w * pr) / divisor);
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
        const sc = target ? target.scissor : this._scissor, rawH = Math.floor(px(sc.w * pr) / divisor);
        const rawY = this._bottomLeftOrigin ? bufferHeight - px(sc.y * pr) - rawH : px(sc.y * pr);
        const sx = Math.max(0, px(sc.x * pr)), sy = Math.max(0, rawY);
        // WebGPU source: clamp each component to >= 0, then fit inside the buffer.
        // WebGL: the GL scissor box is intersected with the buffer, so a negative
        // origin also shortens the box.
        const clip = this._bottomLeftOrigin;
        const sw = Math.max(0, Math.min(Math.floor(px(sc.z * pr) / divisor) + (clip ? Math.min(0, px(sc.x * pr)) : 0), bufferWidth - sx)),
          sh = Math.max(0, Math.min(rawH + (clip ? Math.min(0, rawY) : 0), bufferHeight - sy));
        if (sx !== 0 || sy !== 0 || sw !== bufferWidth || sh !== bufferHeight) frame.scissor = [sx, sy, sw, sh];
      }
      if (hdr) {
        if (!this._compositedAlpha()) frame.clearColor[3] = 1;
        frame.output = {toneMapping: tone, exposure: this.toneMappingExposure};
      }
      if (target && !this._shaderEncodedOutput) {
        const context = inspectThreeMRT(THREE, this);
        // The source background overrides the ordinary primary clear before
        // MRT defaults are selected. Capture it now, before queued preparation
        // or node updates can change the scene; authored MRT clears still win.
        const background = !clearOnly && scene.isScene === true && !scene.backgroundNode?.isNode &&
          scene.background?.isColor ? scene.background : null;
        if (context) frame.clearColors = target.textures.map((texture, index) => {
          const color = context.mrt?.getClearColor(texture.name);
          return color ? [color.r, color.g, color.b, color.a]
            : index === 0 ? background ? [background.r, background.g, background.b, 1] : frame.clearColor.slice() : [0, 0, 0, 1];
        });
      }
      return frame;
    }
    render(scene, camera) {
      if (this._initialized === false && !this._deferUntilInitialized) throw new Error(NOT_INITIALIZED);
      return this._renderCall(scene, camera);
    }
    _beforeRender() {}
    getRenderObjectFunction() { return this._renderObjectFunction; }
    setRenderObjectFunction(value) {
      if (value !== null && typeof value !== 'function') fail('SOURCE', 'A render object function must be callable or null');
      this._renderObjectFunction = value;
    }
    _getShadowNodes(material) { return THREE.Renderer.prototype._getShadowNodes.call(this, material); }
    renderObject(...args) {
      if (typeof this._handleObjectFunction !== 'function')
        fail('SOURCE', 'Native renderObject requires an active source draw capture');
      const scene = args[1], material = scene.overrideMaterial;
      const state = material ? captureThreeShadowMaterialState(material) : null;
      const sourceMaterial = this._currentSourceMaterial;
      try { return THREE.Renderer.prototype.renderObject.apply(this, args); }
      catch (error) {
        // The retained lifecycle restores its overrides on normal completion.
        // A native admission failure must not leak the last caster into later
        // application draws when the source callback stack unwinds early.
        if (state) Object.assign(material, state);
        this._currentSourceMaterial = sourceMaterial; throw error;
      }
    }
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
    _makeRequest(scene, camera, clearOnly = null, captureCameras = true, sourceTarget = this._renderTarget) {
      const target = this._renderTarget, mipLevel = this._activeMipmapLevel, face = this._activeCubeFace, mrt = this._mrt;
      const renderId = this.info.calls;
      if (!this._shaderEncodedOutput) assertThreeNativeTargetWrite(this, target);
      const destination = target ? this._targets.capture(target, {initialize: !this._shaderEncodedOutput}) : this;
      const backgroundState = !this._shaderEncodedOutput ? captureThreeBackground(THREE, scene) : undefined;
      const frame = this._frame(scene, clearOnly, target, mipLevel), key = sceneKey(this, scene, backgroundState);
      // RenderPass/normal/depth passes restore this property immediately after
      // render(). Capture its selector now; source objects stay application-owned.
      const overrideMaterial = scene.isScene === true ? scene.overrideMaterial : null;
      const c = this._clearColor, a = this._clearAlpha;
      // Small callback-free shader passes (FullScreenQuad) are private data views;
      // general scenes and hooked roots retain their established source path.
      const snapshot = this._shaderEncodedOutput ? this._passSnapshots.capture(scene, camera) : null;
      let matricesUpdated = false;
      if (camera?.isArrayCamera === true && captureCameras) {
        // WebGLRenderer has camera-major drawing, physical bottom-left viewports
        // and per-child callbacks. This slice implements the WebGPU contract.
        if (this._shaderEncodedOutput) fail('UNSUPPORTED', 'ArrayCamera on the WebGL-authored route is not admitted');
        if (!Array.isArray(camera.cameras) || camera.cameras.length > 256 || camera.isMultiViewCamera)
          fail('UNSUPPORTED', 'Expected an ordinary ArrayCamera with at most 256 views');
        if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
        // Pinned Renderer._updateCamera changes the public projection convention
        // for all children only when the parent convention changes.
        if (camera.coordinateSystem !== this.coordinateSystem) {
          camera.coordinateSystem = this.coordinateSystem;
          for (const child of camera.cameras) child.coordinateSystem = this.coordinateSystem;
          camera.updateProjectionMatrix();
          for (const child of camera.cameras) child.updateProjectionMatrix();
        }
        if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
        camera = captureThreeArrayCamera(THREE, camera);
        matricesUpdated = true;
        // Source WebGPU viewports are logical canvas units, physical target units.
        frame.arrayCamera = {pixelRatio: target ? 1 : this._pixelRatio, clipSpace: 'webgpu',
          targetSize: target ? [Math.max(1, Math.floor(target.width / 2 ** mipLevel)), Math.max(1, Math.floor(target.height / 2 ** mipLevel))] : this._bufferSize()};
      }
      const native = !this._shaderEncodedOutput && !clearOnly;
      // Before initialization, the source renderer prepares at its asynchronous
      // boundary. Texture feature/layout admission needs the actual device.
      const nodeCaptureDeferred = native && !this._initialized;
      const nodeFrame = native && !nodeCaptureDeferred
        ? this._captureNodeFrame({scene, camera, destination, overrideMaterial, captureCameras, mrt, renderId, sourceTarget}) : null;
      if (nodeFrame) matricesUpdated = true;
      // Native callbacks retain the original camera in _captureNodeFrame.
      // Fixed draws can wait on pipeline/texture preparation; capture their
      // ordinary camera only after the source matrix and node updates finish.
      if (native && !nodeCaptureDeferred && captureCameras && !camera.isArrayCamera) {
        if (!matricesUpdated) this._updateNativeMatrices(scene, camera);
        camera = captureThreeCamera(THREE, camera); matricesUpdated = true;
      }
      let materialFrame = null;
      if (native && !nodeCaptureDeferred) {
        try { materialFrame = this._captureMaterialFrame({scene, nodeFrame, overrideMaterial}); }
        catch (error) { this._releaseRequest({nodeFrame}); throw error; }
      }
      return {scene: snapshot?.root ?? scene, camera: snapshot?.camera ?? camera, snapshot, matricesUpdated, nodeFrame, clearOnly, destination, mipLevel, face, frame, overrideMaterial, mrt,
        nodeCaptureDeferred, captureCameras, backgroundState, materialFrame, renderId, sourceTarget,
        generateCubeMipmaps: !this._shaderEncodedOutput && !clearOnly && target?.isCubeRenderTarget === true && target.texture.generateMipmaps,
        linearClearColor: this.alpha ? [c.r * a, c.g * a, c.b * a, a] : [c.r, c.g, c.b, 1],
        hdr: !target && this._wantsHdr(this._toneMapping()), srgbView: this._canvasSrgbView(), key,
        programFrame: {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
          outputColorSpace: this.outputColorSpace, pixelRatio: this._pixelRatio, height: this._height},
        releaseDestination: destination === this ? null : this._targets.retainGeneration(destination)};
    }
    _updateNativeMatrices(scene, camera) {
      if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
      if (camera.coordinateSystem !== this.coordinateSystem) {
        camera.coordinateSystem = this.coordinateSystem; camera.updateProjectionMatrix();
      }
      if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
    }
    _renderFrame(scene, camera, clearOnly = null) {
      const requests = this._makeOutputRequests(scene, camera, clearOnly);
      for (let i = 0; i < requests.length; i++) {
        try { this._dispatchRequest(requests[i]); }
        catch (error) {
          for (const pending of requests.slice(i + 1)) this._releaseRequest(pending);
          throw error;
        }
      }
    }
    _outputQuad(target) {
      let cache = this._outputQuads.get(target);
      if (!cache) {
        cache = {quads: new Map(), dispose: () => {
          if (cache.resizing) return;
          target.removeEventListener('dispose', cache.dispose);
          for (const quad of cache.quads.values()) quad.material.dispose();
          this._outputQuads.delete(target);
        }};
        target.addEventListener('dispose', cache.dispose);
        this._outputQuads.set(target, cache);
      }
      const key = `${this.toneMapping},${this.currentColorSpace}`;
      let quad = cache.quads.get(key);
      if (!quad) {
        quad = new THREE.QuadMesh(new THREE.NodeMaterial());
        quad.name = 'Output Color Transform'; quad.material.name = 'outputColorTransform';
        // Retain the pinned NodeManager.getOutputNode construction, including
        // alpha un/premultiplication, exposure and the source tone operators.
        quad.material.fragmentNode = THREE.TSL.texture(target.texture, THREE.TSL.screenUV)
          .renderOutput(this.toneMapping, this.currentColorSpace);
        // Distinct graph identities keep earlier queued captures valid when a
        // later call changes the renderer's output transform before preparation.
        cache.quads.set(key, quad);
      }
      return quad;
    }
    _makeOutputRequests(scene, camera, clearOnly = null, captureCameras = true) {
      if (!this._outputRenderTarget || this._shaderEncodedOutput)
        return [this._makeRequest(scene, camera, clearOnly, captureCameras)];
      const previousTarget = this._renderTarget, previousFace = this._activeCubeFace,
        previousMip = this._activeMipmapLevel;
      const outputTarget = previousTarget || this._outputRenderTarget;
      // clear() selects an explicit target first and only otherwise the private
      // framebuffer. With no conversion, a null selection still clears canvas.
      const convert = this.needsFrameBufferTarget && (!clearOnly || previousTarget === null);
      if (!convert && (previousTarget !== null || clearOnly)) {
        const request = this._makeRequest(scene, camera, clearOnly, captureCameras);
        request.outputPass = previousTarget === this._outputRenderTarget;
        return [request];
      }
      const requests = [];
      let completed = false, outputState = null;
      try {
        if (!convert) {
          const initializeDepth = captureCameras && outputTarget.depthBuffer === true && this.depth &&
            (this.autoClear === false || this.autoClear === true && this.autoClearDepth === false);
          if (initializeDepth) {
            // Source clearDepth() still selects canvas when direct default
            // output leaves the public selector null. Preserve that side effect
            // and its call-time clear value ahead of the output draw in the FIFO.
            this._clearScene ??= new THREE.Scene(); this._clearCamera ??= new THREE.PerspectiveCamera();
            const depth = this._makeRequest(this._clearScene, this._clearCamera, {color: false, depth: true, stencil: false});
            depth.initializeOutputDepth = this._targets.capture(outputTarget);
            requests.push(depth);
          }
          this._renderTarget = outputTarget;
          requests.push(this._makeRequest(scene, camera, clearOnly, captureCameras, previousTarget));
        } else {
          this._frameBufferTargets ??= new Map();
          const previousInternal = this._frameBufferTargets.get(this._outputRenderTarget);
          const previousCache = this._outputQuads.get(previousInternal);
          let internal;
          try {
            if (previousCache) previousCache.resizing = true;
            const construct = () => THREE.Renderer.prototype._getFrameBufferTarget.call(this);
            // Source automatic target resizing preserves accepted GPU readers;
            // public output-target disposal still invalidates the whole pair.
            internal = previousInternal ? this._targets.withNativePassResize(previousInternal,
              this._outputRenderTarget.width, this._outputRenderTarget.height, construct) : construct();
          } finally { if (previousCache) previousCache.resizing = false; }
          this._renderTarget = internal; this._activeCubeFace = 0; this._activeMipmapLevel = 0;
          requests.push(this._makeRequest(scene, camera, clearOnly, captureCameras));
          this._renderTarget = outputTarget; this._activeCubeFace = previousFace; this._activeMipmapLevel = previousMip;
          outputState = {autoClear: this.autoClear, xrEnabled: this.xr.enabled};
          this.autoClear = false; this.xr.enabled = false;
          const quad = this._outputQuad(internal);
          if (captureCameras) {
            this.info.calls++; this.info.render.calls++; this.info.render.frameCalls++;
          }
          const output = this._makeRequest(quad, quad.camera, null, captureCameras, clearOnly ? previousTarget : outputTarget);
          output.outputTransform = {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
            outputColorSpace: this.outputColorSpace};
          requests.push(output);
        }
        for (const request of requests) request.outputPass = true;
        completed = true;
        return requests;
      } catch (error) {
        for (const request of requests) this._releaseRequest(request);
        throw error;
      } finally {
        // Source render() leaves the effective output selected after its
        // conversion pass. clear() and compileAsync() preserve the selector.
        this._renderTarget = completed && convert && !clearOnly && captureCameras ? outputTarget : previousTarget;
        this._activeCubeFace = previousFace; this._activeMipmapLevel = previousMip;
        if (outputState) { this.autoClear = outputState.autoClear; this.xr.enabled = outputState.xrEnabled; }
      }
    }
    _captureNodeFrame({scene, camera, destination, overrideMaterial, captureCameras, mrt = this._mrt, outputTransform = null, renderId,
      sourceTarget = destination === this ? null : destination.target}) {
      const target = destination === this ? null : destination.target;
      const stencilAttachment = target ? destination.shape.depthAttachment && destination.shape.depthFormat?.endsWith('-stencil8') === true : this.stencil;
      let raster = this._nodeRaster.get(stencilAttachment);
      if (!raster) { raster = createThreeSceneRaster(THREE, {stencilAttachment, fail}).program; this._nodeRaster.set(stencilAttachment, raster); }
      const backgroundMesh = (this._nodeBackgrounds ??= createThreeNodeBackgrounds(THREE)).get(scene);
      // Pre-init requests can be captured after a PassNode restored the public
      // renderer state. Build and update against that request's source context.
      const previousTarget = this._renderTarget, previousMRT = this._mrt;
      const previousOutput = outputTransform ? {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
        outputColorSpace: this.outputColorSpace} : null;
      let captured;
      const bundleDepth = this._nodeCaptureDepth++;
      try {
        // Direct default output and clear() output quads leave the source's
        // public selector null, although their GPU destination is a target.
        // ScreenNode therefore observes the canvas size on those source paths.
        this._renderTarget = sourceTarget; this._mrt = mrt;
        if (outputTransform) Object.assign(this, outputTransform);
        if (destination === this && this.needsFrameBufferTarget) {
          // Retain source target construction/identity for node observations;
          // actual color storage and submission stay in the owned canvas path.
          this.getCanvasTarget(); this._frameBufferTargets ??= new Map();
          this._renderTarget = THREE.Renderer.prototype._getFrameBufferTarget.call(this);
        }
        captured = captureThreeNodeScene(THREE, scene, camera, {renderer: this, overrideMaterial, raster, backgroundMesh, renderLists: this, renderId,
          bundles: this._nodeBundles ??= createThreeNodeBundles(THREE, {maxEntries: maxDraws, fail}),
          bundleDepth,
          getPrograms: () => this._nodeProgramSupport ??= createThreeNodePrograms({three: THREE, renderer: this, maxPrograms: maxBindings}),
          getTextures: () => this._getSampledTextureInputs(), getStorage: () => this._getStorageInputs(), destination: target,
          withFramebufferCopies: (record, run) => {
            const previous = this._nodeFramebufferCopyRecorder;
            this._nodeFramebufferCopyRecorder = record;
            try { return run(); } finally { this._nodeFramebufferCopyRecorder = previous; }
          },
          maxNodes: sceneLimits.maxNodes ?? 16384, maxDraws, uniforms: captureCameras,
          maxUniformBytes: MAX_NODE_CAPTURE_BYTES - this._nodeCaptureBytes});
      } finally {
        this._nodeCaptureDepth--;
        this._renderTarget = previousTarget; this._mrt = previousMRT;
        if (previousOutput) Object.assign(this, previousOutput);
      }
      if (captured) this._nodeCaptureBytes += captured.bytes;
      return captured;
    }
    _captureMaterialFrame({scene, nodeFrame, overrideMaterial}) {
      const captured = captureThreeMaterialState(THREE, scene, {nodeFrame, overrideMaterial,
        maxNodes: sceneLimits.maxNodes ?? 16384, maxMaterials: maxBindings,
        maxValues: MAX_MATERIAL_CAPTURE_VALUES - this._materialCaptureValues});
      if (captured) this._materialCaptureValues += captured.values;
      return captured;
    }
    _releaseRequest(request) {
      request.releaseFramebuffer?.(); request.releaseFramebuffer = null;
      request.snapshot?.release();
      if (request.materialFrame) {
        this._materialCaptureValues -= request.materialFrame.values; request.materialFrame = null;
      }
      if (request.nodeFrame) {
        const captured = request.nodeFrame;
        for (const lease of captured.textureUses) captured.textureOwner.release(lease);
        for (const lease of captured.storageUses) captured.storageOwner.release(lease);
        for (const lease of captured.bundleGeometries ?? []) lease.release();
        this._nodeCaptureBytes -= captured.bytes; request.nodeFrame = null;
      }
      request.releaseDestination?.(); request.releaseDestination = null;
    }
    _dispatchRequest(request) {
      const {scene, destination, snapshot} = request;
      destination._wanted.set(scene, request.key);
      const entry = destination._dispatcher?.entry(scene);
      if (this._drain || this._preparing || (destination === this &&
          (request.hdr !== this._hdr || request.srgbView !== this._srgbView)) || !entry || entry.key !== request.key || (!entry.clipping && request.overrideMaterial?.clippingPlanes?.length))
        return this._defer(request);
      try {
        const captured = request.nodeFrame;
        if (captured?.textureOwner?.preparation(captured.textureUses)) return this._defer(request);
        this._submit(request); this._releaseRequest(request);
      }
      catch (error) {
        if (isPrepareBoundary(error) || (error?.code === 'THREE_SCENE_MATERIAL' && !entry.clipping && needsClipping(scene, request.overrideMaterial, request.nodeFrame, request.materialFrame)))
          return this._defer(request);
        this._releaseRequest(request);
        throw error;
      }
    }
    _submit({scene, camera, frame, destination, mipLevel = 0, face = 0, generateCubeMipmaps = false, programFrame, snapshot, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, clearOnly, outputPass = false, initializeOutputDepth = null,
      sourceTarget = destination === this ? null : destination.target, deferred = false}) {
      if (initializeOutputDepth) {
        this._targets.assertCurrent(initializeOutputDepth);
        if (this._targetDepthInitialized.has(initializeOutputDepth)) return;
      }
      snapshot?.install();
      nodeFrame?.installAttributes?.();
      const previous = this._programFrame;
      // Immediate source callbacks must still be able to change live program
      // state. Only deferred calls need the queued renderer-state snapshot.
      this._programFrame = deferred ? programFrame : undefined;
      try {
        const issue = () => {
          if (destination === this) this._session.render({scene, camera, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame},
            this._hdr ? {...frame, outputEnabled: nodeFrame?.phases?.output !== false} : frame);
          else {
            // The output pair needs source Renderer's first-use depth clear:
            // its quad loads color, and WebGPU initially stores zero depth.
            // Scope this to output passes and preserve authoritative copies.
            const sourceDepth = destination.target.depthBuffer === true && destination.shape.depthFormat;
            const copiedDepth = mipLevel === 0 && destination.storage?.diagnostics.depthCopies > 0;
            const firstDepth = outputPass && sourceTarget !== null && sourceDepth && !copiedDepth &&
              !this._targetDepthInitialized.has(destination);
            const targetFrame = firstDepth && !clearOnly ? {...frame, depthLoadOp: 'clear'} : frame;
            this._targets.render(destination, (attachments, framebuffer) => destination._dispatcher.render({scene, camera, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame}, {...targetFrame, ...attachments}, framebuffer), {mipLevel, face, generateMipmaps: generateCubeMipmaps});
            // Pinned Renderer tracks this per target, including color-only
            // manual clears; selecting another mip does not reset the flag.
            if (!this._shaderEncodedOutput && sourceDepth) this._targetDepthInitialized.add(destination);
          }
        };
        const textures = nodeFrame?.textureOwner, storage = nodeFrame?.storageOwner;
        const resources = () => {
          if (nodeFrame) nodeFrame = {...nodeFrame, draws: nodeFrame.draws.map(draw => ({...draw,
            nodeResources: [
              ...draw.nodeTextures.map(({group, binding, lease}) => ({group, binding, resource: textures.resolve(lease)})),
              ...draw.nodeStorage.map(({group, binding, lease}) => ({group, binding, resource: storage.resolve(lease)})),
            ]}))};
          issue();
        };
        const readStorage = () => storage && nodeFrame.storageUses.length
          ? storage.withReads(nodeFrame.storageUses, resources) : resources();
        if (textures && nodeFrame.textureUses.length) {
          textures.validateDispatch(nodeFrame.textureUses);
          textures.withDispatch(nodeFrame.textureUses, readStorage);
        } else readStorage();
      } finally { this._programFrame = previous; }
      if (initializeOutputDepth) this._targetDepthInitialized.add(initializeOutputDepth);
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
      try { this._enqueue(request); } catch (error) { this._releaseRequest(request); throw error; }
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
          for (const pending of this._pending) { this._releaseRequest(pending); pending.reject?.(error); }
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
      pending: while (this._pending.length && !this._disposed) {
        const request = this._pending[0];
        if (request.nodeCaptureFailure) throw request.nodeCaptureFailure.error;
        if (request.framebufferCopy) {
          try { const preparation = request.issue(); if (preparation?.then) await preparation; }
          finally { this._releaseRequest(request); }
          this._pending.shift(); continue;
        }
        if (request.readback || request.transfer) {
          // Image compute can await private source readiness before submitting.
          // Ordinary transfers already submit synchronously; their mapping and
          // GPU-completion fences must not stall later commands.
          try {
            const result = request.issue();
            request.resolve(request.awaitSubmission ? await result : result);
          } catch (error) { request.reject(error); }
          this._pending.shift(); continue;
        }
        const {destination, scene} = request;
        for (let attempts = 0; ; attempts++) {
          if (attempts > 8) fail('PREPARE', 'Source structure kept changing during preparation');
          if (destination !== this) this._targets.assertCurrent(destination);
          destination._wanted.set(scene, sceneKey(this, scene, request.backgroundState));
          await this._prepare(destination, request);
          if (this._disposed) return;
          // First-use node capture can accept shadow passes while this parent
          // is already in the FIFO. They must submit before its depth reads.
          // Preparation and callback capture remain valid while those children
          // drain; never shift or release the displaced parent here.
          if (this._pending[0] !== request) continue pending;
          if (request.nodeCaptureFailure) throw request.nodeCaptureFailure.error;
          try {
            const entry = destination._dispatcher?.entry(scene);
            if (!entry || entry.key !== sceneKey(this, scene, request.backgroundState)) fail('PREPARE', 'Scene profile changed during preparation');
            this._submit(request); break;
          } catch (error) {
            if (!isPrepareBoundary(error)) throw error;
          }
        }
        this._releaseRequest(request);
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
      const requests = this._makeOutputRequests(compiled, camera, null, false);
      try {
        await this.init();
        if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
        while (this._drain) await this._drain;
        for (const request of requests) {
          request.destination._wanted.set(request.scene, request.key);
          await this._prepare(request.destination, request);
        }
      } finally { for (const request of requests) this._releaseRequest(request); }
    }
    /** All scene preparations share one queue; targets never rebuild the canvas. */
    _prepare(destination = this, request = null) {
      const run = (this._preparing ?? Promise.resolve()).catch(() => {}).then(async () => {
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        if (request?.nodeCaptureDeferred) {
          // The source's async-before-init path observes scene inputs at this
          // boundary. Refresh the selector before callbacks so fixed background
          // ownership and native sky packets describe that same accepted frame.
          request.backgroundState = captureThreeBackground(THREE, request.scene);
          request.key = sceneKey(this, request.scene, request.backgroundState);
          destination._wanted.set(request.scene, request.key);
          const pendingBefore = this._pending.length;
          try {
            request.nodeFrame = this._captureNodeFrame(request);
            request.matricesUpdated ||= !!request.nodeFrame;
            if (request.captureCameras && !request.camera.isArrayCamera) {
              if (!request.matricesUpdated) this._updateNativeMatrices(request.scene, request.camera);
              request.camera = captureThreeCamera(THREE, request.camera); request.matricesUpdated = true;
            }
            request.materialFrame = this._captureMaterialFrame(request);
          } catch (error) {
            request.nodeCaptureFailure = {error};
          } finally {
            request.nodeCaptureDeferred = false;
            const parent = this._pending.indexOf(request);
            if (parent !== -1 && this._pending.length > pendingBefore) {
              // Capture is synchronous: this suffix contains precisely the
              // recursive calls accepted by its source lifecycle. Preserve
              // their order before this parent and leave preexisting later
              // requests behind it, including on a late capture failure.
              const prerequisites = this._pending.splice(pendingBefore);
              this._pending.splice(parent, 0, ...prerequisites);
            }
          }
          if (request.nodeCaptureFailure) {
            // Accepted children survive a failed parent capture just as they
            // do on the initialized synchronous path. Its error reaches the
            // caller after those writes; the parent is never submitted.
            if (this._pending.includes(request)) return;
            throw request.nodeCaptureFailure.error;
          }
        }
        const captured = request?.nodeFrame;
        const textureReady = captured?.textureOwner?.preparation(captured.textureUses);
        if (textureReady) await textureReady;
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        const previous = this._programFrame, previousRoot = destination._preparationRoot;
        const previousOptions = destination._preparationOptions;
        this._programFrame = request?.programFrame;
        destination._preparationRoot = request?.scene ?? null;
        destination._preparationOptions = request ? {overrideMaterial: request.overrideMaterial, nodeFrame: request.nodeFrame,
          backgroundState: request.backgroundState, materialFrame: request.materialFrame} : undefined;
        try {
          request?.snapshot?.install();
          captured?.prepareAttributes?.();
          captured?.installAttributes?.();
          if (destination !== this) {
            this._targets.ensure(destination);
            if (!destination._dispatcher) createDispatcher(this, destination)(this._device,
              {...destination.storage.rendererOptions, depthFormat: destination.shape.depthAttachment ? destination.shape.depthFormat : null},
              {signal: destination.lifetime.signal});
            await destination._dispatcher.prepare();
            this._targets.assertCurrent(destination);
          } else {
            const tone = request ? TONE_MAPPINGS.get(request.programFrame.toneMapping) : this._toneMapping();
            const selectOutput = () => {
              const hdr = this._wantsHdr(tone, request?.programFrame.outputColorSpace ?? this.outputColorSpace);
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
            const hdr = selectOutput(), srgbView = this._shaderEncodedOutput ? hdr : request?.srgbView ?? this._canvasSrgbView();
            if (request) request.srgbView = srgbView;
            if (hdr !== this._hdr || srgbView !== this._srgbView) { this._session.dispose(); await this._createSession(hdr, srgbView); }
            this._rebuild = false;
            try { await this._session.prepare(); }
            catch (error) {
              if (error?.code !== 'THREE_SCENE_TONE_MAPPING' || this._hdrFallback === true || globalThis.F3D_NO_FALLBACK === true) throw error;
              this._hdrFallback = true;
              this._session.dispose();
              const fallbackHdr = selectOutput();
              if (request && this._shaderEncodedOutput) request.srgbView = fallbackHdr;
              await this._createSession(fallbackHdr, this._shaderEncodedOutput ? fallbackHdr : srgbView);
              this._rebuild = false;
              await this._session.prepare();
            }
          }
          this.info.f3d.preparations++;
        } finally { this._programFrame = previous; destination._preparationRoot = previousRoot; destination._preparationOptions = previousOptions; }
      });
      this._preparing = run;
      run.finally(() => { if (this._preparing === run) this._preparing = null; }).catch(() => {});
      return run;
    }
    async waitForGPU() {
      while (this._drain) await this._drain;
      if (this._preparing) await this._preparing;
      if (this._session) await this._session.whenIdle();
      await this._transfers.whenIdle();
      await this._targets.whenIdle();
    }

    // ---- animation loop (source Animation semantics: callback(time, xrFrame)) ----
    async setAnimationLoop(callback) {
      if (this._initialized === false && callback !== null) await this.init();
      this._setLoop(callback);
    }
    _setLoop(callback) {
      const host = this.domElement.ownerDocument?.defaultView ?? globalThis;
      if (this.isWebGPURenderer) {
        this._animationCallback = callback;
        if (!this._initialized || this._animationStarted || this._disposed) return;
        this._animationStarted = true; this._animationHost = host;
        let starting = true;
        const schedule = () => {
          if (this._animationStarted && !this._disposed)
            this._animationHandle = host.requestAnimationFrame?.(tick) ?? null;
        };
        const tick = (time, xrFrame = null) => {
          if (!this._animationStarted || this._disposed) return;
          const initial = starting; starting = false;
          this._animationHandle = null;
          if (!initial && this._drain && this._pending.some(request =>
            request.animationBarrier === true || !request.transfer && !request.readback)) {
            // Source render preparation blocks its callback synchronously.
            // Our asynchronous equivalent must finish submitting that accepted
            // work before another animation callback can fill the bounded FIFO.
            // Explicit asynchronous compilation/readbacks alone do not block
            // the source frame clock, including Animation.start() during init.
            // Resume on a fresh RAF timestamp, without replaying this host tick
            // or waiting for execution of the submitted GPU commands.
            if (!this._animationWaiting) {
              const pending = this._animationWaiting = this._drain;
              const resume = () => {
                if (this._animationWaiting !== pending) return;
                this._animationWaiting = null; schedule();
              };
              pending.then(resume, resume);
            }
            return;
          }
          schedule();
          this._advanceAnimationFrame();
          this._animationCallback?.(time, xrFrame);
        };
        tick();
        return;
      }
      if (this._animationHandle !== null) { host.cancelAnimationFrame?.(this._animationHandle); this._animationHandle = null; }
      if (this._inspector.isRunning) this._inspector.finish();
      this._animationCallback = callback;
      if (callback === null || this._disposed) return;
      const tick = time => {
        if (this._animationCallback !== callback || this._disposed) return;
        this._animationHandle = host.requestAnimationFrame(tick);
        this._advanceAnimationFrame();
        callback(time, null);
      };
      this._animationHandle = host.requestAnimationFrame(tick);
    }
    _advanceAnimationFrame() {
      // Source Animation order: finish the previous inspected frame, then begin.
      if (this._inspector.isRunning) this._inspector.finish();
      if (this.info.autoReset === true) this.info.reset();
      this._nodes.nodeFrame.update();
      this._renderUid = 0;
      this.info.frame = this._nodes.nodeFrame.frameId;
      this._inspector.begin();
    }
    getAnimationLoop() { return this._animationCallback; }

    dispose() {
      if (this._disposed) return;
      this.setAnimationLoop(null);
      this._disposed = true;
      if (this._animationStarted) {
        this._animationStarted = false;
        if (this._animationHandle !== null) this._animationHost?.cancelAnimationFrame?.(this._animationHandle);
        this._animationHandle = null;
        if (this._inspector.isRunning) this._inspector.finish();
      }
      const stopped = new F3DRendererError('DISPOSED', 'Renderer is disposed');
      this._targetLifetime.abort(stopped);
      this._targets.dispose();
      for (const request of this._pending) { this._releaseRequest(request); request.reject?.(stopped); }
      this._pending = [];
      this._inspector.dispose?.();
      if (!this._drain) this._release();
      else this._drain.finally(() => this._release()).catch(() => {});
    }
  }
  const nativeInit = WebGPURenderer.prototype.init;
  return withThreeCompute(WebGPURenderer, THREE);
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
  // Native WGSL leaves render-target UVs unchanged; WebGL programs retain their
  // bottom-left sampling convention. Keep this owner policy fixed per facade.
  const Base = createWebGPURendererClass(THREE, {...options, targetTextureFlipY: true});
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
    readRenderTargetPixelsAsync(target, x, y, width, height, buffer, activeCubeFace, textureIndex = 0) {
      if (!ArrayBuffer.isView(buffer))
        return Promise.reject(new F3DRendererError('UNSUPPORTED', 'Supply an output array for the selected color attachment'));
      return this._readTargetPixels(target, {x, y, width, height, output: buffer, flipY: true,
        face: activeCubeFace, colorIndex: textureIndex});
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

/** The native WebGPU route retains the pinned PMREM construction/filtering
 * helper and supplies this renderer for every generated draw. The helper does
 * not construct a source renderer or backend. Its real CubeUV targets can be
 * sampled by native node materials and backgrounds through the target owner.
 *
 * The WebGL-authored route still describes the source for the scene bridge's
 * environment owner to prepare at its preparation boundary:
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
  const nativeGenerator = renderer => {
    const generator = new THREE.PMREMGenerator(renderer), source = THREE.PMREMGenerator.prototype;
    const releaseWhenReady = release => {
      if (!renderer._drain && !renderer._preparing) { release(); return; }
      renderer.waitForGPU().then(release, release).catch(error => { renderer._deferredError ??= error; });
    };
    const retire = (method, all) => {
      // r186 replaces these fields on a size change. Capture their resource
      // identities now: cleanup of the live generator after the queue drains
      // would otherwise dispose the next generation's targets and materials.
      const retired = Object.assign(Object.create(source), {
        _blurMaterial: generator._blurMaterial, _ggxMaterial: generator._ggxMaterial,
        _pingPongRenderTarget: generator._pingPongRenderTarget,
        _lodMeshes: generator._lodMeshes.map(mesh => ({geometry: mesh.geometry})),
      });
      if (all) Object.assign(retired, {
        _cubemapMaterial: generator._cubemapMaterial, _equirectMaterial: generator._equirectMaterial,
        _backgroundBox: generator._backgroundBox && {
          geometry: generator._backgroundBox.geometry, material: generator._backgroundBox.material,
        },
      });
      const release = () => source[method].call(retired);
      // Submitted GPU work already owns its resources. Deferred preparation and
      // submission still need the source objects to remain valid. Both rejected
      // drains and successful drains must release the retired private resources;
      // the returned public output target is deliberately absent from this view.
      releaseWhenReady(release);
    };
    Object.defineProperties(generator, {
      _dispose: {configurable: true, writable: true, value() { retire('_dispose', false); }},
      dispose: {configurable: true, writable: true, value() { retire('dispose', true); }},
      _sceneToCubeUV: {configurable: true, writable: true, value(...args) {
        // The source helper reuses a mutable fixed-material background box.
        // Each generation needs its own color while previous draws are queued.
        const previous = generator._backgroundBox; generator._backgroundBox = null;
        if (previous) releaseWhenReady(() => { previous.geometry.dispose(); previous.material.dispose(); });
        return source._sceneToCubeUV.apply(generator, args);
      }},
    });
    return generator;
  };
  class PMREMGenerator {
    constructor(renderer) {
      if (!renderer?.isF3DRenderer) {
        if (exactBackend) return new exactBackend(renderer);
        fail('ROUTE', 'This PMREMGenerator serves the new-backend renderer route');
      }
      if (renderer.isWebGPURenderer === true && renderer._shaderEncodedOutput !== true &&
          typeof THREE.WGSLNodeBuilder === 'function' && typeof THREE.PMREMGenerator === 'function')
        return nativeGenerator(renderer);
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
