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
 *   Animation callbacks on both renderer surfaces wait for deferred submission before
 *   starting another frame; this bounds cold-start work without waiting for
 *   GPU completion or discarding accepted render/compute requests.
 * - Admission failures discovered during deferred preparation are thrown from
 *   the next `render()`/`compileAsync()` call instead of the original call.
 * - Unsupported source state (advanced render targets, XR presentation, custom tone
 *   mapping in direct native canvas rendering, unsupported output color spaces,
 *   transparent clears with tone mapping) throws `F3DRendererError` rather than
 *   silently rendering something else.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createThreeRendererCanvases} from './three_renderer_canvas.mjs';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {createThreeRenderTargets, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {createThreeRendererTransfers} from './three_renderer_transfers.mjs';
import {createThreeTimestampQueries} from './three_timestamp_queries.mjs';
import {createThreeBackendOperations} from './three_backend_operations.mjs';
import {createThreePassSnapshots, captureThreeArrayCamera, captureThreeCamera, captureThreeBackground, updateThreeCameraProjection} from './three_pass_snapshot.mjs';
import {captureThreeMaterialState} from './three_material_snapshot.mjs';
import {createThreePassInstances} from './three_pass_instances.mjs';
import {threeWebGPUStencilValue} from './three_program_stencil.mjs';
import {registerThreePassState, captureThreePassState, finishThreePassState, requiresThreePassCamera, getThreePassExcludedObjects} from './three_pass_state.mjs';
import {registerThreeGpuComputation} from './three_gpu_computation.mjs';
import {createThreeProgramSupport} from './three_program.mjs';
import {gpuProgramModule} from './gpu_program_modules.mjs';
import {createThreeProgramPMREM} from './three_program_pmrem.mjs';
import {hasThreeWebGLTextureResidency} from './three_textures.mjs';
import {createThreeProgramShadows} from './three_program_shadows.mjs';
import {createThreeProgramTransmission} from './three_program_transmission.mjs';
import {withThreeCompute} from './three_compute.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {inspectThreeMRT} from './three_node_passes.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {createThreeNodeBundles} from './three_node_bundles.mjs';
import {createThreeOcclusionQueries} from './three_occlusion_queries.mjs';
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
// runs only in a source shader/TSL output graph, never in a fixed output pass.
const TONE_MAPPINGS = new Map([[0, 'none'], [1, 'linear'], [2, 'reinhard'], [3, 'cineon'], [4, 'aces-filmic'], [6, 'agx'], [7, 'neutral']]);
const PARAMETERS = new Set(['outputType', 'canvas', 'antialias', 'alpha', 'depth', 'stencil', 'samples', 'forceWebGL',
  'logarithmicDepthBuffer', 'reversedDepthBuffer', 'powerPreference', 'requiredLimits', 'requiredFeatures',
  'device', 'outputBufferType', 'multiview', 'trackTimestamp', 'colorBufferType', 'getFallback', 'context']);
// Source GPU bakers issue whole batches synchronously before preparation can
// yield: even LightProbeGrid's default 2x2x2 bake submits 84 render operations.
// Keep a bounded FIFO; captured uniform/material bytes have separate limits.
const MAX_PENDING = 1024;
// Only the retained PMREM helper's own conversion/filter meshes use the
// larger immutable pass profile. Ordinary application meshes keep their route.
const PMREM_CAPTURES = new WeakMap();
const PMREM_INPUTS = new WeakMap();
const PMREM_SOURCES = new WeakMap();
const PMREM_EXECUTIONS = new WeakMap();
const MAX_NODE_CAPTURE_BYTES = 64 * 1024 * 1024;
const MAX_MATERIAL_CAPTURE_VALUES = 1024 * 1024;
const SOURCE_RENDERER_CONSTRUCTORS = new WeakMap();
const ordinaryHasInstance = Function.prototype[Symbol.hasInstance];

// Make this choice at construction, before inspecting source options or owning
// a canvas. The retained constructor keeps its own device/fallback semantics.
// Adapter or device-request failures after native construction do not authorize
// replacing an already exposed renderer with another owner.
const hasWebGPUHost = gpu => typeof (gpu ?? globalThis.navigator?.gpu)?.requestAdapter === 'function';

// The compiler keeps opaque option receivers on the source-constructor path.
// This descriptor check covers ordinary option objects; it is not a purity
// test for Proxies. A selector accessor or inherited/custom container goes
// to the source before reading options, rather than reading and replaying it.
function requiresSourceParameters(parameters, selectors) {
  if (!parameters || typeof parameters !== 'object') return true;
  const prototype = Object.getPrototypeOf(parameters);
  if (prototype !== null && prototype !== Object.prototype) return true;
  for (const name of selectors) {
    const descriptor = Object.getOwnPropertyDescriptor(parameters, name) ??
      (prototype && Object.getOwnPropertyDescriptor(prototype, name));
    if (descriptor && (!Object.hasOwn(descriptor, 'value') || descriptor.value)) return true;
  }
  return false;
}

/** One native device observer per renderer, independent of canvas sessions.
 * Keep the source's event-handler slot and payloads. A disposed owner must not
 * remain reachable from the device's non-removable lost Promise continuation. */
function observeRendererDevice(renderer, device) {
  let owner = renderer;
  device.lost?.then?.(info => {
    const current = owner;
    if (!current || info.reason === 'destroyed') return;
    current.onDeviceLost({api: 'WebGPU', message: info.message || 'Unknown reason',
      reason: info.reason || null, originalEvent: info});
  });
  const onError = event => {
    const current = owner;
    if (!current) return;
    const gpuError = event.error;
    current.onError({api: 'WebGPU', type: gpuError && gpuError.constructor ? gpuError.constructor.name : 'GPUError',
      message: (gpuError && gpuError.message) || 'Unknown uncaptured GPU error', originalEvent: event});
  };
  device.onuncapturederror = onError;
  return () => {
    owner = null;
    if (device.onuncapturederror === onError) device.onuncapturederror = null;
  };
}

/** Compiler-private dispatch for statically source-selected construction sites.
 * No option is read or copied, and unknown constructors keep their own behavior. */
export function sourceThreeRendererConstructor(Constructor) {
  return SOURCE_RENDERER_CONSTRUCTORS.get(Constructor) ?? Constructor;
}

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
    maxTransferBytes = 128 * 1024 * 1024, maxTransfers = 64, maxPendingRenders = MAX_PENDING, targetTextureFlipY = false} = classOptions;
  if (![maxTransferBytes, maxTransfers].every(n => Number.isSafeInteger(n) && n > 0))
    fail('OPTIONS', 'Expected positive renderer transfer byte and request limits');
  if (!Number.isSafeInteger(maxPendingRenders) || maxPendingRenders < 1)
    fail('OPTIONS', 'Expected a positive pending renderer request limit');
  if (exactBackend !== null && typeof exactBackend !== 'function') fail('OPTIONS', 'exactBackend must be a renderer constructor');
  if (typeof targetTextureFlipY !== 'boolean') fail('OPTIONS', 'Expected a target sampling orientation');
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1 || !Number.isSafeInteger(maxBindings) || maxBindings < 1)
    fail('OPTIONS', 'Expected positive draw and binding capacities');
  const {NoToneMapping, SRGBColorSpace} = THREE;
  const sourceError = THREE.error ?? (message => console.error('THREE.' + message));
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
    if (backgroundState?.nativeSceneProfile) return backgroundState.nativeSceneProfile.key;
    const selected = backgroundState ? backgroundState.background : scene.background;
    const background = isScene && selected !== null && selected !== undefined && !selected.isColor;
    return `${isScene && scene.fog ? 1 : 0}${isScene && scene.environment ? 1 : 0}${background ? 1 : 0}${renderer.shadowMap.enabled ? 1 : 0}${renderer._needsGlobalClipping?.() ? 1 : 0}`;
  }
  /** Preparation-time traversal for options that depend on materials. */
  function needsClipping(scene, overrideMaterial = scene.isScene === true ? scene.overrideMaterial : null, nodeFrame = null, materialFrame = null) {
    if (nodeFrame?.sceneHierarchy) return false;
    const state = material => materialFrame?.materials.get(material) ?? material;
    let clipping = false;
    scene.traverse(object => {
      // A queued native render owns its visibility selection. Hidden source
      // objects must not force material admission during later preparation.
      if (nodeFrame?.excludedObjects?.has(object) || materialFrame?.excludedObjects?.has(object)) return;
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
        retire(scene) { const entry = entries.get(scene); if (entry) { entries.delete(scene); entry.bridge.dispose(); } },
        render({scene, camera, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, sortState, onDrawCallbacks}, frame, framebuffer, depthbuffer) {
          const entry = entries.get(scene);
          if (!entry) fail('PREPARE', 'Scene has not been prepared');
          entry.bridge.render(camera, frame, {overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, sortState, framebuffer, depthbuffer, onDrawCallbacks});
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
              const {privateSnapshot, ...bridgeOptions} = passOptions ?? {};
              await previous.bridge.prepare(passOptions ? bridgeOptions : undefined);
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
              sourceAttributeCallbacks: owner._shaderEncodedOutput === true,
              instanceHistory: owner._shaderEncodedOutput && !passOptions?.privateSnapshot ? owner._passInstances.history : null,
              textures: owner._targets.bindingsFor(destination === owner ? null : destination.target,
                owner._shaderEncodedOutput ? owner._framebufferBindings(sceneLimits.textures) : sceneLimits.textures,
                owner._shaderEncodedOutput ? texture => owner._programFrame?.passTextureEntries?.get(texture) : null),
              renderer: {...attachments, instancing: true, renderBundles: true, maxDraws,
                ...(destination !== owner || owner._hdr || !owner._shaderEncodedOutput ? {} : {outputTransfer: 'srgb'})},
              textureTransforms: true, alphaMaps: true,
              // ShaderMaterial programs (WebGLRenderer semantics) on the WebGL surface.
              program: owner._shaderEncodedOutput && (destination !== owner || !owner._hdr) ? destination._programSupport ??= createThreeProgramSupport({three: THREE, pmrem: createThreeProgramPMREM, shadows: createThreeProgramShadows, transmission: createThreeProgramTransmission,
                state: () => ({toneMapping: owner.toneMapping, toneMappingExposure: owner.toneMappingExposure, outputColorSpace: owner.outputColorSpace,
                  canvasOutput: destination === owner,
                  pixelRatio: owner._pixelRatio, height: owner._height, shadowMap: owner.shadowMap, shadowMapType: owner.shadowMap.type, renderer: owner,
                  logarithmicDepthBuffer: owner.logarithmicDepthBuffer === true,
                  reversedDepthBuffer: owner.reversedDepthBuffer === true,
                  floatLinear: device.features?.has?.('float32-filterable') === true, ...owner._programFrame,
                  // Public targets keep top-down native storage (the target owner
                  // provides UV-oriented sampling). Do not use the compiler's
                  // renderTarget:true/GL-row profile reserved for shadow/PMREM maps.
                  // Native sRGB attachments perform their own transfer function.
                  // Keep source point-material size/scale uniforms tied to the
                  // renderer's pixel ratio and logical height, even offscreen.
                  ...(destination === owner ? {} : {toneMapping: NoToneMapping, outputColorSpace: THREE.LinearSRGBColorSpace})})}) : null,
              fog: passOptions?.backgroundState?.nativeSceneProfile ? null : sourceScene && scene.fog ? {} : null,
              environment: passOptions?.backgroundState?.nativeSceneProfile ? null : sourceScene && scene.environment ? {} : null,
              background: background ? {} : null,
              shadow: (passOptions?.backgroundState?.nativeSceneProfile?.shadows ?? owner.shadowMap.enabled) ? {} : null,
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
      const sourcePrototype = exactBackend && overriddenInitPrototype(this);
      if (sourcePrototype) {
        if (typeof exactInit !== 'function') fail('ROUTE', 'An overridden init requires an exact backend with its source init method');
        const retained = new exactBackend(...arguments);
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
      if (exactBackend && !hasWebGPUHost(defaultGpu)) return new exactBackend(...arguments);
      if (exactBackend && requiresSourceParameters(parameters, ['forceWebGL', 'context']))
        return new exactBackend(...arguments);
      if (!parameters || typeof parameters !== 'object') fail('OPTIONS', 'Expected renderer parameters');
      // Choose the owner before other option getters can mutate the selectors.
      // Source-selected constructors receive the untouched original arguments.
      if (!exactBackend && (parameters.forceWebGL || parameters.context))
        fail('ROUTE', 'forceWebGL or a supplied context selects the exact backend route, not this renderer');
      for (const key of Object.keys(parameters)) if (!(this._parameterNames?.() ?? PARAMETERS).has(key)) fail('OPTIONS', `Unknown renderer parameter: ${key}`);
      const {
        canvas = null, antialias = false, alpha = true, depth = true, stencil = false, samples = 0,
        logarithmicDepthBuffer = false, reversedDepthBuffer = false,
      } = parameters;
      this.isRenderer = true;
      this.isWebGPURenderer = true;
      this.isF3DRenderer = true;
      this._canvases = createThreeRendererCanvases(THREE, this, canvas ?? THREE.createCanvasElement(), fail);
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
      this._opaqueSort = null;
      this._transparentSort = null;
      this.opaque = true;
      this.transparent = true;
      this.shadowMap = {enabled: false, transmitted: false, type: THREE.PCFShadowMap};
      this.lighting = THREE.Lighting ? new THREE.Lighting() : {enabled: true};
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
        memoryMap: new WeakMap(),
        reset() {
          this.render.drawCalls = this.render.frameCalls = this.compute.frameCalls = 0;
          this.render.triangles = this.render.points = this.render.lines = 0;
        },
        dispose() {
          this.reset();
          this.calls = this.render.calls = this.compute.calls = 0;
          this.render.timestamp = this.compute.timestamp = 0;
          for (const key in this.memory) this.memory[key] = 0;
          this.memoryMap = new WeakMap();
        },
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
      this._outputQuadLeases = new WeakMap();
      this._targetDepthInitialized = new WeakSet();
      this._outputBufferType = parameters.outputBufferType ?? parameters.colorBufferType ?? THREE.HalfFloatType;
      this._activeCubeFace = 0;
      this._activeMipmapLevel = 0;
      this._targetLifetime = new AbortController();
      this._targets = createThreeRenderTargets(THREE, {getDevice: () => this._device, flipY: targetTextureFlipY});
      this._passSnapshots = createThreePassSnapshots(THREE, {targets: this._targets});
      this._passInstances = createThreePassInstances(THREE, {maxBytes: sceneLimits.maxInstanceBytes ?? 128 * 1024 * 1024,
        maxMeshes: sceneLimits.maxInstanceMeshes ?? Math.min(maxBindings, 4096)});
      this._nodeProgramSupport = null;
      this._nodeBackgrounds = null;
      this._nodeBundles = null;
      this._nodeCaptureDepth = 0;
      this._occlusionContext = null;
      this._occlusionContexts = new WeakMap();
      this._occlusionObjects = new WeakSet();
      this._nodeCaptureBytes = 0;
      this._materialCaptureValues = 0;
      this._nodeRaster = new Map();
      this._renderObjectFunction = null;
      this._currentRenderObjectFunction = null;
      this._handleObjectFunction = null;
      this._cacheShadowNodes = new WeakMap();
      this._currentSourceMaterial = null;
      this._mrt = null;
      this._isDeviceLost = false;
      this.onDeviceLost = this._onDeviceLost;
      this.onError = this._onError;
      this._initialized = false;
      this._initPromise = null;
      this._session = null;
      this._dispatcher = null;
      this._device = null;
      this._ownsDevice = false;
      this._releaseDeviceEvents = null;
      this._rebuild = false;
      this._preparing = null;
      this._hdr = false;
      this._srgbView = true;
      this._outputProfileKey = null;
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
      this._disposePromise = null;
      this._transfers = createThreeRendererTransfers(this, F3DRendererError, {maxBytes: maxTransferBytes, maxRequests: maxTransfers});
      Object.defineProperties(this.info.f3d, {
        pendingTransfers: {enumerable: true, get: () => this._transfers.pendingRequests},
        pendingTransferBytes: {enumerable: true, get: () => this._transfers.allocatedBytes},
      });
      this._renderUid = 0;
      const timestamps = this._timestamps = createThreeTimestampQueries({getDevice: () => this._device,
        enabled: () => this._backend?.trackTimestamp === true, signal: this._targetLifetime.signal, warn: THREE.warnOnce});
      this._backend = Object.defineProperties(createThreeBackendOperations(this), Object.getOwnPropertyDescriptors({isWebGPUBackend: true, isF3DBackend: true, device: null, trackTimestamp: parameters.trackTimestamp === true,
        get hasTimestamp() { return timestamps.supported; },
        capabilities: {getUniformBufferLimit: () => this._device.limits.maxUniformBufferBindingSize},
        hasTimestampQuery: timestamps.hasTimestampQuery, getTimestampUID: timestamps.getTimestampUID,
        getTimestamp: timestamps.getTimestamp, getTimestampFrames: timestamps.getTimestampFrames,
        resolveTimestampsAsync: type => this.resolveTimestampsAsync(type), delete: target => this._canvases.forget(target)}));
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
    _onDeviceLost(info) {
      let message = `THREE.WebGPURenderer: ${info.api} Device Lost:\n\nMessage: ${info.message}`;
      if (info.reason) message += `\nReason: ${info.reason}`;
      sourceError(message);
      this._isDeviceLost = true;
    }
    _onError(info) {
      let message = `WebGPURenderer: Uncaptured ${info.api} ${info.type}`;
      if (info.message) message += `: ${info.message}`;
      sourceError(message);
    }
    hasFeature(name) { return !!this._device?.features?.has(name); }
    async hasFeatureAsync(name) { await this.init(); return this.hasFeature(name); }
    async resolveTimestampsAsync(type = 'render') {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (!this._initialized) await this.init();
      // Resolve at this FIFO position, after already accepted native passes.
      // Mapping the result never holds later render/compute submissions up.
      const duration = await (this._drain || this._preparing || this._timestampScope
        ? new Promise((resolve, reject) => this._enqueue({readback: true,
          issue: () => this._timestamps.resolve(type), resolve, reject}))
        : this._timestamps.resolve(type));
      if (duration !== undefined) this.info[type].timestamp = duration;
      return duration;
    }

    get domElement() { return this._canvasTarget.domElement; }
    get initialized() { return this._initialized; }
    hasInitialized() { return this._initialized; }
    get coordinateSystem() { return THREE.WebGPUCoordinateSystem; }
    get isOutputTarget() { return this._renderTarget === null || this._renderTarget === this._outputRenderTarget; }
    get currentToneMapping() { return this.isOutputTarget ? this.toneMapping : NoToneMapping; }
    get currentColorSpace() { return this.isOutputTarget ? this.outputColorSpace : THREE.LinearSRGBColorSpace; }
    get needsFrameBufferTarget() { return this.currentToneMapping !== NoToneMapping || this.currentColorSpace !== THREE.LinearSRGBColorSpace; }
    get currentSamples() { return this._renderTarget ? this._renderTarget.samples : this.needsFrameBufferTarget ? 0 : this.samples; }
    get backend() { this._backend.device = this._device; return this._backend; }

    init() {
      if (retainedInitReceivers.has(this)) return exactInit.call(this);
      if (this._initPromise) return this._initPromise;
      this._initPromise = this._init();
      return this._initPromise;
    }
    async _init() {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this.stencil && !this.depth) fail('UNSUPPORTED', 'Stencil framebuffers currently require a depth attachment');
      const p = this._parameters;
      if (p.outputType !== undefined && ![THREE.UnsignedByteType, THREE.HalfFloatType].includes(p.outputType))
        fail('UNSUPPORTED', 'Native canvas output requires unsigned-byte or half-float storage');
      if (this.isWebGPURenderer && ![THREE.UnsignedByteType, THREE.HalfFloatType, THREE.FloatType].includes(this._outputBufferType))
        fail('UNSUPPORTED', 'Output buffers require unsigned-byte, half-float or float texture storage');
      const signal = this._targetLifetime.signal;
      let rejectStopped;
      const stopped = new Promise((_, reject) => { rejectStopped = reject; });
      stopped.catch(() => {});
      const live = () => { if (signal.aborted) throw signal.reason; };
      const abort = () => rejectStopped(signal.reason);
      const wait = promise => Promise.race([promise, stopped]);
      signal.addEventListener('abort', abort, {once: true});
      try {
      live();
      // The renderer owns one device for its lifetime so the output path can be
      // rebuilt (tone mapping on/off) without losing it. A supplied device is borrowed.
      if (p.device) this._device = p.device;
      else {
        const gpu = defaultGpu ?? globalThis.navigator?.gpu;
        if (!gpu || typeof gpu.requestAdapter !== 'function') fail('UNAVAILABLE', 'WebGPU is unavailable in this host');
        const adapter = await wait(gpu.requestAdapter(p.powerPreference ? {powerPreference: p.powerPreference} : {}));
        live();
        if (!adapter) fail('UNAVAILABLE', 'No WebGPU adapter is available');
        // As r186 WebGPUBackend: request every GPUFeatureName the adapter supports.
        const requiredFeatures = [...new Set([...R186_GPU_FEATURES.filter(f => adapter.features?.has?.(f)), ...(p.requiredFeatures ?? [])])];
        // Browser requests cannot be cancelled. Transfer ownership in this
        // continuation, or retire a late device without publishing it.
        const acquisition = Promise.resolve(adapter.requestDevice({requiredFeatures, requiredLimits: p.requiredLimits ?? {}})).then(device => {
          if (signal.aborted) { device.destroy(); throw signal.reason; }
          this._device = device; this._ownsDevice = true;
        });
        await wait(acquisition);
        live();
      }
      if (this.isWebGPURenderer) this._releaseDeviceEvents = observeRendererDevice(this, this._device);
      // Synchronous WebGL PMREM captures can precede device acquisition. Bind
      // their accepted geometry histories before any renderer bridge uploads.
      this._passSnapshots.bindDevice(this._device);
      this._preferredFormat = p.outputType === THREE.HalfFloatType ? 'rgba16float' :
        p.outputType === THREE.UnsignedByteType ? 'bgra8unorm' :
        (defaultGpu ?? globalThis.navigator?.gpu)?.getPreferredCanvasFormat?.();
      await wait(this._createSession(this._wantsHdr(this._toneMapping())));
      live();
      this._initialized = true;
      // Source Animation.start() advances once during initialization and keeps
      // the node clock running even without a user animation callback.
      if (this.isWebGPURenderer) this._setLoop(null);
      return this;
      } catch (error) {
        // A failed session never transfers device ownership to a usable
        // renderer. Canvas children borrow that device and cannot retire it.
        this._releaseDeviceEvents?.(); this._releaseDeviceEvents = null;
        try { this._canvases.dispose(); }
        finally { if (this._ownsDevice) { this._ownsDevice = false; this._device.destroy(); } }
        throw signal.aborted ? signal.reason : error;
      } finally { signal.removeEventListener('abort', abort); }
    }
    /** Native tone mapping and output color conversion use the source's linear
     * framebuffer target. Preserve working-linear scene color from the first
     * render, including when no tone mapping is selected. */
    async _createSession(hdr, srgbView = this._canvasSrgbView(this.outputColorSpace, hdr),
      canvasTarget = this.getCanvasTarget(), canvasSize = this._bufferSize(), canvas = canvasTarget.domElement,
      outputProfile = this._captureOutputProfile()) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      const release = this._canvases.retain(canvasTarget, canvas);
      this._canvases.activate(canvasTarget, canvas);
      try {
      const sampleCount = this.antialias || this.samples > 0 ? 4 : 1;
      const depthFormat = this.depth ? (this.stencil ? 'depth24plus-stencil8' : 'depth24plus') : null;
      const [width, height] = canvasSize;
      const common = {device: this._device, signal: this._targetLifetime.signal};
      const format = this._preferredFormat ? {format: this._preferredFormat} : {};
      const extended = this._preferredFormat === 'rgba16float';
      // Half-float presentation has no sRGB attachment view. Its output pass
      // performs the selected transfer while keeping the source HDR range.
      const presentation = {...format, srgbView: srgbView && !extended,
        colorSpace: outputProfile?.canvasColorSpace ?? 'srgb',
        toneMappingMode: extended ? 'extended' : 'standard'};
      this._hdr = hdr;
      this._srgbView = srgbView;
      this._outputProfileKey = outputProfile?.key ?? null;
      this._canvasOutputKey = outputProfile?.canvasKey ?? null;
      this._dispatcher = null;
      const session = hdr
        ? await createGpuHdrCanvasRenderer(canvas, createDispatcher(this), {...common,
          target: {width, height, ...presentation, alphaMode: this._compositedAlpha() ? 'premultiplied' : 'opaque'},
          renderTarget: {depthFormat, sampleCount, format: this._outputBufferFormat()},
          outputColorSpace: outputProfile?.transfer ?? (srgbView ? 'srgb' : 'linear'),
          ...(outputProfile?.matrix ? {outputColorMatrix: outputProfile.matrix} : {}),
          sourceOutput: extended || outputProfile?.sourceOutput === true,
          output: {toneMapping: this._toneMapping(this._programFrame?.toneMapping ?? this.toneMapping),
            exposure: this._programFrame?.toneMappingExposure ?? this.toneMappingExposure}})
        : await createGpuCanvasRenderer(canvas, createDispatcher(this), {...common, lazyAttachments: true,
          target: {width, height, depthFormat, sampleCount, ...presentation,
            preserveDrawingBuffer: this._shaderEncodedOutput && this.preserveDrawingBuffer === true,
            frameHistory: this._shaderEncodedOutput === true,
            premultiplyPresentation: this._straightAlphaCanvas === true,
            alphaMode: this._compositedAlpha() ? 'premultiplied' : 'opaque'}});
      if (this._disposed) { session.dispose(); fail('DISPOSED', 'Renderer was disposed during initialization'); }
      this._session = session;
      this._canvases.save();
      } finally { release(); }
    }
    _release(dispose) {
      for (const cache of this._outputQuads.values()) dispose(() => cache.dispose());
      // The retained framebuffer-target listener releases its private target.
      // Source Renderer.dispose() also disposes custom output targets used as
      // framebuffer-cache keys; an unused output selector owns no such target.
      for (const target of this._frameBufferTargets?.keys() ?? [])
        if (target !== this._canvasTarget) dispose(() => target.dispose());
      dispose(() => this._targets.dispose());
      dispose(() => this._canvases.dispose());
      this._wanted.clear();
      dispose(() => this._passSnapshots.dispose());
      dispose(() => this._passInstances.dispose());
      dispose(() => this._nodeBackgrounds?.dispose()); this._nodeBackgrounds = null;
      dispose(() => this._nodeProgramSupport?.dispose());
      this._nodeProgramSupport = null;
      dispose(() => this._programSupport?.dispose()); this._programSupport = null;
      dispose(() => this._nodeBundles?.dispose()); this._nodeBundles = null;
      dispose(() => this._occlusionQueries?.dispose());
      dispose(() => this._canvasTarget?.dispose());
      if (this._ownsDevice) { this._ownsDevice = false; dispose(() => this._device?.destroy()); }
    }

    /** The whole-image tone-mapping output pass. WebGLRenderer (shader-encoded
     * output) tone-maps inside each material's program instead, unless a scene
     * needed a draw the program route cannot take (sticky fallback). Custom
     * operators require their source program and cannot use that fallback. */
    _wantsHdr(tone, space = this.outputColorSpace) { return !this._shaderEncodedOutput && tone !== 'source' &&
      (tone !== 'none' || space !== THREE.LinearSRGBColorSpace) ||
      tone !== 'none' && tone !== 'custom' && this._hdrFallback === true; }
    _outputBufferFormat() {
      return this._outputBufferType === THREE.UnsignedByteType ? 'rgba8unorm' :
        this._outputBufferType === THREE.FloatType ? 'rgba32float' : 'rgba16float';
    }
    _captureOutputProfile(space = this.outputColorSpace) {
      if (this._shaderEncodedOutput) {
        if (this._hdrFallback === true && space !== SRGBColorSpace)
          fail('UNSUPPORTED', 'Non-sRGB WebGL output requires the source material programs');
        const canvasColorSpace = this._drawingBufferColorSpace ?? 'srgb';
        const canvasKey = JSON.stringify([canvasColorSpace, this._drawingBufferVersion ?? 0]);
        return Object.freeze({space, canvasColorSpace, canvasKey,
          transfer: space === THREE.LinearSRGBColorSpace ? 'linear' : 'srgb', matrix: null, sourceOutput: false,
          key: JSON.stringify([space, canvasKey])});
      }
      const management = THREE.ColorManagement;
      if (management && management.workingColorSpace !== THREE.LinearSRGBColorSpace)
        fail('UNSUPPORTED', 'Native output conversion requires the linear-sRGB working color space');
      let transfer = space === SRGBColorSpace ? 'srgb' : 'linear', matrix = null;
      const standard = space === SRGBColorSpace || space === THREE.LinearSRGBColorSpace;
      // RenderOutputNode delegates to ColorSpaceNode, which skips both the
      // primary conversion and transfer when color management is disabled.
      const enabled = management?.enabled !== false;
      if (!enabled || space === THREE.NoColorSpace) transfer = 'linear';
      else if (management) {
        if (typeof space !== 'string' || !management.spaces[space])
          fail('UNSUPPORTED', `Native output color space is not registered: ${space}`);
        const sourceTransfer = management.getTransfer(space);
        if (sourceTransfer !== THREE.LinearTransfer && sourceTransfer !== THREE.SRGBTransfer)
          fail('UNSUPPORTED', 'Native output color spaces require the source linear or sRGB transfer');
        transfer = sourceTransfer === THREE.SRGBTransfer ? 'srgb' : 'linear';
        if (management.getPrimaries(THREE.LinearSRGBColorSpace) !== management.getPrimaries(space)) {
          // Preserve the pinned node's matrix construction and column ordering,
          // rather than substituting a separately derived color-science matrix.
          const elements = management._getMatrix(new THREE.Matrix3(), THREE.LinearSRGBColorSpace, space).elements;
          if (elements.length !== 9 || elements.some(value => typeof value !== 'number' || !Number.isFinite(Math.fround(value))))
            fail('UNSUPPORTED', 'Native output conversion requires a finite 3x3 color matrix');
          matrix = Object.freeze(Array.from(elements));
        }
      } else if (!standard) fail('UNSUPPORTED', `Native output color space is not registered: ${space}`);
      const sourceOutput = !standard || !enabled || matrix !== null;
      return Object.freeze({space, transfer, matrix, sourceOutput,
        key: JSON.stringify([space, transfer, matrix, sourceOutput])});
    }
    // Source RenderPipeline writes its output transform in the graph and sets
    // working-linear output to suppress a second transfer. Each queued frame
    // captures the attachment view separately from the live renderer property.
    _canvasSrgbView(space = this.outputColorSpace, hdr = this._wantsHdr(this._toneMapping()),
      outputProfile = this._captureOutputProfile(space)) {
      // Registered source output writes its encoded values to the plain canvas
      // view. r186 leaves GPUCanvasContext's display color space at default sRGB;
      // selecting another shader output space does not reconfigure its gamut.
      return this._shaderEncodedOutput ? hdr : space === SRGBColorSpace && !outputProfile.sourceOutput;
    }
    /** Whether the canvas composites with the page by alpha. WebGLRenderer (r186)
     * always creates an alpha:true context; its `alpha` option only sets the
     * default clear alpha, so shader-written alpha and setClearAlpha show. */
    _compositedAlpha() { return this.alpha === true || this._contextAlpha === true; }
    _toneMapping(toneMapping = this.toneMapping) {
      if (this._shaderEncodedOutput && toneMapping === THREE.CustomToneMapping) {
        // A linear fallback framebuffer cannot become shader-encoded history
        // without losing or reinterpreting its retained color/depth contents.
        if (this._hdrFallback === true) fail('UNSUPPORTED', 'Custom tone mapping cannot replace whole-image HDR fallback');
        return 'custom';
      }
      const name = TONE_MAPPINGS.get(toneMapping);
      if (name !== undefined) return name;
      // Keep registration/source graph construction in the retained library.
      // A registered operator may prepare before init() without asking the
      // fixed canvas shader to implement an unknown curve. Only the explicit
      // output-target and RenderPipeline paths execute that source graph.
      const library = this.library, mappings = library && Object.getOwnPropertyDescriptor(library, 'toneMappingNodes');
      if (!this._shaderEncodedOutput && Object.getPrototypeOf(library) === THREE.StandardNodeLibrary.prototype &&
          library.getToneMappingFunction === THREE.StandardNodeLibrary.prototype.getToneMappingFunction &&
          mappings && Object.hasOwn(mappings, 'value') && mappings.value instanceof Map &&
          typeof Map.prototype.get.call(mappings.value, toneMapping) === 'function') return 'source';
      fail('UNSUPPORTED', `Tone mapping ${toneMapping} has no admitted output pass`);
    }
    _bufferSize() {
      return [Math.floor(this._width * this._pixelRatio), Math.floor(this._height * this._pixelRatio)];
    }
    _resize() {
      const [w, h] = this._bufferSize();
      this._canvases.resize(this.getCanvasTarget(), w, h);
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
    getClearDepth() { return this.reversedDepthBuffer === true ? 1 - this._clearDepth : this._clearDepth; }
    setClearDepth(depth) { this._clearDepth = depth; }
    getClearStencil() { return this._clearStencil; }
    setClearStencil(stencil) { this._clearStencil = stencil; }
    getRenderTarget() { return this._renderTarget; }
    getCanvasTarget() { return this._canvasTarget; }
    setCanvasTarget(target) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      this._canvases.select(target);
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
    _capturePMREMTexture(texture, pendingSources = null) {
      if (texture?.onUpdate !== null) fail('UNSUPPORTED', 'PMREM input upload callbacks require their explicit source texture owner');
      if (hasThreeWebGLTextureResidency(this._device, texture))
        fail('UNSUPPORTED', 'PMREM input is already resident in a separate scene texture owner');
      const owner = this._getSampledTextureInputs();
      const lease = owner.captureDetachedBinding(texture, {webglVersioning: true});
      if (pendingSources) pendingSources.add(texture.source);
      else {
        let sources = PMREM_SOURCES.get(this);
        if (!sources) PMREM_SOURCES.set(this, sources = new WeakSet());
        sources.add(texture.source);
      }
      texture = owner.capturedTexture(lease);
      let bindings = PMREM_INPUTS.get(this);
      if (!bindings) PMREM_INPUTS.set(this, bindings = new Map());
      let binding = bindings.get(texture);
      if (!binding) bindings.set(texture, binding = {owner, leases: new Set()});
      binding.leases.add(lease);
      return {texture, owner, lease, release() {
        binding.leases.delete(lease); if (!binding.leases.size) bindings.delete(texture);
        owner.release(lease);
      }};
    }
    _framebufferBindings(retained = new Map()) {
      if (!(retained instanceof Map)) fail('OPTIONS', 'Expected a Map of borrowed texture bindings');
      const renderer = this;
      return new class extends Map {
        has(texture) { return texture?.isFramebufferTexture === true || PMREM_INPUTS.get(renderer)?.has(texture) ||
          PMREM_SOURCES.get(renderer)?.has(texture?.source) || retained.has(texture); }
        get(texture) {
          if (retained.has(texture)) return retained.get(texture);
          const input = PMREM_INPUTS.get(renderer)?.get(texture);
          if (input) return input.owner.capturedBinding(input.leases.values().next().value);
          if (PMREM_SOURCES.get(renderer)?.has(texture?.source)) {
            if (texture.onUpdate !== null) fail('UNSUPPORTED', 'Scene upload callbacks require their explicit source texture owner');
            return renderer._getSampledTextureInputs().webglBinding(texture);
          }
          return texture?.isFramebufferTexture === true
            ? renderer._getSampledTextureInputs().framebufferBinding(texture, {initialize: renderer._webglSubmissionDepth > 0,
              snapshot: renderer._programFramebufferFrame?.bindings.get(texture) ?? null})
            : retained.get(texture);
        }
      }();
    }
    _captureFramebufferBindings() {
      if (!this._shaderEncodedOutput) return null;
      const owner = this._sampledTextureInputs, bindings = owner?.captureFramebufferBindings();
      // Snapshot only the private generations already admitted by this owner.
      // Ordinary materials/custom properties must not be traversed or observed.
      return bindings?.size ? {owner, bindings} : null;
    }
    copyFramebufferToTexture(texture, rectangle = null, level = 0) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (!this._initialized) fail('SOURCE', 'Initialize the renderer before copying its framebuffer');
      const webgl = this._shaderEncodedOutput === true;
      const drawContext = webgl && this._webglSubmissionDepth > 0 ? this._programDrawContext : null;
      if (webgl && this._webglSubmissionDepth > 0 && !drawContext)
        fail('UNSUPPORTED', 'WebGL framebuffer copies inside rendering callbacks require draw-boundary interleaving');
      const depth = texture instanceof THREE.DepthTexture && texture.isDepthTexture === true;
      if (!(texture instanceof THREE.FramebufferTexture) && !depth)
        fail('SOURCE', 'Expected a source FramebufferTexture or detached DepthTexture');
      if (webgl && depth) fail('UNSUPPORTED', 'WebGL framebuffer snapshots currently require a color FramebufferTexture');
      if (depth) {
        const shape = this._renderTarget ? this._targets.capture(this._renderTarget).shape : null;
        if (this._activeMipmapLevel !== 0 || shape && (shape.cube || shape.array || shape.volume))
          fail('COPY', 'Depth framebuffer snapshots require base-level ordinary 2D attachments');
      }
      let region;
      if (webgl) {
        if (!Number.isSafeInteger(level) || level < 0 || level > 31) fail('COPY', 'Expected a non-negative destination mip level');
        region = [rectangle === null ? 0 : Math.trunc(rectangle.x), rectangle === null ? 0 : Math.trunc(rectangle.y),
          Math.floor(texture.image.width / 2 ** level), Math.floor(texture.image.height / 2 ** level)];
        if (!region.every(Number.isSafeInteger) || region[0] < 0 || region[1] < 0 || region[2] < 1 || region[3] < 1)
          fail('COPY', 'Framebuffer copy coordinates and mip extent must fit the source');
      }
      else if (rectangle === null) region = [0, 0, texture.image.width, texture.image.height];
      else if (rectangle.isVector2) region = [rectangle.x, rectangle.y, texture.image.width, texture.image.height].map(Math.floor);
      else if (rectangle.isVector4) region = [rectangle.x, rectangle.y, rectangle.z, rectangle.w].map(Math.floor);
      else { (THREE.error ?? console.error)('Renderer.copyFramebufferToTexture: Invalid rectangle.'); return; }
      // Source draws preceding this call have already issued their GPU effects.
      // Flush before initialization: onUpdate can throw or recursively copy.
      drawContext?.flush();
      const textures = this._getSampledTextureInputs();
      textures.prepareFramebuffer(texture, this._renderTarget?.texture.type ?? THREE.UnsignedByteType);
      drawContext?.check();
      // WebGL initializes the destination before choosing its read framebuffer:
      // onUpdate can throw or change the current target. Keep the earlier extent.
      if (webgl && !this._renderTarget && this._wantsHdr(this._toneMapping()))
        fail('UNSUPPORTED', 'WebGL framebuffer copies require encoded canvas color, unavailable in the HDR fallback');
      const format = depth ? undefined : this._framebufferFormat();
      const source = !this._nodeFramebufferCopyRecorder && this._renderTarget
        ? this._targets.capture(this._renderTarget) : null;
      const sourceLevel = webgl ? this._activeMipmapLevel : 0, sourceFace = webgl ? this._activeCubeFace : 0;
      const [width, height] = source ? [Math.max(1, source.shape.width >> sourceLevel), Math.max(1, source.shape.height >> sourceLevel)] : this._bufferSize();
      const canvasTarget = this.getCanvasTarget(), canvas = canvasTarget.domElement;
      const canvasFrameToken = webgl ? this._webglFrameToken ?? null : null;
      if (!this._nodeFramebufferCopyRecorder &&
          (region[0] + region[2] > width || region[1] + region[3] > height))
        fail('COPY', 'Framebuffer copy rectangle exceeds the selected source');
      const lease = textures.captureFramebuffer(texture, {format,
        rectangle: {x: region[0], y: region[1], z: region[2], w: region[3]},
        ...(webgl ? {level, sourceLevel, sourceFace} : {})});
      if (drawContext) {
        try { drawContext.copy(textures, lease); }
        finally { textures.release(lease); }
      } else if (this._nodeFramebufferCopyRecorder) {
        try { this._nodeFramebufferCopyRecorder(Object.freeze({lease})); }
        catch (error) { textures.release(lease); throw error; }
      } else {
        let releaseSource = null, releaseCanvas = null, released = false;
        const release = () => {
          if (released) return; released = true;
          textures.release(lease); releaseSource?.(); releaseCanvas?.();
        };
        try {
          releaseSource = source ? this._targets.retainGeneration(source) : null;
          releaseCanvas = source ? null : this._canvases.retain(canvasTarget, canvas);
          const hdr = !source && this._wantsHdr(this._toneMapping());
          const outputProfile = source ? null : this._captureOutputProfile();
          const srgbView = source ? false : this._canvasSrgbView(this.outputColorSpace, hdr, outputProfile);
          const copy = () => {
            if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
            textures.validateDispatch([lease]);
            const encode = native => {
              textures.validateDispatch([lease]);
              textures.withDispatch([lease], () => {
                const encoder = this._device.createCommandEncoder({label: 'f3d-public-framebuffer-copy'});
                const copied = textures.encodeFramebufferCopy(lease, encoder, depth ? null : native, depth ? native : null);
                if (copied !== false) this._device.queue.submit([encoder.finish()]);
              });
            };
            if (source) this._targets.withFramebufferTexture(source, encode, {depth});
            else this._session.withFramebufferTexture(encode, {width, height, depth, frameToken: canvasFrameToken});
          };
          const request = {framebufferCopy: true, releaseFramebuffer: release, issue: () => {
            // Invalid destinations must not acquire or replace presentation.
            textures.validateDispatch([lease]);
            if (!source) {
              this._canvases.activate(canvasTarget, canvas);
              // HDR snapshots read the persistent working-linear target. A
              // changed presentation transform must not discard its history.
              if (!this._session || hdr !== this._hdr || !hdr && srgbView !== this._srgbView ||
                  this._shaderEncodedOutput && outputProfile?.canvasKey !== this._canvasOutputKey) {
                this._session?.dispose();
                return this._createSession(hdr, srgbView, canvasTarget, [width, height], canvas, outputProfile).then(copy);
              }
              this._session.resize(width, height);
            }
            return copy();
          }};
          // Public void calls share the render FIFO. An already prepared copy
          // submits now; pending renders keep their original producer boundary.
          if (!this._drain && !this._preparing && !this._pending.length &&
              (source || this._canvases.matches(canvasTarget, width, height, canvas) && hdr === this._hdr &&
                (hdr || srgbView === this._srgbView && (outputProfile?.key ?? null) === this._outputProfileKey))) {
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
        // The target owner validates dimensions, formats and attachment count.
        // Conversion is a render-time choice; working-linear MRT output uses
        // the original fragment outputs directly, with no private framebuffer.
        this._targets.capture(target);
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
      if (renderTarget !== null) inspectThreeRenderTargetMip(this._targets.capture(renderTarget).shape,
        activeMipmapLevel, activeCubeFace, {webgl: this._shaderEncodedOutput, generateMipmaps: renderTarget.texture.generateMipmaps});
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
    _frame(scene, clearOnly = null, target = this._renderTarget, mipLevel = this._activeMipmapLevel,
      outputProfile = target ? null : this._captureOutputProfile(), rasterState = null) {
      const targetShape = target ? this._targets.capture(target).shape : null;
      const mipShape = targetShape ? inspectThreeRenderTargetMip(targetShape, mipLevel, 0,
        {webgl: this._shaderEncodedOutput, generateMipmaps: target.texture.generateMipmaps}) : null;
      if (this.xr.enabled && this.xr.isPresenting) fail('UNSUPPORTED', 'XR presentation is not admitted');
      const tone = this._toneMapping();
      if (!target && tone === 'source')
        fail('UNSUPPORTED', 'Registered native tone mapping requires an output render target or a RenderPipeline output graph');
      // Switching between direct and tone-mapped output rebuilds the canvas
      // session (same device) at the next preparation boundary.
      if (!target) this._rebuild = this._wantsHdr(tone) !== this._hdr ||
        !this._hdr && this._canvasSrgbView(this.outputColorSpace, this._wantsHdr(tone), outputProfile) !== this._srgbView ||
        (outputProfile?.key ?? null) !== this._outputProfileKey;
      const hdr = !target && this._wantsHdr(tone);
      const clear = this.autoClear === true;
      const color = this._clearColor, a = this._clearAlpha;
      // Source Background semantics: the renderer clear color is premultiplied
      // for alpha canvases; an opaque canvas ignores alpha.
      const canvasAlpha = this._compositedAlpha();
      const premultiply = canvasAlpha && (!target || !this._bottomLeftOrigin) && !this._straightAlphaCanvas;
      const frame = {
        loadOp: (clearOnly ? clearOnly.color : clear && this.autoClearColor) ? 'clear' : 'load',
        depthLoadOp: (clearOnly ? clearOnly.depth : clear && this.autoClearDepth) ? 'clear' : 'load',
        clearColor: premultiply ? [color.r * a, color.g * a, color.b * a, a] : [color.r, color.g, color.b, target || canvasAlpha ? a : 1],
      };
      // Source WebGLBackground uses Color.getRGB in the selected output space
      // before premultiplication. GL clamps its normalized drawing-buffer clear.
      if (!target && this._shaderEncodedOutput && !hdr) {
        const converted = color.getRGB({}, outputProfile?.space ?? this.outputColorSpace);
        const encoded = [converted.r, converted.g, converted.b];
        frame.clearColor = (premultiply ? [...encoded.map(c => c * a), a] : [...encoded, canvasAlpha ? a : 1])
          .map(c => Math.max(0, Math.min(1, c)));
      }
      Object.assign(frame, {
        clearDepth: this.getClearDepth(),
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
      const pr = rasterState?.pixelRatio ?? (target ? 1 : this._pixelRatio);
      const [bufferWidth, bufferHeight] = rasterState?.bufferSize ?? (target ? [mipShape.width, mipShape.height] : this._bufferSize());
      const v = rasterState?.viewport ?? (target ? target.viewport : this._viewport), px = this._pixelRound ?? Math.floor;
      // r186 WebGPU scales width/height at a selected mip. WebGL keeps the
      // authored viewport/scissor in physical pixels, including their extents.
      // Neither route mutates the source Vector4s while capturing a request.
      const divisor = target && !this._shaderEncodedOutput ? 2 ** mipLevel : 1;
      const vx = px(v.x * pr), vw = Math.floor(px(v.z * pr) / divisor), vh = Math.floor(px(v.w * pr) / divisor);
      // Program gl_FragCoord (framebuffer) and point-sprite sizes (viewport).
      if (this._shaderEncodedOutput && !hdr) frame.targetSize = [bufferWidth, bufferHeight, vw, vh];
      // r186 _currentViewport size: the transmission target follows it.
      if (this._shaderEncodedOutput) frame.viewportSize = [vw, vh];
      // WebGL viewports/scissors use a bottom-left origin; WebGPU's is top-left.
      const vy = this._bottomLeftOrigin ? bufferHeight - px(v.y * pr) - vh : px(v.y * pr);
      const webglMip = target && this._shaderEncodedOutput && mipLevel > 0;
      if (webglMip && (![vx, vy, vw, vh].every(Number.isFinite) || vx < 0 || vy < 0 || vw < 0 || vh < 0 ||
          vx + vw > bufferWidth || vy + vh > bufferHeight))
        fail('UNSUPPORTED', 'WebGL mip rendering requires a viewport inside the selected mip extent');
      const [minDepth, maxDepth] = rasterState?.depth ?? (target ? [0, 1] : this._viewportDepth);
      if (vx !== 0 || vy !== 0 || vw !== bufferWidth || vh !== bufferHeight || minDepth !== 0 || maxDepth !== 1)
        frame.viewport = [vx, vy, vw, vh, minDepth, maxDepth];
      if (rasterState?.scissorTest ?? (target ? target.scissorTest : this._scissorTest)) {
        // Source clamping: non-negative and inside the drawing buffer.
        const sc = rasterState?.scissor ?? (target ? target.scissor : this._scissor), rawH = Math.floor(px(sc.w * pr) / divisor);
        const rawY = this._bottomLeftOrigin ? bufferHeight - px(sc.y * pr) - rawH : px(sc.y * pr);
        const sx = Math.max(0, px(sc.x * pr)), sy = Math.max(0, rawY);
        // WebGPU source: clamp each component to >= 0, then fit inside the buffer.
        // WebGL: the GL scissor box is intersected with the buffer, so a negative
        // origin also shortens the box.
        const clip = this._bottomLeftOrigin;
        const sw = Math.max(0, Math.min(Math.floor(px(sc.z * pr) / divisor) + (clip ? Math.min(0, px(sc.x * pr)) : 0), bufferWidth - sx)),
          sh = Math.max(0, Math.min(rawH + (clip ? Math.min(0, rawY) : 0), bufferHeight - sy));
        if (sx !== 0 || sy !== 0 || sw !== bufferWidth || sh !== bufferHeight)
          frame.scissor = [webglMip ? Math.min(sx, bufferWidth) : sx, webglMip ? Math.min(sy, bufferHeight) : sy, sw, sh];
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
    setOpaqueSort(method) { this._opaqueSort = method; }
    setTransparentSort(method) { this._transparentSort = method; }
    isOccluded(object) { return this._occlusionContext && this._occlusionContext.occluded?.has(object); }
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
      if (this._isDeviceLost === true && this._initialized) return;
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (!(scene instanceof THREE.Object3D) || !(camera instanceof THREE.Camera)) fail('SOURCE', 'Expected a source scene and camera');
      this._beforeRender();
      this.info.calls++;
      this.info.render.calls++;
      this.info.render.frameCalls++;
      // Source timestamp UID shape `<call id>:f<frame>` (Inspector groups by call id).
      const uid = `render:${++this._renderUid}:f${this.info.frame}`, previousTimestamp = this._timestampScope,
        previousOperations = this._operationScope;
      const timestamp = this._timestamps.begin('render', uid);
      this._inspector.beginRender(uid, scene, camera, this._renderTarget);
      this._timestampScope = timestamp;
      this._operationScope = this._backend._capture({camera, renderTarget: this._renderTarget});
      try { return this._renderFrame(scene, camera); }
      finally { this._timestampScope = previousTimestamp; this._operationScope = previousOperations; this._inspector.finishRender(uid); }
    }
    _makeRequest(scene, camera, clearOnly = null, captureCameras = true, sourceTarget = this._renderTarget, sceneLifecycle = null) {
      const target = this._renderTarget, mipLevel = this._activeMipmapLevel, face = this._activeCubeFace, mrt = this._mrt;
      const renderId = this.info.calls;
      if (sceneLifecycle && this._initialized) this._beginSceneLifecycle(sceneLifecycle, {target, sourceTarget, renderId, sourceFace: face, sourceMip: mipLevel});
      try {
      const sceneReference = sceneLifecycle?.scene ?? scene;
      const overrideMaterial = sceneReference.isScene === true ? sceneReference.overrideMaterial : null;
      const passCameraRequired = this._shaderEncodedOutput && !clearOnly && captureCameras && requiresThreePassCamera(this);
      // Source-verified passes retain temporary jitter and AO exclusions after
      // public state is restored. Check before capturing a destination or retaining GPU
      // generations. Hook-bearing scenes keep their existing public identities;
      // they need a broader synchronous command capture before admission here.
      let requiredMaterialFrame = null;
      if (passCameraRequired) {
        if (this.shadowMap.enabled || this.sortObjects && (this._opaqueSort || this._transparentSort))
          fail('UNSUPPORTED', 'Pinned pass capture requires a callback-free scene without shadows or custom sorting');
        requiredMaterialFrame = captureThreeMaterialState(THREE, scene, {camera, overrideMaterial, selectionOnly: true, allowInstances: true,
          exclusions: getThreePassExcludedObjects(this),
          maxNodes: sceneLimits.maxNodes ?? 16384, maxMaterials: maxBindings,
          maxValues: MAX_MATERIAL_CAPTURE_VALUES - this._materialCaptureValues});
        if (!requiredMaterialFrame?.cameraSafe())
          fail('UNSUPPORTED', 'Pinned pass capture requires ordinary callback-free scene and camera state');
      }
      // Each accepted call keeps its source sort selection while pipelines are
      // prepared. Changing a sorter for the next call cannot reorder this one.
      const sortState = Object.freeze({sortObjects: this.sortObjects === true,
        opaqueSort: this._opaqueSort, transparentSort: this._transparentSort});
      const canvasTarget = this.getCanvasTarget(), canvas = canvasTarget.domElement, canvasSize = this._bufferSize();
      if (!this._shaderEncodedOutput) assertThreeNativeTargetWrite(this, target);
      const destination = target ? this._targets.capture(target, {initialize: !this._shaderEncodedOutput}) : this;
      let backgroundState = !this._shaderEncodedOutput || passCameraRequired ? captureThreeBackground(THREE, sceneLifecycle?.backgroundScene ?? scene) : undefined;
      const outputProfile = target ? null : this._captureOutputProfile();
      const frame = this._frame(scene, clearOnly, target, mipLevel, outputProfile, sceneLifecycle?.rasterState), key = sceneKey(this, scene, backgroundState);
      if (!clearOnly && this._timestampScope) frame.timestamp = this._timestampScope;
      if (!clearOnly && this._operationScope) frame.operations = this._operationScope;
      // RenderPass/normal/depth passes restore this property immediately after
      // render(). Capture its selector now; source objects stay application-owned.
      const c = this._clearColor, a = this._clearAlpha;
      // Small callback-free shader passes (FullScreenQuad) are private data views;
      // general scenes and hooked roots retain their established source path.
      const pmremScope = this._shaderEncodedOutput && PMREM_CAPTURES.get(this), pmrem = pmremScope?.generator;
      const pmremDraw = pmrem && pmrem._lodMeshes.includes(scene) &&
        [pmrem._cubemapMaterial, pmrem._equirectMaterial, pmrem._ggxMaterial, pmrem._blurMaterial].includes(scene.material);
      const pmremScene = pmremScope?.scene && (scene === pmremScope.scene || scene === pmrem._backgroundBox);
      const snapshot = this._shaderEncodedOutput ? pmremDraw ? this._passSnapshots.capturePMREM(scene, camera)
        : pmremScene ? scene === pmremScope.scene ? pmremScope.snapshot.retain(camera)
          : this._passSnapshots.capturePMREMScene(scene, camera, this._device)
        : this._passSnapshots.capture(scene, camera) : null;
      if (pmremDraw && !snapshot) fail('UNSUPPORTED', 'PMREM filtering requires its ordinary source plane and shader values');
      let cameraSource = snapshot ? camera : null;
      if (this._shaderEncodedOutput && destination.shape?.webglCube && !snapshot && captureCameras) {
        // CubeCamera submits six public renders before preparation can finish.
        // Keep those source face poses even if the next capture moves it while
        // the first cube is queued. Scene/material ownership is unchanged.
        if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
        cameraSource = camera;
        camera = captureThreeCamera(THREE, camera);
      }
      let matricesUpdated = sceneLifecycle?.begun === true;
      if (camera?.isArrayCamera === true && captureCameras) {
        if (!Array.isArray(camera.cameras) || camera.cameras.length > 256 || camera.isMultiViewCamera)
          fail('UNSUPPORTED', 'Expected an ordinary ArrayCamera with at most 256 views');
        // WebGL child viewports already use physical bottom-left pixels. Native
        // WebGPU canvas views instead use logical units and a top-left origin.
        frame.arrayCamera = {pixelRatio: this._shaderEncodedOutput || target ? 1 : this._pixelRatio,
          clipSpace: this._shaderEncodedOutput ? 'webgl' : 'webgpu',
          targetSize: target ? [Math.max(1, Math.floor(target.width / 2 ** mipLevel)), Math.max(1, Math.floor(target.height / 2 ** mipLevel))] : this._bufferSize()};
      }
      const native = !this._shaderEncodedOutput && !clearOnly;
      // Before initialization, the source renderer prepares at its asynchronous
      // boundary. Texture feature/layout admission needs the actual device.
      const nodeCaptureDeferred = native && !this._initialized;
      const nodeFrame = native && !nodeCaptureDeferred
        ? this._captureNodeFrame({scene, camera, destination, overrideMaterial, captureCameras, mrt, renderId, sourceTarget, mipLevel, face, sortState, sceneLifecycle, canvasTarget}) : null;
      if (nodeFrame) matricesUpdated = true;
      if (nodeFrame?.sceneHierarchy) backgroundState = Object.freeze({...backgroundState,
        nativeSceneProfile: Object.freeze({key, shadows: this.shadowMap.enabled})});
      // Native callbacks retain the original camera in _captureNodeFrame.
      // Fixed draws can wait on pipeline/texture preparation; capture their
      // camera only after the source matrix and node updates finish. Array
      // callbacks and camera-array nodes must receive the original parent.
      if (native && !nodeCaptureDeferred && captureCameras) {
        if (!matricesUpdated) this._updateNativeMatrices(scene, camera);
        camera = camera.isArrayCamera ? captureThreeArrayCamera(THREE, camera) : captureThreeCamera(THREE, camera);
        matricesUpdated = true;
      }
      let materialFrame = null;
      if (native && !nodeCaptureDeferred) {
        try { materialFrame = this._captureMaterialFrame({scene, nodeFrame, overrideMaterial}); }
        catch (error) { this._releaseRequest({nodeFrame}); throw error; }
      } else if (this._shaderEncodedOutput && !clearOnly && !snapshot && !this.shadowMap.enabled) {
        // UI material changes may arrive while an accepted draw prepares its
        // pipeline. Keep that draw's selector, without freezing live uniforms or
        // changing hooked projection/shadow work's established source ordering.
        materialFrame = requiredMaterialFrame ?? this._captureMaterialFrame({scene, camera, overrideMaterial, selectionOnly: true});
        if (requiredMaterialFrame) this._materialCaptureValues += materialFrame.values;
      }
      // Captures already own source resource leases. Publish each additional
      // lease immediately so a later acquisition failure can release all of it.
      const request = {snapshot, nodeFrame, materialFrame, pmremInput: pmremDraw ? pmremScope.input : null};
      try {
        if (passCameraRequired && overrideMaterial?.isShaderMaterial) {
          request.overrideSnapshot = this._passSnapshots.captureMaterial(overrideMaterial);
          if (!request.overrideSnapshot) fail('UNSUPPORTED', 'Pinned pass shader overrides require ordinary callback-free material values');
        }
        let webglSourceCamera = null;
        if (this._shaderEncodedOutput && captureCameras && !snapshot && materialFrame?.cameraSafe?.() &&
            !(this.sortObjects && (this._opaqueSort || this._transparentSort))) {
          // Keep one camera value per accepted draw, including SSAA's distinct
          // projections. Callback-free rigid draws also retain their accepted
          // geometry selector and world matrix in private views below; live or
          // deformed paths keep their existing source update order.
          if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
          if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
          webglSourceCamera = camera;
          cameraSource ??= camera;
          camera = captureThreeCamera(THREE, camera);
          materialFrame.instanceFrame = this._passInstances.capture(scene, camera, materialFrame, {
            maxValues: MAX_MATERIAL_CAPTURE_VALUES - this._materialCaptureValues, sortObjects: this.sortObjects, required: passCameraRequired});
          if (materialFrame.instanceFrame) {
            materialFrame.values += materialFrame.instanceFrame.values;
            this._materialCaptureValues += materialFrame.instanceFrame.values;
          }
        }
        Object.assign(request, {scene: snapshot?.root ?? scene, camera: snapshot?.camera ?? camera, matricesUpdated, clearOnly, destination, mipLevel, face, frame,
          overrideMaterial: request.overrideSnapshot?.material ?? overrideMaterial, mrt,
          webglSourceCamera, passCameraRequired,
          nodeCaptureDeferred, captureCameras, backgroundState, renderId, sourceTarget, canvasTarget, canvas, canvasSize, sortState, sceneLifecycle,
          canvasFrameToken: this._shaderEncodedOutput ? this._webglFrameToken ?? null : null,
          generateCubeMipmaps: !clearOnly && (this._shaderEncodedOutput
            ? target?.isWebGLCubeRenderTarget === true && mipLevel === 0
            : target?.isCubeRenderTarget === true) && target.texture.generateMipmaps,
          generateColorMipmaps: !clearOnly && target && (target.textures.length > 1 || !this._shaderEncodedOutput &&
            (destination.historyGroup || !destination.shape.cube && !destination.shape.array && !destination.shape.volume))
            ? target.textures.flatMap((texture, index) => texture.generateMipmaps ? [index] : []) : null,
          linearClearColor: this.alpha ? [c.r * a, c.g * a, c.b * a, a] : [c.r, c.g, c.b, 1],
          hdr: !target && this._wantsHdr(this._toneMapping()),
          srgbView: target ? false : this._canvasSrgbView(outputProfile?.space ?? this.outputColorSpace,
            this._wantsHdr(this._toneMapping()), outputProfile), outputProfile, key,
          programFrame: {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
            passTextureEntries: snapshot?.targetEntries ?? request.overrideSnapshot?.targetEntries,
            outputColorSpace: this.outputColorSpace, pixelRatio: this._pixelRatio, height: this._height,
            logarithmicDepthBuffer: this.logarithmicDepthBuffer === true,
            reversedDepthBuffer: this.reversedDepthBuffer === true,
            ...(cameraSource ? {cameraSource, cameraSnapshot: snapshot?.camera ?? camera} : {}),
            ...(this._shaderEncodedOutput ? {
              passClear: clearOnly ? {...clearOnly} : {color: this.autoClear && this.autoClearColor,
                depth: this.autoClear && this.autoClearDepth, stencil: this.autoClear && this.autoClearStencil},
              passBackgroundClear: clearOnly ? null : {color: this.autoClearColor, depth: this.autoClearDepth, stencil: this.autoClearStencil},
            } : {})}});
        if (pmremDraw || pmremScene) {
          let roots = PMREM_EXECUTIONS.get(pmrem);
          if (!roots) PMREM_EXECUTIONS.set(pmrem, roots = new Map());
          let destinations = roots.get(scene);
          if (!destinations) roots.set(scene, destinations = new Map());
          destinations.set(destination, request.scene);
        }
        request.programTextures = this._captureFramebufferBindings();
        request.releaseDestination = destination === this ? null : this._targets.retainGeneration(destination);
        request.releaseCanvas = this._canvases.retain(canvasTarget, canvas);
        // Compilation does not execute GL state. Publish a render's token only
        // after all captures succeed, preserving explicit pass calls around
        // deferred draws without imposing state work on ordinary renderers.
        request.programFrame.passState = this._shaderEncodedOutput && captureCameras ? captureThreePassState(this) : null;
        if (this._shaderEncodedOutput && !target && !clearOnly && this.toneMapping === THREE.CustomToneMapping &&
            (snapshot || materialFrame?.cameraSafe()) && (!sortState.sortObjects || !sortState.opaqueSort && !sortState.transparentSort)) {
          // Keep an ordinary custom curve with its accepted deferred draw.
          // Existing callback-free capture profiles avoid freezing a hook's
          // reads/writes of global shader state. Inspect descriptors so an
          // accessor-backed chunk keeps its original live evaluation boundary.
          const chunk = Object.getOwnPropertyDescriptor(THREE.ShaderChunk, 'tonemapping_pars_fragment');
          if (typeof chunk?.value === 'string') request.programFrame.toneMappingChunk = chunk.value;
        }
        return request;
      } catch (error) { this._releaseRequest(request); throw error; }
      } finally { this._restoreSceneLifecycle(sceneLifecycle); }
    }
    _updateNativeMatrices(scene, camera) {
      if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
      updateThreeCameraProjection(THREE, this, camera);
      if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
    }
    _sceneLifecycle(root, camera, compilation = false, targetScene = null) {
      const scene = root.isScene === true ? root : targetScene?.isScene === true ? targetScene : null;
      if (this._shaderEncodedOutput || !scene || !(camera instanceof THREE.Camera)) return null;
      return {root, scene, camera, compilation, targetScene, backgroundScene: targetScene ?? scene, begun: false, finished: false,
        entrySelection: {target: this._renderTarget, face: this._activeCubeFace, mip: this._activeMipmapLevel}};
    }
    _beginSceneLifecycle(lifecycle, {target, sourceTarget, renderId, sourceFace = this._activeCubeFace, sourceMip = this._activeMipmapLevel}) {
      if (lifecycle.begun) return;
      const {scene, camera, compilation} = lifecycle;
      const frame = this._nodes.nodeFrame;
      lifecycle.previousRenderId = frame.renderId;
      lifecycle.previousDispatcher = this._currentRenderObjectFunction;
      lifecycle.renderObjectFunction = compilation ? null : this.getRenderObjectFunction();
      this._currentRenderObjectFunction = lifecycle.renderObjectFunction || this.renderObject;
      lifecycle.restoreRenderId = true;
      lifecycle.begun = true;
      const selection = compilation ? lifecycle.entrySelection : {target: sourceTarget, face: sourceFace, mip: sourceMip};
      this._renderTarget = selection.target; this._activeCubeFace = selection.face; this._activeMipmapLevel = selection.mip;
      // Native render() selects its private working target before scene hooks.
      // Compilation passes that same target as an argument, without changing
      // the public selector. The owned canvas still submits through its session.
      let callbackTarget = target;
      if (target === null && this.needsFrameBufferTarget) {
        this.getCanvasTarget(); this._frameBufferTargets ??= new Map();
        callbackTarget = THREE.Renderer.prototype._getFrameBufferTarget.call(this);
        if (!compilation) {
          lifecycle.outputSelection = {target: sourceTarget, face: this._activeCubeFace, mip: this._activeMipmapLevel};
          this._renderTarget = callbackTarget; this._activeCubeFace = 0; this._activeMipmapLevel = 0;
        }
      }
      lifecycle.target = callbackTarget;
      frame.renderId = compilation ? frame.renderId + 1 : renderId;
      try {
        this._updateNativeMatrices(lifecycle.root, camera);
        // Source viewport/scissor and the global clipping view precede the
        // scene callback. Its material, background and list edits come later.
        lifecycle.rasterState = {
          viewport: (callbackTarget?.viewport ?? this._viewport).clone(),
          scissor: (callbackTarget?.scissor ?? this._scissor).clone(),
          scissorTest: callbackTarget?.scissorTest ?? this._scissorTest,
          pixelRatio: callbackTarget ? 1 : this._pixelRatio,
          depth: callbackTarget ? [0, 1] : this._viewportDepth.slice(),
          // Default render-context extents precede the callback, but target
          // textures are updated afterward and may have been resized by it.
          bufferSize: callbackTarget ? null : this._bufferSize(),
        };
        lifecycle.clippingView = camera.matrixWorldInverse.clone();
        lifecycle.hooked = lifecycle.targetScene !== null || scene.onBeforeRender !== THREE.Object3D.prototype.onBeforeRender ||
          scene.onAfterRender !== THREE.Object3D.prototype.onAfterRender;
        scene.onBeforeRender(this, lifecycle.root, camera, callbackTarget);
        lifecycle.hooked ||= scene.onAfterRender !== THREE.Object3D.prototype.onAfterRender;
        lifecycle.extraLights = [];
        if (lifecycle.targetScene && lifecycle.targetScene !== lifecycle.root)
          lifecycle.targetScene.traverseVisible(object => {
            if (object.isLight && object.layers.test(camera.layers)) lifecycle.extraLights.push(object);
          });
        lifecycle.publicTarget = this._renderTarget;
        lifecycle.publicFace = this._activeCubeFace; lifecycle.publicMip = this._activeMipmapLevel;
      } catch (error) { this._restoreSceneLifecycle(lifecycle, false); throw error; }
    }
    _restoreSceneLifecycle(lifecycle, restoreOutput = true) {
      if (!lifecycle?.restoreRenderId) return;
      lifecycle.restoreRenderId = false;
      this._nodes.nodeFrame.renderId = lifecycle.previousRenderId;
      this._currentRenderObjectFunction = lifecycle.previousDispatcher;
      if (restoreOutput && lifecycle.outputSelection) {
        const {target, face, mip} = lifecycle.outputSelection;
        this._renderTarget = target; this._activeCubeFace = face; this._activeMipmapLevel = mip;
      }
    }
    _finishSceneLifecycle(lifecycle) {
      if (!lifecycle?.begun || lifecycle.finished || lifecycle.compilation) return;
      lifecycle.finished = true;
      // Read the hook at this boundary: before/object hooks may replace it.
      // Accepted parent requests already precede any nested after-hook work.
      lifecycle.scene.onAfterRender(this, lifecycle.root, lifecycle.camera, lifecycle.target);
    }
    _renderFrame(scene, camera, clearOnly = null) {
      const lifecycle = clearOnly ? null : this._sceneLifecycle(scene, camera);
      const requests = this._makeOutputRequests(scene, camera, clearOnly, true, lifecycle);
      if (lifecycle) lifecycle.lastRequest = requests.at(-1);
      for (let i = 0; i < requests.length; i++) {
        try { this._dispatchRequest(requests[i]); }
        catch (error) {
          for (const pending of requests.slice(i + 1)) this._releaseRequest(pending);
          throw error;
        }
      }
      this._finishSceneLifecycle(lifecycle);
    }
    _outputQuad(target, layer = 0) {
      let cache = this._outputQuads.get(target);
      if (!cache) {
        cache = {quads: new Map(), records: new Set(), epoch: 0, sourceTransforms: false, transform: null, dispose: () => {
          if (cache.resizing) return;
          target.removeEventListener('dispose', cache.dispose);
          for (const record of cache.records) { record.retired = true; record.release(); }
          this._outputQuads.delete(target);
        }};
        target.addEventListener('dispose', cache.dispose);
        this._outputQuads.set(target, cache);
      }
      const transform = `${this.toneMapping},${this.currentColorSpace}`;
      cache.sourceTransforms ||= this._toneMapping() === 'source';
      if (cache.sourceTransforms && cache.transform !== transform) {
        // NodeManager's output key selects one current graph. Switching away
        // and back reconstructs a registered function, including its effects
        // and current library entry. Accepted older graphs keep their leases.
        cache.epoch++;
        for (const record of cache.records) { record.retired = true; record.release(); }
      }
      cache.transform = transform;
      const key = `${transform},${layer},${cache.epoch}`;
      let quad = cache.quads.get(key);
      if (!quad) {
        quad = new THREE.QuadMesh(new THREE.NodeMaterial());
        quad.name = 'Output Color Transform'; quad.material.name = 'outputColorTransform';
        // Retain the pinned NodeManager.getOutputNode construction, including
        // alpha un/premultiplication, exposure and the source tone operators.
        let output = THREE.TSL.texture(target.texture, THREE.TSL.screenUV);
        if (target.texture.isArrayTexture === true && target.texture.image.depth > 1)
          output = output.depth(THREE.TSL.int(layer));
        quad.material.fragmentNode = output.renderOutput(this.toneMapping, this.currentColorSpace);
        // Distinct graph identities keep earlier queued captures valid when a
        // later call changes the renderer's output transform before preparation.
        cache.quads.set(key, quad);
        const record = {quad, references: 0, retired: false, destinations: new Set(), release: () => {
          if (!record.retired || record.references !== 0 || !cache.records.delete(record)) return;
          cache.quads.delete(key); this._outputQuadLeases.delete(quad); quad.material.dispose();
          for (const destination of record.destinations) {
            destination._dispatcher?.retire(quad); destination._wanted.delete(quad);
          }
          record.destinations.clear();
        }};
        cache.records.add(record); this._outputQuadLeases.set(quad, record);
      }
      return quad;
    }
    _retainOutputQuad(quad) {
      const record = this._outputQuadLeases.get(quad);
      record.references++;
      let held = true;
      return () => { if (held) { held = false; record.references--; record.release(); } };
    }
    _makeOutputRequests(scene, camera, clearOnly = null, captureCameras = true, sceneLifecycle = null) {
      if (!this._outputRenderTarget || this._shaderEncodedOutput)
        return [this._makeRequest(scene, camera, clearOnly, captureCameras, this._renderTarget, sceneLifecycle)];
      const previousTarget = this._renderTarget, previousFace = this._activeCubeFace,
        previousMip = this._activeMipmapLevel;
      const outputTarget = previousTarget || this._outputRenderTarget;
      // clear() selects an explicit target first and only otherwise the private
      // framebuffer. With no conversion, a null selection still clears canvas.
      const convert = this.needsFrameBufferTarget && (!clearOnly || previousTarget === null);
      if (convert && this._targets.capture(outputTarget).shape.textures.length !== 1)
        fail('UNSUPPORTED', 'Multiple-color output targets require working-linear output without tone mapping');
      if (convert && captureCameras && (previousFace !== 0 || previousMip !== 0)) {
        const shape = this._targets.capture(outputTarget).shape;
        // The pinned private image has one mip and, for a cube output, one
        // layer. Reject selections that its source render context cannot use.
        if (((shape.cube || shape.array || shape.volume) && previousMip !== 0) ||
            (shape.cube && previousFace !== 0))
          fail('UNSUPPORTED', 'Layered output conversion requires the base mip; cube conversion requires face zero');
      }
      if (!convert && (previousTarget !== null || clearOnly)) {
        const request = this._makeRequest(scene, camera, clearOnly, captureCameras, this._renderTarget, sceneLifecycle);
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
          requests.push(this._makeRequest(scene, camera, clearOnly, captureCameras, previousTarget, sceneLifecycle));
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
          if (sceneLifecycle && captureCameras && !clearOnly) sceneLifecycle.outputSelection = {
            target: outputTarget, face: previousFace, mip: previousMip,
          };
          const input = this._makeRequest(scene, camera, clearOnly, captureCameras, this._renderTarget, sceneLifecycle);
          // Source resets its public selector before callbacks, but its saved
          // render context still writes the selected private array layer.
          input.face = previousFace; input.sourceFace = 0; requests.push(input);
          this._renderTarget = outputTarget; this._activeCubeFace = previousFace; this._activeMipmapLevel = previousMip;
          outputState = {autoClear: this.autoClear, xrEnabled: this.xr.enabled};
          this.autoClear = false; this.xr.enabled = false;
          // Source _renderOutputLayers converts every private array layer,
          // including preserved/empty layers, into the matching output slice.
          const layers = internal.texture.isArrayTexture === true && internal.texture.image.depth > 1
            ? internal.texture.image.depth : 1;
          for (let layer = 0; layer < layers; layer++) {
            this._activeCubeFace = layers > 1 ? layer : previousFace;
            const quad = this._outputQuad(internal, layer);
            if (captureCameras) {
              this.info.calls++; this.info.render.calls++; this.info.render.frameCalls++;
            }
            const releaseOutputQuad = this._retainOutputQuad(quad);
            let output;
            try { output = this._makeRequest(quad, quad.camera, null, captureCameras, clearOnly ? previousTarget : outputTarget); }
            catch (error) { releaseOutputQuad(); throw error; }
            output.releaseOutputQuad = releaseOutputQuad;
            this._outputQuadLeases.get(quad).destinations.add(output.destination);
            output.outputTransform = {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
              outputColorSpace: this.outputColorSpace};
            requests.push(output);
          }
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
        // A direct-output hook may select the next target. Only our temporary
        // selection is restored; the source conversion path selects its output.
        if (sceneLifecycle?.compilation && sceneLifecycle.begun) {
          this._renderTarget = sceneLifecycle.publicTarget;
          this._activeCubeFace = sceneLifecycle.publicFace; this._activeMipmapLevel = sceneLifecycle.publicMip;
        } else if (convert || this._renderTarget === outputTarget && this._activeCubeFace === previousFace && this._activeMipmapLevel === previousMip) {
          this._renderTarget = completed && convert && !clearOnly && captureCameras ? outputTarget : previousTarget;
          this._activeCubeFace = previousFace; this._activeMipmapLevel = previousMip;
        }
        if (outputState) { this.autoClear = outputState.autoClear; this.xr.enabled = outputState.xrEnabled; }
      }
    }
    _captureNodeFrame({scene, camera, destination, overrideMaterial, captureCameras, mrt = this._mrt, outputTransform = null, renderId, sortState = null, sceneLifecycle = null,
      canvasTarget = this.getCanvasTarget(), sourceTarget = destination === this ? null : destination.target, mipLevel = 0, face = 0, sourceFace = face}) {
      const target = destination === this ? null : destination.target;
      const stencilAttachment = target ? destination.shape.depthAttachment && destination.shape.depthFormat?.endsWith('-stencil8') === true : this.stencil;
      let sourceRaster = this._nodeRaster.get(stencilAttachment);
      if (!sourceRaster) { sourceRaster = createThreeSceneRaster(THREE, {stencilAttachment, fail}).program; this._nodeRaster.set(stencilAttachment, sourceRaster); }
      const reversedDepthBuffer = this.reversedDepthBuffer === true;
      const raster = (material, options) => sourceRaster(material, {...options, reversedDepthBuffer});
      const backgroundMesh = (this._nodeBackgrounds ??= createThreeNodeBackgrounds(THREE)).get(sceneLifecycle?.backgroundScene ?? scene);
      // Pre-init requests can be captured after a PassNode restored the public
      // renderer state. Build and update against that request's source context.
      const previousTarget = this._renderTarget, previousMRT = this._mrt, previousCanvas = this.getCanvasTarget();
      const previousMip = this._activeMipmapLevel, previousFace = this._activeCubeFace;
      const restoreCanvas = canvasTarget !== previousCanvas;
      const previousOutput = outputTransform ? {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
        outputColorSpace: this.outputColorSpace} : null;
      let captured;
      const previousOcclusion = this._occlusionContext;
      const bundleDepth = this._nodeCaptureDepth++;
      try {
        // Direct default output and clear() output quads leave the source's
        // public selector null, although their GPU destination is a target.
        // ScreenNode therefore observes the canvas size on those source paths.
        this._renderTarget = sceneLifecycle?.begun ? sceneLifecycle.publicTarget : sourceTarget; this._mrt = mrt;
        this._activeMipmapLevel = sceneLifecycle?.begun ? sceneLifecycle.publicMip : mipLevel;
        this._activeCubeFace = sceneLifecycle?.begun ? sceneLifecycle.publicFace : sourceFace;
        if (restoreCanvas) this._canvases.select(canvasTarget);
        if (outputTransform) Object.assign(this, outputTransform);
        if (destination === this && !sceneLifecycle?.begun && this.needsFrameBufferTarget) {
          // Retain source target construction/identity for node observations;
          // actual color storage and submission stay in the owned canvas path.
          this.getCanvasTarget(); this._frameBufferTargets ??= new Map();
          this._renderTarget = THREE.Renderer.prototype._getFrameBufferTarget.call(this);
        }
        const queries = this._occlusionQueries ??= createThreeOcclusionQueries({getDevice: () => this._device,
          maxQueries: maxDraws, maxContexts: maxBindings, fail});
        this._occlusionContext = queries.context(destination === this ? this._renderTarget : target, mrt,
          captureCameras ? bundleDepth : 0);
        captured = captureThreeNodeScene(THREE, sceneLifecycle?.scene ?? scene, camera, {renderer: this, overrideMaterial, raster, backgroundMesh, renderLists: this, renderId, sortState,
          matricesUpdated: sceneLifecycle?.begun === true, sceneHooks: sceneLifecycle?.hooked === true,
          clippingView: sceneLifecycle?.clippingView, renderRoot: scene, extraLights: sceneLifecycle?.extraLights,
          renderObjectFunction: sceneLifecycle?.begun ? sceneLifecycle.renderObjectFunction : undefined,
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
        if (captured) this._occlusionContexts.set(captured, this._occlusionContext);
      } finally {
        this._nodeCaptureDepth--;
        this._occlusionContext = previousOcclusion;
        if (sceneLifecycle?.begun) {
          sceneLifecycle.publicTarget = this._renderTarget;
          sceneLifecycle.publicMip = this._activeMipmapLevel; sceneLifecycle.publicFace = this._activeCubeFace;
        } else {
          this._renderTarget = previousTarget;
          this._activeMipmapLevel = previousMip; this._activeCubeFace = previousFace;
        }
        this._mrt = previousMRT;
        if (restoreCanvas) this._canvases.select(previousCanvas);
        if (previousOutput) Object.assign(this, previousOutput);
      }
      if (captured) this._nodeCaptureBytes += captured.bytes;
      return captured;
    }
    _captureMaterialFrame({scene, camera, nodeFrame, overrideMaterial, selectionOnly = false}) {
      const captured = captureThreeMaterialState(THREE, scene, {nodeFrame, overrideMaterial, camera, selectionOnly, allowInstances: selectionOnly,
        maxNodes: sceneLimits.maxNodes ?? 16384, maxMaterials: maxBindings,
        maxValues: MAX_MATERIAL_CAPTURE_VALUES - this._materialCaptureValues});
      if (captured) this._materialCaptureValues += captured.values;
      return captured;
    }
    _releaseRequest(request) {
      finishThreePassState(request.programFrame?.passState, {render: false});
      request.releaseFramebuffer?.(); request.releaseFramebuffer = null;
      if (request.programTextures) {
        for (const lease of request.programTextures.bindings.values()) request.programTextures.owner.release(lease);
        request.programTextures = null;
      }
      request.snapshot?.release();
      request.overrideSnapshot?.release();
      if (request.materialFrame) {
        request.materialFrame.instanceFrame?.release();
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
      request.releaseCanvas?.(); request.releaseCanvas = null;
      request.releaseOutputQuad?.(); request.releaseOutputQuad = null;
    }
    _dropInstanceFrame(materialFrame) {
      const frame = materialFrame?.instanceFrame;
      if (!frame) return;
      frame.release(); materialFrame.instanceFrame = null;
      materialFrame.values -= frame.values; this._materialCaptureValues -= frame.values;
    }
    _dispatchRequest(request) {
      const {scene, destination, snapshot} = request;
      destination._wanted.set(scene, request.key);
      const entry = destination._dispatcher?.entry(scene);
      if (this._drain || this._preparing || (destination === this &&
          (!this._canvases.matches(request.canvasTarget, ...request.canvasSize, request.canvas) ||
            request.hdr !== this._hdr || !request.hdr && request.srgbView !== this._srgbView ||
            (request.outputProfile?.key ?? null) !== this._outputProfileKey)) || !entry || entry.key !== request.key || (!entry.clipping && request.overrideMaterial?.clippingPlanes?.length))
        return this._defer(request);
      try {
        const captured = request.nodeFrame;
        if (request.pmremInput?.owner.preparation([request.pmremInput.lease])) return this._defer(request);
        if (request.snapshot?.preparation?.()) return this._defer(request);
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
    _submit(request) {
      return request.frame.operations ? request.frame.operations.run(() => this._submitFrame(request)) : this._submitFrame(request);
    }
    _submitFrame({scene, camera, frame, destination, mipLevel = 0, face = 0, generateCubeMipmaps = false, generateColorMipmaps = null, programFrame, programTextures, snapshot, overrideSnapshot, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, sortState, clearOnly, outputPass = false, initializeOutputDepth = null, canvasFrameToken = null,
      webglSourceCamera = null, passCameraRequired = false,
      sourceTarget = destination === this ? null : destination.target, deferred = false}) {
      const cameraSafe = webglSourceCamera || programFrame?.toneMappingChunk !== undefined ? materialFrame?.cameraSafe?.() : true;
      if (materialFrame && !cameraSafe && programFrame?.toneMappingChunk !== undefined) delete programFrame.toneMappingChunk;
      if (webglSourceCamera && !cameraSafe) {
        if (passCameraRequired) fail('UNSUPPORTED', 'Pinned pass scene or camera gained callbacks before its captured draw submitted');
        // An application may install hooks while preparation is pending. Keep
        // those callbacks on their original camera and live mutation path.
        camera = webglSourceCamera;
        this._dropInstanceFrame(materialFrame);
      }
      if (initializeOutputDepth) {
        this._targets.assertCurrent(initializeOutputDepth);
        if (this._targetDepthInitialized.has(initializeOutputDepth)) return;
      }
      snapshot?.install();
      overrideSnapshot?.install();
      materialFrame?.instanceFrame?.install();
      nodeFrame?.installAttributes?.();
      if (nodeFrame) {
        const entry = destination._dispatcher?.entry(scene);
        if (!entry) fail('PREPARE', 'Scene has not been prepared');
        entry.bridge.assertNativePrepared(nodeFrame);
      }
      const previous = this._programFrame;
      const previousFramebufferFrame = this._programFramebufferFrame;
      this._programFramebufferFrame = programTextures;
      const previousTarget = this._shaderEncodedOutput && deferred
        ? {target: this._renderTarget, face: this._activeCubeFace, mip: this._activeMipmapLevel} : null;
      if (previousTarget) {
        this._renderTarget = sourceTarget; this._activeCubeFace = face; this._activeMipmapLevel = mipLevel;
      }
      const previousSubmissionDepth = this._webglSubmissionDepth ?? 0;
      const previousFrameToken = this._webglFrameToken;
      if (this._shaderEncodedOutput) this._webglSubmissionDepth = previousSubmissionDepth + 1;
      if (this._shaderEncodedOutput) this._webglFrameToken = canvasFrameToken;
      const occlusion = nodeFrame && !clearOnly ? this._occlusionQueries?.begin(this._occlusionContexts.get(nodeFrame),
        {requested: nodeFrame.occlusionRequested, arrayCamera: camera.isArrayCamera === true && camera.cameras.length > 0}) : null;
      if (occlusion) frame = {...frame, occlusion};
      // Immediate source callbacks must still be able to change live program
      // state. Only deferred calls need the queued renderer-state snapshot.
      this._programFrame = deferred ? programFrame : this._shaderEncodedOutput ? {
        passState: programFrame.passState, passClear: programFrame.passClear, passBackgroundClear: programFrame.passBackgroundClear,
        passTextureEntries: programFrame.passTextureEntries,
        cameraSource: programFrame.cameraSource, cameraSnapshot: programFrame.cameraSnapshot,
      } : undefined;
      try {
        const issue = () => {
          if (destination === this) this._session.render({scene, camera, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, sortState},
            this._hdr ? {...frame, outputEnabled: nodeFrame?.phases?.output !== false} : frame, {frameToken: canvasFrameToken});
          else {
            // Source Renderer initializes depth once per destination target,
            // even when that image is borrowed from an already rendered target.
            // Its first borrower render therefore clears the shared depth too.
            const sourceDepth = destination.target.depthBuffer === true && destination.shape.depthFormat;
            const copiedDepth = mipLevel === 0 && destination.storage?.diagnostics.depthCopies > 0;
            const firstDepth = sourceDepth && !this._targetDepthInitialized.has(destination) &&
              (!this._shaderEncodedOutput && destination.shape.sharedDepth ||
                outputPass && sourceTarget !== null && !copiedDepth);
            const targetFrame = firstDepth && !clearOnly ? {...frame, depthLoadOp: 'clear'} : frame;
            // Callback-free queued draws keep the accepted generation flags.
            // WebGL hooks retain live source state and can change those flags
            // before source mip generation; refresh only this execution's copy.
            const colorMipmaps = this._shaderEncodedOutput && generateColorMipmaps !== null ? generateColorMipmaps.slice() : generateColorMipmaps;
            const onDrawCallbacks = this._shaderEncodedOutput && colorMipmaps !== null ? () => {
              const selected = [];
              for (const [index, color] of destination.shape.colors.entries()) if (color.texture.generateMipmaps) {
                selected.push(index);
              }
              colorMipmaps.splice(0, colorMipmaps.length, ...selected);
            } : undefined;
            this._targets.render(destination, (attachments, framebuffer, depthbuffer) => destination._dispatcher.render({scene, camera, overrideMaterial, matricesUpdated, nodeFrame, backgroundState, materialFrame, sortState, onDrawCallbacks}, {...targetFrame, ...attachments}, framebuffer, depthbuffer),
              {mipLevel, face, generateMipmaps: generateCubeMipmaps, generateColorMipmaps: colorMipmaps,
                resolveColor: !(this._shaderEncodedOutput && clearOnly && destination.shape.sampleCount > 1)});
            // Pinned Renderer tracks this per target, including color-only
            // manual clears; selecting another mip does not reset the flag.
            if (!this._shaderEncodedOutput && sourceDepth && (!destination.shape.sharedDepth || !clearOnly))
              this._targetDepthInitialized.add(destination);
          }
        };
        const textures = nodeFrame?.textureOwner, storage = nodeFrame?.storageOwner;
        const resources = () => {
          if (nodeFrame) nodeFrame = {...nodeFrame, draws: nodeFrame.draws.map(draw => ({...draw,
            nodeIndirect: draw.nodeIndirect ? {...storage.resolve(draw.nodeIndirect.lease),
              offsets: draw.nodeIndirect.offsets, skip: draw.nodeIndirect.skip} : null,
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
        occlusion?.commit();
        finishThreePassState(programFrame.passState, {render: !clearOnly});
      } catch (error) { occlusion?.cancel(); if (!isPrepareBoundary(error)) frame.timestamp?.cancel(); throw error; }
      finally {
        this._programFrame = previous; this._programFramebufferFrame = previousFramebufferFrame; this._webglSubmissionDepth = previousSubmissionDepth;
        this._webglFrameToken = previousFrameToken;
        if (previousTarget) {
          this._renderTarget = previousTarget.target; this._activeCubeFace = previousTarget.face; this._activeMipmapLevel = previousTarget.mip;
        }
      }
      if (initializeOutputDepth) this._targetDepthInitialized.add(initializeOutputDepth);
      const d = destination._dispatcher.entry(scene)?.lastDiagnostics;
      if (d && !this._disposed) {
        this.info.render.drawCalls += d.sourceDraws;
        this.info.f3d.nativeDrawCalls = d.drawCalls;
        const memory = this.info.memory;
        memory.geometries = d.geometryCount;
        memory.attributesSize = d.geometryBytes + d.instanceBytes + d.deformationBytes;
        memory.uniformBuffersSize = d.rendererBytes;
        memory.renderTargets = this._targets.size;
        memory.total = memory.attributesSize + memory.uniformBuffersSize + this._targets.allocatedBytes + memory.readbackBuffersSize;
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
      if (this._pending.length >= maxPendingRenders) fail('LIMIT', 'Too many render operations are waiting for preparation');
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
        if (request.sceneHookFailure) throw request.sceneHookFailure;
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
    get compile() { return this.compileAsync; }
    async compileAsync(scene, camera, targetScene = null) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._isDeviceLost === true) return;
      if (camera !== undefined && !(camera instanceof THREE.Camera)) fail('SOURCE', 'Expected a source camera');
      const compiled = this._shaderEncodedOutput ? targetScene ?? scene : scene;
      if (!(compiled instanceof THREE.Object3D)) fail('SOURCE', 'Expected a source Object3D root');
      if (targetScene !== null && !(targetScene instanceof THREE.Object3D)) fail('SOURCE', 'Expected a source target scene');
      // Source compilation initializes before its synchronous scene lifecycle.
      // Once admitted, retries consume its capture without replaying hooks.
      if (!this._shaderEncodedOutput && !this._initialized) await this.init();
      // Capture before initialization or another compile can yield. Preparation
      // and later draws address the same stable execution root.
      if (!this._shaderEncodedOutput) this._nodes.nodeFrame.update();
      const lifecycle = this._sceneLifecycle(compiled, camera, true, targetScene);
      const requests = this._makeOutputRequests(compiled, camera, null, false, lifecycle), compilation = {frameAdvanced: !this._shaderEncodedOutput};
      for (const request of requests) request.compilation = compilation;
      try {
        await this.init();
        if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
        while (this._drain) await this._drain;
        for (const request of requests) {
          request.destination._wanted.set(request.scene, request.key);
          await this._prepare(request.destination, request);
        }
      } finally {
        // Geometry-node compilation can enqueue initialization/update kernels.
        // Drain only after _prepare has released its own queue barrier, even
        // when a later node callback rejected the parent compilation.
        try { while (this._drain) await this._drain; }
        finally { for (const request of requests) this._releaseRequest(request); }
      }
    }
    /** All scene preparations share one queue; targets never rebuild the canvas. */
    _prepare(destination = this, request = null) {
      const run = (this._preparing ?? Promise.resolve()).catch(() => {}).then(async () => {
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        if (request?.pmremInput) await request.pmremInput.owner.preparation([request.pmremInput.lease]);
        const snapshotPreparation = request?.snapshot?.preparation?.();
        if (snapshotPreparation) await snapshotPreparation;
        if (request?.nodeCaptureDeferred) {
          // The source's async-before-init path observes scene inputs at this
          // boundary. Refresh the selector before callbacks so fixed background
          // ownership and native sky packets describe that same accepted frame.
          const pendingBefore = this._pending.length;
          let pendingAfterCapture = null;
          try {
            if (request.sceneLifecycle) this._beginSceneLifecycle(request.sceneLifecycle, {
              target: destination === this ? null : destination.target,
              sourceTarget: request.sourceTarget, renderId: request.renderId,
              sourceFace: request.sourceFace ?? request.face, sourceMip: request.mipLevel,
            });
            if (request.sceneLifecycle?.begun) {
              // The backend selects the current canvas after Scene-before;
              // retain that surface while its raster context stays pre-hook.
              const canvasTarget = this.getCanvasTarget(), canvas = canvasTarget.domElement;
              const releaseCanvas = this._canvases.retain(canvasTarget, canvas);
              request.releaseCanvas?.();
              Object.assign(request, {canvasTarget, canvas, canvasSize: this._bufferSize(), releaseCanvas});
            }
            request.backgroundState = captureThreeBackground(THREE, request.sceneLifecycle?.backgroundScene ?? request.scene);
            const sceneReference = request.sceneLifecycle?.scene ?? request.scene;
            request.overrideMaterial = sceneReference.isScene === true ? sceneReference.overrideMaterial : null;
            request.sortState = Object.freeze({sortObjects: this.sortObjects === true,
              opaqueSort: this._opaqueSort, transparentSort: this._transparentSort});
            if (request.sceneLifecycle?.hooked) {
              const frame = this._frame(request.scene, request.clearOnly, destination === this ? null : destination.target,
                request.mipLevel, request.outputProfile, request.sceneLifecycle.rasterState);
              request.frame = {...frame, timestamp: request.frame.timestamp, operations: request.frame.operations,
                ...(request.frame.arrayCamera ? {arrayCamera: request.frame.arrayCamera} : {})};
              Object.assign(request.programFrame, {toneMapping: this.toneMapping, toneMappingExposure: this.toneMappingExposure,
                outputColorSpace: this.outputColorSpace});
            }
            request.key = sceneKey(this, request.scene, request.backgroundState);
            destination._wanted.set(request.scene, request.key);
            request.nodeFrame = this._captureNodeFrame(request);
            if (request.nodeFrame?.sceneHierarchy) request.backgroundState = Object.freeze({...request.backgroundState,
              nativeSceneProfile: Object.freeze({key: request.key, shadows: this.shadowMap.enabled})});
            request.matricesUpdated ||= !!request.nodeFrame || request.sceneLifecycle?.begun === true;
            if (request.captureCameras) {
              if (!request.matricesUpdated) this._updateNativeMatrices(request.scene, request.camera);
              request.camera = request.camera.isArrayCamera ? captureThreeArrayCamera(THREE, request.camera) :
                captureThreeCamera(THREE, request.camera);
              request.matricesUpdated = true;
            }
            request.materialFrame = this._captureMaterialFrame(request);
            this._restoreSceneLifecycle(request.sceneLifecycle);
            pendingAfterCapture = this._pending.length;
            try { this._finishSceneLifecycle(request.sceneLifecycle); }
            catch (error) { this._pending.push({sceneHookFailure: error}); }
          } catch (error) {
            request.nodeCaptureFailure = {error};
          } finally {
            this._restoreSceneLifecycle(request.sceneLifecycle);
            request.nodeCaptureDeferred = false;
            const parent = this._pending.indexOf(request);
            if (parent !== -1 && this._pending.length > pendingBefore) {
              // Capture is synchronous: this suffix contains precisely the
              // recursive calls accepted by its source lifecycle. Preserve
              // their order before this parent and leave preexisting later
              // requests behind it, including on a late capture failure.
              const children = this._pending.splice(pendingBefore);
              const boundary = pendingAfterCapture === null ? children.length : pendingAfterCapture - pendingBefore;
              const prerequisites = children.slice(0, boundary), following = children.slice(boundary);
              this._pending.splice(parent, 0, ...prerequisites);
              const last = this._pending.indexOf(request.sceneLifecycle?.lastRequest ?? request);
              this._pending.splice((last === -1 ? this._pending.indexOf(request) : last) + 1, 0, ...following);
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
        // Deferred pre-init captures discover geometry kernels only here. The
        // source advances one frame for the entire public compilation, keeping
        // its previous renderId across all output requests.
        if (captured?.compileAttributes && request.compilation && !request.compilation.frameAdvanced) {
          this._nodes.nodeFrame.update(); request.compilation.frameAdvanced = true;
        }
        const textureReady = captured?.textureOwner?.preparation(captured.textureUses);
        if (textureReady) await textureReady;
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        // Selection captures keep live source material objects. Hooks installed
        // after acceptance must retain their global shader reads and writes.
        if (request?.programFrame?.toneMappingChunk !== undefined && request.materialFrame && !request.materialFrame.cameraSafe())
          delete request.programFrame.toneMappingChunk;
        if (request?.materialFrame?.instanceFrame && !request.materialFrame.cameraSafe()) {
          if (request.passCameraRequired) fail('UNSUPPORTED', 'Pinned pass scene or camera gained callbacks before preparation');
          this._dropInstanceFrame(request.materialFrame);
        }
        const previous = this._programFrame, previousRoot = destination._preparationRoot;
        const previousFramebufferFrame = this._programFramebufferFrame;
        this._programFramebufferFrame = request?.programTextures;
        const previousOptions = destination._preparationOptions;
        this._programFrame = request?.programFrame;
        destination._preparationRoot = request?.scene ?? null;
        destination._preparationOptions = request ? {overrideMaterial: request.overrideMaterial, nodeFrame: request.nodeFrame,
          backgroundState: request.backgroundState, materialFrame: request.materialFrame, privateSnapshot: !!request.snapshot} : undefined;
        try {
          request?.snapshot?.install();
          request?.overrideSnapshot?.install();
          request?.materialFrame?.instanceFrame?.install();
          const prepare = async () => {
          captured?.installAttributes?.();
          if (destination !== this) {
            this._targets.ensure(destination);
            if (!destination._dispatcher) createDispatcher(this, destination)(this._device,
              {...destination.storage.rendererOptions, depthFormat: destination.shape.depthAttachment ? destination.shape.depthFormat : null},
              {signal: destination.lifetime.signal});
            await destination._dispatcher.prepare();
            this._targets.assertCurrent(destination);
          } else {
            const canvasTarget = request?.canvasTarget ?? this.getCanvasTarget(), canvasSize = request?.canvasSize ?? this._bufferSize();
            const canvas = request?.canvas ?? canvasTarget.domElement;
            this._canvases.activate(canvasTarget, canvas);
            const tone = this._toneMapping(request?.programFrame.toneMapping ?? this.toneMapping);
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
            const outputProfile = request ? request.outputProfile : this._captureOutputProfile();
            const hdr = selectOutput(), srgbView = this._shaderEncodedOutput ? hdr : request?.srgbView ?? this._canvasSrgbView();
            if (request) request.srgbView = srgbView;
            if (!this._session || hdr !== this._hdr || !hdr && srgbView !== this._srgbView ||
                this._shaderEncodedOutput && outputProfile?.canvasKey !== this._canvasOutputKey) {
              this._session?.dispose(); await this._createSession(hdr, srgbView, canvasTarget, canvasSize, canvas, outputProfile);
            } else this._session.resize(...canvasSize);
            this._rebuild = false;
            try {
              // The source's working-linear color/depth/stencil target survives
              // output-space changes. Only its final conversion program changes.
              // Keep the actual canvas view selection separate from that program.
              const changeOutput = hdr && (outputProfile?.key ?? null) !== this._outputProfileKey;
              await this._session.prepare(changeOutput ? {
                outputColorSpace: outputProfile.transfer,
                outputColorMatrix: outputProfile.matrix ?? undefined,
                sourceOutput: this._preferredFormat === 'rgba16float' || outputProfile.sourceOutput,
              } : undefined);
              this._outputProfileKey = outputProfile?.key ?? null;
              this._canvases.save();
            }
            catch (error) {
              // Encoded history cannot move into a linear whole-image output
              // target: it would be lost or tone-mapped again on later frames.
              // Even auto-clears can be scissored or retain other attachments.
              // Logarithmic fragment depth also requires the source programs.
              // An application-defined tone operator has no fixed-pass substitute.
              if (error?.code !== 'THREE_SCENE_TONE_MAPPING' || tone === 'custom' || this._hdrFallback === true || globalThis.F3D_NO_FALLBACK === true ||
                  this._shaderEncodedOutput && outputProfile?.space !== SRGBColorSpace ||
                  this._shaderEncodedOutput && (this.preserveDrawingBuffer === true || this._straightAlphaCanvas || this.logarithmicDepthBuffer === true || this.reversedDepthBuffer === true)) throw error;
              this._hdrFallback = true;
              this._session.dispose();
              const fallbackHdr = selectOutput();
              if (request && this._shaderEncodedOutput) request.srgbView = fallbackHdr;
              await this._createSession(fallbackHdr, this._shaderEncodedOutput ? fallbackHdr : srgbView, canvasTarget, canvasSize, canvas, outputProfile);
              this._rebuild = false;
              await this._session.prepare();
            }
          }
          this.info.f3d.preparations++;
          };
          if (captured?.compileAttributes) await captured.compileAttributes(prepare);
          else { captured?.prepareAttributes?.(); await prepare(); }
        } finally { this._programFrame = previous; this._programFramebufferFrame = previousFramebufferFrame; destination._preparationRoot = previousRoot; destination._preparationOptions = previousOptions; }
      });
      this._preparing = run;
      run.finally(() => { if (this._preparing === run) this._preparing = null; }).catch(() => {});
      return run;
    }
    async waitForGPU() {
      this._gpuWaiters = (this._gpuWaiters ?? 0) + 1;
      try {
        while (this._drain) await this._drain;
        if (this._preparing) await this._preparing;
        await this._canvases.whenIdle();
        await this._transfers.whenIdle();
        await this._targets.whenIdle();
        await this._occlusionQueries?.whenIdle();
      } finally {
        // Concurrent callers can still be awaiting a private PMREM bridge or
        // ping-pong image after the first fence finishes. Retire only once all
        // active waits have relinquished their resources.
        if (--this._gpuWaiters === 0) {
          const releases = this._pmremIdleReleases ?? []; this._pmremIdleReleases = null;
          for (const release of releases) try { release(); } catch (error) { this._deferredError ??= error; }
        }
      }
    }

    // ---- animation loop (source Animation semantics: callback(time, xrFrame)) ----
    async setAnimationLoop(callback) {
      if (this._initialized === false) await this.init();
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
      // Replacing/stopping the WebGL loop also invalidates a suspended resume,
      // even when the caller reuses the same callback function.
      this._animationWaiting = null;
      if (this._inspector.isRunning) this._inspector.finish();
      this._animationCallback = callback;
      if (callback === null || this._disposed) return;
      const schedule = () => {
        if (this._animationCallback === callback && !this._disposed)
          this._animationHandle = host.requestAnimationFrame(tick);
      };
      const tick = time => {
        if (this._animationCallback !== callback || this._disposed) return;
        this._animationHandle = null;
        if (this._drain && this._pending.some(request =>
          request.animationBarrier === true || !request.transfer && !request.readback)) {
          // Source WebGL render/compile work finishes synchronously before the
          // next RAF can update procedural geometry. Preserve that boundary
          // while our device and pipelines prepare asynchronously; accepted
          // draws stay queued, without advancing another simulation frame.
          const waiting = this._animationWaiting = {};
          const resume = () => {
            if (this._animationWaiting !== waiting) return;
            this._animationWaiting = null; schedule();
          };
          this._drain.then(resume, resume);
          return;
        }
        schedule();
        // Source WebGL submits this whole callback before the browser presents.
        // Deferred preparation may span host turns: accepted draws/copies keep
        // one private image, while a later callback starts a fresh buffer.
        const previous = this._webglFrameToken;
        this._webglFrameToken = {};
        try { this._advanceAnimationFrame(); callback(time, null); }
        finally { this._webglFrameToken = previous; }
      };
      schedule();
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

    dispose() { return this._disposeWithResources(); }
    _disposeWithResources() {
      const asynchronous = this.isWebGPURenderer === true;
      if (this._disposePromise) return asynchronous ? this._disposePromise : undefined;
      let resolve, reject, failure, failed = false;
      const completion = this._disposePromise = new Promise((yes, no) => { resolve = yes; reject = no; });
      // Source callers may ignore dispose(). Keep that use safe while returning
      // the original rejecting promise to callers that await native cleanup.
      completion.catch(() => {});
      const cleanup = [], record = error => { if (!failed) { failed = true; failure = error; } };
      const dispose = operation => {
        try {
          const result = operation();
          if (result?.then) {
            const pending = Promise.resolve(result); pending.catch(() => {}); cleanup.push(pending);
          }
        } catch (error) { record(error); }
      };
      const running = [!this._initialized && this._initPromise, this._preparing, this._drain].filter(Boolean);
      const requests = this._pending; this._pending = [];
      // Disposal must stop the current loop without starting initialization or
      // adding an asynchronous setter behind an already pending start.
      this._disposed = true;
      this._animationCallback = null; this._animationWaiting = null;
      dispose(() => this._setLoop(null));
      this._animationStarted = false;
      if (this._animationHandle !== null)
        dispose(() => this._animationHost?.cancelAnimationFrame?.(this._animationHandle));
      this._animationHandle = null;
      dispose(() => this._releaseDeviceEvents?.()); this._releaseDeviceEvents = null;
      // The pinned native renderer clears public Info before Inspector sees
      // disposal. Neither component is disposed before initialization there.
      dispose(() => {
        if (asynchronous && this._initialized) this.info.dispose();
        if (!asynchronous || this._initialized) this._inspector.dispose?.();
      });
      const stopped = new F3DRendererError('DISPOSED', 'Renderer is disposed');
      // Owners keep their established operation-specific cancellation errors.
      // Admission has already stopped; abort the shared lifetime only after
      // independently owned transfers, compute work and timing reads stop.
      dispose(() => this._disposeResources?.(dispose));
      dispose(() => this._occlusionQueries?.dispose());
      dispose(() => this._timestamps.dispose());
      dispose(() => this._targetLifetime.abort(stopped));
      dispose(() => this._targets.dispose());
      for (const request of requests) {
        dispose(() => this._releaseRequest(request));
        dispose(() => request.reject?.(stopped));
      }
      const finish = async () => {
        // Cancellation settles these renderer-owned tasks without waiting for
        // uncancellable host acquisition or for completion of submitted GPU work.
        // Their operation errors remain on the original render/init promises.
        if (running.length) await Promise.allSettled(running);
        dispose(() => this._release(dispose));
        for (const result of await Promise.allSettled(cleanup))
          if (result.status === 'rejected') record(result.reason);
        if (!failed && asynchronous) dispose(() => { this.setRenderTarget(null); });
        if (failed) reject(failure); else resolve();
      };
      finish().catch(reject);
      if (asynchronous) return completion;
      if (failed) throw failure; // WebGLRenderer keeps its synchronous surface.
    }
  }
  const nativeInit = WebGPURenderer.prototype.init;
  const Routed = withThreeCompute(WebGPURenderer, THREE);
  if (exactBackend) {
    SOURCE_RENDERER_CONSTRUCTORS.set(Routed, exactBackend);
    Object.defineProperty(Routed, Symbol.hasInstance, {value(value) {
      return ordinaryHasInstance.call(this, value) ||
        this === Routed && ordinaryHasInstance.call(exactBackend, value);
    }});
  }
  return Routed;
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
 * - preserveDrawingBuffer retains owned color/depth/MSAA storage across frames.
 */
export function createWebGLRendererClass(THREE, classOptions = {}) {
  const {exactBackend = null, ...options} = classOptions;
  if (exactBackend !== null && typeof exactBackend !== 'function') fail('OPTIONS', 'exactBackend must be a renderer constructor');
  // Native WGSL leaves render-target UVs unchanged; WebGL programs retain their
  // bottom-left sampling convention. Keep this owner policy fixed per facade.
  const Base = createWebGPURendererClass(THREE, {...options, targetTextureFlipY: true});
  const glOnly = name => ({get() { fail('UNSUPPORTED', `WebGLRenderer.${name} describes a GL context; this route has none`); }, configurable: true});
  class WebGLRenderer extends Base {
    // CubeCamera selects its face directions/up vectors from this public
    // source fact. The program compiler converts GL clip depth separately.
    get coordinateSystem() { return THREE.WebGLCoordinateSystem; }
    get outputColorSpace() { return this._outputColorSpace; }
    set outputColorSpace(space) {
      if ((this._webglSubmissionDepth > 0 || this._programCompileCallbackDepth > 0) && space !== this._outputColorSpace)
        fail('UNSUPPORTED', 'Changing WebGL output color space inside rendering callbacks requires exact renderer execution');
      // Source publishes the selector before asking ColorManagement for the
      // drawing-buffer gamut; malformed definitions retain their source error.
      this._outputColorSpace = space;
      const gamut = THREE.ColorManagement._getDrawingBufferColorSpace(space);
      if (!['srgb', 'display-p3'].includes(gamut)) fail('UNSUPPORTED', 'WebGL output requires an sRGB or Display-P3 canvas gamut');
      if (gamut === this._drawingBufferColorSpace) return;
      this._drawingBufferColorSpace = gamut;
      this._drawingBufferVersion = (this._drawingBufferVersion ?? 0) + 1;
      if (!this._shaderEncodedOutput || this._disposed) return;
      // WebGL reallocates its drawing buffer when the display gamut changes,
      // even with preserveDrawingBuffer. Retain this ordered invalidation when
      // A -> B -> A happens without a render in B; shader transfer changes in
      // the same gamut do not invalidate color/depth/stencil history.
      const canvasTarget = this.getCanvasTarget(), canvas = canvasTarget.domElement;
      const profile = this._captureOutputProfile(), request = {framebufferCopy: true,
        releaseCanvas: this._canvases.retain(canvasTarget, canvas), issue: () => {
          this._canvases.activate(canvasTarget, canvas);
          if (this._canvasOutputKey !== profile.canvasKey) {
            this._session?.dispose(); this._session = null; this._dispatcher = null;
            this._outputProfileKey = null; this._canvasOutputKey = profile.canvasKey;
            this._canvases.save();
          }
        }};
      try {
        if (this._initialized && !this._drain && !this._preparing && !this._programDrawContext && !this._pending.length) {
          try { request.issue(); } finally { this._releaseRequest(request); }
        } else this._enqueue(request);
      } catch (error) { this._releaseRequest(request); throw error; }
    }
    constructor(parameters = {}) {
      if (exactBackend && !hasWebGPUHost(options.gpu)) return new exactBackend(...arguments);
      if (exactBackend && requiresSourceParameters(parameters, ['context']))
        return new exactBackend(...arguments);
      if (!parameters || typeof parameters !== 'object') fail('OPTIONS', 'Expected renderer parameters');
      // A supplied GL context is the exact backend by definition.
      if (!exactBackend && parameters.context)
        fail('ROUTE', 'A supplied rendering context selects the exact WebGL backend');
      for (const key of Object.keys(parameters)) if (!WEBGL_PARAMETERS.has(key)) fail('OPTIONS', `Unknown renderer parameter: ${key}`);
      const {premultipliedAlpha = true, preserveDrawingBuffer = false, powerPreference = 'default',
        failIfMajorPerformanceCaveat, precision, outputBufferType, context, ...rest} = parameters;
      super({...rest, alpha: parameters.alpha ?? false,
        ...(powerPreference === 'default' ? {} : {powerPreference})});
      this.isWebGPURenderer = false;
      this.isWebGLRenderer = true;
      delete this.library; // WebGLRenderer has no node library.
      this.premultipliedAlpha = premultipliedAlpha;
      // Context creation attributes do not change after construction. Public
      // property edits must not reinterpret an existing drawing buffer.
      this._straightAlphaCanvas = premultipliedAlpha === false;
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
      registerThreePassState(this, THREE);
      registerThreeGpuComputation(this);
      // Source construction is synchronous: start device negotiation now.
      this.init().catch(error => { this._deferredError ??= error; });
    }
    _parameterNames() { return WEBGL_PARAMETERS; }
    async _init() {
      if (this._outputBufferType !== undefined && this._outputBufferType !== THREE.UnsignedByteType)
        fail('UNSUPPORTED', 'Non-8-bit output buffers are not admitted');
      return super._init();
    }
    _needsGlobalClipping() { return this.clippingPlanes.length > 0; }
    _beforeRender() {
      this._clippingControls.planes = this.clippingPlanes;
      this._clippingControls.localClippingEnabled = this.localClippingEnabled;
    }
    init() { return super.init(); }
    setAnimationLoop(callback) { this._setLoop(callback); }
    getContext() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    setClearStencil(value) {
      super.setClearStencil(value);
      registerThreePassState(this, THREE).buffers.stencil.setClear(value);
    }
    getContextAttributes() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    forceContextLoss() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    forceContextRestore() { fail('UNSUPPORTED', 'This WebGLRenderer route has no GL context'); }
    renderBufferDirect(camera, scene, geometry, material, object, group) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (!this._programDrawContext)
        fail('UNSUPPORTED', 'Direct program draws require an active ordinary-camera rendering callback');
      return this._programDrawContext.draw(camera, scene, geometry, material, object, group);
    }
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
    /** PMREM's retained helper compiles an empty geometry before its texture
     * exists. Capture its actual shader now, without admitting a draw/upload;
     * the same device modules are reused when its conversion planes render. */
    _compilePMREM(object) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      const profile = {toneMapping: THREE.NoToneMapping, outputColorSpace: THREE.LinearSRGBColorSpace,
        logarithmicDepthBuffer: this.logarithmicDepthBuffer === true,
        reversedDepthBuffer: this.reversedDepthBuffer === true, renderer: this};
      const support = createThreeProgramSupport({three: THREE, state: () => profile, maxPrograms: 1});
      let program;
      try { program = support.compile(object.material, object).program; }
      finally { support.dispose(); }
      const signal = this._targetLifetime.signal;
      let rejectStopped;
      const stopped = new Promise((_, reject) => { rejectStopped = reject; });
      const abort = () => rejectStopped(signal.reason);
      signal.addEventListener('abort', abort, {once: true});
      const work = (this._preparing ?? Promise.resolve()).catch(() => {}).then(async () => {
        await this.init();
        if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
        const vertex = gpuProgramModule(this._device, {code: program.vertex, label: 'PMREM/warmup/vertex'});
        const fragment = gpuProgramModule(this._device, {code: program.fragment, label: 'PMREM/warmup/fragment'});
        await Promise.all([vertex.ready, fragment.ready]);
      });
      const run = this._preparing = Promise.race([work, stopped]);
      run.catch(error => { this._deferredError ??= error; });
      run.finally(() => {
        signal.removeEventListener('abort', abort);
        if (this._preparing === run) this._preparing = null;
      }).catch(() => {});
    }
    getCurrentViewport(target) {
      return this._renderTarget ? target.copy(this._renderTarget.viewport).round() : target.copy(this._viewport).multiplyScalar(this._pixelRatio).round();
    }
  }
  for (const name of ['extensions', 'properties', 'renderLists'])
    Object.defineProperty(WebGLRenderer.prototype, name, glOnly(name));
  // WebGLState: only facts true of THIS renderer are served. Reversed depth
  // reports the active owned depth mode. setMask(true) is what
  // CubeCamera/Reflector/Water call before clearing depth. Preserve that request
  // in the private state too, where a retained MaskPass may hold its lock. Any
  // other GL state access (masks off, stencil, raw viewport) throws explicitly.
  const glState = (path, known) => new Proxy(known, {get(target, key) {
    if (typeof key === 'symbol' || Object.hasOwn(target, key)) return target[key];
    fail('UNSUPPORTED', `WebGLRenderer.state.${path}${String(key)} describes a GL context; this route has none`);
  }});
  const webglStates = new WeakMap();
  Object.defineProperty(WebGLRenderer.prototype, 'state', {configurable: true, get() {
    if (!webglStates.has(this)) {
      const depthState = glState('buffers.depth.', {
        getReversed: () => this.reversedDepthBuffer === true,
        setMask: mask => {
          if (mask !== true) fail('UNSUPPORTED', 'WebGLRenderer.state.buffers.depth.setMask(false) needs GL depth-mask state; this route has none');
          registerThreePassState(this, THREE).buffers.depth.setMask(true);
        },
      });
      webglStates.set(this, glState('', {buffers: glState('buffers.', {depth: depthState})}));
    }
    return webglStates.get(this);
  }});
  // Only capabilities that describe THIS renderer's behavior are served: the
  // sampler anisotropy ceiling it applies (WebGPU clamps maxAnisotropy to 16, as
  // r186's WebGPU backend reports) and the depth/precision modes it admits.
  // GL-context limits (texture units, uniform vectors, ...) throw explicitly.
  Object.defineProperty(WebGLRenderer.prototype, 'capabilities', {configurable: true, get() {
    const known = {getMaxAnisotropy: () => 16, logarithmicDepthBuffer: this.logarithmicDepthBuffer === true, reversedDepthBuffer: this.reversedDepthBuffer === true, precision: 'highp',
      getMaxPrecision: () => 'highp', isWebGL2: true};
    return new Proxy(known, {get(target, key) {
      if (typeof key === 'symbol' || Object.hasOwn(target, key)) return target[key];
      fail('UNSUPPORTED', `WebGLRenderer.capabilities.${String(key)} describes a GL context; this route has none`);
    }});
  }});
  if (exactBackend) {
    SOURCE_RENDERER_CONSTRUCTORS.set(WebGLRenderer, exactBackend);
    Object.defineProperty(WebGLRenderer, Symbol.hasInstance, {value(value) {
      return ordinaryHasInstance.call(this, value) ||
        this === WebGLRenderer && ordinaryHasInstance.call(exactBackend, value);
    }});
  }
  return WebGLRenderer;
}

/** The native WebGPU route retains the pinned PMREM construction/filtering
 * helper and supplies this renderer for every generated draw. The helper does
 * not construct a source renderer or backend. Its real CubeUV targets can be
 * sampled by native node materials and backgrounds through the target owner.
 *
 * Once the WebGL-authored renderer is initialized, texture-input calls run the
 * retained source conversion and GGX helper against its existing target owner.
 * They return real CubeUV targets, including an explicitly reused destination;
 * source draws and ordinary input bytes are captured before this call returns.
 * Private reads survive later input disposal and share acknowledged upload
 * history with subsequent ordinary maps and panorama/cube backgrounds.
 *
 * Texture-input calls before initialization retain their earlier descriptors.
 * fromScene captures rigid scenes (including CPU instances and source shader
 * uniforms), runs all six source views and sigma pre-blur, and returns a real
 * target. Untextured scenes can be accepted before device initialization;
 * ordinary material maps use the initialized shared texture owner. Captured
 * geometry keeps WebGL upload
 * history; transferring it to/from an ordinary scene owner requires disposal.
 * Shader warmup retains source material construction and validates WGSL modules
 * shared with later conversion draws; it needs neither geometry nor an input
 * texture and does not submit a render. The legacy delayed bake paths do not
 * claim source PMREM pixel equivalence. Other renderers get the upstream class.
 */
export function createPMREMGeneratorClass(THREE, {exactBackend = null} = {}) {
  const nativeGenerator = renderer => {
    const generator = new THREE.PMREMGenerator(renderer), source = THREE.PMREMGenerator.prototype;
    const releaseWhenReady = release => {
      const retire = () => {
        if (renderer._gpuWaiters > 0) (renderer._pmremIdleReleases ??= []).push(release);
        else release();
      };
      if (!renderer._drain && !renderer._preparing) { retire(); return; }
      renderer.waitForGPU().then(retire, retire).catch(error => { renderer._deferredError ??= error; });
    };
    const retireRoot = root => {
      if (!renderer._shaderEncodedOutput) return;
      const executions = PMREM_EXECUTIONS.get(generator);
      for (const [destination, execution] of executions?.get(root) ?? []) {
        destination._dispatcher?.retire(execution); destination._wanted.delete(execution);
      }
      executions?.delete(root); renderer._passSnapshots.retire(root);
    };
    const retire = (method, all) => {
      // r186 replaces these fields on a size change. Capture their resource
      // identities now: cleanup of the live generator after the queue drains
      // would otherwise dispose the next generation's targets and materials.
      const roots = all ? new Set([...generator._lodMeshes, ...(PMREM_EXECUTIONS.get(generator)?.keys() ?? [])])
        : generator._lodMeshes.slice();
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
      const release = () => {
        for (const root of roots) retireRoot(root);
        source[method].call(retired);
      };
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
        if (previous) releaseWhenReady(() => { retireRoot(previous); previous.geometry.dispose(); previous.material.dispose(); });
        if (renderer._shaderEncodedOutput && !renderer.reversedDepthBuffer) {
          // This output is newly allocated. WebGL initializes its normal depth
          // storage to one; WebGPU initializes to zero. The source helper loads
          // depth with autoClear=false (and explicitly clears reversed depth).
          const target = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel(),
            depth = renderer.getClearDepth();
          try { renderer.setClearDepth(1); renderer.setRenderTarget(args[3]); renderer.clearDepth(); }
          finally { renderer.setClearDepth(depth); renderer.setRenderTarget(target, face, mip); }
        }
        return source._sceneToCubeUV.apply(generator, args);
      }},
    });
    if (renderer._shaderEncodedOutput) {
      // Keep r186's material/empty-mesh construction. Only the GPU compilation
      // boundary changes; no source renderer, draw or texture owner is created.
      generator._compileMaterial = material => source._compileMaterial.call({
        _renderer: {compile: object => renderer._compilePMREM(object)},
      }, material);
    }
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
      this._generator = nativeGenerator(renderer);
    }
    fromScene(scene, sigma = 0, near = 0.1, far = 100, options = {}) {
      if (!(scene instanceof THREE.Object3D)) fail('SOURCE', 'Expected a source Object3D (r186 renders any root)');
      const renderer = this._renderer;
      {
        if (renderer.shadowMap.enabled || renderer.sortObjects && (renderer._opaqueSort || renderer._transparentSort))
          fail('UNSUPPORTED', 'PMREM scene capture requires no shadows or custom sorting');
        // One immutable scene generation serves the six source camera views.
        // Capture before the helper changes background/render state or queues a
        // draw, so unsupported mutable inputs fail at the public call boundary.
        const pendingSources = new Set();
        const capture = () => renderer._passSnapshots.capturePMREMScene(scene, new THREE.PerspectiveCamera(90, 1, near, far), renderer._device ?? renderer._parameters.device,
          {cube: true, position: options.position === undefined ? new THREE.Vector3() : options.position, sortObjects: renderer.sortObjects,
            captureTexture: texture => renderer._capturePMREMTexture(texture, pendingSources)});
        // Rejecting a later map must not establish an earlier map's upload
        // history or redirect its future ordinary consumers to this owner.
        // Only synchronous capture is transactional; source filtering queues
        // its draws after acceptance, outside the resource transaction.
        const snapshot = renderer._device ? renderer._getSampledTextureInputs().captureDispatch(capture) : capture();
        if (pendingSources.size) {
          let sources = PMREM_SOURCES.get(renderer);
          if (!sources) PMREM_SOURCES.set(renderer, sources = new WeakSet());
          for (const source of pendingSources) sources.add(source);
        }
        const previous = PMREM_CAPTURES.get(renderer);
        PMREM_CAPTURES.set(renderer, {generator: this._generator, input: null, scene, snapshot});
        try { return this._generator.fromScene(scene, sigma, near, far, options); }
        finally {
          if (previous) PMREM_CAPTURES.set(renderer, previous); else PMREM_CAPTURES.delete(renderer);
          snapshot.release();
        }
      }
    }
    async fromSceneAsync(...args) { return this.fromScene(...args); }
    fromEquirectangular(equirectangular, renderTarget = null) {
      // Preserve the existing pre-initialization admission. A generator made
      // before init can use real targets on every later initialized call.
      if (!this._renderer._initialized) {
        if (renderTarget !== null) fail('UNSUPPORTED', 'Explicit PMREM targets require an initialized renderer');
        const texture = equirectangular.clone();
        texture.mapping = THREE.EquirectangularReflectionMapping; texture.needsUpdate = true;
        return {isRenderTarget: true, texture, dispose() { texture.dispose(); }};
      }
      return this._fromTexture('fromEquirectangular', equirectangular, renderTarget);
    }
    async fromEquirectangularAsync(...args) { return this.fromEquirectangular(...args); }
    /** Initialized calls capture the retained source conversion/filter passes.
     * Earlier calls preserve the delayed scene-environment descriptor. */
    fromCubemap(cubemap, renderTarget = null) {
      if (!this._renderer._initialized) {
        if (renderTarget !== null) fail('UNSUPPORTED', 'Explicit PMREM targets require an initialized renderer');
        if (!cubemap?.isCubeTexture) fail('SOURCE', 'Expected a source CubeTexture');
        const texture = new THREE.Texture();
        texture.name = 'PMREM.cubeUv'; texture.mapping = THREE.CubeUVReflectionMapping;
        texture.colorSpace = THREE.LinearSRGBColorSpace;
        texture.isF3DPMREMSource = true; texture.f3dPMREMSource = cubemap;
        return {isRenderTarget: true, texture, dispose() { texture.dispose(); }};
      }
      return this._fromTexture('fromCubemap', cubemap, renderTarget);
    }
    async fromCubemapAsync(...args) { return this.fromCubemap(...args); }
    compileCubemapShader() { return this._generator.compileCubemapShader(); }
    compileEquirectangularShader() { return this._generator.compileEquirectangularShader(); }
    _fromTexture(method, texture, target) {
      const renderer = this._renderer, previous = PMREM_CAPTURES.get(renderer);
      let input = null;
      if (!texture?.isRenderTargetTexture) {
        input = renderer._capturePMREMTexture(texture); texture = input.texture;
      }
      PMREM_CAPTURES.set(renderer, {generator: this._generator, input});
      try { return this._generator[method](texture, target); }
      finally {
        if (previous) PMREM_CAPTURES.set(renderer, previous); else PMREM_CAPTURES.delete(renderer);
        if (input) renderer.waitForGPU().then(input.release, input.release).catch(error => { renderer._deferredError ??= error; });
      }
    }
    dispose() { this._generator.dispose(); }
  }
  return PMREMGenerator;
}
