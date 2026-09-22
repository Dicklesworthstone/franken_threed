/**
 * @file browser_native_identity_fixture.mjs
 * Browser-executable test verifying real native context identity and implementation
 * dispatch using the pinned upstream Three.js r186 component (Plan §5.1, §6.9).
 *
 * Verifies:
 * 1. EXACT_BACKEND dispatches to actual pinned WebGLRenderer and acquires genuine WebGL(2)RenderingContext.
 * 2. Irreversible route lock on a real DOM canvas blocks late route switching before context acquisition.
 * 3. Opaque escape preserves the pinned WebGPURenderer class, native backend and context behavior.
 * 4. Defect 1 regression: single invocation and error identity preservation on constructor throw.
 * 5. Defect 3 regression: absence of public reset() method.
 * 6. Defect 4 regression: upfront TypeError on non-function constructor before locking.
 * 7. Binds-then-throws regression: canvas lock is retained on throw to prevent illegal re-binding.
 * 8. Reentrant constructor regression: reentrant route switch on same canvas rejected with RouteLockError.
 * 9. Native object shape preservation on sealed and frozen instances via external WeakMap diagnostics.
 * 10. Connected groups residency conflict rejection when non-exact renderer is already committed.
 * 11. Exact-backend component: pinned WebGLRenderer constructed through router with real WebGL2 context on DOM canvas (f3d-04.3).
 * 12. Actual native WebGL context loss and restore: WEBGL_lose_context lifecycle, positive rendering, event observation, prototype & source equivalence, negative control.
 */

import {
  createExactBackendRouter,
  createExactWebGLRenderer,
  ExactWebGLRenderer,
  ExecutionRoute,
  getRendererRoute,
  RendererConstructionRouter,
  RouteLockError,
  registerExactBackend,
} from "../../../tools/compat/index.mjs";

import {
  BoxGeometry,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  WebGLRenderer as PinnedWebGLRenderer,
  Scene,
} from "../../../upstream/three.js/build/three.module.js";
import { WebGPURenderer as PinnedWebGPURenderer } from "../../../upstream/three.js/build/three.webgpu.js";

export { ExactWebGLRenderer, PinnedWebGLRenderer, PinnedWebGPURenderer };

/**
 * Helper: Wait for the native 'webglcontextlost' event on a canvas with timeout.
 * Calls event.preventDefault() to allow subsequent context restoration.
 */
function waitForContextLost(canvas, timeoutMs = 4000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      canvas.removeEventListener("webglcontextlost", onLost, false);
      reject(new Error(`Timed out waiting for 'webglcontextlost' after ${timeoutMs}ms`));
    }, timeoutMs);
    function onLost(event) {
      clearTimeout(timer);
      event.preventDefault(); // mandatory per WebGL spec to permit restoration
      canvas.removeEventListener("webglcontextlost", onLost, false);
      resolve(event);
    }
    canvas.addEventListener("webglcontextlost", onLost, false);
  });
}

/**
 * Helper: Wait for the native 'webglcontextrestored' event on a canvas with timeout.
 */
function waitForContextRestored(canvas, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      canvas.removeEventListener("webglcontextrestored", onRestored, false);
      reject(
        new Error(
          `Timed out waiting for 'webglcontextrestored' on ${canvas.id} after ${timeoutMs}ms`,
        ),
      );
    }, timeoutMs);
    function onRestored(event) {
      clearTimeout(timer);
      canvas.removeEventListener("webglcontextrestored", onRestored, false);
      resolve(event);
    }
    canvas.addEventListener("webglcontextrestored", onRestored, false);
  });
}

/**
 * Run the browser native identity verification suite against real DOM canvases
 * using actual pinned Three.js r186 components.
 * @returns {Promise<{ passed: boolean, results: Object[], error?: string }>}
 */
export async function runBrowserNativeIdentityVerification() {
  const results = [];

  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: PinnedWebGLRenderer,
      [ExecutionRoute.RETAINED_UPSTREAM]: {
        WebGPURenderer: PinnedWebGPURenderer,
        default: PinnedWebGPURenderer,
      },
    },
  });

  // Test 1: EXACT_BACKEND dispatches to pinned WebGLRenderer and acquires genuine native GL context
  try {
    const canvas1 = document.createElement("canvas");
    canvas1.id = "canvas-exact-gl";
    canvas1.width = 64;
    canvas1.height = 64;
    document.body.appendChild(canvas1);

    const glRenderer = router.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: "WebGLRenderer",
      options: { canvas: canvas1 },
      sourceSpan: "browser_native_identity.html:test1",
    });

    if (!(glRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error("Constructed instance is not an instance of pinned Three.js WebGLRenderer");
    }

    if (!glRenderer.isWebGLRenderer) {
      throw new Error("Renderer instance missing isWebGLRenderer property");
    }

    const rawGl = glRenderer.getContext();
    const isRealGL =
      (typeof WebGL2RenderingContext !== "undefined" && rawGl instanceof WebGL2RenderingContext) ||
      (typeof WebGLRenderingContext !== "undefined" && rawGl instanceof WebGLRenderingContext);

    if (!isRealGL) {
      throw new Error("Acquired context is not an instance of native WebGL(2)RenderingContext");
    }

    results.push({
      test: "exact_native_gl_identity",
      status: "pass",
      route: getRendererRoute(glRenderer),
      implementation: "PinnedWebGLRenderer",
      nativeContext: rawGl.constructor.name,
    });
  } catch (err) {
    results.push({
      test: "exact_native_gl_identity",
      status: "fail",
      error: err.message,
    });
  }

  // Test 2: Irreversible route lock on real DOM canvas rejects second backend
  try {
    const canvas2 = document.createElement("canvas");
    canvas2.id = "canvas-locked-dom";
    canvas2.width = 64;
    canvas2.height = 64;
    document.body.appendChild(canvas2);

    // Initial route lock to exact-backend
    router.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: "WebGLRenderer",
      options: { canvas: canvas2 },
      sourceSpan: "browser_native_identity.html:lock-step1",
    });

    // Attempting to route same canvas to WebGPU must throw RouteLockError before side effects
    let threw = false;
    try {
      router.routeAndConstruct({
        constructorFn: PinnedWebGPURenderer,
        constructorName: "WebGPURenderer",
        options: { canvas: canvas2 },
        hostCapabilities: { hasWebGPU: true },
        sourceSpan: "browser_native_identity.html:lock-step2",
      });
    } catch (err) {
      if (err instanceof RouteLockError) {
        threw = true;
      } else {
        throw err;
      }
    }

    if (!threw) {
      throw new Error("Late route switch on bound DOM canvas did NOT throw RouteLockError");
    }

    results.push({
      test: "canvas_irreversibility_real_dom",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "canvas_irreversibility_real_dom",
      status: "fail",
      error: err.message,
    });
  }

  // Test 3: Compare opaque construction with the unmodified source in both backend modes.
  try {
    const branches = [];
    for (const forceWebGL of [false, true]) {
      const sourceCanvas = document.createElement("canvas");
      const candidateCanvas = document.createElement("canvas");
      sourceCanvas.width = candidateCanvas.width = 64;
      sourceCanvas.height = candidateCanvas.height = 64;
      const source = new PinnedWebGPURenderer({ canvas: sourceCanvas, forceWebGL });
      const candidate = router.routeAndConstruct({
        constructorFn: PinnedWebGPURenderer,
        constructorName: "WebGPURenderer",
        options: { canvas: candidateCanvas, forceWebGL },
        analysis: { hasOpaqueGLEscapes: true },
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
        sourceSpan: "browser_native_identity.html:escape-step",
      });
      if (
        candidate.constructor !== source.constructor ||
        Object.getPrototypeOf(candidate) !== Object.getPrototypeOf(source) ||
        !(candidate instanceof PinnedWebGPURenderer) ||
        candidate.isWebGLRenderer
      ) {
        throw new Error("Opaque escape substituted a different source renderer class");
      }
      if (
        candidate.backend.constructor !== source.backend.constructor ||
        candidate.getContext !== source.getContext
      ) {
        throw new Error("Opaque escape changed the source backend or native context method");
      }
      // Upstream can throw before init; preserve that behavior too.
      const beforeInit = (renderer) => {
        try {
          return { value: renderer.getContext()?.constructor.name };
        } catch (error) {
          return { error: error.name, message: error.message };
        }
      };
      if (JSON.stringify(beforeInit(candidate)) !== JSON.stringify(beforeInit(source))) {
        throw new Error("Opaque escape changed pre-init context behavior");
      }
      await source.init();
      await candidate.init();
      try {
        const referenceContext = source.getContext();
        const context = candidate.getContext();
        const nativeContext = forceWebGL
          ? context instanceof WebGL2RenderingContext
          : (typeof GPUCanvasContext !== "undefined" && context instanceof GPUCanvasContext) ||
            context instanceof WebGL2RenderingContext;
        if (
          !nativeContext ||
          context === referenceContext ||
          Object.getPrototypeOf(context) !== Object.getPrototypeOf(referenceContext) ||
          candidate.backend.constructor !== source.backend.constructor
        ) {
          throw new Error("Opaque escape did not preserve the source native backend/context");
        }
        branches.push({
          forceWebGL,
          backend: candidate.backend.constructor.name,
          nativeContext: context.constructor.name,
          route: getRendererRoute(candidate),
        });
      } finally {
        candidate.dispose();
        source.dispose();
      }
    }
    results.push({
      test: "opaque_escape_preserves_source_backend",
      status: "pass",
      implementation: "PinnedWebGPURenderer",
      branches,
    });
  } catch (err) {
    results.push({
      test: "opaque_escape_preserves_source_backend",
      status: "fail",
      error: err.message,
    });
  }

  // Test 4: Defect 1 regression - Constructor single invocation & error preservation
  try {
    const canvasFail = document.createElement("canvas");
    canvasFail.id = "canvas-fail-dom";
    canvasFail.width = 64;
    canvasFail.height = 64;
    document.body.appendChild(canvasFail);

    let callCount = 0;
    class DomCustomError extends Error {
      constructor(m) {
        super(m);
        this.name = "DomCustomError";
      }
    }
    class FailingConstructor {
      constructor() {
        callCount++;
        throw new DomCustomError("Planned DOM failure");
      }
    }

    let caughtError = null;
    try {
      router.routeAndConstruct({
        constructorFn: FailingConstructor,
        constructorName: "WebGLRenderer",
        options: { canvas: canvasFail },
      });
    } catch (e) {
      caughtError = e;
    }

    if (!(caughtError instanceof DomCustomError) || callCount !== 1) {
      throw new Error(
        `Expected single call and DomCustomError; got calls=${callCount}, error=${caughtError}`,
      );
    }

    results.push({
      test: "constructor_single_invocation_and_error_preservation",
      status: "pass",
      calls: callCount,
    });
  } catch (err) {
    results.push({
      test: "constructor_single_invocation_and_error_preservation",
      status: "fail",
      error: err.message,
    });
  }

  // Test 5: Defect 3 regression - Public reset() method is removed
  try {
    if (typeof router.reset !== "undefined") {
      throw new Error("Public reset() method must be undefined");
    }
    results.push({
      test: "reset_method_removed",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "reset_method_removed",
      status: "fail",
      error: err.message,
    });
  }

  // Test 6: Defect 4 regression - Invalid constructor throws TypeError before construction
  try {
    const canvasInvalid = document.createElement("canvas");
    canvasInvalid.id = "canvas-invalid-dom";
    canvasInvalid.width = 64;
    canvasInvalid.height = 64;
    document.body.appendChild(canvasInvalid);

    let threwType = false;
    try {
      router.routeAndConstruct({
        constructorFn: null,
        constructorName: "WebGLRenderer",
        options: { canvas: canvasInvalid },
      });
    } catch (e) {
      if (e instanceof TypeError) threwType = true;
    }

    if (!threwType) {
      throw new Error("Invalid constructor did not throw TypeError");
    }

    results.push({
      test: "invalid_constructor_rejected_before_lock",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "invalid_constructor_rejected_before_lock",
      status: "fail",
      error: err.message,
    });
  }

  // Test 7: Binds-then-throws regression - Canvas lock is retained after context binding failure
  try {
    const canvasFail = document.createElement("canvas");
    canvasFail.id = "canvas-dom-bind-fail";
    canvasFail.width = 64;
    canvasFail.height = 64;
    document.body.appendChild(canvasFail);

    class DomBindFailRenderer {
      constructor(opts) {
        opts.canvas.getContext("webgl"); // acquire real native context
        throw new Error("Planned context failure after binding");
      }
    }

    let threwFail = false;
    try {
      router.routeAndConstruct({
        constructorFn: DomBindFailRenderer,
        constructorName: "WebGLRenderer",
        options: { canvas: canvasFail },
      });
    } catch {
      threwFail = true;
    }

    if (!threwFail) throw new Error("Constructor did not throw expected error");

    // Attempting to re-route same canvas to WebGPU must throw RouteLockError
    let threwLock = false;
    try {
      router.routeAndConstruct({
        constructorFn: PinnedWebGPURenderer,
        constructorName: "WebGPURenderer",
        options: { canvas: canvasFail },
        hostCapabilities: { hasWebGPU: true },
      });
    } catch (e) {
      if (e instanceof RouteLockError) threwLock = true;
    }

    if (!threwLock) throw new Error("Canvas lock was not retained after constructor throw");

    results.push({
      test: "binds_then_throws_lock_retained",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "binds_then_throws_lock_retained",
      status: "fail",
      error: err.message,
    });
  }

  // Test 8: Reentrant constructor call on same canvas is rejected
  try {
    const canvasReentrant = document.createElement("canvas");
    canvasReentrant.id = "canvas-dom-reentrant";
    canvasReentrant.width = 64;
    canvasReentrant.height = 64;
    document.body.appendChild(canvasReentrant);

    class ReentrantDomRenderer {
      constructor(opts) {
        router.routeAndConstruct({
          constructorFn: PinnedWebGPURenderer,
          constructorName: "WebGPURenderer",
          options: { canvas: opts.canvas },
          hostCapabilities: { hasWebGPU: true },
        });
      }
    }

    let threwReentrantLock = false;
    try {
      router.routeAndConstruct({
        constructorFn: ReentrantDomRenderer,
        constructorName: "WebGLRenderer",
        options: { canvas: canvasReentrant },
      });
    } catch (e) {
      if (e instanceof RouteLockError) threwReentrantLock = true;
    }

    if (!threwReentrantLock) throw new Error("Reentrant call did not trigger RouteLockError");

    results.push({
      test: "reentrant_route_switch_rejected",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "reentrant_route_switch_rejected",
      status: "fail",
      error: err.message,
    });
  }

  // Test 9: Sealed and frozen instances succeed without breaking native shape
  try {
    const canvasSealed = document.createElement("canvas");
    canvasSealed.id = "canvas-dom-sealed";
    canvasSealed.width = 64;
    canvasSealed.height = 64;
    document.body.appendChild(canvasSealed);

    class SealedDomRenderer {
      constructor() {
        this.isSealedRenderer = true;
        Object.seal(this);
      }
    }

    const sealedInstance = router.routeAndConstruct({
      constructorFn: SealedDomRenderer,
      constructorName: "WebGLRenderer",
      options: { canvas: canvasSealed },
    });

    if (!Object.isSealed(sealedInstance)) throw new Error("Instance is not sealed");
    if (getRendererRoute(sealedInstance) !== ExecutionRoute.EXACT_BACKEND) {
      throw new Error("Failed to retrieve route via external diagnostics");
    }

    results.push({
      test: "sealed_instance_preserves_shape",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "sealed_instance_preserves_shape",
      status: "fail",
      error: err.message,
    });
  }

  // Test 10: Connected groups residency hazard rejection
  try {
    const canvasGrp1 = document.createElement("canvas");
    canvasGrp1.id = "canvas-grp-dom-1";
    canvasGrp1.width = 64;
    canvasGrp1.height = 64;
    document.body.appendChild(canvasGrp1);

    const canvasGrp2 = document.createElement("canvas");
    canvasGrp2.id = "canvas-grp-dom-2";
    canvasGrp2.width = 64;
    canvasGrp2.height = 64;
    document.body.appendChild(canvasGrp2);

    const groupRouter = new RendererConstructionRouter({
      implementations: {
        [ExecutionRoute.EXACT_BACKEND]: PinnedWebGLRenderer,
        [ExecutionRoute.RETAINED_UPSTREAM]: {
          WebGPURenderer: PinnedWebGPURenderer,
          default: PinnedWebGPURenderer,
        },
      },
    });

    // Renderer 1 commits to WebGPU route on shared resource
    groupRouter.routeAndConstruct({
      constructorFn: PinnedWebGPURenderer,
      constructorName: "WebGPURenderer",
      options: { canvas: canvasGrp1 },
      hostCapabilities: { hasWebGPU: true },
      sharedResources: ["dom-shared-rt"],
    });

    // Renderer 2 joins same resource but requires EXACT_BACKEND -> must throw residency conflict
    let threwConflict = false;
    try {
      groupRouter.routeAndConstruct({
        constructorFn: PinnedWebGLRenderer,
        constructorName: "WebGPURenderer",
        options: { canvas: canvasGrp2 },
        analysis: { hasOpaqueGLEscapes: true },
        hostCapabilities: { hasWebGPU: true },
        sharedResources: ["dom-shared-rt"],
      });
    } catch (e) {
      if (e.message.includes("Connected group conflict")) threwConflict = true;
    }

    if (!threwConflict) throw new Error("Connected group residency conflict was not thrown");

    results.push({
      test: "connected_groups_residency_hazard_rejected",
      status: "pass",
    });
  } catch (err) {
    results.push({
      test: "connected_groups_residency_hazard_rejected",
      status: "fail",
      error: err.message,
    });
  }

  // Test 11: Exact-backend component constructs through router and asserts pinned class with real WebGL2 context
  let exactRouter = null;
  try {
    const canvas11 = document.createElement("canvas");
    canvas11.id = "canvas-exact-backend-module";
    canvas11.width = 64;
    canvas11.height = 64;
    document.body.appendChild(canvas11);

    // Construct through the registered exact backend router
    exactRouter = createExactBackendRouter();
    const glRenderer = exactRouter.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: "WebGLRenderer",
      options: { canvas: canvas11 },
      sourceSpan: "browser_native_identity.html:test11",
    });

    if (!(glRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error("Constructed instance is not an instance of pinned Three.js WebGLRenderer");
    }

    if (glRenderer.isWebGLRenderer !== true) {
      throw new Error("Renderer instance missing isWebGLRenderer property");
    }

    // Verify native context identity and attributes verbatim
    const rawGl = glRenderer.getContext();
    if (!rawGl) {
      throw new Error("glRenderer.getContext() returned null or undefined");
    }

    const isRealWebGL2 =
      typeof WebGL2RenderingContext !== "undefined" && rawGl instanceof WebGL2RenderingContext;
    const isRealWebGL =
      typeof WebGLRenderingContext !== "undefined" && rawGl instanceof WebGLRenderingContext;

    if (!isRealWebGL2 && !isRealWebGL) {
      throw new Error("Acquired context is not an instance of native WebGL(2)RenderingContext");
    }

    if (rawGl.canvas !== canvas11) {
      throw new Error("Native context canvas does not match DOM canvas");
    }

    const attrs = glRenderer.getContextAttributes();
    if (!attrs || typeof attrs !== "object") {
      throw new Error("getContextAttributes() failed or returned non-object");
    }

    // Also verify direct helper entry point createExactWebGLRenderer
    const canvas11b = document.createElement("canvas");
    canvas11b.id = "canvas-exact-helper-module";
    canvas11b.width = 64;
    canvas11b.height = 64;
    document.body.appendChild(canvas11b);

    const helperRenderer = createExactWebGLRenderer({ canvas: canvas11b }, exactRouter);
    if (!(helperRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error(
        "createExactWebGLRenderer returned instance not matching pinned WebGLRenderer class",
      );
    }

    const helperGl = helperRenderer.getContext();
    if (!helperGl || helperGl.canvas !== canvas11b) {
      throw new Error("Helper renderer GL context canvas mismatch");
    }

    results.push({
      test: "exact_backend_router_pinned_webgl2_identity",
      status: "pass",
      route: getRendererRoute(glRenderer),
      implementation: "PinnedWebGLRenderer",
      nativeContext: rawGl.constructor.name,
      isWebGL2: isRealWebGL2,
    });
  } catch (err) {
    results.push({
      test: "exact_backend_router_pinned_webgl2_identity",
      status: "fail",
      error: err.message,
    });
  }

  // Test 12: Native WebGL context loss and restore (WEBGL_lose_context) across pinned source and routed WebGLRenderer
  try {
    const canvas12Ref = document.createElement("canvas");
    canvas12Ref.id = "canvas-context-loss-ref";
    canvas12Ref.width = 64;
    canvas12Ref.height = 64;
    document.body.appendChild(canvas12Ref);

    const canvas12Cand = document.createElement("canvas");
    canvas12Cand.id = "canvas-context-loss-cand";
    canvas12Cand.width = 64;
    canvas12Cand.height = 64;
    document.body.appendChild(canvas12Cand);

    // 1. Reference: pinned source WebGLRenderer
    const refRenderer = new PinnedWebGLRenderer({ canvas: canvas12Ref });
    const glRef = refRenderer.getContext();

    // 2. Candidate: routed exact-backend WebGLRenderer
    const lossRouter = exactRouter || createExactBackendRouter();
    const candRenderer = lossRouter.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: "WebGLRenderer",
      options: { canvas: canvas12Cand },
      sourceSpan: "browser_native_identity.html:test12",
    });
    const glCand = candRenderer.getContext();

    // Contract verification helper
    function assertExactBackendContract(renderer, glContext, label) {
      if (!(renderer instanceof PinnedWebGLRenderer)) {
        throw new Error(`${label}: instance is not an instance of pinned Three.js WebGLRenderer`);
      }
      if (renderer.isWebGLRenderer !== true) {
        throw new Error(`${label}: instance missing isWebGLRenderer property`);
      }
      if (!glContext) {
        throw new Error(`${label}: getContext() returned null or undefined`);
      }
      const isNativeGL =
        (typeof WebGL2RenderingContext !== "undefined" &&
          glContext instanceof WebGL2RenderingContext) ||
        (typeof WebGLRenderingContext !== "undefined" &&
          glContext instanceof WebGLRenderingContext);
      if (!isNativeGL) {
        throw new Error(
          `${label}: acquired context is not an instance of native WebGL(2)RenderingContext`,
        );
      }
    }

    assertExactBackendContract(refRenderer, glRef, "Reference");
    assertExactBackendContract(candRenderer, glCand, "Candidate");

    // 3. Query native WEBGL_lose_context extension
    const extRef = glRef.getExtension("WEBGL_lose_context");
    const extCand = glCand.getExtension("WEBGL_lose_context");

    if (!extRef || !extCand) {
      results.push({
        test: "exact_backend_native_context_loss_restore",
        status: "host-blocked",
        referenceBlocked: !extRef,
        candidateBlocked: !extCand,
        reason: "WEBGL_lose_context extension unavailable on host WebGL context",
      });
    } else {
      // 4. Render positive pixels before loss (clear red and draw a green mesh)
      refRenderer.setClearColor(0xff0000, 1.0);
      refRenderer.clear();
      candRenderer.setClearColor(0xff0000, 1.0);
      candRenderer.clear();

      const refPixPre = new Uint8Array(4);
      glRef.readPixels(0, 0, 1, 1, glRef.RGBA, glRef.UNSIGNED_BYTE, refPixPre);
      const candPixPre = new Uint8Array(4);
      glCand.readPixels(0, 0, 1, 1, glCand.RGBA, glCand.UNSIGNED_BYTE, candPixPre);

      // Helper: Assert exact 4-channel RGBA equality against expected values and between ref and cand
      function assertRgbaMatch(actual, expected, label) {
        for (let i = 0; i < 4; i++) {
          if (actual[i] !== expected[i]) {
            throw new Error(
              `${label}: channel ${i} expected ${expected[i]}, got ${actual[i]} (actual=[${Array.from(actual).join(",")}], expected=[${expected.join(",")}])`,
            );
          }
        }
      }

      assertRgbaMatch(refPixPre, [255, 0, 0, 255], "Pre-loss reference clear red");
      assertRgbaMatch(candPixPre, [255, 0, 0, 255], "Pre-loss candidate clear red");

      // Render a scene with green mesh on both
      const sceneRef = new Scene();
      const sceneCand = new Scene();
      const cameraRef = new PerspectiveCamera(45, 1, 0.1, 10);
      cameraRef.position.z = 2;
      const cameraCand = new PerspectiveCamera(45, 1, 0.1, 10);
      cameraCand.position.z = 2;
      const geomRef = new BoxGeometry(1, 1, 1);
      const geomCand = new BoxGeometry(1, 1, 1);
      const matRef = new MeshBasicMaterial({ color: 0x00ff00 });
      const matCand = new MeshBasicMaterial({ color: 0x00ff00 });
      sceneRef.add(new Mesh(geomRef, matRef));
      sceneCand.add(new Mesh(geomCand, matCand));

      refRenderer.render(sceneRef, cameraRef);
      candRenderer.render(sceneCand, cameraCand);

      const refCenterPre = new Uint8Array(4);
      glRef.readPixels(32, 32, 1, 1, glRef.RGBA, glRef.UNSIGNED_BYTE, refCenterPre);
      const candCenterPre = new Uint8Array(4);
      glCand.readPixels(32, 32, 1, 1, glCand.RGBA, glCand.UNSIGNED_BYTE, candCenterPre);

      assertRgbaMatch(refCenterPre, [0, 255, 0, 255], "Pre-loss reference mesh green");
      assertRgbaMatch(candCenterPre, [0, 255, 0, 255], "Pre-loss candidate mesh green");

      // 5. Trigger real WEBGL_lose_context.loseContext() and observe native webglcontextlost event
      const refLostPromise = waitForContextLost(canvas12Ref);
      const candLostPromise = waitForContextLost(canvas12Cand);

      extRef.loseContext();
      extCand.loseContext();

      await Promise.all([refLostPromise, candLostPromise]);

      if (!glRef.isContextLost() || !glCand.isContextLost()) {
        throw new Error("gl.isContextLost() returned false after loseContext()");
      }

      // 6. Negative control:
      // (a) Un-restored state check: skipping restoration leaves context in lost state
      if (!glCand.isContextLost() || !glRef.isContextLost()) {
        throw new Error("Negative control failed: un-restored context reports not lost");
      }
      // (b) Substituting wrong class / surrogate without native WebGL prototype fails contract check
      let wrongClassRejected = false;
      try {
        assertExactBackendContract({ isWebGLRenderer: true }, {}, "SurrogateMock");
      } catch (_) {
        wrongClassRejected = true;
      }
      if (!wrongClassRejected) {
        throw new Error("Negative control failed: wrong class substitution was not rejected");
      }

      // Allow browser GPU process event loop to settle context loss before requesting restoration
      await new Promise((r) => setTimeout(r, 100));

      // 7. Restore context via ext.restoreContext() and observe native webglcontextrestored event
      const refRestorePromise = waitForContextRestored(canvas12Ref);
      const candRestorePromise = waitForContextRestored(canvas12Cand);

      extRef.restoreContext();
      extCand.restoreContext();

      await Promise.all([refRestorePromise, candRestorePromise]);

      if (glRef.isContextLost() || glCand.isContextLost()) {
        throw new Error("gl.isContextLost() returned true after restoreContext()");
      }

      // 8. Render again after restore: positive blue clear and yellow mesh
      refRenderer.setClearColor(0x0000ff, 1.0);
      refRenderer.clear();
      candRenderer.setClearColor(0x0000ff, 1.0);
      candRenderer.clear();

      const refPixPost = new Uint8Array(4);
      glRef.readPixels(0, 0, 1, 1, glRef.RGBA, glRef.UNSIGNED_BYTE, refPixPost);
      const candPixPost = new Uint8Array(4);
      glCand.readPixels(0, 0, 1, 1, glCand.RGBA, glCand.UNSIGNED_BYTE, candPixPost);

      assertRgbaMatch(refPixPost, [0, 0, 255, 255], "Post-restore reference clear blue");
      assertRgbaMatch(candPixPost, [0, 0, 255, 255], "Post-restore candidate clear blue");

      const scenePostRef = new Scene();
      const scenePostCand = new Scene();
      const geomPostRef = new BoxGeometry(1, 1, 1);
      const geomPostCand = new BoxGeometry(1, 1, 1);
      const matPostRef = new MeshBasicMaterial({ color: 0xffff00 });
      const matPostCand = new MeshBasicMaterial({ color: 0xffff00 });
      scenePostRef.add(new Mesh(geomPostRef, matPostRef));
      scenePostCand.add(new Mesh(geomPostCand, matPostCand));

      refRenderer.render(scenePostRef, cameraRef);
      candRenderer.render(scenePostCand, cameraCand);

      const refCenterPost = new Uint8Array(4);
      glRef.readPixels(32, 32, 1, 1, glRef.RGBA, glRef.UNSIGNED_BYTE, refCenterPost);
      const candCenterPost = new Uint8Array(4);
      glCand.readPixels(32, 32, 1, 1, glCand.RGBA, glCand.UNSIGNED_BYTE, candCenterPost);

      assertRgbaMatch(refCenterPost, [255, 255, 0, 255], "Post-restore reference mesh yellow");
      assertRgbaMatch(candCenterPost, [255, 255, 0, 255], "Post-restore candidate mesh yellow");

      const allPixelsMatch =
        refPixPre.every((v, i) => v === candPixPre[i]) &&
        refCenterPre.every((v, i) => v === candCenterPre[i]) &&
        refPixPost.every((v, i) => v === candPixPost[i]) &&
        refCenterPost.every((v, i) => v === candCenterPost[i]);

      if (!allPixelsMatch) {
        throw new Error(
          "Candidate and reference pixel bytes diverged across loss/restore checkpoints",
        );
      }

      // 9. Source equivalence assertions across loss and restore
      assertExactBackendContract(candRenderer, glCand, "Post-restore Candidate");
      assertExactBackendContract(refRenderer, glRef, "Post-restore Reference");

      if (Object.getPrototypeOf(candRenderer) !== Object.getPrototypeOf(refRenderer)) {
        throw new Error("Post-restore prototype mismatch between candidate and reference");
      }

      if (Object.getPrototypeOf(glCand) !== Object.getPrototypeOf(glRef)) {
        throw new Error("Post-restore native WebGL context prototype mismatch");
      }

      if (getRendererRoute(candRenderer) !== ExecutionRoute.EXACT_BACKEND) {
        throw new Error("Candidate lost EXACT_BACKEND route after loss and restore");
      }

      results.push({
        test: "exact_backend_native_context_loss_restore",
        status: "pass",
        route: getRendererRoute(candRenderer),
        implementation: "PinnedWebGLRenderer",
        nativeContext: glCand.constructor.name,
        preLossPixels: {
          clearRed: Array.from(candPixPre),
          meshGreen: Array.from(candCenterPre),
        },
        postRestorePixels: {
          clearBlue: Array.from(candPixPost),
          meshYellow: Array.from(candCenterPost),
        },
        events: {
          webglcontextlost: true,
          webglcontextrestored: true,
        },
        sourceEquivalence: {
          rendererPrototypeMatch: true,
          glContextPrototypeMatch: true,
          pixelByteMatch: allPixelsMatch,
        },
        negativeControls: {
          unrestoredContextLossDetected: true,
          wrongClassSubstitutionRejected: true,
        },
      });
    }
  } catch (err) {
    results.push({
      test: "exact_backend_native_context_loss_restore",
      status: "fail",
      error: err.message,
    });
  }

  const allPassed = results.every(
    (r) =>
      r.status === "pass" ||
      (r.status === "host-blocked" && r.referenceBlocked && r.candidateBlocked),
  );
  const decisionLog = [
    ...router.getDecisionLog(),
    ...(exactRouter ? exactRouter.getDecisionLog() : []),
  ];
  const attributionLog = [
    ...router.getAttributionLog(),
    ...(exactRouter ? exactRouter.getAttributionLog() : []),
  ];

  return {
    passed: allPassed,
    results,
    decision_log: decisionLog,
    attribution_log: attributionLog,
  };
}
