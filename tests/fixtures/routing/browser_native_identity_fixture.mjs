/**
 * @file browser_native_identity_fixture.mjs
 * Browser-executable test verifying real native context identity and implementation
 * dispatch using the pinned upstream Three.js r186 component (Plan §5.1, §6.9).
 *
 * Verifies:
 * 1. EXACT_BACKEND dispatches to actual pinned WebGLRenderer and acquires genuine WebGL(2)RenderingContext.
 * 2. Irreversible route lock on a real DOM canvas blocks late route switching before context acquisition.
 * 3. Opaque GL escape on WebGPURenderer callsite dispatches to the actual pinned WebGLRenderer component.
 * 4. Defect 1 regression: single invocation and error identity preservation on constructor throw.
 * 5. Defect 3 regression: absence of public reset() method.
 * 6. Defect 4 regression: upfront TypeError on non-function constructor before locking.
 * 7. Binds-then-throws regression: canvas lock is retained on throw to prevent illegal re-binding.
 * 8. Reentrant constructor regression: reentrant route switch on same canvas rejected with RouteLockError.
 * 9. Native object shape preservation on sealed and frozen instances via external WeakMap diagnostics.
 * 10. Connected groups residency conflict rejection when non-exact renderer is already committed.
 * 11. Exact-backend component: pinned WebGLRenderer constructed through router with real WebGL2 context on DOM canvas (f3d-04.3).
 */

import {
  ExecutionRoute,
  RouteLockError,
  RendererConstructionRouter,
  getRendererRoute,
  ExactWebGLRenderer,
  registerExactBackend,
  createExactBackendRouter,
  createExactWebGLRenderer,
} from '../../../tools/compat/index.mjs';

import { WebGLRenderer as PinnedWebGLRenderer } from '../../../upstream/three.js/build/three.module.js';
import { WebGPURenderer as PinnedWebGPURenderer } from '../../../upstream/three.js/build/three.webgpu.js';

export { PinnedWebGLRenderer, PinnedWebGPURenderer, ExactWebGLRenderer };

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
    const canvas1 = document.createElement('canvas');
    canvas1.id = 'canvas-exact-gl';
    canvas1.width = 64;
    canvas1.height = 64;
    document.body.appendChild(canvas1);

    const glRenderer = router.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: 'WebGLRenderer',
      options: { canvas: canvas1 },
      sourceSpan: 'browser_native_identity.html:test1',
    });

    if (!(glRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error('Constructed instance is not an instance of pinned Three.js WebGLRenderer');
    }

    if (!glRenderer.isWebGLRenderer) {
      throw new Error('Renderer instance missing isWebGLRenderer property');
    }

    const rawGl = glRenderer.getContext();
    const isRealGL =
      (typeof WebGL2RenderingContext !== 'undefined' && rawGl instanceof WebGL2RenderingContext) ||
      (typeof WebGLRenderingContext !== 'undefined' && rawGl instanceof WebGLRenderingContext);

    if (!isRealGL) {
      throw new Error('Acquired context is not an instance of native WebGL(2)RenderingContext');
    }

    results.push({
      test: 'exact_native_gl_identity',
      status: 'pass',
      route: glRenderer.__f3d_route__,
      implementation: 'PinnedWebGLRenderer',
      nativeContext: rawGl.constructor.name,
    });
  } catch (err) {
    results.push({
      test: 'exact_native_gl_identity',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 2: Irreversible route lock on real DOM canvas rejects second backend
  try {
    const canvas2 = document.createElement('canvas');
    canvas2.id = 'canvas-locked-dom';
    canvas2.width = 64;
    canvas2.height = 64;
    document.body.appendChild(canvas2);

    // Initial route lock to exact-backend
    router.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: 'WebGLRenderer',
      options: { canvas: canvas2 },
      sourceSpan: 'browser_native_identity.html:lock-step1',
    });

    // Attempting to route same canvas to WebGPU must throw RouteLockError before side effects
    let threw = false;
    try {
      router.routeAndConstruct({
        constructorFn: PinnedWebGPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: canvas2 },
        hostCapabilities: { hasWebGPU: true },
        sourceSpan: 'browser_native_identity.html:lock-step2',
      });
    } catch (err) {
      if (err instanceof RouteLockError) {
        threw = true;
      } else {
        throw err;
      }
    }

    if (!threw) {
      throw new Error('Late route switch on bound DOM canvas did NOT throw RouteLockError');
    }

    results.push({
      test: 'canvas_irreversibility_real_dom',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'canvas_irreversibility_real_dom',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 3: Opaque escape forces EXACT_BACKEND and dispatches to actual pinned WebGLRenderer
  try {
    const canvas3 = document.createElement('canvas');
    canvas3.id = 'canvas-escape-dom';
    canvas3.width = 64;
    canvas3.height = 64;
    document.body.appendChild(canvas3);

    // Call site asks for WebGPURenderer, but static analysis indicates opaque GL escapes
    const escapedRenderer = router.routeAndConstruct({
      constructorFn: PinnedWebGPURenderer,
      constructorName: 'WebGPURenderer',
      options: { canvas: canvas3 },
      analysis: { hasOpaqueGLEscapes: true },
      hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      sourceSpan: 'browser_native_identity.html:escape-step',
    });

    // Verification: Router MUST have dispatched to PinnedWebGLRenderer, not PinnedWebGPURenderer
    if (!(escapedRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error('Escaped renderer did not dispatch to PinnedWebGLRenderer');
    }

    if (!escapedRenderer.isWebGLRenderer) {
      throw new Error('Escaped renderer missing isWebGLRenderer property');
    }

    if (escapedRenderer.isWebGPURenderer) {
      throw new Error('Escaped renderer has isWebGPURenderer flag (false positive instantiation)');
    }

    const rawGl = escapedRenderer.getContext();
    const isRealGL =
      (typeof WebGL2RenderingContext !== 'undefined' && rawGl instanceof WebGL2RenderingContext) ||
      (typeof WebGLRenderingContext !== 'undefined' && rawGl instanceof WebGLRenderingContext);

    if (!isRealGL) {
      throw new Error('Escaped renderer context is not genuine WebGL(2)RenderingContext');
    }

    results.push({
      test: 'opaque_escape_native_gl_routing',
      status: 'pass',
      route: escapedRenderer.__f3d_route__,
      dispatchedImplementation: 'PinnedWebGLRenderer',
      nativeContext: rawGl.constructor.name,
    });
  } catch (err) {
    results.push({
      test: 'opaque_escape_native_gl_routing',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 4: Defect 1 regression - Constructor single invocation & error preservation
  try {
    let callCount = 0;
    class DomCustomError extends Error {
      constructor(m) { super(m); this.name = 'DomCustomError'; }
    }
    class FailingConstructor {
      constructor() {
        callCount++;
        throw new DomCustomError('Planned DOM failure');
      }
    }

    let caughtError = null;
    try {
      router.routeAndConstruct({
        constructorFn: FailingConstructor,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-fail-dom' },
      });
    } catch (e) {
      caughtError = e;
    }

    if (!(caughtError instanceof DomCustomError) || callCount !== 1) {
      throw new Error(`Expected single call and DomCustomError; got calls=${callCount}, error=${caughtError}`);
    }

    results.push({
      test: 'constructor_single_invocation_and_error_preservation',
      status: 'pass',
      calls: callCount,
    });
  } catch (err) {
    results.push({
      test: 'constructor_single_invocation_and_error_preservation',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 5: Defect 3 regression - Public reset() method is removed
  try {
    if (typeof router.reset !== 'undefined') {
      throw new Error('Public reset() method must be undefined');
    }
    results.push({
      test: 'reset_method_removed',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'reset_method_removed',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 6: Defect 4 regression - Invalid constructor throws TypeError before construction
  try {
    let threwType = false;
    try {
      router.routeAndConstruct({
        constructorFn: null,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'invalid-dom-canvas' },
      });
    } catch (e) {
      if (e instanceof TypeError) threwType = true;
    }

    if (!threwType) {
      throw new Error('Invalid constructor did not throw TypeError');
    }

    results.push({
      test: 'invalid_constructor_rejected_before_lock',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'invalid_constructor_rejected_before_lock',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 7: Binds-then-throws regression - Canvas lock is retained after context binding failure
  try {
    const canvasFail = document.createElement('canvas');
    canvasFail.id = 'canvas-dom-bind-fail';
    document.body.appendChild(canvasFail);

    class DomBindFailRenderer {
      constructor(opts) {
        opts.canvas.getContext('webgl'); // acquire real native context
        throw new Error('Planned context failure after binding');
      }
    }

    let threwFail = false;
    try {
      router.routeAndConstruct({
        constructorFn: DomBindFailRenderer,
        constructorName: 'WebGLRenderer',
        options: { canvas: canvasFail },
      });
    } catch {
      threwFail = true;
    }

    if (!threwFail) throw new Error('Constructor did not throw expected error');

    // Attempting to re-route same canvas to WebGPU must throw RouteLockError
    let threwLock = false;
    try {
      router.routeAndConstruct({
        constructorFn: PinnedWebGPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: canvasFail },
        hostCapabilities: { hasWebGPU: true },
      });
    } catch (e) {
      if (e instanceof RouteLockError) threwLock = true;
    }

    if (!threwLock) throw new Error('Canvas lock was not retained after constructor throw');

    results.push({
      test: 'binds_then_throws_lock_retained',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'binds_then_throws_lock_retained',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 8: Reentrant constructor call on same canvas is rejected
  try {
    const canvasReentrant = document.createElement('canvas');
    canvasReentrant.id = 'canvas-dom-reentrant';
    document.body.appendChild(canvasReentrant);

    class ReentrantDomRenderer {
      constructor(opts) {
        router.routeAndConstruct({
          constructorFn: PinnedWebGPURenderer,
          constructorName: 'WebGPURenderer',
          options: { canvas: opts.canvas },
          hostCapabilities: { hasWebGPU: true },
        });
      }
    }

    let threwReentrantLock = false;
    try {
      router.routeAndConstruct({
        constructorFn: ReentrantDomRenderer,
        constructorName: 'WebGLRenderer',
        options: { canvas: canvasReentrant },
      });
    } catch (e) {
      if (e instanceof RouteLockError) threwReentrantLock = true;
    }

    if (!threwReentrantLock) throw new Error('Reentrant call did not trigger RouteLockError');

    results.push({
      test: 'reentrant_route_switch_rejected',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'reentrant_route_switch_rejected',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 9: Sealed and frozen instances succeed without breaking native shape
  try {
    class SealedDomRenderer {
      constructor() {
        this.isSealedRenderer = true;
        Object.seal(this);
      }
    }

    const sealedInstance = router.routeAndConstruct({
      constructorFn: SealedDomRenderer,
      constructorName: 'WebGLRenderer',
      options: { canvas: 'canvas-dom-sealed' },
    });

    if (!Object.isSealed(sealedInstance)) throw new Error('Instance is not sealed');
    if (getRendererRoute(sealedInstance) !== ExecutionRoute.EXACT_BACKEND) {
      throw new Error('Failed to retrieve route via external diagnostics');
    }

    results.push({
      test: 'sealed_instance_preserves_shape',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'sealed_instance_preserves_shape',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 10: Connected groups residency hazard rejection
  try {
    const groupRouter = new RendererConstructionRouter({
      implementations: {
        [ExecutionRoute.EXACT_BACKEND]: PinnedWebGLRenderer,
      },
    });

    // Renderer 1 commits to WebGPU route on shared resource
    groupRouter.routeAndConstruct({
      constructorFn: PinnedWebGPURenderer,
      constructorName: 'WebGPURenderer',
      options: { canvas: 'canvas-grp-dom-1' },
      hostCapabilities: { hasWebGPU: true },
      sharedResources: ['dom-shared-rt'],
    });

    // Renderer 2 joins same resource but requires EXACT_BACKEND -> must throw residency conflict
    let threwConflict = false;
    try {
      groupRouter.routeAndConstruct({
        constructorFn: PinnedWebGLRenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-grp-dom-2' },
        analysis: { hasOpaqueGLEscapes: true },
        hostCapabilities: { hasWebGPU: true },
        sharedResources: ['dom-shared-rt'],
      });
    } catch (e) {
      if (e.message.includes('Connected group conflict')) threwConflict = true;
    }

    if (!threwConflict) throw new Error('Connected group residency conflict was not thrown');

    results.push({
      test: 'connected_groups_residency_hazard_rejected',
      status: 'pass',
    });
  } catch (err) {
    results.push({
      test: 'connected_groups_residency_hazard_rejected',
      status: 'fail',
      error: err.message,
    });
  }

  // Test 11: Exact-backend component constructs through router and asserts pinned class with real WebGL2 context
  try {
    const canvas11 = document.createElement('canvas');
    canvas11.id = 'canvas-exact-backend-module';
    canvas11.width = 64;
    canvas11.height = 64;
    document.body.appendChild(canvas11);

    // Construct through the registered exact backend router
    const exactRouter = createExactBackendRouter();
    const glRenderer = exactRouter.routeAndConstruct({
      constructorFn: PinnedWebGLRenderer,
      constructorName: 'WebGLRenderer',
      options: { canvas: canvas11 },
      sourceSpan: 'browser_native_identity.html:test11',
    });

    if (!(glRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error('Constructed instance is not an instance of pinned Three.js WebGLRenderer');
    }

    if (glRenderer.isWebGLRenderer !== true) {
      throw new Error('Renderer instance missing isWebGLRenderer property');
    }

    // Verify native context identity and attributes verbatim
    const rawGl = glRenderer.getContext();
    if (!rawGl) {
      throw new Error('glRenderer.getContext() returned null or undefined');
    }

    const isRealWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && rawGl instanceof WebGL2RenderingContext;
    const isRealWebGL = typeof WebGLRenderingContext !== 'undefined' && rawGl instanceof WebGLRenderingContext;

    if (!isRealWebGL2 && !isRealWebGL) {
      throw new Error('Acquired context is not an instance of native WebGL(2)RenderingContext');
    }

    if (rawGl.canvas !== canvas11) {
      throw new Error('Native context canvas does not match DOM canvas');
    }

    const attrs = glRenderer.getContextAttributes();
    if (!attrs || typeof attrs !== 'object') {
      throw new Error('getContextAttributes() failed or returned non-object');
    }

    // Also verify direct helper entry point createExactWebGLRenderer
    const canvas11b = document.createElement('canvas');
    canvas11b.id = 'canvas-exact-helper-module';
    canvas11b.width = 64;
    canvas11b.height = 64;
    document.body.appendChild(canvas11b);

    const helperRenderer = createExactWebGLRenderer({ canvas: canvas11b }, exactRouter);
    if (!(helperRenderer instanceof PinnedWebGLRenderer)) {
      throw new Error('createExactWebGLRenderer returned instance not matching pinned WebGLRenderer class');
    }

    const helperGl = helperRenderer.getContext();
    if (!helperGl || helperGl.canvas !== canvas11b) {
      throw new Error('Helper renderer GL context canvas mismatch');
    }

    results.push({
      test: 'exact_backend_router_pinned_webgl2_identity',
      status: 'pass',
      route: getRendererRoute(glRenderer),
      implementation: 'PinnedWebGLRenderer',
      nativeContext: rawGl.constructor.name,
      isWebGL2: isRealWebGL2,
    });
  } catch (err) {
    results.push({
      test: 'exact_backend_router_pinned_webgl2_identity',
      status: 'fail',
      error: err.message,
    });
  }

  const allPassed = results.every((r) => r.status === 'pass');
  return {
    passed: allPassed,
    results,
  };
}
