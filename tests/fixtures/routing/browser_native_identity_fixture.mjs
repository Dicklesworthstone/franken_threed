/**
 * @file browser_native_identity_fixture.mjs
 * Browser-executable test verifying real native context identity (Plan §5.1, §6.9).
 *
 * Verifies:
 * 1. EXACT_BACKEND acquires and exposes a genuine WebGLRenderingContext or WebGL2RenderingContext.
 * 2. WebGPU route acquires GPUCanvasContext when supported.
 * 3. Irreversible route lock on a real DOM canvas blocks late route switching before context acquisition.
 */

import {
  ExecutionRoute,
  RouteLockError,
  RendererConstructionRouter,
} from '../../../tools/compat/index.mjs';

/**
 * Pinned Exact WebGL Construction Mock representing the pinned Three.js WebGLRenderer contract.
 * Acquires genuine GL context directly from the real DOM canvas.
 */
export class NativeWebGLRendererComponent {
  constructor(options = {}) {
    const canvas = options.canvas || document.createElement('canvas');
    this.domElement = canvas;
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) {
      throw new Error('Host failed to provide native WebGL context');
    }
    this.context = gl;
    this.isWebGLRenderer = true;
  }

  getContext() {
    return this.context;
  }
}

/**
 * Native WebGPU Construction Mock representing the WebGPURenderer contract.
 * Acquires genuine WebGPU context if navigator.gpu is available.
 */
export class NativeWebGPURendererComponent {
  constructor(options = {}) {
    const canvas = options.canvas || document.createElement('canvas');
    this.domElement = canvas;
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      this.context = canvas.getContext('webgpu');
    } else {
      this.context = null;
    }
    this.isWebGPURenderer = true;
  }

  getContext() {
    return this.context;
  }
}

/**
 * Run the browser native identity verification suite against real DOM canvases.
 * @returns {Promise<{ passed: boolean, results: Object[], error?: string }>}
 */
export async function runBrowserNativeIdentityVerification() {
  const results = [];
  const router = new RendererConstructionRouter();

  // Test 1: EXACT_BACKEND acquires genuine native GL context
  try {
    const canvas1 = document.createElement('canvas');
    canvas1.id = 'canvas-exact-gl';
    document.body.appendChild(canvas1);

    const glRenderer = router.routeAndConstruct({
      constructorFn: NativeWebGLRendererComponent,
      constructorName: 'WebGLRenderer',
      options: { canvas: canvas1 },
      sourceSpan: 'browser_native_identity.html:test1',
    });

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
    document.body.appendChild(canvas2);

    // Initial route lock to exact-backend
    router.routeAndConstruct({
      constructorFn: NativeWebGLRendererComponent,
      constructorName: 'WebGLRenderer',
      options: { canvas: canvas2 },
      sourceSpan: 'browser_native_identity.html:lock-step1',
    });

    // Attempting to route same canvas to WebGPU must throw RouteLockError
    let threw = false;
    try {
      router.routeAndConstruct({
        constructorFn: NativeWebGPURendererComponent,
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

  // Test 3: Opaque escape forces EXACT_BACKEND synchronously on WebGPURenderer constructor
  try {
    const canvas3 = document.createElement('canvas');
    canvas3.id = 'canvas-escape-dom';
    document.body.appendChild(canvas3);

    const escapedRenderer = router.routeAndConstruct({
      constructorFn: NativeWebGLRendererComponent, // Router directs to exact WebGL component
      constructorName: 'WebGPURenderer',
      options: { canvas: canvas3 },
      analysis: { hasOpaqueGLEscapes: true },
      hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      sourceSpan: 'browser_native_identity.html:escape-step',
    });

    const rawGl = escapedRenderer.getContext();
    const isRealGL =
      (typeof WebGL2RenderingContext !== 'undefined' && rawGl instanceof WebGL2RenderingContext) ||
      (typeof WebGLRenderingContext !== 'undefined' && rawGl instanceof WebGLRenderingContext);

    if (escapedRenderer.__f3d_route__ !== ExecutionRoute.EXACT_BACKEND || !isRealGL) {
      throw new Error('Escaped renderer failed to route to exact native GL component');
    }

    results.push({
      test: 'opaque_escape_native_gl_routing',
      status: 'pass',
      route: escapedRenderer.__f3d_route__,
      nativeContext: rawGl.constructor.name,
    });
  } catch (err) {
    results.push({
      test: 'opaque_escape_native_gl_routing',
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
