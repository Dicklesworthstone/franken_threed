/**
 * @file routing_adapter.test.mjs
 * Comprehensive unit test suite for RendererConstructionRouter (Bead f3d-04.4).
 *
 * Exercises:
 * - Positive route selection (WebGL -> exact, WebGPU -> retained/specialized, CSS2D -> retained).
 * - Opaque GL escape detection forcing exact-backend synchronously.
 * - Host capability fallback when WebGPU is unavailable.
 * - Connected group propagation over shared resources.
 * - Irreversible route lock rejection on same canvas.
 * - Single-execution invariant (no duplicate constructor calls or side effects).
 * - Report generation with no-claim attestations.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ExecutionRoute,
  EscapeReason,
  RouteLockError,
  ConnectedCompatibilityGroups,
  decideRendererRoute,
  RendererConstructionRouter,
  generateRouteReport,
} from '../../../tools/compat/index.mjs';

test('Positive: WebGLRenderer routes synchronously to EXACT_BACKEND', () => {
  const router = new RendererConstructionRouter();
  let calls = 0;

  class FakeWebGLRenderer {
    constructor(opts) {
      calls++;
      this.isWebGLRenderer = true;
      this.canvas = opts.canvas;
    }
  }

  const instance = router.routeAndConstruct({
    constructorFn: FakeWebGLRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'canvas-1' },
    sourceSpan: 'src/main.js:10:5',
  });

  assert.equal(calls, 1, 'Constructor must execute exactly once');
  assert.equal(instance.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));
  assert.equal(instance.canvas, 'canvas-1');
});

test('Positive: Opaque GL escapes force WebGPURenderer to EXACT_BACKEND synchronously', () => {
  const router = new RendererConstructionRouter();

  class FakeWebGPURenderer {
    constructor(opts) {
      this.isWebGPURenderer = true;
    }
  }

  const instance = router.routeAndConstruct({
    constructorFn: FakeWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-2' },
    analysis: { hasOpaqueGLEscapes: true },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: 'src/app.js:25:3',
  });

  assert.equal(instance.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));
});

test('Positive: Native context access forces EXACT_BACKEND', () => {
  const router = new RendererConstructionRouter();

  class MockRenderer {}

  const instance = router.routeAndConstruct({
    constructorFn: MockRenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-3' },
    analysis: { hasNativeContextAccess: true },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: 'src/custom.js:40:9',
  });

  assert.equal(instance.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.NATIVE_CONTEXT_ACCESS));
});

test('Positive: Host without WebGPU falls back to EXACT_BACKEND with HOST_LIMITATION_FALLBACK', () => {
  const router = new RendererConstructionRouter();

  class FakeWebGPURenderer {}

  const instance = router.routeAndConstruct({
    constructorFn: FakeWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-4' },
    hostCapabilities: { hasWebGPU: false, hasWebGL: true },
    sourceSpan: 'src/fallback.js:12:1',
  });

  assert.equal(instance.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.HOST_LIMITATION_FALLBACK));
});

test('Positive: Non-GPU renderers route to RETAINED_UPSTREAM', () => {
  const router = new RendererConstructionRouter();

  for (const name of ['CSS2DRenderer', 'CSS3DRenderer', 'SVGRenderer']) {
    class FakeRenderer {}
    const instance = router.routeAndConstruct({
      constructorFn: FakeRenderer,
      constructorName: name,
      options: {},
    });
    assert.equal(instance.__f3d_route__, ExecutionRoute.RETAINED_UPSTREAM);
  }
});

test('Positive: Independent canvases maintain distinct route decisions and epochs', () => {
  const router = new RendererConstructionRouter();

  class GLRenderer {}
  class GPURenderer {}

  const inst1 = router.routeAndConstruct({
    constructorFn: GLRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'canvas-A' },
  });

  const inst2 = router.routeAndConstruct({
    constructorFn: GPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-B' },
    hostCapabilities: { hasWebGPU: true },
  });

  assert.equal(inst1.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.equal(inst2.__f3d_route__, ExecutionRoute.RETAINED_UPSTREAM);
  assert.notEqual(inst1.__f3d_decision__.canvas, inst2.__f3d_decision__.canvas);
  assert.notEqual(inst1.__f3d_decision__.rendererId, inst2.__f3d_decision__.rendererId);
});

test('Positive: Connected groups propagate EXACT_BACKEND when sharing mutable resources', () => {
  const router = new RendererConstructionRouter();

  class RendererA {}
  class RendererB {}

  // Renderer A is standard WebGPU
  const instA = router.routeAndConstruct({
    constructorFn: RendererA,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-shared-1' },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: ['render-target-001'],
  });

  // Renderer B has opaque GL escape and shares render-target-001
  const instB = router.routeAndConstruct({
    constructorFn: RendererB,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-shared-2' },
    analysis: { hasOpaqueGLEscapes: true },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: ['render-target-001'],
  });

  assert.equal(instB.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.equal(instA.__f3d_group_id__, instB.__f3d_group_id__, 'Renderers must share connected group ID');
});

test('Negative: Irreversible route lock rejects late route switch on same canvas before side effects', () => {
  const router = new RendererConstructionRouter();
  let glCalls = 0;
  let gpuCalls = 0;

  class GLRenderer {
    constructor() { glCalls++; }
  }
  class GPURenderer {
    constructor() { gpuCalls++; }
  }

  // First lock canvas to EXACT_BACKEND
  router.routeAndConstruct({
    constructorFn: GLRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'locked-canvas' },
    sourceSpan: 'src/init.js:5:1',
  });

  assert.equal(glCalls, 1);

  // Attempt to re-route same canvas to WebGPU must throw RouteLockError
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: GPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'locked-canvas' },
        hostCapabilities: { hasWebGPU: true },
        sourceSpan: 'src/switch.js:15:3',
      });
    },
    (err) => {
      assert.ok(err instanceof RouteLockError);
      assert.equal(err.canvasId, 'locked-canvas');
      assert.equal(err.existingRoute, ExecutionRoute.EXACT_BACKEND);
      assert.equal(err.requestedRoute, ExecutionRoute.RETAINED_UPSTREAM);
      assert.equal(err.sourceSpan, 'src/switch.js:15:3');
      return true;
    },
    'Re-routing locked canvas must throw RouteLockError with source span'
  );

  assert.equal(gpuCalls, 0, 'Side effect / second constructor must NOT have executed');
});

test('Single-execution invariant: constructor executes exactly once', () => {
  const router = new RendererConstructionRouter();
  let executionCount = 0;

  class MonitoredRenderer {
    constructor() {
      executionCount++;
    }
  }

  router.routeAndConstruct({
    constructorFn: MonitoredRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'single-exec-canvas' },
  });

  assert.equal(executionCount, 1, 'Constructor must have executed exactly 1 time');
});

test('Route report formatting includes decisions and no-claim attestations', () => {
  const router = new RendererConstructionRouter();
  class TestRenderer {}

  router.routeAndConstruct({
    constructorFn: TestRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'report-canvas' },
    sourceSpan: 'src/app.js:1:1',
  });

  const report = generateRouteReport(router);
  assert.equal(report.total_renderers, 1);
  assert.equal(report.route_counts[ExecutionRoute.EXACT_BACKEND], 1);
  assert.ok(report.no_claim_attestation.exact_backend.includes('never credited as acceleration'));
});
