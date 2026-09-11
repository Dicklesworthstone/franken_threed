/**
 * @file routing_adapter.test.mjs
 * Comprehensive unit test suite for RendererConstructionRouter (Bead f3d-04.4).
 *
 * Exercises:
 * - Positive route selection (WebGL -> exact, WebGPU -> retained/specialized, CSS2D -> retained).
 * - Opaque GL escape detection forcing exact-backend synchronously with actual implementation dispatch.
 * - Host capability fallback when WebGPU is unavailable.
 * - Connected group propagation over shared resources.
 * - Irreversible route lock rejection on same canvas.
 * - Single-execution invariant (no duplicate constructor calls or side effects).
 * - Defect 1 regression: constructor exception propagation and single invocation.
 * - Defect 2 regression: actual implementation dispatch rather than decorative labeling.
 * - Defect 3 regression: absence of public reset() and permanent canvas locking.
 * - Defect 4 regression: upfront TypeError on non-function constructor before locking.
 * - Remaining correctness: preflight lock reservation, lock retention on throw, reentrancy rejection.
 * - Native object shape preservation on sealed and frozen instances via external WeakMap diagnostics.
 * - Connected groups cross-construction-order validation and residency conflict detection.
 * - Resource mutability in recordResourceSharing: immutable asset sharing keeps independent routes, mutable render targets couple residency.
 * - Module graph JSON consumption (schema 1.0.0) from RubyCrane for real H1 and H2 entries.
 * - forceWebGL literal evaluation: literal true forces EXACT_BACKEND, literal false preserves WebGPU.
 * - Regression: real H1 bundle (!api.webgpu) records UNRESOLVED_FORCE_WEBGL and defers to runtime construction.
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
  extractGraphRoutingFacts,
  evaluateModuleGraphRoutes,
  RendererConstructionRouter,
  getRendererRoute,
  getRendererDecision,
  generateRouteReport,
  formatRouteReport,
  PinnedWebGLRenderer,
  ExactWebGLRenderer,
  registerExactBackend,
  createExactBackendRouter,
  createExactWebGLRenderer,
} from '../../../tools/compat/index.mjs';

import { buildModuleGraph } from '../../../tools/ingest/index.mjs';

class AdmittedWebGLRenderer {
  constructor(opts = {}) {
    this.isWebGLRenderer = true;
    this.isExactBackend = true;
    this.canvas = opts.canvas;
  }
}

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
  assert.equal(router.getInstanceRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.equal(getRendererRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));
  assert.equal(instance.canvas, 'canvas-1');
});

test('Positive: Opaque GL escapes force WebGPURenderer to EXACT_BACKEND synchronously with implementation dispatch', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

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

  assert.ok(instance instanceof AdmittedWebGLRenderer, 'Must instantiate the admitted exact backend implementation');
  assert.equal(instance.isExactBackend, true);
  assert.equal(instance.isWebGPURenderer, undefined, 'Must not instantiate the escaped WebGPURenderer constructor');
  assert.equal(instance.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));
});

test('Positive: Native context access forces EXACT_BACKEND with implementation dispatch', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class MockRenderer {}

  const instance = router.routeAndConstruct({
    constructorFn: MockRenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-3' },
    analysis: { hasNativeContextAccess: true },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: 'src/custom.js:40:9',
  });

  assert.ok(instance instanceof AdmittedWebGLRenderer);
  assert.equal(instance.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(instance.__f3d_decision__.reasons.includes(EscapeReason.NATIVE_CONTEXT_ACCESS));
});

test('Positive: Host without WebGPU falls back to EXACT_BACKEND with HOST_LIMITATION_FALLBACK', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class FakeWebGPURenderer {}

  const instance = router.routeAndConstruct({
    constructorFn: FakeWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-4' },
    hostCapabilities: { hasWebGPU: false, hasWebGL: true },
    sourceSpan: 'src/fallback.js:12:1',
  });

  assert.ok(instance instanceof AdmittedWebGLRenderer);
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

test('Positive: Connected groups propagate EXACT_BACKEND when sharing mutable resources (Order: Exact first)', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class RendererExact {}
  class RendererStandard {}

  // Renderer 1 has opaque GL escape and shares render-target-001 (mutable)
  const inst1 = router.routeAndConstruct({
    constructorFn: RendererExact,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-shared-exact' },
    analysis: { hasOpaqueGLEscapes: true },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: ['render-target-001'],
  });

  // Renderer 2 joins the same mutable resource group and must propagate EXACT_BACKEND
  const inst2 = router.routeAndConstruct({
    constructorFn: RendererStandard,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-shared-standard' },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: ['render-target-001'],
  });

  assert.equal(inst1.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.equal(inst2.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.ok(inst2 instanceof AdmittedWebGLRenderer);
  assert.equal(inst1.__f3d_group_id__, inst2.__f3d_group_id__, 'Renderers must share connected group ID');
});

test('Negative: Connected groups reject residency conflict when non-exact renderer is already committed (Order: Non-exact first)', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class RendererNonExact {}
  class RendererExactSecond {}

  // Renderer 1 commits to RETAINED_UPSTREAM on canvas-group-1 with mutable render target
  router.routeAndConstruct({
    constructorFn: RendererNonExact,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-group-1' },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: ['shared-rt-hazard'],
  });

  // Renderer 2 joins same mutable resource but requires EXACT_BACKEND -> residency hazard must throw
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: RendererExactSecond,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-group-2' },
        analysis: { hasOpaqueGLEscapes: true },
        hostCapabilities: { hasWebGPU: true },
        sharedResources: ['shared-rt-hazard'],
      });
    },
    (err) => {
      assert.match(err.message, /Connected group conflict/);
      assert.match(err.message, /already committed to route 'retained-upstream'/);
      return true;
    }
  );
});

test('Positive: Two renderers sharing only an immutable asset keep independent routes', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class RendererWebGPU {}
  class RendererWebGL {}

  // Renderer 1 on canvas-imm-1 is WebGPU, sharing an immutable read-only texture
  const inst1 = router.routeAndConstruct({
    constructorFn: RendererWebGPU,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-imm-1' },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: [{ id: 'static-skybox-tex', isMutable: false }],
  });

  // Renderer 2 on canvas-imm-2 has opaque GL escape, sharing the same immutable texture
  const inst2 = router.routeAndConstruct({
    constructorFn: RendererWebGL,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-imm-2' },
    analysis: { hasOpaqueGLEscapes: true },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: [{ id: 'static-skybox-tex', isMutable: false }],
  });

  // Both renderers maintain independent routes without conflict because immutable assets upload independently
  assert.equal(inst1.__f3d_route__, ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(inst2.__f3d_route__, ExecutionRoute.EXACT_BACKEND);
  assert.notEqual(inst1.__f3d_group_id__, inst2.__f3d_group_id__, 'Immutable asset sharing must NOT couple connected groups');
});

test('Negative: Two renderers sharing a mutable render target couple backend residency', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class RendererWebGPU {}
  class RendererWebGL {}

  // Renderer 1 commits to WebGPU on canvas-mut-1 with explicit mutable resource
  router.routeAndConstruct({
    constructorFn: RendererWebGPU,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-mut-1' },
    hostCapabilities: { hasWebGPU: true },
    sharedResources: [{ id: 'mutable-rt-hazard', isMutable: true }],
  });

  // Renderer 2 requires EXACT_BACKEND sharing the same mutable resource -> must throw residency conflict
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: RendererWebGL,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-mut-2' },
        analysis: { hasOpaqueGLEscapes: true },
        hostCapabilities: { hasWebGPU: true },
        sharedResources: [{ id: 'mutable-rt-hazard', isMutable: true }],
      });
    },
    (err) => {
      assert.match(err.message, /Connected group conflict/);
      assert.match(err.message, /already committed to route 'retained-upstream'/);
      return true;
    }
  );
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

test('Defect 1 regression: constructor increments side-effect counter and throws, proving single invocation and preserved error identity', () => {
  const router = new RendererConstructionRouter();
  let sideEffects = 0;

  class CustomTestError extends Error {
    constructor(msg) {
      super(msg);
      this.name = 'CustomTestError';
    }
  }

  class FailingRenderer {
    constructor() {
      sideEffects++;
      throw new CustomTestError('Planned constructor failure');
    }
  }

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: FailingRenderer,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'failing-canvas' },
      });
    },
    (err) => {
      assert.ok(err instanceof CustomTestError, 'Error identity must be preserved');
      assert.equal(err.message, 'Planned constructor failure');
      return true;
    }
  );

  assert.equal(sideEffects, 1, 'Constructor must be invoked exactly once, never caught and retried');
  // Canvas lock must be RETAINED after throw to prevent illegal re-binding
  assert.equal(router.getCanvasLock('failing-canvas')?.route, ExecutionRoute.EXACT_BACKEND);
});

test('Binds-then-throws regression: canvas lock is retained after constructor acquires context then throws', () => {
  const router = new RendererConstructionRouter();
  let contextBound = false;

  class ContextBindingErrorRenderer {
    constructor(opts) {
      contextBound = true; // Context acquisition happened
      throw new Error('Shader compilation error after context acquisition');
    }
  }

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: ContextBindingErrorRenderer,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'bound-then-failed-canvas' },
      });
    },
    /Shader compilation error/
  );

  assert.equal(contextBound, true);
  // Canvas lock is retained permanently: cannot be re-routed to WebGPU
  assert.equal(router.getCanvasLock('bound-then-failed-canvas')?.route, ExecutionRoute.EXACT_BACKEND);

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: class OtherRenderer {},
        constructorName: 'WebGPURenderer',
        options: { canvas: 'bound-then-failed-canvas' },
        hostCapabilities: { hasWebGPU: true },
      });
    },
    RouteLockError
  );
});

test('Reentrant constructor regression: reentrant route switch on same canvas is rejected before side effects', () => {
  const router = new RendererConstructionRouter();

  class ReentrantRenderer {
    constructor(opts) {
      // Reentrancy attempt: tries to construct a WebGPU renderer on the same canvas
      router.routeAndConstruct({
        constructorFn: class InnerRenderer {},
        constructorName: 'WebGPURenderer',
        options: { canvas: opts.canvas },
        hostCapabilities: { hasWebGPU: true },
      });
    }
  }

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: ReentrantRenderer,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'reentrant-canvas' },
      });
    },
    (err) => {
      assert.ok(err instanceof RouteLockError);
      assert.equal(err.canvasId, 'reentrant-canvas');
      assert.equal(err.existingRoute, ExecutionRoute.EXACT_BACKEND);
      assert.equal(err.requestedRoute, ExecutionRoute.RETAINED_UPSTREAM);
      return true;
    }
  );
});

test('Preserve native object shape: sealed and frozen instances succeed and maintain route diagnostics via WeakMap', () => {
  const router = new RendererConstructionRouter();

  class SealedRenderer {
    constructor() {
      this.isCustom = true;
      Object.seal(this);
    }
  }

  class FrozenRenderer {
    constructor() {
      this.isCustom = true;
      Object.freeze(this);
    }
  }

  // Sealed instance construction must NOT throw
  const sealedInstance = router.routeAndConstruct({
    constructorFn: SealedRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'sealed-canvas' },
  });

  assert.ok(Object.isSealed(sealedInstance));
  assert.equal(router.getInstanceRoute(sealedInstance), ExecutionRoute.EXACT_BACKEND);
  assert.equal(getRendererRoute(sealedInstance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(router.getInstanceDecision(sealedInstance));

  // Frozen instance construction must NOT throw
  const frozenInstance = router.routeAndConstruct({
    constructorFn: FrozenRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'frozen-canvas' },
  });

  assert.ok(Object.isFrozen(frozenInstance));
  assert.equal(router.getInstanceRoute(frozenInstance), ExecutionRoute.EXACT_BACKEND);
  assert.equal(getRendererRoute(frozenInstance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(getRendererDecision(frozenInstance));
});

test('Defect 2 regression: escaped WebGPU without admitted exact backend implementation throws instead of fake claim', () => {
  const router = new RendererConstructionRouter(); // No implementations registered

  class SomeWebGPURenderer {}

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: SomeWebGPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'escaped-no-impl-canvas' },
        analysis: { hasOpaqueGLEscapes: true },
      });
    },
    (err) => {
      assert.match(err.message, /no admitted exact backend implementation registered/);
      return true;
    }
  );
});

test('Defect 3 regression: reset() method is removed and canvas locks are permanently irreversible', () => {
  const router = new RendererConstructionRouter();
  assert.equal(router.reset, undefined, 'reset() must not exist on router');

  class GLRenderer {}
  class GPURenderer {}

  router.routeAndConstruct({
    constructorFn: GLRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'locked-permanently' },
  });

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: GPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'locked-permanently' },
        hostCapabilities: { hasWebGPU: true },
      });
    },
    RouteLockError
  );
});

test('Defect 4 regression: non-function constructor throws TypeError before construction or canvas locking', () => {
  const router = new RendererConstructionRouter();

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: null,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'invalid-constructor-canvas' },
      });
    },
    (err) => {
      assert.ok(err instanceof TypeError);
      return true;
    }
  );

  assert.equal(
    router.getCanvasLock('invalid-constructor-canvas'),
    undefined,
    'Canvas must not be locked when constructor validation fails'
  );
});

test('Positive: Real H1 module graph JSON (schema 1.0.0) consumed directly by decideRendererRoute', async () => {
  const h1Bundle = await buildModuleGraph('upstream/three.js/examples/webgpu_performance_renderbundle.html');
  assert.equal(h1Bundle.schema_version, '1.0.0');

  // 1. Capable WebGPU host with admitted specialization -> SPECIALIZED_WEBGPU
  const specializedDecision = decideRendererRoute({
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(specializedDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(specializedDecision.constructorName, 'WebGPURenderer');
  assert.ok(specializedDecision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL));

  // 2. Capable WebGPU host without specialization -> RETAINED_UPSTREAM
  const retainedDecision = decideRendererRoute({
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: false,
  });
  assert.equal(retainedDecision.route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(retainedDecision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL));

  // 3. Host lacking WebGPU -> EXACT_BACKEND fallback
  const fallbackDecision = decideRendererRoute({
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: false, hasWebGL: true },
    specializationAvailable: false,
  });
  assert.equal(fallbackDecision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(fallbackDecision.reasons.includes(EscapeReason.HOST_LIMITATION_FALLBACK));
  assert.ok(fallbackDecision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL));
});

test('Positive: Real H2 module graph JSON (schema 1.0.0) consumed directly by decideRendererRoute', async () => {
  const h2Bundle = await buildModuleGraph('upstream/three.js/examples/webgl_marchingcubes.html');
  assert.equal(h2Bundle.schema_version, '1.0.0');

  const h2Decision = decideRendererRoute({
    analysis: h2Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(h2Decision.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(h2Decision.constructorName, 'WebGLRenderer');
  assert.ok(h2Decision.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));
});

test('Positive: evaluateModuleGraphRoutes maps all construction sites in module graph bundle', async () => {
  const h1Bundle = await buildModuleGraph('upstream/three.js/examples/webgpu_performance_renderbundle.html');
  const evaluatedSites = evaluateModuleGraphRoutes(h1Bundle, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });

  assert.equal(evaluatedSites.length, 1);
  assert.equal(evaluatedSites[0].constructorName, 'WebGPURenderer');
  assert.equal(evaluatedSites[0].decision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(evaluatedSites[0].decision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL));
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

  // Extend with one unresolved case
  class UnresolvedRenderer {}
  router.routeAndConstruct({
    constructorFn: UnresolvedRenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'report-unresolved-canvas', forceWebGL: 'unresolved' },
    sourceSpan: 'src/app.js:20:1',
  });

  const report = generateRouteReport(router);
  assert.equal(report.total_renderers, 2);
  assert.equal(report.unresolved_decisions, 1);
  assert.equal(report.route_counts[ExecutionRoute.EXACT_BACKEND], 1);
  assert.equal(report.route_counts[ExecutionRoute.RETAINED_UPSTREAM], 1);
  assert.ok(report.no_claim_attestation.exact_backend.includes('never credited as acceleration'));
  assert.ok(report.no_claim_attestation.retained_upstream.includes('not a Rust rewrite'));
  assert.ok(report.no_claim_attestation.unresolved_facts.includes('runtime validation'));

  const formatted = formatRouteReport(report);
  assert.ok(formatted.includes('Total Renderers: 2'));
  assert.ok(formatted.includes('Unresolved Decisions: 1'));
  assert.ok(formatted.includes(report.no_claim_attestation.exact_backend));
  assert.ok(formatted.includes(report.no_claim_attestation.retained_upstream));
  assert.ok(formatted.includes(report.no_claim_attestation.unresolved_facts));
});

test('Positive: hasUnresolvedContextAccess records UNRESOLVED_NATIVE_CONTEXT_ACCESS reason', () => {
  const decision = decideRendererRoute({
    constructorName: 'WebGPURenderer',
    options: { canvas: 'unresolved-ctx-canvas' },
    analysis: { hasNativeContextAccess: true, hasUnresolvedContextAccess: true },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(decision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(decision.reasons.includes(EscapeReason.UNRESOLVED_NATIVE_CONTEXT_ACCESS));
  assert.ok(!decision.reasons.includes(EscapeReason.NATIVE_CONTEXT_ACCESS));

  // Resolved native context access still uses standard NATIVE_CONTEXT_ACCESS
  const resolvedDecision = decideRendererRoute({
    constructorName: 'WebGPURenderer',
    options: { canvas: 'resolved-ctx-canvas' },
    analysis: { hasNativeContextAccess: true, hasUnresolvedContextAccess: false },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(resolvedDecision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(resolvedDecision.reasons.includes(EscapeReason.NATIVE_CONTEXT_ACCESS));
  assert.ok(!resolvedDecision.reasons.includes(EscapeReason.UNRESOLVED_NATIVE_CONTEXT_ACCESS));
});

test('Positive: forceWebGL literal true forces EXACT_BACKEND, literal false allows WebGPU', () => {
  // 1. Literal forceWebGL: true must force EXACT_BACKEND with EXPLICIT_SOURCE_SELECTION
  const forcedDecision = decideRendererRoute({
    constructorName: 'WebGPURenderer',
    options: { forceWebGL: true },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(forcedDecision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(forcedDecision.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));

  // 2. Literal forceWebGL: false must allow SPECIALIZED_WEBGPU when specialization is available
  const unforcedDecision = decideRendererRoute({
    constructorName: 'WebGPURenderer',
    options: { forceWebGL: false },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(unforcedDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(!unforcedDecision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL));
});

test('Regression: Real H1 bundle with non-literal forceWebGL (! api.webgpu) records UNRESOLVED_FORCE_WEBGL and defers to runtime', async () => {
  // Ingest real H1 source
  const h1Bundle = await buildModuleGraph('upstream/three.js/examples/webgpu_performance_renderbundle.html');
  const graphFacts = extractGraphRoutingFacts(h1Bundle);
  assert.equal(graphFacts.constructionSites.length, 1);
  const site = graphFacts.constructionSites[0];

  // 1. Verify AST extraction preserved unresolved non-literal without coercing to false or true
  assert.equal(site.options.forceWebGL, 'unresolved', 'AST extraction must classify !api.webgpu as "unresolved"');
  assert.equal(site.options.forceWebGLUnresolved, true);

  // 2. Static decision must record EscapeReason.UNRESOLVED_FORCE_WEBGL and NOT coerce to true (EXACT_BACKEND)
  const staticDecision = decideRendererRoute({
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(staticDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(
    staticDecision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL),
    'Decision must include UNRESOLVED_FORCE_WEBGL reason'
  );

  // 3. Passing options.forceWebGL = "unresolved" directly also records UNRESOLVED_FORCE_WEBGL without coercing
  const directUnresolvedDecision = decideRendererRoute({
    constructorName: 'WebGPURenderer',
    options: { forceWebGL: 'unresolved' },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(directUnresolvedDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(directUnresolvedDecision.reasons.includes(EscapeReason.UNRESOLVED_FORCE_WEBGL));

  // 4. Runtime construction router defers to runtime value:
  // When runtime option evaluates to forceWebGL: true (e.g. ?backend=webgl), routes to EXACT_BACKEND
  class MockWebGPURenderer {
    constructor(opts = {}) {
      this.isWebGPURenderer = true;
      this.opts = opts;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
      [ExecutionRoute.SPECIALIZED_WEBGPU]: MockWebGPURenderer,
    },
  });

  const runtimeForcedInstance = router.routeAndConstruct({
    constructorFn: MockWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-runtime-forced', forceWebGL: true },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(getRendererRoute(runtimeForcedInstance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(runtimeForcedInstance instanceof AdmittedWebGLRenderer);

  // When runtime option evaluates to forceWebGL: false (default H1 without ?backend=webgl), routes to SPECIALIZED_WEBGPU
  const runtimeUnforcedInstance = router.routeAndConstruct({
    constructorFn: MockWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-runtime-unforced', forceWebGL: false },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(getRendererRoute(runtimeUnforcedInstance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(runtimeUnforcedInstance.isWebGPURenderer, true);
});

test('Positive: exact_backend component exports pinned WebGLRenderer and registers with construction router', () => {
  assert.equal(typeof PinnedWebGLRenderer, 'function');
  assert.equal(PinnedWebGLRenderer.name, 'WebGLRenderer');
  assert.equal(ExactWebGLRenderer, PinnedWebGLRenderer);

  // Registration on existing router
  const customRouter = new RendererConstructionRouter();
  assert.equal(customRouter.implementations[ExecutionRoute.EXACT_BACKEND], undefined);
  registerExactBackend(customRouter);
  assert.equal(customRouter.implementations[ExecutionRoute.EXACT_BACKEND], PinnedWebGLRenderer);

  // Invalid router throws TypeError
  assert.throws(() => registerExactBackend(null), /TypeError/);
  assert.throws(() => registerExactBackend({}), /TypeError/);

  // Factory creation
  const exactRouter = createExactBackendRouter();
  assert.equal(exactRouter.implementations[ExecutionRoute.EXACT_BACKEND], PinnedWebGLRenderer);

  // Config propagation
  const configuredRouter = createExactBackendRouter({ specializationAvailable: true });
  assert.equal(configuredRouter.specializationAvailable, true);
  assert.equal(configuredRouter.implementations[ExecutionRoute.EXACT_BACKEND], PinnedWebGLRenderer);
});

test('Regression: caller-supplied constructorFn is invoked exactly once and not shadowed by registered implementations', () => {
  class AdmittedBackendRenderer {
    constructor(opts) {
      this.isAdmitted = true;
      this.canvas = opts.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedBackendRenderer,
    },
  });

  let callerCalls = 0;
  class CallerSuppliedRenderer {
    constructor(opts) {
      callerCalls++;
      this.isCallerSupplied = true;
      this.canvas = opts.canvas;
    }
  }

  // 1. Explicitly supplied constructorFn matching resolved route executes exactly once
  const instance = router.routeAndConstruct({
    constructorFn: CallerSuppliedRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas: 'canvas-caller-supplied' },
    sourceSpan: 'src/caller.js:1:1',
  });

  assert.equal(callerCalls, 1, 'Caller-supplied constructorFn must execute exactly once');
  assert.equal(instance.isCallerSupplied, true, 'Instance must be from caller constructorFn, not substituted implementation');
  assert.equal(instance.isAdmitted, undefined, 'Registered implementation must not shadow caller constructorFn');
  assert.ok(instance instanceof CallerSuppliedRenderer);
  assert.equal(getRendererRoute(instance), ExecutionRoute.EXACT_BACKEND);

  // 2. When constructorFn is absent, registered implementation is used
  const absentConstructorInstance = router.routeAndConstruct({
    constructorName: 'WebGLRenderer',
    options: { canvas: 'canvas-absent-constructor' },
    sourceSpan: 'src/absent.js:1:1',
  });

  assert.ok(absentConstructorInstance instanceof AdmittedBackendRenderer, 'Registered implementation must be used when constructorFn is absent');
  assert.equal(absentConstructorInstance.isAdmitted, true);
  assert.equal(getRendererRoute(absentConstructorInstance), ExecutionRoute.EXACT_BACKEND);

  // 3. Single invocation, error identity, and canvas lock retention when caller-supplied constructor throws
  class PlannedError extends Error {
    constructor(msg) {
      super(msg);
      this.name = 'PlannedError';
    }
  }

  let throwCalls = 0;
  class ThrowingCallerRenderer {
    constructor() {
      throwCalls++;
      throw new PlannedError('Caller constructor planned failure');
    }
  }

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: ThrowingCallerRenderer,
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-throwing-caller' },
        sourceSpan: 'src/throwing.js:1:1',
      });
    },
    (err) => {
      assert.ok(err instanceof PlannedError, 'Caller constructor error identity must be preserved');
      assert.equal(err.message, 'Caller constructor planned failure');
      return true;
    }
  );

  assert.equal(throwCalls, 1, 'Throwing caller constructor must be invoked exactly once');
  assert.equal(
    router.getCanvasLock('canvas-throwing-caller')?.route,
    ExecutionRoute.EXACT_BACKEND,
    'Canvas lock must be retained after caller constructor throws'
  );
});

test('Positive: decision log records H1 branches, reasons, group membership, and rejected offending spans; attribution log tracks submissions', async () => {
  const h1Bundle = await buildModuleGraph('upstream/three.js/examples/webgpu_performance_renderbundle.html');

  class MockRenderableWebGLRenderer {
    constructor(opts = {}) {
      this.isWebGLRenderer = true;
      this.canvas = opts.canvas;
      this.renderCalls = 0;
    }
    render(scene, camera) {
      this.renderCalls++;
      return { scene, camera, target: 'gl' };
    }
  }

  class MockRenderableWebGPURenderer {
    constructor(opts = {}) {
      this.isWebGPURenderer = true;
      this.canvas = opts.canvas;
      this.renderCalls = 0;
    }
    render(scene, camera) {
      this.renderCalls++;
      return { scene, camera, target: 'gpu' };
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: MockRenderableWebGLRenderer,
      [ExecutionRoute.SPECIALIZED_WEBGPU]: MockRenderableWebGPURenderer,
    },
  });

  // 1. Initial decision and attribution logs must be empty
  assert.deepEqual(router.getDecisionLog(), []);
  assert.deepEqual(router.getAttributionLog(), []);

  // 2. H1 forceWebGL branch (EXACT_BACKEND)
  const forcedSpan = 'examples/webgpu_performance_renderbundle.html:85:3';
  const forcedInstance = router.routeAndConstruct({
    constructorFn: MockRenderableWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-h1-forced', forceWebGL: true },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: forcedSpan,
  });

  assert.equal(getRendererRoute(forcedInstance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(forcedInstance instanceof MockRenderableWebGLRenderer);

  // Check decision log after H1 forceWebGL construction
  const decisionLog1 = router.getDecisionLog();
  assert.equal(decisionLog1.length, 1);
  const forcedDecision = decisionLog1[0];
  assert.equal(forcedDecision.site, 'WebGPURenderer');
  assert.equal(forcedDecision.span, forcedSpan);
  assert.equal(forcedDecision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(Array.isArray(forcedDecision.reasons));
  assert.ok(
    forcedDecision.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION),
    'H1 forceWebGL branch must record EXPLICIT_SOURCE_SELECTION in reasons'
  );
  assert.ok(typeof forcedDecision.group === 'string' && forcedDecision.group.length > 0, 'Decision must include group membership');

  // 3. H1 WebGPU branch (SPECIALIZED_WEBGPU)
  const unforcedSpan = 'examples/webgpu_performance_renderbundle.html:95:3';
  const unforcedInstance = router.routeAndConstruct({
    constructorFn: MockRenderableWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-h1-unforced', forceWebGL: false },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: unforcedSpan,
  });

  assert.equal(getRendererRoute(unforcedInstance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(unforcedInstance instanceof MockRenderableWebGPURenderer);

  // Check decision log after H1 unforced WebGPU construction
  const decisionLog2 = router.getDecisionLog();
  assert.equal(decisionLog2.length, 2);
  const unforcedDecision = decisionLog2[1];
  assert.equal(unforcedDecision.site, 'WebGPURenderer');
  assert.equal(unforcedDecision.span, unforcedSpan);
  assert.equal(unforcedDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(Array.isArray(unforcedDecision.reasons));
  assert.ok(typeof unforcedDecision.group === 'string' && unforcedDecision.group.length > 0, 'Decision must include group membership');

  // 4. Rejected re-route on locked canvas must log the offending span
  const offendingSpan = 'src/offender_component.js:142:7';
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: MockRenderableWebGPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-h1-forced', forceWebGL: false }, // canvas already locked to EXACT_BACKEND
        analysis: h1Bundle,
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
        sourceSpan: offendingSpan,
      });
    },
    (err) => {
      assert.ok(err instanceof RouteLockError);
      assert.equal(err.sourceSpan, offendingSpan);
      return true;
    }
  );

  // Decision log must record the rejected re-route and its offending span
  const decisionLog3 = router.getDecisionLog();
  assert.equal(decisionLog3.length, 3, 'Decision log must record rejected re-route attempt');
  const rejectedDecision = decisionLog3[2];
  assert.equal(rejectedDecision.site, 'WebGPURenderer');
  assert.equal(rejectedDecision.span, offendingSpan, 'Rejected decision must record the offending source span');
  assert.equal(rejectedDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(Array.isArray(rejectedDecision.reasons));
  assert.ok(typeof rejectedDecision.group === 'string' && rejectedDecision.group.length > 0);

  // 5. Runtime attribution log: hook render call once and count submissions
  assert.deepEqual(router.getAttributionLog(), [], 'Attribution log must be empty before any render call');

  // First render on forcedInstance (EXACT_BACKEND)
  const res1 = forcedInstance.render('sceneA', 'cameraA');
  assert.deepEqual(res1, { scene: 'sceneA', camera: 'cameraA', target: 'gl' });
  assert.equal(forcedInstance.renderCalls, 1);

  const attrLog1 = router.getAttributionLog();
  assert.equal(attrLog1.length, 1);
  assert.deepEqual(attrLog1[0], {
    renderer: forcedInstance.__f3d_renderer_id__,
    route: ExecutionRoute.EXACT_BACKEND,
    submissions: 1,
  });

  // Second render on forcedInstance
  forcedInstance.render('sceneB', 'cameraB');
  assert.equal(forcedInstance.renderCalls, 2);

  const attrLog2 = router.getAttributionLog();
  assert.equal(attrLog2.length, 2);
  assert.deepEqual(attrLog2[1], {
    renderer: forcedInstance.__f3d_renderer_id__,
    route: ExecutionRoute.EXACT_BACKEND,
    submissions: 2,
  });

  // Render on unforcedInstance (SPECIALIZED_WEBGPU)
  const resGPU = unforcedInstance.render('sceneGPU', 'cameraGPU');
  assert.deepEqual(resGPU, { scene: 'sceneGPU', camera: 'cameraGPU', target: 'gpu' });
  assert.equal(unforcedInstance.renderCalls, 1);

  const attrLog3 = router.getAttributionLog();
  assert.equal(attrLog3.length, 3);
  assert.deepEqual(attrLog3[2], {
    renderer: unforcedInstance.__f3d_renderer_id__,
    route: ExecutionRoute.SPECIALIZED_WEBGPU,
    submissions: 1,
  });
});



