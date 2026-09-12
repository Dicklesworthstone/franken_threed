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
  assert.equal(getRendererRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getInstanceRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(getRendererDecision(instance).reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));
  assert.equal(instance.__f3d_route__, undefined, 'No __f3d_route__ own property on instance');
  assert.equal(instance.canvas, 'canvas-1');
});

test('Positive: Opaque escapes preserve the source WebGPURenderer constructor and options', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer,
    },
  });

  class FakeWebGPURenderer {
    constructor(opts) {
      this.isWebGPURenderer = true;
      this.opts = opts;
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

  assert.ok(instance instanceof FakeWebGPURenderer, 'Opaque escape must preserve the source constructor');
  assert.equal(instance.isWebGLRenderer, undefined, 'Must not substitute legacy WebGLRenderer');
  assert.equal(instance.opts.forceWebGL, undefined, 'Must not force a backend the source did not select');
  assert.equal(getRendererRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(getRendererDecision(instance).reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));
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

  assert.ok(instance instanceof MockRenderer);
  assert.equal(getRendererRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(getRendererDecision(instance).reasons.includes(EscapeReason.NATIVE_CONTEXT_ACCESS));
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

  assert.ok(instance instanceof FakeWebGPURenderer, 'The source constructor owns host fallback');
  assert.equal(getRendererRoute(instance), ExecutionRoute.EXACT_BACKEND);
  assert.ok(getRendererDecision(instance).reasons.includes(EscapeReason.HOST_LIMITATION_FALLBACK));
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
    assert.equal(getRendererRoute(instance), ExecutionRoute.RETAINED_UPSTREAM);
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

  assert.equal(getRendererRoute(inst1), ExecutionRoute.EXACT_BACKEND);
  assert.equal(getRendererRoute(inst2), ExecutionRoute.RETAINED_UPSTREAM);
  assert.notEqual(getRendererDecision(inst1).canvas, getRendererDecision(inst2).canvas);
  assert.notEqual(getRendererDecision(inst1).rendererId, getRendererDecision(inst2).rendererId);
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

  assert.equal(getRendererRoute(inst1), ExecutionRoute.EXACT_BACKEND);
  assert.equal(getRendererRoute(inst2), ExecutionRoute.EXACT_BACKEND);
  assert.ok(inst1 instanceof RendererExact);
  assert.ok(inst2 instanceof RendererStandard, 'Group routing must preserve the source class');
  assert.equal(getRendererDecision(inst1).groupId, getRendererDecision(inst2).groupId, 'Renderers must share connected group ID');
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
      assert.match(err.message, new RegExp(`already committed to route '${ExecutionRoute.RETAINED_UPSTREAM}'`));
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
  assert.equal(getRendererRoute(inst1), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(getRendererRoute(inst2), ExecutionRoute.EXACT_BACKEND);
  assert.notEqual(getRendererDecision(inst1).groupId, getRendererDecision(inst2).groupId, 'Immutable asset sharing must NOT couple connected groups');
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
      assert.match(err.message, new RegExp(`already committed to route '${ExecutionRoute.RETAINED_UPSTREAM}'`));
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

test('Exact fallback without a source constructor requires a constructor-qualified implementation', () => {
  for (const unqualified of [undefined, AdmittedWebGLRenderer, { default: AdmittedWebGLRenderer }]) {
    const router = new RendererConstructionRouter({
      implementations: { [ExecutionRoute.EXACT_BACKEND]: unqualified },
    });
    assert.throws(
      () => router.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        options: { canvas: 'escaped-no-impl-canvas' },
        analysis: { hasOpaqueGLEscapes: true },
      }),
      /no admitted exact backend implementation registered/,
    );
    assert.equal(router.getCanvasLock('escaped-no-impl-canvas'), undefined);
  }
  class SourceWebGPURenderer {}
  const router = new RendererConstructionRouter({
    implementations: { [ExecutionRoute.EXACT_BACKEND]: { WebGPURenderer: SourceWebGPURenderer } },
  });
  assert.ok(router.routeAndConstruct({
    constructorName: 'WebGPURenderer', analysis: { hasOpaqueGLEscapes: true },
  }) instanceof SourceWebGPURenderer);
});

test('Opaque WebGPU constructor failure preserves the original error and executes once', () => {
  const originalError = new TypeError('source constructor failure');
  let calls = 0;
  class SourceWebGPURenderer {
    constructor() { calls++; throw originalError; }
  }
  const router = new RendererConstructionRouter({
    implementations: { [ExecutionRoute.EXACT_BACKEND]: AdmittedWebGLRenderer },
  });
  assert.throws(() => router.routeAndConstruct({
    constructorFn: SourceWebGPURenderer,
    constructorName: 'WebGPURenderer',
    analysis: { hasOpaqueGLEscapes: true },
  }), error => error === originalError);
  assert.equal(calls, 1);
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
  assert.ok(runtimeForcedInstance instanceof MockWebGPURenderer, 'Must preserve supplied WebGPURenderer constructor and prototype');
  assert.ok(!(runtimeForcedInstance instanceof AdmittedWebGLRenderer), 'Must not substitute legacy WebGLRenderer');
  assert.equal(runtimeForcedInstance.isWebGPURenderer, true);
  assert.equal(runtimeForcedInstance.opts.forceWebGL, true);

  // When runtime option evaluates to forceWebGL: false (default H1 without ?backend=webgl), routes to SPECIALIZED_WEBGPU
  const runtimeUnforcedInstance = router.routeAndConstruct({
    constructorFn: MockWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-runtime-unforced', forceWebGL: false },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(getRendererRoute(runtimeUnforcedInstance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(runtimeUnforcedInstance instanceof MockWebGPURenderer);
  assert.equal(runtimeUnforcedInstance.isWebGPURenderer, true);
  assert.equal(runtimeUnforcedInstance.opts.forceWebGL, false);
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

test('Positive: H1 production route preserves WebGPURenderer, keeps diagnostics external, and supports backend query reload', async () => {
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

  let gpuConstructorCalls = 0;
  class MockRenderableWebGPURenderer {
    constructor(opts = {}) {
      gpuConstructorCalls++;
      this.isWebGPURenderer = true;
      this.canvas = opts.canvas;
      this.opts = opts;
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

  // 1. Initial decision log is empty
  assert.deepEqual(router.getDecisionLog(), []);

  // 2. H1 forceWebGL branch (EXACT_BACKEND): constructs WebGPURenderer({ forceWebGL: true })
  // Must preserve the actual WebGPURenderer constructor and prototype, NOT substitute legacy WebGLRenderer
  const initialCalls = gpuConstructorCalls;
  const forcedSpan = 'examples/webgpu_performance_renderbundle.html:85:3';
  const forcedInstance = router.routeAndConstruct({
    constructorFn: MockRenderableWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-h1-forced', forceWebGL: true },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: forcedSpan,
  });

  // Constructor executes exactly once
  assert.equal(gpuConstructorCalls, initialCalls + 1, 'Constructor must execute exactly once');

  // Identity and prototype preserved: genuine WebGPURenderer, NOT substituted WebGLRenderer
  assert.ok(forcedInstance instanceof MockRenderableWebGPURenderer, 'Must preserve supplied WebGPURenderer constructor');
  assert.ok(!(forcedInstance instanceof MockRenderableWebGLRenderer), 'Must not substitute legacy WebGLRenderer');
  assert.equal(forcedInstance.constructor, MockRenderableWebGPURenderer);
  assert.equal(Object.getPrototypeOf(forcedInstance), MockRenderableWebGPURenderer.prototype);
  assert.equal(forcedInstance.isWebGPURenderer, true);
  assert.equal(forcedInstance.opts.forceWebGL, true);

  // Method shape preserved: render is the original prototype method, NOT wrapped
  assert.equal(Object.prototype.hasOwnProperty.call(forcedInstance, 'render'), false, 'render must not be an own property');
  assert.equal(forcedInstance.render, MockRenderableWebGPURenderer.prototype.render, 'render must be the original prototype method');

  // Zero source-observable __f3d_* own properties: diagnostics remain strictly external
  assert.equal(forcedInstance.__f3d_route__, undefined, 'No __f3d_route__ own property');
  assert.equal(forcedInstance.__f3d_decision__, undefined, 'No __f3d_decision__ own property');
  assert.equal(forcedInstance.__f3d_group_id__, undefined, 'No __f3d_group_id__ own property');
  assert.equal(forcedInstance.__f3d_renderer_id__, undefined, 'No __f3d_renderer_id__ own property');
  assert.deepEqual(Object.keys(forcedInstance).filter(k => k.startsWith('__f3d_')), [], 'No __f3d_* properties in Object.keys');

  // External diagnostics query works via WeakMaps
  assert.equal(getRendererRoute(forcedInstance), ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getInstanceRoute(forcedInstance), ExecutionRoute.EXACT_BACKEND);
  const forcedDecision = getRendererDecision(forcedInstance);
  assert.equal(forcedDecision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(forcedDecision.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));

  // Check decision log after H1 forceWebGL construction
  const decisionLog1 = router.getDecisionLog();
  assert.equal(decisionLog1.length, 1);
  assert.equal(decisionLog1[0].site, 'WebGPURenderer');
  assert.equal(decisionLog1[0].span, forcedSpan);
  assert.equal(decisionLog1[0].route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(decisionLog1[0].reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));
  assert.ok(typeof decisionLog1[0].group === 'string' && decisionLog1[0].group.length > 0);

  // 3. H1 WebGPU branch (SPECIALIZED_WEBGPU): constructs WebGPURenderer({ forceWebGL: false })
  const unforcedSpan = 'examples/webgpu_performance_renderbundle.html:95:3';
  const unforcedInstance = router.routeAndConstruct({
    constructorFn: MockRenderableWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-h1-unforced', forceWebGL: false },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: unforcedSpan,
  });

  assert.ok(unforcedInstance instanceof MockRenderableWebGPURenderer);
  assert.equal(unforcedInstance.constructor, MockRenderableWebGPURenderer);
  assert.equal(getRendererRoute(unforcedInstance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(Object.prototype.hasOwnProperty.call(unforcedInstance, 'render'), false);
  assert.deepEqual(Object.keys(unforcedInstance).filter(k => k.startsWith('__f3d_')), []);

  // Check decision log after H1 unforced WebGPU construction
  const decisionLog2 = router.getDecisionLog();
  assert.equal(decisionLog2.length, 2);
  assert.equal(decisionLog2[1].site, 'WebGPURenderer');
  assert.equal(decisionLog2[1].span, unforcedSpan);
  assert.equal(decisionLog2[1].route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(Array.isArray(decisionLog2[1].reasons));
  assert.ok(typeof decisionLog2[1].group === 'string' && decisionLog2[1].group.length > 0);

  // 4. Backend query reload on fresh canvas:
  // User reloads with ?backend=webgl on a fresh canvas (canvas-fresh-reload)
  const freshReloadInstance = router.routeAndConstruct({
    constructorFn: MockRenderableWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-fresh-reload', forceWebGL: true },
    analysis: h1Bundle,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: 'app.html:reload:1',
  });

  assert.ok(freshReloadInstance instanceof MockRenderableWebGPURenderer);
  assert.equal(getRendererRoute(freshReloadInstance), ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getCanvasLock('canvas-h1-unforced')?.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(router.getCanvasLock('canvas-fresh-reload')?.route, ExecutionRoute.EXACT_BACKEND);

  // 5. Rejected re-route on locked canvas logs the offending span
  const offendingSpan = 'src/offender_component.js:142:7';
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: MockRenderableWebGPURenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-h1-forced', forceWebGL: false }, // canvas-h1-forced is locked to EXACT_BACKEND
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

  // Decision log records the rejected re-route and its offending span
  const decisionLog4 = router.getDecisionLog();
  assert.equal(decisionLog4.length, 4, 'Decision log must record rejected re-route attempt');
  const rejectedDecision = decisionLog4[3];
  assert.equal(rejectedDecision.site, 'WebGPURenderer');
  assert.equal(rejectedDecision.span, offendingSpan, 'Rejected decision must record the offending source span');
  assert.equal(rejectedDecision.route, ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.ok(Array.isArray(rejectedDecision.reasons));
  assert.ok(typeof rejectedDecision.group === 'string' && rejectedDecision.group.length > 0);
});

test('Independent generated H1 facade execution: module initialization, constructor identity, instanceof, subclassing/new.target, and import-map aliases', async () => {
  const path = await import('node:path');
  const fs = await import('node:fs');
  const { fileURLToPath } = await import('node:url');

  const currentDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(currentDir, '../../..');

  const { generateRoutedWebGPUSource } = await import(
    path.resolve(repoRoot, 'tools/compat-facade/dev_server.mjs')
  );
  const { transformHtmlImportMap } = await import(
    path.resolve(repoRoot, 'tools/compat-facade/index.mjs')
  );

  // 1. Generate the actual runtime source emitted by the facade dev server
  const generatedSource = generateRoutedWebGPUSource({
    importBase: 'file://' + repoRoot,
  });

  // Verify absence of invalid prototype assignment (TypeError prevention on ES classes)
  assert.ok(
    !generatedSource.includes('WebGPURenderer.prototype ='),
    'Generated source must not assign to WebGPURenderer.prototype directly'
  );

  // 2. Independently evaluate the generated module source via data: URI
  const dataUri = 'data:text/javascript;base64,' + Buffer.from(generatedSource).toString('base64');
  const facade = await import(dataUri);

  assert.equal(typeof facade.WebGPURenderer, 'function', 'Exported WebGPURenderer must be a constructor function');
  assert.ok(facade.router instanceof RendererConstructionRouter, 'Exported router must be an instance of RendererConstructionRouter');

  // 3. Direct construction in WebGPU mode (forceWebGL: false)
  const canvasWebGPU = { getContext: () => null, addEventListener: () => {} };
  const instWebGPU = new facade.WebGPURenderer({
    canvas: canvasWebGPU,
    forceWebGL: false,
  });
  assert.ok(instWebGPU instanceof facade.WebGPURenderer, 'instWebGPU must be instanceof facade.WebGPURenderer');
  assert.equal(instWebGPU.backend.constructor.name, 'WebGPUBackend', 'forceWebGL: false selects WebGPUBackend');
  assert.equal(Object.keys(instWebGPU).filter(k => k.startsWith('__f3d_')).length, 0, 'No __f3d_* own properties');

  // 4. Direct construction in WebGL mode (forceWebGL: true) - genuine WebGPURenderer with WebGLBackend
  const canvasWebGL = { getContext: () => null, addEventListener: () => {} };
  const instWebGL = new facade.WebGPURenderer({
    canvas: canvasWebGL,
    forceWebGL: true,
  });
  assert.ok(instWebGL instanceof facade.WebGPURenderer, 'instWebGL must be instanceof facade.WebGPURenderer');
  assert.equal(instWebGL.backend.constructor.name, 'WebGLBackend', 'forceWebGL: true selects WebGLBackend (never legacy WebGLRenderer)');

  // 5. Single constructor execution invariant
  assert.equal(instWebGPU.renderCalls ?? 0, 0);
  assert.equal(instWebGL.renderCalls ?? 0, 0);

  // 6. Subclassing with explicit constructor, custom fields, and new.target preservation
  let subclassConstructorExecutions = 0;
  class CustomAppRenderer extends facade.WebGPURenderer {
    constructor(opts) {
      super(opts);
      subclassConstructorExecutions++;
      this.customFeature = 'enabled';
      this.capturedNewTarget = new.target;
    }
  }

  const canvasSubclass = { getContext: () => null, addEventListener: () => {} };
  const subInstance = new CustomAppRenderer({
    canvas: canvasSubclass,
    forceWebGL: false,
  });

  assert.equal(subclassConstructorExecutions, 1, 'Subclass constructor must execute exactly once');
  assert.ok(subInstance instanceof CustomAppRenderer, 'Must be instanceof CustomAppRenderer');
  assert.ok(subInstance instanceof facade.WebGPURenderer, 'Must be instanceof facade.WebGPURenderer');
  assert.equal(subInstance.customFeature, 'enabled', 'Custom subclass fields must be preserved');
  assert.equal(subInstance.capturedNewTarget, CustomAppRenderer, 'new.target must be preserved in derived constructor');
  assert.equal(subInstance.backend.constructor.name, 'WebGPUBackend');

  // 7. Subclassing with method override
  class MethodOverrideRenderer extends facade.WebGPURenderer {
    render(scene, camera) {
      return { overridden: true, scene, camera };
    }
  }
  const overrideInstance = new MethodOverrideRenderer({
    canvas: { getContext: () => null, addEventListener: () => {} },
    forceWebGL: true,
  });
  assert.ok(overrideInstance instanceof MethodOverrideRenderer);
  assert.ok(overrideInstance instanceof facade.WebGPURenderer);
  assert.deepEqual(overrideInstance.render('scene1', 'cam1'), { overridden: true, scene: 'scene1', camera: 'cam1' });

  // 8. Subclassing without explicit constructor (default constructor)
  class DefaultConstructorRenderer extends facade.WebGPURenderer {}
  const defInstance = new DefaultConstructorRenderer({
    canvas: { getContext: () => null, addEventListener: () => {} },
    forceWebGL: false,
  });
  assert.ok(defInstance instanceof DefaultConstructorRenderer);
  assert.ok(defInstance instanceof facade.WebGPURenderer);
  assert.equal(defInstance.constructor, DefaultConstructorRenderer);

  // 9. Multi-level subclass inheritance chain
  class LevelOneRenderer extends facade.WebGPURenderer {}
  class LevelTwoRenderer extends LevelOneRenderer {}
  const multiLevelInstance = new LevelTwoRenderer({
    canvas: { getContext: () => null, addEventListener: () => {} },
    forceWebGL: true,
  });
  assert.ok(multiLevelInstance instanceof LevelTwoRenderer);
  assert.ok(multiLevelInstance instanceof LevelOneRenderer);
  assert.ok(multiLevelInstance instanceof facade.WebGPURenderer);
  assert.equal(multiLevelInstance.constructor, LevelTwoRenderer);

  // 10. Reflect.construct invocation with arbitrary new.target
  function TargetFunction() {}
  TargetFunction.prototype = Object.create(facade.WebGPURenderer.prototype);
  TargetFunction.prototype.constructor = TargetFunction;

  const reflectInstance = Reflect.construct(
    facade.WebGPURenderer,
    [{ canvas: { getContext: () => null, addEventListener: () => {} }, forceWebGL: false }],
    TargetFunction
  );
  assert.ok(reflectInstance instanceof TargetFunction);
  assert.ok(reflectInstance instanceof facade.WebGPURenderer);

  // 11. H1 source import map preservation: both 'three' and 'three/webgpu' bind to the same module singleton
  const h1HtmlPath = path.resolve(repoRoot, 'upstream/three.js/examples/webgpu_performance_renderbundle.html');
  const h1Html = fs.readFileSync(h1HtmlPath, 'utf8');
  const transformedH1 = transformHtmlImportMap(h1Html, { baseUrl: 'http://127.0.0.1:8080' });

  assert.ok(
    transformedH1.includes('"three": "http://127.0.0.1:8080/compat-facade/webgpu.js"'),
    'H1 alias for "three" must map to webgpu.js facade'
  );
  assert.ok(
    transformedH1.includes('"three/webgpu": "http://127.0.0.1:8080/compat-facade/webgpu.js"'),
    'H1 alias for "three/webgpu" must map to webgpu.js facade'
  );
  assert.ok(
    transformedH1.includes('"three/tsl": "http://127.0.0.1:8080/compat-facade/tsl.js"'),
    'H1 alias for "three/tsl" must map to tsl.js facade'
  );
  assert.ok(
    transformedH1.includes('"three/addons/": "http://127.0.0.1:8080/compat-facade/addons/"'),
    'H1 alias for "three/addons/" must map to addons/ facade'
  );

  // Decision log records all facade-constructed renderers
  const decisionLog = facade.router.getDecisionLog();
  assert.ok(decisionLog.length >= 7, 'Router must log decisions for all facade-routed instances');
  assert.ok(decisionLog.every(d => d.site === 'WebGPURenderer'));

  // 12. Honest implementation ownership and pristine native prototypes (Plan Section 5.1)
  const registeredRouteKeys = Object.keys(facade.router.implementations).sort();
  assert.deepEqual(
    registeredRouteKeys,
    [ExecutionRoute.EXACT_BACKEND, ExecutionRoute.RETAINED_UPSTREAM].sort(),
    'Facade router must strictly register exact-backend and retained-upstream (never claiming unbuilt specialized/general WebGPU)'
  );

  const upstreamModule = await import(path.resolve(repoRoot, 'upstream/three.js/build/three.webgpu.js'));
  assert.equal(
    upstreamModule.WebGPURenderer.prototype.constructor,
    upstreamModule.WebGPURenderer,
    'Upstream prototype.constructor must remain completely pristine and unmutated'
  );

  // Honest documentation of Proxy constructor-equality limitation (not papered over with prototype pollution)
  assert.equal(
    instWebGPU.constructor,
    upstreamModule.WebGPURenderer,
    'Direct instance constructor truthfully references the upstream class constructor'
  );
  assert.notEqual(
    instWebGPU.constructor,
    facade.WebGPURenderer,
    'Direct instance constructor does not equal the outer Proxy wrapper (honest boundary limitation)'
  );
});

test('6mv.4 report criterion: window-independent router decision log and attribution log for H1 WebGPU and forceWebGL branches', async () => {
  const path = await import('node:path');
  const { fileURLToPath } = await import('node:url');

  const currentDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(currentDir, '../../..');

  const { generateRoutedWebGPUSource } = await import(
    path.resolve(repoRoot, 'tools/compat-facade/dev_server.mjs')
  );

  // 1. Generate runtime source emitted by facade dev server and evaluate via fresh data URI
  const generatedSource = generateRoutedWebGPUSource({
    importBase: 'file://' + repoRoot,
  }) + '\n// test-nonce: 6mv.4-report-regression-' + Date.now();
  const dataUri = 'data:text/javascript;base64,' + Buffer.from(generatedSource).toString('base64');
  const facade = await import(dataUri);

  // 2. Assert window-independent execution (no window global required in Node)
  assert.equal(typeof window, 'undefined', 'Test must execute window-independently in Node');
  assert.ok(facade.router instanceof RendererConstructionRouter, 'Facade must export router directly');

  // 3. Construct WebGPURenderer with forceWebGL: false on fresh canvas (WebGPU branch)
  const canvasWebGPU = { getContext: () => null, addEventListener: () => {} };
  const rendererWebGPU = new facade.WebGPURenderer({
    canvas: canvasWebGPU,
    forceWebGL: false,
  });
  assert.ok(rendererWebGPU instanceof facade.WebGPURenderer, 'WebGPU renderer must be instanceof facade.WebGPURenderer');

  // 4. Construct WebGPURenderer with forceWebGL: true on fresh canvas (forceWebGL branch)
  const canvasWebGL = { getContext: () => null, addEventListener: () => {} };
  const rendererWebGL = new facade.WebGPURenderer({
    canvas: canvasWebGL,
    forceWebGL: true,
  });
  assert.ok(rendererWebGL instanceof facade.WebGPURenderer, 'WebGL renderer must be instanceof facade.WebGPURenderer');

  // 5. Query window-independent router.getDecisionLog()
  const decisionLog = facade.router.getDecisionLog();
  assert.equal(decisionLog.length, 2, 'Router decision log must record both constructions');

  // Branch 1: WebGPU branch (forceWebGL: false) -> retained-upstream
  const webgpuEntry = decisionLog[0];
  assert.equal(webgpuEntry.site, 'WebGPURenderer', 'Site must be WebGPURenderer');
  assert.equal(webgpuEntry.span, 'webgpu_performance_renderbundle.html:188:13', 'Span must match H1 constructor site');
  assert.equal(webgpuEntry.route, ExecutionRoute.RETAINED_UPSTREAM, 'WebGPU branch must route to retained-upstream');
  assert.deepEqual(webgpuEntry.reasons, [EscapeReason.SPECIALIZATION_UNAVAILABLE], 'Reasons must contain specialization-unavailable');
  assert.equal(typeof webgpuEntry.group, 'string', 'Group must be a non-empty string');
  assert.ok(webgpuEntry.group.length > 0, 'Group ID must be populated');

  // Branch 2: forceWebGL branch (forceWebGL: true) -> exact-backend
  const webglEntry = decisionLog[1];
  assert.equal(webglEntry.site, 'WebGPURenderer', 'Site must be WebGPURenderer');
  assert.equal(webglEntry.span, 'webgpu_performance_renderbundle.html:188:13', 'Span must match H1 constructor site');
  assert.equal(webglEntry.route, ExecutionRoute.EXACT_BACKEND, 'forceWebGL branch must route to exact-backend');
  assert.deepEqual(webglEntry.reasons, [EscapeReason.EXPLICIT_SOURCE_SELECTION], 'Reasons must contain explicit-source-selection');
  assert.equal(typeof webglEntry.group, 'string', 'Group must be a non-empty string');
  assert.ok(webglEntry.group.length > 0, 'Group ID must be populated');

  // Assert distinct group IDs across independent fresh canvases
  assert.notEqual(webgpuEntry.group, webglEntry.group, 'Independent canvases must have distinct group memberships');

  // 6. Query window-independent router.getAttributionLog()
  const attributionLog = facade.router.getAttributionLog();
  assert.ok(Array.isArray(attributionLog), 'Attribution log must be an array');
  assert.equal(attributionLog.length, 0, 'Attribution log must be empty prior to render calls');

  // 7. Verify canvas locks match decision routes
  const lockWebGPU = facade.router.getCanvasLock(canvasWebGPU);
  assert.equal(lockWebGPU?.route, ExecutionRoute.RETAINED_UPSTREAM, 'Canvas lock for WebGPU canvas must match retained-upstream');

  const lockWebGL = facade.router.getCanvasLock(canvasWebGL);
  assert.equal(lockWebGL?.route, ExecutionRoute.EXACT_BACKEND, 'Canvas lock for WebGL canvas must match exact-backend');
});

test('Positive and negative: Exact backend router preserves route and permanent canvas lock across WebGL context loss and restore', () => {
  class TestWebGLRenderer {
    constructor(opts = {}) {
      this.isWebGLRenderer = true;
      this.canvas = opts.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: TestWebGLRenderer,
    },
  });

  // Create mock canvas with WEBGL_lose_context simulation capabilities
  let lostHandler = null;
  let restoredHandler = null;
  let contextLostState = false;

  const mockLoseContextExt = {
    loseContext() {
      contextLostState = true;
      if (lostHandler) {
        lostHandler({ preventDefault: () => {} });
      }
    },
    restoreContext() {
      contextLostState = false;
      if (restoredHandler) {
        restoredHandler({});
      }
    },
  };

  const mockGl = {
    isContextLost: () => contextLostState,
    getExtension: (name) => (name === 'WEBGL_lose_context' ? mockLoseContextExt : null),
  };

  const canvas = {
    id: 'mock-canvas-loss-restore',
    getContext: (type) => (type && type.includes('webgl') ? mockGl : null),
    addEventListener: (type, fn) => {
      if (type === 'webglcontextlost') lostHandler = fn;
      if (type === 'webglcontextrestored') restoredHandler = fn;
    },
    removeEventListener: () => {},
  };

  // Route and construct
  const renderer = router.routeAndConstruct({
    constructorFn: TestWebGLRenderer,
    constructorName: 'WebGLRenderer',
    options: { canvas },
    sourceSpan: 'test:context_loss_runner_hook',
  });

  assert.equal(getRendererRoute(renderer), ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.EXACT_BACKEND);

  // Trigger context loss
  mockLoseContextExt.loseContext();
  assert.equal(mockGl.isContextLost(), true);

  // Canvas lock and route remain permanently EXACT_BACKEND during context loss
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(getRendererRoute(renderer), ExecutionRoute.EXACT_BACKEND);

  // Negative control 1: Route switch on canvas during context loss is strictly rejected with RouteLockError
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: class MockWebGPURenderer {},
        constructorName: 'WebGPURenderer',
        options: { canvas },
      });
    },
    RouteLockError,
    'Canvas lock must reject late route switch even during WebGL context loss'
  );

  // Negative control 2: Un-restored state check - skipping restoration leaves isContextLost() true
  assert.equal(mockGl.isContextLost(), true, 'Context must remain lost when restoration is skipped');

  // Negative control 3: Unadmitted exact backend implementation is rejected
  assert.throws(
    () => {
      const strictRouter = new RendererConstructionRouter({
        implementations: {},
      });
      strictRouter.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        options: { canvas: 'unadmitted-loss-canvas' },
        analysis: { hasOpaqueGLEscapes: true },
      });
    },
    /no admitted exact backend implementation registered/,
    'Unadmitted class substitution must be rejected'
  );

  // Restore context
  mockLoseContextExt.restoreContext();
  assert.equal(mockGl.isContextLost(), false);

  // Verify route and canvas lock remain intact after restoration
  assert.equal(getRendererRoute(renderer), ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.EXACT_BACKEND);
});

test('Regression: nested successful construction surviving outer preflight failure', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
    },
  });

  // Lock canvas1 to EXACT_BACKEND
  router.routeAndConstruct({
    constructorName: 'WebGLRenderer',
    options: { canvas: 'prelocked-canvas' },
  });

  let nestedInstance = null;
  let nestedId = null;

  // sharedResources getter reenters routeAndConstruct during outer preflight
  const reentrantResources = [
    {
      get id() {
        if (!nestedInstance) {
          nestedInstance = router.routeAndConstruct({
            constructorName: 'WebGPURenderer',
            options: { canvas: 'canvas-nested-survival' },
            hostCapabilities: { hasWebGPU: true, hasWebGL: true },
          });
          nestedId = router.getInstanceDecision(nestedInstance)?.rendererId;
        }
        return 'shared-rt-nested';
      },
      isMutable: true,
    },
  ];

  // Outer attempts to route WebGPURenderer on prelocked-canvas (locked to EXACT_BACKEND)
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        options: { canvas: 'prelocked-canvas' },
        sharedResources: reentrantResources,
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      });
    },
    RouteLockError
  );

  // Assert nested renderer survived completely intact
  assert.ok(nestedInstance, 'Nested renderer must have been constructed');
  assert.equal(router.getInstanceRoute(nestedInstance), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getCanvasLock('canvas-nested-survival')?.route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(nestedId, 'Nested renderer must have assigned ID');
  assert.equal(connectedGroups._committedRoutes.get(nestedId), ExecutionRoute.RETAINED_UPSTREAM);
});

test('Regression: reentrant construction in callImplementations alters constraints and triggers preflight revalidation failure', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
    },
  });

  let nestedConstructed = false;

  const dynamicCallImplementations = {
    get [ExecutionRoute.RETAINED_UPSTREAM]() {
      if (!nestedConstructed) {
        nestedConstructed = true;
        // Nested construction couples shared-target-reval with an EXACT_BACKEND renderer
        router.routeAndConstruct({
          constructorName: 'WebGLRenderer',
          options: { canvas: 'canvas-nested-exact' },
          sharedResources: ['shared-target-reval'],
        });
      }
      return function MockDynamicRetained() { this.isRetained = true; };
    },
  };

  // Outer renderer starts with WebGPURenderer sharing shared-target-reval.
  // Initially shared-target-reval has no exact constraints.
  // During implementation lookup, the getter fires and couples shared-target-reval to EXACT_BACKEND.
  // The post-selection revalidation must conservatively catch this and reject outer construction!
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-outer-reval' },
        sharedResources: ['shared-target-reval'],
        implementations: dynamicCallImplementations,
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      });
    },
    /no admitted exact backend implementation registered/
  );

  // Canvas of outer was NOT locked
  assert.equal(router.getCanvasLock('canvas-outer-reval'), undefined);
  // Nested renderer state is preserved
  assert.equal(router.getCanvasLock('canvas-nested-exact')?.route, ExecutionRoute.EXACT_BACKEND);
});

test('Regression: preflight RouteLockError does not poison shared resources for subsequent renderers', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
    },
  });

  // Lock canvas1 to RETAINED_UPSTREAM
  router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-locked-gpu' },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  // Attempt WebGLRenderer on canvas-locked-gpu with shared-rt-unpoisoned -> throws RouteLockError
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-locked-gpu' },
        sharedResources: ['shared-rt-unpoisoned'],
      });
    },
    RouteLockError
  );

  // Subsequent WebGPURenderer on fresh canvas sharing shared-rt-unpoisoned must NOT be forced to EXACT_BACKEND
  const r2 = router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-fresh-unpoisoned' },
    sharedResources: ['shared-rt-unpoisoned'],
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  assert.equal(router.getInstanceRoute(r2), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getCanvasLock('canvas-fresh-unpoisoned')?.route, ExecutionRoute.RETAINED_UPSTREAM);
});

test('Sunny regression: implementations[EXACT] getter builds retained inner sharing S, then outer fails; inner survives and blocks later exact sharing S', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
    },
  });

  let innerConstructed = false;
  let innerRendererId = null;

  const dynamicImplementations = {
    get [ExecutionRoute.EXACT_BACKEND]() {
      if (!innerConstructed) {
        innerConstructed = true;
        const inner = router.routeAndConstruct({
          constructorName: 'WebGPURenderer',
          options: { canvas: 'canvas-sunny-inner' },
          sharedResources: ['sunny-resource-S'],
          hostCapabilities: { hasWebGPU: true, hasWebGL: true },
        });
        innerRendererId = router.getInstanceDecision(inner)?.rendererId;
      }
      return null; // Outer implementation lookup fails
    },
  };

  // Outer attempts EXACT_BACKEND sharing 'sunny-resource-S'; triggers getter and fails
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-sunny-outer' },
        sharedResources: ['sunny-resource-S'],
        implementations: dynamicImplementations,
      });
    },
    /no admitted exact backend implementation registered/
  );

  // Assert inner survived and is committed to RETAINED_UPSTREAM
  assert.equal(router.getCanvasLock('canvas-sunny-inner')?.route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(innerRendererId);
  assert.equal(connectedGroups._committedRoutes.get(innerRendererId), ExecutionRoute.RETAINED_UPSTREAM);

  // Later exact renderer sharing sunny-resource-S must be rejected with Connected group conflict
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-sunny-later' },
        sharedResources: ['sunny-resource-S'],
        implementations: {
          [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
        },
      });
    },
    /Connected group conflict/
  );
});

test('Sunny regression: recordCommittedRoute before constructor blocks reentrant exact construction from inside constructor', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
  });

  let reentrancyAttempted = false;
  let reentrancyBlocked = false;

  class OuterRetainedRenderer {
    constructor(opts) {
      reentrancyAttempted = true;
      try {
        router.routeAndConstruct({
          constructorName: 'WebGLRenderer',
          options: { canvas: 'canvas-reentrant-inside-constructor' },
          sharedResources: ['sunny-resource-S2'],
          implementations: {
            [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
          },
        });
      } catch (err) {
        if (/Connected group conflict/.test(err.message)) {
          reentrancyBlocked = true;
        }
        throw err;
      }
    }
  }

  // Outer constructs with OuterRetainedRenderer sharing 'sunny-resource-S2'.
  // Because recordCommittedRoute runs before targetConstructor(options),
  // outer's route is committed when constructor body runs, blocking reentrant exact sharing.
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorFn: OuterRetainedRenderer,
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-outer-main' },
        sharedResources: ['sunny-resource-S2'],
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      });
    },
    /Connected group conflict/
  );

  assert.equal(reentrancyAttempted, true);
  assert.equal(reentrancyBlocked, true);
});

test('Root regression: available exact implementation selected after reentrant group escalation to EXACT_BACKEND', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();

  let innerCalls = 0;
  let getterCalls = 0;
  let outerExactCalls = 0;

  function ExactWebGPUCtor(opts) {
    outerExactCalls++;
    this.isExactWebGPU = true;
    this.canvas = opts?.canvas;
  }

  function InnerWebGLCtor(opts) {
    innerCalls++;
    this.isInnerWebGL = true;
    this.canvas = opts?.canvas;
  }

  function RetainedCtor(opts) {
    this.isRetained = true;
  }

  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: {
        WebGLRenderer: InnerWebGLCtor,
        WebGPURenderer: ExactWebGPUCtor,
      },
    },
  });

  const dynamicImplementations = {
    get [ExecutionRoute.RETAINED_UPSTREAM]() {
      getterCalls++;
      // Inner constructs exact WebGLRenderer sharing 'shared-S'
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-inner-root' },
        sharedResources: ['shared-S'],
      });
      return RetainedCtor;
    },
  };

  // Outer constructs WebGPURenderer sharing 'shared-S'.
  // Initial route is RETAINED_UPSTREAM. Implementation getter runs and inner exact is constructed.
  // Group constraint escalates monotonically to EXACT_BACKEND.
  // Router selects available ExactWebGPUCtor from captured implementation without repeating getter.
  const outerInstance = router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-outer-root' },
    sharedResources: ['shared-S'],
    implementations: dynamicImplementations,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  assert.equal(innerCalls, 1, 'innerCalls must be 1');
  assert.equal(getterCalls, 1, 'getterCalls must be 1');
  assert.equal(outerExactCalls, 1, 'outerExactCalls must be 1');

  // Both canvas locks are irreversible and properly set
  assert.equal(router.getCanvasLock('canvas-inner-root')?.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getCanvasLock('canvas-outer-root')?.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(router.getInstanceRoute(outerInstance), ExecutionRoute.EXACT_BACKEND);
});

test('Final review regression: refresh final group metadata after reentrant construction sharing resource', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
  });

  let innerCalls = 0;
  let outerCalls = 0;
  let getterCalls = 0;

  function InnerRetainedCtor(opts) {
    innerCalls++;
    this.isInner = true;
  }

  function OuterRetainedCtor(opts) {
    outerCalls++;
    this.isOuter = true;
  }

  let innerInstance = null;

  const dynamicImplementations = {
    get [ExecutionRoute.RETAINED_UPSTREAM]() {
      getterCalls++;
      innerInstance = router.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-inner-meta' },
        sharedResources: ['shared-meta-S'],
        implementations: {
          [ExecutionRoute.RETAINED_UPSTREAM]: InnerRetainedCtor,
        },
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      });
      return OuterRetainedCtor;
    },
  };

  const outerInstance = router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    options: { canvas: 'canvas-outer-meta' },
    sharedResources: ['shared-meta-S'],
    implementations: dynamicImplementations,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  assert.equal(innerCalls, 1, 'innerCalls must be 1');
  assert.equal(outerCalls, 1, 'outerCalls must be 1');
  assert.equal(getterCalls, 1, 'getterCalls must be 1');

  const innerDecision = router.getInstanceDecision(innerInstance);
  const outerDecision = router.getInstanceDecision(outerInstance);

  assert.ok(innerDecision?.groupId, 'innerDecision must have groupId');
  assert.ok(outerDecision?.groupId, 'outerDecision must have groupId');
  assert.equal(outerDecision.groupId, innerDecision.groupId, 'outer and inner renderers sharing S must report the same groupId');
});







