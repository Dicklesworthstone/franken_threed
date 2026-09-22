/**
 * @file construction_adapter.test.mjs
 * Comprehensive unit test suite for RendererConstructionRouter and diagnostics (Bead f3d-04.4).
 *
 * Tests:
 * - Defect A: Preflight failures do not mutate connected groups or poison shared resources:
 *   - Constructor throw retains irreversible canvas lock (binds-then-throws).
 *   - Preflight RouteLockError does not couple resources or poison future renderers.
 *   - Missing implementation error does not poison connected groups.
 * - Defect B: Zero property reads on instances and native shape preservation:
 *   - getRendererRoute, getRendererDecision, getInstanceRoute, getInstanceDecision do NOT read
 *     __f3d_route__ or __f3d_decision__ from instances.
 *   - Proxies with throwing getters for __f3d_* do not throw.
 *   - Sealed and frozen instances are untouched and diagnosed via external WeakMaps.
 * - Defect C: Unnamed and wrapped resource identity:
 *   - Two distinct unnamed RenderTarget objects do not collide onto a single key.
 *   - Same object reference across renderers couples backend residency.
 *   - Numeric id: 0 is preserved as '0'.
 *   - Wrappers { resource: obj, isMutable } and [obj, isMutable] are respected.
 * - Defect D: Truthful constructor selection and fallback for SPECIALIZED_WEBGPU:
 *   - Registered specialized implementation is invoked instead of caller-supplied constructorFn.
 *   - When specialization is unavailable, falls back to RETAINED_UPSTREAM before effects, never labeling retained source as specialized.
 *   - specialized-island-admitted reason is removed on fallback (admission and unavailability not reported together).
 *   - When specialization is unavailable and constructorFn is absent, falls back to registered RETAINED_UPSTREAM implementation.
 *   - When neither specialized nor fallback implementation exists, throws truthful error before effects.
 *   - targetImplementation override takes precedence.
 *   - Qualified registered constructor takes precedence over default and bypasses source constructorFn.
 *   - Two calls on same canvas with specializationAvailable=true both construct retained without lock collision.
 *   - Re-routing canvas from RETAINED_UPSTREAM to SPECIALIZED_WEBGPU when replacement exists throws RouteLockError.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { BufferGeometry } from "../../upstream/three.js/src/core/BufferGeometry.js";
import { RenderTarget } from "../../upstream/three.js/src/core/RenderTarget.js";
import { Material } from "../../upstream/three.js/src/materials/Material.js";
import { Texture } from "../../upstream/three.js/src/textures/Texture.js";
import { ConnectedCompatibilityGroups } from "./connected_groups.mjs";
import {
  getRendererDecision,
  getRendererRoute,
  RendererConstructionRouter,
  resolveResourceId,
} from "./construction_adapter.mjs";
import { EscapeReason, ExecutionRoute, RouteLockError } from "./route_types.mjs";

test("Resource identity separates actual Three classes and ignores mutable instance labels", () => {
  const texture = new Texture();
  const geometry = new BufferGeometry();
  const material = new Material();
  // r186 allocates these IDs independently, starting at zero for each class family.
  assert.equal(texture.id, geometry.id);
  assert.equal(texture.id, material.id);
  const resources = [texture, geometry, material];
  const ids = resources.map(resolveResourceId);
  assert.equal(new Set(ids).size, resources.length);
  for (const [index, resource] of resources.entries()) {
    assert.equal(resolveResourceId({ resource, isMutable: false }), ids[index]);
    assert.equal(resolveResourceId({ target: resource }), ids[index]);
    resource.name = "same display name";
    assert.equal(resolveResourceId(resource), ids[index]);
  }
  const targets = [new RenderTarget(), new RenderTarget()];
  for (const target of targets) target.name = "same display name";
  assert.notEqual(resolveResourceId(targets[0]), resolveResourceId(targets[1]));
  assert.equal(resolveResourceId({ id: 0 }), "0");
  assert.equal(resolveResourceId({ resourceId: "shared" }), "shared");
});

test("Contract: Constructor throw retains irreversible canvas lock (binds-then-throws)", () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: function FailingWebGL() {
        throw new Error("WebGL context creation failed");
      },
    },
  });

  // 1. Attempt WebGLRenderer construction; constructor throws
  assert.throws(() => {
    router.routeAndConstruct({
      constructorName: "WebGLRenderer",
      options: { canvas: "canvas-failed-gl" },
    });
  }, /WebGL context creation failed/);

  // Canvas lock is permanently retained (binds-then-throws invariant, Plan §3.3)
  assert.equal(router.getCanvasLock("canvas-failed-gl")?.route, ExecutionRoute.EXACT_BACKEND);

  // Late route switch on same canvas must be rejected
  assert.throws(() => {
    router.routeAndConstruct({
      constructorName: "WebGPURenderer",
      options: { canvas: "canvas-failed-gl" },
      hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    });
  }, RouteLockError);
});

test("Defect A: Preflight RouteLockError does not poison connected groups", () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() {
        this.isRetained = true;
      },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() {
        this.isExact = true;
      },
    },
  });

  const canvas1 = { id: "canvas-shared" };
  const sharedTarget = { id: "rt-shared-preflight", isMutable: true };

  // 1. Lock canvas1 to RETAINED_UPSTREAM via WebGPURenderer
  const r1 = router.routeAndConstruct({
    constructorName: "WebGPURenderer",
    options: { canvas: canvas1 },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getInstanceRoute(r1), ExecutionRoute.RETAINED_UPSTREAM);

  // 2. Preflight failure: WebGLRenderer tries to use already-locked canvas1
  assert.throws(() => {
    router.routeAndConstruct({
      constructorName: "WebGLRenderer",
      options: { canvas: canvas1 },
      sharedResources: [sharedTarget],
    });
  }, RouteLockError);

  // Connected groups must NOT keep the rejected WebGLRenderer connected to sharedTarget
  const members = connectedGroups.getGroupMembers("rt-shared-preflight");
  assert.equal(
    members.some((m) => m.startsWith("renderer-")),
    false,
  );

  // 3. Another renderer using sharedTarget on fresh canvas2 routes normally to RETAINED_UPSTREAM
  const r3 = router.routeAndConstruct({
    constructorName: "WebGPURenderer",
    options: { canvas: { id: "canvas-fresh" } },
    sharedResources: [sharedTarget],
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getInstanceRoute(r3), ExecutionRoute.RETAINED_UPSTREAM);
});

test("Defect A: Missing implementation error does not poison connected groups", () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {}, // No implementations registered
  });

  const sharedTarget = { id: "rt-missing-impl", isMutable: true };

  assert.throws(() => {
    router.routeAndConstruct({
      constructorName: "WebGLRenderer",
      sharedResources: [sharedTarget],
    });
  }, /no admitted exact backend implementation registered/);

  assert.equal(connectedGroups.getGroupMembers("rt-missing-impl").length, 0);
  assert.equal(connectedGroups._nodes.has("rt-missing-impl"), false);
});

test("Defect B: Zero property reads on instances preserves proxies, sealed, and frozen objects", () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() {
        this.isRetained = true;
      },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() {
        this.isExact = true;
      },
    },
  });

  // Strict proxy that throws on any __f3d_* access
  let trapped = false;
  const strictProxy = new Proxy(
    {},
    {
      get(target, prop) {
        if (typeof prop === "string" && prop.startsWith("__f3d_")) {
          trapped = true;
          throw new Error(`Forbidden property access: ${prop}`);
        }
        return target[prop];
      },
    },
  );

  assert.equal(getRendererRoute(strictProxy), undefined);
  assert.equal(getRendererDecision(strictProxy), undefined);
  assert.equal(router.getInstanceRoute(strictProxy), undefined);
  assert.equal(router.getInstanceDecision(strictProxy), undefined);
  assert.equal(trapped, false, "Proxy __f3d_* getters must not be invoked");

  // Constructed sealed and frozen instances have zero __f3d_* properties
  class CustomSealed {
    constructor() {
      this.val = 42;
      Object.seal(this);
    }
  }

  const sealedInst = router.routeAndConstruct({
    constructorFn: CustomSealed,
    constructorName: "CustomSealed",
  });

  assert.equal(Object.isSealed(sealedInst), true);
  assert.equal(sealedInst.__f3d_route__, undefined);
  assert.equal(sealedInst.__f3d_decision__, undefined);
  assert.equal(getRendererRoute(sealedInst), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getInstanceRoute(sealedInst), ExecutionRoute.RETAINED_UPSTREAM);
});

test("Defect C: Resource identity resolution distinguishes distinct unnamed objects", () => {
  const groups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups: groups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() {
        this.isRetained = true;
      },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() {
        this.isExact = true;
      },
    },
  });

  // Two distinct unnamed RenderTarget objects
  const rt1 = { width: 512, height: 512 };
  const rt2 = { width: 1024, height: 1024 };

  const id1 = resolveResourceId(rt1);
  const id2 = resolveResourceId(rt2);

  assert.ok(id1.startsWith("resource-obj-"));
  assert.ok(id2.startsWith("resource-obj-"));
  assert.notEqual(id1, id2, "Distinct unnamed objects must receive distinct IDs");
  assert.equal(resolveResourceId(rt1), id1, "Identical object reference must return stable ID");

  // Renderer 1 uses rt1 with WebGLRenderer (EXACT_BACKEND)
  const r1 = router.routeAndConstruct({
    constructorName: "WebGLRenderer",
    sharedResources: [rt1],
  });
  assert.equal(router.getInstanceRoute(r1), ExecutionRoute.EXACT_BACKEND);

  // Renderer 2 uses rt2 with WebGPURenderer -> should NOT collide with rt1
  const r2 = router.routeAndConstruct({
    constructorName: "WebGPURenderer",
    sharedResources: [rt2],
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getInstanceRoute(r2), ExecutionRoute.RETAINED_UPSTREAM);

  // Renderer 3 uses the SAME rt1 with WebGPURenderer -> correctly couples and throws conflict or forces exact
  assert.throws(() => {
    router.routeAndConstruct({
      constructorName: "WebGPURenderer",
      sharedResources: [rt1],
      hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    });
  }, /no admitted exact backend implementation registered/);
});

test("Defect C: Numeric id: 0 and resource wrappers are resolved correctly", () => {
  const resWithZero = { id: 0 };
  assert.equal(resolveResourceId(resWithZero), "0");

  const rawObj = { isRaw: true };
  const wrappedResource = { resource: rawObj, isMutable: false };
  const wrappedTarget = { target: rawObj, isMutable: true };

  assert.equal(resolveResourceId(wrappedResource), resolveResourceId(rawObj));
  assert.equal(resolveResourceId(wrappedTarget), resolveResourceId(rawObj));
});

test("Defect D: SPECIALIZED_WEBGPU selection invokes registered specialized implementation, NOT source constructorFn", () => {
  let sourceConstructorCalls = 0;
  class UpstreamWebGPURenderer {
    constructor(opts) {
      sourceConstructorCalls++;
      this.isUpstreamSource = true;
      this.canvas = opts?.canvas;
    }
  }

  let specializedCalls = 0;
  class AdmittedSpecializedRenderer {
    constructor(opts) {
      specializedCalls++;
      this.isSpecialized = true;
      this.canvas = opts?.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.SPECIALIZED_WEBGPU]: AdmittedSpecializedRenderer,
    },
  });

  const canvas = { id: "canvas-specialized-selection" };
  const instance = router.routeAndConstruct({
    constructorFn: UpstreamWebGPURenderer,
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  // Must invoke registered specialized implementation, NOT source constructorFn
  assert.equal(
    sourceConstructorCalls,
    0,
    "Source constructorFn must NOT be invoked when specialized implementation is registered",
  );
  assert.equal(specializedCalls, 1, "Registered specialized implementation must be invoked");
  assert.ok(instance instanceof AdmittedSpecializedRenderer);
  assert.ok(!(instance instanceof UpstreamWebGPURenderer));
  assert.equal(instance.isSpecialized, true);

  // Diagnostics and lock truthfully record SPECIALIZED_WEBGPU
  assert.equal(getRendererRoute(instance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(router.getInstanceRoute(instance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.SPECIALIZED_WEBGPU);
});

test("Defect D: SPECIALIZED_WEBGPU fallback to RETAINED_UPSTREAM when no specialized implementation is registered", () => {
  let sourceConstructorCalls = 0;
  class UpstreamWebGPURenderer {
    constructor(opts) {
      sourceConstructorCalls++;
      this.isUpstreamSource = true;
      this.canvas = opts?.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    // NO specialized implementation registered!
  });

  const canvas = { id: "canvas-specialized-fallback" };
  const instance = router.routeAndConstruct({
    constructorFn: UpstreamWebGPURenderer,
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: "src/app.js:10:5",
  });

  // Must construct using source constructorFn
  assert.equal(sourceConstructorCalls, 1);
  assert.ok(instance instanceof UpstreamWebGPURenderer);

  // MUST truthfully record RETAINED_UPSTREAM, NEVER SPECIALIZED_WEBGPU
  assert.equal(getRendererRoute(instance), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getInstanceRoute(instance), ExecutionRoute.RETAINED_UPSTREAM);

  // Decision must include SPECIALIZATION_UNAVAILABLE reason and NOT specialized-island-admitted
  const decision = getRendererDecision(instance);
  assert.equal(decision.route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(decision.reasons.includes(EscapeReason.SPECIALIZATION_UNAVAILABLE));
  assert.equal(
    decision.reasons.includes("specialized-island-admitted"),
    false,
    "Must not report admission and unavailability together",
  );

  // Canvas lock must be reserved with RETAINED_UPSTREAM before and after constructor
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.RETAINED_UPSTREAM);

  // Decision log records RETAINED_UPSTREAM
  const log = router.getDecisionLog();
  assert.equal(log.length, 1);
  assert.equal(log[0].route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(log[0].reasons.includes(EscapeReason.SPECIALIZATION_UNAVAILABLE));
});

test("Defect D: SPECIALIZED_WEBGPU fallback selects registered RETAINED_UPSTREAM when constructorFn is absent", () => {
  let retainedCalls = 0;
  class MockRetainedRenderer {
    constructor(opts) {
      retainedCalls++;
      this.isRetained = true;
      this.canvas = opts?.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: MockRetainedRenderer,
    },
  });

  const canvas = { id: "canvas-retained-fallback" };
  const instance = router.routeAndConstruct({
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  assert.equal(retainedCalls, 1);
  assert.ok(instance instanceof MockRetainedRenderer);
  assert.equal(getRendererRoute(instance), ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(
    getRendererDecision(instance).reasons.includes(EscapeReason.SPECIALIZATION_UNAVAILABLE),
  );
  assert.equal(
    getRendererDecision(instance).reasons.includes("specialized-island-admitted"),
    false,
    "Must not report admission and unavailability together",
  );
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.RETAINED_UPSTREAM);
});

test("Defect D: SPECIALIZED_WEBGPU throws truthful error before effects when neither specialized nor fallback implementation exists", () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    specializationAvailable: true,
    implementations: {}, // No implementations
  });

  const canvas = { id: "canvas-unsupported-spec" };
  const sharedRt = { id: "rt-unsupported-spec", isMutable: true };

  assert.throws(() => {
    router.routeAndConstruct({
      constructorName: "WebGPURenderer",
      options: { canvas },
      sharedResources: [sharedRt],
      hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      sourceSpan: "src/render.js:5:1",
    });
  }, /Cannot route construction site 'WebGPURenderer' \(src\/render\.js:5:1\) to 'specialized-webgpu': no admitted specialized WebGPU implementation registered\./);

  // Before effects: canvas must NOT be locked
  assert.equal(router.getCanvasLock(canvas), undefined);

  // Before effects: connected groups must NOT be modified or poisoned
  assert.equal(connectedGroups.getGroupMembers("rt-unsupported-spec").length, 0);
  assert.equal(connectedGroups._nodes.has("rt-unsupported-spec"), false);

  // Before effects: decision log must not record a successful route
  assert.equal(router.getDecisionLog().length, 0);
});

test("Defect D: SPECIALIZED_WEBGPU targetImplementation override takes highest precedence", () => {
  class RegisteredSpecialized {
    constructor() {
      this.type = "registered";
    }
  }
  class ExplicitSpecialized {
    constructor() {
      this.type = "explicit";
    }
  }
  class UpstreamSource {
    constructor() {
      this.type = "upstream";
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.SPECIALIZED_WEBGPU]: RegisteredSpecialized,
    },
  });

  const instance = router.routeAndConstruct({
    targetImplementation: ExplicitSpecialized,
    constructorFn: UpstreamSource,
    constructorName: "WebGPURenderer",
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  assert.ok(instance instanceof ExplicitSpecialized);
  assert.equal(instance.type, "explicit");
  assert.equal(getRendererRoute(instance), ExecutionRoute.SPECIALIZED_WEBGPU);
});

test("Defect D: Qualified registered constructor takes precedence over default and bypasses source constructorFn", () => {
  let sourceConstructorCalls = 0;
  class UpstreamWebGPURenderer {
    constructor(opts) {
      sourceConstructorCalls++;
      this.canvas = opts?.canvas;
    }
  }

  class QualifiedSpecializedRenderer {
    constructor(opts) {
      this.isQualified = true;
      this.canvas = opts?.canvas;
    }
  }

  class DefaultSpecializedRenderer {
    constructor(opts) {
      this.isDefault = true;
      this.canvas = opts?.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.SPECIALIZED_WEBGPU]: {
        WebGPURenderer: QualifiedSpecializedRenderer,
        default: DefaultSpecializedRenderer,
      },
    },
  });

  const canvas = { id: "canvas-qualified-precedence" };
  const instance = router.routeAndConstruct({
    constructorFn: UpstreamWebGPURenderer,
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });

  assert.equal(
    sourceConstructorCalls,
    0,
    "Source constructorFn must not be called when qualified replacement is registered",
  );
  assert.ok(
    instance instanceof QualifiedSpecializedRenderer,
    "Must select qualified constructor over default",
  );
  assert.equal(instance.isQualified, true);
  assert.equal(getRendererRoute(instance), ExecutionRoute.SPECIALIZED_WEBGPU);
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.SPECIALIZED_WEBGPU);

  // When registry lacks qualified key, falls back to default specialized registration
  const routerWithDefaultOnly = new RendererConstructionRouter({
    specializationAvailable: true,
    implementations: {
      [ExecutionRoute.SPECIALIZED_WEBGPU]: {
        default: DefaultSpecializedRenderer,
      },
    },
  });
  const instanceDefault = routerWithDefaultOnly.routeAndConstruct({
    constructorName: "WebGPURenderer",
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.ok(
    instanceDefault instanceof DefaultSpecializedRenderer,
    "Must select default constructor when registry lacks qualified key",
  );
  assert.equal(instanceDefault.isDefault, true);
  assert.equal(getRendererRoute(instanceDefault), ExecutionRoute.SPECIALIZED_WEBGPU);
});

test("Defect D: Two calls on same canvas with specializationAvailable=true both construct retained without lock collision", () => {
  let sourceConstructorCalls = 0;
  class UpstreamWebGPURenderer {
    constructor(opts) {
      sourceConstructorCalls++;
      this.callIndex = sourceConstructorCalls;
      this.canvas = opts?.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: true,
    // NO specialized implementation registered!
  });

  const canvas = { id: "canvas-shared-retained" };

  // Call 1: tentative SPECIALIZED_WEBGPU falls back to RETAINED_UPSTREAM and locks canvas to RETAINED_UPSTREAM
  const r1 = router.routeAndConstruct({
    constructorFn: UpstreamWebGPURenderer,
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: "src/app.js:10:1",
  });

  assert.equal(sourceConstructorCalls, 1);
  assert.ok(r1 instanceof UpstreamWebGPURenderer);
  assert.equal(getRendererRoute(r1), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(getRendererDecision(r1).reasons.includes(EscapeReason.SPECIALIZATION_UNAVAILABLE));
  assert.equal(
    getRendererDecision(r1).reasons.includes("specialized-island-admitted"),
    false,
    "Must not report admission and unavailability together",
  );

  // Call 2 on same canvas: early check must not throw RouteLockError against tentative SPECIALIZED_WEBGPU;
  // it must discover no replacement, fall back to RETAINED_UPSTREAM, and succeed matching the canvas lock
  const r2 = router.routeAndConstruct({
    constructorFn: UpstreamWebGPURenderer,
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    sourceSpan: "src/app.js:20:1",
  });

  assert.equal(sourceConstructorCalls, 2);
  assert.ok(r2 instanceof UpstreamWebGPURenderer);
  assert.equal(getRendererRoute(r2), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.RETAINED_UPSTREAM);
  assert.ok(getRendererDecision(r2).reasons.includes(EscapeReason.SPECIALIZATION_UNAVAILABLE));
  assert.equal(
    getRendererDecision(r2).reasons.includes("specialized-island-admitted"),
    false,
    "Must not report admission and unavailability together",
  );
});

test("Defect D: Re-routing canvas from RETAINED_UPSTREAM to SPECIALIZED_WEBGPU when replacement exists throws RouteLockError", () => {
  class UpstreamWebGPURenderer {
    constructor(opts) {
      this.canvas = opts?.canvas;
    }
  }
  class AdmittedSpecializedRenderer {
    constructor(opts) {
      this.canvas = opts?.canvas;
    }
  }

  const router = new RendererConstructionRouter({
    specializationAvailable: false,
  });

  const canvas = { id: "canvas-retained-then-specialized" };

  // Call 1: Locks canvas to RETAINED_UPSTREAM
  router.routeAndConstruct({
    constructorFn: UpstreamWebGPURenderer,
    constructorName: "WebGPURenderer",
    options: { canvas },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getCanvasLock(canvas)?.route, ExecutionRoute.RETAINED_UPSTREAM);

  // Now register specialized implementation and enable specialization
  router.specializationAvailable = true;
  router.registerImplementation(ExecutionRoute.SPECIALIZED_WEBGPU, AdmittedSpecializedRenderer);

  // Call 2 on same canvas: attempts to switch route to SPECIALIZED_WEBGPU -> must throw RouteLockError
  assert.throws(() => {
    router.routeAndConstruct({
      constructorFn: UpstreamWebGPURenderer,
      constructorName: "WebGPURenderer",
      options: { canvas },
      hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      sourceSpan: "src/app.js:30:1",
    });
  }, RouteLockError);
});
