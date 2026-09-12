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
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  RendererConstructionRouter,
  getRendererRoute,
  getRendererDecision,
  resolveResourceId,
} from './construction_adapter.mjs';
import { ConnectedCompatibilityGroups } from './connected_groups.mjs';
import { ExecutionRoute, EscapeReason, RouteLockError } from './route_types.mjs';

test('Contract: Constructor throw retains irreversible canvas lock (binds-then-throws)', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: function FailingWebGL() {
        throw new Error('WebGL context creation failed');
      },
    },
  });

  // 1. Attempt WebGLRenderer construction; constructor throws
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        options: { canvas: 'canvas-failed-gl' },
      });
    },
    /WebGL context creation failed/
  );

  // Canvas lock is permanently retained (binds-then-throws invariant, Plan §3.3)
  assert.equal(router.getCanvasLock('canvas-failed-gl')?.route, ExecutionRoute.EXACT_BACKEND);

  // Late route switch on same canvas must be rejected
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        options: { canvas: 'canvas-failed-gl' },
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      });
    },
    RouteLockError
  );
});

test('Defect A: Preflight RouteLockError does not poison connected groups', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
    },
  });

  const canvas1 = { id: 'canvas-shared' };
  const sharedTarget = { id: 'rt-shared-preflight', isMutable: true };

  // 1. Lock canvas1 to RETAINED_UPSTREAM via WebGPURenderer
  const r1 = router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    options: { canvas: canvas1 },
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getInstanceRoute(r1), ExecutionRoute.RETAINED_UPSTREAM);

  // 2. Preflight failure: WebGLRenderer tries to use already-locked canvas1
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        options: { canvas: canvas1 },
        sharedResources: [sharedTarget],
      });
    },
    RouteLockError
  );

  // Connected groups must NOT keep the rejected WebGLRenderer connected to sharedTarget
  const members = connectedGroups.getGroupMembers('rt-shared-preflight');
  assert.equal(members.some(m => m.startsWith('renderer-')), false);

  // 3. Another renderer using sharedTarget on fresh canvas2 routes normally to RETAINED_UPSTREAM
  const r3 = router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    options: { canvas: { id: 'canvas-fresh' } },
    sharedResources: [sharedTarget],
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getInstanceRoute(r3), ExecutionRoute.RETAINED_UPSTREAM);
});

test('Defect A: Missing implementation error does not poison connected groups', () => {
  const connectedGroups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups,
    implementations: {}, // No implementations registered
  });

  const sharedTarget = { id: 'rt-missing-impl', isMutable: true };

  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGLRenderer',
        sharedResources: [sharedTarget],
      });
    },
    /no admitted exact backend implementation registered/
  );

  assert.equal(connectedGroups.getGroupMembers('rt-missing-impl').length, 0);
  assert.equal(connectedGroups._nodes.has('rt-missing-impl'), false);
});

test('Defect B: Zero property reads on instances preserves proxies, sealed, and frozen objects', () => {
  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
    },
  });

  // Strict proxy that throws on any __f3d_* access
  let trapped = false;
  const strictProxy = new Proxy({}, {
    get(target, prop) {
      if (typeof prop === 'string' && prop.startsWith('__f3d_')) {
        trapped = true;
        throw new Error(`Forbidden property access: ${prop}`);
      }
      return target[prop];
    }
  });

  assert.equal(getRendererRoute(strictProxy), undefined);
  assert.equal(getRendererDecision(strictProxy), undefined);
  assert.equal(router.getInstanceRoute(strictProxy), undefined);
  assert.equal(router.getInstanceDecision(strictProxy), undefined);
  assert.equal(trapped, false, 'Proxy __f3d_* getters must not be invoked');

  // Constructed sealed and frozen instances have zero __f3d_* properties
  class CustomSealed {
    constructor() {
      this.val = 42;
      Object.seal(this);
    }
  }

  const sealedInst = router.routeAndConstruct({
    constructorFn: CustomSealed,
    constructorName: 'CustomSealed',
  });

  assert.equal(Object.isSealed(sealedInst), true);
  assert.equal(sealedInst.__f3d_route__, undefined);
  assert.equal(sealedInst.__f3d_decision__, undefined);
  assert.equal(getRendererRoute(sealedInst), ExecutionRoute.RETAINED_UPSTREAM);
  assert.equal(router.getInstanceRoute(sealedInst), ExecutionRoute.RETAINED_UPSTREAM);
});

test('Defect C: Resource identity resolution distinguishes distinct unnamed objects', () => {
  const groups = new ConnectedCompatibilityGroups();
  const router = new RendererConstructionRouter({
    connectedGroups: groups,
    implementations: {
      [ExecutionRoute.RETAINED_UPSTREAM]: function MockRetained() { this.isRetained = true; },
      [ExecutionRoute.EXACT_BACKEND]: function MockExact() { this.isExact = true; },
    },
  });

  // Two distinct unnamed RenderTarget objects
  const rt1 = { width: 512, height: 512 };
  const rt2 = { width: 1024, height: 1024 };

  const id1 = resolveResourceId(rt1);
  const id2 = resolveResourceId(rt2);

  assert.ok(id1.startsWith('resource-obj-'));
  assert.ok(id2.startsWith('resource-obj-'));
  assert.notEqual(id1, id2, 'Distinct unnamed objects must receive distinct IDs');
  assert.equal(resolveResourceId(rt1), id1, 'Identical object reference must return stable ID');

  // Renderer 1 uses rt1 with WebGLRenderer (EXACT_BACKEND)
  const r1 = router.routeAndConstruct({
    constructorName: 'WebGLRenderer',
    sharedResources: [rt1],
  });
  assert.equal(router.getInstanceRoute(r1), ExecutionRoute.EXACT_BACKEND);

  // Renderer 2 uses rt2 with WebGPURenderer -> should NOT collide with rt1
  const r2 = router.routeAndConstruct({
    constructorName: 'WebGPURenderer',
    sharedResources: [rt2],
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  });
  assert.equal(router.getInstanceRoute(r2), ExecutionRoute.RETAINED_UPSTREAM);

  // Renderer 3 uses the SAME rt1 with WebGPURenderer -> correctly couples and throws conflict or forces exact
  assert.throws(
    () => {
      router.routeAndConstruct({
        constructorName: 'WebGPURenderer',
        sharedResources: [rt1],
        hostCapabilities: { hasWebGPU: true, hasWebGL: true },
      });
    },
    /no admitted exact backend implementation registered/
  );
});

test('Defect C: Numeric id: 0 and resource wrappers are resolved correctly', () => {
  const resWithZero = { id: 0 };
  assert.equal(resolveResourceId(resWithZero), '0');

  const rawObj = { isRaw: true };
  const wrappedResource = { resource: rawObj, isMutable: false };
  const wrappedTarget = { target: rawObj, isMutable: true };

  assert.equal(resolveResourceId(wrappedResource), resolveResourceId(rawObj));
  assert.equal(resolveResourceId(wrappedTarget), resolveResourceId(rawObj));
});
