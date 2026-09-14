/**
 * @file construction_adapter.mjs
 * Executable renderer construction routing adapter (Plan §3.3, §5.1, §6.9).
 *
 * Intercepts renderer construction sites synchronously, decides the execution route
 * before canvas context acquisition, locks the canvas irreversibly BEFORE invoking
 * effectful constructors, tracks connected compatibility groups across construction orders,
 * dispatches to actual admitted implementations, retains canvas locks after binding throws,
 * preserves native object shapes via external diagnostics, and guarantees single execution.
 */

import { ExecutionRoute, EscapeReason, RouteLockError } from './route_types.mjs';
import { ConnectedCompatibilityGroups } from './connected_groups.mjs';
import { decideRendererRoute } from './route_decider.mjs';

/**
 * Module-level weak lock registry ensuring canvas objects remain irreversibly
 * bound to their execution route across their lifetime.
 * @type {WeakMap<object, { route: string, rendererId: string, sourceSpan: string }>}
 */
const GLOBAL_CANVAS_OBJECT_LOCKS = new WeakMap();

/**
 * External diagnostics registry associating constructed instances with their
 * route decisions without mutating native/sealed/frozen object shapes.
 * @type {WeakMap<object, Object>}
 */
const INSTANCE_DIAGNOSTICS = new WeakMap();

/**
 * Weak registry mapping unnamed resource objects to stable unique IDs.
 * @type {WeakMap<object, string>}
 */
const RESOURCE_OBJECT_IDS = new WeakMap();
let resourceObjectCounter = 0;

/**
 * Resolve a unique string identifier for a shared resource without mutating its shape.
 * Class instances use object identity; their names and numeric IDs are not globally unique.
 * Plain descriptors may supply symbolic IDs or resource wrappers ({ resource, target }).
 * @param {any} res
 * @returns {string}
 */
export function resolveResourceId(res) {
  if (typeof res === 'string') return res;
  if (!res || (typeof res !== 'object' && typeof res !== 'function')) return String(res);
  const prototype = Object.getPrototypeOf(res);
  if (prototype === Object.prototype || prototype === null) {
    if (res.id !== undefined && res.id !== null) return String(res.id);
    if (res.resourceId !== undefined && res.resourceId !== null) return String(res.resourceId);
    if (typeof res.name === 'string' && res.name.length > 0) return res.name;
    if (res.resource && typeof res.resource === 'object') return resolveResourceId(res.resource);
    if (res.target && typeof res.target === 'object') return resolveResourceId(res.target);
  }
  let id = RESOURCE_OBJECT_IDS.get(res);
  if (!id) {
    id = `resource-obj-${++resourceObjectCounter}`;
    RESOURCE_OBJECT_IDS.set(res, id);
  }
  return id;
}

/**
 * Retrieve route metadata from an instance via external diagnostics registry.
 * Does not read properties from instance to preserve native/sealed/frozen/proxy shapes.
 * @param {object} instance
 * @returns {string | undefined}
 */
export function getRendererRoute(instance) {
  if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
  return INSTANCE_DIAGNOSTICS.get(instance)?.route;
}

/**
 * Retrieve full decision record from an instance via external diagnostics registry.
 * Does not read properties from instance to preserve native/sealed/frozen/proxy shapes.
 * @param {object} instance
 * @returns {Object | undefined}
 */
export function getRendererDecision(instance) {
  if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
  return INSTANCE_DIAGNOSTICS.get(instance);
}

export class RendererConstructionRouter {
  /**
   * @param {Object} [config]
   * @param {ConnectedCompatibilityGroups} [config.connectedGroups]
   * @param {boolean} [config.specializationAvailable]
   * @param {Record<string, Function | Record<string, Function>>} [config.implementations]
   *   Mapping of route (e.g. ExecutionRoute.EXACT_BACKEND) to constructor function or factory,
   *   or to a sub-map { [constructorName]: Function }.
   */
  constructor({
    connectedGroups = new ConnectedCompatibilityGroups(),
    specializationAvailable = false,
    implementations = {},
  } = {}) {
    this.connectedGroups = connectedGroups;
    this.specializationAvailable = specializationAvailable;
    /** @type {Record<string, Function | Record<string, Function>>} */
    this.implementations = { ...implementations };
    /** @type {Map<any, { route: string, rendererId: string, sourceSpan: string }>} */
    this._canvasLocks = new Map();
    /** @type {Map<string, Object>} */
    this._decisions = new Map();
    /** @type {Array<{ site: string, span: string, route: string, reasons: string[], group: string }>} */
    this._decisionLog = [];
    /** @type {Array<{ renderer: string, route: string, submissions: number }>} */
    this._attributionLog = [];
    /** @type {WeakMap<object, Object>} */
    this._instanceDiagnostics = new WeakMap();
    this._rendererCounter = 0;
  }

  /**
   * Register an admitted implementation for an execution route.
   * @param {string} route - e.g. ExecutionRoute.EXACT_BACKEND
   * @param {Function} implementation - constructor function or factory
   * @param {Object} [options]
   * @param {string} [options.constructorName] - optional target constructor name qualifier
   */
  registerImplementation(route, implementation, { constructorName } = {}) {
    if (typeof implementation !== 'function') {
      throw new TypeError(`Implementation for route '${route}' must be a constructor function or factory`);
    }
    if (constructorName) {
      if (!this.implementations[route] || typeof this.implementations[route] === 'function') {
        this.implementations[route] = {};
      }
      this.implementations[route][constructorName] = implementation;
    } else {
      this.implementations[route] = implementation;
    }
  }

  /**
   * Helper to obtain a string identifier for a canvas key.
   * @param {any} canvasKey
   * @returns {string}
   */
  _canvasKeyString(canvasKey) {
    if (typeof canvasKey === 'string') return canvasKey;
    if (canvasKey && typeof canvasKey === 'object') {
      if (canvasKey.id) return `#${canvasKey.id}`;
      if (canvasKey.tagName) return `<${canvasKey.tagName.toLowerCase()}>`;
    }
    return 'anonymous-canvas';
  }

  /**
   * Route and construct a renderer instance.
   *
   * @param {Object} params
   * @param {Function} [params.constructorFn] - The constructor class or factory passed at callsite
   * @param {string} [params.constructorName] - e.g. 'WebGLRenderer', 'WebGPURenderer'
   * @param {Object} [params.options] - constructor arguments (e.g. { canvas, forceWebGL })
   * @param {Object} [params.analysis] - AST analysis facts from RubyCrane
   * @param {Object} [params.hostCapabilities] - runtime environment features
   * @param {string} [params.sourceSpan] - callsite span
   * @param {string[]} [params.sharedResources] - resource IDs shared with other renderers
   * @param {Function} [params.targetImplementation] - explicit implementation to construct
   * @param {Record<string, Function | Record<string, Function>>} [params.implementations] - per-call implementation overrides
   * @param {boolean} [params.isFactory] - true if target should be invoked as factory function rather than new
   * @returns {Object} The constructed renderer instance decorated with route metadata
   */
  routeAndConstruct(params = {}) {
    const {
      constructorFn,
      constructorName = constructorFn?.name || 'WebGLRenderer',
      options = {},
      analysis = {},
      hostCapabilities,
      sourceSpan = 'unknown:0:0',
      sharedResources = [],
      targetImplementation,
      implementations: callImplementations,
      isFactory = false,
    } = params;

    // 1. Preflight validation: validate constructor arguments upfront before any lock or side effects
    if (constructorFn !== undefined && typeof constructorFn !== 'function') {
      throw new TypeError(`Invalid constructorFn: expected a constructor function or factory, got ${typeof constructorFn}`);
    }
    if (targetImplementation !== undefined && typeof targetImplementation !== 'function') {
      throw new TypeError(`Invalid targetImplementation: expected a constructor function or factory, got ${typeof targetImplementation}`);
    }

    const rendererId = `renderer-${++this._rendererCounter}`;
    const canvasKey = options.canvas || `auto-canvas-${this._rendererCounter}`;
    const canvasKeyStr = this._canvasKeyString(canvasKey);

    // 2. Initial decision based on constructor, options, analysis, host
    const initialDecision = decideRendererRoute({
      constructorName,
      options,
      analysis,
      hostCapabilities,
      specializationAvailable: this.specializationAvailable,
      sourceSpan,
    });

    // 3. Parse shared resources once and preview route without mutating connected groups
    const parsedResources = [];
    for (const res of sharedResources) {
      if (typeof res === 'string') {
        parsedResources.push({ id: res, isMutable: true });
      } else if (Array.isArray(res)) {
        parsedResources.push({ id: resolveResourceId(res[0]), isMutable: res[1] ?? true });
      } else if (res && typeof res === 'object') {
        const resId = resolveResourceId(res);
        const isMutable = res.isMutable !== undefined ? res.isMutable : (res.mutable !== undefined ? res.mutable : true);
        parsedResources.push({ id: resId, isMutable });
      }
    }

    let resolved = this.connectedGroups.previewRoute(initialDecision, parsedResources);
    let groupId = resolved.groupId || rendererId;

    // 3.5 Preflight check canvas lock: reject known-incompatible re-route attempts immediately before implementation lookup.
    // Defer check only when resolved is tentative SPECIALIZED_WEBGPU and existing lock is RETAINED_UPSTREAM,
    // since specialization may fall back to retained once implementations are inspected.
    // Known-incompatible mismatches are rejected immediately before any effectful getters are read.
    const existingLock = this.getCanvasLock(canvasKey);
    const isPotentiallyCompatible = resolved.route === ExecutionRoute.SPECIALIZED_WEBGPU &&
      existingLock?.route === ExecutionRoute.RETAINED_UPSTREAM;
    if (existingLock && existingLock.route !== resolved.route && !isPotentiallyCompatible) {
      const rejectedEvent = Object.freeze({
        site: constructorName,
        span: sourceSpan,
        route: resolved.route,
        reasons: Object.freeze([...resolved.reasons]),
        group: groupId,
      });
      this._decisionLog.push(rejectedEvent);
      throw new RouteLockError(canvasKeyStr, existingLock.route, resolved.route, sourceSpan);
    }

    // 4. Preflight validate implementation selection for resolved route
    // Cache route implementation lookups so effectful user getters are never repeated
    const cachedRouteImpls = new Map();
    const getRouteImplementation = (route) => {
      if (cachedRouteImpls.has(route)) {
        return cachedRouteImpls.get(route);
      }
      let impl;
      if (callImplementations && route in callImplementations) {
        impl = callImplementations[route];
      }
      if (impl === undefined && this.implementations && route in this.implementations) {
        impl = this.implementations[route];
      }
      cachedRouteImpls.set(route, impl);
      return impl;
    };

    // Four-tier selection order: targetImplementation, route-conforming constructorFn, registered implementation for the resolved route, truthful throw.
    const selectConstructorForRoute = (route) => {
      let ctor = null;
      if (typeof targetImplementation === 'function') {
        ctor = targetImplementation;
      } else if (typeof constructorFn === 'function') {
        // An explicitly supplied constructorFn is invoked for the resolved route
        // when it matches the route's contract.
        if (route === ExecutionRoute.EXACT_BACKEND) {
          // Exact ownership preserves the source constructor and its backend choice.
          // Opaque access is not permission to replace WebGPURenderer with the
          // legacy WebGLRenderer, including on hosts where upstream falls back.
          ctor = constructorFn;
        } else if (route === ExecutionRoute.RETAINED_UPSTREAM && constructorName !== 'WebGLRenderer') {
          ctor = constructorFn;
        } else if (route === ExecutionRoute.GENERAL_WEBGPU && constructorName === 'WebGPURenderer') {
          ctor = constructorFn;
        }
      }

      // Without the source constructor, exact replacements must be qualified by
      // constructor name. The legacy route-level registration is WebGLRenderer only.
      if (!ctor) {
        const routeImpl = getRouteImplementation(route);
        if (typeof routeImpl === 'function') {
          if (route !== ExecutionRoute.EXACT_BACKEND || constructorName === 'WebGLRenderer') {
            ctor = routeImpl;
          }
        } else if (routeImpl && typeof routeImpl === 'object') {
          ctor = routeImpl[constructorName] ||
            (route !== ExecutionRoute.EXACT_BACKEND ? routeImpl['default'] : null);
        }
      }
      return ctor;
    };

    let currentDecision = initialDecision;
    let targetConstructor = selectConstructorForRoute(resolved.route);

    // If SPECIALIZED_WEBGPU has no registered specialized implementation, fall back truthfully
    // to RETAINED_UPSTREAM before any effects (canvas lock reservation, connected group mutation, constructor invocation).
    if (resolved.route === ExecutionRoute.SPECIALIZED_WEBGPU && !targetConstructor) {
      const fallbackConstructor = selectConstructorForRoute(ExecutionRoute.RETAINED_UPSTREAM);
      if (fallbackConstructor) {
        const filteredReasons = resolved.reasons.filter((r) => r !== 'specialized-island-admitted');
        const fallbackReasons = filteredReasons.includes(EscapeReason.SPECIALIZATION_UNAVAILABLE)
          ? Object.freeze([...filteredReasons])
          : Object.freeze([...filteredReasons, EscapeReason.SPECIALIZATION_UNAVAILABLE]);
        currentDecision = Object.freeze({
          ...initialDecision,
          route: ExecutionRoute.RETAINED_UPSTREAM,
          reasons: fallbackReasons,
        });
        resolved = this.connectedGroups.previewRoute(currentDecision, parsedResources);
        groupId = resolved.groupId || rendererId;
        if (resolved.route === ExecutionRoute.EXACT_BACKEND) {
          targetConstructor = selectConstructorForRoute(resolved.route);
        } else {
          targetConstructor = fallbackConstructor;
        }
      }
    }

    // 4.5 Revalidate current shared-group route after implementation selection (without rereading user getters).
    // Group escalation is monotonic to EXACT_BACKEND under supported construction flow,
    // so bound selection to initial then exact; no generic retries/transaction framework.
    if (resolved.route !== ExecutionRoute.EXACT_BACKEND) {
      const revalidated = this.connectedGroups.previewRoute(currentDecision, parsedResources);
      if (revalidated.route === ExecutionRoute.EXACT_BACKEND) {
        resolved = revalidated;
        groupId = resolved.groupId || rendererId;
        targetConstructor = selectConstructorForRoute(resolved.route);
        const revalidatedAfterEscalation = this.connectedGroups.previewRoute(currentDecision, parsedResources);
        if (revalidatedAfterEscalation.route !== resolved.route) {
          throw new Error(
            `Cannot route construction site '${constructorName}' (${sourceSpan}): ` +
            `route changed from '${resolved.route}' to '${revalidatedAfterEscalation.route}' during preflight implementation selection due to reentrant connected group constraints.`
          );
        }
      } else if (revalidated.route !== resolved.route) {
        throw new Error(
          `Cannot route construction site '${constructorName}' (${sourceSpan}): ` +
          `route changed from '${resolved.route}' to '${revalidated.route}' during preflight implementation selection due to reentrant connected group constraints.`
        );
      }
    }

    // If still no implementation, throw truthful route-specific error
    if (!targetConstructor) {
      if (resolved.route === ExecutionRoute.RETAINED_UPSTREAM) {
        throw new Error(
          `Cannot route construction site '${constructorName}' (${sourceSpan}) to '${resolved.route}': ` +
          `caller-supplied constructor is not a valid retained upstream implementation, and no admitted implementation is registered.`
        );
      } else if (resolved.route === ExecutionRoute.EXACT_BACKEND) {
        throw new Error(
          `Cannot route construction site '${constructorName}' (${sourceSpan}) to '${resolved.route}': ` +
          `no admitted exact backend implementation registered (supplied constructor '${constructorName}' does not implement exact backend).`
        );
      } else if (resolved.route === ExecutionRoute.SPECIALIZED_WEBGPU) {
        throw new Error(
          `Cannot route construction site '${constructorName}' (${sourceSpan}) to '${resolved.route}': ` +
          `no admitted specialized WebGPU implementation registered.`
        );
      } else if (resolved.route === ExecutionRoute.GENERAL_WEBGPU) {
        throw new Error(
          `Cannot route construction site '${constructorName}' (${sourceSpan}) to '${resolved.route}': ` +
          `no admitted general WebGPU implementation registered.`
        );
      }
    }

    // Final safety check: targetConstructor must be a valid function
    if (typeof targetConstructor !== 'function') {
      throw new TypeError(
        `Cannot construct renderer: no valid constructor function available for route '${resolved.route}' and constructor '${constructorName}'.`
      );
    }

    // Revalidate route and canvas lock immediately before committing
    const finalRevalidation = this.connectedGroups.previewRoute(currentDecision, parsedResources);
    if (finalRevalidation.route !== resolved.route) {
      throw new Error(
        `Cannot route construction site '${constructorName}' (${sourceSpan}): ` +
        `route changed from '${resolved.route}' to '${finalRevalidation.route}' during preflight implementation selection due to reentrant connected group constraints.`
      );
    }
    resolved = finalRevalidation;
    groupId = resolved.groupId || rendererId;

    const currentLock = this.getCanvasLock(canvasKey);
    if (currentLock && currentLock.route !== resolved.route) {
      const rejectedEvent = Object.freeze({
        site: constructorName,
        span: sourceSpan,
        route: resolved.route,
        reasons: Object.freeze([...resolved.reasons]),
        group: groupId,
      });
      this._decisionLog.push(rejectedEvent);
      throw new RouteLockError(canvasKeyStr, currentLock.route, resolved.route, sourceSpan);
    }

    // 5. Commit to connected groups and reserve irreversible canvas lock BEFORE calling constructor
    this.connectedGroups.registerRenderer(rendererId, resolved.route);
    for (const { id: resId, isMutable } of parsedResources) {
      this.connectedGroups.recordResourceSharing(rendererId, resId, isMutable);
    }
    this.connectedGroups.recordCommittedRoute(rendererId, resolved.route);

    if (!currentLock) {
      const lockEntry = {
        route: resolved.route,
        rendererId,
        sourceSpan,
      };
      this._canvasLocks.set(canvasKey, lockEntry);
      if (typeof canvasKey === 'object' && canvasKey !== null) {
        GLOBAL_CANVAS_OBJECT_LOCKS.set(canvasKey, lockEntry);
      }
    }

    // Emit route decision event for this construction site
    const decisionEvent = Object.freeze({
      site: constructorName,
      span: sourceSpan,
      route: resolved.route,
      reasons: Object.freeze([...resolved.reasons]),
      group: groupId,
    });
    this._decisionLog.push(decisionEvent);

    // 6. Construct the instance exactly once.
    // Lock is already in place. Once constructor begins, canvas locks and committed routes
    // are irreversible: if targetConstructor throws (even after binding context),
    // the original exception propagates directly and the canvas lock remains permanently retained.
    let instance;
    if (isFactory) {
      instance = targetConstructor(options);
    } else {
      instance = new targetConstructor(options);
    }

    if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) {
      throw new TypeError(
        `Renderer constructor for route '${resolved.route}' must return an object, got ${typeof instance}`
      );
    }

    // 6.5 Post-construction registration of internally-created canvas (Plan §3.4, §6.9, Bead 6mv.5).
    // For known native result own data properties, inspect descriptor/value directly without invoking
    // observable getters on subclasses or factories, preserving throwing/missing getter contracts.
    let exposedCanvas = null;
    const ownDomDesc = Object.getOwnPropertyDescriptor(instance, 'domElement');
    if (ownDomDesc && 'value' in ownDomDesc && typeof ownDomDesc.value === 'object' && ownDomDesc.value !== null) {
      exposedCanvas = ownDomDesc.value;
    } else {
      const ownBackendDesc = Object.getOwnPropertyDescriptor(instance, 'backend');
      if (ownBackendDesc && 'value' in ownBackendDesc && ownBackendDesc.value && typeof ownBackendDesc.value === 'object') {
        const backendDomDesc = Object.getOwnPropertyDescriptor(ownBackendDesc.value, 'domElement');
        if (backendDomDesc && 'value' in backendDomDesc && typeof backendDomDesc.value === 'object' && backendDomDesc.value !== null) {
          exposedCanvas = backendDomDesc.value;
        }
      }
    }

    if (exposedCanvas) {
      const lockEntry = {
        route: resolved.route,
        rendererId,
        sourceSpan,
      };
      if (!this._canvasLocks.has(exposedCanvas)) {
        this._canvasLocks.set(exposedCanvas, lockEntry);
      }
      if (!GLOBAL_CANVAS_OBJECT_LOCKS.has(exposedCanvas)) {
        GLOBAL_CANVAS_OBJECT_LOCKS.set(exposedCanvas, lockEntry);
      }
    }

    // 7. Store decision record and diagnostics externally in WeakMaps to preserve native object shape
    const decisionRecord = Object.freeze({
      rendererId,
      route: resolved.route,
      reasons: Object.freeze([...resolved.reasons]),
      groupId,
      canvas: canvasKeyStr,
      sourceSpan,
      timestamp: Date.now(),
    });

    this._decisions.set(rendererId, decisionRecord);
    INSTANCE_DIAGNOSTICS.set(instance, decisionRecord);
    this._instanceDiagnostics.set(instance, decisionRecord);

    return instance;
  }

  /**
   * Check if a canvas is currently locked.
   * @param {any} canvasKey
   * @returns {{ route: string, rendererId: string, sourceSpan: string } | undefined}
   */
  getCanvasLock(canvasKey) {
    if (this._canvasLocks.has(canvasKey)) {
      return this._canvasLocks.get(canvasKey);
    }
    if (typeof canvasKey === 'object' && canvasKey !== null) {
      return GLOBAL_CANVAS_OBJECT_LOCKS.get(canvasKey);
    }
    return undefined;
  }

  /**
   * Retrieve route for a constructed instance via instance WeakMap.
   * Does not read properties from instance to preserve native/sealed/frozen/proxy shapes.
   * @param {object} instance
   * @returns {string | undefined}
   */
  getInstanceRoute(instance) {
    if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
    return this._instanceDiagnostics.get(instance)?.route ?? INSTANCE_DIAGNOSTICS.get(instance)?.route;
  }

  /**
   * Retrieve decision record for a constructed instance via instance WeakMap.
   * Does not read properties from instance to preserve native/sealed/frozen/proxy shapes.
   * @param {object} instance
   * @returns {Object | undefined}
   */
  getInstanceDecision(instance) {
    if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
    return this._instanceDiagnostics.get(instance) ?? INSTANCE_DIAGNOSTICS.get(instance);
  }

  /**
   * Retrieve all recorded route decisions.
   * @returns {Object[]}
   */
  getDecisions() {
    return Array.from(this._decisions.values());
  }

  /**
   * Retrieve route decision events in chronological order.
   * @returns {Array<{ site: string, span: string, route: string, reasons: string[], group: string }>}
   */
  getDecisionLog() {
    return [...this._decisionLog];
  }

  /**
   * Retrieve runtime attribution events in chronological order.
   * @returns {Array<{ renderer: string, route: string, submissions: number }>}
   */
  getAttributionLog() {
    return [...this._attributionLog];
  }
}
