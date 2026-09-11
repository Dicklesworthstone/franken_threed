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

import { ExecutionRoute, RouteLockError } from './route_types.mjs';
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
 * Retrieve route metadata from an instance via external diagnostics or property.
 * @param {object} instance
 * @returns {string | undefined}
 */
export function getRendererRoute(instance) {
  if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
  return INSTANCE_DIAGNOSTICS.get(instance)?.route ?? instance.__f3d_route__;
}

/**
 * Retrieve full decision record from an instance via external diagnostics or property.
 * @param {object} instance
 * @returns {Object | undefined}
 */
export function getRendererDecision(instance) {
  if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
  return INSTANCE_DIAGNOSTICS.get(instance) ?? instance.__f3d_decision__;
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

    // 3. Connect in compatibility groups if sharing resources
    this.connectedGroups.registerRenderer(rendererId, initialDecision.route);
    for (const res of sharedResources) {
      if (typeof res === 'string') {
        this.connectedGroups.recordResourceSharing(rendererId, res, true);
      } else if (Array.isArray(res)) {
        this.connectedGroups.recordResourceSharing(rendererId, res[0], res[1] ?? true);
      } else if (res && typeof res === 'object') {
        const resId = res.id || res.resourceId || res.name;
        const isMutable = res.isMutable !== undefined ? res.isMutable : (res.mutable !== undefined ? res.mutable : true);
        this.connectedGroups.recordResourceSharing(rendererId, resId, isMutable);
      }
    }

    const groupResolutions = this.connectedGroups.resolveGroupRoutes(
      new Map([[rendererId, initialDecision]])
    );
    const resolved = groupResolutions.get(rendererId) || initialDecision;
    const groupId = resolved.groupId || rendererId;

    // 3.5 Preflight check canvas lock: reject re-route attempts immediately before implementation lookup
    const existingLock = this.getCanvasLock(canvasKey);
    if (existingLock && existingLock.route !== resolved.route) {
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
    // Four-tier selection order: targetImplementation, route-conforming constructorFn, registered implementation for the resolved route, truthful throw.
    let targetConstructor = null;

    if (typeof targetImplementation === 'function') {
      targetConstructor = targetImplementation;
    } else if (typeof constructorFn === 'function') {
      // An explicitly supplied constructorFn is invoked for the resolved route
      // when it matches the route's contract.
      if (resolved.route === ExecutionRoute.EXACT_BACKEND) {
        // Exact ownership preserves the source constructor and its backend choice.
        // Opaque access is not permission to replace WebGPURenderer with the
        // legacy WebGLRenderer, including on hosts where upstream falls back.
        targetConstructor = constructorFn;
      } else if (resolved.route === ExecutionRoute.RETAINED_UPSTREAM && constructorName !== 'WebGLRenderer') {
        targetConstructor = constructorFn;
      } else if (resolved.route === ExecutionRoute.GENERAL_WEBGPU && constructorName === 'WebGPURenderer') {
        targetConstructor = constructorFn;
      } else if (resolved.route === ExecutionRoute.SPECIALIZED_WEBGPU && constructorName === 'WebGPURenderer') {
        targetConstructor = constructorFn;
      }
    }

    // Without the source constructor, exact replacements must be qualified by
    // constructor name. The legacy route-level registration is WebGLRenderer only.
    if (!targetConstructor) {
      const mergedImpls = callImplementations
        ? { ...this.implementations, ...callImplementations }
        : this.implementations;

      const routeImpl = mergedImpls[resolved.route];
      if (typeof routeImpl === 'function') {
        if (resolved.route !== ExecutionRoute.EXACT_BACKEND || constructorName === 'WebGLRenderer') {
          targetConstructor = routeImpl;
        }
      } else if (routeImpl && typeof routeImpl === 'object') {
        targetConstructor = routeImpl[constructorName] ||
          (resolved.route !== ExecutionRoute.EXACT_BACKEND ? routeImpl['default'] : null);
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

    // 5. Reserve irreversible canvas lock BEFORE calling constructor
    // (Prevents reentrant route switches on same canvas and preserves lock after constructor throws)
    if (!existingLock) {
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

    // Record committed route in connected groups
    this.connectedGroups.recordCommittedRoute(rendererId, resolved.route);

    // 6. Construct the instance exactly once.
    // Lock is already in place. If targetConstructor throws (even after binding context),
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
   * Retrieve route for a constructed instance via instance WeakMap or property.
   * @param {object} instance
   * @returns {string | undefined}
   */
  getInstanceRoute(instance) {
    if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
    return this._instanceDiagnostics.get(instance)?.route ?? INSTANCE_DIAGNOSTICS.get(instance)?.route ?? instance.__f3d_route__;
  }

  /**
   * Retrieve decision record for a constructed instance via instance WeakMap or property.
   * @param {object} instance
   * @returns {Object | undefined}
   */
  getInstanceDecision(instance) {
    if (!instance || (typeof instance !== 'object' && typeof instance !== 'function')) return undefined;
    return this._instanceDiagnostics.get(instance) ?? INSTANCE_DIAGNOSTICS.get(instance) ?? instance.__f3d_decision__;
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
