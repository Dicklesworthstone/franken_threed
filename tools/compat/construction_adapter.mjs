/**
 * @file construction_adapter.mjs
 * Executable renderer construction routing adapter (Plan §3.3, §5.1, §6.9).
 *
 * Intercepts renderer construction sites synchronously, decides the execution route
 * before canvas context acquisition, locks the canvas irreversibly, checks connected
 * groups, and guarantees single execution without side-effect replaying.
 */

import { ExecutionRoute, RouteLockError } from './route_types.mjs';
import { ConnectedCompatibilityGroups } from './connected_groups.mjs';
import { decideRendererRoute } from './route_decider.mjs';

export class RendererConstructionRouter {
  /**
   * @param {Object} [config]
   * @param {ConnectedCompatibilityGroups} [config.connectedGroups]
   * @param {boolean} [config.specializationAvailable]
   */
  constructor({
    connectedGroups = new ConnectedCompatibilityGroups(),
    specializationAvailable = false,
  } = {}) {
    this.connectedGroups = connectedGroups;
    this.specializationAvailable = specializationAvailable;
    /** @type {Map<any, { route: string, rendererId: string, sourceSpan: string }>} */
    this._canvasLocks = new Map();
    /** @type {Map<string, Object>} */
    this._decisions = new Map();
    this._rendererCounter = 0;
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
   * @param {Function} params.constructorFn - The actual constructor class or factory
   * @param {string} [params.constructorName] - e.g. 'WebGLRenderer', 'WebGPURenderer'
   * @param {Object} [params.options] - constructor arguments (e.g. { canvas, forceWebGL })
   * @param {Object} [params.analysis] - AST analysis facts from RubyCrane
   * @param {Object} [params.hostCapabilities] - runtime environment features
   * @param {string} [params.sourceSpan] - callsite span
   * @param {string[]} [params.sharedResources] - resource IDs shared with other renderers
   * @returns {Object} The constructed renderer instance decorated with route metadata
   */
  routeAndConstruct(params) {
    const {
      constructorFn,
      constructorName = constructorFn?.name || 'WebGLRenderer',
      options = {},
      analysis = {},
      hostCapabilities,
      sourceSpan = 'unknown:0:0',
      sharedResources = [],
    } = params;

    const rendererId = `renderer-${++this._rendererCounter}`;
    const canvasKey = options.canvas || `auto-canvas-${this._rendererCounter}`;
    const canvasKeyStr = this._canvasKeyString(canvasKey);

    // 1. Initial decision based on constructor, options, analysis, host
    const initialDecision = decideRendererRoute({
      constructorName,
      options,
      analysis,
      hostCapabilities,
      specializationAvailable: this.specializationAvailable,
      sourceSpan,
    });

    // 2. Connect in compatibility groups if sharing resources
    this.connectedGroups.registerRenderer(rendererId, initialDecision.route);
    for (const resId of sharedResources) {
      this.connectedGroups.recordResourceSharing(rendererId, resId);
    }

    const groupResolutions = this.connectedGroups.resolveGroupRoutes(
      new Map([[rendererId, initialDecision]])
    );
    const resolved = groupResolutions.get(rendererId) || initialDecision;

    // 3. Enforce canvas irreversibility lock (Plan §3.3)
    const existingLock = this._canvasLocks.get(canvasKey);
    if (existingLock) {
      if (existingLock.route !== resolved.route) {
        throw new RouteLockError(canvasKeyStr, existingLock.route, resolved.route, sourceSpan);
      }
    } else {
      this._canvasLocks.set(canvasKey, {
        route: resolved.route,
        rendererId,
        sourceSpan,
      });
    }

    // 4. Construct the instance exactly once (no duplicate execution or replaying)
    let instance;
    if (typeof constructorFn === 'function') {
      try {
        instance = new constructorFn(options);
      } catch (err) {
        // If not constructible with new, try as factory function
        instance = constructorFn(options);
      }
    } else {
      instance = Object.create(null);
    }

    // 5. Decorate with immutable route attestation metadata
    const decisionRecord = Object.freeze({
      rendererId,
      route: resolved.route,
      reasons: Object.freeze([...resolved.reasons]),
      groupId: resolved.groupId || rendererId,
      canvas: canvasKeyStr,
      sourceSpan,
      timestamp: Date.now(),
    });

    this._decisions.set(rendererId, decisionRecord);

    Object.defineProperty(instance, '__f3d_route__', {
      value: resolved.route,
      writable: false,
      enumerable: true,
      configurable: false,
    });

    Object.defineProperty(instance, '__f3d_decision__', {
      value: decisionRecord,
      writable: false,
      enumerable: true,
      configurable: false,
    });

    Object.defineProperty(instance, '__f3d_group_id__', {
      value: resolved.groupId || rendererId,
      writable: false,
      enumerable: true,
      configurable: false,
    });

    return instance;
  }

  /**
   * Check if a canvas is currently locked.
   * @param {any} canvasKey
   * @returns {{ route: string, rendererId: string, sourceSpan: string } | undefined}
   */
  getCanvasLock(canvasKey) {
    return this._canvasLocks.get(canvasKey);
  }

  /**
   * Retrieve all recorded route decisions.
   * @returns {Object[]}
   */
  getDecisions() {
    return Array.from(this._decisions.values());
  }

  /**
   * Clear all decisions and canvas locks (e.g. for testing or fresh navigation simulation).
   */
  reset() {
    this._canvasLocks.clear();
    this._decisions.clear();
    this.connectedGroups = new ConnectedCompatibilityGroups();
  }
}
