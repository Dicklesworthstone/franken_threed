/**
 * @file route_types.mjs
 * Execution route definitions, escape reasons, and route lock errors
 * for FrankenThreeD renderer construction routing (Plan §3.3, §5.1, §6.9).
 */

/**
 * The four execution routes defined in FrankenThreeD Plan §5.1 / §5.2.
 * Never conflate compatibility ownership with acceleration claims.
 */
export const ExecutionRoute = Object.freeze({
  /** Verified accelerated Rust/Wasm/WebGPU execution */
  SPECIALIZED_WEBGPU: 'specialized-webgpu',
  /** Working WebGPU execution when specialization is unavailable */
  GENERAL_WEBGPU: 'general-webgpu',
  /** Full component functionality through retained JS/host code (e.g. CSS2D/SVG/interim) */
  RETAINED_UPSTREAM: 'retained-upstream',
  /** Exact backend component (pinned upstream WebGLRenderer/backend); never credited as acceleration */
  EXACT_BACKEND: 'exact-backend',
});

/**
 * Reasons triggering a route decision or fallback.
 */
export const EscapeReason = Object.freeze({
  EXPLICIT_SOURCE_SELECTION: 'explicit-source-selection',
  NATIVE_CONTEXT_ACCESS: 'native-context-access',
  OPAQUE_GL_ESCAPE: 'opaque-gl-escape',
  HOST_LIMITATION_FALLBACK: 'host-limitation-fallback',
  CONNECTED_GROUP_CONSTRAINT: 'connected-group-constraint',
  SPECIALIZATION_UNAVAILABLE: 'specialization-unavailable',
});

/**
 * Error thrown when an attempt is made to switch the execution route of an already-bound canvas.
 * The backend decision is irreversible for that canvas/renderer lifetime (Plan §3.3).
 */
export class RouteLockError extends Error {
  /**
   * @param {string} canvasId
   * @param {string} existingRoute
   * @param {string} requestedRoute
   * @param {string} [sourceSpan]
   */
  constructor(canvasId, existingRoute, requestedRoute, sourceSpan = 'unknown') {
    super(
      `Irreversible route lock violation: canvas '${canvasId}' is already bound to route '${existingRoute}'; ` +
      `cannot switch to route '${requestedRoute}' at ${sourceSpan}.`
    );
    this.name = 'RouteLockError';
    this.canvasId = canvasId;
    this.existingRoute = existingRoute;
    this.requestedRoute = requestedRoute;
    this.sourceSpan = sourceSpan;
  }
}
