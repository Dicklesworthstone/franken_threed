/**
 * @file index.mjs
 * Main entry point for tools/compat renderer construction routing.
 */

export {
  evaluateGraphRoutes,
  extractGraphRoutingFacts,
  isInternalLibraryModule,
  prepareRouteInputs,
} from "../ingest/route_bridge.mjs";
export { ConnectedCompatibilityGroups } from "./connected_groups.mjs";
export {
  getRendererDecision,
  getRendererRoute,
  RendererConstructionRouter,
} from "./construction_adapter.mjs";
export {
  createExactBackendForRenderer,
  createExactBackendRouter,
  createExactWebGLRenderer,
  ExactWebGLRenderer,
  getPinnedWebGLBackend,
  PinnedWebGLRenderer,
  registerExactBackend,
} from "./exact_backend.mjs";
export {
  decideRendererRoute,
  evaluateModuleGraphRoutes,
} from "./route_decider.mjs";
export { formatRouteReport, generateRouteReport } from "./route_report.mjs";
export { EscapeReason, ExecutionRoute, RouteLockError } from "./route_types.mjs";
