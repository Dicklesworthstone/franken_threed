/**
 * @file index.mjs
 * Main entry point for tools/compat renderer construction routing.
 */

export { ExecutionRoute, EscapeReason, RouteLockError } from './route_types.mjs';
export { ConnectedCompatibilityGroups } from './connected_groups.mjs';
export {
  decideRendererRoute,
  evaluateModuleGraphRoutes,
} from './route_decider.mjs';
export {
  extractGraphRoutingFacts,
  isInternalLibraryModule,
  evaluateGraphRoutes,
  prepareRouteInputs,
} from '../ingest/route_bridge.mjs';
export {
  RendererConstructionRouter,
  getRendererRoute,
  getRendererDecision,
} from './construction_adapter.mjs';
export { generateRouteReport, formatRouteReport } from './route_report.mjs';
