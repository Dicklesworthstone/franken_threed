/**
 * @file index.mjs
 * Main entry point for tools/compat renderer construction routing.
 */

export { ExecutionRoute, EscapeReason, RouteLockError } from './route_types.mjs';
export { ConnectedCompatibilityGroups } from './connected_groups.mjs';
export { decideRendererRoute } from './route_decider.mjs';
export { RendererConstructionRouter } from './construction_adapter.mjs';
export { generateRouteReport, formatRouteReport } from './route_report.mjs';
