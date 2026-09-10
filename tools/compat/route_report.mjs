/**
 * @file route_report.mjs
 * Generates static and runtime route decision reports (Plan §6.9, §14.3).
 *
 * Truthfully accounts for execution routes without conflating exact component
 * retention with acceleration.
 */

import { ExecutionRoute } from './route_types.mjs';

/**
 * Generate a structured route decision report.
 *
 * @param {import('./construction_adapter.mjs').RendererConstructionRouter} router
 * @returns {Object} Structured JSON report
 */
export function generateRouteReport(router) {
  const decisions = router.getDecisions();

  const routeCounts = {
    [ExecutionRoute.SPECIALIZED_WEBGPU]: 0,
    [ExecutionRoute.GENERAL_WEBGPU]: 0,
    [ExecutionRoute.RETAINED_UPSTREAM]: 0,
    [ExecutionRoute.EXACT_BACKEND]: 0,
  };

  let unresolvedDecisions = 0;

  for (const d of decisions) {
    if (d.route in routeCounts) {
      routeCounts[d.route]++;
    }
    const hasUnresolved = Array.isArray(d.reasons) && d.reasons.some(r =>
      typeof r === 'string' && (r.startsWith('unresolved-') || r.includes('unresolved'))
    );
    if (hasUnresolved) {
      unresolvedDecisions++;
    }
  }

  return {
    schema: 'f3d.route_report.v1',
    generated_at: new Date().toISOString(),
    total_renderers: decisions.length,
    unresolved_decisions: unresolvedDecisions,
    route_counts: routeCounts,
    decisions,
    no_claim_attestation: Object.freeze({
      exact_backend: 'Retained exact compatibility is compositional equivalence; never credited as acceleration or a new renderer (Plan §5.1).',
      retained_upstream: 'Retained upstream JS execution preserves full component functionality; not a Rust rewrite.',
      unresolved_facts: 'Decisions with unresolved facts defer execution boundaries to runtime validation; static optimization cannot be claimed until resolved (Plan §3.3).',
    }),
  };
}

/**
 * Format report into human-readable text for CLI or terminal inspection.
 * @param {Object} report
 * @returns {string}
 */
export function formatRouteReport(report) {
  const lines = [
    '=== FrankenThreeD Construction Route Report ===',
    `Total Renderers: ${report.total_renderers}`,
    `Unresolved Decisions: ${report.unresolved_decisions || 0}`,
    'Route Breakdown:',
    `  Specialized WebGPU: ${report.route_counts[ExecutionRoute.SPECIALIZED_WEBGPU]}`,
    `  General WebGPU:     ${report.route_counts[ExecutionRoute.GENERAL_WEBGPU]}`,
    `  Retained Upstream:  ${report.route_counts[ExecutionRoute.RETAINED_UPSTREAM]}`,
    `  Exact Backend (GL): ${report.route_counts[ExecutionRoute.EXACT_BACKEND]}`,
    '',
    'Decisions Log:',
  ];

  for (const d of report.decisions) {
    lines.push(
      `  [${d.rendererId}] Route: ${d.route} | Group: ${d.groupId} | Canvas: ${d.canvas} | Span: ${d.sourceSpan}`
    );
    lines.push(`    Reasons: ${d.reasons.join(', ')}`);
  }

  const attestations = [];
  const routesPresent = new Set((report.decisions || []).map(d => d.route));
  const hasUnresolved = (report.unresolved_decisions || 0) > 0 ||
    (report.decisions || []).some(d => Array.isArray(d.reasons) && d.reasons.some(r => typeof r === 'string' && (r.startsWith('unresolved-') || r.includes('unresolved'))));

  if (routesPresent.has(ExecutionRoute.EXACT_BACKEND) && report.no_claim_attestation?.exact_backend) {
    attestations.push(report.no_claim_attestation.exact_backend);
  }
  if (routesPresent.has(ExecutionRoute.RETAINED_UPSTREAM) && report.no_claim_attestation?.retained_upstream) {
    attestations.push(report.no_claim_attestation.retained_upstream);
  }
  if (hasUnresolved && report.no_claim_attestation?.unresolved_facts) {
    attestations.push(report.no_claim_attestation.unresolved_facts);
  }

  if (attestations.length > 0) {
    lines.push('');
    for (const att of attestations) {
      lines.push(`Attestation: ${att}`);
    }
  }
  return lines.join('\n');
}
