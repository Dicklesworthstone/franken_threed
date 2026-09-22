/**
 * @file connected_groups.test.mjs
 * Focused unit test suite for ConnectedCompatibilityGroups (Bead f3d-04.4).
 *
 * Tests:
 * - Basic registration of renderers and resources (mutable and immutable).
 * - Path compression and union-find grouping.
 * - Route propagation across mutable shared resources (exact backend propagation).
 * - Conflict detection when committed non-exact renderer is unioned with exact requirement.
 * - previewRoute validation without mutating internal graph state:
 *   - Route propagation preview over mutable resources without mutating graph.
 *   - Conflict detection with committed non-exact renderer during preview.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { ConnectedCompatibilityGroups } from "./connected_groups.mjs";
import { EscapeReason, ExecutionRoute } from "./route_types.mjs";

test("ConnectedCompatibilityGroups: Basic registration and union-find", () => {
  const groups = new ConnectedCompatibilityGroups();

  groups.registerRenderer("renderer-1", ExecutionRoute.RETAINED_UPSTREAM);
  groups.registerResource("tex-1", false);
  groups.registerResource("rt-1", true);

  assert.equal(groups.isResourceMutable("tex-1"), false);
  assert.equal(groups.isResourceMutable("rt-1"), true);

  // tex-1 is immutable, so recordResourceSharing does not union
  groups.recordResourceSharing("renderer-1", "tex-1", false);
  assert.notEqual(groups.find("renderer-1"), groups.find("tex-1"));

  // rt-1 is mutable, so recordResourceSharing unions them
  groups.recordResourceSharing("renderer-1", "rt-1", true);
  assert.equal(groups.find("renderer-1"), groups.find("rt-1"));
});

test("ConnectedCompatibilityGroups: Route propagation over mutable resources", () => {
  const groups = new ConnectedCompatibilityGroups();

  groups.registerRenderer("renderer-exact", ExecutionRoute.EXACT_BACKEND);
  groups.registerRenderer("renderer-gpu", ExecutionRoute.RETAINED_UPSTREAM);

  groups.recordResourceSharing("renderer-exact", "shared-target", true);
  groups.recordResourceSharing("renderer-gpu", "shared-target", true);

  const decisions = groups.resolveGroupRoutes();
  const exactDec = decisions.get("renderer-exact");
  const gpuDec = decisions.get("renderer-gpu");

  assert.equal(exactDec.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(gpuDec.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(gpuDec.reasons.includes(EscapeReason.CONNECTED_GROUP_CONSTRAINT));
});

test("ConnectedCompatibilityGroups: Rejects conflict if non-exact renderer is already committed", () => {
  const groups = new ConnectedCompatibilityGroups();

  groups.registerRenderer("renderer-gpu", ExecutionRoute.RETAINED_UPSTREAM);
  groups.recordCommittedRoute("renderer-gpu", ExecutionRoute.RETAINED_UPSTREAM);

  groups.registerRenderer("renderer-exact", ExecutionRoute.EXACT_BACKEND);

  groups.recordResourceSharing("renderer-gpu", "shared-target", true);
  groups.recordResourceSharing("renderer-exact", "shared-target", true);

  assert.throws(
    () => groups.resolveGroupRoutes(),
    /Connected group conflict: renderer 'renderer-gpu' is already committed to route 'retained-upstream'/,
  );
});

test("ConnectedCompatibilityGroups: previewRoute evaluates exact constraint without mutating graph", () => {
  const groups = new ConnectedCompatibilityGroups();

  // Exact renderer committed to rt-exact
  groups.registerRenderer("renderer-exact", ExecutionRoute.EXACT_BACKEND);
  groups.recordCommittedRoute("renderer-exact", ExecutionRoute.EXACT_BACKEND);
  groups.recordResourceSharing("renderer-exact", "rt-exact", true);

  // Candidate renderer preferring RETAINED_UPSTREAM shares rt-exact
  const preview = groups.previewRoute({ route: ExecutionRoute.RETAINED_UPSTREAM, reasons: [] }, [
    { id: "rt-exact", isMutable: true },
  ]);

  assert.equal(preview.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(preview.reasons.includes(EscapeReason.CONNECTED_GROUP_CONSTRAINT));

  // Graph state must remain unmutated by previewRoute
  assert.equal(groups._nodes.has("candidate"), false);
  assert.equal(groups.getGroupMembers("rt-exact").length, 2); // only renderer-exact and rt-exact
});

test("ConnectedCompatibilityGroups: previewRoute detects conflict with committed non-exact renderer", () => {
  const groups = new ConnectedCompatibilityGroups();

  // GPU renderer committed to rt-shared
  groups.registerRenderer("renderer-gpu", ExecutionRoute.RETAINED_UPSTREAM);
  groups.recordCommittedRoute("renderer-gpu", ExecutionRoute.RETAINED_UPSTREAM);
  groups.recordResourceSharing("renderer-gpu", "rt-shared", true);

  // Candidate renderer requiring EXACT_BACKEND tries to share rt-shared
  assert.throws(
    () =>
      groups.previewRoute({ route: ExecutionRoute.EXACT_BACKEND, reasons: [] }, [
        { id: "rt-shared", isMutable: true },
      ]),
    /Connected group conflict/,
  );

  // Graph state must remain unmutated
  assert.equal(groups.getGroupMembers("rt-shared").length, 2);
});
