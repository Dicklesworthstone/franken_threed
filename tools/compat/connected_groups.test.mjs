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

test("Read-only users stay independent until a resource becomes writable", () => {
  const groups = new ConnectedCompatibilityGroups();
  groups.registerRenderer("source", ExecutionRoute.EXACT_BACKEND);
  groups.registerRenderer("native", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordResourceSharing("source", "image", false);
  groups.recordResourceSharing("native", "image", false);

  assert.equal(groups.resolveGroupRoutes().get("native").route, ExecutionRoute.GENERAL_WEBGPU);
  assert.deepEqual(groups.getGroupMembers("source"), ["source"]);
  assert.deepEqual(groups.getGroupMembers("native"), ["native"]);

  // A newly discovered writer must join both earlier readers, not just itself.
  groups.registerRenderer("writer", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordResourceSharing("writer", "image", true);
  const decisions = groups.resolveGroupRoutes();
  for (const id of ["source", "native", "writer"]) {
    assert.equal(decisions.get(id).route, ExecutionRoute.EXACT_BACKEND);
    assert.equal(decisions.get(id).groupId, decisions.get("source").groupId);
  }
  assert.ok(decisions.get("native").reasons.includes(EscapeReason.CONNECTED_GROUP_CONSTRAINT));
});

test("Resource promotion and later read-only uses preserve mutable GPU history", () => {
  const groups = new ConnectedCompatibilityGroups();
  groups.registerRenderer("source", ExecutionRoute.EXACT_BACKEND);
  groups.recordResourceSharing("source", "image", false);
  groups.registerResource("image", true);
  groups.registerRenderer("reader", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordResourceSharing("reader", "image", false);

  assert.equal(groups.isResourceMutable("image"), true);
  assert.equal(groups.resolveGroupRoutes().get("reader").route, ExecutionRoute.EXACT_BACKEND);
  assert.deepEqual(new Set(groups.getGroupMembers("reader")), new Set(["source", "image", "reader"]));
});

test("Previewing a write includes earlier read-only users without promoting the resource", () => {
  const groups = new ConnectedCompatibilityGroups();
  groups.registerRenderer("source", ExecutionRoute.EXACT_BACKEND);
  groups.recordCommittedRoute("source", ExecutionRoute.EXACT_BACKEND);
  groups.recordResourceSharing("source", "image", false);
  const initial = {route: ExecutionRoute.GENERAL_WEBGPU, reasons: []};

  assert.equal(groups.previewRoute(initial, [{id: "image", isMutable: false}]).route,
    ExecutionRoute.GENERAL_WEBGPU);
  const preview = groups.previewRoute(initial, [{id: "image", isMutable: true}]);
  assert.equal(preview.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(groups.previewRoute(initial, [{id: "image"}]).route, preview.route,
    "omitted mutability is conservatively writable, as in recordResourceSharing");
  assert.ok(preview.reasons.includes(EscapeReason.CONNECTED_GROUP_CONSTRAINT));
  assert.equal(groups.isResourceMutable("image"), false);
  assert.deepEqual(groups.getGroupMembers("source"), ["source"]);
  assert.deepEqual(groups.getGroupMembers("image"), ["image"]);

  groups.registerRenderer("writer", preview.route);
  groups.recordResourceSharing("writer", "image", true);
  assert.equal(groups.resolveGroupRoutes().get("writer").route, preview.route);
});

test("A later read-only declaration cannot hide an existing mutable group from preview", () => {
  for (const priorRoute of [ExecutionRoute.EXACT_BACKEND, ExecutionRoute.GENERAL_WEBGPU]) {
    const groups = new ConnectedCompatibilityGroups();
    groups.registerRenderer("owner", priorRoute);
    groups.recordCommittedRoute("owner", priorRoute);
    groups.recordResourceSharing("owner", "history", true);
    const resources = [{id: "history", isMutable: false}];
    if (priorRoute === ExecutionRoute.EXACT_BACKEND) {
      assert.equal(groups.previewRoute({route: ExecutionRoute.GENERAL_WEBGPU, reasons: []}, resources).route,
        ExecutionRoute.EXACT_BACKEND);
    } else {
      assert.throws(() => groups.previewRoute({route: ExecutionRoute.EXACT_BACKEND, reasons: []}, resources),
        /Connected group conflict/);
    }
  }
});

test("Promotion previews reject committed conflicts across all prospective resource groups", () => {
  const groups = new ConnectedCompatibilityGroups();
  groups.registerRenderer("source", ExecutionRoute.EXACT_BACKEND);
  groups.recordCommittedRoute("source", ExecutionRoute.EXACT_BACKEND);
  groups.recordResourceSharing("source", "source-image", false);
  groups.registerRenderer("native", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordCommittedRoute("native", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordResourceSharing("native", "native-image", false);
  const resources = [{id: "source-image", isMutable: true}, {id: "native-image", isMutable: true}];

  for (const pending of [resources, [...resources].reverse()]) {
    assert.throws(() => groups.previewRoute({route: ExecutionRoute.GENERAL_WEBGPU, reasons: []}, pending),
      /Connected group conflict: renderer 'native'/);
    assert.equal(groups.isResourceMutable("source-image"), false);
    assert.equal(groups.isResourceMutable("native-image"), false);
    assert.deepEqual(groups.getGroupMembers("source"), ["source"]);
    assert.deepEqual(groups.getGroupMembers("native"), ["native"]);
  }
});

test("Direct promotion detects an already committed incompatible residency", () => {
  const groups = new ConnectedCompatibilityGroups();
  groups.registerRenderer("source", ExecutionRoute.EXACT_BACKEND);
  groups.registerRenderer("native", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordCommittedRoute("native", ExecutionRoute.GENERAL_WEBGPU);
  groups.recordResourceSharing("source", "image", false);
  groups.recordResourceSharing("native", "image", false);
  groups.registerResource("image", true);
  assert.throws(() => groups.resolveGroupRoutes(), /Connected group conflict/);
});
