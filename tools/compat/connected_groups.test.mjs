/**
 * @file connected_groups.test.mjs
 * Focused unit test suite for ConnectedCompatibilityGroups (Bead f3d-04.4).
 *
 * Tests:
 * - Basic registration of renderers and resources (mutable and immutable).
 * - Path compression and union-find grouping.
 * - Route propagation across mutable shared resources (exact backend propagation).
 * - Conflict detection when committed non-exact renderer is unioned with exact requirement.
 * - Transactional snapshot and restore:
 *   - Deep copy isolation of nodes, group members, committed routes, and decisions.
 *   - Rollback of intermediate mutations, unions, registrations, and route resolutions.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { ConnectedCompatibilityGroups } from './connected_groups.mjs';
import { ExecutionRoute, EscapeReason } from './route_types.mjs';

test('ConnectedCompatibilityGroups: Basic registration and union-find', () => {
  const groups = new ConnectedCompatibilityGroups();

  groups.registerRenderer('renderer-1', ExecutionRoute.RETAINED_UPSTREAM);
  groups.registerResource('tex-1', false);
  groups.registerResource('rt-1', true);

  assert.equal(groups.isResourceMutable('tex-1'), false);
  assert.equal(groups.isResourceMutable('rt-1'), true);

  // tex-1 is immutable, so recordResourceSharing does not union
  groups.recordResourceSharing('renderer-1', 'tex-1', false);
  assert.notEqual(groups.find('renderer-1'), groups.find('tex-1'));

  // rt-1 is mutable, so recordResourceSharing unions them
  groups.recordResourceSharing('renderer-1', 'rt-1', true);
  assert.equal(groups.find('renderer-1'), groups.find('rt-1'));
});

test('ConnectedCompatibilityGroups: Route propagation over mutable resources', () => {
  const groups = new ConnectedCompatibilityGroups();

  groups.registerRenderer('renderer-exact', ExecutionRoute.EXACT_BACKEND);
  groups.registerRenderer('renderer-gpu', ExecutionRoute.RETAINED_UPSTREAM);

  groups.recordResourceSharing('renderer-exact', 'shared-target', true);
  groups.recordResourceSharing('renderer-gpu', 'shared-target', true);

  const decisions = groups.resolveGroupRoutes();
  const exactDec = decisions.get('renderer-exact');
  const gpuDec = decisions.get('renderer-gpu');

  assert.equal(exactDec.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(gpuDec.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(gpuDec.reasons.includes(EscapeReason.CONNECTED_GROUP_CONSTRAINT));
});

test('ConnectedCompatibilityGroups: Rejects conflict if non-exact renderer is already committed', () => {
  const groups = new ConnectedCompatibilityGroups();

  groups.registerRenderer('renderer-gpu', ExecutionRoute.RETAINED_UPSTREAM);
  groups.recordCommittedRoute('renderer-gpu', ExecutionRoute.RETAINED_UPSTREAM);

  groups.registerRenderer('renderer-exact', ExecutionRoute.EXACT_BACKEND);

  groups.recordResourceSharing('renderer-gpu', 'shared-target', true);
  groups.recordResourceSharing('renderer-exact', 'shared-target', true);

  assert.throws(
    () => groups.resolveGroupRoutes(),
    /Connected group conflict: renderer 'renderer-gpu' is already committed to route 'retained-upstream'/
  );
});

test('ConnectedCompatibilityGroups: Transactional snapshot and restore', () => {
  const groups = new ConnectedCompatibilityGroups();

  // Baseline state: one committed renderer and one resource
  groups.registerRenderer('renderer-base', ExecutionRoute.RETAINED_UPSTREAM);
  groups.recordCommittedRoute('renderer-base', ExecutionRoute.RETAINED_UPSTREAM);
  groups.registerResource('rt-base', true);
  groups.recordResourceSharing('renderer-base', 'rt-base', true);
  groups.resolveGroupRoutes();

  // Capture snapshot
  const snap = groups.snapshot();
  assert.ok(snap);

  // Perform speculative mutations that simulate a failing renderer construction
  groups.registerRenderer('renderer-failing', ExecutionRoute.EXACT_BACKEND);
  groups.registerResource('rt-new', true);
  groups.recordResourceSharing('renderer-failing', 'rt-base', true);
  groups.recordResourceSharing('renderer-failing', 'rt-new', true);

  // At this point, rt-base is unioned with renderer-failing
  assert.equal(groups.find('renderer-failing'), groups.find('rt-base'));
  assert.equal(groups.getGroupMembers('rt-base').includes('renderer-failing'), true);

  // Restore snapshot (simulating catch block rollback)
  groups.restore(snap);

  // Verify full rollback: renderer-failing and rt-new do not exist
  assert.equal(groups._nodes.has('renderer-failing'), false);
  assert.equal(groups._nodes.has('rt-new'), false);
  assert.equal(groups.getGroupMembers('rt-base').includes('renderer-failing'), false);
  assert.deepEqual(groups.getGroupMembers('rt-base').sort(), ['renderer-base', 'rt-base'].sort());

  // Verify baseline route resolution remains valid and uncorrupted
  const restoredDecisions = groups.resolveGroupRoutes();
  assert.equal(restoredDecisions.get('renderer-base').route, ExecutionRoute.RETAINED_UPSTREAM);
});

test('ConnectedCompatibilityGroups: Deep copy isolation in snapshot', () => {
  const groups = new ConnectedCompatibilityGroups();
  groups.registerRenderer('renderer-1', ExecutionRoute.RETAINED_UPSTREAM);
  groups.registerResource('res-1', false);

  const snap = groups.snapshot();

  // Mutating original after snapshot should not affect snapshot
  groups.registerResource('res-1', true); // Promotes isMutable to true
  const nodeOriginal = groups._nodes.get('res-1');
  assert.equal(nodeOriginal.isMutable, true);

  const snapNode = snap.nodes.get('res-1');
  assert.equal(snapNode.isMutable, false, 'Snapshot must keep immutable deep copy of node');

  // Restoring should bring back isMutable: false
  groups.restore(snap);
  assert.equal(groups.isResourceMutable('res-1'), false);
});
