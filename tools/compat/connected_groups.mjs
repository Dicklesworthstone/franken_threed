/**
 * @file connected_groups.mjs
 * Connected compatibility groups tracking shared resources and renderers.
 * (Plan §3.3, §6.9, §8.5; Bead f3d-04.4)
 *
 * Renderers sharing mutable GPU resources (textures, render targets, geometries)
 * form connected groups. If any renderer in a group uses the exact WebGL backend,
 * all connected renderers must select a compatible route or explicitly declare
 * an isolated copy boundary.
 *
 * Read-only / immutable assets uploaded independently to distinct GPU devices
 * do NOT couple backend residency (Plan §3.3; Bead f3d-04.4 review).
 */

import { ExecutionRoute, EscapeReason } from './route_types.mjs';

export class ConnectedCompatibilityGroups {
  constructor() {
    /** @type {Map<string, string>} parent pointers for union-find */
    this._parent = new Map();
    /** @type {Map<string, number>} rank for union-find */
    this._rank = new Map();
    /** @type {Map<string, { type: 'renderer' | 'resource', preferredRoute?: string, forcedRoute?: string, isMutable?: boolean, reasons: string[] }>} */
    this._nodes = new Map();
    /** @type {Map<string, Set<string>>} group id -> set of node ids */
    this._groupMembers = new Map();
    /** @type {Map<string, string>} rendererId -> committed/locked route */
    this._committedRoutes = new Map();
    /** @type {Map<string, { route: string, reasons: string[], groupId?: string }>} */
    this._allDecisions = new Map();
  }

  /**
   * Register a renderer node.
   * @param {string} rendererId
   * @param {string} preferredRoute
   */
  registerRenderer(rendererId, preferredRoute = ExecutionRoute.RETAINED_UPSTREAM) {
    if (!this._nodes.has(rendererId)) {
      this._nodes.set(rendererId, {
        type: 'renderer',
        preferredRoute,
        forcedRoute: undefined,
        reasons: [],
      });
      this._parent.set(rendererId, rendererId);
      this._rank.set(rendererId, 0);
    }
  }

  /**
   * Record that a renderer has been committed/locked to a specific route.
   * @param {string} rendererId
   * @param {string} route
   */
  recordCommittedRoute(rendererId, route) {
    this._committedRoutes.set(rendererId, route);
  }

  /**
   * Register a resource node (e.g. RenderTarget, Texture, BufferGeometry).
   * @param {string} resourceId
   * @param {boolean} [isMutable=true] - Whether the GPU resource is mutable (e.g. render target)
   */
  registerResource(resourceId, isMutable = true) {
    if (!this._nodes.has(resourceId)) {
      this._nodes.set(resourceId, {
        type: 'resource',
        isMutable,
        reasons: [],
      });
      this._parent.set(resourceId, resourceId);
      this._rank.set(resourceId, 0);
    } else {
      const node = this._nodes.get(resourceId);
      if (isMutable && !node.isMutable) {
        node.isMutable = true;
      }
    }
  }

  /**
   * Check if a registered resource is mutable.
   * @param {string} resourceId
   * @returns {boolean}
   */
  isResourceMutable(resourceId) {
    const node = this._nodes.get(resourceId);
    return Boolean(node?.isMutable ?? true);
  }

  /**
   * Find root with path compression.
   * @param {string} id
   * @returns {string}
   */
  find(id) {
    if (!this._parent.has(id)) {
      this._parent.set(id, id);
      this._rank.set(id, 0);
      return id;
    }
    let root = id;
    while (this._parent.get(root) !== root) {
      root = this._parent.get(root);
    }
    let curr = id;
    while (curr !== root) {
      const next = this._parent.get(curr);
      this._parent.set(curr, root);
      curr = next;
    }
    return root;
  }

  /**
   * Union two elements (renderers or resources) into the same connected group.
   * @param {string} idA
   * @param {string} idB
   */
  union(idA, idB) {
    const rootA = this.find(idA);
    const rootB = this.find(idB);
    if (rootA === rootB) return;

    const rankA = this._rank.get(rootA) || 0;
    const rankB = this._rank.get(rootB) || 0;
    if (rankA < rankB) {
      this._parent.set(rootA, rootB);
    } else if (rankA > rankB) {
      this._parent.set(rootB, rootA);
    } else {
      this._parent.set(rootB, rootA);
      this._rank.set(rootA, rankA + 1);
    }
  }

  /**
   * Connect a resource to a renderer that reads or writes it.
   * Only MUTABLE GPU resources (render targets, written storage buffers, shared canvas)
   * couple backend residency; read-only assets uploaded independently do not.
   *
   * @param {string} rendererId
   * @param {string} resourceId
   * @param {boolean} [isMutable=true] - Defaults to true (conservative coupling)
   */
  recordResourceSharing(rendererId, resourceId, isMutable = true) {
    this.registerRenderer(rendererId);
    this.registerResource(resourceId, isMutable);
    if (isMutable) {
      this.union(rendererId, resourceId);
    }
  }

  /**
   * Resolve routes across connected groups.
   * If any member of a group requires EXACT_BACKEND (due to GL escapes or explicit WebGL),
   * all renderers in that group must resolve to EXACT_BACKEND to maintain valid context residency.
   * Checks prior exposed renderer locks to prevent illegal cross-backend resource sharing.
   *
   * @param {Map<string, { route: string, reasons: string[] }>} [individualDecisions]
   * @returns {Map<string, { route: string, reasons: string[], groupId: string }>}
   */
  resolveGroupRoutes(individualDecisions = new Map()) {
    // Merge new individual decisions into historical knowledge
    if (individualDecisions) {
      for (const [id, dec] of individualDecisions.entries()) {
        this._allDecisions.set(id, dec);
      }
    }

    // 1. Group members by representative root
    const groups = new Map();
    for (const id of this._nodes.keys()) {
      const root = this.find(id);
      if (!groups.has(root)) groups.set(root, []);
      groups.get(root).push(id);
    }

    /** @type {Map<string, { route: string, reasons: string[], groupId: string }>} */
    const results = new Map();

    for (const [groupId, members] of groups.entries()) {
      let requiresExact = false;
      const groupReasons = new Set();

      // Check all renderers in this group for exact requirements across committed routes,
      // recorded decisions, and preferred node routes
      for (const memberId of members) {
        const node = this._nodes.get(memberId);
        if (node && node.type === 'renderer') {
          const committed = this._committedRoutes.get(memberId);
          const decision = this._allDecisions.get(memberId);
          const effectiveRoute = committed || decision?.route || node.preferredRoute;

          if (effectiveRoute === ExecutionRoute.EXACT_BACKEND) {
            requiresExact = true;
            if (decision?.reasons) {
              for (const r of decision.reasons) groupReasons.add(r);
            }
          }
        }
      }

      // Check prior exposed renderer locks: if exact is required, no renderer already
      // committed to a non-exact route may be connected to this group
      if (requiresExact) {
        for (const memberId of members) {
          const committed = this._committedRoutes.get(memberId);
          if (committed && committed !== ExecutionRoute.EXACT_BACKEND) {
            throw new Error(
              `Connected group conflict: renderer '${memberId}' is already committed to route '${committed}', ` +
              `but shared resource connects it to a renderer requiring '${ExecutionRoute.EXACT_BACKEND}'. ` +
              `Sharing mutable GPU resources across distinct backend residencies without an isolated copy boundary is prohibited.`
            );
          }
        }
      }

      // Propagate exact requirement to all uncommitted/compatible renderers in the group
      for (const memberId of members) {
        const node = this._nodes.get(memberId);
        if (node && node.type === 'renderer') {
          const committed = this._committedRoutes.get(memberId);
          const decision = this._allDecisions.get(memberId);
          let finalRoute = committed || (decision ? decision.route : node.preferredRoute || ExecutionRoute.RETAINED_UPSTREAM);
          const reasons = decision ? [...decision.reasons] : (node.reasons ? [...node.reasons] : []);

          if (requiresExact && finalRoute !== ExecutionRoute.EXACT_BACKEND) {
            finalRoute = ExecutionRoute.EXACT_BACKEND;
            reasons.push(EscapeReason.CONNECTED_GROUP_CONSTRAINT);
          }

          const resolvedEntry = {
            route: finalRoute,
            reasons: Object.freeze(reasons),
            groupId,
          };
          this._allDecisions.set(memberId, resolvedEntry);
          results.set(memberId, resolvedEntry);
        }
      }
    }

    return results;
  }

  /**
   * Get all members in the group of an element.
   * @param {string} id
   * @returns {string[]}
   */
  getGroupMembers(id) {
    const root = this.find(id);
    const members = [];
    for (const candidate of this._nodes.keys()) {
      if (this.find(candidate) === root) {
        members.push(candidate);
      }
    }
    return members;
  }

  /**
   * Capture a snapshot of current union-find and routing state for transaction rollback.
   * Deep-copies node descriptors and decision records to isolate future mutations.
   * @returns {Object}
   */
  snapshot() {
    const nodesCopy = new Map();
    for (const [k, v] of this._nodes.entries()) {
      nodesCopy.set(k, {
        type: v.type,
        preferredRoute: v.preferredRoute,
        forcedRoute: v.forcedRoute,
        isMutable: v.isMutable,
        reasons: v.reasons ? [...v.reasons] : [],
      });
    }

    const groupMembersCopy = new Map();
    for (const [k, v] of this._groupMembers.entries()) {
      groupMembersCopy.set(k, new Set(v));
    }

    const allDecisionsCopy = new Map();
    for (const [k, v] of this._allDecisions.entries()) {
      allDecisionsCopy.set(k, {
        route: v.route,
        reasons: v.reasons ? [...v.reasons] : [],
        groupId: v.groupId,
      });
    }

    return {
      parent: new Map(this._parent),
      rank: new Map(this._rank),
      nodes: nodesCopy,
      groupMembers: groupMembersCopy,
      committedRoutes: new Map(this._committedRoutes),
      allDecisions: allDecisionsCopy,
    };
  }

  /**
   * Restore union-find and routing state from a previously captured snapshot.
   * @param {Object} snap
   */
  restore(snap) {
    if (!snap) return;
    this._parent = new Map(snap.parent);
    this._rank = new Map(snap.rank);
    this._nodes = new Map();
    for (const [k, v] of snap.nodes.entries()) {
      this._nodes.set(k, {
        type: v.type,
        preferredRoute: v.preferredRoute,
        forcedRoute: v.forcedRoute,
        isMutable: v.isMutable,
        reasons: v.reasons ? [...v.reasons] : [],
      });
    }
    this._groupMembers = new Map();
    for (const [k, v] of snap.groupMembers.entries()) {
      this._groupMembers.set(k, new Set(v));
    }
    this._committedRoutes = new Map(snap.committedRoutes);
    this._allDecisions = new Map();
    for (const [k, v] of snap.allDecisions.entries()) {
      this._allDecisions.set(k, {
        route: v.route,
        reasons: v.reasons ? [...v.reasons] : [],
        groupId: v.groupId,
      });
    }
  }
}

