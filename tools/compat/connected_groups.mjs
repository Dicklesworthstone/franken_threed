/**
 * @file connected_groups.mjs
 * Connected compatibility groups tracking shared resources and renderers.
 * (Plan §3.3, §6.9, §8.5; Bead f3d-04.4)
 *
 * Renderers sharing mutable GPU resources (textures, render targets, geometries)
 * form connected groups. If any renderer in a group uses the exact WebGL backend,
 * all connected renderers must select a compatible route or explicitly declare
 * an isolated copy boundary.
 */

import { ExecutionRoute, EscapeReason } from './route_types.mjs';

export class ConnectedCompatibilityGroups {
  constructor() {
    /** @type {Map<string, string>} parent pointers for union-find */
    this._parent = new Map();
    /** @type {Map<string, number>} rank for union-find */
    this._rank = new Map();
    /** @type {Map<string, { type: 'renderer' | 'resource', preferredRoute?: string, forcedRoute?: string, reasons: string[] }>} */
    this._nodes = new Map();
    /** @type {Map<string, Set<string>>} group id -> set of node ids */
    this._groupMembers = new Map();
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
   * Register a resource node (e.g. RenderTarget, Texture, BufferGeometry).
   * @param {string} resourceId
   */
  registerResource(resourceId) {
    if (!this._nodes.has(resourceId)) {
      this._nodes.set(resourceId, {
        type: 'resource',
        reasons: [],
      });
      this._parent.set(resourceId, resourceId);
      this._rank.set(resourceId, 0);
    }
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
   * @param {string} rendererId
   * @param {string} resourceId
   */
  recordResourceSharing(rendererId, resourceId) {
    this.registerRenderer(rendererId);
    this.registerResource(resourceId);
    this.union(rendererId, resourceId);
  }

  /**
   * Resolve routes across connected groups.
   * If any member of a group requires EXACT_BACKEND (due to GL escapes or explicit WebGL),
   * all renderers in that group must resolve to EXACT_BACKEND to maintain valid context residency.
   * @param {Map<string, { route: string, reasons: string[] }>} [individualDecisions]
   * @returns {Map<string, { route: string, reasons: string[], groupId: string }>}
   */
  resolveGroupRoutes(individualDecisions = new Map()) {
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

      // Check all renderers in this group for exact requirements
      for (const memberId of members) {
        const decision = individualDecisions.get(memberId);
        if (decision) {
          if (decision.route === ExecutionRoute.EXACT_BACKEND) {
            requiresExact = true;
            for (const r of decision.reasons) groupReasons.add(r);
          }
        }
      }

      // If any member requires exact, propagate to all renderers in group
      for (const memberId of members) {
        const node = this._nodes.get(memberId);
        if (node && node.type === 'renderer') {
          const decision = individualDecisions.get(memberId);
          let finalRoute = decision ? decision.route : node.preferredRoute || ExecutionRoute.RETAINED_UPSTREAM;
          const reasons = decision ? [...decision.reasons] : [];

          if (requiresExact && finalRoute !== ExecutionRoute.EXACT_BACKEND) {
            finalRoute = ExecutionRoute.EXACT_BACKEND;
            reasons.push(EscapeReason.CONNECTED_GROUP_CONSTRAINT);
          }

          results.set(memberId, {
            route: finalRoute,
            reasons,
            groupId,
          });
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
}
