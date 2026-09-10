/**
 * @file exact_backend.mjs
 * Exact-backend component: pinned upstream WebGLRenderer and WebGLBackend isolated
 * with the genuine GL contract (Plan §3.3, §5.6, §5.12, §6.9; Bead f3d-04-module-routing-exact-boundaries-6mv.3).
 *
 * This module preserves what cannot be faked:
 * - Direct import of the pinned Three.js r186 build/three.module.js WebGLRenderer (never a copy).
 * - Verbatim preservation of constructor context/canvas options, getContext(), getContextAttributes(),
 *   extensions, and native program handles without wrapper proxies.
 * - Integration with RendererConstructionRouter as the registered EXACT_BACKEND implementation.
 * - Irreversible route locking on real DOM canvases before context acquisition.
 * - No-claim boundary: exact-backend execution is honest compatibility, never claimed as WebGPU acceleration.
 */

import { WebGLRenderer as PinnedWebGLRenderer } from '../../upstream/three.js/build/three.module.js';
import { ExecutionRoute } from './route_types.mjs';
import { RendererConstructionRouter } from './construction_adapter.mjs';

export { PinnedWebGLRenderer };

/**
 * Semantic alias for the pinned WebGLRenderer implementation.
 * @type {typeof PinnedWebGLRenderer}
 */
export const ExactWebGLRenderer = PinnedWebGLRenderer;

/**
 * Register the pinned WebGLRenderer as the admitted EXACT_BACKEND implementation
 * on a RendererConstructionRouter instance.
 *
 * @param {RendererConstructionRouter} router
 * @param {Object} [options]
 * @param {string} [options.constructorName] - optional constructor qualifier (defaults to route-level registration)
 * @returns {RendererConstructionRouter} The router instance with exact backend registered
 */
export function registerExactBackend(router, { constructorName } = {}) {
  if (!router || typeof router.registerImplementation !== 'function') {
    throw new TypeError('router must be an instance of RendererConstructionRouter');
  }
  router.registerImplementation(ExecutionRoute.EXACT_BACKEND, PinnedWebGLRenderer, { constructorName });
  return router;
}

/**
 * Create a RendererConstructionRouter with the pinned WebGLRenderer pre-registered
 * as the admitted EXACT_BACKEND implementation.
 *
 * @param {Object} [config]
 * @param {import('./connected_groups.mjs').ConnectedCompatibilityGroups} [config.connectedGroups]
 * @param {boolean} [config.specializationAvailable]
 * @param {Record<string, Function | Record<string, Function>>} [config.implementations]
 * @returns {RendererConstructionRouter}
 */
export function createExactBackendRouter(config = {}) {
  const implementations = {
    [ExecutionRoute.EXACT_BACKEND]: PinnedWebGLRenderer,
    ...(config.implementations || {}),
  };
  return new RendererConstructionRouter({
    ...config,
    implementations,
  });
}

/**
 * Construct a pinned WebGLRenderer instance through the construction router,
 * ensuring irreversible canvas route locking, connected compatibility group tracking,
 * and verbatim preservation of native GL context and object shape.
 *
 * @param {Object} [options] - WebGLRenderer constructor options (e.g. { canvas, context, antialias })
 * @param {RendererConstructionRouter | Object} [routerOrConfig] - Router instance or router configuration object
 * @returns {PinnedWebGLRenderer} Genuine pinned WebGLRenderer instance (never a wrapper proxy)
 */
export function createExactWebGLRenderer(options = {}, routerOrConfig) {
  const router = routerOrConfig instanceof RendererConstructionRouter
    ? routerOrConfig
    : createExactBackendRouter(routerOrConfig);

  return router.routeAndConstruct({
    constructorFn: PinnedWebGLRenderer,
    constructorName: 'WebGLRenderer',
    options,
    sourceSpan: 'tools/compat/exact_backend.mjs:createExactWebGLRenderer',
  });
}

/**
 * Helper to instantiate a WebGLBackend for WebGPURenderer compatibility when requested.
 * Dynamically imports the pinned WebGLBackend from three.webgpu.js to avoid eager evaluation.
 *
 * @param {Object} [backendOptions]
 * @returns {Promise<Object>} Pinned WebGLBackend instance
 */
export async function createExactBackendForRenderer(backendOptions = {}) {
  const { WebGLBackend } = await import('../../upstream/three.js/build/three.webgpu.js');
  return new WebGLBackend(backendOptions);
}

/**
 * Lazy accessor for the pinned WebGLBackend class from three.webgpu.js.
 * @returns {Promise<Function>} The pinned WebGLBackend constructor
 */
export async function getPinnedWebGLBackend() {
  const { WebGLBackend } = await import('../../upstream/three.js/build/three.webgpu.js');
  return WebGLBackend;
}
