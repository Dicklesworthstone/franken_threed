/**
 * @file route_decider.mjs
 * Route decision evaluator for renderer construction (Plan §3.3, §5.1, §5.6).
 *
 * Determines the execution route before any canvas context acquisition
 * or native handle exposure occurs. Consumes static AST facts or full
 * module graph JSON bundles (schema 1.0.0 from RubyCrane).
 */

import { ExecutionRoute, EscapeReason } from './route_types.mjs';
import {
  extractGraphRoutingFacts,
  isInternalLibraryModule,
  evaluateGraphRoutes,
  prepareRouteInputs,
} from '../ingest/route_bridge.mjs';

export {
  extractGraphRoutingFacts,
  isInternalLibraryModule,
  evaluateGraphRoutes,
  prepareRouteInputs,
};

/**
 * Statically evaluate route decisions for all construction sites in an ingested module graph bundle.
 *
 * @param {Object} bundle - Schema 1.0.0 module graph bundle from tools/ingest/cli.mjs
 * @param {Object} [environment]
 * @param {Object} [environment.hostCapabilities]
 * @param {boolean} [environment.specializationAvailable]
 * @returns {Array<{
 *   moduleId: string,
 *   constructorName: string,
 *   decision: { route: string, reasons: string[], sourceSpan: string }
 * }>}
 */
export function evaluateModuleGraphRoutes(bundle, environment = {}) {
  return evaluateGraphRoutes(bundle, decideRendererRoute, environment);
}

/**
 * Decide the execution route for a renderer construction site.
 * Supports direct parameters, static AST facts, and full module graph JSON bundles (schema 1.0.0).
 *
 * @param {Object} input
 * @param {string} [input.constructorName] - e.g. 'WebGLRenderer', 'WebGPURenderer', 'CSS2DRenderer'
 * @param {Object} [input.options] - constructor parameters (e.g. forceWebGL, canvas)
 * @param {Object} [input.analysis] - static AST facts or schema 1.0.0 module graph bundle
 * @param {Object} [input.bundle] - alternative field for module graph bundle
 * @param {Object} [input.moduleGraph] - alternative field for module graph bundle
 * @param {Object} [input.hostCapabilities]
 * @param {boolean} [input.specializationAvailable]
 * @param {string} [input.sourceSpan]
 * @returns {{ route: string, reasons: string[], sourceSpan: string, constructorName: string }}
 */
export function decideRendererRoute(input = {}) {
  let constructorName = input.constructorName;
  let options = { ...input.options };
  let analysis = input.analysis || {};
  let sourceSpan = input.sourceSpan;

  // Consume RubyCrane's module graph JSON (schema 1.0.0) if passed as analysis or bundle input
  const bundle = input.moduleGraph || input.bundle || (
    analysis && (analysis.schema_version === '1.0.0' || analysis.modules) ? analysis : null
  );

  if (bundle && typeof bundle === 'object' && bundle.modules) {
    const graphFacts = extractGraphRoutingFacts(bundle);
    analysis = {
      hasOpaqueGLEscapes: graphFacts.hasOpaqueGLEscapes,
      hasNativeContextAccess: graphFacts.hasNativeContextAccess,
      hasUnresolvedContextAccess: graphFacts.hasUnresolvedContextAccess,
      ...analysis,
    };

    if (!constructorName && graphFacts.constructionSites.length > 0) {
      const site = graphFacts.constructionSites[0];
      constructorName = site.constructorName;
      if (options.forceWebGL === undefined) {
        options.forceWebGL = site.options.forceWebGL;
      }
      if (options.forceWebGLUnresolved === undefined && site.options.forceWebGLUnresolved !== undefined) {
        options.forceWebGLUnresolved = site.options.forceWebGLUnresolved;
      }
      if (options.canvas === undefined) {
        options.canvas = site.options.canvas;
      }
      sourceSpan = sourceSpan || site.sourceSpan;
    }
  }

  constructorName = constructorName || 'WebGLRenderer';
  sourceSpan = sourceSpan || 'unknown:0:0';

  const isForceWebGLUnresolved = options.forceWebGL === 'unresolved' ||
    options.force_webgl === 'unresolved' ||
    Boolean(options.forceWebGLUnresolved || options.force_webgl_unresolved);

  // Explicit WebGL selection only when options.forceWebGL === true (never coerce 'unresolved' to true or false)
  const isExplicitForceWebGL = !isForceWebGLUnresolved && Boolean(
    options.forceWebGL === true ||
    options.force_webgl === true ||
    options.hasForceWebGL === true ||
    options.has_force_webgl === true
  );

  const hasOpaqueGLEscapes = Boolean(analysis.hasOpaqueGLEscapes || analysis.has_opaque_gl_escapes);
  const hasNativeContextAccess = Boolean(analysis.hasNativeContextAccess || analysis.has_native_context_access);
  const hasUnresolvedContextAccess = Boolean(analysis.hasUnresolvedContextAccess || analysis.has_unresolved_context_access);

  const hostCapabilities = {
    hasWebGPU: typeof navigator !== 'undefined' && 'gpu' in navigator ? true : (input.hostCapabilities?.hasWebGPU ?? false),
    hasWebGL: typeof window !== 'undefined' ? true : (input.hostCapabilities?.hasWebGL ?? true),
    ...input.hostCapabilities,
  };
  const specializationAvailable = input.specializationAvailable ?? false;

  const reasons = [];

  // When forceWebGL is unresolved, record reason UNRESOLVED_FORCE_WEBGL and defer to runtime
  if (isForceWebGLUnresolved) {
    reasons.push(EscapeReason.UNRESOLVED_FORCE_WEBGL);
  }

  // Non-GPU renderers (CSS2D, CSS3D, SVG) are always retained upstream (Plan §5.1 / Bead AC)
  if (['CSS2DRenderer', 'CSS3DRenderer', 'SVGRenderer'].includes(constructorName)) {
    reasons.push(EscapeReason.EXPLICIT_SOURCE_SELECTION);
    return {
      route: ExecutionRoute.RETAINED_UPSTREAM,
      reasons,
      sourceSpan,
      constructorName,
    };
  }

  // Explicit WebGL constructor always maps to exact backend
  if (constructorName === 'WebGLRenderer') {
    reasons.push(EscapeReason.EXPLICIT_SOURCE_SELECTION);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
      constructorName,
    };
  }

  // Explicit forceWebGL in parameters (only when options.forceWebGL === true)
  if (isExplicitForceWebGL === true) {
    reasons.push(EscapeReason.EXPLICIT_SOURCE_SELECTION);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
      constructorName,
    };
  }

  // Opaque GL state / extension / method escapes detected by analysis
  if (hasOpaqueGLEscapes === true) {
    reasons.push(EscapeReason.OPAQUE_GL_ESCAPE);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
      constructorName,
    };
  }

  // Synchronous getContext() / native handle exposure
  if (hasNativeContextAccess === true) {
    const contextReason = hasUnresolvedContextAccess === true
      ? EscapeReason.UNRESOLVED_NATIVE_CONTEXT_ACCESS
      : EscapeReason.NATIVE_CONTEXT_ACCESS;
    reasons.push(contextReason);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
      constructorName,
    };
  }

  // Host lacks WebGPU capability -> fallback to exact GL if supported
  if (!hostCapabilities.hasWebGPU) {
    if (hostCapabilities.hasWebGL) {
      reasons.push(EscapeReason.HOST_LIMITATION_FALLBACK);
      return {
        route: ExecutionRoute.EXACT_BACKEND,
        reasons,
        sourceSpan,
        constructorName,
      };
    }
    throw new Error(`Host lacks both WebGPU and WebGL capability at ${sourceSpan}.`);
  }

  // WebGPURenderer on capable WebGPU host
  if (constructorName === 'WebGPURenderer') {
    if (specializationAvailable) {
      return {
        route: ExecutionRoute.SPECIALIZED_WEBGPU,
        reasons: reasons.length > 0 ? [...reasons, 'specialized-island-admitted'] : ['specialized-island-admitted'],
        sourceSpan,
        constructorName,
      };
    }
    // Interim Phase 0/1 or when specialization is not ready: retained upstream WebGPU
    reasons.push(EscapeReason.SPECIALIZATION_UNAVAILABLE);
    return {
      route: ExecutionRoute.RETAINED_UPSTREAM,
      reasons,
      sourceSpan,
      constructorName,
    };
  }

  // Default fallback to retained component
  reasons.push(EscapeReason.SPECIALIZATION_UNAVAILABLE);
  return {
    route: ExecutionRoute.RETAINED_UPSTREAM,
    reasons,
    sourceSpan,
    constructorName,
  };
}
