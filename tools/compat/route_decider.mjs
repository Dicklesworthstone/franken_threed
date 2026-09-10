/**
 * @file route_decider.mjs
 * Route decision evaluator for renderer construction (Plan §3.3, §5.1, §5.6).
 *
 * Determines the execution route before any canvas context acquisition
 * or native handle exposure occurs.
 */

import { ExecutionRoute, EscapeReason } from './route_types.mjs';

/**
 * @typedef {Object} RouteAnalysisInput
 * @property {string} [constructorName] - e.g. 'WebGLRenderer', 'WebGPURenderer', 'CSS2DRenderer'
 * @property {Object} [options] - constructor parameters (e.g. forceWebGL, canvas)
 * @property {boolean} [options.forceWebGL] - explicit flag requesting WebGL backend
 * @property {Object} [analysis] - static AST or module ingestion facts from RubyCrane
 * @property {boolean} [analysis.hasOpaqueGLEscapes] - opaque code accesses GL state/methods
 * @property {boolean} [analysis.hasNativeContextAccess] - synchronous getContext() / native handles
 * @property {boolean} [analysis.hasXRWebGLFallback] - WebXR fallback requiring exact GL
 * @property {Object} [hostCapabilities]
 * @property {boolean} [hostCapabilities.hasWebGPU] - true if WebGPU is available on host
 * @property {boolean} [hostCapabilities.hasWebGL] - true if WebGL is available on host
 * @property {boolean} [specializationAvailable] - true if specialized Rust/Wasm/WebGPU is admitted
 * @property {string} [sourceSpan] - source code location of the construction site
 */

/**
 * Decide the execution route for a renderer construction site.
 * @param {RouteAnalysisInput} input
 * @returns {{ route: string, reasons: string[], sourceSpan: string }}
 */
export function decideRendererRoute(input = {}) {
  const constructorName = input.constructorName || 'WebGLRenderer';
  const options = input.options || {};
  const analysis = input.analysis || {};
  const hostCapabilities = {
    hasWebGPU: typeof navigator !== 'undefined' && 'gpu' in navigator ? true : (input.hostCapabilities?.hasWebGPU ?? false),
    hasWebGL: typeof window !== 'undefined' ? true : (input.hostCapabilities?.hasWebGL ?? true),
    ...input.hostCapabilities,
  };
  const specializationAvailable = input.specializationAvailable ?? false;
  const sourceSpan = input.sourceSpan || 'unknown:0:0';

  const reasons = [];

  // Non-GPU renderers (CSS2D, CSS3D, SVG) are always retained upstream (Plan §5.1 / Bead AC)
  if (['CSS2DRenderer', 'CSS3DRenderer', 'SVGRenderer'].includes(constructorName)) {
    reasons.push(EscapeReason.EXPLICIT_SOURCE_SELECTION);
    return {
      route: ExecutionRoute.RETAINED_UPSTREAM,
      reasons,
      sourceSpan,
    };
  }

  // Explicit WebGL constructor always maps to exact backend
  if (constructorName === 'WebGLRenderer') {
    reasons.push(EscapeReason.EXPLICIT_SOURCE_SELECTION);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
    };
  }

  // Explicit forceWebGL in parameters (e.g. H1 URL toggle ?renderer=webgl)
  if (options.forceWebGL === true) {
    reasons.push(EscapeReason.EXPLICIT_SOURCE_SELECTION);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
    };
  }

  // Opaque GL state / extension / method escapes detected by analysis
  if (analysis.hasOpaqueGLEscapes === true) {
    reasons.push(EscapeReason.OPAQUE_GL_ESCAPE);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
    };
  }

  // Synchronous getContext() / native handle exposure
  if (analysis.hasNativeContextAccess === true) {
    reasons.push(EscapeReason.NATIVE_CONTEXT_ACCESS);
    return {
      route: ExecutionRoute.EXACT_BACKEND,
      reasons,
      sourceSpan,
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
      };
    }
    throw new Error(`Host lacks both WebGPU and WebGL capability at ${sourceSpan}.`);
  }

  // WebGPURenderer on capable WebGPU host
  if (constructorName === 'WebGPURenderer') {
    if (specializationAvailable) {
      return {
        route: ExecutionRoute.SPECIALIZED_WEBGPU,
        reasons: ['specialized-island-admitted'],
        sourceSpan,
      };
    }
    // Interim Phase 0/1 or when specialization is not ready: retained upstream WebGPU
    reasons.push(EscapeReason.SPECIALIZATION_UNAVAILABLE);
    return {
      route: ExecutionRoute.RETAINED_UPSTREAM,
      reasons,
      sourceSpan,
    };
  }

  // Default fallback to retained component
  reasons.push(EscapeReason.SPECIALIZATION_UNAVAILABLE);
  return {
    route: ExecutionRoute.RETAINED_UPSTREAM,
    reasons,
    sourceSpan,
  };
}
