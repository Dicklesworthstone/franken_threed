/**
 * @file route_bridge.mjs
 * Bridge connecting module ingestion graph facts (f3d-04.1) to renderer construction routing (f3d-04.4).
 *
 * Traverses the module graph bundle emitted by buildModuleGraph to aggregate
 * application-wide static routing facts (opaque GL escapes, native context access)
 * and extract renderer construction sites into structured RouteAnalysisInput objects
 * consumed by decideRendererRoute and RendererConstructionRouter.
 */

/**
 * Check if a module URL belongs to upstream Three.js internal build artifacts.
 * Library-internal WebGL implementation calls do not count as application-level escapes (Plan §3.3).
 * @param {string} moduleId
 * @returns {boolean}
 */
export function isInternalLibraryModule(moduleId) {
  if (!moduleId || typeof moduleId !== "string") return false;
  return moduleId.includes("/build/three.") ||
         moduleId.includes("/build/three.webgpu.") ||
         moduleId.includes("/build/three.core.") ||
         moduleId.includes("/build/three.module.");
}

/**
 * Extract aggregated application-wide routing facts from an ingested module graph bundle.
 *
 * @param {Object} bundle - Module graph bundle from buildModuleGraph
 * @param {Object} [options]
 * @param {boolean} [options.includeLibraryInternalEscapes=false] - Whether to include escapes inside Three.js build files
 * @returns {{
 *   hasOpaqueGLEscapes: boolean,
 *   hasNativeContextAccess: boolean,
 *   escapes: Array<Object>,
 *   constructionSites: Array<{
 *     moduleId: string,
 *     constructorName: string,
 *     options: { forceWebGL: boolean, canvas: string | null },
 *     analysis: { hasOpaqueGLEscapes: boolean, hasNativeContextAccess: boolean },
 *     sourceSpan: string
 *   }>
 * }}
 */
export function extractGraphRoutingFacts(bundle, options = {}) {
  if (!bundle || typeof bundle !== "object") {
    throw new TypeError("extractGraphRoutingFacts requires a valid module graph bundle object");
  }

  const includeInternal = options.includeLibraryInternalEscapes ?? false;
  let hasOpaqueGLEscapes = false;
  let hasNativeContextAccess = false;
  let hasUnresolvedContextAccess = false;
  const escapes = [];
  const constructionSites = [];

  const modules = Object.values(bundle.modules || {});

  // Pass 1: Aggregate application-wide escapes across reachable modules
  for (const mod of modules) {
    if (!includeInternal && isInternalLibraryModule(mod.id)) {
      continue;
    }

    const facts = mod.routing_facts || mod.routingFacts || {};
    if (facts.has_opaque_gl_escapes || facts.hasOpaqueGLEscapes) {
      hasOpaqueGLEscapes = true;
    }
    if (facts.has_native_context_access || facts.hasNativeContextAccess) {
      hasNativeContextAccess = true;
    }
    if (facts.has_unresolved_context_access || facts.hasUnresolvedContextAccess) {
      hasUnresolvedContextAccess = true;
      hasNativeContextAccess = true;
    }
    if (Array.isArray(facts.escapes)) {
      for (const esc of facts.escapes) {
        escapes.push({
          ...esc,
          module_id: mod.id,
          moduleId: mod.id,
        });
      }
    }
  }

  // Pass 2: Extract construction sites with application-wide escape awareness
  for (const mod of modules) {
    const sites = mod.renderer_construction_sites || mod.rendererConstructionSites || [];
    for (const site of sites) {
      const constructorName = site.constructor_name || site.constructorName || "WebGLRenderer";
      const rawForceWebGL = site.force_webgl ?? site.forceWebGL ?? site.has_force_webgl ?? site.hasForceWebGL ?? false;
      const isForceWebGLUnresolved = site.force_webgl_unresolved ?? site.forceWebGLUnresolved ?? (rawForceWebGL === 'unresolved');
      const forceWebGL = isForceWebGLUnresolved ? 'unresolved' : Boolean(rawForceWebGL);
      const canvasOption = site.canvas_option ?? site.canvasOption ?? null;
      const span = site.source_span || site.sourceSpan || `${mod.id}:0:0`;

      constructionSites.push({
        moduleId: mod.id,
        constructorName,
        constructor_name: constructorName,
        options: {
          forceWebGL,
          force_webgl: forceWebGL,
          forceWebGLUnresolved: isForceWebGLUnresolved,
          force_webgl_unresolved: isForceWebGLUnresolved,
          canvas: canvasOption,
        },
        analysis: {
          hasOpaqueGLEscapes,
          has_opaque_gl_escapes: hasOpaqueGLEscapes,
          hasNativeContextAccess,
          has_native_context_access: hasNativeContextAccess,
          hasUnresolvedContextAccess,
          has_unresolved_context_access: hasUnresolvedContextAccess,
        },
        sourceSpan: span,
        source_span: span,
      });
    }
  }

  return {
    hasOpaqueGLEscapes,
    has_opaque_gl_escapes: hasOpaqueGLEscapes,
    hasNativeContextAccess,
    has_native_context_access: hasNativeContextAccess,
    hasUnresolvedContextAccess,
    has_unresolved_context_access: hasUnresolvedContextAccess,
    escapes,
    constructionSites,
    construction_sites: constructionSites,
  };
}

/**
 * Prepare an array of RouteAnalysisInput objects for all renderer construction sites
 * in the module graph bundle, ready to be passed to decideRendererRoute.
 *
 * @param {Object} bundle - Ingested module graph bundle
 * @param {Object} [defaults] - Fallback options if no construction sites were statically found
 * @returns {Array<{
 *   constructorName: string,
 *   options: Object,
 *   analysis: Object,
 *   sourceSpan: string,
 *   moduleId?: string
 * }>}
 */
export function prepareRouteInputs(bundle, defaults = {}) {
  const extracted = extractGraphRoutingFacts(bundle, defaults);

  if (extracted.constructionSites.length === 0) {
    return [{
      constructorName: defaults.constructorName || "WebGLRenderer",
      options: defaults.options || {},
      analysis: {
        hasOpaqueGLEscapes: extracted.hasOpaqueGLEscapes,
        has_opaque_gl_escapes: extracted.hasOpaqueGLEscapes,
        hasNativeContextAccess: extracted.hasNativeContextAccess,
        has_native_context_access: extracted.hasNativeContextAccess,
        hasUnresolvedContextAccess: extracted.hasUnresolvedContextAccess,
        has_unresolved_context_access: extracted.hasUnresolvedContextAccess,
        ...defaults.analysis,
      },
      sourceSpan: defaults.sourceSpan || `${bundle.entry_path || "entry"}:0:0`,
    }];
  }

  return extracted.constructionSites.map(site => ({
    constructorName: site.constructorName,
    options: site.options,
    analysis: site.analysis,
    sourceSpan: site.sourceSpan,
    moduleId: site.moduleId,
  }));
}

/**
 * Statically evaluate route decisions for all construction sites in a module graph bundle.
 *
 * @param {Object} bundle - Ingested module graph bundle
 * @param {Function} decideFn - decideRendererRoute function from tools/compat/route_decider.mjs
 * @param {Object} [environment] - Host capabilities and specialization flags
 * @returns {Array<{
 *   moduleId: string,
 *   decision: { route: string, reasons: string[], sourceSpan: string }
 * }>}
 */
export function evaluateGraphRoutes(bundle, decideFn, environment = {}) {
  if (typeof decideFn !== "function") {
    throw new TypeError("evaluateGraphRoutes requires a decideRendererRoute function");
  }

  const inputs = prepareRouteInputs(bundle, environment);
  return inputs.map(input => {
    const decision = decideFn({
      ...input,
      hostCapabilities: environment.hostCapabilities,
      specializationAvailable: environment.specializationAvailable,
    });
    return {
      moduleId: input.moduleId || bundle.entry_path || "entry",
      constructorName: input.constructorName,
      decision,
    };
  });
}

export const evaluateModuleGraphRoutes = evaluateGraphRoutes;

