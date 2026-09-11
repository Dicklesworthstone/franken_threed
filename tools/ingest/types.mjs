/**
 * Schema types and version constants for FrankenThreeD module ingestion (f3d-04).
 */

export const SCHEMA_VERSION = '1.0.0';

/**
 * @typedef {Object} SourceSpan
 * @property {number} line - 1-based source line number
 * @property {number} column - 0-based source column offset
 * @property {number} offset - 0-based byte/character offset
 */

/**
 * Escape record representing native graphics handle exposure or unresolvable dynamic construct.
 * @typedef {Object} EscapeRecord
 * @property {'webgl_context_acquisition' | 'native_context_acquisition' | 'unresolved_native_context_access' | 'opaque_gl_method_call'} type - Classification of escape
 * @property {'nonliteral'} [classification] - Classification for unresolved dynamic expressions
 * @property {boolean} [unresolved] - True if escape represents an unresolvable dynamic construct
 * @property {string} [context_type] - Requested context type identifier (e.g. 'webgl', 'webgl2')
 * @property {string} [contextType] - CamelCase alias for context_type
 * @property {string} [method] - Invoked WebGL method name (e.g. 'getExtension', 'getParameter')
 * @property {SourceSpan} source_span - Source location span of the escape
 * @property {SourceSpan} sourceSpan - CamelCase alias for source_span
 * @property {string} [module_id] - Module ID where escape occurred (populated during graph aggregation)
 * @property {string} [moduleId] - CamelCase alias for module_id
 */

/**
 * Renderer construction site with statically extracted or unresolved constructor options.
 * @typedef {Object} RendererConstructionSite
 * @property {string} constructor_name - Renderer constructor identifier ('WebGPURenderer', 'WebGLRenderer', etc.)
 * @property {string} constructorName - CamelCase alias for constructor_name
 * @property {boolean | 'unresolved'} force_webgl - Whether forceWebGL is enabled, disabled, or dynamic/unresolved
 * @property {boolean | 'unresolved'} forceWebGL - CamelCase alias for force_webgl
 * @property {boolean | 'unresolved'} has_force_webgl - Compatibility alias for force_webgl
 * @property {boolean | 'unresolved'} hasForceWebGL - CamelCase alias for has_force_webgl
 * @property {boolean} force_webgl_unresolved - True if forceWebGL was specified via non-literal expression
 * @property {boolean} forceWebGLUnresolved - CamelCase alias for force_webgl_unresolved
 * @property {string | null} canvas_option - Canvas identifier or variable name if provided in constructor options
 * @property {string | null} canvasOption - CamelCase alias for canvas_option
 * @property {SourceSpan} source_span - Source location span of the new expression
 * @property {SourceSpan} sourceSpan - CamelCase alias for source_span
 */

/**
 * Routing facts extracted per-module for route evaluation and exact boundary detection.
 * @typedef {Object} RoutingFacts
 * @property {boolean} has_opaque_gl_escapes - True if direct WebGL methods or getExtension are invoked
 * @property {boolean} hasOpaqueGLEscapes - CamelCase alias for has_opaque_gl_escapes
 * @property {boolean} has_native_context_access - True if canvas.getContext is invoked
 * @property {boolean} hasNativeContextAccess - CamelCase alias for has_native_context_access
 * @property {boolean} has_unresolved_context_access - True if canvas.getContext is called with a non-literal argument
 * @property {boolean} hasUnresolvedContextAccess - CamelCase alias for has_unresolved_context_access
 * @property {RendererConstructionSite[]} renderer_construction_sites - Renderer construction sites in this module
 * @property {RendererConstructionSite[]} rendererConstructionSites - CamelCase alias for renderer_construction_sites
 * @property {EscapeRecord[]} escapes - Detailed escape records with source spans
 */

/**
 * @typedef {Object} StaticImport
 * @property {string} specifier - Raw import specifier string
 * @property {string} resolved_id - Normalized canonical URL of target module
 * @property {SourceSpan} source_span - Source location span of import declaration
 * @property {Array<{ imported: string, local: string }>} imported_bindings - Imported symbol bindings
 */

/**
 * @typedef {Object} StaticExport
 * @property {string} type - Export declaration type ('named', 'default', 'all')
 * @property {string | null} [specifier] - Re-export target specifier if re-exporting
 * @property {string | null} [resolved_id] - Canonical URL of re-export target if re-exporting
 * @property {string[]} [exported_names] - Exported binding names
 * @property {SourceSpan} source_span - Source location span of export declaration
 */

/**
 * @typedef {Object} DynamicImport
 * @property {'literal' | 'variable' | 'template' | 'expression' | 'empty' | 'finite_set' | 'nonliteral'} classification - Expression category
 * @property {string | null} specifier - Target specifier if literal
 * @property {string | null} resolved_id - Canonical URL of resolved target if literal
 * @property {string[] | null} [finite_set] - Candidate specifier strings if conditional finite set
 * @property {string[] | null} [finiteSet] - CamelCase alias for finite_set
 * @property {string[] | null} [specifiers] - Specifiers list for finite set
 * @property {string[] | null} [candidates] - Candidate list for finite set
 * @property {Array<{ specifier: string, resolved_id: string | null, resolvedId?: string | null, error?: string }>} [resolved_targets] - Resolved candidate targets for finite set
 * @property {Array<{ specifier: string, resolved_id: string | null, resolvedId?: string | null, error?: string }>} [resolvedTargets] - CamelCase alias for resolved_targets
 * @property {string[]} [resolved_ids] - Canonical URLs of resolved candidates if finite set
 * @property {string[]} [resolvedIds] - CamelCase alias for resolved_ids
 * @property {boolean} unresolved - True if target cannot be statically resolved
 * @property {boolean} [claims_closure] - True if complete module closure is statically claimed
 * @property {boolean} [claimsClosure] - CamelCase alias for claims_closure
 * @property {string} [error] - Error message if literal resolution failed
 * @property {SourceSpan} source_span - Source location span of import() expression
 * @property {SourceSpan} [sourceSpan] - CamelCase alias for source_span
 */

/**
 * @typedef {Object} AssetReference
 * @property {string} specifier - Raw asset specifier string
 * @property {SourceSpan} source_span - Source location span of asset reference
 */

/**
 * @typedef {Object} ClassDeclaration
 * @property {string} name - Class name
 * @property {string | null} super_class - Superclass name if derived
 * @property {SourceSpan} source_span - Source location span of class declaration
 */

/**
 * @typedef {Object} PrototypeWrite
 * @property {string} target - Target class or object name
 * @property {string} property - Property name written to prototype
 * @property {SourceSpan} source_span - Source location span of prototype assignment
 */

/**
 * Module graph node representing an ingested ESM or inline script module.
 * @typedef {Object} ModuleNode
 * @property {string} id - Normalized file:// URL of the module
 * @property {string} content_hash - SHA-256 hash of module content with 'sha256:' prefix
 * @property {boolean} is_inline - True if module was parsed from an inline <script> tag
 * @property {string | null} source_path - Absolute filesystem path (null for synthetic modules)
 * @property {string[]} duplicate_content_with - URLs of other modules with identical byte content
 * @property {StaticImport[]} static_imports - Resolved static import declarations
 * @property {StaticExport[]} static_exports - Static export declarations
 * @property {DynamicImport[]} dynamic_imports - Dynamic import expressions (literal and nonliteral)
 * @property {AssetReference[]} asset_references - Detected asset URL references
 * @property {ClassDeclaration[]} classes - Declared classes and hierarchies
 * @property {PrototypeWrite[]} prototype_writes - Detected prototype modifications
 * @property {boolean} has_top_level_side_effects - True if module body contains top-level side effects
 * @property {boolean} has_live_bindings - True if module exports mutable bindings ('let', 'var', or mutated)
 * @property {string[]} mutable_exported_bindings - Names of mutable exported bindings
 * @property {RendererConstructionSite[]} renderer_construction_sites - Renderer construction sites
 * @property {RendererConstructionSite[]} rendererConstructionSites - CamelCase alias
 * @property {RoutingFacts} routing_facts - Extracted routing facts
 * @property {RoutingFacts} routingFacts - CamelCase alias
 */

/**
 * Aggregate summary statistics across the complete ingested module graph bundle.
 * @typedef {Object} ModuleGraphSummary
 * @property {number} total_modules - Total reachable modules in graph
 * @property {number} total_static_imports - Total static import declarations
 * @property {number} total_dynamic_imports - Total dynamic import expressions
 * @property {number} unresolved_dynamic_imports - Nonliteral/unresolved dynamic imports
 * @property {number} cycles_count - Number of detected dependency cycles
 * @property {number} identical_content_pairs - Number of identical content pairs
 * @property {number} total_renderer_construction_sites - Total renderer construction sites
 * @property {number} [total_unresolved_native_context_access] - Modules with unresolved native context access
 * @property {number} [totalUnresolvedNativeContextAccess] - CamelCase alias
 * @property {number} [total_unresolved_force_webgl] - Renderer sites with non-literal forceWebGL
 * @property {number} [totalUnresolvedForceWebGL] - CamelCase alias
 */

/**
 * Complete serialized module graph bundle output by tools/ingest/cli.mjs and module_graph.mjs.
 * @typedef {Object} ModuleGraphBundle
 * @property {'1.0.0'} schema_version - Schema version string (fixed at '1.0.0')
 * @property {'html' | 'module'} entry_type - Type of entry point
 * @property {string} entry_path - Absolute filesystem path to entry file
 * @property {string[]} root_entries - Canonical URLs of root entry modules
 * @property {Record<string, string> | null} import_map - Extracted import map mappings
 * @property {Record<string, ModuleNode>} modules - Graph nodes keyed by normalized module URL
 * @property {string[][]} cycles - Detected dependency cycles (arrays of canonical URLs)
 * @property {ModuleGraphSummary} summary - Summary statistics
 */

export class IngestionResolutionError extends Error {
  /**
   * @param {string} message
   * @param {string} specifier
   * @param {string} referrerUrl
   * @param {{ line: number, column: number, offset: number } | null} [span]
   */
  constructor(message, specifier, referrerUrl, span = null) {
    super(message);
    this.name = 'IngestionResolutionError';
    this.specifier = specifier;
    this.referrerUrl = referrerUrl;
    this.span = span;
  }
}

export class IngestionParseError extends Error {
  /**
   * @param {string} message
   * @param {string} url
   * @param {{ line: number, column: number, offset: number } | null} [span]
   */
  constructor(message, url, span = null) {
    super(message);
    this.name = 'IngestionParseError';
    this.url = url;
    this.span = span;
  }
}
