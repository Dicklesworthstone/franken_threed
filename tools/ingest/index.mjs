/**
 * @file index.mjs
 * Main entry point for tools/ingest module ingestion frontend (f3d-04.1).
 */

export { decodeGltfGeometry } from "./animation_geometry.mjs";
export { createCpuGltfAnimationModel, decodeGltfAnimationModel } from "./animation_model.mjs";
export { analyzeModuleAst } from "./ast_analyzer.mjs";
export { bundleWithRollup, f3dRollupPlugin } from "./bundler.mjs";
export {
  parseHtmlEntries,
  parseHtmlEntries as parseHtmlEntry,
  parseSrcsetUrls,
  parseTagAttributes,
  stripHtmlComments,
  stripScriptAndStyleBodies,
} from "./html_parser.mjs";
export { buildModuleGraph } from "./module_graph.mjs";
export {
  compileNumericKernel,
  NUMERIC_KERNEL_SECTION,
  NumericKernelCompileError,
} from "./numeric_kernel.mjs";
export { buildNumericKernel } from "./numeric_kernel_build.mjs";
export { instantiateNumericKernel, NumericKernelGuardError } from "./numeric_kernel_runtime.mjs";
export { isAbsoluteUrl, resolveModuleSpecifier, urlToFilePath } from "./resolver.mjs";
export {
  evaluateGraphRoutes,
  evaluateModuleGraphRoutes,
  extractGraphRoutingFacts,
  isInternalLibraryModule,
  prepareRouteInputs,
} from "./route_bridge.mjs";
export { IngestionParseError, IngestionResolutionError, SCHEMA_VERSION } from "./types.mjs";
// GPU model construction is an opt-in import from animation_model_gpu.mjs.
