/**
 * @file index.mjs
 * Main entry point for tools/ingest module ingestion frontend (f3d-04.1).
 */

export { SCHEMA_VERSION, IngestionResolutionError, IngestionParseError } from "./types.mjs";
export { resolveModuleSpecifier, urlToFilePath, isAbsoluteUrl } from "./resolver.mjs";
export { parseHtmlEntries, parseHtmlEntries as parseHtmlEntry, parseSrcsetUrls, stripHtmlComments, stripScriptAndStyleBodies, parseTagAttributes } from "./html_parser.mjs";
export { analyzeModuleAst } from "./ast_analyzer.mjs";
export { buildModuleGraph } from "./module_graph.mjs";
export { bundleWithRollup, f3dRollupPlugin } from "./bundler.mjs";
export {
  isInternalLibraryModule,
  extractGraphRoutingFacts,
  prepareRouteInputs,
  evaluateGraphRoutes,
  evaluateModuleGraphRoutes
} from "./route_bridge.mjs";

export { compileNumericKernel, NumericKernelCompileError, NUMERIC_KERNEL_SECTION } from './numeric_kernel.mjs';
export { instantiateNumericKernel, NumericKernelGuardError } from './numeric_kernel_runtime.mjs';
export { buildNumericKernel } from './numeric_kernel_build.mjs';
