/**
 * Module Graph Builder.
 * Traverses module dependencies starting from HTML or JS entry points,
 * enforces canonical URL identity (preserving distinct instances for identical bytes),
 * resolves cycles and live bindings, classifies dynamic imports,
 * and serializes a versioned structured graph consumed by route analysis.
 */

import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { SCHEMA_VERSION, IngestionResolutionError } from './types.mjs';
import { parseHtmlEntries } from './html_parser.mjs';
import { resolveModuleSpecifier, getBaseUrl } from './resolver.mjs';
import { analyzeModuleAst } from './ast_analyzer.mjs';

/**
 * Computes SHA-256 hash of a string.
 * @param {string} content
 * @returns {string}
 */
function sha256(content) {
  return crypto.createHash('sha256').update(content, 'utf8').digest('hex');
}

/**
 * Finds all elementary cycles in a directed graph using DFS.
 * @param {Map<string, string[]>} adj
 * @returns {string[][]}
 */
function findCycles(adj) {
  const cycles = [];
  const visited = new Set();
  const onStack = new Set();
  const stack = [];

  function dfs(node) {
    visited.add(node);
    onStack.add(node);
    stack.push(node);

    const neighbors = adj.get(node) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        dfs(neighbor);
      } else if (onStack.has(neighbor)) {
        // Cycle detected
        const cycleStartIndex = stack.indexOf(neighbor);
        if (cycleStartIndex !== -1) {
          const cyclePath = stack.slice(cycleStartIndex);
          cyclePath.push(neighbor); // Close cycle
          cycles.push(cyclePath);
        }
      }
    }

    stack.pop();
    onStack.delete(node);
  }

  for (const node of adj.keys()) {
    if (!visited.has(node)) {
      dfs(node);
    }
  }

  return cycles;
}

/**
 * Ingests an application entry point (HTML or ESM module) and builds the complete module graph.
 *
 * @param {string} entryPath - Path to entry file (HTML or JS)
 * @param {Object} [options]
 * @param {string} [options.packageRootUrl] - Base URL for Three.js fallback
 * @returns {Promise<Object>} The versioned module graph bundle
 */
export async function buildModuleGraph(entryPath, options = {}) {
  const resolvedEntryAbs = path.resolve(entryPath);
  if (!fs.existsSync(resolvedEntryAbs)) {
    throw new Error(`Entry file does not exist: "${resolvedEntryAbs}"`);
  }

  const entryUrl = pathToFileURL(resolvedEntryAbs).href;
  const isHtml = entryPath.endsWith('.html') || entryPath.endsWith('.htm');

  let importMap = { imports: {}, scopes: {} };
  const rootEntryIds = [];
  const queue = [];

  // Map of canonical URL -> ModuleNode
  const modules = new Map();
  // Map of sha256 -> Array of canonical URLs with identical bytes
  const contentHashMap = new Map();

  if (isHtml) {
    const htmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const parsedHtml = parseHtmlEntries(htmlContent, entryUrl);
    importMap = parsedHtml.importMap;

    for (const script of parsedHtml.moduleScripts) {
      rootEntryIds.push(script.id);
      queue.push({
        id: script.id,
        isInline: script.inlineContent !== null,
        inlineContent: script.inlineContent,
        offsets: {
          lineOffset: script.startLine - 1,
          columnOffset: script.startColumn - 1,
          charOffset: script.startOffset
        },
        referrerUrl: entryUrl
      });
    }

    if (rootEntryIds.length === 0) {
      throw new Error(`No <script type="module"> entry points found in HTML: "${entryPath}"`);
    }
  } else {
    rootEntryIds.push(entryUrl);
    queue.push({
      id: entryUrl,
      isInline: false,
      inlineContent: null,
      offsets: {},
      referrerUrl: entryUrl
    });
  }

  const mapBaseUrl = entryUrl;

  // Process module queue
  while (queue.length > 0) {
    const item = queue.shift();
    const moduleId = item.id;

    if (modules.has(moduleId)) {
      continue;
    }

    let sourceCode;
    let sourcePath = null;

    if (item.isInline) {
      sourceCode = item.inlineContent;
      sourcePath = resolvedEntryAbs;
    } else {
      sourcePath = fileURLToPath(moduleId);
      try {
        sourceCode = fs.readFileSync(sourcePath, 'utf-8');
      } catch (err) {
        throw new IngestionResolutionError(
          `Failed to read module source file "${sourcePath}": ${err.message}`,
          moduleId,
          item.referrerUrl
        );
      }
    }

    const hash = sha256(sourceCode);

    // Track identical content across distinct URLs (never merge them!)
    const duplicateWith = [];
    if (contentHashMap.has(hash)) {
      const existingUrls = contentHashMap.get(hash);
      for (const existingUrl of existingUrls) {
        duplicateWith.push(existingUrl);
        // Link reciprocally
        const existingNode = modules.get(existingUrl);
        if (existingNode && !existingNode.duplicate_content_with.includes(moduleId)) {
          existingNode.duplicate_content_with.push(moduleId);
        }
      }
      existingUrls.push(moduleId);
    } else {
      contentHashMap.set(hash, [moduleId]);
    }

    // Run AST analysis
    const analysis = analyzeModuleAst(sourceCode, moduleId, item.offsets || {});

    // Resolve static imports
    const resolvedStaticImports = [];
    for (const imp of analysis.staticImports) {
      const resolvedTarget = resolveModuleSpecifier(imp.specifier, moduleId, importMap, {
        mapBaseUrl,
        packageRootUrl: options.packageRootUrl,
        span: imp.sourceSpan
      });

      resolvedStaticImports.push({
        specifier: imp.specifier,
        resolved_id: resolvedTarget,
        source_span: imp.sourceSpan,
        imported_bindings: imp.importedBindings
      });

      // Queue target if not already visited
      if (!modules.has(resolvedTarget)) {
        queue.push({
          id: resolvedTarget,
          isInline: false,
          inlineContent: null,
          offsets: {},
          referrerUrl: moduleId
        });
      }
    }

    // Resolve dynamic imports
    const resolvedDynamicImports = [];
    for (const dyn of analysis.dynamicImports) {
      if (dyn.classification === 'literal' && dyn.specifier) {
        let resolvedTarget = null;
        try {
          resolvedTarget = resolveModuleSpecifier(dyn.specifier, moduleId, importMap, {
            mapBaseUrl,
            packageRootUrl: options.packageRootUrl,
            span: dyn.sourceSpan
          });
        } catch (err) {
          // If literal dynamic import cannot be resolved, mark unresolved with error
          resolvedDynamicImports.push({
            classification: 'literal',
            specifier: dyn.specifier,
            resolved_id: null,
            unresolved: true,
            error: err.message,
            source_span: dyn.sourceSpan
          });
          continue;
        }

        resolvedDynamicImports.push({
          classification: 'literal',
          specifier: dyn.specifier,
          resolved_id: resolvedTarget,
          unresolved: false,
          source_span: dyn.sourceSpan
        });

        if (!modules.has(resolvedTarget)) {
          queue.push({
            id: resolvedTarget,
            isInline: false,
            inlineContent: null,
            offsets: {},
            referrerUrl: moduleId
          });
        }
      } else {
        // Nonliteral dynamic import: classify and preserve as unresolved without claiming closure
        resolvedDynamicImports.push({
          classification: dyn.classification,
          specifier: null,
          resolved_id: null,
          unresolved: true,
          source_span: dyn.sourceSpan
        });
      }
    }

    const node = {
      id: moduleId,
      content_hash: `sha256:${hash}`,
      is_inline: item.isInline,
      source_path: sourcePath,
      duplicate_content_with: duplicateWith,
      static_imports: resolvedStaticImports,
      static_exports: analysis.staticExports,
      dynamic_imports: resolvedDynamicImports,
      asset_references: analysis.assetReferences,
      classes: analysis.classDeclarations,
      prototype_writes: analysis.prototypeWrites,
      has_top_level_side_effects: analysis.hasTopLevelSideEffects,
      has_live_bindings: analysis.hasLiveBindings,
      mutable_exported_bindings: analysis.mutableExportedBindings
    };

    modules.set(moduleId, node);
  }

  // Build adjacency map for cycle detection
  const adjacency = new Map();
  for (const [id, node] of modules.entries()) {
    const targets = node.static_imports.map(i => i.resolved_id);
    adjacency.set(id, targets);
  }

  const cycles = findCycles(adjacency);

  // Compute summary stats
  let totalStaticImports = 0;
  let totalDynamicImports = 0;
  let unresolvedDynamicImports = 0;
  let identicalContentPairs = 0;

  for (const node of modules.values()) {
    totalStaticImports += node.static_imports.length;
    totalDynamicImports += node.dynamic_imports.length;
    unresolvedDynamicImports += node.dynamic_imports.filter(d => d.unresolved).length;
    if (node.duplicate_content_with.length > 0) {
      identicalContentPairs += node.duplicate_content_with.length;
    }
  }

  // Deduplicate pair count (each pair counted twice)
  identicalContentPairs = Math.floor(identicalContentPairs / 2);

  const bundle = {
    schema_version: SCHEMA_VERSION,
    entry_type: isHtml ? 'html' : 'module',
    entry_path: resolvedEntryAbs,
    root_entries: rootEntryIds,
    import_map: importMap,
    modules: Object.fromEntries(modules.entries()),
    cycles,
    summary: {
      total_modules: modules.size,
      total_static_imports: totalStaticImports,
      total_dynamic_imports: totalDynamicImports,
      unresolved_dynamic_imports: unresolvedDynamicImports,
      cycles_count: cycles.length,
      identical_content_pairs: identicalContentPairs
    }
  };

  return bundle;
}
