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
import { resolveModuleSpecifier, getBaseUrl, urlToFilePath } from './resolver.mjs';
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
 * Checks if a URL string has an external (non-filesystem) scheme.
 * @param {string} url
 * @returns {boolean}
 */
function isExternalUrl(url) {
  if (typeof url !== 'string') return false;
  return url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:');
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
  // Set of canonical external URLs (http:, https:, data:)
  const externalModules = new Set();

  if (isHtml) {
    const htmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const parsedHtml = parseHtmlEntries(htmlContent, entryUrl);
    importMap = parsedHtml.importMap;

    for (const script of parsedHtml.moduleScripts) {
      // Preserve the script element in emitted HTML; it has no executable module.
      if (script.src === '') continue;
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

    if (parsedHtml.moduleScripts.length === 0) {
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

    if (modules.has(moduleId) || externalModules.has(moduleId)) {
      continue;
    }

    if (isExternalUrl(moduleId)) {
      externalModules.add(moduleId);
      continue;
    }

    let sourceCode;
    let sourcePath = null;

    if (item.isInline) {
      sourceCode = item.inlineContent;
      sourcePath = resolvedEntryAbs;
    } else {
      sourcePath = urlToFilePath(moduleId);
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
            resolvedId: null,
            unresolved: true,
            claims_closure: false,
            claimsClosure: false,
            error: err.message,
            source_span: dyn.sourceSpan,
            sourceSpan: dyn.sourceSpan
          });
          continue;
        }

        resolvedDynamicImports.push({
          classification: 'literal',
          specifier: dyn.specifier,
          resolved_id: resolvedTarget,
          resolvedId: resolvedTarget,
          unresolved: false,
          claims_closure: true,
          claimsClosure: true,
          source_span: dyn.sourceSpan,
          sourceSpan: dyn.sourceSpan
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
      } else if (dyn.classification === 'finite_set' && (dyn.finite_set || dyn.specifiers)) {
        const candidates = dyn.finite_set || dyn.specifiers;
        const resolvedTargets = [];
        let allResolved = true;

        for (const cand of candidates) {
          try {
            const resolvedTarget = resolveModuleSpecifier(cand, moduleId, importMap, {
              mapBaseUrl,
              packageRootUrl: options.packageRootUrl,
              span: dyn.sourceSpan
            });
            resolvedTargets.push({
              specifier: cand,
              resolved_id: resolvedTarget,
              resolvedId: resolvedTarget,
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
          } catch (err) {
            allResolved = false;
            resolvedTargets.push({
              specifier: cand,
              resolved_id: null,
              resolvedId: null,
              error: err.message
            });
          }
        }

        resolvedDynamicImports.push({
          classification: 'finite_set',
          specifier: null,
          specifiers: candidates,
          candidates,
          finite_set: candidates,
          finiteSet: candidates,
          resolved_targets: resolvedTargets,
          resolvedTargets,
          resolved_ids: resolvedTargets.map(t => t.resolved_id).filter(Boolean),
          resolvedIds: resolvedTargets.map(t => t.resolvedId).filter(Boolean),
          unresolved: !allResolved,
          claims_closure: allResolved,
          claimsClosure: allResolved,
          source_span: dyn.sourceSpan,
          sourceSpan: dyn.sourceSpan
        });
      } else {
        // Nonliteral dynamic import: classify and preserve as unresolved without claiming closure
        resolvedDynamicImports.push({
          classification: dyn.classification,
          specifier: null,
          resolved_id: null,
          resolvedId: null,
          unresolved: true,
          claims_closure: false,
          claimsClosure: false,
          source_span: dyn.sourceSpan,
          sourceSpan: dyn.sourceSpan
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
      mutable_exported_bindings: analysis.mutableExportedBindings,
      renderer_construction_sites: analysis.renderer_construction_sites,
      rendererConstructionSites: analysis.renderer_construction_sites,
      routing_facts: analysis.routing_facts,
      routingFacts: analysis.routing_facts
    };

    modules.set(moduleId, node);
  }

  // Determine tainted modules that reach external or unresolved code using a reverse-edge worklist
  const tainted = new Set();
  const incoming = new Map();

  for (const id of modules.keys()) {
    incoming.set(id, new Set());
  }

  const worklist = [];

  for (const [id, node] of modules.entries()) {
    const hasExternalStatic = node.static_imports.some(imp => isExternalUrl(imp.resolved_id));
    const hasExternalDyn = node.dynamic_imports.some(dyn => {
      if (dyn.unresolved) return true;
      if (dyn.resolved_id && isExternalUrl(dyn.resolved_id)) return true;
      if (dyn.resolved_targets && dyn.resolved_targets.some(t => isExternalUrl(t.resolved_id))) return true;
      return false;
    });

    if (hasExternalStatic || hasExternalDyn) {
      tainted.add(id);
      worklist.push(id);
    }

    for (const imp of node.static_imports) {
      if (modules.has(imp.resolved_id)) {
        incoming.get(imp.resolved_id).add(id);
      }
    }
    for (const dyn of node.dynamic_imports) {
      if (dyn.resolved_id && modules.has(dyn.resolved_id)) {
        incoming.get(dyn.resolved_id).add(id);
      }
      if (dyn.resolved_targets) {
        for (const t of dyn.resolved_targets) {
          if (t.resolved_id && modules.has(t.resolved_id)) {
            incoming.get(t.resolved_id).add(id);
          }
        }
      }
    }
  }

  while (worklist.length > 0) {
    const curr = worklist.pop();
    const dependents = incoming.get(curr) || [];
    for (const dep of dependents) {
      if (!tainted.has(dep)) {
        tainted.add(dep);
        worklist.push(dep);
      }
    }
  }

  for (const node of modules.values()) {
    for (const dyn of node.dynamic_imports) {
      if (dyn.unresolved) {
        dyn.claims_closure = false;
        dyn.claimsClosure = false;
        continue;
      }
      if (dyn.classification === 'literal') {
        const target = dyn.resolved_id;
        const claims = Boolean(target && !isExternalUrl(target) && !tainted.has(target));
        dyn.claims_closure = claims;
        dyn.claimsClosure = claims;
      } else if (dyn.classification === 'finite_set') {
        const targets = (dyn.resolved_targets || []).map(t => t.resolved_id);
        const allClosed = targets.length > 0 && targets.every(t => t && !isExternalUrl(t) && !tainted.has(t));
        dyn.claims_closure = allClosed;
        dyn.claimsClosure = allClosed;
      }
    }
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
  let totalRendererConstructionSites = 0;
  let totalUnresolvedNativeContextAccess = 0;
  let totalUnresolvedForceWebGL = 0;

  for (const node of modules.values()) {
    totalStaticImports += node.static_imports.length;
    totalDynamicImports += node.dynamic_imports.length;
    unresolvedDynamicImports += node.dynamic_imports.filter(d => d.unresolved).length;
    totalRendererConstructionSites += node.renderer_construction_sites ? node.renderer_construction_sites.length : 0;
    const facts = node.routing_facts || node.routingFacts;
    if (facts && (facts.has_unresolved_context_access || facts.hasUnresolvedContextAccess)) {
      totalUnresolvedNativeContextAccess++;
    }
    const sites = node.renderer_construction_sites || node.rendererConstructionSites;
    if (sites) {
      for (const site of sites) {
        if (site.forceWebGL === 'unresolved' || site.force_webgl === 'unresolved' || site.forceWebGLUnresolved || site.force_webgl_unresolved) {
          totalUnresolvedForceWebGL++;
        }
      }
    }
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
    external_modules: Array.from(externalModules).sort(),
    modules: Object.fromEntries(modules.entries()),
    cycles,
    summary: {
      total_modules: modules.size,
      total_external_modules: externalModules.size,
      total_static_imports: totalStaticImports,
      total_dynamic_imports: totalDynamicImports,
      unresolved_dynamic_imports: unresolvedDynamicImports,
      cycles_count: cycles.length,
      identical_content_pairs: identicalContentPairs,
      total_renderer_construction_sites: totalRendererConstructionSites,
      total_unresolved_native_context_access: totalUnresolvedNativeContextAccess,
      totalUnresolvedNativeContextAccess,
      total_unresolved_force_webgl: totalUnresolvedForceWebGL,
      totalUnresolvedForceWebGL
    }
  };

  return bundle;
}
