/**
 * @file tools/compat-facade/index.mjs
 * Compatibility facade and export router generator for Three.js r186 (Plan §3.4, §4.7, §5.1, §5.12, §14.2).
 *
 * Implements bead f3d-04-module-routing-exact-boundaries-6mv.2:
 * - Enumerates every pinned r186 package export entry from reconciled package data:
 *   root ESM, root CJS, three/webgpu, three/tsl, three/addons wildcard, three/src wildcard, examples/jsm wildcard, assets.
 * - Extracts complete export surface (named exports + default export presence) via Acorn without Rollup.
 * - Generates facade module map routing each entry to the retained upstream file with the identical export surface.
 * - Preserves synchronous CJS execution without Promises.
 * - Zero GPU/DOM/Wasm imports for CPU-only entry points.
 * - Truthful attestations: retained upstream JS execution, no GPU acceleration claim.
 */

import fs from 'node:fs';
import path from 'node:path';
import * as acorn from 'acorn';

/**
 * Standard no-claim attestation for retained upstream compatibility facades (§5.1, §5.12).
 */
export const FACADE_NO_CLAIM_ATTESTATION =
  'Retained upstream JS execution preserves full component functionality; not a Rust rewrite and does not claim GPU acceleration (Plan §5.1, §5.12).';

/**
 * Categorize a package export entry based on its key and condition.
 * @param {string} exportKey
 * @param {string} condition
 * @param {string} target
 * @returns {'root_esm' | 'root_cjs' | 'webgpu' | 'tsl' | 'addons' | 'addons_wildcard' | 'src_wildcard' | 'examples_jsm_wildcard' | 'asset'}
 */
export function categorizeExportEntry(exportKey, condition, target) {
  const isCode = target.endsWith('.js') || target.endsWith('.cjs');
  if (!isCode) {
    return 'asset';
  }
  if (exportKey === '.') {
    return condition === 'require' ? 'root_cjs' : 'root_esm';
  }
  if (exportKey === './webgpu') return 'webgpu';
  if (exportKey === './tsl') return 'tsl';
  if (exportKey === './addons') return 'addons';
  if (exportKey.startsWith('./addons/')) return 'addons_wildcard';
  if (exportKey.startsWith('./src/')) return 'src_wildcard';
  if (exportKey.startsWith('./examples/jsm/')) return 'examples_jsm_wildcard';
  return 'asset';
}

/**
 * Enumerates every pinned r186 package export entry from the reconciled package data.
 * @param {Object} [options]
 * @param {string} [options.reconcileJsonPath] Path to reconcile.json artifact (defaults to evidence/01.1/reconcile.json)
 * @param {string} [options.packageDir] Path to upstream package root (defaults to upstream/three.js)
 * @param {Object} [options.reconciliationData] In-memory reconciliation data object
 * @returns {Array<Object>} List of reconciled export entry descriptors
 */
export function enumeratePackageExportEntries(options = {}) {
  let data = options.reconciliationData;

  if (!data) {
    const jsonPath = options.reconcileJsonPath || path.resolve(process.cwd(), 'evidence/01.1/reconcile.json');
    if (fs.existsSync(jsonPath)) {
      data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    } else {
      throw new Error(`Reconciliation data not found at: ${jsonPath}. Run reconcile_package.mjs first.`);
    }
  }

  const rawEntries = data.resolved_exports || [];
  const packageDir = options.packageDir || 'upstream/three.js';

  return rawEntries.map((e, index) => {
    const category = categorizeExportEntry(e.exportKey, e.condition, e.target);
    const isCjs = category === 'root_cjs' || e.target.endsWith('.cjs') || e.condition === 'require';
    const isAsset = category === 'asset' || (!e.target.endsWith('.js') && !e.target.endsWith('.cjs'));
    const moduleType = isAsset ? 'asset' : isCjs ? 'cjs' : 'esm';

    return {
      index,
      exportKey: e.exportKey,
      condition: e.condition,
      target: e.target,
      category,
      moduleType,
      isWildcardExpansion: Boolean(e.isWildcardExpansion),
      retainedRelativePath: path.posix.join(packageDir.replace(/\\/g, '/'), e.target),
      retainedAbsolutePath: path.resolve(process.cwd(), packageDir, e.target),
    };
  });
}

/**
 * Extracts the full export surface (named exports + default export presence) of an upstream module.
 * Traverses local `export * from ...` chains recursively.
 *
 * @param {string} filePath Absolute or relative path to the module file
 * @param {Object} [options]
 * @param {Map<string, Object>} [options.cache] Module export cache
 * @param {boolean} [options.recursive=true] Whether to resolve export * declarations
 * @returns {{ named: string[], hasDefault: boolean, totalCount: number, isCJS: boolean }}
 */
export function extractModuleExportSurface(filePath, options = {}) {
  const cache = options.cache || new Map();
  const absPath = path.resolve(filePath);

  if (cache.has(absPath)) {
    return cache.get(absPath);
  }

  const result = {
    named: new Set(),
    hasDefault: false,
    isCJS: absPath.endsWith('.cjs'),
  };

  // Guard against circular re-exports
  cache.set(absPath, {
    named: [],
    hasDefault: false,
    totalCount: 0,
    isCJS: result.isCJS,
  });

  if (!fs.existsSync(absPath)) {
    return { named: [], hasDefault: false, totalCount: 0, isCJS: result.isCJS };
  }

  const code = fs.readFileSync(absPath, 'utf-8');

  // CommonJS files do not declare static ES exports
  if (result.isCJS) {
    const formatted = {
      named: [],
      hasDefault: false,
      totalCount: 0,
      isCJS: true,
    };
    cache.set(absPath, formatted);
    return formatted;
  }

  const ast = acorn.parse(code, {
    ecmaVersion: 'latest',
    sourceType: 'module',
  });

  for (const node of ast.body) {
    if (node.type === 'ExportDefaultDeclaration') {
      result.hasDefault = true;
    } else if (node.type === 'ExportNamedDeclaration') {
      if (node.declaration) {
        if (node.declaration.id && node.declaration.id.name) {
          result.named.add(node.declaration.id.name);
        } else if (node.declaration.declarations) {
          for (const d of node.declaration.declarations) {
            if (d.id && d.id.name) {
              result.named.add(d.id.name);
            }
          }
        }
      }
      if (Array.isArray(node.specifiers)) {
        for (const spec of node.specifiers) {
          const name = spec.exported ? spec.exported.name : null;
          if (name === 'default') {
            result.hasDefault = true;
          } else if (name) {
            result.named.add(name);
          }
        }
      }
    } else if (node.type === 'ExportAllDeclaration') {
      if (node.exported && node.exported.name) {
        // export * as namespace from '...'
        result.named.add(node.exported.name);
      } else if (options.recursive !== false && node.source && typeof node.source.value === 'string') {
        // export * from '...'
        const targetPath = path.resolve(path.dirname(absPath), node.source.value);
        const sub = extractModuleExportSurface(targetPath, { cache, recursive: true });
        for (const n of sub.named) {
          result.named.add(n);
        }
      }
    }
  }

  const sortedNamed = Array.from(result.named).sort();
  const formatted = {
    named: sortedNamed,
    hasDefault: result.hasDefault,
    totalCount: sortedNamed.length + (result.hasDefault ? 1 : 0),
    isCJS: false,
  };

  cache.set(absPath, formatted);
  return formatted;
}

/**
 * Generates the source code for a compatibility facade module.
 *
 * @param {Object} entry Reconciled package export entry descriptor
 * @param {Object} exportSurface Extracted export surface of retained target
 * @param {Object} [options]
 * @param {string} [options.retainedSpecifier] Relative or bare import specifier to target
 * @returns {string} Generated JavaScript source code
 */
export function generateFacadeModule(entry, exportSurface, options = {}) {
  const retainedSpecifier =
    options.retainedSpecifier ||
    `./${entry.retainedRelativePath.split(path.sep).join('/')}`;

  const header = [
    '/**',
    ` * FrankenThreeD compatibility facade: ${entry.exportKey} [${entry.condition}]`,
    ` * Target: ${entry.target}`,
    ' * Pinned Three.js r186 retained export surface',
    ' *',
    ` * Attestation: ${FACADE_NO_CLAIM_ATTESTATION}`,
    ' */',
  ].join('\n');

  // CommonJS root entry point
  if (entry.moduleType === 'cjs' || entry.condition === 'require') {
    return [
      header,
      "'use strict';",
      '',
      `module.exports = require('${retainedSpecifier}');`,
      '',
    ].join('\n');
  }

  // Pure side-effects module (no exports)
  if (exportSurface.named.length === 0 && !exportSurface.hasDefault) {
    return [
      header,
      `export * from '${retainedSpecifier}';`,
      '',
    ].join('\n');
  }

  const lines = [header];

  // Wildcard re-export for complete library compatibility
  lines.push(`export * from '${retainedSpecifier}';`);

  // Explicit default re-export if retained target exports default
  if (exportSurface.hasDefault) {
    lines.push(`export { default } from '${retainedSpecifier}';`);
  }

  // Explicit named re-exports for 100% transparent AST inspection and autocomplete
  if (exportSurface.named.length > 0) {
    lines.push(`export {`);
    lines.push(`  ${exportSurface.named.join(',\n  ')}`);
    lines.push(`} from '${retainedSpecifier}';`);
  }

  lines.push('');
  return lines.join('\n');
}

/**
 * Parses a generated facade module using Acorn (parse-only, no bundler)
 * and extracts its declared re-export surface.
 *
 * @param {string} facadeSource JavaScript source of the facade module
 * @param {'esm' | 'cjs'} [moduleType='esm']
 * @returns {{ named: string[], hasDefault: boolean, hasWildcard: boolean, isCJS: boolean }}
 */
export function parseFacadeExportSurface(facadeSource, moduleType = 'esm') {
  if (moduleType === 'cjs') {
    const ast = acorn.parse(facadeSource, {
      ecmaVersion: 'latest',
      sourceType: 'script',
    });

    let assignsModuleExports = false;
    let callsRequire = false;

    for (const node of ast.body) {
      if (node.type === 'VariableDeclaration') {
        for (const decl of node.declarations) {
          if (
            decl.init &&
            decl.init.type === 'CallExpression' &&
            decl.init.callee.name === 'require'
          ) {
            callsRequire = true;
          }
        }
      }
      if (
        node.type === 'ExpressionStatement' &&
        node.expression.type === 'AssignmentExpression'
      ) {
        const left = node.expression.left;
        if (
          left.type === 'MemberExpression' &&
          left.object.name === 'module' &&
          left.property.name === 'exports'
        ) {
          assignsModuleExports = true;
          if (
            node.expression.right.type === 'CallExpression' &&
            node.expression.right.callee.name === 'require'
          ) {
            callsRequire = true;
          }
        }
      }
    }

    return {
      named: [],
      hasDefault: false,
      hasWildcard: false,
      isCJS: true,
      assignsModuleExports,
      callsRequire,
    };
  }

  const ast = acorn.parse(facadeSource, {
    ecmaVersion: 'latest',
    sourceType: 'module',
  });

  const named = new Set();
  let hasDefault = false;
  let hasWildcard = false;

  for (const node of ast.body) {
    if (node.type === 'ExportAllDeclaration') {
      hasWildcard = true;
    } else if (node.type === 'ExportNamedDeclaration') {
      for (const spec of node.specifiers) {
        const exportedName = spec.exported.name;
        if (exportedName === 'default') {
          hasDefault = true;
        } else {
          named.add(exportedName);
        }
      }
    }
  }

  return {
    named: Array.from(named).sort(),
    hasDefault,
    hasWildcard,
    isCJS: false,
  };
}

/**
 * Builds the complete facade module map for every pinned Three.js r186 package export.
 *
 * @param {Object} [options]
 * @param {string} [options.reconcileJsonPath]
 * @param {string} [options.packageDir]
 * @returns {{
 *   entries: Array<Object>,
 *   map: Map<string, Object>,
 *   summary: Object
 * }}
 */
export function buildFacadeModuleMap(options = {}) {
  const entries = enumeratePackageExportEntries(options);
  const cache = new Map();
  const facadeMap = new Map();

  const categoryCounts = {};
  let esmCount = 0;
  let cjsCount = 0;
  let assetCount = 0;
  let totalNamedExports = 0;
  let defaultCount = 0;

  for (const entry of entries) {
    categoryCounts[entry.category] = (categoryCounts[entry.category] || 0) + 1;

    if (entry.moduleType === 'asset') {
      assetCount++;
      const item = {
        ...entry,
        exportSurface: { named: [], hasDefault: false, totalCount: 0, isCJS: false },
        facadeSource: null,
        noClaimAttestation: FACADE_NO_CLAIM_ATTESTATION,
      };
      facadeMap.set(`${entry.exportKey}#${entry.condition}`, item);
      continue;
    }

    if (entry.moduleType === 'cjs') {
      cjsCount++;
      const surface = { named: [], hasDefault: false, totalCount: 0, isCJS: true };
      const facadeSource = generateFacadeModule(entry, surface, options);
      const item = {
        ...entry,
        exportSurface: surface,
        facadeSource,
        noClaimAttestation: FACADE_NO_CLAIM_ATTESTATION,
      };
      facadeMap.set(`${entry.exportKey}#${entry.condition}`, item);
      continue;
    }

    // ESM module
    esmCount++;
    const surface = extractModuleExportSurface(entry.retainedAbsolutePath, { cache });
    totalNamedExports += surface.named.length;
    if (surface.hasDefault) defaultCount++;

    const facadeSource = generateFacadeModule(entry, surface, options);
    const item = {
      ...entry,
      exportSurface: surface,
      facadeSource,
      noClaimAttestation: FACADE_NO_CLAIM_ATTESTATION,
    };
    facadeMap.set(`${entry.exportKey}#${entry.condition}`, item);
  }

  const summary = {
    total_entries: entries.length,
    esm_entries: esmCount,
    cjs_entries: cjsCount,
    asset_entries: assetCount,
    total_named_exports_across_esm: totalNamedExports,
    entries_with_default: defaultCount,
    by_category: categoryCounts,
  };

  return {
    entries,
    map: facadeMap,
    summary,
  };
}

/**
 * Emits the generated facade modules to a target directory on disk.
 *
 * @param {string} targetDir Destination directory
 * @param {Object} [options]
 * @returns {Promise<{ emittedFiles: string[], count: number, summary: Object }>}
 */
export async function emitFacadeFiles(targetDir, options = {}) {
  const resolvedTargetDir = fs.existsSync(targetDir) ? fs.realpathSync(targetDir) : path.resolve(targetDir);
  const result = buildFacadeModuleMap(options);
  const emittedFiles = [];

  for (const [key, item] of result.map.entries()) {
    if (!item.facadeSource) continue;

    let subPath;
    if (item.exportKey === '.') {
      subPath = item.condition === 'require' ? 'index.cjs' : 'index.js';
    } else {
      subPath = item.exportKey.replace(/^\.\//, '');
      if (!path.extname(subPath)) {
        subPath += '.js';
      }
    }

    const outPath = path.resolve(resolvedTargetDir, subPath);
    const outDir = path.dirname(outPath);
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }

    // Adjust relative specifier to retained file from outPath
    const retainedRel = path.relative(outDir, item.retainedAbsolutePath).split(path.sep).join('/');
    const retainedSpecifier = retainedRel.startsWith('.') ? retainedRel : `./${retainedRel}`;

    const adjustedSource = generateFacadeModule(item, item.exportSurface, {
      ...options,
      retainedSpecifier,
    });

    fs.writeFileSync(outPath, adjustedSource, 'utf-8');
    emittedFiles.push(outPath);
  }

  return {
    emittedFiles,
    count: emittedFiles.length,
    summary: result.summary,
  };
}

/**
 * Creates the development mode import map routing Three.js packages to the compatibility facade.
 * Preserves the source map's selected targets (e.g. binding both 'three' and 'three/webgpu' to
 * the same routed WebGPU module singleton when the source selected three.webgpu.js for both).
 *
 * NOTE (H1 Invariant): Upstream H1 (examples/webgpu_performance_renderbundle.html) explicitly maps
 * both 'three' and 'three/webgpu' to 'build/three.webgpu.js'. For H1 and WebGPU-first applications,
 * callers must supply `sourceImportMap` (or set `routeThreeToWebgpu: true`) so that both specifiers
 * remain bound to the identical `/compat-facade/webgpu.js` module singleton. Without `sourceImportMap`,
 * 'three' defaults to `/compat-facade/three.js` (root ESM build/three.module.js) for standard Three.js apps.
 *
 * @param {Object} [options]
 * @param {string} [options.baseUrl=''] Optional base URL prefix (e.g. '' or 'http://127.0.0.1:8080')
 * @param {Object} [options.sourceImportMap=null] Original import map from source HTML
 * @param {boolean} [options.routeThreeToWebgpu=false] Explicitly route 'three' to webgpu.js even without sourceImportMap
 * @returns {{ imports: Record<string, string> }}
 */
export function createDevImportMap({ baseUrl = '', sourceImportMap = null, routeThreeToWebgpu = false } = {}) {
  const base = baseUrl.replace(/\/+$/, '');
  const sourceImports = sourceImportMap?.imports || {};

  // If source import map specifically selected three.webgpu.js for 'three', preserve that selection
  const threeTarget = sourceImports['three'];
  const routesThreeToWebgpu =
    routeThreeToWebgpu ||
    (typeof threeTarget === 'string' && threeTarget.includes('three.webgpu.js'));

  return {
    imports: {
      three: routesThreeToWebgpu ? `${base}/compat-facade/webgpu.js` : `${base}/compat-facade/three.js`,
      'three/webgpu': `${base}/compat-facade/webgpu.js`,
      'three/tsl': `${base}/compat-facade/tsl.js`,
      'three/addons/': `${base}/compat-facade/addons/`,
    },
  };
}

/**
 * Transforms an HTML document's `<script type="importmap">` block to route
 * module specifiers through the FrankenThreeD compatibility facade in development mode.
 *
 * Preserves the source map's actual selected targets (binding both 'three' and 'three/webgpu'
 * to the same routed module singleton in H1) while preserving the upstream application script,
 * markup, styles, and assets 100% unchanged.
 *
 * @param {string} htmlContent Full HTML document source
 * @param {Object} [options]
 * @param {string} [options.baseUrl='']
 * @returns {string} Transformed HTML document
 */
export function transformHtmlImportMap(htmlContent, { baseUrl = '' } = {}) {
  const importMapRegex = /<script\s+type=["']importmap["']>([\s\S]*?)<\/script>/i;
  const match = importMapRegex.exec(htmlContent);

  if (!match) {
    throw new Error('No <script type="importmap"> tag found in HTML document');
  }

  let sourceImportMap = null;
  try {
    sourceImportMap = JSON.parse(match[1]);
  } catch {}

  const base = baseUrl.replace(/\/+$/, '');
  const sourceImports = sourceImportMap?.imports || {};
  const newImports = {};

  for (const [specifier, target] of Object.entries(sourceImports)) {
    if (typeof target === 'string') {
      if (target.includes('three.webgpu.js')) {
        newImports[specifier] = `${base}/compat-facade/webgpu.js`;
      } else if (target.includes('three.tsl.js')) {
        newImports[specifier] = `${base}/compat-facade/tsl.js`;
      } else if (target.includes('three.module.js') || target.endsWith('/three.js')) {
        newImports[specifier] = `${base}/compat-facade/three.js`;
      } else if (specifier === 'three/addons/' || target.endsWith('/jsm/') || target === './jsm/') {
        newImports[specifier] = `${base}/compat-facade/addons/`;
      } else {
        newImports[specifier] = target;
      }
    } else {
      newImports[specifier] = target;
    }
  }

  // Ensure mandatory three and three/webgpu mappings if missing from source
  if (!newImports['three/webgpu']) {
    newImports['three/webgpu'] = `${base}/compat-facade/webgpu.js`;
  }
  if (!newImports['three']) {
    newImports['three'] = newImports['three/webgpu'];
  }

  const devMap = { imports: newImports };
  const formattedJson = JSON.stringify(devMap, null, '\t\t\t\t')
    .split('\n')
    .map((line, idx) => (idx === 0 ? line : `\t\t\t${line}`))
    .join('\n');

  const replacementTag = `<script type="importmap">\n\t\t\t${formattedJson}\n\t\t</script>`;
  return htmlContent.replace(importMapRegex, replacementTag);
}

