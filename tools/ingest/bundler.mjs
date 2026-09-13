/**
 * Rollup integration for FrankenThreeD module ingestion (f3d-04).
 * Provides a Rollup plugin backed by our W3C import-map resolver,
 * enabling real Rollup bundle creation and round-trip re-emission.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { rollup } from 'rollup';

import { analyzeModuleAst, classifyDynamicImportArgument } from './ast_analyzer.mjs';
import * as walk from 'acorn-walk';
import { parseHtmlEntries } from './html_parser.mjs';
import { resolveModuleSpecifier, urlToFilePath } from './resolver.mjs';

/**
 * Resolves the directory of a module on disk given its Rollup module ID.
 * Handles file:// URLs, absolute OS paths, inline HTML module IDs, and query/fragment suffixes.
 * @param {string} id
 * @param {string} [fallbackBaseUrl]
 * @returns {string}
 */
function getModuleDir(id, fallbackBaseUrl) {
  let cleanId = id;
  const hashIdx = cleanId.indexOf('#');
  if (hashIdx !== -1) cleanId = cleanId.slice(0, hashIdx);
  const qIdx = cleanId.indexOf('?');
  if (qIdx !== -1) cleanId = cleanId.slice(0, qIdx);

  if (cleanId.startsWith('file://')) {
    return path.dirname(urlToFilePath(cleanId));
  }
  if (path.isAbsolute(cleanId)) {
    return path.dirname(cleanId);
  }
  if (fallbackBaseUrl) {
    const cleanBase = fallbackBaseUrl.split(/[?#]/)[0];
    if (cleanBase.startsWith('file://')) return path.dirname(urlToFilePath(cleanBase));
    if (path.isAbsolute(cleanBase)) return path.dirname(cleanBase);
  }
  return process.cwd();
}

/**
 * Checks whether a specifier is a relative local file path.
 * @param {string} url
 * @returns {boolean}
 */
function isRelativeUrl(url) {
  if (!url || typeof url !== 'string') return false;
  const trimmed = url.trim();
  if (!trimmed) return false;
  if (trimmed.startsWith('#') || trimmed.startsWith('//') || trimmed.startsWith('/')) {
    return false;
  }
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
    return false;
  }
  return true;
}

/**
 * Checks whether a URL is an external runtime route (http:, https:, or data:).
 * @param {string} url
 * @returns {boolean}
 */
function isExternalUrl(url) {
  if (!url || typeof url !== 'string') return false;
  try {
    const protocol = new URL(url, 'file:///').protocol;
    return protocol === 'http:' || protocol === 'https:' || protocol === 'data:';
  } catch {
    return false;
  }
}

/**
 * Creates a Rollup plugin utilizing FrankenThreeD's W3C import-map resolver,
 * static module asset emission (new URL(..., import.meta.url)), and executable
 * finite dynamic import chunk emission (import(flag ? './a.js' : './b.js')).
 * @param {Object} options
 * @param {{ imports?: Record<string, string | null>, scopes?: Record<string, Record<string, string | null>> }} [options.importMap]
 * @param {string} options.mapBaseUrl
 * @param {Map<string, string>} [options.inlineModules]
 * @param {Map<string, string>} [options.retainedModuleUrls] - Exact source URLs mapped to copied output URLs
 * @param {string} [options.packageRootUrl]
 * @returns {import('rollup').Plugin}
 */
export function f3dRollupPlugin(options = {}) {
  const importMap = options.importMap || { imports: {}, scopes: {} };
  const mapBaseUrl = options.mapBaseUrl;
  const inlineModules = options.inlineModules || new Map();
  const retainedModuleUrls = options.retainedModuleUrls || new Map();
  const emittedAssetsByPath = new Map();
  const emittedChunksByUrl = new Map();
  const dynamicChunkRefIds = new Set();

  return {
    name: 'f3d-ingest-resolver',

    resolveFileUrl({ referenceId, relativePath }) {
      if (dynamicChunkRefIds.has(referenceId)) {
        const rel = relativePath.startsWith('./') || relativePath.startsWith('../')
          ? relativePath
          : `./${relativePath}`;
        return JSON.stringify(rel);
      }
      return null;
    },

    resolveId(source, importer) {
      if (inlineModules.has(source)) {
        return source;
      }

      const referrerUrl = importer
        ? (importer.startsWith('file://') ? importer : pathToFileURL(path.resolve(importer)).href)
        : mapBaseUrl;

      const resolvedUrl = resolveModuleSpecifier(source, referrerUrl, importMap, {
        mapBaseUrl,
        packageRootUrl: options.packageRootUrl
      });
      if (importer && retainedModuleUrls.has(resolvedUrl)) {
        // Both retained imports and bundled imports must reach the same browser module.
        // These URLs keep their query/fragment identity and are relative to output chunks.
        return { id: retainedModuleUrls.get(resolvedUrl), external: true };
      }
      if (isExternalUrl(resolvedUrl)) {
        return { id: resolvedUrl, external: true };
      }
      return resolvedUrl;
    },

    load(id) {
      if (inlineModules.has(id)) {
        return inlineModules.get(id);
      }

      if (id.startsWith('file://')) {
        const filePath = urlToFilePath(id);
        const source = fs.readFileSync(filePath, 'utf-8');
        if (retainedModuleUrls.has(id)) {
          // Rollup entry points cannot be external. Forward an HTML entry to its
          // retained module without evaluating a second copy of its body.
          const hasDefault = this.parse(source).body.some(node =>
            node.type === 'ExportDefaultDeclaration' ||
            (node.type === 'ExportAllDeclaration' && (node.exported?.name ?? node.exported?.value) === 'default') ||
            (node.type === 'ExportNamedDeclaration' && node.specifiers.some(spec =>
              (spec.exported.name ?? spec.exported.value) === 'default')));
          return `export * from ${JSON.stringify(id)};\n` +
            (hasDefault ? `export { default } from ${JSON.stringify(id)};\n` : '');
        }
        return source;
      }

      return null;
    },

    transform(code, id) {
      if (!code || typeof code !== 'string') return null;

      const moduleDir = getModuleDir(id, mapBaseUrl);
      const referrerUrl = id.startsWith('file://')
        ? id
        : (path.isAbsolute(id) ? pathToFileURL(id).href : mapBaseUrl);

      const edits = [];

      // 1. Asset reference replacements: new URL(..., import.meta.url)
      const analysis = analyzeModuleAst(code, id);
      const assetRefs = analysis && analysis.assetReferences ? analysis.assetReferences : [];

      for (const ref of assetRefs) {
        const rawSpecifier = ref.specifier;
        if (!rawSpecifier || typeof rawSpecifier !== 'string') continue;

        const qIdx = rawSpecifier.indexOf('?');
        const hIdx = rawSpecifier.indexOf('#');
        let splitIdx = -1;
        if (qIdx !== -1 && hIdx !== -1) splitIdx = Math.min(qIdx, hIdx);
        else if (qIdx !== -1) splitIdx = qIdx;
        else if (hIdx !== -1) splitIdx = hIdx;

        const cleanPath = splitIdx !== -1 ? rawSpecifier.slice(0, splitIdx) : rawSpecifier;
        const suffix = splitIdx !== -1 ? rawSpecifier.slice(splitIdx) : '';

        if (!isRelativeUrl(cleanPath)) continue;

        let assetAbsPath;
        try {
          const dirSlash = moduleDir.endsWith(path.sep) ? moduleDir : moduleDir + path.sep;
          const resolvedUrl = new URL(cleanPath, pathToFileURL(dirSlash));
          assetAbsPath = fileURLToPath(resolvedUrl);
        } catch {
          assetAbsPath = path.resolve(moduleDir, cleanPath);
        }

        if (!fs.existsSync(assetAbsPath) || !fs.statSync(assetAbsPath).isFile()) {
          throw new Error(
            `Unresolved module asset: "${rawSpecifier}" not found at "${assetAbsPath}" referenced from "${id}"`
          );
        }

        let refId = emittedAssetsByPath.get(assetAbsPath);
        if (!refId) {
          const assetSource = fs.readFileSync(assetAbsPath);
          refId = this.emitFile({
            type: 'asset',
            name: path.basename(assetAbsPath),
            source: assetSource
          });
          emittedAssetsByPath.set(assetAbsPath, refId);
        }

        const span = ref.source_span || ref.sourceSpan;
        const start = span.start.offset;
        const end = span.end.offset;

        const replacement = suffix
          ? `new URL(import.meta.ROLLUP_FILE_URL_${refId} + ${JSON.stringify(suffix)})`
          : `new URL(import.meta.ROLLUP_FILE_URL_${refId})`;

        edits.push({
          type: 'replace',
          pos: start,
          end: end,
          priority: 2,
          text: replacement
        });
      }

      // 2. Finite dynamic import replacements: import(flag ? './a.js' : './b.js')
      // Emits candidate branches as separate chunks (with preserveSignature: 'strict')
      // and wraps the dynamic argument with a safe own-key runtime specifier mapping.
      let ast;
      try {
        ast = this.parse(code);
      } catch {
        ast = null;
      }

      if (ast) {
        walk.simple(ast, {
          ImportExpression: (node) => {
            if (!node.source) return;
            const classified = classifyDynamicImportArgument(node.source);
            if (!classified || classified.classification !== 'finite_set' || !classified.candidates || classified.candidates.length === 0) {
              return;
            }

            const uniqueBranches = Array.from(new Set(classified.candidates));
            const specifierMap = new Map();

            for (const cand of uniqueBranches) {
              const resolvedUrl = resolveModuleSpecifier(cand, referrerUrl, importMap, {
                mapBaseUrl,
                packageRootUrl: options.packageRootUrl
              });

              if (retainedModuleUrls.has(resolvedUrl)) {
                specifierMap.set(cand, JSON.stringify(retainedModuleUrls.get(resolvedUrl)));
              } else if (isExternalUrl(resolvedUrl)) {
                specifierMap.set(cand, JSON.stringify(resolvedUrl));
              } else {
                let chunkRefId = emittedChunksByUrl.get(resolvedUrl);
                if (!chunkRefId) {
                  chunkRefId = this.emitFile({
                    type: 'chunk',
                    id: resolvedUrl,
                    preserveSignature: 'strict'
                  });
                  emittedChunksByUrl.set(resolvedUrl, chunkRefId);
                  dynamicChunkRefIds.add(chunkRefId);
                }
                specifierMap.set(cand, `import.meta.ROLLUP_FILE_URL_${chunkRefId}`);
              }
            }

            if (specifierMap.size > 0) {
              const chain = Array.from(specifierMap.entries())
                .map(([cand, targetExpr]) => `s === ${JSON.stringify(cand)} ? ${targetExpr} : `)
                .join('');

              const prefix = `(s => ${chain}s)(`;
              const suffix = `)`;

              // Composable insertions wrapping the argument preserve child edits inside the argument
              edits.push({
                type: 'insert',
                pos: node.source.start,
                priority: 1,
                text: prefix
              });
              edits.push({
                type: 'insert',
                pos: node.source.end,
                priority: 3,
                text: suffix
              });
            }
          }
        });
      }

      if (edits.length === 0) return null;

      // Sort descending by position; for coincident positions: suffix (end) before replace before prefix (start)
      edits.sort((a, b) => {
        if (b.pos !== a.pos) return b.pos - a.pos;
        return (b.priority || 0) - (a.priority || 0);
      });

      let transformedCode = code;
      for (const e of edits) {
        if (e.type === 'insert') {
          transformedCode = transformedCode.slice(0, e.pos) + e.text + transformedCode.slice(e.pos);
        } else if (e.type === 'replace') {
          transformedCode = transformedCode.slice(0, e.pos) + e.text + transformedCode.slice(e.end);
        }
      }

      return {
        code: transformedCode,
        map: null
      };
    }
  };
}

/**
 * @typedef {Object} EmittedChunk
 * @property {string} fileName - Emitted relative file name
 * @property {string} code - Generated JavaScript source code
 * @property {string[]} modules - Module IDs included in this chunk
 * @property {boolean} isEntry - True if this chunk corresponds to a root entry point
 * @property {boolean} isDynamicEntry - True if generated from a dynamic import()
 * @property {string[]} imports - Static dependency chunk file names
 * @property {string[]} dynamicImports - Dynamic dependency chunk file names
 * @property {string | null} facadeModuleId - Module ID for entry chunks
 */

/**
 * Executes an actual Rollup build for an HTML or ESM entry point,
 * producing bundled output code.
 *
 * @param {string} entryPath
 * @param {Object} [options]
 * @returns {Promise<{
 *   code: string,
 *   modules: string[],
 *   isMultiChunk: boolean,
 *   entryFiles: string[],
 *   files: Record<string, string>,
 *   outputChunks: EmittedChunk[],
 *   chunks: EmittedChunk[]
 * }>}
 */
export async function bundleWithRollup(entryPath, options = {}) {
  const resolvedEntryAbs = path.resolve(entryPath);
  const entryUrl = pathToFileURL(resolvedEntryAbs).href;
  const isHtml = entryPath.endsWith('.html') || entryPath.endsWith('.htm');

  let importMap = { imports: {}, scopes: {} };
  let mapBaseUrl = entryUrl;
  const inlineModules = new Map();
  let input;
  const orderedEntryIds = [];

  if (isHtml) {
    const htmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const parsed = parseHtmlEntries(htmlContent, entryUrl);
    importMap = parsed.importMap;
    mapBaseUrl = parsed.baseUrl;

    if (parsed.moduleScripts.length === 0) {
      throw new Error(`No module scripts found in ${entryPath}`);
    }

    // Empty src is a browser script error, never an inline module or an HTML import.
    const localModuleScripts = parsed.moduleScripts.filter(s => s.src !== '' && !isExternalUrl(s.id));

    // Browser-owned external/empty-src scripts do not emit Rollup chunks.
    if (localModuleScripts.length === 0) {
      return {
        code: '',
        modules: [],
        isMultiChunk: false,
        entryFiles: [],
        files: {},
        outputChunks: [],
        chunks: [],
        assets: []
      };
    }

    for (const s of localModuleScripts) {
      if (s.inlineContent !== null) {
        inlineModules.set(s.id, s.inlineContent);
      }
      orderedEntryIds.push(s.id);
    }

    // Deduplicate Rollup input IDs by exact canonical resolved URL id only.
    // Preserves query strings (?query) and fragments (#fragment) as distinct ES module identities.
    const seenInputIds = new Set();
    const uniqueInputScripts = [];

    for (const s of localModuleScripts) {
      if (!seenInputIds.has(s.id)) {
        seenInputIds.add(s.id);
        uniqueInputScripts.push(s);
      }
    }

    if (uniqueInputScripts.length === 1 && uniqueInputScripts[0].src) {
      input = uniqueInputScripts[0].id;
    } else {
      input = {};
      uniqueInputScripts.forEach((s, idx) => {
        let name;
        if (s.src) {
          const cleanSrc = s.src.split('?')[0].split('#')[0];
          name = path.basename(cleanSrc).replace(/\.[^/.]+$/, '') || 'script';
        } else {
          name = `inline_${idx}`;
        }
        let entryKey = name;
        let counter = 1;
        while (Object.prototype.hasOwnProperty.call(input, entryKey)) {
          entryKey = `${name}_${counter++}`;
        }
        input[entryKey] = s.id;
      });
    }
  } else {
    input = entryUrl;
    orderedEntryIds.push(entryUrl);
  }

  let bundle;
  try {
    bundle = await rollup({
      input,
      plugins: [
        f3dRollupPlugin({
          importMap,
          mapBaseUrl,
          inlineModules,
          packageRootUrl: options.packageRootUrl,
          retainedModuleUrls: options.retainedModuleUrls
        })
      ],
      onwarn(warning, warn) {
        // Suppress known non-fatal warnings (e.g. eval in 3rd party libs or circular deps)
        if (warning.code === 'CIRCULAR_DEPENDENCY' || warning.code === 'THIS_IS_UNDEFINED') {
          return;
        }
        // Forward all other warnings honestly
        warn(warning);
      }
    });

    const { output } = await bundle.generate({
      format: 'es',
      entryFileNames: chunk => options.retainedModuleUrls?.has(chunk.facadeModuleId)
        ? 'f3d-entry-[name]-[hash].js'
        : '[name].js'
    });

    const chunks = output.filter(chunk => chunk.type === 'chunk');
    const assets = output.filter(item => item.type === 'asset');

    // Match entry chunks in exact HTML document order.
    // If an HTML document contains repeated module script references with identical URL
    // (e.g. two <script type="module" src="./same.mjs"> tags), entryFiles preserves the exact
    // 1:1 document-order mapping with duplicate file references so HTML re-emission
    // keeps both script tags pointing to the same emitted chunk, while underlying
    // chunks and files remain unique. Distinct URLs (including query variants) map to
    // their own distinct emitted chunks.
    const entryFiles = [];
    const entryChunks = [];
    const seenEntryChunks = new Set();

    for (const entryId of orderedEntryIds) {
      const chunk = chunks.find(c => c.facadeModuleId === entryId);

      if (!chunk) {
        throw new Error(`Failed to find emitted chunk for entry module: ${entryId}`);
      }

      entryFiles.push(chunk.fileName);
      if (!seenEntryChunks.has(chunk)) {
        entryChunks.push(chunk);
        seenEntryChunks.add(chunk);
      }
    }

    const primaryChunk = entryChunks[0] || chunks[0] || null;

    // Collect all modules across all emitted chunks
    const allModules = [];
    const seenModules = new Set();
    for (const chunk of chunks) {
      for (const mod of Object.keys(chunk.modules || {})) {
        if (!seenModules.has(mod)) {
          seenModules.add(mod);
          allModules.push(mod);
        }
      }
    }

    const outputChunks = chunks.map(c => ({
      fileName: c.fileName,
      code: c.code,
      modules: Object.keys(c.modules || {}),
      isEntry: Boolean(c.isEntry),
      isDynamicEntry: Boolean(c.isDynamicEntry),
      imports: c.imports || [],
      dynamicImports: c.dynamicImports || [],
      facadeModuleId: c.facadeModuleId || null
    }));

    const outputAssets = assets.map(a => ({
      fileName: a.fileName,
      name: a.name,
      source: a.source
    }));

    const files = Object.fromEntries([
      ...chunks.map(c => [c.fileName, c.code]),
      ...assets.map(a => [a.fileName, typeof a.source === 'string' ? a.source : Buffer.from(a.source)])
    ]);

    return {
      // Backward compatibility for single-chunk callers
      code: primaryChunk ? primaryChunk.code : '',
      modules: allModules,

      // Multi-chunk & dynamic chunk support
      isMultiChunk: chunks.length > 1,
      entryFiles,
      files,
      outputChunks,
      chunks: outputChunks,
      assets: outputAssets
    };
  } finally {
    if (bundle) {
      await bundle.close();
    }
  }
}
