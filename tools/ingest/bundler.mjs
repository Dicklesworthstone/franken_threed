/**
 * Rollup integration for FrankenThreeD module ingestion (f3d-04).
 * Provides a Rollup plugin backed by our W3C import-map resolver,
 * enabling real Rollup bundle creation and round-trip re-emission.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { rollup } from 'rollup';

import { analyzeModuleAst } from './ast_analyzer.mjs';
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
 * Creates a Rollup plugin utilizing FrankenThreeD's W3C import-map resolver
 * and static module asset emission (new URL(..., import.meta.url)).
 * @param {Object} options
 * @param {{ imports?: Record<string, string | null>, scopes?: Record<string, Record<string, string | null>> }} [options.importMap]
 * @param {string} options.mapBaseUrl
 * @param {Map<string, string>} [options.inlineModules]
 * @param {string} [options.packageRootUrl]
 * @returns {import('rollup').Plugin}
 */
export function f3dRollupPlugin(options = {}) {
  const importMap = options.importMap || { imports: {}, scopes: {} };
  const mapBaseUrl = options.mapBaseUrl;
  const inlineModules = options.inlineModules || new Map();
  const emittedAssetsByPath = new Map();

  return {
    name: 'f3d-ingest-resolver',

    resolveId(source, importer) {
      if (inlineModules.has(source)) {
        return source;
      }

      const referrerUrl = importer
        ? (importer.startsWith('file://') ? importer : pathToFileURL(path.resolve(importer)).href)
        : mapBaseUrl;

      return resolveModuleSpecifier(source, referrerUrl, importMap, {
        mapBaseUrl,
        packageRootUrl: options.packageRootUrl
      });
    },

    load(id) {
      if (inlineModules.has(id)) {
        return inlineModules.get(id);
      }

      if (id.startsWith('file://')) {
        const filePath = urlToFilePath(id);
        return fs.readFileSync(filePath, 'utf-8');
      }

      return null;
    },

    transform(code, id) {
      if (!code || typeof code !== 'string') return null;

      const analysis = analyzeModuleAst(code, id);
      const assetRefs = analysis && analysis.assetReferences ? analysis.assetReferences : [];
      if (assetRefs.length === 0) return null;

      const moduleDir = getModuleDir(id, mapBaseUrl);

      // Sort in descending order of source_span start offset to perform non-shifting slice replacements
      const sortedRefs = [...assetRefs].sort((a, b) => {
        const spanA = a.source_span || a.sourceSpan;
        const spanB = b.source_span || b.sourceSpan;
        return spanB.start.offset - spanA.start.offset;
      });

      let transformedCode = code;
      let hasChanges = false;

      for (const ref of sortedRefs) {
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

        transformedCode = transformedCode.slice(0, start) + replacement + transformedCode.slice(end);
        hasChanges = true;
      }

      if (!hasChanges) return null;

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
  const inlineModules = new Map();
  let input;
  const orderedEntryIds = [];

  if (isHtml) {
    const htmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const parsed = parseHtmlEntries(htmlContent, entryUrl);
    importMap = parsed.importMap;

    if (parsed.moduleScripts.length === 0) {
      throw new Error(`No module scripts found in ${entryPath}`);
    }

    for (const s of parsed.moduleScripts) {
      if (s.inlineContent !== null) {
        inlineModules.set(s.id, s.inlineContent);
      }
      orderedEntryIds.push(s.id);
    }

    // Deduplicate Rollup input IDs by exact canonical resolved URL id only.
    // Preserves query strings (?query) and fragments (#fragment) as distinct ES module identities.
    const seenInputIds = new Set();
    const uniqueInputScripts = [];

    for (const s of parsed.moduleScripts) {
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
          mapBaseUrl: entryUrl,
          inlineModules,
          packageRootUrl: options.packageRootUrl
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
      format: 'es'
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
