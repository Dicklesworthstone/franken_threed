/**
 * Rollup integration for FrankenThreeD module ingestion (f3d-04).
 * Provides a Rollup plugin backed by our W3C import-map resolver,
 * enabling real Rollup bundle creation and round-trip re-emission.
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { rollup } from 'rollup';

import { parseHtmlEntries } from './html_parser.mjs';
import { resolveModuleSpecifier, urlToFilePath } from './resolver.mjs';

/**
 * Creates a Rollup plugin utilizing FrankenThreeD's W3C import-map resolver.
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

    const files = Object.fromEntries(chunks.map(c => [c.fileName, c.code]));

    return {
      // Backward compatibility for single-chunk callers
      code: primaryChunk ? primaryChunk.code : '',
      modules: allModules,

      // Multi-chunk & dynamic chunk support
      isMultiChunk: chunks.length > 1,
      entryFiles,
      files,
      outputChunks,
      chunks: outputChunks
    };
  } finally {
    if (bundle) {
      await bundle.close();
    }
  }
}
