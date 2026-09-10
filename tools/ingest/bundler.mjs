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

      try {
        const resolved = resolveModuleSpecifier(source, referrerUrl, importMap, {
          mapBaseUrl,
          packageRootUrl: options.packageRootUrl
        });
        return resolved;
      } catch (err) {
        return null;
      }
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
 * Executes an actual Rollup build for an HTML or ESM entry point,
 * producing bundled output code.
 *
 * @param {string} entryPath
 * @param {Object} [options]
 * @returns {Promise<{ code: string, modules: string[] }>}
 */
export async function bundleWithRollup(entryPath, options = {}) {
  const resolvedEntryAbs = path.resolve(entryPath);
  const entryUrl = pathToFileURL(resolvedEntryAbs).href;
  const isHtml = entryPath.endsWith('.html') || entryPath.endsWith('.htm');

  let importMap = { imports: {}, scopes: {} };
  const inlineModules = new Map();
  let inputEntry = entryUrl;

  if (isHtml) {
    const htmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const parsed = parseHtmlEntries(htmlContent, entryUrl);
    importMap = parsed.importMap;

    if (parsed.moduleScripts.length === 0) {
      throw new Error(`No module scripts found in ${entryPath}`);
    }

    const firstScript = parsed.moduleScripts[0];
    inputEntry = firstScript.id;

    for (const s of parsed.moduleScripts) {
      if (s.inlineContent !== null) {
        inlineModules.set(s.id, s.inlineContent);
      }
    }
  }

  const bundle = await rollup({
    input: inputEntry,
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
      // Pass other warnings through
    }
  });

  const { output } = await bundle.generate({
    format: 'es'
  });

  const primaryChunk = output.find(chunk => chunk.type === 'chunk');

  return {
    code: primaryChunk ? primaryChunk.code : '',
    modules: primaryChunk ? Object.keys(primaryChunk.modules) : []
  };
}
