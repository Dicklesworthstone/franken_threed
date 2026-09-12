/**
 * Application build emitter for FrankenThreeD (f3d-04).
 *
 * Takes an HTML or ESM entry point, bundles module dependencies using the existing
 * W3C import-map backed Rollup pipeline, and emits a runnable multi-entry application
 * to a fresh destination directory.
 *
 * For HTML entries:
 * - Preserves all original non-module content (DOCTYPE, DOM structure, styles,
 *   non-module scripts, import maps, comments, and meta tags).
 * - Replaces active module script entries in exact source document order with
 *   relative module script references to emitted entry chunks.
 * - Uses quote-aware tag matching so attributes containing ">" (e.g. data-selector="div > span")
 *   are never truncated prematurely.
 * - Uses attribute-aware anchored token matching so attributes like data-src or
 *   data-integrity are never confused with src or integrity, preserving unrelated
 *   attributes with complex quotes (e.g. data-label='a"b') verbatim.
 * - Honestly recomputes Subresource Integrity (SRI) attributes for emitted chunks.
 * - Explicitly rejects <base href="..."> input before output to preserve document.baseURI
 *   semantics and prevent silent asset/link corruption, leaving base-href support open.
 * - Rewrites <link rel="modulepreload"> hrefs to point to emitted chunks instead of raw
 *   source files, updating SRI integrity and skipping static asset copies for bundled modules.
 * - Enforces bounded asset closure by copying relative assets (stylesheets, images,
 *   non-module scripts, and referenced CSS url/@import children) or explicitly rejecting
 *   unresolved relative resources before output.
 * - Enforces entry count equality in both directions between HTML module scripts and emitted chunks.
 * - Strictly rejects collisions, race overwrites, and symlinks using exclusive 'wx' write flags.
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { bundleWithRollup } from './bundler.mjs';
import { parseTagAttributes, parseSrcsetUrls, stripHtmlComments, stripScriptAndStyleBodies } from './html_parser.mjs';

/**
 * Recomputes Subresource Integrity (SRI) string for modified or bundled content,
 * matching the algorithm(s) specified in the original integrity attribute.
 *
 * @param {string} originalIntegrity - e.g. "sha384-..." or "sha256-... sha384-..."
 * @param {string | Buffer} content - The actual emitted chunk content
 * @returns {string}
 */
export function computeIntegrityForContent(originalIntegrity, content) {
  if (!originalIntegrity || typeof originalIntegrity !== 'string') {
    return '';
  }
  const buffer = typeof content === 'string' ? Buffer.from(content, 'utf-8') : content;
  const tokens = originalIntegrity.trim().split(/\s+/);
  const updatedTokens = [];

  for (const token of tokens) {
    const dashIdx = token.indexOf('-');
    if (dashIdx === -1) {
      throw new Error(`Invalid integrity attribute token format: "${token}"`);
    }
    const algo = token.slice(0, dashIdx).toLowerCase();
    if (!['sha256', 'sha384', 'sha512'].includes(algo)) {
      throw new Error(`Unsupported integrity hash algorithm: "${algo}"`);
    }
    const hash = crypto.createHash(algo).update(buffer).digest('base64');
    updatedTokens.push(`${algo}-${hash}`);
  }

  return updatedTokens.join(' ');
}

/**
 * Determines whether a URL string is a relative local file path.
 * Returns false for scheme-qualified (http:, https:, data:, blob:, mailto:, javascript:),
 * protocol-relative (//), root-relative (/), or hash-only (#) references.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isRelativeUrl(url) {
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
 * Looks up whether a modulepreload raw href corresponds to an emitted chunk in preloadChunkMap.
 * Uses exact canonical URL resolution (preserving ?query and #fragment as distinct ES module identities)
 * against the referrer directory URL, without stripping query or fragment.
 *
 * @param {string} rawHref
 * @param {string} referrerDir
 * @param {Map<string, string>} [preloadChunkMap]
 * @returns {string | null} Emitted chunk file name if matched, else null
 */
export function findChunkForPreload(rawHref, referrerDir, preloadChunkMap) {
  if (!rawHref || typeof rawHref !== 'string' || !preloadChunkMap) return null;
  const trimmed = rawHref.trim();
  if (!trimmed) return null;

  // 1. Canonical URL resolution against referrer directory URL (preserving ?query and #fragment)
  if (referrerDir) {
    try {
      const referrerDirSlash = referrerDir.endsWith(path.sep) ? referrerDir : referrerDir + path.sep;
      const baseUrl = pathToFileURL(referrerDirSlash);
      const canonicalUrl = new URL(trimmed, baseUrl).href;
      if (preloadChunkMap.has(canonicalUrl)) {
        return preloadChunkMap.get(canonicalUrl);
      }
    } catch {
      // Ignore invalid URL resolution
    }
  }

  // 2. Direct match fallback (for pre-canonicalized URLs or virtual IDs)
  if (preloadChunkMap.has(trimmed)) {
    return preloadChunkMap.get(trimmed);
  }

  return null;
}


/**
 * Scans HTML content for relative asset references in attributes (e.g. stylesheets,
 * icons, images, non-module scripts, media, unbundled modulepreloads).
 *
 * Uses quote-aware tag scanners so attributes containing ">" are never truncated.
 *
 * @param {string} rawHtmlContent
 * @param {Map<string, string>} [preloadChunkMap] - Mapped bundle chunks to exclude from static copy
 * @param {string} [referrerDir=''] - Base directory for resolving relative URLs
 * @returns {string[]} Deduplicated list of relative resource URLs
 */
export function extractRelativeAssetUrls(rawHtmlContent, preloadChunkMap = null, referrerDir = '') {
  const domHtml = stripScriptAndStyleBodies(rawHtmlContent);
  const assets = new Set();

  // 1. <link ...> quote-aware
  const linkRegex = /<link\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi;
  let match;
  while ((match = linkRegex.exec(domHtml)) !== null) {
    const attrs = parseTagAttributes(match[1]);
    if (attrs.href && isRelativeUrl(attrs.href)) {
      // If this is a modulepreload matching an emitted chunk, it is handled by Rollup bundle emission
      if (
        attrs.rel &&
        attrs.rel.toLowerCase() === 'modulepreload' &&
        preloadChunkMap &&
        findChunkForPreload(attrs.href, referrerDir, preloadChunkMap)
      ) {
        continue;
      }
      assets.add(attrs.href);
    }
  }

  // 2. <img ...> quote-aware
  const imgRegex = /<img\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi;
  while ((match = imgRegex.exec(domHtml)) !== null) {
    const attrs = parseTagAttributes(match[1]);
    if (attrs.src && isRelativeUrl(attrs.src)) {
      assets.add(attrs.src);
    }
    if (attrs.srcset) {
      const srcsetUrls = parseSrcsetUrls(attrs.srcset);
      for (const u of srcsetUrls) {
        if (isRelativeUrl(u)) {
          assets.add(u);
        }
      }
    }
  }

  // 3. <script ...> where type != 'module' quote-aware
  const scriptRegex = /<script\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi;
  while ((match = scriptRegex.exec(domHtml)) !== null) {
    const attrs = parseTagAttributes(match[1]);
    const type = (attrs.type || 'text/javascript').toLowerCase();
    if (type !== 'module' && attrs.src && isRelativeUrl(attrs.src)) {
      assets.add(attrs.src);
    }
  }

  // 4. <video>, <audio>, <source>, <track> quote-aware
  const mediaRegex = /<(?:video|audio|source|track)\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi;
  while ((match = mediaRegex.exec(domHtml)) !== null) {
    const attrs = parseTagAttributes(match[1]);
    if (attrs.src && isRelativeUrl(attrs.src)) {
      assets.add(attrs.src);
    }
    if (attrs.poster && isRelativeUrl(attrs.poster)) {
      assets.add(attrs.poster);
    }
    if (attrs.srcset) {
      const srcsetUrls = parseSrcsetUrls(attrs.srcset);
      for (const u of srcsetUrls) {
        if (isRelativeUrl(u)) {
          assets.add(u);
        }
      }
    }
  }

  return Array.from(assets);
}

/**
 * Strips comments from CSS content.
 * @param {string} css
 * @returns {string}
 */
export function stripCssComments(css) {
  return css.replace(/\/\*[\s\S]*?\*\//g, '');
}

/**
 * Contextual scanner regex for CSS resources.
 * Disambiguates comments, @import rules, url(...) functional notations, and string literals,
 * so that url(...) inside quoted strings (e.g. content: "url(phantom.png)") or inside comments
 * is never extracted as a resource dependency.
 */
export const CSS_RESOURCE_REGEX =
  /(\/\*[\s\S]*?\*\/)|(@import\s+(?:url\(\s*)?(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s();]+))\s*\)?)|(\burl\(\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^'")\s]+))\s*\))|("(?:[^"\\]|\\.)*")|('(?:[^'\\]|\\.)*')/gi;

/**
 * Extracts relative resource URLs (url(...) and @import) from CSS content.
 * Skips url(...) occurrences inside single- or double-quoted CSS strings and comments.
 *
 * @param {string} cssContent
 * @returns {string[]} Deduplicated list of relative URLs
 */
export function extractRelativeCssUrls(cssContent) {
  if (!cssContent || typeof cssContent !== 'string') return [];
  const urls = new Set();
  const regex = new RegExp(CSS_RESOURCE_REGEX.source, 'gi');
  let match;

  while ((match = regex.exec(cssContent)) !== null) {
    // match[1]: comment -> ignore
    // match[10]: double-quoted string -> ignore
    // match[11]: single-quoted string -> ignore
    if (match[1] || match[10] || match[11]) {
      continue;
    }

    // match[2]: @import statement
    if (match[2]) {
      const rawUrl = match[3] !== undefined ? match[3] : (match[4] !== undefined ? match[4] : match[5]);
      if (rawUrl && isRelativeUrl(rawUrl)) {
        urls.add(rawUrl);
      }
      continue;
    }

    // match[6]: url(...) function
    if (match[6]) {
      const rawUrl = match[7] !== undefined ? match[7] : (match[8] !== undefined ? match[8] : match[9]);
      if (rawUrl && isRelativeUrl(rawUrl)) {
        urls.add(rawUrl);
      }
    }
  }

  return Array.from(urls);
}

/**
 * Rewrites attributes of an active <script type="module"> tag.
 *
 * Uses an attribute-aware anchored token match consistent with html_parser.mjs:
 * Requires (?:^|\s+) anchor and [a-zA-Z0-9_:-]+ token name so attributes like data-src
 * or data-integrity are never confused with src or integrity.
 * Preserves exact original quotes, formatting, and values of all unrelated attributes.
 *
 * @param {string} attrString - Raw attribute text between `<script` and `>`
 * @param {string} chunkFileName - Emitted chunk file name to point to
 * @param {string | Buffer} [chunkCode=''] - Emitted chunk content for SRI hashing
 * @returns {string}
 */
export function rewriteScriptTagAttributes(attrString, chunkFileName, chunkCode = '') {
  const attrRegex = /(?:^|\s+)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;

  let match;
  let hasSrc = false;
  let lastIndex = 0;
  const pieces = [];

  while ((match = attrRegex.exec(attrString)) !== null) {
    const fullMatch = match[0];
    const name = match[1].toLowerCase();
    const val = match[2] !== undefined
      ? match[2]
      : (match[3] !== undefined ? match[3] : (match[4] !== undefined ? match[4] : ''));

    // Preserve any whitespace/characters between last token and current match
    if (match.index > lastIndex) {
      pieces.push(attrString.slice(lastIndex, match.index));
    }
    lastIndex = attrRegex.lastIndex;

    if (name === 'src') {
      hasSrc = true;
      pieces.push(` src="./${chunkFileName}"`);
    } else if (name === 'integrity') {
      const newIntegrity = computeIntegrityForContent(val, chunkCode);
      pieces.push(` integrity="${newIntegrity}"`);
    } else {
      // Unrelated attribute (data-src, data-integrity, id, async, etc.) preserved verbatim
      pieces.push(fullMatch);
    }
  }

  // Preserve any trailing characters
  if (lastIndex < attrString.length) {
    pieces.push(attrString.slice(lastIndex));
  }

  // Inject src if not previously present (e.g. inline module script)
  if (!hasSrc) {
    pieces.push(` src="./${chunkFileName}"`);
  }

  return pieces.join('');
}



/**
 * Rewrites attributes of a <link rel="modulepreload"> tag that matches an emitted chunk.
 * Points href to "./${chunkFileName}" and recomputes SRI integrity honestly while preserving
 * unrelated attributes (e.g. data-href, data-integrity, as, crossorigin).
 *
 * @param {string} attrString
 * @param {string} chunkFileName
 * @param {string | Buffer} [chunkCode='']
 * @returns {string}
 */
export function rewriteLinkTagAttributes(attrString, chunkFileName, chunkCode = '') {
  const attrRegex = /(?:^|\s+)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let match;
  let lastIndex = 0;
  const pieces = [];

  while ((match = attrRegex.exec(attrString)) !== null) {
    const fullMatch = match[0];
    const name = match[1].toLowerCase();
    const val = match[2] !== undefined
      ? match[2]
      : (match[3] !== undefined ? match[3] : (match[4] !== undefined ? match[4] : ''));

    if (match.index > lastIndex) {
      pieces.push(attrString.slice(lastIndex, match.index));
    }
    lastIndex = attrRegex.lastIndex;

    if (name === 'href') {
      pieces.push(` href="./${chunkFileName}"`);
    } else if (name === 'integrity') {
      if (chunkCode) {
        const newIntegrity = computeIntegrityForContent(val, chunkCode);
        pieces.push(` integrity="${newIntegrity}"`);
      } else {
        pieces.push(fullMatch);
      }
    } else {
      pieces.push(fullMatch);
    }
  }

  if (lastIndex < attrString.length) {
    pieces.push(attrString.slice(lastIndex));
  }

  return pieces.join('');
}

/**
 * Rewrites raw HTML content, replacing active <script type="module"> tags in document
 * order with references to emitted entry chunks while strictly preserving all comments,
 * DOM elements, styles, non-module scripts, attribute formatting, and import maps.
 *
 * - Uses quote-aware tag regex so attributes containing ">" (e.g. data-selector="div > span")
 *   are never truncated.
 * - Explicitly rejects <base href="..."> input before output to preserve document.baseURI semantics.
 * - Rewrites <link rel="modulepreload"> hrefs to emitted chunks and updates SRI integrity.
 * - Enforces entry count equality in both directions.
 *
 * @param {string} rawHtmlContent
 * @param {string[]} entryFiles - Emitted entry chunk file names in document order
 * @param {Record<string, string>} [chunkFilesMap={}] - Mapping of chunk file name to emitted code
 * @param {Object} [options={}]
 * @param {Map<string, string>} [options.preloadChunkMap] - Mapping of module URL/path to emitted chunk
 * @param {string} [options.entryDir=''] - Directory of the entry HTML file
 * @returns {string}
 */
export function rewriteHtmlForBuild(rawHtmlContent, entryFiles, chunkFilesMap = {}, options = {}) {
  let moduleScriptIndex = 0;
  const preloadChunkMap = options.preloadChunkMap || null;
  const entryDir = options.entryDir || '';

  // Quote-aware tag scanner:
  // Match HTML comments, <script>...</script>, <style>...</style>, <base ...>, or <link ...>
  const tagRegex = /(<!--[\s\S]*?-->)|(<script\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/script\s*>)|(<style\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/style\s*>)|(<base\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>)|(<link\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>)/gi;

  const rewritten = rawHtmlContent.replace(
    tagRegex,
    (match, isComment, isScript, scriptAttrs, scriptBody, isStyle, styleAttrs, styleBody, isBase, baseAttrs, isLink, linkAttrs) => {
      if (isComment || isStyle) {
        return match;
      }

      // Handle <script> tags
      if (isScript) {
        const attrs = parseTagAttributes(scriptAttrs);
        const scriptType = (attrs.type || 'text/javascript').toLowerCase();

        if (scriptType === 'module') {
          if (moduleScriptIndex >= entryFiles.length) {
            throw new Error(
              `Module script at index ${moduleScriptIndex} exceeds emitted entry chunk count (${entryFiles.length})`
            );
          }

          const chunkFileName = entryFiles[moduleScriptIndex++];
          const chunkCode = chunkFilesMap[chunkFileName] || '';

          const updatedAttrString = rewriteScriptTagAttributes(
            scriptAttrs,
            chunkFileName,
            chunkCode
          );

          return `<script${updatedAttrString}></script>`;
        }

        // Non-module scripts (including type="importmap") are preserved verbatim
        return match;
      }

      // Handle <base> tags: explicitly reject if href is specified, to preserve document.baseURI semantics
      if (isBase) {
        const parsedBase = parseTagAttributes(baseAttrs);
        if (parsedBase.href) {
          throw new Error(
            `Explicit rejection: <base href="${parsedBase.href}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`
          );
        }
        return match;
      }

      // Handle <link> tags: if modulepreload matches an emitted chunk, rewrite href and update SRI integrity
      if (isLink) {
        const parsedLink = parseTagAttributes(linkAttrs);
        if (
          parsedLink.rel &&
          parsedLink.rel.toLowerCase() === 'modulepreload' &&
          parsedLink.href &&
          preloadChunkMap
        ) {
          const chunkFileName = findChunkForPreload(parsedLink.href, entryDir, preloadChunkMap);
          if (chunkFileName) {
            const chunkCode = chunkFilesMap[chunkFileName] || '';
            const updatedLink = rewriteLinkTagAttributes(linkAttrs, chunkFileName, chunkCode);
            return `<link${updatedLink}>`;
          }
        }
        return match;
      }

      return match;
    }
  );

  // Enforce entry count equality in both directions
  if (moduleScriptIndex !== entryFiles.length) {
    throw new Error(
      `Module script count in HTML (${moduleScriptIndex}) does not match emitted entry chunk count (${entryFiles.length})`
    );
  }

  return rewritten;
}

/**
 * Checks if a path exists or is an existing symlink (valid or broken).
 * @param {string} targetPath
 * @returns {boolean}
 */
function pathExistsOrSymlink(targetPath) {
  try {
    fs.lstatSync(targetPath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Builds a runnable application package from an HTML or ESM entry point.
 *
 * @param {string} entryPath - Path to HTML or JavaScript entry file
 * @param {string} outDir - Destination directory for emitted application (must be fresh)
 * @param {Object} [options]
 * @param {string} [options.packageRootUrl] - Base URL or directory for package fallback
 * @returns {Promise<{
 *   entryPoint: string,
 *   outDir: string,
 *   isHtml: boolean,
 *   htmlFile: string | null,
 *   entryFiles: string[],
 *   emittedFiles: string[],
 *   isMultiChunk: boolean,
 *   chunks: import('./bundler.mjs').EmittedChunk[],
 *   packageType: string
 * }>}
 */
export async function buildApplication(entryPath, outDir, options = {}) {
  if (!entryPath || typeof entryPath !== 'string') {
    throw new Error('buildApplication requires a valid entryPath string');
  }
  if (!outDir || typeof outDir !== 'string') {
    throw new Error('buildApplication requires a valid outDir string');
  }

  const resolvedEntryAbs = path.resolve(entryPath);
  if (!fs.existsSync(resolvedEntryAbs)) {
    throw new Error(`Entry path does not exist: ${entryPath}`);
  }

  const resolvedOutDir = path.resolve(outDir);
  const isHtml = entryPath.endsWith('.html') || entryPath.endsWith('.htm');
  const entryDir = path.dirname(resolvedEntryAbs);

  if (isHtml) {
    const rawHtml = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const domHtml = stripScriptAndStyleBodies(rawHtml);
    const baseMatch = /<base\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/i.exec(domHtml);
    if (baseMatch) {
      const baseAttrs = parseTagAttributes(baseMatch[1]);
      if (baseAttrs.href) {
        throw new Error(
          `Explicit rejection: <base href="${baseAttrs.href}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`
        );
      }
    }
  }

  // Execute Rollup bundling backed by import-map resolver
  const bundleResult = await bundleWithRollup(resolvedEntryAbs, {
    packageRootUrl: options.packageRootUrl
  });

  const targetFiles = new Map(); // relativePath -> string | Buffer
  const htmlFileName = isHtml ? path.basename(resolvedEntryAbs) : null;

  // Collect all emitted code chunks from Rollup
  for (const [fileName, code] of Object.entries(bundleResult.files)) {
    targetFiles.set(fileName, code);
  }

  // If HTML entry point: process HTML, rewrite module script tags, and enforce bounded asset closure
  if (isHtml) {
    // Build map of canonical module URLs to emitted chunk file names for modulepreload rewrites
    const preloadChunkMap = new Map();
    for (const chunk of bundleResult.chunks) {
      if (chunk.facadeModuleId) {
        preloadChunkMap.set(chunk.facadeModuleId, chunk.fileName);
      }
      if (chunk.modules) {
        const modIds = Array.isArray(chunk.modules) ? chunk.modules : Object.keys(chunk.modules);
        for (const modId of modIds) {
          preloadChunkMap.set(modId, chunk.fileName);
        }
      }
    }

    const rawHtmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const rewrittenHtml = rewriteHtmlForBuild(
      rawHtmlContent,
      bundleResult.entryFiles,
      bundleResult.files,
      { preloadChunkMap, entryDir }
    );
    targetFiles.set(htmlFileName, rewrittenHtml);

    // Extract relative assets referenced by the HTML (stylesheets, images, media, non-module scripts)
    // Excludes modulepreloads that map to bundled chunks
    const relativeAssetUrls = extractRelativeAssetUrls(rawHtmlContent, preloadChunkMap, entryDir);

    // Bounded asset processing queue: handles direct HTML assets and transitive CSS url()/@import children
    const assetQueue = [];
    for (const relUrl of relativeAssetUrls) {
      assetQueue.push({
        relUrl,
        referrerDir: entryDir,
        referrerPath: resolvedEntryAbs
      });
    }

    const visitedCssPaths = new Set();

    while (assetQueue.length > 0) {
      const { relUrl, referrerDir, referrerPath } = assetQueue.shift();
      if (!relUrl || typeof relUrl !== 'string') continue;
      const trimmedRelUrl = relUrl.trim();
      if (!trimmedRelUrl) continue;

      // Browser URI semantics: resolve against referrer directory URL using new URL,
      // then convert file: URL object directly to decoded filesystem path via fileURLToPath.
      // Preserves %20 and other percent-encodings as actual filename bytes on disk.
      const referrerDirSlash = referrerDir.endsWith(path.sep) ? referrerDir : referrerDir + path.sep;
      const referrerBaseUrl = pathToFileURL(referrerDirSlash);
      const resolvedUrl = new URL(trimmedRelUrl, referrerBaseUrl);
      const srcAssetAbs = fileURLToPath(resolvedUrl);

      // Verify that relative asset path does not escape entry directory
      const relFromEntryDir = path.relative(entryDir, srcAssetAbs);
      if (relFromEntryDir.startsWith('..') || path.isAbsolute(relFromEntryDir)) {
        const context = referrerPath !== resolvedEntryAbs ? ` in CSS "${referrerPath}"` : '';
        throw new Error(`Relative resource${context} escapes application root directory: "${relUrl}"`);
      }

      // Check whether this path matches an emitted chunk (e.g. modulepreload referencing an emitted chunk)
      if (targetFiles.has(relFromEntryDir)) {
        continue;
      }

      // Explicitly reject unresolved relative resources before output
      if (!fs.existsSync(srcAssetAbs) || !fs.statSync(srcAssetAbs).isFile()) {
        const context = referrerPath !== resolvedEntryAbs ? ` in CSS referenced from "${referrerPath}"` : '';
        throw new Error(
          `Unresolved relative resource${context}: "${relUrl}" not found at "${srcAssetAbs}"`
        );
      }

      // Copy asset into bounded closure targetFiles
      const assetData = fs.readFileSync(srcAssetAbs);
      targetFiles.set(relFromEntryDir, assetData);

      // If asset is a CSS file, scan for transitive child url() and @import resources
      if (srcAssetAbs.endsWith('.css') && !visitedCssPaths.has(srcAssetAbs)) {
        visitedCssPaths.add(srcAssetAbs);
        const cssContent = assetData.toString('utf-8');
        const childUrls = extractRelativeCssUrls(cssContent);
        const cssDir = path.dirname(srcAssetAbs);
        for (const childUrl of childUrls) {
          assetQueue.push({
            relUrl: childUrl,
            referrerDir: cssDir,
            referrerPath: srcAssetAbs
          });
        }
      }
    }
  }

  // Safe collision check: fail safely without deleting or overwriting (including symlinks)
  for (const relPath of targetFiles.keys()) {
    const destPath = path.join(resolvedOutDir, relPath);
    if (pathExistsOrSymlink(destPath)) {
      throw new Error(
        `Refusing to overwrite existing destination file: ${destPath}. Destination must be fresh; collisions are rejected.`
      );
    }
  }

  // Ensure output directory structure exists
  if (!fs.existsSync(resolvedOutDir)) {
    fs.mkdirSync(resolvedOutDir, { recursive: true });
  }

  // Emit all artifacts to disk using exclusive 'wx' flag to prevent race overwrites and symlink attacks
  const emittedFiles = [];
  for (const [relPath, content] of targetFiles.entries()) {
    const destPath = path.join(resolvedOutDir, relPath);
    const destDir = path.dirname(destPath);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    try {
      fs.writeFileSync(destPath, content, { flag: 'wx' });
    } catch (err) {
      if (err.code === 'EEXIST') {
        throw new Error(
          `Refusing to overwrite existing destination file: ${destPath}. Destination must be fresh; collisions are rejected.`
        );
      }
      throw err;
    }
    emittedFiles.push(relPath);
  }

  return {
    entryPoint: resolvedEntryAbs,
    outDir: resolvedOutDir,
    isHtml,
    htmlFile: htmlFileName,
    entryFiles: bundleResult.entryFiles,
    emittedFiles,
    isMultiChunk: bundleResult.isMultiChunk,
    chunks: bundleResult.chunks,
    packageType: 'module'
  };
}
