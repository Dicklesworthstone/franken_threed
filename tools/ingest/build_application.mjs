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
 * - Positively supports local relative <base href="..."> within application root (directories and file-shaped bases),
 *   rewriting emitted entry chunks and preloads relative to the effective base directory to preserve document.baseURI semantics.
 * - Explicitly rejects remote, root-relative, or escaping bases before output to preserve document.baseURI semantics.
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
import { parseTagAttributes, parseSrcsetUrls, stripHtmlComments, stripScriptAndStyleBodies, parseHtmlEntries } from './html_parser.mjs';
import { resolveModuleSpecifier, urlToFilePath } from './resolver.mjs';
import { analyzeModuleAst } from './ast_analyzer.mjs';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';

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
 * Checks whether a URL string has a canonical external scheme (http:, https:, data:).
 * Does not misclassify root-relative (/...) or local file paths.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isExternalUrl(url) {
  if (!url || typeof url !== 'string') return false;
  try {
    const protocol = new URL(url, 'file:///').protocol;
    return protocol === 'http:' || protocol === 'https:' || protocol === 'data:';
  } catch {
    return false;
  }
}

/**
 * Normalizes a Rollup module ID or filesystem path into a canonical URL string.
 * Preserves existing URL schemes (e.g. file://, http://, https://) and query/fragment identities.
 *
 * @param {string} id
 * @returns {string | null} Canonical URL string, or null if invalid
 */
export function toCanonicalPreloadUrl(id) {
  if (!id || typeof id !== 'string') return null;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(id)) {
    return id;
  }
  try {
    return pathToFileURL(path.resolve(id)).href;
  } catch {
    return null;
  }
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
      const baseUrl = referrerDir.startsWith('file://')
        ? new URL(referrerDir)
        : pathToFileURL(referrerDir.endsWith(path.sep) ? referrerDir : referrerDir + path.sep);
      const canonicalUrl = new URL(trimmed, baseUrl).href;
      if (preloadChunkMap.has(canonicalUrl)) {
        return preloadChunkMap.get(canonicalUrl);
      }
    } catch {
      // Ignore invalid URL resolution
    }
  }

  // 2. Direct match for scheme-qualified URLs (when referrerDir is absent or rawHref is absolute)
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(trimmed)) {
    if (preloadChunkMap.has(trimmed)) {
      return preloadChunkMap.get(trimmed);
    }
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
 * @param {boolean} [skipModulePreloads=false] - Defer preloads until bundle chunks are known
 * @returns {string[]} Deduplicated list of relative resource URLs
 */
export function extractRelativeAssetUrls(rawHtmlContent, preloadChunkMap = null, referrerDir = '', skipModulePreloads = false) {
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
        (skipModulePreloads || (
          preloadChunkMap && findChunkForPreload(attrs.href, referrerDir, preloadChunkMap)
        ))
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
 * Strips comments from CSS content while preserving comment-like text
 * inside single- or double-quoted strings (e.g. url("image/slash-star-not-a-comment-star-slash.png")).
 *
 * @param {string} css
 * @returns {string}
 */
export function stripCssComments(css) {
  if (!css || typeof css !== 'string') return '';
  return css.replace(
    /(\/\*[\s\S]*?\*\/)|("(?:[^"\\]|\\.)*")|('(?:[^'\\]|\\.)*')/g,
    (match, comment) => {
      if (comment) {
        return comment.replace(/[^\r\n]/g, ' ');
      }
      return match;
    }
  );
}

/**
 * Contextual scanner regex for CSS resources.
 * Disambiguates comments, @import rules, url(...) functional notations, and string literals,
 * so that url(...) inside quoted strings (e.g. content: "url(phantom.png)") or inside comments
 * is never extracted as a resource dependency.
 */
export const CSS_RESOURCE_REGEX =
  /(\/\*[\s\S]*?\*\/)|(@import\s+(?:url\(\s*)?(?:"([^"\\]*(?:\\[\s\S][^"\\]*)*)"|'([^'\\]*(?:\\[\s\S][^'\\]*)*)'|((?:\\(?:[0-9a-f]{1,6}(?:\r\n|[ \t\r\n\f])?|[^\r\n\f0-9a-f])|[^\\'"\s();])+))\s*\)?)|(\burl\(\s*(?:"([^"\\]*(?:\\[\s\S][^"\\]*)*)"|'([^'\\]*(?:\\[\s\S][^'\\]*)*)'|((?:\\(?:[0-9a-f]{1,6}(?:\r\n|[ \t\r\n\f])?|[^\r\n\f0-9a-f])|[^\\'"()\s])+))\s*\))|("(?:[^"\\]|\\[\s\S])*")|('(?:[^'\\]|\\[\s\S])*')/gi;

/** Decode CSS string/URL escapes without changing the emitted stylesheet bytes.
 * https://www.w3.org/TR/css-syntax-3/#consume-escaped-code-point
 * Quoted strings additionally discard escaped newlines (CRLF is one newline).
 */
function decodeCssResourceUrl(rawUrl) {
  return rawUrl.replace(
    /\\(?:([0-9a-f]{1,6})(?:\r\n|[ \t\r\n\f])?|(\r\n|[\r\n\f])|([\s\S]))/gi,
    (_escape, hex, continuation, character) => {
      if (continuation) return '';
      if (!hex) return character;
      const codePoint = Number.parseInt(hex, 16);
      return codePoint === 0 || codePoint > 0x10ffff || (codePoint >= 0xd800 && codePoint <= 0xdfff)
        ? '\uFFFD'
        : String.fromCodePoint(codePoint);
    }
  );
}

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
      const url = rawUrl ? decodeCssResourceUrl(rawUrl) : '';
      if (isRelativeUrl(url)) {
        urls.add(url);
      }
      continue;
    }

    // match[6]: url(...) function
    if (match[6]) {
      const rawUrl = match[7] !== undefined ? match[7] : (match[8] !== undefined ? match[8] : match[9]);
      const url = rawUrl ? decodeCssResourceUrl(rawUrl) : '';
      if (isRelativeUrl(url)) {
        urls.add(url);
      }
    }
  }

  return Array.from(urls);
}

const CLASSIC_JS_MIME_TYPES = new Set([
  'text/javascript',
  'application/javascript',
  'text/ecmascript',
  'application/ecmascript',
  'text/jscript',
  'text/livescript',
  'application/x-javascript',
  'application/x-ecmascript',
  'text/javascript1.0',
  'text/javascript1.1',
  'text/javascript1.2',
  'text/javascript1.3',
  'text/javascript1.4',
  'text/javascript1.5',
]);

/**
 * Checks whether a <script> element's type attribute represents classic JavaScript.
 * Returns true if type is omitted, empty, or a standard JavaScript MIME type.
 * Returns false for 'module', 'importmap', or data blocks ('application/json', shaders, templates, etc.).
 *
 * @param {string | null | undefined} typeAttr
 * @returns {boolean}
 */
export function isClassicJavaScriptType(typeAttr) {
  if (!typeAttr || typeof typeAttr !== 'string') {
    return true;
  }
  const trimmed = typeAttr.trim().toLowerCase();
  if (!trimmed) {
    return true;
  }
  return CLASSIC_JS_MIME_TYPES.has(trimmed);
}

/**
 * Recursively extracts static string literals from conditional expression branches.
 * @param {any} node
 * @returns {string[] | null}
 */
function extractConditionalStringLiterals(node) {
  if (!node) return null;
  if (node.type === 'ConditionalExpression') {
    const consequent = extractConditionalStringLiterals(node.consequent);
    const alternate = extractConditionalStringLiterals(node.alternate);
    if (!consequent || !alternate) return null;
    return [...consequent, ...alternate];
  }
  if (node.type === 'Literal' && typeof node.value === 'string') {
    return [node.value];
  }
  if (node.type === 'TemplateLiteral' && node.expressions.length === 0 && node.quasis.length > 0) {
    return [node.quasis.map(q => q.value.cooked ?? q.value.raw).join('')];
  }
  return null;
}

/**
 * Scans JavaScript code (module or classic script) to discover static and dynamic
 * module specifiers as well as static asset references (new URL(..., import.meta.url)).
 *
 * Uses AST analysis via analyzeModuleAst for ESM, falling back to Acorn AST parsing
 * with sourceType: 'script' and acorn-walk ImportExpression visitor for classic scripts.
 * Never uses regex on JavaScript code.
 * Propagates the actual parse error if neither module nor script parses.
 *
 * @param {string} code - Script or module source code
 * @param {string} [contextUrl='script.js'] - File URL or identifier for error reporting
 * @returns {{ moduleSpecifiers: string[], assetSpecifiers: string[] }}
 */
export function extractJsModuleDependencies(code, contextUrl = 'script.js') {
  if (!code || typeof code !== 'string') {
    return { moduleSpecifiers: [], assetSpecifiers: [] };
  }

  const moduleSpecifiers = new Set();
  const assetSpecifiers = new Set();

  let moduleError = null;
  try {
    const analysis = analyzeModuleAst(code, contextUrl);
    if (analysis) {
      if (Array.isArray(analysis.staticImports)) {
        for (const st of analysis.staticImports) {
          if (st.specifier) moduleSpecifiers.add(st.specifier);
        }
      }
      if (Array.isArray(analysis.staticExports)) {
        for (const ex of analysis.staticExports) {
          if (ex.specifier) moduleSpecifiers.add(ex.specifier);
        }
      }
      if (Array.isArray(analysis.dynamicImports)) {
        for (const dyn of analysis.dynamicImports) {
          if (dyn.classification === 'literal' && dyn.specifier) {
            moduleSpecifiers.add(dyn.specifier);
          } else if (dyn.classification === 'finite_set' && Array.isArray(dyn.candidates)) {
            for (const cand of dyn.candidates) {
              if (cand) moduleSpecifiers.add(cand);
            }
          }
        }
      }
      if (Array.isArray(analysis.assetReferences)) {
        for (const assetRef of analysis.assetReferences) {
          if (assetRef.specifier) assetSpecifiers.add(assetRef.specifier);
        }
      }
      return {
        moduleSpecifiers: Array.from(moduleSpecifiers),
        assetSpecifiers: Array.from(assetSpecifiers)
      };
    }
  } catch (err) {
    moduleError = err;
  }

  // Fallback: parse as classic script (sourceType: 'script') via Acorn AST
  // Never regex JS.
  let scriptAst;
  try {
    scriptAst = acorn.parse(code, {
      ecmaVersion: 'latest',
      sourceType: 'script',
      locations: true,
      ranges: true
    });
  } catch (scriptErr) {
    throw moduleError || scriptErr;
  }

  walk.simple(scriptAst, {
    ImportExpression(node) {
      if (!node.source) return;
      if (node.source.type === 'Literal' && typeof node.source.value === 'string') {
        moduleSpecifiers.add(node.source.value);
      } else if (
        node.source.type === 'TemplateLiteral' &&
        node.source.expressions.length === 0 &&
        node.source.quasis.length > 0
      ) {
        const spec = node.source.quasis.map(q => q.value.cooked ?? q.value.raw).join('');
        if (spec) moduleSpecifiers.add(spec);
      } else if (node.source.type === 'ConditionalExpression') {
        const branches = extractConditionalStringLiterals(node.source);
        if (branches && Array.isArray(branches)) {
          for (const b of branches) {
            if (b) moduleSpecifiers.add(b);
          }
        }
      }
    }
  });

  return {
    moduleSpecifiers: Array.from(moduleSpecifiers),
    assetSpecifiers: Array.from(assetSpecifiers)
  };
}

/**
 * Resolves base href details and calculates the chunk prefix relative to the effective base directory.
 * If baseHref is empty, boolean, or absent, returns { effectiveBaseUrl: documentUrl, effectiveBaseDir: entryDir, chunkPrefix: './' }.
 * If baseHref is root-relative (/), protocol-relative (//), remote (http:, etc.), or escapes outside the root directory,
 * throws the explicit rejection error to preserve document.baseURI semantics.
 * Otherwise, computes the relative prefix from the effective base directory to the application root directory.
 *
 * @param {string | null} baseHref
 * @param {string} [entryDir='']
 * @param {string} [documentUrl='']
 * @returns {{ effectiveBaseUrl: string, effectiveBaseDir: string, chunkPrefix: string }}
 */
export function resolveBaseDetails(baseHref, entryDir = '', documentUrl = '') {
  const effectiveDocUrl = documentUrl || (entryDir ? pathToFileURL(path.join(entryDir, 'index.html')).href : 'file:///app/index.html');
  const effectiveEntryDir = entryDir ? path.resolve(entryDir) : fileURLToPath(new URL('./', effectiveDocUrl));

  if (baseHref === null || baseHref === '' || typeof baseHref !== 'string') {
    return {
      effectiveBaseUrl: effectiveDocUrl,
      effectiveBaseDir: effectiveEntryDir,
      chunkPrefix: './'
    };
  }

  const trimmed = baseHref.trim();
  if (trimmed === '') {
    return {
      effectiveBaseUrl: effectiveDocUrl,
      effectiveBaseDir: effectiveEntryDir,
      chunkPrefix: './'
    };
  }

  // Reject explicit URL schemes (e.g. file:, http:, https:), protocol-relative (//),
  // and leading slash/backslash before resolving, to strictly support local relative bases.
  if (
    trimmed.startsWith('/') ||
    trimmed.startsWith('\\') ||
    trimmed.startsWith('//') ||
    /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)
  ) {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`
    );
  }

  let resolvedBaseUrl;
  try {
    resolvedBaseUrl = new URL(trimmed, effectiveDocUrl);
  } catch {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`
    );
  }

  if (resolvedBaseUrl.protocol !== 'file:') {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`
    );
  }

  const baseDir = fileURLToPath(new URL('./', resolvedBaseUrl));
  const relFromEntry = path.relative(effectiveEntryDir, baseDir);

  if (relFromEntry === '..' || relFromEntry.startsWith('..' + path.sep) || path.isAbsolute(relFromEntry)) {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`
    );
  }

  const relFromBase = path.relative(baseDir, effectiveEntryDir).split(path.sep).join('/');
  let chunkPrefix = './';
  if (relFromBase && relFromBase !== '.') {
    chunkPrefix = relFromBase.endsWith('/') ? relFromBase : relFromBase + '/';
  }

  return {
    effectiveBaseUrl: resolvedBaseUrl.href,
    effectiveBaseDir: baseDir,
    chunkPrefix
  };
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
 * @param {string} [chunkPrefix='./'] - Relative directory prefix from base to output root
 * @returns {string}
 */
export function rewriteScriptTagAttributes(attrString, chunkFileName, chunkCode = '', chunkPrefix = './') {
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
      pieces.push(` src="${chunkPrefix}${chunkFileName}"`);
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
    pieces.push(` src="${chunkPrefix}${chunkFileName}"`);
  }

  return pieces.join('');
}



/**
 * Rewrites attributes of a <link rel="modulepreload"> tag that matches an emitted chunk.
 * Points href to "${chunkPrefix}${chunkFileName}" and recomputes SRI integrity honestly while preserving
 * unrelated attributes (e.g. data-href, data-integrity, as, crossorigin).
 *
 * @param {string} attrString
 * @param {string} chunkFileName
 * @param {string | Buffer} [chunkCode='']
 * @param {string} [chunkPrefix='./']
 * @returns {string}
 */
export function rewriteLinkTagAttributes(attrString, chunkFileName, chunkCode = '', chunkPrefix = './') {
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
      pieces.push(` href="${chunkPrefix}${chunkFileName}"`);
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
 * - Positively supports local relative <base href="..."> within application root (directories and file-shaped bases),
 *   rewriting emitted entry chunks and modulepreloads relative to the effective base directory.
 * - Explicitly rejects remote, root-relative, or escaping bases to preserve document.baseURI semantics.
 * - Rewrites <link rel="modulepreload"> hrefs to emitted chunks and updates SRI integrity.
 * - Enforces entry count equality in both directions.
 *
 * @param {string} rawHtmlContent
 * @param {string[]} entryFiles - Emitted entry chunk file names in document order
 * @param {Record<string, string>} [chunkFilesMap={}] - Mapping of chunk file name to emitted code
 * @param {Object} [options={}]
 * @param {Map<string, string>} [options.preloadChunkMap] - Mapping of module URL/path to emitted chunk
 * @param {string} [options.entryDir=''] - Directory of the entry HTML file
 * @param {string} [options.documentUrl=''] - URL of the entry HTML document
 * @returns {string}
 */
export function rewriteHtmlForBuild(rawHtmlContent, entryFiles, chunkFilesMap = {}, options = {}) {
  let moduleScriptIndex = 0;
  const preloadChunkMap = options.preloadChunkMap || null;
  const entryDir = options.entryDir || '';
  const documentUrl = options.documentUrl || (entryDir ? pathToFileURL(path.join(entryDir, 'index.html')).href : 'file:///app/index.html');

  // Discover effective first base href via parseHtmlEntries and validate base containment up front
  const parsedHtml = parseHtmlEntries(rawHtmlContent, documentUrl);
  const { effectiveBaseDir, chunkPrefix } = resolveBaseDetails(parsedHtml.baseHref, entryDir, documentUrl);

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
          // Keep the browser's empty-src error and ignored body, including handlers.
          if (attrs.src === '') return match;
          if (attrs.src && isExternalUrl(attrs.src)) {
            // External module root script (http:, https:, data:) preserved verbatim with original attributes
            return match;
          }

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
            chunkCode,
            chunkPrefix
          );

          return `<script${updatedAttrString}></script>`;
        }

        // Non-module scripts (including type="importmap") are preserved verbatim
        return match;
      }

      // Handle <base> tags: preserved verbatim in emitted HTML to keep document.baseURI semantics
      if (isBase) {
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
          const chunkFileName = findChunkForPreload(parsedLink.href, effectiveBaseDir, preloadChunkMap);
          if (chunkFileName) {
            const chunkCode = chunkFilesMap[chunkFileName] || '';
            const updatedLink = rewriteLinkTagAttributes(linkAttrs, chunkFileName, chunkCode, chunkPrefix);
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
    const entryBaseUrl = pathToFileURL(resolvedEntryAbs).href;
    const parsedHtml = parseHtmlEntries(rawHtml, entryBaseUrl);
    if (parsedHtml.baseHref !== null) {
      resolveBaseDetails(parsedHtml.baseHref, entryDir, entryBaseUrl);
    }
  }

  const targetFiles = new Map(); // relativePath -> string | Buffer
  const htmlFileName = isHtml ? path.basename(resolvedEntryAbs) : null;
  const retainedModuleUrls = new Map(); // exact source URL -> relative output URL

  // The same closure walk runs before bundling to establish retained module identity,
  // then afterward to validate chunk collisions and collect unbundled preloads.
  function collectHtmlAssets(preloadChunkMap, emittedChunkNames, skipModulePreloads = false) {
    if (!isHtml) return;

    const rawHtmlContent = fs.readFileSync(resolvedEntryAbs, 'utf-8');
    const entryBaseUrl = pathToFileURL(resolvedEntryAbs).href;
    const parsedHtml = parseHtmlEntries(rawHtmlContent, entryBaseUrl);
    const importMap = parsedHtml.importMap;
    const { effectiveBaseDir } = resolveBaseDetails(parsedHtml.baseHref, entryDir, entryBaseUrl);

    // Extract relative assets referenced by the HTML (stylesheets, images, media, non-module scripts)
    // Excludes modulepreloads that map to bundled chunks
    const relativeAssetUrls = extractRelativeAssetUrls(rawHtmlContent, preloadChunkMap, effectiveBaseDir, skipModulePreloads);

    // Bounded asset processing queue: handles direct HTML assets, transitive CSS url()/@import children,
    // and literal dynamic imports in retained classic scripts and modules.
    const assetQueue = [];
    for (const relUrl of relativeAssetUrls) {
      assetQueue.push({
        relUrl,
        referrerDir: effectiveBaseDir,
        referrerPath: resolvedEntryAbs,
        isModuleSpecifier: false
      });
    }

    // Scan inline classic scripts in HTML for static literal dynamic imports
    const sanitizedHtml = stripHtmlComments(rawHtmlContent);
    const inlineScriptRegex = /<script\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/script\s*>/gi;
    let inlineMatch;
    while ((inlineMatch = inlineScriptRegex.exec(sanitizedHtml)) !== null) {
      const attrs = parseTagAttributes(inlineMatch[1]);
      if (attrs.src !== undefined || !isClassicJavaScriptType(attrs.type)) {
        continue;
      }
      const scriptBody = inlineMatch[2];
      if (!scriptBody || !scriptBody.trim()) continue;

      const { moduleSpecifiers, assetSpecifiers } = extractJsModuleDependencies(scriptBody, parsedHtml.baseUrl);
      for (const spec of moduleSpecifiers) {
        assetQueue.push({
          specifier: spec,
          referrerUrl: parsedHtml.baseUrl,
          referrerDir: effectiveBaseDir,
          referrerPath: resolvedEntryAbs,
          isModuleSpecifier: true
        });
      }
      for (const assetSpec of assetSpecifiers) {
        assetQueue.push({
          relUrl: assetSpec,
          referrerDir: effectiveBaseDir,
          referrerPath: resolvedEntryAbs,
          isModuleSpecifier: false
        });
      }
    }

    const visitedCssPaths = new Set();
    const visitedJsUrls = new Set();

    while (assetQueue.length > 0) {
      const item = assetQueue.shift();
      let srcAssetAbs;
      let relFromEntryDir;
      let moduleUrl;
      const referrerPath = item.referrerPath || resolvedEntryAbs;

      if (item.isModuleSpecifier) {
        if (!item.specifier || typeof item.specifier !== 'string') continue;
        const trimmedSpecifier = item.specifier.trim();
        if (!trimmedSpecifier) continue;

        let resolvedUrl;
        try {
          resolvedUrl = resolveModuleSpecifier(trimmedSpecifier, item.referrerUrl, importMap, {
            mapBaseUrl: parsedHtml.baseUrl,
            packageRootUrl: options.packageRootUrl
          });
        } catch (err) {
          let context = '';
          if (referrerPath !== resolvedEntryAbs) {
            context = referrerPath.endsWith('.css')
              ? ` in CSS referenced from "${referrerPath}"`
              : ` referenced from "${referrerPath}"`;
          }
          throw new Error(`Unresolved module specifier${context}: "${trimmedSpecifier}" (${err.message})`);
        }

        if (!resolvedUrl.startsWith('file://')) {
          // Non-file URL (e.g. http://, https://, data:); resolved at runtime by browser
          continue;
        }

        srcAssetAbs = urlToFilePath(resolvedUrl);
        relFromEntryDir = path.relative(entryDir, srcAssetAbs);
        moduleUrl = resolvedUrl;
        // File copies share bytes, but query/fragment variants remain distinct modules.
        const entryDirectoryUrl = new URL('./', entryBaseUrl).href;
        retainedModuleUrls.set(resolvedUrl, './' + resolvedUrl.slice(entryDirectoryUrl.length));
      } else {
        const { relUrl, referrerDir } = item;
        if (!relUrl || typeof relUrl !== 'string') continue;
        const trimmedRelUrl = relUrl.trim();
        if (!trimmedRelUrl) continue;

        // Browser URI semantics: resolve against referrer directory URL using new URL,
        // then convert file: URL object directly to decoded filesystem path via fileURLToPath.
        // Preserves %20 and other percent-encodings as actual filename bytes on disk.
        // Wrapped in try/catch to safely handle asset paths containing unencoded literal '%'
        // not followed by two hex digits (e.g. <img src="./100%_sale.png">).
        try {
          const referrerDirSlash = referrerDir.endsWith(path.sep) ? referrerDir : referrerDir + path.sep;
          const referrerBaseUrl = pathToFileURL(referrerDirSlash);
          const resolvedUrl = new URL(trimmedRelUrl, referrerBaseUrl);
          srcAssetAbs = fileURLToPath(resolvedUrl);
        } catch {
          const cleanRelPath = trimmedRelUrl.split(/[?#]/)[0];
          srcAssetAbs = path.resolve(referrerDir, cleanRelPath);
        }

        relFromEntryDir = path.relative(entryDir, srcAssetAbs);
      }

      // Verify that relative asset or module path does not escape entry directory
      if (relFromEntryDir.startsWith('..') || path.isAbsolute(relFromEntryDir)) {
        const target = item.isModuleSpecifier ? item.specifier : item.relUrl;
        let context = '';
        if (referrerPath !== resolvedEntryAbs) {
          context = referrerPath.endsWith('.css') ? ` in CSS "${referrerPath}"` : ` in "${referrerPath}"`;
        }
        throw new Error(`Relative resource${context} escapes application root directory: "${target}"`);
      }

      // Pre-emission collision check: a classic script, stylesheet, module, or asset must NOT collide
      // with or silently overwrite/shadow an emitted chunk or the entry HTML file.
      if (emittedChunkNames.has(relFromEntryDir)) {
        const target = item.isModuleSpecifier ? item.specifier : item.relUrl;
        let context = '';
        if (referrerPath !== resolvedEntryAbs) {
          context = referrerPath.endsWith('.css')
            ? ` referenced from "${referrerPath}"`
            : ` referenced from "${referrerPath}"`;
        }
        throw new Error(
          `Collision detected: relative resource "${target}"${context} collides with emitted bundle chunk or entry file "${relFromEntryDir}". Source assets must not collide with emitted chunk names.`
        );
      }

      // Explicitly reject unresolved relative resources before output
      if (!fs.existsSync(srcAssetAbs) || !fs.statSync(srcAssetAbs).isFile()) {
        const target = item.isModuleSpecifier ? item.specifier : item.relUrl;
        let context = '';
        if (referrerPath !== resolvedEntryAbs) {
          context = referrerPath.endsWith('.css')
            ? ` in CSS referenced from "${referrerPath}"`
            : ` referenced from "${referrerPath}"`;
        }
        throw new Error(
          `Unresolved relative resource${context}: "${target}" not found at "${srcAssetAbs}"`
        );
      }

      // Copy asset/module into bounded closure targetFiles
      const assetData = targetFiles.get(relFromEntryDir) || fs.readFileSync(srcAssetAbs);
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
            referrerPath: srcAssetAbs,
            isModuleSpecifier: false
          });
        }
      }

      // If resource is a module or JS/MJS/CJS file (retained classic script),
      // scan for transitive module dependencies (static/dynamic imports, exports) and asset references
      if (
        (item.isModuleSpecifier || srcAssetAbs.endsWith('.js') || srcAssetAbs.endsWith('.mjs') || srcAssetAbs.endsWith('.cjs')) &&
        !visitedJsUrls.has(moduleUrl || pathToFileURL(srcAssetAbs).href)
      ) {
        const fileUrl = moduleUrl || pathToFileURL(srcAssetAbs).href;
        visitedJsUrls.add(fileUrl);
        const jsContent = assetData.toString('utf-8');
        const { moduleSpecifiers, assetSpecifiers } = extractJsModuleDependencies(jsContent, fileUrl);
        const jsDir = path.dirname(srcAssetAbs);

        for (const childSpec of moduleSpecifiers) {
          assetQueue.push({
            specifier: childSpec,
            referrerUrl: fileUrl,
            referrerDir: jsDir,
            referrerPath: srcAssetAbs,
            isModuleSpecifier: true
          });
        }

        for (const assetSpec of assetSpecifiers) {
          assetQueue.push({
            relUrl: assetSpec,
            referrerDir: jsDir,
            referrerPath: srcAssetAbs,
            isModuleSpecifier: false
          });
        }
      }
    }
  }

  collectHtmlAssets(null, new Set(htmlFileName ? [htmlFileName] : []), true);
  const bundleResult = await bundleWithRollup(resolvedEntryAbs, {
    packageRootUrl: options.packageRootUrl,
    retainedModuleUrls
  });
  const emittedChunkNames = new Set(Object.keys(bundleResult.files));
  if (htmlFileName) emittedChunkNames.add(htmlFileName);

  const preloadChunkMap = new Map();
  for (const chunk of bundleResult.chunks) {
    for (const modId of [chunk.facadeModuleId, ...chunk.modules]) {
      const canonicalKey = toCanonicalPreloadUrl(modId);
      if (canonicalKey) preloadChunkMap.set(canonicalKey, chunk.fileName);
    }
  }
  collectHtmlAssets(preloadChunkMap, emittedChunkNames);

  for (const [fileName, code] of Object.entries(bundleResult.files)) {
    targetFiles.set(fileName, code);
  }
  if (isHtml) {
    const entryBaseUrl = pathToFileURL(resolvedEntryAbs).href;
    targetFiles.set(htmlFileName, rewriteHtmlForBuild(
      fs.readFileSync(resolvedEntryAbs, 'utf-8'),
      bundleResult.entryFiles,
      bundleResult.files,
      { preloadChunkMap, entryDir, documentUrl: entryBaseUrl }
    ));
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
