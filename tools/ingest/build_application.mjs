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
 * - Supports root-relative document resources under an explicit documentRoot;
 *   rejects remote or escaping bases before output to preserve document.baseURI semantics.
 * - Rewrites <link rel="modulepreload"> hrefs to point to emitted chunks instead of raw
 *   source files, updating SRI integrity and skipping static asset copies for bundled modules.
 * - Enforces bounded asset closure by copying relative assets (stylesheets, images,
 *   non-module scripts, and referenced CSS url/@import children) or explicitly rejecting
 *   unresolved relative resources before output.
 * - Enforces entry count equality in both directions between HTML module scripts and emitted chunks.
 * - Strictly rejects collisions, race overwrites, and symlinks using exclusive 'wx' write flags.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as acorn from "acorn";
import * as walk from "acorn-walk";
import { analyzeModuleAst, classifyDynamicImportArgument, classifyImportType } from "./ast_analyzer.mjs";
import { bundleWithRollup } from "./bundler.mjs";
import {
  parseHtmlEntries,
  parseSrcsetUrls,
  parseTagAttributes,
  htmlContextTags,
  rewriteHtmlContextTags,
} from "./html_parser.mjs";
import { normalizeSpecifierKey, resolveModuleSpecifier, urlToFilePath, withCanonicalPackageRoot } from "./resolver.mjs";
import { buildModuleGraph } from "./module_graph.mjs";
import { IngestionParseError } from "./types.mjs";
import { moduleSourcesFor } from "./module_sources.mjs";
import { assertUnbundledModule, commonjsFormat } from "./commonjs_modules.mjs";
import { createImportedStringResolver } from "./module_string_constants.mjs";
import { moduleWorkerRollupPlugin, workerGlobalBindings } from "./module_workers.mjs";
import { analyzeDocumentFetch, collectDocumentFetchAssets } from "./document_fetch_assets.mjs";
import { cssResources } from "./css_resources.mjs";
export { CSS_RESOURCE_REGEX } from "./css_resources.mjs";
import { gltfAssetUris } from "./pack_gltf.mjs";
import { evaluateGraphRoutes, isInternalLibraryModule } from "./route_bridge.mjs";
import { resolveClosedRendererGroups } from "./renderer_resource_groups.mjs";
import { decideRendererRoute } from "../compat/route_decider.mjs";
import { transformThreeGpuComputation } from "./gpu_computation_contracts.mjs";
import { normalizeDocumentRoot, resolveDocumentUrl, documentRelativePath, isRootRelativeDocumentUrl } from "./document_urls.mjs";

/**
 * Recomputes Subresource Integrity (SRI) string for modified or bundled content,
 * matching the algorithm(s) specified in the original integrity attribute.
 *
 * @param {string} originalIntegrity - e.g. "sha384-..." or "sha256-... sha384-..."
 * @param {string | Buffer} content - The actual emitted chunk content
 * @returns {string}
 */
export function computeIntegrityForContent(originalIntegrity, content) {
  if (!originalIntegrity || typeof originalIntegrity !== "string") {
    return "";
  }
  const buffer = typeof content === "string" ? Buffer.from(content, "utf-8") : content;
  const tokens = originalIntegrity.trim().split(/\s+/);
  const updatedTokens = [];

  for (const token of tokens) {
    const dashIdx = token.indexOf("-");
    if (dashIdx === -1) {
      throw new Error(`Invalid integrity attribute token format: "${token}"`);
    }
    const algo = token.slice(0, dashIdx).toLowerCase();
    if (!["sha256", "sha384", "sha512"].includes(algo)) {
      throw new Error(`Unsupported integrity hash algorithm: "${algo}"`);
    }
    const hash = crypto.createHash(algo).update(buffer).digest("base64");
    updatedTokens.push(`${algo}-${hash}`);
  }

  return updatedTokens.join(" ");
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
  if (!url || typeof url !== "string") return false;
  const trimmed = url.trim();
  if (!trimmed) return false;
  if (trimmed.startsWith("#") || trimmed.startsWith("//") || trimmed.startsWith("/")) {
    return false;
  }
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
    return false;
  }
  return true;
}

function isDocumentResourceUrl(url, documentRoot) {
  return isRelativeUrl(url) || Boolean(documentRoot && typeof url === "string" && isRootRelativeDocumentUrl(url));
}

/**
 * Checks whether a URL string has a canonical external scheme (http:, https:, data:).
 * Does not misclassify root-relative (/...) or local file paths.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isExternalUrl(url) {
  if (!url || typeof url !== "string") return false;
  try {
    const protocol = new URL(url, "file:///").protocol;
    return protocol === "http:" || protocol === "https:" || protocol === "data:";
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
  if (!id || typeof id !== "string") return null;
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
export function findChunkForPreload(rawHref, referrerDir, preloadChunkMap, documentRoot = null) {
  if (!rawHref || typeof rawHref !== "string" || !preloadChunkMap) return null;
  const trimmed = rawHref.trim();
  if (!trimmed) return null;

  // 1. Canonical URL resolution against referrer directory URL (preserving ?query and #fragment)
  if (referrerDir) {
    try {
      const baseUrl = referrerDir.startsWith("file://")
        ? new URL(referrerDir)
        : pathToFileURL(referrerDir.endsWith(path.sep) ? referrerDir : referrerDir + path.sep);
      const canonicalUrl = resolveDocumentUrl(trimmed, baseUrl.href, documentRoot);
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
 * icons, images, non-module scripts, media, unbundled modulepreloads) and inline CSS.
 *
 * Uses quote-aware tag scanners so attributes containing ">" are never truncated.
 *
 * @param {string} rawHtmlContent
 * @param {Map<string, string>} [preloadChunkMap] - Mapped bundle chunks to exclude from static copy
 * @param {string} [referrerDir=''] - Base directory for resolving relative URLs
 * @param {boolean} [skipModulePreloads=false] - Defer preloads until bundle chunks are known
 * @returns {string[]} Deduplicated list of relative resource URLs
 */
export function extractRelativeAssetUrls(
  rawHtmlContent,
  preloadChunkMap = null,
  referrerDir = "",
  skipModulePreloads = false,
  documentRoot = null,
) {
  const tags = [...htmlContextTags(rawHtmlContent)].filter(tag => !tag.closing);
  const assets = new Set();

  // 1. <link ...> quote-aware
  for (const tag of tags.filter(tag => tag.name === "link")) {
    const attrs = parseTagAttributes(tag.attributes);
    if (attrs.href && isDocumentResourceUrl(attrs.href, documentRoot)) {
      // If this is a modulepreload matching an emitted chunk, it is handled by Rollup bundle emission
      if (
        attrs.rel &&
        attrs.rel.toLowerCase() === "modulepreload" &&
        (skipModulePreloads ||
          (preloadChunkMap && findChunkForPreload(attrs.href, referrerDir, preloadChunkMap, documentRoot)))
      ) {
        continue;
      }
      assets.add(attrs.href);
    }
  }

  // 2. <img ...> quote-aware
  for (const tag of tags.filter(tag => tag.name === "img")) {
    const attrs = parseTagAttributes(tag.attributes);
    if (attrs.src && isDocumentResourceUrl(attrs.src, documentRoot)) {
      assets.add(attrs.src);
    }
    if (attrs.srcset) {
      const srcsetUrls = parseSrcsetUrls(attrs.srcset);
      for (const u of srcsetUrls) {
        if (isDocumentResourceUrl(u, documentRoot)) {
          assets.add(u);
        }
      }
    }
  }

  // 3. <script ...> where type != 'module' quote-aware
  for (const tag of tags.filter(tag => tag.name === "script")) {
    const attrs = parseTagAttributes(tag.attributes);
    const type = (attrs.type || "text/javascript").toLowerCase();
    if (type !== "module" && attrs.src && isDocumentResourceUrl(attrs.src, documentRoot)) {
      assets.add(attrs.src);
    }
  }

  // 4. <video>, <audio>, <source>, <track> quote-aware
  for (const tag of tags.filter(tag => ["video", "audio", "source", "track"].includes(tag.name))) {
    const attrs = parseTagAttributes(tag.attributes);
    if (attrs.src && isDocumentResourceUrl(attrs.src, documentRoot)) {
      assets.add(attrs.src);
    }
    if (attrs.poster && isDocumentResourceUrl(attrs.poster, documentRoot)) {
      assets.add(attrs.poster);
    }
    if (attrs.srcset) {
      const srcsetUrls = parseSrcsetUrls(attrs.srcset);
      for (const u of srcsetUrls) {
        if (isDocumentResourceUrl(u, documentRoot)) {
          assets.add(u);
        }
      }
    }
  }

  // Inline styles share the document's effective base with markup resources.
  // Consume complete raw-text elements and quoted tags so markup-looking source
  // strings, comments and attribute values cannot create phantom dependencies.
  for (const tag of tags) {
    const name = tag.name;
    if (!name) continue;
    const attrs = parseTagAttributes(tag.attributes);
    if (attrs.style !== undefined) {
      for (const url of extractRelativeCssUrls(decodeHtmlAssetAttribute(attrs.style), documentRoot)) {
        assets.add(url);
      }
    }
    if (name === "style" && ["", "text/css"].includes((attrs.type ?? "").trim().toLowerCase())) {
      // Unlike attributes, a style element's raw text does not decode HTML entities.
      for (const url of extractRelativeCssUrls(tag.body ?? "", documentRoot)) assets.add(url);
    }
    const resource = name === "image"
      ? attrs.href ?? attrs["xlink:href"]
      : name === "input" && (attrs.type ?? "").toLowerCase() === "image"
        ? attrs.src
        : undefined;
    if (resource !== undefined) {
      const url = decodeHtmlAssetAttribute(resource);
      if (isDocumentResourceUrl(url, documentRoot)) assets.add(url);
    }
  }

  return Array.from(assets);
}

// Decode the XML-compatible named references and numeric character references
// used in resource-bearing attributes. Source HTML remains byte-for-byte intact.
function decodeHtmlAssetAttribute(value) {
  const named = { amp: "&", AMP: "&", quot: '"', QUOT: '"', apos: "'", lt: "<", LT: "<", gt: ">", GT: ">" };
  return value.replace(/&(?:#([xX][\da-fA-F]+|\d+);?|([a-zA-Z][\da-zA-Z]+);|(amp|AMP|quot|QUOT|lt|LT|gt|GT)(?![=\da-zA-Z]))/g,
    (whole, number, name, legacy) => {
      if (number === undefined) return named[name ?? legacy] ?? whole;
      const codePoint = /^[xX]/.test(number) ? Number.parseInt(number.slice(1), 16) : Number(number);
      return codePoint === 0 || codePoint > 0x10ffff || (codePoint >= 0xd800 && codePoint <= 0xdfff)
        ? "\uFFFD"
        : String.fromCodePoint(codePoint);
    });
}

/**
 * Strips comments from CSS content while preserving comment-like text
 * inside single- or double-quoted strings (e.g. url("image/slash-star-not-a-comment-star-slash.png")).
 *
 * @param {string} css
 * @returns {string}
 */
export function stripCssComments(css) {
  if (!css || typeof css !== "string") return "";
  return css.replace(
    /(\/\*[\s\S]*?\*\/)|("(?:[^"\\]|\\.)*")|('(?:[^'\\]|\\.)*')/g,
    (match, comment) => {
      if (comment) {
        return comment.replace(/[^\r\n]/g, " ");
      }
      return match;
    },
  );
}

/**
 * Extracts relative resource URLs (url(...), @import and image-set) from CSS content.
 * Skips url(...) occurrences inside single- or double-quoted CSS strings and comments.
 *
 * @param {string} cssContent
 * @returns {string[]} Deduplicated list of relative URLs
 */
export function extractRelativeCssUrls(cssContent, documentRoot = null, options = {}) {
  if (!cssContent || typeof cssContent !== "string") return [];
  const urls = new Set();
  for (const { url } of cssResources(cssContent, options)) {
    if (isDocumentResourceUrl(url, documentRoot)) urls.add(url);
  }

  return Array.from(urls);
}

const CLASSIC_JS_MIME_TYPES = new Set([
  "text/javascript",
  "application/javascript",
  "text/ecmascript",
  "application/ecmascript",
  "text/jscript",
  "text/livescript",
  "application/x-javascript",
  "application/x-ecmascript",
  "text/javascript1.0",
  "text/javascript1.1",
  "text/javascript1.2",
  "text/javascript1.3",
  "text/javascript1.4",
  "text/javascript1.5",
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
  if (!typeAttr || typeof typeAttr !== "string") {
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
  if (node.type === "ConditionalExpression") {
    const consequent = extractConditionalStringLiterals(node.consequent);
    const alternate = extractConditionalStringLiterals(node.alternate);
    if (!consequent || !alternate) return null;
    return [...consequent, ...alternate];
  }
  if (node.type === "Literal" && typeof node.value === "string") {
    return [node.value];
  }
  if (node.type === "TemplateLiteral" && node.expressions.length === 0 && node.quasis.length > 0) {
    return [node.quasis.map((q) => q.value.cooked ?? q.value.raw).join("")];
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
export function extractJsModuleDependencies(code, contextUrl = "script.js") {
  if (!code || typeof code !== "string") {
    return { moduleSpecifiers: [], moduleRequests: [], assetSpecifiers: [] };
  }

  const moduleSpecifiers = new Set();
  const moduleRequests = new Map();
  let runtimeImports = false;
  const addModule = (specifier, moduleType = "javascript") => {
    moduleSpecifiers.add(specifier);
    moduleRequests.set(JSON.stringify([specifier, moduleType]), {specifier, moduleType});
  };
  const assetSpecifiers = new Set();

  let moduleError = null;
  try {
    const analysis = analyzeModuleAst(code, contextUrl);
    if (analysis) {
      if (Array.isArray(analysis.staticImports)) {
        for (const st of analysis.staticImports) {
          if (st.specifier) addModule(st.specifier, st.module_type);
        }
      }
      if (Array.isArray(analysis.staticExports)) {
        for (const ex of analysis.staticExports) {
          if (ex.specifier) addModule(ex.specifier, ex.module_type);
        }
      }
      if (Array.isArray(analysis.dynamicImports)) {
        for (const dyn of analysis.dynamicImports) {
          if (dyn.classification === "nonliteral" || dyn.module_type === "unknown") runtimeImports = true;
          if (dyn.classification === "literal" && dyn.specifier) {
            addModule(dyn.specifier, dyn.module_type);
          } else if (dyn.classification === "finite_set" && Array.isArray(dyn.candidates)) {
            for (const cand of dyn.candidates) {
              if (cand) addModule(cand, dyn.module_type);
            }
          }
        }
      }
      if (Array.isArray(analysis.assetReferences)) {
        for (const assetRef of analysis.assetReferences) {
          for (const specifier of assetRef.candidates ?? [assetRef.specifier]) {
            if (specifier) assetSpecifiers.add(specifier);
          }
        }
      }
      return {
        moduleSpecifiers: Array.from(moduleSpecifiers),
        moduleRequests: Array.from(moduleRequests.values()),
        assetSpecifiers: Array.from(assetSpecifiers),
        runtimeImports,
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
      ecmaVersion: "latest",
      sourceType: "script",
      locations: true,
      ranges: true,
    });
  } catch (scriptErr) {
    throw moduleError || scriptErr;
  }

  walk.simple(scriptAst, {
    ImportExpression(node) {
      if (!node.source) return;
      if (classifyDynamicImportArgument(node.source).classification === "nonliteral") runtimeImports = true;
      const moduleType = classifyImportType(node);
      if (node.source.type === "Literal" && typeof node.source.value === "string") {
        addModule(node.source.value, moduleType);
      } else if (
        node.source.type === "TemplateLiteral" &&
        node.source.expressions.length === 0 &&
        node.source.quasis.length > 0
      ) {
        const spec = node.source.quasis.map((q) => q.value.cooked ?? q.value.raw).join("");
        if (spec) addModule(spec, moduleType);
      } else if (node.source.type === "ConditionalExpression") {
        const branches = extractConditionalStringLiterals(node.source);
        if (branches && Array.isArray(branches)) {
          for (const b of branches) {
            if (b) addModule(b, moduleType);
          }
        }
      }
    },
  });

  return {
    moduleSpecifiers: Array.from(moduleSpecifiers),
    moduleRequests: Array.from(moduleRequests.values()),
    assetSpecifiers: Array.from(assetSpecifiers),
    runtimeImports,
  };
}

/**
 * Resolves base href details and calculates the chunk prefix relative to the effective base directory.
 * The output root is documentRoot when declared, otherwise the HTML entry directory.
 * Root-relative bases require that explicit root; remote or escaping bases are rejected.
 * The prefix keeps emitted chunks reachable from the unchanged source base href.
 *
 * @param {string | null} baseHref
 * @param {string} [entryDir='']
 * @param {string} [documentUrl='']
 * @returns {{ effectiveBaseUrl: string, effectiveBaseDir: string, chunkPrefix: string }}
 */
export function resolveBaseDetails(baseHref, entryDir = "", documentUrl = "", documentRoot = null) {
  const effectiveDocUrl =
    documentUrl ||
    (entryDir ? pathToFileURL(path.join(entryDir, "index.html")).href : "file:///app/index.html");
  const effectiveEntryDir = entryDir
    ? path.resolve(entryDir)
    : fileURLToPath(new URL("./", effectiveDocUrl));
  const declaredRoot = normalizeDocumentRoot(documentRoot);
  const applicationRoot = declaredRoot ?? effectiveEntryDir;
  const trimmed = typeof baseHref === "string" ? baseHref.trim() : "";

  // Reject explicit URL schemes (e.g. file:, http:, https:), protocol-relative (//),
  // and filesystem-only forms. An origin-root base requires its declared local root.
  if (
    (!declaredRoot && trimmed.startsWith("/")) ||
    trimmed.startsWith("\\") ||
    trimmed.startsWith("//") ||
    /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)
  ) {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`,
    );
  }

  let resolvedBaseUrl;
  try {
    resolvedBaseUrl = new URL(trimmed ? resolveDocumentUrl(trimmed, effectiveDocUrl, declaredRoot) : effectiveDocUrl);
  } catch {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`,
    );
  }

  if (resolvedBaseUrl.protocol !== "file:") {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`,
    );
  }

  const baseDir = trimmed ? fileURLToPath(new URL("./", resolvedBaseUrl)) : effectiveEntryDir;
  if (declaredRoot) documentRelativePath(baseDir, declaredRoot);
  const relFromEntry = path.relative(applicationRoot, baseDir);

  if (
    relFromEntry === ".." ||
    relFromEntry.startsWith(".." + path.sep) ||
    path.isAbsolute(relFromEntry)
  ) {
    throw new Error(
      `Explicit rejection: <base href="${baseHref}"> is not currently supported in application build emitter to preserve document.baseURI semantics; base href support remains open.`,
    );
  }

  const relFromBase = path.relative(baseDir, applicationRoot).split(path.sep).join("/");
  let chunkPrefix = "./";
  if (relFromBase && relFromBase !== ".") {
    chunkPrefix = relFromBase.endsWith("/") ? relFromBase : relFromBase + "/";
  }

  return {
    effectiveBaseUrl: resolvedBaseUrl.href,
    effectiveBaseDir: baseDir,
    chunkPrefix,
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
export function rewriteScriptTagAttributes(
  attrString,
  chunkFileName,
  chunkCode = "",
  chunkPrefix = "./",
) {
  const attrRegex = /(?:^|\s+)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;

  let match;
  let hasSrc = false;
  let lastIndex = 0;
  const pieces = [];

  while ((match = attrRegex.exec(attrString)) !== null) {
    const fullMatch = match[0];
    const name = match[1].toLowerCase();
    const val =
      match[2] !== undefined
        ? match[2]
        : match[3] !== undefined
          ? match[3]
          : match[4] !== undefined
            ? match[4]
            : "";

    // Preserve any whitespace/characters between last token and current match
    if (match.index > lastIndex) {
      pieces.push(attrString.slice(lastIndex, match.index));
    }
    lastIndex = attrRegex.lastIndex;

    if (name === "src") {
      hasSrc = true;
      pieces.push(` src="${chunkPrefix}${chunkFileName}"`);
    } else if (name === "integrity") {
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

  return pieces.join("");
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
export function rewriteLinkTagAttributes(
  attrString,
  chunkFileName,
  chunkCode = "",
  chunkPrefix = "./",
) {
  const attrRegex = /(?:^|\s+)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let match;
  let lastIndex = 0;
  const pieces = [];

  while ((match = attrRegex.exec(attrString)) !== null) {
    const fullMatch = match[0];
    const name = match[1].toLowerCase();
    const val =
      match[2] !== undefined
        ? match[2]
        : match[3] !== undefined
          ? match[3]
          : match[4] !== undefined
            ? match[4]
            : "";

    if (match.index > lastIndex) {
      pieces.push(attrString.slice(lastIndex, match.index));
    }
    lastIndex = attrRegex.lastIndex;

    if (name === "href") {
      pieces.push(` href="${chunkPrefix}${chunkFileName}"`);
    } else if (name === "integrity") {
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

  return pieces.join("");
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
 * - Root-relative bases require an explicit documentRoot; remote or escaping bases are rejected.
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
  let sourceModuleIndex = 0;
  let publicMapEmitted = false;
  const preloadChunkMap = options.preloadChunkMap || null;
  const entryDir = options.entryDir || "";
  const documentUrl =
    options.documentUrl ||
    (entryDir ? pathToFileURL(path.join(entryDir, "index.html")).href : "file:///app/index.html");

  // Discover effective first base href via parseHtmlEntries and validate base containment up front
  const parsedHtml = parseHtmlEntries(rawHtmlContent, documentUrl, {documentRoot: options.documentRoot});
  const { effectiveBaseDir, chunkPrefix } = resolveBaseDetails(
    parsedHtml.baseHref,
    entryDir,
    documentUrl,
    options.documentRoot,
  );

  const rewritten = rewriteHtmlContextTags(rawHtmlContent, tag => {
      const match = tag.text;
      if (tag.closing) return match;

      // Handle <script> tags
      if (tag.name === "script" && tag.closed) {
        const scriptAttrs = tag.attributes, scriptBody = tag.body;
        if (options.classicScriptRewrites?.has(match)) return options.classicScriptRewrites.get(match);
        const attrs = parseTagAttributes(scriptAttrs);
        const scriptType = (attrs.type || "text/javascript").toLowerCase();

        if (scriptType === "importmap" && Object.keys(options.publicImports ?? {}).length) {
          const encode = map => JSON.stringify(map).replace(/</g, "\\u003c");
          const sourceMap = JSON.parse(scriptBody);
          const names = Object.keys(sourceMap.imports ?? {}).filter(name => Object.hasOwn(options.publicImports, name));
          for (const name of names) delete sourceMap.imports[name];
          const original = names.length ? `<script${scriptAttrs}>${encode(sourceMap)}</script>` : match;
          if (publicMapEmitted) return original;
          publicMapEmitted = true;
          const imports = Object.fromEntries(Object.entries(options.publicImports).map(([name, file]) => [name, chunkPrefix + file]));
          return `<script type="importmap" data-f3d-public-modules>${encode({imports})}</script>\n${original}`;
        }
        if (scriptType === "module") {
          const sourceScript = parsedHtml.moduleScripts[sourceModuleIndex++];
          const retained = options.retainedHtmlModules?.get(sourceScript.id);
          if (retained !== undefined) {
            if (sourceScript.inlineContent !== null) {
              // Inline imports use the document's native base and import-map
              // scope. Moving this code into a chunk changes runtime imports.
              if (/<\/script[\t\n\f\r />]/i.test(retained))
                throw new Error("A transformed retained inline module contains an HTML script end tag");
              return `<script${scriptAttrs}>${retained}</script>`;
            }
            if (!options.changedRetainedModules?.has(sourceScript.id)) return match;
            return `<script${rewriteScriptTagAttributes(scriptAttrs, attrs.src, retained, "")}>${scriptBody}</script>`;
          }
          // Keep the browser's empty-src error and ignored body, including handlers.
          if (attrs.src === "") return match;
          if (attrs.src && isExternalUrl(attrs.src)) {
            // External module root script (http:, https:, data:) preserved verbatim with original attributes
            return match;
          }

          if (moduleScriptIndex >= entryFiles.length) {
            throw new Error(
              `Module script at index ${moduleScriptIndex} exceeds emitted entry chunk count (${entryFiles.length})`,
            );
          }

          const chunkFileName = entryFiles[moduleScriptIndex++];
          const chunkCode = chunkFilesMap[chunkFileName] || "";

          const updatedAttrString = rewriteScriptTagAttributes(
            scriptAttrs,
            chunkFileName,
            chunkCode,
            chunkPrefix,
          );

          return `<script${updatedAttrString}></script>`;
        }

        // Non-module scripts (including type="importmap") are preserved verbatim
        return match;
      }

      // Handle <base> tags: preserved verbatim in emitted HTML to keep document.baseURI semantics
      if (tag.name === "base") {
        return match;
      }

      // Handle <link> tags: if modulepreload matches an emitted chunk, rewrite href and update SRI integrity
      if (tag.name === "link") {
        const linkAttrs = tag.attributes;
        const parsedLink = parseTagAttributes(linkAttrs);
        if (
          parsedLink.rel &&
          parsedLink.rel.toLowerCase() === "modulepreload" &&
          parsedLink.href &&
          preloadChunkMap
        ) {
          const chunkFileName = findChunkForPreload(
            parsedLink.href,
            effectiveBaseDir,
            preloadChunkMap,
            options.documentRoot,
          );
          if (chunkFileName) {
            const chunkCode = chunkFilesMap[chunkFileName] || "";
            const updatedLink = rewriteLinkTagAttributes(
              linkAttrs,
              chunkFileName,
              chunkCode,
              chunkPrefix,
            );
            return `<link${updatedLink}>`;
          }
        }
        return match;
      }

      return match;
    });

  // Enforce entry count equality in both directions
  if (moduleScriptIndex !== entryFiles.length) {
    throw new Error(
      `Module script count in HTML (${moduleScriptIndex}) does not match emitted entry chunk count (${entryFiles.length})`,
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
 * Module-only output carries its own ESM package scope for relocatable .js chunks.
 *
 * @param {string} entryPath - Path to HTML or JavaScript entry file
 * @param {string} outDir - Destination directory for emitted application (must be fresh)
 * @param {Object} [options]
 * @param {'browser'|'node'} [options.target='browser'] - Deployment host; Node emits a cold ESM package
 * @param {string} [options.packageRootUrl] - Base URL or directory for package fallback
 * @param {Function} [options.transformModule] - Synchronous project frontend: (source, {url, sourceType}) => code | {code} | null
 * @param {boolean} [options.prepareStandaloneLoaders] - Embed finite pinned-loader resources for a following packHtml call; unknown routes remain networked
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
  options = withCanonicalPackageRoot(options);
  const moduleSources = moduleSourcesFor(options);
  const target = moduleSources.target ?? 'browser';
  if (!entryPath || typeof entryPath !== "string") {
    throw new Error("buildApplication requires a valid entryPath string");
  }
  if (!outDir || typeof outDir !== "string") {
    throw new Error("buildApplication requires a valid outDir string");
  }

  let resolvedEntryAbs = path.resolve(entryPath);
  if (!fs.existsSync(resolvedEntryAbs)) {
    throw new Error(`Entry path does not exist: ${entryPath}`);
  }
  if (target === 'node') resolvedEntryAbs = fs.realpathSync(resolvedEntryAbs);

  const resolvedOutDir = path.resolve(outDir);
  const isHtml = entryPath.endsWith(".html") || entryPath.endsWith(".htm");
  const entryDir = path.dirname(resolvedEntryAbs);
  const documentRoot = normalizeDocumentRoot(options.documentRoot);
  if (target === 'node' && (isHtml || documentRoot || options.routeWebGPURenderer ||
      options.routeWebGLRenderer || options.prepareStandaloneLoaders))
    throw new TypeError('The Node target requires a JavaScript entry without browser document, packing or renderer options');
  const applicationRoot = documentRoot ?? entryDir;
  const htmlFileName = isHtml ? documentRoot ? documentRelativePath(resolvedEntryAbs, documentRoot) : path.basename(resolvedEntryAbs) : null;

  if (isHtml) {
    const rawHtml = fs.readFileSync(resolvedEntryAbs, "utf-8");
    const entryBaseUrl = pathToFileURL(resolvedEntryAbs).href;
    const parsedHtml = parseHtmlEntries(rawHtml, entryBaseUrl, {documentRoot});
    if (parsedHtml.baseHref !== null) {
      resolveBaseDetails(parsedHtml.baseHref, entryDir, entryBaseUrl, documentRoot);
    }
  }

  const targetFiles = new Map(); // relativePath -> string | Buffer
  if (!isHtml) {
    // .js chunks must keep their ESM semantics even when deployed beneath a
    // CommonJS package. Reserve the scope file before merging emitted resources
    // so it follows the same collision and exclusive-write rules as the chunks.
    targetFiles.set("package.json", JSON.stringify({ type: "module" }, null, 2) + "\n");
  }
  const retainedModuleUrls = new Map(); // exact source URL -> relative output URL
  const retainedHtmlModules = new Map(), runtimeModuleUrls = new Set(), changedRetainedModules = new Set();
  let hasRuntimeImports = false;
  // Both native import maps and renderer routing inspect the same normalized
  // source graph. Keep it local to this build, including its frontend options.
  let moduleGraphPromise;
  const getModuleGraph = () => moduleGraphPromise ??= buildModuleGraph(resolvedEntryAbs, {
    documentRoot, _moduleSources: moduleSources, packageRootUrl: options.packageRootUrl,
  });
  let nativeMapRouting = false;
  const classicWorkerSources = new Map(), classicScriptRewrites = new Map();
  const documentWorkerGlobals = new Set();
  const workerNumericUnits = new Map();

  if (isHtml) {
    // Opaque runtime imports retain their source URL: bundling into a different
    // directory changes relative imports and scoped import-map selection. This
    // is a networked route, not a proof that the dynamic module graph closes.
    const parsed = parseHtmlEntries(fs.readFileSync(resolvedEntryAbs, "utf8"), pathToFileURL(resolvedEntryAbs).href, {documentRoot});
    if (Object.keys(parsed.importMap.imports ?? {}).length || Object.keys(parsed.importMap.scopes ?? {}).length) {
      const graph = await getModuleGraph();
      // Keep compiler-only package inputs on the established bundler path.
      // A conservative document-wide check also covers parents later retained
      // to avoid resolving an import-map target a second time.
      nativeMapRouting = Object.values(graph.modules).every(node => {
        if (node.source_format && node.source_format !== 'module') return false;
        const requests = [...node.static_imports, ...node.dynamic_imports.flatMap(request => request.resolved_targets ?? [request])];
        return requests.every(request => {
          if (!request.resolved_url) return true;
          try { return resolveModuleSpecifier(request.specifier, node.is_inline ? parsed.baseUrl : node.source_url,
            parsed.importMap, {mapBaseUrl: parsed.baseUrl, documentRoot, _nativeImportsOnly: true}) === request.resolved_url; }
          catch { return false; }
        });
      });
      if (nativeMapRouting) {
        const inline = new Map(parsed.moduleScripts.filter(script => script.inlineContent !== null).map(script => [script.id, script]));
        for (const node of Object.values(graph.modules)) {
          if (!node.dynamic_imports.some(request => request.classification === "nonliteral" || request.module_type === "unknown")) continue;
          if (node.is_inline) retainedHtmlModules.set(node.id, moduleSources.read(node.id, {source: inline.get(node.id).inlineContent}).code);
          else runtimeModuleUrls.add(node.source_url);
        }
      }
    }
  }

  if (isHtml) {
    const rawHtml = fs.readFileSync(resolvedEntryAbs, "utf8"), documentUrl = pathToFileURL(resolvedEntryAbs).href;
    const parsed = parseHtmlEntries(rawHtml, documentUrl, {documentRoot}), scripts = [];
    for (const tag of htmlContextTags(rawHtml)) {
      if (tag.name !== "script" || tag.closing || !tag.closed) continue;
      const attrs = parseTagAttributes(tag.attributes);
      const sourceType = attrs.type?.trim().toLowerCase() === "module" ? "module" : "script";
      if ((sourceType === "script" && !isClassicJavaScriptType(attrs.type)) || attrs.src === "") continue;
      let source = tag.body, file = null, original = Buffer.from(source);
      if (attrs.src !== undefined) {
        const url = new URL(resolveDocumentUrl(attrs.src, parsed.baseUrl, documentRoot));
        if (url.protocol !== "file:" || !isDocumentResourceUrl(attrs.src, documentRoot)) {
          // An uninspected external script can replace the document globals.
          documentWorkerGlobals.add("Worker"); documentWorkerGlobals.add("SharedWorker"); continue;
        }
        file = fileURLToPath(url);
        const relative = documentRoot ? documentRelativePath(file, documentRoot) : path.relative(entryDir, file);
        if (relative.startsWith("..") || path.isAbsolute(relative)) continue; // Existing asset closure reports this.
        if (!fs.existsSync(file) || !fs.statSync(file).isFile()) continue;
        original = fs.readFileSync(file);
        source = moduleSources.read(url.href, {sourceType}).code;
      }
      for (const name of workerGlobalBindings(source, {sourceType})) documentWorkerGlobals.add(name);
      if (sourceType === "script") scripts.push({tag, attrs, source, file, original});
    }
    const plugin = moduleWorkerRollupPlugin({_moduleSources: moduleSources, sourceType: "script",
      specializeNumeric: options.specializeNumeric, _workerNumericUnits: workerNumericUnits,
      documentUrl, documentBaseUrl: parsed.baseUrl, documentRoot, unavailableGlobals: documentWorkerGlobals,
      resolveModule: (specifier, from) => resolveModuleSpecifier(specifier, from, {}, {
        packageRootUrl: options.packageRootUrl, _moduleSources: moduleSources, documentRoot})});
    const context = {emitFile(asset) {
      const previous = targetFiles.get(asset.fileName);
      if (previous !== undefined && !Buffer.from(previous).equals(Buffer.from(asset.source)))
        throw new Error(`Worker resource bytes disagree for ${JSON.stringify(asset.fileName)}`);
      targetFiles.set(asset.fileName, asset.source); return asset.fileName;
    }};
    plugin.buildStart();
    for (const {tag, attrs, source, file, original} of scripts) {
      // Preserve an original SRI failure instead of turning it into a working
      // script while rewriting its URLs. Supported digests use their strongest
      // algorithm, matching the browser's choice among multiple metadata tokens.
      const tokens = (file ? attrs.integrity ?? "" : "").trim().split(/\s+/).filter(Boolean);
      const known = tokens.map(token => /^(sha(?:256|384|512))-([^?]+)(?:\?.*)?$/.exec(token)).filter(Boolean);
      const strongest = Math.max(0, ...known.map(token => Number(token[1].slice(3))));
      if (strongest && !known.some(token => Number(token[1].slice(3)) === strongest &&
          crypto.createHash(token[1]).update(original).digest("base64") === token[2])) continue;
      const transformed = plugin.transform.call(context, source, file ? pathToFileURL(file).href : documentUrl);
      if (!transformed) continue;
      if (file) {
        classicWorkerSources.set(file, {before: Buffer.from(source), after: Buffer.from(transformed.code)});
        if (known.length) {
          const metadata = tokens.map(token => /^(sha(?:256|384|512))-/.test(token)
            ? computeIntegrityForContent(token, transformed.code) : token).join(" ");
          const updated = tag.attributes.replace(/(?:^|\s+)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?/g,
            (attribute, name) => name.toLowerCase() === "integrity"
              ? attribute.replace(/integrity[\s\S]*/i, `integrity="${metadata}"`) : attribute);
          classicScriptRewrites.set(tag.text, `<script${updated}>${tag.body}</script>`);
        }
      } else classicScriptRewrites.set(tag.text, `<script${tag.attributes}>${transformed.code}</script>`);
    }
  }

  // The same closure walk runs before bundling to establish retained module identity,
  // then afterward to validate chunk collisions and collect unbundled preloads.
  function collectHtmlAssets(preloadChunkMap, emittedChunkNames, skipModulePreloads = false) {
    if (!isHtml) return;

    const rawHtmlContent = fs.readFileSync(resolvedEntryAbs, "utf-8");
    const entryBaseUrl = pathToFileURL(resolvedEntryAbs).href;
    const parsedHtml = parseHtmlEntries(rawHtmlContent, entryBaseUrl, {documentRoot});
    const importMap = parsedHtml.importMap;
    const { effectiveBaseDir } = resolveBaseDetails(parsedHtml.baseHref, entryDir, entryBaseUrl, documentRoot);
    const importedStrings = createImportedStringResolver({
      readModule: url => url.startsWith("file:") && ![".json", ".css"].includes(path.extname(urlToFilePath(url)).toLowerCase()) &&
        commonjsFormat(url, moduleSources) === "module" ? moduleSources.read(url).code : null,
      resolveModule: (specifier, from) => resolveModuleSpecifier(specifier, from, importMap, {
        target, mapBaseUrl: parsedHtml.baseUrl, documentRoot,
        _moduleSources: moduleSources, packageRootUrl: options.packageRootUrl,
      }),
    });
    const copyFetchResources = (source, sourceType, moduleUrl = parsedHtml.baseUrl) => {
      if (!source.includes("fetch")) return;
      const {sites} = analyzeDocumentFetch(source, {sourceType,
        resolveImportedString: sourceType === "module" ? importedStrings(moduleUrl) : null});
      for (const [name, asset] of collectDocumentFetchAssets(sites, {
        documentUrl: entryBaseUrl, documentBaseUrl: parsedHtml.baseUrl, documentRoot,
      })) {
        if (emittedChunkNames.has(name))
          throw new Error(`Document fetch resource ${JSON.stringify(name)} collides with an emitted chunk or HTML entry`);
        const previous = targetFiles.get(name);
        if (previous !== undefined && !Buffer.from(previous).equals(asset.source))
          throw new Error(`Document fetch bytes disagree with retained resource ${JSON.stringify(name)}`);
        targetFiles.set(name, asset.source);
      }
    };

    // Extract relative assets referenced by the HTML (stylesheets, images, media, non-module scripts)
    // Excludes modulepreloads that map to bundled chunks
    const relativeAssetUrls = extractRelativeAssetUrls(
      rawHtmlContent,
      preloadChunkMap,
      effectiveBaseDir,
      skipModulePreloads,
      documentRoot,
    );

    // Bounded asset processing queue: handles direct HTML assets, transitive CSS/model children,
    // and literal dynamic imports in retained classic scripts and modules.
    const assetQueue = [];
    let runtimeTargetsQueued = false;
    const retainRuntimeTargets = () => {
      if (runtimeTargetsQueued || !nativeMapRouting) return;
      runtimeTargetsQueued = true;
      hasRuntimeImports = true;
      // Exact import-map addresses are finite even when the selected name is
      // supplied by a user or live data. Prefix mappings remain open routes.
      for (const entries of [importMap.imports, ...Object.values(importMap.scopes ?? {})]) {
        if (!entries || typeof entries !== "object") continue;
        const keys = new Set(Object.keys(entries).map(key => normalizeSpecifierKey(key, parsedHtml.baseUrl, documentRoot)));
        for (const key of keys) {
          if (key === null || key.endsWith("/")) continue;
          let url;
          try {
            url = resolveModuleSpecifier(key, parsedHtml.baseUrl, {imports: entries}, {
              mapBaseUrl: parsedHtml.baseUrl, documentRoot, _moduleSources: moduleSources, allowCommonJS: true});
          } catch (error) {
            if (error.code?.startsWith("F3D_DOCUMENT_") || error.code === "F3D_SOURCE_TRANSFORM") throw error;
            // Blocked, invalid and missing targets must still reject at the
            // native import() boundary, rather than failing an unused branch.
            continue;
          }
          if (url.startsWith("file:")) assetQueue.push({specifier: key, resolvedUrl: url,
            referrerUrl: parsedHtml.baseUrl, isModuleSpecifier: true, runtimeModule: true});
        }
      }
    };
    for (const url of runtimeModuleUrls) assetQueue.push({specifier: url, resolvedUrl: url,
      referrerUrl: parsedHtml.baseUrl, isModuleSpecifier: true, runtimeModule: true});
    for (const [id, source] of retainedHtmlModules) {
      const script = parsedHtml.moduleScripts.find(script => script.id === id);
      if (script?.inlineContent === null) continue;
      const {moduleRequests, assetSpecifiers} = extractJsModuleDependencies(source, id);
      for (const {specifier, moduleType} of moduleRequests) assetQueue.push({specifier, moduleType,
        referrerUrl: parsedHtml.baseUrl, isModuleSpecifier: true, runtimeModule: true});
      for (const relUrl of assetSpecifiers) assetQueue.push({relUrl, referrerDir: effectiveBaseDir,
        referrerPath: resolvedEntryAbs, runtimeModule: true});
      copyFetchResources(source, "module");
    }
    if (runtimeModuleUrls.size || retainedHtmlModules.size) retainRuntimeTargets();
    for (const relUrl of relativeAssetUrls) {
      assetQueue.push({
        relUrl,
        referrerDir: effectiveBaseDir,
        referrerPath: resolvedEntryAbs,
        isModuleSpecifier: false,
      });
    }

    // Scan inline classic scripts in HTML for static literal dynamic imports
    for (const tag of htmlContextTags(rawHtmlContent)) {
      if (tag.name !== "script" || !tag.closed || tag.closing) continue;
      const attrs = parseTagAttributes(tag.attributes);
      if (attrs.src !== undefined || !isClassicJavaScriptType(attrs.type)) {
        continue;
      }
      const scriptBody = tag.body;
      if (!scriptBody || !scriptBody.trim()) continue;
      copyFetchResources(scriptBody, "script");

      const { moduleRequests, assetSpecifiers, runtimeImports } = extractJsModuleDependencies(
        scriptBody,
        parsedHtml.baseUrl,
      );
      if (runtimeImports) retainRuntimeTargets();
      for (const {specifier, moduleType} of moduleRequests) {
        assetQueue.push({
          specifier,
          moduleType,
          referrerUrl: parsedHtml.baseUrl,
          referrerDir: effectiveBaseDir,
          referrerPath: resolvedEntryAbs,
          isModuleSpecifier: true,
        });
      }
      for (const assetSpec of assetSpecifiers) {
        assetQueue.push({
          relUrl: assetSpec,
          referrerDir: effectiveBaseDir,
          referrerPath: resolvedEntryAbs,
          isModuleSpecifier: false,
        });
      }
    }

    const visitedCssPaths = new Set();
    const visitedJsUrls = new Set();
    const visitedModelPaths = new Set();
    const runtimeFiles = new Set();
    let runtimeBytes = 0;

    while (assetQueue.length > 0) {
      const item = assetQueue.shift();
      let srcAssetAbs;
      let relFromEntryDir;
      let moduleUrl;
      const referrerPath = item.referrerPath || resolvedEntryAbs;

      if (item.isModuleSpecifier) {
        if (!item.specifier || typeof item.specifier !== "string") continue;
        const trimmedSpecifier = item.specifier.trim();
        if (!trimmedSpecifier) continue;

        let resolvedUrl;
        try {
          resolvedUrl = item.resolvedUrl ?? resolveModuleSpecifier(trimmedSpecifier, item.referrerUrl, importMap, {
            mapBaseUrl: parsedHtml.baseUrl,
            _moduleSources: moduleSources,
            packageRootUrl: options.packageRootUrl,
            documentRoot,
            allowCommonJS: item.runtimeModule === true,
          });
        } catch (err) {
          if (err.code === "F3D_SOURCE_TRANSFORM") throw err;
          if (item.runtimeModule && !err.code?.startsWith("F3D_DOCUMENT_")) continue;
          let context = "";
          if (referrerPath !== resolvedEntryAbs) {
            context = referrerPath.endsWith(".css")
              ? ` in CSS referenced from "${referrerPath}"`
              : ` referenced from "${referrerPath}"`;
          }
          throw new Error(
            `Unresolved module specifier${context}: "${trimmedSpecifier}" (${err.message})`,
          );
        }

        if (!resolvedUrl.startsWith("file://")) {
          // Non-file URL (e.g. http://, https:, data:); resolved at runtime by browser
          continue;
        }

        srcAssetAbs = urlToFilePath(resolvedUrl);
        relFromEntryDir = documentRoot ? documentRelativePath(srcAssetAbs, documentRoot) : path.relative(entryDir, srcAssetAbs);
        moduleUrl = resolvedUrl;
        // File copies share bytes, but query/fragment variants remain distinct modules.
        const entryDirectoryUrl = pathToFileURL(applicationRoot + path.sep).href;
        retainedModuleUrls.set(resolvedUrl, "./" + resolvedUrl.slice(entryDirectoryUrl.length));
      } else {
        const { relUrl, referrerDir } = item;
        if (!relUrl || typeof relUrl !== "string") continue;
        const trimmedRelUrl = relUrl.trim();
        if (!trimmedRelUrl || documentRoot && !isDocumentResourceUrl(trimmedRelUrl, documentRoot)) continue;

        // Browser URI semantics: resolve against referrer directory URL using new URL,
        // then convert file: URL object directly to decoded filesystem path via fileURLToPath.
        // Preserves %20 and other percent-encodings as actual filename bytes on disk.
        // Wrapped in try/catch to safely handle asset paths containing unencoded literal '%'
        // not followed by two hex digits (e.g. <img src="./100%_sale.png">).
        try {
          const referrerDirSlash = referrerDir.endsWith(path.sep)
            ? referrerDir
            : referrerDir + path.sep;
          const referrerBaseUrl = pathToFileURL(referrerDirSlash);
          const resolvedUrl = resolveDocumentUrl(trimmedRelUrl, referrerBaseUrl.href, documentRoot);
          srcAssetAbs = fileURLToPath(resolvedUrl);
        } catch {
          const cleanRelPath = trimmedRelUrl.split(/[?#]/)[0];
          srcAssetAbs = documentRoot && cleanRelPath.startsWith("/")
            ? path.resolve(documentRoot, cleanRelPath.slice(1)) : path.resolve(referrerDir, cleanRelPath);
        }

        relFromEntryDir = documentRoot ? documentRelativePath(srcAssetAbs, documentRoot) : path.relative(entryDir, srcAssetAbs);
      }

      // Verify that a copied resource stays within the selected application root.
      if (relFromEntryDir.startsWith("..") || path.isAbsolute(relFromEntryDir)) {
        const target = item.isModuleSpecifier ? item.specifier : item.relUrl;
        let context = "";
        if (referrerPath !== resolvedEntryAbs) {
          context = referrerPath.endsWith(".css")
            ? ` in CSS "${referrerPath}"`
            : ` in "${referrerPath}"`;
        }
        throw new Error(
          `Relative resource${context} escapes application root directory: "${target}"`,
        );
      }

      // Pre-emission collision check: a classic script, stylesheet, module, or asset must NOT collide
      // with or silently overwrite/shadow an emitted chunk or the entry HTML file.
      if (emittedChunkNames.has(relFromEntryDir)) {
        const target = item.isModuleSpecifier ? item.specifier : item.relUrl;
        let context = "";
        if (referrerPath !== resolvedEntryAbs) {
          context = referrerPath.endsWith(".css")
            ? ` referenced from "${referrerPath}"`
            : ` referenced from "${referrerPath}"`;
        }
        throw new Error(
          `Collision detected: relative resource "${target}"${context} collides with emitted bundle chunk or entry file "${relFromEntryDir}". Source assets must not collide with emitted chunk names.`,
        );
      }

      // Explicitly reject unresolved relative resources before output
      if (!fs.existsSync(srcAssetAbs) || !fs.statSync(srcAssetAbs).isFile()) {
        if (item.runtimeModule) continue;
        const target = item.isModuleSpecifier ? item.specifier : item.relUrl;
        let context = "";
        if (referrerPath !== resolvedEntryAbs) {
          context = referrerPath.endsWith(".css")
            ? ` in CSS referenced from "${referrerPath}"`
            : ` referenced from "${referrerPath}"`;
        }
        throw new Error(
          `Unresolved relative resource${context}: "${target}" not found at "${srcAssetAbs}"`,
        );
      }

      if (item.runtimeModule && !runtimeFiles.has(srcAssetAbs)) {
        documentRelativePath(srcAssetAbs, applicationRoot);
        if (runtimeFiles.size >= 4096 || (runtimeBytes += fs.statSync(srcAssetAbs).size) > 256 * 1024 * 1024)
          throw new Error("F3D_RUNTIME_IMPORT_LIMIT: retained import-map closure exceeds its file/byte bound");
        runtimeFiles.add(srcAssetAbs);
      }

      const javascript = !item.modelResource && !["json", "css"].includes(item.moduleType) &&
        ![".json", ".css"].includes(path.extname(srcAssetAbs).toLowerCase()) &&
        (item.isModuleSpecifier || /\.(?:mjs|cjs|js)$/.test(srcAssetAbs));
      // Retained scripts must use the same frontend output as bundled imports.
      // URL variants can share a physical output only when their bytes agree.
      if (javascript && item.isModuleSpecifier && !item.runtimeModule) assertUnbundledModule(moduleUrl, moduleSources);
      let assetData = javascript ? moduleSources.read(moduleUrl || pathToFileURL(srcAssetAbs).href,
        {sourceType: item.isModuleSpecifier ? "module" : "script"}).bytes : fs.readFileSync(srcAssetAbs);
      const workerSource = classicWorkerSources.get(srcAssetAbs);
      if (!item.modelResource && workerSource && assetData.equals(workerSource.before)) assetData = workerSource.after;
      const previous = targetFiles.get(relFromEntryDir);
      if (previous !== undefined && !Buffer.from(previous).equals(assetData))
        throw new Error(`Source transforms disagree for retained resource ${JSON.stringify(relFromEntryDir)}`);
      targetFiles.set(relFromEntryDir, assetData);

      // Retained modules bypass the bundler's model emitter. Their models still
      // need every core buffer/image URI, resolved from the model rather than
      // the referring module. Keep model and sidecar bytes and URI strings intact.
      const modelExtension = path.extname(srcAssetAbs).toLowerCase();
      if (!javascript && !item.modelResource && [".gltf", ".glb"].includes(modelExtension) &&
          !visitedModelPaths.has(srcAssetAbs)) {
        visitedModelPaths.add(srcAssetAbs);
        const fail = (code, message) => {
          const error = new Error(`${code}: ${message} (model ${JSON.stringify(srcAssetAbs)})`);
          error.code = code;
          throw error;
        };
        for (const uri of gltfAssetUris(assetData, modelExtension === ".glb", fail)) {
          if (!isDocumentResourceUrl(uri, documentRoot)) continue;
          assetQueue.push({
            relUrl: uri,
            referrerDir: path.dirname(srcAssetAbs),
            referrerPath: srcAssetAbs,
            isModuleSpecifier: false,
            modelResource: true,
            runtimeModule: item.runtimeModule,
          });
        }
      }

      // A CSS module keeps its native constructed-sheet behavior: @import is
      // never a fetched dependency. A separate ordinary stylesheet use of the
      // same URL still traverses its complete stylesheet graph.
      const cssIdentity = `${item.isModuleSpecifier ? "module:" : "stylesheet:"}${srcAssetAbs}`;
      const cssGoal = !item.isModuleSpecifier || !item.moduleType || ["css", "unknown"].includes(item.moduleType);
      if (!item.modelResource && cssGoal && path.extname(srcAssetAbs).toLowerCase() === ".css" && !visitedCssPaths.has(cssIdentity)) {
        visitedCssPaths.add(cssIdentity);
        const cssContent = assetData.toString("utf-8");
        const childUrls = extractRelativeCssUrls(cssContent, documentRoot, {ignoreImports: item.isModuleSpecifier === true});
        const cssDir = path.dirname(srcAssetAbs);
        for (const childUrl of childUrls) {
          assetQueue.push({
            relUrl: childUrl,
            referrerDir: cssDir,
            referrerPath: srcAssetAbs,
            isModuleSpecifier: false,
            runtimeModule: item.runtimeModule,
          });
        }
      }

      // If resource is a module or JS/MJS/CJS file (retained classic script),
      // scan for transitive module dependencies (static/dynamic imports, exports) and asset references
      if (
        javascript &&
        !visitedJsUrls.has(moduleUrl || pathToFileURL(srcAssetAbs).href)
      ) {
        const fileUrl = moduleUrl || pathToFileURL(srcAssetAbs).href;
        visitedJsUrls.add(fileUrl);
        const jsContent = assetData.toString("utf-8");
        let dependencies;
        try { dependencies = extractJsModuleDependencies(jsContent, fileUrl); }
        catch (error) {
          if (!item.runtimeModule || !(error instanceof IngestionParseError)) throw error;
          continue; // Preserve a lazy module's native syntax rejection.
        }
        copyFetchResources(jsContent, item.isModuleSpecifier ? "module" : "script", fileUrl);
        const {moduleRequests, assetSpecifiers, runtimeImports} = dependencies;
        if (runtimeImports) retainRuntimeTargets();
        const jsDir = path.dirname(srcAssetAbs);

        for (const {specifier, moduleType} of moduleRequests) {
          assetQueue.push({
            specifier,
            moduleType,
            referrerUrl: fileUrl,
            referrerDir: jsDir,
            referrerPath: srcAssetAbs,
            isModuleSpecifier: true,
            runtimeModule: item.runtimeModule,
          });
        }

        for (const assetSpec of assetSpecifiers) {
          assetQueue.push({
            relUrl: assetSpec,
            referrerDir: jsDir,
            referrerPath: srcAssetAbs,
            isModuleSpecifier: false,
            runtimeModule: item.runtimeModule,
          });
        }
      }
    }
  }

  collectHtmlAssets(null, new Set(htmlFileName ? [htmlFileName] : []), true);
  if (isHtml && hasRuntimeImports) {
    const parsed = parseHtmlEntries(fs.readFileSync(resolvedEntryAbs, "utf8"), pathToFileURL(resolvedEntryAbs).href, {documentRoot});
    const { modules } = await getModuleGraph();
    const nodes = new Map(Object.values(modules).map(node => [node.source_url, node]));
    const incoming = new Map(), children = new Map();
    for (const node of nodes.values()) {
      const targets = [...node.static_imports, ...node.dynamic_imports.flatMap(request => request.resolved_targets ?? [request])]
        .map(request => request.resolved_url).filter(Boolean);
      children.set(node.source_url, targets);
      for (const url of targets) {
        if (!incoming.has(url)) incoming.set(url, new Set());
        incoming.get(url).add(node.source_url);
      }
    }
    const entries = [parsed.importMap.imports, ...Object.values(parsed.importMap.scopes ?? {})].filter(Boolean);
    const remapped = new Map();
    const needsOriginalImport = url => {
      if (!remapped.has(url)) {
        // A generated canonical URL can match a second map key after an alias
        // was already resolved. Output chunks also have different scopes, so
        // conservatively keep the source importer if any scope can remap it.
        try { remapped.set(url, entries.some(imports => resolveModuleSpecifier(url, parsed.baseUrl, {imports}, {
          mapBaseUrl: parsed.baseUrl, documentRoot, _moduleSources: moduleSources, allowCommonJS: true,
          packageRootUrl: options.packageRootUrl,
        }) !== url)); } catch (error) {
          if (error.code === "F3D_SOURCE_TRANSFORM") throw error;
          remapped.set(url, true);
        }
      }
      return remapped.get(url);
    };
    const pending = [...retainedModuleUrls.keys(), ...retainedHtmlModules.keys()], visited = new Set();
    let expanded = false;
    while (pending.length) {
      const url = pending.pop();
      if (visited.has(url)) continue;
      visited.add(url);
      const node = nodes.get(url);
      if (node) {
        if (node.is_inline && !retainedHtmlModules.has(url)) {
          const script = parsed.moduleScripts.find(script => script.id === node.id);
          retainedHtmlModules.set(url, moduleSources.read(url, {source: script.inlineContent}).code);
          expanded = true;
        } else if (!node.is_inline && !runtimeModuleUrls.has(url)) {
          runtimeModuleUrls.add(url); expanded = true;
        }
        // Reuse proved graph edges too: imported finite selectors can name
        // dependencies that a source-local retained-resource scan cannot see.
        pending.push(...children.get(url));
      }
      if (needsOriginalImport(url)) pending.push(...incoming.get(url) ?? []);
    }
    if (expanded) collectHtmlAssets(null, new Set(htmlFileName ? [htmlFileName] : []), true);
    for (const script of parsed.moduleScripts) if (script.inlineContent === null && retainedModuleUrls.has(script.id)) {
      retainedHtmlModules.set(script.id, moduleSources.read(script.id).code);
      if (moduleSources.changed(script.id)) changedRetainedModules.add(script.id);
    }
  }
  const rendererRoute =
    options.routeWebGPURenderer || options.routeWebGLRenderer
      ? decideRendererRoutes(await getModuleGraph(), resolvedEntryAbs, options.packageRootUrl, {
          webgpu: Boolean(options.routeWebGPURenderer),
          webgl: Boolean(options.routeWebGLRenderer),
        }, moduleSources, documentRoot)
      : null;
  const bundleResult = await bundleWithRollup(resolvedEntryAbs, {
    target,
    _moduleSources: moduleSources,
    packageRootUrl: options.packageRootUrl,
    documentRoot,
    workerHostBindings: documentWorkerGlobals,
    retainedModuleUrls,
    retainedHtmlModules,
    specializeNumeric: options.specializeNumeric,
    _workerNumericUnits: workerNumericUnits,
    prepareStandaloneLoaders: options.prepareStandaloneLoaders,
    networkedImports: options.networkedImports,
    ...(rendererRoute?.routed ? { rendererRoute: { targets: rendererRoute.targets, exactSites: rendererRoute.exactSites,
      postprocessingTransforms: rendererRoute.postprocessingTransforms,
      gpuComputationTransforms: rendererRoute.gpuComputationTransforms } } : {}),
  });
  if (rendererRoute?.routed && bundleResult.routedRendererModules.length === 0) {
    throw new Error("A renderer route was decided but no pinned three build import was substituted");
  }
  const emittedChunkNames = new Set(bundleResult.chunks.map((chunk) => chunk.fileName));
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
    const existing = targetFiles.get(fileName);
    if (existing !== undefined && !Buffer.from(existing).equals(Buffer.from(code))) {
      throw new Error(`Collision detected: relative resource ${JSON.stringify(fileName)} differs from an emitted bundle asset`);
    }
    targetFiles.set(fileName, code);
  }
  if (isHtml) {
    const entryBaseUrl = pathToFileURL(resolvedEntryAbs).href;
    targetFiles.set(
      htmlFileName,
      rewriteHtmlForBuild(
        fs.readFileSync(resolvedEntryAbs, "utf-8"),
        bundleResult.entryFiles,
        bundleResult.files,
        { preloadChunkMap, entryDir, documentUrl: entryBaseUrl, documentRoot, classicScriptRewrites,
          retainedHtmlModules, changedRetainedModules, publicImports: bundleResult.publicImports },
      ),
    );
  }

  // Safe collision check: fail safely without deleting or overwriting (including symlinks)
  for (const relPath of targetFiles.keys()) {
    const destPath = path.join(resolvedOutDir, relPath);
    if (pathExistsOrSymlink(destPath)) {
      throw new Error(
        `Refusing to overwrite existing destination file: ${destPath}. Destination must be fresh; collisions are rejected.`,
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
      fs.writeFileSync(destPath, content, { flag: "wx" });
    } catch (err) {
      if (err.code === "EEXIST") {
        throw new Error(
          `Refusing to overwrite existing destination file: ${destPath}. Destination must be fresh; collisions are rejected.`,
        );
      }
      throw err;
    }
    emittedFiles.push(relPath);
  }

  return {
    target,
    entryPoint: resolvedEntryAbs,
    outDir: resolvedOutDir,
    isHtml,
    htmlFile: htmlFileName,
    entryFiles: bundleResult.entryFiles,
    emittedFiles,
    isMultiChunk: bundleResult.isMultiChunk,
    chunks: bundleResult.chunks,
    packageType: "module",
    ...(bundleResult.numericSpecialization
      ? { numericSpecialization: bundleResult.numericSpecialization }
      : {}),
    ...(rendererRoute
      ? {
          rendererRoute: {
            routed: rendererRoute.routed,
            reason: rendererRoute.reason,
            decisions: rendererRoute.decisions,
            targets: rendererRoute.targets,
            exactConstructionSites: rendererRoute.exactSites.map(({moduleId, sourceSpan}) => ({moduleId, sourceSpan})),
            substitutedModules: bundleResult.routedRendererModules ?? [],
          },
        }
      : {}),
  };
}

/**
 * Build-time renderer route decision for the opt-in general new-backend
 * renderers (three_renderer.mjs). The whole-application module graph and the
 * existing route decider choose. A renderer class is substituted only when every
 * construction site of it routes to GENERAL_WEBGPU or explicitly selects the
 * source WebGL backend, and every graph import of its build file is the admitted
 * pinned build. WebGLRenderer additionally requires that no application module
 * reads GL-context state (capabilities/extensions/context attributes/loss), which
 * this route cannot provide. Anything else keeps the retained upstream build
 * unchanged, with the reason recorded.
 */
function decideRendererRoutes(graph, entryAbs, packageRootUrl, enabled, moduleSources, documentRoot) {
  let decisions = evaluateGraphRoutes(graph, decideRendererRoute, {
    packageRootUrl,
    generalWebGPUAvailable: enabled.webgpu,
    generalWebGLAvailable: enabled.webgl,
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
  }).map(({ moduleId, constructorName, decision }) => ({
    moduleId,
    constructorName,
    route: decision.route,
    reasons: [...decision.reasons],
    sourceSpan: decision.sourceSpan,
  }));
  const fileIds = Object.keys(graph.modules).filter((id) => id.startsWith("file:"));
  const buildFile = (name) => fileIds.filter((id) => new URL(id).pathname.endsWith(`/build/${name}`));
  const candidates = [
    enabled.webgpu && { exportName: "WebGPURenderer", urls: buildFile("three.webgpu.js") },
    enabled.webgl && { exportName: "WebGLRenderer", urls: buildFile("three.module.js") },
  ].filter(Boolean);
  const targets = [];
  const reasons = [];
  // A frontend may change a library as well as application code. The source
  // path alone cannot authorize replacing that changed implementation. Keep
  // the actual normalized library on its ordinary route, including core edits.
  const changedThree = fileIds.some(id => isInternalLibraryModule(id, packageRootUrl) && moduleSources.changed(id));
  // An explicit source selection can coexist with native sites: the bundler
  // preserves its original constructor call without reading any options first.
  // A native site is still required to prove the global escape/closure guards.
  const sourceSelected = decision => decision.constructorName === "WebGPURenderer" &&
    decision.route === "exact-backend" && decision.reasons.length === 1 &&
    decision.reasons[0] === "explicit-source-selection";
  // A separate exact renderer need not force a closed native renderer onto its
  // backend. Only a proved local allocation graph may replace that global veto;
  // shared or unknown resource ownership continues to choose the source route.
  const mixed = decisions.some(decision => decision.route === 'general-webgpu') &&
    decisions.some(decision => decision.route === 'exact-backend');
  const grouped = mixed ? resolveClosedRendererGroups(graph, moduleSources, decisions, packageRootUrl) : null;
  if (grouped) decisions = grouped;
  const keepSource = decision => sourceSelected(decision) || Boolean(grouped &&
    decision.constructorName === 'WebGPURenderer' && decision.route === 'exact-backend');
  for (const { exportName, urls } of candidates) {
    const gl = exportName === "WebGLRenderer";
    const sites = decisions.filter((d) => d.constructorName === exportName);
    let reason = null;
    if (mixed && !grouped) reason = 'mixed-renderer-resource-ownership-unproved';
    else if (sites.length === 0) reason = gl ? "no-webgl-renderer-construction" : "no-webgpu-renderer-construction";
    else if (!sites.some((d) => d.route === "general-webgpu") ||
        sites.some((d) => d.route !== "general-webgpu" && (gl || !keepSource(d))))
      reason = "construction-site-not-general-webgpu";
    else if (!grouped && decisions.some((d) => d.route === "exact-backend" && (gl || !sourceSelected(d))))
      reason = "exact-backend-renderer-present";
    else if (urls.length === 0) reason = gl ? "no-three-module-import" : "no-three-webgpu-import";
    else if (!urls.every((id) => isInternalLibraryModule(id, packageRootUrl)))
      reason = gl ? "unpinned-three-module" : "unpinned-three-webgpu-module";
    else if (changedThree) reason = "transformed-three-module";
    else if (gl && readsGlContextState(graph, entryAbs, packageRootUrl, moduleSources, documentRoot)) reason = "gl-context-state-read";
    if (reason) reasons.push(reason);
    else for (const url of urls) targets.push({ url, exportName });
  }
  const exactSites = targets.some(target => target.exportName === "WebGPURenderer")
    ? decisions.filter(keepSource).map(decision => {
      const module = graph.modules[decision.moduleId];
      const siteIndex = module.renderer_construction_sites.findIndex(site =>
        site.source_span.start.offset === decision.sourceSpan.start.offset &&
        site.source_span.end.offset === decision.sourceSpan.end.offset);
      if (siteIndex < 0) throw new Error("Source-selected renderer construction lost its original site");
      return {moduleId: decision.moduleId, sourceSpan: decision.sourceSpan, siteIndex, sourceHash: module.content_hash,
        ...(grouped && !sourceSelected(decision) ? {connectedGroup: decision.groupId} : {})};
    }) : [];
  const postprocessingTransforms = targets.some(target => target.exportName === 'WebGLRenderer')
    ? Object.entries(graph.modules).filter(([, mod]) => mod.postprocessing_transform)
      .map(([id, mod]) => ({id, ...mod.postprocessing_transform})) : [];
  const gpuComputationTransforms = targets.some(target => target.exportName === 'WebGLRenderer')
    ? Object.entries(graph.modules).filter(([, mod]) => mod.gpu_computation_transform)
      .map(([id, mod]) => ({id, ...mod.gpu_computation_transform})) : [];
  return { routed: targets.length > 0, reason: targets.length ? "routed" : reasons.join(","), decisions, targets, exactSites,
    postprocessingTransforms, gpuComputationTransforms };
}

const GL_CONTEXT_STATE = new Set(["capabilities", "extensions", "getContextAttributes", "forceContextLoss", "forceContextRestore"]);
// Fields the routed WebGLRenderer serves truthfully (three_renderer.mjs).
const SERVED_CAPABILITIES = new Set(["getMaxAnisotropy", "logarithmicDepthBuffer", "reversedDepthBuffer", "precision", "getMaxPrecision", "isWebGL2"]);
/** True when an application (non-pinned-library) module reads GL-context state. */
export function readsGlContextState(graph, entryAbs, packageRootUrl, moduleSources, documentRoot) {
  const propertyNames = node => {
    if (!node.computed) return [node.property.name];
    // Reuse finite string analysis without evaluating keys or their effects.
    // Brackets and optional access expose the same source contracts as dots.
    const key = classifyDynamicImportArgument(node.property);
    return key.classification === "literal" ? [key.specifier] : key.candidates;
  };
  const inline = new Map();
  if (/\.html?$/.test(entryAbs)) {
    const entryUrl = pathToFileURL(entryAbs).href;
    for (const script of parseHtmlEntries(fs.readFileSync(entryAbs, "utf-8"), entryUrl, {documentRoot}).moduleScripts)
      if (script.inlineContent !== null) inline.set(script.id, script.inlineContent);
  }
  for (const id of Object.keys(graph.modules)) {
    if (!id.startsWith("file:") || isInternalLibraryModule(id, packageRootUrl)) continue;
    // Pinned upstream addons are library code that receives the renderer
    // explicitly; their own GL-specific paths are guarded inside upstream.
    if (/\/upstream\/three\.js\/(examples\/jsm|src)\//.test(new URL(id).pathname)) continue;
    let ast;
    try {
      let source = moduleSources.read(id, {source: inline.get(id)}).code;
      if (graph.modules[id].gpu_computation_transform)
        source = transformThreeGpuComputation(source, graph.modules[id].gpu_computation_transform);
      ast = acorn.parse(source, { ecmaVersion: "latest", sourceType: "module" });
    } catch {
      return true; // Unparseable application code cannot be cleared.
    }
    let found = false;
    walk.ancestor(ast, {
      MemberExpression(node, _state, ancestors) {
        const names = propertyNames(node);
        if (!names?.some(name => GL_CONTEXT_STATE.has(name))) return;
        // renderer.capabilities.<served field> describes this route, not a GL context.
        const parent = ancestors.at(-2);
        if (names.every(name => name === "capabilities") && parent?.type === "MemberExpression" && parent.object === node &&
            propertyNames(parent)?.every(name => SERVED_CAPABILITIES.has(name))) return;
        found = true;
      },
    });
    if (found) return true;
  }
  return false;
}
