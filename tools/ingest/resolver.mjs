/**
 * Module Specifier Resolver.
 * Implements W3C Import Map resolution, relative URL resolution,
 * pinned Three.js r186 and installed ESM package resolution, and canonical URL identity.
 *
 * Preserves URL queries (?query) and fragments (#hash) for module instance identity.
 * Strictly respects W3C null mappings (blocked imports) and trailing slash prefix contracts.
 * Supports explicit absolute file://, http://, and https:// URLs.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { IngestionResolutionError } from "./types.mjs";
import { resolvePackageExports } from "./package_exports.mjs";
import { resolveInstalledPackage, resolveInstalledPackageIfPresent, resolvePackageBrowserMapping, resolveCommonJSFile } from "./node_package_resolver.mjs";
import { normalizeDocumentRoot, resolveDocumentUrl, documentRelativePath, isRootRelativeDocumentUrl } from "./document_urls.mjs";
export { resolveExportTargetValue, resolvePackageExports } from "./package_exports.mjs";

/**
 * Normalizes a base URL (strips fragment like #inline-module-1).
 * @param {string} url
 * @returns {string}
 */
export function getBaseUrl(url) {
  const hashIdx = url.indexOf("#");
  return hashIdx !== -1 ? url.slice(0, hashIdx) : url;
}

/**
 * Extracts the physical file system path from a file:// URL, stripping query and fragment.
 * @param {string} url
 * @returns {string}
 */
export function urlToFilePath(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== "file:") {
    throw new Error(`Cannot convert non-file URL "${url}" to a file system path`);
  }
  return fileURLToPath(parsed);
}

/**
 * Checks if a string is a valid absolute URL with an admitted scheme.
 * @param {string} specifier
 * @returns {boolean}
 */
export function isAbsoluteUrl(specifier) {
  try {
    const parsed = new URL(specifier);
    return ["file:", "http:", "https:", "data:"].includes(parsed.protocol);
  } catch {
    return false;
  }
}

/**
 * Matches an import map given specifier and referrer.
 * Returns:
 * - string: resolved target URL
 * - null: no match
 * - { blocked: true }: explicitly mapped to null (blocked import)
 * - { error: string }: invalid prefix contract
 *
 * @param {string} specifier
 * @param {string} referrerUrl
 * @param {{ imports?: Record<string, string | null>, scopes?: Record<string, Record<string, string | null>> }} importMap
 * @param {string} mapBaseUrl
 * @returns {string | { blocked: true } | { error: string } | null}
 */
function matchImportMap(specifier, referrerUrl, importMap, mapBaseUrl, documentRoot) {
  if (!importMap) return null;

  // 1. Check scopes
  if (importMap.scopes && typeof importMap.scopes === "object") {
    // WHATWG sorts serialized scope URLs, not their original relative spellings.
    const scopes = new Map();
    for (const [prefix, entries] of Object.entries(importMap.scopes)) {
      let scopeUrl;
      try {
        scopeUrl = resolveDocumentUrl(prefix, mapBaseUrl, documentRoot);
      } catch (error) {
        if (error.code?.startsWith("F3D_DOCUMENT_")) throw error;
        continue; // Unparseable scope URLs are ignored by import-map normalization.
      }
      scopes.set(scopeUrl, entries);
    }
    for (const scopeBase of [...scopes.keys()].sort().reverse()) {
      if (
        referrerUrl === scopeBase ||
        (scopeBase.endsWith("/") && referrerUrl.startsWith(scopeBase))
      ) {
        const match = matchMapEntries(specifier, referrerUrl, scopes.get(scopeBase), mapBaseUrl, documentRoot);
        if (match !== null) return match;
      }
    }
  }

  // 2. Check top-level imports
  if (importMap.imports && typeof importMap.imports === "object") {
    return matchMapEntries(specifier, referrerUrl, importMap.imports, mapBaseUrl, documentRoot);
  }

  return null;
}

/**
 * Normalizes an import map specifier key per WHATWG:
 * - Empty string returns null (ignored).
 * - Relative/pathname ("/", "./", "../") or absolute URL is resolved against mapBaseUrl.
 * - Bare specifiers return the bare key unmodified.
 * @param {string} key
 * @param {string} mapBaseUrl
 * @returns {string | null}
 */
function normalizeSpecifierKey(key, mapBaseUrl, documentRoot) {
  if (key === "") return null;
  if (key.startsWith("/") || key.startsWith("./") || key.startsWith("../")) {
    try {
      return resolveDocumentUrl(key, mapBaseUrl, documentRoot);
    } catch (error) {
      if (error.code?.startsWith("F3D_DOCUMENT_")) throw error;
      return null;
    }
  }
  try {
    return new URL(key).href;
  } catch {
    return key;
  }
}

/**
 * Normalizes a requested specifier per WHATWG:
 * - Relative/pathname ("/", "./", "../") or absolute URL is resolved against referrerUrl.
 * - Bare specifiers return the bare key unmodified.
 * @param {string} specifier
 * @param {string} referrerUrl
 * @returns {string}
 */
function normalizeRequestSpecifier(specifier, referrerUrl, documentRoot) {
  if (specifier.startsWith("/") || specifier.startsWith("./") || specifier.startsWith("../")) {
    try {
      return resolveDocumentUrl(specifier, referrerUrl, documentRoot);
    } catch (error) {
      if (error.code?.startsWith("F3D_DOCUMENT_")) throw error;
      return specifier;
    }
  }
  try {
    return new URL(specifier).href;
  } catch {
    return specifier;
  }
}

/** Resolve a matched address before checking the import-map slash contract. */
function resolveMapTarget(rawKey, target, mapBaseUrl, documentRoot) {
  if (typeof target !== "string") return { blocked: true };
  let targetUrl;
  try {
    const relative = target.startsWith("/") || target.startsWith("./") || target.startsWith("../");
    targetUrl = resolveDocumentUrl(target, relative ? mapBaseUrl : undefined, documentRoot);
  } catch (error) {
    if (error.code?.startsWith("F3D_DOCUMENT_")) throw error;
    return { blocked: true };
  }
  if (rawKey.endsWith("/") && !targetUrl.endsWith("/")) {
    return {
      error: `Invalid import map prefix mapping: target for prefix "${rawKey}" must end with "/" (got "${target}")`,
    };
  }
  return targetUrl;
}

/**
 * Matches a request specifier against import map entries per WHATWG.
 * Normalizes URL-like keys and request specifier; preserves bare specifiers.
 * Evaluates exact match then longest prefix match.
 *
 * @param {string} specifier
 * @param {string} referrerUrl
 * @param {Record<string, string | null>} entries
 * @param {string} mapBaseUrl
 * @returns {string | { blocked: true } | { error: string } | null}
 */
function matchMapEntries(specifier, referrerUrl, entries, mapBaseUrl, documentRoot) {
  const normalizedSpecifier = normalizeRequestSpecifier(specifier, referrerUrl, documentRoot);

  const normalizedEntries = new Map();
  for (const rawKey of Object.keys(entries)) {
    const normKey = normalizeSpecifierKey(rawKey, mapBaseUrl, documentRoot);
    if (normKey === null) continue;
    normalizedEntries.set(normKey, { rawKey, target: entries[rawKey] });
  }

  // 1. Exact match
  if (normalizedEntries.has(normalizedSpecifier)) {
    const { rawKey, target } = normalizedEntries.get(normalizedSpecifier);
    return resolveMapTarget(rawKey, target, mapBaseUrl, documentRoot);
  }

  // URL specifiers with non-special schemes only participate in exact matches.
  let requestUrl = null;
  try {
    requestUrl = new URL(normalizedSpecifier);
  } catch {
    // Bare specifiers still participate in prefix matching.
  }
  if (
    requestUrl &&
    !["ftp:", "file:", "http:", "https:", "ws:", "wss:"].includes(requestUrl.protocol)
  ) {
    return null;
  }

  // 2. Prefix match uses normalized keys; raw spelling only governs target validation.
  const prefixEntries = [...normalizedEntries.entries()]
    .filter(([normKey]) => normKey.endsWith("/"))
    .sort((a, b) => b[0].length - a[0].length);

  for (const [normKey, { rawKey, target }] of prefixEntries) {
    if (normalizedSpecifier.startsWith(normKey)) {
      const targetUrl = resolveMapTarget(rawKey, target, mapBaseUrl, documentRoot);
      if (typeof targetUrl !== "string") return targetUrl;
      const remainder = normalizedSpecifier.slice(normKey.length);
      let resolvedUrl;
      try {
        resolvedUrl = new URL(remainder, targetUrl).href;
      } catch {
        return {
          error: `Cannot resolve import specifier "${specifier}": suffix cannot be parsed relative to import map prefix "${rawKey}"`,
        };
      }
      if (!resolvedUrl.startsWith(targetUrl)) {
        return {
          error: `Cannot resolve import specifier "${specifier}": backtracking above import map prefix "${rawKey}"`,
        };
      }
      if (documentRoot && isRootRelativeDocumentUrl(target)) {
        const mapBase = new URL(mapBaseUrl);
        if (mapBase.protocol === "file:" && !mapBase.hostname) documentRelativePath(resolvedUrl, documentRoot);
      }
      return resolvedUrl;
    }
  }

  return null;
}

const PACKAGE_JSON_CACHE = new Map();

/**
 * Loads and parses a package.json from disk, with in-memory caching.
 * @param {string} packageJsonPath
 * @returns {any | null}
 */
export function loadPackageJson(packageJsonPath) {
  if (PACKAGE_JSON_CACHE.has(packageJsonPath)) {
    return PACKAGE_JSON_CACHE.get(packageJsonPath);
  }
  if (!fs.existsSync(packageJsonPath)) {
    return null;
  }
  try {
    const raw = fs.readFileSync(packageJsonPath, "utf-8");
    const parsed = JSON.parse(raw);
    PACKAGE_JSON_CACHE.set(packageJsonPath, parsed);
    return parsed;
  } catch {
    return null;
  }
}

/**
 * Clears the package.json cache.
 */
export function clearPackageJsonCache() {
  PACKAGE_JSON_CACHE.clear();
}

/**
 * Resolves a module specifier against a referrer and import map.
 * Preserves canonical URL identity (including query and fragment)
 * and ensures the target file exists.
 *
 * @param {string} specifier - Import specifier string
 * @param {string} referrerUrl - Canonical URL of importing module
 * @param {{ imports?: Record<string, string | null>, scopes?: Record<string, Record<string, string | null>> }} [importMap]
 * @param {Object} [options]
 * @param {string} [options.packageRootUrl] - Explicit Three.js package override; otherwise use the application's installed package before the pinned fallback
 * @param {string} [options.mapBaseUrl] - Base URL for import map resolution
 * @param {{ line: number, column: number, offset: number } | null} [options.span]
 * @returns {string} - Canonical resolved file URL
 */
export function resolveModuleSpecifier(specifier, referrerUrl, importMap = {}, options = {}) {
  const documentRoot = normalizeDocumentRoot(options.documentRoot);
  const span = options.span || null;
  const referrerBase = getBaseUrl(referrerUrl);
  const mapBaseUrl = options.mapBaseUrl || referrerBase;

  let candidateUrl = null;
  const packageOptions = {
    span, browserSkip: options.browserSkip, browserTrail: options.browserTrail,
    _moduleSources: options._moduleSources,
    requestKind: options.requestKind,
    allowCommonJS: options.allowCommonJS,
    // Replacement dependencies and #imports originate at the owner, but keep
    // the HTML map base, scoped host adapters and the pinned Three.js policy.
    resolveExternal: (target, from, state = {}) =>
      resolveModuleSpecifier(target, from, importMap, { ...options, ...state, mapBaseUrl }),
  };

  // 1. Try import map
  const mapped = matchImportMap(specifier, referrerUrl, importMap, mapBaseUrl, documentRoot);
  const threePackage = specifier === "three" || specifier.startsWith("three/");
  const browserTarget = mapped || (threePackage && options.packageRootUrl)
    ? null : resolvePackageBrowserMapping(specifier, referrerUrl, packageOptions);
  if (mapped) {
    if (typeof mapped === "object" && mapped.blocked) {
      throw new IngestionResolutionError(
        `Cannot resolve blocked import specifier "${specifier}" from "${referrerUrl}": mapped to null in import map`,
        specifier,
        referrerUrl,
        span,
      );
    }
    if (typeof mapped === "object" && mapped.error) {
      throw new IngestionResolutionError(mapped.error, specifier, referrerUrl, span);
    }
    candidateUrl = mapped;
  } else if (browserTarget !== null) {
    candidateUrl = browserTarget;
  } else if (isAbsoluteUrl(specifier)) {
    // 2. Explicit absolute URL (file://, http://, https://, data:)
    candidateUrl = new URL(specifier).href;
  } else if (
    specifier.startsWith("./") ||
    specifier.startsWith("../") ||
    specifier.startsWith("/")
  ) {
    // 3. Relative or pathname specifier
    try {
      candidateUrl = resolveDocumentUrl(specifier, referrerBase, documentRoot);
    } catch (err) {
      if (err.code?.startsWith("F3D_DOCUMENT_")) throw err;
      throw new IngestionResolutionError(
        `Failed to parse URL for relative specifier "${specifier}" from "${referrerUrl}": ${err.message}`,
        specifier,
        referrerUrl,
        span,
      );
    }
  } else if (threePackage) {
    // Preserve the application's actual installed version and package instance.
    // Explicit overrides remain deliberate pin choices; an absent installation
    // can use the compiler checkout, but a selected package's errors cannot.
    if (!options.packageRootUrl && new URL(referrerUrl).protocol === "file:") {
      const installed = resolveInstalledPackageIfPresent(specifier, referrerUrl, packageOptions);
      if (installed !== null) return installed;
    }
    // 4. Explicit or fallback Three.js package via its exports map.
    let pkgRootDir;
    let pkgRootUrl;
    if (options.packageRootUrl) {
      if (options.packageRootUrl.startsWith("file://")) {
        pkgRootUrl = options.packageRootUrl.endsWith("/")
          ? options.packageRootUrl
          : options.packageRootUrl + "/";
        pkgRootDir = urlToFilePath(pkgRootUrl);
      } else {
        pkgRootDir = path.resolve(options.packageRootUrl);
        pkgRootUrl = pathToFileURL(pkgRootDir).href + "/";
      }
    } else {
      pkgRootDir = path.resolve("upstream/three.js");
      pkgRootUrl = pathToFileURL(pkgRootDir).href + "/";
    }

    const packageJsonPath = path.join(pkgRootDir, "package.json");
    const pkgJson = loadPackageJson(packageJsonPath);
    if (!pkgJson) {
      throw new IngestionResolutionError(
        `Cannot resolve package import "${specifier}" from "${referrerUrl}": package.json not found at "${packageJsonPath}"`,
        specifier,
        referrerUrl,
        span,
      );
    }

    const subpath = specifier === "three" ? "." : "./" + specifier.slice("three/".length);
    const relativeTarget = resolvePackageExports(
      subpath,
      pkgJson.exports,
      packageJsonPath,
      specifier,
      referrerUrl,
      span,
      options.requestKind === 'require' ? ['browser', 'require', 'default'] : undefined,
    );
    candidateUrl = new URL(relativeTarget, pkgRootUrl).href;
  } else {
    // 4b. Ordinary installed dependencies; never overrides maps or the pin.
    candidateUrl = resolveInstalledPackage(specifier, referrerUrl, packageOptions);
  }

  if (options.requestKind === 'require' && !mapped && browserTarget === null &&
      candidateUrl.startsWith('file:') && /^(?:\.{1,2}\/|\/|file:)/.test(specifier)) {
    candidateUrl = resolveCommonJSFile(candidateUrl);
    if (documentRoot && isRootRelativeDocumentUrl(specifier)) {
      const referrer = new URL(referrerBase);
      if (referrer.protocol === "file:" && !referrer.hostname) documentRelativePath(candidateUrl, documentRoot);
    }
    // Browser objects commonly name './server.js' while require uses './server'.
    candidateUrl = resolvePackageBrowserMapping(candidateUrl, referrerUrl, packageOptions) ?? candidateUrl;
  }

  // 5. Verify target exists if file: URL
  if (candidateUrl.startsWith("file://")) {
    let filePath;
    try {
      filePath = urlToFilePath(candidateUrl);
    } catch (err) {
      throw new IngestionResolutionError(
        `Invalid file URL "${candidateUrl}" for specifier "${specifier}": ${err.message}`,
        specifier,
        referrerUrl,
        span,
      );
    }

    if (!fs.existsSync(filePath)) {
      throw new IngestionResolutionError(
        `Cannot resolve import specifier "${specifier}" from "${referrerUrl}": target file "${filePath}" does not exist`,
        specifier,
        referrerUrl,
        span,
      );
    }

    // Preserve normalized URL identity (including query and fragment)
    return new URL(candidateUrl).href;
  }

  return candidateUrl;
}
