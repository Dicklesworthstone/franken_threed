/**
 * Module Specifier Resolver.
 * Implements W3C Import Map resolution, relative URL resolution,
 * package exports fallback for Three.js r186, and canonical URL identity.
 *
 * Preserves URL queries (?query) and fragments (#hash) for module instance identity.
 * Strictly respects W3C null mappings (blocked imports) and trailing slash prefix contracts.
 * Supports explicit absolute file://, http://, and https:// URLs.
 */

import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { IngestionResolutionError } from './types.mjs';

/**
 * Normalizes a base URL (strips fragment like #inline-module-1).
 * @param {string} url
 * @returns {string}
 */
export function getBaseUrl(url) {
  const hashIdx = url.indexOf('#');
  return hashIdx !== -1 ? url.slice(0, hashIdx) : url;
}

/**
 * Extracts the physical file system path from a file:// URL, stripping query and fragment.
 * @param {string} url
 * @returns {string}
 */
export function urlToFilePath(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== 'file:') {
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
    return ['file:', 'http:', 'https:', 'data:'].includes(parsed.protocol);
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
function matchImportMap(specifier, referrerUrl, importMap, mapBaseUrl) {
  if (!importMap) return null;

  // 1. Check scopes
  if (importMap.scopes && typeof importMap.scopes === 'object') {
    // WHATWG sorts serialized scope URLs, not their original relative spellings.
    const scopes = new Map();
    for (const [prefix, entries] of Object.entries(importMap.scopes)) {
      let scopeUrl;
      try {
        scopeUrl = new URL(prefix, mapBaseUrl).href;
      } catch {
        continue; // Unparseable scope URLs are ignored by import-map normalization.
      }
      scopes.set(scopeUrl, entries);
    }
    for (const scopeBase of [...scopes.keys()].sort().reverse()) {
      if (referrerUrl === scopeBase || (scopeBase.endsWith('/') && referrerUrl.startsWith(scopeBase))) {
        const match = matchMapEntries(specifier, scopes.get(scopeBase), mapBaseUrl);
        if (match !== null) return match;
      }
    }
  }

  // 2. Check top-level imports
  if (importMap.imports && typeof importMap.imports === 'object') {
    return matchMapEntries(specifier, importMap.imports, mapBaseUrl);
  }

  return null;
}

/**
 * @param {string} specifier
 * @param {Record<string, string | null>} entries
 * @param {string} mapBaseUrl
 * @returns {string | { blocked: true } | { error: string } | null}
 */
function matchMapEntries(specifier, entries, mapBaseUrl) {
  // Exact match
  if (Object.prototype.hasOwnProperty.call(entries, specifier)) {
    const target = entries[specifier];
    if (target === null) {
      return { blocked: true };
    }
    return new URL(target, mapBaseUrl).href;
  }

  // Prefix match (for keys ending with '/')
  const prefixKeys = Object.keys(entries).filter(k => k.endsWith('/')).sort((a, b) => b.length - a.length);
  for (const prefix of prefixKeys) {
    if (specifier.startsWith(prefix)) {
      const targetPrefix = entries[prefix];
      if (targetPrefix === null) {
        return { blocked: true };
      }
      if (typeof targetPrefix !== 'string' || !targetPrefix.endsWith('/')) {
        return {
          error: `Invalid import map prefix mapping: target for prefix "${prefix}" must end with "/" (got "${targetPrefix}")`
        };
      }
      const remainder = specifier.slice(prefix.length);
      const combined = targetPrefix + remainder;
      return new URL(combined, mapBaseUrl).href;
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
    const raw = fs.readFileSync(packageJsonPath, 'utf-8');
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
 * Resolves an export target value given admitted conditions ('import', 'default').
 * Handles string targets, conditional object targets, and null targets.
 * @param {any} target
 * @param {string[]} [conditions=['import', 'default']]
 * @returns {string | null | undefined}
 */
export function resolveExportTargetValue(target, conditions = ['import', 'default']) {
  if (typeof target === 'string') {
    return target;
  }
  if (target === null) {
    return null;
  }
  if (typeof target === 'object') {
    for (const key of Object.keys(target)) {
      if (key === 'default' || conditions.includes(key)) {
        const val = resolveExportTargetValue(target[key], conditions);
        if (val !== undefined) return val;
      }
    }
  }
  return undefined;
}

/**
 * Resolves a package subpath using the package.json exports map.
 * Evaluates exact keys, wildcard pattern keys, and condition priorities ('import', 'default').
 * Throws IngestionResolutionError with package.json path if not exported or blocked.
 *
 * @param {string} subpath - Relative subpath e.g. '.', './webgpu', './addons/controls/OrbitControls.js'
 * @param {any} exportsField - The `exports` field from package.json
 * @param {string} packageJsonPath - Path to package.json for error reporting
 * @param {string} specifier - Full import specifier e.g. 'three/webgpu'
 * @param {string} referrerUrl - Referrer URL for error reporting
 * @param {any} [span] - Source span for error reporting
 * @returns {string} - Relative target path e.g. './build/three.webgpu.js'
 */
export function resolvePackageExports(subpath, exportsField, packageJsonPath, specifier, referrerUrl, span) {
  if (!exportsField || typeof exportsField !== 'object') {
    throw new IngestionResolutionError(
      `Package "three" at "${packageJsonPath}" does not define a valid "exports" map`,
      specifier,
      referrerUrl,
      span
    );
  }

  // 1. Direct / exact match
  if (Object.prototype.hasOwnProperty.call(exportsField, subpath)) {
    const targetVal = exportsField[subpath];
    const resolved = resolveExportTargetValue(targetVal, ['import', 'default']);
    if (resolved === null) {
      throw new IngestionResolutionError(
        `Cannot resolve blocked package export "${specifier}" from "${referrerUrl}": mapped to null in "${packageJsonPath}"`,
        specifier,
        referrerUrl,
        span
      );
    }
    if (typeof resolved === 'string') {
      return resolved;
    }
    throw new IngestionResolutionError(
      `Cannot resolve package export "${specifier}" from "${referrerUrl}": no matching condition ('import', 'default') in "${packageJsonPath}"`,
      specifier,
      referrerUrl,
      span
    );
  }

  // 2. Pattern matching for keys containing '*'
  // Node spec: longest prefix match before '*' takes precedence
  const patternKeys = Object.keys(exportsField)
    .filter(k => k.includes('*'))
    .sort((a, b) => {
      const aPrefix = a.slice(0, a.indexOf('*'));
      const bPrefix = b.slice(0, b.indexOf('*'));
      return bPrefix.length - aPrefix.length;
    });

  for (const patternKey of patternKeys) {
    const starIdx = patternKey.indexOf('*');
    const prefix = patternKey.slice(0, starIdx);
    const suffix = patternKey.slice(starIdx + 1);

    if (subpath.startsWith(prefix) && (suffix === '' || subpath.endsWith(suffix))) {
      const wildcardMatch = subpath.slice(prefix.length, suffix ? subpath.length - suffix.length : undefined);
      const targetVal = exportsField[patternKey];
      const resolvedTarget = resolveExportTargetValue(targetVal, ['import', 'default']);

      if (resolvedTarget === null) {
        throw new IngestionResolutionError(
          `Cannot resolve blocked package export "${specifier}" from "${referrerUrl}": pattern "${patternKey}" mapped to null in "${packageJsonPath}"`,
          specifier,
          referrerUrl,
          span
        );
      }

      if (typeof resolvedTarget === 'string') {
        if (resolvedTarget.includes('*')) {
          return resolvedTarget.replace('*', wildcardMatch);
        }
        return resolvedTarget;
      }
    }
  }

  // 3. Subpath not exported
  throw new IngestionResolutionError(
    `Cannot resolve package import "${specifier}" from "${referrerUrl}": subpath "${subpath}" is not exported by package.json at "${packageJsonPath}"`,
    specifier,
    referrerUrl,
    span
  );
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
 * @param {string} [options.packageRootUrl] - Base URL for Three.js package fallback
 * @param {string} [options.mapBaseUrl] - Base URL for import map resolution
 * @param {{ line: number, column: number, offset: number } | null} [options.span]
 * @returns {string} - Canonical resolved file URL
 */
export function resolveModuleSpecifier(specifier, referrerUrl, importMap = {}, options = {}) {
  const span = options.span || null;
  const referrerBase = getBaseUrl(referrerUrl);
  const mapBaseUrl = options.mapBaseUrl || referrerBase;

  let candidateUrl = null;

  // 1. Try import map
  const mapped = matchImportMap(specifier, referrerUrl, importMap, mapBaseUrl);
  if (mapped) {
    if (typeof mapped === 'object' && mapped.blocked) {
      throw new IngestionResolutionError(
        `Cannot resolve blocked import specifier "${specifier}" from "${referrerUrl}": mapped to null in import map`,
        specifier,
        referrerUrl,
        span
      );
    }
    if (typeof mapped === 'object' && mapped.error) {
      throw new IngestionResolutionError(
        mapped.error,
        specifier,
        referrerUrl,
        span
      );
    }
    candidateUrl = mapped;
  } else if (isAbsoluteUrl(specifier)) {
    // 2. Explicit absolute URL (file://, http://, https://, data:)
    candidateUrl = new URL(specifier).href;
  } else if (specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/')) {
    // 3. Relative or pathname specifier
    try {
      candidateUrl = new URL(specifier, referrerBase).href;
    } catch (err) {
      throw new IngestionResolutionError(
        `Failed to parse URL for relative specifier "${specifier}" from "${referrerUrl}": ${err.message}`,
        specifier,
        referrerUrl,
        span
      );
    }
  } else if (specifier === 'three' || specifier.startsWith('three/')) {
    // 4. Pinned Three.js package resolution via package.json exports map
    let pkgRootDir;
    let pkgRootUrl;
    if (options.packageRootUrl) {
      if (options.packageRootUrl.startsWith('file://')) {
        pkgRootUrl = options.packageRootUrl.endsWith('/') ? options.packageRootUrl : options.packageRootUrl + '/';
        pkgRootDir = urlToFilePath(pkgRootUrl);
      } else {
        pkgRootDir = path.resolve(options.packageRootUrl);
        pkgRootUrl = pathToFileURL(pkgRootDir).href + '/';
      }
    } else {
      pkgRootDir = path.resolve('upstream/three.js');
      pkgRootUrl = pathToFileURL(pkgRootDir).href + '/';
    }

    const packageJsonPath = path.join(pkgRootDir, 'package.json');
    const pkgJson = loadPackageJson(packageJsonPath);
    if (!pkgJson) {
      throw new IngestionResolutionError(
        `Cannot resolve package import "${specifier}" from "${referrerUrl}": package.json not found at "${packageJsonPath}"`,
        specifier,
        referrerUrl,
        span
      );
    }

    const subpath = specifier === 'three' ? '.' : './' + specifier.slice('three/'.length);
    const relativeTarget = resolvePackageExports(
      subpath,
      pkgJson.exports,
      packageJsonPath,
      specifier,
      referrerUrl,
      span
    );
    candidateUrl = new URL(relativeTarget, pkgRootUrl).href;
  } else {
    throw new IngestionResolutionError(
      `Cannot resolve bare specifier "${specifier}" from "${referrerUrl}": no import map match and no package fallback`,
      specifier,
      referrerUrl,
      span
    );
  }

  // 5. Verify target exists if file: URL
  if (candidateUrl.startsWith('file://')) {
    let filePath;
    try {
      filePath = urlToFilePath(candidateUrl);
    } catch (err) {
      throw new IngestionResolutionError(
        `Invalid file URL "${candidateUrl}" for specifier "${specifier}": ${err.message}`,
        specifier,
        referrerUrl,
        span
      );
    }

    if (!fs.existsSync(filePath)) {
      throw new IngestionResolutionError(
        `Cannot resolve import specifier "${specifier}" from "${referrerUrl}": target file "${filePath}" does not exist`,
        specifier,
        referrerUrl,
        span
      );
    }

    // Preserve normalized URL identity (including query and fragment)
    return new URL(candidateUrl).href;
  }

  return candidateUrl;
}
