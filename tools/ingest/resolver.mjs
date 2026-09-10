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
  return fileURLToPath(new URL(parsed.pathname, 'file:///'));
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
    const sortedScopes = Object.keys(importMap.scopes).sort((a, b) => b.length - a.length);
    for (const scopePrefix of sortedScopes) {
      const scopeBase = new URL(scopePrefix, mapBaseUrl).href;
      if (referrerUrl.startsWith(scopeBase) || referrerUrl.startsWith(scopePrefix)) {
        const scopeImports = importMap.scopes[scopePrefix];
        const match = matchMapEntries(specifier, scopeImports, mapBaseUrl);
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
    candidateUrl = specifier;
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
  } else {
    // 4. Fallback resolution for pinned Three.js package
    const pkgRoot = options.packageRootUrl || pathToFileURL(path.resolve('upstream/three.js/')).href + '/';
    if (specifier === 'three') {
      candidateUrl = new URL('build/three.module.js', pkgRoot).href;
    } else if (specifier === 'three/webgpu') {
      candidateUrl = new URL('build/three.webgpu.js', pkgRoot).href;
    } else if (specifier === 'three/tsl') {
      candidateUrl = new URL('build/three.tsl.js', pkgRoot).href;
    } else if (specifier.startsWith('three/addons/')) {
      const rest = specifier.slice('three/addons/'.length);
      candidateUrl = new URL(`examples/jsm/${rest}`, pkgRoot).href;
    } else if (specifier.startsWith('three/src/')) {
      const rest = specifier.slice('three/src/'.length);
      candidateUrl = new URL(`src/${rest}`, pkgRoot).href;
    } else {
      throw new IngestionResolutionError(
        `Cannot resolve bare specifier "${specifier}" from "${referrerUrl}": no import map match and no package fallback`,
        specifier,
        referrerUrl,
        span
      );
    }
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
