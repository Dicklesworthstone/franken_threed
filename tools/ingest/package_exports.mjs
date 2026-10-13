/** Package export selection for ingestion, without filesystem probing.
 *
 * Mirrors PACKAGE_EXPORTS_RESOLVE/PACKAGE_TARGET_RESOLVE for ESM export
 * maps. Conditions are tested in manifest order, not caller-priority order.
 * Array fallbacks recover only from invalid targets, not invalid manifests.
 * A missing selected file is NOT an invitation to try another export target.
 * Import-map resolution and the pinned Three.js package choice stay separate.
 */
import { IngestionResolutionError } from "./types.mjs";

class PackageExportError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}
const fail = (code, message) => { throw new PackageExportError(code, message); };
const invalidTarget = message => fail("ERR_INVALID_PACKAGE_TARGET", message);
const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
const arrayIndex = key => String(Number(key)) === key && Number.isInteger(Number(key)) && Number(key) >= 0 && Number(key) < 0xffffffff;

function checkSegments(value, code) {
  if (/[\u0000-\u001f\u007f]/.test(value)) fail(code, "Control characters are not valid package path text");
  // Check before URL normalization erases traversal. Encoded dot/node_modules
  // segments have the same meaning as their unencoded spellings.
  for (const segment of value.split(/[\\/]/)) {
    let decoded;
    try { decoded = decodeURIComponent(segment).toLowerCase(); }
    catch { fail(code, `Invalid percent encoding in package path "${value}"`); }
    if (["", ".", "..", "node_modules"].includes(decoded) || /[/\\\u0000-\u001f\u007f]/.test(decoded))
      fail(code, `Invalid package path segment in "${value}"`);
  }
}

function selectTarget(target, conditions, patternMatch, depth = 0, resolveExternal = null) {
  if (depth > 128) fail("ERR_INVALID_PACKAGE_CONFIG", "Package export nesting exceeds 128 levels");
  if (typeof target === "string") {
    if (!target.startsWith("./")) {
      if (!resolveExternal || target.startsWith("../") || target.startsWith("/") || URL.canParse(target))
        invalidTarget(`Invalid package target "${target}": expected a package-relative path or an imports dependency`);
      // An imports dependency is a bare package, not a second #imports lookup.
      // Do not invent alias recursion (or allow it to bypass the owning scope).
      if (target.startsWith("#")) fail("ERR_INVALID_MODULE_SPECIFIER", "An external imports target must name a package, not another #import");
      if (patternMatch !== null) checkSegments(patternMatch, "ERR_INVALID_MODULE_SPECIFIER");
      return resolveExternal(patternMatch === null ? target : target.split("*").join(patternMatch));
    }
    checkSegments(target.slice(2).split(/[?#]/, 1)[0], "ERR_INVALID_PACKAGE_TARGET");
    if (patternMatch === null) return target;
    checkSegments(patternMatch, "ERR_INVALID_MODULE_SPECIFIER");
    // String replacement syntax ($&, $`, $') must not interpret specifier text.
    return target.split("*").join(patternMatch);
  }
  if (Array.isArray(target)) {
    if (target.length === 0) return null;
    let lastFailure;
    for (const item of target) {
      let resolved;
      try { resolved = selectTarget(item, conditions, patternMatch, depth + 1, resolveExternal); }
      catch (error) {
        if (error.code !== "ERR_INVALID_PACKAGE_TARGET") throw error;
        lastFailure = error;
        continue;
      }
      if (resolved === undefined) continue;
      if (resolved === null) { lastFailure = null; continue; }
      return resolved;
    }
    if (lastFailure instanceof Error) throw lastFailure;
    return lastFailure;
  }
  if (object(target)) {
    const keys = Object.keys(target);
    if (keys.some(arrayIndex)) fail("ERR_INVALID_PACKAGE_CONFIG", "Package export conditions cannot contain numeric property keys");
    for (const key of keys) {
      if (key === "default" || conditions.includes(key)) {
        const resolved = selectTarget(target[key], conditions, patternMatch, depth + 1, resolveExternal);
        if (resolved !== undefined) return resolved;
      }
    }
    return undefined;
  }
  if (target === null) return null;
  invalidTarget(`Invalid package export target of type ${typeof target}`);
}

/** Exact keys win, then longest prefix/trailer. A selected null or unmatched
 * conditional never falls through to a broader key, for either map family. */
function matchSubpath(specifier, entries) {
  if (specifier.endsWith("/")) return null;
  if (!specifier.includes("*") && Object.hasOwn(entries, specifier)) return [entries[specifier], null];
  const patterns = Object.keys(entries)
    .filter(key => key.indexOf("*") !== -1 && key.indexOf("*") === key.lastIndexOf("*"))
    .sort((a, b) => b.indexOf("*") - a.indexOf("*") || b.length - a.length);
  for (const key of patterns) {
    const star = key.indexOf("*"), prefix = key.slice(0, star), suffix = key.slice(star + 1);
    if (specifier.startsWith(prefix) && specifier.length > prefix.length &&
        (suffix === "" || (specifier.endsWith(suffix) && specifier.length >= key.length)))
      return [entries[key], specifier.slice(prefix.length, specifier.length - suffix.length)];
  }
  return null;
}

/** Select the owning package's private mapping. Relative targets are finalized
 * by the caller. External dependencies resolve INSIDE target selection, so a
 * missing/blocked dependency cannot silently activate a different array entry.
 * The callback must preserve the application's import maps and Three.js pin. */
export function resolvePackageImports(specifier, importsField, packageJsonPath, referrerUrl, span, resolveExternal) {
  try {
    if (typeof specifier !== "string" || !specifier.startsWith("#") || specifier === "#" ||
        specifier.startsWith("#/") || specifier.endsWith("/"))
      fail("ERR_INVALID_MODULE_SPECIFIER", `Invalid package import specifier "${specifier}"`);
    if (importsField !== null && importsField !== undefined && !object(importsField))
      fail("ERR_INVALID_PACKAGE_CONFIG", "Package imports must be an object");
    const match = object(importsField) ? matchSubpath(specifier, importsField) : null;
    if (!match) fail("ERR_PACKAGE_IMPORT_NOT_DEFINED", `Package import "${specifier}" is not defined`);
    const resolved = selectTarget(match[0], ["import", "default"], match[1], 0, resolveExternal);
    if (resolved === null || resolved === undefined)
      fail("ERR_PACKAGE_IMPORT_NOT_DEFINED", `Package import "${specifier}" is blocked or has no matching condition`);
    return resolved;
  } catch (cause) {
    if (!(cause instanceof PackageExportError) && !(cause instanceof IngestionResolutionError)) throw cause;
    const error = new IngestionResolutionError(
      `Cannot resolve package import "${specifier}" from "${referrerUrl}" in "${packageJsonPath}": ${cause.message}`,
      specifier, referrerUrl, span,
    );
    error.code = cause.code;
    error.cause = cause;
    throw error;
  }
}

/** Select a target using the admitted conditions; null and undefined differ. */
export function resolveExportTargetValue(target, conditions = ["import", "default"]) {
  return selectTarget(target, conditions, null);
}

/** Resolve an export to a package-relative URL, preserving query/fragment text.
 * The caller verifies existence only after selection. No extension guessing,
 * private-subpath fallback, network access, or CommonJS execution happens here.
 */
export function resolvePackageExports(subpath, exportsField, packageJsonPath, specifier, referrerUrl, span) {
  try {
    if (typeof subpath !== "string" || (subpath !== "." && !subpath.startsWith("./")))
      fail("ERR_INVALID_MODULE_SPECIFIER", `Invalid package subpath "${subpath}"`);
    const keys = object(exportsField) ? Object.keys(exportsField) : [];
    const subpathKeys = keys.filter(key => key.startsWith("."));
    if (subpathKeys.length !== 0 && subpathKeys.length !== keys.length)
      fail("ERR_INVALID_PACKAGE_CONFIG", "Package exports cannot mix subpath keys and condition keys");
    const isSubpathMap = object(exportsField) && subpathKeys.length > 0;
    let target, patternMatch = null, matched = false;
    if (subpath === "." && !isSubpathMap) {
      target = exportsField; matched = true;
    } else if (isSubpathMap) {
      const match = matchSubpath(subpath, exportsField);
      if (match) { [target, patternMatch] = match; matched = true; }
    }
    if (!matched) fail("ERR_PACKAGE_PATH_NOT_EXPORTED", `Subpath "${subpath}" is not exported`);
    const resolved = selectTarget(target, ["import", "default"], patternMatch);
    if (resolved === null) fail("ERR_PACKAGE_PATH_NOT_EXPORTED", `Blocked package export "${specifier}": mapped to null`);
    if (resolved === undefined) fail("ERR_PACKAGE_PATH_NOT_EXPORTED", "Package export has no matching condition ('import', 'default')");
    return resolved;
  } catch (cause) {
    if (!(cause instanceof PackageExportError)) throw cause;
    const error = new IngestionResolutionError(
      `Cannot resolve package export "${specifier}" from "${referrerUrl}" in "${packageJsonPath}": ${cause.message}`,
      specifier, referrerUrl, span,
    );
    error.code = cause.code;
    throw error;
  }
}
