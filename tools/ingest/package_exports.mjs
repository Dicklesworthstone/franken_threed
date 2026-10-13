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
  // Check before URL normalization erases traversal. Encoded dot/node_modules
  // segments have the same meaning as their unencoded spellings.
  for (const segment of value.split(/[\\/]/)) {
    let decoded;
    try { decoded = decodeURIComponent(segment).toLowerCase(); }
    catch { fail(code, `Invalid percent encoding in package path "${value}"`); }
    if (["", ".", "..", "node_modules"].includes(decoded) || /[/\\]/.test(decoded))
      fail(code, `Invalid package path segment in "${value}"`);
  }
}

function selectTarget(target, conditions, patternMatch, depth = 0) {
  if (depth > 128) fail("ERR_INVALID_PACKAGE_CONFIG", "Package export nesting exceeds 128 levels");
  if (typeof target === "string") {
    if (!target.startsWith("./")) invalidTarget(`Package export target "${target}" must start with "./"`);
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
      try { resolved = selectTarget(item, conditions, patternMatch, depth + 1); }
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
        const resolved = selectTarget(target[key], conditions, patternMatch, depth + 1);
        if (resolved !== undefined) return resolved;
      }
    }
    return undefined;
  }
  if (target === null) return null;
  invalidTarget(`Invalid package export target of type ${typeof target}`);
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
      if (!subpath.endsWith("/") && !subpath.includes("*") && Object.hasOwn(exportsField, subpath)) {
        target = exportsField[subpath]; matched = true;
      } else if (!subpath.endsWith("/")) {
        const patterns = keys.filter(key => key.indexOf("*") !== -1 && key.indexOf("*") === key.lastIndexOf("*"))
          .sort((a, b) => b.indexOf("*") - a.indexOf("*") || b.length - a.length);
        for (const key of patterns) {
          const star = key.indexOf("*"), prefix = key.slice(0, star), suffix = key.slice(star + 1);
          if (subpath.startsWith(prefix) && subpath.length > prefix.length &&
              (suffix === "" || (subpath.endsWith(suffix) && subpath.length >= key.length))) {
            target = exportsField[key];
            patternMatch = subpath.slice(prefix.length, subpath.length - suffix.length);
            matched = true;
            break; // A selected pattern never falls through to a less specific one.
          }
        }
      }
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
