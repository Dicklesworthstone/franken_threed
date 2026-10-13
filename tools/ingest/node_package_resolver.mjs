/** Local installed-ESM resolution for application ingestion.
 * Import maps and the pinned Three.js override are handled by the caller first.
 * Never downloads packages, uses global npm paths, or guesses a CommonJS route.
 * Package exports are authoritative, including failures in the nearest package.
 */
import fs from "node:fs";
import path from "node:path";
import { isBuiltin } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolvePackageExports } from "./package_exports.mjs";
import { IngestionResolutionError } from "./types.mjs";

export function resolveInstalledPackage(specifier, referrerUrl, { span = null } = {}) {
  const fail = (code, message) => {
    const error = new IngestionResolutionError(
      `Cannot resolve package import "${specifier}" from "${referrerUrl}": ${message}`,
      specifier, referrerUrl, span,
    );
    error.code = code;
    throw error;
  };
  if (typeof specifier !== "string" || !specifier || /[\\\u0000-\u001f\u007f]/.test(specifier))
    fail("ERR_INVALID_MODULE_SPECIFIER", "Invalid bare package specifier");
  if (isBuiltin(specifier))
    fail("F3D_NODE_BUILTIN", "Node builtins need an explicit import-map host adapter; no browser polyfill is implied");
  if (specifier.startsWith("#"))
    fail("F3D_PACKAGE_IMPORTS", "Package-local #imports require an explicit import-map mapping on this route");
  const parts = specifier.split("/");
  const scoped = specifier.startsWith("@");
  const count = scoped ? 2 : 1;
  const nameParts = parts.slice(0, count);
  if (nameParts.length !== count || nameParts.some((part, index) => {
    const name = scoped && index === 0 ? part.slice(1) : part;
    return !name || name.startsWith(".") || /[\s%:?#@]/.test(name);
  })) fail("ERR_INVALID_MODULE_SPECIFIER", "Invalid package name");
  const name = nameParts.join("/");
  const subpath = parts.length === count ? "." : "./" + parts.slice(count).join("/");
  let referrer;
  try { referrer = new URL(referrerUrl); }
  catch { fail("ERR_INVALID_MODULE_SPECIFIER", "Expected an absolute referrer URL"); }
  if (referrer.protocol !== "file:")
    fail("F3D_PACKAGE_REFERRER", "Local package fallback is only available to file: referrers");

  function stat(filename) {
    try { return fs.statSync(filename); }
    catch (error) {
      if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
      fail("F3D_PACKAGE_IO", `Cannot inspect "${filename}": ${error.message}`);
    }
  }
  function manifest(directory) {
    const filename = path.join(directory, "package.json");
    let text;
    try { text = fs.readFileSync(filename, "utf8"); }
    catch (error) {
      if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
      fail("F3D_PACKAGE_IO", `Cannot read "${filename}": ${error.message}`);
    }
    try {
      const value = JSON.parse(text);
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected a JSON object");
      return value;
    } catch (error) { fail("ERR_INVALID_PACKAGE_CONFIG", `Invalid "${filename}": ${error.message}`); }
  }
  function scope(directory) {
    for (let current = directory; path.basename(current) !== "node_modules";) {
      const value = manifest(current);
      if (value) return { directory: current, value };
      const parent = path.dirname(current);
      if (parent === current) break;
      current = parent;
    }
    return null;
  }
  let referrerPath;
  try {
    referrerPath = fileURLToPath(referrer);
    // Workspace links resolve dependencies from their actual package location.
    if (stat(referrerPath)) referrerPath = fs.realpathSync(referrerPath);
  } catch (error) {
    if (error instanceof IngestionResolutionError) throw error;
    fail("ERR_INVALID_MODULE_SPECIFIER", `Invalid local referrer: ${error.message}`);
  }
  const fromDirectory = path.dirname(referrerPath);

  function resolveAt(directory, value) {
    const packageJsonPath = path.join(directory, "package.json");
    let target;
    if (value?.exports !== null && value?.exports !== undefined) {
      target = resolvePackageExports(subpath, value.exports, packageJsonPath, specifier, referrerUrl, span);
    } else {
      // Legacy ESM packages may use an explicit main file or exact subpaths.
      // Extension/directory guessing is deliberately not a CommonJS fallback.
      const main = typeof value?.main === "string" && value.main ? value.main : "index.js";
      const relative = subpath === "." ? (main.startsWith("./") ? main : "./" + main) : subpath;
      target = resolvePackageExports(".", relative, packageJsonPath, specifier, referrerUrl, span);
    }
    const url = new URL(target, pathToFileURL(directory + path.sep));
    let filename;
    try { filename = fileURLToPath(url); }
    catch (error) { fail("ERR_INVALID_MODULE_SPECIFIER", `Invalid target in "${packageJsonPath}": ${error.message}`); }
    const info = stat(filename);
    if (!info) fail("ERR_MODULE_NOT_FOUND", `Selected target "${filename}" does not exist ("${packageJsonPath}")`);
    if (!info.isFile()) fail("ERR_UNSUPPORTED_DIR_IMPORT", `Selected target "${filename}" is not a regular module file`);
    try { filename = fs.realpathSync(filename); }
    catch (error) { fail("F3D_PACKAGE_IO", `Cannot canonicalize "${filename}": ${error.message}`); }
    const extension = path.extname(filename);
    if (extension !== ".mjs" && !(extension === ".js" && scope(path.dirname(filename))?.value.type === "module"))
      fail("F3D_PACKAGE_FORMAT", `Selected target "${filename}" is not explicit ESM (.mjs or .js in a type:module package); CommonJS and non-JavaScript loaders need a separate route`);
    const canonical = pathToFileURL(filename);
    canonical.search = url.search; canonical.hash = url.hash;
    return canonical.href;
  }

  const own = scope(fromDirectory);
  if (own?.value.name === name && own.value.exports !== null && own.value.exports !== undefined)
    return resolveAt(own.directory, own.value);
  for (let directory = fromDirectory;;) {
    if (path.basename(directory) !== "node_modules") {
      const candidate = path.join(directory, "node_modules", ...nameParts);
      if (stat(candidate)?.isDirectory()) return resolveAt(candidate, manifest(candidate));
    }
    const parent = path.dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  fail("ERR_MODULE_NOT_FOUND", `No import map match and no installed package "${name}" reachable from the referrer`);
}
