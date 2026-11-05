/** Explicit local document-root URL mapping shared by ingestion and packaging. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

function documentError(code, message, cause) {
  return Object.assign(new Error(message, cause ? { cause } : undefined), { code });
}

function localPath(value) {
  if (value instanceof URL || (typeof value === "string" && /^[a-z][a-z\d+.-]*:/i.test(value))) {
    const url = value instanceof URL ? value : new URL(value);
    if (url.protocol !== "file:" || url.hostname) {
      throw documentError("F3D_DOCUMENT_URL", `Expected a local file URL, got "${value}"`);
    }
    try { return fileURLToPath(url); }
    catch (cause) { throw documentError("F3D_DOCUMENT_URL", `Invalid local file URL "${value}": ${cause.message}`, cause); }
  }
  if (typeof value !== "string" || value.length === 0) {
    throw documentError("F3D_DOCUMENT_URL", "Expected a local path or file URL");
  }
  return path.resolve(value);
}

/** Return the absolute lexical root path; retaining its spelling preserves module identity. */
export function normalizeDocumentRoot(value) {
  if (value === null || value === undefined) return null;
  try {
    if (value instanceof URL || (typeof value === "string" && value.startsWith("file:"))) {
      const url = value instanceof URL ? value : new URL(value);
      if (url.search || url.hash) throw new Error("A document root cannot have a query or fragment");
    }
    const root = localPath(value);
    if (!fs.statSync(root).isDirectory()) throw new Error("The document root must be a directory");
    return root;
  } catch (cause) {
    throw documentError("F3D_DOCUMENT_ROOT", `Invalid document root "${value}": ${cause.message}`, cause);
  }
}

function urlText(value) {
  // Match URL parsing's ASCII whitespace handling without trimming valid non-ASCII filenames.
  return String(value).replace(/^[\u0000-\u0020]+|[\u0000-\u0020]+$/g, "").replace(/[\t\n\r]/g, "");
}

/** Protocol-relative references (including URL-parser backslash spelling) are never local. */
export function isRootRelativeDocumentUrl(value) {
  const text = urlText(value);
  return text.startsWith("/") && text[1] !== "/" && text[1] !== "\\";
}

function inside(root, target) {
  const relative = path.relative(root, target);
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

/**
 * Return a POSIX output path, checking both lexical and physical containment.
 * Missing leaves are admitted for import-map prefixes; their existing ancestors must be inside.
 */
export function documentRelativePath(pathOrFileUrl, documentRoot) {
  const root = normalizeDocumentRoot(documentRoot);
  if (!root) throw documentError("F3D_DOCUMENT_ROOT", "A document root is required");
  const target = localPath(pathOrFileUrl);
  const escape = () => documentError("F3D_DOCUMENT_ROOT_ESCAPE", `Document resource "${pathOrFileUrl}" escapes document root "${root}"`);
  if (!inside(root, target)) throw escape();

  let ancestor = target;
  for (;;) {
    try {
      fs.lstatSync(ancestor);
      break;
    } catch (cause) {
      if (cause.code !== "ENOENT" && cause.code !== "ENOTDIR") throw cause;
      const parent = path.dirname(ancestor);
      if (parent === ancestor) throw cause;
      ancestor = parent;
    }
  }
  // realpath also rejects dangling symlinks instead of walking past them.
  let physicalRoot, physicalAncestor;
  try {
    physicalRoot = fs.realpathSync(root);
    physicalAncestor = fs.realpathSync(ancestor);
  } catch (cause) {
    throw documentError("F3D_DOCUMENT_ROOT", `Cannot inspect document resource "${pathOrFileUrl}": ${cause.message}`, cause);
  }
  if (!inside(physicalRoot, physicalAncestor)) throw escape();
  return path.relative(root, target).split(path.sep).join("/");
}

/** Resolve a browser URL, opting local single-leading-slash references into documentRoot. */
export function resolveDocumentUrl(specifier, baseUrl, documentRoot) {
  const root = normalizeDocumentRoot(documentRoot);
  if (!root || !isRootRelativeDocumentUrl(specifier)) return new URL(specifier, baseUrl).href;
  const base = new URL(baseUrl);
  if (base.protocol !== "file:" || base.hostname) return new URL(specifier, baseUrl).href;
  const rootUrl = pathToFileURL(root.endsWith(path.sep) ? root : `${root}${path.sep}`);
  // Browsers clamp leading dot segments at the origin root, before filesystem mapping.
  const browserUrl = new URL(urlText(specifier), "https://f3d-document.invalid/");
  const resolved = new URL(`.${browserUrl.pathname}`, rootUrl);
  resolved.search = browserUrl.search;
  resolved.hash = browserUrl.hash;
  documentRelativePath(resolved, root);
  return resolved.href;
}
