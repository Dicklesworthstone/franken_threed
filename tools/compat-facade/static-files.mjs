import fs from "node:fs";
import path from "node:path";

function isWithin(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function accessDenied() {
  return Object.assign(new Error("Access denied"), { code: "EACCES" });
}

/**
 * Check both lexical and real paths. An alias cannot escape its own directory,
 * and an alias directory cannot itself be a symlink outside the repository.
 * Repository contents are trusted local inputs, not a hostile writable filesystem.
 */
export function resolveStaticFile(root, candidate, repositoryRoot = root) {
  root = path.resolve(root);
  candidate = path.resolve(candidate);
  repositoryRoot = path.resolve(repositoryRoot);
  if (!isWithin(repositoryRoot, root) || !isWithin(root, candidate)) throw accessDenied();
  try {
    const realRepository = fs.realpathSync(repositoryRoot);
    const realRoot = fs.realpathSync(root);
    if (!isWithin(realRepository, realRoot)) throw accessDenied();
    const realFile = fs.realpathSync(candidate);
    if (!isWithin(realRoot, realFile)) throw accessDenied();
    return fs.statSync(realFile).isFile() ? realFile : null;
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
    throw error;
  }
}

export function sendFileError(res, error) {
  if (res.destroyed) return;
  if (res.headersSent) {
    res.destroy();
    return;
  }
  const missing = error.code === "ENOENT" || error.code === "ENOTDIR";
  const denied = error.code === "EACCES" || error.code === "EPERM" || error.code === "ELOOP";
  res.writeHead(missing ? 404 : denied ? 403 : 500, {
    "Content-Type": "text/plain; charset=utf-8",
  });
  res.end(missing ? "File not found" : denied ? "Access denied" : "Unable to read file");
}


/** A server may decline multipart/invalid ranges and return the full asset.
 * A valid but unsatisfiable single range produces 416 instead.
 */
function requestedByteRange(header, size) {
  if (typeof header !== "string" || !header.startsWith("bytes=")) return null;
  const value = header.slice(6).trim();
  if (value.includes(",")) return null;
  const match = /^(\d*)-(\d*)$/.exec(value);
  if (!match || (!match[1] && !match[2])) return null;
  const toIndex = digits => {
    const number = BigInt(digits);
    return number > BigInt(Number.MAX_SAFE_INTEGER) ? Infinity : Number(number);
  };
  let start, end;
  if (!match[1]) {
    const suffix = toIndex(match[2]);
    if (suffix === 0 || size === 0) return { unsatisfiable: true };
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = toIndex(match[1]);
    if (start >= size || size === 0) return { unsatisfiable: true };
    end = match[2] ? Math.min(size - 1, toIndex(match[2])) : size - 1;
    if (end < start) return { unsatisfiable: true };
  }
  return { start, end };
}

// A weak or stale If-Range validator cannot authorize a partial response.
// HTTP dates have second precision while local mtimes may not.
function matchesIfRange(header, etag, mtimeMs) {
  if (typeof header !== "string") return true;
  if (header.startsWith('"')) return header === etag;
  if (header.startsWith("W/")) return false;
  const timestamp = Date.parse(header);
  return Number.isFinite(timestamp) && Math.trunc(mtimeMs / 1000) * 1000 <= timestamp;
}

/** Return false only for absent files/directories, allowing a safe fallback. */
export function serveStaticFile(
  req,
  res,
  filePath,
  root,
  repositoryRoot,
  mimeTypes,
  { fallbackType = "application/octet-stream", headers = {} } = {},
) {
  try {
    const resolved = resolveStaticFile(root, filePath, repositoryRoot);
    if (resolved === null) return false;
    const stats = fs.statSync(resolved);
    const etag = '"' + stats.size.toString(16) + "-" + Math.trunc(stats.mtimeMs).toString(16) + '"';
    const responseHeaders = {
      "Content-Type": mimeTypes[path.extname(filePath).toLowerCase()] || fallbackType,
      ...headers,
      "Accept-Ranges": "bytes",
      "ETag": etag,
      "Last-Modified": stats.mtime.toUTCString(),
    };
    const range = matchesIfRange(req.headers?.["if-range"], etag, stats.mtimeMs)
      ? requestedByteRange(req.headers?.range, stats.size) : null;
    if (range?.unsatisfiable) {
      res.writeHead(416, { ...responseHeaders, "Content-Range": "bytes */" + stats.size, "Content-Length": "0" });
      res.end();
      return true;
    }
    if (range) responseHeaders["Content-Range"] = "bytes " + range.start + "-" + range.end + "/" + stats.size;
    responseHeaders["Content-Length"] = String(range ? range.end - range.start + 1 : stats.size);
    const status = range ? 206 : 200;
    if (req.method === "HEAD") {
      res.writeHead(status, responseHeaders);
      res.end();
      return true;
    }
    // Wait for open before sending success; broken/disappearing files still
    // report errors, and aborting a client tears down its underlying stream.
    const stream = fs.createReadStream(resolved, range ? { start: range.start, end: range.end } : undefined);
    const stop = () => stream.destroy();
    res.once("close", stop);
    stream.once("close", () => res.off("close", stop));
    stream.once("error", (error) => sendFileError(res, error));
    stream.once("open", () => {
      if (res.destroyed) return stream.destroy();
      res.writeHead(status, responseHeaders);
      stream.pipe(res);
    });
    return true;
  } catch (error) {
    sendFileError(res, error);
    return true;
  }
}
