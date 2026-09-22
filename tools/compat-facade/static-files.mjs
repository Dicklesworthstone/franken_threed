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
    const responseHeaders = {
      "Content-Type": mimeTypes[path.extname(filePath).toLowerCase()] || fallbackType,
      ...headers,
    };
    if (req.method === "HEAD") {
      res.writeHead(200, responseHeaders);
      res.end();
      return true;
    }
    // Delay the 200 response until the file actually opens. Disappearance,
    // permission errors and failed reads must not become unhandled events.
    const stream = fs.createReadStream(resolved);
    const stop = () => stream.destroy();
    res.once("close", stop);
    stream.once("close", () => res.off("close", stop));
    stream.once("error", (error) => sendFileError(res, error));
    stream.once("open", () => {
      if (res.destroyed) return stream.destroy();
      res.writeHead(200, responseHeaders);
      stream.pipe(res);
    });
    return true;
  } catch (error) {
    sendFileError(res, error);
    return true;
  }
}
