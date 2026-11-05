/** Serve emitted applications without upstream aliases or HTML rewriting. */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { resolveStaticFile, serveStaticFile } from "../compat-facade/static-files.mjs";
import { documentRelativePath, normalizeDocumentRoot } from "./document_urls.mjs";

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json",
  ".wasm": "application/wasm", ".wgsl": "text/plain; charset=utf-8",
  ".glsl": "text/plain; charset=utf-8", ".gltf": "model/gltf+json", ".glb": "model/gltf-binary",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".webp": "image/webp", ".avif": "image/avif", ".gif": "image/gif", ".ico": "image/x-icon",
  ".ktx2": "image/ktx2", ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf",
  ".mp4": "video/mp4", ".webm": "video/webm", ".mp3": "audio/mpeg", ".wav": "audio/wav",
};

export async function startApplicationServer(target, { port = 8080, isolate = false, documentRoot = null } = {}) {
  if (!Number.isInteger(port) || port < 0 || port > 65535)
    throw new RangeError("Port must be an integer from 0 to 65535");
  const absolute = path.resolve(target), stat = fs.statSync(absolute);
  if (!stat.isDirectory() && (!stat.isFile() || !/\.html?$/i.test(absolute)))
    throw new Error("Serve expects an application directory or an HTML file");
  const declaredRoot = normalizeDocumentRoot(documentRoot);
  const relativeTarget = declaredRoot ? documentRelativePath(absolute, declaredRoot) : null;
  const root = fs.realpathSync(declaredRoot ?? (stat.isDirectory() ? absolute : path.dirname(absolute)));
  const directory = stat.isDirectory() ? relativeTarget ?? "" : "";
  let entry = stat.isFile() ? relativeTarget ?? path.basename(absolute) : null;
  if (entry && !resolveStaticFile(root, path.join(root, entry)))
    throw new Error("The HTML entry is unavailable");
  if (!entry) {
    const index = path.posix.join(directory, "index.html");
    if (resolveStaticFile(root, path.join(root, index))) entry = index;
    else {
      const html = fs.readdirSync(path.join(root, directory), { withFileTypes: true })
        .filter(item => item.isFile() && /\.html?$/i.test(item.name)).map(item => item.name);
      if (html.length === 1) entry = path.posix.join(directory, html[0]);
    }
  }
  const encodePath = value => value.split("/").map(encodeURIComponent).join("/");
  const entryPath = entry ? "/" + encodePath(entry) : directory ? "/" + encodePath(directory) + "/" : "/";
  const server = http.createServer((request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (isolate) {
      response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { Allow: "GET, HEAD" });
      response.end("Method not allowed");
      return;
    }
    let url, pathname;
    try {
      url = new URL(request.url, "http://127.0.0.1");
      pathname = decodeURIComponent(url.pathname);
      if (pathname.includes("\\") || /[\u0000-\u001f\u007f]/.test(pathname)) throw new URIError();
    } catch {
      response.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Invalid request URL");
      return;
    }
    // A named build entry keeps its own document URL, including reload queries.
    if (pathname === "/" && entry && entry !== "index.html") {
      response.writeHead(302, { Location: entryPath + url.search });
      response.end();
      return;
    }
    if (pathname.endsWith("/")) pathname += "index.html";
    const filename = path.resolve(root, pathname.replace(/^\/+/, ""));
    if (serveStaticFile(request, response, filename, root, root, MIME_TYPES)) return;
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("File not found; select an HTML entry explicitly if this directory has multiple pages");
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  let closing;
  return {
    root, server, port: server.address().port, url: baseUrl + entryPath,
    close() {
      return closing ??= new Promise((resolve, reject) => {
        server.close(error => error ? reject(error) : resolve());
        server.closeAllConnections();
      });
    },
  };
}

export async function serveApplicationCommand(args) {
  let target, port = 8080, isolate = false, documentRoot = null;
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      console.log("Usage: node tools/ingest/cli.mjs serve <directory-or-html> [--document-root directory] [--port 8080] [--isolate]");
      return;
    }
    if (arg === "--isolate") isolate = true;
    else if (arg === "--port" && /^\d+$/.test(args[index + 1] ?? "")) port = Number(args[++index]);
    else if (arg === "--document-root" && args[index + 1] && !args[index + 1].startsWith("-")) documentRoot = args[++index];
    else if (!arg.startsWith("-") && !target) target = arg;
    else throw new Error(`Invalid serve argument: ${arg}`);
  }
  if (!target) throw new Error("serve requires an application directory or HTML file");
  const running = await startApplicationServer(target, { port, isolate, documentRoot });
  const stop = async () => {
    process.off("SIGINT", stop);
    process.off("SIGTERM", stop);
    await running.close();
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  console.log(`Serving application at ${running.url}`);
  if (isolate) console.log("Cross-origin isolation enabled (COOP same-origin; COEP require-corp).");
}
