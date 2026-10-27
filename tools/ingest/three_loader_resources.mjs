/** A finite embedded resource route for the pinned Three.js loader host calls.
 * Logical URLs stay in the upstream loader: its manager, caches, deduplication,
 * glTF resource bases and callbacks are unchanged. The final Request, image.src
 * or bitmap fetch receives an embedded response URL; exact source codec-worker
 * sites forward to the native constructor. No global host function or
 * LoadingManager is replaced. Unknown URLs retain their host route.
 */
import { createHash } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";
import * as walk from "acorn-walk";

export const LOADER_RESOURCES_IMPORT = "f3d:standalone-loader-resources";
export const LOADER_RESOURCES_FILE = "f3d-loader-resources.mjs";
const RESOLVE = "__f3dResolveLoaderResource";
const CREATE_WORKER = "__f3dCreateLoaderWorker";
const ROOT = new URL("../../upstream/three.js/", import.meta.url).href;
// Exact modules from r186 source commit 148ef33ecb6d2502ff796d4554abd1549c95d519.
// The surrounding lexical scope is part of the fingerprint: a custom Request,
// fetch or Worker binding must not receive a host hook just because a loader
// class body happens to be unchanged.
const MODULES = new Map([
  ["build/three.core.js", "9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6"],
  ["src/loaders/FileLoader.js", "e3f730f2fdefdd4e9285883fb2eb7ca40e13d58ea96e969a669d3eaacd7d0c03"],
  ["src/loaders/ImageLoader.js", "0507db81b6214afd32f4c781e08acd88005ab4988f93de583f34ded571f1fbcc"],
  ["src/loaders/ImageBitmapLoader.js", "466fc2172c6d1c8d4ee40dc041785a783e09bbef79c3017a945024b8d2166c30"],
  ["examples/jsm/loaders/DRACOLoader.js", "fdafdf4aa11bbc65d89adb338efd844bedf94073702f9049240ff1f1ea48ad5f"],
  ["examples/jsm/loaders/KTX2Loader.js", "3a3233ce3409443076d3414b78832e1405fcfac5ecb34d367365fd0781127d6c"],
  ["examples/jsm/libs/meshopt_decoder.module.js", "d428e73a000057c6c94bfcedc3412d0c2dc14ca8800e88553aab05113ad2bf19"],
]);
const CLASSES = new Set(["FileLoader", "ImageLoader", "ImageBitmapLoader"]);
const WORKER_ARGUMENTS = new Map([
  ["examples/jsm/loaders/DRACOLoader.js", "this.workerSourceURL"],
  ["examples/jsm/loaders/KTX2Loader.js", "this.workerSourceURL"],
  ["examples/jsm/libs/meshopt_decoder.module.js", "url"],
]);
const hash = value => createHash("sha256").update(value).digest("hex");

/** Return focused AST edits; modified or foreign implementations are untouched. */
export function threeLoaderResourceEdits(source, ast, id, packageRootUrl) {
  const canonical = value => {
    const url = value.startsWith("file:") ? new URL(value) : pathToFileURL(path.resolve(value));
    url.search = ""; url.hash = ""; return url.href;
  };
  const root = packageRootUrl ? canonical(packageRootUrl).replace(/\/?$/, "/") : ROOT;
  const module = canonical(id);
  if (!module.startsWith(root) || hash(source) !== MODULES.get(module.slice(root.length))) return [];
  const edits = [], imports = new Set();
  const workerArgument = WORKER_ARGUMENTS.get(module.slice(root.length));
  if (workerArgument) {
    const sites = [];
    walk.simple(ast, {
      NewExpression(value) {
        if (value.callee.type === "Identifier" && value.callee.name === "Worker" &&
            value.arguments.length === 1 &&
            source.slice(value.arguments[0].start, value.arguments[0].end) === workerArgument) sites.push(value);
      },
    });
    if (sites.length !== 1) throw new Error(`Pinned ${module.slice(root.length)} has an unexpected worker constructor shape`);
    // Keep source-created Blobs, self-contained decoder payloads, transferred
    // buffers, pools and disposal. Only native construction crosses this exact
    // source boundary, before bundling erases module identity.
    edits.push({ type: "replace", pos: sites[0].start, end: sites[0].end,
      text: `${CREATE_WORKER}(Worker, ${workerArgument})`, priority: 2 });
    imports.add(`createThreeLoaderWorker as ${CREATE_WORKER}`);
  }
  for (const node of ast.body) {
    if (node.type !== "ClassDeclaration") continue;
    if (!CLASSES.has(node.id?.name)) continue;
    const load = node.body.body.find(method => method.key?.name === "load");
    const sites = [];
    const url = value => value?.type === "Identifier" && value.name === "url";
    walk.simple(load.value.body, {
      NewExpression(value) {
        if (node.id.name === "FileLoader" && value.callee.type === "Identifier" &&
            value.callee.name === "Request" && url(value.arguments[0])) sites.push(value.arguments[0]);
      },
      AssignmentExpression(value) {
        if (node.id.name === "ImageLoader" && value.left.type === "MemberExpression" &&
            !value.left.computed && value.left.object.name === "image" &&
            value.left.property.name === "src" && url(value.right)) sites.push(value.right);
      },
      CallExpression(value) {
        if (node.id.name === "ImageBitmapLoader" && value.callee.type === "Identifier" &&
            value.callee.name === "fetch" && url(value.arguments[0])) sites.push(value.arguments[0]);
      },
    });
    if (sites.length !== 1) throw new Error(`Pinned ${node.id.name} has an unexpected resource request shape`);
    edits.push({ type: "replace", pos: sites[0].start, end: sites[0].end,
      text: `${RESOLVE}(url)`, priority: 2 });
    imports.add(`resolveThreeLoaderResource as ${RESOLVE}`);
  }
  if (edits.length) edits.push({ type: "insert", pos: 0, priority: 0,
    text: `import { ${[...imports].join(", ")} } from ${JSON.stringify(LOADER_RESOURCES_IMPORT)};\n` });
  return edits;
}

const MIME = {
  ".json": "application/json", ".gltf": "model/gltf+json", ".glb": "model/gltf-binary",
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
  ".webp": "image/webp", ".avif": "image/avif", ".svg": "image/svg+xml",
  ".txt": "text/plain", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".wav": "audio/wav",
  ".mp4": "video/mp4", ".webm": "video/webm", ".ktx2": "image/ktx2",
  ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript",
};

/** Embed original bytes, including glTF JSON. Its sidecars resolve from the
 * original logical URL inside GLTFLoader, never from a data URL's fake base.
 * The bound includes encoded table bytes, also charged by packHtml as source.
 */
export function createThreeLoaderResourceModule(assets, { logicalURLs = null, maxBytes = 64 * 1024 * 1024 } = {}) {
  const entries = [];
  let bytes = 0;
  for (const [filename, asset] of [...assets].sort(([a], [b]) => a.localeCompare(b))) {
    const mime = MIME[path.extname(filename).toLowerCase()] ?? "application/octet-stream";
    const encoded = Buffer.from(asset.source).toString("base64");
    const urls = logicalURLs?.get(filename) ?? [filename.split("/").map(encodeURIComponent).join("/")];
    for (const url of [...urls].sort()) {
      const identity = mime === "application/wasm" ? "" : `;f3d-id=${hash(url)}`;
      const data = `data:${mime}${identity};base64,${encoded}`;
      bytes += Buffer.byteLength(data) + Buffer.byteLength(url);
      if (bytes > maxBytes) throw new Error("F3D_LOADER_RESOURCE_LIMIT: Embedded loader assets exceed the byte limit");
      entries.push([url, data]);
    }
  }
  return resourceModule(entries);
}

function resourceModule(entries) {
  return `// Finite source-identified loader assets; unknown URLs retain their network route.
const resources = new Map(${JSON.stringify(entries)});
const resourceRoot = typeof document === 'undefined' ? null : new URL('./', document.URL).href;
export function resolveThreeLoaderResource(url) {
  if (typeof url !== 'string' || resourceRoot === null) return url;
  let requested, key;
  try {
    requested = new URL(url, document.baseURI);
    if (!requested.href.startsWith(resourceRoot)) return url;
    key = requested.href.slice(resourceRoot.length);
  } catch { return url; }
  const embedded = resources.get(key);
  return embedded === undefined ? url : embedded + requested.hash;
}
// Host boundary emitted for exact pinned codec-loader constructor edits. Source
// owns its Blob worker body, messages, transfer lists, pools and URL lifetime.
// Passing the constructor first preserves its lookup before URL/options effects.
export function createThreeLoaderWorker(Worker, ...args) {
  return new Worker(...args);
}
`;
}

/** The packer may retain the native Blob-worker host call only in this exact
 * emitted runtime. A matching filename, export name or comment is insufficient;
 * extra statements, shadowed hosts and changed function bodies are refused.
 */
export function isThreeLoaderResourceModule(source, ast) {
  const declaration = ast.body[0]?.declarations?.[0];
  const entriesNode = declaration?.init?.arguments?.[0];
  if (declaration?.id?.name !== "resources" || entriesNode?.type !== "ArrayExpression") return false;
  try {
    const entries = JSON.parse(source.slice(entriesNode.start, entriesNode.end));
    if (!Array.isArray(entries) || entries.some(entry => !Array.isArray(entry) || entry.length !== 2 ||
        typeof entry[0] !== "string" || typeof entry[1] !== "string" || !entry[1].startsWith("data:"))) return false;
    return source === resourceModule(entries);
  } catch { return false; }
}
