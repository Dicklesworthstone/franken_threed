/**
 * Export an emitted HTML application as one portable HTML file. Modules remain
 * native ESM: an import map addresses embedded data modules, including cycles,
 * live bindings, lazy imports and distinct query/fragment module identities.
 * No eval, source execution, runtime archive loader or network fetch at build time.
 *
 * This closes the static module/markup/CSS/new-URL resource graph, not arbitrary
 * application networking or dynamically created DOM. Such host behavior remains
 * the application's responsibility. Unsupported static routes fail before output;
 * the ordinary multi-file application is never changed or removed.
 */

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as acorn from "acorn";
import * as walk from "acorn-walk";
import { embedGltfAssets } from "./pack_gltf.mjs";
import { createDynamicImportClassifier } from "./ast_analyzer.mjs";

export class HtmlPackingError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "HtmlPackingError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new HtmlPackingError(code, message);
};
const digest = (text) => createHash("sha256").update(text).digest("hex");
const quote = (text) =>
  String(text).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const text = (bytes) => new TextDecoder("utf-8", { fatal: true }).decode(bytes);
const dataUrl = (bytes, mime) => `data:${mime};base64,${Buffer.from(bytes).toString("base64")}`;
const MIME = {
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".otf": "font/otf",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".mp3": "audio/mpeg",
  ".ogg": "audio/ogg",
  ".wav": "audio/wav",
  ".txt": "text/plain",
  ".gltf": "model/gltf+json",
  ".glb": "model/gltf-binary",
};
const within = (root, file) => {
  const rel = path.relative(root, file);
  return rel !== ".." && !rel.startsWith(".." + path.sep) && !path.isAbsolute(rel);
};
function entities(value) {
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z][a-z0-9]+);/gi, (whole, key) => {
    if (key[0] === "#") {
      const n = key[1].toLowerCase() === "x" ? parseInt(key.slice(2), 16) : Number(key.slice(1));
      if (n > 0 && n <= 0x10ffff && !(n >= 0xd800 && n <= 0xdfff)) return String.fromCodePoint(n);
      fail("HTML_ENTITY", `Invalid character reference ${whole}`);
    }
    const known = { amp: "&", quot: '"', apos: "'", lt: "<", gt: ">" };
    if (!Object.hasOwn(known, key.toLowerCase()))
      fail("HTML_ENTITY", `Unsupported URL character reference ${whole}`);
    return known[key.toLowerCase()];
  });
}
const ATTR = /(?:^|\s+)([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
function attributes(raw) {
  const result = Object.create(null);
  for (const m of raw.matchAll(ATTR)) {
    const name = m[1].toLowerCase();
    if (!Object.hasOwn(result, name)) result[name] = m[2] ?? m[3] ?? m[4] ?? "";
  }
  return result;
}
function replaceAttributes(raw, changes) {
  const done = new Set();
  let output = raw.replace(ATTR, (whole, key) => {
    key = key.toLowerCase();
    if (!Object.hasOwn(changes, key)) return whole;
    if (done.has(key)) return ""; // HTML uses the first attribute; do not revive duplicates.
    done.add(key);
    return changes[key] === null ? "" : ` ${key}="${quote(changes[key])}"`;
  });
  for (const [key, value] of Object.entries(changes))
    if (!done.has(key) && value !== null) output += ` ${key}="${quote(value)}"`;
  return output;
}
// Contextual scanning: markup-looking strings inside raw text are not elements.
const TAGS =
  /<!--[\s\S]*?(?:-->|$)|<![^>]*>|<(script|style|textarea|title)\b((?:[^"'>]|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/\1\s*>|<([a-z][a-z0-9:-]*)\b((?:[^"'>]|"[^"]*"|'[^']*')*)>/gi;
const JS_TYPES = new Set([
  "",
  "text/javascript",
  "application/javascript",
  "text/ecmascript",
  "application/ecmascript",
]);
function edits(source, replacements) {
  replacements.sort((a, b) => b.start - a.start || b.end - a.end);
  let end = source.length;
  for (const item of replacements) {
    if (item.end > end) fail("OVERLAPPING_EDITS", "Overlapping source transformations");
    source = source.slice(0, item.start) + item.value + source.slice(item.end);
    end = item.start;
  }
  return source;
}

// Bounded abstract evaluation, never execution of application code. Besides
// literal choices, admit Rollup's emitted `(s => s === 'a' ? 'chunk-a' : ...)(choice)`
// dispatch expressions. The original expression still executes once at runtime.
function finiteImports(node, environment = new Map(), depth = 0, classify = null) {
  if (!node || depth > 64)
    fail("DYNAMIC_IMPORT_OPEN", "Dynamic import analysis exceeded its bound");
  if (node.type === "Literal" && typeof node.value === "string") return new Set([node.value]);
  if (node.type === "TemplateLiteral" && !node.expressions.length)
    return new Set([node.quasis[0].value.cooked]);
  if (node.type === "Identifier" && environment.has(node.name))
    return new Set([environment.get(node.name)]);
  if (node.type === "SequenceExpression")
    return finiteImports(node.expressions.at(-1), environment, depth + 1, classify);
  if (node.type === "ConditionalExpression") {
    const test = node.test;
    let known;
    if (test.type === "BinaryExpression" && ["===", "!=="].includes(test.operator)) {
      const atom = (value) =>
        value.type === "Identifier" && environment.has(value.name)
          ? environment.get(value.name)
          : value.type === "Literal" && typeof value.value === "string"
            ? value.value
            : undefined;
      const a = atom(test.left),
        b = atom(test.right);
      if (a !== undefined && b !== undefined) known = test.operator === "===" ? a === b : a !== b;
    }
    if (known !== undefined)
      return finiteImports(known ? node.consequent : node.alternate, environment, depth + 1, classify);
    const result = new Set([
      ...finiteImports(node.consequent, environment, depth + 1, classify),
      ...finiteImports(node.alternate, environment, depth + 1, classify),
    ]);
    if (result.size > 64) fail("DYNAMIC_IMPORT_OPEN", "Too many dynamic import targets");
    return result;
  }
  if (
    node.type === "CallExpression" &&
    !node.optional &&
    node.arguments.length === 1 &&
    node.callee.type === "ArrowFunctionExpression" &&
    !node.callee.async &&
    node.callee.expression &&
    node.callee.params.length === 1 &&
    node.callee.params[0].type === "Identifier"
  ) {
    const values = finiteImports(node.arguments[0], environment, depth + 1, classify),
      result = new Set();
    for (const value of values) {
      const nested = new Map(environment);
      nested.set(node.callee.params[0].name, value);
      for (const target of finiteImports(node.callee.body, nested, depth + 1, classify)) result.add(target);
      if (result.size > 64) fail("DYNAMIC_IMPORT_OPEN", "Too many mapped import targets");
    }
    return result;
  }
  const classified = classify?.(node);
  if (classified?.classification === "finite_set" && classified.candidates.length <= 64)
    return new Set(classified.candidates);
  fail("DYNAMIC_IMPORT_OPEN", "Dynamic import target is not a bounded literal expression");
}

function rewriteSrcset(source, embed) {
  // Follow the URL/descriptor separation used by HTML: a data URL's internal
  // comma is not a candidate separator, and descriptors stay browser-owned.
  const candidates = [];
  let offset = 0;
  while (offset < source.length) {
    while (offset < source.length && /[\t\n\f\r ,]/.test(source[offset])) offset++;
    if (offset === source.length) break;
    const start = offset;
    while (offset < source.length && !/[\t\n\f\r ]/.test(source[offset])) offset++;
    let url = source.slice(start, offset),
      descriptor = "";
    if (url.endsWith(",")) url = url.replace(/,+$/, "");
    else {
      const begin = offset;
      let parentheses = 0;
      while (offset < source.length) {
        const char = source[offset];
        if (char === "," && !parentheses) break;
        if (char === "(") parentheses++;
        else if (char === ")" && parentheses) parentheses--;
        offset++;
      }
      descriptor = source.slice(begin, offset).trim();
      if (source[offset] === ",") offset++;
    }
    if (!url) fail("SRCSET", "Empty responsive image candidate");
    candidates.push(embed(url) + (descriptor ? " " + descriptor : ""));
  }
  return candidates.join(", ");
}

/**
 * @param {string} entryPath HTML entry in an already-emitted application tree.
 * @param {string} outputPath Fresh single-file destination, never overwritten.
 * @param {{rootDir?: string, maxBytes?: number, maxFiles?: number}} options
 */
export function packHtml(
  entryPath,
  outputPath,
  {
    rootDir = path.dirname(path.resolve(entryPath)),
    maxBytes = 64 * 1024 * 1024,
    maxFiles = 4096,
  } = {},
) {
  if (
    !Number.isSafeInteger(maxBytes) ||
    maxBytes < 1 ||
    !Number.isSafeInteger(maxFiles) ||
    maxFiles < 1
  )
    throw new RangeError("Positive integer packing limits are required");
  const entry = path.resolve(entryPath),
    destination = path.resolve(outputPath),
    root = fs.realpathSync(rootDir);
  try {
    fs.lstatSync(destination);
    fail("OUTPUT_EXISTS", `Refusing to overwrite ${destination}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const documentUrl = pathToFileURL(entry).href;
  const files = new Map(),
    modules = new Map(),
    assets = new Map(),
    cssActive = new Set();
  const workers = new Map(),
    workerModules = new Map(),
    workerClassics = new Map(),
    workerActive = new Set();
  let inputBytes = 0, workerBytes = 0;
  function chargeWorker(encoded) {
    const size = Buffer.byteLength(encoded);
    if (size > maxBytes - workerBytes)
      fail("PACK_LIMIT", "Realm-expanded worker graphs exceed packing limit");
    workerBytes += size;
  }
  function read(url) {
    const parsed = new URL(url);
    if (parsed.protocol !== "file:") fail("EXTERNAL_RESOURCE", `A local file is required: ${url}`);
    const file = fileURLToPath(parsed);
    if (!within(root, file)) fail("ROOT_ESCAPE", `Resource escapes application root: ${url}`);
    const real = fs.realpathSync(file);
    if (!within(root, real)) fail("ROOT_ESCAPE", `Resource escapes application root: ${url}`);
    if (!files.has(real)) {
      if (!fs.statSync(real).isFile()) fail("NOT_A_FILE", `Resource is not a file: ${url}`);
      const size = fs.statSync(real).size;
      if (files.size >= maxFiles || inputBytes + size > maxBytes)
        fail("PACK_LIMIT", "Static resource graph exceeds packing limits");
      const bytes = fs.readFileSync(real);
      inputBytes += bytes.length;
      if (inputBytes > maxBytes)
        fail("PACK_LIMIT", "Static resources grew beyond the packing limit");
      files.set(real, bytes);
    }
    return files.get(real);
  }
  function local(value, base) {
    let url;
    try {
      url = new URL(value, base);
    } catch {
      fail("RESOURCE_URL", `Unresolvable resource ${JSON.stringify(value)}`);
    }
    if (url.protocol !== "file:" && url.protocol !== "data:")
      fail("EXTERNAL_RESOURCE", `Resource requires the normal networked build: ${url.href}`);
    return url.href;
  }
  const html = text(read(documentUrl));
  const tags = [...html.matchAll(TAGS)];
  let base = documentUrl,
    executableSeen = false;
  const sourceMaps = [],
    mapRanges = [];
  for (const tag of tags) {
    const name = (tag[1] ?? tag[4] ?? "").toLowerCase(),
      attrs = attributes(tag[2] ?? tag[5] ?? "");
    if (name === "base" && Object.hasOwn(attrs, "href")) {
      fail("BASE_URL", "Keep the normal build for documents with a base href");
    }
    if (
      name === "meta" &&
      entities(attrs["http-equiv"] ?? "").toLowerCase() === "content-security-policy"
    ) {
      fail(
        "CSP_POLICY",
        "A single-file export requires a separate CSP review; the source policy is not weakened",
      );
    }
    if (["template", "noscript"].includes(name))
      fail("INERT_MARKUP", "Conditional/inert markup requires the normal build in this exporter");
    if (["iframe", "frame", "object", "embed"].includes(name))
      fail("DOCUMENT_RESOURCE", `Nested ${name} documents require the normal build`);
    if (name !== "script") continue;
    const type = entities(attrs.type ?? "")
      .trim()
      .toLowerCase();
    if (type === "importmap") {
      if (executableSeen)
        fail("LATE_IMPORT_MAP", "Import maps after executable scripts require the normal build");
      if (attrs.src !== undefined) fail("IMPORT_MAP", "External import maps are not supported");
      let map;
      try {
        map = JSON.parse(tag[3]);
      } catch {
        fail("IMPORT_MAP", "Invalid import-map JSON");
      }
      if (!map || Array.isArray(map) || typeof map !== "object")
        fail("IMPORT_MAP", "Import map must be an object");
      if (map.integrity && Object.keys(map.integrity).length)
        fail("IMPORT_MAP_INTEGRITY", "Integrity-pinned import maps require the normal build");
      sourceMaps.push(map);
      mapRanges.push({ start: tag.index, end: tag.index + tag[0].length, value: "" });
    } else if (type === "module" || JS_TYPES.has(type)) executableSeen = true;
  }
  // Import maps are resolved in the source environment before any URLs change.
  const imports = new Map(),
    scopes = new Map();
  function merge(target, source) {
    if (source === undefined) return;
    if (!source || typeof source !== "object" || Array.isArray(source))
      fail("IMPORT_MAP", "Specifier mappings must be objects");
    for (let [key, value] of Object.entries(source)) {
      if (!key || (value !== null && typeof value !== "string"))
        fail("IMPORT_MAP", "Invalid import-map entry");
      if (/^(\.?\.?\/|[a-z][\w+.-]*:)/i.test(key)) key = new URL(key, base).href;
      if (value !== null) value = new URL(value, base).href;
      if (!target.has(key)) target.set(key, value); // Earlier import maps win.
    }
  }
  for (const map of sourceMaps) {
    merge(imports, map.imports);
    if (
      map.scopes !== undefined &&
      (!map.scopes || typeof map.scopes !== "object" || Array.isArray(map.scopes))
    )
      fail("IMPORT_MAP", "Scopes must be an object");
    for (const [scope, entries] of Object.entries(map.scopes ?? {})) {
      const url = new URL(scope, base).href;
      if (!scopes.has(url)) scopes.set(url, new Map());
      merge(scopes.get(url), entries);
    }
  }
  function resolveModule(specifier, from) {
    const urlLike = /^(\.?\.?\/|[a-z][\w+.-]*:)/i.test(specifier);
    const normalized = urlLike ? new URL(specifier, from).href : specifier;
    function match(map) {
      for (const key of [...map.keys()].sort((a, b) => b.length - a.length)) {
        if (key !== normalized && !(key.endsWith("/") && normalized.startsWith(key))) continue;
        const value = map.get(key);
        if (value === null) fail("BLOCKED_IMPORT", `Import map blocks ${specifier}`);
        if (key === normalized) return value;
        if (!value.endsWith("/")) fail("IMPORT_MAP", "Prefix mappings require a trailing slash");
        const mapped = new URL(normalized.slice(key.length), value).href;
        if (!mapped.startsWith(value))
          fail("IMPORT_MAP", "Mapped package import backtracks above its prefix");
        return mapped;
      }
      return null;
    }
    for (const [scope, map] of [...scopes].sort((a, b) => b[0].length - a[0].length)) {
      if (from === scope || (scope.endsWith("/") && from.startsWith(scope))) {
        const found = match(map);
        if (found) return found;
      }
    }
    const mapped = match(imports);
    if (mapped) return mapped;
    if (!urlLike)
      fail("UNRESOLVED_IMPORT", `Bare import ${JSON.stringify(specifier)} has no closed mapping`);
    return local(normalized, from);
  }
  function keyFor(url) {
    const value = new URL(url);
    const logical =
      value.protocol === "file:"
        ? path.relative(root, fileURLToPath(value)).split(path.sep).join("/") +
          value.search +
          value.hash
        : url;
    return "f3d-packed/" + digest(logical);
  }
  // Workers have their own module map, NOT the document's import map. Resolve
  // their static graph to absolute data module URLs before creating a same-origin
  // Blob entry. Native Worker owns messaging, transferable storage and lifetime.
  function moduleMime(url) {
    const mime = MIME[path.extname(new URL(url).pathname).toLowerCase()];
    if (!mime) fail("MODULE_MIME", `Module needs a known emitted MIME suffix: ${url}`);
    return mime;
  }
  function workerModuleFor(url, kind, worker) {
    const key = JSON.stringify([worker.kind, worker.entryUrl, kind, url]);
    if (workerActive.has(key))
      fail("WORKER_MODULE_CYCLE", "Bundle cyclic worker modules before single-file export");
    if (workerModules.has(key)) return workerModules.get(key);
    if (
      workerModules.size + workerClassics.size + modules.size >= maxFiles ||
      workerActive.size >= 64
    )
      fail("PACK_LIMIT", "Worker graph exceeds the identity/depth limit");
    if (url.startsWith("data:"))
      fail("DATA_MODULE", "Pre-encoded worker module sources require the normal build");
    workerActive.add(key);
    const original = read(url),
      record = { source: null, data: null, mime: moduleMime(url) };
    workerModules.set(key, record);
    try {
      // Requested type and response MIME are separate. Never turn a missing
      // JSON attribute or a JSON import of JavaScript into a successful load.
      // JSON and MIME-mismatched resources retain exact bytes and native errors.
      record.source = kind === "javascript" && JS_TYPES.has(record.mime)
        ? javascript(text(original), url, "module", worker) : original;
      record.data =
        dataUrl(record.source, record.mime) +
        "#" +
        digest(JSON.stringify([worker.kind, keyFor(worker.entryUrl), kind, keyFor(url)]));
      chargeWorker(record.data);
      return record;
    } finally {
      workerActive.delete(key);
    }
  }
  function workerClassicFor(url, worker) {
    // importScripts and fetch resolve relative to the worker's entry URL even
    // when the call lives in an imported classic script, NOT relative to that
    // imported script. The same source can therefore have several closures.
    const key = `classic:${worker.entryUrl}:${url}`;
    if (workerActive.has(key))
      fail("WORKER_SCRIPT_CYCLE", "Recursive classic imports require the normal build");
    if (workerClassics.has(key)) return workerClassics.get(key);
    if (
      workerModules.size + workerClassics.size + modules.size >= maxFiles ||
      workerActive.size >= 64
    )
      fail("PACK_LIMIT", "Worker graph exceeds the identity/depth limit");
    if (url.startsWith("data:"))
      fail("DATA_MODULE", "Pre-encoded worker scripts require the normal build");
    workerActive.add(key);
    try {
      const source = javascript(text(read(url)), url, "script", worker);
      const data =
        dataUrl(source, "text/javascript") + "#" + digest(keyFor(worker.entryUrl) + keyFor(url));
      chargeWorker(data);
      const record = { source, data };
      workerClassics.set(key, record);
      return record;
    } finally {
      workerActive.delete(key);
    }
  }
  function workerFactory(url, kind) {
    const key = kind + ":" + url;
    if (workers.has(key)) return workers.get(key);
    const active = "worker:" + key;
    if (workerActive.has(active))
      fail("WORKER_NESTING_CYCLE", "Cyclic worker construction requires the normal build");
    if (workerActive.size >= 64) fail("PACK_LIMIT", "Nested workers exceed the depth limit");
    const worker = {
      entryUrl: url, kind, factories: new Map(), names: new Set(),
      bindings: new Set(), hostUses: new Set(), preludeOffset: 0,
    };
    workerActive.add(active);
    let entry;
    try {
      entry = kind === "classic" ? workerClassicFor(url, worker) : workerModuleFor(url, "javascript", worker);
      for (const name of worker.hostUses)
        if (worker.bindings.has(name))
          fail("WORKER_BINDING", `Classic worker scripts rebind the host name ${name}`);
      if (worker.factories.size) {
        const declarations = [];
        for (const [name, expression] of worker.factories) {
          if (worker.names.has(name)) fail("WORKER_BINDING", "Classic worker factory name collision");
          declarations.push(`const ${name} = ${expression};`);
        }
        // Only the entry gets a prelude, after its original directive prologue.
        // importScripts remains repeatable and shares these private lexical
        // bindings without wrapping scripts or exposing new global properties.
        const offset = worker.preludeOffset;
        entry = { ...entry, source: entry.source.slice(0, offset) + "\n" +
          declarations.join("\n") + "\n" + entry.source.slice(offset) };
      }
    } finally { workerActive.delete(active); }
    if (typeof entry.source !== "string")
      fail("WORKER_MIME", "Module worker entry requires JavaScript MIME");
    // One cached URL per worker entry per creating realm, not one allocation per
    // constructor call. Do not revoke on terminate or pagehide: another worker,
    // a later invocation, or bfcache restoration may still need the same entry.
    // The File API releases this bounded URL table when the creating global dies.
    // Factory ESM scope prevents application locals from shadowing Blob or URL.
    const fields = `const source = ${JSON.stringify(entry.source)};\nlet cached;\n`;
    const getURL = `return cached ??= URL.createObjectURL(new Blob([source], {type:'text/javascript'}));`;
    const source = `// ${kind === "classic" ? "classic:" : ""}${keyFor(url)}\n` + fields +
      `export default function workerURL() {\n` +
      `  ${getURL}\n}\n`;
    const factory = dataUrl(source, "text/javascript");
    chargeWorker(factory);
    const record = { factory, classic: `(()=>{${fields}return function workerURL(){${getURL}};})()` };
    workers.set(key, record);
    return record;
  }
  function javascript(source, from, sourceType = "module", worker = false) {
    let ast;
    try {
      ast = acorn.parse(source, { ecmaVersion: "latest", sourceType });
    } catch (error) {
      fail("SCRIPT_PARSE", `${from}: ${error.message}`);
    }
    const replacements = [],
      allowedMeta = new Set(),
      workerURLs = new Set(),
      workerImports = [];
    const workerSites = [],
      names = new Set(),
      bindings = new Set();
    const literal = (node) =>
      node?.type === "Literal" && typeof node.value === "string"
        ? node.value
        : node?.type === "TemplateLiteral" && !node.expressions.length
          ? node.quasis[0].value.cooked
          : null;
    const add = (node, value) => replacements.push({ start: node.start, end: node.end, value });
    const result = (node) => {
      let depth = 0;
      while (node?.type === "SequenceExpression") {
        if (++depth > 64) fail("IMPORT_ATTRIBUTES", `${from}: attribute analysis exceeds its bound`);
        node = node.expressions.at(-1);
      }
      return node;
    };
    function importKind(node) {
      const invalid = () => fail("IMPORT_ATTRIBUTES", `${from}: imports need a statically known plain type dictionary`);
      function dictionary(value) {
        value = result(value);
        if (value?.type !== "ObjectExpression" || value.properties.some((p) =>
          p.type !== "Property" || p.computed || p.method || p.kind !== "init" ||
          (p.key.name ?? p.key.value) === "__proto__")) invalid();
        return value.properties;
      }
      let attributes = node.attributes ?? [];
      if (node.type === "ImportExpression") {
        if (!node.options) return "javascript";
        const options = dictionary(node.options);
        const maps = options.filter((p) => (p.key.name ?? p.key.value) === "with");
        if (maps.length > 1) invalid();
        attributes = maps.length ? dictionary(maps[0].value) : [];
      }
      const types = attributes.filter((p) => (p.key.name ?? p.key.value) === "type");
      if (!types.length) return "javascript";
      if (types.length !== 1 || literal(result(types[0].value)) !== "json") invalid();
      // All original options/attribute expressions remain, including side
      // effects and unsupported extra keys that the native importer rejects.
      return "json";
    }
    function bind(node) {
      if (!node) return;
      if (node.type === "Identifier") bindings.add(node.name);
      else if (node.type === "RestElement") bind(node.argument);
      else if (node.type === "AssignmentPattern") bind(node.left);
      else if (node.type === "ArrayPattern") node.elements.forEach(bind);
      else if (node.type === "ObjectPattern")
        node.properties.forEach((p) => bind(p.type === "RestElement" ? p.argument : p.value));
      else if (node.type === "MemberExpression" && node.object.type === "Identifier" &&
          ["globalThis", "self", "window"].includes(node.object.name)) {
        const name = node.computed ? literal(node.property) : node.property.name;
        if (name !== null) bindings.add(name);
        else ["Worker", "URL", "Blob", "fetch", "importScripts"].forEach((n) => bindings.add(n));
      }
    }
    function writtenHost(node) {
      // Module locals are isolated; direct writes to a classic realm's host
      // globals are not. Track writes even in ESM helpers of that realm.
      if (!node || worker?.kind !== "classic") return;
      if (node.type === "Identifier") {
        if (["Worker", "URL", "Blob", "fetch", "importScripts", "self", "globalThis"].includes(node.name))
          worker.bindings.add(node.name);
      } else if (node.type === "MemberExpression" &&
          ["globalThis", "self"].includes(node.object.name)) {
        const name = node.computed ? literal(node.property) : node.property.name;
        if (name !== null) worker.bindings.add(name);
        else ["Worker", "URL", "Blob", "fetch", "importScripts"].forEach((n) => worker.bindings.add(n));
      } else if (node.type === "ArrayPattern") node.elements.forEach(writtenHost);
      else if (node.type === "ObjectPattern") node.properties.forEach((p) => writtenHost(p.type === "RestElement" ? p.argument : p.value));
      else if (node.type === "RestElement") writtenHost(node.argument);
      else if (node.type === "AssignmentPattern") writtenHost(node.left);
    }
    walk.full(ast, (node) => {
      if (node.type === "Identifier") names.add(node.name);
      if (node.type === "VariableDeclarator") bind(node.id);
      if (/^(FunctionDeclaration|FunctionExpression|ArrowFunctionExpression)$/.test(node.type)) {
        bind(node.id);
        node.params.forEach(bind);
      }
      if (node.type === "ClassDeclaration" || node.type === "ClassExpression") bind(node.id);
      if (/^Import.*Specifier$/.test(node.type)) bind(node.local);
      if (node.type === "CatchClause") bind(node.param);
      if (node.type === "AssignmentExpression") { bind(node.left); writtenHost(node.left); }
      if (node.type === "UpdateExpression" || (node.type === "UnaryExpression" && node.operator === "delete")) {
        bind(node.argument); writtenHost(node.argument);
      }
      if (node.type === "ForInStatement" || node.type === "ForOfStatement") { bind(node.left); writtenHost(node.left); }
      if (
        node.type === "NewExpression" &&
        node.callee.type === "Identifier" &&
        node.callee.name === "Worker"
      )
        workerSites.push(node);
      if (
        worker &&
        ((node.type === "Identifier" && node.name === "location") ||
          (node.type === "MemberExpression" &&
            (node.property.name ?? node.property.value) === "location"))
      )
        fail("WORKER_LOCATION", "Worker location observation requires the normal build");
    });
    if (worker && sourceType === "script") {
      for (const name of names) worker.names.add(name);
      for (const name of bindings) worker.bindings.add(name);
      if (from === worker.entryUrl) {
        worker.preludeOffset = source.startsWith("#!") ? source.indexOf("\n") + 1 : 0;
        for (const node of ast.body) {
          if (node.type !== "ExpressionStatement" || node.directive === undefined) break;
          worker.preludeOffset = node.end;
        }
      }
    }
    function uses(...names) {
      if (worker?.kind === "classic") for (const name of names) worker.hostUses.add(name);
    }
    // Rewrite only finite URL result leaves. Conditions, comma-prefix effects,
    // native URL construction and caller options keep their execution ordering.
    function rewriteURLs(node, origin, emit, code, depth = 0, budget = { count: 0 }) {
      if (!node || depth > 64 || budget.count >= 64) fail(code, `${from}: URL choices exceed their bound`);
      const value = literal(node);
      if (value !== null) {
        if (!value) fail(code, `${from}: URL must be nonempty`);
        budget.count++;
        add(node, emit(value, origin));
      } else if (node.type === "SequenceExpression") {
        rewriteURLs(node.expressions.at(-1), origin, emit, code, depth + 1, budget);
      } else if (node.type === "ConditionalExpression") {
        rewriteURLs(node.consequent, origin, emit, code, depth + 1, budget);
        rewriteURLs(node.alternate, origin, emit, code, depth + 1, budget);
      } else if (node.type === "NewExpression" && node.callee.type === "Identifier" && node.callee.name === "URL") {
        const [ref, base] = node.arguments;
        if (bindings.has("URL") || node.arguments.length !== 2 ||
            base?.type !== "MemberExpression" || base.computed ||
            base.object.type !== "MetaProperty" || base.property.name !== "url")
          fail(code, `${from}: expected an unshadowed module-relative URL`);
        uses("URL");
        allowedMeta.add(base.object);
        workerURLs.add(node);
        rewriteURLs(ref, from, emit, code, depth + 1, budget);
      } else fail(code, `${from}: URL must have finite literal results`);
    }
    for (const site of workerSites) {
      if (!worker && sourceType !== "module")
        fail(
          "WORKER_PARENT",
          "Dedicated workers must be constructed by an emitted document module",
        );
      if (bindings.has("Worker") || bindings.has("URL"))
        fail("WORKER_BINDING", "Shadowed Worker/URL bindings require the normal build");
      if (site.arguments.length < 1 || site.arguments.length > 2)
        fail("WORKER_OPTIONS", "Worker requires a static URL and optional static type");
      const [argument, expression] = site.arguments;
      const options = result(expression);
      if (options && !(options.type === "Literal" && options.value === null) && options.type !== "ObjectExpression")
        fail("WORKER_OPTIONS", "Worker type must be a static option");
      let kind = "classic";
      for (const prop of options?.properties ?? []) {
        if (prop.type !== "Property" || prop.computed || prop.method || prop.kind !== "init" ||
            (prop.key.name ?? prop.key.value) === "__proto__")
          fail("WORKER_OPTIONS", "Worker options must have plain noncomputed properties");
        if ((prop.key.name ?? prop.key.value) === "type") kind = literal(result(prop.value));
      }
      if (!["classic", "module"].includes(kind))
        fail("WORKER_OPTIONS", "Explicit Worker options require a literal classic/module type");
      uses("Worker", "URL", "Blob");
      rewriteURLs(argument, worker ? worker.entryUrl : documentUrl, (value, origin) => {
        const childUrl = local(value, origin), record = workerFactory(childUrl, kind);
        if (sourceType === "script") {
          const name = "__f3d_worker_" + digest(kind + ":" + keyFor(childUrl));
          worker.factories.set(name, record.classic);
          return `${name}()`;
        }
        let name = `__f3d_packed_worker_${workerImports.length}`;
        while (names.has(name) || bindings.has(name)) name += "_";
        names.add(name);
        workerImports.push(`import ${name} from ${JSON.stringify(record.factory)};`);
        return `${name}()`;
      }, "WORKER_URL");
    }
    const importTarget = (value, kind, urlSemantics) => {
      if (!worker) {
        const record = moduleFor(urlSemantics ? local(value, from) : resolveModule(value, from), null, kind);
        // A URL constructor needs an absolute alias, not a bare map key. The
        // map resolves it after traversal, so document cycles still close.
        return urlSemantics ? (record.urlKey ??= "f3d-packed:" + record.key.slice(11)) : record.key;
      }
      // A page's import map must not give a bare worker import invented meaning.
      if (!urlSemantics && !/^(\.?\.?\/|[a-z][\w+.-]*:)/i.test(value))
        fail("WORKER_IMPORT", "Bare worker imports must be bundled before export");
      return workerModuleFor(local(value, from), kind, worker).data;
    };
    const classifyImport = createDynamicImportClassifier(ast);
    function importSource(node, kind, urlSemantics = false, depth = 0) {
      if (depth > 64) fail("DYNAMIC_IMPORT_OPEN", `${from}: import analysis exceeds its bound`);
      const value = literal(node);
      if (value !== null) {
        add(node, JSON.stringify(importTarget(value, kind, urlSemantics)));
        return;
      }
      if (node.type === "SequenceExpression") {
        importSource(node.expressions.at(-1), kind, urlSemantics, depth + 1);
        return;
      }
      if (node.type === "ConditionalExpression") {
        importSource(node.consequent, kind, urlSemantics, depth + 1);
        importSource(node.alternate, kind, urlSemantics, depth + 1);
        return;
      }
      if (node.type === "NewExpression" && node.callee.type === "Identifier" && node.callee.name === "URL") {
        const [ref, base] = node.arguments;
        if (bindings.has("URL") || node.arguments.length !== 2 ||
            base?.type !== "MemberExpression" || base.computed ||
            base.object.type !== "MetaProperty" || base.property.name !== "url")
          fail("IMPORT_URL", `${from}: imports need an unshadowed module-relative URL`);
        uses("URL");
        allowedMeta.add(base.object);
        // Rewrite result leaves to absolute data URLs (workers) or absolute
        // map aliases (documents), retaining construction and selector effects.
        importSource(ref, kind, true, depth + 1);
        return;
      }
      const targets = [...finiteImports(node, new Map(), 0, classifyImport)];
      const chain = targets
        .map(
          (target) =>
            `v === ${JSON.stringify(target)} ? ${JSON.stringify(importTarget(target, kind, urlSemantics))} : `,
        )
        .join("");
      // Insert wrappers rather than replace the expression: nested imports in
      // a selector retain their own edits and original evaluation ordering.
      replacements.push({ start: node.start, end: node.start, value: `((v) => ${chain}v)(` });
      replacements.push({ start: node.end, end: node.end, value: ")" });
    }
    // Mark executable URL operands before bottom-up traversal can mistake them
    // for ordinary binary assets. Do not mark URLs in selector side effects.
    function markImportURLs(node, depth = 0) {
      if (!node) return;
      if (depth > 64) fail("DYNAMIC_IMPORT_OPEN", `${from}: import analysis exceeds its bound`);
      if (node.type === "NewExpression" && node.callee.name === "URL") workerURLs.add(node);
      else if (node.type === "SequenceExpression") markImportURLs(node.expressions.at(-1), depth + 1);
      else if (node.type === "ConditionalExpression") {
        markImportURLs(node.consequent, depth + 1);
        markImportURLs(node.alternate, depth + 1);
      }
    }
    function hostCall(node, name) {
      return node.callee.type === "Identifier" && node.callee.name === name ||
        node.callee.type === "MemberExpression" && !node.callee.computed &&
        ["self", "globalThis"].includes(node.callee.object.name) && node.callee.property.name === name;
    }
    function hostBinding(node, name) {
      return bindings.has(name) || node.callee.type === "MemberExpression" && bindings.has(node.callee.object.name);
    }
    function useHost(node, name) {
      uses(name);
      if (node.callee.type === "MemberExpression") uses(node.callee.object.name);
    }
    const urlResult = (node) => node && (literal(node) !== null ||
      ["SequenceExpression", "ConditionalExpression"].includes(node.type) ||
      node.type === "NewExpression" && node.callee.name === "URL");
    walk.simple(ast, {
      ImportExpression(node) { markImportURLs(node.source); },
      CallExpression(node) {
        if (worker?.kind === "classic" && hostCall(node, "importScripts"))
          node.arguments.forEach((argument) => markImportURLs(argument));
        if (worker && hostCall(node, "fetch") && urlResult(node.arguments[0]))
          markImportURLs(node.arguments[0]);
      },
    });
    walk.simple(ast, {
      ImportDeclaration(node) {
        importSource(node.source, importKind(node));
      },
      ExportNamedDeclaration(node) {
        if (node.source) importSource(node.source, importKind(node));
      },
      ExportAllDeclaration(node) {
        importSource(node.source, importKind(node));
      },
      ImportExpression(node) {
        importSource(node.source, importKind(node));
      },
      NewExpression(node) {
        if (workerURLs.has(node) || workerSites.includes(node)) return;
        if (
          node.callee.type === "MemberExpression" &&
          ["Worker", "SharedWorker"].includes(node.callee.property.name)
        )
          fail("WORKER_BINDING", "Qualified worker constructors require the normal build");
        if (
          node.callee.type === "Identifier" &&
          ["Worker", "SharedWorker", "XMLHttpRequest", "WebSocket", "EventSource"].includes(
            node.callee.name,
          )
        ) {
          fail("HOST_RESOURCE", `${node.callee.name} needs the ordinary application build`);
        }
        if (node.callee.type !== "Identifier" || node.callee.name !== "URL") return;
        const [ref, origin] = node.arguments;
        const isMetaUrl =
          origin?.type === "MemberExpression" &&
          !origin.computed &&
          origin.object.type === "MetaProperty" &&
          origin.property.name === "url";
        if (!isMetaUrl) return;
        const value = literal(ref);
        if (value === null || node.arguments.length !== 2)
          fail("DYNAMIC_ASSET", `${from}: asset URL is not a static literal`);
        if (bindings.has("URL")) fail("WORKER_BINDING", `${from}: shadowed asset URL constructor`);
        uses("URL");
        allowedMeta.add(origin.object);
        add(node, `new URL(${JSON.stringify(assetFor(local(value, from)))})`);
      },
      CallExpression(node) {
        if (worker && hostCall(node, "importScripts")) {
          if (node.optional || hostBinding(node, "importScripts")) {
            fail("WORKER_BINDING", "Shadowed/optional importScripts requires the normal build");
          }
          // A module worker's native importScripts throws, even with no args.
          // Do not read its string targets or turn the call into an ESM import.
          if (worker.kind !== "classic") return;
          useHost(node, "importScripts");
          for (const argument of node.arguments) {
            rewriteURLs(argument, worker.entryUrl, (value, origin) =>
              JSON.stringify(workerClassicFor(local(value, origin), worker).data), "WORKER_SCRIPT_URL");
          }
          return; // Keep the native call, order, repeated evaluation and throws.
        }
        if (worker && hostCall(node, "fetch") && urlResult(node.arguments[0])) {
          const options = result(node.arguments[1]);
          if (node.arguments.length > 2 || node.optional || hostBinding(node, "fetch") ||
              options && !(options.type === "Literal" && options.value === null) &&
              (options.type !== "ObjectExpression" || options.properties.some((p) =>
                p.type !== "Property" || p.computed || p.method || p.kind !== "init" ||
                (p.key.name ?? p.key.value) === "__proto__")))
            fail("WORKER_FETCH", "Only unshadowed static GET worker fetches are packaged");
          const methods = options?.properties?.filter((p) => (p.key.name ?? p.key.value) === "method") ?? [];
          if (methods.length && literal(result(methods.at(-1).value))?.toUpperCase() !== "GET")
            fail("WORKER_FETCH", "Only static GET worker fetches are packaged");
          useHost(node, "fetch");
          rewriteURLs(node.arguments[0], worker.entryUrl, (value, origin) => {
            if (!value.startsWith("data:") && new URL(value, origin).protocol !== "file:")
              fail("FETCH_URL", "External worker fetches require the normal networked build");
            return JSON.stringify(assetFor(local(value, origin)));
          }, "WORKER_FETCH");
          return;
        }
        if (
          node.callee.type === "Identifier" &&
          ["eval", "importScripts"].includes(node.callee.name)
        )
          fail("DYNAMIC_CODE", "Dynamic code cannot establish a closed module graph");
        if (
          node.callee.type === "Identifier" &&
          node.callee.name === "fetch" &&
          literal(node.arguments[0]) !== null
        ) {
          const target = literal(node.arguments[0]);
          if (!target.startsWith("data:"))
            fail(
              "FETCH_URL",
              "Use a static new URL(..., import.meta.url) asset or retain the normal networked build",
            );
        }
      },
    });
    walk.simple(ast, {
      MetaProperty(node) {
        if (!allowedMeta.has(node))
          fail(
            "MODULE_URL_OBSERVATION",
            "Observable import.meta outside a static asset URL requires the normal build",
          );
      },
    });
    if (workerImports.length) {
      const offset = source.startsWith("#!") ? source.indexOf("\n") + 1 : 0;
      replacements.push({
        start: offset,
        end: offset,
        value: "\n" + workerImports.join("\n") + "\n",
      });
    }
    return edits(source, replacements);
  }
  function moduleFor(url, inline = null, kind = "javascript") {
    const identity = JSON.stringify([kind, url]);
    if (modules.has(identity)) return modules.get(identity);
    if (modules.size >= maxFiles) fail("PACK_LIMIT", "Module identity count exceeds packing limit");
    if (url.startsWith("data:"))
      fail("DATA_MODULE", "Pre-encoded module sources require the normal build");
    const record = {
      key: keyFor(url) + (kind === "json" ? "/json" : ""),
      source: null,
      original: inline === null ? read(url) : Buffer.from(inline),
      mime: inline === null ? moduleMime(url) : "text/javascript",
    };
    modules.set(identity, record); // Register before traversal, so cycles close by identity.
    record.source = kind === "javascript" && JS_TYPES.has(record.mime)
      ? javascript(text(record.original), url) : record.original;
    // The suffix prevents same-byte modules at different source URLs coalescing.
    record.data =
      dataUrl(record.source, record.mime) +
      "#" +
      record.key.slice("f3d-packed/".length);
    return record;
  }
  // CSS scanner skips comments and ordinary strings; url() in content text is not an asset.
  const CSS =
    /\/\*[\s\S]*?\*\/|@import\s+(?:url\(\s*)?(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s'"();]+))\s*\)?|\burl\(\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s'"()]+))\s*\)|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/gi;
  function css(source, from) {
    return source.replace(CSS, (whole, ia, ib, ic, ua, ub, uc) => {
      const raw = ia ?? ib ?? ic ?? ua ?? ub ?? uc;
      if (raw === undefined || raw.startsWith("#") || raw.startsWith("data:")) return whole;
      const value = raw.replace(/\\(?:([0-9a-f]{1,6})\s?|(.))/gi, (_m, hex, char) =>
        hex ? String.fromCodePoint(parseInt(hex, 16)) : char,
      );
      const isImport = /^@import/i.test(whole);
      return `${isImport ? "@import " : ""}url("${assetFor(local(value, from), isImport)}")`;
    });
  }
  function assetFor(url, stylesheet = false) {
    if (url.startsWith("data:")) return url;
    const key = (stylesheet ? "css:" : "asset:") + url;
    if (assets.has(key)) return assets.get(key).data;
    if (cssActive.has(key))
      fail(stylesheet ? "CSS_CYCLE" : "ASSET_CYCLE", "Cyclic resources require the ordinary build");
    const parsed = new URL(url),
      bytes = read(url),
      ext = path.extname(parsed.pathname).toLowerCase();
    if (ext === ".svg") {
      // These formats may contain their own external resource graphs. Retain
      // rather than claim a closed graph after embedding only the outer file.
      const source = text(bytes);
      if (ext === ".svg" && /(?:href\s*=\s*["'](?!#|data:)|url\(\s*(?!#|data:))/i.test(source))
        fail("NESTED_ASSET", "External SVG references need the normal build");
    }
    cssActive.add(key);
    let output;
    try {
      output =
        ext === ".gltf" || ext === ".glb"
          ? embedGltfAssets(bytes, ext === ".glb", (uri) => assetFor(local(uri, url)), fail)
          : stylesheet || ext === ".css"
            ? Buffer.from(css(text(bytes), url))
            : bytes;
    } finally {
      cssActive.delete(key);
    }
    // Distinct source URLs must not coalesce solely because their bytes match:
    // browsers associate image density with the selected srcset resource URL.
    // A MIME parameter retains identity without altering an SVG fragment target.
    const mime = stylesheet ? "text/css" : (MIME[ext] ?? "application/octet-stream");
    const identity = keyFor(url).slice("f3d-packed/".length);
    // Wasm streaming requires the exact application/wasm content type.
    const data =
      dataUrl(output, mime.startsWith("image/") ? mime + ";f3d-resource=" + identity : mime) +
      parsed.hash;
    assets.set(key, { data, original: bytes, output });
    return data;
  }
  function integrity(value, original, output) {
    const parts = value.trim().split(/\s+/);
    const supported = parts.map((part) => /^(sha256|sha384|sha512)-([^?]+)(?:\?.*)?$/.exec(part));
    if (supported.some((part) => !part))
      fail("INTEGRITY", "Unsupported integrity metadata is not removed");
    const strongest = supported.reduce((a, b) =>
      Number(a[1].slice(3)) >= Number(b[1].slice(3)) ? a : b,
    )[1];
    if (
      !supported.some(
        (part) =>
          part[1] === strongest &&
          createHash(strongest).update(original).digest("base64") === part[2],
      )
    )
      fail("INTEGRITY", "Source integrity verification failed");
    return supported
      .map((part) => `${part[1]}-${createHash(part[1]).update(output).digest("base64")}`)
      .join(" ");
  }
  const replacements = [...mapRanges];
  let firstScript = html.length,
    inlineId = 0;
  for (const tag of tags) {
    const name = (tag[1] ?? tag[4] ?? "").toLowerCase(),
      raw = tag[2] ?? tag[5] ?? "",
      attrs = attributes(raw),
      changes = {};
    if (!name || ["textarea", "title", "base"].includes(name)) continue;
    let body = tag[3],
      changedBody = false;
    const get = (key) => (attrs[key] === undefined ? null : entities(attrs[key]));
    const rewriteResource = (key, stylesheet = false, module = false) => {
      const value = get(key);
      if (value === null || value.startsWith("#")) return;
      if (value.startsWith("data:")) {
        if (module) fail("DATA_MODULE", "Pre-encoded module sources require the normal build");
        return;
      }
      if (!value)
        fail(
          "EMPTY_RESOURCE",
          `Empty ${name}.${key} keeps browser-specific error behavior; retain the normal build`,
        );
      const url = local(value, base);
      if (module) {
        const record = moduleFor(url);
        changes[key] = record.data;
        if (attrs.integrity)
          changes.integrity = integrity(
            get("integrity"),
            record.original,
            Buffer.from(record.source),
          );
      } else {
        changes[key] = assetFor(url, stylesheet);
        if (attrs.integrity) {
          const record = assets.get((stylesheet ? "css:" : "asset:") + url);
          changes.integrity = integrity(get("integrity"), record.original, record.output);
        }
      }
    };
    if (name === "script") {
      const type = (get("type") ?? "").trim().toLowerCase();
      if (type === "importmap") continue;
      if (type !== "module" && !JS_TYPES.has(type)) continue;
      firstScript = Math.min(firstScript, tag.index);
      if (type === "module") {
        if (attrs.src !== undefined) rewriteResource("src", false, true);
        else {
          const record = moduleFor(documentUrl + `#f3d-inline-${inlineId++}`, body);
          changes.src = record.data;
          body = "";
          changedBody = true;
        }
      } else if (attrs.src !== undefined) {
        const url = local(get("src"), base),
          original = read(url),
          transformed = javascript(text(original), url, "script");
        changes.src = dataUrl(transformed, "text/javascript");
        if (attrs.integrity)
          changes.integrity = integrity(get("integrity"), original, Buffer.from(transformed));
      } else {
        body = javascript(body, base, "script");
        changedBody = body !== tag[3];
      }
    } else if (name === "style") {
      body = css(body, base);
      changedBody = body !== tag[3];
    } else if (name === "link") {
      const rel = (get("rel") ?? "").toLowerCase().split(/\s+/);
      if (rel.includes("stylesheet")) rewriteResource("href", true);
      else if (rel.includes("modulepreload")) {
        firstScript = Math.min(firstScript, tag.index);
        rewriteResource("href", false, true);
      } else if (rel.some((value) => ["icon", "preload"].includes(value))) rewriteResource("href");
      else if (attrs.href !== undefined)
        fail("LINK_RESOURCE", `Unclosed link relation ${rel.join(" ")}`);
    } else if (["img", "audio", "video", "source", "track", "input"].includes(name)) {
      rewriteResource("src");
      rewriteResource("poster");
    }
    if (
      ["image", "use"].includes(name) &&
      (attrs.href !== undefined || attrs["xlink:href"] !== undefined)
    ) {
      const key = attrs.href !== undefined ? "href" : "xlink:href";
      if (!get(key).startsWith("#")) {
        if (name === "use")
          fail("SVG_REFERENCE", "External SVG use elements require the normal build");
        rewriteResource(key);
      }
    }
    if (attrs.srcset !== undefined)
      changes.srcset = rewriteSrcset(get("srcset"), (value) => assetFor(local(value, base)));
    if (attrs.style !== undefined) changes.style = css(get("style"), base);
    if (Object.keys(changes).length || changedBody) {
      const open = `<${tag[1] ?? tag[4]}${replaceAttributes(raw, changes)}>`;
      replacements.push({
        start: tag.index,
        end: tag.index + tag[0].length,
        value: tag[1] ? open + body + `</${tag[1]}>` : open,
      });
    }
  }
  // Emit one map before executable code. All identities are now known; circular
  // modules refer to stable keys, never recursively nested data URLs.
  const map = Object.fromEntries([...modules.values()].flatMap((record) =>
    [[record.key, record.data], ...(record.urlKey ? [[record.urlKey, record.data]] : [])]));
  const insertion = mapRanges.length ? Math.min(firstScript, mapRanges[0].start) : firstScript;
  if (modules.size)
    replacements.push({
      start: insertion,
      end: insertion,
      value: `<script type="importmap">${JSON.stringify({ imports: map }).replace(/</g, "\\u003c")}</script>\n`,
    });
  const output = edits(html, replacements);
  if (Buffer.byteLength(output) > maxBytes)
    fail("PACK_LIMIT", "Encoded HTML exceeds packing limit");
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, output, { flag: "wx" });
  return {
    entryPoint: entry,
    outputFile: destination,
    moduleCount: modules.size,
    assetCount: assets.size,
    sourceFileCount: files.size,
    inputBytes,
    outputBytes: Buffer.byteLength(output),
    ...(workers.size
      ? { workerCount: workers.size, workerScriptCount: workerModules.size + workerClassics.size }
      : {}),
    staticResourceClosure: "modules-markup-css-and-static-module-relative-assets",
    runtimeNetworking: "unchanged-not-analyzed",
  };
}
