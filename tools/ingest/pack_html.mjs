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
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';

export class HtmlPackingError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'HtmlPackingError'; this.code = code; }
}
const fail = (code, message) => { throw new HtmlPackingError(code, message); };
const digest = text => createHash('sha256').update(text).digest('hex');
const quote = text => String(text).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const text = bytes => new TextDecoder('utf-8', { fatal: true }).decode(bytes);
const dataUrl = (bytes, mime) => `data:${mime};base64,${Buffer.from(bytes).toString('base64')}`;
const MIME = {
  '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css',
  '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp', '.avif': 'image/avif', '.ico': 'image/x-icon',
  '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.otf': 'font/otf',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav',
  '.txt': 'text/plain', '.gltf': 'model/gltf+json', '.glb': 'model/gltf-binary',
};
const within = (root, file) => { const rel = path.relative(root, file); return rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel); };
function entities(value) {
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z][a-z0-9]+);/gi, (whole, key) => {
    if (key[0] === '#') {
      const n = key[1].toLowerCase() === 'x' ? parseInt(key.slice(2), 16) : Number(key.slice(1));
      if (n > 0 && n <= 0x10ffff && !(n >= 0xd800 && n <= 0xdfff)) return String.fromCodePoint(n);
      fail('HTML_ENTITY', `Invalid character reference ${whole}`);
    }
    const known = { amp: '&', quot: '"', apos: "'", lt: '<', gt: '>' };
    if (!Object.hasOwn(known, key.toLowerCase())) fail('HTML_ENTITY', `Unsupported URL character reference ${whole}`);
    return known[key.toLowerCase()];
  });
}
const ATTR = /(?:^|\s+)([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
function attributes(raw) {
  const result = Object.create(null);
  for (const m of raw.matchAll(ATTR)) {
    const name = m[1].toLowerCase();
    if (!Object.hasOwn(result, name)) result[name] = m[2] ?? m[3] ?? m[4] ?? '';
  }
  return result;
}
function replaceAttributes(raw, changes) {
  const done = new Set();
  let output = raw.replace(ATTR, (whole, key) => {
    key = key.toLowerCase();
    if (!Object.hasOwn(changes, key)) return whole;
    if (done.has(key)) return ''; // HTML uses the first attribute; do not revive duplicates.
    done.add(key);
    return changes[key] === null ? '' : ` ${key}="${quote(changes[key])}"`;
  });
  for (const [key, value] of Object.entries(changes)) if (!done.has(key) && value !== null) output += ` ${key}="${quote(value)}"`;
  return output;
}
// Contextual scanning: markup-looking strings inside raw text are not elements.
const TAGS = /<!--[\s\S]*?(?:-->|$)|<![^>]*>|<(script|style|textarea|title)\b((?:[^"'>]|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/\1\s*>|<([a-z][a-z0-9:-]*)\b((?:[^"'>]|"[^"]*"|'[^']*')*)>/gi;
const JS_TYPES = new Set(['', 'text/javascript', 'application/javascript', 'text/ecmascript', 'application/ecmascript']);
function edits(source, replacements) {
  replacements.sort((a, b) => b.start - a.start || b.end - a.end);
  let end = source.length;
  for (const item of replacements) {
    if (item.end > end) fail('OVERLAPPING_EDITS', 'Overlapping source transformations');
    source = source.slice(0, item.start) + item.value + source.slice(item.end); end = item.start;
  }
  return source;
}

/**
 * @param {string} entryPath HTML entry in an already-emitted application tree.
 * @param {string} outputPath Fresh single-file destination, never overwritten.
 * @param {{rootDir?: string, maxBytes?: number, maxFiles?: number}} options
 */
export function packHtml(entryPath, outputPath, { rootDir = path.dirname(path.resolve(entryPath)), maxBytes = 64 * 1024 * 1024, maxFiles = 4096 } = {}) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || !Number.isSafeInteger(maxFiles) || maxFiles < 1) throw new RangeError('Positive integer packing limits are required');
  const entry = path.resolve(entryPath), destination = path.resolve(outputPath), root = fs.realpathSync(rootDir);
  try { fs.lstatSync(destination); fail('OUTPUT_EXISTS', `Refusing to overwrite ${destination}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const documentUrl = pathToFileURL(entry).href;
  const files = new Map(), modules = new Map(), assets = new Map(), cssActive = new Set();
  let inputBytes = 0;
  function read(url) {
    const parsed = new URL(url);
    if (parsed.protocol !== 'file:') fail('EXTERNAL_RESOURCE', `A local file is required: ${url}`);
    const file = fileURLToPath(parsed);
    if (!within(root, file)) fail('ROOT_ESCAPE', `Resource escapes application root: ${url}`);
    const real = fs.realpathSync(file);
    if (!within(root, real)) fail('ROOT_ESCAPE', `Resource escapes application root: ${url}`);
    if (!files.has(real)) {
      if (!fs.statSync(real).isFile()) fail('NOT_A_FILE', `Resource is not a file: ${url}`);
      const size = fs.statSync(real).size;
      if (files.size >= maxFiles || inputBytes + size > maxBytes) fail('PACK_LIMIT', 'Static resource graph exceeds packing limits');
      const bytes = fs.readFileSync(real);
      inputBytes += bytes.length;
      if (inputBytes > maxBytes) fail('PACK_LIMIT', 'Static resources grew beyond the packing limit');
      files.set(real, bytes);
    }
    return files.get(real);
  }
  function local(value, base) {
    let url;
    try { url = new URL(value, base); } catch { fail('RESOURCE_URL', `Unresolvable resource ${JSON.stringify(value)}`); }
    if (url.protocol !== 'file:' && url.protocol !== 'data:') fail('EXTERNAL_RESOURCE', `Resource requires the normal networked build: ${url.href}`);
    return url.href;
  }
  const html = text(read(documentUrl));
  const tags = [...html.matchAll(TAGS)];
  let base = documentUrl, executableSeen = false;
  const sourceMaps = [], mapRanges = [];
  for (const tag of tags) {
    const name = (tag[1] ?? tag[4] ?? '').toLowerCase(), attrs = attributes(tag[2] ?? tag[5] ?? '');
    if (name === 'base' && Object.hasOwn(attrs, 'href')) {
      fail('BASE_URL', 'Keep the normal build for documents with a base href');
    }
    if (name === 'meta' && entities(attrs['http-equiv'] ?? '').toLowerCase() === 'content-security-policy') {
      fail('CSP_POLICY', 'A single-file export requires a separate CSP review; the source policy is not weakened');
    }
    if (['template', 'noscript'].includes(name)) fail('INERT_MARKUP', 'Conditional/inert markup requires the normal build in this exporter');
    if (['iframe', 'frame', 'object', 'embed'].includes(name)) fail('DOCUMENT_RESOURCE', `Nested ${name} documents require the normal build`);
    if (name !== 'script') continue;
    const type = entities(attrs.type ?? '').trim().toLowerCase();
    if (type === 'importmap') {
      if (executableSeen) fail('LATE_IMPORT_MAP', 'Import maps after executable scripts require the normal build');
      if (attrs.src !== undefined) fail('IMPORT_MAP', 'External import maps are not supported');
      let map;
      try { map = JSON.parse(tag[3]); } catch { fail('IMPORT_MAP', 'Invalid import-map JSON'); }
      if (!map || Array.isArray(map) || typeof map !== 'object') fail('IMPORT_MAP', 'Import map must be an object');
      if (map.integrity && Object.keys(map.integrity).length) fail('IMPORT_MAP_INTEGRITY', 'Integrity-pinned import maps require the normal build');
      sourceMaps.push(map); mapRanges.push({ start: tag.index, end: tag.index + tag[0].length, value: '' });
    } else if (type === 'module' || JS_TYPES.has(type)) executableSeen = true;
  }
  // Import maps are resolved in the source environment before any URLs change.
  const imports = new Map(), scopes = new Map();
  function merge(target, source) {
    if (source === undefined) return;
    if (!source || typeof source !== 'object' || Array.isArray(source)) fail('IMPORT_MAP', 'Specifier mappings must be objects');
    for (let [key, value] of Object.entries(source)) {
      if (!key || (value !== null && typeof value !== 'string')) fail('IMPORT_MAP', 'Invalid import-map entry');
      if (/^(\.?\.?\/|[a-z][\w+.-]*:)/i.test(key)) key = new URL(key, base).href;
      if (value !== null) value = new URL(value, base).href;
      if (!target.has(key)) target.set(key, value); // Earlier import maps win.
    }
  }
  for (const map of sourceMaps) {
    merge(imports, map.imports);
    if (map.scopes !== undefined && (!map.scopes || typeof map.scopes !== 'object' || Array.isArray(map.scopes))) fail('IMPORT_MAP', 'Scopes must be an object');
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
        if (key !== normalized && !(key.endsWith('/') && normalized.startsWith(key))) continue;
        const value = map.get(key);
        if (value === null) fail('BLOCKED_IMPORT', `Import map blocks ${specifier}`);
        if (key === normalized) return value;
        if (!value.endsWith('/')) fail('IMPORT_MAP', 'Prefix mappings require a trailing slash');
        const mapped = new URL(normalized.slice(key.length), value).href;
        if (!mapped.startsWith(value)) fail('IMPORT_MAP', 'Mapped package import backtracks above its prefix');
        return mapped;
      }
      return null;
    }
    for (const [scope, map] of [...scopes].sort((a, b) => b[0].length - a[0].length)) {
      if (from === scope || (scope.endsWith('/') && from.startsWith(scope))) { const found = match(map); if (found) return found; }
    }
    const mapped = match(imports);
    if (mapped) return mapped;
    if (!urlLike) fail('UNRESOLVED_IMPORT', `Bare import ${JSON.stringify(specifier)} has no closed mapping`);
    return local(normalized, from);
  }
  function keyFor(url) {
    const value = new URL(url);
    const logical = value.protocol === 'file:' ? path.relative(root, fileURLToPath(value)).split(path.sep).join('/') + value.search + value.hash : url;
    return 'f3d-packed/' + digest(logical);
  }
  function javascript(source, from, sourceType = 'module') {
    let ast;
    try { ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType }); }
    catch (error) { fail('SCRIPT_PARSE', `${from}: ${error.message}`); }
    const replacements = [], allowedMeta = new Set();
    const literal = node => node?.type === 'Literal' && typeof node.value === 'string' ? node.value
      : node?.type === 'TemplateLiteral' && !node.expressions.length ? node.quasis[0].value.cooked : null;
    const add = (node, value) => replacements.push({ start: node.start, end: node.end, value });
    function importSource(node) {
      const value = literal(node);
      if (value !== null) { add(node, JSON.stringify(moduleFor(resolveModule(value, from)).key)); return; }
      if (node.type === 'ConditionalExpression') { importSource(node.consequent); importSource(node.alternate); return; }
      fail('DYNAMIC_IMPORT_OPEN', `${from}: dynamic import target is not a finite literal set`);
    }
    walk.simple(ast, {
      ImportDeclaration(node) { importSource(node.source); },
      ExportNamedDeclaration(node) { if (node.source) importSource(node.source); },
      ExportAllDeclaration(node) { importSource(node.source); },
      ImportExpression(node) { importSource(node.source); },
      NewExpression(node) {
        if (node.callee.type === 'Identifier' && ['Worker', 'SharedWorker', 'XMLHttpRequest', 'WebSocket', 'EventSource'].includes(node.callee.name)) {
          fail('HOST_RESOURCE', `${node.callee.name} needs the ordinary application build`);
        }
        if (node.callee.type !== 'Identifier' || node.callee.name !== 'URL') return;
        const [ref, origin] = node.arguments;
        const isMetaUrl = origin?.type === 'MemberExpression' && !origin.computed && origin.object.type === 'MetaProperty' && origin.property.name === 'url';
        if (!isMetaUrl) return;
        const value = literal(ref);
        if (value === null || node.arguments.length !== 2) fail('DYNAMIC_ASSET', `${from}: asset URL is not a static literal`);
        allowedMeta.add(origin.object);
        add(node, `new URL(${JSON.stringify(assetFor(local(value, from)))})`);
      },
      CallExpression(node) {
        if (node.callee.type === 'Identifier' && ['eval', 'importScripts'].includes(node.callee.name)) fail('DYNAMIC_CODE', 'Dynamic code cannot establish a closed module graph');
        if (node.callee.type === 'Identifier' && node.callee.name === 'fetch' && literal(node.arguments[0]) !== null) {
          const target = literal(node.arguments[0]);
          if (!target.startsWith('data:')) fail('FETCH_URL', 'Use a static new URL(..., import.meta.url) asset or retain the normal networked build');
        }
      },
    });
    walk.simple(ast, { MetaProperty(node) { if (!allowedMeta.has(node)) fail('MODULE_URL_OBSERVATION', 'Observable import.meta outside a static asset URL requires the normal build'); } });
    return edits(source, replacements);
  }
  function moduleFor(url, inline = null) {
    if (modules.has(url)) return modules.get(url);
    if (modules.size >= maxFiles) fail('PACK_LIMIT', 'Module identity count exceeds packing limit');
    if (url.startsWith('data:')) fail('DATA_MODULE', 'Pre-encoded module sources require the normal build');
    const record = { key: keyFor(url), source: null, original: inline === null ? read(url) : Buffer.from(inline) };
    modules.set(url, record); // Register before traversal, so cycles close by identity.
    record.source = javascript(text(record.original), url);
    // The suffix prevents same-byte modules at different source URLs coalescing.
    record.data = dataUrl(record.source, 'text/javascript') + '#' + record.key.slice('f3d-packed/'.length);
    return record;
  }
  // CSS scanner skips comments and ordinary strings; url() in content text is not an asset.
  const CSS = /\/\*[\s\S]*?\*\/|@import\s+(?:url\(\s*)?(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s'"();]+))\s*\)?|\burl\(\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s'"()]+))\s*\)|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/gi;
  function css(source, from) {
    return source.replace(CSS, (whole, ia, ib, ic, ua, ub, uc) => {
      const raw = ia ?? ib ?? ic ?? ua ?? ub ?? uc;
      if (raw === undefined || raw.startsWith('#') || raw.startsWith('data:')) return whole;
      const value = raw.replace(/\\(?:([0-9a-f]{1,6})\s?|(.))/gi, (_m, hex, char) => hex ? String.fromCodePoint(parseInt(hex, 16)) : char);
      const isImport = /^@import/i.test(whole);
      return `${isImport ? '@import ' : ''}url("${assetFor(local(value, from), isImport)}")`;
    });
  }
  function assetFor(url, stylesheet = false) {
    if (url.startsWith('data:')) return url;
    const key = (stylesheet ? 'css:' : 'asset:') + url;
    if (assets.has(key)) return assets.get(key).data;
    if (cssActive.has(key)) fail('CSS_CYCLE', 'Cyclic stylesheets require the ordinary build');
    const parsed = new URL(url), bytes = read(url), ext = path.extname(parsed.pathname).toLowerCase();
    if (['.gltf', '.svg'].includes(ext)) {
      // These formats may contain their own external resource graphs. Retain
      // rather than claim a closed graph after embedding only the outer file.
      const source = text(bytes);
      if (ext === '.gltf' && /"uri"\s*:\s*"(?!data:)/.test(source)) fail('NESTED_ASSET', 'External glTF buffers/images need the normal build');
      if (ext === '.svg' && /(?:href\s*=\s*["'](?!#|data:)|url\(\s*(?!#|data:))/i.test(source)) fail('NESTED_ASSET', 'External SVG references need the normal build');
    }
    cssActive.add(key);
    let output;
    try { output = stylesheet || ext === '.css' ? Buffer.from(css(text(bytes), url)) : bytes; }
    finally { cssActive.delete(key); }
    const data = dataUrl(output, stylesheet ? 'text/css' : MIME[ext] ?? 'application/octet-stream') + parsed.hash;
    assets.set(key, { data, original: bytes, output });
    return data;
  }
  function integrity(value, original, output) {
    const parts = value.trim().split(/\s+/);
    const supported = parts.map(part => /^(sha256|sha384|sha512)-([^?]+)(?:\?.*)?$/.exec(part));
    if (supported.some(part => !part)) fail('INTEGRITY', 'Unsupported integrity metadata is not removed');
    const strongest = supported.reduce((a, b) => Number(a[1].slice(3)) >= Number(b[1].slice(3)) ? a : b)[1];
    if (!supported.some(part => part[1] === strongest && createHash(strongest).update(original).digest('base64') === part[2])) fail('INTEGRITY', 'Source integrity verification failed');
    return supported.map(part => `${part[1]}-${createHash(part[1]).update(output).digest('base64')}`).join(' ');
  }
  const replacements = [...mapRanges];
  let firstScript = html.length, inlineId = 0;
  for (const tag of tags) {
    const name = (tag[1] ?? tag[4] ?? '').toLowerCase(), raw = tag[2] ?? tag[5] ?? '', attrs = attributes(raw), changes = {};
    if (!name || ['textarea', 'title', 'base'].includes(name)) continue;
    let body = tag[3], changedBody = false;
    const get = key => attrs[key] === undefined ? null : entities(attrs[key]);
    const rewriteResource = (key, stylesheet = false, module = false) => {
      const value = get(key); if (value === null || value.startsWith('#')) return;
      if (value.startsWith('data:')) {
        if (module) fail('DATA_MODULE', 'Pre-encoded module sources require the normal build');
        return;
      }
      if (!value) fail('EMPTY_RESOURCE', `Empty ${name}.${key} keeps browser-specific error behavior; retain the normal build`);
      const url = local(value, base);
      if (module) {
        const record = moduleFor(url); changes[key] = record.data;
        if (attrs.integrity) changes.integrity = integrity(get('integrity'), record.original, Buffer.from(record.source));
      } else {
        changes[key] = assetFor(url, stylesheet);
        if (attrs.integrity) { const record = assets.get((stylesheet ? 'css:' : 'asset:') + url); changes.integrity = integrity(get('integrity'), record.original, record.output); }
      }
    };
    if (name === 'script') {
      const type = (get('type') ?? '').trim().toLowerCase();
      if (type === 'importmap') continue;
      if (type !== 'module' && !JS_TYPES.has(type)) continue;
      firstScript = Math.min(firstScript, tag.index);
      if (type === 'module') {
        if (attrs.src !== undefined) rewriteResource('src', false, true);
        else {
          const record = moduleFor(documentUrl + `#f3d-inline-${inlineId++}`, body);
          changes.src = record.data; body = ''; changedBody = true;
        }
      } else if (attrs.src !== undefined) {
        const url = local(get('src'), base), original = read(url), transformed = javascript(text(original), url, 'script');
        changes.src = dataUrl(transformed, 'text/javascript');
        if (attrs.integrity) changes.integrity = integrity(get('integrity'), original, Buffer.from(transformed));
      } else { body = javascript(body, base, 'script'); changedBody = body !== tag[3]; }
    } else if (name === 'style') { body = css(body, base); changedBody = body !== tag[3]; }
    else if (name === 'link') {
      const rel = (get('rel') ?? '').toLowerCase().split(/\s+/);
      if (rel.includes('stylesheet')) rewriteResource('href', true);
      else if (rel.includes('modulepreload')) { firstScript = Math.min(firstScript, tag.index); rewriteResource('href', false, true); }
      else if (rel.some(value => ['icon', 'preload'].includes(value))) rewriteResource('href');
      else if (attrs.href !== undefined) fail('LINK_RESOURCE', `Unclosed link relation ${rel.join(' ')}`);
    } else if (['img', 'audio', 'video', 'source', 'track', 'input'].includes(name)) {
      rewriteResource('src'); rewriteResource('poster');
    }
    if (['image', 'use'].includes(name) && (attrs.href !== undefined || attrs['xlink:href'] !== undefined)) {
      const key = attrs.href !== undefined ? 'href' : 'xlink:href';
      if (!get(key).startsWith('#')) {
        if (name === 'use') fail('SVG_REFERENCE', 'External SVG use elements require the normal build');
        rewriteResource(key);
      }
    }
    if (attrs.srcset !== undefined) fail('SRCSET', 'Responsive-image source sets require the normal build in this exporter');
    if (attrs.style !== undefined) changes.style = css(get('style'), base);
    if (Object.keys(changes).length || changedBody) {
      const open = `<${tag[1] ?? tag[4]}${replaceAttributes(raw, changes)}>`;
      replacements.push({ start: tag.index, end: tag.index + tag[0].length, value: tag[1] ? open + body + `</${tag[1]}>` : open });
    }
  }
  // Emit one map before executable code. All identities are now known; circular
  // modules refer to stable keys, never recursively nested data URLs.
  const map = Object.fromEntries([...modules.values()].map(record => [record.key, record.data]));
  const insertion = mapRanges.length ? Math.min(firstScript, mapRanges[0].start) : firstScript;
  if (modules.size) replacements.push({ start: insertion, end: insertion,
    value: `<script type="importmap">${JSON.stringify({ imports: map }).replace(/</g, '\\u003c')}</script>\n` });
  const output = edits(html, replacements);
  if (Buffer.byteLength(output) > maxBytes) fail('PACK_LIMIT', 'Encoded HTML exceeds packing limit');
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, output, { flag: 'wx' });
  return { entryPoint: entry, outputFile: destination, moduleCount: modules.size, assetCount: assets.size,
    sourceFileCount: files.size, inputBytes, outputBytes: Buffer.byteLength(output),
    staticResourceClosure: 'modules-markup-css-and-static-module-relative-assets', runtimeNetworking: 'unchanged-not-analyzed' };
}
