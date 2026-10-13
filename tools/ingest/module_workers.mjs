/** Relocate classic/module Worker and SharedWorker graphs without crossing realms.
 * Worker modules remain native ESM: cycles, live bindings and top-level await
 * are left to the host linker. No application import map, renderer substitution,
 * numeric specialization or worker execution here. Classic scripts stay classic:
 * importScripts retains synchronous ordering, repeated evaluation and globals.
 * Relative static GET fetches use the originating worker entry's URL, not an
 * importing module's directory. Realm-specific closures keep nested entry URLs
 * stable across discovering parents and bound expanded source/output bytes.
 * Rollup receives a content-addressed, bounded multi-file package as assets.
 */
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';

export class ModuleWorkerBuildError extends Error {
  constructor(code, message, url = null) {
    super(`F3D_WORKER_${code}: ${message}${url ? ` (${url})` : ''}`);
    this.name = 'ModuleWorkerBuildError'; this.code = `F3D_WORKER_${code}`; this.url = url;
  }
}
const fail = (code, message, url) => { throw new ModuleWorkerBuildError(code, message, url); };
const parse = (source, url, sourceType = 'module') => {
  try { return acorn.parse(source, {ecmaVersion: 'latest', sourceType}); }
  catch (cause) { throw new ModuleWorkerBuildError('PARSE', cause.message, url); }
};
const text = node => node?.type === 'Literal' && typeof node.value === 'string' ? node.value
  : node?.type === 'TemplateLiteral' && node.expressions.length === 0 ? node.quasis[0].value.cooked : null;
const metaUrl = node => node?.type === 'MemberExpression' && !node.computed && node.property.name === 'url' &&
  node.object.type === 'MetaProperty' && node.object.meta.name === 'import' && node.object.property.name === 'meta';
const urlArgument = node => node?.type === 'NewExpression' && node.callee.type === 'Identifier' &&
  node.callee.name === 'URL' && node.arguments.length === 2 && text(node.arguments[0]) !== null &&
  metaUrl(node.arguments[1]) ? node.arguments[0] : null;
const remote = url => ['https:', 'http:', 'data:'].includes(new URL(url).protocol);
const suffix = url => { const value = new URL(url); return value.search + value.hash; };

// Conservative admission: a binding or write anywhere in the module prevents
// treating that spelling as a host constructor. False negatives are preferable
// to bundling a user-defined constructor's URL as executable worker code.
function changedGlobals(ast) {
  const changed = new Set();
  function binding(node) {
    if (!node) return;
    if (node.type === 'Identifier') changed.add(node.name);
    else if (node.type === 'RestElement') binding(node.argument);
    else if (node.type === 'AssignmentPattern') binding(node.left);
    else if (node.type === 'ArrayPattern') node.elements.forEach(binding);
    else if (node.type === 'ObjectPattern') node.properties.forEach(p => binding(p.type === 'RestElement' ? p.argument : p.value));
    else if (node.type === 'MemberExpression' && node.object.type === 'Identifier' &&
      ['globalThis', 'self', 'window'].includes(node.object.name)) {
      const name = node.computed ? text(node.property) : node.property.name;
      if (name !== null) changed.add(name);
      else ['URL', 'Worker', 'SharedWorker', 'fetch', 'importScripts'].forEach(n => changed.add(n));
    }
  }
  walk.simple(ast, {
    VariableDeclarator: node => binding(node.id),
    Function: node => { binding(node.id); node.params.forEach(binding); },
    Class: node => binding(node.id),
    ImportDeclaration: node => node.specifiers.forEach(s => binding(s.local)),
    CatchClause: node => binding(node.param),
    AssignmentExpression: node => binding(node.left),
    UpdateExpression: node => binding(node.argument),
  });
  return changed;
}
function sitesIn(ast, changed, {classic = false, relative = false} = {}) {
  const sites = [];
  walk.simple(ast, {NewExpression(node) {
    const kind = node.callee.type === 'Identifier' ? node.callee.name : null;
    if (!['Worker', 'SharedWorker'].includes(kind) || changed.has(kind) ||
        node.arguments.length < 1 || node.arguments.length > 2) return;
    const urlNode = node.arguments[0], options = node.arguments[1];
    const url = !changed.has('URL') && urlArgument(urlNode);
    const argument = url || (relative && text(urlNode) !== null ? urlNode : null);
    if (!argument) return;
    let workerType = 'classic';
    if (options && !(options.type === 'Literal' && options.value === null)) {
      if (kind === 'SharedWorker' && text(options) !== null) {
        // The legacy string overload is a name, not a module-worker type.
      } else {
        if (options.type !== 'ObjectExpression' || options.properties.some(p => p.type !== 'Property' ||
            p.computed || p.kind !== 'init' || p.method || (p.key.name ?? p.key.value) === '__proto__')) return;
        const types = options.properties.filter(p => (p.key.name ?? p.key.value) === 'type');
        if (types.length > 1) return;
        if (types.length) workerType = text(types[0].value);
      }
    }
    if (!['classic', 'module'].includes(workerType) || (!classic && workerType !== 'module')) return;
    sites.push({kind, workerType, urlNode, argument, specifier: text(argument), base: url ? 'module' : 'realm'});
  }});
  return sites;
}

/** Backwards-compatible explicit module-URL analysis. */
export function findModuleWorkerSites(source) {
  const ast = typeof source === 'string' ? parse(source) : source;
  return sitesIn(ast, changedGlobals(ast));
}

/** URL-form workers in document ESM; literal URLs additionally inside workers. */
export function findWorkerSites(source, {sourceType = 'module', relative = false} = {}) {
  const ast = typeof source === 'string' ? parse(source, null, sourceType) : source;
  return sitesIn(ast, changedGlobals(ast), {classic: true, relative});
}

const realmKey = (url, workerType) => JSON.stringify([workerType, url]);

// Import attributes choose the host's module goal, not the file's spelling.
// Inspect only plain dictionaries; keep their original expressions so option
// getters/side effects are never executed by the build or moved before import().
function importSourceType(node, url) {
  const tail = value => {
    while (value?.type === 'SequenceExpression') value = value.expressions.at(-1);
    return value;
  };
  function property(value, name) {
    value = tail(value);
    if (!value || (value.type === 'UnaryExpression' && value.operator === 'void')) return null;
    if (value.type !== 'ObjectExpression' || value.properties.some(p => p.type !== 'Property' ||
        p.computed || p.kind !== 'init' || p.method || (p.key.name ?? p.key.value) === '__proto__'))
      fail('IMPORT_ATTRIBUTES', 'Local imports require statically known plain import-attribute dictionaries', url);
    const found = value.properties.filter(p => (p.key.name ?? p.key.value) === name);
    if (found.length > 1) fail('IMPORT_ATTRIBUTES', `Duplicate ${name} in import options`, url);
    return found[0]?.value ?? null;
  }
  const type = node.type === 'ImportExpression'
    ? property(property(node.options, 'with'), 'type')
    : node.attributes?.find(a => (a.key.name ?? a.key.value) === 'type')?.value;
  if (!type) return 'module';
  if (text(tail(type)) === 'json') return 'json';
  fail('IMPORT_ATTRIBUTES', 'Only a statically known type: "json" is supported for typed local worker imports', url);
}

// Browser-native resolution is the standalone default. The application builder
// supplies its installed-package resolver, with an EMPTY worker import map.
function resolveLocal(specifier, referrer) {
  if (!specifier.startsWith('./') && !specifier.startsWith('../') && !specifier.startsWith('/') && !URL.canParse(specifier))
    fail('RESOLUTION', `Bare worker import needs installed-package resolution: ${specifier}`, referrer);
  return new URL(specifier, referrer).href;
}

/** Plan everything before emitFile: no partial graph is emitted on rejection. */
export function buildModuleWorkerGraph(entryUrl, options = {}) {
  return buildWorkerGraph(entryUrl, {...options, workerType: 'module'});
}

/** Classic scripts and ESM are separate source kinds even at the same URL. */
export function buildWorkerGraph(entryUrl, {
  workerType = 'module',
  resolveModule = resolveLocal, maxModules = 4096, maxAssets = 4096, maxBytes = 64 * 1024 * 1024,
} = {}) {
  if (!['classic', 'module'].includes(workerType)) fail('OPTIONS', 'Expected classic or module workerType');
  if (typeof resolveModule !== 'function' || ![maxModules, maxAssets, maxBytes].every(n => Number.isSafeInteger(n) && n > 0))
    fail('OPTIONS', 'Expected a synchronous resolver and positive graph limits');
  const modules = new Map(), assets = new Map(), queue = [], inputs = new Map(), roots = new Map();
  const digest = value => createHash('sha256').update(value).digest('hex');
  let inputBytes = 0, analyzedBytes = 0;
  const normalize = url => {
    try { return new URL(url).href; }
    catch { fail('RESOLUTION', 'Expected an absolute module URL', String(url)); }
  };
  function module(url, realm, sourceType = 'module') {
    url = normalize(url);
    if (remote(url)) return {external: url};
    if (new URL(url).protocol !== 'file:') fail('RESOLUTION', 'Only local files and explicit http/https/data imports are supported', url);
    const key = JSON.stringify([realm, sourceType, url]);
    let record = modules.get(key);
    if (!record) {
      if (modules.size >= maxModules) fail('BUDGET', 'Worker module count exceeds maxModules', url);
      const name = sourceType === 'script' ? `c-${digest(url)}.js`
        : sourceType === 'json' ? `j-${digest(url)}.json` : `m-${digest(url)}.mjs`;
      record = {key, url, realm, sourceType, name, edges: new Set(), hostUses: new Set()};
      modules.set(key, record); queue.push(record);
    }
    return record;
  }
  function worker(url, type) {
    url = normalize(url);
    const realm = realmKey(url, type);
    const record = module(url, realm, type === 'classic' ? 'script' : 'module');
    if (!record.external && !roots.has(realm)) {
      record.workerType = type;
      roots.set(realm, record);
    }
    return record;
  }
  function read(url) {
    let filename, stat, bytes;
    try {
      filename = fileURLToPath(url);
      if (inputs.has(filename)) return inputs.get(filename);
      stat = fs.statSync(filename);
      if (!stat.isFile()) fail('RESOURCE', 'Worker resources must be regular files', url);
      if (stat.size > maxBytes - inputBytes) fail('BUDGET', 'Worker inputs exceed maxBytes', url);
      bytes = fs.readFileSync(filename);
    } catch (cause) {
      if (cause instanceof ModuleWorkerBuildError) throw cause;
      fail('RESOURCE', cause.message, url);
    }
    inputBytes += bytes.length;
    if (inputBytes > maxBytes) fail('BUDGET', 'Worker inputs exceed maxBytes', url);
    inputs.set(filename, bytes);
    return bytes;
  }
  function asset(url) {
    url = normalize(url);
    if (remote(url)) return {external: url};
    if (new URL(url).protocol !== 'file:') fail('RESOURCE', 'Unsupported worker asset URL', url);
    const plain = new URL(url); plain.search = ''; plain.hash = '';
    let record = assets.get(plain.href);
    if (!record) {
      if (assets.size >= maxAssets) fail('BUDGET', 'Worker asset count exceeds maxAssets', url);
      const extension = path.extname(fileURLToPath(plain)).replace(/[^.a-zA-Z0-9_-]/g, '_').slice(0, 32);
      const source = read(plain.href), key = JSON.stringify(['asset', plain.href]);
      const signature = digest(JSON.stringify([key, digest(source)]));
      record = {key, url: plain.href, source, signature, edges: new Set(),
        fileName: `f3d-workers/assets/${signature}/resource${extension}`};
      assets.set(plain.href, record);
    }
    // References retain their suffix without cloning the mutable target record.
    return {asset: record, url};
  }
  function leaves(node, result = []) {
    if (result.length >= 128) return null;
    if (text(node) !== null) { result.push(node); return result; }
    if (node?.type !== 'ConditionalExpression') return null;
    return leaves(node.consequent, result) && leaves(node.alternate, result);
  }
  function staticGet(options) {
    if (!options || (options.type === 'Literal' && options.value === null)) return true;
    if (options.type !== 'ObjectExpression' || options.properties.some(p => p.type !== 'Property' ||
        p.computed || p.kind !== 'init' || p.method || (p.key.name ?? p.key.value) === '__proto__')) return false;
    const methods = options.properties.filter(p => (p.key.name ?? p.key.value) === 'method');
    return methods.length === 0 || (methods.length === 1 && text(methods[0].value)?.toUpperCase() === 'GET');
  }
  const entry = worker(entryUrl, workerType);
  if (entry.external) fail('ENTRY', 'Worker graph entry must be a local file', entryUrl);
  for (let index = 0; index < queue.length; index++) {
    const record = queue[index], bytes = read(record.url);
    analyzedBytes += bytes.length;
    if (analyzedBytes > maxBytes) fail('BUDGET', 'Realm-expanded worker sources exceed maxBytes', record.url);
    if (record.sourceType === 'json') {
      // JSON is a native module, not a JavaScript object-literal wrapper. Keep
      // bytes (including BOM, duplicate keys and invalid lazy input) untouched;
      // parsing, default-export identity and rejection timing belong to the host.
      record.source = bytes;
      record.signature = digest(JSON.stringify([record.key, digest(bytes)]));
      continue;
    }
    const source = bytes.toString('utf8');
    const ast = parse(source, record.url, record.sourceType), changed = changedGlobals(ast), edits = [];
    const root = roots.get(record.realm);
    record.changed = changed;
    function replace(node, target, fetch = false) {
      const actual = target.asset ?? target;
      if (!actual.external) record.edges.add(actual);
      edits.push({start: node.start, end: node.end, target: actual, suffix: target.external ? '' : suffix(target.url), fetch});
    }
    function dependency(node, declaration) {
      let resolved;
      try { resolved = resolveModule(text(node), record.url); }
      catch (cause) { fail('RESOLUTION', cause.message, record.url); }
      if (typeof resolved !== 'string') fail('RESOLUTION', 'Worker resolver must return a URL synchronously', record.url);
      // Remote modules keep host-owned attribute handling and require no local
      // format decision, including runtime-computed options.
      const sourceType = remote(resolved) ? 'module' : importSourceType(declaration, record.url);
      replace(node, module(resolved, record.realm, sourceType));
    }
    function uses(name, callee = null) {
      record.hostUses.add(name);
      if (callee?.type === 'MemberExpression') record.hostUses.add(callee.object.name);
    }
    function hostCall(callee, name) {
      if (changed.has(name)) return false;
      return (callee.type === 'Identifier' && callee.name === name) ||
        (callee.type === 'MemberExpression' && !callee.computed && callee.property.name === name &&
          callee.object.type === 'Identifier' && ['globalThis', 'self'].includes(callee.object.name) && !changed.has(callee.object.name));
    }
    const workers = sitesIn(ast, changed, {classic: true, relative: true});
    const workerURLs = new Set(workers.map(site => site.urlNode.start));
    for (const site of workers) {
      if (site.specifier.trim().startsWith('/')) continue;
      const url = new URL(site.specifier, site.base === 'realm' ? root.url : record.url).href;
      if (new URL(url).protocol !== 'file:') continue;
      replace(site.argument, worker(url, site.workerType), site.base === 'realm');
      uses(site.kind); if (site.base === 'module') uses('URL');
    }
    walk.simple(ast, {
      ImportDeclaration: node => dependency(node.source, node),
      ExportNamedDeclaration: node => { if (node.source) dependency(node.source, node); },
      ExportAllDeclaration: node => dependency(node.source, node),
      ImportExpression(node) {
        const candidates = leaves(node.source);
        if (!candidates) fail('DYNAMIC_IMPORT', 'Use literal imports or conditionals with literal module branches in a relocated worker', record.url);
        candidates.forEach(candidate => dependency(candidate, node));
      },
      CallExpression(node) {
        if (hostCall(node.callee, 'importScripts')) {
          // The native method throws in a module worker, including zero args.
          // Do not fetch a missing file or turn this into a successful import.
          if (root.workerType !== 'classic') return;
          for (const argument of node.arguments) {
            const candidates = leaves(argument);
            if (!candidates) fail('IMPORT_SCRIPTS', 'Classic worker imports require literal or conditional script URLs', record.url);
            for (const candidate of candidates) {
              const value = text(candidate);
              if (value.trim().startsWith('/')) continue;
              const url = new URL(value, root.url).href;
              if (new URL(url).protocol !== 'file:') continue;
              replace(candidate, module(url, record.realm, 'script'), true);
              uses('importScripts', node.callee);
            }
          }
          return;
        }
        if (!hostCall(node.callee, 'fetch')) return;
        const candidates = leaves(node.arguments[0]);
        if (!candidates) return; // Dynamic networking is not claimed closed.
        for (const candidate of candidates) {
          const value = text(candidate);
          if (value.trim().startsWith('/') || URL.canParse(value)) continue;
          if (!staticGet(node.arguments[1])) fail('FETCH', 'Relative worker resource fetches require a statically known GET request', record.url);
          // fetch()/importScripts() use the worker entry, NOT an imported script.
          replace(candidate, asset(new URL(value, root.url).href), true);
          uses('fetch', node.callee);
        }
      },
      NewExpression(node) {
        if (changed.has('URL') || workerURLs.has(node.start)) return;
        const argument = urlArgument(node);
        if (!argument) return;
        const value = text(argument);
        if (value.trim().startsWith('/')) return;
        const url = new URL(value, record.url).href;
        if (new URL(url).protocol !== 'file:') return;
        replace(argument, asset(url)); uses('URL');
      },
    });
    record.original = source; record.edits = edits.sort((a, b) => b.start - a.start);
    // Hash symbolic links, not final output URLs, so mutually spawning workers
    // have a finite content address and SharedWorker URLs don't depend on who
    // happened to discover their graph first.
    record.signature = digest(JSON.stringify([record.key, source, record.edits.map(e =>
      [e.start, e.end, e.target.external ?? e.target.key, e.suffix, e.fetch])]));
  }
  // Classic scripts share a global lexical/var environment. Conservatively reject
  // cross-script host-name rebinding rather than relocate a custom loader's args.
  // Module-local bindings do not enter this shared environment.
  const classicChanges = new Map();
  for (const record of modules.values()) {
    if (record.sourceType !== 'script') continue;
    if (!classicChanges.has(record.realm)) classicChanges.set(record.realm, new Set());
    for (const name of record.changed) classicChanges.get(record.realm).add(name);
  }
  for (const record of modules.values()) {
    const changes = classicChanges.get(record.realm);
    for (const name of record.hostUses) if (changes?.has(name))
      fail('BINDING', `Classic worker scripts rebind the host name ${name}`, record.url);
  }
  let visits = 0;
  const maxVisits = Math.min(Number.MAX_SAFE_INTEGER, 16 * (maxModules + maxAssets));
  const directories = new Map();
  for (const [url, root] of roots) {
    const reached = new Set(), pending = [root];
    for (let i = 0; i < pending.length; i++) {
      const record = pending[i];
      if (reached.has(record)) continue;
      if (++visits > maxVisits) fail('BUDGET', 'Nested worker closure analysis exceeds its work budget', url);
      reached.add(record);
      for (const next of record.edges) if (!reached.has(next)) pending.push(next);
    }
    const hash = createHash('sha256').update('f3d-worker-realms-v2\0').update(url);
    for (const record of [...reached].sort((a, b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0))
      hash.update(record.signature);
    directories.set(url, `f3d-workers/${hash.digest('hex')}`);
  }
  for (const record of modules.values()) record.fileName = `${directories.get(record.realm)}/${record.name}`;
  let outputBytes = [...assets.values()].reduce((n, record) => n + record.source.length, 0);
  for (const record of modules.values()) {
    if (record.sourceType === 'json') {
      outputBytes += record.source.length;
      if (outputBytes > maxBytes) fail('BUDGET', 'Relocated worker output exceeds maxBytes', entryUrl);
      continue;
    }
    const replacements = []; let bytes = Buffer.byteLength(record.original), end = record.original.length;
    for (const edit of record.edits) {
      if (edit.end > end) fail('TRANSFORM', 'Overlapping worker source rewrites', record.url);
      let value = edit.target.external;
      if (value === undefined) {
        const from = edit.fetch ? roots.get(record.realm).fileName : record.fileName;
        const relative = path.posix.relative(path.posix.dirname(from), edit.target.fileName);
        value = (relative.startsWith('.') ? relative : './' + relative) + edit.suffix;
      }
      value = JSON.stringify(value);
      bytes += Buffer.byteLength(value) - Buffer.byteLength(record.original.slice(edit.start, edit.end));
      replacements.push({...edit, value}); end = edit.start;
    }
    outputBytes += bytes;
    if (outputBytes > maxBytes) fail('BUDGET', 'Relocated worker output exceeds maxBytes', entryUrl);
    const parts = []; let offset = 0;
    for (const edit of replacements.reverse()) {
      parts.push(record.original.slice(offset, edit.start), edit.value); offset = edit.end;
    }
    parts.push(record.original.slice(offset)); record.source = parts.join('');
  }
  const files = [...modules.values(), ...assets.values()].map(record => ({
    type: 'asset', fileName: record.fileName, source: record.source,
  }));
  return {entryFile: entry.fileName, files, moduleCount: modules.size, assetCount: assets.size, outputBytes,
    watchFiles: [...inputs.keys()], entries: [...roots.values()].map(root => ({url: root.url, workerType: root.workerType, fileName: root.fileName}))};
}

/** Run before the ordinary URL-asset transform in bundleWithRollup. */
export function moduleWorkerRollupPlugin(options = {}) {
  const emitted = new Map(), references = new Map(), filesByName = new Map();
  let totalBytes = 0, totalModules = 0, totalAssets = 0;
  const maxBytes = options.maxBytes ?? 64 * 1024 * 1024;
  const maxModules = options.maxModules ?? 4096, maxAssets = options.maxAssets ?? 4096;
  return {
    name: 'f3d-module-workers',
    buildStart() { emitted.clear(); references.clear(); filesByName.clear(); totalBytes = totalModules = totalAssets = 0; },
    resolveFileUrl({referenceId, relativePath}) {
      if (!references.has(referenceId)) return null;
      const relative = relativePath.startsWith('.') ? relativePath : './' + relativePath;
      return JSON.stringify(relative + references.get(referenceId));
    },
    transform(code, id) {
      if (!code.includes('Worker')) return null;
      const sites = findWorkerSites(code);
      if (!sites.length) return null;
      const base = id.startsWith('file:') ? id : path.isAbsolute(id) ? pathToFileURL(id).href : null;
      if (!base) fail('ENTRY', 'A module worker URL needs a local source module', id);
      const edits = [];
      for (const site of sites) {
        // Origin-root and protocol-relative workers remain host-owned URLs.
        if (site.specifier.trim().startsWith('/')) continue;
        const url = new URL(site.specifier, base).href;
        if (new URL(url).protocol !== 'file:') continue;
        const key = realmKey(url, site.workerType);
        let referenceId = emitted.get(key);
        if (!referenceId) {
          const graph = buildWorkerGraph(url, {...options, workerType: site.workerType});
          if (graph.outputBytes > maxBytes - totalBytes || graph.moduleCount > maxModules - totalModules || graph.assetCount > maxAssets - totalAssets)
            fail('BUDGET', 'Module worker graphs exceed the shared build budget', url);
          for (const filename of graph.watchFiles) this.addWatchFile?.(filename);
          for (const file of graph.files) {
            let ref = filesByName.get(file.fileName);
            if (!ref) { ref = this.emitFile(file); filesByName.set(file.fileName, ref); }
          }
          for (const entry of graph.entries) {
            const ref = filesByName.get(entry.fileName);
            emitted.set(realmKey(entry.url, entry.workerType), ref); references.set(ref, suffix(entry.url));
          }
          referenceId = emitted.get(key);
          totalBytes += graph.outputBytes; totalModules += graph.moduleCount; totalAssets += graph.assetCount;
        }
        edits.push({start: site.urlNode.start, end: site.urlNode.end,
          text: `new URL(import.meta.ROLLUP_FILE_URL_${referenceId}, import.meta.url)`});
      }
      if (!edits.length) return null;
      let output = code;
      for (const edit of edits.sort((a, b) => b.start - a.start))
        output = output.slice(0, edit.start) + edit.text + output.slice(edit.end);
      return {code: output, map: null};
    },
  };
}
