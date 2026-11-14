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
import {moduleSourcesFor} from './module_sources.mjs';
import {assertUnbundledModule} from './commonjs_modules.mjs';
import {createDynamicImportClassifier} from './ast_analyzer.mjs';
import {isRootRelativeDocumentUrl, normalizeDocumentRoot, resolveDocumentUrl} from './document_urls.mjs';

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
const urlOperand = node => node?.type === 'NewExpression' && node.callee.type === 'Identifier' &&
  node.callee.name === 'URL' && node.arguments.length === 2 &&
  metaUrl(node.arguments[1]) ? node.arguments[0] : null;
const remote = url => ['https:', 'http:', 'data:'].includes(new URL(url).protocol);
const suffix = url => { const value = new URL(url); return value.search + value.hash; };
const hostRootUrl = (value, documentRoot) => value.trim().startsWith('/') &&
  !(documentRoot && isRootRelativeDocumentUrl(value));
const localFileExists = url => {
  try { return fs.statSync(fileURLToPath(url)).isFile(); }
  catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return false;
    throw error;
  }
};

// Select only result positions. Conditions and discarded sequence operands stay
// in place and execute exactly once; a partial set of alternatives is never used.
function literalChoices(node, result = [], depth = 0) {
  if (!node || depth > 64) return null;
  if (text(node) !== null) {
    if (result.length >= 128) return null;
    result.push(node); return result;
  }
  if (node.type === 'SequenceExpression') return literalChoices(node.expressions.at(-1), result, depth + 1);
  if (node.type !== 'ConditionalExpression') return null;
  return literalChoices(node.consequent, result, depth + 1) && literalChoices(node.alternate, result, depth + 1);
}

// Immutable strings can have several runtime values. Map the actual operand,
// never its declaration: other reads and initialization/TDZ effects are source-owned.
function workerStringChoices(node, classify, result = [], depth = 0) {
  if (!node || depth > 64) return null;
  if (text(node) !== null) {
    if (result.length >= 128) return null;
    result.push({argument: node, specifier: text(node)}); return result;
  }
  if (node.type === 'SequenceExpression')
    return workerStringChoices(node.expressions.at(-1), classify, result, depth + 1);
  if (node.type === 'ConditionalExpression')
    return workerStringChoices(node.consequent, classify, result, depth + 1) &&
      workerStringChoices(node.alternate, classify, result, depth + 1);
  const classified = classify(node);
  if (classified.classification !== 'finite_set' || result.length + classified.candidates.length > 128) return null;
  for (const specifier of classified.candidates)
    result.push({argument: node, specifier, selector: specifier});
  return result;
}

function urlChoices(node, changed, relative = true, result = [], depth = 0, classify = null) {
  if (!node || depth > 64) return null;
  const operand = !changed.has('URL') && urlOperand(node);
  if (operand) {
    const choices = classify ? workerStringChoices(operand, classify)
      : literalChoices(operand)?.map(argument => ({argument}));
    if (!choices || result.length + choices.length > 128) return null;
    for (const choice of choices) result.push({urlNode: node, ...choice, base: 'module'});
    return result;
  }
  if (relative && text(node) !== null) {
    if (result.length >= 128) return null;
    result.push({urlNode: node, argument: node, base: 'realm'}); return result;
  }
  if (node.type === 'SequenceExpression') return urlChoices(node.expressions.at(-1), changed, relative, result, depth + 1, classify);
  if (node.type === 'ConditionalExpression')
    return urlChoices(node.consequent, changed, relative, result, depth + 1, classify) &&
      urlChoices(node.alternate, changed, relative, result, depth + 1, classify);
  if (!relative || !classify) return null;
  const choices = workerStringChoices(node, classify);
  if (!choices || result.length + choices.length > 128) return null;
  for (const choice of choices) result.push({urlNode: node, ...choice, base: 'realm'});
  return result;
}

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
// Document modules have lexical scopes. A local URL/Worker binding in one
// factory must not hide the native constructor used by another factory: doing
// so can split equivalent SharedWorker URLs across different output assets.
function scopedGlobals(ast) {
  const ancestorsByNode = new WeakMap(), scopes = new Map(), writes = new Set();
  const functions = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']);
  const lexical = new Set(['Program', 'BlockStatement', 'StaticBlock', 'CatchClause',
    'ForStatement', 'ForInStatement', 'ForOfStatement', 'SwitchStatement', ...functions]);
  const strictScope = node => ['ClassDeclaration', 'ClassExpression', 'StaticBlock'].includes(node.type) ||
    (node.type === 'Program' ? node.body : functions.has(node.type) ? node.body.body : [])?.some(statement => statement.directive === 'use strict');
  function names(node, result = []) {
    if (!node) return result;
    if (node.type === 'Identifier') result.push(node.name);
    else if (node.type === 'RestElement') names(node.argument, result);
    else if (node.type === 'AssignmentPattern') names(node.left, result);
    else if (node.type === 'ArrayPattern') node.elements.forEach(value => names(value, result));
    else if (node.type === 'ObjectPattern') node.properties.forEach(value => names(value.type === 'RestElement' ? value.argument : value.value, result));
    return result;
  }
  const put = (scope, values) => {
    if (!scopes.has(scope)) scopes.set(scope, new Set());
    for (const name of values) scopes.get(scope).add(name);
  };
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    ancestorsByNode.set(node, ancestors.slice());
    const parents = ancestors.slice(0, -1), containing = predicate => parents.findLast(predicate);
    if (node.type === 'VariableDeclaration') {
      const scope = containing(parent => node.kind === 'var'
        ? functions.has(parent.type) || ['Program', 'StaticBlock'].includes(parent.type) : lexical.has(parent.type));
      for (const declaration of node.declarations) put(scope, names(declaration.id));
    } else if (node.type === 'ImportDeclaration') put(ast, node.specifiers.map(value => value.local.name));
    else if (functions.has(node.type)) {
      put(node, node.params.flatMap(value => names(value)));
      if (node.id) {
        put(node, [node.id.name]);
        if (node.type === 'FunctionDeclaration') {
          put(containing(parent => lexical.has(parent.type)), [node.id.name]);
          // Sloppy classic block functions can also create Annex B var bindings.
          if (ast.sourceType === 'script' && !parents.some(strictScope))
            put(containing(parent => functions.has(parent.type) || parent.type === 'Program'), [node.id.name]);
        }
      }
    } else if (node.type === 'CatchClause') put(node, names(node.param));
    else if (['ClassDeclaration', 'ClassExpression'].includes(node.type) && node.id) {
      put(node, [node.id.name]);
      if (node.type === 'ClassDeclaration') put(containing(parent => lexical.has(parent.type)), [node.id.name]);
    }
  });
  const bound = (name, node) => (ancestorsByNode.get(node) ?? []).some(parent => scopes.get(parent)?.has(name));
  function mutation(node) {
    if (node?.type === 'RestElement') return mutation(node.argument);
    if (node?.type === 'AssignmentPattern') return mutation(node.left);
    if (node?.type === 'ArrayPattern') { node.elements.forEach(mutation); return; }
    if (node?.type === 'ObjectPattern') {
      node.properties.forEach(value => mutation(value.type === 'RestElement' ? value.argument : value.value)); return;
    }
    for (const name of names(node)) if (!bound(name, node)) writes.add(name);
    if (node?.type !== 'MemberExpression' || node.object.type !== 'Identifier' ||
        !['globalThis', 'self', 'window'].includes(node.object.name) || bound(node.object.name, node)) return;
    const name = node.computed ? text(node.property) : node.property.name;
    if (name !== null) writes.add(name);
    else ['URL', 'Worker', 'SharedWorker'].forEach(value => writes.add(value));
  }
  walk.simple(ast, {
    AssignmentExpression: node => mutation(node.left), UpdateExpression: node => mutation(node.argument),
    ForInStatement: node => mutation(node.left), ForOfStatement: node => mutation(node.left),
  });
  return node => {
    const ancestors = ancestorsByNode.get(node) ?? [];
    return new Set([...writes, ...ancestors.flatMap(parent => [...(scopes.get(parent) ?? [])]),
      ...(ancestors.some(parent => parent.type === 'WithStatement') ? ['Worker', 'SharedWorker', 'URL'] : [])]);
  };
}
function sitesIn(ast, changed, {classic = false, relative = false, withConstructors = false} = {}) {
  const sites = [], classify = createDynamicImportClassifier(ast);
  walk.simple(ast, {NewExpression(node) {
    const unavailable = typeof changed === 'function' ? changed(node) : changed;
    const kind = node.callee.type === 'Identifier' ? node.callee.name : null;
    if (!['Worker', 'SharedWorker'].includes(kind) || unavailable.has(kind) ||
        node.arguments.length < 1 || node.arguments.length > 2) return;
    const urlNode = node.arguments[0], options = node.arguments[1];
    const choices = urlChoices(urlNode, unavailable, relative, [], 0, classify);
    if (!choices) return;
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
    for (const choice of choices)
      sites.push({kind, workerType, ...(withConstructors ? {constructorNode: node} : {}), ...choice, specifier: choice.specifier ?? text(choice.argument)});
  }});
  return sites;
}

/** Backwards-compatible explicit module-URL analysis. */
export function findModuleWorkerSites(source) {
  const ast = typeof source === 'string' ? parse(source) : source;
  return sitesIn(ast, changedGlobals(ast));
}

/** URL-form workers; literal URLs when their document or worker realm is known. */
export function findWorkerSites(source, {sourceType = 'module', relative = false, unavailableGlobals = []} = {}) {
  const ast = typeof source === 'string' ? parse(source, null, sourceType) : source;
  const globals = scopedGlobals(ast);
  return sitesIn(ast, node => new Set([...globals(node), ...unavailableGlobals]), {classic: true, relative});
}

/** Native worker URL operands belong to the worker transform, including host
 * routes and conditional arms left after other arms have already been emitted. */
export function findWorkerUrlOperands(source, {sourceType = 'module', unavailableGlobals = []} = {}) {
  const ast = typeof source === 'string' ? parse(source, null, sourceType) : source;
  const globals = scopedGlobals(ast), urls = [];
  walk.simple(ast, {NewExpression(node) {
    const kind = node.callee.type === 'Identifier' ? node.callee.name : null;
    const unavailable = new Set([...globals(node), ...unavailableGlobals]);
    if (!['Worker', 'SharedWorker'].includes(kind) || unavailable.has(kind) || unavailable.has('URL')) return;
    function visit(value) {
      if (!value) return;
      if (urlOperand(value)) urls.push(value);
      else if (value.type === 'SequenceExpression') visit(value.expressions.at(-1));
      else if (value.type === 'ConditionalExpression') { visit(value.consequent); visit(value.alternate); }
    }
    visit(node.arguments[0]);
  }});
  return urls;
}

/** Direct global bindings/writes shared by document scripts. Module locals do
 * not become document globals; top-level classic bindings do. */
export function workerGlobalBindings(source, {sourceType = 'script'} = {}) {
  const ast = typeof source === 'string' ? parse(source, null, sourceType) : source;
  return scopedGlobals(ast)(sourceType === 'script' ? ast : null);
}

const realmKey = (url, workerType) => JSON.stringify([workerType, url]);

// All alternatives for one expression share one dispatch. Insertion edits keep
// nested URL/import/worker transformations inside its original evaluation.
function workerEdit(edits, site, value) {
  const {start, end} = site.argument;
  if (site.selector === undefined) { edits.push({start, end, ...value}); return; }
  let group = edits.find(edit => edit.start === start && edit.end === end && edit.choices);
  if (!group) { group = {start, end, choices: []}; edits.push(group); }
  group.choices.push({selector: site.selector, ...value});
}
function workerReplacements(edits, render) {
  const result = [];
  for (const edit of edits) {
    if (!edit.choices) { result.push({...edit, value: render(edit)}); continue; }
    const chain = edit.choices.map(choice =>
      `v === ${JSON.stringify(choice.selector)} ? ${render(choice)} : `).join('');
    const span = edit.end - edit.start;
    result.push({start: edit.start, end: edit.start, side: 1, span, value: `((v) => ${chain}v)(`});
    result.push({start: edit.end, end: edit.end, side: -1, span, value: ')'});
  }
  return result;
}
function applyWorkerEdits(source, replacements, url) {
  replacements.sort((a, b) => a.start - b.start || a.end - b.end ||
    (a.side ?? 0) - (b.side ?? 0) || (a.side === -1 ? a.span - b.span : b.span - a.span));
  const parts = []; let offset = 0;
  for (const edit of replacements) {
    if (edit.start < offset) fail('TRANSFORM', 'Overlapping worker source rewrites', url);
    parts.push(source.slice(offset, edit.start), edit.value); offset = edit.end;
  }
  parts.push(source.slice(offset)); return parts.join('');
}

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
function resolveLocal(specifier, referrer, documentRoot) {
  if (!specifier.startsWith('./') && !specifier.startsWith('../') && !specifier.startsWith('/') && !URL.canParse(specifier))
    fail('RESOLUTION', `Bare worker import needs installed-package resolution: ${specifier}`, referrer);
  return resolveDocumentUrl(specifier, referrer, documentRoot);
}

/** Plan everything before emitFile: no partial graph is emitted on rejection. */
export function buildModuleWorkerGraph(entryUrl, options = {}) {
  return buildWorkerGraph(entryUrl, {...options, workerType: 'module'});
}

/** Classic scripts and ESM are separate source kinds even at the same URL. */
export function buildWorkerGraph(entryUrl, {
  workerType = 'module',
  resolveModule, maxModules = 4096, maxAssets = 4096, maxBytes = 64 * 1024 * 1024,
  requireClosedRuntimeUrls = false,
  documentRoot, transformModule, _moduleSources,
} = {}) {
  documentRoot = normalizeDocumentRoot(documentRoot);
  if (resolveModule === undefined)
    resolveModule = (specifier, referrer) => resolveLocal(specifier, referrer, documentRoot);
  const moduleSources = moduleSourcesFor({transformModule, _moduleSources});
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
    if (remote(url)) {
      if (requireClosedRuntimeUrls)
        fail('DYNAMIC_RESOURCE', 'Uninspected remote worker code needs the original host route', url);
      return {external: url};
    }
    if (new URL(url).protocol !== 'file:') fail('RESOLUTION', 'Only local files and explicit http/https/data imports are supported', url);
    const key = JSON.stringify([realm, sourceType, url]);
    let record = modules.get(key);
    if (!record) {
      if (modules.size >= maxModules) fail('BUDGET', 'Worker module count exceeds maxModules', url);
      const extension = path.extname(fileURLToPath(url));
      const nativeResource = sourceType === 'json' ||
        (sourceType === 'module' && extension.toLowerCase() === '.json');
      // Preserve MIME independently of the requested import type. Missing or
      // incompatible attributes must remain native errors in the worker.
      const name = sourceType === 'script' ? `c-${digest(url)}.js`
        : nativeResource ? `${sourceType === 'json' ? 'j' : 'm'}-${digest(url)}${extension}`
        : `m-${digest(url)}.mjs`;
      record = {key, url, realm, sourceType, nativeResource, name, edges: new Set(), hostUses: new Set()};
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
    const source = record.nativeResource ? null : moduleSources.read(record.url,
      {sourceType: record.sourceType, source: bytes}).code;
    if (!record.nativeResource && record.sourceType === 'module') assertUnbundledModule(record.url, moduleSources);
    analyzedBytes += source === null ? bytes.length : Buffer.byteLength(source);
    if (analyzedBytes > maxBytes) fail('BUDGET', 'Realm-expanded worker sources exceed maxBytes', record.url);
    if (record.nativeResource) {
      // JSON is a native module, not a JavaScript object-literal wrapper. Keep
      // bytes (including BOM, duplicate keys and invalid lazy input) untouched;
      // parsing, default-export identity and rejection timing belong to the host.
      record.source = bytes;
      record.signature = digest(JSON.stringify([record.key, digest(bytes)]));
      continue;
    }
    const ast = parse(source, record.url, record.sourceType), changed = changedGlobals(ast), edits = [];
    const classify = createDynamicImportClassifier(ast);
    const root = roots.get(record.realm);
    record.changed = changed;
    function replace(choice, target, fetch = false) {
      const actual = target.asset ?? target;
      if (!actual.external) record.edges.add(actual);
      workerEdit(edits, choice, {target: actual, suffix: target.external ? '' : suffix(target.url), fetch});
    }
    function dependency(choice, declaration) {
      const value = choice.specifier ?? text(choice.argument), asUrl = choice.base === 'module';
      let resolved;
      try { resolved = asUrl ? resolveDocumentUrl(value, record.url, documentRoot) : resolveModule(value, record.url); }
      catch (cause) {
        if (cause.code?.startsWith('F3D_DOCUMENT_')) throw cause;
        fail('RESOLUTION', cause.message, record.url);
      }
      if (typeof resolved !== 'string') fail('RESOLUTION', 'Worker resolver must return a URL synchronously', record.url);
      resolved = normalize(resolved);
      // Remote modules keep host-owned attribute handling and require no local
      // format decision, including runtime-computed options.
      const sourceType = remote(resolved) ? 'module' : importSourceType(declaration, record.url);
      replace(choice, module(resolved, record.realm, sourceType));
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
    const workers = sitesIn(ast, changed, {classic: true, relative: true, withConstructors: true});
    const workerConstructors = new Set(workers.map(site => site.constructorNode));
    const workerURLs = new Set(workers.map(site => site.urlNode.start));
    if (requireClosedRuntimeUrls) {
      // Unrecognized native URL consumers and escaped host functions can still
      // use relative URLs. Keep those workers at their original document route
      // rather than relocate only the calls this bounded transform recognizes.
      const consumers = new Set(['fetch', 'importScripts', 'Worker', 'SharedWorker', 'URL',
        'XMLHttpRequest', 'Request', 'WebSocket', 'EventSource', 'eval', 'Function']);
      if ([...consumers, 'globalThis', 'self'].some(name => changed.has(name)))
        fail('DYNAMIC_RESOURCE', 'Rebound worker URL consumers need the original host route', record.url);
      walk.fullAncestor(ast, (node, _state, ancestors) => {
        const parent = ancestors.at(-2);
        if (node.type === 'Identifier' && ['globalThis', 'self'].includes(node.name)) {
          if (parent?.type !== 'MemberExpression' || parent.object !== node ||
              (parent.computed && text(parent.property) === null))
            fail('DYNAMIC_RESOURCE', 'An escaped worker global needs the original host route', record.url);
        }
        const name = node.type === 'Identifier' ? node.name
          : node.type === 'MemberExpression' && node.object.type === 'Identifier' &&
            ['globalThis', 'self'].includes(node.object.name) ? node.computed ? text(node.property) : node.property.name : null;
        if (!consumers.has(name)) return;
        const direct = parent?.callee === node;
        if (['fetch', 'importScripts'].includes(name) && direct && parent.type === 'CallExpression' && hostCall(node, name)) return;
        if (['Worker', 'SharedWorker'].includes(name) && direct && parent.type === 'NewExpression' && workerConstructors.has(parent)) return;
        if (name === 'URL' && direct && parent.type === 'NewExpression' && urlOperand(parent)) return;
        fail('DYNAMIC_RESOURCE', `The worker ${name} consumer needs the original host route`, record.url);
      });
    }
    const imports = new Map(), scriptImports = new Map(), fetchUrlArguments = new Map();
    // acorn-walk visits new URL before its enclosing import/call. Claim those
    // operands first so executable dependencies cannot also become raw assets.
    walk.simple(ast, {
      ImportExpression(node) {
        const choices = urlChoices(node.source, changed, true, [], 0, classify);
        if (!choices) fail('DYNAMIC_IMPORT', 'Use bounded string/URL import choices in a relocated worker', record.url);
        imports.set(node, choices);
        for (const choice of choices) if (choice.base === 'module') workerURLs.add(choice.urlNode.start);
      },
      CallExpression(node) {
        if (hostCall(node.callee, 'fetch')) {
          for (const choice of urlChoices(node.arguments[0], changed, true, [], 0, classify) ?? [])
            if (choice.base === 'module') fetchUrlArguments.set(choice.argument, staticGet(node.arguments[1]));
        }
        if (!hostCall(node.callee, 'importScripts')) return;
        const choices = node.arguments.map(argument => urlChoices(argument, changed, true, [], 0, classify));
        if (root.workerType === 'classic' && choices.some(choice => !choice))
          fail('IMPORT_SCRIPTS', 'Classic worker imports require bounded string/URL script choices', record.url);
        scriptImports.set(node, choices);
        for (const choice of choices.flat()) if (choice?.base === 'module') workerURLs.add(choice.urlNode.start);
      },
    });
    for (const site of workers) {
      if (hostRootUrl(site.specifier, documentRoot)) continue;
      const url = resolveDocumentUrl(site.specifier, site.base === 'realm' ? root.url : record.url, documentRoot);
      if (new URL(url).protocol !== 'file:') continue;
      if (documentRoot && isRootRelativeDocumentUrl(site.specifier) && !localFileExists(url)) continue;
      const target = worker(url, site.workerType);
      record.edges.add(target);
      workerEdit(edits, site, {target, suffix: suffix(url), fetch: site.base === 'realm'});
      uses(site.kind); if (site.base === 'module') uses('URL');
    }
    walk.simple(ast, {
      ImportDeclaration: node => dependency({argument: node.source}, node),
      ExportNamedDeclaration: node => { if (node.source) dependency({argument: node.source}, node); },
      ExportAllDeclaration: node => dependency({argument: node.source}, node),
      ImportExpression(node) {
        for (const choice of imports.get(node)) {
          if (choice.base === 'module' && hostRootUrl(choice.specifier ?? text(choice.argument), documentRoot)) continue;
          dependency(choice, node);
          if (choice.base === 'module') uses('URL');
        }
      },
      CallExpression(node) {
        if (hostCall(node.callee, 'importScripts')) {
          // The native method throws in a module worker, including zero args.
          // Do not fetch a missing file or turn this into a successful import.
          if (root.workerType !== 'classic') return;
          for (const choices of scriptImports.get(node)) {
            for (const choice of choices) {
              const value = choice.specifier ?? text(choice.argument);
              if (hostRootUrl(value, documentRoot)) continue;
              const url = resolveDocumentUrl(value, choice.base === 'module' ? record.url : root.url, documentRoot);
              if (new URL(url).protocol !== 'file:') continue;
              replace(choice, module(url, record.realm, 'script'), choice.base === 'realm');
              uses('importScripts', node.callee);
              if (choice.base === 'module') uses('URL');
            }
          }
          return;
        }
        if (!hostCall(node.callee, 'fetch')) return;
        const choices = urlChoices(node.arguments[0], changed, true, [], 0, classify);
        if (!choices) {
          // A document Worker string must retain its host route:
          // relocating an unknown relative request would change its base.
          if (requireClosedRuntimeUrls)
            fail('DYNAMIC_RESOURCE', 'A dynamic fetch URL needs the original worker host route', record.url);
          return;
        }
        for (const choice of choices) {
          // Module-relative URL operands are closed by the URL-asset visit.
          if (choice.base === 'module') continue;
          const value = choice.specifier ?? text(choice.argument);
          if (hostRootUrl(value, documentRoot) || URL.canParse(value)) continue;
          const rooted = documentRoot && isRootRelativeDocumentUrl(value);
          if (!rooted && !staticGet(node.arguments[1]))
            fail('FETCH', 'Relative worker resource fetches require a statically known GET request', record.url);
          const url = resolveDocumentUrl(value, root.url, documentRoot);
          // Root paths keep the same origin when a worker moves. A live API
          // request or absent static file can therefore retain its native URL.
          if (rooted &&
              (!staticGet(node.arguments[1]) || !localFileExists(url))) continue;
          // fetch()/importScripts() use the worker entry, NOT an imported script.
          replace(choice, asset(url), true);
          uses('fetch', node.callee);
        }
      },
      NewExpression(node) {
        if (requireClosedRuntimeUrls && !workerConstructors.has(node) &&
            ['Worker', 'SharedWorker'].some(kind => hostCall(node.callee, kind)))
          fail('DYNAMIC_RESOURCE', 'A dynamic worker URL/options needs the original worker host route', record.url);
        if (changed.has('URL') || workerURLs.has(node.start)) return;
        const operand = urlOperand(node), candidates = workerStringChoices(operand, classify);
        if (operand && !candidates && requireClosedRuntimeUrls)
          fail('DYNAMIC_RESOURCE', 'A dynamic module asset URL needs the original worker host route', record.url);
        if (!candidates) return;
        for (const choice of candidates) {
          const value = choice.specifier;
          if (hostRootUrl(value, documentRoot)) continue;
          const url = resolveDocumentUrl(value, record.url, documentRoot);
          if (new URL(url).protocol !== 'file:') continue;
          if (documentRoot && isRootRelativeDocumentUrl(value) && fetchUrlArguments.has(choice.argument) &&
              (!fetchUrlArguments.get(choice.argument) || !localFileExists(url))) continue;
          replace(choice, asset(url)); uses('URL');
        }
      },
    });
    record.original = source; record.edits = edits.sort((a, b) => b.start - a.start);
    // Hash symbolic links, not final output URLs, so mutually spawning workers
    // have a finite content address and SharedWorker URLs don't depend on who
    // happened to discover their graph first.
    const link = e => [e.target.external ?? e.target.key, e.suffix, e.fetch];
    record.signature = digest(JSON.stringify([record.key, source, record.edits.map(e =>
      [e.start, e.end, ...(e.choices ? [e.choices.map(choice => [choice.selector, ...link(choice)])] : link(e))])]));
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
    if (record.nativeResource) {
      outputBytes += record.source.length;
      if (outputBytes > maxBytes) fail('BUDGET', 'Relocated worker output exceeds maxBytes', entryUrl);
      continue;
    }
    const replacements = workerReplacements(record.edits, edit => {
      let value = edit.target.external;
      if (value === undefined) {
        const from = edit.fetch ? roots.get(record.realm).fileName : record.fileName;
        const relative = path.posix.relative(path.posix.dirname(from), edit.target.fileName);
        value = (relative.startsWith('.') ? relative : './' + relative) + edit.suffix;
      }
      return JSON.stringify(value);
    });
    const bytes = replacements.reduce((total, edit) => total + Buffer.byteLength(edit.value) -
      Buffer.byteLength(record.original.slice(edit.start, edit.end)), Buffer.byteLength(record.original));
    outputBytes += bytes;
    if (outputBytes > maxBytes) fail('BUDGET', 'Relocated worker output exceeds maxBytes', entryUrl);
    record.source = applyWorkerEdits(record.original, replacements, record.url);
  }
  const files = [...modules.values(), ...assets.values()].map(record => ({
    type: 'asset', fileName: record.fileName, source: record.source,
  }));
  return {entryFile: entry.fileName, files, moduleCount: modules.size, assetCount: assets.size, outputBytes,
    watchFiles: [...inputs.keys()], entries: [...roots.values()].map(root => ({url: root.url, workerType: root.workerType, fileName: root.fileName}))};
}

/** Run before the ordinary URL-asset transform in bundleWithRollup. */
export function moduleWorkerRollupPlugin(options = {}) {
  options = {...options, documentRoot: normalizeDocumentRoot(options.documentRoot), _moduleSources: moduleSourcesFor(options)};
  const emitted = new Map(), references = new Map(), filesByName = new Map();
  const closedRealmEntries = new Set();
  let totalBytes = 0, totalModules = 0, totalAssets = 0;
  const maxBytes = options.maxBytes ?? 64 * 1024 * 1024;
  const maxModules = options.maxModules ?? 4096, maxAssets = options.maxAssets ?? 4096;
  return {
    name: 'f3d-module-workers',
    buildStart() { emitted.clear(); references.clear(); filesByName.clear(); closedRealmEntries.clear(); totalBytes = totalModules = totalAssets = 0; },
    resolveFileUrl({referenceId, relativePath}) {
      if (!references.has(referenceId)) return null;
      const relative = relativePath.startsWith('.') ? relativePath : './' + relativePath;
      return JSON.stringify(relative + references.get(referenceId).suffix);
    },
    transform(code, id) {
      if (!code.includes('Worker')) return null;
      const sites = findWorkerSites(code, {sourceType: options.sourceType ?? 'module',
        relative: Boolean(options.documentUrl), unavailableGlobals: options.unavailableGlobals});
      if (!sites.length) return null;
      const base = id.startsWith('file:') ? id : path.isAbsolute(id) ? pathToFileURL(id).href : null;
      if (!base) fail('ENTRY', 'A module worker URL needs a local source module', id);
      const edits = [];
      for (const site of sites) {
        // Origin-root workers need an explicit local root. Protocol-relative
        // workers always retain their original host route.
        if (hostRootUrl(site.specifier, options.documentRoot)) continue;
        const sourceBase = site.base === 'realm' ? options.documentBaseUrl ?? options.documentUrl : base;
        const url = resolveDocumentUrl(site.specifier, sourceBase, options.documentRoot);
        if (new URL(url).protocol !== 'file:') continue;
        const rooted = options.documentRoot && isRootRelativeDocumentUrl(site.specifier);
        if (rooted && !localFileExists(url)) continue;
        const optionalHostRoute = site.base === 'realm' || rooted;
        const key = realmKey(url, site.workerType);
        let referenceId = emitted.get(key);
        if (referenceId && optionalHostRoute && !closedRealmEntries.has(key)) {
          // An earlier import.meta URL site may have used the existing open
          // network profile. That is not proof for relocating a document string.
          try {
            const graph = buildWorkerGraph(url, {...options, workerType: site.workerType, requireClosedRuntimeUrls: true});
            for (const entry of graph.entries) closedRealmEntries.add(realmKey(entry.url, entry.workerType));
          } catch (error) {
            if (error instanceof ModuleWorkerBuildError) continue;
            throw error;
          }
        }
        if (!referenceId) {
          let graph;
          try {
            graph = buildWorkerGraph(url, {...options, workerType: site.workerType,
              requireClosedRuntimeUrls: optionalHostRoute});
          } catch (error) {
            // A document-relative or origin-root URL may name a server-provided worker or
            // an open runtime graph. Preserve its native request/error behavior
            // when local closure cannot be established, without partial output.
            if (optionalHostRoute && error instanceof ModuleWorkerBuildError) continue;
            throw error;
          }
          if (graph.outputBytes > maxBytes - totalBytes || graph.moduleCount > maxModules - totalModules || graph.assetCount > maxAssets - totalAssets) {
            if (optionalHostRoute) continue;
            fail('BUDGET', 'Module worker graphs exceed the shared build budget', url);
          }
          for (const filename of graph.watchFiles) this.addWatchFile?.(filename);
          for (const file of graph.files) {
            let ref = filesByName.get(file.fileName);
            if (!ref) { ref = this.emitFile(file); filesByName.set(file.fileName, ref); }
          }
          for (const entry of graph.entries) {
            const ref = filesByName.get(entry.fileName);
            emitted.set(realmKey(entry.url, entry.workerType), ref);
            if (optionalHostRoute) closedRealmEntries.add(realmKey(entry.url, entry.workerType));
            references.set(ref, {suffix: suffix(entry.url), fileName: entry.fileName});
          }
          referenceId = emitted.get(key);
          totalBytes += graph.outputBytes; totalModules += graph.moduleCount; totalAssets += graph.assetCount;
        }
        let replacement = `import.meta.ROLLUP_FILE_URL_${referenceId}`;
        if (site.base === 'realm') {
          const target = references.get(referenceId);
          // A direct Worker string uses document.baseURI, even when its call is
          // in a nested module or a lazy chunk. Keep it a string so a shadowed
          // URL constructor cannot change previously valid source behavior.
          const documentDir = options.documentRoot ?? path.dirname(fileURLToPath(options.documentUrl));
          const baseDir = fileURLToPath(new URL('.', sourceBase));
          const relative = path.relative(baseDir, path.join(documentDir, target.fileName)).split(path.sep).join('/');
          replacement = JSON.stringify((relative.startsWith('.') ? relative : './' + relative) + target.suffix);
        }
        workerEdit(edits, site, {text: replacement});
      }
      if (!edits.length) return null;
      return {code: applyWorkerEdits(code, workerReplacements(edits, edit => edit.text), id), map: null};
    },
  };
}
