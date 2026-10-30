/** Finite document GET resources. Native fetch, RequestInit and response bodies
 * remain host-owned; discovery never executes selectors or fetches the network.
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {createDynamicImportClassifier, isIdentifierShadowedAtAncestors} from './ast_analyzer.mjs';

const hosts = new Set(['globalThis', 'self', 'window']);
const tail = node => {
  for (let depth = 0; node?.type === 'SequenceExpression'; depth++) {
    if (depth >= 64) return node;
    node = node.expressions.at(-1);
  }
  return node;
};
const literal = node => node?.type === 'Literal' && typeof node.value === 'string' ? node.value : null;
const property = node => node.computed ? literal(node.property) : node.property.name;
export const isRelativeDocumentFetchUrl = value => Boolean(value.trim()) &&
  !/^(?:[a-z][\w+.-]*:|[\/\\#])/i.test(value.trim());

export function analyzeDocumentFetch(source, {sourceType = 'module'} = {}) {
  const ast = typeof source === 'string' ? acorn.parse(source, {ecmaVersion: 'latest', sourceType}) : source;
  const ancestors = new WeakMap(), cache = new WeakMap(), classify = createDynamicImportClassifier(ast);
  walk.fullAncestor(ast, (node, _state, parents) => ancestors.set(node, parents.slice()));
  const bound = (node, name) => isIdentifierShadowedAtAncestors(name, ancestors.get(node) ?? [ast, node], cache);
  const host = node => node?.type === 'Identifier' && hosts.has(node.name) && !bound(node, node.name);
  const dictionaryPrototype = node => node?.type === 'MemberExpression' && node.object.name === 'Object' &&
    !bound(node.object, 'Object') && property(node) === 'prototype';
  let hostMutated = false;
  if (ast.sourceType === 'script') {
    const declares = node => node && (node.type === 'Identifier' ? node.name === 'fetch' || hosts.has(node.name)
      : node.type === 'ArrayPattern' ? node.elements.some(declares)
      : node.type === 'ObjectPattern' ? node.properties.some(p => declares(p.argument ?? p.value))
      : declares(node.argument ?? node.left));
    hostMutated = ast.body.some(node => declares(node.id) ||
      node.type === 'VariableDeclaration' && node.declarations.some(d => declares(d.id)));
  }
  function mutation(node) {
    if (!node) return;
    if (node.type === 'Identifier') {
      if ((node.name === 'fetch' || hosts.has(node.name)) && !bound(node, node.name)) hostMutated = true;
    } else if (node.type === 'MemberExpression' && host(node.object)) {
      if (property(node) === 'fetch' || property(node) === null) hostMutated = true;
    } else if (node.type === 'MemberExpression' && dictionaryPrototype(node.object)) {
      // RequestInit reads inherited dictionary fields too (method, headers,
      // etc.). A changed Object.prototype is not a proved default GET.
      hostMutated = true;
    } else if (node.type === 'RestElement' || node.type === 'AssignmentPattern') mutation(node.argument ?? node.left);
    else if (node.type === 'ArrayPattern') node.elements.forEach(mutation);
    else if (node.type === 'ObjectPattern') node.properties.forEach(p => mutation(p.argument ?? p.value));
  }
  walk.simple(ast, {
    AssignmentExpression: node => mutation(node.left), UpdateExpression: node => mutation(node.argument),
    UnaryExpression: node => { if (node.operator === 'delete') mutation(node.argument); },
    ForInStatement: node => mutation(node.left), ForOfStatement: node => mutation(node.left),
    CallExpression(node) {
      if (node.callee.type === 'MemberExpression' && ['Object', 'Reflect'].includes(node.callee.object.name) &&
          !bound(node.callee.object, node.callee.object.name) && (host(node.arguments[0]) || dictionaryPrototype(node.arguments[0])) &&
          ['assign', 'defineProperty', 'defineProperties', 'set', 'deleteProperty'].includes(property(node.callee)))
        hostMutated = true;
    },
  });
  const sites = [];
  walk.simple(ast, {CallExpression(node) {
    const callee = node.callee;
    if (!(callee.type === 'Identifier' && callee.name === 'fetch' && !bound(callee, 'fetch')) &&
        !(callee.type === 'MemberExpression' && host(callee.object) && property(callee) === 'fetch')) return;
    const argument = node.arguments[0], classified = classify(argument);
    const candidates = classified.classification === 'literal' ? [classified.specifier] : classified.candidates;
    // Request objects, aliases and open URLs keep their existing host route.
    if (!candidates?.length) return;
    const options = tail(node.arguments[1]);
    let admitted = !hostMutated && !node.optional && node.arguments.length <= 2;
    if (options && !(options.type === 'Literal' && options.value === null)) {
      const plain = options.type === 'ObjectExpression' && options.properties.every(p =>
        p.type === 'Property' && !p.computed && !p.method && p.kind === 'init' &&
        (p.key.name ?? p.key.value) !== '__proto__');
      const methods = plain ? options.properties.filter(p => (p.key.name ?? p.key.value) === 'method') : [];
      admitted &&= plain && (!methods.length || literal(tail(methods.at(-1).value))?.toUpperCase() === 'GET');
    }
    sites.push({node, argument, candidates, admitted, reason: hostMutated ? 'host-mutation' : 'request-options'});
  }});
  return {sites, hostMutated};
}

/** Copy responses at the document's original paths, never their importing
 * module's directory. Missing/live/root-relative requests stay network-owned.
 * Raw .js/.css/model bodies are data, not executable or recursive dependencies.
 */
export function collectDocumentFetchAssets(sites, {documentUrl, documentBaseUrl = documentUrl}) {
  const result = new Map(), root = path.dirname(fileURLToPath(documentUrl));
  let bytes = 0;
  for (const site of sites) for (const value of site.admitted ? site.candidates : []) {
    if (!isRelativeDocumentFetchUrl(value)) continue;
    let filename;
    try {
      const url = new URL(value, documentBaseUrl);
      if (url.protocol !== 'file:') continue;
      filename = fileURLToPath(url);
    } catch { continue; }
    const rel = path.relative(root, filename);
    const fileName = rel.split(path.sep).join('/');
    if (rel === '..' || rel.startsWith('..' + path.sep) || path.isAbsolute(rel) || result.has(fileName)) continue;
    if (!fs.existsSync(filename) || !fs.statSync(filename).isFile()) continue;
    const real = path.relative(fs.realpathSync(root), fs.realpathSync(filename));
    if (real === '..' || real.startsWith('..' + path.sep) || path.isAbsolute(real)) continue;
    if (result.size >= 4096 || (bytes += fs.statSync(filename).size) > 256 * 1024 * 1024)
      throw new Error('F3D_FETCH_LIMIT: document resource closure exceeds its file/byte bound');
    result.set(fileName, {source: fs.readFileSync(filename), originalFileName: filename});
  }
  return result;
}
