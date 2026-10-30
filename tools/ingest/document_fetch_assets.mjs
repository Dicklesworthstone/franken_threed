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
  const native = (node, name) => node?.type === 'Identifier' && node.name === name && !bound(node, name) ||
    node?.type === 'MemberExpression' && host(node.object) && property(node) === name;
  let hostMutated = false, requestMutated = false;
  if (ast.sourceType === 'script') {
    // Query the existing Program scope, including vars hoisted out of blocks,
    // without confusing a function-local or block-lexical binding with a global.
    const declared = name => isIdentifierShadowedAtAncestors(name, [ast, null], cache);
    const hostDeclared = [...hosts].some(declared);
    hostMutated = hostDeclared || declared('fetch'); requestMutated = hostDeclared || declared('Request');
    if (!ast.body.some(statement => statement.directive === 'use strict')) walk.simple(ast, {
      FunctionDeclaration(node) {
        // Sloppy classic block functions can also create Annex B global vars.
        if (ancestors.get(node).slice(0, -1).some(parent =>
          /^(?:Function|ArrowFunction|Class)/.test(parent.type) || parent.type === 'StaticBlock')) return;
        if (node.id.name === 'fetch' || hosts.has(node.id.name)) hostMutated = true;
        if (node.id.name === 'Request' || hosts.has(node.id.name)) requestMutated = true;
      },
    });
  }
  function mutation(node) {
    if (!node) return;
    if (node.type === 'Identifier') {
      if ((node.name === 'fetch' || hosts.has(node.name)) && !bound(node, node.name)) hostMutated = true;
      if ((node.name === 'Request' || hosts.has(node.name)) && !bound(node, node.name)) requestMutated = true;
    } else if (node.type === 'MemberExpression' && host(node.object)) {
      if (property(node) === 'fetch' || property(node) === null) hostMutated = true;
      if (property(node) === 'Request' || property(node) === null) requestMutated = true;
    } else if (node.type === 'MemberExpression' && dictionaryPrototype(node.object)) {
      // RequestInit reads inherited dictionary fields too (method, headers,
      // etc.). A changed Object.prototype is not a proved default GET.
      hostMutated = requestMutated = true;
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
        hostMutated = requestMutated = true;
    },
  });
  function plainGet(value) {
    const options = tail(value);
    if (!options || options.type === 'Literal' && options.value === null) return true;
    const plain = options.type === 'ObjectExpression' && options.properties.every(p =>
      p.type === 'Property' && !p.computed && !p.method && p.kind === 'init' &&
      (p.key.name ?? p.key.value) !== '__proto__');
    const methods = plain ? options.properties.filter(p => (p.key.name ?? p.key.value) === 'method') : [];
    return plain && (!methods.length || literal(tail(methods.at(-1).value))?.toUpperCase() === 'GET');
  }
  const sites = [];
  walk.simple(ast, {CallExpression(node) {
    if (!native(node.callee, 'fetch')) return;
    const input = tail(node.arguments[0]);
    const request = input?.type === 'NewExpression' && native(input.callee, 'Request') ? input : null;
    // Rewrite the constructor's URL only: Request creation, its options and
    // outer fetch overrides remain native, in their original evaluation order.
    const argument = request ? request.arguments[0] : node.arguments[0], classified = classify(argument);
    const candidates = classified.classification === 'literal' ? [classified.specifier] : classified.candidates;
    // Existing Request values, aliases and open URLs keep their host route.
    if (!candidates?.length) return;
    const mutated = hostMutated || request && requestMutated;
    const admitted = !mutated && !node.optional && node.arguments.length <= 2 && plainGet(node.arguments[1]) &&
      (!request || request.arguments.length <= 2 && plainGet(request.arguments[1]));
    sites.push({node, argument, request, candidates, admitted, reason: mutated ? 'host-mutation' : 'request-options'});
  }});
  return {sites, hostMutated, requestMutated};
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
