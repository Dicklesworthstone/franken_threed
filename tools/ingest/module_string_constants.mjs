/** Bounded, analysis-only links for primitive string exports. The normal module
 * loader still owns evaluation, live bindings, cycles, effects and TDZ errors.
 * Read/resolve callbacks must use the caller's actual module/source snapshot and
 * containment rules; this helper never resolves a second package universe.
 */
import * as acorn from 'acorn';
import {createDynamicImportClassifier} from './ast_analyzer.mjs';

export function createImportedStringResolver({readModule, resolveModule}) {
  const modules = new Map(), results = new Map(), active = new Set();
  const unknown = Symbol('unproved export');
  let bytes = 0, referenceDepth = 0;
  const nameOf = node => node?.name ?? node?.value;

  function moduleFor(url) {
    if (modules.has(url)) return modules.get(url);
    if (modules.size >= 128) return null;
    modules.set(url, null);
    let ast, source;
    try {
      source = readModule(url);
      if (typeof source !== 'string' || Buffer.byteLength(source) > 2 * 1024 * 1024 - bytes) return null;
      bytes += Buffer.byteLength(source);
      ast = acorn.parse(source, {ecmaVersion: 'latest', sourceType: 'module'});
    } catch (error) {
      if (error.code === 'F3D_SOURCE_TRANSFORM') throw error;
      return null;
    }
    const exports = new Map(), imports = new Map(), stars = [];
    function localExports(pattern) {
      // Mutable/destructured/function/class exports still shadow star exports,
      // even when the string classifier cannot prove their values.
      const pending = [pattern];
      while (pending.length) {
        const node = pending.pop();
        if (!node) continue;
        if (node.type === 'Identifier') exports.set(node.name, {value: node, local: node.name});
        else if (node.type === 'ObjectPattern') {
          for (const property of node.properties) pending.push(property.type === 'RestElement' ? property.argument : property.value);
        } else if (node.type === 'ArrayPattern') {
          for (const element of node.elements) pending.push(element);
        }
        else if (node.type === 'AssignmentPattern') pending.push(node.left);
        else if (node.type === 'RestElement') pending.push(node.argument);
      }
    }
    for (const node of ast.body) {
      if (node.type === 'ImportDeclaration') {
        for (const specifier of node.specifiers) imports.set(specifier.local.name, {
          source: node.source.value, attributes: node.attributes,
          name: specifier.type === 'ImportDefaultSpecifier' ? 'default' : nameOf(specifier.imported),
        });
      }
      if (node.type === 'ExportDefaultDeclaration') exports.set('default', {
        value: node.declaration, local: node.declaration.id?.name ?? '*default*',
      });
      if (node.type === 'ExportAllDeclaration') {
        if (node.exported) exports.set(nameOf(node.exported), {namespace: true});
        else stars.push({source: node.source.value, attributes: node.attributes});
      }
      if (node.type !== 'ExportNamedDeclaration') continue;
      if (node.declaration?.type === 'VariableDeclaration') {
        for (const declaration of node.declaration.declarations) localExports(declaration.id);
      } else if (node.declaration?.id) localExports(node.declaration.id);
      for (const specifier of node.specifiers) {
        const exported = nameOf(specifier.exported);
        exports.set(exported, node.source
          ? {source: node.source.value, name: nameOf(specifier.local), attributes: node.attributes}
          : {value: specifier.local, local: nameOf(specifier.local)});
      }
    }
    const record = {exports, imports, stars, classify: createDynamicImportClassifier(ast, forModule(url))};
    modules.set(url, record);
    return record;
  }

  function sourceUrl(specifier, from) {
    let url;
    try { url = resolveModule(specifier, from); }
    catch (error) {
      if (error.code === 'F3D_SOURCE_TRANSFORM') throw error;
      return null;
    }
    // JSON/CommonJS/virtual/external modules do not establish this ESM proof.
    return typeof url === 'string' && url.startsWith('file:') ? url : null;
  }

  function linkedBinding(from, entry, name, seen) {
    if (entry.attributes?.length || typeof name !== 'string' || referenceDepth >= 64) return unknown;
    const url = sourceUrl(entry.source, from);
    if (!url) return unknown;
    referenceDepth++;
    try { return bindingFor(url, name, seen); }
    finally { referenceDepth--; }
  }

  function bindingFor(url, name, seen) {
    const key = JSON.stringify([url, name]);
    if (seen.has(key)) return null;
    seen.add(key);
    const record = moduleFor(url);
    if (!record) return unknown;
    const entry = record.exports.get(name);
    if (entry) {
      if (entry.namespace) return unknown;
      if (entry.source) return linkedBinding(url, entry, entry.name, seen);
      const imported = record.imports.get(entry.local);
      if (imported) return linkedBinding(url, imported, imported.name, seen);
      return {url, local: entry.local, value: entry.value, record, depth: referenceDepth};
    }
    if (name === 'default') return null;
    let found = null;
    for (const star of record.stars) {
      const candidate = linkedBinding(url, star, name, seen);
      if (candidate === unknown) return unknown;
      // Resolve binding identity, not equal strings: distinct exports with the
      // same value are ambiguous, but a diamond reaching one binding is valid.
      if (candidate && found && (candidate.url !== found.url || candidate.local !== found.local)) return unknown;
      if (candidate) found = candidate;
    }
    return found;
  }

  function exportedStrings(url, name) {
    const key = JSON.stringify([url, name]);
    if (results.has(key)) return results.get(key);
    if (active.has(key) || active.size >= 64) return null;
    active.add(key);
    const previousDepth = referenceDepth;
    try {
      // A resolution's visited set belongs to this search. Caching a partial
      // miss while traversing a star cycle would hide another reachable export.
      const entry = bindingFor(url, name, new Set());
      let candidates = null;
      if (entry && entry !== unknown) {
        referenceDepth = entry.depth;
        const result = entry.record.classify(entry.value);
        candidates = result.classification === 'literal' ? [result.specifier]
          : result.classification === 'finite_set' ? result.candidates : null;
      }
      results.set(key, candidates);
      return candidates;
    } finally { referenceDepth = previousDepth; active.delete(key); }
  }

  function forModule(from) {
    return (specifier, name, localDepth = 0) => {
      // Local aliases and export hops share one stack budget. Separate bounds
      // would permit 64 modules with 64 aliases each to exhaust the host stack.
      if (referenceDepth + localDepth >= 64) return null;
      const previousDepth = referenceDepth;
      referenceDepth += localDepth + 1;
      try {
        const url = sourceUrl(specifier, from);
        return url ? exportedStrings(url, name) : null;
      } finally { referenceDepth = previousDepth; }
    };
  }
  return forModule;
}
