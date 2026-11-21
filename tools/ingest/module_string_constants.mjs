/** Bounded, analysis-only links for primitive string exports. The normal module
 * loader still owns evaluation, live bindings, cycles, effects and TDZ errors.
 * Read/resolve callbacks must use the caller's actual module/source snapshot and
 * containment rules; this helper never resolves a second package universe.
 */
import * as acorn from 'acorn';
import {createDynamicImportClassifier} from './ast_analyzer.mjs';

export function createImportedStringResolver({readModule, resolveModule}) {
  const modules = new Map(), results = new Map(), active = new Set();
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
    const exports = new Map();
    for (const node of ast.body) {
      if (node.type === 'ExportDefaultDeclaration') exports.set('default', {value: node.declaration});
      if (node.type !== 'ExportNamedDeclaration') continue;
      if (node.declaration?.type === 'VariableDeclaration' && node.declaration.kind === 'const') {
        for (const declaration of node.declaration.declarations)
          if (declaration.id.type === 'Identifier') exports.set(declaration.id.name, {value: declaration.id});
      }
      for (const specifier of node.specifiers) {
        const exported = nameOf(specifier.exported);
        exports.set(exported, node.source
          ? {source: node.source.value, name: nameOf(specifier.local), attributes: node.attributes}
          : {value: specifier.local});
      }
    }
    const record = {exports, classify: createDynamicImportClassifier(ast, forModule(url))};
    modules.set(url, record);
    return record;
  }

  function exportedStrings(url, name) {
    const key = JSON.stringify([url, name]);
    if (results.has(key)) return results.get(key);
    if (active.has(key) || active.size >= 64) return null;
    active.add(key);
    try {
      const record = moduleFor(url), entry = record?.exports.get(name);
      let candidates = null;
      if (entry?.source && !entry.attributes?.length) candidates = forModule(url)(entry.source, entry.name);
      else if (entry?.value) {
        const result = record.classify(entry.value);
        candidates = result.classification === 'literal' ? [result.specifier]
          : result.classification === 'finite_set' ? result.candidates : null;
      }
      results.set(key, candidates);
      return candidates;
    } finally { active.delete(key); }
  }

  function forModule(from) {
    return (specifier, name, localDepth = 0) => {
      // Local aliases and export hops share one stack budget. Separate bounds
      // would permit 64 modules with 64 aliases each to exhaust the host stack.
      if (referenceDepth + localDepth >= 64) return null;
      const previousDepth = referenceDepth;
      referenceDepth += localDepth + 1;
      try {
        let url;
        try { url = resolveModule(specifier, from); }
        catch (error) {
          if (error.code === 'F3D_SOURCE_TRANSFORM') throw error;
          return null;
        }
        // JSON/CommonJS/virtual/external modules do not establish this ESM proof.
        if (typeof url !== 'string' || !url.startsWith('file:')) return null;
        return exportedStrings(url, name);
      } finally { referenceDepth = previousDepth; }
    };
  }
  return forModule;
}
