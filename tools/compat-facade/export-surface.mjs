import fs from 'node:fs';
import path from 'node:path';
import * as acorn from 'acorn';

const graphs = new WeakMap();
const AMBIGUOUS = Symbol('ambiguous export');
const NAMESPACE = Symbol('module namespace');
const exportName = (node) => node.name ?? node.value;

function boundNames(pattern, names = []) {
  if (!pattern) return names;
  if (pattern.type === 'Identifier') names.push(pattern.name);
  else if (pattern.type === 'RestElement') boundNames(pattern.argument, names);
  else if (pattern.type === 'AssignmentPattern') boundNames(pattern.left, names);
  else if (pattern.type === 'ArrayPattern') {
    for (const element of pattern.elements) boundNames(element, names);
  } else if (pattern.type === 'ObjectPattern') {
    for (const property of pattern.properties) {
      boundNames(property.type === 'RestElement' ? property.argument : property.value, names);
    }
  }
  return names;
}

/**
 * Extract local ESM exports without evaluating application code. Resolve stars by
 * binding identity, not by name alone (ECMA-262 GetExportedNames/ResolveExport).
 * Only completed root results enter the public cache: an in-progress cycle is
 * not an empty module. A cache represents one immutable source snapshot.
 */
export function extractModuleExportSurface(filePath, options = {}) {
  const cache = options.cache || new Map();
  const root = path.resolve(filePath);
  const recursive = options.recursive !== false;
  // Shallow queries must neither consume nor poison recursive results.
  if (recursive && cache.has(root)) return cache.get(root);
  let graph = graphs.get(cache);
  if (!graph) graphs.set(cache, graph = new Map());

  function read(file) {
    if (recursive && graph.has(file)) return graph.get(file);
    const record = { exports: new Map(), imports: new Map(), stars: [], isCJS: file.endsWith('.cjs') };
    if (!fs.existsSync(file) || record.isCJS) return record;
    // Do not cache failed parses as successful empty modules.
    const ast = acorn.parse(fs.readFileSync(file, 'utf-8'), {
      ecmaVersion: 'latest', sourceType: 'module',
    });
    const target = (source) => {
      const specifier = source.value;
      if (!specifier.startsWith('./') && !specifier.startsWith('../') && !path.isAbsolute(specifier)) {
        throw new Error(`Cannot statically resolve package re-export ${JSON.stringify(specifier)} in ${file}`);
      }
      return path.resolve(path.dirname(file), specifier);
    };
    for (const node of ast.body) {
      if (node.type === 'ImportDeclaration') {
        // Resolve imported bindings only when they are re-exported. Side-effect
        // imports and ordinary package dependencies need no filesystem lookup.
        for (const spec of node.specifiers) {
          record.imports.set(spec.local.name, { source: node.source, name:
            spec.type === 'ImportNamespaceSpecifier' ? NAMESPACE :
            spec.type === 'ImportDefaultSpecifier' ? 'default' : exportName(spec.imported) });
        }
      } else if (node.type === 'ExportDefaultDeclaration') {
        const decl = node.declaration;
        const local = /^(Function|Class)Declaration$/.test(decl.type) && decl.id
          ? decl.id.name : '*default*';
        record.exports.set('default', { file, name: local, local: true });
      } else if (node.type === 'ExportNamedDeclaration') {
        const decl = node.declaration;
        if (decl) {
          const names = decl.type === 'VariableDeclaration'
            ? decl.declarations.flatMap((d) => boundNames(d.id)) : boundNames(decl.id);
          for (const name of names) record.exports.set(name, { file, name, local: true });
        }
        for (const spec of node.specifiers) {
          record.exports.set(exportName(spec.exported), node.source
            ? { file: target(node.source), name: exportName(spec.local) }
            : { file, name: exportName(spec.local), local: true });
        }
      } else if (node.type === 'ExportAllDeclaration') {
        if (node.exported) {
          record.exports.set(exportName(node.exported), { file: target(node.source), name: NAMESPACE });
        } else if (recursive) record.stars.push(target(node.source));
      }
    }
    for (const [name, binding] of record.exports) {
      const imported = binding.local && record.imports.get(binding.name);
      // Namespace imports create local bindings, even when their values are identical.
      if (imported && imported.name !== NAMESPACE) {
        record.exports.set(name, { file: target(imported.source), name: imported.name });
      }
    }
    // Shallow records are deliberately not shared with recursive graph walks.
    if (recursive) graph.set(file, record);
    return record;
  }

  function names(file, seen = new Set()) {
    if (seen.has(file)) return new Set();
    seen.add(file);
    const record = read(file);
    const result = new Set(record.exports.keys());
    for (const star of record.stars) {
      for (const name of names(star, seen)) if (name !== 'default') result.add(name);
    }
    return result;
  }

  function resolve(file, name, seen = new Map()) {
    let visitedNames = seen.get(file);
    if (visitedNames?.has(name)) return null;
    if (!visitedNames) seen.set(file, visitedNames = new Set());
    visitedNames.add(name);
    const record = read(file);
    const direct = record.exports.get(name);
    if (direct) {
      if (direct.local || direct.name === NAMESPACE) return direct;
      return resolve(direct.file, direct.name, seen);
    }
    if (name === 'default') return null;
    let binding = null;
    for (const star of record.stars) {
      const candidate = resolve(star, name, seen);
      if (candidate === AMBIGUOUS) return AMBIGUOUS;
      if (!candidate) continue;
      if (binding && (binding.file !== candidate.file || binding.name !== candidate.name)) return AMBIGUOUS;
      binding = candidate;
    }
    return binding;
  }

  const record = read(root);
  const exported = [...names(root)].filter((name) => {
    if (!recursive) return true;
    const binding = resolve(root, name);
    return binding !== null && binding !== AMBIGUOUS;
  });
  const hasDefault = exported.includes('default');
  const result = {
    named: exported.filter((name) => name !== 'default').sort(),
    hasDefault,
    totalCount: exported.length,
    isCJS: record.isCJS,
  };
  if (recursive) cache.set(root, result);
  return result;
}
