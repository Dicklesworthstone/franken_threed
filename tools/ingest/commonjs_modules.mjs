/** Bounded browser CommonJS ingestion. Execution wrappers belong to the pinned
 * Rollup compiler; source analysis deliberately treats this as an opaque edge. */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parse} from 'acorn';
import * as walk from 'acorn-walk';
import commonjs from '@rollup/plugin-commonjs';
import {isIdentifierShadowedAtAncestors} from './ast_analyzer.mjs';

const ENTRY = '\0f3d-commonjs-entry:';
const JSON_ENTRY = '\0f3d-commonjs-json:';
const EMPTY = new URL('./browser_empty.mjs', import.meta.url).href;
const classifications = new WeakMap();
const originalEntry = id => id?.startsWith(ENTRY) ? id.slice(ENTRY.length, -4) : null;
export const commonjsSourceUrl = id => originalEntry(id?.endsWith('?commonjs-entry')
  ? id.slice(0, -'?commonjs-entry'.length) : id) ?? id;
export const commonjsJsonSourceUrl = id => id?.startsWith(JSON_ENTRY) ? id.slice(JSON_ENTRY.length) : null;

export function commonjsError(code, message, url) {
  const error = new Error(`${message}: ${url}`);
  error.code = `F3D_COMMONJS_${code}`; error.url = url;
  return error;
}

export function commonjsFormat(url, sources, {sourceType = 'module', required = false, inline = false} = {}) {
  if (!url.startsWith('file:')) return 'module';
  const source = sources.read(url, {sourceType});
  const declared = sources.format?.(url);
  const filename = fileURLToPath(url), extension = path.extname(filename);
  if (!inline && !source.frontendHandled && !['.js', '.cjs', '.mjs'].includes(extension)) return 'module';
  let cache = classifications.get(sources);
  if (!cache) classifications.set(sources, cache = new Map());
  const key = JSON.stringify([url, sourceType, required, inline, declared]);
  if (cache.has(key)) return cache.get(key);
  const ast = parse(source.code, {ecmaVersion: 'latest', sourceType: 'module', allowReturnOutsideFunction: true});
  const esm = ast.body.some(node => /^(Import|Export)/.test(node.type));
  let references = false;
  walk.ancestor(ast, {Identifier(node, ancestors) {
    const parent = ancestors.at(-2);
    if (parent?.type === 'UnaryExpression' && parent.operator === 'typeof' && parent.argument === node) return;
    if (['require', 'module', 'exports'].includes(node.name) && !isIdentifierShadowedAtAncestors(node.name, ancestors))
      references = true;
  }});
  let type;
  for (let directory = path.dirname(filename); required && path.basename(directory) !== 'node_modules';) {
    try { type = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8')).type; break; }
    catch (error) { if (error.code !== 'ENOENT' && error.code !== 'ENOTDIR') throw error; }
    const parent = path.dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  // Browser module/script contexts do not inherit Node package type. Explicit
  // ESM containing free CJS globals remains opaque, never a closed ESM proof.
  const result = esm || extension === '.mjs' || inline
    ? references ? 'mixed-commonjs' : 'module'
    : references || declared === 'commonjs' || !source.frontendHandled && extension === '.cjs' ||
        required && declared !== 'module' && type !== 'module' ? 'commonjs' : 'module';
  cache.set(key, result);
  return result;
}

export const isCommonJS = (url, sources, options) => commonjsFormat(url, sources, options) !== 'module';

export function commonjsEntry(url, sources) {
  const format = commonjsFormat(url, sources);
  if (format === 'module') return url;
  if (format === 'mixed-commonjs')
    throw commonjsError('MIXED_MODULE', 'Mixed ESM and free CommonJS bindings need normalization by the project frontend', url);
  const parsed = new URL(url);
  if (parsed.search || parsed.hash)
    throw commonjsError('URL_IDENTITY', 'CommonJS URL variants require a project frontend', url);
  return ENTRY + url + '.cjs';
}

export function assertUnbundledModule(url, sources, sourceType = 'module') {
  if (isCommonJS(url, sources, {sourceType}))
    throw commonjsError('UNBUNDLED', 'CommonJS needs the application bundler; retained modules and worker sources must already be browser JavaScript', url);
}

/** Keep the pinned compiler's initializer, but give each failed attempt its own
 * storage and lexical environment. A stable require entry still exposes partial
 * exports to cycles. Reusing a failed attempt's outer export variables would
 * redirect closures already retained by a successfully initialized dependency.
 */
function retryableCommonjs(result, id, ast) {
  if (result.meta?.commonjs?.isCommonJS !== 'withRequireFunction') return result;
  const fail = () => { throw commonjsError('WRAPPER', 'Unrecognized pinned CommonJS initializer wrapper', id); };
  const exported = ast.body.flatMap(node => node.type === 'ExportNamedDeclaration' ? node.specifiers : [])
    .filter(node => node.exported.name === '__require');
  const name = exported.length === 1 && exported[0].local.name;
  const fn = ast.body.find(node => node.type === 'FunctionDeclaration' && node.id.name === name);
  if (!fn || fn.params.length || fn.async || fn.generator) fail();
  const [guard, mark] = fn.body.body, final = fn.body.body.at(-1);
  const flag = guard?.test?.name;
  if (guard?.type !== 'IfStatement' || guard.test.type !== 'Identifier' || guard.alternate ||
      guard.consequent.type !== 'ReturnStatement' || final?.type !== 'ReturnStatement' ||
      mark?.type !== 'ExpressionStatement' || mark.expression.type !== 'AssignmentExpression' ||
      mark.expression.operator !== '=' || mark.expression.left.type !== 'Identifier' ||
      mark.expression.left.name !== flag || mark.expression.right.value !== 1 ||
      result.code.slice(guard.consequent.argument.start, guard.consequent.argument.end) !==
        result.code.slice(final.argument.start, final.argument.end)) fail();
  const module = final.argument.type === 'MemberExpression' && !final.argument.computed &&
    final.argument.object.type === 'Identifier' && final.argument.property.name === 'exports';
  const storage = module ? final.argument.object.name : final.argument.type === 'Identifier' && final.argument.name;
  if (!storage) fail();
  const declarations = ast.body.filter(node => node.type === 'VariableDeclaration');
  if (!declarations.length || !declarations.some(node => node.kind === 'var' && node.declarations.some(decl =>
    decl.id.type === 'Identifier' && decl.id.name === flag && !decl.init)) ||
      declarations.some(node => node.end > fn.start)) fail();
  const imported = ast.body.find(node => node.type === 'ImportDeclaration' && node.specifiers.some(spec =>
    spec.local.name === storage));
  if (imported ? imported.specifiers.length !== 1 || imported.specifiers[0].type !== 'ImportSpecifier' ||
      imported.specifiers[0].imported.name !== (module ? '__module' : '__exports') ||
      imported.source.value !== `\0${id}?commonjs-${module ? 'module' : 'exports'}`
    : module || !declarations.some(node => node.declarations.some(decl =>
      decl.id.type === 'Identifier' && decl.id.name === storage && !decl.init))) fail();

  const names = new Set();
  walk.full(ast, node => { if (node.type === 'Identifier') names.add(node.name); });
  const unique = base => { while (names.has(base)) base += '_'; names.add(base); return base; };
  const attempt = unique('__f3dCommonjsAttempt'), error = unique('__f3dCommonjsError');
  const code = result.code;
  const lastImport = ast.body.filter(node => node.type === 'ImportDeclaration').at(-1);
  // The compiler can put its module.exports alias between import declarations.
  // Keep imports at module scope and shadow those generated aliases locally.
  const outerDeclarations = declarations.filter(node => node.start < (lastImport?.end ?? 0));
  const first = declarations.find(node => node.start > (lastImport?.end ?? 0)) ?? fn;
  const lineStart = node => {
    const start = code.lastIndexOf('\n', node.start - 1) + 1;
    if (code.slice(start, node.start).trim()) fail();
    return start;
  };
  const afterLine = node => {
    const end = code.indexOf('\n', node.end);
    if (end < 0 || code.slice(node.end, end).trim()) fail();
    return end + 1;
  };
  const edits = [
    [lineStart(first), `var ${attempt};\nfunction ${name} () {\n` +
      `if (!${attempt}) ${attempt} = (function () {\nvar ${storage} = ${module ? '{exports: {}}' : '{}'};\n` +
      outerDeclarations.map(node => code.slice(node.start, node.end) + '\n').join('')],
    [afterLine(mark), 'try {\n'],
    [lineStart(final), `} catch (${error}) {\n${attempt} = null;\nthrow ${error};\n}\n`],
    [afterLine(fn), `return ${name};\n})();\nreturn ${attempt}();\n}\n`],
  ];
  // Insert only whole generated lines. The compiler's mapped source text keeps
  // its columns and source-order deltas; empty map rows account for new glue.
  const map = typeof result.map === 'string' ? JSON.parse(result.map) : result.map;
  const mappings = map?.mappings.split(';');
  let output = code;
  for (const [offset, added] of edits.reverse().sort((a, b) => b[0] - a[0])) {
    output = output.slice(0, offset) + added + output.slice(offset);
    if (mappings) {
      const line = code.slice(0, offset).split('\n').length - 1;
      while (mappings.length < line) mappings.push('');
      mappings.splice(line, 0, ...Array(added.split('\n').length - 1).fill(''));
    }
  }
  return {...result, code: output, map: map ? {...map, mappings: mappings.join(';')} : map};
}

export function commonjsPlugins({moduleSources, resolveModule}) {
  const compiler = commonjs({strictRequires: true, sourceMap: true,
    defaultIsModuleExports: true, requireReturnsDefault: id => id === EMPTY || id.startsWith(JSON_ENTRY),
    ignoreDynamicRequires: false, ignoreTryCatch: false});
  const transform = compiler.transform;
  compiler.transform = async function(code, id) {
    if (id.startsWith('file:') && !isCommonJS(id, moduleSources)) return null;
    const result = await transform.call(this, code, id);
    if (!result?.code) return result;
    const ast = this.parse(result.code);
    if (id.startsWith('file:')) {
      const helpers = new Set();
      for (const node of ast.body) if (node.type === 'ImportDeclaration') {
        if (node.source.value === '\0commonjs-dynamic-modules')
          throw commonjsError('DYNAMIC_REQUIRE', 'Runtime require selection needs a project frontend with an explicit dependency closure', id);
        if (node.source.value === '\0commonjsHelpers.js')
          for (const binding of node.specifiers) helpers.add(binding.local.name);
      }
      walk.simple(ast, {MemberExpression(node) {
        if (helpers.has(node.object.name) && (node.property.name ?? node.property.value) === 'commonjsRequire')
          throw commonjsError('DYNAMIC_REQUIRE', 'Runtime require selection needs a project frontend with an explicit dependency closure', id);
      }});
    }
    return retryableCommonjs(result, id, ast);
  };
  return [compiler, {
    name: 'f3d-commonjs-boundary',
    resolveId(source, importer, options) {
      if (originalEntry(source) || source.startsWith(JSON_ENTRY)) return source;
      if (!options.custom?.['node-resolve']?.isRequire) return null;
      const from = originalEntry(importer) ?? importer;
      // require paths are filesystem strings: percent signs, '?' and '#' in
      // filenames are literal, unlike ESM URL specifiers.
      const request = /^(?:\.{1,2}\/|\/)/.test(source)
        ? pathToFileURL(path.resolve(path.dirname(fileURLToPath(from)), source)).href : source;
      const resolved = resolveModule(request, from, {requestKind: 'require', allowCommonJS: true});
      if (resolved === EMPTY) return resolved;
      if (!resolved.startsWith('file:'))
        throw commonjsError('EXTERNAL_REQUIRE', 'Synchronous require cannot load a live external module', resolved);
      if (path.extname(fileURLToPath(resolved)).toLowerCase() === '.json') {
        // A valid, closed JSON resource has no evaluation effects. Borrow its
        // native JSON-module instance so require() and ESM imports share the
        // same mutable object, including imports made later from lazy chunks.
        // Do not lower to an object literal or a second JSON.parse(): duplicate
        // keys, __proto__, signed zero and module identity belong to the host.
        // Invalid input cannot take this eager path: a source require may be
        // conditional or caught, unlike the native module's linking failure.
        const source = fs.readFileSync(fileURLToPath(resolved), 'utf8').replace(/^\uFEFF/, '');
        try { JSON.parse(source); }
        catch {
          throw commonjsError('JSON_REQUIRE', 'Malformed CommonJS JSON needs a project frontend to preserve synchronous require errors', resolved);
        }
        return JSON_ENTRY + resolved;
      }
      if (!isCommonJS(resolved, moduleSources, {required: true}))
        throw commonjsError('ESM_REQUIRE', 'Requiring an ESM module may change initialization timing; use an ESM import or project frontend', resolved);
      // Even a relative child containing only side effects needs a lazy CJS
      // wrapper. Remember require admission before Rollup loads/transforms it.
      moduleSources.setFormat(resolved, 'commonjs');
      commonjsEntry(resolved, moduleSources); // Validate identity without adding a require facade.
      return resolved;
    },
    load(id) {
      if (id.startsWith(JSON_ENTRY))
        return `export {default} from ${JSON.stringify(id.slice(JSON_ENTRY.length))} with {type: 'json'};`;
      const url = originalEntry(id);
      // Requiring the real source also makes empty/side-effect-only CJS modules
      // receive a wrapper. Every facade forwards the same original instance.
      return url ? `module.exports = require(${JSON.stringify(url)});` : null;
    },
  }];
}
