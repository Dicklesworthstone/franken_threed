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

export function commonjsPlugins({moduleSources, resolveModule}) {
  const compiler = commonjs({strictRequires: true, sourceMap: true,
    defaultIsModuleExports: true, requireReturnsDefault: id => id === EMPTY || id.startsWith(JSON_ENTRY),
    ignoreDynamicRequires: false, ignoreTryCatch: false});
  const transform = compiler.transform;
  compiler.transform = async function(code, id) {
    if (id.startsWith('file:') && !isCommonJS(id, moduleSources)) return null;
    const result = await transform.call(this, code, id);
    if (id.startsWith('file:') && result?.code) {
      const ast = this.parse(result.code), helpers = new Set();
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
    return result;
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
