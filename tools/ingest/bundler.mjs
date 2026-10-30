/**
 * Rollup integration for FrankenThreeD module ingestion (f3d-04).
 * Provides a Rollup plugin backed by our W3C import-map resolver,
 * enabling real Rollup bundle creation and round-trip re-emission.
 */

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as walk from "acorn-walk";
import { rollup } from "rollup";

import { analyzeModuleAst, createDynamicImportClassifier, classifyImportType } from "./ast_analyzer.mjs";
import { parseHtmlEntries } from "./html_parser.mjs";
import { numericKernelRollupPlugin } from "./numeric_rollup.mjs";
import { marchingCubesRollupPlugin } from "./marching_cubes_rollup.mjs";
import { resolveModuleSpecifier, urlToFilePath } from "./resolver.mjs";
import { moduleWorkerRollupPlugin } from "./module_workers.mjs";
import { moduleSourcesFor } from "./module_sources.mjs";
import { analyzeDocumentFetch, collectDocumentFetchAssets } from "./document_fetch_assets.mjs";
import { gltfAssetUris } from "./pack_gltf.mjs";
import { collectThreeLoaderAssets, findThreeLoaderRequests } from "./three_loader_assets.mjs";
import {
  LOADER_RESOURCES_IMPORT, LOADER_RESOURCES_FILE,
  threeLoaderResourceEdits, createThreeLoaderResourceModule,
} from "./three_loader_resources.mjs";

/**
 * Resolves the directory of a module on disk given its Rollup module ID.
 * Handles file:// URLs, absolute OS paths, inline HTML module IDs, and query/fragment suffixes.
 * @param {string} id
 * @param {string} [fallbackBaseUrl]
 * @returns {string}
 */
function getModuleDir(id, fallbackBaseUrl) {
  let cleanId = id;
  const hashIdx = cleanId.indexOf("#");
  if (hashIdx !== -1) cleanId = cleanId.slice(0, hashIdx);
  const qIdx = cleanId.indexOf("?");
  if (qIdx !== -1) cleanId = cleanId.slice(0, qIdx);

  if (cleanId.startsWith("file://")) {
    return path.dirname(urlToFilePath(cleanId));
  }
  if (path.isAbsolute(cleanId)) {
    return path.dirname(cleanId);
  }
  if (fallbackBaseUrl) {
    const cleanBase = fallbackBaseUrl.split(/[?#]/)[0];
    if (cleanBase.startsWith("file://")) return path.dirname(urlToFilePath(cleanBase));
    if (path.isAbsolute(cleanBase)) return path.dirname(cleanBase);
  }
  return process.cwd();
}

/**
 * Checks whether a specifier is a relative local file path.
 * @param {string} url
 * @returns {boolean}
 */
function isRelativeUrl(url) {
  if (!url || typeof url !== "string") return false;
  const trimmed = url.trim();
  if (!trimmed) return false;
  if (trimmed.startsWith("#") || trimmed.startsWith("//") || trimmed.startsWith("/")) {
    return false;
  }
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
    return false;
  }
  return true;
}

/**
 * Checks whether a URL is an external runtime route (http:, https:, or data:).
 * @param {string} url
 * @returns {boolean}
 */
function isExternalUrl(url) {
  if (!url || typeof url !== "string") return false;
  try {
    const protocol = new URL(url, "file:///").protocol;
    return protocol === "http:" || protocol === "https:" || protocol === "data:";
  } catch {
    return false;
  }
}

/**
 * Creates a Rollup plugin utilizing FrankenThreeD's W3C import-map resolver,
 * static module asset emission (new URL(..., import.meta.url)), and executable
 * finite dynamic import chunk emission (import(flag ? './a.js' : './b.js')).
 * @param {Object} options
 * @param {{ imports?: Record<string, string | null>, scopes?: Record<string, Record<string, string | null>> }} [options.importMap]
 * @param {string} options.mapBaseUrl
 * @param {Map<string, string>} [options.inlineModules]
 * @param {Map<string, string>} [options.retainedModuleUrls] - Exact source URLs mapped to copied output URLs
 * @param {string} [options.packageRootUrl]
 * @param {string} [options.documentUrl] - HTML document owning relative loader URLs
 * @returns {import('rollup').Plugin}
 */
export function f3dRollupPlugin(options = {}) {
  const moduleSources = moduleSourcesFor(options);
  const importMap = options.importMap || { imports: {}, scopes: {} };
  const mapBaseUrl = options.mapBaseUrl;
  const inlineModules = options.inlineModules || new Map();
  const retainedModuleUrls = options.retainedModuleUrls || new Map();
  const emittedAssetsByPath = new Map();
  const emittedChunksByUrl = new Map();
  const emittedJsonByUrl = new Map();
  const emittedJsonTargets = new Set();
  const dynamicChunkRefIds = new Set();
  const gltfAssetRefIds = new Set();
  const documentAssets = new Map();
  const documentResourceURLs = new Map();
  const documentRequests = [];
  const documentFetchSites = [];
  let defaultManagerExposed = false;
  let loaderResourceCalls = false;

  function emitJsonModule(context, url) {
    if (emittedJsonByUrl.has(url)) return emittedJsonByUrl.get(url);
    const filename = urlToFilePath(url), source = fs.readFileSync(filename);
    // Distinct URLs must never share a JSON module instance merely because
    // their bytes match. Keep raw bytes, including BOM/duplicate keys/-0 and
    // malformed lazy input, for the host's native JSON parsing and errors.
    const hash = createHash("sha256").update(url).update("\0").update(source).digest("hex");
    // Preserve the source extension/MIME boundary: `type: 'json'` must not
    // relabel JavaScript (or another resource type) into a valid JSON import.
    const fileName = `assets/f3d-json-${hash}${path.extname(filename)}`;
    context.emitFile({type: "asset", fileName, source, originalFileName: filename});
    const target = "./" + fileName.split("/").map(encodeURIComponent).join("/");
    emittedJsonByUrl.set(url, target);
    emittedJsonTargets.add(target);
    return target;
  }

  function emitModuleAsset(context, assetPath) {
    if (emittedAssetsByPath.has(assetPath)) return emittedAssetsByPath.get(assetPath);
    const source = fs.readFileSync(assetPath);
    const extension = path.extname(assetPath).toLowerCase();
    if (extension !== ".gltf" && extension !== ".glb") {
      const refId = context.emitFile({
        type: "asset", name: path.basename(assetPath), source, originalFileName: assetPath,
      });
      emittedAssetsByPath.set(assetPath, refId);
      return refId;
    }

    // A model's resource base is its URL, not the importing module's URL. Keep
    // the model and local sidecars in one directory tree so original URI strings,
    // query/fragment suffixes and GLB binary chunks remain byte-identical.
    const resources = new Map([[assetPath, source]]);
    const fail = (code, message) => context.error({
      code, message: `${message} (model ${JSON.stringify(assetPath)})`,
    });
    for (const uri of gltfAssetUris(source, extension === ".glb", fail)) {
      // Network, data and document-root URLs retain their authored host route.
      // They must not be interpreted as local paths or downloaded during builds.
      if (!isRelativeUrl(uri)) continue;
      let dependency;
      try { dependency = fileURLToPath(new URL(uri, pathToFileURL(assetPath))); }
      catch { fail("GLTF_RESOURCE_URL", `Invalid local glTF resource URI ${JSON.stringify(uri)}`); }
      if (resources.has(dependency)) continue;
      let content;
      try {
        if (!fs.statSync(dependency).isFile()) throw new Error("not a regular file");
        content = fs.readFileSync(dependency);
      } catch (error) {
        fail("GLTF_RESOURCE_MISSING", `Cannot read glTF resource ${JSON.stringify(uri)}: ${error.message}`);
      }
      resources.set(dependency, content);
    }

    let resourceRoot = path.dirname(assetPath);
    for (const filename of resources.keys()) {
      for (;;) {
        const relative = path.relative(resourceRoot, filename);
        if (relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative)) break;
        const parent = path.dirname(resourceRoot);
        if (parent === resourceRoot) fail("GLTF_RESOURCE_ROOT", "Model resources have no common local root");
        resourceRoot = parent;
      }
    }
    const localName = (filename) => path.relative(resourceRoot, filename).split(path.sep).join("/");
    const ordered = [...resources].sort(([a], [b]) => localName(a) < localName(b) ? -1 : localName(a) > localName(b) ? 1 : 0);
    // Include logical location and every dependency, so identically named models
    // do not collide and a changed image/buffer invalidates the model URL too.
    const hash = createHash("sha256");
    hash.update(path.relative(getModuleDir(mapBaseUrl ?? assetPath), assetPath).split(path.sep).join("/"));
    for (const [filename, bytes] of ordered) {
      hash.update("\0" + localName(filename) + "\0" + bytes.length + "\0");
      hash.update(bytes);
    }
    const directory = `assets/gltf-${hash.digest("hex").slice(0, 24)}/`;
    let modelRefId;
    for (const [filename, bytes] of ordered) {
      const refId = context.emitFile({
        type: "asset", fileName: directory + localName(filename), source: bytes, originalFileName: filename,
      });
      if (filename === assetPath) modelRefId = refId;
    }
    emittedAssetsByPath.set(assetPath, modelRefId);
    gltfAssetRefIds.add(modelRefId);
    return modelRefId;
  }

  return {
    name: "f3d-ingest-resolver",

    generateBundle(_outputOptions, bundle) {
      for (const { requests, id } of defaultManagerExposed ? [] : documentRequests) {
        for (const [fileName, asset] of collectThreeLoaderAssets(requests, {
          documentUrl: options.documentUrl,
          documentBaseUrl: mapBaseUrl,
          onResource: (filename, url) => {
            const root = new URL("./", options.documentUrl).href;
            if (!documentResourceURLs.has(filename)) documentResourceURLs.set(filename, new Set());
            documentResourceURLs.get(filename).add(url.slice(root.length));
          },
          fail: (code, message) => this.error({ code, message: `${message} (loader in ${JSON.stringify(id)})` }),
        })) documentAssets.set(fileName, asset);
      }
      const resources = new Map(documentAssets);
      if (options.documentUrl) for (const [name, asset] of collectDocumentFetchAssets(documentFetchSites, {
        documentUrl: options.documentUrl, documentBaseUrl: mapBaseUrl,
      })) {
        if (resources.has(name) && !Buffer.from(resources.get(name).source).equals(asset.source))
          this.error(`Document fetch bytes disagree with loader resource ${JSON.stringify(name)}`);
        resources.set(name, asset);
      }
      for (const [fileName, asset] of resources) {
        if (options.documentUrl && fileName === path.basename(fileURLToPath(options.documentUrl))) {
          this.error(`Document resource ${JSON.stringify(fileName)} collides with the emitted HTML entry`);
        }
        const existing = bundle[fileName];
        if (existing) {
          if (existing.type === "asset" && Buffer.from(existing.source).equals(asset.source)) continue;
          this.error(`Document resource ${JSON.stringify(fileName)} collides with an emitted bundle file`);
        }
        this.emitFile({ type: "asset", fileName, ...asset });
      }
      if (loaderResourceCalls) {
        if (bundle[LOADER_RESOURCES_FILE] || documentAssets.has(LOADER_RESOURCES_FILE))
          this.error("Embedded loader resolver collides with an emitted application resource");
        this.emitFile({ type: "asset", fileName: LOADER_RESOURCES_FILE,
          source: createThreeLoaderResourceModule(documentAssets, { logicalURLs: documentResourceURLs }) });
      }
    },

    resolveFileUrl({ referenceId, relativePath }) {
      if (gltfAssetRefIds.has(referenceId)) {
        // Explicit filenames preserve decoded source names, including literal
        // spaces, percent signs, '?' and '#'; encode each path segment as a URL.
        const url = relativePath.split("/").map(encodeURIComponent).join("/");
        return `new URL(${JSON.stringify(url)}, import.meta.url).href`;
      }
      if (dynamicChunkRefIds.has(referenceId)) {
        const rel =
          relativePath.startsWith("./") || relativePath.startsWith("../")
            ? relativePath
            : `./${relativePath}`;
        return JSON.stringify(rel);
      }
      return null;
    },

    resolveId(source, importer, {attributes = {}} = {}) {
      if (emittedJsonTargets.has(source)) return {id: source, external: true, attributes};
      if (source === LOADER_RESOURCES_IMPORT && options.prepareStandaloneLoaders)
        return { id: "./" + LOADER_RESOURCES_FILE, external: true, moduleSideEffects: false };
      if (inlineModules.has(source)) {
        return source;
      }

      const referrerUrl = importer
        ? importer.startsWith("file://")
          ? importer
          : pathToFileURL(path.resolve(importer)).href
        : mapBaseUrl;

      const resolvedUrl = resolveModuleSpecifier(source, referrerUrl, importMap, {
        mapBaseUrl,
        packageRootUrl: options.packageRootUrl,
      });
      if (importer && retainedModuleUrls.has(resolvedUrl)) {
        // Both retained imports and bundled imports must reach the same browser module.
        // These URLs keep their query/fragment identity and are relative to output chunks.
        return { id: retainedModuleUrls.get(resolvedUrl), external: true };
      }
      if (isExternalUrl(resolvedUrl)) {
        return { id: resolvedUrl, external: true };
      }
      if (importer && (attributes.type === "json" || path.extname(urlToFilePath(resolvedUrl)).toLowerCase() === ".json")) {
        // Untyped .json imports also stay native so missing/invalid attributes
        // cannot accidentally become a successful JavaScript module import.
        return {id: emitJsonModule(this, resolvedUrl), external: true, attributes};
      }
      return resolvedUrl;
    },

    load(id) {
      if (inlineModules.has(id)) {
        return moduleSources.read(id, {source: inlineModules.get(id)}).code;
      }

      if (id.startsWith("file://")) {
        const source = moduleSources.read(id).code;
        if (retainedModuleUrls.has(id)) {
          // Rollup entry points cannot be external. Forward an HTML entry to its
          // retained module without evaluating a second copy of its body.
          const hasDefault = this.parse(source).body.some(
            (node) =>
              node.type === "ExportDefaultDeclaration" ||
              (node.type === "ExportAllDeclaration" &&
                (node.exported?.name ?? node.exported?.value) === "default") ||
              (node.type === "ExportNamedDeclaration" &&
                node.specifiers.some(
                  (spec) => (spec.exported.name ?? spec.exported.value) === "default",
                )),
          );
          return (
            `export * from ${JSON.stringify(id)};\n` +
            (hasDefault ? `export { default } from ${JSON.stringify(id)};\n` : "")
          );
        }
        return source;
      }

      return null;
    },

    transform(code, id) {
      if (!code || typeof code !== "string") return null;

      const moduleDir = getModuleDir(id, mapBaseUrl);
      const referrerUrl = id.startsWith("file://")
        ? id
        : path.isAbsolute(id)
          ? pathToFileURL(id).href
          : mapBaseUrl;

      const edits = [];

      // 1. Asset reference replacements: new URL(..., import.meta.url)
      const analysis = analyzeModuleAst(code, id);
      const assetRefs = analysis && analysis.assetReferences ? analysis.assetReferences : [];

      for (const ref of assetRefs) {
        const rawSpecifier = ref.specifier;
        if (!rawSpecifier || typeof rawSpecifier !== "string") continue;

        const qIdx = rawSpecifier.indexOf("?");
        const hIdx = rawSpecifier.indexOf("#");
        let splitIdx = -1;
        if (qIdx !== -1 && hIdx !== -1) splitIdx = Math.min(qIdx, hIdx);
        else if (qIdx !== -1) splitIdx = qIdx;
        else if (hIdx !== -1) splitIdx = hIdx;

        const cleanPath = splitIdx !== -1 ? rawSpecifier.slice(0, splitIdx) : rawSpecifier;
        const suffix = splitIdx !== -1 ? rawSpecifier.slice(splitIdx) : "";

        if (!isRelativeUrl(cleanPath)) continue;

        let assetAbsPath;
        try {
          const dirSlash = moduleDir.endsWith(path.sep) ? moduleDir : moduleDir + path.sep;
          const resolvedUrl = new URL(cleanPath, pathToFileURL(dirSlash));
          assetAbsPath = fileURLToPath(resolvedUrl);
        } catch {
          assetAbsPath = path.resolve(moduleDir, cleanPath);
        }

        if (!fs.existsSync(assetAbsPath) || !fs.statSync(assetAbsPath).isFile()) {
          throw new Error(
            `Unresolved module asset: "${rawSpecifier}" not found at "${assetAbsPath}" referenced from "${id}"`,
          );
        }

        const refId = emitModuleAsset(this, assetAbsPath);

        const span = ref.source_span || ref.sourceSpan;
        const start = span.start.offset;
        const end = span.end.offset;

        const replacement = suffix
          ? `new URL(import.meta.ROLLUP_FILE_URL_${refId} + ${JSON.stringify(suffix)})`
          : `new URL(import.meta.ROLLUP_FILE_URL_${refId})`;

        edits.push({
          type: "replace",
          pos: start,
          end: end,
          priority: 2,
          text: replacement,
        });
      }

      // 2. Finite dynamic import replacements: import(flag ? './a.js' : './b.js')
      // Emits candidate branches as separate chunks (with preserveSignature: 'strict')
      // and wraps the dynamic argument with a safe own-key runtime specifier mapping.
      let ast;
      try {
        ast = this.parse(code);
      } catch {
        ast = null;
      }

      if (ast) {
        if (options.documentUrl && code.includes("fetch"))
          documentFetchSites.push(...analyzeDocumentFetch(ast).sites);
        const classifyImport = createDynamicImportClassifier(ast);
        const inspectorImports = new Map(analysis.dynamicImports.filter(request => request.runtime_fallback === true)
          .map(request => [request.source_span.start.offset, request]));
        if (options.prepareStandaloneLoaders && options.documentUrl && id.startsWith("file:")) {
          const loaderEdits = threeLoaderResourceEdits(code, ast, id, options.packageRootUrl);
          if (loaderEdits.length) { edits.push(...loaderEdits); loaderResourceCalls = true; }
        }
        if (options.documentUrl && (code.includes("load") || code.includes("LoadingManager"))) {
          const requests = findThreeLoaderRequests(ast, {
            moduleUrl: id, source: code,
            resolveModule: (specifier) => resolveModuleSpecifier(specifier, referrerUrl, importMap, {
              mapBaseUrl, packageRootUrl: options.packageRootUrl,
            }),
            packageRootUrl: options.packageRootUrl,
            onDefaultManagerExposure: () => { defaultManagerExposed = true; },
          });
          if (requests.length) documentRequests.push({ requests, id });
        }
        walk.simple(ast, {
          ImportExpression: (node) => {
            if (!node.source) return;
            const classified = inspectorImports.get(node.start) ?? classifyImport(node.source);
            const moduleType = classifyImportType(node);
            const candidates = classified.classification === "literal" && (moduleType === "json" || node.options)
              ? [classified.specifier] : classified.classification === "finite_set" ? classified.candidates : null;
            if (
              !candidates || candidates.length === 0
            ) {
              return;
            }

            const uniqueBranches = Array.from(new Set(candidates));
            const specifierMap = new Map();
            let bundledTarget = false;

            for (const cand of uniqueBranches) {
              const resolvedUrl = resolveModuleSpecifier(cand, referrerUrl, importMap, {
                mapBaseUrl,
                packageRootUrl: options.packageRootUrl,
              });

              if (retainedModuleUrls.has(resolvedUrl)) {
                specifierMap.set(cand, JSON.stringify(retainedModuleUrls.get(resolvedUrl)));
              } else if (isExternalUrl(resolvedUrl)) {
                specifierMap.set(cand, JSON.stringify(resolvedUrl));
              } else if (moduleType === "json" || path.extname(urlToFilePath(resolvedUrl)).toLowerCase() === ".json") {
                specifierMap.set(cand, JSON.stringify(emitJsonModule(this, resolvedUrl)));
              } else {
                bundledTarget = true;
                let chunkRefId = emittedChunksByUrl.get(resolvedUrl);
                if (!chunkRefId) {
                  chunkRefId = this.emitFile({
                    type: "chunk",
                    id: resolvedUrl,
                    preserveSignature: "strict",
                  });
                  emittedChunksByUrl.set(resolvedUrl, chunkRefId);
                  dynamicChunkRefIds.add(chunkRefId);
                }
                specifierMap.set(cand, `import.meta.ROLLUP_FILE_URL_${chunkRefId}`);
              }
            }

            if (specifierMap.size > 0) {
              if (classified.classification === "literal") {
                // Replacing a literal has no selector effects. Keeping a
                // one-argument identity call for a bundled URL prevents
                // Rollup from dropping import options and remains packable.
                const target = specifierMap.get(classified.specifier);
                edits.push({type: "replace", pos: node.source.start, end: node.source.end,
                  priority: 2, text: bundledTarget ? `(s => s)(${target})` : target});
                return;
              }
              const chain = Array.from(specifierMap.entries())
                .map(([cand, targetExpr]) => `s === ${JSON.stringify(cand)} ? ${targetExpr} : `)
                .join("");

              const prefix = `(s => ${chain}s)(`;
              const suffix = `)`;
              const argument = classified.url_argument_span ?? node.source;

              // Composable insertions wrapping the argument preserve child edits inside the argument
              edits.push({
                type: "insert",
                pos: argument.start,
                priority: 1,
                text: prefix,
              });
              edits.push({
                type: "insert",
                pos: argument.end,
                priority: 3,
                text: suffix,
              });
            }
          },
        });
      }

      if (edits.length === 0) return null;

      // Sort descending by position; for coincident positions: suffix (end) before replace before prefix (start)
      edits.sort((a, b) => {
        if (b.pos !== a.pos) return b.pos - a.pos;
        return (b.priority || 0) - (a.priority || 0);
      });

      let transformedCode = code;
      for (const e of edits) {
        if (e.type === "insert") {
          transformedCode = transformedCode.slice(0, e.pos) + e.text + transformedCode.slice(e.pos);
        } else if (e.type === "replace") {
          transformedCode = transformedCode.slice(0, e.pos) + e.text + transformedCode.slice(e.end);
        }
      }

      return {
        code: transformedCode,
        map: null,
      };
    },
  };
}

/**
 * @typedef {Object} EmittedChunk
 * @property {string} fileName - Emitted relative file name
 * @property {string} code - Generated JavaScript source code
 * @property {string[]} modules - Module IDs included in this chunk
 * @property {boolean} isEntry - True if this chunk corresponds to a root entry point
 * @property {boolean} isDynamicEntry - True if generated from a dynamic import()
 * @property {string[]} imports - Static dependency chunk file names
 * @property {string[]} dynamicImports - Dynamic dependency chunk file names
 * @property {string | null} facadeModuleId - Module ID for entry chunks
 */

/**
 * Executes an actual Rollup build for an HTML or ESM entry point,
 * producing bundled output code.
 *
 * @param {string} entryPath
 * @param {Object} [options]
 * @returns {Promise<{
 *   code: string,
 *   modules: string[],
 *   isMultiChunk: boolean,
 *   entryFiles: string[],
 *   files: Record<string, string>,
 *   outputChunks: EmittedChunk[],
 *   chunks: EmittedChunk[]
 * }>}
 */
const ROUTED_RENDERER_PREFIX = "\0f3d-renderer-route:";
const THREE_RENDERER_URL = new URL("./three_renderer.mjs", import.meta.url).href;
const RENDERER_FACTORIES = {
  WebGPURenderer: "createWebGPURendererClass",
  WebGLRenderer: "createWebGLRendererClass",
};

/**
 * Renderer-construction route substitution for an already decided build.
 *
 * Every import that resolves to a target URL (an admitted pinned build such as
 * `build/three.webgpu.js` or `build/three.module.js`) is redirected to a facade
 * module that re-exports the complete upstream namespace -- every class keeps its
 * upstream module identity -- and replaces only the named renderer binding with
 * the general new-backend implementation from three_renderer.mjs. Requests the
 * new backend cannot own (forceWebGL, a supplied context) construct the unchanged
 * upstream renderer. The caller decided the route; this plugin decides nothing.
 *
 * @param {{ targets: Array<{ url: string, exportName: 'WebGPURenderer' | 'WebGLRenderer' }> }} options
 * @returns {import('rollup').Plugin}
 */
export function rendererRoutePlugin({ targets, exactSites = [] }) {
  const byUrl = new Map(targets.map((t) => [t.url, t.exportName]));
  if (byUrl.size === 0) throw new Error("rendererRoutePlugin requires target modules");
  for (const name of byUrl.values())
    if (!RENDERER_FACTORIES[name]) throw new Error(`Unsupported routed renderer export: ${name}`);
  const routed = new Set();
  const sourceSites = new Map();
  for (const site of exactSites) {
    if (!sourceSites.has(site.moduleId)) sourceSites.set(site.moduleId, []);
    sourceSites.get(site.moduleId).push(site);
  }
  return {
    name: "f3d-renderer-route",
    transform(code, id) {
      const sites = sourceSites.get(id);
      if (!sites?.length) return null;
      const hash = 'sha256:' + createHash('sha256').update(code).digest('hex');
      if (sites.some(site => site.sourceHash !== hash))
        throw new Error(`Source-selected renderer module changed before emission: ${id}`);
      const analysis = analyzeModuleAst(code, id), ast = this.parse(code), spans = new Map();
      for (const site of sites) {
        const current = analysis.renderer_construction_sites[site.siteIndex];
        if (current?.constructor_name !== 'WebGPURenderer' || current.force_webgl !== true)
          throw new Error(`Source-selected renderer construction changed before emission: ${id}`);
        spans.set(current.source_span.start.offset, current.source_span.end.offset);
      }
      const names = new Set();
      walk.full(ast, node => { if (node.type === 'Identifier') names.add(node.name); });
      let helper = '__f3dSourceRenderer';
      while (names.has(helper)) helper += '_';
      const edits = [];
      walk.simple(ast, {NewExpression(node) {
        if (spans.get(node.start) !== node.end) return;
        // Evaluate the original callee once, before its untouched argument list.
        // The private registry selects the actual source constructor belonging
        // to this facade, even with multiple independent Three module copies.
        edits.push({start: node.callee.start, end: node.callee.end,
          text: `(${helper}(${code.slice(node.callee.start, node.callee.end)}))`});
      }});
      if (edits.length !== sites.length) throw new Error(`Source-selected renderer sites were not emitted: ${id}`);
      for (const edit of edits.sort((a, b) => b.start - a.start))
        code = code.slice(0, edit.start) + edit.text + code.slice(edit.end);
      code += `\nimport { sourceThreeRendererConstructor as ${helper} } from ${JSON.stringify(THREE_RENDERER_URL)};\n`;
      return {code, map: null};
    },
    async resolveId(source, importer, resolveOptions) {
      if (source.startsWith(ROUTED_RENDERER_PREFIX)) return source;
      // Resolve facade imports from the backing module's URL. Passing the
      // virtual ID to the ingestion resolver turns its NUL prefix into an
      // invalid filesystem path during package browser-map inspection. Skip
      // this plugin so the facade reaches the real upstream module once.
      if (importer?.startsWith(ROUTED_RENDERER_PREFIX)) {
        return this.resolve(source, importer.slice(ROUTED_RENDERER_PREFIX.length), {
          ...resolveOptions, skipSelf: true,
        });
      }
      const resolved = await this.resolve(source, importer, { ...resolveOptions, skipSelf: true });
      if (!resolved || resolved.external || !byUrl.has(resolved.id)) return null;
      routed.add(resolved.id);
      return ROUTED_RENDERER_PREFIX + resolved.id;
    },
    load(id) {
      if (!id.startsWith(ROUTED_RENDERER_PREFIX)) return null;
      const url = id.slice(ROUTED_RENDERER_PREFIX.length);
      const upstream = JSON.stringify(url), name = byUrl.get(url), factory = RENDERER_FACTORIES[name];
      return (
        `export * from ${upstream};\n` +
        `import * as F3D_THREE from ${upstream};\n` +
        `import { ${factory}, createPMREMGeneratorClass } from ${JSON.stringify(THREE_RENDERER_URL)};\n` +
        `export const ${name} = ${factory}(F3D_THREE, { exactBackend: F3D_THREE.${name} });\n` +
        // PMREM needs renderer internals; new-backend renderers get the
        // environment-capture substitute, any other renderer the upstream class.
        `export const PMREMGenerator = createPMREMGeneratorClass(F3D_THREE, { exactBackend: F3D_THREE.PMREMGenerator });\n`
      );
    },
    api: {
      get routedModules() {
        return [...routed];
      },
    },
  };
}

export async function bundleWithRollup(entryPath, options = {}) {
  const moduleSources = moduleSourcesFor(options);
  const specialization = options.specializeNumeric;
  const numericPlugin =
    specialization === undefined || specialization === false
      ? null
      : numericKernelRollupPlugin(specialization === true ? {} : specialization);
  // The existing application opt-in covers source-pinned library kernels too.
  // Both passes share caller budgets, but retain their own default ceilings.
  // Library kernels are accounted separately from generic inferred functions.
  const marchingPlugin = numericPlugin
    ? marchingCubesRollupPlugin({
        maxMemoryPages: specialization === true ? undefined : specialization.maxMemoryPages,
        maxIterations: specialization === true ? undefined : specialization.maxIterations,
      })
    : null;
  const resolvedEntryAbs = path.resolve(entryPath);
  const entryUrl = pathToFileURL(resolvedEntryAbs).href;
  const isHtml = entryPath.endsWith(".html") || entryPath.endsWith(".htm");

  let importMap = { imports: {}, scopes: {} };
  let mapBaseUrl = entryUrl;
  const inlineModules = new Map();
  let input;
  const orderedEntryIds = [];

  if (isHtml) {
    const htmlContent = fs.readFileSync(resolvedEntryAbs, "utf-8");
    const parsed = parseHtmlEntries(htmlContent, entryUrl);
    importMap = parsed.importMap;
    mapBaseUrl = parsed.baseUrl;

    if (parsed.moduleScripts.length === 0) {
      throw new Error(`No module scripts found in ${entryPath}`);
    }

    // Empty src is a browser script error, never an inline module or an HTML import.
    const localModuleScripts = parsed.moduleScripts.filter(
      (s) => s.src !== "" && !isExternalUrl(s.id),
    );

    // Browser-owned external/empty-src scripts do not emit Rollup chunks.
    if (localModuleScripts.length === 0) {
      return {
        code: "",
        modules: [],
        isMultiChunk: false,
        entryFiles: [],
        files: {},
        outputChunks: [],
        chunks: [],
        assets: [],
      };
    }

    for (const s of localModuleScripts) {
      if (s.inlineContent !== null) {
        inlineModules.set(s.id, s.inlineContent);
      }
      orderedEntryIds.push(s.id);
    }

    // Deduplicate Rollup input IDs by exact canonical resolved URL id only.
    // Preserves query strings (?query) and fragments (#fragment) as distinct ES module identities.
    const seenInputIds = new Set();
    const uniqueInputScripts = [];

    for (const s of localModuleScripts) {
      if (!seenInputIds.has(s.id)) {
        seenInputIds.add(s.id);
        uniqueInputScripts.push(s);
      }
    }

    if (uniqueInputScripts.length === 1 && uniqueInputScripts[0].src) {
      input = uniqueInputScripts[0].id;
    } else {
      input = {};
      uniqueInputScripts.forEach((s, idx) => {
        let name;
        if (s.src) {
          const cleanSrc = s.src.split("?")[0].split("#")[0];
          name = path.basename(cleanSrc).replace(/\.[^/.]+$/, "") || "script";
        } else {
          name = `inline_${idx}`;
        }
        let entryKey = name;
        let counter = 1;
        while (Object.hasOwn(input, entryKey)) {
          entryKey = `${name}_${counter++}`;
        }
        input[entryKey] = s.id;
      });
    }
  } else {
    input = entryUrl;
    orderedEntryIds.push(entryUrl);
  }

  const routePlugin = options.rendererRoute ? rendererRoutePlugin(options.rendererRoute) : null;
  let bundle;
  try {
    bundle = await rollup({
      input,
      plugins: [
        ...(routePlugin ? [routePlugin] : []),
        // Resolve the compiler-owned virtual modules before the import-map
        // resolver sees them; they are not application package/asset paths.
        ...(marchingPlugin ? [marchingPlugin] : []),
        // A worker is an independent ESM realm, not a raw URL asset and not a
        // document chunk. Do not apply the page import map or renderer route.
        moduleWorkerRollupPlugin({
          _moduleSources: moduleSources,
          ...(isHtml ? { documentUrl: entryUrl, documentBaseUrl: mapBaseUrl } : {}),
          resolveModule: (specifier, from) => resolveModuleSpecifier(specifier, from, {}, {
            packageRootUrl: options.packageRootUrl,
          }),
        }),
        f3dRollupPlugin({
          _moduleSources: moduleSources,
          importMap,
          mapBaseUrl,
          inlineModules,
          packageRootUrl: options.packageRootUrl,
          retainedModuleUrls: options.retainedModuleUrls,
          prepareStandaloneLoaders: options.prepareStandaloneLoaders,
          ...(isHtml ? { documentUrl: entryUrl } : {}),
        }),
        ...(numericPlugin ? [numericPlugin] : []),
      ],
      onwarn(warning, warn) {
        // Suppress known non-fatal warnings (e.g. eval in 3rd party libs or circular deps)
        if (warning.code === "CIRCULAR_DEPENDENCY" || warning.code === "THIS_IS_UNDEFINED") {
          return;
        }
        // Forward all other warnings honestly
        warn(warning);
      },
    });

    const { output } = await bundle.generate({
      format: "es",
      // r186 applications use native import attributes; Rollup's default
      // legacy `assert` syntax is rejected by current supported hosts.
      importAttributesKey: "with",
      // An application may await compileAsync/renderAsync at module scope. The
      // renderer then imports optional UV/fog/background modules, which share
      // helpers with its static runtime. Keep that runtime (and its static
      // dependencies) outside the still-evaluating application entry; otherwise
      // Rollup's shared exports introduce a dynamic-import/TLA deadlock.
      ...(routePlugin ? {
        manualChunks: (id) => id.startsWith(ROUTED_RENDERER_PREFIX)
          ? "f3d-renderer-runtime" : undefined,
      } : {}),
      entryFileNames: (chunk) =>
        options.retainedModuleUrls?.has(chunk.facadeModuleId)
          ? "f3d-entry-[name]-[hash].js"
          : "[name].js",
    });

    let numericReport = numericPlugin?.api.getReport();
    if (numericReport && marchingPlugin) {
      numericReport = {
        ...numericReport,
        libraryKernels: { marchingCubes: marchingPlugin.api.getReport(),
          geometryNormals: marchingPlugin.api.getGeometryNormalsReport(),
          geometryTangents: marchingPlugin.api.getGeometryTangentsReport(),
          geometryTransforms: marchingPlugin.api.getGeometryTransformsReport(),
          geometryBounds: marchingPlugin.api.getGeometryBoundsReport(),
          matrixMultiplication: marchingPlugin.api.getMatrixMultiplicationReport(),
          quaternionFlat: marchingPlugin.api.getQuaternionFlatReport(),
          transformBatches: marchingPlugin.api.getTransformBatchesReport(),
          eulerUpdates: marchingPlugin.api.getEulerUpdatesReport(),
          instancedMatrixUpdates: marchingPlugin.api.getInstancedMatrixUpdatesReport() },
      };
      const reportAsset = output.find(
        (item) => item.type === "asset" && item.fileName === numericReport.reportFile,
      );
      if (!reportAsset) throw new Error("Numeric specialization report asset is missing");
      // This explicitly named JSON asset is not executable and has no content
      // hash in its filename. Keep emitted JSON and the build manifest identical.
      reportAsset.source = JSON.stringify(numericReport, null, 2) + "\n";
    }

    const chunks = output.filter((chunk) => chunk.type === "chunk");
    const assets = output.filter((item) => item.type === "asset");

    // Match entry chunks in exact HTML document order.
    // If an HTML document contains repeated module script references with identical URL
    // (e.g. two <script type="module" src="./same.mjs"> tags), entryFiles preserves the exact
    // 1:1 document-order mapping with duplicate file references so HTML re-emission
    // keeps both script tags pointing to the same emitted chunk, while underlying
    // chunks and files remain unique. Distinct URLs (including query variants) map to
    // their own distinct emitted chunks.
    const entryFiles = [];
    const entryChunks = [];
    const seenEntryChunks = new Set();

    for (const entryId of orderedEntryIds) {
      const chunk = chunks.find((c) => c.facadeModuleId === entryId);

      if (!chunk) {
        throw new Error(`Failed to find emitted chunk for entry module: ${entryId}`);
      }

      entryFiles.push(chunk.fileName);
      if (!seenEntryChunks.has(chunk)) {
        entryChunks.push(chunk);
        seenEntryChunks.add(chunk);
      }
    }

    const primaryChunk = entryChunks[0] || chunks[0] || null;

    // Collect all modules across all emitted chunks
    const allModules = [];
    const seenModules = new Set();
    for (const chunk of chunks) {
      for (const mod of Object.keys(chunk.modules || {})) {
        if (!seenModules.has(mod)) {
          seenModules.add(mod);
          allModules.push(mod);
        }
      }
    }

    const outputChunks = chunks.map((c) => ({
      fileName: c.fileName,
      code: c.code,
      modules: Object.keys(c.modules || {}),
      isEntry: Boolean(c.isEntry),
      isDynamicEntry: Boolean(c.isDynamicEntry),
      imports: c.imports || [],
      dynamicImports: c.dynamicImports || [],
      facadeModuleId: c.facadeModuleId || null,
    }));

    const outputAssets = assets.map((a) => ({
      fileName: a.fileName,
      name: a.name,
      source: a.source,
    }));

    const files = Object.fromEntries([
      ...chunks.map((c) => [c.fileName, c.code]),
      ...assets.map((a) => [
        a.fileName,
        typeof a.source === "string" ? a.source : Buffer.from(a.source),
      ]),
    ]);

    return {
      // Backward compatibility for single-chunk callers
      code: primaryChunk ? primaryChunk.code : "",
      modules: allModules,

      // Multi-chunk & dynamic chunk support
      isMultiChunk: chunks.length > 1,
      entryFiles,
      files,
      outputChunks,
      chunks: outputChunks,
      assets: outputAssets,
      ...(numericReport ? { numericSpecialization: numericReport } : {}),
      ...(routePlugin ? { routedRendererModules: routePlugin.api.routedModules } : {}),
    };
  } finally {
    if (bundle) {
      await bundle.close();
    }
  }
}
