/**
 * Finite document-relative resource discovery for source-identified Three loaders.
 * This pass copies bytes at their authored URLs; it does not rewrite loading calls,
 * intercept requests, evaluate application code, or change LoadingManager behavior.
 */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as walk from "acorn-walk";
import { isInternalLibraryModule } from "./route_bridge.mjs";
import { gltfAssetUris } from "./pack_gltf.mjs";
import { fbxAssetUris } from "./pack_fbx.mjs";
import { colladaAssetUris } from "./pack_collada.mjs";

const PINNED_ROOT = new URL("../../upstream/three.js/", import.meta.url).href;
const CORE_LOADERS = new Set([
  "TextureLoader", "CubeTextureLoader", "ImageLoader", "ImageBitmapLoader",
  "FileLoader", "AudioLoader", "BufferGeometryLoader", "ObjectLoader",
]);
const SETTINGS = new Set(["setPath", "setResourcePath", "setCrossOrigin", "setWithCredentials", "setRequestHeader"]);
// These pinned loaders fetch one source file; their parsers do not fetch model
// sidecars. OBJ's material creator executes upstream and has its own MTL closure.
// The setters below change decoding, not the URL, and still execute upstream.
const SINGLE_FILE_ADDONS = new Map([
  ["HDRLoader", new Set(["setDataType"])],
  ["RGBELoader", new Set(["setDataType"])],
  ["EXRLoader", new Set(["setDataType", "setOutputFormat", "setPart"])],
  ["FontLoader", new Set()],
  ["STLLoader", new Set()],
  ["PLYLoader", new Set(["setPropertyNameMapping", "setCustomPropertyNameMapping"])],
  ["OBJLoader", new Set(["setMaterials"])],
  // The archive owns its model parts and images. The retained parser creates
  // their Blob URLs; no ZIP contents are extracted or rewritten by packaging.
  ["ThreeMFLoader", new Set()],
]);
const ADDON_FILES = new Map([["ThreeMFLoader", "3MFLoader.js"], ["Rhino3dmLoader", "3DMLoader.js"]]);
const RHINO_LIBRARIES = ["rhino3dm.js", "rhino3dm.wasm"];
const ADDON_SETTINGS = new Map([
  ...SINGLE_FILE_ADDONS,
  ["MTLLoader", new Set(["setMaterialOptions"])],
  ["FBXLoader", new Set()],
  ["ColladaLoader", new Set()],
  ["GLTFLoader", new Set(["setDRACOLoader", "setKTX2Loader", "setMeshoptDecoder"])],
  ["DRACOLoader", new Set(["setDecoderPath", "setDecoderConfig", "setWorkerLimit", "preload"])],
  ["KTX2Loader", new Set(["setTranscoderPath", "setWorkerLimit", "detectSupport"])],
  ["Rhino3dmLoader", new Set(["setLibraryPath", "setWorkerLimit", "setSubdivisionLevel"])],
]);
const ADDON_USES = new Map([
  ["GLTFLoader", new Set(["parse", "parseAsync"])],
  ["DRACOLoader", new Set(["decodeDracoFile", "decodeGeometry"])],
  ["KTX2Loader", new Set(["parse", "init", "detectSupportAsync"])],
  ["MTLLoader", new Set(["parse"])],
  ["FBXLoader", new Set(["parse"])],
  ["ColladaLoader", new Set(["parse"])],
  ["Rhino3dmLoader", new Set(["parse", "parseAsync", "decodeObjects"])],
]);
const ALL_SETTINGS = new Set([...SETTINGS, ...[...ADDON_SETTINGS.values()].flatMap(value => [...value])]);
const isSetting = (loader, method) => SETTINGS.has(method) || ADDON_SETTINGS.get(loader)?.has(method);
const isUse = (loader, method) => ["load", "loadAsync", "dispose"].includes(method) ||
  (SINGLE_FILE_ADDONS.has(loader) && method === "parse") || ADDON_USES.get(loader)?.has(method);
const FUNCTION_TYPES = new Set(["FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"]);
const SCOPE_TYPES = new Set([
  "Program", "BlockStatement", "StaticBlock", "SwitchStatement", "CatchClause",
  "ForStatement", "ForInStatement", "ForOfStatement", "ClassDeclaration", "ClassExpression",
  ...FUNCTION_TYPES,
]);
const LIMIT = 128;

function names(pattern, result = []) {
  if (!pattern) return result;
  if (pattern.type === "Identifier") result.push(pattern.name);
  else if (pattern.type === "RestElement") names(pattern.argument, result);
  else if (pattern.type === "AssignmentPattern") names(pattern.left, result);
  else if (pattern.type === "ArrayPattern") for (const element of pattern.elements) names(element, result);
  else if (pattern.type === "ObjectPattern") for (const property of pattern.properties) names(property.value ?? property.argument, result);
  return result;
}

function memberName(node) {
  if (node?.type !== "MemberExpression" || node.optional) return null;
  if (!node.computed && node.property.type === "Identifier") return node.property.name;
  return node.computed && node.property.type === "Literal" && typeof node.property.value === "string"
    ? node.property.value : null;
}

function canonicalUrl(value) {
  const url = value.startsWith("file:") ? new URL(value) : pathToFileURL(path.resolve(value));
  url.search = "";
  url.hash = "";
  return url.href;
}

function relativeUrl(value) {
  const text = value.trim();
  return text !== "" && !/^(?:[/#]|[a-z][a-z0-9+.-]*:)/i.test(text);
}

/** Returns finite load requests, following lexical const bindings without guessing names. */
export function findThreeLoaderRequests(ast, { resolveModule, packageRootUrl, onDefaultManagerExposure, moduleUrl, source } = {}) {
  const root = packageRootUrl ? canonicalUrl(packageRootUrl).replace(/\/?$/, "/") : PINNED_ROOT;
  // This exact retained module only reads the default manager as the fallback
  // for MaterialCreator.loadTexture(). It neither changes nor exposes it to app
  // code. A modified module or an identically named file keeps the escape rule.
  if (moduleUrl && typeof source === "string" &&
      canonicalUrl(moduleUrl) === root + "examples/jsm/loaders/MTLLoader.js" &&
      createHash("sha256").update(source).digest("hex") === "f1f8f0c6fcfa09671cb609954ea436521d835606853c110c5aff2dc8188f5e61") return [];
  const importedBindings = new Map();
  for (const node of ast.body) {
    if (node.type !== "ImportDeclaration") continue;
    let resolved;
    try { resolved = canonicalUrl(resolveModule(node.source.value)); } catch { continue; }
    const core = isInternalLibraryModule(resolved, root);
    const addons = [...ADDON_SETTINGS.keys()].filter(name =>
      resolved === root + "examples/jsm/Addons.js" ||
      resolved === root + `examples/jsm/loaders/${ADDON_FILES.get(name) ?? name + ".js"}`);
    for (const specifier of node.specifiers) {
      let value;
      if (specifier.type === "ImportNamespaceSpecifier" && core) value = { namespace: true };
      if (specifier.type === "ImportNamespaceSpecifier" && addons.length) value = { addonNamespace: new Set(addons) };
      if (specifier.type === "ImportSpecifier") {
        const imported = specifier.imported.name ?? specifier.imported.value;
        if (core && imported === "DefaultLoadingManager") value = { manager: true };
        if ((core && CORE_LOADERS.has(imported)) ||
            (CORE_LOADERS.has(imported) && resolved === root + `src/loaders/${imported}.js`) ||
            addons.includes(imported)) {
          value = { loader: imported };
        }
      }
      importedBindings.set(specifier.local.name, { kind: "import", value });
    }
  }
  if (![...importedBindings.values()].some((binding) => binding.value)) return [];
  const ancestorsByNode = new WeakMap(), scopes = new Map(), calls = [];
  const put = (scope, name, binding) => {
    if (!scopes.has(scope)) scopes.set(scope, new Map());
    scopes.get(scope).set(name, binding);
  };
  const containing = (ancestors, predicate) => [...ancestors].reverse().find(predicate);
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    ancestorsByNode.set(node, ancestors.slice());
    const parents = ancestors.slice(0, -1);
    if (node.type === "ImportDeclaration") {
      for (const specifier of node.specifiers) {
        put(ast, specifier.local.name, importedBindings.get(specifier.local.name) ?? { kind: "import" });
      }
    } else if (node.type === "VariableDeclaration") {
      for (const declaration of node.declarations) {
        const scope = containing(parents, (parent) => node.kind === "var"
          ? FUNCTION_TYPES.has(parent.type) || parent.type === "Program"
          : SCOPE_TYPES.has(parent.type));
        for (const name of names(declaration.id)) {
          const binding = { kind: node.kind, init: declaration.id.type === "Identifier" ? declaration.init : null, declaration };
          put(scope, name, binding);
        }
      }
    } else if (FUNCTION_TYPES.has(node.type)) {
      for (const parameter of node.params) for (const name of names(parameter)) put(node, name, { kind: "parameter" });
      if (node.id) {
        put(node, node.id.name, { kind: "function" });
        if (node.type === "FunctionDeclaration") put(containing(parents, (parent) => SCOPE_TYPES.has(parent.type)), node.id.name, { kind: "function" });
      }
    } else if (node.type === "CatchClause") {
      for (const name of names(node.param)) put(node, name, { kind: "catch" });
    } else if (node.type === "ClassDeclaration" || node.type === "ClassExpression") {
      if (node.id) {
        put(node, node.id.name, { kind: "class" });
        if (node.type === "ClassDeclaration") put(containing(parents, (parent) => SCOPE_TYPES.has(parent.type)), node.id.name, { kind: "class" });
      }
    }
    if (node.type === "CallExpression") calls.push(node);
  });
  const lookup = (name, node) => {
    for (const parent of [...(ancestorsByNode.get(node) ?? [])].reverse()) {
      const binding = scopes.get(parent)?.get(name);
      if (binding) return binding;
    }
    return null;
  };
  const refs = new Map();
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    if (node.type !== "Identifier") return;
    const binding = lookup(node.name, node);
    if (!binding) return;
    const parent = ancestors.at(-2);
    // Declaration identifiers and uncomputed property keys are not reads.
    if (parent?.type === "VariableDeclarator" && parent.id === node) return;
    if (parent?.type === "MemberExpression" && parent.property === node && !parent.computed) return;
    if (!refs.has(binding)) refs.set(binding, []);
    refs.get(binding).push({ node, parent, ancestors: ancestors.slice() });
  });
  const defaultManager = (node, active = new Set()) => {
    if (node?.type === "Identifier") {
      const binding = lookup(node.name, node);
      if (binding?.value?.manager) return true;
      if (binding?.kind === "const" && !active.has(binding)) return defaultManager(binding.init, new Set(active).add(binding));
    }
    return node?.type === "MemberExpression" && memberName(node) === "DefaultLoadingManager" &&
      node.object.type === "Identifier" && lookup(node.object.name, node.object)?.value?.namespace;
  };
  let managerExposed = false;
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    if (!defaultManager(node)) return;
    const parent = ancestors.at(-2);
    if (parent?.type === "ImportSpecifier" || parent?.type === "VariableDeclarator" && parent.id === node) return;
    // Once the shared manager is exposed to application code, another module
    // can change URL mapping too. Keep its host route across the whole bundle.
    managerExposed = true;
  });
  if (managerExposed) { onDefaultManagerExposure?.(); return []; }
  const combine = (left, right, concatenate) => {
    if (!left || !right || left.length * right.length > LIMIT) return null;
    return [...new Set(left.flatMap((a) => right.map((b) => concatenate(a, b))))];
  };
  const finite = (node, active = new Set()) => {
    if (!node) return null;
    if (node.type === "Literal" && typeof node.value === "string") return [node.value];
    if (node.type === "Identifier") {
      const binding = lookup(node.name, node);
      if (binding?.kind !== "const" || !binding.init || active.has(binding)) return null;
      const next = new Set(active).add(binding);
      return finite(binding.init, next);
    }
    if (node.type === "BinaryExpression" && node.operator === "+") return combine(finite(node.left, active), finite(node.right, active), (a, b) => a + b);
    if (node.type === "ConditionalExpression") {
      const yes = finite(node.consequent, active), no = finite(node.alternate, active);
      return yes && no && yes.length + no.length <= LIMIT ? [...new Set([...yes, ...no])] : null;
    }
    if (node.type === "TemplateLiteral") {
      let result = [node.quasis[0].value.cooked];
      for (let i = 0; i < node.expressions.length; i++) {
        result = combine(result, finite(node.expressions[i], active), (a, b) => a + b + node.quasis[i + 1].value.cooked);
      }
      return result;
    }
    return null;
  };
  const constructor = (node, active = new Set()) => {
    if (node?.type === "Identifier") {
      const binding = lookup(node.name, node);
      if (binding?.kind === "import") return binding.value?.loader ?? null;
      if (binding?.kind === "const" && !active.has(binding)) return constructor(binding.init, new Set(active).add(binding));
    }
    if (node?.type === "MemberExpression" && node.object.type === "Identifier") {
      const binding = lookup(node.object.name, node.object), name = memberName(node);
      if ((binding?.value?.namespace && CORE_LOADERS.has(name)) || binding?.value?.addonNamespace?.has(name)) return name;
    }
    return null;
  };
  const statementList = (node) => containing(ancestorsByNode.get(node) ?? [], (parent) => parent.type === "BlockStatement" || parent.type === "Program");
  const receiverSetter = (use, call) => {
    let receiver = use.callee.object;
    while (receiver?.type === "CallExpression" && ALL_SETTINGS.has(memberName(receiver.callee))) {
      if (receiver === call) return true;
      receiver = receiver.callee.object;
    }
    return false;
  };
  // Only inline data dictionaries are inspected. A const object can still be
  // mutated or aliased; its runtime URL configuration remains source-owned.
  const decoderDictionary = (node) => {
    if (node?.type !== "ObjectExpression" || node.properties.length !== 2) return null;
    const values = new Map();
    for (const property of node.properties) {
      if (property.type !== "Property" || property.kind !== "init" || property.method || property.computed) return null;
      const key = property.key.name ?? property.key.value;
      if (!["js", "wasm"].includes(key) || values.has(key)) return null;
      const candidates = finite(property.value);
      if (!candidates) return null;
      values.set(key, candidates);
    }
    return [...values.values()].flat();
  };
  const setting = (value, method, call) => {
    if (!isSetting(value.loader, method)) return null;
    if (method === "setPath" || method === "setResourcePath") {
      const candidates = finite(call.arguments[0]);
      if (!candidates) return null;
      value[method === "setPath" ? "paths" : "resourcePaths"] = candidates;
    } else if (method === "setDRACOLoader" || method === "setKTX2Loader") {
      value.codecs.set(method, call.arguments[0]);
    } else if (method === "setDecoderPath") {
      const prefixes = finite(call.arguments[0]);
      // The pinned string setter uses LoaderUtils.resolveURL: prefix + name.
      // Preserve both source JS/Wasm branches; do not bake a host capability or
      // decoderConfig.type into the artifact. The dictionary has no JS fallback.
      value.codecUrls = prefixes
        ? prefixes.flatMap(prefix => ["draco_wasm_wrapper.js", "draco_decoder.wasm", "draco_decoder.js"].map(name => prefix + name))
        : decoderDictionary(call.arguments[0]);
      if (!value.codecUrls || value.codecUrls.length > LIMIT) return null;
    } else if (method === "setTranscoderPath") {
      const prefixes = finite(call.arguments[0]);
      if (!prefixes) return null;
      // Empty selects the module-relative defaults, already emitted by the
      // ordinary new URL(..., import.meta.url) asset path.
      value.codecUrls = prefixes.filter(Boolean).flatMap(prefix => ["basis_transcoder.js", "basis_transcoder.wasm"].map(name => prefix + name));
      if (value.codecUrls.length > LIMIT) return null;
    } else if (method === "setLibraryPath") {
      const prefixes = finite(call.arguments[0]);
      if (!prefixes) return null;
      // Rhino's FileLoaders concatenate this prefix; an empty path requests
      // both libraries from the document, rather than module-relative defaults.
      value.codecUrls = prefixes.flatMap(prefix => RHINO_LIBRARIES.map(name => prefix + name));
      if (value.codecUrls.length > LIMIT) return null;
    }
    return value;
  };
  const instance = (node, use, active = new Set()) => {
    if (node?.type === "NewExpression") {
      const loader = constructor(node.callee);
      // An application-supplied manager can map a logical URL to an unrelated
      // runtime resource. Keep that authored route without assuming disk files.
      return loader && node.arguments.length === 0
        ? { loader, origin: node, paths: [""], resourcePaths: [""], codecs: new Map(),
          codecUrls: loader === "Rhino3dmLoader" ? [...RHINO_LIBRARIES] : [] } : null;
    }
    if (node?.type === "CallExpression" && ALL_SETTINGS.has(memberName(node.callee))) {
      const value = instance(node.callee.object, use, active);
      if (!value) return null;
      return setting(value, memberName(node.callee), node);
    }
    if (node?.type !== "Identifier") return null;
    const binding = lookup(node.name, node);
    if (binding?.kind !== "const" || active.has(binding)) return null;
    const value = instance(binding.init, use, new Set(active).add(binding));
    if (!value) return null;
    const mutations = [];
    for (const ref of refs.get(binding) ?? []) {
      // Passing a codec to the pinned GLTFLoader is an understood consumer,
      // not an arbitrary escape. Its setters retain the same codec object and
      // do not change its path. Resolve only the receiving loader here, so
      // inspecting a GLTF/codec relationship cannot recursively inspect itself.
      const attached = ref.parent;
      const attachment = memberName(attached?.callee);
      if (attached?.type === "CallExpression" && attached.arguments[0] === ref.node &&
          (value.loader === "DRACOLoader" && attachment === "setDRACOLoader" ||
            value.loader === "KTX2Loader" && attachment === "setKTX2Loader") &&
          instance(attached.callee.object, attached, new Set(active).add(binding))?.loader === "GLTFLoader") continue;
      const member = ref.parent, call = ref.ancestors.at(-3), method = memberName(member);
      // Only boolean configuration and simple value reads are admitted. An app
      // may replace the field with a method whose `this` can mutate loader paths.
      if (value.loader === "FBXLoader" && member?.object === ref.node && method === "trimAnimationClips") {
        const booleanAssignment = call?.type === "AssignmentExpression" && call.left === member && call.operator === "=" &&
          call.right.type === "Literal" && typeof call.right.value === "boolean";
        const read = call?.type === "VariableDeclarator" && call.init === member ||
          ["IfStatement", "ConditionalExpression"].includes(call?.type) && call.test === member ||
          call?.type === "UnaryExpression" && call.operator === "!" && call.argument === member ||
          call?.type === "ReturnStatement" && call.argument === member;
        if (booleanAssignment || read) continue;
        return null;
      }
      if (member?.object !== ref.node || call?.type !== "CallExpression" || call.callee !== member ||
          !(isUse(value.loader, method) || isSetting(value.loader, method))) return null;
      if (isSetting(value.loader, method)) {
        // Standalone straight-line setters dominate subsequent loads in the
        // same statement list. Conditional or escaping mutations stay dynamic.
        if (receiverSetter(use, call)) continue; // Arguments are not part of the receiver chain.
        let index = ref.ancestors.indexOf(call), outer = call;
        const settings = [{ call, method }];
        while (index >= 2) {
          const member = ref.ancestors[index - 1], next = ref.ancestors[index - 2];
          if (member?.type !== "MemberExpression" || member.object !== outer || next?.type !== "CallExpression" || next.callee !== member) break;
          const name = memberName(member);
          if (!isSetting(value.loader, name) && !isUse(value.loader, name)) return null;
          if (isSetting(value.loader, name)) settings.push({ call: next, method: name });
          outer = next; index -= 2;
          if (!isSetting(value.loader, name)) break; // load() does not return the loader.
        }
        const statement = ref.ancestors[index - 1], list = ref.ancestors[index - 2];
        if (statement?.type !== "ExpressionStatement" || statement.expression !== outer ||
            list !== statementList(call) || list !== statementList(use) || list !== statementList(binding.declaration)) return null;
        if (call.start < use.start) mutations.push(...settings);
      }
    }
    mutations.sort((a, b) => a.call.start - b.call.start || a.call.end - b.call.end);
    for (const { call, method } of mutations) {
      if (!setting(value, method, call)) return null;
    }
    return value;
  };
  const cubeUrls = (node, active = new Set()) => {
    if (node?.type === "Identifier") {
      const binding = lookup(node.name, node);
      if (binding?.kind !== "const" || active.has(binding)) return null;
      // Const arrays remain mutable. Only admit arrays whose references are
      // direct load/loadAsync arguments; mutations and escapes are unresolved.
      for (const ref of refs.get(binding) ?? []) {
        if (ref.parent?.type !== "CallExpression" || ref.parent.arguments[0] !== ref.node ||
            !["load", "loadAsync"].includes(memberName(ref.parent.callee)) ||
            instance(ref.parent.callee.object, ref.parent)?.loader !== "CubeTextureLoader") return null;
      }
      return cubeUrls(binding.init, new Set(active).add(binding));
    }
    if (node?.type !== "ArrayExpression" || node.elements.length !== 6) return null;
    const elements = node.elements.map((element) => finite(element));
    return elements.every(Boolean) && elements.flat().length <= LIMIT ? [...new Set(elements.flat())] : null;
  };
  // Identity alone is enough to invalidate an already admitted ObjectLoader
  // image closure after an unknown setting; this never admits a new request.
  const instanceOrigin = (node, active = new Set()) => {
    if (node?.type === "NewExpression") return node;
    if (node?.type === "CallExpression" && ALL_SETTINGS.has(memberName(node.callee)))
      return instanceOrigin(node.callee.object, active);
    if (node?.type !== "Identifier") return null;
    const binding = lookup(node.name, node);
    return binding?.kind === "const" && !active.has(binding)
      ? instanceOrigin(binding.init, new Set(active).add(binding)) : null;
  };
  const requests = [], objectLoads = new Map(), materialLoads = new Map(), rhinoLoads = new Map(), unresolvedLoads = new Set();
  for (const call of calls) {
    const method = memberName(call.callee);
    if (!["load", "loadAsync", "parse", "parseAsync", "preload", "init", "decodeGeometry", "decodeDracoFile", "decodeObjects"].includes(method)) continue;
    const value = instance(call.callee.object, call);
    if (!value) {
      if (["load", "loadAsync"].includes(method)) unresolvedLoads.add(instanceOrigin(call.callee.object));
      continue;
    }
    if (["load", "loadAsync"].includes(method)) {
      const urls = value.loader === "CubeTextureLoader" ? cubeUrls(call.arguments[0]) : finite(call.arguments[0]);
      let objectGroup;
      if (value.loader === "ObjectLoader") {
        // ObjectLoader keeps its first default base, but reads resourcePath
        // again when asynchronous JSON loading finishes. Preserve all finite
        // first-load bases and explicit resets, including for pending requests.
        if (!objectLoads.has(value.origin)) objectLoads.set(value.origin, { paths: new Set(), requests: [], unknown: false });
        objectGroup = objectLoads.get(value.origin);
        for (const prefix of value.resourcePaths) {
          if (prefix) objectGroup.paths.add(prefix);
          else for (const loaderPath of value.paths) {
            if (loaderPath) objectGroup.paths.add(loaderPath);
            else if (urls) for (const url of urls) objectGroup.paths.add(extractUrlBase(url));
            else objectGroup.unknown = true;
          }
        }
      }
      let materialGroup;
      if (["MTLLoader", "FBXLoader", "ColladaLoader"].includes(value.loader)) {
        // These loaders capture their default path when load() starts, but resourcePath is
        // read later by parse(). Each pending request keeps its own fallback.
        if (!materialLoads.has(value.origin)) materialLoads.set(value.origin, { paths: new Set(), requests: [], unknown: false });
        materialGroup = materialLoads.get(value.origin);
        for (const prefix of value.resourcePaths) materialGroup.paths.add(prefix);
      }
      if (urls && urls.length * value.paths.length * value.resourcePaths.length <= LIMIT) {
        for (const url of urls) for (const prefix of value.paths) for (const resourcePath of value.resourcePaths) {
          const request = { loader: value.loader, url: prefix + url, resourcePath, path: prefix, originalUrl: url };
          requests.push(request);
          objectGroup?.requests.push(request);
          materialGroup?.requests.push(request);
        }
      }
    } else if (!isUse(value.loader, method) && !(value.loader === "DRACOLoader" && method === "preload")) continue;
    const codecs = value.loader === "GLTFLoader"
      ? [...value.codecs.values()].map(node => instance(node, call)).filter(Boolean) : [value];
    if (value.loader === "Rhino3dmLoader") {
      // Initialization starts after the model response arrives and remains
      // cached across loads. Include every finite library selection below,
      // including a setter that runs while an earlier model is still pending.
      if (!rhinoLoads.has(value.origin)) rhinoLoads.set(value.origin, { urls: new Set(), unknown: false });
      for (const url of value.codecUrls) rhinoLoads.get(value.origin).urls.add(url);
      continue;
    }
    for (const codec of codecs) for (const url of codec.codecUrls) {
      requests.push({ loader: codec.loader, url, resourcePath: "", path: "", originalUrl: url });
    }
  }
  for (const call of calls) {
    if (memberName(call.callee) === "setLibraryPath") {
      const group = rhinoLoads.get(instanceOrigin(call.callee.object));
      if (group) {
        const paths = finite(call.arguments[0]);
        if (!paths) group.unknown = true;
        else for (const prefix of paths) for (const name of RHINO_LIBRARIES) group.urls.add(prefix + name);
      }
    }
    if (memberName(call.callee) !== "setResourcePath") continue;
    const origin = instanceOrigin(call.callee.object);
    const paths = finite(call.arguments[0]);
    for (const group of [objectLoads.get(origin), materialLoads.get(origin)]) {
      if (!group) continue;
      if (!paths) group.unknown = true;
      else for (const prefix of paths) group.paths.add(prefix);
    }
  }
  for (const [origin, group] of objectLoads) {
    for (const request of group.requests) request.imagePaths = group.unknown || unresolvedLoads.has(origin) || group.paths.size > LIMIT
      ? null : [...group.paths];
  }
  for (const group of materialLoads.values()) {
    for (const request of group.requests) request.texturePaths = group.unknown || group.paths.size > LIMIT
      ? null : [...new Set([...group.paths].map(prefix => prefix || request.path || extractUrlBase(request.originalUrl)))];
  }
  for (const group of rhinoLoads.values()) {
    if (group.unknown || group.urls.size > LIMIT) continue;
    for (const url of group.urls) requests.push({ loader: "Rhino3dmLoader", url, resourcePath: "", path: "", originalUrl: url });
  }
  return requests;
}

// Discover references, not materials: the original bytes and parser remain in
// the artifact. Match r186 MTLLoader's duplicate keys/materials, first bump map,
// literal-space keyword separator and getTextureParams option-removal order.
function mtlTextureUris(text) {
  const maps = new Map([
    ["map_kd", "map"], ["map_ks", "specularMap"], ["map_ke", "emissiveMap"],
    ["norm", "normalMap"], ["map_bump", "bumpMap"], ["bump", "bumpMap"],
    ["disp", "displacementMap"], ["map_d", "alphaMap"],
  ]);
  const materials = new Map();
  let material;
  for (let line of text.split("\n")) {
    line = line.trim();
    if (!line || line.startsWith("#")) continue;
    const space = line.indexOf(" ");
    const key = (space < 0 ? line : line.slice(0, space)).toLowerCase();
    const value = space < 0 ? "" : line.slice(space + 1).trim();
    if (key === "newmtl") { material = new Map(); materials.set(value, material); }
    else if (material && maps.has(key)) material.set(key, value);
  }
  const uris = new Set();
  for (const properties of materials.values()) {
    const usedMaps = new Set();
    for (const [key, value] of properties) {
      const type = maps.get(key);
      if (!value || usedMaps.has(type)) continue;
      usedMaps.add(type);
      const items = value.split(/\s+/);
      for (const [option, count] of [["-bm", 2], ["-mm", 3], ["-s", 4], ["-o", 4]]) {
        const index = items.indexOf(option);
        if (index >= 0) items.splice(index, count);
      }
      const uri = items.join(" ").trim();
      if (uri) uris.add(uri);
    }
  }
  return uris;
}

/** Copies the finite local closure without changing the loader's original URL strings. */
export function collectThreeLoaderAssets(requests, { documentUrl, documentBaseUrl, fail, onResource }) {
  const result = new Map(), entryDir = path.dirname(fileURLToPath(documentUrl));
  const add = (uri) => {
    if (!relativeUrl(uri)) return null;
    let filename;
    try { filename = fileURLToPath(new URL(uri, documentBaseUrl)); }
    catch { return null; }
    const relative = path.relative(entryDir, filename);
    // Root/external resources keep their original host route. An out-of-root
    // authored route cannot be reproduced inside this application's directory.
    if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) return null;
    // Loader URLs can also name responses provided by the host at runtime.
    // Unlike an explicit module file asset, absence on disk does not prove a
    // missing resource. Retain that request and its source success/error path.
    if (!fs.existsSync(filename) || !fs.statSync(filename).isFile()) return null;
    const fileName = relative.split(path.sep).join("/");
    onResource?.(fileName, new URL(uri, documentBaseUrl).href);
    if (!result.has(fileName)) result.set(fileName, { source: fs.readFileSync(filename), originalFileName: filename });
    return result.get(fileName);
  };
  for (const request of requests) {
    const asset = add(request.url);
    if (!asset) continue;
    if (request.loader === "ColladaLoader") {
      if (!request.texturePaths) continue;
      let uris;
      try { uris = colladaAssetUris(asset.source); }
      catch (error) {
        // The original XML and native parser retain malformed-input behavior.
        if (error.code === "COLLADA_XML") continue;
        if (error.code === "COLLADA_BUDGET") fail?.(error.code, error.message);
        throw error;
      }
      // Both the retained TextureLoader and TGALoader use literal setPath()
      // concatenation; an image's URL is not resolved against the model URL.
      for (const uri of uris) for (const base of request.texturePaths) add(base + uri);
      continue;
    }
    if (request.loader === "FBXLoader") {
      if (!request.texturePaths) continue;
      let uris;
      try { uris = fbxAssetUris(asset.source); }
      catch (error) {
        // Preserve malformed model bytes and the retained loader's parse error.
        // Work bounds remain explicit failures, never a partial resource set.
        if (["FBX_HEADER", "FBX_VERSION", "FBX_NODE", "FBX_PROPERTY"].includes(error.code)) continue;
        if (error.code === "FBX_BUDGET") fail?.(error.code, error.message);
        throw error;
      }
      for (const uri of uris) {
        // FBX uses TextureLoader.setPath(), which prefixes even HTTP URLs.
        for (const base of request.texturePaths) add(base + uri);
      }
      continue;
    }
    if (request.loader === "MTLLoader") {
      if (!request.texturePaths) continue;
      for (const uri of mtlTextureUris(asset.source.toString("utf8"))) {
        // Pinned MaterialCreator only bypasses its prefix for HTTP(S) URLs.
        for (const base of request.texturePaths) add(/^https?:\/\//i.test(uri) ? uri : base + uri);
      }
      continue;
    }
    if (request.loader === "ObjectLoader") {
      if (!request.imagePaths) continue; // An unknown earlier load can own the sticky base.
      let json;
      try { json = JSON.parse(asset.source.toString("utf8")); } catch { continue; }
      // Keep the JSON bytes and source parse errors intact. DataTexture image
      // objects are already embedded; string URLs include ordinary/cube images.
      for (const image of Array.isArray(json?.images) ? json.images : []) {
        for (const uri of Array.isArray(image?.url) ? image.url : [image?.url]) {
          if (typeof uri !== "string") continue;
          for (const base of request.imagePaths) {
            // Match the pinned deserializeImage rule, including a single '/'
            // being concatenated with resourcePath rather than replacing it.
            add(/^(\/\/)|([a-z]+:(\/\/)?)/i.test(uri) ? uri : base + uri);
          }
        }
      }
      continue;
    }
    if (request.loader !== "GLTFLoader") continue;
    const binary = asset.source.length >= 4 && asset.source.readUInt32LE(0) === 0x46546c67;
    const base = request.resourcePath || (request.path
      ? request.path + extractUrlBase(request.originalUrl)
      : extractUrlBase(request.originalUrl));
    for (const uri of gltfAssetUris(asset.source, binary, fail)) {
      if (relativeUrl(uri)) add(base + uri);
    }
  }
  return result;
}

function extractUrlBase(url) {
  const index = url.lastIndexOf("/");
  return index === -1 ? "./" : url.slice(0, index + 1);
}
