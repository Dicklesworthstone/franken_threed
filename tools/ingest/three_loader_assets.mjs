/**
 * Finite document-relative resource discovery for source-identified Three loaders.
 * This pass copies bytes at their authored URLs; it does not rewrite loading calls,
 * intercept requests, evaluate application code, or change LoadingManager behavior.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as walk from "acorn-walk";
import { isInternalLibraryModule } from "./route_bridge.mjs";
import { gltfAssetUris } from "./pack_gltf.mjs";

const PINNED_ROOT = new URL("../../upstream/three.js/", import.meta.url).href;
const CORE_LOADERS = new Set([
  "TextureLoader", "CubeTextureLoader", "ImageLoader", "ImageBitmapLoader",
  "FileLoader", "AudioLoader", "BufferGeometryLoader", "ObjectLoader",
]);
const SETTINGS = new Set(["setPath", "setResourcePath", "setCrossOrigin", "setWithCredentials", "setRequestHeader"]);
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
export function findThreeLoaderRequests(ast, { resolveModule, packageRootUrl, onDefaultManagerExposure } = {}) {
  const root = packageRootUrl ? canonicalUrl(packageRootUrl).replace(/\/?$/, "/") : PINNED_ROOT;
  const importedBindings = new Map();
  for (const node of ast.body) {
    if (node.type !== "ImportDeclaration") continue;
    let resolved;
    try { resolved = canonicalUrl(resolveModule(node.source.value)); } catch { continue; }
    const core = isInternalLibraryModule(resolved, root);
    for (const specifier of node.specifiers) {
      let value;
      if (specifier.type === "ImportNamespaceSpecifier" && core) value = { namespace: true };
      if (specifier.type === "ImportSpecifier") {
        const imported = specifier.imported.name ?? specifier.imported.value;
        if (core && imported === "DefaultLoadingManager") value = { manager: true };
        if ((core && CORE_LOADERS.has(imported)) ||
            (CORE_LOADERS.has(imported) && resolved === root + `src/loaders/${imported}.js`) ||
            (imported === "GLTFLoader" && resolved === root + "examples/jsm/loaders/GLTFLoader.js")) {
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
      if (binding?.value?.namespace && CORE_LOADERS.has(name)) return name;
    }
    return null;
  };
  const statementList = (node) => containing(ancestorsByNode.get(node) ?? [], (parent) => parent.type === "BlockStatement" || parent.type === "Program");
  const receiverSetter = (use, call) => {
    let receiver = use.callee.object;
    while (receiver?.type === "CallExpression" && SETTINGS.has(memberName(receiver.callee))) {
      if (receiver === call) return true;
      receiver = receiver.callee.object;
    }
    return false;
  };
  const instance = (node, use, active = new Set()) => {
    if (node?.type === "NewExpression") {
      const loader = constructor(node.callee);
      // An application-supplied manager can map a logical URL to an unrelated
      // runtime resource. Keep that authored route without assuming disk files.
      return loader && node.arguments.length === 0 ? { loader, paths: [""], resourcePaths: [""] } : null;
    }
    if (node?.type === "CallExpression" && SETTINGS.has(memberName(node.callee))) {
      const value = instance(node.callee.object, use, active);
      if (!value) return null;
      const setting = memberName(node.callee);
      if (setting === "setPath" || setting === "setResourcePath") {
        const candidates = finite(node.arguments[0]);
        if (!candidates) return null;
        value[setting === "setPath" ? "paths" : "resourcePaths"] = candidates;
      }
      return value;
    }
    if (node?.type !== "Identifier") return null;
    const binding = lookup(node.name, node);
    if (binding?.kind !== "const" || active.has(binding)) return null;
    const value = instance(binding.init, use, new Set(active).add(binding));
    if (!value) return null;
    const mutations = [];
    for (const ref of refs.get(binding) ?? []) {
      const member = ref.parent, call = ref.ancestors.at(-3), method = memberName(member);
      if (member?.object !== ref.node || call?.type !== "CallExpression" || call.callee !== member ||
          !(method === "load" || method === "loadAsync" || SETTINGS.has(method))) return null;
      if (SETTINGS.has(method)) {
        // Standalone straight-line setters dominate subsequent loads in the
        // same statement list. Conditional or escaping mutations stay dynamic.
        if (receiverSetter(use, call)) continue; // Arguments are not part of the receiver chain.
        let index = ref.ancestors.indexOf(call), outer = call;
        const settings = [{ call, method }];
        while (index >= 2) {
          const member = ref.ancestors[index - 1], next = ref.ancestors[index - 2];
          if (member?.type !== "MemberExpression" || member.object !== outer || next?.type !== "CallExpression" || next.callee !== member) break;
          const name = memberName(member);
          if (!SETTINGS.has(name) && name !== "load" && name !== "loadAsync") return null;
          if (SETTINGS.has(name)) settings.push({ call: next, method: name });
          outer = next; index -= 2;
          if (!SETTINGS.has(name)) break; // load() does not return the loader.
        }
        const statement = ref.ancestors[index - 1], list = ref.ancestors[index - 2];
        if (statement?.type !== "ExpressionStatement" || statement.expression !== outer ||
            list !== statementList(call) || list !== statementList(use) || list !== statementList(binding.declaration)) return null;
        if (call.start < use.start) mutations.push(...settings);
      }
    }
    mutations.sort((a, b) => a.call.start - b.call.start || a.call.end - b.call.end);
    for (const { call, method } of mutations) {
      if (method !== "setPath" && method !== "setResourcePath") continue;
      const candidates = finite(call.arguments[0]);
      if (!candidates) return null;
      value[method === "setPath" ? "paths" : "resourcePaths"] = candidates;
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
  const requests = [];
  for (const call of calls) {
    if (!["load", "loadAsync"].includes(memberName(call.callee))) continue;
    const value = instance(call.callee.object, call);
    if (!value) continue;
    const urls = value.loader === "CubeTextureLoader" ? cubeUrls(call.arguments[0]) : finite(call.arguments[0]);
    if (!urls || urls.length * value.paths.length * value.resourcePaths.length > LIMIT) continue;
    for (const url of urls) for (const prefix of value.paths) for (const resourcePath of value.resourcePaths) {
      requests.push({ loader: value.loader, url: prefix + url, resourcePath, path: prefix, originalUrl: url });
    }
  }
  return requests;
}

/** Copies the finite local closure without changing the loader's original URL strings. */
export function collectThreeLoaderAssets(requests, { documentUrl, documentBaseUrl, fail }) {
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
    if (!result.has(fileName)) result.set(fileName, { source: fs.readFileSync(filename), originalFileName: filename });
    return result.get(fileName);
  };
  for (const request of requests) {
    const asset = add(request.url);
    if (!asset || request.loader !== "GLTFLoader") continue;
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
