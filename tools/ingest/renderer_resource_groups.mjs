/** A bounded allocation/provenance proof for independent mixed renderers.
 *
 * This supplements, never weakens, the existing whole-graph escape proof. A
 * closed static ESM graph may contain exact and native renderers when every
 * object they consume has a known local origin. All shared objects conservatively
 * couple their users, including CPU assets; no immutability is guessed. Source
 * statements and constructor arguments are never evaluated or rewritten here.
 */
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {ConnectedCompatibilityGroups} from '../compat/connected_groups.mjs';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {isInternalLibraryModule} from './route_bridge.mjs';
import {isPinnedRendererBuild} from './renderer_module_ownership.mjs';
import {parseTagAttributes, stripHtmlComments, stripScriptAndStyleBodies} from './html_parser.mjs';

const CONSTRUCTORS = new Set([
  'WebGLRenderer', 'WebGPURenderer', 'Scene', 'Group', 'Mesh', 'Line', 'LineSegments', 'Points',
  'PerspectiveCamera', 'OrthographicCamera', 'Color', 'Vector2', 'Vector3', 'Vector4', 'Euler', 'Quaternion',
  'Matrix3', 'Matrix4', 'Box3', 'Sphere', 'BufferGeometry', 'BufferAttribute', 'Float32BufferAttribute',
  'Uint16BufferAttribute', 'Uint32BufferAttribute', 'InterleavedBuffer', 'InterleavedBufferAttribute',
  'BoxGeometry', 'PlaneGeometry', 'SphereGeometry', 'ConeGeometry', 'CylinderGeometry', 'TorusGeometry',
  'MeshBasicMaterial', 'MeshLambertMaterial', 'MeshPhongMaterial', 'MeshStandardMaterial', 'MeshToonMaterial',
  'MeshBasicNodeMaterial', 'MeshLambertNodeMaterial', 'MeshPhongNodeMaterial', 'MeshStandardNodeMaterial',
  'MeshToonNodeMaterial', 'LineBasicMaterial', 'PointsMaterial', 'SpriteMaterial',
  'Texture', 'DataTexture', 'DepthTexture', 'RenderTarget', 'WebGLRenderTarget', 'WebGLCubeRenderTarget',
  'AmbientLight', 'DirectionalLight', 'PointLight', 'HemisphereLight', 'SpotLight',
]);
const ARRAYS = new Set(['Float32Array', 'Float64Array', 'Uint8Array', 'Uint8ClampedArray', 'Int8Array',
  'Uint16Array', 'Int16Array', 'Uint32Array', 'Int32Array', 'ArrayBuffer']);
const METHODS = new Set([
  'init', 'render', 'renderAsync', 'compile', 'compileAsync', 'dispose', 'setAnimationLoop',
  'setSize', 'setPixelRatio', 'setDrawingBufferSize', 'getSize', 'getDrawingBufferSize',
  'setClearColor', 'setClearAlpha', 'clear', 'clearColor', 'clearDepth', 'clearStencil',
  'setRenderTarget', 'getRenderTarget', 'setOutputRenderTarget', 'getOutputRenderTarget',
  'setViewport', 'setScissor', 'setScissorTest', 'initTexture', 'initTextureAsync', 'initRenderTarget',
  'waitForGPU', 'readRenderTargetPixelsAsync', 'writeRenderTargetPixelsAsync',
  'copyTextureToTexture', 'copyFramebufferToTexture', 'copyRenderTargetPixelsAsync',
  'add', 'remove', 'clear', 'set', 'setScalar', 'setRGB', 'setHex', 'copy', 'lookAt',
  'updateProjectionMatrix', 'updateMatrix', 'updateMatrixWorld', 'updateWorldMatrix',
  'rotateX', 'rotateY', 'rotateZ', 'translateX', 'translateY', 'translateZ',
  'setIndex', 'setDrawRange', 'addGroup', 'clearGroups', 'computeVertexNormals',
  'computeBoundingBox', 'computeBoundingSphere', 'scale', 'rotateX', 'rotateY', 'rotateZ', 'translate',
]);
// These public data fields are either scalar or owned by the pinned allocation.
// Unknown properties are not assumed to be fresh: inherited functions, material
// defines dictionaries and DOM ownerDocument/parentNode can be shared globally.
// In particular Array.prototype.map, BufferGeometry.prototype.scale and vector
// length methods prevent treating those names as data on an untyped receiver.
const OWN_DATA = new Set([
  'target', 'texture', 'textures', 'geometry', 'material', 'children', 'position', 'rotation',
  'quaternion', 'up', 'matrix', 'matrixWorld', 'projectionMatrix', 'projectionMatrixInverse',
  'matrixWorldInverse', 'color', 'emissive', 'background', 'environment', 'domElement',
  'width', 'height', 'depth', 'x', 'y', 'z', 'w', 'r', 'g', 'b',
  'isF3DRenderer', 'aspect', 'fov', 'near', 'far', 'visible', 'opacity', 'transparent',
]);
const MATERIALS = new Set([...CONSTRUCTORS].filter(name => name.endsWith('Material')));
const FORBIDDEN_PROPERTIES = new Set(['__proto__', 'prototype', 'constructor', 'getContext', 'backend',
  'context', 'extensions', 'xr', 'innerHTML', 'outerHTML', 'srcdoc', 'setAttribute', 'setAttributeNS',
  'onBeforeRender', 'onAfterRender', 'onBeforeCompile', 'customProgramCacheKey']);
const scalar = Object.freeze({kind: 'scalar', ids: []});
const reject = () => { throw new Error('Unproved renderer resource provenance'); };
const property = node => {
  const key = node.property ?? node.key;
  const name = !node.computed && key?.type === 'Identifier' ? key.name :
    key?.type === 'Literal' && ['string', 'number'].includes(typeof key.value) ? String(key.value) : null;
  if (name === null || FORBIDDEN_PROPERTIES.has(name) || name.startsWith('_') || name.toLowerCase().startsWith('on')) reject();
  return name;
};

// A module graph does not include classic scripts, event attributes or child
// browsing contexts. Keep those documents outside this additional proof.
function closedDocument(graph) {
  if (graph.entry_type !== 'html') return true;
  const html = stripScriptAndStyleBodies(stripHtmlComments(fs.readFileSync(graph.entry_path, 'utf8')));
  const tags = new Set(['html', 'head', 'body', 'script', 'style', 'meta', 'title', 'div', 'section',
    'main', 'canvas', 'p', 'h1', 'h2', 'span']);
  const attributes = new Set(['id', 'class', 'width', 'height', 'style', 'type', 'src', 'charset', 'name', 'content', 'lang']);
  for (const match of html.matchAll(/<([a-z][\w:-]*)\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi)) {
    const tag = match[1].toLowerCase(), attrs = parseTagAttributes(match[2]);
    if (!tags.has(tag) || Object.keys(attrs).some(name => !attributes.has(name) && !name.startsWith('data-'))) return false;
    if (tag === 'script' && !['module', 'importmap'].includes(attrs.type?.toLowerCase())) return false;
    if (attrs.src && (tag !== 'script' || attrs.type?.toLowerCase() !== 'module')) return false;
  }
  return true;
}

/** Return per-construction group routes, or null without changing the graph.
 * Unknown imports, mutable bindings, native observations, callbacks carrying
 * unknown parameters and resource escapes keep the existing exact route.
 */
export function resolveClosedRendererGroups(graph, moduleSources, decisions, packageRootUrl) {
  try {
    if (graph.root_entries?.length !== 1 || graph.external_modules?.length || decisions.length < 2 ||
        decisions.some(site => !['WebGLRenderer', 'WebGPURenderer'].includes(site.constructorName)) || !closedDocument(graph)) return null;
    const modules = Object.values(graph.modules), libraries = new Set(), applications = new Map();
    for (const module of modules) {
      if (isInternalLibraryModule(module.id, packageRootUrl)) {
        if (!isPinnedRendererBuild(module) || module.dynamic_imports.length) return null;
        libraries.add(module.id);
      } else applications.set(module.id, module);
    }
    if (!applications.has(graph.root_entries[0]) || applications.size > 128) return null;
    for (const module of modules.filter(module => libraries.has(module.id)))
      if (module.static_imports.some(request => !libraries.has(request.resolved_id) ||
          new URL(request.specifier, module.id).href !== request.resolved_id)) return null;
    // Module URL, not source bytes or a local identifier, owns every binding and
    // allocation. Equal source at separate URLs must not share renderer groups.
    const contexts = new Map(), sites = new Map(), nodes = [], enclosing = new WeakMap(), owners = new WeakMap();
    const groups = new ConnectedCompatibilityGroups(), rendererIds = new Set();
    let sourceBytes = 0;
    for (const module of applications.values()) {
      const facts = module.routing_facts;
      if (module.analysis_complete === false || module.module_type !== 'javascript' || module.dynamic_imports.length ||
          module.postprocessing_transform || module.gpu_computation_transform ||
          facts.has_opaque_gl_escapes || facts.has_native_context_access || facts.has_unresolved_context_access ||
          facts.escapes.length || module.static_imports.some(request =>
            request.module_type !== 'javascript' || !libraries.has(request.resolved_id) && !applications.has(request.resolved_id))) return null;
      const code = moduleSources.read(module.id).code;
      sourceBytes += Buffer.byteLength(code);
      if (module.content_hash !== 'sha256:' + createHash('sha256').update(code).digest('hex') || sourceBytes > 128 * 1024) return null;
      const localSites = analyzeModuleAst(code, module.id).renderer_construction_sites;
      const localDecisions = decisions.filter(site => site.moduleId === module.id);
      if (localSites.length !== localDecisions.length || localSites.length !== module.renderer_construction_sites.length) return null;
      for (const [index, site] of localSites.entries()) {
        const original = module.renderer_construction_sites[index];
        const matches = localDecisions.filter(value => value.sourceSpan.start.offset === original.source_span.start.offset &&
          value.sourceSpan.end.offset === original.source_span.end.offset);
        if (matches.length !== 1 || matches[0].constructorName !== site.constructor_name ||
            original.constructor_name !== site.constructor_name) return null;
        sites.set(`${module.id}\0${site.source_span.start.offset}`, {id: `renderer:${sites.size}`, decision: matches[0]});
      }
      contexts.set(module.id, {module, ast: acorn.parse(code, {ecmaVersion: 'latest', sourceType: 'module'}),
        bindings: new Map(), exports: new Map()});
    }
    if (sites.size !== decisions.length) return null;
    // Cycles and disconnected code need a distinct initialization/entry proof.
    // This walk admits only the complete acyclic static application region.
    const reached = new Set(), visiting = new Set();
    const visit = id => {
      if (libraries.has(id) || reached.has(id)) return;
      if (visiting.has(id) || visiting.size >= 64) reject();
      visiting.add(id);
      for (const request of applications.get(id).static_imports) visit(request.resolved_id);
      visiting.delete(id); reached.add(id);
    };
    visit(graph.root_entries[0]);
    if (reached.size !== applications.size) return null;
    for (const context of contexts.values()) {
      const {module, ast, bindings} = context;
      const define = (name, source, scope) => { if (bindings.has(name)) reject(); bindings.set(name, {source, scope}); };
      const imported = (specifier, name) => {
        const requests = module.static_imports.filter(request => request.specifier === specifier);
        if (!requests.length || requests.some(request => request.resolved_id !== requests[0].resolved_id)) reject();
        const id = requests[0].resolved_id;
        if (libraries.has(id)) {
          if (name === '*') return {kind: 'namespace'};
          if (!CONSTRUCTORS.has(name)) reject();
          return {kind: 'constructor', name};
        }
        return name === '*' ? {kind: 'module-namespace', id} : {imported: {id, name}};
      };
      const expose = (name, source) => {
        if (context.exports.has(name)) reject();
        // acorn-walk does not visit the local identifiers in export specifiers.
        if (source.type === 'Identifier' && !owners.has(source)) {
          owners.set(source, context); enclosing.set(source, [ast]);
        }
        context.exports.set(name, source);
      };
      walk.fullAncestor(ast, (node, _state, ancestors) => {
        nodes.push(node);
        enclosing.set(node, [...ancestors]);
        owners.set(node, context);
        const scope = ancestors.slice(0, -1).findLast(parent => ['Program', 'BlockStatement'].includes(parent.type));
        if (nodes.length > 12000) reject();
        if (node.type === 'ImportDeclaration') {
          for (const item of node.specifiers) {
            const name = item.type === 'ImportNamespaceSpecifier' ? '*' :
              item.type === 'ImportDefaultSpecifier' ? 'default' : item.imported.name ?? item.imported.value;
            define(item.local.name, imported(node.source.value, name), ast);
          }
        } else if (node.type === 'VariableDeclaration') {
          if (node.kind !== 'const') reject();
          for (const item of node.declarations) {
            if (item.id.type !== 'Identifier' || !item.init) reject();
            define(item.id.name, item.init, scope);
          }
        } else if (node.type === 'FunctionDeclaration') {
          if (!node.id) reject();
          define(node.id.name, node, scope);
        } else if (node.type === 'ExportNamedDeclaration') {
          for (const item of node.specifiers) expose(item.exported.name ?? item.exported.value,
            node.source ? imported(node.source.value, item.local.name ?? item.local.value) : item.local);
          for (const item of node.declaration?.declarations ?? []) expose(item.id.name, item.init);
          if (node.declaration?.type === 'FunctionDeclaration') expose(node.declaration.id.name, node.declaration);
        } else if (node.type === 'ExportDefaultDeclaration') {
          expose('default', node.declaration);
        }
        if (['ClassDeclaration', 'ClassExpression', 'ImportExpression', 'SwitchStatement', 'ForStatement', 'ForOfStatement', 'ForInStatement',
          'WhileStatement', 'DoWhileStatement', 'WithStatement', 'ThisExpression', 'Super', 'YieldExpression', 'TaggedTemplateExpression',
          'ExportAllDeclaration'].includes(node.type) ||
            node.type === 'Identifier' && node.name === 'arguments' || node.type === 'CatchClause' && node.param) reject();
      });
    }
    const memo = new WeakMap(), pending = new WeakSet(), resolvingExports = new Set(), returns = [];
    const exported = (id, name) => {
      const key = `${id}\0${name}`, source = contexts.get(id)?.exports.get(name);
      if (!source || resolvingExports.has(key) || resolvingExports.size >= 64) reject();
      resolvingExports.add(key);
      const result = bindingValue(source);
      resolvingExports.delete(key);
      return result;
    };
    const bindingValue = source => source.imported ? exported(source.imported.id, source.imported.name) :
      source.kind ? source : value(source);
    const siteKey = node => `${owners.get(node).module.id}\0${node.start}`;
    const connect = (left, right) => {
      for (const a of left.ids ?? []) for (const b of right.ids ?? []) groups.union(a, b);
    };
    const object = node => {
      const key = siteKey(node), site = sites.get(key), id = site?.id ?? `object:${key}`;
      if (site) { groups.registerRenderer(id, site.decision.route); rendererIds.add(id); }
      else groups.registerResource(id, true);
      return {kind: 'object', ids: [id]};
    };
    const data = result => { if (!['object', 'scalar'].includes(result.kind)) reject(); return result; };
    const value = (node, depth = 0) => {
      if (!node) return scalar;
      if (depth > 64 || pending.has(node)) reject();
      if (memo.has(node)) return memo.get(node);
      const read = child => value(child, depth + 1);
      pending.add(node);
      let result;
      switch (node.type) {
        case 'Literal': if (node.regex || typeof node.value === 'bigint') reject(); result = scalar; break;
        case 'Identifier': {
          const {bindings, ast} = owners.get(node);
          if (bindings.has(node.name)) {
            const {source, scope} = bindings.get(node.name);
            if (!(enclosing.get(node) ?? [ast]).includes(scope)) reject();
            result = source.kind || source.imported ? bindingValue(source) : read(source);
          } else if (node.name === 'document') result = {kind: 'document'};
          else if (node.name === 'Math') result = {kind: 'math'};
          else if (ARRAYS.has(node.name)) result = {kind: 'constructor', name: node.name};
          else if (['undefined', 'NaN', 'Infinity'].includes(node.name)) result = scalar;
          else reject();
          break;
        }
        case 'NewExpression': {
          const ctor = read(node.callee);
          if (ctor.kind !== 'constructor' || node.arguments.some(item => item.type === 'SpreadElement')) reject();
          if (['WebGLRenderer', 'WebGPURenderer'].includes(ctor.name) && !sites.has(siteKey(node))) reject();
          result = {...object(node), allocationType: ctor.name};
          for (const argument of node.arguments) connect(result, data(read(argument)));
          break;
        }
        case 'ObjectExpression': case 'ArrayExpression': {
          result = object(node);
          const items = node.type === 'ArrayExpression' ? node.elements : node.properties.map(item => {
            if (item.type !== 'Property' || item.kind !== 'init' || item.method) reject();
            property(item);
            return item.value;
          });
          for (const item of items) connect(result, data(read(item)));
          break;
        }
        case 'MemberExpression': {
          const owner = read(node.object), name = property(node);
          if (owner.kind === 'namespace') {
            if (!CONSTRUCTORS.has(name)) reject();
            result = {kind: 'constructor', name};
          } else if (owner.kind === 'module-namespace') {
            result = exported(owner.id, name);
          } else if (owner.kind === 'document') {
            if (!['body', 'documentElement', 'createElement', 'createElementNS', 'appendChild', 'append', 'prepend'].includes(name)) reject();
            result = {kind: 'document', name};
          } else if (owner.kind === 'math') result = {kind: 'math', name};
          else if (owner.kind === 'object') {
            const parent = enclosing.get(node)?.at(-2);
            if (parent?.type === 'CallExpression' && parent.callee === node && METHODS.has(name)) {
              result = {kind: 'method', ids: owner.ids, member: name};
            } else {
              if (!OWN_DATA.has(name) && !/^\d+$/.test(name) &&
                  !(name === 'map' && MATERIALS.has(owner.allocationType))) reject();
              result = {kind: 'object', ids: owner.ids};
            }
          }
          else reject();
          break;
        }
        case 'CallExpression': {
          if (node.optional || node.arguments.some(item => item.type === 'SpreadElement')) reject();
          const callee = read(node.callee);
          if (callee.kind === 'document') {
            if (['createElement', 'createElementNS'].includes(callee.name)) {
              const name = node.arguments.at(-1);
              if (name?.type !== 'Literal' || name.value !== 'canvas' ||
                  node.arguments.length !== (callee.name === 'createElement' ? 1 : 2)) reject();
              if (callee.name === 'createElementNS' && (node.arguments[0]?.type !== 'Literal' ||
                  node.arguments[0].value !== 'http://www.w3.org/1999/xhtml')) reject();
              node.arguments.forEach(item => data(read(item)));
              result = object(node);
            } else if (['appendChild', 'append', 'prepend'].includes(callee.name)) {
              node.arguments.forEach(item => data(read(item)));
              result = node.arguments.length === 1 ? read(node.arguments[0]) : scalar;
            } else reject();
          } else if (callee.kind === 'math') {
            if (!callee.name || node.arguments.some(item => read(item).kind !== 'scalar')) reject();
            result = scalar;
          } else if (callee.kind === 'callback') {
            if (node.arguments.length) reject();
            result = scalar;
          } else if (callee.kind === 'method') {
            for (const argument of node.arguments) {
              const input = read(argument);
              if (input.kind === 'callback' && callee.member === 'setAnimationLoop') continue;
              connect(callee, data(input));
            }
            result = {kind: 'object', ids: callee.ids};
          } else reject();
          break;
        }
        case 'ArrowFunctionExpression': case 'FunctionExpression': case 'FunctionDeclaration':
          if (node.params.length || node.async || node.generator || node.type === 'FunctionExpression' && node.id) reject();
          if (node.body.type !== 'BlockStatement') returns.push(node.body);
          result = {kind: 'callback'}; break;
        case 'BinaryExpression':
          if (node.operator === 'instanceof') {
            data(read(node.left)); if (read(node.right).kind !== 'constructor') reject();
          } else { data(read(node.left)); data(read(node.right)); }
          result = scalar; break;
        case 'UnaryExpression':
          if (node.operator === 'delete') reject();
          data(read(node.argument)); result = scalar; break;
        case 'LogicalExpression': case 'ConditionalExpression': {
          if (node.test) data(read(node.test));
          const left = data(read(node.consequent ?? node.left)), right = data(read(node.alternate ?? node.right));
          result = left.kind === 'scalar' && right.kind === 'scalar' ? scalar :
            {kind: 'object', ids: [...new Set([...left.ids, ...right.ids])]}; break;
        }
        case 'AssignmentExpression': {
          if (node.left.type !== 'MemberExpression') reject();
          const owner = read(node.left.object);
          if (owner.kind !== 'object') reject();
          read(node.left); // Also prove the destination is an admitted data field.
          result = data(read(node.right)); connect(owner, result); break;
        }
        case 'UpdateExpression':
          if (node.argument.type !== 'MemberExpression' || read(node.argument.object).kind !== 'object') reject();
          read(node.argument);
          result = scalar; break;
        case 'AwaitExpression': result = read(node.argument); break;
        case 'SequenceExpression': node.expressions.forEach(read); result = read(node.expressions.at(-1)); break;
        case 'TemplateLiteral': node.expressions.forEach(item => data(read(item))); result = scalar; break;
        default: reject();
      }
      pending.delete(node); memo.set(node, result); return result;
    };
    for (const node of nodes) {
      if (['NewExpression', 'CallExpression', 'AssignmentExpression', 'UpdateExpression', 'MemberExpression',
        'ArrowFunctionExpression', 'FunctionExpression', 'FunctionDeclaration'].includes(node.type)) value(node);
      else if (node.type === 'VariableDeclarator') value(node.init);
      else if (node.type === 'ExpressionStatement') value(node.expression);
      else if (node.type === 'IfStatement') data(value(node.test));
      else if (node.type === 'ReturnStatement' || node.type === 'ThrowStatement') returns.push(node.argument);
    }
    // Returning a data object can reveal it through a callback or source helper.
    // Internal ESM exports stay inside the proved graph; its public root still
    // cannot expose a renderer-connected object to an untracked consumer.
    for (const node of returns) if (value(node).kind !== 'scalar') reject();
    for (const name of contexts.get(graph.root_entries[0]).exports.keys()) {
      const output = exported(graph.root_entries[0], name);
      if (output.kind === 'callback' || output.kind === 'scalar') continue;
      if (output.kind !== 'object' || output.ids.some(id => groups.getGroupMembers(id).some(member => rendererIds.has(member)))) reject();
    }
    if (rendererIds.size !== decisions.length) return null;
    const routes = groups.resolveGroupRoutes();
    return decisions.map(decision => {
      const site = [...sites.values()].find(site => site.decision === decision), route = routes.get(site.id);
      return {...decision, route: route.route, reasons: [...new Set([...decision.reasons, ...route.reasons])], groupId: route.groupId};
    });
  } catch { return null; }
}
