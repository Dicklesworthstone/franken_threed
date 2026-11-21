/** A bounded allocation/provenance proof for independent mixed renderers.
 *
 * This supplements, never weakens, the existing whole-graph escape proof. One
 * closed application module may contain exact and native renderers when every
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
    const modules = Object.values(graph.modules), libraries = new Set(), applications = [];
    for (const module of modules) {
      if (isInternalLibraryModule(module.id, packageRootUrl)) {
        if (!isPinnedRendererBuild(module) || module.dynamic_imports.length) return null;
        libraries.add(module.id);
      } else applications.push(module);
    }
    if (applications.length !== 1 || applications[0].id !== graph.root_entries[0]) return null;
    for (const module of modules.filter(module => libraries.has(module.id)))
      if (module.static_imports.some(request => !libraries.has(request.resolved_id) ||
          new URL(request.specifier, module.id).href !== request.resolved_id)) return null;
    const module = applications[0], facts = module.routing_facts;
    if (module.analysis_complete === false || module.module_type !== 'javascript' || module.dynamic_imports.length ||
        facts.has_opaque_gl_escapes || facts.has_native_context_access || facts.has_unresolved_context_access ||
        facts.escapes.length || module.static_imports.some(request => !libraries.has(request.resolved_id))) return null;
    const code = moduleSources.read(module.id).code;
    if (module.content_hash !== 'sha256:' + createHash('sha256').update(code).digest('hex') || code.length > 128 * 1024) return null;
    const localSites = analyzeModuleAst(code, module.id).renderer_construction_sites;
    if (localSites.length !== decisions.length || decisions.some(site => site.moduleId !== module.id)) return null;
    const sites = new Map(localSites.map((site, index) => {
      const original = module.renderer_construction_sites[index];
      const decision = decisions.find(value => value.sourceSpan.start.offset === original.source_span.start.offset);
      if (!decision || decision.constructorName !== site.constructor_name) reject();
      return [site.source_span.start.offset, {id: `renderer:${index}`, decision}];
    }));
    const ast = acorn.parse(code, {ecmaVersion: 'latest', sourceType: 'module'});
    const nodes = [], bindings = new Map(), enclosing = new WeakMap();
    const groups = new ConnectedCompatibilityGroups(), rendererIds = new Set();
    const define = (name, source, scope) => { if (bindings.has(name)) reject(); bindings.set(name, {source, scope}); };
    walk.fullAncestor(ast, (node, _state, ancestors) => {
      nodes.push(node);
      enclosing.set(node, [...ancestors]);
      const scope = ancestors.slice(0, -1).findLast(parent => ['Program', 'BlockStatement'].includes(parent.type));
      if (nodes.length > 12000) reject();
      if (node.type === 'ImportDeclaration') {
        const request = module.static_imports.find(value => value.specifier === node.source.value);
        if (!request || !libraries.has(request.resolved_id)) reject();
        for (const item of node.specifiers) {
          if (item.type === 'ImportNamespaceSpecifier') define(item.local.name, {kind: 'namespace'}, ast);
          else if (item.type === 'ImportSpecifier' && CONSTRUCTORS.has(item.imported.name ?? item.imported.value))
            define(item.local.name, {kind: 'constructor', name: item.imported.name ?? item.imported.value}, ast);
          else reject();
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
      }
      if (['ClassDeclaration', 'ClassExpression', 'ImportExpression', 'SwitchStatement', 'ForStatement', 'ForOfStatement', 'ForInStatement',
        'WhileStatement', 'DoWhileStatement', 'WithStatement', 'ThisExpression', 'Super', 'YieldExpression', 'TaggedTemplateExpression'].includes(node.type) ||
          node.type === 'Identifier' && node.name === 'arguments' || node.type === 'CatchClause' && node.param) reject();
    });
    const memo = new WeakMap(), pending = new WeakSet(), exports = [], returns = [];
    const connect = (left, right) => {
      for (const a of left.ids ?? []) for (const b of right.ids ?? []) groups.union(a, b);
    };
    const object = node => {
      const site = sites.get(node.start), id = site?.id ?? `object:${node.start}`;
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
          if (bindings.has(node.name)) {
            const {source, scope} = bindings.get(node.name);
            if (!(enclosing.get(node) ?? [ast]).includes(scope)) reject();
            result = source.kind ? source : read(source);
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
          if (['WebGLRenderer', 'WebGPURenderer'].includes(ctor.name) && !sites.has(node.start)) reject();
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
      else if (node.type === 'ExportDefaultDeclaration') exports.push(node.declaration);
      else if (node.type === 'ExportNamedDeclaration') {
        if (node.source) reject();
        for (const item of node.specifiers) exports.push(item.local);
        for (const item of node.declaration?.declarations ?? []) exports.push(item.init);
      } else if (node.type === 'ExportAllDeclaration') reject();
    }
    // Returning a data object can reveal it through a callback or source helper.
    // Scalar-only diagnostics and closed zero-argument frame functions may be
    // exported; no renderer-connected object may cross the module boundary.
    for (const node of returns) if (value(node).kind !== 'scalar') reject();
    for (const node of exports) {
      const output = value(node);
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
