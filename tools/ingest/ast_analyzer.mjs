/**
 * AST Analyzer for ES Modules.
 * Uses pinned Acorn (8.14.0) to extract imports, exports, dynamic imports,
 * live bindings, classes, prototype writes, asset patterns, and renderer
 * routing facts with precise source spans.
 */

import * as acorn from "acorn";
import * as walk from "acorn-walk";
import { IngestionParseError } from "./types.mjs";
import { verifiedInspectorExtensionImports } from "./inspector_extension_imports.mjs";

/**
 * @typedef {Object} SourceLocationPoint
 * @property {number} line - 1-based line number
 * @property {number} column - 0-based column number
 * @property {number} offset - 0-based character offset
 */

/**
 * @typedef {Object} SourceSpan
 * @property {SourceLocationPoint} start
 * @property {SourceLocationPoint} end
 */

/**
 * Adjusts an Acorn AST node's source location for inline scripts.
 * @param {any} node
 * @param {Object} [offsets]
 * @param {number} [offsets.lineOffset=0]
 * @param {number} [offsets.columnOffset=0]
 * @param {number} [offsets.charOffset=0]
 * @returns {SourceSpan}
 */
function toSourceSpan(node, offsets = {}) {
  const lineOffset = offsets.lineOffset || 0;
  const colOffset = offsets.columnOffset || 0;
  const charOffset = offsets.charOffset || 0;

  const startLine = node.loc ? node.loc.start.line + lineOffset : 1;
  const startCol = node.loc
    ? node.loc.start.line === 1
      ? node.loc.start.column + colOffset
      : node.loc.start.column
    : 0;
  const startChar = (node.start !== undefined ? node.start : 0) + charOffset;

  const endLine = node.loc ? node.loc.end.line + lineOffset : 1;
  const endCol = node.loc
    ? node.loc.end.line === 1
      ? node.loc.end.column + colOffset
      : node.loc.end.column
    : 0;
  const endChar = (node.end !== undefined ? node.end : 0) + charOffset;

  return {
    start: { line: startLine, column: startCol, offset: startChar },
    end: { line: endLine, column: endCol, offset: endChar },
  };
}

/**
 * Recursively extracts all string candidate branches from a finite string expression.
 * Supports:
 * - String Literal (typeof node.value === 'string')
 * - TemplateLiteral (with 0 or more finite string expressions)
 * - BinaryExpression with operator '+' (both sides finite string expressions)
 * - ConditionalExpression (both branches finite string expressions)
 *
 * Strictly rejects:
 * - Unresolved Identifiers, CallExpressions, MemberExpressions, TaggedTemplateExpressions
 * - Non-string Literals (numbers, booleans, null, regex)
 * - Binary operators other than '+'
 * - Expressions whose Cartesian expansion exceeds limit
 *
 * @param {any} node - Acorn AST node
 * @param {number} [limit=128] - Maximum number of candidates before aborting to nonliteral
 * @returns {string[] | null}
 */
function extractFiniteStringCandidates(node, limit = 128, resolveIdentifier = null, depth = 0) {
  if (!node || depth > 64) return null;
  const expand = child => extractFiniteStringCandidates(child, limit, resolveIdentifier, depth + 1);

  switch (node.type) {
    case "Literal": {
      if (typeof node.value === "string") {
        return [node.value];
      }
      return null;
    }

    case "Identifier":
      return resolveIdentifier?.(node) ?? null;

    case "TemplateLiteral": {
      if (node.expressions.length === 0) {
        if (node.quasis.length > 0) {
          return [node.quasis.map((q) => q.value.cooked ?? q.value.raw).join("")];
        }
        return [""];
      }

      let current = [node.quasis[0].value.cooked ?? node.quasis[0].value.raw];
      for (let i = 0; i < node.expressions.length; i++) {
        const exprCands = expand(node.expressions[i]);
        if (!exprCands || exprCands.length === 0) return null;

        const nextQuasi = node.quasis[i + 1].value.cooked ?? node.quasis[i + 1].value.raw;
        if (current.length * exprCands.length > limit) return null;

        const next = [];
        for (const prefix of current) {
          for (const cand of exprCands) {
            if (prefix.length + cand.length + nextQuasi.length > 65536) return null;
            next.push(prefix + cand + nextQuasi);
          }
        }
        current = Array.from(new Set(next));
        if (current.length > limit) return null;
      }
      return current;
    }

    case "BinaryExpression": {
      if (node.operator !== "+") return null;
      const left = expand(node.left);
      if (!left || left.length === 0) return null;
      const right = expand(node.right);
      if (!right || right.length === 0) return null;

      if (left.length * right.length > limit) return null;

      const combined = [];
      for (const l of left) {
        for (const r of right) {
          if (l.length + r.length > 65536) return null;
          combined.push(l + r);
        }
      }
      const unique = Array.from(new Set(combined));
      if (unique.length > limit) return null;
      return unique;
    }

    case "ConditionalExpression": {
      // Preserve effects in conditional tests by only collecting alternatives and leaving source unchanged
      const consequent = expand(node.consequent);
      if (!consequent || consequent.length === 0) return null;
      const alternate = expand(node.alternate);
      if (!alternate || alternate.length === 0) return null;

      const unique = Array.from(new Set([...consequent, ...alternate]));
      if (unique.length > limit) return null;
      return unique;
    }

    case "ParenthesizedExpression":
      return expand(node.expression);

    case "SequenceExpression":
      return expand(node.expressions.at(-1));

    default:
      return null;
  }
}

/**
 * Classifies the argument of an ImportExpression (dynamic import).
 * Exported for Ruby bundler reuse on executable finite conditional imports.
 * @param {any} sourceNode
 * @returns {{ classification: 'literal' | 'finite_set' | 'nonliteral', specifier: string | null, specifiers?: string[], finite_set?: string[], finiteSet?: string[], candidates?: string[] }}
 */
export function classifyDynamicImportArgument(sourceNode, resolveIdentifier = null) {
  if (!sourceNode) {
    return { classification: "nonliteral", specifier: null };
  }

  // Fastpaths: actual literal string or 0-expression template
  if (sourceNode.type === "Literal" && typeof sourceNode.value === "string") {
    return { classification: "literal", specifier: sourceNode.value };
  }

  if (
    sourceNode.type === "TemplateLiteral" &&
    sourceNode.expressions.length === 0 &&
    sourceNode.quasis.length > 0
  ) {
    const specifier = sourceNode.quasis.map((q) => q.value.cooked ?? q.value.raw).join("");
    return { classification: "literal", specifier };
  }

  // Complex finite expressions (binary '+', template interpolation, conditionals)
  const candidates = extractFiniteStringCandidates(sourceNode, 128, resolveIdentifier);
  if (candidates && candidates.length > 0) {
    return {
      classification: "finite_set",
      specifier: null,
      specifiers: candidates,
      candidates,
      finite_set: candidates,
      finiteSet: candidates,
    };
  }

  return { classification: "nonliteral", specifier: null };
}

/** Resolve only immutable lexical string bindings, without evaluating their
 * initializers or rewriting reads. All other declarations still shadow outer
 * constants. Runtime selection, TDZ failures and initializer effects stay in
 * the source; this proves a set of possible import URLs, not a constant value.
 */
export function createDynamicImportClassifier(ast) {
  // Sloppy scripts can introduce bindings with direct eval/with. Their existing
  // literal-only path remains conservative; ES modules are always strict.
  if (ast.sourceType === "script") return node => classifyDynamicImportArgument(node);
  const scopes = new WeakMap(), enclosing = new WeakMap(), values = new WeakMap();
  const functions = new Set(["FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"]);
  const lexical = new Set(["Program", "BlockStatement", "StaticBlock", "CatchClause",
    "ForStatement", "ForInStatement", "ForOfStatement", "SwitchStatement", ...functions]);
  const classes = new Set(["ClassDeclaration", "ClassExpression"]);
  const put = (scope, names, initializer = null) => {
    if (!scopes.has(scope)) scopes.set(scope, new Map());
    const bindings = scopes.get(scope);
    for (const name of names) bindings.set(name, bindings.has(name) ? null : initializer);
  };
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    const parents = ancestors.slice(0, -1);
    const containing = predicate => parents.findLast(predicate);
    if (node.type === "Identifier") {
      enclosing.set(node, parents.filter((parent, index) =>
        (lexical.has(parent.type) || classes.has(parent.type)) &&
        // A switch's case bindings are not in scope in its discriminant.
        !(parent.type === "SwitchStatement" && ancestors[index + 1] === parent.discriminant)));
    } else if (node.type === "VariableDeclaration") {
      let scope;
      if (node.kind === "var") {
        scope = containing(parent => functions.has(parent.type) ||
          parent.type === "Program" || parent.type === "StaticBlock");
        // Body vars do not shadow a function's default parameter expressions.
        if (functions.has(scope.type)) scope = scope.body;
      } else scope = containing(parent => lexical.has(parent.type));
      for (const declaration of node.declarations) {
        put(scope, collectPatternIdentifiers(declaration.id),
          node.kind === "const" && declaration.id.type === "Identifier" ? declaration.init : null);
      }
    } else if (node.type === "ImportDeclaration") {
      put(ast, node.specifiers.map(specifier => specifier.local.name));
    } else if (functions.has(node.type)) {
      for (const parameter of node.params) put(node, collectPatternIdentifiers(parameter));
      if (node.id) {
        put(node, [node.id.name]);
        if (node.type === "FunctionDeclaration") put(containing(parent => lexical.has(parent.type)), [node.id.name]);
      }
    } else if (node.type === "CatchClause") {
      put(node, collectPatternIdentifiers(node.param));
    } else if (classes.has(node.type) && node.id) {
      put(node, [node.id.name]);
      if (node.type === "ClassDeclaration") put(containing(parent => lexical.has(parent.type)), [node.id.name]);
    }
  });
  let depth = 0;
  const resolveIdentifier = node => {
    const chain = enclosing.get(node) ?? [];
    for (let index = chain.length - 1; index >= 0; index--) {
      const bindings = scopes.get(chain[index]);
      if (!bindings?.has(node.name)) continue;
      const initializer = bindings.get(node.name);
      if (!initializer || depth >= 64) return null;
      if (values.has(initializer)) return values.get(initializer);
      // A pending value breaks alias cycles; memoization also bounds repeated
      // aliases whose source expression would otherwise expand exponentially.
      values.set(initializer, null);
      depth++;
      const result = extractFiniteStringCandidates(initializer, 128, resolveIdentifier);
      depth--;
      values.set(initializer, result);
      return result;
    }
    return null;
  };
  return node => classifyDynamicImportArgument(node, resolveIdentifier);
}

/** The native module goal, without evaluating import options. Sequence prefixes
 * remain runtime effects; opaque dictionaries/getters stay unresolved.
 */
export function classifyImportType(node) {
  const tail = (value) => {
    let depth = 0;
    while (value?.type === "SequenceExpression") {
      if (++depth > 64) return null;
      value = value.expressions.at(-1);
    }
    return value;
  };
  const dictionary = (value) => {
    value = tail(value);
    if (value?.type !== "ObjectExpression" || value.properties.some((p) =>
      p.type !== "Property" || p.computed || p.kind !== "init" || p.method ||
      (p.key.name ?? p.key.value) === "__proto__")) return null;
    return value.properties;
  };
  let attributes = node.attributes ?? [];
  if (node.type === "ImportExpression") {
    if (!node.options) return "javascript";
    const options = dictionary(node.options);
    if (!options) return "unknown";
    const maps = options.filter((p) => (p.key.name ?? p.key.value) === "with");
    // Import() reads options.with normally, so an absent own property can
    // still resolve through Object.prototype. Do not assume that is absent.
    if (maps.length !== 1) return "unknown";
    attributes = dictionary(maps[0].value);
    if (!attributes) return "unknown";
  }
  const types = attributes.filter((p) => (p.key.name ?? p.key.value) === "type");
  if (!types.length) return "javascript";
  return types.length === 1 && extractStaticString(tail(types[0].value)) === "json"
    ? "json" : "unknown";
}

/**
 * Recursively collects all identifier names bound by a pattern.
 * Supports Identifiers, AssignmentPatterns, RestElements, ObjectPatterns, and ArrayPatterns.
 * @param {any} pattern
 * @param {Set<string>} [out]
 * @returns {Set<string>}
 */
function collectPatternIdentifiers(pattern, out = new Set()) {
  if (!pattern) return out;
  switch (pattern.type) {
    case "Identifier":
      out.add(pattern.name);
      break;
    case "AssignmentPattern":
      collectPatternIdentifiers(pattern.left, out);
      break;
    case "RestElement":
      collectPatternIdentifiers(pattern.argument, out);
      break;
    case "ObjectPattern":
      for (const prop of pattern.properties) {
        if (prop.type === "Property") {
          collectPatternIdentifiers(prop.value, out);
        } else if (prop.type === "RestElement") {
          collectPatternIdentifiers(prop.argument, out);
        }
      }
      break;
    case "ArrayPattern":
      for (const elem of pattern.elements) {
        if (elem) collectPatternIdentifiers(elem, out);
      }
      break;
  }
  return out;
}

const defaultVarBindingCache = new WeakMap();

/**
 * Checks whether any `var` declaration within a function or module body binds the target name,
 * without descending across nested function or class boundaries.
 * @param {any} bodyNode
 * @param {string} name
 * @param {WeakMap<any, Map<string, boolean>>} [cache]
 * @returns {boolean}
 */
function hasVarBindingInFunctionScope(bodyNode, name, cache = defaultVarBindingCache) {
  if (!bodyNode) return false;
  let map = cache.get(bodyNode);
  if (!map) {
    map = new Map();
    cache.set(bodyNode, map);
  }
  if (map.has(name)) return map.get(name);

  let found = false;
  walk.recursive(bodyNode, null, {
    VariableDeclaration(decl, state, c) {
      if (decl.kind === "var") {
        for (const d of decl.declarations) {
          if (collectPatternIdentifiers(d.id).has(name)) {
            found = true;
          }
        }
      }
      for (const d of decl.declarations) {
        if (d.init) c(d.init, state);
      }
    },
    FunctionDeclaration() {},
    FunctionExpression() {},
    ArrowFunctionExpression() {},
    ClassDeclaration() {},
    ClassExpression() {},
  });

  map.set(name, found);
  return found;
}

/**
 * Conservatively verifies whether an identifier (e.g. 'URL') is shadowed by any
 * local declaration, import, parameter, loop header, catch clause, or hoisted `var`
 * in any scope enclosing the node corresponding to `ancestors`.
 * @param {string} name
 * @param {any[]} ancestors
 * @param {WeakMap<any, Map<string, boolean>>} [varCache]
 * @returns {boolean}
 */
export function isIdentifierShadowedAtAncestors(name, ancestors, varCache = defaultVarBindingCache) {
  for (let i = ancestors.length - 2; i >= 0; i--) {
    const ancestor = ancestors[i];
    const child = ancestors[i + 1];
    if (!ancestor) continue;

    // 1. Function boundaries: parameters, function expression / declaration name, hoisted vars
    if (
      ancestor.type === "FunctionDeclaration" ||
      ancestor.type === "FunctionExpression" ||
      ancestor.type === "ArrowFunctionExpression"
    ) {
      if (ancestor.id && ancestor.id.name === name) return true;
      if (ancestor.params) {
        for (const param of ancestor.params) {
          if (collectPatternIdentifiers(param).has(name)) return true;
        }
      }
      // Body var declarations only scope the function body, not default parameter initializers
      if (child === ancestor.body && hasVarBindingInFunctionScope(ancestor.body, name, varCache))
        return true;
    }

    // 2. Class declaration / expression self-name in class body
    if (ancestor.type === "ClassDeclaration" || ancestor.type === "ClassExpression") {
      if (ancestor.id && ancestor.id.name === name) return true;
    }

    // 3. BlockStatement / StaticBlock / Program body statements
    if (
      ancestor.type === "BlockStatement" ||
      ancestor.type === "StaticBlock" ||
      ancestor.type === "Program"
    ) {
      const body = ancestor.body || [];
      for (const stmt of body) {
        if (stmt.type === "VariableDeclaration") {
          for (const d of stmt.declarations) {
            if (collectPatternIdentifiers(d.id).has(name)) return true;
          }
        } else if (stmt.type === "FunctionDeclaration") {
          if (stmt.id && stmt.id.name === name) return true;
        } else if (stmt.type === "ClassDeclaration") {
          if (stmt.id && stmt.id.name === name) return true;
        } else if (stmt.type === "ImportDeclaration") {
          for (const spec of stmt.specifiers) {
            if (spec.local && spec.local.name === name) return true;
          }
        } else if (stmt.type === "ExportNamedDeclaration" && stmt.declaration) {
          const decl = stmt.declaration;
          if (decl.type === "VariableDeclaration") {
            for (const d of decl.declarations) {
              if (collectPatternIdentifiers(d.id).has(name)) return true;
            }
          } else if (decl.id && decl.id.name === name) {
            return true;
          }
        } else if (stmt.type === "ExportDefaultDeclaration" && stmt.declaration) {
          const decl = stmt.declaration;
          if (decl.id && decl.id.name === name) {
            return true;
          }
        }
      }
      if (ancestor.type === "Program") {
        if (hasVarBindingInFunctionScope(ancestor, name, varCache)) return true;
      }
    }

    // 4. Switch statement: cases share a single block scope
    if (ancestor.type === "SwitchStatement" && ancestor.cases) {
      for (const sc of ancestor.cases) {
        if (!sc.consequent) continue;
        for (const stmt of sc.consequent) {
          if (
            stmt.type === "VariableDeclaration" &&
            (stmt.kind === "let" || stmt.kind === "const")
          ) {
            for (const d of stmt.declarations) {
              if (collectPatternIdentifiers(d.id).has(name)) return true;
            }
          } else if (stmt.type === "FunctionDeclaration" || stmt.type === "ClassDeclaration") {
            if (stmt.id && stmt.id.name === name) return true;
          }
        }
      }
    }

    // 5. For loop variable declarations
    if (ancestor.type === "ForStatement") {
      if (ancestor.init && ancestor.init.type === "VariableDeclaration") {
        for (const d of ancestor.init.declarations) {
          if (collectPatternIdentifiers(d.id).has(name)) return true;
        }
      }
    }
    if (ancestor.type === "ForInStatement" || ancestor.type === "ForOfStatement") {
      if (ancestor.left && ancestor.left.type === "VariableDeclaration") {
        for (const d of ancestor.left.declarations) {
          if (collectPatternIdentifiers(d.id).has(name)) return true;
        }
      }
    }

    // 6. Catch clause parameter
    if (ancestor.type === "CatchClause") {
      if (ancestor.param && collectPatternIdentifiers(ancestor.param).has(name)) return true;
    }
  }
  return false;
}

/**
 * Extracts a static string literal or single-quasi template literal from an AST node.
 * Returns null if the expression is dynamic or not a string.
 * @param {any} node
 * @returns {string | null}
 */
function extractStaticString(node) {
  if (!node) return null;
  if (node.type === "Literal" && typeof node.value === "string") {
    return node.value;
  }
  if (node.type === "TemplateLiteral" && node.expressions.length === 0 && node.quasis.length > 0) {
    return node.quasis.map((q) => q.value.cooked ?? q.value.raw).join("");
  }
  return null;
}

/** Finite property names without evaluating computed keys or their effects. */
function staticPropertyNames(node) {
  if (!node || (node.type !== "MemberExpression" && node.type !== "Property")) return null;
  const key = node.property ?? node.key;
  if (!node.computed && key?.type === "Identifier") return [key.name];
  return extractFiniteStringCandidates(key);
}

/** Read a constructor option from object syntax in source order. Undefined means
 * absent, null means unknown, and an AST node is the captured property value.
 * Spreads copy own properties only; explicit __proto__ can supply inherited ones.
 * Getters, opaque keys/sources and excessive nesting stay runtime decisions.
 */
function sourceOptionValue(node, name, ownOnly = false, depth = 0) {
  if (depth > 32) return null;
  if (!node) return undefined;
  if (node.type === "SequenceExpression")
    return sourceOptionValue(node.expressions.at(-1), name, ownOnly, depth + 1);
  if (node.type === "Literal" && (node.value === null || typeof node.value !== "object"))
    return undefined;
  if (node.type !== "ObjectExpression") return null;
  let own, inherited;
  for (const property of node.properties) {
    if (property.type === "SpreadElement") {
      const value = sourceOptionValue(property.argument, name, true, depth + 1);
      if (value !== undefined) own = value;
      continue;
    }
    const names = staticPropertyNames(property);
    if (!names || names.length !== 1) {
      if (!names || names.includes(name)) own = null;
      continue;
    }
    if (names[0] === name) {
      own = property.kind === "init" && !property.method ? property.value : null;
    } else if (names[0] === "__proto__" && !property.computed &&
        property.kind === "init" && !property.method) {
      inherited = sourceOptionValue(property.value, name, false, depth + 1);
    }
  }
  return own !== undefined ? own : ownOnly ? undefined : inherited;
}

const OPAQUE_GL_METHODS = new Set([
  "getExtension", "getParameter", "createBuffer", "bindBuffer", "createTexture", "bindTexture",
]);

/**
 * A renderer's route must be chosen before an opaque consumer can observe it.
 * Track lexical aliases monotonically: reassignment cannot prove that an earlier
 * consumer lost its reference. Containers, returns and unknown calls are escape
 * boundaries, not invitations to guess what another component will do.
 *
 * Imported consumers remain opaque unless the graph verified their source and
 * dependency contract. Even then their returned resource carriers remain in
 * this alias proof. No source code is evaluated.
 */
function analyzeRendererResourceEscapes(ast, offsets, consumerContract = null) {
  const RENDERER = 1, CANVAS = 2, WEBGL = 4, WEBGPU = 8, NAMESPACE = 16, DOM = 32;
  const ORBIT_CTOR = 64, CONTROLS = 128, BASE_CTOR = 256, ORBIT_NAMESPACE = 512, BASE_NAMESPACE = 1024;
  const CONSTRUCTOR = WEBGL | WEBGPU, ALL_CONSTRUCTORS = CONSTRUCTOR | ORBIT_CTOR | BASE_CTOR;
  const RESOURCE = RENDERER | CANVAS | ALL_CONSTRUCTORS | NAMESPACE | CONTROLS;
  const consumerKind = role => role === "orbit" ? ORBIT_CTOR : role === "base" ? BASE_CTOR :
    role === "webgpu" ? WEBGPU : role === "webgl" ? WEBGL : 0;
  const functions = new Set(["FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"]);
  const lexical = new Set(["Program", "BlockStatement", "StaticBlock", "CatchClause",
    "ForStatement", "ForInStatement", "ForOfStatement", "SwitchStatement", ...functions]);
  const classes = new Set(["ClassDeclaration", "ClassExpression"]);
  const scopes = new WeakMap(), enclosing = new WeakMap(), parents = new WeakMap();
  const nodes = [], globals = new Map(), escapes = [], reported = new WeakSet();
  const ctorKind = name => name === "WebGLRenderer" ? WEBGL : name === "WebGPURenderer" ? WEBGPU : 0;
  const declare = (scope, names) => {
    if (!scopes.has(scope)) scopes.set(scope, new Map());
    const bindings = scopes.get(scope);
    for (const name of names) if (!bindings.has(name)) bindings.set(name, {kind: 0, local: true});
  };
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    nodes.push(node);
    const chain = ancestors.slice(0, -1);
    parents.set(node, chain.at(-1));
    const containing = predicate => chain.findLast(predicate);
    if (node.type === "Identifier") {
      enclosing.set(node, chain.filter((parent, index) =>
        (lexical.has(parent.type) || classes.has(parent.type)) &&
        !(parent.type === "SwitchStatement" && ancestors[index + 1] === parent.discriminant)));
    } else if (node.type === "VariableDeclaration") {
      let scope = containing(parent => node.kind === "var"
        ? functions.has(parent.type) || parent.type === "Program" || parent.type === "StaticBlock"
        : lexical.has(parent.type));
      if (node.kind === "var" && functions.has(scope.type)) scope = scope.body;
      for (const declaration of node.declarations) {
        const names = collectPatternIdentifiers(declaration.id);
        declare(scope, names);
        // Parameter/var aliasing is outside the retained-consumer proof. A
        // monotonic resource bit alone does not establish a closed call target.
        if (node.kind !== "var") for (const name of names) scopes.get(scope).get(name).consumerVariable = true;
      }
    } else if (node.type === "ImportDeclaration") {
      declare(ast, node.specifiers.map(spec => spec.local.name));
      for (const spec of node.specifiers) {
        if (spec.type === "ImportSpecifier") {
          const cell = scopes.get(ast).get(spec.local.name);
          cell.consumerSource = consumerKind(consumerContract?.bindings.get(spec.local.name));
          cell.kind |= ctorKind(spec.imported.name ?? spec.imported.value) | cell.consumerSource;
        }
        // An import map or intermediate reexport may expose the replaced
        // renderer constructor under any namespace name. Reading an ordinary
        // named member remains local; handing out the whole namespace does not.
        else if (spec.type === "ImportNamespaceSpecifier") {
          const role = consumerContract?.namespaces.get(spec.local.name);
          const cell = scopes.get(ast).get(spec.local.name);
          cell.consumerSource = role === "orbit" ? ORBIT_NAMESPACE : role === "base" ? BASE_NAMESPACE : 0;
          cell.kind |= NAMESPACE | cell.consumerSource;
        }
      }
    } else if (functions.has(node.type)) {
      for (const parameter of node.params) declare(node, collectPatternIdentifiers(parameter));
      if (node.id) {
        declare(node, [node.id.name]);
        if (node.type === "FunctionDeclaration") declare(containing(parent => lexical.has(parent.type)), [node.id.name]);
      }
    } else if (node.type === "CatchClause") {
      declare(node, collectPatternIdentifiers(node.param));
    } else if (classes.has(node.type) && node.id) {
      declare(node, [node.id.name]);
      if (node.type === "ClassDeclaration") declare(containing(parent => lexical.has(parent.type)), [node.id.name]);
    }
  });
  const binding = node => {
    const chain = enclosing.get(node) ?? [ast];
    for (let index = chain.length - 1; index >= 0; index--) {
      const found = scopes.get(chain[index])?.get(node.name);
      if (found) return found;
    }
    if (!globals.has(node.name)) globals.set(node.name, {kind: ctorKind(node.name), local: false});
    return globals.get(node.name);
  };
  const report = (node, reason) => {
    if (reported.has(node)) return;
    reported.add(node);
    const span = toSourceSpan(node, offsets);
    escapes.push({type: "renderer_resource_escape", reason, source_span: span, sourceSpan: span});
  };
  const propertyKind = (object, names) => {
    if (!names) return 0;
    let result = names.reduce((kind, name) => kind | ctorKind(name), 0);
    if (object & ORBIT_NAMESPACE && names.includes("OrbitControls")) result |= ORBIT_CTOR;
    if (object & BASE_NAMESPACE && names.some(name => name === "Controls" || name === "EventDispatcher")) result |= BASE_CTOR;
    if (object & (RENDERER | CONTROLS) && names.some(name => name === "domElement")) result |= CANVAS;
    if (object & DOM && names.some(name => ["body", "documentElement", "parentElement", "ownerDocument"].includes(name))) result |= DOM;
    return result;
  };
  let values = new WeakMap();
  const kind = (node, depth = 0) => {
    if (!node) return 0;
    if (depth > 64) { report(node, "alias-expression-limit"); return RESOURCE; }
    if (values.has(node)) return values.get(node);
    const child = value => kind(value, depth + 1);
    let value = 0;
    switch (node.type) {
      case "Identifier": {
        const cell = binding(node);
        value = cell.kind | (!cell.local && node.name === "document" ? DOM : 0);
        break;
      }
      case "NewExpression": {
        const constructor = child(node.callee);
        value = (constructor & CONSTRUCTOR ? RENDERER : 0) | (constructor & ORBIT_CTOR ? CONTROLS : 0);
        break;
      }
      case "MemberExpression": value = propertyKind(child(node.object), staticPropertyNames(node)); break;
      case "ChainExpression": case "AwaitExpression": value = child(node.expression ?? node.argument); break;
      case "SequenceExpression": value = child(node.expressions.at(-1)); break;
      case "ConditionalExpression": value = child(node.consequent) | child(node.alternate); break;
      case "LogicalExpression": value = child(node.left) | child(node.right); break;
      case "AssignmentExpression": value = child(node.right); break;
      case "CallExpression": {
        const callee = node.callee;
        if (callee.type === "MemberExpression" && child(callee.object) & DOM &&
            staticPropertyNames(callee)?.every(name => ["createElement", "createElementNS", "getElementById", "querySelector"].includes(name))) value = DOM;
        if (callee.type === "MemberExpression" && child(callee.object) & RENDERER &&
            staticPropertyNames(callee)?.every(name => name === "setMRT")) value = RENDERER;
        break;
      }
    }
    values.set(node, value);
    return value;
  };
  let changed = false;
  const mark = (node, value) => {
    if (!node || node.type !== "Identifier") return;
    const cell = binding(node), next = cell.kind | value;
    if (next !== cell.kind) { cell.kind = next; changed = true; }
  };
  const assign = (pattern, value) => {
    if (pattern.type === "Identifier") mark(pattern, value);
    else if (pattern.type === "ObjectPattern") {
      for (const property of pattern.properties) {
        if (property.type === "Property" && property.value.type === "Identifier")
          mark(property.value, propertyKind(value, staticPropertyNames(property)));
      }
    }
  };
  // A direct call to an immutable local function is another part of this
  // ownership graph, not an opaque consumer. Flow its actual resources into
  // the lexical parameter cells; the ordinary sinks below then examine the
  // body, nested calls and closures with the same conservative rules. Returning
  // a resource still escapes: this does not invent a call-result alias model.
  const helperTargets = new Map(), helperWrites = new Set(), helperSafety = new WeakMap();
  const patternIdentifiers = pattern => {
    if (!pattern) return [];
    if (pattern.type === "Identifier") return [pattern];
    if (pattern.type === "AssignmentPattern") return patternIdentifiers(pattern.left);
    if (pattern.type === "RestElement") return patternIdentifiers(pattern.argument);
    if (pattern.type === "ArrayPattern") return pattern.elements.flatMap(patternIdentifiers);
    if (pattern.type === "ObjectPattern") return pattern.properties.flatMap(property =>
      patternIdentifiers(property.type === "RestElement" ? property.argument : property.value));
    return [];
  };
  const registerHelper = (cell, fn) => {
    if (!cell) return;
    if (helperTargets.has(cell) && helperTargets.get(cell) !== fn) helperTargets.set(cell, null);
    else helperTargets.set(cell, fn);
  };
  for (const node of nodes) {
    if (functions.has(node.type) && node.id) {
      registerHelper(scopes.get(node)?.get(node.id.name), node);
      if (node.type === "FunctionDeclaration") {
        let scope = parents.get(node);
        while (scope && !lexical.has(scope.type)) scope = parents.get(scope);
        registerHelper(scopes.get(scope)?.get(node.id.name), node);
      }
    } else if (node.type === "VariableDeclarator") {
      for (const identifier of patternIdentifiers(node.id)) {
        registerHelper(binding(identifier), parents.get(node)?.kind !== "var" && node.id.type === "Identifier" &&
          functions.has(node.init?.type) ? node.init : null);
      }
    }
    const target = node.type === "AssignmentExpression" ? node.left : node.type === "UpdateExpression" ? node.argument :
      ["ForInStatement", "ForOfStatement"].includes(node.type) && node.left.type !== "VariableDeclaration" ? node.left : null;
    for (const identifier of patternIdentifiers(target)) helperWrites.add(binding(identifier));
  }
  // Function declarations have registered outer and recursive-name cells.
  // A write through either spelling invalidates every call to that function.
  const reassignedHelpers = new Set();
  for (const [cell, fn] of helperTargets)
    if (fn && helperWrites.has(cell)) reassignedHelpers.add(fn);
  const localHelper = call => {
    if (call.type !== "CallExpression" || call.arguments.some(value => value.type === "SpreadElement")) return null;
    const callee = call.callee;
    const fn = functions.has(callee.type) ? callee : callee.type === "Identifier" && !helperWrites.has(binding(callee))
      ? helperTargets.get(binding(callee)) : null;
    if (!fn || reassignedHelpers.has(fn)) return null;
    if (!helperSafety.has(fn)) {
      const names = new Set(fn.params.filter(value => value.type === "Identifier").map(value => value.name));
      let safe = !fn.async && !fn.generator && names.size === fn.params.length && !names.has(fn.id?.name);
      // `arguments` can expose the original call frame without referring to a
      // parameter. A var/function redeclaration of a parameter also needs a
      // shared binding model that this lexical proof deliberately does not use.
      walk.full(fn.body, value => {
        if (value.type === "Identifier" && value.name === "arguments" ||
            value.type === "VariableDeclaration" && value.kind === "var" && value.declarations.some(declaration =>
              [...collectPatternIdentifiers(declaration.id)].some(name => names.has(name))) ||
            value.type === "FunctionDeclaration" && names.has(value.id?.name)) safe = false;
      });
      helperSafety.set(fn, safe);
    }
    return helperSafety.get(fn) ? fn : null;
  };
  // A bounded fixpoint covers aliases assigned before their source declaration,
  // closures and later reassignments without simulating application control flow.
  for (let iteration = 0; iteration < 64; iteration++) {
    changed = false;
    values = new WeakMap();
    for (const node of nodes) {
      if (node.type === "VariableDeclarator") assign(node.id, kind(node.init));
      else if (node.type === "AssignmentExpression") assign(node.left, kind(node.right));
      else if (node.type === "NewExpression" && kind(node.callee) & CONSTRUCTOR) {
        const canvas = sourceOptionValue(node.arguments[0], "canvas");
        if (canvas?.type === "Identifier") mark(canvas, CANVAS);
        if (node.callee.type === "MemberExpression" && node.callee.object.type === "Identifier")
          mark(node.callee.object, NAMESPACE);
      } else if (node.type === "NewExpression" && kind(node.callee) & ORBIT_CTOR) {
        mark(node.arguments[1], CANVAS);
      } else if (node.type === "CallExpression" && node.callee.type === "MemberExpression" &&
          kind(node.callee.object) & CONTROLS && staticPropertyNames(node.callee)?.every(name => name === "connect")) {
        mark(node.arguments[0], CANVAS);
      }
      const helper = localHelper(node);
      if (helper) for (let index = 0; index < helper.params.length; index++)
        mark(helper.params[index], kind(node.arguments[index]));
    }
    if (!changed) break;
    if (iteration === 63) report(ast, "alias-analysis-limit");
  }
  values = new WeakMap();
  // A known DOM receiver does not establish a native host operation after the
  // source replaces that operation. Unknown keys are conservative as well.
  const replacedDomMethods = new Set();
  let unknownDomWrite = false;
  for (const node of nodes) {
    const target = node.type === "AssignmentExpression" ? node.left :
      node.type === "UpdateExpression" || node.type === "UnaryExpression" && node.operator === "delete" ? node.argument : null;
    if (target?.type !== "MemberExpression") continue;
    const names = staticPropertyNames(target);
    if (names) for (const name of names) replacedDomMethods.add(name);
    else if (kind(target.object) & DOM) unknownDomWrite = true;
  }
  const origins = new Map();
  for (const node of nodes) {
    const target = node.type === "VariableDeclarator" ? node.id :
      node.type === "AssignmentExpression" ? node.left : null;
    const source = node.type === "VariableDeclarator" ? node.init : node.right;
    if (target?.type !== "Identifier" || !source) continue;
    const cell = binding(target);
    if (!origins.has(cell)) origins.set(cell, []);
    origins.get(cell).push(source);
  }
  const hostMethods = new Set(["createElement", "createElementNS", "getElementById", "querySelector"]);
  const closedConsumerValue = (node, expected, visiting = new Set(), depth = 0) => {
    if (!node || depth > 64) return false;
    const visit = (value, role = expected) => closedConsumerValue(value, role, visiting, depth + 1);
    if (node.type === "Identifier") {
      const cell = binding(node);
      if (!origins.has(cell)) return Boolean(cell.consumerSource & expected);
      if (!cell.consumerVariable || visiting.has(cell)) return false;
      visiting.add(cell);
      const result = origins.get(cell).every(value => visit(value));
      visiting.delete(cell);
      return result;
    }
    if (node.type === "ChainExpression") return visit(node.expression);
    if (node.type === "SequenceExpression") return visit(node.expressions.at(-1));
    if (node.type === "ConditionalExpression" || node.type === "LogicalExpression")
      return visit(node.consequent ?? node.left) && visit(node.alternate ?? node.right);
    if (node.type === "NewExpression") return expected === CONTROLS ? visit(node.callee, ORBIT_CTOR) :
      expected === RENDERER && visit(node.callee, CONSTRUCTOR);
    if (node.type === "MemberExpression") {
      const names = staticPropertyNames(node);
      if (!names) return false;
      if (expected === ORBIT_CTOR) return names.every(name => name === "OrbitControls") && visit(node.object, ORBIT_NAMESPACE);
      if (expected === CONSTRUCTOR) return names.every(name => ctorKind(name)) && visit(node.object, BASE_NAMESPACE);
    }
    return false;
  };
  const closedDomValue = (node, visiting = new Set(), depth = 0) => {
    if (!node || depth > 64) return false;
    const visit = value => closedDomValue(value, visiting, depth + 1);
    if (node.type === "Identifier") {
      const cell = binding(node);
      if (!cell.local) return node.name === "document";
      if (visiting.has(cell) || !origins.has(cell)) return false;
      visiting.add(cell);
      const result = origins.get(cell).every(visit);
      visiting.delete(cell);
      return result;
    }
    if (node.type === "ChainExpression") return visit(node.expression);
    if (node.type === "SequenceExpression") return visit(node.expressions.at(-1));
    if (node.type === "ConditionalExpression") return visit(node.consequent) && visit(node.alternate);
    if (node.type === "MemberExpression") {
      const names = staticPropertyNames(node);
      if (!names) return false;
      if (names.every(name => name === "domElement")) {
        if (kind(node.object) & RENDERER && (!consumerContract || closedConsumerValue(node.object, RENDERER))) return true;
        if (kind(node.object) & CONTROLS && closedConsumerValue(node.object, CONTROLS)) return true;
      }
      return names.every(name => ["body", "documentElement", "parentElement", "ownerDocument"].includes(name)) && visit(node.object);
    }
    if (node.type === "CallExpression" && node.callee.type === "MemberExpression")
      return !unknownDomWrite && staticPropertyNames(node.callee)?.every(name =>
        hostMethods.has(name) && !replacedDomMethods.has(name)) && visit(node.callee.object);
    return node.type === "NewExpression" && node.callee.type === "Identifier" &&
      node.callee.name === "OffscreenCanvas" && !binding(node.callee).local;
  };
  const isCanvasAttachment = (call, argument) => {
    const callee = call.callee;
    return (kind(argument) & RESOURCE) === CANVAS && callee.type === "MemberExpression" &&
      closedDomValue(callee.object) && !unknownDomWrite &&
      staticPropertyNames(callee)?.every(name => !replacedDomMethods.has(name) &&
        ["appendChild", "append", "prepend", "insertBefore", "replaceChild"].includes(name));
  };
  const nativeState = new Set(["backend", "state", "properties", "context", "extensions", "xr", "constructor", "__proto__"]);
  // These retained methods never hand their receiver to an application callback
  // unless event registration or a hook was changed; those exits stay opaque.
  const controlsMethods = new Set(["connect", "disconnect", "dispose", "update", "saveState", "reset",
    "getPolarAngle", "getAzimuthalAngle", "getDistance", "pan", "dollyIn", "dollyOut", "rotateLeft", "rotateUp"]);
  const controlsSettings = new Set(["enabled", "state", "autoRotate", "autoRotateSpeed", "enableDamping", "dampingFactor",
    "minDistance", "maxDistance", "minZoom", "maxZoom", "minPolarAngle", "maxPolarAngle", "minAzimuthAngle", "maxAzimuthAngle",
    "minTargetRadius", "maxTargetRadius", "enableZoom", "zoomSpeed", "zoomToCursor", "enableRotate", "rotateSpeed",
    "enablePan", "panSpeed", "keyPanSpeed", "keyRotateSpeed", "screenSpacePanning"]);
  const controlsViews = new Set(["domElement", "target", "cursor", "object", "keys", "mouseButtons", "touches"]);
  const reflection = new Set(["prototype", "__proto__", "constructor", "getPrototypeOf", "setPrototypeOf",
    "defineProperty", "defineProperties", "getOwnPropertyDescriptor", "getOwnPropertyDescriptors",
    "__defineGetter__", "__defineSetter__", "__lookupGetter__", "__lookupSetter__", "eval", "Function", "Reflect"]);
  const controlsHostMethods = new Set(["addEventListener", "removeEventListener", "setPointerCapture", "releasePointerCapture",
    "getRootNode", "getBoundingClientRect", ...hostMethods, "appendChild", "append", "prepend", "insertBefore", "replaceChild"]);
  const isControlsCanvasInput = (call, argument, index) => (kind(argument) & RESOURCE) === CANVAS &&
    !call.arguments.some(value => value.type === "SpreadElement") && closedDomValue(argument) &&
    (call.type === "NewExpression" && closedConsumerValue(call.callee, ORBIT_CTOR) && index === 1 ||
      call.type === "CallExpression" && call.callee.type === "MemberExpression" && closedConsumerValue(call.callee.object, CONTROLS) &&
      staticPropertyNames(call.callee)?.every(name => name === "connect") && index === 0);
  const publicMethods = new Set([
    "init", "render", "renderAsync", "compile", "compileAsync", "compute", "computeAsync", "dispose",
    "setAnimationLoop", "setSize", "setPixelRatio", "setDrawingBufferSize", "getSize", "getPixelRatio",
    "getDrawingBufferSize", "getClearColor", "getClearAlpha", "setClearColor", "setClearAlpha", "clear",
    "clearColor", "clearDepth", "clearStencil", "setRenderTarget", "getRenderTarget", "getActiveCubeFace",
    "getActiveMipmapLevel", "setViewport", "getViewport", "setScissor", "getScissor", "setScissorTest",
    "getScissorTest", "setOpaqueSort", "setTransparentSort", "readRenderTargetPixelsAsync",
    "copyTextureToTexture", "copyFramebufferToTexture", "initTexture", "initRenderTarget", "hasFeature",
    "hasFeatureAsync", "hasInitialized", "getMaxAnisotropy", "clearAsync", "clearColorAsync",
    "clearDepthAsync", "clearStencilAsync", "getClearDepth", "setClearDepth", "getClearStencil",
    "setClearStencil", "getCanvasTarget", "getOutputRenderTarget", "setOutputRenderTarget",
    "getOutputBufferType", "getMRT", "setMRT", "writeRenderTargetPixelsAsync", "copyRenderTargetPixelsAsync",
    "waitForGPU", "getAnimationLoop", "getCurrentViewport",
  ]);
  const hasRenderer = nodes.some(node => node.type === "NewExpression" && kind(node) & RENDERER);
  for (const node of nodes) {
    // A retained component contract also depends on its inherited methods and
    // host methods. Keep reflective mutation/code generation outside this slice,
    // including in a different application module from the constructor call.
    if (consumerContract && (node.type === "MemberExpression" && staticPropertyNames(node)?.some(name => reflection.has(name)) ||
        node.type === "Identifier" && ["eval", "Function", "Reflect"].includes(node.name) && !binding(node).local))
      report(node, "controls-contract-reflection");
    if (consumerContract) {
      const write = node.type === "AssignmentExpression" ? node.left :
        node.type === "UpdateExpression" || node.type === "UnaryExpression" && node.operator === "delete" ? node.argument : null;
      if (write?.type === "MemberExpression") {
        const names = staticPropertyNames(write);
        if (names?.some(name => controlsHostMethods.has(name)) || !names && kind(write.object) & (DOM | CANVAS))
          report(node, "controls-host-method-mutation");
        if (kind(write.object) & CANVAS && !names?.every(name => ["width", "height", "id", "className", "tabIndex", "style"].includes(name)))
          report(node, "opaque-canvas-write");
      }
    }
    if (node.type === "CallExpression" || node.type === "NewExpression") {
      if (node.type === "NewExpression" && kind(node.callee) & CONSTRUCTOR) {
        const canvas = sourceOptionValue(node.arguments[0], "canvas");
        if (canvas === null || canvas &&
            !(canvas.type === "Literal" && canvas.value == null) && !closedDomValue(canvas))
          report(node, "opaque-canvas-input");
      }
      if (node.type === "NewExpression" && kind(node.callee) & ORBIT_CTOR) {
        const canvas = node.arguments[1];
        if (!closedConsumerValue(node.callee, ORBIT_CTOR) || node.arguments.some(value => value.type === "SpreadElement"))
          report(node, "unclosed-controls-constructor");
        if (canvas && !(canvas.type === "Literal" && canvas.value == null) && !closedDomValue(canvas))
          report(node, "opaque-controls-canvas-input");
      }
      if (node.type === "CallExpression" && node.callee.type === "MemberExpression" &&
          kind(node.callee.object) & RENDERER &&
          !staticPropertyNames(node.callee)?.every(name => publicMethods.has(name)))
        report(node, "opaque-renderer-method");
      if (node.type === "CallExpression" && node.callee.type === "MemberExpression") {
        if (kind(node.callee.object) & CONTROLS && (!closedConsumerValue(node.callee.object, CONTROLS) ||
            !staticPropertyNames(node.callee)?.every(name => controlsMethods.has(name))))
          report(node, "opaque-controls-method");
        // DOM event callbacks receive the canvas as this/event.target. They are
        // not closed merely because the canvas came from a verified controls.
        if (consumerContract && kind(node.callee.object) & CANVAS) report(node, "opaque-canvas-method");
      }
      if (hasRenderer && node.callee.type === "Identifier" && node.callee.name === "eval")
        report(node, "opaque-eval");
      for (const [index, argument] of node.arguments.entries()) {
        const value = kind(argument.type === "SpreadElement" ? argument.argument : argument);
        const helper = localHelper(node);
        if (value & RESOURCE && !isCanvasAttachment(node, argument) && !isControlsCanvasInput(node, argument, index) &&
            !(helper && index < helper.params.length))
          report(node, "opaque-call-argument");
      }
    } else if (node.type === "AssignmentExpression") {
      if (kind(node.right) & RESOURCE &&
          (node.left.type !== "Identifier" || !binding(node.left).local)) report(node, "external-store");
      if (node.left.type === "MemberExpression" && kind(node.left.object) & RENDERER &&
          staticPropertyNames(node.left)?.some(name => publicMethods.has(name)))
        report(node, "renderer-method-replacement");
      if (node.left.type === "MemberExpression" && kind(node.left.object) & ALL_CONSTRUCTORS)
        report(node, "renderer-constructor-mutation");
      if (node.left.type === "MemberExpression" && kind(node.left.object) & CONTROLS &&
          !staticPropertyNames(node.left)?.every(name => controlsSettings.has(name))) report(node, "controls-member-write");
    } else if ((node.type === "UpdateExpression" || node.type === "UnaryExpression" && node.operator === "delete") &&
        node.argument.type === "MemberExpression" && kind(node.argument.object) & CONTROLS) {
      report(node, "controls-member-write");
    } else if (node.type === "ReturnStatement" || node.type === "YieldExpression" || node.type === "ThrowStatement" ||
        node.type === "ArrowFunctionExpression" && node.body.type !== "BlockStatement") {
      if (kind(node.argument ?? node.body) & RESOURCE) report(node, node.type === "ThrowStatement" ? "thrown-resource" : "returned-resource");
    } else if (node.type === "AssignmentPattern" && kind(node.right) & RESOURCE) {
      report(node, "resource-default-value");
    } else if (node.type === "PropertyDefinition" && kind(node.value) & RESOURCE) {
      report(node, "resource-container");
    } else if (node.type === "TaggedTemplateExpression" && node.quasi.expressions.some(value => kind(value) & RESOURCE)) {
      report(node, "opaque-call-argument");
    } else if (consumerContract && node.type === "BinaryExpression" && node.operator === "instanceof" && kind(node.left) & RESOURCE &&
        !closedConsumerValue(node.right, kind(node.left) & CONTROLS ? ORBIT_CTOR : CONSTRUCTOR)) {
      report(node, "opaque-instance-check");
    } else if (node.type === "ExportNamedDeclaration") {
      if (node.source && node.specifiers.some(spec =>
        consumerContract?.reexports.has(`${node.source.value}\0${spec.local.name}`))) report(node, "exported-resource");
      if (node.declaration?.type === "VariableDeclaration") {
        for (const declaration of node.declaration.declarations)
          if ((kind(declaration.init) | kind(declaration.id)) & RESOURCE) report(node, "exported-resource");
      }
      for (const spec of node.specifiers)
        if (kind(spec.local) & RESOURCE) report(node, "exported-resource");
    } else if (node.type === "ExportDefaultDeclaration") {
      if (kind(node.declaration) & RESOURCE) report(node, "exported-resource");
    } else if (node.type === "ExportAllDeclaration") {
      report(node, "reexported-namespace");
    } else if (node.type === "MemberExpression" && kind(node.object) & RESOURCE) {
      const names = staticPropertyNames(node);
      if (!names) report(node, "unresolved-resource-member");
      else if (kind(node.object) & RENDERER && names.some(name => nativeState.has(name)))
        report(node, "native-renderer-state");
      else if (kind(node.object) & ALL_CONSTRUCTORS && names.some(name => name === "prototype"))
        report(node, "renderer-prototype-access");
      else if (kind(node.object) & CONTROLS && !names.every(name => controlsSettings.has(name) || controlsViews.has(name) ||
          controlsMethods.has(name) && parents.get(node)?.type === "CallExpression" && parents.get(node).callee === node))
        report(node, "opaque-controls-member");
    } else if (node.type === "VariableDeclarator" && node.id.type !== "Identifier" && kind(node.init) & RESOURCE) {
      if (!(node.id.type === "ObjectPattern" && node.id.properties.every(property =>
          property.type === "Property" && property.value.type === "Identifier" &&
          staticPropertyNames(property)?.every(name => name === "domElement" || ctorKind(name)))))
        report(node, "destructured-resource");
    } else if (node.type === "Property" && kind(node.value) & RESOURCE && parents.get(node)?.type !== "ObjectPattern") {
      // A local canvas in a literal constructor options object is a declared
      // renderer input. Every other container is beyond this alias proof.
      const object = parents.get(node), owner = parents.get(object);
      if (!(node.kind === "init" && staticPropertyNames(node)?.every(name => name === "canvas") &&
          kind(node.value) & CANVAS && owner?.type === "NewExpression" &&
          owner.arguments[0] === object && kind(owner.callee) & CONSTRUCTOR)) report(node, "resource-container");
    } else if (node.type === "ArrayExpression") {
      if (node.elements.some(element => kind(element?.argument ?? element) & RESOURCE)) report(node, "resource-container");
    } else if (node.type === "SpreadElement" && kind(node.argument) & RESOURCE) report(node, "resource-container");
    else if (classes.has(node.type) && kind(node.superClass) & ALL_CONSTRUCTORS) report(node, "renderer-subclass");
  }
  return {
    escapes,
    constructorName(node) {
      const value = kind(node) & CONSTRUCTOR;
      return value === WEBGL ? "WebGLRenderer" : value === WEBGPU ? "WebGPURenderer" : null;
    },
  };
}

/**
 * Verifies that a node is strictly an import.meta.url property access:
 * noncomputed .url or computed literal ['url'] / [`url`].
 * Strictly rejects new.target.url, variable computed keys like import.meta[url], etc.
 * @param {any} node
 * @returns {boolean}
 */
function isImportMetaUrl(node) {
  if (!node || node.type !== "MemberExpression") return false;

  const obj = node.object;
  if (!obj || obj.type !== "MetaProperty") return false;
  if (!obj.meta || obj.meta.name !== "import") return false;
  if (!obj.property || obj.property.name !== "meta") return false;

  if (!node.computed) {
    return Boolean(
      node.property && node.property.type === "Identifier" && node.property.name === "url",
    );
  } else {
    if (!node.property) return false;
    if (node.property.type === "Literal") {
      return node.property.value === "url";
    }
    if (
      node.property.type === "TemplateLiteral" &&
      node.property.expressions.length === 0 &&
      node.property.quasis.length > 0
    ) {
      const val = node.property.quasis.map((q) => q.value.cooked ?? q.value.raw).join("");
      return val === "url";
    }
    return false;
  }
}

/**
 * Analyzes ES module source code using Acorn.
 *
 * @param {string} code - JavaScript module source
 * @param {string} moduleUrl - Canonical URL of the module
 * @param {Object} [offsets] - Line/column offsets for inline scripts
 * @returns {Object} Structured module AST analysis
 */
export function analyzeModuleAst(code, moduleUrl, offsets = {}, consumerContract = null) {
  let ast;
  try {
    ast = acorn.parse(code, {
      ecmaVersion: "latest",
      sourceType: "module",
      locations: true,
      ranges: true,
    });
  } catch (err) {
    const span = err.loc
      ? {
          start: {
            line: err.loc.line + (offsets.lineOffset || 0),
            column:
              err.loc.line === 1 ? err.loc.column + (offsets.columnOffset || 0) : err.loc.column,
            offset: (err.pos || 0) + (offsets.charOffset || 0),
          },
          end: {
            line: err.loc.line + (offsets.lineOffset || 0),
            column:
              (err.loc.line === 1 ? err.loc.column + (offsets.columnOffset || 0) : err.loc.column) +
              1,
            offset: (err.pos || 0) + (offsets.charOffset || 0) + 1,
          },
        }
      : null;
    throw new IngestionParseError(
      `Failed to parse module "${moduleUrl}": ${err.message}`,
      moduleUrl,
      span,
    );
  }

  const staticImports = [];
  const staticExports = [];
  const dynamicImports = [];
  const assetReferences = [];
  const classDeclarations = [];
  const prototypeWrites = [];
  const rendererConstructionSites = [];
  const escapes = [];
  let hasNativeContextAccess = false;
  let hasOpaqueGLEscapes = false;
  let hasUnresolvedContextAccess = false;

  // Track declarations and mutations for live bindings analysis
  const topLevelDeclarations = new Map(); // name -> kind ('const' | 'let' | 'var' | 'function' | 'class')
  const mutatedIdentifiers = new Set();
  const exportedBindingNames = new Set();
  let hasTopLevelSideEffects = false;

  // Process top-level body statements
  for (const node of ast.body) {
    switch (node.type) {
      case "ImportDeclaration": {
        const specifier = node.source.value;
        const importedBindings = [];
        for (const spec of node.specifiers) {
          if (spec.type === "ImportDefaultSpecifier") {
            importedBindings.push({
              local: spec.local.name,
              imported: "default",
              type: "default",
            });
            topLevelDeclarations.set(spec.local.name, "import");
          } else if (spec.type === "ImportNamespaceSpecifier") {
            importedBindings.push({
              local: spec.local.name,
              imported: "*",
              type: "namespace",
            });
            topLevelDeclarations.set(spec.local.name, "import");
          } else if (spec.type === "ImportSpecifier") {
            importedBindings.push({
              local: spec.local.name,
              imported: spec.imported.name,
              type: "named",
            });
            topLevelDeclarations.set(spec.local.name, "import");
          }
        }
        const span = toSourceSpan(node, offsets);
        staticImports.push({
          specifier,
          module_type: classifyImportType(node),
          import_attributes: Object.fromEntries((node.attributes ?? []).map((a) =>
            [a.key.name ?? a.key.value, a.value.value])),
          source_span: span,
          sourceSpan: span,
          imported_bindings: importedBindings,
        });
        break;
      }

      case "ExportNamedDeclaration": {
        const reexportSpecifier = node.source ? node.source.value : null;
        const specifiers = [];
        if (node.specifiers && node.specifiers.length > 0) {
          for (const spec of node.specifiers) {
            specifiers.push({
              local: spec.local.name,
              exported: spec.exported.name,
            });
            exportedBindingNames.add(spec.local.name);
          }
        }
        if (node.declaration) {
          const decl = node.declaration;
          if (decl.type === "VariableDeclaration") {
            for (const d of decl.declarations) {
              for (const name of collectPatternIdentifiers(d.id)) {
                topLevelDeclarations.set(name, decl.kind);
                exportedBindingNames.add(name);
                specifiers.push({ local: name, exported: name });
              }
            }
          } else if (decl.type === "FunctionDeclaration" && decl.id) {
            topLevelDeclarations.set(decl.id.name, "function");
            exportedBindingNames.add(decl.id.name);
            specifiers.push({ local: decl.id.name, exported: decl.id.name });
          } else if (decl.type === "ClassDeclaration" && decl.id) {
            topLevelDeclarations.set(decl.id.name, "class");
            exportedBindingNames.add(decl.id.name);
            specifiers.push({ local: decl.id.name, exported: decl.id.name });
          }
        }
        const span = toSourceSpan(node, offsets);
        staticExports.push({
          type: "named",
          specifier: reexportSpecifier,
          ...(node.source ? {module_type: classifyImportType(node)} : {}),
          specifiers,
          source_span: span,
          sourceSpan: span,
        });
        break;
      }

      case "ExportDefaultDeclaration": {
        let localName = null;
        if (node.declaration) {
          if (node.declaration.id && node.declaration.id.name) {
            localName = node.declaration.id.name;
            topLevelDeclarations.set(
              localName,
              node.declaration.type === "FunctionDeclaration" ? "function" : "class",
            );
          }
        }
        const span = toSourceSpan(node, offsets);
        staticExports.push({
          type: "default",
          local_name: localName,
          localName,
          source_span: span,
          sourceSpan: span,
        });
        break;
      }

      case "ExportAllDeclaration": {
        const span = toSourceSpan(node, offsets);
        staticExports.push({
          type: "all",
          specifier: node.source.value,
          module_type: classifyImportType(node),
          exported: node.exported ? node.exported.name : null,
          source_span: span,
          sourceSpan: span,
        });
        break;
      }

      case "VariableDeclaration": {
        for (const d of node.declarations) {
          for (const name of collectPatternIdentifiers(d.id)) {
            topLevelDeclarations.set(name, node.kind);
          }
        }
        break;
      }

      case "FunctionDeclaration": {
        if (node.id) {
          topLevelDeclarations.set(node.id.name, "function");
        }
        break;
      }

      case "ClassDeclaration": {
        if (node.id) {
          topLevelDeclarations.set(node.id.name, "class");
        }
        break;
      }

      default:
        // Top level statements like call expressions, conditionals, etc. constitute side effects
        hasTopLevelSideEffects = true;
        break;
    }
  }

  // Traverse AST with walk to discover:
  // 1. Dynamic imports (ImportExpression)
  // 2. Mutations to top-level identifiers (AssignmentExpression, UpdateExpression)
  // 3. Classes and prototype writes
  // 4. new URL(..., import.meta.url) asset patterns
  // 5. Renderer construction sites and WebGL escapes
  const classifyImport = createDynamicImportClassifier(ast);
  const inspectorImports = verifiedInspectorExtensionImports(code, ast);
  const rendererResources = analyzeRendererResourceEscapes(ast, offsets, consumerContract);
  walk.ancestor(ast, {
    ImportExpression(node) {
      const classified = inspectorImports.get(node.start) ?? classifyImport(node.source);
      const span = toSourceSpan(node, offsets);
      dynamicImports.push({
        ...classified,
        module_type: classifyImportType(node),
        unresolved: classified.classification === "nonliteral" || classified.runtime_fallback === true,
        source_span: span,
        sourceSpan: span,
      });
    },

    AssignmentExpression(node) {
      if (node.left.type === "Identifier") {
        mutatedIdentifiers.add(node.left.name);
      } else if (
        node.left.type === "MemberExpression" &&
        node.left.object.type === "MemberExpression" &&
        node.left.object.property.name === "prototype" &&
        node.left.object.object.type === "Identifier"
      ) {
        const className = node.left.object.object.name;
        const propName =
          node.left.property.name ||
          (node.left.property.value ? String(node.left.property.value) : null);
        const span = toSourceSpan(node, offsets);
        prototypeWrites.push({
          class_name: className,
          className,
          property_name: propName,
          propertyName: propName,
          source_span: span,
          sourceSpan: span,
        });
      }
    },

    UpdateExpression(node) {
      if (node.argument.type === "Identifier") {
        mutatedIdentifiers.add(node.argument.name);
      }
    },

    ClassDeclaration(node) {
      const className = node.id ? node.id.name : null;
      const superClass =
        node.superClass && node.superClass.type === "Identifier" ? node.superClass.name : null;
      const methods = [];
      if (node.body && node.body.body) {
        for (const item of node.body.body) {
          if (item.type === "MethodDefinition" && item.key && item.key.name) {
            methods.push(item.key.name);
          }
        }
      }
      const span = toSourceSpan(node, offsets);
      classDeclarations.push({
        name: className,
        super_class: superClass,
        superClass,
        methods,
        source_span: span,
        sourceSpan: span,
      });
    },

    NewExpression(node, ancestors) {
      // 1. Check for URL asset references: new URL(relativeLiteral, import.meta.url)
      if (
        node.callee &&
        node.callee.type === "Identifier" &&
        node.callee.name === "URL" &&
        node.arguments.length === 2 &&
        !isIdentifierShadowedAtAncestors("URL", ancestors)
      ) {
        const firstArg = node.arguments[0];
        const secondArg = node.arguments[1];
        const specifier = extractStaticString(firstArg);

        if (specifier !== null && isImportMetaUrl(secondArg)) {
          const pathPart = specifier.split(/[?#]/)[0];
          const isDirectory =
            pathPart.endsWith("/") ||
            pathPart === "." ||
            pathPart === ".." ||
            pathPart.endsWith("/.") ||
            pathPart.endsWith("/..");

          if (!isDirectory) {
            const span = toSourceSpan(node, offsets);
            assetReferences.push({
              specifier,
              source_span: span,
              sourceSpan: span,
            });
          }
        }
      }

      // 2. Check for Renderer construction: new THREE.WebGLRenderer, new WebGPURenderer, etc.
      let constructorName = rendererResources.constructorName(node.callee);
      if (!constructorName && node.callee.type === "Identifier") {
        constructorName = node.callee.name;
      } else if (!constructorName) {
        const names = staticPropertyNames(node.callee);
        if (names?.length === 1) constructorName = names[0];
      }

      if (
        [
          "WebGLRenderer",
          "WebGPURenderer",
          "CSS2DRenderer",
          "CSS3DRenderer",
          "SVGRenderer",
        ].includes(constructorName)
      ) {
        const forceValue = sourceOptionValue(node.arguments[0], "forceWebGL");
        const forceWebGLUnresolved = forceValue === null ||
          (forceValue !== undefined && forceValue.type !== "Literal");
        const forceWebGL = forceWebGLUnresolved ? "unresolved" :
          forceValue === undefined ? false : Boolean(forceValue.value);
        const canvasValue = sourceOptionValue(node.arguments[0], "canvas");
        const canvasOption = canvasValue?.type === "Literal" ? String(canvasValue.value) :
          canvasValue?.type === "Identifier" ? canvasValue.name : null;

        const span = toSourceSpan(node, offsets);
        rendererConstructionSites.push({
          constructor_name: constructorName,
          constructorName,
          force_webgl: forceWebGL,
          forceWebGL,
          has_force_webgl: forceWebGL,
          hasForceWebGL: forceWebGL,
          force_webgl_unresolved: forceWebGLUnresolved,
          forceWebGLUnresolved,
          canvas_option: canvasOption,
          canvasOption,
          source_span: span,
          sourceSpan: span,
        });
      }
    },

    MemberExpression(node, ancestors) {
      // A borrowed accessor still exposes a native handle: e.g.
      // const get = renderer['getContext']; get.call(renderer).
      // Direct calls are classified below with their actual context arguments.
      const parent = ancestors.at(-2);
      if (parent?.type === "CallExpression" && parent.callee === node) return;
      const names = staticPropertyNames(node);
      if (!names) return;
      if (names.includes("getContext")) {
        hasNativeContextAccess = true;
        hasUnresolvedContextAccess = true;
        const span = toSourceSpan(node, offsets);
        escapes.push({type: "unresolved_native_context_access", classification: "accessor",
          unresolved: true, source_span: span, sourceSpan: span});
      }
      for (const name of names) if (OPAQUE_GL_METHODS.has(name)) {
        hasOpaqueGLEscapes = true;
        escapes.push({type: "opaque_gl_method_access", method: name,
          source_span: toSourceSpan(node, offsets)});
      }
    },

    CallExpression(node) {
      const names = staticPropertyNames(node.callee);
      if (!names) return;
      if (names.includes("getContext")) {
        const contexts = extractFiniteStringCandidates(node.arguments[0]);
        const span = toSourceSpan(node, offsets);
        if (!contexts) {
          hasNativeContextAccess = true;
          hasUnresolvedContextAccess = true;
          escapes.push({
            type: "unresolved_native_context_access", classification: "nonliteral",
            unresolved: true, source_span: span, sourceSpan: span,
          });
        } else for (const ctxType of contexts) {
          // A finite choice of 2D contexts is still provably non-native. Any
          // native alternative keeps the whole application on its exact route.
          if (ctxType === "2d" || ctxType === "bitmaprenderer") continue;
          hasNativeContextAccess = true;
          const webgl = ctxType.includes("webgl");
          if (webgl) hasOpaqueGLEscapes = true;
          escapes.push({
            type: webgl ? "webgl_context_acquisition" : "native_context_acquisition",
            context_type: ctxType, contextType: ctxType,
            source_span: span, sourceSpan: span,
          });
        }
      }

      for (const name of names) if (OPAQUE_GL_METHODS.has(name)) {
        hasOpaqueGLEscapes = true;
        escapes.push({type: "opaque_gl_method_call", method: name,
          source_span: toSourceSpan(node, offsets)});
      }
    },
  });

  if (rendererResources.escapes.length) {
    hasOpaqueGLEscapes = true;
    escapes.push(...rendererResources.escapes);
  }

  // Calculate live bindings:
  // Any exported binding whose declaration is 'let' or 'var', or that is mutated anywhere in the module
  const mutableExportedBindings = [];
  for (const name of exportedBindingNames) {
    const kind = topLevelDeclarations.get(name);
    const isMutated = mutatedIdentifiers.has(name);
    if (kind === "let" || kind === "var" || isMutated) {
      mutableExportedBindings.push(name);
    }
  }

  const routingFacts = {
    has_opaque_gl_escapes: hasOpaqueGLEscapes,
    has_native_context_access: hasNativeContextAccess,
    has_unresolved_context_access: hasUnresolvedContextAccess,
    hasOpaqueGLEscapes,
    hasNativeContextAccess,
    hasUnresolvedContextAccess,
    renderer_construction_sites: rendererConstructionSites,
    rendererConstructionSites,
    escapes,
  };

  return {
    staticImports,
    staticExports,
    dynamicImports,
    assetReferences,
    asset_references: assetReferences,
    classDeclarations,
    prototypeWrites,
    rendererConstructionSites,
    renderer_construction_sites: rendererConstructionSites,
    routingFacts,
    routing_facts: routingFacts,
    hasTopLevelSideEffects,
    hasLiveBindings: mutableExportedBindings.length > 0,
    mutableExportedBindings,
  };
}
