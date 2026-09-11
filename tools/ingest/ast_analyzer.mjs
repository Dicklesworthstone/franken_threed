/**
 * AST Analyzer for ES Modules.
 * Uses pinned Acorn (8.14.0) to extract imports, exports, dynamic imports,
 * live bindings, classes, prototype writes, asset patterns, and renderer
 * routing facts with precise source spans.
 */

import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import { IngestionParseError } from './types.mjs';

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
    ? (node.loc.start.line === 1 ? node.loc.start.column + colOffset : node.loc.start.column)
    : 0;
  const startChar = (node.start !== undefined ? node.start : 0) + charOffset;

  const endLine = node.loc ? node.loc.end.line + lineOffset : 1;
  const endCol = node.loc
    ? (node.loc.end.line === 1 ? node.loc.end.column + colOffset : node.loc.end.column)
    : 0;
  const endChar = (node.end !== undefined ? node.end : 0) + charOffset;

  return {
    start: { line: startLine, column: startCol, offset: startChar },
    end: { line: endLine, column: endCol, offset: endChar }
  };
}

/**
 * Recursively extracts all string literal candidate branches from a conditional expression.
 * Returns null if any branch is not a string literal or purely literal template.
 * @param {any} node
 * @returns {string[] | null}
 */
function extractConditionalStringLiterals(node) {
  if (!node) return null;
  if (node.type === 'ConditionalExpression') {
    const consequent = extractConditionalStringLiterals(node.consequent);
    const alternate = extractConditionalStringLiterals(node.alternate);
    if (!consequent || !alternate) return null;
    return [...consequent, ...alternate];
  }
  if (node.type === 'Literal' && typeof node.value === 'string') {
    return [node.value];
  }
  if (node.type === 'TemplateLiteral' && node.expressions.length === 0 && node.quasis.length > 0) {
    return [node.quasis.map(q => q.value.cooked ?? q.value.raw).join('')];
  }
  return null;
}

/**
 * Classifies the argument of an ImportExpression (dynamic import).
 * @param {any} sourceNode
 * @returns {{ classification: 'literal' | 'finite_set' | 'nonliteral', specifier: string | null, specifiers?: string[], finite_set?: string[], finiteSet?: string[], candidates?: string[] }}
 */
function classifyDynamicImportArgument(sourceNode) {
  if (!sourceNode) {
    return { classification: 'nonliteral', specifier: null };
  }

  if (sourceNode.type === 'Literal' && typeof sourceNode.value === 'string') {
    return { classification: 'literal', specifier: sourceNode.value };
  }

  if (sourceNode.type === 'TemplateLiteral') {
    if (sourceNode.expressions.length === 0 && sourceNode.quasis.length > 0) {
      const specifier = sourceNode.quasis.map(q => q.value.cooked ?? q.value.raw).join('');
      return { classification: 'literal', specifier };
    }
    // Template with expressions: not a simple literal
    return { classification: 'nonliteral', specifier: null };
  }

  if (sourceNode.type === 'ConditionalExpression') {
    const branches = extractConditionalStringLiterals(sourceNode);
    if (branches && branches.length > 0) {
      const unique = Array.from(new Set(branches));
      return {
        classification: 'finite_set',
        specifier: null,
        specifiers: unique,
        candidates: unique,
        finite_set: unique,
        finiteSet: unique,
      };
    }
  }

  return { classification: 'nonliteral', specifier: null };
}

/**
 * Analyzes ES module source code using Acorn.
 *
 * @param {string} code - JavaScript module source
 * @param {string} moduleUrl - Canonical URL of the module
 * @param {Object} [offsets] - Line/column offsets for inline scripts
 * @returns {Object} Structured module AST analysis
 */
export function analyzeModuleAst(code, moduleUrl, offsets = {}) {
  let ast;
  try {
    ast = acorn.parse(code, {
      ecmaVersion: 'latest',
      sourceType: 'module',
      locations: true,
      ranges: true
    });
  } catch (err) {
    const span = err.loc
      ? {
          start: {
            line: err.loc.line + (offsets.lineOffset || 0),
            column: (err.loc.line === 1 ? err.loc.column + (offsets.columnOffset || 0) : err.loc.column),
            offset: (err.pos || 0) + (offsets.charOffset || 0)
          },
          end: {
            line: err.loc.line + (offsets.lineOffset || 0),
            column: (err.loc.line === 1 ? err.loc.column + (offsets.columnOffset || 0) : err.loc.column) + 1,
            offset: (err.pos || 0) + (offsets.charOffset || 0) + 1
          }
        }
      : null;
    throw new IngestionParseError(
      `Failed to parse module "${moduleUrl}": ${err.message}`,
      moduleUrl,
      span
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
      case 'ImportDeclaration': {
        const specifier = node.source.value;
        const importedBindings = [];
        for (const spec of node.specifiers) {
          if (spec.type === 'ImportDefaultSpecifier') {
            importedBindings.push({
              local: spec.local.name,
              imported: 'default',
              type: 'default'
            });
          } else if (spec.type === 'ImportNamespaceSpecifier') {
            importedBindings.push({
              local: spec.local.name,
              imported: '*',
              type: 'namespace'
            });
          } else if (spec.type === 'ImportSpecifier') {
            importedBindings.push({
              local: spec.local.name,
              imported: spec.imported.name,
              type: 'named'
            });
          }
        }
        const span = toSourceSpan(node, offsets);
        staticImports.push({
          specifier,
          source_span: span,
          sourceSpan: span,
          imported_bindings: importedBindings
        });
        break;
      }

      case 'ExportNamedDeclaration': {
        const reexportSpecifier = node.source ? node.source.value : null;
        const specifiers = [];
        if (node.specifiers && node.specifiers.length > 0) {
          for (const spec of node.specifiers) {
            specifiers.push({
              local: spec.local.name,
              exported: spec.exported.name
            });
            exportedBindingNames.add(spec.local.name);
          }
        }
        if (node.declaration) {
          const decl = node.declaration;
          if (decl.type === 'VariableDeclaration') {
            for (const d of decl.declarations) {
              if (d.id.type === 'Identifier') {
                topLevelDeclarations.set(d.id.name, decl.kind);
                exportedBindingNames.add(d.id.name);
                specifiers.push({ local: d.id.name, exported: d.id.name });
              }
            }
          } else if (decl.type === 'FunctionDeclaration' && decl.id) {
            topLevelDeclarations.set(decl.id.name, 'function');
            exportedBindingNames.add(decl.id.name);
            specifiers.push({ local: decl.id.name, exported: decl.id.name });
          } else if (decl.type === 'ClassDeclaration' && decl.id) {
            topLevelDeclarations.set(decl.id.name, 'class');
            exportedBindingNames.add(decl.id.name);
            specifiers.push({ local: decl.id.name, exported: decl.id.name });
          }
        }
        const span = toSourceSpan(node, offsets);
        staticExports.push({
          type: 'named',
          specifier: reexportSpecifier,
          specifiers,
          source_span: span,
          sourceSpan: span
        });
        break;
      }

      case 'ExportDefaultDeclaration': {
        let localName = null;
        if (node.declaration) {
          if (node.declaration.id && node.declaration.id.name) {
            localName = node.declaration.id.name;
            topLevelDeclarations.set(localName, node.declaration.type === 'FunctionDeclaration' ? 'function' : 'class');
          }
        }
        const span = toSourceSpan(node, offsets);
        staticExports.push({
          type: 'default',
          local_name: localName,
          localName,
          source_span: span,
          sourceSpan: span
        });
        break;
      }

      case 'ExportAllDeclaration': {
        const span = toSourceSpan(node, offsets);
        staticExports.push({
          type: 'all',
          specifier: node.source.value,
          exported: node.exported ? node.exported.name : null,
          source_span: span,
          sourceSpan: span
        });
        break;
      }

      case 'VariableDeclaration': {
        for (const d of node.declarations) {
          if (d.id.type === 'Identifier') {
            topLevelDeclarations.set(d.id.name, node.kind);
          }
        }
        break;
      }

      case 'FunctionDeclaration': {
        if (node.id) {
          topLevelDeclarations.set(node.id.name, 'function');
        }
        break;
      }

      case 'ClassDeclaration': {
        if (node.id) {
          topLevelDeclarations.set(node.id.name, 'class');
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
  walk.simple(ast, {
    ImportExpression(node) {
      const classified = classifyDynamicImportArgument(node.source);
      const span = toSourceSpan(node, offsets);
      dynamicImports.push({
        ...classified,
        unresolved: classified.classification === 'nonliteral',
        source_span: span,
        sourceSpan: span
      });
    },

    AssignmentExpression(node) {
      if (node.left.type === 'Identifier') {
        mutatedIdentifiers.add(node.left.name);
      } else if (
        node.left.type === 'MemberExpression' &&
        node.left.object.type === 'MemberExpression' &&
        node.left.object.property.name === 'prototype' &&
        node.left.object.object.type === 'Identifier'
      ) {
        const className = node.left.object.object.name;
        const propName = node.left.property.name || (node.left.property.value ? String(node.left.property.value) : null);
        const span = toSourceSpan(node, offsets);
        prototypeWrites.push({
          class_name: className,
          className,
          property_name: propName,
          propertyName: propName,
          source_span: span,
          sourceSpan: span
        });
      }
    },

    UpdateExpression(node) {
      if (node.argument.type === 'Identifier') {
        mutatedIdentifiers.add(node.argument.name);
      }
    },

    ClassDeclaration(node) {
      const className = node.id ? node.id.name : null;
      const superClass = node.superClass && node.superClass.type === 'Identifier' ? node.superClass.name : null;
      const methods = [];
      if (node.body && node.body.body) {
        for (const item of node.body.body) {
          if (item.type === 'MethodDefinition' && item.key && item.key.name) {
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
        sourceSpan: span
      });
    },

    NewExpression(node) {
      // 1. Check for URL asset references: new URL('...', import.meta.url)
      if (
        node.callee &&
        node.callee.type === 'Identifier' &&
        node.callee.name === 'URL' &&
        node.arguments.length >= 2
      ) {
        const firstArg = node.arguments[0];
        const secondArg = node.arguments[1];
        if (
          firstArg.type === 'Literal' &&
          typeof firstArg.value === 'string' &&
          secondArg.type === 'MemberExpression' &&
          secondArg.object.type === 'MetaProperty' &&
          secondArg.property.name === 'url'
        ) {
          const span = toSourceSpan(node, offsets);
          assetReferences.push({
            specifier: firstArg.value,
            source_span: span,
            sourceSpan: span
          });
        }
      }

      // 2. Check for Renderer construction: new THREE.WebGLRenderer, new WebGPURenderer, etc.
      let constructorName = null;
      if (node.callee.type === 'Identifier') {
        constructorName = node.callee.name;
      } else if (node.callee.type === 'MemberExpression' && node.callee.property.type === 'Identifier') {
        constructorName = node.callee.property.name;
      }

      if (['WebGLRenderer', 'WebGPURenderer', 'CSS2DRenderer', 'CSS3DRenderer', 'SVGRenderer'].includes(constructorName)) {
        let forceWebGL = false;
        let forceWebGLUnresolved = false;
        let canvasOption = null;

        if (node.arguments.length > 0 && node.arguments[0].type === 'ObjectExpression') {
          for (const prop of node.arguments[0].properties) {
            if (prop.type === 'Property') {
              const propKey = prop.key.name || prop.key.value;
              if (propKey === 'forceWebGL') {
                if (prop.value.type === 'Literal') {
                  forceWebGL = Boolean(prop.value.value);
                  forceWebGLUnresolved = false;
                } else {
                  // Non-literal value (variable, template, expression like !api.webgpu):
                  // Must be classified as unresolved, not false
                  forceWebGL = 'unresolved';
                  forceWebGLUnresolved = true;
                }
              } else if (propKey === 'canvas') {
                if (prop.value.type === 'Literal') canvasOption = String(prop.value.value);
                else if (prop.value.type === 'Identifier') canvasOption = prop.value.name;
              }
            }
          }
        }

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
          sourceSpan: span
        });
      }
    },

    CallExpression(node) {
      // Check for native context access: e.g. canvas.getContext('webgl' | 'webgl2')
      if (
        node.callee.type === 'MemberExpression' &&
        node.callee.property.type === 'Identifier' &&
        node.callee.property.name === 'getContext'
      ) {
        const arg = node.arguments[0];
        const span = toSourceSpan(node, offsets);
        const isLiteralString = arg && arg.type === 'Literal' && typeof arg.value === 'string';

        if (!isLiteralString) {
          // getContext with a non-literal argument (variable, template literal, expression, or missing)
          // MUST be classified as unresolved native-context access with a source span, never as non-native (Plan §3.3, §5.1)
          hasNativeContextAccess = true;
          hasUnresolvedContextAccess = true;
          escapes.push({
            type: 'unresolved_native_context_access',
            classification: 'nonliteral',
            unresolved: true,
            source_span: span,
            sourceSpan: span
          });
        } else {
          const ctxType = arg.value;
          if (ctxType === '2d' || ctxType === 'bitmaprenderer') {
            // Explicitly non-native 2D canvas context: ignore
          } else {
            hasNativeContextAccess = true;
            if (ctxType.includes('webgl')) {
              hasOpaqueGLEscapes = true;
              escapes.push({
                type: 'webgl_context_acquisition',
                context_type: ctxType,
                contextType: ctxType,
                source_span: span,
                sourceSpan: span
              });
            } else {
              escapes.push({
                type: 'native_context_acquisition',
                context_type: ctxType,
                contextType: ctxType,
                source_span: span,
                sourceSpan: span
              });
            }
          }
        }
      }

      // Check for direct WebGL method calls / extension queries
      if (node.callee.type === 'MemberExpression' && node.callee.property.type === 'Identifier') {
        const propName = node.callee.property.name;
        if (['getExtension', 'getParameter', 'createBuffer', 'bindBuffer', 'createTexture', 'bindTexture'].includes(propName)) {
          hasOpaqueGLEscapes = true;
          escapes.push({
            type: 'opaque_gl_method_call',
            method: propName,
            source_span: toSourceSpan(node, offsets)
          });
        }
      }
    }
  });

  // Calculate live bindings:
  // Any exported binding whose declaration is 'let' or 'var', or that is mutated anywhere in the module
  const mutableExportedBindings = [];
  for (const name of exportedBindingNames) {
    const kind = topLevelDeclarations.get(name);
    const isMutated = mutatedIdentifiers.has(name);
    if (kind === 'let' || kind === 'var' || isMutated) {
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
    escapes
  };

  return {
    staticImports,
    staticExports,
    dynamicImports,
    assetReferences,
    classDeclarations,
    prototypeWrites,
    rendererConstructionSites,
    renderer_construction_sites: rendererConstructionSites,
    routingFacts,
    routing_facts: routingFacts,
    hasTopLevelSideEffects,
    hasLiveBindings: mutableExportedBindings.length > 0,
    mutableExportedBindings
  };
}
