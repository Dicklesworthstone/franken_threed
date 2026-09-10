/**
 * AST Analyzer for ES Modules.
 * Uses pinned Acorn (8.14.0) to extract imports, exports, dynamic imports,
 * live bindings, classes, prototype writes, and asset patterns with source spans.
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
 * Classifies the argument of an ImportExpression (dynamic import).
 * @param {any} sourceNode
 * @returns {{ classification: 'literal' | 'template_enumerable' | 'nonliteral', specifier: string | null }}
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
      return { classification: 'literal', specifier: sourceNode.quasis[0].value.raw };
    }
    // Template with expressions: not a simple literal
    return { classification: 'nonliteral', specifier: null };
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
          line: err.loc.line + (offsets.lineOffset || 0),
          column: (err.loc.line === 1 ? err.loc.column + (offsets.columnOffset || 0) : err.loc.column),
          offset: (err.pos || 0) + (offsets.charOffset || 0)
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
        staticImports.push({
          specifier,
          sourceSpan: toSourceSpan(node, offsets),
          importedBindings
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
        staticExports.push({
          type: 'named',
          specifier: reexportSpecifier,
          specifiers,
          sourceSpan: toSourceSpan(node, offsets)
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
        staticExports.push({
          type: 'default',
          localName,
          sourceSpan: toSourceSpan(node, offsets)
        });
        break;
      }

      case 'ExportAllDeclaration': {
        staticExports.push({
          type: 'all',
          specifier: node.source.value,
          exported: node.exported ? node.exported.name : null,
          sourceSpan: toSourceSpan(node, offsets)
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
  walk.simple(ast, {
    ImportExpression(node) {
      const { classification, specifier } = classifyDynamicImportArgument(node.source);
      dynamicImports.push({
        classification,
        specifier,
        unresolved: classification === 'nonliteral',
        sourceSpan: toSourceSpan(node, offsets)
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
        prototypeWrites.push({
          className,
          propertyName: propName,
          sourceSpan: toSourceSpan(node, offsets)
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
      classDeclarations.push({
        name: className,
        superClass,
        methods,
        sourceSpan: toSourceSpan(node, offsets)
      });
    },

    NewExpression(node) {
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
          assetReferences.push({
            specifier: firstArg.value,
            sourceSpan: toSourceSpan(node, offsets)
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

  return {
    staticImports,
    staticExports,
    dynamicImports,
    assetReferences,
    classDeclarations,
    prototypeWrites,
    hasTopLevelSideEffects,
    hasLiveBindings: mutableExportedBindings.length > 0,
    mutableExportedBindings
  };
}
