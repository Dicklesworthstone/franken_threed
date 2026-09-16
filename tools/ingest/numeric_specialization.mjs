/**
 * Discover closed update loops and specialize ordinary ESM call sites.
 *
 * Original declarations/exports/identities remain untouched. Only direct calls
 * in this source unit are rewritten, with a runtime callee-identity guard.
 * Applying this after Rollup links a chunk also covers calls across merged
 * source modules. Reachable immutable scalar helpers execute in the same Wasm
 * module as their loop. Calls through exports in other chunks remain JavaScript.
 *
 * Array types are speculative: native type, ownership, alias and length
 * guards decide each invocation. Unsupported code is retained, never rejected
 * as an application feature. This is not a whole-application acceleration claim.
 */
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';

function span(node) {
  return { start: node.start, end: node.end, line: node.loc.start.line, column: node.loc.start.column };
}

function patternNames(node, result) {
  if (!node) return;
  if (node.type === 'Identifier') result.add(node.name);
  else if (node.type === 'RestElement') patternNames(node.argument, result);
  else if (node.type === 'AssignmentPattern') patternNames(node.left, result);
  else if (node.type === 'ArrayPattern') node.elements.forEach(item => patternNames(item, result));
  else if (node.type === 'ObjectPattern') {
    node.properties.forEach(item => patternNames(item.type === 'RestElement' ? item.argument : item.value, result));
  }
}

/**
 * @param {string} source ESM source (or an ES-format rendered Rollup chunk)
 * @param {{sourceName?: string, runtimeModule?: string | (() => string), maxKernels?: number, maxMemoryPages?: number}} options
 * @returns {{code: string, changed: boolean, report: object}}
 */
export function specializeNumericModule(source, {
  sourceName = '<module>', runtimeModule = './numeric_dispatch.mjs',
  maxKernels = 64, maxMemoryPages = 1024,
} = {}) {
  if (typeof source !== 'string') throw new TypeError('Numeric specialization requires source text');
  if ((typeof runtimeModule !== 'string' || !runtimeModule) && typeof runtimeModule !== 'function') {
    throw new TypeError('runtimeModule must be a nonempty module specifier or a synchronous resolver');
  }
  if (!Number.isInteger(maxKernels) || maxKernels < 1 || maxKernels > 256) throw new RangeError('maxKernels must be between 1 and 256');
  if (!Number.isInteger(maxMemoryPages) || maxMemoryPages < 1 || maxMemoryPages > 16384) throw new RangeError('maxMemoryPages must be between 1 and 16384');
  const report = {
    version: 1, sourceName: String(sourceName), route: 'retained-js',
    scope: 'direct-calls-in-source-unit', accelerated: false,
    compiledKernels: 0, rewrittenCalls: 0, candidates: [], refusal: null,
  };
  const unchanged = () => ({ code: source, changed: false, report });
  let ast;
  const tokens = [];
  try {
    ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module', locations: true, onToken: tokens });
  } catch (error) {
    report.refusal = { code: 'MODULE_PARSE_UNSUPPORTED', message: error.message };
    return unchanged();
  }
  const names = new Set(), mutations = new Set(), calls = [];
  let hasDirectEval = false;
  walk.full(ast, node => {
    if (node.type === 'Identifier') names.add(node.name);
    if (node.type === 'AssignmentExpression') patternNames(node.left, mutations);
    if (node.type === 'UpdateExpression') patternNames(node.argument, mutations);
    if (node.type === 'VariableDeclarator' && node.init) patternNames(node.id, mutations);
    if (node.type === 'ForInStatement' || node.type === 'ForOfStatement') patternNames(node.left, mutations);
    if (node.type === 'CallExpression' && node.callee.type === 'Identifier') {
      if (node.callee.name === 'eval') hasDirectEval = true;
      if (!node.optional) calls.push(node);
    }
  });
  if (hasDirectEval) {
    report.refusal = { code: 'DIRECT_EVAL', message: 'Dynamic lexical access requires the original source unit' };
    return unchanged();
  }
  // Only hoisted declarations have a known initialized binding throughout module
  // evaluation. Const/arrow helpers need a separate TDZ/initialization proof.
  // Keep every original declaration/export intact; helper closure is codegen,
  // not function replacement or source evaluation. The compiler rejects free
  // variables, shadowed callees, recursion and any unclosed transitive helper.
  const helperSources = new Map();
  const helperSpans = new Map();
  for (const statement of ast.body) {
    const fn = ['ExportNamedDeclaration', 'ExportDefaultDeclaration'].includes(statement.type)
      ? statement.declaration : statement;
    if (fn?.type !== 'FunctionDeclaration' || !fn.id || mutations.has(fn.id.name)) continue;
    helperSources.set(fn.id.name, source.slice(fn.start, fn.end));
    helperSpans.set(fn.id.name, span(fn));
  }
  // Include every identifier token, including binding/property positions skipped
  // by a semantic walker. Generated bindings cannot collide in nested scopes.
  for (const token of tokens) if (token.type.label === 'name') names.add(token.value);
  let sequence = 0;
  function fresh(role) {
    let name;
    do { name = `__f3d_numeric_${role}_${sequence++}`; } while (names.has(name));
    names.add(name);
    return name;
  }
  const createName = fresh('create'), dispatchName = fresh('dispatch');
  const edits = [], registrations = [], helpers = [];
  const openingParens = tokens.filter(token => token.type.label === '(');
  function callParen(call) {
    // Callees may be parenthesized and comments may contain misleading '('s.
    let low = 0, high = openingParens.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (openingParens[mid].start < call.callee.end) low = mid + 1;
      else high = mid;
    }
    const token = openingParens[low];
    if (!token || token.start >= call.end) throw new Error('Missing call argument delimiter');
    return token.end;
  }
  for (const statement of ast.body) {
    const fn = ['ExportNamedDeclaration', 'ExportDefaultDeclaration'].includes(statement.type)
      ? statement.declaration : statement;
    if (fn?.type !== 'FunctionDeclaration' || !fn.id) continue;
    const loopPosition = fn.body.body.length - (fn.body.body.at(-1)?.type === 'ReturnStatement' ? 2 : 1);
    if (fn.body.body[loopPosition]?.type !== 'ForStatement' ||
        fn.body.body.slice(0, loopPosition).some(node => node.type !== 'VariableDeclaration')) continue;
    const item = { functionName: fn.id.name, sourceSpan: span(fn), route: 'retained-js', reason: null, calls: [] };
    report.candidates.push(item);
    // Rebinding before this module's evaluation is possible through an ESM
    // cycle. Do not capture an altered function and associate it with old code.
    if (mutations.has(fn.id.name)) { item.reason = 'MUTABLE_FUNCTION_BINDING'; continue; }
    const sites = calls.filter(call => call.callee.name === fn.id.name);
    if (!sites.length) { item.reason = 'NO_LOCAL_DIRECT_CALLS'; continue; }
    if (report.compiledKernels >= maxKernels) { item.reason = 'KERNEL_BUDGET'; continue; }
    const arrays = new Set();
    walk.simple(fn.body, {
      MemberExpression(node) {
        if (node.object.type === 'Identifier') arrays.add(node.object.name);
      },
    });
    const parameterTypes = fn.params.map(param => arrays.has(param.name) ? 'f64[]' : 'f64');
    let artifact;
    try {
      artifact = compileNumericKernel(source.slice(fn.start, fn.end), {
        parameterTypes, helperSources, sourceName: `${sourceName}:${fn.id.name}`, maxMemoryPages,
      });
    } catch (error) {
      if (!(error instanceof NumericKernelCompileError)) throw error;
      item.reason = error.code;
      item.detail = error.message;
      continue;
    }
    // At most four AOT variants: homogeneous storage, plus the two combinations
    // of streamed geometry and uniform storage. Do not enumerate 2^N ABIs.
    const float32Types = parameterTypes.map(type => type === 'f64[]' ? 'f32[]' : type);
    const layouts = [float32Types];
    if (artifact.manifest.parameters.some(param => param.access?.minimumLength > 0 && !param.access.indexed)) {
      for (const uniformType of ['f64[]', 'f32[]']) {
        layouts.push(artifact.manifest.parameters.map(param => param.type === 'f64' ? 'f64'
          : param.access.indexed ? (uniformType === 'f64[]' ? 'f32[]' : 'f64[]') : uniformType));
      }
    }
    const seenLayouts = new Set([parameterTypes.join(',')]);
    const alternatives = [];
    for (const types of layouts) {
      if (seenLayouts.has(types.join(','))) continue;
      seenLayouts.add(types.join(','));
      const variant = compileNumericKernel(source.slice(fn.start, fn.end), {
        parameterTypes: types, helperSources, sourceName: `${sourceName}:${fn.id.name}`, maxMemoryPages,
      });
      alternatives.push({ parameterTypes: types, bytes: [...variant.wasm] });
    }
    const tokenName = fresh('token'), helperName = fresh('call');
    // var + a hoisted helper preserve calls that occur before module evaluation
    // in a cycle: an undefined token routes to the supplied original callee.
    registrations.push(`var ${tokenName} = ${createName}(${fn.id.name}, [${artifact.wasm.join(',')}], ${JSON.stringify(alternatives)});`);
    helpers.push(`function ${helperName}(callee, ...args) { return ${dispatchName}(${tokenName}, callee, args); }`);
    for (const call of sites) {
      edits.push({ start: call.callee.start, end: call.callee.end, text: helperName });
      const pos = callParen(call);
      edits.push({ start: pos, end: pos, text: `${source.slice(call.callee.start, call.callee.end)},` });
      item.calls.push(span(call));
    }
    item.route = 'guarded-numeric-wasm';
    item.parameterTypes = parameterTypes;
    item.loopStride = artifact.manifest.loopStride ?? 1;
    item.resultType = artifact.manifest.resultType ?? 'void';
    if (artifact.manifest.iterationSemantics) item.iterationSemantics = artifact.manifest.iterationSemantics;
    item.variants = [
      { parameterTypes, wasmBytes: artifact.wasm.length },
      ...alternatives.map(variant => ({ parameterTypes: variant.parameterTypes, wasmBytes: variant.bytes.length })),
    ];
    item.wasmBytes = item.variants.reduce((sum, variant) => sum + variant.wasmBytes, 0);
    if (artifact.helpers.length) {
      item.scalarHelpers = artifact.helpers.map(helper => ({ ...helper, sourceSpan: helperSpans.get(helper.name) }));
    }
    item.guardFallback = 'retained-original-js';
    report.compiledKernels++;
    report.rewrittenCalls += sites.length;
  }
  if (!report.compiledKernels) return unchanged();
  const runtimeSpecifier = typeof runtimeModule === 'function' ? runtimeModule() : runtimeModule;
  if (typeof runtimeSpecifier !== 'string' || !runtimeSpecifier) throw new TypeError('Runtime resolver must return a nonempty module specifier');
  // Keep hashbangs and directive prologues intact; imports remain static ESM.
  let preludeEnd = source.startsWith('#!') ? source.indexOf('\n') + 1 : 0;
  for (const statement of ast.body) {
    if (statement.type !== 'ExpressionStatement' || !statement.directive) break;
    preludeEnd = statement.end;
  }
  edits.push({ start: preludeEnd, end: preludeEnd, text:
    `\nimport { createNumericDispatch as ${createName}, dispatchNumericCall as ${dispatchName} } from ${JSON.stringify(runtimeSpecifier)};\n` +
    registrations.join('\n') + '\n' });
  // Apply disjoint token edits backwards. Nested calls retain their own edits
  // and all original argument expressions, comments, spreads and evaluation order.
  edits.sort((a, b) => b.start - a.start || b.end - a.end);
  let code = source;
  for (const edit of edits) code = code.slice(0, edit.start) + edit.text + code.slice(edit.end);
  code += '\n' + helpers.join('\n') + '\n';
  report.route = 'mixed-js-and-guarded-numeric-wasm';
  return { code, changed: true, report };
}
