/**
 * Discover closed update loops/pipelines and specialize ordinary ESM call sites.
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
import * as acorn from "acorn";
import * as walk from "acorn-walk";
import { compileNumericCandidate } from "./numeric_candidate.mjs";
import { NumericKernelCompileError } from "./numeric_kernel.mjs";
import { hasNumericLoop } from "./numeric_loop_discovery.mjs";
import { discoverNumericStorageHints } from "./numeric_storage_hints.mjs";

function span(node) {
  return {
    start: node.start,
    end: node.end,
    line: node.loc.start.line,
    column: node.loc.start.column,
  };
}

function patternNames(node, result) {
  if (!node) return;
  if (node.type === "Identifier") result.add(node.name);
  else if (node.type === "RestElement") patternNames(node.argument, result);
  else if (node.type === "AssignmentPattern") patternNames(node.left, result);
  else if (node.type === "ArrayPattern")
    node.elements.forEach((item) => patternNames(item, result));
  else if (node.type === "ObjectPattern") {
    node.properties.forEach((item) =>
      patternNames(item.type === "RestElement" ? item.argument : item.value, result),
    );
  }
}

/**
 * Storage hints only, never a bounds/alias/closure proof. Follow scalar index
 * expressions back to read-only parameters (e.g. a = indices[i] * 3). False
 * positives merely spend an AOT variant; native slot guards decide every call.
 */
function indexedLayouts(fn, parameters) {
  const dependencies = new Map(),
    required = new Set();
  const uses = (node) => {
    const names = new Set();
    if (node)
      walk.simple(node, {
        Identifier(identifier) {
          names.add(identifier.name);
        },
      });
    return names;
  };
  const define = (binding, value) => {
    if (binding.type !== "Identifier") return;
    if (!dependencies.has(binding.name)) dependencies.set(binding.name, new Set());
    for (const name of uses(value)) dependencies.get(binding.name).add(name);
  };
  walk.simple(fn.body, {
    VariableDeclarator(node) {
      define(node.id, node.init);
    },
    AssignmentExpression(node) {
      define(node.left, node.right);
    },
    MemberExpression(node) {
      if (node.computed) for (const name of uses(node.property)) required.add(name);
    },
  });
  const pending = [...required];
  while (pending.length) {
    for (const name of dependencies.get(pending.pop()) ?? []) {
      if (!required.has(name)) {
        required.add(name);
        pending.push(name);
      }
    }
  }
  const topology = parameters.flatMap((param, index) =>
    param.type !== "f64" && !param.write && required.has(param.name) ? [index] : [],
  );
  const bases = [];
  for (const [read, write] of [
    ["f64[]", "f64[]"],
    ["f32[]", "f32[]"],
    ["f64[]", "f32[]"],
    ["f32[]", "f64[]"],
  ]) {
    bases.push(
      parameters.map((param) => (param.type === "f64" ? "f64" : param.write ? write : read)),
    );
  }
  const layouts = [],
    seen = new Set();
  const add = (types) => {
    const key = types.join(",");
    if (layouts.length < 17 && !seen.has(key)) {
      seen.add(key);
      layouts.push(types);
    }
  };
  bases.forEach(add);
  // Bound compile size, not application functionality. Try shared topology
  // first, then individual index inputs; unlisted storage combinations retain JS.
  for (const group of [topology, ...topology.map((index) => [index])]) {
    if (!group.length) continue;
    for (const integer of ["u16[]", "u32[]"])
      for (const base of bases) {
        add(base.map((type, index) => (group.includes(index) ? integer : type)));
      }
    if (layouts.length === 17) break;
  }
  return layouts;
}

/**
 * @param {string} source ESM source (or an ES-format rendered Rollup chunk)
 * @param {{sourceName?: string, runtimeModule?: string | (() => string), maxKernels?: number, maxMemoryPages?: number, maxIterations?: number}} options
 * @returns {{code: string, changed: boolean, report: object}}
 */
export function specializeNumericModule(
  source,
  {
    sourceName = "<module>",
    runtimeModule = "./numeric_dispatch.mjs",
    maxKernels = 64,
    maxMemoryPages = 1024,
    maxIterations = 1000000,
  } = {},
) {
  if (typeof source !== "string")
    throw new TypeError("Numeric specialization requires source text");
  if (
    (typeof runtimeModule !== "string" || !runtimeModule) &&
    typeof runtimeModule !== "function"
  ) {
    throw new TypeError(
      "runtimeModule must be a nonempty module specifier or a synchronous resolver",
    );
  }
  if (!Number.isInteger(maxKernels) || maxKernels < 1 || maxKernels > 256)
    throw new RangeError("maxKernels must be between 1 and 256");
  if (!Number.isInteger(maxMemoryPages) || maxMemoryPages < 1 || maxMemoryPages > 16384)
    throw new RangeError("maxMemoryPages must be between 1 and 16384");
  if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 1000000000)
    throw new RangeError("maxIterations must be between 1 and 1000000000");
  const report = {
    version: 1,
    sourceName: String(sourceName),
    route: "retained-js",
    scope: "direct-calls-in-source-unit",
    accelerated: false,
    compiledKernels: 0,
    rewrittenCalls: 0,
    candidates: [],
    refusal: null,
  };
  const unchanged = () => ({ code: source, changed: false, report });
  let ast;
  const tokens = [];
  try {
    ast = acorn.parse(source, {
      ecmaVersion: "latest",
      sourceType: "module",
      locations: true,
      onToken: tokens,
    });
  } catch (error) {
    report.refusal = { code: "MODULE_PARSE_UNSUPPORTED", message: error.message };
    return unchanged();
  }
  const names = new Set(),
    mutations = new Set(),
    calls = [];
  let hasDirectEval = false;
  walk.full(ast, (node) => {
    if (node.type === "Identifier") names.add(node.name);
    if (node.type === "AssignmentExpression") patternNames(node.left, mutations);
    if (node.type === "UpdateExpression") patternNames(node.argument, mutations);
    if (node.type === "VariableDeclarator" && node.init) patternNames(node.id, mutations);
    if (node.type === "ForInStatement" || node.type === "ForOfStatement")
      patternNames(node.left, mutations);
    if (node.type === "CallExpression" && node.callee.type === "Identifier") {
      if (node.callee.name === "eval") hasDirectEval = true;
      if (!node.optional) calls.push(node);
    }
  });
  if (hasDirectEval) {
    report.refusal = {
      code: "DIRECT_EVAL",
      message: "Dynamic lexical access requires the original source unit",
    };
    return unchanged();
  }
  // Only hoisted declarations have a known initialized binding throughout module
  // evaluation. Const/arrow helpers need a separate TDZ/initialization proof.
  // Keep every original declaration/export intact; helper closure is codegen,
  // not function replacement or source evaluation. The compiler rejects free
  // variables, shadowed callees, recursion and any unclosed transitive helper.
  const storageHints = discoverNumericStorageHints(ast);
  const helperSources = new Map();
  const helperSpans = new Map();
  for (const statement of ast.body) {
    const fn = ["ExportNamedDeclaration", "ExportDefaultDeclaration"].includes(statement.type)
      ? statement.declaration
      : statement;
    if (fn?.type !== "FunctionDeclaration" || !fn.id || mutations.has(fn.id.name)) continue;
    helperSources.set(fn.id.name, source.slice(fn.start, fn.end));
    helperSpans.set(fn.id.name, span(fn));
  }
  // Include every identifier token, including binding/property positions skipped
  // by a semantic walker. Generated bindings cannot collide in nested scopes.
  for (const token of tokens) if (token.type.label === "name") names.add(token.value);
  let sequence = 0;
  function fresh(role) {
    let name;
    do {
      name = `__f3d_numeric_${role}_${sequence++}`;
    } while (names.has(name));
    names.add(name);
    return name;
  }
  const createName = fresh("create"),
    dispatchName = fresh("dispatch");
  const edits = [],
    registrations = [],
    helpers = [];
  const openingParens = tokens.filter((token) => token.type.label === "(");
  function callParen(call) {
    // Callees may be parenthesized and comments may contain misleading '('s.
    let low = 0,
      high = openingParens.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (openingParens[mid].start < call.callee.end) low = mid + 1;
      else high = mid;
    }
    const token = openingParens[low];
    if (!token || token.start >= call.end) throw new Error("Missing call argument delimiter");
    return token.end;
  }
  for (const statement of ast.body) {
    const fn = ["ExportNamedDeclaration", "ExportDefaultDeclaration"].includes(statement.type)
      ? statement.declaration
      : statement;
    if (fn?.type !== "FunctionDeclaration" || !fn.id) continue;
    // Discovery reaches loops under blocks/branches, but never callbacks or
    // nested declarations. Admission still compiles the WHOLE source function.
    if (!hasNumericLoop(fn.body)) continue;
    const loops = fn.body.body.filter((node) => node.type === "ForStatement");
    const item = {
      functionName: fn.id.name,
      sourceSpan: span(fn),
      route: "retained-js",
      reason: null,
      calls: [],
    };
    report.candidates.push(item);
    // Rebinding before this module's evaluation is possible through an ESM
    // cycle. Do not capture an altered function and associate it with old code.
    if (mutations.has(fn.id.name)) {
      item.reason = "MUTABLE_FUNCTION_BINDING";
      continue;
    }
    const sites = calls.filter((call) => call.callee.name === fn.id.name);
    if (!sites.length) {
      item.reason = "NO_LOCAL_DIRECT_CALLS";
      continue;
    }
    if (report.compiledKernels >= maxKernels) {
      item.reason = "KERNEL_BUDGET";
      continue;
    }
    const arrays = new Set();
    walk.simple(fn.body, {
      MemberExpression(node) {
        if (node.object.type === "Identifier") arrays.add(node.object.name);
      },
    });
    const parameterTypes = fn.params.map((param) => (arrays.has(param.name) ? "f64[]" : "f64"));
    let artifact;
    try {
      artifact = compileNumericCandidate(source.slice(fn.start, fn.end), {
        parameterTypes,
        helperSources,
        allowMath: true,
        sourceName: `${sourceName}:${fn.id.name}`,
        maxMemoryPages,
        maxIterations,
      });
    } catch (error) {
      if (!(error instanceof NumericKernelCompileError)) throw error;
      item.reason = error.code;
      item.detail = error.message;
      continue;
    }
    // Keep legacy layouts unchanged. Checked-index kernels additionally cover
    // topology and mixed input/output precision, within the dispatch AOT budget.
    const float32Types = parameterTypes.map((type) => (type === "f64[]" ? "f32[]" : type));
    const layouts =
      (artifact.manifest.version === 7 || artifact.manifest.version === 8)
        ? indexedLayouts(fn, artifact.manifest.parameters)
        : [float32Types];
    if (
      artifact.manifest.parameters.some(
        (param) => param.access?.minimumLength > 0 && !param.access.indexed,
      )
    ) {
      for (const uniformType of ["f64[]", "f32[]"]) {
        layouts.push(
          artifact.manifest.parameters.map((param) =>
            param.type === "f64"
              ? "f64"
              : param.access.indexed
                ? uniformType === "f64[]"
                  ? "f32[]"
                  : "f64[]"
                : uniformType,
          ),
        );
      }
    }
    const seenLayouts = new Set([parameterTypes.join(",")]);
    const alternatives = [];
    // Concrete allocation/wrapper hints select additional integer storage ABIs,
    // never eliminate runtime guards. Prefer observed layouts over speculative
    // topology combinations, retaining the primary and the 16-alternative cap.
    // Without integer hints, existing successful output is byte-for-byte stable.
    const hintedLayouts = storageHints(sites, artifact.manifest.parameters);
    for (const types of [...hintedLayouts, ...layouts]) {
      if (alternatives.length === 16) break;
      if (seenLayouts.has(types.join(","))) continue;
      seenLayouts.add(types.join(","));
      try {
        const variant = compileNumericCandidate(source.slice(fn.start, fn.end), {
          parameterTypes: types,
          helperSources,
          allowMath: true,
          sourceName: `${sourceName}:${fn.id.name}`,
          maxMemoryPages,
          maxIterations,
        });
        alternatives.push({ parameterTypes: types, bytes: [...variant.wasm] });
      } catch (error) {
        // One unsuitable speculative layout must not discard an admitted
        // primary or reject the application's original JavaScript.
        if (!(error instanceof NumericKernelCompileError)) throw error;
      }
    }
    const tokenName = fresh("token"),
      helperName = fresh("call");
    // var + a hoisted helper preserve calls that occur before module evaluation
    // in a cycle: an undefined token routes to the supplied original callee.
    // The resolver closes over exactly the module environment shared by this
    // declaration and its top-level helpers. Do not read Math at registration:
    // imports, mutable lexical bindings and TDZ/ESM-cycle calls need live guards.
    // Every current compiler route preserves source-ordered memory operations,
    // including fixed-loop expansion, helpers and checked structured loops.
    // Assert that producer contract for all AOT variants, not a guessed lack of
    // aliases. Future no-alias optimizations must not reuse this assertion.
    const mathResolver = artifact.manifest.mathIntrinsics ? "() => Math" : "null";
    registrations.push(
      `var ${tokenName} = ${createName}(${fn.id.name}, [${artifact.wasm.join(",")}], ${JSON.stringify(alternatives)}, ${mathResolver}, true);`,
    );
    helpers.push(
      `function ${helperName}(callee, ...args) { return ${dispatchName}(${tokenName}, callee, args); }`,
    );
    for (const call of sites) {
      edits.push({ start: call.callee.start, end: call.callee.end, text: helperName });
      const pos = callParen(call);
      edits.push({
        start: pos,
        end: pos,
        text: `${source.slice(call.callee.start, call.callee.end)},`,
      });
      item.calls.push(span(call));
    }
    item.route = "guarded-numeric-wasm";
    item.parameterTypes = parameterTypes;
    if (artifact.manifest.mathIntrinsics)
      item.mathIntrinsics = [...artifact.manifest.mathIntrinsics];
    if (artifact.manifest.version === 8) {
      item.controlSemantics = artifact.manifest.controlSemantics;
      item.maxIterations = artifact.manifest.maxIterations;
      item.loopCount = artifact.manifest.loopCount;
      item.maxLoopDepth = artifact.manifest.maxLoopDepth;
      item.lengthParameters = [...artifact.manifest.lengthParameters];
      item.indexSemantics = artifact.manifest.indexSemantics;
      // v8 compiles the original function slice. Translate nested loop spans
      // back into this source unit, not the legacy top-level-pass coordinate set.
      item.loops = artifact.controlLoops.map(({ kind, depth, sourceSpan }) => ({
        kind, depth,
        sourceSpan: {
          start: fn.start + sourceSpan.start,
          end: fn.start + sourceSpan.end,
          line: fn.loc.start.line + sourceSpan.line - 1,
          column: sourceSpan.column + (sourceSpan.line === 1 ? fn.loc.start.column : 0),
        },
      }));
    } else if (artifact.manifest.version === 6 || artifact.manifest.version === 7) {
      item.loopCount = artifact.manifest.loops.length;
      if (artifact.manifest.version === 7) {
        item.lengthParameters = [...artifact.manifest.lengthParameters];
        item.indexSemantics = artifact.manifest.indexSemantics;
      } else {
        item.boundParameters = [...artifact.manifest.boundParameters];
      }
      item.loops = artifact.manifest.loops.map((pass, index) => ({
        ...pass,
        sourceSpan: span(loops[index]),
      }));
    } else {
      item.loopStride = artifact.manifest.loopStride ?? 1;
    }
    item.resultType = artifact.manifest.resultType ?? "void";
    if (artifact.manifest.iterationSemantics)
      item.iterationSemantics = artifact.manifest.iterationSemantics;
    item.variants = [
      { parameterTypes, wasmBytes: artifact.wasm.length },
      ...alternatives.map((variant) => ({
        parameterTypes: variant.parameterTypes,
        wasmBytes: variant.bytes.length,
      })),
    ];
    item.wasmBytes = item.variants.reduce((sum, variant) => sum + variant.wasmBytes, 0);
    if (artifact.helpers.length) {
      item.scalarHelpers = artifact.helpers.map((helper) => ({
        ...helper,
        sourceSpan: helperSpans.get(helper.name),
      }));
    }
    item.storageSemantics = "same-type-alias-preserving-v1";
    item.guardFallback = "retained-original-js";
    report.compiledKernels++;
    report.rewrittenCalls += sites.length;
  }
  if (!report.compiledKernels) return unchanged();
  const runtimeSpecifier = typeof runtimeModule === "function" ? runtimeModule() : runtimeModule;
  if (typeof runtimeSpecifier !== "string" || !runtimeSpecifier)
    throw new TypeError("Runtime resolver must return a nonempty module specifier");
  // Keep hashbangs and directive prologues intact; imports remain static ESM.
  let preludeEnd = source.startsWith("#!") ? source.indexOf("\n") + 1 : 0;
  for (const statement of ast.body) {
    if (statement.type !== "ExpressionStatement" || !statement.directive) break;
    preludeEnd = statement.end;
  }
  edits.push({
    start: preludeEnd,
    end: preludeEnd,
    text:
      `\nimport { createNumericDispatch as ${createName}, dispatchNumericCall as ${dispatchName} } from ${JSON.stringify(runtimeSpecifier)};\n` +
      registrations.join("\n") +
      "\n",
  });
  // Apply disjoint token edits backwards. Nested calls retain their own edits
  // and all original argument expressions, comments, spreads and evaluation order.
  edits.sort((a, b) => b.start - a.start || b.end - a.end);
  let code = source;
  for (const edit of edits) code = code.slice(0, edit.start) + edit.text + code.slice(edit.end);
  code += "\n" + helpers.join("\n") + "\n";
  report.route = "mixed-js-and-guarded-numeric-wasm";
  return { code, changed: true, report };
}
