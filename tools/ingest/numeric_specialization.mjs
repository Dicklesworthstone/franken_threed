/**
 * Discover closed update loops/pipelines and specialize ordinary ESM call sites.
 *
 * Original kernel declarations/exports/identities remain untouched. Only direct calls
 * in this source unit are rewritten, with a runtime callee-identity guard.
 * Applying this after Rollup links a chunk also covers calls across merged
 * source modules. Reachable immutable numeric helpers execute in the same Wasm
 * module as their loop, sharing typed-array storage and per-view bounds. Array
 * slots propagate through wrappers; no host adapter or per-helper copy is used.
 * Loop-bearing helper graphs and scalar delegating roots
 * are discovered transitively, but still need whole-function closure. With crossModule, admitted exported functions register
 * their original identity in the shared dispatcher; imported direct calls can
 * then select that same lazy Wasm instance across chunk and re-export boundaries.
 *
 * loopIslands additionally compiles closed for/while/do-while statements inside
 * retained callbacks, closures and methods. Adjacent closed loops share one
 * native transaction. Array outputs and mutable numeric lexical bindings are
 * published at the original boundary; property access, escaping returns and
 * yielding control stay in JS. The original statements remain the fallback.
 *
 * Array types are speculative: native type, ownership, alias and length
 * guards decide each invocation. Unsupported code is retained, never rejected
 * as an application feature. This is not a whole-application acceleration claim.
 */
import * as acorn from "acorn";
import * as walk from "acorn-walk";
import MagicString from "magic-string";
import { compileNumericCandidate } from "./numeric_candidate.mjs";
import { NumericKernelCompileError } from "./numeric_kernel.mjs";
import { hasNumericLoop, discoverNumericArrayParameters } from "./numeric_loop_discovery.mjs";
import { discoverNumericStorageHints } from "./numeric_storage_hints.mjs";
import { planNumericLoopIslands, numericLoopScopeBindings } from "./numeric_loop_islands.mjs";
import { createNumericHelperCapturePlanner, numericHelperSource } from "./numeric_helper_captures.mjs";

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

/** Shared opt-in application settings for document chunks and worker ESM. */
export function numericApplicationSettings(options = {}) {
  if (!options || typeof options !== "object" || Array.isArray(options))
    throw new TypeError("Numeric specialization options must be an object");
  for (const key of Object.keys(options)) {
    if (!["maxKernels", "maxMemoryPages", "maxIterations", "crossModule", "loopIslands"].includes(key))
      throw new TypeError(`Unknown numeric specialization option: ${key}`);
  }
  // Source-transform callers retain their conservative defaults. Application
  // builds explicitly opt in and connect native calls across their ESM graph.
  const settings = { ...options,
    crossModule: options.crossModule === undefined ? true : options.crossModule,
    loopIslands: options.loopIslands === undefined ? options.crossModule !== false : options.loopIslands,
  };
  specializeNumericModule("", settings);
  return settings;
}

/**
 * @param {string} source ESM source (or an ES-format rendered Rollup chunk)
 * @param {{sourceName?: string, runtimeModule?: string | (() => string), maxKernels?: number, maxMemoryPages?: number, maxIterations?: number, crossModule?: boolean, loopIslands?: boolean, sourcemap?: boolean}} options
 * @returns {{code: string, changed: boolean, report: object, map?: object | null}}
 */
export function specializeNumericModule(
  source,
  {
    sourceName = "<module>",
    runtimeModule = "./numeric_dispatch.mjs",
    maxKernels = 64,
    maxMemoryPages = 1024,
    maxIterations = 1000000,
    crossModule = false,
    loopIslands = false,
    sourcemap = false,
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
  if (typeof crossModule !== "boolean")
    throw new TypeError("crossModule must be a boolean");
  if (typeof loopIslands !== "boolean")
    throw new TypeError("loopIslands must be a boolean");
  if (typeof sourcemap !== "boolean")
    throw new TypeError("sourcemap must be a boolean");
  const report = {
    version: 1,
    sourceName: String(sourceName),
    route: "retained-js",
    scope: crossModule ? "direct-local-and-imported-binding-calls" : "direct-calls-in-source-unit",
    accelerated: false,
    compiledKernels: 0,
    rewrittenCalls: 0,
    candidates: [],
    refusal: null,
    ...(crossModule ? { registeredKernels: 0, importedCalls: [] } : {}),
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
  const constHelpers = new Map(), constInitializations = new WeakSet();
  for (const statement of ast.body) {
    const declaration = statement.declaration ?? statement;
    if (declaration.type !== "VariableDeclaration" || declaration.kind !== "const") continue;
    for (const variable of declaration.declarations) {
      const fn = variable.init;
      if (variable.id.type !== "Identifier" ||
          !["FunctionExpression", "ArrowFunctionExpression"].includes(fn?.type) ||
          fn.async || fn.generator || fn.params.some(param => param.type !== "Identifier")) continue;
      // A private expression name is a different lexical binding. It may be
      // erased only when unused; recursive/self-name access retains source JS.
      if (fn.id && fn.id.name !== variable.id.name) {
        let selfReference = false;
        walk.full(fn.body, node => { if (node.type === "Identifier" && node.name === fn.id.name) selfReference = true; });
        if (selfReference) continue;
      }
      constHelpers.set(variable.id.name, {fn, initializedAfter:statement.end});
      constInitializations.add(variable);
    }
  }
  const names = new Set(),
    mutations = new Set(),
    calls = [];
  let hasDirectEval = false;
  walk.full(ast, (node) => {
    if (node.type === "Identifier") names.add(node.name);
    if (node.type === "AssignmentExpression") patternNames(node.left, mutations);
    if (node.type === "UpdateExpression") patternNames(node.argument, mutations);
    if (node.type === "VariableDeclarator" && node.init && !constInitializations.has(node)) patternNames(node.id, mutations);
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
  // Imports identify eligible call-site spellings, NOT native targets. The
  // runtime checks the evaluated function identity, including when a nested
  // binding shadows a direct-call spelling. Namespace members additionally need
  // a lexical proof: reading the receiver twice is inert only for the immutable
  // imported namespace or const awaited-import binding, not for mutable or
  // shadowed locals. An awaited import can resolve to an arbitrary then-export
  // value; only the binding's inert reread is proven, never its properties.
  const importedBindings = new Map();
  const namespaceBindings = new Map();
  const destructuredImports = new Map();
  const bindings = numericLoopScopeBindings();
  const addNamespace = (name, owner, metadata) => {
    if (!namespaceBindings.has(name)) namespaceBindings.set(name, new Map());
    namespaceBindings.get(name).set(owner, metadata);
  };
  const exportedBindings = new Set();
  if (crossModule) {
    const aliases = new Map();
    for (const statement of ast.body) {
      if (statement.type === "ImportDeclaration") {
        for (const specifier of statement.specifiers) {
          if (specifier.type === "ImportNamespaceSpecifier") {
            addNamespace(specifier.local.name, ast, {
              moduleSpecifier: statement.source.value,
              receiverSemantics: "original-module-namespace",
            });
            continue;
          }
          importedBindings.set(specifier.local.name, {
            moduleSpecifier: statement.source.value,
            importedName: specifier.type === "ImportDefaultSpecifier"
              ? "default" : specifier.imported.name ?? specifier.imported.value,
          });
        }
      }
      const declaration = statement.declaration ?? statement;
      if (declaration.type === "VariableDeclaration") {
        for (const variable of declaration.declarations) {
          if (variable.id.type === "Identifier" && variable.init?.type === "Identifier")
            aliases.set(variable.id.name, variable.init.name);
        }
      }
      if (statement.type === "ExportNamedDeclaration" && !statement.source) {
        if (statement.declaration?.type === "FunctionDeclaration")
          exportedBindings.add(statement.declaration.id.name);
        if (statement.declaration?.type === "VariableDeclaration") {
          for (const variable of statement.declaration.declarations)
            patternNames(variable.id, exportedBindings);
        }
        for (const specifier of statement.specifiers) exportedBindings.add(specifier.local.name);
      }
      if (statement.type === "ExportDefaultDeclaration") {
        if (declaration.type === "Identifier") exportedBindings.add(declaration.name);
        else if (declaration.type === "FunctionDeclaration" && declaration.id)
          exportedBindings.add(declaration.id.name);
      }
    }
    walk.fullAncestor(ast, (declaration, _state, ancestors) => {
      if (declaration.type !== "VariableDeclaration" || declaration.kind !== "const") return;
      for (const variable of declaration.declarations) {
        if (variable.init?.type !== "AwaitExpression" ||
            variable.init.argument.type !== "ImportExpression") continue;
        const imported = variable.init.argument;
        const metadata = {
          moduleSpecifier: imported.source.type === "Literal" && typeof imported.source.value === "string"
            ? imported.source.value : null,
          importSourceSpan: span(imported),
        };
        if (variable.id.type === "Identifier") {
          const owner = [...ancestors].reverse().find(scope => bindings(scope, variable.id).has(variable.id.name));
          if (owner) addNamespace(variable.id.name, owner, {...metadata,
            receiverSemantics: "original-dynamic-import-value"});
        } else if (variable.id.type === "ObjectPattern") {
          const properties = variable.id.properties.map(property => {
            if (property.type !== "Property") return null;
            const importedName = !property.computed && property.key.type === "Identifier" ? property.key.name
              : property.key.type === "Literal" && typeof property.key.value === "string" ? property.key.value : null;
            const local = property.value.type === "AssignmentPattern" ? property.value.left : property.value;
            return importedName !== null && local.type === "Identifier" ? {importedName, local} : null;
          });
          if (properties.some(property => !property)) continue;
          // Destructuring, getters and defaults still execute exactly once at
          // the original await. Only later direct calls are routed; the value
          // may come from arbitrary then-export assimilation or a default, so
          // native admission always uses the evaluated function identity.
          for (const {importedName, local} of properties) {
            const owner = [...ancestors].reverse().find(scope => bindings(scope, local).has(local.name));
            if (!owner) continue;
            if (!destructuredImports.has(local.name)) destructuredImports.set(local.name, new Map());
            destructuredImports.get(local.name).set(owner, {...metadata, importedName,
              receiverSemantics: "unbound-destructured-import-value"});
          }
        }
      }
    });
    // Alias discovery may overapproximate reachability, never the target proof.
    // Only immutable ORIGINAL declarations are registered, not alias values.
    // A mutable exported alias can change later; lookup then follows its value.
    const pending = [...exportedBindings];
    while (pending.length) {
      const target = aliases.get(pending.pop());
      if (target && !exportedBindings.has(target)) {
        exportedBindings.add(target);
        pending.push(target);
      }
    }
  }
  const destructuredCalls = new Map();
  if (destructuredImports.size) walk.fullAncestor(ast, (call, _state, ancestors) => {
    if (call.type !== "CallExpression" || call.optional || call.callee.type !== "Identifier" ||
        !destructuredImports.has(call.callee.name)) return;
    const owner = [...ancestors].reverse().find(scope => bindings(scope, call.callee).has(call.callee.name));
    const binding = destructuredImports.get(call.callee.name).get(owner);
    if (binding) destructuredCalls.set(call, binding);
  });
  // Hoisted declarations are initialized throughout module evaluation. Const
  // helpers need a later registration point: until all reachable initializers
  // execute, an undefined token retains the original call and its TDZ behavior.
  // Keep every original declaration/export intact; helper closure is codegen,
  // not function replacement or source evaluation. The compiler rejects free
  // variables, shadowed callees, recursion and any unclosed transitive helper.
  const storageHints = discoverNumericStorageHints(ast);
  const helperSources = new Map();
  const helperSpans = new Map();
  const helperDeclarations = new Map();
  const helperInitializers = new Map(), helperOriginalSpans = new Map();
  for (const statement of ast.body) {
    const fn = ["ExportNamedDeclaration", "ExportDefaultDeclaration"].includes(statement.type)
      ? statement.declaration
      : statement;
    if (fn?.type !== "FunctionDeclaration" || !fn.id || mutations.has(fn.id.name)) continue;
    helperSources.set(fn.id.name, source.slice(fn.start, fn.end));
    helperSpans.set(fn.id.name, span(fn));
    helperDeclarations.set(fn.id.name, fn);
  }
  for (const [name, {fn, initializedAfter}] of constHelpers) {
    if (mutations.has(name)) continue;
    const normalized = numericHelperSource(source, tokens, fn, name);
    helperSources.set(name, normalized.source);
    helperSpans.set(name, span(fn));
    helperDeclarations.set(name, fn);
    helperInitializers.set(name, initializedAfter);
    helperOriginalSpans.set(name, normalized.originalSpan);
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
  const helperCapturePlan = createNumericHelperCapturePlanner({
    source, ast, tokens, helperSources, helperDeclarations, fresh,
    bindings: numericLoopScopeBindings(),
  });
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
  // Keep previously eligible local-call kernels ahead of newly discovered
  // export-only kernels when the unit's compilation budget is small.
  const calledNames = new Set(calls.map(call => call.callee.name));
  const declarationCandidates = ast.body.flatMap(statement => {
    const fn = statement.declaration ?? statement;
    return fn.type === "FunctionDeclaration" && fn.id ? [{fn, name:fn.id.name}] : [];
  });
  const priority = candidate => calledNames.has(candidate.name) ? 0 : 1;
  const orderedCandidates = crossModule
    ? declarationCandidates.sort((a, b) => priority(a) - priority(b)) : declarationCandidates;
  const directLoops = new Set(orderedCandidates.filter(({fn}) => hasNumericLoop(fn.body, null, false)));
  // Do not let a new delegating root consume the last slot before a previously
  // eligible loop function. Within both groups, retain the prior call priority.
  // Existing declarations keep their compilation-budget priority. Immutable
  // expression roots reuse helper normalization only for code generation; the
  // actual const initializer, lexical function and its public shape stay put.
  const expressionCandidates = [...constHelpers].map(([name, {fn}]) => ({fn, name}));
  if (crossModule) expressionCandidates.sort((a, b) => priority(a) - priority(b));
  const candidates = [...orderedCandidates.filter(candidate => directLoops.has(candidate)),
    ...orderedCandidates.filter(candidate => !directLoops.has(candidate)), ...expressionCandidates];
  const preservedSourceRegions = [];
  const inside = (region, node) => region.start <= node.start && region.end >= node.end;
  for (const candidate of candidates) {
    const {fn, name} = candidate, expressionRoot = fn.type !== "FunctionDeclaration";
    // Include roots whose only iteration lives in reachable immutable helpers.
    // Discovery never substitutes for scalar closure or callee-identity guards.
    if (!directLoops.has(candidate) && !hasNumericLoop(fn.body, helperDeclarations, true)) continue;
    // A helper in a compiled declaration already executes inside that native
    // transaction. Do not add another dispatch inside its original source, or
    // spend a second root on a private expression used only by that graph.
    const nativeRegions = expressionRoot ? report.candidates.filter(item => item.route === "guarded-numeric-wasm")
      .flatMap(item => [item.sourceSpan, ...(item.scalarHelpers ?? []).map(helper => helper.sourceSpan)]) : [];
    const sites = calls.filter(call => call.callee.name === name && !destructuredCalls.has(call) &&
      !nativeRegions.some(region => region && inside(region, call)));
    // Uncalled delegating wrappers offer no route in local-call mode. Avoid
    // adding irrelevant candidates solely because one of their callees loops.
    if (!directLoops.has(candidate) && !sites.length && !(crossModule && exportedBindings.has(name))) continue;
    const loops = fn.body.type === "BlockStatement" ? fn.body.body.filter(node => node.type === "ForStatement") : [];
    const item = {
      functionName: name,
      sourceSpan: span(fn),
      route: "retained-js",
      reason: null,
      calls: [],
    };
    report.candidates.push(item);
    // Rebinding before this module's evaluation is possible through an ESM
    // cycle. Do not capture an altered function and associate it with old code.
    if (mutations.has(name)) {
      item.reason = "MUTABLE_FUNCTION_BINDING";
      continue;
    }
    if (!sites.length && !(crossModule && exportedBindings.has(name))) {
      item.reason = "NO_LOCAL_DIRECT_CALLS";
      continue;
    }
    if (report.compiledKernels >= maxKernels) {
      item.reason = "KERNEL_BUDGET";
      continue;
    }
    // A source wrapper can pass an array straight to a helper without owning
    // a subscript itself. Infer those slots across argument positions, then
    // demand the same whole-graph compiler proof and per-invocation guards.
    const arrays = discoverNumericArrayParameters(fn, helperDeclarations);
    const lifted = helperCapturePlan(fn, [ast], expressionRoot ? name : null);
    if (lifted?.refusal) { item.reason = lifted.refusal; continue; }
    const parameterTypes = [...fn.params.map((param) => (arrays.has(param.name) ? "f64[]" : "f64")),
      ...(lifted?.captureKinds.map(kind => kind === "array" ? "f64[]" : "f64") ?? [])];
    const kernelSource = lifted ? lifted.rewrite() : expressionRoot
      ? helperSources.get(name) : source.slice(fn.start, fn.end);
    const candidateHelpers = lifted?.helperSources ?? helperSources;
    let artifact;
    try {
      artifact = compileNumericCandidate(kernelSource, {
        parameterTypes,
        helperSources: candidateHelpers,
        allowMath: true,
        sourceName: `${sourceName}:${name}`,
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
    const captureArguments = lifted?.captures.map(name => ({type:"Identifier", name}));
    const hintSites = captureArguments
      ? (sites.length ? sites : [{arguments:fn.params}]).map(site =>
        ({arguments:[...site.arguments, ...captureArguments]}))
      : sites;
    const hintedLayouts = storageHints(hintSites, artifact.manifest.parameters,
      {includeFloat:!!lifted?.arrayParameters.size});
    for (const types of [...hintedLayouts, ...layouts]) {
      if (alternatives.length === 16) break;
      if (seenLayouts.has(types.join(","))) continue;
      seenLayouts.add(types.join(","));
      try {
        const variant = compileNumericCandidate(kernelSource, {
          parameterTypes: types,
          helperSources: candidateHelpers,
          allowMath: true,
          sourceName: `${sourceName}:${name}`,
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
    const captureKinds = lifted?.arrayParameters.size ? `, captureKinds:${JSON.stringify(lifted.captureKinds)}` : "";
    const captureOptions = lifted ? `, {parameterCount:${fn.params.length}, captureCount:${lifted.captures.length}${captureKinds}, resolve:() => [${lifted.captures.join(",")}]}` : "";
    const registration = `var ${tokenName} = ${createName}(${name}, [${artifact.wasm.join(",")}], ${JSON.stringify(alternatives)}, ${mathResolver}, true${captureOptions});`;
    const initializedAfter = Math.max(helperInitializers.get(name) ?? 0,
      ...artifact.helpers.map(helper => helperInitializers.get(helper.name) ?? 0));
    if (initializedAfter) edits.push({start:initializedAfter, end:initializedAfter, text:`;\n${registration}\n`});
    else registrations.push(registration);
    helpers.push(
      `function ${helperName}(callee, ...args) { return ${dispatchName}(${tokenName}, callee, args); }`,
    );
    for (const call of sites) {
      edits.push({ start: call.callee.start, end: call.callee.end, text: helperName, sourceCallStart:call.start });
      const pos = callParen(call);
      edits.push({
        start: pos,
        end: pos,
        text: `${source.slice(call.callee.start, call.callee.end)},`,
        sourceCallStart:call.start,
      });
      item.calls.push(span(call));
    }
    item.route = "guarded-numeric-wasm";
    item.parameterTypes = parameterTypes;
    if (lifted) {
      item.helperCaptures = [...lifted.captures];
      item.helperCaptureSemantics = lifted.arrayParameters.size
        ? "live-readonly-module-bindings" : "live-readonly-module-numbers";
      if (lifted.arrayParameters.size) item.helperCaptureKinds = [...lifted.captureKinds];
    }
    if (artifact.manifest.typedArrayMethods)
      item.typedArrayMethods = [...artifact.manifest.typedArrayMethods];
    if (artifact.manifest.mathIntrinsics)
      item.mathIntrinsics = [...artifact.manifest.mathIntrinsics];
    if (artifact.manifest.mathImports)
      item.mathImports = [...artifact.manifest.mathImports];
    if (artifact.manifest.version === 8) {
      item.controlSemantics = artifact.manifest.controlSemantics;
      item.maxIterations = artifact.manifest.maxIterations;
      item.loopCount = artifact.manifest.loopCount;
      item.maxLoopDepth = artifact.manifest.maxLoopDepth;
      item.lengthParameters = [...artifact.manifest.lengthParameters];
      item.indexSemantics = artifact.manifest.indexSemantics;
      // Each helper was parsed in its OWN source slice. Never attribute its
      // loops to the root's coordinates; keep helper identity in the report.
      item.loops = artifact.controlLoops.map(({ kind, depth, sourceSpan, functionName }) => {
        if (lifted) return {kind, depth, ...(functionName ? {functionName} : {}),
          sourceSpan: lifted.originalSpan(functionName, sourceSpan)};
        if (helperOriginalSpans.has(functionName ?? name)) return {kind, depth, ...(functionName ? {functionName} : {}),
          sourceSpan:helperOriginalSpans.get(functionName ?? name)(sourceSpan)};
        const owner = functionName ? helperSpans.get(functionName) : span(fn);
        return {
          kind, depth, ...(functionName ? { functionName } : {}),
          sourceSpan: {
            start: owner.start + sourceSpan.start,
            end: owner.start + sourceSpan.end,
            line: owner.line + sourceSpan.line - 1,
            column: sourceSpan.column + (sourceSpan.line === 1 ? owner.column : 0),
          },
        };
      });
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
      if (initializedAfter) item.helperBindingSemantics = "immutable-module-bindings-after-initialization";
    }
    item.storageSemantics = "same-type-alias-preserving-v1";
    item.guardFallback = "retained-original-js";
    if (expressionRoot) {
      item.bindingSemantics = "immutable-module-expression-after-initialization";
    }
    preservedSourceRegions.push(item.sourceSpan, ...(item.scalarHelpers ?? []).map(helper => helper.sourceSpan));
    if (crossModule) {
      item.dispatchSemantics = "shared-original-function-identity-v1";
      item.exportReachable = exportedBindings.has(name);
      report.registeredKernels++;
    }
    report.compiledKernels++;
    report.rewrittenCalls += sites.length;
  }
  // Roots and their helpers may be discovered in either source order. Their
  // bodies must stay byte-for-byte intact regardless of declaration shape:
  // native execution already owns the helper calls, and a rejected transaction
  // must resume the original JS without attempting nested native transactions.
  const absorbedSourceCalls = report.candidates.flatMap(item => item.calls)
    .filter(call => preservedSourceRegions.some(region => region && inside(region, call)));
  if (absorbedSourceCalls.length) {
    const absorbed = new Set(absorbedSourceCalls.map(call => call.start));
    // A semicolon-free concise arrow may end at the call's end coordinate.
    // Its registration shares that offset but is not part of the call edit.
    for (let index = edits.length - 1; index >= 0; index--)
      if (absorbed.has(edits[index].sourceCallStart)) edits.splice(index, 1);
    for (const item of report.candidates) {
      const calls = item.calls.filter(call => absorbed.has(call.start));
      if (calls.length) {
        item.absorbedCalls = calls;
        item.calls = item.calls.filter(call => !absorbed.has(call.start));
      }
    }
    report.rewrittenCalls -= absorbedSourceCalls.length;
    report.absorbedCalls = absorbedSourceCalls.length;
  }
  // Consumer-only chunks need no kernel or Wasm bytes of their own. The source
  // dependency still initializes on the normal ESM schedule; do not import or
  // evaluate any producer through the registry, and never replace live imports.
  const runtimeImports = [];
  if (report.compiledKernels) {
    runtimeImports.push(`${crossModule ? "registerNumericDispatch" : "createNumericDispatch"} as ${createName}`,
      `dispatchNumericCall as ${dispatchName}`);
  }
  if (loopIslands) {
    const planned = planNumericLoopIslands(source, {
      ast, fresh, sourceName, maxKernels: maxKernels - report.compiledKernels,
      maxMemoryPages, maxIterations, reservedEdits: edits, helperSources, helperDeclarations, helperInitializers, helperCapturePlan,
      replaceableCalls: report.candidates.flatMap(item => item.calls),
      excludedSpans: preservedSourceRegions.filter(Boolean),
    });
    // Replace smaller direct-call token edits only after a larger original
    // statement has passed full closure. Its JS fallback is already embedded
    // unchanged. Retain standalone registrations for calls elsewhere/exports;
    // absorbed call sites must not count as separate host dispatch transitions.
    for (let i = edits.length - 1; i >= 0; i--)
      if (planned.absorbedEdits.has(edits[i])) edits.splice(i, 1);
    const absorbed = new Set(planned.report.candidates.flatMap(item =>
      (item.absorbedCalls ?? []).map(call => call.start)));
    if (absorbed.size) {
      for (const item of report.candidates) {
        const calls = item.calls.filter(call => absorbed.has(call.start));
        if (calls.length) {
          item.absorbedCalls = calls;
          item.calls = item.calls.filter(call => !absorbed.has(call.start));
        }
      }
      report.rewrittenCalls -= absorbed.size;
      report.absorbedCalls = (report.absorbedCalls ?? 0) + absorbed.size;
    }
    report.functionKernels = report.compiledKernels;
    report.loopIslands = planned.report;
    report.compiledKernels += planned.report.compiledKernels;
    edits.push(...planned.edits);
    registrations.push(...planned.registrations);
    runtimeImports.push(...planned.runtimeImports);
  }
  if (crossModule) {
    const selectedCalls = new Map(calls.filter(call => importedBindings.has(call.callee.name) &&
      !preservedSourceRegions.some(region => region && inside(region, call)))
      .map(call => [call, importedBindings.get(call.callee.name)]));
    for (const [call, binding] of destructuredCalls) {
      if (!preservedSourceRegions.some(region => region && inside(region, call)) &&
          !edits.some(edit => edit.start <= call.start && edit.end >= call.end))
        selectedCalls.set(call, binding);
    }
    const sites = [...selectedCalls];
    if (sites.length) {
      const importedDispatchName = fresh("imported_dispatch"), helperName = fresh("imported_call");
      runtimeImports.push(`dispatchImportedNumericCall as ${importedDispatchName}`);
      helpers.push(`function ${helperName}(callee, ...args) { return ${importedDispatchName}(callee, args); }`);
      for (const [call, binding] of sites) {
        edits.push({ start: call.callee.start, end: call.callee.end, text: helperName });
        const pos = callParen(call);
        edits.push({ start: pos, end: pos, text: `${source.slice(call.callee.start, call.callee.end)},` });
        report.importedCalls.push({
          ...binding, localName: call.callee.name,
          sourceSpan: span(call), route: "shared-identity-lookup-with-retained-fallback",
        });
      }
      report.rewrittenCalls += sites.length;
    }
    if (namespaceBindings.size) {
      const namespaceSites = [];
      const nativeFunctions = report.candidates.filter(item => item.route === "guarded-numeric-wasm")
        .flatMap(item => [item.sourceSpan, ...(item.scalarHelpers ?? []).map(helper => helper.sourceSpan)]);
      for (const item of report.loopIslands?.candidates ?? [])
        if (item.route === "guarded-loop-wasm")
          nativeFunctions.push(...(item.helpers ?? []).map(helper => helper.sourceSpan));
      walk.fullAncestor(ast, (call, _state, ancestors) => {
        if (call.type !== "CallExpression" || call.optional ||
            call.callee.type !== "MemberExpression" || call.callee.optional) return;
        const {object, property, computed} = call.callee;
        if (object.type !== "Identifier" || !namespaceBindings.has(object.name)) return;
        const importedName = !computed && property.type === "Identifier" ? property.name
          : computed && property.type === "Literal" && typeof property.value === "string" ? property.value : null;
        if (importedName === null) return;
        const owner = [...ancestors].reverse().find(scope => bindings(scope, object).has(object.name));
        const binding = namespaceBindings.get(object.name).get(owner);
        if (!binding) return;
        // Do not overlap an extracted loop or alter an original admitted
        // function (notably a namespace named Math whose operations already
        // have the numeric compiler's independent intrinsic guards).
        if (nativeFunctions.some(region => region && region.start <= call.start && region.end >= call.end) ||
            edits.some(edit => edit.start <= call.start && edit.end >= call.end)) return;
        namespaceSites.push({call, importedName, binding});
      });
      if (namespaceSites.length) {
        const namespaceDispatchName = fresh("namespace_dispatch"), helperName = fresh("namespace_call");
        runtimeImports.push(`dispatchImportedNumericCall as ${namespaceDispatchName}`);
        helpers.push(`function ${helperName}(callee, receiver, ...args) { return ${namespaceDispatchName}(callee, args, receiver); }`);
        for (const {call, importedName, binding} of namespaceSites) {
          const receiver = call.callee.object.name;
          edits.push({start:call.callee.start, end:call.callee.end, text:helperName});
          // The live export is selected before all source arguments. Preserve
          // that selected function even if an argument rebinds the export, and
          // retain the actual receiver as this on every source fallback. A
          // dynamic import's then-export value may have an effectful getter;
          // that member access still occurs only once, at its source position.
          const pos = callParen(call);
          edits.push({start:pos, end:pos,
            text:`${source.slice(call.callee.start, call.callee.end)},${receiver},`});
          report.importedCalls.push({...binding,
            importedName, localName:receiver, sourceSpan:span(call),
            route:"shared-identity-lookup-with-retained-fallback"});
        }
        report.rewrittenCalls += namespaceSites.length;
      }
    }
  }
  if (!runtimeImports.length) return unchanged();
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
      `\nimport { ${runtimeImports.join(", ")} } from ${JSON.stringify(runtimeSpecifier)};\n` +
      registrations.join("\n") +
      "\n",
  });
  // Apply disjoint token edits backwards. Nested calls retain their own edits
  // and all original argument expressions, comments, spreads and evaluation order.
  edits.sort((a, b) => b.start - a.start || b.end - a.end);
  const transformed = new MagicString(source);
  for (const edit of edits) {
    if (edit.start === edit.end) {
      // Preserve the backwards token edits' order for same-position insertions.
      transformed.prependLeft(edit.start, edit.text);
    } else if (edit.retainedOffset !== undefined) {
      // The original loop is the production fallback. Its individual source
      // coordinates remain useful even when the native guard declines a call.
      const end = edit.retainedOffset + edit.end - edit.start;
      if (edit.text.slice(edit.retainedOffset, end) !== source.slice(edit.start, edit.end))
        throw new Error("Numeric loop source-map boundary lost its original statement");
      transformed.prependLeft(edit.start, edit.text.slice(0, edit.retainedOffset));
      // Adjacent independently admitted loops may share this boundary. Earlier
      // suffixes precede the next loop's prefix, which was inserted first by
      // the descending edit walk.
      transformed.prependLeft(edit.end, edit.text.slice(end));
    } else {
      transformed.overwrite(edit.start, edit.end, edit.text);
    }
  }
  transformed.append("\n" + helpers.join("\n") + "\n");
  report.route = report.compiledKernels ? "mixed-js-and-guarded-numeric-wasm" : "shared-numeric-dispatch-with-retained-fallback";
  return { code: transformed.toString(), changed: true, report,
    map: sourcemap ? transformed.generateMap({hires: true, includeContent: true, source: String(sourceName)}) : null };
}
