/**
 * In-place numeric loop islands for ordinary ESM callbacks, closures and methods.
 * Only a closed loop moves to Wasm; its enclosing application code stays put.
 * This planner reuses the numeric compiler and its transactional array ABI.
 * No source evaluation, inferred property purity, host-call lowering or JIT.
 */
import * as walk from 'acorn-walk';
import {compileNumericCandidate} from './numeric_candidate.mjs';
import {NumericKernelCompileError} from './numeric_kernel.mjs';
import {discoverNumericStorageHints} from './numeric_storage_hints.mjs';
import {discoverNumericArrayParameters, isNumericBulkCall} from './numeric_loop_discovery.mjs';
import {isNumericForEachCall, isNumericReductionCall, isNumericSearchCall,
  numericForEachCallbackBody, numericReductionCallbackBody} from './numeric_typed_array_ops.mjs';

const FUNCTIONS = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']);
const LOOPS = new Set(['ForStatement', 'ForOfStatement', 'WhileStatement', 'DoWhileStatement']);
const bulkStatement = node => node.type === 'ExpressionStatement' && isNumericBulkCall(node.expression);
const numericWork = node => LOOPS.has(node.type) || bulkStatement(node);
const span = node => ({start:node.start, end:node.end,
  line:node.loc.start.line, column:node.loc.start.column});
const contains = (outer, inner) => outer.start <= inner.start && inner.end <= outer.end;

function pattern(node, names) {
  if (!node) return;
  if (node.type === 'Identifier') names.add(node.name);
  else if (node.type === 'RestElement') pattern(node.argument, names);
  else if (node.type === 'AssignmentPattern') pattern(node.left, names);
  else if (node.type === 'ArrayPattern') node.elements.forEach(item => pattern(item, names));
  else if (node.type === 'ObjectPattern') node.properties.forEach(item =>
    pattern(item.type === 'RestElement' ? item.argument : item.value, names));
}

// This is only a proof that reading a capture cannot invoke a global-object
// getter. Its current type, initialization, lengths and ownership remain guarded
// at the source boundary. No initializer values or constness are assumed.
export function numericLoopScopeBindings() {
  const cache = new WeakMap();
  return (node, position) => {
    // Parameter initializers do not see body var bindings; switch discriminants
    // do not see case-block lexical bindings. Treating either as a capture could
    // hoist a global getter out of the original loop and change its effects.
    const parametersOnly = FUNCTIONS.has(node.type) && !contains(node.body, position);
    const switchHead = node.type === 'SwitchStatement' && contains(node.discriminant, position);
    const key = parametersOnly ? node.params : switchHead ? node.discriminant : node;
    if (cache.has(key)) return cache.get(key);
    const names = new Map();
    function bind(node, kind) {
      const found = new Set(); pattern(node, found);
      for (const name of found) names.set(name, kind);
    }
    function declaration(statement) {
      const value = statement.declaration ?? statement;
      if (value.type === 'VariableDeclaration') value.declarations.forEach(d => bind(d.id, value.kind));
      else if (value.type === 'FunctionDeclaration' || value.type === 'ClassDeclaration') bind(value.id, 'declaration');
      else if (value.type === 'ImportDeclaration') value.specifiers.forEach(s => bind(s.local, 'import'));
    }
    if (node.type === 'Program' || node.type === 'BlockStatement' || node.type === 'StaticBlock')
      node.body.forEach(declaration);
    if (node.type === 'SwitchStatement' && !switchHead) node.cases.forEach(c => c.consequent.forEach(declaration));
    if (node.type === 'CatchClause') bind(node.param, 'catch');
    if (node.type === 'ForStatement' && node.init?.type === 'VariableDeclaration') declaration(node.init);
    if (['ForInStatement', 'ForOfStatement'].includes(node.type) && node.left.type === 'VariableDeclaration')
      declaration(node.left);
    if (FUNCTIONS.has(node.type)) {
      bind(node.id, 'self-name');
      node.params.forEach(p => bind(p, 'parameter'));
    }
    if (node.type === 'ClassDeclaration' || node.type === 'ClassExpression') bind(node.id, 'self-name');
    if (node.type === 'Program' || (FUNCTIONS.has(node.type) && !parametersOnly) || node.type === 'StaticBlock') {
      // var may be declared in a different branch/block of this function. Do
      // not borrow declarations from a nested function or class static block.
      walk.recursive(FUNCTIONS.has(node.type) ? node.body : node, null, {
        Function() {},
        StaticBlock(value, state, next) {
          if (value === node) for (const statement of value.body) next(statement, state);
        },
        VariableDeclaration(value, state, next) {
          if (value.kind === 'var') value.declarations.forEach(d => bind(d.id, value.kind));
          for (const d of value.declarations) if (d.init) next(d.init, state);
        },
      });
    }
    cache.set(key, names);
    return names;
  };
}

function capturesFor(loop, ancestors, bindings, helperDeclarations) {
  const locals = new Set(), references = [], arrays = new Set(), assigned = new Set();
  const varInitializers = [], varNames = new Set();
  const callbacks = new WeakSet();
  walk.simple(loop, {CallExpression(node) {
    const callback = node.arguments[0];
    if ((isNumericReductionCall(node) && numericReductionCallbackBody(callback)) ||
        ((isNumericForEachCall(node) || isNumericSearchCall(node)) && numericForEachCallbackBody(callback))) {
      callbacks.add(callback);
    }
  }});
  let refusal = null, usesMath = false;
  const ownerOf = (name, scopes, position) => [...scopes].reverse()
    .find(scope => bindings(scope, position).has(name));
  walk.fullAncestor(loop, (node, _state, innerAncestors) => {
    const callbackIndex = innerAncestors.findIndex(scope => callbacks.has(scope));
    const callbackBinding = name => callbackIndex !== -1 && innerAncestors.slice(callbackIndex)
      .some(scope => bindings(scope, node).has(name));
    if (node.type === 'CallExpression' && node.callee.type === 'Identifier' &&
        helperDeclarations.has(node.callee.name)) {
      const owner = ownerOf(node.callee.name, [...ancestors, ...innerAncestors.slice(0, -1)], node);
      // The compiler resolves bindings INSIDE the extracted region. A helper
      // outside it must resolve to the proven module declaration, not to a
      // callback parameter, catch binding, block function, import or body var.
      // Check EACH call: the union of inner locals below can hide an outer
      // capture when a disjoint block declares the same spelling.
      if (owner !== ancestors[0] && (!owner || !contains(loop, owner)))
        refusal = 'ISLAND_SHADOWED_HELPER';
    }
    if (node.type === 'VariableDeclaration') {
      if (node.kind === 'var') {
        const parent = innerAncestors.at(-2);
        // Function-scoped headers publish through the scalar-state channel.
        // Body declarations and destructuring need separate initialization and
        // observation proofs; do not turn them into lexical loop temporaries.
        if (callbackIndex !== -1 || parent?.type !== 'ForStatement' || parent.init !== node ||
            node.declarations.some(d => d.id.type !== 'Identifier'))
          refusal = 'ISLAND_ESCAPING_DECLARATION';
        varInitializers.push(node);
        for (const d of node.declarations) {
          pattern(d.id, varNames);
          if (d.init) pattern(d.id, assigned);
        }
      } else if (callbackIndex === -1) for (const d of node.declarations) pattern(d.id, locals);
    }
    if ((FUNCTIONS.has(node.type) && !callbacks.has(node)) || node.type === 'ClassDeclaration' || node.type === 'ClassExpression')
      refusal = 'ISLAND_NESTED_CLOSURE';
    if ((node.type === 'ReturnStatement' && !callbacks.has(innerAncestors.findLast(scope => FUNCTIONS.has(scope.type)))) ||
        ['AwaitExpression', 'YieldExpression', 'ThisExpression', 'Super', 'MetaProperty'].includes(node.type) ||
        (['BreakStatement', 'ContinueStatement'].includes(node.type) && node.label))
      refusal = 'ISLAND_ESCAPING_CONTROL';
    if (node.type === 'AssignmentExpression' || node.type === 'UpdateExpression') {
      const names = new Set();
      pattern(node.type === 'AssignmentExpression' ? node.left : node.argument, names);
      for (const name of names) if (!callbackBinding(name)) assigned.add(name);
    }
    if (node.type === 'MemberExpression' && node.object.type === 'Identifier' && !callbackBinding(node.object.name)) {
      arrays.add(node.object.name);
      if (node.object.name === 'Math') usesMath = true;
    }
    if (node.type === 'Identifier') {
      if (callbackBinding(node.name)) return;
      // Acorn's semantic walker omits ordinary property names and labels. Keep
      // initialized binding identifiers too; an uninitialized var declaration
      // is only hoisting, not a read of its incoming value.
      const declaration = innerAncestors.at(-2);
      if (declaration?.type === 'VariableDeclarator' && declaration.id === node &&
          !declaration.init && innerAncestors.at(-3)?.kind === 'var') return;
      references.push(node);
    }
  });
  if (refusal) return {refusal};
  if (usesMath && (locals.has('Math') || varNames.has('Math'))) return {refusal:'ISLAND_LOCAL_MATH'};
  const captures = [...new Set(references.sort((a,b) => a.start-b.start).map(n => n.name))]
    .filter(name => !locals.has(name) && !(usesMath && name === 'Math') &&
      !(helperDeclarations.has(name) && ownerOf(name, ancestors, loop) === ancestors[0]));
  // A captured view can be accessed only through local references or helper
  // arguments. Discover both, even when the module has no helper declarations.
  // This proposes types; lexical closure and native slot guards still decide.
  for (const name of discoverNumericArrayParameters({
    params:captures.map(name => ({type:'Identifier', name})), body:loop,
  }, helperDeclarations)) arrays.add(name);
  // A syntactic union of inner declarations can cause conservative refusals
  // under shadowing; it cannot authorize a missing lexical binding, because the
  // numeric compiler independently resolves every reference in the whole loop.
  const scalarOutputs = captures.filter(name => assigned.has(name));
  for (const name of scalarOutputs) {
    if (arrays.has(name)) return {refusal:'ISLAND_REBOUND_ARRAY'};
    const owner = [...ancestors].reverse().map(node => bindings(node, loop)).find(scope => scope.has(name));
    // Publishing into const/import/self-name bindings can throw even when a
    // source loop takes zero trips. Only ordinary mutable environments qualify.
    if (!['let', 'var', 'parameter', 'catch'].includes(owner?.get(name)))
      return {refusal:'ISLAND_IMMUTABLE_SCALAR_OUTPUT'};
  }
  if (captures.length > 64) return {refusal:'ISLAND_CAPTURE_BUDGET'};
  if (captures.some(name => name === 'arguments' || !ancestors.some(node => bindings(node, loop).has(name))))
    return {refusal:'ISLAND_NONLEXICAL_CAPTURE'};
  return {captures, arrays, usesMath, scalarOutputs, varInitializers};
}

// Each top-level for initializer executes on every successful completion of
// this region, even when the loop takes zero trips. A direct assignment before
// any occurrence of its binding makes the incoming value irrelevant. Scan all
// earlier statements conservatively; nested/conditional writes never qualify.
function outputOnlyScalars(loop, scalarOutputs) {
  const outputs = new Set(scalarOutputs), overwritten = new Set(), seen = new Set();
  const references = (node, names) => walk.full(node, child => {
    if (child.type === 'Identifier') names.add(child.name);
  });
  function assignment(id, value) {
    const reads = new Set(); references(value, reads);
    if (id.type === 'Identifier' && outputs.has(id.name) &&
        !seen.has(id.name) && !reads.has(id.name)) overwritten.add(id.name);
    references(id, seen); for (const name of reads) seen.add(name);
  }
  function initializer(node) {
    if (!node) return;
    if (node.type === 'SequenceExpression') node.expressions.forEach(initializer);
    else if (node.type === 'VariableDeclaration') {
      for (const d of node.declarations) if (d.init) assignment(d.id, d.init);
    } else if (node.type === 'AssignmentExpression' && node.operator === '=') assignment(node.left, node.right);
    else references(node, seen);
  }
  for (const statement of loop.type === 'BlockStatement' ? loop.body : [loop]) {
    if (statement.type === 'ForStatement') initializer(statement.init);
    references(statement, seen);
  }
  return overwritten;
}

/**
 * Plan disjoint statement replacements in the ORIGINAL parsed source unit.
 * Called after whole-function discovery so its admitted regions keep priority.
 * maxKernels is the remaining shared unit budget. fresh must reserve names
 * against every original identifier token, including property/binding positions.
 * helperSources/helperDeclarations must be the matching immutable MODULE
 * declarations proved by the caller. This planner independently resolves each
 * root call's lexical owner; the numeric compiler closes the transitive graph.
 */
export function planNumericLoopIslands(source, {
  ast, fresh, sourceName = '<module>', maxKernels = 64, maxMemoryPages = 1024,
  maxIterations = 1000000, excludedSpans = [], reservedEdits = [],
  helperSources = new Map(), helperDeclarations = new Map(), replaceableCalls = [],
}) {
  const report = {version:1, scope:'closed-loop-statements-in-original-lexical-environment',
    compiledKernels:0, candidates:[], accelerated:false};
  const edits = [], registrations = [], runtimeImports = [], accepted = [], absorbedEdits = new Set();
  const candidates = [], loopContainers = new WeakSet();
  const bindings = numericLoopScopeBindings(), hints = discoverNumericStorageHints(ast);
  const helperMath = new Map();
  function usesHelperMath(name) {
    if (!helperMath.has(name)) {
      let uses = false;
      walk.simple(helperDeclarations.get(name).body, {
        MemberExpression(node) { if (node.object.type === 'Identifier' && node.object.name === 'Math') uses = true; },
      });
      helperMath.set(name, uses);
    }
    return helperMath.get(name);
  }
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    // The walk is postorder: a containing callback operation inherits the
    // priority of its explicit loops before either replacement is selected.
    if (LOOPS.has(node.type)) for (const scope of ancestors) loopContainers.add(scope);
    if (numericWork(node)) candidates.push({node, ancestors:ancestors.slice(0,-1),
      loopPriority:loopContainers.has(node), ...(bulkStatement(node) ? {kind:'TypedArrayBulkStatement'} : {})});
    // A sequence is closed only if ALL its statements are closed. Combining
    // adjacent numeric work removes intermediate copy/publication boundaries
    // without crossing arbitrary calls, declarations or suspension points. Bulk
    // methods join only when their entire original call passes native closure.
    // If the larger proof fails, the individual candidates remain available.
    if (['Program', 'BlockStatement', 'StaticBlock'].includes(node.type)) {
      for (let first=0; first<node.body.length;) {
        if (!numericWork(node.body[first])) { first++; continue; }
        let end=first+1;
        while (end<node.body.length && numericWork(node.body[end])) end++;
        if (end-first>1) {
          const body=node.body.slice(first,end), start=body[0], last=body.at(-1);
          candidates.push({kind:body.every(statement => LOOPS.has(statement.type)) ? 'LoopSequence' : 'NumericWorkSequence',
            loopPriority:body.some(statement => loopContainers.has(statement)), ancestors:[...ancestors], node:{
            type:'BlockStatement', body, start:start.start, end:last.end,
            loc:{start:start.loc.start,end:last.loc.end},
          }});
        }
        first=end;
      }
    }
  });
  // Prefer a closed sequence or outer loop over multiple native transitions. If
  // the larger region has effects, independently closed loops remain eligible.
  // Newly discovered standalone bulk work must not displace an established
  // loop when compilation capacity is small. A larger region containing that
  // loop retains priority, subject to the same full-region closure proof.
  candidates.sort((a,b) => Number(b.loopPriority)-Number(a.loopPriority) || a.node.start-b.node.start || b.node.end-a.node.end);
  let createName, dispatchName, stateDispatchName;
  for (const {node, ancestors, kind=node.type} of candidates) {
    // Never splice a second edit across an accepted region, even if a future
    // candidate priority rule orders an inner operation before its container.
    if (excludedSpans.some(range => contains(range,node)) ||
        accepted.some(range => contains(range,node) || contains(node,range))) continue;
    const item = {sourceSpan:span(node), kind, route:'retained-js', reason:null};
    report.candidates.push(item);
    if (report.compiledKernels >= maxKernels) { item.reason='KERNEL_BUDGET'; continue; }
    if (ancestors.at(-1)?.type === 'LabeledStatement') { item.reason='ISLAND_LABELED_LOOP'; continue; }
    const nestedEdits = reservedEdits.filter(edit => contains(node, edit));
    const nestedCalls = replaceableCalls.filter(call => contains(node, call));
    // Previously selected direct-call routes are replaceable ONLY after the
    // larger ORIGINAL region and all its helper bindings close successfully.
    // Unknown edits stay reserved. A failed proof/budget never removes an
    // existing route, and fallback keeps the original statement in place.
    if (nestedEdits.some(edit => !nestedCalls.some(call => contains(call, edit)))) {
      item.reason='ISLAND_EXISTING_CALL_ROUTE'; continue;
    }
    const closure = capturesFor(node, ancestors, bindings, helperDeclarations);
    if (closure.refusal) { item.reason=closure.refusal; continue; }
    const {captures, arrays, usesMath, scalarOutputs, varInitializers} = closure;
    const outputOnly = outputOnlyScalars(node, scalarOutputs);
    const inputs = captures.filter(name => !scalarOutputs.includes(name));
    const stateName = scalarOutputs.length ? fresh('loop_state') : null;
    const parameters = stateName ? [...inputs, stateName] : inputs;
    const prologue = scalarOutputs.map((name, i) => `let ${name} = ${stateName}[${i}];`).join('\n');
    const epilogue = scalarOutputs.map((name, i) => `${stateName}[${i}] = ${name};`).join('\n');
    const functionName = fresh('loop_kernel');
    let body = source.slice(node.start, node.end);
    for (const declaration of [...varInitializers].sort((a, b) => b.start - a.start)) {
      const assignments = declaration.declarations.filter(d => d.init)
        .map(d => `${d.id.name} = (${source.slice(d.init.start, d.init.end)})`).join(', ');
      body = body.slice(0, declaration.start - node.start) + assignments + body.slice(declaration.end - node.start);
    }
    const kernelSource = `function ${functionName}(${parameters.join(',')}) {\n${prologue}\n${body}\n${epilogue}\n}`;
    const parameterTypes = parameters.map(name => arrays.has(name) || name === stateName ? 'f64[]' : 'f64');
    const compile = types => compileNumericCandidate(kernelSource, {
      parameterTypes:types, helperSources, allowMath:usesMath || helperSources.size > 0, generalControl:true,
      maxMemoryPages, maxIterations, sourceName:`${sourceName}:loop@${node.start}`,
    });
    let artifact;
    try {
      artifact = compile(parameterTypes);
      if (artifact.manifest.resultType !== 'void') throw new Error('Loop compiler produced a non-void result');
    } catch (error) {
      if (!(error instanceof NumericKernelCompileError)) throw error;
      item.reason=error.code; item.detail=error.message; continue;
    }
    const params=artifact.manifest.parameters;
    const layouts=hints([{arguments:parameters.map(name => ({type:'Identifier',name}))}],params);
    for (const [read,write] of [['f32[]','f32[]'],['f64[]','f32[]'],['f32[]','f64[]']])
      layouts.push(params.map(p => p.type==='f64' ? 'f64' : p.write ? write : read));
    const alternatives=[], seen=new Set([parameterTypes.join(',')]);
    for (const layout of layouts) {
      // Scalar observations must never narrow to f32 just because a geometry
      // output uses that storage. The private output channel is always f64.
      const types = layout.map((type, i) => parameters[i] === stateName ? 'f64[]' : type);
      if (alternatives.length===16) break;
      const key=types.join(','); if(seen.has(key)) continue; seen.add(key);
      try { alternatives.push({parameterTypes:types,bytes:[...compile(types).wasm]}); }
      catch(error) { if(!(error instanceof NumericKernelCompileError)) throw error; }
    }
    if (!createName) {
      createName=fresh('create_loop');
      runtimeImports.push(`createNumericLoopDispatch as ${createName}`);
    }
    const token=fresh('loop_token');
    // A helper resolves Math in the MODULE, not in this callback's environment.
    // Capture that resolver lazily at module registration, without reading its
    // value during initialization (including ESM cycles and lexical TDZ).
    const moduleMath = !!artifact.manifest.mathIntrinsics && artifact.helpers.some(helper => usesHelperMath(helper.name));
    registrations.push(`var ${token} = ${createName}([${artifact.wasm.join(',')}], ${JSON.stringify(alternatives)}${moduleMath ? ', () => Math' : ''});`);
    // An extra block prevents a dangling else from rebinding to our predicate.
    // The original statement (including lexical declarations and comments) is
    // unchanged in the fallback. Capture failures have no source effects.
    const mathResolver=usesMath && artifact.manifest.mathIntrinsics ? '() => Math' : 'null';
    if (stateName) {
      if (!stateDispatchName) {
        stateDispatchName=fresh('dispatch_state_loop');
        runtimeImports.push(`dispatchNumericStateLoop as ${stateDispatchName}`);
      }
      // Even an overwritten lexical binding must be read: a TDZ write in the
      // source must fall back before any native array publication. The comma
      // expression ignores only its old value, without coercion or property access.
      const result=fresh('loop_result'), values=[...inputs,
        ...scalarOutputs.map(name => outputOnly.has(name) ? `(${name}, 0)` : name)].join(',');
      const publish=scalarOutputs.map((name,i) => `${name} = ${result}[${i}];`).join('\n');
      edits.push({start:node.start,end:node.end,text:
        `{ const ${result} = ${stateDispatchName}(${token}, () => [${values}], ${scalarOutputs.length}, ${mathResolver});\n` +
        `if (${result} === null) {\n${source.slice(node.start,node.end)}\n} else {\n${publish}\n} }`});
    } else {
      if (!dispatchName) {
        dispatchName=fresh('dispatch_loop');
        runtimeImports.push(`dispatchNumericLoop as ${dispatchName}`);
      }
      edits.push({start:node.start,end:node.end,text:
        `{ if (!${dispatchName}(${token}, () => [${captures.join(',')}], ${mathResolver})) {\n${source.slice(node.start,node.end)}\n} }`});
    }
    accepted.push(node); report.compiledKernels++;
    for (const edit of nestedEdits) absorbedEdits.add(edit);
    if (nestedCalls.length) item.absorbedCalls = nestedCalls.map(call => ({...call}));
    Object.assign(item, {route:'guarded-loop-wasm', captures:params.map(p => ({...p})),
      ...(stateName ? {scalarOutputs:[...scalarOutputs], scalarOutputSemantics:'mutable-lexical-f64-publication'} : {}),
      ...(outputOnly.size ? {outputOnlyScalars:[...outputOnly]} : {}),
      storageSemantics:'same-type-alias-preserving-v1', guardFallback:'original-loop-in-place',
      controlSemantics:artifact.manifest.controlSemantics, maxIterations:artifact.manifest.maxIterations,
      loopCount:artifact.manifest.loopCount, maxLoopDepth:artifact.manifest.maxLoopDepth,
      variants:[{parameterTypes,wasmBytes:artifact.wasm.length}, ...alternatives.map(v =>
        ({parameterTypes:v.parameterTypes,wasmBytes:v.bytes.length}))],
      ...(artifact.manifest.typedArrayMethods ? {typedArrayMethods:[...artifact.manifest.typedArrayMethods]} : {}),
      ...(artifact.manifest.mathIntrinsics ? {mathIntrinsics:[...artifact.manifest.mathIntrinsics]} : {}),
      ...(artifact.manifest.mathImports ? {mathImports:[...artifact.manifest.mathImports]} : {}),
      ...(artifact.helpers.length ? {
        helperBindingSemantics:'immutable-module-declarations',
        helpers:artifact.helpers.map(helper => ({...helper, sourceSpan:span(helperDeclarations.get(helper.name))})),
        ...(moduleMath ? {helperMathBinding:'live-module-lexical-environment'} : {}),
      } : {}),
    });
    item.wasmBytes=item.variants.reduce((sum,v) => sum+v.wasmBytes,0);
  }
  return {report, edits, registrations, runtimeImports, absorbedEdits};
}
