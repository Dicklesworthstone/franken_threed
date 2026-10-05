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

const FUNCTIONS = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']);
const LOOPS = new Set(['ForStatement', 'WhileStatement', 'DoWhileStatement']);
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
function scopeBindings() {
  const cache = new WeakMap();
  return (node, position) => {
    // Parameter initializers do not see body var bindings; switch discriminants
    // do not see case-block lexical bindings. Treating either as a capture could
    // hoist a global getter out of the original loop and change its effects.
    const parametersOnly = FUNCTIONS.has(node.type) && !contains(node.body, position);
    const switchHead = node.type === 'SwitchStatement' && contains(node.discriminant, position);
    const key = parametersOnly ? node.params : switchHead ? node.discriminant : node;
    if (cache.has(key)) return cache.get(key);
    const names = new Set();
    function declaration(statement) {
      const value = statement.declaration ?? statement;
      if (value.type === 'VariableDeclaration') value.declarations.forEach(d => pattern(d.id, names));
      else if (value.type === 'FunctionDeclaration' || value.type === 'ClassDeclaration') pattern(value.id, names);
      else if (value.type === 'ImportDeclaration') value.specifiers.forEach(s => pattern(s.local, names));
    }
    if (node.type === 'Program' || node.type === 'BlockStatement' || node.type === 'StaticBlock')
      node.body.forEach(declaration);
    if (node.type === 'SwitchStatement' && !switchHead) node.cases.forEach(c => c.consequent.forEach(declaration));
    if (node.type === 'CatchClause') pattern(node.param, names);
    if (node.type === 'ForStatement' && node.init?.type === 'VariableDeclaration') declaration(node.init);
    if (['ForInStatement', 'ForOfStatement'].includes(node.type) && node.left.type === 'VariableDeclaration')
      declaration(node.left);
    if (FUNCTIONS.has(node.type)) {
      node.params.forEach(p => pattern(p, names));
      pattern(node.id, names);
    }
    if (node.type === 'ClassDeclaration' || node.type === 'ClassExpression') pattern(node.id, names);
    if (node.type === 'Program' || (FUNCTIONS.has(node.type) && !parametersOnly) || node.type === 'StaticBlock') {
      // var may be declared in a different branch/block of this function. Do
      // not borrow declarations from a nested function or class static block.
      walk.recursive(FUNCTIONS.has(node.type) ? node.body : node, null, {
        Function() {},
        StaticBlock(value, state, next) {
          if (value === node) for (const statement of value.body) next(statement, state);
        },
        VariableDeclaration(value, state, next) {
          if (value.kind === 'var') value.declarations.forEach(d => pattern(d.id, names));
          for (const d of value.declarations) if (d.init) next(d.init, state);
        },
      });
    }
    cache.set(key, names);
    return names;
  };
}

function capturesFor(loop, ancestors, bindings) {
  const locals = new Set(), references = [], arrays = new Set(), assigned = new Set();
  let refusal = null, usesMath = false;
  walk.fullAncestor(loop, (node) => {
    if (node.type === 'VariableDeclaration') {
      if (node.kind === 'var') refusal = 'ISLAND_ESCAPING_DECLARATION';
      for (const d of node.declarations) pattern(d.id, locals);
    }
    if (FUNCTIONS.has(node.type) || node.type === 'ClassDeclaration' || node.type === 'ClassExpression')
      refusal = 'ISLAND_NESTED_CLOSURE';
    if (['ReturnStatement', 'AwaitExpression', 'YieldExpression', 'ThisExpression', 'Super', 'MetaProperty'].includes(node.type) ||
        (['BreakStatement', 'ContinueStatement'].includes(node.type) && node.label))
      refusal = 'ISLAND_ESCAPING_CONTROL';
    if (node.type === 'AssignmentExpression') pattern(node.left, assigned);
    if (node.type === 'UpdateExpression') pattern(node.argument, assigned);
    if (node.type === 'MemberExpression' && node.object.type === 'Identifier') {
      arrays.add(node.object.name);
      if (node.object.name === 'Math') usesMath = true;
    }
    if (node.type === 'Identifier') {
      // Acorn's semantic walker omits ordinary property names and labels. Keep
      // binding identifiers too; the closed compiler resolves their real scopes.
      references.push(node);
    }
  });
  if (refusal) return {refusal};
  if (usesMath && locals.has('Math')) return {refusal:'ISLAND_LOCAL_MATH'};
  const captures = [...new Set(references.sort((a,b) => a.start-b.start).map(n => n.name))]
    .filter(name => !locals.has(name) && !(usesMath && name === 'Math'));
  // A syntactic union of inner declarations can cause conservative refusals
  // under shadowing; it cannot authorize a missing lexical binding, because the
  // numeric compiler independently resolves every reference in the whole loop.
  if (captures.some(name => assigned.has(name))) return {refusal:'ISLAND_OUTER_SCALAR_WRITE'};
  if (captures.length > 64) return {refusal:'ISLAND_CAPTURE_BUDGET'};
  if (captures.some(name => name === 'arguments' || !ancestors.some(node => bindings(node, loop).has(name))))
    return {refusal:'ISLAND_NONLEXICAL_CAPTURE'};
  return {captures, arrays, usesMath};
}

/**
 * Plan disjoint statement replacements in the ORIGINAL parsed source unit.
 * Called after whole-function discovery so its admitted regions keep priority.
 * maxKernels is the remaining shared unit budget. fresh must reserve names
 * against every original identifier token, including property/binding positions.
 */
export function planNumericLoopIslands(source, {
  ast, fresh, sourceName = '<module>', maxKernels = 64, maxMemoryPages = 1024,
  maxIterations = 1000000, excludedSpans = [], reservedEdits = [],
}) {
  const report = {version:1, scope:'closed-loop-statements-in-original-lexical-environment',
    compiledKernels:0, candidates:[], accelerated:false};
  const edits = [], registrations = [], runtimeImports = [], accepted = [];
  const candidates = [], bindings = scopeBindings(), hints = discoverNumericStorageHints(ast);
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    if (LOOPS.has(node.type)) candidates.push({node, ancestors:ancestors.slice(0,-1)});
  });
  // Prefer one whole closed outer loop over many native transitions. If an outer
  // loop has effects, independently closed inner loops may still be compiled.
  candidates.sort((a,b) => a.node.start-b.node.start || b.node.end-a.node.end);
  let createName, dispatchName;
  for (const {node, ancestors} of candidates) {
    if ([...excludedSpans, ...accepted].some(range => contains(range,node))) continue;
    const item = {sourceSpan:span(node), kind:node.type, route:'retained-js', reason:null};
    report.candidates.push(item);
    if (report.compiledKernels >= maxKernels) { item.reason='KERNEL_BUDGET'; continue; }
    if (ancestors.at(-1)?.type === 'LabeledStatement') { item.reason='ISLAND_LABELED_LOOP'; continue; }
    if (reservedEdits.some(edit => contains(node,edit))) { item.reason='ISLAND_EXISTING_CALL_ROUTE'; continue; }
    const closure = capturesFor(node, ancestors, bindings);
    if (closure.refusal) { item.reason=closure.refusal; continue; }
    const {captures, arrays, usesMath} = closure;
    const functionName = fresh('loop_kernel');
    const kernelSource = `function ${functionName}(${captures.join(',')}) {\n${source.slice(node.start,node.end)}\n}`;
    const parameterTypes = captures.map(name => arrays.has(name) ? 'f64[]' : 'f64');
    const compile = types => compileNumericCandidate(kernelSource, {
      parameterTypes:types, allowMath:usesMath, generalControl:true,
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
    const layouts=hints([{arguments:captures.map(name => ({type:'Identifier',name}))}],params);
    for (const [read,write] of [['f32[]','f32[]'],['f64[]','f32[]'],['f32[]','f64[]']])
      layouts.push(params.map(p => p.type==='f64' ? 'f64' : p.write ? write : read));
    const alternatives=[], seen=new Set([parameterTypes.join(',')]);
    for (const types of layouts) {
      if (alternatives.length===16) break;
      const key=types.join(','); if(seen.has(key)) continue; seen.add(key);
      try { alternatives.push({parameterTypes:types,bytes:[...compile(types).wasm]}); }
      catch(error) { if(!(error instanceof NumericKernelCompileError)) throw error; }
    }
    if (!createName) {
      createName=fresh('create_loop'); dispatchName=fresh('dispatch_loop');
      runtimeImports.push(`createNumericLoopDispatch as ${createName}`, `dispatchNumericLoop as ${dispatchName}`);
    }
    const token=fresh('loop_token');
    registrations.push(`var ${token} = ${createName}([${artifact.wasm.join(',')}], ${JSON.stringify(alternatives)});`);
    // An extra block prevents a dangling else from rebinding to our predicate.
    // The original statement (including lexical declarations and comments) is
    // unchanged in the fallback. Capture failures have no source effects.
    const mathResolver=artifact.manifest.mathIntrinsics ? '() => Math' : 'null';
    edits.push({start:node.start,end:node.end,text:
      `{ if (!${dispatchName}(${token}, () => [${captures.join(',')}], ${mathResolver})) {\n${source.slice(node.start,node.end)}\n} }`});
    accepted.push(node); report.compiledKernels++;
    Object.assign(item, {route:'guarded-loop-wasm', captures:params.map(p => ({...p})),
      storageSemantics:'same-type-alias-preserving-v1', guardFallback:'original-loop-in-place',
      controlSemantics:artifact.manifest.controlSemantics, maxIterations:artifact.manifest.maxIterations,
      loopCount:artifact.manifest.loopCount, maxLoopDepth:artifact.manifest.maxLoopDepth,
      variants:[{parameterTypes,wasmBytes:artifact.wasm.length}, ...alternatives.map(v =>
        ({parameterTypes:v.parameterTypes,wasmBytes:v.bytes.length}))],
      ...(artifact.manifest.mathIntrinsics ? {mathIntrinsics:[...artifact.manifest.mathIntrinsics]} : {}),
    });
    item.wasmBytes=item.variants.reduce((sum,v) => sum+v.wasmBytes,0);
  }
  return {report, edits, registrations, runtimeImports};
}
