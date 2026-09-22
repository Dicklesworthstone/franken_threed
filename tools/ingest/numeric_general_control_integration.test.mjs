import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import * as acorn from 'acorn';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { hasNumericLoop } from './numeric_loop_discovery.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';

const dataURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
let sequence = 0;
async function application(source, options = {}, emittedAssets = false) {
  let dispatchURL = new URL('./numeric_dispatch.mjs', import.meta.url).href;
  if (emittedAssets) {
    const runtime = fs.readFileSync(new URL('./numeric_kernel_runtime.mjs', import.meta.url), 'utf8');
    const dispatch = fs.readFileSync(new URL('./numeric_dispatch.mjs', import.meta.url), 'utf8');
    dispatchURL = dataURL(dispatch.replace('"./numeric_kernel_runtime.mjs"', JSON.stringify(dataURL(runtime))));
  }
  const wrapper = dataURL(`
    import { createNumericDispatch as create, dispatchNumericCall, numericDispatchDiagnostics } from ${JSON.stringify(dispatchURL)};
    const tokens = []; // isolated fixture ${sequence++}
    export { dispatchNumericCall };
    export function createNumericDispatch(...args) { const token = create(...args); tokens.push(token); return token; }
    export function diagnostics() { return tokens.map(numericDispatchDiagnostics); }
  `);
  const result = specializeNumericModule(source, { ...options, runtimeModule: wrapper });
  assert.equal(result.changed, true);
  assert.equal(result.report.accelerated, false);
  assert.deepEqual(result, specializeNumericModule(source, { ...options, runtimeModule: wrapper }));
  const app = await import(dataURL(result.code)), original = await import(dataURL(source));
  assert.deepEqual(Object.keys(app), Object.keys(original));
  for (const candidate of result.report.candidates.filter(item => item.route === 'guarded-numeric-wasm')) {
    if (app[candidate.functionName]) assert.equal(app[candidate.functionName].toString(), original[candidate.functionName].toString());
  }
  const { diagnostics } = await import(wrapper);
  assert.ok(diagnostics().every(item => !item.initialized), 'registration must not eagerly instantiate Wasm');
  return { app, original, result, diagnostics };
}
function compare(fn, oracle, args) {
  const expected = args.map(value => ArrayBuffer.isView(value) ? value.slice() : value);
  assert.equal(fn(...args), oracle(...expected));
  assert.deepEqual(args, expected);
}

const rangeSource = `
  export function update(out, input, start, end, step) {
    for (let i = start; i < end; i += step) out[i] += input[i] / 3;
  }
  export function run(...args) { return update(...args); }
  export function shadow(update, ...args) { return update(...args); }
`;

test('automatic candidate routing admits general loops only after existing routes and respects opt-outs', () => {
  const source = 'function count(n){let total=0;while(n>0){total+=n;n--;}return total;}';
  const types = { parameterTypes: ['f64'] };
  const artifact = compileNumericCandidate(source, { ...types, maxIterations: 37 });
  assert.equal(artifact.manifest.version, 8);
  assert.equal(artifact.manifest.maxIterations, 37);
  assert.deepEqual(artifact.wasm, compileNumericKernel(source, { ...types, generalControl: true, maxIterations: 37 }).wasm);
  for (const option of [{ generalControl: false }, { checkedIndexing: false }, { structuredLoops: false }]) {
    assert.throws(() => compileNumericCandidate(source, { ...types, ...option }), { code: 'KERNEL_NOT_CLOSED' });
  }
  for (const option of [{ generalControl: 1 }, { maxIterations: 0 }, { maxIterations: '10' }]) {
    assert.throws(() => compileNumericCandidate(source, { ...types, ...option }), { code: 'INVALID_KERNEL_ABI' });
  }
  const legacy = [
    'function update(out){for(let i=0;i<out.length;i++)out[i]+=1;}',
    'function update(out){for(let i=0;i<out.length;i++){for(let j=0;j<3;j++)out[i]+=j;}}',
    'function update(out,n){for(let i=0;i<out.length;i++){for(let j=0;j<n;j++){if(j===1)continue;out[i]+=j;}}}',
  ];
  for (const source of legacy) {
    const settings = { parameterTypes: source.includes('out,n') ? ['f64[]', 'f64'] : ['f64[]'] };
    const normal = compileNumericCandidate(source, settings);
    assert.ok(normal.manifest.version < 8);
    assert.deepEqual(normal.wasm, compileNumericCandidate(source, { ...settings, generalControl: false }).wasm);
    assert.equal(compileNumericCandidate(source, { ...settings, generalControl: true }).manifest.version, 8);
  }
});

test('ESM range calls execute aliased Float32/64 views without replacing declarations or shadowed callees', async () => {
  const { app, original, diagnostics, result } = await application(rangeSource);
  const candidate = result.report.candidates.find(item => item.functionName === 'update');
  assert.equal(candidate.controlSemantics, 'budgeted-source-order-v1');
  assert.equal(candidate.loopStride, undefined);
  assert.equal(candidate.storageSemantics, 'same-type-alias-preserving-v1');
  for (const ArrayType of [Float32Array, Float64Array]) {
    const data = new ArrayType([1, 2, 3, 4, 5, 6]), expected = data.slice();
    for (const step of [1, 2, 3]) {
      app.run(data.subarray(1), data.subarray(0, 5), 1, 5, step);
      original.run(expected.subarray(1), expected.subarray(0, 5), 1, 5, step);
      assert.deepEqual(data, expected);
    }
    assert.equal(diagnostics()[0].kernel.wasmCalls, 3);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  }
  let calls = 0;
  assert.equal(app.shadow(() => { calls++; return 99; }), 99);
  assert.equal(calls, 1);
  assert.equal(diagnostics()[0].identityMisses, 1);
});

test('ordinary scalar-only do-while and zero-parameter functions execute without invented array bounds', async () => {
  const source = `
    export function root(x, tolerance) {
      if (x === 0) return x;
      let current = x, previous = 0;
      do { previous = current; current = (current + x / current) / 2; }
      while (Math.abs(current - previous) > tolerance);
      return current;
    }
    export function constant() { let total = 0; while (total < 3) total++; return total; }
    export function run(x) { return root(x, 0.000001); }
    export function fixed() { return constant(); }
  `;
  const { app, original, diagnostics, result } = await application(source);
  for (const value of [-0, 1, 2, 7, 100]) compare(app.run, original.run, [value]);
  assert.equal(app.fixed(), 3);
  for (const item of result.report.candidates) {
    assert.deepEqual(item.lengthParameters, []);
    assert.equal(item.variants.length, 1);
  }
  assert.equal(diagnostics()[0].kernel.wasmCalls, 5);
  assert.equal(diagnostics()[1].kernel.wasmCalls, 1);
  assert.ok(diagnostics().every(item => item.kernel.copiedBytes === 0 && item.kernel.fallbackCalls === 0));
});

test('generic control automatically selects integer topology variants for pointer chasing', async () => {
  const source = `
    export function traverse(next, weights, cursor, stop) {
      let total = 0;
      while (cursor !== stop) { total += weights[cursor]; cursor = next[cursor]; }
      return total;
    }
    export function run(...args) { return traverse(...args); }
  `;
  const { app, original, diagnostics, result } = await application(source);
  assert.ok(result.report.candidates[0].variants.length <= 17);
  for (const ArrayType of [Float32Array, Float64Array]) for (const IndexType of [Uint16Array, Uint32Array]) {
    compare(app.run, original.run, [new IndexType([3, 2, 5, 1, 0, 6]), new ArrayType([1, 2, 3, 4, 5, 6]), 4, 6]);
    assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  }
});

test('nested loops under branches are discovered and reported in original module source coordinates', async () => {
  const source = `// Unicode text before the function: π 🦀
export function update(out, n) { if (n > 0) { for (let i=1; i<n; i++) {
  let j=0; while (j<i) { j++; if (j===2) continue; out[i]+=j; }
} } else { do { out[0]+=1; } while (false); }
return out[0]; }
export function run(...args) { return update(...args); }`;
  const { app, original, diagnostics, result } = await application(source);
  const candidate = result.report.candidates[0];
  assert.deepEqual(candidate.loops.map(loop => loop.kind), ['for', 'while', 'do-while']);
  assert.equal(candidate.maxLoopDepth, 2);
  assert.equal(candidate.loopCount, 3);
  for (const loop of candidate.loops) {
    const before = source.slice(0, loop.sourceSpan.start).split('\n');
    assert.equal(loop.sourceSpan.line, before.length);
    assert.equal(loop.sourceSpan.column, before.at(-1).length);
    const text = source.slice(loop.sourceSpan.start, loop.sourceSpan.end);
    assert.ok(text.startsWith(loop.kind === 'do-while' ? 'do ' : loop.kind));
    assert.equal(acorn.parse(text, { ecmaVersion: 'latest' }).body.length, 1);
  }
  for (const n of [0, 1, 5, 8]) compare(app.run, original.run, [new Float64Array(8), n]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 4);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
});

test('all selected precision variants receive the same budget and fall back without partial publication', async () => {
  const { app, original, diagnostics, result } = await application(rangeSource, { maxIterations: 2 });
  assert.equal(result.report.candidates[0].maxIterations, 2);
  for (const ArrayType of [Float32Array, Float64Array]) {
    const data = new ArrayType([1, 2, 3, 4, 5]), expected = data.slice();
    app.run(data.subarray(1), data.subarray(0, 4), 0, 4, 1);
    original.run(expected.subarray(1), expected.subarray(0, 4), 0, 4, 1);
    assert.deepEqual(data, expected);
    assert.equal(diagnostics()[0].kernel.wasmCalls, 0);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 1);
    assert.equal(diagnostics()[0].kernel.lastGuardFailure, 'KERNEL_EXECUTION_FAILED');
    app.run(data.subarray(1), data.subarray(0, 4), 0, 2, 1);
    original.run(expected.subarray(1), expected.subarray(0, 4), 0, 2, 1);
    assert.deepEqual(data, expected);
    assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
    assert.equal(diagnostics()[0].kernel.lastGuardFailure, null);
  }
  for (const maxIterations of [0, 1.5, Infinity, '2']) {
    assert.throws(() => specializeNumericModule(rangeSource, { maxIterations }), RangeError);
  }
});

test('late checked-range failures retain the full original result and original coercion counts', async () => {
  const { app, original, diagnostics } = await application(rangeSource);
  compare(app.run, original.run, [new Float64Array([1, 2, 3]), new Float64Array([4, 5]), 0, 3, 1]);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 1);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 0);
  let conversions = 0;
  const end = { valueOf() { conversions++; return 2; } };
  app.run(new Float64Array(2), new Float64Array(2), 0, end, 1);
  assert.equal(conversions, 3);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 2);
  const failure = {};
  assert.throws(() => app.run(new Float64Array(1), new Float64Array(1), 0,
    { valueOf() { throw failure; } }, 1), error => error === failure);
});

test('general loops preserve live Math resolution and execute no extra replacement calls', async () => {
  const source = `
    let Math = globalThis.Math;
    export function replaceMath(value) { Math = value; }
    export function update(out, n) {
      let i=0;
      while (i<n) { out[i] += Math.abs(out[i]); i++; }
    }
    export function run(...args) { return update(...args); }
  `;
  const { app, diagnostics } = await application(source);
  const data = new Float64Array([-1, -2]);
  app.run(data, 2);
  assert.deepEqual([...data], [0, 0]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
  let calls = 0;
  app.replaceMath({ abs() { calls++; return 7; } });
  app.run(data, 2);
  assert.deepEqual([...data], [7, 7]);
  assert.equal(calls, 2);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 1);
  assert.equal(diagnostics()[0].kernel.lastGuardFailure, 'KERNEL_MATH_BINDING');
});

test('new discovery never licenses captures, side effects, mutable functions, or nested callback scopes', async () => {
  const source = `
    const state = { total: 0 };
    export function captured(n) { while(n>0) { state.total++; n--; } return state.total; }
    export function outer(n) { const callback=()=>{while(n>0)n--;return n;};return callback(); }
    export function mutable(n) { while(n>0)n--;return n; }
    export function change() { mutable = x => x + 1; }
    export function run(n) { return [captured(n), outer(n), mutable(n)]; }
  `;
  const result = specializeNumericModule(source);
  assert.equal(result.changed, false);
  assert.equal(result.code, source);
  assert.equal(result.report.candidates.find(item => item.functionName === 'captured').route, 'retained-js');
  assert.equal(result.report.candidates.find(item => item.functionName === 'mutable').reason, 'MUTABLE_FUNCTION_BINDING');
  assert.ok(!result.report.candidates.some(item => item.functionName === 'outer'));
  const directEval = specializeNumericModule('function f(n){while(n>0)n--;return n;} eval("f(2)"); f(1);');
  assert.equal(directEval.changed, false);
  assert.equal(directEval.report.refusal.code, 'DIRECT_EVAL');
  const ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module' });
  const outer = ast.body.find(node => node.declaration?.id?.name === 'outer').declaration;
  assert.equal(hasNumericLoop(outer.body), false);
});

test('relocated standalone dispatch/runtime assets execute generic loops without compiler imports', async () => {
  const { app, original, diagnostics } = await application(rangeSource, {}, true);
  compare(app.run, original.run, [new Float32Array([1, 2, 3, 4]), new Float32Array([5, 6, 7, 8]), 1, 4, 1]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
});

// Each fixture creates only new paths and intentionally does not remove caller-
// or repository-owned files. Package output's exclusive creation is also tested.
function packageFixture(source, options) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-general-control-'));
  const entry = path.join(directory, 'entry.mjs'), output = path.join(directory, 'package');
  fs.writeFileSync(entry, source, { flag: 'wx' });
  const report = buildNumericKernel(entry, output, options);
  assert.equal(report.kernel.version, 8);
  assert.equal(report.accelerationClaim, false);
  const binary = fs.readFileSync(path.join(output, 'kernel.wasm'));
  assert.equal(report.wasmSha256, createHash('sha256').update(binary).digest('hex'));
  assert.equal(WebAssembly.validate(binary), true);
  assert.equal(fs.readFileSync(path.join(output, 'retained.mjs'), 'utf8'),
    `${source}\nexport default ${report.kernel.functionName};\n`);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(output, 'kernel.json'), 'utf8')).kernel, report.kernel);
  return { directory, entry, output, report };
}

test('standalone package builder discovers a branch-contained loop entry and closes its scalar helpers', async () => {
  const source = `
    function distance(a,b) { return Math.abs(a-b); }
    function midpoint(a,b) { return (a+b)/2; }
    export function converge(out,target) {
      let count=0;
      if (target > 0) {
        while (distance(out[0], target) > 0.001) { out[0] = midpoint(out[0], target); count++; }
      }
      return count;
    }
  `;
  const fixture = packageFixture(source, { parameterTypes: ['f32[]', 'f64'], maxIterations: 30 });
  assert.equal(fixture.report.selectedFunction.name, 'converge');
  assert.deepEqual(fixture.report.selectedFunction.scalarHelpers.map(helper => helper.name), ['distance', 'midpoint']);
  assert.equal(fixture.report.kernel.maxIterations, 30);
  const { createKernel, retained } = await import(pathToFileURL(path.join(fixture.output, 'kernel.mjs')));
  const kernel = createKernel();
  assert.ok(Object.isFrozen(kernel.manifest.lengthParameters));
  compare(kernel.run, retained, [new Float32Array([1]), 7]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
  const moved = path.join(fixture.directory, 'relocated');
  fs.cpSync(fixture.output, moved, { recursive: true, errorOnExist: true, force: false });
  const relocated = await import(pathToFileURL(path.join(moved, 'kernel.mjs')));
  const second = relocated.createKernel();
  compare(second.run, relocated.retained, [new Float32Array([2]), 5]);
  assert.equal(second.diagnostics.wasmCalls, 1);
});

test('single-function package forwards executable budgets and retains whole-call results on exhaustion', async () => {
  const source = 'export function update(out,n){let i=0;while(i<n){out[i]+=1;i++;}return i;}';
  const fixture = packageFixture(source, { parameterTypes: ['f64[]', 'f64'], maxIterations: 2 });
  const { createKernel, retained } = await import(pathToFileURL(path.join(fixture.output, 'kernel.mjs')));
  const kernel = createKernel();
  compare(kernel.run, retained, [new Float64Array([1, 2, 3, 4]), 3]);
  assert.equal(kernel.manifest.maxIterations, 2);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
  compare(kernel.run, retained, [new Float64Array([1, 2, 3, 4]), 2]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('no-Wasm package fallback keeps immutable v8 metadata and zero-argument original execution', async () => {
  const fixture = packageFixture('function constant(){let x=0;do{x++;}while(x<3);return x;}', { parameterTypes: [] });
  const { createKernel } = await import(pathToFileURL(path.join(fixture.output, 'kernel.mjs')));
  const originalWasm = globalThis.WebAssembly;
  let kernel;
  try { globalThis.WebAssembly = undefined; kernel = createKernel(); }
  finally { globalThis.WebAssembly = originalWasm; }
  assert.deepEqual(kernel.manifest.lengthParameters, []);
  assert.ok(Object.isFrozen(kernel.manifest.lengthParameters));
  assert.ok(Object.isFrozen(kernel.manifest.parameters));
  assert.ok(Object.isFrozen(kernel.manifest));
  assert.equal(kernel.run(), 3);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_WASM_UNAVAILABLE');
  kernel.dispose();
  assert.throws(() => kernel.run(), { code: 'KERNEL_DISPOSED' });
});

test('package explicit controls select v8 without unrolling and reject disabled/invalid routes before output', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-general-options-'));
  const entry = path.join(directory, 'entry.mjs');
  fs.writeFileSync(entry, 'function count(n){let s=0;for(let i=0;i<n;i++)s+=i;return s;}', { flag: 'wx' });
  let index = 0;
  for (const options of [
    { generalControl: false }, { checkedIndexing: false }, { structuredLoops: false },
    { maxIterations: 0 }, { generalControl: true, checkedIndexing: false },
  ]) {
    const output = path.join(directory, `invalid-${index++}`);
    assert.throws(() => buildNumericKernel(entry, output, { parameterTypes: ['f64'], ...options }));
    assert.equal(fs.existsSync(output), false);
  }
  const explicit = packageFixture('function f(out){for(let i=0;i<out.length;i++)out[i]+=1;}',
    { parameterTypes: ['f64[]'], generalControl: true, maxIterations: 17 });
  assert.equal(explicit.report.kernel.maxIterations, 17);
  const before = fs.readFileSync(path.join(explicit.output, 'kernel.json'));
  assert.throws(() => buildNumericKernel(explicit.entry, explicit.output, { parameterTypes: ['f64[]'], generalControl: true }));
  assert.deepEqual(fs.readFileSync(path.join(explicit.output, 'kernel.json')), before);
});
