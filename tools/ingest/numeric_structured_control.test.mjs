import test from 'node:test';
import assert from 'node:assert/strict';
import {compileNumericKernel, NumericKernelCompileError} from './numeric_kernel.mjs';
import {compileNumericCandidate} from './numeric_candidate.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics} from './numeric_dispatch.mjs';
import {specializeNumericModule} from './numeric_specialization.mjs';

function build(fn, parameterTypes, options = {}) {
  const source = typeof fn === 'string' ? fn : fn.toString();
  const settings = {parameterTypes, checkedIndexing: true, structuredLoops: true, ...options};
  const artifact = compileNumericKernel(source, settings);
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
  assert.deepEqual(artifact.wasm, compileNumericKernel(source, settings).wasm);
  return artifact;
}
const instance = artifact => instantiateNumericKernel(artifact.wasm);

test('root continue executes the increment and preserves skipped caller values across calls', () => {
  function update(out, mask) {
    for (let i = 0; i < out.length; i++) {
      if (!mask[i]) continue;
      out[i] = i + 10;
    }
  }
  for (const ArrayType of [Float32Array, Float64Array]) {
    const type = ArrayType === Float32Array ? 'f32[]' : 'f64[]';
    const artifact = build(update, [type, type]), kernel = instance(artifact);
    assert.equal(artifact.manifest.parameters[0].read, true);
    const actual = new ArrayType([91, 92, 93, 94]), mask = new ArrayType([1, 1, 1, 1]);
    kernel.run(actual, mask);
    actual.set([7, -0, 9, 11]); mask.set([0, 1, 0, 1]);
    const expected = actual.slice(); update(expected, mask); kernel.run(actual, mask);
    assert.deepEqual(actual, expected); assert.equal(kernel.diagnostics.wasmCalls, 2);
  }
});

test('root break avoids invalid later reads but continues the next ordered pass', () => {
  function update(out, input, stop) {
    for (let i = 0; i < out.length; i++) {
      if (i >= stop) break;
      out[i] += input[i];
    }
    for (let j = 0; j < out.length; j++) out[j] *= 2;
  }
  const kernel = instance(build(update, ['f64[]', 'f64[]', 'f64']));
  for (const stop of [0, 1, 2]) {
    const actual = new Float64Array([1, 2, 3, 4]), expected = actual.slice();
    const input = new Float64Array([10, 20]);
    update(expected, input, stop); kernel.run(actual, input, stop); assert.deepEqual(actual, expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 3); assert.equal(kernel.diagnostics.fallbackCalls, 0);
});

test('nested continue and break target the nearest loop through if/else and lexical blocks', () => {
  function update(out, n) {
    for (let i = 0; i < out.length; i++) {
      if (i & 1) continue;
      for (let j = n; j >= 0; j--) {
        if (j > 3) { if (j & 1) continue; else { continue; } }
        else { if (j === 1) break; }
        out[i] += 10 * i + j;
      }
      out[i] += 100;
      if (i > 4) break;
    }
  }
  const kernel = instance(build(update, ['f64[]', 'f64']));
  for (const n of [0, 1, 4, 9]) {
    const actual = new Float64Array(9).fill(91), expected = actual.slice();
    update(expected, n); kernel.run(actual, n); assert.deepEqual(actual, expected);
  }
});

test('nested early numeric return exits all passes with exactly the source prefix published', () => {
  function find(out, input, width, threshold) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < width; j++) {
        const value = input[i * width + j];
        if (value < 0) continue;
        out[i] += value;
        if (out[i] >= threshold) return i * width + j;
      }
    }
    for (let i = 0; i < out.length; i++) out[i] = -1;
    return -1;
  }
  const kernel = instance(build(find, ['f64[]', 'f64[]', 'f64', 'f64']));
  const input = new Float64Array([-1, 2, 3, 4, 5, 6, 7, 8, 9]);
  for (const threshold of [0, 4, 9, 99]) {
    const actual = new Float64Array([1, 1, 1]), expected = actual.slice();
    assert.equal(kernel.run(actual, input, 3, threshold), find(expected, input, 3, threshold));
    assert.deepEqual(actual, expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 4);
});

test('early returns do not speculate into invalid later geometry or failed predicates', () => {
  function probe(out, input, stop) {
    for (let i = 0; i < out.length; i++) {
      if (i === stop) return -0;
      out[i] = input[i];
    }
    return 1;
  }
  const kernel = instance(build(probe, ['f64[]', 'f64[]', 'f64']));
  const actual = new Float64Array([91, 92]), input = new Float64Array([7]);
  assert.ok(Object.is(kernel.run(actual, input, 1), -0));
  assert.deepEqual(actual, new Float64Array([7, 92])); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('void return preserves all untouched output elements', () => {
  function update(out, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j++) {
        if (i + j > 3) return;
        out[i] += j;
      }
    }
  }
  const kernel = instance(build(update, ['f32[]', 'f64']));
  const actual = new Float32Array([1, 2, 3, 4, 5]), expected = actual.slice();
  assert.equal(kernel.run(actual, 3), update(expected, 3)); assert.deepEqual(actual, expected);
  assert.equal(kernel.manifest.resultType, 'void'); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('continue cannot bypass nested fuel or skip its numeric update', () => {
  function skip(out, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j += 0.5) { if (j < count) continue; out[i] = 0; }
      out[i] += 1;
    }
  }
  const artifact = build(skip, ['f64[]', 'f64'], {maxNestedIterations: 4});
  const kernel = instance(artifact), actual = new Float64Array([91]);
  kernel.run(actual, 2); assert.equal(actual[0], 92);
  assert.throws(() => kernel.run(actual, 2.5), {code: 'KERNEL_EXECUTION_FAILED'});
  assert.equal(actual[0], 92); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('branch depths beyond one-byte LEB encoding target the correct loop', () => {
  const source = `function f(out,n) {
    for(let i=0;i<out.length;i++) {
      for(let j=0;j<n;j++) {${'if (j === 1) {'.repeat(126)}break;${'}'.repeat(126)} out[i]+=1;}
      out[i]+=10;
    }
  }`;
  const kernel = instance(build(source, ['f64[]', 'f64']));
  const out = new Float64Array([0, 7]); kernel.run(out, 3);
  assert.deepEqual(out, new Float64Array([11, 18]));
});

test('early-result types and labeled transfers remain conservatively rejected', () => {
  for (const source of [
    'function f(out) {for(let i=0;i<out.length;i++) {if(i) return;out[i]=1;}return 3;}',
    'function f(out) {for(let i=0;i<out.length;i++) {if(i) return 2;out[i]=1;}}',
    'function f(out) {for(let i=0;i<out.length;i++) {if(i) return true;out[i]=1;}return 3;}',
    'function f(out) {for(let i=0;i<out.length;i++) {label: {out[i]=1;break label;}}}',
    'function f(out) {for(let i=0;i<out.length;i++) {label: for(let j=0;j<3;j++) continue label;out[i]=1;}}',
  ]) assert.throws(() => build(source, ['f64[]']), NumericKernelCompileError);
  for (const transfer of ['break', 'continue', 'return']) {
    const source = `function f(out) {for(let i=0;i<out.length;i++){if(i) ${transfer};out[i]=1;}}`;
    assert.throws(() => compileNumericKernel(source, {parameterTypes: ['f64[]'], checkedIndexing: true}), NumericKernelCompileError);
  }
});

test('candidate routing retains legacy unrolling but admits dynamic loops without expansion', () => {
  const fixed = 'function f(out) {for(let i=0;i<out.length;i++) for(let j=0;j<3;j++) out[i]+=j;}';
  const old = compileNumericCandidate(fixed, {parameterTypes: ['f64[]']});
  assert.equal(old.fixedLoops.expandedIterations, 3); assert.equal(old.nestedLoops, undefined);
  function dynamic(out, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j++) {if (j & 1) continue; out[i] += j;}
    }
  }
  const artifact = compileNumericCandidate(dynamic.toString(), {parameterTypes: ['f64[]', 'f64']});
  assert.equal(artifact.fixedLoops, undefined); assert.equal(artifact.nestedLoops.count, 1);
  const actual = new Float64Array([1, 2]), expected = actual.slice();
  const token = createNumericDispatch(dynamic, artifact.wasm);
  dynamic(expected, 5); dispatchNumericCall(token, dynamic, [actual, 5]); assert.deepEqual(actual, expected);
  assert.equal(numericDispatchDiagnostics(token).kernel.wasmCalls, 1);
  assert.equal(numericDispatchDiagnostics(token).kernel.fallbackCalls, 0);
  for (const options of [{checkedIndexing: false}, {structuredLoops: false}]) {
    assert.throws(() => compileNumericCandidate(dynamic.toString(), {parameterTypes: ['f64[]', 'f64'], ...options}), NumericKernelCompileError);
  }
  assert.equal(compileNumericCandidate(dynamic.toString(), {parameterTypes: ['f64[]', 'f64'], structuredLoops: true}).nestedLoops.count, 1);
});

test('candidate retry starts from original source when partial fixed-loop expansion failed', () => {
  function dynamic(out, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j++) {
        for (let k = 0; k < 2; k++) out[i] += j + k;
      }
    }
  }
  const artifact = compileNumericCandidate(dynamic.toString(), {parameterTypes: ['f64[]', 'f64']});
  assert.equal(artifact.fixedLoops, undefined); assert.equal(artifact.nestedLoops.count, 2);
  assert.equal(artifact.manifest.sourceName, '<numeric-kernel>');
  const actual = new Float64Array([1, 2]), expected = actual.slice();
  dynamic(expected, 3); instance(artifact).run(actual, 3); assert.deepEqual(actual, expected);
});

test('automatic dispatch retains the whole original on budget failure and callee replacement', () => {
  function update(out, count) {
    for (let i = 0; i < out.length; i++) for (let j = 0; j < count; j++) out[i] += 1;
  }
  const artifact = compileNumericCandidate(update.toString(), {parameterTypes: ['f64[]', 'f64'], maxNestedIterations: 3});
  const token = createNumericDispatch(update, artifact.wasm), actual = new Float64Array([10, 20]);
  dispatchNumericCall(token, update, [actual, 3]); assert.deepEqual(actual, new Float64Array([13, 23]));
  const state = numericDispatchDiagnostics(token);
  assert.equal(state.kernel.fallbackCalls, 1); assert.equal(state.kernel.wasmCalls, 0);
  let replacementCalls = 0;
  const replacement = out => {replacementCalls++; out[0] = 99; return 7;};
  assert.equal(dispatchNumericCall(token, replacement, [actual, 3]), 7);
  assert.equal(actual[0], 99); assert.equal(replacementCalls, 1);
  assert.equal(numericDispatchDiagnostics(token).identityMisses, 1);
});

const moduleUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
let appSequence = 0;
async function application(source) {
  // Observe tokens through the real dispatch host, not a replacement executor.
  const runtime = moduleUrl(`
    import {createNumericDispatch as create, dispatchNumericCall, numericDispatchDiagnostics} from
      ${JSON.stringify(new URL('./numeric_dispatch.mjs', import.meta.url).href)};
    const tokens=[]; // fixture ${appSequence++}
    export {dispatchNumericCall};
    export function createNumericDispatch(...args) {const token=create(...args);tokens.push(token);return token;}
    export function diagnostics() {return tokens.map(numericDispatchDiagnostics);}
  `);
  const result = specializeNumericModule(source, {runtimeModule: runtime, sourceName: 'structured-fixture.mjs'});
  assert.equal(result.changed, true);
  assert.equal(result.report.accelerated, false);
  assert.deepEqual(specializeNumericModule(source, {runtimeModule: runtime, sourceName: 'structured-fixture.mjs'}), result);
  return {result, app: await import(moduleUrl(result.code)), original: await import(moduleUrl(source)),
    diagnostics: (await import(runtime)).diagnostics};
}

test('ordinary ESM call sites execute sparse native loops with inferred integer/float variants', async () => {
  const source = `
    export function neighbors(out, offsets, indices, values, limit) {
      for(let i=0;i<out.length;i++) {
        let sum=0;
        for(let p=offsets[i];p<offsets[i+1];p++) {
          const value=values[indices[p]];
          if(value<0) continue;
          sum+=value;
          if(sum>=limit) break;
        }
        out[i]=sum;
      }
    }
    export function run(...args) {return neighbors(...args);}
    export function shadow(neighbors,...args) {return neighbors(...args);}
  `;
  const {app, original, result, diagnostics} = await application(source);
  assert.equal(result.report.compiledKernels, 1); assert.equal(result.report.rewrittenCalls, 2);
  assert.deepEqual(Object.keys(app), Object.keys(original));
  assert.equal(app.neighbors.toString(), original.neighbors.toString());
  const offsets = new Uint32Array([0, 4, 7]), indices = new Uint32Array([0, 1, 2, 3, 3, 0, 2]);
  for (const ArrayType of [Float32Array, Float64Array]) {
    const values = new ArrayType([1 / 3, -1, 5, 8]);
    const actual = new ArrayType([91, 92]), expected = actual.slice();
    original.run(expected, offsets, indices, values, 4);
    app.run(actual, offsets, indices, values, 4); assert.deepEqual(actual, expected);
    assert.equal(diagnostics()[0].kernel.wasmCalls, 1); assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  }
  const out = new Float64Array([91]); let calls = 0;
  assert.equal(app.shadow(array => {calls++;array[0]=42;return 7;}, out), 7);
  assert.equal(calls, 1); assert.equal(out[0], 42); assert.equal(diagnostics()[0].identityMisses, 1);
});

test('module discovery admits early-exit search but retains effectful dynamic-loop code unchanged', async () => {
  const source = `
    export function find(input, threshold) {
      for(let i=0;i<input.length;i++) {if(input[i]>=threshold) return i;}
      return -1;
    }
    export function run(...args) {return find(...args);}
  `;
  const {app, original, diagnostics} = await application(source);
  for (const threshold of [0, 2, 4, 99, NaN]) {
    const input = new Float64Array([-1, 1, 3, 5]);
    assert.equal(app.run(input, threshold), original.run(input, threshold));
  }
  assert.equal(diagnostics()[0].kernel.wasmCalls, 5);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  const effectful = `
    export function update(out, count, callback) {
      for(let i=0;i<out.length;i++) for(let j=0;j<count;j++) {out[i]+=j;callback();}
    }
    export function run(...args) {return update(...args);}
  `;
  const retained = specializeNumericModule(effectful);
  assert.equal(retained.changed, false); assert.equal(retained.code, effectful);
  assert.equal(retained.report.compiledKernels, 0);
});
