import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics } from './numeric_dispatch.mjs';

const url = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
let sequence = 0;
async function application(source, options = {}, emittedAssets = false) {
  let dispatchUrl = new URL('./numeric_dispatch.mjs', import.meta.url).href;
  if (emittedAssets) {
    // Load the actual runtime as a standalone emitted asset, without local
    // relative imports. Only relocate dispatch's existing runtime import.
    const runtimeSource = await readFile(new URL('./numeric_kernel_runtime.mjs', import.meta.url), 'utf8');
    const dispatchSource = await readFile(new URL('./numeric_dispatch.mjs', import.meta.url), 'utf8');
    dispatchUrl = url(dispatchSource.replace('"./numeric_kernel_runtime.mjs"', JSON.stringify(url(runtimeSource))));
  }
  const wrapper = url(`
    import { createNumericDispatch as create, dispatchNumericCall, numericDispatchDiagnostics } from ${JSON.stringify(dispatchUrl)};
    const tokens = []; // isolate fixture ${sequence++}
    export { dispatchNumericCall };
    export function createNumericDispatch(...args) { const token = create(...args); tokens.push(token); return token; }
    export function diagnostics() { return tokens.map(numericDispatchDiagnostics); }
  `);
  const result = specializeNumericModule(source, { ...options, runtimeModule: wrapper });
  assert.equal(result.changed, true);
  assert.equal(result.report.accelerated, false);
  assert.deepEqual(result, specializeNumericModule(source, { ...options, runtimeModule: wrapper }));
  const original = await import(url(source)), app = await import(url(result.code));
  assert.deepEqual(Object.keys(app), Object.keys(original));
  for (const candidate of result.report.candidates.filter(item => item.route === 'guarded-numeric-wasm')) {
    assert.equal(candidate.storageSemantics, 'same-type-alias-preserving-v1');
    if (typeof app[candidate.functionName] === 'function') {
      assert.equal(app[candidate.functionName].toString(), original[candidate.functionName].toString());
    }
  }
  return { app, original, result, diagnostics: (await import(wrapper)).diagnostics };
}

const shiftedSource = `
  export function update(out, input) {
    for (let i = 0; i < out.length; i++) out[i] += input[i];
  }
  export function run(...args) { return update(...args); }
  export function shadow(update, ...args) { return update(...args); }
`;

test('dispatch opts in all variants without changing the strict low-level default', () => {
  function update(out, input) { for (let i = 0; i < out.length; i++) out[i] += input[i]; }
  const source = update.toString();
  const primary = compileNumericCandidate(source, { parameterTypes: ['f64[]', 'f64[]'] });
  const alternative = compileNumericCandidate(source, { parameterTypes: ['f32[]', 'f32[]'] });
  const variants = [{ parameterTypes: ['f32[]', 'f32[]'], bytes: [...alternative.wasm] }];
  for (const preserveAliasing of [false, true]) {
    const token = preserveAliasing ? createNumericDispatch(update, primary.wasm, variants, null, true)
      : createNumericDispatch(update, primary.wasm, variants);
    for (const ArrayType of [Float32Array, Float64Array]) {
      const data = new ArrayType([1, 2, 3, 4]);
      dispatchNumericCall(token, update, [data.subarray(1), data.subarray(0, 3)]);
      assert.deepEqual([...data], [1, 3, 6, 10]);
      const state = numericDispatchDiagnostics(token);
      assert.equal(state.kernel.wasmCalls, Number(preserveAliasing));
      assert.equal(state.kernel.fallbackCalls, Number(!preserveAliasing));
      assert.equal(state.kernel.lastGuardFailure, preserveAliasing ? null : 'KERNEL_ARRAY_ALIAS');
    }
  }
  for (const flag of [null, 1, 'true', {}]) {
    assert.throws(() => createNumericDispatch(update, primary.wasm, variants, null, flag), TypeError);
  }
});

test('ordinary Float32/64 ESM calls execute shifted and identical aliases natively', async () => {
  const { app, original, diagnostics } = await application(shiftedSource);
  for (const ArrayType of [Float32Array, Float64Array]) {
    const data = new ArrayType([1, 2, 3, 4]), expected = data.slice();
    for (let frame = 0; frame < 4; frame++) {
      original.run(expected.subarray(1), expected.subarray(0, 3));
      app.run(data.subarray(1), data.subarray(0, 3));
      assert.deepEqual(data, expected);
    }
    original.run(expected, expected); app.run(data, data);
    assert.deepEqual(data, expected);
    assert.equal(diagnostics()[0].kernel.wasmCalls, 5);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  }
  let calls = 0;
  const data = new Float64Array([1, 2]);
  assert.equal(app.shadow(out => { calls++; out[0] = 99; return 7; }, data, data), 7);
  assert.equal(calls, 1); assert.equal(data[0], 99);
  assert.equal(diagnostics()[0].identityMisses, 1);
});

test('standalone emitted runtime and dispatch assets retain native alias execution', async () => {
  const { app, diagnostics } = await application(shiftedSource, {}, true);
  const data = new Float64Array([1, 2, 3, 4]);
  app.run(data.subarray(1), data.subarray(0, 3));
  assert.deepEqual([...data], [1, 3, 6, 10]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
});

test('sparse AOT variants preserve colliding aliases, early exits, and whole-call rollback', async () => {
  const source = `
    export function scatter(out, input, indices, limit) {
      for (let i = 0; i < indices.length; i++) {
        out[indices[i]] += input[i];
        if (out[indices[i]] > limit) return i;
      }
      return -1;
    }
    export function run(...args) { return scatter(...args); }
  `;
  const { app, original, diagnostics } = await application(source);
  for (const ArrayType of [Float32Array, Float64Array]) for (const IndexType of [Uint16Array, Uint32Array]) {
    for (const limit of [0, 5, 100]) {
      const data = new ArrayType([1, 2, 3, 4]), expected = data.slice();
      const indices = new IndexType([1, 1, 2]);
      assert.equal(app.run(data.subarray(1), data.subarray(0, 3), indices, limit),
        original.run(expected.subarray(1), expected.subarray(0, 3), indices, limit));
      assert.deepEqual(data, expected);
    }
    assert.equal(diagnostics()[0].kernel.wasmCalls, 3);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
    const data = new ArrayType([1, 2, 3, 4]), expected = data.slice();
    const invalid = new IndexType([1, 1, 100]);
    assert.equal(app.run(data.subarray(1), data.subarray(0, 3), invalid, 100),
      original.run(expected.subarray(1), expected.subarray(0, 3), invalid, 100));
    assert.deepEqual(data, expected);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 1);
    assert.equal(diagnostics()[0].kernel.lastGuardFailure, 'KERNEL_EXECUTION_FAILED');
  }
});

test('mixed-type alias alternatives retain JS while disjoint mixed-type views still execute', async () => {
  const source = `
    export function update(out, input, indices) {
      for (let i = 0; i < indices.length; i++) out[indices[i]] += input[i];
    }
    export function run(...args) { return update(...args); }
  `;
  const { app, original, diagnostics } = await application(source);
  const indices = new Uint16Array([0, 1, 2]);
  const buffer = new ArrayBuffer(32), data = new Float64Array(buffer); data.set([1, 2, 3, 4]);
  const expected = buffer.slice(0);
  app.run(new Float32Array(buffer, 0, 3), data, indices);
  original.run(new Float32Array(expected, 0, 3), new Float64Array(expected), indices);
  assert.deepEqual(new Uint8Array(buffer), new Uint8Array(expected));
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 1);
  assert.equal(diagnostics()[0].kernel.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
  const out = new Float32Array([1, 2, 3]);
  app.run(out, new Float64Array([4, 5, 6]), indices);
  assert.deepEqual([...out], [5, 7, 9]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
});

test('ordinary in-place calls fit the configured memory cap after union packing', async () => {
  const { app, diagnostics } = await application(shiftedSource, { maxMemoryPages: 1 });
  const data = new Float64Array(6000).fill(1);
  app.run(data, data);
  assert.ok(data.every(value => value === 2));
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
  assert.equal(diagnostics()[0].kernel.memoryBytes, 65536);
  assert.equal(diagnostics()[0].kernel.copiedBytes, 96000);
});

test('aliased Math kernels preserve live lexical resolution and fallback without speculative publication', async () => {
  const source = `
    let Math = globalThis.Math;
    export function replaceMath(value) { Math = value; }
    export function update(out, input) {
      for (let i = 0; i < out.length; i++) out[i] += Math.abs(input[i]);
    }
    export function run(...args) { return update(...args); }
  `;
  const { app, diagnostics } = await application(source);
  const data = new Float64Array([-1, -2, -3, -4]);
  app.run(data.subarray(1), data.subarray(0, 3));
  assert.deepEqual([...data], [-1, -1, -2, -2]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
  const original = data.slice(); let calls = 0;
  app.replaceMath({ abs(value) { if (!calls) assert.deepEqual(data, original); calls++; return 7 + value; } });
  app.run(data.subarray(1), data.subarray(0, 3));
  const expected = original.slice();
  for (let i = 1; i < expected.length; i++) expected[i] += 7 + expected[i - 1];
  assert.deepEqual(data, expected); assert.equal(calls, 3);
  assert.equal(diagnostics()[0].kernel.lastGuardFailure, 'KERNEL_MATH_BINDING');
  assert.equal(diagnostics()[0].kernel.fallbackCalls, 1);
});

test('dynamic nested alias updates retain source ordering across bound mutations and returns', async () => {
  const source = `
    export function update(out, bounds, threshold) {
      let total = 0;
      for (let i = 0; i < out.length; i++) {
        for (let j = 0; j < bounds[i]; j++) {
          if (j === 1) continue;
          out[i] -= 1; total++;
          if (total >= threshold) return total;
        }
      }
      return total;
    }
    export function run(...args) { return update(...args); }
  `;
  const { app, original, diagnostics } = await application(source);
  for (const ArrayType of [Float32Array, Float64Array]) for (const threshold of [0, 4, 100]) {
    const data = new ArrayType([4, 7, 2, 10]), expected = data.slice();
    assert.equal(app.run(data, data, threshold), original.run(expected, expected, threshold));
    assert.deepEqual(data, expected);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  }
});

test('fixed-unrolled alias kernels retain each rounded store and subsequent uniform read', async () => {
  const source = `
    export function update(out, uniform) {
      for (let i = 0; i < out.length; i++) {
        for (let j = 0; j < 3; j++) out[i] += uniform[0] / 3;
      }
    }
    export function run(...args) { return update(...args); }
  `;
  const { app, original, diagnostics } = await application(source);
  for (const ArrayType of [Float32Array, Float64Array]) {
    const data = new ArrayType([1 / 3, 2, 3, 4]), expected = data.slice();
    app.run(data, data.subarray(0, 1)); original.run(expected, expected.subarray(0, 1));
    assert.deepEqual(data, expected);
    assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
    assert.equal(diagnostics()[0].kernel.fallbackCalls, 0);
  }
});
