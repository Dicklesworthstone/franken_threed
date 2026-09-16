import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';
import { createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics } from './numeric_dispatch.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';

function integrate(a, v, dt) { for (let i = 0; i < a.length; i++) a[i] += v[i] * dt; }
function kernel(fn, parameterTypes, options = {}) {
  const artifact = compileNumericKernel(fn.toString(), { parameterTypes });
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  return instantiateNumericKernel(artifact.wasm, { fallback: fn, ...options });
}

test('float32 storage uses Number arithmetic and rounds at every store, not every operator', () => {
  function update(a, b, one) { for (let i = 0; i < a.length; i++) {
    a[i] += one;
    b[i] = a[i] - 16777216;
    a[i] = 16777216 + one - 16777216;
  } }
  const run = kernel(update, ['f32[]', 'f64[]', 'f64']);
  const a = new Float32Array([16777216]), b = new Float64Array([9]); run.run(a, b, 1);
  assert.equal(b[0], 0, 'first store must round before subsequent read');
  assert.equal(a[0], 1, 'arithmetic must not round 16777216+1 prematurely');
  assert.equal(run.diagnostics.wasmCalls, 1); assert.equal(run.diagnostics.fallbackCalls, 0);
  assert.equal(run.manifest.version, 2);
});

test('mixed f32/f64 ABI aligns odd-length views and preserves tails over repeated frames', () => {
  const run = kernel(integrate, ['f32[]', 'f64[]', 'f64']);
  const a = new Float32Array([1, 2, 3]), expected = a.slice();
  const v = new Float64Array([1 / 7, -1 / 13, 1 / 19, 123]);
  for (let i = 0; i < 400; i++) { run.run(a, v, 1 / 60); integrate(expected, v, 1 / 60); assert.deepEqual(a, expected); }
  assert.equal(v[3], 123); assert.equal(run.diagnostics.wasmCalls, 400);
  assert.equal(run.diagnostics.copiedBytes, 400 * (3 * 4 * 2 + 3 * 8));
});

test('f32 boundary values, subnormals, overflow and signed zeros match JavaScript', () => {
  function arithmetic(a, b, divisor) { for (let i = 0; i < a.length; i++) {
    const previous = a[i]; a[i] = previous * b[i]; b[i] = a[i] / divisor;
  } }
  const run = kernel(arithmetic, ['f32[]', 'f32[]', 'f64']);
  const samples = [0, -0, NaN, Infinity, -Infinity, 1.401298464324817e-45, -1.401298464324817e-45,
    1.1754943508222875e-38, 3.4028234663852886e38, -3.4028234663852886e38, 1 / 3, 16777216];
  for (const multiplier of [-2, -0, 0, 0.5, 1, 2, Infinity, NaN]) {
    const a = new Float32Array(samples), b = new Float32Array(samples.length).fill(multiplier), aa = a.slice(), bb = b.slice();
    run.run(a, b, 3); arithmetic(aa, bb, 3); assert.deepEqual(a, aa); assert.deepEqual(b, bb);
  }
  assert.equal(run.diagnostics.fallbackCalls, 0);
});

test('conditional float32 stores preserve skipped elements across changing input arrays', () => {
  function update(a, v, limit) { for (let i = 0; i < a.length; i++) {
    if (a[i] > limit) { a[i] = -a[i]; v[i] = v[i] * -1; }
  } }
  const run = kernel(update, ['f32[]', 'f32[]', 'f64']);
  for (let frame = 0; frame < 30; frame++) {
    const a = new Float32Array([frame, -frame, frame / 3]), v = new Float32Array([1, 2, 3]);
    const aa = a.slice(), vv = v.slice(); run.run(a, v, 10); update(aa, vv, 10);
    assert.deepEqual(a, aa); assert.deepEqual(v, vv);
  }
  assert.equal(run.diagnostics.wasmCalls, 30);
});

test('float32 typed-array identity, ownership and scalar coercion guards retain JS', () => {
  const run = kernel(integrate, ['f32[]', 'f32[]', 'f64']);
  let conversions = 0; const dt = { valueOf() { conversions++; return 2; } };
  const a = new Float32Array([1, 2]); run.run(a, new Float32Array([2, 3]), dt);
  assert.deepEqual([...a], [5, 8]); assert.equal(conversions, 2);
  class Derived extends Float32Array {}
  const b = new Derived([1, 2]); run.run(b, new Float32Array([2, 3]), 2); assert.deepEqual([...b], [5, 8]);
  const shared = new Float32Array(new SharedArrayBuffer(8)); run.run(shared, new Float32Array([2, 3]), 2);
  assert.deepEqual([...shared], [4, 6]); assert.equal(run.diagnostics.fallbackCalls, 3);
});

test('mixed-width overlapping prefixes use original sequential JavaScript semantics', () => {
  const run = kernel(integrate, ['f32[]', 'f64[]', 'f64']);
  const buffer = new ArrayBuffer(32), expected = new ArrayBuffer(32);
  new Float64Array(buffer).set([1, 2, 3, 4]); new Uint8Array(expected).set(new Uint8Array(buffer));
  const a = new Float32Array(buffer, 8, 2), v = new Float64Array(buffer, 0, 2);
  run.run(a, v, 2); integrate(new Float32Array(expected, 8, 2), new Float64Array(expected, 0, 2), 2);
  assert.deepEqual(new Uint8Array(buffer), new Uint8Array(expected));
  assert.equal(run.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
});

test('float32 growth recreates scratch views and enforces byte budgets', () => {
  const run = kernel(integrate, ['f32[]', 'f32[]', 'f64'], { maxMemoryBytes: 2 * 65536 });
  for (const count of [0, 3, 9000, 5, 20000]) {
    const a = new Float32Array(count).fill(1), v = new Float32Array(count).fill(2); run.run(a, v, 3);
    assert.ok(a.every(value => value === 7));
  }
  assert.equal(run.diagnostics.memoryBytes, 2 * 65536);
  assert.equal(run.diagnostics.wasmCalls, 4); assert.equal(run.diagnostics.fallbackCalls, 1);
  assert.equal(run.diagnostics.lastGuardFailure, 'KERNEL_MEMORY_LIMIT');
});

test('AOT dispatch switches f32/f64 variants lazily and reuses both native instances', () => {
  const double = compileNumericKernel(integrate.toString(), { parameterTypes: ['f64[]', 'f64[]', 'f64'] });
  const single = compileNumericKernel(integrate.toString(), { parameterTypes: ['f32[]', 'f32[]', 'f64'] });
  const token = createNumericDispatch(integrate, double.wasm, [{ parameterTypes: ['f32[]', 'f32[]', 'f64'], bytes: single.wasm }]);
  assert.equal(numericDispatchDiagnostics(token).initialized, false);
  for (let frame = 0; frame < 60; frame++) {
    const ArrayType = frame % 2 ? Float32Array : Float64Array;
    const a = new ArrayType([1, 2]), v = new ArrayType([2, 3]); dispatchNumericCall(token, integrate, [a, v, 2]);
    assert.deepEqual([...a], [5, 8]);
  }
  const diagnostics = numericDispatchDiagnostics(token);
  assert.deepEqual(diagnostics.variants.map(variant => variant.kernel.wasmCalls), [30, 30]);
  assert.equal(diagnostics.retainedCalls, 0);
});

test('mismatched variant metadata cannot select a binary with the wrong storage ABI', () => {
  const artifact = compileNumericKernel(integrate.toString(), { parameterTypes: ['f64[]', 'f64[]', 'f64'] });
  const token = createNumericDispatch(integrate, artifact.wasm, [{ parameterTypes: ['f32[]', 'f32[]', 'f64'], bytes: artifact.wasm }]);
  const a = new Float32Array([1]), v = new Float32Array([2]); dispatchNumericCall(token, integrate, [a, v, 3]);
  assert.equal(a[0], 7); assert.equal(numericDispatchDiagnostics(token).retainedCalls, 1);
});

test('automatic source specialization reaches f32 Wasm and conservatively retains mixed arrays', async t => {
  const native = WebAssembly.Instance; let calls = 0;
  t.after(() => { WebAssembly.Instance = native; });
  WebAssembly.Instance = function (...args) {
    const instance = Reflect.construct(native, args);
    return { exports: { memory: instance.exports.memory, run(...xs) { calls++; return instance.exports.run(...xs); } } };
  };
  const source = `${integrate.toString()}\nexport function tick(a, v, dt) { return integrate(a, v, dt); }`;
  const result = specializeNumericModule(source); assert.equal(result.report.candidates[0].variants.length, 2);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-auto-f32-'));
  for (const name of ['numeric_dispatch.mjs', 'numeric_kernel_runtime.mjs']) fs.copyFileSync(new URL(name, import.meta.url), path.join(dir, name));
  fs.writeFileSync(path.join(dir, 'entry.mjs'), result.code);
  const module = await import(pathToFileURL(path.join(dir, 'entry.mjs')));
  const a = new Float32Array([1]), v = new Float32Array([2]); module.tick(a, v, 3); assert.equal(a[0], 7);
  assert.equal(calls, 1);
  module.tick(a, new Float64Array([2]), 3); assert.equal(a[0], 13); assert.equal(calls, 1);
});
