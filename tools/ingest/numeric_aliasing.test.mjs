import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const bytes = buffer => new Uint8Array(buffer);
const type = ArrayType => ArrayType === Float32Array ? 'f32[]' : 'f64[]';
function artifact(fn, parameterTypes, options = {}) {
  const source = fn.toString();
  const settings = { parameterTypes, ...options };
  const result = compileNumericKernel(source, settings);
  assert.equal(WebAssembly.validate(result.wasm), true);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(result.wasm)), []);
  assert.deepEqual(compileNumericKernel(source, settings).wasm, result.wasm);
  return result;
}
const engine = (compiled, options = {}) => instantiateNumericKernel(compiled.wasm, { preserveAliasing: true, ...options });

function compare(fn, kernel, storage, views, scalars = []) {
  const expected = storage.slice(0);
  const actualArgs = views.map(([ArrayType, offset, length]) => new ArrayType(storage, offset, length));
  const expectedArgs = views.map(([ArrayType, offset, length]) => new ArrayType(expected, offset, length));
  assert.equal(kernel.run(...actualArgs, ...scalars), fn(...expectedArgs, ...scalars));
  assert.deepEqual(bytes(storage), bytes(expected));
  return actualArgs;
}

for (const ArrayType of [Float32Array, Float64Array]) {
  const width = ArrayType.BYTES_PER_ELEMENT, types = [type(ArrayType), type(ArrayType)];
  test(`${ArrayType.name}: identical views preserve each source-ordered store and copy storage once`, () => {
    function update(a, b) {
      for (let i = 0; i < a.length; i++) {
        a[i] += b[i] / 3;
        b[i] += a[i] / 7;
      }
    }
    const kernel = engine(artifact(update, types));
    const data = new ArrayType([1 / 3, -0, 16777216, -3, Infinity]);
    compare(update, kernel, data.buffer, [[ArrayType, 0, 5], [ArrayType, 0, 5]]);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
    assert.equal(kernel.diagnostics.copiedBytes, data.byteLength * 2);
    const expected = data.slice(); update(expected, expected);
    kernel.run(data, data); assert.deepEqual(data, expected);
    assert.equal(kernel.diagnostics.copiedBytes, data.byteLength * 4);
  });

  test(`${ArrayType.name}: shifted overlapping views retain both directions of loop-carried dependence`, () => {
    function shift(out, input) {
      for (let i = 0; i < out.length; i++) out[i] += input[i];
    }
    const kernel = engine(artifact(shift, types));
    for (const [start, end] of [[1, 0], [0, 1], [2, 0], [0, 2]]) {
      const data = new ArrayType([1, 2, 3, 4, 5, 6, 7]);
      compare(shift, kernel, data.buffer, [[ArrayType, start * width, 5], [ArrayType, end * width, 5]]);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 4);
  });

  test(`${ArrayType.name}: transitive aliases share one region regardless of parameter order`, () => {
    function update(a, b, c) {
      for (let i = 0; i < a.length; i++) {
        a[i] += c[i] / 7;
        b[i] = a[i] + 1 / 3;
        c[i] -= b[i];
      }
    }
    const kernel = engine(artifact(update, [...types, type(ArrayType)]));
    for (const offsets of [[0, 2, 4], [4, 0, 2], [2, 4, 0], [0, 4, 2], [4, 2, 0], [2, 0, 4]]) {
      const data = ArrayType.from({ length: 9 }, (_, i) => i / 3 + 1);
      compare(update, kernel, data.buffer, offsets.map(offset => [ArrayType, offset * width, 3]));
    }
    assert.equal(kernel.diagnostics.wasmCalls, 6);
  });

  test(`${ArrayType.name}: strided overlapping records preserve untouched channels`, () => {
    function update(out, input) {
      for (let i = 0; i < out.length; i += 3) {
        out[i] = input[i] / 3;
        out[i + 2] += input[i + 1];
      }
    }
    const compiled = artifact(update, types), kernel = engine(compiled);
    assert.equal(compiled.manifest.version, 3);
    for (const offset of [0, 1, 2, 3]) {
      const data = ArrayType.from({ length: 16 }, (_, i) => i + 1);
      compare(update, kernel, data.buffer, [[ArrayType, offset * width, 12], [ArrayType, 0, 12]]);
    }
  });

  test(`${ArrayType.name}: aliased fixed-index uniforms are read at their original program points`, () => {
    function update(out, uniform) {
      for (let i = 0; i < out.length; i++) out[i] += uniform[0];
    }
    const compiled = artifact(update, types), kernel = engine(compiled);
    assert.equal(compiled.manifest.version, 4);
    const data = new ArrayType([1, 2, 3, 4, 5]);
    compare(update, kernel, data.buffer, [[ArrayType, 0, 5], [ArrayType, 0, 1]]);
    assert.deepEqual([...data], [2, 4, 5, 6, 7]);
    assert.equal(kernel.diagnostics.copiedBytes, data.byteLength * 2);
  });

  test(`${ArrayType.name}: ordered reductions observe aliases and return the correct scalar`, () => {
    function update(a, b) {
      let total = 0;
      for (let i = 0; i < a.length; i++) { a[i] += b[i]; total += b[i]; }
      return total;
    }
    const compiled = artifact(update, types), kernel = engine(compiled);
    assert.equal(compiled.manifest.version, 5);
    const data = new ArrayType([1, 2, 3, 4]);
    compare(update, kernel, data.buffer, [[ArrayType, width, 3], [ArrayType, 0, 3]]);
  });

  test(`${ArrayType.name}: an ordered multi-pass transaction retains cross-parameter and cross-pass aliases`, () => {
    function pipeline(a, b) {
      let total = 0;
      for (let i = 0; i < a.length; i++) a[i] += b[i];
      for (let j = 0; j < b.length; j++) { b[j] = b[j] / 3; total += b[j]; }
      return total;
    }
    const compiled = artifact(pipeline, types), kernel = engine(compiled);
    assert.equal(compiled.manifest.version, 6);
    const data = new ArrayType([1, 2, 3, 4, 5, 6]);
    compare(pipeline, kernel, data.buffer, [[ArrayType, 2 * width, 4], [ArrayType, 0, 6]]);
    assert.equal(kernel.diagnostics.copiedBytes, data.byteLength * 2);
  });
}

test('checked gathers/scatters see earlier stores through an aliased parameter', () => {
  function scatter(out, input, indices) {
    for (let i = 0; i < indices.length; i++) out[indices[i]] += input[i];
  }
  const compiled = artifact(scatter, ['f32[]', 'f32[]', 'u16[]'], { checkedIndexing: true });
  const kernel = engine(compiled), data = new Float32Array([1, 2, 3, 4, 5]);
  const expected = data.slice(), indices = new Uint16Array([1, 1, 2, 0]);
  scatter(expected.subarray(1), expected.subarray(0, 4), indices);
  kernel.run(data.subarray(1), data.subarray(0, 4), indices);
  assert.deepEqual(data, expected); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('nested bounds read through aliases are reevaluated and early returns retain only original effects', () => {
  function update(out, limits, threshold) {
    let total = 0;
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < limits[i]; j++) {
        if (j === 1) continue;
        out[i] -= 1; total += 1;
        if (total >= threshold) return total;
      }
    }
    return total;
  }
  const kernel = engine(artifact(update, ['f64[]', 'f64[]', 'f64'], { checkedIndexing: true, structuredLoops: true }));
  for (const threshold of [0, 3, 100]) {
    const data = new Float64Array([6, 4, 8, 10]);
    compare(update, kernel, data.buffer, [[Float64Array, 0, 4], [Float64Array, 0, 4]], [threshold]);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 3);
});

test('late checked access failure publishes nothing before exactly one alias-aware fallback', () => {
  function update(out, input, indices) {
    for (let i = 0; i < indices.length; i++) out[indices[i]] += input[i];
  }
  const compiled = artifact(update, ['f64[]', 'f64[]', 'u32[]'], { checkedIndexing: true });
  const data = new Float64Array([1, 2, 3, 4]), original = data.slice(), expected = data.slice();
  const indices = new Uint32Array([0, 1, 100]), receiver = {}; let calls = 0;
  const kernel = engine(compiled, { fallback(...args) {
    calls++; assert.equal(this, receiver); assert.deepEqual(data, original);
    return update(...args);
  } });
  kernel.run.call(receiver, data.subarray(1), data.subarray(0, 3), indices);
  update(expected.subarray(1), expected.subarray(0, 3), indices);
  assert.deepEqual(data, expected); assert.equal(calls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0); assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_EXECUTION_FAILED');
  const strict = engine(compiled), untouched = original.slice();
  assert.throws(() => strict.run(untouched.subarray(1), untouched.subarray(0, 3), indices), { code: 'KERNEL_EXECUTION_FAILED' });
  assert.deepEqual(untouched, original);
});

test('nested work-budget failure rolls back every alias before fallback, and the next call can succeed', () => {
  function update(out, input, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j++) out[i] += input[i];
    }
  }
  const compiled = artifact(update, ['f64[]', 'f64[]', 'f64'],
    { checkedIndexing: true, structuredLoops: true, maxNestedIterations: 6 });
  const data = new Float64Array([1, 2, 3, 4]), original = data.slice(); let calls = 0;
  const kernel = engine(compiled, { fallback(...args) { calls++; assert.deepEqual(data, original); return update(...args); } });
  const views = [[Float64Array, 8, 3], [Float64Array, 0, 3]];
  compare(update, kernel, data.buffer, views, [3]); assert.equal(calls, 1);
  compare(update, kernel, data.buffer, views, [2]);
  assert.equal(kernel.diagnostics.wasmCalls, 1); assert.equal(kernel.diagnostics.fallbackCalls, 1);
});

test('memory limits charge unique storage instead of duplicate parameter copies', () => {
  function update(a, b, c) {
    for (let i = 0; i < a.length; i++) { a[i] += b[i]; c[i] += a[i]; }
  }
  const compiled = artifact(update, ['f64[]', 'f64[]', 'f64[]']);
  const kernel = engine(compiled, { maxMemoryBytes: 65536 });
  const data = new Float64Array(5000).fill(1);
  kernel.run(data, data, data); assert.ok(data.every(value => value === 4));
  assert.equal(kernel.diagnostics.memoryBytes, 65536);
  assert.equal(kernel.diagnostics.copiedBytes, 80000);
  const separate = new Float64Array(5000).fill(3);
  assert.throws(() => kernel.run(data, data, separate), { code: 'KERNEL_MEMORY_LIMIT' });
  assert.ok(data.every(value => value === 4)); assert.ok(separate.every(value => value === 3));
});

test('distant views of a large buffer do not allocate or copy the gap', () => {
  function update(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }
  const compiled = artifact(update, ['f64[]', 'f64[]']);
  const kernel = engine(compiled, { maxMemoryBytes: 65536 });
  const data = new Float64Array(131072); data.set([1, 2, 3]); data.set([4, 5, 6], 131069);
  compare(update, kernel, data.buffer, [[Float64Array, 0, 3], [Float64Array, 131069 * 8, 3]]);
  assert.equal(kernel.diagnostics.memoryBytes, 65536); assert.equal(kernel.diagnostics.copiedBytes, 72);
});

test('one-page Float32 views with a four-byte backing offset do not pay unnecessary alignment padding', () => {
  function update(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }
  const kernel = engine(artifact(update, ['f32[]', 'f32[]']), { maxMemoryBytes: 65536 });
  const storage = new ArrayBuffer(65540), data = new Float32Array(storage, 4, 16384); data.fill(1);
  kernel.run(data, data); assert.ok(data.every(value => value === 2));
  assert.equal(kernel.diagnostics.memoryBytes, 65536);
});

test('scratch pointers and copy views are rebuilt after growth and changing alias relationships', () => {
  function update(a, b) { for (let i = 0; i < a.length; i++) { a[i] += b[i]; b[i] *= 2; } }
  const kernel = engine(artifact(update, ['f64[]', 'f64[]']));
  for (const length of [3, 10000, 0, 1, 30000, 8]) for (const shift of [0, 1, length + 1]) {
    const data = Float64Array.from({ length: 2 * length + 3 }, (_, i) => i / 3);
    compare(update, kernel, data.buffer, [[Float64Array, 0, length], [Float64Array, shift * 8, length]]);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 18);
});

test('untouched bytes including NaN payloads survive skipped stores and overlapping output views', () => {
  function update(a, b, stop) {
    for (let i = 0; i < a.length; i++) { if (i >= stop) break; a[i] += b[i]; }
  }
  const kernel = engine(artifact(update, ['f64[]', 'f64[]', 'f64'], { checkedIndexing: true, structuredLoops: true }));
  const storage = new ArrayBuffer(48), raw = new DataView(storage);
  raw.setFloat64(0, 1, true); raw.setFloat64(8, 2, true); raw.setFloat64(16, 3, true);
  raw.setBigUint64(24, 0x7ff0000000000001n, true);
  raw.setBigUint64(32, 0x8000000000000000n, true);
  raw.setBigUint64(40, 0x7ff800000000abcdn, true);
  compare(update, kernel, storage, [[Float64Array, 8, 5], [Float64Array, 0, 5]], [1]);
});

test('mixed-type writable overlaps retain the original once without speculative writes', () => {
  function update(out, input) { for (let i = 0; i < out.length; i++) out[i] += input[i]; }
  const compiled = artifact(update, ['f32[]', 'f64[]'], { checkedIndexing: true });
  const storage = new ArrayBuffer(32); new Float64Array(storage).set([1, 2, 3, 4]);
  const initial = storage.slice(0); let calls = 0;
  const kernel = engine(compiled, { fallback(...args) { calls++; assert.deepEqual(bytes(storage), bytes(initial)); return update(...args); } });
  compare(update, kernel, storage, [[Float32Array, 0, 4], [Float64Array, 0, 4]]);
  assert.equal(calls, 1); assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('mixed read-only overlaps retain proper alignment and numeric interpretation', () => {
  function total(a, b, c) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] + b[i] + c[i];
    return sum;
  }
  const kernel = engine(artifact(total, ['u16[]', 'u32[]', 'f64[]'], { checkedIndexing: true }));
  const storage = new ArrayBuffer(32); new Float64Array(storage).set([1, 2, 3, 4]);
  compare(total, kernel, storage, [[Uint16Array, 2, 2], [Uint32Array, 4, 2], [Float64Array, 8, 2]]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('strict default and invalid options retain their public contract', () => {
  function update(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }
  const compiled = artifact(update, ['f64[]', 'f64[]']);
  for (const options of [{}, { preserveAliasing: false }]) {
    const kernel = instantiateNumericKernel(compiled.wasm, options), data = new Float64Array([1, 2]);
    assert.throws(() => kernel.run(data, data), { code: 'KERNEL_ARRAY_ALIAS' });
    assert.deepEqual(data, new Float64Array([1, 2]));
  }
  for (const preserveAliasing of [null, 1, 0, 'true', {}, []]) {
    assert.throws(() => engine(compiled, { preserveAliasing }), TypeError);
  }
});

test('shared, resizable, detached and proxy-backed arrays never enter shared scratch storage', () => {
  function update(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }
  const kernel = engine(artifact(update, ['f64[]', 'f64[]']));
  const detached = new Float64Array(2); structuredClone(detached.buffer, { transfer: [detached.buffer] });
  let traps = 0;
  const proxy = new Proxy(new Float64Array(2), { get() { traps++; throw Error(); }, getPrototypeOf() { traps++; throw Error(); } });
  for (const bad of [new Float64Array(new SharedArrayBuffer(16)),
    new Float64Array(new ArrayBuffer(16, { maxByteLength: 32 })), detached, proxy]) {
    const good = new Float64Array([1, 2]);
    assert.throws(() => kernel.run(good, bad)); assert.deepEqual(good, new Float64Array([1, 2]));
  }
  assert.equal(traps, 0); assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('read-only bridges share scratch storage without becoming published outputs', () => {
  function update(a, bridge, c, d) {
    for (let i = 0; i < a.length; i++) {
      a[i] += bridge[i]; c[i] += bridge[i + 6];
      if (i === 0) d[0] += 1;
    }
  }
  const kernel = engine(artifact(update, ['f64[]', 'f64[]', 'f64[]', 'f64[]'], { checkedIndexing: true }));
  const data = Float64Array.from({ length: 12 }, (_, i) => i + 1);
  compare(update, kernel, data.buffer, [[Float64Array, 8, 2], [Float64Array, 16, 8],
    [Float64Array, 64, 2], [Float64Array, 16, 1]]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
  // One 72-byte input union; only the two 16-byte declared output intervals.
  assert.equal(kernel.diagnostics.copiedBytes, 72 + 16 + 16);
});
