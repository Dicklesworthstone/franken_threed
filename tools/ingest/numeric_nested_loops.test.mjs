import test from 'node:test';
import assert from 'node:assert/strict';
import {compileNumericKernel, NumericKernelCompileError} from './numeric_kernel.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';

const type = ArrayType => ArrayType === Float32Array ? 'f32[]' : 'f64[]';
function compile(fn, parameterTypes, options = {}) {
  const source = typeof fn === 'string' ? fn : fn.toString();
  const settings = {parameterTypes, checkedIndexing: true, structuredLoops: true, ...options};
  const artifact = compileNumericKernel(source, settings);
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
  assert.deepEqual(compileNumericKernel(source, settings).wasm, artifact.wasm);
  assert.equal(artifact.manifest.version, 7);
  return artifact;
}
const host = (artifact, options) => instantiateNumericKernel(artifact.wasm, options);

for (const ArrayType of [Float32Array, Float64Array]) {
  test(`runtime-width matrix/vector reduction preserves ordered stores in ${ArrayType.name}`, () => {
    function product(out, matrix, vector, width) {
      for (let i = 0; i < out.length; i++) {
        out[i] = 0;
        for (let j = 0; j < width; j++) out[i] += matrix[i * width + j] * vector[j];
      }
    }
    const artifact = compile(product, [type(ArrayType), type(ArrayType), type(ArrayType), 'f64']);
    const kernel = host(artifact);
    assert.deepEqual(artifact.nestedLoops, {count: 1, maxDepth: 1, maxIterations: 1000000});
    assert.ok(Object.isFrozen(artifact.nestedLoops));
    for (const rows of [0, 1, 17, 129]) for (const width of [0, 1, 3, 31]) {
      const matrix = ArrayType.from({length: rows * width}, (_, i) => (i % 17 - 8) / 7);
      const vector = ArrayType.from({length: width}, (_, j) => (j % 13 - 6) / 11);
      const actual = new ArrayType(rows).fill(91), expected = actual.slice();
      product(expected, matrix, vector, width); kernel.run(actual, matrix, vector, width);
      assert.deepEqual(actual, expected, `rows=${rows}, width=${width}`);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 16);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });
}

test('CSR adjacency reduction uses dynamic offset bounds and unsigned topology', () => {
  function accumulate(out, offsets, indices, values) {
    for (let i = 0; i < out.length; i++) {
      for (let p = offsets[i]; p < offsets[i + 1]; p++) out[i] += values[indices[p]];
    }
  }
  const kernel = host(compile(accumulate, ['f32[]', 'u32[]', 'u16[]', 'f32[]']));
  const offsets = new Uint32Array([0, 3, 3, 8]);
  const indices = new Uint16Array([0, 1, 0, 2, 1, 2, 0, 2]);
  const values = new Float32Array([1 / 3, 16777216, -16777216]);
  const actual = new Float32Array([1, -0, 3]), expected = actual.slice();
  for (let frame = 0; frame < 8; frame++) {
    accumulate(expected, offsets, indices, values); kernel.run(actual, offsets, indices, values);
    assert.deepEqual(actual, expected);
  }
  assert.ok(Object.is(actual[1], -0));
  assert.equal(kernel.diagnostics.wasmCalls, 8);
});

test('three-level loops reinitialize inner indices and locals for every parent iteration', () => {
  function stencil(out, width, radius) {
    for (let i = 0; i < out.length; i++) {
      let sum = 0;
      for (let y = -radius; y <= radius; y++) {
        for (let x = width; x > 0; x -= 2) {
          const offset = (i + x) * (y + 1);
          sum += offset;
        }
      }
      out[i] = sum;
    }
  }
  const artifact = compile(stencil, ['f64[]', 'f64', 'f64']);
  assert.deepEqual(artifact.nestedLoops, {count: 2, maxDepth: 2, maxIterations: 1000000});
  const kernel = host(artifact);
  for (const width of [0, 1, 3, 4, 7]) for (const radius of [0, 1, 3]) {
    const actual = new Float64Array(10), expected = actual.slice();
    stencil(expected, width, radius); kernel.run(actual, width, radius);
    assert.deepEqual(actual, expected);
  }
});

test('Number loop indices preserve fractional starts, steps, signed zero and comparisons', () => {
  for (const [operator, update, start, limit] of [
    ['<', 'j += 0.25', -0, 1], ['<=', 'j++', -2, 1],
    ['>', '--j', 4.5, 0.5], ['>=', 'j -= 0.5', 1, -0],
    ['<', '++j', NaN, 4], ['>=', 'j--', 3, NaN],
    ['<', 'j++', Infinity, Infinity], ['>', 'j--', -Infinity, -Infinity],
  ]) {
    const source = `function fold(out,start,limit) {
      for(let i=0;i<out.length;i++) { for(let j=start;j ${operator} limit;${update}) out[i] += 1 / j; }
    }`;
    const reference = Function(`return (${source});`)();
    const actual = new Float64Array([0, -0]), expected = actual.slice();
    reference(expected, start, limit);
    const kernel = host(compile(source, ['f64[]', 'f64', 'f64']));
    kernel.run(actual, start, limit); assert.deepEqual(actual, expected);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
  }
});

test('mutable array and scalar bounds are reevaluated rather than hoisted', () => {
  function update(out, limits, limit) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < limits[i]; j++) {
        out[i] += j; limits[i] -= 1;
      }
      for (let k = 0; k < limit; k++) {
        out[i] += k; limit -= 1;
      }
    }
    return limit;
  }
  const kernel = host(compile(update, ['f64[]', 'f64[]', 'f64']));
  const actual = new Float64Array([7, 8, 9]), expected = actual.slice();
  const limits = new Float64Array([8, 6, 10]), expectedLimits = limits.slice();
  assert.equal(kernel.run(actual, limits, 10), update(expected, expectedLimits, 10));
  assert.deepEqual(actual, expected); assert.deepEqual(limits, expectedLimits);
});

test('empty inner loops spend no credit and preserve arrays without reading their bodies', () => {
  function empty(out, missing, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j++) out[i] = missing[j];
    }
  }
  const kernel = host(compile(empty, ['f64[]', 'f64[]', 'f64'], {maxNestedIterations: 1}));
  const actual = new Float64Array(200).fill(-0), expected = actual.slice();
  kernel.run(actual, new Float64Array(), 0); assert.deepEqual(actual, expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('one shared budget covers siblings, nesting, all root iterations and all passes', () => {
  function work(out) {
    for (let i = 0; i < out.length; i++) {
      out[i] += 1;
      for (let j = 0; j < 2; j++) {
        for (let k = 0; k < 3; k++) out[i] += j + k;
      }
      for (let p = 0; p < 2; p++) out[i] += p;
    }
    for (let i = 0; i < out.length; i++) {
      for (let q = 0; q < 2; q++) out[i] += q;
    }
  }
  // Each output costs (2 parent + 6 child + 2 sibling + 2 second-pass) entries.
  for (const budget of [23, 24]) {
    let fallbacks = 0;
    const actual = new Float64Array([91, 92]), initial = actual.slice(), expected = actual.slice();
    const kernel = host(compile(work, ['f64[]'], {maxNestedIterations: budget}), {
      fallback(out) { fallbacks++; assert.deepEqual(out, initial); return work(out); },
    });
    work(expected); kernel.run(actual); assert.deepEqual(actual, expected);
    assert.equal(fallbacks, budget === 23 ? 1 : 0);
    assert.equal(kernel.diagnostics.wasmCalls, budget === 23 ? 0 : 1);
  }
});

test('budget fallback is atomic, preserves the receiver/result, and resets on the next call', () => {
  function count(out, n) {
    let total = 0;
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < n; j++) { out[i] += j + 1; total++; }
    }
    return total;
  }
  const receiver = {}, actual = new Float64Array([10, 20]), initial = actual.slice();
  let calls = 0;
  const kernel = host(compile(count, ['f64[]', 'f64'], {maxNestedIterations: 6}), {
    fallback(out, n) { calls++; assert.equal(this, receiver); assert.deepEqual(out, initial); return count(out, n); },
  });
  assert.equal(kernel.run.call(receiver, actual, 4), 8);
  assert.equal(calls, 1); assert.deepEqual(actual, new Float64Array([20, 30]));
  assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_EXECUTION_FAILED');
  const expected = actual.slice();
  assert.equal(kernel.run(actual, 3), count(expected, 3)); assert.deepEqual(actual, expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1); assert.equal(kernel.diagnostics.lastGuardFailure, null);
});

test('f64 index stagnation and divergent bounds hit the budget without publishing', () => {
  function stalled(out, start, limit) {
    for (let i = 0; i < out.length; i++) {
      for (let j = start; j < limit; j++) out[i] += 1;
    }
  }
  const artifact = compile(stalled, ['f64[]', 'f64', 'f64'], {maxNestedIterations: 3});
  // Deliberately do not run the nonterminating JavaScript oracle here.
  const kernel = host(artifact), actual = new Float64Array([91]);
  for (const [start, limit] of [[2 ** 53, 2 ** 53 + 2], [0, Infinity]]) {
    assert.throws(() => kernel.run(actual, start, limit), {code: 'KERNEL_EXECUTION_FAILED'});
    assert.equal(actual[0], 91);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('late nested bounds failure discards the entire pipeline before the original effects', () => {
  function update(out, input, n) {
    for (let i = 0; i < out.length; i++) out[i] += 10;
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < n; j++) out[i] += input[i + j];
    }
  }
  const actual = new Float64Array([1, 2]), initial = actual.slice(), expected = actual.slice();
  const input = new Float64Array([3, 4]); let calls = 0;
  const kernel = host(compile(update, ['f64[]', 'f64[]', 'f64']), {
    fallback(...args) { calls++; assert.deepEqual(actual, initial); return update(...args); },
  });
  update(expected, input, 2); kernel.run(actual, input, 2); assert.deepEqual(actual, expected);
  assert.equal(calls, 1); assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('non-Number bounds and aliased views retain their original observable execution', () => {
  function update(out, input, count) {
    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < count; j++) out[i] += input[j];
    }
  }
  let coercions = 0;
  const kernel = host(compile(update, ['f64[]', 'f64[]', 'f64']), {fallback: update});
  const actual = new Float64Array([1, 2]), input = new Float64Array([3, 4]);
  kernel.run(actual, input, {valueOf() {coercions++; return 2;}});
  assert.deepEqual(actual, new Float64Array([8, 9])); assert.equal(coercions, 6);
  const expected = actual.slice(); update(expected, expected, 2); kernel.run(actual, actual, 2);
  assert.deepEqual(actual, expected); assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('nested lexical scopes reject TDZ, escapes, index mutation and unsupported loop shapes', () => {
  const bodies = [
    'for(let j=j;j<3;j++) out[i]=j;',
    'for(let j=0;j<3;j++) {j+=1;out[i]=j;}',
    'for(let j=0;j<3;j++) {let j=2;out[i]=j;}',
    'let j=2;for(let j=0;j<3;j++) out[i]=j;',
    'for(let i=0;i<3;i++) out[i]=i;',
    'for(let j=0;j<3;j++) out[i]=callback(j);',
    'for(let j=0;j<3;j++) out[i]=j;out[i]=j;',
    'for(var j=0;j<3;j++) out[i]=j;',
    'for(let j=0;j<3;j*=2) out[i]=j;',
    'for(let j=0;j<3;j+=0) out[i]=j;',
    'for(let j=0;j<3;j+=i) out[i]=j;',
    'for(let j=0;3>j;j++) out[i]=j;',
    'for(let j=0;j<3;j++) out[i]=()=>j;',
    'while(out[i]) out[i]=0;',
  ];
  for (const body of bodies) assert.throws(() => compileNumericKernel(
    `function f(out) {for(let i=0;i<out.length;i++) {${body}}}`,
    {parameterTypes: ['f64[]'], checkedIndexing: true, structuredLoops: true}), NumericKernelCompileError, body);
  const source = 'function f(out) {for(let i=0;i<out.length;i++) for(let j=0;j<2;j++) out[i]=j;}';
  assert.throws(() => compileNumericKernel(source, {parameterTypes: ['f64[]']}), NumericKernelCompileError);
  assert.throws(() => compileNumericKernel(source, {parameterTypes: ['f64[]'], checkedIndexing: true}), NumericKernelCompileError);
  for (const options of [{structuredLoops: true}, {checkedIndexing: true, structuredLoops: 1}]) {
    assert.throws(() => compileNumericKernel(source, {parameterTypes: ['f64[]'], ...options}), {code: 'INVALID_KERNEL_ABI'});
  }
  for (const maxNestedIterations of [0, -1, 1.5, NaN, Infinity, 1000000001, '3']) {
    assert.throws(() => compileNumericKernel(source, {parameterTypes: ['f64[]'], checkedIndexing: true, maxNestedIterations}),
      {code: 'INVALID_KERNEL_ABI'});
  }
});

test('nested source limits refuse oversized loop graphs instead of expanding them', () => {
  for (const count of [8, 9]) {
    const prefix = Array.from({length: count}, (_, n) => `for(let j${n}=0;j${n}<1;j${n}++){`).join('');
    const source = `function f(out) {for(let i=0;i<out.length;i++){${prefix}out[i]=1;${'}'.repeat(count)}}}`;
    if (count === 9) assert.throws(() => compile(source, ['f64[]']), NumericKernelCompileError);
    else {
      const artifact = compile(source, ['f64[]']), out = new Float64Array(1);
      host(artifact).run(out); assert.equal(out[0], 1); assert.equal(artifact.nestedLoops.maxDepth, 8);
    }
  }
  const body = Array.from({length: 65}, (_, i) => `for(let j${i}=0;j${i}<1;j${i}++) out[i]+=1;`).join('');
  assert.throws(() => compile(`function f(out) {for(let i=0;i<out.length;i++){${body}}}`, ['f64[]']), NumericKernelCompileError);
});
