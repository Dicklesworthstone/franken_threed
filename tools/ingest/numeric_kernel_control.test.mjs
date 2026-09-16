import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

function runComparison(fn, types, inputs, repeats = 1) {
  const artifact = compileNumericKernel(fn.toString(), { parameterTypes: types });
  const kernel = instantiateNumericKernel(artifact.wasm);
  const expected = inputs.map(value => value instanceof Float64Array ? value.slice() : value);
  for (let i = 0; i < repeats; i++) {
    fn(...expected);
    kernel.run(...inputs);
    for (let p = 0; p < types.length; p++) {
      if (types[p] === 'f64[]') assert.deepEqual(inputs[p], expected[p]);
    }
  }
  assert.equal(kernel.diagnostics.wasmCalls, repeats);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
  return artifact;
}

test('compiles bounded particle integration with nested conditional expressions and stores', () => {
  function bounce(x, v, dt, limit) {
    for (let i = 0; i < x.length; i++) {
      const next = x[i] + v[i] * dt;
      x[i] = next < -limit ? -limit : (next > limit ? limit : next);
      if (next < -limit || next > limit) {
        v[i] = -v[i];
      }
    }
  }
  runComparison(bounce, ['f64[]', 'f64[]', 'f64', 'f64'], [
    new Float64Array([-1, -0.5, 0, 0.9, 1]), new Float64Array([-1, 2, 0, 4, 1]), 1 / 60, 1,
  ], 200);
});

test('conditional write-only arrays preserve skipped elements across reused Wasm memory', () => {
  function selective(output, mask, value) {
    for (let i = 0; i < output.length; i++) {
      if (mask[i] > 0) output[i] = value;
    }
  }
  const compiled = compileNumericKernel(selective.toString(), { parameterTypes: ['f64[]', 'f64[]', 'f64'] });
  assert.equal(compiled.manifest.parameters[0].read, true);
  const engine = instantiateNumericKernel(compiled.wasm);
  const out = new Float64Array([1, 2, 3, 4]), mask = new Float64Array([1, 0, 1, 0]);
  engine.run(out, mask, 10);
  assert.deepEqual([...out], [10, 2, 10, 4]);
  out.set([31, 32, 33, 34]);
  mask.set([0, 1, 0, 1]);
  engine.run(out, mask, 20);
  assert.deepEqual([...out], [31, 20, 33, 20]);
});

test('supports if/else chains and lexical branch-local names without scope leakage', () => {
  function classify(x, out) {
    for (let i = 0; i < x.length; i++) {
      const value = x[i];
      if (value < 0) {
        const result = -value;
        out[i] = result;
      } else if (value > 0) {
        const result = value * 2;
        { const result = value * 3; out[i] = result; }
        out[i] += result;
      } else {
        const result = 7;
        out[i] = result;
      }
      out[i] += value;
    }
  }
  runComparison(classify, ['f64[]', 'f64[]'], [
    new Float64Array([-5, -1, -0, 0, 1, 5, NaN]), new Float64Array(7),
  ], 3);
});

test('numeric truthiness matches JavaScript for both zero signs, NaN and infinities', () => {
  function truthy(input, output) {
    for (let i = 0; i < input.length; i++) {
      output[i] = input[i] ? 1 : 0;
      if (!input[i]) output[i] += 2;
      if (input[i] ? false : true) output[i] += 4;
      if (true && !false) output[i] += 8;
    }
  }
  runComparison(truthy, ['f64[]', 'f64[]'], [
    new Float64Array([0, -0, NaN, Infinity, -Infinity, 1, -1, 1e-320]), new Float64Array(8),
  ]);
});

test('strict numeric comparisons preserve ordered and unordered IEEE semantics', () => {
  function compare(x, y, out) {
    for (let i = 0; i < x.length; i++) {
      out[i] = 0;
      if (x[i] === y[i]) out[i] += 1;
      if (x[i] !== y[i]) out[i] += 2;
      if (x[i] < y[i]) out[i] += 4;
      if (x[i] <= y[i]) out[i] += 8;
      if (x[i] > y[i]) out[i] += 16;
      if (x[i] >= y[i]) out[i] += 32;
    }
  }
  runComparison(compare, ['f64[]', 'f64[]', 'f64[]'], [
    new Float64Array([0, -0, NaN, Infinity, -Infinity, 1, -1, NaN]),
    new Float64Array([-0, 0, NaN, Infinity, Infinity, -1, 1, 1]), new Float64Array(8),
  ]);
});

test('conditional and logical lowering are lazy in the actual Wasm engine', () => {
  // Deliberately use the raw ABI to put the RHS array outside memory. Closed
  // runtime calls never do this, but a trap detects eager lowering unambiguously.
  for (const [predicate, initial] of [['x[i] > 0 && y[i] > 0', 0], ['x[i] > 0 || y[i] > 0', 1]]) {
    const source = `function lazy(x,y) { for(let i=0;i<x.length;i++) x[i]=(${predicate})?3:4; }`;
    const { wasm } = compileNumericKernel(source, { parameterTypes: ['f64[]', 'f64[]'] });
    const { memory, run } = new WebAssembly.Instance(new WebAssembly.Module(wasm)).exports;
    const x = new Float64Array(memory.buffer, 0, 1);
    x[0] = initial;
    assert.doesNotThrow(() => run(0, memory.buffer.byteLength, 1));
    assert.equal(x[0], initial ? 3 : 4);
    x[0] = initial ? 0 : 1;
    assert.throws(() => run(0, memory.buffer.byteLength, 1), WebAssembly.RuntimeError);
  }
  const source = 'function lazy(x,y) { for(let i=0;i<x.length;i++) x[i]=x[i]>0?y[i]:42; }';
  const { wasm } = compileNumericKernel(source, { parameterTypes: ['f64[]', 'f64[]'] });
  const { memory, run } = new WebAssembly.Instance(new WebAssembly.Module(wasm)).exports;
  assert.doesNotThrow(() => run(0, memory.buffer.byteLength, 1));
  assert.equal(new Float64Array(memory.buffer, 0, 1)[0], 42);
});

test('refuses effects in untaken branches, boolean-as-number confusion, and invalid scopes', async t => {
  for (const body of [
    'if (false) x[i]=external(); else x[i]=1;',
    'x[i]=(1===true)?1:2;',
    'const flag=x[i]>0; x[i]=flag===1?2:3;',
    'if(x[i]) { const hidden=1; } x[i]=hidden;',
    'const outer=1; { x[i]=outer; const outer=2; }',
    'if(x[i]) { const t=t+1; x[i]=t; }',
    'if(x[i]) { const i=1; x[i]=2; }',
    'if(x[i]) { const x=1; } x[i]=2;',
    'x[i]=x[i]||2;',
    'if(x[i]) continue; x[i]=1;',
    'if(x[i]) break; x[i]=1;',
    'var t=1; x[i]=t;',
    'if(x[i]) x[i]=y[i+1];',
  ]) await t.test(body, () => {
    assert.throws(() => compileNumericKernel(`function f(x) { for(let i=0;i<x.length;i++) { ${body} } }`,
      { parameterTypes: ['f64[]'] }), NumericKernelCompileError);
  });
});
