import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const sources = helpers => new Map(helpers.map(fn => [fn.name, fn.toString()]));
function compile(fn, helpers, parameterTypes, options = {}) {
  const settings = { parameterTypes, generalControl: true, helperSources: sources(helpers), ...options };
  const source = typeof fn === 'string' ? fn : fn.toString();
  const artifact = compileNumericKernel(source, settings);
  assert.deepEqual(artifact.wasm, compileNumericKernel(source, settings).wasm);
  assert.ok(WebAssembly.validate(artifact.wasm));
  const module = new WebAssembly.Module(artifact.wasm);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  assert.deepEqual(WebAssembly.Module.exports(module).map(entry => [entry.name, entry.kind]),
    [['run', 'function'], ['memory', 'memory']]);
  assert.equal(artifact.controlLoops.length, artifact.manifest.loopCount);
  assert.ok(Object.isFrozen(artifact.controlLoops));
  return artifact;
}
function compare(fn, kernel, args) {
  const expected = args.map(value => ArrayBuffer.isView(value) ? value.slice() : value);
  assert.ok(Object.is(kernel.run(...args), fn(...expected)));
  args.forEach((value, p) => {
    if (ArrayBuffer.isView(value)) value.forEach((number, i) =>
      assert.ok(Object.is(number, expected[p][i]), `parameter ${p}, element ${i}: ${number} != ${expected[p][i]}`));
    else assert.ok(Object.is(value, expected[p]));
  });
}
function refine(value, iterations) {
  let result = value;
  for (let j = 0; j < iterations; j++) result = (result + value / result) / 2;
  return result;
}
function update(out, values, iterations) {
  for (let i = 0; i < out.length; i++) out[i] = refine(values[i], iterations);
}
for (const [ArrayType, type] of [[Float32Array, 'f32[]'], [Float64Array, 'f64[]']]) {
  test(`${ArrayType.name}: repeated source update calls close iterative helpers into import-free Wasm`, () => {
    const artifact = compile(update, [refine], [type, type, 'f64']);
    assert.equal(artifact.manifest.maxLoopDepth, 2);
    assert.deepEqual(artifact.controlLoops.map(loop => [loop.kind, loop.functionName]),
      [['for', undefined], ['for', 'refine']]);
    assert.deepEqual(artifact.helpers, [{ name: 'refine', arity: 2 }]);
    const kernel = instantiateNumericKernel(artifact.wasm);
    for (let frame = 0; frame < 120; frame++) {
      compare(update, kernel, [new ArrayType(10000), ArrayType.from({ length: 10000 }, (_, i) => (i + 1) / 7), frame % 5]);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 120);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });
}

test('candidate selection admits a loop-bearing helper without weakening explicit opt-outs', () => {
  const options = { parameterTypes: ['f64[]', 'f64[]', 'f64'], helperSources: sources([refine]) };
  assert.throws(() => compileNumericKernel(update.toString(), options), { code: 'KERNEL_NOT_CLOSED' });
  assert.equal(compileNumericCandidate(update.toString(), options).manifest.version, 8);
  for (const flag of ['generalControl', 'structuredLoops', 'checkedIndexing']) {
    assert.throws(() => compileNumericCandidate(update.toString(), { ...options, [flag]: false }), { code: 'KERNEL_NOT_CLOSED' });
  }
});

test('a root may delegate all iteration, including from its final return, to transitive helpers', () => {
  function outer(n) { return middle(n) + 1; }
  function middle(n) { return inner(n) * 2; }
  function inner(n) { let total = 0; while (n > 0) { total += n; n--; } return total; }
  const artifact = compile(outer, [middle, inner], ['f64']);
  assert.equal(artifact.manifest.loopCount, 1);
  assert.equal(artifact.controlLoops[0].functionName, 'inner');
  assert.equal(artifact.manifest.maxLoopDepth, 1);
  const kernel = instantiateNumericKernel(artifact.wasm);
  for (const n of [0, 1, 10, 1234, NaN, -Infinity]) compare(outer, kernel, [n]);
  assert.equal(kernel.diagnostics.wasmCalls, 6);
});

test('nested helper for/while/do-while preserve continue, break, early return and update order', () => {
  function nested(n) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      let j = 0;
      while (j < 5) {
        j++;
        if (j === 2) continue;
        let k = 0;
        do {
          k++;
          if (k === 1) continue;
          if (k === 4) break;
          total += i + j + k;
          if (i > 4) return total;
        } while (k < j);
        if (j === 4) break;
      }
      if (i === 3) continue;
      total -= 1;
    }
    return total;
  }
  function root(n) { return nested(n); }
  const artifact = compile(root, [nested], ['f64']);
  assert.equal(artifact.manifest.maxLoopDepth, 3);
  const kernel = instantiateNumericKernel(artifact.wasm);
  for (let n = -1; n < 30; n++) compare(root, kernel, [n]);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
});

test('helper for initializers, comma updates, body shadows and outer lexical bindings stay separate', () => {
  function scope(n) {
    let i = 100, total = 0;
    for (let i = 1, j = i + 1; i < n; i++, j += i) {
      total += i + j;
      { const i = 17; total += i; }
      if (j > 30) break;
      if (i === 2) continue;
      total *= 2;
    }
    for (i = 0; i < 3; i++) total += i;
    return total + i;
  }
  function root(n) { return scope(n); }
  const kernel = instantiateNumericKernel(compile(root, [scope], ['f64']).wasm);
  for (const n of [0, 1, 2, 5, 100]) compare(root, kernel, [n]);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
});

test('dynamic Number induction preserves fractions, signed zero, NaN and indices larger than u32', () => {
  function sequence(start, end, step) {
    let total = start;
    for (let i = start; i < end; i += step) total += i;
    return total;
  }
  function root(start, end, step) { return sequence(start, end, step); }
  const kernel = instantiateNumericKernel(compile(root, [sequence], ['f64', 'f64', 'f64']).wasm);
  for (const args of [[0.25, 2, 0.25], [-1.5, 2, 0.5], [-0, 0, 1], [NaN, 3, 1],
    [4294967295, 4294967298, 1], [9007199254740988, 9007199254740992, 1]]) compare(root, kernel, args);
});

function count(n) { let sum = 0; while (n > 0) { sum += n; n--; } return sum; }
function sumCounts(n, repetitions) {
  let result = 0;
  for (let i = 0; i < repetitions; i++) result += count(n);
  return result;
}
test('all root and helper loop bodies spend one invocation budget, reset on every entry after success or trap', () => {
  const artifact = compile(sumCounts, [count], ['f64', 'f64'], { maxIterations: 6 });
  const kernel = instantiateNumericKernel(artifact.wasm);
  assert.equal(kernel.run(2, 2), 6); // Two root bodies plus four helper bodies.
  assert.throws(() => kernel.run(2, 3), { code: 'KERNEL_EXECUTION_FAILED' });
  assert.equal(kernel.run(2, 2), 6);
  assert.equal(kernel.run(5, 1), 15);
  assert.throws(() => kernel.run(6, 1), { code: 'KERNEL_EXECUTION_FAILED' });
  assert.equal(kernel.run(0, 6), 0);
  assert.throws(() => kernel.run(0, 7), { code: 'KERNEL_EXECUTION_FAILED' });
});

test('shared helper budget also covers calls in initializers, tests, updates and final returns', () => {
  function root(n) {
    let total = 0;
    for (let i = count(1); i < count(n); i += count(1)) total += i;
    return total + count(2);
  }
  // initializer 1; tests 2+2+2; root 2; updates 1+1; final 2 = 13.
  const exact = instantiateNumericKernel(compile(root, [count], ['f64'], { maxIterations: 13 }).wasm);
  compare(root, exact, [2]);
  const short = instantiateNumericKernel(compile(root, [count], ['f64'], { maxIterations: 12 }).wasm);
  assert.throws(() => short.run(2), { code: 'KERNEL_EXECUTION_FAILED' });
});

test('false while/for tests spend no body credit; do-while and continued bodies do', () => {
  function once(n) {
    let result = 0;
    while (n < 0) { result++; break; }
    for (let i = 0; i < 0; i++) result++;
    do { result++; if (n >= 0) continue; result++; } while (result < n);
    return result;
  }
  function root(n) { return once(n); }
  const kernel = instantiateNumericKernel(compile(root, [once], ['f64'], { maxIterations: 1 }).wasm);
  assert.equal(kernel.run(0), 1);
  assert.equal(kernel.run(1), 1);
  assert.throws(() => kernel.run(2), { code: 'KERNEL_EXECUTION_FAILED' });
});

test('empty and infinite helper loops trap at the private cap rather than freezing the caller', () => {
  const helperSources = new Map([['spin', 'function spin(){for(;;);return 1;}']]);
  const artifact = compileNumericKernel('function root(){return spin();}', { parameterTypes: [], helperSources,
    generalControl: true, maxIterations: 5 });
  const kernel = instantiateNumericKernel(artifact.wasm);
  assert.throws(() => kernel.run(), { code: 'KERNEL_EXECUTION_FAILED' });
});

test('budget exhaustion publishes no partial array writes and invokes the original once with original arguments and receiver', () => {
  function edit(out, n) {
    for (let i = 0; i < out.length; i++) { out[i] += 1; out[i] += count(n); }
  }
  const artifact = compile(edit, [count], ['f64[]', 'f64'], { maxIterations: 4 });
  const owner = {}, data = new Float64Array([10, 20, 30]);
  let calls = 0;
  const kernel = instantiateNumericKernel(artifact.wasm, { fallback(...args) {
    calls++;
    assert.equal(this, owner);
    assert.equal(args[0], data);
    assert.deepEqual(data, new Float64Array([10, 20, 30]));
    return Reflect.apply(edit, this, args);
  } });
  const reference = data.slice(); edit(reference, 2);
  assert.equal(Reflect.apply(kernel.run, owner, [data, 2]), undefined);
  assert.deepEqual(data, reference);
  assert.equal(calls, 1);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('checked aliased storage retains source-ordered helper results without extra copyback', () => {
  function edited(out, input, n) {
    for (let i = 0; i < out.length; i++) out[i] += input[i] + count(n);
  }
  const artifact = compile(edited, [count], ['f32[]', 'f32[]', 'f64']);
  const kernel = instantiateNumericKernel(artifact.wasm, { preserveAliasing: true });
  const data = new Float32Array([1, 2, 3, 4, 5]), expected = data.slice();
  kernel.run(data.subarray(1), data.subarray(0, 4), 2);
  edited(expected.subarray(1), expected.subarray(0, 4), 2);
  assert.deepEqual(data, expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('guarded Math and integer operations inside helper loops stay native and propagate guards', () => {
  function mix(x, n) {
    for (let i = 0; i < n; i++) { x = Math.imul(x ^ (x >>> 16), 2246822507); x += Math.round(-0.5); }
    return x;
  }
  function root(out, n) { for (let i = 0; i < out.length; i++) out[i] = mix(out[i], n); }
  const artifact = compile(root, [mix], ['u32[]', 'f64'], { allowMath: true });
  assert.equal(artifact.manifest.version, 9);
  assert.deepEqual(artifact.manifest.mathIntrinsics, ['imul', 'round']);
  const kernel = instantiateNumericKernel(artifact.wasm, { resolveMath: () => Math });
  compare(root, kernel, [new Uint32Array([0, 1, 0xffffffff, 0x80000000]), 5]);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
});

test('loop nesting is checked across helper calls, not only within each source function', () => {
  const helperSources = new Map();
  for (let i = 0; i < 9; i++) helperSources.set(`h${i}`,
    `function h${i}(n){let sum=0;for(let j=0;j<n;j++)sum+=${i === 8 ? 'j' : `h${i+1}(n)`};return sum;}`);
  assert.throws(() => compileNumericKernel('function root(n){return h0(n);}', {
    parameterTypes: ['f64'], generalControl: true, helperSources,
  }), /8-level/);
  const admitted = compileNumericKernel('function root(n){return h1(n);}', {
    parameterTypes: ['f64'], generalControl: true, helperSources,
  });
  assert.equal(admitted.manifest.maxLoopDepth, 8);
  assert.equal(admitted.manifest.loopCount, 8);
  assert.equal(instantiateNumericKernel(admitted.wasm).run(1), 0);
});

test('the static 64-loop limit spans the whole reachable helper graph', () => {
  const helperSources = new Map([['first', `function first(n){${'while(n<0){n++;}'.repeat(33)}return n;}`],
    ['second', `function second(n){${'while(n<0){n++;}'.repeat(32)}return n;}`]]);
  assert.throws(() => compileNumericKernel('function root(n){return first(n)+second(n);}', {
    parameterTypes: ['f64'], generalControl: true, helperSources,
  }), /64-loop/);
});

test('source closure, return completeness, lexical TDZ and no-recursion rules remain enforced', async t => {
  const rejected = [
    'function helper(n){while(n>0){n--;return n;}}',
    'function helper(n){for(;;){break;return 1;}}',
    'function helper(n){while(n>0){if(n>2)break;return 1;}}',
    'function helper(n){for(let i=i;i<n;i++);return n;}',
    'function helper(n){for(let i=j,j=0;i<n;i++);return n;}',
    'function helper(n){for(const i=0;i<n;i++);return n;}',
    'function helper(n){while(n>0){n--;const n=1;}return n;}',
    'function helper(n){while(n>0){n--;console.log(n);}return n;}',
    'function helper(n){while(n>0){return helper(n-1);}return n;}',
    'function helper(n){let sum=0;for(let x of n)sum+=x;return sum;}',
    'function helper(n){while(n>0){return;}return n;}',
    'function helper(n){outer:for(let i=0;i<n;i++){break outer;}return n;}',
    'function helper(n){while(n>0){n--;sum+=n;}return n;}',
    'function helper(n){while(n>0){n--;return arguments[0];}return n;}',
  ];
  for (const source of rejected) await t.test(source, () => {
    assert.throws(() => compileNumericKernel('function root(n){return helper(n);}', {
      parameterTypes: ['f64'], helperSources: new Map([['helper', source]]), generalControl: true,
    }), { code: 'KERNEL_NOT_CLOSED' });
  });
});
