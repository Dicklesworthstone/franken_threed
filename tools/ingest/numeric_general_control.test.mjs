import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

function compile(fn, parameterTypes, options = {}) {
  const source = typeof fn === 'string' ? fn : fn.toString();
  const settings = { parameterTypes, generalControl: true, ...options };
  const artifact = compileNumericKernel(source, settings);
  assert.deepEqual(artifact.wasm, compileNumericKernel(source, settings).wasm);
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
  assert.equal(artifact.manifest.version, 8);
  assert.equal(artifact.manifest.kind, 'closed-numeric-control');
  assert.equal(artifact.manifest.loops, undefined);
  assert.equal(artifact.manifest.boundParameter, undefined);
  assert.equal(artifact.manifest.automaticRouteAdmission, false);
  assert.equal(Object.isFrozen(artifact.controlLoops), true);
  assert.equal(artifact.controlLoops.length, artifact.manifest.loopCount);
  return artifact;
}
const runtime = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm, options);
const cloneArgs = args => args.map(value => ArrayBuffer.isView(value) ? value.slice() : value);
function compare(fn, kernel, args) {
  const expected = cloneArgs(args);
  assert.equal(kernel.run(...args), fn(...expected));
  assert.deepEqual(args, expected);
}
const types = ArrayType => ArrayType === Float32Array ? 'f32[]' : 'f64[]';

for (const ArrayType of [Float32Array, Float64Array]) {
  test(`${ArrayType.name}: dynamic ranges and steps execute partial records without divisibility guesses`, () => {
    function update(out, input, start, end, step) {
      for (let i = start; i < end; i += step) out[i] += input[i] / 3;
    }
    const kernel = runtime(compile(update, [types(ArrayType), types(ArrayType), 'f64', 'f64', 'f64']));
    for (const [start, end, step] of [[1, 9, 2], [2, 7, 3], [5, 5, 1], [0, 9, 1], [1, 0, 2]]) {
      compare(update, kernel, [new ArrayType(9).fill(1), new ArrayType(9).fill(5 / 7), start, end, step]);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 5);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });

  test(`${ArrayType.name}: reverse traversal and mutable induction keep source-ordered stores`, () => {
    function update(out, step) {
      for (let i = out.length - 1; i >= 0; i -= step) {
        out[i] += 1 / 3;
        if (i > 2) i -= 1;
      }
    }
    const kernel = runtime(compile(update, [types(ArrayType), 'f64']));
    for (const size of [0, 1, 4, 10000]) compare(update, kernel, [new ArrayType(size).fill(16777216), 1]);
    assert.equal(kernel.diagnostics.wasmCalls, 4);
  });

  test(`${ArrayType.name}: while binary search returns original boundary on every insertion position`, () => {
    function lowerBound(values, target) {
      let low = 0, high = values.length;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (values[middle] < target) low = middle + 1;
        else high = middle;
      }
      return low;
    }
    const kernel = runtime(compile(lowerBound, [types(ArrayType), 'f64']));
    const data = ArrayType.from({ length: 257 }, (_, i) => i * 2 - 17);
    for (let target = -20; target < 505; target++) compare(lowerBound, kernel, [data, target]);
    compare(lowerBound, kernel, [new ArrayType(0), 4]);
    compare(lowerBound, kernel, [data, NaN]);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });

  test(`${ArrayType.name}: while predicates see mutations through shifted aliases`, () => {
    function update(out, input) {
      let i = 0;
      while (i < out.length && input[i] < 20) { out[i] += input[i]; i++; }
      return i;
    }
    const kernel = runtime(compile(update, [types(ArrayType), types(ArrayType)]), { preserveAliasing: true });
    const data = new ArrayType([1, 2, 3, 4, 5, 6, 7]), expected = data.slice();
    assert.equal(kernel.run(data.subarray(1), data.subarray(0, 6)), update(expected.subarray(1), expected.subarray(0, 6)));
    assert.deepEqual(data, expected);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
  });
}

test('scalar-only Newton iteration uses guarded Math without a synthetic array parameter', () => {
  function root(x, tolerance) {
    if (x === 0) return x;
    let current = x, previous = 0;
    do {
      previous = current;
      current = (current + x / current) / 2;
    } while (Math.abs(current - previous) > tolerance);
    return current;
  }
  const artifact = compile(root, ['f64', 'f64'], { allowMath: true });
  const kernel = runtime(artifact, { resolveMath: () => Math });
  assert.deepEqual(artifact.manifest.lengthParameters, []);
  for (const value of [0, -0, 1, 2, 3, 1000000, NaN]) compare(root, kernel, [value, 1e-12]);
  assert.equal(kernel.diagnostics.copiedBytes, 0);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
});

test('numeric iteration retains fractional values, signed zero, NaN and f64 indices beyond i32', () => {
  function sequence(start, end, step) {
    let sum = start;
    for (let i = start; i < end; i += step) sum += i;
    return sum;
  }
  const kernel = runtime(compile(sequence, ['f64', 'f64', 'f64']));
  for (const values of [[0.25, 2, 0.25], [-1.5, 2, 0.5], [-0, 0, 1], [NaN, 3, 1],
    [4294967295, 4294967298, 1], [9007199254740988, 9007199254740992, 1]]) compare(sequence, kernel, values);
  assert.equal(kernel.diagnostics.wasmCalls, 6);
});

test('zero-parameter loops produce a scalar without spurious array/argument guards', () => {
  function constant() { let n = 0; do { n += 1; } while (n < 3); return n; }
  const kernel = runtime(compile(constant, []));
  assert.equal(kernel.run(), 3);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
  assert.throws(() => kernel.run(1), { code: 'KERNEL_ARGUMENT_COUNT' });
});

test('unsigned topology pointer chasing executes native while loops and checked gathers', () => {
  function traverse(next, values, start, stop) {
    let cursor = start, total = 0;
    while (cursor !== stop) { total += values[cursor]; cursor = next[cursor]; }
    return total;
  }
  for (const IndexType of [Uint16Array, Uint32Array]) {
    const kernel = runtime(compile(traverse, [IndexType === Uint16Array ? 'u16[]' : 'u32[]', 'f32[]', 'f64', 'f64']));
    compare(traverse, kernel, [new IndexType([3, 2, 5, 1, 0, 6]), new Float32Array([1, 2, 3, 4, 5, 6]), 4, 6]);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
  }
});

test('nested for/while/do-while break and continue reach their own update/test, not their parent', () => {
  function nested(out, n) {
    let total = 0;
    if (n > 0) {
      for (let i = 1; i < n; i += 1) {
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
            out[i] = total;
          } while (k < j);
          if (j === 4) break;
        }
        if (i === 3) continue;
        total -= 1;
      }
    }
    out[0] = total;
    return total;
  }
  const artifact = compile(nested, ['f64[]', 'f64']);
  assert.deepEqual(artifact.controlLoops.map(loop => [loop.kind, loop.depth]), [['for', 1], ['while', 2], ['do-while', 3]]);
  assert.equal(artifact.manifest.maxLoopDepth, 3);
  const kernel = runtime(artifact);
  for (const n of [0, 1, 4, 8]) compare(nested, kernel, [new Float64Array(8), n]);
  assert.equal(kernel.diagnostics.wasmCalls, 4);
});

test('for initializers, comma updates, shadowed body locals and post-loop scopes stay separate', () => {
  function scopes(n) {
    let i = 100, total = 0;
    for (let i = 1, j = i + 1; i < n; i++, j += i) {
      total += i + j;
      { const i = 17; total += i; }
      if (j > 30) break;
    }
    total += i;
    for (i = 0, total += 1; i < 2; i++, total += i) ;
    return total + i;
  }
  const kernel = runtime(compile(scopes, ['f64']));
  for (const n of [0, 2, 6, 9]) compare(scopes, kernel, [n]);
});

test('for without clauses, const initializers and update stores retain control order', () => {
  function update(out, stop) {
    let i = 0;
    for (;;) {
      if (i >= stop) break;
      out[i] += 1; i++;
    }
    for (const value = 7; ; ) { out[0] += value; break; }
    for (i = 1; i < stop; out[i] += i, i++) { if (i === 2) continue; out[i] *= 2; }
    return i;
  }
  const kernel = runtime(compile(update, ['f64[]', 'f64']));
  for (const stop of [0, 1, 5]) compare(update, kernel, [new Float64Array(6).fill(2), stop]);
});

test('do-while executes once on a false test and skips the test on break/return', () => {
  function once(out, input, stop) {
    do { out[0] += 1; if (stop === 1) break; if (stop === 2) return out[0]; }
    while (input[0] > 0);
    return out[0];
  }
  const kernel = runtime(compile(once, ['f64[]', 'f64[]', 'f64'], { maxIterations: 1 }));
  compare(once, kernel, [new Float64Array([2]), new Float64Array([0]), 0]);
  compare(once, kernel, [new Float64Array([2]), new Float64Array(0), 1]);
  compare(once, kernel, [new Float64Array([2]), new Float64Array(0), 2]);
  assert.equal(kernel.diagnostics.wasmCalls, 3);
});

test('short-circuit loop predicates do not evaluate unexecuted out-of-bounds reads', () => {
  function empty(a, n) {
    let sum = 0;
    while (n < 0 && a[99] > 1) sum += 1;
    do { sum += 2; } while (n > 0 && a[99] > 1);
    for (; n < 0 ? a[99] > 1 : false; ) sum += 4;
    return sum;
  }
  const kernel = runtime(compile(empty, ['f64[]', 'f64']));
  compare(empty, kernel, [new Float64Array(0), 0]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('void early returns and final return preserve untouched array bytes', () => {
  function update(out, stop) {
    let i = 0;
    while (i < out.length) { if (i >= stop) return; out[i] += 1; i++; }
    return;
  }
  const kernel = runtime(compile(update, ['f64[]', 'f64']));
  for (const stop of [0, 2, 10]) compare(update, kernel, [new Float64Array([1, -0, NaN, 5]), stop]);
  assert.equal(kernel.diagnostics.wasmCalls, 3);
});

test('top-level and nested loops share one exact budget, reset for every invocation', () => {
  function update(out, n) {
    let i = 0;
    while (i < n) { out[i] += 1; i++; }
    for (let j = 0; j < n; j++) { let k = 0; do { out[j] += 2; k++; } while (k < 2); }
  }
  const artifact = compile(update, ['f64[]', 'f64'], { maxIterations: 8 });
  const kernel = runtime(artifact);
  for (let call = 0; call < 3; call++) compare(update, kernel, [new Float64Array(3).fill(1), 2]);
  const data = new Float64Array([1, 2, 3]), before = data.slice();
  assert.throws(() => kernel.run(data, 3), { code: 'KERNEL_EXECUTION_FAILED' });
  assert.deepEqual(data, before);
  compare(update, kernel, [data, 2]);
  assert.equal(kernel.diagnostics.wasmCalls, 4);
});

test('false pre-tests consume no body budget, including after the final allowed body', () => {
  function count(n) {
    let total = 0;
    while (total < n) total++;
    for (let i = 0; i < 0; i++) total++;
    while (false) total++;
    return total;
  }
  const kernel = runtime(compile(count, ['f64'], { maxIterations: 3 }));
  assert.equal(kernel.run(3), 3);
  assert.equal(kernel.run(0), 0);
  assert.throws(() => kernel.run(4), { code: 'KERNEL_EXECUTION_FAILED' });
});

test('infinite/continue-only loops and non-progressing Number induction exhaust native budget', () => {
  for (const source of [
    'function infinite() { while (true) continue; return 0; }',
    'function infinite() { do { continue; } while (true); return 0; }',
    'function infinite() { for (;;) continue; return 0; }',
    'function infinite() { for (let i=9007199254740992; i<9007199254740994; i++) {} return 0; }',
  ]) {
    const kernel = runtime(compile(source, [], { maxIterations: 7 }));
    assert.throws(() => kernel.run(), { code: 'KERNEL_EXECUTION_FAILED' });
    assert.equal(kernel.diagnostics.wasmCalls, 0);
  }
});

test('budget failure publishes no speculative alias writes before exactly one original fallback', () => {
  function update(out, input, n) {
    let i = 0;
    while (i < n) { out[i] += input[i]; i++; }
    return i;
  }
  const artifact = compile(update, ['f64[]', 'f64[]', 'f64'], { maxIterations: 2 });
  const data = new Float64Array([1, 2, 3, 4]), before = data.slice(), expected = data.slice();
  const receiver = {}; let calls = 0;
  const kernel = runtime(artifact, { preserveAliasing: true, fallback(...args) {
    calls++; assert.equal(this, receiver); assert.deepEqual(data, before); return update(...args);
  } });
  assert.equal(kernel.run.call(receiver, data.subarray(1), data.subarray(0, 3), 3),
    update(expected.subarray(1), expected.subarray(0, 3), 3));
  assert.deepEqual(data, expected); assert.equal(calls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.copiedBytes, 0);
});

test('late predicate, body, and update address failures all abort before publication', () => {
  const functions = [
    function predicate(out, input) { let i = 0; while (input[i] > 0) { out[0] += 1; i++; } },
    function body(out, input) { let i = 0; do { out[0] += input[i]; i++; } while (i < 4); },
    function update(out, input) { for (let i = 0; i < 4; out[0] += input[i], i++) out[0] += 1; },
  ];
  for (const fn of functions) {
    const artifact = compile(fn, ['f64[]', 'f64[]']);
    const out = new Float64Array([1]), input = new Float64Array([2, 3]);
    const strict = runtime(artifact);
    assert.throws(() => strict.run(out, input), { code: 'KERNEL_EXECUTION_FAILED' });
    assert.deepEqual(out, new Float64Array([1]));
    let calls = 0;
    const kernel = runtime(artifact, { fallback(...args) { calls++; assert.equal(out[0], 1); return fn(...args); } });
    compare(fn, kernel, [out, input]);
    assert.equal(calls, 1); assert.equal(kernel.diagnostics.fallbackCalls, 1);
  }
});

test('argument coercions and fallback exceptions are not duplicated by new loop admission', () => {
  function count(out, n) { let i = 0; while (i < n) { out[i] += 1; i++; } return i; }
  const artifact = compile(count, ['f64[]', 'f64']);
  let conversions = 0;
  const number = { valueOf() { conversions++; return 2; } };
  const kernel = runtime(artifact, { fallback: count });
  assert.equal(kernel.run(new Float64Array(2), number), 2);
  assert.equal(conversions, 3);
  const failure = {}, bad = { valueOf() { throw failure; } };
  assert.throws(() => kernel.run(new Float64Array(2), bad), error => error === failure);
  assert.equal(kernel.diagnostics.fallbackCalls, 2);
});

test('source closure, lexical TDZ, immutable bindings and return types remain enforced', () => {
  const sources = [
    'function f(a) { while (true) { a[0] = external(); break; } }',
    'function f(a) { while (true) { a[0] = this.x; break; } }',
    'function f(a) { for (let i=i; i<2; i++) a[i]=1; }',
    'function f(a) { for (let i=j, j=0; i<2; i++) a[i]=1; }',
    'function f(a) { while (true) { a[0]=i; let i=0; break; } }',
    'function f(a) { for (const i=0; i<2; i++) a[i]=1; }',
    'function f(a) { for (let i=0; i<2; i++) a[i]=1; return i; }',
    'function f(a) { while (true) { a[0]=1; return; } return 1; }',
    'function f(a) { while (true) { a[0]=1; return 1; } }',
    'function f(a) { outer: while (true) { a[0]=1; break outer; } }',
    'function f(a) { while (true) { a[0]=1; eval("a"); break; } }',
    'function f(a) { for (var i=0; i<2; i++) a[i]=1; }',
    'function f(a) { for (let i=0; i<2; i++) a[i]=(() => i)(); }',
  ];
  for (const source of sources) assert.throws(() => compileNumericKernel(source, { parameterTypes: ['f64[]'], generalControl: true }),
    error => error instanceof NumericKernelCompileError && error.code === 'KERNEL_NOT_CLOSED', source);
});

test('generic control has bounded compilation depth/count and validated explicit options', () => {
  const settings = { parameterTypes: ['f64'], generalControl: true };
  const source = 'function f(n) { while(n>0) n--; return n; }';
  for (const maxIterations of [0, -1, NaN, Infinity, 1.5, 1000000001, '3']) {
    assert.throws(() => compileNumericKernel(source, { ...settings, maxIterations }), { code: 'INVALID_KERNEL_ABI' });
  }
  for (const option of [{ generalControl: 1 }, { checkedIndexing: false }, { structuredLoops: false }]) {
    assert.throws(() => compileNumericKernel(source, { ...settings, ...option }), { code: 'INVALID_KERNEL_ABI' });
  }
  const loops = depth => 'function f(n) {' + 'while(n>0){'.repeat(depth) + 'n--;' + '}'.repeat(depth) + 'return n;}';
  compile(loops(8), ['f64']);
  assert.throws(() => compile(loops(9), ['f64']), { code: 'KERNEL_NOT_CLOSED' });
  const many = n => 'function f(n){' + 'while(false){};'.repeat(n) + 'return n;}';
  compile(many(64), ['f64']);
  assert.throws(() => compile(many(65), ['f64']), { code: 'KERNEL_NOT_CLOSED' });
});

function u32(n) { const bytes = []; do { const b = n & 127; n >>>= 7; bytes.push(b | (n ? 128 : 0)); } while (n); return bytes; }
function changeManifest(artifact, update) {
  const manifest = structuredClone(artifact.manifest); update(manifest);
  const wasm = artifact.wasm; let offset = 8, customStart = null;
  while (offset < wasm.length) {
    const start = offset, id = wasm[offset++]; let size = 0, shift = 0, b;
    do { b = wasm[offset++]; size += (b & 127) * 2 ** shift; shift += 7; } while (b & 128);
    if (id === 0) { customStart = start; break; }
    offset += size;
  }
  assert.notEqual(customStart, null);
  const name = new TextEncoder().encode('f3d.numeric-kernel');
  const content = [...u32(name.length), ...name, ...new TextEncoder().encode(JSON.stringify(manifest))];
  return new Uint8Array([...wasm.subarray(0, customStart), 0, ...u32(content.length), ...content]);
}

test('v8 manifest rejects missing budgets, fabricated array trip counts and inconsistent storage', () => {
  function f(out, n) { while (n > 0) { out[0] += 1; n--; } }
  const artifact = compile(f, ['f64[]', 'f64']);
  const mutations = [
    m => { delete m.maxIterations; }, m => { m.maxIterations = 0; }, m => { m.maxIterations = 1000000001; },
    m => { m.controlSemantics = 'unchecked'; }, m => { m.loops = []; }, m => { m.loopStride = 1; },
    m => { m.boundParameter = 0; }, m => { m.boundParameters = [0]; }, m => { m.lengthParameters = [1]; },
    m => { m.loopCount = 0; }, m => { m.loopCount = 65; }, m => { m.maxLoopDepth = 2; },
    m => { m.parameters[0].read = false; }, m => { m.parameters[0].access = {}; },
    m => { m.parameters[0].type = 'u32[]'; }, m => { m.parameters[1].read = true; },
    m => { m.version = 7; }, m => { m.version = 9; }, m => { m.resultType = 'i32'; },
  ];
  for (const mutate of mutations) assert.throws(() => instantiateNumericKernel(changeManifest(artifact, mutate)), { code: 'KERNEL_ABI_MISMATCH' });
});

// Deterministic v1-v7 bytecode fixtures captured from the parent compiler.
const legacy = [
  ['function f(a){for(let i=0;i<a.length;i++)a[i]+=1;}', ['f64[]'], {}],
  ['function f(a){for(let i=0;i<a.length;i++)a[i]+=1;}', ['f32[]'], {}],
  ['function f(a){for(let i=0;i<a.length;i+=3)a[i]+=1;}', ['f32[]'], {}],
  ['function f(a,u){for(let i=0;i<a.length;i++)a[i]+=u[0];}', ['f64[]','f64[]'], {}],
  ['function f(a){let sum=0;for(let i=0;i<a.length;i++)sum+=a[i];return sum;}', ['f64[]'], {}],
  ['function f(a){for(let i=0;i<a.length;i++)a[i]+=1;for(let j=0;j<a.length;j++)a[j]*=2;}', ['f64[]'], {}],
  ['function f(a){for(let i=0;i<a.length;i++){for(let j=0;j<3;j++){if(j===1)continue;a[i]+=j;}}}', ['f64[]'], {checkedIndexing:true,structuredLoops:true}],
];
const legacyHashes = [
  "a4cc7cb2e124584426f18c7e59fe7abd0354a16502af81307f105d2001faeb7e",
  "12d9cd132f51e28eccf6bb62aae9a1162e4f61af7c5791b0bb3628760608d4fc",
  "6418c1c4079b86a91561facb50317febde6e8e1f8311046f0b9aa6423c0dfcfc",
  "cc23467b91f03ec87c4406086875e1533211e0cbdb30c7015b8dc713ef869464",
  "90648f1fec2383afd855a6215739581a2db54018e6f57f1cbe8e498397755536",
  "5ffa86b35dc90f76ae10343656dd858e35cb5e2550010922c285642a1e1aa956",
  "69ceff237e0ac9e9e89b4f125fe56ca705d21860b1c2d87b46531ed3166df1fe"
];
for (let index = 0; index < legacy.length; index++) test(`legacy ABI v${index + 1} keeps identical executable bytes`, () => {
  const [source, parameterTypes, options] = legacy[index];
  const artifact = compileNumericKernel(source, { parameterTypes, ...options });
  assert.equal(artifact.manifest.version, index + 1);
  assert.equal(createHash('sha256').update(artifact.wasm).digest('hex'), legacyHashes[index]);
});

test('closed scalar helper graph executes inside native loop tests and bodies', () => {
  function difference(a, b) { return Math.abs(a - b); }
  function midpoint(a, b) { return (a + b) / 2; }
  function converge(out, target) {
    let i = 0;
    while (difference(out[0], target) > 0.001) {
      out[0] = midpoint(out[0], target); i++;
    }
    return i;
  }
  const artifact = compile(converge, ['f32[]', 'f64'], { allowMath: true,
    helperSources: new Map([['difference', difference.toString()], ['midpoint', midpoint.toString()]]) });
  assert.deepEqual(artifact.helpers.map(helper => helper.name), ['difference', 'midpoint']);
  const kernel = runtime(artifact, { resolveMath: () => Math });
  for (const target of [-7, 0, 9]) compare(converge, kernel, [new Float32Array([1]), target]);
  assert.equal(kernel.diagnostics.wasmCalls, 3);
});

test('deterministic mixed-loop differential matrix preserves dynamic steps and nested exits', () => {
  function mixed(out, n, stop, initialStep) {
    let total = 0, step = initialStep;
    for (let i = 0; i < n; i += step) {
      let j = n;
      while (j > 0) {
        j -= 1;
        if (j === stop) continue;
        let k = 0;
        do {
          k += 1;
          if (k === 2) continue;
          total += i + j + k / 3;
          if (total > 1000) break;
        } while (k < initialStep);
        out[i] += total;
        if (j < stop) break;
      }
      step = step < 3 ? step + 1 : 1;
      if (i === stop) continue;
      out[i] /= 3;
    }
    return total;
  }
  for (const ArrayType of [Float32Array, Float64Array]) {
    const kernel = runtime(compile(mixed, [types(ArrayType), 'f64', 'f64', 'f64']));
    for (let seed = 0; seed < 128; seed++) {
      const n = seed % 12, stop = (seed * 7) % 13, step = 1 + seed % 4;
      compare(mixed, kernel, [ArrayType.from({ length: 13 }, (_, i) => (seed + i) / 3), n, stop, step]);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 128);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  }
});
