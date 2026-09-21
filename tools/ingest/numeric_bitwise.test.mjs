import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const values = [0, -0, NaN, Infinity, -Infinity, 0.5, -0.5, 1.9, -1.9,
  31, 32, 33, -33, 0x7fffffff, 0x80000000, 0xffffffff, 2 ** 32, 2 ** 32 + 3,
  -(2 ** 32) - 3, 2 ** 53 - 1, -(2 ** 53) + 1, 2 ** 63 + 2048,
  2 ** 84 - 2 ** 31, -(2 ** 84) + 2 ** 31, 2 ** 84, Number.MAX_VALUE, Number.MIN_VALUE];
const operators = ['&', '|', '^', '<<', '>>', '>>>'];
// Evaluate only trusted fixture source to obtain the independent JS oracle.
const original = source => Function(`return (${source});`)();
function build(source, parameterTypes, options = {}) {
  const result = compileNumericKernel(source, {parameterTypes, ...options});
  assert.equal(WebAssembly.validate(result.wasm), true);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(result.wasm)), []);
  assert.deepEqual(compileNumericKernel(source, {parameterTypes, ...options}).wasm, result.wasm);
  return result;
}

for (const operator of operators) test(`compiled ${operator} preserves signedness and converted shift counts`, () => {
  const source = `function update(out, input, shift) {
    for (let i=0; i<out.length; i++) out[i] = input[i] ${operator} shift;
  }`;
  const reference = original(source);
  for (const ArrayType of [Float64Array, Float32Array]) {
    const type = ArrayType === Float64Array ? 'f64[]' : 'f32[]';
    const artifact = build(source, [type, type, 'f64']);
    assert.equal(artifact.manifest.mathIntrinsics, undefined);
    const kernel = instantiateNumericKernel(artifact.wasm);
    const input = new ArrayType(values), actual = new ArrayType(values.length), expected = new ArrayType(values.length);
    for (const shift of values) {
      reference(expected, input, shift); kernel.run(actual, input, shift);
      assert.deepEqual(actual, expected, `${ArrayType.name}, shift=${shift}`);
    }
    assert.equal(kernel.diagnostics.wasmCalls, values.length);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  }
});

test('unary not and nested bitwise predicates remain numeric rather than Boolean', () => {
  function update(out, input) {
    for (let i=0; i<out.length; i++) {
      const bits = ~input[i];
      if ((bits & 1) && ((bits >>> 0) !== 0)) out[i] = bits;
      else out[i] = ~(bits ^ input[i]);
    }
  }
  const {wasm} = build(update.toString(), ['f64[]', 'f64[]']);
  const actual = new Float64Array(values.length), expected = actual.slice(), input = new Float64Array(values);
  update(expected, input); instantiateNumericKernel(wasm).run(actual, input);
  assert.deepEqual(actual, expected);
});

test('every compound bitwise operation updates arrays and scalar state in order', () => {
  for (const operator of operators) {
    const source = `function update(out, input, state) {
      for (let i=0; i<out.length; i++) {
        out[i] ${operator}= input[i]; state ${operator}= out[i];
      }
      return state;
    }`;
    for (const ArrayType of [Float64Array, Float32Array]) {
      const type = ArrayType === Float64Array ? 'f64[]' : 'f32[]';
      const kernel = instantiateNumericKernel(build(source, [type, type, 'f64']).wasm);
      for (const seed of values) {
        const actual = new ArrayType(values), expected = actual.slice();
        const input = new ArrayType(values.slice().reverse());
        const result = original(source)(expected, input, seed);
        assert.equal(kernel.run(actual, input, seed), result);
        assert.deepEqual(actual, expected);
      }
    }
  }
});

test('loop-carried xorshift generates an exact reusable ordered sequence and scalar return', () => {
  function generate(out, seed) {
    let state = seed | 0;
    for (let i=0; i<out.length; i++) {
      state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
      out[i] = state >>> 0;
    }
    return state >>> 0;
  }
  const kernel = instantiateNumericKernel(build(generate.toString(), ['f64[]', 'f64']).wasm);
  for (const length of [0, 1, 4097]) for (const seed of values) {
    const actual = new Float64Array(length), expected = actual.slice();
    assert.equal(kernel.run(actual, seed), generate(expected, seed));
    assert.deepEqual(actual, expected);
  }
});

test('nested scalar helpers combine rotations, Math.imul, clz32 and parameter assignments', () => {
  function rotate(value, bits) { return (value << bits) | (value >>> (32 - bits)); }
  function mix(value, salt) {
    value ^= salt; value = Math.imul(value, 0x85ebca6b); value ^= value >>> 16;
    const flipped = ~value;
    return rotate(flipped, Math.clz32(value));
  }
  function update(out, input, salt) {
    for (let i=0; i<out.length; i++) out[i] = mix(input[i], salt);
  }
  const options = {allowMath:true, helperSources:new Map([['mix', mix.toString()], ['rotate', rotate.toString()]])};
  const artifact = build(update.toString(), ['f64[]', 'f64[]', 'f64'], options);
  assert.deepEqual(artifact.manifest.mathIntrinsics, ['clz32', 'imul']);
  assert.deepEqual(artifact.helpers.map(item => item.name), ['mix', 'rotate']);
  const kernel = instantiateNumericKernel(artifact.wasm, {resolveMath:()=>Math});
  const input = new Float64Array(values);
  for (const salt of values) {
    const actual = new Float64Array(values.length), expected = actual.slice();
    update(expected, input, salt); kernel.run(actual, input, salt);
    assert.deepEqual(actual, expected);
  }
});

test('checked masked scatters preserve collisions, Float32 store rounding and untouched tails', () => {
  for (const operator of operators) {
    const source = `function scatter(out, indices, input) {
      for (let i=0; i<indices.length; i++) out[indices[i] & 3] ${operator}= input[i];
    }`;
    const indices = new Uint32Array(values.map((_, i) => i * 7));
    const input = new Float32Array(values);
    const actual = new Float32Array([-1, 16777215, -0, 33, 91, 92]), expected = actual.slice();
    const kernel = instantiateNumericKernel(build(source, ['f32[]', 'u32[]', 'f32[]'], {checkedIndexing:true}).wasm);
    original(source)(expected, indices, input); kernel.run(actual, indices, input);
    assert.deepEqual(actual, expected);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
  }
});

test('out-of-view bitwise gather aborts all private writes before one original fallback', () => {
  function gather(out, indices, input) {
    for (let i=0; i<out.length; i++) out[i] = input[indices[i] >>> 0];
  }
  const actual = new Float64Array([91, 92]), initial = actual.slice();
  const indices = new Uint32Array([0, 0xffffffff]), input = new Float64Array([7]);
  let calls = 0;
  const kernel = instantiateNumericKernel(build(gather.toString(), ['f64[]', 'u32[]', 'f64[]'], {checkedIndexing:true}).wasm, {
    fallback(...args) {calls++; assert.deepEqual(actual, initial); return gather(...args);},
  });
  kernel.run(actual, indices, input);
  assert.deepEqual(actual, new Float64Array([7, NaN]));
  assert.equal(calls, 1); assert.equal(kernel.diagnostics.wasmCalls, 0);
  assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_EXECUTION_FAILED');
});

test('bitwise scalar coercion runs only in original JS when an argument is not a Number', () => {
  function update(out, value) {for (let i=0; i<out.length; i++) out[i] = out[i] | value;}
  let conversions = 0, calls = 0;
  const kernel = instantiateNumericKernel(build(update.toString(), ['f64[]', 'f64']).wasm, {
    fallback(...args) {calls++; return update(...args);},
  });
  const actual = new Float64Array([2, 4]);
  kernel.run(actual, {valueOf() {conversions++; return 1;}});
  assert.deepEqual(actual, new Float64Array([3, 5]));
  assert.equal(calls, 1); assert.equal(conversions, 2);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
  assert.throws(() => kernel.run(actual, 1n), TypeError);
  assert.equal(calls, 2); assert.deepEqual(actual, new Float64Array([3, 5]));
});

test('new Math methods retain descriptor guards, original getter counts and recovery', () => {
  for (const name of ['imul', 'clz32']) {
    const source = `function update(out) {
      for (let i=0; i<out.length; i++) out[i] = Math.${name}(out[i]${name === 'imul' ? ', 3' : ''});
    }`;
    const reference = original(source), artifact = build(source, ['f64[]'], {allowMath:true});
    let fallbacks = 0, gets = 0;
    const kernel = instantiateNumericKernel(artifact.wasm, {
      resolveMath:()=>Math, fallback(out) {fallbacks++; return reference(out);},
    });
    const descriptor = Object.getOwnPropertyDescriptor(Math, name);
    try {
      Object.defineProperty(Math, name, {configurable:true, get() {gets++; return () => 77;}});
      const out = new Float64Array([1, 2]); kernel.run(out);
      assert.deepEqual(out, new Float64Array([77, 77]));
      assert.equal(gets, 2); assert.equal(fallbacks, 1);
      assert.equal(kernel.diagnostics.wasmCalls, 0);
      assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_MATH_BINDING');
    } finally {Object.defineProperty(Math, name, descriptor);}
    const out = new Float64Array([1, 2]), expected = out.slice();
    reference(expected); kernel.run(out); assert.deepEqual(out, expected);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
  }
});

test('bitwise admission does not relax binding, type, loop-index or topology-write constraints', () => {
  const rejects = source => assert.throws(() => compileNumericKernel(source, {parameterTypes:['f64[]']}), NumericKernelCompileError);
  for (const expression of ['1n & 1n', 'true | 0', '"1" << 1', 'null ^ 0', 'a[0] % 2']) {
    rejects(`function f(a) {for(let i=0;i<a.length;i++) a[i] = ${expression};}`);
  }
  rejects('function f(a) {for(let i=0;i<a.length;i++) {const x=1; x|=2; a[i]=x;}}');
  rejects('function f(a) {for(let i=0;i<a.length;i++) {i|=1; a[i]=0;}}');
  rejects('function f(a) {for(let i=0;i<a.length;i++) {a[i]=x|0; let x=1;}}');
  assert.throws(() => compileNumericKernel('function f(a) {for(let i=0;i<a.length;i++) a[i] |= 1;}',
    {parameterTypes:['u32[]'], checkedIndexing:true}), NumericKernelCompileError);
  assert.throws(() => compileNumericKernel('function f(a) {for(let i=0;i<a.length;i++) a[i]=bad(a[i]);}', {
    parameterTypes:['f64[]'], helperSources:new Map([['bad', 'function bad(x) {const y=1; y>>>=x; return y;}']]),
  }), NumericKernelCompileError);
});
