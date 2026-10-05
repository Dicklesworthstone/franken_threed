import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const sources = functions => new Map(functions.map(fn => [fn.name, fn.toString()]));
function compile(fn, helpers, parameterTypes, options = {}) {
  const settings = { parameterTypes, generalControl: true, helperSources: sources(helpers), ...options };
  const result = compileNumericKernel(fn.toString(), settings);
  assert.deepEqual(result.wasm, compileNumericKernel(fn.toString(), settings).wasm);
  assert.ok(WebAssembly.validate(result.wasm));
  const module = new WebAssembly.Module(result.wasm);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  assert.deepEqual(WebAssembly.Module.exports(module).map(entry => [entry.name, entry.kind]),
    [['run', 'function'], ['memory', 'memory']]);
  return result;
}
const engine = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm, { preserveAliasing: true, ...options });
function equal(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Object.is(actual[i], expected[i]), `element ${i}: ${actual[i]} != ${expected[i]}`);
}
function get(a, index) { return a[index]; }
function put(a, index, value) { a[index] = value; return a[index]; }
function update(a, s) { for (let i = 0; i < a.length; i++) put(a, i, get(a, i) * s); }
const types = [
  [Float64Array, 'f64[]'], [Float32Array, 'f32[]'], [Int8Array, 'i8[]'],
  [Uint8Array, 'u8[]'], [Uint8ClampedArray, 'u8c[]'], [Int16Array, 'i16[]'],
  [Uint16Array, 'u16[]'], [Int32Array, 'i32[]'], [Uint32Array, 'u32[]'],
];
for (const [ArrayType, type] of types) {
  test(`${ArrayType.name}: helper reads/writes round at each store, over repeated native frames`, () => {
    const artifact = compile(update, [get, put], [type, 'f64']);
    assert.deepEqual(artifact.manifest.parameters.map(p => [p.read, p.write]), [[true, true], [false, false]]);
    const kernel = engine(artifact);
    const data = new ArrayType([0, -0, 1 / 3, 0.5, 1.5, 2.5, -3.5, 65537, 16777217, 2 ** 32 - 1]);
    const expected = data.slice();
    for (const scalar of [1, 1 / 3, 1.5, -2.5, 0, -0, 2 ** 90, Infinity, -Infinity, NaN]) {
      assert.equal(kernel.run(data, scalar), update(expected, scalar)); equal(data, expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 10); assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });
}

test('one graph specializes the same helper for f32, f64 and integer views without host calls', () => {
  function root(a, b, c) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
      result += put(a, i, get(b, i) / 3);
      result += put(c, i, get(a, i) * 255);
    }
    return result;
  }
  const artifact = compile(root, [get, put], ['f32[]', 'f64[]', 'u8c[]']);
  assert.deepEqual(artifact.helpers.filter(h => h.name === 'put').map(h => h.parameterTypes[0]), ['f32[]', 'u8c[]']);
  const kernel = engine(artifact);
  const input = new Float64Array([1, 0.5, 1 / 3, -1, 3]), a = new Float32Array(5), c = new Uint8ClampedArray(5);
  const ea = a.slice(), ec = c.slice();
  assert.equal(kernel.run(a, input, c), root(ea, input, ec)); equal(a, ea); equal(c, ec);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('void helpers, transitive array forwarding and early returns form one transaction', () => {
  function adjust(a, s) { for (let i = 0; i < a.length; i++) { if (i === 3) return; a[i] *= s; } }
  function forward(a, b, s) { adjust(a, s); adjust(b, s + 1); }
  function root(a, b, s) { forward(a, b, s); }
  const artifact = compile(root, [forward, adjust], ['f32[]', 'f32[]', 'f64']);
  assert.equal(artifact.manifest.loopCount, 1);
  assert.deepEqual(artifact.helpers.map(h => h.resultType), ['void', 'void']);
  assert.deepEqual(artifact.manifest.parameters.map(p => p.write), [true, true, false]);
  const kernel = engine(artifact), a = new Float32Array([1, 2, 3, 4]), b = new Float32Array([5, 6, 7, 8]);
  const ea = a.slice(), eb = b.slice(); assert.equal(kernel.run(a, b, 1 / 3), root(ea, eb, 1 / 3));
  equal(a, ea); equal(b, eb); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

for (const [ArrayType, type] of types.slice(0, 2)) {
  test(`${ArrayType.name}: identical and shifted aliases remain source ordered through cached helper calls`, () => {
    function mutate(a, b, i) { a[i] += b[i] / 3; b[i] -= a[i] / 7; }
    function root(a, b) { for (let i = 0; i < a.length; i++) mutate(a, b, i); }
    const kernel = engine(compile(root, [mutate], [type, type]));
    for (const [ao, bo] of [[0, 0], [0, 1], [1, 0], [0, 2], [2, 0]]) {
      const data = new ArrayType([1, 2, 3, 4, 5, 6, 7]), expected = data.slice();
      kernel.run(data.subarray(ao, ao + 5), data.subarray(bo, bo + 5));
      root(expected.subarray(ao, ao + 5), expected.subarray(bo, bo + 5)); equal(data, expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 5); assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });
}

test('nested calls evaluate all arguments left-to-right exactly once, including Math and bitwise conversions', () => {
  function tick(a, value) { a[0] += 1; return value + a[0]; }
  function pair(x, a, y) { a[1] = x * 100 + y; return a[1]; }
  function root(a) {
    let result = 0;
    for (let i = 0; i < 4; i++) {
      result += pair(tick(a, i), a, tick(a, 2 * i));
      result += Math.imul(tick(a, i), tick(a, i + 1));
      result += Math.min(tick(a, 0), tick(a, 1), tick(a, 2));
      result += tick(a, 1) << tick(a, 2);
    }
    return result;
  }
  const kernel = engine(compile(root, [tick, pair], ['f64[]'], { allowMath: true }), { resolveMath: () => Math });
  const data = new Float64Array(2), expected = data.slice();
  assert.equal(kernel.run(data), root(expected)); equal(data, expected); assert.equal(data[0], 36);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('a compound destination index and its old value precede RHS helper writes', () => {
  function index(a) { a[1] += 1; return a[1] - 1; }
  function rhs(a) { a[0] = 100; return 3; }
  function change(a) { a[index(a)] += rhs(a); }
  function root(a) { for (let i = 0; i < 1; i++) change(a); return a[0]; }
  const kernel = engine(compile(root, [index, rhs, change], ['f64[]']));
  const a = new Float64Array([2, 0]), expected = a.slice();
  assert.equal(kernel.run(a), root(expected)); equal(a, expected); assert.deepEqual([...a], [5, 1]);
});

test('helper index checks use each view, including the shorter alias, before any publication', () => {
  function root(a, b) { for (let i = 0; i < a.length; i++) put(a, i, get(b, i) + 1); }
  const artifact = compile(root, [get, put], ['f64[]', 'f64[]']);
  const data = new Float64Array([1, 2, 3, 4]), before = data.slice();
  const native = engine(artifact);
  assert.throws(() => native.run(data, data.subarray(0, 2)), { code: 'KERNEL_EXECUTION_FAILED' }); equal(data, before);
  let fallbacks = 0;
  const kernel = engine(artifact, { fallback(a, b) { fallbacks++; equal(data, before); return root(a, b); } });
  const expected = data.slice(); root(expected, expected.subarray(0, 2));
  kernel.run(data, data.subarray(0, 2)); equal(data, expected);
  assert.equal(fallbacks, 1); assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('negative/fractional/nonfinite helper subscripts abort; numeric negative zero is index zero', () => {
  function root(a, at) { for (let i = 0; i < 1; i++) put(a, at, 7); return get(a, at); }
  const kernel = engine(compile(root, [get, put], ['f64[]', 'f64']));
  for (const at of [-1, 0.5, NaN, Infinity, -Infinity, 2, 2 ** 32]) {
    const data = new Float64Array([2, 3]);
    assert.throws(() => kernel.run(data, at), { code: 'KERNEL_EXECUTION_FAILED' }); equal(data, [2, 3]);
  }
  const data = new Float64Array([2, 3]); assert.equal(kernel.run(data, -0), 7); equal(data, [7, 3]);
});

test('root/helper loops share one budget, rollback writes, and reset even after a trap', () => {
  function pass(a) { for (let j = 0; j < a.length; j++) a[j] += 1; }
  function root(a, repetitions) { for (let i = 0; i < repetitions; i++) pass(a); }
  const artifact = compile(root, [pass], ['f32[]', 'f64'], { maxIterations: 6 });
  const native = engine(artifact), a = new Float32Array([1, 2]);
  native.run(a, 2); equal(a, [3, 4]);
  assert.throws(() => native.run(a, 3), { code: 'KERNEL_EXECUTION_FAILED' }); equal(a, [3, 4]);
  native.run(a, 2); equal(a, [5, 6]);
  const receiver = { tag: 1 }; let calls = 0;
  const kernel = engine(artifact, { fallback(data, n) { assert.equal(this, receiver); calls++; return root(data, n); } });
  kernel.run.call(receiver, a, 3); equal(a, [8, 9]); assert.equal(calls, 1);
});

test('length-only helper inputs need no memory copy; array identities never escape', () => {
  function length(a) { return a.length; }
  function root(a, b) { let result = 0; for (let i = 0; i < 2; i++) result += length(a) + length(b); return result; }
  const artifact = compile(root, [length], ['f32[]', 'f64[]']);
  assert.ok(artifact.manifest.parameters.every(p => !p.read && !p.write));
  const kernel = engine(artifact); assert.equal(kernel.run(new Float32Array(100), new Float64Array(200)), 600);
  assert.equal(kernel.diagnostics.copiedBytes, 0);
});

test('array parameters shadowed by lexical locals remain scalar, with TDZ refusal before initialization', () => {
  function helper(a) { let total = 0; for (let i = 0; i < a.length; i++) { const a = 2; total += a; } return total + a[0]; }
  function root(a) { return helper(a); }
  const kernel = engine(compile(root, [helper], ['f64[]'])); assert.equal(kernel.run(new Float64Array([3, 4])), 7);
  const bad = new Map([['helper', 'function helper(a){ { const b = a[0]; let a = 2; return b + a; } }']]);
  assert.throws(() => compileNumericKernel(root.toString(), { parameterTypes: ['f64[]'], generalControl: true, helperSources: bad }), { code: 'KERNEL_NOT_CLOSED' });
});

test('mixed writable storage aliases retain JavaScript instead of reinterpreting native floats', () => {
  function helper(a, b) { a[0] = 1 / 3; b[0] += 1; }
  function root(a, b) { for (let i = 0; i < 1; i++) helper(a, b); }
  const artifact = compile(root, [helper], ['f32[]', 'u32[]']);
  const buffer = new ArrayBuffer(4), expected = buffer.slice(0); let calls = 0;
  const kernel = engine(artifact, { fallback(a, b) { calls++; return root(a, b); } });
  kernel.run(new Float32Array(buffer), new Uint32Array(buffer)); root(new Float32Array(expected), new Uint32Array(expected));
  assert.deepEqual(new Uint8Array(buffer), new Uint8Array(expected)); assert.equal(calls, 1); assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('array-reference reassignment, captures, object escape, unsupported calls and void-as-number refuse', () => {
  function root(a) { for (let i = 0; i < a.length; i++) a[i] = helper(a, i); }
  for (const source of [
    'function helper(a,i){return a;}', 'function helper(a,i){a=3;return a;}',
    'function helper(a,i){const b=a;return b[i];}', 'function helper(a,i){return other[i];}',
    'function helper(a,i){return a.slice(i);}', 'function helper(a,i){a[i]=1;}',
    'function helper(a,i){if(i)return 1;return;}', 'function helper(a,i){return helper(a,i);}',
    'function helper(a,i){return nested(a,i);} function nested(a,i){return a[i];}',
    'function helper(a,i){return a[i]=2;}', 'function helper(a,i){return a[i]++;}',
  ]) assert.throws(() => compileNumericKernel(root.toString(), { parameterTypes: ['f64[]'], generalControl: true,
    helperSources: new Map([['helper', source]]) }), { code: 'KERNEL_NOT_CLOSED' }, source);
});

test('array helpers use the checked ABI, while explicit checked-index opt-out remains a refusal', () => {
  const options = { parameterTypes: ['f32[]', 'f64'], helperSources: sources([get, put]) };
  const artifact = compileNumericCandidate(update.toString(), options);
  assert.equal(artifact.manifest.version, 7);
  const kernel = engine(artifact), data = new Float32Array([1, 2]); kernel.run(data, 2); equal(data, [2, 4]);
  assert.throws(() => compileNumericCandidate(update.toString(), { ...options, checkedIndexing: false }), { code: 'KERNEL_NOT_CLOSED' });
});
