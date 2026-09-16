import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { instantiateNumericKernel, NumericKernelGuardError } from './numeric_kernel_runtime.mjs';

function integrate(position, velocity, dt) {
  for (let i = 0; i < position.length; i++) {
    position[i] += velocity[i] * dt;
    velocity[i] = velocity[i] - dt;
  }
}
const types = ['f64[]', 'f64[]', 'f64'];
const artifact = compileNumericKernel(integrate.toString(), { parameterTypes: types });
const kernel = options => instantiateNumericKernel(artifact.wasm, options);

test('executes real Wasm repeatedly and publishes into original array identities', () => {
  const engine = kernel();
  const position = new Float64Array([1, 2, 3]);
  const velocity = new Float64Array([4, 5, 6]);
  const expectedPosition = position.slice(), expectedVelocity = velocity.slice();
  for (let frame = 0; frame < 100; frame++) {
    const dt = frame / 1000;
    assert.equal(engine.run(position, velocity, dt), undefined);
    integrate(expectedPosition, expectedVelocity, dt);
  }
  assert.deepEqual(position, expectedPosition);
  assert.deepEqual(velocity, expectedVelocity);
  assert.equal(engine.diagnostics.wasmCalls, 100);
  assert.equal(engine.diagnostics.fallbackCalls, 0);
  assert.equal(engine.diagnostics.copiedBytes, 100 * 3 * 8 * 4);
});

test('reacquires memory views after growth and accepts changing batch sizes', () => {
  const engine = kernel();
  for (const length of [1, 8193, 0, 3, 20000, 16]) {
    const a = new Float64Array(length).fill(2), b = new Float64Array(length).fill(3);
    engine.run(a, b, 0.5);
    assert.ok(a.every(value => value === 3.5));
    assert.ok(b.every(value => value === 2.5));
  }
  assert.ok(engine.diagnostics.memoryBytes >= 20000 * 16);
  assert.equal(engine.diagnostics.wasmCalls, 6);
});

test('handles disjoint views of one buffer and leaves unused tails untouched', () => {
  const memory = new ArrayBuffer(8 * 12);
  const position = new Float64Array(memory, 8, 3);
  const velocity = new Float64Array(memory, 8 * 5, 5);
  position.set([1, 2, 3]); velocity.set([10, 20, 30, 99, 98]);
  const engine = kernel();
  engine.run(position, velocity, 2);
  assert.deepEqual([...position], [21, 42, 63]);
  assert.deepEqual([...velocity], [8, 18, 28, 99, 98]);
  assert.deepEqual([...new Float64Array(memory, 0, 1)], [0]);
});

test('aliased views deopt before writes, retaining loop-carried alias semantics', () => {
  const source = 'function shift(x, y, dt) { for(let i=0; i<x.length; i++) x[i] += y[i]*dt; }';
  function shift(x, y, dt) { for (let i = 0; i < x.length; i++) x[i] += y[i] * dt; }
  const bytes = compileNumericKernel(source, { parameterTypes: types }).wasm;
  const storage = new Float64Array([1, 2, 3, 4]);
  let calls = 0;
  const engine = instantiateNumericKernel(bytes, { fallback(...args) {
    calls++;
    assert.deepEqual([...storage], [1, 2, 3, 4]);
    return shift(...args);
  } });
  engine.run(storage.subarray(1), storage.subarray(0, 3), 1);
  assert.deepEqual([...storage], [1, 3, 6, 10]);
  assert.equal(calls, 1);
  assert.equal(engine.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
  assert.equal(engine.diagnostics.wasmCalls, 0);
});

test('all argument guards run before publication, including late invalid arguments', async t => {
  const cases = [
    { value: new Float32Array(3), code: 'KERNEL_ARRAY_TYPE' },
    { value: [1, 2, 3], code: 'KERNEL_ARRAY_TYPE' },
    { value: null, code: 'KERNEL_ARRAY_TYPE' },
    { value: new (class extends Float64Array {})(3), code: 'KERNEL_ARRAY_TYPE' },
    { value: vm.runInNewContext('new Float64Array(3)'), code: 'KERNEL_ARRAY_TYPE' },
    { value: new Float64Array(2), code: 'KERNEL_ARRAY_LENGTH' },
    { value: new Float64Array(new SharedArrayBuffer(24)), code: 'KERNEL_ARRAY_OWNERSHIP' },
    { value: new Float64Array(new ArrayBuffer(24, { maxByteLength: 48 })), code: 'KERNEL_ARRAY_OWNERSHIP' },
  ];
  const detached = new Float64Array(3);
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  cases.push({ value: detached, code: 'KERNEL_ARRAY_OWNERSHIP' });
  for (const { value, code } of cases) await t.test(code, () => {
    const position = new Float64Array([1, 2, 3]);
    assert.throws(() => kernel().run(position, value, 0.5), error => error instanceof NumericKernelGuardError && error.code === code);
    assert.deepEqual([...position], [1, 2, 3]);
  });
});

test('does not execute proxy traps or spoofed length getters during guard evaluation', () => {
  const engine = kernel();
  const a = new Float64Array([1, 2, 3]);
  let traps = 0;
  const b = new Proxy(new Float64Array(3), {
    get() { traps++; throw new Error('get trap'); },
    getPrototypeOf() { traps++; throw new Error('prototype trap'); },
    getOwnPropertyDescriptor() { traps++; throw new Error('descriptor trap'); },
  });
  assert.throws(() => engine.run(a, b, 1), /KERNEL_ARRAY_TYPE/);
  assert.equal(traps, 0);
  let reads = 0;
  Object.defineProperty(a, 'length', { get() { reads++; return 3; } });
  assert.throws(() => engine.run(a, new Float64Array(3), 1), /KERNEL_MUTABLE_LENGTH/);
  assert.equal(reads, 0);
});

test('numeric objects deopt without an extra coercion and fallback keeps receiver and result', () => {
  let conversions = 0, calls = 0;
  const dt = { valueOf() { conversions++; return 0.5; } };
  const context = { marker: true };
  const engine = kernel({ fallback(...args) {
    assert.equal(this, context);
    calls++;
    integrate(...args);
    return 'retained-result';
  } });
  const a = new Float64Array([1, 2]), b = new Float64Array([4, 6]);
  assert.equal(engine.run.call(context, a, b, dt), 'retained-result');
  assert.equal(conversions, 4); // Two source coercions per iteration; none from guards.
  assert.equal(calls, 1);
  assert.deepEqual([...a], [3, 5]);
  assert.equal(engine.diagnostics.lastGuardFailure, 'KERNEL_SCALAR_TYPE');
});

test('fallback exceptions propagate unchanged and are never retried', () => {
  const sentinel = new Error('source failure');
  let calls = 0;
  const engine = kernel({ fallback() { calls++; throw sentinel; } });
  assert.throws(() => engine.run([], [], 1), error => error === sentinel);
  assert.equal(calls, 1);
  assert.equal(engine.diagnostics.fallbackCalls, 1);
});

test('memory budget refusal is transactional and a later smaller batch still runs', () => {
  const engine = kernel({ maxMemoryBytes: 65536 });
  const a = new Float64Array(5000).fill(2), b = new Float64Array(5000).fill(3);
  assert.throws(() => engine.run(a, b, 1), /KERNEL_MEMORY_LIMIT/);
  assert.ok(a.every(value => value === 2));
  assert.ok(b.every(value => value === 3));
  assert.equal(engine.diagnostics.memoryBytes, 65536);
  const small = new Float64Array([2]);
  engine.run(small, new Float64Array([3]), 1);
  assert.equal(small[0], 5);
  assert.equal(engine.diagnostics.lastGuardFailure, null);
  const unalignedBudget = kernel({ maxMemoryBytes: 70000 });
  assert.throws(() => unalignedBudget.run(new Float64Array(4097), new Float64Array(4097), 1), /KERNEL_MEMORY_LIMIT/);
  assert.equal(unalignedBudget.diagnostics.memoryBytes, 65536);
});

test('ignores overridable array methods and publishes only declared writes', () => {
  const compiled = compileNumericKernel('function fill(x, y) { for(let i=0;i<x.length;i++) x[i]=y[i]+1; }',
    { parameterTypes: ['f64[]', 'f64[]'] });
  const a = new Float64Array([1, 2]), b = new Float64Array([3, 4]);
  a.set = b.subarray = () => { throw new Error('User method must not be invoked'); };
  const engine = instantiateNumericKernel(compiled.wasm);
  engine.run(a, b);
  assert.deepEqual([...a], [4, 5]);
  assert.deepEqual([...b], [3, 4]);
  assert.equal(engine.diagnostics.copiedBytes, 32); // One input prefix + one output prefix.
});

test('zero-sized batches, nonfinite numbers, and signed zero retain numeric semantics', () => {
  const engine = kernel();
  engine.run(new Float64Array(0), new Float64Array(0), NaN);
  const a = new Float64Array([0, -0, Infinity, NaN]);
  const b = new Float64Array([0, -0, -Infinity, 1]);
  const expectedA = a.slice(), expectedB = b.slice();
  integrate(expectedA, expectedB, -0);
  engine.run(a, b, -0);
  a.forEach((value, i) => assert.ok(Object.is(value, expectedA[i])));
  b.forEach((value, i) => assert.ok(Object.is(value, expectedB[i])));
});

test('validates binary ABI and options; disposal releases the private memory reference', () => {
  assert.throws(() => instantiateNumericKernel(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])), /KERNEL_ABI_MISMATCH/);
  assert.throws(() => kernel({ fallback: 1 }), /fallback/);
  for (const maxMemoryBytes of [0, -1, Infinity, 1.5, 2 ** 32]) {
    assert.throws(() => kernel({ maxMemoryBytes }), RangeError);
  }
  const engine = kernel({ fallback: integrate });
  assert.throws(() => kernel().run(new Float64Array(1)), /KERNEL_ARGUMENT_COUNT/);
  engine.dispose(); engine.dispose();
  assert.equal(engine.diagnostics.disposed, true);
  assert.equal(engine.diagnostics.memoryBytes, 0);
  assert.throws(() => engine.run(new Float64Array(1), new Float64Array(1), 1), /KERNEL_DISPOSED/);
  assert.equal(engine.diagnostics.fallbackCalls, 0);
});

test('runtime module imports with DOM, GPU and WebAssembly globals unavailable', async () => {
  const prior = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');
  Object.defineProperty(globalThis, 'WebAssembly', { configurable: true, get() { throw new Error('eager Wasm access'); } });
  try {
    const runtime = await import(`./numeric_kernel_runtime.mjs?cpu-only=${Date.now()}`);
    assert.equal(typeof runtime.instantiateNumericKernel, 'function');
  } finally { Object.defineProperty(globalThis, 'WebAssembly', prior); }
});
