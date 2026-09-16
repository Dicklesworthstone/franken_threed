import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel, NumericKernelCompileError, NUMERIC_KERNEL_SECTION } from './numeric_kernel.mjs';

// Runs the emitted binary in the actual Wasm engine; no interpreter or mocked kernel.
function execute(source, types, args) {
  const artifact = compileNumericKernel(source, { parameterTypes: types });
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  const module = new WebAssembly.Module(artifact.wasm);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  const { memory, run } = new WebAssembly.Instance(module).exports;
  const count = args[artifact.manifest.boundParameter].length;
  const arrays = types.filter(type => type === 'f64[]').length;
  const pages = Math.ceil(count * arrays * 8 / 65536);
  if (pages > 1) memory.grow(pages - 1);
  let cursor = 0;
  const views = [];
  const values = args.map((arg, index) => {
    if (types[index] === 'f64') return arg;
    const ptr = cursor;
    const view = new Float64Array(memory.buffer, ptr, count);
    view.set(arg.subarray(0, count));
    views.push({ index, view });
    cursor += count * 8;
    return ptr;
  });
  assert.equal(run(...values, count), undefined);
  return { artifact, arrays: views.map(({ view }) => Float64Array.from(view)) };
}

function compare(reference, types, args) {
  const expected = args.map(value => value instanceof Float64Array ? value.slice() : value);
  reference(...expected);
  const { arrays } = execute(reference.toString(), types, args);
  const outputs = expected.filter((_, index) => types[index] === 'f64[]');
  arrays.forEach((array, index) => {
    assert.equal(array.length, outputs[index].length);
    array.forEach((value, i) => assert.ok(Object.is(value, outputs[index][i]),
      `array ${index}, index ${i}: ${value} !== ${outputs[index][i]}`));
  });
}

test('emits an import-free deterministic module with an embedded matching ABI', () => {
  const source = 'export function integrate(x, v, dt) { for (let i=0; i<x.length; i++) { x[i] += v[i]*dt; } }';
  const options = { parameterTypes: ['f64[]', 'f64[]', 'f64'], sourceName: 'scene/update.mjs' };
  const a = compileNumericKernel(source, options);
  const b = compileNumericKernel(source, options);
  assert.deepEqual(a.wasm, b.wasm);
  const module = new WebAssembly.Module(a.wasm);
  const embedded = WebAssembly.Module.customSections(module, NUMERIC_KERNEL_SECTION);
  assert.equal(embedded.length, 1);
  assert.deepEqual(JSON.parse(new TextDecoder().decode(embedded[0])), a.manifest);
  assert.equal(a.manifest.parameters[0].write, true);
  assert.equal(a.manifest.parameters[1].write, false);
  assert.equal(a.manifest.automaticRouteAdmission, false);
  assert.equal(Object.isFrozen(a.manifest.parameters[0]), true);
});

test('lowers real per-frame position and velocity updates with f64 temporaries', () => {
  function integrate(x, v, dt, acceleration) {
    for (let i = 0; i < x.length; ++i) {
      const nextVelocity = v[i] + acceleration * dt;
      x[i] += (v[i] + nextVelocity) * (dt / 2);
      v[i] = nextVelocity;
    }
  }
  compare(integrate, ['f64[]', 'f64[]', 'f64', 'f64'], [
    new Float64Array([0, 1, -2, 1e6]), new Float64Array([5, -3, 4, 0]), 1 / 60, -9.81,
  ]);
});

test('preserves operation order, compound stores, index conversion, and unary operators', () => {
  function update(x, y, a, b) {
    for (let i = 0; i < x.length; i++) {
      let delta = x[i] - y[i];
      x[i] = -((delta * a) / b) + i;
      y[i] += +x[i];
      y[i] -= x[i] / 3;
      y[i] *= a;
      y[i] /= b;
    }
  }
  compare(update, ['f64[]', 'f64[]', 'f64', 'f64'], [
    new Float64Array([2, -9, 0, 1e20]), new Float64Array([3, 1, -0, -1e20]), -0.25, 7,
  ]);
});

test('retains signed zero, infinities and NaN numeric semantics without reassociation', () => {
  function arithmetic(x, divisor) {
    for (let i = 0; i < x.length; i++) {
      x[i] = -(x[i] / divisor);
    }
  }
  const values = new Float64Array([0, -0, 1, -1, Infinity, -Infinity, NaN, 1e-300]);
  for (const divisor of [0, -0, Infinity, -Infinity, NaN, 1e100]) {
    compare(arithmetic, ['f64[]', 'f64'], [values, divisor]);
  }
  const result = execute('function f(x) { for (let i=0; i<x.length; i++) { x[i] = (1e16 + -1e16) + 1; } }',
    ['f64[]'], [new Float64Array(1)]);
  assert.equal(result.arrays[0][0], 1);
});

test('matches native JavaScript across deterministic randomized inputs and multiple memory pages', () => {
  function step(x, y, dt) {
    for (let i = 0; i < x.length; i++) {
      const sum = x[i] + y[i];
      x[i] = sum * dt - y[i] / (dt + 1);
      y[i] = sum / (i + 1);
    }
  }
  let state = 123456789;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return (state / 2 ** 32 - 0.5) * 1e6; };
  for (const size of [0, 1, 7, 257, 8193]) {
    compare(step, ['f64[]', 'f64[]', 'f64'], [
      Float64Array.from({ length: size }, random), Float64Array.from({ length: size }, random), 0.125,
    ]);
  }
});

test('supports more than 127 locals and instructions with multi-byte binary lengths', () => {
  const locals = Array.from({ length: 140 }, (_, i) => `const t${i} = ${i} + x[i];`).join('\n');
  const source = `function f(x) { for (let i=0; i<x.length; i++) { ${locals} x[i] = t139; } }`;
  assert.deepEqual([...execute(source, ['f64[]'], [new Float64Array([2, 3])]).arrays[0]], [141, 142]);
});

test('supports a non-first array loop bound and a single-statement loop body', () => {
  const source = 'function f(scale, x) { for (let i=0; i<x.length; i++) x[i] *= scale; }';
  const result = execute(source, ['f64', 'f64[]'], [2.5, new Float64Array([2, 4])]);
  assert.equal(result.artifact.manifest.boundParameter, 1);
  assert.deepEqual([...result.arrays[0]], [5, 10]);
});

test('rejects unclosed and effectful source before producing executable code', async t => {
  const refused = [
    'function f(x) { for (let i=0; i<x.length; i++) { x[i] = Math.sin(x[i]); } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i] = external; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i] = x[i+1]; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i] %= 2; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i] = "1" + x[i]; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { const t=t+1; x[i]=t; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { const t=later; const later=1; x[i]=t; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i]=this.value; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i]++; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { while (i) x[i]=1; } }',
    'function f(x) { for (let i=0; i<x.length; i++) { x[i]=1; return; } }',
    'function f(x) { sideEffect(); for (let i=0; i<x.length; i++) x[i]=1; }',
    'function f(x) { for (let i=1; i<x.length; i++) x[i]=1; }',
    'function f(x) { for (let i=0; i<=x.length; i++) x[i]=1; }',
    'function f(x) { for (let i=0; i<x.length; i+=2) x[i]=1; }',
    'async function f(x) { for (let i=0; i<x.length; i++) x[i]=1; }',
    'function f(x=[]) { for (let i=0; i<x.length; i++) x[i]=1; }',
    'function f(x) { for (let i=0; i<x.length; i++) { const nothing=1; } }',
  ];
  for (const source of refused) await t.test(source, () => {
    assert.throws(() => compileNumericKernel(source, { parameterTypes: ['f64[]'] }), NumericKernelCompileError);
  });
});

test('validates ABI, source bounds and refusal source locations', () => {
  const source = 'function f(x) { for (let i=0; i<x.length; i++) x[i]=1; }';
  for (const options of [{}, { parameterTypes: ['i32[]'] }, { parameterTypes: [] },
    { parameterTypes: ['f64[]'], maxMemoryPages: 0 }, { parameterTypes: ['f64[]'], maxMemoryPages: 16385 }]) {
    assert.throws(() => compileNumericKernel(source, options), /INVALID_KERNEL_ABI/);
  }
  assert.throws(() => compileNumericKernel(' '.repeat(65537)), /INVALID_KERNEL_SOURCE/);
  assert.throws(() => compileNumericKernel('function {'), /INVALID_KERNEL_SOURCE/);
  assert.throws(() => compileNumericKernel('function f(x) { for (let i=0; i<x.length; i++) x[i]=unknown; }',
    { parameterTypes: ['f64[]'] }), error => error.span.line === 1 && error.span.end > error.span.start);
});
