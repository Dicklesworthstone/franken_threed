import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';

function project(position, matrix) {
  const m00 = matrix[0], m01 = matrix[4], m02 = matrix[8], tx = matrix[12];
  const m10 = matrix[1], m11 = matrix[5], m12 = matrix[9], ty = matrix[13];
  const m20 = matrix[2], m21 = matrix[6], m22 = matrix[10], tz = matrix[14];
  for (let i = 0; i < position.length; i += 3) {
    const x = position[i], y = position[i + 1], z = position[i + 2];
    const w = 1 / (matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15]);
    position[i] = (m00 * x + m01 * y + m02 * z + tx) * w;
    position[i + 1] = (m10 * x + m11 * y + m12 * z + ty) * w;
    position[i + 2] = (m20 * x + m21 * y + m22 * z + tz) * w;
  }
}
function affine(a, m) { const scale = m[0]; for (let i = 0; i < a.length; i++) a[i] = a[i] * scale + m[1]; }
function compiled(fn, parameterTypes, options = {}) {
  const artifact = compileNumericKernel(fn.toString(), { parameterTypes });
  assert.ok(WebAssembly.validate(artifact.wasm));
  return instantiateNumericKernel(artifact.wasm, { fallback: fn, ...options });
}
const matrixValues = [1, 0.001, 0, 0.00001, -0.001, 1, 0.001, 0, 0.001, 0, 1, 0.00002, 0.0001, -0.0001, 0.0002, 1];

for (const [ArrayType, type] of [[Float32Array, 'f32[]'], [Float64Array, 'f64[]']]) {
  test(`${type} geometry projects through a 16-element f64 matrix with exact operator order`, () => {
    const kernel = compiled(project, [type, 'f64[]']);
    const actual = ArrayType.from({ length: 30000 }, (_, i) => (i % 37 - 18) / 11);
    const expected = actual.slice(), matrix = new Float64Array(matrixValues);
    for (let frame = 0; frame < 120; frame++) {
      matrix[12] = frame / 100000; // Uniforms are fresh on every invocation.
      project(expected, matrix); kernel.run(actual, matrix); assert.deepEqual(actual, expected);
    }
    assert.equal(kernel.manifest.version, 4);
    assert.deepEqual(kernel.manifest.parameters[1].access, { indexed: false, minimumLength: 16 });
    assert.equal(kernel.diagnostics.wasmCalls, 120);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
    assert.equal(kernel.diagnostics.copiedBytes, 120 * (30000 * ArrayType.BYTES_PER_ELEMENT * 2 + 16 * 8));
  });
}

test('scalar setup stays f64 even when both input and output storage are f32', () => {
  function update(a, m, one) { const factor = m[0] + one - m[0]; for (let i = 0; i < a.length; i++) a[i] = factor; }
  const kernel = compiled(update, ['f32[]', 'f32[]', 'f64']);
  const a = new Float32Array(3); kernel.run(a, new Float32Array([16777216]), 1);
  assert.deepEqual([...a], [1, 1, 1]); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('direct fixed reads without a prelude do not require vertex-sized uniform arrays', () => {
  function update(a, m) { for (let i = 0; i < a.length; i++) a[i] += m[15]; }
  const kernel = compiled(update, ['f64[]', 'f32[]']);
  const a = new Float64Array(1000), m = new Float32Array(16).fill(2); kernel.run(a, m);
  assert.ok(a.every(x => x === 2)); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('pre-loop declarations share lexical scope and nested loop bindings retain their TDZ', () => {
  function update(a, scale) { const x = scale + 1, y = x * 2; for (let i = 0; i < a.length; i++) {
    a[i] = y; { const x = 3; a[i] += x; }
  } }
  const kernel = compiled(update, ['f64[]', 'f64']);
  const a = new Float64Array(2); kernel.run(a, 2); assert.deepEqual([...a], [9, 9]);
  for (const setup of ['const x = y; const y = 1;', 'const x = x;', 'const x = i;', 'const x = a[i];']) {
    assert.throws(() => compileNumericKernel(`function update(a) { ${setup} for(let i=0;i<a.length;i++) a[i]=x; }`,
      { parameterTypes: ['f64[]'] }));
  }
  assert.throws(() => compileNumericKernel(`function update(a) { const x=1; for(let i=0;i<a.length;i++) {
    a[i]=x; const x=2; } }`, { parameterTypes: ['f64[]'] }), /before initialization/);
});

test('an array used only for its loop bound is not copied or assigned a vertex-sized scratch region', () => {
  function update(bound, out, m) { for (let i = 0; i < bound.length; i++) out[i] = m[0]; }
  const kernel = compiled(update, ['f32[]', 'f64[]', 'f64[]'], { maxMemoryBytes: 65536 });
  const out = new Float64Array(7000);
  kernel.run(new Float32Array(7000), out, new Float64Array([7]));
  assert.ok(out.every(value => value === 7)); assert.equal(kernel.diagnostics.wasmCalls, 1);
  assert.equal(kernel.diagnostics.copiedBytes, 7000 * 8 + 8);
  assert.deepEqual(kernel.manifest.parameters[0].access, { indexed: false, minimumLength: 0 });
});

test('uniform read extents can coexist with streamed reads on a read-only parameter', () => {
  function update(out, input) { for (let i = 0; i < out.length; i++) out[i] = input[i] + input[7]; }
  const kernel = compiled(update, ['f32[]', 'f64[]']);
  const out = new Float32Array(3); kernel.run(out, new Float64Array([1,2,3,4,5,6,7,8]));
  assert.deepEqual([...out], [9,10,11]); assert.equal(kernel.diagnostics.wasmCalls, 1);
  assert.deepEqual(kernel.manifest.parameters[1].access, { indexed: true, minimumLength: 8 });
});

test('undersized uniforms fallback once before any output is published', () => {
  let calls = 0;
  const artifact = compileNumericKernel(affine.toString(), { parameterTypes: ['f32[]', 'f64[]'] });
  const kernel = instantiateNumericKernel(artifact.wasm, { fallback(...args) { calls++; return affine(...args); } });
  const a = new Float32Array([1,2,3]); kernel.run(a, new Float64Array([2]));
  assert.ok(a.every(Number.isNaN)); assert.equal(calls, 1);
  assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_LENGTH');
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('proxy/getter uniforms are evaluated only by the original function in source order', () => {
  const events = [];
  const m = new Proxy({}, { get(_target, key) { events.push(key); return key === '0' ? 2 : 3; } });
  const kernel = compiled(affine, ['f32[]', 'f64[]']); const a = new Float32Array([1,2,3]);
  kernel.run(a, m);
  assert.deepEqual(events, ['0','1','1','1']); assert.deepEqual([...a], [5,7,9]);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
});

test('overlapping uniform/output prefixes use original sequential behavior', () => {
  function update(a, m) { for (let i = 0; i < a.length; i++) a[i] += m[0]; }
  const kernel = compiled(update, ['f32[]', 'f32[]']);
  const actual = new Float32Array([1,2,3]), expected = actual.slice();
  kernel.run(actual, actual.subarray(0,1)); update(expected, expected.subarray(0,1));
  assert.deepEqual(actual, expected); assert.deepEqual([...actual], [2,4,5]);
  assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
});

test('unused uniform tails can overlap outputs without false alias failure', () => {
  const buffer = new ArrayBuffer(48), m = new Float64Array(buffer);
  m[0] = 2; m[1] = 3;
  const a = new Float32Array(buffer, 16, 6).fill(1);
  const kernel = compiled(affine, ['f32[]', 'f64[]']); kernel.run(a, m);
  assert.deepEqual([...a], [5,5,5,5,5,5]); assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('zero records still execute original prelude effects when guards require fallback', () => {
  const events = [], kernel = compiled(affine, ['f32[]', 'f64[]']);
  const m = { get 0() { events.push('setup'); return 2; }, get 1() { throw Error('loop must not run'); } };
  kernel.run(new Float32Array(0), m); assert.deepEqual(events, ['setup']);
  kernel.run(new Float32Array(0), new Float64Array([2,3]));
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('incomplete geometry records, growing scratch memory, and changed uniforms match JS', () => {
  const kernel = compiled(project, ['f32[]', 'f64[]']); const matrix = new Float64Array(matrixValues);
  for (const length of [0,3,30000,6,4,9]) {
    const a = new Float32Array(length).fill(1), expected = a.slice();
    matrix[12] += 0.01; kernel.run(a, matrix); project(expected, matrix); assert.deepEqual(a, expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 5); assert.equal(kernel.diagnostics.fallbackCalls, 1);
});

test('fixed writes, mutable uniforms, unbounded indices and setup effects stay refused', () => {
  const bodies = [
    'for(let i=0;i<a.length;i++) a[0]=1;',
    'for(let i=0;i<a.length;i++) a[i]=a[0];',
    'const x=sideEffect(); for(let i=0;i<a.length;i++) a[i]=x;',
    'for(let i=0;i<a.length;i++) a[i]=m[65536];',
    'for(let i=0;i<a.length;i++) a[i]=m[-1];',
    'for(let i=0;i<a.length;i++) a[i]=m[1.5];',
    'for(let i=0;i<a.length;i++) a[i]=m[index];',
  ];
  for (const body of bodies) assert.throws(() => compileNumericKernel(`function update(a,m) { ${body} }`,
    { parameterTypes: ['f64[]','f64[]'] }));
});

test('the maximum fixed index uses bounded allocation and unsigned Wasm offsets', () => {
  function update(a, m) { for (let i = 0; i < a.length; i++) a[i] = m[65535]; }
  const kernel = compiled(update, ['f64[]','f64[]']); const m = new Float64Array(65536); m[65535] = 7;
  const a = new Float64Array(1); kernel.run(a,m); assert.equal(a[0],7); assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('automatic specialization admits four bounded stream/uniform layouts and executes each natively', async t => {
  const native = WebAssembly.Instance; let calls = 0, instances = 0;
  t.after(() => { WebAssembly.Instance = native; });
  WebAssembly.Instance = function (...args) {
    const instance = Reflect.construct(native, args); instances++;
    return { exports: { memory: instance.exports.memory, run(...xs) { calls++; return instance.exports.run(...xs); } } };
  };
  const source = `${project.toString()}\nexport { project }; export function tick(a,m) { return project(a,m); }`;
  const result = specializeNumericModule(source);
  assert.equal(result.report.compiledKernels, 1); assert.equal(result.report.candidates[0].variants.length, 4);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'f3d-uniform-'));
  for (const name of ['numeric_dispatch.mjs','numeric_kernel_runtime.mjs']) fs.copyFileSync(new URL(name,import.meta.url),path.join(dir,name));
  fs.writeFileSync(path.join(dir,'entry.mjs'),result.code);
  const module = await import(pathToFileURL(path.join(dir,'entry.mjs')));
  for (let frame=0;frame<12;frame++) for (const A of [Float32Array,Float64Array]) for (const M of [Float32Array,Float64Array]) {
    const a = new A([1,2,3,4,5,6]), expected = a.slice(), m = new M(matrixValues);
    module.tick(a,m); project(expected,m); assert.deepEqual(a,expected);
  }
  assert.equal(calls,48); assert.equal(instances,4);
  assert.equal(module.project.toString(),project.toString());
});

test('setup without uniforms retains only two variants and conservative mixed-stream fallback', () => {
  function update(a,b,dt) { const factor=dt*2; for(let i=0;i<a.length;i++) a[i]+=b[i]*factor; }
  const source = `${update.toString()}\nexport const tick=(a,b,dt)=>update(a,b,dt);`;
  const result = specializeNumericModule(source);
  assert.equal(result.report.candidates[0].variants.length,2);
});

test('uniform access manifests are deeply immutable and reject malformed extents', t => {
  const artifact = compileNumericKernel(affine.toString(), { parameterTypes: ['f32[]','f64[]'] });
  assert.ok(Object.isFrozen(artifact.manifest.parameters[1].access));
  const customSections = WebAssembly.Module.customSections;
  t.after(() => { WebAssembly.Module.customSections = customSections; });
  for (const mutate of [
    m => { m.parameters[1].access.minimumLength = -1; },
    m => { m.parameters[1].access.minimumLength = 65537; },
    m => { m.parameters[1].access.indexed = 1; },
    m => { delete m.parameters[1].access; },
    m => { m.parameters[0].access.indexed = false; },
    m => { m.parameters[0].access.minimumLength = 1; },
    m => { m.loopStride = 0; },
  ]) {
    const invalid = structuredClone(artifact.manifest); mutate(invalid);
    WebAssembly.Module.customSections = () => [new TextEncoder().encode(JSON.stringify(invalid)).buffer];
    assert.throws(() => instantiateNumericKernel(artifact.wasm), /KERNEL_ABI_MISMATCH/);
  }
  WebAssembly.Module.customSections = customSections;
  const kernel = instantiateNumericKernel(artifact.wasm);
  assert.ok(Object.isFrozen(kernel.manifest.parameters[1].access));
});
