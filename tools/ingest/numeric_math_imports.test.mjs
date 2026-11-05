import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const compile = source => compileNumericKernel(source, { parameterTypes: ['f64[]'], allowMath: true });
const original = source => Function(`'use strict'; return (${source});`)();
const host = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm, {
  resolveMath: () => Math, ...options,
});
const source = 'function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sin(a[i])+Math.cos(a[i]); }';
const artifact = compile(source);

function u32(value) {
  const bytes = [];
  do {
    const byte = value & 127;
    value >>>= 7;
    bytes.push(byte | (value ? 128 : 0));
  } while (value);
  return bytes;
}

// Change one bounded section while leaving the executable otherwise intact.
function editSection(bytes, section, edit) {
  let offset = 8;
  const read = () => {
    let value = 0, shift = 0, byte;
    do {
      byte = bytes[offset++];
      value |= (byte & 127) << shift;
      shift += 7;
    } while (byte & 128);
    return value;
  };
  while (offset < bytes.length) {
    const start = offset, id = bytes[offset++], length = read(), end = offset + length;
    if (id === section) {
      const content = edit(bytes.slice(offset, end));
      return new Uint8Array([...bytes.slice(0, start), id, ...u32(content.length), ...content, ...bytes.slice(end)]);
    }
    offset = end;
  }
  throw new Error(`Missing section ${section}`);
}

function editManifest(bytes, edit) {
  return editSection(bytes, 0, content => {
    // The compiler's short, fixed custom-section name has a one-byte length.
    const start = content[0] + 1;
    assert.equal(new TextDecoder().decode(content.slice(1, start)), 'f3d.numeric-kernel');
    const manifest = JSON.parse(new TextDecoder().decode(content.slice(start)));
    edit(manifest);
    return new Uint8Array([...content.slice(0, start), ...new TextEncoder().encode(JSON.stringify(manifest))]);
  });
}

test('native Math import requirements are frozen and match actual imports', () => {
  const kernel = host(artifact);
  assert.deepEqual(kernel.manifest.mathImports, ['cos', 'sin']);
  assert.ok(Object.isFrozen(kernel.manifest.mathImports));
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), [
    { module: 'f3d.math', name: 'cos', kind: 'function' },
    { module: 'f3d.math', name: 'sin', kind: 'function' },
  ]);
  const actual = new Float64Array([-0, 0.5, Infinity, NaN]);
  const expected = actual.slice();
  original(source)(expected);
  kernel.run(actual);
  assert.deepEqual(actual, expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

for (const [name, edit] of [
  ['missing', m => { delete m.mathImports; }],
  ['empty', m => { m.mathImports = []; }],
  ['unsorted', m => { m.mathImports.reverse(); }],
  ['duplicate', m => { m.mathImports = ['cos', 'cos']; }],
  ['unlisted', m => { m.mathImports = ['cos', 'random']; }],
  ['unguarded', m => { m.mathIntrinsics = ['cos']; }],
  ['wrong semantics', m => { m.numericSemantics = 'f64-operator-order'; }],
]) test(`rejects ${name} native Math import metadata before instantiation`, () => {
  const wasm = editManifest(artifact.wasm, edit);
  assert.ok(WebAssembly.validate(wasm));
  assert.throws(() => host({ wasm }), { code: 'KERNEL_ABI_MISMATCH' });
});

for (const [from, to] of [['f3d.math', 'f3d.evil'], ['sin', 'tan']])
  test(`rejects an actual import changed from ${from} to ${to}`, () => {
    const wasm = editSection(artifact.wasm, 2, bytes => new TextEncoder().encode(
      new TextDecoder().decode(bytes).replaceAll(from, to)));
    assert.ok(WebAssembly.validate(wasm));
    assert.throws(() => host({ wasm }), { code: 'KERNEL_ABI_MISMATCH' });
  });

test('old import-free artifacts cannot claim an absent native Math import', () => {
  const legacy = compile('function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }');
  assert.equal(legacy.manifest.mathImports, undefined);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(legacy.wasm)), []);
  const wasm = editManifest(legacy.wasm, m => {
    m.mathIntrinsics = ['sin'];
    m.mathImports = ['sin'];
  });
  assert.throws(() => host({ wasm }), { code: 'KERNEL_ABI_MISMATCH' });
});

test('changed native Math falls back once per call and recovers when restored', () => {
  const body = 'function update(a) { for(let i=0;i<a.length;i++) a[i]+=1; for(let i=0;i<a.length;i++) a[i]=Math.sin(a[i]); }';
  const kernel = host(compile(body), { fallback: original(body) });
  const saved = Object.getOwnPropertyDescriptor(Math, 'sin');
  let calls = 0;
  try {
    Math.sin = x => { calls++; return x + 10; };
    const values = new Float64Array([2, 3]);
    kernel.run(values);
    assert.deepEqual(values, new Float64Array([13, 14]));
    assert.equal(calls, 2);
    assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_MATH_BINDING');
  } finally { Object.defineProperty(Math, 'sin', saved); }
  const values = new Float64Array([2, 3]), expected = values.slice();
  original(body)(expected);
  kernel.run(values);
  assert.deepEqual(values, expected);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('throwing native Math getter is read only by the original source', () => {
  const body = 'function update(a) { for(let i=0;i<a.length;i++) a[i]+=1; for(let i=0;i<a.length;i++) a[i]=Math.sin(a[i]); }';
  const kernel = host(compile(body), { fallback: original(body) });
  const saved = Object.getOwnPropertyDescriptor(Math, 'sin'), sentinel = {};
  let reads = 0;
  const values = new Float64Array([2, 3]);
  try {
    Object.defineProperty(Math, 'sin', { configurable: true, get() { reads++; throw sentinel; } });
    assert.throws(() => kernel.run(values), error => error === sentinel);
  } finally { Object.defineProperty(Math, 'sin', saved); }
  assert.equal(reads, 1);
  assert.deepEqual(values, new Float64Array([3, 4]));
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('native imports require the original lexical Math without trapping replacements', () => {
  let gets = 0, descriptors = 0;
  const replacement = new Proxy(Math, {
    get(target, key) { gets++; return target[key]; },
    getOwnPropertyDescriptor(target, key) { descriptors++; return Object.getOwnPropertyDescriptor(target, key); },
  });
  const fn = Function('Math', `return (${source});`)(replacement);
  const kernel = host(artifact, { resolveMath: () => replacement, fallback: fn });
  const values = new Float64Array([1, 2]);
  kernel.run(values);
  assert.equal(gets, 4);
  assert.equal(descriptors, 0);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  const unresolved = instantiateNumericKernel(artifact.wasm);
  assert.throws(() => unresolved.run(values), { code: 'KERNEL_MATH_BINDING' });
});

test('revalidates imported native Math after memory growth, before publication', () => {
  const body = 'function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sin(a[i]); }';
  const kernel = host(compile(body), { fallback: original(body) });
  const grow = WebAssembly.Memory.prototype.grow, sin = Math.sin;
  const values = new Float64Array(10000).fill(2);
  let calls = 0;
  try {
    WebAssembly.Memory.prototype.grow = function (...args) {
      Math.sin = x => { calls++; return x + 20; };
      return Reflect.apply(grow, this, args);
    };
    kernel.run(values);
  } finally {
    WebAssembly.Memory.prototype.grow = grow;
    Math.sin = sin;
  }
  assert.ok(values.every(value => value === 22));
  assert.equal(calls, values.length);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
});

test('pre-bootstrap replacements and getters are never installed as Wasm imports', async () => {
  const runtime = await readFile(new URL('./numeric_kernel_runtime.mjs', import.meta.url), 'utf8');
  const url = `data:text/javascript;base64,${Buffer.from(runtime).toString('base64')}`;
  const saved = Object.getOwnPropertyDescriptor(Math, 'sin');
  let reads = 0, calls = 0;
  try {
    Object.defineProperty(Math, 'sin', { configurable: true, get() { reads++; return saved.value; } });
    const getterRuntime = await import(`${url}#getter`);
    assert.throws(() => getterRuntime.instantiateNumericKernel(artifact.wasm), { code: 'KERNEL_MATH_BINDING' });
    Object.defineProperty(Math, 'sin', { configurable: true, value() { calls++; return 0; } });
    const patchedRuntime = await import(`${url}#replacement`);
    assert.throws(() => patchedRuntime.instantiateNumericKernel(artifact.wasm), { code: 'KERNEL_MATH_BINDING' });
  } finally { Object.defineProperty(Math, 'sin', saved); }
  assert.equal(reads, 0);
  assert.equal(calls, 0);
});
