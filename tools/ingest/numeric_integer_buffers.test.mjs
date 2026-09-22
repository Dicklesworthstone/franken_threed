import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';

const TYPES = [[Int8Array, 'i8[]'], [Uint8Array, 'u8[]'], [Uint8ClampedArray, 'u8c[]'],
  [Int16Array, 'i16[]'], [Uint16Array, 'u16[]'], [Int32Array, 'i32[]'], [Uint32Array, 'u32[]']];
const compile = (fn, parameterTypes, options = {}) =>
  compileNumericCandidate(fn.toString(), { parameterTypes, ...options });
const host = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm, options);
function copy(out, input) {
  for (let i = 0; i < input.length; i++) out[i] = input[i];
}
function accumulate(out, input, index) {
  for (let i = 0; i < index.length; i++) {
    out[index[i]] += input[i];
    out[index[i]]++;
    --out[index[i]];
  }
  return out[0];
}

// Test conversion against the engine's typed-element implementation, including
// arbitrary binary64 exponents/payloads. No application source is evaluated by
// the compiler; ordinary JS functions here are independent execution oracles.
const numbers = [NaN, Infinity, -Infinity, -0, 0, Number.MIN_VALUE, -Number.MIN_VALUE,
  Number.MAX_VALUE, -Number.MAX_VALUE];
for (const edge of [0, 1, 127, 128, 254, 255, 256, 32767, 32768, 65535, 65536,
  2 ** 31, 2 ** 32, 2 ** 53, 2 ** 84])
  for (const delta of [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    numbers.push(edge + delta, -edge + delta);
const bits = new DataView(new ArrayBuffer(8));
let seed = 0x4f334431;
const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0);
for (let i = 0; i < 4096; i++) {
  bits.setUint32(0, random()); bits.setUint32(4, random());
  numbers.push(bits.getFloat64(0));
}

for (const [ArrayType, type] of TYPES) {
  test(`${type}: exact stores and signed/unsigned loads across all numeric storage`, () => {
    const artifact = compile(copy, [type, 'f64[]']);
    assert.equal(artifact.manifest.version, 9);
    assert.equal(artifact.manifest.integerSemantics, 'ecmascript-integer-elements-v1');
    assert.ok(WebAssembly.validate(artifact.wasm));
    assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
    assert.deepEqual(compile(copy, [type, 'f64[]']).wasm, artifact.wasm);
    const kernel = host(artifact), input = Float64Array.from(numbers);
    const backing = new ArrayType(input.length + 4).fill(73), out = backing.subarray(2, -2);
    for (let frame = 0; frame < 2; frame++) {
      kernel.run(out, input);
      assert.deepEqual(out, new ArrayType(input));
      input.reverse();
    }
    assert.deepEqual([...backing.subarray(0, 2), ...backing.subarray(-2)], [73, 73, 73, 73]);
    assert.equal(kernel.diagnostics.wasmCalls, 2);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
    for (const [Input, inputType] of [...TYPES, [Float32Array, 'f32[]'], [Float64Array, 'f64[]']]) {
      const source = new Input(numbers), actual = new ArrayType(source.length);
      const k = host(compile(copy, [type, inputType]));
      k.run(actual, source);
      assert.deepEqual(actual, new ArrayType(source));
      assert.equal(k.diagnostics.wasmCalls, 1);
    }
    const widened = new Float64Array(out.length), read = host(compile(copy, ['f64[]', type]));
    read.run(widened, out);
    assert.deepEqual(widened, Float64Array.from(out));
  });

  test(`${type}: every colliding compound store and increment converts before the next read`, () => {
    const index = new Uint32Array([0, 0, 1, 0, 1, 0]);
    const input = new Float64Array([255.5, -1.5, 2 ** 32 - 1, 0.5, NaN, -300.75]);
    const actual = new ArrayType([254, 127, 61]), expected = actual.slice();
    const kernel = host(compile(accumulate, [type, 'f64[]', 'u32[]']));
    for (let frame = 0; frame < 3; frame++) {
      assert.equal(kernel.run(actual, input, index), accumulate(expected, input, index));
      assert.deepEqual(actual, expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 3);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
  });

  test(`${type}: transactional bounds failure, one fallback, then recovery`, () => {
    const artifact = compile(accumulate, [type, 'f64[]', 'f64[]']);
    const input = new Float64Array([2, 3, 4]), index = new Float64Array([0, 99, 1]);
    const actual = new ArrayType([3, 4, 5]), expected = actual.slice();
    assert.throws(() => host(artifact).run(actual, input, index), {code: 'KERNEL_EXECUTION_FAILED'});
    assert.deepEqual(actual, expected);
    const receiver = {}; let calls = 0;
    const k = host(artifact, {fallback(...args) {
      calls++; assert.equal(this, receiver); assert.deepEqual(actual, expected);
      return accumulate(...args);
    }});
    const result = k.run.call(receiver, actual, input, index);
    assert.equal(result, accumulate(expected, input, index));
    assert.deepEqual(actual, expected);
    assert.equal(calls, 1); assert.equal(k.diagnostics.wasmCalls, 0);
    assert.equal(k.diagnostics.copiedBytes, 0);
    index[1] = 2;
    assert.equal(k.run(actual, input, index), accumulate(expected, input, index));
    assert.deepEqual(actual, expected);
    assert.equal(k.diagnostics.wasmCalls, 1); assert.equal(calls, 1);
  });

  test(`${type}: same-type overlapping views preserve ordered writes when explicitly admitted`, () => {
    function chain(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }
    const actual = new ArrayType([1, 2, 3, 4, 5]), expected = actual.slice();
    const k = host(compile(chain, [type, type]), {preserveAliasing: true});
    chain(expected.subarray(1), expected.subarray(0, 4));
    k.run(actual.subarray(1), actual.subarray(0, 4));
    assert.deepEqual(actual, expected);
    assert.equal(k.diagnostics.wasmCalls, 1);
  });
}

test('Uint8ClampedArray half ties round to even, not Math.round or wrapping', () => {
  const out = new Uint8ClampedArray(10);
  host(compile(copy, ['u8c[]', 'f64[]'])).run(out,
    new Float64Array([0.5, 1.5, 2.5, 253.5, 254.5, 255.5, NaN, Infinity, -Infinity, -1]));
  assert.deepEqual([...out], [0, 2, 2, 254, 254, 255, 0, 255, 0, 0]);
});

test('procedural grid indices and a dependent checksum execute in one ordered Wasm call', () => {
  function grid(indices, width) {
    for (let i = 0; i < indices.length; i += 6) {
      const cell = i / 6, row = Math.floor(cell / width), a = cell + row;
      indices[i] = a; indices[i + 1] = a + 1; indices[i + 2] = a + width + 1;
      indices[i + 3] = a + 1; indices[i + 4] = a + width + 2; indices[i + 5] = a + width + 1;
    }
    let checksum = 0;
    for (let i = 0; i < indices.length; i++) checksum += indices[i];
    return checksum;
  }
  for (const [ArrayType, type] of [[Uint16Array, 'u16[]'], [Uint32Array, 'u32[]']]) {
    const k = host(compile(grid, [type, 'f64'], {allowMath: true}), {resolveMath: () => Math});
    for (const width of [1, 100, 257, 4]) {
      const actual = new ArrayType(width * width * 6), expected = actual.slice();
      assert.equal(k.run(actual, width), grid(expected, width));
      assert.deepEqual(actual, expected);
    }
    assert.equal(k.diagnostics.wasmCalls, 4);
    assert.ok(k.diagnostics.memoryBytes > 65536);
    assert.equal(k.manifest.loops.length, 2);
  }
});

test('mixed-type writable overlap stays on JS; native slot guards do not run Proxy traps', () => {
  const artifact = compile(copy, ['u8[]', 'u16[]']);
  const buffer = new ArrayBuffer(8), out = new Uint8Array(buffer), input = new Uint16Array(buffer);
  input.set([257, 258, 259, 260]);
  const expectedBuffer = buffer.slice(0);
  copy(new Uint8Array(expectedBuffer), new Uint16Array(expectedBuffer));
  const k = host(artifact, {fallback: copy, preserveAliasing: true});
  k.run(out, input);
  assert.deepEqual(out, new Uint8Array(expectedBuffer));
  assert.equal(k.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
  let traps = 0;
  const proxy = new Proxy(out, {get() {traps++; throw Error('unexpected trap');}});
  assert.throws(() => host(artifact).run(proxy, input), {code: 'KERNEL_ARRAY_TYPE'});
  assert.equal(traps, 0);
});

test('integer output ownership, lengths, and budgets are checked before publication', () => {
  const artifact = compile(copy, ['u8[]', 'f64[]'], {maxMemoryPages: 1});
  const input = new Float64Array([7, 8]);
  for (const out of [new Uint8Array(new SharedArrayBuffer(2)),
    new Uint8Array(new ArrayBuffer(2, {maxByteLength: 4})), new (class extends Uint8Array {})(2)]) {
    assert.throws(() => host(artifact).run(out, input));
    assert.deepEqual([...out], [0, 0]);
  }
  const detached = new Uint8Array(2);
  structuredClone(detached.buffer, {transfer: [detached.buffer]});
  assert.throws(() => host(artifact).run(detached, input), {code: 'KERNEL_ARRAY_OWNERSHIP'});
  const large = new Uint8Array(65536), k = host(artifact, {fallback: copy});
  k.run(large, input);
  assert.deepEqual([...large.subarray(0, 3)], [7, 8, 0]);
  assert.equal(k.diagnostics.lastGuardFailure, 'KERNEL_MEMORY_LIMIT');
});

// Mutate only the compiler's bounded custom section, keeping the executable
// module valid, to prove old hosts/contracts cannot silently misinterpret v9.
function manifestBytes(artifact, change) {
  const wasm = artifact.wasm, reader = {at: 8};
  const read = () => {let v = 0, shift = 0, b; do {b = wasm[reader.at++]; v += (b & 127) * 2 ** shift; shift += 7;} while (b & 128); return v;};
  let start = -1;
  while (reader.at < wasm.length) {
    const at = reader.at, id = wasm[reader.at++], size = read();
    if (id === 0) {start = at; break;} reader.at += size;
  }
  assert.ok(start > 0);
  const encode = n => {const bytes = []; do {const b = n & 127; n >>>= 7; bytes.push(b | (n ? 128 : 0));} while (n); return bytes;};
  const manifest = structuredClone(artifact.manifest); change(manifest);
  const name = new TextEncoder().encode('f3d.numeric-kernel');
  const payload = [...encode(name.length), ...name, ...new TextEncoder().encode(JSON.stringify(manifest))];
  return new Uint8Array([...wasm.subarray(0, start), 0, ...encode(payload.length), ...payload]);
}

test('integer ABI fails closed for missing semantics, invalid types and forged v7 writes', () => {
  const artifact = compile(copy, ['u32[]', 'f64[]']);
  for (const change of [m => {delete m.integerSemantics;}, m => {m.integerSemantics = 'saturate';},
    m => {m.version = 7; delete m.integerSemantics;}, m => {m.parameters[0].read = false;},
    m => {m.parameters[0].type = 'bigint64[]';}]) {
    assert.throws(() => instantiateNumericKernel(manifestBytes(artifact, change)), {code: 'KERNEL_ABI_MISMATCH'});
  }
  assert.throws(() => compile(copy, ['u8[]', 'f64[]'], {checkedIndexing: false}), {code: 'INVALID_KERNEL_ABI'});
  assert.throws(() => compile(copy, ['bigint64[]', 'f64[]']), {code: 'INVALID_KERNEL_ABI'});
});

test('legacy v1-v7 compiler bytecode remains deterministic and unchanged', () => {
  const cases = [
    ['function f(a,b){for(let i=0;i<a.length;i++)a[i]+=b;}', ['f64[]','f64'], 'bd427a2810ba4c4c92742d3d362943bfb8080fe0710d5b393d1663cf7d789b48'],
    ['function f(a,b){for(let i=0;i<a.length;i++)a[i]+=b;}', ['f32[]','f64'], 'fb2eed4c8a155339dcd62037943da18d05da89309fe43918345104e24afa5b52'],
    ['function f(a,b){for(let i=0;i<a.length;i+=3)a[i]+=b;}', ['f32[]','f64'], '5261d7dde5adf84903d3c8fe7b42250403b8367d885e94c08e11867bb6502225'],
    ['function f(a,b){const c=b;for(let i=0;i<a.length;i++)a[i]*=c;}', ['f32[]','f64'], '7f5f80a8a81444b37ef979e704017ab52f274535906fce4e329623e38936c1e4'],
    ['function f(a){let s=0;for(let i=0;i<a.length;i++)s+=a[i];return s;}', ['f64[]'], '03d8feb90ba8b1dbe1e88f8c3876dab8b7c60cbf9f79e5c7d9fbad92a02ffa02'],
    ['function f(a,b){for(let i=0;i<a.length;i++)a[i]+=1;for(let j=0;j<b.length;j++)b[j]+=2;}', ['f32[]','f64[]'], '5428cd8c3aaa78f7c175f65766026ca3759c057cd6152ab3c73f940a44cb025a'],
    ['function f(a,b,c){for(let i=0;i<c.length;i++)a[c[i]]=b[i];}', ['f32[]','f64[]','u16[]'], '731251ac35e2b6cb03e021724357c55a77b924161c3397506f62e579daea5fef'],
  ];
  cases.forEach(([source, parameterTypes, hash], i) => {
    const artifact = compileNumericKernel(source, {parameterTypes, checkedIndexing: i === 6});
    assert.equal(artifact.manifest.version, i + 1);
    assert.equal(createHash('sha256').update(artifact.wasm).digest('hex'), hash);
    assert.doesNotThrow(() => host(artifact));
  });
});

test('portable integer package executes and its no-Wasm fallback keeps the frozen ABI', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-integer-package-'));
  const entry = path.join(root, 'source.mjs'), output = path.join(root, 'package');
  fs.writeFileSync(entry, copy.toString());
  const report = buildNumericKernel(entry, output, {parameterTypes: ['u8c[]', 'f64[]']});
  assert.equal(report.kernel.version, 9);
  assert.equal(report.accelerationClaim, false);
  const {createKernel} = await import(pathToFileURL(path.join(output, 'kernel.mjs')));
  const input = new Float64Array([0.5, 1.5, 255.5]), actual = new Uint8ClampedArray(3);
  const k = createKernel(); k.run(actual, input);
  assert.deepEqual(actual, new Uint8ClampedArray(input));
  assert.equal(k.diagnostics.wasmCalls, 1);
  const wasm = globalThis.WebAssembly;
  try {
    globalThis.WebAssembly = undefined;
    const retained = createKernel();
    actual.fill(0); retained.run(actual, input);
    assert.deepEqual(actual, new Uint8ClampedArray(input));
    assert.equal(retained.diagnostics.fallbackCalls, 1);
    assert.ok(Object.isFrozen(retained.manifest.lengthParameters));
    assert.ok(Object.isFrozen(retained.manifest.loops));
    assert.ok(Object.isFrozen(retained.manifest.loops[0]));
    retained.dispose(); assert.throws(() => retained.run(actual, input), {code: 'KERNEL_DISPOSED'});
  } finally {globalThis.WebAssembly = wasm;}
});

for (const [ArrayType, type] of TYPES) {
  test(`${type}: general-control ABI v9 composes integer stores, dynamic loops and transactional fuel`, () => {
    function update(out, count) {
      let i = 0;
      while (i < count) {
        out[i] += 255.5;
        do { out[i]--; } while (out[i] > 250);
        i++;
      }
      return i;
    }
    const artifact = compileNumericKernel(update.toString(), {
      parameterTypes: [type, 'f64'], generalControl: true, maxIterations: 30,
    });
    assert.equal(artifact.manifest.version, 9);
    assert.equal(artifact.manifest.kind, 'closed-numeric-control');
    assert.equal(artifact.manifest.loops, undefined);
    const k = host(artifact), actual = new ArrayType([1, 2, 3]), expected = actual.slice();
    assert.equal(k.run(actual, 3), update(expected, 3));
    assert.deepEqual(actual, expected);
    const exhausted = host(compileNumericKernel(update.toString(), {
      parameterTypes: [type, 'f64'], generalControl: true, maxIterations: 1,
    }));
    assert.throws(() => exhausted.run(actual, 3), {code: 'KERNEL_EXECUTION_FAILED'});
    assert.deepEqual(actual, expected);
    assert.equal(k.diagnostics.wasmCalls, 1);
  });
}
