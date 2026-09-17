import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildNumericKernel } from './numeric_kernel_build.mjs';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { recomputeNormals } from './fixtures/numeric/indexed_normals.mjs';

const fixture = fileURLToPath(new URL('./fixtures/numeric/indexed_normals.mjs', import.meta.url));
const fresh = () => fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-indexed-package-'));
const load = directory => import(pathToFileURL(path.join(directory, 'kernel.mjs')).href);
const same = (a, b) => assert.deepEqual([...a], [...b]);
function sourceFile(root, source) {
  const entry = path.join(root, 'input.mjs');
  fs.writeFileSync(entry, source, { flag: 'wx' });
  return entry;
}

for (const Float of [Float32Array, Float64Array]) for (const Index of [Uint16Array, Uint32Array]) {
  test(`relocatable indexed normals package: ${Float.name}/${Index.name}`, async () => {
    const root = fresh(), output = path.join(root, 'output');
    const types = [Float === Float32Array ? 'f32[]' : 'f64[]', Index === Uint16Array ? 'u16[]' : 'u32[]', Float === Float32Array ? 'f32[]' : 'f64[]'];
    const result = buildNumericKernel(fixture, output, { parameterTypes: types });
    assert.equal(result.kernel.version, 7);
    assert.equal(result.kernel.loops.length, 3);
    assert.equal(result.accelerationClaim, false);
    const relocated = path.join(root, 'relocated');
    fs.cpSync(output, relocated, { recursive: true, errorOnExist: true, force: false });
    const { createKernel, retained } = await load(relocated);
    const kernel = createKernel();
    const p = new Float([0,0,0, 1,0,0, 1,1,0, 0,1,1]);
    const indices = new Index([0,1,2, 0,2,3]);
    const n = new Float(p.length), expected = n.slice();
    for (let frame = 0; frame < 12; frame++) {
      p[2] += 0.125;
      assert.equal(kernel.run(p, indices, n), recomputeNormals(p, indices, expected));
      same(n, expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 12);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
    assert.equal(retained.name, 'recomputeNormals');
    // A bad face after valid writes must restart the ORIGINAL function, with
    // neither cleared normals nor a partially accumulated result published.
    indices[5] = 99;
    assert.ok(Object.is(kernel.run(p, indices, n), retained(p, indices, expected)));
    same(n, expected);
    assert.equal(kernel.diagnostics.fallbackCalls, 1);
    indices[5] = 3;
    assert.equal(kernel.run(p, indices, n), retained(p, indices, expected));
    same(n, expected);
    assert.equal(kernel.diagnostics.wasmCalls, 13);
  });
}

test('float-only indirect source selects checked ABI without changing the low-level default', async () => {
  const root = fresh();
  const source = 'function gather(a,b,index){for(let i=0;i<index.length;i++)a[i]=b[index[i]];}';
  const options = { parameterTypes: ['f64[]','f64[]','f64[]'] };
  assert.throws(() => compileNumericKernel(source, options), { code: 'KERNEL_NOT_CLOSED' });
  assert.equal(compileNumericCandidate(source, options).manifest.version, 7);
  const output = path.join(root, 'output');
  buildNumericKernel(sourceFile(root, source), output, options);
  const kernel = (await load(output)).createKernel();
  const a = new Float64Array(2);
  kernel.run(a, new Float64Array([2,3,5]), new Float64Array([2,0]));
  same(a, [5,2]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('indexed entry packages close immutable scalar helpers', async () => {
  const root = fresh(), output = path.join(root, 'output');
  const entry = sourceFile(root, `function offset(i){return i*2;}
export function update(a,index){for(let i=0;i<index.length;i++)a[offset(index[i])]+=3;}`);
  const result = buildNumericKernel(entry, output, { parameterTypes: ['f32[]','u16[]'], functionName: 'update' });
  assert.deepEqual(result.selectedFunction.scalarHelpers.map(helper => helper.name), ['offset']);
  const kernel = (await load(output)).createKernel(), a = new Float32Array(5);
  kernel.run(a, new Uint16Array([2,0,2]));
  same(a, [3,0,0,0,6]);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('no-Wasm package retains execution and deeply freezes checked-index descriptors', async () => {
  const root = fresh(), output = path.join(root, 'output');
  buildNumericKernel(fixture, output, { parameterTypes: ['f32[]','u32[]','f32[]'] });
  const { createKernel, retained } = await load(output);
  const saved = globalThis.WebAssembly;
  let kernel;
  try { globalThis.WebAssembly = undefined; kernel = createKernel(); }
  finally { globalThis.WebAssembly = saved; }
  const p = new Float32Array([0,0,0, 1,0,0, 0,1,0]), index = new Uint32Array([0,1,2]);
  const a = new Float32Array(9), expected = a.slice();
  assert.equal(kernel.run(p,index,a), retained(p,index,expected));
  same(a, expected);
  assert.equal(kernel.diagnostics.fallbackCalls, 1);
  assert.equal(kernel.diagnostics.lastGuardFailure, 'KERNEL_WASM_UNAVAILABLE');
  for (const value of [kernel.manifest, kernel.manifest.lengthParameters, kernel.manifest.loops, ...kernel.manifest.loops]) {
    assert.ok(Object.isFrozen(value));
  }
  assert.throws(() => { kernel.manifest.loops[0].boundParameter = 99; }, TypeError);
  kernel.dispose();
  assert.throws(() => kernel.run(p,index,a), { code: 'KERNEL_DISPOSED' });
});

test('unsafe source and explicitly disabled checked indexing fail before creating output', () => {
  const cases = [
    ['function f(a){for(let i=0;i<a.length;i++)a[i]=1;}', { parameterTypes: ['u32[]'] }],
    ['function f(a,b){for(let i=0;i<b.length;i++)a[b[i]]=external();}', { parameterTypes: ['f64[]','u16[]'] }],
    ['function f(a,b){for(let i=0;i<b.length;i++)a[b[i]]=1;}', { parameterTypes: ['f64[]','f64[]'], checkedIndexing: false }],
    ['function f(a){for(let i=0;i<a.length;i++)a[i]=1;}', { parameterTypes: ['f64[]'], checkedIndexing: 'yes' }],
  ];
  for (const [source, options] of cases) {
    const root = fresh(), output = path.join(root, 'output');
    assert.throws(() => buildNumericKernel(sourceFile(root, source), output, options));
    assert.equal(fs.existsSync(output), false);
  }
});

test('legacy prefix kernels keep deterministic module bytes and manifests', () => {
  const sources = [
    'function f(a,b){for(let i=0;i<a.length;i++)a[i]+=b[i];}',
    'function f(a,b){let sum=0;for(let i=0;i<a.length;i++)sum+=a[i]*b[i];return sum;}',
    'function f(a,b){for(let i=0;i<a.length;i++)a[i]+=1;for(let j=0;j<b.length;j++)b[j]+=2;}',
  ];
  for (const source of sources) for (const storage of ['f64[]','f32[]']) {
    const root = fresh(), output = path.join(root, 'output');
    const options = { parameterTypes: [storage, storage], sourceName: 'input.mjs', allowMath: true };
    const original = compileNumericKernel(source, options), candidate = compileNumericCandidate(source, options);
    assert.deepEqual(candidate.wasm, original.wasm);
    assert.deepEqual(candidate.manifest, original.manifest);
    const built = buildNumericKernel(sourceFile(root, source), output, options);
    assert.deepEqual(built.kernel, original.manifest);
    assert.deepEqual(new Uint8Array(fs.readFileSync(path.join(output, 'kernel.wasm'))), original.wasm);
  }
});
