import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { discoverNumericStorageHints } from './numeric_storage_hints.mjs';
import { compileNumericCandidate } from './numeric_candidate.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';
import { createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics } from './numeric_dispatch.mjs';

const INTEGER_TYPES = [['Int8Array', 'i8[]'], ['Uint8Array', 'u8[]'], ['Uint8ClampedArray', 'u8c[]'],
  ['Int16Array', 'i16[]'], ['Uint16Array', 'u16[]'], ['Int32Array', 'i32[]'], ['Uint32Array', 'u32[]']];
const FLOAT_TYPES = [['Float32Array', 'f32[]'], ['Float64Array', 'f64[]']];
const parse = source => acorn.parse(source, {ecmaVersion: 'latest', sourceType: 'module'});
const url = value => pathToFileURL(value).href;

// The generated module uses the real dispatcher/runtime/compiler. The wrapper
// observes private tokens without changing application exports or function IDs.
async function fixture(source, options = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-integer-specialization-'));
  const wrapper = path.join(dir, 'dispatch.mjs');
  fs.writeFileSync(wrapper, `
import {createNumericDispatch as create, dispatchNumericCall, numericDispatchDiagnostics} from ${JSON.stringify(new URL('./numeric_dispatch.mjs', import.meta.url).href)};
export {dispatchNumericCall};
export const tokens = [];
export function createNumericDispatch(...args) {const token = create(...args); tokens.push(token); return token;}
export function diagnostics() {return tokens.map(numericDispatchDiagnostics);}
`);
  const result = specializeNumericModule(source, {...options, runtimeModule: url(wrapper)});
  const original = path.join(dir, 'original.mjs'), emitted = path.join(dir, 'emitted.mjs');
  fs.writeFileSync(original, source); fs.writeFileSync(emitted, result.code);
  const reference = await import(url(original)), actual = await import(url(emitted));
  return {reference, actual, result, diagnostics: (await import(url(wrapper))).diagnostics};
}
function hints(source, names = ['out']) {
  const ast = parse(source), sites = [];
  walk.simple(ast, {CallExpression(node) {if (node.callee.name === 'kernel') sites.push(node);}});
  return discoverNumericStorageHints(ast)(sites, names.map(name => ({name, type: 'f64[]', read: true, write: name === 'out'})));
}

for (const [Output, outputType] of INTEGER_TYPES) {
  test(`${Output}: automatic guarded dispatch across all nine numeric input types`, async () => {
    for (const [Input, inputType] of [...INTEGER_TYPES, ...FLOAT_TYPES]) {
      const source = `
export function convert(out, input, scale) {
  for (let i = 0; i < input.length; i++) out[i] += input[i] * scale;
}
export const out = new ${Output}([254, 127, 32767, 0, 1, 2, 3]);
const input = new ${Input}([0.5, 1.5, -1.25, 4294967295, NaN, Infinity, -Infinity]);
export const original = convert;
export function frame(scale) { convert(out, input, scale); }
`;
      const f = await fixture(source);
      assert.equal(f.result.report.compiledKernels, 1);
      assert.ok(f.result.report.candidates[0].variants.some(v =>
        v.parameterTypes.join(',') === [outputType, inputType, 'f64'].join(',')));
      assert.equal(f.actual.original, f.actual.convert);
      assert.equal(f.actual.convert.toString(), f.reference.convert.toString());
      for (const scale of [1, -0.5, 2.75]) {
        f.actual.frame(scale); f.reference.frame(scale);
        assert.deepEqual(f.actual.out, f.reference.out, `${Output} <- ${Input}`);
      }
      const [d] = f.diagnostics();
      assert.equal(d.kernel.wasmCalls, 3); assert.equal(d.kernel.fallbackCalls, 0);
      assert.equal(d.retainedCalls, 0);
      assert.equal(d.variants.filter(v => v.initialized).length, 1, 'only selected AOT variant instantiates');
      assert.equal(f.result.report.accelerated, false);
    }
  });
}

test('allocation hints follow aliases, subviews, static factories and wrapper arguments', async () => {
  const source = `
export function paint(pixels, count) {
  let i = 0;
  while (i < count) { pixels[i] += 2.5; i++; }
  return i;
}
function forward(data, count) { return paint(data, count); }
export const storage = Uint8ClampedArray.of(90, 0, 1, 254, 90);
let view = storage.subarray(1, 4), alias = view;
export function frame(count) { return forward(alias, count); }
`;
  const f = await fixture(source);
  assert.equal(f.actual.frame(3), f.reference.frame(3));
  assert.deepEqual(f.actual.storage, f.reference.storage);
  assert.deepEqual([...f.actual.storage], [90, 2, 4, 255, 90]);
  assert.equal(f.diagnostics()[0].kernel.wasmCalls, 1);
  assert.equal(f.result.report.candidates[0].controlSemantics, 'budgeted-source-order-v1');
});

test('mixed integer outputs, float values and topology scatter execute natively in source order', async () => {
  const f = await fixture(`
function scatter(out, input, indices) {
  for (let i = 0; i < indices.length; i++) { out[indices[i]] += input[i]; out[indices[i]]++; }
  return out[0];
}
export const out = new Int8Array([126, 127, 5]);
const input = Float32Array.from([1.5, 255, -1.5, 2, -300]), indices = new Uint16Array([0,0,1,0,1]);
export function frame() { return scatter(out, input, indices); }
`);
  for (let frame = 0; frame < 5; frame++) {
    assert.equal(f.actual.frame(), f.reference.frame());
    assert.deepEqual(f.actual.out, f.reference.out);
  }
  assert.equal(f.diagnostics()[0].kernel.wasmCalls, 5);
  assert.equal(f.diagnostics()[0].kernel.fallbackCalls, 0);
});

test('general integer control retains transactional fallback on fuel exhaustion', async () => {
  const f = await fixture(`
function paint(a,n) { let i=0; while(i<n) {a[i]+=2; i++;} return i; }
export const out = new Uint8Array([3,4,5]);
export function frame(n) {return paint(out,n);}
`, {maxIterations: 1});
  assert.equal(f.actual.frame(3), f.reference.frame(3));
  assert.deepEqual(f.actual.out, f.reference.out);
  let d = f.diagnostics()[0];
  assert.equal(d.kernel.fallbackCalls, 1); assert.equal(d.kernel.copiedBytes, 0);
  assert.equal(f.actual.frame(1), f.reference.frame(1));
  assert.deepEqual(f.actual.out, f.reference.out);
  d = f.diagnostics()[0]; assert.equal(d.kernel.wasmCalls, 1);
});

test('incorrect shadowed-constructor hints never invoke getters or replace the original behavior', async () => {
  const f = await fixture(`
export let constructions = 0, tags = 0;
function Uint8Array() {
  constructions++;
  const a = [0,1,2];
  Object.defineProperty(a, Symbol.toStringTag, {get(){tags++; throw Error('unexpected tag access');}});
  return a;
}
function paint(a) { for(let i=0;i<a.length;i++) a[i]+=300.5; }
export const out = new Uint8Array();
export function frame() {paint(out);}
`);
  f.actual.frame(); f.reference.frame();
  assert.deepEqual([...f.actual.out], [...f.reference.out]);
  assert.equal(f.actual.out[0], 300.5);
  assert.equal(f.actual.constructions, 1); assert.equal(f.actual.tags, 0);
  assert.equal(f.diagnostics()[0].kernel.fallbackCalls, 1);
  assert.equal(f.diagnostics()[0].kernel.wasmCalls, 0);
});

test('unsupported storage and later rebinding use the actual callee, with effects once', async () => {
  const f = await fixture(`
function paint(a) {for(let i=0;i<a.length;i++)a[i]++;}
export const original=paint;
export let out = new Uint8Array([255]);
export function replace(value) {out=value;}
export function frame() {paint(out);}
export function shadow(paint) {paint(out);}
`);
  f.actual.frame(); f.reference.frame(); assert.deepEqual(f.actual.out, f.reference.out);
  f.actual.replace([255]); f.reference.replace([255]);
  f.actual.frame(); f.reference.frame(); assert.deepEqual(f.actual.out, [256]);
  let called = 0;
  f.actual.shadow(a => {called++; a[0] = 7;});
  assert.equal(called, 1); assert.deepEqual(f.actual.out, [7]);
  const d = f.diagnostics()[0]; assert.equal(d.identityMisses, 1);
  assert.equal(d.variants.reduce((n,v)=>n+(v.kernel?.wasmCalls ?? 0),0), 1);
});

test('same-type aliases share ordered scratch; overlapping mixed storage retains JS', async () => {
  const f = await fixture(`
function add(a,b) {for(let i=0;i<a.length;i++)a[i]+=b[i];}
export const same = new Uint8Array([1,2,3,4]);
const alias = same.subarray(1);
export const buffer = new ArrayBuffer(8), bytes = new Uint8Array(buffer), words = new Uint16Array(buffer);
words.set([257,258,259,260]);
export function frame() {add(alias,same.subarray(0,3));}
export function mixed() {add(bytes.subarray(0,4),words);}
`);
  f.actual.frame(); f.reference.frame(); assert.deepEqual(f.actual.same, f.reference.same);
  assert.equal(f.diagnostics()[0].kernel.wasmCalls, 1);
  f.actual.mixed(); f.reference.mixed(); assert.deepEqual(f.actual.bytes, f.reference.bytes);
  assert.equal(f.diagnostics()[0].kernel.lastGuardFailure, 'KERNEL_ARRAY_ALIAS');
});

test('no-Wasm hosts retain integer applications and instantiate no variant eagerly', async () => {
  const saved = globalThis.WebAssembly;
  try {
    globalThis.WebAssembly = undefined;
    const f = await fixture(`function fill(a){for(let i=0;i<a.length;i++)a[i]=300.5;}
export const out = new Uint8ClampedArray(3); export function frame(){fill(out);}`);
    assert.equal(f.diagnostics()[0].initialized, false);
    f.actual.frame(); f.reference.frame(); assert.deepEqual(f.actual.out, f.reference.out);
    assert.equal(f.diagnostics()[0].retainedCalls, 1);
    assert.equal(f.diagnostics()[0].initializationFailure, 'KERNEL_INITIALIZATION_FAILED');
  } finally { globalThis.WebAssembly = saved; }
});

test('hint generation never evaluates constructor arguments and does not admit unclosed code', () => {
  assert.ok(hints('kernel(new Uint8Array(thrower()));')[0].includes('u8[]'));
  const source = `function work(a) {for(let i=0;i<a.length;i++)a[i]=external(i);}
const a=new Uint8Array(4); work(a);`;
  const result = specializeNumericModule(source);
  assert.equal(result.changed, false); assert.equal(result.code, source);
  assert.equal(result.report.candidates[0].reason, 'KERNEL_NOT_CLOSED');
});

test('hint graphs handle cycles, reverse declarations, unknown views and ambiguous assignments', () => {
  assert.ok(hints('kernel(a); let a=b; let b=c; let c=new Int16Array(2); c=a;')
    .some(types => types[0] === 'i16[]'));
  const result = hints('let a; a=new Int8Array(2); a=new Uint32Array(2); kernel(a.slice());');
  assert.ok(result.some(types => types[0] === 'i8[]'));
  assert.ok(result.some(types => types[0] === 'u32[]'));
  assert.deepEqual(hints('kernel(unknown.array);'), []);
  assert.deepEqual(hints('kernel(...values);'), []);
  assert.deepEqual(hints('kernel(new Float64Array(2));'), []);
  assert.ok(hints('kernel(new globalThis.Uint8ClampedArray(2));').some(types => types[0] === 'u8c[]'));
});

test('AOT work and dispatch alternatives remain bounded without removing application features', () => {
  const declarations = INTEGER_TYPES.map(([name], i) => `const a${i}=new ${name}(2);`).join('\n');
  const calls = INTEGER_TYPES.flatMap((_,i)=>INTEGER_TYPES.map((_,j)=>`copy(a${i},a${j});`)).join('\n');
  const source = `function copy(a,b){for(let i=0;i<a.length;i++)a[i]=b[i];}${declarations}${calls}`;
  const result = specializeNumericModule(source);
  assert.equal(result.report.compiledKernels, 1);
  assert.equal(result.report.candidates[0].variants.length, 17);
  const overload = Array.from({length: 4100}, (_,i)=>`let a${i}=new Uint8Array(1);`).join('\n');
  assert.deepEqual(hints(overload+'kernel(a0);'), []);
  const maskSource = `function kernel(out){for(let i=0;i<out.length;i++)out[i]++;}${overload}kernel(a0);`;
  assert.equal(specializeNumericModule(maskSource).report.candidates[0].variants.length, 2);
});

test('dispatch validates all integer ABI tags and checks the embedded manifest before execution', () => {
  function fill(a) {for(let i=0;i<a.length;i++)a[i]=255;}
  const primary = compileNumericCandidate(fill.toString(), {parameterTypes:['f64[]']});
  for (const [name, type] of INTEGER_TYPES) {
    const variant = compileNumericCandidate(fill.toString(), {parameterTypes:[type]});
    const token = createNumericDispatch(fill, primary.wasm, [{parameterTypes:[type],bytes:variant.wasm}]);
    const out = new globalThis[name](2); dispatchNumericCall(token,fill,[out]);
    const expected = new globalThis[name](2); fill(expected); assert.deepEqual(out,expected);
    assert.equal(numericDispatchDiagnostics(token).kernel.wasmCalls,1);
  }
  for (const type of ['__proto__','toString','bigint64[]'])
    assert.throws(()=>createNumericDispatch(fill,primary.wasm,[{parameterTypes:[type],bytes:primary.wasm}]),TypeError);
  const token = createNumericDispatch(fill,primary.wasm,[{parameterTypes:['u8[]'],bytes:primary.wasm}]);
  const out = new Uint8Array(2); dispatchNumericCall(token,fill,[out]);
  assert.deepEqual([...out],[255,255]);
  assert.equal(numericDispatchDiagnostics(token).initializationFailure,'KERNEL_INITIALIZATION_FAILED');
  assert.equal(numericDispatchDiagnostics(token).retainedCalls,1);
});

test('portable integer package with general control retains its v9 ABI without Wasm', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'f3d-integer-control-package-'));
  const entry = path.join(dir,'entry.mjs'), output = path.join(dir,'package');
  fs.writeFileSync(entry,'function fill(a){let i=0;while(i<a.length){a[i]=i+0.5;i++;}return i;}');
  const report = buildNumericKernel(entry,output,{parameterTypes:['u8c[]'],generalControl:true,maxIterations:10});
  assert.equal(report.kernel.version,9); assert.equal(report.kernel.kind,'closed-numeric-control');
  const {createKernel} = await import(url(path.join(output,'kernel.mjs')));
  const out = new Uint8ClampedArray(3), k = createKernel();
  assert.equal(k.run(out),3); assert.deepEqual([...out],[0,2,2]); assert.equal(k.diagnostics.wasmCalls,1);
  const saved = globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const retained=createKernel(); out.fill(0); assert.equal(retained.run(out),3);
    assert.deepEqual([...out],[0,2,2]); assert.equal(retained.diagnostics.fallbackCalls,1);
    assert.ok(Object.isFrozen(retained.manifest.lengthParameters));
    assert.equal(retained.manifest.loops,undefined);
  } finally {globalThis.WebAssembly=saved;}
});

test('float-only transformed modules preserve the parent compiler output byte for byte', () => {
  // Captured from 340ca8a's specializer with the same source and default options.
  const fixtures = [
    ['function scale(a,s){for(let i=0;i<a.length;i++)a[i]*=s;} export function frame(a,s){scale(a,s);}', '11dc05a87f1a721dc3c96730f17d2d926a8cfb81f2fa47a1afcf9c893f1cbf98'],
    ['function gather(a,b,index){for(let i=0;i<index.length;i++)a[i]=b[index[i]];} export function frame(a,b,index){gather(a,b,index);}', '3134617e57b8048e51a4a16c8a854bbcd5157c1fbfbada0470484970af1307f9'],
    ['function count(n){let s=0;for(let i=0;i<n;i++){s+=i;}return s;} export function frame(n){return count(n);}', 'f41d9050e1f1229e4453ae00326dc05af94a09f17302f0b1b01770f75c4d524b'],
    ['function f(a){for(let i=0;i<a.length;i++)a[i]*=3;} const a=new Float32Array(3); f(a);', '4ecf41e42440f98741c41da60adb06efae7151fbae3f3a1cbb9dd7e5e0ce2517'],
  ];
  for (const [source, hash] of fixtures) {
    const result = specializeNumericModule(source);
    assert.equal(createHash('sha256').update(result.code).digest('hex'), hash);
    assert.equal(result.changed, true);
  }
});
