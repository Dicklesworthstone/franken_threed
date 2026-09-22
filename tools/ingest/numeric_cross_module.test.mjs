import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';
import {
  createNumericDispatch, registerNumericDispatch, dispatchNumericCall,
  dispatchImportedNumericCall, numericDispatchDiagnostics,
  importedNumericDispatchDiagnostics as diagnostics,
} from './numeric_dispatch.mjs';

const runtime = new URL('./numeric_dispatch.mjs', import.meta.url).href;
const UPDATE = 'export function update(out, input, dt) { for (let i=0; i<out.length; i++) out[i] += input[i] * dt; }';
const SUM = 'export function sum(n) { let total=0; while(n>0) { total+=n; n--; } return total; }';

// Real separate ESM files, real compiled Wasm and the production shared host.
// Every graph gets fresh identities and an independent original-source oracle.
function graph(sources, options = {}, perFile = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-cross-module-'));
  const reports = {}, outputs = {};
  for (const [name, source] of Object.entries(sources)) {
    const settings = { crossModule: true, runtimeModule: runtime, sourceName: name, ...options, ...perFile[name] };
    const result = specializeNumericModule(source, settings);
    assert.deepEqual(result, specializeNumericModule(source, settings));
    assert.equal(result.report.accelerated, false);
    reports[name] = result.report;
    outputs[name] = result.code;
    for (const [directory, content] of [['native', result.code], ['original', source]]) {
      const filename = path.join(root, directory, name);
      fs.mkdirSync(path.dirname(filename), { recursive: true });
      fs.writeFileSync(filename, content, { flag: 'wx' });
    }
  }
  return {
    root, reports, outputs,
    load: name => import(pathToFileURL(path.join(root, 'native', name))),
    original: name => import(pathToFileURL(path.join(root, 'original', name))),
  };
}
function compare(fn, oracle, args) {
  const expected = args.map(value => ArrayBuffer.isView(value) ? value.slice() : value);
  assert.equal(fn(...args), oracle(...expected));
  assert.deepEqual(args, expected);
}
function assertOriginalFunction(actual, expected) {
  assert.equal(actual.toString(), expected.toString());
  assert.equal(actual.name, expected.name);
  assert.equal(actual.length, expected.length);
  assert.deepEqual(Reflect.ownKeys(actual), Reflect.ownKeys(expected));
}

for (const ArrayType of [Float32Array, Float64Array]) {
  test(`${ArrayType.name}: separate consumers share one lazy producer kernel and preserve aliases`, async () => {
    const files = graph({
      'producer.mjs': UPDATE,
      'one.mjs': "import {update} from './producer.mjs'; export function run(...args) { return update(...args); }",
      'nested/two.mjs': "import {update as advance} from '../producer.mjs'; export function run(...args) { return advance(...args); }",
    });
    const producer = await files.load('producer.mjs'), original = await files.original('producer.mjs');
    const one = await files.load('one.mjs'), two = await files.load('nested/two.mjs');
    assertOriginalFunction(producer.update, original.update);
    assert.deepEqual(Object.keys(producer), Object.keys(original));
    assert.equal(diagnostics(producer.update).initialized, false);
    assert.equal(files.reports['producer.mjs'].compiledKernels, 1);
    assert.equal(files.reports['producer.mjs'].rewrittenCalls, 0);
    assert.equal(files.reports['one.mjs'].compiledKernels, 0);
    assert.equal(files.reports['one.mjs'].importedCalls.length, 1);
    const actual = new ArrayType([1, -0, 3, 4, 5]), expected = actual.slice();
    for (let frame = 0; frame < 100; frame++) {
      (frame & 1 ? one : two).run(actual.subarray(1), actual.subarray(0, 4), 1/60);
      original.update(expected.subarray(1), expected.subarray(0, 4), 1/60);
      assert.deepEqual(actual, expected);
    }
    const info = diagnostics(producer.update);
    assert.equal(info.kernel.wasmCalls, 100);
    assert.equal(info.kernel.fallbackCalls, 0);
    assert.equal(info.variants.filter(variant => variant.initialized).length, 1);
  });
}

test('default exports, renamed imports and star/re-export chains retain the same function identity', async () => {
  const files = graph({
    'producer.mjs': SUM + '\nexport default sum; export const alias = sum;',
    'bridge.mjs': "export * from './producer.mjs'; export {default as operation} from './producer.mjs';",
    'consumer.mjs': "import direct from './producer.mjs'; import {operation as renamed, alias} from './bridge.mjs'; export {direct, renamed, alias}; export function run(n){return direct(n)+renamed(n)+alias(n);}",
  });
  const app = await files.load('consumer.mjs'), original = await files.original('consumer.mjs');
  assert.equal(app.direct, app.renamed);
  assert.equal(app.alias, app.renamed);
  assertOriginalFunction(app.direct, original.direct);
  assert.equal(files.outputs['bridge.mjs'], "export * from './producer.mjs'; export {default as operation} from './producer.mjs';");
  for (const n of [0, -0, 1, 10, NaN]) compare(app.run, original.run, [n]);
  assert.equal(diagnostics(app.direct).kernel.wasmCalls, 15);
});

test('export discovery follows alias chains without registering mutable alias values', async () => {
  const files = graph({
    'producer.mjs': 'function advance(out,dt){for(let i=0;i<out.length;i++)out[i]+=dt;} const a=advance; let b=a; export {b as update}; export default a; export function replace(fn){b=fn;}',
    'consumer.mjs': "import original, {update,replace} from './producer.mjs'; export {original}; export function run(...args){return update(...args);} export function swap(out,fn){return update((replace(fn),out),2);}",
  });
  const app = await files.load('consumer.mjs');
  const out = new Float64Array([1]); let calls = 0;
  const replacement = function(out, dt) { assert.equal(this, undefined); calls++; out[0]+=dt*10; return 77; };
  // The old imported callee is read before the argument rebinds the live alias.
  app.swap(out, replacement);
  assert.equal(out[0], 3);
  assert.equal(calls, 0);
  assert.equal(diagnostics(app.original).kernel.wasmCalls, 1);
  assert.equal(app.run(out, 2), 77);
  assert.equal(out[0], 23);
  assert.equal(calls, 1);
  assert.equal(diagnostics(app.original).kernel.wasmCalls, 1);
});

test('same export spelling in different producers cannot select another function by name', async () => {
  const files = graph({
    'one.mjs': UPDATE,
    'two.mjs': UPDATE.replace(' * dt', ' * dt * 2'),
    'consumer.mjs': "import {update as a} from './one.mjs'; import {update as b} from './two.mjs'; export {a,b}; export function run(out,input,dt){a(out,input,dt);b(out,input,dt);}",
  });
  const app = await files.load('consumer.mjs'), original = await files.original('consumer.mjs');
  assert.notEqual(app.a, app.b);
  compare(app.run, original.run, [new Float64Array([1,2]), new Float64Array([3,4]), 2]);
  assert.equal(diagnostics(app.a).kernel.wasmCalls, 1);
  assert.equal(diagnostics(app.b).kernel.wasmCalls, 1);
});

test('ESM-cycle calls before producer evaluation use JavaScript, then the registered Wasm', async () => {
  const files = graph({
    'producer.mjs': "import {early} from './consumer.mjs';\n" + UPDATE + '\nexport const value=early;',
    'consumer.mjs': "import {update} from './producer.mjs'; const out=new Float64Array([1]); update(out,new Float64Array([2]),3); export const early=out[0]; export function run(...args){return update(...args);}",
  });
  const producer = await files.load('producer.mjs');
  assert.equal(producer.value, 7);
  assert.equal(diagnostics(producer.update).initialized, false);
  assert.equal((await files.original('producer.mjs')).value, 7);
  const app = await files.load('consumer.mjs'), original = await files.original('consumer.mjs');
  compare(app.run, original.run, [new Float64Array([1]), new Float64Array([2]), 3]);
  assert.equal(diagnostics(producer.update).kernel.wasmCalls, 1);
});

test('cyclic Math TDZ exceptions remain original and late producer initialization permits Wasm', async () => {
  const files = graph({
    'producer.mjs': "import {early} from './consumer.mjs'; const Math=globalThis.Math; export {early}; export function absolute(out){for(let i=0;i<out.length;i++)out[i]=Math.abs(out[i]);}",
    'consumer.mjs': "import {absolute} from './producer.mjs'; let early=false; try {absolute(new Float64Array([-1]));} catch(error){early=error instanceof ReferenceError;} export {early}; export function run(a){return absolute(a);}",
  });
  const producer = await files.load('producer.mjs'), consumer = await files.load('consumer.mjs');
  assert.equal(producer.early, true);
  assert.equal(diagnostics(producer.absolute).initialized, false);
  const out = new Float64Array([-3]); consumer.run(out);
  assert.equal(out[0], 3);
  assert.equal(diagnostics(producer.absolute).kernel.wasmCalls, 1);
});

test('dynamic entry loading and top-level await do not duplicate producer initialization', async () => {
  const files = graph({
    'producer.mjs': 'export const events=[]; events.push("before"); await Promise.resolve(); events.push("after");\n' + SUM,
    'lazy.mjs': "import {sum} from './producer.mjs'; export function run(n){return sum(n);}",
    'entry.mjs': "export async function run(n){const app=await import('./lazy.mjs');return app.run(n);}",
  });
  const app = await files.load('entry.mjs');
  assert.equal(await app.run(10), 55);
  assert.equal(await app.run(5), 15);
  const producer = await files.load('producer.mjs');
  assert.deepEqual(producer.events, ['before','after']);
  assert.equal(diagnostics(producer.sum).kernel.wasmCalls, 2);
});

test('nested imported calls preserve callee/argument order, source spans and token boundaries', async () => {
  const files = graph({
    'producer.mjs': SUM,
    'consumer.mjs': "import {sum} from './producer.mjs';\nexport function run(events){return ((sum)) /* ( misleading */ ((events.push('outer'),sum((events.push('inner'),3))));}",
  });
  const app = await files.load('consumer.mjs'), original = await files.original('consumer.mjs');
  const events = [], expected = [];
  assert.equal(app.run(events), original.run(expected));
  assert.deepEqual(events, ['outer','inner']); assert.deepEqual(events, expected);
  for (const site of files.reports['consumer.mjs'].importedCalls) {
    assert.equal(site.localName, 'sum');
    assert.equal(site.moduleSpecifier, './producer.mjs');
    assert.equal(site.sourceSpan.line, 2);
  }
  const producer = await files.load('producer.mjs');
  assert.equal(diagnostics(producer.sum).kernel.wasmCalls, 2);
});

test('shadowed imports, callable proxies, argument failures and original receivers retain exact calls', async () => {
  const files = graph({
    'producer.mjs': SUM,
    'consumer.mjs': "import {sum} from './producer.mjs'; export function run(sum,...args){return sum(...args);} export function fail(sum,events){return sum((events.push('arg'),9));}",
  });
  const app = await files.load('consumer.mjs');
  const failure = {}, events = []; let applied = 0, properties = 0;
  const proxy = new Proxy(function(){}, {
    get() { properties++; throw failure; },
    apply(_target, receiver, args) { applied++; assert.equal(receiver, undefined); return args; },
  });
  assert.deepEqual(app.run(proxy, 1, 2, 3), [1,2,3]);
  assert.equal(applied, 1); assert.equal(properties, 0);
  assert.throws(() => app.fail(null, events), TypeError);
  assert.deepEqual(events, ['arg']);
  assert.throws(() => app.run(() => { throw failure; }), error => error === failure);
  const badSpread = { *[Symbol.iterator]() { throw failure; } };
  assert.throws(() => app.run(proxy, ...badSpread), error => error === failure);
  assert.equal(applied, 1);
});

test('zero-argument, spread and string-named imports work; optional and namespace calls remain intact', async () => {
  const files = graph({
    'producer.mjs': 'function constant(){let n=0;while(n<3)n++;return n;} export {constant as "odd name",constant};',
    'consumer.mjs': "import {\"odd name\" as count} from './producer.mjs'; import * as ns from './producer.mjs'; export function run(){return count();} export function spread(...args){return count(...args);} export function optional(){return count?.();} export function namespace(){return ns.constant();}",
  });
  const app = await files.load('consumer.mjs'), producer = await files.load('producer.mjs');
  assert.equal(app.run(), 3); assert.equal(app.spread(), 3);
  assert.equal(app.optional(), 3); assert.equal(app.namespace(), 3);
  assert.equal(diagnostics(producer.constant).kernel.wasmCalls, 2);
  assert.equal(files.reports['consumer.mjs'].importedCalls.length, 2);
  assert.equal(files.reports['consumer.mjs'].importedCalls[0].importedName, 'odd name');
});

test('mutable original functions are never paired with stale compiled code', async () => {
  const files = graph({
    'producer.mjs': SUM + '\nexport function replace(fn){sum=fn;}',
    'consumer.mjs': "import {sum,replace} from './producer.mjs'; export {replace}; export function run(n){return sum(n);}",
  });
  const app = await files.load('consumer.mjs'), producer = await files.load('producer.mjs');
  assert.equal(files.reports['producer.mjs'].candidates[0].reason, 'MUTABLE_FUNCTION_BINDING');
  assert.equal(diagnostics(producer.sum), null);
  assert.equal(app.run(4), 10);
  app.replace(n => n*11);
  assert.equal(app.run(4), 44);
  assert.equal(diagnostics(producer.sum), null);
});

test('uncompiled dependencies preserve side effects and are not reported as compiled kernels', async () => {
  const files = graph({
    'external.mjs': 'export const events=[]; export function host(value){events.push(value);return value+1;}',
    'consumer.mjs': "import {host,events} from './external.mjs'; export {events}; export function run(n){return host(n);}",
  });
  const app = await files.load('consumer.mjs');
  assert.equal(app.run(7), 8); assert.deepEqual(app.events, [7]);
  const report = files.reports['consumer.mjs'];
  assert.equal(report.compiledKernels, 0);
  assert.equal(report.registeredKernels, 0);
  assert.equal(report.route, 'shared-numeric-dispatch-with-retained-fallback');
});

test('source closure and direct eval remain conservative in producer and consumer units', async () => {
  const files = graph({
    'producer.mjs': 'const captured=7; export function sum(n){let s=0;while(n>0){s+=captured;n--;}return s;}',
    'consumer.mjs': "import {sum} from './producer.mjs'; export function run(n){return eval('sum(n)');}",
  });
  assert.equal(files.reports['producer.mjs'].compiledKernels, 0);
  assert.equal(files.reports['consumer.mjs'].refusal.code, 'DIRECT_EVAL');
  assert.equal(files.reports['consumer.mjs'].importedCalls.length, 0);
  const app = await files.load('consumer.mjs');
  assert.equal(app.run(3), 21);
});

test('producer Math resolution is live and independent of consumer bindings', async () => {
  const files = graph({
    'producer.mjs': 'let Math=globalThis.Math; export function replace(value){Math=value;} export function absolute(out){for(let i=0;i<out.length;i++)out[i]=Math.abs(out[i]);}',
    'consumer.mjs': "import {absolute,replace} from './producer.mjs'; const Math=null; export {replace}; export function run(out){return absolute(out);}",
  });
  const app = await files.load('consumer.mjs'), producer = await files.load('producer.mjs');
  const out = new Float64Array([-1,-2]); app.run(out);
  assert.deepEqual([...out], [1,2]);
  let calls=0; app.replace({abs(n){calls++;return n+10;}}); app.run(out);
  assert.deepEqual([...out], [11,12]); assert.equal(calls, 2);
  assert.equal(diagnostics(producer.absolute).kernel.wasmCalls, 1);
  assert.equal(diagnostics(producer.absolute).kernel.fallbackCalls, 1);
  assert.equal(diagnostics(producer.absolute).kernel.lastGuardFailure, 'KERNEL_MATH_BINDING');
});

test('exported general loops propagate budgets and late bounds rollback through the shared host', async () => {
  const files = graph({
    'producer.mjs': 'export function update(out,input,n){let i=0;while(i<n){out[i]+=input[i];i++;}return i;}',
    'consumer.mjs': "import {update} from './producer.mjs'; export function run(...args){return update(...args);}",
  }, {maxIterations: 2});
  const app = await files.load('consumer.mjs'), producer = await files.load('producer.mjs');
  const original = await files.original('consumer.mjs');
  const actual = new Float64Array([1,2,3,4]), expected=actual.slice();
  assert.equal(app.run(actual.subarray(1), actual.subarray(0,3), 3),
    original.run(expected.subarray(1), expected.subarray(0,3), 3));
  assert.deepEqual(actual, expected);
  let info = diagnostics(producer.update).kernel;
  assert.equal(info.wasmCalls, 0); assert.equal(info.fallbackCalls, 1); assert.equal(info.copiedBytes, 0);
  compare(app.run, original.run, [new Float64Array([1,2]), new Float64Array([3]), 2]);
  assert.equal(diagnostics(producer.update).kernel.fallbackCalls, 2);
  compare(app.run, original.run, [new Float64Array([1,2]), new Float64Array([3,4]), 2]);
  assert.equal(diagnostics(producer.update).kernel.wasmCalls, 1);
});

test('custom array length and scalar coercion getters run only in the retained original', async () => {
  const files = graph({
    'producer.mjs': UPDATE,
    'consumer.mjs': "import {update} from './producer.mjs'; export function run(...args){return update(...args);}",
  });
  const app = await files.load('consumer.mjs'), producer = await files.load('producer.mjs');
  const out = new Float64Array([1,2]); let lengths=0, numbers=0;
  Object.defineProperty(out,'length',{get(){lengths++;return 2;}});
  app.run(out,new Float64Array([3,4]),{valueOf(){numbers++;return 2;}});
  assert.deepEqual([...out],[7,10]); assert.equal(lengths,3); assert.equal(numbers,2);
  assert.equal(diagnostics(producer.update).kernel.fallbackCalls,1);
  assert.equal(diagnostics(producer.update).kernel.wasmCalls,0);
});

test('Wasm-unavailable module graphs import and execute original functions without retries', async () => {
  const files = graph({
    'producer.mjs': SUM,
    'consumer.mjs': "import {sum} from './producer.mjs'; export function run(n){return sum(n);}",
  });
  const native = globalThis.WebAssembly;
  let app, producer;
  try {
    globalThis.WebAssembly=undefined;
    app=await files.load('consumer.mjs'); producer=await files.load('producer.mjs');
    assert.equal(diagnostics(producer.sum).initialized,false);
    assert.equal(app.run(4),10); assert.equal(app.run(5),15);
  } finally {globalThis.WebAssembly=native;}
  assert.equal(app.run(6),21);
  assert.equal(diagnostics(producer.sum).retainedCalls,3);
  assert.equal(diagnostics(producer.sum).kernel,null);
});

test('private tokens stay private; duplicate registration does not reset shared state', () => {
  function sum(n){let total=0;while(n>0){total+=n;n--;}return total;}
  const bytes=compileNumericKernel(sum.toString(),{parameterTypes:['f64'],generalControl:true}).wasm;
  const privateToken=createNumericDispatch(sum,bytes);
  assert.equal(diagnostics(sum),null);
  assert.equal(dispatchImportedNumericCall(sum,[3]),6);
  assert.equal(numericDispatchDiagnostics(privateToken).initialized,false);
  const token=registerNumericDispatch(sum,bytes);
  assert.equal(dispatchNumericCall(token,sum,[4]),10);
  const duplicate=registerNumericDispatch(sum,bytes);
  assert.equal(dispatchImportedNumericCall(sum,[5]),15);
  assert.equal(diagnostics(sum).kernel.wasmCalls,2);
  assert.equal(numericDispatchDiagnostics(duplicate).initialized,false);
  assert.equal(numericDispatchDiagnostics(privateToken).initialized,false);
  assert.ok(Object.isFrozen(diagnostics(sum)));
});

test('registry misses and diagnostics do not read properties, coerce keys or invoke proxy traps', () => {
  const keys=[null,undefined,1,NaN,'fn',Symbol('fn'),{},()=>{}];
  for(const key of keys) assert.equal(diagnostics(key),null);
  let traps=0, calls=0;
  const proxy=new Proxy(function(){calls++;return 9;},{get(){traps++;throw Error('unexpected');}});
  assert.equal(diagnostics(proxy),null);
  assert.equal(dispatchImportedNumericCall(proxy,[]),9);
  assert.equal(traps,0);assert.equal(calls,1);
  for(const key of keys.slice(0,-1)) assert.throws(()=>dispatchImportedNumericCall(key,[]),TypeError);
  assert.throws(()=>registerNumericDispatch(null,[]),TypeError);
});

test('legacy defaults, explicit opt-out, fresh bindings and export budgets are deterministic', () => {
  const source=UPDATE+'\nexport function run(...args){return update(...args);}';
  assert.deepEqual(specializeNumericModule(source),specializeNumericModule(source,{crossModule:false}));
  assert.equal(specializeNumericModule(UPDATE).changed,false);
  assert.equal(specializeNumericModule(UPDATE).report.candidates[0].reason,'NO_LOCAL_DIRECT_CALLS');
  const producer=specializeNumericModule(UPDATE,{crossModule:true});
  assert.equal(producer.report.registeredKernels,1);
  const hidden=specializeNumericModule(UPDATE.replace('export ',''),{crossModule:true});
  assert.equal(hidden.changed,false);
  for(const crossModule of [null,0,'true',{},[]])assert.throws(()=>specializeNumericModule('',{crossModule}),TypeError);
  const budget=specializeNumericModule(UPDATE+'\n'+SUM,{crossModule:true,maxKernels:1});
  assert.equal(budget.report.compiledKernels,1);
  assert.equal(budget.report.candidates[1].reason,'KERNEL_BUDGET');
  const consumer="#!/usr/bin/env node\n'use strict'; import {sum} from './producer.mjs'; export function run(n){let __f3d_numeric_imported_call_3=null;return sum(n);}";
  const result=specializeNumericModule(consumer,{crossModule:true});
  assert.ok(result.code.startsWith("#!/usr/bin/env node\n'use strict';"));
  assert.ok(result.code.includes('__f3d_numeric_imported_call_4'));
  assert.equal(result.report.importedCalls[0].sourceSpan.start,consumer.indexOf('sum(n)'));
});


test('export-only kernels cannot displace existing local-call kernels under a unit budget', () => {
  const source=UPDATE+'\n'+SUM+'\nexport function local(n){return sum(n);}';
  const result=specializeNumericModule(source,{crossModule:true,maxKernels:1});
  assert.equal(result.report.compiledKernels,1);
  assert.equal(result.report.candidates.find(item=>item.functionName==='sum').route,'guarded-numeric-wasm');
  assert.equal(result.report.candidates.find(item=>item.functionName==='update').reason,'KERNEL_BUDGET');
});

test('reentrant Wasm initialization retains the original call without recursive construction', () => {
  function total(n){let s=0;while(n>0){s+=n;n--;}return s;}
  const artifact=compileNumericKernel(total.toString(),{parameterTypes:['f64'],generalControl:true});
  registerNumericDispatch(total,artifact.wasm);
  const NativeModule=WebAssembly.Module; let constructions=0,reentrant;
  try {
    WebAssembly.Module=class extends NativeModule {
      constructor(bytes){constructions++;reentrant=dispatchImportedNumericCall(total,[2]);super(bytes);}
    };
    assert.equal(dispatchImportedNumericCall(total,[4]),10);
  } finally {WebAssembly.Module=NativeModule;}
  assert.equal(constructions,1);assert.equal(reentrant,3);
  assert.equal(diagnostics(total).retainedCalls,1);
  assert.equal(diagnostics(total).kernel.wasmCalls,1);
});

test('distinct dispatcher module instances never borrow registrations from one another', async () => {
  function total(n){let s=0;while(n>0){s+=n;n--;}return s;}
  const bytes=compileNumericKernel(total.toString(),{parameterTypes:['f64'],generalControl:true}).wasm;
  const separate=await import(new URL('./numeric_dispatch.mjs?independent-registry',import.meta.url));
  registerNumericDispatch(total,bytes);
  assert.equal(separate.importedNumericDispatchDiagnostics(total),null);
  assert.equal(separate.dispatchImportedNumericCall(total,[3]),6);
  assert.equal(diagnostics(total).initialized,false);
  assert.equal(dispatchImportedNumericCall(total,[3]),6);
  assert.equal(diagnostics(total).kernel.wasmCalls,1);
});
