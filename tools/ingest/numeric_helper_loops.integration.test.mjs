import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import * as acorn from 'acorn';
import { hasNumericLoop } from './numeric_loop_discovery.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';

const dispatchURL = new URL('./numeric_dispatch.mjs', import.meta.url).href;
const asURL = (dir, name) => pathToFileURL(path.join(dir, name)).href;
function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-helper-loops-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}
async function application(t, source, options = {}) {
  const dir = fixture(t);
  fs.writeFileSync(path.join(dir, 'observer.mjs'), `
    import { createNumericDispatch as create, registerNumericDispatch as register,
      dispatchNumericCall, dispatchImportedNumericCall, numericDispatchDiagnostics } from ${JSON.stringify(dispatchURL)};
    export { dispatchNumericCall, dispatchImportedNumericCall };
    const tokens = [];
    export function createNumericDispatch(target, ...args) { const token=create(target,...args); tokens.push({name:target.name,token}); return token; }
    export function registerNumericDispatch(target, ...args) { const token=register(target,...args); tokens.push({name:target.name,token}); return token; }
    export function diagnostics() { return tokens.map(({name,token})=>({name,...numericDispatchDiagnostics(token)})); }
  `);
  const result = specializeNumericModule(source, { sourceName: 'entry.mjs', runtimeModule: asURL(dir, 'observer.mjs'), ...options });
  fs.writeFileSync(path.join(dir, 'entry.mjs'), result.code);
  fs.writeFileSync(path.join(dir, 'reference.mjs'), source);
  const module = await import(asURL(dir, 'entry.mjs'));
  const reference = await import(asURL(dir, 'reference.mjs'));
  const observer = await import(asURL(dir, 'observer.mjs'));
  return { result, module, reference, diagnostics: observer.diagnostics };
}
const helper = `function count(n) { let total=0; while(n>0) {total+=n;n--;} return total; }`;

test('ordinary source array updates now close loop-bearing helpers without caller rewrites', async t => {
  const source = `${helper}
    export function update(out, n) { for(let i=0;i<out.length;i++) out[i]+=count(n); }
    export const identity=update;
    export function frame(out,n){return update(out,n);}`;
  const app = await application(t, source);
  const candidate = app.result.report.candidates.find(item => item.functionName === 'update');
  assert.equal(candidate.route, 'guarded-numeric-wasm');
  assert.equal(candidate.maxLoopDepth, 2);
  assert.equal(candidate.loopCount, 2);
  assert.deepEqual(candidate.scalarHelpers.map(item => item.name), ['count']);
  assert.equal(app.module.identity, app.module.update);
  assert.equal(app.result.report.accelerated, false);
  for (const ArrayType of [Float32Array, Float64Array]) {
    const actual = new ArrayType([0, 1, -7, 1 / 3]), expected = actual.slice();
    for (let n = 0; n < 20; n++) {
      app.module.frame(actual, n); app.reference.frame(expected, n);
      assert.deepEqual(actual, expected);
    }
    const stats = app.diagnostics().find(item => item.name === 'update');
    assert.equal(stats.kernel.wasmCalls, 20);
    assert.equal(stats.kernel.fallbackCalls, 0);
  }
  // The callee is private Wasm code in the entry's instance, not a host crossing.
  assert.equal(app.diagnostics().find(item => item.name === 'count').initialized, false);
});

test('transitive scalar delegation is discovered and executes as one native invocation', async t => {
  const source = `${helper}
    function middle(n){return count(n)*2;}
    export function delegated(n){return middle(n)+1;}
    export const identity=delegated;
    export function run(n){return delegated(n);}`;
  const app = await application(t, source);
  const candidate = app.result.report.candidates.find(item => item.functionName === 'delegated');
  assert.equal(candidate.route, 'guarded-numeric-wasm');
  assert.deepEqual(candidate.scalarHelpers.map(item => item.name), ['middle', 'count']);
  assert.equal(candidate.loopCount, 1);
  assert.equal(candidate.maxLoopDepth, 1);
  for (const n of [0, 1, 2, 10, NaN]) assert.ok(Object.is(app.module.run(n), app.reference.run(n)));
  const stats = app.diagnostics().find(item => item.name === 'delegated');
  assert.equal(stats.kernel.wasmCalls, 5);
  assert.equal(stats.kernel.fallbackCalls, 0);
  assert.ok(app.diagnostics().filter(item => item.name !== 'delegated').every(item => !item.initialized));
  assert.equal(app.module.delegated, app.module.identity);
});

test('helper loop spans refer to their original module source, not to the caller slice', async t => {
  const source = `// Prefix matters to every span.
    function early(n){ let result=0; while(n>0){result+=n;n--;} return result; }
    export function root(out,n){for(let i=0;i<out.length;i++)out[i]=late(n);}
    function late(n){let result=0;
      for(let j=0;j<n;j++){result+=early(j);}
      return result;
    }
    export function run(out,n){root(out,n);}`;
  const app = await application(t, source);
  const item = app.result.report.candidates.find(item => item.functionName === 'root');
  assert.equal(item.loopCount, 3);
  assert.equal(item.maxLoopDepth, 3);
  for (const loop of item.loops) {
    const expected = loop.functionName === 'early' ? 'while(n>0)' : loop.functionName === 'late' ? 'for(let j=0' : 'for(let i=0';
    assert.ok(source.slice(loop.sourceSpan.start, loop.sourceSpan.end).startsWith(expected));
    const before = source.slice(0, loop.sourceSpan.start), lines = before.split('\n');
    assert.equal(loop.sourceSpan.line, lines.length);
    assert.equal(loop.sourceSpan.column, lines.at(-1).length);
  }
  const data = new Float64Array(5), expected = data.slice();
  app.module.run(data, 8); app.reference.run(expected, 8);
  assert.deepEqual(data, expected);
});

test('source helper exhaustion retains the whole array update and resets on the next call', async t => {
  const source = `${helper}
    function update(out,n){for(let i=0;i<out.length;i++){out[i]+=1;out[i]+=count(n);}}
    export function run(out,n){return update(out,n);}`;
  const app = await application(t, source, { maxIterations: 4 });
  const actual = new Float64Array([10, 20, 30]), expected = actual.slice();
  app.module.run(actual, 3); app.reference.run(expected, 3);
  assert.deepEqual(actual, expected);
  let stats = app.diagnostics().find(item => item.name === 'update');
  assert.equal(stats.kernel.fallbackCalls, 1);
  assert.equal(stats.kernel.wasmCalls, 0);
  app.module.run(actual, 0); app.reference.run(expected, 0);
  assert.deepEqual(actual, expected);
  stats = app.diagnostics().find(item => item.name === 'update');
  assert.equal(stats.kernel.wasmCalls, 1);
  assert.equal(stats.kernel.fallbackCalls, 1);
});

test('non-Number arguments retain the original coercion order without speculative host effects', async t => {
  const source = `${helper}
    function delegated(n){return count(n)+1;}
    export function run(n){return delegated(n);}`;
  const app = await application(t, source);
  let reads = 0, originalReads = 0;
  const actual = { valueOf(){reads++;return 3;} }, reference = { valueOf(){originalReads++;return 3;} };
  assert.equal(app.module.run(actual), app.reference.run(reference));
  assert.equal(reads, originalReads);
  assert.equal(app.diagnostics().find(item => item.name === 'delegated').kernel.fallbackCalls, 1);
});

test('new delegated candidates do not steal the last compilation slot from direct loop kernels', () => {
  const source = `function delegated(n){return count(n)+1;}
    export function run(n){return delegated(n);}
    ${helper}`;
  const result = specializeNumericModule(source, { maxKernels: 1 });
  const admitted = result.report.candidates.filter(item => item.route === 'guarded-numeric-wasm');
  assert.deepEqual(admitted.map(item => item.functionName), ['count']);
  assert.equal(result.report.candidates.find(item => item.functionName === 'delegated').reason, 'KERNEL_BUDGET');
});

test('mutable and shadowed helper names cannot authorize transitive native closure', async t => {
  const mutable = `${helper}
    function delegated(n){return count(n)+1;}
    export function run(n){return delegated(n);}
    export function replace(fn){count=fn;}`;
  const app = await application(t, mutable);
  assert.equal(app.result.report.compiledKernels, 0);
  let calls = 0;
  app.module.replace(n => { calls++; return n * 10; });
  assert.equal(app.module.run(4), 41);
  assert.equal(calls, 1);
  const shadowed = await application(t, `${helper}
    function delegated(count,n){return count(n);}
    export function run(fn,n){return delegated(fn,n);}`);
  assert.notEqual(shadowed.result.report.candidates.find(item => item.functionName === 'delegated')?.route, 'guarded-numeric-wasm');
  assert.equal(shadowed.module.run(n => n * 7, 5), 35);
});

test('immutable helper discovery terminates on cycles and does not enter nested execution scopes', () => {
  const source = `function a(){return b();} function b(){return a();}
    function local(){function hidden(){while(true){}}return 0;}
    function callback(){return ()=>{while(true){}};}
    function wrapper(){return local()+callback();}
    function count(n){while(n>0)n--;return n;}
    function called(n){return count(n);}`;
  const ast = acorn.parse(source, { ecmaVersion: 'latest' });
  const helpers = new Map(ast.body.map(fn => [fn.id.name, fn]));
  for (const name of ['a', 'b', 'local', 'callback', 'wrapper']) assert.equal(hasNumericLoop(helpers.get(name).body, helpers), false);
  assert.equal(hasNumericLoop(helpers.get('called').body), false);
  assert.equal(hasNumericLoop(helpers.get('called').body, helpers), true);
});

test('cross-module registrations admit exported scalar roots with only transitive helper iteration', async t => {
  const source = `${helper}
    export function delegated(n){return count(n)*3;}`;
  const app = await application(t, source, { crossModule: true });
  const item = app.result.report.candidates.find(item => item.functionName === 'delegated');
  assert.equal(item.route, 'guarded-numeric-wasm');
  assert.equal(item.exportReachable, true);
  const { dispatchImportedNumericCall } = await import(dispatchURL);
  assert.equal(dispatchImportedNumericCall(app.module.delegated, [10]), 165);
  assert.equal(app.diagnostics().find(item => item.name === 'delegated').kernel.wasmCalls, 1);
});

for (const crossModule of [false, true]) {
  for (const rootFirst of [false, true]) {
    test(`admitted declaration bodies remain original with ${crossModule ? 'shared' : 'local'} dispatch and ${rootFirst ? 'root' : 'helper'} first`, async t => {
      const helper = `export function add(out,input,count){for(let i=0;i<count;i++)out[i]+=input[i];}`;
      const root = `export function update(out,input,count){out[0]+=2;add(out,input,count);return out[0];}`;
      const source = `${(rootFirst ? [root, helper] : [helper, root]).join('\n')}
        export const identity=update, events=[];
        export function frame(out,input,count){events.push('before');const result=update(out,input,count);events.push(result);return result;}`;
      const app = await application(t, source, {crossModule, loopIslands:true});
      assert.equal(app.module.update, app.module.identity);
      for (const name of ['add', 'update'])
        assert.equal(app.module[name].toString(), app.reference[name].toString(), name);
      assert.equal(app.result.report.absorbedCalls, 1);
      assert.equal(app.result.report.rewrittenCalls, 1);
      assert.equal(app.result.report.candidates.find(item => item.functionName === 'add').calls.length, 0);
      assert.equal(app.result.report.loopIslands.compiledKernels, 0);
      for (const Type of [Float32Array, Float64Array]) {
        const values = new Type([1, 2, 3, 4, 5]), expected = values.slice();
        for (let step = 0; step < 3; step++) {
          // Shifted aliases exercise source-ordered writes inside the helper.
          assert.equal(app.module.frame(values.subarray(1), values.subarray(0, 4), 4),
            app.reference.frame(expected.subarray(1), expected.subarray(0, 4), 4));
          assert.deepEqual(values, expected);
        }
        assert.equal(app.diagnostics().find(item => item.name === 'update').kernel.wasmCalls, 3);
      }
      assert.deepEqual(app.module.events, app.reference.events);
      assert.equal(app.diagnostics().find(item => item.name === 'add').initialized, false);
      if (crossModule) {
        const {dispatchImportedNumericCall} = await import(dispatchURL);
        const actual = new Float64Array([10, 20]), expected = actual.slice(), input = new Float64Array([3, 4]);
        assert.equal(dispatchImportedNumericCall(app.module.update, [actual, input, 2]),
          app.reference.update(expected, input, 2));
        assert.deepEqual(actual, expected);
        assert.equal(app.diagnostics().find(item => item.name === 'update').kernel.wasmCalls, 4);
        assert.equal(app.diagnostics().find(item => item.name === 'add').initialized, false);
      }
    });
  }
}

test('a failed outer declaration resumes untouched helper source without nested native attempts', async t => {
  const source = `
    export function add(out,input,count){for(let i=0;i<count;i++)out[i]+=input[i];}
    export function update(out,input,count){out[0]+=2;add(out,input,count);return out[0];}
    export const events=[];
    export function frame(out,input,count){events.push('before');const result=update(out,input,count);events.push(result);return result;}`;
  const app = await application(t, source, {crossModule:true, maxIterations:2});
  const values = new Float64Array([1, 2, 3, 4]), expected = values.slice(), input = new Float64Array([0.5, 1, 1.5, 2]);
  assert.equal(app.module.frame(values, input, 4), app.reference.frame(expected, input, 4));
  assert.deepEqual(values, expected, 'speculative writes roll back before the original update');
  assert.deepEqual(app.module.events, app.reference.events);
  const stats = () => app.diagnostics().find(item => item.name === 'update');
  assert.equal(stats().kernel.fallbackCalls, 1);
  assert.equal(stats().kernel.wasmCalls, 0);
  assert.equal(app.diagnostics().find(item => item.name === 'add').initialized, false);

  // A source accessor observes exactly the original partial publication and
  // throws the same object; a nested dispatcher must never inspect it first.
  const marker = {}, seen = [], referenceSeen = [];
  const observed = (output, log) => ({0:3, get 1(){log.push([...output]);throw marker;}});
  assert.throws(() => app.module.frame(values, observed(values, seen), 2), error => error === marker);
  assert.throws(() => app.reference.frame(expected, observed(expected, referenceSeen), 2), error => error === marker);
  assert.deepEqual(values, expected);
  assert.deepEqual(seen, referenceSeen);
  assert.deepEqual(app.module.events, app.reference.events);
  assert.equal(app.diagnostics().find(item => item.name === 'add').initialized, false);
  assert.equal(app.module.update.toString(), app.reference.update.toString());

  assert.equal(app.module.frame(values, input, 1), app.reference.frame(expected, input, 1));
  assert.deepEqual(values, expected);
  assert.equal(stats().kernel.wasmCalls, 1, 'the failed transaction does not poison later dispatch');
  assert.equal(app.diagnostics().find(item => item.name === 'add').initialized, false);
});

test('a relocated generated package closes iterative helpers without parser, source tree, or host imports', async t => {
  const dir = fixture(t), entry = path.join(dir, 'source.mjs'), build = path.join(dir, 'build'), relocated = path.join(dir, 'relocated');
  const source = `${helper}\nexport function root(n){return count(n)*2+1;}`;
  fs.writeFileSync(entry, source);
  const built = buildNumericKernel(entry, build, { functionName: 'root', parameterTypes: ['f64'], maxIterations: 4 });
  assert.equal(built.kernel.version, 8);
  assert.equal(built.kernel.loopCount, 1);
  assert.equal(built.selectedFunction.name, 'root');
  assert.deepEqual(built.selectedFunction.scalarHelpers.map(item => item.name), ['count']);
  assert.ok(fs.readFileSync(path.join(build, 'retained.mjs'), 'utf8').startsWith(source));
  fs.renameSync(build, relocated);
  // This fresh process has no access to build-time imports through the package.
  const script = `import assert from 'node:assert/strict';
    import {createKernel,retained} from ${JSON.stringify(asURL(relocated, 'kernel.mjs'))};
    const kernel=createKernel();
    assert.equal(kernel.run(4),retained(4));
    assert.equal(kernel.diagnostics.wasmCalls,1);
    assert.equal(kernel.run(5),retained(5));
    assert.equal(kernel.diagnostics.fallbackCalls,1);
    assert.equal(kernel.run(4),retained(4));
    assert.equal(kernel.diagnostics.wasmCalls,2);
    console.log('relocated-helper-package-native-pass');`;
  const native = spawnSync(process.execPath, ['--input-type=module', '-e', script], { encoding: 'utf8', timeout: 10000 });
  assert.equal(native.status, 0, native.stderr);
  assert.match(native.stdout, /native-pass/);
  const noWasm = spawnSync(process.execPath, ['--input-type=module', '-e', `
    globalThis.WebAssembly=undefined;
    const {createKernel,retained}=await import(${JSON.stringify(asURL(relocated, 'kernel.mjs'))});
    const kernel=createKernel();
    if(kernel.run(5)!==retained(5)||kernel.diagnostics.fallbackCalls!==1)throw Error('fallback');
    console.log('no-wasm-helper-package-pass');`], { encoding: 'utf8', timeout: 10000 });
  assert.equal(noWasm.status, 0, noWasm.stderr);
  assert.match(noWasm.stdout, /package-pass/);
  const module = new WebAssembly.Module(fs.readFileSync(path.join(relocated, 'kernel.wasm')));
  assert.deepEqual(WebAssembly.Module.imports(module), []);
});
