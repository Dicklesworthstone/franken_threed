/** Production source discovery/dispatch and real Wasm; no substitute compiler. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import * as acorn from 'acorn';
import { discoverNumericArrayParameters } from './numeric_loop_discovery.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';
import { buildApplication } from './build_application.mjs';

const dispatchURL = new URL('./numeric_dispatch.mjs', import.meta.url).href;
const url = (dir, name) => pathToFileURL(path.join(dir, name)).href;
const fixture = () => fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-array-helpers-'));
async function application(source, options = {}) {
  const dir = fixture(), observer = url(dir, 'observer.mjs');
  fs.writeFileSync(path.join(dir, 'observer.mjs'), `
    import {createNumericDispatch as create, registerNumericDispatch as register,
      dispatchNumericCall, dispatchImportedNumericCall, numericDispatchDiagnostics} from ${JSON.stringify(dispatchURL)};
    export {dispatchNumericCall, dispatchImportedNumericCall};
    const tokens=[];
    export function createNumericDispatch(fn,...args){const token=create(fn,...args);tokens.push({name:fn.name,token});return token;}
    export function registerNumericDispatch(fn,...args){const token=register(fn,...args);tokens.push({name:fn.name,token});return token;}
    export function diagnostics(){return tokens.map(({name,token})=>({name,...numericDispatchDiagnostics(token)}));}
  `);
  const settings = {sourceName:'entry.mjs', runtimeModule:observer, ...options};
  const result = specializeNumericModule(source, settings);
  assert.deepEqual(result, specializeNumericModule(source, settings));
  fs.writeFileSync(path.join(dir, 'entry.mjs'), result.code);
  fs.writeFileSync(path.join(dir, 'reference.mjs'), source);
  const module = await import(url(dir, 'entry.mjs')), reference = await import(url(dir, 'reference.mjs'));
  const {diagnostics} = await import(observer);
  assert.ok(diagnostics().every(item => !item.initialized));
  assert.deepEqual(Object.keys(module), Object.keys(reference));
  return {dir, result, module, reference, diagnostics};
}
const stats = (app, name) => app.diagnostics().find(item => item.name === name);
const candidate = (app, name) => app.result.report.candidates.find(item => item.functionName === name);
const source = `
  function leaf(scale,input,out){for(let i=0;i<out.length;i++)out[i]+=input[i]*scale;}
  function forward(out,scale,input){leaf(scale,input,out);}
  export function update(out,input,scale){forward(out,scale,input);}
  export const identity=update;
  export function frame(out,input,scale){return update(out,input,scale);}
`;

test('array slots propagate through reordered helper arguments, without entering nested executions', () => {
  const ast = acorn.parse(`${source}
    function shadow(x){function hidden(){return x[0];}return 1;}
    function callback(x){return ()=>x.length;}
    function typeOnly(x){class Hidden{method(){return x[0];}}return 1;}`, {ecmaVersion:'latest',sourceType:'module'});
  const helpers = new Map(ast.body.map(statement => statement.declaration ?? statement)
    .filter(fn => fn.type === 'FunctionDeclaration').map(fn => [fn.id.name, fn]));
  assert.deepEqual([...discoverNumericArrayParameters(helpers.get('update'), helpers)], ['out','input']);
  assert.deepEqual([...discoverNumericArrayParameters(helpers.get('forward'), helpers)], ['out','input']);
  assert.deepEqual([...discoverNumericArrayParameters(helpers.get('leaf'), helpers)], ['input','out']);
  for (const name of ['shadow','callback','typeOnly']) assert.equal(discoverNumericArrayParameters(helpers.get(name), helpers).size, 0);
});

test('cyclic array constraints terminate, but discovery never admits recursion or unresolved calls', () => {
  const text = `function root(a,b){first(b,a);}function first(x,y){second(y,x);}
    function second(p,q){p[0]=q[0];first(q,p);}function external(a){unknown(a);}`;
  const ast = acorn.parse(text, {ecmaVersion:'latest'}), helpers = new Map(ast.body.map(fn => [fn.id.name,fn]));
  assert.deepEqual([...discoverNumericArrayParameters(helpers.get('root'), helpers)], ['a','b']);
  assert.equal(discoverNumericArrayParameters(helpers.get('external'), helpers).size, 0);
  const result = specializeNumericModule(text+'export function run(a,b){root(a,b);}');
  assert.equal(result.report.compiledKernels, 0);
});

test('ordinary delegating wrappers execute one native transaction over mixed Float32/64 storage', async () => {
  const app = await application(source), item = candidate(app, 'update');
  assert.equal(item.route, 'guarded-numeric-wasm');
  assert.deepEqual(item.parameterTypes, ['f64[]','f64[]','f64']);
  assert.deepEqual(item.scalarHelpers.map(h => [h.name,h.parameterTypes,h.resultType]), [
    ['forward',['f64[]','f64','f64[]'],'void'], ['leaf',['f64','f64[]','f64[]'],'void']]);
  assert.equal(app.module.update, app.module.identity);
  for (const ArrayType of [Float32Array, Float64Array]) {
    const actual = new ArrayType([1,2,3,4]), expected = actual.slice(), input = new Float64Array([1/3,-1,2/3,1]);
    for (let frame=0;frame<20;frame++) {
      app.module.frame(actual,input,1/60); app.reference.frame(expected,input,1/60);
      assert.deepEqual(actual,expected);
    }
    assert.equal(stats(app,'update').kernel.wasmCalls,20);
    assert.equal(stats(app,'update').kernel.fallbackCalls,0);
  }
  assert.ok(app.diagnostics().filter(item=>item.name!=='update').every(item=>!item.initialized), 'no helper crosses into a host dispatcher');
  assert.equal(app.result.report.accelerated,false);
});

test('callee effects determine mixed variants and preserve shifted and identical source views', async () => {
  const app = await application(source);
  for (const ArrayType of [Float32Array,Float64Array]) {
    for (const [a,b] of [[0,0],[0,1],[1,0],[0,2],[2,0]]) {
      const actual = new ArrayType([1,2,3,4,5,6,7]), expected = actual.slice();
      app.module.frame(actual.subarray(a,a+5),actual.subarray(b,b+5),1/3);
      app.reference.frame(expected.subarray(a,a+5),expected.subarray(b,b+5),1/3);
      assert.deepEqual(actual,expected);
    }
    assert.equal(stats(app,'update').kernel.wasmCalls,5);
    assert.equal(stats(app,'update').kernel.fallbackCalls,0);
  }
});

test('allocation hints select integer helper inputs and clamped outputs through source wrappers', async () => {
  const app = await application(`
    function gather(out,input,indices){for(let i=0;i<out.length;i++)out[i]=input[indices[i]];}
    function delegated(out,input,indices){gather(out,input,indices);}
    export const out=new Uint8ClampedArray(4);
    export const input=new Float64Array([1.5,2.5,300,-5]);
    export const indices=new Uint32Array([3,2,0,1]);
    export function run(){delegated(out,input,indices);}
  `);
  const item = candidate(app,'delegated');
  assert.ok(item.variants.some(v=>v.parameterTypes.join(',')==='u8c[],f64[],u32[]'));
  app.module.run(); app.reference.run();
  assert.deepEqual(app.module.out,app.reference.out); assert.deepEqual([...app.module.out],[0,255,2,2]);
  assert.equal(stats(app,'delegated').kernel.wasmCalls,1);
  assert.equal(stats(app,'gather').initialized,false);
});

test('short helper views abort before publication and retain surrounding application effects once', async () => {
  const app = await application(source+`
    export const events=[];
    export function observed(out,input,scale){events.push('before');update(out,input,scale);events.push('after');}
  `);
  const actual = new Float64Array([1,2,3,4]), expected = actual.slice();
  app.module.observed(actual,actual.subarray(0,2),1);
  app.reference.observed(expected,expected.subarray(0,2),1);
  assert.deepEqual(actual,expected); assert.deepEqual(app.module.events,['before','after']);
  assert.equal(stats(app,'update').kernel.fallbackCalls,1);
  assert.equal(stats(app,'update').kernel.wasmCalls,0);
});

test('non-native objects retain original getter/coercion ordering; guards never call source getters', async () => {
  const app = await application(source);
  const calls = [], expectedCalls = [];
  function args(events) {
    return [{get length(){events.push('length');return 2;},get 0(){events.push('read0');return 2;},
      set 0(value){events.push(['write0',value]);},get 1(){events.push('read1');return 3;},
      set 1(value){events.push(['write1',value]);}},new Float64Array([2,3]),
      {valueOf(){events.push('scale');return 2;}}];
  }
  app.module.frame(...args(calls)); app.reference.frame(...args(expectedCalls));
  assert.deepEqual(calls,expectedCalls);
  assert.equal(stats(app,'update').kernel.fallbackCalls,1);
  assert.equal(stats(app,'update').kernel.wasmCalls,0);
});

test('helper work-limit fallback retains the whole wrapper and native credit resets on the next call', async () => {
  const app = await application(source,{maxIterations:2});
  const actual = new Float64Array([1,2,3,4]), expected = actual.slice(), input = new Float64Array([1,2,3,4]);
  app.module.frame(actual,input,2); app.reference.frame(expected,input,2); assert.deepEqual(actual,expected);
  assert.equal(stats(app,'update').kernel.fallbackCalls,1);
  app.module.frame(actual.subarray(0,2),input.subarray(0,2),2);
  app.reference.frame(expected.subarray(0,2),input.subarray(0,2),2); assert.deepEqual(actual,expected);
  assert.equal(stats(app,'update').kernel.wasmCalls,1);
});

test('mutable or shadowed helper bindings never authorize typed-array native calls', async () => {
  const app = await application(source+'export function replace(fn){leaf=fn;}');
  assert.notEqual(candidate(app,'update')?.route,'guarded-numeric-wasm');
  let count=0; app.module.replace((scale,input,out)=>{count++;out[0]=scale+input[0];});
  const a=new Float64Array([1]);app.module.frame(a,new Float64Array([2]),3);
  assert.equal(a[0],5);assert.equal(count,1);
  const shadow = await application(`function leaf(a){for(let i=0;i<a.length;i++)a[i]++;}
    function delegated(leaf,a){leaf(a);}
    export function run(fn,a){delegated(fn,a);}`);
  assert.notEqual(candidate(shadow,'delegated')?.route,'guarded-numeric-wasm');
  let calls=0;shadow.module.run(a=>{calls++;a[0]=7;},a);assert.equal(calls,1);assert.equal(a[0],7);
});

test('source helper array rebinding and void-as-number expressions retain the original', async () => {
  const app = await application(`function mutate(a){for(let i=0;i<a.length;i++)a[i]+=1;}
    export function delegated(a){return mutate(a)+1;}
    export function run(a){return delegated(a);}`);
  assert.equal(candidate(app,'delegated').route,'retained-js');
  const a=new Float64Array([2]);assert.ok(Number.isNaN(app.module.run(a)));assert.equal(a[0],3);
  const rebound = await application(`function mutate(a,b){a=b;for(let i=0;i<a.length;i++)a[i]+=1;}
    function delegated(a,b){mutate(a,b);}
    export function run(a,b){delegated(a,b);}`);
  assert.equal(rebound.result.report.compiledKernels,0);
  const b=new Float64Array([9]);rebound.module.run(a,b);assert.equal(a[0],3);assert.equal(b[0],10);
});

test('imported original identities select the complete helper graph without replacing live exports', async () => {
  const app = await application(source,{crossModule:true});
  const consumerSource = `import {update,identity} from './entry.mjs';
    export function run(out,input,scale){update(out,input,scale);}
    export function same(){return update===identity;}`;
  const consumer = specializeNumericModule(consumerSource,{crossModule:true,runtimeModule:dispatchURL});
  fs.writeFileSync(path.join(app.dir,'consumer.mjs'),consumer.code);
  const module = await import(url(app.dir,'consumer.mjs'));
  assert.equal(consumer.report.compiledKernels,0);assert.equal(module.same(),true);
  const a=new Float32Array([1,2,3]);module.run(a,new Float64Array([3,6,9]),1/3);
  assert.deepEqual([...a],[2,4,6]);assert.equal(stats(app,'update').kernel.wasmCalls,1);
  assert.ok(app.diagnostics().filter(item=>item.name!=='update').every(item=>!item.initialized));
});

function runProcess(script) {
  const result=spawnSync(process.execPath,['--input-type=module','-e',script],{encoding:'utf8',timeout:20000});
  assert.equal(result.status,0,result.stderr);return result.stdout;
}
const observe = `let calls=0;
  const NativeInstance=WebAssembly.Instance;
  WebAssembly.Instance=new Proxy(NativeInstance,{construct(Target,args){
    const instance=Reflect.construct(Target,args);return {exports:{memory:instance.exports.memory,
      run(...values){calls++;return instance.exports.run(...values);}}};
  }});`;

test('relocated standalone array-helper packages execute native code and retain a no-Wasm fallback', () => {
  const dir=fixture(), entry=path.join(dir,'input.mjs'), out=path.join(dir,'build'), relocated=path.join(dir,'moved');
  const text=`function scale(a,s){for(let i=0;i<a.length;i++)a[i]*=s;}
    function sum(a){let result=0;for(let i=0;i<a.length;i++)result+=a[i];return result;}
    export function root(a,s){scale(a,s);return sum(a);}`;
  fs.writeFileSync(entry,text);
  const result=buildNumericKernel(entry,out,{functionName:'root',parameterTypes:['f32[]','f64'],maxIterations:6});
  assert.equal(result.kernel.version,8);assert.equal(result.kernel.parameters[0].write,true);
  assert.deepEqual(result.selectedFunction.scalarHelpers.map(h=>h.resultType),['void','f64']);
  fs.renameSync(out,relocated);
  const check=`const a=new Float32Array([1,2,3]),b=a.slice();
    assert.equal(kernel.run(a,1/3),retained(b,1/3));assert.deepEqual(a,b);`;
  runProcess(`import assert from 'node:assert/strict';
    import {createKernel,retained} from ${JSON.stringify(url(relocated,'kernel.mjs'))};
    const kernel=createKernel();${check}
    assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,0);
    const c=new Float32Array([1,2,3,4]),d=c.slice();assert.equal(kernel.run(c,2),retained(d,2));
    assert.deepEqual(c,d);assert.equal(kernel.diagnostics.fallbackCalls,1);`);
  runProcess(`import assert from 'node:assert/strict';globalThis.WebAssembly=undefined;
    const {createKernel,retained}=await import(${JSON.stringify(url(relocated,'kernel.mjs'))});
    const kernel=createKernel();${check}assert.equal(kernel.diagnostics.fallbackCalls,1);`);
  const module=new WebAssembly.Module(fs.readFileSync(path.join(relocated,'kernel.wasm')));
  assert.deepEqual(WebAssembly.Module.imports(module),[]);
});

test('real linked application builds close imported array helpers and execute one native call per frame', async () => {
  const dir=fixture(),entry=path.join(dir,'entry.mjs'),out=path.join(dir,'build');
  fs.writeFileSync(path.join(dir,'helper.mjs'),'export function leaf(out,input,s){for(let i=0;i<out.length;i++)out[i]+=input[i]*s;}');
  fs.writeFileSync(entry,`import {leaf} from './helper.mjs';
    function update(out,input,s){leaf(out,input,s);}
    export function frame(out,input,s){update(out,input,s);}`);
  const built=await buildApplication(entry,out,{specializeNumeric:true});
  assert.ok(built.numericSpecialization.units.some(unit=>unit.candidates.some(item=>item.functionName==='update'&&item.route==='guarded-numeric-wasm')));
  const moved=path.join(dir,'moved');fs.renameSync(out,moved);
  const target=url(moved,built.entryFiles[0]);
  runProcess(`import assert from 'node:assert/strict';${observe}
    const app=await import(${JSON.stringify(target)});
    assert.equal(calls,0);const a=new Float32Array([1,2,3]),b=new Float64Array([2,4,6]),expected=a.slice();
    for(let i=0;i<30;i++){app.frame(a,b,1/60);for(let j=0;j<a.length;j++)expected[j]+=b[j]/60;}
    assert.deepEqual(a,expected);assert.equal(calls,30);`);
  runProcess(`import assert from 'node:assert/strict';globalThis.WebAssembly=undefined;
    const app=await import(${JSON.stringify(target)});const a=new Float32Array([1,2,3]);
    app.frame(a,new Float64Array([2,4,6]),.5);assert.deepEqual([...a],[2,4,6]);`);
});

test('CLI builds preserve dynamic modules and their array-helper graphs after relocation', () => {
  const dir=fixture(),entry=path.join(dir,'entry.mjs'),out=path.join(dir,'build');
  fs.writeFileSync(entry,`export const load=()=>import('./lazy.mjs');`);
  fs.writeFileSync(path.join(dir,'lazy.mjs'),source);
  const cli=spawnSync(process.execPath,[new URL('./cli.mjs',import.meta.url).pathname,
    '--entry',entry,'--build-app',out,'--specialize-numeric'],{encoding:'utf8',timeout:20000});
  assert.equal(cli.status,0,cli.stderr);
  const report=JSON.parse(fs.readFileSync(path.join(out,'f3d-numeric-specialization.json'),'utf8'));
  assert.ok(report.units.some(unit=>unit.candidates.some(item=>item.functionName==='update'&&item.route==='guarded-numeric-wasm')));
  const entryFile=report.units.find(unit=>unit.moduleIds.includes(pathToFileURL(entry).href)).fileName;
  const moved=path.join(dir,'moved');fs.renameSync(out,moved);
  runProcess(`import assert from 'node:assert/strict';${observe}
    const entry=await import(${JSON.stringify(url(moved,entryFile))});
    const app=await entry.load(),a=new Float64Array([1,2]);
    app.frame(a,new Float64Array([3,6]),1/3);assert.deepEqual([...a],[2,4]);assert.equal(calls,1);`);
});
