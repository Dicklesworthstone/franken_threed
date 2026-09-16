import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics } from './numeric_dispatch.mjs';

const runtimeUrl = new URL('./numeric_dispatch.mjs', import.meta.url).href;
const runtimeKernelUrl = new URL('./numeric_kernel_runtime.mjs', import.meta.url).href;
const sha256 = data => createHash('sha256').update(data).digest('hex');
const same = (a, b) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Object.is(a[i], b[i]), `element ${i}: ${a[i]} !== ${b[i]} including zero sign`);
};
const temp = t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-math-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
};
async function application(t, source, files = {}) {
  const dir = temp(t);
  const result = specializeNumericModule(source, { sourceName: 'app.mjs', runtimeModule: runtimeUrl });
  const tokens = [...result.code.matchAll(/var (__f3d_numeric_token_\d+) =/g)].map(match => match[1]);
  // Test-only observation of real dispatch tokens; production exports stay intact.
  const observation = tokens.length ? `\nimport {numericDispatchDiagnostics as __testStats} from ${JSON.stringify(runtimeUrl)};\nexport const __stats=()=>[${tokens.map(name => `__testStats(${name})`).join(',')}];\n` : '';
  for (const [file, contents] of Object.entries(files)) fs.writeFileSync(path.join(dir, file), contents);
  fs.writeFileSync(path.join(dir, 'app.mjs'), result.code + observation);
  const module = await import(pathToFileURL(path.join(dir, 'app.mjs')).href);
  return { module, result, dir };
}
const rootSource = 'export function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
const rootApp = rootSource + '\nexport function run(a) { return update(a); }';

for (const name of ['abs', 'ceil', 'floor', 'fround', 'max', 'min', 'round', 'sign', 'sqrt', 'trunc']) {
  test(`ordinary application calls discover Math.${name} and select f32/f64 native variants`, async t => {
    const declaration = `export function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.${name}(a[i]); }`;
    const { module, result } = await application(t, declaration + '\nexport function run(a) { return update(a); }\nexport const identity=update;');
    assert.equal(result.report.compiledKernels, 1);
    assert.deepEqual(result.report.candidates[0].mathIntrinsics, [name]);
    assert.equal(module.identity, module.update);
    assert.equal(module.update.toString(), declaration.slice(7));
    assert.equal(module.__stats()[0].initialized, false);
    for (const ArrayType of [Float32Array, Float64Array]) {
      const actual = new ArrayType([NaN, Infinity, -Infinity, 0, -0, -0.5, 0.49999999999999994, 2.5, -2.5, 9]);
      const expected = actual.slice();
      module.update(expected); module.run(actual); same(actual, expected);
      const stats = module.__stats()[0];
      assert.equal(stats.kernel.wasmCalls, 1); assert.equal(stats.kernel.fallbackCalls, 0);
      assert.equal(stats.identityMisses, 0);
    }
  });
}

const geometrySource = `
function magnitude(x,y,z) { return Math.sqrt(x*x+y*y+z*z); }
function quantize(x) { return Math.fround(Math.round(Math.max(-1,Math.min(1,x))*127)/127); }
export function update(position,colors,uniform,gain) {
  let longest=0;
  for(let i=0;i<position.length;i+=3) {
    const x=position[i], y=position[i+1], z=position[i+2];
    const n=magnitude(x,y,z); longest=Math.max(longest,n);
    if(n>0) { position[i]=x/n*uniform[0]; position[i+1]=y/n*uniform[1]; position[i+2]=z/n*uniform[2]; }
  }
  for(let j=0;j<colors.length;j++) colors[j]=quantize(colors[j]*gain);
  return Math.sqrt(longest);
}
export function frame(position,colors,uniform,gain) { return update(position,colors,uniform,gain); }
`;
for (const Storage of [Float32Array, Float64Array]) for (const Uniform of [Float32Array, Float64Array]) {
  test(`automatic 10k-vertex + 40k-color pipeline: ${Storage.name} / ${Uniform.name}`, async t => {
    const { module, result } = await application(t, geometrySource);
    assert.equal(result.report.compiledKernels, 1);
    const candidate = result.report.candidates[0];
    assert.equal(candidate.loopCount, 2); assert.equal(candidate.variants.length, 4);
    assert.deepEqual(candidate.mathIntrinsics, ['fround', 'max', 'min', 'round', 'sqrt']);
    assert.deepEqual(candidate.scalarHelpers.map(helper => helper.name), ['magnitude', 'quantize']);
    const p = new Storage(30000), c = new Storage(40000), u = new Uniform([1.1, 0.9, 1.2]);
    for (let i=0;i<p.length;i++) p[i]=(i%113-56)/31;
    for (let i=0;i<c.length;i++) c[i]=(i%79-39)/41;
    const ep=p.slice(), ec=c.slice(), eu=u.slice();
    for (let frame=0;frame<30;frame++) {
      const expected=module.update(ep,ec,eu,1.013);
      assert.ok(Object.is(module.frame(p,c,u,1.013),expected)); same(p,ep); same(c,ec); same(u,eu);
    }
    const stats=module.__stats()[0];
    assert.equal(stats.kernel.wasmCalls,30); assert.equal(stats.kernel.fallbackCalls,0);
    assert.equal(stats.kernel.copiedBytes,30*(p.byteLength*2+c.byteLength*2+u.byteLength));
    assert.equal(stats.variants.filter(variant=>variant.initialized).length,1);
  });
}

test('live module-local Math identity follows replacement and restoration', async t => {
  const {module}=await application(t, `let Math=globalThis.Math;\n${rootApp}\nexport function setMath(value){Math=value;}`);
  const a=new Float64Array([4,9]);module.run(a);same(a,[2,3]);
  module.setMath({sqrt:x=>x+7});module.run(a);same(a,[9,10]);
  assert.equal(module.__stats()[0].kernel.lastGuardFailure,'KERNEL_MATH_BINDING');
  module.setMath(Math);module.run(a);same(a,[3,Math.sqrt(10)]);
  assert.equal(module.__stats()[0].kernel.wasmCalls,2);assert.equal(module.__stats()[0].kernel.fallbackCalls,1);
});

test('a renamed import named Math uses the original module binding, not the runtime global',async t=>{
  const {module}=await application(t,`import {methods as Math} from './methods.mjs';\n${rootApp}`,
    {'methods.mjs':'export const methods={sqrt:x=>x+40};'});
  const a=new Float64Array([4,9]);module.run(a);same(a,[44,49]);
  assert.equal(module.__stats()[0].kernel.wasmCalls,0);assert.equal(module.__stats()[0].kernel.fallbackCalls,1);
});

test('caller-local Math does not replace the top-level function/helper environment',async t=>{
  const {module}=await application(t, `${rootSource}\nexport function run(a,Math){return update(a);} `);
  const a=new Float64Array([4,9]);module.run(a,{sqrt(){throw new Error('caller binding must not be used');}});same(a,[2,3]);
  assert.equal(module.__stats()[0].kernel.wasmCalls,1);
});

test('registration does not read a TDZ Math binding; early empty calls preserve behavior',async t=>{
  const {module}=await application(t, `${rootApp}
    export const early=run(new Float64Array());
    export const partial=new Float64Array([4]);
    export let failure;
    try {run(partial);}catch(error){failure=error;}
    let Math=globalThis.Math;
  `);
  assert.equal(module.early,undefined);assert.ok(module.failure instanceof ReferenceError);same(module.partial,[4]);
  assert.equal(module.__stats()[0].kernel.fallbackCalls,2);
  module.run(module.partial);same(module.partial,[2]);assert.equal(module.__stats()[0].kernel.wasmCalls,1);
});

test('ESM-cycle calls before registration use original code; subsequent calls use native Math',async t=>{
  const {module}=await application(t, `import {early} from './cycle.mjs';\n${rootApp}\nexport {early};`,
    {'cycle.mjs':`import {run} from './app.mjs';export const early=new Float64Array([4,9]);run(early);`});
  same(module.early,[2,3]);assert.equal(module.__stats()[0].initialized,false);
  module.run(module.early);same(module.early,[Math.sqrt(2),Math.sqrt(3)]);
  assert.equal(module.__stats()[0].kernel.wasmCalls,1);
});

test('argument evaluation mutating Math happens before invocation guards, in source order',async t=>{
  const {module}=await application(t,`${rootSource}
    export function run(a,change,log){return update((log.push('argument'),change(),a));}
  `);
  const saved=Math.sqrt, log=[],a=new Float64Array([4,9]);
  try{module.run(a,()=>{log.push('change');Math.sqrt=x=>{log.push('sqrt');return x+1;};},log);}
  finally{Math.sqrt=saved;}
  same(a,[5,10]);assert.deepEqual(log,['argument','change','sqrt','sqrt']);
  assert.equal(module.__stats()[0].kernel.wasmCalls,0);assert.equal(module.__stats()[0].kernel.fallbackCalls,1);
});

test('overridden method getter before lazy instantiation is invoked only by retained code',async t=>{
  const {module}=await application(t,rootApp);
  const saved=Object.getOwnPropertyDescriptor(Math,'sqrt');let calls=0;
  const a=new Float64Array([4,9]);
  try{Object.defineProperty(Math,'sqrt',{configurable:true,get(){calls++;return saved.value;}});module.run(a);}
  finally{Object.defineProperty(Math,'sqrt',saved);}
  same(a,[2,3]);assert.equal(calls,2);assert.equal(module.__stats()[0].kernel.fallbackCalls,1);
});

test('callee identity misses and missing ESM-cycle tokens do not evaluate Math resolvers',()=>{
  const source='function update(a){for(let i=0;i<a.length;i++)a[i]=Math.sqrt(a[i]);}';
  const artifact=compileNumericKernel(source,{parameterTypes:['f64[]'],allowMath:true});
  const target=()=>{throw new Error('wrong identity');};
  const token=createNumericDispatch(target,artifact.wasm,[],()=>{throw new Error('must not resolve');});
  const other=a=>a.length;
  assert.equal(dispatchNumericCall(token,other,[[1,2]]),2);
  assert.equal(dispatchNumericCall(undefined,other,[[1]]),1);
  assert.equal(numericDispatchDiagnostics(token).identityMisses,1);
  assert.equal(numericDispatchDiagnostics(token).initialized,false);
  assert.throws(()=>createNumericDispatch(target,artifact.wasm,[],42),TypeError);
});

for(const expression of ['Math.sin(a[i])','Math.pow(a[i],2)','Math.hypot(a[i],1)','Math["sqrt"](a[i])','Math.sqrt(...a)']){
  test(`unsupported later pass keeps entire function and emits no runtime import: ${expression}`,()=>{
    const source=`function update(a){for(let i=0;i<a.length;i++)a[i]+=1;for(let i=0;i<a.length;i++)a[i]=${expression};}\nupdate(data);`;
    let resolutions=0;
    const result=specializeNumericModule(source,{runtimeModule(){resolutions++;return runtimeUrl;}});
    assert.equal(result.changed,false);assert.equal(result.code,source);assert.equal(resolutions,0);
    assert.equal(result.report.compiledKernels,0);
  });
}

test('unavailable WebAssembly at first invocation keeps original functions working',async t=>{
  const {module}=await application(t,rootApp);const saved=globalThis.WebAssembly;
  const a=new Float64Array([4,9]);
  try{globalThis.WebAssembly=undefined;module.run(a);}
  finally{globalThis.WebAssembly=saved;}
  same(a,[2,3]);assert.equal(module.__stats()[0].retainedCalls,1);
  assert.equal(module.__stats()[0].initializationFailure,'KERNEL_INITIALIZATION_FAILED');
});

async function packageOf(t, source, parameterTypes, options={}){
  const dir=temp(t),entry=path.join(dir,'source.mjs'),out=path.join(dir,'package');
  fs.writeFileSync(entry,source);
  const result=buildNumericKernel(entry,out,{parameterTypes,...options});
  const moved=path.join(dir,'relocated');fs.renameSync(out,moved);
  const module=await import(pathToFileURL(path.join(moved,'kernel.mjs')).href);
  return{dir,entry,out:moved,module,result};
}
for(const Storage of [Float32Array,Float64Array])test(`standalone Math helpers/pipeline package relocates and executes: ${Storage.name}`,async t=>{
  const source=geometrySource.slice(0,geometrySource.indexOf('export function frame'));
  const storage=Storage===Float32Array?'f32[]':'f64[]';
  const {module,result,out}=await packageOf(t,source,[storage,storage,'f64[]','f64']);
  const kernel=module.createKernel();
  const retained=await import(pathToFileURL(path.join(out,'retained.mjs')).href);
  assert.equal(module.retained,retained.update);assert.equal(retained.default,retained.update);
  assert.equal(result.sourceSha256,sha256(source));assert.equal(result.wasmSha256,sha256(fs.readFileSync(path.join(out,'kernel.wasm'))));
  assert.deepEqual(result.kernel.mathIntrinsics,['fround','max','min','round','sqrt']);
  assert.ok(Object.isFrozen(kernel.manifest.mathIntrinsics));
  const a=new Storage([3,4,0,0,0,0,-1,-2,3]),c=new Storage([-2,-0.5,0,0.5,2]),u=new Float64Array([1,2,3]);
  const ea=a.slice(),ec=c.slice();
  for(let i=0;i<5;i++){assert.ok(Object.is(kernel.run(a,c,u,0.9),module.retained(ea,ec,u,0.9)));same(a,ea);same(c,ec);}
  assert.equal(kernel.diagnostics.wasmCalls,5);assert.equal(kernel.diagnostics.fallbackCalls,0);
  assert.equal(fs.readdirSync(out).length,5);
});

test('single-function standalone source enables Math without explicit compiler options',async t=>{
  const {module,result}=await packageOf(t,rootSource,['f64[]']);
  assert.deepEqual(result.kernel.mathIntrinsics,['sqrt']);
  const kernel=module.createKernel(),a=new Float64Array([4,9]);kernel.run(a);same(a,[2,3]);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('no-Wasm standalone fallback exposes frozen intrinsic metadata and original exceptions',async t=>{
  const {module}=await packageOf(t,rootSource,['f64[]']);
  const saved=globalThis.WebAssembly;let kernel;
  try{globalThis.WebAssembly=undefined;kernel=module.createKernel();}finally{globalThis.WebAssembly=saved;}
  assert.ok(Object.isFrozen(kernel.manifest.mathIntrinsics));
  assert.throws(()=>kernel.manifest.mathIntrinsics.push('cos'),TypeError);
  const a=[4,9];kernel.run(a);same(a,[2,3]);assert.equal(kernel.diagnostics.fallbackCalls,1);
  const sqrt=Math.sqrt,sentinel={};let caught;
  try{Math.sqrt=()=>{throw sentinel;};try{kernel.run([4]);}catch(error){caught=error;}}finally{Math.sqrt=sqrt;}
  assert.equal(caught,sentinel);assert.equal(kernel.diagnostics.fallbackCalls,2);
  kernel.dispose();assert.throws(()=>kernel.run(a),{code:'KERNEL_DISPOSED'});
});

for(const suffix of [
  'function Math() { return 1; }',
  'function Math() { return 1; } export function replace() { Math=42; }',
])test('standalone rejects shadowed Math even when declaration is unused/mutable: '+suffix, t=>{
  const dir=temp(t),entry=path.join(dir,'source.mjs'),out=path.join(dir,'package');
  fs.writeFileSync(entry,rootSource+'\n'+suffix);
  assert.throws(()=>buildNumericKernel(entry,out,{parameterTypes:['f64[]']}),{code:'KERNEL_NOT_CLOSED'});
  assert.equal(fs.existsSync(out),false);
});

test('standalone source with direct local Math helper remains original on runtime replacement',async t=>{
  const {module}=await packageOf(t,rootSource,['f64[]']);const kernel=module.createKernel(),sqrt=Math.sqrt;
  const a=new Float64Array([4,9]);
  try{Math.sqrt=x=>x+10;kernel.run(a);}finally{Math.sqrt=sqrt;}
  same(a,[14,19]);assert.equal(kernel.diagnostics.fallbackCalls,1);assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_MATH_BINDING');
});

for(const replacement of ['function','getter','bound','proxy'])test(`runtime loaded after a Math method ${replacement} replacement refuses intrinsic admission`,async t=>{
  const artifact=compileNumericKernel('function update(a){for(let i=0;i<a.length;i++)a[i]=Math.sqrt(a[i]);}',{parameterTypes:['f64[]'],allowMath:true});
  const saved=Object.getOwnPropertyDescriptor(Math,'sqrt');let gets=0;
  try{
    let value;
    if(replacement==='function') value=function sqrt(x){return x+40;};
    if(replacement==='bound') value=saved.value.bind(Math);
    if(replacement==='proxy') value=new Proxy(saved.value,{apply(target,receiver,args){gets++;return Reflect.apply(target,receiver,args);}});
    if(replacement==='getter')Object.defineProperty(Math,'sqrt',{configurable:true,get(){gets++;return saved.value;}});
    else Object.defineProperty(Math,'sqrt',{...saved,value});
    const {instantiateNumericKernel}=await import(runtimeKernelUrl+'?prepatched='+replacement);
    assert.equal(gets,0);
    const kernel=instantiateNumericKernel(artifact.wasm,{resolveMath:()=>Math,fallback(a){for(let i=0;i<a.length;i++)a[i]=Math.sqrt(a[i]);}});
    const a=new Float64Array([4,9]);kernel.run(a);
    same(a,replacement==='function'?[44,49]:[2,3]);
    assert.equal(gets,replacement==='getter'||replacement==='proxy'?2:0);
    assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,1);
  }finally{Object.defineProperty(Math,'sqrt',saved);}
});
