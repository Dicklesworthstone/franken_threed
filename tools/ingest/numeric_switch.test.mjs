import test from 'node:test';
import assert from 'node:assert/strict';
import {Color} from '../../upstream/three.js/src/math/Color.js';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {specializeNumericModule} from './numeric_specialization.mjs';

function compile(fn, parameterTypes, helpers = [], options = {}) {
  const source = typeof fn === 'string' ? fn : fn.toString();
  const settings = {parameterTypes, generalControl:true, allowMath:true,
    helperSources:new Map(helpers.map(helper => [helper.name,helper.toString()])), ...options};
  const artifact = compileNumericKernel(source, settings);
  assert.deepEqual(artifact.wasm, compileNumericKernel(source, settings).wasm);
  assert.ok(WebAssembly.validate(artifact.wasm));
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
  return artifact;
}
const runtime = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm,
  {preserveAliasing:true, resolveMath:() => Math, ...options});
const type = Type => Type === Float32Array ? 'f32[]' : 'f64[]';

// The pinned Color.getHSL numeric algorithm over packed RGB/HSL records. NaN
// initialization models its uninitialized hue when no case matches, without
// admitting undefined into the numeric ABI; color-space conversion stays out.
function hsl(out, rgb) {
  for (let i = 0; i < rgb.length; i += 3) {
    const r = rgb[i], g = rgb[i+1], b = rgb[i+2];
    const max = Math.max(r,g,b), min = Math.min(r,g,b), lightness = (min+max)/2;
    let hue = 0/0, saturation = 0/0;
    if (min === max) { hue = 0; saturation = 0; }
    else {
      const delta = max-min;
      saturation = lightness <= .5 ? delta/(max+min) : delta/(2-max-min);
      switch (max) {
        case r: hue = (g-b)/delta + (g < b ? 6 : 0); break;
        case g: hue = (b-r)/delta + 2; break;
        case b: hue = (r-g)/delta + 4; break;
      }
      hue /= 6;
    }
    out[i] = hue; out[i+1] = saturation; out[i+2] = lightness;
  }
}
for (const Type of [Float32Array,Float64Array]) test(Type.name + ': packed HSL conversion matches pinned Color.getHSL, including tied cases', () => {
  const input = new Type([1,0,0, 0,1,0, 0,0,1, 1,1,0, 0,1,1, 1,0,1,
    .5,.5,.5, .1,.6,.3, -0,0,-0, -1,.3,2, NaN,1,0, Infinity,1,0]);
  const expected = new Type(input.length), actual = new Type(input.length), color = new Color(), target = {};
  for (let i=0;i<input.length;i+=3) {
    color.setRGB(input[i],input[i+1],input[i+2]).getHSL(target);
    expected.set([target.h,target.s,target.l],i);
  }
  const kernel = runtime(compile(hsl,[type(Type),type(Type)]));
  kernel.run(actual,input);
  for (let i=0;i<actual.length;i++)
    assert.ok(Object.is(actual[i],expected[i]),`HSL element ${i} differs, including zero sign`);
  assert.equal(kernel.diagnostics.wasmCalls,1); assert.equal(kernel.diagnostics.fallbackCalls,0);
  kernel.run(new Type(0),new Type(0));
  assert.equal(kernel.diagnostics.wasmCalls,2);
});

test('Number cases use strict equality, preserve first match, and never truncate discriminants', () => {
  function route(value) {
    let result = -1;
    switch (value) {
      case 0: result = 10; break;
      case -0: result = 20; break;
      case 0/0: result = 30; break;
      case 1/0: result = 40; break;
      case 4294967297: result = 50; break;
      case .5: result = 60; break;
    }
    return result;
  }
  const artifact = compile(route,['f64']), kernel = runtime(artifact);
  assert.equal(artifact.manifest.loopCount,0);
  for (const value of [0,-0,NaN,Infinity,-Infinity,1,4294967297,.5])
    assert.equal(kernel.run(value),route(value));
  assert.equal(kernel.diagnostics.wasmCalls,8);
});

test('Boolean discriminants and cases cannot become numeric zero/one matches', () => {
  const update = 'function update(out,value){out[0]=choose(value);}';
  for (const [head,label] of [
    ['value','value > 0'], ['value > 0','1'], ['value','true'], ['false','0'],
    ['value','!value'], ['!value','1'], ['value','value > 0 ? 1 : false'],
  ]) {
    const source = `function choose(value){let result=7;switch(${head}){case ${label}:result=5;break;}return result;}`;
    assert.throws(()=>compile(source,['f64']),{code:'KERNEL_NOT_CLOSED'});
    assert.throws(()=>compile(update,['f64[]','f64'],[],{
      helperSources:new Map([['choose',source]]),
    }),{code:'KERNEL_NOT_CLOSED'});
  }
  function route(value) {
    let result = 7;
    switch(value) { case 0: result=2; break; case 1: result=5; break; }
    return result;
  }
  let calls=0;
  const kernel=runtime(compile(route,['f64']),{fallback(value){calls++;return route(value);}});
  for (const value of [true,false]) assert.equal(kernel.run(value),7);
  assert.equal(calls,2); assert.equal(kernel.diagnostics.wasmCalls,0);
  for (const value of [0,1]) assert.equal(kernel.run(value),route(value));
  assert.equal(kernel.diagnostics.wasmCalls,2);
});

test('default at every position, empty case labels, missing matches and fallthrough retain source order', () => {
  for (const cases of [
    'default: out[0]+=1; case 0: out[0]*=2; case 1: case 2: out[0]+=3; break; case 3: out[0]=7;',
    'case 0: out[0]*=2; default: out[0]+=1; case 1: case 2: out[0]+=3; break; case 3: out[0]=7;',
    'case 0: out[0]*=2; case 1: case 2: out[0]+=3; break; case 3: out[0]=7; default: out[0]+=1;',
    'case 0: out[0]*=2; case 1: case 2: out[0]+=3; break; case 3: out[0]=7;',
    '',
  ]) {
    const source = `function route(out,mode){switch(mode){${cases}}return out[0];}`;
    const original = Function('return ('+source+')')(), kernel = runtime(compile(source,['f64[]','f64']));
    for (const mode of [-1,0,1,2,3,4]) {
      const out = new Float64Array([5]), expected = out.slice();
      assert.equal(kernel.run(out,mode),original(expected,mode));
      assert.deepEqual(out,expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls,6);
  }
});

test('discriminant and effectful helper cases run once, lazily and before the selected body', () => {
  function mark(state,value) { state[0] = state[0]*10+value; return value; }
  function route(state,value) {
    switch (mark(state,value)) {
      case mark(state,1): state[0]+=100; break;
      default: state[0]+=200;
      case mark(state,2): state[0]+=300; break;
      case mark(state,3): state[0]+=400;
    }
    return state[0];
  }
  const kernel = runtime(compile(route,['f64[]','f64'],[mark]));
  for (const value of [0,1,2,3,4]) {
    const state = new Float64Array(1), expected = state.slice();
    assert.equal(kernel.run(state,value),route(expected,value));
    assert.deepEqual(state,expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls,5);
});

test('nested switches break locally while continue reaches the correct enclosing loop', () => {
  function route(out,values) {
    for (const value of values) {
      switch (value) {
        case 0: continue;
        case 1:
          for (let i=0;i<4;i++) {
            switch(i) { case 1: continue; case 2: break; default: out[0]+=i; }
            out[0]+=10;
          }
          break;
        default:
          switch(value) { case 2: out[0]+=100; break; default: out[0]+=200; }
      }
      out[0]+=1000;
    }
    return out[0];
  }
  const kernel = runtime(compile(route,['f64[]','f64[]']));
  const values = new Float64Array([0,1,2,3]), out = new Float64Array(1), expected = out.slice();
  assert.equal(kernel.run(out,values),route(expected,values));
  assert.deepEqual(out,expected); assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('helper switch returns are complete through default/fallthrough and preserve outer loops', () => {
  function choose(value) {
    switch(value) {
      case 0: case 1: return value+10;
      default: { const negative = -value; return negative; }
    }
  }
  function update(out,values) { for(const value of values) out[0]+=choose(value); return out[0]; }
  const kernel = runtime(compile(update,['f64[]','f64[]'],[choose]));
  const values = new Float64Array([0,1,2,-1]), out = new Float64Array(1), expected = out.slice();
  assert.equal(kernel.run(out,values),update(expected,values));
  assert.deepEqual(out,expected); assert.equal(kernel.diagnostics.wasmCalls,1);
  for (const source of [
    'function choose(value){switch(value){case 0:return 1;}}',
    'function choose(value){switch(value){case 0:break;return 1;default:return 2;}}',
  ]) assert.throws(()=>compile(update,['f64[]','f64[]'],[],{helperSources:new Map([['choose',source]])}),{code:'KERNEL_NOT_CLOSED'});
});

test('switch discriminant uses the outer scope and cases keep lexical TDZ and block-local declarations', () => {
  function route(out,mode) {
    let value = mode;
    switch(value) {
      case 0: let value = 3; out[0]+=value; break;
      case 1: { const value = 4; out[0]+=value; break; }
      default: out[0]+=5;
    }
    return value+out[0];
  }
  const kernel = runtime(compile(route,['f64[]','f64']));
  for (const mode of [0,1,2]) {
    const out = new Float64Array(1), expected = out.slice();
    assert.equal(kernel.run(out,mode),route(expected,mode));
    assert.deepEqual(out,expected);
  }
  for (const source of [
    'function f(x){let y=1;switch(x){case y:let y=2;break;}return x;}',
    'function f(x){switch(x){case 0:let y=2;case 1:return y;}return x;}',
    'function f(x){switch(x){case 0:let y=2;}return y;}',
    'function f(x){switch(x){case 0:var y=2;}return x;}',
    'function f(x){switch(x){case "0":return 2;}return x;}',
  ]) assert.throws(()=>compile(source,['f64']),{code:'KERNEL_NOT_CLOSED'});
});

test('skipped invalid case reads stay native; late failures roll back before one source fallback', () => {
  function route(out,values,mode) {
    out[0]+=1;
    switch(mode) { case 0: out[0]+=2; break; case values[4]: out[0]+=3; break; default: out[0]+=4; }
    return out[0];
  }
  let calls=0;
  const kernel=runtime(compile(route,['f64[]','f64[]','f64']),{fallback(...args){calls++;return route(...args);}});
  const out=new Float64Array([10]),values=new Float64Array([1]);
  assert.equal(kernel.run(out,values,0),13); assert.equal(kernel.diagnostics.wasmCalls,1);
  const expected=out.slice();
  assert.equal(kernel.run(out,values,2),route(expected,values,2));
  assert.deepEqual(out,expected); assert.equal(calls,1); assert.equal(kernel.diagnostics.fallbackCalls,1);
  function budget(out,mode) { out[0]++; switch(mode){case 0:for(let i=0;i<5;i++)out[0]+=i;break;default:out[0]--;} }
  const bounded=runtime(compile(budget,['f64[]','f64'],[],{maxIterations:2}),{fallback:budget});
  const actual=new Float64Array([1]),reference=actual.slice();
  bounded.run(actual,0);budget(reference,0);
  assert.deepEqual(actual,reference);assert.equal(bounded.diagnostics.fallbackCalls,1);
});

test('automatic function and callback routes discover switches without replaying surrounding effects', async () => {
  const url=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
  const runtimeURL=new URL('./numeric_dispatch.mjs',import.meta.url).href;
  const observer=url(`import * as n from ${JSON.stringify(runtimeURL)};export * from ${JSON.stringify(runtimeURL)};
    const records=[];export function createNumericDispatch(...a){const t=n.createNumericDispatch(...a);records.push(()=>n.numericDispatchDiagnostics(t));return t;}
    export function createNumericLoopDispatch(...a){const t=n.createNumericLoopDispatch(...a);records.push(()=>n.numericLoopDispatchDiagnostics(t));return t;}
    export function diagnostics(){return records.map(r=>r());}`);
  const source=`function update(a,modes){for(let i=0;i<a.length;i++){switch(modes[i]){case 0:a[i]+=2;break;default:a[i]*=3;}}}
    export function run(a,modes){update(a,modes);}
    export function make(a,modes,events){return ()=>{events.push('before');for(let i=0;i<a.length;i++){switch(modes[i]){case 0:a[i]+=1;break;default:a[i]-=2;}}events.push('after');};}`;
  const result=specializeNumericModule(source,{runtimeModule:observer,loopIslands:true});
  assert.equal(result.report.functionKernels,1);assert.equal(result.report.loopIslands.compiledKernels,1);
  const app=await import(url(result.code)),reference=await import(url(source));
  const modes=new Float64Array([0,1]),actual=new Float64Array([1,2]),expected=actual.slice();
  app.run(actual,modes);reference.run(expected,modes);assert.deepEqual(actual,expected);
  const events=[],originalEvents=[];
  app.make(actual,modes,events)();reference.make(expected,modes,originalEvents)();
  assert.deepEqual(actual,expected);assert.deepEqual(events,originalEvents);
  const diagnostics=(await import(observer)).diagnostics();
  assert.equal(diagnostics.reduce((sum,item)=>sum+item.variants.reduce((s,v)=>s+(v.kernel?.wasmCalls??0),0),0),2);
});
