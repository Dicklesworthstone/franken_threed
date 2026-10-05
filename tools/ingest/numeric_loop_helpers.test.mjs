import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {createNumericLoopDispatch} from './numeric_dispatch.mjs';

const runtime = new URL('./numeric_dispatch.mjs', import.meta.url).href;
async function application(t, source, options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-loop-helpers-'));
  t.after(() => fs.rmSync(root, {recursive:true, force:true}));
  const url = name => pathToFileURL(path.join(root, name)).href;
  fs.writeFileSync(path.join(root, 'observer.mjs'), `
    import {createNumericLoopDispatch as create, numericLoopDispatchDiagnostics} from ${JSON.stringify(runtime)};
    export * from ${JSON.stringify(runtime)};
    const tokens = [];
    export function createNumericLoopDispatch(...args) { const token=create(...args); tokens.push(token); return token; }
    export function diagnostics() { return tokens.map(numericLoopDispatchDiagnostics); }
  `);
  const result = specializeNumericModule(source, {
    sourceName:'application.mjs', runtimeModule:url('observer.mjs'), loopIslands:true, ...options,
  });
  fs.writeFileSync(path.join(root, 'application.mjs'), result.code);
  fs.writeFileSync(path.join(root, 'reference.mjs'), source);
  return {result, module:await import(url('application.mjs')), reference:await import(url('reference.mjs')),
    diagnostics:(await import(url('observer.mjs'))).diagnostics};
}
const regions = app => app.result.report.loopIslands?.candidates.filter(c => c.route === 'guarded-loop-wasm') ?? [];
const calls = (app, key = 'wasmCalls') => app.diagnostics().reduce((sum, record) =>
  sum + record.variants.reduce((n, variant) => n + (variant.kernel?.[key] ?? 0), 0), 0);
const same = (a, b) => {
  assert.equal(a.length, b.length);
  for (let i=0;i<a.length;i++) assert.ok(Object.is(a[i],b[i]), `element ${i}: ${a[i]} != ${b[i]}`);
};

test('callback/method helper chains execute once per region with live per-instance captures and scalar publication', async t => {
  const source = `
    export const events=[];
    function clamp(x){if(x<0)return 0;if(x>1)return 1;return x;}
    function mix(a,b,t){return a+(b-a)*clamp(t);}
    export {mix}; export const identity=mix;
    export function make(out,goal){let total=0;return {tag:'frame',step(dt){
      events.push(['before',this.tag]);
      for(let i=0;i<out.length;i++){out[i]=mix(out[i],goal[i],dt);total+=out[i];}
      events.push(['after',total]);return total;
    }}}
  `;
  const app = await application(t, source);
  assert.equal(regions(app).length,1);
  assert.deepEqual(regions(app)[0].helpers.map(h=>h.name),['mix','clamp']);
  assert.deepEqual(regions(app)[0].scalarOutputs,['total']);
  assert.equal(app.module.mix,app.module.identity);
  assert.equal(app.module.mix.toString(),app.reference.mix.toString());
  assert.deepEqual(Object.keys(app.module),Object.keys(app.reference));
  for(const Type of [Float32Array,Float64Array]) {
    const a=new Type([1,2,3]), b=new Type([4,5,6]), ea=a.slice(), eb=b.slice(), goal=new Type([7,8,9]);
    const actors=[app.module.make(a,goal),app.module.make(b,goal)];
    const refs=[app.reference.make(ea,goal),app.reference.make(eb,goal)];
    for(const dt of [0,0.2,1.2,-1]) for(let i=0;i<actors.length;i++)
      assert.equal(actors[i].step(dt),refs[i].step(dt));
    same(a,ea);same(b,eb);
  }
  assert.deepEqual(app.module.events,app.reference.events);
  assert.equal(calls(app),16); assert.equal(calls(app,'fallbackCalls'),0);
  assert.equal(app.result.report.accelerated,false);
  assert.equal(app.result.report.loopIslands.accelerated,false);
  for(const helper of regions(app)[0].helpers)
    assert.ok(source.slice(helper.sourceSpan.start,helper.sourceSpan.end).startsWith(`function ${helper.name}`));
});

for(const Type of [Float64Array,Float32Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array]) {
  test(`${Type.name}: helper-only array slots, reordered forwarding and per-store conversion`, async t => {
    const app=await application(t,`
      export const data=new ${Type.name}([0,-0,1/3,1.5,2.5,-3.5,255,65537]);
      export const effects=[];
      function put(a,i,value){a[i]+=value;return a[i];}
      function forward(value,a,i){return put(a,i,value);}
      export function frame(count,delta){let sum=0;effects.push('before');
        for(let i=0;i<count;i++)sum+=forward(delta,data,i);
        effects.push(sum);return sum;
      }
    `);
    assert.equal(regions(app).length,1);
    assert.ok(regions(app)[0].captures.some(p=>p.name==='data' && p.read && p.write));
    for(const delta of [0,0.5,-1.5,2**32+1,NaN]) {
      assert.ok(Object.is(app.module.frame(8,delta),app.reference.frame(8,delta)));
      same(app.module.data,app.reference.data);
    }
    assert.deepEqual(app.module.effects,app.reference.effects);
    assert.equal(calls(app),5);assert.equal(calls(app,'fallbackCalls'),0);
  });
}

for(const Type of [Float32Array,Float64Array]) {
  test(`${Type.name}: void helper calls share shifted/identical aliases without intermediate publication`, async t => {
    const app=await application(t,`
      function mutate(a,b,i){a[i]+=b[i]/3;b[i]-=a[i]/7;}
      function forward(i,b,a){mutate(a,b,i);}
      export function run(a,b,n){for(let i=0;i<n;i++)forward(i,b,a);}
    `);
    assert.equal(regions(app).length,1);
    for(const [ao,bo] of [[0,0],[0,1],[1,0],[0,2],[2,0]]) {
      const data=new Type([1,2,3,4,5,6,7]), expected=data.slice();
      app.module.run(data.subarray(ao,ao+5),data.subarray(bo,bo+5),5);
      app.reference.run(expected.subarray(ao,ao+5),expected.subarray(bo,bo+5),5);
      same(data,expected);
    }
    assert.equal(calls(app),5);assert.equal(calls(app,'fallbackCalls'),0);
  });
}

test('a bounds trap rolls back helper writes and scalar state before the original loop runs once', async t => {
  const app=await application(t,`
    export const effects=[];
    function change(a,b,i){a[i]+=1;a[i]+=b[i];return a[i];}
    export function make(a,b){let total=100;return n=>{
      effects.push(['before',total]);
      for(let i=0;i<n;i++)total+=change(a,b,i);
      effects.push(['after',total]);return total;
    }}
  `);
  const a=new Float64Array([1,2,3,4]), expected=a.slice();
  const actual=app.module.make(a,a.subarray(0,2)), reference=app.reference.make(expected,expected.subarray(0,2));
  assert.ok(Object.is(actual(4),reference(4)));same(a,expected);
  assert.deepEqual(app.module.effects,app.reference.effects);
  assert.equal(calls(app),0);assert.equal(calls(app,'fallbackCalls'),1);
});

test('non-number captures preserve coercion ordering and do not replay callback effects', async t => {
  const app=await application(t,`
    export const effects=[];
    function add(x,y){return x+y;}
    export function run(a,value){effects.push('before');
      for(let i=0;i<a.length;i++)a[i]=add(a[i],value);
      effects.push('after');
    }
  `);
  const a=new Float64Array([1,2,3]), expected=a.slice();
  const value=effects=>({valueOf(){effects.push('coerce');return 2;}});
  app.module.run(a,value(app.module.effects));app.reference.run(expected,value(app.reference.effects));
  same(a,expected);assert.deepEqual(app.module.effects,app.reference.effects);
  assert.equal(calls(app),0);assert.equal(calls(app,'fallbackCalls'),1);
});

const shadows=[
  ['parameter plus disjoint block local',`export function run(a,h){for(let i=0;i<a.length;i++){a[i]=h(a[i]);{let h=3;a[i]+=h;}}}`,true],
  ['catch binding',`export function run(a,h){try{throw h;}catch(h){for(let i=0;i<a.length;i++)a[i]=h(a[i]);}}`,true],
  ['block function',`export function run(a){ {function h(x){return x+7;}for(let i=0;i<a.length;i++)a[i]=h(a[i]);}}`,true],
  ['body var',`export function run(a){for(let i=0;i<a.length;i++)a[i]=h(a[i]);var h;}`,false],
  ['loop-local binding',`export function run(a){for(let i=0;i<a.length;i++){let h=3;a[i]=h(a[i]);}}`,false],
];
for(const [name,body,works] of shadows) {
  test(`retains the actual ${name}, never a same-named module helper`, async t => {
    const app=await application(t,`function h(x){return x*2;} ${body}`);
    assert.equal(regions(app).length,0);
    const a=new Float64Array([1,2]), expected=a.slice(), custom=x=>x+9;
    if(works){app.module.run(a,custom);app.reference.run(expected,custom);}
    else {assert.throws(()=>app.module.run(a),TypeError);assert.throws(()=>app.reference.run(expected),TypeError);}
    same(a,expected);assert.equal(calls(app),0);
  });
}

test('parameter initializer loops do not see body var bindings', async t => {
  const app=await application(t,`
    function h(x){return x*2;}
    export function run(a,value=(()=>{for(let i=0;i<a.length;i++)a[i]=h(a[i]);return a;})()){
      var h;return value;
    }
  `);
  assert.equal(regions(app).length,1);
  const a=new Float64Array([1,2]), expected=a.slice();
  app.module.run(a);app.reference.run(expected);same(a,expected);assert.equal(calls(app),1);
});

test('switch discriminant loops do not borrow a case-block helper binding', async t => {
  const app=await application(t,`
    function h(x){return x*2;}
    export function run(a){switch((()=>{for(let i=0;i<a.length;i++)a[i]=h(a[i]);return 0;})()){
      case 0: function h(x){return x+100;} break;
    }}
  `);
  assert.equal(regions(app).length,1);
  const a=new Float64Array([1,2]), expected=a.slice();
  app.module.run(a);app.reference.run(expected);same(a,expected);assert.equal(calls(app),1);
});

for(const [name,source] of [
  ['mutable helper',`function h(x){return x*2;} export function replace(value){h=value;}`],
  ['captured helper state',`let offset=3;function h(x){return x+offset;}`],
  ['recursive helper',`function h(x){if(x>0)return h(x-1)+1;return 0;}`],
]) {
  test(`${name} remains original JavaScript`, async t=>{
    const app=await application(t,`${source} export function run(a){for(let i=0;i<a.length;i++)a[i]=h(a[i]);}`);
    assert.equal(regions(app).length,0);
    if(app.module.replace){app.module.replace(x=>x+8);app.reference.replace(x=>x+8);}
    const a=new Float64Array([1,2,3]), expected=a.slice();
    app.module.run(a);app.reference.run(expected);same(a,expected);assert.equal(calls(app),0);
  });
}

test('helper-only Math uses its live module binding, not an unrelated callback parameter', async t=>{
  const app=await application(t,`
    export let Math=globalThis.Math;
    export function setMath(value){Math=value;}
    function magnitude(x){return Math.abs(x);}
    export function run(a,Math){for(let i=0;i<a.length;i++)a[i]=magnitude(a[i]);}
  `);
  assert.equal(regions(app)[0].helperMathBinding,'live-module-lexical-environment');
  const a=new Float64Array([-2,-3]), expected=a.slice();
  const unrelated={abs(){throw Error('must not read callback Math');}};
  app.module.run(a,unrelated);app.reference.run(expected,unrelated);same(a,expected);assert.equal(calls(app),1);
  const custom={abs:x=>x+100};app.module.setMath(custom);app.reference.setMath(custom);
  app.module.run(a,Math);app.reference.run(expected,Math);same(a,expected);
  assert.equal(calls(app),1);assert.equal(calls(app,'fallbackCalls'),1);
  app.module.setMath(Math);app.reference.setMath(Math);
  app.module.run(a,unrelated);app.reference.run(expected,unrelated);same(a,expected);assert.equal(calls(app),2);
});

test('root and helper Math environments must both pass their own live guard', async t=>{
  const app=await application(t,`
    export let Math=globalThis.Math;
    export function setMath(value){Math=value;}
    function magnitude(x){return Math.abs(x);}
    export function run(a,Math){for(let i=0;i<a.length;i++)a[i]=magnitude(a[i])+Math.sqrt(a[i]*a[i]);}
  `);
  const a=new Float64Array([-2,-3]), expected=a.slice();
  for(const [moduleMath,localMath] of [[Math,Math],[Math,{sqrt:x=>x+1}],[{abs:x=>x-10},Math],[Math,Math]]) {
    app.module.setMath(moduleMath);app.reference.setMath(moduleMath);
    app.module.run(a,localMath);app.reference.run(expected,localMath);same(a,expected);
  }
  assert.equal(calls(app),2);assert.equal(calls(app,'fallbackCalls'),2);
});

test('lazy module Math guards preserve zero-trip TDZ behavior and recover after initialization', async t=>{
  const app=await application(t,`
    export const effects=[],data=new Float64Array([-2,-3]);
    function magnitude(x){return Math.abs(x);}
    export function run(n){effects.push('before');for(let i=0;i<n;i++)data[i]=magnitude(data[i]);effects.push('after');}
    run(0);
    let Math=globalThis.Math;
  `);
  assert.equal(regions(app).length,1);
  same(app.module.data,app.reference.data);
  assert.deepEqual(app.module.effects,['before','after']);
  assert.equal(calls(app),0);assert.equal(calls(app,'fallbackCalls'),1);
  app.module.run(2);app.reference.run(2);same(app.module.data,app.reference.data);
  assert.deepEqual(app.module.effects,app.reference.effects);assert.equal(calls(app),1);
});

test('a global Math accessor is not invoked speculatively by either lexical resolver', async t=>{
  const app=await application(t,`
    function magnitude(x){return Math.abs(x);}
    export function run(a){for(let i=0;i<a.length;i++)a[i]=magnitude(a[i]);}
  `);
  const original=Object.getOwnPropertyDescriptor(globalThis,'Math'), intrinsic=Math;
  const a=new Float64Array([-2,-3]), expected=a.slice();let reads=0;
  try {
    Object.defineProperty(globalThis,'Math',{configurable:true,get(){reads++;return intrinsic;}});
    app.module.run(a);const nativeReads=reads;reads=0;app.reference.run(expected);
    assert.equal(nativeReads,reads);assert.equal(reads,2);same(a,expected);
  } finally {Object.defineProperty(globalThis,'Math',original);}
  assert.equal(calls(app),0);assert.equal(calls(app,'fallbackCalls'),1);
});

test('bounds/work fallback refreshes captures and never publishes a partial reduction', async t=>{
  const app=await application(t,`
    function step(a,i){a[i]+=1;return a[i];}
    export function make(a){let total=0;return n=>{for(let i=0;i<n;i++)total+=step(a,i);return total;}}
  `,{maxIterations:2});
  const a=new Float32Array([1,2,3]), expected=a.slice(), run=app.module.make(a), ref=app.reference.make(expected);
  for(const n of [2,3,2]) {assert.equal(run(n),ref(n));same(a,expected);}
  assert.equal(calls(app),2);assert.equal(calls(app,'fallbackCalls'),1);
});

test('missing Wasm preserves ordinary callback and helper execution', async t=>{
  const app=await application(t,`
    export const effects=[];function h(x){return x*2;}
    export function run(a){effects.push('before');for(let i=0;i<a.length;i++)a[i]=h(a[i]);effects.push('after');}
  `);
  const wasm=globalThis.WebAssembly, a=new Float64Array([1,2]), expected=a.slice();
  try {globalThis.WebAssembly=undefined;app.module.run(a);} finally {globalThis.WebAssembly=wasm;}
  app.reference.run(expected);same(a,expected);assert.deepEqual(app.module.effects,app.reference.effects);
  assert.equal(calls(app),0);assert.equal(app.diagnostics()[0].retainedCalls,1);
});

test('helper support remains opt-in and rejects invalid module Math resolvers',()=>{
  const source=`function h(x){return x*2;} export function make(a){return ()=>{for(let i=0;i<a.length;i++)a[i]=h(a[i]);}}`;
  assert.equal(specializeNumericModule(source).changed,false);
  assert.throws(()=>createNumericLoopDispatch([],[],42),TypeError);
});
