import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {compileMarchingCubesFields} from './marching_cubes_fields.mjs';
import {readMarchingCubesOracle,loadOracle,sameEffect} from './fixtures/marching_cubes_oracle.mjs';

const source=readMarchingCubesOracle(),artifacts=compileMarchingCubesFields(source);
const runtime=new URL('./numeric_kernel_runtime.mjs',import.meta.url).href;
const expected=await loadOracle();
// Only this numerical harness reads arbitrary properties. Production admission
// is tested separately through the real addon adapter and genuine Three objects.
async function instrument(options) {
  const fields=options?compileMarchingCubesFields(source,options):artifacts;
  let code=source;
  for (const field of [...fields].sort((a,b)=>b.sourceSpan.start-a.sourceSpan.start)) {
    const {start,end}=field.sourceSpan;
    const invocation=`__testField(${JSON.stringify(field.name)},this,[${field.locals.join(',')}],${field.color?'ballColor':'null'})`;
    code=code.slice(0,start)+`if (!${invocation}) {\n`+code.slice(start,end)+'\n}'+code.slice(end);
  }
  code=`import {runField as __testField} from './field-harness.mjs';\n`+code;
  const records=fields.map(field=>({name:field.name,parameters:field.parameters,locals:field.locals,
    base64:Buffer.from(field.wasm).toString('base64')}));
  const harness=`import {instantiateNumericKernel} from ${JSON.stringify(runtime)};
    const records=${JSON.stringify(records)};
    const kernels=new Map(records.map(record=>[record.name,{...record,
      kernel:instantiateNumericKernel(Uint8Array.from(atob(record.base64),c=>c.charCodeAt(0)),{resolveMath:()=>Math}),calls:0,refusals:0}]));
    export function diagnostics(name) {const r=kernels.get(name);return {calls:r.calls,refusals:r.refusals,...r.kernel.diagnostics};}
    export function runField(name,owner,locals,color) {
      const record=kernels.get(name);
      const args=record.parameters.map(p=>p.kind==='owner'?owner[p.property]:p.kind==='color'?color[p.property]:locals[record.locals.indexOf(p.property)]);
      try {record.kernel.run(...args);record.calls++;return true;}
      catch {record.refusals++;return false;}
    }`;
  const oracle=await loadOracle({source:code,files:{'field-harness.mjs':harness}});
  const bridge=await import(pathToFileURL(path.join(oracle.root,'field-harness.mjs')).href);
  return {...oracle,...bridge};
}
const actual=await instrument();
const pair=(size=8)=>[actual,expected].map(m=>new m.MarchingCubes(size,{flatShading:false},true,true,1000));
function invoke(a,b,name,args=[]) {
  assert.equal(a[name](...args),undefined);assert.equal(b[name](...args),undefined);sameEffect(a,b);
}
const cleanup=[expected.root,actual.root];
process.on('exit',()=>{for (const root of cleanup)fs.rmSync(root,{recursive:true,force:true});});

test('six field kernels are derived from exact source loops with bounded import-free executables',()=>{
  const again=compileMarchingCubesFields(source);
  assert.deepEqual(artifacts.map(a=>a.name),['addBall','addPlaneX','addPlaneY','addPlaneZ','blur','reset']);
  artifacts.forEach((artifact,i)=>{
    assert.deepEqual(artifact.wasm,again[i].wasm);
    assert.ok(WebAssembly.validate(artifact.wasm));
    assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)),[]);
    assert.equal(artifact.manifest.automaticRouteAdmission,false);
    assert.equal(artifact.manifest.maxMemoryPages,2048);
    assert.equal(artifact.manifest.maxIterations,100000000);
    assert.equal(artifact.manifest.resultType,'void');
    assert.ok(artifact.manifest.loopCount>=1);
  });
  assert.deepEqual(artifacts[0].manifest.mathIntrinsics,['sqrt']);
  assert.throws(()=>compileMarchingCubesFields(source+'\n'),/pin mismatch/);
});

for(const size of [1,3,4,9,18,32])test(`whole field-building sequences preserve Float32 stores and cache state at resolution ${size}`,()=>{
  const [a,b]=pair(size),before=actual.diagnostics('addBall').calls;
  for(const o of [a,b])o.normal_cache.fill(12.5);
  for(let frame=0;frame<5;frame++) {
    invoke(a,b,'reset');
    invoke(a,b,'addBall',[0.41+frame/53,0.51,0.57,0.8,12,[0.1,0.4,0.8]]);
    invoke(a,b,'addBall',[0.61,0.43,0.49,-0.19,8,[-2,0.3,0.7]]);
    invoke(a,b,'addPlaneX',[0.11,5]);invoke(a,b,'addPlaneY',[0.16,7]);invoke(a,b,'addPlaneZ',[0.12,9]);
    if(frame===2)invoke(a,b,'blur',[0.375]);
    if(frame===3)invoke(a,b,'blur');
  }
  assert.equal(actual.diagnostics('addBall').calls-before,10);
  for(const o of [a,b])o.init(6);
  invoke(a,b,'addBall',[0.4,0.5,0.6,0.6,8]);invoke(a,b,'blur',[0.7]);
});

test('seeded field edits and blur preserve edge-neighbor ordering, nonfinite values, and signed zeros',()=>{
  const [a,b]=pair(7),before=actual.diagnostics('blur').calls;
  let seed=0x1425abcd;
  for(let frame=0;frame<16;frame++) {
    for(let i=0;i<a.field.length;i++) {
      seed=(Math.imul(seed,1664525)+1013904223)>>>0;
      a.field[i]=b.field[i]=(seed%1009-509)/33;
    }
    a.field[0]=b.field[0]=-0;
    a.field[19]=b.field[19]=[NaN,Infinity,-Infinity,1e-45][frame%4];
    invoke(a,b,'blur',[[0,1,-0.25,NaN,Infinity,-Infinity][frame%6]]);
  }
  assert.equal(actual.diagnostics('blur').calls-before,16);
});

test('reset clears only the normal-cache x sentinels and leaves output arrays and list history intact',()=>{
  const [a,b]=pair(5);
  for(const o of [a,b]){
    o.normal_cache.forEach((_,i)=>o.normal_cache[i]=i+0.5);
    o.field.fill(-0);o.palette.fill(NaN);o.positionArray.fill(-0);o.normalArray.fill(93);
    o.__f3dLists[0].fill(71);
  }
  invoke(a,b,'reset');
  assert.equal(a.normal_cache[0],0);assert.equal(a.normal_cache[1],1.5);assert.equal(a.normal_cache[2],2.5);
  assert.equal(a.__f3dLists[0][0],71);assert.ok(Object.is(a.positionArray[0],-0));
});

for(const options of [{maxMemoryPages:1},{maxIterations:1}])test(`field transactions roll back and run original loops with budget ${JSON.stringify(options)}`,async()=>{
  const limited=await instrument(options);cleanup.push(limited.root);
  const [a,b]=[limited,expected].map(m=>new m.MarchingCubes(18,{flatShading:false},true,true,1000));
  for(const name of ['addBall','addPlaneX','addPlaneY','addPlaneZ','blur','reset']){
    const args=name==='addBall'?[0.5,0.5,0.5,1.2,8,[0.1,0.2,0.3]]:name.startsWith('addPlane')?[1.2,8]:[];
    invoke(a,b,name,args);
    // A single field fits in one page, but addBall/reset pack multiple views.
    if(options.maxIterations || name==='addBall' || name==='reset')assert.equal(limited.diagnostics(name).refusals,1,name);
  }
});

test('late checked-index failure and writable aliasing retain the complete original field operation',()=>{
  const [a,b]=pair(8);
  a.palette=a.field;b.palette=b.field;
  invoke(a,b,'addBall',[0.5,0.5,0.5,0.8,9,[0.2,0.3,0.4]]);
  assert.equal(actual.diagnostics('addBall').lastGuardFailure,'KERNEL_ARRAY_ALIAS');
  a.palette=new Float32Array(7);b.palette=new Float32Array(7);
  invoke(a,b,'addBall',[0.5,0.5,0.5,0.8,9,[0.2,0.3,0.4]]);
  assert.equal(actual.diagnostics('addBall').lastGuardFailure,'KERNEL_EXECUTION_FAILED');
  a.init(8);b.init(8);invoke(a,b,'addBall',[0.5,0.5,0.5,0.8,9]);
  assert.equal(actual.diagnostics('addBall').lastGuardFailure,null);
});
