import assert from 'node:assert/strict';
import {after,test} from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import * as acorn from 'acorn';
import {compileMarchingCubesBallBatch} from './marching_cubes_fields.mjs';
import {specializeMarchingCubesBallLoops} from './marching_cubes_ball_loops.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {readMarchingCubesOracle,loadOracle,sameEffect} from './fixtures/marching_cubes_oracle.mjs';

const source=readMarchingCubesOracle(),artifact=compileMarchingCubesBallBatch(source);
const oracle=await loadOracle();
after(()=>fs.rmSync(oracle.root,{recursive:true,force:true}));
const make=(size=12)=>new oracle.MarchingCubes(size,{flatShading:false},true,true,1000);
function balls(count) {
  const values=new Float64Array(count*8);
  for(let i=0;i<count;i++)values.set([.5+Math.sin(i*.7)*.2,.4+Math.cos(i)*.23,.5+Math.sin(i)*.19,
    i%4===0?-.3:.8,8+i%3,i%3/3,(i+1)%3/3,(i+2)%3/3],i*8);
  return values;
}
function reference(effect,values,count) {
  for(let i=0;i<count;i++)effect.addBall(...values.subarray(i*8,i*8+5),new oracle.three.Color(...values.subarray(i*8+5,i*8+8)));
}
test('multi-ball executable is derived from the complete pinned source and shares one transaction',()=>{
  assert.ok(WebAssembly.validate(artifact.wasm));
  assert.deepEqual(artifact.wasm,compileMarchingCubesBallBatch(source).wasm);
  assert.throws(()=>compileMarchingCubesBallBatch(source+'\n'),/pin mismatch/);
  assert.equal(artifact.manifest.parameters.filter(p=>p.write).length,2);
});
for(const size of [1,4,8,19,32])test(`one native batch preserves every ordered float store at resolution ${size}`,()=>{
  const a=make(size),b=make(size),kernel=instantiateNumericKernel(artifact.wasm,{resolveMath:()=>Math});
  for(const count of [0,1,7,50]) {
    const values=balls(count);
    for(const effect of [a,b]) {effect.reset();effect.field[0]=-0;effect.palette[1]=NaN;}
    kernel.run(a.field,a.palette,a.size,a.size2,values,count);reference(b,values,count);sameEffect(a,b);
  }
  assert.equal(kernel.diagnostics.wasmCalls,4);
});
test('batch retains nonfinite and signed-zero inputs without reassociating contributions',()=>{
  const a=make(),b=make(),kernel=instantiateNumericKernel(artifact.wasm,{resolveMath:()=>Math});
  const values=balls(8);
  values[0]=NaN;values[8+3]=-0;values[16+4]=Infinity;
  values[24+5]=NaN;values[32+6]=Infinity;values[40+7]=-Infinity;values[48+3]=-Infinity;
  kernel.run(a.field,a.palette,a.size,a.size2,values,8);reference(b,values,8);sameEffect(a,b);
});
for(const budget of [{maxIterations:1},{maxMemoryPages:1}])test(`batch guard ${JSON.stringify(budget)} publishes nothing before original replay`,()=>{
  const a=make(18),b=make(18),values=balls(7);
  const kernel=instantiateNumericKernel(compileMarchingCubesBallBatch(source,budget).wasm,{resolveMath:()=>Math});
  assert.throws(()=>kernel.run(a.field,a.palette,a.size,a.size2,values,7));
  sameEffect(a,b);
  reference(a,values,7);reference(b,values,7);sameEffect(a,b);
});
test('late array bounds and writable aliases retain all pre-call state',()=>{
  const a=make(),kernel=instantiateNumericKernel(artifact.wasm,{resolveMath:()=>Math}),values=balls(7);
  const before=a.field.slice();
  assert.throws(()=>kernel.run(a.field,new Float32Array(3),a.size,a.size2,values,7));
  assert.deepEqual(a.field,before);
  assert.throws(()=>kernel.run(a.field,a.field,a.size,a.size2,values,7));
  assert.deepEqual(a.field,before);
});

function transform(source,runtimeModule='adapter') {
  const result=specializeMarchingCubesBallLoops(source,{runtimeModule});
  let code=source;
  for(const edit of result.edits.sort((a,b)=>b.start-a.start))code=code.slice(0,edit.start)+edit.text+code.slice(edit.end);
  code=result.header+code;
  acorn.parse(code,{ecmaVersion:'latest',sourceType:'module'});
  return {...result,code};
}
const ordinary='export function update(owner,count,time){for(let i=0;i<count;i++){const x=Math.sin(i+time)*.2+.5;owner.addBall(x,.5,.5,.8,12);}}';
test('recognition is structural and preserves source scalar expressions, names and formulas',()=>{
  const result=transform(ordinary);
  assert.equal(result.report.compiledLoops,1);assert.ok(result.code.includes('Math.sin(i+time)*.2+.5'));
  const alternate=ordinary.replaceAll('owner','metaballs').replaceAll('count','amount').replaceAll('time','clock');
  assert.equal(transform(alternate).report.compiledLoops,1);
  const colored=ordinary.replace('time){','time,colors,mode){').replace('owner.addBall(x,.5,.5,.8,12);',
    'if(mode==="rgb")owner.addBall(x,.5,.5,.8,12,colors[i%7]);else owner.addBall(x,.5,.5,.8,12);');
  assert.equal(transform(colored).report.compiledLoops,1);
});
test('an ESM cycle can call the source function before its private recipe initializes',async()=>{
  const main=path.join(oracle.root,'cycle-main.mjs'),caller=path.join(oracle.root,'cycle-caller.mjs');
  fs.writeFileSync(main,transform('import "./cycle-caller.mjs"; export {observed} from "./cycle-caller.mjs";'+ordinary,
    new URL('./marching_cubes_adapter.mjs',import.meta.url).href).code,{flag:'wx'});
  fs.writeFileSync(caller,'import {update} from "./cycle-main.mjs"; export let observed=0;'+
    'update({addBall(){observed++;}},3,0);',{flag:'wx'});
  assert.equal((await import(pathToFileURL(main))).observed,3);
});
for(const [name,changed] of [
  ['field observation',ordinary.replace('Math.sin(i+time)*.2+.5','owner.field[i]')],
  ['helper effects',ordinary.replace('Math.sin(i+time)','observe(i+time)')],
  ['property capture',ordinary.replace('i+time','i+time.value')],
  ['written capture',ordinary.replace('const x=','time++;const x=')],
  ['source callback',ordinary.replace('owner.addBall','observe(owner);owner.addBall')],
  ['shadowed Math',ordinary.replace('time){','time,Math){')],
  ['global capture',ordinary.replace('count,time','count')],
  ['direct eval',ordinary+'\neval("0")'],
  ['other receiver',ordinary.replace('owner.addBall','getOwner().addBall')],
  ['escaped loop control',ordinary.replace('const x=','if(i===2)break;const x=')],
])test(`observable ${name} keeps the original whole loop`,()=>{
  const result=transform(changed);assert.equal(result.changed,false);assert.equal(result.code,changed);
});
