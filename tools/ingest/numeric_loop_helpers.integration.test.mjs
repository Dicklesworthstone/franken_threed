/** Real application builds and relocated execution; no fake linker or Wasm VM. */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {bundleWithRollup} from './bundler.mjs';

const HELPERS=`
  export function integrate(out,velocity,dt){for(let j=0;j<out.length;j++)out[j]+=velocity[j]*dt;}
  export function damp(velocity,factor){for(let j=0;j<velocity.length;j++)velocity[j]*=factor;}
  export function energy(velocity){let total=0;for(let j=0;j<velocity.length;j++)total+=velocity[j]*velocity[j];return total;}
`;
const FRAME=`
  import {integrate,damp,energy} from './helpers.mjs';
  export function make(out,velocity,upload){let total=0;return {tag:'frame',step(dt,count){
    const tag=this.tag;
    for(let i=0;i<count;i++){integrate(out,velocity,dt);damp(velocity,0.99);total+=energy(velocity);}
    upload({tag,positions:[...out],total});return total;
  }}}
`;
function fixture(t,frame=FRAME) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-loop-helper-build-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  const src=path.join(root,'src');fs.mkdirSync(src);
  for(const [name,code] of Object.entries({'entry.mjs':`export const load=()=>import('./frame.mjs');`,'frame.mjs':frame,'helpers.mjs':HELPERS}))
    fs.writeFileSync(path.join(src,name),code);
  return {root,src,entry:path.join(src,'entry.mjs'),out:path.join(root,'dist')};
}
function observeWasm(t) {
  const Native=WebAssembly.Instance,counts={instances:0,attempts:0,calls:0};
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){
    const instance=Reflect.construct(Native,args);counts.instances++;
    return {exports:{...instance.exports,run(...parameters){
      counts.attempts++;const result=instance.exports.run(...parameters);counts.calls++;return result;
    }}};
  };
  return counts;
}
const same=(a,b)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Object.is(a[i],b[i]),`element ${i}: ${a[i]} != ${b[i]}`);};
function verifyFrames(module,reference,Type=Float32Array,frames=30) {
  // Shifted writable aliases cross helper boundaries in both source and Wasm.
  const data=new Type([1,2,3,4,5,6,7,8,9]),expected=data.slice();
  const out=data.subarray(0,8),velocity=data.subarray(1),eo=expected.subarray(0,8),ev=expected.subarray(1);
  const uploads=[],referenceUploads=[];
  const actor=module.make(out,velocity,snapshot=>uploads.push(snapshot));
  const oracle=reference.make(eo,ev,snapshot=>referenceUploads.push(snapshot));
  for(let frame=0;frame<frames;frame++){
    actor.tag=oracle.tag=`frame-${frame}`;
    const dt=1/(frame+30),count=frame%3+1;
    assert.ok(Object.is(actor.step(dt,count),oracle.step(dt,count)));same(data,expected);
  }
  assert.deepEqual(uploads,referenceUploads);
  return [...data];
}

test('linked dynamic application callbacks execute helper graphs in one real native transaction per frame',async t=>{
  const files=fixture(t),result=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  const report=result.numericSpecialization;
  assert.equal(report.accelerated,false);
  assert.equal(report.compiledLoopIslands,1);
  assert.ok(report.units.some(unit=>unit.absorbedCalls===3));
  assert.equal(report.rewrittenCalls,0);
  const region=report.units.flatMap(unit=>unit.loopIslands?.candidates??[]).find(item=>item.route==='guarded-loop-wasm');
  assert.equal(region.loopCount,4);
  assert.equal(region.maxLoopDepth,2);
  assert.equal(region.helpers.length,3);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(files.out,report.reportFile),'utf8')),report);
  for(const asset of report.runtimeAssets){
    const bytes=fs.readFileSync(path.join(files.out,asset.fileName));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),asset.sha256);
  }
  const counts=observeWasm(t);
  const entry=await import(pathToFileURL(path.join(result.outDir,result.entryFiles[0]))),module=await entry.load();
  const reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));
  assert.equal(counts.instances,0);
  verifyFrames(module,reference,Float32Array);verifyFrames(module,reference,Float64Array);
  assert.deepEqual(counts,{instances:2,attempts:60,calls:60});
});

test('a packaged nested-helper budget aborts all speculative passes without duplicating uploads',async t=>{
  const files=fixture(t),result=await buildApplication(files.entry,files.out,{specializeNumeric:{maxIterations:25}});
  // Each iteration spends 1 root + 8 integrate + 8 damp + 8 energy = 25.
  const counts=observeWasm(t),entry=await import(pathToFileURL(path.join(files.out,result.entryFiles[0])));
  const module=await entry.load(),reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));
  verifyFrames(module,reference,Float32Array,9);
  assert.deepEqual(counts,{instances:1,attempts:9,calls:3});
});

test('CLI builds relocate independently and preserve native execution or original JS without Wasm',async t=>{
  const files=fixture(t),manifest=path.join(files.root,'manifest.json');
  const build=spawnSync(process.execPath,[fileURLToPath(new URL('./cli.mjs',import.meta.url)),
    '--entry',files.entry,'--build-app',files.out,'--specialize-numeric','--output',manifest],{encoding:'utf8'});
  assert.equal(build.status,0,build.stderr);
  const result=JSON.parse(fs.readFileSync(manifest,'utf8'));
  assert.equal(result.numericSpecialization.compiledLoopIslands,1);
  const moved=path.join(files.root,'relocated');fs.renameSync(files.out,moved);
  const entry=pathToFileURL(path.join(moved,result.entryFiles[0])).href;
  const reference=pathToFileURL(path.join(files.src,'frame.mjs')).href;
  for(const wasm of [true,false]){
    const run=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';
      let calls=0,instances=0;
      if(${wasm}){
        const Native=WebAssembly.Instance;
        WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);instances++;
          return {exports:{...instance.exports,run(...parameters){const result=instance.exports.run(...parameters);calls++;return result;}}};};
      } else globalThis.WebAssembly=undefined;
      const module=await (await import(${JSON.stringify(entry)})).load();
      const reference=await import(${JSON.stringify(reference)});
      ${same.toString().replace(/^\(a,b\)=>/,'function same(a,b)')}
      ${verifyFrames.toString()}
      const values=verifyFrames(module,reference,Float32Array);
      assert.equal(calls,${wasm?30:0});assert.equal(instances,${wasm?1:0});
      console.log(JSON.stringify({calls,instances,values}));
    `],{encoding:'utf8'});
    assert.equal(run.status,0,run.stderr);
    const execution=JSON.parse(run.stdout);assert.equal(execution.calls,wasm?30:0);
  }
});

test('application opt-out emits original callbacks, while explicit loop opt-in works in local-call mode',async t=>{
  const files=fixture(t),ordinary=await buildApplication(files.entry,files.out);
  assert.equal(ordinary.numericSpecialization,undefined);
  const options={crossModule:false,loopIslands:true};
  const result=await bundleWithRollup(files.entry,{specializeNumeric:options});
  assert.equal(result.numericSpecialization.compiledLoopIslands,1);
  assert.equal(result.numericSpecialization.rewrittenCalls,0);
  const out=path.join(files.root,'local');fs.mkdirSync(out);
  for(const [name,bytes] of Object.entries(result.files)){
    const target=path.join(out,name);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,bytes);
  }
  const counts=observeWasm(t),module=await (await import(pathToFileURL(path.join(out,result.entryFiles[0])))).load();
  const reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));
  verifyFrames(module,reference,Float32Array,3);assert.equal(counts.calls,3);
});
