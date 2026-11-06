/** Production application builds with live captured buffers and real Wasm. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {buildApplication} from './build_application.mjs';

const PARTICLES = {
  'parameters.mjs': `
    export let positions=new Float32Array(0),velocities=new Float32Array(0),damping=1;
    export function configure(nextPositions,nextVelocities,nextDamping){
      positions=nextPositions;velocities=nextVelocities;damping=nextDamping;
    }
  `,
  'entry.mjs': `
    export {configure,positions,velocities} from './parameters.mjs';
    export const load=()=>import('./particle_update.mjs');
  `,
  'particle_update.mjs': `
    import {positions,velocities,damping} from './parameters.mjs';
    function displacement(i,dt){velocities[i]*=damping;return velocities[i]*dt;}
    function integrate(i,dt){positions[i]+=displacement(i,dt);return positions[i];}
    function update(out,dt){for(let i=0;i<out.length;i++)out[i]=integrate(i,dt);}
    export function frame(out,dt,upload){upload('before');update(out,dt);upload('after');}
  `,
};

const asURL = (directory, file) => pathToFileURL(path.join(directory, file)).href;
function fixture(modules) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-array-capture-build-'));
  const src=path.join(root,'src'),out=path.join(root,'dist');
  fs.mkdirSync(src);
  fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  for(const [file,source] of Object.entries(modules)) fs.writeFileSync(path.join(src,file),source);
  return {root,src,out,entry:path.join(src,'entry.mjs')};
}

function observeWasm(t) {
  const Native=WebAssembly.Instance,counts={instances:0,calls:0};
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){
    const instance=Reflect.construct(Native,args);counts.instances++;
    return {exports:{...instance.exports,run(...values){
      const result=instance.exports.run(...values);counts.calls++;return result;
    }}};
  };
  return counts;
}

// All three writable views may share one backing store. The explicit output
// overlaps a hidden capture, and integrate reads its destination before a
// transitive helper mutates a different overlapping captured view.
function verifyParticles(entry,app,referenceEntry,reference) {
  let frames=0;
  for(const Type of [Float32Array,Float64Array]) for(const overlap of [false,true]) {
    const positions=new Type([1/3,2,3,4,5,6,7,8,9,10]),expected=positions.slice();
    const velocity=overlap ? positions.subarray(2) : new Type([2,3,5,7,11,13,17,19]);
    const referenceVelocity=overlap ? expected.subarray(2) : velocity.slice();
    const output=positions.subarray(1,9),referenceOutput=expected.subarray(1,9);
    entry.configure(positions,velocity,0.875);
    referenceEntry.configure(expected,referenceVelocity,0.875);
    assert.equal(entry.positions,positions);assert.equal(entry.velocities,velocity);
    const uploads=[],referenceUploads=[];
    for(let i=0;i<3;i++) {
      const dt=1/(30+i);
      app.frame(output,dt,phase=>uploads.push([phase,[...positions],[...velocity]]));
      reference.frame(referenceOutput,dt,phase=>referenceUploads.push([phase,[...expected],[...referenceVelocity]]));
      assert.deepEqual(positions,expected);assert.deepEqual(velocity,referenceVelocity);
      frames++;
    }
    assert.deepEqual(uploads,referenceUploads);
  }
  return frames;
}

test('a relocated lazy particle application reads live imported captures and preserves overlapping views',async t=>{
  const files=fixture(PARTICLES);
  const built=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  const native=built.numericSpecialization.units.flatMap(unit=>unit.candidates??[])
    .find(item=>item.functionName==='update' && item.route==='guarded-numeric-wasm');
  assert.ok(native,'the particle update must have a native whole-function route');
  assert.ok(native.helperCaptureKinds.includes('array'));
  const relocated=path.join(files.root,'relocated');fs.renameSync(files.out,relocated);
  const counts=observeWasm(t);
  const entry=await import(asURL(relocated,built.entryFiles[0])),app=await entry.load();
  const referenceEntry=await import(asURL(files.src,'entry.mjs')),reference=await referenceEntry.load();
  assert.deepEqual(counts,{instances:0,calls:0},'lazy loading alone must not instantiate Wasm');
  const frames=verifyParticles(entry,app,referenceEntry,reference);
  assert.equal(counts.calls,frames,'one native transaction owns all transitive helper work per frame');
  assert.equal(counts.instances,2,'the same Float32 and Float64 kernels survive capture rebinding');
});

test('mesh color callbacks publish scalar state with captured Float32, Uint32 and clamped buffers',async t=>{
  const files=fixture({
    'entry.mjs':`export const load=()=>import('./mesh_colors.mjs');`,
    'mesh_colors.mjs':`
      export let colors=new Uint8ClampedArray(6);
      let values=new Float32Array([-5,1.5,2.5,255.7,300,1/3]);
      let indices=new Uint32Array([4,2,0,5,3,1]);
      export function replace(nextColors,nextValues,nextIndices){
        colors=nextColors;values=nextValues;indices=nextIndices;
      }
      function colorAt(i,scale){colors[i]=values[indices[i]]*scale;return colors[i];}
      function shade(i,scale){return colorAt(i,scale);}
      export function make(upload){let energy=0;return function frame(scale){
        upload('before',energy);
        for(var i=0;i<colors.length;i++)energy+=shade(i,scale);
        upload('after',{energy,index:i,colors:[...colors]});return energy;
      }}
    `,
  });
  const built=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  const islands=built.numericSpecialization.units.flatMap(unit=>unit.loopIslands?.candidates??[])
    .filter(item=>item.route==='guarded-loop-wasm');
  assert.equal(islands.length,1);
  assert.ok(islands[0].helperCaptureKinds.includes('array'));
  const counts=observeWasm(t);
  const app=await (await import(asURL(files.out,built.entryFiles[0]))).load();
  const reference=await (await import(asURL(files.src,'entry.mjs'))).load();
  const events=[],expectedEvents=[];
  const frame=app.make((...args)=>events.push(args));
  const original=reference.make((...args)=>expectedEvents.push(args));
  for(const scale of [1,0.5,2,0.25]) {
    assert.equal(frame(scale),original(scale));assert.deepEqual(app.colors,reference.colors);
  }
  const colors=new Uint8ClampedArray(4),values=new Float32Array([0.5,1.5,2.5,3.5]);
  const indices=new Uint32Array([3,2,1,0]),expected=colors.slice();
  app.replace(colors,values,indices);reference.replace(expected,values.slice(),indices.slice());
  for(const scale of [1,3]) {
    assert.equal(frame(scale),original(scale));assert.deepEqual(colors,expected);
  }
  assert.equal(app.colors,colors);assert.deepEqual(events,expectedEvents);
  assert.deepEqual(counts,{instances:1,calls:6},'integer allocation hints must select a native captured-buffer ABI');
});

test('an independent moved package preserves source execution with and without Wasm',async()=>{
  const files=fixture(PARTICLES);
  const built=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  const relocated=path.join(files.root,'relocated'),oracle=path.join(files.root,'oracle');
  fs.renameSync(files.out,relocated);fs.renameSync(files.src,oracle);
  for(const wasm of [true,false]) {
    const execution=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';
      let nativeCalls=0;
      if(${wasm}) {
        const Native=WebAssembly.Instance;
        WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);
          return {exports:{...instance.exports,run(...values){
            const result=instance.exports.run(...values);nativeCalls++;return result;
          }}};
        };
      } else globalThis.WebAssembly=undefined;
      const entry=await import(${JSON.stringify(asURL(relocated,built.entryFiles[0]))}),app=await entry.load();
      const referenceEntry=await import(${JSON.stringify(asURL(oracle,'entry.mjs'))}),reference=await referenceEntry.load();
      ${verifyParticles.toString()}
      const frames=verifyParticles(entry,app,referenceEntry,reference);
      assert.equal(nativeCalls,${wasm?'frames':'0'});
      // An inadmissible captured view keeps the original getters, coercions,
      // stores and surrounding upload callbacks, including their multiplicity.
      const events=[],expectedEvents=[];
      function view(log){let value=3;return {
        get 0(){log.push('read');return value;},
        set 0(next){log.push(['write',next]);value=next;}
      };}
      const output=new Float32Array(1),expected=output.slice();
      entry.configure(view(events),new Float32Array([2]),0.5);
      referenceEntry.configure(view(expectedEvents),new Float32Array([2]),0.5);
      app.frame(output,0.25,phase=>events.push(phase));
      reference.frame(expected,0.25,phase=>expectedEvents.push(phase));
      assert.deepEqual(output,expected);assert.deepEqual(events,expectedEvents);
      assert.equal(nativeCalls,${wasm?'frames':'0'});
    `],{encoding:'utf8',timeout:20000});
    assert.equal(execution.status,0,`${wasm?'native':'no-Wasm'} execution failed:\n${execution.stderr}`);
  }
});
