import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {baseSource,addonSource} from '../../tests/fixtures/marching_cubes/adapter_fixture.mjs';
import {reference,clone,testTables} from '../../tests/fixtures/marching_cubes/reference.mjs';
import {specializeMarchingCubesModule,specializeMarchingCubesBase,sourceBlob} from './marching_cubes_specialization.mjs';
import {marchingCubesDiagnostics} from './marching_cubes_adapter.mjs';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
const runtimeModule=new URL('./marching_cubes_adapter.mjs',import.meta.url).href;
const referenceUrl=new URL('../../tests/fixtures/marching_cubes/reference.mjs',import.meta.url).href;
async function fixture(t) {
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-marching-'));
  t.after(()=>fs.rm(dir,{recursive:true,force:true}));
  const baseUrl=pathToFileURL(path.join(dir,'base.mjs')).href;
  const base=specializeMarchingCubesBase(baseSource,{runtimeModule,expectedBlob:sourceBlob(baseSource)});
  await fs.writeFile(path.join(dir,'base.mjs'),base.code);
  const source=addonSource(referenceUrl,baseUrl);
  const built=specializeMarchingCubesModule(source,{runtimeModule,expectedBlob:sourceBlob(source)});
  await fs.writeFile(path.join(dir,'addon.mjs'),built.code);
  const module=await import(pathToFileURL(path.join(dir,'addon.mjs')).href);
  const object=new module.MarchingCubes(5,new module.Material(),true,true);
  for(let k=0;k<object.field.length;k++)object.field[k]=(k%7)-3;
  return {object,module,source,built};
}
function parameters(o) {
  const [edgePositions,edgeNormals,edgeColors]=o.testEdgeLists();
  return {edgePositions,edgeNormals,edgeColors,size:o.size,isolation:o.isolation,field:o.field,normalCache:o.normal_cache,palette:o.palette,
    positions:o.positionArray,normals:o.normalArray,uvs:o.enableUvs ? o.uvArray : null,
    colors:o.enableColors ? o.colorArray : null,flatShading:o.material.flatShading===true};
}
function compare(o,tables) {
  const expected=clone(parameters(o)),count=reference(expected,tables);
  const method=o.update;assert.equal(o.update(),undefined);assert.equal(o.update,method);assert.equal(o.count,count);
  for(const [name,key] of [['normalCache','normal_cache'],['positions','positionArray'],['normals','normalArray'],['uvs','uvArray'],['colors','colorArray']]) {
    if(expected[name])assert.deepEqual(o[key],expected[name],name);
  }
  assert.deepEqual(o.geometry.drawRange,{start:0,count});
}

test('real transformed ESM runs Wasm and retains constructor/method identities and original publication tail',async t=>{
  const {object:o,module:m,source,built}=await fixture(t);
  assert.equal(built.report.accelerationClaim,false);
  assert.equal(Object.getPrototypeOf(o),m.MarchingCubes.prototype);
  assert.equal(o.update.name,'');assert.equal(o.update.length,0);
  const tail=source.slice(built.report.preservedPublicationSpan.start,built.report.preservedPublicationSpan.end);
  assert.ok(built.code.includes(tail));
  compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,1);
  assert.equal(marchingCubesDiagnostics(o).fallbackCalls,0);
  for(const attribute of Object.values(o.geometry.attributes))assert.equal(attribute.version,1);
  assert.ok(Object.isFrozen(marchingCubesDiagnostics(o)));
});

test('re-init, material switching and field/cache changes use fresh live storage without replacing update',async t=>{
  const {object:o,module:m}=await fixture(t),method=o.update;
  compare(o,m);const old=o.positionArray;
  o.init(8);assert.notEqual(o.positionArray,old);o.material=new m.Material(true);
  for(let k=0;k<o.field.length;k++)o.field[k]=(k%11)-6;
  compare(o,m);o.field[73]+=10;o.palette.fill(0.7);compare(o,m);
  o.init(4);o.field.fill(1);o.field[21]=-1;compare(o,m);
  assert.equal(o.update,method);assert.equal(marchingCubesDiagnostics(o).wasmCalls,4);
});

test('replaced public geometry changes draw target but notifications retain the original construction geometry',async t=>{
  const {object:o,module:m}=await fixture(t),original=o.geometry;
  o.geometry=new m.Geometry();compare(o,m);
  assert.equal(original.getAttribute('position').version,1);
  assert.equal(original.getAttribute('normal').version,1);
  assert.deepEqual(Object.keys(o.geometry.attributes),[]);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,1);
});

test('source callbacks see complete buffers and their exceptions never trigger retained replay',async t=>{
  const {object:o}=await fixture(t),sentinel={};let callbacks=0;
  o.geometry.onRange=(start,count)=>{callbacks++;assert.equal(o.count,count);assert.ok(count>0);assert.ok(o.positionArray.some(x=>x!==0));throw sentinel;};
  assert.throws(()=>o.update(),error=>error===sentinel);
  assert.equal(callbacks,1);assert.equal(o.geometry.getAttribute('position').version,0);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,1);assert.equal(marchingCubesDiagnostics(o).fallbackCalls,0);
});

test('post-computation changes to UV/color flags and attribute setters execute in source order',async t=>{
  const {object:o}=await fixture(t),trace=[];
  o.geometry.onRange=()=>{trace.push('range');o.enableUvs=false;Object.defineProperty(o,'enableColors',{get(){trace.push('colors?');return true;},configurable:true});};
  for(const name of ['position','normal','uv','color'])Object.defineProperty(o.geometry.attributes[name],'needsUpdate',{set(){trace.push(name);}});
  o.update();assert.deepEqual(trace,['range','position','normal','colors?','color']);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,1);
});

test('nested source callbacks reuse the kernel only after the outer numeric transaction completes',async t=>{
  const {object:o}=await fixture(t);let nested=false,ranges=0;
  o.geometry.onRange=()=>{ranges++;if(!nested){nested=true;o.field[31]+=1;o.update();}};
  o.update();assert.equal(ranges,2);
  assert.equal(o.geometry.getAttribute('position').version,2);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,2);
});

test('unknown material Proxy and borrowed Proxy receivers receive no speculative descriptor traps',async t=>{
  const {object:o}=await fixture(t);let descriptors=0,reads=0;
  const real=o.material;
  o.material=new Proxy(real,{get(target,key){if(key==='flatShading')reads++;return Reflect.get(target,key);},
    getOwnPropertyDescriptor(){descriptors++;throw Error('unexpected descriptor');}});
  o.update();assert.equal(reads,1);assert.equal(descriptors,0);assert.equal(marchingCubesDiagnostics(o).wasmCalls,0);
  o.material=real;
  const proxy=new Proxy(o,{set(target,key,value){target[key]=value;return true;},
    getOwnPropertyDescriptor(){descriptors++;throw Error('unexpected descriptor');}});
  o.update.call(proxy);assert.equal(descriptors,0);
  assert.equal(marchingCubesDiagnostics(o).fallbackCalls,2);
});

test('live public table changes remain native and preserve source triangle order',async t=>{
  const {object:o,module:m}=await fixture(t);compare(o,m);
  const old=m.triTable[16];m.triTable[16]=m.triTable[17];compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).lastFailure,null);
  m.triTable[16]=old;compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,3);assert.equal(marchingCubesDiagnostics(o).fallbackCalls,0);
});

test('late capacity failure runs the retained numeric body once and warns through the untouched tail',async t=>{
  const {module:m}=await fixture(t),o=new m.MarchingCubes(5,new m.Material(),true,true,0);
  o.field.fill(1);o.field[31]=-1;let warnings=0;const warn=console.warn;
  try {console.warn=()=>warnings++;o.update();} finally {console.warn=warn;}
  assert.ok(o.count>0);assert.equal(warnings,1);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,0);assert.equal(marchingCubesDiagnostics(o).fallbackCalls,1);
  assert.equal(marchingCubesDiagnostics(o).lastFailure,'KERNEL_EXECUTION_FAILED');
});

test('unavailable Wasm leaves the original method fully executable without repeated initialization',async t=>{
  const {object:o,module:m}=await fixture(t),saved=globalThis.WebAssembly;
  try {globalThis.WebAssembly=undefined;compare(o,m);compare(o,m);} finally {globalThis.WebAssembly=saved;}
  assert.equal(marchingCubesDiagnostics(o).fallbackCalls,2);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,0);
});

test('NaN computation preserves the source numeric result',async t=>{
  const {object:o}=await fixture(t);o.field[31]=Infinity;o.normal_cache.fill(NaN);
  const expected=clone(parameters(o));const count=reference(expected,testTables());o.update();
  assert.equal(o.count,count);
  for(let k=0;k<o.normalArray.length;k++)assert.ok(Object.is(o.normalArray[k],expected.normals[k]));
});

test('source pins reject modified/lookalike code, and ordinary unselected modules stay byte-identical',()=>{
  const source=addonSource(referenceUrl,'./base.mjs');
  assert.equal(specializeMarchingCubesModule(source).changed,false);
  assert.equal(specializeMarchingCubesModule(source).code,source);
  assert.equal(specializeMarchingCubesBase(baseSource).changed,false);
  const plugin=marchingCubesRollupPlugin();plugin.buildStart();
  assert.equal(plugin.transform(source,'/fake/MarchingCubes.js'),null);
  assert.equal(plugin.transform('export const x=1','/app.mjs'),null);
  assert.equal(plugin.api.getReport().compiledAddons,0);
  assert.equal(plugin.api.getReport().modules[0].reason,'SOURCE_PIN_MISMATCH');
  assert.equal(plugin.resolveId('\0f3d-marching-cubes-adapter'), '\0f3d-marching-cubes-adapter');
  assert.equal(plugin.resolveId('./numeric_kernel_runtime.mjs','\0f3d-marching-cubes-adapter'),'\0f3d-marching-cubes-runtime');
  assert.ok(plugin.load('\0f3d-marching-cubes-adapter').includes('tryMarchingCubesUpdate'));
  assert.ok(plugin.load('\0f3d-marching-cubes-runtime').includes('instantiateNumericKernel'));
  assert.equal(plugin.resolveId('./user.mjs'),null);assert.equal(plugin.load('/app.mjs'),null);
});

test('registered material accessors are not speculatively invoked or hoisted',async t=>{
  const {object:o}=await fixture(t);let reads=0;
  Object.defineProperty(o.material,'flatShading',{get(){reads++;return true;},configurable:true});
  o.update();assert.equal(reads,1);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,0);
  assert.equal(marchingCubesDiagnostics(o).fallbackCalls,1);
});

test('EventDispatcher registration preserves base name/length and does not trust foreign returned proxies',async t=>{
  const {object:o,module:m}=await fixture(t);
  // A source-constructed object's Proxy wrapper is never its recorded identity.
  let gets=0;
  const material=new Proxy(o.material,{get(target,key){gets++;return Reflect.get(target,key);}});
  o.material=material;o.update();assert.equal(gets,1);
  o.material=new m.Material();o.update();assert.equal(marchingCubesDiagnostics(o).wasmCalls,1);
  const base=Object.getPrototypeOf(m.Material);
  assert.equal(base.name,'EventDispatcher');assert.equal(base.length,0);
  assert.equal(Object.getPrototypeOf(new base()),base.prototype);
});

test('changed WebAssembly bindings retain source without invoking injected platform getters or constructors',async t=>{
  const {object:o,module:m}=await fixture(t),saved=Object.getOwnPropertyDescriptor(globalThis,'WebAssembly');let gets=0;
  try {
    Object.defineProperty(globalThis,'WebAssembly',{configurable:true,get(){gets++;throw Error('unexpected platform getter');}});
    compare(o,m);
  } finally {Object.defineProperty(globalThis,'WebAssembly',saved);}
  assert.equal(gets,0);assert.equal(marchingCubesDiagnostics(o).fallbackCalls,1);
});


test('native updates preserve instance-private edge history across shared kernels and table edits',async t=>{
  const {object:o,module:m}=await fixture(t);
  o.init(4);o.field.fill(1);o.field[26]=-1; // Case 4 writes edge 1.
  o.palette.fill(0.75);compare(o,m);
  const history=o.testEdgeLists().map(a=>a.slice());
  assert.notEqual(history[0][4],0);
  // A second instance shares the module but not any retained private lists.
  const other=new m.MarchingCubes(4,new m.Material(),true,true);
  other.field.fill(2);other.field[26]=-3;other.palette.fill(0.25);compare(other,m);
  for(let k=0;k<3;k++)assert.deepEqual(o.testEdgeLists()[k],history[k]);
  o.field.fill(1);o.field[21]=-1; // Case 1 does not write edge 1.
  const prior=m.triTable[16];m.triTable[16]=1; // A public edit reads its old contents.
  compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).lastFailure,null);
  assert.deepEqual(o.positionArray.slice(0,3),history[0].slice(3,6));
  m.triTable[16]=prior;compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,3);
  assert.equal(marchingCubesDiagnostics(o).fallbackCalls,0);
});

test('aliased outputs retain the complete source operation and recover when disjoint',async t=>{
  const {object:o,module:m}=await fixture(t),original=o.normalArray;
  o.normalArray=o.positionArray;
  o.update();
  assert.equal(marchingCubesDiagnostics(o).lastFailure,'KERNEL_ARRAY_ALIAS');
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,0);
  o.normalArray=original;compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,1);
});

test('accessor count is not speculatively written or read before retained execution',async t=>{
  const {object:o}=await fixture(t);let count=99,writes=0,reads=0;
  Object.defineProperty(o,'count',{get(){reads++;return count;},set(v){writes++;count=v;},configurable:true});
  o.update();assert.equal(writes,2);assert.ok(reads>0);
  assert.equal(marchingCubesDiagnostics(o).lastFailure,'MARCHING_CUBES_OBJECT_GUARD');
});

test('mutated memory-growth hooks are never invoked by a speculative update',async t=>{
  const {object:o,module:m}=await fixture(t);compare(o,m);
  const original=Object.getOwnPropertyDescriptor(WebAssembly.Memory.prototype,'grow');let calls=0;
  try {
    Object.defineProperty(WebAssembly.Memory.prototype,'grow',{configurable:true,get(){calls++;throw Error('hook');}});
    compare(o,m);
  } finally {Object.defineProperty(WebAssembly.Memory.prototype,'grow',original);}
  assert.equal(calls,0);assert.equal(marchingCubesDiagnostics(o).lastFailure,'MARCHING_CUBES_PLATFORM');
  compare(o,m);assert.equal(marchingCubesDiagnostics(o).wasmCalls,2);
});

test('fuel budgets are forwarded to the shared compiler and preserve whole-source fallback',async t=>{
  const dir=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-marching-fuel-'));
  t.after(()=>fs.rm(dir,{recursive:true,force:true}));
  const baseUrl=pathToFileURL(path.join(dir,'base.mjs')).href;
  await fs.writeFile(path.join(dir,'base.mjs'),specializeMarchingCubesBase(baseSource,{runtimeModule,expectedBlob:sourceBlob(baseSource)}).code);
  const source=addonSource(referenceUrl,baseUrl);
  const built=specializeMarchingCubesModule(source,{runtimeModule,expectedBlob:sourceBlob(source),maxIterations:1});
  await fs.writeFile(path.join(dir,'addon.mjs'),built.code);
  const m=await import(pathToFileURL(path.join(dir,'addon.mjs')).href),o=new m.MarchingCubes(5,new m.Material(),true,true);
  o.field.fill(1);o.field[31]=-1;compare(o,m);
  assert.equal(marchingCubesDiagnostics(o).wasmCalls,0);
  assert.equal(marchingCubesDiagnostics(o).lastFailure,'KERNEL_EXECUTION_FAILED');
});

test('plugin validates budgets even with no candidates and resets reports between builds',()=>{
  for(const options of [{maxMemoryPages:0},{maxIterations:0},{maxIterations:1.5},{unexpected:true},null]) {
    assert.throws(()=>marchingCubesRollupPlugin(options));
  }
  const plugin=marchingCubesRollupPlugin({maxIterations:100});plugin.buildStart();
  plugin.transform('class MarchingCubes {}','/unverified.mjs');
  assert.equal(plugin.api.getReport().modules.length,1);plugin.buildStart();
  assert.equal(plugin.api.getReport().modules.length,0);assert.equal(plugin.api.getReport().maxIterations,100);
});
