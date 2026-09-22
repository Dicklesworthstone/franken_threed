/** End-to-end build tests against unmodified pinned Three.js, not host doubles. */
import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {bundleWithRollup} from './bundler.mjs';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';
import {cubeField,sameArrays,readMarchingCubesOracle} from './fixtures/marching_cubes_oracle.mjs';

const oracleRoot=process.env.F3D_THREE_ROOT ?? fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const packageRootUrl=pathToFileURL(path.resolve(oracleRoot)+path.sep).href;
const ordinaryEntry=`export {MarchingCubes,edgeTable,triTable} from 'three/addons/objects/MarchingCubes.js';
export {MeshPhongMaterial,MeshStandardMaterial,MeshNormalMaterial,MeshBasicMaterial,BufferGeometry,Color} from 'three';\n`;
// Test-only import of the same private virtual runtime used by the source pass.
// No upstream method, table, constructor or public field is replaced by this.
const diagnosticEntry=ordinaryEntry+`export {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};\n`;
let root,actualModule,expectedModule,actualBuild;
async function write(name,content) {
  const file=path.join(root,name);await fs.mkdir(path.dirname(file),{recursive:true});await fs.writeFile(file,content);return file;
}
async function executeBuild(build) {
  // Build output .js files are browser ES modules; declare that in the Node harness.
  await fs.writeFile(path.join(build.outDir,'package.json'),'{"type":"module"}',{flag:'wx'});
  return import(pathToFileURL(path.join(build.outDir,build.entryFiles[0])).href);
}
function compare(actual,expected) {
  assert.equal(actual.count,expected.count);
  assert.deepEqual(actual.geometry.drawRange,expected.geometry.drawRange);
  for(const name of ['field','normal_cache','palette','positionArray','normalArray','uvArray','colorArray']) {
    if(actual[name] || expected[name])sameArrays(actual[name],expected[name],name);
  }
  for(const name of ['position','normal','uv','color']) {
    const a=actual.geometry.getAttribute(name),b=expected.geometry.getAttribute(name);
    assert.equal(a?.version,b?.version,`${name} upload version`);
    assert.equal(a?.itemSize,b?.itemSize,`${name} item size`);
  }
}
function pair(size=8,flags=7,capacity=2000,material='MeshPhongMaterial') {
  return [actualModule,expectedModule].map(m=>new m.MarchingCubes(size,
    new m[material]({flatShading:!!(flags&1)}),!!(flags&2),!!(flags&4),capacity));
}
function update(a,b) {assert.equal(a.update(),undefined);assert.equal(b.update(),undefined);compare(a,b);}

before(async()=>{
  readMarchingCubesOracle(); // Fails rather than silently swapping/omitting the oracle.
  const core=await fs.readFile(path.join(oracleRoot,'build/three.core.js'));
  assert.equal(createHash('sha256').update(core).digest('hex'),
    '9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6');
  root=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-marching-build-'));
  const actualEntry=await write('native-entry.mjs',diagnosticEntry);
  const expectedEntry=await write('retained-entry.mjs',ordinaryEntry);
  actualBuild=await buildApplication(actualEntry,path.join(root,'native'),{packageRootUrl,specializeNumeric:true});
  actualModule=await executeBuild(actualBuild);
  const expectedBuild=await buildApplication(expectedEntry,path.join(root,'retained'),{packageRootUrl});
  assert.equal(expectedBuild.numericSpecialization,undefined);
  expectedModule=await executeBuild(expectedBuild);
});
after(async()=>{if(root)await fs.rm(root,{recursive:true,force:true});});

test('existing application opt-in emits a real pinned addon and reports it without inflating generic counts',async()=>{
  const report=actualBuild.numericSpecialization;
  const library=report.libraryKernels.marchingCubes;
  assert.equal(library.compiledAddons,1);assert.equal(library.registeredBaseModules,1);
  assert.equal(library.compiledFieldKernels,6);assert.equal(library.registeredColorModules,1);
  assert.deepEqual(library.modules.find(m=>m.compiledFieldKernels===6).fieldKernels.map(k=>k.name),
    ['addBall','addPlaneX','addPlaneY','addPlaneZ','blur','reset']);
  assert.equal(library.accelerationClaim,false);assert.equal(report.accelerated,false);
  assert.equal(library.maxMemoryPages,2048);assert.equal(library.maxIterations,100000000);
  assert.equal(report.compiledKernels,report.units.reduce((sum,unit)=>sum+unit.compiledKernels,0));
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(actualBuild.outDir,report.reportFile),'utf8')),report);
  assert.ok(actualBuild.emittedFiles.includes(report.reportFile));
  assert.equal(actualModule.MarchingCubes.name,'MarchingCubes');
  assert.equal(actualModule.MarchingCubes.length,2);
  const [a,b]=pair();for(const effect of [a,b])effect.addBall(0.5,0.5,0.5,1.2,8);
  const method=a.update,position=a.positionArray;
  update(a,b);assert.ok(a.count>0);
  assert.equal(a.update,method);assert.equal(a.positionArray,position);
  assert.equal(Object.getPrototypeOf(a),actualModule.MarchingCubes.prototype);
  assert.equal(actualModule.diagnostics(a).wasmCalls,1);assert.equal(actualModule.diagnostics(a).fallbackCalls,0);
});

for(let flags=0;flags<8;flags++)test(`full pinned addon: all 256 cases under output flags ${flags}`,()=>{
  const [a,b]=pair(4,flags,7);
  for(let mask=0;mask<256;mask++) {
    for(const effect of [a,b]) {effect.reset();cubeField(effect,mask);}
    update(a,b);
  }
  assert.equal(actualModule.diagnostics(a).wasmCalls,256);
  assert.equal(actualModule.diagnostics(a).fallbackCalls,0);
});

test('full objects retain animated fields, material changes, stale normals and reinitialization',()=>{
  const [a,b]=pair(),method=a.update;
  for(let frame=0;frame<7;frame++) {
    for(const [effect,m] of [[a,actualModule],[b,expectedModule]]) {
      if(frame!==2)effect.reset();
      effect.addBall(0.4+frame/50,0.53,0.47,1.1,12,[0.2,0.4,0.9]);
      effect.addPlaneX(0.1,7);effect.addPlaneY(0.12,8);effect.addPlaneZ(0.08,6);
      if(frame===3)effect.blur(0.4);
      if(frame===2)effect.setCell(2,2,2,90);
      effect.material=new m[['MeshPhongMaterial','MeshStandardMaterial','MeshNormalMaterial'][frame%3]]({flatShading:frame%2===0});
      effect.enableUvs=frame!==1;effect.enableColors=frame!==4;
    }
    update(a,b);
  }
  for(const effect of [a,b]) {effect.enableUvs=true;effect.enableColors=true;effect.init(5);cubeField(effect,69);}
  update(a,b);assert.equal(a.update,method);
  assert.equal(actualModule.diagnostics(a).wasmCalls,8);
});

test('actual source callbacks run once after complete publication; thrown errors never replay the numeric body',()=>{
  const [a]=pair(4);cubeField(a,1);
  const sentinel={},geometry=a.geometry;let calls=0;
  geometry.setDrawRange=function(start,count){calls++;assert.equal(count,a.count);assert.ok(count>0);throw sentinel;};
  assert.throws(()=>a.update(),value=>value===sentinel);assert.equal(calls,1);
  assert.equal(geometry.getAttribute('position').version,0);
  assert.equal(actualModule.diagnostics(a).wasmCalls,1);assert.equal(actualModule.diagnostics(a).fallbackCalls,0);
});

test('real source falls back on aliasing or material accessors then recovers native execution',()=>{
  const [a,b]=pair(4);for(const effect of [a,b])cubeField(effect,1);
  const aNormal=a.normalArray,bNormal=b.normalArray;
  a.normalArray=a.positionArray;b.normalArray=b.positionArray;update(a,b);
  assert.equal(actualModule.diagnostics(a).lastFailure,'KERNEL_ARRAY_ALIAS');
  a.normalArray=aNormal;b.normalArray=bNormal;
  let aReads=0,bReads=0;
  Object.defineProperty(a.material,'flatShading',{get(){aReads++;return true;},configurable:true});
  Object.defineProperty(b.material,'flatShading',{get(){bReads++;return true;},configurable:true});
  update(a,b);assert.equal(aReads,bReads);assert.ok(aReads>0);
  a.material=new actualModule.MeshPhongMaterial();b.material=new expectedModule.MeshPhongMaterial();
  update(a,b);assert.equal(actualModule.diagnostics(a).wasmCalls,1);assert.equal(actualModule.diagnostics(a).fallbackCalls,2);
});

test('capacity fallback preserves the exact upstream warning/count and recovers on a smaller surface',()=>{
  const [a,b]=pair(4,7,1);for(const effect of [a,b])cubeField(effect,61);
  const warn=console.warn,warnings=[];
  try {console.warn=message=>warnings.push(message);update(a,b);} finally {console.warn=warn;}
  assert.equal(warnings.length,2);assert.equal(warnings[0],warnings[1]);assert.ok(a.count>3);
  assert.equal(actualModule.diagnostics(a).lastFailure,'KERNEL_EXECUTION_FAILED');
  for(const effect of [a,b]) {effect.reset();cubeField(effect,1);}
  update(a,b);assert.equal(actualModule.diagnostics(a).wasmCalls,1);
});

test('caller memory/fuel budgets propagate through the existing build API to whole-source fallback',async()=>{
  for(const settings of [{maxMemoryPages:1},{maxIterations:1}]) {
    const suffix=Object.keys(settings)[0];
    const built=await buildApplication(path.join(root,'native-entry.mjs'),path.join(root,suffix),{packageRootUrl,specializeNumeric:settings});
    const m=await executeBuild(built);
    assert.equal(built.numericSpecialization.libraryKernels.marchingCubes[suffix],1);
    const a=new m.MarchingCubes(18,new m.MeshPhongMaterial(),true,true,2000);
    const b=new expectedModule.MarchingCubes(18,new expectedModule.MeshPhongMaterial(),true,true,2000);
    for(const effect of [a,b])effect.addBall(0.5,0.5,0.5,1,9);
    update(a,b);assert.equal(m.diagnostics(a).wasmCalls,0);assert.equal(m.diagnostics(a).fallbackCalls,1);
    assert.equal(m.diagnostics(a).lastFailure,suffix==='maxMemoryPages' ? 'KERNEL_MEMORY_LIMIT' : 'KERNEL_EXECUTION_FAILED');
    assert.equal(m.diagnostics(a).fieldKernels.addBall.wasmCalls,0);
    assert.equal(m.diagnostics(a).fieldKernels.addBall.fallbackCalls,1);
    assert.equal(m.diagnostics(a).fieldKernels.addBall.lastFailure,
      suffix==='maxMemoryPages' ? 'KERNEL_MEMORY_LIMIT' : 'KERNEL_EXECUTION_FAILED');
    for(const effect of [a,b]) {effect.init(2);effect.addBall(0.5,0.5,0.5,1,9);}
    compare(a,b);assert.equal(m.diagnostics(a).fieldKernels.addBall.wasmCalls,1);
  }
});

test('CLI --specialize-numeric reaches the addon, emits matching JSON and preserves the default opt-out',async()=>{
  const cli=fileURLToPath(new URL('./cli.mjs',import.meta.url));
  const out=path.join(root,'cli-out'),manifest=path.join(root,'cli-manifest.json');
  const stdout=execFileSync(process.execPath,[cli,'--entry',path.join(root,'retained-entry.mjs'),'--build-app',out,
    '--specialize-numeric','--package-root',packageRootUrl,'--output',manifest],{encoding:'utf8',timeout:120000});
  assert.match(stdout,/MarchingCubes specialization: 1 pinned addons, 6 field kernels plus polygonization/);
  const result=JSON.parse(await fs.readFile(manifest,'utf8'));
  assert.equal(result.numericSpecialization.libraryKernels.marchingCubes.compiledAddons,1);
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(out,result.numericSpecialization.reportFile),'utf8')),result.numericSpecialization);
  const module=await executeBuild(result),a=new module.MarchingCubes(4,new module.MeshPhongMaterial());
  cubeField(a,1);a.update();assert.equal(a.count,3);
  const plain=await bundleWithRollup(path.join(root,'retained-entry.mjs'),{packageRootUrl,specializeNumeric:false});
  assert.equal(plain.numericSpecialization,undefined);
  assert.ok(!plain.modules.some(id=>id.includes('f3d-marching-cubes')));
});

test('HTML module entries retain document structure and can pack the specialized runtime into one file',async()=>{
  const entry=await write('page.html','<!doctype html><title>Marching fixture</title><div id="host"></div>\n<script type="module" src="./retained-entry.mjs"></script>');
  const built=await buildApplication(entry,path.join(root,'html-out'),{packageRootUrl,specializeNumeric:true});
  assert.equal(built.numericSpecialization.libraryKernels.marchingCubes.compiledAddons,1);
  const html=await fs.readFile(path.join(built.outDir,built.htmlFile),'utf8');
  assert.ok(html.includes('<div id="host"></div>'));assert.ok(html.includes('<title>Marching fixture</title>'));
  const packed=packHtml(path.join(built.outDir,built.htmlFile),path.join(root,'packed.html'));
  assert.ok(packed.outputBytes>0);assert.ok(packed.moduleCount>0);
});

test('lookalike library code remains retained and is reported as a pin mismatch',async()=>{
  const entry=await write('lookalike.mjs','export class MarchingCubes { update() { return 42; } }');
  const built=await buildApplication(entry,path.join(root,'lookalike-out'),{specializeNumeric:true});
  const report=built.numericSpecialization.libraryKernels.marchingCubes;
  assert.equal(report.compiledAddons,0);assert.equal(report.modules[0].reason,'SOURCE_PIN_MISMATCH');
  const module=await executeBuild(built);assert.equal(new module.MarchingCubes().update(),42);
});


const fieldStats=(object,name)=>actualModule.diagnostics(object).fieldKernels[name];
function fieldCall(a,b,name,args=[],expectedArgs=args) {
  assert.equal(a[name](...args),undefined);assert.equal(b[name](...expectedArgs),undefined);compare(a,b);
}

test('all six field operations actually run native behind unchanged public methods and synchronous storage',()=>{
  const [a,b]=pair(9),methods=Object.fromEntries(['addBall','addPlaneX','addPlaneY','addPlaneZ','blur','reset'].map(name=>[name,a[name]]));
  const field=a.field,palette=a.palette;
  for(const o of [a,b])o.normal_cache.fill(31.25);
  fieldCall(a,b,'addBall',[0.5,0.5,0.5,0.9,9,[0.2,0.4,0.8]]);
  assert.ok(a.field.some(value=>value>0));assert.ok(a.palette.some(value=>value>0));
  assert.equal(a.count,0);assert.equal(a.geometry.getAttribute('position').version,0);
  for(const axis of ['X','Y','Z'])fieldCall(a,b,'addPlane'+axis,[0.15,8]);
  fieldCall(a,b,'blur',[0.5]);fieldCall(a,b,'reset');
  for(const [name,method] of Object.entries(methods)) {
    assert.equal(a[name],method);assert.equal(a[name].length,b[name].length);
    assert.equal(fieldStats(a,name).wasmCalls,1,name);assert.equal(fieldStats(a,name).fallbackCalls,0,name);
  }
  assert.equal(a.field,field);assert.equal(a.palette,palette);
  assert.equal(a.normal_cache[0],0);assert.equal(a.normal_cache[1],31.25);
  assert.equal(actualModule.Color.length,3);
});

test('metaball color conversion stays upstream for defaults, arrays, CSS, hex and genuine live Colors',()=>{
  const [a,b]=pair(8),args=[0.47,0.52,0.49,0.8,9];
  for(const value of [undefined,null,[2,-0.2,0.8],'skyblue','hsl(25, 70%, 60%)',0x52ab73]) {
    fieldCall(a,b,'addBall',[...args,value]);
  }
  const ac=new actualModule.Color(0.1,0.2,0.3),bc=new expectedModule.Color(0.1,0.2,0.3);
  fieldCall(a,b,'addBall',[...args,ac],[...args,bc]);
  ac.r=bc.r=0.91;ac.g=bc.g=-0.1;
  fieldCall(a,b,'addBall',[...args,ac],[...args,bc]);
  assert.equal(fieldStats(a,'addBall').wasmCalls,8);assert.equal(fieldStats(a,'addBall').fallbackCalls,0);
  update(a,b);
});

test('accessor Colors retain per-cell observations and partial JavaScript writes when a getter throws',()=>{
  const [a,b]=pair(8),ac=new actualModule.Color(),bc=new expectedModule.Color(),args=[0.5,0.5,0.5,0.8,9];
  let ar=0,br=0;
  Object.defineProperty(ac,'r',{get(){return 0.1+(++ar)/1000;},configurable:true});
  Object.defineProperty(bc,'r',{get(){return 0.1+(++br)/1000;},configurable:true});
  fieldCall(a,b,'addBall',[...args,ac],[...args,bc]);
  assert.equal(ar,br);assert.ok(ar>1);assert.equal(fieldStats(a,'addBall').wasmCalls,0);
  const sentinel={};ar=br=0;
  Object.defineProperty(ac,'r',{get(){if(++ar===3)throw sentinel;return 0.2;},configurable:true});
  Object.defineProperty(bc,'r',{get(){if(++br===3)throw sentinel;return 0.2;},configurable:true});
  assert.throws(()=>a.addBall(...args,ac),error=>error===sentinel);
  assert.throws(()=>b.addBall(...args,bc),error=>error===sentinel);
  assert.equal(ar,3);assert.equal(br,3);compare(a,b);
  assert.equal(fieldStats(a,'addBall').fallbackCalls,2);
});

test('proxy Colors and objects returned by overridden Color.set are never speculatively inspected',()=>{
  for(const returned of [false,true]) {
    const [a,b]=pair(8),traces=[[],[]];let descriptors=0;
    const colors=[actualModule,expectedModule].map((m,i)=>{
      const wrap=target=>new Proxy(target,{get(target,key){traces[i].push(key);return Reflect.get(target,key);},
        getOwnPropertyDescriptor(){descriptors++;throw Error('unexpected speculative descriptor');}});
      if (!returned)return wrap(new m.Color(0.2,0.3,0.4));
      class ReturningColor extends m.Color {set(...args){return wrap(super.set(...args));}}
      return new ReturningColor(0.2,0.3,0.4);
    });
    const args=[0.5,0.5,0.5,0.8,9];
    fieldCall(a,b,'addBall',[...args,colors[0]],[...args,colors[1]]);
    assert.deepEqual(traces[0],traces[1]);assert.equal(descriptors,0);
    assert.equal(fieldStats(a,'addBall').lastFailure,'MARCHING_CUBES_FIELD_GUARD');
  }
});

test('color-conversion exceptions are caught once by original setup, then the recovered default color can run native',()=>{
  const [a,b]=pair(8);let ar=0,br=0;
  const ac=[0.2,0.3,0.4],bc=[0.2,0.3,0.4];
  Object.defineProperty(ac,0,{get(){ar++;throw Error('color input');}});
  Object.defineProperty(bc,0,{get(){br++;throw Error('color input');}});
  const args=[0.5,0.5,0.5,0.8,9];fieldCall(a,b,'addBall',[...args,ac],[...args,bc]);
  assert.equal(ar,1);assert.equal(br,1);assert.equal(fieldStats(a,'addBall').wasmCalls,1);
});

test('numeric coercions in plane parameters occur only in the original prefix and retained loop',()=>{
  const [a,b]=pair(8);let ar=0,br=0;
  const strengthA={valueOf(){ar++;return 0.5;}},strengthB={valueOf(){br++;return 0.5;}};
  for(const name of ['addPlaneX','addPlaneY','addPlaneZ']) {
    fieldCall(a,b,name,[strengthA,9],[strengthB,9]);
    assert.equal(ar,br);assert.equal(fieldStats(a,name).wasmCalls,0);
    assert.equal(fieldStats(a,name).lastFailure,'KERNEL_SCALAR_TYPE');
  }
  assert.ok(ar>3);
});

test('blur executes an overridden slice once; an alias result retains the original in-place update order',()=>{
  for(const alias of [false,true]) {
    const [a,b]=pair(7);let ac=0,bc=0;
    for(let i=0;i<a.field.length;i++)a.field[i]=b.field[i]=(i%19-7)/9;
    const slice=Float32Array.prototype.slice;
    a.field.slice=function(){ac++;return alias?this:Reflect.apply(slice,this,[]);};
    b.field.slice=function(){bc++;return alias?this:Reflect.apply(slice,this,[]);};
    fieldCall(a,b,'blur',[0.7]);assert.equal(ac,1);assert.equal(bc,1);
    assert.equal(fieldStats(a,'blur').wasmCalls,alias?0:1);
    assert.equal(fieldStats(a,'blur').lastFailure,alias?'KERNEL_ARRAY_ALIAS':null);
  }
});

test('changed sqrt accessors select the original loop without speculative getter calls and recover after restoration',()=>{
  const [a,b]=pair(8),d=Object.getOwnPropertyDescriptor(Math,'sqrt');let reads=0;
  const args=[0.5,0.5,0.5,0.8,9];let actualReads,expectedReads;
  try {
    Object.defineProperty(Math,'sqrt',{configurable:true,get(){reads++;return d.value;}});
    a.addBall(...args);actualReads=reads;reads=0;b.addBall(...args);expectedReads=reads;
  } finally {Object.defineProperty(Math,'sqrt',d);}
  compare(a,b);assert.equal(actualReads,expectedReads);assert.ok(actualReads>1);
  assert.equal(fieldStats(a,'addBall').lastFailure,'KERNEL_MATH_BINDING');
  fieldCall(a,b,'addBall',args);assert.equal(fieldStats(a,'addBall').wasmCalls,1);
});

test('owner accessors and borrowed receivers preserve original field access traces without speculative reads',()=>{
  const [a,b]=pair(8),af=a.field,bf=b.field;let ar=0,br=0;
  Object.defineProperty(a,'field',{get(){ar++;return af;},configurable:true});
  Object.defineProperty(b,'field',{get(){br++;return bf;},configurable:true});
  a.addBall(0.5,0.5,0.5,0.8,9);b.addBall(0.5,0.5,0.5,0.8,9);
  assert.equal(ar,br);assert.ok(ar>1);compare(a,b);
  Object.defineProperty(a,'field',{value:af,writable:true,configurable:true});
  Object.defineProperty(b,'field',{value:bf,writable:true,configurable:true});
  let traps=0;
  const proxy=target=>new Proxy(target,{getOwnPropertyDescriptor(){traps++;throw Error('descriptor');}});
  a.addPlaneX.call(proxy(a),0.1,8);b.addPlaneX.call(proxy(b),0.1,8);
  assert.equal(traps,0);compare(a,b);
  assert.equal(fieldStats(a,'addPlaneX').lastFailure,'MARCHING_CUBES_RECEIVER');
});

test('late field bounds and alias failures do not publish speculative density before retained palette writes',()=>{
  const [a,b]=pair(8),args=[0.5,0.5,0.5,0.8,9,[0.2,0.3,0.4]];
  a.palette=a.field;b.palette=b.field;fieldCall(a,b,'addBall',args);
  assert.equal(fieldStats(a,'addBall').lastFailure,'KERNEL_ARRAY_ALIAS');
  a.palette=new Float32Array(7);b.palette=new Float32Array(7);fieldCall(a,b,'addBall',args);
  assert.equal(fieldStats(a,'addBall').lastFailure,'KERNEL_EXECUTION_FAILED');
  a.init(8);b.init(8);fieldCall(a,b,'addBall',args);
  assert.equal(fieldStats(a,'addBall').wasmCalls,1);assert.equal(fieldStats(a,'addBall').fallbackCalls,2);
});

test('the source-tree import-map route registers Color and EventDispatcher without requiring the built core',async()=>{
  const map={imports:{three:new URL('src/Three.js',packageRootUrl).href,'three/addons/':new URL('examples/jsm/',packageRootUrl).href}};
  const entry=await write('source-route.html',`<script type="importmap">${JSON.stringify(map)}</script><script type="module" src="./native-entry.mjs"></script>`);
  const built=await buildApplication(entry,path.join(root,'source-route'),{specializeNumeric:true});
  const m=await executeBuild(built),a=new m.MarchingCubes(8,new m.MeshPhongMaterial());
  a.addBall(0.5,0.5,0.5,0.8,9);a.update();assert.ok(a.count>0);
  assert.equal(m.diagnostics(a).fieldKernels.addBall.wasmCalls,1);assert.equal(m.diagnostics(a).wasmCalls,1);
  assert.equal(built.numericSpecialization.libraryKernels.marchingCubes.registeredColorModules,1);
});

test('dynamic chunks share genuine color ownership and lazy field dispatch with their consumer',async()=>{
  await write('field-maker.mjs',`import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
    import {Color,MeshPhongMaterial} from 'three';
    export function make(){const effect=new MarchingCubes(8,new MeshPhongMaterial());
      effect.addBall(0.5,0.5,0.5,0.8,9,new Color(0.2,0.4,0.7));return effect;}`);
  const entry=await write('field-consumer.mjs',`export {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};
    export async function make(){return (await import('./field-maker.mjs')).make();}`);
  const built=await buildApplication(entry,path.join(root,'dynamic-fields'),{packageRootUrl,specializeNumeric:true});
  assert.ok(built.isMultiChunk);const m=await executeBuild(built),a=await m.make();
  assert.equal(m.diagnostics(a).fieldKernels.addBall.wasmCalls,1);a.update();assert.equal(m.diagnostics(a).wasmCalls,1);
});
