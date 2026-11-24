/** Source-module ownership for the existing whole-addBall Wasm transaction.
 * Every public algorithm and class comes from the pinned r186 source tree. */
import assert from 'node:assert/strict';
import {before,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const colorPath=path.join(upstream,'src/math/Color.js');
const managementPath=path.join(upstream,'src/math/ColorManagement.js');
const sourceModule=name=>pathToFileURL(path.join(upstream,name)).href;
const publicSource=`
import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
import {Color,MeshBasicMaterial,ColorManagement} from 'three';
export {Color,ColorManagement};
export function create(){return new MarchingCubes(8,new MeshBasicMaterial(),true,true,2000);}
export function update(effect,count,time,mode='default'){
  const colors=[new Color(.2,.4,.6),new Color(.7,.1,.3)];
  for(let i=0;i<count;i++){
    const x=Math.sin(time+i)*.2+.5;
    if(mode==='colors')effect.addBall(x,.5,.4,.8,12,colors[i%2]);
    else effect.addBall(x,.5,.4,.8,12);
  }
  effect.update();
}
`;
let root,serial=0,ordinary;
before(async()=>{root=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-source-ball-colors-'));});

async function build(mode='direct',specialize=true) {
  const dir=path.join(root,`${mode}-${specialize}-${serial++}`);
  await fs.mkdir(dir);
  const barrel=path.join(dir,'three.mjs'),bridge=path.join(dir,'management.mjs');
  const entries=[['BufferAttribute','src/core/BufferAttribute.js'],['BufferGeometry','src/core/BufferGeometry.js'],
    ['Color','src/math/Color.js'],['DynamicDrawUsage','src/constants.js'],['Mesh','src/objects/Mesh.js'],
    ['Sphere','src/math/Sphere.js'],['Vector3','src/math/Vector3.js'],['MeshBasicMaterial','src/materials/MeshBasicMaterial.js'],
    ['ColorManagement','src/math/ColorManagement.js']];
  await fs.writeFile(barrel,entries.map(([name,file])=>`export {${name}} from ${JSON.stringify(sourceModule(file))};`).join('\n'));
  let prefix='';
  if(mode==='reexport'||mode==='cycle'){
    const body=mode==='cycle'
      ?`import {Color} from ${JSON.stringify(pathToFileURL(colorPath).href)};
        import {ColorManagement as original,SRGBToLinear,LinearToSRGB} from ${JSON.stringify(pathToFileURL(managementPath).href)};
        export const ColorManagement=original;
        export {SRGBToLinear,LinearToSRGB}; export const observedClass=Color;`
      :`export {ColorManagement,SRGBToLinear,LinearToSRGB} from ${JSON.stringify(pathToFileURL(managementPath).href)};`;
    await fs.writeFile(bridge,body);
    if(mode==='cycle')prefix=`export {observedClass} from ${JSON.stringify(pathToFileURL(bridge).href)};\n`;
  }else if(mode==='proxy'||mode==='live'){
    await fs.writeFile(bridge,`
      import {ColorManagement as original,SRGBToLinear,LinearToSRGB} from ${JSON.stringify(pathToFileURL(managementPath).href)};
      export {SRGBToLinear,LinearToSRGB};
      export let traps=0;
      const proxy=new Proxy(original,{
        get(target,key,receiver){traps++;return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(target,key){traps++;return Reflect.getOwnPropertyDescriptor(target,key);}
      });
      export let ColorManagement=${mode==='proxy'?'proxy':'original'};
      export function replace(){ColorManagement=proxy;}
    `);
    prefix=`export {traps,replace} from ${JSON.stringify(pathToFileURL(bridge).href)};\n`;
  }
  const entry=path.join(dir,'entry.mjs');
  await fs.writeFile(entry,prefix+publicSource+(specialize?
    `export {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};`:''));
  const plugin=specialize?marchingCubesRollupPlugin():null;
  const bundle=await rollup({input:entry,plugins:[{
    name:'source-module-color-fixture',
    resolveId(id,importer){
      if(id==='three')return barrel;
      if(id==='three/addons/objects/MarchingCubes.js')return path.join(upstream,'examples/jsm/objects/MarchingCubes.js');
      if(id.startsWith('file:'))return fileURLToPath(id);
      if(importer===colorPath&&id==='./ColorManagement.js'&&['reexport','cycle','proxy','live'].includes(mode))return bridge;
      return null;
    },
    async load(id){
      if(mode==='modified'&&id===managementPath)return await fs.readFile(id,'utf8')+'\n// User-modified dependency.\n';
      return null;
    },
  },...(plugin?[plugin]:[])],onwarn(warning,handler){if(warning.code!=='CIRCULAR_DEPENDENCY')handler(warning);}});
  const output=path.join(dir,'compiled.mjs');
  try{await bundle.write({file:output,format:'es'});}finally{await bundle.close();}
  // Execute after moving the complete emitted module, without source lookups.
  const relocated=path.join(dir,'relocated.mjs');await fs.rename(output,relocated);
  return {module:await import(pathToFileURL(relocated).href),report:plugin?.api.getReport()};
}
function same(a,b){
  assert.equal(a.count,b.count);assert.deepEqual(a.geometry.drawRange,b.geometry.drawRange);
  for(const name of ['field','palette','normal_cache','positionArray','normalArray','uvArray','colorArray'])assert.deepEqual(a[name],b[name],name);
  for(const name of ['position','normal','uv','color'])assert.equal(a.geometry.getAttribute(name).version,b.geometry.getAttribute(name).version,name);
}
function runPair(actual,expected,mode='default',count=7){
  const a=actual.create(),b=expected.create(),field=a.field,palette=a.palette;
  for(const time of [.125,.375]){actual.update(a,count,time,mode);expected.update(b,count,time,mode);same(a,b);}
  assert.equal(a.field,field);assert.equal(a.palette,palette);
  return [a,b];
}

test('pinned source Color and management connect default and palette loops to real Wasm',async()=>{
  ordinary=await build();const reference=await build('direct',false);
  assert.equal(ordinary.report.compiledBallLoops,1);
  for(const mode of ['default','colors']){
    const [a]=runPair(ordinary.module,reference.module,mode,17);
    assert.equal(ordinary.module.diagnostics(a).ballLoops.wasmCalls,2);
    assert.equal(ordinary.module.diagnostics(a).ballLoops.balls,34);
    assert.equal(ordinary.module.diagnostics(a).ballLoops.fallbackCalls,0);
    assert.equal(ordinary.module.diagnostics(a).fieldKernels.addBall?.wasmCalls??0,0);
  }
});

test('source management mutation retains every callback and permits later native recovery',async()=>{
  ordinary??=await build();
  const actual=ordinary.module,expected=(await build('direct',false)).module;
  for(const key of ['convert','colorSpaceToWorking']){
    const a=actual.create(),b=expected.create(),counts=[0,0],original=[actual,expected].map(module=>module.ColorManagement[key]);
    for(const [i,module] of [actual,expected].entries())module.ColorManagement[key]=function(...args){counts[i]++;return original[i].apply(this,args);};
    try{actual.update(a,5,.2,'colors');expected.update(b,5,.2,'colors');}
    finally{for(const [i,module] of [actual,expected].entries())module.ColorManagement[key]=original[i];}
    same(a,b);assert.ok(counts[0]>0);assert.equal(counts[0],counts[1]);
    assert.equal(actual.diagnostics(a).ballLoops.wasmCalls,0);
    assert.equal(actual.diagnostics(a).ballLoops.fallbackCalls,1);
    actual.update(a,5,.3,'colors');expected.update(b,5,.3,'colors');same(a,b);
    assert.equal(actual.diagnostics(a).ballLoops.wasmCalls,1);
  }
});

for(const mode of ['reexport','cycle'])test(`${mode} preserves the actual source dependency and its initialization order`,async()=>{
  const actual=await build(mode),expected=await build(mode,false);
  const [a]=runPair(actual.module,expected.module,'colors');
  if(mode==='cycle'){
    assert.equal(actual.module.observedClass,actual.module.Color);
    assert.equal(expected.module.observedClass,expected.module.Color);
  }
  assert.equal(actual.module.diagnostics(a).ballLoops.wasmCalls,2);
});

for(const mode of ['modified','proxy'])test(`${mode} management keeps the original loop without speculative dependency observations`,async()=>{
  const actual=await build(mode),expected=await build(mode,false);
  const [a]=runPair(actual.module,expected.module,'colors');
  assert.equal(actual.module.diagnostics(a).ballLoops.wasmCalls,0);
  assert.equal(actual.module.diagnostics(a).ballLoops.fallbackCalls,2);
  if(mode==='proxy'){assert.ok(actual.module.traps>0);assert.equal(actual.module.traps,expected.module.traps);}
});

test('a re-exported live binding is rechecked after a successful batch',async()=>{
  const actual=(await build('live')).module,expected=(await build('live',false)).module;
  const [a,b]=runPair(actual,expected);
  assert.equal(actual.diagnostics(a).ballLoops.wasmCalls,2);
  actual.replace();expected.replace();
  actual.update(a,5,.6);expected.update(b,5,.6);same(a,b);
  assert.equal(actual.diagnostics(a).ballLoops.wasmCalls,2);
  assert.equal(actual.diagnostics(a).ballLoops.fallbackCalls,1);
  assert.ok(actual.traps>0);assert.equal(actual.traps,expected.traps);
});
