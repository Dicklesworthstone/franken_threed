import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeInstancedMatrixBase,specializeInstancedMatrixLoops} from './instanced_matrix_update_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const html=fs.readFileSync(path.join(upstream,'examples/webgpu_performance_renderbundle.html'),'utf8');
const moduleSource=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
let originalLoop;
walk.simple(acorn.parse(moduleSource,{ecmaVersion:'latest',sourceType:'module'}),{FunctionDeclaration(node){
  if(node.id.name==='animateMeshes'){
    const loop=node.body.body.filter(statement=>statement.type==='ForStatement')[1];
    originalLoop=moduleSource.slice(loop.start,loop.end);
  }
}});
assert.ok(originalLoop);
const original=new Function('instancedMesh','rotationSpeeds','rotationMatrix','instanceMatrix',originalLoop+
  '\ninstancedMesh.instanceMatrix.needsUpdate=true;');
const renamed=`for(let slot=0;slot<instances.count;++slot){turn.makeRotationFromEuler(speeds[slot]);
  instances.getMatrixAt(slot,temp);temp.premultiply(turn);instances.setMatrixAt(slot,temp);}`;
const reverse=new Function('instances','speeds','turn','temp',renamed);
const constructorsSource=`function selectedConstructors(trace){
  class Example{constructor(value=17){trace.push('construct');this.value=value;}}
  function argument(){trace.push('argument');return 23;}
  const first=new [Example][0]().value;
  const second=new [Example][0](argument()).value;
  const nested=new [[Example][0]][0](29).value;
  return [first,second,nested];
}`;
const sourceConstructors=new Function('return ('+constructorsSource+')')();
async function fixture(options={}){
  const entry='\0instance-update-fixture',plugin=marchingCubesRollupPlugin(options);
  const code='import * as T from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';export {T};\n'+
    'export {instancedMatrixUpdateDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-instanced-matrix-update-adapter')+';\n'+
    'export {matrixMultiplyDiagnostics as multiplyDiagnostics} from '+JSON.stringify('\0f3d-matrix-multiply-adapter')+';\n'+
    'export function speeds(count){const list=[];for(let i=0;i<count;i++)list.push(new T.Euler((i+1)/107,(i+2)/211,-(i+3)/307,["XYZ","YXZ","ZXY","ZYX","YZX","XZY"][i%6]));return list;}\n'+
    'export function update(instancedMesh,rotationSpeeds,rotationMatrix,instanceMatrix){'+originalLoop+'\ninstancedMesh.instanceMatrix.needsUpdate=true;}\n'+
    'export function reverse(instances,speeds,turn,temp){'+renamed+'}\nexport '+constructorsSource;
  const bundle=await rollup({input:entry,plugins:[{name:'instance-update-fixture',resolveId:id=>id===entry?id:null,
    load:id=>id===entry?code:null},plugin],onwarn(warning,warn){if(warning.code!=='CIRCULAR_DEPENDENCY')warn(warning);}});
  try{
    const result=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64')),
      report:plugin.api.getInstancedMatrixUpdatesReport()};
  }finally{await bundle.close();}
}
let compiled,limited;
before(async()=>{compiled=await fixture();limited=await fixture({maxIterations:1});});
const makeSpeeds=(T,count)=>Array.from({length:count},(_,i)=>new T.Euler((i+1)/107,(i+2)/211,-(i+3)/307,
  ['XYZ','YXZ','ZXY','ZYX','YZX','XZY'][i%6]));
function state(T,speeds,count){
  const mesh=new T.InstancedMesh(new T.BufferGeometry(),new T.MeshBasicMaterial(),count),initial=new T.Matrix4();
  for(let i=0;i<count;i++){
    for(let j=0;j<16;j++)initial.elements[j]=(j%5===0?1:0)+(i+1)*(j+1)/200;
    mesh.setMatrixAt(i,initial);
  }
  return {mesh,speeds,rotation:new T.Matrix4(),scratch:new T.Matrix4()};
}
function pair(count=10,api=compiled.api){return {api,
  actual:state(api.T,api.speeds(count),count),expected:state(reference,makeSpeeds(reference,count),count)};}
const args=f=>[f.mesh,f.speeds,f.rotation,f.scratch];
function same(a,b){
  assert.deepEqual(a.mesh.instanceMatrix.array,b.mesh.instanceMatrix.array);
  assert.deepEqual(a.rotation.elements,b.rotation.elements);assert.deepEqual(a.scratch.elements,b.scratch.elements);
  assert.equal(a.mesh.instanceMatrix.version,b.mesh.instanceMatrix.version);
  assert.deepEqual(a.speeds.map(value=>value.toArray()),b.speeds.map(value=>value.toArray()));
}
function advance(f){f.api.update(...args(f.actual));original(...args(f.expected));same(f.actual,f.expected);}

test('unchanged H1 instance loop and renamed premultiply shape derive arithmetic from the pinned core',()=>{
  const source=fs.readFileSync(path.join(upstream,'build/three.core.js'),'utf8');
  assert.equal(specializeInstancedMatrixBase(source+'\n// modified').changed,false);
  assert.equal(specializeInstancedMatrixLoops(moduleSource,{runtimeModule:'./runtime.mjs'}).report.compiledLoops,1);
  assert.equal(compiled.report.compiledModules,1);assert.equal(compiled.report.compiledLoops,2);
  assert.equal(compiled.report.accelerationClaim,false);
  const kernel=compiled.report.modules.find(record=>record.kind==='pinned-instanced-matrix-kernel');
  assert.ok(kernel.wasmBytes>0);assert.deepEqual(kernel.retainedMath,['cos','sin']);
});

test('the H1 update uses one transaction, preserves every instance, final scratch state, identities and upload version',()=>{
  const f=pair(),a=f.actual,identity=[a.mesh.instanceMatrix,a.mesh.instanceMatrix.array,a.rotation,a.rotation.elements,a.scratch,a.scratch.elements];
  for(let i=0;i<5;i++){
    advance(f);assert.deepEqual(f.api.diagnostics(a.mesh),{wasmCalls:i+1,instances:(i+1)*10,fallbackCalls:0,lastFailure:null});
  }
  assert.deepEqual([a.mesh.instanceMatrix,a.mesh.instanceMatrix.array,a.rotation,a.rotation.elements,a.scratch,a.scratch.elements],identity);
  assert.equal(f.api.multiplyDiagnostics(a.scratch).wasmCalls,0,'no per-instance multiply bridge crossings');
});

test('literal allocation witnesses preserve new-expression precedence for array-selected constructors',()=>{
  const trace=[],sourceTrace=[];
  assert.deepEqual(compiled.api.selectedConstructors(trace),sourceConstructors(sourceTrace));
  assert.deepEqual(compiled.api.selectedConstructors([]),[17,23,29]);assert.deepEqual(trace,sourceTrace);
});

test('all Euler orders, projective matrices, premultiply and Float64 instance storage retain Number results',()=>{
  const f=pair(30);
  for(const side of [f.actual,f.expected])side.mesh.instanceMatrix.array=new Float64Array(side.mesh.instanceMatrix.array);
  for(let i=0;i<4;i++){f.api.reverse(...args(f.actual));reverse(...args(f.expected));same(f.actual,f.expected);}
  assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,4);
  for(const side of [f.actual,f.expected]){
    side.speeds[0]._x=-0;side.speeds[1]._y=NaN;side.speeds[2]._z=Infinity;
    side.mesh.instanceMatrix.array[50]=-Infinity;
  }
  advance(f);assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,5);
});

test('current counts and Euler values are recaptured while zero iterations preserve both scratch matrices',()=>{
  const f=pair(12);
  for(const count of [3,0,12,7]){
    for(const side of [f.actual,f.expected]){side.mesh.count=count;side.speeds[0].x+=.125;}
    advance(f);
  }
  assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,3);assert.equal(f.api.diagnostics(f.actual.mesh).instances,22);
});

test('unknown arrays, proxy wrappers and getters retain exactly the source observations',()=>{
  for(const mode of ['mesh','speeds','speed','component','slot','count']){
    const f=pair(4),traces=[[],[]];
    for(const [which,side]of [f.actual,f.expected].entries()){
      const wrap=object=>new Proxy(object,{get(target,key,receiver){traces[which].push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(){assert.fail('guard inspected a proxy');}});
      if(mode==='mesh')side.mesh=wrap(side.mesh);
      if(mode==='speeds')side.speeds=wrap(side.speeds);
      if(mode==='speed')side.speeds[1]=wrap(side.speeds[1]);
      if(mode==='component')Object.defineProperty(side.speeds[1],'x',{get(){traces[which].push('x');return .25;}});
      if(mode==='slot'){const speed=side.speeds[1];Object.defineProperty(side.speeds,1,{get(){traces[which].push('slot');return speed;}});}
      if(mode==='count')Object.defineProperty(side.mesh,'count',{get(){traces[which].push('count');return 4;}});
    }
    f.api.update(...args(f.actual));original(...args(f.expected));
    assert.deepEqual(traces[0],traces[1]);same(f.actual,f.expected);
  }
  const f=pair();f.actual.speeds=[...f.actual.speeds];advance(f);
  assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,0);
});

test('overridden methods retain callback order, reentry and source intermediate instance writes',()=>{
  const f=pair(4),traces=[[],[]];
  for(const [which,side]of [f.actual,f.expected].entries()){
    const originalSet=side.mesh.setMatrixAt;let inside=false;
    side.mesh.setMatrixAt=function(index,matrix){
      traces[which].push([index,this.instanceMatrix.array[0],matrix.elements[0]]);
      const result=originalSet.call(this,index,matrix);
      if(index===0&&!inside){inside=true;const count=this.count;this.count=1;
        (which===0?f.api.update:original)(...args(side));this.count=count;inside=false;}
      return result;
    };
  }
  advance(f);assert.deepEqual(traces[0],traces[1]);assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,0);
});

test('aliased scratch objects or elements keep the original matrix read/write order',()=>{
  for(const alias of ['object','elements']){
    const f=pair(4);
    for(const side of [f.actual,f.expected]){
      if(alias==='object')side.rotation=side.scratch;else side.rotation.elements=side.scratch.elements;
    }
    advance(f);assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,0);
  }
});

test('invalid later input and nonwritable scratch cells retain source partial updates and errors',()=>{
  for(const mode of ['missing-speed','nonwritable','accessor-error']){
    const f=pair(4);
    for(const side of [f.actual,f.expected]){
      if(mode==='missing-speed')side.speeds.length=2;
      if(mode==='nonwritable')Object.defineProperty(side.rotation.elements,8,{writable:false});
      if(mode==='accessor-error')Object.defineProperty(side.speeds[2],'x',{get(){throw new Error('authored read error');}});
    }
    assert.throws(()=>f.api.update(...args(f.actual)));assert.throws(()=>original(...args(f.expected)));
    assert.deepEqual(f.actual.mesh.instanceMatrix.array,f.expected.mesh.instanceMatrix.array);
    assert.deepEqual(f.actual.scratch.elements,f.expected.scratch.elements);assert.deepEqual(f.actual.rotation.elements,f.expected.rotation.elements);
    assert.equal(f.actual.mesh.instanceMatrix.version,f.expected.mesh.instanceMatrix.version);
    assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,0);
  }
});

test('native budget failures fall back once without publishing speculative output',()=>{
  const f=pair(4,limited.api);advance(f);
  assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,0);assert.equal(f.api.diagnostics(f.actual.mesh).fallbackCalls,1);
  assert.equal(f.api.diagnostics(f.actual.mesh).lastFailure,'KERNEL_EXECUTION_FAILED');
});

test('unproved method sequences, escaping induction variables and global captures keep source loops',()=>{
  const wrap=body=>'function update(instancedMesh,rotationSpeeds,rotationMatrix,instanceMatrix){'+body+'}';
  for(const code of [originalLoop,wrap(originalLoop.replace('let i = 0','var i = 0')),
    wrap(originalLoop.replace('instanceMatrix.multiply( rotationMatrix );','observe(instanceMatrix);')),
    wrap(originalLoop.replace('i ++','i += 2')),wrap(originalLoop)+'\neval(source);'])
    assert.equal(specializeInstancedMatrixLoops(code,{runtimeModule:'./runtime.mjs'}).changed,false);
});
