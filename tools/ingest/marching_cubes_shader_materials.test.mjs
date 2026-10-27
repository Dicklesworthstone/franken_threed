/** Actual H2 materials keep source polygonization and public geometry current. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import * as acorn from 'acorn';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeMarchingCubesBase} from './marching_cubes_specialization.mjs';
import * as Reference from '../../upstream/three.js/build/three.core.js';
import * as referenceShaders from '../../upstream/three.js/examples/jsm/shaders/ToonShader.js';
import {MarchingCubes as ReferenceCubes} from '../../upstream/three.js/examples/jsm/objects/MarchingCubes.js';
import {OBJExporter} from '../../upstream/three.js/examples/jsm/exporters/OBJExporter.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const html=fs.readFileSync(path.join(upstream,'examples/webgl_marchingcubes.html'),'utf8');
const entrySource=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const entryAst=acorn.parse(entrySource,{ecmaVersion:'latest',sourceType:'module'});
const applicationFunctions=entryAst.body.filter(node=>node.type==='FunctionDeclaration' &&
  ['generateMaterials','createShaderMaterial','updateCubes'].includes(node.id.name))
  .map(node=>entrySource.slice(node.start,node.end)).join('\n');
function application(T,shaders=T) {
  // This test's resource boundary needs the real texture identities; no pixels
  // are rendered. Material definitions and numeric updates are the original H2.
  const THREE={...T,CubeTextureLoader:class{load(){return new T.CubeTexture();}},
    TextureLoader:class{load(){return new T.Texture();}}};
  return new Function('THREE','shaders',
    'const {ToonShader1,ToonShader2,ToonShaderHatching,ToonShaderDotted}=shaders;'+
    'const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(.5,.5,1);'+
    'const ambientLight=new THREE.AmbientLight(0x323232,3);let current_material="shiny";'+
    applicationFunctions+';return {materials:generateMaterials(),update:(effect,name,time)=>{'+
    'current_material=name;updateCubes(effect,time,4,true,false,false);}};')(THREE,shaders);
}
async function compiled(core,{changeMaterial=false}={}) {
  const entry='\0marching-shader-fixture',packageEntry=path.join(upstream,core?'build/three.core.js':'src/Three.Core.js');
  const materialFile=path.join(upstream,'src/materials/Material.js');
  const source='export * from '+JSON.stringify(packageEntry)+';'+
    (core?'':'export {UniformsUtils} from '+JSON.stringify(path.join(upstream,'src/renderers/shaders/UniformsUtils.js'))+';')+
    'export {MarchingCubes} from '+JSON.stringify(path.join(upstream,'examples/jsm/objects/MarchingCubes.js'))+';'+
    'export {ToonShader1,ToonShader2,ToonShaderHatching,ToonShaderDotted} from '+
      JSON.stringify(path.join(upstream,'examples/jsm/shaders/ToonShader.js'))+';'+
    'export {marchingCubesDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-marching-cubes-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'marching-shader-fixture',
    resolveId(id){if(id===entry)return id;if(id==='three')return packageEntry;return null;},
    load(id){if(id===entry)return source;if(changeMaterial && id===materialFile)return fs.readFileSync(id,'utf8')+'\n// application edit\n';return null;}},
    marchingCubesRollupPlugin()]});
  try {
    const output=await bundle.generate({format:'es'});
    return await import('data:text/javascript;base64,'+Buffer.from(output.output[0].code).toString('base64'));
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=await Promise.all([compiled(false),compiled(true)]);});
function pair(T,material=new T.ShaderMaterial(),referenceMaterial=new Reference.ShaderMaterial()) {
  return [new T.MarchingCubes(8,material,true,true,2000),new ReferenceCubes(8,referenceMaterial,true,true,2000)];
}
function seed(effect) {effect.reset();effect.addBall(.4,.5,.5,1.2,12);effect.addBall(.6,.5,.5,1.1,12);}
function equal(actual,expected) {
  assert.equal(actual.count,expected.count);
  assert.deepEqual(actual.geometry.drawRange,expected.geometry.drawRange);
  for(const key of ['field','normal_cache','palette','positionArray','normalArray','uvArray','colorArray'])
    assert.deepEqual(actual[key],expected[key],key);
  for(const name of ['position','normal','uv','color']) {
    assert.equal(actual.geometry.getAttribute(name).version,expected.geometry.getAttribute(name).version,name);
    assert.equal(actual.geometry.getAttribute(name).array,
      actual[{position:'positionArray',normal:'normalArray',uv:'uvArray',color:'colorArray'}[name]]);
  }
}
function update(actual,expected) {seed(actual);seed(expected);actual.update();expected.update();equal(actual,expected);}
function withProperty(object,key,description,run) {
  const original=Object.getOwnPropertyDescriptor(object,key);
  Object.defineProperty(object,key,{configurable:true,...description});
  try{return run();}finally{if(original)Object.defineProperty(object,key,original);else delete object[key];}
}

test('all 13 actual H2 material branches use the native polygonizer in source and core builds',()=>{
  for(const T of fixtures) {
    const actualApp=application(T),referenceApp=application(Reference,referenceShaders);
    const [actual,expected]=pair(T,actualApp.materials.shiny,referenceApp.materials.shiny);
    const method=actual.update,arrays=[actual.positionArray,actual.normalArray,actual.uvArray,actual.colorArray];
    assert.equal(Object.keys(actualApp.materials).length,13);
    for(const name of Object.keys(actualApp.materials)) {
      actual.material=actualApp.materials[name];expected.material=referenceApp.materials[name];
      for(const effect of [actual,expected]) {
        effect.enableUvs=name==='textured';effect.enableColors=name==='colors'||name==='multiColors';
      }
      actualApp.update(actual,name,.125);referenceApp.update(expected,name,.125);equal(actual,expected);
      if(actual.material.isShaderMaterial) {
        assert.equal(Object.hasOwn(actual.material,'flatShading'),false);
        assert.equal(Object.getPrototypeOf(actual.material),T.ShaderMaterial.prototype);
      }
    }
    assert.equal(actual.update,method);
    assert.deepEqual([actual.positionArray,actual.normalArray,actual.uvArray,actual.colorArray],arrays);
    assert.equal(T.diagnostics(actual).wasmCalls,13);assert.equal(T.diagnostics(actual).fallbackCalls,0);
  }
});

test('own and witnessed inherited flatShading data use the exact strict-true source condition',()=>{
  for(const T of fixtures)for(const flatShading of [true,false,undefined,1,null]) {
    const [actual,expected]=pair(T);
    actual.material.flatShading=flatShading;expected.material.flatShading=flatShading;
    update(actual,expected);assert.equal(T.diagnostics(actual).wasmCalls,1);
    const [inherited,reference]=pair(T);
    withProperty(T.Material.prototype,'flatShading',{value:flatShading,writable:true},()=>
      withProperty(Reference.Material.prototype,'flatShading',{value:flatShading,writable:true},()=>update(inherited,reference)));
    assert.equal(Object.hasOwn(inherited.material,'flatShading'),false);
    assert.equal(T.diagnostics(inherited).wasmCalls,1);
  }
});

test('inherited material accessors retain the original reads and numeric body',()=>{
  for(const T of fixtures)for(const name of ['ShaderMaterial','Material']) {
    const [actual,expected]=pair(T);let actualReads=0,expectedReads=0;
    withProperty(T[name].prototype,'flatShading',{get(){actualReads++;return true;}},()=>
      withProperty(Reference[name].prototype,'flatShading',{get(){expectedReads++;return true;}},()=>update(actual,expected)));
    assert.ok(actualReads>0);assert.equal(actualReads,expectedReads);
    assert.equal(T.diagnostics(actual).wasmCalls,0);assert.equal(T.diagnostics(actual).fallbackCalls,1);
  }
});

test('Object.prototype flatShading remains observable instead of being assumed absent',()=>{
  for(const T of fixtures) {
    const [actual,expected]=pair(T);seed(actual);seed(expected);const reads={actual:0,expected:0};let side='actual';
    withProperty(Object.prototype,'flatShading',{get(){reads[side]++;return false;}},()=>{
      actual.update();side='expected';expected.update();
    });
    equal(actual,expected);assert.ok(reads.actual>0);assert.equal(reads.actual,reads.expected);
    assert.equal(T.diagnostics(actual).wasmCalls,0);assert.equal(T.diagnostics(actual).fallbackCalls,1);
  }
});

test('unknown and proxied prototype chains are never inspected speculatively',()=>{
  for(const T of fixtures)for(const proxy of [false,true]) {
    const [actual,expected]=pair(T),actualTrace=[],expectedTrace=[];
    const parent=(material,trace)=>{
      const target=Object.getPrototypeOf(material);
      if(!proxy)return Object.create(target);
      return new Proxy(target,{
        get(object,key,receiver){trace.push('get:'+String(key));return Reflect.get(object,key,receiver);},
        getOwnPropertyDescriptor(object,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(object,key);},
        getPrototypeOf(object){trace.push('prototype');return Reflect.getPrototypeOf(object);}
      });
    };
    Object.setPrototypeOf(actual.material,parent(actual.material,actualTrace));
    Object.setPrototypeOf(expected.material,parent(expected.material,expectedTrace));
    update(actual,expected);assert.deepEqual(actualTrace,expectedTrace);
    assert.equal(actualTrace.some(event=>event.startsWith('descriptor:')||event==='prototype'),false);
    assert.equal(T.diagnostics(actual).wasmCalls,0);assert.equal(T.diagnostics(actual).fallbackCalls,1);
  }
});

test('late WeakSet mutation cannot forge the instance or prototype witness',()=>{
  const T=fixtures[1],[actual,expected]=pair(T),actualTrace=[],expectedTrace=[];
  const wrap=(material,trace)=>new Proxy(material,{
    get(object,key,receiver){trace.push('get:'+String(key));return Reflect.get(object,key,receiver);},
    getOwnPropertyDescriptor(object,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(object,key);},
    getPrototypeOf(object){trace.push('prototype');return Reflect.getPrototypeOf(object);}
  });
  actual.material=wrap(actual.material,actualTrace);expected.material=wrap(expected.material,expectedTrace);
  seed(actual);seed(expected);let calls=0;
  withProperty(WeakSet.prototype,'has',{value(){calls++;return true;},writable:true},()=>{
    actual.update();expected.update();
  });
  equal(actual,expected);assert.equal(calls,0);assert.deepEqual(actualTrace,expectedTrace);
  assert.equal(actualTrace.some(event=>event.startsWith('descriptor:')||event==='prototype'),false);
  assert.equal(T.diagnostics(actual).wasmCalls,0);assert.equal(T.diagnostics(actual).fallbackCalls,1);
});

test('changed source material definitions keep the original execution path',async()=>{
  for(const name of ['Material','ShaderMaterial']) {
    const source=fs.readFileSync(path.join(upstream,'src/materials',name+'.js'),'utf8');
    assert.equal(specializeMarchingCubesBase(source).changed,true);
    assert.equal(specializeMarchingCubesBase(source+'\n// changed').changed,false);
  }
  const T=await compiled(false,{changeMaterial:true}),[actual,expected]=pair(T);
  update(actual,expected);
  assert.equal(T.diagnostics(actual).wasmCalls,0);assert.equal(T.diagnostics(actual).fallbackCalls,1);
});

test('resolution changes publish current arrays, draw ranges and OBJ exports after shader updates',()=>{
  const exporter=new OBJExporter();
  for(const T of fixtures) {
    const [actual,expected]=pair(T);
    actual.name=expected.name='h2-current-state';
    for(const effect of [actual,expected]){effect.position.set(2,-3,4);effect.scale.set(3,2,-1);}
    for(const resolution of [8,14,9]) {
      for(const effect of [actual,expected]){effect.init(resolution);effect.isolation=70+resolution;}
      update(actual,expected);
      actual.updateMatrixWorld(true);expected.updateMatrixWorld(true);
      assert.deepEqual(actual.matrixWorld.elements,expected.matrixWorld.elements);
      assert.equal(exporter.parse(actual),exporter.parse(expected));
    }
    assert.equal(T.diagnostics(actual).wasmCalls,3);assert.equal(T.diagnostics(actual).fallbackCalls,0);
  }
});
