import assert from 'node:assert/strict';
import test from 'node:test';
import {compileMarchingCubesKernel, MARCHING_CUBES_PARAMETERS} from './marching_cubes_compile.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {loadOracle, sameEffect, sameArrays, cubeField, outputNames} from './fixtures/marching_cubes_oracle.mjs';

const oracle = await loadOracle();
const artifact = compileMarchingCubesKernel();
const empty = new Float32Array(0);
const args = effect => [effect.field,effect.normal_cache,effect.palette,effect.positionArray,effect.normalArray,
  effect.enableUvs ? effect.uvArray : empty,effect.enableColors ? effect.colorArray : empty,
  ...effect.__f3dLists,oracle.edgeTable,oracle.triTable,
  effect.size,effect.size2,effect.halfsize,effect.delta,effect.yd,effect.zd,effect.isolation,
  Number(effect.material.flatShading===true),Number(effect.enableUvs),Number(effect.enableColors)];
const pair = (size=4,flags=7,capacity=100) => Array.from({length:2},()=>new oracle.MarchingCubes(
  size,{flatShading:!!(flags&1)},!!(flags&2),!!(flags&4),capacity));
const update = (kernel, actual, expected) => {
  expected.update();
  actual.count=kernel.run(...args(actual));
  sameEffect(actual,expected);
};
const arrays = effect => [...outputNames.flatMap(name=>effect[name]?[effect[name]]:[]),...effect.__f3dLists];
const snapshot = effect => arrays(effect).map(value=>value.slice());
const unchanged = (effect, before) => arrays(effect).forEach((value,index)=>sameArrays(value,before[index]));

for(let flags=0;flags<8;flags++)test(`all 256 cube cases: flat/UV/color flags ${flags}`,()=>{
  const [actual,expected]=pair(4,flags,7), kernel=instantiateNumericKernel(artifact.wasm);
  const identities=arrays(actual);
  for(let bits=0;bits<256;bits++) {
    for(const effect of [actual,expected]) {
      effect.reset();cubeField(effect,bits);
      // Dirty tails must not be cleared, even when a smaller case follows.
      effect.positionArray.fill(-0);effect.normalArray.fill(123.5);
    }
    update(kernel,actual,expected);
    assert.ok(actual.count<=15);
  }
  arrays(actual).forEach((array,i)=>assert.equal(array,identities[i]));
  assert.equal(kernel.diagnostics.wasmCalls,256);assert.equal(kernel.diagnostics.fallbackCalls,0);
});

for(const size of [4,9,18,32])test(`animated metaballs, all planes, blur, cache reuse and re-init: ${size}`,()=>{
  const [actual,expected]=pair(size,7,50000),kernel=instantiateNumericKernel(artifact.wasm);
  for(let frame=0;frame<5;frame++) {
    for(const effect of [actual,expected]) {
      if(frame!==2)effect.reset();
      effect.addBall(0.43+frame/47,0.51,0.47,0.8,12,[0.2,0.4,0.7]);
      effect.addBall(0.61,0.4,0.55,-0.17,8,[0.3,0.1,0.9]);
      effect.addPlaneX(0.12,5);effect.addPlaneY(0.17,7);effect.addPlaneZ(0.11,9);
      if(frame===3)effect.blur(0.375);
      effect.material.flatShading=frame%2===0;
      effect.enableUvs=frame!==1;effect.enableColors=frame!==2;
      if(frame===2)effect.setCell(2,2,2,91.25); // No cache reset: preserve upstream stale normals.
    }
    update(kernel,actual,expected);
  }
  for(const effect of [actual,expected]){effect.enableUvs=true;effect.enableColors=true;effect.init(6);cubeField(effect,69);}
  update(kernel,actual,expected);
  assert.equal(kernel.diagnostics.wasmCalls,6);
});

test('seeded fields retain triangle order, degeneracies and Float32 rounding across 80 updates',()=>{
  const kernel=instantiateNumericKernel(artifact.wasm),[actual,expected]=pair(7,7,2000);
  let seed=0x23bd719;
  for(let frame=0;frame<80;frame++) {
    if(frame%3===0){actual.reset();expected.reset();}
    for(let i=0;i<actual.field.length;i++) {
      seed=(Math.imul(seed,1664525)+1013904223)>>>0;
      actual.field[i]=expected.field[i]=(seed%1024-512)/71;
    }
    for(let i=0;i<actual.palette.length;i++)actual.palette[i]=expected.palette[i]=(i%13-6)/17;
    actual.isolation=expected.isolation=(frame%7-3)/5;
    actual.material.flatShading=expected.material.flatShading=frame%2===0;
    update(kernel,actual,expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls,80);
});

for(const size of [1,2,3,4])test(`empty/NaN-isolation field size ${size} leaves tails and edge caches untouched`,()=>{
  const [actual,expected]=pair(size,7,4),kernel=instantiateNumericKernel(artifact.wasm);
  for(const effect of [actual,expected]) {
    for(const a of arrays(effect))a.fill(-0);
    effect.isolation=NaN;
    effect.__f3dLists[0][0]=19;
  }
  update(kernel,actual,expected);
  assert.equal(actual.count,0);assert.equal(actual.__f3dLists[0][0],19);
  assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('NaN, infinities, tiny values, signed zeros and equal-to-isolation corners match upstream',()=>{
  const kernel=instantiateNumericKernel(artifact.wasm);
  for(const value of [NaN,Infinity,-Infinity,-0,0,1e-45,3.4028234663852886e38]) {
    const [actual,expected]=pair(4,7,7);
    for(const effect of [actual,expected]) {
      cubeField(effect,89);effect.field[21]=value;
      effect.palette[63]=value;effect.normal_cache[63]=value;
    }
    update(kernel,actual,expected);
  }
});

test('output exhaustion aborts ALL scratch writes; original overflow warning/count and later recovery survive',()=>{
  const kernel=instantiateNumericKernel(artifact.wasm),[actual,expected]=pair(4,7,1);
  for(const effect of [actual,expected])cubeField(effect,61);
  const before=snapshot(actual);
  assert.throws(()=>kernel.run(...args(actual)),{code:'KERNEL_EXECUTION_FAILED'});
  unchanged(actual,before);assert.equal(kernel.diagnostics.wasmCalls,0);
  const warn=console.warn,warnings=[];
  try{console.warn=message=>warnings.push(message);actual.update();expected.update();}
  finally{console.warn=warn;}
  sameEffect(actual,expected);assert.ok(actual.count>3);assert.equal(warnings.length,2);
  assert.equal(warnings[0],warnings[1]);
  for(const effect of [actual,expected]){effect.reset();cubeField(effect,1);}
  update(kernel,actual,expected);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('fuel exhaustion and memory budgets fail before publication, not after partial geometry',()=>{
  const [effect]=pair(8,7,2000);cubeField(effect,69);
  for(const options of [{maxIterations:4},{maxMemoryPages:1}]) {
    const k=instantiateNumericKernel(compileMarchingCubesKernel(options).wasm),before=snapshot(effect);
    assert.throws(()=>k.run(...args(effect)));
    unchanged(effect,before);assert.equal(k.diagnostics.wasmCalls,0);
  }
});

test('aliased output storage is refused because upstream cross-attribute store order is observable',()=>{
  const [effect]=pair();cubeField(effect,1);
  effect.normalArray=effect.positionArray;
  const k=instantiateNumericKernel(artifact.wasm),before=snapshot(effect);
  assert.throws(()=>k.run(...args(effect)),{code:'KERNEL_ARRAY_ALIAS'});unchanged(effect,before);
});

test('live table edits and original edge-list history are consumed, not baked into the module',()=>{
  const k=instantiateNumericKernel(artifact.wasm),[actual,expected]=pair();
  for(const effect of [actual,expected])cubeField(effect,1);
  update(k,actual,expected);
  const saved=oracle.edgeTable[1],savedEntry=oracle.triTable[16];
  try {
    oracle.edgeTable[1]=2;oracle.triTable[16]=0; // Edge zero is stale but still referenced.
    for(const effect of [actual,expected]){effect.field[21]=-13;effect.palette[63]=7;}
    update(k,actual,expected);
    oracle.triTable[16]=14; // JS reads undefined and stores NaN; Wasm must abort, not approximate.
    const before=snapshot(actual);
    assert.throws(()=>k.run(...args(actual)),{code:'KERNEL_EXECUTION_FAILED'});unchanged(actual,before);
    actual.update();expected.update();sameEffect(actual,expected);
  }finally{oracle.edgeTable[1]=saved;oracle.triTable[16]=savedEntry;}
});

test('deterministic bounded artifact uses the existing host ABI and no host imports',()=>{
  assert.deepEqual(artifact.wasm,compileMarchingCubesKernel().wasm);
  assert.equal(WebAssembly.validate(artifact.wasm),true);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)),[]);
  assert.deepEqual(artifact.manifest.parameters.map(p=>p.type),MARCHING_CUBES_PARAMETERS);
  assert.equal(artifact.manifest.version,9);assert.equal(artifact.manifest.resultType,'f64');
  assert.equal(artifact.manifest.automaticRouteAdmission,false);
  assert.equal(artifact.manifest.loopCount,9);assert.equal(artifact.manifest.maxLoopDepth,5);
  assert.equal(artifact.manifest.mathIntrinsics,undefined);
});
