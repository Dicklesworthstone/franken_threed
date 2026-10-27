import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeTransformBatch} from './transform_batch_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';
import {BundleGroup as ReferenceBundleGroup} from '../../upstream/three.js/build/three.webgpu.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
async function fixture(options={}) {
  const entry='\0transform-batch-fixture',plugin=marchingCubesRollupPlugin(options);
  const source='export {Object3D,Group,Scene,Bone,Mesh,InstancedMesh,Matrix4,Euler,Vector3,Quaternion,BufferGeometry,MeshBasicMaterial} from '+
    JSON.stringify(path.join(upstream,'build/three.core.js'))+';export {BundleGroup} from '+
    JSON.stringify(path.join(upstream,'build/three.webgpu.js'))+';'+
    'export {transformBatchDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-transform-batch-adapter')+';'+
    'export {matrixMultiplyDiagnostics as matrixDiagnostics} from '+JSON.stringify('\0f3d-matrix-multiply-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'transform-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?source:null;}},plugin]});
  try{
    const result=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64')),
      report:plugin.api.getTransformBatchesReport()};
  }finally{await bundle.close();}
}
let native,limited;
before(async()=>{native=await fixture();limited=await fixture({maxIterations:1});});
function fill(matrix,seed=0){for(let i=0;i<16;i++)matrix.elements[i]=((i*5+seed)%23-11)/7;return matrix;}
function group(api,count=3,Bundle=api.Group){
  const parent=new Bundle(),geometry=new api.BufferGeometry(),material=new api.MeshBasicMaterial();
  for(let i=0;i<count;i++){
    const child=new api.Mesh(geometry,material);
    child.position.set(i/7,2-i/9,-1);child.rotation.set(i/11,0.7,-0.2);child.scale.set(-2,0.3+i/100,1.2);
    parent.add(child);
  }
  return parent;
}
function same(actual,expected){
  assert.deepEqual(actual.matrix.elements,expected.matrix.elements);
  assert.deepEqual(actual.matrixWorld.elements,expected.matrixWorld.elements);
  assert.equal(actual.matrixWorldNeedsUpdate,expected.matrixWorldNeedsUpdate);
  assert.equal(actual.children.length,expected.children.length);
  for(let i=0;i<actual.children.length;i++)same(actual.children[i],expected.children[i]);
}
function configureManual(parent){
  parent.matrixAutoUpdate=false;parent.matrixWorldNeedsUpdate=false;fill(parent.matrix,4);fill(parent.matrixWorld,8);
}
function subtree(api){
  const root=new api.Scene(),left=new api.Group(),right=new api.Object3D();
  left.add(new api.Mesh(new api.BufferGeometry(),new api.MeshBasicMaterial()),new api.Bone());
  left.children[1].add(new api.Object3D());right.add(new api.Group());root.add(left,right);
  let index=0;
  root.traverse(node=>{
    const i=index++;node.position.set(i/3,1-i/7,-2);node.rotation.set(i/11,0.7,-i/13);
    node.scale.set(i===3?0:-2,0.4+i/9,1.3);
  });
  return root;
}

test('only exact pinned built-core source admits the source-derived child transaction',()=>{
  const source=fs.readFileSync(path.join(upstream,'build/three.core.js'),'utf8');
  assert.equal(specializeTransformBatch(source+'\n// changed').changed,false);
  assert.equal(specializeTransformBatch(fs.readFileSync(path.join(upstream,'src/core/Object3D.js'),'utf8')).changed,false);
  assert.equal(native.report.compiledModules,1);assert.equal(native.report.accelerationClaim,false);
  assert.equal(native.report.modules[0].sourceDerived,true);
  assert.equal(native.report.modules[0].scope,'guarded-ordinary-subtrees');
  assert.ok(native.report.modules[0].wasmBytes>0);
});

test('nested Groups, Meshes and Bones propagate current parent matrices in one source-derived transaction',()=>{
  const {api}=native,actual=subtree(api),expected=subtree(reference),identities=[];
  actual.traverse(node=>identities.push([node,node.matrix,node.matrix.elements,node.matrixWorld,node.matrixWorld.elements,
    node.position,node.quaternion,node.scale]));
  for(let frame=0;frame<3;frame++){
    for(const root of [actual,expected]){
      root.children[0].rotation.y+=0.2;root.children[0].children[1].position.x-=0.3;
      root.updateMatrixWorld();
    }
    same(actual,expected);
    assert.deepEqual(api.diagnostics(actual),{wasmCalls:frame+1,objects:6*(frame+1),fallbackCalls:0,lastFailure:null});
  }
  for(const [node,matrix,elements,world,worldElements,position,quaternion,scale]of identities){
    assert.equal(node.matrix,matrix);assert.equal(matrix.elements,elements);
    assert.equal(node.matrixWorld,world);assert.equal(world.elements,worldElements);
    assert.equal(node.position,position);assert.equal(node.quaternion,quaternion);assert.equal(node.scale,scale);
    if(node!==actual)assert.equal(api.diagnostics(node).wasmCalls,0);
  }
});

test('mixed subtree flags keep inherited force local even when an ancestor world matrix is manual',()=>{
  const {api}=native;
  for(let seed=0;seed<24;seed++)for(const force of [undefined,false,true]){
    const actual=subtree(api),expected=subtree(reference);
    for(const root of [actual,expected]){
      configureManual(root);
      let index=0;
      root.traverse(node=>{
        if(node===root)return;
        const flags=(seed+index*5)%8;
        node.matrixAutoUpdate=!!(flags&1);node.matrixWorldAutoUpdate=!!(flags&2);node.matrixWorldNeedsUpdate=!!(flags&4);
        fill(node.matrix,seed+index);fill(node.matrixWorld,seed+index+11);index++;
      });
    }
    actual.updateMatrixWorld(force);expected.updateMatrixWorld(force);same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,1);
  }
  const actual=subtree(api),expected=subtree(reference);
  for(const root of [actual,expected]){
    let index=0;
    root.traverse(node=>{configureManual(node);fill(node.matrixWorld,++index);});
    root.children[0].matrixWorldNeedsUpdate=true;root.children[0].matrixWorldAutoUpdate=false;
  }
  const staleParent=[...actual.children[0].matrixWorld.elements],cleanSibling=[...actual.children[1].matrixWorld.elements];
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.deepEqual(actual.children[0].matrixWorld.elements,staleParent);
  assert.deepEqual(actual.children[1].matrixWorld.elements,cleanSibling);
  assert.equal(api.diagnostics(actual).wasmCalls,1);
});

test('a single-child chain is batched and add/remove/reparent edits rebuild source traversal on the next call',()=>{
  const {api}=native,actual=new api.Group(),expected=new reference.Group();
  for(const [root,types]of [[actual,api],[expected,reference]]){
    let parent=root;
    for(let i=0;i<64;i++){
      const child=new types.Group();child.position.set(i/3,1,-1);child.rotation.y=i/31;
      parent.add(child);parent=child;
    }
  }
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.deepEqual(api.diagnostics(actual),{wasmCalls:1,objects:64,fallbackCalls:0,lastFailure:null});
  const a=subtree(api),b=subtree(reference);
  const operations=[
    (root,types)=>root.children[1].add(new types.Object3D()),
    root=>root.children[1].add(root.children[0].children[0]),
    root=>root.children[0].remove(root.children[0].children[0]),
    (root,types)=>root.children[0].add(new types.Group()),
  ];
  let frames=0,objects=0;
  for(const edit of operations){
    edit(a,api);edit(b,reference);a.updateMatrixWorld();b.updateMatrixWorld();same(a,b);
    a.traverse(node=>{if(node!==a)objects++;});
    assert.deepEqual(api.diagnostics(a),{wasmCalls:++frames,objects,fallbackCalls:0,lastFailure:null});
  }
});

test('H1 BundleGroup with 4000 Mesh children and one InstancedMesh updates all 4001 leaves in one Wasm call per frame',()=>{
  const {api}=native,actual=group(api,4000,api.BundleGroup),expected=group(reference,4000,ReferenceBundleGroup);
  const instanced=new api.InstancedMesh(new api.BufferGeometry(),new api.MeshBasicMaterial(),10);
  const refInstanced=new reference.InstancedMesh(new reference.BufferGeometry(),new reference.MeshBasicMaterial(),10);
  actual.add(instanced);expected.add(refInstanced);
  const identities=actual.children.map(child=>[child.matrix,child.matrix.elements,child.matrixWorld,child.matrixWorld.elements,
    child.position,child.rotation,child.quaternion,child.scale]);
  const matrix=new api.Matrix4(),rotation=new api.Matrix4(),refMatrix=new reference.Matrix4(),refRotation=new reference.Matrix4();
  for(let frame=0;frame<3;frame++){
    for(let i=0;i<4000;i++){
      actual.children[i].rotation.set(frame/31,i/73,-frame/41);
      expected.children[i].rotation.set(frame/31,i/73,-frame/41);
    }
    for(let i=0;i<10;i++){
      rotation.makeRotationFromEuler(new api.Euler(frame/71,(i+1)/100,-i/83));
      refRotation.makeRotationFromEuler(new reference.Euler(frame/71,(i+1)/100,-i/83));
      instanced.getMatrixAt(i,matrix);refInstanced.getMatrixAt(i,refMatrix);
      matrix.multiply(rotation);refMatrix.multiply(refRotation);
      instanced.setMatrixAt(i,matrix);refInstanced.setMatrixAt(i,refMatrix);
    }
    actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
    assert.deepEqual(instanced.instanceMatrix.array,refInstanced.instanceMatrix.array);
    assert.deepEqual(api.diagnostics(actual),{wasmCalls:frame+1,objects:4001*(frame+1),fallbackCalls:0,lastFailure:null});
  }
  for(let i=0;i<actual.children.length;i++){
    const child=actual.children[i];
    const current=[child.matrix,child.matrix.elements,child.matrixWorld,child.matrixWorld.elements,
      child.position,child.rotation,child.quaternion,child.scale];
    for(let j=0;j<current.length;j++)assert.equal(current[j],identities[i][j]);
    assert.equal(api.matrixDiagnostics(child.matrixWorld).wasmCalls,0);
  }
});

test('all 24 manual flags/force combinations retain projective math and sibling-local force',()=>{
  const {api}=native;
  for(const automatic of [false,true])for(const worldAutomatic of [false,true])for(const dirty of [false,true])
    for(const force of [undefined,false,true]){
      const actual=group(api),expected=group(reference);
      for(const parent of [actual,expected]){
        configureManual(parent);
        for(let i=0;i<parent.children.length;i++){
          const child=parent.children[i];child.matrixAutoUpdate=automatic;child.matrixWorldAutoUpdate=worldAutomatic;
          child.matrixWorldNeedsUpdate=dirty;fill(child.matrix,i+2);fill(child.matrixWorld,i+7);
        }
      }
      actual.updateMatrixWorld(force);expected.updateMatrixWorld(force);same(actual,expected);
      assert.equal(api.diagnostics(actual).wasmCalls,1);
    }
  const actual=group(api),expected=group(reference);
  for(const parent of [actual,expected]){
    configureManual(parent);
    for(let i=0;i<parent.children.length;i++){
      const child=parent.children[i];child.matrixAutoUpdate=false;child.matrixWorldNeedsUpdate=i===0;
      fill(child.matrix,i+3);fill(child.matrixWorld,i+12);
    }
  }
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.equal(api.diagnostics(actual).wasmCalls,1);
  for(const parent of [actual,expected])parent.matrixWorldNeedsUpdate=true;
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.equal(api.diagnostics(actual).wasmCalls,2);
});

test('manual read-only channels are not written or probed through unused updateMatrix getters',()=>{
  const {api}=native,actual=group(api),expected=group(reference);
  let callbacks=0;
  for(const parent of [actual,expected]){
    configureManual(parent);
    for(const child of parent.children){
      child.matrixAutoUpdate=false;child.matrixWorldAutoUpdate=false;
      Object.defineProperty(child,'matrixWorldNeedsUpdate',{value:false,writable:false});
      Object.freeze(child.matrix.elements);Object.freeze(child.matrixWorld.elements);
      Object.defineProperty(child,'updateMatrix',{get(){callbacks++;throw new Error('unused getter');}});
    }
  }
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.equal(callbacks,0);assert.equal(api.diagnostics(actual).wasmCalls,1);
});

test('parent effects and authored Euler callbacks retain order, identities and current clone/serialization state',()=>{
  const {api}=native,actual=group(api),expected=group(reference),actualTrace=[],expectedTrace=[];
  for(const [parent,trace]of [[actual,actualTrace],[expected,expectedTrace]]){
    const original=parent.updateMatrix;
    parent.updateMatrix=function(){trace.push('parent');this.children[0].position.x+=2;return original.call(this);};
    for(let i=0;i<parent.children.length;i++){
      const child=parent.children[i],callback=child.rotation._onChangeCallback;
      child.rotation._onChangeCallback=function(){trace.push('euler:'+i);return callback.call(this);};
      child.rotation.set(0.4,i/7,-0.8);
    }
  }
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.deepEqual(actualTrace,expectedTrace);assert.equal(actualTrace.filter(x=>x==='parent').length,1);
  assert.equal(api.diagnostics(actual).wasmCalls,1);
  assert.deepEqual(actual.children[0].clone().matrix.elements,expected.children[0].clone().matrix.elements);
  assert.deepEqual(actual.children[0].toJSON().object.matrix,expected.children[0].toJSON().object.matrix);
});

test('observable methods, getters and proxy allocations retain the complete original child loop',()=>{
  const {api}=native;
  const cases=[
    (p,trace)=>{const child=p.children[0],method=child.updateMatrixWorld;Object.defineProperty(child,'updateMatrixWorld',{get(){trace.push('method');return method;}});},
    (p,trace)=>{const child=p.children[0];child.matrix=new Proxy(child.matrix,{get(target,key,receiver){trace.push('matrix:'+String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    (p,trace)=>{const child=p.children[0];child.matrix.elements=new Proxy(child.matrix.elements,{get(target,key,receiver){trace.push('elements:'+String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    (p,trace)=>{const child=p.children[0],value=child.quaternion._x;Object.defineProperty(child.quaternion,'_x',{get(){trace.push('quat');return value;}});},
    (p,trace)=>{const children=p.children;p.children=new Proxy(children,{get(target,key,receiver){trace.push('children:'+String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    (p,trace)=>{const child=p.children[0],method=child.updateMatrix;child.updateMatrix=function(){trace.push('override');return method.call(this);};}
  ];
  for(const prepare of cases){
    const actual=group(api),expected=group(reference),a=[],b=[];prepare(actual,a);prepare(expected,b);
    actual.updateMatrixWorld();expected.updateMatrixWorld();
    assert.deepEqual(a,b);assert.equal(api.diagnostics(actual).wasmCalls,0);
    for(let i=0;i<actual.children.length;i++)assert.deepEqual(actual.children[i].matrixWorld.elements,expected.children[i].matrixWorld.elements);
  }
});

test('deep observable hooks refuse the ancestor transaction before effects and retain original DFS observations',()=>{
  const {api}=native;
  const cases=[
    (root,trace)=>{
      const child=root.children[0].children[1].children[0],method=child.updateMatrixWorld;
      child.updateMatrixWorld=function(force){
        trace.push(['deep',force,...root.children[0].children[0].matrixWorld.elements]);
        root.children[1].position.x+=2;return method.call(this,force);
      };
    },
    (root,trace)=>{
      const child=root.children[1].children[0],value=child.matrix;
      Object.defineProperty(child,'matrix',{get(){trace.push(['matrix',...root.children[0].matrixWorld.elements]);return value;}});
    },
    (root,trace)=>{
      const branch=root.children[0].children[1],child=branch.children[0];
      branch.children[0]=new Proxy(child,{
        get(target,key,receiver){trace.push('get:'+String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);},
      });
    },
  ];
  for(const prepare of cases){
    const actual=subtree(api),expected=subtree(reference),a=[],b=[];
    prepare(actual,a);prepare(expected,b);actual.updateMatrixWorld();expected.updateMatrixWorld();
    assert.deepEqual(a,b);assert.equal(api.diagnostics(actual).wasmCalls,0);
    same(actual,expected);
  }
});

test('deep and cross-branch aliases, invalid topology and pivots keep the original subtree execution',()=>{
  const {api}=native;
  const cases=[
    root=>{root.children[1].children[0].matrixWorld=root.children[0].children[0].matrixWorld;},
    root=>{root.children[0].children[1].children[0].matrix.elements=root.children[0].matrix.elements;},
    root=>{Object.defineProperty(root.children[1].children[0],'scale',{value:root.children[0].children[0].scale});},
    root=>{root.children[0].children[1].children[0].parent=null;},
    root=>{root.children[1].children[0]=root.children[0].children[0];},
    (root,types)=>{root.children[0].children[1].pivot=new types.Vector3(1,2,-3);},
  ];
  for(const prepare of cases){
    const actual=subtree(api),expected=subtree(reference);prepare(actual,api);prepare(expected,reference);
    actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,0);
  }
});

test('parent/sibling matrix and TRS aliases, duplicate children and wrong parents reject bulk execution',()=>{
  const {api}=native;
  const cases=[
    p=>{p.children[0].matrixWorld=p.children[0].matrix;},
    p=>{p.children[0].matrix.elements=p.children[1].matrix.elements;},
    p=>{p.children[1].matrixWorld.elements=p.children[0].matrixWorld.elements;},
    p=>{p.children[0].matrixWorld=p.matrixWorld;},
    p=>{Object.defineProperty(p.children[1],'position',{value:p.children[0].position});},
    p=>{Object.defineProperty(p.children[1],'quaternion',{value:p.children[0].quaternion});},
    p=>{p.children[1]=p.children[0];},
    p=>{p.children[0].parent=null;}
  ];
  for(const prepare of cases){
    const actual=group(api),expected=group(reference);prepare(actual);prepare(expected);
    actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,0);
  }
});

test('kernel budget failure and read-only output preserve original writes and exception order',()=>{
  const actual=group(limited.api),expected=group(reference);
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.equal(limited.api.diagnostics(actual).wasmCalls,0);assert.equal(limited.api.diagnostics(actual).fallbackCalls,1);
  const broken=group(native.api),refBroken=group(reference);
  for(const parent of [broken,refBroken])Object.defineProperty(parent.children[1].matrix.elements,'5',{writable:false});
  assert.throws(()=>broken.updateMatrixWorld(),TypeError);assert.throws(()=>refBroken.updateMatrixWorld(),TypeError);
  same(broken,refBroken);assert.equal(native.api.diagnostics(broken).wasmCalls,0);
});

test('subtree budget exhaustion and a late unwritable descendant preserve source publication and exceptions',()=>{
  const actual=subtree(limited.api),expected=subtree(reference),trace=[];
  const original=actual.updateMatrix;
  actual.updateMatrix=function(){trace.push('owner');return original.call(this);};
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.deepEqual(trace,['owner']);
  assert.deepEqual(limited.api.diagnostics(actual),{
    wasmCalls:0,objects:0,fallbackCalls:1,lastFailure:'KERNEL_EXECUTION_FAILED'});
  const broken=subtree(native.api),refBroken=subtree(reference);
  for(const root of [broken,refBroken])Object.defineProperty(root.children[1].children[0].matrix.elements,'5',{writable:false});
  assert.throws(()=>broken.updateMatrixWorld(),TypeError);assert.throws(()=>refBroken.updateMatrixWorld(),TypeError);
  same(broken,refBroken);assert.equal(native.api.diagnostics(broken).wasmCalls,0);
});

test('late Error dispatch and inherited error setters cannot run inside a budget-failure fallback',()=>{
  const {api}=limited;
  for(const [target,key]of [[Error,Symbol.hasInstance],[Error.prototype,'code'],[Error.prototype,'name']]){
    const actual=group(api),expected=group(reference),saved=Object.getOwnPropertyDescriptor(target,key);
    let callbacks=0;
    const replacement=target===Error
      ?{configurable:true,value(){callbacks++;Object.defineProperty(actual.children[0].matrix.elements,'5',{writable:false});return false;}}
      :{configurable:true,get(){callbacks++;return 'changed';},set(){callbacks++;throw new Error('unexpected inherited setter');}};
    try{Object.defineProperty(target,key,replacement);actual.updateMatrixWorld();}
    finally{if(saved)Object.defineProperty(target,key,saved);else delete target[key];}
    expected.updateMatrixWorld();same(actual,expected);
    assert.equal(callbacks,0);assert.deepEqual(api.diagnostics(actual),{
      wasmCalls:0,objects:0,fallbackCalls:1,lastFailure:'TRANSFORM_BATCH_PLATFORM'});
  }
});
