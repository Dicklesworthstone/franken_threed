import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeTransformBatch} from './transform_batch_specialization.mjs';
import {specializeMatrixMultiply} from './matrix_multiply_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';
import {BundleGroup as ReferenceBundleGroup} from '../../upstream/three.js/build/three.webgpu.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
async function fixture(options={},withBundleGroup=false) {
  const entry='\0transform-batch-fixture',plugin=marchingCubesRollupPlugin(options);
  const reports=[];
  // Exercise these two production source transforms directly. Unrelated asset,
  // geometry and animation compilers do not belong in every matrix fixture;
  // transform_batch_application.test.mjs still runs the complete production build.
  plugin.transform=(code,id)=>{
    if(!code.includes('class Matrix4'))return null;
    const matrix=specializeMatrixMultiply(code,{...options,runtimeModule:'\0f3d-matrix-multiply-adapter'});
    const batch=specializeTransformBatch(code,{...options,runtimeModule:'\0f3d-transform-batch-adapter'});
    if(!matrix.changed && !batch.changed)return null;
    if(batch.report)reports.push({id,...batch.report});
    let transformed=code;
    for(const edit of [...matrix.edits,...batch.edits].sort((a,b)=>b.start-a.start))
      transformed=transformed.slice(0,edit.start)+edit.text+transformed.slice(edit.end);
    return {code:matrix.header+batch.header+transformed+matrix.tail+batch.tail,map:null};
  };
  const source='export {Object3D,Group,Scene,Bone,Mesh,InstancedMesh,Matrix4,Euler,Vector3,Quaternion,BufferGeometry,MeshBasicMaterial,Camera,SkinnedMesh,Light,AmbientLight,DirectionalLight,HemisphereLight,LightProbe,PointLight,RectAreaLight,SpotLight,Line,LineLoop,LineSegments,Points,Sprite,LOD} from '+
    JSON.stringify(path.join(upstream,'build/three.core.js'))+';'+
    (withBundleGroup?'export {BundleGroup} from '+JSON.stringify(path.join(upstream,'build/three.webgpu.js'))+';':'')+
    'export {transformBatchDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-transform-batch-adapter')+';'+
    'export {matrixMultiplyDiagnostics as matrixDiagnostics} from '+JSON.stringify('\0f3d-matrix-multiply-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'transform-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?source:null;}},plugin]});
  try{
    const result=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64')),
      report:{compiledModules:reports.length,modules:reports,accelerationClaim:false}};
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

test('built-core pin remains exact and source modules require their Matrix4 input',()=>{
  const source=fs.readFileSync(path.join(upstream,'build/three.core.js'),'utf8');
  assert.equal(specializeTransformBatch(source+'\n// changed').changed,false);
  assert.equal(specializeTransformBatch(fs.readFileSync(path.join(upstream,'src/core/Object3D.js'),'utf8')).changed,false);
  assert.equal(native.report.compiledModules,1);assert.equal(native.report.accelerationClaim,false);
  assert.equal(native.report.modules[0].sourceDerived,true);
  assert.equal(native.report.modules[0].scope,'guarded-ordinary-subtrees');
  assert.deepEqual(native.report.modules[0].compiledMethods,['Object3D.updateMatrixWorld','Object3D.updateWorldMatrix']);
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

test('H1 BundleGroup with 4000 Mesh children and one InstancedMesh updates all 4001 leaves in one Wasm call per frame',async()=>{
  const {api}=await fixture({},true),actual=group(api,4000,api.BundleGroup),expected=group(reference,4000,ReferenceBundleGroup);
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

test('observable methods, getters and proxy allocations retain source effects before guarded later spans',()=>{
  const {api}=native;
  const cases=[
    (p,trace)=>{const child=p.children[0],method=child.updateMatrixWorld;Object.defineProperty(child,'updateMatrixWorld',{get(){trace.push('method');return method;}});},
    (p,trace)=>{const child=p.children[0];child.matrix=new Proxy(child.matrix,{get(target,key,receiver){trace.push('matrix:'+String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    (p,trace)=>{const child=p.children[0];child.matrix.elements=new Proxy(child.matrix.elements,{get(target,key,receiver){trace.push('elements:'+String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    (p,trace)=>{const child=p.children[0],value=child.quaternion._x;Object.defineProperty(child.quaternion,'_x',{get(){trace.push('quat');return value;}});},
    (p,trace)=>{const children=p.children;p.children=new Proxy(children,{get(target,key,receiver){trace.push('children:'+String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    (p,trace)=>{const child=p.children[0],method=child.updateMatrix;child.updateMatrix=function(){trace.push('override');return method.call(this);};}
  ];
  for(const [index,prepare]of cases.entries()){
    const actual=group(api),expected=group(reference),a=[],b=[];prepare(actual,a);prepare(expected,b);
    actual.updateMatrixWorld();expected.updateMatrixWorld();
    assert.deepEqual(a,b);
    assert.equal(api.diagnostics(actual).wasmCalls,index===4?0:1);
    assert.equal(api.diagnostics(actual).objects,index===4?0:2);
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
  for(const [index,prepare]of cases.entries()){
    const actual=subtree(api),expected=subtree(reference),a=[],b=[];
    prepare(actual,a);prepare(expected,b);actual.updateMatrixWorld();expected.updateMatrixWorld();
    assert.deepEqual(a,b);assert.equal(api.diagnostics(actual).wasmCalls,1);
    assert.equal(api.diagnostics(actual).objects,index===1?4:2);
    same(actual,expected);
  }
});

test('deep and cross-branch aliases and invalid topology preserve source state across separate spans',()=>{
  const {api}=native;
  const cases=[
    root=>{root.children[1].children[0].matrixWorld=root.children[0].children[0].matrixWorld;},
    root=>{root.children[0].children[1].children[0].matrix.elements=root.children[0].matrix.elements;},
    root=>{Object.defineProperty(root.children[1].children[0],'scale',{value:root.children[0].children[0].scale});},
    root=>{root.children[0].children[1].children[0].parent=null;},
    root=>{root.children[1].children[0]=root.children[0].children[0];},
  ];
  for(const [index,prepare]of cases.entries()){
    const actual=subtree(api),expected=subtree(reference);prepare(actual,api);prepare(expected,reference);
    actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,[2,1,2,1,1][index]);
    assert.equal(api.diagnostics(actual).objects,[6,2,6,2,4][index]);
  }
});

test('parent/sibling aliases, duplicate children and wrong parents stay outside the same transaction',()=>{
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
    assert.equal(api.diagnostics(actual).wasmCalls,1);
    assert.equal(api.diagnostics(actual).objects,2);
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
  same(broken,refBroken);assert.equal(native.api.diagnostics(broken).wasmCalls,1);
  assert.equal(native.api.diagnostics(broken).objects,4);
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

const updateSubtree=(root,method,force)=>method==='updateWorldMatrix'
  ?root.updateWorldMatrix(false,true,force):root.updateMatrixWorld(force);

test('both public subtree entry points apply live pivots with exact r186 arithmetic and stable public identities',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix']){
    const actual=subtree(api),expected=subtree(reference),identities=[];
    for(const [root,types]of [[actual,api],[expected,reference]]){
      let i=0;root.traverse(node=>{node.pivot=new types.Vector3(i/7,-i/11,++i/5);});
    }
    actual.traverse(node=>identities.push([node,node.matrix,node.matrix.elements,node.matrixWorld,node.matrixWorld.elements,node.pivot]));
    for(let frame=0;frame<4;frame++){
      for(const [root,types]of [[actual,api],[expected,reference]]){
        root.children[0].rotation.z+=0.3;
        root.children[0].pivot.y-=1;
        root.children[1].pivot=frame%2?new types.Vector3(-3,2,0):null;
        assert.equal(updateSubtree(root,method),undefined);
      }
      same(actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,frame+1);
      assert.deepEqual(actual.children[0].toJSON().object.matrix,expected.children[0].toJSON().object.matrix);
      assert.deepEqual(actual.children[0].toJSON().object.pivot,expected.children[0].toJSON().object.pivot);
      assert.deepEqual(actual.children[0].clone().matrix.elements,expected.children[0].clone().matrix.elements);
      assert.deepEqual(actual.children[0].getWorldPosition(new api.Vector3()).toArray(),
        expected.children[0].getWorldPosition(new reference.Vector3()).toArray());
    }
    for(const [node,matrix,local,world,elements,pivot]of identities){
      assert.equal(node.matrix,matrix);assert.equal(matrix.elements,local);
      assert.equal(node.matrixWorld,world);assert.equal(world.elements,elements);
      if(node!==actual.children[1])assert.equal(node.pivot,pivot);
    }
  }
});

test('pivot arithmetic preserves signed zeros and exceptional Numbers without adding unused pivot reads',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const value of [-0,0,NaN,Infinity,-Infinity,1e200]){
    const actual=group(api),expected=group(reference);
    for(const [root,types]of [[actual,api],[expected,reference]]){
      root.children[0].pivot=new types.Vector3(value,0,-0);
      root.children[1].pivot=new types.Vector3(-2,3,4);root.children[1].scale.set(-1,0,2);
      root.children[2].matrixAutoUpdate=false;
      Object.defineProperty(root.children[2],'pivot',{get(){throw new Error('unused pivot');}});
      updateSubtree(root,method,true);
    }
    same(actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,1);
  }
});

test('selective updates retain ancestor and owner effects once, without touching unrequested siblings',()=>{
  const {api}=native;
  for(const updateParents of [false,true,1])for(const updateChildren of [false,true,1])for(const force of [false,true]){
    const actual=subtree(api),expected=subtree(reference),traces=[[],[]];
    for(const [index,root]of [actual,expected].entries()){
      const branch=root.children[0],updateParent=root.updateMatrix,updateBranch=branch.updateMatrix;
      root.updateMatrix=function(){traces[index].push('parent');branch.position.x+=2;return updateParent.call(this);};
      branch.updateMatrix=function(){traces[index].push('branch');branch.children[1].position.y-=3;return updateBranch.call(this);};
      branch.updateWorldMatrix(updateParents,updateChildren,force);
    }
    same(actual,expected);assert.deepEqual(traces[0],traces[1]);
    assert.deepEqual(traces[0],updateParents===true?['parent','branch']:['branch']);
    assert.equal(api.diagnostics(actual).wasmCalls,0);
    assert.equal(api.diagnostics(actual.children[0]).wasmCalls,updateChildren===true?1:0);
    assert.deepEqual(actual.children[1].matrixWorld.elements,new api.Matrix4().elements);
  }
});

test('selective updates preserve per-branch force, manual matrices and changing parent topology',()=>{
  const {api}=native;
  for(let seed=0;seed<8;seed++)for(const force of [undefined,false,true]){
    const actual=subtree(api),expected=subtree(reference);
    for(const [root,types]of [[actual,api],[expected,reference]]){
      configureManual(root);let i=0;
      root.traverse(node=>{
        if(node===root)return;const flags=(seed+i++)%8;
        node.matrixAutoUpdate=!!(flags&1);node.matrixWorldAutoUpdate=!!(flags&2);node.matrixWorldNeedsUpdate=!!(flags&4);
        node.pivot=new types.Vector3(i,-i/2,i/3);fill(node.matrix,i);fill(node.matrixWorld,11+i);
      });
      root.updateWorldMatrix(false,true,force);
    }
    same(actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,1);
    for(const root of [actual,expected]){
      root.children[1].add(root.children[0].children[0]);
      root.children[1].rotation.x+=0.2;root.updateWorldMatrix(true,true,force);
    }
    same(actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,2);
  }
});

test('selective descendant methods and pivot hooks fall back before observable source effects',()=>{
  const {api}=native;
  const cases=[
    (root,types,trace)=>{const child=root.children[0],method=child.updateWorldMatrix;
      Object.defineProperty(child,'updateWorldMatrix',{get(){trace.push('method');return method;}});},
    (root,types,trace)=>{const child=root.children[0].children[1],method=child.updateWorldMatrix;
      child.updateWorldMatrix=function(...args){trace.push(['override',...args]);root.children[1].position.x++;return method.apply(this,args);};},
    (root,types,trace)=>{const child=root.children[0],pivot=new types.Vector3(1,2,3);
      Object.defineProperty(child,'pivot',{get(){trace.push('pivot');return pivot;}});},
    (root,types,trace)=>{const pivot=new types.Vector3(1,2,3);Object.defineProperty(pivot,'x',{get(){trace.push('x');return 2;}});
      root.children[1].pivot=pivot;},
    (root,types,trace)=>{root.children[1].pivot=new Proxy(new types.Vector3(1,2,3),{
      get(target,key,receiver){trace.push(String(key));return Reflect.get(target,key,receiver);},
      getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}});},
    root=>{root.children[0].pivot=root.children[1].position;},
    root=>{root.children[0].children[0].matrix.elements=root.children[1].matrix.elements;},
  ];
  for(const [index,prepare]of cases.entries()){
    const actual=subtree(api),expected=subtree(reference),traces=[[],[]];
    prepare(actual,api,traces[0]);prepare(expected,reference,traces[1]);
    actual.updateWorldMatrix(true,true);expected.updateWorldMatrix(true,true);
    assert.deepEqual(traces[0],traces[1]);same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,index<5?1:2);
    assert.equal(api.diagnostics(actual).objects,[2,2,2,4,4,6,6][index]);
  }
  // Only the selected traversal method matters. An unused hook is not read.
  for(const method of ['updateMatrixWorld','updateWorldMatrix']){
    const actual=subtree(api),expected=subtree(reference),unused=method==='updateMatrixWorld'?'updateWorldMatrix':'updateMatrixWorld';
    for(const root of [actual,expected])Object.defineProperty(root.children[0],unused,{get(){throw new Error('unused traversal');}});
    updateSubtree(actual,method);updateSubtree(expected,method);same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,1);
  }
});

test('selective budget and late pivot failures preserve partial source updates with no replay',()=>{
  const actual=subtree(limited.api),expected=subtree(reference);
  actual.updateWorldMatrix(true,true);expected.updateWorldMatrix(true,true);same(actual,expected);
  assert.equal(limited.api.diagnostics(actual).wasmCalls,0);
  assert.equal(limited.api.diagnostics(actual).lastFailure,'KERNEL_EXECUTION_FAILED');
  for(const failure of ['pivot','matrix']){
    const broken=subtree(native.api),referenceBroken=subtree(reference),traces=[[],[]];
    for(const [index,root]of [broken,referenceBroken].entries()){
      const original=root.updateMatrix;
      root.updateMatrix=function(){traces[index].push('owner');return original.call(this);};
      const child=root.children[1].children[0];
      if(failure==='pivot')Object.defineProperty(child,'pivot',{get(){traces[index].push('pivot');throw new Error('pivot failure');}});
      else Object.defineProperty(child.matrix.elements,'5',{writable:false});
      assert.throws(()=>root.updateWorldMatrix(true,true),failure==='pivot'?/pivot failure/:TypeError);
    }
    assert.deepEqual(traces[0],traces[1]);same(broken,referenceBroken);assert.equal(native.api.diagnostics(broken).wasmCalls,1);
    assert.equal(native.api.diagnostics(broken).objects,4);
  }
});

test('ordinary lights, primitives and LOD retain one whole-scene transform transaction',()=>{
  const {api}=native;
  const kinds=['Light','AmbientLight','DirectionalLight','HemisphereLight','LightProbe','PointLight','RectAreaLight','SpotLight',
    'Line','LineLoop','LineSegments','Points','Sprite','LOD'];
  for(const kind of kinds)for(const method of ['updateMatrixWorld','updateWorldMatrix']) {
    const make=library=>{
      const scene=new library.Scene(),child=new library[kind](),leaf=new library.Object3D();
      scene.add(new library.Mesh(new library.BufferGeometry(),new library.MeshBasicMaterial()),child);child.add(leaf);
      let i=0;scene.traverse(node=>{
        node.position.set(++i/7,-i/11,2);node.rotation.set(i/13,-0.4,0.2);node.scale.set(-1,0.7,2);
        node.pivot=new library.Vector3(0.4,1,-0.3);
      });
      leaf.matrixAutoUpdate=false;leaf.matrix.makeTranslation(2,-1,3);
      return scene;
    };
    const actual=make(api),expected=make(reference),elements=actual.children[1].matrixWorld.elements;
    for(let frame=0;frame<3;frame++) {
      for(const root of [actual,expected]) {
        root.children[1].position.z+=0.2;root.children[1].pivot.x-=0.1;
        if(method==='updateMatrixWorld')root.updateMatrixWorld(frame===1);
        else root.updateWorldMatrix(false,true,frame===1);
      }
      same(actual,expected);
      assert.equal(api.diagnostics(actual).wasmCalls,frame+1,kind+' '+method);
      assert.equal(api.diagnostics(actual).fallbackCalls,0,kind+' '+method);
      assert.equal(actual.children[1].matrixWorld.elements,elements);
    }
  }
});

test('changed light methods and opaque primitive prototypes retain source effects once',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const failure of ['override','opaque']) {
    const traces=[[],[]],roots=[];
    for(const [index,library]of [api,reference].entries()) {
      const scene=new library.Scene(),child=failure==='override'?new library.PointLight():new library.LineSegments();
      child.position.set(3,-2,1);scene.add(new library.Object3D(),child);roots.push(scene);
      if(failure==='override') {
        const original=child[method];
        Object.defineProperty(child,method,{get(){
          traces[index].push('method');return function(...args){
            traces[index].push('call');return original.apply(this,args);
          };
        }});
      }else {
        const parent=Object.getPrototypeOf(child);
        Object.setPrototypeOf(child,new Proxy(parent,{
          get(target,key,receiver){traces[index].push(String(key));return Reflect.get(target,key,receiver);},
          getOwnPropertyDescriptor(){throw new Error('Unknown prototype must not be inspected');},
          getPrototypeOf(){throw new Error('Unknown prototype must not be traversed');},
        }));
      }
      if(method==='updateMatrixWorld')scene.updateMatrixWorld();else scene.updateWorldMatrix(false,true);
    }
    same(roots[0],roots[1]);assert.deepEqual(traces[0],traces[1]);
    if(failure==='override')assert.deepEqual(traces[0],['method','call']);
    assert.equal(api.diagnostics(roots[0]).wasmCalls,0);
    assert.equal(api.diagnostics(roots[0]).lastFailure,'TRANSFORM_BATCH_GUARD');
  }
});

test('camera and skinned-mesh world-update side effects stay on their original route',()=>{
  const {api}=native;
  for(const kind of ['Camera','SkinnedMesh'])for(const method of ['updateMatrixWorld','updateWorldMatrix']) {
    const roots=[api,reference].map(library=>{
      const scene=new library.Scene(),child=new library[kind]();
      child.position.set(3,-2,1);child.rotation.set(0.3,-0.4,0.1);
      scene.add(new library.Object3D(),child);
      if(method==='updateMatrixWorld')scene.updateMatrixWorld();else scene.updateWorldMatrix(false,true);
      return scene;
    });
    same(roots[0],roots[1]);assert.equal(api.diagnostics(roots[0]).wasmCalls,0);
    assert.equal(api.diagnostics(roots[0]).lastFailure,'TRANSFORM_BATCH_GUARD');
    const field=kind==='Camera'?'matrixWorldInverse':'bindMatrixInverse';
    assert.deepEqual(roots[0].children[1][field].elements,roots[1].children[1][field].elements);
  }
});

function segmentedTree(library,kind='Object3D',count=4){
  const root=new library.Scene(),before=[],after=[],barrier=new library[kind]();
  for(const list of [before,after])for(let i=0;i<count;i++){
    const node=new library.Object3D();node.position.set(i/3+1,-i/7,2);
    node.rotation.set(i/11,-0.2,0.4);node.scale.set(-1,2,0.5);list.push(node);
  }
  barrier.position.set(2,-3,1);barrier.scale.set(1.5,2,0.7);
  root.add(...before,barrier,...after);
  return {root,before,barrier,after,captured:root.children};
}

test('segmented batches keep large ordinary spans around Camera and SkinnedMesh source updates',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const kind of ['Camera','SkinnedMesh']){
    const actual=segmentedTree(api,kind,128),expected=segmentedTree(reference,kind,128);
    for(const [scene,library]of [[actual,api],[expected,reference]])
      scene.barrier.add(new library.Object3D(),new library.Object3D());
    const identities=actual.captured.map(node=>[node,node.matrix,node.matrix.elements,node.matrixWorld,node.matrixWorld.elements]);
    for(let frame=0;frame<2;frame++){
      for(const scene of [actual,expected]){
        scene.root.position.x+=3;scene.barrier.rotation.y+=0.2;
        updateSubtree(scene.root,method);
      }
      same(actual.root,expected.root);
      const field=kind==='Camera'?'matrixWorldInverse':'bindMatrixInverse';
      assert.deepEqual(actual.barrier[field].elements,expected.barrier[field].elements);
      assert.equal(api.diagnostics(actual.root).wasmCalls,2*(frame+1));
      assert.equal(api.diagnostics(actual.root).objects,256*(frame+1));
      assert.equal(api.diagnostics(actual.barrier).wasmCalls,frame+1,'Native post-super work follows its batched children');
    }
    for(const [node,matrix,local,world,elements]of identities){
      assert.equal(node.matrix,matrix);assert.equal(matrix.elements,local);
      assert.equal(node.matrixWorld,world);assert.equal(world.elements,elements);
    }
  }
});

test('segmented batches materialize before hooks and reread later transforms, parent matrices and topology',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix']){
    const scenes=[segmentedTree(api),segmentedTree(reference)],traces=[[],[]];
    for(const [index,scene]of scenes.entries()){
      const library=index===0?api:reference,{root,before,barrier,after}=scene;
      after[0].add(new library.Object3D());after[1].add(new library.Object3D());
      const original=barrier[method];
      barrier[method]=function(...args){
        traces[index].push([args,before.map(node=>node.matrixWorld.elements.slice()),before.map(node=>node.matrixWorldNeedsUpdate)]);
        root.matrixWorld.elements[12]+=50;root.matrixWorldNeedsUpdate=true;
        after[0].position.x+=7;after[0].add(after[1].children[0]);
        [root.children[5],root.children[6]]=[root.children[6],root.children[5]];
        const replacement=new library.Object3D();replacement.position.y=9;replacement.parent=root;root.children[7]=replacement;
        return original.apply(this,args);
      };
      updateSubtree(root,method);
    }
    same(scenes[0].root,scenes[1].root);assert.deepEqual(traces[0],traces[1]);
    assert.equal(traces[0].length,1);assert.equal(scenes[0].root.matrixWorldNeedsUpdate,true);
    assert.equal(api.diagnostics(scenes[0].root).wasmCalls,2);
    assert.equal(api.diagnostics(scenes[0].root).objects,10);
  }
});

test('segmented batches preserve the captured child array and count across callback mutations',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const change of ['append','shrink','replace-array','index-getter']){
    const scenes=[segmentedTree(api),segmentedTree(reference)],traces=[[],[]];
    for(const [index,scene]of scenes.entries()){
      const library=index===0?api:reference,{root,barrier,after,captured}=scene,original=barrier[method];
      barrier[method]=function(...args){
        traces[index].push('hook');
        if(change==='append'){
          const added=new library.Object3D();added.position.x=100;root.add(added);scene.added=added;
        }else if(change==='shrink')root.remove(after[3]);
        else if(change==='replace-array')root.children=[];
        else Object.defineProperty(captured,'5',{configurable:true,get(){traces[index].push('index');return after[0];}});
        return original.apply(this,args);
      };
      if(change==='shrink')assert.throws(()=>updateSubtree(root,method),TypeError);
      else updateSubtree(root,method);
    }
    same(scenes[0].root,scenes[1].root);assert.deepEqual(traces[0],traces[1]);
    for(let i=0;i<scenes[0].captured.length;i++)same(scenes[0].captured[i],scenes[1].captured[i]);
    assert.equal(traces[0].filter(value=>value==='hook').length,1);
    assert.equal(api.diagnostics(scenes[0].root).wasmCalls,2,method+' '+change);
    if(change==='append')assert.deepEqual(scenes[0].added.matrixWorld.elements,new api.Matrix4().elements,
      'Children appended beyond the original cached length are not visited');
  }
});

test('segmented batches preserve sibling-local force and exact fallback call arguments',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const force of [undefined,false,true]){
    const scenes=[segmentedTree(api),segmentedTree(reference)],traces=[[],[]];
    for(const [index,scene]of scenes.entries()){
      const {root,before,barrier}=scene;configureManual(root);
      let i=0;root.traverse(node=>{
        if(node===root)return;node.matrixAutoUpdate=false;node.matrixWorldNeedsUpdate=false;
        fill(node.matrix,++i);fill(node.matrixWorld,20+i);
      });
      before[0].matrixWorldNeedsUpdate=true;
      const original=barrier[method];barrier[method]=function(...args){
        traces[index].push(args);return original.apply(this,args);
      };
      updateSubtree(root,method,force);
    }
    same(scenes[0].root,scenes[1].root);assert.deepEqual(traces[0],traces[1]);
    if(force!==true)assert.deepEqual(scenes[0].after[0].matrixWorld.elements,fill(new api.Matrix4(),26).elements);
    assert.equal(api.diagnostics(scenes[0].root).wasmCalls,2);
  }
});

test('segmented exceptions keep prior commits and stop before unvisited children without replay',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const failure of ['hook','readonly']){
    const scenes=[segmentedTree(api),segmentedTree(reference)],traces=[[],[]];
    for(const [index,scene]of scenes.entries()){
      const {root,before,barrier,after}=scene,owner=root.updateMatrix,original=barrier[method];
      root.updateMatrix=function(){traces[index].push('owner');return owner.call(this);};
      barrier[method]=function(...args){
        traces[index].push(['hook',before[3].matrixWorld.elements.slice()]);
        if(failure==='hook')throw new Error('source barrier');return original.apply(this,args);
      };
      if(failure==='readonly')Object.defineProperty(after[2].matrix.elements,'5',{writable:false});
      assert.throws(()=>updateSubtree(root,method),failure==='hook'?/source barrier/:TypeError);
    }
    same(scenes[0].root,scenes[1].root);assert.deepEqual(traces[0],traces[1]);
    assert.equal(traces[0].length,2);
    assert.equal(api.diagnostics(scenes[0].root).wasmCalls,failure==='hook'?1:2);
    assert.deepEqual(scenes[0].after[3].matrixWorld.elements,new api.Matrix4().elements);
  }
});

test('segmented kernel budget failure retains earlier spans and executes the remaining source once',async()=>{
  const {api}=await fixture({maxIterations:4});
  for(const method of ['updateMatrixWorld','updateWorldMatrix']){
    const scenes=[segmentedTree(api,'Object3D',2),segmentedTree(reference,'Object3D',2)],traces=[[],[]];
    for(const [index,scene]of scenes.entries()){
      const library=index===0?api:reference,{root,before,barrier}=scene,owner=root.updateMatrix,original=barrier[method];
      for(let i=0;i<20;i++){const child=new library.Object3D();child.position.x=i+10;root.add(child);}
      root.updateMatrix=function(){traces[index].push('owner');return owner.call(this);};
      barrier[method]=function(...args){
        traces[index].push(['hook',before[1].matrixWorld.elements.slice()]);return original.apply(this,args);
      };
      updateSubtree(root,method);
    }
    same(scenes[0].root,scenes[1].root);assert.deepEqual(traces[0],traces[1]);
    assert.deepEqual(api.diagnostics(scenes[0].root),{
      wasmCalls:1,objects:2,fallbackCalls:1,lastFailure:'KERNEL_EXECUTION_FAILED'});
  }
});
