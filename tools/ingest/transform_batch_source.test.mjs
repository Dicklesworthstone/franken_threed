import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeTransformBatch,transformBatchSourceRole} from './transform_batch_specialization.mjs';
import {specializeMatrixMultiply} from './matrix_multiply_specialization.mjs';
import * as reference from '../../upstream/three.js/src/Three.Core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const matrixSource=fs.readFileSync(path.join(upstream,'src/math/Matrix4.js'),'utf8');
const paths={Object3D:'core/Object3D',Group:'objects/Group',Scene:'scenes/Scene',Bone:'objects/Bone',
  Mesh:'objects/Mesh',InstancedMesh:'objects/InstancedMesh',Matrix4:'math/Matrix4',Vector3:'math/Vector3',
  Quaternion:'math/Quaternion',BufferGeometry:'core/BufferGeometry',MeshBasicMaterial:'materials/MeshBasicMaterial',
  Light:'lights/Light',AmbientLight:'lights/AmbientLight',DirectionalLight:'lights/DirectionalLight',
  HemisphereLight:'lights/HemisphereLight',LightProbe:'lights/LightProbe',PointLight:'lights/PointLight',
  RectAreaLight:'lights/RectAreaLight',SpotLight:'lights/SpotLight',Line:'objects/Line',LineLoop:'objects/LineLoop',
  LineSegments:'objects/LineSegments',Points:'objects/Points',Sprite:'objects/Sprite',LOD:'objects/LOD',
  Camera:'cameras/Camera',SkinnedMesh:'objects/SkinnedMesh'};
async function fixture(options={},changedMatrix=false,withCore=false,production=false) {
  const entry='\0source-world-transform',plugin=marchingCubesRollupPlugin(options),reports=[];
  // Isolate the two production transforms here; application tests exercise the
  // complete compiler and its resolved dependency seam after relocation.
  if(!production)plugin.transform=(code,id)=>{
    if(!transformBatchSourceRole(code))return null;
    if(changedMatrix && id.endsWith('/src/math/Matrix4.js'))return {code:code+`
      Matrix4 = new Proxy(Matrix4, {getOwnPropertyDescriptor(...args) {
        globalThis.__f3dSourceMatrixDescriptorCalls++;
        return Reflect.getOwnPropertyDescriptor(...args);
      }});`,map:null};
    const matrix=code.includes('class Matrix4')
      ?specializeMatrixMultiply(code,{...options,runtimeModule:'\0f3d-matrix-multiply-adapter'})
      :{changed:false,edits:[],header:'',tail:''};
    const batch=specializeTransformBatch(code,{...options,matrixSource,runtimeModule:'\0f3d-transform-batch-adapter'});
    if(batch.report)reports.push({id,...batch.report});
    let transformed=code;
    for(const edit of [...matrix.edits,...batch.edits].sort((a,b)=>b.start-a.start))
      transformed=transformed.slice(0,edit.start)+edit.text+transformed.slice(edit.end);
    return {code:matrix.header+batch.header+transformed+matrix.tail+batch.tail,map:null};
  };
  const source=Object.entries(paths).map(([name,file])=>'export {'+name+'} from '+
    JSON.stringify(path.join(upstream,'src',file+'.js'))+';').join('\n')+
    (withCore?'export {Object3D as CoreObject3D,Group as CoreGroup} from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';':'')+
    'export {transformBatchDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-transform-batch-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'source-world-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?source:null;}},plugin]});
  try {
    const result=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64')),reports};
  } finally {await bundle.close();}
}
let native,limited,modified;
before(async()=>{
  native=await fixture();limited=await fixture({maxIterations:1});
  globalThis.__f3dSourceMatrixDescriptorCalls=0;modified=await fixture({},true);
});
function tree(api) {
  const root=new api.Scene(),left=new api.Group(),right=new api.Bone();
  const mesh=new api.Mesh(new api.BufferGeometry(),new api.MeshBasicMaterial());
  const instances=new api.InstancedMesh(mesh.geometry,mesh.material,2);
  root.add(left,right);left.add(mesh,new api.Object3D());right.add(instances);
  let i=0;
  root.traverse(node=>{
    const n=++i;node.position.set(n/3,-n/7,1);node.rotation.set(n/11,-0.2,0.3);
    node.scale.set(n===4?0:-2,1+n/9,0.5);node.pivot=new api.Vector3(n/8,-1,n/13);
  });
  return root;
}
function same(actual,expected) {
  assert.deepEqual(actual.matrix.elements,expected.matrix.elements);
  assert.deepEqual(actual.matrixWorld.elements,expected.matrixWorld.elements);
  assert.equal(actual.matrixWorldNeedsUpdate,expected.matrixWorldNeedsUpdate);
  assert.equal(actual.children.length,expected.children.length);
  actual.children.forEach((child,i)=>same(child,expected.children[i]));
}
const update=(root,method,force)=>method==='updateWorldMatrix'
  ?root.updateWorldMatrix(false,true,force):root.updateMatrixWorld(force);

test('source Object3D batching requires exact object and Matrix4 inputs',()=>{
  const object=fs.readFileSync(path.join(upstream,'src/core/Object3D.js'),'utf8');
  assert.equal(transformBatchSourceRole(object),'Object3D');
  assert.equal(specializeTransformBatch(object).changed,false);
  assert.equal(specializeTransformBatch(object,{matrixSource:matrixSource+'\n'}).changed,false);
  assert.equal(specializeTransformBatch(object+'\n',{matrixSource}).changed,false);
  const sources=native.reports.filter(report=>report.role==='Object3D');
  assert.equal(sources.length,1);assert.equal(sources[0].sourceDerived,true);
  assert.deepEqual(sources[0].compiledMethods,['Object3D.updateMatrixWorld','Object3D.updateWorldMatrix']);
  assert.equal(sources[0].accelerationClaim,false);
});

test('source hierarchies execute both update methods in Wasm and preserve current public state',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix']) {
    const actual=tree(api),expected=tree(reference),identities=[];
    actual.traverse(node=>identities.push([node,node.matrix,node.matrix.elements,node.matrixWorld,node.matrixWorld.elements,
      node.position,node.quaternion,node.scale,node.pivot]));
    for(let frame=0;frame<3;frame++) {
      for(const root of [actual,expected]) {
        root.children[0].rotation.y+=0.2;root.children[1].pivot.z-=0.7;
        update(root,method,frame===1);
      }
      same(actual,expected);
      assert.equal(api.diagnostics(actual).wasmCalls,frame+1);
      assert.deepEqual(actual.children[0].getWorldPosition(new api.Vector3()).toArray(),
        expected.children[0].getWorldPosition(new reference.Vector3()).toArray());
      assert.deepEqual(actual.children[0].clone().matrix.elements,expected.children[0].clone().matrix.elements);
    }
    for(const [node,matrix,local,world,elements,position,quaternion,scale,pivot]of identities) {
      assert.equal(node.matrix,matrix);assert.equal(matrix.elements,local);assert.equal(node.matrixWorld,world);
      assert.equal(world.elements,elements);assert.equal(node.position,position);assert.equal(node.quaternion,quaternion);
      assert.equal(node.scale,scale);assert.equal(node.pivot,pivot);
    }
  }
});

test('source batching rechecks live topology and manual matrix flags on every call',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(let seed=0;seed<8;seed++) {
    const actual=tree(api),expected=tree(reference);
    for(const root of [actual,expected]) {
      let i=0;root.traverse(node=>{
        const bits=(seed+i++)%8;node.matrixAutoUpdate=Boolean(bits&1);
        node.matrixWorldAutoUpdate=Boolean(bits&2);node.matrixWorldNeedsUpdate=Boolean(bits&4);
      });
      update(root,method,seed%2===0);
    }
    same(actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,1);
    for(const root of [actual,expected]) {
      root.children[1].add(root.children[0].children[0]);root.children[0].position.x+=2;
      update(root,method,true);
    }
    same(actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,2);
  }
});

test('source and built-core classes keep distinct identities and never borrow another dispatch contract',async()=>{
  const {api}=await fixture({},false,true);
  assert.notEqual(api.Object3D,api.CoreObject3D);
  const actual=tree(api),expected=tree(reference),foreign=new api.CoreObject3D(),expectedForeign=new reference.Object3D();
  foreign.position.set(7,8,9);expectedForeign.position.set(7,8,9);
  actual.add(foreign);expected.add(expectedForeign);
  actual.updateMatrixWorld();expected.updateMatrixWorld();same(actual,expected);
  assert.equal(api.diagnostics(actual).wasmCalls,1,'Own-family subtrees batch before the foreign-family barrier');
  assert.equal(api.diagnostics(actual).objects,5);
  assert.equal(api.diagnostics(actual).lastFailure,'TRANSFORM_BATCH_GUARD');
  const core=new api.CoreGroup();core.add(new api.CoreObject3D(),new api.CoreObject3D());core.updateMatrixWorld();
  assert.equal(api.diagnostics(core).wasmCalls,1,'Each module family retains its own valid transaction');
});

test('source overrides, getters and partial failures run once at their original observation boundaries',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const throwing of [false,true]) {
    const actual=tree(api),expected=tree(reference),traces=[[],[]];
    for(const [index,root]of [actual,expected].entries()) {
      const owner=root.updateMatrix;root.updateMatrix=function(){traces[index].push('owner');return owner.call(this);};
      const child=root.children[1],original=child[method];
      Object.defineProperty(child,method,{get(){traces[index].push('method');return function(...args){
        traces[index].push('child');if(throwing)throw new Error('source child failure');return original.apply(this,args);
      };}});
      if(throwing)assert.throws(()=>update(root,method),/source child failure/);
      else update(root,method);
    }
    same(actual,expected);assert.deepEqual(traces[0],traces[1]);assert.deepEqual(traces[0],['owner','method','child']);
    assert.equal(api.diagnostics(actual).wasmCalls,1,'The preceding complete subtree is materialized before the getter');
    assert.equal(api.diagnostics(actual).objects,3);
  }
});

test('source module spans preserve camera and skin barriers between ordinary objects',()=>{
  const {api}=native;
  for(const method of ['updateMatrixWorld','updateWorldMatrix'])for(const kind of ['Camera','SkinnedMesh']){
    const make=library=>{
      const root=new library.Scene(),barrier=new library[kind]();
      for(let i=0;i<9;i++){
        const child=i===4?barrier:new library.Object3D();
        child.position.set(i/3,-i/5,1);child.scale.set(1.5,2,0.7);root.add(child);
      }
      return {root,barrier};
    };
    const actual=make(api),expected=make(reference);
    update(actual.root,method);update(expected.root,method);same(actual.root,expected.root);
    const field=kind==='Camera'?'matrixWorldInverse':'bindMatrixInverse';
    assert.deepEqual(actual.barrier[field].elements,expected.barrier[field].elements);
    assert.deepEqual(api.diagnostics(actual.root),{wasmCalls:2,objects:8,fallbackCalls:1,lastFailure:null});
  }
});

test('unregistered imported matrices and exhausted source budgets retain correct JavaScript execution',()=>{
  for(const {api}of [modified,limited])for(const method of ['updateMatrixWorld','updateWorldMatrix']) {
    const actual=tree(api),expected=tree(reference);
    update(actual,method);update(expected,method);same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,0);
    assert.equal(api.diagnostics(actual).lastFailure,api===modified.api?'TRANSFORM_BATCH_GUARD':'KERNEL_EXECUTION_FAILED');
  }
  assert.equal(globalThis.__f3dSourceMatrixDescriptorCalls,0,'Binding an unknown constructor must not invoke Proxy traps');
});

test('production source imports admit pinned inherited classes in mixed scenes',async()=>{
  const kinds=['Light','AmbientLight','DirectionalLight','HemisphereLight','LightProbe','PointLight','RectAreaLight','SpotLight',
    'Line','LineLoop','LineSegments','Points','Sprite','LOD'];
  for(const kind of kinds) {
    const source=fs.readFileSync(path.join(upstream,'src',paths[kind]+'.js'),'utf8');
    assert.equal(transformBatchSourceRole(source),kind);
    assert.equal(specializeTransformBatch(source+'\n').changed,false,'Modified '+kind+' must remain source-owned');
  }
  const {api}=await fixture({},false,false,true);
  for(const method of ['updateMatrixWorld','updateWorldMatrix']) {
    const make=library=>{
      const scene=new library.Scene(),branch=new library.Group();scene.add(branch,new library.Object3D());
      for(const [i,kind]of kinds.entries()) {
        const child=new library[kind]();child.position.set(i/3,-i/7,1);child.rotation.set(i/11,-0.2,0.4);
        child.scale.set(-1,2,0.5);branch.add(child);
      }
      return scene;
    };
    const actual=make(api),expected=make(reference);
    update(actual,method);update(expected,method);same(actual,expected);
    assert.equal(api.diagnostics(actual).wasmCalls,1,method);
    assert.equal(api.diagnostics(actual).fallbackCalls,0,method);
    assert.equal(api.diagnostics(actual).objects,kinds.length+2,method);
  }
});
