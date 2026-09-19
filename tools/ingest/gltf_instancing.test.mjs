import test from 'node:test';
import assert from 'node:assert/strict';
import {expandGltfInstances} from './gltf_instancing.mjs';
import {decodeGltfAnimation} from './animation_gltf.mjs';
const EXT='EXT_mesh_gpu_instancing';
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${v} != ${b[i]}`));};
function fixture() {
  const model={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{name:'batch',mesh:0,translation:[10,0,0]}],
    meshes:[{primitives:[{attributes:{}}]}],accessors:[],bufferViews:[],buffers:[],extensionsRequired:[EXT],extensionsUsed:[EXT]},buffers=[];
  function attribute(values,type='VEC3',C=Float32Array,normalized=false) {
    const data=new C(values),buffer=buffers.push(data)-1;model.buffers.push({byteLength:data.byteLength});
    const bufferView=model.bufferViews.push({buffer,byteLength:data.byteLength})-1;
    const width={SCALAR:1,VEC3:3,VEC4:4}[type],componentType=new Map([[Float32Array,5126],[Int8Array,5120],[Int16Array,5122],[Uint8Array,5121]]).get(C);
    return model.accessors.push({bufferView,componentType,type,count:values.length/width,...(normalized?{normalized:true}:{})})-1;
  }
  model.nodes[0].extensions={[EXT]:{attributes:{TRANSLATION:attribute([1,2,3,4,5,6])}}};
  return {model,buffers,attribute,get attrs(){return model.nodes[0].extensions[EXT].attributes;}};
}
const expand=(f,options)=>expandGltfInstances(f.model,f.buffers,options);
function unread(f,options,expected) {
  let reads=0;
  assert.throws(()=>expandGltfInstances(f.model,i=>{reads++;return f.buffers[i];},options),expected);
  assert.equal(reads,0,'preflight must precede buffer reads');
}

test('ordinary input is an identity operation without touching buffers',()=>{
  const f=fixture();delete f.model.nodes[0].extensions;delete f.model.extensionsRequired;delete f.model.extensionsUsed;
  const r=expandGltfInstances(f.model,()=>assert.fail('no I/O'));
  assert.equal(r.json,f.model);assert.equal(r.instanceCount,0);assert.equal(r.accessorComponents,0);assert.deepEqual(r.instanceOrigins,{});
});
test('unused extension declarations are consumed without decoding unrelated data',()=>{
  const f=fixture();delete f.model.nodes[0].extensions;
  const r=expandGltfInstances(f.model,()=>assert.fail('no I/O'));
  assert.deepEqual(r.json.extensionsRequired,[]);assert.deepEqual(f.model.extensionsRequired,[EXT]);
});
test('only meshes are multiplied: stable parents, children, cameras, lights and source identities',()=>{
  const f=fixture();f.model.nodes.push({name:'child',camera:0},{mesh:0});f.model.nodes[0].children=[1,2];
  f.model.nodes[0].camera=0;f.model.nodes[0].extensions.KHR_lights_punctual={light:0};
  f.model.extensionsRequired.push('KHR_lights_punctual');const before=structuredClone(f.model),r=expand(f);
  assert.deepEqual(f.model,before);assert.equal(r.json.nodes.length,5);assert.equal(r.json.nodes[0].mesh,undefined);
  assert.deepEqual(r.json.nodes[0].children,[1,2,3,4]);assert.equal(r.json.nodes[0].camera,0);
  assert.deepEqual(r.json.nodes[0].extensions,{KHR_lights_punctual:{light:0}});
  assert.equal(r.json.nodes[1],f.model.nodes[1]);assert.equal(r.json.nodes[2].mesh,0);
  for(const n of r.json.nodes.slice(3)){assert.equal(n.mesh,0);assert.equal(n.children,undefined);assert.equal(n.camera,undefined);}
  assert.deepEqual(r.instanceOrigins,{3:{node:0,instance:0},4:{node:0,instance:1}});
  assert.ok(Object.isFrozen(r.instanceOrigins)&&Object.isFrozen(r.instanceOrigins[3]));
  assert.deepEqual(r.json.extensionsRequired,['KHR_lights_punctual']);
  const pose=decodeGltfAnimation(r.json,f.buffers);assert.deepEqual(pose.nodes.map(n=>n.parent),[-1,0,0,0,0]);
});
test('TRS order is represented by synthetic child transforms, including reflected nonuniform scale',()=>{
  const f=fixture();f.attrs.ROTATION=f.attribute([0,0,0,1,0,0,1,0],'VEC4');f.attrs.SCALE=f.attribute([2,3,4,-2,1,0.5]);
  const r=expand(f),definition=decodeGltfAnimation(r.json,f.buffers);
  assert.deepEqual(definition.nodes[1],{name:'batch',translation:[1,2,3],rotation:[0,0,0,1],scale:[2,3,4],parent:0});
  assert.deepEqual(definition.nodes[2],{name:'batch',translation:[4,5,6],rotation:[0,0,1,0],scale:[-2,1,0.5],parent:0});
  assert.deepEqual(definition.nodes[0].translation,[10,0,0]);assert.equal(r.accessorComponents,20);assert.equal(r.expandedComponents,20);
});
for(const semantic of ['TRANSLATION','ROTATION','SCALE'])test(`${semantic}-only instancing leaves other TRS defaults implicit`,()=>{
  const f=fixture(),values=semantic==='ROTATION'?[0,0,0,1,0,0,0,1]:[1,2,3,4,5,6];
  f.model.nodes[0].extensions[EXT].attributes={[semantic]:f.attribute(values,semantic==='ROTATION'?'VEC4':'VEC3')};
  const r=expand(f);assert.equal(r.instanceCount,2);
  for(const n of r.json.nodes.slice(1))for(const [s,key] of [['TRANSLATION','translation'],['ROTATION','rotation'],['SCALE','scale']])assert.equal(Object.hasOwn(n,key),s===semantic);
});
for(const [C,k] of [[Int8Array,90],[Int16Array,23170]])test(`${C.name} normalized rotation becomes a unit FLOAT node quaternion`,()=>{
  const f=fixture();f.attrs.ROTATION=f.attribute([0,0,k,k,0,0,-k,k],'VEC4',C,true);
  const r=expand(f);close(r.json.nodes[1].rotation,[0,0,Math.SQRT1_2,Math.SQRT1_2]);
  close(r.json.nodes[2].rotation,[0,0,-Math.SQRT1_2,Math.SQRT1_2]);assert.equal(Math.hypot(...r.json.nodes[1].rotation),1);
});
test('repeated instance accessors and buffers are decoded once and source storage is not borrowed',()=>{
  const f=fixture();f.attrs.SCALE=f.attrs.TRANSLATION;let reads=0;
  const r=expandGltfInstances(f.model,()=>{reads++;return f.buffers[0];});
  assert.equal(reads,1);assert.equal(r.accessorComponents,6);
  f.buffers[0].fill(99);assert.deepEqual(r.json.nodes[1].translation,[1,2,3]);assert.deepEqual(r.json.nodes[1].scale,[1,2,3]);
  r.json.nodes[1].translation[0]=123;assert.equal(r.json.nodes[1].scale[0],1);
});
test('interleaved accessor stride uses the existing bounded decoder',()=>{
  const f=fixture(),data=new Float32Array([1,2,3,999,4,5,6,999]);f.buffers[0]=data;
  f.model.buffers[0].byteLength=data.byteLength;Object.assign(f.model.bufferViews[0],{byteLength:data.byteLength,byteStride:16});
  const r=expand(f);assert.deepEqual(r.json.nodes[1].translation,[1,2,3]);assert.deepEqual(r.json.nodes[2].translation,[4,5,6]);
});
test('sparse instance translations overlay their implicit zero base',()=>{
  const f=fixture(),indices=f.attribute([1],'SCALAR',Uint8Array),values=f.attribute([7,8,9]);
  const source=f.model.accessors[f.attrs.TRANSLATION];delete source.bufferView;
  source.sparse={count:1,indices:{bufferView:f.model.accessors[indices].bufferView,componentType:5121},values:{bufferView:f.model.accessors[values].bufferView}};
  const r=expand(f);assert.deepEqual(r.json.nodes[1].translation,[0,0,0]);assert.deepEqual(r.json.nodes[2].translation,[7,8,9]);
});
function animated(f=fixture()) {
  f.model.meshes[0].primitives[0].targets=[{},{}];f.model.meshes[0].weights=[0.1,0.2];f.model.nodes[0].weights=[0.25,0.75];
  const input=f.attribute([0,1],'SCALAR');Object.assign(f.model.accessors[input],{min:[0],max:[1]});
  const weights=f.attribute([0,1,1,0],'SCALAR'),translation=f.attribute([10,0,0,20,0,0]);
  f.model.animations=[{name:'move and morph',samplers:[{input,output:weights},{input,output:translation}],
    channels:[{sampler:0,target:{node:0,path:'weights'}},{sampler:1,target:{node:0,path:'translation'}}]}];return f;
}
test('morph rest overrides and animated weights reach every instance, while parent TRS stays one channel',()=>{
  const f=animated(),before=structuredClone(f.model),r=expand(f),d=decodeGltfAnimation(r.json,f.buffers);
  assert.deepEqual(f.model,before);assert.equal(r.json.nodes[0].weights,undefined);
  assert.deepEqual(d.nodes[1].weights,[0.25,0.75]);assert.deepEqual(d.nodes[2].weights,[0.25,0.75]);
  assert.deepEqual(d.clips[0].channels.map(c=>[c.node,c.path]),[[1,'weights'],[2,'weights'],[0,'translation']]);
  assert.deepEqual(d.clips[0].channels[0].values,[0,1,1,0]);assert.notEqual(d.clips[0].channels[0].values,d.clips[0].channels[1].values);
  assert.equal(r.json.animations[0].samplers,f.model.animations[0].samplers);assert.equal(r.expandedComponents,36);
});
test('mesh-default morph weights and untouched clips remain shared source metadata',()=>{
  const f=animated();delete f.model.nodes[0].weights;f.model.animations[0].channels.shift();
  const r=expand(f),d=decodeGltfAnimation(r.json,f.buffers);
  assert.equal(r.json.animations,f.model.animations);assert.deepEqual(d.nodes[1].weights,[0.1,0.2]);assert.deepEqual(d.nodes[2].weights,[0.1,0.2]);
});
test('multiple and nested batches get disjoint appended pose IDs without duplicating original children',()=>{
  const f=fixture();f.model.nodes.push({...structuredClone(f.model.nodes[0]),children:undefined});f.model.nodes[0].children=[1];
  const r=expand(f);assert.deepEqual(r.json.nodes[0].children,[1,2,3]);assert.deepEqual(r.json.nodes[1].children,[4,5]);
  assert.deepEqual(r.instanceOrigins,{2:{node:0,instance:0},3:{node:0,instance:1},4:{node:1,instance:0},5:{node:1,instance:1}});
  assert.equal(r.instanceCount,4);
});
for(const [name,mutate] of [
  ['missing mesh',f=>{delete f.model.nodes[0].mesh;}],['skin',f=>{f.model.nodes[0].skin=0;}],
  ['custom attributes',f=>{f.attrs._COLOR_0=f.attrs.TRANSLATION;}],['unknown semantic',f=>{f.attrs.POSITION=f.attrs.TRANSLATION;}],
  ['empty attributes',f=>{f.model.nodes[0].extensions[EXT].attributes={};}],['null extension',f=>{f.model.nodes[0].extensions[EXT]=null;}],
  ['extra extension fields',f=>{f.model.nodes[0].extensions[EXT].behavior='unknown';}],
  ['nested extension',f=>{f.model.nodes[0].extensions[EXT].extensions={EXT_unknown:{}};}],
  ['mismatched count',f=>{f.attrs.SCALE=f.attribute([1,1,1]);}],['fractional count',f=>{f.model.accessors[0].count=1.5;}],
  ['bad accessor index',f=>{f.attrs.TRANSLATION=99;}],['unsigned rotation',f=>{f.attrs.ROTATION=f.attribute([0,0,0,255,0,0,0,255],'VEC4',Uint8Array,true);} ],
  ['unnormalized integer rotation',f=>{f.attrs.ROTATION=f.attribute([0,0,0,127,0,0,0,127],'VEC4',Int8Array);} ],
  ['normalized FLOAT',f=>{f.model.accessors[0].normalized=true;}],['bad translation shape',f=>{f.model.accessors[0].type='VEC4';}],
  ['invalid rest weights',f=>{f.model.nodes[0].weights=[1];}],
])test(`preflight rejects ${name} without reading buffers`,()=>{
  const f=fixture();mutate(f);unread(f);
});
for(const q of [[0,0,0,0],[0,0,1,1],[NaN,0,0,1]])test(`invalid instance quaternion ${q} is not repaired`,()=>{
  const f=fixture();f.attrs.ROTATION=f.attribute([...q,...q],'VEC4');assert.throws(()=>expand(f));
});
test('aggregate instance and output budgets are checked before buffer reads, with exact boundaries',()=>{
  const f=fixture();assert.equal(expand(f,{maxInstances:2,maxComponents:20}).instanceCount,2);
  for(const options of [{maxInstances:1},{maxComponents:19},{maxInstances:0},{maxComponents:Infinity}])unread(f,options);
  f.model.nodes.push(structuredClone(f.model.nodes[0]));
  unread(f,{maxInstances:3});
});
test('replicated morph keyframes are charged before downstream decode or instance-buffer reads',()=>{
  const f=animated();assert.equal(expand(f,{maxComponents:36}).expandedComponents,36);
  unread(f,{maxComponents:35},{code:'GLTF_INSTANCING_LIMIT'});
  const sampler=f.model.animations[0].samplers[0];f.model.accessors[sampler.output].count=Number.MAX_SAFE_INTEGER;
  unread(f,undefined,{code:'GLTF_INSTANCING_LIMIT'});
});
test('synthetic nodes respect the existing pose-node ceiling',()=>{
  const f=fixture();f.model.nodes.push(...Array.from({length:65534},()=>({})));
  unread(f,undefined,{code:'GLTF_INSTANCING_LIMIT'});
});
test('short, shared and nonfinite instance buffers never publish partial output',()=>{
  for(const buffer of [new Float32Array(1),new Float32Array(new SharedArrayBuffer(24)),new Float32Array([NaN,0,0,0,0,0])]) {
    const f=fixture();f.buffers[0]=buffer;const before=structuredClone(f.model);assert.throws(()=>expand(f));assert.deepEqual(f.model,before);
  }
});

for(const [name,mutate] of [
  ['animation target',f=>{animated(f);f.model.animations[0].channels[1].target.node=1;}],
  ['skin joint',f=>{f.model.skins=[{joints:[1]}];}],
  ['skin skeleton',f=>{f.model.skins=[{joints:[0],skeleton:1}];}],
  ['scene root',f=>{f.model.scenes[0].nodes=[1];}],
  ['child',f=>{f.model.nodes[0].children=[1];}],
])test(`appended node IDs cannot legalize an invalid source ${name}`,()=>{
  const f=fixture();mutate(f);unread(f,undefined,{code:'GLTF_INSTANCING_LIMIT'});
});
