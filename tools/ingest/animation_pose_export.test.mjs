import test from 'node:test';
import assert from 'node:assert/strict';
import {exportAnimationPoseGLB, AnimationExportError} from './animation_pose_export.mjs';
const I = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const png = Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==','base64'));
function fixture() {
  const pose = {version: 2, disposed: false};
  const deformer = {vertexCount: 3, poseVersion: 2, disposed: false,
    positions: new Float32Array([0,0,1, 1,0,1, 0,1,1]),
    normals: new Float32Array([0,0,1, 0,0,1, 0,0,1]),
    tangents: new Float32Array([1,0,0,1, 1,0,0,1, 1,0,0,1]), worldMatrix: I()};
  const drawable = {shading: 'metallic-roughness', baseColor: [0.2,0.3,0.4,0.5],
    metallicFactor: 0.25, roughnessFactor: 0.75, emissiveFactor: [0.1,0.2,0.3],
    indices: [0,1,2], alphaMode: 'MASK', alphaCutoff: 0.4, doubleSided: true};
  return {pose, deformer, drawable, source: {node: 4, mesh: 2, primitive: 1, material: 3}};
}
// Independent GLB inspection: does not call the writer's serialization helpers.
function inspect(buffer) {
  const header = new DataView(buffer); assert.equal(header.getUint32(0,true),0x46546c67);
  assert.equal(header.getUint32(4,true),2); assert.equal(header.getUint32(8,true),buffer.byteLength);
  const jsonLength=header.getUint32(12,true); assert.equal(jsonLength%4,0); assert.equal(header.getUint32(16,true),0x4e4f534a);
  const json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,jsonLength)));
  const binOffset=28+jsonLength, binLength=header.getUint32(20+jsonLength,true);
  assert.equal(header.getUint32(24+jsonLength,true),0x004e4942);assert.equal(binLength%4,0);
  assert.equal(binOffset+binLength,buffer.byteLength); assert.ok(binLength-json.buffers[0].byteLength<4);
  for(const view of json.bufferViews){assert.equal(view.byteOffset%4,0);assert.ok(view.byteLength>0);assert.ok(view.byteOffset+view.byteLength<=json.buffers[0].byteLength);}
  const data = new DataView(buffer,binOffset,binLength);
  function accessor(index){const a=json.accessors[index],v=json.bufferViews[a.bufferView],width={SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[a.type];
    const bytes={5126:4,5123:2,5125:4}[a.componentType],get={5126:'getFloat32',5123:'getUint16',5125:'getUint32'}[a.componentType];
    assert.equal(v.byteLength,a.count*width*bytes);assert.equal(a.normalized,undefined);
    return Array.from({length:a.count*width},(_,i)=>data[get](v.byteOffset+i*bytes,true));}
  const attribute=(semantic,mesh=0)=>accessor(json.meshes[mesh].primitives[0].attributes[semantic]);
  return {json,data,accessor,attribute};
}
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${i}: ${v} != ${b[i]}`));};
const run=(f,options)=>exportAnimationPoseGLB(f.pose,[{deformer:f.deformer,drawable:f.drawable,source:f.source}],options);

test('exports current positions, core materials, source IDs and a valid aligned static GLB',async()=>{
  const f=fixture(),buffer=await run(f),r=inspect(buffer),p=r.json.meshes[0].primitives[0];
  close(r.attribute('POSITION'),f.deformer.positions);close(r.attribute('NORMAL'),f.deformer.normals);close(r.attribute('TANGENT'),f.deformer.tangents);
  assert.deepEqual(r.accessor(p.indices),[0,1,2]);assert.deepEqual(r.json.accessors[p.attributes.POSITION].min,[0,0,1]);
  assert.deepEqual(r.json.accessors[p.attributes.POSITION].max,[1,1,1]);assert.equal(r.json.buffers[0].uri,undefined);
  const m=r.json.materials[0];assert.equal(m.alphaMode,'MASK');assert.equal(m.alphaCutoff,0.4);assert.equal(m.doubleSided,true);
  assert.equal(m.pbrMetallicRoughness.metallicFactor,0.25);assert.deepEqual(m.emissiveFactor,[0.1,0.2,0.3]);
  assert.deepEqual(r.json.nodes[0].extras.f3dSource,f.source);assert.equal(r.json.extras.f3d.poseVersion,2);
  for(const field of ['skins','animations'])assert.equal(r.json[field],undefined);
  assert.equal(p.targets,undefined);assert.equal(r.json.nodes[0].matrix,undefined);
});

test('bakes translation and nonuniform/sheared transforms with inverse-transpose unit normals',async()=>{
  const f=fixture();f.deformer.worldMatrix=[2,0,1,0, 0,3,0,0, 0,0,4,0, 10,20,30,1];
  const r=inspect(await run(f));close(r.attribute('POSITION'),[10,20,34,12,20,35,10,23,34]);
  const n=[-1/Math.sqrt(5),0,2/Math.sqrt(5)],t=[2/Math.sqrt(5),0,1/Math.sqrt(5),1];
  close(r.attribute('NORMAL'),[...n,...n,...n]);close(r.attribute('TANGENT'),[...t,...t,...t]);
});

test('reflections reverse indexed winding and tangent handedness without changing the model',async()=>{
  const f=fixture();f.deformer.worldMatrix[0]=-2;const r=inspect(await run(f));
  assert.deepEqual(r.accessor(r.json.meshes[0].primitives[0].indices),[0,2,1]);
  close(r.attribute('TANGENT'),[-1,0,0,-1,-1,0,0,-1,-1,0,0,-1]);
  assert.deepEqual(f.drawable.indices,[0,1,2]);assert.equal(f.deformer.tangents[3],1);
});

test('unindexed reflected geometry acquires valid triangle indices; ordinary unindexed remains unindexed',async()=>{
  const f=fixture();delete f.drawable.indices;
  assert.equal(inspect(await run(f)).json.meshes[0].primitives[0].indices,undefined);
  f.deformer.worldMatrix[5]=-1;const r=inspect(await run(f));assert.deepEqual(r.accessor(r.json.meshes[0].primitives[0].indices),[0,2,1]);
});

test('multiple mesh instances preserve distinct world poses and original identities',async()=>{
  const a=fixture(),b=fixture();b.deformer.worldMatrix[12]=9;b.source.node=10;
  const r=inspect(await exportAnimationPoseGLB(a.pose,[{deformer:a.deformer,drawable:a.drawable,source:a.source},{deformer:b.deformer,drawable:b.drawable,source:b.source}]));
  close(r.attribute('POSITION',1),[9,0,1,10,0,1,9,1,1]);assert.equal(r.json.nodes[1].extras.f3dSource.node,10);
  assert.deepEqual(r.json.scenes[0].nodes,[0,1]);
});

test('unlit, vertex colors and alpha settings survive without a texture resolver',async()=>{
  const f=fixture();f.drawable={shading:'unlit',baseColor:[1,0,0,1],alphaMode:'BLEND',vertexColors:[1,0,0,0.5,0,1,0,0.5,0,0,1,0.5]};
  const r=inspect(await run(f));assert.deepEqual(r.json.extensionsRequired,['KHR_materials_unlit']);
  assert.deepEqual(r.json.materials[0].extensions,{KHR_materials_unlit:{}});assert.equal(r.json.materials[0].alphaMode,'BLEND');
  close(r.attribute('COLOR_0'),f.drawable.vertexColors);assert.equal(r.json.images,undefined);
});

function textured(f=fixture()) {
  const color={view:{},sampler:{}},data={view:{},sampler:{}};
  Object.assign(f.drawable,{texCoords:[0,0,1,0,0,1],uvTransform:[2,0,0,3,4,5],baseColorTexture:color,
    metallicRoughnessTexture:data,normalTexture:data,normalScale:0.25,emissiveTexture:color,
    mapCoordinates:{normalTexture:{texCoords:[0.25,0.5,0.75,0.5,0.25,1],uvTransform:[0,1,-1,0,0.5,0.25]}}});
  return f;
}

test('four maps retain independent baked UVs and embedded images/samplers with resource deduplication',async()=>{
  const f=textured();let calls=0;
  const r=inspect(await run(f,{resolveTexture:async()=>{calls++;return {bytes:png,mimeType:'image/png',sampler:{wrapS:33071,wrapT:33648,minFilter:9987,magFilter:9728}};}}));
  assert.equal(calls,2);assert.equal(r.json.textures.length,2);assert.equal(r.json.images.length,1);assert.equal(r.json.samplers.length,1);
  close(r.attribute('TEXCOORD_0'),[4,5,6,5,4,8]);close(r.attribute('TEXCOORD_2'),[4,6.5,4,8,3,6.5]);
  close(r.attribute('TEXCOORD_1'),r.attribute('TEXCOORD_0'));close(r.attribute('TEXCOORD_3'),r.attribute('TEXCOORD_0'));
  const material=r.json.materials[0];assert.equal(material.normalTexture.texCoord,2);assert.equal(material.normalTexture.scale,0.25);
  assert.equal(material.emissiveTexture.index,material.pbrMetallicRoughness.baseColorTexture.index);
  const v=r.json.bufferViews[r.json.images[0].bufferView];assert.deepEqual(new Uint8Array(r.data.buffer,r.data.byteOffset+v.byteOffset,v.byteLength),png);
  assert.deepEqual(r.json.samplers[0],{wrapS:33071,wrapT:33648,magFilter:9728,minFilter:9987});
});

test('different samplers share an encoded image but retain their source sampling',async()=>{
  const f=textured(),first=f.drawable.baseColorTexture.sampler;
  const r=inspect(await run(f,{resolveTexture:t=>({bytes:png,mimeType:'image/png',sampler:{minFilter:t.sampler===first?9728:9987}})}));
  assert.equal(r.json.images.length,1);assert.equal(r.json.samplers.length,2);assert.notEqual(r.json.textures[0].sampler,r.json.textures[1].sampler);
});

test('snapshot precedes asynchronous image work: advancing, mutating or disposing live data cannot change output',async()=>{
  const f=textured();let finish;
  const pending=run(f,{resolveTexture:()=>new Promise(r=>{finish=r;})});
  assert.equal(typeof finish,'function');f.deformer.positions.fill(99);f.deformer.worldMatrix[12]=100;
  f.drawable.baseColor.fill(1);f.drawable.texCoords.fill(99);f.source.node=99;f.pose.version++;f.pose.disposed=true;f.deformer.disposed=true;
  finish({bytes:png,mimeType:'image/png'});await Promise.resolve();await Promise.resolve();
  // The second texture resolves independently; no live arrays are read again.
  finish({bytes:png,mimeType:'image/png'});
  const r=inspect(await pending);close(r.attribute('POSITION'),[0,0,1,1,0,1,0,1,1]);
  assert.deepEqual(r.json.materials[0].pbrMetallicRoughness.baseColorFactor,[0.2,0.3,0.4,0.5]);assert.equal(r.json.nodes[0].extras.f3dSource.node,4);
  close(r.attribute('TEXCOORD_0'),[4,5,6,5,4,8]);
});

test('all geometry/material preflight completes before the first texture resolver call',async()=>{
  const a=textured(),b=fixture();b.drawable.shading='lambert';let calls=0;
  await assert.rejects(exportAnimationPoseGLB(a.pose,[{deformer:a.deformer,drawable:a.drawable},{deformer:b.deformer,drawable:b.drawable}],{resolveTexture:()=>{calls++;}}),{code:'ANIMATION_EXPORT_UNSUPPORTED'});
  assert.equal(calls,0);
});

for(const [name,change] of [
  ['nonfinite position',f=>{f.deformer.positions[0]=NaN;}],['overflow',f=>{f.deformer.worldMatrix[0]=1e100;}],
  ['zero normal',f=>{f.deformer.normals.fill(0);} ],['singular lit world',f=>{f.deformer.worldMatrix[0]=0;}],
  ['nonaffine world',f=>{f.deformer.worldMatrix[3]=1;}],['zero tangent',f=>{f.deformer.tangents.fill(0);} ],
  ['wrong tangent handedness',f=>{f.deformer.tangents[3]=0;}],['missing normal',f=>{f.deformer.normals=null;}],
  ['invalid indices',f=>{f.drawable.indices=[0,1,3];}],['index holes',f=>{f.drawable.indices=[0,,2];}],
  ['index fractions',f=>{f.drawable.indices=[0,0.5,2];}],['incomplete triangles',f=>{f.drawable.indices=[0,1];}],
  ['invalid material',f=>{f.drawable.baseColor[0]=2;}],['unsupported shading',f=>{f.drawable.shading='lambert';}],
  ['malformed UV',f=>{f.drawable.texCoords=[0,0];f.drawable.normalTexture={view:{},sampler:{}};}],
])test(`refuses ${name} instead of writing a misleading/invalid file`,async()=>{
  const f=fixture();change(f);await assert.rejects(run(f,{resolveTexture:()=>({bytes:png,mimeType:'image/png'})}),AnimationExportError);
});

test('live pose and matching deformer versions are required',async()=>{
  for(const change of [f=>{f.pose.disposed=true;},f=>{f.deformer.poseVersion=1;},f=>{f.deformer.disposed=true;}]){
    const f=fixture();change(f);await assert.rejects(run(f),AnimationExportError);
  }
  const f=fixture();Object.defineProperty(f.drawable,'baseColor',{get(){f.pose.version++;return [1,1,1,1];}});
  await assert.rejects(run(f),{code:'ANIMATION_EXPORT_STALE'});
});

test('detached and shared arrays are rejected',async()=>{
  const f=fixture();structuredClone(f.deformer.positions.buffer,{transfer:[f.deformer.positions.buffer]});await assert.rejects(run(f));
  const g=fixture();g.deformer.positions=new Float32Array(new SharedArrayBuffer(36));await assert.rejects(run(g),{code:'ANIMATION_EXPORT_STORAGE'});
});

test('final output limit includes JSON, alignment, and image bytes; exact limit succeeds',async()=>{
  const f=textured(),settings={resolveTexture:()=>({bytes:png,mimeType:'image/png'})},bytes=await run(f,settings);
  await assert.rejects(run(f,{...settings,maxBytes:bytes.byteLength-1}),{code:'ANIMATION_EXPORT_LIMIT'});
  assert.equal((await run(f,{...settings,maxBytes:bytes.byteLength})).byteLength,bytes.byteLength);
  await assert.rejects(run(f,{...settings,maxVertices:2}),{code:'ANIMATION_EXPORT_LIMIT'});
  await assert.rejects(run(f,{...settings,maxBytes:64}),{code:'ANIMATION_EXPORT_LIMIT'});
});

test('maximum uint16 index uses uint32 rather than a forbidden primitive-restart index',async()=>{
  const f=fixture(),count=65536;f.deformer.vertexCount=count;f.deformer.positions=new Float32Array(count*3);
  f.deformer.normals=f.deformer.tangents=null;f.drawable.indices=[0,1,65535];
  const r=inspect(await run(f)),p=r.json.meshes[0].primitives[0];
  assert.equal(r.json.accessors[p.indices].componentType,5125);assert.deepEqual(r.accessor(p.indices),[0,1,65535]);
});

test('abort prevents work, propagates during async image resolution, and does not dispose model/resources',async()=>{
  const f=textured(),a=new AbortController();a.abort();let calls=0;
  await assert.rejects(run(f,{signal:a.signal,resolveTexture:()=>{calls++;}}),{name:'AbortError'});assert.equal(calls,0);
  const b=new AbortController();
  await assert.rejects(run(f,{signal:b.signal,resolveTexture:async(_,{signal})=>{assert.equal(signal,b.signal);b.abort();return {bytes:png,mimeType:'image/png'};}}),{name:'AbortError'});
  assert.equal(f.pose.disposed,false);assert.equal(f.deformer.disposed,false);
});

test('invalid texture encodings and sampler metadata fail without partial output',async()=>{
  for(const result of [null,{bytes:png,mimeType:'image/jpeg'},{bytes:new Uint8Array([1]),mimeType:'image/png'},
    {bytes:png,mimeType:'image/png',sampler:{wrapS:99}},{bytes:png,mimeType:'image/png',sampler:{minFilter:0}},
    {bytes:png,mimeType:'image/png',url:'ignored.png'}])await assert.rejects(run(textured(),{resolveTexture:()=>result}),AnimationExportError);
  await assert.rejects(run(textured()),{code:'ANIMATION_EXPORT_TEXTURE'});
});

test('repeated export is deterministic and contains no external URI or executable metadata',async()=>{
  const f=textured(),settings={resolveTexture:()=>({bytes:png,mimeType:'image/png'})};
  const a=await run(f,settings),b=await run(f,settings);assert.deepEqual(new Uint8Array(a),new Uint8Array(b));
  assert.equal(JSON.stringify(inspect(a).json).includes('uri'),false);
});

test('copyright is retained as inert JSON text',async()=>{
  const text='Artist "name" <script>not code</script> ☃';
  const r=inspect(await run(fixture(),{copyright:text}));assert.equal(r.json.asset.copyright,text);
});
