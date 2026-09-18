import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeDracoMeshes, prepareDracoMeshes} from './gltf_draco.mjs';
import {fixture, EXT} from '../../tests/fixtures/draco/fixture.mjs';
const code = name => error => error.code === 'GLTF_DRACO_' + name;
const deferred = () => { let resolve, reject; const promise = new Promise((r,j) => {resolve=r;reject=j;}); return {promise,resolve,reject}; };
const tick = () => new Promise(resolve => setImmediate(resolve));
function read(result, at) {
  const a = result.json.accessors[at], v = result.json.bufferViews[a.bufferView], bytes = result.buffers[v.buffer];
  const [size,get] = {5120:[1,'getInt8'],5121:[1,'getUint8'],5122:[2,'getInt16'],5123:[2,'getUint16'],5125:[4,'getUint32'],5126:[4,'getFloat32']}[a.componentType];
  const width = {SCALAR:1,VEC2:2,VEC3:3,VEC4:4}[a.type], d = new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength), values=[];
  for(let n=0;n<a.count;n++) for(let c=0;c<width;c++) values.push(d[get]((v.byteOffset??0)+(a.byteOffset??0)+n*(v.byteStride??width*size)+c*size,true));
  return values;
}

for (const componentType of [5120,5121,5122,5123,5125,5126]) test(`raw component ${componentType}: typed requests, padded little-endian storage and untouched source`,async()=>{
  const f=fixture({componentType,normalized:[5120,5121,5122,5123].includes(componentType)}), before=structuredClone(f.json), bytes=f.bytes.slice();
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}), p=r.json.meshes[0].primitives[0];
  assert.deepEqual(f.json,before);assert.deepEqual(f.bytes,bytes);assert.deepEqual(r.sourceJson,before);
  assert.notEqual(r.sourceJson,f.json);assert.deepEqual(r.json.accessors.slice(0,2),f.json.accessors);
  assert.deepEqual(f.calls[0].config,{attributeIDs:{a0:7},attributeTypes:{a0:f.Type.name},useUniqueIDs:true,vertexColorSpace:'srgb-linear'});
  assert.deepEqual(new Uint8Array(f.calls[0].buffer),bytes.subarray(4,12));
  assert.deepEqual(read(r,p.attributes.POSITION),[0,0,0,10,0,0,0,20,0]);assert.deepEqual(read(r,p.indices),[0,1,2]);
  assert.equal(r.json.accessors[p.attributes.POSITION].normalized,f.json.accessors[0].normalized);
  assert.deepEqual(r.json.extensionsRequired,[]);assert.deepEqual(r.json.extensionsUsed,[]);assert.equal(p.extensions,undefined);
  assert.equal(r.decodedPrimitives,1);assert.equal(f.geometries[0].disposed,1);
  const v=r.json.bufferViews[r.json.accessors[p.attributes.POSITION].bufferView];assert.equal(v.byteStride%4,0);
  assert.equal(r.decodedBytes,r.buffers.slice(1).reduce((n,b)=>n+b.byteLength,0));
});

test('uncompressed inputs do not inspect or initialize the borrowed decoder',async()=>{
  const json={asset:{version:'2.0'}},buffers=[];
  const r=await decodeDracoMeshes(json,buffers,{decoder:new Proxy({}, {get(){assert.fail('cold import');}})});
  assert.equal(r.json,json);assert.equal(r.buffers,buffers);assert.equal(r.decodedBytes,0);
});

test('shared source accessor is remapped per compressed primitive and remains unchanged elsewhere',async()=>{
  const f=fixture();f.json.meshes[0].primitives.push(structuredClone(f.primitive),{attributes:{POSITION:0}});
  const decode=f.decoder.decodeGeometry.bind(f.decoder);let call=0;
  f.decoder.decodeGeometry=(...args)=>decode(...args).then(g=>{g.attributes.a0.array[0]=++call;return g;});
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),[a,b,c]=r.json.meshes[0].primitives;
  assert.notEqual(a.attributes.POSITION,b.attributes.POSITION);assert.equal(c.attributes.POSITION,0);
  assert.equal(read(r,a.attributes.POSITION)[0],1);assert.equal(read(r,b.attributes.POSITION)[0],2);
  assert.deepEqual(r.json.accessors[0],f.json.accessors[0]);assert.equal(f.geometries.length,2);assert.ok(f.geometries.every(g=>g.disposed===1));
});

test('direct callers sharing primitive and attribute objects get independent projected storage',async()=>{
  const f=fixture();f.json.meshes.push(f.json.meshes[0]);
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder});
  const a=r.json.meshes[0].primitives[0],b=r.json.meshes[1].primitives[0];
  assert.notEqual(a,b);assert.notEqual(a.attributes,b.attributes);assert.notEqual(a.attributes.POSITION,b.attributes.POSITION);
  assert.ok(f.json.meshes[0].primitives[0].extensions[EXT]);
});

test('uncompressed attributes, morphs and unrelated extensions survive normalization',async()=>{
  const f=fixture();f.json.accessors.push({type:'VEC2',componentType:5126,count:3,bufferView:1},{type:'VEC3',componentType:5126,count:3,bufferView:2});
  f.json.buffers.push({byteLength:24},{byteLength:36});f.buffers.push(new Uint8Array(24),new Uint8Array(36));
  f.json.bufferViews.push({buffer:1,byteLength:24},{buffer:2,byteLength:36});
  Object.assign(f.primitive,{material:5,targets:[{POSITION:3}],extras:{name:'original'}});f.primitive.attributes.TEXCOORD_1=2;
  f.primitive.extensions.UNRELATED={enabled:true};f.json.extensionsRequired.push('KHR_mesh_quantization');
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),p=r.json.meshes[0].primitives[0];
  assert.equal(p.attributes.TEXCOORD_1,2);assert.deepEqual(p.targets,[{POSITION:3}]);assert.equal(p.material,5);
  assert.deepEqual(p.extras,{name:'original'});assert.deepEqual(p.extensions,{UNRELATED:{enabled:true}});
  assert.deepEqual(r.json.extensionsRequired,['KHR_mesh_quantization']);assert.equal(r.buffers[1],f.buffers[1]);
});

test('all skin/color/custom mappings preserve unique IDs and use raw rather than normalized getters',async()=>{
  const f=fixture(),names=['POSITION','NORMAL','COLOR_0','JOINTS_0','WEIGHTS_0','TEXCOORD_7','_TEMPERATURE'];
  f.primitive.attributes={};f.primitive.extensions[EXT].attributes={};f.json.accessors=[];delete f.primitive.indices;
  names.forEach((name,i)=>{f.primitive.attributes[name]=i;f.primitive.extensions[EXT].attributes[name]=37-i;
    f.json.accessors.push({type:'VEC3',count:3,componentType:5126});});
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),p=r.json.meshes[0].primitives[0];
  assert.equal(f.calls.length,1);assert.deepEqual(Object.values(f.calls[0].config.attributeIDs),names.map((_,i)=>37-i));
  for(const name of names)assert.deepEqual(read(r,p.attributes[name]),[0,0,0,10,0,0,0,20,0]);
});

test('interleaved padded decoder attributes are repacked without normalization or leaking padding',async()=>{
  const f=fixture({componentType:5122,normalized:true}),base=f.decoder.decodeGeometry.bind(f.decoder);
  f.decoder.decodeGeometry=(...args)=>base(...args).then(g=>{g.attributes.a0={isInterleavedBufferAttribute:true,
    itemSize:3,count:3,offset:1,normalized:true,data:{stride:5,array:new Int16Array([99,-32768,0,32767,99,99,1,2,3,99,99,4,5,6,99])},getX(){assert.fail('no normalization');}};return g;});
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),p=r.json.meshes[0].primitives[0];
  assert.deepEqual(read(r,p.attributes.POSITION),[-32768,0,32767,1,2,3,4,5,6]);
  const v=r.json.bufferViews[r.json.accessors[p.attributes.POSITION].bufferView],bytes=r.buffers[v.buffer];
  assert.equal(bytes.length,24);assert.equal(bytes[6],0);assert.equal(bytes[7],0);
});

test('decoder-owned output arenas are copied before disposal or reuse',async()=>{
  const f=fixture(),base=f.decoder.decodeGeometry.bind(f.decoder);
  f.decoder.decodeGeometry=(...args)=>base(...args).then(g=>{g.dispose=function(){this.disposed++;this.attributes.a0.array.fill(99);this.index.array.fill(2);};return g;});
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),p=r.json.meshes[0].primitives[0];
  assert.equal(read(r,p.attributes.POSITION)[0],0);assert.deepEqual(read(r,p.indices),[0,1,2]);assert.equal(f.geometries[0].disposed,1);
});

test('geometry without an index accessor gains private decoded connectivity, not sequential guessed faces',async()=>{
  const f=fixture({indices:false}),base=f.decoder.decodeGeometry.bind(f.decoder);
  f.decoder.decodeGeometry=(...args)=>base(...args).then(g=>{g.index.array.set([2,0,1]);return g;});
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),p=r.json.meshes[0].primitives[0];
  assert.deepEqual(read(r,p.indices),[2,0,1]);assert.equal(r.json.accessors[p.indices].componentType,5125);
  assert.equal(f.primitive.indices,undefined);
});

test('nonindexed decoder output is accepted only for complete source vertex triples',async()=>{
  const f=fixture({indices:false}),base=f.decoder.decodeGeometry.bind(f.decoder);
  f.decoder.decodeGeometry=(...args)=>base(...args).then(g=>{g.index=null;return g;});
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder});assert.equal(r.json.meshes[0].primitives[0].indices,undefined);
  f.primitive.indices=1;await assert.rejects(decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),code('DECODE'));assert.equal(f.geometries[1].disposed,1);
});

test('optional compression uses real fallback without invoking an unavailable decoder',async()=>{
  for(const decoder of [null,{supported:false,get decodeGeometry(){assert.fail('not invoked');}}]) {
    const f=fixture();f.json.extensionsRequired=[];f.json.accessors[0].bufferView=1;f.json.accessors[1].bufferView=2;
    f.json.buffers.push({byteLength:36},{byteLength:6});f.json.bufferViews.push({buffer:1,byteLength:36},{buffer:2,byteLength:6});
    f.buffers.push(new Uint8Array(36),new Uint8Array([0,0,1,0,2,0]));
    const r=await decodeDracoMeshes(f.json,f.buffers,{decoder});assert.equal(r.decodedBytes,0);assert.equal(r.decodedPrimitives,0);
    assert.equal(r.json.meshes[0].primitives[0].attributes.POSITION,0);assert.deepEqual(r.json.accessors,f.json.accessors);
    assert.equal(r.buffers[1],f.buffers[1]);assert.equal(r.json.meshes[0].primitives[0].extensions,undefined);
  }
});

test('required decoder or missing optional fallback fails at preflight, before any decoder call',()=>{
  const f=fixture();assert.throws(()=>prepareDracoMeshes(f.json),code('DECODER'));
  assert.throws(()=>prepareDracoMeshes(f.json,{decoder:{}}),code('DECODER'));
  f.json.extensionsRequired=[];assert.throws(()=>prepareDracoMeshes(f.json),code('FALLBACK'));
});

test('fallback accessor byte offsets are ignored only for mapped decoded data',async()=>{
  const f=fixture();f.json.accessors[0].bufferView=999;f.json.accessors[0].byteOffset=999;
  const r=await decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder});assert.equal(read(r,r.json.meshes[0].primitives[0].attributes.POSITION)[3],10);
});

test('compressed strips refuse the topology-aware gap instead of reinterpreting triangle lists',()=>{
  const f=fixture();f.primitive.mode=5;assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder}),code('TOPOLOGY'));
  f.primitive.mode=1;assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder}),code('TOPOLOGY'));
});

for(const [field,value] of [['count',0],['count',-1],['count',1.5],['count',Number.MAX_SAFE_INTEGER],
  ['componentType',999],['normalized','true'],['type','MAT4']]) test(`invalid accessor ${field}=${value} rejects before decoding`,()=>{
  const f=fixture();f.json.accessors[0][field]=value;assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder}));assert.equal(f.calls.length,0);
});

test('invalid ID maps, source view ranges, attribute counts and sparse overlays are rejected early',()=>{
  for(const change of [f=>{f.primitive.extensions[EXT].attributes.NORMAL=1;},f=>{f.primitive.extensions[EXT].attributes.POSITION=-1;},
    f=>{f.primitive.extensions[EXT].attributes.POSITION=2**32;},f=>{f.json.bufferViews[0].byteOffset=9;},
    f=>{f.json.bufferViews[0].byteStride=4;},f=>{f.json.accessors[0].sparse={};},f=>{f.json.accessors[1].count=4;},
    f=>{f.json.accessors[1].normalized=true;},f=>{f.primitive.attributes.NORMAL=f.json.accessors.push({type:'VEC3',componentType:5126,count:4})-1;},f=>{f.json.extensionsUsed={};}]) {
    const f=fixture();change(f);assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder}));assert.equal(f.calls.length,0);
  }
});

test('exact byte budgets include alignment, all mapped attributes, index buffers and aliased primitives',async()=>{
  const f=fixture();assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder,maxEncodedBytes:7}),code('LIMIT'));
  assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder,maxDecodedBytes:43}),code('LIMIT'));
  assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder,maxDecodedBufferBytes:35}),code('LIMIT'));
  const plan=prepareDracoMeshes(f.json,{decoder:f.decoder,maxEncodedBytes:8,maxDecodedBytes:44,maxDecodedBufferBytes:36});
  assert.equal(plan.decodedBytes,44);assert.equal((await plan.decode(f.buffers)).decodedBytes,44);
  f.json.meshes[0].primitives.push(structuredClone(f.primitive));
  assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder,maxDecodedBytes:87}),code('LIMIT'));
  assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder,maxEncodedBytes:15}),code('LIMIT'));
  assert.throws(()=>prepareDracoMeshes(f.json,{decoder:f.decoder,maxPrimitives:1}),code('LIMIT'));
});

test('generated index storage is checked before its allocation and failed output is disposed',async()=>{
  const f=fixture({indices:false});await assert.rejects(decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder,maxDecodedBytes:47}),code('LIMIT'));
  assert.equal(f.geometries[0].disposed,1);assert.equal(f.primitive.indices,undefined);
});

for(const kind of ['width','count','type','missing','nonfinite','detached','shared','stride','offset','indices','indexType','indexCount','indexMissing'])
  test(`invalid decoded ${kind} unwinds owned geometry without changing source`,async()=>{
    const f=fixture(),before=structuredClone(f.json),base=f.decoder.decodeGeometry.bind(f.decoder);
    f.decoder.decodeGeometry=(...args)=>base(...args).then(g=>{
      const a=g.attributes.a0;
      if(kind==='width')a.itemSize=2;if(kind==='count')a.count=4;if(kind==='type')a.array=new Float64Array(9);
      if(kind==='missing')delete g.attributes.a0;if(kind==='nonfinite')a.array[0]=NaN;
      if(kind==='detached')structuredClone(a.array.buffer,{transfer:[a.array.buffer]});
      if(kind==='shared')a.array=new Float32Array(new SharedArrayBuffer(36));
      if(kind==='stride'||kind==='offset'){a.isInterleavedBufferAttribute=true;a.data={array:a.array,stride:kind==='stride'?2:3};a.offset=kind==='offset'?1:0;}
      if(kind==='indices')g.index.array[2]=3;if(kind==='indexType')g.index.array=new Float32Array([0,1,2]);
      if(kind==='indexCount')g.index.count=2;if(kind==='indexMissing')g.index=null;return g;
    });
    await assert.rejects(decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}));
    assert.deepEqual(f.json,before);assert.equal(f.geometries[0].disposed,1);
  });

test('source snapshots precede awaits, including shared compressed buffers transferred by the decoder',async()=>{
  const f=fixture(),gate=deferred(),entered=deferred(),base=f.decoder.decodeGeometry.bind(f.decoder);f.json.meshes[0].primitives.push(structuredClone(f.primitive));
  let calls=0;f.decoder.decodeGeometry=async(buffer,config)=>{
    assert.deepEqual([...new Uint8Array(buffer)],[68,82,65,67,79,1,2,3]);
    const g=await base(buffer,config);structuredClone(buffer,{transfer:[buffer]});
    if(++calls===1){entered.resolve();await gate.promise;}return g;
  };
  const pending=decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder});await entered.promise;
  f.bytes.fill(99);f.json.meshes[0].primitives[1].attributes.POSITION=999;gate.resolve();
  const r=await pending;assert.equal(r.decodedPrimitives,2);assert.equal(calls,2);assert.ok(f.geometries.every(g=>g.disposed===1));
});

test('abort returns before foreign decoding finishes and disposes its late result exactly once',async()=>{
  const f=fixture(),gate=deferred(),started=deferred(),controller=new AbortController(),reason=new Error('cancel');
  f.decoder.decodeGeometry=()=>{started.resolve();return gate.promise;};
  const pending=decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder,signal:controller.signal});await started.promise;
  controller.abort(reason);await assert.rejects(pending,e=>e===reason);
  let disposed=0;gate.resolve({dispose(){disposed++;}});await tick();assert.equal(disposed,1);
  assert.ok(f.primitive.extensions[EXT]);
});

test('late decoder rejection is observed after cancellation, and preaborted input never invokes decoding',async()=>{
  const f=fixture(),gate=deferred(),controller=new AbortController();f.decoder.decodeGeometry=()=>gate.promise;
  const pending=decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder,signal:controller.signal});controller.abort();
  await assert.rejects(pending,e=>e.name==='AbortError');gate.reject(new Error('late'));await tick();
  const g=fixture();await assert.rejects(decodeDracoMeshes(g.json,g.buffers,{decoder:g.decoder,signal:controller.signal}),e=>e.name==='AbortError');assert.equal(g.calls.length,0);
});

test('later failures do not publish earlier decoded storage and preserve original cause/provenance',async()=>{
  const f=fixture(),error=new Error('native decode failed'),base=f.decoder.decodeGeometry.bind(f.decoder),before=structuredClone(f.json);
  f.json.meshes[0].primitives.push(structuredClone(f.primitive));let call=0;
  f.decoder.decodeGeometry=(...args)=>{if(++call===2)throw error;return base(...args);};
  await assert.rejects(decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}),e=>e.code==='GLTF_DRACO_DECODE'&&e.cause===error&&e.message.includes('primitive 1'));
  assert.deepEqual(f.json.meshes[0].primitives[0],before.meshes[0].primitives[0]);assert.equal(f.geometries[0].disposed,1);assert.equal(f.buffers.length,1);
});

test('source bounds, shared/detached bytes and unresolved meshopt views fail before codec invocation',async()=>{
  for(const change of [f=>{f.buffers[0]=new Uint8Array(8);},f=>{f.buffers[0]=new Uint8Array(new SharedArrayBuffer(16));},
    f=>{structuredClone(f.bytes.buffer,{transfer:[f.bytes.buffer]});},f=>{f.json.bufferViews[0].extensions={EXT_meshopt_compression:{}};}]) {
    const f=fixture();change(f);await assert.rejects(decodeDracoMeshes(f.json,f.buffers,{decoder:f.decoder}));assert.equal(f.calls.length,0);
  }
});

test('plan is single-use and cleanup exceptions never replace original decode failures',async()=>{
  const f=fixture(),plan=prepareDracoMeshes(f.json,{decoder:f.decoder});await plan.decode(f.buffers);
  await assert.rejects(plan.decode(f.buffers),code('STATE'));
  const g=fixture();g.decoder.decodeGeometry=()=>({attributes:{},dispose(){throw new Error('cleanup');}});
  await assert.rejects(decodeDracoMeshes(g.json,g.buffers,{decoder:g.decoder}),code('LAYOUT'));
});
