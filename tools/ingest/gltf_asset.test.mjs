import test from 'node:test';
import assert from 'node:assert/strict';
import {parseGltfAsset,loadGltfAsset,GltfAssetError} from './gltf_asset.mjs';
const encode=value=>new TextEncoder().encode(JSON.stringify(value));
const base=()=>({asset:{version:'2.0'}});
const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==','base64'));
const uri=(data,mime='application/octet-stream')=>`data:${mime};base64,${Buffer.from(data).toString('base64')}`;
function glb(json,bin=null,extra=[]) {
  const text=encode(json),length=Math.ceil(text.length/4)*4,j=new Uint8Array(length);j.fill(32);j.set(text);
  const chunks=[[0x4e4f534a,j]];
  if(bin!==null){const b=new Uint8Array(Math.ceil(bin.length/4)*4);b.set(bin);chunks.push([0x004e4942,b]);}
  chunks.push(...extra);
  const output=new Uint8Array(12+chunks.reduce((n,[,b])=>n+8+b.length,0)),view=new DataView(output.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,output.length,true);
  let offset=12;for(const [kind,data]of chunks){view.setUint32(offset,data.length,true);view.setUint32(offset+4,kind,true);output.set(data,offset+8);offset+=8+data.length;}
  return output;
}
function server(entries) {
  const calls=[];
  const fetch=async(url,options)=>{calls.push({url,options});const data=entries[url];return new Response(data??null,{status:data===undefined?404:200});};
  return {calls,fetch};
}
test('JSON and offset GLB views are parsed without borrowing input storage',()=>{
  const json={...base(),buffers:[{byteLength:3}]},data=glb(json,Uint8Array.of(1,2,3));
  const padded=new Uint8Array(data.length+16);padded.set(data,8);
  const result=parseGltfAsset(padded.subarray(8,8+data.length));padded.fill(0);
  assert.deepEqual(result.json,json);assert.deepEqual([...result.bin],[1,2,3,0]);
  assert.deepEqual(parseGltfAsset(encode(base())).json,base());
});
test('unknown trailing GLB chunks are ignored',()=>{
  assert.deepEqual(parseGltfAsset(glb(base(),null,[[0x12345678,new Uint8Array(4)]])).json,base());
});
for(const [name,alter]of [
  ['version',b=>new DataView(b.buffer).setUint32(4,1,true)],
  ['length',b=>new DataView(b.buffer).setUint32(8,b.length-1,true)],
  ['chunk extent',b=>new DataView(b.buffer).setUint32(12,b.length,true)],
  ['alignment',b=>new DataView(b.buffer).setUint32(12,3,true)],
  ['first type',b=>new DataView(b.buffer).setUint32(16,0x004e4942,true)],
])test(`rejects invalid GLB ${name}`,()=>{const b=glb(base());alter(b);assert.throws(()=>parseGltfAsset(b),{code:'GLTF_ASSET_GLB'});});
test('rejects duplicate JSON and out-of-order or duplicate BIN chunks',()=>{
  for(const b of [glb(base(),null,[[0x4e4f534a,new Uint8Array(4)]]),
    glb(base(),null,[[123,new Uint8Array(4)],[0x004e4942,new Uint8Array(4)]]),
    glb(base(),new Uint8Array(4),[[0x004e4942,new Uint8Array(4)]])])assert.throws(()=>parseGltfAsset(b),{code:'GLTF_ASSET_GLB'});
});
test('rejects malformed UTF-8, JSON, unsupported versions and unsafe storage',()=>{
  for(const bytes of [Uint8Array.of(255),new TextEncoder().encode('{'),encode({asset:{version:'1.0'}}),encode({asset:{version:'2.0',minVersion:'2.1'}})])assert.throws(()=>parseGltfAsset(bytes),GltfAssetError);
  assert.throws(()=>parseGltfAsset(new SharedArrayBuffer(8)),{code:'GLTF_ASSET_BYTES'});
  const detached=new ArrayBuffer(4);structuredClone(detached,{transfer:[detached]});assert.throws(()=>parseGltfAsset(detached),{code:'GLTF_ASSET_BYTES'});
  assert.throws(()=>parseGltfAsset(new ArrayBuffer(4,{maxByteLength:8})),{code:'GLTF_ASSET_BYTES'});
});
test('loads GLB BIN padding and external buffers without truncating declared data',async()=>{
  const json={...base(),buffers:[{byteLength:3},{byteLength:2,uri:'extra.bin'}]};
  const s=server({'https://a.test/models/extra.bin':Uint8Array.of(8,9,10)});
  const asset=await loadGltfAsset(glb(json,Uint8Array.of(1,2,3)),{baseURL:'https://a.test/models/a.glb',fetch:s.fetch});
  assert.deepEqual(asset.buffers.map(b=>[...b]),[[1,2,3],[8,9]]);assert.equal(s.calls.length,1);
});
test('missing BIN, excessive padding, nonzero padding and short buffers fail',async()=>{
  const json={...base(),buffers:[{byteLength:1}]};
  for(const bytes of [encode(json),glb(json,new Uint8Array(8)),glb(json,Uint8Array.of(1,2))])await assert.rejects(loadGltfAsset(bytes),{code:'GLTF_ASSET_BUFFER'});
  await assert.rejects(loadGltfAsset(encode({...base(),buffers:[{byteLength:2,uri:uri([1])}]})),{code:'GLTF_ASSET_BUFFER'});
});
test('URL resolution, query identity and fragment deduplication preserve fetch semantics',async()=>{
  const json={...base(),buffers:[{uri:'./a%20b.bin?v=1#x',byteLength:2},{uri:'a%20b.bin?v=1#y',byteLength:1},{uri:'a%20b.bin?v=2',byteLength:1}]};
  const s=server({'https://a.test/models/m.gltf':encode(json),'https://a.test/models/a%20b.bin?v=1':Uint8Array.of(7,8),'https://a.test/models/a%20b.bin?v=2':Uint8Array.of(9)});
  const asset=await loadGltfAsset('models/m.gltf',{baseURL:'https://a.test/',fetch:s.fetch});
  assert.deepEqual(asset.buffers.map(b=>[...b]),[[7,8],[7],[9]]);assert.equal(s.calls.length,3);
  for(const {options}of s.calls){assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');}
  assert.equal(asset.bytesLoaded,encode(json).length+3);
});
test('data URI percent octets, base64 and shared storage work without fetch',async()=>{
  const json={...base(),buffers:[{uri:'data:application/gltf-buffer,%00%ff+',byteLength:3},{uri:uri([3,4]),byteLength:2},{uri:uri([3,4]),byteLength:2}]};
  const asset=await loadGltfAsset(encode(json),{fetch:()=>assert.fail('No network expected')});
  assert.deepEqual(asset.buffers.map(b=>[...b]),[[0,255,43],[3,4],[3,4]]);
  assert.equal(asset.bytesLoaded,encode(json).length+5);
});
for(const value of ['data:application/octet-stream;base64,AB==','data:application/octet-stream;base64,a','data:application/octet-stream,%xx','data:application/octet-stream,é'])
  test(`rejects malformed data URI ${value}`,async()=>{await assert.rejects(loadGltfAsset(encode({...base(),buffers:[{uri:value,byteLength:1}]})),{code:'GLTF_ASSET_URI'});});
test('external dependency origins and dangerous schemes fail before fetch',async()=>{
  for(const resource of ['https://other.test/a.bin','file:///a.bin','javascript:alert(1)','https://user:pass@a.test/x']) {
    const json={...base(),buffers:[{uri:resource,byteLength:1}]};
    await assert.rejects(loadGltfAsset(encode(json),{baseURL:'https://a.test/x.gltf',fetch:()=>assert.fail('Blocked before request')}),{code:'GLTF_ASSET_URI'});
  }
});
test('explicit additional origin is supported without credentials',async()=>{
  const json={...base(),buffers:[{uri:'https://cdn.test/a.bin',byteLength:1}]};const s=server({'https://cdn.test/a.bin':Uint8Array.of(1)});
  await loadGltfAsset(encode(json),{baseURL:'https://a.test/',allowedOrigins:['https://cdn.test'],fetch:s.fetch});assert.equal(s.calls.length,1);
});
test('invalid response, redirects and response URL changes fail and cancel the body',async()=>{
  for(const extra of [{ok:false,status:404},{redirected:true},{url:'https://evil.test/stolen'}]) {
    let cancelled=0;
    const response={ok:true,status:200,headers:new Headers(),body:new ReadableStream({cancel(){cancelled++;}}),...extra};
    await assert.rejects(loadGltfAsset('https://a.test/model.gltf',{fetch:async()=>response}),{code:'GLTF_ASSET_HTTP'});assert.equal(cancelled,1);
  }
});
test('streaming limits stop body consumption without trusting Content-Length',async()=>{
  for(const headers of [{},{'content-length':'1'}]) {
    let cancelled=0;
    const fetch=async()=>new Response(new ReadableStream({start(c){c.enqueue(new Uint8Array(16));},cancel(){cancelled++;}}),{headers});
    await assert.rejects(loadGltfAsset('https://a.test/x',{fetch,maxResourceBytes:8}),{code:'GLTF_ASSET_LIMIT'});assert.equal(cancelled,1);
  }
});
test('declared and cumulative budgets fail before allocations/dependency fetches',async()=>{
  let calls=0;
  await assert.rejects(loadGltfAsset('https://a.test/x',{maxBytes:8,fetch:async()=>new Response('x',{headers:{'content-length':'100'}})}),{code:'GLTF_ASSET_LIMIT'});
  const json={...base(),buffers:[{byteLength:999,uri:'b.bin'}]};
  await assert.rejects(loadGltfAsset(encode(json),{maxResourceBytes:200,fetch:()=>{calls++;}}),{code:'GLTF_ASSET_BUFFER'});assert.equal(calls,0);
  const embedded={...base(),buffers:[{byteLength:2,uri:uri([1,2])}]};
  await assert.rejects(loadGltfAsset(encode(embedded),{maxBytes:encode(embedded).length+1}),{code:'GLTF_ASSET_LIMIT'});
});
test('pre-abort avoids fetch and mid-stream abort cancels a pending reader',async()=>{
  const c=new AbortController();c.abort();let calls=0;
  await assert.rejects(loadGltfAsset('https://a.test/x',{signal:c.signal,fetch:()=>{calls++;}}),{name:'AbortError'});assert.equal(calls,0);
  const next=new AbortController();let cancelled=0;
  const pending=loadGltfAsset('https://a.test/x',{signal:next.signal,fetch:async()=>new Response(new ReadableStream({cancel(){cancelled++;}}))});
  await new Promise(resolve=>setImmediate(resolve));next.abort();
  await assert.rejects(pending,{name:'AbortError'});assert.equal(cancelled,1);
});
test('loads only requested images and deduplicates concurrent reads',async()=>{
  const json={...base(),images:[{uri:'used.png'},{uri:'https://unreachable.test/unused.png'}]};const s=server({'https://a.test/used.png':png});
  const asset=await loadGltfAsset(encode(json),{baseURL:'https://a.test/model.gltf',fetch:s.fetch});assert.equal(s.calls.length,0);
  const [a,b]=await Promise.all([asset.readImage(0),asset.readImage(0)]);
  assert.equal(a,b);assert.equal(a.mimeType,'image/png');assert.deepEqual(a.bytes,png);assert.equal(s.calls.length,1);
});
test('embedded images use buffer-view offsets and snapshot metadata',async()=>{
  const bin=new Uint8Array(png.length+4);bin.set(png,4);
  const json={...base(),buffers:[{byteLength:bin.length}],bufferViews:[{buffer:0,byteOffset:4,byteLength:png.length}],images:[{bufferView:0,mimeType:'image/png'}]};
  const asset=await loadGltfAsset(glb(json,bin));asset.json.bufferViews[0].byteOffset=0;asset.json.images[0].mimeType='image/jpeg';
  assert.deepEqual((await asset.readImage(0)).bytes,png);
});
test('core data-URI images load; invalid ranges, MIME and codecs fail',async()=>{
  const model={...base(),images:[{uri:uri(png,'image/png')},{uri:uri(png,'image/jpeg')},{uri:uri([1,2,3],'image/webp')},{bufferView:0,mimeType:'image/png'}]};
  const asset=await loadGltfAsset(encode(model));assert.equal((await asset.readImage(0)).mimeType,'image/png');
  for(const i of [1,2,3,-1,99])await assert.rejects(asset.readImage(i),{code:'GLTF_ASSET_IMAGE'});
});
test('resource count, input byte limits and invalid options are bounded',async()=>{
  assert.throws(()=>parseGltfAsset(encode(base()),{maxBytes:1}),{code:'GLTF_ASSET_LIMIT'});
  for(const options of [{maxBytes:0},{maxResourceBytes:0},{maxResources:1.5},{allowedOrigins:'*'}])await assert.rejects(loadGltfAsset(encode(base()),options),GltfAssetError);
  await assert.rejects(loadGltfAsset(encode({...base(),buffers:[{uri:uri([1]),byteLength:1},{uri:uri([2]),byteLength:1}]}),{maxResources:1}),{code:'GLTF_ASSET_LIMIT'});
});
