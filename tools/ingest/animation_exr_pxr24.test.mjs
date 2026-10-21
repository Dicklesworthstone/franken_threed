import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {deflateSync,inflateSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {decodeAnimationExr, inspectAnimationExr} from './animation_exr.mjs';
// Genuine OpenCV 4.13/native OpenEXR files. Goldens are native-decoded RGBA16F,
// not pre-encode floats: compressed PXR24 FLOAT already has authored 24-bit precision.
const native = {
  "1_8x4": {
    "base64": "eJwr0zdkZGJgYEjOSMzLS80pBjJyMotLGDyBYo4MjAwQwAjFThgi7hgiQRgiDMn5uQVFqcXFmfl5KGyQJGtKYklieGZeSn45Q1J+hVEmgwADArADMTMQp2QWF+QkVhJUl5OZl+pflJJahMQCO6EgsyI1x7G4IDW5JCixJDOfIS0nP7GEgQWsu8G+OLkoNTUPYrxzal4JUFuZURoDB5IVyErCM1NKMtCMYAhmRCh+BMQVczb//r26bW5n58egtv9b1+j/kuGZdsanp4eHh+dX/Olk1/r7pYxGkv/+/78vmrFa8dP3o6nhCr//76//f8bs7GvhyirNk+/ev+fyWzAt83Oq3/QjxiJGEQyWCrbs+y19f39/Y+Kzf+H01lUX7l+/HrPg1/q3EdOUPn3f2v92v739rMiZEvcjznc/nbhX8GgC39b4H4vP9Zy4ennugl/Gn5p0l/y7Z//kj3ySw/Mps6Y4/O6vM3P4dv++/rLaAL7vXvtnlU89yLHq/1/3y5wvtdVuO36tP8tpMvnP9wcdh+s5Ouz3GtleuRnCztCTVC0DAJ/VwBA=",
    "sha256": "dc68db55f9425408e29847680f5a25bde28353dcbbd131dbe7ff0abb2cea25a4"
  },
  "2_8x4": {
    "base64": "eJwr0zdkZGJgYEjOSMzLS80pBjJyMotLGDyBYo4MIBkQYIRiJwwRdwyRIAwRhuT83IKi1OLizPw8FDZIkjUlsSQxPDMvJb+cISm/wiiTQYABAdiBmBmIUzKLC3ISKwmqy8nMS/UvSkktQmKBnVCQWZGa41hckJpcEpRYkpnPkJaTn1jCwALW3WBfnFyUmpoHMd45Na8EqK3MKI2BA8kKZCXhmSklGWhGMAQzIhR7AdkVczb//79/HiNH3tqvc76anQs96PDryMGGPun5Em8kFRy2nvOP9oyb9fDvmeDXYct8yhk6//7/X68kbq19+f/+/XtaxT4tzU/tK//Bz3h6X+2/Hy/OrTVOD82TPXlnkWAGv8l8wRVPLz1L8dybw75X1/50Uvf8zycPn52+4rPBpOkTUy7cfrfn1xNN+xk/c+fk6Ch8nprzT9/w5tMa2bR/a/Lj79nsqv/+ZN/6cq8osaqtzGf/32e4sun7j5R3R44G3trqejq9/Py/p0cS2E3/B85dPXHbCgHnXRWX09++8Tb4XFu588aM////Hdxk/nnn7TPbHQ0+c9vvrf/86ufeyV62O4NZ44tlt7B3/8ud9f3JlZOmoXu2/9ib7xo/58MC9ry6nLmzF/Z1CBzmbf2aL/9Pc0v7rHr+nH/z5blP9fJ3v0ld0zTz/N/Pr2eH68+zrZ1x8vNmzeRiPQaJM7uvAgAYpvz/",
    "sha256": "67fb25a8f151388ae947aa6237c620eec7cf36400472f581f2a23755de75bc6a"
  },
  "piz_panorama": {
    "base64": "eJwr0zdkZGJgYEjOSMzLS80pBjJyMotLGDyBYo4MjAwQwAjFThgi7hgiQRgiDMn5uQVFqcXFmfl5KGyQJEtKYklieGZeSn45Q1J+hVEmgwADAtgDsTwQp2QWF+QkVhJUl5OZl+pflJJahMQCO6EgsyI1x7G4IDW5JCixJDOfIS0nP7GEgQWsu8G+OLkoNTUPYrxzal4JUFuZURoDB5IVyErCM1NKMtCMYAhmRCiWkQUSbA3iSEIDAii1n9b66S0/UPyBpAc6DY6CUTAKRsEoGAWjYBSMglEwCkbBSAAD3fca8f3/ydwIpVeAqsN8ITyORA/1TRwMTScY+k4YMB5QYPzPLP9/FAxB8FT5g+BuDQMmjgMMfOeTvlVX23bffMNgDSTfvvmmn75zJiqYvPPOu3LrueeeF8MYlWgqJlZXi2/fiEqQbsr0zbdBDgC5Z++8e+fzB9AhN0AOgIbMaJiAwIRARhYmAV6Gb9UuLAIcDibMN9+IODgytDA72M67CJOSMskoPHOmV8rE4lmaW6+UTLGxslsvnyVQM1PAjZh35awCqryt554XoHILFx43NjYC6Ulza5c5c+bEhcdAC9nfKAOdikmiuWsmiqvBDHQVYAebvvnWwaEg0JB9862AgIMiw/XNtxQUGpwY16dvXAyy9cTFxc59aSkZF4GOMTLJADkBu2thXJDvjJWBeoCagJ4+A/LnzAl1h6zFt2/AIEkP7ykYgQsLd4j7L/APaofHnc9oaGLkYNsLdX+17SImDhYFqcGaRhiUGBfYNgBDm+GC3AG58wksDibsG9iBaUSK+QYwqSMCaiLCIc6JIKOkuYFmFoCcat12CGj4vTRUkP5GGuii32xv4u79O7hbfPv+DXlH375DU5SCFqiYxqQK8DICC4IOtgZgYeDC7AAsEBYBnbx33jFEcCDFrJk40JiHG0DmgYKveEeOHtDgZ2iG1sEd9B7hSnRFfmhBh2lMICTQMhoy+BP407eMBh3xQee4G1bDbMiDVTik+zAl7t7Rt29QCTLCCbme2Q6seAbQIfpAByBXwiM+TFIAWRMc9A==",
    "sha256": "8b71da11b84031f5ff32acc35c3708043a567ed6f6ffd297dd4d5821930b0eaf"
  }
};
const fixture=name=>new Uint8Array(inflateSync(Buffer.from(native[name].base64,'base64')));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytesOf=image=>new Uint8Array(image.data.buffer);
for(const name of ['1_8x4','2_8x4'])test('native PXR24 '+name+' matches every RGBA16F byte',async()=>{
  const input=fixture(name),image=await decodeAnimationExr(input);
  assert.equal(image.compression,'pxr24');assert.equal(digest(bytesOf(image)),native[name].sha256);
  assert.deepEqual([image.width,image.height],[8,4]);assert.equal(image.clampedComponents,0);
});
const word=(n,signed=false)=>{const b=Buffer.alloc(4);signed?b.writeInt32LE(n):b.writeUInt32LE(n);return b;};
const str=s=>Buffer.from(s+'\0');
const attribute=(name,type,bytes)=>Buffer.concat([str(name),str(type),word(bytes.length),bytes]);
// Independent packet writer for exact word patterns and corrupted inflate extents.
// Each row/channel has high-byte-first delta planes; floats are supplied as bits.
function make({width=8,height=19,types={A:1,B:2,G:0,R:1},value=(x,y,c)=>({A:0x3800,B:0xbf800000,G:y*8+x,R:x%2?0x8000:0x3c00}[c]),raw=false,packed}={}){
  const names=Object.keys(types).sort(),box=Buffer.concat([0,0,width-1,height-1].map(n=>word(n,true)));
  const channels=Buffer.concat([...names.map(n=>Buffer.concat([str(n),word(types[n]),Buffer.alloc(4),word(1),word(1)])),Buffer.of(0)]);
  const header=Buffer.concat([word(20000630),word(2),
    attribute('channels','chlist',channels),attribute('compression','compression',Buffer.of(5)),
    attribute('dataWindow','box2i',box),attribute('displayWindow','box2i',box),
    attribute('lineOrder','lineOrder',Buffer.of(0)),attribute('pixelAspectRatio','float',word(0x3f800000)),
    attribute('screenWindowCenter','v2f',Buffer.alloc(8)),attribute('screenWindowWidth','float',word(0x3f800000)),Buffer.of(0)]);
  const blocks=[];
  for(let y=0;y<height;y+=16){const rows=Math.min(16,height-y),parts=[],literal=[];
    for(let dy=0;dy<rows;dy++)for(const c of names){const type=types[c],size=type===1?2:4,planes=type===2?3:size;
      const data=Buffer.alloc(width*planes);let previous=0;
      for(let x=0;x<width;x++){
        const pixel=value(x,y+dy,c)>>>0,delta=(pixel-previous)>>>0;previous=pixel;
        const v=Buffer.alloc(size);size===2?v.writeUInt16LE(pixel):v.writeUInt32LE(pixel);literal.push(v);
        for(let p=0;p<planes;p++)data[p*width+x]=(delta>>>((size-p-1)*8))&255;
      }parts.push(data);
    }
    const payload=Buffer.concat(parts),full=Buffer.concat(literal),compressed=packed?packed(payload):deflateSync(payload);
    const data=raw||(!packed&&compressed.length>=full.length)?full:compressed;
    blocks.push(Buffer.concat([word(y,true),word(data.length),data]));
  }
  let offset=header.length+blocks.length*8;const table=Buffer.alloc(blocks.length*8);
  for(let i=0;i<blocks.length;i++){table.writeBigUInt64LE(BigInt(offset),i*8);offset+=blocks[i].length;}
  return Buffer.concat([header,table,...blocks]);
}
test('mixed HALF/FLOAT/UINT words reset each row/channel across partial 16-row blocks',async()=>{
  const image=await decodeAnimationExr(make());
  const halfInt=n=>{if(!n)return 0;const e=Math.floor(Math.log2(n));return ((e+15)<<10)+(n*2**(10-e)-1024);};
  for(let y=0;y<19;y++)for(let x=0;x<8;x++)assert.deepEqual([...image.data.slice((y*8+x)*4,(y*8+x+1)*4)],
    [x%2?0x8000:0x3c00,halfInt(y*8+x),0xbc00,0x3800]);
});
test('PXR24 UINT delta wraps and overflow use the existing explicit-clamp policy',async()=>{
  const input=make({types:{Y:0},value:x=>[0,42,0xffffffff,1][x%4]});
  await assert.rejects(decodeAnimationExr(input),{code:'ANIMATION_EXR_VALUE'});
  const image=await decodeAnimationExr(input,{overflow:'clamp'});
  assert.deepEqual([...image.data.slice(0,16)],[0,0,0,0x3c00,0x5140,0x5140,0x5140,0x3c00,0x7bff,0x7bff,0x7bff,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00]);
  assert.equal(image.clampedComponents,19*2*3);
});
test('PXR24 raw fallback preserves low FLOAT bits rather than applying codec quantization',async()=>{
  // Just above the halfway point: chopping eight low bits would instead round to 1.
  const image=await decodeAnimationExr(make({width:1,height:1,types:{Y:2},value:()=>0x3f801001,raw:true}));
  assert.deepEqual([...image.data],[0x3c01,0x3c01,0x3c01,0x3c00]);
});
test('signed zero/subnormals and full-width UINT precision survive separate delta planes',async()=>{
  const input=make({types:{Y:1},value:x=>[0,0x8000,1,0x8001,0x3ff,0x400,0x7bff,0xfbff][x]});
  const image=await decodeAnimationExr(input);
  assert.deepEqual([...image.data.slice(0,32)].filter((_,i)=>i%4===0),[0,0x8000,1,0x8001,0x3ff,0x400,0x7bff,0xfbff]);
});
for(const packed of [()=>Buffer.of(1,2,3),()=>deflateSync(Buffer.alloc(1)),p=>deflateSync(Buffer.alloc(p.length+1))])
  test('PXR24 rejects corrupt, short or overexpanded deflate streams',async()=>{
    await assert.rejects(decodeAnimationExr(make({packed})),e=>e.name==='AnimationExrError');
  });
test('PXR24 bounds output plus two full blocks before decompressing',async()=>{
  const input=make(),info=inspectAnimationExr(input),minimum=8*19*8+2*8*16*12;
  assert.equal(info.peakDecodedBytes,minimum);
  await assert.rejects(decodeAnimationExr(input,{maxDecodedBytes:minimum-1}),{code:'ANIMATION_EXR_LIMIT'});
  assert.equal((await decodeAnimationExr(input,{maxDecodedBytes:minimum})).byteLength,8*19*8);
});
test('PXR24 observes mid-inflate abort and snapshots transferred input before awaiting',async()=>{
  const input=fixture('2_8x4'),pending=decodeAnimationExr(input);structuredClone(input,{transfer:[input.buffer]});
  assert.equal(digest(bytesOf(await pending)),native['2_8x4'].sha256);
  const controller=new AbortController(),reason=new Error('stop inflate');
  const cancelled=decodeAnimationExr(fixture('2_8x4'),{signal:controller.signal});controller.abort(reason);
  await assert.rejects(cancelled,e=>e===reason);
});
test('PXR24 nonfinite words are never silently clamped or published',async()=>{
  for(const value of [0x7f800000,0xff800000,0x7fc00000])
    await assert.rejects(decodeAnimationExr(make({types:{Y:2},value:()=>value}),{overflow:'clamp'}),{code:'ANIMATION_EXR_VALUE'});
});

// Run the unmodified production loader against the real EXR module. Only GPU,
// IBL filtering and the unused RGBE boundary are substitutes. No shader claim.
test('PIZ/PXR24 native panoramas load from bytes and streams with exact uploads and owned cleanup',()=>{
  const child=String.raw`
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {inflateSync} from 'node:zlib';
import {createHash} from 'node:crypto';
const root=new URL(process.argv[1]),fixtures=JSON.parse(process.argv[2]);
const exr=await import(new URL('./animation_exr.mjs',root));
let uploads=[],textures=[],filtered=0;
class BoundaryError extends Error {constructor(code,message){super(message);this.code=code;}}
const dependencies={
  './animation_exr.mjs':exr,
  './animation_hdr.mjs':{AnimationHdrError:BoundaryError,decodeAnimationHdr(){assert.fail('EXR must not retry RGBE');}},
  './animation_environment.mjs':{AnimationEnvironmentError:BoundaryError,planAnimationEnvironment:()=>({uniformBytes:256,textureBytes:64}),
    async createGpuAnimationEnvironment(device,texture,options){assert.equal(texture.destroyed,0);assert.equal(options.signal.aborted,false);filtered++;let disposed=false;
      return {dispose(){disposed=true;},get disposed(){return disposed;},failed:false,textureBytes:64,sample:d=>{assert.equal(d,device);return {profile:'test-ibl'};},whenIdle:async()=>{}};}}
};
const module=new vm.SourceTextModule(await readFile(root,'utf8'),{identifier:root.href});
await module.link(specifier=>{const values=dependencies[specifier];assert.ok(values,specifier);return new vm.SyntheticModule(Object.keys(values),function(){for(const [k,v]of Object.entries(values))this.setExport(k,v);});});
await module.evaluate();const load=module.namespace.loadGpuAnimationEnvironment;
const device={limits:{maxTextureDimension2D:4096,minUniformBufferOffsetAlignment:256,maxBufferSize:1048576},lost:new Promise(()=>{}),
  createTexture(desc){const texture={...desc,destroyed:0,destroy(){this.destroyed++;}};textures.push(texture);return texture;},
  pushErrorScope(){},popErrorScope:async()=>null,queue:{writeTexture(destination,data,layout,size){uploads.push({data:new Uint8Array(data.buffer).slice(),layout,size});},writeBuffer(){},submit(){},onSubmittedWorkDone:async()=>{}}};
for(const key of ['createBuffer','createSampler','createBindGroupLayout','createPipelineLayout','createShaderModule','createRenderPipelineAsync','createBindGroup','createCommandEncoder'])device[key]=()=>{throw Error('Unused GPU boundary');};
for(const [name,fixture]of Object.entries(fixtures))for(const streamed of [false,true]){
  uploads=[];textures=[];const bytes=new Uint8Array(inflateSync(Buffer.from(fixture.base64,'base64'))),length=bytes.length;
  const settings=streamed?{fetch:async(url,options)=>{
    assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');
    return new Response(new ReadableStream({start(c){c.enqueue(bytes.subarray(0,17));c.enqueue(bytes.subarray(17));c.close();}}),{headers:{'content-type':'image/png'}});
  }}:{};
  const pending=load(device,streamed?'https://assets.test/misleading.png':bytes,settings);
  if(!streamed)structuredClone(bytes,{transfer:[bytes.buffer]});
  const map=await pending;assert.equal(uploads.length,1);assert.equal(textures[0].destroyed,1);
  assert.equal(createHash('sha256').update(uploads[0].data).digest('hex'),fixture.sha256);
  assert.equal(map.sourceInfo.inputBytes,length);assert.equal(map.sourceInfo.compression,name==='piz_panorama'?'piz':'pxr24');
  assert.equal(uploads[0].layout.bytesPerRow,map.sourceInfo.width*8);assert.equal(map.disposed,false);
  await map.whenIdle();map.dispose();assert.equal(map.disposed,true);
}
const bytes=()=>new Uint8Array(inflateSync(Buffer.from(fixtures['2_8x4'].base64,'base64')));
const before=filtered;uploads=[];
const c=new AbortController(),reason=new Error('caller stopped');const pending=load(device,bytes(),{signal:c.signal});c.abort(reason);
await assert.rejects(pending,e=>e===reason);assert.equal(filtered,before);assert.equal(uploads.length,0);
const info=exr.inspectAnimationExr(bytes());
await assert.rejects(load(device,bytes(),{maxDecodedBytes:info.peakDecodedBytes-1}),{code:'ANIMATION_EXR_LIMIT'});
await assert.rejects(load(device,bytes(),{maxTextureBytes:info.byteLength+63}),{code:'ANIMATION_HDR_LIMIT'});
assert.equal(filtered,before);assert.equal(uploads.length,0);
console.log('six native panorama byte/stream routes, exact uploads, transfer, cancellation, budgets and disposal passed');
`;
  const output=execFileSync(process.execPath,['--experimental-vm-modules','--input-type=module','-e',child,
    new URL('./animation_environment_loader.mjs',import.meta.url).href,JSON.stringify(native)],
    {encoding:'utf8',timeout:20000,stdio:['ignore','pipe','pipe']});
  assert.match(output,/six native panorama.*passed/);
});
