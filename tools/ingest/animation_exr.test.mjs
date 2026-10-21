import test from 'node:test';
import assert from 'node:assert/strict';
import {deflateSync} from 'node:zlib';
import {decodeAnimationExr,inspectAnimationExr} from './animation_exr.mjs';
const u32=n=>{const b=Buffer.alloc(4);b.writeUInt32LE(n);return b;};
const i32=n=>{const b=Buffer.alloc(4);b.writeInt32LE(n);return b;};
const f32=n=>{const b=Buffer.alloc(4);b.writeFloatLE(n);return b;};
const str=s=>Buffer.from(s+'\0');
const attribute=(name,type,data)=>Buffer.concat([str(name),str(type),i32(data.length),data]);
function preprocess(raw) {
  const a=Buffer.alloc(raw.length),h=Math.ceil(raw.length/2);
  for(let i=0;i<raw.length;i++)a[(i&1)?h+(i>>1):i>>1]=raw[i];
  const out=Buffer.from(a);for(let i=1;i<a.length;i++)out[i]=(a[i]-a[i-1]+128)&255;return out;
}
function encodeRle(raw) {
  const out=[]; let i=0;
  const run=at=>{let n=1;while(n<127&&at+n<raw.length&&raw[at+n]===raw[at])n++;return n;};
  while(i<raw.length){const n=run(i);
    if(n>=3){out.push(n-1,raw[i]);i+=n;}
    else {const start=i;i+=n;while(i<raw.length&&i-start<127&&run(i)<3)i++;
      out.push((256-(i-start))&255,...raw.subarray(start,i));}
  }
  return Buffer.from(out);
}
// Independent EXR writer for adversarial fixtures. Pixel bytes are assembled
// scanline/channel-major, ZIP compressed by Node zlib, never by product code.
function fixture({w=16,h=3,compression=0,types={B:2,G:2,R:2,A:2},value=(x,y,c)=>({R:x/4,G:y/2,B:-0.25,A:0.75}[c]),
  minX=0,minY=0,order=0,reverse=false,extra=[],replace={},remove=[],version=2,packed=null,sampling=1}={}) {
  const channels=Object.keys(types).sort(),channelBytes=Buffer.concat([...channels.map(c=>Buffer.concat([str(c),i32(types[c]),Buffer.alloc(4),i32(sampling),i32(1)])),Buffer.of(0)]);
  const entries=[['channels','chlist',channelBytes],['compression','compression',Buffer.of(compression)],
    ['dataWindow','box2i',Buffer.concat([minX,minY,minX+w-1,minY+h-1].map(i32))],
    ['displayWindow','box2i',Buffer.concat([minX,minY,minX+w-1,minY+h-1].map(i32))],
    ['lineOrder','lineOrder',Buffer.of(order)],['pixelAspectRatio','float',f32(1)],
    ['screenWindowCenter','v2f',Buffer.alloc(8)],['screenWindowWidth','float',f32(1)],...extra];
  const header=Buffer.concat([u32(20000630),u32(version),...entries.filter(e=>!remove.includes(e[0])).map(([n,t,d])=>attribute(n,t,replace[n]??d)),Buffer.of(0)]);
  const rows=compression===3?16:1,blocks=[];
  for(let y=0;y<h;y+=rows){const raw=[];
    for(let dy=0;dy<Math.min(rows,h-y);dy++)for(const c of channels)for(let x=0;x<w;x++){
      const v=value(x,y+dy,c),type=types[c];
      if(type===1){const b=Buffer.alloc(2);b.writeUInt16LE(v);raw.push(b);}else raw.push(type===0?u32(v):f32(v));
    }
    const bytes=Buffer.concat(raw);let encoded=bytes;
    if(compression){const candidate=compression===1?encodeRle(preprocess(bytes)):deflateSync(preprocess(bytes));if(candidate.length<bytes.length)encoded=candidate;}
    if(packed)encoded=packed(bytes,y);
    blocks.push({y: minY+y,bytes:Buffer.concat([i32(minY+y),i32(encoded.length),encoded]),raw:bytes.length});
  }
  const table=Buffer.alloc(blocks.length*8),physical=reverse?blocks.toReversed():blocks;let offset=header.length+table.length;
  for(const b of physical){const i=blocks.indexOf(b);table.writeBigUInt64LE(BigInt(offset),i*8);b.offset=offset;offset+=b.bytes.length;}
  return {bytes:Buffer.concat([header,table,...physical.map(b=>b.bytes)]),table:header.length,blocks};
}
const halfFloat=bits=>{const sign=bits&0x8000?-1:1,e=(bits>>10)&31,m=bits&1023;return sign*(e? (1+m/1024)*2**(e-15):m*2**-24);};
const values=image=>Array.from(image.data,halfFloat);
const error=code=>({code:'ANIMATION_EXR_'+code});
for(const compression of [0,1,2,3])test(`decodes actual ${['raw','RLE','ZIPS','ZIP'][compression]} blocks, including partial final ZIP block`,async()=>{
  const f=fixture({h:19,compression, ...(compression===1 ? {value:(x,y,c)=>({R:.5,G:y/2,B:-.25,A:.75}[c])} : {})}),image=await decodeAnimationExr(f.bytes);
  assert.equal(image.width,16);assert.equal(image.height,19);
  for(let y=0;y<19;y++)for(let x=0;x<16;x++)assert.deepEqual(values(image).slice((y*16+x)*4,(y*16+x+1)*4),[compression===1?.5:x/4,y/2,-.25,.75]);
  assert.equal(image.format,'rgba16float');assert.equal(image.clampedComponents,0);
  if(compression)assert.ok(f.blocks.some(b=>b.bytes.length-8<b.raw),'fixture really compresses');
});
test('raw fallback within a ZIP file bypasses predictor and decompression',async()=>{
  const f=fixture({w:1,h:1,compression:3,types:{R:2,G:2,B:2},value:()=>1.2345,packed:raw=>raw});
  const image=await decodeAnimationExr(f.bytes);assert.equal(image.data[0],0x3cf0);assert.equal(image.data[3],0x3c00);
});
test('offset table, decreasing/random physical order and negative data origins preserve top-down pixels',async()=>{
  for(const order of [0,1,2]){const image=await decodeAnimationExr(fixture({compression:3,h:19,minX:-11,minY:-31,reverse:true,order}).bytes);
    assert.deepEqual(image.dataWindow,[-11,-31,4,-13]);assert.deepEqual(values(image).slice(16*18*4,16*18*4+4),[0,9,-.25,.75]);}
});
test('HALF copies finite bits exactly, including signed zero and subnormals',async()=>{
  const bits=[0,0x8000,1,0x8001,0x3ff,0x400,0x3c00,0xbc00,0x7bff,0xfbff];
  const image=await decodeAnimationExr(fixture({w:bits.length,h:1,types:{R:1,G:1,B:1},value:x=>bits[x]}).bytes);
  for(let x=0;x<bits.length;x++)assert.deepEqual([...image.data.slice(x*4,x*4+4)],[bits[x],bits[x],bits[x],0x3c00]);
});
test('mixed UINT/FLOAT/HALF channels are decoded with independent widths',async()=>{
  const image=await decodeAnimationExr(fixture({w:2,h:1,types:{R:0,G:1,B:2,A:1},value:(x,y,c)=>({R:42,G:0x3c00,B:-2,A:0x3800}[c])}).bytes);
  assert.deepEqual(values(image),[42,1,-2,.5,42,1,-2,.5]);
});
test('Y and YA populate all RGB components without rescaling alpha',async()=>{
  const a=await decodeAnimationExr(fixture({w:2,h:1,types:{Y:2,A:2},value:(x,y,c)=>c==='Y'?x+2:.25}).bytes);
  assert.deepEqual(values(a),[2,2,2,.25,3,3,3,.25]);
  assert.deepEqual(values(await decodeAnimationExr(fixture({w:1,h:1,types:{Y:2},value:()=>4}).bytes)),[4,4,4,1]);
});
test('FLOAT narrows with ties-to-even through normal/subnormal boundaries',async()=>{
  const samples=[0,-0,2**-25,3*2**-25,1023.5*2**-24,1+2**-11,1+3*2**-11,2-2**-11];
  const expected=[0,0x8000,0,2,0x400,0x3c00,0x3c02,0x4000];
  const image=await decodeAnimationExr(fixture({w:samples.length,h:1,types:{Y:2},value:x=>samples[x]}).bytes);
  assert.deepEqual(Array.from(image.data).filter((_,i)=>i%4===0),expected);
});
test('signed overflow requires explicit clamp; nonfinite values never clamp',async()=>{
  const f=fixture({w:1,h:1,types:{Y:2},value:()=>-70000});await assert.rejects(decodeAnimationExr(f.bytes),error('VALUE'));
  const image=await decodeAnimationExr(f.bytes,{overflow:'clamp'});assert.deepEqual(values(image),[-65504,-65504,-65504,1]);assert.equal(image.clampedComponents,3);
  for(const value of [Infinity,NaN,-Infinity])await assert.rejects(decodeAnimationExr(fixture({value:()=>value}).bytes,{overflow:'clamp'}),error('VALUE'));
  for(const value of [0x7c00,0xfc00,0x7e00])await assert.rejects(decodeAnimationExr(fixture({types:{Y:1},value:()=>value}).bytes),error('VALUE'));
});
test('metadata is immutable; exposure is reported, never reapplied to radiance',async()=>{
  const image=await decodeAnimationExr(fixture({w:1,h:1,types:{Y:2},value:()=>2,extra:[['exposure','float',f32(3)],['whiteLuminance','float',f32(100)],['envmap','envmap',Buffer.of(0)]]}).bytes);
  assert.equal(image.exposure,3);assert.equal(image.whiteLuminance,100);assert.equal(image.environmentMap,'latlong');assert.equal(values(image)[0],2);
  assert.ok(Object.isFrozen(image));assert.ok(Object.isFrozen(image.dataWindow));assert.ok(Object.isFrozen(image.channels));
});
test('offset views and async input mutation do not alter the owned decoded image',async()=>{
  const f=fixture({compression:3}),buffer=Buffer.concat([Buffer.alloc(7),f.bytes,Buffer.alloc(3)]),part=buffer.subarray(7,7+f.bytes.length);
  const promise=decodeAnimationExr(part);buffer.fill(0);const image=await promise;assert.deepEqual(values(image).slice(0,8),[0,0,-.25,.75,.25,0,-.25,.75]);
});
test('exact output plus scratch budget is enforced before decompression',async()=>{
  const f=fixture({compression:3}),meta=inspectAnimationExr(f.bytes),minimum=meta.byteLength+16*3*4*4*2;
  assert.equal(meta.peakDecodedBytes,minimum);
  await assert.rejects(decodeAnimationExr(f.bytes,{maxDecodedBytes:minimum-1}),error('LIMIT'));
  assert.equal((await decodeAnimationExr(f.bytes,{maxDecodedBytes:minimum})).data.byteLength,meta.byteLength);
  assert.throws(()=>inspectAnimationExr(f.bytes,{maxInputBytes:f.bytes.length-1}),error('LIMIT'));
  assert.throws(()=>inspectAnimationExr(f.bytes,{maxHeaderBytes:32}),error('HEADER'));
  assert.throws(()=>inspectAnimationExr(f.bytes,{maxDimension:15}),error('LIMIT'));
});
for(const version of [1,2|512,2|2048,2|4096,2|256])test(`unsupported version/flags ${version} reject before pixel effects`,()=>{
  assert.throws(()=>inspectAnimationExr(fixture({version}).bytes),error('PROFILE'));
});
test('unsupported compression/channel interpretation and color metadata reject explicitly',()=>{
  for(const opts of [{compression:6},{types:{R:2,G:2}},{types:{R:2,G:2,B:2,Y:2}},{types:{'layer.R':2}},
    {sampling:2},{extra:[['gamma','float',f32(2.2)]]},{replace:{pixelAspectRatio:f32(2)}},
    {extra:[['chromaticities','chromaticities',Buffer.alloc(32)]]},{extra:[['adoptedNeutral','v2f',Buffer.alloc(8)]]}])
    assert.throws(()=>inspectAnimationExr(fixture(opts).bytes),error('PROFILE'));
});
test('mandatory fields, duplicate fields and truncated headers reject',()=>{
  for(const name of ['channels','compression','dataWindow','displayWindow','lineOrder','pixelAspectRatio','screenWindowCenter','screenWindowWidth'])
    assert.throws(()=>inspectAnimationExr(fixture({remove:[name]}).bytes),error('HEADER'));
  assert.throws(()=>inspectAnimationExr(fixture({extra:[['compression','compression',Buffer.of(0)]]}).bytes),error('HEADER'));
  const f=fixture();for(const length of [0,4,8,31,f.table-1,f.table+7])assert.throws(()=>inspectAnimationExr(f.bytes.subarray(0,length)),AnimationExrErrorPredicate);
});
function AnimationExrErrorPredicate(e){return e.name==='AnimationExrError';}
test('all chunk offsets/rows/extents are validated before decoding the first block',()=>{
  for(const mutate of [f=>f.bytes.writeBigUInt64LE(1n,f.table),f=>f.bytes.writeBigUInt64LE(2n**60n,f.table),
    f=>f.bytes.writeBigUInt64LE(f.bytes.readBigUInt64LE(f.table),f.table+8),
    f=>f.bytes.writeInt32LE(123,f.blocks[1].offset),f=>f.bytes.writeInt32LE(0,f.blocks[1].offset+4),
    f=>f.bytes.writeInt32LE(1000000,f.blocks[1].offset+4)]){const f=fixture();mutate(f);assert.throws(()=>inspectAnimationExr(f.bytes),error('DATA'));}
});
test('invalid, truncated, short and overexpanded ZIP streams reject without partial publication',async()=>{
  for(const packed of [()=>Buffer.from([1,2,3]),()=>deflateSync(Buffer.alloc(1)),()=>deflateSync(Buffer.alloc(65536))]){
    await assert.rejects(decodeAnimationExr(fixture({compression:3,packed}).bytes),AnimationExrErrorPredicate);
  }
});
test('malformed RLE runs/literals reject instead of overrunning a block',async()=>{
  for(const packed of [()=>Buffer.of(127),()=>Buffer.of(255),()=>Buffer.of(127,0,127,0,127,0),()=>Buffer.of(0,0)])
    await assert.rejects(decodeAnimationExr(fixture({w:8,h:1,compression:1,packed}).bytes),error('DATA'));
});
test('pre-abort and abort during native ZIP decompression retain the original reason',async()=>{
  const a=new AbortController(),reason=new Error('stop decoding');a.abort(reason);
  await assert.rejects(decodeAnimationExr(fixture().bytes,{signal:a.signal}),e=>e===reason);
  const b=new AbortController(),pending=decodeAnimationExr(fixture({compression:3}).bytes,{signal:b.signal});b.abort(reason);
  await assert.rejects(pending,e=>e===reason);
});
test('unsafe storage and invalid limits fail with structured errors',async()=>{
  const detached=new ArrayBuffer(8);structuredClone(detached,{transfer:[detached]});
  for(const input of [detached,new SharedArrayBuffer(16),new ArrayBuffer(16,{maxByteLength:32}),new Uint16Array(8),null])
    await assert.rejects(decodeAnimationExr(input),error('STORAGE'));
  for(const opt of [{maxDecodedBytes:0},{maxDimension:NaN},{maxHeaderBytes:1.5},{overflow:'ignore'},{signal:{}},{unknown:1}])
    await assert.rejects(decodeAnimationExr(fixture().bytes,opt),AnimationExrErrorPredicate);
});

// Reference files produced by OpenCV 4.13.0 with its native OpenEXR codec.
// These are complete binary EXRs, independent of the JS test writer above.
test('native OpenEXR RLE reference decodes every RGBA pixel',async()=>{
  const data=Buffer.from('di8xAQIAAABjaGFubmVscwBjaGxpc3QASQAAAEEAAQAAAAAAAAABAAAAAQAAAEIAAQAAAAAAAAABAAAAAQAAAEcAAQAAAAAAAAABAAAAAQAAAFIAAQAAAAAAAAABAAAAAQAAAABjb21wcmVzc2lvbgBjb21wcmVzc2lvbgABAAAAAWRhdGFXaW5kb3cAYm94MmkAEAAAAAAAAAAAAAAADwAAABAAAABkaXNwbGF5V2luZG93AGJveDJpABAAAAAAAAAAAAAAAA8AAAAQAAAAbGluZU9yZGVyAGxpbmVPcmRlcgABAAAAAHBpeGVsQXNwZWN0UmF0aW8AZmxvYXQABAAAAAAAgD9zY3JlZW5XaW5kb3dDZW50ZXIAdjJmAAgAAAAAAAAAAAAAAHNjcmVlbldpbmRvd1dpZHRoAGZsb2F0AAQAAAAAAIA/ANMBAAAAAAAA7wEAAAAAAAALAgAAAAAAACcCAAAAAAAAQwIAAAAAAABbAgAAAAAAAHcCAAAAAAAAkwIAAAAAAACvAgAAAAAAAMsCAAAAAAAA7wIAAAAAAAALAwAAAAAAAC8DAAAAAAAASwMAAAAAAABvAwAAAAAAAIsDAAAAAAAArwMAAAAAAAAAAAAAFAAAAP8APoD/ug6A//oOgP/MDoD/uA6AAQAAABQAAAD/AD6A/7oOgP/6DoD//A6A/4gOgAIAAAAUAAAA/wA+gP+6DoD/+g6A/wAOgP+EDoADAAAAFAAAAP8APoD/ug6A//oOgP8CDoD/gg6ABAAAABAAAAD/AD6A/7oOgP/6DoD/BB6ABQAAABQAAAD/AD6A/7oOgP/6DoD/BQ6A/38OgAYAAAAUAAAA/wA+gP+6DoD/+g6A/wYOgP9+DoAHAAAAFAAAAP8APoD/ug6A//oOgP8HDoD/fQ6ACAAAABQAAAD/AD6A/7oOgP/6DoD/CA6A/3wOgAkAAAAcAAAA/wAegP8ADoD/AA6A/7oOgP/6DoD/CA6A/3wOgAoAAAAUAAAA/wA+gP+6DoD/+g6A/wkOgP97DoALAAAAHAAAAP8AHoD/AA6A/wAOgP+6DoD/+g6A/wkOgP97DoAMAAAAFAAAAP8APoD/ug6A//oOgP8KDoD/eg6ADQAAABwAAAD/AB6A/wAOgP8ADoD/ug6A//oOgP8KDoD/eg6ADgAAABQAAAD/AD6A/7oOgP/6DoD/Cw6A/3kOgA8AAAAcAAAA/wAegP8ADoD/AA6A/7oOgP/6DoD/Cw6A/3kOgBAAAAAUAAAA/wA+gP+6DoD/+g6A/wwOgP94DoA=','base64');
  const image=await decodeAnimationExr(data);
  assert.deepEqual([image.width,image.height],[16,17]);
  const actual=values(image);
  for(let y=0;y<17;y++)for(let x=0;x<16;x++)assert.deepEqual(actual.slice((y*16+x)*4,(y*16+x+1)*4),[.5,y/8,-.25,.75]);
});
test('native OpenEXR ZIP reference decodes every RGBA pixel',async()=>{
  const data=Buffer.from('di8xAQIAAABjaGFubmVscwBjaGxpc3QASQAAAEEAAQAAAAAAAAABAAAAAQAAAEIAAQAAAAAAAAABAAAAAQAAAEcAAQAAAAAAAAABAAAAAQAAAFIAAQAAAAAAAAABAAAAAQAAAABjb21wcmVzc2lvbgBjb21wcmVzc2lvbgABAAAAA2RhdGFXaW5kb3cAYm94MmkAEAAAAAAAAAAAAAAADwAAABAAAABkaXNwbGF5V2luZG93AGJveDJpABAAAAAAAAAAAAAAAA8AAAAQAAAAbGluZU9yZGVyAGxpbmVPcmRlcgABAAAAAHBpeGVsQXNwZWN0UmF0aW8AZmxvYXQABAAAAAAAgD9zY3JlZW5XaW5kb3dDZW50ZXIAdjJmAAgAAAAAAAAAAAAAAHNjcmVlbldpbmRvd1dpZHRoAGZsb2F0AAQAAAAAAIA/AFsBAAAAAAAAywEAAAAAAAAAAAAAaAAAAHic7c07CoBADEXRQfy7Yn/jIBYuSVyDKxIL+1sYAjo2ed2BXOKCzfbdnGD7/+//HT7hA97gWegveFX2Dl6UfSLcS30anif1GTwp+xz2yr6AR2VfwkPkvoL7yH0Nd5H7Bm5f7m90DP3mEAAAABcAAAB4nGNooAzsQuP/QuPzoPEr0PgA98E/uQ==','base64');
  const image=await decodeAnimationExr(data);
  assert.deepEqual([image.width,image.height],[16,17]);
  const actual=values(image);
  for(let y=0;y<17;y++)for(let x=0;x<16;x++)assert.deepEqual(actual.slice((y*16+x)*4,(y*16+x+1)*4),[.5,y/8,-.25,.75]);
});
