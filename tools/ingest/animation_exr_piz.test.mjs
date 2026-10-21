import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {inflateSync} from 'node:zlib';
import {decodeAnimationExr, inspectAnimationExr, AnimationExrError} from './animation_exr.mjs';
import {decodeExrPiz, EXR_PIZ_TABLE_BYTES} from './animation_exr_piz.mjs';
// Genuine files encoded by OpenCV 4.13's native OpenEXR codec; zlib below only
// shrinks the checked-in fixture text, not the EXR compression being tested.
// Goldens are SHA256 of native-decoded RGBA narrowed to little-endian float16.
// Odd fixtures: BGRA[c] = (x%16)/8 + (y%8)/4 + c/2 - 1, 65x35.
// Full LUT: Y float bits = 0x3f800000 | (pixelIndex%65536), 2048x32.
// Zero fixture: 1x65 HALF RGBA, covering one-wide and partial raw-fallback blocks.
const fixtures = {
  "half_odd": {
    "encoded": "eNrtV1toE0EUvdv0EW1tVZBUTTVYEcWP2Afa2qbdqPgAQVHBD6G4STbNQsym3dBWojhq1YooPn7aD+sL/DVSsfURiKjYj9Lav9RKfaI/a/FrFVR0tm3SdBaTICRRnAN7d7L3zt1z792ZO2kxlzFZAGB3cR4P75bwwC1IPtiGn1mBgUkwU9cGzZMtmie7NE/ALh7wNvOSJIieGWNVme3gfNxeweMQW8EmtpULMBemweJrGb4cguR1cwcT2rkFD7+j2cE3x4wmKHiFNt5tlby83beL8wkiON0i54PsidmoXrI387xn0v1G3uPD01rKnaCPeUWsyV7B4XMRLmDfVNzvl0xPOlqCRS4yMJBZMH/5/HTrM/U7k/dMf4MUFBQUFBQUFBQUFBT/A+j//wzrv+ZPm4axdcPOyV96butKpDdd3G66XmMK5AHSfy/wzv586yfFP4hDV0Kb5ashfSmApfuprCh+Q0+YkRVzY29PONj9aIVzJuz3Xo631lwe/ChFBl7jAAGzaFMUQjDbCaNnEunpTQWJkU8qIb8l2D02JKqElpOvmklFHei6CJz4JDeMPZ0pQqsIo85xjScycqffckYlNJWZsSGb4s8mX3buIxlWSCKCWhMMnhkNEyKHDL1kkPTUTsY+hImMxqZo/yXCSRlZK6mvkIzK0Hvb7ycE3ExcdTLyTiNrQtd1pqLGO5FBoAmOZXUUIEv3cGQQst7YfMo40D98Y+M5p8M1fO1ZRXml6/m7rq7iKrTHPOfy4Ou3zM6Ha1trjnfkbPg88nK80LQg/3Gj1JdbWfXB6VQnD/Sr9l2LjDizpbPk1ZiwVs4n49R8KK2a4l2YYBmcYqu9/wPkY3M9TLOfNvIIWEdRwNDzwlSeF84fkSE7l/3G+C14HRxGmHsZZovpEjyKGxar+16bG7s+8aYHezacVBcpuY7XhbfYlC/KPLxX/GhvMYtHuBK8Z/zJZniBOV83NuRC/UtFvE4dRb0TjEdHZN1kTnBqcW6bsLOzhdXYPJZVDNnT5HZ1OErr1TTXY4n3/NfVJMO5C3Vyjqx0FBxVshT/JprGP0pjZ2m0R+mjLetJ4m6l6fecpjkYegPGxCnT9HtNt7IlEdalZHpnYeL1qmlojdH0RLv67SSOOtvINGtOFljkJe7laHfCo04SvXx5MgcvXRJHFM1pyITP+cW5WFRBNayHGqhFtWBBFqhDdVCP6oFlWcSG2DToH8IDuA/3oA/uwh0IYE5roBLWRmemXv87TYRz6vW/00SylXp9vBpZWWsa9PFqpM5MvT5ejVTOqdfHqxHOVhr0cWqErCFr6vV0Nf4Nu/Ivmq+ldg==",
    "sha256": "c1c3d3e52e1a4899232fcb174bbabcc2a5c1c0edc31f31581cc8ad7e65c095ce"
  },
  "float_odd": {
    "encoded": "eNrtWl9oWlcYP1dNVUxNmtJ1Yd1iiy2DyWrqSDsm5rYba8Y6Bl0hjI2i808q3KlTaQwGZtOUUMjAtxmEMPaw16bzZYOmvSbQh2HoYC8NjGD24FMhQmDpw2D7zr13Tbx+HvHiwyb3k5/3u8f7nXN+3/m+cz7Fm2dHOQMhJHgjEIuFhRQoQjSVJh9A20VCP6HCKbjU1HK5qeVqUwsJxr9MJMOpVDQea9Dph6ZQIB2YjMZC8WnyRTxzLkoGyb7wgFOAUDSVEAIzbZ8TorHwx8lQOHlAk6aQiGbCwsVUIhxMXw2ko3ESEeKBNDFJ1rnxVDAZDsfk7t8Nx9JgdvNchFgODHHwkcloKH1D1QX5jJMf/OPkvtHXDkJEc/04RxqFQ66NUL1UcoTooosuuuiiiy666KKLLrrooosu/zXp+Pv/+4P7jz6Fll+S8p3pq7uuvNX5/TfOH7NO0ULy1r/6c7b6o791+R/K7D3x8rP7xOLhc/2bi5t3iqNFr3J1c/StFPFMKdeIaXUnu7Wneru/HPQ+nlncHa55KlOFJ6ONt277HmK0trEssCAeQwZabTWGfDsXR0x2xoJjM6Xh0kYBen0s7MpkKMNaJVKxYhYsNqPL/lexmd2BLlnIvYEMlGWxWdw1YSZbQGJNIYNdC0OYEYtQ0Cu+g41kA7+xQCaRgf5kERquWbH4eSAvxhN0pdynOw+E3DQ2s5egSxbIPWSgZyw2nspRLH78mQmr1bDG/04olSHDJX4lf554IH3M87l83uU8LNJM23Y4nTarMWEE3v6hkYGF8jb4wjcycHt9dScz6xsxL2S36s9nfQOQQdtiPX98rnZYWq626gnJBjp5jQZr/bmXLvB2PUP9sj4XqRwbC4C7/Uulvo8Kn4LDzwjlTyQfWIvua1K8zmtIdHSlFJaz+4Rbq7kJNWtQKIX1bZnDgnGNv24rvEVHbauSRCPvap1OSPLy6o79BVOWO7Tk0reY51qte8tI4U6qqa/LPAYWZCInBqxc5k3/b3TotirXSF6KsOyW5Oq9BxeaVx/xyYed71+G65j7FJYHCLdUCyY1a3WWJLzOvurPt6/Q8Gurmht5N+ZIlZEML9wRONv5FoEeMX4yf+snuyC6pHiIGYtuG90F8FY7OmEluMupQ/J4DGUNW4jNovs9id1SySQRX1I3dO8ogQg3S3s7Q2Gnq5LTwzWGgk63sjl3puKC82WjcEqqLJoa0Ixtc45x6NE8dKj6donQeglbx8zEK7laxUDj2ogmorK+ic/LP8gbKkNB5xypHJHCNCKUj0oR3NTQtfqtnj/PScc0Q2lzDMlnFZRtrRX0FCqN+e9KBYgnYoGqdKroUTd0r2awqAvgpoZuHRFtyxM05J6ydp6pgqHzcktDMqBBtULXgQW7BsehXwpoHe9S6nns2rVSpm2NP92579Cik/2dRUNw30LPLboJsmDU4Lg+jA7vIIS4+6UfD8YJcQB4gH9c+s8CyQO+A6wARMCvgCqgDqD/qrAABgEv871hn3sEtgAe4AAQgPgQ2gHkofKTiw/uAaIPGbMH7Du2Ufm5F+w7tWmKrR6w15pHDsDrfG/Ya84jZcxesNeaR//6uRfsNeeRElu9YK81j9yAC3xv2Otno3426rUytf8H07OZvA==",
    "sha256": "c1c3d3e52e1a4899232fcb174bbabcc2a5c1c0edc31f31581cc8ad7e65c095ce"
  },
  "gray_2048x32": {
    "encoded": "eNrtnEFIU3Ecx//vbU7NJFFYpYQdQgORNhHpkJh5iGBgSGDZxc098cHcxjbMvGTYIRAiqsNO0ml66uQ9JYRuHQYRnuymNyPrEKQ99xz6D3o/kpDEz5fvvs79vvtv3/8b7PF7//8mroQNUyk1MhZNJq1E1rmTsLM51eA8dk/tVnZh7N3USGo8nbGyWTuV1O7vFv3xaC46aCfjqQcqlprssFWd2sdOpVLNzt+4nU0nog9FX8JOWv2ZuJU5cK/0FtL2pJXozaatkdxANGen1GgiFc0pf+nZ0z3ZkYxlJd3h+6xkznnaRMeoqjrwEgctg3Y8N/bbEOqCsW/Ovym9q+btHQAAAAAAAAAAAAAAAAAAAAAAAAAcdzx/5q4H2NxR6r6plBl1lwu0bT7pOd/35dU/kJule0cTZ731cM/7OZn6c/HH50cH//02o1U/Vh7pAetr+/7irdG6vFQXOpoX3I7/7aH2lF6lLqqXwXwwv7g0HrYf27MRn3OriQwNdzkPLa4U1A2jpT3UHranV2NFsxjMB4qaXT01HfNQrGg45opCe6i2oNmNc353oPK4TbrdvF7tDlQet0u3+1/XuwOVxx3X7dXTl9yB9sZVq7q9YfqWlqf8umX7ZSH+gBB/QYh/V4i/5B2/QohfI8Q/K8QPCfENoR4Tpue9MD157+nxCZ+OgDA9tcL0NAnTc1WInxHirwvxPwnxhU9HpRD/jBBfCfFNIX5AiF/nHV9d9o6vhr3jqzXv+Kbyju8X4p8S4geF+CEh/pgQf807viHE9wvxq4T49UL8FiH+bSH+gnd8Qzj6phA/IMQ/LcRvFOJ3CfHTQvx14egrYXqET4fPe3o+XPsK4Ynlu/5I5xaCIMgJFL4B4EnmcqG9O2zPRIamYsVAfnFrpVAbtjdmI42xonOC3Lky75w127P9zol0MTiXX8SOHTt27MfEzjccpL+BIAhCfwPCk8SlO6XLhN3uRb8K99rh3pVCSpQoUaJE6X8t8Q0O6d8gCILQv4GQ9Slc2cWOHTt21qdASH8DQRCE/gaEx2h9SqO2O31D31uvbaXv1ze+B7R97lP6JvdObQ+/viPf0Dbgz+nb5Wu13fFb+k8BaDv/5/V9/D7t3XXre/IJSUhCEpKQhDxcSM4UIH0iBEEQ+kQQsg6GK8jYsWPHzjoYCOlvIAiC0N+AkN9poUSJEiVKlPidFgjp3yAIgtC/gZD1KdixY8eOnfUpENLfQBAEob8BYWl9yi8fuPpj",
    "sha256": "e826637dc7c216c63ba8af1f0c62fb9b70bf4803ced2d5cd447be90e4fffd640"
  },
  "half_zero_one": {
    "encoded": "eNqNUDsOwjAMdVAHxNSRDW6A6AX4DYgJiYURhcQVlkJSNRGfjZ3jcAFuBmmJRKADWHqK/fKeZfswGLIWAIgd1xqV9Yki62DhuQkweAULmDaYeYNZNRgQZl+UaC0Z/ZFXn4nkjq9JS3OErTllBCl8xthDki0UP//UKdK4LCWWUVaPUNAJ1cQWKNyKOzKQK8MdJLXzMrKiRNSv9jPUztsOWQ7tqH0sWZN0u68WIMLe1/De2Nvc93j0ALrRaaqzd0KdpJt7/09dtWc82BNWM0mK",
    "sha256": "20aa497d9bd4c19e851e3df6e386700faada213db38acf7679f6365832830b3d"
  }
};
const fixture=name=>new Uint8Array(inflateSync(Buffer.from(fixtures[name].encoded,'base64')));
const fail=(code,message)=>{throw new AnimationExrError('ANIMATION_EXR_'+code,message);};
const dataError={code:'ANIMATION_EXR_DATA'};
function layout(bytes){
  const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),attributes={};let at=8;
  const text=()=>{const start=at;while(bytes[at])at++;return new TextDecoder().decode(bytes.subarray(start,at++));};
  while(bytes[at]){const name=text(),type=text(),size=v.getUint32(at,true);at+=4;attributes[name]={type,at,size};at+=size;}
  return {attributes,table:at+1};
}
function block(bytes,index=0){const table=layout(bytes).table,v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),at=Number(v.getBigUint64(table+index*8,true));return bytes.subarray(at+8,at+8+v.getUint32(at+4,true));}
for(const name of Object.keys(fixtures))test('native PIZ '+name+' matches all RGBA16F bytes',async()=>{
  const input=fixture(name),before=input.slice(),image=await decodeAnimationExr(input);
  assert.equal(image.compression,'piz');assert.equal(image.clampedComponents,0);
  assert.equal(createHash('sha256').update(new Uint8Array(image.data.buffer)).digest('hex'),fixtures[name].sha256);
  assert.deepEqual(input,before);assert.ok(Object.isFrozen(image));
});
test('full 16-bit LUT and independent FLOAT word planes reconstruct before RGBA narrowing',()=>{
  const input=fixture('gray_2048x32'),packed=block(input),v=new DataView(packed.buffer,packed.byteOffset,packed.byteLength);
  assert.equal(v.getUint16(0,true),0);assert.equal(v.getUint16(2,true),8191);
  const result=decodeExrPiz(packed,2048,32,[{bytes:4}],fail),raw=new DataView(result.buffer);
  for(let i=0;i<65536;i++)assert.equal(raw.getUint32(i*4,true),0x3f800000+i);
});
test('PIZ needs no platform ZIP decompressor and has no implicit dependency on it',async()=>{
  const old=globalThis.DecompressionStream;
  try{globalThis.DecompressionStream=undefined;assert.equal((await decodeAnimationExr(fixture('half_odd'))).compression,'piz');}
  finally{globalThis.DecompressionStream=old;}
});
test('PIZ output, two block buffers and typed tables share the preflight memory bound',async()=>{
  const input=fixture('half_odd'),meta=inspectAnimationExr(input);
  assert.equal(meta.peakDecodedBytes,65*35*8+2*65*32*8+EXR_PIZ_TABLE_BYTES);
  const bad=input.slice(),payload=block(bad);payload[0]=255;payload[1]=255;payload[2]=255;payload[3]=255;
  assert.throws(()=>inspectAnimationExr(input,{maxDecodedBytes:meta.peakDecodedBytes-1}),{code:'ANIMATION_EXR_LIMIT'});
  await assert.rejects(decodeAnimationExr(bad,{maxDecodedBytes:meta.peakDecodedBytes-1}),{code:'ANIMATION_EXR_LIMIT'});
  await assert.rejects(decodeAnimationExr(bad,{maxDecodedBytes:meta.peakDecodedBytes}),dataError);
  assert.equal((await decodeAnimationExr(input,{maxDecodedBytes:meta.peakDecodedBytes})).byteLength,meta.byteLength);
});
test('invalid PIZ chunk offsets, lengths and rows reject before entropy decoding',()=>{
  const original=fixture('half_odd'),table=layout(original).table;
  for(const edit of [
    (v)=>v.setBigUint64(table,BigInt(table),true),
    (v)=>v.setBigUint64(table+8,v.getBigUint64(table,true),true),
    (v)=>v.setInt32(Number(v.getBigUint64(table,true)),1,true),
    (v)=>v.setInt32(Number(v.getBigUint64(table,true))+4,65*32*8+1,true),
  ]){const data=original.slice();edit(new DataView(data.buffer));assert.throws(()=>inspectAnimationExr(data),dataError);}
});
test('caller input may be transferred immediately and cancellation never publishes partial pixels',async()=>{
  const input=fixture('half_odd'),pending=decodeAnimationExr(input);structuredClone(input,{transfer:[input.buffer]});
  assert.equal(createHash('sha256').update(new Uint8Array((await pending).data.buffer)).digest('hex'),fixtures.half_odd.sha256);
  const controller=new AbortController(),reason=new Error('cancel decode');controller.abort(reason);
  await assert.rejects(decodeAnimationExr(fixture('half_odd'),{signal:controller.signal}),error=>error===reason);
});
// Tiny, independently described prefix codes isolate malformed-stream contracts.
// The image layer's native-codec tests above validate the complete wavelet route.
function packed(lengths,payload,values=[]){
  const tableBits=lengths.map(v=>v.toString(2).padStart(6,'0')).join('');
  const encode=s=>Uint8Array.from({length:Math.ceil(s.length/8)},(_,i)=>parseInt(s.slice(i*8,i*8+8).padEnd(8,'0'),2));
  const table=encode(tableBits),data=encode(payload),bitmap=values.length?new Uint8Array((Math.max(...values)>>>3)+1):new Uint8Array();
  for(const value of values)bitmap[value>>>3]|=1<<(value&7);
  const out=new Uint8Array(8+bitmap.length+20+table.length+data.length),v=new DataView(out.buffer);
  v.setUint16(0,bitmap.length?0:8191,true);v.setUint16(2,bitmap.length?bitmap.length-1:0,true);out.set(bitmap,4);
  const h=8+bitmap.length;v.setUint32(h-4,20+table.length+data.length,true);
  v.setUint32(h,0,true);v.setUint32(h+4,lengths.length-1,true);v.setUint32(h+8,table.length,true);v.setUint32(h+12,payload.length,true);
  out.set(table,h+20);out.set(data,h+20+table.length);return out;
}
const decode=(input,count)=>decodeExrPiz(input,1,count,[{bytes:2}],fail);
const binary=(n,width)=>n.toString(2).padStart(width,'0');
test('literal and repeat codes preserve exact output extents, including a legal zero repeat',()=>{
  assert.deepEqual([...decode(packed([1,1],'01'+binary(7,8)),8)],new Array(16).fill(0));
  assert.deepEqual([...decode(packed([1,1],'01'+binary(0,8)),1)],[0,0]);
});
test('58-bit Huffman codes and their repeat marker decode exactly without JS shift truncation',()=>{
  const lengths=[...Array.from({length:57},(_,i)=>i+1),58,58];
  const input=packed(lengths,'0'.repeat(58)+'0'.repeat(57)+'1'+binary(2,8),Array.from({length:58},(_,i)=>i+1));
  assert.deepEqual([...new Uint16Array(decode(input,3).buffer)],[57,57,57]);
});
for(const [name,lengths,payload,count] of [
  ['repeat before literal',[1,1],'1'+binary(1,8),1],
  ['repeat exceeds output',[1,1],'01'+binary(2,8),2],
  ['repeat missing count bits',[1,1],'010',2],
  ['too few literals',[1,1],'0',2],
  ['too many literals',[1,1],'00',1],
  ['unused code',[2,2],'11',1],
  ['oversubscribed table',[1,1,1],'0',1],
  ['prefix collision',[1,0,3],'0',1],
  ['missing run marker',[1,0],'0',1],
  ['zero run escapes alphabet',[1,59],'0',1],
])test('rejects malformed PIZ '+name,()=>assert.throws(()=>decode(packed(lengths,payload),count),dataError));
test('PIZ byte extents, symbol bounds, reserved header and inverse range values are checked',()=>{
  const source=packed([1,1],'0');
  for(const edit of [
    v=>v.setUint16(2,8192,true), v=>v.setUint32(4,0xffffffff,true),
    v=>v.setUint32(8,65537,true), v=>v.setUint32(12,65537,true),
    v=>v.setUint32(16,0xffffffff,true), v=>v.setUint32(20,0xffffffff,true),
    v=>v.setUint32(24,1,true),
  ]){const input=source.slice();edit(new DataView(input.buffer));assert.throws(()=>decode(input,1),dataError);}
  // Codes: literal 0 -> 1; literal 1 -> 00; run marker 2 -> 01.
  assert.throws(()=>decode(packed([1,2,2],'00'),1),dataError);
  for(let size=0;size<source.length;size++)assert.throws(()=>decode(source.subarray(0,size),1),dataError);
});
