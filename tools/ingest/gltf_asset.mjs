/** Runtime glTF/GLB bytes, meshopt/Draco geometry and lazy PNG/JPEG/KTX2 image loading. No Node, DOM or
 * GPU dependency and no work at import time. All I/O uses the supplied Fetch API.
 * The byte budget counts unique encoded resources, not decoded meshes/images.
 * Parsed data is owned by the caller; do not mutate it during model construction.
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
 */
import {prepareMeshoptBuffers} from './gltf_meshopt.mjs';
import {prepareDracoMeshes} from './gltf_draco.mjs';
export class GltfAssetError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name='GltfAssetError'; this.code=code; }
}
const fail=(code,message)=>{throw new GltfAssetError('GLTF_ASSET_'+code,message);};
const positive=(n,label)=>{if(!Number.isSafeInteger(n)||n<1)fail('LIMIT',`Invalid ${label}`);return n;};
const abort=signal=>{if(signal?.aborted)throw signal.reason ?? new DOMException('Aborted','AbortError');};
function bytes(value) {
  const buffer=ArrayBuffer.isView(value)?value.buffer:value;
  if(!(buffer instanceof ArrayBuffer)||buffer.resizable)fail('BYTES','Expected fixed, unshared bytes');
  try{return ArrayBuffer.isView(value)?new Uint8Array(buffer,value.byteOffset,value.byteLength):new Uint8Array(buffer);}
  catch{fail('BYTES','Detached input bytes');}
}

/** Parse a complete JSON glTF or GLB 2.0 file. BIN storage is a view into a private
 * copy. Unknown GLB chunks are skipped, but JSON/BIN order and extents are strict.
 */
export function parseGltfAsset(input,{maxBytes=64*1024*1024}={}) {
  positive(maxBytes,'file byte limit');
  const raw=bytes(input);if(raw.length>maxBytes)fail('LIMIT','Model file exceeds byte limit');
  const data=raw.slice(),view=new DataView(data.buffer);let source=data,bin=null;
  if(data.length>=4&&view.getUint32(0,true)===0x46546c67) {
    if(data.length<20||view.getUint32(4,true)!==2||view.getUint32(8,true)!==data.length)fail('GLB','Invalid GLB header');
    let cursor=12,count=0;
    while(cursor<data.length) {
      if(cursor+8>data.length)fail('GLB','Truncated chunk header');
      const length=view.getUint32(cursor,true),kind=view.getUint32(cursor+4,true),end=cursor+8+length;
      if(length%4||end>data.length)fail('GLB','Invalid chunk extent');
      if(count===0) {
        if(kind!==0x4e4f534a)fail('GLB','JSON must be the first chunk');
        source=data.subarray(cursor+8,end);
      } else if(kind===0x4e4f534a)fail('GLB','Duplicate JSON chunk');
      else if(kind===0x004e4942) {
        if(bin!==null||count!==1)fail('GLB','BIN must be the second chunk');
        bin=data.subarray(cursor+8,end);
      }
      count++;cursor=end;
    }
  }
  let json;try{json=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(source));}
  catch{fail('JSON','Invalid UTF-8 model JSON');}
  if(!json||Array.isArray(json)||typeof json!=='object'||json.asset?.version!=='2.0'||
     (json.asset.minVersion!==undefined&&json.asset.minVersion!=='2.0'))fail('VERSION','Expected glTF 2.0');
  return {json,bin};
}
function dataUri(uri,limit) {
  const match=/^data:([^;,]*)(;base64)?,([\s\S]*)$/i.exec(uri);
  if(!match)fail('URI','Unsupported data URI');
  const body=match[3];let output;
  if(match[2]) {
    if(!/^[A-Za-z0-9+/]*={0,2}$/.test(body)||body.replace(/=+$/,'').length%4===1)fail('URI','Invalid base64');
    const length=Math.floor(body.replace(/=+$/,'').length*6/8);
    if(length>limit)fail('LIMIT','Data URI exceeds byte budget');
    let decoded;try{decoded=atob(body);}catch{fail('URI','Invalid base64');}
    if(btoa(decoded).replace(/=+$/,'')!==body.replace(/=+$/,''))fail('URI','Noncanonical base64');
    output=Uint8Array.from(decoded,c=>c.charCodeAt(0));
  } else {
    let length=0;
    for(let i=0;i<body.length;i++,length++) {
      if(body[i]==='%') {if(!/^[\da-f]{2}$/i.test(body.slice(i+1,i+3)))fail('URI','Invalid escaped octet');i+=2;}
      else if(body.charCodeAt(i)>127)fail('URI','Escape non-ASCII data octets');
    }
    if(length>limit)fail('LIMIT','Data URI exceeds byte budget');
    output=new Uint8Array(length);
    for(let i=0,j=0;i<body.length;i++,j++) {
      if(body[i]==='%'){output[j]=parseInt(body.slice(i+1,i+3),16);i+=2;}
      else output[j]=body.charCodeAt(i);
    }
  }
  return {bytes:output,mimeType:match[1].toLowerCase()};
}
function imageType(data) {
  if(data.length>=8&&[137,80,78,71,13,10,26,10].every((v,i)=>data[i]===v))return 'image/png';
  if(data.length>=3&&data[0]===255&&data[1]===216&&data[2]===255)return 'image/jpeg';
  // Type detection only: the texture stage validates the KTX2 container/profile
  // and its bounded transcode output. Keep the encoded bytes for export/cache.
  if(data.length>=12&&[171,75,84,88,32,50,48,187,13,10,26,10].every((v,i)=>data[i]===v))return 'image/ktx2';
  fail('IMAGE','Only PNG/JPEG/KTX2 images are supported');
}

/** source: absolute/relative HTTP(S) URL (relative needs baseURL), or complete
 * JSON/GLB bytes. Buffers are loaded before returning; readImage(index) is lazy
 * so unreachable images are never requested. Identical URLs/data URIs share one
 * fetch, including concurrent image requests. Query strings remain significant.
 *
 * Dependencies are same-origin by default. Extra origins must be explicitly
 * listed; credentials and redirects are never sent/followed. This is not an
 * SSRF sandbox for a caller-supplied fetch implementation. Requests are streamed
 * and stopped at maxResourceBytes/maxBytes even without Content-Length.
 */
export async function loadGltfAsset(source,{
  baseURL,fetch:fetcher=globalThis.fetch,signal,meshoptDecoder=null,dracoDecoder=null,
  maxDecodedBytes=128*1024*1024,maxDecodedBufferBytes=64*1024*1024,
  maxBytes=128*1024*1024,maxResourceBytes=64*1024*1024,maxResources=4096,allowedOrigins=[],
}={}) {
  positive(maxBytes,'total byte limit');positive(maxResourceBytes,'resource byte limit');positive(maxResources,'resource count');
  if(!Array.isArray(allowedOrigins)||allowedOrigins.some(v=>typeof v!=='string'))fail('URI','Invalid allowed origins');
  const origins=new Set(allowedOrigins);let base=null,total=0;
  if(baseURL!==undefined) {try{base=new URL(baseURL);}catch{fail('URI','baseURL must be absolute');}}
  const networkSource=typeof source==='string'||source instanceof URL;
  if(networkSource) {try{base=new URL(source,base ?? undefined);}catch{fail('URI','Model URL must be absolute or have baseURL');}}
  if(base) {
    if(!['http:','https:'].includes(base.protocol)||base.username||base.password)fail('URI','Use credential-free HTTP(S) URLs');
    origins.add(base.origin);
  }
  const cache=new Map();
  function charge(length) {if(length>maxBytes-total)fail('LIMIT','Total input byte budget exceeded');total+=length;}
  async function read(uri) {
    abort(signal);
    if(typeof uri!=='string'||!uri)fail('URI','Expected a resource URI');
    let key=uri;
    if(!/^data:/i.test(uri)) {
      let url;try{url=new URL(uri,base ?? undefined);}catch{fail('URI','External resources need an absolute baseURL');}
      if(!['http:','https:'].includes(url.protocol)||url.username||url.password||!origins.has(url.origin))fail('URI','Resource origin or scheme is not allowed');
      url.hash='';key=url.href;
    }
    if(!cache.has(key)) {
      if(cache.size>=maxResources)fail('LIMIT','Too many resource requests');
      // Start after caching so an injected fetch cannot reenter before deduplication.
      cache.set(key,Promise.resolve().then(async()=>{
        abort(signal);
        if(/^data:/i.test(key)) {
          const result=dataUri(key,Math.min(maxResourceBytes,maxBytes-total));charge(result.bytes.length);return result;
        }
        if(typeof fetcher!=='function')fail('FETCH','No Fetch API supplied');
        const response=await fetcher(key,{signal,credentials:'omit',redirect:'error'});
        const cancel=()=>{try{Promise.resolve(response?.body?.cancel()).catch(()=>{});}catch{}};
        try {
          abort(signal);
          if(!response?.ok||response.redirected)fail('HTTP',`Resource request failed: ${response?.status ?? 'no response'}`);
          if(response.url) {
            let actual;try{actual=new URL(response.url);actual.hash='';}catch{fail('HTTP','Invalid response URL');}
            if(actual.href!==key)fail('HTTP','Unexpected resource redirect');
          }
          const header=response.headers?.get('content-length');
          if(header!==null&&header!==undefined&&/^\d+$/.test(header)&&Number(header)>Math.min(maxResourceBytes,maxBytes-total))fail('LIMIT','Resource exceeds byte budget');
          if(!response.body||typeof response.body.getReader!=='function')fail('FETCH','A streaming response body is required');
        } catch(error) {cancel();throw error;}
        const reader=response.body.getReader(),chunks=[];let length=0,complete=false;
        const onAbort=()=>{try{Promise.resolve(reader.cancel(signal.reason)).catch(()=>{});}catch{}};
        signal?.addEventListener('abort',onAbort,{once:true});
        try {
          while(true) {
            abort(signal);const chunk=await reader.read();abort(signal);
            if(chunk.done){complete=true;break;}
            const data=bytes(chunk.value);
            if(data.length>maxResourceBytes-length)fail('LIMIT','Stream exceeds resource byte budget');
            charge(data.length);length+=data.length;chunks.push(data.slice());
          }
          const data=new Uint8Array(length);let at=0;for(const chunk of chunks){data.set(chunk,at);at+=chunk.length;}
          return {bytes:data,mimeType:(response.headers?.get('content-type') ?? '').split(';')[0].trim().toLowerCase()};
        } finally {
          signal?.removeEventListener('abort',onAbort);
          if(!complete)try{await reader.cancel();}catch{}
          reader.releaseLock();
        }
      }));
    }
    return cache.get(key);
  }
  abort(signal);
  let input;
  if(networkSource)input=(await read(base.href)).bytes;
  else {input=bytes(source);if(input.length>maxResourceBytes)fail('LIMIT','Model file exceeds resource limit');charge(input.length);}
  const parsed=parseGltfAsset(input,{maxBytes:maxResourceBytes}),bin=parsed.bin;
  let json=parsed.json;
  const definitions=json.buffers ?? [],images=structuredClone(json.images ?? []);
  let views=structuredClone(json.bufferViews ?? []);
  if(!Array.isArray(definitions)||!Array.isArray(images)||!Array.isArray(views)||definitions.length>maxResources||images.length>maxResources||views.length>maxResources*16)fail('LIMIT','Invalid or excessive resources');
  // Expansion and fallback decisions precede dependency I/O, not just decoding.
  const meshopt=prepareMeshoptBuffers(json,{decoder:meshoptDecoder,binaryBuffer:bin!==null,
    maxEncodedBytes:maxBytes,maxDecodedBytes,maxDecodedBufferBytes,maxBufferViews:maxResources*16});
  const dracoOptions={decoder:dracoDecoder,maxEncodedBytes:maxBytes,
    maxDecodedBytes:maxDecodedBytes-meshopt.decodedBytes,maxDecodedBufferBytes,maxPrimitives:maxResources};
  const draco=prepareDracoMeshes(json,dracoOptions);
  const skipped=new Set([...meshopt.skippedBuffers,...draco.skippedBuffers]);
  // Unused fallback declarations may be large; never allocate/fetch fake bytes.
  for(let i=0;i<definitions.length;i++) {
    const item=definitions[i];
    if(!item||!Number.isSafeInteger(item.byteLength)||item.byteLength<1||(!skipped.has(i)&&item.byteLength>maxResourceBytes))fail('BUFFER','Invalid buffer byteLength');
  }
  let buffers=[];
  for(let i=0;i<definitions.length;i++) {
    abort(signal);if(skipped.has(i)){buffers.push(null);continue;}
    const item=definitions[i];let data;
    if(item.uri===undefined) {
      if(i!==0||bin===null||bin.length<item.byteLength||bin.length-item.byteLength>3)fail('BUFFER','Missing or invalid GLB BIN buffer');
      for(let j=item.byteLength;j<bin.length;j++)if(bin[j]!==0)fail('BUFFER','Nonzero BIN padding');
      data=bin;
    } else {
      data=(await read(item.uri)).bytes;
      if(data.length<item.byteLength)fail('BUFFER','Truncated buffer');
    }
    buffers.push(data.subarray(0,item.byteLength));
  }
  const decoded=await meshopt.decode(buffers,{signal});abort(signal);
  // Meshopt may relocate the very bufferView carrying the Draco bitstream.
  // Rebind the metadata plan to those decoded views, without decoding twice.
  const preparedDraco=decoded.json===json?draco:prepareDracoMeshes(decoded.json,dracoOptions);
  const geometry=await preparedDraco.decode(decoded.buffers,{signal});abort(signal);
  const sourceJson=parsed.json;
  json=geometry.json;buffers=geometry.buffers;views=structuredClone(json.bufferViews ?? []);
  const imageCache=new Map();
  function readImage(index) {
    try {
      abort(signal);
      if(!Number.isSafeInteger(index)||index<0||index>=images.length)fail('IMAGE','Invalid image index');
    }catch(error){return Promise.reject(error);}
    if(!imageCache.has(index))imageCache.set(index,Promise.resolve().then(async()=>{
      const image=images[index];
      if(!image||((image.uri===undefined)===(image.bufferView===undefined)))fail('IMAGE','Image needs exactly one URI or bufferView');
      if(image.extensions&&Object.keys(image.extensions).length)fail('IMAGE','Image extensions need the source loader');
      let data,declared=image.mimeType;
      if(image.uri!==undefined) {
        const resource=await read(image.uri);data=resource.bytes;
        if(declared===undefined&&/^data:/i.test(image.uri))declared=resource.mimeType;
      } else {
        const view=Number.isSafeInteger(image.bufferView)&&image.bufferView>=0?views[image.bufferView]:null;
        const buffer=view&&Number.isSafeInteger(view.buffer)&&view.buffer>=0?buffers[view.buffer]:null;
        const offset=view?.byteOffset ?? 0,length=view?.byteLength;
        if(!buffer||!Number.isSafeInteger(offset)||offset<0||!Number.isSafeInteger(length)||length<1||length>buffer.length-offset||view.byteStride!==undefined||Object.keys(view.extensions ?? {}).length)fail('IMAGE','Invalid embedded image range');
        if(declared===undefined)fail('IMAGE','Embedded image needs mimeType');
        data=buffer.subarray(offset,offset+length);
      }
      abort(signal);const mimeType=imageType(data);
      if(declared!==undefined&&declared!==mimeType)fail('IMAGE','Image MIME type disagrees with image bytes');
      return Object.freeze({bytes:data,mimeType});
    }));
    return imageCache.get(index);
  }
  abort(signal);
  return Object.freeze({json,sourceJson,buffers:Object.freeze(buffers),readImage,
    decodedBytes:decoded.decodedBytes+geometry.decodedBytes,decodedBufferViews:decoded.decodedBufferViews,
    decodedPrimitives:geometry.decodedPrimitives,
    get bytesLoaded(){return total;}});
}
