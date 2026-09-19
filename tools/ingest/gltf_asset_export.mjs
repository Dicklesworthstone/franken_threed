/** Self-contained GLB export of a loaded glTF asset, not a sampled pose.
 * Preserve authored scenes, local transforms, skins, morph targets, animations,
 * cameras, lights and materials. Only resource storage is repacked: buffer indices
 * become one BIN buffer, while bufferView/accessor/image/node indices stay fixed.
 * Use loadGltfAsset's normalized json/buffers after meshopt/Draco decoding, not
 * sourceJson with its original compression references. No GPU or codec executes.
 */
export class GltfAssetExportError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name='GltfAssetExportError'; this.code=code; }
}
const fail=(code,message)=>{throw new GltfAssetExportError('GLTF_EXPORT_'+code,message);};
const align=n=>Math.ceil(n/4)*4;
// These extensions reference stable node/accessor/texture/image indices, never
// buffers directly. Unknown extensions cannot be blindly carried through a buffer
// relocation: an opaque extension might hide another buffer offset or URI.
const EXTENSIONS=new Set(['KHR_materials_unlit','KHR_materials_emissive_strength','KHR_materials_clearcoat',
  'KHR_texture_transform','KHR_texture_basisu','KHR_lights_punctual','KHR_mesh_quantization','EXT_mesh_gpu_instancing']);
function integer(n,min,max,label) {
  if(!Number.isSafeInteger(n)||n<min||n>max)fail('LIMIT',`Invalid ${label}`);
  return n;
}
function object(value,label) {
  if(!value||typeof value!=='object'||Array.isArray(value))fail('SHAPE',`Expected ${label} object`);
  return value;
}
function fields(value,allowed,label) {
  object(value,label);
  for(const key of Object.keys(value))if(!allowed.includes(key))fail('OPTIONS',`Unsupported ${label}: ${key}`);
}
const abort=signal=>{if(signal?.aborted)throw signal.reason??new DOMException('Aborted','AbortError');};
function bytes(value) {
  const buffer=ArrayBuffer.isView(value)?value.buffer:value;
  if(!(buffer instanceof ArrayBuffer)||buffer.resizable)fail('STORAGE','Resources require fixed unshared storage');
  try{return ArrayBuffer.isView(value)?new Uint8Array(buffer,value.byteOffset,value.byteLength):new Uint8Array(buffer);}
  catch{fail('STORAGE','Resource storage is detached');}
}
function mime(data) {
  if(data.length>=8&&[137,80,78,71,13,10,26,10].every((n,i)=>data[i]===n))return 'image/png';
  if(data.length>=3&&data[0]===255&&data[1]===216&&data[2]===255)return 'image/jpeg';
  if(data.length>=12&&[171,75,84,88,32,50,48,187,13,10,26,10].every((n,i)=>data[i]===n))return 'image/ktx2';
  fail('IMAGE','Expected encoded PNG, JPEG or KTX2 image bytes');
}
// Capture only JSON data properties. Do not call user getters/toJSON while
// snapshotting. Cycles, nonfinite numbers and unsupported JavaScript values must
// not be silently erased by JSON.stringify. Extras are retained as data too.
function snapshot(value,limit) {
  let units=0;const active=new Set();
  const charge=n=>{units+=n;if(units>limit)fail('LIMIT','Model JSON exceeds its byte budget');};
  function copy(input,depth) {
    if(depth>128)fail('LIMIT','Model JSON nesting exceeds 128 levels');
    charge(1);
    if(input===null||typeof input==='boolean')return input;
    if(typeof input==='number') {if(!Number.isFinite(input))fail('JSON','Nonfinite JSON number');return input;}
    if(typeof input==='string'){charge(input.length);return input;}
    if(!input||typeof input!=='object'||active.has(input))fail('JSON','Expected acyclic JSON data');
    if(!Array.isArray(input)&&![Object.prototype,null].includes(Object.getPrototypeOf(input)))fail('JSON','Expected plain JSON objects');
    active.add(input);
    const array=Array.isArray(input),out=array?[]:Object.create(null);
    if(array)integer(input.length,0,limit,'JSON array length');
    const properties=Object.getOwnPropertyDescriptors(input);
    const keys=array?Array.from({length:input.length},(_,i)=>String(i)):Object.keys(properties).filter(k=>properties[k].enumerable);
    for(const key of keys) {
      const property=properties[key];
      if(!property||!Object.hasOwn(property,'value'))fail('JSON','JSON accessors and sparse arrays are not supported');
      charge(key.length);const child=copy(property.value,depth+1);
      Object.defineProperty(out,key,{value:child,writable:true,enumerable:true,configurable:true});
    }
    active.delete(input);return out;
  }
  return copy(value,0);
}
function encodeJson(value) {
  if(typeof value==='number')return Object.is(value,-0)?'-0':JSON.stringify(value);
  if(Array.isArray(value))return '['+value.map(encodeJson).join(',')+']';
  if(value&&typeof value==='object')return '{'+Object.entries(value).map(([k,v])=>JSON.stringify(k)+':'+encodeJson(v)).join(',')+'}';
  return JSON.stringify(value);
}
function extensions(json) {
  for(const field of ['extensionsUsed','extensionsRequired'])if(json[field]!==undefined) {
    if(!Array.isArray(json[field])||json[field].some(n=>typeof n!=='string'||!EXTENSIONS.has(n)))
      fail('EXTENSION','Export requires a supported, resource-closed extension profile');
  }
  function inspect(value) {
    if(!value||typeof value!=='object')return;
    for(const [key,child]of Object.entries(value)) {
      if(key==='extras')continue; // Application metadata is not a loader resource.
      if(key==='extensions') {
        object(child,'extensions');
        for(const name of Object.keys(child))if(!EXTENSIONS.has(name))fail('EXTENSION',`Cannot relocate opaque extension: ${name}`);
      }
      // Core buffer/image URIs are checked separately. Any URI under one of the
      // allowed extension records is still unsupported, not an offline success.
      inspect(child);
    }
  }
  inspect(json);
}
function wait(pending,signal) {
  if(!signal)return Promise.resolve(pending);
  return new Promise((resolve,reject)=>{
    const cleanup=()=>signal.removeEventListener('abort',onAbort);
    const onAbort=()=>{cleanup();reject(signal.reason??new DOMException('Aborted','AbortError'));};
    signal.addEventListener('abort',onAbort,{once:true});
    Promise.resolve(pending).then(value=>{cleanup();resolve(value);},error=>{cleanup();reject(error);});
    if(signal.aborted)onAbort();
  });
}

/** asset: {json, buffers, readImage(index)} from loadGltfAsset.
 * All JSON and supplied buffer bytes are captured before the first image callback.
 * URI images are resolved sequentially through readImage, not a hidden fetch;
 * identical URI strings share one result. Embedded image ranges stay in place.
 * Every scene/image is retained, including resources not used by the active scene.
 * maxBytes bounds the finished GLB and binary staging independently, not total
 * process memory. maxJsonBytes bounds source/output JSON; maxResources bounds
 * each buffer/image table (bufferViews may have sixteen times as many entries).
 * A caller's image resolver may ignore abort; late settlements are still observed.
 * Images are signature-checked, not decoded or fully validated by a codec.
 * Unknown/compressed extension records fail before I/O. The asset loader removes
 * successfully decoded geometry compression; skipped unused buffer slots may be
 * null, but a referenced buffer must always have real bytes. No zero stand-ins.
 */
export async function exportGltfAssetGLB(asset,options={}) {
  fields(options,['signal','maxBytes','maxJsonBytes','maxResources'],'export option');
  const {signal,maxBytes=128*1024*1024,maxJsonBytes=Math.min(maxBytes,16*1024*1024),maxResources=4096}=options;
  integer(maxBytes,20,0xffffffff,'GLB byte limit');integer(maxJsonBytes,1,maxBytes,'JSON byte limit');
  integer(maxResources,1,65536,'resource count limit');abort(signal);object(asset,'loaded asset');
  const json=snapshot(asset.json,maxJsonBytes);
  if(!json||Array.isArray(json)||json.asset?.version!=='2.0'||
    (json.asset.minVersion!==undefined&&json.asset.minVersion!=='2.0'))fail('VERSION','Expected glTF 2.0');
  if(new TextEncoder().encode(encodeJson(json)).length>maxJsonBytes)fail('LIMIT','Source JSON exceeds its byte budget');
  extensions(json);
  const table=(field,max)=>{const list=json[field]??[];if(!Array.isArray(list)||list.length>max)fail('LIMIT',`Invalid or excessive ${field}`);return list;};
  const definitions=table('buffers',maxResources),views=table('bufferViews',maxResources*16),images=table('images',maxResources);
  if(!Array.isArray(asset.buffers)||asset.buffers.length!==definitions.length)fail('BUFFER','Supply the original buffer slots');
  const supplied=asset.buffers.slice(),readImage=asset.readImage,used=new Set(),offsets=[],chunks=[];
  let binaryLength=0;
  const reserve=size=>{
    integer(size,1,maxBytes,'resource byte length');
    if(align(binaryLength)+size+28>maxBytes)fail('LIMIT','Binary storage exceeds the GLB budget');
  };
  const append=data=>{reserve(data.length);const offset=align(binaryLength);chunks.push({offset,bytes:data});binaryLength=offset+data.length;return offset;};
  for(const definition of definitions) {
    object(definition,'buffer');integer(definition.byteLength,1,0xffffffff,'declared buffer length');
    if(definition.extensions!==undefined&&Object.keys(object(definition.extensions,'buffer extensions')).length)fail('EXTENSION','Buffer extensions require their source packer');
  }
  for(const view of views) {
    object(view,'bufferView');integer(view.buffer,0,definitions.length-1,'buffer reference');
    const offset=view.byteOffset??0,length=view.byteLength;
    integer(offset,0,0xffffffff,'view offset');integer(length,1,0xffffffff,'view length');
    if(length>definitions[view.buffer].byteLength-offset)fail('BOUNDS','bufferView exceeds the declared buffer');
    if(view.extensions!==undefined&&Object.keys(object(view.extensions,'bufferView extensions')).length)fail('EXTENSION','Decode extended bufferViews before exporting');
    used.add(view.buffer);
  }
  // Validate all image metadata before reading/copying or invoking a provider.
  const external=new Map();
  for(let i=0;i<images.length;i++) {
    const image=object(images[i],'image');
    if((image.uri===undefined)===(image.bufferView===undefined))fail('IMAGE','Image needs exactly one URI or bufferView');
    if(image.mimeType!==undefined&&!['image/png','image/jpeg','image/ktx2'].includes(image.mimeType))fail('IMAGE','Unsupported image MIME type');
    if(image.uri!==undefined) {
      if(typeof image.uri!=='string'||!image.uri)fail('IMAGE','Invalid image URI');
      if(typeof readImage!=='function')fail('IMAGE','URI images require asset.readImage');
      const previous=external.get(image.uri);
      if(previous?.mimeType!==undefined&&image.mimeType!==undefined&&previous.mimeType!==image.mimeType)fail('IMAGE','Conflicting MIME types for one URI');
      external.set(image.uri,{index:previous?.index??i,mimeType:previous?.mimeType??image.mimeType});
    } else {
      integer(image.bufferView,0,views.length-1,'image bufferView');
      if(image.mimeType===undefined||views[image.bufferView].byteStride!==undefined)fail('IMAGE','Embedded images need MIME and unstrided bytes');
    }
  }
  // No supported extension has resource URIs of its own. Reject unknown resource
  // fields even when placed inside an otherwise supported extension record.
  if(views.length+external.size>maxResources*16)fail('LIMIT','Embedded image views exceed the resource count limit');
  const coreResources=new Set([...definitions,...images]);
  function resourceUris(value,core=false,inExtension=false) {
    if(!value||typeof value!=='object')return;
    for(const [key,child]of Object.entries(value)) {
      if(key==='extras')continue;
      if(key==='uri'&&!core)fail('EXTENSION','Unrecognized resource URI prevents self-contained export');
      if(inExtension&&key==='buffer')fail('EXTENSION','Opaque extension buffer references cannot be relocated');
      if(key==='extensions')resourceUris(child,false,true);
      else if(child&&typeof child==='object')resourceUris(child,coreResources.has(child),inExtension);
    }
  }
  resourceUris(json);
  for(let i=0;i<definitions.length;i++) {
    abort(signal);
    if(supplied[i]==null) {if(used.has(i))fail('BUFFER',`Referenced buffer ${i} has no bytes`);continue;}
    const data=bytes(supplied[i]),size=definitions[i].byteLength;
    if(data.length<size)fail('BUFFER',`Buffer ${i} is truncated`);
    reserve(size);offsets[i]=append(data.slice(0,size));
  }
  for(const image of images)if(image.bufferView!==undefined) {
    const view=views[image.bufferView],chunk=chunks.find(c=>c.offset===offsets[view.buffer]);
    const kind=mime(chunk.bytes.subarray(view.byteOffset??0,(view.byteOffset??0)+view.byteLength));
    if(kind!==image.mimeType)fail('IMAGE','Embedded MIME disagrees with image bytes');
  }
  for(const view of views){view.byteOffset=offsets[view.buffer]+(view.byteOffset??0);view.buffer=0;}
  // Original buffer names/extras remain recoverable even when several buffers
  // are consolidated. Core identities elsewhere in the document are unchanged.
  const metadata=definitions.map((d,i)=>({byteLength:d.byteLength,byteOffset:offsets[i]??null,
    ...(d.name===undefined?{}:{name:d.name}),...(d.extras===undefined?{}:{extras:d.extras})}));
  const uriViews=new Map();
  for(const [uri,request]of external) {
    abort(signal);
    const encoded=await wait(Reflect.apply(readImage,asset,[request.index]),signal);abort(signal);
    object(encoded,'encoded image');const data=bytes(encoded.bytes),kind=mime(data);
    if(encoded.mimeType!==kind||(request.mimeType!==undefined&&request.mimeType!==kind))fail('IMAGE','Image MIME disagrees with encoded bytes');
    reserve(data.length);const byteOffset=append(data.slice());
    const bufferView=views.push({buffer:0,byteOffset,byteLength:data.length})-1;
    uriViews.set(uri,{bufferView,mimeType:kind});
  }
  for(const image of images)if(image.uri!==undefined) {
    Object.assign(image,uriViews.get(image.uri));delete image.uri;
  }
  if(views.length)json.bufferViews=views;
  if(binaryLength) {
    const buffer={byteLength:binaryLength};
    if(definitions.length===1) {
      if(definitions[0].name!==undefined)buffer.name=definitions[0].name;
      if(definitions[0].extras!==undefined)buffer.extras=definitions[0].extras;
    } else if(metadata.some(m=>m.name!==undefined||m.extras!==undefined))buffer.extras={f3dSourceBuffers:metadata};
    json.buffers=[buffer];
  } else delete json.buffers;
  abort(signal);
  const text=new TextEncoder().encode(encodeJson(json));
  if(text.length>maxJsonBytes)fail('LIMIT','Output JSON exceeds its byte budget');
  const jsonLength=align(text.length),binLength=align(binaryLength),length=20+jsonLength+(binaryLength?8+binLength:0);
  if(length>maxBytes)fail('LIMIT','Complete GLB exceeds its byte limit');
  const output=new Uint8Array(length),header=new DataView(output.buffer);
  header.setUint32(0,0x46546c67,true);header.setUint32(4,2,true);header.setUint32(8,length,true);
  header.setUint32(12,jsonLength,true);header.setUint32(16,0x4e4f534a,true);
  output.fill(32,20,20+jsonLength);output.set(text,20);
  if(binaryLength){
    header.setUint32(20+jsonLength,binLength,true);header.setUint32(24+jsonLength,0x004e4942,true);
    for(const chunk of chunks)output.set(chunk.bytes,28+jsonLength+chunk.offset);
  }
  return output.buffer;
}
