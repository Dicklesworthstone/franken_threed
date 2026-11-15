/** One synchronous tangent transaction over witnessed source objects. Source
 * validation, attribute allocation and the final public flag stay in Three.
 * Unknown methods, groups, storage and effects resume the original algorithm.
 */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {isNumericObject} from './numeric_objects.mjs';
import {matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf,apply=Reflect.apply;
const hasOwn=Object.hasOwn,integer=Number.isSafeInteger,freeze=Object.freeze;
const own=(object,key)=>{const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>object && own(object,key)?.value;
const getMethod=WeakMap.prototype.get,setMethod=WeakMap.prototype.set,hasMethod=WeakSet.prototype.has,addMethod=WeakSet.prototype.add;
const get=(map,key)=>apply(getMethod,map,[key]),set=(map,key,value)=>apply(setMethod,map,[key,value]);
const F32=Float32Array,F64=Float64Array,U8=Uint8Array,objectPrototype=Object.prototype;
const typed=prototype(F64.prototype),tag=descriptor(typed,Symbol.toStringTag).get;
const length=descriptor(typed,'length').get,buffer=descriptor(typed,'buffer').get,offset=descriptor(typed,'byteOffset').get;
const copy=data(typed,'set'),bufferLength=descriptor(ArrayBuffer.prototype,'byteLength').get;
const resizable=descriptor(ArrayBuffer.prototype,'resizable')?.get;
const decode=globalThis.atob,charCode=String.prototype.charCodeAt;
const finite=data(globalThis,'isFinite'),nativeFinite=typeof finite==='function' &&
  apply(Function.prototype.toString,finite,[])==='function isFinite() { [native code] }';
const layouts={Float32Array:{type:'f32[]',Type:F32,bytes:4},Float64Array:{type:'f64[]',Type:F64,bytes:8},
  Uint16Array:{type:'u16[]',Type:Uint16Array,bytes:2},Uint32Array:{type:'u32[]',Type:Uint32Array,bytes:4}};
const classes=new WeakMap(),allocations=new WeakSet(),dispatches=new WeakMap(),statistics=new WeakMap();
const methods={Vector3:['fromBufferAttribute','sub','copy','multiplyScalar','addScaledVector','add','dot','normalize','divideScalar','length','crossVectors'],
  Vector2:['fromBufferAttribute','sub'],attribute:['getX','getY','getZ','setXYZW']};

/** Emitted only at pinned array/object literals and class declarations. A Proxy
 * around an allocation never acquires its target's witness. */
export function registerGeometryTangentAllocation(value) {apply(addMethod,allocations,[value]);return value;}
export function registerGeometryTangentClass(kind,Class,children=[],layout='contiguous') {
  const proto=data(Class,'prototype'),record={kind,proto,parent:prototype(proto),Class,layout,
    countGetter:layout==='interleaved'?descriptor(proto,'count')?.get:null,
    methods:methods[kind].map(name=>[name,data(proto,name)])};
  set(classes,Class,record);set(classes,proto,record);
  for(const Child of children) {
    const child=data(Child,'prototype');
    if(prototype(child)===proto)set(classes,child,{...record,proto:child,parent:proto});
  }
}
export function createGeometryTangentDispatch(variants,Vector3,Vector2,resolveMath,resolveFinite,maxMemoryBytes) {
  const token=freeze({});set(dispatches,token,{Vector3,Vector2,resolveMath,resolveFinite,maxMemoryBytes,
    variants:variants.map(value=>({...value,attempted:false}))});return token;
}
function vectorsIntact(record) {
  for(const Class of [record.Vector3,record.Vector2]) {
    const known=get(classes,Class);
    if(!known || data(Class,'prototype')!==known.proto || prototype(known.proto)!==objectPrototype)return false;
    for(const key of known.kind==='Vector2'?['x','y']:['x','y','z'])
      if(descriptor(known.proto,key) || descriptor(objectPrototype,key))return false;
    for(const [name,method]of known.methods)if(!method || data(known.proto,name)!==method)return false;
  }
  return true;
}
function attributeInput(object,size) {
  if(!isNumericObject(object))return null;
  const proto=prototype(object),known=get(classes,proto);
  if(!known || known.kind!=='attribute' || prototype(proto)!==known.parent)return null;
  for(const [name,method]of known.methods) {
    if(!method)return null;
    let current=object,found=false;
    while(current) {
      const d=descriptor(current,name);
      if(d){if(!hasOwn(d,'value') || d.value!==method)return null;found=true;break;}
      current=prototype(current);
      const entry=get(classes,current);
      if(!entry || entry.kind!=='attribute' || prototype(current)!==entry.parent)return null;
    }
    if(!found)return null;
  }
  const itemSize=data(object,'itemSize');
  if(data(object,'normalized')!==false || (known.layout==='interleaved'
    ? size===1 || !integer(itemSize) || itemSize<size : itemSize!==size))return null;
  let array,count,stride=size,componentOffset=0;
  if(known.layout==='interleaved') {
    // Inspect descriptors and witnessed backing storage, never invoke a source
    // accessor during speculative admission. The original count getter reads
    // this same own data field. Custom getters and Proxies retain source code.
    const countDescriptor=descriptor(proto,'count');
    if(descriptor(object,'count') || !known.countGetter || countDescriptor?.get!==known.countGetter || countDescriptor.set)return null;
    const interleaved=data(object,'data');if(!isNumericObject(interleaved))return null;
    array=data(interleaved,'array');count=data(interleaved,'count');stride=data(interleaved,'stride');componentOffset=data(object,'offset');
    if(!integer(stride) || stride<itemSize || !integer(componentOffset) || componentOffset<0 || componentOffset>stride-itemSize)return null;
  }else {array=data(object,'array');count=data(object,'count');}
  if(!integer(count) || count<0 || !integer(count*stride))return null;
  try {
    const layout=layouts[apply(tag,array,[])];
    if(!layout || prototype(array)!==layout.Type.prototype || apply(length,array,[])!==count*stride)return null;
    const storage=apply(buffer,array,[]),byteLength=apply(bufferLength,storage,[]),byteOffset=apply(offset,array,[]);
    if(resizable && apply(resizable,storage,[]))return null;
    const bytes=count*stride*layout.bytes;
    if(bytes+byteOffset>byteLength)return null;
    return {...layout,array,count,storage,byteOffset,bytes,size,stride,componentOffset};
  }catch{return null;}
}
function groupsInput(owner,indexCount,maxBytes) {
  const groups=data(owner,'groups');
  if(!apply(hasMethod,allocations,[groups]))return null;
  const count=data(groups,'length');
  if(!integer(count) || count<0 || count*16>maxBytes)return null;
  if(count===0)return indexCount%3===0?new F64([0,indexCount]):null;
  const packed=new F64(count*2);
  for(let i=0;i<count;i++) {
    const group=data(groups,i);
    if(!apply(hasMethod,allocations,[group]))return null;
    const start=data(group,'start'),size=data(group,'count');
    if(!integer(start) || !integer(size) || start<0 || size<0 || size%3!==0 || start+size>indexCount)return null;
    packed[i*2]=start;packed[i*2+1]=size;
  }
  return packed;
}
function inputs(owner,record,position,normal,uv,index,tangent) {
  if(!vectorsIntact(record) || !nativeFinite || data(globalThis,'isFinite')!==finite || record.resolveFinite()!==finite)return null;
  const values=[attributeInput(position,3),attributeInput(normal,3),attributeInput(uv,2),attributeInput(index,1),attributeInput(tangent,4)];
  if(values.some(value=>!value) || values.some((value,i)=>!((i===3?['u16[]','u32[]']:['f32[]','f64[]']).includes(value.type))) ||
      values.some((value,i)=>i!==3 && value.count!==values[0].count))return null;
  const target=values[4];
  for(let i=0;i<4;i++) {
    const input=values[i];
    if(input.storage===target.storage && input.bytes && target.bytes && input.byteOffset<target.byteOffset+target.bytes &&
        target.byteOffset<input.byteOffset+input.bytes) {
      // A common interleaved record shares its array across attributes. Native
      // tangents may write disjoint lanes in that record without changing a
      // later normal/index read. Unknown or genuinely intersecting layouts
      // keep the source recurrence instead of separating aliased state.
      const separateLanes=input.type===target.type && input.byteOffset===target.byteOffset &&
        input.count===target.count && input.stride===target.stride &&
        (input.componentOffset+input.size<=target.componentOffset || target.componentOffset+target.size<=input.componentOffset);
      if(!separateLanes)return null;
    }
  }
  const groups=groupsInput(owner,values[3].count,record.maxMemoryBytes);
  if(!groups)return null;
  const groupCount=apply(length,groups,[])/2;
  if(values.reduce((sum,value)=>sum+value.bytes,0)+groupCount*16+values[0].count*48>record.maxMemoryBytes)return null;
  return {values,target,groups,groupCount};
}
function kernel(record,variant) {
  if(!variant.attempted) {
    variant.attempted=true;
    try {
      const binary=apply(decode,globalThis,[variant.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      variant.kernel=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath});
    }catch{variant.failure='GEOMETRY_TANGENT_INITIALIZATION';}
    variant.base64=null;
  }
  return variant.kernel;
}
export function tryGeometryTangents(owner,token,position,normal,uv,index,tangent) {
  if(!isNumericObject(owner))return false;
  let stats=get(statistics,owner);
  if(!stats){stats={wasmCalls:0,fallbackCalls:0,lastFailure:null};set(statistics,owner,stats);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;},record=get(dispatches,token);
  if(!record || !matrixNumericPlatformIntact())return fallback('GEOMETRY_TANGENT_PLATFORM');
  let input;
  try{input=inputs(owner,record,position,normal,uv,index,tangent);}catch{return fallback('GEOMETRY_TANGENT_GUARD');}
  if(!input)return fallback('GEOMETRY_TANGENT_GUARD');
  const variant=record.variants.find(value=>value.types.every((type,i)=>type===input.values[i].type));
  if(!variant)return fallback('GEOMETRY_TANGENT_STORAGE');
  const runtime=kernel(record,variant);if(!runtime)return fallback(variant.failure);
  // Host initialization can encounter policy hooks. Re-read every source object
  // before any array is borrowed; no application code runs inside the kernel.
  if(!matrixNumericPlatformIntact())return fallback('GEOMETRY_TANGENT_PLATFORM');
  try{input=inputs(owner,record,position,normal,uv,index,tangent);}catch{return fallback('GEOMETRY_TANGENT_GUARD');}
  if(!input)return fallback('GEOMETRY_TANGENT_GUARD');
  try {
    const {values,target,groups,groupCount}=input,count=values[0].count;
    const output=new target.Type(count*target.stride),tan1=new F64(count*3),tan2=new F64(count*3);
    apply(copy,output,[target.array]);
    runtime.run(values[0].array,values[1].array,values[2].array,values[3].array,output,groups,tan1,tan2,groupCount,
      values[0].stride,values[0].componentOffset,values[1].stride,values[1].componentOffset,
      values[2].stride,values[2].componentOffset,target.stride,target.componentOffset);
    // Engines may choose different NaN payloads for Wasm arithmetic. Keep those
    // uncommon results on the source path before publishing a single byte.
    for(let i=0;i<count;i++)for(let j=0;j<4;j++) {
      const value=output[i*target.stride+target.componentOffset+j];
      if(value!==value)return fallback('GEOMETRY_TANGENT_NAN');
    }
    apply(copy,target.array,[output]);
  }catch{return fallback(runtime.diagnostics.lastGuardFailure??'GEOMETRY_TANGENT_EXECUTION');}
  stats.wasmCalls++;stats.lastFailure=null;return true;
}
export function geometryTangentsDiagnostics(owner) {
  return freeze({...get(statistics,owner)??{wasmCalls:0,fallbackCalls:0,lastFailure:null}});
}
