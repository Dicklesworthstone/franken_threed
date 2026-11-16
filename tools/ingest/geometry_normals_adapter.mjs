/** Guarded source-library geometry loops; all public Three objects stay retained.
 * Unknown identities/prototypes, custom methods and mixed-type overlap keep
 * the original method body. The numeric runtime stages every write before one
 * synchronous publication. This is a copying baseline, not a speedup claim.
 */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {isNumericObject} from './numeric_objects.mjs';
const descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf,apply=Reflect.apply;
const hasOwn=Object.hasOwn,integer=Number.isSafeInteger;
const own=(object,key)=>{const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>object && own(object,key)?.value;
const objectPrototype=Object.prototype,U8=Uint8Array,F64=Float64Array,decode=globalThis.atob,charCode=String.prototype.charCodeAt;
const typedPrototype=prototype(Float64Array.prototype);
const tag=descriptor(typedPrototype,Symbol.toStringTag).get,length=descriptor(typedPrototype,'length').get;
const tags={Float32Array:'f32[]',Float64Array:'f64[]',Uint16Array:'u16[]',Uint32Array:'u32[]',
  Uint8Array:'u8[]',Uint8ClampedArray:'u8c[]',Int8Array:'i8[]',Int16Array:'i16[]',Int32Array:'i32[]'};
const normalizedTypes=[Float32Array,Uint32Array,Uint16Array,Uint8Array,Uint8ClampedArray,Int32Array,Int16Array,Int8Array]
  .map(Type=>({type:tags[Type.name],name:Type.name,Type,proto:Type.prototype}));
const emptyIndex=new Uint16Array(0),attributes=new WeakMap(),vectors=new WeakMap(),dispatches=new WeakMap(),statistics=new WeakMap();
const attributeMethods=['getX','getY','getZ','setXYZ'];
const vectorMethods=['fromBufferAttribute','subVectors','cross','crossVectors','add','normalize','divideScalar','length','multiplyScalar'];
const wasm=data(globalThis,'WebAssembly'),platform=[];
if(wasm)for(const name of ['Module','Instance','Memory'])platform.push([wasm,name,data(wasm,name)]);
const moduleType=data(wasm,'Module'),memoryType=data(wasm,'Memory'),memoryPrototype=data(memoryType,'prototype');
if(moduleType)for(const name of ['customSections','imports','exports'])platform.push([moduleType,name,data(moduleType,name)]);
if(memoryType)platform.push([memoryType,'prototype',memoryPrototype],[memoryPrototype,'grow',data(memoryPrototype,'grow')]);
const bufferGetter=memoryPrototype && descriptor(memoryPrototype,'buffer')?.get;
const platformIntact=()=>wasm && data(globalThis,'WebAssembly')===wasm &&
  platform.every(([object,key,value])=>data(object,key)===value) && descriptor(memoryPrototype,'buffer')?.get===bufferGetter;

/** Compiler-emitted registrations for exact source class definitions. */
export function registerGeometryNormalAttributes(Base,children,layout='contiguous',resolveDenormalize=null) {
  const base=data(Base,'prototype'),methods=new Map(attributeMethods.map(name=>[name,data(base,name)]));
  const countGetter=layout==='interleaved'?descriptor(base,'count')?.get:null;
  const arrayGetter=layout==='interleaved'?descriptor(base,'array')?.get:null;
  const denormalize=resolveDenormalize?.();
  attributes.set(base,{parent:prototype(base),methods,layout,countGetter,arrayGetter,resolveDenormalize,denormalize});
  for(const Child of children) {
    const child=data(Child,'prototype');
    if(prototype(child)===base)attributes.set(child,{parent:base,methods,layout,countGetter,arrayGetter,resolveDenormalize,denormalize});
  }
}
export function registerGeometryNormalVector(Vector) {
  const proto=data(Vector,'prototype');
  if(prototype(proto)!==objectPrototype)return;
  vectors.set(Vector,{proto,methods:new Map(vectorMethods.map(name=>[name,data(proto,name)]))});
}
export function createGeometryNormalDispatch(variants,Vector,resolveMath) {
  const token=Object.freeze({});
  dispatches.set(token,{Vector,resolveMath,variants:variants.map(value=>({...value,attempted:false,kernel:null}))});
  return token;
}
function vectorIntact(record) {
  const known=vectors.get(record.Vector);
  if(!known || data(record.Vector,'prototype')!==known.proto || prototype(known.proto)!==objectPrototype)return null;
  for(const key of ['x','y','z'])if(descriptor(known.proto,key) || descriptor(objectPrototype,key))return null;
  for(const [key,method] of known.methods)if(!method || data(known.proto,key)!==method)return null;
  return known;
}
function attributeInput(object,size,allowNormalized=false) {
  if(!isNumericObject(object))return null;
  const proto=prototype(object),record=attributes.get(proto);
  if(!record)return null;
  for(const [name,method] of record.methods) {
    if(!method)return null;
    let current=object,found=false;
    while(current) {
      if(current!==object) {
        const type=attributes.get(current);
        if(!type || prototype(current)!==type.parent)return null;
      }
      const d=descriptor(current,name);
      if(d) {if(!hasOwn(d,'value') || d.value!==method)return null;found=true;break;}
      current=current===object?proto:attributes.get(current).parent;
    }
    if(!found)return null;
  }
  const itemSize=data(object,'itemSize'),normalized=data(object,'normalized');
  if(!(normalized===false || allowNormalized && normalized===true) || (record.layout==='interleaved'
    ? size!==3 || !integer(itemSize) || itemSize<size : itemSize!==size))return null;
  let array,count,stride=size,offset=0;
  if(record.layout==='interleaved') {
    const countDescriptor=descriptor(proto,'count');
    if(descriptor(object,'count') || !record.countGetter || countDescriptor?.get!==record.countGetter || countDescriptor.set)return null;
    if(normalized) {
      const arrayDescriptor=descriptor(proto,'array');
      // The source reads data.array for the component, then this.array for
      // denormalize's constructor dispatch. Neither lookup may gain a hook.
      if(descriptor(object,'array') || !record.arrayGetter || arrayDescriptor?.get!==record.arrayGetter || arrayDescriptor.set)return null;
    }
    const buffer=data(object,'data');if(!isNumericObject(buffer))return null;
    array=data(buffer,'array');count=data(buffer,'count');stride=data(buffer,'stride');offset=data(object,'offset');
    if(!integer(stride) || stride<itemSize || !integer(offset) || offset<0 || offset>stride-itemSize)return null;
  }else {array=data(object,'array');count=data(object,'count');}
  if(!integer(count) || count<0 || !integer(count*stride))return null;
  try {
    const type=tags[apply(tag,array,[])];
    if(!type || apply(length,array,[])!==count*stride)return null;
    if(normalized) {
      let expected;
      // Indexed access to private records cannot invoke replaced Map methods.
      for(let i=0;i<normalizedTypes.length;i++) {
        const entry=normalizedTypes[i];
        if(data(globalThis,entry.name)!==entry.Type)return null;
        if(entry.type===type)expected=entry;
      }
      if(!expected || prototype(array)!==expected.proto || !record.resolveDenormalize ||
          record.resolveDenormalize()!==record.denormalize)return null;
      const constructor=descriptor(array,'constructor')??descriptor(expected.proto,'constructor');
      if(!constructor || !hasOwn(constructor,'value') || constructor.value!==expected.Type)return null;
    }
    return {array,count,type,stride,offset,normalized};
  }catch{return null;}
}
function kernel(record,variant) {
  if(!variant.attempted) {
    variant.attempted=true;
    try {
      const binary=apply(decode,globalThis,[variant.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      // The source loops keep every load/store in order. Same-type aliases
      // share staged storage; mixed-type writable overlap still retains JS.
      variant.kernel=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath,preserveAliasing:true});
    }catch{variant.failure='GEOMETRY_NORMAL_INITIALIZATION';}
    variant.base64=null;
  }
  return variant.kernel;
}
function state(owner,method) {
  let methods=statistics.get(owner);if(!methods){methods={};statistics.set(owner,methods);}
  return methods[method]??=( {wasmCalls:0,fallbackCalls:0,lastFailure:null} );
}
function inputs(record,method,position,index,normal,scratch) {
  const vector=vectorIntact(record),normals=attributeInput(normal,3);
  if(!vector || !normals || !['f32[]','f64[]'].includes(normals.type))return null;
  if(method==='compute') {
    const positions=attributeInput(position,3,true),indices=index===null?{array:emptyIndex,count:0,type:'u16[]'}:attributeInput(index,1);
    if(!positions || !indices || !['u16[]','u32[]'].includes(indices.type) ||
        normals.count!==positions.count)return null;
    return {types:[positions.type,indices.type,normals.type],normalized:positions.normalized,args:[positions.array,indices.array,normals.array,
      positions.count,normals.count,indices.count,index===null?0:1,positions.stride,positions.offset,normals.stride,normals.offset,
      positions.normalized?1:0]};
  }
  if(!isNumericObject(scratch) || prototype(scratch)!==vector.proto)return null;
  for(const [name,method] of vector.methods) {
    const d=descriptor(scratch,name);
    if(d && (!hasOwn(d,'value') || d.value!==method))return null;
  }
  const values=[];
  for(const name of ['x','y','z']) {
    const d=own(scratch,name);if(!d?.writable || typeof d.value!=='number')return null;values.push(d.value);
  }
  const staged=new F64(values);
  return {types:[normals.type],args:[normals.array,staged,normals.count,normals.stride,normals.offset],staged};
}
/** Source prefixes already ran. A miss resumes only the original numeric block. */
export function tryGeometryNormals(owner,token,method,position,index,normal,scratch=null) {
  if(!isNumericObject(owner))return false;
  const stats=state(owner,method),fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  const record=dispatches.get(token);
  if(!record || !platformIntact())return fallback('GEOMETRY_NORMAL_PLATFORM');
  let input=inputs(record,method,position,index,normal,scratch);
  if(!input)return fallback('GEOMETRY_NORMAL_GUARD');
  const variant=record.variants.find(value=>value.method===method && value.types.every((type,i)=>type===input.types[i]));
  if(!variant)return fallback('GEOMETRY_NORMAL_STORAGE');
  if(input.normalized && !variant.normalized)return fallback('GEOMETRY_NORMAL_GUARD');
  const runtime=kernel(record,variant);
  if(!runtime)return fallback(variant.failure);
  // Instantiation can encounter host policy hooks. No property snapshot is
  // reused across initialization, and no source callback runs while borrowed.
  if(!platformIntact() || !(input=inputs(record,method,position,index,normal,scratch)) || input.normalized && !variant.normalized)
    return fallback('GEOMETRY_NORMAL_GUARD');
  try {runtime.run(...input.args);}
  catch{return fallback(runtime.diagnostics.lastGuardFailure??'GEOMETRY_NORMAL_EXECUTION');}
  if(method==='normalize') {
    // A retained override may previously have exposed this module's scratch
    // vector. Preserve its identity and final unrounded Number components.
    scratch.x=input.staged[0];scratch.y=input.staged[1];scratch.z=input.staged[2];
  }
  stats.wasmCalls++;stats.lastFailure=null;return true;
}
export function geometryNormalsDiagnostics(object) {
  const result={};
  for(const [method,stats] of Object.entries(statistics.get(object)??{}))result[method]=Object.freeze({...stats});
  return Object.freeze(result);
}
