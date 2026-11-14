/** Checked native BufferAttribute loops. Public arrays and the source's scratch
 * Vector3 retain identity and synchronous observable values. No renderer upload
 * is requested here: needsUpdate remains at the original call boundary. */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {isNumericObject} from './numeric_objects.mjs';
import {matrixNumericPlatformIntact,matrixNumericElements} from './matrix_multiply_adapter.mjs';
const descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf,apply=Reflect.apply;
const hasOwn=Object.hasOwn,integer=Number.isSafeInteger,freeze=Object.freeze;
const own=(object,key)=>{const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>object && own(object,key)?.value;
const F64=Float64Array,U8=Uint8Array,decode=globalThis.atob,charCode=String.prototype.charCodeAt;
const typedPrototype=prototype(F64.prototype),tag=descriptor(typedPrototype,Symbol.toStringTag).get;
const length=descriptor(typedPrototype,'length').get,objectPrototype=Object.prototype;
const attributes=new WeakMap(),vectors=new WeakMap(),dispatches=new WeakMap(),statistics=new WeakMap();
const matrixObjects=new WeakSet(),matrixArrays=new WeakSet();
const setHas=WeakSet.prototype.has,setAdd=WeakSet.prototype.add;
const mapGet=WeakMap.prototype.get,mapSet=WeakMap.prototype.set;
const get=(map,key)=>apply(mapGet,map,[key]),set=(map,key,value)=>apply(mapSet,map,[key,value]);
const attributeMethods=['getX','getY','getZ','setXYZ'];
const vectorMethods=['fromBufferAttribute','applyMatrix4','applyMatrix3','applyNormalMatrix','transformDirection',
  'normalize','divideScalar','length','multiplyScalar'];

/** These witnesses are emitted at verified Matrix3 allocations only. */
export function registerGeometryTransformMatrix(object){apply(setAdd,matrixObjects,[object]);}
export function registerGeometryTransformElements(array){apply(setAdd,matrixArrays,[array]);return array;}
export function registerGeometryTransformVector(Vector) {
  const proto=data(Vector,'prototype');
  if(prototype(proto)===objectPrototype)set(vectors,Vector,{proto,
    methods:new Map(vectorMethods.map(name=>[name,data(proto,name)]))});
}
export function registerGeometryTransformAttributes(Base,children,layout='contiguous') {
  const base=data(Base,'prototype'),methods=new Map(attributeMethods.map(name=>[name,data(base,name)]));
  const countGetter=layout==='interleaved'?descriptor(base,'count')?.get:null;
  set(attributes,base,{parent:prototype(base),methods,layout,countGetter});
  for(const Child of children) {
    const child=data(Child,'prototype');if(prototype(child)===base)set(attributes,child,{parent:base,methods,layout,countGetter});
  }
}
export function createGeometryTransformDispatch(variants,Vector,resolveMath) {
  const token=freeze({});set(dispatches,token,{Vector,resolveMath,
    variants:variants.map(value=>({...value,attempted:false,kernel:null}))});return token;
}
function vectorInput(record,scratch) {
  if(!isNumericObject(scratch))return null;
  const known=get(vectors,record.Vector);
  if(!known || data(record.Vector,'prototype')!==known.proto || prototype(known.proto)!==objectPrototype ||
      prototype(scratch)!==known.proto)return null;
  for(const [key,method]of known.methods) {
    const d=descriptor(scratch,key);
    if(!method || data(known.proto,key)!==method || d && (!hasOwn(d,'value') || d.value!==method))return null;
  }
  const staged=new F64(3);
  for(const [index,key]of ['x','y','z'].entries()) {
    const d=own(scratch,key);
    if(!d?.writable || typeof d.value!=='number' || descriptor(known.proto,key) || descriptor(objectPrototype,key))return null;
    staged[index]=d.value;
  }
  return staged;
}
function attributeInput(object) {
  const proto=prototype(object),record=get(attributes,proto);
  const itemSize=data(object,'itemSize');
  if(!record || data(object,'normalized')!==false ||
      (record.layout==='interleaved'? !integer(itemSize) || itemSize<3 : itemSize!==3))return null;
  for(const [name,method]of record.methods) {
    if(!method)return null;
    let current=object,found=false;
    while(current) {
      if(current!==object) {
        const type=get(attributes,current);if(!type || prototype(current)!==type.parent)return null;
      }
      const d=descriptor(current,name);
      if(d){if(!hasOwn(d,'value') || d.value!==method)return null;found=true;break;}
      current=current===object?proto:get(attributes,current).parent;
    }
    if(!found)return null;
  }
  let array,count,stride=3,offset=0;
  if(record.layout==='interleaved') {
    // The ordinary count getter is source code, not a speculative read. Check
    // its identity and the allocation of data before reading own data fields;
    // a Proxy or accessor resumes the original source loop without extra calls.
    const countDescriptor=descriptor(proto,'count');
    if(descriptor(object,'count') || !record.countGetter || countDescriptor?.get!==record.countGetter || countDescriptor.set)return null;
    const buffer=data(object,'data');if(!isNumericObject(buffer))return null;
    array=data(buffer,'array');count=data(buffer,'count');stride=data(buffer,'stride');offset=data(object,'offset');
    if(!integer(stride) || stride<itemSize || !integer(offset) || offset<0 || offset>stride-itemSize)return null;
  }else {array=data(object,'array');count=data(object,'count');}
  if(!integer(count) || count<0 || !integer(count*stride))return null;
  try {
    const type=apply(tag,array,[])==='Float32Array'?'f32[]':apply(tag,array,[])==='Float64Array'?'f64[]':null;
    return type && apply(length,array,[])===count*stride?{array,count,type,stride,offset,layout:record.layout}:null;
  }catch{return null;}
}
function matrixInput(matrix,method) {
  const small=method==='applyMatrix3' || method==='applyNormalMatrix';
  let array;
  if(small) {
    if(!apply(setHas,matrixObjects,[matrix]))return null;
    array=data(matrix,'elements');if(!apply(setHas,matrixArrays,[array]))return null;
  }else array=matrixNumericElements(matrix);
  if(!array)return null;
  const staged=new F64(small?9:16);
  for(let i=0;i<staged.length;i++) {
    const d=own(array,i);if(!d || typeof d.value!=='number')return null;staged[i]=d.value;
  }
  return staged;
}
function inputs(record,owner,method,matrix,scratch) {
  const attribute=attributeInput(owner),vector=vectorInput(record,scratch),elements=matrixInput(matrix,method);
  return attribute && vector && elements?{...attribute,vector,elements}:null;
}
function kernel(record,variant) {
  if(!variant.attempted) {
    variant.attempted=true;
    try {
      const binary=apply(decode,globalThis,[variant.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      variant.kernel=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath});
    }catch{variant.failure='GEOMETRY_TRANSFORM_INITIALIZATION';}
    variant.base64=null;
  }
  return variant.kernel;
}
export function tryGeometryTransform(owner,token,method,matrix,scratch) {
  if(!isNumericObject(owner))return false;
  // Guard platform methods before using collections or producing source data.
  if(!matrixNumericPlatformIntact())return false;
  let methods=get(statistics,owner);if(!methods){methods={};set(statistics,owner,methods);}
  const stats=methods[method]??={wasmCalls:0,fallbackCalls:0,lastFailure:null};
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;},record=get(dispatches,token);
  if(!record)return fallback('GEOMETRY_TRANSFORM_PLATFORM');
  let input=inputs(record,owner,method,matrix,scratch);
  if(!input)return fallback('GEOMETRY_TRANSFORM_GUARD');
  const variant=record.variants.find(value=>value.method===method && value.type===input.type && value.layout===input.layout);
  if(!variant)return fallback('GEOMETRY_TRANSFORM_STORAGE');
  const runtime=kernel(record,variant);if(!runtime)return fallback(variant.failure);
  // Initialization may encounter a host policy hook. Discard old snapshots and
  // revalidate before the single checked synchronous kernel publication.
  if(!matrixNumericPlatformIntact() || !(input=inputs(record,owner,method,matrix,scratch)))return fallback('GEOMETRY_TRANSFORM_GUARD');
  try{
    if(input.layout==='interleaved')runtime.run(input.array,input.elements,input.vector,input.count,input.stride,input.offset);
    else runtime.run(input.array,input.elements,input.vector,input.count);
  }
  catch{return fallback(runtime.diagnostics.lastGuardFailure??'GEOMETRY_TRANSFORM_EXECUTION');}
  scratch.x=input.vector[0];scratch.y=input.vector[1];scratch.z=input.vector[2];
  stats.wasmCalls++;stats.lastFailure=null;return true;
}
export function geometryTransformsDiagnostics(object) {
  const result={};for(const [method,stats]of Object.entries(get(statistics,object)??{}))result[method]=freeze({...stats});
  return freeze(result);
}
