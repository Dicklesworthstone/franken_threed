/** Closed, synchronous skeletal palette transaction. Public bone matrices keep
 * identity and the original texture setter executes after the palette commits. */
import {instantiateNumericKernel,numericArrayIterationIntact} from './numeric_kernel_runtime.mjs';
import {matrixNumericElements,matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,freeze=Object.freeze,integer=Number.isSafeInteger;
const has=WeakSet.prototype.has,add=WeakSet.prototype.add,get=WeakMap.prototype.get,set=WeakMap.prototype.set;
const F32=Float32Array,F64=Float64Array,U8=Uint8Array,decode=host.atob,charCode=String.prototype.charCodeAt;
const arrayPrototype=Array.prototype,slice=Array.prototype.slice;
const typed=prototype(F64.prototype),tag=descriptor(typed,Symbol.toStringTag).get;
const length=descriptor(typed,'length').get,buffer=descriptor(typed,'buffer').get;
const byteLength=descriptor(ArrayBuffer.prototype,'byteLength').get,resizable=descriptor(ArrayBuffer.prototype,'resizable')?.get;
const immutable=descriptor(ArrayBuffer.prototype,'immutable')?.get;
const own=(object,key)=>{const d=descriptor(object,key);return d&&hasOwn(d,'value')?d:null;};
const data=(object,key)=>object&&own(object,key)?.value;
const objects=new WeakSet(),owners=new WeakSet(),arrays=new WeakSet(),records=new WeakMap(),classes=new WeakMap();
const statistics=()=>({wasmCalls:0,bones:0,fallbackCalls:0,lastFailure:null});
export function registerSkeletonObject(object){apply(add,objects,[object]);}
export function registerSkeletonOwner(object){apply(add,owners,[object]);}
export function registerSkeletonArray(array){apply(add,arrays,[array]);return array;}
const platform=()=>numericArrayIterationIntact()&&matrixNumericPlatformIntact();

/** A successful native slice creates an ordinary Array only when its receiver
 * and constructor/species lookup are proven before any indexed getter runs.
 * Otherwise the original source call performs every lookup and effect once. */
export function trySkeletonBoneSlice(array) {
  if(!apply(has,arrays,[array])||!platform()||prototype(array)!==arrayPrototype)return null;
  const method=descriptor(array,'slice'),constructor=descriptor(array,'constructor');
  if(method&&(!hasOwn(method,'value')||method.value!==slice)||
    constructor&&(!hasOwn(constructor,'value')||constructor.value!==Array))return null;
  return registerSkeletonArray(apply(slice,array,[0]));
}
export function createSkeletonUpdateDispatch(Class,Matrix,variants,maxMemoryBytes) {
  const token=freeze({}),proto=data(Matrix,'prototype');
  const record={variants,maxMemoryBytes,proto,parent:prototype(proto),
    multiply:data(proto,'multiplyMatrices'),toArray:data(proto,'toArray'),kernels:{},statistics:statistics()};
  apply(set,records,[token,record]);apply(set,classes,[Class,record]);return token;
}
function matrixMethod(object,key,expected,record) {
  if(prototype(object)!==record.proto||prototype(record.proto)!==record.parent)return false;
  const value=descriptor(object,key)??descriptor(record.proto,key);
  return !!value&&hasOwn(value,'value')&&typeof expected==='function'&&value.value===expected;
}
function arrayCount(array) {
  return apply(has,arrays,[array])&&prototype(array)===arrayPrototype?data(array,'length'):null;
}
function storage(view,count) {
  try {
    const kind=apply(tag,view,[]),backing=apply(buffer,view,[]);
    if(kind!=='Float32Array'&&kind!=='Float64Array'||apply(length,view,[])<count*16)return null;
    apply(byteLength,backing,[]);
    if(resizable&&apply(resizable,backing,[])||immutable&&apply(immutable,backing,[]))return null;
    new U8(backing,0,0);
    return kind==='Float32Array'?'f32':'f64';
  }catch{return null;}
}
function inputs(record,bones,inverses,output,scratch,identity) {
  const count=arrayCount(bones);
  if(!integer(count)||count<0||count>(record.maxMemoryBytes-256)/512)return null;
  if(count===0)return {count};
  const inverseCount=arrayCount(inverses),kind=storage(output,count),target=matrixNumericElements(scratch,true);
  if(inverseCount===null||inverseCount<count||!kind||!target||
    !matrixMethod(scratch,'multiplyMatrices',record.multiply,record)||!matrixMethod(scratch,'toArray',record.toArray,record))return null;
  const state=new F64(count*48),result=kind==='f32'?new F32(count*16):new F64(count*16);
  for(let i=0;i<count;i++) {
    const boneEntry=descriptor(bones,i),inverseEntry=own(inverses,i);
    if(boneEntry&&!hasOwn(boneEntry,'value')||!inverseEntry)return null;
    const bone=boneEntry?.value;
    if(bone&&!apply(has,objects,[bone]))return null;
    const left=matrixNumericElements(bone?data(bone,'matrixWorld'):identity),right=matrixNumericElements(inverseEntry.value);
    // The source scratch can escape through a previously replaced method. Its
    // sequential feedback is deliberately left to JS if reused as an operand.
    if(!left||!right||left===target||right===target)return null;
    const offset=i*48;
    for(let j=0;j<16;j++){state[offset+j]=left[j];state[offset+16+j]=right[j];}
  }
  return {count,kind,state,result,output,target};
}
function kernel(record,kind) {
  let result=record.kernels[kind];
  if(result===undefined) {
    result=null;
    try {
      const binary=apply(decode,host,[record.variants[kind]]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      result=instantiateNumericKernel(bytes);
    }catch{/* Initialization failure resumes the complete source loop. */}
    record.kernels[kind]=result;record.variants[kind]=null;
  }
  return result;
}
export function trySkeletonUpdate(owner,token,bones,inverses,output,scratch,identity) {
  const record=apply(get,records,[token]);if(!record||!apply(has,owners,[owner]))return false;
  const stats=record.statistics,fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  if(!platform())return fallback('SKELETON_PLATFORM');
  let input,runtime;
  try {
    input=inputs(record,bones,inverses,output,scratch,identity);if(!input)return fallback('SKELETON_GUARD');
    if(input.count===0)return true;
    const initialized=record.kernels[input.kind]!==undefined;
    runtime=kernel(record,input.kind);if(!runtime)return fallback('SKELETON_INITIALIZATION');
    if(!platform()||!initialized&&!(input=inputs(record,bones,inverses,output,scratch,identity)))return fallback('SKELETON_GUARD');
    runtime.run(input.state,input.result,input.count);
  }catch{return fallback(runtime?.diagnostics.lastGuardFailure??'SKELETON_EXECUTION');}
  // NaN payloads are visible through typed aliases; refuse before publication.
  const values=input.count*16;
  for(let i=0;i<values;i++)if(input.result[i]!==input.result[i])return fallback('SKELETON_NAN');
  const last=(input.count-1)*48+32;
  for(let j=0;j<16;j++)if(input.state[last+j]!==input.state[last+j])return fallback('SKELETON_NAN');
  for(let i=0;i<values;i++)input.output[i]=input.result[i];
  for(let row=0;row<4;row++)for(let column=0;column<4;column++) {
    const index=row+column*4;input.target[index]=input.state[last+index];
  }
  stats.wasmCalls++;stats.bones+=input.count;stats.lastFailure=null;return true;
}
export function skeletonUpdateDiagnostics(Class) {
  return freeze({...apply(get,classes,[Class])?.statistics??statistics()});
}
