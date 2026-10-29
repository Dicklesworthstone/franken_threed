/** Synchronous flat-quaternion arithmetic with one authoritative source array.
 * Both source quaternions are read before any destination store, as upstream
 * does. Four-component snapshots preserve overlapping and mixed-layout views
 * without staging an entire animation track for each interpolated quaternion.
 */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,freeze=Object.freeze,source=Function.prototype.toString;
const get=WeakMap.prototype.get,set=WeakMap.prototype.set,F64=Float64Array,U8=Uint8Array;
const decode=host.atob,charCode=String.prototype.charCodeAt,isInteger=Number.isSafeInteger;
const records=new WeakMap(),classes=new WeakMap();
const data=(object,key)=>{const d=object && descriptor(object,key);return d && hasOwn(d,'value')?d.value:undefined;};
const typed=prototype(F64.prototype),length=descriptor(typed,'length').get,buffer=descriptor(typed,'buffer').get;
const tag=descriptor(typed,Symbol.toStringTag).get,bufferLength=descriptor(ArrayBuffer.prototype,'byteLength').get;
const resizable=descriptor(ArrayBuffer.prototype,'resizable')?.get,immutable=descriptor(ArrayBuffer.prototype,'immutable')?.get;
const math=data(host,'Math'),acos=data(math,'acos'),sin=data(math,'sin'),sqrt=data(math,'sqrt');
const native=(value,name)=>typeof value==='function' && apply(source,value,[])===`function ${name}() { [native code] }`;
const nativeMath=native(acos,'acos') && native(sin,'sin') && native(sqrt,'sqrt');
const wasm=data(host,'WebAssembly');
const nativeWasm=wasm && ['Module','Instance','Memory'].every(name=>native(data(wasm,name),name));
const numericTags=['Float64Array','Float32Array','Int8Array','Uint8Array','Uint8ClampedArray','Int16Array','Uint16Array','Int32Array','Uint32Array'];
const stats=()=>({wasmCalls:0,fallbackCalls:0,lastFailure:null});

export function createQuaternionFlatDispatch(Class,variants,prepare,resolveMath) {
  const token=freeze({});
  const record={variants,prepare,resolveMath,kernels:{},attempted:{},failures:{},
    output:new F64(4),left:new F64(4),right:new F64(4),mathValues:new F64(4),
    statistics:{slerpFlat:stats(),multiplyQuaternionsFlat:stats()}};
  apply(set,records,[token,record]);apply(set,classes,[Class,record]);return token;
}
function platform(record,slerp) {
  if(!nativeWasm || !matrixNumericPlatformIntact())return false;
  if(!slerp)return true;
  // Compare the global object's identity before resolving/inspecting a possible
  // replacement. A getter or Proxy in Math receives no speculative calls.
  return nativeMath && data(host,'Math')===math && data(math,'acos')===acos &&
    data(math,'sin')===sin && data(math,'sqrt')===sqrt && record.resolveMath()===math;
}
function storage(view,offset,writable) {
  if(typeof offset!=='number' || !isInteger(offset) || offset<0)return false;
  try {
    const count=apply(length,view,[]),type=apply(tag,view,[]);
    let numeric=false;for(let i=0;i<numericTags.length;i++)if(type===numericTags[i]){numeric=true;break;}
    if(!numeric || offset>count-4)return false;
    const backing=apply(buffer,view,[]);
    // Native ArrayBuffer getters refuse SharedArrayBuffer and Proxy receivers.
    // Fixed, attached storage cannot change during the callback-free attempt.
    if(!apply(bufferLength,backing,[]) || resizable && apply(resizable,backing,[]) ||
      writable && immutable && apply(immutable,backing,[]))return false;
    return true;
  }catch{return false;}
}
function input(dst,dstOffset,left,leftOffset,right,rightOffset,t,slerp) {
  return (!slerp || typeof t==='number') && storage(dst,dstOffset,true) &&
    storage(left,leftOffset,false) && storage(right,rightOffset,false);
}
function kernel(record,method) {
  if(!record.attempted[method]) {
    record.attempted[method]=true;
    try {
      const binary=apply(decode,host,[record.variants[method]]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      record.kernels[method]=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath});
    }catch{record.failures[method]='QUATERNION_FLAT_INITIALIZATION';}
  }
  return record.kernels[method];
}
export function tryQuaternionFlat(token,method,dst,dstOffset,left,leftOffset,right,rightOffset,t) {
  const record=apply(get,records,[token]);if(!record)return false;
  const current=record.statistics[method],slerp=method==='slerpFlat';
  const fallback=reason=>{current.fallbackCalls++;current.lastFailure=reason;return false;};
  if(!platform(record,slerp))return fallback('QUATERNION_FLAT_PLATFORM');
  if(!input(dst,dstOffset,left,leftOffset,right,rightOffset,t,slerp))return fallback('QUATERNION_FLAT_STORAGE');
  const runtime=kernel(record,method);if(!runtime)return fallback(record.failures[method]);
  if(!platform(record,slerp) || !input(dst,dstOffset,left,leftOffset,right,rightOffset,t,slerp))
    return fallback('QUATERNION_FLAT_STORAGE');
  for(let i=0;i<4;i++)record.left[i]=left[leftOffset+i];
  for(let i=0;i<4;i++)record.right[i]=right[rightOffset+i];
  try {
    if(slerp) {
      record.prepare(record.left,record.right,t,record.mathValues);
      runtime.run(record.output,0,record.left,0,record.right,0,t,record.mathValues);
    }else runtime.run(record.output,0,record.left,0,record.right,0);
  }catch{return fallback(runtime.diagnostics.lastGuardFailure??'QUATERNION_FLAT_EXECUTION');}
  // Number-level NaN equivalence does not prove identical payload/sign bytes
  // when a caller inspects a typed buffer through another view. Let the exact
  // source arithmetic produce those bytes; no destination was touched yet.
  for(let i=0;i<4;i++)if(record.output[i]!==record.output[i])return fallback('QUATERNION_FLAT_NAN');
  // This is the first visible mutation. The original method also stores in
  // x/y/z/w order; each destination's native integer/f32 conversion occurs here.
  for(let i=0;i<4;i++)dst[dstOffset+i]=record.output[i];
  current.wasmCalls++;current.lastFailure=null;return true;
}
export function quaternionFlatDiagnostics(Class) {
  const record=apply(get,classes,[Class]);
  return freeze({slerpFlat:freeze({...record?.statistics.slerpFlat??stats()}),
    multiplyQuaternionsFlat:freeze({...record?.statistics.multiplyQuaternionsFlat??stats()})});
}
