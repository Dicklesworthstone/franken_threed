/** One source-ordered interpolation segment. Only its active samples are packed;
 * the complete animation track is never copied for one evaluation. Unknown
 * storage, overlapping output and observable effects resume the source suffix.
 */
import {instantiateNumericKernel,numericArrayIterationIntact} from './numeric_kernel_runtime.mjs';
import {matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,freeze=Object.freeze,integer=Number.isSafeInteger;
const get=WeakMap.prototype.get,set=WeakMap.prototype.set,F64=Float64Array,U8=Uint8Array;
const decode=host.atob,charCode=String.prototype.charCodeAt,toSource=Function.prototype.toString;
const records=new WeakMap(),classes=new WeakMap();
const data=(object,key)=>{const d=object && descriptor(object,key);return d && hasOwn(d,'value')?d.value:undefined;};
const typed=prototype(F64.prototype),length=descriptor(typed,'length').get,buffer=descriptor(typed,'buffer').get;
const byteOffset=descriptor(typed,'byteOffset').get,byteLength=descriptor(typed,'byteLength').get;
const tag=descriptor(typed,Symbol.toStringTag).get,bufferLength=descriptor(ArrayBuffer.prototype,'byteLength').get;
const resizable=descriptor(ArrayBuffer.prototype,'resizable')?.get,immutable=descriptor(ArrayBuffer.prototype,'immutable')?.get;
const numericTags=['Float64Array','Float32Array','Int8Array','Uint8Array','Uint8ClampedArray','Int16Array','Uint16Array','Int32Array','Uint32Array'];
const native=(value,name)=>typeof value==='function' && apply(toSource,value,[])===`function ${name}() { [native code] }`;
const wasm=data(host,'WebAssembly'),nativeWasm=wasm && ['Module','Instance','Memory'].every(name=>native(data(wasm,name),name));
const math=data(host,'Math'),mathMethods=['abs','min','max'].map(name=>[name,data(math,name)]);
const nativeMath=mathMethods.every(([name,method])=>native(method,name));
const stats=()=>({wasmCalls:0,fallbackCalls:0,lastFailure:null});

export function createAnimationInterpolantDispatch(Class,mode,base64,resolveMath,maxMemoryBytes) {
  const token=freeze({}),record={mode,base64,resolveMath,maxMemoryBytes,attempted:false,kernel:null,statistics:stats()};
  apply(set,records,[token,record]);apply(set,classes,[Class,record]);return token;
}
function platform(record) {
  if(!nativeWasm || !numericArrayIterationIntact() || !matrixNumericPlatformIntact())return false;
  if(record.mode!=='bezier')return true;
  if(!nativeMath || data(host,'Math')!==math)return false;
  for(const [name,method]of mathMethods)if(data(math,name)!==method)return false;
  return record.resolveMath()===math;
}
function storage(view,writable=false) {
  try {
    const type=apply(tag,view,[]);let numeric=false;
    for(let i=0;i<numericTags.length;i++)if(type===numericTags[i]){numeric=true;break;}
    if(!numeric)return null;
    const count=apply(length,view,[]),backing=apply(buffer,view,[]);
    if(!apply(bufferLength,backing,[]) || resizable && apply(resizable,backing,[]) ||
        writable && immutable && apply(immutable,backing,[]))return null;
    return {view,count,backing,offset:apply(byteOffset,view,[]),bytes:apply(byteLength,view,[])};
  }catch{return null;}
}
const span=(input,start,size)=>typeof start==='number' && integer(start) && start>=0 && start<=input.count-size;
const overlaps=(a,b)=>a.backing===b.backing && a.offset<b.offset+b.bytes && b.offset<a.offset+a.bytes;
function inputs(record,result,values,stride,args) {
  if(typeof stride!=='number' || !integer(stride) || stride<0 || stride>record.maxMemoryBytes/80)return null;
  const target=storage(result,true),samples=storage(values);
  if(!target || !samples || target.count<stride || overlaps(target,samples))return null;
  const mode=record.mode;
  for(let i=0;i<args.length;i++)if(!(mode==='bezier' && (i===2 || i===3)) && typeof args[i]!=='number')return null;
  let offsets,tangents=null;
  if(mode==='linear')offsets=[(args[0]-1)*stride,args[0]*stride];
  else if(mode==='cubic')offsets=[args[2],args[1],args[0],args[3]];
  else {
    offsets=[args[1],args[0]];
    const incoming=storage(args[2]),outgoing=storage(args[3]),current=args[4]*stride*2,previous=(args[4]-1)*stride*2;
    if(!incoming || !outgoing || overlaps(target,incoming) || overlaps(target,outgoing) ||
        !span(incoming,current,stride*2) || !span(outgoing,previous,stride*2))return null;
    tangents={incoming,outgoing,current,previous};
  }
  for(const offset of offsets)if(!span(samples,offset,stride))return null;
  return {target,samples,offsets,tangents};
}
function kernel(record) {
  if(!record.attempted) {
    record.attempted=true;
    try {
      const binary=apply(decode,host,[record.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      record.kernel=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath});
    }catch{record.failure='INTERPOLATION_INITIALIZATION';}
    record.base64=null;
  }
  return record.kernel;
}
export function tryAnimationInterpolant(token,result,values,stride,...args) {
  const record=apply(get,records,[token]);if(!record)return false;
  const current=record.statistics,fallback=reason=>{current.fallbackCalls++;current.lastFailure=reason;return false;};
  if(!platform(record))return fallback('INTERPOLATION_PLATFORM');
  let input=inputs(record,result,values,stride,args);if(!input)return fallback('INTERPOLATION_STORAGE');
  const runtime=kernel(record);if(!runtime)return fallback(record.failure);
  // Instantiation can encounter host policy hooks; do not reuse a source proof
  // across it. No application code runs while the packed arrays are borrowed.
  if(!platform(record) || !(input=inputs(record,result,values,stride,args)))return fallback('INTERPOLATION_STORAGE');
  let output;
  try {
    output=new F64(stride);
    const packed=new F64(input.offsets.length*stride);
    for(let key=0;key<input.offsets.length;key++)for(let i=0;i<stride;i++)
      packed[key*stride+i]=values[input.offsets[key]+i];
    if(record.mode==='linear')runtime.run(output,packed,stride,1,args[1],args[2],args[3]);
    else if(record.mode==='cubic')runtime.run(output,packed,stride,2*stride,stride,0,3*stride,...args.slice(4));
    else {
      const incoming=new F64(4*stride),outgoing=new F64(2*stride),t=input.tangents;
      for(let i=0;i<stride*2;i++) {
        incoming[stride*2+i]=t.incoming.view[t.current+i];outgoing[i]=t.outgoing.view[t.previous+i];
      }
      runtime.run(output,packed,stride,stride,0,incoming,outgoing,1,args[5],args[6],args[7]);
    }
  }catch{return fallback(runtime.diagnostics.lastGuardFailure??'INTERPOLATION_EXECUTION');}
  // Different engine NaN payloads must not escape through a typed-buffer alias.
  for(let i=0;i<stride;i++)if(output[i]!==output[i])return fallback('INTERPOLATION_NAN');
  for(let i=0;i<stride;i++)result[i]=output[i];
  current.wasmCalls++;current.lastFailure=null;return true;
}
export function animationInterpolantDiagnostics(Class) {
  return freeze({...apply(get,classes,[Class])?.statistics??stats()});
}
