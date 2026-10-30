/** One guarded PropertyMixer numeric operation. The public buffer keeps its
 * identity; binding callbacks run in the source method after materialization. */
import {instantiateNumericKernel,numericArrayIterationIntact} from './numeric_kernel_runtime.mjs';
import {matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,freeze=Object.freeze,integer=Number.isSafeInteger;
const get=WeakMap.prototype.get,set=WeakMap.prototype.set,has=WeakSet.prototype.has,add=WeakSet.prototype.add;
const F64=Float64Array,U8=Uint8Array,decode=host.atob,charCode=String.prototype.charCodeAt,toSource=Function.prototype.toString;
const objects=new WeakSet(),records=new WeakMap(),classes=new WeakMap();
const own=(object,key)=>{const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>object && own(object,key)?.value;
const typed=prototype(F64.prototype),length=descriptor(typed,'length').get,buffer=descriptor(typed,'buffer').get;
const tag=descriptor(typed,Symbol.toStringTag).get,bufferLength=descriptor(ArrayBuffer.prototype,'byteLength').get;
const resizable=descriptor(ArrayBuffer.prototype,'resizable')?.get,immutable=descriptor(ArrayBuffer.prototype,'immutable')?.get;
const wasm=data(host,'WebAssembly'),nativeWasm=wasm && ['Module','Instance','Memory'].every(name=>
  typeof data(wasm,name)==='function' && apply(toSource,data(wasm,name),[])===`function ${name}() { [native code] }`);
const statistics=()=>({wasmCalls:0,fallbackCalls:0,lastFailure:null,operations:{accumulate:0,accumulateAdditive:0,apply:0}});

export function registerPropertyMixer(object){apply(add,objects,[object]);}
export function createPropertyMixerDispatch(Class,variants,maxMemoryBytes) {
  const token=freeze({}),methods={lerp:data(Class.prototype,'_lerp'),additive:data(Class.prototype,'_lerpAdditive'),
    identity:data(Class.prototype,'_setAdditiveIdentityNumeric')};
  const record={variants,maxMemoryBytes,methods,kernels:{},statistics:statistics()};
  apply(set,records,[token,record]);apply(set,classes,[Class,record]);return token;
}
const platform=()=>nativeWasm && numericArrayIterationIntact() && matrixNumericPlatformIntact();
function storage(view) {
  try {
    if(apply(tag,view,[])!=='Float64Array')return null;
    const backing=apply(buffer,view,[]),count=apply(length,view,[]);
    apply(bufferLength,backing,[]); // The intrinsic rejects SharedArrayBuffer.
    if(resizable && apply(resizable,backing,[]) || immutable && apply(immutable,backing,[]))return null;
    new U8(backing,0,0); // Reject detached storage, including empty views.
    return {view,count};
  }catch{return null;}
}
function inputs(owner,record,name,args) {
  if(!apply(has,objects,[owner]))return null;
  const values=data(owner,'buffer'),stride=data(owner,'valueSize'),target=storage(values);
  if(!target || typeof stride!=='number' || !integer(stride) || stride<0)return null;
  const weight=own(owner,'cumulativeWeight'),additive=own(owner,'cumulativeWeightAdditive');
  if(!weight || typeof weight.value!=='number' || !additive || typeof additive.value!=='number')return null;
  if(name!=='accumulateAdditive' && !weight.writable || name!=='accumulate' && !additive.writable)return null;
  for(let i=0;i<args.length;i++)if(typeof args[i]!=='number')return null;
  let count=0,offset=0,origIndex=0,addIndex=0,binding,writes=true;
  const span=start=>{
    if(!integer(start) || start<0 || start>target.count-stride)return false;
    const end=start+stride;if(end>count)count=end;return true;
  };
  if(name==='accumulate') {
    offset=args[0]*stride+stride;
    if(!span(0) || !span(offset) || weight.value!==0 && data(owner,'_mixBufferRegion')!==record.methods.lerp)return null;
  }else if(name==='accumulateAdditive') {
    addIndex=data(owner,'_addIndex');if(typeof addIndex!=='number')return null;
    offset=stride*addIndex;
    if(!span(0) || !span(offset) || data(owner,'_mixBufferRegionAdditive')!==record.methods.additive ||
        additive.value===0 && data(owner,'_setIdentity')!==record.methods.identity)return null;
  }else {
    const field=own(owner,'binding');if(!field)return null;binding=field.value;
    offset=args[0]*stride+stride;writes=weight.value<1 || additive.value>0;
    if(!span(stride) || !span(stride+stride) || !span(offset))return null;
    if(weight.value<1) {
      origIndex=data(owner,'_origIndex');if(typeof origIndex!=='number')return null;
      if(!span(stride*origIndex) || data(owner,'_mixBufferRegion')!==record.methods.lerp)return null;
    }
    if(additive.value>0) {
      addIndex=data(owner,'_addIndex');if(typeof addIndex!=='number')return null;
      if(!span(stride*addIndex) || data(owner,'_mixBufferRegionAdditive')!==record.methods.additive)return null;
    }
  }
  // Copy only the property regions in this operation, never an oversized tail.
  if(count>(record.maxMemoryBytes-1024)/16)return null;
  return {values,stride,count,offset,origIndex,addIndex,binding,writes,weight:weight.value,additive:additive.value};
}
function kernel(record,name) {
  let result=record.kernels[name];
  if(result===undefined) {
    result=null;
    try {
      const binary=apply(decode,host,[record.variants[name]]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      result=instantiateNumericKernel(bytes);
    }catch{/* Native initialization refusal resumes the unchanged source body. */}
    record.kernels[name]=result;record.variants[name]=null;
  }
  return result;
}
export function tryPropertyMixer(owner,token,name,...args) {
  const record=apply(get,records,[token]);if(!record)return null;
  const stats=record.statistics,fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return null;};
  if(!platform())return fallback('PROPERTY_MIXER_PLATFORM');
  let input=inputs(owner,record,name,args);if(!input)return fallback('PROPERTY_MIXER_GUARD');
  const runtime=kernel(record,name);if(!runtime)return fallback('PROPERTY_MIXER_INITIALIZATION');
  // Instantiation may encounter host policy hooks. Recheck all state before the
  // callback-free numerical transaction rather than relying on an earlier proof.
  if(!platform() || !(input=inputs(owner,record,name,args)))return fallback('PROPERTY_MIXER_GUARD');
  let packed,state,changed;
  try {
    packed=new F64(input.count);state=new F64([input.weight,input.additive]);
    for(let i=0;i<input.count;i++)packed[i]=input.values[i];
    changed=runtime.run(packed,state,input.stride,input.origIndex,input.addIndex,...args);
  }catch{return fallback(runtime.diagnostics.lastGuardFailure??'PROPERTY_MIXER_EXECUTION');}
  // Source NaN payloads can be observed through aliases of the public buffer.
  // No staged native write becomes visible before this refusal decision.
  for(let i=0;i<input.count;i++)if(packed[i]!==packed[i])return fallback('PROPERTY_MIXER_NAN');
  if(state[0]!==state[0] || state[1]!==state[1])return fallback('PROPERTY_MIXER_NAN');
  if(name==='apply'){owner.cumulativeWeight=state[0];owner.cumulativeWeightAdditive=state[1];}
  if(input.writes)for(let i=0;i<input.stride;i++)input.values[input.offset+i]=packed[input.offset+i];
  if(name==='accumulate')owner.cumulativeWeight=state[0];
  if(name==='accumulateAdditive')owner.cumulativeWeightAdditive=state[1];
  stats.wasmCalls++;stats.operations[name]++;stats.lastFailure=null;
  return name==='apply'?(changed?{binding:input.binding,buffer:input.values,offset:input.offset}:false):true;
}
export function propertyMixerDiagnostics(Class) {
  const value=apply(get,classes,[Class])?.statistics??statistics();
  return freeze({...value,operations:freeze({...value.operations})});
}
