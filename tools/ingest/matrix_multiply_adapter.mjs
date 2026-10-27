/** Guarded native Matrix4 multiplication; source-visible arrays keep identity.
 * This is a copying baseline, not a speedup or whole-update-island claim. */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
const host=globalThis,descriptor=Object.getOwnPropertyDescriptor,hasOwn=Object.hasOwn,apply=Reflect.apply;
const prototype=Object.getPrototypeOf,ownKeys=Reflect.ownKeys,freeze=Object.freeze;
const setHas=WeakSet.prototype.has,setAdd=WeakSet.prototype.add,mapGet=WeakMap.prototype.get,mapSet=WeakMap.prototype.set;
const own=(object,key)=>{const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>object && own(object,key)?.value;
const F64=Float64Array,U8=Uint8Array,decode=host.atob,charCode=String.prototype.charCodeAt;
const owners=new WeakSet(),arrays=new WeakSet(),dispatches=new WeakMap(),statistics=new WeakMap();
// The generic packing runtime uses ordinary collection methods and iterators.
// Refuse late replacements BEFORE entering it: callbacks there could invalidate
// destination proofs or observe work absent from the original matrix method.
const wasm=data(host,'WebAssembly'),platform=[],surfaces=[];
const remember=(object,key)=>platform.push([object,key,descriptor(object,key)]);
const surface=object=>{const keys=ownKeys(object);surfaces.push([object,prototype(object),keys]);
  for(let i=0;i<keys.length;i++)remember(object,keys[i]);};
for(const name of ['globalThis','WebAssembly','Array','Object','Number','JSON','Map','Set','TextDecoder','ArrayBuffer','Error'])remember(host,name);
for(const [object,keys]of [
  [Array,['isArray',Symbol.species]],[Object,['freeze']],[Number,['isInteger','isSafeInteger']],
  [JSON,['parse']],[Map.prototype,['get','set','has',Symbol.iterator]],
  [Set.prototype,['has','add',Symbol.iterator]],
])for(const key of keys)remember(object,key);
surface(Object.prototype);surface(Array.prototype);surface(U8.prototype);surface(F64.prototype);
// Budget failures construct NumericKernelGuardError and use instanceof. Keep
// their inherited name/code setters and Symbol.hasInstance route callback-free.
surface(Error);surface(Error.prototype);surface(Function.prototype);
for(const iterator of [[][Symbol.iterator](),new Map()[Symbol.iterator](),new Set()[Symbol.iterator]()])
  remember(prototype(iterator),'next');
remember(ArrayBuffer.prototype,'byteLength');
remember(prototype(F64.prototype),'length');
for(const name of ['Array','Object','Number','Map','Set','TextDecoder','ArrayBuffer']) {
  const constructor=data(host,name);if(constructor)remember(constructor,'prototype');
}
const decoder=data(host,'TextDecoder'),decoderPrototype=data(decoder,'prototype');
if(decoderPrototype)remember(decoderPrototype,'decode');
if(wasm)for(const name of ['Module','Instance','Memory'])remember(wasm,name);
const moduleType=data(wasm,'Module'),instanceType=data(wasm,'Instance'),memoryType=data(wasm,'Memory');
const instancePrototype=data(instanceType,'prototype'),memoryPrototype=data(memoryType,'prototype');
if(moduleType)for(const name of ['customSections','imports','exports'])remember(moduleType,name);
if(instanceType){remember(instanceType,'prototype');remember(instancePrototype,'exports');}
if(memoryType){remember(memoryType,'prototype');remember(memoryPrototype,'grow');remember(memoryPrototype,'buffer');}
function platformIntact() {
  if(!wasm)return false;
  // Indexed traversal itself must not consult a potentially replaced iterator.
  for(let i=0;i<surfaces.length;i++) {
    const entry=surfaces[i],keys=ownKeys(entry[0]),expected=entry[2];
    if(prototype(entry[0])!==entry[1] || keys.length!==expected.length)return false;
    for(let j=0;j<keys.length;j++)if(keys[j]!==expected[j])return false;
  }
  for(let i=0;i<platform.length;i++) {
    const entry=platform[i],current=descriptor(entry[0],entry[1]),expected=entry[2];
    if(!current || !expected){if(current!==expected)return false;continue;}
    if(hasOwn(current,'value')!==hasOwn(expected,'value'))return false;
    if(hasOwn(expected,'value')){if(current.value!==expected.value || current.writable!==expected.writable)return false;}
    else if(current.get!==expected.get || current.set!==expected.set)return false;
    if(current.enumerable!==expected.enumerable || current.configurable!==expected.configurable)return false;
  }
  return true;
}

/** Emitted only at verified base-constructor and array-literal allocations.
 * A proxy, replacement array or borrowed method never acquires its target's witness. */
export function registerMatrixObject(object) {apply(setAdd,owners,[object]);}
export function registerMatrixElements(array) {apply(setAdd,arrays,[array]);return array;}
export function createMatrixMultiplyDispatch(base64) {
  const token=freeze({});
  apply(mapSet,dispatches,[token,{base64,attempted:false,kernel:null}]);
  return token;
}
function snapshot(matrix,writable) {
  if(!apply(setHas,owners,[matrix]))return null;
  const array=data(matrix,'elements');
  if(!apply(setHas,arrays,[array]))return null;
  const staged=new F64(16);
  for(let i=0;i<16;i++) {
    const d=own(array,i);
    if(!d || typeof d.value!=='number' || (writable && !d.writable))return null;
    staged[i]=d.value;
  }
  return {array,staged};
}
function inputs(owner,left,right) {
  // Descriptor-only checks on witnessed ordinary allocations invoke no user
  // getters or proxy traps. Unknown objects resume the complete source method.
  const a=snapshot(left,false),b=snapshot(right,false),target=snapshot(owner,true);
  return a && b && target?{a,b,target}:null;
}
function kernel(record) {
  if(!record.attempted) {
    record.attempted=true;
    try {
      const binary=apply(decode,host,[record.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      record.kernel=instantiateNumericKernel(bytes);
    }catch{record.failure='MATRIX_MULTIPLY_INITIALIZATION';}
    record.base64=null;
  }
  return record.kernel;
}
export function tryMatrixMultiply(owner,token,left,right) {
  if(!apply(setHas,owners,[owner]))return false;
  let stats=apply(mapGet,statistics,[owner]);
  if(!stats){stats={wasmCalls:0,fallbackCalls:0,lastFailure:null};apply(mapSet,statistics,[owner,stats]);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  const record=apply(mapGet,dispatches,[token]);
  if(!record || !platformIntact())return fallback('MATRIX_MULTIPLY_PLATFORM');
  let input=inputs(owner,left,right);
  if(!input)return fallback('MATRIX_MULTIPLY_GUARD');
  const runtime=kernel(record);
  if(!runtime)return fallback(record.failure);
  // Initialization can encounter host policy hooks. Discard every prior
  // snapshot before executing; no callbacks occur while numeric data is borrowed.
  if(!platformIntact() || !(input=inputs(owner,left,right)))return fallback('MATRIX_MULTIPLY_GUARD');
  try {runtime.run(input.a.staged,input.b.staged,input.target.staged);}
  catch{return fallback(runtime.diagnostics.lastGuardFailure??'MATRIX_MULTIPLY_EXECUTION');}
  // The source reads all operands before writing, so identical matrices and
  // shared witnessed element arrays preserve multiply/premultiply aliasing.
  // Every destination is already a writable own data property: publication
  // cannot call user code or fall back after any visible write.
  for(let row=0;row<4;row++)for(let column=0;column<4;column++) {
    const index=row+column*4;input.target.array[index]=input.target.staged[index];
  }
  stats.wasmCalls++;stats.lastFailure=null;return true;
}
export function matrixMultiplyDiagnostics(object) {
  return freeze({...apply(mapGet,statistics,[object])??{wasmCalls:0,fallbackCalls:0,lastFailure:null}});
}

/** Compiler-private shared proof for a larger source-derived transform batch.
 * Callers validate the platform before borrowing these ordinary numeric cells. */
export function matrixNumericPlatformIntact(){return platformIntact();}
export function matrixNumericElements(matrix,writable=false){
  if(!apply(setHas,owners,[matrix]))return null;
  const array=data(matrix,'elements');if(!apply(setHas,arrays,[array]))return null;
  for(let i=0;i<16;i++){
    const d=own(array,i);if(!d || typeof d.value!=='number' || (writable && !d.writable))return null;
  }
  return array;
}
