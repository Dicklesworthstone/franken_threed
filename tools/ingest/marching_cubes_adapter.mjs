/**
 * Compiler-owned bridge for the source-verified MarchingCubes addon. Construct
 * genuine retained objects and keep every original export, prototype and field.
 * Instrument only its numeric update body; the original method identity and
 * publication tail remain in place. All other addon methods stay retained.
 *
 * Registration is emitted INSIDE the verified base EventDispatcher constructor
 * where `this` is freshly allocated, never an arbitrary caller-owned object.
 * Unknown owners/materials, including proxies around known objects, take the
 * original path without property inspection. Bootstrap trusted platform APIs.
 */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
const apply=Reflect.apply, descriptor=Object.getOwnPropertyDescriptor, hasOwn=Object.hasOwn;
const isInteger=Number.isInteger, U8=Uint8Array, decode=globalThis.atob, charCodeAt=String.prototype.charCodeAt;
const empty=new Float32Array(0);
// Do not invoke replacement platform getters/constructors during admission.
const data=(object,key)=>object && descriptor(object,key)?.value;
const wasm=data(globalThis,'WebAssembly');
const moduleType=data(wasm,'Module'),instanceType=data(wasm,'Instance'),memoryType=data(wasm,'Memory');
const memoryPrototype=memoryType?.prototype;
const moduleMethods=['customSections','imports','exports'].map(key=>[key,data(moduleType,key)]);
const grow=data(memoryPrototype,'grow'),bufferGetter=memoryPrototype && descriptor(memoryPrototype,'buffer')?.get;
function platformIntact() {
  if (!wasm || data(globalThis,'WebAssembly')!==wasm || data(wasm,'Module')!==moduleType ||
      data(wasm,'Instance')!==instanceType || data(wasm,'Memory')!==memoryType ||
      data(memoryType,'prototype')!==memoryPrototype || data(memoryPrototype,'grow')!==grow ||
      descriptor(memoryPrototype,'buffer')?.get!==bufferGetter) return false;
  for (let i=0;i<moduleMethods.length;i++) {
    const [key,value]=moduleMethods[i];if (data(moduleType,key)!==value) return false;
  }
  return true;
}
const objects=new WeakSet(),dispatches=new WeakMap(),updates=new WeakMap();
const own=(object,key)=>{
  const d=descriptor(object,key);
  return d && hasOwn(d,'value') ? d : null;
};
const names=['size','size2','size3','halfsize','delta','yd','zd','isolation','field','normal_cache',
  'palette','positionArray','normalArray','material','geometry','enableUvs','enableColors','count'];

/** Called only by the verified base constructor with its genuine fresh `this`. */
export function registerMarchingCubesObject(object) { objects.add(object); }

/** No binary decoding or Wasm allocation until the first admitted update. */
export function createMarchingCubesDispatch(base64,edgeTable,triTable) {
  if (typeof base64!=='string' || base64.length>1024*1024) throw new TypeError('Expected bounded compiled Wasm');
  const token=Object.freeze({});
  dispatches.set(token,{base64,edgeTable,triTable,attempted:false,kernel:null,initializationFailure:null});
  return token;
}
function kernelFor(record) {
  if (!record.attempted) {
    record.attempted=true;
    try {
      const binary=apply(decode,globalThis,[record.base64]),bytes=new U8(binary.length);
      for (let k=0;k<binary.length;k++) bytes[k]=apply(charCodeAt,binary,[k]);
      // Independent storage is essential: do NOT enable preserveAliasing.
      record.kernel=instantiateNumericKernel(bytes);
    } catch {
      // Host policy may throw arbitrary values with effectful accessors.
      record.initializationFailure='MARCHING_CUBES_INITIALIZATION';
    }
    record.base64=null;
  }
  return record.kernel;
}
function snapshot(object) {
  const values={};
  for (const key of names) {
    const d=own(object,key);
    if (!d || (key==='count' && !d.writable)) return null;
    values[key]=d.value;
  }
  const s=values.size;
  if (!isInteger(s) || s<1 || s>256 || values.size2!==s*s || values.size3!==s*s*s ||
      values.halfsize!==s/2 || values.delta!==2/s || values.yd!==s || values.zd!==s*s ||
      typeof values.isolation!=='number' || typeof values.enableUvs!=='boolean' || typeof values.enableColors!=='boolean') return null;
  // WeakSet.has never invokes a Proxy's traps. Only a compiler registration of
  // the real constructed identity authorizes own-descriptor inspection.
  if (!objects.has(values.material)) return null;
  const shading=own(values.material,'flatShading');
  if (!shading) return null;
  let uvs=null,colors=null;
  if (values.enableUvs) {const d=own(object,'uvArray');if (!d) return null;uvs=d.value;}
  if (values.enableColors) {const d=own(object,'colorArray');if (!d) return null;colors=d.value;}
  return {values,uvs:uvs??empty,colors:colors??empty,flat:shading.value===true ? 1 : 0};
}

/**
 * Compiler-emitted branch at the start of the original update function. False
 * means execute its retained numeric body once; true skips only that body.
 * The original geometry/attribute/warning tail ALWAYS runs in the source method,
 * outside this helper's exception handling. Its identity/name/length are not
 * replaced, and it retains the construction geometry and maxPolyCount closure.
 */
export function tryMarchingCubesUpdate(receiver,object,token,edgePositions,edgeNormals,edgeColors) {
  let stats=updates.get(object);
  if (!stats) {stats={wasmCalls:0,fallbackCalls:0,lastFailure:null,token:null};updates.set(object,stats);}
  stats.token=token;
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  // No inspection of a borrowed/proxied receiver: preserve original this/scope.
  if (receiver!==object || !objects.has(object)) return fallback('MARCHING_CUBES_RECEIVER');
  const record=dispatches.get(token);
  if (!record) return fallback('MARCHING_CUBES_UNREGISTERED');
  if (!snapshot(object)) return fallback('MARCHING_CUBES_OBJECT_GUARD');
  if (!platformIntact()) return fallback('MARCHING_CUBES_PLATFORM');
  const kernel=kernelFor(record);
  if (!kernel) return fallback(record.initializationFailure);
  // Initialization can encounter host policy hooks. Re-read admitted data.
  const input=snapshot(object);
  if (!input) return fallback('MARCHING_CUBES_OBJECT_GUARD');
  const v=input.values;
  let count;
  try {
    count=kernel.run(v.field,v.normal_cache,v.palette,v.positionArray,v.normalArray,input.uvs,input.colors,
      edgePositions,edgeNormals,edgeColors,record.edgeTable,record.triTable,
      v.size,v.size2,v.halfsize,v.delta,v.yd,v.zd,v.isolation,
      input.flat,v.enableUvs ? 1 : 0,v.enableColors ? 1 : 0);
  } catch {return fallback(kernel.diagnostics.lastGuardFailure ?? 'MARCHING_CUBES_EXECUTION');}
  // Own writable data count was guarded. No caller code has run since the last
  // snapshot. This is after native output publication: no fallback below here.
  object.count=count;
  stats.wasmCalls++;stats.lastFailure=null;
  return true;
}

/** Opt-in diagnostics; the generated Three module gains no new public exports. */
export function marchingCubesDiagnostics(object) {
  const stats=updates.get(object);
  if (!stats) return Object.freeze({wasmCalls:0,fallbackCalls:0,lastFailure:null,initialized:false,kernel:null});
  const dispatch=dispatches.get(stats.token);
  return Object.freeze({wasmCalls:stats.wasmCalls,fallbackCalls:stats.fallbackCalls,lastFailure:stats.lastFailure,
    initialized:dispatch?.attempted??false,kernel:dispatch?.kernel?.diagnostics??null});
}
