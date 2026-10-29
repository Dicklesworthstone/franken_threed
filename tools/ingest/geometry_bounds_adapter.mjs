/** Closed bounds scans over source-owned buffers and Vector3 cells. The source
 * makeEmpty call has already executed. A miss resumes only its original scan;
 * a success materializes the same min/max and escaped private scratch objects.
 */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {isNumericObject} from './numeric_objects.mjs';
import {matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,integer=Number.isSafeInteger,freeze=Object.freeze;
const own=(object,key)=>{const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>object && own(object,key)?.value;
const mapGet=WeakMap.prototype.get,mapSet=WeakMap.prototype.set;
const get=(map,key)=>apply(mapGet,map,[key]),set=(map,key,value)=>apply(mapSet,map,[key,value]);
const F64=Float64Array,U8=Uint8Array,decode=globalThis.atob,charCode=String.prototype.charCodeAt;
const objectPrototype=Object.prototype,typed=prototype(F64.prototype);
const tag=descriptor(typed,Symbol.toStringTag).get,length=descriptor(typed,'length').get;
const classes=new WeakMap(),dispatches=new WeakMap(),statistics=new WeakMap();
const methodNames={box:['expandByPoint'],vector:['fromBufferAttribute','min','max'],attribute:['getX','getY','getZ']};

/** Compiler-emitted registrations refer only to exact source class definitions. */
export function registerGeometryBoundsClass(kind,Class,Child) {
  const proto=data(Class,'prototype'),parent=prototype(proto);
  const methods=methodNames[kind].map(name=>[name,data(proto,name)]);
  set(classes,proto,{kind,parent,methods});
  if(Child) {
    const child=data(Child,'prototype');if(prototype(child)===proto)set(classes,child,{kind,parent:proto,methods});
  }
}
export function createGeometryBoundsDispatch(variants,resolveMath) {
  const token=freeze({});set(dispatches,token,{resolveMath,variants:variants.map(value=>({...value,attempted:false}))});return token;
}
function shape(object,kind) {
  if(!isNumericObject(object))return false;
  const proto=prototype(object),known=get(classes,proto);
  if(!known || known.kind!==kind || prototype(proto)!==known.parent)return false;
  for(const [key,method]of known.methods) {
    if(!method)return false;
    let current=object;
    while(current) {
      const d=descriptor(current,key);
      if(d){if(!hasOwn(d,'value') || d.value!==method)return false;break;}
      current=prototype(current);
      const entry=get(classes,current);
      if(!entry || entry.kind!==kind || prototype(current)!==entry.parent)return false;
    }
  }
  return true;
}
function vector(object,state,offset) {
  if(!shape(object,'vector'))return false;
  const proto=prototype(object);
  if(prototype(proto)!==objectPrototype)return false;
  for(let i=0;i<3;i++) {
    const key=['x','y','z'][i],d=own(object,key);
    if(!d?.writable || typeof d.value!=='number' || descriptor(proto,key) || descriptor(objectPrototype,key))return false;
    state[offset+i]=d.value;
  }
  return true;
}
function inputs(owner,attribute,scratch) {
  if(!shape(owner,'box') || !shape(attribute,'attribute'))return null;
  const min=data(owner,'min'),max=data(owner,'max'),state=new F64(9);
  // Shared vector identities change the source's in-place recurrence. Keep
  // that uncommon shape on its exact original loop, without touching it here.
  if(min===max || min===scratch || max===scratch || !vector(min,state,0) ||
      !vector(max,state,3) || !vector(scratch,state,6))return null;
  const values=data(attribute,'array'),count=data(attribute,'count'),itemSize=data(attribute,'itemSize');
  if(data(attribute,'normalized')!==false || !integer(count) || count<0 || !integer(itemSize) || itemSize<3)return null;
  try {
    const name=apply(tag,values,[]),type=name==='Float32Array'?'f32[]':name==='Float64Array'?'f64[]':null;
    return type && apply(length,values,[])===count*itemSize?{min,max,scratch,values,state,count,itemSize,type}:null;
  }catch{return null;}
}
function kernel(record,variant) {
  if(!variant.attempted) {
    variant.attempted=true;
    try {
      const binary=apply(decode,globalThis,[variant.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      variant.kernel=instantiateNumericKernel(bytes,{resolveMath:record.resolveMath});
    }catch{variant.failure='GEOMETRY_BOUNDS_INITIALIZATION';}
    variant.base64=null;
  }
  return variant.kernel;
}
export function tryGeometryBoxBounds(owner,token,attribute,scratch) {
  if(!isNumericObject(owner) || !matrixNumericPlatformIntact())return false;
  let stats=get(statistics,owner);if(!stats){stats={wasmCalls:0,fallbackCalls:0,lastFailure:null};set(statistics,owner,stats);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;},record=get(dispatches,token);
  if(!record)return fallback('GEOMETRY_BOUNDS_PLATFORM');
  let input=inputs(owner,attribute,scratch);if(!input)return fallback('GEOMETRY_BOUNDS_GUARD');
  const variant=record.variants.find(value=>value.type===input.type),runtime=kernel(record,variant);
  if(!runtime)return fallback(variant.failure);
  // Host initialization can encounter a policy hook. Revalidate live objects
  // before borrowing data; the admitted scan and publication invoke no hooks.
  if(!matrixNumericPlatformIntact() || !(input=inputs(owner,attribute,scratch)))return fallback('GEOMETRY_BOUNDS_GUARD');
  try{runtime.run(input.values,input.state,input.count,input.itemSize);}
  catch{return fallback(runtime.diagnostics.lastGuardFailure??'GEOMETRY_BOUNDS_EXECUTION');}
  // Preserve source-produced NaN bits for later typed-buffer observations.
  for(let i=0;i<9;i++)if(input.state[i]!==input.state[i])return fallback('GEOMETRY_BOUNDS_NAN');
  for(let i=0;i<3;i++) {
    const target=[input.min,input.max,input.scratch][i];
    target.x=input.state[i*3];target.y=input.state[i*3+1];target.z=input.state[i*3+2];
  }
  stats.wasmCalls++;stats.lastFailure=null;return true;
}
export function geometryBoundsDiagnostics(owner) {
  return freeze({...get(statistics,owner)??{wasmCalls:0,fallbackCalls:0,lastFailure:null}});
}
