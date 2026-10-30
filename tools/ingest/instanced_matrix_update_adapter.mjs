/** Guarded whole-loop instance-matrix updates. Public storage and both scratch
 * Matrix4 objects keep identity; upload requests remain in the source program. */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {matrixNumericElements} from './matrix_multiply_adapter.mjs';
import {eulerUpdateNumericSnapshot,eulerUpdatePlatformIntact} from './euler_update_adapter.mjs';
import {isNumericObject} from './numeric_objects.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,freeze=Object.freeze,create=Object.create,integer=Number.isSafeInteger;
const get=WeakMap.prototype.get,set=WeakMap.prototype.set,has=WeakSet.prototype.has,add=WeakSet.prototype.add;
const F64=Float64Array,U8=Uint8Array,decode=host.atob,charCode=String.prototype.charCodeAt;
const typed=prototype(F64.prototype),tag=descriptor(typed,Symbol.toStringTag).get;
const length=descriptor(typed,'length').get,buffer=descriptor(typed,'buffer').get;
const byteLength=descriptor(ArrayBuffer.prototype,'byteLength').get,resizable=descriptor(ArrayBuffer.prototype,'resizable')?.get;
const immutable=descriptor(ArrayBuffer.prototype,'immutable')?.get;
const own=(object,key)=>{const d=descriptor(object,key);return d&&hasOwn(d,'value')?d:null;};
const data=(object,key)=>object&&own(object,key)?.value;
const math=data(host,'Math'),cos=data(math,'cos'),sin=data(math,'sin');
const records=new WeakMap(),owners=new WeakMap(),arrays=new WeakSet(),statistics=new WeakMap();
const lookup=(map,key)=>apply(get,map,[key]);
const methods=['makeRotationFromEuler','multiply','premultiply','multiplyMatrices','fromArray','toArray','getMatrixAt','setMatrixAt'];

export function createInstancedMatrixKernel(base64,orders,maxMemoryPages){
  const token=freeze({}),capacity=(maxMemoryPages*65536-128)/(39*8);
  apply(set,records,[token,{base64,orders,maxRows:capacity-capacity%1,classes:create(null),attempted:false,kernel:null}]);
  return token;
}
export function registerInstancedMatrixClass(token,name,Class){
  const record=lookup(records,token),proto=data(Class,'prototype'),values=create(null);
  for(let i=0;i<methods.length;i++)values[methods[i]]=data(proto,methods[i]);
  record.classes[name]={proto,parent:prototype(proto),methods:values};
}
export function registerInstancedMatrixOwner(object,token){apply(set,owners,[object,lookup(records,token)]);}
export function registerInstancedMatrixArray(array){apply(add,arrays,[array]);return array;}
function method(object,key,cls){
  if(!cls||prototype(object)!==cls.proto||prototype(cls.proto)!==cls.parent)return false;
  const value=descriptor(object,key)??descriptor(cls.proto,key);
  return !!value&&hasOwn(value,'value')&&typeof value.value==='function'&&value.value===cls.methods[key];
}
function input(record,mesh,speeds,rotation,scratch,reverse){
  const count=data(mesh,'count');
  if(!integer(count)||count<0||count>record.maxRows)return null;
  if(count===0)return {count}; // The source does not observe any body operands.
  if(!method(mesh,'getMatrixAt',record.classes.InstancedMesh)||!method(mesh,'setMatrixAt',record.classes.InstancedMesh)||
    !apply(has,arrays,[speeds])||data(speeds,'length')<count||rotation===scratch)return null;
  const matrix=record.classes.Matrix4;
  const rotate=matrixNumericElements(rotation,true),result=matrixNumericElements(scratch,true);
  if(!rotate||!result||rotate===result||!method(rotation,'makeRotationFromEuler',matrix)||
    !method(scratch,reverse?'premultiply':'multiply',matrix)||!method(scratch,'multiplyMatrices',matrix)||
    !method(scratch,'fromArray',matrix)||!method(scratch,'toArray',matrix))return null;
  const attribute=data(mesh,'instanceMatrix');
  if(!isNumericObject(attribute))return null;
  const storage=data(attribute,'array');
  try {
    const kind=apply(tag,storage,[]),backing=apply(buffer,storage,[]);
    if(!['Float32Array','Float64Array'].includes(kind)||apply(length,storage,[])<count*16||
      !apply(byteLength,backing,[])||(resizable&&apply(resizable,backing,[]))||
      (immutable&&apply(immutable,backing,[])))return null;
  }catch{return null;}
  const state=new F64(count*39);
  for(let i=0;i<count;i++){
    const speed=data(speeds,i),angle=eulerUpdateNumericSnapshot(speed);
    if(!angle)return null;
    let order=-1;
    for(let j=0;j<record.orders.length;j++)if(angle.order===record.orders[j]){order=j;break;}
    if(order<0)return null;
    const offset=i*39;
    for(let j=0;j<16;j++)state[offset+j]=storage[i*16+j];
    // Match the source's cos(x),sin(x),cos(y),sin(y),cos(z),sin(z) order.
    for(let j=0;j<3;j++){state[offset+16+j*2]=cos(angle.values[j]);state[offset+17+j*2]=sin(angle.values[j]);}
    state[offset+22]=order;
  }
  return {count,state,storage,rotate,result};
}
function kernel(record){
  if(!record.attempted){
    record.attempted=true;
    try{
      const binary=apply(decode,host,[record.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      record.kernel=instantiateNumericKernel(bytes);
    }catch{record.failure='INSTANCE_MATRIX_INITIALIZATION';}
    record.base64=null;
  }
  return record.kernel;
}
export function tryInstancedMatrixLoop(mesh,speeds,rotation,scratch,reverse=false){
  const record=lookup(owners,mesh);if(!record)return false;
  let stats=lookup(statistics,mesh);
  if(!stats){stats={wasmCalls:0,instances:0,fallbackCalls:0,lastFailure:null};apply(set,statistics,[mesh,stats]);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  if(!eulerUpdatePlatformIntact()||data(host,'Math')!==math||data(math,'cos')!==cos||data(math,'sin')!==sin)
    return fallback('INSTANCE_MATRIX_PLATFORM');
  let prepared,runtime;
  try{
    prepared=input(record,mesh,speeds,rotation,scratch,reverse);if(!prepared)return fallback('INSTANCE_MATRIX_GUARD');
    if(prepared.count===0)return true;
    const initialized=record.attempted;
    runtime=kernel(record);if(!runtime)return fallback(record.failure);
    if(!eulerUpdatePlatformIntact())return fallback('INSTANCE_MATRIX_PLATFORM');
    if(!initialized&&!(prepared=input(record,mesh,speeds,rotation,scratch,reverse)))return fallback('INSTANCE_MATRIX_GUARD');
    runtime.run(prepared.state,prepared.count,reverse?1:0);
  }catch{return fallback(runtime?.diagnostics.lastGuardFailure??'INSTANCE_MATRIX_EXECUTION');}
  // Every write target is native unshared storage or writable witnessed cells.
  // No callback or source observation occurs between these original iterations.
  // Float32 assignment rounds only the instance stores; scratch matrices retain
  // the full Number result of the last iteration, including projective terms.
  for(let i=0;i<prepared.count;i++)for(let j=0;j<16;j++)prepared.storage[i*16+j]=prepared.state[i*39+j];
  const last=(prepared.count-1)*39;
  for(let j=0;j<16;j++){prepared.rotate[j]=prepared.state[last+23+j];prepared.result[j]=prepared.state[last+j];}
  stats.wasmCalls++;stats.instances+=prepared.count;stats.lastFailure=null;return true;
}
export function instancedMatrixUpdateDiagnostics(mesh){
  return freeze({...lookup(statistics,mesh)??{wasmCalls:0,instances:0,fallbackCalls:0,lastFailure:null}});
}
