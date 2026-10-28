/** A guarded, synchronous Euler/quaternion transaction for ordinary source
 * objects. Math.sin/cos stay in JS to preserve the engine's exact source values.
 * Packing and identity-preserving publication are included; no speed claim. */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
import {isTransformChildren} from './transform_batch_adapter.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const hasOwn=Object.hasOwn,freeze=Object.freeze,create=Object.create,source=Function.prototype.toString;
const get=WeakMap.prototype.get,set=WeakMap.prototype.set,has=WeakSet.prototype.has,add=WeakSet.prototype.add;
const setHas=Set.prototype.has,setAdd=Set.prototype.add,F64=Float64Array,U8=Uint8Array,SetType=Set;
const decode=host.atob,charCode=String.prototype.charCodeAt,isInteger=Number.isSafeInteger;
const kernels=new WeakMap(),plans=new WeakMap(),owners=new WeakMap(),numbers=new WeakMap(),dataObjects=new WeakSet(),statistics=new WeakMap();
const own=(object,key)=>{if(object==null)return null;const d=descriptor(object,key);return d && hasOwn(d,'value')?d:null;};
const data=(object,key)=>own(object,key)?.value;
const math=data(host,'Math'),cos=data(math,'cos'),sin=data(math,'sin');
const native=(value,name)=>typeof value==='function' && apply(source,value,[])===`function ${name}() { [native code] }`;
const nativeTrig=native(cos,'cos') && native(sin,'sin');
const wasm=data(host,'WebAssembly');
const nativeWasm=wasm && ['Module','Instance','Memory'].every(name=>native(data(wasm,name),name));
const fields=['_x','_y','_z'],quaternionFields=['_x','_y','_z','_w'];
const lookup=(map,key)=>apply(get,map,[key]);

export function createEulerUpdateKernel(base64,orders,maxMemoryPages){
  const token=freeze({}),capacity=(maxMemoryPages*65536-256)/(17*8);
  apply(set,kernels,[token,{base64,orders,classes:create(null),maxRows:capacity-capacity%1,attempted:false,kernel:null}]);
  return token;
}
export function registerEulerUpdateClass(token,name,Class){
  const record=lookup(kernels,token),proto=data(Class,'prototype'),properties=create(null);
  for(const key of ['x','y','z','set','setFromEuler'])properties[key]=descriptor(proto,key);
  record.classes[name]={proto,parent:prototype(proto),properties};
}
export function registerEulerUpdateNumeric(object,token,name){
  apply(set,numbers,[object,{record:lookup(kernels,token),name}]);
}
export function registerEulerUpdateOwner(object,rotation,quaternion,callback,token){
  apply(set,owners,[object,{record:lookup(kernels,token),rotation,quaternion,callback}]);
}
export function registerEulerUpdateData(object){apply(add,dataObjects,[object]);return object;}
export function createEulerUpdatePlan(metadata){
  const token=freeze({}),operations=new F64(9);
  for(let i=0;i<9;i++)operations[i]=metadata.operations[i];
  apply(set,plans,[token,{boundField:metadata.boundField,childrenBound:metadata.childrenBound===true,
    path:freeze(metadata.path),operations}]);
  return token;
}
function platform(){return nativeTrig && nativeWasm && matrixNumericPlatformIntact() &&
  data(host,'Math')===math && data(math,'cos')===cos && data(math,'sin')===sin;}
function sameDescriptor(actual,expected){
  if(!actual || !expected)return actual===expected;
  return hasOwn(actual,'value')===hasOwn(expected,'value') && actual.value===expected.value &&
    actual.get===expected.get && actual.set===expected.set;
}
function numeric(object,name,record,writable=false){
  const witness=lookup(numbers,object);
  if(!witness || witness.name!==name || record && witness.record!==record)return null;
  const cls=witness.record?.classes[name];
  if(!cls || prototype(object)!==cls.proto || prototype(cls.proto)!==cls.parent)return null;
  if(name==='Euler')for(const key of ['x','y','z']){
    if(descriptor(object,key) || !sameDescriptor(descriptor(cls.proto,key),cls.properties[key]))return null;
  }
  const keys=name==='Euler'?fields:quaternionFields,values=[];
  for(let i=0;i<keys.length;i++){
    const d=own(object,keys[i]);if(!d || typeof d.value!=='number' || writable && !d.writable)return null;
    values[i]=d.value;
  }
  return {cls,values};
}
function method(object,key,cls){
  const value=descriptor(object,key)??descriptor(cls.proto,key),expected=cls.properties[key];
  return !!value && hasOwn(value,'value') && value.value===expected?.value && typeof value.value==='function';
}
function term(a,b,operation){return operation===0?a+b:operation===1?a-b:operation===2?a*b:a/b;}
function input(group,bound,plan){
  const root=lookup(owners,group),children=data(group,'children');
  if(!root || !isTransformChildren(children))return null;
  let count;
  if(plan.childrenBound)count=data(children,'length');
  else if(plan.boundField===null)count=bound;
  else {
    if(!apply(has,dataObjects,[bound]))return null;
    count=data(bound,plan.boundField);
  }
  if(!isInteger(count) || count<0 || count>root.record.maxRows || count>data(children,'length'))return null;
  const state=new F64(count*17),entries=[],targets=new SetType(),speeds=[];
  for(let i=0;i<count;i++){
    const child=data(children,i),owner=lookup(owners,child);
    if(!owner || owner.record!==root.record || apply(setHas,targets,[child]) ||
      data(child,'rotation')!==owner.rotation || data(child,'quaternion')!==owner.quaternion)return null;
    apply(setAdd,targets,[child]);
    const euler=numeric(owner.rotation,'Euler',root.record,true),quaternion=numeric(owner.quaternion,'Quaternion',root.record,true);
    const order=own(owner.rotation,'_order');
    if(!euler || !quaternion || !order || !order.writable ||
      !method(owner.rotation,'set',euler.cls) || !method(owner.quaternion,'setFromEuler',quaternion.cls) ||
      data(owner.rotation,'_onChangeCallback')!==owner.callback ||
      apply(setHas,targets,[owner.rotation]) || apply(setHas,targets,[owner.quaternion]))return null;
    let orderIndex=-1;
    for(let j=0;j<root.record.orders.length;j++)if(order.value===root.record.orders[j]){orderIndex=j;break;}
    if(orderIndex<0)return null;
    apply(setAdd,targets,[owner.rotation]);apply(setAdd,targets,[owner.quaternion]);
    let speed=child;
    for(let j=0;j<plan.path.length;j++){
      if(j>0 && !apply(has,dataObjects,[speed]))return null;
      speed=data(speed,plan.path[j]);
    }
    const velocity=numeric(speed,'Euler',null,false);if(!velocity)return null;
    speeds[i]=speed;
    const offset=i*17,ops=plan.operations,angles=[];
    for(let j=0;j<3;j++){
      state[offset+j]=euler.values[j];state[offset+j+3]=velocity.values[j];
      angles[j]=term(euler.values[ops[j*3]],velocity.values[ops[j*3+1]],ops[j*3+2]);
    }
    // Pinned Quaternion.setFromEuler takes cos(x/2),cos(y/2),cos(z/2),
    // then the corresponding sin calls. Native Number inputs cannot call user
    // code; modified Math functions refuse the entire attempt before packing.
    for(let j=0;j<3;j++)state[offset+6+j]=cos(angles[j]/2);
    for(let j=0;j<3;j++)state[offset+9+j]=sin(angles[j]/2);
    state[offset+12]=orderIndex;
    entries[i]={rotation:owner.rotation,quaternion:owner.quaternion,order:order.value,offset};
  }
  // Read-only speeds can be shared, but none may alias ANY output, including
  // a future target. Otherwise earlier source writes could change later reads.
  for(let i=0;i<speeds.length;i++)if(apply(setHas,targets,[speeds[i]]))return null;
  return {record:root.record,state,count,entries};
}
function kernel(record){
  if(!record.attempted){
    record.attempted=true;
    try{
      const binary=apply(decode,host,[record.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      record.kernel=instantiateNumericKernel(bytes);
    }catch{record.failure='EULER_UPDATE_INITIALIZATION';}
    record.base64=null;
  }
  return record.kernel;
}
export function tryEulerUpdateLoop(group,bound,token){
  if(!lookup(owners,group))return false;
  let stats=lookup(statistics,group);
  if(!stats){stats={wasmCalls:0,objects:0,fallbackCalls:0,lastFailure:null};apply(set,statistics,[group,stats]);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  const plan=lookup(plans,token);
  if(!plan || !platform())return fallback('EULER_UPDATE_PLATFORM');
  let prepared,runtime;
  try{
    prepared=input(group,bound,plan);if(!prepared)return fallback('EULER_UPDATE_GUARD');
    if(prepared.count===0)return true;
    const initialized=prepared.record.attempted;
    runtime=kernel(prepared.record);if(!runtime)return fallback(prepared.record.failure);
    if(!platform())return fallback('EULER_UPDATE_PLATFORM');
    if(!initialized && !(prepared=input(group,bound,plan)))return fallback('EULER_UPDATE_GUARD');
    runtime.run(prepared.state,plan.operations,prepared.count);
  }catch{return fallback(runtime?.diagnostics.lastGuardFailure??'EULER_UPDATE_EXECUTION');}
  // Publication is outside the fallback catch. Each destination is a witnessed
  // ordinary allocation with writable own cells, and every callback that would
  // run was proven to be the exact inlined source rotation callback.
  for(let i=0;i<prepared.entries.length;i++){
    const entry=prepared.entries[i],offset=entry.offset;
    for(let j=0;j<3;j++)entry.rotation[fields[j]]=prepared.state[offset+j];
    entry.rotation._order=entry.order;
    for(let j=0;j<4;j++)entry.quaternion[quaternionFields[j]]=prepared.state[offset+13+j];
  }
  stats.wasmCalls++;stats.objects+=prepared.count;stats.lastFailure=null;return true;
}
export function eulerUpdateDiagnostics(group){
  return freeze({...lookup(statistics,group)??{wasmCalls:0,objects:0,fallbackCalls:0,lastFailure:null}});
}
