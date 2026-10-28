/** A copying bulk baseline for source-pinned ordinary subtrees. All public Euler
 * callbacks and owner update effects stay in their original source positions.
 * Each call rebuilds guarded DFS order; this is not a persistent dirty plan. */
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {matrixNumericElements,matrixNumericPlatformIntact} from './matrix_multiply_adapter.mjs';
const host=globalThis,apply=Reflect.apply,descriptor=Object.getOwnPropertyDescriptor,prototype=Object.getPrototypeOf;
const ownKey=Object.hasOwn,freeze=Object.freeze,create=Object.create,F64=Float64Array,U8=Uint8Array,S=Set;
const weakHas=WeakSet.prototype.has,weakAdd=WeakSet.prototype.add,mapGet=WeakMap.prototype.get,mapSet=WeakMap.prototype.set;
const setHas=Set.prototype.has,setAdd=Set.prototype.add,decode=host.atob,charCode=String.prototype.charCodeAt,floor=Math.floor;
const objects=new WeakSet(),numbers=new WeakSet(),arrays=new WeakSet(),dispatches=new WeakMap(),statistics=new WeakMap();
const own=(object,key)=>{const d=descriptor(object,key);return d && ownKey(d,'value')?d:null;};
const data=(object,key)=>own(object,key)?.value;
const nodeKinds=['Object3D','Group','Scene','Bone','Mesh','InstancedMesh'],matrixKinds=['Matrix4'],fields=['position','quaternion','scale'];
const xyz=['x','y','z'],xyzw=['_x','_y','_z','_w'];
export function registerTransformObject(object){apply(weakAdd,objects,[object]);}
export function registerTransformNumeric(object){apply(weakAdd,numbers,[object]);}
export function registerTransformChildren(array){apply(weakAdd,arrays,[array]);return array;}
// Shared by source update islands; inspection never invokes a Proxy trap.
export function isTransformChildren(array){return apply(weakHas,arrays,[array]);}
export function createTransformBatchDispatch(base64,stride,maxMemoryPages){
  const token=freeze({});
  apply(mapSet,dispatches,[token,{base64,stride,maxRows:floor((maxMemoryPages*65536-128)/(stride*8)),
    attempted:false,kernel:null,classes:create(null)}]);
  return token;
}
/** Emitted immediately after each pinned class definition, before later source
 * initializers can expose its methods through ordinary application callbacks. */
export function registerTransformBatchClass(token,name,Class){
  const record=apply(mapGet,dispatches,[token]),proto=data(Class,'prototype');
  record.classes[name]={proto,parent:prototype(proto),updateMatrix:data(proto,'updateMatrix'),
    updateMatrixWorld:data(proto,'updateMatrixWorld'),compose:data(proto,'compose'),multiplyMatrices:data(proto,'multiplyMatrices')};
}
function method(object,key,expected,kinds,classes){
  if(typeof expected!=='function')return false;
  let current=object,remaining=4;
  while(current && remaining--){
    if(current!==object){
      let known=null;
      for(let i=0;i<kinds.length;i++){const value=classes[kinds[i]];if(value?.proto===current){known=value;break;}}
      if(!known || prototype(current)!==known.parent)return false;
    }
    const d=descriptor(current,key);
    if(d)return ownKey(d,'value') && d.value===expected;
    current=prototype(current);
  }
  return false;
}
function vector(object,keys,state,offset,seen){
  if(!apply(weakHas,numbers,[object]) || apply(setHas,seen,[object]))return false;
  apply(setAdd,seen,[object]);
  for(let i=0;i<keys.length;i++){
    const d=own(object,keys[i]);if(!d || typeof d.value!=='number')return false;state[offset+i]=d.value;
  }
  return true;
}
function hierarchy(record,owner,children){
  const entries=[],nodes=new S(),objectClass=record.classes.Object3D;
  const count=data(children,'length');
  if(!objectClass || count===0 || count>record.maxRows)return null;
  apply(setAdd,nodes,[owner]);
  const stack=[{owner,children,count,index:0,parent:-1}];
  while(stack.length){
    const frame=stack[stack.length-1];
    if(frame.index===frame.count){stack.length--;continue;}
    const child=own(frame.children,frame.index++)?.value;
    if(entries.length>=record.maxRows || !apply(weakHas,objects,[child]) ||
      apply(setHas,nodes,[child]) || data(child,'parent')!==frame.owner)return null;
    apply(setAdd,nodes,[child]);
    if(!method(child,'updateMatrixWorld',objectClass.updateMatrixWorld,nodeKinds,record.classes))return null;
    const descendants=data(child,'children');
    if(!apply(weakHas,arrays,[descendants]))return null;
    const row=entries.length,descendantCount=data(descendants,'length');
    entries[row]={child,parent:frame.parent};
    // Source traversal is pre-order, with each child's local force inherited
    // from its own parent. A DFS stack bounds depth without native recursion.
    if(descendantCount)stack[stack.length]={owner:child,children:descendants,count:descendantCount,index:0,parent:row};
  }
  return entries.length>=2?entries:null;
}
function input(record,owner,children,force){
  if(!apply(weakHas,arrays,[children]) || data(owner,'children')!==children ||
    (force!==undefined && typeof force!=='boolean'))return null;
  const objectClass=record.classes.Object3D,matrixClass=record.classes.Matrix4;
  const entries=hierarchy(record,owner,children);
  if(!entries || !matrixClass)return null;
  const count=entries.length;
  const parent=matrixNumericElements(data(owner,'matrixWorld'),false),local=matrixNumericElements(data(owner,'matrix'),false);
  if(!parent || !local || parent===local)return null;
  const references=new S();
  apply(setAdd,references,[parent]);apply(setAdd,references,[local]);
  for(let i=0;i<fields.length;i++){
    const d=own(owner,fields[i]);if(!d)return null;apply(setAdd,references,[d.value]);
  }
  const state=new F64(count*record.stride),parentState=new F64(16);
  for(let i=0;i<16;i++)parentState[i]=parent[i];
  for(let i=0;i<count;i++){
    const entry=entries[i],child=entry.child;
    const automatic=data(child,'matrixAutoUpdate'),worldAutomatic=data(child,'matrixWorldAutoUpdate'),dirty=own(child,'matrixWorldNeedsUpdate');
    if(typeof automatic!=='boolean' || typeof worldAutomatic!=='boolean' || !dirty || typeof dirty.value!=='boolean')return null;
    const inheritedForce=entry.parent<0?force===true:entries[entry.parent].dirtyWrite;
    const update=automatic || dirty.value || inheritedForce,worldWrite=update && worldAutomatic;
    if(update && !dirty.writable)return null;
    if(automatic && (data(child,'pivot')!==null ||
      !method(child,'updateMatrix',objectClass.updateMatrix,nodeKinds,record.classes)))return null;
    const matrix=data(child,'matrix'),world=data(child,'matrixWorld');
    const elements=matrixNumericElements(matrix,automatic),worldElements=matrixNumericElements(world,worldWrite);
    if(!elements || !worldElements || elements===worldElements ||
      apply(setHas,references,[elements]) || apply(setHas,references,[worldElements]))return null;
    if(automatic && !method(matrix,'compose',matrixClass.compose,matrixKinds,record.classes))return null;
    if(worldWrite && !method(world,'multiplyMatrices',matrixClass.multiplyMatrices,matrixKinds,record.classes))return null;
    apply(setAdd,references,[elements]);apply(setAdd,references,[worldElements]);
    const offset=i*record.stride;
    state[offset]=automatic?1:0;state[offset+1]=worldAutomatic?1:0;state[offset+2]=dirty.value?1:0;
    state[offset+45]=entry.parent;
    if(automatic){
      if(!vector(data(child,'position'),xyz,state,offset+3,references) ||
        !vector(data(child,'quaternion'),xyzw,state,offset+6,references) ||
        !vector(data(child,'scale'),xyz,state,offset+10,references))return null;
    }else{
      for(let j=0;j<fields.length;j++){
        const d=own(child,fields[j]);if(!d || apply(setHas,references,[d.value]))return null;
        apply(setAdd,references,[d.value]);
      }
    }
    for(let j=0;j<16;j++){state[offset+13+j]=elements[j];state[offset+29+j]=worldElements[j];}
    entry.elements=elements;entry.worldElements=worldElements;entry.localWrite=automatic;
    entry.worldWrite=worldWrite;entry.dirtyWrite=update;entry.offset=offset;
  }
  return {state,parent:parentState,entries,count,force:force===true?1:0};
}
function kernel(record){
  if(!record.attempted){
    record.attempted=true;
    try{
      const binary=apply(decode,host,[record.base64]),bytes=new U8(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=apply(charCode,binary,[i]);
      record.kernel=instantiateNumericKernel(bytes);
    }catch{record.failure='TRANSFORM_BATCH_INITIALIZATION';}
    record.base64=null;
  }
  return record.kernel;
}
export function tryTransformBatch(owner,children,force,token){
  if(!apply(weakHas,objects,[owner]) || !apply(weakHas,arrays,[children]) || data(children,'length')===0)return false;
  let stats=apply(mapGet,statistics,[owner]);
  if(!stats){stats={wasmCalls:0,objects:0,fallbackCalls:0,lastFailure:null};apply(mapSet,statistics,[owner,stats]);}
  const fallback=reason=>{stats.fallbackCalls++;stats.lastFailure=reason;return false;};
  const record=apply(mapGet,dispatches,[token]);
  if(!record || !matrixNumericPlatformIntact())return fallback('TRANSFORM_BATCH_PLATFORM');
  let prepared,runtime;
  try{
    prepared=input(record,owner,children,force);
    if(!prepared)return fallback('TRANSFORM_BATCH_GUARD');
    const initialized=record.attempted;
    runtime=kernel(record);if(!runtime)return fallback(record.failure);
    // No source snapshots survive potentially reentrant native initialization.
    if(!matrixNumericPlatformIntact())return fallback('TRANSFORM_BATCH_PLATFORM');
    if(!initialized && !(prepared=input(record,owner,children,force)))return fallback('TRANSFORM_BATCH_GUARD');
    runtime.run(prepared.state,prepared.parent,prepared.count,prepared.force);
  }catch{return fallback(runtime?.diagnostics.lastGuardFailure??'TRANSFORM_BATCH_EXECUTION');}
  // All writes target proven writable own data properties, with no aliases or
  // callbacks in the borrowed interval. Publish only fields the source writes.
  for(let i=0;i<prepared.entries.length;i++){
    const entry=prepared.entries[i],offset=entry.offset;
    if(entry.localWrite)for(let j=0;j<16;j++)entry.elements[j]=prepared.state[offset+13+j];
    if(entry.worldWrite)for(let row=0;row<4;row++)for(let column=0;column<4;column++){
      const j=row+column*4;entry.worldElements[j]=prepared.state[offset+29+j];
    }
    if(entry.dirtyWrite)entry.child.matrixWorldNeedsUpdate=false;
  }
  stats.wasmCalls++;stats.objects+=prepared.count;stats.lastFailure=null;return true;
}
export function transformBatchDiagnostics(object){
  return freeze({...apply(mapGet,statistics,[object])??{wasmCalls:0,objects:0,fallbackCalls:0,lastFailure:null}});
}
