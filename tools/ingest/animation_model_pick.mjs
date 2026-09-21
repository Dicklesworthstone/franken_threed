/** Opt-in selection for the existing model factories. CPU models borrow their
 * current deformation outputs. GPU models snapshot position-only source geometry
 * before initialization yields, then materialize/update CPU positions only for
 * selected draws. The CPU reference path is intentional: no synchronous GPU
 * readback or claim that GPU arithmetic/raster coverage is reproduced exactly.
 */
import {createAnimationDeformer} from './animation_deformer.mjs';
import {createAnimationRaycaster,rayFromAnimationCamera,AnimationRaycastError} from './animation_raycast.mjs';
export {AnimationRaycastError} from './animation_raycast.mjs';
const fail=(code,message)=>{throw new AnimationRaycastError('ANIMATION_PICK_'+code,message);};

export function createAnimationModelPicker(pose,view,drawables,source,picking=false,deformers=null) {
  let disposed=false,busy=false,inputs=null,caster=null,owned=[],lastQuery=null;
  let materializedMeshes=0,updatedMeshes=0;
  const enabled=picking!==false;
  if(!enabled)return Object.freeze({enabled:false,get lastQuery(){return null;},
    raycast(){fail('DISABLED','Enable picking when creating this model');},
    pick(){fail('DISABLED','Enable picking when creating this model');},dispose(){disposed=true;},
  });
  const options=picking===true?{}:picking;
  if(!options||typeof options!=='object'||Array.isArray(options))fail('OPTION','picking must be false, true, or a limits object');
  for(const key of Object.keys(options))if(!['maxComponents','maxTriangles','maxBytes','sceneBvh'].includes(key))fail('OPTION',`Unsupported picking option: ${key}`);
  const maxComponents=options.maxComponents ?? 16777216;
  const sceneBvh=options.sceneBvh;
  const limits={maxTriangles:options.maxTriangles ?? 1048576,maxBytes:options.maxBytes ?? 128*1024*1024,
    sceneBvh:sceneBvh===undefined?true:sceneBvh};
  if(typeof limits.sceneBvh!=='boolean')fail('OPTION','sceneBvh must be boolean');
  if(!Number.isSafeInteger(maxComponents)||maxComponents<1||!Number.isSafeInteger(limits.maxTriangles)||limits.maxTriangles<1||limits.maxTriangles>1048576||!Number.isSafeInteger(limits.maxBytes)||limits.maxBytes<1)fail('LIMIT','Invalid picking limits');
  if(!pose||!view||typeof view.sample!=='function'||!Array.isArray(drawables)||drawables.length>4096||!Array.isArray(source)||source.length!==drawables.length)fail('SHAPE','Expected a decoded model and its bound scene view');
  if(deformers!==null&&(!Array.isArray(deformers)||deformers.length!==drawables.length))fail('SHAPE','Expected matching CPU deformers');
  const initialVersion=pose.version,drawCount=drawables.length;
  let consumed=0,triangles=0;
  function copy(value,length,label) {
    if((!Array.isArray(value)&&!ArrayBuffer.isView(value))||value instanceof DataView||!Number.isSafeInteger(length)||length<0||value.length!==length)fail('SHAPE',`Invalid ${label}`);
    if(ArrayBuffer.isView(value)){
      if(!(value.buffer instanceof ArrayBuffer)||value.buffer.resizable)fail('STORAGE',`${label} needs fixed unshared storage`);
      try{new Uint8Array(value.buffer,0,0);}catch{fail('STORAGE',`${label} is detached`);}
    }
    consumed+=length;if(consumed>maxComponents)fail('LIMIT','Aggregate picking source component budget exceeded');
    return Float64Array.from(value,x=>{if(typeof x!=='number'||!Number.isFinite(x))fail('VALUE',`${label} must be finite`);return x;});
  }
  const metadata=(drawable,i)=>({indices:drawable.indices ?? null,texCoords:drawable.texCoords ?? null,
    doubleSided:drawable.doubleSided ?? false,source:{...source[i]}});
  if(deformers!==null) {
    const descriptors=drawables.map((drawable,i)=>{
      if(deformers[i]?.node!==drawable?.geometry?.node||deformers[i]?.vertexCount!==drawable?.geometry?.positions?.length/3)fail('SHAPE','CPU deformer does not match its drawable');
      return {...metadata(drawable,i),deformer:deformers[i]};
    });
    caster=createAnimationRaycaster(pose,descriptors,limits);
  }else {
    inputs=drawables.map((drawable,i)=>{
      const g=drawable?.geometry,length=g?.positions?.length;
      if(!Number.isSafeInteger(length)||length<3||length%3)fail('SHAPE','Expected XYZ source geometry');
      const count=drawable.indices==null?length/3:drawable.indices.length;
      if(!Number.isSafeInteger(count)||count<0||count%3||(triangles+=count/3)>limits.maxTriangles)fail('LIMIT','Invalid or excessive picking topology');
      const geometry={node:g.node,positions:copy(g.positions,length,'source positions')};
      const targets=g.morphTargets ?? [];
      if(!Array.isArray(targets)||targets.length>4096)fail('SHAPE','Invalid morph targets');
      geometry.morphTargets=targets.map(target=>{
        if(!target||typeof target!=='object'||Array.isArray(target))fail('SHAPE','Invalid morph target');
        return target.positions===undefined?{}:{positions:copy(target.positions,length,'morph positions')};
      });
      for(const key of ['joints','weights'])if(g[key]!==undefined)geometry[key]=copy(g[key],g[key].length,key);
      if(g.influences!==undefined)geometry.influences=g.influences;
      const meta=metadata(drawable,i);
      if(meta.indices!==null)meta.indices=copy(meta.indices,count,'indices');
      if(meta.texCoords!==null)meta.texCoords=copy(meta.texCoords,length/3*2,'texture coordinates');
      return {geometry,meta};
    });
  }
  if(pose.disposed||pose.version!==initialVersion){caster?.dispose();fail('CHANGED','Pose changed while preparing picking');}
  function live() {
    if(disposed)fail('DISPOSED','Model picker has been disposed');
    if(pose.disposed||!Number.isSafeInteger(pose.version)||pose.version<0)fail('POSE','Model pose is unavailable');
  }
  function prepare(slot) {
    if(!slot.deformer) {
      slot.deformer=createAnimationDeformer(pose,slot.geometry,{maxComponents});
      // The deformer owns its source snapshot now; do not retain a second copy.
      slot.geometry=null;materializedMeshes++;updatedMeshes++;
    }else if(slot.deformer.poseVersion!==pose.version) {
      slot.deformer.update();updatedMeshes++;
    }
    return slot.deformer;
  }
  function initialize() {
    if(caster)return;
    const slots=inputs.map(input=>({geometry:input.geometry,deformer:null}));
    // Admission reads only node/vertexCount. The raycaster first validates the
    // ray and selection, then accesses only selected deformer properties. Thus
    // malformed queries and empty selections never materialize CPU geometry.
    caster=createAnimationRaycaster(pose,inputs.map((input,i)=>({...input.meta,deformer:{
      node:input.geometry.node,vertexCount:input.geometry.positions.length/3,
      get disposed(){return disposed||!!slots[i].deformer?.disposed;},
      get poseVersion(){return prepare(slots[i]).poseVersion;},
      get version(){return prepare(slots[i]).version;},
      get positions(){return prepare(slots[i]).positions;},
      get worldMatrix(){return prepare(slots[i]).worldMatrix;},
    }})),limits);
    owned=slots;inputs=null;
  }
  function captureQuery(input={}) {
    if(!input||typeof input!=='object'||Array.isArray(input))fail('OPTION','Expected picking query options');
    const version=pose.version,copy={...input};
    if(Object.hasOwn(copy,'lodSelection')) {
      if(Object.hasOwn(copy,'drawIndices'))fail('OPTION','Use lodSelection or drawIndices, not both');
      const selection=copy.lodSelection;
      if(!selection||typeof selection!=='object'||Array.isArray(selection))fail('OPTION','Expected a submitted LOD selection');
      const selectedVersion=selection.poseVersion,indices=selection.drawIndices;
      if(!Number.isSafeInteger(selectedVersion)||selectedVersion<0)fail('OPTION','Invalid LOD pose version');
      if(selectedVersion!==version)fail('STALE','Render the current pose before using its LOD selection');
      if(!Array.isArray(indices)||indices.length>drawCount)fail('OPTION','Invalid LOD draw selection');
      copy.drawIndices=new Array(indices.length);
      for(let i=0;i<copy.drawIndices.length;i++)copy.drawIndices[i]=indices[i];
      delete copy.lodSelection;
    }
    live();if(pose.version!==version)fail('CHANGED','Pose changed during picking option capture');
    return copy;
  }
  function query(ray,options) {
    const captured=captureQuery(options);
    initialize();return caster.raycast(ray,captured);
  }
  function exclusive(operation) {
    live();if(busy)fail('REENTRANT','Model picking cannot be reentered');busy=true;
    const version=pose.version;updatedMeshes=0;
    try {
      const result=operation();live();
      if(pose.version!==version)fail('CHANGED','Pose changed during model picking');
      lastQuery=Object.freeze({...caster.lastQuery,materializedMeshes,updatedMeshes});return result;
    }finally{busy=false;}
  }
  return Object.freeze({enabled:true,get lastQuery(){return lastQuery;},
    raycast(ray,options){return exclusive(()=>query(ray,options));},
    pick(ndc,cameraOptions,queryOptions){return exclusive(()=>{
      const sample=view.sample(cameraOptions),ray=rayFromAnimationCamera(sample,ndc);
      if(sample.poseVersion!==pose.version)fail('CHANGED','Camera and geometry must use one pose');
      return query(ray,queryOptions);
    });},
    dispose(){
      if(busy)fail('REENTRANT','Cannot dispose during model picking');
      if(disposed)return;caster?.dispose();for(const slot of owned)slot.deformer?.dispose();
      owned=[];inputs=null;caster=null;lastQuery=null;disposed=true;
    },
  });
}
