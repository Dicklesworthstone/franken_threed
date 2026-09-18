/** Authored glTF cameras and KHR_lights_punctual, evaluated from the existing
 * animation pose. No scene graph, animation clock, device or frame loop is added.
 * Projection matrices use WebGPU depth 0..1, NOT glTF's OpenGL example matrices.
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#cameras
 * https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual
 */
export class GltfSceneViewError extends Error {
  constructor(code,message){super(`${code}: ${message}`);this.name='GltfSceneViewError';this.code=code;}
}
const fail=(code,message)=>{throw new GltfSceneViewError('GLTF_VIEW_'+code,message);};
const object=(value,label)=>{
  if(!value||typeof value!=='object'||Array.isArray(value))fail('SHAPE',`Expected ${label} object`);
  return value;
};
const list=(value,label)=>{if(!Array.isArray(value))fail('SHAPE',`Expected ${label} array`);return value;};
const index=(array,i,label)=>{
  if(!Number.isSafeInteger(i)||i<0||i>=array.length)fail('INDEX',`Invalid ${label} index`);
  return array[i];
};
const finite=(value,label,min=-Infinity,max=Infinity)=>{
  if(typeof value!=='number'||!Number.isFinite(value)||value<min||value>max)fail('VALUE',`Invalid ${label}`);
  return value;
};
const positive=(value,label)=>{finite(value,label);if(value<=0)fail('VALUE',`${label} must be positive`);return value;};
const name=value=>{if(value!==undefined&&typeof value!=='string')fail('SHAPE','Names must be strings');return value ?? '';};
function fields(value,allowed,label) {
  object(value,label);
  for(const key of Object.keys(value))if(!allowed.includes(key)&&key!=='extras'&&key!=='extensions')fail('UNSUPPORTED',`Unsupported ${label} field: ${key}`);
  if(Object.keys(object(value.extensions ?? {},'extensions')).length)fail('UNSUPPORTED',`Extended ${label} needs the source route`);
}
function projection(camera) {
  fields(camera,['name','type','perspective','orthographic'],'camera');
  const type=camera.type;
  if(type==='perspective') {
    if(camera.orthographic!==undefined)fail('CAMERA','Camera has two projection types');
    const p=camera.perspective;fields(p,['yfov','znear','zfar','aspectRatio'],'perspective');
    const yfov=positive(p.yfov,'vertical field of view'),znear=positive(p.znear,'near plane');
    if(yfov>=Math.PI)fail('CAMERA','Perspective field of view must be less than PI');
    const result={yfov,znear};
    if(p.zfar!==undefined){if(finite(p.zfar,'far plane')<=znear)fail('CAMERA','Far plane must exceed near plane');result.zfar=p.zfar;}
    if(p.aspectRatio!==undefined)result.aspectRatio=positive(p.aspectRatio,'camera aspect ratio');
    return Object.freeze(result);
  }
  if(type==='orthographic') {
    if(camera.perspective!==undefined)fail('CAMERA','Camera has two projection types');
    const p=camera.orthographic;fields(p,['xmag','ymag','znear','zfar'],'orthographic');
    const result={xmag:positive(p.xmag,'orthographic xmag'),ymag:positive(p.ymag,'orthographic ymag'),
      znear:finite(p.znear,'near plane',0),zfar:finite(p.zfar,'far plane')};
    if(result.zfar<=result.znear)fail('CAMERA','Far plane must exceed near plane');
    return Object.freeze(result);
  }
  fail('CAMERA','Unsupported camera projection');
}
function lightProperties(light) {
  fields(light,['name','type','color','intensity','range','spot'],'punctual light');
  if(!['point','directional','spot'].includes(light.type))fail('LIGHT','Unsupported light type');
  const color=list(light.color ?? [1,1,1],'light color');
  if(color.length!==3)fail('LIGHT','Light color needs three components');
  const result={type:light.type,color:Object.freeze(color.map(v=>finite(v,'light color',0,1))),intensity:finite(light.intensity ?? 1,'intensity',0)};
  if(light.range!==undefined) {
    if(light.type==='directional')fail('LIGHT','Directional lights cannot have range');
    result.range=positive(light.range,'light range');
  }
  if(light.type==='spot') {
    fields(light.spot,['innerConeAngle','outerConeAngle'],'spot cone');
    result.innerConeAngle=finite(light.spot.innerConeAngle ?? 0,'inner cone',0);
    result.outerConeAngle=finite(light.spot.outerConeAngle ?? Math.PI/4,'outer cone',0,Math.PI/2);
    if(result.innerConeAngle>=result.outerConeAngle)fail('LIGHT','Spot cones must satisfy inner < outer');
  }else if(light.spot!==undefined)fail('LIGHT','Only spot lights can have a cone');
  return result;
}

/** Snapshot selected-scene camera/light instances with original node/source IDs.
 * Repeated references to one camera/light definition stay distinct instances.
 * Uninstantiated definitions and other scenes are not imported. The selected
 * scene is limited to the existing renderer's eight lights, never truncated.
 */
export function decodeGltfSceneView(model,{scene=model?.scene ?? 0}={}) {
  if(model?.asset?.version!=='2.0')fail('VERSION','Expected glTF 2.0');
  const nodes=list(model.nodes ?? [],'nodes'),scenes=list(model.scenes,'scenes');
  if(nodes.length>65536)fail('LIMIT','Too many scene nodes');
  const selected=object(index(scenes,scene,'scene'),'scene');
  const parents=new Int32Array(nodes.length).fill(-1),children=[];
  for(let n=0;n<nodes.length;n++) {
    const node=object(nodes[n],'node');children[n]=list(node.children ?? [],'children');
    for(const child of children[n]) {
      index(nodes,child,'child');
      if(child===n||parents[child]!==-1)fail('HIERARCHY','Self-parent, repeated child or multiple parents');
      parents[child]=n;
    }
  }
  const forest=[];for(let n=0;n<nodes.length;n++)if(parents[n]===-1)forest.push(n);
  for(let i=0;i<forest.length;i++)for(const child of children[forest[i]])forest.push(child);
  if(forest.length!==nodes.length)fail('HIERARCHY','Cyclic scene hierarchy');
  const order=[],seen=new Set();
  for(const root of list(selected.nodes ?? [],'scene roots')) {
    index(nodes,root,'scene root');
    if(parents[root]!==-1||seen.has(root))fail('HIERARCHY','Scene roots must be distinct forest roots');
    seen.add(root);order.push(root);
  }
  for(let i=0;i<order.length;i++)for(const child of children[order[i]])order.push(child);
  const cameraDefs=list(model.cameras ?? [],'cameras'),rootExtension=model.extensions?.KHR_lights_punctual;
  if(rootExtension!==undefined)fields(rootExtension,['lights'],'punctual lights extension');
  const lightDefs=list(rootExtension?.lights ?? [],'lights'),cameras=[],lights=[];
  for(const nodeIndex of order) {
    const node=nodes[nodeIndex],ref=node.extensions?.KHR_lights_punctual;
    if(node.camera!==undefined) {
      if(cameras.length>=4096)fail('LIMIT','Too many camera instances');
      const camera=object(index(cameraDefs,node.camera,'camera'),'camera');
      cameras.push(Object.freeze({node:nodeIndex,camera:node.camera,name:name(camera.name),nodeName:name(node.name),type:camera.type,projection:projection(camera)}));
    }
    if(ref!==undefined) {
      fields(ref,['light'],'node light');
      if(lights.length>=8)fail('LIMIT','Selected scene exceeds the renderer limit of eight punctual lights');
      const light=object(index(lightDefs,ref.light,'light'),'light');
      lights.push(Object.freeze({node:nodeIndex,light:ref.light,name:name(light.name),nodeName:name(node.name),...lightProperties(light)}));
    }
  }
  return Object.freeze({format:'f3d-gltf-scene-view-v1',scene,nodeCount:nodes.length,cameras:Object.freeze(cameras),lights:Object.freeze(lights)});
}
function world(pose,node) {
  const data=pose.worldMatrices;
  if(!(data instanceof Float64Array)&&!(data instanceof Float32Array))fail('POSE','Expected packed world matrices');
  if(!(data.buffer instanceof ArrayBuffer)||data.buffer.resizable||data.length!==pose.nodeCount*16)fail('POSE','Invalid world matrix storage');
  let m;try{m=Array.from(data.subarray(node*16,node*16+16));}catch{fail('POSE','Detached world matrix storage');}
  for(const x of m)finite(x,'world matrix');
  if(m[3]!==0||m[7]!==0||m[11]!==0||m[15]!==1)fail('TRANSFORM','World transform must be affine');
  return m;
}
const unit=v=>{const n=Math.hypot(...v);if(!Number.isFinite(n)||n===0)fail('TRANSFORM','Undefined camera/light direction');return v.map(x=>x/n);};
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function cameraMatrices(camera,m,aspectRatio) {
  // Preserve the world forward axis and orthogonalize up. This removes scale
  // and handles shear inherited from rotated, nonuniformly scaled ancestors
  // without mirroring the image. Parallel/zero forward and up are ambiguous.
  const z=unit(m.slice(8,11)),up=unit(m.slice(4,7)),right=cross(up,z);
  if(Math.hypot(...right)<1e-10)fail('TRANSFORM','Camera forward and up axes are parallel');
  const x=unit(right),y=unit(cross(z,x)),position=m.slice(12,15);
  const view=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,position),-dot(y,position),-dot(z,position),1];
  const projection=new Array(16).fill(0),p=camera.projection,n=p.znear,f=p.zfar;
  if(camera.type==='perspective') {
    if(aspectRatio!==undefined)positive(aspectRatio,'viewport aspect ratio');
    const aspect=p.aspectRatio ?? aspectRatio;
    if(aspect===undefined)fail('ASPECT','Perspective camera without aspectRatio needs a viewport aspect ratio');
    projection[0]=1/Math.tan(p.yfov/2)/aspect;projection[5]=1/Math.tan(p.yfov/2);
    projection[10]=f===undefined?-1:-1/(1-n/f);
    projection[11]=-1;projection[14]=n*projection[10];
  }else {
    projection[0]=1/p.xmag;projection[5]=1/p.ymag;projection[10]=-1/(f-n);projection[14]=-n/(f-n);projection[15]=1;
  }
  const viewProjection=new Array(16).fill(0);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)viewProjection[c*4+r]+=projection[k*4+r]*view[c*4+k];
  for(const matrix of [view,projection,viewProjection]) {
    for(const v of matrix)if(!Number.isFinite(Math.fround(v)))fail('VALUE','Camera matrix cannot be represented by the renderer');
    Object.freeze(matrix);
  }
  return {viewMatrix:view,projectionMatrix:projection,viewProjection,cameraPosition:Object.freeze(position),viewDirection:Object.freeze(z)};
}

/** Bind immutable decoded metadata to a borrowed animation pose. sample() and
 * sampleLights() only read the current pose; they never advance it. Each result
 * is an independent snapshot so retaining a prior submitted frame is safe.
 * cameraNode selects the original NODE index (not the shared camera definition).
 * With multiple cameras an explicit selection is required; no arbitrary choice.
 */
export function createGltfSceneView(pose,definition) {
  if(definition?.format!=='f3d-gltf-scene-view-v1'||!Number.isSafeInteger(definition.nodeCount)||definition.nodeCount<0||definition.nodeCount>65536||
     !pose||pose.nodeCount!==definition.nodeCount)fail('POSE','Scene view needs its matching animation pose');
  // Revalidate and snapshot public decoded data too: never trust writable foreign
  // descriptors passed to this lower-level entry, or expose borrowed metadata.
  const nodeCount=definition.nodeCount;
  const cameras=list(definition.cameras,'camera instances'),lights=list(definition.lights,'light instances');
  if(cameras.length>4096||lights.length>8)fail('LIMIT','Excessive scene view instances');
  const seen=new Set();
  const node=n=>{if(!Number.isSafeInteger(n)||n<0||n>=nodeCount)fail('INDEX','Invalid instance node');return n;};
  const ids=(entry,key)=>{if(!Number.isSafeInteger(entry[key])||entry[key]<0)fail('INDEX',`Invalid ${key} ID`);return {[key]:entry[key]};};
  const preparedCameras=cameras.map(c=>{
    node(c.node);if(seen.has(c.node))fail('CAMERA','Duplicate camera node');seen.add(c.node);
    return Object.freeze({node:c.node,...ids(c,'camera'),name:name(c.name),nodeName:name(c.nodeName),type:c.type,projection:projection({type:c.type,[c.type]:c.projection})});
  });
  seen.clear();
  const preparedLights=lights.map(l=>{
    node(l.node);if(seen.has(l.node))fail('LIGHT','Duplicate light node');seen.add(l.node);
    const source={type:l.type,color:l.color,intensity:l.intensity,...(l.range===undefined?{}:{range:l.range}),
      ...(l.type==='spot'?{spot:{innerConeAngle:l.innerConeAngle,outerConeAngle:l.outerConeAngle}}:{})};
    return Object.freeze({node:l.node,...ids(l,'light'),name:name(l.name),nodeName:name(l.nodeName),...lightProperties(source)});
  });
  let busy=false;
  function live() {
    if(pose.disposed)fail('POSE','Animation pose has been disposed');
    if(pose.nodeCount!==nodeCount||!Number.isSafeInteger(pose.version)||pose.version<0)fail('POSE','Invalid animation pose');
  }
  function snapshot(operation) {
    live();if(busy)fail('REENTRANT','Scene view sampling cannot be reentered');busy=true;
    const version=pose.version;
    try {const result=operation(version);live();if(pose.version!==version)fail('CHANGED','Pose changed during scene view sampling');return result;}
    finally {busy=false;}
  }
  function evaluateLights() {
    return Object.freeze(preparedLights.map(l=>{
      const m=world(pose,l.node),out={type:l.type,color:l.color,intensity:l.intensity};
      if(l.type!=='directional')out.position=Object.freeze(m.slice(12,15));
      if(l.type!=='point')out.direction=Object.freeze(unit([-m[8],-m[9],-m[10]]));
      if(l.range!==undefined)out.range=l.range;
      if(l.type==='spot'){out.innerConeAngle=l.innerConeAngle;out.outerConeAngle=l.outerConeAngle;}
      for(const v of [...out.color,...(out.position ?? []),out.intensity,out.range ?? 1])if(!Number.isFinite(Math.fround(v)))fail('VALUE','Light cannot be represented by the renderer');
      return Object.freeze(out);
    }));
  }
  return Object.freeze({cameras:Object.freeze(preparedCameras),lights:Object.freeze(preparedLights),
    sampleLights(){return snapshot(()=>evaluateLights());},
    sample(options={}) {return snapshot(poseVersion=>{
      const {cameraNode,aspectRatio}=object(options,'camera options');
      for(const key of Object.keys(options))if(!['cameraNode','aspectRatio'].includes(key))fail('CAMERA',`Unsupported camera option: ${key}`);
      const camera=cameraNode===undefined?(preparedCameras.length===1?preparedCameras[0]:null):preparedCameras.find(c=>c.node===cameraNode);
      if(!camera)fail('CAMERA','Select an instantiated cameraNode in this scene');
      const matrices=cameraMatrices(camera,world(pose,camera.node),aspectRatio),lights=evaluateLights();
      const lighting=Object.freeze({lights,...(camera.type==='orthographic'?{viewDirection:matrices.viewDirection}:{cameraPosition:matrices.cameraPosition})});
      return Object.freeze({...matrices,lighting,poseVersion,cameraNode:camera.node,cameraIndex:camera.camera,type:camera.type});
    });},
  });
}
