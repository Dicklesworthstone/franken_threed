/**
 * Packed glTF animation poses. No parser, filesystem, DOM, renderer or Wasm
 * initialization is needed at import time. Instances own all mutable storage.
 *
 * This is an explicit pose sampler, not an AnimationMixer replacement:
 * sample() resets untargeted values to the imported rest pose, and loop timing
 * is an explicit caller choice. Outputs retain identity across samples. Input
 * definitions are copied once; edits to published output arrays are not inputs.
 * edit() publishes transactional local-pose changes; sample/blend/reset replace
 * those changes on the next call. snapshotLocalPose() returns independent data.
 *
 * Conventions: glTF 2.0 section 3.11 / Appendix C; column-major T*R*S;
 * mesh-local palette = inverse(meshWorld) * jointWorld * inverseBindMatrix.
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
 */
export class AnimationPoseError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationPoseError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationPoseError(code, message); };
// Capture publication operations before caller-owned output views are exposed.
const apply = Reflect.apply;
const typedPrototype = Object.getPrototypeOf(Float64Array.prototype);
const typedSet = typedPrototype.set;
const typedLength = Object.getOwnPropertyDescriptor(typedPrototype, 'length').get;
const typedBuffer = Object.getOwnPropertyDescriptor(typedPrototype, 'buffer').get;
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const finite = (value, label) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail('ANIMATION_VALUE', `${label} must be finite`);
  return value;
};
const integer = (value, size, label) => {
  if (!Number.isInteger(value) || value < 0 || value >= size) fail('ANIMATION_INDEX', `${label} is outside its domain`);
  return value;
};
function numbers(value, count, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== count) {
    fail('ANIMATION_SHAPE', `${label} requires ${count} numbers`);
  }
  return Float64Array.from(value, item => finite(item, label));
}
function affine(value, label) {
  const matrix = numbers(value, 16, label);
  if (matrix[3] !== 0 || matrix[7] !== 0 || matrix[11] !== 0 || matrix[15] !== 1) {
    fail('ANIMATION_MATRIX', `${label} must be affine`);
  }
  return matrix;
}
function normalize(q, offset) {
  const length = Math.hypot(q[offset], q[offset+1], q[offset+2], q[offset+3]);
  if (!(length > 0) || !Number.isFinite(length)) fail('ANIMATION_QUATERNION', 'Interpolated rotation is not a finite nonzero quaternion');
  for (let i=0;i<4;i++) q[offset+i] /= length;
}
function unit(q, offset, label, tolerance=1e-3) {
  const norm = Math.hypot(q[offset],q[offset+1],q[offset+2],q[offset+3]);
  if (Math.abs(norm-1)>tolerance) fail('ANIMATION_QUATERNION', `${label} must be normalized`);
}
function compose(t, q, s, node, out) {
  const a=node*3,b=node*4,o=node*16;
  const x=q[b],y=q[b+1],z=q[b+2],w=q[b+3],x2=x+x,y2=y+y,z2=z+z;
  const xx=x*x2,xy=x*y2,xz=x*z2,yy=y*y2,yz=y*z2,zz=z*z2,wx=w*x2,wy=w*y2,wz=w*z2;
  out[o]=(1-(yy+zz))*s[a]; out[o+1]=(xy+wz)*s[a]; out[o+2]=(xz-wy)*s[a]; out[o+3]=0;
  out[o+4]=(xy-wz)*s[a+1]; out[o+5]=(1-(xx+zz))*s[a+1]; out[o+6]=(yz+wx)*s[a+1]; out[o+7]=0;
  out[o+8]=(xz+wy)*s[a+2]; out[o+9]=(yz-wx)*s[a+2]; out[o+10]=(1-(xx+yy))*s[a+2]; out[o+11]=0;
  out[o+12]=t[a]; out[o+13]=t[a+1]; out[o+14]=t[a+2]; out[o+15]=1;
}
// Output must not alias either input range. Multiplication order is explicit;
// in particular, pose propagation is never a parallel reduction over ancestors.
function multiply(a, ao, b, bo, out, offset) {
  for (let column=0;column<4;column++) for (let row=0;row<4;row++) {
    out[offset+column*4+row] = a[ao+row]*b[bo+column*4] + a[ao+4+row]*b[bo+column*4+1]
      + a[ao+8+row]*b[bo+column*4+2] + a[ao+12+row]*b[bo+column*4+3];
  }
}
function inverseAffine(m, o, out) {
  const a=m[o],b=m[o+4],c=m[o+8],d=m[o+1],e=m[o+5],f=m[o+9],g=m[o+2],h=m[o+6],i=m[o+10];
  const A=e*i-f*h,B=c*h-b*i,C=b*f-c*e,D=f*g-d*i,E=a*i-c*g,F=c*d-a*f,G=d*h-e*g,H=b*g-a*h,I=a*e-b*d;
  const determinant=a*A+b*D+c*G;
  if (determinant===0 || !Number.isFinite(determinant)) fail('ANIMATION_SINGULAR_MESH', 'Skinned mesh world transform is not invertible');
  out[0]=A/determinant; out[4]=B/determinant; out[8]=C/determinant;
  out[1]=D/determinant; out[5]=E/determinant; out[9]=F/determinant;
  out[2]=G/determinant; out[6]=H/determinant; out[10]=I/determinant;
  out[3]=out[7]=out[11]=0; out[15]=1;
  for(let row=0;row<3;row++) out[12+row]=-(out[row]*m[o+12]+out[4+row]*m[o+13]+out[8+row]*m[o+14]);
}
function interval(channel, time) {
  const times=channel.times,n=times.length;
  if(time<=times[0]) return 0;
  if(time>=times[n-1]) return n-1;
  let lo=channel.cursor;
  // Coherent playback examines at most two neighboring intervals, then uses
  // binary search. Large seeks/backwards playback never linearly scan all keys.
  if(lo<n-1 && times[lo]<=time && time<times[lo+1]) return lo;
  if(lo+2<n && times[lo+1]<=time && time<times[lo+2]) return ++channel.cursor;
  let hi=n-1;lo=0;
  while(lo+1<hi) { const mid=lo+Math.floor((hi-lo)/2);if(times[mid]<=time)lo=mid;else hi=mid; }
  channel.cursor=lo;return lo;
}
function sampleChannel(channel, time, destination, offset) {
  const {times,values,width,interpolation,path}=channel;
  const key=interval(channel,time),cubic=interpolation==='CUBICSPLINE',stride=width*(cubic?3:1);
  const left=key*stride+(cubic?width:0);
  if(interpolation==='STEP'||key===times.length-1||time<=times[0]||time===times[key]) {
    for(let i=0;i<width;i++) destination[offset+i]=values[left+i];
    if(cubic&&path==='rotation')normalize(destination,offset);
    return;
  }
  const duration=times[key+1]-times[key],t=(time-times[key])/duration,right=left+stride;
  if(cubic) {
    const t2=t*t,t3=t2*t,h00=2*t3-3*t2+1,h10=t3-2*t2+t,h01=-2*t3+3*t2,h11=t3-t2;
    for(let i=0;i<width;i++) destination[offset+i]=h00*values[left+i]+h10*duration*values[left+width+i]
      +h01*values[right+i]+h11*duration*values[right-width+i];
    // glTF cubic quaternion tangents are Hermite derivatives, not quaternion
    // endpoints. Do not flip antipodal signs as done for linear SLERP.
    if(path==='rotation')normalize(destination,offset);
  } else if(path==='rotation') {
    let dot=0;for(let i=0;i<4;i++)dot+=values[left+i]*values[right+i];
    const sign=dot<0?-1:1;dot=Math.min(1,Math.abs(dot));
    let a=1-t,b=t;
    if(1-dot>Number.EPSILON) { const angle=Math.acos(dot),sin=Math.sin(angle);a=Math.sin((1-t)*angle)/sin;b=Math.sin(t*angle)/sin; }
    for(let i=0;i<4;i++)destination[offset+i]=a*values[left+i]+b*sign*values[right+i];
    normalize(destination,offset);
  } else {
    for(let i=0;i<width;i++)destination[offset+i]=(1-t)*values[left+i]+t*values[right+i];
  }
}

// Alias-safe quaternion operations for local-pose mixing. Normalize inputs to
// avoid magnifying the small unit-length error tolerated by glTF accessors.
function mixQuaternion(out,offset,source,start,t) {
  const an=Math.hypot(out[offset],out[offset+1],out[offset+2],out[offset+3]);
  const bn=Math.hypot(source[start],source[start+1],source[start+2],source[start+3]);
  if(!(an>0)||!(bn>0)||!Number.isFinite(an)||!Number.isFinite(bn))fail('ANIMATION_QUATERNION','Cannot mix invalid rotations');
  let dot=0;for(let i=0;i<4;i++)dot+=(out[offset+i]/an)*(source[start+i]/bn);
  const sign=dot<0?-1:1;dot=Math.min(1,Math.abs(dot));
  let a=1-t,b=t;
  if(1-dot>Number.EPSILON) {
    const angle=Math.acos(dot),sin=Math.sin(angle);
    a=Math.sin((1-t)*angle)/sin;b=Math.sin(t*angle)/sin;
  }
  for(let i=0;i<4;i++)out[offset+i]=a*(out[offset+i]/an)+b*sign*(source[start+i]/bn);
  normalize(out,offset);
}
function multiplyQuaternion(out,o,a,ao,b,bo) {
  const x=a[ao],y=a[ao+1],z=a[ao+2],w=a[ao+3];
  const X=b[bo],Y=b[bo+1],Z=b[bo+2],W=b[bo+3];
  out[o]=x*W+w*X+y*Z-z*Y;out[o+1]=y*W+w*Y+z*X-x*Z;
  out[o+2]=z*W+w*Z+x*Y-y*X;out[o+3]=w*W-x*X-y*Y-z*Z;
}

/**
 * Create a reusable pose instance from a build-time decoded definition.
 * All indices use original glTF node/skin order. `instances` lists each skinned
 * mesh node, including several mesh nodes referencing the same skin. Palettes
 * are mesh-local, so those instances must NOT accidentally share a palette.
 * No implicit wall-clock, event scheduling or public Three.js mutation.
 */
export function createAnimationPlayer(definition) {
  if(definition?.format!=='f3d-animation-v1'||!Array.isArray(definition.nodes)||definition.nodes.length>65536)fail('ANIMATION_FORMAT','Expected a bounded f3d-animation-v1 definition');
  const nodes=definition.nodes,n=nodes.length,parents=new Int32Array(n),children=Array.from({length:n},()=>[]);
  const baseT=new Float64Array(n*3),baseQ=new Float64Array(n*4),baseS=new Float64Array(n*3),matrices=new Map();
  const morphOffsets=new Uint32Array(n+1),baseWeights=[];
  for(let j=0;j<n;j++) {
    const node=nodes[j];if(!node||typeof node!=='object')fail('ANIMATION_NODE','Invalid node');
    const parent=node.parent??-1;parents[j]=parent;
    if(parent!==-1){integer(parent,n,'Parent');if(parent===j)fail('ANIMATION_HIERARCHY','A node cannot parent itself');children[parent].push(j);}
    baseT.set(numbers(node.translation??[0,0,0],3,'Translation'),j*3);
    baseQ.set(numbers(node.rotation??[0,0,0,1],4,'Rotation'),j*4);unit(baseQ,j*4,'Rest rotation');
    baseS.set(numbers(node.scale??[1,1,1],3,'Scale'),j*3);
    if(node.matrix!==undefined){if(node.translation!==undefined||node.rotation!==undefined||node.scale!==undefined)fail('ANIMATION_MATRIX','Matrix and TRS cannot both be present');matrices.set(j,affine(node.matrix,'Node matrix'));}
    const weights=node.weights??[];if(!Array.isArray(weights)&&!ArrayBuffer.isView(weights))fail('ANIMATION_SHAPE','Morph weights must be an array');
    if(weights.length>4096||baseWeights.length+weights.length>1048576)fail('ANIMATION_LIMIT','Too many morph weights');
    for(const w of weights)baseWeights.push(finite(w,'Morph weight'));
    morphOffsets[j+1]=baseWeights.length;
  }
  const rawSkins=definition.skins??[],rawClips=definition.clips??[],rawInstances=definition.instances??[];
  if(!Array.isArray(rawSkins)||!Array.isArray(rawClips)||!Array.isArray(rawInstances)||rawSkins.length>65536||rawClips.length>4096||rawInstances.length>65536)fail('ANIMATION_LIMIT','Invalid or excessive skins/clips/instances');
  const order=[];for(let j=0;j<n;j++)if(parents[j]===-1)order.push(j);
  for(let i=0;i<order.length;i++)for(const child of children[order[i]])order.push(child);
  if(order.length!==n)fail('ANIMATION_HIERARCHY','Cyclic node hierarchy');
  let inverseBindComponents=0;
  const skins=rawSkins.map(skin=>{
    if(!skin||!Array.isArray(skin.joints)||!skin.joints.length||skin.joints.length>65536)fail('ANIMATION_SKIN','Skin requires a bounded joint list');
    inverseBindComponents+=skin.joints.length*16;
    if(inverseBindComponents>16777216)fail('ANIMATION_LIMIT','Inverse bind storage exceeds component budget');
    const joints=Uint32Array.from(skin.joints,index=>integer(index,n,'Joint'));
    if(new Set(joints).size!==joints.length)fail('ANIMATION_SKIN','Duplicate joint');
    const inverseBind=skin.inverseBindMatrices===undefined?new Float64Array(joints.length*16):numbers(skin.inverseBindMatrices,joints.length*16,'Inverse bind matrices');
    for(let i=0;i<joints.length;i++) {
      if(skin.inverseBindMatrices===undefined)inverseBind.set(identity(),i*16);
      else affine(inverseBind.subarray(i*16,i*16+16),'Inverse bind matrix');
    }
    return {joints,inverseBind};
  });
  let paletteSize=0;const instanceNodes=new Set();
  const instances=rawInstances.map(instance=>{
    const node=integer(instance?.node,n,'Mesh node'),skin=integer(instance?.skin,skins.length,'Skin');
    if(instanceNodes.has(node))fail('ANIMATION_SKIN','Duplicate skinned mesh instance');instanceNodes.add(node);
    const result={node,skin,offset:paletteSize,jointCount:skins[skin].joints.length};paletteSize+=result.jointCount*16;
    if(paletteSize>16777216)fail('ANIMATION_LIMIT','Joint palette exceeds component budget');
    return Object.freeze(result);
  });
  let components=0;
  const clips=rawClips.map((clip,clipIndex)=>{
    if(!clip||!Array.isArray(clip.channels)||clip.channels.length>262144)fail('ANIMATION_CHANNEL','Invalid channel list');
    const used=new Set();let duration=0;
    const channels=clip.channels.map(channel=>{
      const node=integer(channel?.node,n,'Animation target'),path=channel.path;
      if(!['translation','rotation','scale','weights'].includes(path))fail('ANIMATION_CHANNEL','Unsupported animation target');
      if(path!=='weights'&&matrices.has(node))fail('ANIMATION_CHANNEL','Cannot animate TRS on a matrix node');
      const width=path==='weights'?morphOffsets[node+1]-morphOffsets[node]:path==='rotation'?4:3;
      if(!width||used.has(`${node}:${path}`))fail('ANIMATION_CHANNEL','Empty morph target or duplicate animation target');used.add(`${node}:${path}`);
      if(channel.quantizedRotation!==undefined&&typeof channel.quantizedRotation!=='boolean')fail('ANIMATION_CHANNEL','Invalid rotation quantization marker');
      const interpolation=channel.interpolation??'LINEAR';
      if(!['LINEAR','STEP','CUBICSPLINE'].includes(interpolation))fail('ANIMATION_INTERPOLATION','Unknown interpolation');
      const count=channel.times?.length;
      if(!Number.isInteger(count)||count<(interpolation==='CUBICSPLINE'?2:1)||count>1048576)fail('ANIMATION_KEYS','Invalid keyframe count');
      const expected=count*width*(interpolation==='CUBICSPLINE'?3:1);
      components+=expected+count;if(components>16777216)fail('ANIMATION_LIMIT','Keyframe component budget exceeded');
      const times=numbers(channel.times,count,'Keyframe times'),values=numbers(channel.values,expected,'Keyframe values');
      for(let i=0;i<count;i++)if(times[i]<0||(i&&times[i]<=times[i-1]))fail('ANIMATION_KEYS','Times must be nonnegative and strictly increasing');
      if(path==='rotation')for(let i=0;i<count;i++)unit(values,(i*(interpolation==='CUBICSPLINE'?3:1)+(interpolation==='CUBICSPLINE'?1:0))*4,'Keyframe rotation',channel.quantizedRotation?0.01:1e-3);
      duration=Math.max(duration,times[count-1]);
      return {node,path,width,interpolation,times,values,cursor:0};
    });
    return {name:String(clip.name??`animation_${clipIndex}`),duration,channels};
  });
  const restW=new Float64Array(baseWeights);
  const makeState=()=>({translations:baseT.slice(),rotations:baseQ.slice(),scales:baseS.slice(),morphWeights:restW.slice(),worldMatrices:new Float64Array(n*16),jointMatrices:new Float32Array(paletteSize)});
  const published=makeState(),scratch=makeState(),local=new Float64Array(n*16),inverse=new Float64Array(16),product=new Float64Array(16),result=new Float64Array(16);
  const fields=Object.keys(published);
  let version=0,currentTime=0,currentClip=-1,currentMode='rest',disposed=false,busy=false;
  // One accumulator per property binding, not per clip: a clip that does not
  // animate a property must not dilute another clip's contribution to it.
  const paths=['translation','rotation','scale','weights'];
  const poseFields=['translations','rotations','scales','morphWeights'];
  // Private double-buffered local state: failed evaluations and caller writes to
  // exposed arrays must never become the baseline of a subsequent edit. Swap
  // these four buffers at commit; no extra full-pose copy on sample/blend.
  const committed=Object.fromEntries(poseFields.map(field=>[field,scratch[field].slice()]));
  let currentRoot=null,currentMatrices=matrices;
  const rest=[baseT,baseQ,baseS,restW],bindings=new Map();
  let maxWidth=4;
  for(const clip of clips)for(const channel of clip.channels) {
    const kind=paths.indexOf(channel.path),key=channel.node*4+kind;
    if(!bindings.has(key))bindings.set(key,{
      key,kind,width:channel.width,
      offset:kind===3?morphOffsets[channel.node]:channel.node*channel.width,
    });
    channel.binding=bindings.get(key);maxWidth=Math.max(maxWidth,channel.width);
  }
  const totals=new Float64Array(n*4),values=new Float64Array(maxWidth);
  const delta=new Float64Array(4),weightedDelta=new Float64Array(4);
  const layerScratch=[];
  function checkStorage() {
    for(const field of fields){
      try{new Uint8Array(apply(typedBuffer,published[field],[]),0,0);if(apply(typedLength,published[field],[])!==scratch[field].length)throw new Error();}
      catch{fail('ANIMATION_OUTPUT_STORAGE','Published pose buffers must not be detached');}
    }
  }
  function run(operation,input,options) {
    if(disposed)fail('ANIMATION_DISPOSED','Animation player has been disposed');
    if(busy)fail('ANIMATION_REENTRANT','Pose evaluation cannot be reentered');
    busy=true;
    try { checkStorage();return operation(input,options); }
    finally { busy=false; }
  }
  function resetScratch() {
    scratch.translations.set(baseT);scratch.rotations.set(baseQ);
    scratch.scales.set(baseS);scratch.morphWeights.set(restW);
  }
  function clipTime(time,clip,loop) {
    const remainder=loop&&clip.duration>0?time%clip.duration:time;
    return loop&&clip.duration>0&&remainder<0?remainder+clip.duration:remainder;
  }
  function evaluate(time,{clip=0,loop=false,rootMatrix=null}={}) {
    finite(time,'Sample time');if(typeof loop!=='boolean')fail('ANIMATION_TIME','loop must be boolean');
    if(clip!==-1)integer(clip,clips.length,'Clip');
    const root=rootMatrix===null?null:affine(rootMatrix,'Root matrix');
    const selected=clip===-1?null:clips[clip],sampled=selected?clipTime(time,selected,loop):time;
    resetScratch();
    if(selected)for(const channel of selected.channels) {
      const {kind,offset}=channel.binding;
      sampleChannel(channel,sampled,scratch[poseFields[kind]],offset);
    }
    return publishPose(root,sampled,clip,clip===-1?'rest':'sample');
  }
  /**
   * blend([{clip,time,weight=1,loop=false,mode='normal',mask=null}], options)
   * combines LOCAL TRS/morph channels, then evaluates world/skin matrices once.
   * Normal weights normalize per binding above one; below one the rest pose
   * supplies the remainder. Rotations use ordered shortest-path SLERP, not a
   * matrix average. An optional mask has nodeCount weights in [0,1].
   *
   * Additive layers follow all normal layers, in input order. Numeric deltas
   * (including scale) are sample minus imported rest; quaternion deltas are
   * inverse(rest) * sample and are post-multiplied at their weighted strength.
   * This is NOT a preconverted Three.js additive-clip input contract.
   * No per-layer pose/matrix buffers are allocated while blending.
   * Descriptor/mask scratch grows only when a new layer slot first needs it.
   * blend publishes mode='blend', time=0, clip=-1; sample/reset keep their API.
   */
  function evaluateBlend(layers,{rootMatrix=null}={}) {
    if(!Array.isArray(layers)||layers.length>256)fail('ANIMATION_LAYERS','Expected at most 256 layers');
    const root=rootMatrix===null?null:affine(rootMatrix,'Root matrix');
    const count=layers.length;
    // Snapshot caller-owned inputs before touching pose scratch. In particular,
    // a mask may be a view into published pose storage.
    for(let i=0;i<count;i++) {
      const input=layers[i];
      if(!input||typeof input!=='object')fail('ANIMATION_LAYER','Invalid layer');
      const clip=integer(input.clip,clips.length,'Layer clip');
      const time=finite(input.time,'Layer time'),weight=finite(input.weight??1,'Layer weight');
      const loop=input.loop??false,mode=input.mode??'normal',mask=input.mask??null;
      if(weight<0)fail('ANIMATION_WEIGHT','Layer weight must be nonnegative');
      if(typeof loop!=='boolean')fail('ANIMATION_TIME','loop must be boolean');
      if(mode!=='normal'&&mode!=='additive')fail('ANIMATION_BLEND_MODE','Unknown layer mode');
      const layer=layerScratch[i]??(layerScratch[i]={mask:null});
      layer.clip=clips[clip];layer.time=clipTime(time,layer.clip,loop);
      layer.weight=weight;layer.mode=mode;layer.masked=mask!==null;
      if(mask!==null) {
        if((!Array.isArray(mask)&&!ArrayBuffer.isView(mask))||mask.length!==n)fail('ANIMATION_MASK','Mask must have nodeCount weights');
        layer.mask??=new Float64Array(n);
        for(let node=0;node<n;node++) {
          const value=finite(mask[node],'Mask weight');
          if(value<0||value>1)fail('ANIMATION_MASK','Mask weights must be in [0,1]');
          layer.mask[node]=value;
        }
      }
    }
    resetScratch();totals.fill(0);
    for(let i=0;i<count;i++) {
      const layer=layerScratch[i];if(layer.mode!=='normal'||layer.weight===0)continue;
      for(const channel of layer.clip.channels) {
        const weight=layer.weight*(layer.masked?layer.mask[channel.node]:1);
        if(weight===0)continue;
        const {key,kind,offset,width}=channel.binding,destination=scratch[poseFields[kind]];
        sampleChannel(channel,layer.time,values,0);
        const previous=totals[key],total=finite(previous+weight,'Accumulated weight');
        if(previous===0) {
          for(let j=0;j<width;j++)destination[offset+j]=values[j];
          if(kind===1)normalize(destination,offset);
        } else if(kind===1)mixQuaternion(destination,offset,values,0,weight/total);
        else for(let j=0;j<width;j++)destination[offset+j]=(1-weight/total)*destination[offset+j]+weight/total*values[j];
        totals[key]=total;
      }
    }
    for(const {key,kind,offset,width} of bindings.values()) {
      const weight=totals[key];if(weight===0||weight>=1)continue;
      const destination=scratch[poseFields[kind]],base=rest[kind];
      if(kind===1)mixQuaternion(destination,offset,base,offset,1-weight);
      else for(let j=0;j<width;j++)destination[offset+j]=weight*destination[offset+j]+(1-weight)*base[offset+j];
    }
    for(let i=0;i<count;i++) {
      const layer=layerScratch[i];if(layer.mode!=='additive'||layer.weight===0)continue;
      for(const channel of layer.clip.channels) {
        const weight=layer.weight*(layer.masked?layer.mask[channel.node]:1);
        if(weight===0)continue;
        const {kind,offset,width}=channel.binding,destination=scratch[poseFields[kind]],base=rest[kind];
        sampleChannel(channel,layer.time,values,0);
        if(kind===1) {
          for(let j=0;j<4;j++)delta[j]=base[offset+j]*(j===3?1:-1);
          normalize(delta,0);normalize(values,0);
          multiplyQuaternion(delta,0,delta,0,values,0);
          weightedDelta[0]=weightedDelta[1]=weightedDelta[2]=0;weightedDelta[3]=1;
          mixQuaternion(weightedDelta,0,delta,0,weight);
          multiplyQuaternion(destination,offset,destination,offset,weightedDelta,0);
          normalize(destination,offset);
        } else for(let j=0;j<width;j++)destination[offset+j]+=weight*(values[j]-base[offset+j]);
      }
    }
    return publishPose(root,0,-1,'blend');
  }
  /** Absolute local edits, evaluated together in hierarchy order. Omitted
   * fields and rootMatrix preserve the last committed pose, NOT public-array
   * mutations or dirty scratch from a failed operation. Matrix nodes keep their
   * representation; edits never silently decompose shear or retarget tracks.
   */
  function evaluateEdit(edits,options={}) {
    if(!options||typeof options!=='object'||Array.isArray(options)||
       Object.keys(options).some(key=>key!=='rootMatrix'))fail('ANIMATION_EDIT','Invalid edit options');
    if(!Array.isArray(edits)||edits.length>n)fail('ANIMATION_EDIT','Expected at most one edit per node');
    const inputRoot=options.rootMatrix;
    const root=inputRoot===undefined?currentRoot:inputRoot===null?null:affine(inputRoot,'Root matrix');
    const changes=[],seen=new Set();let nextMatrices=currentMatrices;
    // Complete validation and snapshotting precedes any scratch writes. Inputs
    // may explicitly borrow published storage; overlapping reads remain stable.
    for(const input of edits) {
      if(!input||typeof input!=='object'||Array.isArray(input)||
         Object.keys(input).some(key=>!['node','translation','rotation','scale','weights','matrix'].includes(key)))fail('ANIMATION_EDIT','Invalid node edit');
      const node=integer(input.node,n,'Edited node');
      if(seen.has(node))fail('ANIMATION_EDIT','Duplicate edited node');seen.add(node);
      const change={node};let count=0;
      for(const key of ['translation','rotation','scale','weights','matrix']) {
        const value=input[key];if(value===undefined)continue;count++;
        if(key==='matrix') {
          if(!matrices.has(node))fail('ANIMATION_EDIT','TRS nodes require TRS edits');
          change.matrix=affine(value,'Edited matrix');
        } else {
          if(key!=='weights'&&matrices.has(node))fail('ANIMATION_EDIT','Matrix nodes require matrix edits');
          const width=key==='weights'?morphOffsets[node+1]-morphOffsets[node]:key==='rotation'?4:3;
          if(!width)fail('ANIMATION_EDIT','Node has no morph weights');
          change[key]=numbers(value,width,'Edited '+key);
          if(key==='rotation'){unit(change[key],0,'Edited rotation');normalize(change[key],0);}
        }
      }
      if(!count)fail('ANIMATION_EDIT','Empty node edit');changes.push(change);
    }
    for(const field of poseFields)scratch[field].set(committed[field]);
    for(const change of changes) {
      const node=change.node;
      for(let kind=0;kind<4;kind++) {
        const value=change[paths[kind]];if(value===undefined)continue;
        scratch[poseFields[kind]].set(value,kind===3?morphOffsets[node]:node*(kind===1?4:3));
      }
      if(change.matrix) {
        if(nextMatrices===currentMatrices)nextMatrices=new Map(currentMatrices);
        nextMatrices.set(node,change.matrix);
      }
    }
    return publishPose(root,currentTime,currentClip,'edit',nextMatrices);
  }
  // The arrays in this snapshot belong to the caller. In particular, neither
  // source definitions nor writable published matrices are the solver's truth.
  function snapshotLocalPose() {
    return Object.freeze({format:'f3d-local-pose-v1',nodeCount:n,version,
      ...Object.fromEntries(poseFields.map(field=>[field,committed[field].slice()])),
      parents:parents.slice(),morphOffsets:morphOffsets.slice(),restRotations:baseQ.slice(),
      matrices:Object.freeze([...currentMatrices].map(([node,matrix])=>Object.freeze({node,matrix:matrix.slice()}))),
      rootMatrix:currentRoot===null?null:currentRoot.slice()});
  }
  function publishPose(root,sampled,clip,mode,matrixState=matrices) {
    for(const node of order) {
      if(matrixState.has(node))local.set(matrixState.get(node),node*16);
      else compose(scratch.translations,scratch.rotations,scratch.scales,node,local);
      if(parents[node]!==-1)multiply(scratch.worldMatrices,parents[node]*16,local,node*16,scratch.worldMatrices,node*16);
      else if(root)multiply(root,0,local,node*16,scratch.worldMatrices,node*16);
      else for(let k=0;k<16;k++)scratch.worldMatrices[node*16+k]=local[node*16+k];
    }
    for(const instance of instances) {
      const skin=skins[instance.skin];inverseAffine(scratch.worldMatrices,instance.node*16,inverse);
      for(let j=0;j<skin.joints.length;j++) {
        multiply(scratch.worldMatrices,skin.joints[j]*16,skin.inverseBind,j*16,product,0);
        multiply(inverse,0,product,0,result,0);
        scratch.jointMatrices.set(result,instance.offset+j*16);
      }
    }
    // A malformed sample, singular mesh or overflowing palette cannot expose a
    // half-updated pose. No callbacks or source effects run during evaluation.
    for(const field of fields)for(const value of scratch[field])if(!Number.isFinite(value))fail('ANIMATION_VALUE',`Non-finite ${field}`);
    checkStorage();
    for(const field of fields)apply(typedSet,published[field],[scratch[field]]);
    for(const field of poseFields){const previous=committed[field];committed[field]=scratch[field];scratch[field]=previous;}
    currentRoot=root;currentMatrices=matrixState;
    version++;currentTime=sampled;currentClip=clip;currentMode=mode;return player;
  }
  const player=Object.freeze({ ...published,nodeCount:n,morphOffsets:morphOffsets.slice(),
    clips:Object.freeze(clips.map(({name,duration})=>Object.freeze({name,duration}))),instances:Object.freeze(instances),
    sample(time,options){return run(evaluate,time,options);},
    blend(layers,options){return run(evaluateBlend,layers,options);},
    edit(edits,options){return run(evaluateEdit,edits,options);},
    snapshotLocalPose(){return run(snapshotLocalPose);},
    reset(){return run(evaluate,0,{clip:-1});},
    get version(){return version;},get time(){return currentTime;},get clip(){return currentClip;},get mode(){return currentMode;},
    dispose(){if(busy)fail('ANIMATION_REENTRANT','Cannot dispose during evaluation');disposed=true;},get disposed(){return disposed;},
  });
  evaluate(0,{clip:-1});version=0;
  return player;
}
