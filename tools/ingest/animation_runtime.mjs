/**
 * Packed glTF animation poses. No parser, filesystem, DOM, renderer or Wasm
 * initialization is needed at import time. Instances own all mutable storage.
 *
 * This is an explicit single-clip sampler, not an AnimationMixer replacement:
 * sample() resets untargeted values to the imported rest pose, and loop timing
 * is an explicit caller choice. Outputs retain identity across samples. Input
 * definitions are copied once; edits to published output arrays are not inputs.
 *
 * Conventions: glTF 2.0 section 3.11 / Appendix C; column-major T*R*S;
 * mesh-local palette = inverse(meshWorld) * jointWorld * inverseBindMatrix.
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
 */
export class AnimationPoseError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationPoseError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationPoseError(code, message); };
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

/**
 * Create a reusable pose instance from a build-time decoded definition.
 * All indices use original glTF node/skin order. `instances` lists each skinned
 * mesh node, including several mesh nodes referencing the same skin. Palettes
 * are mesh-local, so those instances must NOT accidentally share a palette.
 * No implicit wall-clock, event scheduling, blending or public Three.js mutation.
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
  const fields=Object.keys(published);let version=0,currentTime=0,currentClip=-1,disposed=false;
  function evaluate(time,{clip=0,loop=false,rootMatrix=null}={}) {
    if(disposed)fail('ANIMATION_DISPOSED','Animation player has been disposed');
    // Published arrays are reusable player-owned storage. Reject detached buffers
    // before copying any field, including empty outputs transferred to workers.
    for(const field of fields){
      try{new Uint8Array(published[field].buffer,0,0);if(published[field].length!==scratch[field].length)throw new Error();}
      catch{fail('ANIMATION_OUTPUT_STORAGE','Published pose buffers must not be detached');}
    }
    finite(time,'Sample time');if(typeof loop!=='boolean')fail('ANIMATION_TIME','loop must be boolean');
    if(clip!==-1)integer(clip,clips.length,'Clip');
    const root=rootMatrix===null?null:affine(rootMatrix,'Root matrix');
    const selected=clip===-1?null:clips[clip];
    const remainder=loop&&selected?.duration>0?time%selected.duration:time;
    const sampled=loop&&selected?.duration>0&&remainder<0?remainder+selected.duration:remainder;
    scratch.translations.set(baseT);scratch.rotations.set(baseQ);scratch.scales.set(baseS);scratch.morphWeights.set(restW);
    if(selected)for(const channel of selected.channels) {
      const field=channel.path==='translation'?'translations':channel.path==='rotation'?'rotations':channel.path==='scale'?'scales':'morphWeights';
      const offset=channel.path==='weights'?morphOffsets[channel.node]:channel.node*channel.width;
      sampleChannel(channel,sampled,scratch[field],offset);
    }
    for(const node of order) {
      if(matrices.has(node))local.set(matrices.get(node),node*16);
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
    for(const field of fields)published[field].set(scratch[field]);
    version++;currentTime=sampled;currentClip=clip;return player;
  }
  const player=Object.freeze({ ...published,nodeCount:n,morphOffsets:morphOffsets.slice(),
    clips:Object.freeze(clips.map(({name,duration})=>Object.freeze({name,duration}))),instances:Object.freeze(instances),
    sample:evaluate,reset(){return evaluate(0,{clip:-1});},
    get version(){return version;},get time(){return currentTime;},get clip(){return currentClip;},
    dispose(){disposed=true;},get disposed(){return disposed;},
  });
  evaluate(0,{clip:-1});version=0;
  return player;
}
