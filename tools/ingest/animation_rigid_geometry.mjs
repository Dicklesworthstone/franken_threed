/** Shared, immutable GPU vertices for rigid animated nodes. The 40-byte layout
 * matches animation_webgpu: position XYZ, normal XYZ, tangent XYZW. No compute
 * shader, palette buffer, queue write or submit is needed on update: only the
 * per-node world snapshot changes. This is NOT hardware-instanced drawing.
 *
 * Opt-in only. Skinning, morph targets and regenerated flat normals stay on the
 * existing compute path. Float32 conversion happens once during registration;
 * identical packed bytes AND attribute layouts share a buffer. Hash collisions
 * are resolved by complete word comparison, never by hash alone.
 *
 * The pool owns GPU buffers and retained comparison bytes. Each is bounded by
 * maxBytes; one incoming packed snapshot adds at most maxBytes transient CPU
 * bytes. Per-handle transforms are bounded by maxMeshes. A handle owns a shared
 * reference, not its buffer (bufferBytes=0; sharedBufferBytes reports its size).
 * Dispose the last handle or the pool to retire the allocation. Do not write to
 * borrowed vertex buffers. Pose/device/materials and submitted draws are not owned.
 */
export class AnimationRigidError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name='AnimationRigidError'; this.code=code; }
}
const fail=(code,message)=>{throw new AnimationRigidError('ANIMATION_RIGID_'+code,message);};
const FIELDS=['node','positions','normals','tangents','morphTargets','flatNormals'];
function integer(n,min,max,label) {
  if(!Number.isSafeInteger(n)||n<min||n>max)fail('LIMIT',`Invalid ${label}`);
  return n;
}
function fixed(value,length,label) {
  if(!ArrayBuffer.isView(value)||value instanceof DataView||!(value.buffer instanceof ArrayBuffer)||value.buffer.resizable||value.length!==length)fail('STORAGE',`Invalid ${label} storage`);
  try{new Uint8Array(value.buffer,0,0);}catch{fail('STORAGE',`Detached ${label}`);}
}
function numbers(value,length,label) {
  if((!Array.isArray(value)&&!ArrayBuffer.isView(value))||value.length!==length)fail('GEOMETRY',`Invalid ${label}`);
  if(ArrayBuffer.isView(value))fixed(value,length,label);
  return value;
}
function finite(value) {
  if(typeof value!=='number'||!Number.isFinite(value)||!Number.isFinite(Math.fround(value)))fail('VALUE','Expected finite f32-representable data');
  return value;
}
/** A conservative optimization decision; false retains ordinary deformation. */
export function canUseRigidAnimationGeometry(pose,geometry) {
  return !!geometry && typeof geometry==='object' && !Array.isArray(geometry) &&
    Object.keys(geometry).every(key=>FIELDS.includes(key)) &&
    (geometry.flatNormals===undefined||geometry.flatNormals===false) &&
    (geometry.morphTargets===undefined||(Array.isArray(geometry.morphTargets)&&geometry.morphTargets.length===0)) &&
    Number.isSafeInteger(geometry.node) && geometry.node>=0 && geometry.node<pose?.nodeCount &&
    Array.isArray(pose.instances) && !pose.instances.some(s=>s.node===geometry.node) &&
    pose.morphOffsets?.length===pose.nodeCount+1 && Number.isSafeInteger(pose.morphOffsets[geometry.node]) &&
    pose.morphOffsets[geometry.node]>=0 && pose.morphOffsets[geometry.node]===pose.morphOffsets[geometry.node+1];
}

export function createGpuRigidGeometryPool(device,pose,{
  maxBytes=128*1024*1024,maxMeshes=4096,maxComponents=16777216,label='f3d-rigid',
}={}) {
  integer(maxBytes,1,Number.MAX_SAFE_INTEGER,'byte budget');integer(maxMeshes,1,4096,'mesh budget');integer(maxComponents,1,Number.MAX_SAFE_INTEGER,'component budget');
  if(typeof label!=='string'||!device?.limits||typeof device.createBuffer!=='function'||typeof device.pushErrorScope!=='function'||
    typeof device.popErrorScope!=='function'||typeof device.queue?.onSubmittedWorkDone!=='function'||typeof device.lost?.then!=='function')fail('DEVICE','Lend a WebGPU device');
  const nodeCount=integer(pose?.nodeCount,0,65536,'pose node count'),worlds=pose.worldMatrices;
  fixed(worlds,nodeCount*16,'pose matrices');fixed(pose.morphOffsets,nodeCount+1,'morph offsets');
  if(!Array.isArray(pose.instances))fail('POSE','Expected skin instance metadata');
  const entries=new Set(),buckets=new Map(),handles=new Set();
  let bytes=0,busy=false,capturing=false,disposed=false,terminal=null,pending=Promise.resolve(),stop;
  const stopped=new Promise((_,reject)=>{stop=reject;});stopped.catch(()=>{});
  function retire(entry) {
    if(entry.retired)return;entry.retired=true;entry.buffer?.destroy();
    if(entries.delete(entry))bytes-=entry.words.byteLength;
    const bucket=buckets.get(entry.key);if(bucket){const at=bucket.indexOf(entry);if(at!==-1)bucket.splice(at,1);if(!bucket.length)buckets.delete(entry.key);}
    entry.words=null;
  }
  function release(){for(const entry of entries)retire(entry);handles.clear();}
  function stopWith(error){terminal??=error;release();stop(terminal);return terminal;}
  // Observe device loss even while an error-scope/completion promise is pending.
  device.lost.then(info=>{if(!disposed)stopWith(new AnimationRigidError('ANIMATION_RIGID_LOST',info?.message||'WebGPU device lost'));},
    error=>{if(!disposed)stopWith(error);}).catch(()=>{});
  function live(){if(terminal)throw terminal;if(disposed)fail('DISPOSED','Rigid geometry pool was disposed');if(pose.disposed)fail('POSE','Borrowed pose was disposed');}
  function world(node,out) {
    live();if(pose.nodeCount!==nodeCount||pose.worldMatrices!==worlds)fail('POSE','Pose storage identity changed');
    fixed(worlds,nodeCount*16,'pose matrices');const stamp=integer(pose.version,0,Number.MAX_SAFE_INTEGER,'pose version');
    for(let i=0;i<16;i++)out[i]=finite(worlds[node*16+i]);
    if(out[3]!==0||out[7]!==0||out[11]!==0||out[15]!==1)fail('TRANSFORM','World matrix must be affine');
    if(pose.version!==stamp||pose.disposed)fail('CHANGED','Pose changed during transform capture');return stamp;
  }
  function pack(geometry) {
    if(!canUseRigidAnimationGeometry(pose,geometry))fail('DYNAMIC','This geometry needs the ordinary deformation path');
    const node=geometry.node,length=geometry.positions?.length;
    integer(length,3,maxComponents,'position extent');if(length%3)fail('GEOMETRY','Positions must be XYZ');
    const count=length/3,positions=numbers(geometry.positions,length,'positions');
    const normals=geometry.normals===undefined?null:numbers(geometry.normals,length,'normals');
    const tangents=geometry.tangents===undefined?null:numbers(geometry.tangents,count*4,'tangents');
    if(length+(normals?length:0)+(tangents?count*4:0)>maxComponents)fail('LIMIT','Geometry exceeds component budget');
    const size=count*40;
    if(size>maxBytes||!Number.isSafeInteger(device.limits.maxBufferSize)||size>device.limits.maxBufferSize)fail('LIMIT','Packed vertices exceed byte/device budget');
    const packed=new Float32Array(count*10);
    for(let v=0;v<count;v++) {
      for(let a=0;a<3;a++){
        packed[v*10+a]=finite(positions[v*3+a]);
        packed[v*10+3+a]=normals?finite(normals[v*3+a]):0;
        packed[v*10+6+a]=tangents?finite(tangents[v*4+a]):0;
      }
      const w=tangents?finite(tangents[v*4+3]):1;if(Math.abs(w)!==1)fail('GEOMETRY','Tangent handedness must be -1 or 1');packed[v*10+9]=w;
    }
    const words=new Uint32Array(packed.buffer);let hash=2166136261;
    for(const word of words)hash=Math.imul(hash^word,16777619)>>>0;
    const mask=(normals?1:0)|(tangents?2:0),key=`${mask}:${count}:${hash}`;
    const attributes=[{shaderLocation:0,offset:0,format:'float32x3'}];
    if(normals)attributes.push({shaderLocation:1,offset:12,format:'float32x3'});
    if(tangents)attributes.push({shaderLocation:2,offset:24,format:'float32x4'});
    const vertexLayout=Object.freeze({arrayStride:40,stepMode:'vertex',attributes:Object.freeze(attributes.map(Object.freeze))});
    return {node,count,words,key,vertexLayout};
  }
  async function allocate(input) {
    const entry={...input,buffer:null,refs:0,retired:false};entries.add(entry);bytes+=input.words.byteLength;
    let syncError,depth=0;
    try {
      device.pushErrorScope('validation');depth++;device.pushErrorScope('out-of-memory');depth++;
      entry.buffer=device.createBuffer({label,size:input.words.byteLength,usage:32|4,mappedAtCreation:true});
      new Uint32Array(entry.buffer.getMappedRange()).set(input.words);entry.buffer.unmap();
    }catch(error){syncError=error;}
    // Pop both scopes synchronously. The device stack is shared with callers.
    const scopes=[];while(depth-->0){try{scopes.push(device.popErrorScope());}catch(error){syncError??=error;}}
    pending=Promise.all(scopes).then(errors=>{
      if(syncError)throw syncError;const error=errors.find(Boolean);
      if(error)fail('DEVICE',error.message||'GPU buffer allocation failed');
    }).catch(error=>{throw stopWith(error);});pending.catch(()=>{});
    capturing=false;
    try{await Promise.race([pending,stopped]);live();return entry;}
    catch(error){retire(entry);throw error;}
  }
  function instance(entry,node,initialWorld,stamp) {
    let released=false,updating=false,version=0,poseVersion=stamp;
    const worldMatrix=initialWorld,scratch=new Float64Array(16);
    const handle=Object.freeze({node,vertexCount:entry.count,vertexLayout:entry.vertexLayout,vertexBuffer:entry.buffer,
      bufferBytes:0,sharedBufferBytes:entry.words.byteLength,
      worldMatrix,get version(){return version;},get poseVersion(){return poseVersion;},
      get disposed(){return released||disposed;},get failed(){return terminal!==null;},
      update(){
        live();if(released)fail('DISPOSED','Rigid mesh was disposed');if(updating)fail('REENTRANT','Rigid mesh update cannot be reentered');
        updating=true;try{fixed(worldMatrix,16,'mesh transform');const next=world(node,scratch);worldMatrix.set(scratch);poseVersion=next;version++;return handle;}finally{updating=false;}
      },
      async whenIdle(){live();if(released)fail('DISPOSED','Rigid mesh was disposed');await result.whenIdle();return handle;},
      dispose(){if(updating)fail('REENTRANT','Cannot dispose while updating');if(!released){released=true;handles.delete(handle);if(--entry.refs===0)retire(entry);}},
    });
    entry.refs++;handles.add(handle);return handle;
  }
  const result=Object.freeze({
    /** maxAdditionalBytes limits new GPU storage; an existing match needs zero. */
    async addMesh(geometry,{maxAdditionalBytes=maxBytes}={}) {
      live();if(busy)fail('REENTRANT','Register rigid meshes sequentially');busy=true;capturing=true;
      let entry,created=false;
      try {
        integer(maxAdditionalBytes,0,Number.MAX_SAFE_INTEGER,'remaining GPU budget');
        if(handles.size>=maxMeshes)fail('LIMIT','Too many rigid meshes');
        const input=pack(geometry),matrix=new Float64Array(16),stamp=world(input.node,matrix);
        entry=buckets.get(input.key)?.find(e=>e.words.every((word,i)=>word===input.words[i]));
        if(!entry){
          if(input.words.byteLength>Math.min(maxBytes-bytes,maxAdditionalBytes))fail('LIMIT','Unique rigid vertices exceed GPU budget');
          created=true;entry=await allocate(input);
          if(pose.version!==stamp||pose.disposed)fail('CHANGED','Pose changed while preparing rigid geometry');
          const bucket=buckets.get(input.key)??[];bucket.push(entry);buckets.set(input.key,bucket);
        }
        live();return instance(entry,input.node,matrix,stamp);
      }catch(error){if(created&&entry&&entry.refs===0)retire(entry);throw error;}
      finally{capturing=false;busy=false;}
    },
    get bufferBytes(){return bytes;},get uniqueGeometries(){return entries.size;},get meshCount(){return handles.size;},
    get disposed(){return disposed;},get failed(){return terminal!==null;},
    async whenIdle(){
      live();try{await Promise.race([Promise.all([pending,device.queue.onSubmittedWorkDone()]),stopped]);}
      catch(error){throw stopWith(error);}live();return result;
    },
    dispose(){if(capturing)fail('REENTRANT','Cannot dispose during synchronous capture');if(!disposed){disposed=true;release();stop(new AnimationRigidError('ANIMATION_RIGID_DISPOSED','Rigid geometry pool was disposed'));}},
  });
  return result;
}
