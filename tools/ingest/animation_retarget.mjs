/** Reference-relative motion transfer between packed animation poses.
 * Explicit node IDs, no guessed names, sampling clock, GPU work or source edits.
 * Reference poses default to the committed local poses at construction. Capture
 * both rigs in their intended reference stance, or pass independent snapshots.
 * This is a model-space rotational/root-motion profile, not SkeletonUtils parity.
 */
export class AnimationRetargetError extends Error {
  constructor(code,message){super(`${code}: ${message}`);this.name='AnimationRetargetError';this.code=code;}
}
const fail=(code,message)=>{throw new AnimationRetargetError('ANIMATION_RETARGET_'+code,message);};
function object(v,label){if(!v||typeof v!=='object'||Array.isArray(v))fail('SHAPE',`Expected ${label}`);return v;}
function fields(v,allowed,label){object(v,label);for(const key of Object.keys(v))if(!allowed.includes(key))fail('OPTIONS',`Unknown ${label} field: ${key}`);}
function finite(v,label){if(typeof v!=='number'||!Number.isFinite(v))fail('VALUE',`Invalid ${label}`);return v;}
function integer(v,min,max,label){if(!Number.isSafeInteger(v)||v<min||v>max)fail('LIMIT',`Invalid ${label}`);return v;}
function numbers(v,n,label){
  if((!Array.isArray(v)&&!ArrayBuffer.isView(v))||v instanceof DataView||v.length!==n)fail('SHAPE',`Invalid ${label}`);
  if(ArrayBuffer.isView(v)){
    if(!(v.buffer instanceof ArrayBuffer)||v.buffer.resizable)fail('STORAGE',`${label} needs fixed unshared storage`);
    try{new Uint8Array(v.buffer,0,0);}catch{fail('STORAGE',`Detached ${label}`);}
  }
  return Float64Array.from(v,x=>finite(x,label));
}
const identity=[0,0,0,1],zero=[0,0,0];
const inverse=q=>[-q[0],-q[1],-q[2],q[3]];
function unit(q){
  const length=Math.hypot(...q);
  if(!(length>0)||!Number.isFinite(length))fail('ROTATION','Undefined quaternion');
  return q.map(v=>v/length);
}
function quaternion(v){const q=Array.from(numbers(v,4,'rotation'));if(Math.abs(Math.hypot(...q)-1)>1e-3)fail('ROTATION','Rotation must be unit length');return unit(q);}
function multiply(a,b){
  const [x,y,z,w]=a,[X,Y,Z,W]=b;
  return unit([w*X+x*W+y*Z-z*Y,w*Y+y*W+z*X-x*Z,w*Z+z*W+x*Y-y*X,w*W-x*X-y*Y-z*Z]);
}
function rotate(q,v){
  const [x,y,z,w]=q,[a,b,c]=v,tx=2*(y*c-z*b),ty=2*(z*a-x*c),tz=2*(x*b-y*a);
  return [a+w*tx+y*tz-z*ty,b+w*ty+z*tx-x*tz,c+w*tz+x*ty-y*tx];
}
function slerp(a,b,t){
  let dot=a.reduce((s,v,i)=>s+v*b[i],0);const sign=dot<0?-1:1;dot=Math.min(1,Math.abs(dot));
  let x=1-t,y=t;
  if(1-dot>1e-12){const angle=Math.acos(dot),s=Math.sin(angle);x=Math.sin((1-t)*angle)/s;y=Math.sin(t*angle)/s;}
  return unit(a.map((v,i)=>x*v+y*sign*b[i]));
}
function snapshot(input,n){
  object(input,'reference pose');
  if(input.format!=='f3d-local-pose-v1'||input.nodeCount!==n)fail('SHAPE','Expected matching local-pose snapshot');
  const state={nodeCount:n,version:integer(input.version,0,Number.MAX_SAFE_INTEGER,'snapshot version'),
    parents:numbers(input.parents,n,'parents'),translations:numbers(input.translations,n*3,'translations'),
    rotations:numbers(input.rotations,n*4,'rotations'),scales:numbers(input.scales,n*3,'scales'),matrices:new Map()};
  if(!Array.isArray(input.matrices)||input.matrices.length>n)fail('SHAPE','Invalid matrix nodes');
  for(const entry of input.matrices){
    const node=integer(entry?.node,0,n-1,'matrix node');
    if(state.matrices.has(node))fail('SHAPE','Duplicate matrix node');
    state.matrices.set(node,numbers(entry.matrix,16,'node matrix'));
  }
  return state;
}
function capture(pose){
  const version=pose.version,result=snapshot(pose.snapshotLocalPose(),pose.nodeCount);
  if(pose.disposed||pose.version!==version||result.version!==version)fail('STALE','Pose changed during snapshot');
  return result;
}
function hierarchy(state){
  const n=state.nodeCount,children=Array.from({length:n},()=>[]),order=[];
  for(let i=0;i<n;i++){
    const p=integer(state.parents[i],-1,n-1,'parent');
    if(p===i)fail('HIERARCHY','Self-parent');
    if(p===-1)order.push(i);else children[p].push(i);
  }
  for(let i=0;i<order.length;i++)for(const child of children[order[i]])order.push(child);
  if(order.length!==n)fail('HIERARCHY','Cyclic hierarchy');return order;
}
function closure(state,nodes,order){
  const active=new Set();
  for(let node of nodes)while(node!==-1&&!active.has(node)){active.add(node);node=state.parents[node];}
  return order.filter(node=>active.has(node));
}
function sameRig(reference,current){
  if(current.parents.some((p,i)=>p!==reference.parents[i])||current.matrices.size!==reference.matrices.size||
    [...current.matrices.keys()].some(i=>!reference.matrices.has(i)))fail('HIERARCHY','Rig hierarchy/representation changed');
}
// Similarity transforms admit an unambiguous rotation and invertible position
// transfer. Do not silently polar-decompose reflected, sheared or stretched rigs.
function local(state,node){
  const t=Array.from(state.translations.subarray(node*3,node*3+3)),m=state.matrices.get(node);
  if(!m){
    const s=Array.from(state.scales.subarray(node*3,node*3+3)),scale=s[0];
    if(!(scale>0)||s.some(v=>Math.abs(v/scale-1)>1e-8))fail('TRANSFORM','Active rig ancestors require positive uniform scale');
    return {t,q:quaternion(state.rotations.subarray(node*4,node*4+4)),s:scale};
  }
  if(m[3]!==0||m[7]!==0||m[11]!==0||m[15]!==1)fail('TRANSFORM','Matrix must be affine');
  const s=Math.hypot(m[0],m[1],m[2]);if(!(s>0)||!Number.isFinite(s))fail('TRANSFORM','Singular matrix ancestor');
  const r=[m[0]/s,m[1]/s,m[2]/s,m[4]/s,m[5]/s,m[6]/s,m[8]/s,m[9]/s,m[10]/s];
  for(let a=0;a<3;a++)for(let b=0;b<3;b++)if(Math.abs(r[a*3]*r[b*3]+r[a*3+1]*r[b*3+1]+r[a*3+2]*r[b*3+2]-(a===b?1:0))>1e-8)fail('TRANSFORM','Matrix ancestor has shear or nonuniform scale');
  if(r[0]*(r[4]*r[8]-r[5]*r[7])-r[3]*(r[1]*r[8]-r[2]*r[7])+r[6]*(r[1]*r[5]-r[2]*r[4])<0)fail('TRANSFORM','Reflected matrix ancestor');
  let q;const trace=r[0]+r[4]+r[8];
  if(trace>0){const k=2*Math.sqrt(1+trace);q=[(r[5]-r[7])/k,(r[6]-r[2])/k,(r[1]-r[3])/k,k/4];}
  else{
    let a=0;if(r[4]>r[0])a=1;if(r[8]>r[a*3+a])a=2;const b=(a+1)%3,c=(a+2)%3,k=2*Math.sqrt(1+r[a*3+a]-r[b*3+b]-r[c*3+c]);
    q=[0,0,0,0];q[a]=k/4;q[b]=(r[a*3+b]+r[b*3+a])/k;q[c]=(r[a*3+c]+r[c*3+a])/k;q[3]=(r[b*3+c]-r[c*3+b])/k;
  }
  return {t:[m[12],m[13],m[14]],q:unit(q),s};
}
function append(world,state,node,value){
  const parent=world.get(state.parents[node]),rotated=parent?rotate(parent.q,value.t.map(x=>x*parent.s)):value.t;
  const result={q:parent?multiply(parent.q,value.q):value.q,t:parent?rotated.map((v,i)=>v+parent.t[i]):rotated,s:(parent?.s??1)*value.s};
  if(!(result.s>0)||!Number.isFinite(result.s)||result.t.some(v=>!Number.isFinite(v)))fail('TRANSFORM','Rig transform overflow/underflow');
  world.set(node,result);return result;
}
function worldPose(state,order){const world=new Map();for(const node of order)append(world,state,node,local(state,node));return world;}

/** mapping:[{source,target,weight?}] transfers model-space rotation deltas from
 * sourceReference to targetReference. Destination translations/scale/morphs and
 * unmapped nodes stay current. rootMotion:{source,target,scale?} additionally
 * transfers reference-relative model-space displacement (no bone stretching).
 * alignment is a unit quaternion mapping source-model axes to target-model axes.
 * External pose rootMatrix placement is deliberately ignored and preserved.
 * All mapped destination nodes must be TRS; active ancestors may use positive
 * uniform affine matrices. Unrelated matrix/sheared nodes are not interpreted.
 * maxNodes bounds the combined rigs; maxMappings bounds rotation bindings.
 */
export function createAnimationRetargeter(source,target,options={}){
  fields(options,['mapping','rootMotion','alignment','sourceReference','targetReference','maxNodes','maxMappings'],'retarget options');
  const {mapping,rootMotion=null,alignment=identity,sourceReference,targetReference,maxNodes=65536,maxMappings=4096}=options;
  integer(maxNodes,2,131072,'node budget');integer(maxMappings,1,65536,'mapping budget');
  function live(){
    if(disposed)fail('DISPOSED','Retargeter is disposed');
    for(const p of [source,target])if(!p||p.disposed||typeof p.snapshotLocalPose!=='function'||!Number.isSafeInteger(p.version)||p.version<0)fail('POSE','Expected live snapshot-capable poses');
    if(typeof target.edit!=='function'||source===target)fail('POSE','Source and editable target must be distinct poses');
  }
  let disposed=false,busy=false;live();
  const sn=integer(source.nodeCount,1,maxNodes,'source node count'),tn=integer(target.nodeCount,1,maxNodes-sn,'target node count');
  if(!Array.isArray(mapping)||mapping.length>maxMappings)fail('LIMIT','Expected bounded mapping array');
  const align=quaternion(alignment),mapped=new Map();
  const bindings=mapping.map(entry=>{
    fields(entry,['source','target','weight'],'mapping');
    const from=integer(entry.source,0,sn-1,'source node'),to=integer(entry.target,0,tn-1,'target node'),weight=finite(entry.weight??1,'mapping weight');
    if(mapped.has(to)||weight<0||weight>1)fail('MAPPING','Duplicate target or invalid weight');
    const binding=Object.freeze({source:from,target:to,weight});mapped.set(to,binding);return binding;
  });
  let root=null;
  if(rootMotion!==null){
    fields(rootMotion,['source','target','scale'],'root motion');
    root={source:integer(rootMotion.source,0,sn-1,'source root'),target:integer(rootMotion.target,0,tn-1,'target root'),scale:finite(rootMotion.scale??1,'root-motion scale')};
    if(root.scale<0)fail('VALUE','Root-motion scale must be nonnegative');
  }
  if(!bindings.length&&!root)fail('MAPPING','At least one rotation or root-motion binding is required');
  let src=capture(source),dst=capture(target);
  let sr=sourceReference===undefined?src:snapshot(sourceReference,sn),tr=targetReference===undefined?dst:snapshot(targetReference,tn);
  sameRig(sr,src);sameRig(tr,dst);
  const so=closure(sr,[...bindings.map(b=>b.source),...(root?[root.source]:[])],hierarchy(sr));
  const to=closure(tr,[...bindings.map(b=>b.target),...(root?[root.target]:[])],hierarchy(tr));
  for(const node of [...mapped.keys(),...(root?[root.target]:[])])if(tr.matrices.has(node))fail('MAPPING','Mapped destination nodes must use TRS');
  let sw=worldPose(sr,so),tw=worldPose(tr,to);
  // A * sourceWorld * inverse(sourceReferenceWorld) * inverse(A) * targetReferenceWorld.
  const corrections=new Map(bindings.map(b=>[b.target,multiply(multiply(inverse(sw.get(b.source).q),inverse(align)),tw.get(b.target).q)]));
  worldPose(src,so);worldPose(dst,to);live();
  if(source.version!==src.version||target.version!==dst.version)fail('STALE','Pose changed during retarget setup');
  src=dst=null;
  const result=Object.freeze({mapping:Object.freeze(bindings),
    apply(settings={}){
      live();if(busy)fail('REENTRANT','Retargeting cannot be reentered');busy=true;
      try{
        fields(settings,['weight'],'apply options');const weight=finite(settings.weight??1,'weight');
        if(weight<0||weight>1)fail('VALUE','Weight must be in [0,1]');
        const before=target.version,sourceVersion=source.version;
        if(weight===0)return Object.freeze({sourceVersion,targetVersion:before,updatedNodes:Object.freeze([])});
        const s=capture(source),t=capture(target);sameRig(sr,s);sameRig(tr,t);
        const current=worldPose(s,so),output=new Map(),edits=[];
        for(const node of to){
          const value=local(t,node),originalQ=value.q,originalT=value.t,binding=mapped.get(node);let changed=false;
          if(binding&&binding.weight>0){
            const desired=multiply(multiply(align,current.get(binding.source).q),corrections.get(node));
            const parent=output.get(t.parents[node])?.q??identity;
            value.q=multiply(inverse(parent),desired);changed=true;
          }
          const edit={node};if(changed)edit.rotation=slerp(originalQ,value.q,weight*binding.weight);
          if(root?.target===node){
            const delta=rotate(align,current.get(root.source).t.map((v,i)=>(v-sw.get(root.source).t[i])*root.scale));
            const desired=tw.get(node).t.map((v,i)=>v+delta[i]),parent=output.get(t.parents[node]);
            const translation=parent?rotate(inverse(parent.q),desired.map((v,i)=>(v-parent.t[i])/parent.s)):desired;
            value.t=translation;edit.translation=originalT.map((v,i)=>(1-weight)*v+weight*translation[i]);changed=true;
          }
          // Solve descendants against the complete target solution. Blending
          // an ancestor before solving a child would amplify inherited motion.
          // Only the eventual local edits are weighted against the input pose.
          append(output,t,node,value);if(changed)edits.push(edit);
        }
        live();if(source.version!==sourceVersion||target.version!==before)fail('STALE','Pose changed before retarget publication');
        if(before===Number.MAX_SAFE_INTEGER)fail('LIMIT','Destination version exhausted');
        // One existing transactional publication, including all affected skin
        // palettes. A final palette/storage failure leaves destination unchanged.
        if(edits.length)target.edit(edits);
        return Object.freeze({sourceVersion,targetVersion:target.version,updatedNodes:Object.freeze(edits.map(e=>e.node))});
      }finally{busy=false;}
    },
    get disposed(){return disposed;},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during retargeting');disposed=true;source=target=sr=tr=sw=tw=null;corrections.clear();},
  });
  return result;
}
