/** Conservative mesh-local bounds for the existing f32 morph/skin shader.
 * Scan source positions/weights once, then evaluate O(targets + used joints),
 * independent of vertex count. Skin weights form a nonnegative weighted hull;
 * their actual f32 sums (not an assumed sum of 1) and arithmetic error enlarge it.
 * This deliberately trades tightness for bounded update cost. No CPU skinning,
 * GPU readback, GPU resource, retained source vertex arrays, or animation clock.
 *
 * Floating-point expansion accounts for f32 rounding, contraction/reassociation
 * of short dot products, and optional subnormal flushing. Overflow/indeterminate
 * bounds fail OPEN: keep drawing rather than dropping potentially visible meshes.
 * Only the current animation_webgpu.mjs position profile is described, not custom
 * displacement shaders. WebGPU clipping is -w<=x,y<=w and 0<=z<=w, without division.
 * https://www.w3.org/TR/WGSL/#floating-point-evaluation
 * https://gpuweb.github.io/gpuweb/#coordinate-systems
 */
export class AnimationBoundsError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationBoundsError'; this.code = code; }
}
const fail = (code, text) => { throw new AnimationBoundsError(`ANIMATION_BOUNDS_${code}`, text); };
const EPS = 2 ** -23, MIN_NORMAL = 2 ** -126, MAX = 3.4028234663852886e38;
const WHOLE = Object.freeze([-Infinity, Infinity]);
const scalar = new Float32Array(1), bits = new Uint32Array(scalar.buffer);
function adjacent(value, up) {
  if (!Number.isFinite(value)) return value;
  if (value === 0) return up ? 2 ** -149 : -(2 ** -149);
  scalar[0] = value;
  bits[0] += (value > 0) === up ? 1 : -1;
  return scalar[0];
}
function expand(lo, hi, error = 0) {
  lo -= error; hi += error;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < -MAX || hi > MAX) return WHOLE;
  // Adjacent representable values enclose either rounding direction and exact
  // intermediates retained by contraction; subnormal operands/results may be 0.
  lo = adjacent(Math.fround(lo), false); hi = adjacent(Math.fround(hi), true);
  if (lo > -MIN_NORMAL && hi < MIN_NORMAL) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  else if (lo > 0 && lo < MIN_NORMAL) lo = 0;
  else if (hi < 0 && hi > -MIN_NORMAL) hi = 0;
  return [lo, hi];
}
function inputInterval(value) {
  const f = Math.fround(value);
  return f !== 0 && Math.abs(f) < MIN_NORMAL ? [Math.min(f, 0), Math.max(f, 0)] : [f, f];
}
function product(a, b) {
  if ((a[0] === 0 && a[1] === 0) || (b[0] === 0 && b[1] === 0)) return [0, 0];
  const values = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]];
  return expand(Math.min(...values), Math.max(...values));
}
function add(a, b) { return expand(a[0] + b[0], a[1] + b[1]); }
function dot(matrix, row, box) {
  if (box.some(range=>!Number.isFinite(range[0])||!Number.isFinite(range[1]))) return WHOLE;
  const terms = [0,1,2].map(i => product(inputInterval(matrix[i*4+row]), box[i]));
  terms.push(inputInterval(matrix[12+row]));
  const magnitude = terms.reduce((sum, t) => sum + Math.max(Math.abs(t[0]), Math.abs(t[1])), 0);
  // Four products and three additions, plus slack for permitted alternate dot
  // accumulation order and subnormal flushing. Cancellation uses absolute error.
  return expand(terms.reduce((sum,t)=>sum+t[0],0), terms.reduce((sum,t)=>sum+t[1],0),
    16 * EPS * magnitude + 16 * MIN_NORMAL);
}
function fixed(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value instanceof DataView || value.length !== length) fail('SHAPE', `Invalid ${label}`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('STORAGE', `${label} must be fixed and unshared`);
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail('STORAGE', `${label} is detached`); }
  }
  return value;
}
function finite(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) fail('VALUE', `${label} must fit finite f32`);
  return Math.fround(value);
}
function positive(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) fail('LIMIT', `Invalid ${label}`);
  return value;
}
function matrix(value, label) {
  fixed(value, 16, label);
  for (const v of value) if (typeof v !== 'number' || !Number.isFinite(v)) fail('VALUE', `${label} must be finite`);
  return value;
}

/** Return true unless the complete bound is definitely outside a clip plane.
 * Multiply viewProjection * world exactly as the renderer's host-side packer,
 * then apply conservative f32 dot bounds. No inverse or perspective divide;
 * supports orthographic/infinite/reversed depth projections and behind-eye boxes.
 */
export function animationBoundsVisible(bounds, viewProjection, worldMatrix) {
  matrix(viewProjection, 'View projection'); matrix(worldMatrix, 'World matrix');
  if (worldMatrix[3] !== 0 || worldMatrix[7] !== 0 || worldMatrix[11] !== 0 || worldMatrix[15] !== 1) fail('VALUE', 'World matrix must be affine');
  if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max) || bounds.min.length !== 3 || bounds.max.length !== 3) fail('SHAPE', 'Expected a bounds snapshot');
  const box = bounds.min.map((v,i) => [v, bounds.max[i]]);
  if (box.some(([lo,hi]) => !Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi)) return true;
  const clip = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) {
    clip[c*4+r] = viewProjection[r]*worldMatrix[c*4] + viewProjection[4+r]*worldMatrix[c*4+1]
      + viewProjection[8+r]*worldMatrix[c*4+2] + viewProjection[12+r]*worldMatrix[c*4+3];
  }
  if (clip.some(v=>!Number.isFinite(v))) return true;
  const q = [0,1,2,3].map(row=>dot(clip,row,box));
  if (q.some(([lo,hi])=>!Number.isFinite(lo)||!Number.isFinite(hi))) return true;
  // Each test is the largest possible signed plane distance. Strict comparison
  // preserves touching/crossing boxes, even if no individual corner is visible.
  return !(q[0][1]+q[3][1]<0 || q[3][1]-q[0][0]<0 ||
    q[1][1]+q[3][1]<0 || q[3][1]-q[1][0]<0 || q[2][1]<0 || q[3][1]-q[2][0]<0);
}

/** createAnimationBounds(pose, decodedGeometry, {maxComponents?, maxBytes?})
 * Only position-affecting geometry is read. maxComponents limits the source
 * scan; maxBytes counts retained typed summaries, not general JS heap overhead.
 * update() returns an immutable snapshot. Call after uploading the same pose.
 */
export function createAnimationBounds(pose, geometry, {maxComponents=16777216, maxBytes=16*1024*1024}={}) {
  positive(maxComponents, 'source component limit'); positive(maxBytes, 'summary byte limit');
  if (!pose || !Number.isSafeInteger(pose.nodeCount) || pose.nodeCount < 1 || !Array.isArray(pose.instances)) fail('POSE', 'Expected a packed pose');
  const node=geometry?.node, count=geometry?.positions?.length;
  if (!Number.isInteger(node)||node<0||node>=pose.nodeCount||!Number.isSafeInteger(count)||count<3||count%3) fail('SHAPE','Expected node and XYZ positions');
  const offsets=fixed(pose.morphOffsets,pose.nodeCount+1,'Morph offsets');
  const start=offsets[node],end=offsets[node+1];
  if (!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||end<start||end>(pose.morphWeights?.length??-1)) fail('POSE','Invalid morph range');
  const targets=geometry.morphTargets??[];
  if (!Array.isArray(targets)||targets.length!==end-start||targets.length>4096) fail('SHAPE','Mismatched morph target count');
  const targetCount=targets.length;
  const found=pose.instances.filter(x=>x.node===node);
  if(found.length>1)fail('POSE','Ambiguous mesh-local joint palette');
  const skin=found.length?{...found[0]}:null, vertices=count/3;
  if(skin&&(!Number.isSafeInteger(skin.offset)||skin.offset<0||skin.offset%16||!Number.isSafeInteger(skin.jointCount)||skin.jointCount<1||
    skin.offset+skin.jointCount*16>(pose.jointMatrices?.length??-1))) fail('POSE','Invalid instance palette');
  const influences=skin?(geometry.influences??4):0;
  if(skin&&(!Number.isInteger(influences)||influences<1||influences>32))fail('SHAPE','Expected 1..32 influences');
  if(!skin&&(geometry.weights!==undefined||geometry.joints!==undefined||geometry.influences!==undefined))fail('SHAPE','Skin attributes require an instance');
  // Determine the charged scan size before traversing caller vertex arrays.
  let components=count+(skin?vertices*influences*2:0);
  for(const target of targets) {
    if(!target||typeof target!=='object'||Array.isArray(target))fail('SHAPE','Invalid morph target');
    if(target.positions!==undefined)components+=count;
  }
  if(components>maxComponents)fail('LIMIT','Bounds source component limit exceeded');
  const summaryBytes=(targetCount+1)*6*8;
  if(summaryBytes>maxBytes)fail('LIMIT','Bounds summary byte limit exceeded');
  let summary=new Float64Array((targetCount+1)*6),usedJoints=null,smin=1,smax=1;
  function scan(values, at) {
    fixed(values,count,'Positions');
    summary.fill(Infinity,at,at+3);summary.fill(-Infinity,at+3,at+6);
    for(let i=0;i<count;i++) {
      const f=finite(values[i],'Position'),range=inputInterval(f),axis=i%3;
      summary[at+axis]=Math.min(summary[at+axis],range[0]);
      summary[at+3+axis]=Math.max(summary[at+3+axis],range[1]);
    }
  }
  const initialVersion=pose.version;
  scan(geometry.positions,0);
  for(let i=0;i<targets.length;i++)if(targets[i].positions!==undefined)scan(targets[i].positions,(i+1)*6);
  if(skin) {
    const length=vertices*influences,joints=fixed(geometry.joints,length,'Joint indices'),weights=fixed(geometry.weights,length,'Skin weights');
    const used=new Set();smin=Infinity;smax=0;
    for(let vertex=0;vertex<vertices;vertex++) {
      let original=0,sum=0,low=0;
      for(let k=0;k<influences;k++) {
        const at=vertex*influences+k,j=joints[at],w=finite(weights[at],'Skin weight');
        if(!Number.isInteger(j)||j<0||j>=skin.jointCount||weights[at]<0)fail('SHAPE','Invalid joint or negative skin weight');
        original+=weights[at];sum+=w;low+=w<MIN_NORMAL?0:w;
        if(w>0)used.add(j);
      }
      if(Math.abs(original-1)>1e-4)fail('SHAPE','Skin weights must sum to one within 1e-4');
      smin=Math.min(smin,low);smax=Math.max(smax,sum);
    }
    if(summaryBytes+used.size*4>maxBytes)fail('LIMIT','Joint summary byte limit exceeded');
    usedJoints=Uint32Array.from(used);
  }
  const byteLength=summary.byteLength+(usedJoints?.byteLength??0);
  const morphLength=pose.morphWeights.length,paletteLength=pose.jointMatrices?.length;
  let disposed=false,busy=false,current=null;
  function live() {
    if(disposed)fail('DISPOSED','Bounds have been disposed');
    if(pose.disposed||!Number.isSafeInteger(pose.version)||pose.version<0)fail('POSE','Pose is unavailable');
  }
  if(pose.version!==initialVersion)fail('CHANGED','Pose changed while preparing bounds');
  function update() {
    live();if(busy)fail('REENTRANT','Bounds update cannot be reentered');busy=true;
    const version=pose.version;
    try {
      fixed(pose.morphWeights,morphLength,'Morph weights');
      let box=[0,1,2].map(i=>[summary[i],summary[i+3]]);
      for(let t=0;t<targetCount;t++) {
        const w=finite(pose.morphWeights[start+t],'Morph weight');
        if(w===0)continue;
        for(let a=0;a<3;a++) box[a]=add(box[a],product(inputInterval(w),[summary[(t+1)*6+a],summary[(t+1)*6+3+a]]));
      }
      if(skin) {
        fixed(pose.jointMatrices,paletteLength,'Joint palette');
        const hull=[[Infinity,-Infinity],[Infinity,-Infinity],[Infinity,-Infinity]];
        const m=new Float32Array(16);
        for(const joint of usedJoints) {
          const offset=skin.offset+joint*16;
          for(let i=0;i<16;i++)m[i]=finite(pose.jointMatrices[offset+i],'Joint matrix');
          if(m[3]!==0||m[7]!==0||m[11]!==0||m[15]!==1)fail('POSE','Joint matrix must be affine');
          for(let a=0;a<3;a++) {
            const range=dot(m,a,box);hull[a][0]=Math.min(hull[a][0],range[0]);hull[a][1]=Math.max(hull[a][1],range[1]);
          }
        }
        // Any exact nonnegative weighted sum is inside this hull times the
        // actual weight sum. Account for n products + n accumulating additions.
        const gamma=(4*influences*EPS)/(1-4*influences*EPS);
        box=hull.map(range=>{
          const values=[range[0]*smin,range[0]*smax,range[1]*smin,range[1]*smax];
          return expand(Math.min(...values),Math.max(...values),
            gamma*Math.max(Math.abs(range[0]),Math.abs(range[1]))*smax+4*influences*MIN_NORMAL);
        });
      }
      live();if(pose.version!==version)fail('CHANGED','Pose changed during bounds update');
      current=Object.freeze({min:Object.freeze(box.map(x=>x[0])),max:Object.freeze(box.map(x=>x[1])),
        poseVersion:version,bounded:box.every(x=>Number.isFinite(x[0])&&Number.isFinite(x[1]))});
      return current;
    }finally{busy=false;}
  }
  const result=Object.freeze({node,vertexCount:vertices,sourceComponents:components,update,
    get snapshot(){live();return current;},get byteLength(){return disposed?0:byteLength;},
    get disposed(){return disposed;},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during bounds update');disposed=true;summary=null;usedJoints=null;current=null;},
  });
  update();return result;
}
