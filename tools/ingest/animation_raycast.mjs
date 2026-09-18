/** Current-pose geometric queries over borrowed CPU deformation outputs.
 * BVHs are built once, refitted on changed deformation versions, and traversed
 * synchronously. No device, readback, animation clock, events or source routing.
 * Positions are the deformer's mesh-local Float32 outputs; world-space query
 * coordinates must remain finite/f32-representable, as on the renderer route.
 * This is triangle selection, not raster/alpha-test or Three.js Raycaster parity.
 */
export class AnimationRaycastError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationRaycastError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationRaycastError('ANIMATION_PICK_' + code, message); };
const finite = (v, label) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) fail('VALUE', `${label} must be finite`);
  return v;
};
const object = (v, label) => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) fail('SHAPE', `Expected ${label} object`);
  return v;
};
function fields(v, allowed, label) {
  object(v, label);
  for (const key of Object.keys(v)) if (!allowed.includes(key)) fail('OPTION', `Unsupported ${label} field: ${key}`);
}
function array(v, length, label) {
  if ((!Array.isArray(v) && !ArrayBuffer.isView(v)) || v instanceof DataView || v.length !== length) fail('SHAPE', `Invalid ${label}`);
  if (ArrayBuffer.isView(v)) {
    if (!(v.buffer instanceof ArrayBuffer) || v.buffer.resizable) fail('STORAGE', `${label} requires fixed unshared storage`);
    try { new Uint8Array(v.buffer, 0, 0); } catch { fail('STORAGE', `${label} is detached`); }
  }
  return v;
}
function vector(v, length, label) { return Array.from(array(v, length, label), x => finite(x, label)); }
function unit(v) {
  const scale = Math.max(...v.map(Math.abs));
  if (scale === 0) fail('RAY', 'Ray direction must be nonzero');
  const scaled = v.map(x => x / scale), n = Math.hypot(...scaled);
  return scaled.map(x => x / n);
}
function rayInput(input) {
  fields(input, ['origin', 'direction', 'near', 'far'], 'ray');
  const origin = vector(input.origin, 3, 'ray origin'), direction = unit(vector(input.direction, 3, 'ray direction'));
  for (const x of origin) if (!Number.isFinite(Math.fround(x))) fail('RAY', 'Ray origin exceeds the renderer coordinate range');
  const near = finite(input.near ?? 0, 'near distance'), far = input.far ?? Infinity;
  if (near < 0 || typeof far !== 'number' || Number.isNaN(far) || far < near) fail('RAY', 'Require 0 <= near <= far');
  let kz = 0; for (let a = 1; a < 3; a++) if (Math.abs(direction[a]) > Math.abs(direction[kz])) kz = a;
  let kx = (kz + 1) % 3, ky = (kx + 1) % 3;
  if (direction[kz] < 0) [kx, ky] = [ky, kx];
  return {origin, direction, near, far, kz, kx, ky, sx: -direction[kx] / direction[kz], sy: -direction[ky] / direction[kz], sz: 1 / direction[kz]};
}

/** Convert an existing gltf_scene_view sample and NDC [x,y] (Y up) to a ray.
 * The ray starts on the camera plane (orthographic) or at its optical center.
 * Camera clipping is NOT imposed: raycast's near/far are world-ray distances.
 */
export function rayFromAnimationCamera(sample, ndc) {
  object(sample, 'camera sample');
  const point = vector(ndc, 2, 'NDC point');
  if (point.some(x => x < -1 || x > 1)) fail('CAMERA', 'NDC coordinates must be in [-1,1]');
  const v = vector(sample.viewMatrix, 16, 'view matrix'), p = vector(sample.projectionMatrix, 16, 'projection matrix');
  const position = vector(sample.cameraPosition, 3, 'camera position'), type = sample.type;
  if (!['perspective', 'orthographic'].includes(type) || p[0] <= 0 || p[5] <= 0 ||
      [1,2,3,4,6,7].some(i => p[i] !== 0) || v[3] !== 0 || v[7] !== 0 || v[11] !== 0 || v[15] !== 1) fail('CAMERA', 'Unsupported camera matrices');
  if (type === 'perspective' ? p[11] !== -1 || p[15] !== 0 || p[12] !== 0 || p[13] !== 0 :
      p[11] !== 0 || p[15] !== 1 || p[8] !== 0 || p[9] !== 0) fail('CAMERA', 'Unsupported camera projection');
  const axes = [0,1,2].map(r => [v[r],v[4+r],v[8+r]]);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    const dot = axes[a].reduce((n, x, i) => n + x * axes[b][i], 0);
    if (Math.abs(dot - (a === b ? 1 : 0)) > 1e-8) fail('CAMERA', 'Camera view must be orthonormal');
  }
  const x = (point[0] + (type === 'perspective' ? p[8] : -p[12])) / p[0];
  const y = (point[1] + (type === 'perspective' ? p[9] : -p[13])) / p[5];
  const origin = type === 'perspective' ? position : position.map((c, i) => c + axes[0][i]*x + axes[1][i]*y);
  const direction = type === 'perspective' ? axes[2].map((c, i) => axes[0][i]*x + axes[1][i]*y - c) : axes[2].map(c => -c);
  const ray = rayInput({origin, direction});
  return Object.freeze({origin: Object.freeze(ray.origin), direction: Object.freeze(ray.direction)});
}
function sourceInfo(source, node, drawIndex) {
  if (source === undefined) return Object.freeze({node, mesh: null, primitive: drawIndex, material: null});
  fields(source, ['node','mesh','primitive','material'], 'source IDs');
  const result = {};
  for (const key of ['node','mesh','primitive','material']) {
    const value = source[key];
    if (value !== null && (!Number.isSafeInteger(value) || value < 0)) fail('SOURCE', 'Source IDs must be nonnegative integers or null');
    result[key] = value;
  }
  if (result.node !== node) fail('SOURCE', 'Source node must match the deformer');
  return Object.freeze(result);
}
const compareHits = (a, b) => a.distance - b.distance || a.drawIndex - b.drawIndex || a.faceIndex - b.faceIndex;

/** descriptors: [{deformer, indices?, texCoords?, doubleSided?, source?}].
 * Borrowed pose/deformers are never updated or disposed. They must all refer to
 * the same current pose version. Treat their output arrays as read-only.
 * maxBytes charges owned typed storage, including transient BVH build storage;
 * it does not bound borrowed geometry, JS objects, or returned hit snapshots.
 */
export function createAnimationRaycaster(pose, descriptors, {
  maxTriangles = 1048576, maxBytes = 128 * 1024 * 1024,
} = {}) {
  if (!pose || !Number.isSafeInteger(pose.nodeCount) || pose.nodeCount < 0 || !Array.isArray(descriptors) || descriptors.length > 4096) fail('SHAPE', 'Expected a pose and at most 4096 meshes');
  if (!Number.isSafeInteger(maxTriangles) || maxTriangles < 1 || maxTriangles > 1048576 || !Number.isSafeInteger(maxBytes) || maxBytes < 1) fail('LIMIT', 'Invalid raycaster budget');
  let triangles = 0, plannedBytes = 0, disposed = false, busy = false, lastQuery = null;
  const meshes = [];
  for (let drawIndex = 0; drawIndex < descriptors.length; drawIndex++) {
    const descriptor = descriptors[drawIndex];
    fields(descriptor, ['deformer','indices','texCoords','doubleSided','source'], 'mesh');
    const d = object(descriptor.deformer, 'deformer'), node = d.node, vertexCount = d.vertexCount;
    if (!Number.isSafeInteger(node) || node < 0 || node >= pose.nodeCount || !Number.isSafeInteger(vertexCount) || vertexCount < 1 || vertexCount > 16777216) fail('MESH', 'Invalid mesh node/vertex count');
    const raw = descriptor.indices ?? null, count = raw === null ? vertexCount : raw.length;
    if (!Number.isSafeInteger(count) || count < 0 || count % 3 || count / 3 > maxTriangles - triangles) fail('LIMIT', 'Invalid or excessive triangle count');
    const n = count / 3; triangles += n;
    const doubleSided = descriptor.doubleSided ?? false;
    if (typeof doubleSided !== 'boolean') fail('MESH', 'doubleSided must be boolean');
    const hasUV = descriptor.texCoords != null;
    // Median splits with eight triangles per leaf need at most this many nodes.
    let leaves = 1; while (leaves < Math.ceil(n / 8)) leaves *= 2;
    const capacity = n ? 2 * leaves - 1 : 0;
    const bytes = count * 4 + (hasUV ? vertexCount * 16 : 0) + (n ? vertexCount * 24 + capacity * 60 + n * 28 : 0);
    plannedBytes += bytes;
    if (!Number.isSafeInteger(plannedBytes) || plannedBytes > maxBytes) fail('LIMIT', 'Raycaster typed storage budget exceeded');
    const indices = new Uint32Array(count);
    if (raw !== null) array(raw, count, 'indices');
    for (let i = 0; i < count; i++) {
      const value = raw === null ? i : raw[i];
      if (!Number.isSafeInteger(value) || value < 0 || value >= vertexCount) fail('INDEX', 'Triangle index is outside the vertex array');
      indices[i] = value;
    }
    const uv = hasUV ? Float64Array.from(array(descriptor.texCoords, vertexCount * 2, 'texture coordinates'), x => finite(x, 'texture coordinate')) : null;
    meshes.push({d,node,vertexCount,indices,uv,doubleSided,source:sourceInfo(descriptor.source,node,drawIndex),drawIndex,n,capacity,
      tree:null,stamp:-1,valid:false,orientation:1});
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Raycaster has been disposed');
    if (pose.disposed || !Number.isSafeInteger(pose.version) || pose.version < 0) fail('POSE', 'Pose is disposed or invalid');
  }
  function check(mesh, version) {
    const d = mesh.d;
    if (d.disposed || d.failed) fail('MESH', 'Deformer is no longer usable');
    if (d.poseVersion !== version) fail('STALE', 'Update all mesh deformers to the current pose before querying');
    if (!Number.isSafeInteger(d.version) || d.version < 0) fail('MESH', 'Invalid deformation version');
    if (!(d.positions instanceof Float32Array)) fail('STORAGE', 'Expected Float32 deformed positions');
    array(d.positions, mesh.vertexCount * 3, 'deformed positions');
    if (!ArrayBuffer.isView(d.worldMatrix)) fail('STORAGE', 'Expected packed world matrix');
    array(d.worldMatrix, 16, 'world matrix');
  }
  function refit(mesh, stats) {
    const {d,n,capacity,vertexCount,indices} = mesh;
    if (!n || mesh.valid && mesh.stamp === d.version) return;
    mesh.valid = false;
    mesh.tree ??= {positions:new Float64Array(vertexCount*3),bounds:new Float64Array(capacity*6),
      right:new Uint32Array(capacity),start:new Uint32Array(capacity),count:new Uint32Array(capacity),order:Uint32Array.from({length:n},(_,i)=>i),used:0};
    const t = mesh.tree, m = vector(d.worldMatrix,16,'world matrix');
    if (m[3] !== 0 || m[7] !== 0 || m[11] !== 0 || m[15] !== 1) fail('MATRIX', 'World matrix must be affine');
    const determinant = m[0]*(m[5]*m[10]-m[9]*m[6])-m[4]*(m[1]*m[10]-m[9]*m[2])+m[8]*(m[1]*m[6]-m[5]*m[2]);
    if (!Number.isFinite(determinant)) fail('MATRIX', 'World determinant is not finite');
    mesh.orientation = determinant < 0 ? -1 : 1;
    for (let i=0;i<vertexCount;i++) {
      const x=finite(d.positions[i*3],'position'),y=finite(d.positions[i*3+1],'position'),z=finite(d.positions[i*3+2],'position');
      for (let a=0;a<3;a++) {
        const value=m[a]*x+m[a+4]*y+m[a+8]*z+m[a+12];
        if (!Number.isFinite(Math.fround(value))) fail('VALUE','World position exceeds the renderer coordinate range');
        t.positions[i*3+a]=value;
      }
    }
    if (!t.used) {
      const centers=new Float64Array(n*3);
      for(let f=0;f<n;f++)for(let a=0;a<3;a++)centers[f*3+a]=(t.positions[indices[f*3]*3+a]+t.positions[indices[f*3+1]*3+a]+t.positions[indices[f*3+2]*3+a])/3;
      function build(start,count) {
        const node=t.used++;t.start[node]=start;
        if(count<=8){t.count[node]=count;return node;}
        const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
        for(let i=start;i<start+count;i++)for(let a=0;a<3;a++){const v=centers[t.order[i]*3+a];lo[a]=Math.min(lo[a],v);hi[a]=Math.max(hi[a],v);}
        let axis=0;for(let a=1;a<3;a++)if(hi[a]-lo[a]>hi[axis]-lo[axis])axis=a;
        t.order.subarray(start,start+count).sort((a,b)=>centers[a*3+axis]-centers[b*3+axis]||a-b);
        const half=Math.floor(count/2);build(start,half);t.right[node]=build(start+half,count-half);return node;
      }
      build(0,n);
    }
    for(let node=t.used-1;node>=0;node--) {
      const offset=node*6;
      if(t.count[node]) {
        for(let a=0;a<3;a++){t.bounds[offset+a]=Infinity;t.bounds[offset+3+a]=-Infinity;}
        for(let i=t.start[node];i<t.start[node]+t.count[node];i++)for(let c=0;c<3;c++) {
          const vertex=indices[t.order[i]*3+c]*3;
          for(let a=0;a<3;a++){const v=t.positions[vertex+a];t.bounds[offset+a]=Math.min(t.bounds[offset+a],v);t.bounds[offset+3+a]=Math.max(t.bounds[offset+3+a],v);}
        }
      }else for(let a=0;a<3;a++) {
        t.bounds[offset+a]=Math.min(t.bounds[(node+1)*6+a],t.bounds[t.right[node]*6+a]);
        t.bounds[offset+3+a]=Math.max(t.bounds[(node+1)*6+3+a],t.bounds[t.right[node]*6+3+a]);
      }
    }
    mesh.stamp=d.version;mesh.valid=true;stats.refittedMeshes++;
  }
  function box(t,node,ray,far,stats) {
    stats.boxesTested++;
    let lo=ray.near,hi=far;
    for(let a=0;a<3;a++) {
      const min=t.bounds[node*6+a],max=t.bounds[node*6+3+a],o=ray.origin[a],d=ray.direction[a];
      if(d===0){if(o<min||o>max)return Infinity;continue;}
      let first=(min-o)/d,last=(max-o)/d;if(first>last)[first,last]=[last,first];
      // Expand slab endpoints outwards, including flat boxes/shared boundaries.
      const e=8*Number.EPSILON*Math.max(Math.abs(first),Math.abs(last));
      if(Number.isFinite(e)){first-=e;last+=e;}
      lo=Math.max(lo,first);hi=Math.min(hi,last);if(lo>hi)return Infinity;
    }
    return lo;
  }
  function triangle(mesh,face,ray,far) {
    const p=mesh.tree.positions,ids=[mesh.indices[face*3],mesh.indices[face*3+1],mesh.indices[face*3+2]],v=ids.map(i=>[p[i*3],p[i*3+1],p[i*3+2]]);
    // Dominant-axis shear edge functions, using JS binary64 arithmetic.
    // Algorithm background: https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes
    // No arbitrary determinant epsilon: tiny valid triangles remain queryable.
    const q=v.map(point=>point.map((x,a)=>x-ray.origin[a]));
    const x=q.map(a=>a[ray.kx]+ray.sx*a[ray.kz]),y=q.map(a=>a[ray.ky]+ray.sy*a[ray.kz]);
    const e=[x[1]*y[2]-y[1]*x[2],x[2]*y[0]-y[2]*x[0],x[0]*y[1]-y[0]*x[1]];
    if(e.some(a=>a<0)&&e.some(a=>a>0))return null;
    const determinant=e[0]+e[1]+e[2];if(determinant===0)return null;
    const distance=(e[0]*q[0][ray.kz]+e[1]*q[1][ray.kz]+e[2]*q[2][ray.kz])*ray.sz/determinant;
    if(!Number.isFinite(distance)||distance<ray.near||distance>far)return null;
    const a=v[1].map((x,i)=>x-v[0][i]),b=v[2].map((x,i)=>x-v[0][i]);
    const cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=Math.hypot(...cross);
    if(norm===0)return null;
    const normal=cross.map(x=>x/norm*mesh.orientation),frontFacing=normal.reduce((n,x,i)=>n+x*ray.direction[i],0)<0;
    if(!mesh.doubleSided&&!frontFacing)return null;
    if(!frontFacing)for(let i=0;i<3;i++)normal[i]=-normal[i];
    const barycentric=e.map(x=>x/determinant),point=ray.origin.map((x,i)=>x+distance*ray.direction[i]);
    const hit={drawIndex:mesh.drawIndex,node:mesh.node,source:mesh.source,faceIndex:face,distance,
      point:Object.freeze(point),normal:Object.freeze(normal),barycentric:Object.freeze(barycentric),frontFacing};
    if(mesh.uv)hit.uv=Object.freeze([0,1].map(a=>finite(barycentric.reduce((sum,w,i)=>sum+w*mesh.uv[ids[i]*2+a],0),'interpolated UV')));
    return Object.freeze(hit);
  }
  function raycast(input,options={}) {
    live();if(busy)fail('REENTRANT','Raycasting cannot be reentered');busy=true;
    const poseVersion=pose.version;
    try {
      const ray=rayInput(input);fields(options,['firstHitOnly','maxHits','drawIndices'],'query');
      const firstHitOnly=options.firstHitOnly ?? false,maxHits=options.maxHits ?? 4096;
      if(typeof firstHitOnly!=='boolean'||!Number.isSafeInteger(maxHits)||maxHits<1||maxHits>1048576)fail('OPTION','Invalid hit options');
      let selected=null;
      if(options.drawIndices!==undefined){const values=options.drawIndices;if(!Array.isArray(values)||values.length>meshes.length)fail('OPTION','Invalid draw selection');selected=new Set();for(const i of values){if(!Number.isSafeInteger(i)||i<0||i>=meshes.length||selected.has(i))fail('OPTION','Draw indices must be distinct existing meshes');selected.add(i);}}
      live();if(pose.version!==poseVersion)fail('CHANGED','Pose changed during query input capture');
      const stamps=meshes.map(mesh=>{check(mesh,poseVersion);return mesh.d.version;});
      const stats={poseVersion,meshesTested:0,boxesTested:0,trianglesTested:0,refittedMeshes:0,hitCount:0},hits=[];
      let best=null;
      for(const mesh of meshes) {
        if(selected&&!selected.has(mesh.drawIndex)||!mesh.n)continue;
        refit(mesh,stats);stats.meshesTested++;
        const t=mesh.tree,far=best?.distance ?? ray.far,entry=box(t,0,ray,far,stats);
        if(entry===Infinity)continue;
        const stack=[[0,entry]];
        while(stack.length) {
          const [node,start]=stack.pop(),limit=best?.distance ?? ray.far;
          if(start>limit)continue;
          if(t.count[node]) {
            for(let i=t.start[node];i<t.start[node]+t.count[node];i++) {
              stats.trianglesTested++;const hit=triangle(mesh,t.order[i],ray,best?.distance ?? ray.far);
              if(!hit)continue;
              if(firstHitOnly){if(!best||compareHits(hit,best)<0)best=hit;}
              else {if(hits.length===maxHits)fail('LIMIT','Hit budget exceeded; narrow the ray/draw selection or request firstHitOnly');hits.push(hit);}
            }
          }else {
            const left=node+1,right=t.right[node],a=box(t,left,ray,limit,stats),b=box(t,right,ray,limit,stats);
            if(a<b){if(b!==Infinity)stack.push([right,b]);if(a!==Infinity)stack.push([left,a]);}
            else {if(a!==Infinity)stack.push([left,a]);if(b!==Infinity)stack.push([right,b]);}
          }
        }
      }
      live();if(pose.version!==poseVersion||meshes.some((m,i)=>m.d.version!==stamps[i]||m.d.poseVersion!==poseVersion||m.d.disposed||m.d.failed))fail('CHANGED','Pose or deformation changed during raycasting');
      if(best)hits.push(best);hits.sort(compareHits);stats.hitCount=hits.length;lastQuery=Object.freeze(stats);
      return Object.freeze(hits);
    }finally{busy=false;}
  }
  return Object.freeze({raycast,get lastQuery(){return lastQuery;},get disposed(){return disposed;},
    get bufferBytes(){return meshes.reduce((sum,m)=>sum+m.indices.byteLength+(m.uv?.byteLength??0)+(m.tree?Object.values(m.tree).reduce((n,x)=>n+(ArrayBuffer.isView(x)?x.byteLength:0),0):0),0);},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during raycasting');disposed=true;meshes.length=0;lastQuery=null;},
  });
}
