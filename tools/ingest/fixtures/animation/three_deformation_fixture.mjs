/** Source-contract recorder, NOT the retained Three.js implementation. */
export const identity = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
export class Matrix4 {
  constructor(elements = identity()) { this.elements = [...elements]; }
  copy(m) { this.elements = [...m.elements]; return this; }
  multiplyMatrices(a,b) {
    const e=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)e[c*4+r]+=a.elements[k*4+r]*b.elements[c*4+k];
    this.elements=e;return this;
  }
}
class Events {
  listeners = new Map();
  addEventListener(type, fn) { if (!this.listeners.has(type)) this.listeners.set(type, new Set()); this.listeners.get(type).add(fn); }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  dispose() { for (const fn of this.listeners.get('dispose') ?? []) fn(); }
}
export class BufferAttribute {
  constructor(array, itemSize, normalized = false) { Object.assign(this, {array, itemSize, normalized, count: array.length / itemSize, version: 0}); }
  set needsUpdate(v) { if (v) this.version++; }
  onUploadCallback() {}
  component(i, c) { const n = this.array[i * this.itemSize + c]; return this.normalized && this.array instanceof Uint8Array ? n / 255 : n; }
  getX(i) { return this.component(i, 0); } getY(i) { return this.component(i, 1); }
  getZ(i) { return this.component(i, 2); } getW(i) { return this.component(i, 3); }
}
export class InterleavedBuffer { constructor(array, stride) { Object.assign(this, {array, stride, version: 0}); } onUploadCallback() {} }
export class InterleavedBufferAttribute {
  constructor(data, itemSize, offset, normalized = false) { Object.assign(this, {data, itemSize, offset, normalized, count: data.array.length / data.stride, isInterleavedBufferAttribute: true}); }
  component(i, c) { return this.data.array[i * this.data.stride + this.offset + c]; }
  getX(i) { return this.component(i, 0); } getY(i) { return this.component(i, 1); }
  getZ(i) { return this.component(i, 2); } getW(i) { return this.component(i, 3); }
}
export class BufferGeometry extends Events {
  constructor() { super(); Object.assign(this, {attributes: {}, index: null, morphAttributes: {}, morphTargetsRelative: false, groups: [], drawRange: {start:0,count:Infinity}}); }
}
let nextId=0;
export class Object3D extends Events {
  constructor(){super();Object.assign(this,{id:nextId++,children:[],parent:null,matrixWorld:new Matrix4(),matrixWorldAutoUpdate:true,
    visible:true,layers:{test(){return true;}},renderOrder:0});}
  onBeforeRender(){} onAfterRender(){}
  add(object){this.children.push(object);object.parent=this;return this;}
  updateMatrixWorld(){for(const child of this.children)child.updateMatrixWorld();}
}
export class Mesh extends Object3D {
  constructor(geometry,material){super();Object.assign(this,{geometry,material,morphTargetInfluences:[],isMesh:true,frustumCulled:false,
    modelViewMatrix:new Matrix4(),normalMatrix:{getNormalMatrix(){}}});}
  intersectsFrustum(){return true;}
}
export class Bone extends Object3D { constructor(matrixWorld = new Matrix4()) { super();this.matrixWorld = matrixWorld; } }
export class Skeleton { constructor(bones) { this.bones = bones; this.boneInverses = bones.map(() => new Matrix4()); } }
export class SkinnedMesh extends Mesh {
  constructor(geometry) { super(geometry); Object.assign(this, {isSkinnedMesh: true, skeleton: new Skeleton([new Bone()]), bindMode: 'attached', bindMatrix: new Matrix4(), bindMatrixInverse: new Matrix4()}); }
}
export class Scene extends Object3D { constructor(){super();this.fog=null;this.environment=null;this.background=null;this.overrideMaterial=null;} }
export class Camera extends Object3D { constructor(){super();this.isPerspectiveCamera=true;this.coordinateSystem=2000;this.projectionMatrix=new Matrix4();this.matrixWorldInverse=new Matrix4();} }
export class Frustum {setFromProjectionMatrix(){return this;}}
export class Vector3 {constructor(){this.x=0;this.y=0;this.z=0;}copy(v){Object.assign(this,{x:v.x,y:v.y,z:v.z});return this;}applyMatrix4(){return this;}}
export class Material extends Events {
  constructor(){super();Object.assign(this,{id:nextId++,visible:true,allowOverride:true,side:0,depthFunc:3,depthTest:true,depthWrite:true,
    colorWrite:true,transparent:false,vertexColors:false,forceSinglePass:false,blending:1,alphaTest:0,opacity:1,
    color:{r:1,g:1,b:1},emissive:{r:0,g:0,b:0},emissiveIntensity:1,metalness:0,roughness:1,version:0});}
  onBeforeRender(){} onBeforeCompile(){} customProgramCacheKey(){}
  set needsUpdate(value){if(value)this.version++;}
}
export class MeshBasicMaterial extends Material {}
export class MeshLambertMaterial extends Material {}
export class MeshPhongMaterial extends Material {}
export class MeshToonMaterial extends Material {}
export class MeshStandardMaterial extends Material {}
export const THREE = {REVISION: '186', Matrix4, BufferGeometry, BufferAttribute, InterleavedBuffer,
  InterleavedBufferAttribute, Mesh, SkinnedMesh, Skeleton, Bone,Object3D,Scene,Camera,Frustum,Vector3,Material,
  MeshBasicMaterial,MeshLambertMaterial,MeshPhongMaterial,MeshToonMaterial,MeshStandardMaterial,
  NormalBlending:1,NoBlending:0,FrontSide:0,BackSide:1,DoubleSide:2,WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001};
export const attr = (array, width = 3) => new BufferAttribute(new Float32Array(array), width);
export function fixture({skin = false, morph = false} = {}) {
  const geometry = new BufferGeometry();
  geometry.attributes.position = attr([1,2,3, 4,5,6, 7,8,9]);
  geometry.attributes.normal = attr([0,1,0, 0,1,0, 0,1,0]);
  const mesh = skin ? new SkinnedMesh(geometry) : new Mesh(geometry);
  if (skin) {
    geometry.attributes.skinIndex = attr(Array(12).fill(0), 4);
    geometry.attributes.skinWeight = attr([1,0,0,0, 1,0,0,0, 1,0,0,0], 4);
  }
  if (morph) {
    geometry.morphTargetsRelative = true;
    geometry.morphAttributes.position = [attr([2,0,0, 2,0,0, 2,0,0])];
    mesh.morphTargetInfluences = [0.5];
  }
  return {mesh, geometry, three: THREE};
}
export function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; }
