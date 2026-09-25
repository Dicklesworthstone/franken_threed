/** Source-contract recorder, NOT the retained Three.js implementation. */
export const identity = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
export class Matrix4 { constructor(elements = identity()) { this.elements = [...elements]; } }
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
  constructor() { super(); Object.assign(this, {attributes: {}, index: null, morphAttributes: {}, morphTargetsRelative: false}); }
}
export class Mesh { constructor(geometry) { Object.assign(this, {geometry, matrixWorld: new Matrix4(), morphTargetInfluences: []}); } }
export class Bone { constructor(matrixWorld = new Matrix4()) { this.matrixWorld = matrixWorld; } }
export class Skeleton { constructor(bones) { this.bones = bones; this.boneInverses = bones.map(() => new Matrix4()); } }
export class SkinnedMesh extends Mesh {
  constructor(geometry) { super(geometry); Object.assign(this, {isSkinnedMesh: true, skeleton: new Skeleton([new Bone()]), bindMode: 'attached', bindMatrix: new Matrix4(), bindMatrixInverse: new Matrix4()}); }
}
export const THREE = {REVISION: '186', Matrix4, BufferGeometry, BufferAttribute, InterleavedBuffer,
  InterleavedBufferAttribute, Mesh, SkinnedMesh, Skeleton, Bone};
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
