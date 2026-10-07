/** Minimal source-data objects for snapshot tests, not a Three.js oracle. */
class Events {
  addEventListener(name, fn) { (this._listeners ??= new Map()); if (!this._listeners.has(name)) this._listeners.set(name, new Set()); this._listeners.get(name).add(fn); }
  removeEventListener(name, fn) { this._listeners?.get(name)?.delete(fn); }
  dispose() { for (const fn of this._listeners?.get('dispose') ?? []) fn(); }
}
class Matrix4 { constructor() { this.elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; } }
class Vector3 { constructor(x = 0, y = 0, z = 0) { Object.assign(this, {x, y, z}); } }
class Vector2 { constructor(x = 0, y = 0) { Object.assign(this, {x, y}); } }
class Layers { mask = 1; }
let id = 0;
class Object3D extends Events {
  constructor() {
    super(); Object.defineProperty(this, 'id', {value: id++});
    Object.assign(this, {isObject3D: true, name: '', children: [], parent: null, visible: true, layers: new Layers(),
      matrix: new Matrix4(), matrixWorld: new Matrix4(), modelViewMatrix: new Matrix4(), normalMatrix: new Matrix4(),
      position: new Vector3(), matrixAutoUpdate: true, matrixWorldAutoUpdate: true, renderOrder: 0,
      frustumCulled: true, castShadow: false, receiveShadow: false, updates: 0});
  }
  updateMatrix() { this.matrix.elements[12] = this.position.x; }
  updateMatrixWorld() { this.updates++; if (this.matrixAutoUpdate) this.updateMatrix(); this.matrixWorld.elements = this.matrix.elements.slice(); }
  onBeforeRender() {}
  onAfterRender() {}
  traverse(fn) { fn(this); this.children.forEach(c => c.traverse(fn)); }
}
class Camera extends Object3D {
  constructor() { super(); this.isOrthographicCamera = true; this.projectionMatrix = new Matrix4(); this.matrixWorldInverse = new Matrix4(); }
  updateMatrixWorld() { super.updateMatrixWorld(); this.matrixWorldInverse.elements[12] = -this.position.x; }
}
class Material extends Events {
  constructor() { super(); this.id = id++; this.name = ''; this.opacity = 1; this.transparent = false; this.side = 0; this.defines = {}; this.version = 0; }
  onBeforeRender() {}
  onBeforeCompile() {}
  customProgramCacheKey() { return ''; }
}
class ShaderMaterial extends Material {
  constructor() { super(); this.isShaderMaterial = true; this.uniforms = {}; this.vertexShader = 'vertex'; this.fragmentShader = 'fragment'; }
}
class RawShaderMaterial extends ShaderMaterial { isRawShaderMaterial = true; }
class BufferAttribute {
  constructor(array, itemSize) { Object.assign(this, {array, itemSize, count: array.length / itemSize, normalized: false, version: 0, updateRanges: []}); }
  onUploadCallback() {}
}
class InterleavedBuffer extends BufferAttribute { constructor(array, stride) { super(array, stride); this.stride = stride; this.isInterleavedBuffer = true; } }
class InterleavedBufferAttribute {
  constructor(data, itemSize, offset) { Object.assign(this, {data, itemSize, offset, count: data.count, isInterleavedBufferAttribute: true}); }
}
class BufferGeometry extends Events {
  constructor() { super(); this.id = id++; this.isBufferGeometry = true; this.attributes = {}; this.index = null; this.morphAttributes = {}; this.groups = []; this.drawRange = {start: 0, count: Infinity}; }
}
class Mesh extends Object3D { constructor(geometry, material) { super(); this.isMesh = true; Object.assign(this, {geometry, material}); } }
class Texture extends Events {
  constructor() { super(); this.isTexture = true; this.version = 0; this.source = {version: 0}; this.matrixAutoUpdate = true; this.matrix = new Matrix4(); this.offset = new Vector2(); this.repeat = new Vector2(1, 1); this.center = new Vector2(); this.rotation = 0; this.minFilter = 1006; }
}
export const passThree = {REVISION: '186', Object3D, Camera, Mesh, Material, ShaderMaterial, RawShaderMaterial, BufferAttribute,
  BufferGeometry, InterleavedBuffer, InterleavedBufferAttribute, Matrix4, Vector3, Vector2, Layers, Texture};
export function passFixture() {
  const geometry = new BufferGeometry(), material = new ShaderMaterial();
  geometry.attributes.position = new BufferAttribute(new Float32Array([-1, 3, 0, -1, -1, 0, 3, -1, 0]), 3);
  geometry.attributes.uv = new BufferAttribute(new Float32Array([0, 2, 0, 0, 2, 0]), 2);
  material.uniforms = {factor: {value: 1}, direction: {value: new Vector2(1, 0)}};
  const root = new Mesh(geometry, material); root.name = 'fullscreen';
  return {root, geometry, material, camera: new Camera()};
}
