// A source-object/GPU boundary double, not a shader or rasterization oracle.
let nextId = 1;
class Events {
  constructor() { this.id = nextId++; this.listeners = new Map(); }
  addEventListener(name, fn) { if (!this.listeners.has(name)) this.listeners.set(name, new Set()); this.listeners.get(name).add(fn); }
  removeEventListener(name, fn) { this.listeners.get(name)?.delete(fn); }
}
class Matrix4 {
  elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  multiplyMatrices() { return this; }
  copy(m) { this.elements = [...m.elements]; return this; }
  determinant() { return 1; }
}
class Matrix3 { getNormalMatrix() { return this; } }
class Vector3 { x = 0; y = 0; z = 0; copy(v) { Object.assign(this, v); return this; } applyMatrix4() { return this; } }
class Object3D extends Events {
  constructor() {
    super(); Object.assign(this, {children: [], parent: null, visible: true, layers: {test: () => true}, renderOrder: 0,
      matrixWorldAutoUpdate: false, matrixWorld: new Matrix4(), modelViewMatrix: new Matrix4(), normalMatrix: new Matrix3()});
  }
  onBeforeRender() {} onAfterRender() {}
}
class Scene extends Object3D {
  isScene = true; background = null; environment = null; fog = null; overrideMaterial = null;
}
class Camera extends Object3D {
  isPerspectiveCamera = true; coordinateSystem = 2000;
  projectionMatrix = new Matrix4(); matrixWorldInverse = new Matrix4();
}
class BufferGeometry extends Events {
  attributes = {}; groups = []; morphAttributes = {}; index = null;
  boundingSphere = {center: new Vector3()}; drawRange = {start: 0, count: Infinity};
}
class Material extends Events {
  type = 'ShaderMaterial'; isShaderMaterial = true; side = 0; transparent = false; visible = true; allowOverride = true;
  uniforms = {value: {value: 0}};
  onBeforeCompile() {} onBeforeRender() {} customProgramCacheKey() {}
}
class Mesh extends Object3D {
  isMesh = true; frustumCulled = false; castShadow = false; receiveShadow = false;
  constructor(geometry = new BufferGeometry(), material = new Material()) { super(); this.geometry = geometry; this.material = material; }
  intersectsFrustum() { return true; }
}
export const T = {REVISION: '186', Matrix4, Matrix3, Vector3, Object3D, Scene, Camera, Mesh, BufferGeometry,
  Texture: class extends Events { version = 0; source = {version: 0}; }, Material,
  MeshBasicMaterial: class extends Material {}, MeshLambertMaterial: class extends Material {},
  MeshPhongMaterial: class extends Material {}, MeshToonMaterial: class extends Material {}, MeshStandardMaterial: class extends Material {},
  Frustum: class { setFromProjectionMatrix() {} }, WebGLCoordinateSystem: 2000, WebGPUCoordinateSystem: 2001,
  NoToneMapping: 0, DoubleSide: 2, FrontSide: 0, BackSide: 1};
export function createGpuBufferGeometry() { throw Error('unexpected core geometry path'); }
export function bufferGeometrySnapshot() { throw Error('unexpected core geometry path'); }
export function createGpuInstanceAttributes() { throw Error('unexpected instance path'); }
export function instanceAttributesSnapshot() { throw Error('unexpected instance path'); }
export function inspectInstanceAttributes() { throw Error('unexpected instance path'); }
export function createGpuThreeTextures() { throw Error('unexpected texture upload: source textures are borrowed'); }
export const hasThreeDeformation = () => false;
export function inspectThreeDeformation() { throw Error('unexpected deformation'); }
export function createGpuThreeDeformation() { throw Error('unexpected deformation'); }
export function updateGpuThreeDeformations(list) { if (list.length) throw Error('unexpected deformation'); }
export async function createGpuAnimationRenderer(device, options) {
  return {allocatedBytes: 0, failed: false, whenIdle: async () => {}, dispose() {},
    async addMesh(geometry, description) {
      const mesh = {geometry, description, disposed: false, dispose() { this.disposed = true; }};
      device.registrations.push(mesh); return mesh;
    },
    render(frame) { device.frames.push({options, frame}); },
  };
}
export function fixture() {
  const device = {frames: [], registrations: [], geometries: []};
  const scene = new T.Scene(), mesh = new T.Mesh(), camera = new T.Camera(); scene.children.push(mesh);
  const texture = new T.Texture(), binding = {view: {}, sampler: {}, version: 0, sourceVersion: 0};
  const textures = new Map([[texture, binding]]), sizes = [];
  const program = {state: () => ({toneMapping: 0}), setLights() {}, setLightsView() {}, needsLights: () => false,
    shaderLibMaterial: () => true, clippingState: () => ({numPlanes: 0, numIntersection: 0}), raster: () => ({}),
    uniformsFor: m => m.uniforms, refresh: m => m.uniforms, bindsClippingPlanes: () => false,
    compile(m) {
      return {key: 'test-program', attributesKey: 'position', program: {reflection: {uniformBufferSize: 16, attributes: [],
        textures: m.uniforms.map ? [{name: 'map', element: null, dimension: '2d'}] : []}}};
    },
    createGeometry() {
      const g = {disposed: false, bufferBytes: 36, update() {}, whenIdle: async () => {}, dispose() { this.disposed = true; }};
      device.geometries.push(g); return g;
    },
    geometrySnapshot: () => ({signature: 'position'}),
    pack(reflection, uniforms, object, camera, bytes, options) {
      sizes.push(options.targetSize); new DataView(bytes.buffer).setFloat32(0, uniforms.value.value, true);
      return uniforms.map ? [uniforms.map.value] : [];
    },
  };
  return {device, scene, mesh, camera, textures, texture, binding, program, sizes};
}
