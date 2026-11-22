/** Ordinary source deformation inputs for queued WebGL scene passes. Like the
 * PMREM scene capture, private skeletons retain bone worlds and inverse binds
 * and use the pinned Skeleton methods; no alternative skinning math is used. */
const accessor = Symbol('accessor');
const data = (object, key) => {
  for (let owner = object; owner; owner = Object.getPrototypeOf(owner)) {
    const descriptor = Object.getOwnPropertyDescriptor(owner, key);
    if (descriptor) return Object.hasOwn(descriptor, 'value') ? descriptor.value : accessor;
  }
};
const dataOnly = object => object !== null && typeof object === 'object' &&
  Object.values(Object.getOwnPropertyDescriptors(object)).every(descriptor => Object.hasOwn(descriptor, 'value'));
const ordinaryArray = (array, maximum) => {
  if (!Array.isArray(array) || Object.getPrototypeOf(array) !== Array.prototype || array.length > maximum) return false;
  const descriptors = Object.getOwnPropertyDescriptors(array);
  return Reflect.ownKeys(descriptors).every(key => typeof key === 'string' &&
    (key === 'length' || /^(0|[1-9][0-9]*)$/.test(key) && Number(key) < array.length) && Object.hasOwn(descriptors[key], 'value'));
};
const typed = new Set([Float32Array, Float64Array, Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array,
  Int32Array, Uint32Array, ...(typeof Float16Array === 'function' ? [Float16Array] : [])].map(Type => Type.prototype));
const fixedArray = array => typed.has(Object.getPrototypeOf(array ?? {})) &&
  !['buffer', 'byteOffset', 'byteLength', 'length', 'constructor', 'slice', 'subarray'].some(key => Object.hasOwn(array, key)) &&
  array.buffer instanceof ArrayBuffer && !array.buffer.resizable;
const ordinaryMatrix = (T, value) => value instanceof T.Matrix4 && dataOnly(value) &&
  ordinaryArray(data(value, 'elements'), 16) && data(value, 'elements').length === 16 &&
  data(value, 'elements').every(component => typeof component === 'number');
const componentPrototype = (T, attribute) => attribute instanceof T.InterleavedBufferAttribute ? T.InterleavedBufferAttribute.prototype :
  attribute instanceof T.Float16BufferAttribute ? T.Float16BufferAttribute.prototype : T.BufferAttribute.prototype;
const ordinaryComponents = (T, attribute) => ['getX', 'getY', 'getZ', 'getW'].every(key =>
  data(attribute, key) === componentPrototype(T, attribute)[key]);
function ordinaryAttribute(T, attribute, widths, vertexCount) {
  const interleaved = attribute instanceof T.InterleavedBufferAttribute;
  const owner = interleaved ? data(attribute, 'data') : attribute;
  if (!(attribute instanceof T.BufferAttribute) && !interleaved || !dataOnly(attribute) || !dataOnly(owner)) return false;
  const width = data(attribute, 'itemSize'), count = data(owner, 'count'), array = data(owner, 'array');
  const stride = interleaved ? data(owner, 'stride') : width, offset = interleaved ? data(attribute, 'offset') : 0;
  return widths.includes(width) && count === vertexCount && fixedArray(array) &&
    Number.isSafeInteger(stride) && stride >= width && Number.isSafeInteger(offset) && offset >= 0 && offset + width <= stride &&
    count * stride <= array.length && ordinaryComponents(T, attribute);
}

/** Descriptor-only admission: do not run an application getter, component
 * reader or Skeleton.update merely to decide whether capture is possible. */
export function isOrdinaryPassDeformation(T, object, {maxValues = 1048576} = {}) {
  const geometry = data(object, 'geometry'), skinned = data(object, 'isSkinnedMesh') === true;
  if (geometry === undefined) return !skinned;
  const attributes = data(geometry, 'attributes'), morphs = data(geometry, 'morphAttributes');
  if (!dataOnly(attributes) || !dataOnly(morphs)) return false;
  const position = data(attributes, 'position'), positionOwner = data(position, 'isInterleavedBufferAttribute') === true ? data(position, 'data') : position;
  const vertexCount = data(positionOwner, 'count');
  const morphList = data(morphs, 'position') ?? data(morphs, 'normal') ?? data(morphs, 'color');
  const morphCount = morphList === undefined ? 0 : data(morphList, 'length');
  if (morphCount) {
    const influences = data(object, 'morphTargetInfluences');
    if (data(object, 'isInstancedMesh') || !Number.isSafeInteger(vertexCount) || vertexCount < 1 ||
        !ordinaryArray(morphList, maxValues) || !ordinaryArray(influences, maxValues) || influences.length !== morphCount ||
        influences.some(value => typeof value !== 'number') ||
        Object.keys(morphs).some(name => !['position', 'normal', 'color'].includes(name))) return false;
    // Match the PMREM capture's ordinary source position-target contract.
    if (!ordinaryArray(data(morphs, 'position'), maxValues) || data(morphs, 'position').length !== morphCount) return false;
    for (const name of Object.keys(morphs)) {
      const list = data(morphs, name);
      if (!ordinaryArray(list, maxValues) || list.length !== morphCount ||
          list.some(attribute => !ordinaryAttribute(T, attribute, name === 'color' ? [3, 4] : [3], vertexCount))) return false;
    }
  }
  if (!skinned) return true;
  const skeleton = data(object, 'skeleton');
  if (Object.getPrototypeOf(object) !== T.SkinnedMesh.prototype || Object.getPrototypeOf(skeleton ?? {}) !== T.Skeleton.prototype ||
      !dataOnly(skeleton) || data(skeleton, 'update') !== T.Skeleton.prototype.update ||
      data(skeleton, 'computeBoneTexture') !== T.Skeleton.prototype.computeBoneTexture ||
      ![T.AttachedBindMode, T.DetachedBindMode].includes(data(object, 'bindMode')) ||
      !ordinaryMatrix(T, data(object, 'bindMatrix')) || !ordinaryMatrix(T, data(object, 'bindMatrixInverse')) ||
      data(object, 'getVertexPosition') !== T.SkinnedMesh.prototype.getVertexPosition ||
      data(object, 'applyBoneTransform') !== T.SkinnedMesh.prototype.applyBoneTransform ||
      !ordinaryAttribute(T, data(attributes, 'skinIndex'), [4], vertexCount) ||
      !ordinaryAttribute(T, data(attributes, 'skinWeight'), [4], vertexCount)) return false;
  const bones = data(skeleton, 'bones'), inverses = data(skeleton, 'boneInverses');
  if (!ordinaryArray(bones, maxValues) || !bones.length || !ordinaryArray(inverses, maxValues) || inverses.length !== bones.length) return false;
  for (let index = 0; index < bones.length; index++) {
    const bone = bones[index];
    if (Object.getPrototypeOf(bone ?? {}) !== T.Bone.prototype || !dataOnly(bone) ||
        !ordinaryMatrix(T, data(bone, 'matrixWorld')) || !ordinaryMatrix(T, inverses[index])) return false;
  }
  const palette = data(skeleton, 'boneMatrices'), texture = data(skeleton, 'boneTexture');
  const size = Math.max(4, Math.ceil(Math.sqrt(bones.length * 4) / 4) * 4);
  if (!fixedArray(palette) || Object.getPrototypeOf(palette) !== Float32Array.prototype ||
      palette.length < bones.length * 16 || palette.length > size * size * 4) return false;
  if (texture !== null) {
    const source = data(texture, 'source'), image = data(source, 'data');
    if (Object.getPrototypeOf(texture ?? {}) !== T.DataTexture.prototype || !dataOnly(texture) || !dataOnly(source) || !dataOnly(image) ||
        data(image, 'data') !== palette || data(image, 'width') !== size || data(image, 'height') !== size || palette.length !== size * size * 4 ||
        data(texture, 'format') !== T.RGBAFormat || data(texture, 'type') !== T.FloatType || data(texture, 'internalFormat') !== null ||
        !ordinaryArray(data(texture, 'updateRanges'), 0) || data(texture, 'onUpdate') !== null || data(source, 'dataReady') !== true ||
        data(texture, 'flipY') || data(texture, 'premultiplyAlpha') || data(texture, 'colorSpace') !== T.NoColorSpace ||
        data(texture, 'generateMipmaps') || !ordinaryArray(data(texture, 'mipmaps'), 0) || data(texture, 'unpackAlignment') !== 1 ||
        data(texture, 'minFilter') !== T.NearestFilter || data(texture, 'magFilter') !== T.NearestFilter) return false;
  }
  return true;
}

export function createThreePassSkeletons(T) {
  let states = new WeakMap();
  const active = new Set(); let closed = false;
  const fields = values => Object.fromEntries(Object.entries(values).map(([key, value]) => [key,
    {value, writable: true, configurable: true, enumerable: true}]));
  const fail = () => { throw Object.assign(new Error('F3D_RENDERER_UNSUPPORTED: A queued skeleton was released before submission'), {code: 'F3D_RENDERER_UNSUPPORTED'}); };
  const retire = state => {
    state.view.boneTexture?.dispose(); state.view.boneTexture = null;
    state.view.bones = []; state.view.boneInverses = []; state.view.boneMatrices = null;
  };
  return {
    capture(source) {
      if (closed) fail();
      let state = states.get(source);
      if (!state) {
        const view = Object.create(T.Skeleton.prototype, fields({uuid: source.uuid, bones: [], boneInverses: [], boneMatrices: null, boneTexture: null,
          computeBoneTexture() {
            if (closed || !state.references) fail();
            return T.Skeleton.prototype.computeBoneTexture.call(this);
          }}));
        state = {view, references: 0}; states.set(source, state);
      }
      const view = state.view; state.references++; active.add(state);
      const bones = source.bones.map(bone => Object.create(T.Bone.prototype, fields({matrixWorld: new T.Matrix4().copy(bone.matrixWorld)})));
      const boneInverses = source.boneInverses.map(matrix => new T.Matrix4().copy(matrix));
      const size = Math.max(4, Math.ceil(Math.sqrt(bones.length * 4) / 4) * 4);
      const palette = new Float32Array(size * size * 4); palette.set(source.boneMatrices);
      let released = false;
      return {view,
        install() {
          if (closed || released) fail();
          const texture = view.boneTexture;
          if (texture && texture.image.data.length !== palette.length) {
            texture.dispose(); view.boneTexture = null;
          } else if (texture && texture.image.data !== palette) {
            texture.image.data = palette; texture.needsUpdate = true;
          }
          view.bones = bones; view.boneInverses = boneInverses; view.boneMatrices = palette;
        },
        safe: () => data(source, 'update') === T.Skeleton.prototype.update &&
          data(source, 'computeBoneTexture') === T.Skeleton.prototype.computeBoneTexture,
        release() {
          if (released) return;
          released = true;
          if (--state.references === 0) { retire(state); active.delete(state); }
        },
      };
    },
    dispose() { if (!closed) { closed = true; for (const state of active) retire(state); active.clear(); states = new WeakMap(); } },
  };
}
