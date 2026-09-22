/**
 * @file tools/compat/mesh_adapter.mjs
 * Explicit experimental dynamic Mesh adapter for FrankenThreeD (f3d-05.6 / 19:26Z product wave).
 *
 * Extracts dynamic input from real pinned Three.js Mesh + Camera instances into
 * isolated typed arrays for the Wasm/WebGPU execution core:
 * - BufferGeometry position (itemSize 3, unnormalized, BufferAttribute or InterleavedBufferAttribute) + optional index + drawRange
 * - Opaque untextured MeshBasicMaterial color (Float32Array[4]), with material-side culling
 * - Dynamic depth state: depthTest, depthWrite, and depthFunc mapped to wire compare codes 1..8
 * - f64 model-view and projection matrices with WebGL-to-WebGPU depth coordinate tracking
 *
 * Invariants:
 * - Keeps JavaScript ownership of Three.js scene objects.
 * - Does not silently ignore unsupported features; explicitly refuses them.
 * - Takes a fresh typed-array snapshot per call; zero per-frame JSON.
 * - Preserves source matrix-update boundaries.
 * - Retains unsupported full application route without automatic rerouting.
 */

import {
  AlwaysDepth,
  EqualDepth,
  Frustum,
  GreaterDepth,
  GreaterEqualDepth,
  LessDepth,
  LessEqualDepth,
  Material,
  Matrix4,
  Mesh,
  NeverDepth,
  NotEqualDepth,
  Object3D,
  Vector4,
} from "../../upstream/three.js/build/three.module.js";

// Base prototypes default hooks for override detection (avoiding function.toString)
const DEFAULT_MATERIAL_ON_BEFORE_COMPILE = Material.prototype.onBeforeCompile;
const DEFAULT_MATERIAL_CUSTOM_PROGRAM_CACHE_KEY = Material.prototype.customProgramCacheKey;
const DEFAULT_MATERIAL_ON_BEFORE_RENDER = Material.prototype.onBeforeRender;

const DEFAULT_OBJECT3D_ON_BEFORE_RENDER = Object3D.prototype.onBeforeRender;
const DEFAULT_OBJECT3D_ON_AFTER_RENDER = Object3D.prototype.onAfterRender;
const DEFAULT_MESH_INTERSECTS_FRUSTUM = Mesh.prototype.intersectsFrustum;

export const ADMISSION_REJECTION = Object.freeze({
  NOT_A_MESH: "NOT_A_MESH: Object is not an instance of THREE.Mesh",
  UNSUPPORTED_MESH_SUBCLASS:
    "UNSUPPORTED_MESH_SUBCLASS: InstancedMesh, SkinnedMesh, and BatchedMesh are not supported in this scalar slice",
  NOT_VISIBLE: "NOT_VISIBLE: Mesh is not visible (mesh.visible === false)",
  MATERIAL_NOT_VISIBLE:
    "MATERIAL_NOT_VISIBLE: Material is not visible (material.visible === false)",
  UNSUPPORTED_CALLBACK:
    "UNSUPPORTED_CALLBACK: onBeforeRender and onAfterRender callbacks on mesh or material are not supported in this slice",
  LAYER_MISMATCH:
    "LAYER_MISMATCH: Camera layers do not intersect mesh layers (camera.layers.test(mesh.layers) === false)",
  INVALID_GEOMETRY: "INVALID_GEOMETRY: Mesh geometry must be an instance of THREE.BufferGeometry",
  MISSING_POSITION:
    'MISSING_POSITION: BufferGeometry must have a "position" attribute with itemSize === 3',
  UNSUPPORTED_ATTRIBUTE:
    "UNSUPPORTED_ATTRIBUTE: Interleaved or normalized vertex attributes are not supported in this slice",
  UNSUPPORTED_GEOMETRY:
    "UNSUPPORTED_GEOMETRY: Morph targets and multiple geometry groups are not supported in this slice",
  INVALID_MATERIAL: "INVALID_MATERIAL: Material must be an instance of THREE.MeshBasicMaterial",
  UNSUPPORTED_MATERIAL:
    "UNSUPPORTED_MATERIAL: Textured maps, transparency, wireframe, or custom blending are not supported in this slice",
  UNSUPPORTED_MATERIAL_FEATURE:
    "UNSUPPORTED_MATERIAL_FEATURE: clippingPlanes, alphaTest/alphaHash, or custom shader hooks are not supported in this slice",
  INCOMPATIBLE_COLOR_WRITE:
    "INCOMPATIBLE_COLOR_WRITE: Meshes with colorWrite=false require Wasm export f3d_build_mesh_batch_cull_depth_color_packet",
  INCOMPATIBLE_VERTEX_COLORS:
    "INCOMPATIBLE_VERTEX_COLORS: Meshes with vertexColors require Wasm export f3d_build_mesh_batch_vertex_color_packet",
  INCOMPATIBLE_BACKGROUND_CLEAR:
    "INCOMPATIBLE_BACKGROUND_CLEAR: scene.background Color requires Wasm export f3d_build_mesh_batch_vertex_color_clear_packet",
  MISSING_COLOR_ATTRIBUTE:
    "MISSING_COLOR_ATTRIBUTE: Meshes with material.vertexColors=true require geometry.attributes.color attribute (WebGL/WebGPU discrepancy)",
  INVALID_COLOR_ATTRIBUTE:
    "INVALID_COLOR_ATTRIBUTE: BufferGeometry color attribute must have itemSize 3 or 4 and match vertex count",
  UNSUPPORTED_DEPTH:
    "UNSUPPORTED_DEPTH: Material must have depthTest === false and depthWrite === false in this slice (pipeline has no depth buffer)",
  UNSUPPORTED_STENCIL:
    "UNSUPPORTED_STENCIL: Stencil operations are not supported in this slice (material.stencilWrite === true)",
  UNSUPPORTED_POLYGON_OFFSET:
    "UNSUPPORTED_POLYGON_OFFSET: Polygon offset is not supported in this slice (material.polygonOffset === true)",
  UNSUPPORTED_REVERSED_DEPTH:
    "UNSUPPORTED_REVERSED_DEPTH: Reversed depth buffer is not supported in this slice",
  INVALID_DEPTH_FUNC: "INVALID_DEPTH_FUNC: Invalid or unsupported depthFunc",
  AMBIGUOUS_DEPTH_PAIR:
    'AMBIGUOUS_DEPTH_PAIR: Material with depthTest=false and depthWrite=true is ambiguous across backends (WebGL suppresses writes, WebGPU permits writes); provide options.sourceBackend ("webgl" | "webgpu")',
  UNSUPPORTED_SIDE:
    "UNSUPPORTED_SIDE: Material side must be FrontSide (0), BackSide (1), or DoubleSide (2)",
  MISSING_CULL_EXPORT:
    "MISSING_CULL_EXPORT: FrontSide or BackSide mesh rendering requires Wasm export f3d_build_mesh_batch_cull_packet; silent DoubleSide fallback is strictly forbidden",
  INVALID_CULL_MODE: "INVALID_CULL_MODE: Invalid or unknown cull mode wire value",
  INVALID_FRONT_FACE: "INVALID_FRONT_FACE: Invalid or unknown front face wire value",
  INVALID_CAMERA:
    "INVALID_CAMERA: Camera must be an instance of THREE.Camera with valid projectionMatrix and matrixWorldInverse",
  INVALID_DIMENSIONS: "INVALID_DIMENSIONS: Viewport dimensions must be positive integers",
  INDEX_OUT_OF_BOUNDS: "INDEX_OUT_OF_BOUNDS: Index references vertex out of bounds",
  INVALID_DRAWRANGE:
    "INVALID_DRAWRANGE: Invalid drawRange: start and count must be non-negative integers",
  EMPTY_MESH_BATCH: "EMPTY_MESH_BATCH: Mesh batch must be a non-empty array of meshes",
  INCOMPATIBLE_BATCH_DEPTH:
    "INCOMPATIBLE_BATCH_DEPTH: Meshes in batch have incompatible depth settings for the available legacy Wasm exports; mixed settings require f3d_build_mesh_batch_cull_depth_packet",
  UNSUPPORTED_RENDERABLE:
    "UNSUPPORTED_RENDERABLE: Non-mesh renderable objects (Line, Points, Sprite, Light) are not supported in this slice",
  UNSUPPORTED_SCENE_FEATURE:
    "UNSUPPORTED_SCENE_FEATURE: Scene-level features (background, fog, overrideMaterial, environment) are not supported in this slice",
  UNSUPPORTED_UPLOAD_CALLBACK:
    "UNSUPPORTED_UPLOAD_CALLBACK: Custom onUploadCallback on BufferAttribute is not supported in this slice",
  INVALID_ATTRIBUTE_RESIZE:
    "INVALID_ATTRIBUTE_RESIZE: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.",
  AMBIGUOUS_ATTRIBUTE_UPDATE:
    'AMBIGUOUS_ATTRIBUTE_UPDATE: Attribute usage or multiple updateRanges behavior is ambiguous across backends; provide options.sourceBackend ("webgl" | "webgpu")',
  INVALID_UPDATE_RANGE:
    "INVALID_UPDATE_RANGE: Invalid updateRange: start must be non-negative integer, count must be positive integer, and start + count must not exceed attribute array length",
});

// Supported upstream source backends for resolving backend-specific semantics
export const SOURCE_BACKEND = Object.freeze({
  WEBGL: "webgl",
  WEBGPU: "webgpu",
});

// Pinned Three.js coordinate system constants
export const COORDINATE_SYSTEM = Object.freeze({
  WEBGL: 2000,
  WEBGPU: 2001,
});

// Pinned Three.js material side constants matching upstream FrontSide, BackSide, DoubleSide
export const THREE_SIDE = Object.freeze({
  FRONT_SIDE: 0,
  BACK_SIDE: 1,
  DOUBLE_SIDE: 2,
});

// Wire cull mode codes matching WebGPU GPUCullMode ("none", "front", "back")
export const CULL_MODE_WIRE = Object.freeze({
  NONE: 0,
  FRONT: 1,
  BACK: 2,
});

// Wire front face codes matching WebGPU GPUFrontFace ("ccw", "cw")
export const FRONT_FACE_WIRE = Object.freeze({
  CCW: 0,
  CW: 1,
});

// Wire compare codes matching crates/f3d-runtime/src/gpu_host.rs and bridge_runtime.js
export const DEPTH_WIRE_COMPARE = Object.freeze({
  NEVER: 1, // "never"
  LESS: 2, // "less"
  EQUAL: 3, // "equal"
  LESS_EQUAL: 4, // "less-equal"
  GREATER: 5, // "greater"
  NOT_EQUAL: 6, // "not-equal"
  GREATER_EQUAL: 7, // "greater-equal"
  ALWAYS: 8, // "always"
});

// Pinned Three.js depth constants (0..7) to wire compare codes (1..8)
export const THREE_DEPTH_FUNC_TO_WIRE_COMPARE = Object.freeze({
  [NeverDepth]: DEPTH_WIRE_COMPARE.NEVER, // 0 -> 1
  [AlwaysDepth]: DEPTH_WIRE_COMPARE.ALWAYS, // 1 -> 8
  [LessDepth]: DEPTH_WIRE_COMPARE.LESS, // 2 -> 2
  [LessEqualDepth]: DEPTH_WIRE_COMPARE.LESS_EQUAL, // 3 -> 4
  [EqualDepth]: DEPTH_WIRE_COMPARE.EQUAL, // 4 -> 3
  [GreaterEqualDepth]: DEPTH_WIRE_COMPARE.GREATER_EQUAL, // 5 -> 7
  [GreaterDepth]: DEPTH_WIRE_COMPARE.GREATER, // 6 -> 5
  [NotEqualDepth]: DEPTH_WIRE_COMPARE.NOT_EQUAL, // 7 -> 6
});

/**
 * Computes the 3x3 determinant of the upper-left affine transform block of a 4x4 matrix.
 * Matches Three.js Matrix4.determinantAffine().
 *
 * @param {any} matrixWorld - Three.js Matrix4 or object with elements array
 * @returns {number}
 */
export function computeAffineDeterminant(matrixWorld) {
  if (typeof matrixWorld?.determinantAffine === "function") {
    return matrixWorld.determinantAffine();
  }
  const te = matrixWorld?.elements;
  if (!te || te.length < 16) return 1.0;
  const n11 = te[0],
    n12 = te[4],
    n13 = te[8];
  const n21 = te[1],
    n22 = te[5],
    n23 = te[9];
  const n31 = te[2],
    n32 = te[6],
    n33 = te[10];
  return (
    n11 * (n22 * n33 - n23 * n32) - n12 * (n21 * n33 - n23 * n31) + n13 * (n21 * n32 - n22 * n31)
  );
}

/**
 * Creates an Error representing an admission or batch rejection.
 * Ensures the error message carries the stable code prefix (e.g. 'KEY: ...') and sets err.reason = key.
 *
 * @param {string} code - Stable error key (e.g. 'EMPTY_MESH_BATCH', 'INCOMPATIBLE_BATCH_DEPTH')
 * @param {string} [detail] - Additional contextual detail to append
 * @returns {Error}
 */
export function createAdmissionError(code, detail = "") {
  const reasonText = ADMISSION_REJECTION[code] ?? "Admission rejected";
  const prefix = `${code}: `;
  const baseMessage = reasonText.startsWith(prefix) ? reasonText : `${prefix}${reasonText}`;
  const message = detail ? `${baseMessage}: ${detail}` : baseMessage;
  const err = new Error(message);
  err.reason = code;
  return err;
}

function rejectMesh(code) {
  return {
    admitted: false,
    reason: ADMISSION_REJECTION[code],
    reasonCode: code,
    code,
  };
}

function createRefusalItem(uuid, code, reason) {
  const item = {
    uuid,
    reason: reason || ADMISSION_REJECTION[code] || code,
  };
  Object.defineProperty(item, "code", {
    value: code,
    enumerable: false,
    writable: true,
    configurable: true,
  });
  return item;
}

/**
 * Resolves the effective groupOrder for a scene object per Three.js Renderer.js:3252 / WebGLRenderer.js:1870.
 * Only groups visited in the camera's layers contribute their renderOrder.
 *
 * @param {any} object
 * @returns {number}
 */
function getMeshGroupOrder(object, camera, root) {
  let cur = object?.parent;
  while (cur) {
    if (cur.isGroup && camera.layers.test(cur.layers)) {
      return Number.isFinite(cur.renderOrder) ? cur.renderOrder : 0;
    }
    if (cur === root) break;
    cur = cur.parent;
  }
  return 0;
}

/**
 * Three.js WebGPU opaque sort comparator matching common/RenderList.js:14-34.
 * Order: groupOrder -> renderOrder -> projected z -> id
 *
 * @param {object} a
 * @param {object} b
 * @returns {number}
 */
function painterSortWebGPU(a, b) {
  if (a.groupOrder !== b.groupOrder) {
    return a.groupOrder - b.groupOrder;
  } else if (a.renderOrder !== b.renderOrder) {
    return a.renderOrder - b.renderOrder;
  } else if (a.z !== b.z) {
    return a.z - b.z;
  } else {
    return a.id - b.id;
  }
}

/**
 * Three.js WebGL opaque sort comparator matching webgl/WebGLRenderLists.js:1-29.
 * Order: groupOrder -> renderOrder -> material.id -> materialVariant -> projected z -> id
 *
 * @param {object} a
 * @param {object} b
 * @returns {number}
 */
function painterSortWebGL(a, b) {
  if (a.groupOrder !== b.groupOrder) {
    return a.groupOrder - b.groupOrder;
  } else if (a.renderOrder !== b.renderOrder) {
    return a.renderOrder - b.renderOrder;
  } else if (a.materialId !== b.materialId) {
    return a.materialId - b.materialId;
  } else if (a.materialVariant !== b.materialVariant) {
    return a.materialVariant - b.materialVariant;
  } else if (a.z !== b.z) {
    return a.z - b.z;
  } else {
    return a.id - b.id;
  }
}

function hasCustomUploadCallback(attr) {
  if (!attr) return false;
  if (Object.hasOwn(attr, "onUploadCallback")) return true;
  if (typeof attr.onUploadCallback === "function") {
    const proto = Object.getPrototypeOf(attr);
    if (!proto || attr.onUploadCallback !== proto.onUploadCallback) {
      return true;
    }
  }
  if (attr.isInterleavedBufferAttribute && attr.data) {
    return hasCustomUploadCallback(attr.data);
  }
  return false;
}

/**
 * Checks if a Three.js Mesh and Camera can be admitted into the dynamic Wasm mesh rendering slice.
 * @param {any} mesh
 * @param {any} camera
 * @param {object} [options]
 * @returns {{ admitted: boolean, reason?: string, reasonCode?: string, code?: string }}
 */
export function canAdmitMesh(mesh, camera, options = {}) {
  if (!mesh || !mesh.isMesh) {
    return rejectMesh("NOT_A_MESH");
  }

  // Explicitly decline subclasses that cannot silently render scalar (root 19:47Z)
  if (mesh.isInstancedMesh || mesh.isSkinnedMesh || mesh.isBatchedMesh) {
    return rejectMesh("UNSUPPORTED_MESH_SUBCLASS");
  }

  // Check mesh and ancestor visibility (Three.js Renderer.js:3244 subtree culling)
  let cur = mesh;
  while (cur) {
    if (cur.visible === false) {
      return rejectMesh("NOT_VISIBLE");
    }
    cur = cur.parent;
  }

  // Reject unsupported object callbacks (root 19:47Z point 2)
  if (
    (mesh.onBeforeRender && mesh.onBeforeRender !== DEFAULT_OBJECT3D_ON_BEFORE_RENDER) ||
    Object.hasOwn(mesh, "onBeforeRender") ||
    (mesh.onAfterRender && mesh.onAfterRender !== DEFAULT_OBJECT3D_ON_AFTER_RENDER) ||
    Object.hasOwn(mesh, "onAfterRender")
  ) {
    return rejectMesh("UNSUPPORTED_CALLBACK");
  }

  // Camera check
  if (!camera || !camera.isCamera || !camera.projectionMatrix || !camera.matrixWorldInverse) {
    return rejectMesh("INVALID_CAMERA");
  }

  // Reject reversed depth buffer configurations (root review invariant)
  if (
    camera.reversedDepth === true ||
    camera.reversedDepthBuffer === true ||
    camera._reversedDepth === true
  ) {
    return rejectMesh("UNSUPPORTED_REVERSED_DEPTH");
  }

  // Layer intersection check (root 19:47Z)
  if (camera.layers && mesh.layers && !camera.layers.test(mesh.layers)) {
    return rejectMesh("LAYER_MISMATCH");
  }

  const geometry = mesh.geometry;
  if (!geometry || !geometry.isBufferGeometry) {
    return rejectMesh("INVALID_GEOMETRY");
  }

  const posAttr = geometry.attributes?.position;
  if (!posAttr || posAttr.itemSize !== 3) {
    return rejectMesh("MISSING_POSITION");
  }

  // Reject instanced, normalized, or invalid interleaved position attributes explicitly
  if (
    posAttr.isInstancedBufferAttribute ||
    posAttr.data?.isInstancedInterleavedBuffer ||
    posAttr.normalized
  ) {
    return rejectMesh("UNSUPPORTED_ATTRIBUTE");
  }
  if (posAttr.isInterleavedBufferAttribute) {
    if (!posAttr.data || !(posAttr.data.array instanceof Float32Array)) {
      return rejectMesh("UNSUPPORTED_ATTRIBUTE");
    }
  }

  // Reject interleaved index buffers explicitly
  if (geometry.index?.isInterleavedBufferAttribute) {
    return rejectMesh("UNSUPPORTED_ATTRIBUTE");
  }

  // Reject custom upload callbacks on BufferAttribute or InterleavedBuffer (root 18462)
  if (hasCustomUploadCallback(posAttr) || hasCustomUploadCallback(geometry.index)) {
    return rejectMesh("UNSUPPORTED_UPLOAD_CALLBACK");
  }

  if (geometry.morphAttributes && Object.keys(geometry.morphAttributes).length > 0) {
    return rejectMesh("UNSUPPORTED_GEOMETRY");
  }
  if (geometry.groups && geometry.groups.length > 1) {
    return rejectMesh("UNSUPPORTED_GEOMETRY");
  }

  const material = mesh.material;
  if (!material || Array.isArray(material) || !material.isMeshBasicMaterial) {
    return rejectMesh("INVALID_MATERIAL");
  }

  // Material visibility check (root 19:47Z point 1)
  if (material.visible === false) {
    return rejectMesh("MATERIAL_NOT_VISIBLE");
  }

  // Reject unsupported material callbacks (root 19:47Z point 2)
  if (
    (material.onBeforeRender && material.onBeforeRender !== DEFAULT_MATERIAL_ON_BEFORE_RENDER) ||
    Object.hasOwn(material, "onBeforeRender") ||
    material.onAfterRender ||
    Object.hasOwn(material, "onAfterRender")
  ) {
    return rejectMesh("UNSUPPORTED_CALLBACK");
  }

  if (material.transparent === true || (material.opacity !== undefined && material.opacity < 1.0)) {
    return rejectMesh("UNSUPPORTED_MATERIAL");
  }
  if (material.map || material.envMap || material.alphaMap || material.lightMap || material.aoMap) {
    return rejectMesh("UNSUPPORTED_MATERIAL");
  }
  if (material.wireframe === true) {
    return rejectMesh("UNSUPPORTED_MATERIAL");
  }

  // Reject stencil operations
  if (material.stencilWrite === true) {
    return rejectMesh("UNSUPPORTED_STENCIL");
  }

  // Reject polygon offset
  if (material.polygonOffset === true) {
    return rejectMesh("UNSUPPORTED_POLYGON_OFFSET");
  }

  // Validate depth function if specified
  const rawDepthFunc = material.depthFunc;
  if (rawDepthFunc !== undefined && THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc] === undefined) {
    return rejectMesh("INVALID_DEPTH_FUNC");
  }

  // Handle depthTest=false and depthWrite=true ambiguity across backends (root review invariant)
  const depthTest = material.depthTest !== false;
  const depthWrite = material.depthWrite !== false;
  if (!depthTest && depthWrite) {
    const backend = options?.sourceBackend?.toLowerCase();
    if (backend !== SOURCE_BACKEND.WEBGL && backend !== SOURCE_BACKEND.WEBGPU) {
      return rejectMesh("AMBIGUOUS_DEPTH_PAIR");
    }
  }

  // Handle attribute usage / updateRanges ambiguity across backends (root review 18405 / 18499)
  let colorAttr = null;
  if (material.vertexColors === true) {
    colorAttr = geometry.attributes?.color;
    if (!colorAttr) {
      return rejectMesh("MISSING_COLOR_ATTRIBUTE");
    }
    if (colorAttr.isInstancedBufferAttribute || colorAttr.data?.isInstancedInterleavedBuffer) {
      return rejectMesh("UNSUPPORTED_ATTRIBUTE");
    }
    if (colorAttr.isInterleavedBufferAttribute && (!colorAttr.data || !colorAttr.data.array)) {
      return rejectMesh("UNSUPPORTED_ATTRIBUTE");
    }
    if (hasCustomUploadCallback(colorAttr)) {
      return rejectMesh("UNSUPPORTED_UPLOAD_CALLBACK");
    }
    if (colorAttr.itemSize !== 3 && colorAttr.itemSize !== 4) {
      return rejectMesh("INVALID_COLOR_ATTRIBUTE");
    }
    const colorArray = colorAttr.array;
    if (
      !colorArray ||
      (!(colorArray instanceof Float32Array) &&
        !(colorArray instanceof Uint8Array) &&
        !(colorArray instanceof Uint8ClampedArray))
    ) {
      return rejectMesh("INVALID_COLOR_ATTRIBUTE");
    }
    if (
      (colorArray instanceof Uint8Array || colorArray instanceof Uint8ClampedArray) &&
      !colorAttr.normalized
    ) {
      return rejectMesh("INVALID_COLOR_ATTRIBUTE");
    }
    if (colorAttr.count < posAttr.count) {
      return rejectMesh("INVALID_COLOR_ATTRIBUTE");
    }
  }

  const posTarget = posAttr.isInterleavedBufferAttribute ? posAttr.data : posAttr;
  const colorTarget = colorAttr?.isInterleavedBufferAttribute ? colorAttr.data : colorAttr;
  const posAttrUsage = posTarget?.usage;
  const indexAttrUsage = geometry.index?.usage;
  const colorAttrUsage = colorTarget?.usage;
  const hasDynamicUsage =
    posAttrUsage === 35048 || indexAttrUsage === 35048 || colorAttrUsage === 35048;
  const hasMultiRanges =
    (posTarget?.updateRanges && posTarget.updateRanges.length > 1) ||
    (geometry.index?.updateRanges && geometry.index.updateRanges.length > 1) ||
    (colorTarget?.updateRanges && colorTarget.updateRanges.length > 1);
  if (hasDynamicUsage || hasMultiRanges) {
    const backend = options?.sourceBackend?.toLowerCase();
    if (backend !== SOURCE_BACKEND.WEBGL && backend !== SOURCE_BACKEND.WEBGPU) {
      return rejectMesh("AMBIGUOUS_ATTRIBUTE_UPDATE");
    }
  }

  // Side admission: FrontSide (0), BackSide (1), and DoubleSide (2) are supported
  const side = material.side ?? THREE_SIDE.FRONT_SIDE;
  if (
    side !== THREE_SIDE.FRONT_SIDE &&
    side !== THREE_SIDE.BACK_SIDE &&
    side !== THREE_SIDE.DOUBLE_SIDE
  ) {
    return rejectMesh("UNSUPPORTED_SIDE");
  }

  // Strict check on advanced material features
  if (material.clippingPlanes && material.clippingPlanes.length > 0) {
    return rejectMesh("UNSUPPORTED_MATERIAL_FEATURE");
  }
  if ((material.alphaTest && material.alphaTest > 0) || material.alphaHash === true) {
    return rejectMesh("UNSUPPORTED_MATERIAL_FEATURE");
  }
  if (material.blending !== undefined && material.blending !== 1) {
    // 1 = NormalBlending
    return rejectMesh("UNSUPPORTED_MATERIAL_FEATURE");
  }

  // Check for custom shader hooks by comparing against base Material prototype methods
  if (
    (material.onBeforeCompile && material.onBeforeCompile !== DEFAULT_MATERIAL_ON_BEFORE_COMPILE) ||
    Object.hasOwn(material, "onBeforeCompile")
  ) {
    return rejectMesh("UNSUPPORTED_MATERIAL_FEATURE");
  }
  if (
    (material.customProgramCacheKey &&
      material.customProgramCacheKey !== DEFAULT_MATERIAL_CUSTOM_PROGRAM_CACHE_KEY) ||
    Object.hasOwn(material, "customProgramCacheKey")
  ) {
    return rejectMesh("UNSUPPORTED_MATERIAL_FEATURE");
  }

  return { admitted: true };
}

/**
 * Computes 4x4 matrix product C = A * B in column-major Float64Array.
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {Float64Array}
 */
export function multiplyMatrices4x4(a, b) {
  const ae = a.elements || a;
  const be = b.elements || b;
  const out = new Float64Array(16);
  for (let col = 0; col < 4; col++) {
    const b0 = be[col * 4];
    const b1 = be[col * 4 + 1];
    const b2 = be[col * 4 + 2];
    const b3 = be[col * 4 + 3];
    out[col * 4] = ae[0] * b0 + ae[4] * b1 + ae[8] * b2 + ae[12] * b3;
    out[col * 4 + 1] = ae[1] * b0 + ae[5] * b1 + ae[9] * b2 + ae[13] * b3;
    out[col * 4 + 2] = ae[2] * b0 + ae[6] * b1 + ae[10] * b2 + ae[14] * b3;
    out[col * 4 + 3] = ae[3] * b0 + ae[7] * b1 + ae[11] * b2 + ae[15] * b3;
  }
  return out;
}

/**
 * Expands indexed vertex positions into flat unindexed triangle vertices.
 * Truncates incomplete tail indices (indices.length not divisible by 3) per native triangle-list semantics.
 * @param {Float32Array} positions
 * @param {Uint32Array} indices
 * @returns {Float32Array}
 */
export function expandIndexedPositions(positions, indices) {
  const completeTriangles = Math.floor(indices.length / 3);
  const count = completeTriangles * 3;
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const idx = indices[i];
    out[i * 3] = positions[idx * 3];
    out[i * 3 + 1] = positions[idx * 3 + 1];
    out[i * 3 + 2] = positions[idx * 3 + 2];
  }
  return out;
}

/**
 * Expands indexed RGBA vertex colors into flat unindexed triangle vertex colors.
 * Truncates incomplete tail indices per native triangle-list semantics matching expandIndexedPositions.
 * @param {Float32Array} colors - Per-vertex RGBA floats (totalVertexCount * 4)
 * @param {Uint32Array} indices
 * @returns {Float32Array}
 */
export function expandIndexedColors(colors, indices) {
  const completeTriangles = Math.floor(indices.length / 3);
  const count = completeTriangles * 3;
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const idx = indices[i];
    out[i * 4] = colors[idx * 4];
    out[i * 4 + 1] = colors[idx * 4 + 1];
    out[i * 4 + 2] = colors[idx * 4 + 2];
    out[i * 4 + 3] = colors[idx * 4 + 3];
  }
  return out;
}

/**
 * Converts scene.background (THREE.Color) into 4 clear_color floats.
 * Pinned WebGLBackground.js:241 calls:
 *   color.getRGB( _rgb, getUnlitUniformColorSpace( renderer ) )
 * which defaults to renderer.outputColorSpace ('srgb').
 * Alpha is fixed to 1.0 per WebGLBackground.js:57.
 *
 * @param {any} background - THREE.Color instance (background.isColor === true)
 * @param {object} [options] - Options (colorSpace, outputColorSpace, renderer)
 * @returns {Float32Array} 4 floats [r, g, b, 1.0]
 */
export function convertBackgroundColorToClearColor(background, options = {}) {
  if (!background || !background.isColor) {
    throw createAdmissionError(
      "UNSUPPORTED_SCENE_FEATURE",
      "scene.background must be an instance of THREE.Color with isColor === true",
    );
  }

  const target = { r: background.r ?? 0, g: background.g ?? 0, b: background.b ?? 0 };

  const colorSpace =
    options.colorSpace ?? options.outputColorSpace ?? options.renderer?.outputColorSpace ?? "srgb";

  if (typeof background.getRGB === "function") {
    background.getRGB(target, colorSpace);
  }

  return new Float32Array([target.r, target.g, target.b, 1.0]);
}

// Internal Symbol to pass residency context safely without polluting public options
const RESIDENCY_CONTEXT = Symbol("f3d.residencyContext");

// Module-level WeakMap: bridgeHost -> { deviceGeneration: number, device: any, attributes: WeakMap<BufferAttribute, AttributeGpuRecord>, bindings: WeakMap<BufferGeometry, Map<string, HostBindingRecord>>, disposalRegistered: WeakSet<BufferGeometry> }
const hostResidencyMap = new WeakMap();

function getHostResidency(bridgeHost) {
  if (!bridgeHost) return null;
  const currentGen = bridgeHost.deviceGeneration ?? 0;
  const currentDevice = bridgeHost.device ?? null;
  let residency = hostResidencyMap.get(bridgeHost);
  if (!residency) {
    residency = {
      deviceGeneration: currentGen,
      device: currentDevice,
      attributes: new WeakMap(),
      bindings: new WeakMap(),
      disposalRegistered: new WeakSet(),
    };
    hostResidencyMap.set(bridgeHost, residency);
  } else if (residency.deviceGeneration !== currentGen || residency.device !== currentDevice) {
    residency.deviceGeneration = currentGen;
    residency.device = currentDevice;
    residency.attributes = new WeakMap();
    residency.bindings = new WeakMap();
  }
  return residency;
}

function resolveHostGeometryBinding(
  residencyContext,
  geometry,
  programKey,
  posAttr,
  colorAttr,
  hasVertexColors,
) {
  if (!residencyContext || !geometry) {
    return null;
  }
  const { residency, pendingCommits, stagedBindingsThisPass } = residencyContext;
  if (!residency) return null;

  // Invalidate on geometry disposal (WebGLGeometries.js:8-24, WebGLBindingStates.js:636)
  if (
    typeof geometry.addEventListener === "function" &&
    !residency.disposalRegistered.has(geometry)
  ) {
    residency.disposalRegistered.add(geometry);
    geometry.addEventListener("dispose", () => {
      if (residency.bindings) {
        residency.bindings.delete(geometry);
      }
      if (geometry.index) {
        const target = geometry.index.isInterleavedBufferAttribute
          ? geometry.index.data
          : geometry.index;
        if (target) residency.attributes.delete(target);
      }
      if (geometry.attributes) {
        for (const attrName in geometry.attributes) {
          const attr = geometry.attributes[attrName];
          const target = attr?.isInterleavedBufferAttribute ? attr.data : attr;
          if (target) residency.attributes.delete(target);
        }
      }
    });
  }

  const currentIndex = geometry.index ?? null;
  const posData = posAttr?.isInterleavedBufferAttribute ? posAttr.data : null;
  const colorData =
    hasVertexColors && colorAttr?.isInterleavedBufferAttribute ? colorAttr.data : null;

  // 1. Check if binding was already resolved in this render pass for this geometry and program variant
  let stagedGeomBindings = stagedBindingsThisPass?.get(geometry);
  if (!stagedGeomBindings && stagedBindingsThisPass) {
    stagedGeomBindings = new Map();
    stagedBindingsThisPass.set(geometry, stagedGeomBindings);
  }
  const staged = stagedGeomBindings?.get(programKey);

  let geomBindings = residency.bindings?.get(geometry);
  if (!geomBindings && residency.bindings) {
    geomBindings = new Map();
    residency.bindings.set(geometry, geomBindings);
  }
  const cached = staged || geomBindings?.get(programKey);

  // needsUpdate (WebGLBindingStates.js:149-192):
  // Rebinds ALL consumed vertex pointers for this program when any consumed attribute/data identity or index identity changes
  let needsUpdate = !cached;
  if (cached) {
    if (
      cached.index !== currentIndex ||
      cached.posAttribute !== posAttr ||
      cached.posData !== posData
    ) {
      needsUpdate = true;
    } else if (
      hasVertexColors &&
      (cached.colorAttribute !== colorAttr || cached.colorData !== colorData)
    ) {
      needsUpdate = true;
    }
  }

  if (needsUpdate) {
    const fresh = {
      index: currentIndex,
      programKey,
      hasVertexColors,
      posAttribute: posAttr,
      posData,
      posOffset: posAttr?.offset ?? 0,
      posStride: posAttr?.data?.stride ?? 0,
      colorAttribute: colorAttr,
      colorData,
      colorOffset: colorAttr?.offset ?? 0,
      colorStride: colorAttr?.data?.stride ?? 0,
    };
    if (stagedGeomBindings) {
      stagedGeomBindings.set(programKey, fresh);
    }
    if (pendingCommits) {
      pendingCommits.push(() => {
        geomBindings.set(programKey, fresh);
      });
    } else if (geomBindings) {
      geomBindings.set(programKey, fresh);
    }
    return fresh;
  }

  return cached;
}

function resolveHostAttribute(residencyContext, attribute, sourceBackend, geometry) {
  if (!residencyContext || !attribute) return null;
  const { residency, pendingCommits, stagedThisPass } = residencyContext;
  if (!residency) return null;

  // Invalidate on geometry disposal (WebGLGeometries.js:8-24, WebGLAttributes.js:165-178, WebGLBindingStates.js:636)
  // Per-residency WeakSet registration without mutating public geometry object (root review 18499)
  if (
    geometry &&
    typeof geometry.addEventListener === "function" &&
    !residency.disposalRegistered.has(geometry)
  ) {
    residency.disposalRegistered.add(geometry);
    geometry.addEventListener("dispose", () => {
      if (residency.bindings) {
        residency.bindings.delete(geometry);
      }
      if (geometry.index) {
        const target = geometry.index.isInterleavedBufferAttribute
          ? geometry.index.data
          : geometry.index;
        if (target) residency.attributes.delete(target);
      }
      if (geometry.attributes) {
        for (const name in geometry.attributes) {
          const attr = geometry.attributes[name];
          const target = attr?.isInterleavedBufferAttribute ? attr.data : attr;
          if (target) residency.attributes.delete(target);
        }
      }
    });
  }

  const target = attribute.isInterleavedBufferAttribute ? attribute.data : attribute;
  if (!target || !target.array) return null;

  // If attribute was already staged in this render pass (e.g. shared by multiple meshes or views)
  if (stagedThisPass && stagedThisPass.has(target)) {
    return stagedThisPass.get(target);
  }

  let record = residency.attributes.get(target);

  // Initial upload: first use on this host & deviceGeneration & device
  if (!record) {
    const stagedShadow = target.array.slice();
    record = {
      version: target.version,
      shadow: stagedShadow,
      size: target.array.byteLength,
    };
    if (stagedThisPass) {
      stagedThisPass.set(target, stagedShadow);
    }
    // WebGLAttributes.js:5-16: initial creation uploads whole array and does NOT clear existing ranges
    pendingCommits.push(() => {
      residency.attributes.set(target, record);
    });
    return stagedShadow;
  }

  // Check if update is needed based on sourceBackend
  // Three.js constants: DynamicDrawUsage = 35048
  const backend = sourceBackend?.toLowerCase();
  const isWebGPU = backend === SOURCE_BACKEND.WEBGPU;
  const isWebGL = backend === SOURCE_BACKEND.WEBGL;
  const isDynamic = target.usage === 35048;
  const versionBumped = record.version < target.version;

  if (!versionBumped && isDynamic) {
    if (!isWebGPU && !isWebGL) {
      throw createAdmissionError(
        "AMBIGUOUS_ATTRIBUTE_UPDATE",
        'Attribute with DynamicDrawUsage and unchanged version is ambiguous across backends; provide options.sourceBackend ("webgl" | "webgpu")',
      );
    }
  }

  const needsUpload = versionBumped || (isWebGPU && isDynamic);

  // If no upload requested, return GPU-stale shadow!
  if (!needsUpload) {
    if (stagedThisPass) {
      stagedThisPass.set(target, record.shadow);
    }
    return record.shadow;
  }

  // Resizing buffer attributes is not supported (WebGLAttributes.js:212-215)
  if (record.size !== target.array.byteLength) {
    throw createAdmissionError(
      "INVALID_ATTRIBUTE_RESIZE",
      "THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.",
    );
  }

  const stagedShadow = record.shadow.slice();
  const updateRanges = target.updateRanges;
  let clearRanges = false;

  if (!updateRanges || updateRanges.length === 0) {
    stagedShadow.set(target.array);
  } else {
    // Validate consumed ranges before any commit (root 18546)
    // Must be positive integral count/start, within array bounds
    for (let i = 0; i < updateRanges.length; i++) {
      const range = updateRanges[i];
      if (
        !range ||
        !Number.isInteger(range.start) ||
        range.start < 0 ||
        !Number.isInteger(range.count) ||
        range.count <= 0 ||
        range.start + range.count > target.array.length
      ) {
        throw createAdmissionError(
          "INVALID_UPDATE_RANGE",
          `Invalid updateRange at index ${i}: start must be non-negative integer, count must be positive integer, and start + count must not exceed attribute array length`,
        );
      }
    }

    clearRanges = true;
    if (updateRanges.length === 1) {
      const range = updateRanges[0];
      stagedShadow.set(target.array.subarray(range.start, range.start + range.count), range.start);
    } else {
      // Multiple update ranges: WebGL merges with +1, WebGPU updates individually
      if (!isWebGPU && !isWebGL) {
        throw createAdmissionError(
          "AMBIGUOUS_ATTRIBUTE_UPDATE",
          'Multiple updateRanges on BufferAttribute have ambiguous merging across backends; provide options.sourceBackend ("webgl" | "webgpu")',
        );
      }
      if (isWebGPU) {
        // WebGPU applies ranges individually (WebGPUAttributeUtils.js:223-272)
        for (let i = 0; i < updateRanges.length; i++) {
          const range = updateRanges[i];
          stagedShadow.set(
            target.array.subarray(range.start, range.start + range.count),
            range.start,
          );
        }
      } else {
        // WebGL exact +1 merge: range.start <= prev.start + prev.count + 1 (WebGLAttributes.js:119)
        // Deep-copy each { start, count } so originals are never mutated during preflight!
        const rangesCopy = updateRanges
          .map((r) => ({ start: r.start, count: r.count }))
          .sort((a, b) => a.start - b.start);
        let mergeIndex = 0;
        for (let i = 1; i < rangesCopy.length; i++) {
          const prev = rangesCopy[mergeIndex];
          const cur = rangesCopy[i];
          if (cur.start <= prev.start + prev.count + 1) {
            prev.count = Math.max(prev.count, cur.start + cur.count - prev.start);
          } else {
            mergeIndex++;
            rangesCopy[mergeIndex] = cur;
          }
        }
        rangesCopy.length = mergeIndex + 1;
        for (let i = 0; i < rangesCopy.length; i++) {
          const range = rangesCopy[i];
          stagedShadow.set(
            target.array.subarray(range.start, range.start + range.count),
            range.start,
          );
        }
      }
    }
  }

  if (stagedThisPass) {
    stagedThisPass.set(target, stagedShadow);
  }

  const targetVersion = target.version;
  pendingCommits.push(() => {
    record.shadow = stagedShadow;
    record.version = targetVersion;
    if (clearRanges) {
      target.clearUpdateRanges();
    }
  });

  return stagedShadow;
}

/**
 * Extracts a complete, isolated typed-array snapshot of a Three.js Mesh + Camera.
 * Preserves source matrix-update boundaries without mutating application state.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} [options]
 * @returns {object} Isolated typed-array snapshot
 */
export function extractMeshRenderData(mesh, camera, width, height, options = {}) {
  const admission = canAdmitMesh(mesh, camera, options);
  if (!admission.admitted) {
    throw createAdmissionError(admission.code);
  }
  if (
    !Number.isInteger(width) ||
    width <= 0 ||
    width > 0xffffffff ||
    !Number.isInteger(height) ||
    height <= 0 ||
    height > 0xffffffff
  ) {
    throw createAdmissionError("INVALID_DIMENSIONS");
  }

  const geometry = mesh.geometry;
  const posAttr = geometry.attributes.position;
  const indexAttr = geometry.index;
  const mat = mesh.material;
  const matSide = mat?.side ?? THREE_SIDE.FRONT_SIDE;
  const hasVertexColors = Boolean(mat.vertexColors === true && geometry.attributes?.color);
  const colorAttr = hasVertexColors ? geometry.attributes.color : null;
  const colorItemSize = hasVertexColors ? (colorAttr?.itemSize ?? 0) : 0;
  // Bounded base-MeshBasicMaterial program key (WebGLPrograms.js:310-311, 373-374).
  // Limitation: does not track precision, dithering, premultipliedAlpha, defines, or wider renderer state;
  // full multi-feature shader-key tracking remains future compatibility work.
  const programKey = `s:${matSide}|vc:${hasVertexColors ? (colorItemSize === 4 ? "rgba" : "rgb") : "none"}`;

  const residencyContext = options?.[RESIDENCY_CONTEXT];
  const posShadow = residencyContext
    ? resolveHostAttribute(residencyContext, posAttr, options.sourceBackend, geometry)
    : null;
  const indexShadow =
    residencyContext && indexAttr
      ? resolveHostAttribute(residencyContext, indexAttr, options.sourceBackend, geometry)
      : null;
  const colorShadow =
    residencyContext && colorAttr
      ? resolveHostAttribute(residencyContext, colorAttr, options.sourceBackend, geometry)
      : null;

  const binding = resolveHostGeometryBinding(
    residencyContext,
    geometry,
    programKey,
    posAttr,
    colorAttr,
    hasVertexColors,
  );

  const boundPosOffset =
    binding && posAttr.isInterleavedBufferAttribute ? binding.posOffset : posAttr.offset;
  const boundPosStride =
    binding && posAttr.isInterleavedBufferAttribute ? binding.posStride : posAttr.data?.stride;

  const effectivePosAttr = posAttr.isInterleavedBufferAttribute
    ? Object.create(posAttr, {
        offset: { value: boundPosOffset, writable: true, configurable: true, enumerable: true },
        data: {
          value: Object.create(posAttr.data, {
            stride: { value: boundPosStride, writable: true, configurable: true, enumerable: true },
            array: {
              value: posShadow ?? posAttr.data.array,
              writable: true,
              configurable: true,
              enumerable: true,
            },
          }),
          writable: true,
          configurable: true,
          enumerable: true,
        },
      })
    : posShadow
      ? Object.create(posAttr, {
          array: { value: posShadow, writable: true, configurable: true, enumerable: true },
        })
      : posAttr;

  const totalVertexCount = posAttr.count;
  const effectiveIndexSource = indexShadow ?? indexAttr?.array;

  const boundColorOffset =
    binding && colorAttr?.isInterleavedBufferAttribute ? binding.colorOffset : colorAttr?.offset;
  const boundColorStride =
    binding && colorAttr?.isInterleavedBufferAttribute
      ? binding.colorStride
      : colorAttr?.data?.stride;

  const effectiveColorAttr = colorAttr
    ? colorAttr.isInterleavedBufferAttribute
      ? Object.create(colorAttr, {
          offset: { value: boundColorOffset, writable: true, configurable: true, enumerable: true },
          data: {
            value: Object.create(colorAttr.data, {
              stride: {
                value: boundColorStride,
                writable: true,
                configurable: true,
                enumerable: true,
              },
              array: {
                value: colorShadow ?? colorAttr.data.array,
                writable: true,
                configurable: true,
                enumerable: true,
              },
            }),
            writable: true,
            configurable: true,
            enumerable: true,
          },
        })
      : colorShadow
        ? Object.create(colorAttr, {
            array: { value: colorShadow, writable: true, configurable: true, enumerable: true },
          })
        : colorAttr
    : null;

  // Validate drawRange parameters
  const drawStart = geometry.drawRange?.start ?? 0;
  const rawDrawCount = geometry.drawRange?.count;
  if (!Number.isInteger(drawStart) || drawStart < 0) {
    throw createAdmissionError("INVALID_DRAWRANGE", "start must be non-negative integer");
  }
  if (
    rawDrawCount !== undefined &&
    rawDrawCount !== Infinity &&
    (!Number.isInteger(rawDrawCount) || rawDrawCount < 0)
  ) {
    throw createAdmissionError("INVALID_DRAWRANGE", "count must be non-negative integer");
  }

  let positions;
  let indices;
  let rawColors = null;
  let unindexedColors = null;
  let triangleCount = 0;

  if (indexAttr) {
    const totalIndexCount = indexAttr.count;
    // Clamp count to remaining indices past drawStart
    let effectiveCount = 0;
    if (drawStart < totalIndexCount) {
      const maxAvailable = totalIndexCount - drawStart;
      effectiveCount =
        rawDrawCount !== undefined && rawDrawCount !== Infinity
          ? Math.min(rawDrawCount, maxAvailable)
          : maxAvailable;
    }

    if (effectiveCount === 0) {
      // Empty indexed range must pass empty positions, because empty indices
      // mean unindexed and would draw all positions!
      positions = new Float32Array(0);
      indices = new Uint32Array(0);
      triangleCount = 0;
    } else {
      indices = new Uint32Array(effectiveCount);
      for (let i = 0; i < effectiveCount; i++) {
        // Element buffers consume raw integer indices; normalized applies to
        // vertex attributes only. getX() would normalize these into fractions.
        const idx = effectiveIndexSource[drawStart + i];
        // Explicit bounds check to prevent silent conversion of undefined to 0
        if (idx === undefined || idx < 0 || idx >= totalVertexCount) {
          throw createAdmissionError(
            "INDEX_OUT_OF_BOUNDS",
            `index ${idx} at position ${drawStart + i} exceeds vertex count ${totalVertexCount}`,
          );
        }
        indices[i] = idx;
      }

      // Read vertex positions safely
      positions = new Float32Array(totalVertexCount * 3);
      for (let i = 0; i < totalVertexCount; i++) {
        positions[i * 3] = effectivePosAttr.getX(i);
        positions[i * 3 + 1] = effectivePosAttr.getY(i);
        positions[i * 3 + 2] = effectivePosAttr.getZ(i);
      }
      if (hasVertexColors) {
        rawColors = new Float32Array(totalVertexCount * 4);
        for (let i = 0; i < totalVertexCount; i++) {
          rawColors[i * 4] = effectiveColorAttr.getX(i);
          rawColors[i * 4 + 1] = effectiveColorAttr.getY(i);
          rawColors[i * 4 + 2] = effectiveColorAttr.getZ(i);
          // Pinned Three.js OPAQUE shader forces output alpha to 1.0 even for RGBA vertex colors
          rawColors[i * 4 + 3] = 1.0;
        }
      }
      triangleCount = Math.floor(effectiveCount / 3);
    }
  } else {
    // Non-indexed
    let effectiveCount = 0;
    if (drawStart < totalVertexCount) {
      const maxAvailable = totalVertexCount - drawStart;
      effectiveCount =
        rawDrawCount !== undefined && rawDrawCount !== Infinity
          ? Math.min(rawDrawCount, maxAvailable)
          : maxAvailable;
    }

    positions = new Float32Array(effectiveCount * 3);
    if (hasVertexColors) {
      unindexedColors = new Float32Array(effectiveCount * 4);
      for (let i = 0; i < effectiveCount; i++) {
        const vertIdx = drawStart + i;
        positions[i * 3] = effectivePosAttr.getX(vertIdx);
        positions[i * 3 + 1] = effectivePosAttr.getY(vertIdx);
        positions[i * 3 + 2] = effectivePosAttr.getZ(vertIdx);
        unindexedColors[i * 4] = effectiveColorAttr.getX(vertIdx);
        unindexedColors[i * 4 + 1] = effectiveColorAttr.getY(vertIdx);
        unindexedColors[i * 4 + 2] = effectiveColorAttr.getZ(vertIdx);
        // Pinned Three.js OPAQUE shader forces output alpha to 1.0 even for RGBA vertex colors
        unindexedColors[i * 4 + 3] = 1.0;
      }
    } else {
      for (let i = 0; i < effectiveCount; i++) {
        const vertIdx = drawStart + i;
        positions[i * 3] = effectivePosAttr.getX(vertIdx);
        positions[i * 3 + 1] = effectivePosAttr.getY(vertIdx);
        positions[i * 3 + 2] = effectivePosAttr.getZ(vertIdx);
      }
    }
    indices = new Uint32Array(0);
    triangleCount = Math.floor(effectiveCount / 3);
  }

  // Expanded unindexed positions for backends without index buffer support
  const expandedPositions =
    indices.length > 0 ? expandIndexedPositions(positions, indices) : new Float32Array(positions);

  let expandedVertexColors = null;
  if (hasVertexColors) {
    if (indices.length > 0) {
      expandedVertexColors = expandIndexedColors(rawColors, indices);
    } else {
      expandedVertexColors = unindexedColors;
    }
  }

  // Model-View Matrix: MV = camera.matrixWorldInverse * mesh.matrixWorld
  const modelView = multiplyMatrices4x4(camera.matrixWorldInverse, mesh.matrixWorld);

  // Source coordinate system check: default in Three.js is WebGL (2000)
  // When options.sourceBackend is explicitly WebGL, hardware uses WebGL clip depth [-1, 1], requiring webglDepth=true.
  const isExplicitWebGL = options?.sourceBackend?.toLowerCase() === SOURCE_BACKEND.WEBGL;
  const isWebGPUCoord = !isExplicitWebGL && camera.coordinateSystem === COORDINATE_SYSTEM.WEBGPU;
  const webglDepth = !isWebGPUCoord;

  // Raw projection matrix (16 f64 elements)
  const projection = new Float64Array(camera.projectionMatrix.elements);

  // Material Color: [r, g, b, opacity] in linear sRGB
  const color = new Float32Array([
    mat.color?.r ?? 1.0,
    mat.color?.g ?? 1.0,
    mat.color?.b ?? 1.0,
    mat.opacity ?? 1.0,
  ]);

  // Material Color & Depth Settings
  const colorWrite = mat.colorWrite !== false;
  const depthTest = mat.depthTest !== false;
  const rawDepthWrite = mat.depthWrite !== false;
  const rawDepthFunc = mat.depthFunc ?? LessEqualDepth;
  if (THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc] === undefined) {
    throw createAdmissionError("INVALID_DEPTH_FUNC", `${rawDepthFunc}`);
  }

  // Resolve effective depthWrite:
  // - Under WebGL backend semantics, gl.disable(gl.DEPTH_TEST) suppresses depth writes in hardware,
  //   so effective depthWrite is false.
  // - Under WebGPU backend semantics (r186 WebGPUPipelineUtils.js:224), depthWrite is passed directly
  //   to depthWriteEnabled even when depthTest is false (compare Always), so effective depthWrite is true.
  // - Requires options.sourceBackend ('webgl' | 'webgpu') to disambiguate; otherwise refused.
  // - For depthTest=true, or depthWrite=false, no backend parameter is needed.
  let depthWrite = rawDepthWrite;
  if (!depthTest && rawDepthWrite) {
    const backend = options?.sourceBackend?.toLowerCase();
    if (backend === SOURCE_BACKEND.WEBGL) {
      depthWrite = false;
    } else if (backend === SOURCE_BACKEND.WEBGPU) {
      depthWrite = true;
    } else {
      throw createAdmissionError("AMBIGUOUS_DEPTH_PAIR");
    }
  }

  // When depthTest is false, WebGPU pipeline uses GPUCompareFunction.Always (wire 8)
  const depthCompare = depthTest
    ? THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc]
    : DEPTH_WIRE_COMPARE.ALWAYS;

  // Material Side & Culling Settings
  const side = mat.side ?? THREE_SIDE.FRONT_SIDE;
  if (
    side !== THREE_SIDE.FRONT_SIDE &&
    side !== THREE_SIDE.BACK_SIDE &&
    side !== THREE_SIDE.DOUBLE_SIDE
  ) {
    throw createAdmissionError("UNSUPPORTED_SIDE", `material side ${side} is not supported`);
  }

  const det = computeAffineDeterminant(mesh.matrixWorld);
  const isReflected = det < 0;
  let flipSided = side === THREE_SIDE.BACK_SIDE;
  if (isReflected) flipSided = !flipSided;

  const cullMode = side === THREE_SIDE.DOUBLE_SIDE ? CULL_MODE_WIRE.NONE : CULL_MODE_WIRE.BACK;
  const frontFace = flipSided ? FRONT_FACE_WIRE.CW : FRONT_FACE_WIRE.CCW;

  return Object.freeze({
    positions,
    indices,
    expandedPositions,
    vertexColors: mat.vertexColors === true,
    hasVertexColors,
    expandedVertexColors,
    modelView,
    projection,
    color,
    colorWrite,
    width,
    height,
    webglDepth,
    depthTest,
    depthWrite,
    depthFunc: rawDepthFunc,
    depthCompare,
    vertexCount: positions.length / 3,
    triangleCount,
    isIndexed: indices.length > 0,
    side,
    isReflected,
    flipSided,
    cullMode,
    frontFace,
  });
}

/**
 * Helper to route single-mesh FrontSide and BackSide draws through f3d_build_mesh_batch_cull_packet (N=1).
 * Invariant: Never falls back to no-cull Wasm entry point for non-DoubleSide meshes.
 */
function buildSingleMeshCullPacket(snapshot, width, height, wasmModule, isCanvas, options = {}) {
  const vertexColorClearBatchFn = wasmModule?.f3d_build_mesh_batch_vertex_color_clear_packet;
  const vertexColorBatchFn = wasmModule?.f3d_build_mesh_batch_vertex_color_packet;
  const cullDepthColorBatchFn = wasmModule?.f3d_build_mesh_batch_cull_depth_color_packet;
  const cullDepthBatchFn = wasmModule?.f3d_build_mesh_batch_cull_depth_packet;
  const cullBatchFn = wasmModule?.f3d_build_mesh_batch_cull_packet;

  if (options.clearColor) {
    if (typeof vertexColorClearBatchFn !== "function") {
      throw createAdmissionError(
        "INCOMPATIBLE_BACKGROUND_CLEAR",
        "scene.background Color requires Wasm export f3d_build_mesh_batch_vertex_color_clear_packet",
      );
    }
  } else if (snapshot.vertexColors === true) {
    if (typeof vertexColorBatchFn !== "function") {
      throw createAdmissionError(
        "INCOMPATIBLE_VERTEX_COLORS",
        "Meshes with vertexColors require Wasm export f3d_build_mesh_batch_vertex_color_packet",
      );
    }
  } else {
    if (snapshot.colorWrite === false && typeof cullDepthColorBatchFn !== "function") {
      throw createAdmissionError(
        "INCOMPATIBLE_COLOR_WRITE",
        "Meshes with colorWrite=false require Wasm export f3d_build_mesh_batch_cull_depth_color_packet",
      );
    }

    if (
      typeof cullDepthColorBatchFn !== "function" &&
      typeof cullDepthBatchFn !== "function" &&
      typeof cullBatchFn !== "function"
    ) {
      throw createAdmissionError(
        "MISSING_CULL_EXPORT",
        `wasmModule is missing f3d_build_mesh_batch_cull_packet export for side ${snapshot.side}. Silent DoubleSide fallback is strictly forbidden.`,
      );
    }
  }

  if (
    snapshot.cullMode !== CULL_MODE_WIRE.NONE &&
    snapshot.cullMode !== CULL_MODE_WIRE.FRONT &&
    snapshot.cullMode !== CULL_MODE_WIRE.BACK
  ) {
    throw createAdmissionError(
      "INVALID_CULL_MODE",
      `snapshot has invalid cullMode ${snapshot.cullMode}`,
    );
  }
  if (snapshot.frontFace !== FRONT_FACE_WIRE.CCW && snapshot.frontFace !== FRONT_FACE_WIRE.CW) {
    throw createAdmissionError(
      "INVALID_FRONT_FACE",
      `snapshot has invalid frontFace ${snapshot.frontFace}`,
    );
  }

  const positionsToUse = snapshot.expandedPositions;
  const vertexCount = positionsToUse.length / 3;
  const vertexCounts = new Uint32Array([vertexCount]);
  const cullModes = new Uint8Array([snapshot.cullMode]);
  const frontFaces = new Uint8Array([snapshot.frontFace]);

  if (options.clearColor && typeof vertexColorClearBatchFn === "function") {
    const depthTests = new Uint8Array([snapshot.depthTest ? 1 : 0]);
    const depthWrites = new Uint8Array([snapshot.depthWrite ? 1 : 0]);
    const depthCompares = new Uint32Array([snapshot.depthCompare]);
    const colorWrites = new Uint8Array([snapshot.colorWrite ? 1 : 0]);
    const vertexColorsToUse =
      snapshot.hasVertexColors && snapshot.expandedVertexColors
        ? snapshot.expandedVertexColors
        : new Float32Array(vertexCount * 4).fill(1.0);
    return vertexColorClearBatchFn(
      positionsToUse,
      vertexCounts,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      cullModes,
      frontFaces,
      depthTests,
      depthWrites,
      depthCompares,
      colorWrites,
      width,
      height,
      snapshot.webglDepth,
      isCanvas,
      vertexColorsToUse,
      options.clearColor,
    );
  }

  if (snapshot.vertexColors === true && typeof vertexColorBatchFn === "function") {
    const depthTests = new Uint8Array([snapshot.depthTest ? 1 : 0]);
    const depthWrites = new Uint8Array([snapshot.depthWrite ? 1 : 0]);
    const depthCompares = new Uint32Array([snapshot.depthCompare]);
    const colorWrites = new Uint8Array([snapshot.colorWrite ? 1 : 0]);
    const vertexColorsToUse =
      snapshot.hasVertexColors && snapshot.expandedVertexColors
        ? snapshot.expandedVertexColors
        : new Float32Array(vertexCount * 4).fill(1.0);
    return vertexColorBatchFn(
      positionsToUse,
      vertexCounts,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      cullModes,
      frontFaces,
      depthTests,
      depthWrites,
      depthCompares,
      colorWrites,
      width,
      height,
      snapshot.webglDepth,
      isCanvas,
      vertexColorsToUse,
    );
  }

  if (typeof cullDepthColorBatchFn === "function") {
    const depthTests = new Uint8Array([snapshot.depthTest ? 1 : 0]);
    const depthWrites = new Uint8Array([snapshot.depthWrite ? 1 : 0]);
    const depthCompares = new Uint32Array([snapshot.depthCompare]);
    const colorWrites = new Uint8Array([snapshot.colorWrite ? 1 : 0]);
    return cullDepthColorBatchFn(
      positionsToUse,
      vertexCounts,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      cullModes,
      frontFaces,
      depthTests,
      depthWrites,
      depthCompares,
      colorWrites,
      width,
      height,
      snapshot.webglDepth,
      isCanvas,
    );
  }

  if (typeof cullDepthBatchFn === "function") {
    const depthTests = new Uint8Array([snapshot.depthTest ? 1 : 0]);
    const depthWrites = new Uint8Array([snapshot.depthWrite ? 1 : 0]);
    const depthCompares = new Uint32Array([snapshot.depthCompare]);
    return cullDepthBatchFn(
      positionsToUse,
      vertexCounts,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      cullModes,
      frontFaces,
      depthTests,
      depthWrites,
      depthCompares,
      width,
      height,
      snapshot.webglDepth,
      isCanvas,
    );
  }

  return cullBatchFn(
    positionsToUse,
    vertexCounts,
    snapshot.modelView,
    snapshot.projection,
    snapshot.color,
    cullModes,
    frontFaces,
    width,
    height,
    snapshot.webglDepth,
    snapshot.depthTest,
    snapshot.depthWrite,
    snapshot.depthCompare,
    isCanvas,
  );
}

/**
 * Prepares a binary submission packet targeting a visible canvas presentation.
 * Automatically selects 11-argument depth export when depth is enabled or available,
 * preserving legacy 8-argument export when depth is disabled.
 *
 * Invariants:
 * - Requires wasmModule to expose f3d_build_canvas_mesh_depth_packet or f3d_build_canvas_mesh_packet.
 * - Explicitly refuses execution if the visible canvas export is missing;
 *   silent offscreen fallback is strictly forbidden.
 * - Explicitly refuses execution if mesh requires depth but depth export is missing.
 * - If mesh is FrontSide or BackSide, routes through f3d_build_mesh_batch_cull_packet (N=1);
 *   silent DoubleSide fallback is strictly forbidden.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} wasmModule - Loaded Wasm module
 * @param {object} [options]
 * @returns {{ packetBytes: Uint8Array, snapshot: object, target: 'canvas' }}
 */
export function prepareCanvasMeshPacket(mesh, camera, width, height, wasmModule, options = {}) {
  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (
    snapshot.side !== THREE_SIDE.DOUBLE_SIDE ||
    snapshot.colorWrite === false ||
    snapshot.vertexColors === true ||
    options.clearColor
  ) {
    const packetBytes = buildSingleMeshCullPacket(
      snapshot,
      width,
      height,
      wasmModule,
      true,
      options,
    );
    return { packetBytes, snapshot, target: "canvas" };
  }

  const canvasDepthFn =
    wasmModule?.f3d_build_canvas_mesh_depth_packet ||
    wasmModule?.gpu_bridge_build_canvas_mesh_depth_packet;
  const canvasLegacyFn =
    wasmModule?.f3d_build_canvas_mesh_packet || wasmModule?.gpu_bridge_build_canvas_mesh_packet;

  if (
    !wasmModule ||
    (typeof canvasDepthFn !== "function" && typeof canvasLegacyFn !== "function")
  ) {
    throw new Error(
      "Visible canvas mesh packet preparation failed: wasmModule is missing f3d_build_canvas_mesh_packet export. " +
        "Silent offscreen-as-visible fallback is strictly forbidden.",
    );
  }

  const requiresDepth = snapshot.depthTest === true || snapshot.depthWrite === true;
  if (requiresDepth && typeof canvasDepthFn !== "function") {
    throw new Error(
      "Visible canvas mesh requires depth (depthTest or depthWrite enabled), but wasmModule is missing " +
        "f3d_build_canvas_mesh_depth_packet export. Silent offscreen-as-visible fallback or retained renderer masquerading is strictly forbidden.",
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  let packetBytes;
  if (typeof canvasDepthFn === "function") {
    packetBytes = canvasDepthFn(
      positionsToUse,
      indicesToUse,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      width,
      height,
      snapshot.webglDepth,
      snapshot.depthTest,
      snapshot.depthWrite,
      snapshot.depthCompare,
    );
  } else {
    packetBytes = canvasLegacyFn(
      positionsToUse,
      indicesToUse,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      width,
      height,
      snapshot.webglDepth,
    );
  }

  return { packetBytes, snapshot, target: "canvas" };
}

/**
 * Prepares a binary submission packet with explicit depth settings targeting a visible canvas presentation.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} wasmModule - Loaded Wasm module exposing f3d_build_canvas_mesh_depth_packet
 * @param {object} [options]
 * @returns {{ packetBytes: Uint8Array, snapshot: object, target: 'canvas' }}
 */
export function prepareCanvasMeshDepthPacket(
  mesh,
  camera,
  width,
  height,
  wasmModule,
  options = {},
) {
  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (
    snapshot.side !== THREE_SIDE.DOUBLE_SIDE ||
    snapshot.colorWrite === false ||
    snapshot.vertexColors === true ||
    options.clearColor
  ) {
    const packetBytes = buildSingleMeshCullPacket(
      snapshot,
      width,
      height,
      wasmModule,
      true,
      options,
    );
    return { packetBytes, snapshot, target: "canvas" };
  }

  const buildFn =
    wasmModule?.f3d_build_canvas_mesh_depth_packet ||
    wasmModule?.gpu_bridge_build_canvas_mesh_depth_packet;

  if (typeof buildFn !== "function") {
    throw new Error(
      "Visible canvas mesh depth packet preparation failed: wasmModule is missing f3d_build_canvas_mesh_depth_packet / gpu_bridge_build_canvas_mesh_depth_packet export. " +
        "Silent offscreen-as-visible fallback or retained renderer masquerading is strictly forbidden.",
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  const packetBytes = buildFn(
    positionsToUse,
    indicesToUse,
    snapshot.modelView,
    snapshot.projection,
    snapshot.color,
    width,
    height,
    snapshot.webglDepth,
    snapshot.depthTest,
    snapshot.depthWrite,
    snapshot.depthCompare,
  );

  return { packetBytes, snapshot, target: "canvas" };
}

/**
 * Prepares a binary GpuSubmissionPacket from real Three.js Mesh inputs via dynamic Wasm export.
 * Defaults to offscreen rendering unless options.target === 'canvas'.
 * Automatically selects 11-argument depth export when depth is enabled or available,
 * preserving legacy 8-argument export when depth is disabled.
 *
 * Invariants:
 * - Requires wasmModule to expose f3d_build_mesh_depth_packet or f3d_build_mesh_packet.
 * - Explicitly refuses execution if mesh requires depth but depth export is missing.
 * - If mesh is FrontSide or BackSide, routes through f3d_build_mesh_batch_cull_packet (N=1);
 *   silent DoubleSide fallback is strictly forbidden.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} wasmModule - Loaded Wasm module
 * @param {object} [options]
 * @returns {{ packetBytes: Uint8Array, snapshot: object, target: 'offscreen' | 'canvas' }}
 */
export function prepareMeshPacket(mesh, camera, width, height, wasmModule, options = {}) {
  if (options.target === "canvas") {
    return prepareCanvasMeshPacket(mesh, camera, width, height, wasmModule, options);
  }

  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (
    snapshot.side !== THREE_SIDE.DOUBLE_SIDE ||
    snapshot.colorWrite === false ||
    snapshot.vertexColors === true ||
    options.clearColor
  ) {
    const packetBytes = buildSingleMeshCullPacket(
      snapshot,
      width,
      height,
      wasmModule,
      false,
      options,
    );
    return { packetBytes, snapshot, target: "offscreen" };
  }

  const depthFn =
    wasmModule?.f3d_build_mesh_depth_packet || wasmModule?.gpu_bridge_build_mesh_depth_packet;
  const legacyFn = wasmModule?.f3d_build_mesh_packet || wasmModule?.gpu_bridge_build_mesh_packet;

  if (!wasmModule || (typeof depthFn !== "function" && typeof legacyFn !== "function")) {
    throw new Error("Invalid wasmModule: must expose f3d_build_mesh_packet export");
  }

  const requiresDepth = snapshot.depthTest === true || snapshot.depthWrite === true;
  if (requiresDepth && typeof depthFn !== "function") {
    throw new Error(
      "Mesh requires depth (depthTest or depthWrite enabled), but wasmModule is missing " +
        "f3d_build_mesh_depth_packet export. Silent no-depth fallback or retained renderer masquerading is strictly forbidden.",
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  let packetBytes;
  if (typeof depthFn === "function") {
    packetBytes = depthFn(
      positionsToUse,
      indicesToUse,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      width,
      height,
      snapshot.webglDepth,
      snapshot.depthTest,
      snapshot.depthWrite,
      snapshot.depthCompare,
    );
  } else {
    packetBytes = legacyFn(
      positionsToUse,
      indicesToUse,
      snapshot.modelView,
      snapshot.projection,
      snapshot.color,
      width,
      height,
      snapshot.webglDepth,
    );
  }

  return { packetBytes, snapshot, target: "offscreen" };
}

/**
 * Prepares a binary GpuSubmissionPacket with explicit depth settings via dynamic Wasm export.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} wasmModule - Loaded Wasm module exposing f3d_build_mesh_depth_packet
 * @param {object} [options]
 * @returns {{ packetBytes: Uint8Array, snapshot: object, target: 'offscreen' }}
 */
export function prepareMeshDepthPacket(mesh, camera, width, height, wasmModule, options = {}) {
  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (
    snapshot.side !== THREE_SIDE.DOUBLE_SIDE ||
    snapshot.colorWrite === false ||
    snapshot.vertexColors === true ||
    options.clearColor
  ) {
    const packetBytes = buildSingleMeshCullPacket(
      snapshot,
      width,
      height,
      wasmModule,
      false,
      options,
    );
    return { packetBytes, snapshot, target: "offscreen" };
  }

  const buildFn =
    wasmModule?.f3d_build_mesh_depth_packet || wasmModule?.gpu_bridge_build_mesh_depth_packet;

  if (typeof buildFn !== "function") {
    throw new Error(
      "Mesh depth packet preparation failed: wasmModule is missing f3d_build_mesh_depth_packet / gpu_bridge_build_mesh_depth_packet export. " +
        "Silent no-depth fallback or retained renderer masquerading is strictly forbidden.",
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  const packetBytes = buildFn(
    positionsToUse,
    indicesToUse,
    snapshot.modelView,
    snapshot.projection,
    snapshot.color,
    width,
    height,
    snapshot.webglDepth,
    snapshot.depthTest,
    snapshot.depthWrite,
    snapshot.depthCompare,
  );

  return { packetBytes, snapshot, target: "offscreen" };
}

/**
 * Explicit bridge execution entry: prepares and executes a Mesh render through WebGpuBridgeHost.
 *
 * Invariants:
 * - If canvasContext is provided, routes through wasmModule.f3d_build_canvas_mesh_packet or f3d_build_canvas_mesh_depth_packet.
 *   If wasmModule lacks canvas exports, EXPLICITLY REFUSES canvasContext;
 *   silent offscreen rendering as visible is strictly forbidden.
 * - If canvasContext is null/undefined, executes an honest offscreen render pass.
 *
 * @param {object} bridgeHost - Initialized WebGpuBridgeHost
 * @param {any} mesh
 * @param {any} camera
 * @param {any} [canvasContext] - HTMLCanvasElement / GPUCanvasContext (optional)
 * @param {object} wasmModule - Loaded Wasm module
 * @param {object} [options]
 * @returns {Promise<{ result: any, snapshot: object, target: 'canvas' | 'offscreen' }>}
 */
export async function renderMesh(
  bridgeHost,
  mesh,
  camera,
  canvasContext,
  wasmModule,
  options = {},
) {
  if (!bridgeHost || typeof bridgeHost.executePacket !== "function") {
    throw new Error("Invalid bridgeHost: must expose executePacket method");
  }

  const isCanvasTarget = canvasContext !== null && canvasContext !== undefined;

  const residency = getHostResidency(bridgeHost);
  const pendingCommits = [];
  const residencyContext = residency
    ? {
        residency,
        pendingCommits,
        stagedThisPass: new Map(),
        stagedBindingsThisPass: new Map(),
      }
    : null;

  const renderOptions = {
    ...options,
    [RESIDENCY_CONTEXT]: residencyContext,
  };

  let packetBytes;
  let snapshot;
  let target;

  if (isCanvasTarget) {
    const side = mesh?.material?.side ?? THREE_SIDE.FRONT_SIDE;
    const isSidedMesh = side !== THREE_SIDE.DOUBLE_SIDE;
    const colorWriteDisabled = mesh?.material?.colorWrite === false;
    const vertexColorsEnabled = mesh?.material?.vertexColors === true;
    const hasVertexColorExport =
      wasmModule && typeof wasmModule.f3d_build_mesh_batch_vertex_color_packet === "function";
    const hasColorCanvasExport =
      wasmModule && typeof wasmModule.f3d_build_mesh_batch_cull_depth_color_packet === "function";
    const hasSidedCanvasExport =
      wasmModule &&
      (hasColorCanvasExport ||
        typeof wasmModule.f3d_build_mesh_batch_cull_depth_packet === "function" ||
        typeof wasmModule.f3d_build_mesh_batch_cull_packet === "function");
    const hasLegacyCanvasExport =
      wasmModule &&
      (typeof wasmModule.f3d_build_canvas_mesh_packet === "function" ||
        typeof wasmModule.f3d_build_canvas_mesh_depth_packet === "function" ||
        typeof wasmModule.gpu_bridge_build_canvas_mesh_packet === "function" ||
        typeof wasmModule.gpu_bridge_build_canvas_mesh_depth_packet === "function");
    if (vertexColorsEnabled) {
      if (!hasVertexColorExport) {
        throw createAdmissionError(
          "INCOMPATIBLE_VERTEX_COLORS",
          "Meshes with vertexColors require Wasm export f3d_build_mesh_batch_vertex_color_packet",
        );
      }
    } else if (colorWriteDisabled) {
      if (!hasColorCanvasExport) {
        throw createAdmissionError(
          "INCOMPATIBLE_COLOR_WRITE",
          "Meshes with colorWrite=false require Wasm export f3d_build_mesh_batch_cull_depth_color_packet",
        );
      }
    } else {
      const hasCanvasExport = (isSidedMesh && hasSidedCanvasExport) || hasLegacyCanvasExport;
      if (!hasCanvasExport) {
        throw new Error(
          "renderMesh refused: canvasContext provided for visible canvas rendering, but wasmModule " +
            "does not export f3d_build_canvas_mesh_packet. Silent offscreen-as-visible rendering is strictly forbidden.",
        );
      }
    }
    const width = canvasContext?.canvas?.width ?? options.width ?? 64;
    const height = canvasContext?.canvas?.height ?? options.height ?? 64;
    const prep = prepareCanvasMeshPacket(mesh, camera, width, height, wasmModule, renderOptions);
    packetBytes = prep.packetBytes;
    snapshot = prep.snapshot;
    target = "canvas";
  } else {
    // Honest offscreen execution
    const width = options.width ?? 64;
    const height = options.height ?? 64;
    const prep = prepareMeshPacket(mesh, camera, width, height, wasmModule, {
      ...renderOptions,
      target: "offscreen",
    });
    packetBytes = prep.packetBytes;
    snapshot = prep.snapshot;
    target = "offscreen";
  }

  const onSubmitted = () => {
    pendingCommits.forEach((commit) => commit());
  };

  const result = await bridgeHost.executePacket(
    packetBytes,
    isCanvasTarget ? canvasContext : null,
    onSubmitted,
  );
  return { result, snapshot, target };
}

/**
 * Prepares a variable-length binary submission packet for an explicit ordered list of compatible meshes.
 *
 * Invariants:
 * - Meshes must be an array with length >= 1. Empty or non-array inputs are strictly rejected.
 * - Extracts per-draw snapshots via extractMeshRenderData, preserving per-draw transforms and colors.
 * - Refuses incompatible shared depth settings across meshes rather than silently dropping differences.
 * - Flattens expanded positions and concatenates per-draw uniforms (modelViews, colors, vertexCounts).
 * - Invokes wasmModule.f3d_build_mesh_batch_packet (or gpu_bridge_build_mesh_batch_packet).
 * - Target selection: options.target === 'canvas' sets canvas: true, otherwise offscreen.
 *
 * @param {Array<any>} meshes - Explicit ordered list of compatible Three.js Mesh instances
 * @param {any} camera - Shared Three.js Camera instance
 * @param {number} width - Viewport width
 * @param {number} height - Viewport height
 * @param {object} wasmModule - Loaded Wasm module exposing f3d_build_mesh_batch_packet
 * @param {object} [options] - Additional options (target, canvas, sourceBackend)
 * @returns {{ packetBytes: Uint8Array, snapshots: Array<object>, meshCount: number, totalVertices: number, target: 'canvas' | 'offscreen' }}
 */
export function prepareMeshBatchPacket(meshes, camera, width, height, wasmModule, options = {}) {
  if (!Array.isArray(meshes) || meshes.length === 0) {
    throw createAdmissionError("EMPTY_MESH_BATCH");
  }

  // If options.autoUpdate is explicitly true (standalone batch preparation requesting auto-update),
  // update world matrices according to Three.js boundaries without repeating if autoUpdate is false.
  if (options.autoUpdate === true) {
    if (
      camera.parent === null &&
      camera.matrixWorldAutoUpdate === true &&
      typeof camera.updateMatrixWorld === "function"
    ) {
      camera.updateMatrixWorld();
    }
    for (const mesh of meshes) {
      if (mesh.matrixWorldAutoUpdate === true && typeof mesh.updateMatrixWorld === "function") {
        mesh.updateMatrixWorld();
      }
    }
  }

  const vertexColorClearBatchFn = wasmModule?.f3d_build_mesh_batch_vertex_color_clear_packet;
  const vertexColorBatchFn = wasmModule?.f3d_build_mesh_batch_vertex_color_packet;
  const cullDepthColorBatchFn = wasmModule?.f3d_build_mesh_batch_cull_depth_color_packet;
  const cullDepthBatchFn = wasmModule?.f3d_build_mesh_batch_cull_depth_packet;
  const cullBatchFn = wasmModule?.f3d_build_mesh_batch_cull_packet;
  const legacyBatchFn = wasmModule?.f3d_build_mesh_batch_packet;

  if (
    typeof vertexColorClearBatchFn !== "function" &&
    typeof vertexColorBatchFn !== "function" &&
    typeof cullDepthColorBatchFn !== "function" &&
    typeof cullDepthBatchFn !== "function" &&
    typeof cullBatchFn !== "function" &&
    typeof legacyBatchFn !== "function"
  ) {
    throw new Error(
      "Mesh batch packet preparation failed: wasmModule is missing f3d_build_mesh_batch_packet, f3d_build_mesh_batch_cull_packet, f3d_build_mesh_batch_cull_depth_packet, f3d_build_mesh_batch_cull_depth_color_packet, f3d_build_mesh_batch_vertex_color_packet, or f3d_build_mesh_batch_vertex_color_clear_packet export.",
    );
  }

  // Extract snapshot for each mesh in the explicit order
  const snapshots = meshes.map((mesh, index) => {
    try {
      return extractMeshRenderData(mesh, camera, width, height, options);
    } catch (err) {
      const code = err.reason ?? "ADMISSION_REJECTED";
      const batchErr = new Error(
        `${code}: Mesh batch admission rejected at index ${index}: ${err.message}`,
      );
      batchErr.reason = code;
      throw batchErr;
    }
  });

  // Verify shared batch pipeline configuration: webglDepth (camera coordinate system)
  // and check for mixed depth settings across meshes
  const first = snapshots[0];
  const sharedDepthTest = first.depthTest;
  const sharedDepthWrite = first.depthWrite;
  const sharedDepthCompare = first.depthCompare;
  const sharedWebglDepth = first.webglDepth;

  let firstDepthMismatch = "";
  for (let i = 1; i < snapshots.length; i++) {
    const s = snapshots[i];
    if (s.webglDepth !== sharedWebglDepth) {
      throw createAdmissionError(
        "INCOMPATIBLE_BATCH_DEPTH",
        `mesh 0 has webglDepth=${sharedWebglDepth}, mesh ${i} has webglDepth=${s.webglDepth}`,
      );
    }
    const field =
      s.depthTest !== sharedDepthTest
        ? "depthTest"
        : s.depthWrite !== sharedDepthWrite
          ? "depthWrite"
          : s.depthCompare !== sharedDepthCompare
            ? "depthCompare"
            : null;
    if (field && !firstDepthMismatch) {
      firstDepthMismatch = `mesh 0 has ${field}=${first[field]}, mesh ${i} has ${field}=${s[field]}`;
    }
  }

  const hasVertexColors = snapshots.some((s) => s.vertexColors === true);
  const hasClearColor = Boolean(options.clearColor);
  if (hasClearColor) {
    if (typeof vertexColorClearBatchFn !== "function") {
      throw createAdmissionError(
        "INCOMPATIBLE_BACKGROUND_CLEAR",
        "scene.background Color requires Wasm export f3d_build_mesh_batch_vertex_color_clear_packet",
      );
    }
  } else if (hasVertexColors) {
    if (typeof vertexColorBatchFn !== "function") {
      throw createAdmissionError(
        "INCOMPATIBLE_VERTEX_COLORS",
        "Meshes with vertexColors require Wasm export f3d_build_mesh_batch_vertex_color_packet",
      );
    }
  } else {
    // Uncolored batch must not be admitted by or route through vertexColor export
    const hasColorWriteDisabled = snapshots.some((s) => s.colorWrite === false);
    if (hasColorWriteDisabled && typeof cullDepthColorBatchFn !== "function") {
      throw createAdmissionError(
        "INCOMPATIBLE_COLOR_WRITE",
        "Meshes with colorWrite=false require Wasm export f3d_build_mesh_batch_cull_depth_color_packet",
      );
    }

    if (
      firstDepthMismatch &&
      typeof cullDepthColorBatchFn !== "function" &&
      typeof cullDepthBatchFn !== "function"
    ) {
      throw createAdmissionError(
        "INCOMPATIBLE_BATCH_DEPTH",
        `${firstDepthMismatch}; wasmModule lacks f3d_build_mesh_batch_cull_depth_packet export. Silent uniform-depth fallback is strictly forbidden.`,
      );
    }

    if (
      typeof cullDepthColorBatchFn !== "function" &&
      typeof cullDepthBatchFn !== "function" &&
      typeof cullBatchFn !== "function" &&
      typeof legacyBatchFn !== "function"
    ) {
      throw new Error(
        "Mesh batch packet preparation failed: wasmModule is missing f3d_build_mesh_batch_packet, f3d_build_mesh_batch_cull_packet, f3d_build_mesh_batch_cull_depth_packet, or f3d_build_mesh_batch_cull_depth_color_packet export.",
      );
    }
  }

  // Material side / cull / depth checks and typed array creation
  const n = snapshots.length;
  const cullModes = new Uint8Array(n);
  const frontFaces = new Uint8Array(n);
  const depthTests = new Uint8Array(n);
  const depthWrites = new Uint8Array(n);
  const depthCompares = new Uint32Array(n);
  const colorWrites = new Uint8Array(n);
  let hasNonDoubleSide = false;

  for (let i = 0; i < n; i++) {
    const s = snapshots[i];
    if (s.side !== THREE_SIDE.DOUBLE_SIDE) {
      hasNonDoubleSide = true;
    }
    if (
      s.cullMode !== CULL_MODE_WIRE.NONE &&
      s.cullMode !== CULL_MODE_WIRE.FRONT &&
      s.cullMode !== CULL_MODE_WIRE.BACK
    ) {
      throw createAdmissionError(
        "INVALID_CULL_MODE",
        `mesh ${i} has invalid cullMode ${s.cullMode}`,
      );
    }
    if (s.frontFace !== FRONT_FACE_WIRE.CCW && s.frontFace !== FRONT_FACE_WIRE.CW) {
      throw createAdmissionError(
        "INVALID_FRONT_FACE",
        `mesh ${i} has invalid frontFace ${s.frontFace}`,
      );
    }
    cullModes[i] = s.cullMode;
    frontFaces[i] = s.frontFace;
    depthTests[i] = s.depthTest ? 1 : 0;
    depthWrites[i] = s.depthWrite ? 1 : 0;
    depthCompares[i] = s.depthCompare;
    colorWrites[i] = s.colorWrite ? 1 : 0;
  }

  if (
    !hasVertexColors &&
    hasNonDoubleSide &&
    typeof cullDepthColorBatchFn !== "function" &&
    typeof cullDepthBatchFn !== "function" &&
    typeof cullBatchFn !== "function"
  ) {
    throw createAdmissionError(
      "MISSING_CULL_EXPORT",
      "Mesh batch contains FrontSide or BackSide meshes, but wasmModule is missing f3d_build_mesh_batch_cull_packet export. Silent DoubleSide fallback is strictly forbidden.",
    );
  }

  // Calculate total expanded vertex count across all meshes
  let totalVertices = 0;
  for (let i = 0; i < snapshots.length; i++) {
    totalVertices += snapshots[i].expandedPositions.length / 3;
  }

  const flatPositions = new Float32Array(totalVertices * 3);
  const vertexCounts = new Uint32Array(n);
  const modelViews = new Float64Array(n * 16);
  const colors = new Float32Array(n * 4);
  const projection = first.projection; // Shared camera projection (16 f64)

  let posOffset = 0;
  for (let i = 0; i < n; i++) {
    const s = snapshots[i];
    const vertCount = s.expandedPositions.length / 3;
    vertexCounts[i] = vertCount;

    flatPositions.set(s.expandedPositions, posOffset);
    posOffset += s.expandedPositions.length;

    modelViews.set(s.modelView, i * 16);
    colors.set(s.color, i * 4);
  }

  const isCanvas = options.target === "canvas";

  let packetBytes;
  try {
    if (hasClearColor && typeof vertexColorClearBatchFn === "function") {
      const flatVertexColors = new Float32Array(totalVertices * 4);
      let colorOffset = 0;
      for (let i = 0; i < n; i++) {
        const s = snapshots[i];
        const vertCount = s.expandedPositions.length / 3;
        if (s.hasVertexColors && s.expandedVertexColors) {
          flatVertexColors.set(s.expandedVertexColors, colorOffset);
        } else {
          flatVertexColors.fill(1.0, colorOffset, colorOffset + vertCount * 4);
        }
        colorOffset += vertCount * 4;
      }
      packetBytes = vertexColorClearBatchFn(
        flatPositions,
        vertexCounts,
        modelViews,
        projection,
        colors,
        cullModes,
        frontFaces,
        depthTests,
        depthWrites,
        depthCompares,
        colorWrites,
        width,
        height,
        sharedWebglDepth,
        isCanvas,
        flatVertexColors,
        options.clearColor,
      );
    } else if (hasVertexColors && typeof vertexColorBatchFn === "function") {
      const flatVertexColors = new Float32Array(totalVertices * 4);
      let colorOffset = 0;
      for (let i = 0; i < n; i++) {
        const s = snapshots[i];
        const vertCount = s.expandedPositions.length / 3;
        if (s.hasVertexColors && s.expandedVertexColors) {
          flatVertexColors.set(s.expandedVertexColors, colorOffset);
        } else {
          flatVertexColors.fill(1.0, colorOffset, colorOffset + vertCount * 4);
        }
        colorOffset += vertCount * 4;
      }
      packetBytes = vertexColorBatchFn(
        flatPositions,
        vertexCounts,
        modelViews,
        projection,
        colors,
        cullModes,
        frontFaces,
        depthTests,
        depthWrites,
        depthCompares,
        colorWrites,
        width,
        height,
        sharedWebglDepth,
        isCanvas,
        flatVertexColors,
      );
    } else if (typeof cullDepthColorBatchFn === "function") {
      packetBytes = cullDepthColorBatchFn(
        flatPositions,
        vertexCounts,
        modelViews,
        projection,
        colors,
        cullModes,
        frontFaces,
        depthTests,
        depthWrites,
        depthCompares,
        colorWrites,
        width,
        height,
        sharedWebglDepth,
        isCanvas,
      );
    } else if (typeof cullDepthBatchFn === "function") {
      packetBytes = cullDepthBatchFn(
        flatPositions,
        vertexCounts,
        modelViews,
        projection,
        colors,
        cullModes,
        frontFaces,
        depthTests,
        depthWrites,
        depthCompares,
        width,
        height,
        sharedWebglDepth,
        isCanvas,
      );
    } else if (typeof cullBatchFn === "function") {
      packetBytes = cullBatchFn(
        flatPositions,
        vertexCounts,
        modelViews,
        projection,
        colors,
        cullModes,
        frontFaces,
        width,
        height,
        sharedWebglDepth,
        sharedDepthTest,
        sharedDepthWrite,
        sharedDepthCompare,
        isCanvas,
      );
    } else {
      packetBytes = legacyBatchFn(
        flatPositions,
        vertexCounts,
        modelViews,
        projection,
        colors,
        width,
        height,
        sharedWebglDepth,
        sharedDepthTest,
        sharedDepthWrite,
        sharedDepthCompare,
        isCanvas,
      );
    }
  } catch (err) {
    const msg = err?.message ?? String(err);
    if (
      msg.includes("EmptyMeshList") ||
      msg.includes("must contain at least one mesh") ||
      msg.includes("EMPTY_MESH_BATCH")
    ) {
      throw createAdmissionError("EMPTY_MESH_BATCH");
    }
    if (msg.includes("INVALID_CULL_MODE")) {
      throw createAdmissionError("INVALID_CULL_MODE", msg);
    }
    if (msg.includes("INVALID_FRONT_FACE")) {
      throw createAdmissionError("INVALID_FRONT_FACE", msg);
    }
    if (msg.includes("INVALID_COLOR_WRITE") || msg.includes("InvalidColorWrite")) {
      throw createAdmissionError("INCOMPATIBLE_COLOR_WRITE", msg);
    }
    throw err;
  }

  return {
    packetBytes,
    snapshots,
    meshCount: n,
    totalVertices,
    cullModes,
    frontFaces,
    depthTests,
    depthWrites,
    depthCompares,
    colorWrites,
    target: isCanvas ? "canvas" : "offscreen",
  };
}

/**
 * Executes an explicit ordered list of compatible meshes through WebGpuBridgeHost.
 * If canvasContext is provided, targets the visible canvas swapchain (canvas: true).
 * Otherwise, executes honest offscreen rendering.
 *
 * @param {object} bridgeHost - Initialized WebGpuBridgeHost
 * @param {Array<any>} meshes - Ordered list of THREE.Mesh instances
 * @param {any} camera - THREE.Camera instance
 * @param {any} [canvasContext] - HTMLCanvasElement / GPUCanvasContext (optional)
 * @param {object} wasmModule - Loaded Wasm module
 * @param {object} [options]
 * @returns {Promise<{ result: any, snapshots: Array<object>, meshCount: number, totalVertices: number, cullModes: Uint8Array, frontFaces: Uint8Array, target: 'canvas' | 'offscreen' }>}
 */
export async function renderMeshBatch(
  bridgeHost,
  meshes,
  camera,
  canvasContext,
  wasmModule,
  options = {},
) {
  if (!bridgeHost || typeof bridgeHost.executePacket !== "function") {
    throw new Error("Invalid bridgeHost: must expose executePacket method");
  }

  if (!Array.isArray(meshes) || meshes.length === 0) {
    throw createAdmissionError("EMPTY_MESH_BATCH");
  }

  const isCanvasTarget = canvasContext !== null && canvasContext !== undefined;
  const width = canvasContext?.canvas?.width ?? options.width ?? 64;
  const height = canvasContext?.canvas?.height ?? options.height ?? 64;

  const residency = getHostResidency(bridgeHost);
  const pendingCommits = [];
  const residencyContext = residency
    ? {
        residency,
        pendingCommits,
        stagedThisPass: new Map(),
        stagedBindingsThisPass: new Map(),
      }
    : null;

  const batchResult = prepareMeshBatchPacket(meshes, camera, width, height, wasmModule, {
    ...options,
    target: isCanvasTarget ? "canvas" : "offscreen",
    [RESIDENCY_CONTEXT]: residencyContext,
  });

  const onSubmitted = () => {
    pendingCommits.forEach((commit) => commit());
  };

  const result = await bridgeHost.executePacket(
    batchResult.packetBytes,
    isCanvasTarget ? canvasContext : null,
    onSubmitted,
  );
  return {
    result,
    snapshots: batchResult.snapshots,
    meshCount: batchResult.meshCount,
    totalVertices: batchResult.totalVertices,
    cullModes: batchResult.cullModes,
    frontFaces: batchResult.frontFaces,
    depthTests: batchResult.depthTests,
    depthWrites: batchResult.depthWrites,
    depthCompares: batchResult.depthCompares,
    target: batchResult.target,
  };
}

/**
 * Renders admitted Three.js Mesh instances in a Scene using a single batch packet.
 *
 * Traversal & Admission:
 * - Updates scene/camera matrices at the pinned renderer's automatic update boundaries.
 * - Traverses the scene graph to collect visible THREE.Mesh instances.
 * - Evaluates each mesh via canAdmitMesh(obj, camera, options), honoring sourceBackend rules.
 * - Sorts admitted opaque meshes with the selected source backend's ordering.
 * - Any unsupported visible content refuses the whole submission before packet construction.
 *
 * Submission:
 * - If admitted set is empty, explicitly refuses without building or submitting a packet,
 *   returning { admitted: [], refused }.
 * - If admitted meshes exist, builds ONE batch via prepareMeshBatchPacket and executes
 *   it via bridgeHost.executePacket.
 * - Returns { admitted: [uuid...], refused: [{ uuid, reason }] }.
 *
 * @param {object} bridgeHost - Initialized WebGpuBridgeHost exposing executePacket
 * @param {any} scene - THREE.Scene / Object3D hierarchy
 * @param {any} camera - THREE.Camera instance
 * @param {any} [canvasContext] - HTMLCanvasElement / GPUCanvasContext (optional)
 * @param {object} wasmModule - Loaded Wasm module exposing f3d_build_mesh_batch_packet
 * @param {object} [options] - Additional options (sourceBackend, width, height, etc.)
 * @returns {Promise<{ admitted: Array<string>, refused: Array<{ uuid: string, reason: string }> }>}
 */
export async function renderScene(
  bridgeHost,
  scene,
  camera,
  canvasContext,
  wasmModule,
  options = {},
) {
  if (!bridgeHost || typeof bridgeHost.executePacket !== "function") {
    throw new Error("Invalid bridgeHost: must expose executePacket method");
  }
  if (!scene || typeof scene.traverse !== "function") {
    throw new Error("Invalid scene: must expose traverse method");
  }
  if (!camera || !camera.isCamera) {
    throw createAdmissionError("INVALID_CAMERA");
  }

  // Update world matrices according to Three.js renderer source boundaries
  // (Renderer.js:1755, 3677 / WebGLRenderer.js:1663, 1667)
  if (scene.matrixWorldAutoUpdate === true && typeof scene.updateMatrixWorld === "function") {
    scene.updateMatrixWorld();
  }
  if (
    camera.parent === null &&
    camera.matrixWorldAutoUpdate === true &&
    typeof camera.updateMatrixWorld === "function"
  ) {
    camera.updateMatrixWorld();
  }

  const projScreenMatrix = new Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  const frustum = new Frustum();
  // WebGLRenderer.js:1687 unconditionally uses WebGLCoordinateSystem for frustum extraction against projScreenMatrix.
  const isExplicitWebGL = options?.sourceBackend?.toLowerCase() === SOURCE_BACKEND.WEBGL;
  const coordinateSystem = isExplicitWebGL
    ? COORDINATE_SYSTEM.WEBGL
    : (camera.coordinateSystem ?? COORDINATE_SYSTEM.WEBGL);
  frustum.setFromProjectionMatrix(
    projScreenMatrix,
    coordinateSystem,
    Boolean(camera.reversedDepth),
  );
  const vector4 = new Vector4();

  const admittedItems = [];
  const refused = [];

  // Scene-level unsupported properties check (Root Mail 14355)
  if (
    scene.onBeforeRender !== DEFAULT_OBJECT3D_ON_BEFORE_RENDER ||
    scene.onAfterRender !== DEFAULT_OBJECT3D_ON_AFTER_RENDER
  ) {
    refused.push(createRefusalItem(scene.uuid, "UNSUPPORTED_CALLBACK"));
  }
  if (scene.background !== null && scene.background !== undefined) {
    if (!scene.background?.isColor) {
      refused.push(
        createRefusalItem(
          scene.uuid,
          "UNSUPPORTED_SCENE_FEATURE",
          `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.background is not supported`,
        ),
      );
    }
  }
  if (scene.fog !== null && scene.fog !== undefined) {
    refused.push(
      createRefusalItem(
        scene.uuid,
        "UNSUPPORTED_SCENE_FEATURE",
        `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.fog is not supported`,
      ),
    );
  }
  if (scene.overrideMaterial !== null && scene.overrideMaterial !== undefined) {
    refused.push(
      createRefusalItem(
        scene.uuid,
        "UNSUPPORTED_SCENE_FEATURE",
        `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.overrideMaterial is not supported`,
      ),
    );
  }
  if (scene.environment !== null && scene.environment !== undefined) {
    refused.push(
      createRefusalItem(
        scene.uuid,
        "UNSUPPORTED_SCENE_FEATURE",
        `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.environment is not supported`,
      ),
    );
  }

  function isNodeVisible(node) {
    let cur = node;
    while (cur) {
      if (cur.visible === false) return false;
      cur = cur.parent;
    }
    return true;
  }

  function isNodeInCameraLayers(node) {
    if (!camera.layers || !node.layers) return true;
    return camera.layers.test(node.layers);
  }

  // Traverse scene to collect visible in-layer renderables; cull hidden/layer-filtered objects legitimately
  scene.traverse((obj) => {
    if (!obj || obj === scene) {
      return;
    }

    // Ignore hidden or camera-layer-filtered content legitimately (Root Mail 14355)
    if (!isNodeVisible(obj) || !isNodeInCameraLayers(obj)) {
      return;
    }
    if (obj.material?.visible === false) return;

    // Visible, in-layer non-mesh renderables
    if (
      obj.isLine ||
      obj.isLineSegments ||
      obj.isLineLoop ||
      obj.isPoints ||
      obj.isSprite ||
      obj.isLight ||
      (obj.geometry && !obj.isMesh)
    ) {
      refused.push(
        createRefusalItem(
          obj.uuid,
          "UNSUPPORTED_RENDERABLE",
          `${ADMISSION_REJECTION.UNSUPPORTED_RENDERABLE}: ${obj.type || "Non-mesh renderable"} is not supported`,
        ),
      );
      return;
    }

    // Visible, in-layer Mesh
    if (obj.isMesh) {
      const admission = canAdmitMesh(obj, camera, options);
      if (!admission.admitted) {
        refused.push(
          createRefusalItem(obj.uuid, admission.code ?? "ADMISSION_REJECTED", admission.reason),
        );
        return;
      }

      // Three.js frustum culling: WebGLRenderer:1914 / Renderer:3268
      if (obj.frustumCulled) {
        if (obj.intersectsFrustum !== DEFAULT_MESH_INTERSECTS_FRUSTUM) {
          refused.push(
            createRefusalItem(
              obj.uuid,
              "UNSUPPORTED_CALLBACK",
              `${ADMISSION_REJECTION.UNSUPPORTED_CALLBACK}: custom intersectsFrustum is not supported`,
            ),
          );
          return;
        }
        if (!obj.intersectsFrustum(frustum)) {
          return;
        }
      }

      // Calculate projected z (Three.js WebGLRenderer.js:1924-1936 / Renderer.js:3300-3306)
      let z = 0;
      const geom = obj.geometry;
      if (options.sortObjects !== false && geom) {
        if (geom.boundingSphere === null && typeof geom.computeBoundingSphere === "function") {
          geom.computeBoundingSphere();
        }
        if (geom.boundingSphere) {
          vector4.copy(geom.boundingSphere.center);
        } else {
          vector4.set(0, 0, 0, 1);
        }
        vector4.applyMatrix4(obj.matrixWorld).applyMatrix4(projScreenMatrix);
        z = vector4.z;
      }

      admittedItems.push({
        mesh: obj,
        groupOrder: getMeshGroupOrder(obj, camera, scene),
        renderOrder: Number.isFinite(obj.renderOrder) ? obj.renderOrder : 0,
        z,
        id: obj.id,
        materialId: obj.material?.id ?? 0,
        materialVariant: 0,
      });
    }
  });

  // Verify shared batch pipeline configuration across admitted meshes in scene:
  // If wasmModule lacks per-mesh depth export, refuse mixed depth settings;
  // if per-mesh depth export is present, admit mixed depth into the batch.
  const hasBackgroundColor = Boolean(scene.background && scene.background.isColor);
  const anyHasVertexColors = admittedItems.some(
    (item) => item.mesh.material?.vertexColors === true,
  );
  const hasVertexColorClearExport =
    wasmModule && typeof wasmModule.f3d_build_mesh_batch_vertex_color_clear_packet === "function";
  const hasSceneClearExport =
    wasmModule && typeof wasmModule.f3d_build_scene_clear_packet === "function";
  const hasVertexColorExport =
    wasmModule && typeof wasmModule.f3d_build_mesh_batch_vertex_color_packet === "function";
  const hasCullDepthColorExport =
    wasmModule && typeof wasmModule.f3d_build_mesh_batch_cull_depth_color_packet === "function";
  const hasCullDepthExport =
    wasmModule && typeof wasmModule.f3d_build_mesh_batch_cull_depth_packet === "function";

  // The new clear+vertex export accepts cull, depth, colorWrite, vertexColors, and clear_color.
  // When active for a background scene, it satisfies depth, colorWrite, and vertexColors capabilities.
  const hasActiveClearExport = hasBackgroundColor && hasVertexColorClearExport;
  const hasVertexColorCapability = hasVertexColorExport || hasActiveClearExport;
  const hasColorExport =
    hasActiveClearExport ||
    (anyHasVertexColors ? hasVertexColorCapability : hasCullDepthColorExport);
  const hasStateExport =
    hasActiveClearExport ||
    (anyHasVertexColors ? hasVertexColorCapability : hasCullDepthColorExport || hasCullDepthExport);

  if (!hasStateExport && admittedItems.length > 1) {
    const firstMat = admittedItems[0].mesh.material;
    const sharedDepthTest = firstMat?.depthTest !== false;
    const sharedDepthWrite = firstMat?.depthWrite !== false;
    const sharedDepthFunc = firstMat?.depthFunc ?? LessEqualDepth;

    for (let i = 1; i < admittedItems.length; i++) {
      const mat = admittedItems[i].mesh.material;
      const dt = mat?.depthTest !== false;
      const dw = mat?.depthWrite !== false;
      const df = mat?.depthFunc ?? LessEqualDepth;

      if (dt !== sharedDepthTest || dw !== sharedDepthWrite || df !== sharedDepthFunc) {
        refused.push(
          createRefusalItem(
            admittedItems[i].mesh.uuid,
            "INCOMPATIBLE_BATCH_DEPTH",
            `${ADMISSION_REJECTION.INCOMPATIBLE_BATCH_DEPTH}: scene meshes have conflicting depth settings`,
          ),
        );
      }
    }
  }

  // Verify vertexColors export availability across admitted meshes in scene:
  // If wasmModule lacks vertex color capability, refuse meshes with vertexColors=true
  if (!hasVertexColorCapability) {
    for (let i = 0; i < admittedItems.length; i++) {
      const mat = admittedItems[i].mesh.material;
      if (mat?.vertexColors === true) {
        refused.push(
          createRefusalItem(
            admittedItems[i].mesh.uuid,
            "INCOMPATIBLE_VERTEX_COLORS",
            `${ADMISSION_REJECTION.INCOMPATIBLE_VERTEX_COLORS}: material.vertexColors=true requires f3d_build_mesh_batch_vertex_color_packet`,
          ),
        );
      }
    }
  }

  // Verify colorWrite export availability across admitted meshes in scene:
  // If wasmModule lacks f3d_build_mesh_batch_cull_depth_color_packet, refuse meshes with colorWrite=false
  if (!hasColorExport) {
    for (let i = 0; i < admittedItems.length; i++) {
      const mat = admittedItems[i].mesh.material;
      if (mat?.colorWrite === false) {
        refused.push(
          createRefusalItem(
            admittedItems[i].mesh.uuid,
            "INCOMPATIBLE_COLOR_WRITE",
            `${ADMISSION_REJECTION.INCOMPATIBLE_COLOR_WRITE}: material.colorWrite=false requires f3d_build_mesh_batch_cull_depth_color_packet`,
          ),
        );
      }
    }
  }

  // Verify background clear color export availability:
  // If scene has admitted meshes and background Color, requires f3d_build_mesh_batch_vertex_color_clear_packet.
  // If scene has zero admitted meshes and background Color, requires f3d_build_scene_clear_packet.
  if (hasBackgroundColor) {
    if (admittedItems.length > 0 && !hasVertexColorClearExport) {
      refused.push(
        createRefusalItem(
          scene.uuid,
          "INCOMPATIBLE_BACKGROUND_CLEAR",
          `${ADMISSION_REJECTION.INCOMPATIBLE_BACKGROUND_CLEAR}: scene.background Color requires f3d_build_mesh_batch_vertex_color_clear_packet`,
        ),
      );
    } else if (admittedItems.length === 0 && !hasSceneClearExport) {
      refused.push(
        createRefusalItem(
          scene.uuid,
          "INCOMPATIBLE_BACKGROUND_CLEAR",
          `${ADMISSION_REJECTION.INCOMPATIBLE_BACKGROUND_CLEAR}: empty scene with scene.background Color requires f3d_build_scene_clear_packet`,
        ),
      );
    }
  }

  // Refuse WHOLE submission on any visible unsupported renderable or scene effect (Root Mail 14355)
  if (refused.length > 0) {
    const response = {
      admitted: [],
      refused,
    };
    const primaryCode = refused[0].code ?? "UNSUPPORTED_SCENE_CONTENT";
    const primaryReason = refused[0].reason ?? ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE;
    Object.defineProperty(response, "reason", {
      value: primaryCode,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(response, "reasonMessage", {
      value: primaryReason,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(response, "refusalReason", {
      value: primaryCode,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    return response;
  }

  // Empty admitted set:
  // If scene has a background Color, submit a clear packet using f3d_build_scene_clear_packet.
  // Otherwise, retain ordinary empty-batch refusal without submitting.
  if (admittedItems.length === 0) {
    if (!hasBackgroundColor) {
      const response = {
        admitted: [],
        refused: [],
      };
      Object.defineProperty(response, "reason", {
        value: "EMPTY_MESH_BATCH",
        enumerable: false,
        writable: true,
        configurable: true,
      });
      Object.defineProperty(response, "reasonMessage", {
        value: ADMISSION_REJECTION.EMPTY_MESH_BATCH,
        enumerable: false,
        writable: true,
        configurable: true,
      });
      Object.defineProperty(response, "refusalReason", {
        value: "EMPTY_MESH_BATCH",
        enumerable: false,
        writable: true,
        configurable: true,
      });
      return response;
    }

    const isCanvasTarget = canvasContext !== null && canvasContext !== undefined;
    const width = canvasContext?.canvas?.width ?? options.width ?? 64;
    const height = canvasContext?.canvas?.height ?? options.height ?? 64;

    if (
      !Number.isInteger(width) ||
      width <= 0 ||
      width > 0xffffffff ||
      !Number.isInteger(height) ||
      height <= 0 ||
      height > 0xffffffff
    ) {
      throw createAdmissionError("INVALID_DIMENSIONS");
    }

    const clearColor = convertBackgroundColorToClearColor(scene.background, options);

    const packetBytes = wasmModule.f3d_build_scene_clear_packet(
      width,
      height,
      clearColor,
      isCanvasTarget,
    );

    const result = await bridgeHost.executePacket(
      packetBytes,
      isCanvasTarget ? canvasContext : null,
    );

    const response = {
      admitted: [],
      refused: [],
    };

    if (result !== undefined) {
      Object.defineProperty(response, "result", {
        value: result,
        enumerable: false,
        writable: true,
        configurable: true,
      });
    }

    return response;
  }

  // Sort admitted meshes: If options.sortObjects !== false (default true), sort per sourceBackend
  // WebGL: groupOrder -> renderOrder -> material.id -> materialVariant -> projected z -> id
  // WebGPU: groupOrder -> renderOrder -> projected z -> id
  if (options.sortObjects !== false) {
    const sourceBackend = (options.sourceBackend ?? SOURCE_BACKEND.WEBGPU).toLowerCase();
    const sortFn = sourceBackend === SOURCE_BACKEND.WEBGL ? painterSortWebGL : painterSortWebGPU;
    admittedItems.sort(sortFn);
  }

  const admittedMeshes = admittedItems.map((item) => item.mesh);
  const admitted = admittedMeshes.map((m) => m.uuid);

  const isCanvasTarget = canvasContext !== null && canvasContext !== undefined;
  const width = canvasContext?.canvas?.width ?? options.width ?? 64;
  const height = canvasContext?.canvas?.height ?? options.height ?? 64;

  const residency = getHostResidency(bridgeHost);
  const pendingCommits = [];
  const residencyContext = residency
    ? {
        residency,
        pendingCommits,
        stagedThisPass: new Map(),
        stagedBindingsThisPass: new Map(),
      }
    : null;

  const clearColor = hasBackgroundColor
    ? convertBackgroundColorToClearColor(scene.background, options)
    : (options.clearColor ?? null);

  const batchResult = prepareMeshBatchPacket(admittedMeshes, camera, width, height, wasmModule, {
    ...options,
    autoUpdate: false,
    target: isCanvasTarget ? "canvas" : "offscreen",
    clearColor,
    [RESIDENCY_CONTEXT]: residencyContext,
  });

  const onSubmitted = () => {
    pendingCommits.forEach((commit) => commit());
  };

  const result = await bridgeHost.executePacket(
    batchResult.packetBytes,
    isCanvasTarget ? canvasContext : null,
    onSubmitted,
  );

  const response = {
    admitted,
    refused,
  };

  if (result !== undefined) {
    Object.defineProperty(response, "result", {
      value: result,
      enumerable: false,
      writable: true,
      configurable: true,
    });
  }

  return response;
}

/**
 * Truthful retained fallback route for full scenes and unsupported features.
 * Retains JavaScript ownership and issues calls directly to the retained Three.js renderer.
 *
 * @param {any} renderer - Upstream WebGLRenderer or WebGPURenderer
 * @param {any} scene
 * @param {any} camera
 * @returns {{ implementationOwner: 'retained-js', rendered: boolean }}
 */
export function renderRetainedFallback(renderer, scene, camera) {
  if (!renderer || typeof renderer.render !== "function") {
    throw new Error("Cannot execute retained fallback: renderer.render is not a function");
  }
  renderer.render(scene, camera);
  return {
    implementationOwner: "retained-js",
    rendered: true,
  };
}
