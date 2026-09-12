/**
 * @file tools/compat/mesh_adapter.mjs
 * Explicit experimental dynamic Mesh adapter for FrankenThreeD (f3d-05.6 / 19:26Z product wave).
 *
 * Extracts dynamic input from real pinned Three.js Mesh + Camera instances into
 * isolated typed arrays for the Wasm/WebGPU execution core:
 * - BufferGeometry position (itemSize 3, non-interleaved, unnormalized) + optional index + drawRange
 * - Opaque untextured MeshBasicMaterial color (Float32Array[4]), DoubleSide only
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
  Material,
  Object3D,
  Matrix4,
  Vector4,
  NeverDepth,
  AlwaysDepth,
  LessDepth,
  LessEqualDepth,
  EqualDepth,
  GreaterEqualDepth,
  GreaterDepth,
  NotEqualDepth,
} from '../../upstream/three.js/build/three.module.js';

// Base prototypes default hooks for override detection (avoiding function.toString)
const DEFAULT_MATERIAL_ON_BEFORE_COMPILE = Material.prototype.onBeforeCompile;
const DEFAULT_MATERIAL_CUSTOM_PROGRAM_CACHE_KEY = Material.prototype.customProgramCacheKey;
const DEFAULT_MATERIAL_ON_BEFORE_RENDER = Material.prototype.onBeforeRender;

const DEFAULT_OBJECT3D_ON_BEFORE_RENDER = Object3D.prototype.onBeforeRender;
const DEFAULT_OBJECT3D_ON_AFTER_RENDER = Object3D.prototype.onAfterRender;

export const ADMISSION_REJECTION = Object.freeze({
  NOT_A_MESH: 'NOT_A_MESH: Object is not an instance of THREE.Mesh',
  UNSUPPORTED_MESH_SUBCLASS: 'UNSUPPORTED_MESH_SUBCLASS: InstancedMesh, SkinnedMesh, and BatchedMesh are not supported in this scalar slice',
  NOT_VISIBLE: 'NOT_VISIBLE: Mesh is not visible (mesh.visible === false)',
  MATERIAL_NOT_VISIBLE: 'MATERIAL_NOT_VISIBLE: Material is not visible (material.visible === false)',
  UNSUPPORTED_CALLBACK: 'UNSUPPORTED_CALLBACK: onBeforeRender and onAfterRender callbacks on mesh or material are not supported in this slice',
  LAYER_MISMATCH: 'LAYER_MISMATCH: Camera layers do not intersect mesh layers (camera.layers.test(mesh.layers) === false)',
  INVALID_GEOMETRY: 'INVALID_GEOMETRY: Mesh geometry must be an instance of THREE.BufferGeometry',
  MISSING_POSITION: 'MISSING_POSITION: BufferGeometry must have a "position" attribute with itemSize === 3',
  UNSUPPORTED_ATTRIBUTE: 'UNSUPPORTED_ATTRIBUTE: Interleaved or normalized vertex attributes are not supported in this slice',
  UNSUPPORTED_GEOMETRY: 'UNSUPPORTED_GEOMETRY: Morph targets and multiple geometry groups are not supported in this slice',
  INVALID_MATERIAL: 'INVALID_MATERIAL: Material must be an instance of THREE.MeshBasicMaterial',
  UNSUPPORTED_MATERIAL: 'UNSUPPORTED_MATERIAL: Textured maps, transparency, wireframe, or custom blending are not supported in this slice',
  UNSUPPORTED_MATERIAL_FEATURE: 'UNSUPPORTED_MATERIAL_FEATURE: vertexColors, colorWrite=false, clippingPlanes, alphaTest/alphaHash, or custom shader hooks are not supported in this slice',
  UNSUPPORTED_DEPTH: 'UNSUPPORTED_DEPTH: Material must have depthTest === false and depthWrite === false in this slice (pipeline has no depth buffer)',
  UNSUPPORTED_STENCIL: 'UNSUPPORTED_STENCIL: Stencil operations are not supported in this slice (material.stencilWrite === true)',
  UNSUPPORTED_POLYGON_OFFSET: 'UNSUPPORTED_POLYGON_OFFSET: Polygon offset is not supported in this slice (material.polygonOffset === true)',
  UNSUPPORTED_REVERSED_DEPTH: 'UNSUPPORTED_REVERSED_DEPTH: Reversed depth buffer is not supported in this slice',
  INVALID_DEPTH_FUNC: 'INVALID_DEPTH_FUNC: Invalid or unsupported depthFunc',
  AMBIGUOUS_DEPTH_PAIR: 'AMBIGUOUS_DEPTH_PAIR: Material with depthTest=false and depthWrite=true is ambiguous across backends (WebGL suppresses writes, WebGPU permits writes); provide options.sourceBackend ("webgl" | "webgpu")',
  UNSUPPORTED_SIDE: 'UNSUPPORTED_SIDE: Material side must be FrontSide (0), BackSide (1), or DoubleSide (2)',
  MISSING_CULL_EXPORT: 'MISSING_CULL_EXPORT: FrontSide or BackSide mesh rendering requires Wasm export f3d_build_mesh_batch_cull_packet; silent DoubleSide fallback is strictly forbidden',
  INVALID_CULL_MODE: 'INVALID_CULL_MODE: Invalid or unknown cull mode wire value',
  INVALID_FRONT_FACE: 'INVALID_FRONT_FACE: Invalid or unknown front face wire value',
  INVALID_CAMERA: 'INVALID_CAMERA: Camera must be an instance of THREE.Camera with valid projectionMatrix and matrixWorldInverse',
  INVALID_DIMENSIONS: 'INVALID_DIMENSIONS: Viewport dimensions must be positive integers',
  INDEX_OUT_OF_BOUNDS: 'INDEX_OUT_OF_BOUNDS: Index references vertex out of bounds',
  INVALID_DRAWRANGE: 'INVALID_DRAWRANGE: Invalid drawRange: start and count must be non-negative integers',
  EMPTY_MESH_BATCH: 'EMPTY_MESH_BATCH: Mesh batch must be a non-empty array of meshes',
  INCOMPATIBLE_BATCH_DEPTH: 'INCOMPATIBLE_BATCH_DEPTH: Meshes in batch have incompatible depth settings; all meshes in batch must share depthTest, depthWrite, and depthCompare',
  UNSUPPORTED_RENDERABLE: 'UNSUPPORTED_RENDERABLE: Non-mesh renderable objects (Line, Points, Sprite, Light) are not supported in this slice',
  UNSUPPORTED_SCENE_FEATURE: 'UNSUPPORTED_SCENE_FEATURE: Scene-level features (background, fog, overrideMaterial, environment) are not supported in this slice',
});

// Supported upstream source backends for resolving backend-specific semantics
export const SOURCE_BACKEND = Object.freeze({
  WEBGL: 'webgl',
  WEBGPU: 'webgpu',
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
  NEVER: 1,         // "never"
  LESS: 2,          // "less"
  EQUAL: 3,         // "equal"
  LESS_EQUAL: 4,    // "less-equal"
  GREATER: 5,       // "greater"
  NOT_EQUAL: 6,     // "not-equal"
  GREATER_EQUAL: 7, // "greater-equal"
  ALWAYS: 8,        // "always"
});

// Pinned Three.js depth constants (0..7) to wire compare codes (1..8)
export const THREE_DEPTH_FUNC_TO_WIRE_COMPARE = Object.freeze({
  [NeverDepth]: DEPTH_WIRE_COMPARE.NEVER,                 // 0 -> 1
  [AlwaysDepth]: DEPTH_WIRE_COMPARE.ALWAYS,               // 1 -> 8
  [LessDepth]: DEPTH_WIRE_COMPARE.LESS,                   // 2 -> 2
  [LessEqualDepth]: DEPTH_WIRE_COMPARE.LESS_EQUAL,         // 3 -> 4
  [EqualDepth]: DEPTH_WIRE_COMPARE.EQUAL,                 // 4 -> 3
  [GreaterEqualDepth]: DEPTH_WIRE_COMPARE.GREATER_EQUAL,  // 5 -> 7
  [GreaterDepth]: DEPTH_WIRE_COMPARE.GREATER,             // 6 -> 5
  [NotEqualDepth]: DEPTH_WIRE_COMPARE.NOT_EQUAL,          // 7 -> 6
});

/**
 * Computes the 3x3 determinant of the upper-left affine transform block of a 4x4 matrix.
 * Matches Three.js Matrix4.determinantAffine().
 *
 * @param {any} matrixWorld - Three.js Matrix4 or object with elements array
 * @returns {number}
 */
export function computeAffineDeterminant(matrixWorld) {
  if (typeof matrixWorld?.determinantAffine === 'function') {
    return matrixWorld.determinantAffine();
  }
  const te = matrixWorld?.elements;
  if (!te || te.length < 16) return 1.0;
  const n11 = te[0], n12 = te[4], n13 = te[8];
  const n21 = te[1], n22 = te[5], n23 = te[9];
  const n31 = te[2], n32 = te[6], n33 = te[10];
  return (
    n11 * (n22 * n33 - n23 * n32) -
    n12 * (n21 * n33 - n23 * n31) +
    n13 * (n21 * n32 - n22 * n31)
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
export function createAdmissionError(code, detail = '') {
  const reasonText = ADMISSION_REJECTION[code] ?? 'Admission rejected';
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
  Object.defineProperty(item, 'code', {
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

/**
 * Checks if a Three.js Mesh and Camera can be admitted into the dynamic Wasm mesh rendering slice.
 * @param {any} mesh
 * @param {any} camera
 * @param {object} [options]
 * @returns {{ admitted: boolean, reason?: string, reasonCode?: string, code?: string }}
 */
export function canAdmitMesh(mesh, camera, options = {}) {
  if (!mesh || !mesh.isMesh) {
    return rejectMesh('NOT_A_MESH');
  }

  // Explicitly decline subclasses that cannot silently render scalar (root 19:47Z)
  if (mesh.isInstancedMesh || mesh.isSkinnedMesh || mesh.isBatchedMesh) {
    return rejectMesh('UNSUPPORTED_MESH_SUBCLASS');
  }

  // Check mesh and ancestor visibility (Three.js Renderer.js:3244 subtree culling)
  let cur = mesh;
  while (cur) {
    if (cur.visible === false) {
      return rejectMesh('NOT_VISIBLE');
    }
    cur = cur.parent;
  }

  // Reject unsupported object callbacks (root 19:47Z point 2)
  if (
    (mesh.onBeforeRender && mesh.onBeforeRender !== DEFAULT_OBJECT3D_ON_BEFORE_RENDER) ||
    Object.hasOwn(mesh, 'onBeforeRender') ||
    (mesh.onAfterRender && mesh.onAfterRender !== DEFAULT_OBJECT3D_ON_AFTER_RENDER) ||
    Object.hasOwn(mesh, 'onAfterRender')
  ) {
    return rejectMesh('UNSUPPORTED_CALLBACK');
  }

  // Camera check
  if (!camera || !camera.isCamera || !camera.projectionMatrix || !camera.matrixWorldInverse) {
    return rejectMesh('INVALID_CAMERA');
  }

  // Reject reversed depth buffer configurations (root review invariant)
  if (camera.reversedDepth === true || camera.reversedDepthBuffer === true || camera._reversedDepth === true) {
    return rejectMesh('UNSUPPORTED_REVERSED_DEPTH');
  }

  // Layer intersection check (root 19:47Z)
  if (camera.layers && mesh.layers && !camera.layers.test(mesh.layers)) {
    return rejectMesh('LAYER_MISMATCH');
  }

  const geometry = mesh.geometry;
  if (!geometry || !geometry.isBufferGeometry) {
    return rejectMesh('INVALID_GEOMETRY');
  }

  const posAttr = geometry.attributes?.position;
  if (!posAttr || posAttr.itemSize !== 3) {
    return rejectMesh('MISSING_POSITION');
  }

  // Reject interleaved or normalized attributes explicitly
  if (posAttr.isInterleavedBufferAttribute || posAttr.normalized) {
    return rejectMesh('UNSUPPORTED_ATTRIBUTE');
  }

  if (geometry.morphAttributes && Object.keys(geometry.morphAttributes).length > 0) {
    return rejectMesh('UNSUPPORTED_GEOMETRY');
  }
  if (geometry.groups && geometry.groups.length > 1) {
    return rejectMesh('UNSUPPORTED_GEOMETRY');
  }

  const material = mesh.material;
  if (!material || Array.isArray(material) || !material.isMeshBasicMaterial) {
    return rejectMesh('INVALID_MATERIAL');
  }

  // Material visibility check (root 19:47Z point 1)
  if (material.visible === false) {
    return rejectMesh('MATERIAL_NOT_VISIBLE');
  }

  // Reject unsupported material callbacks (root 19:47Z point 2)
  if (
    (material.onBeforeRender && material.onBeforeRender !== DEFAULT_MATERIAL_ON_BEFORE_RENDER) ||
    Object.hasOwn(material, 'onBeforeRender') ||
    material.onAfterRender ||
    Object.hasOwn(material, 'onAfterRender')
  ) {
    return rejectMesh('UNSUPPORTED_CALLBACK');
  }

  if (material.transparent === true || (material.opacity !== undefined && material.opacity < 1.0)) {
    return rejectMesh('UNSUPPORTED_MATERIAL');
  }
  if (material.map || material.envMap || material.alphaMap || material.lightMap || material.aoMap) {
    return rejectMesh('UNSUPPORTED_MATERIAL');
  }
  if (material.wireframe === true) {
    return rejectMesh('UNSUPPORTED_MATERIAL');
  }

  // Reject stencil operations
  if (material.stencilWrite === true) {
    return rejectMesh('UNSUPPORTED_STENCIL');
  }

  // Reject polygon offset
  if (material.polygonOffset === true) {
    return rejectMesh('UNSUPPORTED_POLYGON_OFFSET');
  }

  // Validate depth function if specified
  const rawDepthFunc = material.depthFunc;
  if (rawDepthFunc !== undefined && THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc] === undefined) {
    return rejectMesh('INVALID_DEPTH_FUNC');
  }

  // Handle depthTest=false and depthWrite=true ambiguity across backends (root review invariant)
  const depthTest = material.depthTest !== false;
  const depthWrite = material.depthWrite !== false;
  if (!depthTest && depthWrite) {
    const backend = options?.sourceBackend?.toLowerCase();
    if (backend !== SOURCE_BACKEND.WEBGL && backend !== SOURCE_BACKEND.WEBGPU) {
      return rejectMesh('AMBIGUOUS_DEPTH_PAIR');
    }
  }

  // Side admission: FrontSide (0), BackSide (1), and DoubleSide (2) are supported
  const side = material.side ?? THREE_SIDE.FRONT_SIDE;
  if (side !== THREE_SIDE.FRONT_SIDE && side !== THREE_SIDE.BACK_SIDE && side !== THREE_SIDE.DOUBLE_SIDE) {
    return rejectMesh('UNSUPPORTED_SIDE');
  }

  // Strict check on advanced material features
  if (material.vertexColors === true) {
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
  }
  if (material.colorWrite === false) {
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
  }
  if (material.clippingPlanes && material.clippingPlanes.length > 0) {
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
  }
  if ((material.alphaTest && material.alphaTest > 0) || material.alphaHash === true) {
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
  }
  if (material.blending !== undefined && material.blending !== 1) { // 1 = NormalBlending
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
  }

  // Check for custom shader hooks by comparing against base Material prototype methods
  if (
    (material.onBeforeCompile && material.onBeforeCompile !== DEFAULT_MATERIAL_ON_BEFORE_COMPILE) ||
    Object.hasOwn(material, 'onBeforeCompile')
  ) {
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
  }
  if (
    (material.customProgramCacheKey && material.customProgramCacheKey !== DEFAULT_MATERIAL_CUSTOM_PROGRAM_CACHE_KEY) ||
    Object.hasOwn(material, 'customProgramCacheKey')
  ) {
    return rejectMesh('UNSUPPORTED_MATERIAL_FEATURE');
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
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw createAdmissionError('INVALID_DIMENSIONS');
  }

  const geometry = mesh.geometry;
  const posAttr = geometry.attributes.position;
  const totalVertexCount = posAttr.count;
  const indexAttr = geometry.index;

  // Validate drawRange parameters
  const drawStart = geometry.drawRange?.start ?? 0;
  const rawDrawCount = geometry.drawRange?.count;
  if (!Number.isInteger(drawStart) || drawStart < 0) {
    throw createAdmissionError('INVALID_DRAWRANGE', 'start must be non-negative integer');
  }
  if (rawDrawCount !== undefined && rawDrawCount !== Infinity && (!Number.isInteger(rawDrawCount) || rawDrawCount < 0)) {
    throw createAdmissionError('INVALID_DRAWRANGE', 'count must be non-negative integer');
  }

  let positions;
  let indices;
  let triangleCount = 0;

  if (indexAttr) {
    const totalIndexCount = indexAttr.count;
    // Clamp count to remaining indices past drawStart
    let effectiveCount = 0;
    if (drawStart < totalIndexCount) {
      const maxAvailable = totalIndexCount - drawStart;
      effectiveCount = (rawDrawCount !== undefined && rawDrawCount !== Infinity)
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
        const idx = indexAttr.array[drawStart + i];
        // Explicit bounds check to prevent silent conversion of undefined to 0
        if (idx === undefined || idx < 0 || idx >= totalVertexCount) {
          throw createAdmissionError(
            'INDEX_OUT_OF_BOUNDS',
            `index ${idx} at position ${drawStart + i} exceeds vertex count ${totalVertexCount}`
          );
        }
        indices[i] = idx;
      }

      // Read vertex positions safely
      positions = new Float32Array(totalVertexCount * 3);
      for (let i = 0; i < totalVertexCount; i++) {
        positions[i * 3] = posAttr.getX(i);
        positions[i * 3 + 1] = posAttr.getY(i);
        positions[i * 3 + 2] = posAttr.getZ(i);
      }
      triangleCount = Math.floor(effectiveCount / 3);
    }
  } else {
    // Non-indexed
    let effectiveCount = 0;
    if (drawStart < totalVertexCount) {
      const maxAvailable = totalVertexCount - drawStart;
      effectiveCount = (rawDrawCount !== undefined && rawDrawCount !== Infinity)
        ? Math.min(rawDrawCount, maxAvailable)
        : maxAvailable;
    }

    positions = new Float32Array(effectiveCount * 3);
    for (let i = 0; i < effectiveCount; i++) {
      const vertIdx = drawStart + i;
      positions[i * 3] = posAttr.getX(vertIdx);
      positions[i * 3 + 1] = posAttr.getY(vertIdx);
      positions[i * 3 + 2] = posAttr.getZ(vertIdx);
    }
    indices = new Uint32Array(0);
    triangleCount = Math.floor(effectiveCount / 3);
  }

  // Expanded unindexed positions for backends without index buffer support
  const expandedPositions = indices.length > 0
    ? expandIndexedPositions(positions, indices)
    : new Float32Array(positions);

  // Model-View Matrix: MV = camera.matrixWorldInverse * mesh.matrixWorld
  const modelView = multiplyMatrices4x4(camera.matrixWorldInverse, mesh.matrixWorld);

  // Source coordinate system check: default in Three.js is WebGL (2000)
  const isWebGPUCoord = camera.coordinateSystem === COORDINATE_SYSTEM.WEBGPU;
  const webglDepth = !isWebGPUCoord;

  // Raw projection matrix (16 f64 elements)
  const projection = new Float64Array(camera.projectionMatrix.elements);

  // Material Color: [r, g, b, opacity] in linear sRGB
  const mat = mesh.material;
  const color = new Float32Array([
    mat.color?.r ?? 1.0,
    mat.color?.g ?? 1.0,
    mat.color?.b ?? 1.0,
    mat.opacity ?? 1.0,
  ]);

  // Material Depth Settings
  const depthTest = mat.depthTest !== false;
  const rawDepthWrite = mat.depthWrite !== false;
  const rawDepthFunc = mat.depthFunc ?? LessEqualDepth;
  if (THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc] === undefined) {
    throw createAdmissionError('INVALID_DEPTH_FUNC', `${rawDepthFunc}`);
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
      throw createAdmissionError('AMBIGUOUS_DEPTH_PAIR');
    }
  }

  // When depthTest is false, WebGPU pipeline uses GPUCompareFunction.Always (wire 8)
  const depthCompare = depthTest
    ? THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc]
    : DEPTH_WIRE_COMPARE.ALWAYS;

  // Material Side & Culling Settings
  const side = mat.side ?? THREE_SIDE.FRONT_SIDE;
  if (side !== THREE_SIDE.FRONT_SIDE && side !== THREE_SIDE.BACK_SIDE && side !== THREE_SIDE.DOUBLE_SIDE) {
    throw createAdmissionError('UNSUPPORTED_SIDE', `material side ${side} is not supported`);
  }

  const det = computeAffineDeterminant(mesh.matrixWorld);
  const isReflected = det < 0;
  let flipSided = (side === THREE_SIDE.BACK_SIDE);
  if (isReflected) flipSided = !flipSided;

  const cullMode = (side === THREE_SIDE.DOUBLE_SIDE) ? CULL_MODE_WIRE.NONE : CULL_MODE_WIRE.BACK;
  const frontFace = flipSided ? FRONT_FACE_WIRE.CW : FRONT_FACE_WIRE.CCW;

  return Object.freeze({
    positions,
    indices,
    expandedPositions,
    modelView,
    projection,
    color,
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
  const cullBatchFn = wasmModule?.f3d_build_mesh_batch_cull_packet;
  if (typeof cullBatchFn !== 'function') {
    throw createAdmissionError(
      'MISSING_CULL_EXPORT',
      `wasmModule is missing f3d_build_mesh_batch_cull_packet export for side ${snapshot.side}. Silent DoubleSide fallback is strictly forbidden.`
    );
  }

  if (snapshot.cullMode !== CULL_MODE_WIRE.NONE && snapshot.cullMode !== CULL_MODE_WIRE.FRONT && snapshot.cullMode !== CULL_MODE_WIRE.BACK) {
    throw createAdmissionError('INVALID_CULL_MODE', `snapshot has invalid cullMode ${snapshot.cullMode}`);
  }
  if (snapshot.frontFace !== FRONT_FACE_WIRE.CCW && snapshot.frontFace !== FRONT_FACE_WIRE.CW) {
    throw createAdmissionError('INVALID_FRONT_FACE', `snapshot has invalid frontFace ${snapshot.frontFace}`);
  }

  const positionsToUse = snapshot.expandedPositions;
  const vertexCount = positionsToUse.length / 3;
  const vertexCounts = new Uint32Array([vertexCount]);
  const cullModes = new Uint8Array([snapshot.cullMode]);
  const frontFaces = new Uint8Array([snapshot.frontFace]);

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
    isCanvas
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

  if (snapshot.side !== THREE_SIDE.DOUBLE_SIDE) {
    const packetBytes = buildSingleMeshCullPacket(snapshot, width, height, wasmModule, true, options);
    return { packetBytes, snapshot, target: 'canvas' };
  }

  const canvasDepthFn =
    wasmModule?.f3d_build_canvas_mesh_depth_packet ||
    wasmModule?.gpu_bridge_build_canvas_mesh_depth_packet;
  const canvasLegacyFn =
    wasmModule?.f3d_build_canvas_mesh_packet ||
    wasmModule?.gpu_bridge_build_canvas_mesh_packet;

  if (!wasmModule || (typeof canvasDepthFn !== 'function' && typeof canvasLegacyFn !== 'function')) {
    throw new Error(
      'Visible canvas mesh packet preparation failed: wasmModule is missing f3d_build_canvas_mesh_packet export. ' +
      'Silent offscreen-as-visible fallback is strictly forbidden.'
    );
  }

  const requiresDepth = snapshot.depthTest === true || snapshot.depthWrite === true;
  if (requiresDepth && typeof canvasDepthFn !== 'function') {
    throw new Error(
      'Visible canvas mesh requires depth (depthTest or depthWrite enabled), but wasmModule is missing ' +
      'f3d_build_canvas_mesh_depth_packet export. Silent offscreen-as-visible fallback or retained renderer masquerading is strictly forbidden.'
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  let packetBytes;
  if (typeof canvasDepthFn === 'function') {
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

  return { packetBytes, snapshot, target: 'canvas' };
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
export function prepareCanvasMeshDepthPacket(mesh, camera, width, height, wasmModule, options = {}) {
  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (snapshot.side !== THREE_SIDE.DOUBLE_SIDE) {
    const packetBytes = buildSingleMeshCullPacket(snapshot, width, height, wasmModule, true, options);
    return { packetBytes, snapshot, target: 'canvas' };
  }

  const buildFn =
    wasmModule?.f3d_build_canvas_mesh_depth_packet ||
    wasmModule?.gpu_bridge_build_canvas_mesh_depth_packet;

  if (typeof buildFn !== 'function') {
    throw new Error(
      'Visible canvas mesh depth packet preparation failed: wasmModule is missing f3d_build_canvas_mesh_depth_packet / gpu_bridge_build_canvas_mesh_depth_packet export. ' +
      'Silent offscreen-as-visible fallback or retained renderer masquerading is strictly forbidden.'
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

  return { packetBytes, snapshot, target: 'canvas' };
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
  if (options.target === 'canvas') {
    return prepareCanvasMeshPacket(mesh, camera, width, height, wasmModule, options);
  }

  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (snapshot.side !== THREE_SIDE.DOUBLE_SIDE) {
    const packetBytes = buildSingleMeshCullPacket(snapshot, width, height, wasmModule, false, options);
    return { packetBytes, snapshot, target: 'offscreen' };
  }

  const depthFn =
    wasmModule?.f3d_build_mesh_depth_packet ||
    wasmModule?.gpu_bridge_build_mesh_depth_packet;
  const legacyFn =
    wasmModule?.f3d_build_mesh_packet ||
    wasmModule?.gpu_bridge_build_mesh_packet;

  if (!wasmModule || (typeof depthFn !== 'function' && typeof legacyFn !== 'function')) {
    throw new Error('Invalid wasmModule: must expose f3d_build_mesh_packet export');
  }

  const requiresDepth = snapshot.depthTest === true || snapshot.depthWrite === true;
  if (requiresDepth && typeof depthFn !== 'function') {
    throw new Error(
      'Mesh requires depth (depthTest or depthWrite enabled), but wasmModule is missing ' +
      'f3d_build_mesh_depth_packet export. Silent no-depth fallback or retained renderer masquerading is strictly forbidden.'
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  let packetBytes;
  if (typeof depthFn === 'function') {
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

  return { packetBytes, snapshot, target: 'offscreen' };
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

  if (snapshot.side !== THREE_SIDE.DOUBLE_SIDE) {
    const packetBytes = buildSingleMeshCullPacket(snapshot, width, height, wasmModule, false, options);
    return { packetBytes, snapshot, target: 'offscreen' };
  }

  const buildFn =
    wasmModule?.f3d_build_mesh_depth_packet ||
    wasmModule?.gpu_bridge_build_mesh_depth_packet;

  if (typeof buildFn !== 'function') {
    throw new Error(
      'Mesh depth packet preparation failed: wasmModule is missing f3d_build_mesh_depth_packet / gpu_bridge_build_mesh_depth_packet export. ' +
      'Silent no-depth fallback or retained renderer masquerading is strictly forbidden.'
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

  return { packetBytes, snapshot, target: 'offscreen' };
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
export async function renderMesh(bridgeHost, mesh, camera, canvasContext, wasmModule, options = {}) {
  if (!bridgeHost || typeof bridgeHost.executePacket !== 'function') {
    throw new Error('Invalid bridgeHost: must expose executePacket method');
  }

  const isCanvasTarget = canvasContext !== null && canvasContext !== undefined;

  if (isCanvasTarget) {
    const hasCanvasExport = wasmModule && (
      typeof wasmModule.f3d_build_canvas_mesh_packet === 'function' ||
      typeof wasmModule.f3d_build_canvas_mesh_depth_packet === 'function' ||
      typeof wasmModule.gpu_bridge_build_canvas_mesh_packet === 'function' ||
      typeof wasmModule.gpu_bridge_build_canvas_mesh_depth_packet === 'function'
    );
    if (!hasCanvasExport) {
      throw new Error(
        'renderMesh refused: canvasContext provided for visible canvas rendering, but wasmModule ' +
        'does not export f3d_build_canvas_mesh_packet. Silent offscreen-as-visible rendering is strictly forbidden.'
      );
    }
    const width = canvasContext?.canvas?.width ?? options.width ?? 64;
    const height = canvasContext?.canvas?.height ?? options.height ?? 64;
    const { packetBytes, snapshot } = prepareCanvasMeshPacket(
      mesh,
      camera,
      width,
      height,
      wasmModule,
      options,
    );
    const result = await bridgeHost.executePacket(packetBytes, canvasContext);
    return { result, snapshot, target: 'canvas' };
  } else {
    // Honest offscreen execution
    const width = options.width ?? 64;
    const height = options.height ?? 64;
    const { packetBytes, snapshot } = prepareMeshPacket(
      mesh,
      camera,
      width,
      height,
      wasmModule,
      { ...options, target: 'offscreen' },
    );
    const result = await bridgeHost.executePacket(packetBytes, null);
    return { result, snapshot, target: 'offscreen' };
  }
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
    throw createAdmissionError('EMPTY_MESH_BATCH');
  }

  // If options.autoUpdate is explicitly true (standalone batch preparation requesting auto-update),
  // update world matrices according to Three.js boundaries without repeating if autoUpdate is false.
  if (options.autoUpdate === true) {
    if (camera.parent === null && camera.matrixWorldAutoUpdate === true && typeof camera.updateMatrixWorld === 'function') {
      camera.updateMatrixWorld();
    }
    for (const mesh of meshes) {
      if (mesh.matrixWorldAutoUpdate === true && typeof mesh.updateMatrixWorld === 'function') {
        mesh.updateMatrixWorld();
      }
    }
  }

  const cullBatchFn = wasmModule?.f3d_build_mesh_batch_cull_packet;
  const legacyBatchFn = wasmModule?.f3d_build_mesh_batch_packet;

  if (typeof cullBatchFn !== 'function' && typeof legacyBatchFn !== 'function') {
    throw new Error(
      'Mesh batch packet preparation failed: wasmModule is missing f3d_build_mesh_batch_packet or f3d_build_mesh_batch_cull_packet export.'
    );
  }

  // Extract snapshot for each mesh in the explicit order
  const snapshots = meshes.map((mesh, index) => {
    try {
      return extractMeshRenderData(mesh, camera, width, height, options);
    } catch (err) {
      const code = err.reason ?? 'ADMISSION_REJECTED';
      const batchErr = new Error(`${code}: Mesh batch admission rejected at index ${index}: ${err.message}`);
      batchErr.reason = code;
      throw batchErr;
    }
  });

  // Verify shared batch pipeline configuration: depthTest, depthWrite, depthCompare, webglDepth
  const first = snapshots[0];
  const sharedDepthTest = first.depthTest;
  const sharedDepthWrite = first.depthWrite;
  const sharedDepthCompare = first.depthCompare;
  const sharedWebglDepth = first.webglDepth;

  for (let i = 1; i < snapshots.length; i++) {
    const s = snapshots[i];
    if (s.depthTest !== sharedDepthTest) {
      throw createAdmissionError(
        'INCOMPATIBLE_BATCH_DEPTH',
        `mesh 0 has depthTest=${sharedDepthTest}, mesh ${i} has depthTest=${s.depthTest}`
      );
    }
    if (s.depthWrite !== sharedDepthWrite) {
      throw createAdmissionError(
        'INCOMPATIBLE_BATCH_DEPTH',
        `mesh 0 has depthWrite=${sharedDepthWrite}, mesh ${i} has depthWrite=${s.depthWrite}`
      );
    }
    if (s.depthCompare !== sharedDepthCompare) {
      throw createAdmissionError(
        'INCOMPATIBLE_BATCH_DEPTH',
        `mesh 0 has depthCompare=${sharedDepthCompare}, mesh ${i} has depthCompare=${s.depthCompare}`
      );
    }
    if (s.webglDepth !== sharedWebglDepth) {
      throw createAdmissionError(
        'INCOMPATIBLE_BATCH_DEPTH',
        `mesh 0 has webglDepth=${sharedWebglDepth}, mesh ${i} has webglDepth=${s.webglDepth}`
      );
    }
  }

  // Material side / cull checks and typed array creation
  const n = snapshots.length;
  const cullModes = new Uint8Array(n);
  const frontFaces = new Uint8Array(n);
  let hasNonDoubleSide = false;

  for (let i = 0; i < n; i++) {
    const s = snapshots[i];
    if (s.side !== THREE_SIDE.DOUBLE_SIDE) {
      hasNonDoubleSide = true;
    }
    if (s.cullMode !== CULL_MODE_WIRE.NONE && s.cullMode !== CULL_MODE_WIRE.FRONT && s.cullMode !== CULL_MODE_WIRE.BACK) {
      throw createAdmissionError('INVALID_CULL_MODE', `mesh ${i} has invalid cullMode ${s.cullMode}`);
    }
    if (s.frontFace !== FRONT_FACE_WIRE.CCW && s.frontFace !== FRONT_FACE_WIRE.CW) {
      throw createAdmissionError('INVALID_FRONT_FACE', `mesh ${i} has invalid frontFace ${s.frontFace}`);
    }
    cullModes[i] = s.cullMode;
    frontFaces[i] = s.frontFace;
  }

  if (hasNonDoubleSide && typeof cullBatchFn !== 'function') {
    throw createAdmissionError(
      'MISSING_CULL_EXPORT',
      'Mesh batch contains FrontSide or BackSide meshes, but wasmModule is missing f3d_build_mesh_batch_cull_packet export. Silent DoubleSide fallback is strictly forbidden.'
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

  const isCanvas = options.target === 'canvas';

  let packetBytes;
  try {
    if (typeof cullBatchFn === 'function') {
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
      msg.includes('EmptyMeshList') ||
      msg.includes('must contain at least one mesh') ||
      msg.includes('EMPTY_MESH_BATCH')
    ) {
      throw createAdmissionError('EMPTY_MESH_BATCH');
    }
    if (msg.includes('INVALID_CULL_MODE')) {
      throw createAdmissionError('INVALID_CULL_MODE', msg);
    }
    if (msg.includes('INVALID_FRONT_FACE')) {
      throw createAdmissionError('INVALID_FRONT_FACE', msg);
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
    target: isCanvas ? 'canvas' : 'offscreen',
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
export async function renderMeshBatch(bridgeHost, meshes, camera, canvasContext, wasmModule, options = {}) {
  if (!bridgeHost || typeof bridgeHost.executePacket !== 'function') {
    throw new Error('Invalid bridgeHost: must expose executePacket method');
  }

  if (!Array.isArray(meshes) || meshes.length === 0) {
    throw createAdmissionError('EMPTY_MESH_BATCH');
  }

  const isCanvasTarget = canvasContext !== null && canvasContext !== undefined;
  const width = canvasContext?.canvas?.width ?? options.width ?? 64;
  const height = canvasContext?.canvas?.height ?? options.height ?? 64;

  const batchResult = prepareMeshBatchPacket(
    meshes,
    camera,
    width,
    height,
    wasmModule,
    { ...options, target: isCanvasTarget ? 'canvas' : 'offscreen' }
  );

  const result = await bridgeHost.executePacket(batchResult.packetBytes, isCanvasTarget ? canvasContext : null);
  return {
    result,
    snapshots: batchResult.snapshots,
    meshCount: batchResult.meshCount,
    totalVertices: batchResult.totalVertices,
    cullModes: batchResult.cullModes,
    frontFaces: batchResult.frontFaces,
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
export async function renderScene(bridgeHost, scene, camera, canvasContext, wasmModule, options = {}) {
  if (!bridgeHost || typeof bridgeHost.executePacket !== 'function') {
    throw new Error('Invalid bridgeHost: must expose executePacket method');
  }
  if (!scene || typeof scene.traverse !== 'function') {
    throw new Error('Invalid scene: must expose traverse method');
  }
  if (!camera || !camera.isCamera) {
    throw createAdmissionError('INVALID_CAMERA');
  }

  // Update world matrices according to Three.js renderer source boundaries
  // (Renderer.js:1755, 3677 / WebGLRenderer.js:1663, 1667)
  if (scene.matrixWorldAutoUpdate === true && typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld();
  }
  if (camera.parent === null && camera.matrixWorldAutoUpdate === true && typeof camera.updateMatrixWorld === 'function') {
    camera.updateMatrixWorld();
  }

  const projScreenMatrix = new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const vector4 = new Vector4();

  const admittedItems = [];
  const refused = [];

  // Scene-level unsupported properties check (Root Mail 14355)
  if (scene.onBeforeRender !== DEFAULT_OBJECT3D_ON_BEFORE_RENDER ||
      scene.onAfterRender !== DEFAULT_OBJECT3D_ON_AFTER_RENDER) {
    refused.push(createRefusalItem(scene.uuid, 'UNSUPPORTED_CALLBACK'));
  }
  if (scene.background !== null && scene.background !== undefined) {
    refused.push(createRefusalItem(
      scene.uuid,
      'UNSUPPORTED_SCENE_FEATURE',
      `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.background is not supported`
    ));
  }
  if (scene.fog !== null && scene.fog !== undefined) {
    refused.push(createRefusalItem(
      scene.uuid,
      'UNSUPPORTED_SCENE_FEATURE',
      `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.fog is not supported`
    ));
  }
  if (scene.overrideMaterial !== null && scene.overrideMaterial !== undefined) {
    refused.push(createRefusalItem(
      scene.uuid,
      'UNSUPPORTED_SCENE_FEATURE',
      `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.overrideMaterial is not supported`
    ));
  }
  if (scene.environment !== null && scene.environment !== undefined) {
    refused.push(createRefusalItem(
      scene.uuid,
      'UNSUPPORTED_SCENE_FEATURE',
      `${ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE}: scene.environment is not supported`
    ));
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
    if (obj.isLine || obj.isLineSegments || obj.isLineLoop || obj.isPoints || obj.isSprite || obj.isLight || (obj.geometry && !obj.isMesh)) {
      refused.push(createRefusalItem(
        obj.uuid,
        'UNSUPPORTED_RENDERABLE',
        `${ADMISSION_REJECTION.UNSUPPORTED_RENDERABLE}: ${obj.type || 'Non-mesh renderable'} is not supported`
      ));
      return;
    }

    // Visible, in-layer Mesh
    if (obj.isMesh) {
      const admission = canAdmitMesh(obj, camera, options);
      if (!admission.admitted) {
        refused.push(createRefusalItem(
          obj.uuid,
          admission.code ?? 'ADMISSION_REJECTED',
          admission.reason
        ));
        return;
      }

      // Calculate projected z (Three.js WebGLRenderer.js:1924-1936 / Renderer.js:3300-3306)
      let z = 0;
      const geom = obj.geometry;
      if (options.sortObjects !== false && geom) {
        if (geom.boundingSphere === null && typeof geom.computeBoundingSphere === 'function') {
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

  // Verify shared batch pipeline configuration across admitted meshes in scene
  if (admittedItems.length > 1) {
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
        refused.push(createRefusalItem(
          admittedItems[i].mesh.uuid,
          'INCOMPATIBLE_BATCH_DEPTH',
          `${ADMISSION_REJECTION.INCOMPATIBLE_BATCH_DEPTH}: scene meshes have conflicting depth settings`
        ));
      }
    }
  }

  // Refuse WHOLE submission on any visible unsupported renderable or scene effect (Root Mail 14355)
  if (refused.length > 0) {
    const response = {
      admitted: [],
      refused,
    };
    const primaryCode = refused[0].code ?? 'UNSUPPORTED_SCENE_CONTENT';
    const primaryReason = refused[0].reason ?? ADMISSION_REJECTION.UNSUPPORTED_SCENE_FEATURE;
    Object.defineProperty(response, 'reason', {
      value: primaryCode,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(response, 'reasonMessage', {
      value: primaryReason,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(response, 'refusalReason', {
      value: primaryCode,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    return response;
  }

  // Empty admitted set -> explicit refusal, no submit
  if (admittedItems.length === 0) {
    const response = {
      admitted: [],
      refused: [],
    };
    Object.defineProperty(response, 'reason', {
      value: 'EMPTY_MESH_BATCH',
      enumerable: false,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(response, 'reasonMessage', {
      value: ADMISSION_REJECTION.EMPTY_MESH_BATCH,
      enumerable: false,
      writable: true,
      configurable: true,
    });
    Object.defineProperty(response, 'refusalReason', {
      value: 'EMPTY_MESH_BATCH',
      enumerable: false,
      writable: true,
      configurable: true,
    });
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

  const batchResult = prepareMeshBatchPacket(
    admittedMeshes,
    camera,
    width,
    height,
    wasmModule,
    { ...options, autoUpdate: false, target: isCanvasTarget ? 'canvas' : 'offscreen' }
  );

  const result = await bridgeHost.executePacket(
    batchResult.packetBytes,
    isCanvasTarget ? canvasContext : null
  );

  const response = {
    admitted,
    refused,
  };

  if (result !== undefined) {
    Object.defineProperty(response, 'result', {
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
  if (!renderer || typeof renderer.render !== 'function') {
    throw new Error('Cannot execute retained fallback: renderer.render is not a function');
  }
  renderer.render(scene, camera);
  return {
    implementationOwner: 'retained-js',
    rendered: true,
  };
}
