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
  NOT_A_MESH: 'Object is not an instance of THREE.Mesh',
  UNSUPPORTED_MESH_SUBCLASS: 'InstancedMesh, SkinnedMesh, and BatchedMesh are not supported in this scalar slice',
  NOT_VISIBLE: 'Mesh is not visible (mesh.visible === false)',
  MATERIAL_NOT_VISIBLE: 'Material is not visible (material.visible === false)',
  UNSUPPORTED_CALLBACK: 'onBeforeRender and onAfterRender callbacks on mesh or material are not supported in this slice',
  LAYER_MISMATCH: 'Camera layers do not intersect mesh layers (camera.layers.test(mesh.layers) === false)',
  INVALID_GEOMETRY: 'Mesh geometry must be an instance of THREE.BufferGeometry',
  MISSING_POSITION: 'BufferGeometry must have a "position" attribute with itemSize === 3',
  UNSUPPORTED_ATTRIBUTE: 'Interleaved or normalized vertex attributes are not supported in this slice',
  UNSUPPORTED_GEOMETRY: 'Morph targets and multiple geometry groups are not supported in this slice',
  INVALID_MATERIAL: 'Material must be an instance of THREE.MeshBasicMaterial',
  UNSUPPORTED_MATERIAL: 'Textured maps, transparency, wireframe, or custom blending are not supported in this slice',
  UNSUPPORTED_MATERIAL_FEATURE: 'vertexColors, colorWrite=false, clippingPlanes, alphaTest/alphaHash, or custom shader hooks are not supported in this slice',
  UNSUPPORTED_DEPTH: 'Material must have depthTest === false and depthWrite === false in this slice (pipeline has no depth buffer)',
  UNSUPPORTED_STENCIL: 'Stencil operations are not supported in this slice (material.stencilWrite === true)',
  UNSUPPORTED_POLYGON_OFFSET: 'Polygon offset is not supported in this slice (material.polygonOffset === true)',
  UNSUPPORTED_REVERSED_DEPTH: 'Reversed depth buffer is not supported in this slice',
  INVALID_DEPTH_FUNC: 'Invalid or unsupported depthFunc',
  AMBIGUOUS_DEPTH_PAIR: 'Material with depthTest=false and depthWrite=true is ambiguous across backends (WebGL suppresses writes, WebGPU permits writes); provide options.sourceBackend ("webgl" | "webgpu")',
  UNSUPPORTED_SIDE: 'Only DoubleSide (2) is supported in this slice (pipeline has no culling state)',
  INVALID_CAMERA: 'Camera must be an instance of THREE.Camera with valid projectionMatrix and matrixWorldInverse',
  INVALID_DIMENSIONS: 'Viewport dimensions must be positive integers',
  INDEX_OUT_OF_BOUNDS: 'Index references vertex out of bounds',
  INVALID_DRAWRANGE: 'Invalid drawRange: start and count must be non-negative integers',
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
 * Checks if a Three.js Mesh and Camera can be admitted into the dynamic Wasm mesh rendering slice.
 * @param {any} mesh
 * @param {any} camera
 * @param {object} [options]
 * @returns {{ admitted: boolean, reason?: string }}
 */
export function canAdmitMesh(mesh, camera, options = {}) {
  if (!mesh || !mesh.isMesh) {
    return { admitted: false, reason: ADMISSION_REJECTION.NOT_A_MESH };
  }

  // Explicitly decline subclasses that cannot silently render scalar (root 19:47Z)
  if (mesh.isInstancedMesh || mesh.isSkinnedMesh || mesh.isBatchedMesh) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MESH_SUBCLASS };
  }

  // Check mesh visibility
  if (mesh.visible === false) {
    return { admitted: false, reason: ADMISSION_REJECTION.NOT_VISIBLE };
  }

  // Reject unsupported object callbacks (root 19:47Z point 2)
  if (
    (mesh.onBeforeRender && mesh.onBeforeRender !== DEFAULT_OBJECT3D_ON_BEFORE_RENDER) ||
    Object.hasOwn(mesh, 'onBeforeRender') ||
    (mesh.onAfterRender && mesh.onAfterRender !== DEFAULT_OBJECT3D_ON_AFTER_RENDER) ||
    Object.hasOwn(mesh, 'onAfterRender')
  ) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_CALLBACK };
  }

  // Camera check
  if (!camera || !camera.isCamera || !camera.projectionMatrix || !camera.matrixWorldInverse) {
    return { admitted: false, reason: ADMISSION_REJECTION.INVALID_CAMERA };
  }

  // Reject reversed depth buffer configurations (root review invariant)
  if (camera.reversedDepth === true || camera.reversedDepthBuffer === true || camera._reversedDepth === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_REVERSED_DEPTH };
  }

  // Layer intersection check (root 19:47Z)
  if (camera.layers && mesh.layers && !camera.layers.test(mesh.layers)) {
    return { admitted: false, reason: ADMISSION_REJECTION.LAYER_MISMATCH };
  }

  const geometry = mesh.geometry;
  if (!geometry || !geometry.isBufferGeometry) {
    return { admitted: false, reason: ADMISSION_REJECTION.INVALID_GEOMETRY };
  }

  const posAttr = geometry.attributes?.position;
  if (!posAttr || posAttr.itemSize !== 3) {
    return { admitted: false, reason: ADMISSION_REJECTION.MISSING_POSITION };
  }

  // Reject interleaved or normalized attributes explicitly
  if (posAttr.isInterleavedBufferAttribute || posAttr.normalized) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_ATTRIBUTE };
  }

  if (geometry.morphAttributes && Object.keys(geometry.morphAttributes).length > 0) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_GEOMETRY };
  }
  if (geometry.groups && geometry.groups.length > 1) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_GEOMETRY };
  }

  const material = mesh.material;
  if (!material || Array.isArray(material) || !material.isMeshBasicMaterial) {
    return { admitted: false, reason: ADMISSION_REJECTION.INVALID_MATERIAL };
  }

  // Material visibility check (root 19:47Z point 1)
  if (material.visible === false) {
    return { admitted: false, reason: ADMISSION_REJECTION.MATERIAL_NOT_VISIBLE };
  }

  // Reject unsupported material callbacks (root 19:47Z point 2)
  if (
    (material.onBeforeRender && material.onBeforeRender !== DEFAULT_MATERIAL_ON_BEFORE_RENDER) ||
    Object.hasOwn(material, 'onBeforeRender') ||
    material.onAfterRender ||
    Object.hasOwn(material, 'onAfterRender')
  ) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_CALLBACK };
  }

  if (material.transparent === true || (material.opacity !== undefined && material.opacity < 1.0)) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL };
  }
  if (material.map || material.envMap || material.alphaMap || material.lightMap || material.aoMap) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL };
  }
  if (material.wireframe === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL };
  }

  // Reject stencil operations
  if (material.stencilWrite === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_STENCIL };
  }

  // Reject polygon offset
  if (material.polygonOffset === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_POLYGON_OFFSET };
  }

  // Validate depth function if specified
  const rawDepthFunc = material.depthFunc;
  if (rawDepthFunc !== undefined && THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc] === undefined) {
    return { admitted: false, reason: ADMISSION_REJECTION.INVALID_DEPTH_FUNC };
  }

  // Handle depthTest=false and depthWrite=true ambiguity across backends (root review invariant)
  const depthTest = material.depthTest !== false;
  const depthWrite = material.depthWrite !== false;
  if (!depthTest && depthWrite) {
    const backend = options?.sourceBackend?.toLowerCase();
    if (backend !== SOURCE_BACKEND.WEBGL && backend !== SOURCE_BACKEND.WEBGPU) {
      return { admitted: false, reason: ADMISSION_REJECTION.AMBIGUOUS_DEPTH_PAIR };
    }
  }

  // Pipeline has no cull state; require DoubleSide (2) explicitly per 13043 / 13062 point 1
  if (material.side !== 2) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_SIDE };
  }

  // Strict check on advanced material features
  if (material.vertexColors === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
  }
  if (material.colorWrite === false) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
  }
  if (material.clippingPlanes && material.clippingPlanes.length > 0) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
  }
  if ((material.alphaTest && material.alphaTest > 0) || material.alphaHash === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
  }
  if (material.blending !== undefined && material.blending !== 1) { // 1 = NormalBlending
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
  }

  // Check for custom shader hooks by comparing against base Material prototype methods
  if (
    (material.onBeforeCompile && material.onBeforeCompile !== DEFAULT_MATERIAL_ON_BEFORE_COMPILE) ||
    Object.hasOwn(material, 'onBeforeCompile')
  ) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
  }
  if (
    (material.customProgramCacheKey && material.customProgramCacheKey !== DEFAULT_MATERIAL_CUSTOM_PROGRAM_CACHE_KEY) ||
    Object.hasOwn(material, 'customProgramCacheKey')
  ) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE };
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
    throw new Error(`Mesh admission rejected: ${admission.reason}`);
  }
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error(ADMISSION_REJECTION.INVALID_DIMENSIONS);
  }

  const geometry = mesh.geometry;
  const posAttr = geometry.attributes.position;
  const totalVertexCount = posAttr.count;
  const indexAttr = geometry.index;

  // Validate drawRange parameters
  const drawStart = geometry.drawRange?.start ?? 0;
  const rawDrawCount = geometry.drawRange?.count;
  if (!Number.isInteger(drawStart) || drawStart < 0) {
    throw new Error(ADMISSION_REJECTION.INVALID_DRAWRANGE);
  }
  if (rawDrawCount !== undefined && rawDrawCount !== Infinity && (!Number.isInteger(rawDrawCount) || rawDrawCount < 0)) {
    throw new Error(ADMISSION_REJECTION.INVALID_DRAWRANGE);
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
        const idx = indexAttr.getX(drawStart + i);
        // Explicit bounds check to prevent silent conversion of undefined to 0
        if (idx === undefined || idx < 0 || idx >= totalVertexCount) {
          throw new Error(`${ADMISSION_REJECTION.INDEX_OUT_OF_BOUNDS}: index ${idx} at position ${drawStart + i} exceeds vertex count ${totalVertexCount}`);
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
    throw new Error(`${ADMISSION_REJECTION.INVALID_DEPTH_FUNC}: ${rawDepthFunc}`);
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
      throw new Error(`Mesh admission rejected: ${ADMISSION_REJECTION.AMBIGUOUS_DEPTH_PAIR}`);
    }
  }

  // When depthTest is false, WebGPU pipeline uses GPUCompareFunction.Always (wire 8)
  const depthCompare = depthTest
    ? THREE_DEPTH_FUNC_TO_WIRE_COMPARE[rawDepthFunc]
    : DEPTH_WIRE_COMPARE.ALWAYS;

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
    side: mat.side,
  });
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
    const width = canvasContext?.canvas?.width || options.width || 64;
    const height = canvasContext?.canvas?.height || options.height || 64;
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
    const width = options.width || 64;
    const height = options.height || 64;
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
