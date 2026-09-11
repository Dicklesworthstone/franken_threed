/**
 * @file tools/compat/mesh_adapter.mjs
 * Explicit experimental dynamic Mesh adapter for FrankenThreeD (f3d-05.6 / 19:26Z product wave).
 *
 * Extracts dynamic input from real pinned Three.js Mesh + Camera instances into
 * isolated typed arrays for the Wasm/WebGPU execution core:
 * - BufferGeometry position (itemSize 3, non-interleaved, unnormalized) + optional index + drawRange
 * - Opaque untextured MeshBasicMaterial color (Float32Array[4]), DoubleSide only, depthTest=false, depthWrite=false
 * - f64 model-view and projection matrices with WebGL-to-WebGPU depth coordinate tracking
 *
 * Invariants:
 * - Keeps JavaScript ownership of Three.js scene objects.
 * - Does not silently ignore unsupported features; explicitly refuses them.
 * - Takes a fresh typed-array snapshot per call; zero per-frame JSON.
 * - Preserves source matrix-update boundaries.
 * - Retains unsupported full application route without automatic rerouting.
 */

import { Material, Object3D } from '../../upstream/three.js/build/three.module.js';

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
  UNSUPPORTED_SIDE: 'Only DoubleSide (2) is supported in this slice (pipeline has no culling state)',
  INVALID_CAMERA: 'Camera must be an instance of THREE.Camera with valid projectionMatrix and matrixWorldInverse',
  INVALID_DIMENSIONS: 'Viewport dimensions must be positive integers',
  INDEX_OUT_OF_BOUNDS: 'Index references vertex out of bounds',
  INVALID_DRAWRANGE: 'Invalid drawRange: start and count must be non-negative integers',
});

// Pinned Three.js coordinate system constants
export const COORDINATE_SYSTEM = Object.freeze({
  WEBGL: 2000,
  WEBGPU: 2001,
});

/**
 * Checks if a Three.js Mesh and Camera can be admitted into the dynamic Wasm mesh rendering slice.
 * @param {any} mesh
 * @param {any} camera
 * @returns {{ admitted: boolean, reason?: string }}
 */
export function canAdmitMesh(mesh, camera) {
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

  // Pipeline has no depth buffer; require explicit depthTest=false and depthWrite=false
  if (material.depthTest === true || material.depthWrite === true) {
    return { admitted: false, reason: ADMISSION_REJECTION.UNSUPPORTED_DEPTH };
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
  const admission = canAdmitMesh(mesh, camera);
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
    vertexCount: positions.length / 3,
    triangleCount,
    isIndexed: indices.length > 0,
    side: mat.side,
  });
}

/**
 * Prepares a binary submission packet targeting a visible canvas presentation.
 *
 * Invariants:
 * - Requires wasmModule to expose f3d_build_canvas_mesh_packet.
 * - Explicitly refuses execution if the visible canvas export is missing;
 *   silent offscreen fallback is strictly forbidden.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} wasmModule - Loaded Wasm module exposing f3d_build_canvas_mesh_packet
 * @param {object} [options]
 * @returns {{ packetBytes: Uint8Array, snapshot: object, target: 'canvas' }}
 */
export function prepareCanvasMeshPacket(mesh, camera, width, height, wasmModule, options = {}) {
  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (!wasmModule || typeof wasmModule.f3d_build_canvas_mesh_packet !== 'function') {
    throw new Error(
      'Visible canvas mesh packet preparation failed: wasmModule is missing f3d_build_canvas_mesh_packet export. ' +
      'Silent offscreen-as-visible fallback is strictly forbidden.'
    );
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  const packetBytes = wasmModule.f3d_build_canvas_mesh_packet(
    positionsToUse,
    indicesToUse,
    snapshot.modelView,
    snapshot.projection,
    snapshot.color,
    width,
    height,
    snapshot.webglDepth,
  );

  return { packetBytes, snapshot, target: 'canvas' };
}

/**
 * Prepares a binary GpuSubmissionPacket from real Three.js Mesh inputs via dynamic Wasm export.
 * Defaults to offscreen rendering unless options.target === 'canvas'.
 *
 * @param {any} mesh
 * @param {any} camera
 * @param {number} width
 * @param {number} height
 * @param {object} wasmModule - Loaded Wasm module exposing f3d_build_mesh_packet
 * @param {object} [options]
 * @returns {{ packetBytes: Uint8Array, snapshot: object, target: 'offscreen' | 'canvas' }}
 */
export function prepareMeshPacket(mesh, camera, width, height, wasmModule, options = {}) {
  if (options.target === 'canvas') {
    return prepareCanvasMeshPacket(mesh, camera, width, height, wasmModule, options);
  }

  const snapshot = extractMeshRenderData(mesh, camera, width, height, options);

  if (!wasmModule || typeof wasmModule.f3d_build_mesh_packet !== 'function') {
    throw new Error('Invalid wasmModule: must expose f3d_build_mesh_packet export');
  }

  const positionsToUse = options.expandIndices ? snapshot.expandedPositions : snapshot.positions;
  const indicesToUse = options.expandIndices ? new Uint32Array(0) : snapshot.indices;

  // CobaltOrchid signature (8 arguments, confirmed by NavyAspen 13068):
  // f3d_build_mesh_packet(positions, indices, model_view, projection, color, width, height, webgl_depth)
  const packetBytes = wasmModule.f3d_build_mesh_packet(
    positionsToUse,
    indicesToUse,
    snapshot.modelView,
    snapshot.projection,
    snapshot.color,
    width,
    height,
    snapshot.webglDepth,
  );

  return { packetBytes, snapshot, target: 'offscreen' };
}

/**
 * Explicit bridge execution entry: prepares and executes a Mesh render through WebGpuBridgeHost.
 *
 * Invariants:
 * - If canvasContext is provided, routes through wasmModule.f3d_build_canvas_mesh_packet.
 *   If wasmModule lacks f3d_build_canvas_mesh_packet, EXPLICITLY REFUSES canvasContext;
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
    if (!wasmModule || typeof wasmModule.f3d_build_canvas_mesh_packet !== 'function') {
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
