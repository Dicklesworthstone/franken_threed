/**
 * Explicit CPU-deformed MeshBasicMaterial rendering through Rust/Wasm/WebGPU.
 * This extends the ordered mesh-packet API, not the exact retained renderer or
 * its scene traversal. The caller owns sorting, visibility, world/bind updates,
 * skeleton palette updates, scheduling, and disposal of source Three.js objects.
 *
 * Every call captures current attribute data (including normalized/interleaved
 * morph and skin attributes), evaluates the whole batch in one Rust call, then
 * uses the existing mesh packet builder for indices/drawRange, colors, reflected
 * winding, depth, colorWrite, and canvas/offscreen target selection. It does not
 * emulate persistent Three.js GPU attribute residency/version semantics: callers
 * needing that contract must stay on the exact retained route. CPU f64 skinning
 * is not a claim of shader-bitwise equivalence, GPU skinning, or measured speedup.
 */
import { BufferAttribute } from '../../upstream/three.js/build/three.module.js';
import { canAdmitMesh, createAdmissionError, prepareMeshBatchPacket } from './mesh_adapter.mjs';
import { captureDeformationBatch, evaluateDeformationBatch } from './deformation_inputs.mjs';

// Copy descriptors, not just values: own callback overrides must remain own
// properties so existing admission checks cannot be bypassed by the staging view.
function view(source, replacements) {
  const descriptors = Object.getOwnPropertyDescriptors(source);
  for (const [name, value] of Object.entries(replacements)) {
    descriptors[name] = { value, enumerable: true, writable: false, configurable: false };
  }
  return Object.create(Object.getPrototypeOf(source), descriptors);
}
function drawViews(input, positions) {
  return input.rows.map(row => {
    const position = new BufferAttribute(positions.subarray(row.positionOffset, row.positionOffset + row.count * 3), 3);
    const geometry = view(row.geometry, {
      attributes: { ...row.geometry.attributes, position }, morphAttributes: {},
      // This path does not run frustum traversal. Never expose the old bind-pose
      // bounds on a geometry with changed positions, including returned snapshots.
      boundingBox: null, boundingSphere: null,
    });
    return view(row.mesh, { geometry, isSkinnedMesh: false });
  });
}

/**
 * Build one existing F3DP packet for an explicit ordered mixed mesh batch.
 * Opaque, untextured MeshBasicMaterial restrictions remain unchanged. No source
 * attribute, hierarchy, palette, callback, or texture version is modified.
 * Pass the same target/sourceBackend/clearColor options as prepareMeshBatchPacket;
 * limits may be supplied as options.deformationLimits. autoUpdate=true rejects:
 * perform source updates explicitly BEFORE capturing palettes and bind matrices.
 *
 * Returns the existing packet/snapshots plus deformation counts. This is actual
 * packet production, not a substitute JavaScript skinning/rendering algorithm.
 */
export function prepareDeformedMeshBatchPacket(meshes, camera, width, height, wasmModule, options = {}) {
  if (options.autoUpdate === true) {
    throw createAdmissionError('DEFORMATION_UPDATE_BOUNDARY', 'update world matrices, bind state, and skeleton palettes before packet preparation');
  }
  for (const dimension of [width, height]) {
    if (!Number.isInteger(dimension) || dimension < 1 || dimension > 0xffffffff) throw createAdmissionError('INVALID_DIMENSIONS');
  }
  const input = captureDeformationBatch(meshes, options.deformationLimits);
  // Preflight the actual legacy material/camera/callback contract before entering
  // native deformation. The temporary views retain ALL non-deformation fields.
  const staged = drawViews(input, new Float32Array(input.positions));
  for (let index = 0; index < staged.length; index++) {
    const admission = canAdmitMesh(staged[index], camera, options);
    if (!admission.admitted) throw createAdmissionError(admission.code, `mesh ${index}`);
  }
  const positions = evaluateDeformationBatch(input, wasmModule);
  const prepared = prepareMeshBatchPacket(drawViews(input, positions), camera, width, height,
    wasmModule, { ...options, autoUpdate: false });
  return { ...prepared, deformedMeshCount: input.deformedMeshCount,
    skinnedMeshCount: input.skinnedCount, deformationVertexCount: input.vertexCount };
}

/**
 * Submit the prepared packet through the existing WebGpuBridgeHost, with no
 * implicit frame loop. Canvas presence selects visible rendering; absent canvas
 * selects honest offscreen execution. Host submission failures propagate intact.
 */
export async function renderDeformedMeshBatch(
  meshes, camera, bridgeHost, wasmModule, canvasContext = null, options = {},
) {
  if (typeof bridgeHost?.executePacket !== 'function') throw new TypeError('bridgeHost.executePacket is required');
  const target = canvasContext ? 'canvas' : 'offscreen';
  if (options.target !== undefined && options.target !== target) {
    throw createAdmissionError('DEFORMATION_TARGET', 'target conflicts with canvasContext presence');
  }
  const width = canvasContext?.canvas?.width ?? options.width ?? 64;
  const height = canvasContext?.canvas?.height ?? options.height ?? 64;
  const prepared = prepareDeformedMeshBatchPacket(meshes, camera, width, height, wasmModule, { ...options, target });
  const result = await bridgeHost.executePacket(prepared.packetBytes, canvasContext);
  return { ...prepared, result };
}
