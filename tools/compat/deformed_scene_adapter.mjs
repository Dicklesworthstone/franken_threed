/**
 * Opt-in scene traversal for the compiled CPU deformation / native packet path.
 * Keeps renderScene's admission, ordering, clear, culling and submission policy;
 * it does not replace the exact retained renderer or add a JavaScript deformer.
 */
import { BufferAttribute, BufferGeometry, Mesh, SkinnedMesh } from "../../upstream/three.js/build/three.module.js";
import { captureDeformationBatch, evaluateDeformationBatch } from "./deformation_inputs.mjs";
import { canAdmitMesh, createAdmissionError, renderScene } from "./mesh_adapter.mjs";

function requireThat(condition, code, detail) {
  if (!condition) throw createAdmissionError(code, detail);
}

// Retain own hooks, IDs and material state. Only our private views are changed.
function view(source, replacements, prototype = Object.getPrototypeOf(source)) {
  const descriptors = Object.getOwnPropertyDescriptors(source);
  for (const [name, value] of Object.entries(replacements)) {
    descriptors[name] = { value, enumerable: true, writable: true, configurable: true };
  }
  return Object.create(prototype, descriptors);
}

function collect(scene, camera, options) {
  const maxNodes = options.maxSceneNodes ?? 100_000;
  const maxDepth = options.maxSceneDepth ?? 512;
  for (const limit of [maxNodes, maxDepth])
    requireThat(Number.isSafeInteger(limit) && limit > 0, "DEFORMATION_SCENE_BUDGET", "scene limits must be positive safe integers");
  const ancestors = new Set();
  let rootVisible = true;
  for (let parent = scene.parent; parent != null; parent = parent.parent) {
    requireThat(!ancestors.has(parent) && parent !== scene, "DEFORMATION_SCENE_GRAPH", "cyclic scene ancestry");
    ancestors.add(parent);
    requireThat(ancestors.size <= maxDepth, "DEFORMATION_SCENE_BUDGET", "scene ancestry exceeds maxSceneDepth");
    rootVisible &&= parent.visible !== false;
  }
  const rows = [], meshes = [], seen = new Set();
  const stack = [{ node: scene, parent: scene.parent ?? null, visible: rootVisible, depth: 0 }];
  while (stack.length) {
    const row = stack.pop(), node = row.node;
    requireThat(node && Array.isArray(node.children) && !seen.has(node) && !ancestors.has(node), "DEFORMATION_SCENE_GRAPH", "children must form a tree without cycles or shared nodes");
    requireThat(rows.length < maxNodes && row.depth <= maxDepth, "DEFORMATION_SCENE_BUDGET", "scene exceeds node or depth budget");
    requireThat(node === scene || node.parent === row.parent, "DEFORMATION_SCENE_GRAPH", "child.parent does not match its owner");
    seen.add(node);
    row.visible &&= node.visible !== false;
    rows.push(row);
    const inLayers = !camera.layers || !node.layers || camera.layers.test(node.layers);
    if (node !== scene && row.visible && inLayers && node.isMesh && node.material?.visible !== false) {
      if (node.frustumCulled !== false) {
        requireThat(
          node.intersectsFrustum === Mesh.prototype.intersectsFrustum ||
            (node.isSkinnedMesh === true && node.intersectsFrustum === SkinnedMesh.prototype.intersectsFrustum),
          "UNSUPPORTED_CALLBACK", "custom intersectsFrustum is not supported",
        );
      }
      requireThat(
        node.getVertexPosition === Mesh.prototype.getVertexPosition ||
          (node.isSkinnedMesh === true && node.getVertexPosition === SkinnedMesh.prototype.getVertexPosition),
        "UNSUPPORTED_CALLBACK", "custom getVertexPosition is not supported",
      );
      meshes.push(node);
    }
    // Bound pending work as well as visited rows before pushing a wide hierarchy.
    requireThat(rows.length + stack.length + node.children.length <= maxNodes, "DEFORMATION_SCENE_BUDGET", "scene exceeds maxSceneNodes");
    for (let index = node.children.length - 1; index >= 0; index--)
      stack.push({ node: node.children[index], parent: node, visible: row.visible, depth: row.depth + 1 });
  }
  return { rows, meshes };
}

function meshView(row, positions, bounds) {
  const position = new BufferAttribute(positions.subarray(row.positionOffset, row.positionOffset + row.count * 3), 3);
  // Use a private ordinary geometry for bounds, never bind-pose object bounds or
  // a source getVertexPosition override that might apply skinning a second time.
  const scratch = new BufferGeometry();
  scratch.setAttribute("position", position);
  if (bounds) {
    scratch.computeBoundingBox();
    scratch.computeBoundingSphere();
  }
  const geometry = view(row.geometry, {
    attributes: { ...row.geometry.attributes, position },
    morphAttributes: {},
    boundingBox: scratch.boundingBox,
    boundingSphere: scratch.boundingSphere,
  });
  return view(row.mesh, {
    geometry,
    parent: row.mesh.parent,
    children: row.mesh.children,
    isSkinnedMesh: false,
    boundingBox: undefined,
    boundingSphere: undefined,
    intersectsFrustum: Mesh.prototype.intersectsFrustum,
    getVertexPosition: Mesh.prototype.getVertexPosition,
  });
}

/**
 * Capture one live scene pose into a private hierarchy suitable for renderScene.
 * By default, honor source scene/unparented-camera matrixWorldAutoUpdate flags,
 * then update each visible, in-layer skeleton once before consuming its palette.
 * autoUpdate:false instead consumes application-managed matrices/palettes as-is.
 * AnimationMixer time remains application-owned. Geometry, hierarchy, influences,
 * materials and source bounds are never overwritten.
 *
 * Only the existing opaque untextured MeshBasicMaterial slice is admitted. The
 * entire scene remains subject to renderScene's fail-closed policy, including
 * unsupported visible non-mesh objects, scene features and render callbacks.
 * Frustum rejection and sorting happen AFTER deformation using fresh bounds.
 * This is fresh-data snapshot behavior, not retained GPU attribute versioning.
 */
export function prepareDeformedScene(scene, camera, wasmModule, options = {}) {
  requireThat(scene && typeof scene.traverse === "function", "DEFORMATION_SCENE_GRAPH", "an Object3D hierarchy is required");
  requireThat(camera?.isCamera, "INVALID_CAMERA", "a camera is required");
  requireThat(options.autoUpdate === undefined || typeof options.autoUpdate === "boolean", "DEFORMATION_UPDATE_BOUNDARY", "autoUpdate must be boolean");
  const { rows, meshes } = collect(scene, camera, options);
  if (options.autoUpdate !== false) {
    if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
    if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
    const skeletons = new Set();
    for (const mesh of meshes) if (mesh.isSkinnedMesh === true) {
      requireThat(typeof mesh.skeleton?.update === "function", "DEFORMATION_SKELETON", "an updatable skeleton is required, or use autoUpdate:false");
      if (!skeletons.has(mesh.skeleton)) {
        mesh.skeleton.update();
        skeletons.add(mesh.skeleton);
      }
    }
  }
  const cameraView = view(camera, { matrixWorldAutoUpdate: false });
  const stagedMeshes = new Map();
  let input = null;
  if (meshes.length) {
    input = captureDeformationBatch(meshes, options.deformationLimits);
    // Preserve the batch API's preflight: material/camera/render-hook rejection
    // must not run the compiled evaluator or submit a partial scene.
    const basePositions = new Float32Array(input.positions);
    for (const row of input.rows) {
      const admission = canAdmitMesh(meshView(row, basePositions, false), cameraView, options);
      if (!admission.admitted) throw createAdmissionError(admission.code, admission.reason);
    }
    const positions = evaluateDeformationBatch(input, wasmModule);
    for (const row of input.rows) stagedMeshes.set(row.mesh, meshView(row, positions, true));
  }
  const views = new Map(rows.map(({ node }) => [node, stagedMeshes.get(node) ?? view(node, {
    parent: node.parent,
    children: node.children,
    ...(node === scene ? { matrixWorldAutoUpdate: false, traverse: node.traverse } : {}),
  })]));
  for (const { node } of rows) {
    const staged = views.get(node);
    // Define rather than assign: upstream may make structural fields read-only.
    Object.defineProperties(staged, {
      parent: { value: views.get(node.parent) ?? node.parent ?? null, writable: true, configurable: true, enumerable: true },
      children: { value: node.children.map((child) => views.get(child)), writable: true, configurable: true, enumerable: true },
    });
  }
  const sceneView = views.get(scene);
  Object.defineProperties(sceneView, {
    matrixWorldAutoUpdate: { value: false, writable: true, configurable: true, enumerable: true },
    // Iterative, already-validated traversal also supports deeply nested assets.
    traverse: { value: (callback) => { for (const { node } of rows) callback(views.get(node)); }, configurable: true },
  });
  return {
    scene: sceneView,
    camera: cameraView,
    deformedMeshCount: input?.deformedMeshCount ?? 0,
    skinnedMeshCount: input?.skinnedCount ?? 0,
    deformationVertexCount: input?.vertexCount ?? 0,
  };
}

/** Deform, cull, order and submit one scene through the existing native renderer. */
export async function renderDeformedScene(bridgeHost, scene, camera, canvasContext, wasmModule, options = {}) {
  requireThat(typeof bridgeHost?.executePacket === "function", "DEFORMATION_HOST", "bridgeHost.executePacket is required");
  const target = canvasContext == null ? "offscreen" : "canvas";
  requireThat(options.target === undefined || options.target === target, "DEFORMATION_TARGET", "target conflicts with canvasContext presence");
  for (const value of [canvasContext?.canvas?.width ?? options.width ?? 64, canvasContext?.canvas?.height ?? options.height ?? 64])
    requireThat(Number.isInteger(value) && value >= 1 && value <= 0xffffffff, "INVALID_DIMENSIONS", "viewport dimensions must be positive u32 values");
  const prepared = prepareDeformedScene(scene, camera, wasmModule, options);
  const result = await renderScene(bridgeHost, prepared.scene, prepared.camera, canvasContext ?? null, wasmModule, { ...options, autoUpdate: false, target });
  return {
    ...result,
    deformedMeshCount: prepared.deformedMeshCount,
    skinnedMeshCount: prepared.skinnedMeshCount,
    deformationVertexCount: prepared.deformationVertexCount,
  };
}
